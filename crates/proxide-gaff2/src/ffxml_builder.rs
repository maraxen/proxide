//! OpenMM FFXML builder — converts GAFF2 atom types + parameters into an
//! FFXML `<ForceField>` string for a single small-molecule residue.
//!
//! Ports `load_gaff2_parameters`, `assign_pdb_atom_names`,
//! `_lookup_bond_params`, `_lookup_angle_params`, and `build_gaff2_ffxml`
//! (gaff2.py:1260-1677). This is a **behavior-preserving** port: known
//! residual quirks of the Python (e.g. the VdW-section parser never turning
//! back off once a `MOD4` header is seen, or the improper-parsing branch
//! requiring `int(parts[1])` to succeed even though the parsed value is
//! never used for impropers) are reproduced verbatim, not "fixed".
//!
//! ## Deviations forced by this crate's existing contracts
//!
//! * `crate::orchestrate::assign_gaff2_atom_types` (owned by a different
//!   port task) returns `Result<Vec<String>, String>`, whereas the Python
//!   `assign_gaff2_atom_types` never raises. [`build_gaff2_ffxml`] threads
//!   that `Result` through via [`Gaff2FfxmlError::AtomTyping`].
//! * Python resolves the `.dat` parameter file at runtime from
//!   `Path(__file__).parent.parent / "assets" / "gaff" / "dat" /
//!   f"{gaff_version}.dat"`. Rust has no direct equivalent of
//!   `Path(__file__)`-relative dynamic loading that also survives standalone
//!   (e.g. wasm) compilation, and this crate's established convention
//!   (`rules_loader.rs`) is to `include_str!()` the canonical asset at
//!   compile time. [`build_gaff2_ffxml`] therefore takes a pre-loaded
//!   [`Gaff2Params`] rather than a `gaff_version` string; the default
//!   `gaff-2.2.20.dat` is embedded and reachable via
//!   [`load_gaff2_parameters_default`], matching Python's default arg
//!   (`gaff_version: str = "gaff-2.2.20"`) for the only version any test
//!   exercises. `gaff_version` is still accepted as a label (mirrors the
//!   Python signature) and is otherwise unused, since it's implied by
//!   whichever `Gaff2Params` the caller loaded.
//! * `MolGraph` (this crate's owned molecular representation) carries no
//!   RDKit `AtomPDBResidueInfo`-equivalent "existing name" field, so
//!   [`assign_pdb_atom_names`] takes an explicit `existing_names: Option<&[Option<String>]>`
//!   in place of Python's `atom.GetMonomerInfo()` read. Passing `None`
//!   reproduces the common case (fresh molecule, no pre-existing PDB names).

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::mol::MolGraph;

// ---------------------------------------------------------------------------
// Unit conversion constants (gaff2.py:1391-1399)
// ---------------------------------------------------------------------------

const KCAL_TO_KJ: f64 = 4.184;
/// kcal/mol/Å² → kJ/mol/nm²
const BOND_K_CONV: f64 = 2.0 * KCAL_TO_KJ / (0.1 * 0.1);
/// Å → nm
const BOND_R_CONV: f64 = 0.1;
/// kcal/mol/rad² → kJ/mol/rad²
const ANGLE_K_CONV: f64 = 2.0 * KCAL_TO_KJ;
/// deg → rad
const ANGLE_T_CONV: f64 = std::f64::consts::PI / 180.0;
/// kcal/mol → kJ/mol
const TORSION_K_CONV: f64 = KCAL_TO_KJ;
/// deg → rad
const TORSION_P_CONV: f64 = std::f64::consts::PI / 180.0;
/// Rmin/2 (Å) → σ (nm). `f64::powf` isn't usable in a `const` context, so
/// this is a `fn` rather than a `const` like its sibling conversions —
/// computes the identical value Python's `2.0 * (2.0 ** (-1.0/6.0)) * 0.1`
/// does at runtime.
fn lj_sigma_conv() -> f64 {
    2.0 * (2.0_f64).powf(-1.0 / 6.0) * 0.1
}
/// kcal/mol → kJ/mol
const LJ_EPS_CONV: f64 = KCAL_TO_KJ;

/// The `gaff_version` label whose `.dat` file is embedded at compile time
/// (see module-level deviation note on dynamic `gaff_version` resolution).
pub const DEFAULT_GAFF_VERSION: &str = "gaff-2.2.20";

fn elem_mass_default(elem: &str) -> f64 {
    match elem {
        "C" => 12.011,
        "O" => 15.999,
        "H" => 1.008,
        "N" => 14.007,
        "S" => 32.06,
        "P" => 30.974,
        "F" => 18.998,
        "Cl" => 35.453,
        "Br" => 79.904,
        "I" => 126.904,
        _ => 12.011,
    }
}

fn vdw_elem_default(elem: &str) -> (f64, f64) {
    match elem {
        "C" => (1.9080, 0.0860),
        "O" => (1.6612, 0.2100),
        "H" => (1.4593, 0.0208),
        "N" => (1.8240, 0.1700),
        "S" => (2.0000, 0.2500),
        "P" => (2.1000, 0.2000),
        _ => (1.9080, 0.0860),
    }
}

/// Bond-type substitution table used as a last-resort fallback when a direct
/// (or reversed) parameter lookup misses (gaff2.py:1413-1418).
fn bond_type_sub(t: &str) -> String {
    match t {
        "cx" | "cy" | "c5" | "c6" => "c3",
        "cs" | "cz" | "ca" | "cc" | "cd" | "ce" | "cf" | "cp" | "cq" => "c2",
        "cg" | "ch" => "c1",
        other => other,
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// GAFF2 parameter table (load_gaff2_parameters, gaff2.py:1260-1384)
// ---------------------------------------------------------------------------

/// Parsed GAFF2 `.dat` parameter tables.
///
/// All maps are keyed purely for lookup (never iterated in an
/// order-sensitive way in the Python source), so plain `HashMap` is
/// sufficient — unlike e.g. `MolGraph::bonds`, whose *iteration* order is
/// load-bearing. See `atom_bond_facts.rs` for the same reasoning applied to
/// `AtomBondFacts`. The per-key `Vec` of torsion terms *is* order-sensitive
/// (multiple periodicity terms accumulate in file-encounter order) and a
/// `Vec` naturally preserves that regardless of the outer map type.
#[derive(Debug, Clone, Default)]
pub struct Gaff2Params {
    pub masses: HashMap<String, f64>,
    /// (type1, type2) -> (kb, r0_angstrom)
    pub bonds: HashMap<(String, String), (f64, f64)>,
    /// (type1, type2, type3) -> (kt, t0_deg)
    pub angles: HashMap<(String, String, String), (f64, f64)>,
    /// (type1, type2, type3, type4) -> [(periodicity, kt_kcal, phase_deg), ...]
    pub torsions: HashMap<(String, String, String, String), Vec<(i32, f64, f64)>>,
    /// (type1, type2, type3, type4) -> (kt_kcal, phase_deg)
    pub impropers: HashMap<(String, String, String, String), (f64, f64)>,
    /// atom_type -> (rmin_half_angstrom, epsilon_kcal_mol)
    pub vdw: HashMap<String, (f64, f64)>,
}

static GAFF_2_2_20_DAT: &str =
    include_str!("../../../src/proxide/assets/gaff/dat/gaff-2.2.20.dat");

/// Load the embedded default (`gaff-2.2.20.dat`) parameter table.
///
/// Equivalent to Python's `load_gaff2_parameters(None)`, which resolves to
/// the same bundled file.
pub fn load_gaff2_parameters_default() -> Gaff2Params {
    parse_gaff2_parameters(GAFF_2_2_20_DAT)
}

/// Read and parse a GAFF2 `.dat` file from an arbitrary path.
///
/// Equivalent to Python's `load_gaff2_parameters(dat_path)` for an explicit
/// path (the implicit `Path(__file__)`-relative default is
/// [`load_gaff2_parameters_default`] here — see module deviation note).
pub fn load_gaff2_parameters_from_path(
    path: &std::path::Path,
) -> std::io::Result<Gaff2Params> {
    let content = std::fs::read_to_string(path)?;
    Ok(parse_gaff2_parameters(&content))
}

/// True iff `s` has at least one alphabetic character and every alphabetic
/// character is lowercase — mirrors Python's `str.islower()` closely enough
/// for the ASCII GAFF2 type-name alphabet this parser sees.
fn py_islower(s: &str) -> bool {
    let mut has_cased = false;
    for c in s.chars() {
        if c.is_alphabetic() {
            has_cased = true;
            if !c.is_lowercase() {
                return false;
            }
        }
    }
    has_cased
}

/// Parse GAFF2 `.dat` file content into parameter tables.
///
/// Direct port of `load_gaff2_parameters`'s parsing loop (gaff2.py:1283-1384).
/// Preserves every quirk verbatim, including:
/// * the VdW section (`in_vdw`) is entered on a `MOD4` header line and is
///   **never turned back off** for the rest of the file (the Python has no
///   code path that resets it) — every non-blank line for the remainder of
///   the file is attempted as a VdW entry;
/// * the single-char-type padding reconstruction (`"c -o kb r0"` splitting
///   as `["c", "-o", ...]` gets its first two tokens rejoined into `"c-o"`);
/// * the dash-count==3 branch requires `int(parts[1])` (the periodicity
///   field) to parse successfully even on the improper path, where that
///   value is never used — a malformed periodicity field silently drops an
///   otherwise well-formed improper line, not just a torsion line.
pub fn parse_gaff2_parameters(content: &str) -> Gaff2Params {
    let mut params = Gaff2Params::default();
    let mut in_vdw = false;

    for line in content.split('\n') {
        let line_stripped = line.trim();

        if line_stripped.starts_with("MOD4") {
            in_vdw = true;
            continue;
        }

        if in_vdw {
            if line_stripped.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line_stripped.split_whitespace().collect();
            if parts.len() >= 3 {
                if let (Ok(rmin_half), Ok(epsilon)) =
                    (parts[1].parse::<f64>(), parts[2].parse::<f64>())
                {
                    params
                        .vdw
                        .insert(parts[0].to_string(), (rmin_half, epsilon));
                }
            }
            continue;
        }

        if line_stripped.is_empty() {
            continue;
        }

        let raw_parts: Vec<String> = line_stripped
            .split_whitespace()
            .map(String::from)
            .collect();
        if raw_parts.len() < 2 {
            continue;
        }

        // AMBER .dat quirk: single-char types are padded, so "c -o kb r0"
        // splits as ["c", "-o", "kb", "r0"]. Reconstruct the dash-joined
        // key: parts = [parts[0]+parts[1]] + parts[2:] (gaff2.py:1315-1316).
        let parts: Vec<String> = if !raw_parts[0].contains('-') && raw_parts[1].starts_with('-') {
            let mut out = vec![format!("{}{}", raw_parts[0], raw_parts[1])];
            out.extend(raw_parts.into_iter().skip(2));
            out
        } else {
            raw_parts
        };
        if parts.len() < 2 {
            continue;
        }

        let first = parts[0].as_str();

        if first.contains('-') {
            let dash_count = first.matches('-').count();

            if dash_count == 1 {
                let mut it = first.splitn(2, '-');
                let t1 = it.next().unwrap_or("");
                let t2 = it.next().unwrap_or("");
                if t1.len() <= 3 && t2.len() <= 3 {
                    if let (Some(p1), Some(p2)) = (parts.get(1), parts.get(2)) {
                        if let (Ok(kb), Ok(r0)) = (p1.parse::<f64>(), p2.parse::<f64>()) {
                            if kb > 0.0 {
                                params
                                    .bonds
                                    .insert((t1.to_string(), t2.to_string()), (kb, r0));
                            }
                        }
                    }
                }
            } else if dash_count == 2 {
                let mut it1 = first.splitn(2, '-');
                let t1 = it1.next().unwrap_or("");
                let rest = it1.next().unwrap_or("");
                let mut it2 = rest.splitn(2, '-');
                let t2 = it2.next().unwrap_or("");
                let t3 = it2.next().unwrap_or("");
                if t1.len() <= 3 && t2.len() <= 3 && t3.len() <= 3 {
                    if let (Some(p1), Some(p2)) = (parts.get(1), parts.get(2)) {
                        if let (Ok(kt), Ok(t0)) = (p1.parse::<f64>(), p2.parse::<f64>()) {
                            if kt > 0.0 {
                                params.angles.insert(
                                    (t1.to_string(), t2.to_string(), t3.to_string()),
                                    (kt, t0),
                                );
                            }
                        }
                    }
                }
            } else if dash_count == 3 {
                let mut it1 = first.splitn(2, '-');
                let t1 = it1.next().unwrap_or("");
                let rest = it1.next().unwrap_or("");
                let mut it2 = rest.splitn(2, '-');
                let t2 = it2.next().unwrap_or("");
                let rest2 = it2.next().unwrap_or("");
                let mut it3 = rest2.splitn(2, '-');
                let t3 = it3.next().unwrap_or("");
                let t4 = it3.next().unwrap_or("");

                if t1.len() <= 3 && t2.len() <= 3 && t3.len() <= 3 && t4.len() <= 3 {
                    // Outer try: periodicity=int(parts[1]), kt=float(parts[2]),
                    // phase=float(parts[3]) must ALL succeed, even on the
                    // improper path where `periodicity` is unused — this is
                    // a faithfully-preserved quirk, not a bug we're fixing.
                    let outer = (|| -> Option<(i32, f64, f64)> {
                        let periodicity = parts.get(1)?.parse::<i32>().ok()?;
                        let kt = parts.get(2)?.parse::<f64>().ok()?;
                        let phase = parts.get(3)?.parse::<f64>().ok()?;
                        Some((periodicity, kt, phase))
                    })();

                    if let Some((periodicity, kt, phase)) = outer {
                        if parts.len() >= 5 {
                            let key = (
                                t1.to_string(),
                                t2.to_string(),
                                t3.to_string(),
                                t4.to_string(),
                            );
                            params
                                .torsions
                                .entry(key)
                                .or_default()
                                .push((periodicity, kt, phase));
                        } else {
                            // Inner try: kt_imp=float(parts[2]), phase_imp=float(parts[3]).
                            let inner = (|| -> Option<(f64, f64)> {
                                let kt_imp = parts.get(2)?.parse::<f64>().ok()?;
                                let phase_imp = parts.get(3)?.parse::<f64>().ok()?;
                                Some((kt_imp, phase_imp))
                            })();
                            if let Some((kt_imp, phase_imp)) = inner {
                                if kt_imp > 0.0 {
                                    params.impropers.insert(
                                        (
                                            t1.to_string(),
                                            t2.to_string(),
                                            t3.to_string(),
                                            t4.to_string(),
                                        ),
                                        (kt_imp, phase_imp),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        } else if first.len() <= 3 && py_islower(&first.replace('+', "")) {
            if let Some(p1) = parts.get(1) {
                if let Ok(mass) = p1.parse::<f64>() {
                    params.masses.insert(first.to_string(), mass);
                }
            }
        }
    }

    params
}

// ---------------------------------------------------------------------------
// Parameter lookup with substitution fallback
// (_lookup_bond_params / _lookup_angle_params, gaff2.py:1448-1477)
// ---------------------------------------------------------------------------

/// Return `(kb, r0)` for a bond, trying substitutions on miss.
fn lookup_bond_params(ti: &str, tj: &str, params: &Gaff2Params) -> (f64, f64) {
    for (a, b) in [(ti, tj), (tj, ti)] {
        if let Some(&v) = params.bonds.get(&(a.to_string(), b.to_string())) {
            return v;
        }
    }
    let ti_s = bond_type_sub(ti);
    let tj_s = bond_type_sub(tj);
    for (a, b) in [(ti_s.as_str(), tj_s.as_str()), (tj_s.as_str(), ti_s.as_str())] {
        if let Some(&v) = params.bonds.get(&(a.to_string(), b.to_string())) {
            return v;
        }
    }
    (0.0, 1.50)
}

/// Return `(kt, t0)` for an angle, trying substitutions on miss.
fn lookup_angle_params(ti: &str, tj: &str, tk: &str, params: &Gaff2Params) -> (f64, f64) {
    for (a, b, c) in [(ti, tj, tk), (tk, tj, ti)] {
        if let Some(&v) = params
            .angles
            .get(&(a.to_string(), b.to_string(), c.to_string()))
        {
            return v;
        }
    }
    let ti_s = bond_type_sub(ti);
    let tj_s = bond_type_sub(tj);
    let tk_s = bond_type_sub(tk);
    for (a, b, c) in [
        (ti_s.as_str(), tj_s.as_str(), tk_s.as_str()),
        (tk_s.as_str(), tj_s.as_str(), ti_s.as_str()),
    ] {
        if let Some(&v) = params
            .angles
            .get(&(a.to_string(), b.to_string(), c.to_string()))
        {
            return v;
        }
    }
    (0.0, 120.0)
}

// ---------------------------------------------------------------------------
// PDB atom naming (assign_pdb_atom_names, gaff2.py:1421-1445)
// ---------------------------------------------------------------------------

/// Assign unique PDB atom names.
///
/// `existing_names[i]`, when `Some` and non-blank after trimming, is kept
/// as-is (mirrors a non-empty RDKit `AtomPDBResidueInfo` name). Otherwise a
/// fresh name of the form `{element}{counter}` is assigned, with the
/// counter scoped per-element and incrementing in atom-index order (e.g.
/// C1, C2, O1, H1, H2, ...).
///
/// `existing_names` may be `None` (equivalent to every atom lacking prior
/// MonomerInfo) — see the module-level deviation note on why `MolGraph`
/// can't carry this itself.
pub fn assign_pdb_atom_names(
    elements: &[String],
    existing_names: Option<&[Option<String>]>,
) -> Vec<String> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    let mut names = Vec::with_capacity(elements.len());
    for (idx, elem) in elements.iter().enumerate() {
        let existing = existing_names
            .and_then(|s| s.get(idx))
            .and_then(|o| o.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        if let Some(name) = existing {
            names.push(name.to_string());
        } else {
            let counter = counts.entry(elem.clone()).or_insert(0);
            *counter += 1;
            names.push(format!("{elem}{counter}"));
        }
    }
    names
}

// ---------------------------------------------------------------------------
// build_gaff2_ffxml (gaff2.py:1480-1677)
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum Gaff2FfxmlError {
    #[error("charges length {actual} != mol atom count {expected}")]
    ChargesLengthMismatch { expected: usize, actual: usize },
    #[error("atom type assignment failed: {0}")]
    AtomTyping(String),
}

/// Per-atom incident-bond indices, in **molecule (bond-list) order** —
/// mirrors RDKit `atom.GetBonds()` order, which follows bond-addition order.
/// Load-bearing for angle/torsion enumeration order, per the same discipline
/// documented on `MolGraph::bonds` and `atadjust()`.
fn atom_incident_bonds(mol: &MolGraph) -> Vec<Vec<usize>> {
    let n = mol.elements.len();
    let mut result: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (bidx, bond) in mol.bonds.iter().enumerate() {
        result[bond.i].push(bidx);
        result[bond.j].push(bidx);
    }
    result
}

fn other_atom(bond: &crate::mol::Bond, atom: usize) -> usize {
    if bond.i == atom {
        bond.j
    } else {
        bond.i
    }
}

/// Build a complete OpenMM FFXML string for a single small-molecule residue.
///
/// Faithful port of `build_gaff2_ffxml` (gaff2.py:1480-1677). Calls
/// `crate::orchestrate::assign_gaff2_atom_types` internally, exactly as the
/// Python calls `assign_gaff2_atom_types(rdmol)` — see module deviation
/// note on that function's `Result`-returning signature here vs. Python's
/// non-raising one.
///
/// `gaff_version` is accepted for signature parity with Python but is
/// otherwise unused: `params` must already reflect whichever `.dat` file
/// the caller wants (see module deviation note on `.dat` resolution).
pub fn build_gaff2_ffxml(
    mol: &MolGraph,
    resname: &str,
    charges: &[f64],
    gaff_version: &str,
    params: &Gaff2Params,
) -> Result<String, Gaff2FfxmlError> {
    let _ = gaff_version; // signature parity only; see module deviation note

    let atom_names = assign_pdb_atom_names(&mol.elements, None);
    let n_atoms = mol.elements.len();

    if charges.len() != n_atoms {
        return Err(Gaff2FfxmlError::ChargesLengthMismatch {
            expected: n_atoms,
            actual: charges.len(),
        });
    }

    let atom_types = crate::orchestrate::assign_gaff2_atom_types(mol)
        .map_err(Gaff2FfxmlError::AtomTyping)?;

    Ok(build_gaff2_ffxml_from_types(
        mol,
        resname,
        charges,
        params,
        &atom_types,
        &atom_names,
    ))
}

/// Core FFXML-writing logic, factored out of [`build_gaff2_ffxml`] so it can
/// be exercised in tests without depending on the (separately-ported, and
/// currently stubbed) `assign_gaff2_atom_types` engine. `build_gaff2_ffxml`
/// is the faithful full port of the Python public API; this function is a
/// pure internal decomposition, not a behavior change — every line below
/// corresponds 1:1 to gaff2.py:1524-1677.
///
/// Callers must supply `atom_types`/`atom_names` already aligned with
/// `mol`'s atom-index order and of matching length; `charges` must already
/// be length-validated (see `build_gaff2_ffxml`).
pub fn build_gaff2_ffxml_from_types(
    mol: &MolGraph,
    resname: &str,
    charges: &[f64],
    params: &Gaff2Params,
    atom_types: &[String],
    atom_names: &[String],
) -> String {
    let n_atoms = mol.elements.len();
    let idx_to_type = atom_types; // index-aligned Vec plays the role of idx_to_type dict

    let mut lines: Vec<String> = vec![
        r#"<?xml version="1.0" encoding="utf-8"?>"#.to_string(),
        "<ForceField>".to_string(),
    ];

    // --- AtomTypes ---
    lines.push("  <AtomTypes>".to_string());
    for idx in 0..n_atoms {
        let gaff_type = &idx_to_type[idx];
        let elem = &mol.elements[idx];
        let mass = params
            .masses
            .get(gaff_type)
            .copied()
            .unwrap_or_else(|| elem_mass_default(elem));
        lines.push(format!(
            r#"    <Type name="{resname}_{idx}" class="{gaff_type}" element="{elem}" mass="{mass:.4}"/>"#
        ));
    }
    lines.push("  </AtomTypes>".to_string());

    // --- Residues ---
    lines.push("  <Residues>".to_string());
    lines.push(format!(r#"    <Residue name="{resname}">"#));
    for idx in 0..n_atoms {
        let name = &atom_names[idx];
        let q = charges[idx];
        lines.push(format!(
            r#"      <Atom name="{name}" type="{resname}_{idx}" charge="{q:.6}"/>"#
        ));
    }
    for bond in &mol.bonds {
        let i = bond.i;
        let j = bond.j;
        lines.push(format!(
            r#"      <Bond atomName1="{}" atomName2="{}"/>"#,
            atom_names[i], atom_names[j]
        ));
    }
    lines.push("    </Residue>".to_string());
    lines.push("  </Residues>".to_string());

    // --- HarmonicBondForce ---
    lines.push("  <HarmonicBondForce>".to_string());
    let mut missing_bonds = 0u32;
    for bond in &mol.bonds {
        let i = bond.i;
        let j = bond.j;
        let ti = &idx_to_type[i];
        let tj = &idx_to_type[j];
        let (kb, r0) = lookup_bond_params(ti, tj, params);
        if kb == 0.0 {
            missing_bonds += 1;
        }
        let r0_nm = r0 * BOND_R_CONV;
        let k_kj = kb * BOND_K_CONV;
        lines.push(format!(
            r#"    <Bond type1="{resname}_{i}" type2="{resname}_{j}" length="{r0_nm:.6}" k="{k_kj:.2}"/>"#
        ));
    }
    if missing_bonds > 0 {
        log::warn!(
            "build_gaff2_ffxml: {missing_bonds} bonds missing GAFF2 parameters (kb=0)"
        );
    }
    lines.push("  </HarmonicBondForce>".to_string());

    // --- HarmonicAngleForce ---
    lines.push("  <HarmonicAngleForce>".to_string());
    let incident = atom_incident_bonds(mol);
    for j in 0..n_atoms {
        let bonds_j = &incident[j];
        for b1 in 0..bonds_j.len() {
            let i = other_atom(&mol.bonds[bonds_j[b1]], j);
            for b2 in (b1 + 1)..bonds_j.len() {
                let k = other_atom(&mol.bonds[bonds_j[b2]], j);
                let ti = &idx_to_type[i];
                let tj = &idx_to_type[j];
                let tk = &idx_to_type[k];
                let (kt, t0) = lookup_angle_params(ti, tj, tk, params);
                let t0_rad = t0 * ANGLE_T_CONV;
                let k_kj = kt * ANGLE_K_CONV;
                lines.push(format!(
                    r#"    <Angle type1="{resname}_{i}" type2="{resname}_{j}" type3="{resname}_{k}" angle="{t0_rad:.6}" k="{k_kj:.2}"/>"#
                ));
            }
        }
    }
    lines.push("  </HarmonicAngleForce>".to_string());

    // --- PeriodicTorsionForce ---
    lines.push("  <PeriodicTorsionForce>".to_string());
    let mut seen_torsions: HashSet<(usize, usize, usize, usize)> = HashSet::new();
    for bond in &mol.bonds {
        let j = bond.i;
        let k = bond.j;
        for &bi in &incident[j] {
            let i = other_atom(&mol.bonds[bi], j);
            if i == k {
                continue;
            }
            for &bl in &incident[k] {
                let ll = other_atom(&mol.bonds[bl], k);
                if ll == j || ll == i {
                    continue;
                }
                let fwd = (i, j, k, ll);
                let rev = (ll, k, j, i);
                let key = fwd.min(rev);
                if seen_torsions.contains(&key) {
                    continue;
                }
                seen_torsions.insert(key);

                let ti = idx_to_type[i].clone();
                let tj = idx_to_type[j].clone();
                let tk_ = idx_to_type[k].clone();
                let tl = idx_to_type[ll].clone();

                let torsion_key = (ti.clone(), tj.clone(), tk_.clone(), tl.clone());
                let mut torsion_params: Vec<(i32, f64, f64)> = params
                    .torsions
                    .get(&torsion_key)
                    .cloned()
                    .unwrap_or_default();
                if torsion_params.is_empty() {
                    let rev_key = (tl.clone(), tk_.clone(), tj.clone(), ti.clone());
                    torsion_params = params.torsions.get(&rev_key).cloned().unwrap_or_default();
                }
                if torsion_params.is_empty() {
                    let ti_s = bond_type_sub(&ti);
                    let tj_s = bond_type_sub(&tj);
                    let tk_s = bond_type_sub(&tk_);
                    let tl_s = bond_type_sub(&tl);
                    for sub_key in [
                        (ti_s.clone(), tj_s.clone(), tk_s.clone(), tl_s.clone()),
                        (tl_s, tk_s, tj_s, ti_s),
                    ] {
                        if let Some(v) = params.torsions.get(&sub_key) {
                            if !v.is_empty() {
                                torsion_params = v.clone();
                                break;
                            }
                        }
                    }
                }
                if torsion_params.is_empty() {
                    continue;
                }

                let mut attrs = format!(
                    r#"type1="{resname}_{i}" type2="{resname}_{j}" type3="{resname}_{k}" type4="{resname}_{ll}""#
                );
                for (term_idx, (n, kt, phase)) in torsion_params.iter().enumerate() {
                    let term_idx = term_idx + 1;
                    let phase_rad = phase * TORSION_P_CONV;
                    let k_kj = kt * TORSION_K_CONV;
                    let _ = write!(
                        attrs,
                        r#" periodicity{term_idx}="{n}" phase{term_idx}="{phase_rad:.6}" k{term_idx}="{k_kj:.4}""#
                    );
                }
                lines.push(format!("    <Proper {attrs}/>"));
            }
        }
    }
    lines.push("  </PeriodicTorsionForce>".to_string());

    // --- NonbondedForce ---
    lines.push(
        r#"  <NonbondedForce coulomb14scale="0.8333333333" lj14scale="0.5">"#.to_string(),
    );
    for idx in 0..n_atoms {
        let gaff_type = &idx_to_type[idx];
        let elem = &mol.elements[idx];
        let q = charges[idx];
        let (rmin_half, epsilon) = params
            .vdw
            .get(gaff_type)
            .copied()
            .unwrap_or_else(|| vdw_elem_default(elem));
        let sigma_nm = rmin_half * lj_sigma_conv();
        let eps_kj = epsilon * LJ_EPS_CONV;
        lines.push(format!(
            r#"    <Atom type="{resname}_{idx}" charge="{q:.6}" sigma="{sigma_nm:.8}" epsilon="{eps_kj:.6}"/>"#
        ));
    }
    lines.push("  </NonbondedForce>".to_string());
    lines.push("</ForceField>".to_string());

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mol::{Bond, BondOrder};

    fn make_bond(i: usize, j: usize) -> Bond {
        Bond {
            i,
            j,
            order: BondOrder::Single,
            aromatic: false,
        }
    }

    fn mol_from(elements: &[&str], bonds: Vec<Bond>) -> MolGraph {
        MolGraph {
            elements: elements.iter().map(|s| s.to_string()).collect(),
            formal_charges: vec![0; elements.len()],
            bonds,
            rings: None,
        }
    }

    // -- load_gaff2_parameters / parse_gaff2_parameters -----------------
    // Ports tests/test_gaff2.py::test_parameter_lookups and
    // ::test_parameter_values against the real bundled gaff-2.2.20.dat.

    #[test]
    fn test_parameter_lookups_masses() {
        let params = load_gaff2_parameters_default();
        for atom_type in ["c3", "cp", "cx", "oh", "op", "n3", "ni", "hc"] {
            assert!(
                params.masses.contains_key(atom_type),
                "mass lookup failed for {atom_type}"
            );
        }
    }

    #[test]
    fn test_parameter_lookups_bonds() {
        let params = load_gaff2_parameters_default();
        for (a, b) in [("c3", "c3"), ("cx", "op"), ("c3", "oh"), ("c3", "hc")] {
            let mut pair = [a, b];
            pair.sort_unstable();
            let key = (pair[0].to_string(), pair[1].to_string());
            assert!(
                params.bonds.contains_key(&key),
                "bond lookup failed for {a}-{b} (sorted key {key:?})"
            );
        }
    }

    #[test]
    fn test_parameter_lookups_angles() {
        let params = load_gaff2_parameters_default();
        for (a, b, c) in [("c3", "c3", "c3"), ("c3", "c3", "oh"), ("c3", "c3", "hc")] {
            let key = (a.to_string(), b.to_string(), c.to_string());
            assert!(
                params.angles.contains_key(&key),
                "angle lookup failed for {a}-{b}-{c}"
            );
        }
    }

    #[test]
    fn test_parameter_lookups_torsions_with_substitution() {
        let params = load_gaff2_parameters_default();
        // propane H torsions become c3-c3-c3-hc via cx->c3, x->hc substitution
        let key = (
            "c3".to_string(),
            "c3".to_string(),
            "c3".to_string(),
            "hc".to_string(),
        );
        let v = params.torsions.get(&key);
        assert!(
            v.is_some_and(|v| !v.is_empty()),
            "torsion lookup failed for substituted c3-c3-c3-hc"
        );

        let key2 = (
            "c3".to_string(),
            "c3".to_string(),
            "c3".to_string(),
            "c3".to_string(),
        );
        let v2 = params.torsions.get(&key2);
        assert!(
            v2.is_some_and(|v| !v.is_empty()),
            "torsion lookup failed for direct c3-c3-c3-c3"
        );
    }

    // NOTE on divergence from tests/test_gaff2.py::test_parameter_values:
    // that Python test's `known_masses` (c3=1.9069, cp=1.8606, oh=1.82,
    // op=1.7713) and `known_bonds` (("c3","op") -> (273.64, 1.4368)) are
    // never actually checked by pytest -- the function ends with
    // `return failed == 0` rather than an `assert`, the exact same
    // never-fails-regardless-of-content bug this test file's own module
    // docstring calls out for the *removed* `test_atom_types`. Grepping the
    // real gaff-2.2.20.dat confirms those reference numbers are simply
    // wrong: `c3`/`cp` masses are 12.01, `oh`/`op` are 16.00 (see the file's
    // `MASS` section), and no `c3-op` bond entry exists at all. Porting the
    // Python file's literal numbers forward as real Rust `assert`s would
    // fail against a *correct* parser and silently encode that stale bug --
    // so this asserts the values the real file actually contains instead,
    // plus the same volume-sanity checks the Python test also performs
    // (which ARE meaningful, since they don't depend on the wrong numbers).
    #[test]
    fn test_parameter_values_masses() {
        let params = load_gaff2_parameters_default();
        let known: &[(&str, f64, f64)] = &[
            ("c3", 12.01, 0.001),
            ("cp", 12.01, 0.001),
            ("oh", 16.00, 0.001),
            ("op", 16.00, 0.001),
        ];
        for &(t, expected, tol) in known {
            let actual = *params
                .masses
                .get(t)
                .unwrap_or_else(|| panic!("mass missing for {t}"));
            assert!(
                (actual - expected).abs() <= tol,
                "{t} mass = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_parameter_values_volume_sanity() {
        // Mirrors the (functional, non-`known_*`) part of
        // tests/test_gaff2.py::test_parameter_values.
        let params = load_gaff2_parameters_default();
        assert!(params.masses.len() >= 90, "only {} masses loaded", params.masses.len());
        assert!(params.bonds.len() >= 1000, "only {} bonds loaded", params.bonds.len());
        assert!(params.angles.len() >= 7000, "only {} angles loaded", params.angles.len());
    }

    // -- lookup_bond_params / lookup_angle_params substitution chain ----
    // Synthetic params isolate the fallback-ladder logic from real .dat
    // content availability.

    #[test]
    fn test_lookup_bond_params_direct_hit() {
        let mut params = Gaff2Params::default();
        params
            .bonds
            .insert(("c3".to_string(), "hc".to_string()), (340.0, 1.09));
        assert_eq!(lookup_bond_params("c3", "hc", &params), (340.0, 1.09));
    }

    #[test]
    fn test_lookup_bond_params_reversed_hit() {
        let mut params = Gaff2Params::default();
        params
            .bonds
            .insert(("hc".to_string(), "c3".to_string()), (340.0, 1.09));
        assert_eq!(lookup_bond_params("c3", "hc", &params), (340.0, 1.09));
    }

    #[test]
    fn test_lookup_bond_params_substitution_hit() {
        let mut params = Gaff2Params::default();
        // only c3-c3 present; querying cx-c3 should hit via cx->c3 sub
        params
            .bonds
            .insert(("c3".to_string(), "c3".to_string()), (300.9, 1.535));
        assert_eq!(lookup_bond_params("cx", "c3", &params), (300.9, 1.535));
    }

    #[test]
    fn test_lookup_bond_params_default_fallback() {
        let params = Gaff2Params::default();
        assert_eq!(lookup_bond_params("zz", "yy", &params), (0.0, 1.50));
    }

    #[test]
    fn test_lookup_angle_params_default_fallback() {
        let params = Gaff2Params::default();
        assert_eq!(lookup_angle_params("zz", "yy", "xx", &params), (0.0, 120.0));
    }

    // -- assign_pdb_atom_names ------------------------------------------

    #[test]
    fn test_assign_pdb_atom_names_fresh() {
        let elements = vec!["C".to_string(), "C".to_string(), "H".to_string(), "H".to_string()];
        let names = assign_pdb_atom_names(&elements, None);
        assert_eq!(names, vec!["C1", "C2", "H1", "H2"]);
    }

    #[test]
    fn test_assign_pdb_atom_names_keeps_existing() {
        let elements = vec!["C".to_string(), "C".to_string()];
        let existing = vec![Some("CA".to_string()), None];
        let names = assign_pdb_atom_names(&elements, Some(&existing));
        assert_eq!(names, vec!["CA", "C1"]);
    }

    #[test]
    fn test_assign_pdb_atom_names_blank_existing_treated_as_absent() {
        let elements = vec!["O".to_string()];
        let existing = vec![Some("   ".to_string())];
        let names = assign_pdb_atom_names(&elements, Some(&existing));
        assert_eq!(names, vec!["O1"]);
    }

    // -- build_gaff2_ffxml_from_types (core FFXML logic) -----------------

    /// Small hand-built params table covering everything a 4-atom butane-like
    /// chain (i-j-k-l, all c3, with terminal hc on i and l) needs.
    fn synthetic_params() -> Gaff2Params {
        let mut params = Gaff2Params::default();
        params.masses.insert("c3".to_string(), 12.01);
        params.masses.insert("hc".to_string(), 1.008);
        params
            .bonds
            .insert(("c3".to_string(), "c3".to_string()), (300.9, 1.535));
        params
            .bonds
            .insert(("c3".to_string(), "hc".to_string()), (330.6, 1.092));
        params.angles.insert(
            ("c3".to_string(), "c3".to_string(), "c3".to_string()),
            (58.35, 111.51),
        );
        params.angles.insert(
            ("hc".to_string(), "c3".to_string(), "c3".to_string()),
            (46.4, 109.8),
        );
        params.torsions.insert(
            (
                "c3".to_string(),
                "c3".to_string(),
                "c3".to_string(),
                "c3".to_string(),
            ),
            vec![(3, 0.18, 0.0), (1, 0.25, 180.0)],
        );
        params
            .vdw
            .insert("c3".to_string(), (1.9080, 0.1094));
        params
            .vdw
            .insert("hc".to_string(), (1.4870, 0.0157));
        params
    }

    #[test]
    fn test_ffxml_bond_unit_conversion() {
        // Two-atom mol: c3-c3.
        let mol = mol_from(&["C", "C"], vec![make_bond(0, 1)]);
        let params = synthetic_params();
        let types = vec!["c3".to_string(), "c3".to_string()];
        let names = vec!["C1".to_string(), "C2".to_string()];
        let charges = [0.0, 0.0];

        let xml = build_gaff2_ffxml_from_types(&mol, "LIG", &charges, &params, &types, &names);

        // r0_nm = 1.535 * 0.1 = 0.1535 ; k_kj = 300.9 * 2 * 4.184 / 0.01 = 251793.12
        assert!(
            xml.contains(r#"length="0.153500" k="251793.12""#),
            "xml:\n{xml}"
        );
    }

    #[test]
    fn test_ffxml_missing_bond_params_default_and_zero_k() {
        let mol = mol_from(&["C", "C"], vec![make_bond(0, 1)]);
        let params = Gaff2Params::default(); // no bond params at all
        let types = vec!["c3".to_string(), "c3".to_string()];
        let names = vec!["C1".to_string(), "C2".to_string()];
        let charges = [0.0, 0.0];

        let xml = build_gaff2_ffxml_from_types(&mol, "LIG", &charges, &params, &types, &names);
        // r0 default 1.50 Å -> 0.15 nm; kb default 0.0 -> k=0.00
        assert!(
            xml.contains(r#"length="0.150000" k="0.00""#),
            "xml:\n{xml}"
        );
    }

    #[test]
    fn test_ffxml_angle_and_torsion_dedup_chain() {
        // i(0)-j(1)-k(2)-l(3), all c3, straight chain.
        let mol = mol_from(
            &["C", "C", "C", "C"],
            vec![make_bond(0, 1), make_bond(1, 2), make_bond(2, 3)],
        );
        let params = synthetic_params();
        let types = vec![
            "c3".to_string(),
            "c3".to_string(),
            "c3".to_string(),
            "c3".to_string(),
        ];
        let names = vec![
            "C1".to_string(),
            "C2".to_string(),
            "C3".to_string(),
            "C4".to_string(),
        ];
        let charges = [0.0, 0.0, 0.0, 0.0];

        let xml = build_gaff2_ffxml_from_types(&mol, "LIG", &charges, &params, &types, &names);

        // Exactly one angle 0-1-2 and one angle 1-2-3 (no reverse duplicates
        // since the j/k double loop only walks each pair once already).
        let angle_count = xml.matches("<Angle ").count();
        assert_eq!(angle_count, 2, "xml:\n{xml}");

        // Exactly one torsion 0-1-2-3 despite being reachable from either
        // bond direction / neighbor order -- dedup via canonical min(fwd,rev).
        let proper_count = xml.matches("<Proper ").count();
        assert_eq!(proper_count, 1, "xml:\n{xml}");

        // Both torsion terms present (periodicity1/2), each with the right
        // unit-converted k and phase.
        assert!(xml.contains(r#"periodicity1="3""#), "xml:\n{xml}");
        assert!(xml.contains(r#"periodicity2="1""#), "xml:\n{xml}");
        // phase2 = 180 deg -> pi rad
        assert!(xml.contains(&format!(r#"phase2="{:.6}""#, std::f64::consts::PI)));
    }

    #[test]
    fn test_ffxml_nonbonded_conversion_and_fallback() {
        let mol = mol_from(&["C", "Cl"], vec![make_bond(0, 1)]);
        let params = synthetic_params(); // no "cl" vdw entry -> element fallback
        let types = vec!["c3".to_string(), "cl".to_string()];
        let names = vec!["C1".to_string(), "CL1".to_string()];
        let charges = [-0.1, -0.1];

        let xml = build_gaff2_ffxml_from_types(&mol, "LIG", &charges, &params, &types, &names);

        // c3: sigma = 1.9080 * 2*2^(-1/6)*0.1 ; eps = 0.1094*4.184
        let sigma_c3 = 1.9080 * lj_sigma_conv();
        let eps_c3 = 0.1094 * LJ_EPS_CONV;
        assert!(
            xml.contains(&format!(r#"sigma="{sigma_c3:.8}" epsilon="{eps_c3:.6}""#)),
            "xml:\n{xml}"
        );

        // Cl has no vdw entry in synthetic_params and no _ELEM_VDW_DEFAULT
        // entry either (Cl isn't in the Python default table) -> falls back
        // to the generic default (1.9080, 0.0860), same as an unmapped elem.
        let sigma_default = 1.9080 * lj_sigma_conv();
        let eps_default = 0.0860 * LJ_EPS_CONV;
        assert!(
            xml.contains(&format!(
                r#"sigma="{sigma_default:.8}" epsilon="{eps_default:.6}""#
            )),
            "xml:\n{xml}"
        );
    }

    #[test]
    fn test_ffxml_atom_type_mass_fallback_to_element_default() {
        let mol = mol_from(&["N"], vec![]);
        let params = Gaff2Params::default(); // no masses at all
        let types = vec!["n3".to_string()];
        let names = vec!["N1".to_string()];
        let charges = [0.0];

        let xml = build_gaff2_ffxml_from_types(&mol, "LIG", &charges, &params, &types, &names);
        assert!(
            xml.contains(r#"mass="14.0070""#),
            "expected elemental N default mass, xml:\n{xml}"
        );
    }

    // -- build_gaff2_ffxml (public entry point) --------------------------

    #[test]
    fn test_build_gaff2_ffxml_charges_length_mismatch() {
        let mol = mol_from(&["C", "H"], vec![make_bond(0, 1)]);
        let params = synthetic_params();
        let err = build_gaff2_ffxml(&mol, "LIG", &[0.0], DEFAULT_GAFF_VERSION, &params)
            .expect_err("expected charges-length mismatch error");
        match err {
            Gaff2FfxmlError::ChargesLengthMismatch { expected, actual } => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            other => panic!("expected ChargesLengthMismatch, got {other:?}"),
        }
    }
}
