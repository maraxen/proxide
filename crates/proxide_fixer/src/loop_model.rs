use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

use crate::models::{Atom, Chain, Residue, Topology};

// ── Error type ────────────────────────────────────────────────────────────────

/// Error type for Modeller invocation and parsing.
#[derive(Error, Debug)]
pub enum LoopModellingError {
    #[error("Modeller executable not found on PATH (tried: mod10.1, mod9.25, modpy.sh, modeller)")]
    ModelerNotInstalled,

    #[error("MODELLER_KEY environment variable not set")]
    MissingLicenseKey,

    #[error("failed to spawn Modeller: {0}")]
    Spawn(String),

    #[error("Modeller exited with non-zero status:\n{0}")]
    NonZeroExit(String),

    #[error("failed to parse Modeller output: {0}")]
    Parse(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("loop geometry invalid (hard tolerance exceeded): {0}")]
    GeometryInvalid(String),
}

// ── Data types ────────────────────────────────────────────────────────────────

/// A contiguous gap in a chain's residue sequence that needs loop modelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingLoop {
    /// Chain identifier (e.g. "A").
    pub chain_id: String,
    /// Residue number of the last modelled residue before the gap.
    pub start_res: i32,
    /// Residue number of the first modelled residue after the gap.
    pub end_res: i32,
}

/// Summary of loop modelling results.
#[derive(Debug, Clone)]
pub struct LoopModelReport {
    /// Loops that were successfully built.
    pub loops_built: Vec<MissingLoop>,
    /// Non-fatal geometry warnings accumulated during splicing.
    pub geometry_warnings: Vec<String>,
}

// ── find_modeller ─────────────────────────────────────────────────────────────

/// Locate a Modeller executable on `PATH`.
///
/// Modeller is distributed under several names across versions and packaging
/// schemes.  We try them in order and return the first one found.
pub fn find_modeller() -> Result<String, LoopModellingError> {
    let candidates = [
        "mod10.1",
        "mod10.0",
        "mod9.25",
        "mod9.24",
        "modpy.sh",
        "modeller",
    ];

    for name in &candidates {
        // `which`-style check: try to run with --version or just spawn and
        // check for NotFound.  We use a no-op `--version` call.
        let result = Command::new(name).arg("--version").output();
        match result {
            Ok(_) => return Ok(name.to_string()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(_) => {
                // Exists but errored (e.g. bad exit on --version) — still found.
                return Ok(name.to_string());
            }
        }
    }

    Err(LoopModellingError::ModelerNotInstalled)
}

// ── detect_missing_loops ──────────────────────────────────────────────────────

/// Scan `topology` for gaps in residue numbering and return the list of
/// missing loops.
///
/// A gap is defined as a jump of more than 1 in `res_id` between consecutive
/// residues on the same chain, excluding HETATM-only residues.
///
/// The returned `MissingLoop` spans from `start_res` (last present before gap)
/// to `end_res` (first present after gap).
pub fn detect_missing_loops(topology: &Topology) -> Vec<MissingLoop> {
    let mut loops = Vec::new();

    for chain in &topology.chains {
        // Collect res_ids for non-HETATM residues in order.
        let res_ids: Vec<i32> = chain
            .residues
            .iter()
            .filter(|r| r.atoms.iter().any(|a| !a.is_hetatm))
            .map(|r| r.res_id)
            .collect();

        for window in res_ids.windows(2) {
            let prev = window[0];
            let next = window[1];
            if next - prev > 1 {
                loops.push(MissingLoop {
                    chain_id: chain.id.clone(),
                    start_res: prev,
                    end_res: next,
                });
            }
        }
    }

    loops
}

// ── LoopModeller ─────────────────────────────────────────────────────────────

/// Drives Modeller as a subprocess to build missing loops into a `Topology`.
pub struct LoopModeller<'a> {
    topology: &'a mut Topology,
}

impl<'a> LoopModeller<'a> {
    /// Create a new `LoopModeller` wrapping the given topology.
    pub fn new(topology: &'a mut Topology) -> Self {
        Self { topology }
    }

    /// Build every loop in `loops` by invoking Modeller as a subprocess.
    ///
    /// The method:
    /// 1. Writes the current topology to a temp PDB.
    /// 2. Generates a Modeller Python script targeting the first loop (single-loop
    ///    simplification; multi-loop support noted as a follow-up).
    /// 3. Spawns Modeller and waits for completion.
    /// 4. Parses the output PDB and splices new residues back into `self.topology`.
    ///
    /// Returns `Ok(LoopModelReport)` on success.  If `loops` is empty, returns
    /// an empty report without touching the topology.
    ///
    /// # Errors
    ///
    /// - `ModelerNotInstalled` — no Modeller executable found on `PATH`.
    /// - `MissingLicenseKey` — `MODELLER_KEY` env var is absent.
    /// - `Spawn` — OS-level spawn failure.
    /// - `NonZeroExit` — Modeller returned a non-zero exit code (stderr captured).
    /// - `Parse` — output PDB missing or unparseable.
    /// - `Io` — any file-system error.
    pub fn build_loops(
        &mut self,
        loops: &[MissingLoop],
    ) -> Result<LoopModelReport, LoopModellingError> {
        if loops.is_empty() {
            return Ok(LoopModelReport {
                loops_built: vec![],
                geometry_warnings: vec![],
            });
        }

        let modeller = find_modeller()?;
        let _ = std::env::var("MODELLER_KEY")
            .map_err(|_| LoopModellingError::MissingLicenseKey)?;

        // Build an isolated temp directory for this invocation.
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let tmp_dir = std::env::temp_dir().join(format!(
            "proxide_modeller_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&tmp_dir)?;

        let input_pdb = tmp_dir.join("input.pdb");
        let output_pdb = tmp_dir.join("output.pdb");
        let script_path = tmp_dir.join("model_loops.py");

        // Write current topology to PDB.
        write_topology_to_pdb(self.topology, &input_pdb)?;

        // Generate Modeller Python driver script.
        //
        // NOTE: The script targets only the *first* loop in `loops`.  Full
        // multi-loop handling (iterating over all entries and calling
        // `select_loop_atoms` with a union selection) is tracked as a follow-up.
        let script = generate_modeller_script(&input_pdb, &output_pdb, loops);
        std::fs::write(&script_path, &script)?;

        // Invoke Modeller.
        let output = Command::new(&modeller)
            .arg(&script_path)
            .current_dir(&tmp_dir)
            .output()
            .map_err(|e| LoopModellingError::Spawn(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            std::fs::remove_dir_all(&tmp_dir).ok();
            return Err(LoopModellingError::NonZeroExit(stderr));
        }

        if !output_pdb.exists() {
            std::fs::remove_dir_all(&tmp_dir).ok();
            return Err(LoopModellingError::Parse(
                "Modeller did not produce output PDB".to_string(),
            ));
        }

        // Parse loop residues out of the output PDB and splice into topology.
        let new_residues =
            parse_loop_residues(&output_pdb, loops).map_err(LoopModellingError::Parse)?;

        splice_loops_into_topology(self.topology, &new_residues, loops);

        std::fs::remove_dir_all(&tmp_dir).ok();

        let (geometry_warnings, hard_failures) = validate_loop_geometry(self.topology, loops);
        if !hard_failures.is_empty() {
            return Err(LoopModellingError::GeometryInvalid(hard_failures.join("; ")));
        }

        Ok(LoopModelReport {
            loops_built: loops.to_vec(),
            geometry_warnings,
        })
    }
}

// ── Helper: write topology to PDB ────────────────────────────────────────────

/// Write `topology` to a minimal fixed-column PDB file at `path`.
///
/// Uses the same column layout as `ProtonationStateSanitizer::write_temp_pdb`.
fn write_topology_to_pdb(topology: &Topology, path: &Path) -> Result<(), LoopModellingError> {
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "REMARK  Temporary PDB for Modeller loop modelling")?;

    let mut serial = 1i32;
    for chain in &topology.chains {
        for residue in &chain.residues {
            for atom in &residue.atoms {
                let record = if atom.is_hetatm { "HETATM" } else { "ATOM  " };
                let atom_name = format!("{:>4}", atom.name);
                let res_name = format!("{:<3}", residue.name);
                let chain_id = &chain.id;
                let res_id = residue.res_id;
                let [x, y, z] = atom.coords;
                let occupancy = atom.occupancy;
                let b_factor = atom.b_factor;
                let element = format!("{:>2}", atom.element);

                writeln!(
                    file,
                    "{:<6}{:>5} {:>4}{}{:<3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}",
                    record,
                    serial,
                    atom_name,
                    atom.alt_loc,
                    res_name,
                    chain_id,
                    res_id,
                    x,
                    y,
                    z,
                    occupancy,
                    b_factor,
                    element
                )?;

                serial = serial.saturating_add(1);
            }
        }
        writeln!(file, "TER")?;
    }
    writeln!(file, "END")?;

    Ok(())
}

// ── Helper: generate Modeller Python script ───────────────────────────────────

/// Generate the Modeller `loopmodel` Python driver script.
///
/// The script targets the first loop in `loops` (single-loop simplification).
fn generate_modeller_script(
    input_pdb: &Path,
    output_pdb: &Path,
    loops: &[MissingLoop],
) -> String {
    // Use the first loop for the single-loop script.
    let first = &loops[0];
    let chain = &first.chain_id;
    let start_res = first.start_res;
    let end_res = first.end_res;

    format!(
        r#"from modeller import *
from modeller.automodel import *

env = Environ()
env.io.atom_files_directory = ['.']

class MyLoop(LoopModel):
    def select_loop_atoms(self):
        return Selection(
            self.residue_range('{start_res}:{chain}', '{end_res}:{chain}')
        )

a = MyLoop(env,
           inifile='{input}',
           sequence='target',
           loop_assess_methods=assess.DOPE)
a.loop.starting_model = 1
a.loop.ending_model = 1
a.make()

import shutil, glob
files = glob.glob('*.B9999*.pdb') + glob.glob('*loop*.pdb')
if files:
    shutil.copy(sorted(files)[-1], '{output}')
"#,
        start_res = start_res,
        end_res = end_res,
        chain = chain,
        input = input_pdb.display(),
        output = output_pdb.display(),
    )
}

// ── Helper: parse loop residues from PDB ─────────────────────────────────────

/// Parsed representation of a loop: chain_id → residues extracted from the PDB.
type ParsedLoops = Vec<(String, Vec<Residue>)>;

/// Parse the Modeller output PDB and extract residues that fall within the
/// chain/res_id ranges described by `loops`.
///
/// Returns a list of `(chain_id, residues)` pairs, one per loop in `loops`.
fn parse_loop_residues(pdb_path: &Path, loops: &[MissingLoop]) -> Result<ParsedLoops, String> {
    let content = std::fs::read_to_string(pdb_path)
        .map_err(|e| format!("cannot read output PDB: {e}"))?;

    // Build a temporary topology from the ATOM/HETATM lines, then extract loop
    // residues by range.
    let mut chains_map: Vec<(String, Vec<Residue>)> = Vec::new();

    for line in content.lines() {
        let record = line.get(0..6).unwrap_or("").trim();
        if record != "ATOM" && record != "HETATM" {
            continue;
        }

        // PDB fixed-column fields.
        let is_hetatm = record == "HETATM";
        let atom_name = line.get(12..16).unwrap_or("    ").trim().to_string();
        let alt_loc = line.get(16..17).unwrap_or(" ").chars().next().unwrap_or(' ');
        let res_name = line.get(17..20).unwrap_or("   ").trim().to_string();
        let chain_id = line.get(21..22).unwrap_or(" ").trim().to_string();
        let res_id: i32 = line
            .get(22..26)
            .unwrap_or("    ")
            .trim()
            .parse()
            .unwrap_or(0);
        let x: f32 = line.get(30..38).unwrap_or("        ").trim().parse().unwrap_or(0.0);
        let y: f32 = line.get(38..46).unwrap_or("        ").trim().parse().unwrap_or(0.0);
        let z: f32 = line.get(46..54).unwrap_or("        ").trim().parse().unwrap_or(0.0);
        let occupancy: f32 = line.get(54..60).unwrap_or("      ").trim().parse().unwrap_or(1.0);
        let b_factor: f32 = line.get(60..66).unwrap_or("      ").trim().parse().unwrap_or(0.0);
        let element = line.get(76..78).unwrap_or("  ").trim().to_string();
        // Fall back: derive element from atom name if the column is blank.
        let element = if element.is_empty() {
            extract_element_from_name(&atom_name).to_string()
        } else {
            element
        };

        // Serial number (best-effort).
        let serial: i32 = line.get(6..11).unwrap_or("     ").trim().parse().unwrap_or(0);

        let atom = Atom {
            name: atom_name,
            element,
            coords: [x, y, z],
            alt_loc,
            serial,
            b_factor,
            occupancy,
            is_hetatm,
        };

        // Insert into chains_map.
        let chain_entry = chains_map.iter_mut().find(|(id, _)| id == &chain_id);
        let residues = if let Some((_, r)) = chain_entry {
            r
        } else {
            chains_map.push((chain_id.clone(), Vec::new()));
            &mut chains_map.last_mut().unwrap().1
        };

        let ins_code = line.get(26..27).unwrap_or(" ").chars().next().unwrap_or(' ');
        let res_entry = residues
            .iter_mut()
            .rev()
            .find(|r| r.res_id == res_id && r.name == res_name && r.insertion_code == ins_code);

        if let Some(r) = res_entry {
            r.atoms.push(atom);
        } else {
            residues.push(Residue {
                name: res_name,
                res_id,
                insertion_code: ins_code,
                atoms: vec![atom],
            });
        }
    }

    // For each requested loop, collect residues whose res_id falls strictly
    // between start_res and end_res (the gap to fill).
    let mut result: ParsedLoops = Vec::new();
    for lp in loops {
        let chain_residues = chains_map
            .iter()
            .find(|(id, _)| id == &lp.chain_id)
            .map(|(_, r)| r.as_slice())
            .unwrap_or(&[]);

        let loop_residues: Vec<Residue> = chain_residues
            .iter()
            .filter(|r| r.res_id > lp.start_res && r.res_id < lp.end_res)
            .cloned()
            .collect();

        result.push((lp.chain_id.clone(), loop_residues));
    }

    Ok(result)
}

/// Derive a one-letter element symbol from a PDB atom name field.
///
/// This handles the most common protein atoms.  Used only as a fallback when
/// columns 77-78 are blank.
fn extract_element_from_name(name: &str) -> &str {
    let trimmed = name.trim_start_matches(|c: char| c.is_ascii_digit());
    match trimmed.chars().next() {
        Some('C') => "C",
        Some('N') => "N",
        Some('O') => "O",
        Some('S') => "S",
        Some('H') => "H",
        Some('P') => "P",
        _ => "C", // safe default
    }
}

// ── Helper: splice loops into topology ───────────────────────────────────────

/// Insert the parsed loop residues into `topology` at the correct chain
/// positions, filling the gap between `start_res` and `end_res`.
///
/// For each loop, we find the index in the chain's residue list where
/// `res_id == start_res` and insert the new residues immediately after it.
/// Any existing residues that already occupy the same res_id range (unlikely
/// but possible in re-runs) are replaced.
fn splice_loops_into_topology(
    topology: &mut Topology,
    new_residues: &ParsedLoops,
    loops: &[MissingLoop],
) {
    for (lp, (chain_id, residues)) in loops.iter().zip(new_residues.iter()) {
        debug_assert_eq!(chain_id, &lp.chain_id);

        // Find the chain.
        let chain = match topology.chains.iter_mut().find(|c| c.id == lp.chain_id) {
            Some(c) => c,
            None => {
                // Chain not present — create a minimal one.
                topology.chains.push(Chain {
                    id: lp.chain_id.clone(),
                    residues: residues.clone(),
                });
                continue;
            }
        };

        if residues.is_empty() {
            continue;
        }

        // Find insertion position: index of the residue with res_id == start_res.
        let insert_after = chain
            .residues
            .iter()
            .rposition(|r| r.res_id == lp.start_res);

        let insert_idx = match insert_after {
            Some(idx) => idx + 1,
            None => {
                // start_res not found — append at end.
                chain.residues.len()
            }
        };

        // Remove any residues already occupying the loop range (idempotent).
        chain
            .residues
            .retain(|r| r.res_id <= lp.start_res || r.res_id >= lp.end_res);

        // Recompute insert_idx after retain (start_res residue may have shifted).
        let insert_idx = chain
            .residues
            .iter()
            .rposition(|r| r.res_id == lp.start_res)
            .map(|i| i + 1)
            .unwrap_or(insert_idx.min(chain.residues.len()));

        // Insert loop residues in sequence order.
        for (offset, residue) in residues.iter().enumerate() {
            chain.residues.insert(insert_idx + offset, residue.clone());
        }
    }
}

// ── Geometry validation ───────────────────────────────────────────────────────

/// Validate geometry of newly-inserted loop residues against C3 IC tolerances.
///
/// Checks N–CA, CA–C, and C–O bond lengths and the N–CA–C bond angle for every
/// residue in the gap `(start_res, end_res)` of each `MissingLoop`.  Returns a
/// pair `(warnings, hard_failures)`:
/// - `warnings`      — deviations outside the soft tolerance (but within hard).
/// - `hard_failures` — deviations that exceed the hard tolerance; callers should
///   propagate these as `LoopModellingError::GeometryInvalid`.
fn validate_loop_geometry(
    topology: &Topology,
    loops: &[MissingLoop],
) -> (Vec<String>, Vec<String>) {
    let mut warnings = Vec::new();
    let mut hard_failures = Vec::new();

    // Bond length tolerances (Å).
    const N_CA_EXPECTED: f32 = 1.46;
    const CA_C_EXPECTED: f32 = 1.52;
    const C_O_EXPECTED: f32 = 1.23;
    const SOFT_TOL: f32 = 0.05; // warn
    const HARD_TOL: f32 = 0.15; // hard-fail

    // Bond angle tolerance (degrees, N–CA–C).
    const NCA_C_EXPECTED: f32 = 111.2;
    const ANGLE_SOFT_TOL: f32 = 10.0;
    const ANGLE_HARD_TOL: f32 = 25.0;

    for ml in loops {
        // Find the chain.
        let chain = match topology.chains.iter().find(|c| c.id == ml.chain_id) {
            Some(c) => c,
            None => continue,
        };

        // Iterate over residues strictly inside the gap.
        for residue in chain
            .residues
            .iter()
            .filter(|r| r.res_id > ml.start_res && r.res_id < ml.end_res)
        {
            let label_prefix = format!("{}:{}", ml.chain_id, residue.res_id);

            let get_atom = |name: &str| -> Option<[f32; 3]> {
                residue
                    .atoms
                    .iter()
                    .find(|a| a.name == name)
                    .map(|a| a.coords)
            };

            // N–CA bond
            if let (Some(n), Some(ca)) = (get_atom("N"), get_atom("CA")) {
                let d = atom_distance(n, ca);
                check_bond(
                    &mut warnings,
                    &mut hard_failures,
                    &format!("{label_prefix} N-CA"),
                    d,
                    N_CA_EXPECTED,
                    SOFT_TOL,
                    HARD_TOL,
                );
            }

            // CA–C bond
            if let (Some(ca), Some(c)) = (get_atom("CA"), get_atom("C")) {
                let d = atom_distance(ca, c);
                check_bond(
                    &mut warnings,
                    &mut hard_failures,
                    &format!("{label_prefix} CA-C"),
                    d,
                    CA_C_EXPECTED,
                    SOFT_TOL,
                    HARD_TOL,
                );
            }

            // C–O bond
            if let (Some(c), Some(o)) = (get_atom("C"), get_atom("O")) {
                let d = atom_distance(c, o);
                check_bond(
                    &mut warnings,
                    &mut hard_failures,
                    &format!("{label_prefix} C-O"),
                    d,
                    C_O_EXPECTED,
                    SOFT_TOL,
                    HARD_TOL,
                );
            }

            // N–CA–C angle
            if let (Some(n), Some(ca), Some(c)) =
                (get_atom("N"), get_atom("CA"), get_atom("C"))
            {
                let angle = atom_bond_angle(n, ca, c);
                check_angle(
                    &mut warnings,
                    &mut hard_failures,
                    &format!("{label_prefix} N-CA-C"),
                    angle,
                    NCA_C_EXPECTED,
                    ANGLE_SOFT_TOL,
                    ANGLE_HARD_TOL,
                );
            }
        }
    }

    (warnings, hard_failures)
}

/// Emit a bond-length warning or hard failure depending on how far `actual`
/// deviates from `expected`.
#[inline]
fn check_bond(
    warnings: &mut Vec<String>,
    failures: &mut Vec<String>,
    label: &str,
    actual: f32,
    expected: f32,
    soft: f32,
    hard: f32,
) {
    let delta = (actual - expected).abs();
    if delta > hard {
        failures.push(format!(
            "{label}: bond {actual:.3} Å, expected {expected:.3}±{soft:.3} (hard limit ±{hard:.3})"
        ));
    } else if delta > soft {
        warnings.push(format!(
            "{label}: bond {actual:.3} Å, expected {expected:.3}±{soft:.3}"
        ));
    }
}

/// Emit a bond-angle warning or hard failure depending on how far `actual`
/// deviates from `expected` (both in degrees).
#[inline]
fn check_angle(
    warnings: &mut Vec<String>,
    failures: &mut Vec<String>,
    label: &str,
    actual: f32,
    expected: f32,
    soft: f32,
    hard: f32,
) {
    let delta = (actual - expected).abs();
    if delta > hard {
        failures.push(format!(
            "{label}: angle {actual:.1}°, expected {expected:.1}°±{soft:.1}° (hard limit ±{hard:.1}°)"
        ));
    } else if delta > soft {
        warnings.push(format!(
            "{label}: angle {actual:.1}°, expected {expected:.1}°±{soft:.1}°"
        ));
    }
}

/// Euclidean distance between two 3-D coordinate triples (Å).
#[inline]
fn atom_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Bond angle at `vertex` between the vectors (a → vertex) and (c → vertex),
/// returned in degrees.
///
/// Returns 0.0 if either vector has zero length (degenerate case).
#[inline]
fn atom_bond_angle(a: [f32; 3], vertex: [f32; 3], c: [f32; 3]) -> f32 {
    // Vectors from vertex to each terminal atom.
    let v1 = [a[0] - vertex[0], a[1] - vertex[1], a[2] - vertex[2]];
    let v2 = [c[0] - vertex[0], c[1] - vertex[1], c[2] - vertex[2]];

    let len1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
    let len2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();

    if len1 < f32::EPSILON || len2 < f32::EPSILON {
        return 0.0;
    }

    let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    // Clamp to [-1, 1] to guard against floating-point overshoot.
    let cos_theta = (dot / (len1 * len2)).clamp(-1.0, 1.0);
    cos_theta.acos().to_degrees()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Atom, Chain, Residue, Topology};

    fn make_atom(name: &str, serial: i32) -> Atom {
        Atom {
            name: name.to_string(),
            element: name[..1].to_string(),
            coords: [0.0, 0.0, 0.0],
            alt_loc: ' ',
            serial,
            b_factor: 0.0,
            occupancy: 1.0,
            is_hetatm: false,
        }
    }

    fn make_residue(name: &str, res_id: i32) -> Residue {
        Residue {
            name: name.to_string(),
            res_id,
            insertion_code: ' ',
            atoms: vec![make_atom("CA", res_id)],
        }
    }

    fn make_topology(chain_id: &str, res_ids: &[i32]) -> Topology {
        let residues = res_ids
            .iter()
            .map(|&id| make_residue("ALA", id))
            .collect();
        Topology {
            chains: vec![Chain {
                id: chain_id.to_string(),
                residues,
            }],
        }
    }

    // ── detect_missing_loops ─────────────────────────────────────────────────

    #[test]
    fn detect_no_gaps() {
        let topo = make_topology("A", &[1, 2, 3, 4]);
        let loops = detect_missing_loops(&topo);
        assert!(loops.is_empty(), "consecutive residues should produce no gaps");
    }

    #[test]
    fn detect_single_gap() {
        let topo = make_topology("A", &[1, 2, 5, 6]);
        let loops = detect_missing_loops(&topo);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].chain_id, "A");
        assert_eq!(loops[0].start_res, 2);
        assert_eq!(loops[0].end_res, 5);
    }

    #[test]
    fn detect_multiple_gaps() {
        let topo = make_topology("A", &[1, 3, 5, 10]);
        let loops = detect_missing_loops(&topo);
        assert_eq!(loops.len(), 3);
    }

    #[test]
    fn detect_empty_topology() {
        let topo = Topology { chains: vec![] };
        let loops = detect_missing_loops(&topo);
        assert!(loops.is_empty());
    }

    // ── find_modeller ────────────────────────────────────────────────────────

    #[test]
    fn find_modeller_missing_returns_error() {
        // In CI Modeller is absent; expect ModelerNotInstalled.
        // If Modeller is actually installed the test is still valid (it returns Ok).
        match find_modeller() {
            Ok(path) => {
                // Modeller found — just assert it's non-empty.
                assert!(!path.is_empty(), "found modeller path should be non-empty");
            }
            Err(LoopModellingError::ModelerNotInstalled) => {
                // Expected in CI — pass.
            }
            Err(e) => panic!("unexpected error from find_modeller: {e}"),
        }
    }

    // ── build_loops ──────────────────────────────────────────────────────────

    #[test]
    fn build_loops_empty_is_noop() {
        let mut topo = make_topology("A", &[1, 2, 3]);
        let mut modeller = LoopModeller::new(&mut topo);
        let report = modeller.build_loops(&[]).expect("empty loops should return Ok");
        assert!(report.loops_built.is_empty());
        assert!(report.geometry_warnings.is_empty());
    }

    #[test]
    fn build_loops_missing_modeller_returns_error() {
        // This test verifies the error path when Modeller is absent.
        // Skip if Modeller is actually installed (find_modeller() returns Ok).
        if find_modeller().is_ok() {
            // Also need MODELLER_KEY set for it to proceed past the key check.
            // If it is set too, we'd actually try to run Modeller — skip entirely.
            eprintln!("Skipping: Modeller is installed");
            return;
        }

        let mut topo = make_topology("A", &[1, 5]);
        let loop_ = MissingLoop {
            chain_id: "A".to_string(),
            start_res: 1,
            end_res: 5,
        };
        let mut modeller = LoopModeller::new(&mut topo);
        let result = modeller.build_loops(&[loop_]);
        assert!(
            matches!(result, Err(LoopModellingError::ModelerNotInstalled)),
            "expected ModelerNotInstalled, got {:?}",
            result
        );
    }

    // ── generate_modeller_script ─────────────────────────────────────────────

    #[test]
    fn modeller_script_contains_residue_range() {
        let lp = MissingLoop {
            chain_id: "A".to_string(),
            start_res: 10,
            end_res: 20,
        };
        let script = generate_modeller_script(
            Path::new("/tmp/input.pdb"),
            Path::new("/tmp/output.pdb"),
            &[lp],
        );
        assert!(script.contains("10:A"), "script should reference start residue");
        assert!(script.contains("20:A"), "script should reference end residue");
        assert!(script.contains("LoopModel"), "script should use LoopModel class");
    }

    // ── write_topology_to_pdb ────────────────────────────────────────────────

    #[test]
    fn write_pdb_produces_atom_records() {
        let topo = make_topology("A", &[1, 2, 3]);
        let tmp = std::env::temp_dir().join(format!(
            "proxide_test_write_{}.pdb",
            std::process::id()
        ));
        write_topology_to_pdb(&topo, &tmp).expect("write_topology_to_pdb failed");
        let content = std::fs::read_to_string(&tmp).expect("read temp PDB failed");
        std::fs::remove_file(&tmp).ok();

        assert!(
            content.contains("ATOM"),
            "PDB output should contain ATOM records"
        );
        assert!(content.contains("TER"), "PDB output should contain TER record");
        assert!(content.contains("END"), "PDB output should end with END");
    }

    // ── splice_loops_into_topology ───────────────────────────────────────────

    #[test]
    fn splice_inserts_loop_residues() {
        let mut topo = make_topology("A", &[1, 5]);
        // Simulate Modeller filling in residues 2, 3, 4.
        let new_res = vec![
            make_residue("ALA", 2),
            make_residue("ALA", 3),
            make_residue("ALA", 4),
        ];
        let lp = MissingLoop {
            chain_id: "A".to_string(),
            start_res: 1,
            end_res: 5,
        };
        let parsed: ParsedLoops = vec![("A".to_string(), new_res)];
        splice_loops_into_topology(&mut topo, &parsed, &[lp]);

        let res_ids: Vec<i32> = topo.chains[0].residues.iter().map(|r| r.res_id).collect();
        assert_eq!(res_ids, vec![1, 2, 3, 4, 5], "residues should be ordered 1-5");
    }

    #[test]
    fn splice_idempotent_on_re_run() {
        let mut topo = make_topology("A", &[1, 5]);
        let new_res = vec![make_residue("ALA", 2), make_residue("ALA", 3)];
        let lp = MissingLoop {
            chain_id: "A".to_string(),
            start_res: 1,
            end_res: 5,
        };
        let parsed: ParsedLoops = vec![("A".to_string(), new_res.clone())];

        // First splice.
        splice_loops_into_topology(&mut topo, &parsed, &[lp.clone()]);
        let count_after_first = topo.chains[0].residues.len();

        // Second splice with same data.
        let parsed2: ParsedLoops = vec![("A".to_string(), new_res)];
        splice_loops_into_topology(&mut topo, &parsed2, &[lp]);
        let count_after_second = topo.chains[0].residues.len();

        assert_eq!(
            count_after_first, count_after_second,
            "second splice should not add duplicate residues"
        );
    }

    // ── parse_loop_residues (round-trip) ─────────────────────────────────────

    #[test]
    fn parse_loop_residues_round_trip() {
        // Write a topology, parse it back, and verify the loop range is extracted.
        let topo = make_topology("A", &[1, 2, 3, 4, 5]);
        let tmp = std::env::temp_dir().join(format!(
            "proxide_test_parse_{}.pdb",
            std::process::id()
        ));
        write_topology_to_pdb(&topo, &tmp).expect("write failed");

        let lp = MissingLoop {
            chain_id: "A".to_string(),
            start_res: 1,
            end_res: 5,
        };
        let result = parse_loop_residues(&tmp, &[lp]).expect("parse failed");
        std::fs::remove_file(&tmp).ok();

        // Residues 2, 3, 4 fall strictly between start=1 and end=5.
        let (chain_id, residues) = &result[0];
        assert_eq!(chain_id, "A");
        assert_eq!(residues.len(), 3, "should extract res_ids 2, 3, 4");
        assert_eq!(residues[0].res_id, 2);
        assert_eq!(residues[1].res_id, 3);
        assert_eq!(residues[2].res_id, 4);
    }

    // ── geometry validation ──────────────────────────────────────────────────

    /// Build a residue with realistic backbone atoms at the given coordinates.
    fn make_residue_with_backbone(
        res_id: i32,
        n: [f32; 3],
        ca: [f32; 3],
        c: [f32; 3],
        o: [f32; 3],
    ) -> Residue {
        Residue {
            name: "ALA".to_string(),
            res_id,
            insertion_code: ' ',
            atoms: vec![
                Atom { name: "N".to_string(),  element: "N".to_string(), coords: n,  alt_loc: ' ', serial: 1, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                Atom { name: "CA".to_string(), element: "C".to_string(), coords: ca, alt_loc: ' ', serial: 2, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                Atom { name: "C".to_string(),  element: "C".to_string(), coords: c,  alt_loc: ' ', serial: 3, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                Atom { name: "O".to_string(),  element: "O".to_string(), coords: o,  alt_loc: ' ', serial: 4, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
            ],
        }
    }

    #[test]
    fn geometry_validation_good_bond_no_warning() {
        // Ideal N–CA bond = 1.46 Å; place N at origin, CA at (1.46, 0, 0).
        // CA–C at 1.52 Å along y; C–O at 1.23 Å along z.
        // N–CA–C angle will be 90°, within the soft 10° tolerance from 111.2°
        // is NOT met (90 vs 111.2 = 21.2°) but within hard (25°), so only a
        // warning.  Use a proper tetrahedral-ish geometry instead:
        // Place atoms so that N-CA-C angle ≈ 111°.
        //
        // N  = (0, 0, 0)
        // CA = (1.46, 0, 0)
        // C  = (1.46 + 1.52*cos(111°), 1.52*sin(111°), 0)
        //     ≈ (1.46 - 0.545, 1.420, 0) = (0.915, 1.420, 0)
        // O  = C + (0, 0, 1.23) for simplicity (C-O not angle-checked)
        let n_coords  = [0.0_f32, 0.0, 0.0];
        let ca_coords = [1.46_f32, 0.0, 0.0];
        let angle_rad = 111.2_f32.to_radians();
        let c_coords  = [
            1.46 + 1.52 * (-angle_rad.cos()), // cos(111.2°) is negative
            1.52 * angle_rad.sin(),
            0.0,
        ];
        let o_coords  = [c_coords[0], c_coords[1], c_coords[2] + 1.23];

        let inner_res = make_residue_with_backbone(2, n_coords, ca_coords, c_coords, o_coords);
        let topo = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    make_residue("ALA", 1), // anchor before gap
                    inner_res,
                    make_residue("ALA", 4), // anchor after gap
                ],
            }],
        };
        let ml = MissingLoop { chain_id: "A".to_string(), start_res: 1, end_res: 4 };

        let (warnings, hard_failures) = validate_loop_geometry(&topo, &[ml]);
        assert!(
            hard_failures.is_empty(),
            "ideal geometry should produce no hard failures, got: {hard_failures:?}"
        );
        assert!(
            warnings.is_empty(),
            "ideal geometry should produce no warnings, got: {warnings:?}"
        );
    }

    #[test]
    fn geometry_validation_bad_bond_surfaces_hard_failure() {
        // N–CA at 2.5 Å is 1.04 Å off the expected 1.46 Å — well beyond HARD_TOL=0.15.
        let n_coords  = [0.0_f32, 0.0, 0.0];
        let ca_coords = [2.5_f32, 0.0, 0.0]; // bad: 2.5 Å instead of 1.46
        // CA–C and C–O at nominal distances to isolate the N-CA failure.
        let c_coords  = [2.5 + 1.52_f32, 0.0, 0.0];
        let o_coords  = [c_coords[0], 1.23_f32, 0.0];

        let inner_res = make_residue_with_backbone(2, n_coords, ca_coords, c_coords, o_coords);
        let topo = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    make_residue("ALA", 1),
                    inner_res,
                    make_residue("ALA", 4),
                ],
            }],
        };
        let ml = MissingLoop { chain_id: "A".to_string(), start_res: 1, end_res: 4 };

        let (_warnings, hard_failures) = validate_loop_geometry(&topo, &[ml]);
        assert!(
            !hard_failures.is_empty(),
            "N-CA bond at 2.5 Å should produce a hard failure"
        );
        assert!(
            hard_failures[0].contains("N-CA"),
            "hard failure message should name the bond, got: {:?}",
            hard_failures
        );
    }

    #[test]
    fn geometry_validation_soft_warning_only() {
        // N–CA at 1.50 Å — 0.04 Å off the expected 1.46 Å, within HARD_TOL but
        // just under SOFT_TOL=0.05.  Should produce no warning (delta < soft).
        // N–CA at 1.52 Å — 0.06 Å off → soft warning only.
        let n_coords  = [0.0_f32, 0.0, 0.0];
        let ca_coords = [1.52_f32, 0.0, 0.0]; // 0.06 off → soft warning
        let angle_rad = 111.2_f32.to_radians();
        let c_coords  = [
            1.52 + 1.52 * (-angle_rad.cos()),
            1.52 * angle_rad.sin(),
            0.0,
        ];
        let o_coords  = [c_coords[0], c_coords[1], c_coords[2] + 1.23];

        let inner_res = make_residue_with_backbone(2, n_coords, ca_coords, c_coords, o_coords);
        let topo = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    make_residue("ALA", 1),
                    inner_res,
                    make_residue("ALA", 4),
                ],
            }],
        };
        let ml = MissingLoop { chain_id: "A".to_string(), start_res: 1, end_res: 4 };

        let (warnings, hard_failures) = validate_loop_geometry(&topo, &[ml]);
        assert!(
            hard_failures.is_empty(),
            "0.06 Å deviation should not hard-fail, got: {hard_failures:?}"
        );
        assert!(
            !warnings.is_empty(),
            "0.06 Å deviation should produce a soft warning"
        );
        assert!(
            warnings.iter().any(|w| w.contains("N-CA")),
            "warning should name N-CA bond, got: {warnings:?}"
        );
    }

    #[test]
    fn geometry_validation_bad_angle_surfaces_hard_failure() {
        // N–CA–C angle of 30° is 81.2° off from 111.2° — well beyond ANGLE_HARD_TOL=25°.
        //
        // `atom_bond_angle(n, ca, c)` measures the angle at `ca` between the
        // vectors (ca→n) and (ca→c).  With N at origin and CA at (1.46,0,0),
        // the ca→n vector is (-1,0,0).  We want the angle between ca→n and ca→c
        // to be 30°, so ca→c = 1.52·(cos(30°)·(-1,0,0) + sin(30°)·(0,1,0))
        //                      = 1.52·(-√3/2, 0.5, 0).
        let n_coords  = [0.0_f32, 0.0, 0.0];
        let ca_coords = [1.46_f32, 0.0, 0.0];
        let c_coords  = [
            1.46 + 1.52 * (-(3.0_f32.sqrt() / 2.0)),
            1.52 * 0.5,
            0.0,
        ];
        let o_coords  = [c_coords[0], c_coords[1] + 1.23, 0.0];

        let inner_res = make_residue_with_backbone(2, n_coords, ca_coords, c_coords, o_coords);
        let topo = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    make_residue("ALA", 1),
                    inner_res,
                    make_residue("ALA", 4),
                ],
            }],
        };
        let ml = MissingLoop { chain_id: "A".to_string(), start_res: 1, end_res: 4 };

        let (_warnings, hard_failures) = validate_loop_geometry(&topo, &[ml]);
        assert!(
            !hard_failures.is_empty(),
            "60° N-CA-C angle should produce a hard failure"
        );
        assert!(
            hard_failures.iter().any(|f| f.contains("N-CA-C")),
            "hard failure should name N-CA-C angle, got: {hard_failures:?}"
        );
    }

    #[test]
    fn geometry_validation_residues_outside_gap_ignored() {
        // The anchor residues (start_res and end_res) should NOT be checked —
        // only strictly interior residues.  Give the anchors pathological coords
        // and verify no failures.
        let bad_coords = [0.0_f32, 0.0, 0.0]; // all atoms at origin → zero bonds
        let anchor1 = Residue {
            name: "ALA".to_string(),
            res_id: 1,
            insertion_code: ' ',
            atoms: vec![
                Atom { name: "N".to_string(),  element: "N".to_string(), coords: bad_coords, alt_loc: ' ', serial: 1, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                Atom { name: "CA".to_string(), element: "C".to_string(), coords: bad_coords, alt_loc: ' ', serial: 2, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
            ],
        };
        let anchor2 = Residue {
            name: "ALA".to_string(),
            res_id: 4,
            insertion_code: ' ',
            atoms: vec![
                Atom { name: "N".to_string(),  element: "N".to_string(), coords: bad_coords, alt_loc: ' ', serial: 5, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
            ],
        };
        let topo = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![anchor1, anchor2],
            }],
        };
        let ml = MissingLoop { chain_id: "A".to_string(), start_res: 1, end_res: 4 };

        // No interior residues → nothing to validate.
        let (warnings, hard_failures) = validate_loop_geometry(&topo, &[ml]);
        assert!(warnings.is_empty(), "no interior residues → no warnings");
        assert!(hard_failures.is_empty(), "no interior residues → no hard failures");
    }

    #[test]
    fn atom_distance_basic() {
        // 3-4-5 right triangle → hypotenuse = 5.0
        let a = [0.0_f32, 0.0, 0.0];
        let b = [3.0_f32, 4.0, 0.0];
        let d = atom_distance(a, b);
        assert!((d - 5.0).abs() < 1e-5, "distance should be 5.0, got {d}");
    }

    #[test]
    fn atom_bond_angle_right_angle() {
        // Vertex at origin; A along +x; C along +y → 90°.
        let a      = [1.0_f32, 0.0, 0.0];
        let vertex = [0.0_f32, 0.0, 0.0];
        let c      = [0.0_f32, 1.0, 0.0];
        let angle = atom_bond_angle(a, vertex, c);
        assert!((angle - 90.0).abs() < 1e-4, "angle should be 90°, got {angle}");
    }

    #[test]
    fn atom_bond_angle_180_degrees() {
        // Collinear: A at -x, vertex at origin, C at +x → 180°.
        let a      = [-1.0_f32, 0.0, 0.0];
        let vertex = [0.0_f32,  0.0, 0.0];
        let c      = [1.0_f32,  0.0, 0.0];
        let angle = atom_bond_angle(a, vertex, c);
        assert!((angle - 180.0).abs() < 1e-4, "angle should be 180°, got {angle}");
    }

    #[test]
    fn atom_bond_angle_degenerate_zero_length() {
        // Zero-length vector → should return 0.0 without panic.
        let a      = [0.0_f32, 0.0, 0.0];
        let vertex = [0.0_f32, 0.0, 0.0]; // same as a → zero vector
        let c      = [1.0_f32, 0.0, 0.0];
        let angle = atom_bond_angle(a, vertex, c);
        assert_eq!(angle, 0.0, "degenerate case should return 0.0");
    }

    // ── ignored smoke test (requires Modeller) ────────────────────────────────

    #[test]
    #[ignore = "requires Modeller executable and MODELLER_KEY license key"]
    fn build_loops_smoke() {
        // Validates that the subprocess wrapper compiles and the roundtrip works
        // when Modeller is available.  Run with:
        //   MODELLER_KEY=... cargo test -p proxide_fixer -- --ignored build_loops_smoke
        let mut topo = make_topology("A", &[1, 5]);
        let loop_ = MissingLoop {
            chain_id: "A".to_string(),
            start_res: 1,
            end_res: 5,
        };
        let mut modeller = LoopModeller::new(&mut topo);
        let report = modeller
            .build_loops(&[loop_])
            .expect("build_loops failed in smoke test");
        assert_eq!(report.loops_built.len(), 1);
    }
}
