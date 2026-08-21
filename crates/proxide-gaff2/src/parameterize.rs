//! End-to-end GAFF2 parameterization.
//!
//! Port of `parameterize_gaff_with_rdkit()` (`src/proxide/chem/gaff2.py`,
//! lines 1737-1905) — the top-level wrapper that assigns atom types, then
//! walks the molecule's topology to look up bond, angle, torsion, mass and
//! charge parameters.
//!
//! This module owns the topology walk + inline parameter lookups only. Its
//! three dependencies each already have their own port elsewhere in this
//! crate, and this module composes them rather than re-implementing any of
//! them:
//!
//! - Atom typing: [`crate::orchestrate::assign_gaff2_atom_types`].
//! - The `gaff-2.2.20.dat` parameter-table loader (`load_gaff2_parameters`,
//!   gaff2.py:1260-1384): [`crate::param_loader`].
//! - Charges (`_get_espaloma_charges`, gaff2.py:1680-1735):
//!   [`crate::charges::get_espaloma_charges`].
//!
//! `extract_atom_features` (gaff2.py:863-973) is intentionally not ported
//! anywhere: recon confirmed it is dead code, superseded by
//! `_atom_bond_facts` (ported in `atom_bond_facts.rs`), with no callers
//! anywhere in the module, including `parameterize_gaff_with_rdkit`.
//!
//! # `gaff_version` is intentionally unused
//!
//! Python's `parameterize_gaff_with_rdkit(mol, gaff_version="gaff-2.2.20")`
//! accepts a `gaff_version` argument but never forwards it: it always calls
//! `load_gaff2_parameters()` with zero arguments, which always resolves the
//! bundled `gaff-2.2.20.dat` regardless of what `gaff_version` was passed.
//! This is a known residual mismatch in the Python source (not something to
//! "fix" during a behavior-preserving port) — [`parameterize_gaff2`] keeps
//! the parameter for signature parity but likewise never uses it.
//!
//! # Two lookup helpers exist in this crate and this module intentionally
//! # calls neither
//!
//! `param_lookup.rs` ports `_lookup_bond_params`/`_lookup_angle_params`
//! (gaff2.py:1448-1477) — bidirectional lookup with a `_BOND_TYPE_SUB`
//! substitution fallback. `parameterize_gaff_with_rdkit`'s own inline bond
//! lookup (gaff2.py:1825-1834) and angle lookup (gaff2.py:1836-1846) are
//! textually different and strictly narrower: bonds sort the pair once
//! (`tuple(sorted([t1, t2]))`, no substitution fallback) and angles try only
//! the exact `(t1, t2, t3)` order (no reversed probe, no substitution). The
//! two richer helpers are real Python functions, just not ones this
//! particular function calls. Preserved faithfully below — see
//! [`build_topology`]'s inline comments at each lookup site.

use indexmap::IndexMap;

use crate::charges::{get_espaloma_charges, EspalomaAssigner, EspalomaFeaturizer};
use crate::mol::{Bond, BondOrder, MolGraph};
use crate::orchestrate::assign_gaff2_atom_types;
use crate::param_loader::{self, Gaff2Parameters};

// ---------------------------------------------------------------------------
// Topology walk + parameter lookup (port of `parameterize_gaff_with_rdkit`'s body)
// ---------------------------------------------------------------------------

/// One bonded pair, with its assigned GAFF2 types and looked-up bond
/// parameters. Field-for-field port of the `dict` Python builds per bond
/// (gaff2.py:1790-1797, 1825-1834).
#[derive(Debug, Clone, PartialEq)]
pub struct BondRecord {
    pub i: usize,
    pub j: usize,
    /// "sb" | "db" | "tb"
    pub bond_type: &'static str,
    pub order: f64,
    pub gaff_type_i: String,
    pub gaff_type_j: String,
    pub kb: f64,
    pub r0: f64,
}

/// One 1-2-3 angle, with looked-up angle parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct AngleRecord {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub types: (String, String, String),
    pub kt: f64,
    pub t0: f64,
}

/// One 1-2-3-4 torsion, with (possibly substitution-fallback) looked-up
/// torsion terms.
#[derive(Debug, Clone, PartialEq)]
pub struct TorsionRecord {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub l: usize,
    pub types: (String, String, String, String),
    pub params: Vec<(i64, f64, f64)>,
}

/// Full output of a GAFF2 parameterization run — the Rust equivalent of the
/// `dict` `parameterize_gaff_with_rdkit` returns (gaff2.py:1898-1905).
#[derive(Debug, Clone)]
pub struct Gaff2Parameterization {
    pub atom_types: Vec<String>,
    pub charges: Vec<f64>,
    pub masses: IndexMap<String, f64>,
    pub bonds: Vec<BondRecord>,
    pub angles: Vec<AngleRecord>,
    pub torsions: Vec<TorsionRecord>,
}

/// GAFF2 torsion-type substitutions used only as a fallback when an exact
/// torsion-quad lookup misses (gaff2.py:1852-1861). `"x"` (the fallback atom
/// type placeholder) becomes `"hc"`; everything else not in the table passes
/// through unchanged.
///
/// Note this table is textually distinct from `param_lookup::bond_type_sub`
/// (a different Python constant, `_BOND_TYPE_SUB`, gaff2.py:1413-1418, used
/// by the two lookup helpers this module does not call — see module docs):
/// different key sets, different targets (`type_substitutions` maps several
/// nitrogen variants to `"n3"`/`"n"`, which `_BOND_TYPE_SUB` does not do at
/// all), and only this one is reachable from
/// `parameterize_gaff_with_rdkit`.
fn substitute_torsion_type(t: &str) -> String {
    if t == "x" {
        return "hc".to_string();
    }
    match t {
        "cx" | "cy" | "c5" | "c6" => "c3",
        "n7" | "n8" | "nx" | "ny" | "nu" | "nv" => "n3",
        "ni" => "n",
        other => other,
    }
    .to_string()
}

/// Atom type at `idx`, or `"x"` if `idx` is out of range for `atom_types` —
/// port of the `atom_types[i] if i < len(atom_types) else "x"` fallback used
/// throughout the Python function.
fn type_at(atom_types: &[String], idx: usize) -> String {
    atom_types
        .get(idx)
        .cloned()
        .unwrap_or_else(|| "x".to_string())
}

/// Numeric bond order for a Kekulized [`BondOrder`] (1.0/2.0/3.0), the Rust
/// equivalent of RDKit's `bond.GetBondTypeAsDouble()` for a Kekulized bond
/// (no fractional 1.5 aromatic value is reachable here: `MolGraph` requires
/// Kekulized bonds — see `mol.rs` — so the classification below never
/// exercises its `>= 1.4`-but-`< 1.9` branch on anything but an exact 2.0).
fn bond_order_value(order: BondOrder) -> f64 {
    match order {
        BondOrder::Single => 1.0,
        BondOrder::Double => 2.0,
        BondOrder::Triple => 3.0,
    }
}

/// Classify a numeric bond order into GAFF2's sb/db/tb bucket — port of the
/// `bo >= 1.9` / `bo >= 1.4` / else threshold ladder (gaff2.py:1783-1788).
///
/// Preserved residual mismatch: the `>= 1.9` ("tb") check runs BEFORE the
/// `>= 1.4` ("db") check, so a plain double bond (`bo == 2.0`, which
/// satisfies `>= 1.9`) is classified `"tb"`, never `"db"`. The `"db"` bucket
/// is only reachable for `1.4 <= bo < 1.9`, which no Kekulized
/// [`BondOrder`] variant ever produces (see [`bond_order_value`]). This is
/// exactly what `gaff2.py` does; not "fixed" here.
fn classify_bond_type(bo: f64) -> &'static str {
    if bo >= 1.9 {
        "tb"
    } else if bo >= 1.4 {
        "db"
    } else {
        "sb"
    }
}

/// Per-atom adjacency in bond-addition ("molecule") order: for atom `n`,
/// `adjacency[n]` lists `(other_atom, bond)` in the order those bonds appear
/// in `mol.bonds`. This is the Rust stand-in for RDKit's
/// `atom.GetBonds()`, which likewise yields an atom's incident bonds in
/// molecule-addition order — the angle/torsion walk below depends on this
/// order exactly as the Python nested-loop walk depends on RDKit's.
fn build_adjacency(mol: &MolGraph) -> Vec<Vec<(usize, &Bond)>> {
    let n = mol.elements.len();
    let mut adjacency: Vec<Vec<(usize, &Bond)>> = (0..n).map(|_| Vec::new()).collect();
    for bond in &mol.bonds {
        adjacency[bond.i].push((bond.j, bond));
        adjacency[bond.j].push((bond.i, bond));
    }
    adjacency
}

/// Pure topology walk + parameter lookup: builds bonds/angles/torsions/masses
/// from an already-typed molecule. Split out from [`parameterize_gaff2`] so
/// it can be unit-tested without depending on `assign_gaff2_atom_types`
/// (this is an internal decomposition for testability, not a behavior
/// change — [`parameterize_gaff2`] below calls this with exactly the atom
/// types `assign_gaff2_atom_types` produced, same as Python's single
/// function body).
pub fn build_topology(
    mol: &MolGraph,
    atom_types: &[String],
    params: &Gaff2Parameters,
) -> (
    IndexMap<String, f64>,
    Vec<BondRecord>,
    Vec<AngleRecord>,
    Vec<TorsionRecord>,
) {
    let n_atoms = mol.elements.len();
    let adjacency = build_adjacency(mol);

    // --- bonds: one record per entry in mol.bonds, in molecule order -------
    let mut bonds = Vec::with_capacity(mol.bonds.len());
    for bond in &mol.bonds {
        let t_i = type_at(atom_types, bond.i);
        let t_j = type_at(atom_types, bond.j);
        let bo = bond_order_value(bond.order);
        let bond_type = classify_bond_type(bo);

        // key = tuple(sorted([t1, t2])); default (0.0, 0.0) on miss. This is
        // NOT `param_lookup::lookup_bond_params` -- see module docs.
        let key = if t_i <= t_j {
            (t_i.clone(), t_j.clone())
        } else {
            (t_j.clone(), t_i.clone())
        };
        let (kb, r0) = params.bonds.get(&key).copied().unwrap_or((0.0, 0.0));

        bonds.push(BondRecord {
            i: bond.i,
            j: bond.j,
            bond_type,
            order: bo,
            gaff_type_i: t_i,
            gaff_type_j: t_j,
            kb,
            r0,
        });
    }

    // --- angles: 1-2-3 walk, i<j<k with k != i, dedup via index ordering --
    let mut angles = Vec::new();
    for i in 0..n_atoms {
        for &(j, _bond1) in &adjacency[i] {
            if j <= i {
                continue;
            }
            for &(k, _bond2) in &adjacency[j] {
                if k <= j || k == i {
                    continue;
                }
                let t_i = type_at(atom_types, i);
                let t_j = type_at(atom_types, j);
                let t_k = type_at(atom_types, k);

                // key = (t1, t2, t3) exact order only; default (0.0, 0.0) on
                // miss. NOT `param_lookup::lookup_angle_params` (no (k,j,i)
                // reverse fallback, no substitution) -- see module docs.
                let key = (t_i.clone(), t_j.clone(), t_k.clone());
                let (kt, t0) = params.angles.get(&key).copied().unwrap_or((0.0, 0.0));

                angles.push(AngleRecord {
                    i,
                    j,
                    k,
                    types: (t_i, t_j, t_k),
                    kt,
                    t0,
                });
            }
        }
    }

    // --- torsions: 1-2-3-4 walk, i<j<k<l with l != j (and k != i, from the
    // angle stage feeding it) -----------------------------------------------
    let mut torsions = Vec::new();
    for i in 0..n_atoms {
        for &(j, _bond1) in &adjacency[i] {
            if j <= i {
                continue;
            }
            for &(k, _bond2) in &adjacency[j] {
                if k <= j || k == i {
                    continue;
                }
                for &(l, _bond3) in &adjacency[k] {
                    if l <= k || l == j {
                        continue;
                    }
                    let t_i = type_at(atom_types, i);
                    let t_j = type_at(atom_types, j);
                    let t_k = type_at(atom_types, k);
                    let t_l = type_at(atom_types, l);

                    let key = (t_i.clone(), t_j.clone(), t_k.clone(), t_l.clone());

                    // Exact match first, then substitution fallback -- only
                    // if the exact match is missing OR present-but-empty
                    // (Python: `torsion_params = ...get(key, []); if not
                    // torsion_params: try substitution`).
                    let mut torsion_params =
                        params.torsions.get(&key).cloned().unwrap_or_default();
                    if torsion_params.is_empty() {
                        let key_sub = (
                            substitute_torsion_type(&t_i),
                            substitute_torsion_type(&t_j),
                            substitute_torsion_type(&t_k),
                            substitute_torsion_type(&t_l),
                        );
                        torsion_params =
                            params.torsions.get(&key_sub).cloned().unwrap_or_default();
                    }

                    torsions.push(TorsionRecord {
                        i,
                        j,
                        k,
                        l,
                        types: key,
                        params: torsion_params,
                    });
                }
            }
        }
    }

    // --- masses: one entry per distinct atom type actually assigned -------
    // Python builds this from `set(atom_types)`, whose iteration order is a
    // CPython hash-seed artifact with no semantic meaning here (the result
    // is consumed as a mapping, not iterated positionally downstream) -- so
    // rather than trying to fabricate an unreproducible "set order", this
    // inserts in first-occurrence order of `atom_types`, which is
    // deterministic and observably equivalent (same keys, same values).
    let mut masses = IndexMap::new();
    for at in atom_types {
        masses
            .entry(at.clone())
            .or_insert_with(|| params.masses.get(at).copied().unwrap_or(0.0));
    }

    (masses, bonds, angles, torsions)
}

/// End-to-end GAFF2 parameterization. Port of `parameterize_gaff_with_rdkit`
/// (gaff2.py:1737-1905).
///
/// `mol` is an already-Kekulized, owned [`MolGraph`] in place of Python's
/// `rdkit.Chem.Mol` (see `mol.rs`). `featurizer`/`assigner` are forwarded
/// as-is to `crate::charges::get_espaloma_charges` — pass `(None, None)` for
/// the common case (matching Python whenever the optional `proxide._proxider`
/// / `expaloma` imports are absent), or real implementations to exercise the
/// native-espaloma path. `gaff_version` is accepted for signature parity but
/// unused — see the module doc comment.
///
/// Returns `Err` if the bundled `.dat` file can't be read (`io::Error`,
/// stringified) or if atom-type assignment fails.
pub fn parameterize_gaff2(
    mol: &MolGraph,
    featurizer: Option<&dyn EspalomaFeaturizer>,
    assigner: Option<&dyn EspalomaAssigner>,
    _gaff_version: &str,
) -> Result<Gaff2Parameterization, String> {
    let params = param_loader::load_gaff2_parameters(None).map_err(|e| e.to_string())?;
    let atom_types = assign_gaff2_atom_types(mol)?;
    let (masses, bonds, angles, torsions) = build_topology(mol, &atom_types, &params);
    let charges = get_espaloma_charges(mol, featurizer, assigner);

    Ok(Gaff2Parameterization {
        atom_types,
        charges,
        masses,
        bonds,
        angles,
        torsions,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mol_graph(elements: Vec<&str>, bonds: Vec<(usize, usize, BondOrder)>) -> MolGraph {
        let elements: Vec<String> = elements.into_iter().map(|s| s.to_string()).collect();
        // One entry per atom, all-zero -- charges.rs's `gasteiger_charges`
        // seeds its iterative PEOE solve from `mol.formal_charges`, so an
        // under-length vec here (this helper used to hardcode `vec![]`
        // regardless of atom count) silently zero-length every downstream
        // per-atom charges vector instead of erroring, which only surfaced
        // once a test here actually exercised the `parameterize_gaff2` ->
        // `get_espaloma_charges` path.
        let formal_charges = vec![0i8; elements.len()];
        MolGraph {
            elements,
            formal_charges,
            bonds: bonds
                .into_iter()
                .map(|(i, j, order)| Bond {
                    i,
                    j,
                    order,
                    aromatic: false,
                })
                .collect(),
            rings: None,
        }
    }

    #[test]
    fn linear_chain_produces_expected_angle_and_torsion_counts() {
        // n-butane heavy-atom skeleton: 0-1-2-3, all single bonds.
        let mol = mol_graph(
            vec!["C", "C", "C", "C"],
            vec![
                (0, 1, BondOrder::Single),
                (1, 2, BondOrder::Single),
                (2, 3, BondOrder::Single),
            ],
        );
        let atom_types = vec![
            "c3".to_string(),
            "c3".to_string(),
            "c3".to_string(),
            "c3".to_string(),
        ];
        let params = Gaff2Parameters::default();
        let (masses, bonds, angles, torsions) = build_topology(&mol, &atom_types, &params);

        assert_eq!(bonds.len(), 3);
        assert_eq!(angles.len(), 2); // (0,1,2) and (1,2,3)
        assert_eq!(torsions.len(), 1); // (0,1,2,3)
        assert_eq!(torsions[0].i, 0);
        assert_eq!(torsions[0].j, 1);
        assert_eq!(torsions[0].k, 2);
        assert_eq!(torsions[0].l, 3);
        // No params loaded -> masses default to 0.0 but every distinct type
        // used still gets an entry.
        assert_eq!(masses.get("c3"), Some(&0.0));
        assert_eq!(masses.len(), 1);
    }

    #[test]
    fn bond_order_classifies_to_sb_db_tb() {
        // NOTE (real residual mismatch in gaff2.py, preserved faithfully):
        // `if bo >= 1.9: "tb" elif bo >= 1.4: "db" else: "sb"` checks the
        // triple-bond threshold FIRST, so a plain Kekulized DOUBLE bond
        // (bo == 2.0, which is >= 1.9) is classified "tb", not "db". The
        // "db" bucket is only ever reachable for `1.4 <= bo < 1.9` -- for a
        // Kekulized bond (Single/Double/Triple -> 1.0/2.0/3.0 exactly,
        // never a fractional aromatic 1.5) that range is empty, so "db" is
        // unreachable in practice from this crate's `MolGraph`. This is
        // exactly what the Python source does; it is not "fixed" here.
        let mol = mol_graph(
            vec!["C", "C", "C", "C", "C", "C"],
            vec![
                (0, 1, BondOrder::Single),
                (1, 2, BondOrder::Double),
                (2, 3, BondOrder::Triple),
            ],
        );
        let atom_types = vec!["c3".to_string(); 6];
        let params = Gaff2Parameters::default();
        let (_masses, bonds, _angles, _torsions) = build_topology(&mol, &atom_types, &params);

        assert_eq!(bonds[0].bond_type, "sb");
        assert_eq!(bonds[0].order, 1.0);
        assert_eq!(bonds[1].bond_type, "tb"); // double bond, but bo=2.0 >= 1.9
        assert_eq!(bonds[1].order, 2.0);
        assert_eq!(bonds[2].bond_type, "tb");
        assert_eq!(bonds[2].order, 3.0);
    }

    #[test]
    fn db_bucket_is_reachable_only_for_a_fractional_bond_order_in_1_4_to_1_9() {
        // Direct test of `classify_bond_type`'s middle branch in isolation,
        // since no reachable `BondOrder` variant exercises it via
        // `build_topology` (see the note above).
        assert_eq!(classify_bond_type(1.0), "sb");
        assert_eq!(classify_bond_type(1.4), "db");
        assert_eq!(classify_bond_type(1.5), "db"); // RDKit's un-Kekulized aromatic value
        assert_eq!(classify_bond_type(1.899), "db");
        assert_eq!(classify_bond_type(1.9), "tb");
        assert_eq!(classify_bond_type(2.0), "tb");
        assert_eq!(classify_bond_type(3.0), "tb");
    }

    #[test]
    fn out_of_range_atom_type_index_falls_back_to_x() {
        let mol = mol_graph(vec!["C", "H"], vec![(0, 1, BondOrder::Single)]);
        // Deliberately shorter than n_atoms, mirroring Python's
        // `atom_types[i] if i < len(atom_types) else "x"` fallback path.
        let atom_types = vec!["c3".to_string()];
        let params = Gaff2Parameters::default();
        let (_masses, bonds, _angles, _torsions) = build_topology(&mol, &atom_types, &params);

        assert_eq!(bonds[0].gaff_type_i, "c3");
        assert_eq!(bonds[0].gaff_type_j, "x");
    }

    #[test]
    fn bond_lookup_uses_sorted_key_with_zero_default_on_miss() {
        let mol = mol_graph(vec!["C", "O"], vec![(0, 1, BondOrder::Single)]);
        let atom_types = vec!["oh".to_string(), "c3".to_string()];
        let mut params = Gaff2Parameters::default();
        // Stored sorted (c3 < oh); lookup must sort (oh, c3) -> (c3, oh) to hit it.
        params
            .bonds
            .insert(("c3".to_string(), "oh".to_string()), (300.0, 1.4));

        let (_masses, bonds, _angles, _torsions) = build_topology(&mol, &atom_types, &params);
        assert_eq!(bonds[0].kb, 300.0);
        assert_eq!(bonds[0].r0, 1.4);

        // Miss -> exact (0.0, 0.0) default, not the (0.0, 1.50) default that
        // `param_lookup::lookup_bond_params` (not called here) uses.
        let mol2 = mol_graph(vec!["C", "N"], vec![(0, 1, BondOrder::Single)]);
        let atom_types2 = vec!["c3".to_string(), "n3".to_string()];
        let (_m, bonds2, _a, _t) = build_topology(&mol2, &atom_types2, &params);
        assert_eq!((bonds2[0].kb, bonds2[0].r0), (0.0, 0.0));
    }

    #[test]
    fn angle_lookup_is_exact_order_only_no_reverse_fallback() {
        // Distinguishes this from `param_lookup::lookup_angle_params`, which
        // DOES try the reversed (k,j,i) key -- parameterize_gaff_with_rdkit's
        // own inline lookup does not, and that must be preserved.
        let mol = mol_graph(
            vec!["C", "C", "O"],
            vec![(0, 1, BondOrder::Single), (1, 2, BondOrder::Single)],
        );
        let atom_types = vec!["c3".to_string(), "c3".to_string(), "oh".to_string()];
        let mut params = Gaff2Parameters::default();
        // Only the reversed order is present.
        params.angles.insert(
            ("oh".to_string(), "c3".to_string(), "c3".to_string()),
            (50.0, 109.0),
        );

        let (_masses, _bonds, angles, _torsions) = build_topology(&mol, &atom_types, &params);
        assert_eq!(angles.len(), 1);
        // Forward key (c3, c3, oh) was never inserted -> default, despite
        // the reverse key being present in the table.
        assert_eq!((angles[0].kt, angles[0].t0), (0.0, 0.0));
    }

    #[test]
    fn torsion_falls_back_to_substituted_key_only_when_exact_key_is_empty() {
        let mol = mol_graph(
            vec!["C", "C", "C", "H"],
            vec![
                (0, 1, BondOrder::Single),
                (1, 2, BondOrder::Single),
                (2, 3, BondOrder::Single),
            ],
        );
        // "cx" substitutes to "c3", and an out-of-range 4th atom would
        // substitute "x" -> "hc"; here atom 3 has an explicit type "x".
        let atom_types = vec![
            "cx".to_string(),
            "cx".to_string(),
            "cx".to_string(),
            "x".to_string(),
        ];
        let mut params = Gaff2Parameters::default();
        params.torsions.insert(
            (
                "c3".to_string(),
                "c3".to_string(),
                "c3".to_string(),
                "hc".to_string(),
            ),
            vec![(3, 0.150, 0.0)],
        );

        let (_masses, _bonds, _angles, torsions) = build_topology(&mol, &atom_types, &params);
        assert_eq!(torsions.len(), 1);
        assert_eq!(
            torsions[0].types,
            (
                "cx".to_string(),
                "cx".to_string(),
                "cx".to_string(),
                "x".to_string()
            )
        );
        assert_eq!(torsions[0].params, vec![(3, 0.150, 0.0)]);
    }

    #[test]
    fn torsion_direct_match_wins_even_when_substituted_key_also_present() {
        // The exact key must win over the substituted fallback when both
        // are present and the exact one is non-empty -- Python only tries
        // substitution when the direct lookup is missing OR empty.
        let mol = mol_graph(
            vec!["C", "C", "C", "C"],
            vec![
                (0, 1, BondOrder::Single),
                (1, 2, BondOrder::Single),
                (2, 3, BondOrder::Single),
            ],
        );
        let atom_types = vec![
            "cx".to_string(),
            "cx".to_string(),
            "cx".to_string(),
            "cx".to_string(),
        ];
        let mut params = Gaff2Parameters::default();
        params.torsions.insert(
            (
                "cx".to_string(),
                "cx".to_string(),
                "cx".to_string(),
                "cx".to_string(),
            ),
            vec![(1, 9.9, 9.9)],
        );
        params.torsions.insert(
            (
                "c3".to_string(),
                "c3".to_string(),
                "c3".to_string(),
                "c3".to_string(),
            ),
            vec![(1, 0.180, 0.0)],
        );

        let (_masses, _bonds, _angles, torsions) = build_topology(&mol, &atom_types, &params);
        assert_eq!(torsions[0].params, vec![(1, 9.9, 9.9)]);
    }

    // --- integration with the real bundled gaff-2.2.20.dat -----------------

    #[test]
    fn real_dat_propane_topology_walk_matches_known_bond_and_angle_values() {
        // Propane heavy-atom skeleton C0-C1-C2, both bonds/angle sp3-sp3.
        let mol = mol_graph(
            vec!["C", "C", "C"],
            vec![(0, 1, BondOrder::Single), (1, 2, BondOrder::Single)],
        );
        let atom_types = vec!["c3".to_string(), "c3".to_string(), "c3".to_string()];
        let params = param_loader::load_default_gaff2_parameters();

        let (masses, bonds, angles, _torsions) = build_topology(&mol, &atom_types, &params);
        assert_eq!(masses.get("c3"), Some(&12.01));
        assert_eq!(bonds.len(), 2);
        assert!(bonds[0].kb > 0.0, "c3-c3 bond should resolve to a real kb");
        assert_eq!(angles.len(), 1);
        assert!(angles[0].kt > 0.0, "c3-c3-c3 angle should resolve to a real kt");
    }

    #[test]
    fn full_parameterize_gaff2_runs_end_to_end_against_the_real_bundled_def_and_dat() {
        // Was `full_parameterize_gaff2_errors_until_atom_typing_is_implemented`
        // while `assign_gaff2_atom_types` was still a `todo!()` stub. Now
        // that `orchestrate.rs` is wired up (real `ATOMTYPE_GFF2.DEF` rules
        // + fallback ladder), this exercises the true end-to-end path: real
        // atom typing -> real `gaff-2.2.20.dat` parameter lookup -> real
        // topology walk. A single unbonded carbon has no attached atoms, so
        // it cannot match any DEF rule that requires `num_attached`
        // (every carbon rule in the bundled file does) and falls through to
        // the fallback ladder's `"c3"` -- matching gaff2.py:1188-1201's
        // `atomic_num == 6 -> "c3"` branch exactly.
        let mol = mol_graph(vec!["C"], vec![]);
        let result = parameterize_gaff2(&mol, None, None, "gaff-2.2.20")
            .expect("real DEF + real .dat should parameterize a bare carbon atom");

        assert_eq!(result.atom_types, vec!["c3".to_string()]);
        assert_eq!(result.charges.len(), 1);
        // A single, bondless atom contributes no bonds/angles/torsions, but
        // its own type's mass should still resolve from the real .dat.
        assert!(result.bonds.is_empty());
        assert!(result.angles.is_empty());
        assert!(result.torsions.is_empty());
        assert_eq!(result.masses.get("c3"), Some(&12.01));
    }
}
