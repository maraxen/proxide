//! Atom and bond feature extraction (GAFF2 f8 "atomic property" facts).
//!
//! Faithful, behavior-preserving port of `AtomBondFacts` and its computation
//! from `src/proxide/chem/gaff2.py` (lines 351-668): ring membership, ring
//! sizes, per-ring AR1..AR5 aromaticity classification, per-atom bond-
//! category tallies (SB/DB/TB/AB/DL/sb/db/tb), and the H-only electron-
//! withdrawing-neighbor count (f7) used by `h`-rule matching.
//!
//! This is a BEHAVIOR-PRESERVING port. Known residual mismatches against
//! real AmberTools (155 atoms / ~45 signatures on the full geostd corpus, as
//! of the Python implementation) are preserved as-is -- triage is a separate,
//! explicitly-gated follow-up. Do not "fix" anything here that looks
//! questionable.
//!
//! # CRITICAL: `classify_ring_aromaticity` is UNCONDITIONAL, not aromaticity-gated
//!
//! [`classify_ring_aromaticity`] is a direct, unconditional port of
//! AmberTools' real per-ring algorithm (`aromatic()`, `ring.c`,
//! Amber-MD/AmberClassic) -- it is **NOT** gated on the caller's own
//! aromaticity perception (RDKit's `GetIsAromatic()` in the Python source, a
//! materially different, stricter Huckel-based test). Confirmed 260821
//! against real geostd ligand `679`: a genuinely non-aromatic-by-Huckel-
//! rules but planar, conjugated 5-ring (a maleimide-like ring with one
//! internal C=C flanked by two exocyclic C=O groups) is still real
//! antechamber's AR3, which the `cc`/`ce`-style DEF rules need to
//! distinguish a "ring" conjugated sp2 atom from a "chain" one. Gating this
//! classification on Huckel aromaticity left such rings with no AR class at
//! all, silently falling through to the wrong rule. [`atom_bond_facts`]
//! enumerates EVERY ring an atom belongs to and aggregates independently
//! per-ring (AR1 if ANY ring gives AR1, else AR23 if any ring gives AR2 or
//! AR3, else AR5, else AR4) -- see its doc comment for the full aggregation
//! rationale, ported verbatim from gaff2.py:542-577.
//!
//! # `HashMap` fields are safe here
//!
//! `ring_counts_by_size` and `bond_counts` are plain `HashMap`s
//! deliberately: both are consulted only via key lookup (`.get(key)`) by
//! every current and Python-source consumer, never iterated in an
//! order-sensitive way (the accumulation loops in [`atom_bond_facts`] and
//! `_atom_bond_facts` alike only ever do `counts[key] += 1` against a fixed,
//! pre-known key set, so the *iteration* order of the thing being summed
//! over is provably irrelevant to the final map contents) -- unlike e.g.
//! `mol.rs::Bond` ordering, which genuinely IS load-bearing. This mirrors
//! the established convention elsewhere in this port (see
//! `alternation.rs`/`param_lookup.rs`'s own HashMap-safety notes): reach for
//! an order-preserving structure only when a *consumer* actually iterates,
//! not defensively for every dict-shaped value.

use std::collections::HashMap;

use crate::mol::{Bond, BondOrder, MolGraph};

/// Per-atom bond/ring facts shared by f8 (`atomic_prop`) and f9 (`chem_env`)
/// matching. Ports `AtomBondFacts` (gaff2.py:351-355).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomBondFacts {
    pub in_ring: bool,
    pub ring_counts_by_size: HashMap<usize, usize>,
    /// `"AR1"`/`"AR23"`/`"AR4"`/`"AR5"`, or `None` if not in any ring. `AR2`
    /// and `AR3` are collapsed into the single `"AR23"` string -- no
    /// current DEF rule distinguishes them (see module doc).
    pub aromaticity_class: Option<String>,
    pub bond_counts: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Per-ring aromaticity classification
// ---------------------------------------------------------------------------

/// Per-ring AR1..AR5 aromaticity classification, as computed by
/// [`classify_ring_aromaticity`] for exactly ONE ring. [`atom_bond_facts`]
/// aggregates this across every ring an atom belongs to and collapses AR2/
/// AR3 into the single `"AR23"` string in [`AtomBondFacts::aromaticity_class`]
/// -- no current DEF rule distinguishes AR2 from AR3 (gaff2.py:560-562).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingAromaticity {
    /// Pure aromatic (benzene/pyridine-like).
    Ar1,
    /// Other planar ring.
    Ar2,
    /// Planar ring formed by "outside" (exocyclic) double bonds.
    Ar3,
    /// Fallback / hypervalent-or-sp3-containing non-pure-aliphatic ring.
    Ar4,
    /// Pure aliphatic ring (every atom sp3 carbon).
    Ar5,
}

/// Matches one f8 aromaticity token (`"AR1"`..`"AR5"`) against an atom's
/// aggregated aromaticity-class string. Direct port of
/// `_matches_aromaticity_token` (gaff2.py:598-607).
///
/// Note: `atomic_prop.rs` (the f8 grammar consumer, ported before this
/// module landed) carries its own private copy of this exact function,
/// since it needed one to compile/test independently while this module was
/// still a stub -- see this crate's port lessons log
/// (`.praxia/docs/reference/260821_gaff2-rust-port-lessons.md`, "Port:
/// atomic_prop"). Both copies are faithful ports of the same 10-line Python
/// function and cannot diverge in practice; a follow-up cleanup could have
/// `atomic_prop.rs` import this one instead of keeping its own, but that
/// edit is out of this module's scope.
pub fn matches_aromaticity_token(word: &str, aromaticity_class: Option<&str>) -> bool {
    let Some(ac) = aromaticity_class else {
        return false;
    };
    if word == "AR4" || word == "AR5" {
        return ac == word;
    }
    if word == "AR1" {
        return ac == "AR1";
    }
    if word == "AR2" || word == "AR3" {
        return ac == "AR23";
    }
    false
}

// ---------------------------------------------------------------------------
// Bond-category facts
// ---------------------------------------------------------------------------

/// Per-bond category flags. Direct port of `_bond_category_facts`
/// (gaff2.py:363-410).
///
/// Keys are the exact DEF token text: `"SB"`/`"DB"`/`"TB"` (exact,
/// non-aromatic-only bond identity), `"AB"`/`"DL"` (aromatic / delocalized
/// -- identical by this module's documented DL-detection simplification,
/// see gaff2.py:379-386), and `"sb"`/`"db"`/`"tb"` (the lowercase,
/// aromatic-inclusive unions the DEF footer describes). Requires `bond` to
/// come from a Kekulized structure where `bond.order` is the true Kekule
/// single/double/triple identity and `bond.aromatic` preserves the
/// original (pre-Kekulize) aromaticity flag -- the same invariant the
/// Python source requires via `Chem.Kekulize(mol, clearAromaticFlags=False)`.
///
/// # Panics
///
/// Mirrors Python's `raise AssertionError` precondition check
/// (gaff2.py:393-400): panics if `bond.aromatic` is true but `bond.order`
/// is neither `Single` nor `Double` (an aromatic `Triple` bond is
/// chemically nonsensical and indicates a `MolGraph` that was not built
/// consistently with the Kekulization invariant above -- sb/db would
/// otherwise silently collapse to always-false for this bond).
pub fn bond_category_facts(bond: &Bond) -> HashMap<&'static str, bool> {
    let is_aromatic = bond.aromatic;
    let is_single = bond.order == BondOrder::Single;
    let is_double = bond.order == BondOrder::Double;
    let is_triple = bond.order == BondOrder::Triple;

    if is_aromatic && !(is_single || is_double) {
        panic!(
            "bond_category_facts precondition violated: an aromatic bond \
             reports order {:?}, not Single/Double -- the owning MolGraph \
             was not built consistently with the Kekulization invariant \
             (clearAromaticFlags=False semantics preserved) documented on \
             this function. sb/db would silently collapse to always-false \
             for this bond.",
            bond.order
        );
    }

    HashMap::from([
        ("SB", is_single && !is_aromatic),
        ("DB", is_double && !is_aromatic),
        ("TB", is_triple),
        ("AB", is_aromatic),
        ("DL", is_aromatic),
        ("sb", is_single),
        ("db", is_double),
        ("tb", is_triple),
    ])
}

// ---------------------------------------------------------------------------
// Adjacency helpers (private -- MolGraph itself has no adjacency index;
// bond order in `mol.bonds` is molecule-file order and is iterated
// faithfully here, mirroring RDKit's per-atom `GetBonds()` order for a
// freshly-Kekulized, non-bond-mutated molecule).
// ---------------------------------------------------------------------------

/// Bonds incident to `atom_idx`, each paired with the *other* atom's index,
/// in `mol.bonds` order.
fn bonds_incident(mol: &MolGraph, atom_idx: usize) -> impl Iterator<Item = (&Bond, usize)> {
    mol.bonds.iter().filter_map(move |b| {
        if b.i == atom_idx {
            Some((b, b.j))
        } else if b.j == atom_idx {
            Some((b, b.i))
        } else {
            None
        }
    })
}

/// Number of rings (out of `mol.rings`, molecule-wide) that `atom_idx`
/// belongs to. Mirrors RDKit's `RingInfo.NumAtomRings(idx)`.
fn num_atom_rings(mol: &MolGraph, atom_idx: usize) -> usize {
    mol.rings
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .filter(|ring| ring.contains(&atom_idx))
        .count()
}

// ---------------------------------------------------------------------------
// Core facts computation
// ---------------------------------------------------------------------------

/// H-inclusive attached-atom count (matches DEF f5 convention). Direct port
/// of `_attached_count` (gaff2.py:358-360): Python computes
/// `atom.GetDegree() + atom.GetNumImplicitHs()`.
///
/// `MolGraph` has no implicit-hydrogen concept: every real atom, hydrogens
/// included, is expected to already be an explicit graph node/bond by the
/// time a `MolGraph` is constructed -- this mirrors
/// `Molecule._to_rdkit()`'s unconditional `atom.SetNoImplicit(True)`
/// (molecule.py:410-414), which forces `GetNumImplicitHs() == 0` for every
/// atom on the real production path (mol2/sdf inputs are expected to list
/// every atom, hydrogens included, explicitly). The implicit-H term is
/// therefore always 0 here and `attached_count` reduces to plain bonded
/// degree -- a faithful translation of an invariant that already holds in
/// the Python pipeline by construction, not a behavior change.
pub fn attached_count(mol: &MolGraph, atom_idx: usize) -> usize {
    bonds_incident(mol, atom_idx).count()
}

/// H-neighbor count for f6 (`num_h`) matching. Direct port of
/// `atom.GetTotalNumHs(includeNeighbors=True)` (gaff2.py:701 --
/// `Gaff2Rule.matches`'s `num_h` computation).
///
/// `includeNeighbors=True` is load-bearing in the Python: after
/// `AllChem.AddHs`, H atoms are explicit graph nodes, so the default
/// (implicit/explicit-count-only) form of `GetTotalNumHs` would return 0
/// for every atom. `MolGraph` has no implicit-hydrogen concept at all
/// (every H is expected to already be an explicit bonded atom -- see
/// [`attached_count`]'s doc for the same invariant), so this reduces to a
/// plain count of `atom_idx`'s bonded neighbors whose element is `"H"`.
pub fn num_h_neighbors(mol: &MolGraph, atom_idx: usize) -> usize {
    bonds_incident(mol, atom_idx)
        .filter(|(_, other)| mol.elements[*other] == "H")
        .count()
}

/// Per-atom contribution to a ring's aromaticity-classification sum --
/// direct port of `initarom[i]` in AmberTools' `aromatic()` (`ring.c`,
/// Amber-MD/AmberClassic), ported here as `_ring_initarom_atom`
/// (gaff2.py:413-451).
///
/// Uses the atom's element symbol (`mol.elements[atom_idx]`) in place of
/// Python's `atom.GetAtomicNum()`, since `MolGraph` stores standard element
/// symbols rather than atomic numbers -- a data-model-forced, behavior-
/// preserving substitution (`"C"`==6, `"N"`==7, `"O"`==8, `"P"`==15,
/// `"S"`==16).
pub fn ring_initarom_atom(mol: &MolGraph, atom_idx: usize) -> i32 {
    let connum = attached_count(mol, atom_idx) as i32;
    match mol.elements[atom_idx].as_str() {
        "C" => match connum {
            3 => 2,
            4 => -2,
            _ => 0,
        },
        "N" => {
            if connum <= 3 {
                2
            } else {
                0
            }
        }
        "O" => {
            if connum == 2 {
                1
            } else {
                0
            }
        }
        "P" => match connum {
            2 => 2,
            3 => 1,
            c if c >= 4 => -1,
            _ => 0,
        },
        "S" => match connum {
            2 => 1,
            c if c >= 3 => -1,
            _ => 0,
        },
        _ => 0,
    }
}

/// Classify one ring as AR1..AR5. Direct, UNCONDITIONAL port of
/// AmberTools' real per-ring algorithm (`aromatic()`, `ring.c`,
/// Amber-MD/AmberClassic), ported here as `_classify_ring_aromaticity`
/// (gaff2.py:454-527) -- **NOT** gated on the caller's own aromaticity
/// perception. See this module's top-level doc for the full rationale.
///
/// Algorithm (condition order matters -- first match wins, mirrored here
/// as early returns):
///   1. `total = sum(ring_initarom_atom(a) for a in ring)`.
///   2. `total == -2 * len(ring)` (every atom sp3 carbon): `Ar5`.
///   3. Any ring atom has a negative contribution (sp3 carbon, or
///      hypervalent P/S): `Ar4`.
///   4. `len(ring) <= total <= 2 * len(ring)` AND some ring atom has a
///      Kekule `Double` bond to a neighbor with NO ring membership at all
///      (a fused-ring bridgehead's neighbor in its OTHER ring still counts
///      as ring-connected and must not trigger this): `Ar3`.
///   5. `total == 12 and len(ring) == 6` AND no ring N/P atom lacks a
///      `Double` bond anywhere among ALL its bonds (checked molecule-wide,
///      not just the two bonds internal to `ring`): `Ar1`.
///   6. `total >= len(ring) + 3`: `Ar2`.
///   7. Otherwise: `Ar4` (fallback).
///
/// `mol` must already be Kekulized (bond orders are the true Kekule
/// single/double/triple identity) -- see [`bond_category_facts`]'s doc for
/// the same invariant.
pub fn classify_ring_aromaticity(mol: &MolGraph, ring: &[usize]) -> RingAromaticity {
    let n = ring.len() as i32;
    let total: i32 = ring.iter().map(|&idx| ring_initarom_atom(mol, idx)).sum();

    if total == -2 * n {
        return RingAromaticity::Ar5;
    }
    if ring.iter().any(|&idx| ring_initarom_atom(mol, idx) < 0) {
        return RingAromaticity::Ar4;
    }

    if n <= total && total <= 2 * n {
        for &ring_idx in ring {
            for (bond, other) in bonds_incident(mol, ring_idx) {
                if num_atom_rings(mol, other) > 0 {
                    continue;
                }
                if bond.order == BondOrder::Double {
                    return RingAromaticity::Ar3;
                }
            }
        }
    }

    if total == 12 && n == 6 {
        let mut has_lone_pair_donor = false;
        for &ring_idx in ring {
            let elem = mol.elements[ring_idx].as_str();
            if elem != "N" && elem != "P" {
                continue;
            }
            let has_double =
                bonds_incident(mol, ring_idx).any(|(b, _)| b.order == BondOrder::Double);
            if !has_double {
                has_lone_pair_donor = true;
                break;
            }
        }
        if !has_lone_pair_donor {
            return RingAromaticity::Ar1;
        }
    }

    if total >= n + 3 {
        return RingAromaticity::Ar2;
    }
    RingAromaticity::Ar4
}

/// Compute the full [`AtomBondFacts`] for one atom. Direct port of
/// `_atom_bond_facts` (gaff2.py:530-595).
///
/// Aromaticity-class aggregation: classify EVERY ring this atom belongs to
/// independently (via [`classify_ring_aromaticity`], itself unconditional
/// -- see module doc), then aggregate: `"AR1"` if ANY ring gives `Ar1`
/// (checked first because this atom-type's DEF rule always precedes the
/// AR2/AR3-requiring one in file order for every case that needs this);
/// else `"AR23"` if ANY ring gives `Ar2` or `Ar3` (collapsed, as before --
/// no current DEF rule distinguishes them); else `"AR5"` if some ring
/// gives that; else `"AR4"` if some ring gives that. `None` if the atom is
/// not in any ring.
///
/// Bond-count tallies include bonds to hydrogen (unlike f9's neighbor-
/// pattern matching, which is heavy-atom-only): a terminal alkyne carbon's
/// only single bond is to H (`cg`'s `[sb,tb]` requires it to be counted),
/// and the DEF file gives no indication f8's bond-count facts should be
/// H-exclusive.
pub fn atom_bond_facts(mol: &MolGraph, atom_idx: usize) -> AtomBondFacts {
    let rings: &[Vec<usize>] = mol.rings.as_deref().unwrap_or(&[]);

    let mut ring_counts_by_size: HashMap<usize, usize> = HashMap::new();
    let mut in_ring = false;
    let mut containing_rings: Vec<&Vec<usize>> = Vec::new();
    for ring in rings {
        if ring.contains(&atom_idx) {
            in_ring = true;
            *ring_counts_by_size.entry(ring.len()).or_insert(0) += 1;
            containing_rings.push(ring);
        }
    }

    let aromaticity_class: Option<String> = if in_ring {
        let ring_classes: Vec<RingAromaticity> = containing_rings
            .iter()
            .map(|ring| classify_ring_aromaticity(mol, ring))
            .collect();
        if ring_classes.contains(&RingAromaticity::Ar1) {
            Some("AR1".to_string())
        } else if ring_classes
            .iter()
            .any(|c| matches!(c, RingAromaticity::Ar2 | RingAromaticity::Ar3))
        {
            Some("AR23".to_string())
        } else if ring_classes.contains(&RingAromaticity::Ar5) {
            Some("AR5".to_string())
        } else if ring_classes.contains(&RingAromaticity::Ar4) {
            Some("AR4".to_string())
        } else {
            None
        }
    } else {
        None
    };

    let mut bond_counts: HashMap<String, usize> = ["SB", "DB", "TB", "AB", "DL", "sb", "db", "tb"]
        .iter()
        .map(|k| (k.to_string(), 0))
        .collect();
    for (bond, _other) in bonds_incident(mol, atom_idx) {
        for (key, present) in bond_category_facts(bond) {
            if present {
                *bond_counts.entry(key.to_string()).or_insert(0) += 1;
            }
        }
    }

    AtomBondFacts {
        in_ring,
        ring_counts_by_size,
        aromaticity_class,
        bond_counts,
    }
}

// ---------------------------------------------------------------------------
// f7: H-only electron-withdrawing-neighbor count
// ---------------------------------------------------------------------------

/// Electron-withdrawing element symbols for f7 (H-only) matching. Direct
/// port of `_EW_ATOMS` (gaff2.py:650).
///
/// Independently confirmed via AmberTools' real source (`aromatic()`,
/// Amber-MD/AmberClassic/src/antechamber/aromatic.c): its `ewd`
/// (electron-withdrawing) flag is set for exactly atomic numbers
/// 7/8/16/9/17/35/53 (N/O/S/F/Cl/Br/I) and nothing else -- S is explicitly
/// commented "S is considered electron withdraw group" there. A prior
/// version of this set omitted sulfur (a plausible-looking guess never
/// checked against a live reference), which caused a real, reproducible
/// bug: thiophene's ring H's beta to the ring sulfur mistyped as `ha`
/// instead of `h4` (confirmed against real antechamber output; see this
/// module's tests for a regression covering it). Membership-only usage, so
/// order carries no meaning and a plain array is fine (unlike
/// `bond_counts`, where every key is always present and only ever summed
/// by key -- see module doc's `HashMap`-safety note; this array isn't even
/// a map, just a membership list).
pub const EW_ATOMS: [&str; 7] = ["N", "O", "S", "F", "Cl", "Br", "I"];

/// Count electron-withdrawing atoms bonded to `h_idx`'s own heavy
/// attachment atom. Direct port of `_h_ew_neighbor_count`
/// (gaff2.py:653-668).
///
/// f7 is meaningful only for H atoms (per the DEF footer); an H atom has
/// exactly one real neighbor (its attachment atom), whose OTHER neighbors
/// (not the H itself) are what f7 counts.
pub fn h_ew_neighbor_count(mol: &MolGraph, h_idx: usize) -> usize {
    let neighbors: Vec<usize> = bonds_incident(mol, h_idx).map(|(_, other)| other).collect();
    if neighbors.len() != 1 {
        return 0;
    }
    let attachment = neighbors[0];
    bonds_incident(mol, attachment)
        .filter(|(_, nb)| *nb != h_idx && EW_ATOMS.contains(&mol.elements[*nb].as_str()))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mol::{Bond, BondOrder};

    fn mg(elements: &[&str], bonds: Vec<Bond>, rings: Option<Vec<Vec<usize>>>) -> MolGraph {
        MolGraph {
            elements: elements.iter().map(|s| s.to_string()).collect(),
            formal_charges: vec![0; elements.len()],
            bonds,
            rings,
        }
    }

    fn bond(i: usize, j: usize, order: BondOrder, aromatic: bool) -> Bond {
        Bond {
            i,
            j,
            order,
            aromatic,
        }
    }

    // -- ring_initarom_atom ---------------------------------------------

    #[test]
    fn ring_initarom_atom_carbon_connum3_is_2() {
        let mol = mg(
            &["C", "C", "C", "C"],
            vec![
                bond(0, 1, BondOrder::Single, false),
                bond(0, 2, BondOrder::Single, false),
                bond(0, 3, BondOrder::Single, false),
            ],
            None,
        );
        assert_eq!(ring_initarom_atom(&mol, 0), 2);
    }

    #[test]
    fn ring_initarom_atom_carbon_connum4_is_negative_2() {
        let mol = mg(
            &["C", "C", "C", "C", "C"],
            vec![
                bond(0, 1, BondOrder::Single, false),
                bond(0, 2, BondOrder::Single, false),
                bond(0, 3, BondOrder::Single, false),
                bond(0, 4, BondOrder::Single, false),
            ],
            None,
        );
        assert_eq!(ring_initarom_atom(&mol, 0), -2);
    }

    #[test]
    fn ring_initarom_atom_carbon_other_connum_is_0() {
        let mol = mg(
            &["C", "C"],
            vec![bond(0, 1, BondOrder::Single, false)],
            None,
        );
        assert_eq!(ring_initarom_atom(&mol, 0), 0); // connum 1
    }

    #[test]
    fn ring_initarom_atom_nitrogen_connum_le3_is_2_else_0() {
        let n3 = mg(
            &["N", "C", "C", "C"],
            vec![
                bond(0, 1, BondOrder::Single, false),
                bond(0, 2, BondOrder::Single, false),
                bond(0, 3, BondOrder::Single, false),
            ],
            None,
        );
        assert_eq!(ring_initarom_atom(&n3, 0), 2);

        let n4 = mg(
            &["N", "C", "C", "C", "C"],
            vec![
                bond(0, 1, BondOrder::Single, false),
                bond(0, 2, BondOrder::Single, false),
                bond(0, 3, BondOrder::Single, false),
                bond(0, 4, BondOrder::Single, false),
            ],
            None,
        );
        assert_eq!(ring_initarom_atom(&n4, 0), 0);
    }

    #[test]
    fn ring_initarom_atom_oxygen_connum2_is_1_else_0() {
        let o2 = mg(
            &["O", "C", "C"],
            vec![
                bond(0, 1, BondOrder::Single, false),
                bond(0, 2, BondOrder::Single, false),
            ],
            None,
        );
        assert_eq!(ring_initarom_atom(&o2, 0), 1);

        let o1 = mg(
            &["O", "C"],
            vec![bond(0, 1, BondOrder::Single, false)],
            None,
        );
        assert_eq!(ring_initarom_atom(&o1, 0), 0);
    }

    #[test]
    fn ring_initarom_atom_phosphorus_and_sulfur_branches() {
        // P: connum 2 -> 2, connum 4 -> -1
        let p2 = mg(
            &["P", "C", "C"],
            vec![
                bond(0, 1, BondOrder::Single, false),
                bond(0, 2, BondOrder::Single, false),
            ],
            None,
        );
        assert_eq!(ring_initarom_atom(&p2, 0), 2);

        let p4 = mg(
            &["P", "C", "C", "C", "C"],
            vec![
                bond(0, 1, BondOrder::Single, false),
                bond(0, 2, BondOrder::Single, false),
                bond(0, 3, BondOrder::Single, false),
                bond(0, 4, BondOrder::Single, false),
            ],
            None,
        );
        assert_eq!(ring_initarom_atom(&p4, 0), -1);

        // S: connum 2 -> 1, connum 3 -> -1
        let s2 = mg(
            &["S", "C", "C"],
            vec![
                bond(0, 1, BondOrder::Single, false),
                bond(0, 2, BondOrder::Single, false),
            ],
            None,
        );
        assert_eq!(ring_initarom_atom(&s2, 0), 1);

        let s3 = mg(
            &["S", "C", "C", "C"],
            vec![
                bond(0, 1, BondOrder::Single, false),
                bond(0, 2, BondOrder::Single, false),
                bond(0, 3, BondOrder::Single, false),
            ],
            None,
        );
        assert_eq!(ring_initarom_atom(&s3, 0), -1);
    }

    // -- bond_category_facts ---------------------------------------------

    #[test]
    fn bond_category_facts_plain_single() {
        let b = bond(0, 1, BondOrder::Single, false);
        let f = bond_category_facts(&b);
        assert_eq!(f.len(), 8);
        assert_eq!(f["SB"], true);
        assert_eq!(f["DB"], false);
        assert_eq!(f["TB"], false);
        assert_eq!(f["AB"], false);
        assert_eq!(f["DL"], false);
        assert_eq!(f["sb"], true);
        assert_eq!(f["db"], false);
        assert_eq!(f["tb"], false);
    }

    #[test]
    fn bond_category_facts_aromatic_double() {
        let b = bond(0, 1, BondOrder::Double, true);
        let f = bond_category_facts(&b);
        // Aromatic excludes exact SB/DB but is included in sb/db/AB/DL.
        assert!(!f["SB"] && !f["DB"]);
        assert!(f["AB"] && f["DL"] && f["db"] && !f["sb"]);
    }

    #[test]
    fn bond_category_facts_triple() {
        let b = bond(0, 1, BondOrder::Triple, false);
        let f = bond_category_facts(&b);
        assert!(f["TB"] && f["tb"]);
        assert!(!f["SB"] && !f["DB"] && !f["AB"] && !f["DL"] && !f["sb"] && !f["db"]);
    }

    #[test]
    #[should_panic(expected = "precondition violated")]
    fn bond_category_facts_aromatic_triple_panics() {
        // A malformed MolGraph: aromatic flag set on a Kekule triple bond
        // is nonsensical -- mirrors Python's AssertionError precondition.
        let b = bond(0, 1, BondOrder::Triple, true);
        let _ = bond_category_facts(&b);
    }

    // -- matches_aromaticity_token ----------------------------------------

    #[test]
    fn matches_aromaticity_token_all_combinations() {
        assert!(!matches_aromaticity_token("AR1", None));
        assert!(matches_aromaticity_token("AR1", Some("AR1")));
        assert!(!matches_aromaticity_token("AR1", Some("AR23")));
        assert!(matches_aromaticity_token("AR2", Some("AR23")));
        assert!(matches_aromaticity_token("AR3", Some("AR23")));
        assert!(!matches_aromaticity_token("AR2", Some("AR1")));
        assert!(matches_aromaticity_token("AR4", Some("AR4")));
        assert!(matches_aromaticity_token("AR5", Some("AR5")));
        assert!(!matches_aromaticity_token("AR4", Some("AR5")));
        assert!(!matches_aromaticity_token("bogus", Some("AR1")));
    }

    // -- classify_ring_aromaticity / atom_bond_facts: real-molecule fixtures

    /// Benzene: 6 aromatic carbons, each with one H.
    fn benzene() -> MolGraph {
        let elements = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"];
        let mut bonds = vec![
            bond(0, 1, BondOrder::Double, true),
            bond(1, 2, BondOrder::Single, true),
            bond(2, 3, BondOrder::Double, true),
            bond(3, 4, BondOrder::Single, true),
            bond(4, 5, BondOrder::Double, true),
            bond(5, 0, BondOrder::Single, true),
        ];
        for (ring_idx, h_idx) in (0..6).zip(6..12) {
            bonds.push(bond(ring_idx, h_idx, BondOrder::Single, false));
        }
        mg(&elements, bonds, Some(vec![vec![0, 1, 2, 3, 4, 5]]))
    }

    #[test]
    fn benzene_ring_classifies_ar1() {
        let mol = benzene();
        assert_eq!(
            classify_ring_aromaticity(&mol, &[0, 1, 2, 3, 4, 5]),
            RingAromaticity::Ar1
        );
    }

    #[test]
    fn benzene_atom_bond_facts_ar1_and_ring_counts() {
        let mol = benzene();
        let facts = atom_bond_facts(&mol, 0);
        assert!(facts.in_ring);
        assert_eq!(facts.ring_counts_by_size.get(&6), Some(&1));
        assert_eq!(facts.aromaticity_class.as_deref(), Some("AR1"));
        // Ring atom 0: aromatic double bond to 1, aromatic single to 5,
        // plain non-aromatic single to its H.
        assert_eq!(facts.bond_counts["AB"], 2);
        assert_eq!(facts.bond_counts["DL"], 2);
        assert_eq!(facts.bond_counts["db"], 1);
        assert_eq!(facts.bond_counts["sb"], 2); // aromatic single (1) + C-H (1)
        assert_eq!(facts.bond_counts["SB"], 1); // only the non-aromatic C-H bond
    }

    /// Maleimide-like 5-ring (real geostd ligand 679's core, see
    /// test_ar23_classification_is_not_gated_on_rdkit_aromaticity in
    /// test_gaff2_golden.py:638-696): one internal C=C flanked by two
    /// exocyclic C=O groups, bridged by a ring N-H. RDKit does NOT call
    /// this ring aromatic, yet real antechamber classifies it AR3 -- this
    /// is the load-bearing regression this module exists to preserve.
    fn maleimide_like_ring() -> MolGraph {
        // p=0(C), q=1(C), r=2(C), s=3(C), t=4(N), o_r=5(O), o_s=6(O),
        // H on p=7, H on q=8, H on t=9.
        let elements = ["C", "C", "C", "C", "N", "O", "O", "H", "H", "H"];
        let bonds = vec![
            bond(0, 1, BondOrder::Double, false), // p=q, the "ene" bond
            bond(0, 3, BondOrder::Single, false), // p-s, closes the ring
            bond(1, 2, BondOrder::Single, false), // q-r
            bond(2, 4, BondOrder::Single, false), // r-t
            bond(4, 3, BondOrder::Single, false), // t-s
            bond(2, 5, BondOrder::Double, false), // r=o_r, exocyclic
            bond(3, 6, BondOrder::Double, false), // s=o_s, exocyclic
            bond(0, 7, BondOrder::Single, false), // p-H
            bond(1, 8, BondOrder::Single, false), // q-H
            bond(4, 9, BondOrder::Single, false), // t-H
        ];
        mg(&elements, bonds, Some(vec![vec![0, 1, 2, 3, 4]]))
    }

    #[test]
    fn maleimide_like_ring_classifies_ar3_though_not_rdkit_aromatic() {
        let mol = maleimide_like_ring();
        assert_eq!(
            classify_ring_aromaticity(&mol, &[0, 1, 2, 3, 4]),
            RingAromaticity::Ar3
        );
        // atom_bond_facts must surface this as the collapsed AR23 class so
        // cc/cd-style [sb,db,AR2/AR3] rules can fire.
        let facts = atom_bond_facts(&mol, 0);
        assert_eq!(facts.aromaticity_class.as_deref(), Some("AR23"));
    }

    /// Fused 6+5-ring system whose bridgehead nitrogen (`f`) has zero
    /// double bonds anywhere (a pyrrole-type lone-pair donor). Direct port
    /// of the fixture in
    /// test_fused_bicyclic_lone_pair_donor_not_promoted_to_ar1
    /// (test_gaff2_golden.py:491-556): confirms this bridgehead must NOT
    /// be promoted to AR1 just because it sits in rings that could
    /// otherwise look aromatic-ish, and confirms the has-a-double-bond
    /// check is molecule-wide (all of `f`'s bonds), not ring-internal only.
    fn fused_bicyclic_lone_pair_donor() -> MolGraph {
        // a=0(C) b=1(C) c=2(N) d=3(C) e=4(C) g=5(N) h=6(C) i=7(C) f=8(N,
        // bridgehead). H's: a=9, b=10, d=11, h=12, i=13 (c, e, g, f are
        // valence-saturated by their heavy bonds alone and need no H).
        let elements = [
            "C", "C", "N", "C", "C", "N", "C", "C", "N", "H", "H", "H", "H", "H",
        ];
        let bonds = vec![
            bond(0, 1, BondOrder::Double, false), // a=b
            bond(1, 2, BondOrder::Single, false), // b-c
            bond(2, 3, BondOrder::Double, false), // c=d
            bond(3, 4, BondOrder::Single, false), // d-e
            bond(4, 8, BondOrder::Single, false), // e-f, shared bridgehead edge
            bond(8, 0, BondOrder::Single, false), // f-a
            bond(4, 5, BondOrder::Double, false), // e=g
            bond(5, 6, BondOrder::Single, false), // g-h
            bond(6, 7, BondOrder::Double, false), // h=i
            bond(7, 8, BondOrder::Single, false), // i-f
            bond(0, 9, BondOrder::Single, false),
            bond(1, 10, BondOrder::Single, false),
            bond(3, 11, BondOrder::Single, false),
            bond(6, 12, BondOrder::Single, false),
            bond(7, 13, BondOrder::Single, false),
        ];
        let ring1 = vec![0, 1, 2, 3, 4, 8]; // a b c d e f (6-ring)
        let ring2 = vec![4, 5, 6, 7, 8]; // e g h i f (5-ring)
        mg(&elements, bonds, Some(vec![ring1, ring2]))
    }

    #[test]
    fn fused_bicyclic_bridgehead_lone_pair_donor_not_ar1() {
        let mol = fused_bicyclic_lone_pair_donor();
        // Both rings must classify AR2 (not AR1) precisely because `f`
        // (idx 8) has no double bond anywhere among its 3 single bonds.
        assert_eq!(
            classify_ring_aromaticity(&mol, &[0, 1, 2, 3, 4, 8]),
            RingAromaticity::Ar2
        );
        assert_eq!(
            classify_ring_aromaticity(&mol, &[4, 5, 6, 7, 8]),
            RingAromaticity::Ar2
        );
        let f_facts = atom_bond_facts(&mol, 8);
        assert!(f_facts.in_ring);
        assert_eq!(f_facts.aromaticity_class.as_deref(), Some("AR23"));
        assert_ne!(f_facts.aromaticity_class.as_deref(), Some("AR1"));
    }

    /// Naphthalene: both bridgeheads sit in two AR1 6-rings each --
    /// confirms `atom_bond_facts`'s aggregation surfaces AR1 for a
    /// bridgehead via EITHER of its rings, and that ring classification is
    /// based on real double-bond placement across the whole molecule, not
    /// just the ring's own two bonds at each atom. Ground truth: real
    /// antechamber assigns all 10 naphthalene ring carbons -- bridgeheads
    /// included -- `ca` (AR1); see
    /// test_naphthalene_bridgehead_confirmed_by_external_reference in
    /// test_gaff2_golden.py:459-488.
    fn naphthalene() -> MolGraph {
        // Ring1 atoms 0,1,2,3,4(4a),5(8a); ring2 atoms 6,7,8,9,5(8a),4(4a).
        // H's on 0,1,2,3,6,7,8,9 -> indices 10..18. Bridgeheads 4,5 have no H.
        let elements = [
            "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H", "H",
            "H",
        ];
        let bonds = vec![
            bond(0, 1, BondOrder::Double, true),
            bond(1, 2, BondOrder::Single, true),
            bond(2, 3, BondOrder::Double, true),
            bond(3, 4, BondOrder::Single, true),
            bond(4, 5, BondOrder::Double, true), // shared bridge bond
            bond(5, 0, BondOrder::Single, true),
            bond(4, 6, BondOrder::Single, true),
            bond(6, 7, BondOrder::Double, true),
            bond(7, 8, BondOrder::Single, true),
            bond(8, 9, BondOrder::Double, true),
            bond(9, 5, BondOrder::Single, true),
            bond(0, 10, BondOrder::Single, false),
            bond(1, 11, BondOrder::Single, false),
            bond(2, 12, BondOrder::Single, false),
            bond(3, 13, BondOrder::Single, false),
            bond(6, 14, BondOrder::Single, false),
            bond(7, 15, BondOrder::Single, false),
            bond(8, 16, BondOrder::Single, false),
            bond(9, 17, BondOrder::Single, false),
        ];
        let ring1 = vec![0, 1, 2, 3, 4, 5];
        let ring2 = vec![6, 7, 8, 9, 5, 4];
        mg(&elements, bonds, Some(vec![ring1, ring2]))
    }

    #[test]
    fn naphthalene_both_rings_classify_ar1() {
        let mol = naphthalene();
        assert_eq!(
            classify_ring_aromaticity(&mol, &[0, 1, 2, 3, 4, 5]),
            RingAromaticity::Ar1
        );
        assert_eq!(
            classify_ring_aromaticity(&mol, &[6, 7, 8, 9, 5, 4]),
            RingAromaticity::Ar1
        );
    }

    #[test]
    fn naphthalene_bridgehead_is_ar1_and_in_two_six_rings() {
        let mol = naphthalene();
        // Bridgehead atom 4 (4a) is a member of BOTH 6-rings.
        let facts = atom_bond_facts(&mol, 4);
        assert!(facts.in_ring);
        assert_eq!(facts.ring_counts_by_size.get(&6), Some(&2));
        assert_eq!(facts.aromaticity_class.as_deref(), Some("AR1"));

        let facts5 = atom_bond_facts(&mol, 5);
        assert_eq!(facts5.ring_counts_by_size.get(&6), Some(&2));
        assert_eq!(facts5.aromaticity_class.as_deref(), Some("AR1"));
    }

    // -- h_ew_neighbor_count / EW_ATOMS -----------------------------------

    #[test]
    fn ew_atoms_contains_sulfur_and_halogens() {
        // Regression coverage for the 260820 fix: S was previously
        // omitted, mistyping thiophene's beta ring H's as `ha` instead of
        // `h4`.
        for elem in ["N", "O", "S", "F", "Cl", "Br", "I"] {
            assert!(EW_ATOMS.contains(&elem), "{elem} must be in EW_ATOMS");
        }
        assert!(!EW_ATOMS.contains(&"C"));
        assert!(!EW_ATOMS.contains(&"H"));
    }

    /// Thiophene: 5-ring C1-C2-C3-S-C5(-C1). Ring carbons directly bonded
    /// to S (C3, C5) must report h_ew_neighbor_count == 1 for their H's
    /// (S counts as the electron-withdrawing neighbor of the attachment
    /// carbon itself); carbons NOT bonded to S (C1, C2) must report 0.
    /// Direct-connectivity analog of
    /// TestHAtomTyping::test_thiophene_beta_h_is_h4_not_ha
    /// (test_gaff2_golden.py:1004-1022), which exercises the same fix
    /// through the full rule-matching pipeline (out of this module's
    /// scope) rather than `h_ew_neighbor_count` directly.
    fn thiophene() -> MolGraph {
        // c1=0, c2=1, c3=2, s=3, c5=4; H1=5 (on c1), H2=6 (on c2),
        // H3=7 (on c3), H5=8 (on c5). S has no H.
        let elements = ["C", "C", "C", "S", "C", "H", "H", "H", "H"];
        let bonds = vec![
            bond(0, 1, BondOrder::Double, true),
            bond(1, 2, BondOrder::Single, true),
            bond(2, 3, BondOrder::Single, true),
            bond(3, 4, BondOrder::Single, true),
            bond(4, 0, BondOrder::Double, true),
            bond(0, 5, BondOrder::Single, false),
            bond(1, 6, BondOrder::Single, false),
            bond(2, 7, BondOrder::Single, false),
            bond(4, 8, BondOrder::Single, false),
        ];
        mg(&elements, bonds, Some(vec![vec![0, 1, 2, 3, 4]]))
    }

    #[test]
    fn thiophene_h_ew_neighbor_count_via_sulfur() {
        let mol = thiophene();
        // H3 (idx 7) is on C3 (idx 2), which is directly bonded to S (idx 3).
        assert_eq!(h_ew_neighbor_count(&mol, 7), 1);
        // H5 (idx 8) is on C5 (idx 4), also directly bonded to S.
        assert_eq!(h_ew_neighbor_count(&mol, 8), 1);
        // H1 (idx 5) is on C1 (idx 0), whose neighbors are C2 and C5 --
        // neither is electron-withdrawing.
        assert_eq!(h_ew_neighbor_count(&mol, 5), 0);
        // H2 (idx 6) is on C2 (idx 1), whose neighbors are C1 and C3 --
        // also neither electron-withdrawing.
        assert_eq!(h_ew_neighbor_count(&mol, 6), 0);
    }

    #[test]
    fn h_ew_neighbor_count_is_zero_for_non_h_or_multiply_bonded_atom() {
        let mol = thiophene();
        // Called on a heavy atom (more than one neighbor): returns 0 per
        // the `len(neighbors) != 1` guard, mirroring Python exactly.
        assert_eq!(h_ew_neighbor_count(&mol, 2), 0);
    }

    // -- attached_count -----------------------------------------------------

    #[test]
    fn attached_count_matches_bonded_degree() {
        let mol = mg(
            &["C", "H", "H", "H", "H"],
            vec![
                bond(0, 1, BondOrder::Single, false),
                bond(0, 2, BondOrder::Single, false),
                bond(0, 3, BondOrder::Single, false),
                bond(0, 4, BondOrder::Single, false),
            ],
            None,
        );
        assert_eq!(attached_count(&mol, 0), 4);
        assert_eq!(attached_count(&mol, 1), 1);
    }

    // -- bond_counts key set ------------------------------------------------

    #[test]
    fn atom_bond_facts_bond_counts_always_has_all_eight_keys() {
        // Even an atom with zero bonds must report all 8 keys at 0 (matches
        // Python's dict literal initialization, gaff2.py:583).
        let mol = mg(&["C"], vec![], None);
        let facts = atom_bond_facts(&mol, 0);
        for key in ["SB", "DB", "TB", "AB", "DL", "sb", "db", "tb"] {
            assert_eq!(facts.bond_counts.get(key), Some(&0), "missing key {key}");
        }
    }
}
