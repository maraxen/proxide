//! Conjugated/biphenyl atom-type alternation — `cc`↔`cd`, `ce`↔`cf`, ... and
//! `cp`↔`cq` relabeling via 2-coloring the Kekule bond-parity graph.
//!
//! Line-for-line port of Python's `_relabel_conjugated_alternation`
//! (`src/proxide/chem/gaff2.py:1009-1112`), which is itself a line-for-line
//! port of real AmberTools' `atadjust()` (`src/antechamber/atomtype.c`,
//! `Amber-MD/AmberClassic`) — **not** a naive per-component BFS 2-coloring,
//! which is a materially different (and, for disconnected multi-component
//! molecules, WRONG) algorithm.
//!
//! # The reseed mechanism (the load-bearing subtlety of this port)
//!
//! Real `atadjust()` uses a GLOBAL, PASS-SCOPED reseed flag, not a
//! per-component one:
//!
//! - One atom overall (the first family atom, by ascending atom index) is
//!   unconditionally seeded with sign `+1` before any bond is examined.
//! - The algorithm then runs up to `num_family_atoms - 1` outer passes. Each
//!   pass does a single Gauss-Seidel-style sweep over every family-family
//!   bond IN MOLECULE BOND ORDER (i.e. input-file bond order, **not** sorted
//!   by atom index — `MolGraph::bonds` order is load-bearing here),
//!   propagating sign to any atom whose bonded neighbor already has one,
//!   using already-updated values from earlier in the SAME pass.
//! - At most ONE new (still fully-unsigned) connected component may be
//!   "reseeded" with sign `+1` per pass — gated by a single `flag` that is
//!   reset only once at the top of each pass, not per bond or per
//!   component. The reseed fires on the first still-untouched family-family
//!   bond encountered in bond order during a pass, and immediately
//!   propagates across that one bond in the same iteration (which sets the
//!   flag), blocking any further reseeding for the rest of that pass.
//! - There is no cross-component consistency check: if a molecule's family
//!   subgraph is NOT actually 2-colorable as one connected whole (e.g. a
//!   genuinely ambiguous fused-ring system), real antechamber does not
//!   detect or repair the resulting local contradiction — it just prints a
//!   "may be wrong" warning and keeps the result. This port reproduces that
//!   same silently-approximate behavior; it does not attempt to be "more
//!   correct" than the reference it ports.
//!
//! Net effect: for a molecule whose family atoms form more than one
//! connected component, WHICH atom seeds each later component (and hence
//! that component's overall same/primed orientation) depends on bond file
//! order, not just molecular topology — a naive independent-BFS-per-component
//! implementation disagrees with real antechamber on such molecules.
//!
//! `mol` must already be Kekulized by the caller — an un-Kekulized aromatic
//! bond is "not `Single`" (a flip), silently corrupting the coloring for
//! ring systems exactly like the `sb`/`db` precondition this mirrors.
//!
//! `atadjust()` vs `cpadjust()` are NOT structurally identical, despite
//! sharing the same coloring rule: `atadjust()`'s propagation loop has the
//! `flag`-gated reseed described above, so it colors EVERY disconnected
//! same-family subgraph in the molecule. `cpadjust()` has no such reseed: it
//! seeds exactly ONE atom overall (the first `cp` atom found) and only ever
//! colors that one component — any other disconnected `cp` system in the
//! same molecule is left untouched (stays `cp`, never becomes `cq`).
//! `single_seed_only = true` on the cp/cq call reproduces this real,
//! intentional asymmetry rather than treating both calls the same way.

use std::collections::HashMap;

use once_cell::sync::Lazy;

use crate::mol::{BondOrder, MolGraph};

/// Conjugated-alternation family pairs: same-family sp2 ring/chain atom
/// types whose Kekule bond-parity coloring flips paired members to their
/// "primed" type. Mirrors Python's `_CONJUGATED_ALTERNATION_PAIRS`
/// (`gaff2.py:1002-1005`). Fed to [`relabel_conjugated_alternation`] with
/// `single_seed_only = false` (real `atadjust()` semantics).
pub static CONJUGATED_ALTERNATION_PAIRS: Lazy<HashMap<&'static str, &'static str>> =
    Lazy::new(|| {
        HashMap::from([
            ("cc", "cd"),
            ("ce", "cf"),
            ("cg", "ch"),
            ("pc", "pd"),
            ("pe", "pf"),
            ("nc", "nd"),
            ("ne", "nf"),
        ])
    });

/// Biphenyl-connector alternation pair. Mirrors Python's
/// `_BIPHENYL_ALTERNATION_PAIR` (`gaff2.py:1006`). Fed to
/// [`relabel_conjugated_alternation`] with `single_seed_only = true` (real
/// `cpadjust()` semantics — no reseed, only the first `cp` component is
/// ever colored).
pub static BIPHENYL_ALTERNATION_PAIR: Lazy<HashMap<&'static str, &'static str>> =
    Lazy::new(|| HashMap::from([("cp", "cq")]));

/// Mutate `atom_types` in place: 2-color the family subgraph by Kekule bond
/// parity (single bond = same label, double/triple = flipped label),
/// relabeling flipped atoms to their paired ("primed") type.
///
/// Port of `_relabel_conjugated_alternation` (`gaff2.py:1009-1112`). See the
/// module doc for the reseed mechanism this must reproduce exactly.
///
/// `single_seed_only`: disables the reseed check on every pass (as if the
/// pass-scoped `flag` were permanently set), so only the one atom seeded
/// before the pass loop is ever colored — reproducing real AmberTools'
/// `cpadjust()`, which has no reseed step at all.
///
/// # Panics
/// Does not panic on its own; `atom_types.len()` is expected by the caller
/// to equal `mol.elements.len()` (mirroring `mol.GetNumAtoms()` in the
/// Python original). A shorter `atom_types` will panic on out-of-bounds
/// indexing when a bond references an atom beyond its length — the same
/// implicit precondition the Python has via `mol.GetNumAtoms()`.
pub fn relabel_conjugated_alternation(
    mol: &MolGraph,
    atom_types: &mut [String],
    pairs: &HashMap<&str, &str>,
    single_seed_only: bool,
) {
    let n = mol.elements.len();

    // in_family is computed once up front and never mutated — atom_types
    // itself is only mutated at the very end, after sign propagation is
    // fully resolved, so this snapshot stays valid for the whole function.
    let in_family: Vec<bool> = (0..n)
        .map(|i| pairs.contains_key(atom_types[i].as_str()))
        .collect();

    let mut sign: Vec<i8> = vec![0; n];
    let mut num: usize = 0;
    let mut seeded = false;
    for i in 0..n {
        if in_family[i] {
            if !seeded {
                sign[i] = 1;
                seeded = true;
            }
            num += 1;
        }
    }
    if num == 0 {
        return;
    }
    num -= 1;

    // Bonds are swept IN MOLECULE BOND ORDER (mol.bonds, load-bearing —
    // never sort or use a HashMap/HashSet here).
    while num > 0 {
        num -= 1;
        let mut flag = single_seed_only;
        for bond in &mol.bonds {
            let bi = bond.i;
            let bj = bond.j;
            if !(in_family[bi] && in_family[bj]) {
                continue;
            }
            if !flag && sign[bi] == 0 && sign[bj] == 0 {
                sign[bi] = 1;
            }
            let same = bond.order == BondOrder::Single;
            if sign[bi] == 0 && sign[bj] != 0 {
                flag = true;
                sign[bi] = if same { sign[bj] } else { -sign[bj] };
            }
            if sign[bj] == 0 && sign[bi] != 0 {
                flag = true;
                sign[bj] = if same { sign[bi] } else { -sign[bi] };
            }
        }
    }

    for (idx, &s) in sign.iter().enumerate() {
        if s == -1 {
            if let Some(&paired) = pairs.get(atom_types[idx].as_str()) {
                atom_types[idx] = paired.to_string();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mol::Bond;

    /// Build a minimal `MolGraph` for these unit tests — element identity
    /// and formal charges are irrelevant to this algorithm, only atom count,
    /// bond endpoints, and bond order (hence bond ORDER in the `Vec`) matter.
    fn mol_graph(n_atoms: usize, bonds: Vec<(usize, usize, BondOrder)>) -> MolGraph {
        MolGraph {
            elements: vec!["C".to_string(); n_atoms],
            formal_charges: vec![0; n_atoms],
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

    /// Port of Python's `test_relabel_conjugated_alternation_single_seed_only`
    /// (`tests/test_gaff2_golden.py:838-875`). Two disconnected ethene-like
    /// fragments (`C=C . C=C`, 4 heavy atoms, no explicit H needed since the
    /// algorithm only cares about family-typed atoms and bond order): with
    /// `single_seed_only = false` (atadjust()-style) both components get
    /// colored via the pass-scoped reseed; with `single_seed_only = true`
    /// (cpadjust()-style) only the first (lowest-index) component is colored
    /// and the second is left completely untouched.
    #[test]
    fn test_relabel_conjugated_alternation_single_seed_only() {
        let mol = mol_graph(
            4,
            vec![(0, 1, BondOrder::Double), (2, 3, BondOrder::Double)],
        );
        let pairs: HashMap<&str, &str> = HashMap::from([("X", "Y")]);

        // single_seed_only=false (atadjust()-style): both components colored.
        let mut types_multi: Vec<String> = vec!["X".to_string(); 4];
        relabel_conjugated_alternation(&mol, &mut types_multi, &pairs, false);
        assert_eq!(types_multi, vec!["X", "Y", "X", "Y"]);

        // single_seed_only=true (cpadjust()-style): only the first component
        // (lowest atom index) is colored; the second fragment is untouched.
        let mut types_single: Vec<String> = vec!["X".to_string(); 4];
        relabel_conjugated_alternation(&mol, &mut types_single, &pairs, true);
        assert_eq!(types_single, vec!["X", "Y", "X", "X"]);
    }

    /// Port of the topology + bond order exercised by Python's
    /// `test_atadjust_reseed_is_pass_scoped_not_per_component`
    /// (`tests/test_gaff2_golden.py:559-635`), but calling
    /// `relabel_conjugated_alternation` directly with hand-seeded family
    /// types instead of going through the whole rule-matching engine (which
    /// is not part of this module) — this isolates the exact mechanism the
    /// Python test's docstring describes: a single connected chain
    /// a-b-c-d-e-f-g-h that a naive independent-BFS 2-coloring gets half
    /// right, half wrong, because bond file order causes the real algorithm
    /// to effectively re-seed at `g` partway through a pass.
    ///
    /// Atom order/types/bonds mirror real geostd ligand X4Q's minimal fused
    /// 6+5-ring cc/cd/nc/nd core (bridgehead nitrogen `x` excluded from the
    /// family as type `na`, mirroring real N24 in that structure).
    /// Independently hand-traced against the ported algorithm (not just
    /// copied from the Python's asserted expectation) to confirm the port's
    /// pass/reseed bookkeeping reproduces it exactly.
    #[test]
    fn test_atadjust_reseed_is_pass_scoped_not_per_component() {
        // indices: a=0 b=1 c=2 d=3 e=4 f=5 g=6 h=7 x=8
        let mol = mol_graph(
            9,
            vec![
                (0, 1, BondOrder::Double), // a-b
                (0, 8, BondOrder::Single), // a-x (x excluded from family)
                (6, 5, BondOrder::Single), // g-f
                (6, 7, BondOrder::Double), // g-h
                (5, 4, BondOrder::Double), // f-e
                (4, 3, BondOrder::Single), // e-d
                (4, 8, BondOrder::Single), // e-x
                (2, 1, BondOrder::Single), // c-b
                (2, 3, BondOrder::Double), // c-d
                (7, 8, BondOrder::Single), // h-x
            ],
        );
        let pairs: HashMap<&str, &str> = HashMap::from([("cc", "cd"), ("nc", "nd")]);

        let mut types: Vec<String> = vec![
            "cc".to_string(), // a
            "cc".to_string(), // b
            "cc".to_string(), // c
            "nc".to_string(), // d
            "cc".to_string(), // e
            "cc".to_string(), // f
            "cc".to_string(), // g
            "nc".to_string(), // h
            "na".to_string(), // x -- not in `pairs`, must stay untouched
        ];
        relabel_conjugated_alternation(&mol, &mut types, &pairs, false);

        let expected: Vec<String> = ["cc", "cd", "cd", "nc", "cd", "cc", "cc", "nd", "na"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(types, expected);
    }

    /// A single unreachable family atom (no family-family bonds at all)
    /// stays seeded at its default sign (+1, i.e. unflipped) and is never
    /// relabeled -- `num == 0` after the initial count-only loop for a
    /// zero-family molecule short-circuits immediately (mirrors Python's
    /// `if num == 0: return`), and a lone family atom with no family
    /// neighbor never gets a `-1` sign to flip on.
    #[test]
    fn test_no_family_atoms_is_a_no_op() {
        let mol = mol_graph(2, vec![(0, 1, BondOrder::Single)]);
        let pairs: HashMap<&str, &str> = HashMap::from([("X", "Y")]);
        let mut types: Vec<String> = vec!["A".to_string(), "B".to_string()];
        relabel_conjugated_alternation(&mol, &mut types, &pairs, false);
        assert_eq!(types, vec!["A".to_string(), "B".to_string()]);
    }
}
