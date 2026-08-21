//! GAFF2 atom-type assignment orchestration.
//!
//! Top-level entry point, ported from `assign_gaff2_atom_types`
//! (`src/proxide/chem/gaff2.py:1115-1216`): loads GAFF2 rules in DEF-file
//! declaration order, applies them to every atom -- heavy AND hydrogen, per
//! the 260820 fix that put H atoms through the same rule loop instead of a
//! separate heuristic -- with first-match-in-file-order precedence, falls
//! back to a fixed element-keyed default for any atom no rule matched, then
//! runs the two independent bond-alternation passes over the result.
//!
//! ## Kekulization is not this module's job
//!
//! Python's `assign_gaff2_atom_types` makes a local `Chem.Mol` copy and
//! calls `Chem.Kekulize(mol_for_matching, clearAromaticFlags=False)` before
//! matching (gaff2.py:1162-1173), deliberately re-raising (after a warning)
//! on `Chem.KekulizeException` rather than silently matching against an
//! un-Kekulized molecule -- because the f8/f9 `sb`/`db` bond-category
//! tokens require true Kekule single/double bond identity, and RDKit
//! reports `AROMATIC` (not `SINGLE`/`DOUBLE`) for un-Kekulized ring bonds.
//! `clearAromaticFlags=False` is load-bearing too: it keeps
//! `GetIsAromatic()` intact so AB/DL and aromaticity-class matching, which
//! read that flag rather than bond type, are unaffected by the Kekulize
//! call.
//!
//! This crate's [`MolGraph`] makes the "must be Kekulized" precondition
//! structural instead of a runtime check: [`crate::mol::BondOrder`] has
//! only `Single`, `Double`, `Triple` variants -- there is no `Aromatic`
//! variant, so an un-Kekulized bond cannot be represented here at all.
//! Each [`crate::mol::Bond`]'s separate `aromatic` flag carries over
//! RDKit's `GetIsAromatic()` (as preserved by `clearAromaticFlags=False`)
//! for AB/DL / aromaticity-class matching, exactly mirroring the Python
//! contract's two-track (Kekule bond type + preserved aromaticity flag)
//! design.
//!
//! Concretely, this pushes "Kekulize a local copy, fail loudly (don't
//! silently degrade) on failure" to whatever constructs the `MolGraph` in
//! the first place -- an RDKit-backed bridge outside this pure-Rust crate.
//! By the time `assign_gaff2_atom_types` runs here, that step has already
//! succeeded or the `MolGraph` could not have been built; there is nothing
//! left to silently fall back on at this point, which is the actual
//! property Python's guard exists to protect. This is a deliberate
//! architectural deviation from a line-for-line port of gaff2.py:1162-1173,
//! not an oversight -- see this port's final report for why re-deriving
//! Kekulization in pure Rust was out of scope for the orchestrate module.

use crate::alternation;
use crate::def_parser::WildatomMap;
use crate::mol::MolGraph;

/// Contract a GAFF2 rule type must satisfy to participate in the top-level
/// assignment loop below.
///
/// `def_parser::Gaff2Rule` (one parsed `ATD` line from `ATOMTYPE_GFF2.DEF`)
/// is the type this describes -- its `matches()` (port of
/// `Gaff2Rule.matches`, gaff2.py:699-725) depends on `atom_bond_facts`,
/// `atomic_prop`, and `chem_env`, which are separate, independently-gated
/// ports still in progress as of this writing. Expressing the loop below
/// generically over this trait, rather than directly against the concrete
/// `Gaff2Rule` (which cannot yet honor it), lets [`assign_types_with_rules`]
/// be exercised by real, non-panicking unit tests today, and lets
/// `def_parser::Gaff2Rule` opt in with a plain `impl` once its own porting
/// pass lands -- no change needed here at that point.
pub trait Gaff2RuleMatch {
    /// The GAFF2 atom type this rule assigns on a match (DEF field f2).
    fn atom_type(&self) -> &str;

    /// Whether this rule matches the atom at `atom_idx` in `mol`, given
    /// `wildatom_map` for WILDATOM-expanded f9 chem-env matching. Port of
    /// `Gaff2Rule.matches(atom, wildatom_map)` (gaff2.py:699-725).
    fn matches(&self, atom_idx: usize, mol: &MolGraph, wildatom_map: &WildatomMap) -> bool;
}

/// Fallback ladder for any atom no rule matched, ported verbatim from
/// gaff2.py:1188-1201's `else` branch (`atomic_num == 1/6/7/8/16` /
/// anything-else chain): H -> `ha` (DEF line 91's unconstrained H
/// catch-all), C -> `c3`, N -> `n3`, O -> `oh`, S -> `s`, anything else ->
/// `x`.
///
/// Ported against `MolGraph::elements`'s element-symbol representation
/// (`"H"`/`"C"`/`"N"`/`"O"`/`"S"`) rather than Python's `atom.GetAtomicNum()`
/// integer -- the two are equivalent for exactly these five elements, and
/// `MolGraph` (see `mol.rs`) does not expose atomic numbers at all, only
/// symbols, so there is no faithful alternative representation available
/// to this module.
fn fallback_atom_type(element: &str) -> String {
    match element {
        "H" => "ha",
        "C" => "c3",
        "N" => "n3",
        "O" => "oh",
        "S" => "s",
        _ => "x",
    }
    .to_string()
}

/// Core orchestration logic: assign a GAFF2 atom type to every atom in
/// `mol` (heavy and H, index-aligned with `mol.elements`) using `rules` in
/// precedence order and `wildatom_map` for chem-env matching, then run the
/// two independent bond-alternation passes over the result.
///
/// Ported from `assign_gaff2_atom_types`'s body (gaff2.py:1146-1216), minus
/// the Kekulization step (see the module doc comment) and the
/// `rules is None` / `wildatom_map is None` default-loading branch, which
/// [`assign_gaff2_atom_types`] below is the (currently blocked) wrapper
/// for. Generic over `R: Gaff2RuleMatch` purely for testability -- see that
/// trait's doc comment; this is not a behavioral change from the Python,
/// which took a concrete `list[Gaff2Rule]` for the same
/// injectable-for-testing purpose (`rules: list[Gaff2Rule] | None = None`).
pub fn assign_types_with_rules<R: Gaff2RuleMatch>(
    mol: &MolGraph,
    rules: &[R],
    wildatom_map: &WildatomMap,
) -> Vec<String> {
    let n = mol.elements.len();
    let mut atom_types: Vec<String> = Vec::with_capacity(n);

    // Precedence: first-match-in-file-order -- `rules` must preserve
    // ATOMTYPE_GFF2.DEF's declaration order (per the DEF file's own
    // "defination order is crucial" rule); a slice/`Vec`, iterated in
    // order, never a `HashSet`/`HashMap` over rules, upholds that. H atoms
    // run through this SAME loop as heavy atoms (260820 fix,
    // gaff2.py:1177-1179) -- no atomic-number branch skips or
    // special-cases them here, unlike the pre-260820 Python.
    for atom_idx in 0..n {
        let mut assigned: Option<&str> = None;
        for rule in rules {
            if rule.matches(atom_idx, mol, wildatom_map) {
                assigned = Some(rule.atom_type());
                break;
            }
        }
        atom_types.push(match assigned {
            Some(t) => t.to_string(),
            None => fallback_atom_type(&mol.elements[atom_idx]),
        });
    }

    // Two independent alternation passes (gaff2.py:1203-1214): cc/ce/cg/
    // pc/pe/nc/ne and cp are disjoint families that must never
    // cross-propagate through each other even if adjacent in the molecule
    // -- each is its own call with its own pair map, not one merged pass.
    // `single_seed_only=true` on the second call reproduces real
    // AmberTools' `cpadjust()`, which (unlike `atadjust()`) never reseeds a
    // second disconnected same-family component -- see `alternation`'s
    // module docs for why this is a real, intentional asymmetry between
    // the two AmberTools C functions this call site distinguishes, not a
    // flag on one shared algorithm.
    alternation::relabel_conjugated_alternation(
        mol,
        &mut atom_types,
        &alternation::CONJUGATED_ALTERNATION_PAIRS,
        false,
    );
    alternation::relabel_conjugated_alternation(
        mol,
        &mut atom_types,
        &alternation::BIPHENYL_ALTERNATION_PAIR,
        true,
    );

    atom_types
}

/// Assign GAFF2 atom types to every atom in `mol`, using the default
/// (bundled `ATOMTYPE_GFF2.DEF`-parsed) rule set.
///
/// # Returns
/// A vector of atom type strings, one per atom, in `mol`'s atom index
/// order -- including H atoms. Never returns `None`/empty-for-an-atom:
/// the fallback ladder in [`assign_types_with_rules`] guarantees every
/// atom gets *some* type even if no DEF rule matches it.
///
/// Port of gaff2.py:1140-1216's `rules is None` default-loading branch:
/// loads the bundled `ATOMTYPE_GFF2.DEF` (parsed at most once per process,
/// see [`crate::rules_loader::get_default_rules`]'s own doc), then delegates
/// to [`assign_types_with_rules`] for the rest of the body. `def_parser::
/// Gaff2Rule`'s [`Gaff2RuleMatch`] impl lives in `def_parser.rs` itself (per
/// this function's original doc note) rather than here, so no change was
/// needed in this module once that impl landed.
///
/// # Errors
/// Propagates any I/O or parse error from loading the bundled DEF file (see
/// [`crate::rules_loader::get_default_rules`]) as a `String`, matching this
/// function's `Result` contract -- Python's equivalent has no such failure
/// mode (`_get_default_rules()` never raises), so this is a Rust-idiomatic
/// widening of the error surface, not a behavior difference in the success
/// path.
pub fn assign_gaff2_atom_types(mol: &MolGraph) -> Result<Vec<String>, String> {
    let (rules, wildatom_map) = crate::rules_loader::get_default_rules()?;
    Ok(assign_types_with_rules(mol, &rules, &wildatom_map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mol::{Bond, BondOrder};

    /// Build a minimal `MolGraph` directly via struct literal, bypassing
    /// `MolGraph::new()` (still a `todo!()` stub as of this writing) --
    /// same convention already established by `alternation.rs`'s own test
    /// helper. Element identity/formal charges are irrelevant to most of
    /// these tests beyond feeding the fallback ladder.
    fn mol_graph(elements: Vec<&str>, bonds: Vec<(usize, usize, BondOrder)>) -> MolGraph {
        let elements: Vec<String> = elements.into_iter().map(str::to_string).collect();
        let formal_charges: Vec<i8> = vec![0; elements.len()];
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

    /// A rule whose match decision is an injected predicate over
    /// `(atom_idx, mol)`, entirely decoupled from the real (unimplemented)
    /// `atom_bond_facts`/`atomic_prop`/`chem_env` machinery -- this is what
    /// [`Gaff2RuleMatch`] exists to make possible.
    struct MockRule {
        atom_type: &'static str,
        predicate: fn(usize, &MolGraph) -> bool,
    }

    impl Gaff2RuleMatch for MockRule {
        fn atom_type(&self) -> &str {
            self.atom_type
        }

        fn matches(&self, atom_idx: usize, mol: &MolGraph, _wildatom_map: &WildatomMap) -> bool {
            (self.predicate)(atom_idx, mol)
        }
    }

    fn is_element(idx: usize, mol: &MolGraph, sym: &str) -> bool {
        mol.elements[idx] == sym
    }

    #[test]
    fn precedence_is_first_match_in_rule_order() {
        // Two rules that would BOTH match atom 0 (a carbon) -- the first
        // one in the slice must win, mirroring gaff2.py:1180-1186's
        // `for rule in rules: ... break` first-match-wins loop.
        let mol = mol_graph(vec!["C"], vec![]);
        let rules = vec![
            MockRule {
                atom_type: "c3",
                predicate: |i, m| is_element(i, m, "C"),
            },
            MockRule {
                atom_type: "ca",
                predicate: |i, m| is_element(i, m, "C"),
            },
        ];
        let wildatom_map = WildatomMap::new();
        let types = assign_types_with_rules(&mol, &rules, &wildatom_map);
        assert_eq!(types, vec!["c3".to_string()]);
    }

    #[test]
    fn h_atoms_run_through_the_same_rule_loop_as_heavy_atoms() {
        // Port of the intent behind TestHAtomTyping (gaff2.py's 260820
        // rework, tests/test_gaff2_golden.py:926-1033): H atoms must be
        // real candidates for real DEF rules, not skipped or shunted to a
        // separate heuristic. A rule that only ever matches H atoms must
        // fire for the H atom here, proving it entered the loop at all.
        let mol = mol_graph(
            vec!["C", "H"],
            vec![(0, 1, BondOrder::Single)],
        );
        let rules = vec![
            MockRule {
                atom_type: "c3",
                predicate: |i, m| is_element(i, m, "C"),
            },
            MockRule {
                atom_type: "hc",
                predicate: |i, m| is_element(i, m, "H"),
            },
        ];
        let wildatom_map = WildatomMap::new();
        let types = assign_types_with_rules(&mol, &rules, &wildatom_map);
        assert_eq!(types, vec!["c3".to_string(), "hc".to_string()]);
    }

    #[test]
    fn fallback_ladder_matches_python_exactly_for_unmatched_atoms() {
        // Port of gaff2.py:1188-1201's fallback chain, exercised with no
        // rules at all so every atom falls through to it.
        let mol = mol_graph(vec!["H", "C", "N", "O", "S", "P", "Cl"], vec![]);
        let rules: Vec<MockRule> = vec![];
        let wildatom_map = WildatomMap::new();
        let types = assign_types_with_rules(&mol, &rules, &wildatom_map);
        assert_eq!(
            types,
            vec!["ha", "c3", "n3", "oh", "s", "x", "x"]
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn unmatched_atom_falls_back_even_when_other_atoms_matched_a_rule() {
        // A rule that matches only carbon leaves the nitrogen to the
        // fallback ladder in the SAME call -- fallback is per-atom, not
        // all-or-nothing for the molecule.
        let mol = mol_graph(vec!["C", "N"], vec![]);
        let rules = vec![MockRule {
            atom_type: "c3",
            predicate: |i, m| is_element(i, m, "C"),
        }];
        let wildatom_map = WildatomMap::new();
        let types = assign_types_with_rules(&mol, &rules, &wildatom_map);
        assert_eq!(types, vec!["c3".to_string(), "n3".to_string()]);
    }

    #[test]
    fn two_alternation_passes_run_disjoint_families_without_cross_propagation() {
        // CRITICAL risk-note coverage (gaff2.py:1211-1214): cc/cd-family
        // and cp/cq-family must be two independent calls. Build one
        // disconnected cc-family pair (double bond -> second atom flips to
        // cd) and one disconnected cp-family pair (double bond -> second
        // atom flips to cq), all typed via rules in one `assign_types_with_
        // rules` call, and confirm both flips happen and neither family
        // leaks into the other (a "cc"-typed atom never becomes "cq" and
        // vice versa).
        let mol = mol_graph(
            vec!["C", "C", "C", "C"],
            vec![
                (0, 1, BondOrder::Double), // cc-family pair
                (2, 3, BondOrder::Double), // cp-family pair (disjoint)
            ],
        );
        let rules = vec![
            MockRule {
                atom_type: "cc",
                predicate: |i, _m| i == 0 || i == 1,
            },
            MockRule {
                atom_type: "cp",
                predicate: |i, _m| i == 2 || i == 3,
            },
        ];
        let wildatom_map = WildatomMap::new();
        let types = assign_types_with_rules(&mol, &rules, &wildatom_map);
        assert_eq!(
            types,
            vec!["cc", "cd", "cp", "cq"]
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn empty_molecule_produces_empty_types() {
        let mol = mol_graph(vec![], vec![]);
        let rules: Vec<MockRule> = vec![];
        let wildatom_map = WildatomMap::new();
        let types = assign_types_with_rules(&mol, &rules, &wildatom_map);
        assert!(types.is_empty());
    }

    /// End-to-end coverage of [`assign_gaff2_atom_types`] itself (not just
    /// [`assign_types_with_rules`]) against the real bundled
    /// `ATOMTYPE_GFF2.DEF`, real `def_parser::Gaff2Rule::matches`, and real
    /// `atom_bond_facts`/`chem_env`/`atomic_prop` matching -- the exact gap
    /// this port's lessons log (D3, "zero end-to-end coverage ... the
    /// module is dead code pending wiring") flagged before this module was
    /// wired up. Benzene, explicit-H, Kekulized (atoms 0-5 the ring, 6-11
    /// each singly bonded to the ring atom of the same index minus 6):
    /// every ring carbon has attached_count 3 (2 ring C neighbors + 1 H)
    /// and, per `classify_ring_aromaticity`'s AR1 condition (total ==
    /// 2*len(ring) == 12, no ring N/P), classifies AR1 -- matching
    /// `ATD ca * 6 3 * * [AR1] &` (DEF line ~54) exactly, not a wildcard
    /// fallback. Each ring H has num_attached=1 and an EW-neighbor count of
    /// 0 (its one carbon neighbor's other neighbors are C/C/H, none of
    /// which are in `EW_ATOMS`), so it fails every `h4`/`h5`/`hx`/... f9
    /// neighbor-pattern rule and falls through to the unconstrained `ha`
    /// catch-all (DEF line 91) -- also a real rule match, not the
    /// orchestrate-level fallback ladder (which would only fire if `ha`
    /// itself were absent from the DEF file).
    #[test]
    fn real_def_file_types_benzene_ring_carbons_ca_and_ring_hydrogens_ha() {
        let mut bonds = Vec::new();
        // Kekulized ring, alternating single/double, all six bonds flagged
        // aromatic (preserves RDKit's GetIsAromatic() through Kekulize --
        // see this module's doc comment).
        let ring_orders = [
            BondOrder::Single,
            BondOrder::Double,
            BondOrder::Single,
            BondOrder::Double,
            BondOrder::Single,
            BondOrder::Double,
        ];
        for i in 0..6 {
            bonds.push(Bond {
                i,
                j: (i + 1) % 6,
                order: ring_orders[i],
                aromatic: true,
            });
        }
        // Each ring carbon's explicit H, non-aromatic single bonds.
        for i in 0..6 {
            bonds.push(Bond {
                i,
                j: 6 + i,
                order: BondOrder::Single,
                aromatic: false,
            });
        }

        let mut elements = vec!["C".to_string(); 6];
        elements.extend(vec!["H".to_string(); 6]);

        let mol = MolGraph::new(elements, bonds, None, None, Some(vec![vec![0, 1, 2, 3, 4, 5]]))
            .expect("well-formed benzene MolGraph");

        let types = assign_gaff2_atom_types(&mol).expect("real bundled DEF should parse and match");

        assert_eq!(types.len(), 12);
        for i in 0..6 {
            assert_eq!(types[i], "ca", "ring carbon {i} should type as ca (AR1)");
        }
        for i in 6..12 {
            assert_eq!(types[i], "ha", "ring hydrogen {i} should type as ha");
        }
    }
}
