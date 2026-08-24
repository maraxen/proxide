//! Chemical environment parsing and matching — f9 neighbor-spec grammar.
//!
//! Parses and matches the f9 field of GAFF2 rules: neighbor-connectivity
//! patterns like `(N3,N3,N3)`, `(C3(C3))`, `(XX[AR1],XX[AR1],XX[AR1])`,
//! `(XD3[sb',db])`. Direct, faithful, behavior-preserving port of
//! `proxide.chem.gaff2`'s f9 grammar: `NeighborSpec`, `ChemEnvExpr`,
//! `_TokenStream`, `parse_chem_env`, `_parse_paren_group`,
//! `_parse_neighbor_spec`, `chem_env_matches`, `_match_neighbor_specs`,
//! `_neighbor_matches` (`src/proxide/chem/gaff2.py:115-342`).
//!
//! This is a **behavior-preserving port** — known residual quirks in the
//! Python (e.g. the unused `predecessor` parameter of `chem_env_matches`,
//! see its doc comment below) are preserved exactly, not "fixed".
//!
//! # Atom representation: `&MolGraph` + index, not an RDKit-style object
//!
//! The Python operates on RDKit atom objects via duck typing (`GetSymbol`,
//! `GetDegree()`/`GetNumImplicitHs()`, `GetNeighbors()`,
//! `GetOwningMol().GetBondBetweenAtoms(...)`). This crate has no RDKit
//! dependency; per the established convention already used by the sibling
//! `atomic_prop.rs`/`atom_bond_facts.rs` ports (own `&MolGraph` + a plain
//! `atom_idx: usize`, not a wrapper object), this module's matching
//! functions take the same shape: `(mol: &MolGraph, atom_idx: usize)`
//! everywhere the Python passes an atom. `MolGraph` itself has no adjacency
//! index (bonds are a flat, order-load-bearing `Vec`, see `mol.rs`'s doc),
//! so this module derives per-atom neighbor lists and edge lookups locally
//! ([`neighbor_indices`], [`find_bond`]) rather than depending on a private
//! helper from `atom_bond_facts.rs`.
//!
//! f9's own-property (`own_props`) and edge-bond (`edge_bond_reqs`)
//! evaluation delegate to the real, already-ported sibling modules:
//! [`crate::atomic_prop::atomic_prop_matches`] for `own_props`, and
//! [`crate::atom_bond_facts::bond_category_facts`] for `edge_bond_reqs`.

use crate::atom_bond_facts::{atom_bond_facts, attached_count, bond_category_facts};
use crate::atomic_prop::{atomic_prop_matches, parse_atomic_prop, AtomicPropExpr};
use crate::def_parser::WildatomMap;
use crate::mol::{Bond, MolGraph};

/// One entry of a parsed f9 pattern: an element-or-WILDATOM token,
/// optionally with that neighbor's own attached-atom count, an `[...]`
/// bracket of edge-bond requirements and/or the neighbor's own f8-style
/// properties, and an optional nested `(...)` recursing one more hop out.
/// Direct port of `gaff2.py:114-120`'s `NeighborSpec`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighborSpec {
    pub elem_or_wild: String,
    pub attached_count: Option<usize>,
    pub own_props: Option<AtomicPropExpr>,
    /// `(bond_word, must_have)` pairs, extracted from `'`/`''`-suffixed
    /// bracket tokens.
    pub edge_bond_reqs: Option<Vec<(String, bool)>>,
    pub nested: Option<ChemEnvExpr>,
}

/// A parsed f9 pattern: an AND of existential, pairwise-distinct neighbor
/// requirements. Direct port of `gaff2.py:123-125`'s `ChemEnvExpr`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChemEnvExpr {
    pub neighbors: Vec<NeighborSpec>,
}

/// Character cursor over one f9 pattern string. Direct port of
/// `gaff2.py:128-142`'s `_TokenStream`. `peek`/`advance` return `'\0'` past
/// EOF, mirroring Python's `""` sentinel (never matches any real grammar
/// character, so callers' `==` comparisons behave identically).
struct TokenStream {
    chars: Vec<char>,
    pos: usize,
}

impl TokenStream {
    fn new(raw: &str) -> Self {
        TokenStream {
            chars: raw.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> char {
        self.chars.get(self.pos).copied().unwrap_or('\0')
    }

    fn advance(&mut self) -> char {
        let ch = self.peek();
        self.pos += 1;
        ch
    }

    fn eof(&self) -> bool {
        self.pos >= self.chars.len()
    }
}

/// Parse an f9 pattern (including its outer parens) into an AST. Direct
/// port of `gaff2.py:145-155`'s `parse_chem_env`.
pub fn parse_chem_env(raw: &str) -> Option<ChemEnvExpr> {
    let raw = raw.trim();
    if raw.is_empty() || raw == "*" {
        return None;
    }
    if !raw.starts_with('(') {
        return None;
    }

    let mut stream = TokenStream::new(raw);
    parse_paren_group(&mut stream)
}

/// Direct port of `gaff2.py:158-196`'s `_parse_paren_group`: a single
/// flat, depth-tracking scan of one `(...)` group that segments top-level
/// comma-separated neighbor entries (verbatim-swallowing `[...]` brackets
/// so an internal `,`/`.` never gets mistaken for a top-level separator),
/// deferring recursive re-parsing of any nested `(...)` to
/// `_parse_neighbor_spec`.
fn parse_paren_group(stream: &mut TokenStream) -> Option<ChemEnvExpr> {
    if stream.peek() != '(' {
        return None;
    }
    stream.advance(); // consume "("

    let mut neighbors: Vec<Option<NeighborSpec>> = Vec::new();
    let mut current = String::new();
    let mut depth: i32 = 0;

    while !stream.eof() {
        let ch = stream.peek();
        if ch == '(' || ch == '[' {
            if ch == '(' {
                depth += 1;
            }
            current.push(stream.advance());
            if ch == '[' {
                // consume through matching "]" verbatim (no nesting of [] within [])
                while !stream.eof() && stream.peek() != ']' {
                    current.push(stream.advance());
                }
                if !stream.eof() {
                    current.push(stream.advance());
                }
            }
            continue;
        }
        if ch == ')' {
            if depth == 0 {
                stream.advance(); // consume closing ")"
                break;
            }
            depth -= 1;
            current.push(stream.advance());
            continue;
        }
        if ch == ',' && depth == 0 {
            neighbors.push(parse_neighbor_spec(&current));
            current.clear();
            stream.advance();
            continue;
        }
        current.push(stream.advance());
    }

    if !current.is_empty() {
        neighbors.push(parse_neighbor_spec(&current));
    }

    let resolved: Vec<NeighborSpec> = neighbors.into_iter().flatten().collect();
    if resolved.is_empty() {
        None
    } else {
        Some(ChemEnvExpr {
            neighbors: resolved,
        })
    }
}

/// Direct port of `gaff2.py:202-262`'s `_parse_neighbor_spec`. The element
/// head is matched against `^([A-Za-z]+)(\d*)` (no `$` anchor — this is
/// `gaff2.py:199`'s `_NEIGHBOR_HEAD_RE`, distinct from the f8 grammar's
/// fully-anchored `_PROP_TOKEN_RE`), hand-rolled since no `regex` crate is
/// available in this crate (see `atomic_prop.rs`'s analogous
/// `match_prop_token` for the same approach on the f8 side).
///
/// CRITICAL (recon note, 260820 bug): a bracket body containing `.` (the
/// DEF footer's documented ring/aromaticity OR, e.g. nv/nm/nn's
/// `(XX[AR1.AR2.AR3])`) must be parsed as `parse_atomic_prop(bracket_body)`
/// directly, *without* first splitting on `,` -- silently rewriting that
/// `.` into an AND-joining `,` before `parse_atomic_prop` ever sees it made
/// `(XX[AR1.AR2.AR3])` unsatisfiable (an atom has exactly one aromaticity
/// class, so `AND(AR1, AR2, AR3)` can never hold) and was the root cause of
/// a real mistyping (aniline's N falling through to `n8` instead of `nv`).
/// See `test_parse_chem_env_dot_bracket_is_or_not_and` below.
fn parse_neighbor_spec(raw: &str) -> Option<NeighborSpec> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }

    let chars: Vec<char> = raw.chars().collect();
    let n = chars.len();

    let mut i = 0;
    while i < n && chars[i].is_ascii_alphabetic() {
        i += 1;
    }
    if i == 0 {
        return None; // `[A-Za-z]+` requires >=1 letter at the start
    }
    let elem_or_wild: String = chars[..i].iter().collect();

    let mut j = i;
    while j < n && chars[j].is_ascii_digit() {
        j += 1;
    }
    let attached_count: Option<usize> = if j > i {
        chars[i..j].iter().collect::<String>().parse().ok()
    } else {
        None
    };

    let mut rest: String = chars[j..].iter().collect();

    let mut own_props: Option<AtomicPropExpr> = None;
    let mut edge_bond_reqs: Vec<(String, bool)> = Vec::new();
    let mut nested: Option<ChemEnvExpr> = None;

    if rest.starts_with('[') {
        // f9 bracket content is ASCII-only in every real ATOMTYPE_GFF2.DEF
        // pattern (element symbols, digits, bond-word letters, `'`/`,`/`.`),
        // so byte indices below coincide with char indices exactly as
        // Python's `str.find`/slicing do for this grammar.
        if let Some(close) = rest.find(']') {
            let bracket_body = rest[1..close].to_string();
            rest = rest[close + 1..].to_string();

            if bracket_body.contains('.') {
                own_props = parse_atomic_prop(&bracket_body);
            } else {
                let mut plain_tokens: Vec<String> = Vec::new();
                for raw_tok in bracket_body.split(',') {
                    let raw_tok = raw_tok.trim();
                    if raw_tok.is_empty() {
                        continue;
                    }
                    if let Some(stem) = raw_tok.strip_suffix("''") {
                        edge_bond_reqs.push((stem.to_string(), false));
                    } else if let Some(stem) = raw_tok.strip_suffix('\'') {
                        edge_bond_reqs.push((stem.to_string(), true));
                    } else {
                        plain_tokens.push(raw_tok.to_string());
                    }
                }
                if !plain_tokens.is_empty() {
                    own_props = parse_atomic_prop(&plain_tokens.join(","));
                }
            }
        }
        // If no closing "]" was found, `rest` is left untouched (still
        // starts with "[") and bracket parsing is skipped entirely --
        // mirrors the Python, which never reassigns `rest` in that case.
    }

    if rest.starts_with('(') {
        let mut nested_stream = TokenStream::new(&rest);
        nested = parse_paren_group(&mut nested_stream);
    }

    Some(NeighborSpec {
        elem_or_wild,
        attached_count,
        own_props,
        edge_bond_reqs: if edge_bond_reqs.is_empty() {
            None
        } else {
            Some(edge_bond_reqs)
        },
        nested,
    })
}

/// Real graph neighbor indices of `atom_idx`, in `mol.bonds` order --
/// mirrors RDKit's `GetNeighbors()` iteration order, which
/// `_match_neighbor_specs`'s first-match-wins backtracking depends on.
/// `MolGraph` has no adjacency index (see module doc), so this is a linear
/// scan; local to this module since `atom_bond_facts.rs`'s equivalent
/// (`bonds_incident`) is private there.
fn neighbor_indices(mol: &MolGraph, atom_idx: usize) -> Vec<usize> {
    mol.bonds
        .iter()
        .filter_map(|b| {
            if b.i == atom_idx {
                Some(b.j)
            } else if b.j == atom_idx {
                Some(b.i)
            } else {
                None
            }
        })
        .collect()
}

/// The bond directly connecting atoms `a` and `b`, if any. Mirrors RDKit's
/// `GetOwningMol().GetBondBetweenAtoms(a, b)`, used by `edge_bond_reqs`
/// matching.
fn find_bond(mol: &MolGraph, a: usize, b: usize) -> Option<&Bond> {
    mol.bonds
        .iter()
        .find(|bond| (bond.i == a && bond.j == b) || (bond.i == b && bond.j == a))
}

/// Direct port of `gaff2.py:265-266`'s `_resolve_wildatom`.
fn resolve_wildatom(token: &str, wildatom_map: &WildatomMap) -> Vec<String> {
    wildatom_map
        .get(token)
        .cloned()
        .unwrap_or_else(|| vec![token.to_string()])
}

/// Check whether the real neighbor graph of `mol`'s atom `atom_idx`
/// satisfies a parsed f9 pattern. Direct port of `gaff2.py:269-297`'s
/// `chem_env_matches`.
///
/// **Preserved quirk (behavior-preserving port, not a fix):** the Python's
/// `predecessor` parameter is accepted but never read in the function
/// body -- `atom` itself is always what gets threaded through as the
/// "predecessor" reference for the recursive neighbor matching (see
/// `_neighbor_matches`'s edge-bond lookup and its own recursive
/// `chem_env_matches(..., predecessor=cand)` call, which passes the *same*
/// atom it just walked to, not any ancestor). The docstring's talk of
/// "excluding predecessor" describes `_match_neighbor_specs`'s
/// pairwise-distinct candidate consumption, not this unused parameter.
/// Confirmed against the Python source (`gaff2.py:269-297`) and every call
/// site (`gaff2.py:339`, `tests/test_gaff2_golden.py:831,836`) -- ported
/// exactly, including the dead parameter, per this task's
/// behavior-preservation mandate.
pub fn chem_env_matches(
    expr: Option<&ChemEnvExpr>,
    mol: &MolGraph,
    atom_idx: usize,
    wildatom_map: &WildatomMap,
    _predecessor: Option<usize>,
) -> bool {
    let Some(expr) = expr else {
        return true;
    };
    let candidates = neighbor_indices(mol, atom_idx);
    match_neighbor_specs(&expr.neighbors, &candidates, mol, atom_idx, wildatom_map)
}

/// Direct port of `gaff2.py:300-309`'s `_match_neighbor_specs`: backtracking
/// bipartite matching -- each successive spec claims one distinct candidate
/// (by index-into-`candidates`, not atom identity, matching Python's
/// `candidates[:i] + candidates[i+1:]`), trying every candidate and
/// backtracking on failure.
fn match_neighbor_specs(
    specs: &[NeighborSpec],
    candidates: &[usize],
    mol: &MolGraph,
    predecessor: usize,
    wildatom_map: &WildatomMap,
) -> bool {
    let Some((spec, rest)) = specs.split_first() else {
        return true;
    };
    for i in 0..candidates.len() {
        if neighbor_matches(spec, candidates[i], mol, predecessor, wildatom_map) {
            let mut remaining: Vec<usize> = Vec::with_capacity(candidates.len() - 1);
            remaining.extend_from_slice(&candidates[..i]);
            remaining.extend_from_slice(&candidates[i + 1..]);
            if match_neighbor_specs(rest, &remaining, mol, predecessor, wildatom_map) {
                return true;
            }
        }
    }
    false
}

/// Direct port of `gaff2.py:312-342`'s `_neighbor_matches`.
fn neighbor_matches(
    spec: &NeighborSpec,
    cand_idx: usize,
    mol: &MolGraph,
    predecessor: usize,
    wildatom_map: &WildatomMap,
) -> bool {
    let resolved = resolve_wildatom(&spec.elem_or_wild, wildatom_map);
    let cand_symbol = mol.elements[cand_idx].as_str();
    if !resolved.iter().any(|s| s.as_str() == cand_symbol) {
        return false;
    }

    if let Some(want) = spec.attached_count {
        if attached_count(mol, cand_idx) != want {
            return false;
        }
    }

    // `if spec.edge_bond_reqs:` in Python is falsy for both `None` and an
    // empty list -- `_parse_neighbor_spec` only ever produces `None` for
    // "no requirements" (`edge_bond_reqs or None`), but a directly
    // constructed `Some(vec![])` (e.g. in a test) must behave identically.
    if let Some(edge_bond_reqs) = spec.edge_bond_reqs.as_ref().filter(|v| !v.is_empty()) {
        let Some(bond) = find_bond(mol, cand_idx, predecessor) else {
            return false;
        };
        let edge_facts = bond_category_facts(bond);
        for (bond_word, must_have) in edge_bond_reqs {
            let has_it = edge_facts.get(bond_word.as_str()).copied().unwrap_or(false);
            if has_it != *must_have {
                return false;
            }
        }
    }

    if let Some(own_props) = spec.own_props.as_ref() {
        let cand_facts = atom_bond_facts(mol, cand_idx);
        if !atomic_prop_matches(Some(own_props), &cand_facts) {
            return false;
        }
    }

    if let Some(nested) = spec.nested.as_ref() {
        if !chem_env_matches(Some(nested), mol, cand_idx, wildatom_map, Some(cand_idx)) {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mol::BondOrder;

    fn mk_bond(i: usize, j: usize) -> Bond {
        Bond {
            i,
            j,
            order: BondOrder::Single,
            aromatic: false,
        }
    }

    // -----------------------------------------------------------------
    // f9 grammar parsing fixtures, ported from
    // `tests/test_gaff2_golden.py::TestGaff2Grammar`.
    // -----------------------------------------------------------------

    /// Port of `test_parse_chem_env_simple_count_list`
    /// (`tests/test_gaff2_golden.py:730-735`).
    #[test]
    fn test_parse_chem_env_simple_count_list() {
        let expr = parse_chem_env("(N3,N3,N3)").expect("should parse");
        assert_eq!(expr.neighbors.len(), 3);
        assert!(expr
            .neighbors
            .iter()
            .all(|n| n.elem_or_wild == "N" && n.attached_count == Some(3)));
    }

    /// Port of `test_parse_chem_env_bracket_with_edge_suffix`
    /// (`tests/test_gaff2_golden.py:737-745`).
    #[test]
    fn test_parse_chem_env_bracket_with_edge_suffix() {
        let expr = parse_chem_env("(XD3[sb',db])").expect("should parse");
        assert_eq!(expr.neighbors.len(), 1);
        let n = &expr.neighbors[0];
        assert_eq!(n.elem_or_wild, "XD");
        assert_eq!(n.attached_count, Some(3));
        assert_eq!(n.edge_bond_reqs, Some(vec![("sb".to_string(), true)]));
        let own_props = n.own_props.as_ref().expect("own_props");
        assert_eq!(
            own_props
                .tokens
                .iter()
                .map(|t| t.word.as_str())
                .collect::<Vec<_>>(),
            vec!["db"]
        );
    }

    /// Port of `test_parse_chem_env_nested_recursion`
    /// (`tests/test_gaff2_golden.py:747-757`).
    #[test]
    fn test_parse_chem_env_nested_recursion() {
        let expr = parse_chem_env("(C3(C3))").expect("should parse");
        assert_eq!(expr.neighbors.len(), 1);
        let n = &expr.neighbors[0];
        assert_eq!(n.elem_or_wild, "C");
        assert_eq!(n.attached_count, Some(3));
        let nested = n.nested.as_ref().expect("nested");
        assert_eq!(nested.neighbors.len(), 1);
        assert_eq!(nested.neighbors[0].elem_or_wild, "C");
        assert_eq!(nested.neighbors[0].attached_count, Some(3));
    }

    /// Port of `test_parse_chem_env_aromatic_neighbor_bracket`
    /// (`tests/test_gaff2_golden.py:759-768`) -- "the exact syntax that was
    /// silently no-op'd before this fix (cp's rule)."
    #[test]
    fn test_parse_chem_env_aromatic_neighbor_bracket() {
        let expr = parse_chem_env("(XX[AR1],XX[AR1],XX[AR1])").expect("should parse");
        assert_eq!(expr.neighbors.len(), 3);
        for n in &expr.neighbors {
            assert_eq!(n.elem_or_wild, "XX");
            let own_props = n.own_props.as_ref().expect("own_props");
            assert_eq!(
                own_props
                    .tokens
                    .iter()
                    .map(|t| t.word.as_str())
                    .collect::<Vec<_>>(),
                vec!["AR1"]
            );
        }
    }

    /// Regression test for the recon-flagged CRITICAL bug (260820): a
    /// bracket body containing "." must be parsed as
    /// `parse_atomic_prop(bracket_body)` directly (preserving "." as OR),
    /// NOT split on "," first (which would silently turn "." into an
    /// unsatisfiable AND). This is nv/nm/nn's real pattern; not a literal
    /// existing Python fixture at the `parse_chem_env` level (only
    /// `parse_atomic_prop("AR1.AR2.AR3")` is tested directly, at the f8
    /// level, in `atomic_prop.rs::test_parse_atomic_prop_or_list`), but the
    /// exact scenario the recon notes call out as the single highest-risk
    /// part of this module.
    #[test]
    fn test_parse_chem_env_dot_bracket_is_or_not_and() {
        let expr = parse_chem_env("(XX[AR1.AR2.AR3])").expect("should parse");
        assert_eq!(expr.neighbors.len(), 1);
        let n = &expr.neighbors[0];
        assert_eq!(n.elem_or_wild, "XX");
        assert!(n.edge_bond_reqs.is_none());
        let own_props = n.own_props.as_ref().expect("own_props");
        assert_eq!(
            own_props
                .tokens
                .iter()
                .map(|t| t.word.as_str())
                .collect::<Vec<_>>(),
            vec!["AR1", "AR2", "AR3"]
        );
    }

    // -----------------------------------------------------------------
    // chem_env_matches fixture, ported from
    // `tests/test_gaff2_golden.py::TestGaff2Grammar
    //     ::test_chem_env_matches_hw_pattern_reachable_though_unreached`.
    //
    // The Python test builds real RDKit molecules (`Chem.AddHs(Chem
    // .MolFromSmiles(...))`); this crate has no RDKit dependency, so a
    // `MolGraph` literal stands in, wired to reproduce the exact same
    // connectivity (water: O bonded to two explicit H's; methane: one C
    // bonded to one explicit H, matching the Python test's actual access
    // pattern of only ever inspecting `methane_h`).
    // -----------------------------------------------------------------

    /// Port of `test_chem_env_matches_hw_pattern_reachable_though_unreached`
    /// (`tests/test_gaff2_golden.py:808-836`): confirms the 260820
    /// H-typing rework's fix to `chem_env_matches`'s neighbor-candidate
    /// filtering (candidates must include H atoms, not just heavy atoms --
    /// `hw`'s `(O(H1))` pattern is the only f9 pattern anywhere in
    /// ATOMTYPE_GFF2.DEF that targets H as an explicit element).
    #[test]
    fn test_chem_env_matches_hw_pattern_reachable_though_unreached() {
        // O(0) bonded to H(1) and H(2), mirroring
        // `Chem.AddHs(Chem.MolFromSmiles("O"))`.
        let water = MolGraph {
            elements: vec!["O".to_string(), "H".to_string(), "H".to_string()],
            formal_charges: vec![0, 0, 0],
            bonds: vec![mk_bond(0, 1), mk_bond(0, 2)],
            rings: None,
        };
        let expr = parse_chem_env("(O(H1))").expect("should parse");
        let wildatom_map = WildatomMap::new();

        // h_atoms[0] (atom_idx 1)'s own f9 pattern is "(O(H1))": its
        // neighbor must be O, and that O's neighbor (predecessor loop-back
        // onto atom_idx 1 itself is a permitted candidate, per the Python
        // docstring) must be a 1-attached H -- atom_idx 1 itself satisfies
        // this.
        assert!(chem_env_matches(
            Some(&expr),
            &water,
            1,
            &wildatom_map,
            None
        ));

        // Sanity check the negative: methane's H has no O neighbor at all.
        // C(0) bonded to H(1), mirroring
        // `Chem.AddHs(Chem.MolFromSmiles("C"))` (only the H atom the
        // Python test actually inspects is modeled).
        let methane = MolGraph {
            elements: vec!["C".to_string(), "H".to_string()],
            formal_charges: vec![0, 0],
            bonds: vec![mk_bond(0, 1)],
            rings: None,
        };
        assert!(!chem_env_matches(
            Some(&expr),
            &methane,
            1,
            &wildatom_map,
            None
        ));
    }
}
