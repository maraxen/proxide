//! GAFF2 rule parser.
//!
//! Parses ATOMTYPE_GFF2.DEF rule definitions and wildcard atom definitions.
//! Responsible for extracting `Gaff2Rule` and wildcard atom data structures.
//!
//! Line-for-line port of Python's `src/proxide/chem/gaff2.py` (verified
//! against the current file content at port time — see the note at the
//! bottom of this doc comment if these line numbers ever look stale):
//! - `Gaff2WildAtomDef` (lines 28-36)
//! - `Gaff2Rule` — parse-relevant fields (lines 672-698)
//! - `parse_wildatom_defs` (lines 728-748)
//! - `parse_gaff2_rules` (lines 751-860)
//!
//! # Scope note: `Gaff2Rule.matches()` is NOT ported here
//!
//! Python's `Gaff2Rule` class (lines 672-726) also defines a `matches()`
//! method (lines 699-725) that applies a rule to an RDKit atom. That method
//! depends on atom-level feature extraction (`_attached_count`,
//! `_h_ew_neighbor_count`, `_atom_bond_facts`) and on interpreting the parsed
//! f8/f9 expression trees (`atomic_prop_matches`, `chem_env_matches`) — all
//! of which live in sibling modules (`atom_bond_facts.rs`, `atomic_prop.rs`,
//! `chem_env.rs`) that are still stubs as of this port, and depend on a
//! genuine atom-view abstraction this crate doesn't have yet (`mol::MolGraph`
//! itself is still `todo!()`). This module's job, per its own description,
//! is parsing the DEF file into `Gaff2Rule`/WILDATOM data — rule
//! *application* belongs in a follow-up module once its dependencies exist.
//!
//! # Scope note: f8/f9 are kept as raw, unparsed text
//!
//! Python's `Gaff2Rule.atomic_prop` / `.chem_env` fields hold typed
//! `AtomicPropExpr | None` / `ChemEnvExpr | None` values, produced by
//! `parse_atomic_prop()` / `parse_chem_env()` (the f8 bracket grammar and f9
//! neighbor-spec grammar respectively). Those grammars are separate,
//! still-stubbed sibling modules outside this task's scope, and their
//! Rust signatures/types are not yet settled. This module therefore
//! extracts and stores the *raw* f8/f9 substrings (`atomic_prop_raw`,
//! `chem_env_raw`) — byte-for-byte what Python would hand to
//! `parse_atomic_prop`/`parse_chem_env` — so a follow-up module can parse
//! them into structured exprs without re-deriving the substring boundaries
//! (which are the fiddly, order-of-operations-sensitive part of this parser;
//! see `parse_rule_fields` below).

use indexmap::IndexMap;

use crate::atom_bond_facts::{
    atom_bond_facts, attached_count, h_ew_neighbor_count, num_h_neighbors,
};
use crate::atomic_prop::{atomic_prop_matches, parse_atomic_prop};
use crate::chem_env::{chem_env_matches, parse_chem_env};
use crate::mol::MolGraph;
use crate::orchestrate::Gaff2RuleMatch;

/// WILDATOM symbol -> element-list map (Python's `dict[str, list[str]]`).
///
/// `IndexMap`, not a plain `HashMap`: every current consumer (Python's
/// `_resolve_wildatom`, gaff2.py:265-266) only ever does a keyed
/// `.get(symbol, default)` lookup, never iterates the map, so plain
/// `HashMap` would be behaviorally safe today — but `IndexMap` costs
/// nothing here, matches this crate's established convention for
/// GAFF2-DEF-derived maps, and (unlike `HashMap`) reproduces Python dict's
/// exact "re-assigning an existing key updates its value in place without
/// moving it" insertion-order semantics if a same-symbol WILDATOM line is
/// ever redefined, so a future order-sensitive consumer stays correct for
/// free.
pub type WildatomMap = IndexMap<String, Vec<String>>;

/// WILDATOM definition: a symbol shortcut for a set of element symbols
/// usable in chem-env rule matching (e.g. `XX = C,N,O,S,P`).
///
/// Ports `Gaff2WildAtomDef` (gaff2.py lines 28-36). Note: in the Python
/// source this dataclass is defined but never actually instantiated —
/// `parse_wildatom_defs` returns a plain `dict[str, list[str]]`
/// (`WildatomMap` here), not a list of `Gaff2WildAtomDef`. Ported here for
/// fidelity to the Python module's declared surface, not because anything
/// downstream constructs one (yet).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gaff2WildAtomDef {
    pub symbol: String,
    pub elements: Vec<String>,
}

/// A single GAFF2 atom-typing rule from ATOMTYPE_GFF2.DEF.
///
/// Field correspondence to the ATD line format (per the DEF file's own
/// footer), ported from `Gaff2Rule`'s field declarations (gaff2.py
/// lines 690-697):
/// - f2: `atom_type` — the GAFF2 type to assign (e.g. c3, ca, n)
/// - f3: `residue` — residue name filter (`*` means any)
/// - f4: `atomic_num` — atomic number
/// - f5: `num_attached` — number of attached atoms (heavy + H); `None` means
///   wildcard (`*` or absent in the DEF row), NOT zero
/// - f6: `num_h` — number of hydrogen attachments; `None` means wildcard,
///   NOT zero
/// - f7: `h_ew_count` — H-only electron-withdrawing-neighbor count
///   (meaningful only when atomic_num == 1)
/// - f8: `atomic_prop_raw` — raw bracket-interior text of the atomic-property
///   expression (ring/aromaticity/bond-type-count facts), unparsed — see the
///   module-level "f8/f9 are kept as raw" note
/// - f9: `chem_env_raw` — raw parenthesized chemical-environment neighbor
///   pattern text (including the parens), unparsed — see the module-level
///   "f8/f9 are kept as raw" note
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gaff2Rule {
    pub atom_type: String,
    pub residue: String,
    pub atomic_num: i64,
    pub num_attached: Option<i64>,
    pub num_h: Option<i64>,
    pub h_ew_count: Option<i64>,
    pub atomic_prop_raw: Option<String>,
    pub chem_env_raw: Option<String>,
}

/// Element symbol -> atomic number, for f4 (`atomic_num`) matching against
/// `MolGraph::elements`'s symbol representation. Covers every atomic number
/// actually present in the bundled `ATOMTYPE_GFF2.DEF` (1-109, verified at
/// port time via `awk` over the file's f4 column) plus the standard symbols
/// in between, for a complete, order-correct periodic table rather than a
/// partial one that would silently need extending later. `LP`/`lp`
/// (lone-pair placeholder rows, atomic_num 0) are deliberately absent: no
/// real `MolGraph` atom is ever a lone pair, so a rule keyed on atomic_num
/// 0 can never match a real atom regardless of what this table contains.
const ELEMENT_ATOMIC_NUMBERS: [&str; 109] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
];

fn element_atomic_number(symbol: &str) -> Option<i64> {
    ELEMENT_ATOMIC_NUMBERS
        .iter()
        .position(|&s| s == symbol)
        .map(|i| (i + 1) as i64)
}

impl Gaff2RuleMatch for Gaff2Rule {
    fn atom_type(&self) -> &str {
        &self.atom_type
    }

    /// Port of `Gaff2Rule.matches` (gaff2.py:699-725).
    ///
    /// **Preserved quirk:** `self.residue` (f3) is parsed and stored (see
    /// the struct doc above) but never consulted here -- confirmed against
    /// the Python source, whose `matches()` body never references
    /// `self.residue` at all. Faithfully preserved, not an omission.
    fn matches(&self, atom_idx: usize, mol: &MolGraph, wildatom_map: &WildatomMap) -> bool {
        let Some(atomic_num) = element_atomic_number(&mol.elements[atom_idx]) else {
            return false;
        };
        if self.atomic_num != atomic_num {
            return false;
        }

        if let Some(want) = self.num_attached {
            if want != attached_count(mol, atom_idx) as i64 {
                return false;
            }
        }

        // includeNeighbors=True in Python's GetTotalNumHs -- see
        // num_h_neighbors' own doc for why this is a plain H-element
        // neighbor count against a `MolGraph` (no implicit-H concept).
        if let Some(want) = self.num_h {
            if want != num_h_neighbors(mol, atom_idx) as i64 {
                return false;
            }
        }

        if let Some(want) = self.h_ew_count {
            if want != h_ew_neighbor_count(mol, atom_idx) as i64 {
                return false;
            }
        }

        // Re-parses the raw f8 text on every call rather than caching a
        // pre-parsed `AtomicPropExpr` on the rule (unlike Python, which
        // parses once at rule-construction time) -- see this module's
        // "f8/f9 are kept as raw" scope note for why `Gaff2Rule` stores raw
        // text. `parse_atomic_prop` is a pure function of its input, so
        // re-deriving the same `AtomicPropExpr` per call is a performance
        // deviation only, not a behavioral one; a wildcard/absent f8
        // (`atomic_prop_raw: None`, or raw text that parses to `None`)
        // takes the same "always matches" path Python's
        // `if self.atomic_prop is not None:` guard takes, via
        // `atomic_prop_matches(None, _) == true`.
        if let Some(raw) = self.atomic_prop_raw.as_deref() {
            let expr = parse_atomic_prop(raw);
            let facts = atom_bond_facts(mol, atom_idx);
            if !atomic_prop_matches(expr.as_ref(), &facts) {
                return false;
            }
        }

        // Same re-parse-per-call tradeoff as f8 above, and the same
        // None-means-always-match neutrality via `chem_env_matches`.
        if let Some(raw) = self.chem_env_raw.as_deref() {
            let expr = parse_chem_env(raw);
            if !chem_env_matches(expr.as_ref(), mol, atom_idx, wildatom_map, None) {
                return false;
            }
        }

        true
    }
}

/// Parse WILDATOM definitions from the rule file header.
///
/// Ports `parse_wildatom_defs` (gaff2.py lines 728-748). Lines are
/// space-separated element lists, e.g. `"WILDATOM XX C N O S P"` ->
/// `{"XX": ["C","N","O","S","P"]}` — **not** comma-joined. (A prior bug
/// fixed upstream: a comma-split on `parts[2]` alone silently captured only
/// the first element.)
///
/// Takes pre-split lines (matching Python's `lines: list[str]` parameter),
/// not raw file content — `parse_gaff2_rules` below does the splitting once
/// and shares the result with this function, exactly as Python's
/// `parse_gaff2_rules` does (`lines = content.split("\n")` then
/// `parse_wildatom_defs(lines)`).
pub fn parse_wildatom_defs(lines: &[&str]) -> WildatomMap {
    let mut wildatom_map: WildatomMap = WildatomMap::new();

    for raw_line in lines {
        let line = raw_line.trim();
        if !line.starts_with("WILDATOM") {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let symbol = parts[1].to_string();
            let elements: Vec<String> = parts[2..].iter().map(|s| s.to_string()).collect();
            wildatom_map.insert(symbol, elements);
        }
    }

    wildatom_map
}

/// Parse a single already-tokenized ATD rule body (post `ATD`-prefix-strip,
/// post `&`-suffix-strip, whitespace-split) into a `Gaff2Rule`.
///
/// Returns `None` on any field parse failure, mirroring Python's
/// `except (ValueError, IndexError): continue` (gaff2.py lines 796-858):
/// a malformed rule is silently dropped, not surfaced as an error, and
/// *no partial rule* is ever emitted.
///
/// `parts.len() >= 3` must already be checked by the caller (mirrors
/// gaff2.py lines 793-794 — the minimum valid row is `<atom_type> <residue>
/// <atomic_num>`; this was previously `< 4`, which silently dropped every
/// bare single-element rule — roughly half the file, mostly halogens/metals/
/// late-periodic-table fallback types).
fn parse_rule_fields(parts: &[&str]) -> Option<Gaff2Rule> {
    let atom_type = parts[0].to_string();
    // Python: `residue = parts[1] if parts[1] != "*" else "*"` — both
    // branches produce parts[1] either way; preserved verbatim (as a no-op
    // ternary) rather than simplified, per the behavior-preservation mandate
    // for this port.
    let residue = parts[1].to_string();
    let atomic_num: i64 = parts[2].parse().ok()?;

    let mut num_attached: Option<i64> = None;
    let mut num_h: Option<i64> = None;
    let mut h_ew_count: Option<i64> = None;

    let mut idx = 3;
    if idx < parts.len() && parts[idx] != "*" {
        // f5: default is None (wildcard), NOT 0 — do not coerce a missing/`*`
        // field to zero.
        num_attached = Some(parts[idx].parse().ok()?);
    }
    idx += 1;

    if idx < parts.len() && parts[idx] != "*" {
        // f6: same None-not-zero default as f5.
        num_h = Some(parts[idx].parse().ok()?);
    }
    idx += 1;

    let mut remaining: String = if idx < parts.len() {
        parts[idx..].join(" ")
    } else {
        String::new()
    };
    remaining = remaining.trim_start().to_string();

    // f7 (h_ew_count): H-only field. A bare integer, or "*" for every
    // non-H row. Consume exactly one token positionally rather than
    // silently discarding it (gaff2.py lines 817-826).
    if let Some(rest) = remaining.strip_prefix('*') {
        remaining = rest.trim().to_string();
    } else {
        let digit_len = remaining.chars().take_while(|c| c.is_ascii_digit()).count();
        if digit_len > 0 {
            let (num_str, rest) = remaining.split_at(digit_len);
            h_ew_count = Some(num_str.parse().ok()?);
            remaining = rest.trim().to_string();
        }
        // else: remaining doesn't start with a digit (and isn't "*") — leave
        // h_ew_count as None and remaining untouched, matching Python's
        // `re.match` returning no match (no branch mutates anything then).
    }

    // f8: bracketed atomic-property expression, "[...]" or "*" or absent.
    let mut atomic_prop_raw: Option<String> = None;
    if let Some(after_bracket) = remaining.strip_prefix('[') {
        if let Some(close_rel) = after_bracket.find(']') {
            atomic_prop_raw = Some(after_bracket[..close_rel].to_string());
            remaining = after_bracket[close_rel + 1..].trim().to_string();
        }
        // else: unmatched "[" — leave remaining untouched (still starting
        // with "["), atomic_prop_raw stays None. Matches Python's
        // `if close != -1:` guard, which has no else-branch mutation.
    } else if let Some(rest) = remaining.strip_prefix('*') {
        remaining = rest.trim().to_string();
    }

    // f9: parenthesized chemical-environment neighbor pattern, "(...)" or
    // "*" or absent (end of line before "&"). There is no true f10 —
    // anything left after this is a malformed line, not a further field
    // (gaff2.py lines 838-840) — so, unlike f7/f8, there is no "*" branch
    // here and `remaining` is not consumed any further.
    let chem_env_raw: Option<String> = if remaining.starts_with('(') {
        Some(remaining)
    } else {
        None
    };

    Some(Gaff2Rule {
        atom_type,
        residue,
        atomic_num,
        num_attached,
        num_h,
        h_ew_count,
        atomic_prop_raw,
        chem_env_raw,
    })
}

/// Parse ATOMTYPE_GFF2.DEF content into rules and a WILDATOM map.
///
/// Ports `parse_gaff2_rules` (gaff2.py lines 751-860). Unlike the Python
/// version, this takes already-loaded DEF *content* rather than a path —
/// file IO / embedding is `rules_loader`'s job, this module is pure parsing.
///
/// Returns `Result` (always `Ok` today) rather than the bare tuple: given
/// content, this function can't actually fail — every per-line failure is
/// caught and the line is silently skipped (see `parse_rule_fields`'s doc
/// comment), exactly like Python, which never raises to its caller either.
/// The `Result` wrapper exists purely to match `rules_loader`'s
/// `impl FnOnce(&str) -> Result<(Vec<Gaff2Rule>, WildatomMap), String>`
/// injection point (its `read_and_parse`/`get_default_rules_with` helpers,
/// which this function is wired into directly) — an infallible signature
/// here would not compose with that already-established caller contract.
///
/// # CRITICAL: declaration order is load-bearing
///
/// Rule precedence at match time is first-match-in-file-order (see
/// ATOMTYPE_GFF2.DEF's own "defination order is crucial" comment, line
/// 1176). The returned `Vec<Gaff2Rule>` MUST preserve DEF-file declaration
/// order — it does, because `Vec::push` inside a single forward pass over
/// `content`'s lines is inherently order-preserving. Do not sort, dedupe
/// via a hash-keyed structure, or otherwise reorder this vector.
pub fn parse_gaff2_rules(content: &str) -> Result<(Vec<Gaff2Rule>, WildatomMap), String> {
    let lines: Vec<&str> = content.split('\n').collect();

    let wildatom_map = parse_wildatom_defs(&lines);

    let mut rules: Vec<Gaff2Rule> = Vec::new();
    let mut in_definition = false;

    for raw_line in &lines {
        let line = raw_line.trim();

        // Deliberately matches the DEF file's own "Defination" spelling
        // (see the file header) case-insensitively; do not "fix" the typo
        // in the match string, it must track the source file's own wording.
        if line.to_lowercase().contains("efination begin") {
            in_definition = true;
            continue;
        }

        if !in_definition || !line.starts_with("ATD") {
            continue;
        }

        if !line.contains('&') {
            continue;
        }

        let line = line.strip_suffix('&').unwrap_or(line).trim();
        let line = if let Some(rest) = line.strip_prefix("ATD") {
            rest.trim()
        } else {
            line
        };

        let parts: Vec<&str> = line.split_whitespace().collect();
        // Minimum valid row is "<atom_type> <residue> <atomic_num>" (3
        // tokens) — bare single-element rules like "ATD f * 9 &" have no
        // further fields. (Previously "< 4", which silently dropped every
        // such rule — see parse_rule_fields' doc comment.)
        if parts.len() < 3 {
            continue;
        }

        if let Some(rule) = parse_rule_fields(&parts) {
            rules.push(rule);
        }
        // else: mirrors Python's `except (ValueError, IndexError): continue`.
    }

    Ok((rules, wildatom_map))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // parse_wildatom_defs
    // ------------------------------------------------------------------

    #[test]
    fn test_wildatom_defs_space_separated_not_comma_joined() {
        // Regression fixture for the "space-separated, not comma-joined"
        // bug: a naive `parts[2]`-with-comma-split would previously
        // silently capture only "C".
        let lines = ["WILDATOM XX C N O S P"];
        let map = parse_wildatom_defs(&lines);
        assert_eq!(
            map.get("XX").cloned(),
            Some(vec![
                "C".to_string(),
                "N".to_string(),
                "O".to_string(),
                "S".to_string(),
                "P".to_string(),
            ])
        );
    }

    #[test]
    fn test_wildatom_defs_multiple_symbols() {
        let lines = [
            "WILDATOM XX C N O S P",
            "WILDATOM XA O S",
            "WILDATOM XB N P",
            "WILDATOM XC F Cl Br I",
            "WILDATOM XD S P",
        ];
        let map = parse_wildatom_defs(&lines);
        assert_eq!(map.len(), 5);
        assert_eq!(
            map.get("XC").cloned(),
            Some(vec![
                "F".to_string(),
                "Cl".to_string(),
                "Br".to_string(),
                "I".to_string(),
            ])
        );
    }

    #[test]
    fn test_wildatom_defs_ignores_non_wildatom_lines() {
        let lines = ["// a comment", "ATD c3 * 6 4 &", "WILDATOM XX C N"];
        let map = parse_wildatom_defs(&lines);
        assert_eq!(map.len(), 1);
        assert!(map.contains_key("XX"));
    }

    #[test]
    fn test_wildatom_defs_duplicate_symbol_last_wins() {
        // Matches Python dict-assignment overwrite semantics.
        let lines = ["WILDATOM XX C N", "WILDATOM XX O S P"];
        let map = parse_wildatom_defs(&lines);
        assert_eq!(
            map.get("XX").cloned(),
            Some(vec!["O".to_string(), "S".to_string(), "P".to_string()])
        );
    }

    // ------------------------------------------------------------------
    // parse_gaff2_rules — structural / precedence-critical behavior
    // ------------------------------------------------------------------

    fn wrap_def(body: &str) -> String {
        format!(
            "Defination begin\n\
             --------------------------------------------------------------------------------------------\n\
             {body}\n"
        )
    }

    #[test]
    fn test_minimum_three_token_rule_parses() {
        // Regression fixture for the "< 4" -> "< 3" fix: a bare
        // single-element rule (atom_type, residue, atomic_num only, no f5+)
        // must parse, not be dropped.
        let content = wrap_def("ATD  f     *   9   &");
        let (rules, _) = parse_gaff2_rules(&content).unwrap();
        assert_eq!(rules.len(), 1);
        let rule = &rules[0];
        assert_eq!(rule.atom_type, "f");
        assert_eq!(rule.residue, "*");
        assert_eq!(rule.atomic_num, 9);
        assert_eq!(rule.num_attached, None);
        assert_eq!(rule.num_h, None);
        assert_eq!(rule.h_ew_count, None);
        assert_eq!(rule.atomic_prop_raw, None);
        assert_eq!(rule.chem_env_raw, None);
    }

    #[test]
    fn test_two_token_rule_is_dropped() {
        // Below the 3-token minimum -- must NOT parse (unlike the 3-token
        // case above).
        let content = wrap_def("ATD  DU    &");
        let (rules, _) = parse_gaff2_rules(&content).unwrap();
        assert!(rules.is_empty());
    }

    #[test]
    fn test_num_attached_and_num_h_default_to_none_not_zero() {
        // f5 present ("4"), f6 absent (wildcard "*") -- num_h must be None,
        // not coerced to 0.
        let content = wrap_def("ATD  c3    *   6   4   &");
        let (rules, _) = parse_gaff2_rules(&content).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].num_attached, Some(4));
        assert_eq!(rules[0].num_h, None);
    }

    #[test]
    fn test_h_ew_count_consumed_positionally_not_discarded() {
        // Real DEF fixture (line ~84): h3 = H atom, 1 attachment, wildcard
        // num_h, 3 electron-withdrawing neighbors (f7), chem_env "(C4)".
        let content = wrap_def("ATD  h3    *   1   1   *   3   *       \t\t(C4)     &");
        let (rules, _) = parse_gaff2_rules(&content).unwrap();
        assert_eq!(rules.len(), 1);
        let rule = &rules[0];
        assert_eq!(rule.atom_type, "h3");
        assert_eq!(rule.atomic_num, 1);
        assert_eq!(rule.num_attached, Some(1));
        assert_eq!(rule.num_h, None);
        assert_eq!(rule.h_ew_count, Some(3));
        assert_eq!(rule.atomic_prop_raw, None);
        assert_eq!(rule.chem_env_raw.as_deref(), Some("(C4)"));
    }

    #[test]
    fn test_f8_and_f9_both_present_f9_may_contain_nested_bracket() {
        // Real DEF fixture (cp): f8 is "[AR1,1RG6]", f9 is
        // "(XX[AR1],XX[AR1],XX[AR1])" -- the "[" inside f9's neighbor spec
        // must NOT be mistaken for a second f8 field or truncate chem_env_raw.
        let content =
            wrap_def("ATD  cp    *   6   3   *   *   [AR1,1RG6]       (XX[AR1],XX[AR1],XX[AR1]) &");
        let (rules, _) = parse_gaff2_rules(&content).unwrap();
        assert_eq!(rules.len(), 1);
        let rule = &rules[0];
        assert_eq!(rule.atom_type, "cp");
        assert_eq!(rule.atomic_prop_raw.as_deref(), Some("AR1,1RG6"));
        assert_eq!(
            rule.chem_env_raw.as_deref(),
            Some("(XX[AR1],XX[AR1],XX[AR1])")
        );
    }

    #[test]
    fn test_declaration_order_is_preserved() {
        let content = wrap_def(
            "ATD  cx    *   6   4   *   *   [RG3]            &\n\
             ATD  cy    *   6   4   *   *   [RG4]            &\n\
             ATD  c5    *   6   4   *   *   [RG5]            &\n\
             ATD  c6    *   6   4   *   *   [RG6]            &",
        );
        let (rules, _) = parse_gaff2_rules(&content).unwrap();
        let order: Vec<&str> = rules.iter().map(|r| r.atom_type.as_str()).collect();
        assert_eq!(order, vec!["cx", "cy", "c5", "c6"]);
    }

    #[test]
    fn test_lines_before_defination_begin_are_ignored() {
        let content = "// header comment mentioning ATD but not inside the rule block\n\
             ATD  bogus * 6 4 &\n\
             Defination begin\n\
             --------------------------------------------------------------------------------------------\n\
             ATD  c3    *   6   4   &\n";
        let (rules, _) = parse_gaff2_rules(content).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].atom_type, "c3");
    }

    #[test]
    fn test_defination_marker_is_case_insensitive() {
        let content = "DEFINATION BEGIN\nATD  c3 * 6 4 &\n";
        let (rules, _) = parse_gaff2_rules(content).unwrap();
        assert_eq!(rules.len(), 1);
    }

    #[test]
    fn test_line_without_ampersand_is_skipped() {
        let content = wrap_def("ATD  c3    *   6   4");
        let (rules, _) = parse_gaff2_rules(&content).unwrap();
        assert!(rules.is_empty());
    }

    #[test]
    fn test_non_numeric_atomic_num_drops_whole_rule() {
        // Mirrors Python's except-continue: a ValueError anywhere in the
        // required-field parse aborts the whole rule, not just that field.
        let content = wrap_def("ATD  bogus * NOTANUM 4 &");
        let (rules, _) = parse_gaff2_rules(&content).unwrap();
        assert!(rules.is_empty());
    }

    // ------------------------------------------------------------------
    // Full real DEF file smoke test — mirrors Python's
    // test_def_file_parses_without_dropping_fields (test_gaff2_golden.py).
    // ------------------------------------------------------------------

    static ATOMTYPE_GFF2_DEF: &str =
        include_str!("../../../src/proxide/assets/gaff/dat/ATOMTYPE_GFF2.DEF");

    #[test]
    fn test_real_def_file_parses_317_rules() {
        let (rules, wildatom_map) = parse_gaff2_rules(ATOMTYPE_GFF2_DEF).unwrap();

        // Pinned to the actual count from the vendored DEF file (317 of 318
        // raw ATD rows -- the remaining one is the bare "ATD DU &" sentinel,
        // which has no residue/atomic_num fields at all and is intentionally
        // not a parseable rule). Update deliberately if the asset is ever
        // regenerated/upgraded, not to silence a regression.
        assert_eq!(
            rules.len(),
            317,
            "expected 317 parsed rules, got {}",
            rules.len()
        );

        let xx: std::collections::HashSet<&str> =
            wildatom_map["XX"].iter().map(|s| s.as_str()).collect();
        assert_eq!(
            xx,
            std::collections::HashSet::from(["C", "N", "O", "S", "P"])
        );
    }

    #[test]
    fn test_real_def_file_no_dropped_bracket_or_paren() {
        // Re-derive each rule's raw line and confirm no bracket/paren was
        // silently dropped -- same audit Python's
        // test_def_file_parses_without_dropping_fields performs.
        let (rules, _) = parse_gaff2_rules(ATOMTYPE_GFF2_DEF).unwrap();

        let atd_lines: Vec<&str> = ATOMTYPE_GFF2_DEF
            .split('\n')
            .map(|l| l.trim())
            .filter(|l| l.starts_with("ATD") && l.contains('&'))
            .collect();
        assert!(atd_lines.len() >= rules.len());

        let mut dropped_bracket: Vec<&str> = Vec::new();
        let mut dropped_paren: Vec<&str> = Vec::new();
        for (rule, raw) in rules.iter().zip(atd_lines.iter()) {
            // An f9 neighbor spec can itself contain a "[...]" bracket (e.g.
            // "(XX[AR1])"), so only a "[" appearing BEFORE any "(" is really
            // in the f8 position.
            let before_paren = raw.split('(').next().unwrap_or(raw);
            if before_paren.contains('[') && rule.atomic_prop_raw.is_none() {
                dropped_bracket.push(rule.atom_type.as_str());
            }
            if raw.contains('(') && rule.chem_env_raw.is_none() {
                dropped_paren.push(rule.atom_type.as_str());
            }
        }
        assert!(
            dropped_bracket.is_empty(),
            "rules with a raw f8 '[' but no parsed atomic_prop_raw: {dropped_bracket:?}"
        );
        assert!(
            dropped_paren.is_empty(),
            "rules with a raw '(' but no parsed chem_env_raw: {dropped_paren:?}"
        );
    }
}
