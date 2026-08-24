//! Atomic property parsing and matching — f8 bracket grammar.
//!
//! Parses and evaluates the f8 field of GAFF2 rules: a bracket-delimited
//! grammar of atomic properties, e.g. `[sb,db,AR2]`, `[AR1,1RG6]`, `[2DL]`.
//!
//! Ports gaff2.py's f8 grammar section verbatim:
//! - `PropToken` (55-57), `AtomicPropExpr` (61-63)
//! - `_tokenize_prop_list` (66-71), `parse_atomic_prop` (74-88)
//! - `atomic_prop_matches` (91-97)
//! - `_matches_aromaticity_token` (598-607), `_prop_token_matches` (610-632)
//!
//! Per ATOMTYPE_GFF2.DEF's own footer ("Field descriptions" / "Specific
//! symbols" / "Predefined words" / "Miscellaneous"): comma = AND, "." = OR
//! (ring/aromaticity descriptions); a leading digit is an exact-count prefix
//! (e.g. "2DL" = exactly 2 delocalized bonds, "1RG6" = exactly 1 six-membered
//! ring); uppercase bond words (SB/DB/TB/AB/DL) are exact bond-type identity,
//! lowercase (sb/db/tb) are the inclusive unions the footer describes ("sb"
//! includes aromatic-single + delocalized, "db" includes aromatic-double).

use crate::atom_bond_facts::AtomBondFacts;

/// A single parsed f8 token, e.g. `RG3`, `AR1`, `SB`, `sb`, `NR`, `DL`.
///
/// Ports `PropToken` (gaff2.py:55-57).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropToken {
    pub word: String,
    /// Leading digit prefix, e.g. `Some(2)` for "2DL". `None` = presence-only
    /// (match if count > 0).
    pub count: Option<u32>,
}

/// Top-level combinator for a parsed f8 bracket body.
///
/// Ports `AtomicPropExpr` (gaff2.py:61-63).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropOp {
    /// comma-separated: all tokens must match.
    And,
    /// dot-separated: any token must match.
    Or,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicPropExpr {
    pub op: PropOp,
    pub tokens: Vec<PropToken>,
}

/// Split a bracket body on its top-level separator, returning `(op, raw_tokens)`.
///
/// Ports `_tokenize_prop_list` (gaff2.py:66-71). A "." anywhere in the body
/// selects OR semantics (dot-separated); otherwise the body is comma-separated
/// AND. Empty entries (from leading/trailing/doubled separators) are dropped,
/// matching the Python `if t.strip()` filter.
fn tokenize_prop_list(raw: &str) -> (PropOp, Vec<String>) {
    let raw = raw.trim();
    if raw.contains('.') {
        let tokens = raw
            .split('.')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        (PropOp::Or, tokens)
    } else {
        let tokens = raw
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        (PropOp::And, tokens)
    }
}

/// Match one raw token against the `^(\d*)([A-Za-z]+\d*)$` shape used by
/// gaff2.py's `_PROP_TOKEN_RE`.
///
/// Because the leading `\d*` and the following `[A-Za-z]+` are disjoint
/// character classes, there is no backtracking ambiguity to reproduce: the
/// leading digit run (if any) is exactly the maximal run of digit characters
/// from the start of the token, and everything after it must be a non-empty
/// run of letters optionally followed by a run of digits, with nothing left
/// over. Returns `None` on any other shape (mirrors the Python `if not m:
/// continue` skip-on-no-match behavior).
fn match_prop_token(raw_tok: &str) -> Option<PropToken> {
    let chars: Vec<char> = raw_tok.chars().collect();

    let mut i = 0;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    let count_chars = &chars[..i];
    let rest_chars = &chars[i..];

    let mut j = 0;
    while j < rest_chars.len() && rest_chars[j].is_ascii_alphabetic() {
        j += 1;
    }
    if j == 0 {
        // No letters at all (e.g. rest is empty, or starts with a non-letter
        // that isn't a digit either) -- word group requires >= 1 letter.
        return None;
    }
    if !rest_chars[j..].iter().all(|c| c.is_ascii_digit()) {
        // Anything after the letter run that isn't itself all digits (e.g.
        // more letters, or a stray symbol) doesn't match `\d*$`.
        return None;
    }

    let count = if count_chars.is_empty() {
        None
    } else {
        // count_chars is all-ASCII-digit by construction; parse failure would
        // only occur on overflow, which the DEF grammar never produces.
        count_chars.iter().collect::<String>().parse::<u32>().ok()
    };
    let word: String = rest_chars.iter().collect();
    Some(PropToken { word, count })
}

/// Parse an f8 bracket body (without the surrounding `[...]`) into an AST.
///
/// Ports `parse_atomic_prop` (gaff2.py:74-88). Returns `None` for an empty or
/// wildcard (`*`) body, or if no raw token matched the token shape (empty
/// token list after filtering unmatched tokens).
pub fn parse_atomic_prop(raw: &str) -> Option<AtomicPropExpr> {
    let raw = raw.trim();
    if raw.is_empty() || raw == "*" {
        return None;
    }

    let (op, raw_tokens) = tokenize_prop_list(raw);
    let tokens: Vec<PropToken> = raw_tokens
        .iter()
        .filter_map(|t| match_prop_token(t))
        .collect();

    if tokens.is_empty() {
        None
    } else {
        Some(AtomicPropExpr { op, tokens })
    }
}

// `_matches_aromaticity_token` (gaff2.py:598-607) is ported as
// `crate::atom_bond_facts::matches_aromaticity_token`, called directly
// below -- this module carries no private copy (a prior one was removed as
// unreachable dead code; see that function's doc comment for the AR1/AR2/
// AR3/AR4/AR5 matching rules themselves).

/// Evaluate one parsed token against precomputed per-atom facts.
///
/// Ports `_prop_token_matches` (gaff2.py:610-632). Branch order matches the
/// Python exactly (each branch returns early; there is no fallthrough once a
/// branch's guard is taken).
fn prop_token_matches(tok: &PropToken, facts: &AtomBondFacts) -> bool {
    let word = tok.word.as_str();

    if word == "NR" {
        return !facts.in_ring;
    }
    if word == "RG" {
        return facts.in_ring;
    }
    if let Some(size_str) = word.strip_prefix("RG") {
        if !size_str.is_empty() && size_str.chars().all(|c| c.is_ascii_digit()) {
            // Grammar never produces a size large enough to overflow usize;
            // fall through to no-match (rather than panic) if it somehow did.
            if let Ok(size) = size_str.parse::<usize>() {
                let count = facts.ring_counts_by_size.get(&size).copied().unwrap_or(0);
                return match tok.count {
                    Some(c) => count == c as usize,
                    None => count > 0,
                };
            }
        }
    }
    if let Some(digits) = word.strip_prefix("AR") {
        if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
            return crate::atom_bond_facts::matches_aromaticity_token(
                word,
                facts.aromaticity_class.as_deref(),
            );
        }
    }

    if let Some(&count) = facts.bond_counts.get(word) {
        return match tok.count {
            Some(c) => count == c as usize,
            None => count > 0,
        };
    }

    false
}

/// Evaluate a parsed f8 expression against one atom's precomputed bond/ring
/// facts.
///
/// Ports `atomic_prop_matches` (gaff2.py:91-97). `expr = None` (unbracketed
/// wildcard field) always matches.
pub fn atomic_prop_matches(expr: Option<&AtomicPropExpr>, facts: &AtomBondFacts) -> bool {
    let Some(expr) = expr else {
        return true;
    };

    let results: Vec<bool> = expr
        .tokens
        .iter()
        .map(|tok| prop_token_matches(tok, facts))
        .collect();

    match expr.op {
        PropOp::Or => results.iter().any(|&b| b),
        PropOp::And => results.iter().all(|&b| b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Ports `TestGaff2Grammar.test_parse_atomic_prop_and_list`
    /// (test_gaff2_golden.py:702-709).
    #[test]
    fn test_parse_atomic_prop_and_list() {
        let expr = parse_atomic_prop("sb,db,AR2").unwrap();
        assert_eq!(expr.op, PropOp::And);
        let got: Vec<(&str, Option<u32>)> = expr
            .tokens
            .iter()
            .map(|t| (t.word.as_str(), t.count))
            .collect();
        assert_eq!(got, vec![("sb", None), ("db", None), ("AR2", None)]);
    }

    /// Ports `TestGaff2Grammar.test_parse_atomic_prop_or_list`
    /// (test_gaff2_golden.py:711-716).
    #[test]
    fn test_parse_atomic_prop_or_list() {
        let expr = parse_atomic_prop("AR1.AR2.AR3").unwrap();
        assert_eq!(expr.op, PropOp::Or);
        let words: Vec<&str> = expr.tokens.iter().map(|t| t.word.as_str()).collect();
        assert_eq!(words, vec!["AR1", "AR2", "AR3"]);
    }

    /// Ports `TestGaff2Grammar.test_parse_atomic_prop_exact_counts`
    /// (test_gaff2_golden.py:718-727).
    #[test]
    fn test_parse_atomic_prop_exact_counts() {
        let expr = parse_atomic_prop("2DL").unwrap();
        let got: Vec<(&str, Option<u32>)> = expr
            .tokens
            .iter()
            .map(|t| (t.word.as_str(), t.count))
            .collect();
        assert_eq!(got, vec![("DL", Some(2))]);

        let expr = parse_atomic_prop("1DB,0DL").unwrap();
        let got: Vec<(&str, Option<u32>)> = expr
            .tokens
            .iter()
            .map(|t| (t.word.as_str(), t.count))
            .collect();
        assert_eq!(got, vec![("DB", Some(1)), ("DL", Some(0))]);

        let expr = parse_atomic_prop("3sb").unwrap();
        let got: Vec<(&str, Option<u32>)> = expr
            .tokens
            .iter()
            .map(|t| (t.word.as_str(), t.count))
            .collect();
        assert_eq!(got, vec![("sb", Some(3))]);

        let expr = parse_atomic_prop("AR1,1RG6").unwrap();
        let got: Vec<(&str, Option<u32>)> = expr
            .tokens
            .iter()
            .map(|t| (t.word.as_str(), t.count))
            .collect();
        assert_eq!(got, vec![("AR1", None), ("RG6", Some(1))]);
    }

    fn facts_with_bond_counts(overrides: &[(&str, usize)]) -> AtomBondFacts {
        let mut bond_counts: HashMap<String, usize> =
            ["SB", "DB", "TB", "AB", "DL", "sb", "db", "tb"]
                .iter()
                .map(|k| (k.to_string(), 0))
                .collect();
        for (k, v) in overrides {
            bond_counts.insert(k.to_string(), *v);
        }
        AtomBondFacts {
            in_ring: false,
            ring_counts_by_size: HashMap::new(),
            aromaticity_class: None,
            bond_counts,
        }
    }

    /// Ports `TestGaff2Grammar.test_atomic_prop_matches_2dl_and_3sb_exact_counts`
    /// (test_gaff2_golden.py:770-806).
    #[test]
    fn test_atomic_prop_matches_2dl_and_3sb_exact_counts() {
        let two_dl = parse_atomic_prop("2DL").unwrap();
        assert!(atomic_prop_matches(
            Some(&two_dl),
            &facts_with_bond_counts(&[("DL", 2), ("AB", 2)])
        ));
        assert!(!atomic_prop_matches(
            Some(&two_dl),
            &facts_with_bond_counts(&[("DL", 1), ("AB", 1)])
        ));
        assert!(!atomic_prop_matches(
            Some(&two_dl),
            &facts_with_bond_counts(&[("DL", 3), ("AB", 3)])
        ));

        let three_sb = parse_atomic_prop("3sb").unwrap();
        assert!(atomic_prop_matches(
            Some(&three_sb),
            &facts_with_bond_counts(&[("sb", 3)])
        ));
        assert!(!atomic_prop_matches(
            Some(&three_sb),
            &facts_with_bond_counts(&[("sb", 2)])
        ));
        assert!(!atomic_prop_matches(
            Some(&three_sb),
            &facts_with_bond_counts(&[("sb", 4)])
        ));
    }

    /// `expr = None` (unbracketed wildcard) always matches, whatever the
    /// facts. No direct Python unit test exercises this in isolation, but
    /// it's exercised transitively wherever an f8 field is `*` or empty; the
    /// behavior itself is `parse_atomic_prop`/`atomic_prop_matches`'s
    /// explicit contract (gaff2.py:78-79, 93-94).
    #[test]
    fn test_atomic_prop_matches_none_expr_always_true() {
        assert!(parse_atomic_prop("").is_none());
        assert!(parse_atomic_prop("*").is_none());
        assert!(atomic_prop_matches(None, &facts_with_bond_counts(&[])));
    }

    /// AR1/AR2/AR3/AR4/AR5 matching via `_matches_aromaticity_token`
    /// (gaff2.py:598-607). Not covered by a standalone Python unit test, but
    /// flagged in the port's recon notes as the one non-obvious branch in
    /// this module: AR2 and AR3 both collapse onto the single "AR23" class,
    /// while AR1/AR4/AR5 are each exact matches against their own class name.
    #[test]
    fn test_aromaticity_token_matching() {
        fn facts_with_aromaticity(class: Option<&str>) -> AtomBondFacts {
            AtomBondFacts {
                in_ring: true,
                ring_counts_by_size: HashMap::new(),
                aromaticity_class: class.map(|s| s.to_string()),
                bond_counts: HashMap::new(),
            }
        }

        let ar1 = parse_atomic_prop("AR1").unwrap();
        let ar2 = parse_atomic_prop("AR2").unwrap();
        let ar3 = parse_atomic_prop("AR3").unwrap();
        let ar4 = parse_atomic_prop("AR4").unwrap();
        let ar5 = parse_atomic_prop("AR5").unwrap();

        // AR1 class matches only the AR1 token.
        let f = facts_with_aromaticity(Some("AR1"));
        assert!(atomic_prop_matches(Some(&ar1), &f));
        assert!(!atomic_prop_matches(Some(&ar2), &f));
        assert!(!atomic_prop_matches(Some(&ar3), &f));

        // AR23 class matches both AR2 and AR3 tokens, not AR1.
        let f = facts_with_aromaticity(Some("AR23"));
        assert!(!atomic_prop_matches(Some(&ar1), &f));
        assert!(atomic_prop_matches(Some(&ar2), &f));
        assert!(atomic_prop_matches(Some(&ar3), &f));

        // AR4/AR5 are exact, distinct classes.
        let f = facts_with_aromaticity(Some("AR4"));
        assert!(atomic_prop_matches(Some(&ar4), &f));
        assert!(!atomic_prop_matches(Some(&ar5), &f));
        let f = facts_with_aromaticity(Some("AR5"));
        assert!(atomic_prop_matches(Some(&ar5), &f));
        assert!(!atomic_prop_matches(Some(&ar4), &f));

        // No aromaticity class at all never matches any AR token.
        let f = facts_with_aromaticity(None);
        assert!(!atomic_prop_matches(Some(&ar1), &f));
        assert!(!atomic_prop_matches(Some(&ar2), &f));
    }

    /// `NR`/`RG`/`RGn` ring-membership and ring-count tokens (gaff2.py:614-621).
    #[test]
    fn test_ring_membership_and_count_tokens() {
        let nr = parse_atomic_prop("NR").unwrap();
        let rg = parse_atomic_prop("RG").unwrap();
        let rg6 = parse_atomic_prop("RG6").unwrap();
        let one_rg6 = parse_atomic_prop("1RG6").unwrap();

        let mut ring_counts = HashMap::new();
        ring_counts.insert(6usize, 2usize);
        let in_ring = AtomBondFacts {
            in_ring: true,
            ring_counts_by_size: ring_counts,
            aromaticity_class: None,
            bond_counts: HashMap::new(),
        };
        let not_in_ring = AtomBondFacts {
            in_ring: false,
            ring_counts_by_size: HashMap::new(),
            aromaticity_class: None,
            bond_counts: HashMap::new(),
        };

        assert!(!atomic_prop_matches(Some(&nr), &in_ring));
        assert!(atomic_prop_matches(Some(&nr), &not_in_ring));

        assert!(atomic_prop_matches(Some(&rg), &in_ring));
        assert!(!atomic_prop_matches(Some(&rg), &not_in_ring));

        // Presence-only: any nonzero count of 6-membered rings matches.
        assert!(atomic_prop_matches(Some(&rg6), &in_ring));
        assert!(!atomic_prop_matches(Some(&rg6), &not_in_ring));

        // Exact count: "1RG6" requires EXACTLY one 6-membered ring; this
        // atom is in two, so it does not match.
        assert!(!atomic_prop_matches(Some(&one_rg6), &in_ring));
    }
}
