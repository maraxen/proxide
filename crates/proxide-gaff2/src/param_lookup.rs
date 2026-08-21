//! Bond and angle parameter lookup with bidirectional matching and
//! substitution fallback.
//!
//! Ports `_lookup_bond_params` (gaff2.py:1448-1461), `_lookup_angle_params`
//! (gaff2.py:1464-1477), and the `_BOND_TYPE_SUB` table (gaff2.py:1413-1418)
//! from the ciMIST reference implementation, verbatim.
//!
//! # Lookup order (load-bearing — do not reorder)
//!
//! For a bond `(ti, tj)`:
//! 1. Try `(ti, tj)` in the bond table.
//! 2. Try `(tj, ti)` in the bond table.
//! 3. Only if *both* miss: substitute both types via [`bond_type_sub`],
//!    then try `(ti_sub, tj_sub)`.
//! 4. Try `(tj_sub, ti_sub)`.
//! 5. If all four probes miss, return [`DEFAULT_BOND_PARAMS`].
//!
//! The critical behavior preserved from Python: substitution is applied
//! **only on a full miss of the unsubstituted forward+reverse pair**, never
//! on the first attempt. A table that has both the exact type pair and the
//! substituted-generic pair present must resolve to the exact pair.
//! Angles follow the identical five-step shape with a third type `tj` held
//! fixed in the center position (the reversed probe swaps `ti`/`tk` only,
//! matching Python's `(tk, tj, ti)`).
//!
//! # Why `HashMap`, not an order-preserving map
//!
//! Unlike other modules in this crate (see `mol.rs`, `alternation.rs`),
//! this module's Python source never *iterates* `params['bonds']` /
//! `params['angles']` — it only ever performs `in`/`[]` key lookups against
//! explicit, hand-ordered probe lists (`[(ti, tj), (tj, ti)]`, etc.). Dict
//! insertion order has no observable effect on `_lookup_bond_params` /
//! `_lookup_angle_params`'s behavior, so a plain `HashMap` is behavior-
//! preserving here. (Whoever builds the `BondParams`/`AngleParams` table
//! from the GAFF2 `.dat` file — a separate, not-yet-ported concern — should
//! re-check this reasoning if that construction step relies on last-write-
//! wins semantics over a duplicate-keyed source.)

use std::collections::HashMap;

/// Key into a [`BondParams`] table: `(type_i, type_j)`.
pub type BondKey = (String, String);
/// Key into an [`AngleParams`] table: `(type_i, type_j, type_k)`.
pub type AngleKey = (String, String, String);

/// Bond parameter table: `(type_i, type_j) -> (kb, r0)`.
///
/// `kb` is the force constant (kcal/mol/Å²), `r0` the equilibrium bond
/// length (Å) — units as stored by the GAFF2 `.dat` parser, unconverted.
pub type BondParams = HashMap<BondKey, (f64, f64)>;

/// Angle parameter table: `(type_i, type_j, type_k) -> (kt, t0)`.
///
/// `kt` is the force constant (kcal/mol/rad²), `t0` the equilibrium angle
/// (degrees) — units as stored by the GAFF2 `.dat` parser, unconverted.
pub type AngleParams = HashMap<AngleKey, (f64, f64)>;

/// Value returned by [`lookup_bond_params`] when no entry is found under
/// any of the four (direct/reversed × unsubstituted/substituted) probes.
///
/// Ported verbatim from `_lookup_bond_params`'s `return (0.0, 1.50)`
/// (gaff2.py:1461).
pub const DEFAULT_BOND_PARAMS: (f64, f64) = (0.0, 1.50);

/// Value returned by [`lookup_angle_params`] when no entry is found under
/// any of the four probes.
///
/// Ported verbatim from `_lookup_angle_params`'s `return (0.0, 120.0)`
/// (gaff2.py:1477).
pub const DEFAULT_ANGLE_PARAMS: (f64, f64) = (0.0, 120.0);

/// GAFF2 bond-type substitution table used as a parameter-lookup fallback.
///
/// Ported verbatim from `_BOND_TYPE_SUB` (gaff2.py:1413-1418). Collapses
/// several conjugated/exotic sp3/sp2/sp carbon subtypes onto a generic
/// representative type, so that a bond/angle table miss on the specific
/// type can still resolve against a generic `.dat` entry:
///
/// - `cx`, `cy`, `c5`, `c6` -> `c3`
/// - `cs`, `cz`, `ca`, `cc`, `cd`, `ce`, `cf`, `cp`, `cq` -> `c2`
/// - `cg`, `ch` -> `c1`
/// - anything else -> itself (Python's `dict.get(t, t)` default).
///
/// This table is intentionally *not* used to "fix" any residual mismatch
/// against real AmberTools output — it is a byte-for-byte port of the
/// Python constant, including whatever known mismatches that table's
/// exact membership produces on the full geostd corpus.
pub fn bond_type_sub(t: &str) -> &str {
    match t {
        "cx" | "cy" | "c5" | "c6" => "c3",
        "cs" | "cz" | "ca" | "cc" | "cd" | "ce" | "cf" | "cp" | "cq" => "c2",
        "cg" | "ch" => "c1",
        other => other,
    }
}

/// Look up `(kb, r0)` for a bond between atom types `ti` and `tj`.
///
/// Ports `_lookup_bond_params` (gaff2.py:1448-1461) exactly:
/// 1. Try `(ti, tj)`, then `(tj, ti)`, against `bonds` (no substitution).
/// 2. On a full miss, substitute both types via [`bond_type_sub`] and
///    retry both orderings.
/// 3. If every probe misses, return [`DEFAULT_BOND_PARAMS`].
pub fn lookup_bond_params(ti: &str, tj: &str, bonds: &BondParams) -> (f64, f64) {
    for (a, b) in [(ti, tj), (tj, ti)] {
        if let Some(&params) = bonds.get(&(a.to_string(), b.to_string())) {
            return params;
        }
    }

    let ti_s = bond_type_sub(ti);
    let tj_s = bond_type_sub(tj);
    for (a, b) in [(ti_s, tj_s), (tj_s, ti_s)] {
        if let Some(&params) = bonds.get(&(a.to_string(), b.to_string())) {
            return params;
        }
    }

    DEFAULT_BOND_PARAMS
}

/// Look up `(kt, t0)` for an angle `ti-tj-tk` (`tj` is the vertex atom).
///
/// Ports `_lookup_angle_params` (gaff2.py:1464-1477) exactly:
/// 1. Try `(ti, tj, tk)`, then `(tk, tj, ti)`, against `angles` (no
///    substitution). Note `tj` never moves — only the two terminal types
///    swap ends.
/// 2. On a full miss, substitute all three types via [`bond_type_sub`] and
///    retry both orderings.
/// 3. If every probe misses, return [`DEFAULT_ANGLE_PARAMS`].
pub fn lookup_angle_params(ti: &str, tj: &str, tk: &str, angles: &AngleParams) -> (f64, f64) {
    for (a, b, c) in [(ti, tj, tk), (tk, tj, ti)] {
        if let Some(&params) = angles.get(&(a.to_string(), b.to_string(), c.to_string())) {
            return params;
        }
    }

    let ti_s = bond_type_sub(ti);
    let tj_s = bond_type_sub(tj);
    let tk_s = bond_type_sub(tk);
    for (a, b, c) in [(ti_s, tj_s, tk_s), (tk_s, tj_s, ti_s)] {
        if let Some(&params) = angles.get(&(a.to_string(), b.to_string(), c.to_string())) {
            return params;
        }
    }

    DEFAULT_ANGLE_PARAMS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bonds_with(entries: &[((&str, &str), (f64, f64))]) -> BondParams {
        entries
            .iter()
            .map(|&((a, b), v)| ((a.to_string(), b.to_string()), v))
            .collect()
    }

    fn angles_with(entries: &[((&str, &str, &str), (f64, f64))]) -> AngleParams {
        entries
            .iter()
            .map(|&((a, b, c), v)| ((a.to_string(), b.to_string(), c.to_string()), v))
            .collect()
    }

    // --- bond_type_sub: ported from _BOND_TYPE_SUB (gaff2.py:1413-1418) ---

    #[test]
    fn bond_type_sub_c3_group() {
        for t in ["cx", "cy", "c5", "c6"] {
            assert_eq!(bond_type_sub(t), "c3");
        }
    }

    #[test]
    fn bond_type_sub_c2_group() {
        for t in ["cs", "cz", "ca", "cc", "cd", "ce", "cf", "cp", "cq"] {
            assert_eq!(bond_type_sub(t), "c2");
        }
    }

    #[test]
    fn bond_type_sub_c1_group() {
        for t in ["cg", "ch"] {
            assert_eq!(bond_type_sub(t), "c1");
        }
    }

    #[test]
    fn bond_type_sub_default_is_identity() {
        // Python: _BOND_TYPE_SUB.get(ti, ti) -- unmapped types pass through.
        for t in ["c3", "oh", "op", "hc", "n3", "zz"] {
            assert_eq!(bond_type_sub(t), t);
        }
    }

    // --- lookup_bond_params ---

    #[test]
    fn bond_direct_lookup() {
        // Real .dat-derived value, ported from tests/test_gaff2.py
        // known_bonds: ("c3", "op") -> (273.64, 1.4368) (alcohol O-H).
        let bonds = bonds_with(&[(("c3", "op"), (273.64, 1.4368))]);
        assert_eq!(lookup_bond_params("c3", "op", &bonds), (273.64, 1.4368));
    }

    #[test]
    fn bond_bidirectional_lookup() {
        // Only the reversed ordering is present -- gaff2.py:1452-1454 must
        // still find it via the (tj, ti) probe before falling to
        // substitution.
        let bonds = bonds_with(&[(("op", "c3"), (273.64, 1.4368))]);
        assert_eq!(lookup_bond_params("c3", "op", &bonds), (273.64, 1.4368));
    }

    #[test]
    fn bond_substitution_fallback_forward() {
        // "cx" is not itself in the table, but substitutes to "c3"
        // (_BOND_TYPE_SUB), which resolves against the forward probe.
        let bonds = bonds_with(&[(("c3", "op"), (273.64, 1.4368))]);
        assert_eq!(lookup_bond_params("cx", "op", &bonds), (273.64, 1.4368));
    }

    #[test]
    fn bond_substitution_fallback_reversed() {
        // Only the substituted+reversed ordering is present.
        let bonds = bonds_with(&[(("op", "c3"), (273.64, 1.4368))]);
        assert_eq!(lookup_bond_params("cx", "op", &bonds), (273.64, 1.4368));
    }

    #[test]
    fn bond_exact_match_wins_over_substitution() {
        // Risk-note case: substitution must apply ONLY on a full miss of
        // the unsubstituted pair, never pre-emptively. If both the exact
        // "cx"-"op" entry and the generic substituted "c3"-"op" entry
        // exist, the exact entry must win.
        let bonds = bonds_with(&[
            (("cx", "op"), (111.0, 1.10)),
            (("c3", "op"), (273.64, 1.4368)),
        ]);
        assert_eq!(lookup_bond_params("cx", "op", &bonds), (111.0, 1.10));
    }

    #[test]
    fn bond_miss_returns_default() {
        let bonds = bonds_with(&[(("c3", "op"), (273.64, 1.4368))]);
        assert_eq!(lookup_bond_params("zz", "yy", &bonds), DEFAULT_BOND_PARAMS);
    }

    // --- lookup_angle_params ---

    #[test]
    fn angle_direct_lookup() {
        // Synthetic value (no numeric angle fixture exists in the Python
        // test suite for this pair) -- exercises the direct-probe path.
        let angles = angles_with(&[(("c3", "c3", "oh"), (63.21, 109.47))]);
        assert_eq!(
            lookup_angle_params("c3", "c3", "oh", &angles),
            (63.21, 109.47)
        );
    }

    #[test]
    fn angle_bidirectional_lookup() {
        // Only the reversed ordering (tk, tj, ti) is present -- the vertex
        // type tj must stay fixed in the middle position on both probes.
        let angles = angles_with(&[(("oh", "c3", "c3"), (63.21, 109.47))]);
        assert_eq!(
            lookup_angle_params("c3", "c3", "oh", &angles),
            (63.21, 109.47)
        );
    }

    #[test]
    fn angle_substitution_fallback_forward() {
        let angles = angles_with(&[(("c3", "c3", "oh"), (63.21, 109.47))]);
        // "cx" substitutes to "c3" on both ends; middle atom is unaffected.
        assert_eq!(
            lookup_angle_params("cx", "c3", "oh", &angles),
            (63.21, 109.47)
        );
    }

    #[test]
    fn angle_substitution_fallback_reversed() {
        let angles = angles_with(&[(("oh", "c3", "c3"), (63.21, 109.47))]);
        assert_eq!(
            lookup_angle_params("cx", "c3", "oh", &angles),
            (63.21, 109.47)
        );
    }

    #[test]
    fn angle_exact_match_wins_over_substitution() {
        let angles = angles_with(&[
            (("cx", "c3", "oh"), (10.0, 100.0)),
            (("c3", "c3", "oh"), (63.21, 109.47)),
        ]);
        assert_eq!(
            lookup_angle_params("cx", "c3", "oh", &angles),
            (10.0, 100.0)
        );
    }

    #[test]
    fn angle_miss_returns_default() {
        let angles = angles_with(&[(("c3", "c3", "oh"), (63.21, 109.47))]);
        assert_eq!(
            lookup_angle_params("zz", "yy", "xx", &angles),
            DEFAULT_ANGLE_PARAMS
        );
    }

    #[test]
    fn angle_vertex_type_also_substitutes_on_fallback() {
        // The vertex type tj is substituted too (gaff2.py:1472
        // `tj_s = _BOND_TYPE_SUB.get(tj, tj)`), even though it never moves
        // position between the two probes.
        let angles = angles_with(&[(("c3", "c2", "c3"), (44.0, 111.0))]);
        // ti="c3" (no-op sub), tj="ca" (subs to "c2"), tk="cx" (subs to "c3")
        assert_eq!(
            lookup_angle_params("c3", "ca", "cx", &angles),
            (44.0, 111.0)
        );
    }
}
