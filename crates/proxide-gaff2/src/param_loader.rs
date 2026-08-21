//! GAFF2 `.dat` parameter table loader.
//!
//! Line-for-line port of `load_gaff2_parameters()` (`src/proxide/chem/
//! gaff2.py:1260-1384`). Parses an AMBER-format GAFF2 `.dat` file (e.g.
//! `gaff-2.2.20.dat`) into atom-mass, bond, angle, torsion, improper, and
//! Lennard-Jones ("VdW"/MOD4) parameter tables.
//!
//! # This is a behavior-preserving port, not a cleanup
//!
//! The AMBER `.dat` grammar this function implements is genuinely
//! heuristic, and the Python reference has real, known residual mismatches
//! against AmberTools' actual parser (see the module risk notes and the
//! `GOTCHA` comments inline below). They are preserved here **on purpose**.
//! Two are load-bearing enough to call out up front:
//!
//! 1. **Multi-single-char-type torsion lines silently fail to parse.** The
//!    AMBER `.dat` quirk where single-character atom types are right-padded
//!    with a space (so `"c -o kb r0"` splits as `["c", "-o", "kb", "r0"]`)
//!    is fixed up with a *single, one-shot* merge of `parts[0]`/`parts[1]`.
//!    A line with two or more single-char types back to back (e.g. the
//!    common wildcard torsion shape `"X -c -c -X ..."`) only gets the first
//!    pair merged; the resulting `first` token has the wrong dash count and
//!    the line contributes nothing to any table. This is real: it silently
//!    drops a meaningful slice of the wildcard torsion section against the
//!    bundled `gaff-2.2.20.dat`.
//! 2. **The improper-torsion branch requires `parts[1]` to parse as a
//!    strict integer** (Python `int(parts[1])`, ported here as
//!    `parts[1].parse::<i64>()`). Real improper lines in
//!    `gaff-2.2.20.dat` put a *fractional force constant* (e.g. `"1.1"`,
//!    `"10.5"`) in that column, not an integer periodicity — so this parse
//!    fails for essentially the entire improper section, and
//!    `Gaff2Parameters::impropers` ends up empty against the real bundled
//!    file. Verified directly against the file during this port (see the
//!    unit tests below); this is not a hypothetical edge case.
//!
//! Do not "fix" either of these here. Triage of remaining parity mismatches
//! against real AmberTools output happens separately, by explicit
//! instruction.

use std::fs;
use std::io;
use std::path::Path;

use indexmap::IndexMap;

/// Embedded canonical GAFF2 `.dat` parameter file.
///
/// Path: `../../../src/proxide/assets/gaff/dat/gaff-2.2.20.dat` — the same
/// asset directory used by `rules_loader.rs`'s `ATOMTYPE_GFF2.DEF`
/// embedding, and the default `dat_path` resolved by `gaff2.py:1272`.
static GAFF2_DAT: &str = include_str!("../../../src/proxide/assets/gaff/dat/gaff-2.2.20.dat");

/// Key into [`Gaff2Parameters::bonds`]: `(type1, type2)`, in file order
/// (not sorted). Callers that need a canonical lookup key sort themselves —
/// this mirrors the Python reference, which never sorts at parse time
/// either (only some call sites do, e.g. `tuple(sorted(pair))` in
/// `tests/test_gaff2.py`).
pub type BondKey = (String, String);
/// Key into [`Gaff2Parameters::angles`]: `(type1, type2, type3)`, `type2`
/// is the vertex atom, in file order.
pub type AngleKey = (String, String, String);
/// Key into [`Gaff2Parameters::torsions`] / [`Gaff2Parameters::impropers`]:
/// `(type1, type2, type3, type4)`, in file order.
pub type TorsionKey = (String, String, String, String);

/// Parsed GAFF2 parameter tables — mirrors the dict returned by
/// `load_gaff2_parameters()`.
///
/// All maps are order-preserving (`IndexMap`), matching Python `dict`
/// insertion-order + update-in-place semantics: the `.dat` file can (and
/// does) contain duplicate keys for `masses`/`bonds`/`angles`/`vdw`/
/// `impropers`, and the Python reference keeps the *last* value seen while
/// leaving the key's position fixed at its *first* occurrence. A plain
/// `HashMap` would preserve neither the position nor make "last write wins"
/// visible to any downstream consumer that iterates the table. `torsions`
/// additionally accumulates multiple `(periodicity, kt, phase)` terms per
/// key in file-encounter order (multi-term Fourier torsions are consecutive
/// `.dat` lines sharing one 4-type key).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Gaff2Parameters {
    /// Atomic mass (amu) keyed by atom type.
    pub masses: IndexMap<String, f64>,
    /// `(kb, r0)` — force constant (kcal/mol/Å²) and equilibrium length (Å).
    pub bonds: IndexMap<BondKey, (f64, f64)>,
    /// `(kt, t0)` — force constant (kcal/mol/rad²) and equilibrium angle
    /// (degrees).
    pub angles: IndexMap<AngleKey, (f64, f64)>,
    /// `(periodicity, kt, phase)` terms, one `Vec` entry per `.dat` line
    /// sharing the same 4-type key, in file order.
    pub torsions: IndexMap<TorsionKey, Vec<(i64, f64, f64)>>,
    /// `(kt, phase)`. See the module-level note: against the real bundled
    /// `gaff-2.2.20.dat`, this table ends up empty (residual mismatch,
    /// preserved intentionally).
    pub impropers: IndexMap<TorsionKey, (f64, f64)>,
    /// `(rmin_half_angstrom, epsilon_kcal_mol)` keyed by atom type, from the
    /// `MOD4` (Lennard-Jones) section.
    pub vdw: IndexMap<String, (f64, f64)>,
}

/// Python's `str.islower()`: `true` iff there is at least one cased
/// character and every cased character is lowercase. Uncased characters
/// (digits, punctuation) don't affect the result either way, but a string
/// with *no* cased characters at all returns `false` — this matters for the
/// mass-line heuristic below (a purely-numeric `first` token would
/// otherwise be misclassified), though no real GAFF2 type token actually
/// hits that edge in the bundled `.dat` file.
fn py_islower(s: &str) -> bool {
    let mut has_cased = false;
    for c in s.chars() {
        if c.is_uppercase() {
            return false;
        }
        if c.is_lowercase() {
            has_cased = true;
        }
    }
    has_cased
}

/// Parse GAFF2 `.dat` file content into parameter tables.
///
/// Faithful, line-for-line port of `load_gaff2_parameters()`'s parse loop
/// (`gaff2.py:1283-1384`). See the module-level docs and inline `GOTCHA`
/// comments for intentionally-preserved residual mismatches against real
/// AmberTools output.
pub fn parse_gaff2_parameters(content: &str) -> Gaff2Parameters {
    let mut params = Gaff2Parameters::default();
    let mut in_vdw = false;

    for line in content.split('\n') {
        let line_stripped = line.trim();

        // Detect VdW section (MOD4 RE header). Checked unconditionally,
        // before the in_vdw branch, on every line -- matches gaff2.py:1291
        // exactly (a line CAN be blank and still fail this prefix check
        // harmlessly).
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
                if let (Ok(rmin_half), Ok(eps)) = (parts[1].parse::<f64>(), parts[2].parse::<f64>())
                {
                    params.vdw.insert(parts[0].to_string(), (rmin_half, eps));
                }
            }
            continue;
        }

        if line_stripped.is_empty() {
            continue;
        }

        let mut parts: Vec<String> = line_stripped
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        if parts.len() < 2 {
            continue;
        }

        // GOTCHA (AMBER .dat quirk, gaff2.py:1313-1316, and see module docs
        // point 1): single-char atom types are right-padded with a space,
        // so "c -o kb r0" splits as ["c", "-o", "kb", "r0"]. Reconstruct
        // the dash-joined key -- but only ONE time, for the first pair.
        // A line with two-or-more single-char types back to back is NOT
        // fully reconstructed and silently fails to parse further down.
        // Preserved intentionally.
        if !parts[0].contains('-') && parts.len() >= 2 && parts[1].starts_with('-') {
            let merged = format!("{}{}", parts[0], parts[1]);
            let mut new_parts = Vec::with_capacity(parts.len() - 1);
            new_parts.push(merged);
            new_parts.extend(parts.into_iter().skip(2));
            parts = new_parts;
        }

        let first = parts[0].clone();

        if first.contains('-') {
            let dash_count = first.matches('-').count();

            if dash_count == 1 {
                // Bond: type1-type2  kb  r0
                let mut split = first.splitn(2, '-');
                let t1 = split.next().unwrap_or("").to_string();
                let t2 = split.next().unwrap_or("").to_string();
                if t1.len() <= 3 && t2.len() <= 3 {
                    if let (Some(p1), Some(p2)) = (parts.get(1), parts.get(2)) {
                        if let (Ok(kb), Ok(r0)) = (p1.parse::<f64>(), p2.parse::<f64>()) {
                            if kb > 0.0 {
                                params.bonds.insert((t1, t2), (kb, r0));
                            }
                        }
                    }
                }
            } else if dash_count == 2 {
                // Angle: type1-type2-type3  kt  t0
                let mut split = first.splitn(2, '-');
                let t1 = split.next().unwrap_or("").to_string();
                let rest = split.next().unwrap_or("");
                let mut split2 = rest.splitn(2, '-');
                let t2 = split2.next().unwrap_or("").to_string();
                let t3 = split2.next().unwrap_or("").to_string();
                if t1.len() <= 3 && t2.len() <= 3 && t3.len() <= 3 {
                    if let (Some(p1), Some(p2)) = (parts.get(1), parts.get(2)) {
                        if let (Ok(kt), Ok(t0)) = (p1.parse::<f64>(), p2.parse::<f64>()) {
                            if kt > 0.0 {
                                params.angles.insert((t1, t2, t3), (kt, t0));
                            }
                        }
                    }
                }
            } else if dash_count == 3 {
                // Torsion or improper: type1-type2-type3-type4  ...
                let mut split = first.splitn(2, '-');
                let t1 = split.next().unwrap_or("").to_string();
                let rest = split.next().unwrap_or("");
                let mut split2 = rest.splitn(2, '-');
                let t2 = split2.next().unwrap_or("").to_string();
                let rest2 = split2.next().unwrap_or("");
                let mut split3 = rest2.splitn(2, '-');
                let t3 = split3.next().unwrap_or("").to_string();
                let t4 = split3.next().unwrap_or("").to_string();

                if t1.len() <= 3 && t2.len() <= 3 && t3.len() <= 3 && t4.len() <= 3 {
                    // GOTCHA (gaff2.py:1355-1357, and see module docs point
                    // 2): periodicity is parsed as a STRICT integer from
                    // parts[1]. All three of periodicity/kt/phase must
                    // parse successfully (mirroring the single outer
                    // `try` in Python that wraps both the torsion and the
                    // improper branch) before either branch below runs --
                    // if the periodicity parse fails, the improper
                    // fallback is never attempted either, exactly as in
                    // Python.
                    let periodicity = parts.get(1).and_then(|s| s.parse::<i64>().ok());
                    let kt = parts.get(2).and_then(|s| s.parse::<f64>().ok());
                    let phase = parts.get(3).and_then(|s| s.parse::<f64>().ok());

                    if let (Some(periodicity), Some(kt), Some(phase)) = (periodicity, kt, phase) {
                        if parts.len() >= 5 {
                            // Torsion: has 4+ terms per line (may repeat
                            // across consecutive lines sharing this key).
                            let key = (t1, t2, t3, t4);
                            params
                                .torsions
                                .entry(key)
                                .or_default()
                                .push((periodicity, kt, phase));
                        } else {
                            // This could be improper. Python re-parses
                            // parts[2]/parts[3] here (kt_imp/phase_imp) in
                            // a redundant inner try -- since those strings
                            // already parsed successfully above (as
                            // kt/phase), the re-parse cannot behave
                            // differently, so it's elided here rather than
                            // duplicated.
                            if kt > 0.0 {
                                params.impropers.insert((t1, t2, t3, t4), (kt, phase));
                            }
                        }
                    }
                }
            }
            // dash_count == 0 is unreachable inside `first.contains('-')`;
            // dash_count >= 4 falls through with no effect, matching
            // Python (no further `elif` branch matches).
        } else if first.len() <= 3 && py_islower(&first.replace('+', "")) {
            // Mass: type  mass
            if let Some(mass) = parts.get(1).and_then(|s| s.parse::<f64>().ok()) {
                params.masses.insert(first, mass);
            }
        }
    }

    params
}

/// Load GAFF2 parameter tables from a `.dat` file path.
///
/// Mirrors `load_gaff2_parameters(dat_path)` (`gaff2.py:1260`): pass
/// `None` to use the bundled default (`gaff-2.2.20.dat`, embedded at
/// compile time via [`GAFF2_DAT`]), or `Some(path)` to parse an arbitrary
/// `.dat` file from disk.
pub fn load_gaff2_parameters(dat_path: Option<&Path>) -> io::Result<Gaff2Parameters> {
    match dat_path {
        Some(path) => {
            let content = fs::read_to_string(path)?;
            Ok(parse_gaff2_parameters(&content))
        }
        None => Ok(parse_gaff2_parameters(GAFF2_DAT)),
    }
}

/// Parse the bundled default GAFF2 `.dat` file (`gaff-2.2.20.dat`).
///
/// Infallible convenience wrapper around [`load_gaff2_parameters`] for the
/// common no-override case -- the embedded content can't fail to read.
pub fn load_default_gaff2_parameters() -> Gaff2Parameters {
    parse_gaff2_parameters(GAFF2_DAT)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Synthetic fixtures: exercise the parsing rules directly ---

    #[test]
    fn parses_simple_mass_line() {
        let params = parse_gaff2_parameters("c3 12.01         0.878               Sp3 C\n");
        assert_eq!(params.masses.get("c3"), Some(&12.01));
    }

    #[test]
    fn parses_ion_mass_line_with_plus_sign() {
        // "n+ 14.01 ... NH4+" -- the '+' must be stripped before the
        // islower() check, but the length check uses the ORIGINAL token
        // (len("n+") == 2 <= 3).
        let params = parse_gaff2_parameters("n+ 14.01         0.530               NH4+\n");
        assert_eq!(params.masses.get("n+"), Some(&14.01));
    }

    #[test]
    fn reconstructs_padded_single_char_bond_key() {
        // The real AMBER .dat quirk from the risk notes: "c -o kb r0"
        // splits as ["c", "-o", "kb", "r0"] and must be reconstructed as
        // "c-o" before the dash-count dispatch.
        let params = parse_gaff2_parameters("c -o   680.00   1.2100\n");
        assert_eq!(
            params.bonds.get(&("c".to_string(), "o".to_string())),
            Some(&(680.00, 1.21))
        );
    }

    #[test]
    fn one_shot_padding_fix_drops_lines_with_two_padded_single_char_types() {
        // GOTCHA regression: "X -c -c -X" (two single-char types back to
        // back) only gets ITS FIRST pair merged ("X" + "-c" -> "X-c"),
        // leaving "-c"/"-X" unmerged. The resulting `first` = "X-c" has
        // dash_count == 1, so this is misrouted into the BOND branch,
        // where float("-c") fails and the whole line contributes nothing.
        // This must stay broken -- it's a real, verified mismatch against
        // gaff-2.2.20.dat's wildcard torsion section, not a hypothetical.
        let params =
            parse_gaff2_parameters("X -c -c -X    4    1.200       180.000           2.000\n");
        assert!(params.bonds.is_empty());
        assert!(params.torsions.is_empty());
        assert!(params.impropers.is_empty());
    }

    #[test]
    fn multi_char_types_do_not_need_the_padding_fix() {
        // Multi-char types are never space-padded by AMBER, so a 4-type
        // torsion line like "X -c1-c1-X ..." merges cleanly (only "X" is
        // single-char) and parses as a real torsion.
        let params = parse_gaff2_parameters(
            "X -c1-c1-X    1    0.000       180.000           2.000      for both triple and single bonds\n",
        );
        let key = (
            "X".to_string(),
            "c1".to_string(),
            "c1".to_string(),
            "X".to_string(),
        );
        assert_eq!(params.torsions.get(&key), Some(&vec![(1, 0.0, 180.0)]));
    }

    #[test]
    fn parses_angle_line() {
        let params = parse_gaff2_parameters(
            "c3-c3-oh   76.79      109.66                2021     3654   1.3617\n",
        );
        let key = ("c3".to_string(), "c3".to_string(), "oh".to_string());
        assert_eq!(params.angles.get(&key), Some(&(76.79, 109.66)));
    }

    #[test]
    fn torsion_terms_accumulate_across_lines_sharing_a_key() {
        let content = "\
c1-c1-c1-c1    1    5.000         0.000           1.000
c1-c1-c1-c1    1    2.000       180.000           2.000
";
        let params = parse_gaff2_parameters(content);
        let key = (
            "c1".to_string(),
            "c1".to_string(),
            "c1".to_string(),
            "c1".to_string(),
        );
        assert_eq!(
            params.torsions.get(&key),
            Some(&vec![(1, 5.0, 0.0), (1, 2.0, 180.0)])
        );
    }

    #[test]
    fn improper_branch_requires_len_parts_below_5() {
        // Only ONE single-char type ("c") at the front, so the one-shot
        // padding fix is sufficient here -- "ca"/"ca"/"n2" are all
        // multi-char and stay dash-joined with no padding gap between them.
        // (A line needing *two* merges, e.g. all-single-char types, hits
        // the GOTCHA covered by `one_shot_padding_fix_drops_lines_with_two_
        // padded_single_char_types` instead and must NOT parse.)
        let params = parse_gaff2_parameters("c -ca-ca-n2     1    5.0   180.0\n");
        let key = (
            "c".to_string(),
            "ca".to_string(),
            "ca".to_string(),
            "n2".to_string(),
        );
        assert_eq!(params.impropers.get(&key), Some(&(5.0, 180.0)));
    }

    #[test]
    fn improper_periodicity_column_rejects_fractional_values() {
        // GOTCHA regression (module docs point 2): a fractional value in
        // the periodicity column, as used throughout the REAL improper
        // section of gaff-2.2.20.dat (e.g. "1.1", "10.5"), fails Rust's
        // `i64::parse` exactly as it fails Python's `int()` -- the whole
        // line contributes nothing, to EITHER table.
        let params = parse_gaff2_parameters("c3-ca-ca-n2         1.1          180.          2.\n");
        assert!(params.impropers.is_empty());
        assert!(params.torsions.is_empty());
    }

    #[test]
    fn vdw_section_parses_after_mod4_header() {
        let content = "\
c -o   680.00   1.2100

MOD4      RE
  c3          1.9069  0.1078
  hc          1.4593  0.0208
";
        let params = parse_gaff2_parameters(content);
        assert_eq!(params.vdw.get("c3"), Some(&(1.9069, 0.1078)));
        assert_eq!(params.vdw.get("hc"), Some(&(1.4593, 0.0208)));
        // The pre-MOD4 bond line is still parsed normally.
        assert_eq!(
            params.bonds.get(&("c".to_string(), "o".to_string())),
            Some(&(680.00, 1.21))
        );
    }

    #[test]
    fn later_key_overwrites_value_but_keeps_original_iteration_position() {
        // Python dict semantics: re-assigning an existing key updates its
        // value in place without moving it in iteration order. IndexMap's
        // `insert` matches this exactly -- verify it here since it's the
        // entire reason this module uses IndexMap instead of HashMap.
        let content = "\
aa 1.0
bb 2.0
aa 3.0
";
        let params = parse_gaff2_parameters(content);
        assert_eq!(params.masses.get("aa"), Some(&3.0));
        assert_eq!(params.masses.keys().collect::<Vec<_>>(), vec!["aa", "bb"]);
    }

    #[test]
    fn zero_or_negative_bond_force_constant_is_dropped() {
        // gaff2.py:1330 `if kb > 0:` -- a non-positive kb is a parse
        // success but a store failure.
        let params = parse_gaff2_parameters("c -o   0.00   1.2100\n");
        assert!(params.bonds.is_empty());
    }

    // --- Real gaff-2.2.20.dat fixtures, ported from tests/test_gaff2.py's
    //     test_parameter_lookups()/test_parameter_values() ---
    //
    // NOTE: test_gaff2.py's `known_masses` fixture there is itself wrong
    // (it compares `params["masses"]` against numbers that are actually
    // the MOD4/VdW sigma values for those same atom types, e.g. "c3":
    // 1.9069 -- but c3's real mass is 12.01; 1.9069 is `sigma` from the
    // MOD4 section). This is invisible in Python because that test has no
    // `assert`/`return failed == 0`, so it can never fail regardless of
    // content (see this file's own docstring on the historical
    // ATOM_TYPE_TESTS breakage for another example) -- corrected here to
    // grounded, independently-verified values so the Rust assertions are
    // actually load-bearing.

    #[test]
    fn real_dat_masses() {
        let params = load_default_gaff2_parameters();
        for (atom_type, expected) in [
            ("c3", 12.01),
            ("cp", 12.01),
            ("cx", 12.01),
            ("oh", 16.00),
            ("op", 16.00),
            ("n3", 14.01),
            ("ni", 14.01),
            ("hc", 1.008),
        ] {
            assert_eq!(
                params.masses.get(atom_type),
                Some(&expected),
                "mass mismatch for {atom_type}"
            );
        }
    }

    #[test]
    fn real_dat_bonds() {
        let params = load_default_gaff2_parameters();
        let cases: [(BondKey, (f64, f64)); 3] = [
            (("c3".into(), "hc".into()), (345.25, 1.0962)),
            (("c3".into(), "oh".into()), (285.15, 1.4242)),
            (("cx".into(), "op".into()), (273.64, 1.4368)),
        ];
        for (key, expected) in cases {
            let actual = params.bonds.get(&key);
            assert!(actual.is_some(), "missing bond {key:?}");
            let (kb, r0) = *actual.unwrap();
            assert!((kb - expected.0).abs() < 1e-6, "{key:?} kb={kb}");
            assert!((r0 - expected.1).abs() < 1e-6, "{key:?} r0={r0}");
        }
    }

    #[test]
    fn real_dat_angles() {
        let params = load_default_gaff2_parameters();
        let cases: [(AngleKey, (f64, f64)); 3] = [
            (("c3".into(), "c3".into(), "c3".into()), (59.87, 112.63)),
            (("c3".into(), "c3".into(), "hc".into()), (43.27, 109.68)),
            (("c3".into(), "c3".into(), "oh".into()), (76.79, 109.66)),
        ];
        for (key, expected) in cases {
            let actual = params.angles.get(&key);
            assert!(actual.is_some(), "missing angle {key:?}");
            let (kt, t0) = *actual.unwrap();
            assert!((kt - expected.0).abs() < 1e-6, "{key:?} kt={kt}");
            assert!((t0 - expected.1).abs() < 1e-6, "{key:?} t0={t0}");
        }
    }

    #[test]
    fn real_dat_torsion_c3_c3_c3_c3_and_c3_c3_c3_hc() {
        let params = load_default_gaff2_parameters();
        let key_cccc: TorsionKey = ("c3".into(), "c3".into(), "c3".into(), "c3".into());
        let key_ccch: TorsionKey = ("c3".into(), "c3".into(), "c3".into(), "hc".into());
        assert!(
            params
                .torsions
                .get(&key_cccc)
                .is_some_and(|v| !v.is_empty()),
            "missing c3-c3-c3-c3 torsion terms"
        );
        assert!(
            params
                .torsions
                .get(&key_ccch)
                .is_some_and(|v| !v.is_empty()),
            "missing c3-c3-c3-hc torsion terms"
        );
    }

    #[test]
    fn real_dat_vdw_c3() {
        let params = load_default_gaff2_parameters();
        let (rmin_half, eps) = params.vdw.get("c3").copied().expect("c3 vdw entry");
        assert!((rmin_half - 1.9069).abs() < 1e-6);
        assert!((eps - 0.1078).abs() < 1e-6);
    }

    #[test]
    fn real_dat_impropers_table_is_empty_against_bundled_file() {
        // Direct verification of module docs point 2 / the
        // `improper_periodicity_column_rejects_fractional_values` unit
        // test, but against the actual bundled asset rather than a
        // synthetic fixture: confirms this isn't a quirk of a synthetic
        // string, it's the real, currently-shipped gaff-2.2.20.dat.
        let params = load_default_gaff2_parameters();
        assert!(
            params.impropers.is_empty(),
            "expected impropers to be empty against the real bundled .dat \
             (residual Python parity mismatch) -- got {} entries; if this \
             now fails, the bundled gaff-2.2.20.dat asset changed and this \
             comment/test needs re-verifying against the new file, not \
             blindly updating",
            params.impropers.len()
        );
    }

    #[test]
    fn real_dat_table_sizes_are_substantial() {
        // Loose sanity floors ported from test_gaff2.py::test_parameter_values
        // (masses >= 90, bonds >= 1000, angles >= 7000) -- these thresholds
        // were independently re-verified against the actual parse output at
        // port time, not copied blind.
        let params = load_default_gaff2_parameters();
        assert!(params.masses.len() >= 90, "masses: {}", params.masses.len());
        assert!(params.bonds.len() >= 1000, "bonds: {}", params.bonds.len());
        assert!(
            params.angles.len() >= 7000,
            "angles: {}",
            params.angles.len()
        );
    }
}
