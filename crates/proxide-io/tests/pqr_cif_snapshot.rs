//! Step 0 of sprint 25 (task 260922_autonomous-loop, track a, debt #1919/#1920):
//! a bit-exact snapshot of `parse_pqr_file`/`parse_mmcif_file`/`parse_pdb_file`'s
//! output for a fixed, ENUMERATED set of PQR/mmCIF (+ one parity PDB) fixtures,
//! taken BEFORE this sprint's fail-loud rewrites touch `pqr.rs` (track a) or
//! `mmcif.rs` (track b).
//!
//! Mirrors `pdb_snapshot.rs` (sprint 24) exactly in spirit: this is the
//! regression oracle that later steps' rewrites must not silently change for
//! any fixture that is not explicitly listed in `EXPECTED_DIVERGENCES` with a
//! reviewed reason (ledger B1/B5).
//!
//! Unlike `pdb_snapshot.rs`, the fixture set here is a fixed, ENUMERATED list
//! (`FIXTURES`), not a directory walk -- `enumerated_fixture_set_is_exact`
//! below fails if a fixture is silently added to or removed from that list
//! without a matching, reviewed regeneration of the checked-in snapshot.
//!
//! Two generator/comparison tests, same pattern as `pdb_snapshot.rs`:
//! - `regenerate_snapshot` (`#[ignore]`): re-parses every fixture and
//!   OVERWRITES `tests/data/pqr_cif_snapshot/snapshot.json`. Run explicitly.
//! - `snapshot_matches_checked_in_baseline`: compares (not overwrites)
//!   against the checked-in file.

use proxide_core::structure::RawAtomData;
use proxide_io::formats::mmcif::parse_mmcif_file;
use proxide_io::formats::pdb::parse_pdb_file;
use proxide_io::formats::pqr::parse_pqr_file;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The fixed, enumerated fixture set for this snapshot (decision i / Step 0).
/// Paths are relative to the workspace root. Rotlib CCD `.cif` files are
/// deliberately excluded (decision i) -- they are a different tokenizer's
/// input (`ccd_parser.rs`, out of scope, tracked as debt).
const FIXTURES: &[&str] = &[
    "tests/data/1a00.pqr",
    "tests/io/parsing/altloc_two_conf.cif",
    "crates/proxide_rs/tests/fixtures/test.cif",
    "crates/proxide-io/tests/data/1CRN.cif",
    "crates/proxide-io/tests/data/1CRN.pdb",
    "crates/proxide-io/tests/data/1UBQ.cif",
];

/// Fixtures whose snapshot is EXPECTED to change in a later step of this
/// sprint, with a reviewed reason. Empty right now: Step 0 runs before any
/// parser change, so every fixture above must snapshot identically through
/// Step 2 (pqr.rs) unless a later fixer step adds a reviewed entry here.
const EXPECTED_DIVERGENCES: &[(&str, &str)] = &[];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("proxide-io is two levels below the workspace root")
        .to_path_buf()
}

fn snapshot_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pqr_cif_snapshot/snapshot.json")
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn char_field(c: char) -> String {
    json_escape(&c.to_string())
}

fn opt_bits(v: &Option<Vec<f32>>, i: usize) -> Option<u32> {
    v.as_ref().and_then(|vec| vec.get(i)).map(|f| f.to_bits())
}

fn opt_bits_json(bits: Option<u32>) -> String {
    match bits {
        Some(b) => format!("\"{b:08x}\""),
        None => "null".to_string(),
    }
}

/// Dispatch to the format-appropriate parser by extension. `.pqr` has no
/// model concept, so its model-id list is always empty.
fn parse_fixture(path: &Path) -> Result<(RawAtomData, Vec<usize>), String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "pqr" => parse_pqr_file(path)
            .map(|raw| (raw, Vec::new()))
            .map_err(|e| e.to_string()),
        "cif" => parse_mmcif_file(path).map_err(|e| e.to_string()),
        "pdb" => parse_pdb_file(path).map_err(|e| e.to_string()),
        other => Err(format!("pqr_cif_snapshot: unsupported extension {other:?}")),
    }
}

fn serialize_fixture(path: &Path) -> String {
    match parse_fixture(path) {
        Ok((raw, model_ids)) => {
            let mut out = String::new();
            out.push_str("{\"status\":\"ok\",\"num_atoms\":");
            let _ = write!(out, "{}", raw.num_atoms);
            out.push_str(",\"model_ids\":[");
            for (i, m) in model_ids.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let _ = write!(out, "{m}");
            }
            out.push_str("],\"atoms\":[");
            for i in 0..raw.num_atoms {
                if i > 0 {
                    out.push(',');
                }
                let x_bits = raw.coords[3 * i].to_bits();
                let y_bits = raw.coords[3 * i + 1].to_bits();
                let z_bits = raw.coords[3 * i + 2].to_bits();
                let occ_bits = raw.occupancy[i].to_bits();
                let b_bits = raw.b_factors[i].to_bits();
                let charge_json = opt_bits_json(opt_bits(&raw.charges, i));
                let radius_json = opt_bits_json(opt_bits(&raw.radii, i));
                let _ = write!(
                    out,
                    "{{\"serial\":{},\"atom_name\":\"{}\",\"alt_loc\":\"{}\",\"res_name\":\"{}\",\"chain_id\":\"{}\",\"res_seq\":{},\"i_code\":\"{}\",\"element\":\"{}\",\"is_hetatm\":{},\"x_bits\":\"{:08x}\",\"y_bits\":\"{:08x}\",\"z_bits\":\"{:08x}\",\"occ_bits\":\"{:08x}\",\"b_bits\":\"{:08x}\",\"charge_bits\":{},\"radius_bits\":{}}}",
                    raw.serial_numbers[i],
                    json_escape(&raw.atom_names[i]),
                    char_field(raw.alt_locs[i]),
                    json_escape(&raw.res_names[i]),
                    json_escape(&raw.chain_ids[i]),
                    raw.res_ids[i],
                    char_field(raw.insertion_codes[i]),
                    json_escape(&raw.elements[i]),
                    raw.is_hetatm[i],
                    x_bits,
                    y_bits,
                    z_bits,
                    occ_bits,
                    b_bits,
                    charge_json,
                    radius_json,
                );
            }
            out.push_str("]}");
            out
        }
        Err(msg) => {
            let mut out = String::new();
            out.push_str("{\"status\":\"error\",\"message\":");
            let _ = write!(out, "{:?}", msg);
            out.push('}');
            out
        }
    }
}

fn build_snapshot() -> BTreeMap<String, String> {
    let root = workspace_root();
    let mut map = BTreeMap::new();
    for fixture in FIXTURES {
        let path = root.join(fixture);
        map.insert(fixture.to_string(), serialize_fixture(&path));
    }
    map
}

fn render_snapshot(entries: &BTreeMap<String, String>) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    let n = entries.len();
    for (i, (key, line)) in entries.iter().enumerate() {
        let _ = write!(out, "  {:?}: {}", key, line);
        if i + 1 < n {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("}\n");
    out
}

fn parse_checked_in_lines(checked_in: &str) -> BTreeMap<String, String> {
    checked_in
        .lines()
        .filter(|l| l.trim_start().starts_with('"'))
        .filter_map(|l| {
            let colon = l.find("\": ")?;
            let key = l[..colon].trim().trim_matches('"').to_string();
            Some((key, l.trim_end_matches(',').to_string()))
        })
        .collect()
}

#[test]
fn enumerated_fixture_set_is_exact() {
    // Guards against a fixture being silently added to or removed from
    // FIXTURES without a reviewed, regenerated snapshot (ledger B5 -- a gate
    // satisfiable by narrowing the claim). Every fixture must also actually
    // exist on disk.
    let root = workspace_root();
    for fixture in FIXTURES {
        assert!(
            root.join(fixture).is_file(),
            "enumerated fixture {fixture:?} does not exist at {:?}",
            root.join(fixture)
        );
    }

    let checked_in = std::fs::read_to_string(snapshot_path()).unwrap_or_else(|e| {
        panic!(
            "missing checked-in snapshot at {:?}: {e}. Run with --ignored regenerate_snapshot first.",
            snapshot_path()
        )
    });
    let checked_keys: BTreeSet<String> = parse_checked_in_lines(&checked_in).into_keys().collect();
    let want: BTreeSet<String> = FIXTURES.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        checked_keys, want,
        "FIXTURES const and the checked-in snapshot's fixture set diverged -- \
         a fixture was added or removed without a reviewed --ignored regenerate_snapshot"
    );
}

#[test]
#[ignore = "generator; run explicitly (cargo test -p proxide-io --test pqr_cif_snapshot regenerate_snapshot -- --ignored) to intentionally re-baseline snapshot.json"]
fn regenerate_snapshot() {
    let entries = build_snapshot();
    let rendered = render_snapshot(&entries);
    std::fs::create_dir_all(snapshot_path().parent().unwrap()).expect("create snapshot dir");
    std::fs::write(snapshot_path(), rendered).expect("write snapshot.json");
}

#[test]
fn snapshot_matches_checked_in_baseline() {
    let checked_in = std::fs::read_to_string(snapshot_path()).unwrap_or_else(|e| {
        panic!(
            "missing checked-in snapshot at {:?}: {e}. Run with --ignored regenerate_snapshot \
             first (Step 0 must precede any parser change).",
            snapshot_path()
        )
    });

    let current = build_snapshot();
    let rendered = render_snapshot(&current);

    if rendered == checked_in {
        return;
    }

    let checked_lines = parse_checked_in_lines(&checked_in);

    let mut unexpected = Vec::new();
    for fixture in &current {
        let (fixture, _) = fixture;
        let new_serialized = serialize_fixture(&workspace_root().join(fixture));
        let old_line = checked_lines.get(fixture);
        let expected_line = format!("{:?}: {}", fixture, new_serialized);
        let changed = old_line != Some(&expected_line.trim_end_matches(',').to_string());
        if changed {
            let allowed = EXPECTED_DIVERGENCES.iter().any(|(f, _)| f == fixture);
            if !allowed {
                unexpected.push(fixture.clone());
            }
        }
    }

    assert!(
        unexpected.is_empty(),
        "unreviewed parse output change for fixture(s): {unexpected:?}. If this divergence is \
         intentional, add it to EXPECTED_DIVERGENCES with a reason and re-run --ignored \
         regenerate_snapshot. Do NOT regenerate blindly."
    );
}

// ---------------------------------------------------------------------
// Independent atom-count checks for the real wwPDB fixtures (decision i).
// A separate, minimal scan -- deliberately NOT reusing any parser's own
// tokenizer/row-assembly logic -- so a parser bug that drops/adds rows can't
// also make the "independent" count agree by construction.
// ---------------------------------------------------------------------

/// Count `_atom_site` loop data rows in a real wwPDB mmCIF file by the
/// simplest possible rule: a line beginning with `ATOM ` or `HETATM `
/// (column 1, as wwPDB always writes them -- no leading whitespace, no
/// wrapped rows). This is intentionally cruder than the real tokenizer.
fn independent_cif_atom_site_row_count(path: &Path) -> usize {
    let text = std::fs::read_to_string(path).expect("read cif fixture");
    text.lines()
        .filter(|l| l.starts_with("ATOM ") || l.starts_with("HETATM "))
        .count()
}

/// Count ATOM/HETATM lines in a legacy PDB file (column-1 record name).
fn independent_pdb_atom_line_count(path: &Path) -> usize {
    let text = std::fs::read_to_string(path).expect("read pdb fixture");
    text.lines()
        .filter(|l| l.starts_with("ATOM") || l.starts_with("HETATM"))
        .count()
}

#[test]
fn independent_count_matches_1crn_cif() {
    let path = workspace_root().join("crates/proxide-io/tests/data/1CRN.cif");
    let expected = independent_cif_atom_site_row_count(&path);
    let (raw, _) =
        parse_mmcif_file(&path).unwrap_or_else(|e| panic!("1CRN.cif: parser errored: {e}"));
    assert_eq!(
        raw.num_atoms, expected,
        "1CRN.cif: parser num_atoms ({}) disagrees with independent _atom_site row count ({}) \
         -- STOP: the old mmCIF parser drops or adds rows on a real file",
        raw.num_atoms, expected
    );
}

#[test]
fn independent_count_matches_1ubq_cif() {
    let path = workspace_root().join("crates/proxide-io/tests/data/1UBQ.cif");
    let expected = independent_cif_atom_site_row_count(&path);
    let (raw, _) =
        parse_mmcif_file(&path).unwrap_or_else(|e| panic!("1UBQ.cif: parser errored: {e}"));
    assert_eq!(
        raw.num_atoms, expected,
        "1UBQ.cif: parser num_atoms ({}) disagrees with independent _atom_site row count ({}) \
         -- STOP: the old mmCIF parser drops or adds rows on a real file",
        raw.num_atoms, expected
    );
}

#[test]
fn independent_count_matches_1crn_pdb() {
    let path = workspace_root().join("crates/proxide-io/tests/data/1CRN.pdb");
    let expected = independent_pdb_atom_line_count(&path);
    let (raw, _) =
        parse_pdb_file(&path).unwrap_or_else(|e| panic!("1CRN.pdb: parser errored: {e}"));
    assert_eq!(
        raw.num_atoms, expected,
        "1CRN.pdb: parser num_atoms ({}) disagrees with independent line count ({})",
        raw.num_atoms, expected
    );
}

#[test]
fn crn_cif_and_pdb_parity() {
    // decision i: same deposition in both formats must parse to the same
    // atom count, names, and coordinates (to file precision).
    let cif_path = workspace_root().join("crates/proxide-io/tests/data/1CRN.cif");
    let pdb_path = workspace_root().join("crates/proxide-io/tests/data/1CRN.pdb");
    let (cif_raw, _) = parse_mmcif_file(&cif_path).expect("1CRN.cif parses");
    let (pdb_raw, _) = parse_pdb_file(&pdb_path).expect("1CRN.pdb parses");

    assert_eq!(
        cif_raw.num_atoms, pdb_raw.num_atoms,
        "1CRN cif/pdb atom count parity"
    );
    assert_eq!(
        cif_raw.atom_names, pdb_raw.atom_names,
        "1CRN cif/pdb atom_name parity"
    );

    for i in 0..cif_raw.num_atoms {
        let cif_xyz = (
            cif_raw.coords[3 * i],
            cif_raw.coords[3 * i + 1],
            cif_raw.coords[3 * i + 2],
        );
        let pdb_xyz = (
            pdb_raw.coords[3 * i],
            pdb_raw.coords[3 * i + 1],
            pdb_raw.coords[3 * i + 2],
        );
        assert!(
            (cif_xyz.0 - pdb_xyz.0).abs() < 1e-3
                && (cif_xyz.1 - pdb_xyz.1).abs() < 1e-3
                && (cif_xyz.2 - pdb_xyz.2).abs() < 1e-3,
            "atom {i} coordinate mismatch: cif {cif_xyz:?} vs pdb {pdb_xyz:?}"
        );
    }
}
