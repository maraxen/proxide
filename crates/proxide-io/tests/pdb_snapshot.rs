//! Step 0 of sprint 24 (task 260922_autonomous-loop, track a, debt #1776+#1883):
//! a bit-exact snapshot of `parse_pdb_file`'s output for every `*.pdb` fixture
//! under `tests/data/` and `crates/*/tests/data/`, taken BEFORE the fixed-column
//! reader in `pdb_fields.rs` replaces the hand-rolled parser in `pdb.rs`.
//!
//! Purpose: the rewrite in later steps (fail-loud on malformed serial/res_seq/
//! coord/occupancy/B, non-ASCII, non-finite, residue reappearance) must not
//! silently change what the *valid* fixtures parse to. This snapshot is the
//! regression oracle for that claim -- see `snapshot_matches_checked_in_baseline`
//! below, which every later step must keep green (ledger B1: a report saying
//! "unchanged" is worthless; only a machine-checked byte comparison counts).
//!
//! Two tests:
//! - `regenerate_snapshot` (`#[ignore]`): re-parses every fixture and
//!   OVERWRITES `tests/data/parse_snapshot_base.json`. Run explicitly, only
//!   when an intentional, reviewed change to fixture parsing occurs.
//! - `snapshot_matches_checked_in_baseline`: re-parses every fixture and
//!   compares (not overwrites) against the checked-in file. A mismatch means
//!   either an unreviewed behaviour change (bug) or a fixture that must be
//!   added to `EXPECTED_DIVERGENCES` below with a reason (see module docs at
//!   the top of that const).

use proxide_io::formats::pdb::parse_pdb_file;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Fixtures whose snapshot is EXPECTED to change (e.g. a later step makes a
/// previously-silently-dropped/zeroed atom into a hard error, or -- as here --
/// a companion fix in the same sprint corrects a wrong inferred value). Empty
/// would mean every fixture snapshotted at Step 0 must still parse
/// bit-identically after every later step. Adding an entry here is a
/// reviewed decision, not a silent allowance -- see ledger B5 (a gate
/// satisfiable by narrowing the claim): removing a fixture from scope instead
/// of listing it here would be exactly that anti-pattern.
const EXPECTED_DIVERGENCES: &[(&str, &str)] = &[(
    "tests/data/trajectories/native.pdb",
    "Step 1 (decision g, same sprint) made proxide_core::infer_element strip a \
     leading digit before inference. This fixture's ACE/NME methyl hydrogens are \
     named '1HH3'/'2HH3'/'3HH3' with a blank element column (short line, no cols \
     77-78). Before Step 1: '1HH3' -> first char '1' -> no match -> silently \
     defaulted to element \"C\" (wrong -- backlog #5052-adjacent bug named in \
     decision g). After Step 1: '1HH3' -> strip '1' -> \"HH3\" -> 'H' (correct). \
     Atoms 1, 3, 4, 20, 21, 22 change element \"C\" -> \"H\"; no other field \
     changes. This is the intended effect of Step 1, not a Step 2/3 regression.",
)];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("proxide-io is two levels below the workspace root")
        .to_path_buf()
}

fn walk_pdb_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut children: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    children.sort();
    for path in children {
        if path.is_dir() {
            walk_pdb_files(&path, out);
        } else if path
            .extension()
            .map(|ext| ext.eq_ignore_ascii_case("pdb"))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
}

/// Every `*.pdb` under `tests/data/` (repo root) and `crates/*/tests/data/`,
/// as paths relative to the workspace root, in sorted order. Deliberately
/// scoped to exactly these two locations per the Step 0 instruction --
/// `crates/proxide_rs/foldcomp/test/test_af.pdb` (a different directory
/// shape) is out of scope.
fn discover_fixtures() -> Vec<PathBuf> {
    let root = workspace_root();
    let mut out = Vec::new();

    let top_tests_data = root.join("tests/data");
    if top_tests_data.is_dir() {
        walk_pdb_files(&top_tests_data, &mut out);
    }

    let crates_dir = root.join("crates");
    if let Ok(entries) = std::fs::read_dir(&crates_dir) {
        let mut crate_dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        crate_dirs.sort();
        for crate_dir in crate_dirs {
            let crate_tests_data = crate_dir.join("tests/data");
            if crate_tests_data.is_dir() {
                walk_pdb_files(&crate_tests_data, &mut out);
            }
        }
    }

    out.sort();
    out
}

fn rel_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
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
    // ' ' (blank) is the overwhelmingly common case; escape whatever else
    // shows up (rare non-ASCII alt_loc/i_code) the same way as strings.
    json_escape(&c.to_string())
}

/// Serialize one fixture's parse result as a single deterministic JSON object
/// (compact, one line, atoms in original parse order). Floats are stored as
/// their raw `f32::to_bits()` hex -- exact, and immune to any float-formatting
/// drift between Rust versions that decimal text would not be.
fn serialize_fixture(rel: &str, path: &Path) -> String {
    let mut out = String::new();
    let _ = write!(out, "  {:?}: ", rel);

    match parse_pdb_file(path) {
        Ok((raw, model_ids)) => {
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
                let _ = write!(
                    out,
                    "{{\"serial\":{},\"atom_name\":\"{}\",\"alt_loc\":\"{}\",\"res_name\":\"{}\",\"chain_id\":\"{}\",\"res_seq\":{},\"i_code\":\"{}\",\"element\":\"{}\",\"is_hetatm\":{},\"x_bits\":\"{:08x}\",\"y_bits\":\"{:08x}\",\"z_bits\":\"{:08x}\",\"occ_bits\":\"{:08x}\",\"b_bits\":\"{:08x}\"}}",
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
                );
            }
            out.push_str("]}");
        }
        Err(e) => {
            out.push_str("{\"status\":\"error\",\"message\":");
            let _ = write!(out, "{:?}", e.to_string());
            out.push('}');
        }
    }
    out
}

fn build_snapshot() -> BTreeMap<String, String> {
    let root = workspace_root();
    let mut map = BTreeMap::new();
    for path in discover_fixtures() {
        let rel = rel_path(&root, &path);
        map.insert(rel.clone(), serialize_fixture(&rel, &path));
    }
    map
}

fn snapshot_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/parse_snapshot_base.json")
}

fn render_snapshot(entries: &BTreeMap<String, String>) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    let n = entries.len();
    for (i, (_, line)) in entries.iter().enumerate() {
        out.push_str(line);
        if i + 1 < n {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("}\n");
    out
}

#[test]
#[ignore = "generator; run explicitly (cargo test -p proxide-io --test pdb_snapshot regenerate_snapshot -- --ignored) to intentionally re-baseline parse_snapshot_base.json"]
fn regenerate_snapshot() {
    let entries = build_snapshot();
    assert!(!entries.is_empty(), "no *.pdb fixtures discovered");
    let rendered = render_snapshot(&entries);
    std::fs::write(snapshot_path(), rendered).expect("write parse_snapshot_base.json");
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

    // Not byte-identical: every fixture whose line actually changed must be
    // an explicitly reviewed entry in EXPECTED_DIVERGENCES, not a silent
    // pass. Parse the checked-in file back into per-fixture lines the same
    // way it was written (one fixture per line) so we can diff line-by-line
    // instead of failing on the whole blob, which would hide which fixture
    // moved and let an unreviewed regression hide behind a reviewed one.
    let checked_lines: BTreeMap<String, String> = checked_in
        .lines()
        .filter(|l| l.trim_start().starts_with('"'))
        .filter_map(|l| {
            let colon = l.find("\": ")?;
            let key = l[..colon].trim().trim_matches('"').to_string();
            Some((key, l.trim_end_matches(',').to_string()))
        })
        .collect();

    let mut unexpected = Vec::new();
    for (fixture, new_line) in &current {
        let new_serialized = serialize_fixture(fixture, &workspace_root().join(fixture));
        let old_line = checked_lines.get(fixture);
        let changed =
            old_line.map(|o| o.trim_end_matches(',')) != Some(new_serialized.trim_end_matches(','));
        if changed {
            let allowed = EXPECTED_DIVERGENCES.iter().any(|(f, _)| f == fixture);
            if !allowed {
                unexpected.push(fixture.clone());
            }
        }
        let _ = new_line; // silence unused warning if map iteration order differs
    }

    assert!(
        unexpected.is_empty(),
        "unreviewed parse_pdb_file output change for fixture(s): {unexpected:?}. \
         If this divergence is intentional (e.g. a fixture that now correctly \
         errors instead of silently dropping/zeroing data), add it to \
         EXPECTED_DIVERGENCES with a reason and re-run --ignored regenerate_snapshot. \
         Do NOT regenerate blindly."
    );
}
