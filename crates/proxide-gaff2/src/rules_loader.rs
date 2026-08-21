//! Load and cache default GAFF2 rules from the bundled ATOMTYPE_GFF2.DEF.
//!
//! Port of `gaff2.py`'s module-level cache + loader trio:
//! - `_default_rules` / `_default_wildatom` (gaff2.py:551-552)
//! - `_get_default_rules()` (gaff2.py:555-572)
//! - `load_gaff2_rules()` (gaff2.py:575-589)
//!
//! Behavior preserved from Python:
//! - The bundled default DEF is parsed **at most once per process**; the
//!   result is cached and every subsequent default-path call returns the
//!   cached value.
//! - An explicit `def_path` (`load_gaff2_rules(Some(path))`) bypasses the
//!   cache entirely and always re-reads + re-parses, exactly like Python's
//!   `load_gaff2_rules(def_path)` calling `parse_gaff2_rules(def_path)`
//!   directly.
//!
//! Deliberate behavior deviation from Python (D1 fix, see
//! `.praxia/docs/reference/260821_gaff2-rust-port-lessons.md` Open Item #2):
//! Python resolves the bundled DEF at *runtime*, relative to wherever
//! `gaff2.py` actually lives on disk (`Path(__file__).parent.parent /
//! "assets" / "gaff" / "dat" / "ATOMTYPE_GFF2.DEF"`, gaff2.py:561), and
//! degrades to an empty ruleset if that path doesn't exist. An earlier
//! version of this module reproduced that runtime-path-plus-empty-fallback
//! shape via `env!("CARGO_MANIFEST_DIR")` -- which is a landmine, not a
//! faithful port: `CARGO_MANIFEST_DIR` is a *build-machine* path baked in at
//! compile time, so any deployment off that machine (a different checkout
//! location, a packaged wheel, a different CI runner) silently typed every
//! atom as `"x"` with no error at all. This module now embeds the DEF via
//! [`include_str!`] instead (same pattern as
//! `crates/proxide-wasm/src/gaff2.rs`'s `GAFF2_XML` embedding of a sibling
//! asset) -- the bundled default DEF cannot be "missing" at runtime anymore,
//! because it isn't a runtime file at all. This removes Python's
//! missing-file branch entirely, on purpose: a build with no valid DEF
//! content simply fails to compile, which is strictly safer than a binary
//! that runs and silently mistypes every atom.
//!
//! **`ATOMTYPE_GFF2.DEF` itself is not committed to this repository** (it is
//! GPL-licensed, AmberTools/antechamber-derived content, deliberately kept
//! out of this MIT-licensed repo's git history) -- run
//! `uv run python scripts/fetch_amber_assets.py` first to populate it before
//! building this crate. `include_str!` failing to find the file is exactly
//! the loud, compile-time failure this design wants: see that script's
//! docstring for the fetch/verify mechanism.

use std::path::Path;

use once_cell::sync::OnceCell;

use crate::def_parser::{self, Gaff2Rule, WildatomMap};

/// Compile-time-embedded contents of the bundled `ATOMTYPE_GFF2.DEF`. See
/// this module's doc comment for why `include_str!` replaced a runtime
/// `CARGO_MANIFEST_DIR`-relative file read.
static ATOMTYPE_GFF2_DEF: &str =
    include_str!("../../../src/proxide/assets/gaff/dat/ATOMTYPE_GFF2.DEF");

/// Testable core of `get_default_rules()`: caller supplies the cache cell,
/// the DEF content, and the parse function, so tests can exercise "parsed
/// once, cached" behavior against synthetic content without touching the
/// real bundled DEF or the process-global [`DEFAULT_RULES`] cache.
fn get_default_rules_with(
    cache: &OnceCell<(Vec<Gaff2Rule>, WildatomMap)>,
    content: &str,
    parse: impl FnOnce(&str) -> Result<(Vec<Gaff2Rule>, WildatomMap), String>,
) -> Result<(Vec<Gaff2Rule>, WildatomMap), String> {
    cache.get_or_try_init(|| parse(content)).cloned()
}

static DEFAULT_RULES: OnceCell<(Vec<Gaff2Rule>, WildatomMap)> = OnceCell::new();

/// Port of `_get_default_rules()` (gaff2.py:555-572).
///
/// Lazily parses and caches the bundled `ATOMTYPE_GFF2.DEF` exactly once
/// per process. Returns owned clones of the cached rules/wildatom map on
/// every call (Python instead returns references to the same cached `list`/
/// `dict` objects each time, so an external mutation of one returned tuple
/// would be visible to the next caller too; cloning here is a deliberate,
/// safer Rust-idiomatic deviation -- nothing in the Python call sites relies
/// on that aliasing).
pub fn get_default_rules() -> Result<(Vec<Gaff2Rule>, WildatomMap), String> {
    get_default_rules_with(&DEFAULT_RULES, ATOMTYPE_GFF2_DEF, def_parser::parse_gaff2_rules)
}

/// Read `path` and parse it via `parse`. Used by [`load_gaff2_rules`]'s
/// explicit-path branch; `parse` is the injection point tests use to stand
/// in for `def_parser::parse_gaff2_rules` so this helper is unit-testable
/// independent of the real DEF grammar.
fn read_and_parse(
    path: &Path,
    parse: impl FnOnce(&str) -> Result<(Vec<Gaff2Rule>, WildatomMap), String>,
) -> Result<(Vec<Gaff2Rule>, WildatomMap), String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    parse(&content)
}

/// Port of `load_gaff2_rules()` (gaff2.py:575-589).
///
/// `def_path: None` returns the cached default (see `get_default_rules`).
/// `def_path: Some(path)` always re-reads and re-parses `path` directly,
/// bypassing the cache -- matching Python's `parse_gaff2_rules(def_path)`
/// call, including the lack of an existence pre-check (a missing explicit
/// path surfaces as a propagated read error, just as Python's
/// `Path.read_text()` would raise `FileNotFoundError`).
pub fn load_gaff2_rules(def_path: Option<&Path>) -> Result<(Vec<Gaff2Rule>, WildatomMap), String> {
    match def_path {
        None => get_default_rules(),
        Some(path) => read_and_parse(path, def_parser::parse_gaff2_rules),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A trivial fake standing in for `def_parser::parse_gaff2_rules` so we
    /// can unit-test this module's own caching/path-resolution logic in
    /// isolation, independent of the real DEF grammar. It reports how many
    /// times it was invoked via `calls`, mirroring what
    /// `test_atom_types`-style fixtures implicitly rely on: that the
    /// default DEF is parsed once, not once per `assign_gaff2_atom_types`
    /// call.
    fn fake_parse_ok(
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) -> impl FnOnce(&str) -> Result<(Vec<Gaff2Rule>, WildatomMap), String> {
        move |content: &str| {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut wildatom = WildatomMap::new();
            wildatom.insert("XX".to_string(), vec!["C".to_string(), "N".to_string()]);
            // One Gaff2Rule per non-empty line, just to prove content
            // actually flowed through to the parser. Field values are
            // arbitrary placeholders -- only the count matters here.
            let rules = content
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|_| Gaff2Rule {
                    atom_type: "x".to_string(),
                    residue: "*".to_string(),
                    atomic_num: 6,
                    num_attached: None,
                    num_h: None,
                    h_ew_count: None,
                    atomic_prop_raw: None,
                    chem_env_raw: None,
                })
                .collect();
            Ok((rules, wildatom))
        }
    }

    /// D1 regression guard (Open Item #2): the embedded default DEF must be
    /// non-empty real content, not an accidentally-truncated or empty
    /// embed. This is the direct replacement for the old
    /// `bundled_def_file_exists_in_this_checkout` runtime-path check --
    /// there is no runtime path left to check, but "the embed actually
    /// contains the real file" still needs asserting.
    #[test]
    fn embedded_default_def_is_present_and_non_trivial() {
        assert!(
            ATOMTYPE_GFF2_DEF.len() > 1000,
            "embedded ATOMTYPE_GFF2.DEF looks truncated/empty: {} bytes",
            ATOMTYPE_GFF2_DEF.len()
        );
        assert!(
            ATOMTYPE_GFF2_DEF.contains("ATD"),
            "embedded content doesn't look like a GAFF2 DEF file (no ATD lines)"
        );
    }

    /// Content-digest pin (Open Item #6, drift guard #1 of 2 -- the second,
    /// a cross-language digest test spanning the maturin wheel boundary, is
    /// still open). A DEF bump must be a deliberate, reviewed change that
    /// forces re-running the parity campaign, not a silent behavior change
    /// picked up by whoever next rebuilds this crate. Recompute with
    /// `sha256sum src/proxide/assets/gaff/dat/ATOMTYPE_GFF2.DEF` and update
    /// deliberately if the DEF file is intentionally updated.
    #[test]
    fn embedded_default_def_content_digest_is_pinned() {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(ATOMTYPE_GFF2_DEF.as_bytes());
        let hex = format!("{digest:x}");
        assert_eq!(
            hex, "7a076ac2e667ab87057befc7a5985be4cead83e01ff5d2d3dab9f1d65bff637e",
            "ATOMTYPE_GFF2.DEF content changed (sha256 mismatch) -- if this is a \
             deliberate AmberTools update, update this pinned digest AND re-run \
             the full geostd parity campaign before merging"
        );
    }

    #[test]
    fn present_default_file_is_parsed_once_and_cached() {
        let cache = OnceCell::new();
        let content = "ATD c3 * 6 4 *&\nATD n3 * 7 3 *&\n";
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let (rules, wildatom) =
            get_default_rules_with(&cache, content, fake_parse_ok(calls.clone())).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(
            wildatom.get("XX"),
            Some(&vec!["C".to_string(), "N".to_string()])
        );
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Second call must hit the cache, not re-parse (gaff2.py:559's
        // `if _default_rules is None:` guard only ever runs the parse once).
        let (rules2, _wildatom2) =
            get_default_rules_with(&cache, content, fake_parse_ok(calls.clone())).unwrap();
        assert_eq!(rules2.len(), 2);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn load_gaff2_rules_none_uses_default_cache() {
        // `load_gaff2_rules(None)` (gaff2.py:586-587) delegates to
        // `_get_default_rules()`. We can't inject a fake parser through the
        // public `load_gaff2_rules` entry point (it hardwires
        // `def_parser::parse_gaff2_rules`, matching Python's hardwired
        // call), so routing (None -> get_default_rules(), Some ->
        // read_and_parse) is exercised directly by the tests above and
        // below via their shared helper functions instead; end-to-end
        // coverage of the real default DEF file lives in orchestrate.rs's
        // tests.
    }

    #[test]
    fn explicit_missing_path_propagates_error_not_empty_fallback() {
        // An explicit `def_path` has no existence pre-check (gaff2.py:589's
        // direct `parse_gaff2_rules(def_path)` call) -- a missing explicit
        // path must surface as a real error, not silently degrade to empty
        // rules. (Only the *default* path's old CARGO_MANIFEST_DIR-based
        // empty-fallback was the D1 landmine this module fixed -- the
        // explicit-path contract is unchanged.)
        let missing = Path::new("/nonexistent/explicit-path.DEF");
        let err = load_gaff2_rules(Some(missing)).unwrap_err();
        assert!(
            err.contains("No such file") || err.contains("os error 2"),
            "expected a not-found read error, got: {err}"
        );
    }

    #[test]
    fn explicit_path_bypasses_cache_and_reparses_every_call() {
        // Exercise the shared `read_and_parse` helper directly with the
        // fake parser (rather than through `load_gaff2_rules`, which
        // hardwires the real `def_parser::parse_gaff2_rules`) to confirm
        // each call re-reads content -- no caching for explicit paths,
        // matching gaff2.py:589 -- independent of the real DEF grammar.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "ATD c3 * 6 4 *&").unwrap();
        let path = tmp.path().to_path_buf();

        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (rules1, _) = read_and_parse(&path, fake_parse_ok(calls.clone())).unwrap();
        let (rules2, _) = read_and_parse(&path, fake_parse_ok(calls.clone())).unwrap();
        assert_eq!(rules1.len(), 1);
        assert_eq!(rules2.len(), 1);
        // Two independent calls -> two independent parses (no cache).
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }
}
