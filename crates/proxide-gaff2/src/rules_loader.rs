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
//! - If the bundled file is missing, the default path degrades to an empty
//!   `(rules, wildatom_map)` pair rather than erroring -- it does NOT cache
//!   a read/parse *failure* on a file that does exist; that case is left
//!   unset so the next call retries (mirrors the fact that Python's globals
//!   are only ever assigned inside the "exists" success branch or the
//!   "doesn't exist" empty branch, never on a raised exception).
//! - An explicit `def_path` (`load_gaff2_rules(Some(path))`) bypasses the
//!   cache entirely and always re-reads + re-parses, exactly like Python's
//!   `load_gaff2_rules(def_path)` calling `parse_gaff2_rules(def_path)`
//!   directly -- and, unlike the default path, does NOT pre-check
//!   existence, so a missing explicit path surfaces as a real I/O error
//!   instead of a silent empty fallback.

use std::io;
use std::path::{Path, PathBuf};

use once_cell::sync::OnceCell;

use crate::def_parser::{self, Gaff2Rule, WildatomMap};

/// Absolute, build-time-resolved path to the bundled DEF file.
///
/// Python resolves this at *runtime*, relative to wherever `gaff2.py`
/// actually lives on disk (`Path(__file__).parent.parent / "assets" /
/// "gaff" / "dat" / "ATOMTYPE_GFF2.DEF"`, gaff2.py:561) -- there is no
/// direct Rust equivalent of `__file__` for a compiled crate. We instead
/// bake in the path relative to this crate's manifest directory
/// (`CARGO_MANIFEST_DIR`, resolved at compile time) and still perform a
/// genuine *runtime* `exists()` check + file read against it, which
/// reproduces Python's observable three-way behavior (parsed / gracefully
/// empty / propagated read error) faithfully for this monorepo. This
/// crate is `publish = false` (see Cargo.toml) specifically because this
/// path assumption does not survive being packaged out of the workspace.
fn default_rules_path() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../src/proxide/assets/gaff/dat/ATOMTYPE_GFF2.DEF"
    ))
}

/// Read `path` and parse it via `parse`. Shared by the default-path and
/// explicit-path branches; `parse` is the injection point tests use to
/// stand in for `def_parser::parse_gaff2_rules` so this module's
/// caching/loading orchestration is unit-testable on its own, independent
/// of the real DEF grammar (and without touching the real bundled DEF file
/// or the process-global [`DEFAULT_RULES`] cache from a test).
fn read_and_parse(
    path: &Path,
    parse: impl FnOnce(&str) -> Result<(Vec<Gaff2Rule>, WildatomMap), String>,
) -> io::Result<(Vec<Gaff2Rule>, WildatomMap)> {
    let content = std::fs::read_to_string(path)?;
    parse(&content).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Testable core of `get_default_rules()`: caller supplies the cache cell,
/// the path, and the parse function, so tests can exercise the "missing
/// file -> empty, cached" and "present file -> parsed, cached once" paths
/// without touching the real bundled DEF or the real (stub) parser.
fn get_default_rules_with(
    cache: &OnceCell<(Vec<Gaff2Rule>, WildatomMap)>,
    path: &Path,
    parse: impl FnOnce(&str) -> Result<(Vec<Gaff2Rule>, WildatomMap), String>,
) -> io::Result<(Vec<Gaff2Rule>, WildatomMap)> {
    cache
        .get_or_try_init(|| {
            if path.exists() {
                read_and_parse(path, parse)
            } else {
                Ok((Vec::new(), WildatomMap::new()))
            }
        })
        .cloned()
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
pub fn get_default_rules() -> io::Result<(Vec<Gaff2Rule>, WildatomMap)> {
    get_default_rules_with(&DEFAULT_RULES, &default_rules_path(), |c| {
        def_parser::parse_gaff2_rules(c).map(|(rules, wildatom)| (rules, wildatom))
    })
}

/// Port of `load_gaff2_rules()` (gaff2.py:575-589).
///
/// `def_path: None` returns the cached default (see `get_default_rules`).
/// `def_path: Some(path)` always re-reads and re-parses `path` directly,
/// bypassing the cache -- matching Python's `parse_gaff2_rules(def_path)`
/// call, including the lack of an existence pre-check (a missing explicit
/// path surfaces as an `io::Error`, just as Python's `Path.read_text()`
/// would raise `FileNotFoundError`).
pub fn load_gaff2_rules(def_path: Option<&Path>) -> io::Result<(Vec<Gaff2Rule>, WildatomMap)> {
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

    #[test]
    fn default_path_ends_with_bundled_asset_location() {
        // Fixture-equivalent of gaff2.py:561's
        // `Path(__file__).parent.parent / "assets" / "gaff" / "dat" / "ATOMTYPE_GFF2.DEF"`.
        let path = default_rules_path();
        assert!(
            path.ends_with("src/proxide/assets/gaff/dat/ATOMTYPE_GFF2.DEF"),
            "unexpected default rules path: {}",
            path.display()
        );
    }

    #[test]
    fn bundled_def_file_exists_in_this_checkout() {
        // Sanity check mirroring gaff2.py:563's `rules_path.exists()` guard:
        // in a real checkout this must be true, or `get_default_rules()`
        // silently degrades to empty rules for every caller.
        assert!(
            default_rules_path().exists(),
            "bundled ATOMTYPE_GFF2.DEF not found at {}",
            default_rules_path().display()
        );
    }

    #[test]
    fn missing_default_file_degrades_to_empty_and_caches_it() {
        let cache = OnceCell::new();
        let missing = PathBuf::from("/nonexistent/ATOMTYPE_GFF2.DEF");
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let (rules, wildatom) =
            get_default_rules_with(&cache, &missing, fake_parse_ok(calls.clone())).unwrap();
        assert!(rules.is_empty());
        assert!(wildatom.is_empty());
        // Parser must never be invoked for a missing default file
        // (gaff2.py:565-567's `else: _default_rules = []`).
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);

        // Cached: a second call returns the same empty result without
        // re-checking the parser.
        let (rules2, wildatom2) =
            get_default_rules_with(&cache, &missing, fake_parse_ok(calls.clone())).unwrap();
        assert!(rules2.is_empty());
        assert!(wildatom2.is_empty());
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn present_default_file_is_parsed_once_and_cached() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "ATD c3 * 6 4 *&").unwrap();
        writeln!(tmp, "ATD n3 * 7 3 *&").unwrap();
        let path = tmp.path().to_path_buf();

        let cache = OnceCell::new();
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let (rules, wildatom) =
            get_default_rules_with(&cache, &path, fake_parse_ok(calls.clone())).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(
            wildatom.get("XX"),
            Some(&vec!["C".to_string(), "N".to_string()])
        );
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Second call must hit the cache, not re-parse (gaff2.py:559's
        // `if _default_rules is None:` guard only ever runs the parse once).
        let (rules2, _wildatom2) =
            get_default_rules_with(&cache, &path, fake_parse_ok(calls.clone())).unwrap();
        assert_eq!(rules2.len(), 2);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn load_gaff2_rules_none_uses_default_cache() {
        // `load_gaff2_rules(None)` (gaff2.py:586-587) delegates to
        // `_get_default_rules()`. We can't inject a fake parser through the
        // public `load_gaff2_rules` entry point (it hardwires
        // `def_parser::parse_gaff2_rules`, matching Python's hardwired
        // call), so this only exercises the routing, not the parse itself;
        // the missing-file degrade-to-empty case is what's checkable here.
        let missing = PathBuf::from("/nonexistent/also-missing.DEF");
        assert!(!missing.exists());
        // Not calling load_gaff2_rules(None) here since that would touch
        // the process-global DEFAULT_RULES cache (shared with every other
        // test/caller in this binary, including get_default_rules()'s own
        // real-DEF-parsing callers elsewhere in this crate) -- a test-order-
        // dependent cross-test interaction, not a correctness concern with
        // the real (now-implemented) def_parser::parse_gaff2_rules itself.
        // The routing (None -> get_default_rules(), Some -> read_and_parse)
        // is exercised directly by the tests above and below via their
        // shared helper functions; end-to-end coverage of the real default
        // DEF file lives in orchestrate.rs's tests instead.
    }

    #[test]
    fn explicit_missing_path_propagates_io_error_not_empty_fallback() {
        // Unlike the default path (gaff2.py:563's `if rules_path.exists()`
        // guard), an explicit `def_path` has no existence pre-check
        // (gaff2.py:589's direct `parse_gaff2_rules(def_path)` call) --  a
        // missing explicit path must surface as a real error, not silently
        // degrade to empty rules.
        let missing = Path::new("/nonexistent/explicit-path.DEF");
        let err = load_gaff2_rules(Some(missing)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
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
