//! LRU Cache for formatted structure outputs
//!
//! Provides caching of formatted structure data to avoid re-formatting
//! the same structure multiple times. An entry is keyed on the canonical
//! path, a fingerprint of the exact bytes that were parsed, and a description
//! of every option that shapes the output. A file overwritten at the same path
//! is parsed again, and a file touched without a change in bytes stays a hit.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use proxide_core::spec::OutputSpec;

/// Identity of a file's bytes: their length and a hash of their content. The
/// hash is std's `DefaultHasher`: it is not stable across Rust releases, which
/// is fine because the cache lives only in this process. mtime is deliberately
/// not part of the identity, so touching a file without changing it keeps the
/// entry valid.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct FileFingerprint {
    len: u64,
    content_hash: u64,
}

impl FileFingerprint {
    fn of_bytes(bytes: &[u8]) -> Self {
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        Self {
            len: bytes.len() as u64,
            content_hash: hasher.finish(),
        }
    }
}

/// Key for a cached parse: the canonical source path, the fingerprint of the
/// bytes parsed, and the options that shape the output (see `options` below).
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CacheKey {
    path: PathBuf,
    fingerprint: FileFingerprint,
    options: String,
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path.hash(state);
        self.fingerprint.hash(state);
        self.options.hash(state);
    }
}

impl CacheKey {
    /// Key for `bytes` read from `path`, parsed with `options`. The key is only
    /// correct for the bytes it hashed, so pass the same buffer the parse reads.
    /// `options` must describe every setting that changes the parse result.
    pub fn from_bytes(path: impl AsRef<Path>, bytes: &[u8], options: &str) -> io::Result<Self> {
        Ok(Self {
            path: std::fs::canonicalize(path)?,
            fingerprint: FileFingerprint::of_bytes(bytes),
            options: options.to_string(),
        })
    }

    /// Key for the file at `path` as it is on disk now. Reads the file once.
    pub fn for_file(path: impl AsRef<Path>, options: &str) -> io::Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        Self::from_bytes(path, &bytes, options)
    }

    /// Same file and options, regardless of content.
    fn same_slot(&self, other: &Self) -> bool {
        self.path == other.path && self.options == other.options
    }
}

/// Cached formatted structure data
#[derive(Debug, Clone)]
pub struct CachedStructure {
    pub coordinates: Vec<f32>,
    pub atom_mask: Vec<f32>,
    pub aatype: Vec<i8>,
    pub residue_index: Vec<i32>,
    pub chain_index: Vec<i32>,
    pub _num_residues: usize,
    /// Optional atom names (for Full format)
    pub atom_names: Option<Vec<String>>,
    /// Optional coordinate shape (for Full format)
    pub coord_shape: Option<(usize, usize, usize)>,
    /// Per-atom Full-format fields, so a cache hit returns the same dict as
    /// the first call (it used to drop them).
    pub full_per_atom: Option<FullPerAtom>,
    /// The chain vocabulary `chain_index` indexes, so a cache hit returns the
    /// same `unique_chain_ids` as the first call.
    pub unique_chain_ids: Option<Vec<String>>,
}

/// Per-atom arrays the Full format carries (see `FormattedFull`).
#[derive(Debug, Clone)]
pub struct FullPerAtom {
    pub atom_residue_ids: Vec<i32>,
    pub elements: Vec<String>,
    pub res_names: Vec<String>,
    pub atom_chain_ids: Vec<String>,
}

pub struct CachedStructureArgs {
    pub coordinates: Vec<f32>,
    pub atom_mask: Vec<f32>,
    pub aatype: Vec<i8>,
    pub residue_index: Vec<i32>,
    pub chain_index: Vec<i32>,
    pub num_residues: usize,
    pub atom_names: Option<Vec<String>>,
    pub coord_shape: Option<(usize, usize, usize)>,
    pub full_per_atom: Option<FullPerAtom>,
    pub unique_chain_ids: Option<Vec<String>>,
}

impl CachedStructure {
    pub fn new(args: CachedStructureArgs) -> Self {
        Self {
            coordinates: args.coordinates,
            atom_mask: args.atom_mask,
            aatype: args.aatype,
            residue_index: args.residue_index,
            chain_index: args.chain_index,
            _num_residues: args.num_residues,
            atom_names: args.atom_names,
            coord_shape: args.coord_shape,
            full_per_atom: args.full_per_atom,
            unique_chain_ids: args.unique_chain_ids,
        }
    }
}

/// LRU Cache with fixed capacity
pub struct FormatCache {
    cache: HashMap<CacheKey, CachedStructure>,
    order: Vec<CacheKey>,
    capacity: usize,
}

impl FormatCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            order: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn get(&mut self, key: &CacheKey) -> Option<&CachedStructure> {
        if self.cache.contains_key(key) {
            // Move to end (most recently used)
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                let k = self.order.remove(pos);
                self.order.push(k);
            }
            self.cache.get(key)
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: CacheKey, value: CachedStructure) {
        // The same file, format and flags under an older fingerprint is stale: drop it.
        let stale: Vec<CacheKey> = self
            .order
            .iter()
            .filter(|k| k.same_slot(&key) && **k != key)
            .cloned()
            .collect();
        for old in &stale {
            self.cache.remove(old);
        }
        self.order.retain(|k| !stale.contains(k));

        if self.cache.len() >= self.capacity && !self.cache.contains_key(&key) {
            // Evict least recently used
            if let Some(lru_key) = self.order.first().cloned() {
                self.cache.remove(&lru_key);
                self.order.remove(0);
            }
        }

        if !self.cache.contains_key(&key) {
            self.order.push(key.clone());
        }
        self.cache.insert(key, value);
    }

    pub fn _len(&self) -> usize {
        self.cache.len()
    }

    pub fn _is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn _clear(&mut self) {
        self.cache.clear();
        self.order.clear();
    }
}

// Global cache instance (thread-safe)
lazy_static::lazy_static! {
    static ref GLOBAL_CACHE: Mutex<FormatCache> = Mutex::new(FormatCache::new(100));
}

static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Get a structure from the global cache. Counts a hit or a miss.
pub fn get_cached(key: &CacheKey) -> Option<CachedStructure> {
    let mut cache = GLOBAL_CACHE.lock().unwrap();
    let found = cache.get(key).cloned();
    if found.is_some() {
        HITS.fetch_add(1, Ordering::Relaxed);
    } else {
        MISSES.fetch_add(1, Ordering::Relaxed);
    }
    found
}

/// Hits, misses and entry count of the global cache since process start.
pub fn cache_stats() -> (u64, u64, usize) {
    (
        HITS.load(Ordering::Relaxed),
        MISSES.load(Ordering::Relaxed),
        _cache_size(),
    )
}

/// Cache-key description of every option that shapes a parse's output. The whole
/// spec is described: Debug covers every field, so a field added later is keyed
/// automatically. `enable_caching` is the one field left out: it only decides
/// whether a result is stored, not what the result is.
pub fn options_key(spec: &OutputSpec) -> String {
    let mut keyed = spec.clone();
    keyed.enable_caching = false;
    format!("{keyed:?}")
}

/// Insert a structure into the global cache
pub fn insert_cached(key: CacheKey, value: CachedStructure) {
    let mut cache = GLOBAL_CACHE.lock().unwrap();
    cache.insert(key, value);
}

/// Clear the global cache
pub fn _clear_cache() {
    let mut cache = GLOBAL_CACHE.lock().unwrap();
    cache._clear();
}

/// Get the current cache size
pub fn _cache_size() -> usize {
    let cache = GLOBAL_CACHE.lock().unwrap();
    cache._len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const ATOM_A: &[u8] = b"ATOM      1  CA  ALA A   1       1.000   0.000   0.000\n";
    const ATOM_B: &[u8] = b"ATOM      1  CA  ALA A   1       9.000   0.000   0.000\n";
    const OPTS: &str = "unit=angstrom;target=general";

    fn sample_structure() -> CachedStructure {
        CachedStructure::new(CachedStructureArgs {
            coordinates: vec![1.0, 2.0, 3.0],
            atom_mask: vec![1.0],
            aatype: vec![0],
            residue_index: vec![1],
            chain_index: vec![0],
            num_residues: 1,
            atom_names: None,
            coord_shape: None,
            full_per_atom: None,
            unique_chain_ids: None,
        })
    }

    fn write_file(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn key(path: &Path) -> CacheKey {
        CacheKey::for_file(path, OPTS).unwrap()
    }

    #[test]
    fn test_cache_basic() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = write_file(&dir, "test1.pdb", ATOM_A);
        let p2 = write_file(&dir, "test2.pdb", ATOM_A);
        let p3 = write_file(&dir, "test3.pdb", ATOM_A);
        let mut cache = FormatCache::new(2);

        let key1 = key(&p1);
        let key2 = key(&p2);
        let key3 = key(&p3);

        cache.insert(key1.clone(), sample_structure());
        cache.insert(key2.clone(), sample_structure());

        assert_eq!(cache._len(), 2);
        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_some());

        // Order after inserts: key1, key2. After the gets: key1, key2, so key1
        // is the LRU entry and is evicted when key3 is inserted.
        cache.insert(key3.clone(), sample_structure());

        assert_eq!(cache._len(), 2);
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_some());
        assert!(cache.get(&key3).is_some());
    }

    #[test]
    fn test_from_bytes_matches_for_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "same.pdb", ATOM_A);
        let from_file = key(&path);
        let bytes = std::fs::read(&path).unwrap();
        let from_bytes = CacheKey::from_bytes(&path, &bytes, OPTS).unwrap();
        assert_eq!(from_file, from_bytes);
    }

    #[test]
    fn test_key_follows_the_bytes_it_is_given() {
        // The key must describe the buffer the parse reads, not a later read of the path.
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "given.pdb", ATOM_A);
        let from_disk = key(&path);
        let from_other_bytes = CacheKey::from_bytes(&path, ATOM_B, OPTS).unwrap();
        assert_ne!(from_disk, from_other_bytes);
    }

    #[test]
    fn test_different_options_are_separate_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "opts.pdb", ATOM_A);
        let mut cache = FormatCache::new(4);
        let angstrom = CacheKey::for_file(&path, "unit=angstrom;target=general").unwrap();
        let nanometre = CacheKey::for_file(&path, "unit=nm;target=general").unwrap();
        let alphafold = CacheKey::for_file(&path, "unit=angstrom;target=alphafold").unwrap();
        assert_ne!(angstrom, nanometre);
        assert_ne!(angstrom, alphafold);

        cache.insert(angstrom.clone(), sample_structure());
        cache.insert(nanometre.clone(), sample_structure());
        cache.insert(alphafold.clone(), sample_structure());
        assert_eq!(
            cache._len(),
            3,
            "option variants must not replace each other"
        );
        assert!(cache.get(&angstrom).is_some());
        assert!(cache.get(&nanometre).is_some());
        assert!(cache.get(&alphafold).is_some());
    }

    #[test]
    fn test_same_path_changed_bytes_is_a_miss() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "structure.pdb", ATOM_A);
        let mut cache = FormatCache::new(4);

        let before = key(&path);
        cache.insert(before.clone(), sample_structure());
        assert!(cache.get(&before).is_some(), "unchanged file must hit");

        std::fs::write(&path, ATOM_B).unwrap();
        let after = key(&path);
        assert!(
            cache.get(&after).is_none(),
            "same path with different bytes must not return the old entry"
        );

        // Storing the new parse replaces the stale entry for the same file.
        cache.insert(after.clone(), sample_structure());
        assert_eq!(cache._len(), 1);
        assert!(cache.get(&before).is_none());
        assert!(cache.get(&after).is_some());
    }

    #[test]
    fn test_same_size_same_mtime_changed_bytes_is_a_miss() {
        // Length and mtime both match; only the content hash can tell these apart.
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "same_size.pdb", ATOM_A);
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        let mut cache = FormatCache::new(4);

        let before = key(&path);
        cache.insert(before.clone(), sample_structure());

        std::fs::write(&path, ATOM_B).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(mtime)
            .unwrap();
        let after = key(&path);
        assert_ne!(before, after);
        assert!(cache.get(&after).is_none());
    }

    #[test]
    fn test_identical_bytes_touch_is_a_hit() {
        // An mtime change with identical bytes must not invalidate the entry.
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "touched.pdb", ATOM_A);
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        let mut cache = FormatCache::new(4);
        cache.insert(key(&path), sample_structure());

        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(mtime + Duration::from_secs(3600))
            .unwrap();
        assert!(cache.get(&key(&path)).is_some());
    }

    #[test]
    fn test_unchanged_file_is_a_hit() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "unchanged.pdb", ATOM_A);
        let mut cache = FormatCache::new(4);
        cache.insert(key(&path), sample_structure());

        let again = key(&path);
        assert!(cache.get(&again).is_some());
    }

    #[test]
    fn test_key_for_missing_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.pdb");
        assert!(CacheKey::for_file(&missing, OPTS).is_err());
    }

    #[test]
    fn test_options_key_is_deterministic_and_field_sensitive() {
        let base = OutputSpec::default();
        assert_eq!(options_key(&base), options_key(&OutputSpec::default()));

        let mut caching_only = OutputSpec::default();
        caching_only.enable_caching = !base.enable_caching;
        assert_eq!(
            options_key(&base),
            options_key(&caching_only),
            "enable_caching only decides storage, so it is not keyed"
        );

        let mut other = OutputSpec::default();
        other.remove_solvent = !base.remove_solvent;
        assert_ne!(options_key(&base), options_key(&other));
    }
}
