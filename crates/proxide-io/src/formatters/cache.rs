//! LRU Cache for formatted structure outputs
//!
//! Provides caching of formatted structure data to avoid re-formatting
//! the same structure multiple times. Entries are keyed on the identity of
//! the file they were parsed from (canonical path plus a content
//! fingerprint), so a file overwritten at the same path is parsed again
//! instead of being served from the old entry.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

use proxide_core::spec::CoordFormat;

/// Identity of a file's bytes. Length and mtime are the cheap pre-check; the
/// content hash is what catches an edit that keeps both (same size, same
/// mtime). The hash is std's `DefaultHasher`: it is not stable across Rust
/// releases, which is fine because the cache lives only in this process.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct FileFingerprint {
    len: u64,
    /// Nanoseconds since the Unix epoch; 0 when the platform reports no mtime.
    mtime_ns: i128,
    content_hash: u64,
}

impl FileFingerprint {
    fn of_file(path: &Path) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let mtime_ns = std::fs::metadata(path)?
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos() as i128);
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        Ok(Self {
            len: bytes.len() as u64,
            mtime_ns,
            content_hash: hasher.finish(),
        })
    }
}

/// Key for a cached parse: the source file (canonical path and content
/// fingerprint) plus the output format and flags.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CacheKey {
    path: PathBuf,
    fingerprint: FileFingerprint,
    format: CoordFormat,
    remove_solvent: bool,
    include_hetatm: bool,
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path.hash(state);
        self.fingerprint.hash(state);
        (self.format as u8).hash(state);
        self.remove_solvent.hash(state);
        self.include_hetatm.hash(state);
    }
}

impl CacheKey {
    /// Key for the parse of `path` as its bytes are on disk now. Reads the
    /// file to fingerprint it. Errors if the file cannot be resolved or read:
    /// a parse that cannot be fingerprinted must not be cached.
    pub fn for_file(
        path: impl AsRef<Path>,
        format: CoordFormat,
        remove_solvent: bool,
        include_hetatm: bool,
    ) -> io::Result<Self> {
        let path = std::fs::canonicalize(path)?;
        let fingerprint = FileFingerprint::of_file(&path)?;
        Ok(Self {
            path,
            fingerprint,
            format,
            remove_solvent,
            include_hetatm,
        })
    }

    /// Same file, format and flags, regardless of content.
    fn same_slot(&self, other: &Self) -> bool {
        self.path == other.path
            && self.format == other.format
            && self.remove_solvent == other.remove_solvent
            && self.include_hetatm == other.include_hetatm
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

/// Get a structure from the global cache
pub fn get_cached(key: &CacheKey) -> Option<CachedStructure> {
    let mut cache = GLOBAL_CACHE.lock().unwrap();
    cache.get(key).cloned()
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

    const ATOM_A: &[u8] = b"ATOM      1  CA  ALA A   1       1.000   0.000   0.000\n";
    const ATOM_B: &[u8] = b"ATOM      1  CA  ALA A   1       9.000   0.000   0.000\n";

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

    fn key(path: &Path, format: CoordFormat) -> CacheKey {
        CacheKey::for_file(path, format, true, false).unwrap()
    }

    #[test]
    fn test_cache_basic() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = write_file(&dir, "test1.pdb", ATOM_A);
        let p2 = write_file(&dir, "test2.pdb", ATOM_A);
        let p3 = write_file(&dir, "test3.pdb", ATOM_A);
        let mut cache = FormatCache::new(2);

        let key1 = key(&p1, CoordFormat::Atom37);
        let key2 = key(&p2, CoordFormat::Atom37);
        let key3 = key(&p3, CoordFormat::Atom37);

        cache.insert(key1.clone(), sample_structure());
        cache.insert(key2.clone(), sample_structure());

        assert_eq!(cache._len(), 2);
        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_some());

        // Insert third, should evict LRU
        // Order after inserts: key1, key2
        // After get(key1): key2, key1
        // After get(key2): key1, key2
        // So key1 is LRU and should be evicted when key3 is inserted
        cache.insert(key3.clone(), sample_structure());

        assert_eq!(cache._len(), 2);
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_some());
        assert!(cache.get(&key3).is_some());
    }

    #[test]
    fn test_cache_key_different_formats() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "test.pdb", ATOM_A);
        let key1 = key(&path, CoordFormat::Atom37);
        let key2 = key(&path, CoordFormat::Atom14);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_same_path_changed_bytes_is_a_miss() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "structure.pdb", ATOM_A);
        let mut cache = FormatCache::new(4);

        let before = key(&path, CoordFormat::Atom37);
        cache.insert(before.clone(), sample_structure());
        assert!(cache.get(&before).is_some(), "unchanged file must hit");

        std::fs::write(&path, ATOM_B).unwrap();
        let after = key(&path, CoordFormat::Atom37);
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

        let before = key(&path, CoordFormat::Atom37);
        cache.insert(before.clone(), sample_structure());

        std::fs::write(&path, ATOM_B).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(mtime)
            .unwrap();
        let after = key(&path, CoordFormat::Atom37);
        assert_ne!(before, after);
        assert!(cache.get(&after).is_none());
    }

    #[test]
    fn test_unchanged_file_is_a_hit() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(&dir, "unchanged.pdb", ATOM_A);
        let mut cache = FormatCache::new(4);
        cache.insert(key(&path, CoordFormat::Atom37), sample_structure());

        let again = key(&path, CoordFormat::Atom37);
        assert!(cache.get(&again).is_some());
    }

    #[test]
    fn test_key_for_missing_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.pdb");
        assert!(CacheKey::for_file(&missing, CoordFormat::Atom37, true, false).is_err());
    }
}
