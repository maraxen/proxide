//! LRU Cache for formatted structure outputs
//!
//! Provides caching of formatted structure data to avoid re-formatting
//! the same structure multiple times.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use crate::spec::CoordFormat;

/// Cache key combining file path hash and format specification
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CacheKey {
    path_hash: u64,
    format: CoordFormat,
    remove_solvent: bool,
    include_hetatm: bool,
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path_hash.hash(state);
        (self.format as u8).hash(state);
        self.remove_solvent.hash(state);
        self.include_hetatm.hash(state);
    }
}

impl CacheKey {
    pub fn new(
        path: &str,
        format: CoordFormat,
        remove_solvent: bool,
        include_hetatm: bool,
    ) -> Self {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        path.hash(&mut hasher);
        let path_hash = hasher.finish();

        Self {
            path_hash,
            format,
            remove_solvent,
            include_hetatm,
        }
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
}

impl CachedStructure {
    /// Convert to Python dictionary
    pub fn to_py_dict(&self, py: pyo3::Python) -> pyo3::PyResult<pyo3::PyObject> {
        use numpy::PyArray1;
        use pyo3::prelude::*;
        use pyo3::types::PyDict;

        let dict = PyDict::new_bound(py);

        // Convert to NumPy arrays (zero-copy when possible)
        dict.set_item(
            "coordinates",
            PyArray1::from_vec_bound(py, self.coordinates.clone()),
        )?;
        dict.set_item(
            "atom_mask",
            PyArray1::from_vec_bound(py, self.atom_mask.clone()),
        )?;
        dict.set_item("aatype", PyArray1::from_vec_bound(py, self.aatype.clone()))?;
        dict.set_item(
            "residue_index",
            PyArray1::from_vec_bound(py, self.residue_index.clone()),
        )?;
        dict.set_item(
            "chain_index",
            PyArray1::from_vec_bound(py, self.chain_index.clone()),
        )?;

        if let Some(ref names) = self.atom_names {
            let atom_names_list: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            dict.set_item("atom_names", atom_names_list)?;
        }

        if let Some(shape) = self.coord_shape {
            dict.set_item("coord_shape", shape)?;
        }

        Ok(dict.into())
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

    #[test]
    fn test_cache_basic() {
        let mut cache = FormatCache::new(2);

        let key1 = CacheKey::new("test1.pdb", CoordFormat::Atom37, true, false);
        let key2 = CacheKey::new("test2.pdb", CoordFormat::Atom37, true, false);
        let key3 = CacheKey::new("test3.pdb", CoordFormat::Atom37, true, false);

        let value = CachedStructure {
            coordinates: vec![1.0, 2.0, 3.0],
            atom_mask: vec![1.0],
            aatype: vec![0],
            residue_index: vec![1],
            chain_index: vec![0],
            _num_residues: 1,
            atom_names: None,
            coord_shape: None,
        };

        cache.insert(key1.clone(), value.clone());
        cache.insert(key2.clone(), value.clone());

        assert_eq!(cache._len(), 2);
        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_some());

        // Insert third, should evict LRU
        // Order after inserts: key1, key2
        // After get(key1): key2, key1
        // After get(key2): key1, key2
        // So key1 is LRU and should be evicted when key3 is inserted
        cache.insert(key3.clone(), value);

        assert_eq!(cache._len(), 2);
        // key1 should be evicted (was LRU after accessing key1 then key2)
        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_some());
        assert!(cache.get(&key3).is_some());
    }

    #[test]
    fn test_cache_key_different_formats() {
        let key1 = CacheKey::new("test.pdb", CoordFormat::Atom37, true, false);
        let key2 = CacheKey::new("test.pdb", CoordFormat::Atom14, true, false);

        assert_ne!(key1, key2);
    }
}
