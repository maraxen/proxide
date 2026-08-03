//! Persistence layer for FragmentDb: serialization and deserialization via bincode.

use crate::db::{FragmentDb, FragmentDbEntry, SourceLabel};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: u32 = 0x5058_4D44; // "PXMD" in ASCII
const VERSION: u16 = 1;

// ---------------------------------------------------------------------------
// Error Type
// ---------------------------------------------------------------------------

/// Errors that can occur during persistence operations.
#[derive(Debug, Error)]
pub enum PersistError {
    /// I/O error (file not found, permission denied, etc.)
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Bincode error (encoding or decoding)
    #[error("bincode error: {0}")]
    Bincode(#[from] bincode::Error),

    /// Magic number mismatch (file is not a proxide-frag database)
    #[error("bad magic number: {found:#010x} (expected {MAGIC:#010x})")]
    BadMagic { found: u32 },

    /// Version mismatch (file format incompatible)
    #[error("version mismatch: found {found}, expected {expected}")]
    VersionMismatch { found: u16, expected: u16 },

    /// Arity mismatch (const generic N mismatch)
    #[error("arity mismatch: expected {expected} residues per fragment, found {found}")]
    ArityMismatch { expected: usize, found: usize },
}

// ---------------------------------------------------------------------------
// Serde Mirror Types
// ---------------------------------------------------------------------------

/// Serializable representation of a fragment entry.
/// Used to work around serde's inability to handle const-generic fixed arrays.
#[derive(serde::Serialize, serde::Deserialize)]
struct PersistedEntry {
    /// Backbone coordinates as Vec (flattened from [[[f32;3];4];N]).
    coords: Vec<[[f32; 3]; 4]>,
    /// Centroid of the original raw fragment.
    centroid: [f32; 3],
    /// Precomputed squared norm.
    norm_sq: f32,
    /// Source label.
    label: SourceLabel,
}

/// Serializable representation of the entire database.
#[derive(serde::Serialize, serde::Deserialize)]
struct PersistedDb {
    /// Magic number identifying this file format.
    magic: u32,
    /// Version of this file format.
    version: u16,
    /// Number of residues per fragment (const generic N).
    residues_n: u32,
    /// All fragments in the database.
    entries: Vec<PersistedEntry>,
}

// ---------------------------------------------------------------------------
// Persistence Implementation
// ---------------------------------------------------------------------------

impl<const N: usize> FragmentDb<N> {
    /// Serialize and save this database to a bincode file.
    ///
    /// The file begins with a magic number and version for validation.
    /// Returns `PersistError::Io` if the file cannot be written,
    /// and `PersistError::Bincode` if serialization fails.
    pub fn save(&self, path: &Path) -> Result<(), PersistError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Convert entries to PersistedEntry format.
        let entries: Vec<PersistedEntry> = self
            .entries
            .iter()
            .map(|entry| PersistedEntry {
                coords: entry.coords.to_vec(),
                centroid: entry.centroid,
                norm_sq: entry.norm_sq,
                label: entry.label.clone(),
            })
            .collect();

        let persisted = PersistedDb {
            magic: MAGIC,
            version: VERSION,
            residues_n: N as u32,
            entries,
        };

        bincode::serialize_into(&mut writer, &persisted)?;

        Ok(())
    }

    /// Deserialize and load a database from a bincode file.
    ///
    /// Validates the magic number and version before deserializing.
    /// Returns appropriate errors if validation fails or deserialization fails.
    pub fn load(path: &Path) -> Result<Self, PersistError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let persisted: PersistedDb = bincode::deserialize_from(reader)?;

        // Validate magic number.
        if persisted.magic != MAGIC {
            return Err(PersistError::BadMagic {
                found: persisted.magic,
            });
        }

        // Validate version.
        if persisted.version != VERSION {
            return Err(PersistError::VersionMismatch {
                found: persisted.version,
                expected: VERSION,
            });
        }

        // Validate arity.
        let expected_residues = N as u32;
        if persisted.residues_n != expected_residues {
            return Err(PersistError::ArityMismatch {
                expected: N,
                found: persisted.residues_n as usize,
            });
        }

        // Reconstruct entries from PersistedEntry.
        let mut entries = Vec::with_capacity(persisted.entries.len());
        for persisted_entry in persisted.entries {
            // Convert coords Vec to fixed array — safe, length-checked.
            // `TryFrom<Vec<T>> for [T; N]` hands back the Vec on a length
            // mismatch, which we surface as an ArityMismatch.
            let coords: [[[f32; 3]; 4]; N] =
                persisted_entry
                    .coords
                    .try_into()
                    .map_err(|v: Vec<[[f32; 3]; 4]>| PersistError::ArityMismatch {
                        expected: N,
                        found: v.len(),
                    })?;

            entries.push(FragmentDbEntry {
                coords,
                centroid: persisted_entry.centroid,
                norm_sq: persisted_entry.norm_sq,
                label: persisted_entry.label,
            });
        }

        Ok(FragmentDb { entries })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::FragmentDbBuilder;
    use crate::fragment::Fragment;
    use std::env;

    #[test]
    fn test_roundtrip() {
        // Create a small database with a few fragments.
        let mut builder = FragmentDbBuilder::<5>::new();

        // Fragment 1: Simple non-degenerate coordinates.
        let coords1 = [
            [
                [0.0, 0.0, 10.0],
                [1.0, 0.0, 10.0],
                [1.0, 1.0, 10.0],
                [1.0, 1.0, 11.0],
            ],
            [
                [2.0, 0.0, 10.0],
                [3.0, 0.0, 10.0],
                [3.0, 1.0, 10.0],
                [3.0, 1.0, 11.0],
            ],
            [
                [4.0, 0.0, 10.0],
                [5.0, 0.0, 10.0],
                [5.0, 1.0, 10.0],
                [5.0, 1.0, 11.0],
            ],
            [
                [6.0, 0.0, 10.0],
                [7.0, 0.0, 10.0],
                [7.0, 1.0, 10.0],
                [7.0, 1.0, 11.0],
            ],
            [
                [8.0, 0.0, 10.0],
                [9.0, 0.0, 10.0],
                [9.0, 1.0, 10.0],
                [9.0, 1.0, 11.0],
            ],
        ];
        let frag1 = Fragment::new(coords1);
        let label1 = SourceLabel::new("1abc", 'A', 1, ' ', 5, ' ');
        builder.add_fragment(frag1, label1.clone()).unwrap();

        // Fragment 2: Different coordinates and label.
        let coords2 = [
            [
                [0.5, 0.5, 20.0],
                [1.5, 0.5, 20.0],
                [1.5, 1.5, 20.0],
                [1.5, 1.5, 21.0],
            ],
            [
                [2.5, 0.5, 20.0],
                [3.5, 0.5, 20.0],
                [3.5, 1.5, 20.0],
                [3.5, 1.5, 21.0],
            ],
            [
                [4.5, 0.5, 20.0],
                [5.5, 0.5, 20.0],
                [5.5, 1.5, 20.0],
                [5.5, 1.5, 21.0],
            ],
            [
                [6.5, 0.5, 20.0],
                [7.5, 0.5, 20.0],
                [7.5, 1.5, 20.0],
                [7.5, 1.5, 21.0],
            ],
            [
                [8.5, 0.5, 20.0],
                [9.5, 0.5, 20.0],
                [9.5, 1.5, 20.0],
                [9.5, 1.5, 21.0],
            ],
        ];
        let frag2 = Fragment::new(coords2);
        let label2 = SourceLabel::new("2xyz", 'B', 10, 'A', 14, 'B');
        builder.add_fragment(frag2, label2.clone()).unwrap();

        let db = builder.build();

        // Save to temp file.
        let temp_path = env::temp_dir().join("proxide_frag_persist_roundtrip.bin");
        db.save(&temp_path).expect("failed to save database");

        // Load it back.
        let loaded: FragmentDb<5> = FragmentDb::load(&temp_path).expect("failed to load database");

        // Verify basic properties.
        assert_eq!(loaded.len(), 2, "database should have 2 entries");

        // Verify first entry.
        let entry1 = &loaded.entries[0];
        assert_eq!(entry1.label, label1, "first label should match");
        // The DB stores *centered* coords, so the round-trip must reproduce the
        // in-memory centered values (not the raw input `coords1`).
        assert_eq!(
            entry1.coords, db.entries[0].coords,
            "first coords should round-trip exactly"
        );
        assert_eq!(
            entry1.norm_sq, db.entries[0].norm_sq,
            "first norm_sq should match"
        );

        // Verify second entry.
        let entry2 = &loaded.entries[1];
        assert_eq!(entry2.label, label2, "second label should match");
        assert_eq!(
            entry2.coords, db.entries[1].coords,
            "second coords should round-trip exactly"
        );
        assert_eq!(
            entry2.norm_sq, db.entries[1].norm_sq,
            "second norm_sq should match"
        );

        // Clean up.
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_arity_mismatch_on_load() {
        // Create a database with N=5.
        let mut builder = FragmentDbBuilder::<5>::new();
        let coords = [
            [
                [0.0, 0.0, 10.0],
                [1.0, 0.0, 10.0],
                [1.0, 1.0, 10.0],
                [1.0, 1.0, 11.0],
            ],
            [
                [2.0, 0.0, 10.0],
                [3.0, 0.0, 10.0],
                [3.0, 1.0, 10.0],
                [3.0, 1.0, 11.0],
            ],
            [
                [4.0, 0.0, 10.0],
                [5.0, 0.0, 10.0],
                [5.0, 1.0, 10.0],
                [5.0, 1.0, 11.0],
            ],
            [
                [6.0, 0.0, 10.0],
                [7.0, 0.0, 10.0],
                [7.0, 1.0, 10.0],
                [7.0, 1.0, 11.0],
            ],
            [
                [8.0, 0.0, 10.0],
                [9.0, 0.0, 10.0],
                [9.0, 1.0, 10.0],
                [9.0, 1.0, 11.0],
            ],
        ];
        let frag = Fragment::new(coords);
        let label = SourceLabel::new("1abc", 'A', 1, ' ', 5, ' ');
        builder.add_fragment(frag, label).unwrap();

        let db = builder.build();

        // Save to temp file.
        let temp_path = env::temp_dir().join("proxide_frag_persist_arity_mismatch.bin");
        db.save(&temp_path).expect("failed to save database");

        // Try to load with N=4 (wrong arity).
        let result: Result<FragmentDb<4>, _> = FragmentDb::load(&temp_path);
        assert!(result.is_err(), "loading with wrong arity should fail");

        match result {
            Err(PersistError::ArityMismatch {
                expected: 4,
                found: 5,
            }) => {
                // Correct error.
            }
            _ => panic!("expected ArityMismatch error"),
        }

        // Clean up.
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_bad_magic_number() {
        // Create a corrupt file with wrong magic number.
        let temp_path = env::temp_dir().join("proxide_frag_persist_bad_magic.bin");

        let bad_db = PersistedDb {
            magic: 0xDEADBEEF,
            version: VERSION,
            residues_n: 5,
            entries: vec![],
        };

        let file = File::create(&temp_path).expect("failed to create temp file");
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &bad_db).expect("failed to encode");
        drop(writer); // flush buffered bytes to disk before reading back

        // Try to load it.
        let result: Result<FragmentDb<5>, _> = FragmentDb::load(&temp_path);
        assert!(result.is_err(), "loading with bad magic should fail");

        match result {
            Err(PersistError::BadMagic { found: 0xDEADBEEF }) => {
                // Correct error.
            }
            _ => panic!("expected BadMagic error"),
        }

        // Clean up.
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_version_mismatch() {
        // Create a file with wrong version number.
        let temp_path = env::temp_dir().join("proxide_frag_persist_version_mismatch.bin");

        let bad_db = PersistedDb {
            magic: MAGIC,
            version: 99,
            residues_n: 5,
            entries: vec![],
        };

        let file = File::create(&temp_path).expect("failed to create temp file");
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &bad_db).expect("failed to encode");
        drop(writer); // flush buffered bytes to disk before reading back

        // Try to load it.
        let result: Result<FragmentDb<5>, _> = FragmentDb::load(&temp_path);
        assert!(result.is_err(), "loading with wrong version should fail");

        match result {
            Err(PersistError::VersionMismatch {
                found: 99,
                expected: 1,
            }) => {
                // Correct error.
            }
            _ => panic!("expected VersionMismatch error"),
        }

        // Clean up.
        let _ = std::fs::remove_file(&temp_path);
    }
}
