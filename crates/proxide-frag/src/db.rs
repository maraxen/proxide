//! Fragment database: storage and builder.
//!
//! [`FragmentDbBuilder`] accepts raw [`Fragment<N, Raw>`] values, centers them
//! at build time, computes their squared norms, and collects them into an
//! immutable [`FragmentDb<N>`] ready for parallel search.

use crate::fragment::{AlreadyCenteredError, Fragment, Raw};

// ---------------------------------------------------------------------------
// SourceLabel
// ---------------------------------------------------------------------------

/// Identifies the structural origin of a database fragment.
///
/// Encodes PDB entry, chain identifier, and residue range in a compact,
/// copy-friendly form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SourceLabel {
    /// First 4 bytes of the PDB ID, zero-padded (ASCII).
    pub pdb_id: [u8; 4],
    /// Chain identifier as a single ASCII byte.
    pub chain: u8,
    /// Sequence number of the first residue.
    pub start_res: i32,
    /// Insertion code of the first residue (ASCII; space = none).
    pub start_icode: u8,
    /// Sequence number of the last residue.
    pub end_res: i32,
    /// Insertion code of the last residue (ASCII; space = none).
    pub end_icode: u8,
}

impl SourceLabel {
    /// Construct a new label.
    ///
    /// `pdb_id` is copied up to 4 bytes, zero-padded if shorter.
    pub fn new(
        pdb_id: &str,
        chain: char,
        start_res: i32,
        start_icode: char,
        end_res: i32,
        end_icode: char,
    ) -> Self {
        let mut id_bytes = [0u8; 4];
        for (i, b) in pdb_id.bytes().take(4).enumerate() {
            id_bytes[i] = b;
        }
        Self {
            pdb_id: id_bytes,
            chain: chain as u8,
            start_res,
            start_icode: start_icode as u8,
            end_res,
            end_icode: end_icode as u8,
        }
    }
}

// ---------------------------------------------------------------------------
// FragmentDbEntry (crate-private)
// ---------------------------------------------------------------------------

/// Internal storage record for a single centered fragment in the database.
pub(crate) struct FragmentDbEntry<const N: usize> {
    /// Centered backbone coordinates `[residue][atom][xyz]`.
    pub(crate) coords: [[[f32; 3]; 4]; N],
    /// Centroid of the original raw fragment (to reconstruct world-frame geometry).
    /// Stored for future use; not accessed during search.
    #[allow(dead_code)]
    pub(crate) centroid: [f32; 3],
    /// Precomputed `||coords||²` for the fast inner-product RMSD formula.
    pub(crate) norm_sq: f32,
    /// Structural provenance of this fragment.
    pub(crate) label: SourceLabel,
}

// ---------------------------------------------------------------------------
// FragmentDb
// ---------------------------------------------------------------------------

/// Immutable database of centered backbone fragments, ready for parallel search.
///
/// Construct via [`FragmentDbBuilder`].
pub struct FragmentDb<const N: usize> {
    pub(crate) entries: Vec<FragmentDbEntry<N>>,
}

impl<const N: usize> FragmentDb<N> {
    /// Number of fragments in the database.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the database contains no fragments.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// FragmentDbBuilder
// ---------------------------------------------------------------------------

/// Builder for [`FragmentDb<N>`].
///
/// Accepts raw fragments (which it centers on insertion), then builds an
/// immutable database.
pub struct FragmentDbBuilder<const N: usize> {
    entries: Vec<FragmentDbEntry<N>>,
}

impl<const N: usize> FragmentDbBuilder<N> {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a raw fragment to the database.
    ///
    /// The fragment is centered immediately; returns
    /// [`AlreadyCenteredError`] if its centroid is already at the origin.
    pub fn add_fragment(
        &mut self,
        fragment: Fragment<N, Raw>,
        label: SourceLabel,
    ) -> Result<(), AlreadyCenteredError> {
        let (centered, centroid) = fragment.center()?;
        let norm_sq = centered.norm_sq();
        self.entries.push(FragmentDbEntry {
            coords: centered.coords,
            centroid,
            norm_sq,
            label,
        });
        Ok(())
    }

    /// Consume the builder and produce an immutable [`FragmentDb<N>`].
    pub fn build(self) -> FragmentDb<N> {
        FragmentDb {
            entries: self.entries,
        }
    }
}

impl<const N: usize> Default for FragmentDbBuilder<N> {
    fn default() -> Self {
        Self::new()
    }
}
