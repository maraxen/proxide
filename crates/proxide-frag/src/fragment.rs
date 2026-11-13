//! Type-state Fragment — Raw vs Centered backbone coordinate arrays.
//!
//! A [`Fragment<N, State>`] holds N residues of backbone atoms (N, CA, C, O),
//! each with a 3D coordinate, in `f32` precision. The `State` type parameter
//! encodes whether the fragment has been centered (centroid subtracted) at the
//! type level, preventing callers from accidentally passing a raw fragment where
//! a centered one is required.

use std::marker::PhantomData;
use thiserror::Error;

// ---------------------------------------------------------------------------
// State markers
// ---------------------------------------------------------------------------

/// Marker: fragment coordinates are in the original (world-frame) coordinate system.
#[derive(Debug, Clone, Copy)]
pub struct Raw;

/// Marker: fragment coordinates have been shifted so their centroid is at the origin.
#[derive(Debug, Clone, Copy)]
pub struct Centered;

// ---------------------------------------------------------------------------
// BackboneAtom
// ---------------------------------------------------------------------------

/// The four canonical backbone atom types in a peptide residue.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackboneAtom {
    N = 0,
    CA = 1,
    C = 2,
    O = 3,
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

/// An N-residue backbone fragment with 4 atoms per residue (N, CA, C, O).
///
/// `coords[residue][atom][xyz]` — residue index in `0..N`, atom index in
/// `0..4` matching [`BackboneAtom`] ordinals, xyz in `0..3`.
///
/// The `State` parameter is a ZST phantom: use [`Raw`] for freshly extracted
/// coordinates, [`Centered`] after [`Fragment::center`] has been called.
#[derive(Debug, Clone)]
pub struct Fragment<const N: usize, State> {
    pub coords: [[[f32; 3]; 4]; N],
    _state: PhantomData<State>,
}

impl<const N: usize> Fragment<N, Raw> {
    /// Construct a new raw fragment from the given coordinate array.
    pub fn new(coords: [[[f32; 3]; 4]; N]) -> Self {
        Self {
            coords,
            _state: PhantomData,
        }
    }

    /// Center the fragment by subtracting the centroid from every atom coordinate.
    ///
    /// Returns `Err(AlreadyCenteredError)` if the centroid norm is below 1e-6 Å,
    /// indicating the fragment is already centered.
    pub fn center(self) -> Result<(Fragment<N, Centered>, [f32; 3]), AlreadyCenteredError> {
        let n_atoms = (N * 4) as f32;

        // Accumulate centroid sum.
        let mut sum = [0.0_f32; 3];
        for residue in &self.coords {
            for atom in residue {
                sum[0] += atom[0];
                sum[1] += atom[1];
                sum[2] += atom[2];
            }
        }
        let centroid = [sum[0] / n_atoms, sum[1] / n_atoms, sum[2] / n_atoms];

        // Reject if already centered.
        let norm =
            (centroid[0] * centroid[0] + centroid[1] * centroid[1] + centroid[2] * centroid[2])
                .sqrt();
        if norm < 1e-6_f32 {
            return Err(AlreadyCenteredError { norm });
        }

        // Subtract centroid.
        let mut centered_coords = self.coords;
        // Use range loops because N is const generic; iter_mut() over const arrays is not ergonomic.
        #[allow(clippy::needless_range_loop)]
        for r in 0..N {
            #[allow(clippy::needless_range_loop)]
            for at in 0..4 {
                centered_coords[r][at][0] -= centroid[0];
                centered_coords[r][at][1] -= centroid[1];
                centered_coords[r][at][2] -= centroid[2];
            }
        }

        Ok((
            Fragment {
                coords: centered_coords,
                _state: PhantomData,
            },
            centroid,
        ))
    }
}

impl<const N: usize> Fragment<N, Centered> {
    /// Crate-internal constructor — wraps already-centered coords without
    /// re-centering them.  Called by [`crate::search`] to create temporary
    /// views over database entry coordinate arrays.
    pub(crate) fn new_centered(coords: [[[f32; 3]; 4]; N]) -> Self {
        Self {
            coords,
            _state: PhantomData,
        }
    }

    /// Sum of squared magnitudes of all atom coordinate vectors.
    ///
    /// Equals `||X||²` where X is the N×4×3 matrix of centered coordinates.
    /// Pre-computed and cached in [`FragmentDbEntry`] for fast RMSD evaluation.
    ///
    /// [`FragmentDbEntry`]: crate::db::FragmentDbEntry
    pub fn norm_sq(&self) -> f32 {
        let mut acc = 0.0_f32;
        for residue in &self.coords {
            for atom in residue {
                let (x, y, z) = (atom[0], atom[1], atom[2]);
                acc += x * x + y * y + z * z;
            }
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Error returned when [`Fragment::center`] is called on an already-centered
/// fragment (centroid norm < 1e-6 Å).
#[derive(Debug, Error)]
#[error("fragment is already centered (centroid norm {norm:.2e} < 1e-6 Å)")]
pub struct AlreadyCenteredError {
    /// Norm of the computed centroid.
    pub norm: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two residues, integer coordinates, so every sum below is exact in f32.
    /// x sums to 32, y to 32, z to 8 over 8 atoms: centroid (4, 4, 1).
    fn two_residue_fragment() -> Fragment<2, Raw> {
        Fragment::new([
            [
                [1.0, 2.0, 0.0],
                [3.0, 2.0, 0.0],
                [5.0, 2.0, 0.0],
                [7.0, 2.0, 0.0],
            ],
            [
                [1.0, 6.0, 0.0],
                [3.0, 6.0, 0.0],
                [5.0, 6.0, 0.0],
                [7.0, 6.0, 8.0],
            ],
        ])
    }

    #[test]
    fn center_subtracts_the_centroid_of_all_atoms() {
        let (centered, centroid) = two_residue_fragment().center().unwrap();
        assert_eq!(centroid, [4.0, 4.0, 1.0]);
        assert_eq!(
            centered.coords,
            [
                [
                    [-3.0, -2.0, -1.0],
                    [-1.0, -2.0, -1.0],
                    [1.0, -2.0, -1.0],
                    [3.0, -2.0, -1.0]
                ],
                [
                    [-3.0, 2.0, -1.0],
                    [-1.0, 2.0, -1.0],
                    [1.0, 2.0, -1.0],
                    [3.0, 2.0, 7.0]
                ],
            ]
        );
    }

    #[test]
    fn norm_sq_is_the_sum_of_squared_coordinates() {
        // x: 4 * (9 + 1 + 1 + 9) = 40, y: 8 * 4 = 32, z: 7 * 1 + 49 = 56.
        let (centered, _) = two_residue_fragment().center().unwrap();
        assert_eq!(centered.norm_sq(), 128.0);
    }

    #[test]
    fn centering_an_already_centered_fragment_is_refused() {
        let (centered, _) = two_residue_fragment().center().unwrap();
        let raw_again = Fragment::<2, Raw>::new(centered.coords);
        assert!(raw_again.center().is_err());
    }
}
