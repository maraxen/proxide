//! Projection from AtomicSystem to MPNNBatch
//!
//! Converts full atomic data to residue-level features suitable for
//! protein structure learning models.

use crate::geometry::neighbors::find_k_nearest_neighbors;
use crate::geometry::radial_basis::compute_radial_basis;
use crate::processing::noising::gaussian_backbone_noise_5atom;
use crate::processing::residues::ProcessedStructure;
use crate::spec::OutputFormatTarget;

/// Result of projecting to MPNN batch format
#[derive(Debug, Clone)]
pub struct MPNNBatchResult {
    /// Amino acid type per residue (0-20). Shape: (N_res,)
    pub aatype: Vec<i32>,
    /// Residue index (PDB numbering). Shape: (N_res,)
    pub residue_index: Vec<i32>,
    /// Chain index per residue. Shape: (N_res,)
    pub chain_index: Vec<i32>,
    /// Validity mask. Shape: (N_res,)
    pub mask: Vec<f32>,
    /// RBF features. Flattened (N_res, K, 400)
    pub rbf_features: Vec<f32>,
    /// Neighbor indices. Flattened (N_res, K)
    pub neighbor_indices: Vec<i32>,
    /// Optional physics features. (N_res, F_phys) flattened
    pub physics_features: Option<Vec<f32>>,
    /// Number of residues
    pub n_residues: usize,
    /// Number of neighbors per residue
    pub n_neighbors: usize,
}

/// Error type for projection failures
#[derive(Debug)]
pub enum ProjectionError {
    /// No protein atoms found in structure
    NoProteinAtoms,
}

impl std::fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProjectionError::NoProteinAtoms => write!(f, "No protein atoms found in structure"),
        }
    }
}

impl std::error::Error for ProjectionError {}

/// Project ProcessedStructure to MPNNBatch format
///
/// # Arguments
/// * `structure` - ProcessedStructure with residue grouping
/// * `num_neighbors` - Number of K-nearest neighbors per residue
/// * `noise_std` - Optional Gaussian noise standard deviation for coordinates
/// * `noise_seed` - Random seed for noising
/// * `compute_physics` - Whether to compute physics features
///
/// # Returns
/// MPNNBatchResult with all features computed
pub fn project_to_mpnn_batch(
    structure: &ProcessedStructure,
    num_neighbors: usize,
    noise_std: Option<f32>,
    noise_seed: u64,
    compute_physics: bool,
) -> Result<MPNNBatchResult, ProjectionError> {
    let n_residues = structure.num_residues;

    if n_residues == 0 {
        return Err(ProjectionError::NoProteinAtoms);
    }

    // Extract backbone coordinates (N_res, 5, 3)
    let mut backbone = structure.extract_backbone_coords(OutputFormatTarget::Mpnn);

    // Apply noising if requested (before RBF computation)
    if let Some(std) = noise_std {
        gaussian_backbone_noise_5atom(&mut backbone, std, noise_seed);
    }

    // Extract CA coordinates for neighbor search
    let ca_coords = structure.extract_ca_coords();

    // Find K nearest neighbors
    let neighbors = find_k_nearest_neighbors(&ca_coords, num_neighbors);

    // Compute RBF features
    let rbf_features = compute_radial_basis(&backbone, &neighbors);

    // Build aatype, residue_index, chain_index, mask
    let mut aatype = Vec::with_capacity(n_residues);
    let mut residue_index = Vec::with_capacity(n_residues);
    let mut chain_index = Vec::with_capacity(n_residues);
    let mut mask = Vec::with_capacity(n_residues);

    for res in &structure.residue_info {
        aatype.push(res.res_type as i32);
        residue_index.push(res.res_id);

        // Map chain_id to numeric index
        let chain_idx = structure
            .chain_indices
            .get(&res.chain_id)
            .copied()
            .unwrap_or(0);
        chain_index.push(chain_idx as i32);

        // All residues are valid (mask = 1.0)
        mask.push(1.0);
    }

    // Flatten neighbor indices
    let effective_k = neighbors.first().map(|v| v.len()).unwrap_or(0);
    let mut neighbor_indices: Vec<i32> = Vec::with_capacity(n_residues * effective_k);
    for res_neighbors in &neighbors {
        for &idx in res_neighbors {
            neighbor_indices.push(idx as i32);
        }
        // Pad if this residue has fewer neighbors
        let pad_count = effective_k - res_neighbors.len();
        if pad_count > 0 {
            neighbor_indices.extend(std::iter::repeat_n(0, pad_count));
        }
    }

    // Compute physics features if requested
    let physics_features = if compute_physics {
        let charges = structure.extract_backbone_charges(OutputFormatTarget::Mpnn);
        let sigmas = structure.extract_backbone_sigmas(OutputFormatTarget::Mpnn);
        let epsilons = structure.extract_backbone_epsilons(OutputFormatTarget::Mpnn);

        // Combine into per-residue features: 5 atoms × 3 features = 15 per residue
        let mut features = Vec::with_capacity(n_residues * 15);
        for i in 0..n_residues {
            for j in 0..5 {
                let idx = i * 5 + j;
                features.push(charges[idx]);
                features.push(sigmas[idx]);
                features.push(epsilons[idx]);
            }
        }
        Some(features)
    } else {
        None
    };

    Ok(MPNNBatchResult {
        aatype,
        residue_index,
        chain_index,
        mask,
        rbf_features,
        neighbor_indices,
        physics_features,
        n_residues,
        n_neighbors: effective_k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{AtomRecord, RawAtomData};

    fn make_test_structure() -> ProcessedStructure {
        let mut raw = RawAtomData::with_capacity(10);

        // Add 2 residues (ALA at positions 0,0,0 and GLY at 10,0,0)
        let atoms = ["N", "CA", "C", "CB", "O"];
        for (atom_idx, atom_name) in atoms.iter().enumerate() {
            raw.add_atom(AtomRecord {
                serial: atom_idx as i32 + 1,
                atom_name: atom_name.to_string(),
                alt_loc: ' ',
                res_name: "ALA".to_string(),
                chain_id: "A".to_string(),
                res_seq: 1,
                i_code: ' ',
                x: atom_idx as f32,
                y: 0.0,
                z: 0.0,
                occupancy: 1.0,
                temp_factor: 20.0,
                element: "C".to_string(),
                charge: Some(0.1),
                radius: None,
                is_hetatm: false,
            });
        }

        for (atom_idx, atom_name) in atoms.iter().enumerate() {
            raw.add_atom(AtomRecord {
                serial: (atom_idx + 5) as i32 + 1,
                atom_name: atom_name.to_string(),
                alt_loc: ' ',
                res_name: "GLY".to_string(),
                chain_id: "A".to_string(),
                res_seq: 2,
                i_code: ' ',
                x: 10.0 + atom_idx as f32,
                y: 0.0,
                z: 0.0,
                occupancy: 1.0,
                temp_factor: 20.0,
                element: "C".to_string(),
                charge: Some(-0.1),
                radius: None,
                is_hetatm: false,
            });
        }

        ProcessedStructure::from_raw(raw).unwrap()
    }

    #[test]
    fn test_project_basic() {
        let structure = make_test_structure();
        let result = project_to_mpnn_batch(&structure, 30, None, 0, false).unwrap();

        assert_eq!(result.n_residues, 2);
        assert_eq!(result.aatype.len(), 2);
        assert_eq!(result.mask.len(), 2);
        assert!(result.physics_features.is_none());
    }

    #[test]
    fn test_project_with_physics() {
        let structure = make_test_structure();
        let result = project_to_mpnn_batch(&structure, 30, None, 0, true).unwrap();

        assert!(result.physics_features.is_some());
        let features = result.physics_features.unwrap();
        // 2 residues × 5 atoms × 3 features = 30
        assert_eq!(features.len(), 30);
    }

    #[test]
    fn test_project_with_noise() {
        let structure = make_test_structure();

        // Without noise
        let result_no_noise = project_to_mpnn_batch(&structure, 1, None, 0, false).unwrap();

        // With noise
        let result_with_noise = project_to_mpnn_batch(&structure, 1, Some(1.0), 42, false).unwrap();

        // RBF features should differ
        assert_ne!(result_no_noise.rbf_features, result_with_noise.rbf_features);
    }

    #[test]
    fn test_neighbor_indices_shape() {
        let structure = make_test_structure();
        let k = 1; // Only 1 neighbor since we have 2 residues
        let result = project_to_mpnn_batch(&structure, k, None, 0, false).unwrap();

        // 2 residues × 1 neighbor
        assert_eq!(result.neighbor_indices.len(), 2);
        assert_eq!(result.n_neighbors, 1);
    }
}
