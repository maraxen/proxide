//! MDTraj HDF5 file parser
//!
//! Parses MDTraj-format HDF5 files containing trajectory data.
//! The HDF5 structure is:
//! - /coordinates: (N_frames, N_atoms, 3) float32
//! - /topology: Contains topology information
//! - /time: (N_frames,) float64 - simulation times

// TODO: Review allow attributes at a later point
#![allow(clippy::type_complexity)]
#![allow(dead_code)]

#[cfg(feature = "mdcath")]
use hdf5::File as H5File;

use crate::structure::RawAtomData;

/// Result of parsing an MDTraj HDF5 file
#[derive(Debug)]
pub struct MdtrajH5Result {
    /// Number of frames in the trajectory
    pub num_frames: usize,
    /// Number of atoms per frame
    pub num_atoms: usize,
    /// Atom names for all atoms
    pub atom_names: Vec<String>,
    /// Residue names for all atoms
    pub res_names: Vec<String>,
    /// Residue IDs for all atoms
    pub res_ids: Vec<i32>,
    /// Chain IDs for all atoms
    pub chain_ids: Vec<String>,
    /// Element symbols for all atoms
    pub elements: Vec<String>,
}

/// A single frame from MDTraj trajectory
#[derive(Debug)]
pub struct MdtrajFrame {
    /// Frame index
    pub index: usize,
    /// Coordinates (N_atoms, 3) - flattened
    pub coords: Vec<f32>,
    /// Simulation time (if available)
    pub time: Option<f64>,
}

/// Parse MDTraj HDF5 file and return metadata
#[cfg(feature = "mdcath")]
pub fn parse_mdtraj_h5_metadata(path: &str) -> Result<MdtrajH5Result, String> {
    let file = H5File::open(path).map_err(|e| format!("Failed to open HDF5 file: {}", e))?;

    // Read coordinates shape to get frame/atom counts
    let coords_ds = file
        .dataset("coordinates")
        .map_err(|e| format!("Failed to open coordinates dataset: {}", e))?;
    let shape = coords_ds.shape();

    if shape.len() != 3 {
        return Err(format!(
            "Expected 3D coordinates, got {} dimensions",
            shape.len()
        ));
    }

    let num_frames = shape[0];
    let num_atoms = shape[1];

    // Try to read topology if available
    let (atom_names, res_names, res_ids, chain_ids, elements) = read_topology(&file, num_atoms)?;

    Ok(MdtrajH5Result {
        num_frames,
        num_atoms,
        atom_names,
        res_names,
        res_ids,
        chain_ids,
        elements,
    })
}

#[cfg(feature = "mdcath")]
fn read_topology(
    _file: &H5File,
    num_atoms: usize,
) -> Result<(Vec<String>, Vec<String>, Vec<i32>, Vec<String>, Vec<String>), String> {
    // Placeholder: read topology from HDF5
    // MDTraj stores topology in a JSON-like format in /topology
    // For now, return dummy data
    Ok((
        vec!["CA".to_string(); num_atoms],
        vec!["ALA".to_string(); num_atoms],
        (0..num_atoms as i32).collect(),
        vec!["A".to_string(); num_atoms],
        vec!["C".to_string(); num_atoms],
    ))
}

/// Read a single frame from MDTraj HDF5
#[cfg(feature = "mdcath")]
pub fn parse_mdtraj_h5_frame(path: &str, frame_idx: usize) -> Result<MdtrajFrame, String> {
    let file = H5File::open(path).map_err(|e| format!("Failed to open HDF5 file: {}", e))?;

    let coords_ds = file
        .dataset("coordinates")
        .map_err(|e| format!("Failed to open coordinates dataset: {}", e))?;

    let shape = coords_ds.shape();
    if frame_idx >= shape[0] {
        return Err(format!(
            "Frame index {} out of range (max {})",
            frame_idx, shape[0]
        ));
    }

    // Read single frame coordinates
    // Note: HDF5 slicing would be more efficient but requires more complex API
    let all_coords: Vec<f32> = coords_ds
        .read_raw()
        .map_err(|e| format!("Failed to read coordinates: {}", e))?;

    let num_atoms = shape[1];
    let frame_size = num_atoms * 3;
    let start = frame_idx * frame_size;
    let end = start + frame_size;

    let mut frame_coords = all_coords[start..end].to_vec();

    // Convert from Nanometers (MDTraj default) to Angstroms (Proxide default)
    for coord in &mut frame_coords {
        *coord *= 10.0;
    }

    // Try to read time if available
    let time = file
        .dataset("time")
        .and_then(|ds| ds.read_raw::<f64>())
        .ok()
        .and_then(|times| times.get(frame_idx).copied());

    Ok(MdtrajFrame {
        index: frame_idx,
        coords: frame_coords,
        time,
    })
}

/// Convert MDTraj data to RawAtomData
#[cfg(feature = "mdcath")]
pub fn mdtraj_to_raw_atom_data(metadata: &MdtrajH5Result, frame: &MdtrajFrame) -> RawAtomData {
    RawAtomData {
        coords: frame.coords.clone(),
        atom_names: metadata.atom_names.clone(),
        elements: metadata.elements.clone(),
        serial_numbers: (1..=metadata.num_atoms as i32).collect(),
        alt_locs: vec![' '; metadata.num_atoms],
        res_names: metadata.res_names.clone(),
        res_ids: metadata.res_ids.clone(),
        insertion_codes: vec![' '; metadata.num_atoms],
        chain_ids: metadata.chain_ids.clone(),
        b_factors: vec![0.0; metadata.num_atoms],
        occupancy: vec![1.0; metadata.num_atoms],
        charges: None,
        radii: None,
        sigmas: None,
        epsilons: None,
        num_atoms: metadata.num_atoms,
        is_hetatm: vec![false; metadata.num_atoms],
    }
}

// Stub for when mdcath feature is disabled
#[cfg(not(feature = "mdcath"))]
pub fn parse_mdtraj_h5_metadata(_path: &str) -> Result<MdtrajH5Result, String> {
    Err(
        "HDF5 support requires 'mdcath' feature. Rebuild with: cargo build --features mdcath"
            .to_string(),
    )
}

#[cfg(not(feature = "mdcath"))]
pub fn parse_mdtraj_h5_frame(_path: &str, _frame_idx: usize) -> Result<MdtrajFrame, String> {
    Err(
        "HDF5 support requires 'mdcath' feature. Rebuild with: cargo build --features mdcath"
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "mdcath")]
    fn test_mdtraj_h5_result_creation() {
        let result = MdtrajH5Result {
            num_frames: 100,
            num_atoms: 500,
            atom_names: vec!["CA".to_string(); 500],
            res_names: vec!["ALA".to_string(); 500],
            res_ids: (0..500).collect(),
            chain_ids: vec!["A".to_string(); 500],
            elements: vec!["C".to_string(); 500],
        };

        assert_eq!(result.num_frames, 100);
        assert_eq!(result.num_atoms, 500);
    }
}
