//! MDCATH HDF5 file parser
//!
//! Parses mdCATH-format HDF5 files containing molecular dynamics trajectory data.
//! The HDF5 structure is:
//! - /{domain_id}/: Domain group
//!   - /resname: (N_res,) string - residue names
//!   - /chain: (N_res,) string - chain IDs
//!   - /{temperature}/{replica}/coords: (N_frames, N_atoms, 3) - coordinates
//!   - /{temperature}/{replica}/dssp: (N_frames, N_res) - secondary structure

#![allow(dead_code)]

#[cfg(feature = "mdcath")]
use hdf5::{File as H5File, Group};

use crate::structure::RawAtomData;

/// Metadata for an MDCATH domain
#[derive(Debug, Clone)]
pub struct MdcathDomain {
    /// Domain identifier (e.g., "1a2b00")
    pub domain_id: String,
    /// Number of residues
    pub num_residues: usize,
    /// Residue names
    pub resnames: Vec<String>,
    /// Chain IDs (residue-level)
    pub chain_ids: Vec<String>,
    /// Available temperature keys (e.g., "320", "348", "379", "413", "450")
    pub temperatures: Vec<String>,
}

/// A single frame from MDCATH trajectory
#[derive(Debug)]
pub struct MdcathFrame {
    /// Temperature key
    pub temperature: String,
    /// Replica key
    pub replica: String,
    /// Frame index within replica
    pub frame_idx: usize,
    /// Coordinates (N_atoms, 3) - flattened
    pub coords: Vec<f32>,
}

/// Parse MDCATH HDF5 file and return domain metadata
#[cfg(feature = "mdcath")]
pub fn parse_mdcath_metadata(path: &str) -> Result<MdcathDomain, String> {
    let file = H5File::open(path).map_err(|e| format!("Failed to open HDF5 file: {}", e))?;

    // Get first domain ID (there's usually only one per file)
    let domain_id = file
        .member_names()
        .map_err(|e| format!("Failed to list file contents: {}", e))?
        .into_iter()
        .next()
        .ok_or_else(|| "Empty HDF5 file".to_string())?;

    let domain_group = file
        .group(&domain_id)
        .map_err(|e| format!("Failed to open domain group: {}", e))?;

    // Read residue names
    let resnames = read_string_dataset(&domain_group, "resname")?;
    let num_residues = resnames.len();

    // Read chain IDs
    let chain_ids = if domain_group.dataset("chain").is_ok() {
        read_string_dataset(&domain_group, "chain")?
    } else {
        vec!["A".to_string(); num_residues]
    };

    // Find temperature keys (numeric subgroups)
    let temperatures: Vec<String> = domain_group
        .member_names()
        .map_err(|e| format!("Failed to list domain contents: {}", e))?
        .into_iter()
        .filter(|name| name.parse::<u32>().is_ok())
        .collect();

    Ok(MdcathDomain {
        domain_id,
        num_residues,
        resnames,
        chain_ids,
        temperatures,
    })
}

#[cfg(feature = "mdcath")]
fn read_string_dataset(group: &Group, name: &str) -> Result<Vec<String>, String> {
    let dataset = group
        .dataset(name)
        .map_err(|e| format!("Failed to open dataset '{}': {}", name, e))?;

    // Read as fixed-length strings
    let data: Vec<hdf5::types::FixedAscii<8>> = dataset
        .read_raw()
        .map_err(|e| format!("Failed to read dataset '{}': {}", name, e))?;

    Ok(data
        .into_iter()
        .map(|s| s.to_string().trim_end_matches('\0').to_string())
        .collect())
}

/// Get list of replicas for a temperature
#[cfg(feature = "mdcath")]
pub fn get_replicas(path: &str, domain_id: &str, temperature: &str) -> Result<Vec<String>, String> {
    let file = H5File::open(path).map_err(|e| format!("Failed to open HDF5 file: {}", e))?;
    let domain_group = file
        .group(domain_id)
        .map_err(|e| format!("Failed to open domain group: {}", e))?;
    let temp_group = domain_group
        .group(temperature)
        .map_err(|e| format!("Failed to open temperature group '{}': {}", temperature, e))?;

    temp_group
        .member_names()
        .map_err(|e| format!("Failed to list replicas: {}", e))
}

/// Read a single frame from MDCATH trajectory
#[cfg(feature = "mdcath")]
pub fn parse_mdcath_frame(
    path: &str,
    domain_id: &str,
    temperature: &str,
    replica: &str,
    frame_idx: usize,
) -> Result<MdcathFrame, String> {
    let file = H5File::open(path).map_err(|e| format!("Failed to open HDF5 file: {}", e))?;

    let coords_path = format!("{}/{}/{}/coords", domain_id, temperature, replica);
    let coords_ds = file
        .dataset(&coords_path)
        .map_err(|e| format!("Failed to open coords dataset at '{}': {}", coords_path, e))?;

    let shape = coords_ds.shape();
    if shape.len() != 3 {
        return Err(format!(
            "Expected 3D coords, got {} dimensions",
            shape.len()
        ));
    }

    let num_frames = shape[0];
    let num_atoms = shape[1];

    if frame_idx >= num_frames {
        return Err(format!(
            "Frame {} out of range (max {})",
            frame_idx, num_frames
        ));
    }

    // Read all coordinates and extract frame
    // (More efficient would be HDF5 slicing)
    let all_coords: Vec<f32> = coords_ds
        .read_raw()
        .map_err(|e| format!("Failed to read coordinates: {}", e))?;

    let frame_size = num_atoms * 3;
    let start = frame_idx * frame_size;
    let end = start + frame_size;

    Ok(MdcathFrame {
        temperature: temperature.to_string(),
        replica: replica.to_string(),
        frame_idx,
        coords: all_coords[start..end].to_vec(),
    })
}

/// Convert MDCATH data to RawAtomData
/// Note: MDCATH coordinates are at residue level (CA atoms only)
/// so this creates a simplified representation
#[cfg(feature = "mdcath")]
pub fn mdcath_to_raw_atom_data(metadata: &MdcathDomain, frame: &MdcathFrame) -> RawAtomData {
    // MDCATH typically stores CA-only coordinates
    // We treat each residue as a single CA atom
    let num_atoms = metadata.num_residues;

    RawAtomData {
        coords: frame.coords.clone(),
        atom_names: vec!["CA".to_string(); num_atoms],
        elements: vec!["C".to_string(); num_atoms],
        serial_numbers: (1..=num_atoms as i32).collect(),
        alt_locs: vec![' '; num_atoms],
        res_names: metadata.resnames.clone(),
        res_ids: (0..num_atoms as i32).collect(),
        insertion_codes: vec![' '; num_atoms],
        chain_ids: metadata.chain_ids.clone(),
        b_factors: vec![0.0; num_atoms],
        occupancy: vec![1.0; num_atoms],
        charges: None,
        radii: None,
        sigmas: None,
        epsilons: None,
        num_atoms,
        is_hetatm: vec![false; num_atoms],
    }
}

// Stubs for when mdcath feature is disabled
#[cfg(not(feature = "mdcath"))]
pub fn parse_mdcath_metadata(_path: &str) -> Result<MdcathDomain, String> {
    Err(
        "HDF5 support requires 'mdcath' feature. Rebuild with: cargo build --features mdcath"
            .to_string(),
    )
}

#[cfg(not(feature = "mdcath"))]
pub fn get_replicas(
    _path: &str,
    _domain_id: &str,
    _temperature: &str,
) -> Result<Vec<String>, String> {
    Err("HDF5 support requires 'mdcath' feature".to_string())
}

#[cfg(not(feature = "mdcath"))]
pub fn parse_mdcath_frame(
    _path: &str,
    _domain_id: &str,
    _temperature: &str,
    _replica: &str,
    _frame_idx: usize,
) -> Result<MdcathFrame, String> {
    Err("HDF5 support requires 'mdcath' feature".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mdcath_domain_creation() {
        let domain = MdcathDomain {
            domain_id: "1abc00".to_string(),
            num_residues: 100,
            resnames: vec!["ALA".to_string(); 100],
            chain_ids: vec!["A".to_string(); 100],
            temperatures: vec!["320".to_string(), "348".to_string()],
        };

        assert_eq!(domain.num_residues, 100);
        assert_eq!(domain.temperatures.len(), 2);
    }

    #[test]
    fn test_mdcath_frame_creation() {
        let frame = MdcathFrame {
            temperature: "320".to_string(),
            replica: "0".to_string(),
            frame_idx: 0,
            coords: vec![0.0; 300],
        };

        assert_eq!(frame.temperature, "320");
        assert_eq!(frame.coords.len(), 300);
    }
}
