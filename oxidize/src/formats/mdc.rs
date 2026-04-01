//! MDCompress trajectory file format parser using the rust_mdc bridge.
//!
//! MDC is a custom compressed trajectory format.
//! This implementation bridges to the C++ library via rust_mdc.

use rust_mdc::{Reader, QueryResult};
use std::path::Path;

/// MDC trajectory data structure
#[derive(Debug, Clone)]
pub struct MdcTrajectory {
    /// Number of frames
    pub num_frames: usize,
    /// Number of atoms per frame
    pub num_atoms: usize,
    /// Time for each frame in ps
    pub times: Vec<f32>,
    /// Coordinates for each frame (n_frames x n_atoms*3) in Angstroms
    pub coords: Vec<Vec<f32>>,
}

/// Read an MDC file using the rust_mdc bridge
/// Returns coordinates in Angstroms (converts from nm).
pub fn read_mdc<P: AsRef<Path>>(path: P) -> Result<MdcTrajectory, Box<dyn std::error::Error>> {
    let path_str = path.as_ref().to_str().ok_or("Invalid path encoding")?;
    let reader = Reader::new(path_str).map_err(|e| format!("MDC Reader Error: {}", e))?;
    
    let num_frames = reader.get_no_frames() as usize;
    if num_frames == 0 {
        return Ok(MdcTrajectory {
            num_frames: 0,
            num_atoms: 0,
            times: Vec::new(),
            coords: Vec::new(),
        });
    }

    // 2. Initialize Query Engine for ALL atoms and ALL frames
    let engine = reader
        .get_query_engine(&[], &[])
        .map_err(|e| format!("MDC QueryEngine Error: {}", e))?;
    
    let original_atom_ids = engine
        .get_original_atom_ids()
        .map_err(|e| format!("MDC Atom IDs Error: {}", e))?;
    let num_atoms = original_atom_ids.len();

    let mut times = Vec::with_capacity(num_frames);
    let mut all_coords = Vec::with_capacity(num_frames);
    
    // Create query result and frame IDs
    let mut query_result = QueryResult::new().map_err(|e| format!("MDC QueryResult Error: {}", e))?;
    let frame_ids: Vec<u32> = (0..num_frames as u32).collect();
    
    // Query all frames
    engine
        .query(&frame_ids, &mut query_result)
        .map_err(|e| format!("MDC Query Error: {}", e))?;
    
    if query_result.frames.len() != num_frames {
        return Err(format!(
            "MDC Query Frame Mismatch: expected {}, got {}",
            num_frames,
            query_result.frames.len()
        )
        .into());
    }

    for frame in query_result.frames.iter() {
        times.push(frame.time);
        
        // Safety check for atom count consistency
        if frame.coords.len() != num_atoms {
            return Err(format!(
                "MDC Frame Atom Mismatch: expected {}, got {}",
                num_atoms,
                frame.coords.len()
            )
            .into());
        }
        
        // Convert coordinates from nm to Angstroms (multiply by 10.0)
        let mut frame_coords = Vec::with_capacity(num_atoms * 3);
        for atom in frame.coords.iter() {
            frame_coords.push(atom.x * 10.0);
            frame_coords.push(atom.y * 10.0);
            frame_coords.push(atom.z * 10.0);
        }
        all_coords.push(frame_coords);
    }

    Ok(MdcTrajectory {
        num_frames,
        num_atoms,
        times,
        coords: all_coords,
    })
}
