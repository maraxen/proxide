//! DCD trajectory format parser
//!
//! DCD is a binary format used by CHARMM and NAMD.
//! Uses chemfiles crate for parsing (feature-gated by 'trajectories').

#![allow(dead_code)]

/// DCD trajectory data
#[derive(Debug, Clone)]
pub struct DcdTrajectory {
    /// Number of frames
    pub num_frames: usize,
    /// Number of atoms per frame
    pub num_atoms: usize,
    /// Timestamps for each frame (ps)
    pub times: Vec<f32>,
    /// Coordinates per frame: Vec of flattened (N_atoms * 3)
    pub coords: Vec<Vec<f32>>,
    /// Unit cell dimensions per frame (a, b, c, alpha, beta, gamma)
    pub unit_cells: Option<Vec<[f64; 6]>>,
}

#[cfg(feature = "trajectories")]
pub mod chemfiles_impl {
    use super::*;
    use chemfiles::{Frame, Trajectory};

    /// Read DCD trajectory using chemfiles
    pub fn read_dcd_chemfiles(path: &str) -> Result<DcdTrajectory, Box<dyn std::error::Error>> {
        let mut trajectory = Trajectory::open(path, 'r')?;
        let num_frames = trajectory.nsteps();

        if num_frames == 0 {
            return Err("Empty DCD trajectory".into());
        }

        let mut frame = Frame::new();
        trajectory.read(&mut frame)?;
        let num_atoms = frame.size();

        let mut times = Vec::with_capacity(num_frames);
        let mut coords = Vec::with_capacity(num_frames);
        let mut unit_cells = Vec::with_capacity(num_frames);

        // Reset to beginning
        trajectory = Trajectory::open(path, 'r')?;

        for _i in 0..num_frames {
            let mut frame = Frame::new();
            trajectory.read(&mut frame)?;

            // Get time (chemfiles may not provide this for DCD)
            times.push(0.0); // DCD doesn't always have timestamps

            // Get positions
            let positions = frame.positions();
            let mut frame_coords = Vec::with_capacity(num_atoms * 3);
            for pos in positions {
                frame_coords.push(pos[0] as f32);
                frame_coords.push(pos[1] as f32);
                frame_coords.push(pos[2] as f32);
            }
            coords.push(frame_coords);

            // Get unit cell
            let cell = frame.cell();
            let lengths = cell.lengths();
            let angles = cell.angles();
            unit_cells.push([
                lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2],
            ]);
        }

        Ok(DcdTrajectory {
            num_frames,
            num_atoms,
            times,
            coords,
            unit_cells: Some(unit_cells),
        })
    }
}

#[cfg(not(feature = "trajectories"))]
pub mod chemfiles_impl {
    use super::*;

    /// Stub when trajectories feature is disabled
    pub fn read_dcd_chemfiles(_path: &str) -> Result<DcdTrajectory, Box<dyn std::error::Error>> {
        Err("DCD support requires compiling with 'trajectories' feature".into())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_dcd_trajectory_struct() {
        use super::*;

        let traj = DcdTrajectory {
            num_frames: 10,
            num_atoms: 100,
            times: vec![0.0; 10],
            coords: vec![vec![0.0; 300]; 10],
            unit_cells: None,
        };

        assert_eq!(traj.num_frames, 10);
        assert_eq!(traj.num_atoms, 100);
    }
}
