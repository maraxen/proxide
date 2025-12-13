//! TRR trajectory format parser
//!
//! TRR is a binary format used by GROMACS, containing positions, velocities, and forces.
//! Uses chemfiles crate for parsing (feature-gated by 'trajectories').

#![allow(dead_code)]

/// TRR trajectory data
#[derive(Debug, Clone)]
pub struct TrrTrajectory {
    /// Number of frames
    pub num_frames: usize,
    /// Number of atoms per frame
    pub num_atoms: usize,
    /// Timestamps for each frame (ps)
    pub times: Vec<f32>,
    /// Coordinates per frame: Vec of flattened (N_atoms * 3)
    pub coords: Vec<Vec<f32>>,
    /// Velocities per frame (optional)
    pub velocities: Option<Vec<Vec<f32>>>,
    /// Forces per frame (optional)
    pub forces: Option<Vec<Vec<f32>>>,
    /// Box vectors per frame
    pub box_vectors: Option<Vec<[[f64; 3]; 3]>>,
}

#[cfg(feature = "trajectories")]
pub mod chemfiles_impl {
    use super::*;
    use chemfiles::{Frame, Trajectory};

    /// Read TRR trajectory using chemfiles
    pub fn read_trr_chemfiles(path: &str) -> Result<TrrTrajectory, Box<dyn std::error::Error>> {
        let mut trajectory = Trajectory::open(path, 'r')?;
        let num_frames = trajectory.nsteps();

        if num_frames == 0 {
            return Err("Empty TRR trajectory".into());
        }

        let mut frame = Frame::new();
        trajectory.read(&mut frame)?;
        let num_atoms = frame.size();

        let mut times = Vec::with_capacity(num_frames);
        let mut coords = Vec::with_capacity(num_frames);
        let mut velocities = Vec::with_capacity(num_frames);
        let mut box_vectors = Vec::with_capacity(num_frames);

        // Reset to beginning
        trajectory = Trajectory::open(path, 'r')?;

        let mut has_velocities = false;

        for _i in 0..num_frames {
            let mut frame = Frame::new();
            trajectory.read(&mut frame)?;

            // Get time
            times.push(0.0); // TRR time handling varies

            // Get positions
            let positions = frame.positions();
            let mut frame_coords = Vec::with_capacity(num_atoms * 3);
            for pos in positions {
                frame_coords.push(pos[0] as f32);
                frame_coords.push(pos[1] as f32);
                frame_coords.push(pos[2] as f32);
            }
            coords.push(frame_coords);

            // Get velocities if available
            // Get velocities if available
            if let Some(vels) = frame.velocities() {
                has_velocities = true;
                let mut frame_vels = Vec::with_capacity(num_atoms * 3);
                for i in 0..num_atoms {
                    frame_vels.push(vels[i][0] as f32);
                    frame_vels.push(vels[i][1] as f32);
                    frame_vels.push(vels[i][2] as f32);
                }
                velocities.push(frame_vels);
            }

            // Get box vectors
            let cell = frame.cell();
            let matrix = cell.matrix();
            box_vectors.push([
                [matrix[0][0], matrix[0][1], matrix[0][2]],
                [matrix[1][0], matrix[1][1], matrix[1][2]],
                [matrix[2][0], matrix[2][1], matrix[2][2]],
            ]);
        }

        Ok(TrrTrajectory {
            num_frames,
            num_atoms,
            times,
            coords,
            velocities: if has_velocities {
                Some(velocities)
            } else {
                None
            },
            forces: None, // chemfiles may not expose forces
            box_vectors: Some(box_vectors),
        })
    }
}

#[cfg(not(feature = "trajectories"))]
pub mod chemfiles_impl {
    use super::*;

    /// Stub when trajectories feature is disabled
    pub fn read_trr_chemfiles(_path: &str) -> Result<TrrTrajectory, Box<dyn std::error::Error>> {
        Err("TRR support requires compiling with 'trajectories' feature".into())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_trr_trajectory_struct() {
        use super::*;

        let traj = TrrTrajectory {
            num_frames: 10,
            num_atoms: 100,
            times: vec![0.0; 10],
            coords: vec![vec![0.0; 300]; 10],
            velocities: None,
            forces: None,
            box_vectors: None,
        };

        assert_eq!(traj.num_frames, 10);
        assert_eq!(traj.num_atoms, 100);
    }
}
