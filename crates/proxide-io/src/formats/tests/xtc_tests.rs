use crate::formats::xtc::molly_impl::read_xtc_molly;
use std::path::Path;

#[test]
fn test_read_xtc_real_file() {
    // Determine path to test.xtc relative to workspace root
    // Environment Context shows it at tests/data/trajectories/test.xtc
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let xtc_path = project_root.join("tests/data/trajectories/test.xtc");
    
    if !xtc_path.exists() {
        // Fallback for different test environments if needed, 
        // but based on environment context it should be there.
        return;
    }

    let traj = read_xtc_molly(xtc_path).expect("Failed to read XTC file");
    
    // We expect some frames and atoms. 
    // Let's verify based on typical test.xtc properties if known, 
    // or just check they are non-zero.
    assert!(traj.num_frames > 0);
    assert!(traj.num_atoms > 0);
    assert_eq!(traj.coords.len(), traj.num_frames);
    assert_eq!(traj.coords[0].len(), traj.num_atoms * 3);
    assert_eq!(traj.times.len(), traj.num_frames);
}

#[test]
fn test_read_xtc_nonexistent() {
    let result = read_xtc_molly("nonexistent.xtc");
    assert!(result.is_err());
}
