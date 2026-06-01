use proxide_confind::coords::{ProteinBackbone, ResidueBackbone};
use proxide_core::processing::residues::ResidueId;
use proxide_rotlib::RotamerLibrary;
use std::path::Path;
use std::sync::Arc;

/// Build a synthetic ProteinBackbone with `n_res` residues on a single chain.
/// Residues are placed along the x-axis at `spacing` Å apart (CA positions).
/// phi=-60.0, psi=-45.0 for all interior residues (α-helix geometry).
/// Terminal residues get sentinel phi/psi (9999.0).
pub fn make_synthetic_backbone(n_res: usize, spacing: f64) -> Arc<ProteinBackbone> {
    let mut bb = Vec::new();
    let mut ids = Vec::new();
    let mut chain_map = Vec::new();

    for i in 0..n_res {
        let x = i as f64 * spacing;

        // Place CA at (x, 0, 0), N at (x-0.5, 0, 0), C at (x+1.0, 0, 0), O at (x+1.1, 1.0, 0)
        let n_pos = Some([x - 0.5, 0.0, 0.0]);
        let ca_pos = Some([x, 0.0, 0.0]);
        let c_pos = Some([x + 1.0, 0.0, 0.0]);
        let o_pos = Some([x + 1.1, 1.0, 0.0]);

        // Terminal residues: phi/psi = 9999.0 (sentinel for missing)
        let phi = if i == 0 { 9999.0 } else { -60.0 };
        let psi = if i == n_res - 1 { 9999.0 } else { -45.0 };

        bb.push(ResidueBackbone {
            res_name: "ALA".to_string(),
            n: n_pos,
            ca: ca_pos,
            c: c_pos,
            o: o_pos,
            phi,
            psi,
        });

        ids.push(ResidueId {
            chain_id: "A".to_string(),
            res_id: (i + 1) as i32,
            insertion_code: ' ',
        });

        chain_map.push(0); // All on chain 0
    }

    Arc::new(ProteinBackbone { bb, ids, chain_map })
}

/// Build a synthetic ProteinBackbone with partial atoms (for testing atom building)
pub fn make_synthetic_backbone_partial_atoms(
    n_res: usize,
    spacing: f64,
    skip_indices: &[usize], // indices where we skip C and O atoms
) -> Arc<ProteinBackbone> {
    let mut bb = Vec::new();
    let mut ids = Vec::new();
    let mut chain_map = Vec::new();

    for i in 0..n_res {
        let x = i as f64 * spacing;

        let n_pos = Some([x - 0.5, 0.0, 0.0]);
        let ca_pos = Some([x, 0.0, 0.0]);
        let c_pos = if skip_indices.contains(&i) { None } else { Some([x + 1.0, 0.0, 0.0]) };
        let o_pos = if skip_indices.contains(&i) { None } else { Some([x + 1.1, 1.0, 0.0]) };

        let phi = if i == 0 { 9999.0 } else { -60.0 };
        let psi = if i == n_res - 1 { 9999.0 } else { -45.0 };

        bb.push(ResidueBackbone {
            res_name: "ALA".to_string(),
            n: n_pos,
            ca: ca_pos,
            c: c_pos,
            o: o_pos,
            phi,
            psi,
        });

        ids.push(ResidueId {
            chain_id: "A".to_string(),
            res_id: (i + 1) as i32,
            insertion_code: ' ',
        });

        chain_map.push(0);
    }

    Arc::new(ProteinBackbone { bb, ids, chain_map })
}

/// Build a synthetic backbone with all atoms as None
pub fn make_synthetic_backbone_no_atoms(n_res: usize) -> Arc<ProteinBackbone> {
    let mut bb = Vec::new();
    let mut ids = Vec::new();
    let mut chain_map = Vec::new();

    for i in 0..n_res {
        bb.push(ResidueBackbone {
            res_name: "ALA".to_string(),
            n: None,
            ca: None,
            c: None,
            o: None,
            phi: 9999.0,
            psi: 9999.0,
        });

        ids.push(ResidueId {
            chain_id: "A".to_string(),
            res_id: (i + 1) as i32,
            insertion_code: ' ',
        });

        chain_map.push(0);
    }

    Arc::new(ProteinBackbone { bb, ids, chain_map })
}

/// Build a synthetic ProteinBackbone with two chains.
/// Chain A: n_res residues on chain 0, spacing `spacing`.
/// Chain B: n_res residues on chain 1, at offset position.
pub fn make_two_chain_backbone(n_res: usize, spacing: f64, chain_b_offset: [f64; 3]) -> Arc<ProteinBackbone> {
    let mut bb = Vec::new();
    let mut ids = Vec::new();
    let mut chain_map = Vec::new();

    // Chain A
    for i in 0..n_res {
        let x = i as f64 * spacing;
        let n_pos = Some([x - 0.5, 0.0, 0.0]);
        let ca_pos = Some([x, 0.0, 0.0]);
        let c_pos = Some([x + 1.0, 0.0, 0.0]);
        let o_pos = Some([x + 1.1, 1.0, 0.0]);

        let phi = if i == 0 { 9999.0 } else { -60.0 };
        let psi = if i == n_res - 1 { 9999.0 } else { -45.0 };

        bb.push(ResidueBackbone {
            res_name: "ALA".to_string(),
            n: n_pos,
            ca: ca_pos,
            c: c_pos,
            o: o_pos,
            phi,
            psi,
        });

        ids.push(ResidueId {
            chain_id: "A".to_string(),
            res_id: (i + 1) as i32,
            insertion_code: ' ',
        });

        chain_map.push(0);
    }

    // Chain B
    for i in 0..n_res {
        let x = i as f64 * spacing + chain_b_offset[0];
        let y = chain_b_offset[1];
        let z = chain_b_offset[2];

        let n_pos = Some([x - 0.5, y, z]);
        let ca_pos = Some([x, y, z]);
        let c_pos = Some([x + 1.0, y, z]);
        let o_pos = Some([x + 1.1, y + 1.0, z]);

        let phi = if i == 0 { 9999.0 } else { -60.0 };
        let psi = if i == n_res - 1 { 9999.0 } else { -45.0 };

        bb.push(ResidueBackbone {
            res_name: "ALA".to_string(),
            n: n_pos,
            ca: ca_pos,
            c: c_pos,
            o: o_pos,
            phi,
            psi,
        });

        ids.push(ResidueId {
            chain_id: "B".to_string(),
            res_id: (i + 1) as i32,
            insertion_code: ' ',
        });

        chain_map.push(1);
    }

    Arc::new(ProteinBackbone { bb, ids, chain_map })
}

/// Load RotamerLibrary from RLIB env var. Returns None if unset or file not found.
pub fn load_rotlib_or_skip() -> Option<Arc<RotamerLibrary>> {
    let path_str = std::env::var("RLIB").ok()?;
    let path = Path::new(&path_str);
    RotamerLibrary::load(path).ok().map(Arc::new)
}

/// Load the small PDB fixture. Uses PDB_PATH env var or defaults to Mosaist testfiles.
pub fn real_pdb_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("PDB_PATH") {
        std::path::PathBuf::from(p)
    } else {
        std::path::PathBuf::from("/home/marielle/repos/mosaist/testfiles/small.pdb")
    }
}

/// Load RotamerLibrary path. Uses ROTLIB_PATH env var or defaults to Mosaist testfiles.
pub fn real_rotlib_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("ROTLIB_PATH") {
        std::path::PathBuf::from(p)
    } else {
        std::path::PathBuf::from("/home/marielle/repos/mosaist/testfiles/rotlib.bin")
    }
}

/// Load backbone from the real PDB fixture. Returns None if file not found.
pub fn load_real_backbone() -> Option<Arc<ProteinBackbone>> {
    let path = real_pdb_path();
    if !path.exists() {
        return None;
    }
    proxide_confind::load_pdb_f64(&path).ok().map(Arc::new)
}

/// Find the ResidueIndex for a given chain_id + res_id in a backbone.
pub fn res_idx(bb: &ProteinBackbone, chain: &str, res_id: i32) -> proxide_confind::coords::ResidueIndex {
    let i = bb.ids
        .iter()
        .position(|id| id.chain_id == chain && id.res_id == res_id)
        .unwrap_or_else(|| panic!("residue {},{} not found", chain, res_id));
    proxide_confind::coords::ResidueIndex(i as u32)
}
