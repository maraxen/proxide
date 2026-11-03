#![allow(dead_code)]

use proxide_confind::coords::{ProteinBackbone, ResidueBackbone};
use proxide_core::processing::residues::ResidueId;
use proxide_rotlib::RotamerLibrary;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Build a synthetic ProteinBackbone with `n_res` residues on a single chain.
/// Residues are placed along the x-axis at `spacing` Å apart (CA positions).
/// phi=-60.0, psi=-45.0 for all interior residues (α-helix geometry).
/// Terminal residues have None for missing dihedrals (Debt #1897).
pub fn make_synthetic_backbone(n_res: usize, spacing: f64) -> Arc<ProteinBackbone> {
    let mut bb = Vec::new();
    let mut ids = Vec::new();
    let mut chain_map = Vec::new();

    for i in 0..n_res {
        let x = i as f64 * spacing;

        // Non-collinear peptide geometry (standard planar peptide bond):
        // N-CA-C angle ~112.6° (Engh & Huber tetrahedral CA ~111.2°),
        // C=O bond length 1.2 Å, ensuring non-zero Gram-Schmidt cross-product.
        let n_pos = Some([x - 1.2, 0.8, 0.0]);
        let ca_pos = Some([x, 0.0, 0.0]);
        let c_pos = Some([x + 1.2, 0.8, 0.0]);
        let o_pos = Some([x + 1.2, 2.0, 0.0]);

        let phi = if i == 0 { None } else { Some(-60.0) };
        let psi = if i == n_res - 1 { None } else { Some(-45.0) };
        let omega = if i == 0 { None } else { Some(180.0) };

        bb.push(ResidueBackbone {
            res_name: "ALA".to_string(),
            n: n_pos,
            ca: ca_pos,
            c: c_pos,
            o: o_pos,
            phi,
            psi,
            omega,
            is_cis_peptide: false,
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

/// Build a synthetic ProteinBackbone with partial atoms (for testing atom building)
pub fn make_synthetic_backbone_partial_atoms(
    n_res: usize,
    spacing: f64,
    skip_indices: &[usize],
) -> Arc<ProteinBackbone> {
    let mut bb = Vec::new();
    let mut ids = Vec::new();
    let mut chain_map = Vec::new();

    for i in 0..n_res {
        let x = i as f64 * spacing;

        let n_pos = Some([x - 1.2, 0.8, 0.0]);
        let ca_pos = Some([x, 0.0, 0.0]);
        let c_pos = if skip_indices.contains(&i) {
            None
        } else {
            Some([x + 1.2, 0.8, 0.0])
        };
        let o_pos = if skip_indices.contains(&i) {
            None
        } else {
            Some([x + 1.2, 2.0, 0.0])
        };

        let phi = if i == 0 { None } else { Some(-60.0) };
        let psi = if i == n_res - 1 { None } else { Some(-45.0) };
        let omega = if i == 0 { None } else { Some(180.0) };

        bb.push(ResidueBackbone {
            res_name: "ALA".to_string(),
            n: n_pos,
            ca: ca_pos,
            c: c_pos,
            o: o_pos,
            phi,
            psi,
            omega,
            is_cis_peptide: false,
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
            phi: None,
            psi: None,
            omega: None,
            is_cis_peptide: false,
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
pub fn make_two_chain_backbone(
    n_res: usize,
    spacing: f64,
    chain_b_offset: [f64; 3],
) -> Arc<ProteinBackbone> {
    let mut bb = Vec::new();
    let mut ids = Vec::new();
    let mut chain_map = Vec::new();

    // Chain A
    for i in 0..n_res {
        let x = i as f64 * spacing;
        let n_pos = Some([x - 1.2, 0.8, 0.0]);
        let ca_pos = Some([x, 0.0, 0.0]);
        let c_pos = Some([x + 1.2, 0.8, 0.0]);
        let o_pos = Some([x + 1.2, 2.0, 0.0]);

        let phi = if i == 0 { None } else { Some(-60.0) };
        let psi = if i == n_res - 1 { None } else { Some(-45.0) };
        let omega = if i == 0 { None } else { Some(180.0) };

        bb.push(ResidueBackbone {
            res_name: "ALA".to_string(),
            n: n_pos,
            ca: ca_pos,
            c: c_pos,
            o: o_pos,
            phi,
            psi,
            omega,
            is_cis_peptide: false,
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

        let n_pos = Some([x - 1.2, y + 0.8, z]);
        let ca_pos = Some([x, y, z]);
        let c_pos = Some([x + 1.2, y + 0.8, z]);
        let o_pos = Some([x + 1.2, y + 2.0, z]);

        let phi = if i == 0 { None } else { Some(-60.0) };
        let psi = if i == n_res - 1 { None } else { Some(-45.0) };
        let omega = if i == 0 { None } else { Some(180.0) };

        bb.push(ResidueBackbone {
            res_name: "ALA".to_string(),
            n: n_pos,
            ca: ca_pos,
            c: c_pos,
            o: o_pos,
            phi,
            psi,
            omega,
            is_cis_peptide: false,
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

/// Load RotamerLibrary from RLIB env var or default committed workspace asset.
pub fn load_rotlib_or_skip() -> Option<Arc<RotamerLibrary>> {
    if let Ok(path_str) = std::env::var("RLIB") {
        let path = Path::new(&path_str);
        if let Ok(lib) = RotamerLibrary::load_pb_zst(path) {
            return Some(Arc::new(lib));
        }
        if let Ok(lib) = RotamerLibrary::load_pb(path) {
            return Some(Arc::new(lib));
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../data/rotlibs/proxide-rotlib-dunbrack2010-ccd.pb.zst"),
        manifest_dir.join("../../data/rotlibs/proxide-rotlib-bbdep2010.pb.zst"),
    ];
    for p in &candidates {
        if p.exists() {
            if let Ok(lib) = RotamerLibrary::load_pb_zst(p) {
                return Some(Arc::new(lib));
            }
            if let Ok(lib) = RotamerLibrary::load_pb(p) {
                return Some(Arc::new(lib));
            }
        }
    }

    None
}

/// Create a lightweight in-memory RotamerLibrary for tests that do not evaluate sidechain rotamers.
pub fn dummy_rotlib() -> Arc<RotamerLibrary> {
    let mut pb = proxide_rotlib::pb::rotlib_v1::RotamerLibrary::default();
    pb.attribution = "synthetic-test".to_string();
    pb.geometry_mode = proxide_rotlib::pb::rotlib_v1::GeometryMode::Precomputed as i32;
    Arc::new(RotamerLibrary::from_protobuf(&pb).expect("failed to construct dummy rotlib"))
}

/// Return path to committed PDB fixture.
pub fn real_pdb_path() -> PathBuf {
    if let Ok(p) = std::env::var("PDB_PATH") {
        PathBuf::from(p)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/1crn.pdb")
    }
}

/// Load backbone from committed PDB fixture.
pub fn load_real_backbone() -> Option<Arc<ProteinBackbone>> {
    let path = real_pdb_path();
    if !path.exists() {
        return None;
    }
    let backbone = proxide_confind::load_pdb_f64(&path).ok()?;
    Some(Arc::new(backbone))
}

/// Find the ResidueIndex for a given chain_id + res_id in a backbone.
pub fn res_idx(
    bb: &ProteinBackbone,
    chain: &str,
    res_id: i32,
) -> proxide_confind::coords::ResidueIndex {
    let i = bb
        .ids
        .iter()
        .position(|id| id.chain_id == chain && id.res_id == res_id)
        .unwrap_or_else(|| panic!("residue {},{} not found", chain, res_id));
    proxide_confind::coords::ResidueIndex(i as u32)
}
