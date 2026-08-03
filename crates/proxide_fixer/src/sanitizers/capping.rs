use crate::models::{Atom, Residue, Topology};
use proxide_geometry::geometry::nerf::Nerf;
use std::f32::consts::PI;
use thiserror::Error;

/// Constant threshold for detecting chain breaks (same as proxide-confind)
const CHAIN_BREAK_CA_ANGSTROM: f32 = 4.5;

#[derive(Error, Debug)]
pub enum CappingError {
    #[error("Missing backbone atom {0} in residue")]
    MissingBackboneAtom(String),
    #[error("Failed to cap residue: {0}")]
    General(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TerminusKind {
    NTerm,
    CTerm,
}

#[derive(Debug, Clone)]
pub struct TerminusSite {
    pub chain: String,
    pub res_id: i32,
    pub kind: TerminusKind,
}

#[derive(Debug, Clone)]
pub struct BreakSite {
    pub chain: String,
    pub res_id_upstream: i32,
    pub res_id_downstream: i32,
}

/// CappingSanitizer replaces placeholder H1/OXT atoms with proper termini patches.
/// Detects true N/C termini (first/last residues per chain) and internal chain breaks.
/// Uses NeRF to compute hydrogen and oxygen coordinates with realistic geometry.
pub struct CappingSanitizer<'a> {
    topology: &'a mut Topology,
}

impl<'a> CappingSanitizer<'a> {
    pub fn new(topology: &'a mut Topology) -> Self {
        Self { topology }
    }

    /// Detect true termini (first and last residue per chain).
    /// Does not mutate topology.
    pub fn detect_termini(&self) -> Vec<TerminusSite> {
        let mut sites = Vec::new();
        for chain in &self.topology.chains {
            if let Some(first_res) = chain.residues.first() {
                sites.push(TerminusSite {
                    chain: chain.id.clone(),
                    res_id: first_res.res_id,
                    kind: TerminusKind::NTerm,
                });
            }
            if let Some(last_res) = chain.residues.last() {
                sites.push(TerminusSite {
                    chain: chain.id.clone(),
                    res_id: last_res.res_id,
                    kind: TerminusKind::CTerm,
                });
            }
        }
        sites
    }

    /// Detect internal chain breaks: consecutive residues in the same chain
    /// with CA-CA distance > CHAIN_BREAK_CA_ANGSTROM.
    /// Does not mutate topology.
    pub fn detect_breaks(&self) -> Vec<BreakSite> {
        let mut breaks = Vec::new();
        for chain in &self.topology.chains {
            for i in 0..chain.residues.len().saturating_sub(1) {
                let res_i = &chain.residues[i];
                let res_i_plus_1 = &chain.residues[i + 1];

                if let (Some(ca_i), Some(ca_i_plus_1)) = (
                    res_i.atoms.iter().find(|a| a.name == "CA"),
                    res_i_plus_1.atoms.iter().find(|a| a.name == "CA"),
                ) {
                    let dx = ca_i_plus_1.coords[0] - ca_i.coords[0];
                    let dy = ca_i_plus_1.coords[1] - ca_i.coords[1];
                    let dz = ca_i_plus_1.coords[2] - ca_i.coords[2];
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                    if distance > CHAIN_BREAK_CA_ANGSTROM {
                        breaks.push(BreakSite {
                            chain: chain.id.clone(),
                            res_id_upstream: res_i.res_id,
                            res_id_downstream: res_i_plus_1.res_id,
                        });
                    }
                }
            }
        }
        breaks
    }

    /// Cap termini with proper N-terminal NH3+ and C-terminal OXT coordinates.
    /// Uses NeRF algorithm to place hydrogens and OXT at realistic positions.
    /// Only caps true termini (first/last residue per chain), not internal breaks.
    pub fn run(&mut self) -> Result<(), CappingError> {
        let termini = self.detect_termini();

        for terminus in termini {
            let chain_idx = self
                .topology
                .chains
                .iter()
                .position(|c| c.id == terminus.chain)
                .ok_or_else(|| CappingError::General("Chain not found".to_string()))?;

            let res_idx = self.topology.chains[chain_idx]
                .residues
                .iter()
                .position(|r| r.res_id == terminus.res_id)
                .ok_or_else(|| CappingError::General("Residue not found".to_string()))?;

            let residue = &mut self.topology.chains[chain_idx].residues[res_idx];

            match terminus.kind {
                TerminusKind::NTerm => {
                    cap_n_terminus(residue)?;
                }
                TerminusKind::CTerm => {
                    cap_c_terminus(residue)?;
                }
            }
        }

        Ok(())
    }
}

/// Add H1, H2, H3 to N-terminus with proper geometry.
/// Assumes N atom exists and ideally also CA (for angle/torsion reference).
fn cap_n_terminus(residue: &mut Residue) -> Result<(), CappingError> {
    let n_atom = residue
        .atoms
        .iter()
        .find(|a| a.name == "N")
        .ok_or(CappingError::MissingBackboneAtom("N".to_string()))?
        .clone();

    let ca_atom = residue
        .atoms
        .iter()
        .find(|a| a.name == "CA")
        .ok_or(CappingError::MissingBackboneAtom("CA".to_string()))?
        .clone();

    // Try to find CB for better reference; fall back to C if needed
    let ref_atom = residue
        .atoms
        .iter()
        .find(|a| a.name == "CB")
        .or_else(|| residue.atoms.iter().find(|a| a.name == "C"))
        .ok_or(CappingError::MissingBackboneAtom("CB or C".to_string()))?
        .clone();

    // Ideal N-H bond length: ~1.01 Å
    // Ideal H-N-H angle: ~109.5° (tetrahedral)
    let n_h_bond = 1.01;
    let h_n_h_angle = 109.5;

    // NeRF::place_atom([A, B, C], bond_len, angle, torsion) places D bonded to C.
    // We want H bonded to N, so N must be at index 2.
    // Use [CA, ref_atom, N] so that H is placed from N at the specified bond length.
    let h1_coords = Nerf::place_atom(
        &[ca_atom.coords, ref_atom.coords, n_atom.coords],
        n_h_bond,
        h_n_h_angle,
        0.0, // torsion for first H
    );

    let mut serial = residue.atoms.iter().map(|a| a.serial).max().unwrap_or(0) + 1;

    let h1 = Atom {
        name: "H1".to_string(),
        element: "H".to_string(),
        coords: h1_coords,
        alt_loc: ' ',
        serial,
        b_factor: 0.0,
        occupancy: 1.0,
        is_hetatm: false,
    };
    residue.atoms.push(h1);

    // H2: rotate H1 by ~120° about the N-CA axis
    serial += 1;
    let h2_coords = rotate_around_axis(h1_coords, n_atom.coords, ca_atom.coords, 120.0);
    let h2 = Atom {
        name: "H2".to_string(),
        element: "H".to_string(),
        coords: h2_coords,
        alt_loc: ' ',
        serial,
        b_factor: 0.0,
        occupancy: 1.0,
        is_hetatm: false,
    };
    residue.atoms.push(h2);

    // H3: rotate H1 by ~240° (or -120°) about the N-CA axis
    serial += 1;
    let h3_coords = rotate_around_axis(h1_coords, n_atom.coords, ca_atom.coords, 240.0);
    let h3 = Atom {
        name: "H3".to_string(),
        element: "H".to_string(),
        coords: h3_coords,
        alt_loc: ' ',
        serial,
        b_factor: 0.0,
        occupancy: 1.0,
        is_hetatm: false,
    };
    residue.atoms.push(h3);

    Ok(())
}

/// Add OXT (second carboxylate oxygen) to C-terminus with proper geometry.
fn cap_c_terminus(residue: &mut Residue) -> Result<(), CappingError> {
    let ca_atom = residue
        .atoms
        .iter()
        .find(|a| a.name == "CA")
        .ok_or(CappingError::MissingBackboneAtom("CA".to_string()))?
        .clone();

    let c_atom = residue
        .atoms
        .iter()
        .find(|a| a.name == "C")
        .ok_or(CappingError::MissingBackboneAtom("C".to_string()))?
        .clone();

    let o_atom = residue
        .atoms
        .iter()
        .find(|a| a.name == "O")
        .ok_or(CappingError::MissingBackboneAtom("O".to_string()))?
        .clone();

    // Ideal C=O bond length: ~1.25 Å
    // Ideal O-C-OXT angle: ~123° (sp2 carbon)
    let c_o_bond = 1.25;
    let o_c_o_angle = 123.0;

    // NeRF::place_atom([A, B, C], bond_len, angle, torsion) places D bonded to C.
    // We want OXT bonded to C, so C must be at index 2.
    // Use [CA, O, C] so that OXT is placed from C at the specified bond length.
    let oxt_coords = Nerf::place_atom(
        &[ca_atom.coords, o_atom.coords, c_atom.coords],
        c_o_bond,
        o_c_o_angle,
        180.0, // trans (opposite side of existing O)
    );

    let serial = residue.atoms.iter().map(|a| a.serial).max().unwrap_or(0) + 1;

    let oxt = Atom {
        name: "OXT".to_string(),
        element: "O".to_string(),
        coords: oxt_coords,
        alt_loc: ' ',
        serial,
        b_factor: 0.0,
        occupancy: 1.0,
        is_hetatm: false,
    };
    residue.atoms.push(oxt);

    Ok(())
}

/// Rotate a point around an axis by an angle (in degrees).
/// axis_start and axis_end define the rotation axis.
fn rotate_around_axis(
    point: [f32; 3],
    axis_start: [f32; 3],
    axis_end: [f32; 3],
    angle_deg: f32,
) -> [f32; 3] {
    let angle_rad = angle_deg * PI / 180.0;

    // Vector along the axis
    let axis = [
        axis_end[0] - axis_start[0],
        axis_end[1] - axis_start[1],
        axis_end[2] - axis_start[2],
    ];
    let axis_len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if axis_len == 0.0 {
        return point; // Degenerate axis
    }

    let axis_norm = [axis[0] / axis_len, axis[1] / axis_len, axis[2] / axis_len];

    // Translate point relative to axis start
    let v = [
        point[0] - axis_start[0],
        point[1] - axis_start[1],
        point[2] - axis_start[2],
    ];

    // Rodrigues' rotation formula
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    // dot product: v · axis_norm
    let dot = v[0] * axis_norm[0] + v[1] * axis_norm[1] + v[2] * axis_norm[2];

    // cross product: axis_norm × v
    let cross = [
        axis_norm[1] * v[2] - axis_norm[2] * v[1],
        axis_norm[2] * v[0] - axis_norm[0] * v[2],
        axis_norm[0] * v[1] - axis_norm[1] * v[0],
    ];

    // Rotated vector
    let v_rot = [
        v[0] * cos_a + cross[0] * sin_a + axis_norm[0] * dot * (1.0 - cos_a),
        v[1] * cos_a + cross[1] * sin_a + axis_norm[1] * dot * (1.0 - cos_a),
        v[2] * cos_a + cross[2] * sin_a + axis_norm[2] * dot * (1.0 - cos_a),
    ];

    // Translate back
    [
        axis_start[0] + v_rot[0],
        axis_start[1] + v_rot[1],
        axis_start[2] + v_rot[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Atom, Chain, Residue, Topology};

    /// Helper: squared distance between two 3D points
    fn dist_sq(p1: [f32; 3], p2: [f32; 3]) -> f32 {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let dz = p2[2] - p1[2];
        dx * dx + dy * dy + dz * dz
    }

    /// Helper: Euclidean distance
    fn dist(p1: [f32; 3], p2: [f32; 3]) -> f32 {
        dist_sq(p1, p2).sqrt()
    }

    /// Helper: Build a minimal backbone residue (N, CA, C, O)
    fn build_backbone_residue(name: &str, res_id: i32, serial_base: i32) -> Residue {
        Residue {
            name: name.to_string(),
            res_id,
            insertion_code: ' ',
            atoms: vec![
                Atom {
                    name: "N".to_string(),
                    element: "N".to_string(),
                    coords: [0.0, 0.0, 0.0],
                    alt_loc: ' ',
                    serial: serial_base,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "CA".to_string(),
                    element: "C".to_string(),
                    coords: [1.458, 0.0, 0.0],
                    alt_loc: ' ',
                    serial: serial_base + 1,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "C".to_string(),
                    element: "C".to_string(),
                    coords: [2.009, 1.391, 0.0],
                    alt_loc: ' ',
                    serial: serial_base + 2,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
                Atom {
                    name: "O".to_string(),
                    element: "O".to_string(),
                    coords: [1.221, 2.338, 0.0],
                    alt_loc: ' ',
                    serial: serial_base + 3,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
            ],
        }
    }

    #[test]
    fn test_detect_termini_single_residue() {
        let res = build_backbone_residue("ALA", 1, 1);
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res],
            }],
        };

        let sanitizer = CappingSanitizer::new(&mut topology);
        let termini = sanitizer.detect_termini();

        assert_eq!(termini.len(), 2);
        assert!(termini
            .iter()
            .any(|t| t.kind == TerminusKind::NTerm && t.res_id == 1));
        assert!(termini
            .iter()
            .any(|t| t.kind == TerminusKind::CTerm && t.res_id == 1));
    }

    #[test]
    fn test_detect_termini_three_residues() {
        let res1 = build_backbone_residue("ALA", 1, 1);
        let res2 = build_backbone_residue("GLY", 2, 5);
        let res3 = build_backbone_residue("SER", 3, 9);

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res1, res2, res3],
            }],
        };

        let sanitizer = CappingSanitizer::new(&mut topology);
        let termini = sanitizer.detect_termini();

        assert_eq!(termini.len(), 2);
        let nterm = termini
            .iter()
            .find(|t| t.kind == TerminusKind::NTerm)
            .unwrap();
        let cterm = termini
            .iter()
            .find(|t| t.kind == TerminusKind::CTerm)
            .unwrap();
        assert_eq!(nterm.res_id, 1);
        assert_eq!(cterm.res_id, 3);
    }

    #[test]
    fn test_detect_breaks_no_breaks_continuous() {
        let res1 = build_backbone_residue("ALA", 1, 1);
        let res2 = build_backbone_residue("GLY", 2, 5);

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res1, res2],
            }],
        };

        let sanitizer = CappingSanitizer::new(&mut topology);
        let breaks = sanitizer.detect_breaks();
        assert_eq!(breaks.len(), 0);
    }

    #[test]
    fn test_detect_breaks_internal_break() {
        // Residue 1 with CA at (1.458, 0, 0)
        let res1 = build_backbone_residue("ALA", 1, 1);

        // Residue 2 with CA far away at (15, 0, 0) -> distance > 4.5 Å
        let mut res2 = build_backbone_residue("GLY", 2, 5);
        res2.atoms
            .iter_mut()
            .find(|a| a.name == "CA")
            .unwrap()
            .coords = [15.0, 0.0, 0.0];

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res1, res2],
            }],
        };

        let sanitizer = CappingSanitizer::new(&mut topology);
        let breaks = sanitizer.detect_breaks();

        assert_eq!(breaks.len(), 1);
        let break_site = &breaks[0];
        assert_eq!(break_site.chain, "A");
        assert_eq!(break_site.res_id_upstream, 1);
        assert_eq!(break_site.res_id_downstream, 2);
    }

    #[test]
    fn test_cap_n_terminus_adds_h1_h2_h3() {
        let res = build_backbone_residue("ALA", 1, 1);
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res],
            }],
        };

        let mut sanitizer = CappingSanitizer::new(&mut topology);
        sanitizer.run().unwrap();

        let residue = &topology.chains[0].residues[0];
        let h1 = residue.atoms.iter().find(|a| a.name == "H1");
        let h2 = residue.atoms.iter().find(|a| a.name == "H2");
        let h3 = residue.atoms.iter().find(|a| a.name == "H3");

        assert!(h1.is_some(), "H1 should be present");
        assert!(h2.is_some(), "H2 should be present");
        assert!(h3.is_some(), "H3 should be present");

        // Verify no atom has [0,0,0] coords
        assert_ne!(h1.unwrap().coords, [0.0, 0.0, 0.0]);
        assert_ne!(h2.unwrap().coords, [0.0, 0.0, 0.0]);
        assert_ne!(h3.unwrap().coords, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cap_n_terminus_h_bond_length() {
        let res = build_backbone_residue("ALA", 1, 1);
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res],
            }],
        };

        let mut sanitizer = CappingSanitizer::new(&mut topology);
        sanitizer.run().unwrap();

        let residue = &topology.chains[0].residues[0];
        let n_atom = residue.atoms.iter().find(|a| a.name == "N").unwrap();
        let h1 = residue.atoms.iter().find(|a| a.name == "H1").unwrap();

        let bond_len = dist(n_atom.coords, h1.coords);
        // Ideal N-H bond is ~1.01 Å, allow ±0.3 Å tolerance (NeRF approximation)
        assert!(
            (bond_len - 1.01).abs() < 0.3,
            "N-H bond length {} should be ~1.01 Å",
            bond_len
        );
    }

    #[test]
    fn test_cap_c_terminus_adds_oxt() {
        let res = build_backbone_residue("ALA", 1, 1);
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res],
            }],
        };

        let mut sanitizer = CappingSanitizer::new(&mut topology);
        sanitizer.run().unwrap();

        let residue = &topology.chains[0].residues[0];
        let oxt = residue.atoms.iter().find(|a| a.name == "OXT");

        assert!(oxt.is_some(), "OXT should be present");
        assert_ne!(oxt.unwrap().coords, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cap_c_terminus_o_c_o_angle() {
        let res = build_backbone_residue("ALA", 1, 1);
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res],
            }],
        };

        let mut sanitizer = CappingSanitizer::new(&mut topology);
        sanitizer.run().unwrap();

        let residue = &topology.chains[0].residues[0];
        let c_atom = residue.atoms.iter().find(|a| a.name == "C").unwrap();
        let o_atom = residue.atoms.iter().find(|a| a.name == "O").unwrap();
        let oxt = residue.atoms.iter().find(|a| a.name == "OXT").unwrap();

        // Compute O-C-OXT angle
        let oc = [
            o_atom.coords[0] - c_atom.coords[0],
            o_atom.coords[1] - c_atom.coords[1],
            o_atom.coords[2] - c_atom.coords[2],
        ];
        let oxt_c = [
            oxt.coords[0] - c_atom.coords[0],
            oxt.coords[1] - c_atom.coords[1],
            oxt.coords[2] - c_atom.coords[2],
        ];

        let oc_len = (oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2]).sqrt();
        let oxt_c_len = (oxt_c[0] * oxt_c[0] + oxt_c[1] * oxt_c[1] + oxt_c[2] * oxt_c[2]).sqrt();

        if oc_len > 0.0 && oxt_c_len > 0.0 {
            let dot =
                (oc[0] * oxt_c[0] + oc[1] * oxt_c[1] + oc[2] * oxt_c[2]) / (oc_len * oxt_c_len);
            let clamped_dot = dot.clamp(-1.0, 1.0);
            let angle_rad = clamped_dot.acos();
            let angle_deg = angle_rad * 180.0 / PI;

            // Ideal O-C-OXT angle is ~123°, allow ±30° tolerance (NeRF approximation)
            assert!(
                (angle_deg - 123.0).abs() < 30.0,
                "O-C-OXT angle {} should be ~123°",
                angle_deg
            );
        }
    }

    #[test]
    fn test_cap_c_terminus_bond_length() {
        let res = build_backbone_residue("ALA", 1, 1);
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res],
            }],
        };

        let mut sanitizer = CappingSanitizer::new(&mut topology);
        sanitizer.run().unwrap();

        let residue = &topology.chains[0].residues[0];
        let c_atom = residue.atoms.iter().find(|a| a.name == "C").unwrap();
        let oxt = residue.atoms.iter().find(|a| a.name == "OXT").unwrap();

        let bond_len = dist(c_atom.coords, oxt.coords);
        // Ideal C-O bond is ~1.25 Å, allow ±0.4 Å tolerance (NeRF approximation)
        assert!(
            (bond_len - 1.25).abs() < 0.4,
            "C-OXT bond length {} should be ~1.25 Å",
            bond_len
        );
    }

    #[test]
    fn test_no_placeholder_zeros() {
        let res = build_backbone_residue("ALA", 1, 1);
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res],
            }],
        };

        let mut sanitizer = CappingSanitizer::new(&mut topology);
        sanitizer.run().unwrap();

        let residue = &topology.chains[0].residues[0];
        // Check only the added atoms (H1, H2, H3, OXT) don't have placeholder zeros
        for atom in &residue.atoms {
            if atom.name == "H1" || atom.name == "H2" || atom.name == "H3" || atom.name == "OXT" {
                assert_ne!(
                    atom.coords,
                    [0.0, 0.0, 0.0],
                    "Added atom {} has placeholder [0,0,0] coords",
                    atom.name
                );
            }
        }
    }

    #[test]
    fn test_chain_break_not_capped() {
        // Residue 1: N at origin, CA at (1.458, 0, 0)
        let res1 = build_backbone_residue("ALA", 1, 1);

        // Residue 2: CA far away at (15, 0, 0), creating a break
        let mut res2 = build_backbone_residue("GLY", 2, 5);
        res2.atoms
            .iter_mut()
            .find(|a| a.name == "CA")
            .unwrap()
            .coords = [15.0, 0.0, 0.0];

        // Residue 3: Normal continuation from residue 2
        let mut res3 = build_backbone_residue("SER", 3, 9);
        res3.atoms
            .iter_mut()
            .find(|a| a.name == "CA")
            .unwrap()
            .coords = [16.458, 0.0, 0.0];

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![res1, res2, res3],
            }],
        };

        let mut sanitizer = CappingSanitizer::new(&mut topology);
        sanitizer.run().unwrap();

        // Res 1 should have H1/H2/H3 (it's NTerm)
        let res1_atom_names: Vec<_> = topology.chains[0].residues[0]
            .atoms
            .iter()
            .map(|a| a.name.clone())
            .collect();
        assert!(res1_atom_names.contains(&"H1".to_string()));
        assert!(res1_atom_names.contains(&"H2".to_string()));
        assert!(res1_atom_names.contains(&"H3".to_string()));

        // Res 2 (middle of break) should NOT have H or OXT added
        let res2_atom_names: Vec<_> = topology.chains[0].residues[1]
            .atoms
            .iter()
            .map(|a| a.name.clone())
            .collect();
        assert!(!res2_atom_names.contains(&"H1".to_string()));
        assert!(!res2_atom_names.contains(&"OXT".to_string()));

        // Res 3 should have OXT (it's CTerm)
        let res3_atom_names: Vec<_> = topology.chains[0].residues[2]
            .atoms
            .iter()
            .map(|a| a.name.clone())
            .collect();
        assert!(res3_atom_names.contains(&"OXT".to_string()));
    }

    #[test]
    fn test_multiple_chains() {
        let res_a = build_backbone_residue("ALA", 1, 1);
        let res_b = build_backbone_residue("GLY", 1, 5);

        let mut topology = Topology {
            chains: vec![
                Chain {
                    id: "A".to_string(),
                    residues: vec![res_a],
                },
                Chain {
                    id: "B".to_string(),
                    residues: vec![res_b],
                },
            ],
        };

        let mut sanitizer = CappingSanitizer::new(&mut topology);
        sanitizer.run().unwrap();

        // Chain A should have both N and C caps
        let chain_a = &topology.chains[0];
        let chain_a_names: Vec<_> = chain_a.residues[0]
            .atoms
            .iter()
            .map(|a| a.name.clone())
            .collect();
        assert!(chain_a_names.contains(&"H1".to_string()));
        assert!(chain_a_names.contains(&"OXT".to_string()));

        // Chain B should also have both
        let chain_b = &topology.chains[1];
        let chain_b_names: Vec<_> = chain_b.residues[0]
            .atoms
            .iter()
            .map(|a| a.name.clone())
            .collect();
        assert!(chain_b_names.contains(&"H1".to_string()));
        assert!(chain_b_names.contains(&"OXT".to_string()));
    }
}
