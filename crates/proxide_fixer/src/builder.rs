use crate::models::{Topology, Atom};
use crate::templates::ResidueLibrary;
use nalgebra::{Matrix3, Matrix3x1};
use log::{warn, info};

pub struct Builder;

impl Builder {
    pub fn add_missing_atoms(topology: &mut Topology, library: &ResidueLibrary) {
        for chain in &mut topology.chains {
            for residue in &mut chain.residues {
                if let Some(template) = library.get(&residue.name) {
                    let existing_atom_names: std::collections::HashSet<String> = residue.atoms.iter().map(|a| a.name.clone()).collect();
                    let missing_atoms: Vec<_> = template.atoms.iter()
                        .filter(|ta| !existing_atom_names.contains(&ta.name))
                        .collect();

                    if missing_atoms.is_empty() {
                        continue;
                    }

                    info!("Fixing residue {} (res_id={}): missing {} atoms", residue.name, residue.res_id, missing_atoms.len());

                    // Find common atoms
                    let common_atoms: Vec<_> = residue.atoms.iter()
                        .filter(|a| template.atoms.iter().any(|ta| ta.name == a.name))
                        .collect();

                    if common_atoms.len() >= 3 {
                        // Align using Kabsch
                        let mut source_pts = Vec::new();
                        let mut target_pts = Vec::new();
                        
                        for atom in &common_atoms {
                            let t_atom = template.atoms.iter().find(|ta| ta.name == atom.name).unwrap();
                            source_pts.push(Matrix3x1::new(atom.coords[0], atom.coords[1], atom.coords[2]));
                            target_pts.push(Matrix3x1::new(t_atom.coords[0] as f32, t_atom.coords[1] as f32, t_atom.coords[2] as f32));
                        }

                        // Compute centroids
                        let mut centroid_src = Matrix3x1::zeros();
                        let mut centroid_tgt = Matrix3x1::zeros();
                        for (s, t) in source_pts.iter().zip(target_pts.iter()) {
                            centroid_src += s;
                            centroid_tgt += t;
                        }
                        centroid_src /= source_pts.len() as f32;
                        centroid_tgt /= target_pts.len() as f32;

                        // Center pts
                        let centered_src: Vec<_> = source_pts.iter().map(|p| p - centroid_src).collect();
                        let centered_tgt: Vec<_> = target_pts.iter().map(|p| p - centroid_tgt).collect();

                        // Compute H = S^T * T (sum of outer products)
                        let mut h = Matrix3::zeros();
                        for (s, t) in centered_src.iter().zip(centered_tgt.iter()) {
                            h += s * t.transpose();
                        }

                        // Compute SVD
                        let svd = h.svd(true, true);
                        let u = svd.u.expect("SVD U failed");
                        let v_t = svd.v_t.expect("SVD V failed");
                        let mut rot: Matrix3<f32> = u * v_t;
                        if rot.determinant() < 0.0 {
                            let mut s = Matrix3::identity();
                            s[(2, 2)] = -1.0;
                            rot = u * s * v_t;
                        }

                        // Apply transformation
                        for ta in missing_atoms {
                            let pt = Matrix3x1::new(ta.coords[0] as f32, ta.coords[1] as f32, ta.coords[2] as f32);
                            let centered_ta = pt - centroid_tgt;
                            let rotated = rot * centered_ta;
                            let final_pt = rotated + centroid_src;

                            residue.atoms.push(Atom {
                                name: ta.name.clone(),
                                element: ta.element.clone(),
                                coords: [final_pt.x, final_pt.y, final_pt.z],
                                alt_loc: ' ',
                                serial: 0, // Placeholder
                                b_factor: 0.0,
                                occupancy: 1.0,
                                is_hetatm: false,
                            });
                        }
                    } else {
                        warn!("Not enough atoms to reconstruct for {} ({} found, need 3)", residue.name, common_atoms.len());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Topology, Residue, Atom, Chain};
    use crate::templates::{ResidueLibrary, ResidueTemplate, TemplateAtom};

    #[test]
    fn test_add_missing_atoms() {
        let mut lib = ResidueLibrary::new();
        // ALA template: N, CA, CB, C, O
        lib.insert(ResidueTemplate {
            name: "ALA".to_string(),
            atoms: vec![
                TemplateAtom { name: "N".to_string(), element: "N".to_string(), coords: [0.0, 0.0, 0.0] },
                TemplateAtom { name: "CA".to_string(), element: "C".to_string(), coords: [1.0, 0.0, 0.0] },
                TemplateAtom { name: "CB".to_string(), element: "C".to_string(), coords: [1.0, 1.0, 0.0] },
                TemplateAtom { name: "C".to_string(), element: "C".to_string(), coords: [1.0, 1.0, 1.0] }, // Make them non-collinear
                TemplateAtom { name: "O".to_string(), element: "O".to_string(), coords: [2.0, 0.0, 1.0] },
            ],
            bonds: vec![],
        });

        // Offset the input by [10, 10, 10]
        let offset = [10.0, 10.0, 10.0];
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![Residue {
                    name: "ALA".to_string(),
                    res_id: 1,
                    insertion_code: ' ',
                    atoms: vec![
                        Atom { name: "N".to_string(), element: "N".to_string(), coords: [0.0 + offset[0], 0.0 + offset[1], 0.0 + offset[2]], alt_loc: ' ', serial: 1, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                        Atom { name: "CA".to_string(), element: "C".to_string(), coords: [1.0 + offset[0], 0.0 + offset[1], 0.0 + offset[2]], alt_loc: ' ', serial: 2, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                        Atom { name: "C".to_string(), element: "C".to_string(), coords: [1.0 + offset[0], 1.0 + offset[1], 1.0 + offset[2]], alt_loc: ' ', serial: 3, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                    ],
                }],
            }],
        };

        Builder::add_missing_atoms(&mut topology, &lib);

        let res = &topology.chains[0].residues[0];
        assert_eq!(res.atoms.len(), 5);
        
        // Verify CB position
        let cb = res.atoms.iter().find(|a| a.name == "CB").unwrap();
        assert!((cb.coords[0] - (1.0 + offset[0])).abs() < 1e-3, "CB x expected {}, got {}", 1.0 + offset[0], cb.coords[0]);
        assert!((cb.coords[1] - (1.0 + offset[1])).abs() < 1e-3, "CB y expected {}, got {}", 1.0 + offset[1], cb.coords[1]);
        assert!((cb.coords[2] - (0.0 + offset[2])).abs() < 1e-3, "CB z expected {}, got {}", 0.0 + offset[2], cb.coords[2]);
    }
}
