use crate::models::Topology;
use log::warn;
use proxide_rs::structure::systems::AtomicSystem;

pub struct Builder;

impl Builder {
    pub fn map_seqres(
        topology: &Topology,
        seqres: &std::collections::HashMap<String, Vec<(i32, String)>>,
    ) {
        for chain in &topology.chains {
            if let Some(expected_residues) = seqres.get(&chain.id) {
                let existing_residues: std::collections::HashSet<(i32, String)> = chain
                    .residues
                    .iter()
                    .filter(|r| !r.atoms.iter().any(|a| a.is_hetatm))
                    .map(|r| (r.res_id, r.name.clone()))
                    .collect();

                for (res_id, res_name) in expected_residues {
                    if !existing_residues.contains(&(*res_id, res_name.clone())) {
                        warn!(
                            "Residue {} (res_id={}) missing in chain {}",
                            res_name, res_id, chain.id
                        );
                    }
                }
            }
        }
    }

    pub fn resolve_clashes(system: &mut AtomicSystem, radius: f32) {
        let max_iter = 100;
        let convergence_threshold = 0.01;

        for _ in 0..max_iter {
            let mut max_displacement: f32 = 0.0;
            let coords = system.coordinates.clone();
            let num_atoms = coords.len() / 3;

            for i in 0..num_atoms {
                for j in (i + 1)..num_atoms {
                    let d2 = (coords[i * 3] - coords[j * 3]).powi(2)
                        + (coords[i * 3 + 1] - coords[j * 3 + 1]).powi(2)
                        + (coords[i * 3 + 2] - coords[j * 3 + 2]).powi(2);
                    let d = d2.sqrt();
                    let min_dist = radius * 2.0;

                    if d < min_dist && d > 1e-6 {
                        let shift = (min_dist - d) / 2.0;
                        let direction = [
                            (coords[i * 3] - coords[j * 3]) / d,
                            (coords[i * 3 + 1] - coords[j * 3 + 1]) / d,
                            (coords[i * 3 + 2] - coords[j * 3 + 2]) / d,
                        ];

                        system.coordinates[i * 3] += direction[0] * shift;
                        system.coordinates[i * 3 + 1] += direction[1] * shift;
                        system.coordinates[i * 3 + 2] += direction[2] * shift;
                        system.coordinates[j * 3] -= direction[0] * shift;
                        system.coordinates[j * 3 + 1] -= direction[1] * shift;
                        system.coordinates[j * 3 + 2] -= direction[2] * shift;

                        max_displacement = max_displacement.max(shift);
                    }
                }
            }
            if max_displacement < convergence_threshold {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Atom, Chain, Residue, Topology};
    use proxide_rs::structure::systems::AtomicSystemArgs;
    use std::collections::HashMap;

    #[test]
    fn test_map_seqres() {
        let topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "ALA".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "HOH".to_string(),
                        res_id: 2,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "O".to_string(),
                            element: "O".to_string(),
                            coords: [1.0, 1.0, 1.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: true,
                        }],
                    },
                ],
            }],
        };

        let mut seqres = HashMap::new();
        seqres.insert(
            "A".to_string(),
            vec![
                (1, "ALA".to_string()),
                (2, "VAL".to_string()), // Missing
            ],
        );

        Builder::map_seqres(&topology, &seqres);
    }

    #[test]
    fn test_clash_resolution() {
        let mut system = AtomicSystem::new(AtomicSystemArgs {
            coordinates: vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            atom_mask: vec![1.0, 1.0],
            atom_names: None,
            elements: None,
            bonds: None,
            charges: None,
            sigmas: None,
            epsilons: None,
            radii: None,
            residue_index: None,
            chain_index: None,
        });

        // Radius of 1.0 Å, collision min_dist = 2.0 Å. Currently 0.5 Å.
        Builder::resolve_clashes(&mut system, 1.0);

        let d2 = (system.coordinates[0] - system.coordinates[3]).powi(2)
            + (system.coordinates[1] - system.coordinates[4]).powi(2)
            + (system.coordinates[2] - system.coordinates[5]).powi(2);
        let d = d2.sqrt();
        assert!(d >= 2.0, "Clash not resolved: distance is {}", d);
    }
}
