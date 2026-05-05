use crate::models::Topology;
use crate::templates::ResidueLibrary;

#[derive(Debug, Clone)]
pub struct MissingAtom {
    pub chain_id: String,
    pub res_name: String,
    pub res_id: i32,
    pub insertion_code: char,
    pub atom_name: String,
}

impl Topology {
    pub fn find_missing_atoms(&self, library: &ResidueLibrary) -> Vec<MissingAtom> {
        let mut missing = Vec::new();

        for chain in &self.chains {
            for residue in &chain.residues {
                if let Some(template) = library.get(&residue.name) {
                    for expected_atom in &template.atoms {
                        if !residue.atoms.iter().any(|a| a.name == expected_atom.name) {
                            missing.push(MissingAtom {
                                chain_id: chain.id.clone(),
                                res_name: residue.name.clone(),
                                res_id: residue.res_id,
                                insertion_code: residue.insertion_code,
                                atom_name: expected_atom.name.clone(),
                            });
                        }
                    }
                } else {
                    eprintln!("Warning: Template for residue {} not found in library", residue.name);
                }
            }
        }
        missing
    }
}

#[cfg(test)]
mod tests {
    use crate::models::{Atom, Residue, Chain, Topology};
    use crate::templates::{ResidueLibrary, ResidueTemplate, TemplateAtom};

    #[test]
    fn test_find_missing_atoms() {
        let mut library = ResidueLibrary::new();
        library.insert(ResidueTemplate {
            name: "ALA".to_string(),
            atoms: vec![
                TemplateAtom { name: "N".to_string(), element: "N".to_string(), coords: [0.0, 0.0, 0.0] },
                TemplateAtom { name: "CA".to_string(), element: "C".to_string(), coords: [0.0, 0.0, 0.0] },
                TemplateAtom { name: "CB".to_string(), element: "C".to_string(), coords: [0.0, 0.0, 0.0] },
            ],
            bonds: vec![],
        });

        let topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![Residue {
                    name: "ALA".to_string(),
                    res_id: 1,
                    insertion_code: ' ',
                    atoms: vec![
                        Atom { name: "N".to_string(), element: "N".to_string(), coords: [0.0, 0.0, 0.0], alt_loc: ' ', serial: 1, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                        Atom { name: "CA".to_string(), element: "C".to_string(), coords: [0.0, 0.0, 0.0], alt_loc: ' ', serial: 2, b_factor: 0.0, occupancy: 1.0, is_hetatm: false },
                    ],
                }],
            }],
        };

        let missing = topology.find_missing_atoms(&library);
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].atom_name, "CB");
        assert_eq!(missing[0].res_name, "ALA");
    }
}
