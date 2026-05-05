use crate::models::{Topology};
use crate::templates::ResidueLibrary;
use crate::models::Atom;

impl Topology {
    pub fn add_hydrogens(&mut self, _ph: f32, _library: &ResidueLibrary) {
        // Basic implementation for ALA: add 'HA'
        for chain in &mut self.chains {
            for residue in &mut chain.residues {
                if residue.name == "ALA" {
                    // Check if HA is missing
                    if !residue.atoms.iter().any(|a| a.name == "HA") {
                        // We could use Builder::add_missing_atoms here if HA is in the template.
                        // For now, let's manually add a dummy HA atom to test the logic.
                        residue.atoms.push(Atom {
                            name: "HA".to_string(),
                            element: "H".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 0,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        });
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Atom, Chain, Residue};
    use crate::templates::ResidueLibrary;

    #[test]
    fn test_add_hydrogens_ala() {
        let residue = Residue {
            name: "ALA".to_string(),
            res_id: 1,
            insertion_code: ' ',
            atoms: vec![
                Atom {
                    name: "CA".to_string(),
                    element: "C".to_string(),
                    coords: [0.0, 0.0, 0.0],
                    alt_loc: ' ',
                    serial: 1,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                },
            ],
        };

        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![residue],
            }],
        };

        let library = ResidueLibrary::new();
        topology.add_hydrogens(7.0, &library);

        assert!(topology.chains[0].residues[0].atoms.iter().any(|a| a.name == "HA"));
    }
}
