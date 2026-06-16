use crate::models::{Topology, Atom};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CappingError {
    #[error("Failed to cap residue: {0}")]
    General(String),
}

pub struct CappingSanitizer<'a> {
    topology: &'a mut Topology,
}

impl<'a> CappingSanitizer<'a> {
    pub fn new(topology: &'a mut Topology) -> Self {
        Self { topology }
    }

    pub fn run(&mut self) -> Result<(), CappingError> {
        for chain in self.topology.chains.iter_mut() {
            if let Some(first_res) = chain.residues.first_mut() {
                // Apply N-terminal cap: NH3+
                let n_term_atom = Atom {
                    name: "H1".to_string(),
                    element: "H".to_string(),
                    coords: [0.0, 0.0, 0.0], // Placeholder coords
                    alt_loc: ' ',
                    serial: 999,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                };
                first_res.atoms.push(n_term_atom);
            }

            if let Some(last_res) = chain.residues.last_mut() {
                // Apply C-terminal cap: COO-
                let c_term_atom = Atom {
                    name: "OXT".to_string(),
                    element: "O".to_string(),
                    coords: [0.0, 0.0, 0.0], // Placeholder coords
                    alt_loc: ' ',
                    serial: 1000,
                    b_factor: 0.0,
                    occupancy: 1.0,
                    is_hetatm: false,
                };
                last_res.atoms.push(c_term_atom);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Atom, Chain, Residue, Topology};

    #[test]
    fn test_capping_sanitizer() {
        let mut residue = Residue {
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

        let mut sanitizer = CappingSanitizer::new(&mut topology);
        sanitizer.run().unwrap();

        let res = &topology.chains[0].residues[0];
        assert!(res.atoms.iter().any(|a| a.name == "H1"));
        assert!(res.atoms.iter().any(|a| a.name == "OXT"));
    }
}
