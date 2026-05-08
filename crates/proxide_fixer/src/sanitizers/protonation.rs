use crate::models::Topology;
use crate::templates::ResidueLibrary;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProtonationError {
    #[error("Failed to protonate residue: {0}")]
    General(String),
}

pub enum ProtonationStrategy {
    ConstantPH(f32),
}

pub struct ProtonationSanitizer<'a> {
    topology: &'a mut Topology,
    strategy: ProtonationStrategy,
    library: &'a ResidueLibrary,
}

impl<'a> ProtonationSanitizer<'a> {
    pub fn new(
        topology: &'a mut Topology,
        strategy: ProtonationStrategy,
        library: &'a ResidueLibrary,
    ) -> Self {
        Self {
            topology,
            strategy,
            library,
        }
    }

    pub fn run(&mut self) -> Result<(), ProtonationError> {
        let ph = match self.strategy {
            ProtonationStrategy::ConstantPH(val) => val,
        };
        
        self.topology.add_hydrogens(ph, self.library);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Atom, Chain, Residue, Topology};
    use crate::templates::ResidueLibrary;

    #[test]
    fn test_protonation_sanitizer() {
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
        let mut sanitizer = ProtonationSanitizer::new(
            &mut topology,
            ProtonationStrategy::ConstantPH(7.0),
            &library,
        );
        sanitizer.run().unwrap();

        assert!(topology.chains[0].residues[0].atoms.iter().any(|a| a.name == "HA"));
    }
}
