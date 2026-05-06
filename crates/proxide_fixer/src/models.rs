use proxide_rs::structure::RawAtomData;
use proxide_rs::structure::systems::{AtomicSystem, AtomicSystemArgs};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Atom {
    pub name: String,
    pub element: String,
    pub coords: [f32; 3],
    pub alt_loc: char,
    pub serial: i32,
    pub b_factor: f32,
    pub occupancy: f32,
    pub is_hetatm: bool,
}

#[derive(Debug, Clone)]
pub struct Residue {
    pub name: String,
    pub res_id: i32,
    pub insertion_code: char,
    pub atoms: Vec<Atom>,
}

#[derive(Debug, Clone)]
pub struct Chain {
    pub id: String,
    pub residues: Vec<Residue>,
}

#[derive(Debug, Clone)]
pub struct Topology {
    pub chains: Vec<Chain>,
}

impl Topology {
    pub fn from_raw_atom_data(data: &RawAtomData) -> Self {
        let mut chains = Vec::new();
        
        if data.num_atoms == 0 {
            return Topology { chains };
        }

        for i in 0..data.num_atoms {
            let atom = Atom {
                name: data.atom_names[i].clone(),
                element: data.elements[i].clone(),
                coords: [data.coords[i * 3], data.coords[i * 3 + 1], data.coords[i * 3 + 2]],
                alt_loc: data.alt_locs[i],
                serial: data.serial_numbers[i],
                b_factor: data.b_factors[i],
                occupancy: data.occupancy[i],
                is_hetatm: data.is_hetatm[i],
            };

            let chain_id = &data.chain_ids[i];
            let res_name = &data.res_names[i];
            let res_id = data.res_ids[i];
            let ins_code = data.insertion_codes[i];

            if chains.is_empty() || chains.last().unwrap().id != *chain_id {
                chains.push(Chain {
                    id: chain_id.clone(),
                    residues: Vec::new(),
                });
            }

            let chain = chains.last_mut().unwrap();
            if chain.residues.is_empty() || 
               chain.residues.last().unwrap().res_id != res_id ||
               chain.residues.last().unwrap().insertion_code != ins_code ||
               chain.residues.last().unwrap().name != *res_name 
            {
                chain.residues.push(Residue {
                    name: res_name.clone(),
                    res_id,
                    insertion_code: ins_code,
                    atoms: Vec::new(),
                });
            }

            chain.residues.last_mut().unwrap().atoms.push(atom);
        }

        Topology { chains }
    }

    pub fn to_atomic_system(&self) -> AtomicSystem {
        let mut coordinates = Vec::new();
        let mut atom_names = Vec::new();
        let mut elements = Vec::new();
        let mut residue_indices = Vec::new();
        let mut chain_indices = Vec::new();
        let mut unique_chain_ids = Vec::new();
        let mut chain_id_map = HashMap::new();

        let mut res_idx = 0;
        let mut chain_idx = 0;

        for chain in &self.chains {
            if !chain_id_map.contains_key(&chain.id) {
                chain_id_map.insert(chain.id.clone(), chain_idx);
                unique_chain_ids.push(chain.id.clone());
                chain_idx += 1;
            }
            let current_chain_idx = *chain_id_map.get(&chain.id).unwrap();

            for residue in &chain.residues {
                for atom in &residue.atoms {
                    coordinates.extend_from_slice(&atom.coords);
                    atom_names.push(atom.name.clone());
                    elements.push(atom.element.clone());
                    residue_indices.push(res_idx);
                    chain_indices.push(current_chain_idx);
                }
                res_idx += 1;
            }
        }

        let num_atoms = atom_names.len();
        let mut system = AtomicSystem::new(AtomicSystemArgs {
            coordinates,
            atom_mask: vec![1.0; num_atoms],
            atom_names: Some(atom_names),
            elements: Some(elements),
            bonds: None,
            charges: None,
            sigmas: None,
            epsilons: None,
            radii: None,
            residue_index: None,
            chain_index: None,
        });
        system.residue_index = Some(residue_indices);
        system.chain_index = Some(chain_indices);
        system.unique_chain_ids = Some(unique_chain_ids);
        system
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_to_atomic_system() {
        let data = RawAtomData {
            coords: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            atom_names: vec!["N".to_string(), "CA".to_string()],
            elements: vec!["N".to_string(), "C".to_string()],
            serial_numbers: vec![1, 2],
            alt_locs: vec![' ', ' '],
            res_names: vec!["ALA".to_string(), "ALA".to_string()],
            res_ids: vec![1, 1],
            insertion_codes: vec![' ', ' '],
            chain_ids: vec!["A".to_string(), "A".to_string()],
            b_factors: vec![0.0, 0.0],
            occupancy: vec![1.0, 1.0],
            charges: None,
            radii: None,
            sigmas: None,
            epsilons: None,
            num_atoms: 2,
            is_hetatm: vec![false, false],
        };

        let topology = Topology::from_raw_atom_data(&data);
        let system = topology.to_atomic_system();

        assert_eq!(system.coordinates.len(), 6);
        assert_eq!(system.atom_names.len(), 2);
        assert_eq!(system.residue_index.as_ref().unwrap()[0], 0);
        assert_eq!(system.residue_index.as_ref().unwrap()[1], 0);
        assert_eq!(system.chain_index.as_ref().unwrap()[0], 0);
        assert_eq!(system.unique_chain_ids.as_ref().unwrap()[0], "A");
    }

    #[test]
    fn test_topology_conversion_sequential() {
        let data = RawAtomData {
            coords: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            atom_names: vec!["N".to_string(), "CA".to_string(), "C".to_string()],
            elements: vec!["N".to_string(), "C".to_string(), "C".to_string()],
            serial_numbers: vec![1, 2, 3],
            alt_locs: vec![' ', ' ', ' '],
            res_names: vec!["ALA".to_string(), "ALA".to_string(), "ALA".to_string()],
            res_ids: vec![1, 1, 1],
            insertion_codes: vec![' ', ' ', ' '],
            chain_ids: vec!["A".to_string(), "A".to_string(), "A".to_string()],
            b_factors: vec![0.0, 0.0, 0.0],
            occupancy: vec![1.0, 1.0, 1.0],
            charges: None,
            radii: None,
            sigmas: None,
            epsilons: None,
            num_atoms: 3,
            is_hetatm: vec![false, false, false],
        };

        let topology = Topology::from_raw_atom_data(&data);
        assert_eq!(topology.chains.len(), 1);
        assert_eq!(topology.chains[0].residues.len(), 1);
        assert_eq!(topology.chains[0].residues[0].atoms.len(), 3);
        assert_eq!(topology.chains[0].residues[0].name, "ALA");
    }

    #[test]
    fn test_topology_multiple_chains_residues() {
        let data = RawAtomData {
            coords: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            atom_names: vec!["N".to_string(), "CA".to_string()],
            elements: vec!["N".to_string(), "C".to_string()],
            serial_numbers: vec![1, 2],
            alt_locs: vec![' ', ' '],
            res_names: vec!["ALA".to_string(), "GLY".to_string()],
            res_ids: vec![1, 2],
            insertion_codes: vec![' ', ' '],
            chain_ids: vec!["A".to_string(), "B".to_string()],
            b_factors: vec![0.0, 0.0],
            occupancy: vec![1.0, 1.0],
            charges: None,
            radii: None,
            sigmas: None,
            epsilons: None,
            num_atoms: 2,
            is_hetatm: vec![false, false],
        };

        let topology = Topology::from_raw_atom_data(&data);
        assert_eq!(topology.chains.len(), 2);
        assert_eq!(topology.chains[0].id, "A");
        assert_eq!(topology.chains[1].id, "B");
        assert_eq!(topology.chains[0].residues[0].name, "ALA");
        assert_eq!(topology.chains[1].residues[0].name, "GLY");
    }
}
