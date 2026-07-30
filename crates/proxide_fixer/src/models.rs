use proxide_rs::structure::systems::{AtomicSystem, AtomicSystemArgs};
use proxide_rs::structure::RawAtomData;
use std::collections::HashMap;

/// Atom index + selection info for altloc filtering
#[derive(Clone, Copy)]
struct AtomIndexWithOccupancy {
    index: usize,
    occupancy: f32,
    alt_loc: char,
}

/// Group key for atom deduplication by position
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AtomPositionKey {
    chain_id: String,
    res_id: i32,
    insertion_code: char,
    atom_name: String,
}

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

/// Filter atoms to keep only one altloc per atom position.
/// For each unique (chain_id, res_id, insertion_code, atom_name), keeps:
/// - The atom with alt_loc == ' ' (blank) unconditionally
/// - Otherwise, the one with highest occupancy (ties broken by alphabetical alt_loc)
fn filter_altlocs(data: &RawAtomData) -> Vec<usize> {
    if data.num_atoms == 0 {
        return Vec::new();
    }

    let mut groups: HashMap<AtomPositionKey, Vec<AtomIndexWithOccupancy>> = HashMap::new();

    // Group atoms by position
    for i in 0..data.num_atoms {
        let key = AtomPositionKey {
            chain_id: data.chain_ids[i].clone(),
            res_id: data.res_ids[i],
            insertion_code: data.insertion_codes[i],
            atom_name: data.atom_names[i].clone(),
        };

        groups
            .entry(key)
            .or_default()
            .push(AtomIndexWithOccupancy {
                index: i,
                occupancy: data.occupancy[i],
                alt_loc: data.alt_locs[i],
            });
    }

    let mut selected_indices = Vec::new();

    // For each group, select the best atom
    for mut atoms in groups.into_values() {
        // If any atom has blank alt_loc, keep all blank-altloc atoms
        let has_blank = atoms.iter().any(|a| a.alt_loc == ' ');
        if has_blank {
            for atom in atoms {
                if atom.alt_loc == ' ' {
                    selected_indices.push(atom.index);
                }
            }
        } else {
            // No blank altloc: sort by occupancy (descending), then by alt_loc (ascending)
            atoms.sort_by(|a, b| {
                b.occupancy
                    .partial_cmp(&a.occupancy)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.alt_loc.cmp(&b.alt_loc))
            });
            // Keep the first (highest occupancy, alphabetically first on tie)
            selected_indices.push(atoms[0].index);
        }
    }

    // Sort indices to preserve order from original data
    selected_indices.sort_unstable();
    selected_indices
}

impl Topology {
    pub fn from_raw_atom_data(data: &RawAtomData) -> Self {
        let mut chains = Vec::new();

        if data.num_atoms == 0 {
            return Topology { chains };
        }

        // Filter atoms to keep only highest-occupancy altloc per position
        let selected_indices = filter_altlocs(data);
        let selected_set: std::collections::HashSet<usize> = selected_indices.into_iter().collect();

        for i in 0..data.num_atoms {
            // Skip atoms that didn't survive altloc filtering
            if !selected_set.contains(&i) {
                continue;
            }
            let atom = Atom {
                name: data.atom_names[i].clone(),
                element: data.elements[i].clone(),
                coords: [
                    data.coords[i * 3],
                    data.coords[i * 3 + 1],
                    data.coords[i * 3 + 2],
                ],
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
            if chain.residues.is_empty()
                || chain.residues.last().unwrap().res_id != res_id
                || chain.residues.last().unwrap().insertion_code != ins_code
                || chain.residues.last().unwrap().name != *res_name
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

    #[test]
    fn test_altloc_selection_keeps_highest_occupancy() {
        // Two SG atoms in the same CYS residue: alt_loc A (occupancy 0.6) and B (occupancy 0.4)
        // Only the A copy (highest occupancy) should be kept
        let data = RawAtomData {
            coords: vec![1.0, 2.0, 3.0, 1.1, 2.1, 3.1],
            atom_names: vec!["SG".to_string(), "SG".to_string()],
            elements: vec!["S".to_string(), "S".to_string()],
            serial_numbers: vec![1, 2],
            alt_locs: vec!['A', 'B'],
            res_names: vec!["CYS".to_string(), "CYS".to_string()],
            res_ids: vec![10, 10],
            insertion_codes: vec![' ', ' '],
            chain_ids: vec!["A".to_string(), "A".to_string()],
            b_factors: vec![20.0, 20.0],
            occupancy: vec![0.6, 0.4],
            charges: None,
            radii: None,
            sigmas: None,
            epsilons: None,
            num_atoms: 2,
            is_hetatm: vec![false, false],
        };

        let topology = Topology::from_raw_atom_data(&data);

        // Should have exactly one chain, one residue
        assert_eq!(topology.chains.len(), 1);
        assert_eq!(topology.chains[0].residues.len(), 1);

        // Should have exactly one SG atom
        assert_eq!(topology.chains[0].residues[0].atoms.len(), 1);

        // The remaining atom should be the A copy (higher occupancy)
        let sg_atom = &topology.chains[0].residues[0].atoms[0];
        assert_eq!(sg_atom.name, "SG");
        assert_eq!(sg_atom.alt_loc, 'A');
        assert_eq!(sg_atom.occupancy, 0.6);
    }

    #[test]
    fn test_altloc_selection_ties_broken_alphabetically() {
        // Two SG atoms with same occupancy: A and B should keep A (alphabetical)
        let data = RawAtomData {
            coords: vec![1.0, 2.0, 3.0, 1.1, 2.1, 3.1],
            atom_names: vec!["SG".to_string(), "SG".to_string()],
            elements: vec!["S".to_string(), "S".to_string()],
            serial_numbers: vec![1, 2],
            alt_locs: vec!['B', 'A'],
            res_names: vec!["CYS".to_string(), "CYS".to_string()],
            res_ids: vec![10, 10],
            insertion_codes: vec![' ', ' '],
            chain_ids: vec!["A".to_string(), "A".to_string()],
            b_factors: vec![20.0, 20.0],
            occupancy: vec![0.5, 0.5],
            charges: None,
            radii: None,
            sigmas: None,
            epsilons: None,
            num_atoms: 2,
            is_hetatm: vec![false, false],
        };

        let topology = Topology::from_raw_atom_data(&data);

        // Should have exactly one SG atom
        assert_eq!(topology.chains[0].residues[0].atoms.len(), 1);

        // The remaining atom should be the A copy (alphabetically first)
        let sg_atom = &topology.chains[0].residues[0].atoms[0];
        assert_eq!(sg_atom.alt_loc, 'A');
    }

    #[test]
    fn test_altloc_selection_keeps_blank_altloc_unconditionally() {
        // Atom with blank alt_loc (no alternate) should be kept even if other altlocs exist
        let data = RawAtomData {
            coords: vec![1.0, 2.0, 3.0, 1.1, 2.1, 3.1],
            atom_names: vec!["SG".to_string(), "SG".to_string()],
            elements: vec!["S".to_string(), "S".to_string()],
            serial_numbers: vec![1, 2],
            alt_locs: vec![' ', 'A'],
            res_names: vec!["CYS".to_string(), "CYS".to_string()],
            res_ids: vec![10, 10],
            insertion_codes: vec![' ', ' '],
            chain_ids: vec!["A".to_string(), "A".to_string()],
            b_factors: vec![20.0, 20.0],
            occupancy: vec![0.5, 0.8],
            charges: None,
            radii: None,
            sigmas: None,
            epsilons: None,
            num_atoms: 2,
            is_hetatm: vec![false, false],
        };

        let topology = Topology::from_raw_atom_data(&data);

        // Should have exactly one SG atom
        assert_eq!(topology.chains[0].residues[0].atoms.len(), 1);

        // The remaining atom should be the blank alt_loc copy
        let sg_atom = &topology.chains[0].residues[0].atoms[0];
        assert_eq!(sg_atom.alt_loc, ' ');
        assert_eq!(sg_atom.occupancy, 0.5);
    }
}
