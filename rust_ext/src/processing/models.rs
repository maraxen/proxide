//! Multi-model handling for NMR structures and trajectories
//!
//! Provides utilities to parse and filter multiple models from PDB files.

use crate::structure::RawAtomData;

/// Split a multi-model structure into separate RawAtomData for each model.
/// The input `model_ids` must have the same length as the number of atoms.
pub fn split_by_model(raw: &RawAtomData, model_ids: &[usize]) -> Vec<RawAtomData> {
    if model_ids.len() != raw.num_atoms {
        return vec![raw.clone()];
    }

    // Find unique model IDs (preserving order)
    let mut unique_models: Vec<usize> = Vec::new();
    for &model in model_ids {
        if !unique_models.contains(&model) {
            unique_models.push(model);
        }
    }

    // Create a RawAtomData for each model
    unique_models
        .iter()
        .map(|&target_model| {
            let mut model_data = RawAtomData::new();

            for i in 0..raw.num_atoms {
                if model_ids[i] == target_model {
                    // Copy atom data
                    model_data.coords.push(raw.coords[i * 3]);
                    model_data.coords.push(raw.coords[i * 3 + 1]);
                    model_data.coords.push(raw.coords[i * 3 + 2]);
                    model_data.atom_names.push(raw.atom_names[i].clone());
                    model_data.elements.push(raw.elements[i].clone());
                    model_data.serial_numbers.push(raw.serial_numbers[i]);
                    model_data.alt_locs.push(raw.alt_locs[i]);
                    model_data.res_names.push(raw.res_names[i].clone());
                    model_data.res_ids.push(raw.res_ids[i]);
                    model_data.insertion_codes.push(raw.insertion_codes[i]);
                    model_data.chain_ids.push(raw.chain_ids[i].clone());
                    model_data.b_factors.push(raw.b_factors[i]);
                    model_data.occupancy.push(raw.occupancy[i]);
                    model_data.is_hetatm.push(raw.is_hetatm[i]);

                    // Copy optional fields if present
                    if let Some(ref charges) = raw.charges {
                        model_data
                            .charges
                            .get_or_insert_with(Vec::new)
                            .push(charges[i]);
                    }
                    if let Some(ref radii) = raw.radii {
                        model_data.radii.get_or_insert_with(Vec::new).push(radii[i]);
                    }
                    if let Some(ref sigmas) = raw.sigmas {
                        model_data
                            .sigmas
                            .get_or_insert_with(Vec::new)
                            .push(sigmas[i]);
                    }
                    if let Some(ref epsilons) = raw.epsilons {
                        model_data
                            .epsilons
                            .get_or_insert_with(Vec::new)
                            .push(epsilons[i]);
                    }

                    model_data.num_atoms += 1;
                }
            }

            model_data
        })
        .collect()
}

/// Filter atoms to include only specified models.
/// Returns a new RawAtomData with atoms from the requested models.
pub fn filter_models(raw: &RawAtomData, model_ids: &[usize], keep_models: &[usize]) -> RawAtomData {
    if model_ids.len() != raw.num_atoms {
        return raw.clone();
    }

    let mut filtered = RawAtomData::new();

    for i in 0..raw.num_atoms {
        if keep_models.contains(&model_ids[i]) {
            // Copy atom data
            filtered.coords.push(raw.coords[i * 3]);
            filtered.coords.push(raw.coords[i * 3 + 1]);
            filtered.coords.push(raw.coords[i * 3 + 2]);
            filtered.atom_names.push(raw.atom_names[i].clone());
            filtered.elements.push(raw.elements[i].clone());
            filtered.serial_numbers.push(raw.serial_numbers[i]);
            filtered.alt_locs.push(raw.alt_locs[i]);
            filtered.res_names.push(raw.res_names[i].clone());
            filtered.res_ids.push(raw.res_ids[i]);
            filtered.insertion_codes.push(raw.insertion_codes[i]);
            filtered.chain_ids.push(raw.chain_ids[i].clone());
            filtered.b_factors.push(raw.b_factors[i]);
            filtered.occupancy.push(raw.occupancy[i]);
            filtered.is_hetatm.push(raw.is_hetatm[i]);

            // Copy optional fields if present
            if let Some(ref charges) = raw.charges {
                filtered
                    .charges
                    .get_or_insert_with(Vec::new)
                    .push(charges[i]);
            }
            if let Some(ref radii) = raw.radii {
                filtered.radii.get_or_insert_with(Vec::new).push(radii[i]);
            }
            if let Some(ref sigmas) = raw.sigmas {
                filtered.sigmas.get_or_insert_with(Vec::new).push(sigmas[i]);
            }
            if let Some(ref epsilons) = raw.epsilons {
                filtered
                    .epsilons
                    .get_or_insert_with(Vec::new)
                    .push(epsilons[i]);
            }

            filtered.num_atoms += 1;
        }
    }

    filtered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::AtomRecord;

    fn create_atom(serial: i32, model: usize) -> AtomRecord {
        AtomRecord {
            serial,
            atom_name: "CA".to_string(),
            alt_loc: ' ',
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: serial,
            i_code: ' ',
            x: model as f32 * 10.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        }
    }

    #[test]
    fn test_split_by_model() {
        let mut raw = RawAtomData::new();
        // Add 2 atoms for model 1, 2 atoms for model 2
        raw.add_atom(create_atom(1, 1));
        raw.add_atom(create_atom(2, 1));
        raw.add_atom(create_atom(3, 2));
        raw.add_atom(create_atom(4, 2));

        let model_ids = vec![1, 1, 2, 2];
        let models = split_by_model(&raw, &model_ids);

        assert_eq!(models.len(), 2);
        assert_eq!(models[0].num_atoms, 2);
        assert_eq!(models[1].num_atoms, 2);
        // Model 1 coords should have x=10.0, Model 2 should have x=20.0
        assert!((models[0].coords[0] - 10.0).abs() < 0.01);
        assert!((models[1].coords[0] - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_filter_models() {
        let mut raw = RawAtomData::new();
        raw.add_atom(create_atom(1, 1));
        raw.add_atom(create_atom(2, 1));
        raw.add_atom(create_atom(3, 2));
        raw.add_atom(create_atom(4, 3));

        let model_ids = vec![1, 1, 2, 3];
        let filtered = filter_models(&raw, &model_ids, &[1, 3]);

        // Should only keep atoms from models 1 and 3
        assert_eq!(filtered.num_atoms, 3);
    }
}
