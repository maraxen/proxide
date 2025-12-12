//! Hydrogen addition utilities
//!
//! Adds missing hydrogen atoms to protein structures using a fragment library
//! and geometric superimposition (Kabsch algorithm).
//!
//! Reference: hydride library (biotite-dev/hydride)

use crate::chem::bonds::get_bond_order;
use crate::geometry::fragment_library::{calculate_hydrogen_positions, FragmentLibrary};
use crate::geometry::relax::{relax_hydrogens, RelaxOptions};
use crate::processing::residues::ProcessedStructure;
use crate::structure::AtomRecord;
use std::collections::HashMap;

// Embed the fragment library binary at compile time.
// This ensures the data is always available regardless of working directory.
static FRAGMENTS_BIN: &[u8] = include_bytes!("../../data/fragments.bin");

// Lazy load the fragment library from embedded data
lazy_static::lazy_static! {
    static ref FRAGMENT_LIBRARY: FragmentLibrary = {
        match FragmentLibrary::from_binary(FRAGMENTS_BIN) {
            Ok(lib) => {
                log::info!("Loaded fragment library with {} entries", lib.len());
                lib
            }
            Err(e) => {
                log::warn!("Failed to parse embedded fragment library: {}", e);
                FragmentLibrary::new()
            }
        }
    };
}

/// Add hydrogen atoms to the structure.
///
/// Use `include_hetatm` to restrict to protein only?
/// Currently adds to everything matching fragments.
pub fn add_hydrogens(
    structure: &mut ProcessedStructure,
    bonds: &mut Vec<[usize; 2]>,
) -> Result<usize, String> {
    if structure.raw_atoms.num_atoms == 0 {
        return Ok(0);
    }

    // 1. Build adjacency list for efficient neighbor lookup
    // Map atom_idx -> Vec<(neighbor_idx, bond_type)>
    // Note: inferred bonds don't have types yet, usually single.
    // We assume single unless we define aromatic logic.
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); structure.raw_atoms.num_atoms];
    for bond in bonds.iter() {
        if bond[0] < structure.raw_atoms.num_atoms && bond[1] < structure.raw_atoms.num_atoms {
            adjacency[bond[0]].push(bond[1]);
            adjacency[bond[1]].push(bond[0]);
        }
    }

    // 2. Identify heavy atoms to hydrogenate
    let mut new_atoms: Vec<AtomRecord> = Vec::new();
    let mut new_bonds: Vec<[usize; 2]> = Vec::new();
    let n_original = structure.raw_atoms.num_atoms;

    // We need to know existing hydrogen counts to name new ones (simplistic)
    let mut res_h_counts: HashMap<i32, usize> = HashMap::new();

    // Identify Terminals
    // Map ChainID -> (MinResId, MaxResId)
    // Note: This simple logic assumes residues are sorted by ID in chain.
    let mut chain_min_max: HashMap<String, (i32, i32)> = HashMap::new();
    for i in 0..n_original {
        let chain = &structure.raw_atoms.chain_ids[i];
        let res_id = structure.raw_atoms.res_ids[i];
        let entry = chain_min_max
            .entry(chain.clone())
            .or_insert((i32::MAX, i32::MIN));
        if res_id < entry.0 {
            entry.0 = res_id;
        }
        if res_id > entry.1 {
            entry.1 = res_id;
        }
    }

    for i in 0..n_original {
        let element = &structure.raw_atoms.elements[i].to_uppercase();
        if element == "H" {
            continue;
        }

        // Skip if not in fragment library? handled by calculate_hydrogen_positions returning None

        let neighbors = &adjacency[i];

        let chain_id = &structure.raw_atoms.chain_ids[i];
        let res_id = structure.raw_atoms.res_ids[i];
        let res_name = &structure.raw_atoms.res_names[i];
        let atom_name = &structure.raw_atoms.atom_names[i];

        // Terminal Logic
        let mut charge = 0;
        if let Some(&(min_res, _)) = chain_min_max.get(chain_id) {
            // N-terminal Nitrogen: standard pH 7 -> NH3+ (charge +1, 3 H if single bonds)
            if res_id == min_res && atom_name == "N" {
                charge = 1;
            }
        }

        let stereo = 0; // TODO: Implement stereo detection

        // Determine bond types for neighbors using lookup
        // Filter out H neighbors for the key? Hydride excludes H from fragment definition
        let heavy_neighbors: Vec<usize> = neighbors
            .iter()
            .filter(|&&idx| structure.raw_atoms.elements[idx].to_uppercase() != "H")
            .cloned()
            .collect();

        // Determine bond types and sort to match FragmentKey expectations
        let mut neighbor_data: Vec<(usize, u8)> = Vec::with_capacity(heavy_neighbors.len());
        for &neighbor_idx in &heavy_neighbors {
            let neighbor_name = &structure.raw_atoms.atom_names[neighbor_idx];
            let order = get_bond_order(res_name, atom_name, neighbor_name);
            neighbor_data.push((neighbor_idx, order));
        }

        // Sort by bond order, then by index for stability
        neighbor_data.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let sorted_bond_types: Vec<u8> = neighbor_data.iter().map(|nd| nd.1).collect();
        let sorted_neighbors: Vec<usize> = neighbor_data.iter().map(|nd| nd.0).collect();

        // Get coordinates helper
        let get_coords = |idx: usize| -> [f32; 3] {
            [
                structure.raw_atoms.coords[idx * 3],
                structure.raw_atoms.coords[idx * 3 + 1],
                structure.raw_atoms.coords[idx * 3 + 2],
            ]
        };

        // Prepare heavy coords for alignment (up to 3)
        // We match "hydride" logic: if < 3 neighbors, fill slots intelligently
        // to define a coordinate frame that matches the fragment library's reference.
        let mut heavy_coords = [[0.0; 3]; 3];

        match sorted_neighbors.len() {
            0 => {
                // No existing bonds (e.g. single atom/ion)
                // Use center 3 times
                let c = get_coords(i);
                heavy_coords = [c, c, c];
            }
            1 => {
                // 1 Bond: Use [n1, n1, n2] where n2 is neighbor of n1
                // This defines a plane containing n1-center and n1-n2
                let n1 = sorted_neighbors[0];
                let c1 = get_coords(n1);

                // Find n2: a heavy neighbor of n1 that is NOT current atom (i)
                // Use adjacency list
                let mut c2 = c1; // fallback
                if let Some(n1_neighbors) = adjacency.get(n1) {
                    for &n2 in n1_neighbors {
                        if n2 != i && structure.raw_atoms.elements[n2].to_uppercase() != "H" {
                            c2 = get_coords(n2);
                            break; // just take first valid one
                        }
                    }
                }

                heavy_coords[0] = c1;
                heavy_coords[1] = c1;
                heavy_coords[2] = c2;
            }
            2 => {
                // 2 Bonds: Use [n1, n2, center]
                heavy_coords[0] = get_coords(sorted_neighbors[0]);
                heavy_coords[1] = get_coords(sorted_neighbors[1]);
                heavy_coords[2] = get_coords(i);
            }
            _ => {
                // 3 or more: Use first 3
                for j in 0..3 {
                    heavy_coords[j] = get_coords(sorted_neighbors[j]);
                }
            }
        }

        // Get the central atom's coordinates
        let center_coord = get_coords(i);

        // Calculate H positions
        if let Some(h_coords) = calculate_hydrogen_positions(
            &FRAGMENT_LIBRARY,
            element,
            charge,
            stereo,
            sorted_bond_types,
            center_coord,
            &heavy_coords,
        ) {
            for h_pos in h_coords.iter() {
                // Name generation
                let res_id = structure.raw_atoms.res_ids[i];
                let count = res_h_counts.entry(res_id).or_insert(0);
                *count += 1;
                let atom_name = format!("H{}", count); // TODO: Proper naming

                // Create new atom
                let atom = AtomRecord {
                    serial: (n_original + new_atoms.len() + 1) as i32,
                    atom_name,
                    alt_loc: structure.raw_atoms.alt_locs[i],
                    res_name: structure.raw_atoms.res_names[i].clone(),
                    chain_id: structure.raw_atoms.chain_ids[i].clone(),
                    res_seq: res_id,
                    i_code: structure.raw_atoms.insertion_codes[i],
                    x: h_pos[0],
                    y: h_pos[1],
                    z: h_pos[2],
                    occupancy: 1.0,
                    temp_factor: structure.raw_atoms.b_factors[i],
                    element: "H".to_string(),
                    charge: None, // Will be set by parameterization later
                    radius: None,
                    is_hetatm: structure.raw_atoms.is_hetatm[i],
                };

                new_atoms.push(atom);

                // Bond to parent. Parent is i.
                // New atom index is n_original + new_atoms.len() - 1
                new_bonds.push([i, n_original + new_atoms.len() - 1]);
            }
        }
    }

    let added_count = new_atoms.len();

    // 3. Append new atoms to structure
    for atom in new_atoms {
        structure.raw_atoms.add_atom(atom);
        // Also update molecule_type, etc.?
        // ProcessedStructure might need re-processing or manual update
        // Append 0/1 based on parent
        let molecule_type = structure.molecule_type[new_bonds.last().unwrap()[0]];
        structure.molecule_type.push(molecule_type);
    }

    // 4. Update bonds
    bonds.extend(new_bonds);

    Ok(added_count)
}

/// Add hydrogen atoms and optionally relax their positions.
///
/// # Arguments
/// * `structure` - Structure to add hydrogens to
/// * `bonds` - Bonds array (will be updated with new H bonds)
/// * `relax` - Whether to run energy relaxation
/// * `max_relax_iterations` - Maximum iterations for relaxation (None = until convergence)
///
/// # Returns
/// (num_added, num_iterations, final_energy) if relax=true
/// (num_added, 0, 0.0) if relax=false
pub fn add_hydrogens_with_relax(
    structure: &mut ProcessedStructure,
    bonds: &mut Vec<[usize; 2]>,
    relax: bool,
    max_relax_iterations: Option<usize>,
) -> Result<(usize, usize, f32), String> {
    // First add the hydrogens
    let added_count = add_hydrogens(structure, bonds)?;

    if added_count == 0 || !relax {
        return Ok((added_count, 0, 0.0));
    }

    // Prepare data for relaxation
    let n_atoms = structure.raw_atoms.num_atoms;

    // Extract coordinates as [[f32; 3]]
    let mut coords: Vec<[f32; 3]> = (0..n_atoms)
        .map(|i| {
            [
                structure.raw_atoms.coords[i * 3],
                structure.raw_atoms.coords[i * 3 + 1],
                structure.raw_atoms.coords[i * 3 + 2],
            ]
        })
        .collect();

    // Get elements
    let elements = structure.raw_atoms.elements.clone();

    // Get charges if available
    let charges: Option<Vec<f32>> = structure.raw_atoms.charges.clone();

    // Convert bonds to [[usize; 2]]
    let bond_array: Vec<[usize; 2]> = bonds.clone();

    // Set up relaxation options
    let options = RelaxOptions {
        max_iterations: max_relax_iterations,
        angle_increment: 10.0_f32.to_radians(),
        force_cutoff: 10.0,
    };

    // Run relaxation
    let (iterations, final_energy) = relax_hydrogens(
        &mut coords,
        &elements,
        charges.as_deref(),
        &bond_array,
        &options,
    );

    // Update structure coordinates from relaxed positions
    for i in 0..n_atoms {
        structure.raw_atoms.coords[i * 3] = coords[i][0];
        structure.raw_atoms.coords[i * 3 + 1] = coords[i][1];
        structure.raw_atoms.coords[i * 3 + 2] = coords[i][2];
    }

    log::info!(
        "Relaxed {} hydrogen positions in {} iterations (energy: {:.2})",
        added_count,
        iterations,
        final_energy
    );

    Ok((added_count, iterations, final_energy))
}
