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
use once_cell::sync::OnceCell;
use std::collections::HashMap;

// Embed the fragment library binary at compile time.
// This ensures the data is always available regardless of working directory.
static FRAGMENTS_BIN: &[u8] = include_bytes!("../../data/fragments.bin");

// Fragment library - initialized eagerly at module load to avoid GIL deadlock
static FRAGMENT_LIBRARY: OnceCell<FragmentLibrary> = OnceCell::new();

/// Initialize the fragment library. Call this at module initialization time.
/// This avoids lazy initialization which can deadlock with Python's GIL.
pub fn init_fragment_library() {
    eprintln!("[OXIDIZE H] init_fragment_library: starting...");
    let _ = FRAGMENT_LIBRARY.get_or_init(|| {
        eprintln!(
            "[OXIDIZE H] init_fragment_library: parsing binary ({} bytes)...",
            FRAGMENTS_BIN.len()
        );
        match FragmentLibrary::from_binary(FRAGMENTS_BIN) {
            Ok(lib) => {
                eprintln!(
                    "[OXIDIZE H] init_fragment_library: loaded {} entries",
                    lib.len()
                );
                log::info!("Loaded fragment library with {} entries", lib.len());
                lib
            }
            Err(e) => {
                eprintln!("[OXIDIZE H] init_fragment_library: FAILED - {}", e);
                log::warn!("Failed to parse embedded fragment library: {}", e);
                FragmentLibrary::new()
            }
        }
    });
    eprintln!("[OXIDIZE H] init_fragment_library: done");
}

/// Get the fragment library reference.
/// Uses .get() instead of get_or_init() to avoid potential GIL deadlock issues.
/// The library should already be initialized during module load via init_fragment_library().
fn get_fragment_library() -> &'static FragmentLibrary {
    // Use get() to avoid blocking - library should be initialized at module load
    FRAGMENT_LIBRARY.get().unwrap_or_else(|| {
        // Fallback: This should never happen if init_fragment_library() was called
        eprintln!("[OXIDIZE H] WARNING: Fragment library not initialized, returning empty library");
        // Return a static empty library as last resort
        static EMPTY: once_cell::sync::OnceCell<FragmentLibrary> = once_cell::sync::OnceCell::new();
        EMPTY.get_or_init(FragmentLibrary::new)
    })
}

/// Add hydrogen atoms to the structure.
///
/// Uses sequential computation for hydrogen position calculations
/// (parallel was causing GIL deadlock issues with PyO3).
pub fn add_hydrogens(
    structure: &mut ProcessedStructure,
    bonds: &mut Vec<[usize; 2]>,
) -> Result<usize, String> {
    // File-based debug logging for hang investigation
    {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/oxidize_debug.log")
        {
            let _ = writeln!(
                f,
                "[hydrogens.rs] add_hydrogens ENTRY: {} atoms, {} bonds",
                structure.raw_atoms.num_atoms,
                bonds.len()
            );
            let _ = f.flush();
        }
    }

    log::debug!(
        "add_hydrogens: starting with {} atoms, {} bonds",
        structure.raw_atoms.num_atoms,
        bonds.len()
    );

    if structure.raw_atoms.num_atoms == 0 {
        return Ok(0);
    }

    let n_original = structure.raw_atoms.num_atoms;

    // 1. Build adjacency list for efficient neighbor lookup
    eprintln!("[OXIDIZE H] Step 1: Building adjacency list...");
    log::debug!("add_hydrogens: building adjacency list...");
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_original];
    for bond in bonds.iter() {
        if bond[0] < n_original && bond[1] < n_original {
            adjacency[bond[0]].push(bond[1]);
            adjacency[bond[1]].push(bond[0]);
        }
    }
    log::debug!("add_hydrogens: adjacency list built");

    // 2. Identify terminal residues per chain
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

    // 3. Prepare read-only data for parallel iteration
    // Extract all data needed for H calculation (avoids mutable borrow conflicts)
    let elements: Vec<String> = structure.raw_atoms.elements.iter().cloned().collect();
    let coords: Vec<f32> = structure.raw_atoms.coords.clone();
    let chain_ids: Vec<String> = structure.raw_atoms.chain_ids.iter().cloned().collect();
    let res_ids: Vec<i32> = structure.raw_atoms.res_ids.clone();
    let res_names: Vec<String> = structure.raw_atoms.res_names.iter().cloned().collect();
    let atom_names: Vec<String> = structure.raw_atoms.atom_names.iter().cloned().collect();
    let alt_locs: Vec<char> = structure.raw_atoms.alt_locs.clone();
    let insertion_codes: Vec<char> = structure.raw_atoms.insertion_codes.clone();
    let b_factors: Vec<f32> = structure.raw_atoms.b_factors.clone();
    let is_hetatm: Vec<bool> = structure.raw_atoms.is_hetatm.clone();

    // Helper to get coords
    let get_coords =
        |idx: usize| -> [f32; 3] { [coords[idx * 3], coords[idx * 3 + 1], coords[idx * 3 + 2]] };

    // 4. Calculate hydrogen positions for each heavy atom (sequential to avoid GIL deadlock)
    // Returns: Vec<(parent_idx, res_id, Vec<[f32; 3]>)>
    eprintln!(
        "[OXIDIZE H] Step 4: Calculating H positions for {} atoms...",
        n_original
    );
    log::debug!(
        "add_hydrogens: starting H position calculation for {} atoms...",
        n_original
    );
    let h_positions: Vec<(usize, i32, Vec<[f32; 3]>)> = (0..n_original)
        .into_iter()
        .filter_map(|i| {
            let element = elements[i].to_uppercase();
            if element == "H" {
                return None;
            }

            let neighbors = &adjacency[i];
            let chain_id = &chain_ids[i];
            let res_id = res_ids[i];
            let res_name = &res_names[i];
            let atom_name = &atom_names[i];

            // Terminal logic
            let mut charge = 0;
            if let Some(&(min_res, _)) = chain_min_max.get(chain_id) {
                if res_id == min_res && atom_name == "N" {
                    charge = 1;
                }
            }

            let stereo = 0;

            // Filter heavy neighbors
            let heavy_neighbors: Vec<usize> = neighbors
                .iter()
                .filter(|&&idx| elements[idx].to_uppercase() != "H")
                .cloned()
                .collect();

            // Determine bond types
            let mut neighbor_data: Vec<(usize, u8)> = Vec::with_capacity(heavy_neighbors.len());
            for &neighbor_idx in &heavy_neighbors {
                let neighbor_name = &atom_names[neighbor_idx];
                let order = get_bond_order(res_name, atom_name, neighbor_name);
                neighbor_data.push((neighbor_idx, order));
            }

            neighbor_data.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

            let sorted_bond_types: Vec<u8> = neighbor_data.iter().map(|nd| nd.1).collect();
            let sorted_neighbors: Vec<usize> = neighbor_data.iter().map(|nd| nd.0).collect();

            // Prepare heavy coords for alignment
            let mut heavy_coords = [[0.0; 3]; 3];
            match sorted_neighbors.len() {
                0 => {
                    let c = get_coords(i);
                    heavy_coords = [c, c, c];
                }
                1 => {
                    let n1 = sorted_neighbors[0];
                    let c1 = get_coords(n1);
                    let mut c2 = c1;
                    for &n2 in &adjacency[n1] {
                        if n2 != i && elements[n2].to_uppercase() != "H" {
                            c2 = get_coords(n2);
                            break;
                        }
                    }
                    heavy_coords[0] = c1;
                    heavy_coords[1] = c1;
                    heavy_coords[2] = c2;
                }
                2 => {
                    heavy_coords[0] = get_coords(sorted_neighbors[0]);
                    heavy_coords[1] = get_coords(sorted_neighbors[1]);
                    heavy_coords[2] = get_coords(i);
                }
                _ => {
                    for j in 0..3 {
                        heavy_coords[j] = get_coords(sorted_neighbors[j]);
                    }
                }
            }

            let center_coord = get_coords(i);

            // Calculate H positions
            if let Some(h_coords) = calculate_hydrogen_positions(
                get_fragment_library(),
                &element,
                charge,
                stereo,
                sorted_bond_types,
                center_coord,
                &heavy_coords,
            ) {
                if !h_coords.is_empty() {
                    return Some((i, res_id, h_coords));
                }
            }

            None
        })
        .collect();

    eprintln!(
        "[OXIDIZE H] Step 4 done: calculated {} H placement sites",
        h_positions.len()
    );
    log::debug!(
        "add_hydrogens: calculated {} H placement sites",
        h_positions.len()
    );

    // 5. SEQUENTIAL: Append atoms, update bonds, and fix residue_info
    // Count hydrogens per residue for naming
    let mut res_h_counts: HashMap<i32, usize> = HashMap::new();

    // Track hydrogens added per residue for residue_info update
    let mut res_h_added: HashMap<i32, usize> = HashMap::new();
    let mut new_atoms: Vec<AtomRecord> = Vec::new();
    let mut new_bonds: Vec<[usize; 2]> = Vec::new();

    for (parent_idx, res_id, h_coords) in h_positions {
        for h_pos in h_coords {
            let count = res_h_counts.entry(res_id).or_insert(0);
            *count += 1;
            let h_name = format!("H{}", count);

            let atom = AtomRecord {
                serial: (n_original + new_atoms.len() + 1) as i32,
                atom_name: h_name,
                alt_loc: alt_locs[parent_idx],
                res_name: res_names[parent_idx].clone(),
                chain_id: chain_ids[parent_idx].clone(),
                res_seq: res_id,
                i_code: insertion_codes[parent_idx],
                x: h_pos[0],
                y: h_pos[1],
                z: h_pos[2],
                occupancy: 1.0,
                temp_factor: b_factors[parent_idx],
                element: "H".to_string(),
                charge: None,
                radius: None,
                is_hetatm: is_hetatm[parent_idx],
            };

            new_atoms.push(atom);
            new_bonds.push([parent_idx, n_original + new_atoms.len() - 1]);

            // Track for residue_info update
            *res_h_added.entry(res_id).or_insert(0) += 1;
        }
    }

    let added_count = new_atoms.len();

    // 6. Append atoms to structure
    for (idx, atom) in new_atoms.into_iter().enumerate() {
        structure.raw_atoms.add_atom(atom);
        // Copy molecule_type from parent
        let parent_idx = new_bonds[idx][0];
        let molecule_type = structure.molecule_type[parent_idx];
        structure.molecule_type.push(molecule_type);
    }

    // 7. Update residue_info.num_atoms for each residue that received hydrogens
    for res_info in structure.residue_info.iter_mut() {
        if let Some(&h_count) = res_h_added.get(&res_info.res_id) {
            res_info.num_atoms += h_count;
        }
    }

    // 8. Extend bonds
    bonds.extend(new_bonds);

    log::debug!("Added {} hydrogens in parallel", added_count);

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
