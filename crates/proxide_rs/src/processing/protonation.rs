use proxide_geometry::geometry::cell_list::CellList;
use crate::processing::residues::ProcessedStructure;

/// Normalizes a single residue's name across both the ResidueInfo and its constituent RawAtoms
fn rename_residue(processed: &mut ProcessedStructure, res_idx: usize, new_name: &str) {
    processed.residue_info[res_idx].res_name = new_name.to_string();
    let start = processed.residue_info[res_idx].start_atom;
    let num = processed.residue_info[res_idx].num_atoms;
    for i in start..(start + num) {
        processed.raw_atoms.res_names[i] = new_name.to_string();
    }
}

/// Determine protonation states for titratable residues based on local topology and pH.
/// Follows OpenMM semantics for HIS, CYS, ASP, GLU, LYS rules.
pub fn determine_protonation_states(
    processed: &mut ProcessedStructure,
    ph: f32,
) -> Result<(), String> {
    // Check if we even need to bother with topology checks
    let has_titratable = processed
        .residue_info
        .iter()
        .any(|r| matches!(r.res_name.as_str(), "HIS" | "CYS" | "ASP" | "GLU" | "LYS"));

    if !has_titratable {
        return Ok(());
    }

    let all_coords = processed.extract_all_coords();
    // Use a reasonable cutoff for H-bonding and disulfide checks (3.5 Angstroms is typical max for H-bond, we use 4.0 (0.4nm) for safety)
    let cell_list = CellList::new(&all_coords, 0.4);

    let mut modifications = 0;

    // We can't mutate `processed.residue_info` while iterating over it safely if we also need to access `raw_atoms` and other things.
    // So we'll run a pass to gather updates.
    let mut updates = Vec::new();

    for (res_idx, res_info) in processed.residue_info.iter().enumerate() {
        let name = res_info.res_name.as_str();

        match name {
            "ASP" => {
                let pka = 4.4;
                if ph < pka {
                    updates.push((res_idx, "ASH"));
                }
            }
            "GLU" => {
                let pka = 4.4;
                if ph < pka {
                    updates.push((res_idx, "GLH"));
                }
            }
            "LYS" => {
                let pka = 10.5;
                if ph > pka {
                    updates.push((res_idx, "LYN"));
                }
            }
            "CYS" => {
                // Check for disulfide bridge
                // Find SG atom
                let mut sg_idx = None;
                for i in res_info.start_atom..(res_info.start_atom + res_info.num_atoms) {
                    if processed.raw_atoms.atom_names[i] == "SG" {
                        sg_idx = Some(i);
                        break;
                    }
                }

                if let Some(sg) = sg_idx {
                    // CYS SG-SG disulfide bond distance is ~2.05 A (0.205 nm). We look up to 2.5 A (0.25 nm) for safety.
                    let max_dist_sq = 0.25 * 0.25;
                    let neighbors = cell_list.query_neighbors(&all_coords[sg], &all_coords, 0.25);
                    let mut is_disulfide = false;
                    for nb in neighbors {
                        // Ensure it's not the same atom and not in the same residue
                        if nb < res_info.start_atom
                            || nb >= res_info.start_atom + res_info.num_atoms
                        {
                            let dist_sq = distance_sq(&all_coords[sg], &all_coords[nb]);
                            // Ensure it's another Sulfur (or at least heavy atom for general bridging, but typically another CYS SG)
                            if dist_sq < max_dist_sq
                                && processed.raw_atoms.elements[nb].to_uppercase() == "S"
                            {
                                is_disulfide = true;
                                break;
                            }
                        }
                    }

                    if is_disulfide {
                        updates.push((res_idx, "CYX"));
                    }
                }
            }
            "HIS" => {
                // HIS determination is the most complex
                let result =
                    determine_histidine_state(res_info, processed, &all_coords, &cell_list, ph);
                updates.push((res_idx, result));
            }
            _ => {}
        }
    }

    for (res_idx, new_name) in updates {
        rename_residue(processed, res_idx, new_name);
        modifications += 1;
    }

    if modifications > 0 {
        log::debug!(
            "Normalized {} titratable residues based on pH {}",
            modifications,
            ph
        );
    }

    Ok(())
}

fn determine_histidine_state(
    res_info: &crate::processing::residues::ResidueInfo,
    processed: &ProcessedStructure,
    all_coords: &[[f32; 3]],
    cell_list: &CellList,
    ph: f32,
) -> &'static str {
    // Collect coordinates and statuses for the ring
    let mut cg = None;
    let mut nd1 = None;
    let mut ce1 = None;
    let mut ne2 = None;
    let mut cd2 = None;

    let mut hd1_exists = false;
    let mut he2_exists = false;

    for i in res_info.start_atom..(res_info.start_atom + res_info.num_atoms) {
        let atom_name = processed.raw_atoms.atom_names[i].as_str();
        match atom_name {
            "CG" => cg = Some(i),
            "ND1" => nd1 = Some(i),
            "CE1" => ce1 = Some(i),
            "NE2" => ne2 = Some(i),
            "CD2" => cd2 = Some(i),
            "HD1" => hd1_exists = true,
            "HE2" => he2_exists = true,
            _ => {}
        }
    }

    // 1. Check existing hydrogens first
    if hd1_exists && he2_exists {
        return "HIP";
    } else if hd1_exists {
        return "HID";
    } else if he2_exists {
        return "HIE";
    }

    // 2. Fallback to pH if we can't do structural analysis
    if nd1.is_none() || ne2.is_none() || cg.is_none() || ce1.is_none() || cd2.is_none() {
        return if ph <= 6.5 { "HIP" } else { "HIE" }; // Most common neutral
    }

    let p_cg = &all_coords[cg.unwrap()];
    let p_nd1 = &all_coords[nd1.unwrap()];
    let p_ce1 = &all_coords[ce1.unwrap()];
    let p_ne2 = &all_coords[ne2.unwrap()];
    let p_cd2 = &all_coords[cd2.unwrap()];

    // 3. Low pH default
    if ph <= 6.5 {
        return "HIP";
    }

    // 4. H-Bond Scoring
    // Calculate predicted normal vectors representing the lone pairs.
    // We approximate the H position by bisecting the outer angle.
    // For ND1, bonded to CG and CE1.
    // Vector CG->ND1 + Vector CE1->ND1 is roughly pointing OUT of the ring.
    let vec_out_nd1 = add_vec(&sub_vec(p_nd1, p_cg), &sub_vec(p_nd1, p_ce1));
    let h_nd1_pred = add_vec(p_nd1, &scale_to(vec_out_nd1, 0.1)); // 0.1 nm = 1.0 A

    // For NE2, bonded to CE1 and CD2.
    let vec_out_ne2 = add_vec(&sub_vec(p_ne2, p_ce1), &sub_vec(p_ne2, p_cd2));
    let h_ne2_pred = add_vec(p_ne2, &scale_to(vec_out_ne2, 0.1));

    let nd1_forms_hbond = check_h_bond(
        p_nd1,
        &h_nd1_pred,
        cell_list,
        all_coords,
        processed,
        res_info,
    );
    let ne2_forms_hbond = check_h_bond(
        p_ne2,
        &h_ne2_pred,
        cell_list,
        all_coords,
        processed,
        res_info,
    );

    // If only one forms an H-bond, choose it. Otherwise, HIE is the generic default base.
    if ne2_forms_hbond && !nd1_forms_hbond {
        "HIE"
    } else if nd1_forms_hbond && !ne2_forms_hbond {
        "HID"
    } else {
        "HIE" // Default common tautomer
    }
}

/// Checks if a predicted hydrogen forms a valid H-bond with nearby acceptors
fn check_h_bond(
    donor_pos: &[f32; 3],
    h_pred_pos: &[f32; 3],
    cell_list: &CellList,
    all_coords: &[[f32; 3]],
    processed: &ProcessedStructure,
    res_info: &crate::processing::residues::ResidueInfo,
) -> bool {
    let cutoff = 0.35; // 3.5 A
    let neighbors = cell_list.query_neighbors(donor_pos, all_coords, cutoff);

    for nb in neighbors {
        // Must be external to the residue
        if nb >= res_info.start_atom && nb < res_info.start_atom + res_info.num_atoms {
            continue;
        }

        let element = processed.raw_atoms.elements[nb].to_uppercase();
        if element == "O" || element == "N" {
            let acceptor_pos = &all_coords[nb];
            // Check Angle: angle between (Donor -> H) and (Donor -> Acceptor)
            // It should be small (OpenMM uses < 50 deg D-H-A, meaning H is between Donor and Acceptor)
            // Let's implement Donor-H-Acceptor angle > 130 deg (or < 50 deg for Donor-Acceptor to Donor-H)

            let vec_dh = sub_vec(h_pred_pos, donor_pos);
            let vec_da = sub_vec(acceptor_pos, donor_pos);

            let dot = dot_product(&vec_dh, &vec_da);
            let mag_dh = (dot_product(&vec_dh, &vec_dh)).sqrt();
            let mag_da = (dot_product(&vec_da, &vec_da)).sqrt();

            if mag_dh > 0.0 && mag_da > 0.0 {
                let cos_theta = dot / (mag_dh * mag_da);
                let cos_theta = cos_theta.clamp(-1.0, 1.0);
                let angle_rad = cos_theta.acos();
                // 50 degrees in radians is ~0.872
                if angle_rad < 0.872 {
                    return true;
                }
            }
        }
    }

    false
}

#[inline]
fn distance_sq(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[inline]
fn sub_vec(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add_vec(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn dot_product(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn scale_to(v: [f32; 3], target_mag: f32) -> [f32; 3] {
    let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if mag > 0.0 {
        let scale = target_mag / mag;
        [v[0] * scale, v[1] * scale, v[2] * scale]
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proxide_core::structure::{AtomRecord, RawAtomData};

    fn make_test_structure() -> ProcessedStructure {
        let mut raw = RawAtomData::with_capacity(10);
        // ASP
        raw.add_atom(AtomRecord {
            serial: 1,
            atom_name: "CA".to_string(),
            alt_loc: ' ',
            res_name: "ASP".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 0.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 0.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });
        // LYS
        raw.add_atom(AtomRecord {
            serial: 2,
            atom_name: "CA".to_string(),
            alt_loc: ' ',
            res_name: "LYS".to_string(),
            chain_id: "A".to_string(),
            res_seq: 2,
            i_code: ' ',
            x: 10.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 0.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });
        // CYS 1
        raw.add_atom(AtomRecord {
            serial: 3,
            atom_name: "SG".to_string(),
            alt_loc: ' ',
            res_name: "CYS".to_string(),
            chain_id: "A".to_string(),
            res_seq: 3,
            i_code: ' ',
            x: 20.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 0.0,
            element: "S".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });
        // CYS 2 (close to CYS 1 SG)
        raw.add_atom(AtomRecord {
            serial: 4,
            atom_name: "SG".to_string(),
            alt_loc: ' ',
            res_name: "CYS".to_string(),
            chain_id: "A".to_string(),
            res_seq: 4,
            i_code: ' ',
            x: 20.2,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 0.0,
            element: "S".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });

        ProcessedStructure::from_raw(raw).unwrap()
    }

    #[test]
    fn test_ph_effects() {
        let mut p = make_test_structure();
        // pH 2.0 -> ASP should be ASH, LYS should stay LYS
        determine_protonation_states(&mut p, 2.0).unwrap();
        assert_eq!(p.residue_info[0].res_name, "ASH");
        assert_eq!(p.residue_info[1].res_name, "LYS");

        let mut p2 = make_test_structure();
        // pH 7.0 -> ASP stays ASP, LYS stays LYS
        determine_protonation_states(&mut p2, 7.0).unwrap();
        assert_eq!(p2.residue_info[0].res_name, "ASP");
        assert_eq!(p2.residue_info[1].res_name, "LYS");

        let mut p3 = make_test_structure();
        // pH 12.0 -> ASP stays ASP, LYS becomes LYN
        determine_protonation_states(&mut p3, 12.0).unwrap();
        assert_eq!(p3.residue_info[0].res_name, "ASP");
        assert_eq!(p3.residue_info[1].res_name, "LYN");
    }

    #[test]
    fn test_disulfide() {
        let mut p = make_test_structure();
        // CYS at res 3 and 4 are 0.2nm apart. Should form CYX.
        determine_protonation_states(&mut p, 7.0).unwrap();
        assert_eq!(p.residue_info[2].res_name, "CYX");
        assert_eq!(p.residue_info[3].res_name, "CYX");
    }

    #[test]
    fn test_his_preexisting_h() {
        // Test HID, HIE, HIP overrides based on existing H atoms
        let mut raw = RawAtomData::with_capacity(10);
        raw.add_atom(AtomRecord {
            serial: 1,
            atom_name: "ND1".to_string(),
            alt_loc: ' ',
            res_name: "HIS".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 0.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 0.0,
            element: "N".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });
        raw.add_atom(AtomRecord {
            serial: 2,
            atom_name: "HD1".to_string(),
            alt_loc: ' ',
            res_name: "HIS".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 0.0,
            y: 0.0,
            z: 1.0,
            occupancy: 1.0,
            temp_factor: 0.0,
            element: "H".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });
        let mut p = ProcessedStructure::from_raw(raw).unwrap();
        determine_protonation_states(&mut p, 7.0).unwrap();
        assert_eq!(p.residue_info[0].res_name, "HID");

        let mut raw = RawAtomData::with_capacity(10);
        raw.add_atom(AtomRecord {
            serial: 1,
            atom_name: "NE2".to_string(),
            alt_loc: ' ',
            res_name: "HIS".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 0.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 0.0,
            element: "N".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });
        raw.add_atom(AtomRecord {
            serial: 2,
            atom_name: "HE2".to_string(),
            alt_loc: ' ',
            res_name: "HIS".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 0.0,
            y: 0.0,
            z: 1.0,
            occupancy: 1.0,
            temp_factor: 0.0,
            element: "H".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });
        let mut p = ProcessedStructure::from_raw(raw).unwrap();
        determine_protonation_states(&mut p, 7.0).unwrap();
        assert_eq!(p.residue_info[0].res_name, "HIE");
    }

    #[test]
    fn test_his_h_bond_scoring() {
        // Build a HIS ring
        let mut raw = RawAtomData::with_capacity(10);

        let cg = [0.0, 0.0, 0.0];
        let nd1 = [-1.0, 1.0, 0.0];
        let ce1 = [0.0, 2.0, 0.0];
        let ne2 = [1.0, 1.0, 0.0];
        let cd2 = [1.0, -0.5, 0.0];

        // An Oxygen positioned perfectly to hydrogen bond with NE2
        // H normally points from NE2 outwards
        // vec_out_ne2 = (NE2-CE1) + (NE2-CD2) = ( [1,-1,0] ) + ( [0,1.5,0] ) = [1,0.5,0]
        // normal is approx [0.89, 0.44, 0]
        // Let's stick an Oxygen at [3.0, 2.0, 0.0] which is roughly 2.2A away from NE2 in direction of H
        let oxygen = [2.5, 1.75, 0.0];

        for (i, (name, coord, el)) in [
            ("CG", cg, "C"),
            ("ND1", nd1, "N"),
            ("CE1", ce1, "C"),
            ("NE2", ne2, "N"),
            ("CD2", cd2, "C"),
        ]
        .iter()
        .enumerate()
        {
            raw.add_atom(AtomRecord {
                serial: i as i32 + 1,
                atom_name: name.to_string(),
                alt_loc: ' ',
                res_name: "HIS".to_string(),
                chain_id: "A".to_string(),
                res_seq: 1,
                i_code: ' ',
                x: coord[0],
                y: coord[1],
                z: coord[2],
                occupancy: 1.0,
                temp_factor: 0.0,
                element: el.to_string(),
                charge: None,
                radius: None,
                is_hetatm: false,
            });
        }

        // Add O in another residue
        raw.add_atom(AtomRecord {
            serial: 10,
            atom_name: "O".to_string(),
            alt_loc: ' ',
            res_name: "HOH".to_string(),
            chain_id: "W".to_string(),
            res_seq: 2,
            i_code: ' ',
            x: oxygen[0],
            y: oxygen[1],
            z: oxygen[2],
            occupancy: 1.0,
            temp_factor: 0.0,
            element: "O".to_string(),
            charge: None,
            radius: None,
            is_hetatm: true,
        });

        let mut p = ProcessedStructure::from_raw(raw).unwrap();
        determine_protonation_states(&mut p, 7.0).unwrap();

        // Since O is matched to NE2 but not ND1, we expect HIE
        assert_eq!(p.residue_info[0].res_name, "HIE");
    }
}
