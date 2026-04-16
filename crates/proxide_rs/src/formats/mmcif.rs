//! mmCIF/PDBx file format parser
//!
//! High-performance parser for macromolecular Crystallographic Information File (mmCIF) format.
//! mmCIF is the primary deposition format for the Protein Data Bank.
//!
//! Format reference: https://mmcif.wwpdb.org/

use crate::structure::{AtomRecord, RawAtomData};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parse mmCIF file and return raw atom data with model IDs
pub fn parse_mmcif_file<P: AsRef<Path>>(
    path: P,
) -> Result<(RawAtomData, Vec<usize>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut raw_data = RawAtomData::new();
    let mut model_ids: Vec<usize> = Vec::new();
    let mut in_atom_site_loop = false;
    let mut column_names: Vec<String> = Vec::new();
    let mut column_map: HashMap<String, usize> = HashMap::new();
    let mut current_model = 1;

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Detect start of _atom_site loop
        if trimmed == "loop_" {
            // Reset - next items are column definitions
            in_atom_site_loop = false;
            column_names.clear();
            column_map.clear();
            continue;
        }

        // Collect column names for _atom_site
        if trimmed.starts_with("_atom_site.") {
            let col_name = trimmed.trim_start_matches("_atom_site.");
            column_map.insert(col_name.to_string(), column_names.len());
            column_names.push(col_name.to_string());
            in_atom_site_loop = true;
            continue;
        }

        // If we hit a different category, we're done with atom_site
        if trimmed.starts_with('_') && !trimmed.starts_with("_atom_site") {
            if in_atom_site_loop && !raw_data.coords.is_empty() {
                // We have data, stop parsing
                break;
            }
            in_atom_site_loop = false;
            column_names.clear();
            continue;
        }

        // Parse data rows in atom_site loop
        if in_atom_site_loop
            && !column_names.is_empty()
            && !trimmed.starts_with('_')
            && !trimmed.starts_with("loop_")
        {
            // Parse the data row
            let values = parse_cif_values(trimmed);

            if values.len() != column_names.len() {
                continue; // Skip malformed lines
            }

            // Get model number if available
            if let Some(&model_idx) = column_map.get("pdbx_PDB_model_num") {
                if let Ok(model_num) = values[model_idx].parse::<usize>() {
                    current_model = model_num;
                }
            }

            // Extract atom record
            if let Some(atom) = extract_atom_record(&values, &column_map) {
                raw_data.add_atom(atom);
                model_ids.push(current_model);
            }
        }
    }

    if raw_data.num_atoms == 0 {
        return Err("No atoms found in mmCIF file".into());
    }

    Ok((raw_data, model_ids))
}

/// Parse a CIF data line into values, handling quoted strings
fn parse_cif_values(line: &str) -> Vec<&str> {
    let mut values = Vec::new();
    let mut chars = line.char_indices().peekable();
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut start = 0;
    let mut last_end = 0;

    while let Some((i, c)) = chars.next() {
        match c {
            '\'' if !in_double_quote => {
                if !in_single_quote {
                    in_single_quote = true;
                    start = i + 1;
                } else {
                    // Check if next char is whitespace or end
                    if chars.peek().is_none_or(|(_, nc)| nc.is_whitespace()) {
                        values.push(&line[start..i]);
                        in_single_quote = false;
                        last_end = i + 1;
                    }
                }
            }
            '"' if !in_single_quote => {
                if !in_double_quote {
                    in_double_quote = true;
                    start = i + 1;
                } else if chars.peek().is_none_or(|(_, nc)| nc.is_whitespace()) {
                    values.push(&line[start..i]);
                    in_double_quote = false;
                    last_end = i + 1;
                }
            }
            ' ' | '\t' if !in_single_quote && !in_double_quote => {
                if i > last_end {
                    values.push(&line[last_end..i]);
                }
                last_end = i + 1;
                start = i + 1;
            }
            _ => {}
        }
    }

    // Handle last value
    if last_end < line.len() && !in_single_quote && !in_double_quote {
        let remaining = line[last_end..].trim();
        if !remaining.is_empty() {
            values.push(remaining);
        }
    }

    values
}

/// Extract an AtomRecord from parsed CIF values
fn extract_atom_record(values: &[&str], column_map: &HashMap<String, usize>) -> Option<AtomRecord> {
    // Helper to get value by column name
    let get_val =
        |name: &str| -> Option<&str> { column_map.get(name).and_then(|&i| values.get(i).copied()) };

    let get_str = |name: &str| -> String {
        get_val(name)
            .map(|s| if s == "." || s == "?" { "" } else { s })
            .unwrap_or("")
            .to_string()
    };

    let get_i32 = |name: &str| -> Option<i32> { get_val(name).and_then(|s| s.parse().ok()) };

    let get_f32 = |name: &str| -> Option<f32> { get_val(name).and_then(|s| s.parse().ok()) };

    // Required fields
    let x = get_f32("Cartn_x")?;
    let y = get_f32("Cartn_y")?;
    let z = get_f32("Cartn_z")?;

    // Check group_PDB for HETATM
    let group_pdb = get_str("group_PDB");
    let is_hetatm = group_pdb == "HETATM";

    Some(AtomRecord {
        serial: get_i32("id").unwrap_or(0),
        atom_name: get_str("label_atom_id"),
        alt_loc: get_str("label_alt_id").chars().next().unwrap_or(' '),
        res_name: get_str("label_comp_id"),
        chain_id: get_str("label_asym_id"),
        res_seq: get_i32("label_seq_id")
            .or_else(|| get_i32("auth_seq_id"))
            .unwrap_or(0),
        i_code: get_str("pdbx_PDB_ins_code").chars().next().unwrap_or(' '),
        x,
        y,
        z,
        occupancy: get_f32("occupancy").unwrap_or(1.0),
        temp_factor: get_f32("B_iso_or_equiv").unwrap_or(0.0),
        element: get_str("type_symbol"),
        charge: None,
        radius: None,
        is_hetatm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cif_values_simple() {
        let line =
            "ATOM 1 N N . ALA A 1 1 ? 20.154 29.699 5.276 1.00 49.05 ? ? ? ? ? ? 1 ALA A N 1";
        let values = parse_cif_values(line);
        assert!(values.len() > 10);
        assert_eq!(values[0], "ATOM");
        assert_eq!(values[1], "1");
    }

    #[test]
    fn test_parse_cif_values_quoted() {
        let line = "ATOM 1 'C1' 'test name' A";
        let values = parse_cif_values(line);
        assert_eq!(values[0], "ATOM");
        assert_eq!(values[2], "C1");
        assert_eq!(values[3], "test name");
    }

    #[test]
    fn test_extract_atom_record() {
        let values = vec![
            "ATOM", "1", "N", "N", ".", "ALA", "A", "1", "1", "?", "20.154", "29.699", "5.276",
            "1.00", "49.05", "N",
        ];
        let mut column_map = HashMap::new();
        column_map.insert("group_PDB".to_string(), 0);
        column_map.insert("id".to_string(), 1);
        column_map.insert("type_symbol".to_string(), 2);
        column_map.insert("label_atom_id".to_string(), 3);
        column_map.insert("label_alt_id".to_string(), 4);
        column_map.insert("label_comp_id".to_string(), 5);
        column_map.insert("label_asym_id".to_string(), 6);
        column_map.insert("label_seq_id".to_string(), 7);
        column_map.insert("pdbx_PDB_ins_code".to_string(), 8);
        column_map.insert("pdbx_PDB_model_num".to_string(), 9);
        column_map.insert("Cartn_x".to_string(), 10);
        column_map.insert("Cartn_y".to_string(), 11);
        column_map.insert("Cartn_z".to_string(), 12);
        column_map.insert("occupancy".to_string(), 13);
        column_map.insert("B_iso_or_equiv".to_string(), 14);

        let atom = extract_atom_record(&values, &column_map);
        assert!(atom.is_some());

        let atom = atom.unwrap();
        assert_eq!(atom.serial, 1);
        assert_eq!(atom.atom_name, "N");
        assert_eq!(atom.res_name, "ALA");
        assert!((atom.x - 20.154).abs() < 0.001);
    }
}
