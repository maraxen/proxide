//! PDB file format parser
//!  
//! High-performance parser for Protein Data Bank (PDB) files.
//! Returns raw atom data matching biotite's AtomArray format.
//!
//! # Examples
//! ```
//! use oxidize::formats::pdb::parse_pdb_file;
//! let data = parse_pdb_file("tests/data/1abc.pdb").unwrap();
//! ```

use crate::structure::{AtomRecord, RawAtomData};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parse a PDB ATOM/HETATM line using fixed-width fields
/// Format: https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
fn parse_atom_line(line: &str) -> Option<AtomRecord> {
    if line.len() < 54 {
        return None;
    }

    let record_type = line[0..6].trim();
    if record_type != "ATOM" && record_type != "HETATM" {
        return None;
    }

    // Helper to parse float from fixed-width field
    let parse_f32 = |s: &str| -> Option<f32> { s.trim().parse().ok() };

    // Helper to parse int from fixed-width field
    let parse_i32 = |s: &str| -> Option<i32> { s.trim().parse().ok() };

    Some(AtomRecord {
        serial: parse_i32(&line[6..11])?,
        atom_name: line[12..16].trim().to_string(),
        alt_loc: line.chars().nth(16).unwrap_or(' '),
        res_name: line[17..20].trim().to_string(),
        chain_id: line[21..22].trim().to_string(),
        res_seq: parse_i32(&line[22..26])?,
        i_code: line.chars().nth(26).unwrap_or(' '),
        x: parse_f32(&line[30..38])?,
        y: parse_f32(&line[38..46])?,
        z: parse_f32(&line[46..54])?,
        occupancy: if line.len() >= 60 {
            parse_f32(&line[54..60]).unwrap_or(1.0)
        } else {
            1.0
        },
        temp_factor: if line.len() >= 66 {
            parse_f32(&line[60..66]).unwrap_or(0.0)
        } else {
            0.0
        },
        element: if line.len() >= 78 {
            line[76..78].trim().to_string()
        } else {
            // Infer from atom name (first character)
            line[12..16]
                .trim()
                .chars().next()
                .map(|c| c.to_string())
                .unwrap_or_default()
        },
        charge: None,
        radius: None,
        is_hetatm: record_type == "HETATM",
    })
}

/// Parse PDB file and return raw atom data with model IDs.
/// Parses all models by default. Use `filter_models()` to select specific models.
pub fn parse_pdb_file<P: AsRef<Path>>(
    path: P,
) -> Result<(RawAtomData, Vec<usize>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut raw_data = RawAtomData::new();
    let mut model_ids: Vec<usize> = Vec::new();
    let mut current_model: usize = 1; // Default model 1 if no MODEL record

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();

        if trimmed.starts_with("MODEL") {
            // Parse model number from MODEL record
            if let Some(model_str) = trimmed.get(10..) {
                if let Ok(model_num) = model_str.trim().parse::<usize>() {
                    current_model = model_num;
                }
            }
        } else if trimmed.starts_with("ENDMDL") {
            // Model ends, next atoms belong to next model
            // current_model will be updated by next MODEL record
        } else if trimmed.starts_with("ATOM") || trimmed.starts_with("HETATM") {
            if let Some(atom) = parse_atom_line(&line) {
                raw_data.add_atom(atom);
                model_ids.push(current_model);
            }
        }
    }

    if raw_data.num_atoms == 0 {
        return Err("No atoms found in PDB file".into());
    }

    Ok((raw_data, model_ids))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_atom_line() {
        let line =
            "ATOM      1  N   MET A   1      20.154  29.699   5.276  1.00 49.05           N  ";
        let atom = parse_atom_line(line);
        assert!(atom.is_some());

        let atom = atom.unwrap();
        assert_eq!(atom.serial, 1);
        assert_eq!(atom.atom_name, "N");
        assert_eq!(atom.res_name, "MET");
        assert_eq!(atom.chain_id, "A");
        assert_eq!(atom.res_seq, 1);
        assert!((atom.x - 20.154).abs() < 0.001);
        assert!((atom.temp_factor - 49.05).abs() < 0.01);
        assert_eq!(atom.element, "N");
    }

    #[test]
    fn test_parse_hetatm() {
        let line =
            "HETATM 2242  O   HOH A 301      24.243  16.452  10.158  1.00 20.12           O  ";
        let atom = parse_atom_line(line);
        assert!(atom.is_some());

        let atom = atom.unwrap();
        assert_eq!(atom.atom_name, "O");
        assert_eq!(atom.res_name, "HOH");
        assert_eq!(atom.chain_id, "A");
    }

    #[test]
    fn test_raw_atom_data_accumulation() {
        let mut data = RawAtomData::new();

        let atom1 = AtomRecord {
            serial: 1,
            atom_name: "N".to_string(),
            alt_loc: ' ',
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 1.0,
            y: 2.0,
            z: 3.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "N".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        };

        let atom2 = AtomRecord {
            serial: 2,
            atom_name: "CA".to_string(),
            alt_loc: ' ',
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 4.0,
            y: 5.0,
            z: 6.0,
            occupancy: 1.0,
            temp_factor: 25.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        };

        data.add_atom(atom1);
        data.add_atom(atom2);

        assert_eq!(data.num_atoms, 2);
        assert_eq!(data.coords.len(), 6);
        assert_eq!(data.atom_names.len(), 2);
        assert_eq!(data.atom_names[0], "N");
        assert_eq!(data.atom_names[1], "CA");
        assert_eq!(data.res_names[0], "ALA");
        assert_eq!(data.b_factors[1], 25.0);
    }
}
