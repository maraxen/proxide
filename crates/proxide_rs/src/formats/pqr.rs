//! PQR file format parser
//!
//! Parses PQR files which contain atom coordinates along with
//! partial charges and radii (used for electrostatics calculations).
//!
//! Format: Similar to PDB but with charge and radius instead of occupancy/B-factor
//! `ATOM serial name resName chainID resSeq x y z charge radius`

#![allow(dead_code)]

use crate::structure::{AtomRecord, RawAtomData};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parse a PQR ATOM/HETATM line
///
/// PQR format varies but typically:
/// ATOM serial name resName chainID resSeq x y z charge radius
fn parse_pqr_line(line: &str) -> Option<AtomRecord> {
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.is_empty() {
        return None;
    }

    let record_type = parts[0];
    if record_type != "ATOM" && record_type != "HETATM" {
        return None;
    }

    // Minimum: ATOM serial name resName chainID resSeq x y z charge radius
    // That's 11 fields
    if parts.len() < 11 {
        return None;
    }

    let serial = parts[1].parse::<i32>().ok()?;
    let atom_name = parts[2].to_string();
    let res_name = parts[3].to_string();
    let chain_id = parts[4].to_string();

    // Handle insertion codes like "52A", "52B" - extract numeric part
    let res_seq_str = parts[5];
    let res_seq: i32 = res_seq_str
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect::<String>()
        .parse()
        .unwrap_or(0);

    // Extract insertion code if present
    let i_code: char = res_seq_str
        .chars()
        .find(|c| c.is_ascii_alphabetic())
        .unwrap_or(' ');

    let x = parts[6].parse::<f32>().ok()?;
    let y = parts[7].parse::<f32>().ok()?;
    let z = parts[8].parse::<f32>().ok()?;
    let charge = parts[9].parse::<f32>().ok()?;
    let radius = parts[10].parse::<f32>().ok()?;

    // Infer element from atom name
    let element = atom_name
        .chars()
        .next()
        .map(|c| c.to_string())
        .unwrap_or_else(|| "C".to_string());

    Some(AtomRecord {
        serial,
        atom_name,
        alt_loc: ' ',
        res_name,
        chain_id,
        res_seq,
        i_code,
        x,
        y,
        z,
        occupancy: 1.0,
        temp_factor: 0.0,
        element,
        charge: Some(charge),
        radius: Some(radius),
        is_hetatm: record_type == "HETATM",
    })
}

/// Parse PQR file and return raw atom data with charges and radii
pub fn parse_pqr_file<P: AsRef<Path>>(path: P) -> Result<RawAtomData, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut raw_data = RawAtomData::new();

    for line in reader.lines() {
        let line = line?;

        if let Some(atom) = parse_pqr_line(&line) {
            raw_data.add_atom(atom);
        }
    }

    if raw_data.num_atoms == 0 {
        return Err("No atoms found in PQR file".into());
    }

    Ok(raw_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pqr_line() {
        let line = "ATOM      1  N   MET A   1      20.154  29.699   5.276  -0.4157  1.8240";
        let atom = parse_pqr_line(line);

        assert!(atom.is_some());
        let atom = atom.unwrap();
        assert_eq!(atom.serial, 1);
        assert_eq!(atom.atom_name, "N");
        assert_eq!(atom.res_name, "MET");
        assert_eq!(atom.chain_id, "A");
        assert!((atom.charge.unwrap() - (-0.4157)).abs() < 0.001);
        assert!((atom.radius.unwrap() - 1.8240).abs() < 0.001);
    }

    #[test]
    fn test_parse_hetatm() {
        let line = "HETATM 2242  O   HOH A 301      24.243  16.452  10.158  -0.8340  1.5000";
        let atom = parse_pqr_line(line);

        assert!(atom.is_some());
        let atom = atom.unwrap();
        assert!(atom.is_hetatm);
        assert_eq!(atom.res_name, "HOH");
    }

    #[test]
    fn test_parse_insertion_code() {
        let line = "ATOM      5  N   ALA A  52A     14.000  24.000  34.000  -0.500   1.850";
        let atom = parse_pqr_line(line);

        assert!(atom.is_some());
        let atom = atom.unwrap();
        assert_eq!(atom.res_seq, 52);
        assert_eq!(atom.i_code, 'A');
    }
}
