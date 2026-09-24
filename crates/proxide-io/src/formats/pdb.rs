//! PDB file format parser
//!
//! High-performance parser for Protein Data Bank (PDB) files.
//! Returns raw atom data matching the proxide format.
//!
//! The actual fixed-column field reading (and its fail-loud error handling)
//! lives in [`crate::formats::pdb_fields`] -- see that module's docs for the
//! full rationale (sprint 24, task 260922_autonomous-loop, debt
//! #1776+#1883, OBS-105 option a). This module is now a thin adapter from
//! [`pdb_fields::PdbAtomRecord`] to the crate-wide [`RawAtomData`] /
//! [`AtomRecord`] shape, plus the public file/reader entry points. There is
//! no atom-dropping path left here: every malformed record aborts the whole
//! parse with a [`crate::formats::pdb_fields::PdbFieldError`] (via `?`, which
//! `Box<dyn std::error::Error>` accepts automatically since `PdbFieldError`
//! implements `std::error::Error`).

use crate::formats::pdb_fields::parse_pdb_records;
use proxide_core::structure::{AtomRecord, RawAtomData};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parse PDB file and return raw atom data with model IDs.
/// Parses all models by default. Use `filter_models()` to select specific models.
pub fn parse_pdb_file<P: AsRef<Path>>(
    path: P,
) -> Result<(RawAtomData, Vec<usize>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    parse_pdb_from_reader(reader)
}

/// Internal parser implementation taking any reader
pub fn parse_pdb_from_reader<R: BufRead>(
    reader: R,
) -> Result<(RawAtomData, Vec<usize>), Box<dyn std::error::Error>> {
    let records = parse_pdb_records(reader)?;

    let mut raw_data = RawAtomData::with_capacity(records.len());
    let mut model_ids: Vec<usize> = Vec::with_capacity(records.len());

    for rec in records {
        model_ids.push(rec.model);
        raw_data.add_atom(AtomRecord {
            serial: rec.serial,
            atom_name: rec.atom_name,
            alt_loc: rec.alt_loc,
            res_name: rec.res_name,
            chain_id: rec.chain_id,
            res_seq: rec.res_seq,
            i_code: rec.i_code,
            x: rec.x,
            y: rec.y,
            z: rec.z,
            occupancy: rec.occupancy,
            temp_factor: rec.temp_factor,
            element: rec.element,
            charge: rec.charge,
            radius: rec.radius,
            is_hetatm: rec.is_hetatm,
        });
    }

    if raw_data.num_atoms == 0 {
        return Err("No atoms found in PDB file".into());
    }

    Ok((raw_data, model_ids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::pdb_fields::PdbFieldError;

    /// Parse a single ATOM/HETATM line through the full reader (mirrors what
    /// the old private `parse_atom_line` unit tests exercised, now routed
    /// through the shared `pdb_fields` reader -- there is no atom-level
    /// parsing left in this module to unit test directly).
    fn parse_one_line(line: &str) -> Result<RawAtomData, Box<dyn std::error::Error>> {
        let (raw, _) = parse_pdb_from_reader(line.as_bytes())?;
        Ok(raw)
    }

    fn downcast_kind(err: &Box<dyn std::error::Error>) -> Option<&PdbFieldError> {
        err.downcast_ref::<PdbFieldError>()
    }

    #[test]
    fn test_parse_atom_line() {
        let line =
            "ATOM      1  N   MET A   1      20.154  29.699   5.276  1.00 49.05           N  ";
        let raw = parse_one_line(line).unwrap();
        assert_eq!(raw.num_atoms, 1);
        assert_eq!(raw.serial_numbers[0], 1);
        assert_eq!(raw.atom_names[0], "N");
        assert_eq!(raw.res_names[0], "MET");
        assert_eq!(raw.chain_ids[0], "A");
        assert_eq!(raw.res_ids[0], 1);
        assert!((raw.coords[0] - 20.154).abs() < 0.001);
        assert!((raw.b_factors[0] - 49.05).abs() < 0.01);
        assert_eq!(raw.elements[0], "N");
    }

    #[test]
    fn test_parse_hetatm() {
        let line =
            "HETATM 2242  O   HOH A 301      24.243  16.452  10.158  1.00 20.12           O  ";
        let raw = parse_one_line(line).unwrap();
        assert_eq!(raw.num_atoms, 1);
        assert_eq!(raw.atom_names[0], "O");
        assert_eq!(raw.res_names[0], "HOH");
        assert_eq!(raw.chain_ids[0], "A");
        assert!(raw.is_hetatm[0]);
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

    #[test]
    fn test_parse_4_char_atom_name() {
        // 1H5' should be captured correctly
        let line =
            "ATOM      1 1H5' ALA A   1      20.154  29.699   5.276  1.00 49.05           H  ";
        let raw = parse_one_line(line).unwrap();
        assert_eq!(raw.atom_names[0], "1H5'");
    }

    #[test]
    fn test_parse_atom_line_edge_cases() {
        // Short line -- was `None` (silently dropped); now a hard,
        // kind-specific error (decision c, OBS-105 option a).
        let err = parse_one_line("ATOM").unwrap_err();
        let kind_err =
            downcast_kind(&err).expect("Box<dyn Error> should downcast to PdbFieldError");
        assert_eq!(
            kind_err.kind,
            crate::formats::pdb_fields::PdbFieldErrorKind::LineTooShort
        );

        // Temp factor fallback and element inference (line ends before the
        // occupancy/B-factor/element columns -- all blank, default applies).
        let line = "ATOM      1  N   ALA A   1      20.154  29.699   5.276";
        let raw = parse_one_line(line).unwrap();
        assert_eq!(raw.b_factors[0], 0.0);
        assert_eq!(raw.elements[0], "N");

        // Explicit occupancy but no temp factor.
        let line = "ATOM      1  N   ALA A   1      20.154  29.699   5.276  1.00";
        let raw = parse_one_line(line).unwrap();
        assert_eq!(raw.occupancy[0], 1.0);
        assert_eq!(raw.b_factors[0], 0.0);
    }

    #[test]
    fn test_malformed_occupancy_is_an_error() {
        // OBS-105 option a, named test 1/2.
        let line =
            "ATOM      1  N   ALA A   1      20.154  29.699   5.276  x.xx 49.05           N  ";
        let err = parse_one_line(line).unwrap_err();
        let kind_err = downcast_kind(&err).unwrap();
        assert_eq!(
            kind_err.kind,
            crate::formats::pdb_fields::PdbFieldErrorKind::Unparseable
        );
        assert_eq!(kind_err.field, "occupancy");
    }

    #[test]
    fn test_blank_occupancy_takes_documented_default() {
        // OBS-105 option a, named test 2/2.
        let line =
            "ATOM      1  N   ALA A   1      20.154  29.699   5.276       49.05           N  ";
        let raw = parse_one_line(line).unwrap();
        assert_eq!(raw.occupancy[0], 1.0);
    }

    #[test]
    fn test_two_letter_element_fallback_not_truncated() {
        // backlog #5052 (prolix): a chloride ion named "Cl" on a short line (no
        // element column at all) used to infer element "C" (first character only)
        // instead of "Cl". Also cover a long-enough line whose element column is
        // present but blank -- that must fall back to name-based inference too,
        // not silently accept an empty element string.
        let short_line = "HETATM 7506  Cl  CL  A 500      12.000   3.000   4.000  1.00  0.00";
        let raw = parse_one_line(short_line).unwrap();
        assert_eq!(raw.atom_names[0], "Cl");
        assert_eq!(raw.elements[0], "Cl");

        let blank_column_line =
            "HETATM 7506  Cl  CL  A 500      12.000   3.000   4.000  1.00  0.00              ";
        let raw = parse_one_line(blank_column_line).unwrap();
        assert_eq!(raw.elements[0], "Cl");
    }

    #[test]
    fn test_pdb_with_ter_and_anisou() {
        let pdb_content =
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ANISOU    1  N   ALA A   1   1000   1000   1000      0      0      0       N
TER       2      ALA A   1
ATOM      3  N   ALA B   1      10.000   0.000   0.000  1.00  0.00           N";

        let (raw_data, _) = parse_pdb_from_reader(pdb_content.as_bytes()).unwrap();

        assert_eq!(raw_data.num_atoms, 2);
        assert_eq!(raw_data.chain_ids[0], "A");
        assert_eq!(raw_data.chain_ids[1], "B");
        assert_eq!(raw_data.res_ids[0], 1);
        assert_eq!(raw_data.res_ids[1], 1);
    }
}
