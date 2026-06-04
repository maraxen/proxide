//! Parse CHARMM RTF IC section into ResidueGeometryTable protobuf.
//!
//! Reads CHARMM36 top_all36_prot.rtf file, extracts IC (internal coordinates)
//! records from RESI blocks, and converts them into a ResidueGeometryTable proto
//! with source, version, license, and citation metadata.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::fs::File;
use crate::RotlibError;
use crate::pb::proxide::rotlib::v1::{ResidueGeometryTable, ResidueGeometry, IcRecord};

/// Parse CHARMM RTF file and extract IC geometry table.
///
/// # Arguments
/// * `rtf_path` - Path to the CHARMM top_all36_prot.rtf file
///
/// # Returns
/// `ResidueGeometryTable` with source="charmm36", containing all parsed IC records
/// grouped by residue name.
///
/// # Errors
/// Returns `RotlibError::InvalidFormat` if the file cannot be read or parsed.
pub fn parse_rtf_ic_table(rtf_path: &str) -> Result<ResidueGeometryTable, RotlibError> {
    let file = File::open(rtf_path)
        .map_err(|e| RotlibError::InvalidFormat(format!("Failed to open RTF file: {}", e)))?;
    let reader = BufReader::new(file);

    let mut residues_map: HashMap<String, Vec<IcRecord>> = HashMap::new();
    let mut current_resi: Option<String> = None;

    for line in reader.lines() {
        let line = line
            .map_err(|e| RotlibError::InvalidFormat(format!("IO error reading RTF: {}", e)))?;

        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('!') {
            continue;
        }

        // Track current RESI block
        if trimmed.starts_with("RESI ") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                current_resi = Some(parts[1].to_string());
                if let Some(ref resi_name) = current_resi {
                    residues_map.insert(resi_name.clone(), Vec::new());
                }
            }
            continue;
        }

        // Skip PRES blocks (patch residues)
        if trimmed.starts_with("PRES ") {
            current_resi = None;
            continue;
        }

        // Parse IC records
        if trimmed.starts_with("IC ") {
            if let Some(ref resi_name) = current_resi {
                if let Ok(ic_record) = parse_ic_line(trimmed) {
                    // Skip dummy coordinates (0.0000 values)
                    if ic_record.b_ij > 0.0 && ic_record.b_kl > 0.0 {
                        residues_map.entry(resi_name.clone())
                            .and_modify(|v: &mut Vec<IcRecord>| v.push(ic_record.clone()))
                            .or_insert_with(|| vec![ic_record]);
                    }
                }
            }
        }
    }

    // Convert map to sorted ResidueGeometry messages
    let mut residues = Vec::new();
    let mut resi_names: Vec<_> = residues_map.keys().cloned().collect();
    resi_names.sort();

    for resi_name in resi_names {
        residues.push(ResidueGeometry {
            name: resi_name.clone(),
            ic: residues_map.remove(&resi_name).unwrap_or_default(),
        });
    }

    Ok(ResidueGeometryTable {
        source: "charmm36".to_string(),
        version: "charmm36_rtf".to_string(),
        license: "NOT-OSI: MacKerell lab academic use only; see https://mackerell.umaryland.edu/charmm_ff.shtml".to_string(),
        citation: "doi:10.1021/jp973084f".to_string(),
        residues,
    })
}

/// Parse a single IC line from RTF format.
///
/// Format: IC atom_i atom_j [*]atom_k atom_l b_ij theta_ijk phi_ijkl theta_jkl b_kl
///
/// The * prefix on atom_k marks a branch point.
fn parse_ic_line(line: &str) -> Result<IcRecord, RotlibError> {
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.len() < 10 {
        return Err(RotlibError::InvalidFormat(
            format!("IC line has {} fields, need at least 10", parts.len()),
        ));
    }

    // Skip "IC" prefix
    let atom_i = parts[1].to_string();
    let atom_j = parts[2].to_string();

    // atom_k may have * prefix
    let atom_k_raw = parts[3];
    let branch = atom_k_raw.starts_with('*');
    let atom_k = if branch {
        atom_k_raw[1..].to_string()
    } else {
        atom_k_raw.to_string()
    };

    let atom_l = parts[4].to_string();

    // Parse 5 floats: b_ij, theta_ijk, phi, theta_jkl, b_kl
    let b_ij: f32 = parts[5]
        .parse()
        .map_err(|_| RotlibError::InvalidFormat(format!("Invalid float: {}", parts[5])))?;
    let theta_ijk: f32 = parts[6]
        .parse()
        .map_err(|_| RotlibError::InvalidFormat(format!("Invalid float: {}", parts[6])))?;
    let phi_ijkl: f32 = parts[7]
        .parse()
        .map_err(|_| RotlibError::InvalidFormat(format!("Invalid float: {}", parts[7])))?;
    let theta_jkl: f32 = parts[8]
        .parse()
        .map_err(|_| RotlibError::InvalidFormat(format!("Invalid float: {}", parts[8])))?;
    let b_kl: f32 = parts[9]
        .parse()
        .map_err(|_| RotlibError::InvalidFormat(format!("Invalid float: {}", parts[9])))?;

    Ok(IcRecord {
        atom_i,
        atom_j,
        atom_k,
        atom_l,
        branch,
        b_ij,
        theta_ijk,
        phi_ijkl,
        theta_jkl,
        b_kl,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rtf_ic_table_real_file() {
        // Test with the actual CHARMM36 RTF file
        let rtf_path = "/home/marielle/.local/share/mamba/envs/espaloma-bench/dat/chamber/top_all36_prot.rtf";
        let result = parse_rtf_ic_table(rtf_path);

        assert!(result.is_ok(), "Failed to parse RTF: {:?}", result);
        let table = result.unwrap();

        assert_eq!(table.source, "charmm36");
        assert_eq!(table.version, "charmm36_rtf");
        assert!(!table.residues.is_empty(), "No residues parsed");

        // Should have at least 20 standard amino acids
        assert!(table.residues.len() >= 20, "Expected at least 20 residues, got {}", table.residues.len());
    }

    #[test]
    fn test_parse_rtf_ic_table_has_met() {
        let rtf_path = "/home/marielle/.local/share/mamba/envs/espaloma-bench/dat/chamber/top_all36_prot.rtf";
        let table = parse_rtf_ic_table(rtf_path).expect("Parse RTF");

        let met = table.residues.iter().find(|r| r.name == "MET");
        assert!(met.is_some(), "MET residue not found");

        let met_res = met.unwrap();
        assert!(!met_res.ic.is_empty(), "MET has no IC records");
    }

    #[test]
    fn test_met_ce_bond() {
        // Verify MET CE (carbon-sulfur) bond parameters from the spec
        let rtf_path = "/home/marielle/.local/share/mamba/envs/espaloma-bench/dat/chamber/top_all36_prot.rtf";
        let table = parse_rtf_ic_table(rtf_path).expect("Parse RTF");

        let met = table.residues.iter().find(|r| r.name == "MET")
            .expect("MET residue not found");

        // Find IC record: CB CG SD CE
        // Expected: b_ij=1.5460, theta_ijk=110.2800, phi=180.0, theta_jkl=98.94, b_kl=1.8206
        let ce_record = met.ic.iter()
            .find(|ic| ic.atom_i == "CB" && ic.atom_j == "CG" && ic.atom_k == "SD" && ic.atom_l == "CE");

        assert!(ce_record.is_some(), "CB-CG-SD-CE IC record not found");

        let ic = ce_record.unwrap();
        assert!((ic.b_kl - 1.8206).abs() < 0.001, "CE bond length: expected ~1.8206 Å, got {}", ic.b_kl);
        assert!((ic.theta_jkl - 98.94).abs() < 0.1, "CE angle: expected ~98.94°, got {}", ic.theta_jkl);
    }

    #[test]
    fn test_all_20_standard_aa_present() {
        let rtf_path = "/home/marielle/.local/share/mamba/envs/espaloma-bench/dat/chamber/top_all36_prot.rtf";
        let table = parse_rtf_ic_table(rtf_path).expect("Parse RTF");

        let standard_aa = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HSD", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        ];

        for aa in &standard_aa {
            let found = table.residues.iter().any(|r| r.name == *aa);
            assert!(found, "Standard amino acid {} not found in parsed RTF", aa);
        }
    }
}
