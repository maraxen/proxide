//! Parse PDB CCD (Chemical Component Dictionary) `.cif` files into residue IC geometry.
//!
//! Extracts `_chem_comp_bond.value_dist_ideal` and `_chem_comp_angle.value_angle_ideal`
//! from CIF files and converts them into IC records following the ResidueTemplate atom
//! ordering and connectivity.
//!
//! The CIF format is a column-based text format used by the Protein Data Bank:
//! - `loop_` introduces a multi-row block
//! - Field names prefixed with `_` define the columns
//! - Data rows follow in space-separated or quoted-string format

use super::template::ResidueTemplate;
use crate::RotlibError;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::{debug, warn};

/// Fallback parameter kind for stereochemical fallbacks (Debt #1933).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FallbackKind {
    BondLength,
    ValenceAngle,
}

/// Provenance record for non-silent stereochemical fallbacks (Debt #1933).
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryFallback {
    pub component: String,
    pub atom_names: Vec<String>,
    pub parameter_kind: FallbackKind,
    pub fallback_value: f64,
    pub citation: &'static str,
}

impl From<&GeometryFallback> for crate::pb::proxide::rotlib::v1::GeometryFallback {
    fn from(f: &GeometryFallback) -> Self {
        Self {
            component: f.component.clone(),
            atom_names: f.atom_names.clone(),
            parameter_kind: match f.parameter_kind {
                FallbackKind::BondLength => 0,
                FallbackKind::ValenceAngle => 1,
            },
            fallback_value: f.fallback_value,
            citation: f.citation.to_string(),
        }
    }
}

/// A simple CIF parser for extracting bond and angle records.
pub struct CifParser {
    lines: Vec<String>,
    pos: usize,
}

impl CifParser {
    pub fn new(content: &str) -> Self {
        let lines: Vec<String> = content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        Self { lines, pos: 0 }
    }

    /// Parse a loop block that starts with field names and is followed by data rows.
    /// Returns a map of field_name -> Vec<values> for all fields in the loop.
    ///
    /// # Errors
    /// Returns `RotlibError::InvalidFormat` if a row contains extra or missing tokens
    /// relative to the declared field definitions (Debt #1933 strict token validation).
    pub fn parse_loop(&mut self) -> Result<HashMap<String, Vec<String>>, RotlibError> {
        let mut fields = Vec::new();
        let mut result = HashMap::new();

        // Read field names (all lines starting with _)
        while self.pos < self.lines.len() {
            let line = self.lines[self.pos].trim();
            if line.starts_with('_') {
                fields.push(line.to_string());
                self.pos += 1;
            } else {
                break;
            }
        }

        if fields.is_empty() {
            return Err(RotlibError::InvalidFormat(
                "loop_ encountered without any field definitions".to_string(),
            ));
        }

        // Initialize result map
        for field in &fields {
            result.insert(field.clone(), Vec::new());
        }

        // Read data rows
        let mut current_row = Vec::new();
        while self.pos < self.lines.len() {
            let line = self.lines[self.pos].trim();
            if line.starts_with('_') || line.starts_with("loop_") || line.starts_with("data_") {
                // Next section starts
                break;
            }
            if line.is_empty() || line.starts_with('#') {
                self.pos += 1;
                continue;
            }

            // Parse tokens (quoted strings or space-separated)
            let tokens = self.tokenize(line);
            current_row.extend(tokens);

            if current_row.len() == fields.len() {
                for (i, field) in fields.iter().enumerate() {
                    result.get_mut(field).unwrap().push(current_row[i].clone());
                }
                current_row.clear();
                self.pos += 1;
            } else if current_row.len() > fields.len() {
                return Err(RotlibError::InvalidFormat(format!(
                    "CIF table row has {} tokens, expected exact field count {}",
                    current_row.len(),
                    fields.len()
                )));
            } else {
                // Partial row spanning multiple lines
                self.pos += 1;
            }
        }

        if !current_row.is_empty() {
            return Err(RotlibError::InvalidFormat(format!(
                "CIF table has incomplete row with {} tokens, expected exact field count {}",
                current_row.len(),
                fields.len()
            )));
        }

        Ok(result)
    }

    /// Tokenize a CIF data line, handling quoted strings and spaces.
    pub fn tokenize(&self, line: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        let mut chars = line.chars().peekable();

        for c in chars.by_ref() {
            match c {
                '\'' | '"' => {
                    in_quotes = !in_quotes;
                }
                ' ' | '\t' if !in_quotes => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                }
                _ => {
                    current.push(c);
                }
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }
}

/// Bond map: (atom1, atom2) -> ideal distance
type BondMap = HashMap<(String, String), f32>;

/// Angle map: (atom1, center, atom3) -> ideal angle
type AngleMap = HashMap<(String, String, String), f32>;

/// Parse a CIF file and extract bond and angle ideal values.
pub fn parse_cif_file(path: &Path) -> Result<(BondMap, AngleMap), RotlibError> {
    let content = fs::read_to_string(path).map_err(|e| {
        RotlibError::InvalidFormat(format!("Failed to read CIF file {}: {}", path.display(), e))
    })?;

    let mut bonds = HashMap::new();
    let mut angles = HashMap::new();
    let mut parser = CifParser::new(&content);

    while parser.pos < parser.lines.len() {
        let line = &parser.lines[parser.pos];

        if line.starts_with("loop_") {
            parser.pos += 1;
            let block = parser.parse_loop()?;

            // Check if this is a _chem_comp_bond block
            if let (Some(atom_id_1), Some(atom_id_2), Some(dist_ideal)) = (
                block.get("_chem_comp_bond.atom_id_1"),
                block.get("_chem_comp_bond.atom_id_2"),
                block.get("_chem_comp_bond.value_dist_ideal"),
            ) {
                for i in 0..atom_id_1.len() {
                    let a1 = &atom_id_1[i];
                    let a2 = &atom_id_2[i];
                    let val_str = &dist_ideal[i];
                    // CIF null/unmeasured handling (Debt #1933)
                    if val_str == "?" || val_str == "." {
                        continue;
                    }
                    let d: f32 = val_str.parse().map_err(|_| {
                        RotlibError::InvalidFormat(format!(
                            "Invalid bond distance: {}",
                            val_str
                        ))
                    })?;
                    let key = if a1 <= a2 {
                        (a1.clone(), a2.clone())
                    } else {
                        (a2.clone(), a1.clone())
                    };
                    bonds.insert(key, d);
                }
            }

            // Check if this is a _chem_comp_angle block
            if let (Some(atom_id_1), Some(atom_id_2), Some(atom_id_3), Some(angle_ideal)) = (
                block.get("_chem_comp_angle.atom_id_1"),
                block.get("_chem_comp_angle.atom_id_2"),
                block.get("_chem_comp_angle.atom_id_3"),
                block.get("_chem_comp_angle.value_angle_ideal"),
            ) {
                for i in 0..atom_id_1.len() {
                    let a1 = &atom_id_1[i];
                    let a2 = &atom_id_2[i];
                    let a3 = &atom_id_3[i];
                    let val_str = &angle_ideal[i];
                    // CIF null/unmeasured handling (Debt #1933)
                    if val_str == "?" || val_str == "." {
                        continue;
                    }
                    let ang: f32 = val_str.parse().map_err(|_| {
                        RotlibError::InvalidFormat(format!("Invalid angle: {}", val_str))
                    })?;
                    angles.insert((a1.clone(), a2.clone(), a3.clone()), ang);
                }
            }
        } else {
            parser.pos += 1;
        }
    }

    Ok((bonds, angles))
}

/// Parse all CCD CIF files in a directory and return a ResidueGeometryTable (proto).
///
/// Wraps [`parse_ccd_ic_table_raw`] and converts local IcRecord and GeometryFallback to proto.
pub fn parse_ccd_ic_table(
    ccd_dir: &str,
) -> Result<crate::pb::proxide::rotlib::v1::ResidueGeometryTable, RotlibError> {
    use crate::pb::proxide::rotlib::v1::{
        IcRecord as ProtoIcRecord, ResidueGeometry, ResidueGeometryTable,
    };
    let (raw, fallbacks) = parse_ccd_ic_table_raw(ccd_dir)?;
    let residues = raw
        .into_iter()
        .map(|(name, records)| ResidueGeometry {
            name,
            ic: records
                .into_iter()
                .map(|r| ProtoIcRecord {
                    atom_i: r.atom_i,
                    atom_j: r.atom_j,
                    atom_k: r.atom_k,
                    atom_l: r.atom_l,
                    branch: r.branch,
                    b_ij: r.b_ij,
                    theta_ijk: r.theta_ijk,
                    phi_ijkl: r.phi_ijkl,
                    theta_jkl: r.theta_jkl,
                    b_kl: r.b_kl,
                })
                .collect(),
        })
        .collect();
    let proto_fallbacks = fallbacks.iter().map(Into::into).collect();
    Ok(ResidueGeometryTable {
        source: "pdb_ccd".to_string(),
        version: "pdb_ccd_2024".to_string(),
        license: "CC0".to_string(),
        citation: "doi:10.1093/nar/gku1178".to_string(),
        residues,
        fallbacks: proto_fallbacks,
    })
}

/// Parsed raw CCD IC records and recorded fallbacks.
pub type ParsedCcdRecords = (Vec<(String, Vec<IcRecord>)>, Vec<GeometryFallback>);

/// Parse all CCD CIF files in a directory and return raw IC records and recorded fallbacks.
pub fn parse_ccd_ic_table_raw(
    ccd_dir: &str,
) -> Result<ParsedCcdRecords, RotlibError> {
    let path = Path::new(ccd_dir);
    if !path.is_dir() {
        return Err(RotlibError::InvalidFormat(format!(
            "{} is not a directory",
            ccd_dir
        )));
    }

    let mut result = Vec::new();
    let mut all_fallbacks = Vec::new();

    // Standard amino acids (3-letter codes)
    let standard_aas = vec![
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
        "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ];

    for aa_code in standard_aas {
        let cif_path = path.join(format!("{}.cif", aa_code));
        if !cif_path.exists() {
            warn!("CCD file not found for {}: {}", aa_code, cif_path.display());
            continue;
        }

        match parse_cif_file(&cif_path) {
            Ok((bonds, angles)) => {
                // Build IC records from the bonds and angles
                let ic_records = build_ic_records_for_residue(aa_code, &bonds, &angles, &mut all_fallbacks)?;
                result.push((aa_code.to_string(), ic_records));
                debug!(
                    "Parsed CCD for {}: {} bonds, {} angles, {} IC records",
                    aa_code,
                    bonds.len(),
                    angles.len(),
                    result.last().unwrap().1.len()
                );
            }
            Err(e) => {
                warn!("Failed to parse CCD for {}: {}", aa_code, e);
            }
        }
    }

    Ok((result, all_fallbacks))
}

/// Represents one IC record (NERF placement: i, j, k, l with geometry).
#[derive(Clone, Debug)]
pub struct IcRecord {
    pub atom_i: String,
    pub atom_j: String,
    pub atom_k: String,
    pub atom_l: String,
    pub branch: bool,
    pub b_ij: f32,
    pub theta_ijk: f32,
    pub phi_ijkl: f32,
    pub theta_jkl: f32,
    pub b_kl: f32,
}

/// Build IC records for a residue by walking its template atom tree.
fn build_ic_records_for_residue(
    aa_code: &str,
    bonds: &HashMap<(String, String), f32>,
    angles: &HashMap<(String, String, String), f32>,
    fallbacks: &mut Vec<GeometryFallback>,
) -> Result<Vec<IcRecord>, RotlibError> {
    use super::template::standard_residue_template;

    // Get the template for this residue
    let template = standard_residue_template(aa_code)
        .ok_or_else(|| RotlibError::InvalidFormat(format!("No template for {}", aa_code)))?;

    let mut records = Vec::new();

    // Walk atoms in template order; starting from atom 3 (O), build IC records
    for atom_idx in 3..template.num_atoms() {
        if let Some(bond_def) = template.bonds[atom_idx] {
            let atom_k = &template.atom_names[atom_idx];
            let atom_j = &template.atom_names[bond_def.parent_idx];

            // Resolve grandparent (B) and great-grandparent (A)
            let (b_idx, a_idx) = resolve_atom_parents(&template, bond_def.parent_idx);
            let atom_i = &template.atom_names[a_idx];
            let atom_j_check = &template.atom_names[b_idx];

            // Find a child of atom_k for the L atom (or use a sibling)
            let (atom_l, b_kl, _theta_jkl) = find_atom_l(&template, atom_idx, bonds, angles, aa_code, fallbacks);

            // Look up b(i-j), theta(i-j-k), theta(j-k-l), b(k-l) from CCD
            let b_ij = lookup_bond(aa_code, bonds, atom_i, atom_j_check, fallbacks);
            let theta_ijk = lookup_angle(aa_code, angles, atom_i, atom_j, atom_k, fallbacks);
            let phi_ijkl = 0.0; // CCD does not provide dihedral; chi placement from Dunbrack
            let theta_jkl = if !atom_l.is_empty() {
                lookup_angle(aa_code, angles, atom_j, atom_k, &atom_l, fallbacks)
            } else {
                0.0
            };

            let record = IcRecord {
                atom_i: atom_i.clone(),
                atom_j: atom_j.clone(),
                atom_k: atom_k.clone(),
                atom_l,
                branch: false, // CCD doesn't explicitly mark branches
                b_ij,
                theta_ijk,
                phi_ijkl,
                theta_jkl,
                b_kl,
            };

            records.push(record);
        }
    }

    Ok(records)
}

/// Resolve the grandparent (B) and great-grandparent (A) indices for a parent atom.
fn resolve_atom_parents(template: &ResidueTemplate, parent_idx: usize) -> (usize, usize) {
    match parent_idx {
        1 => (0, 2), // parent=CA: B=N, A=backbone_C
        2 => (1, 0), // parent=backbone_C: B=CA, A=N
        _ => {
            // Parent is a sidechain atom — walk up via template.bonds
            if let Some(bond) = template.bonds[parent_idx] {
                let b_idx = bond.parent_idx;
                let a_idx = match b_idx {
                    0 => 2, // B=N: A=backbone_C
                    1 => 0, // B=CA: A=N
                    2 => 1, // B=backbone_C: A=CA
                    _ => template.bonds[b_idx].map(|bb| bb.parent_idx).unwrap_or(0),
                };
                (b_idx, a_idx)
            } else {
                (0, 1)
            }
        }
    }
}

/// Find the child of atom_k (L atom); return (atom_l_name, b_kl, theta_jkl).
fn find_atom_l(
    template: &ResidueTemplate,
    atom_k_idx: usize,
    bonds: &HashMap<(String, String), f32>,
    _angles: &HashMap<(String, String, String), f32>,
    component: &str,
    fallbacks: &mut Vec<GeometryFallback>,
) -> (String, f32, f32) {
    // Look for the first child of atom_k in the template
    for (idx, bond_def) in template.bonds.iter().enumerate() {
        if let Some(bd) = bond_def {
            if bd.parent_idx == atom_k_idx {
                let atom_l = template.atom_names[idx].clone();
                let atom_k = &template.atom_names[atom_k_idx];
                let b_kl = lookup_bond(component, bonds, atom_k, &atom_l, fallbacks);
                return (atom_l, b_kl, 0.0);
            }
        }
    }

    // No child found
    ("".to_string(), 0.0, 0.0)
}

/// Look up bond length from CCD data. Records fallback and emits tracing::warn! if not found (Debt #1933).
fn lookup_bond(
    component: &str,
    bonds: &HashMap<(String, String), f32>,
    atom_a: &str,
    atom_b: &str,
    fallbacks: &mut Vec<GeometryFallback>,
) -> f32 {
    let key = if atom_a <= atom_b {
        (atom_a.to_string(), atom_b.to_string())
    } else {
        (atom_b.to_string(), atom_a.to_string())
    };
    if let Some(&d) = bonds.get(&key) {
        d
    } else {
        let fallback_val = 1.50;
        let citation = "Engh & Huber 1991";
        tracing::warn!(
            component = %component,
            atom_a = %atom_a,
            atom_b = %atom_b,
            fallback = fallback_val,
            citation = citation,
            "CCD missing ideal bond length; recording Engh & Huber 1991 fallback"
        );
        fallbacks.push(GeometryFallback {
            component: component.to_string(),
            atom_names: vec![atom_a.to_string(), atom_b.to_string()],
            parameter_kind: FallbackKind::BondLength,
            fallback_value: fallback_val,
            citation,
        });
        fallback_val as f32
    }
}

/// Look up bond angle from CCD data. Records fallback and emits tracing::warn! if not found (Debt #1933).
fn lookup_angle(
    component: &str,
    angles: &HashMap<(String, String, String), f32>,
    a1: &str,
    center: &str,
    a3: &str,
    fallbacks: &mut Vec<GeometryFallback>,
) -> f32 {
    if let Some(&ang) = angles.get(&(a1.to_string(), center.to_string(), a3.to_string())) {
        return ang;
    }
    if let Some(&ang) = angles.get(&(a3.to_string(), center.to_string(), a1.to_string())) {
        return ang;
    }
    let fallback_val = 109.47;
    let citation = "Ideal tetrahedral sp3 valence angle arccos(-1/3)";
    tracing::warn!(
        component = %component,
        atom_1 = %a1,
        center = %center,
        atom_3 = %a3,
        fallback = fallback_val,
        citation = citation,
        "CCD missing ideal valence angle; recording ideal tetrahedral fallback"
    );
    fallbacks.push(GeometryFallback {
        component: component.to_string(),
        atom_names: vec![a1.to_string(), center.to_string(), a3.to_string()],
        parameter_kind: FallbackKind::ValenceAngle,
        fallback_value: fallback_val,
        citation,
    });
    fallback_val as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ccd_ic_table() {
        // Try relative paths from different locations
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let ccd_paths = vec![
            format!("{}/tests/data/ccd", manifest_dir),
            "crates/proxide-rotlib/tests/data/ccd".to_string(),
            "tests/data/ccd".to_string(),
            "./tests/data/ccd".to_string(),
        ];

        let mut last_err = None;
        let mut result = None;
        for p in &ccd_paths {
            match parse_ccd_ic_table(p) {
                Ok(t) => {
                    result = Some(t);
                    break;
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }
        if result.is_none() {
            panic!("Should parse test CCD data directory, last error: {:?}", last_err);
        }
    }

    #[test]
    fn test_cif_tokenizer() {
        let parser = CifParser::new("PRO N CA SING");
        let tokens = parser.tokenize("PRO N CA SING");
        assert_eq!(tokens, vec!["PRO", "N", "CA", "SING"]);
    }

    #[test]
    fn test_cif_quoted_tokens() {
        let parser = CifParser::new("");
        let tokens = parser.tokenize("'quoted string' unquoted 'another'");
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_parse_loop_strict_token_count_extra_token_errors() {
        let cif = "\
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_dist_ideal
N CA 1.458 EXTRA_TOKEN
";
        let mut parser = CifParser::new(cif);
        parser.pos = 1; // skip loop_ line
        let res = parser.parse_loop();
        assert!(res.is_err(), "Extra token must trigger RotlibError::InvalidFormat");
        if let Err(RotlibError::InvalidFormat(msg)) = res {
            assert!(msg.contains("CIF table row has 4 tokens, expected exact field count 3"));
        } else {
            panic!("Expected InvalidFormat error");
        }
    }

    #[test]
    fn test_parse_loop_strict_token_count_missing_token_errors() {
        let cif = "\
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_dist_ideal
N CA
";
        let mut parser = CifParser::new(cif);
        parser.pos = 1; // skip loop_ line
        let res = parser.parse_loop();
        assert!(res.is_err(), "Incomplete row must trigger RotlibError::InvalidFormat");
        if let Err(RotlibError::InvalidFormat(msg)) = res {
            assert!(msg.contains("incomplete row with 2 tokens"));
        } else {
            panic!("Expected InvalidFormat error");
        }
    }

    #[test]
    fn test_cif_null_tokens_treated_as_absent() {
        let dir = tempfile::tempdir().unwrap();
        let cif_content = "\
data_ALA
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_dist_ideal
N CA ?
CA C .
CA CB 1.540
#
loop_
_chem_comp_angle.atom_id_1
_chem_comp_angle.atom_id_2
_chem_comp_angle.atom_id_3
_chem_comp_angle.value_angle_ideal
N CA C ?
N CA CB 110.5
";
        let cif_file = dir.path().join("ALA.cif");
        fs::write(&cif_file, cif_content).unwrap();

        let parsed = parse_cif_file(&cif_file);
        assert!(parsed.is_ok(), "CIF with '?' and '.' null tokens must parse successfully");
        let (bonds, angles) = parsed.unwrap();
        // CA-CB should be present
        assert_eq!(bonds.get(&("CA".to_string(), "CB".to_string())), Some(&1.540));
        // N-CA was '?', so absent
        assert!(!bonds.contains_key(&("CA".to_string(), "N".to_string())));
        // N-CA-CB was 110.5, so present
        assert_eq!(angles.get(&("N".to_string(), "CA".to_string(), "CB".to_string())), Some(&110.5));
    }

    #[test]
    fn test_fallback_recording_and_provenance() {
        let dir = tempfile::tempdir().unwrap();
        // Minimal ALA where CB bond and angle are omitted
        let cif_content = "\
data_ALA
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_dist_ideal
N CA 1.458
#
loop_
_chem_comp_angle.atom_id_1
_chem_comp_angle.atom_id_2
_chem_comp_angle.atom_id_3
_chem_comp_angle.value_angle_ideal
N CA C 111.0
";
        let cif_file = dir.path().join("ALA.cif");
        fs::write(&cif_file, cif_content).unwrap();

        let (records, fallbacks) = parse_ccd_ic_table_raw(dir.path().to_str().unwrap()).unwrap();
        assert!(!records.is_empty());
        assert!(!fallbacks.is_empty(), "Missing bond/angle in CCD must record fallbacks");

        // Verify fallback citations and kinds
        let bond_fb = fallbacks.iter().find(|f| f.parameter_kind == FallbackKind::BondLength);
        assert!(bond_fb.is_some());
        let b = bond_fb.unwrap();
        assert_eq!(b.citation, "Engh & Huber 1991");
        assert_eq!(b.fallback_value, 1.50);

        let angle_fb = fallbacks.iter().find(|f| f.parameter_kind == FallbackKind::ValenceAngle);
        assert!(angle_fb.is_some());
        let a = angle_fb.unwrap();
        assert_eq!(a.citation, "Ideal tetrahedral sp3 valence angle arccos(-1/3)");
        assert_eq!(a.fallback_value, 109.47);
    }
}
