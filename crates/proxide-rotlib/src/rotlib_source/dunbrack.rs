//! Dunbrack BBDEP2010 rotamer library source.
//!
//! Reads backbone-dependent rotamer data from a Dunbrack text file (*.bbdep.rotamers.lib format).

use super::{BinData, RotamerEntry, RotlibSource};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tracing::debug;

/// Internal representation of a single rotamer entry during parsing.
#[derive(Debug, Clone)]
struct DunbrackRotamer {
    res_code: String,
    phi: f64,
    psi: f64,
    count: u32,
    probability: f64,
    chi_values: [f32; 4],
    chi_sigmas: [f32; 4],
}

/// Dunbrack BBDEP2010 rotamer library source.
///
/// Implements the RotlibSource trait by reading a Dunbrack BBDEP text file
/// and organizing rotamer data by residue code and (phi, psi) bin.
pub struct DunbrackSource {
    /// Map from residue code to list of bins (phi, psi) with rotamer populations.
    data: HashMap<String, Vec<BinData>>,
    /// Map from residue code to index of the default bin (highest frequency).
    default_bin_indices: HashMap<String, usize>,
}

impl DunbrackSource {
    /// Load a Dunbrack rotamer library from a text file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Dunbrack *.bbdep.rotamers.lib file.
    ///
    /// # Returns
    ///
    /// A new DunbrackSource instance, or an error if the file cannot be read or parsed.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // Read and parse the raw rotamer entries
        let rotamers = read_rotamer_library(path)?;

        // Group by residue code and (phi, psi) bin
        let grouped = group_rotamers(rotamers);

        // Transform grouped data into BinData vectors, and track default bin indices
        let mut data = HashMap::new();
        let mut default_bin_indices = HashMap::new();

        for (res_code, bins) in grouped.iter() {
            let mut bin_data_vec = Vec::new();
            let mut bin_counts: Vec<((i32, i32), u64)> = bins
                .iter()
                .map(|(&key, rots)| (key, rots.iter().map(|r| r.count as u64).sum()))
                .collect();
            bin_counts.sort_by(|a, b| b.1.cmp(&a.1)); // descending
            let best_bin_key = bin_counts.first().map(|&(k, _)| k).unwrap_or((0, 0));

            for (&(phi_bin, psi_bin), rots) in bins.iter() {
                let mut rotamers = Vec::new();
                for rot in rots {
                    rotamers.push(RotamerEntry {
                        chi_values: rot.chi_values[..].to_vec(),
                        chi_sigmas: rot.chi_sigmas[..].to_vec(),
                        probability: rot.probability,
                        count: rot.count,
                    });
                }

                // Sort rotamers by probability (descending)
                rotamers.sort_by(|a, b| b.probability.total_cmp(&a.probability));

                bin_data_vec.push(BinData {
                    phi: phi_bin as f64,
                    psi: psi_bin as f64,
                    freq: rots.iter().map(|r| r.count as f64).sum(),
                    rotamers,
                });
            }

            // Find the index of the default bin (matching best_bin_key)
            let default_idx = bin_data_vec
                .iter()
                .position(|b| {
                    (b.phi - best_bin_key.0 as f64).abs() < 1.0
                        && (b.psi - best_bin_key.1 as f64).abs() < 1.0
                })
                .unwrap_or(0);

            default_bin_indices.insert(res_code.clone(), default_idx);
            data.insert(res_code.clone(), bin_data_vec);
        }

        Ok(DunbrackSource {
            data,
            default_bin_indices,
        })
    }
}

impl RotlibSource for DunbrackSource {
    fn residue_codes(&self) -> Vec<String> {
        let mut codes: Vec<String> = self.data.keys().cloned().collect();
        codes.sort();
        codes
    }

    fn bins(&self, code: &str) -> Vec<BinData> {
        self.data.get(code).cloned().unwrap_or_default()
    }

    fn default_bin_index(&self, code: &str) -> usize {
        self.default_bin_indices.get(code).copied().unwrap_or(0)
    }

    fn data_license(&self) -> &str {
        "ODC-BY-1.0"
    }

    fn attribution(&self) -> &str {
        "Dunbrack BBDEP2010 SimpleOpt1-5; Shapovalov & Dunbrack 2011"
    }

    fn source_tag(&self) -> &str {
        "dunbrack2010_simpleopt1"
    }
}

/// Read the Dunbrack rotamer library text file.
fn read_rotamer_library(path: &Path) -> Result<Vec<DunbrackRotamer>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut rotamers = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;

        // Skip comment lines
        if line.starts_with('#') {
            continue;
        }

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Parse the line
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 17 {
            debug!("Line {} has too few fields, skipping", line_num + 1);
            continue;
        }

        let res_code = parts[0].to_string();
        let phi: f64 = parts[1].parse()?;
        let psi: f64 = parts[2].parse()?;
        let count: u32 = parts[3].parse().unwrap_or(0);
        // parts[4-7]: r1..r4 (skip)
        let probability: f64 = parts[8].parse()?;
        let chi1: f32 = parts[9].parse()?;
        let chi2: f32 = parts[10].parse()?;
        let chi3: f32 = parts[11].parse()?;
        let chi4: f32 = parts[12].parse()?;
        let chi1_sig: f32 = parts[13].parse()?;
        let chi2_sig: f32 = parts[14].parse()?;
        let chi3_sig: f32 = parts[15].parse()?;
        let chi4_sig: f32 = parts[16].parse()?;

        rotamers.push(DunbrackRotamer {
            res_code,
            phi,
            psi,
            count,
            probability,
            chi_values: [chi1, chi2, chi3, chi4],
            chi_sigmas: [chi1_sig, chi2_sig, chi3_sig, chi4_sig],
        });
    }

    Ok(rotamers)
}

/// Group rotamers by (residue_code, phi, psi).
type GroupedRotamers = BTreeMap<String, BTreeMap<(i32, i32), Vec<DunbrackRotamer>>>;

fn group_rotamers(rotamers: Vec<DunbrackRotamer>) -> GroupedRotamers {
    let mut grouped: GroupedRotamers = BTreeMap::new();

    for rot in rotamers {
        // Discretize phi/psi to nearest degree for grouping
        let phi_bin = (rot.phi.round()) as i32;
        let psi_bin = (rot.psi.round()) as i32;

        grouped
            .entry(rot.res_code.clone())
            .or_default()
            .entry((phi_bin, psi_bin))
            .or_default()
            .push(rot);
    }

    grouped
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper to create a minimal synthetic Dunbrack file for testing.
    fn create_test_dunbrack_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");

        // Write minimal Dunbrack format header and entries.
        // Format: res_code phi psi count r1 r2 r3 r4 prob chi1 chi2 chi3 chi4 chi1_sig chi2_sig chi3_sig chi4_sig
        let content = "# Test Dunbrack file\nALA -60.0 -40.0 100 1.0 1.0 1.0 1.0 0.5 -62.3 -41.0 0.0 0.0 20.0 25.0 0.0 0.0\n\
                      ALA -60.0 -40.0 80 1.0 1.0 1.0 1.0 0.3 62.3 41.0 0.0 0.0 20.0 25.0 0.0 0.0\n\
                      ALA -120.0 -120.0 50 1.0 1.0 1.0 1.0 0.2 -173.0 67.0 0.0 0.0 20.0 25.0 0.0 0.0\n\
                      GLY 0.0 0.0 200 0.0 0.0 0.0 0.0 0.9 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n";

        file.write_all(content.as_bytes())
            .expect("Failed to write to temp file");
        file.flush().expect("Failed to flush temp file");
        file
    }

    #[test]
    fn test_dunbrack_source_from_file() {
        let file = create_test_dunbrack_file();
        let source = DunbrackSource::from_file(file.path()).expect("Failed to load DunbrackSource");

        let codes = source.residue_codes();
        assert!(!codes.is_empty(), "Expected non-empty residue codes");
        assert!(
            codes.contains(&"ALA".to_string()),
            "Expected ALA in residue codes"
        );
    }

    #[test]
    fn test_residue_codes_sorted() {
        let file = create_test_dunbrack_file();
        let source = DunbrackSource::from_file(file.path()).expect("Failed to load DunbrackSource");

        let codes = source.residue_codes();
        // Verify sorted order
        let mut sorted_codes = codes.clone();
        sorted_codes.sort();
        assert_eq!(
            codes, sorted_codes,
            "residue_codes() should return sorted codes"
        );
    }

    #[test]
    fn test_bins_for_known_residue() {
        let file = create_test_dunbrack_file();
        let source = DunbrackSource::from_file(file.path()).expect("Failed to load DunbrackSource");

        let ala_bins = source.bins("ALA");
        assert!(!ala_bins.is_empty(), "Expected bins for ALA");

        // ALA should have at least 2 distinct (phi, psi) bins based on our test data
        assert!(ala_bins.len() >= 2, "Expected multiple bins for ALA");
    }

    #[test]
    fn test_bins_for_unknown_residue() {
        let file = create_test_dunbrack_file();
        let source = DunbrackSource::from_file(file.path()).expect("Failed to load DunbrackSource");

        let unknown_bins = source.bins("UNKNOWN");
        assert!(
            unknown_bins.is_empty(),
            "Expected empty vec for unknown residue"
        );
    }

    #[test]
    fn test_default_bin_index_valid() {
        let file = create_test_dunbrack_file();
        let source = DunbrackSource::from_file(file.path()).expect("Failed to load DunbrackSource");

        let idx = source.default_bin_index("ALA");
        let ala_bins = source.bins("ALA");
        assert!(
            idx < ala_bins.len(),
            "default_bin_index should return valid index"
        );
    }

    #[test]
    fn test_default_bin_index_unknown() {
        let file = create_test_dunbrack_file();
        let source = DunbrackSource::from_file(file.path()).expect("Failed to load DunbrackSource");

        let idx = source.default_bin_index("UNKNOWN");
        assert_eq!(idx, 0, "Expected default index 0 for unknown residue");
    }

    #[test]
    fn test_data_license() {
        let file = create_test_dunbrack_file();
        let source = DunbrackSource::from_file(file.path()).expect("Failed to load DunbrackSource");

        assert_eq!(source.data_license(), "ODC-BY-1.0");
    }

    #[test]
    fn test_attribution_contains_required_text() {
        let file = create_test_dunbrack_file();
        let source = DunbrackSource::from_file(file.path()).expect("Failed to load DunbrackSource");

        let attr = source.attribution();
        assert!(!attr.is_empty(), "attribution should not be empty");
        assert!(
            attr.contains("Shapovalov"),
            "attribution should mention Shapovalov"
        );
        assert!(
            attr.contains("Dunbrack"),
            "attribution should mention Dunbrack"
        );
    }

    #[test]
    fn test_source_tag() {
        let file = create_test_dunbrack_file();
        let source = DunbrackSource::from_file(file.path()).expect("Failed to load DunbrackSource");

        assert_eq!(source.source_tag(), "dunbrack2010_simpleopt1");
    }

    #[test]
    fn test_bin_data_structure() {
        let file = create_test_dunbrack_file();
        let source = DunbrackSource::from_file(file.path()).expect("Failed to load DunbrackSource");

        let ala_bins = source.bins("ALA");
        assert!(!ala_bins.is_empty());

        let first_bin = &ala_bins[0];
        assert!(first_bin.rotamers.len() > 0, "Expected rotamers in bin");

        // Verify each rotamer has valid structure
        for rot in &first_bin.rotamers {
            assert!(!rot.chi_values.is_empty(), "Rotamer should have chi_values");
            assert!(!rot.chi_sigmas.is_empty(), "Rotamer should have chi_sigmas");
            assert!(
                rot.probability >= 0.0 && rot.probability <= 1.0,
                "Probability should be 0-1"
            );
            assert!(rot.count > 0, "Count should be positive");
        }
    }
}
