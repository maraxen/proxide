//! Dunbrack BBDEP2010 rotamer library source.
//!
//! Reads backbone-dependent rotamer data from a Dunbrack text file (*.bbdep.rotamers.lib format).

use super::{RotlibSource, RotamerEntry, BinData};
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
                .position(|b| (b.phi - best_bin_key.0 as f64).abs() < 1.0 && (b.psi - best_bin_key.1 as f64).abs() < 1.0)
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
        self.data
            .get(code)
            .map(|bins| bins.clone())
            .unwrap_or_default()
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
            .or_insert_with(BTreeMap::new)
            .entry((phi_bin, psi_bin))
            .or_insert_with(Vec::new)
            .push(rot);
    }

    grouped
}
