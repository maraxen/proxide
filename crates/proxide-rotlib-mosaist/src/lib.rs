//! Mosaist binary rotlib.bin format reader and Protobuf converter.
//!
//! Licensed under CC BY-NC-SA 4.0 (Non-Commercial Research Use Only).
//! See NOTICE for full licensing details.

use std::collections::{HashMap, HashSet};
use std::io::{BufReader, Read};
use std::path::Path;
use proxide_rotlib::{AaGridSummary, RotlibError};

fn angle_to_standard(a: f64) -> f64 {
    let mut v = a.rem_euclid(360.0);
    if v >= 180.0 {
        v -= 360.0;
    }
    v
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).abs() % 360.0;
    if d > 180.0 {
        360.0 - d
    } else {
        d
    }
}

fn find_closest_angle(centers: &[f64], query: f64) -> usize {
    centers
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| angle_diff(a, query).total_cmp(&angle_diff(b, query)))
        .unwrap()
        .0
}

fn read_i32(r: &mut impl Read) -> std::io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_cstring(r: &mut impl Read) -> std::io::Result<String> {
    let mut bytes = Vec::new();
    loop {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        if b[0] == 0 {
            break;
        }
        bytes.push(b[0]);
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Validate rectangular grid and assert strict phi-major ordering.
///
/// For an N_phi x N_psi grid, the i-th bin must match
/// (phi_centers[i / n_psi], psi_centers[i % n_psi]) within 1e-4 tolerance.
pub fn validate_grid(bins: &[(f64, f64, f64)]) -> Result<(Vec<f64>, Vec<f64>), RotlibError> {
    let mut pair_set: HashSet<(u64, u64)> = HashSet::new();
    let mut phi_set: HashSet<u64> = HashSet::new();
    let mut psi_set: HashSet<u64> = HashSet::new();
    for &(phi, psi, _) in bins {
        let pk = (phi.to_bits(), psi.to_bits());
        if !pair_set.insert(pk) {
            return Err(RotlibError::InvalidFormat(format!(
                "duplicate (phi, psi) pair ({phi:.2}, {psi:.2})"
            )));
        }
        phi_set.insert(phi.to_bits());
        psi_set.insert(psi.to_bits());
    }
    let n_phi = phi_set.len();
    let n_psi = psi_set.len();
    if n_phi * n_psi != bins.len() {
        return Err(RotlibError::InvalidFormat(format!(
            "non-rectangular grid: {}φ × {}ψ = {} ≠ {} bins",
            n_phi,
            n_psi,
            n_phi * n_psi,
            bins.len()
        )));
    }
    let mut phi_centers: Vec<f64> = phi_set.iter().map(|&b| f64::from_bits(b)).collect();
    let mut psi_centers: Vec<f64> = psi_set.iter().map(|&b| f64::from_bits(b)).collect();
    phi_centers.sort_by(f64::total_cmp);
    psi_centers.sort_by(f64::total_cmp);

    // Phi-major sorting check per spec §4.2:
    // Assert the i-th bin matches (phi_centers[i / n_psi], psi_centers[i % n_psi])
    for (i, &(phi, psi, _)) in bins.iter().enumerate() {
        let expected_phi = phi_centers[i / n_psi];
        let expected_psi = psi_centers[i % n_psi];
        if (phi - expected_phi).abs() > 1e-4 || (psi - expected_psi).abs() > 1e-4 {
            return Err(RotlibError::InvalidFormat(format!(
                "bin {i} ({phi:.2}, {psi:.2}) out of phi-major order, expected ({expected_phi:.2}, {expected_psi:.2})"
            )));
        }
    }

    Ok((phi_centers, psi_centers))
}

/// Per-rotamer data within a single phi/psi bin.
#[derive(Clone, Debug)]
pub struct BinData {
    pub probs: Vec<f64>,
    pub coords: Vec<Vec<[f64; 3]>>,
}

/// Per amino-acid rotamer library entry.
#[derive(Clone, Debug)]
pub struct AaEntry {
    pub atom_names: Vec<String>,
    pub bin_phi_centers: Vec<f64>,
    pub bin_psi_centers: Vec<f64>,
    pub default_bin: u32,
    pub rotamers: Vec<BinData>,
}

const CIS_PRO_KEY: &str = "CPR";

/// Mosaist binary rotamer library loaded from MSL `rotlib.bin`.
#[derive(Debug)]
pub struct MosaistRotamerLibrary {
    pub entries: HashMap<String, AaEntry>,
}

pub type RotamerLibrary = MosaistRotamerLibrary;

impl MosaistRotamerLibrary {
    /// Load from MSL binary file (`rotlib.bin`).
    pub fn load(path: &Path) -> Result<Self, RotlibError> {
        use std::fs::File;
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut entries: HashMap<String, AaEntry> = HashMap::new();

        loop {
            let aa_name = match read_cstring(&mut reader) {
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(RotlibError::Io(e)),
                Ok(s) if s.is_empty() => break,
                Ok(s) => s,
            };

            let nc = read_i32(&mut reader)? as usize;
            let na = read_i32(&mut reader)? as usize;
            let nb = read_i32(&mut reader)? as usize;

            // chi definitions — 4 strings each, discard
            for _ in 0..nc {
                for _ in 0..4 {
                    read_cstring(&mut reader)?;
                }
            }

            // atom names
            let mut atom_names = Vec::with_capacity(na);
            for _ in 0..na {
                atom_names.push(read_cstring(&mut reader)?);
            }

            // bin descriptors
            let mut bin_descriptors: Vec<(f64, f64, f64)> = Vec::with_capacity(nb);
            for _ in 0..nb {
                let phi = angle_to_standard(read_f32(&mut reader)? as f64);
                let psi = angle_to_standard(read_f32(&mut reader)? as f64);
                let freq = read_f32(&mut reader)? as f64;
                bin_descriptors.push((phi, psi, freq));
            }

            let (bin_phi_centers, bin_psi_centers) = validate_grid(&bin_descriptors)?;

            // default_bin: argmax frequency, first-maximum wins
            let default_bin = bin_descriptors
                .iter()
                .enumerate()
                .fold(
                    (0usize, f64::NEG_INFINITY),
                    |(best_i, best_f), (i, &(_, _, f))| {
                        if f > best_f {
                            (i, f)
                        } else {
                            (best_i, best_f)
                        }
                    },
                )
                .0 as u32;

            // rotamer data
            let mut rotamers: Vec<BinData> = Vec::with_capacity(nb);
            for _ in 0..nb {
                let nr = read_i32(&mut reader)? as usize;
                let mut probs = Vec::with_capacity(nr);
                let mut coords: Vec<Vec<[f64; 3]>> = Vec::with_capacity(nr);
                for _ in 0..nr {
                    probs.push(read_f32(&mut reader)? as f64);
                    for _ in 0..nc {
                        read_f32(&mut reader)?; // chi
                        read_f32(&mut reader)?; // sigma
                    }
                    let mut atom_coords = Vec::with_capacity(na);
                    for _ in 0..na {
                        let x = read_f32(&mut reader)? as f64;
                        let y = read_f32(&mut reader)? as f64;
                        let z = read_f32(&mut reader)? as f64;
                        atom_coords.push([x, y, z]);
                    }
                    coords.push(atom_coords);
                }
                rotamers.push(BinData { probs, coords });
            }

            entries.insert(
                aa_name,
                AaEntry {
                    atom_names,
                    bin_phi_centers,
                    bin_psi_centers,
                    default_bin,
                    rotamers,
                },
            );
        }

        Ok(MosaistRotamerLibrary { entries })
    }

    /// Return `true` if the library contains rotamer data for amino acid `aa`.
    pub fn contains_aa(&self, aa: &str) -> bool {
        self.entries.contains_key(aa)
    }

    /// Return all residue codes present in the library.
    pub fn residue_codes(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Return sidechain atom names for residue `aa`.
    pub fn sidechain_atom_names(&self, aa: &str) -> Result<&[String], RotlibError> {
        self.entries
            .get(aa)
            .map(|e| e.atom_names.as_slice())
            .ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))
    }

    /// Aggregate, non-identifying statistics for one amino acid's rotamer grid.
    pub fn aa_grid_summary(&self, aa: &str) -> Result<AaGridSummary, RotlibError> {
        let entry = self
            .entries
            .get(aa)
            .ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))?;

        let bin_count = entry.rotamers.len();
        let mut min_rotamers_per_bin = usize::MAX;
        let mut max_rotamers_per_bin = 0usize;
        let mut min_probability = f64::INFINITY;
        let mut max_probability = f64::NEG_INFINITY;

        for bin in &entry.rotamers {
            let n = bin.probs.len();
            min_rotamers_per_bin = min_rotamers_per_bin.min(n);
            max_rotamers_per_bin = max_rotamers_per_bin.max(n);
            for &p in &bin.probs {
                min_probability = min_probability.min(p);
                max_probability = max_probability.max(p);
            }
        }

        if bin_count == 0 {
            min_rotamers_per_bin = 0;
            min_probability = 0.0;
            max_probability = 0.0;
        }

        let cb_idx = entry.atom_names.iter().position(|n| n == "CB");
        let max_cb_spread_angstrom = cb_idx.map(|idx| {
            let points: Vec<[f64; 3]> = entry
                .rotamers
                .iter()
                .filter_map(|bin| bin.coords.first())
                .filter_map(|rot0_coords| rot0_coords.get(idx).copied())
                .collect();

            let mut max_d = 0.0f64;
            for i in 0..points.len() {
                for j in (i + 1)..points.len() {
                    let d = ((points[i][0] - points[j][0]).powi(2)
                        + (points[i][1] - points[j][1]).powi(2)
                        + (points[i][2] - points[j][2]).powi(2))
                    .sqrt();
                    max_d = max_d.max(d);
                }
            }
            max_d
        });

        Ok(AaGridSummary {
            bin_count,
            min_rotamers_per_bin,
            max_rotamers_per_bin,
            min_probability,
            max_probability,
            max_cb_spread_angstrom,
        })
    }

    /// Resolve entry and bin index for query.
    fn resolve_entry_bin(
        &self,
        aa: &str,
        phi: Option<f64>,
        psi: Option<f64>,
        cis_proline: bool,
    ) -> Result<(&AaEntry, usize), RotlibError> {
        let effective_aa = if cis_proline && aa == "PRO" {
            if self.entries.contains_key(CIS_PRO_KEY) {
                CIS_PRO_KEY
            } else {
                "PRO"
            }
        } else {
            aa
        };

        let entry = self
            .entries
            .get(effective_aa)
            .ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))?;

        let bin = match (phi, psi) {
            (Some(p), Some(s)) => {
                let phi_ind = find_closest_angle(&entry.bin_phi_centers, p);
                let psi_ind = find_closest_angle(&entry.bin_psi_centers, s);
                phi_ind * entry.bin_psi_centers.len() + psi_ind
            }
            _ => entry.default_bin as usize,
        };

        Ok((entry, bin))
    }

    /// Number of rotamers in the bin closest to `(phi, psi)` for amino acid `aa`.
    pub fn num_rotamers(
        &self,
        aa: &str,
        phi: Option<f64>,
        psi: Option<f64>,
        cis_proline: bool,
    ) -> Result<usize, RotlibError> {
        let (entry, bin) = self.resolve_entry_bin(aa, phi, psi, cis_proline)?;
        Ok(entry.rotamers[bin].probs.len())
    }

    /// Convert Mosaist library to protobuf format for parse_master.
    pub fn to_protobuf(
        &self,
        geometry_source: String,
        geometry_license: String,
    ) -> proxide_rotlib::rotlib_v1::RotamerLibrary {
        let mut residues = Vec::new();

        for (code, entry) in &self.entries {
            let phi_centers = entry.bin_phi_centers.clone();
            let psi_centers = entry.bin_psi_centers.clone();
            let mut bins = Vec::new();

            for (bin_idx, bin_data) in entry.rotamers.iter().enumerate() {
                let n_psi = psi_centers.len();
                let phi_ind = bin_idx / n_psi;
                let psi_ind = bin_idx % n_psi;

                let phi = phi_centers[phi_ind];
                let psi = psi_centers[psi_ind];

                let mut rotamers = Vec::new();

                for (rot_idx, prob) in bin_data.probs.iter().enumerate() {
                    let coords = &bin_data.coords[rot_idx];
                    let coord_vec = coords
                        .iter()
                        .map(|&[x, y, z]| proxide_rotlib::rotlib_v1::Vec3 {
                            x: x as f32,
                            y: y as f32,
                            z: z as f32,
                        })
                        .collect();

                    rotamers.push(proxide_rotlib::rotlib_v1::Rotamer {
                        prob: *prob as f32,
                        chi: Vec::new(),
                        coords: coord_vec,
                    });
                }

                bins.push(proxide_rotlib::rotlib_v1::Bin {
                    phi,
                    psi,
                    freq: 0.0,
                    rotamers,
                });
            }

            let residue_entry = proxide_rotlib::rotlib_v1::ResidueEntry {
                code: code.clone(),
                atom_names: entry.atom_names.clone(),
                num_chi: 0,
                phi_centers,
                psi_centers,
                default_bin: entry.default_bin,
                bins,
            };

            residues.push(residue_entry);
        }

        proxide_rotlib::rotlib_v1::RotamerLibrary {
            version: 1,
            provenance: "Mosaist Grigoryan lab; testfiles/rotlib.bin; NON-COMMERCIAL ONLY; \
                NOT FOR REDISTRIBUTION"
                .to_string(),
            attribution: "Mosaist (https://github.com/Grigoryanlab/Mosaist), CC-BY-NC-SA 4.0, \
                Grigoryan lab, Dartmouth"
                .to_string(),
            data_license: "CC-BY-NC-SA-4.0".to_string(),
            geometry_mode: proxide_rotlib::rotlib_v1::GeometryMode::Precomputed as i32,
            residues,
            geometry_source,
            geometry_license,
        }
    }
}
