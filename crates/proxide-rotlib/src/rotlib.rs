use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::Path;
use crate::error::RotlibError;
use crate::rotamer_id::RotamerId;
use crate::binning::{angle_to_standard, find_closest_angle};
use prost::Message;

/// Per-rotamer data within a single phi/psi bin.
#[derive(Clone, Debug)]
pub(crate) struct BinData {
    /// Rotamer probabilities (len = nr).
    pub(crate) probs:  Vec<f64>,
    /// Atom coordinates in canonical backbone-relative frame, indexed [rot_index][atom_index].
    pub(crate) coords: Vec<Vec<[f64; 3]>>,
}

/// Per amino-acid rotamer library entry.
#[derive(Clone, Debug)]
pub(crate) struct AaEntry {
    /// Sidechain heavy-atom names in library order.
    pub(crate) atom_names:      Vec<String>,
    /// Sorted unique phi bin centers (degrees, ascending).
    pub(crate) bin_phi_centers: Vec<f64>,
    /// Sorted unique psi bin centers (degrees, ascending).
    pub(crate) bin_psi_centers: Vec<f64>,
    /// Index of the bin with the highest frequency (first-maximum wins).
    pub(crate) default_bin:     u32,
    /// Rotamer data indexed by linear bin index (phi-major: phi_ind * n_psi + psi_ind).
    pub(crate) rotamers:        Vec<BinData>,
}

/// Map key for the cis-proline rotamer entry. The Dunbrack 2010 backbone-
/// dependent library codes cis-proline as `CPR` (and trans-proline as `TPR`);
/// this must match whatever the loaded library names its cis-PRO entry.
const CIS_PRO_KEY: &str = "CPR";

/// Backbone-dependent rotamer library loaded from an MSL binary file.
#[derive(Debug)]
pub struct RotamerLibrary {
    pub(crate) entries: HashMap<String, AaEntry>,
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
        if b[0] == 0 { break; }
        bytes.push(b[0]);
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn validate_grid(bins: &[(f64, f64, f64)]) -> Result<(Vec<f64>, Vec<f64>), RotlibError> {
    use std::collections::HashSet;
    let mut pair_set: HashSet<(u64, u64)> = HashSet::new();
    let mut phi_set: HashSet<u64> = HashSet::new();
    let mut psi_set: HashSet<u64> = HashSet::new();
    for &(phi, psi, _) in bins {
        let pk = (phi.to_bits(), psi.to_bits());
        if !pair_set.insert(pk) {
            return Err(RotlibError::InvalidFormat(
                format!("duplicate (phi, psi) pair ({phi:.2}, {psi:.2})")
            ));
        }
        phi_set.insert(phi.to_bits());
        psi_set.insert(psi.to_bits());
    }
    if phi_set.len() * psi_set.len() != bins.len() {
        return Err(RotlibError::InvalidFormat(format!(
            "non-rectangular grid: {}φ × {}ψ = {} ≠ {} bins",
            phi_set.len(), psi_set.len(),
            phi_set.len() * psi_set.len(), bins.len()
        )));
    }
    let mut phi_centers: Vec<f64> = phi_set.iter().map(|&b| f64::from_bits(b)).collect();
    let mut psi_centers: Vec<f64> = psi_set.iter().map(|&b| f64::from_bits(b)).collect();
    phi_centers.sort_by(f64::total_cmp);
    psi_centers.sort_by(f64::total_cmp);
    Ok((phi_centers, psi_centers))
}

impl RotamerLibrary {
    /// Load from MSL binary file.
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
                for _ in 0..4 { read_cstring(&mut reader)?; }
            }

            // atom names
            let mut atom_names = Vec::with_capacity(na);
            for _ in 0..na { atom_names.push(read_cstring(&mut reader)?); }

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
            let default_bin = bin_descriptors.iter().enumerate()
                .fold((0usize, f64::NEG_INFINITY), |(best_i, best_f), (i, &(_, _, f))| {
                    if f > best_f { (i, f) } else { (best_i, best_f) }
                })
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

            entries.insert(aa_name, AaEntry {
                atom_names,
                bin_phi_centers,
                bin_psi_centers,
                default_bin,
                rotamers,
            });
        }

        Ok(RotamerLibrary { entries })
    }

    /// Load from protobuf format (zstd-compressed).
    ///
    /// Reads a `.pb.zst` file, decompresses using zstd, decodes the protobuf,
    /// validates required fields, and populates the rotamer library entries.
    ///
    /// # Errors
    ///
    /// - `RotlibError::Io` if file cannot be read
    /// - `RotlibError::InvalidFormat` if decompression fails
    /// - `RotlibError::Protobuf` if protobuf decoding fails
    /// - `RotlibError::MissingAttribution` if attribution field is empty
    /// - `RotlibError::UnsupportedGeometryMode` if geometry_mode is not PRECOMPUTED
    pub fn load_pb(path: &Path) -> Result<Self, RotlibError> {
        use std::fs::File;

        // 1. Read file
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut compressed = Vec::new();
        reader.read_to_end(&mut compressed)?;

        // 2. Decompress zstd
        let decompressed = zstd::decode_all(compressed.as_slice())
            .map_err(|e| RotlibError::InvalidFormat(format!("zstd decompression failed: {}", e)))?;

        // 3. Decode protobuf
        let pb_lib = crate::pb::rotlib_v1::RotamerLibrary::decode(decompressed.as_slice())?;

        // 4. Validate attribution
        if pb_lib.attribution.is_empty() {
            return Err(RotlibError::MissingAttribution);
        }

        // 5. Validate geometry mode (must be PRECOMPUTED)
        if pb_lib.geometry_mode != crate::pb::rotlib_v1::GeometryMode::Precomputed as i32 {
            return Err(RotlibError::UnsupportedGeometryMode(pb_lib.geometry_mode));
        }

        // 6. Build entries map
        let mut entries: HashMap<String, AaEntry> = HashMap::new();

        for residue in pb_lib.residues {
            let mut bin_phi_centers = residue.phi_centers.clone();
            let mut bin_psi_centers = residue.psi_centers.clone();

            // Sort phi and psi centers defensively
            bin_phi_centers.sort_by(f64::total_cmp);
            bin_psi_centers.sort_by(f64::total_cmp);

            let n_phi = bin_phi_centers.len();
            let n_psi = bin_psi_centers.len();
            let expected_grid_size = n_phi * n_psi;

            // Verify grid is rectangular
            if residue.bins.len() != expected_grid_size {
                return Err(RotlibError::InvalidFormat(format!(
                    "residue '{}': non-rectangular grid: {}φ × {}ψ = {} ≠ {} bins",
                    residue.code, n_phi, n_psi, expected_grid_size, residue.bins.len()
                )));
            }

            // Build linear rotamer array indexed phi-major
            let mut rotamers: Vec<BinData> = vec![
                BinData {
                    probs: Vec::new(),
                    coords: Vec::new(),
                };
                expected_grid_size
            ];

            // Place bins in linear array
            for bin in &residue.bins {
                // Find indices of this bin's phi and psi in the sorted centers
                let phi_ind = bin_phi_centers.iter().position(|&p| {
                    (p - bin.phi).abs() < 1e-9
                }).ok_or_else(|| {
                    RotlibError::InvalidFormat(format!(
                        "residue '{}': bin phi {} not in phi_centers",
                        residue.code, bin.phi
                    ))
                })?;

                let psi_ind = bin_psi_centers.iter().position(|&p| {
                    (p - bin.psi).abs() < 1e-9
                }).ok_or_else(|| {
                    RotlibError::InvalidFormat(format!(
                        "residue '{}': bin psi {} not in psi_centers",
                        residue.code, bin.psi
                    ))
                })?;

                let linear_index = phi_ind * n_psi + psi_ind;

                // Check if already filled (duplicate bin)
                if !rotamers[linear_index].probs.is_empty() {
                    return Err(RotlibError::InvalidFormat(format!(
                        "residue '{}': duplicate bin at (phi={}, psi={})",
                        residue.code, bin.phi, bin.psi
                    )));
                }

                // Build BinData from protobuf Bin
                let mut probs = Vec::new();
                let mut coords: Vec<Vec<[f64; 3]>> = Vec::new();

                for rotamer in &bin.rotamers {
                    probs.push(rotamer.prob as f64);

                    // Verify coords length matches atom_names
                    if rotamer.coords.len() != residue.atom_names.len() {
                        return Err(RotlibError::InvalidFormat(format!(
                            "residue '{}': rotamer has {} coords but {} atom names",
                            residue.code, rotamer.coords.len(), residue.atom_names.len()
                        )));
                    }

                    // Convert Vec3 to [f64; 3]
                    let coord_array: Vec<[f64; 3]> = rotamer.coords.iter()
                        .map(|v| [v.x as f64, v.y as f64, v.z as f64])
                        .collect();

                    coords.push(coord_array);
                }

                rotamers[linear_index] = BinData { probs, coords };
            }

            // Create AaEntry
            let entry = AaEntry {
                atom_names: residue.atom_names.clone(),
                bin_phi_centers,
                bin_psi_centers,
                default_bin: residue.default_bin,
                rotamers,
            };

            entries.insert(residue.code.clone(), entry);
        }

        Ok(RotamerLibrary { entries })
    }

    /// Return `true` if the library contains rotamer data for amino acid `aa`.
    pub fn contains_aa(&self, aa: &str) -> bool {
        self.entries.contains_key(aa)
    }

    /// Resolve the effective entry and bin index, handling cis-proline routing.
    ///
    /// This helper ensures that when `cis_proline=true` and `aa="PRO"`, both the
    /// entry AND the bin come from the CPR (cis-proline) entry if available.
    /// Otherwise, uses the standard amino acid entry and computes the bin from
    /// its own grid.
    ///
    /// Returns a reference to the resolved AaEntry and the computed bin index.
    ///
    /// # Errors
    ///
    /// Returns `RotlibError::UnknownAa` if the effective amino acid is not in the library.
    fn resolve_entry_bin(&self, aa: &str, phi: f64, psi: f64, cis_proline: bool)
        -> Result<(&AaEntry, usize), RotlibError>
    {
        // Determine effective key: use CPR if cis-PRO is requested and available
        let effective_aa = if cis_proline && aa == "PRO" {
            if self.entries.contains_key(CIS_PRO_KEY) {
                CIS_PRO_KEY
            } else {
                tracing::warn!(
                    "cis-PRO requested but library has no cis-PRO data; using standard PRO bin"
                );
                "PRO"
            }
        } else {
            aa
        };

        let entry = self.entries.get(effective_aa)
            .ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))?;

        // Compute bin from entry's OWN grid
        let bin = if phi == 9999.0 || psi == 9999.0 {
            entry.default_bin as usize
        } else {
            let phi_ind = find_closest_angle(&entry.bin_phi_centers, phi);
            let psi_ind = find_closest_angle(&entry.bin_psi_centers, psi);
            phi_ind * entry.bin_psi_centers.len() + psi_ind
        };

        Ok((entry, bin))
    }

    /// Number of rotamers in the φ/ψ bin closest to `(phi, psi)` for amino acid `aa`.
    pub fn num_rotamers(&self, aa: &str, phi: f64, psi: f64, cis_proline: bool) -> Result<usize, RotlibError> {
        let (entry, bin) = self.resolve_entry_bin(aa, phi, psi, cis_proline)?;
        Ok(entry.rotamers[bin].probs.len())
    }

    /// Probability of rotamer `rot_index` in the φ/ψ bin closest to `(phi, psi)`.
    ///
    /// Returns `Err(RotIndexOob)` if `rot_index ≥ num_rotamers`.
    pub fn rotamer_probability(&self, aa: &str, rot_index: usize, phi: f64, psi: f64) -> Result<f64, RotlibError> {
        let entry = self.entries.get(aa).ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))?;
        let bin = self.backbone_bin(aa, phi, psi, false)? as usize;
        let nr = entry.rotamers[bin].probs.len();
        if rot_index >= nr {
            return Err(RotlibError::RotIndexOob(aa.to_string(), rot_index, nr));
        }
        Ok(entry.rotamers[bin].probs[rot_index])
    }

    pub fn rotamer_probability_by_id(&self, id: &RotamerId) -> Result<f64, RotlibError> {
        let entry = self.entries.get(&id.aa).ok_or_else(|| RotlibError::UnknownAa(id.aa.clone()))?;
        // bin_index must be library-derived; OOB panics by documented precondition
        let bin = id.bin_index as usize;
        let nr = entry.rotamers[bin].probs.len();
        let rot_index = id.rot_index as usize;
        if rot_index >= nr {
            return Err(RotlibError::RotIndexOob(id.aa.clone(), rot_index, nr));
        }
        Ok(entry.rotamers[bin].probs[rot_index])
    }

    /// Place rotamer `rot_index` of amino acid `aa` into world coordinates.
    ///
    /// Looks up canonical backbone-relative sidechain coordinates from the
    /// library bin closest to `(phi, psi)`, then transforms them into the
    /// world frame defined by the backbone atoms `(n, ca, c)` using
    /// [`backbone_frame`](crate::frame::backbone_frame) and
    /// [`Transform::switch_frames`](crate::frame::Transform::switch_frames).
    ///
    /// Returns a [`PlacedRotamer`](crate::rotamer_id::PlacedRotamer) whose
    /// `atoms` field lists every sidechain heavy atom (excluding backbone).
    #[allow(clippy::too_many_arguments)]
    pub fn place_rotamer(&self, aa: &str, phi: f64, psi: f64, rot_index: usize, cis_proline: bool, n: [f64; 3], ca: [f64; 3], c: [f64; 3]) -> Result<crate::rotamer_id::PlacedRotamer, RotlibError> {
        use crate::frame::{backbone_frame, Frame, Transform};
        use crate::rotamer_id::{PlacedAtom, PlacedRotamer, RotamerId};

        let (entry, bin) = self.resolve_entry_bin(aa, phi, psi, cis_proline)?;
        let bin_data = &entry.rotamers[bin];
        let nr = bin_data.coords.len();
        if rot_index >= nr {
            return Err(RotlibError::RotIndexOob(aa.to_string(), rot_index, nr));
        }

        let res_frame = backbone_frame(n, ca, c);
        let lab_frame = Frame::identity();
        let xform = Transform::switch_frames(&res_frame, &lab_frame);

        let atoms = bin_data.coords[rot_index].iter()
            .zip(&entry.atom_names)
            .map(|(&xyz, name)| PlacedAtom { name: name.clone(), xyz: xform.apply(xyz) })
            .collect();

        Ok(PlacedRotamer {
            id: RotamerId { aa: aa.to_string(), bin_index: bin as u32, rot_index: rot_index as u32 },
            atoms,
        })
    }

    /// Ordered sidechain heavy-atom names for `aa` as stored in the library.
    ///
    /// The slice order matches the coordinate vectors returned by
    /// [`place_rotamer`](RotamerLibrary::place_rotamer).
    pub fn sidechain_atom_names(&self, aa: &str) -> Result<&[String], RotlibError> {
        self.entries.get(aa)
            .map(|e| e.atom_names.as_slice())
            .ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))
    }

    /// Look up the backbone-dependent bin index for `aa` at (`phi`, `psi`).
    ///
    /// Uses nearest-neighbour mapping into the Dunbrack BBdep 20°×20° grid
    /// (10° bin centers). The returned bin index may be passed to
    /// [`place_rotamer`](RotamerLibrary::place_rotamer).
    ///
    /// # `cis_proline` handling
    ///
    /// When `aa == "PRO"` and `cis_proline` is `true`:
    /// - If the library contains a dedicated `CPR` entry, routes to that entry (uses its bin grid).
    /// - Otherwise, emits a warning via `tracing::warn!` and falls back to the standard PRO bin.
    ///
    /// For all other amino acids, the `cis_proline` flag is ignored.
    ///
    /// # Warning: sparse φ region
    ///
    /// Accuracy degrades for φ≥−30°: these bins have <3 crystallographic observations
    /// in the Dunbrack library. Nearest-neighbor lookup is still correct but rotamer
    /// probabilities in this region have poor statistical support. Placement in this
    /// region should be treated as a low-confidence estimate.
    pub fn backbone_bin(&self, aa: &str, phi: f64, psi: f64, cis_proline: bool) -> Result<u32, RotlibError> {
        let (_, bin) = self.resolve_entry_bin(aa, phi, psi, cis_proline)?;
        Ok(bin as u32)
    }

    /// Get all residue codes present in the library.
    pub fn residue_codes(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Convert MASTER library to protobuf format. Used by parse_master binary.
    pub fn to_protobuf(&self, geometry_source: String, geometry_license: String) -> crate::pb::rotlib_v1::RotamerLibrary {
        let mut residues = Vec::new();

        for (code, entry) in &self.entries {
            // Collect phi/psi centers
            let phi_centers = entry.bin_phi_centers.clone();
            let psi_centers = entry.bin_psi_centers.clone();

            // Convert bins: linear array to protobuf structure
            let mut bins = Vec::new();

            for (bin_idx, bin_data) in entry.rotamers.iter().enumerate() {
                // Reconstruct phi, psi from linear index (phi-major order: phi_ind * n_psi + psi_ind)
                let n_psi = psi_centers.len();
                let phi_ind = bin_idx / n_psi;
                let psi_ind = bin_idx % n_psi;

                let phi = phi_centers[phi_ind];
                let psi = psi_centers[psi_ind];

                // Convert rotamers within this bin
                let mut rotamers = Vec::new();

                for (rot_idx, prob) in bin_data.probs.iter().enumerate() {
                    let coords = &bin_data.coords[rot_idx];
                    let coord_vec = coords
                        .iter()
                        .map(|&[x, y, z]| crate::pb::rotlib_v1::Vec3 {
                            x: x as f32,
                            y: y as f32,
                            z: z as f32,
                        })
                        .collect();

                    rotamers.push(crate::pb::rotlib_v1::Rotamer {
                        prob: *prob as f32,
                        chi: Vec::new(), // No explicit chi in PRECOMPUTED mode
                        coords: coord_vec,
                    });
                }

                bins.push(crate::pb::rotlib_v1::Bin {
                    phi,
                    psi,
                    freq: 0.0, // Not available from MASTER; placeholder
                    rotamers,
                });
            }

            // Build ResidueEntry
            let residue_entry = crate::pb::rotlib_v1::ResidueEntry {
                code: code.clone(),
                atom_names: entry.atom_names.clone(),
                num_chi: 0, // Not tracked in MASTER binary
                phi_centers,
                psi_centers,
                default_bin: entry.default_bin,
                bins,
            };

            residues.push(residue_entry);
        }

        crate::pb::rotlib_v1::RotamerLibrary {
            version: 1,
            provenance: "Mosaist Grigoryan lab; testfiles/rotlib.bin; NON-COMMERCIAL ONLY; \
                NOT FOR REDISTRIBUTION"
                .to_string(),
            attribution: "Mosaist (https://github.com/Grigoryanlab/Mosaist), CC-BY-NC-SA 4.0, \
                Grigoryan lab, Dartmouth"
                .to_string(),
            data_license: "CC-BY-NC-SA-4.0".to_string(),
            geometry_mode: crate::pb::rotlib_v1::GeometryMode::Precomputed as i32,
            residues,
            geometry_source,
            geometry_license,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backbone_bin_unknown_aa() {
        let lib = RotamerLibrary { entries: std::collections::HashMap::new() };
        let result = lib.backbone_bin("UNK", -60.0, -45.0, false);
        assert!(result.is_err());
        match result {
            Err(RotlibError::UnknownAa(aa)) => assert_eq!(aa, "UNK"),
            _ => panic!("Expected UnknownAa error"),
        }
    }

    #[test]
    fn test_backbone_bin_sentinel_phi() {
        use std::collections::HashMap;
        let entry = AaEntry {
            atom_names: vec!["CB".to_string()],
            bin_phi_centers: vec![-120.0, -60.0, 60.0],
            bin_psi_centers: vec![-45.0, -20.0, 20.0, 45.0],
            default_bin: 5,
            rotamers: vec![
                BinData { probs: vec![0.5], coords: vec![vec![[1.0, 2.0, 3.0]]] };
                12
            ],
        };
        let mut entries = HashMap::new();
        entries.insert("ALA".to_string(), entry);
        let lib = RotamerLibrary { entries };

        let result = lib.backbone_bin("ALA", 9999.0, -45.0, false);
        assert_eq!(result.unwrap(), 5);
    }

    #[test]
    fn test_backbone_bin_sentinel_psi() {
        use std::collections::HashMap;
        let entry = AaEntry {
            atom_names: vec!["CB".to_string()],
            bin_phi_centers: vec![-120.0, -60.0, 60.0],
            bin_psi_centers: vec![-45.0, -20.0, 20.0, 45.0],
            default_bin: 7,
            rotamers: vec![
                BinData { probs: vec![0.5], coords: vec![vec![[1.0, 2.0, 3.0]]] };
                12
            ],
        };
        let mut entries = HashMap::new();
        entries.insert("ALA".to_string(), entry);
        let lib = RotamerLibrary { entries };

        let result = lib.backbone_bin("ALA", -60.0, 9999.0, false);
        assert_eq!(result.unwrap(), 7);
    }

    #[test]
    fn test_place_rotamer_unknown_aa() {
        let lib = RotamerLibrary { entries: std::collections::HashMap::new() };
        let result = lib.place_rotamer("UNK", -60.0, -45.0, 0, false, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
        assert!(result.is_err());
        match result {
            Err(RotlibError::UnknownAa(aa)) => assert_eq!(aa, "UNK"),
            _ => panic!("Expected UnknownAa error"),
        }
    }

    #[test]
    fn test_place_rotamer_rot_index_oob() {
        use std::collections::HashMap;
        let entry = AaEntry {
            atom_names: vec!["CB".to_string()],
            bin_phi_centers: vec![-120.0, -60.0, 60.0],
            bin_psi_centers: vec![-45.0, -20.0, 20.0, 45.0],
            default_bin: 0,
            rotamers: vec![
                BinData {
                    probs: vec![0.5],
                    coords: vec![vec![[1.0, 2.0, 3.0]]]
                };
                12
            ],
        };
        let mut entries = HashMap::new();
        entries.insert("ALA".to_string(), entry);
        let lib = RotamerLibrary { entries };

        let result = lib.place_rotamer("ALA", -120.0, -45.0, 5, false, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
        assert!(result.is_err());
        match result {
            Err(RotlibError::RotIndexOob(aa, idx, nr)) => {
                assert_eq!(aa, "ALA");
                assert_eq!(idx, 5);
                assert_eq!(nr, 1);
            },
            _ => panic!("Expected RotIndexOob error"),
        }
    }

    #[test]
    fn test_place_rotamer_creates_placed_atom() {
        use std::collections::HashMap;
        let entry = AaEntry {
            atom_names: vec!["CB".to_string(), "CG".to_string()],
            bin_phi_centers: vec![-120.0, -60.0, 60.0],
            bin_psi_centers: vec![-45.0, -20.0, 20.0, 45.0],
            default_bin: 0,
            rotamers: vec![
                BinData {
                    probs: vec![0.8],
                    coords: vec![vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
                };
                12
            ],
        };
        let mut entries = HashMap::new();
        entries.insert("VAL".to_string(), entry);
        let lib = RotamerLibrary { entries };

        // Use backbone coordinates that form identity-like frame
        let n = [1.0, 0.0, 0.0];
        let ca = [0.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];

        let result = lib.place_rotamer("VAL", -120.0, -45.0, 0, false, n, ca, c);
        assert!(result.is_ok());
        let placed = result.unwrap();

        assert_eq!(placed.id.aa, "VAL");
        assert_eq!(placed.id.bin_index, 0);
        assert_eq!(placed.id.rot_index, 0);
        assert_eq!(placed.atoms.len(), 2);
        assert_eq!(placed.atoms[0].name, "CB");
        assert_eq!(placed.atoms[1].name, "CG");
    }

    #[test]
    fn test_backbone_bin_cis_pro_routing() {
        use std::collections::HashMap;
        // Create two entries: standard PRO and cis-PRO (CPR)
        // They will have different grids so we can verify the correct one is selected
        let pro_entry = AaEntry {
            atom_names: vec!["CB".to_string()],
            bin_phi_centers: vec![-120.0, -60.0, 60.0],
            bin_psi_centers: vec![-45.0, -20.0, 20.0, 45.0],
            default_bin: 0,
            rotamers: vec![
                BinData { probs: vec![0.5], coords: vec![vec![[1.0, 2.0, 3.0]]] };
                12
            ],
        };

        let cpro_entry = AaEntry {
            atom_names: vec!["CB".to_string()],
            // Different grid for cis-PRO
            bin_phi_centers: vec![-100.0, -40.0, 80.0],
            bin_psi_centers: vec![-30.0, 0.0, 30.0, 60.0],
            default_bin: 1,
            rotamers: vec![
                BinData { probs: vec![0.6], coords: vec![vec![[4.0, 5.0, 6.0]]] };
                12
            ],
        };

        let mut entries = HashMap::new();
        entries.insert("PRO".to_string(), pro_entry.clone());
        entries.insert("CPR".to_string(), cpro_entry);
        let lib = RotamerLibrary { entries };

        // Test 1: cis_proline=true routes to CPR
        // Query with angles near -100, -30 (cis-PRO grid)
        let result_cis = lib.backbone_bin("PRO", -100.0, -30.0, true);
        assert!(result_cis.is_ok());
        let bin_cis = result_cis.unwrap() as usize;
        // The cis-PRO grid is 3×4 = 12 bins, -100 is at index 0, -30 is at index 0
        // So we expect bin 0
        assert_eq!(bin_cis, 0);

        // Test 2: cis_proline=false routes to standard PRO
        let result_trans = lib.backbone_bin("PRO", -120.0, -45.0, false);
        assert!(result_trans.is_ok());
        let bin_trans = result_trans.unwrap() as usize;
        // The standard PRO grid is 3×4 = 12 bins, -120 is at index 0, -45 is at index 0
        // So we expect bin 0
        assert_eq!(bin_trans, 0);

        // Test 3: cis_proline=true without CPR entry falls back and logs warning
        let mut entries_no_cpro = HashMap::new();
        entries_no_cpro.insert("PRO".to_string(), pro_entry);
        let lib_no_cpro = RotamerLibrary { entries: entries_no_cpro };

        let result_fallback = lib_no_cpro.backbone_bin("PRO", -120.0, -45.0, true);
        assert!(result_fallback.is_ok());
        let bin_fallback = result_fallback.unwrap() as usize;
        // Without CPR, should fall back to PRO and return bin 0
        assert_eq!(bin_fallback, 0);
    }
}
