use crate::binning::{angle_to_standard, find_closest_angle};
use crate::error::RotlibError;
use crate::rotamer_id::RotamerId;
use prost::Message;
use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::Path;

/// Per-rotamer data within a single phi/psi bin.
#[derive(Clone, Debug)]
pub(crate) struct BinData {
    /// Rotamer probabilities (len = nr).
    pub(crate) probs: Vec<f64>,
    /// Atom coordinates in canonical backbone-relative frame, indexed [rot_index][atom_index].
    pub(crate) coords: Vec<Vec<[f64; 3]>>,
}

/// Per amino-acid rotamer library entry.
#[derive(Clone, Debug)]
pub(crate) struct AaEntry {
    /// Sidechain heavy-atom names in library order.
    pub(crate) atom_names: Vec<String>,
    /// Sorted unique phi bin centers (degrees, ascending).
    pub(crate) bin_phi_centers: Vec<f64>,
    /// Sorted unique psi bin centers (degrees, ascending).
    pub(crate) bin_psi_centers: Vec<f64>,
    /// Index of the bin with the highest frequency (first-maximum wins).
    pub(crate) default_bin: u32,
    /// Rotamer data indexed by linear bin index (phi-major: phi_ind * n_psi + psi_ind).
    pub(crate) rotamers: Vec<BinData>,
}

/// Aggregate, non-identifying statistics for one amino acid's rotamer grid.
/// See [`RotamerLibrary::aa_grid_summary`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AaGridSummary {
    /// Total number of (phi, psi) bins for this residue.
    pub bin_count: usize,
    /// Fewest rotamers in any single bin.
    pub min_rotamers_per_bin: usize,
    /// Most rotamers in any single bin.
    pub max_rotamers_per_bin: usize,
    /// Lowest rotamer probability observed across all bins.
    pub min_probability: f64,
    /// Highest rotamer probability observed across all bins.
    pub max_probability: f64,
    /// Max pairwise distance (Angstrom) between the rotamer-0 CB position across all
    /// bins, or `None` if this residue has no CB atom (e.g. GLY) or fewer than 2 bins.
    pub max_cb_spread_angstrom: Option<f64>,
}

/// Map key for the cis-proline rotamer entry. The Dunbrack 2010 backbone-
/// dependent library codes cis-proline as `CPR` (and trans-proline as `TPR`);
/// this must match whatever the loaded library names its cis-PRO entry.
const CIS_PRO_KEY: &str = "CPR";

/// Backbone-dependent rotamer library loaded from a Protobuf asset.
#[derive(Debug, Clone)]
pub struct RotamerLibrary {
    pub(crate) entries: HashMap<String, AaEntry>,
}

impl RotamerLibrary {
    /// Decode a RotamerLibrary from an in-memory Protobuf message.
    pub fn from_protobuf(pb_lib: &crate::pb::rotlib_v1::RotamerLibrary) -> Result<Self, RotlibError> {
        // 1. Validate attribution
        if pb_lib.attribution.is_empty() {
            return Err(RotlibError::MissingAttribution);
        }

        // 2. Validate geometry mode (must be PRECOMPUTED)
        if pb_lib.geometry_mode != crate::pb::rotlib_v1::GeometryMode::Precomputed as i32 {
            return Err(RotlibError::UnsupportedGeometryMode(pb_lib.geometry_mode));
        }

        // 3. Build entries map
        let mut entries: HashMap<String, AaEntry> = HashMap::new();

        for residue in &pb_lib.residues {
            let mut bin_phi_centers = residue.phi_centers.clone();
            let mut bin_psi_centers = residue.psi_centers.clone();

            bin_phi_centers.sort_by(f64::total_cmp);
            bin_psi_centers.sort_by(f64::total_cmp);

            let n_phi = bin_phi_centers.len();
            let n_psi = bin_psi_centers.len();
            let expected_grid_size = n_phi * n_psi;

            if residue.bins.len() != expected_grid_size {
                return Err(RotlibError::InvalidFormat(format!(
                    "residue '{}': non-rectangular grid: {}φ × {}ψ = {} ≠ {} bins",
                    residue.code,
                    n_phi,
                    n_psi,
                    expected_grid_size,
                    residue.bins.len()
                )));
            }

            let mut rotamers: Vec<BinData> = vec![
                BinData {
                    probs: Vec::new(),
                    coords: Vec::new(),
                };
                expected_grid_size
            ];

            for bin in &residue.bins {
                let phi_ind = bin_phi_centers
                    .iter()
                    .position(|&p| (p - bin.phi).abs() < 1e-9)
                    .ok_or_else(|| {
                        RotlibError::InvalidFormat(format!(
                            "residue '{}': bin phi {} not in phi_centers",
                            residue.code, bin.phi
                        ))
                    })?;

                let psi_ind = bin_psi_centers
                    .iter()
                    .position(|&p| (p - bin.psi).abs() < 1e-9)
                    .ok_or_else(|| {
                        RotlibError::InvalidFormat(format!(
                            "residue '{}': bin psi {} not in psi_centers",
                            residue.code, bin.psi
                        ))
                    })?;

                let linear_index = phi_ind * n_psi + psi_ind;

                if !rotamers[linear_index].probs.is_empty() {
                    return Err(RotlibError::InvalidFormat(format!(
                        "residue '{}': duplicate bin at (phi={}, psi={})",
                        residue.code, bin.phi, bin.psi
                    )));
                }

                let mut probs = Vec::new();
                let mut coords: Vec<Vec<[f64; 3]>> = Vec::new();

                for rotamer in &bin.rotamers {
                    probs.push(rotamer.prob as f64);

                    if rotamer.coords.len() != residue.atom_names.len() {
                        return Err(RotlibError::InvalidFormat(format!(
                            "residue '{}': rotamer has {} coords but {} atom names",
                            residue.code,
                            rotamer.coords.len(),
                            residue.atom_names.len()
                        )));
                    }

                    let coord_array: Vec<[f64; 3]> = rotamer
                        .coords
                        .iter()
                        .map(|v| [v.x as f64, v.y as f64, v.z as f64])
                        .collect();

                    coords.push(coord_array);
                }

                rotamers[linear_index] = BinData { probs, coords };
            }

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

    /// Load from uncompressed protobuf file (.pb) or automatically decompress if zstd.
    pub fn load_pb(path: &Path) -> Result<Self, RotlibError> {
        use std::fs::File;

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;

        let decoded_bytes = if bytes.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
            zstd::decode_all(bytes.as_slice())
                .map_err(|e| RotlibError::InvalidFormat(format!("zstd decompression failed: {}", e)))?
        } else {
            bytes
        };

        let pb_lib = crate::pb::rotlib_v1::RotamerLibrary::decode(decoded_bytes.as_slice())?;
        Self::from_protobuf(&pb_lib)
    }

    /// Load from zstd-compressed protobuf format (.pb.zst).
    pub fn load_pb_zst(path: &Path) -> Result<Self, RotlibError> {
        Self::load_pb(path)
    }

    /// Return `true` if the library contains rotamer data for amino acid `aa`.
    pub fn contains_aa(&self, aa: &str) -> bool {
        self.entries.contains_key(aa)
    }

    /// Aggregate, non-identifying statistics for one amino acid's rotamer grid: bin
    /// count, rotamer-per-bin range, probability range, and CB-position spread across
    /// bins. Exists so callers (and tests) can characterize a loaded library's shape for
    /// a residue -- e.g. confirming a single-bin/single-rotamer synthetic entry -- without
    /// exposing or copying any per-bin raw data.
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

        // CB spread: max pairwise distance (A) between the rotamer-0 CB position of
        // every bin, if this residue has a CB atom. None for residues without CB.
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
    fn resolve_entry_bin(
        &self,
        aa: &str,
        phi: Option<f64>,
        psi: Option<f64>,
        cis_proline: bool,
    ) -> Result<(&AaEntry, usize), RotlibError> {
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

        let entry = self
            .entries
            .get(effective_aa)
            .ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))?;

        // Under Debt #1897, missing dihedral angles are represented strictly as None.
        // Non-finite dihedrals (NaN or infinite) raise an explicit error.
        // If either angle is None, route to entry.default_bin.
        if phi.is_some_and(|v| !v.is_finite()) || psi.is_some_and(|v| !v.is_finite()) {
            return Err(RotlibError::InvalidFormat(format!(
                "NaN or non-finite dihedral angle provided: phi={phi:?}, psi={psi:?}"
            )));
        }

        let bin = match (phi, psi) {
            (Some(p), Some(s)) => {
                let norm_phi = angle_to_standard(p);
                let norm_psi = angle_to_standard(s);
                let phi_ind = find_closest_angle(&entry.bin_phi_centers, norm_phi);
                let psi_ind = find_closest_angle(&entry.bin_psi_centers, norm_psi);
                phi_ind * entry.bin_psi_centers.len() + psi_ind
            }
            (None, _) | (_, None) => entry.default_bin as usize,
        };

        Ok((entry, bin))
    }

    /// Number of rotamers available for residue at given dihedral angles.
    ///
    /// For PRO residues with cis_proline = true, routes to the CPR library definition.
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

    /// Probability of specific rotamer index.
    pub fn rotamer_probability(
        &self,
        aa: &str,
        phi: Option<f64>,
        psi: Option<f64>,
        cis_proline: bool,
        rot_index: usize,
    ) -> Result<f64, RotlibError> {
        let (entry, bin) = self.resolve_entry_bin(aa, phi, psi, cis_proline)?;
        let nr = entry.rotamers[bin].probs.len();
        if rot_index >= nr {
            return Err(RotlibError::RotIndexOob(aa.to_string(), rot_index, nr));
        }
        Ok(entry.rotamers[bin].probs[rot_index])
    }

    pub fn rotamer_probability_by_id(&self, id: &RotamerId) -> Result<f64, RotlibError> {
        let entry = self
            .entries
            .get(&id.aa)
            .ok_or_else(|| RotlibError::UnknownAa(id.aa.clone()))?;
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
    /// Constructs a standard right-handed orthonormal coordinate frame at Cα via
    /// Gram-Schmidt orthonormalization (Dunbrack & Cohen 1997; spec §2.7, §4.4).
    /// For Glycine (`GLY`), returns an empty atom list.
    #[allow(clippy::too_many_arguments)]
    pub fn place_rotamer(
        &self,
        aa: &str,
        phi: Option<f64>,
        psi: Option<f64>,
        rot_index: usize,
        cis_proline: bool,
        n: [f64; 3],
        ca: [f64; 3],
        c: [f64; 3],
    ) -> Result<crate::rotamer_id::PlacedRotamer, RotlibError> {
        use crate::rotamer_id::{PlacedAtom, PlacedRotamer, RotamerId};

        if aa == "GLY" {
            return Ok(PlacedRotamer {
                id: RotamerId {
                    aa: "GLY".to_string(),
                    bin_index: 0,
                    rot_index: 0,
                },
                used_default_bin: phi.is_none() || psi.is_none(),
                atoms: Vec::new(),
            });
        }

        let (entry, bin) = self.resolve_entry_bin(aa, phi, psi, cis_proline)?;
        let used_default_bin = phi.is_none() || psi.is_none();
        let bin_data = &entry.rotamers[bin];
        let nr = bin_data.coords.len();
        if rot_index >= nr {
            return Err(RotlibError::RotIndexOob(aa.to_string(), rot_index, nr));
        }

        // Construct standard right-handed orthonormal triad ux, uy, uz at CA
        // via Gram-Schmidt orthonormalization (Dunbrack & Cohen 1997; spec §2.7, §4.4)
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let scale = |v: [f64; 3], s: f64| [v[0] / s, v[1] / s, v[2] / s];
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };

        let vx = sub(ca, n);
        let ux = scale(vx, norm(vx));
        let vc = sub(c, ca);
        let vz = cross(ux, vc);
        let uz = scale(vz, norm(vz));
        let uy = cross(uz, ux);

        let atoms = bin_data.coords[rot_index]
            .iter()
            .zip(&entry.atom_names)
            .map(|(&local, name)| {
                let x = local[0];
                let y = local[1];
                let z = local[2];
                let global = [
                    ca[0] + x * ux[0] + y * uy[0] + z * uz[0],
                    ca[1] + x * ux[1] + y * uy[1] + z * uz[1],
                    ca[2] + x * ux[2] + y * uy[2] + z * uz[2],
                ];
                PlacedAtom {
                    name: name.clone(),
                    xyz: global,
                }
            })
            .collect();

        Ok(PlacedRotamer {
            id: RotamerId {
                aa: aa.to_string(),
                bin_index: bin as u32,
                rot_index: rot_index as u32,
            },
            used_default_bin,
            atoms,
        })
    }

    /// Ordered sidechain heavy-atom names for `aa` as stored in the library.
    pub fn sidechain_atom_names(&self, aa: &str) -> Result<&[String], RotlibError> {
        self.entries
            .get(aa)
            .map(|e| e.atom_names.as_slice())
            .ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))
    }

    /// Look up the discrete Ramachandran bin index for `aa` at (`phi`, `psi`).
    pub fn backbone_bin(
        &self,
        aa: &str,
        phi: Option<f64>,
        psi: Option<f64>,
        cis_proline: bool,
    ) -> Result<usize, RotlibError> {
        let (_, bin) = self.resolve_entry_bin(aa, phi, psi, cis_proline)?;
        Ok(bin)
    }

    /// Look up the discrete Ramachandran bin index for `aa` at (`phi`, `psi`),
    /// returning `(bin_index, used_default_bin)` so callers can observe when an
    /// undefined angle (`None`) routed to `default_bin` (Debt #1897).
    pub fn backbone_bin_with_fallback(
        &self,
        aa: &str,
        phi: Option<f64>,
        psi: Option<f64>,
        cis_proline: bool,
    ) -> Result<(usize, bool), RotlibError> {
        let (_, bin) = self.resolve_entry_bin(aa, phi, psi, cis_proline)?;
        Ok((bin, phi.is_none() || psi.is_none()))
    }

    /// Get all residue codes present in the library.
    pub fn residue_codes(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backbone_bin_unknown_aa() {
        let lib = RotamerLibrary {
            entries: std::collections::HashMap::new(),
        };
        let result = lib.backbone_bin("UNK", Some(-60.0), Some(-45.0), false);
        assert!(result.is_err());
        match result {
            Err(RotlibError::UnknownAa(aa)) => assert_eq!(aa, "UNK"),
            _ => panic!("Expected UnknownAa error"),
        }
    }

    #[test]
    fn test_backbone_bin_none_dihedrals() {
        use std::collections::HashMap;
        let entry = AaEntry {
            atom_names: vec!["CB".to_string()],
            bin_phi_centers: vec![-120.0, -60.0, 60.0],
            bin_psi_centers: vec![-45.0, -20.0, 20.0, 45.0],
            default_bin: 5,
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

        // None phi routes to default_bin
        let result = lib.backbone_bin("ALA", None, Some(-45.0), false);
        assert_eq!(result.unwrap(), 5);

        // None psi routes to default_bin
        let result = lib.backbone_bin("ALA", Some(-60.0), None, false);
        assert_eq!(result.unwrap(), 5);

        // Both None route to default_bin
        let result = lib.backbone_bin("ALA", None, None, false);
        assert_eq!(result.unwrap(), 5);
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

        let result = lib.backbone_bin("ALA", None, Some(-45.0), false);
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

        let result = lib.backbone_bin("ALA", Some(-60.0), None, false);
        assert_eq!(result.unwrap(), 7);
    }

    #[test]
    fn test_place_rotamer_unknown_aa() {
        let lib = RotamerLibrary {
            entries: std::collections::HashMap::new(),
        };
        let result = lib.place_rotamer(
            "UNK",
            Some(-60.0),
            Some(-45.0),
            0,
            false,
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        );
        assert!(result.is_err());
        match result {
            Err(RotlibError::UnknownAa(aa)) => assert_eq!(aa, "UNK"),
            _ => panic!("Expected UnknownAa error"),
        }
    }

    #[test]
    fn test_place_rotamer_gly_returns_empty() {
        let lib = RotamerLibrary {
            entries: std::collections::HashMap::new(),
        };
        let result = lib.place_rotamer(
            "GLY",
            Some(-60.0),
            Some(-45.0),
            0,
            false,
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        );
        assert!(result.is_ok());
        let placed = result.unwrap();
        assert_eq!(placed.id.aa, "GLY");
        assert!(placed.atoms.is_empty(), "GLY must return empty atoms vector");
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

        let result = lib.place_rotamer(
            "ALA",
            Some(-120.0),
            Some(-45.0),
            5,
            false,
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        );
        assert!(result.is_err());
        match result {
            Err(RotlibError::RotIndexOob(aa, idx, nr)) => {
                assert_eq!(aa, "ALA");
                assert_eq!(idx, 5);
                assert_eq!(nr, 1);
            }
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

        let n = [1.0, 0.0, 0.0];
        let ca = [0.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];

        let result = lib.place_rotamer("VAL", Some(-120.0), Some(-45.0), 0, false, n, ca, c);
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
        let pro_entry = AaEntry {
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

        let cpro_entry = AaEntry {
            atom_names: vec!["CB".to_string()],
            bin_phi_centers: vec![-100.0, -40.0, 80.0],
            bin_psi_centers: vec![-30.0, 0.0, 30.0, 60.0],
            default_bin: 1,
            rotamers: vec![
                BinData {
                    probs: vec![0.6],
                    coords: vec![vec![[4.0, 5.0, 6.0]]]
                };
                12
            ],
        };

        let mut entries = HashMap::new();
        entries.insert("PRO".to_string(), pro_entry.clone());
        entries.insert("CPR".to_string(), cpro_entry);
        let lib = RotamerLibrary { entries };

        // Test 1: cis_proline=true routes to CPR
        let result_cis = lib.backbone_bin("PRO", Some(-100.0), Some(-30.0), true);
        assert!(result_cis.is_ok());
        let bin_cis = result_cis.unwrap();
        assert_eq!(bin_cis, 0);

        // Test 2: cis_proline=false routes to standard PRO
        let result_trans = lib.backbone_bin("PRO", Some(-120.0), Some(-45.0), false);
        assert!(result_trans.is_ok());
        let bin_trans = result_trans.unwrap();
        assert_eq!(bin_trans, 0);

        // Test 3: cis_proline=true without CPR entry falls back and logs warning
        let mut entries_no_cpro = HashMap::new();
        entries_no_cpro.insert("PRO".to_string(), pro_entry);
        let lib_no_cpro = RotamerLibrary {
            entries: entries_no_cpro,
        };

        let result_fallback = lib_no_cpro.backbone_bin("PRO", Some(-120.0), Some(-45.0), true);
        assert!(result_fallback.is_ok());
        let bin_fallback = result_fallback.unwrap();
        assert_eq!(bin_fallback, 0);
    }
}
