use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::Path;
use crate::error::RotlibError;
use crate::rotamer_id::RotamerId;
use crate::binning::angle_to_standard;

/// Per-rotamer data within a single phi/psi bin.
#[derive(Clone)]
pub(crate) struct BinData {
    /// Rotamer probabilities (len = nr).
    pub(crate) probs:  Vec<f64>,
    /// Atom coordinates in canonical backbone-relative frame, indexed [rot_index][atom_index].
    pub(crate) coords: Vec<Vec<[f64; 3]>>,
}

/// Per amino-acid rotamer library entry.
#[derive(Clone)]
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

/// Backbone-dependent rotamer library loaded from an MSL binary file.
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
    phi_centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    psi_centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
                .max_by(|(_, (_, _, fa)), (_, (_, _, fb))|
                    fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

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

    pub fn contains_aa(&self, aa: &str) -> bool {
        self.entries.contains_key(aa)
    }

    pub fn num_rotamers(&self, aa: &str, phi: f64, psi: f64) -> Result<usize, RotlibError> {
        let entry = self.entries.get(aa).ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))?;
        let bin = self.backbone_bin(aa, phi, psi)? as usize;
        Ok(entry.rotamers[bin].probs.len())
    }

    pub fn rotamer_probability(&self, aa: &str, rot_index: usize, phi: f64, psi: f64) -> Result<f64, RotlibError> {
        let entry = self.entries.get(aa).ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))?;
        let bin = self.backbone_bin(aa, phi, psi)? as usize;
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

    pub fn place_rotamer(&self, aa: &str, phi: f64, psi: f64, rot_index: usize, n: [f64; 3], ca: [f64; 3], c: [f64; 3]) -> Result<crate::rotamer_id::PlacedRotamer, RotlibError> {
        use crate::frame::{backbone_frame, Frame, Transform};
        use crate::rotamer_id::{PlacedAtom, PlacedRotamer, RotamerId};

        let entry = self.entries.get(aa)
            .ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))?;
        let bin = self.backbone_bin(aa, phi, psi)? as usize;
        // bin is always valid (produced by backbone_bin which indexes entry.rotamers)
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

    pub fn sidechain_atom_names(&self, aa: &str) -> Result<&[String], RotlibError> {
        self.entries.get(aa)
            .map(|e| e.atom_names.as_slice())
            .ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))
    }

    pub fn backbone_bin(&self, aa: &str, phi: f64, psi: f64) -> Result<u32, RotlibError> {
        use crate::binning::find_closest_angle;
        let entry = self.entries.get(aa)
            .ok_or_else(|| RotlibError::UnknownAa(aa.to_string()))?;
        // Sentinel: either angle is 9999.0 → return default_bin
        if phi == 9999.0 || psi == 9999.0 {
            return Ok(entry.default_bin);
        }
        let phi_ind = find_closest_angle(&entry.bin_phi_centers, phi);
        let psi_ind = find_closest_angle(&entry.bin_psi_centers, psi);
        let n_psi = entry.bin_psi_centers.len();
        Ok((phi_ind * n_psi + psi_ind) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backbone_bin_unknown_aa() {
        let lib = RotamerLibrary { entries: std::collections::HashMap::new() };
        let result = lib.backbone_bin("UNK", -60.0, -45.0);
        assert!(result.is_err());
        match result {
            Err(RotlibError::UnknownAa(aa)) => assert_eq!(aa, "UNK"),
            _ => panic!("Expected UnknownAa error"),
        }
    }

    #[test]
    fn test_backbone_bin_sentinel_phi() {
        use std::collections::HashMap;
        let mut entry = AaEntry {
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

        let result = lib.backbone_bin("ALA", 9999.0, -45.0);
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

        let result = lib.backbone_bin("ALA", -60.0, 9999.0);
        assert_eq!(result.unwrap(), 7);
    }

    #[test]
    fn test_place_rotamer_unknown_aa() {
        let lib = RotamerLibrary { entries: std::collections::HashMap::new() };
        let result = lib.place_rotamer("UNK", -60.0, -45.0, 0, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
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

        let result = lib.place_rotamer("ALA", -120.0, -45.0, 5, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
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

        let result = lib.place_rotamer("VAL", -120.0, -45.0, 0, n, ca, c);
        assert!(result.is_ok());
        let placed = result.unwrap();

        assert_eq!(placed.id.aa, "VAL");
        assert_eq!(placed.id.bin_index, 0);
        assert_eq!(placed.id.rot_index, 0);
        assert_eq!(placed.atoms.len(), 2);
        assert_eq!(placed.atoms[0].name, "CB");
        assert_eq!(placed.atoms[1].name, "CG");
    }
}
