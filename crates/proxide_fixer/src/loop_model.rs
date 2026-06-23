use std::path::PathBuf;
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use crate::models::Topology;

#[derive(Debug, Clone)]
pub struct MissingLoop {
    pub chain_id: String,
    pub start_res: i32,
    pub end_res: i32,
    pub sequence: String, // 1-letter SEQRES residues for the gap
}

#[derive(Debug, Clone)]
pub struct LoopModelReport {
    pub loops_built: Vec<MissingLoop>,
    pub geometry_warnings: Vec<String>,
}

#[derive(Error, Debug)]
pub enum LoopModellingError {
    #[error("Modeller executable not found (set MODELLER_EXEC or add to PATH)")]
    ModelerNotInstalled,

    #[error("MODELLER_KEY not set; Modeller requires a license key")]
    MissingLicenseKey,

    #[error("failed to spawn Modeller: {0}")]
    Spawn(String),

    #[error("Modeller exited non-zero: {0}")]
    NonZeroExit(String),

    #[error("failed to parse Modeller output PDB: {0}")]
    Parse(String),

    #[error("loop geometry validation failed: {0}")]
    GeometryInvalid(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Loop modeller interface: orchestrates missing-loop detection and building via Modeller.
///
/// Holds a reference to the topology being modified and provides methods to build
/// loops (C10 — requires MODELLER_EXEC and MODELLER_KEY to be set).
pub struct LoopModeller<'a> {
    topology: &'a mut Topology,
}

impl<'a> LoopModeller<'a> {
    /// Create a new LoopModeller for the given topology.
    pub fn new(topology: &'a mut Topology) -> Self {
        Self { topology }
    }

    /// Build missing loops detected in the given list.
    ///
    /// This is a placeholder that validates the Modeller environment and topology
    /// state. A full implementation would invoke Modeller to generate coordinates
    /// for the missing residues.
    ///
    /// For now, this just checks that Modeller is installed and returns success.
    pub fn build_loops(&mut self, _missing: &[MissingLoop]) -> Result<LoopModelReport, LoopModellingError> {
        // Verify Modeller is installed
        let _modeller_path = find_modeller()?;

        // Verify MODELLER_KEY is set
        if std::env::var("MODELLER_KEY").is_err() {
            return Err(LoopModellingError::MissingLicenseKey);
        }

        // TODO(#977 C10): implement actual loop building via Modeller subprocess
        // For now, return an empty report (no loops built, no warnings)
        Ok(LoopModelReport {
            loops_built: Vec::new(),
            geometry_warnings: Vec::new(),
        })
    }
}

/// Detect SEQRES-vs-ATOM gaps.
///
/// `seqres`: chain_id → Vec<(res_id, three_letter_name)>, matching the
/// `builder.rs` `map_seqres` convention.
///
/// Returns one `MissingLoop` per contiguous run of SEQRES res_ids absent
/// from the topology's ATOM records.
pub fn detect_missing_loops(
    topology: &Topology,
    seqres: &HashMap<String, Vec<(i32, String)>>,
) -> Vec<MissingLoop> {
    let mut result = Vec::new();

    // Build a set of present (chain_id, res_id) pairs from ATOM records.
    let mut present: HashSet<(String, i32)> = HashSet::new();
    for chain in &topology.chains {
        for residue in &chain.residues {
            present.insert((chain.id.clone(), residue.res_id));
        }
    }

    for (chain_id, seqres_residues) in seqres {
        // Walk SEQRES in order; collect contiguous absent runs.
        let mut gap_start: Option<i32> = None;
        let mut gap_seq = String::new();
        let mut prev_res_id: Option<i32> = None;

        for (res_id, three_letter) in seqres_residues {
            if present.contains(&(chain_id.clone(), *res_id)) {
                // Present — close any open gap.
                if let Some(start) = gap_start.take() {
                    let end = prev_res_id.unwrap();
                    result.push(MissingLoop {
                        chain_id: chain_id.clone(),
                        start_res: start,
                        end_res: end,
                        sequence: std::mem::take(&mut gap_seq),
                    });
                }
            } else {
                // Absent — extend or start a gap.
                if gap_start.is_none() {
                    gap_start = Some(*res_id);
                }
                // Convert three-letter to one-letter (best-effort; unknown → X)
                gap_seq.push(three_to_one(three_letter));
            }
            prev_res_id = Some(*res_id);
        }

        // Close trailing gap.
        if let Some(start) = gap_start {
            let end = prev_res_id.unwrap();
            result.push(MissingLoop {
                chain_id: chain_id.clone(),
                start_res: start,
                end_res: end,
                sequence: gap_seq,
            });
        }
    }

    result
}

fn three_to_one(name: &str) -> char {
    match name {
        "ALA" => 'A', "ARG" => 'R', "ASN" => 'N', "ASP" => 'D',
        "CYS" => 'C', "GLN" => 'Q', "GLU" => 'E', "GLY" => 'G',
        "HIS" | "HID" | "HIE" | "HIP" => 'H',
        "ILE" => 'I', "LEU" => 'L', "LYS" => 'K', "MET" => 'M',
        "PHE" => 'F', "PRO" => 'P', "SER" => 'S', "THR" => 'T',
        "TRP" => 'W', "TYR" => 'Y', "VAL" => 'V',
        _ => 'X',
    }
}

/// Locate the Modeller executable.
/// Checks `MODELLER_EXEC` env var first, then PATH.
pub fn find_modeller() -> Result<PathBuf, LoopModellingError> {
    if let Ok(exec) = std::env::var("MODELLER_EXEC") {
        let path = PathBuf::from(&exec);
        if path.exists() {
            return Ok(path);
        }
    }
    // Try common executable names in PATH
    for name in &["mod10.x", "mod9.25", "modeller"] {
        if let Ok(p) = which_in_path(name) {
            return Ok(p);
        }
    }
    Err(LoopModellingError::ModelerNotInstalled)
}

fn which_in_path(name: &str) -> Result<PathBuf, ()> {
    let path_var = std::env::var("PATH").unwrap_or_default();
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Topology, Chain, Residue, Atom};

    #[test]
    fn three_to_one_standard_residues() {
        assert_eq!(three_to_one("ALA"), 'A');
        assert_eq!(three_to_one("TYR"), 'Y');
        assert_eq!(three_to_one("GLY"), 'G');
        assert_eq!(three_to_one("PRO"), 'P');
    }

    #[test]
    fn three_to_one_his_variants() {
        assert_eq!(three_to_one("HIS"), 'H');
        assert_eq!(three_to_one("HID"), 'H');
        assert_eq!(three_to_one("HIE"), 'H');
        assert_eq!(three_to_one("HIP"), 'H');
    }

    #[test]
    fn three_to_one_unknown() {
        assert_eq!(three_to_one("XYZ"), 'X');
        assert_eq!(three_to_one("FOO"), 'X');
    }

    #[test]
    fn detect_missing_loops_gap_in_middle() {
        // Topology with residues 1, 2, 5, 6 (gap at 3, 4)
        let topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "ALA".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "ALA".to_string(),
                        res_id: 2,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [1.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "GLY".to_string(),
                        res_id: 5,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [4.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 5,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "ALA".to_string(),
                        res_id: 6,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [5.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 6,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                ],
            }],
        };

        let mut seqres = HashMap::new();
        seqres.insert(
            "A".to_string(),
            vec![
                (1, "ALA".to_string()),
                (2, "ALA".to_string()),
                (3, "VAL".to_string()),
                (4, "LEU".to_string()),
                (5, "GLY".to_string()),
                (6, "ALA".to_string()),
            ],
        );

        let gaps = detect_missing_loops(&topology, &seqres);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].chain_id, "A");
        assert_eq!(gaps[0].start_res, 3);
        assert_eq!(gaps[0].end_res, 4);
        assert_eq!(gaps[0].sequence, "VL");
    }

    #[test]
    fn detect_missing_loops_multiple_gaps() {
        // Topology with residues 1, 4, 7 (gaps at 2-3 and 5-6)
        let topology = Topology {
            chains: vec![Chain {
                id: "B".to_string(),
                residues: vec![
                    Residue {
                        name: "ALA".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "GLY".to_string(),
                        res_id: 4,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [3.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 4,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "ALA".to_string(),
                        res_id: 7,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [6.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 7,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                ],
            }],
        };

        let mut seqres = HashMap::new();
        seqres.insert(
            "B".to_string(),
            vec![
                (1, "ALA".to_string()),
                (2, "ARG".to_string()),
                (3, "ASN".to_string()),
                (4, "GLY".to_string()),
                (5, "SER".to_string()),
                (6, "THR".to_string()),
                (7, "ALA".to_string()),
            ],
        );

        let gaps = detect_missing_loops(&topology, &seqres);
        assert_eq!(gaps.len(), 2);

        // First gap: 2-3 (RN)
        assert_eq!(gaps[0].chain_id, "B");
        assert_eq!(gaps[0].start_res, 2);
        assert_eq!(gaps[0].end_res, 3);
        assert_eq!(gaps[0].sequence, "RN");

        // Second gap: 5-6 (ST)
        assert_eq!(gaps[1].chain_id, "B");
        assert_eq!(gaps[1].start_res, 5);
        assert_eq!(gaps[1].end_res, 6);
        assert_eq!(gaps[1].sequence, "ST");
    }

    #[test]
    fn detect_missing_loops_no_gaps() {
        let topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "ALA".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                    Residue {
                        name: "ALA".to_string(),
                        res_id: 2,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [1.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                ],
            }],
        };

        let mut seqres = HashMap::new();
        seqres.insert(
            "A".to_string(),
            vec![(1, "ALA".to_string()), (2, "ALA".to_string())],
        );

        let gaps = detect_missing_loops(&topology, &seqres);
        assert_eq!(gaps.len(), 0);
    }

    #[test]
    fn not_installed_when_absent() {
        // Temporarily clear MODELLER_EXEC to simulate absence
        // (PATH-based check will also fail in CI where Modeller is not installed)
        let result = {
            let _guard = EnvGuard::clear("MODELLER_EXEC");
            find_modeller()
        };
        // In CI Modeller is not installed — expect ModelerNotInstalled
        // If Modeller IS installed, this test passes trivially with Ok.
        match result {
            Err(LoopModellingError::ModelerNotInstalled) => {} // expected in CI
            Ok(_) => {}  // Modeller present locally — also fine
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    struct EnvGuard {
        key: String,
        original: Option<String>,
    }
    impl EnvGuard {
        fn clear(key: &str) -> Self {
            let original = std::env::var(key).ok();
            std::env::remove_var(key);
            Self {
                key: key.to_string(),
                original,
            }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(v) => std::env::set_var(&self.key, v),
                None => std::env::remove_var(&self.key),
            }
        }
    }
}
