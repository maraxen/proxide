use crate::models::Topology;
use crate::propka_wrap::{self, PropkaError};
use log::warn;
use thiserror::Error;

/// Error type for protonation state assignment.
#[derive(Error, Debug)]
pub enum ProtonationStateError {
    #[error("propka error: {0}")]
    Propka(#[from] PropkaError),

    #[error("IO error during temp file handling: {0}")]
    Io(#[from] std::io::Error),
}

/// Report of protonation state changes applied.
#[derive(Debug, Clone)]
pub struct ProtonationReport {
    /// List of (chain_id, res_id, new_res_name) for renamed residues
    pub renamed: Vec<(String, i32, String)>,
    /// Whether PROPKA3 was actually used (vs. fallback)
    pub used_propka: bool,
}

/// Sanitizer for pH-dependent protonation state assignment.
///
/// Wraps PROPKA3 (if available) to assign AMBER-style protonation state residue names.
/// Falls back to fixed pH-7.4 rules if PROPKA3 is not installed.
pub struct ProtonationStateSanitizer<'a> {
    topology: &'a mut Topology,
    ph: f32,
}

impl<'a> ProtonationStateSanitizer<'a> {
    /// Create a new protonation state sanitizer at the given pH.
    pub fn new(topology: &'a mut Topology, ph: f32) -> Self {
        Self { topology, ph }
    }

    /// Run protonation state assignment.
    ///
    /// Writes topology to a temp PDB, calls PROPKA3 if available,
    /// or falls back to fixed pH-7.4 rules. Returns a report of changes.
    pub fn run(&mut self) -> Result<ProtonationReport, ProtonationStateError> {
        // Write topology to temp PDB
        let temp_pdb = self.write_temp_pdb()?;
        let temp_pdb_path = std::path::Path::new(&temp_pdb);

        // Try PROPKA3
        match propka_wrap::run_propka(temp_pdb_path, self.ph) {
            Ok(pka_table) => {
                // PROPKA3 succeeded; apply pH-based assignment
                let renamed = self.assign_by_pka(&pka_table)?;

                // Clean up temp file
                let _ = std::fs::remove_file(temp_pdb_path);

                Ok(ProtonationReport {
                    renamed,
                    used_propka: true,
                })
            }
            Err(PropkaError::NotInstalled) => {
                // PROPKA3 not on PATH; fall back to fixed pH-7.4 rules
                warn!("PROPKA3 not found; using fixed pH-7.4 protonation rules");

                let renamed = self.assign_fixed_ph_7_4()?;

                // Clean up temp file
                let _ = std::fs::remove_file(temp_pdb_path);

                Ok(ProtonationReport {
                    renamed,
                    used_propka: false,
                })
            }
            Err(e) => {
                // PROPKA3 was found but failed; hard error
                let _ = std::fs::remove_file(temp_pdb_path);
                Err(ProtonationStateError::Propka(e))
            }
        }
    }

    /// Write topology to a minimal fixed-column PDB file.
    fn write_temp_pdb(&self) -> Result<String, std::io::Error> {
        use std::io::Write;

        use std::sync::atomic::{AtomicU64, Ordering};
        static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!(
            "proxide_temp_{}_{}.pdb",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));

        let mut file = std::fs::File::create(&temp_file)?;

        // Write minimal PDB header
        writeln!(file, "REMARK  Temporary PDB for PROPKA3 pKa prediction")?;

        let mut serial = 1i32;
        for chain in &self.topology.chains {
            for residue in &chain.residues {
                for atom in &residue.atoms {
                    // Fixed-column PDB ATOM line format
                    let record = if atom.is_hetatm { "HETATM" } else { "ATOM  " };
                    let atom_name = format!("{:>4}", atom.name); // Right-aligned, 4 chars
                    let res_name = format!("{:<3}", residue.name); // Left-aligned, 3 chars
                    let chain_id = &chain.id;
                    let res_id = residue.res_id;
                    let x = atom.coords[0];
                    let y = atom.coords[1];
                    let z = atom.coords[2];
                    let occupancy = atom.occupancy;
                    let b_factor = atom.b_factor;
                    let element = format!("{:>2}", atom.element); // Right-aligned, 2 chars

                    // PDB format: columns 1-6, 7-11, 13-16, 17, 18-20, 22, 23-26, 27, 31-38, 39-46, 47-54, 55-60, 61-66, 77-78
                    // We use a simplified approach: write the essentials
                    writeln!(
                        file,
                        "{:<6}{:>5} {:>4}{}{:<3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}",
                        record,
                        serial,
                        atom_name,
                        atom.alt_loc,
                        res_name,
                        chain_id,
                        res_id,
                        x,
                        y,
                        z,
                        occupancy,
                        b_factor,
                        element
                    )?;

                    serial = serial.saturating_add(1);
                }
            }
            writeln!(file, "TER")?;
        }
        writeln!(file, "END")?;

        Ok(temp_file.to_string_lossy().to_string())
    }

    /// Assign protonation states based on pKa table.
    /// Residue is protonated if pH < pKa.
    fn assign_by_pka(
        &mut self,
        pka_table: &propka_wrap::PkaTable,
    ) -> Result<Vec<(String, i32, String)>, ProtonationStateError> {
        let mut renamed = Vec::new();

        for chain in &mut self.topology.chains {
            for residue in &mut chain.residues {
                if let Some(&pka) = pka_table.get(&(chain.id.clone(), residue.res_id)) {
                    let protonated = self.ph < pka;
                    let new_name = match residue.name.as_str() {
                        "HIS" => {
                            // pH < pKa => doubly-protonated HIP (+1). Neutral HIS defaults
                            // to HIE (Nepsilon2-H, the more common neutral tautomer).
                            // HID-vs-HIE selection needs PROPKA H-bond determinant analysis
                            // (tracked as a follow-up).
                            if protonated {
                                "HIP".to_string()
                            } else {
                                "HIE".to_string()
                            }
                        }
                        "ASP" => {
                            if protonated {
                                "ASH".to_string() // Protonated aspartic acid
                            } else {
                                "ASP".to_string()
                            }
                        }
                        "GLU" => {
                            if protonated {
                                "GLH".to_string() // Protonated glutamic acid
                            } else {
                                "GLU".to_string()
                            }
                        }
                        "LYS" => {
                            if protonated {
                                "LYS".to_string()
                            } else {
                                "LYN".to_string() // Neutral lysine
                            }
                        }
                        "CYS" => {
                            if protonated {
                                "CYS".to_string()
                            } else {
                                "CYM".to_string() // Deprotonated cysteine
                            }
                        }
                        "TYR" => {
                            if protonated {
                                "TYR".to_string()
                            } else {
                                "TYM".to_string() // Deprotonated tyrosine
                            }
                        }
                        "ARG" => "ARG".to_string(), // Always protonated
                        other => other.to_string(), // Keep unchanged
                    };

                    if new_name != residue.name {
                        renamed.push((chain.id.clone(), residue.res_id, new_name.clone()));
                        residue.name = new_name;
                    }
                }
            }
        }

        Ok(renamed)
    }

    /// Assign protonation states using fixed pH-7.4 rules.
    fn assign_fixed_ph_7_4(&mut self) -> Result<Vec<(String, i32, String)>, ProtonationStateError> {
        let mut renamed = Vec::new();

        for chain in &mut self.topology.chains {
            for residue in &mut chain.residues {
                let new_name = match residue.name.as_str() {
                    "HIS" => "HIE".to_string(), // Default to HIE at neutral pH
                    "ASP" => "ASP".to_string(), // Deprotonated at pH 7.4 (pKa ~3.9)
                    "GLU" => "GLU".to_string(), // Deprotonated at pH 7.4 (pKa ~4.2)
                    "LYS" => "LYS".to_string(), // Protonated at pH 7.4 (pKa ~10.5)
                    "CYS" => "CYS".to_string(), // Neutral (protonated) at pH 7.4 (pKa ~8.3)
                    "TYR" => "TYR".to_string(), // Neutral (protonated) at pH 7.4 (pKa ~10.1)
                    "ARG" => "ARG".to_string(), // Always protonated
                    other => other.to_string(), // Keep unchanged
                };

                if new_name != residue.name {
                    renamed.push((chain.id.clone(), residue.res_id, new_name.clone()));
                    residue.name = new_name;
                }
            }
        }

        Ok(renamed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Atom, Chain, Residue};

    fn make_test_topology() -> Topology {
        Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "HIS".to_string(),
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
                        name: "ASP".to_string(),
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
                        name: "LYS".to_string(),
                        res_id: 3,
                        insertion_code: ' ',
                        atoms: vec![Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [2.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 3,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        }],
                    },
                ],
            }],
        }
    }

    #[test]
    fn test_protonation_state_fallback_no_propka() {
        // Test: PROPKA3 not on PATH -> fallback to fixed pH-7.4 rules
        // NO panic, NO error. Should succeed and return used_propka=false.
        let mut topology = make_test_topology();
        let mut sanitizer = ProtonationStateSanitizer::new(&mut topology, 7.4);

        let result = sanitizer.run();

        // Should succeed (fallback)
        let report = result.expect("fallback should not error");
        assert_eq!(
            report.used_propka, false,
            "Should use fallback, not PROPKA3"
        );

        // Verify state changes were applied
        // HIS -> HIE, ASP stays ASP (deprot), LYS stays LYS (prot)
        let his_res = &sanitizer.topology.chains[0].residues[0];
        assert_eq!(his_res.name, "HIE", "HIS should become HIE");

        let asp_res = &sanitizer.topology.chains[0].residues[1];
        assert_eq!(asp_res.name, "ASP", "ASP should stay ASP");

        let lys_res = &sanitizer.topology.chains[0].residues[2];
        assert_eq!(lys_res.name, "LYS", "LYS should stay LYS");
    }

    #[test]
    fn test_fixed_ph_7_4_rules() {
        // Test the fixed pH-7.4 assignment logic directly
        let mut topology = make_test_topology();
        let mut sanitizer = ProtonationStateSanitizer::new(&mut topology, 7.4);

        let renamed = sanitizer.assign_fixed_ph_7_4().expect("assign failed");

        // HIS should be renamed to HIE
        assert_eq!(
            renamed
                .iter()
                .find(|r| r.0 == "A" && r.1 == 1)
                .map(|r| &r.2),
            Some(&"HIE".to_string()),
            "HIS at res 1 should map to HIE"
        );

        // ASP should stay ASP (but renamed = false in this case)
        // LYS should stay LYS
    }

    #[test]
    fn test_protonation_state_run_no_panic() {
        // Minimal test: just ensure run() doesn't panic or error fatally
        // when PROPKA3 is missing (which it will be in CI).
        let mut topology = make_test_topology();
        let mut sanitizer = ProtonationStateSanitizer::new(&mut topology, 7.4);

        // This should NOT panic, even if PROPKA3 is not installed
        let result = sanitizer.run();

        // Result should be Ok (fallback applied)
        assert!(result.is_ok(), "Should succeed with fallback");
    }

    #[test]
    fn test_protonation_state_canonicalize_his() {
        let mut topology = Topology {
            chains: vec![Chain {
                id: "X".to_string(),
                residues: vec![Residue {
                    name: "HIS".to_string(),
                    res_id: 42,
                    insertion_code: ' ',
                    atoms: vec![],
                }],
            }],
        };

        let mut sanitizer = ProtonationStateSanitizer::new(&mut topology, 7.4);
        let renamed = sanitizer.assign_fixed_ph_7_4().expect("assign failed");

        assert!(
            renamed
                .iter()
                .any(|r| r.0 == "X" && r.1 == 42 && r.2 == "HIE"),
            "HIS should be canonicalized to HIE"
        );
    }

    #[test]
    fn test_assign_by_pka_his_tautomers() {
        let mut topology = Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![
                    Residue {
                        name: "HIS".to_string(),
                        res_id: 1,
                        insertion_code: ' ',
                        atoms: vec![],
                    },
                    Residue {
                        name: "HIS".to_string(),
                        res_id: 2,
                        insertion_code: ' ',
                        atoms: vec![],
                    },
                ],
            }],
        };
        let mut table: crate::propka_wrap::PkaTable = std::collections::HashMap::new();
        table.insert(("A".to_string(), 1), 8.0); // pKa 8.0 > pH 7.4 => protonated => HIP
        table.insert(("A".to_string(), 2), 5.0); // pKa 5.0 < pH 7.4 => neutral  => HIE
        let mut sanitizer = ProtonationStateSanitizer::new(&mut topology, 7.4);
        let renamed = sanitizer
            .assign_by_pka(&table)
            .expect("assign_by_pka failed");
        assert!(
            renamed.iter().any(|r| r.1 == 1 && r.2 == "HIP"),
            "pH<pKa HIS -> HIP"
        );
        assert!(
            renamed.iter().any(|r| r.1 == 2 && r.2 == "HIE"),
            "pH>pKa HIS -> HIE"
        );
    }
}
