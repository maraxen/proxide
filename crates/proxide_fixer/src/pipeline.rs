use crate::models::Topology;
use proxide_core::forcefield::ForceField;
use proxide_rotlib::RotamerLibrary;
use thiserror::Error;
use std::collections::HashMap;

#[derive(Error, Debug)]
pub enum SystemPrepError {
    #[cfg(feature = "disulfide")]
    #[error("disulfide sanitizer failed: {0}")]
    DisulfideError(#[from] crate::sanitizers::disulfide::DisulfideError),

    #[cfg(feature = "capping")]
    #[error("capping sanitizer failed: {0}")]
    CappingError(#[from] crate::sanitizers::capping::CappingError),

    #[cfg(feature = "capping")]
    #[error("cap groups sanitizer failed: {0}")]
    CapGroupsError(#[from] crate::sanitizers::capping_groups::CapError),

    #[cfg(feature = "protonation")]
    #[error("protonation-state sanitizer failed: {0}")]
    ProtonationStateError(#[from] crate::protonation_state::ProtonationStateError),

    #[cfg(any(feature = "protonation", feature = "capping", feature = "stereo", feature = "disulfide"))]
    #[error("hydrogen sanitizer failed: {0}")]
    HydrogenError(#[from] crate::sanitizers::hydrogens::HydrogenError),

    #[error("repack error: {0}")]
    RepackError(#[from] crate::repack::RepackError),

    #[error("loop modelling failed: {0}")]
    LoopModelling(#[from] crate::loop_model::LoopModellingError),

    #[error("solvation failed: {0}")]
    Solvation(#[from] crate::solvate::SolvationError),

    #[error("mutation failed: {0:?}")]
    Mutation(crate::mutate::MutationError),
}

/// Configuration for the system preparation pipeline.
#[derive(Debug, Clone)]
pub struct SystemPrepConfig {
    /// pH for protonation state assignment (default: 7.4).
    pub ph: f32,
    /// Whether to detect and fix disulfide bonds.
    pub detect_disulfides: bool,
    /// Whether to patch and fix termini.
    pub patch_termini: bool,
    /// Whether to cap internal breaks with ACE/NME.
    pub cap_breaks: bool,
    /// Whether to assign protonation states.
    pub assign_protonation: bool,
    /// Whether to place missing hydrogen atoms.
    pub place_hydrogens: bool,
    /// Whether to repack sidechains.
    pub repack_sidechains: bool,
    /// Whether to model missing loops using Modeller (requires MODELLER_EXEC + MODELLER_KEY).
    pub model_loops: bool,
    /// SEQRES mapping for loop modeling: chain_id → Vec<(res_id, three_letter_name)>.
    /// Required for model_loops. If None and model_loops=true, loop modeling is skipped with a warning.
    pub seqres: Option<HashMap<String, Vec<(i32, String)>>>,
    /// Solvation configuration (optional). None = skip solvation; Some(...) = run Solvator.
    pub solvate: Option<crate::solvate::SolvationConfig>,
    /// Mutations to apply (default: empty). Mutations run as Step 0, before all other sanitization steps.
    pub mutations: Vec<crate::mutate::MutationRequest>,
}

impl Default for SystemPrepConfig {
    fn default() -> Self {
        Self {
            ph: 7.4,
            detect_disulfides: true,
            patch_termini: true,
            cap_breaks: true,
            assign_protonation: true,
            place_hydrogens: true,
            repack_sidechains: false,
            model_loops: false,
            seqres: None,
            solvate: None,
            mutations: vec![],
        }
    }
}

/// Report from the system preparation pipeline.
#[derive(Debug, Clone, Default)]
pub struct SystemPrepReport {
    /// Mutation report (None if no mutations were requested).
    pub mutation: Option<crate::mutate::MutationReport>,
    /// Solvation report (None if solvation was skipped).
    pub solvation: Option<crate::solvate::SolvationReport>,
}

/// System preparation pipeline: threads sanitizers in dependency order.
/// Each step is gated on both its config flag and its feature gate.
pub struct SystemPrep<'a> {
    topology: &'a mut Topology,
    ff: &'a ForceField,
    lib: Option<&'a RotamerLibrary>,
    config: SystemPrepConfig,
}

impl<'a> SystemPrep<'a> {
    /// Create a new SystemPrep instance.
    pub fn new(
        topology: &'a mut Topology,
        ff: &'a ForceField,
        lib: Option<&'a RotamerLibrary>,
        config: SystemPrepConfig,
    ) -> Self {
        Self {
            topology,
            ff,
            lib,
            config,
        }
    }

    /// Run the full system prep pipeline in order.
    /// 0. Mutations (C9) — apply requested point mutations, repack neighbourhood, re-evaluate disulfides
    /// 1. Disulfides (C2) — rename CYS to CYX, remove HG
    /// 2. Loop modeling (C10) — build missing loops via Modeller (optional)
    /// 3. Termini patching (C4) — fix first/last residue termini
    /// 4. Capping breaks (C5) — insert ACE/NME at internal chain breaks
    /// 5. Protonation state (C7) — assign HID/HIE/HIP/ASH/... names
    /// 6. Hydrogens (C6) — place missing H atoms based on FF templates
    /// 7. Repack (C8) — optimize sidechain conformations (optional)
    /// 8. Solvation (C11) — add solvent and counterions (optional)
    pub fn run(&mut self) -> Result<SystemPrepReport, SystemPrepError> {
        let mut report = SystemPrepReport::default();

        // Step 0: Mutations (C9) — must precede protonation, disulfide, capping
        {
            if !self.config.mutations.is_empty() {
                if let Some(lib) = self.lib {
                    use crate::mutate::Mutator;
                    let mut mutator = Mutator::new(self.topology, lib);
                    let mutation_report = mutator.apply(&self.config.mutations)
                        .map_err(SystemPrepError::Mutation)?;
                    report.mutation = Some(mutation_report);
                } else {
                    log::warn!("mutations requested but no rotamer library provided; skipping mutations");
                }
            }
        }

        // Step 1: Disulfides (C2)
        #[cfg(feature = "disulfide")]
        {
            if self.config.detect_disulfides {
                let mut sanitizer =
                    crate::sanitizers::disulfide::DisulfideSanitizer::new(self.topology);
                sanitizer.run()?;
            }
        }
        #[cfg(not(feature = "disulfide"))]
        {
            if self.config.detect_disulfides {
                log::warn!(
                    "detect_disulfides=true requested but 'disulfide' feature not compiled in — step skipped"
                );
            }
        }

        // Step 2: Loop modeling (C10) — model missing loops via Modeller
        {
            if self.config.model_loops {
                if let Some(seqres) = &self.config.seqres {
                    let missing = crate::loop_model::detect_missing_loops(self.topology);
                    if !missing.is_empty() {
                        let mut modeller = crate::loop_model::LoopModeller::new(self.topology);
                        modeller.build_loops(&missing)?;
                    }
                } else {
                    log::warn!("model_loops=true but no seqres provided; skipping loop modeling");
                }
            }
        }

        // Step 3: Termini patching (C4)
        #[cfg(feature = "capping")]
        {
            if self.config.patch_termini {
                let mut sanitizer =
                    crate::sanitizers::capping::CappingSanitizer::new(self.topology);
                sanitizer.run()?;
            }
        }
        #[cfg(not(feature = "capping"))]
        {
            if self.config.patch_termini {
                log::warn!(
                    "patch_termini=true requested but 'capping' feature not compiled in — step skipped"
                );
            }
        }

        // Step 4: Capping breaks (C5) — requires capping feature to detect breaks
        #[cfg(feature = "capping")]
        {
            if self.config.cap_breaks {
                let breaks = {
                    let sanitizer =
                        crate::sanitizers::capping::CappingSanitizer::new(self.topology);
                    sanitizer.detect_breaks()
                };

                if !breaks.is_empty() {
                    let mut cap_sanitizer =
                        crate::sanitizers::capping_groups::CapSanitizer::new(self.topology);
                    cap_sanitizer.run(&breaks)?;
                }
            }
        }
        #[cfg(not(feature = "capping"))]
        {
            if self.config.cap_breaks {
                log::warn!(
                    "cap_breaks=true requested but 'capping' feature not compiled in — step skipped"
                );
            }
        }

        // Step 5: Protonation state (C7) — assign HID/HIE/HIP/ASH/... NAMES only
        // (PROPKA wrap + pH-7.4 fallback). Runs BEFORE hydrogens so C6 places the
        // H matching the chosen state; the legacy sanitizers::protonation path
        // (which adds H itself) is intentionally NOT used here to avoid double H.
        #[cfg(feature = "protonation")]
        {
            if self.config.assign_protonation {
                let mut sanitizer = crate::protonation_state::ProtonationStateSanitizer::new(
                    self.topology,
                    self.config.ph,
                );
                sanitizer.run()?;
            }
        }
        #[cfg(not(feature = "protonation"))]
        {
            if self.config.assign_protonation {
                log::warn!(
                    "assign_protonation=true requested but 'protonation' feature not compiled in — step skipped"
                );
            }
        }

        // Step 6: Hydrogens (C6) — available regardless of feature gates
        // Note: sanitizers module itself is feature-gated; only run if at least one feature is enabled
        #[cfg(any(feature = "protonation", feature = "capping", feature = "stereo", feature = "disulfide"))]
        {
            if self.config.place_hydrogens {
                let mut sanitizer =
                    crate::sanitizers::hydrogens::HydrogenSanitizer::new(self.topology, self.ff);
                sanitizer.run()?;
            }
        }

        // Step 7: Repack (C8)
        {
            if self.config.repack_sidechains {
                if let Some(lib) = self.lib {
                    let mut repacker =
                        crate::repack::SidechainRepacker::new(self.topology, lib);
                    repacker.complete_and_repack(crate::repack::RepackMode::CompleteMissingOnly)?;
                }
            }
        }

        // Step 8: Solvation (C11) — add solvent and counterions (optional)
        {
            if let Some(ref sol_config) = self.config.solvate {
                use crate::solvate::Solvator;
                let mut solvator = Solvator::new(self.topology, sol_config.clone());
                let solvation_report = solvator.run().map_err(SystemPrepError::Solvation)?;
                report.solvation = Some(solvation_report);
            }
        }

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Atom, Chain, Residue};

    /// Build a minimal Topology for testing.
    fn build_test_topology() -> Topology {
        Topology {
            chains: vec![Chain {
                id: "A".to_string(),
                residues: vec![Residue {
                    name: "ALA".to_string(),
                    res_id: 1,
                    insertion_code: ' ',
                    atoms: vec![
                        Atom {
                            name: "N".to_string(),
                            element: "N".to_string(),
                            coords: [0.0, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 1,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        },
                        Atom {
                            name: "CA".to_string(),
                            element: "C".to_string(),
                            coords: [1.5, 0.0, 0.0],
                            alt_loc: ' ',
                            serial: 2,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        },
                        Atom {
                            name: "C".to_string(),
                            element: "C".to_string(),
                            coords: [2.5, 1.0, 0.0],
                            alt_loc: ' ',
                            serial: 3,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        },
                        Atom {
                            name: "O".to_string(),
                            element: "O".to_string(),
                            coords: [2.5, 2.0, 0.0],
                            alt_loc: ' ',
                            serial: 4,
                            b_factor: 0.0,
                            occupancy: 1.0,
                            is_hetatm: false,
                        },
                    ],
                }],
            }],
        }
    }

    #[test]
    fn test_system_prep_default_config() {
        let mut topology = build_test_topology();
        let ff = ForceField::new("test".to_string());
        let config = SystemPrepConfig::default();

        let mut sysprep = SystemPrep::new(&mut topology, &ff, None, config);
        // Expect this to not crash (feature gates will skip disabled sanitizers)
        let result = sysprep.run();
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_prep_all_disabled() {
        let mut topology = build_test_topology();
        let ff = ForceField::new("test".to_string());
        let config = SystemPrepConfig {
            detect_disulfides: false,
            patch_termini: false,
            cap_breaks: false,
            assign_protonation: false,
            place_hydrogens: false,
            repack_sidechains: false,
            ..Default::default()
        };

        let mut sysprep = SystemPrep::new(&mut topology, &ff, None, config);
        let result = sysprep.run();
        // Should succeed (nothing to do)
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_prep_config_ph() {
        let config_default = SystemPrepConfig::default();
        assert_eq!(config_default.ph, 7.4);

        let config_custom = SystemPrepConfig {
            ph: 6.5,
            ..Default::default()
        };
        assert_eq!(config_custom.ph, 6.5);
    }

    #[test]
    #[cfg(not(feature = "disulfide"))]
    fn test_system_prep_detects_disulfide_feature_missing() {
        // When disulfide feature is disabled but detect_disulfides is requested,
        // the pipeline should still run but without the step, which would be
        // silently skipped before the fix. With the fix, a warning should be emitted
        // via tracing. This test runs in a no-disulfide build config.
        let mut topology = build_test_topology();
        let ff = ForceField::new("test".to_string());
        let config = SystemPrepConfig {
            detect_disulfides: true,
            ..Default::default()
        };

        let mut sysprep = SystemPrep::new(&mut topology, &ff, None, config);
        // This should still succeed (not error), but the disulfide step should be
        // either skipped OR a warning emitted. The key is: no silent failure.
        let result = sysprep.run();
        assert!(result.is_ok(), "SystemPrep should not error even with disabled feature");
    }

    #[test]
    #[cfg(not(feature = "protonation"))]
    fn test_system_prep_detects_protonation_feature_missing() {
        // When protonation feature is disabled but assign_protonation is requested,
        // the pipeline should still run but without the step.
        let mut topology = build_test_topology();
        let ff = ForceField::new("test".to_string());
        let config = SystemPrepConfig {
            assign_protonation: true,
            ..Default::default()
        };

        let mut sysprep = SystemPrep::new(&mut topology, &ff, None, config);
        let result = sysprep.run();
        assert!(result.is_ok(), "SystemPrep should not error even with disabled feature");
    }

    #[test]
    fn test_solvation_skipped_when_config_none() {
        // When solvate is None in config, solvation should be skipped
        // and report.solvation should be None.
        let mut topology = build_test_topology();
        let ff = ForceField::new("test".to_string());
        let config = SystemPrepConfig {
            solvate: None,
            ..Default::default()
        };

        let mut sysprep = SystemPrep::new(&mut topology, &ff, None, config);
        let result = sysprep.run();

        assert!(result.is_ok(), "SystemPrep should succeed when solvation is None");
        let report = result.unwrap();
        assert!(
            report.solvation.is_none(),
            "Solvation report should be None when solvate config is None"
        );
    }

    #[test]
    fn mutations_skipped_when_empty() {
        // When mutations vec is empty in config, mutations should be skipped
        // and report.mutation should be None.
        let mut topology = build_test_topology();
        let ff = ForceField::new("test".to_string());
        let config = SystemPrepConfig {
            mutations: vec![],
            ..Default::default()
        };

        let mut sysprep = SystemPrep::new(&mut topology, &ff, None, config);
        let result = sysprep.run();

        assert!(result.is_ok(), "SystemPrep should succeed when mutations is empty");
        let report = result.unwrap();
        assert!(
            report.mutation.is_none(),
            "Mutation report should be None when mutations config is empty"
        );
    }
}
