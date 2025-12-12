//! MD Parameterization - Assigns force field parameters to structures
//!
//! This module uses parsed force field data to assign charges, LJ parameters,
//! and GBSA radii to atoms in a ProcessedStructure.

use std::collections::HashMap;
use thiserror::Error;

use crate::forcefield::{ForceField, GBSAOBCParam, NonbondedParam, ResidueTemplate};
use crate::processing::ProcessedStructure;

/// Errors during parameterization
#[derive(Error, Debug)]
pub enum ParamError {
    #[error("Missing residue template: {0}")]
    MissingTemplate(String),

    #[error("Missing atom in template: residue={0}, atom={1}")]
    MissingAtom(String, String),

    #[error("Missing nonbonded params for atom type: {0}")]
    MissingNonbonded(String),
}

/// How to handle missing residue templates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingResidueMode {
    /// Skip residue and log warning (default)
    #[default]
    SkipWarn,
    /// Fail with error
    Fail,
    /// Try GAFF fallback (future - not implemented)
    GaffFallback,
    /// Match closest residue by shared atom names
    ClosestMatch,
}

/// MD parameters assigned to a structure
#[derive(Debug, Clone)]
pub struct MDParameters {
    /// Partial charges per atom (elementary charge units)
    pub charges: Vec<f32>,
    /// LJ sigma per atom (nm)
    pub sigmas: Vec<f32>,
    /// LJ epsilon per atom (kJ/mol)
    pub epsilons: Vec<f32>,
    /// GBSA radius per atom (nm) - None if GBSA not available
    pub radii: Option<Vec<f32>>,
    /// GBSA scaling factor per atom - None if GBSA not available
    pub scales: Option<Vec<f32>>,
    /// Atom type name per atom
    pub atom_types: Vec<String>,
    /// Number of atoms that were successfully parameterized
    pub num_parameterized: usize,
    /// Number of atoms that were skipped
    pub num_skipped: usize,
}

/// Options for parameterization
#[derive(Debug, Clone)]
pub struct ParamOptions {
    /// Auto-detect terminal residue variants (NALA, CALA, etc.)
    pub auto_terminal_caps: bool,
    /// How to handle missing residue templates
    pub missing_mode: MissingResidueMode,
}

impl Default for ParamOptions {
    fn default() -> Self {
        Self {
            auto_terminal_caps: true,
            missing_mode: MissingResidueMode::SkipWarn,
        }
    }
}

/// Parameterize a structure using force field templates
pub fn parameterize_structure(
    processed: &ProcessedStructure,
    ff: &ForceField,
    options: &ParamOptions,
) -> Result<MDParameters, ParamError> {
    let n_atoms = processed.raw_atoms.num_atoms;
    let n_residues = processed.num_residues;

    // Initialize output arrays
    let mut charges = vec![0.0f32; n_atoms];
    let mut sigmas = vec![0.0f32; n_atoms];
    let mut epsilons = vec![0.0f32; n_atoms];
    let mut atom_types = vec![String::new(); n_atoms];

    // GBSA if available
    let has_gbsa = !ff.gbsa_obc_params.is_empty();
    let mut radii = if has_gbsa {
        Some(vec![0.0f32; n_atoms])
    } else {
        None
    };
    let mut scales = if has_gbsa {
        Some(vec![0.0f32; n_atoms])
    } else {
        None
    };

    // Build lookup tables
    let nonbonded_map = build_nonbonded_map(&ff.nonbonded_params);
    let gbsa_map = build_gbsa_map(&ff.gbsa_obc_params);

    let mut num_parameterized = 0usize;
    let mut num_skipped = 0usize;

    // Process each residue
    for (res_idx, res_info) in processed.residue_info.iter().enumerate() {
        // Determine template name (with terminal caps if enabled)
        let template_name = if options.auto_terminal_caps {
            get_terminal_template_name(
                &res_info.res_name,
                res_idx,
                n_residues,
                &res_info.chain_id,
                processed,
                ff,
            )
        } else {
            res_info.res_name.clone()
        };

        // Look up template
        let template = match ff.get_residue(&template_name) {
            Some(t) => t,
            None => {
                // Try base name if terminal variant not found
                match ff.get_residue(&res_info.res_name) {
                    Some(t) => t,
                    None => {
                        match options.missing_mode {
                            MissingResidueMode::Fail => {
                                return Err(ParamError::MissingTemplate(res_info.res_name.clone()));
                            }
                            MissingResidueMode::SkipWarn => {
                                log::warn!(
                                    "Missing template for residue {}, skipping",
                                    res_info.res_name
                                );
                                num_skipped += res_info.num_atoms;
                                continue;
                            }
                            MissingResidueMode::ClosestMatch => {
                                // Find closest match
                                match find_closest_template(res_info, ff) {
                                    Some(t) => t,
                                    None => {
                                        log::warn!(
                                            "No matching template for {}",
                                            res_info.res_name
                                        );
                                        num_skipped += res_info.num_atoms;
                                        continue;
                                    }
                                }
                            }
                            MissingResidueMode::GaffFallback => {
                                log::warn!(
                                    "GAFF fallback not yet implemented, skipping {}",
                                    res_info.res_name
                                );
                                num_skipped += res_info.num_atoms;
                                continue;
                            }
                        }
                    }
                }
            }
        };

        // Build atom name -> template atom lookup
        let template_atoms: HashMap<&str, _> = template
            .atoms
            .iter()
            .map(|a| (a.name.as_str(), a))
            .collect();

        // Assign parameters to each atom in residue
        for atom_idx in res_info.start_atom..(res_info.start_atom + res_info.num_atoms) {
            let atom_name = &processed.raw_atoms.atom_names[atom_idx];

            // Find matching template atom
            if let Some(template_atom) = template_atoms.get(atom_name.as_str()) {
                // Assign charge from template
                charges[atom_idx] = template_atom.charge.unwrap_or(0.0);
                atom_types[atom_idx] = template_atom.atom_type.clone();

                // Look up LJ params by atom type
                if let Some(nb) = nonbonded_map.get(&template_atom.atom_type) {
                    sigmas[atom_idx] = nb.sigma;
                    epsilons[atom_idx] = nb.epsilon;
                } else {
                    log::debug!("No nonbonded params for type {}", template_atom.atom_type);
                }

                // Look up GBSA params if available
                if has_gbsa {
                    if let Some(gbsa) = gbsa_map.get(&template_atom.atom_type) {
                        if let Some(ref mut r) = radii {
                            r[atom_idx] = gbsa.radius;
                        }
                        if let Some(ref mut s) = scales {
                            s[atom_idx] = gbsa.scale;
                        }
                    }
                }

                num_parameterized += 1;
            } else {
                // Atom not in template
                log::debug!("Atom {} not in template {}", atom_name, template.name);
                num_skipped += 1;
            }
        }
    }

    Ok(MDParameters {
        charges,
        sigmas,
        epsilons,
        radii,
        scales,
        atom_types,
        num_parameterized,
        num_skipped,
    })
}

/// Build lookup map from atom type -> nonbonded params
fn build_nonbonded_map(params: &[NonbondedParam]) -> HashMap<String, &NonbondedParam> {
    params.iter().map(|p| (p.atom_type.clone(), p)).collect()
}

/// Build lookup map from atom type -> GBSA params
fn build_gbsa_map(params: &[GBSAOBCParam]) -> HashMap<String, &GBSAOBCParam> {
    params.iter().map(|p| (p.atom_type.clone(), p)).collect()
}

/// Get template name with terminal cap detection
fn get_terminal_template_name(
    base_name: &str,
    res_idx: usize,
    n_residues: usize,
    chain_id: &str,
    processed: &ProcessedStructure,
    ff: &ForceField,
) -> String {
    // Check if this is first residue in chain
    let is_n_terminal =
        res_idx == 0 || (res_idx > 0 && processed.residue_info[res_idx - 1].chain_id != chain_id);

    // Check if this is last residue in chain
    let is_c_terminal = res_idx == n_residues - 1
        || (res_idx < n_residues - 1 && processed.residue_info[res_idx + 1].chain_id != chain_id);

    // Try N-terminal variant first
    if is_n_terminal {
        let n_name = format!("N{}", base_name);
        if ff.get_residue(&n_name).is_some() {
            return n_name;
        }
    }

    // Try C-terminal variant
    if is_c_terminal {
        let c_name = format!("C{}", base_name);
        if ff.get_residue(&c_name).is_some() {
            return c_name;
        }
    }

    // Fall back to base name
    base_name.to_string()
}

/// Find the closest matching template based on shared atom names
fn find_closest_template<'a>(
    res_info: &crate::processing::ResidueInfo,
    ff: &'a ForceField,
) -> Option<&'a ResidueTemplate> {
    // This is a simplified version - we just try common variants
    // A more sophisticated version would score by atom overlap

    // Try adding/removing common suffixes
    let variants = [
        res_info.res_name.clone(),
        res_info.res_name.to_uppercase(),
        format!("N{}", res_info.res_name),
        format!("C{}", res_info.res_name),
    ];

    for name in &variants {
        if let Some(template) = ff.get_residue(name) {
            return Some(template);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forcefield::ForceField;
    use crate::structure::{AtomRecord, RawAtomData};

    fn make_test_forcefield() -> ForceField {
        let mut ff = ForceField::new("test".to_string());

        // Add a simple ALA template
        ff.residue_templates
            .push(crate::forcefield::ResidueTemplate {
                name: "ALA".to_string(),
                atoms: vec![
                    crate::forcefield::ResidueAtom {
                        name: "N".to_string(),
                        atom_type: "N".to_string(),
                        charge: Some(-0.4157),
                    },
                    crate::forcefield::ResidueAtom {
                        name: "CA".to_string(),
                        atom_type: "CX".to_string(),
                        charge: Some(0.0337),
                    },
                    crate::forcefield::ResidueAtom {
                        name: "C".to_string(),
                        atom_type: "C".to_string(),
                        charge: Some(0.5973),
                    },
                ],
                bonds: vec![],
                external_bonds: vec![],
                override_level: None,
            });

        // Add nonbonded params
        ff.nonbonded_params.push(NonbondedParam {
            atom_type: "N".to_string(),
            charge: 0.0,
            sigma: 0.325,
            epsilon: 0.711,
        });
        ff.nonbonded_params.push(NonbondedParam {
            atom_type: "CX".to_string(),
            charge: 0.0,
            sigma: 0.339,
            epsilon: 0.457,
        });
        ff.nonbonded_params.push(NonbondedParam {
            atom_type: "C".to_string(),
            charge: 0.0,
            sigma: 0.339,
            epsilon: 0.359,
        });

        ff.build_indices();
        ff
    }

    fn make_test_structure() -> ProcessedStructure {
        let mut raw = RawAtomData::with_capacity(3);

        raw.add_atom(AtomRecord {
            serial: 1,
            atom_name: "N".to_string(),
            alt_loc: ' ',
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 0.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "N".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });
        raw.add_atom(AtomRecord {
            serial: 2,
            atom_name: "CA".to_string(),
            alt_loc: ' ',
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 1.5,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });
        raw.add_atom(AtomRecord {
            serial: 3,
            atom_name: "C".to_string(),
            alt_loc: ' ',
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 3.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });

        ProcessedStructure::from_raw(raw).unwrap()
    }

    #[test]
    fn test_parameterize_simple() {
        let ff = make_test_forcefield();
        let structure = make_test_structure();
        let options = ParamOptions::default();

        let params = parameterize_structure(&structure, &ff, &options).unwrap();

        assert_eq!(params.charges.len(), 3);
        assert!((params.charges[0] - (-0.4157)).abs() < 1e-4); // N
        assert!((params.charges[1] - 0.0337).abs() < 1e-4); // CA
        assert!((params.charges[2] - 0.5973).abs() < 1e-4); // C

        assert_eq!(params.atom_types[0], "N");
        assert_eq!(params.atom_types[1], "CX");
        assert_eq!(params.atom_types[2], "C");

        assert_eq!(params.num_parameterized, 3);
        assert_eq!(params.num_skipped, 0);
    }

    #[test]
    fn test_missing_template_skip() {
        let ff = ForceField::new("empty".to_string());
        let structure = make_test_structure();
        let options = ParamOptions {
            auto_terminal_caps: false,
            missing_mode: MissingResidueMode::SkipWarn,
        };

        let params = parameterize_structure(&structure, &ff, &options).unwrap();

        // All atoms should be skipped (template not found)
        assert_eq!(params.num_skipped, 3);
        assert_eq!(params.num_parameterized, 0);
    }

    #[test]
    fn test_missing_template_fail() {
        let ff = ForceField::new("empty".to_string());
        let structure = make_test_structure();
        let options = ParamOptions {
            auto_terminal_caps: false,
            missing_mode: MissingResidueMode::Fail,
        };

        let result = parameterize_structure(&structure, &ff, &options);
        assert!(result.is_err());
    }
}
