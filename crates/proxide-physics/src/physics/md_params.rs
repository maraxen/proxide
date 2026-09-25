//! MD Parameterization - Assigns force field parameters to structures
//!
//! This module uses parsed force field data to assign charges, LJ parameters,
//! and GBSA radii to atoms in a ProcessedStructure.

use std::collections::{HashMap, HashSet};
use thiserror::Error;

use proxide_core::forcefield::{
    ForceField, GBSAOBCParam, HarmonicAngleParam, HarmonicBondParam, ImproperTorsionParam,
    NonbondedException, NonbondedParam, ProperTorsionParam, Topology,
};
use proxide_core::processing::ProcessedStructure;
use proxide_geometry::geometry::topology::assign_template_hydrogens;
use proxide_units::{ANGSTROM_TO_NM, KCAL_TO_KJ};

/// Errors during parameterization
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ParamError {
    #[error("Missing residue template: {0}")]
    MissingTemplate(String),

    #[error("Missing atom in template: residue={0}, atom={1}")]
    _MissingAtom(String, String),

    #[error("Missing nonbonded params for atom type: {0}")]
    _MissingNonbonded(String),

    #[error(
        "{0} atom(s) could not be parameterized (strict mode); see the \
         `unparameterized_atoms` report for indices"
    )]
    UnparameterizedAtoms(usize),

    #[error("topology {term} term {indices:?} references atom index >= n_atoms ({n_atoms})")]
    TopologyIndexOutOfRange {
        term: &'static str,
        indices: Vec<usize>,
        n_atoms: usize,
    },
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

    // --- Topology ---
    /// Bonds as [atom1_idx, atom2_idx]
    pub bonds: Vec<[usize; 2]>,
    /// Bond parameters (length, k)
    pub bond_params: Vec<[f32; 2]>,

    /// Angles as [atom1, atom2, atom3]
    pub angles: Vec<[usize; 3]>,
    /// Angle parameters (angle, k)
    pub angle_params: Vec<[f32; 2]>,

    /// Proper dihedrals as [atom1, atom2, atom3, atom4]
    pub dihedrals: Vec<[usize; 4]>,
    /// Dihedral parameters (periodicity, phase, k)
    /// Note: This is now a multi-row array of shape (num_dihedrals * max_proper_terms, 3)
    pub dihedral_params: Vec<[f32; 3]>,
    /// Maximum number of terms per proper dihedral
    pub max_proper_terms: usize,

    /// Improper dihedrals as [atom1, atom2, atom3, atom4]
    pub impropers: Vec<[usize; 4]>,
    /// Improper parameters (periodicity, phase, k)
    /// Note: This is now a multi-row array of shape (num_impropers * max_improper_terms, 3)
    pub improper_params: Vec<[f32; 3]>,
    /// Maximum number of terms per improper dihedral
    pub max_improper_terms: usize,

    /// 1-4 Pairs for scaling (atom1, atom2)
    pub pairs_14: Vec<[usize; 2]>,
    /// Per-pair exception parameters: (type1, type2, chargeProd, sigma, epsilon)
    pub nonbonded_exceptions: Vec<(String, String, f32, f32, f32)>,
    /// Per-atom-pair resolved 1-4 params, row-aligned with pairs_14: [chargeProd, sigma, epsilon].
    /// Uses explicit FF <Exception> override for the atom-type pair if present,
    /// else Lorentz-Berthelot combining rules scaled by ff.lj14scale / ff.coulomb14scale.
    pub resolved_nonbonded_14_params: Vec<[f32; 3]>,

    // --- CMAP ---
    /// CMAP torsions as [atom1, atom2, atom3, atom4, atom5]
    pub cmap_torsions: Vec<[usize; 5]>,
    /// CMAP map indices into cmap_grids
    pub cmap_map_indices: Vec<usize>,
    /// CMAP energy grids
    pub cmap_grids: Vec<proxide_core::forcefield::CMAPGrid>,
    // We'll assume global for now, handled by OpenMM, but we list the pairs.
    /// Indices of atoms this call did NOT confidently parameterize: atoms
    /// with no matching template atom, atoms in a residue with no matching
    /// residue template (only crude element-based LJ fallback, charge left
    /// at 0.0), solvent atoms that couldn't be matched to a water model, and
    /// -- critically -- any atom this function never visits at all (e.g.
    /// ligand/ion atoms, which are outside `residue_info` and are not
    /// handled here; see `parameterize_molecule` for ligands). Empty means
    /// every atom received real force-field- or water-model-sourced values.
    /// See [`ParamOptions::strict`] to turn a non-empty report into a hard
    /// error instead.
    pub unparameterized_atoms: Vec<usize>,
}

/// Options for parameterization
#[derive(Debug, Clone)]
pub struct ParamOptions {
    /// Auto-detect terminal residue variants (NALA, CALA, etc.)
    pub auto_terminal_caps: bool,
    /// How to handle missing residue templates
    pub missing_mode: MissingResidueMode,
    /// Water model used to parameterize solvent atoms (molecule_type == 2,
    /// e.g. `HOH`/`WAT`/`TIP3`/`SOL`/`DOD`), which are excluded from
    /// `residue_info` and therefore invisible to the residue-template loop
    /// above -- without this, solvent silently keeps charge=sigma=epsilon=0.
    /// Passed to [`crate::physics::water::get_water_model`]; supported
    /// names are `"TIP3P"` (default), `"SPCE"`/`"SPC/E"`, `"TIP4PEW"`/
    /// `"TIP4P-EW"`. An unrecognized name leaves solvent atoms
    /// unparameterized (reported via `unparameterized_atoms`, not silently
    /// zeroed) rather than erroring, so a typo degrades gracefully into a
    /// visible report instead of a hard failure mid-pipeline.
    pub water_model: String,
    /// If true, return `Err(ParamError::UnparameterizedAtoms)` when the
    /// resulting `unparameterized_atoms` report is non-empty, instead of
    /// returning `Ok` with zeroed values for those atoms. Note: `strict` does
    /// not govern invariant violations such as out-of-range topology indices,
    /// which always error regardless of this flag.
    pub strict: bool,
}

impl Default for ParamOptions {
    fn default() -> Self {
        Self {
            auto_terminal_caps: true,
            missing_mode: MissingResidueMode::SkipWarn,
            water_model: "TIP3P".to_string(),
            strict: false,
        }
    }
}

/// Parameterize a structure using force field templates
pub fn parameterize_structure(
    processed: &ProcessedStructure,
    topology: &Topology,
    ff: &ForceField,
    options: &ParamOptions,
) -> Result<MDParameters, ParamError> {
    let n_atoms = processed.raw_atoms.num_atoms;

    // Validate topology indices before processing
    for bond in &topology.bonds {
        if bond.i >= n_atoms || bond.j >= n_atoms {
            return Err(ParamError::TopologyIndexOutOfRange {
                term: "bond",
                indices: vec![bond.i, bond.j],
                n_atoms,
            });
        }
    }

    for angle in &topology.angles {
        if angle.i >= n_atoms || angle.j >= n_atoms || angle.k >= n_atoms {
            return Err(ParamError::TopologyIndexOutOfRange {
                term: "angle",
                indices: vec![angle.i, angle.j, angle.k],
                n_atoms,
            });
        }
    }

    for dih in &topology.proper_dihedrals {
        if dih.i >= n_atoms || dih.j >= n_atoms || dih.k >= n_atoms || dih.l >= n_atoms {
            return Err(ParamError::TopologyIndexOutOfRange {
                term: "proper",
                indices: vec![dih.i, dih.j, dih.k, dih.l],
                n_atoms,
            });
        }
    }

    for imp in &topology.improper_dihedrals {
        if imp.i >= n_atoms || imp.j >= n_atoms || imp.k >= n_atoms || imp.l >= n_atoms {
            return Err(ParamError::TopologyIndexOutOfRange {
                term: "improper",
                indices: vec![imp.i, imp.j, imp.k, imp.l],
                n_atoms,
            });
        }
    }

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

    // Tracks which atoms received a confident, force-field- (or water-
    // model-)sourced charge assignment. Everything left `false` at the end
    // -- unmatched atoms, atoms in a residue with no template at all (which
    // still get a crude element-based LJ fallback below but keep charge
    // 0.0), unhandled solvent, and any atom this function never visits
    // (ligand/ion atoms, which live outside `residue_info`) -- is surfaced
    // via `unparameterized_atoms` rather than silently returned as zeros.
    let mut touched = vec![false; n_atoms];

    // Mapping from (class1, class2) -> BondParam
    // We need atom classes for lookup, so let's store them
    let mut atom_classes = vec![String::new(); n_atoms];

    // Process each residue
    for (res_idx, res_info) in processed.residue_info.iter().enumerate() {
        // Determine template name (with terminal caps if enabled)
        let template_name = if options.auto_terminal_caps {
            get_terminal_template_name(
                &res_info.res_name,
                res_idx,
                processed.num_residues,
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
                match ff.get_residue(&res_info.res_name) {
                    Some(t) => t,
                    None => {
                        if options.missing_mode == MissingResidueMode::Fail {
                            return Err(ParamError::MissingTemplate(res_info.res_name.clone()));
                        }

                        // ClosestMatch is currently a stub, behaves like SkipWarn but could be expanded
                        if options.missing_mode == MissingResidueMode::ClosestMatch {
                            // TODO: Implement actual atom-name based matching
                        }

                        // Apply element-based LJ fallbacks
                        for atom_idx in
                            res_info.start_atom..(res_info.start_atom + res_info.num_atoms)
                        {
                            let element = &processed.raw_atoms.elements[atom_idx];
                            let (fb_sigma, fb_epsilon) = match element.to_uppercase().as_str() {
                                "H" => (0.1069, 0.065),
                                "C" => (0.34, 0.36),
                                "N" => (0.325, 0.71),
                                "O" => (0.296, 0.88),
                                "S" => (0.356, 1.04),
                                _ => (
                                    crate::physics::constants::DEFAULT_SIGMA,
                                    crate::physics::constants::DEFAULT_EPSILON,
                                ),
                            };
                            sigmas[atom_idx] = fb_sigma;
                            epsilons[atom_idx] = fb_epsilon;
                        }
                        num_skipped += res_info.num_atoms;
                        continue;
                    }
                }
            }
        };

        let template_atoms: HashMap<&str, _> = template
            .atoms
            .iter()
            .map(|a| (a.name.as_str(), a))
            .collect();
        let mut local_to_global: HashMap<&str, usize> = HashMap::new();
        let mut claimed_template_atoms: HashSet<String> = HashSet::new();
        let mut unmatched_h_indices: Vec<usize> = Vec::new();

        // PASS 1: Exact name match
        for atom_idx in res_info.start_atom..(res_info.start_atom + res_info.num_atoms) {
            let atom_name = &processed.raw_atoms.atom_names[atom_idx];
            let template_atom_opt =
                template_atoms
                    .get(atom_name.as_str())
                    .or_else(|| match atom_name.as_str() {
                        "H" => template_atoms.get("H1"),
                        "H1" => template_atoms.get("H"),
                        _ => None,
                    });

            if let Some(template_atom) = template_atom_opt {
                charges[atom_idx] = template_atom.charge.unwrap_or(0.0);
                atom_types[atom_idx] = template_atom.atom_type.clone();

                let atom_class = if let Some(at) = ff.get_atom_type(&template_atom.atom_type) {
                    at.class.clone()
                } else {
                    template_atom.atom_type.clone()
                };
                atom_classes[atom_idx] = atom_class.clone();

                if let Some(nb) = nonbonded_map.get(&template_atom.atom_type) {
                    sigmas[atom_idx] = nb.sigma;
                    epsilons[atom_idx] = nb.epsilon;
                } else if let Some(nb) = nonbonded_map.get(&atom_class) {
                    sigmas[atom_idx] = nb.sigma;
                    epsilons[atom_idx] = nb.epsilon;
                }

                if has_gbsa {
                    if let Some(gbsa) = gbsa_map.get(&template_atom.atom_type) {
                        if let Some(ref mut r) = radii {
                            r[atom_idx] = gbsa.radius;
                        }
                        if let Some(ref mut s) = scales {
                            s[atom_idx] = gbsa.scale;
                        }
                    } else if let Some(gbsa) = gbsa_map.get(&atom_class) {
                        if let Some(ref mut r) = radii {
                            r[atom_idx] = gbsa.radius;
                        }
                        if let Some(ref mut s) = scales {
                            s[atom_idx] = gbsa.scale;
                        }
                    }
                }

                local_to_global.insert(atom_name.as_str(), atom_idx);
                claimed_template_atoms.insert(template_atom.name.clone());
                touched[atom_idx] = true;
                num_parameterized += 1;
            } else {
                let element = &processed.raw_atoms.elements[atom_idx];
                if element.eq_ignore_ascii_case("H") {
                    unmatched_h_indices.push(atom_idx);
                } else {
                    num_skipped += 1;
                }
            }
        }

        // PASS 2: H fallback using geometry crate
        if !unmatched_h_indices.is_empty() {
            let h_mapping = assign_template_hydrogens(
                processed,
                template,
                &local_to_global,
                &mut claimed_template_atoms,
                &unmatched_h_indices,
                ff,
            );

            for (h_idx, t_name) in h_mapping {
                if let Some(template_atom) = template_atoms.get(t_name.as_str()) {
                    charges[h_idx] = template_atom.charge.unwrap_or(0.0);
                    atom_types[h_idx] = template_atom.atom_type.clone();

                    let atom_class = if let Some(at) = ff.get_atom_type(&template_atom.atom_type) {
                        at.class.clone()
                    } else {
                        template_atom.atom_type.clone()
                    };
                    atom_classes[h_idx] = atom_class.clone();

                    if let Some(nb) = nonbonded_map.get(&template_atom.atom_type) {
                        sigmas[h_idx] = nb.sigma;
                        epsilons[h_idx] = nb.epsilon;
                    } else if let Some(nb) = nonbonded_map.get(&atom_class) {
                        sigmas[h_idx] = nb.sigma;
                        epsilons[h_idx] = nb.epsilon;
                    }

                    if has_gbsa {
                        if let Some(gbsa) = gbsa_map.get(&template_atom.atom_type) {
                            if let Some(ref mut r) = radii {
                                r[h_idx] = gbsa.radius;
                            }
                            if let Some(ref mut s) = scales {
                                s[h_idx] = gbsa.scale;
                            }
                        } else if let Some(gbsa) = gbsa_map.get(&atom_class) {
                            if let Some(ref mut r) = radii {
                                r[h_idx] = gbsa.radius;
                            }
                            if let Some(ref mut s) = scales {
                                s[h_idx] = gbsa.scale;
                            }
                        }
                    }
                    touched[h_idx] = true;
                    num_parameterized += 1;
                }
            }
        }
    }

    // --- Assign solvent (water) parameters ---
    // Solvent atoms (HOH/WAT/TIP3/SOL/DOD) are intentionally excluded from
    // `residue_info` -- see `proxide_core::processing::residues` -- so the
    // per-residue template loop above never visits them, and they would
    // otherwise keep charge=sigma=epsilon=0.0 forever, silently, even with
    // a force field that happens to define an HOH template (there is
    // currently no multi-force-field merging in this crate, so a protein FF
    // like ff14SB and a water FF like tip3p.xml can't both be loaded into
    // one `ForceField` at once anyway). Parameterize solvent here from the
    // hard-coded `WaterModel` catalog (`crate::physics::water`) instead.
    //
    // Units: `get_water_model` returns AMBER-convention values (Angstroms,
    // kcal/mol -- see doc comment on `WaterModel`). Everything else in this
    // function (charges/sigmas/epsilons parsed from the OpenMM-XML force
    // field via `NonbondedParam`, whose field docs specify nm and kJ/mol)
    // is in OpenMM/GROMACS convention, and the Python boundary applies a
    // single unit-system conversion pass over the whole `sigmas`/`epsilons`
    // arrays assuming they're uniformly in that convention (see
    // `proxide_units::registry` + `py_parsers.rs`'s `conv.*` scaling). So
    // water values are converted right here, at assignment time, to match
    // -- getting this wrong would silently replace one wrong answer with a
    // different wrong answer.
    if !processed.solvent_atoms.is_empty() {
        match crate::physics::water::get_water_model(&options.water_model, true) {
            Ok(model) => {
                let mut i = 0;
                while i < processed.solvent_atoms.len() {
                    let first_idx = processed.solvent_atoms[i];
                    let key = (
                        processed.raw_atoms.chain_ids[first_idx].as_str(),
                        processed.raw_atoms.res_ids[first_idx],
                        processed.raw_atoms.insertion_codes[first_idx],
                    );
                    let mut j = i + 1;
                    while j < processed.solvent_atoms.len() {
                        let idx = processed.solvent_atoms[j];
                        let k = (
                            processed.raw_atoms.chain_ids[idx].as_str(),
                            processed.raw_atoms.res_ids[idx],
                            processed.raw_atoms.insertion_codes[idx],
                        );
                        if k != key {
                            break;
                        }
                        j += 1;
                    }
                    let mol_atoms = &processed.solvent_atoms[i..j];
                    i = j;

                    if mol_atoms.len() != model.atoms.len() {
                        // Atom count doesn't match the chosen model (e.g. an
                        // O-only crystallographic water against TIP3P, which
                        // expects O+H1+H2). Leave unparameterized -- it will
                        // surface via `unparameterized_atoms` -- rather than
                        // guess at missing atoms.
                        continue;
                    }

                    // Assign each atom in this water molecule to a model
                    // site name by element, consuming "H1" before "H2" in
                    // file order. The two hydrogens are physically
                    // equivalent in every model here, so file order is an
                    // arbitrary but stable and harmless choice.
                    let mut slot_names: Vec<&'static str> = Vec::with_capacity(mol_atoms.len());
                    let mut h_seen = 0usize;
                    let mut recognized = true;
                    for &atom_idx in mol_atoms {
                        let element = processed.raw_atoms.elements[atom_idx].to_uppercase();
                        let slot: &'static str = match element.as_str() {
                            "O" => "O",
                            "H" => {
                                h_seen += 1;
                                if h_seen == 1 {
                                    "H1"
                                } else {
                                    "H2"
                                }
                            }
                            _ if model.has_virtual_sites => "M",
                            _ => {
                                recognized = false;
                                break;
                            }
                        };
                        slot_names.push(slot);
                    }
                    if !recognized {
                        continue;
                    }

                    for (&atom_idx, &slot) in mol_atoms.iter().zip(slot_names.iter()) {
                        let charge = model.charges.get(slot);
                        let sigma_a = model.sigmas.get(slot);
                        let epsilon_kcal = model.epsilons.get(slot);
                        let (Some(&charge), Some(&sigma_a), Some(&epsilon_kcal)) =
                            (charge, sigma_a, epsilon_kcal)
                        else {
                            continue;
                        };
                        charges[atom_idx] = charge;
                        sigmas[atom_idx] = sigma_a * ANGSTROM_TO_NM;
                        epsilons[atom_idx] = epsilon_kcal * KCAL_TO_KJ;
                        atom_types[atom_idx] = format!("{}-{}", model.name, slot);
                        atom_classes[atom_idx] = atom_types[atom_idx].clone();
                        touched[atom_idx] = true;
                        num_parameterized += 1;
                    }
                }
            }
            Err(e) => {
                log::warn!(
                    "Unknown water model '{}' ({e}); {} solvent atom(s) left unparameterized",
                    options.water_model,
                    processed.solvent_atoms.len()
                );
            }
        }
    }

    // --- Unparameterized-atom report ---
    // Never hand back charge=0/sigma=0/epsilon=0 for an atom this function
    // didn't confidently visit without saying so: `touched[i] == false`
    // covers unmatched template atoms, whole residues with no template at
    // all (crude element-based LJ fallback above, charge left at 0.0),
    // unresolved solvent, AND any atom outside `residue_info`/solvent
    // entirely (ligand/ion atoms -- this function never visits those; see
    // `parameterize_molecule` for standalone ligand/GAFF parameterization).
    let unparameterized_atoms: Vec<usize> =
        (0..n_atoms).filter(|&idx| !touched[idx]).collect();

    if !unparameterized_atoms.is_empty() {
        log::warn!(
            "parameterize_structure: {} of {} atom(s) could not be confidently \
             parameterized (charge/sigma/epsilon may be 0.0 or a crude element-based \
             fallback) -- see MDParameters::unparameterized_atoms for indices",
            unparameterized_atoms.len(),
            n_atoms
        );
        if options.strict {
            return Err(ParamError::UnparameterizedAtoms(unparameterized_atoms.len()));
        }
    }

    // --- Assign bonded parameters using Topology ---
    let mut bonds_vec = Vec::new();
    let mut bond_params = Vec::new();
    for bond in &topology.bonds {
        let (i, j) = (bond.i, bond.j);
        bonds_vec.push([i, j]);
        if let Some(params) = lookup_bond(&atom_classes[i], &atom_classes[j], ff) {
            bond_params.push([params.length, params.k]);
        } else {
            bond_params.push([0.0, 0.0]);
        }
    }

    let mut angles_vec = Vec::new();
    let mut angle_params = Vec::new();
    for angle in &topology.angles {
        let (i, j, k) = (angle.i, angle.j, angle.k);
        angles_vec.push([i, j, k]);
        if let Some(params) = lookup_angle(&atom_classes[i], &atom_classes[j], &atom_classes[k], ff)
        {
            angle_params.push([params.angle, params.k]);
        } else {
            angle_params.push([0.0, 0.0]);
        }
    }

    let mut dihedrals_vec = Vec::new();
    let mut all_proper_terms = Vec::new();
    let mut max_proper_terms = 0;
    for dih in &topology.proper_dihedrals {
        let proper_matches = lookup_proper(
            &atom_classes[dih.i],
            &atom_types[dih.i],
            &atom_classes[dih.j],
            &atom_types[dih.j],
            &atom_classes[dih.k],
            &atom_types[dih.k],
            &atom_classes[dih.l],
            &atom_types[dih.l],
            ff,
        );

        if !proper_matches.is_empty() {
            let mut terms_collected = Vec::new();
            // Use a set to deduplicate by (periodicity, phase)
            let mut seen_terms: HashSet<(u32, String)> = HashSet::new();

            // Collect terms from all matching parameters
            for params in proper_matches {
                for term in &params.terms {
                    if term.k.abs() > 1e-6 {
                        // Deduplicate by (periodicity, phase) to avoid double-counting
                        let phase_str = format!("{:.10}", term.phase);
                        let term_key = (term.periodicity, phase_str);
                        if !seen_terms.contains(&term_key) {
                            seen_terms.insert(term_key);
                            terms_collected.push([term.periodicity as f32, term.phase, term.k]);
                        }
                    }
                }
            }

            if !terms_collected.is_empty() {
                max_proper_terms = max_proper_terms.max(terms_collected.len());
                dihedrals_vec.push([dih.i, dih.j, dih.k, dih.l]);
                all_proper_terms.push(terms_collected);
            }
        }
    }

    let mut dihedral_params = Vec::with_capacity(dihedrals_vec.len() * max_proper_terms);
    for terms in all_proper_terms {
        for i in 0..max_proper_terms {
            if i < terms.len() {
                dihedral_params.push(terms[i]);
            } else {
                dihedral_params.push([0.0, 0.0, 0.0]);
            }
        }
    }

    let mut impropers_vec = Vec::new();
    let mut all_improper_terms = Vec::new();
    let mut max_improper_terms = 0;
    for imp in &topology.improper_dihedrals {
        if let Some(params) = lookup_improper(
            ImproperLookupParams {
                c1: &atom_classes[imp.i],
                t1: &atom_types[imp.i],
                c_center: &atom_classes[imp.j],
                t_center: &atom_types[imp.j],
                c3: &atom_classes[imp.k],
                t3: &atom_types[imp.k],
                c4: &atom_classes[imp.l],
                t4: &atom_types[imp.l],
            },
            ff,
        ) {
            let mut terms_collected = Vec::new();
            for term in &params.terms {
                if term.k.abs() > 1e-6 {
                    terms_collected.push([term.periodicity as f32, term.phase, term.k]);
                }
            }
            if !terms_collected.is_empty() {
                max_improper_terms = max_improper_terms.max(terms_collected.len());
                impropers_vec.push([imp.i, imp.j, imp.k, imp.l]);
                all_improper_terms.push(terms_collected);
            }
        }
    }

    let mut improper_params = Vec::with_capacity(impropers_vec.len() * max_improper_terms);
    for terms in all_improper_terms {
        for i in 0..max_improper_terms {
            if i < terms.len() {
                improper_params.push(terms[i]);
            } else {
                improper_params.push([0.0, 0.0, 0.0]);
            }
        }
    }

    // --- Nonbonded Exceptions (1-4) ---
    // Pre-compute 1-2 and 1-3 exclusions for filtering 1-4 exceptions
    let mut exclusions_123 = HashSet::new();
    for bond in &topology.bonds {
        let (i, j) = if bond.i < bond.j {
            (bond.i, bond.j)
        } else {
            (bond.j, bond.i)
        };
        exclusions_123.insert((i, j));
    }
    for angle in &topology.angles {
        let (i, k) = if angle.i < angle.k {
            (angle.i, angle.k)
        } else {
            (angle.k, angle.i)
        };
        exclusions_123.insert((i, k));
    }

    // Solvent atoms (water, per `ProcessedStructure::solvent_atoms`) never carry
    // torsion terms in any standard water model (TIP3P/TIP4P/...), so a *real*
    // 1-4 nonbonded exception can never involve one. `topology` here is built
    // purely from distance + covalent-radius inference over the WHOLE system
    // at once (see `proxide_geometry::geometry::topology::infer_bonds`), with
    // no residue-boundary or force-field-template awareness -- so a geometric
    // dihedral chain that touches a solvent atom only exists because two
    // different molecules got spuriously bonded together (e.g. a water's H
    // sitting anomalously close to another water's O, or to a protein atom).
    // Without this filter, such a spurious bond silently fabricates a "scaled
    // 1-4" nonbonded exception with a nonzero energy for atoms that have no
    // real covalent relationship at all. Found via prolix's
    // dhfr_pme_exclusion_census.py (task 260909_dhfr_gap_tranche2): a solvated
    // 1VII system (7507 atoms, only 596 of them protein) reported ~1563-1564
    // `pairs_14` entries, reproduced consistently across separate runs --
    // implausibly high for a 596-atom protein alone (low hundreds at most)
    // *and* including water, which must structurally contribute exactly zero
    // real 1-4 pairs (no water model has torsion terms) -- and at least one
    // entry referenced an atom index that didn't even exist in the 7507-atom
    // system. (An initial OpenMM-side "~40 exceptions" comparison baseline
    // for the same run was later retracted as unreliable -- a separate,
    // unrelated OpenMM/SWIG binding issue on that environment, prolix
    // backlog #5055 -- so it is deliberately not cited here; the anomaly
    // stands on proxide's own numbers alone.) See
    // `test_pairs_14_excludes_spurious_cross_water_bond` below.
    let solvent_set: HashSet<usize> = processed.solvent_atoms.iter().copied().collect();

    let mut pairs_14 = Vec::new();
    let mut seen_14_pairs = HashSet::new();
    let mut nonbonded_exceptions = Vec::new();
    for exc in &ff.exceptions {
        nonbonded_exceptions.push((
            exc.type1.clone(),
            exc.type2.clone(),
            exc.charge_prod,
            exc.sigma,
            exc.epsilon,
        ));
    }

    for dih in &topology.proper_dihedrals {
        if solvent_set.contains(&dih.i)
            || solvent_set.contains(&dih.j)
            || solvent_set.contains(&dih.k)
            || solvent_set.contains(&dih.l)
        {
            continue;
        }

        let pair_key = if dih.i < dih.l {
            (dih.i, dih.l)
        } else {
            (dih.l, dih.i)
        };
        if !seen_14_pairs.contains(&pair_key) && !exclusions_123.contains(&pair_key) {
            seen_14_pairs.insert(pair_key);
            pairs_14.push([dih.i, dih.l]);
        }
    }

    // --- CMAP ---
    let mut cmap_torsions = Vec::new();
    let mut cmap_map_indices = Vec::new();
    if let Some(cmap_data) = &ff.cmap_data {
        for i in 0..processed.num_residues {
            if i == 0 || i + 1 >= processed.num_residues {
                continue;
            }
            let res_prev = &processed.residue_info[i - 1];
            let res_curr = &processed.residue_info[i];
            let res_next = &processed.residue_info[i + 1];

            let find_atom =
                |ri: &proxide_core::processing::ResidueInfo, name: &str| -> Option<usize> {
                    (ri.start_atom..(ri.start_atom + ri.num_atoms))
                        .find(|&idx| processed.raw_atoms.atom_names[idx] == name)
                };

            if let (Some(idx1), Some(idx2), Some(idx3), Some(idx4), Some(idx5)) = (
                find_atom(res_prev, "C"),
                find_atom(res_curr, "N"),
                find_atom(res_curr, "CA"),
                find_atom(res_curr, "C"),
                find_atom(res_next, "N"),
            ) {
                let c1 = &atom_classes[idx1];
                let t2 = &atom_types[idx2];
                let t3 = &atom_types[idx3];
                let t4 = &atom_types[idx4];
                let c5 = &atom_classes[idx5];

                for torsion in &cmap_data.torsions {
                    if torsion.class1 == *c1
                        && torsion.type2 == *t2
                        && torsion.type3 == *t3
                        && torsion.type4 == *t4
                        && torsion.class5 == *c5
                    {
                        cmap_torsions.push([idx1, idx2, idx3, idx4, idx5]);
                        cmap_map_indices.push(torsion.map_index);
                        break;
                    }
                }
            }
        }
    }

    let resolved_nonbonded_14_params = resolve_14_params(
        &pairs_14,
        &charges,
        &sigmas,
        &epsilons,
        &atom_types,
        &ff.exceptions,
        ff.lj14scale,
        ff.coulomb14scale,
    );
    Ok(MDParameters {
        charges,
        sigmas,
        epsilons,
        radii,
        scales,
        atom_types,
        num_parameterized,
        num_skipped,
        bonds: bonds_vec,
        bond_params,
        angles: angles_vec,
        angle_params,
        dihedrals: dihedrals_vec,
        dihedral_params,
        max_proper_terms,
        impropers: impropers_vec,
        improper_params,
        max_improper_terms,
        pairs_14,
        resolved_nonbonded_14_params,
        nonbonded_exceptions,
        cmap_torsions,
        cmap_map_indices,

        cmap_grids: if let Some(cmap_data) = &ff.cmap_data {
            cmap_data.maps.clone()
        } else {
            Vec::new()
        },
        unparameterized_atoms,
    })
}

/// Parameterize a molecule using GAFF (for ligands and small molecules)
///
/// This function is for molecules that don't have residue templates.
/// It infers topology from coordinates, assigns GAFF atom types, and
/// looks up LJ parameters from the GAFF parameter set.
///
/// Note: GAFF does not provide partial charges. Use antechamber or AM1-BCC
/// for accurate charges. This function assigns zero charges by default.
pub fn parameterize_molecule(
    coords: &[[f32; 3]],
    elements: &[String],
    bond_tolerance: f32,
) -> Result<MDParameters, ParamError> {
    use proxide_core::forcefield::topology::Topology;
    use proxide_gaff::gaff::{assign_gaff_types, GaffParameters};

    let n_atoms = elements.len();
    if coords.len() != n_atoms {
        return Err(ParamError::MissingTemplate(format!(
            "Coordinate/element count mismatch: {} vs {}",
            coords.len(),
            n_atoms
        )));
    }

    // Infer topology from coordinates
    let topology =
        proxide_geometry::geometry::topology::generate_topology(coords, elements, bond_tolerance);
    let gaff = GaffParameters::new();

    // Assign GAFF atom types
    let gaff_types = assign_gaff_types(elements, &topology, &gaff);

    // Initialize parameter arrays
    let charges = vec![0.0f32; n_atoms]; // GAFF doesn't provide charges
    let mut sigmas = vec![0.0f32; n_atoms];
    let mut epsilons = vec![0.0f32; n_atoms];
    let mut atom_types = vec![String::new(); n_atoms];
    let mut num_parameterized = 0usize;
    let mut num_skipped = 0usize;
    let mut unparameterized_atoms = Vec::new();

    // Assign LJ parameters from GAFF atom types
    for (i, gaff_type_opt) in gaff_types.iter().enumerate() {
        if let Some(gaff_type) = gaff_type_opt {
            if let Some(type_params) = gaff.atom_types.get(gaff_type) {
                sigmas[i] = type_params.sigma;
                epsilons[i] = type_params.epsilon;
                atom_types[i] = gaff_type.clone();
                num_parameterized += 1;
            } else {
                num_skipped += 1;
                unparameterized_atoms.push(i);
            }
        } else {
            num_skipped += 1;
            unparameterized_atoms.push(i);
        }
    }

    // Convert topology bonds to our format
    let bonds_vec: Vec<[usize; 2]> = topology.bonds.iter().map(|b| [b.i, b.j]).collect();

    // Generate angles from topology
    let angles = Topology::generate_angles(&topology.adjacency);
    let angles_vec: Vec<[usize; 3]> = angles.iter().map(|a| [a.i, a.j, a.k]).collect();

    // Generate dihedrals from topology
    let dihedrals = Topology::generate_proper_dihedrals(&topology.adjacency);
    let dihedrals_vec: Vec<[usize; 4]> = dihedrals.iter().map(|d| [d.i, d.j, d.k, d.l]).collect();

    // Generate impropers from topology
    let impropers = Topology::generate_improper_dihedrals(&topology.adjacency, elements);
    let impropers_vec: Vec<[usize; 4]> = impropers.iter().map(|d| [d.i, d.j, d.k, d.l]).collect();

    // For bond/angle/dihedral params, use default values since GAFF bond params
    // require specific type pairs. This is a simplified implementation.
    // A full implementation would use GAFF bond/angle/dihedral parameter tables.
    let bond_params: Vec<[f32; 2]> = bonds_vec
        .iter()
        .map(|_| [0.15, 300.0]) // Default: 1.5 Å, 300 kJ/mol/nm²
        .collect();

    let angle_params: Vec<[f32; 2]> = angles_vec
        .iter()
        .map(|_| [1.91, 100.0]) // Default: ~109.5°, 100 kJ/mol/rad²
        .collect();

    let dihedral_params: Vec<[f32; 3]> = dihedrals_vec
        .iter()
        .map(|_| [1.0, 0.0, 0.0]) // Default: periodicity 1, phase 0, k 0
        .collect();

    let improper_params: Vec<[f32; 3]> = impropers_vec
        .iter()
        .map(|_| [2.0, std::f32::consts::PI, 10.0]) // Default: periodicity 2, phase π, k 10
        .collect();

    // 1-4 pairs from dihedrals
    let pairs_14: Vec<[usize; 2]> = dihedrals_vec.iter().map(|d| [d[0], d[3]]).collect();
    let resolved_nonbonded_14_params = resolve_14_params(
        &pairs_14,
        &charges,
        &sigmas,
        &epsilons,
        &atom_types,
        &[],
        0.5,
        0.833333,
    );

    Ok(MDParameters {
        charges,
        sigmas,
        epsilons,
        radii: None,
        scales: None,
        atom_types,
        num_parameterized,
        num_skipped,
        bonds: bonds_vec,
        bond_params,
        angles: angles_vec,
        angle_params,
        dihedrals: dihedrals_vec,
        dihedral_params,
        max_proper_terms: 1,
        impropers: impropers_vec,
        improper_params,
        max_improper_terms: 1,
        resolved_nonbonded_14_params,
        pairs_14,
        nonbonded_exceptions: Vec::new(),
        cmap_torsions: Vec::new(),
        cmap_map_indices: Vec::new(),
        cmap_grids: Vec::new(),
        unparameterized_atoms,
    })
}

// --- Lookup Helpers ---

fn lookup_bond<'a>(c1: &str, c2: &str, ff: &'a ForceField) -> Option<&'a HarmonicBondParam> {
    // Try c1-c2, then c2-c1
    // Optimization: Store map (class1, class2) -> Param in ForceField
    // For now, linear search is okay or we'd duplicate build logic.
    // Actually FF struct "harmonic_bonds: Vec<HarmonicBondParam>".
    ff.harmonic_bonds
        .iter()
        .find(|&b| (b.class1 == c1 && b.class2 == c2) || (b.class1 == c2 && b.class2 == c1))
        .map(|v| v as _)
}

fn lookup_angle<'a>(
    c1: &str,
    c2: &str,
    c3: &str,
    ff: &'a ForceField,
) -> Option<&'a HarmonicAngleParam> {
    // Try c1-c2-c3, c3-c2-c1
    for a in &ff.harmonic_angles {
        if a.class2 != c2 {
            continue;
        }
        if (a.class1 == c1 && a.class3 == c3) || (a.class1 == c3 && a.class3 == c1) {
            return Some(a);
        }
    }
    None
}

fn matches(def: &str, cls: &str, typ: &str) -> bool {
    def == cls || def == typ || def == "X" || def.is_empty()
}

#[allow(clippy::too_many_arguments)]
fn lookup_proper<'a>(
    c1: &str,
    t1: &str,
    c2: &str,
    t2: &str,
    c3: &str,
    t3: &str,
    c4: &str,
    t4: &str,
    ff: &'a ForceField,
) -> Vec<&'a ProperTorsionParam> {
    let mut specific_matches: Vec<&'a ProperTorsionParam> = Vec::new();
    let mut wildcard_matches: Vec<&'a ProperTorsionParam> = Vec::new();

    for t in &ff.proper_torsions {
        // Forward match?
        let fwd_match = matches(&t.class2, c2, t2)
            && matches(&t.class3, c3, t3)
            && matches(&t.class1, c1, t1)
            && matches(&t.class4, c4, t4);

        // Reverse match?
        let rev_match = matches(&t.class2, c3, t3)
            && matches(&t.class3, c2, t2)
            && matches(&t.class1, c4, t4)
            && matches(&t.class4, c1, t1);

        if fwd_match || rev_match {
            let has_wildcard = t.class1.is_empty()
                || t.class1 == "X"
                || t.class2.is_empty()
                || t.class2 == "X"
                || t.class3.is_empty()
                || t.class3 == "X"
                || t.class4.is_empty()
                || t.class4 == "X";

            if has_wildcard {
                wildcard_matches.push(t);
            } else {
                specific_matches.push(t);
            }
        }
    }

    // Return specific matches if any exist; otherwise return wildcard matches
    if !specific_matches.is_empty() {
        specific_matches
    } else {
        wildcard_matches
    }
}

struct ImproperLookupParams<'a> {
    c1: &'a str,
    t1: &'a str,
    c_center: &'a str,
    t_center: &'a str,
    c3: &'a str,
    t3: &'a str,
    c4: &'a str,
    t4: &'a str,
}

// TODO(parity): Proxide places the improper center atom at position j (index 1) in the
// quad (i, j=center, k, l), following the topology.rs `new_improper` convention. OpenMM's
// PeriodicTorsionForce places the center at position k (index 2): (other_a, other_b, CENTER,
// other_c). This axis mismatch means ~25 of the impropers we generate are not recognized by
// OpenMM's atom-ordered lookup, and ~9 additional quads OpenMM registers are not generated
// at all by `generate_improper_dihedrals`. Combined, ~34 OpenMM torsion terms are absent
// from prolix. For nearly-planar sp2 centers (carbonyl, amide) the energy contribution is
// ~0.003 kcal/mol total (angles near the minimum), so this has negligible numerical impact,
// but the topology is architecturally wrong. Fix: change `new_improper` to put center at
// index 2, then update `ImproperLookupParams` ordering to match.
fn lookup_improper<'a>(
    params: ImproperLookupParams<'a>,
    ff: &'a ForceField,
) -> Option<&'a ImproperTorsionParam> {
    let mut best_match: Option<&'a ImproperTorsionParam> = None;

    for t in &ff.improper_torsions {
        // 1. Central atom must match t.class1 (AMBER ff.xml convention: center is class1)
        if !matches(&t.class1, params.c_center, params.t_center) {
            continue;
        }

        // 2. The other 3 atoms (c1, c3, c4) must match t.class2, t.class3, t.class4 in ANY order.
        let def_others = [&t.class2, &t.class3, &t.class4];
        let target_others = vec![
            (params.c1, params.t1),
            (params.c3, params.t3),
            (params.c4, params.t4),
        ];

        // Simple greedy match for the 3 others
        let mut matched_count = 0;
        let mut def_used = [false; 3];
        for (tc, tt) in target_others {
            for i in 0..3 {
                if !def_used[i] && matches(def_others[i], tc, tt) {
                    def_used[i] = true;
                    matched_count += 1;
                    break;
                }
            }
        }

        if matched_count == 3 {
            let has_wildcard = t.class1.is_empty()
                || t.class1 == "X"
                || t.class2.is_empty()
                || t.class2 == "X"
                || t.class3.is_empty()
                || t.class3 == "X"
                || t.class4.is_empty()
                || t.class4 == "X";

            if best_match.is_none() || !has_wildcard {
                best_match = Some(t);
            }
            if !has_wildcard {
                break;
            }
        }
    }
    best_match
}

#[allow(clippy::too_many_arguments)]
fn resolve_14_params(
    pairs_14: &[[usize; 2]],
    charges: &[f32],
    sigmas: &[f32],
    epsilons: &[f32],
    atom_types: &[String],
    exceptions: &[NonbondedException],
    lj14scale: f32,
    coulomb14scale: f32,
) -> Vec<[f32; 3]> {
    pairs_14
        .iter()
        .map(|&[i, j]| {
            let ti = &atom_types[i];
            let tj = &atom_types[j];
            if let Some(exc) = exceptions
                .iter()
                .find(|e| (&e.type1 == ti && &e.type2 == tj) || (&e.type1 == tj && &e.type2 == ti))
            {
                [exc.charge_prod, exc.sigma, exc.epsilon]
            } else {
                let charge_prod = coulomb14scale * charges[i] * charges[j];
                let sigma = 0.5 * (sigmas[i] + sigmas[j]);
                let epsilon = lj14scale * (epsilons[i] * epsilons[j]).sqrt();
                [charge_prod, sigma, epsilon]
            }
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use proxide_core::forcefield::ForceField;
    use proxide_core::structure::{AtomRecord, RawAtomData};

    fn make_test_forcefield() -> ForceField {
        let mut ff = ForceField::new("test".to_string());

        // Add GBSA param
        ff.gbsa_obc_params.push(GBSAOBCParam {
            atom_type: "N".to_string(),
            radius: 0.15,
            scale: 0.8,
        });

        // Add a simple ALA template
        ff.residue_templates
            .push(proxide_core::forcefield::ResidueTemplate {
                name: "ALA".to_string(),
                atoms: vec![
                    proxide_core::forcefield::ResidueAtom {
                        name: "N".to_string(),
                        atom_type: "N".to_string(),
                        charge: Some(-0.4157),
                    },
                    proxide_core::forcefield::ResidueAtom {
                        name: "CA".to_string(),
                        atom_type: "CX".to_string(),
                        charge: Some(0.0337),
                    },
                    proxide_core::forcefield::ResidueAtom {
                        name: "C".to_string(),
                        atom_type: "C".to_string(),
                        charge: Some(0.5973),
                    },
                    proxide_core::forcefield::ResidueAtom {
                        name: "N2".to_string(),
                        atom_type: "N".to_string(),
                        charge: Some(-0.4157),
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

        let coords_slice: &[[f32; 3]] = bytemuck::cast_slice(&structure.raw_atoms.coords);
        let topology = proxide_geometry::geometry::topology::generate_topology(
            coords_slice,
            &structure.raw_atoms.elements,
            1.3,
        );

        let params = parameterize_structure(&structure, &topology, &ff, &options).unwrap();

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
            ..Default::default()
        };

        let coords_slice: &[[f32; 3]] = bytemuck::cast_slice(&structure.raw_atoms.coords);
        let topology = proxide_geometry::geometry::topology::generate_topology(
            coords_slice,
            &structure.raw_atoms.elements,
            1.3,
        );

        let params = parameterize_structure(&structure, &topology, &ff, &options).unwrap();

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
            ..Default::default()
        };

        let coords_slice: &[[f32; 3]] = bytemuck::cast_slice(&structure.raw_atoms.coords);
        let topology = proxide_geometry::geometry::topology::generate_topology(
            coords_slice,
            &structure.raw_atoms.elements,
            1.3,
        );

        let result = parameterize_structure(&structure, &topology, &ff, &options);
        assert!(result.is_err());
    }

    #[test]
    fn test_dihedral_and_improper_params() {
        let mut ff = make_test_forcefield();

        // Add Proper Torsion
        ff.proper_torsions
            .push(proxide_core::forcefield::ProperTorsionParam {
                class1: "N".to_string(),
                class2: "CX".to_string(),
                class3: "C".to_string(),
                class4: "N".to_string(),
                terms: vec![proxide_core::forcefield::TorsionTerm {
                    periodicity: 3,
                    phase: 0.0,
                    k: 1.5,
                }],
            });

        // Add Improper
        ff.improper_torsions
            .push(proxide_core::forcefield::ImproperTorsionParam {
                class1: "N".to_string(),
                class2: "N".to_string(),
                class3: "CX".to_string(), // Center
                class4: "C".to_string(),
                terms: vec![proxide_core::forcefield::TorsionTerm {
                    periodicity: 2,
                    phase: 3.14159,
                    k: 10.0,
                }],
            });

        // Set up 4-atom linear topology: N-CA-C-N
        let mut raw = RawAtomData::with_capacity(4);
        for (i, name) in ["N", "CA", "C", "N"].iter().enumerate() {
            raw.add_atom(AtomRecord {
                serial: (i + 1) as i32,
                atom_name: name.to_string(),
                res_name: "ALA".to_string(),
                chain_id: "A".to_string(),
                res_seq: 1,
                x: i as f32,
                y: 0.0,
                z: 0.0,
                element: (if *name == "N" { "N" } else { "C" }).to_string(),
                ..AtomRecord::default()
            });
        }
        let structure = ProcessedStructure::from_raw(raw).unwrap();

        let bonds = vec![
            proxide_core::forcefield::Bond::new(0, 1),
            proxide_core::forcefield::Bond::new(1, 2),
            proxide_core::forcefield::Bond::new(2, 3),
        ];
        let elements = vec![
            "N".to_string(),
            "C".to_string(),
            "C".to_string(),
            "N".to_string(),
        ];
        let mut topology = proxide_core::forcefield::Topology::new(bonds, &elements);

        // Dihedral N-CA-C-N
        topology
            .proper_dihedrals
            .push(proxide_core::forcefield::Dihedral::new_proper(0, 1, 2, 3));

        let options = ParamOptions::default();
        let params = parameterize_structure(&structure, &topology, &ff, &options).unwrap();

        assert!(!params.dihedral_params.is_empty());
        assert!(!params.pairs_14.is_empty()); // Check 1-4 exception scaling logic
    }

    #[test]
    fn test_cmap_params() {
        let mut ff = make_test_forcefield();
        ff.cmap_data = Some(proxide_core::forcefield::CMAPData {
            torsions: vec![proxide_core::forcefield::CMAPTorsion {
                class1: "N".to_string(),
                type2: "CX".to_string(),
                type3: "C".to_string(),
                type4: "N".to_string(),
                class5: "CX".to_string(),
                map_index: 0,
            }],
            maps: vec![proxide_core::forcefield::CMAPGrid {
                size: 24,
                energies: vec![0.0; 24 * 24],
            }],
        });

        let structure = make_test_structure();
        let bonds = vec![
            proxide_core::forcefield::Bond::new(0, 1),
            proxide_core::forcefield::Bond::new(1, 2),
        ];
        let elements = vec!["N".to_string(), "C".to_string(), "C".to_string()];
        let topology = proxide_core::forcefield::Topology::new(bonds, &elements);
        let options = ParamOptions::default();

        let _ = parameterize_structure(&structure, &topology, &ff, &options).unwrap();
    }

    #[test]
    fn test_missing_residue_closest_match_stub() {
        let ff = ForceField::new("empty".to_string());
        let structure = make_test_structure();
        let options = ParamOptions {
            auto_terminal_caps: false,
            missing_mode: MissingResidueMode::ClosestMatch,
            ..Default::default()
        };

        let coords_slice: &[[f32; 3]] = bytemuck::cast_slice(&structure.raw_atoms.coords);
        let topology = proxide_geometry::geometry::topology::generate_topology(
            coords_slice,
            &structure.raw_atoms.elements,
            1.3,
        );

        let params = parameterize_structure(&structure, &topology, &ff, &options).unwrap();
        // Should fall back to elements since it's a stub
        assert_eq!(params.num_skipped, 3);
    }

    #[test]
    fn test_param_error_display() {
        assert!(format!("{}", ParamError::MissingTemplate("FOO".into())).contains("FOO"));
        // Using _ to suppress warning and still call Display for coverage
        let _ = format!("{}", ParamError::_MissingAtom("RES".into(), "ATM".into()));
        let _ = format!("{}", ParamError::_MissingNonbonded("TYPE".into()));
    }

    #[test]
    fn test_missing_template_fail_message() {
        let ff = ForceField::new("empty".to_string());
        let structure = make_test_structure();
        let options = ParamOptions {
            auto_terminal_caps: false,
            missing_mode: MissingResidueMode::Fail,
            ..Default::default()
        };

        let coords_slice: &[[f32; 3]] = bytemuck::cast_slice(&structure.raw_atoms.coords);
        let topology = proxide_geometry::geometry::topology::generate_topology(
            coords_slice,
            &structure.raw_atoms.elements,
            1.3,
        );

        let result = parameterize_structure(&structure, &topology, &ff, &options);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("ALA"));
    }

    #[test]
    fn test_terminal_caps_last_only() {
        let mut ff = make_test_forcefield();
        // Add CALA
        ff.residue_templates
            .push(proxide_core::forcefield::ResidueTemplate {
                name: "CALA".to_string(),
                atoms: vec![
                    proxide_core::forcefield::ResidueAtom {
                        name: "N".to_string(),
                        atom_type: "N".to_string(),
                        charge: Some(-0.6),
                    },
                    proxide_core::forcefield::ResidueAtom {
                        name: "CA".to_string(),
                        atom_type: "CX".to_string(),
                        charge: Some(0.01),
                    },
                    proxide_core::forcefield::ResidueAtom {
                        name: "C".to_string(),
                        atom_type: "C".to_string(),
                        charge: Some(0.5),
                    },
                ],
                bonds: vec![],
                external_bonds: vec![],
                override_level: None,
            });
        ff.build_indices();

        let mut raw = RawAtomData::new();
        // Chain A: 2 residues.
        // Residue 1: N, CA, C (ALA)
        // Residue 2: N, CA, C (ALA) -> Should be CALA
        raw.add_atom(AtomRecord {
            serial: 1,
            atom_name: "N".to_string(),
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            element: "N".to_string(),
            ..AtomRecord::default()
        });
        raw.add_atom(AtomRecord {
            serial: 2,
            atom_name: "CA".to_string(),
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            x: 1.0,
            y: 0.0,
            z: 0.0,
            element: "C".to_string(),
            ..AtomRecord::default()
        });
        raw.add_atom(AtomRecord {
            serial: 3,
            atom_name: "C".to_string(),
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            x: 2.0,
            y: 0.0,
            z: 0.0,
            element: "C".to_string(),
            ..AtomRecord::default()
        });

        raw.add_atom(AtomRecord {
            serial: 4,
            atom_name: "N".to_string(),
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 2,
            x: 10.0,
            y: 0.0,
            z: 0.0,
            element: "N".to_string(),
            ..AtomRecord::default()
        });
        raw.add_atom(AtomRecord {
            serial: 5,
            atom_name: "CA".to_string(),
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 2,
            x: 11.0,
            y: 0.0,
            z: 0.0,
            element: "C".to_string(),
            ..AtomRecord::default()
        });
        raw.add_atom(AtomRecord {
            serial: 6,
            atom_name: "C".to_string(),
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 2,
            x: 12.0,
            y: 0.0,
            z: 0.0,
            element: "C".to_string(),
            ..AtomRecord::default()
        });

        let structure = ProcessedStructure::from_raw(raw).unwrap();
        let topology = proxide_core::forcefield::Topology::new(
            vec![],
            &[
                "N".to_string(),
                "C".to_string(),
                "C".to_string(),
                "N".to_string(),
                "C".to_string(),
                "C".to_string(),
            ],
        );
        let options = ParamOptions {
            auto_terminal_caps: true,
            missing_mode: MissingResidueMode::SkipWarn,
            ..Default::default()
        };

        let params = parameterize_structure(&structure, &topology, &ff, &options).unwrap();

        // Residue 2 (atoms 3,4,5): CALA
        assert_eq!(params.charges[3], -0.6);
        assert_eq!(params.charges[4], 0.01);
    }

    #[test]
    fn test_parameterize_molecule_error() {
        let elements = vec!["O".to_string()];
        let coords = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let res = parameterize_molecule(&coords, &elements, 1.3);
        assert!(res.is_err());
        assert!(format!("{}", res.unwrap_err()).contains("mismatch"));
    }

    /// One ALA protein residue (atoms 0,1,2; far from the waters) plus two
    /// TIP3P-geometry `HOH` water molecules (atoms 3,4,5 and 6,7,8), well
    /// separated from each other and from the protein so `infer_bonds` can't
    /// spuriously connect them. `is_hetatm: true` on the water atoms is
    /// required: solvent classification only triggers on HETATM records
    /// (see `ProcessedStructure::from_raw_with_config`).
    fn make_water_structure() -> ProcessedStructure {
        let mut raw = RawAtomData::with_capacity(9);

        for (i, name) in ["N", "CA", "C"].iter().enumerate() {
            raw.add_atom(AtomRecord {
                serial: (i + 1) as i32,
                atom_name: name.to_string(),
                res_name: "ALA".to_string(),
                chain_id: "A".to_string(),
                res_seq: 1,
                x: i as f32,
                y: 100.0,
                z: 0.0,
                element: (if *name == "N" { "N" } else { "C" }).to_string(),
                ..AtomRecord::default()
            });
        }

        // r(OH) = 0.9572 A, theta(HOH) = 104.52 deg -- matches the TIP3P
        // model geometry documented on `water::tip3p()`.
        let water_geom: [[f32; 3]; 3] = [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2397, 0.9266, 0.0],
        ];
        let names = ["O", "H1", "H2"];
        let elements = ["O", "H", "H"];
        for water_idx in 0..2usize {
            let offset_x = water_idx as f32 * 50.0;
            for atom_idx in 0..3usize {
                raw.add_atom(AtomRecord {
                    serial: (10 + water_idx * 3 + atom_idx) as i32,
                    atom_name: names[atom_idx].to_string(),
                    res_name: "HOH".to_string(),
                    chain_id: "W".to_string(),
                    res_seq: 100 + water_idx as i32,
                    x: water_geom[atom_idx][0] + offset_x,
                    y: water_geom[atom_idx][1],
                    z: water_geom[atom_idx][2],
                    element: elements[atom_idx].to_string(),
                    is_hetatm: true,
                    ..AtomRecord::default()
                });
            }
        }

        ProcessedStructure::from_raw(raw).unwrap()
    }

    #[test]
    fn test_water_oh_bonds_locked() {
        // Locks pre-existing, correct behavior (independent of the solvent
        // parameterization fix): geometric bond inference already finds
        // exactly the 2 O-H bonds per water molecule, with no spurious H-H
        // or cross-molecule bonds, because it works on distance + covalent
        // radii and never consults `residue_info`/`molecule_type` at all.
        let structure = make_water_structure();
        assert_eq!(structure.solvent_atoms.len(), 6); // 2 waters x 3 atoms

        let coords_slice: &[[f32; 3]] = bytemuck::cast_slice(&structure.raw_atoms.coords);
        let topology = proxide_geometry::geometry::topology::generate_topology(
            coords_slice,
            &structure.raw_atoms.elements,
            1.3,
        );

        let water_set: HashSet<usize> = structure.solvent_atoms.iter().copied().collect();
        let water_bonds: Vec<_> = topology
            .bonds
            .iter()
            .filter(|b| water_set.contains(&b.i) && water_set.contains(&b.j))
            .collect();
        assert_eq!(water_bonds.len(), 4); // 2 O-H bonds x 2 waters
    }

    #[test]
    fn test_parameterize_solvent_water_tip3p() {
        // Reproduces the diagnosed defect: before the fix, solvent atoms
        // were excluded from `residue_info` and therefore invisible to this
        // function's residue-template loop, so charges/sigmas/epsilons
        // stayed at 0.0 for every water atom no matter what force field was
        // supplied. This asserts the fixed behavior: nonzero charges and a
        // nonzero oxygen epsilon, matching the TIP3P model converted from
        // its native AMBER units (Angstrom, kcal/mol) to the nm/kJ-mol
        // convention used everywhere else in `MDParameters`.
        let ff = make_test_forcefield();
        let structure = make_water_structure();
        let options = ParamOptions::default(); // water_model defaults to "TIP3P"

        let coords_slice: &[[f32; 3]] = bytemuck::cast_slice(&structure.raw_atoms.coords);
        let topology = proxide_geometry::geometry::topology::generate_topology(
            coords_slice,
            &structure.raw_atoms.elements,
            1.3,
        );

        let params = parameterize_structure(&structure, &topology, &ff, &options).unwrap();

        // Atom order: 0,1,2 = ALA N/CA/C; 3,4,5 = water 1 O/H1/H2.
        let o1 = 3;
        let h1a = 4;
        let h1b = 5;

        assert!(
            (params.charges[o1] - (-0.834)).abs() < 1e-4,
            "O charge: {}",
            params.charges[o1]
        );
        assert!((params.charges[h1a] - 0.417).abs() < 1e-4);
        assert!((params.charges[h1b] - 0.417).abs() < 1e-4);

        let expected_sigma_nm = 3.150_61 * ANGSTROM_TO_NM;
        let expected_epsilon_kj = 0.1521 * KCAL_TO_KJ;
        assert!(
            (params.sigmas[o1] - expected_sigma_nm).abs() < 1e-4,
            "O sigma (nm): {}",
            params.sigmas[o1]
        );
        assert!(
            (params.epsilons[o1] - expected_epsilon_kj).abs() < 1e-4,
            "O epsilon (kJ/mol): {}",
            params.epsilons[o1]
        );
        // TIP3P hydrogens carry zero LJ epsilon by design (not to be
        // confused with the "never visited" zero this whole fix addresses).
        assert_eq!(params.epsilons[h1a], 0.0);
        assert_eq!(params.epsilons[h1b], 0.0);

        // Every water atom got parameterized -- none of them are flagged in
        // the unparameterized-atom report.
        for &idx in &structure.solvent_atoms {
            assert!(
                !params.unparameterized_atoms.contains(&idx),
                "water atom {idx} should not be reported as unparameterized"
            );
        }

        // Sanity: the ordinary protein path is untouched by the solvent fix.
        assert!((params.charges[0] - (-0.4157)).abs() < 1e-4); // ALA N
    }

    #[test]
    fn test_pairs_14_excludes_spurious_cross_water_bond() {
        // Bug 2 (task 260909_dhfr_gap_tranche2, proxide side of a prolix-reported
        // anomaly): a solvated 1VII system (7507 atoms, only 596 of them
        // protein) reported ~1563-1564 `pairs_14` entries, reproduced across
        // separate runs -- implausibly high for a 596-atom protein alone, and
        // definitely wrong insofar as it includes any water atom at all
        // (water has zero real 1-4 pairs in every standard water model,
        // independent of any protein-side count) -- and at least one
        // prolix-reported pair referenced an atom index that didn't exist in
        // the 7507-atom system. (A same-run OpenMM-side "~40 exceptions"
        // comparison baseline was later retracted as unreliable -- an
        // unrelated OpenMM/SWIG binding issue, prolix backlog #5055 -- so
        // it's deliberately not relied on here.) Root cause: `topology`
        // is built by pure distance + covalent-radius bond inference over the
        // WHOLE system at once (`infer_bonds`, no residue-boundary or
        // force-field-template awareness), so two DIFFERENT water molecules
        // placed close enough together get a spurious inter-molecular "bond" --
        // and, before the fix, `pairs_14` was built directly from raw geometric
        // `topology.proper_dihedrals` with no check that the resulting chain
        // stayed within a single, real, force-field-bonded molecule. Since water
        // has zero real torsions in every standard water model, ANY water-
        // touching entry in `pairs_14` is definitionally spurious.
        //
        // This test manufactures exactly that spurious bond (water 1's H2 placed
        // 1.0 A from water 2's O -- well inside the O-H bonding threshold of
        // (0.66+0.31)*1.3 = 1.261 A -- while every other inter-water distance
        // stays outside any bonding threshold) and asserts:
        //   1. the spurious bond really does form (sanity: the mechanism fires);
        //   2. it really does create a geometric proper dihedral entirely among
        //      water atoms (sanity: the dihedral-generation step is exercised);
        //   3. `parameterize_structure`'s `pairs_14` contains NO entry touching
        //      any solvent atom (the actual fix -- this would have failed
        //      before it, since the raw geometric dihedral from point 2 would
        //      have been pushed straight into `pairs_14`).
        let ff = make_test_forcefield();
        let mut raw = RawAtomData::with_capacity(6);

        // Water 1: standard TIP3P geometry at the origin.
        let water1_geom: [[f32; 3]; 3] = [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2397, 0.9266, 0.0],
        ];
        // Water 2: translated so its O sits exactly 1.0 A from water 1's H2
        // (spurious-bond distance), with every other cross-molecule pair kept
        // outside any bonding threshold (see test docstring for the arithmetic).
        let water2_origin = [-0.2397_f32, 1.9266_f32, 0.0_f32];
        let water2_geom: [[f32; 3]; 3] = [
            water2_origin,
            [water2_origin[0] + 0.9572, water2_origin[1], water2_origin[2]],
            [
                water2_origin[0] - 0.2397,
                water2_origin[1] + 0.9266,
                water2_origin[2],
            ],
        ];
        let names = ["O", "H1", "H2"];
        let elements = ["O", "H", "H"];
        for (water_idx, geom) in [water1_geom, water2_geom].iter().enumerate() {
            for atom_idx in 0..3usize {
                raw.add_atom(AtomRecord {
                    serial: (water_idx * 3 + atom_idx + 1) as i32,
                    atom_name: names[atom_idx].to_string(),
                    res_name: "HOH".to_string(),
                    chain_id: "W".to_string(),
                    res_seq: 100 + water_idx as i32,
                    x: geom[atom_idx][0],
                    y: geom[atom_idx][1],
                    z: geom[atom_idx][2],
                    element: elements[atom_idx].to_string(),
                    is_hetatm: true,
                    ..AtomRecord::default()
                });
            }
        }

        let structure = ProcessedStructure::from_raw(raw).unwrap();
        assert_eq!(structure.solvent_atoms.len(), 6);
        let solvent_set: HashSet<usize> = structure.solvent_atoms.iter().copied().collect();

        let coords_slice: &[[f32; 3]] = bytemuck::cast_slice(&structure.raw_atoms.coords);
        let topology = proxide_geometry::geometry::topology::generate_topology(
            coords_slice,
            &structure.raw_atoms.elements,
            1.3,
        );

        // Sanity 1: the spurious cross-molecule bond (water1 H2 [index 2] --
        // water2 O [index 3]) really did form.
        let has_spurious_bond = topology
            .bonds
            .iter()
            .any(|b| (b.i == 2 && b.j == 3) || (b.i == 3 && b.j == 2));
        assert!(
            has_spurious_bond,
            "expected the manufactured close contact to be inferred as a bond; \
             got bonds: {:?}",
            topology.bonds
        );

        // Sanity 2: that spurious bond really does create a geometric proper
        // dihedral entirely among water atoms (all 4 chain atoms in
        // solvent_set).
        let has_all_water_dihedral = topology.proper_dihedrals.iter().any(|d| {
            solvent_set.contains(&d.i)
                && solvent_set.contains(&d.j)
                && solvent_set.contains(&d.k)
                && solvent_set.contains(&d.l)
        });
        assert!(
            has_all_water_dihedral,
            "expected the spurious bond to produce an all-water geometric \
             dihedral; got dihedrals: {:?}",
            topology.proper_dihedrals
        );

        // The actual fix: parameterize_structure must never surface a pairs_14
        // entry touching a solvent atom, no matter what spurious geometric
        // dihedral the raw distance-based topology contains.
        let options = ParamOptions::default();
        let params = parameterize_structure(&structure, &topology, &ff, &options).unwrap();
        for pair in &params.pairs_14 {
            assert!(
                !solvent_set.contains(&pair[0]) && !solvent_set.contains(&pair[1]),
                "pairs_14 must never include a solvent atom (spurious 1-4 \
                 exception), got pair {:?}",
                pair
            );
        }
    }

    #[test]
    fn test_out_of_range_bond_index_is_an_error() {
        let ff = make_test_forcefield();
        let mut raw = RawAtomData::with_capacity(2);
        raw.add_atom(AtomRecord {
            serial: 1,
            atom_name: "CA".to_string(),
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            element: "C".to_string(),
            is_hetatm: false,
            ..AtomRecord::default()
        });
        raw.add_atom(AtomRecord {
            serial: 2,
            atom_name: "N".to_string(),
            res_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            res_seq: 1,
            x: 1.5,
            y: 0.0,
            z: 0.0,
            element: "N".to_string(),
            is_hetatm: false,
            ..AtomRecord::default()
        });
        let structure = ProcessedStructure::from_raw(raw).unwrap();
        let n_atoms = structure.raw_atoms.num_atoms;

        let coords_slice: &[[f32; 3]] = bytemuck::cast_slice(&structure.raw_atoms.coords);
        let mut topology = proxide_geometry::geometry::topology::generate_topology(
            coords_slice,
            &structure.raw_atoms.elements,
            1.3,
        );
        topology.bonds.push(proxide_core::forcefield::Bond {
            i: n_atoms,
            j: 0,
        });

        let options = ParamOptions::default();
        let result = parameterize_structure(&structure, &topology, &ff, &options);

        match result {
            Err(ParamError::TopologyIndexOutOfRange {
                term,
                indices,
                n_atoms: na,
            }) => {
                assert_eq!(term, "bond");
                assert!(indices.contains(&n_atoms));
                assert_eq!(na, n_atoms);
            }
            _ => panic!(
                "Expected TopologyIndexOutOfRange error for bond, got: {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_out_of_range_proper_index_is_an_error() {
        let ff = make_test_forcefield();
        let mut raw = RawAtomData::with_capacity(4);
        for i in 0..4 {
            raw.add_atom(AtomRecord {
                serial: (i + 1) as i32,
                atom_name: format!("A{}", i),
                res_name: "ALA".to_string(),
                chain_id: "A".to_string(),
                res_seq: 1,
                x: i as f32 * 1.5,
                y: 0.0,
                z: 0.0,
                element: if i == 0 || i == 3 { "C" } else { "N" }.to_string(),
                is_hetatm: false,
                ..AtomRecord::default()
            });
        }
        let structure = ProcessedStructure::from_raw(raw).unwrap();
        let n_atoms = structure.raw_atoms.num_atoms;

        let coords_slice: &[[f32; 3]] = bytemuck::cast_slice(&structure.raw_atoms.coords);
        let mut topology = proxide_geometry::geometry::topology::generate_topology(
            coords_slice,
            &structure.raw_atoms.elements,
            1.3,
        );
        topology.proper_dihedrals.push(proxide_core::forcefield::Dihedral {
            i: 0,
            j: 1,
            k: 2,
            l: n_atoms + 5,
            is_improper: false,
        });

        let options = ParamOptions::default();
        let result = parameterize_structure(&structure, &topology, &ff, &options);

        match result {
            Err(ParamError::TopologyIndexOutOfRange {
                term,
                indices,
                n_atoms: na,
            }) => {
                assert_eq!(term, "proper");
                assert!(indices.contains(&(n_atoms + 5)));
                assert_eq!(na, n_atoms);
            }
            _ => panic!(
                "Expected TopologyIndexOutOfRange error for proper dihedral, got: {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_unparameterized_report_and_strict_mode() {
        // Ligand atoms are, structurally, in exactly the same boat solvent
        // was before this fix: excluded from `residue_info`, so
        // `parameterize_structure` never visits them at all (ligand
        // parameterization is a separate opt-in path -- see
        // `parameterize_molecule` / py_chemistry). This checks the new
        // safety net (task B) catches that silently-zeroed case generically,
        // not just for water.
        let ff = make_test_forcefield();
        let mut raw = RawAtomData::with_capacity(2);
        for (i, name) in ["C1", "C2"].iter().enumerate() {
            raw.add_atom(AtomRecord {
                serial: (i + 1) as i32,
                atom_name: name.to_string(),
                res_name: "LIG".to_string(),
                chain_id: "L".to_string(),
                res_seq: 1,
                x: i as f32 * 1.5,
                y: 200.0,
                z: 0.0,
                element: "C".to_string(),
                is_hetatm: true,
                ..AtomRecord::default()
            });
        }
        let structure = ProcessedStructure::from_raw(raw).unwrap();
        assert_eq!(structure.ligand_groups.len(), 1);
        assert!(structure.residue_info.is_empty());

        let coords_slice: &[[f32; 3]] = bytemuck::cast_slice(&structure.raw_atoms.coords);
        let topology = proxide_geometry::geometry::topology::generate_topology(
            coords_slice,
            &structure.raw_atoms.elements,
            1.3,
        );

        // Lenient (default) mode: still returns Ok, but now says so instead
        // of staying silent.
        let lenient_options = ParamOptions::default();
        let params =
            parameterize_structure(&structure, &topology, &ff, &lenient_options).unwrap();
        assert_eq!(params.unparameterized_atoms.len(), 2);
        assert!(params.unparameterized_atoms.contains(&0));
        assert!(params.unparameterized_atoms.contains(&1));
        assert_eq!(params.charges[0], 0.0);

        // Strict mode: the exact same input now errors instead of quietly
        // returning zeros.
        let strict_options = ParamOptions {
            strict: true,
            ..Default::default()
        };
        let result = parameterize_structure(&structure, &topology, &ff, &strict_options);
        match result {
            Err(ParamError::UnparameterizedAtoms(n)) => assert_eq!(n, 2),
            other => panic!("expected UnparameterizedAtoms(2), got {other:?}"),
        }
    }
}
