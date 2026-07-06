//! Residue processing and grouping
//!
//! Groups atoms by residue and builds residue-level metadata
//!
//! Note: Some fields are used for debugging/future features.

#![allow(dead_code)]

use crate::chem::{
    build_resname_to_idx, chem_comp_type, is_peptide_linking_type, resolve_variant_alias,
    AliasMatchConfig, ResidueNamingConvention, UNK_RESTYPE_INDEX,
};
use crate::spec::OutputFormatTarget;
use crate::structure::RawAtomData;
use std::collections::{HashMap, HashSet};

// Using hardcoded constants for now to avoid circular dependencies during the refactor.
// They should be moved to a shared 'constants' crate eventually.
const DEFAULT_SIGMA: f32 = 3.5;
const DEFAULT_EPSILON: f32 = 0.1;

/// Residue identifier (unique combination of chain, resid, insertion code)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResidueId {
    pub chain_id: String,
    pub res_id: i32,
    pub insertion_code: char,
}

/// Information about a single residue
#[derive(Debug, Clone)]
pub struct ResidueInfo {
    pub res_id: i32,
    pub res_name: String,
    pub res_type: usize, // Index into residue types (0-20)
    pub chain_id: String,
    pub insertion_code: char,
    pub start_atom: usize, // Index of first atom
    pub num_atoms: usize,  // Number of atoms in this residue
}

/// Information about a ligand
#[derive(Debug, Clone)]
pub struct LigandInfo {
    pub res_name: String,
    pub atom_indices: Vec<usize>,
    pub chain_id: String,
    pub res_id: i32,
}

/// Configuration for how [`ProcessedStructure::from_raw_with_config`]
/// classifies residues whose name matches a known CHARMM/AMBER
/// protonation-variant/disulfide-state alias (e.g. `HSD`, `CYX`). See
/// [`crate::chem::ResidueNamingConvention`] for the tradeoff between the two
/// modes -- the default (`Standard` + strict atom-composition check) is safe
/// for files of unknown provenance.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidueClassificationConfig {
    pub naming_convention: ResidueNamingConvention,
    pub alias_match: AliasMatchConfig,
}

/// Processed structure with residue-level organization
#[derive(Debug)]
pub struct ProcessedStructure {
    pub raw_atoms: RawAtomData,
    pub residue_info: Vec<ResidueInfo>,
    pub chain_indices: HashMap<String, usize>, // chain_id -> index

    // HETATM components
    pub ligand_groups: Vec<LigandInfo>,
    pub solvent_atoms: Vec<usize>,
    pub ion_atoms: Vec<usize>,

    // Per-atom molecule type: 0=protein, 1=ligand, 2=solvent, 3=ion
    pub molecule_type: Vec<i8>,

    pub num_residues: usize,
    pub num_chains: usize,
}

impl ProcessedStructure {
    /// Create a ProcessedStructure from RawAtomData, using the default
    /// residue-classification config (`Standard` naming convention, strict
    /// atom-composition check). See [`Self::from_raw_with_config`] to opt into
    /// `ForceFieldPrepped` mode for files of known simulation-tool provenance.
    pub fn from_raw(raw_atoms: RawAtomData) -> Result<Self, String> {
        Self::from_raw_with_config(raw_atoms, ResidueClassificationConfig::default())
    }

    /// Create a ProcessedStructure from RawAtomData with an explicit
    /// residue-classification config. See [`ResidueClassificationConfig`].
    pub fn from_raw_with_config(
        raw_atoms: RawAtomData,
        config: ResidueClassificationConfig,
    ) -> Result<Self, String> {
        if raw_atoms.num_atoms == 0 {
            return Err("No atoms in structure".to_string());
        }

        // Build residue ID to atom indices mapping
        let mut residue_atoms: HashMap<ResidueId, Vec<usize>> = HashMap::new();

        for i in 0..raw_atoms.num_atoms {
            let res_id = ResidueId {
                chain_id: raw_atoms.chain_ids[i].clone(),
                res_id: raw_atoms.res_ids[i],
                insertion_code: raw_atoms.insertion_codes[i],
            };

            residue_atoms.entry(res_id).or_default().push(i);
        }

        // Sort residues by chain and residue number
        let mut residue_ids: Vec<_> = residue_atoms.keys().cloned().collect();
        residue_ids.sort_by(|a, b| {
            a.chain_id
                .cmp(&b.chain_id)
                .then(a.res_id.cmp(&b.res_id))
                .then(a.insertion_code.cmp(&b.insertion_code))
        });

        // Initialize classification containers
        let mut residue_info = Vec::with_capacity(residue_ids.len());
        let mut ligand_groups = Vec::new();
        let mut solvent_atoms = Vec::new();
        let mut ion_atoms = Vec::new();
        let mut molecule_type = vec![0i8; raw_atoms.num_atoms];

        let resname_to_idx = build_resname_to_idx();
        // Common solvent residue names
        let solvent_names = ["HOH", "WAT", "TIP3", "SOL", "DOD"];

        for res_id in &residue_ids {
            let atom_indices = &residue_atoms[res_id];
            let start_atom = atom_indices[0];
            let res_name = &raw_atoms.res_names[start_atom];
            let is_hetatm = raw_atoms.is_hetatm[start_atom];

            // Observed heavy atom names (element != "H") for this residue
            // instance, used by the CHARMM/AMBER alias-resolution
            // atom-composition check below. Computed regardless of is_hetatm,
            // since a variant name (e.g. HSD) can in principle arrive under
            // either record type.
            let observed_heavy_atoms: HashSet<&str> = atom_indices
                .iter()
                .filter(|&&idx| raw_atoms.elements[idx] != "H")
                .map(|&idx| raw_atoms.atom_names[idx].as_str())
                .collect();

            // Try the two "this is really a protein residue" resolution paths
            // BEFORE trusting the raw is_hetatm flag, so a residue like the
            // CHARMM/AMBER protonation-variant `HSD` (which real-world QM/MM
            // structure-prep tools frequently tag HETATM) is recognized as
            // HIS rather than silently dropped into ligand_groups:
            //   1. `resolve_variant_alias` (see chem::residues) -- known
            //      simulation-tool naming conventions, atom-verified by
            //      default (or trusted unconditionally under
            //      ForceFieldPrepped, per `config`).
            //   2. Genuine CCD (`chem::ccd`) peptide-linking type -- catches
            //      real, unambiguous non-canonical residues (MSE, D-amino
            //      acids, etc.) regardless of the file's HETATM flag.
            let alias_resolved = resolve_variant_alias(
                res_name,
                &observed_heavy_atoms,
                config.naming_convention,
                &config.alias_match,
            );
            let ccd_is_protein = chem_comp_type(res_name)
                .map(is_peptide_linking_type)
                .unwrap_or(false);

            // Name to use for res_type lookup: the alias's canonical parent if
            // resolved; otherwise the residue's own (verbatim) name, but ONLY
            // if it's a plain ATOM record or a genuine CCD-recognized protein
            // residue -- this preserves today's exact behavior for ordinary
            // ATOM records and un-recognized HETATM records alike.
            let type_lookup_name = alias_resolved
                .or_else(|| (!is_hetatm || ccd_is_protein).then_some(res_name.as_str()));

            if let Some(type_lookup_name) = type_lookup_name {
                // Protein residue.
                for &idx in atom_indices {
                    molecule_type[idx] = 0;
                }

                let res_type = resname_to_idx
                    .get(type_lookup_name)
                    .copied()
                    .unwrap_or(UNK_RESTYPE_INDEX);

                residue_info.push(ResidueInfo {
                    res_id: res_id.res_id,
                    // Preserve the ORIGINAL, as-read name (e.g. "HSD"), not
                    // the resolved canonical parent -- downstream consumers
                    // that care about protonation/disulfide state (e.g.
                    // force-field parameterization) must not lose it.
                    res_name: res_name.clone(),
                    res_type,
                    chain_id: res_id.chain_id.clone(),
                    insertion_code: res_id.insertion_code,
                    start_atom,
                    num_atoms: atom_indices.len(),
                });
            } else {
                // HETATM Processing
                let is_solvent = solvent_names.contains(&res_name.as_str());

                if is_solvent {
                    // Solvent (2)
                    for &idx in atom_indices {
                        molecule_type[idx] = 2;
                        solvent_atoms.push(idx);
                    }
                } else if atom_indices.len() == 1 && res_name.len() <= 2 {
                    // Heuristic: single atom, short name (e.g. "CL", "ZN") -> Ion (3)
                    let idx = atom_indices[0];
                    molecule_type[idx] = 3;
                    ion_atoms.push(idx);
                } else {
                    // Ligand (1)
                    for &idx in atom_indices {
                        molecule_type[idx] = 1;
                    }

                    ligand_groups.push(LigandInfo {
                        res_name: res_name.clone(),
                        atom_indices: atom_indices.clone(),
                        chain_id: res_id.chain_id.clone(),
                        res_id: res_id.res_id,
                    });
                }
            }
        }

        // Build chain index mapping based on protein residues
        // But also include chains from ligands/solvent if needed?
        // For now, let's keep chain indices mapped to all unique chain IDs encountered
        // in residue_ids (which includes everything)
        let mut chain_ids: Vec<_> = residue_ids.iter().map(|r| r.chain_id.clone()).collect();
        chain_ids.sort();
        chain_ids.dedup();

        let chain_indices: HashMap<String, usize> = chain_ids
            .iter()
            .enumerate()
            .map(|(i, chain)| (chain.clone(), i))
            .collect();

        // Compute counts before moving values
        let num_residues = residue_info.len();
        let num_chains = chain_indices.len();

        log::debug!(
            "Processed structure: {} atoms -> {} residues, {} ligands, {} solvent atoms",
            raw_atoms.num_atoms,
            num_residues,
            ligand_groups.len(),
            solvent_atoms.len()
        );

        Ok(ProcessedStructure {
            raw_atoms,
            residue_info,
            chain_indices,
            ligand_groups,
            solvent_atoms,
            ion_atoms,
            molecule_type,
            num_residues,
            num_chains,
        })
    }

    /// Get atom indices for a specific residue
    pub fn get_residue_atoms(&self, residue_idx: usize) -> Vec<usize> {
        if residue_idx >= self.residue_info.len() {
            return Vec::new();
        }

        let res = &self.residue_info[residue_idx];
        (res.start_atom..res.start_atom + res.num_atoms).collect()
    }

    /// Extract AlphaFold-style backbone coordinates (N, CA, C, CB, O)
    /// Returns (N_res, 5, 3) array. Missing atoms are NaN.
    pub fn extract_backbone_coords(&self, target: OutputFormatTarget) -> Vec<[[f32; 3]; 5]> {
        let mut backbone = vec![[[f32::NAN; 3]; 5]; self.num_residues];

        let (idx_cb, idx_o) = match target {
            OutputFormatTarget::General => (3, 4),
            OutputFormatTarget::Mpnn => (4, 3),
        };

        for (i, res) in self.residue_info.iter().enumerate() {
            for atom_idx in res.start_atom..(res.start_atom + res.num_atoms) {
                let name = &self.raw_atoms.atom_names[atom_idx];
                let coords = [
                    self.raw_atoms.coords[atom_idx * 3],
                    self.raw_atoms.coords[atom_idx * 3 + 1],
                    self.raw_atoms.coords[atom_idx * 3 + 2],
                ];

                // Map to 0..4 indices
                match name.as_str() {
                    "N" => backbone[i][0] = coords,
                    "CA" => backbone[i][1] = coords,
                    "C" => backbone[i][2] = coords,
                    "CB" => backbone[i][idx_cb] = coords,
                    "O" => backbone[i][idx_o] = coords,
                    _ => {}
                }
            }
        }
        backbone
    }

    /// Extract CA coordinates for neighbor search
    /// Returns (N_res, 3). Missing CA is NaN.
    pub fn extract_ca_coords(&self) -> Vec<[f32; 3]> {
        let mut ca_coords = vec![[f32::NAN; 3]; self.num_residues];

        for (i, res) in self.residue_info.iter().enumerate() {
            for atom_idx in res.start_atom..(res.start_atom + res.num_atoms) {
                if self.raw_atoms.atom_names[atom_idx] == "CA" {
                    ca_coords[i] = [
                        self.raw_atoms.coords[atom_idx * 3],
                        self.raw_atoms.coords[atom_idx * 3 + 1],
                        self.raw_atoms.coords[atom_idx * 3 + 2],
                    ];
                    break;
                }
            }
        }
        ca_coords
    }

    /// Extract all atom positions as (N_atoms, 3)
    pub fn extract_all_coords(&self) -> Vec<[f32; 3]> {
        let n = self.raw_atoms.num_atoms;
        let mut coords = Vec::with_capacity(n);
        for i in 0..n {
            coords.push([
                self.raw_atoms.coords[i * 3],
                self.raw_atoms.coords[i * 3 + 1],
                self.raw_atoms.coords[i * 3 + 2],
            ]);
        }
        coords
    }

    /// Extract charges at backbone positions (N_res * 5)
    pub fn extract_backbone_charges(&self, target: OutputFormatTarget) -> Vec<f32> {
        let mut values = vec![0.0f32; self.num_residues * 5];
        let (idx_cb, idx_o) = match target {
            OutputFormatTarget::General => (3, 4),
            OutputFormatTarget::Mpnn => (4, 3),
        };
        if let Some(ref data) = self.raw_atoms.charges {
            for (i, res) in self.residue_info.iter().enumerate() {
                for atom_idx in res.start_atom..(res.start_atom + res.num_atoms) {
                    if atom_idx >= data.len() {
                        continue;
                    }
                    let val = data[atom_idx];
                    match self.raw_atoms.atom_names[atom_idx].as_str() {
                        "N" => values[i * 5] = val,
                        "CA" => values[i * 5 + 1] = val,
                        "C" => values[i * 5 + 2] = val,
                        "CB" => values[i * 5 + idx_cb] = val,
                        "O" => values[i * 5 + idx_o] = val,
                        _ => {}
                    }
                }
            }
        }
        values
    }

    /// Extract sigmas at backbone positions (N_res * 5)
    pub fn extract_backbone_sigmas(&self, target: OutputFormatTarget) -> Vec<f32> {
        // Default sigma? Or 0.0?
        // Using DEFAULT_SIGMA if missing might be safer if used in VdW equation
        let mut values = vec![DEFAULT_SIGMA; self.num_residues * 5];
        let (idx_cb, idx_o) = match target {
            OutputFormatTarget::General => (3, 4),
            OutputFormatTarget::Mpnn => (4, 3),
        };
        if let Some(ref data) = self.raw_atoms.sigmas {
            for (i, res) in self.residue_info.iter().enumerate() {
                for atom_idx in res.start_atom..(res.start_atom + res.num_atoms) {
                    if atom_idx >= data.len() {
                        continue;
                    }
                    let val = data[atom_idx];
                    match self.raw_atoms.atom_names[atom_idx].as_str() {
                        "N" => values[i * 5] = val,
                        "CA" => values[i * 5 + 1] = val,
                        "C" => values[i * 5 + 2] = val,
                        "CB" => values[i * 5 + idx_cb] = val,
                        "O" => values[i * 5 + idx_o] = val,
                        _ => {}
                    }
                }
            }
        }
        values
    }

    /// Extract epsilons at backbone positions (N_res * 5)
    pub fn extract_backbone_epsilons(&self, target: OutputFormatTarget) -> Vec<f32> {
        let mut values = vec![DEFAULT_EPSILON; self.num_residues * 5];
        let (idx_cb, idx_o) = match target {
            OutputFormatTarget::General => (3, 4),
            OutputFormatTarget::Mpnn => (4, 3),
        };
        if let Some(ref data) = self.raw_atoms.epsilons {
            for (i, res) in self.residue_info.iter().enumerate() {
                for atom_idx in res.start_atom..(res.start_atom + res.num_atoms) {
                    if atom_idx >= data.len() {
                        continue;
                    }
                    let val = data[atom_idx];
                    match self.raw_atoms.atom_names[atom_idx].as_str() {
                        "N" => values[i * 5] = val,
                        "CA" => values[i * 5 + 1] = val,
                        "C" => values[i * 5 + 2] = val,
                        "CB" => values[i * 5 + idx_cb] = val,
                        "O" => values[i * 5 + idx_o] = val,
                        _ => {}
                    }
                }
            }
        }
        values
    }
}

/// Rebuild ProcessedStructure and Bonds by sorting atoms to ensure contiguous residues.
/// This is necessary after adding atoms (like Hydrogens) which are appended to the end.
pub fn rebuild_topology(
    structure: ProcessedStructure,
    bonds: Vec<[usize; 2]>,
) -> Result<(ProcessedStructure, Vec<[usize; 2]>), String> {
    let n = structure.raw_atoms.num_atoms;
    if n == 0 {
        return Ok((structure, bonds));
    }

    // 1. Create permutation
    let mut indices: Vec<usize> = (0..n).collect();

    // Sort key: (ChainID, ResID, AtomOrder/Index)
    indices.sort_by(|&a, &b| {
        let chain_a = &structure.raw_atoms.chain_ids[a];
        let chain_b = &structure.raw_atoms.chain_ids[b];
        let res_a = structure.raw_atoms.res_ids[a];
        let res_b = structure.raw_atoms.res_ids[b];

        chain_a.cmp(chain_b).then(res_a.cmp(&res_b)).then(a.cmp(&b)) // Stable relative order
    });

    // 2. Build mapping (Old -> New)
    let mut old_to_new = vec![0usize; n];
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        old_to_new[old_idx] = new_idx;
    }

    // 3. Permute RawAtomData
    let old_raw = structure.raw_atoms;
    let mut new_raw = RawAtomData::new();
    new_raw.num_atoms = n;

    // Pre-allocate vectors
    // new_raw.ids removed - not a field
    // RawAtomData fields are: coords, atom_names, elements, serial_numbers, ...
    // Wait, check definition in mod.rs: "pub serial_numbers: Vec<i32>".
    // "pub res_ids: Vec<i32>".
    // I need to match fields exactly.

    new_raw.coords = Vec::with_capacity(n * 3);
    new_raw.atom_names = Vec::with_capacity(n);
    new_raw.elements = Vec::with_capacity(n);
    new_raw.serial_numbers = Vec::with_capacity(n);
    new_raw.alt_locs = Vec::with_capacity(n);
    new_raw.res_names = Vec::with_capacity(n);
    new_raw.res_ids = Vec::with_capacity(n);
    new_raw.insertion_codes = Vec::with_capacity(n);
    new_raw.chain_ids = Vec::with_capacity(n);
    new_raw.b_factors = Vec::with_capacity(n);
    new_raw.occupancy = Vec::with_capacity(n);
    new_raw.is_hetatm = Vec::with_capacity(n);

    // Manual loop over sorted indices
    for &idx in &indices {
        new_raw
            .coords
            .extend_from_slice(&old_raw.coords[idx * 3..idx * 3 + 3]);
        new_raw.atom_names.push(old_raw.atom_names[idx].clone());
        new_raw.elements.push(old_raw.elements[idx].clone());
        new_raw.serial_numbers.push(old_raw.serial_numbers[idx]);
        new_raw.alt_locs.push(old_raw.alt_locs[idx]);
        new_raw.res_names.push(old_raw.res_names[idx].clone());
        new_raw.res_ids.push(old_raw.res_ids[idx]);
        new_raw.insertion_codes.push(old_raw.insertion_codes[idx]);
        new_raw.chain_ids.push(old_raw.chain_ids[idx].clone());
        new_raw.b_factors.push(old_raw.b_factors[idx]);
        new_raw.occupancy.push(old_raw.occupancy[idx]);
        new_raw.is_hetatm.push(old_raw.is_hetatm[idx]);

        if let Some(ref c) = old_raw.charges {
            if new_raw.charges.is_none() {
                new_raw.charges = Some(Vec::with_capacity(n));
            }
            new_raw.charges.as_mut().unwrap().push(c[idx]);
        }
        if let Some(ref r) = old_raw.radii {
            if new_raw.radii.is_none() {
                new_raw.radii = Some(Vec::with_capacity(n));
            }
            new_raw.radii.as_mut().unwrap().push(r[idx]);
        }
        if let Some(ref s) = old_raw.sigmas {
            if new_raw.sigmas.is_none() {
                new_raw.sigmas = Some(Vec::with_capacity(n));
            }
            new_raw.sigmas.as_mut().unwrap().push(s[idx]);
        }
        if let Some(ref e) = old_raw.epsilons {
            if new_raw.epsilons.is_none() {
                new_raw.epsilons = Some(Vec::with_capacity(n));
            }
            new_raw.epsilons.as_mut().unwrap().push(e[idx]);
        }
    }

    // 4. Remap Bonds
    let mut new_bonds = Vec::with_capacity(bonds.len());
    for bond in bonds {
        let (i, j) = (bond[0], bond[1]);
        if i < n && j < n {
            new_bonds.push([old_to_new[i], old_to_new[j]]);
        }
    }

    // 5. Build new ProcessedStructure
    let new_structure = ProcessedStructure::from_raw(new_raw)?;

    Ok((new_structure, new_bonds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::AtomRecord;

    #[test]
    fn test_residue_grouping() {
        let mut raw = RawAtomData::with_capacity(6);

        // Add 2 residues (ALA and GLY) with 3 atoms each
        for i in 0..3 {
            raw.add_atom(AtomRecord {
                serial: i + 1,
                atom_name: if i == 0 {
                    "N".to_string()
                } else if i == 1 {
                    "CA".to_string()
                } else {
                    "C".to_string()
                },
                alt_loc: ' ',
                res_name: "ALA".to_string(),
                chain_id: "A".to_string(),
                res_seq: 1,
                i_code: ' ',
                x: i as f32,
                y: 0.0,
                z: 0.0,
                occupancy: 1.0,
                temp_factor: 20.0,
                element: "C".to_string(),
                charge: None,
                radius: None,
                is_hetatm: false,
            });
        }

        for i in 0..3 {
            raw.add_atom(AtomRecord {
                serial: i + 4,
                atom_name: if i == 0 {
                    "N".to_string()
                } else if i == 1 {
                    "CA".to_string()
                } else {
                    "C".to_string()
                },
                alt_loc: ' ',
                res_name: "GLY".to_string(),
                chain_id: "A".to_string(),
                res_seq: 2,
                i_code: ' ',
                x: i as f32 + 3.0,
                y: 0.0,
                z: 0.0,
                occupancy: 1.0,
                temp_factor: 20.0,
                element: "C".to_string(),
                charge: None,
                radius: None,
                is_hetatm: false,
            });
        }

        let processed = ProcessedStructure::from_raw(raw).unwrap();

        assert_eq!(processed.num_residues, 2);
        assert_eq!(processed.num_chains, 1);
        assert_eq!(processed.residue_info[0].res_name, "ALA");
        assert_eq!(processed.residue_info[1].res_name, "GLY");
        assert_eq!(processed.residue_info[0].num_atoms, 3);
        assert_eq!(processed.residue_info[1].num_atoms, 3);
    }

    #[test]
    fn test_multi_chain() {
        let mut raw = RawAtomData::with_capacity(2);

        raw.add_atom(AtomRecord {
            serial: 1,
            atom_name: "CA".to_string(),
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
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });

        raw.add_atom(AtomRecord {
            serial: 2,
            atom_name: "CA".to_string(),
            alt_loc: ' ',
            res_name: "GLY".to_string(),
            chain_id: "B".to_string(),
            res_seq: 1,
            i_code: ' ',
            x: 10.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });

        let processed = ProcessedStructure::from_raw(raw).unwrap();

        assert_eq!(processed.num_residues, 2);
        assert_eq!(processed.num_chains, 2);
        assert!(processed.chain_indices.contains_key("A"));
        assert!(processed.chain_indices.contains_key("B"));
    }

    #[test]
    fn test_hetatm_grouping() {
        let mut raw = RawAtomData::with_capacity(5);

        // Protein residue (ATOM)
        raw.add_atom(AtomRecord {
            serial: 1,
            atom_name: "CA".to_string(),
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
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: false,
        });

        // Ligand (HETATM, not water, multi-atom)
        raw.add_atom(AtomRecord {
            serial: 2,
            atom_name: "C1".to_string(),
            alt_loc: ' ',
            res_name: "LIG".to_string(),
            chain_id: "A".to_string(),
            res_seq: 100,
            i_code: ' ',
            x: 10.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "C".to_string(),
            charge: None,
            radius: None,
            is_hetatm: true,
        });

        // Solvent (HETATM, water)
        raw.add_atom(AtomRecord {
            serial: 3,
            atom_name: "O".to_string(),
            alt_loc: ' ',
            res_name: "HOH".to_string(),
            chain_id: "A".to_string(),
            res_seq: 200,
            i_code: ' ',
            x: 20.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "O".to_string(),
            charge: None,
            radius: None,
            is_hetatm: true,
        });

        // Ion (HETATM, single atom, short name)
        raw.add_atom(AtomRecord {
            serial: 4,
            atom_name: "ZN".to_string(),
            alt_loc: ' ',
            res_name: "ZN".to_string(),
            chain_id: "A".to_string(),
            res_seq: 300,
            i_code: ' ',
            x: 30.0,
            y: 0.0,
            z: 0.0,
            occupancy: 1.0,
            temp_factor: 20.0,
            element: "ZN".to_string(),
            charge: None,
            radius: None,
            is_hetatm: true,
        });

        let processed = ProcessedStructure::from_raw(raw).unwrap();

        // Only protein residue in residue_info
        assert_eq!(processed.num_residues, 1);
        assert_eq!(processed.residue_info[0].res_name, "ALA");

        // Ligand grouping
        assert_eq!(processed.ligand_groups.len(), 1);
        assert_eq!(processed.ligand_groups[0].res_name, "LIG");

        // Solvent atoms
        assert_eq!(processed.solvent_atoms.len(), 1);

        // Ion atoms
        assert_eq!(processed.ion_atoms.len(), 1);

        // Molecule type assignment
        assert_eq!(processed.molecule_type[0], 0); // Protein
        assert_eq!(processed.molecule_type[1], 1); // Ligand
        assert_eq!(processed.molecule_type[2], 2); // Solvent
        assert_eq!(processed.molecule_type[3], 3); // Ion
    }

    /// Builds one residue's worth of AtomRecords from a bare atom-name list.
    /// Element is derived as the atom name's first character, which is
    /// correct for every heavy-atom name used by these tests (no hydrogens
    /// are constructed here, so the element != "H" filter in `from_raw` is a
    /// no-op by construction -- exactly what these tests want to exercise).
    fn make_residue_atoms(
        res_name: &str,
        atom_names: &[&str],
        is_hetatm: bool,
        serial_start: i32,
    ) -> Vec<AtomRecord> {
        atom_names
            .iter()
            .enumerate()
            .map(|(i, name)| AtomRecord {
                serial: serial_start + i as i32,
                atom_name: name.to_string(),
                alt_loc: ' ',
                res_name: res_name.to_string(),
                chain_id: "A".to_string(),
                res_seq: 1,
                i_code: ' ',
                x: i as f32,
                y: 0.0,
                z: 0.0,
                occupancy: 1.0,
                temp_factor: 20.0,
                element: name[0..1].to_string(),
                charge: None,
                radius: None,
                is_hetatm,
            })
            .collect()
    }

    #[test]
    fn test_charmm_his_variant_recognized_as_protein_via_alias() {
        // Real CHARMM HSD: exactly His's heavy-atom set, tagged HETATM (as
        // real QM/MM structure-prep tools commonly do) -- must resolve to HIS
        // under the default (Standard) config, not be dropped as a ligand.
        let his_atoms = crate::chem::residues::get_residue_atoms()["HIS"].clone();
        let mut raw = RawAtomData::with_capacity(his_atoms.len());
        for atom in make_residue_atoms("HSD", &his_atoms, true, 1) {
            raw.add_atom(atom);
        }

        let processed = ProcessedStructure::from_raw(raw).unwrap();

        assert_eq!(processed.num_residues, 1);
        assert!(
            processed.ligand_groups.is_empty(),
            "HSD must not be dropped as a ligand"
        );
        assert_eq!(
            processed.residue_info[0].res_name, "HSD",
            "original name must be preserved"
        );
        let his_idx = build_resname_to_idx()["HIS"];
        assert_eq!(processed.residue_info[0].res_type, his_idx);
        assert_eq!(processed.molecule_type[0], 0);
    }

    #[test]
    fn test_real_ccd_collision_recognized_as_protein_but_not_misaliased() {
        // Real CCD CYM is S-methylcysteine: CYS's heavy atoms plus an extra
        // methyl carbon ("CM"). Must NOT be aliased to CYS (the atom check
        // rejects it), but IS a genuine CCD peptide-linking residue, so it
        // should still land in residue_info (as UNK type), not ligand_groups.
        let mut atoms: Vec<&str> = crate::chem::residues::get_residue_atoms()["CYS"].clone();
        atoms.push("CM");
        let mut raw = RawAtomData::with_capacity(atoms.len());
        for atom in make_residue_atoms("CYM", &atoms, true, 1) {
            raw.add_atom(atom);
        }

        let processed = ProcessedStructure::from_raw(raw).unwrap();

        assert_eq!(processed.num_residues, 1);
        assert!(processed.ligand_groups.is_empty());
        assert_eq!(processed.residue_info[0].res_name, "CYM");
        assert_eq!(
            processed.residue_info[0].res_type, UNK_RESTYPE_INDEX,
            "must not be misaliased to CYS's type"
        );
    }

    #[test]
    fn test_unrelated_ccd_ligand_with_variant_name_stays_a_ligand() {
        // Stand-in for the REAL CCD HSD (an unrelated small molecule, NOT
        // histidine) -- atom names that don't remotely match His's imidazole
        // ring template. Real CCD also says HSD is NON-POLYMER, so this must
        // land in ligand_groups, not be aliased and not be CCD-rescued.
        let atoms = ["C1", "C2", "O1", "O2", "O3"];
        let mut raw = RawAtomData::with_capacity(atoms.len());
        for atom in make_residue_atoms("HSD", &atoms, true, 1) {
            raw.add_atom(atom);
        }

        let processed = ProcessedStructure::from_raw(raw).unwrap();

        assert_eq!(processed.num_residues, 0);
        assert_eq!(processed.ligand_groups.len(), 1);
        assert_eq!(processed.ligand_groups[0].res_name, "HSD");
    }

    #[test]
    fn test_force_field_prepped_convention_trusts_alias_unconditionally() {
        // Same mismatched atom set as the collision test above, but under
        // ForceFieldPrepped the caller asserts the file's provenance is known
        // simulation output -- the alias should apply with no atom check.
        let mut atoms: Vec<&str> = crate::chem::residues::get_residue_atoms()["CYS"].clone();
        atoms.push("CM");
        let mut raw = RawAtomData::with_capacity(atoms.len());
        for atom in make_residue_atoms("CYM", &atoms, true, 1) {
            raw.add_atom(atom);
        }

        let config = ResidueClassificationConfig {
            naming_convention: ResidueNamingConvention::ForceFieldPrepped,
            alias_match: AliasMatchConfig::default(),
        };
        let processed = ProcessedStructure::from_raw_with_config(raw, config).unwrap();

        assert_eq!(processed.num_residues, 1);
        assert!(processed.ligand_groups.is_empty());
        let cys_idx = build_resname_to_idx()["CYS"];
        assert_eq!(processed.residue_info[0].res_type, cys_idx);
    }

    #[test]
    fn test_mse_recognized_as_protein_via_ccd_alone() {
        // MSE (selenomethionine) is deliberately NOT in the alias table --
        // it's an unambiguous, genuine CCD entry, so the CCD path (Part 2)
        // alone must recognize it as protein, regardless of HETATM tagging.
        let atoms = ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"];
        let mut raw = RawAtomData::with_capacity(atoms.len());
        for atom in make_residue_atoms("MSE", &atoms, true, 1) {
            raw.add_atom(atom);
        }

        let processed = ProcessedStructure::from_raw(raw).unwrap();

        assert_eq!(processed.num_residues, 1);
        assert!(processed.ligand_groups.is_empty());
        assert_eq!(processed.residue_info[0].res_name, "MSE");
        assert_eq!(processed.molecule_type[0], 0);
    }
}
