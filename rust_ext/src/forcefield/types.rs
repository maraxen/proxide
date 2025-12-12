//! Force field data types for molecular dynamics.
//!
//! This module contains all the data structures used to represent
//! force field parameters parsed from OpenMM-style XML files.
//!
//! Note: Some fields are parsed but not yet fully utilized in Python API.

#![allow(dead_code)]

use std::collections::HashMap;

/// Atom type definition from the force field
#[derive(Debug, Clone)]
pub struct AtomType {
    /// Unique name of the atom type (e.g., "protein-C", "gaff-ca")
    pub name: String,
    /// Atom class for parameter lookup (e.g., "protein-C")
    pub class: String,
    /// Element symbol (e.g., "C", "N", "O")
    pub element: String,
    /// Atomic mass in atomic mass units (Da)
    pub mass: f32,
    /// Partial charge (optional, often defined in residue templates instead)
    pub charge: Option<f32>,
}

/// Atom within a residue template
#[derive(Debug, Clone)]
pub struct ResidueAtom {
    /// Atom name within the residue (e.g., "CA", "N", "O")
    pub name: String,
    /// Atom type for parameter lookup
    pub atom_type: String,
    /// Partial charge in elementary charge units (optional)
    pub charge: Option<f32>,
}

/// Residue template defining the topology of a residue
#[derive(Debug, Clone)]
pub struct ResidueTemplate {
    /// Residue name (3-letter code, e.g., "ALA", "GLY")
    pub name: String,
    /// Atoms in the residue
    pub atoms: Vec<ResidueAtom>,
    /// Internal bonds as (atom_name1, atom_name2) pairs
    pub bonds: Vec<(String, String)>,
    /// External bond connection points
    pub external_bonds: Vec<String>,
    /// Override priority (for N/C-terminal variants)
    pub override_level: Option<u32>,
}

/// Harmonic bond parameters: V(r) = k/2 * (r - r0)^2
#[derive(Debug, Clone)]
pub struct HarmonicBondParam {
    /// Atom type/class 1
    pub class1: String,
    /// Atom type/class 2
    pub class2: String,
    /// Force constant in kJ/mol/nm^2
    pub k: f32,
    /// Equilibrium bond length in nm
    pub length: f32,
}

/// Harmonic angle parameters: V(θ) = k/2 * (θ - θ0)^2
#[derive(Debug, Clone)]
pub struct HarmonicAngleParam {
    /// Atom type/class 1
    pub class1: String,
    /// Atom type/class 2 (central atom)
    pub class2: String,
    /// Atom type/class 3
    pub class3: String,
    /// Force constant in kJ/mol/rad^2
    pub k: f32,
    /// Equilibrium angle in radians
    pub angle: f32,
}

/// Periodic torsion (dihedral) term
#[derive(Debug, Clone)]
pub struct TorsionTerm {
    /// Periodicity (1, 2, 3, ...)
    pub periodicity: u32,
    /// Phase offset in radians
    pub phase: f32,
    /// Force constant in kJ/mol
    pub k: f32,
}

/// Proper torsion parameters for dihedrals
#[derive(Debug, Clone)]
pub struct ProperTorsionParam {
    /// Atom type/class 1
    pub class1: String,
    /// Atom type/class 2
    pub class2: String,
    /// Atom type/class 3
    pub class3: String,
    /// Atom type/class 4
    pub class4: String,
    /// Torsion terms (can have multiple periodicities)
    pub terms: Vec<TorsionTerm>,
}

/// Improper torsion parameters (for planarity)
#[derive(Debug, Clone)]
pub struct ImproperTorsionParam {
    /// Atom type/class 1
    pub class1: String,
    /// Atom type/class 2 (central atom)
    pub class2: String,
    /// Atom type/class 3
    pub class3: String,
    /// Atom type/class 4
    pub class4: String,
    /// Torsion terms
    pub terms: Vec<TorsionTerm>,
}

/// Nonbonded (Lennard-Jones) parameters per atom type
#[derive(Debug, Clone)]
pub struct NonbondedParam {
    /// Atom type this applies to
    pub atom_type: String,
    /// Charge in elementary charge units
    pub charge: f32,
    /// LJ sigma parameter in nm
    pub sigma: f32,
    /// LJ epsilon parameter in kJ/mol
    pub epsilon: f32,
}

/// GBSA-OBC implicit solvent parameters
#[derive(Debug, Clone)]
pub struct GBSAOBCParam {
    /// Atom type this applies to
    pub atom_type: String,
    /// Atomic radius in nm
    pub radius: f32,
    /// Scaling factor
    pub scale: f32,
}

/// CMAP torsion correction data
#[derive(Debug, Clone)]
pub struct CMAPData {
    /// Map index to energy grid
    pub maps: Vec<CMAPGrid>,
    /// Torsion definitions (which atom types use which map)
    pub torsions: Vec<CMAPTorsion>,
}

/// CMAP energy grid (typically 24x24)
#[derive(Debug, Clone)]
pub struct CMAPGrid {
    /// Grid size (e.g., 24)
    pub size: usize,
    /// Energy values in kJ/mol, row-major order
    pub energies: Vec<f32>,
}

/// CMAP torsion assignment
#[derive(Debug, Clone)]
pub struct CMAPTorsion {
    /// Five atom types defining the two overlapping dihedrals
    pub class1: String,
    pub type2: String,
    pub type3: String,
    pub type4: String,
    pub class5: String,
    /// Index into CMAPData.maps
    pub map_index: usize,
}

/// Complete parsed force field
#[derive(Debug, Clone)]
pub struct ForceField {
    /// Force field name/identifier
    pub name: String,

    /// Atom type definitions
    pub atom_types: Vec<AtomType>,

    /// Residue templates
    pub residue_templates: Vec<ResidueTemplate>,

    /// Harmonic bond parameters (class1-class2 → param)
    pub harmonic_bonds: Vec<HarmonicBondParam>,

    /// Harmonic angle parameters
    pub harmonic_angles: Vec<HarmonicAngleParam>,

    /// Proper torsion parameters
    pub proper_torsions: Vec<ProperTorsionParam>,

    /// Improper torsion parameters
    pub improper_torsions: Vec<ImproperTorsionParam>,

    /// Nonbonded parameters (indexed by atom type)
    pub nonbonded_params: Vec<NonbondedParam>,

    /// GBSA-OBC implicit solvent parameters
    pub gbsa_obc_params: Vec<GBSAOBCParam>,

    /// CMAP correction data (optional)
    pub cmap_data: Option<CMAPData>,

    // === Lookup tables for fast access ===
    /// Atom type name → index
    pub atom_type_map: HashMap<String, usize>,

    /// Residue name → template index
    pub residue_map: HashMap<String, usize>,
}

impl ForceField {
    /// Create an empty force field
    pub fn new(name: String) -> Self {
        Self {
            name,
            atom_types: Vec::new(),
            residue_templates: Vec::new(),
            harmonic_bonds: Vec::new(),
            harmonic_angles: Vec::new(),
            proper_torsions: Vec::new(),
            improper_torsions: Vec::new(),
            nonbonded_params: Vec::new(),
            gbsa_obc_params: Vec::new(),
            cmap_data: None,
            atom_type_map: HashMap::new(),
            residue_map: HashMap::new(),
        }
    }

    /// Build lookup indices after parsing
    pub fn build_indices(&mut self) {
        self.atom_type_map = self
            .atom_types
            .iter()
            .enumerate()
            .map(|(i, at)| (at.name.clone(), i))
            .collect();

        self.residue_map = self
            .residue_templates
            .iter()
            .enumerate()
            .map(|(i, rt)| (rt.name.clone(), i))
            .collect();
    }

    /// Get residue template by name
    pub fn get_residue(&self, name: &str) -> Option<&ResidueTemplate> {
        self.residue_map
            .get(name)
            .map(|&i| &self.residue_templates[i])
    }

    /// Get atom type by name
    pub fn get_atom_type(&self, name: &str) -> Option<&AtomType> {
        self.atom_type_map.get(name).map(|&i| &self.atom_types[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forcefield_new() {
        let ff = ForceField::new("test".to_string());
        assert_eq!(ff.name, "test");
        assert!(ff.atom_types.is_empty());
    }

    #[test]
    fn test_build_indices() {
        let mut ff = ForceField::new("test".to_string());
        ff.atom_types.push(AtomType {
            name: "C".to_string(),
            class: "C".to_string(),
            element: "C".to_string(),
            mass: 12.01,
            charge: None,
        });
        ff.residue_templates.push(ResidueTemplate {
            name: "ALA".to_string(),
            atoms: Vec::new(),
            bonds: Vec::new(),
            external_bonds: Vec::new(),
            override_level: None,
        });

        ff.build_indices();

        assert_eq!(ff.atom_type_map.get("C"), Some(&0));
        assert_eq!(ff.residue_map.get("ALA"), Some(&0));
    }
}
