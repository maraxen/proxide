//! GAFF Template Generator
//!
//! Rust implementation of the GAFFTemplateGenerator from openmmforcefields.
//! Generates residue templates and force field parameters for small molecules
//! using GAFF (General Amber Force Field) atom typing.
//!
//! This module provides:
//! - GAFF atom type assignment based on element, hybridization, and connectivity
//! - Residue template generation for small molecules
//! - Parameter lookup from pre-loaded GAFF force field XML files
//!
//! # Usage
//!
//! ```rust,ignore
//! use priox_rs::forcefield::gaff_generator::GaffTemplateGenerator;
//!
//! // Create generator with GAFF 2.11 force field
//! let generator = GaffTemplateGenerator::new("gaff-2.11")?;
//!
//! // Generate template for molecule from topology
//! let template = generator.generate_template(&molecule_name, &elements, &topology)?;
//! ```

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use super::topology::Topology;
use super::types::{
    ForceField, HarmonicAngleParam, HarmonicBondParam, ImproperTorsionParam, NonbondedParam,
    ProperTorsionParam, ResidueAtom, ResidueTemplate,
};
use super::xml_parser::parse_forcefield_xml;

/// Supported GAFF force field versions
pub const INSTALLED_FORCEFIELDS: &[&str] = &[
    "gaff-1.4",
    "gaff-1.7",
    "gaff-1.8",
    "gaff-1.81",
    "gaff-2.1",
    "gaff-2.11",
    "gaff-2.2.20",
];

/// Error type for GAFF template generation
#[derive(Debug)]
pub enum GaffError {
    /// Invalid force field version
    InvalidForceField(String),
    /// Force field file not found
    FileNotFound(String),
    /// Parse error
    ParseError(String),
    /// Missing parameters
    MissingParameters(String),
}

impl std::fmt::Display for GaffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GaffError::InvalidForceField(msg) => write!(f, "Invalid force field: {}", msg),
            GaffError::FileNotFound(path) => write!(f, "Force field file not found: {}", path),
            GaffError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            GaffError::MissingParameters(msg) => write!(f, "Missing parameters: {}", msg),
        }
    }
}

impl std::error::Error for GaffError {}

/// GAFF atom type assignment rules
///
/// Based on GAFF2 atom typing rules derived from element, hybridization,
/// and local chemical environment.
#[derive(Debug, Clone)]
pub struct GaffAtomTyper {
    /// Map of element -> base atom types
    element_types: HashMap<String, Vec<GaffTypeRule>>,
}

/// Rule for assigning GAFF atom types
#[derive(Debug, Clone)]
pub struct GaffTypeRule {
    /// GAFF atom type name
    pub atom_type: String,
    /// Number of neighbors (hybridization hint)
    pub num_neighbors: Option<usize>,
    /// Is aromatic
    pub is_aromatic: Option<bool>,
    /// Is in ring
    pub is_ring: Option<bool>,
    /// Specific neighbor elements
    pub neighbor_elements: Option<Vec<String>>,
    /// Description
    pub description: String,
}

impl GaffAtomTyper {
    /// Create a new GAFF atom typer with standard rules
    pub fn new() -> Self {
        let mut element_types = HashMap::new();

        // Carbon types
        element_types.insert(
            "C".to_string(),
            vec![
                GaffTypeRule {
                    atom_type: "ca".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(true),
                    is_ring: Some(true),
                    neighbor_elements: None,
                    description: "Sp2 C in aromatic ring".to_string(),
                },
                GaffTypeRule {
                    atom_type: "c".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: Some(vec!["O".to_string()]),
                    description: "Sp2 C carbonyl group".to_string(),
                },
                GaffTypeRule {
                    atom_type: "c1".to_string(),
                    num_neighbors: Some(2),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "Sp C".to_string(),
                },
                GaffTypeRule {
                    atom_type: "c2".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "Sp2 C".to_string(),
                },
                GaffTypeRule {
                    atom_type: "c3".to_string(),
                    num_neighbors: Some(4),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "Sp3 C".to_string(),
                },
                GaffTypeRule {
                    atom_type: "cc".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(false),
                    is_ring: Some(true),
                    neighbor_elements: None,
                    description: "Sp2 C in non-pure aromatic ring".to_string(),
                },
                GaffTypeRule {
                    atom_type: "cx".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(false),
                    is_ring: Some(true),
                    neighbor_elements: None,
                    description: "Sp3 C in 3-membered ring".to_string(),
                },
                GaffTypeRule {
                    atom_type: "cy".to_string(),
                    num_neighbors: Some(4),
                    is_aromatic: Some(false),
                    is_ring: Some(true),
                    neighbor_elements: None,
                    description: "Sp3 C in 4-membered ring".to_string(),
                },
            ],
        );

        // Nitrogen types
        element_types.insert(
            "N".to_string(),
            vec![
                GaffTypeRule {
                    atom_type: "na".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(true),
                    is_ring: Some(true),
                    neighbor_elements: None,
                    description: "Sp2 N in aromatic ring with H".to_string(),
                },
                GaffTypeRule {
                    atom_type: "nb".to_string(),
                    num_neighbors: Some(2),
                    is_aromatic: Some(true),
                    is_ring: Some(true),
                    neighbor_elements: None,
                    description: "Sp2 N in pure aromatic ring".to_string(),
                },
                GaffTypeRule {
                    atom_type: "nc".to_string(),
                    num_neighbors: Some(2),
                    is_aromatic: Some(false),
                    is_ring: Some(true),
                    neighbor_elements: None,
                    description: "Sp2 N in non-pure aromatic ring".to_string(),
                },
                GaffTypeRule {
                    atom_type: "n".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: Some(vec!["C".to_string()]),
                    description: "Sp2 N in amide group".to_string(),
                },
                GaffTypeRule {
                    atom_type: "n1".to_string(),
                    num_neighbors: Some(1),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "Sp N".to_string(),
                },
                GaffTypeRule {
                    atom_type: "n2".to_string(),
                    num_neighbors: Some(2),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "Sp2 N (aliphatic with double bond)".to_string(),
                },
                GaffTypeRule {
                    atom_type: "n3".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "Sp3 N".to_string(),
                },
                GaffTypeRule {
                    atom_type: "n4".to_string(),
                    num_neighbors: Some(4),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "Sp3 N with 4 substituents".to_string(),
                },
                GaffTypeRule {
                    atom_type: "nh".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: Some(vec!["C".to_string()]), // connected to aromatic
                    description: "Amine N connected to aromatic ring".to_string(),
                },
                GaffTypeRule {
                    atom_type: "no".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: Some(vec!["O".to_string()]),
                    description: "N in nitro group".to_string(),
                },
            ],
        );

        // Oxygen types
        element_types.insert(
            "O".to_string(),
            vec![
                GaffTypeRule {
                    atom_type: "o".to_string(),
                    num_neighbors: Some(1),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "Sp2 O in carbonyl/carboxylate".to_string(),
                },
                GaffTypeRule {
                    atom_type: "oh".to_string(),
                    num_neighbors: Some(2),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: Some(vec!["H".to_string()]),
                    description: "O in hydroxyl group".to_string(),
                },
                GaffTypeRule {
                    atom_type: "os".to_string(),
                    num_neighbors: Some(2),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "Sp3 O in ether/ester".to_string(),
                },
                GaffTypeRule {
                    atom_type: "ow".to_string(),
                    num_neighbors: Some(2),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: Some(vec!["H".to_string(), "H".to_string()]),
                    description: "O in water".to_string(),
                },
            ],
        );

        // Hydrogen types
        element_types.insert(
            "H".to_string(),
            vec![
                GaffTypeRule {
                    atom_type: "ha".to_string(),
                    num_neighbors: Some(1),
                    is_aromatic: None,
                    is_ring: None,
                    neighbor_elements: None,
                    description: "H on aromatic C".to_string(),
                },
                GaffTypeRule {
                    atom_type: "hc".to_string(),
                    num_neighbors: Some(1),
                    is_aromatic: None,
                    is_ring: None,
                    neighbor_elements: Some(vec!["C".to_string()]),
                    description: "H on aliphatic C without electron-withdrawing".to_string(),
                },
                GaffTypeRule {
                    atom_type: "h1".to_string(),
                    num_neighbors: Some(1),
                    is_aromatic: None,
                    is_ring: None,
                    neighbor_elements: Some(vec!["C".to_string()]),
                    description: "H on C with 1 electron-withdrawing group".to_string(),
                },
                GaffTypeRule {
                    atom_type: "hn".to_string(),
                    num_neighbors: Some(1),
                    is_aromatic: None,
                    is_ring: None,
                    neighbor_elements: Some(vec!["N".to_string()]),
                    description: "H on N".to_string(),
                },
                GaffTypeRule {
                    atom_type: "ho".to_string(),
                    num_neighbors: Some(1),
                    is_aromatic: None,
                    is_ring: None,
                    neighbor_elements: Some(vec!["O".to_string()]),
                    description: "H in hydroxyl group".to_string(),
                },
                GaffTypeRule {
                    atom_type: "hp".to_string(),
                    num_neighbors: Some(1),
                    is_aromatic: None,
                    is_ring: None,
                    neighbor_elements: Some(vec!["P".to_string()]),
                    description: "H on P".to_string(),
                },
                GaffTypeRule {
                    atom_type: "hs".to_string(),
                    num_neighbors: Some(1),
                    is_aromatic: None,
                    is_ring: None,
                    neighbor_elements: Some(vec!["S".to_string()]),
                    description: "H on S".to_string(),
                },
                GaffTypeRule {
                    atom_type: "hw".to_string(),
                    num_neighbors: Some(1),
                    is_aromatic: None,
                    is_ring: None,
                    neighbor_elements: Some(vec!["O".to_string()]),
                    description: "H in water".to_string(),
                },
            ],
        );

        // Sulfur types
        element_types.insert(
            "S".to_string(),
            vec![
                GaffTypeRule {
                    atom_type: "s".to_string(),
                    num_neighbors: Some(1),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "S with one connected atom".to_string(),
                },
                GaffTypeRule {
                    atom_type: "s2".to_string(),
                    num_neighbors: Some(2),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "S with 2 connected atoms (thioether)".to_string(),
                },
                GaffTypeRule {
                    atom_type: "s4".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "S with 3 connected atoms (sulfoxide)".to_string(),
                },
                GaffTypeRule {
                    atom_type: "s6".to_string(),
                    num_neighbors: Some(4),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "S with 4 connected atoms (sulfone)".to_string(),
                },
                GaffTypeRule {
                    atom_type: "sh".to_string(),
                    num_neighbors: Some(2),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: Some(vec!["H".to_string()]),
                    description: "S in thiol group".to_string(),
                },
                GaffTypeRule {
                    atom_type: "ss".to_string(),
                    num_neighbors: Some(2),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: Some(vec!["S".to_string()]),
                    description: "S in disulfide".to_string(),
                },
            ],
        );

        // Phosphorus types
        element_types.insert(
            "P".to_string(),
            vec![
                GaffTypeRule {
                    atom_type: "p2".to_string(),
                    num_neighbors: Some(2),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "Sp2 P".to_string(),
                },
                GaffTypeRule {
                    atom_type: "p3".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "Sp3 P (phosphine)".to_string(),
                },
                GaffTypeRule {
                    atom_type: "p4".to_string(),
                    num_neighbors: Some(3),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: Some(vec!["O".to_string()]),
                    description: "Sp3 P (phosphine oxide)".to_string(),
                },
                GaffTypeRule {
                    atom_type: "p5".to_string(),
                    num_neighbors: Some(4),
                    is_aromatic: Some(false),
                    is_ring: None,
                    neighbor_elements: None,
                    description: "Sp3 P in phosphate".to_string(),
                },
            ],
        );

        // Halogens
        element_types.insert(
            "F".to_string(),
            vec![GaffTypeRule {
                atom_type: "f".to_string(),
                num_neighbors: Some(1),
                is_aromatic: None,
                is_ring: None,
                neighbor_elements: None,
                description: "Fluorine".to_string(),
            }],
        );

        element_types.insert(
            "Cl".to_string(),
            vec![GaffTypeRule {
                atom_type: "cl".to_string(),
                num_neighbors: Some(1),
                is_aromatic: None,
                is_ring: None,
                neighbor_elements: None,
                description: "Chlorine".to_string(),
            }],
        );

        element_types.insert(
            "Br".to_string(),
            vec![GaffTypeRule {
                atom_type: "br".to_string(),
                num_neighbors: Some(1),
                is_aromatic: None,
                is_ring: None,
                neighbor_elements: None,
                description: "Bromine".to_string(),
            }],
        );

        element_types.insert(
            "I".to_string(),
            vec![GaffTypeRule {
                atom_type: "i".to_string(),
                num_neighbors: Some(1),
                is_aromatic: None,
                is_ring: None,
                neighbor_elements: None,
                description: "Iodine".to_string(),
            }],
        );

        Self { element_types }
    }

    /// Assign GAFF atom types to a molecule
    pub fn assign_types(&self, elements: &[String], topology: &Topology) -> Vec<String> {
        // Compute aromaticity
        let is_aromatic_map = topology.compute_aromaticity(elements);
        // TODO: compute ring membership
        let is_ring_map: Vec<bool> = vec![false; elements.len()]; // placeholder

        let mut types = Vec::with_capacity(elements.len());

        // First pass: assign types based on local environment
        for (i, element) in elements.iter().enumerate() {
            let num_neighbors = topology.adjacency.get(&i).map(|v| v.len()).unwrap_or(0);
            let is_aromatic = is_aromatic_map[i];
            let is_ring = is_ring_map[i];

            // Get neighbor elements
            let neighbor_elements: Vec<String> = topology
                .adjacency
                .get(&i)
                .map(|neighbors| {
                    neighbors
                        .iter()
                        .filter_map(|&n| elements.get(n).cloned())
                        .collect()
                })
                .unwrap_or_default();

            let atom_type = self.assign_single_type(
                element,
                num_neighbors,
                is_aromatic,
                is_ring,
                &neighbor_elements,
            );

            types.push(atom_type);
        }

        // Second pass: refine hydrogen types based on parent atom
        let initial_types = types.clone();
        for (i, element) in elements.iter().enumerate() {
            if element.to_uppercase() == "H" {
                if let Some(neighbors) = topology.adjacency.get(&i) {
                    if let Some(&parent_idx) = neighbors.first() {
                        if let Some(parent_type) = initial_types.get(parent_idx) {
                            let refined =
                                self.refine_hydrogen_type(parent_type, elements.get(parent_idx));
                            if let Some(new_type) = refined {
                                types[i] = new_type;
                            }
                        }
                    }
                }
            }
        }

        types
    }

    /// Assign type to a single atom
    fn assign_single_type(
        &self,
        element: &str,
        num_neighbors: usize,
        is_aromatic: bool,
        is_ring: bool,
        neighbor_elements: &[String],
    ) -> String {
        let elem_normalized = element.to_uppercase();
        let elem_key = match elem_normalized.as_str() {
            "CL" => "Cl".to_string(),
            "BR" => "Br".to_string(),
            other => {
                // Capitalize first letter only
                let mut s = other.to_lowercase();
                if let Some(c) = s.get_mut(0..1) {
                    c.make_ascii_uppercase();
                }
                s
            }
        };

        if let Some(rules) = self.element_types.get(&elem_key) {
            // Find best matching rule
            for rule in rules {
                let mut matches = true;

                if let Some(expected_neighbors) = rule.num_neighbors {
                    if num_neighbors != expected_neighbors {
                        matches = false;
                    }
                }

                if let Some(expected_aromatic) = rule.is_aromatic {
                    if is_aromatic != expected_aromatic {
                        matches = false;
                    }
                }

                if let Some(expected_ring) = rule.is_ring {
                    if is_ring != expected_ring {
                        matches = false;
                    }
                }

                // Check neighbor elements if specified
                if let Some(expected_neighbors) = &rule.neighbor_elements {
                    let has_required = expected_neighbors.iter().all(|req| {
                        neighbor_elements
                            .iter()
                            .any(|n| n.to_uppercase() == req.to_uppercase())
                    });
                    if !has_required {
                        matches = false;
                    }
                }

                if matches {
                    return rule.atom_type.clone();
                }
            }

            // Fallback to first rule (default for element)
            if let Some(first_rule) = rules.first() {
                return first_rule.atom_type.clone();
            }
        }

        // Ultimate fallback based on element
        match elem_key.as_str() {
            "C" => "c3".to_string(),
            "N" => "n3".to_string(),
            "O" => "os".to_string(),
            "H" => "hc".to_string(),
            "S" => "ss".to_string(),
            "P" => "p5".to_string(),
            "F" => "f".to_string(),
            "Cl" => "cl".to_string(),
            "Br" => "br".to_string(),
            "I" => "i".to_string(),
            _ => "du".to_string(), // dummy type
        }
    }

    /// Refine hydrogen type based on parent atom type
    fn refine_hydrogen_type(
        &self,
        parent_type: &str,
        parent_element: Option<&String>,
    ) -> Option<String> {
        let parent_elem = parent_element.map(|s| s.to_uppercase()).unwrap_or_default();

        match parent_type {
            "ca" | "cc" | "cd" | "cp" | "cq" => Some("ha".to_string()),
            "n" | "na" | "nb" | "nc" | "nd" | "ne" | "nf" | "nh" | "no" | "n3" | "n4" => {
                Some("hn".to_string())
            }
            "o" | "oh" | "os" | "ow" => Some("ho".to_string()),
            "sh" | "ss" | "s" | "s2" | "s4" | "s6" => Some("hs".to_string()),
            "p2" | "p3" | "p4" | "p5" => Some("hp".to_string()),
            _ => {
                // Based on parent element
                match parent_elem.as_str() {
                    "N" => Some("hn".to_string()),
                    "O" => Some("ho".to_string()),
                    "S" => Some("hs".to_string()),
                    "P" => Some("hp".to_string()),
                    _ => None,
                }
            }
        }
    }
}

impl Default for GaffAtomTyper {
    fn default() -> Self {
        Self::new()
    }
}

/// GAFF Template Generator
///
/// Generates residue templates and force field parameters for small molecules
/// using GAFF atom typing.
#[derive(Debug)]
pub struct GaffTemplateGenerator {
    /// GAFF force field version
    forcefield_version: String,
    /// Major version (1 or 2)
    major_version: u32,
    /// Minor version string
    minor_version: String,
    /// Loaded force field parameters
    forcefield: ForceField,
    /// Atom typer
    typer: GaffAtomTyper,
    /// LJ 1-4 scale factor (GAFF default: 0.5)
    lj14scale: f32,
    /// Coulomb 1-4 scale factor (GAFF default: 0.8333)
    coulomb14scale: f32,
}

impl GaffTemplateGenerator {
    /// Create a new GAFF template generator
    ///
    /// # Arguments
    /// * `forcefield` - GAFF version string (e.g., "gaff-2.11")
    /// * `ff_path` - Optional path to force field XML. If None, will look in assets.
    pub fn new(forcefield: &str, ff_path: Option<&str>) -> Result<Self, GaffError> {
        // Validate force field version
        if !INSTALLED_FORCEFIELDS.contains(&forcefield) {
            return Err(GaffError::InvalidForceField(format!(
                "'{}' not in {:?}",
                forcefield, INSTALLED_FORCEFIELDS
            )));
        }

        // Parse version
        let (major, minor) = Self::parse_version(forcefield)?;

        // Load force field
        let ff = if let Some(path) = ff_path {
            parse_forcefield_xml(path).map_err(|e| GaffError::ParseError(format!("{}", e)))?
        } else {
            // Try to find in assets directory (relative to workspace)
            let path = format!("src/priox/assets/gaff/ffxml/{}.xml", forcefield);
            parse_forcefield_xml(&path)
                .map_err(|e| GaffError::FileNotFound(format!("{}: {}", path, e)))?
        };

        Ok(Self {
            forcefield_version: forcefield.to_string(),
            major_version: major,
            minor_version: minor,
            forcefield: ff,
            typer: GaffAtomTyper::new(),
            lj14scale: 0.5,
            coulomb14scale: 0.8333333,
        })
    }

    /// Parse GAFF version string
    fn parse_version(forcefield: &str) -> Result<(u32, String), GaffError> {
        // Expected format: gaff-X.Y or gaff-X.Y.Z
        let parts: Vec<&str> = forcefield.split('-').collect();
        if parts.len() != 2 || parts[0] != "gaff" {
            return Err(GaffError::InvalidForceField(
                "Format must be 'gaff-X.Y'".to_string(),
            ));
        }

        let version_parts: Vec<&str> = parts[1].splitn(2, '.').collect();
        if version_parts.is_empty() {
            return Err(GaffError::InvalidForceField(
                "Version must have major.minor format".to_string(),
            ));
        }

        let major: u32 = version_parts[0]
            .parse()
            .map_err(|_| GaffError::InvalidForceField("Invalid major version".to_string()))?;

        let minor = if version_parts.len() > 1 {
            version_parts[1].to_string()
        } else {
            "0".to_string()
        };

        Ok((major, minor))
    }

    /// Get force field version
    pub fn version(&self) -> &str {
        &self.forcefield_version
    }

    /// Get major version
    pub fn major_version(&self) -> u32 {
        self.major_version
    }

    /// Get minor version
    pub fn minor_version(&self) -> &str {
        &self.minor_version
    }

    /// Assign GAFF atom types to a molecule
    pub fn assign_atom_types(&self, elements: &[String], topology: &Topology) -> Vec<String> {
        self.typer.assign_types(elements, topology)
    }

    /// Generate a residue template for a molecule
    ///
    /// # Arguments
    /// * `name` - Residue name (typically SMILES or molecule name)
    /// * `elements` - Element symbols for each atom
    /// * `topology` - Molecular topology (bonds)
    /// * `charges` - Optional partial charges for each atom
    ///
    /// # Returns
    /// A ResidueTemplate with atom types and charges
    pub fn generate_template(
        &self,
        name: &str,
        elements: &[String],
        topology: &Topology,
        charges: Option<&[f32]>,
    ) -> Result<ResidueTemplate, GaffError> {
        let atom_types = self.assign_atom_types(elements, topology);
        let n_atoms = elements.len();

        // Create atoms with assigned types
        let mut atoms = Vec::with_capacity(n_atoms);
        for i in 0..n_atoms {
            let atom_name = format!("{}{}", elements[i], i + 1);
            let charge = charges
                .map(|c| c.get(i).copied().unwrap_or(0.0))
                .unwrap_or(0.0);

            atoms.push(ResidueAtom {
                name: atom_name,
                atom_type: atom_types[i].clone(),
                charge: Some(charge),
            });
        }

        // Create bonds from topology
        let mut bonds = Vec::new();
        for (&atom_i, neighbors) in &topology.adjacency {
            for &atom_j in neighbors {
                if atom_i < atom_j {
                    let name1 = format!("{}{}", elements[atom_i], atom_i + 1);
                    let name2 = format!("{}{}", elements[atom_j], atom_j + 1);
                    bonds.push((name1, name2));
                }
            }
        }

        Ok(ResidueTemplate {
            name: name.to_string(),
            atoms,
            bonds,
            external_bonds: Vec::new(),
            override_level: None,
        })
    }

    /// Look up bond parameters for a given pair of atom types
    pub fn get_bond_parameters(&self, type1: &str, type2: &str) -> Option<&HarmonicBondParam> {
        // Try both orderings
        self.forcefield.harmonic_bonds.iter().find(|b| {
            (b.class1 == type1 && b.class2 == type2) || (b.class1 == type2 && b.class2 == type1)
        })
    }

    /// Look up angle parameters for a triplet of atom types
    pub fn get_angle_parameters(
        &self,
        type1: &str,
        type2: &str,
        type3: &str,
    ) -> Option<&HarmonicAngleParam> {
        self.forcefield.harmonic_angles.iter().find(|a| {
            (a.class1 == type1 && a.class2 == type2 && a.class3 == type3)
                || (a.class1 == type3 && a.class2 == type2 && a.class3 == type1)
        })
    }

    /// Look up proper torsion parameters
    pub fn get_proper_torsion_parameters(
        &self,
        type1: &str,
        type2: &str,
        type3: &str,
        type4: &str,
    ) -> Option<&ProperTorsionParam> {
        // First try exact match
        if let Some(param) = self.forcefield.proper_torsions.iter().find(|t| {
            (t.class1 == type1 && t.class2 == type2 && t.class3 == type3 && t.class4 == type4)
                || (t.class1 == type4
                    && t.class2 == type3
                    && t.class3 == type2
                    && t.class4 == type1)
        }) {
            return Some(param);
        }

        // Try wildcard matches (empty string = wildcard)
        self.forcefield.proper_torsions.iter().find(|t| {
            let matches_forward = (t.class1.is_empty() || t.class1 == type1)
                && (t.class2 == type2 || t.class2.is_empty())
                && (t.class3 == type3 || t.class3.is_empty())
                && (t.class4.is_empty() || t.class4 == type4);
            let matches_reverse = (t.class4.is_empty() || t.class4 == type1)
                && (t.class3 == type2 || t.class3.is_empty())
                && (t.class2 == type3 || t.class2.is_empty())
                && (t.class1.is_empty() || t.class1 == type4);
            matches_forward || matches_reverse
        })
    }

    /// Look up improper torsion parameters
    pub fn get_improper_torsion_parameters(
        &self,
        type1: &str,
        type2: &str,
        type3: &str,
        type4: &str,
    ) -> Option<&ImproperTorsionParam> {
        self.forcefield.improper_torsions.iter().find(|t| {
            t.class1 == type1 && t.class2 == type2 && t.class3 == type3 && t.class4 == type4
        })
    }

    /// Look up nonbonded parameters for an atom type
    pub fn get_nonbonded_parameters(&self, atom_type: &str) -> Option<&NonbondedParam> {
        self.forcefield
            .nonbonded_params
            .iter()
            .find(|p| p.atom_type == atom_type)
    }

    /// Get all parameters needed for a molecule
    ///
    /// Returns the parameters that would be needed to simulate the molecule,
    /// including bonds, angles, and torsions based on topology.
    pub fn get_molecule_parameters(
        &self,
        elements: &[String],
        topology: &Topology,
    ) -> MoleculeParameters {
        let atom_types = self.assign_atom_types(elements, topology);

        // Collect bonds
        let mut bonds = Vec::new();
        let mut seen_bonds: HashSet<(usize, usize)> = HashSet::new();
        for (&i, neighbors) in &topology.adjacency {
            for &j in neighbors {
                let key = if i < j { (i, j) } else { (j, i) };
                if !seen_bonds.contains(&key) {
                    seen_bonds.insert(key);
                    if let Some(param) = self.get_bond_parameters(&atom_types[i], &atom_types[j]) {
                        bonds.push(AssignedBond {
                            atom1: i,
                            atom2: j,
                            k: param.k,
                            length: param.length,
                        });
                    }
                }
            }
        }

        // Collect angles
        let angles = self.collect_angles(&atom_types, topology);

        // Collect proper dihedrals
        let proper_dihedrals = self.collect_proper_dihedrals(&atom_types, topology);

        // Collect nonbonded
        let nonbonded: Vec<AssignedNonbonded> = atom_types
            .iter()
            .enumerate()
            .filter_map(|(i, atype)| {
                self.get_nonbonded_parameters(atype)
                    .map(|p| AssignedNonbonded {
                        atom: i,
                        sigma: p.sigma,
                        epsilon: p.epsilon,
                    })
            })
            .collect();

        MoleculeParameters {
            atom_types,
            bonds,
            angles,
            proper_dihedrals,
            nonbonded,
        }
    }

    /// Collect all angles from topology
    fn collect_angles(&self, atom_types: &[String], topology: &Topology) -> Vec<AssignedAngle> {
        let mut angles = Vec::new();
        let mut seen: HashSet<(usize, usize, usize)> = HashSet::new();

        // For each atom, look at all pairs of neighbors
        for (&center, neighbors) in &topology.adjacency {
            let neighbor_list: Vec<usize> = neighbors.to_vec();
            for i in 0..neighbor_list.len() {
                for j in (i + 1)..neighbor_list.len() {
                    let a = neighbor_list[i];
                    let b = neighbor_list[j];

                    // Canonical ordering: (min, center, max)
                    let key = if a < b {
                        (a, center, b)
                    } else {
                        (b, center, a)
                    };
                    if !seen.contains(&key) {
                        seen.insert(key);

                        let type1 = &atom_types[key.0];
                        let type2 = &atom_types[key.1];
                        let type3 = &atom_types[key.2];

                        if let Some(param) = self.get_angle_parameters(type1, type2, type3) {
                            angles.push(AssignedAngle {
                                atom1: key.0,
                                atom2: key.1,
                                atom3: key.2,
                                k: param.k,
                                angle: param.angle,
                            });
                        }
                    }
                }
            }
        }

        angles
    }

    /// Collect all proper dihedrals from topology
    fn collect_proper_dihedrals(
        &self,
        atom_types: &[String],
        topology: &Topology,
    ) -> Vec<AssignedDihedral> {
        let mut dihedrals = Vec::new();
        let mut seen: HashSet<(usize, usize, usize, usize)> = HashSet::new();

        // For each bond (i-j), find all i-j-k-l dihedrals
        for (&i, i_neighbors) in &topology.adjacency {
            for &j in i_neighbors {
                if i >= j {
                    continue;
                } // Only process each bond once

                if let Some(j_neighbors) = topology.adjacency.get(&j) {
                    for &k in j_neighbors {
                        if k == i {
                            continue;
                        }

                        if let Some(k_neighbors) = topology.adjacency.get(&k) {
                            for &l in k_neighbors {
                                if l == j {
                                    continue;
                                }

                                // Canonical ordering
                                let key = if i < l {
                                    (i, j, k, l)
                                } else if i > l {
                                    (l, k, j, i)
                                } else if j < k {
                                    (i, j, k, l)
                                } else {
                                    (l, k, j, i)
                                };

                                if seen.contains(&key) {
                                    continue;
                                }
                                seen.insert(key);

                                let type1 = &atom_types[key.0];
                                let type2 = &atom_types[key.1];
                                let type3 = &atom_types[key.2];
                                let type4 = &atom_types[key.3];

                                if let Some(param) =
                                    self.get_proper_torsion_parameters(type1, type2, type3, type4)
                                {
                                    for term in &param.terms {
                                        dihedrals.push(AssignedDihedral {
                                            atom1: key.0,
                                            atom2: key.1,
                                            atom3: key.2,
                                            atom4: key.3,
                                            periodicity: term.periodicity,
                                            phase: term.phase,
                                            k: term.k,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        dihedrals
    }
}

/// Assigned bond parameters for a specific molecule
#[derive(Debug, Clone)]
pub struct AssignedBond {
    pub atom1: usize,
    pub atom2: usize,
    pub k: f32,
    pub length: f32,
}

/// Assigned angle parameters for a specific molecule
#[derive(Debug, Clone)]
pub struct AssignedAngle {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
    pub k: f32,
    pub angle: f32,
}

/// Assigned dihedral parameters for a specific molecule
#[derive(Debug, Clone)]
pub struct AssignedDihedral {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
    pub atom4: usize,
    pub periodicity: u32,
    pub phase: f32,
    pub k: f32,
}

/// Assigned nonbonded parameters
#[derive(Debug, Clone)]
pub struct AssignedNonbonded {
    pub atom: usize,
    pub sigma: f32,
    pub epsilon: f32,
}

/// Complete molecule parameters
#[derive(Debug, Clone)]
pub struct MoleculeParameters {
    pub atom_types: Vec<String>,
    pub bonds: Vec<AssignedBond>,
    pub angles: Vec<AssignedAngle>,
    pub proper_dihedrals: Vec<AssignedDihedral>,
    pub nonbonded: Vec<AssignedNonbonded>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaff_atom_typer() {
        let typer = GaffAtomTyper::new();

        // Test carbon typing
        let c_type = typer.assign_single_type("C", 4, false, false, &[]);
        assert_eq!(c_type, "c3");

        let ca_type = typer.assign_single_type("C", 3, true, true, &[]);
        assert_eq!(ca_type, "ca");
    }

    #[test]
    fn test_version_parsing() {
        let (major, minor) = GaffTemplateGenerator::parse_version("gaff-2.11").unwrap();
        assert_eq!(major, 2);
        assert_eq!(minor, "11");

        let (major, minor) = GaffTemplateGenerator::parse_version("gaff-1.81").unwrap();
        assert_eq!(major, 1);
        assert_eq!(minor, "81");
    }

    #[test]
    fn test_invalid_version() {
        assert!(GaffTemplateGenerator::parse_version("amber-14").is_err());
        assert!(GaffTemplateGenerator::parse_version("gaff").is_err());
    }
}
