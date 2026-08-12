//! Residue template: ideal internal coordinates and atom connectivity.
//!
//! Defines a residue's atoms, bond connectivity, ideal bond lengths/angles,
//! and dihedral angle definitions (χ).

/// Internal coordinate definition: bond length, angle, torsion.
#[derive(Clone, Copy, Debug)]
pub struct BondDef {
    /// Parent atom (by index, relative to the atom list).
    pub parent_idx: usize,
    /// Bond length (Å) — from parent to this atom.
    pub bond_length: f32,
    /// Bond angle (degrees) — defined by grandparent→parent→this.
    pub bond_angle_deg: f32,
    /// Torsion angle (degrees) — base value.
    /// If `relative_chi` is None, this is the absolute dihedral (used for backbone atoms,
    /// chi-defining atoms with placeholder 0.0, and rigid atoms like backbone O).
    /// If `relative_chi` is Some(i), torsion = chi_values[i] + torsion_deg (used for
    /// branch atoms whose orientation follows a chi angle with a fixed offset).
    pub torsion_deg: f32,
    /// If Some(chi_idx), the actual torsion = chi_values[chi_idx] + torsion_deg.
    /// Used for branch atoms (e.g. CG2 in VAL: torsion = chi1 + 120°).
    pub relative_chi: Option<usize>,
}

/// Dihedral angle definition.
#[derive(Clone, Debug)]
pub struct DihedralDef {
    /// Name of the dihedral (e.g., "χ1", "χ2").
    pub name: String,
    /// Indices of the four atoms defining the dihedral (in order).
    pub atom_indices: [usize; 4],
}

/// Residue template: atom names, connectivity, idealized geometry.
#[derive(Clone, Debug)]
pub struct ResidueTemplate {
    /// Residue code (e.g., "PRO", "CPR").
    pub code: String,
    /// Atom names in library order (e.g., ["N", "CA", "C", "O", "CB", "CG", "CD"]).
    pub atom_names: Vec<String>,
    /// Internal coordinate definitions for each atom (indexed by atom_names).
    /// atom_names[0] (N) has no definition (backbone N).
    /// atom_names[1] (CA) has no definition (backbone CA).
    /// atom_names[2] (C) has no definition (backbone C).
    /// Remaining atoms have BondDef entries.
    pub bonds: Vec<Option<BondDef>>,
    /// Dihedral angle definitions.
    pub dihedrals: Vec<DihedralDef>,
}

impl ResidueTemplate {
    /// Create a new residue template.
    pub fn new(code: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            atom_names: Vec::new(),
            bonds: Vec::new(),
            dihedrals: Vec::new(),
        }
    }

    /// Add an atom to the template (by name).
    pub fn add_atom(&mut self, name: impl Into<String>) {
        self.atom_names.push(name.into());
        self.bonds.push(None);
    }

    /// Set internal coordinates for an atom (by index).
    pub fn set_bond(&mut self, atom_idx: usize, bond: BondDef) {
        if atom_idx < self.bonds.len() {
            self.bonds[atom_idx] = Some(bond);
        }
    }

    /// Add a dihedral definition.
    pub fn add_dihedral(&mut self, dihedral: DihedralDef) {
        self.dihedrals.push(dihedral);
    }

    /// Get the number of atoms.
    pub fn num_atoms(&self) -> usize {
        self.atom_names.len()
    }

    /// Get atom index by name.
    pub fn atom_index(&self, name: &str) -> Option<usize> {
        self.atom_names.iter().position(|n| n == name)
    }

    /// Get dihedral by name (e.g., "χ1").
    pub fn dihedral(&self, name: &str) -> Option<&DihedralDef> {
        self.dihedrals.iter().find(|d| d.name == name)
    }
}

/// Get the standard residue template for a given residue code.
///
/// Returns templates for the 19 non-proline amino acids (ALA, ARG, ASN, ASP, CYS, CYH, CYD, GLN, GLU, HIS, ILE, LEU, LYS, MET, PHE, SER, THR, TRP, TYR, VAL).
/// All geometry uses Engh-Huber ideal parameters.
///
/// # Parameters
/// * `code` - Three-letter residue code (case-sensitive, e.g., "ARG", "ALA")
///
/// # Returns
/// `Some(ResidueTemplate)` if code is recognized, `None` otherwise.
pub fn standard_residue_template(code: &str) -> Option<ResidueTemplate> {
    match code {
        "ALA" => Some(alanine_template()),
        "ARG" => Some(arginine_template()),
        "ASN" => Some(asparagine_template()),
        "ASP" => Some(aspartic_acid_template()),
        "CYS" => Some(cysteine_template()),
        "CYH" => Some(cysteine_protonated_template()),
        "CYD" => Some(cysteine_disulfide_template()),
        "GLN" => Some(glutamine_template()),
        "GLU" => Some(glutamic_acid_template()),
        "HIS" => Some(histidine_template()),
        "ILE" => Some(isoleucine_template()),
        "LEU" => Some(leucine_template()),
        "LYS" => Some(lysine_template()),
        "MET" => Some(methionine_template()),
        "PHE" => Some(phenylalanine_template()),
        "SER" => Some(serine_template()),
        "THR" => Some(threonine_template()),
        "TRP" => Some(tryptophan_template()),
        "TYR" => Some(tyrosine_template()),
        "VAL" => Some(valine_template()),
        _ => None,
    }
}

fn alanine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("ALA");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 2, 4],
    });

    t
}

fn serine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("SER");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("OG");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.417,
            bond_angle_deg: 111.1,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t
}

fn threonine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("THR");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("OG1");
    t.add_atom("CG2");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.433,
            bond_angle_deg: 109.6,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 4,
            bond_length: 1.525,
            bond_angle_deg: 111.5,
            torsion_deg: 120.0,

            relative_chi: Some(0), // χ1 + 120° for gauche
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t
}

fn valine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("VAL");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG1");
    t.add_atom("CG2");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.524,
            bond_angle_deg: 110.9,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 4,
            bond_length: 1.524,
            bond_angle_deg: 110.9,
            torsion_deg: 120.0,

            relative_chi: Some(0), // χ1 + 120° for gauche
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t
}

fn leucine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("LEU");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("CD1");
    t.add_atom("CD2");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.530,
            bond_angle_deg: 116.3,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.524,
            bond_angle_deg: 111.1,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 5,
            bond_length: 1.524,
            bond_angle_deg: 111.1,
            torsion_deg: 120.0,

            relative_chi: Some(1), // χ2 + 120° for second branch
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });

    t
}

fn isoleucine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("ILE");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG1");
    t.add_atom("CG2");
    t.add_atom("CD1");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.530,
            bond_angle_deg: 110.7,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 4,
            bond_length: 1.521,
            bond_angle_deg: 110.7,
            torsion_deg: 120.0,

            relative_chi: Some(0), // χ1 + 120° for CG2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 5,
            bond_length: 1.513,
            bond_angle_deg: 113.8,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 7],
    });

    t
}

fn methionine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("MET");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("SD");
    t.add_atom("CE");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.530,
            bond_angle_deg: 114.1,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.807,
            bond_angle_deg: 112.7,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 6,
            bond_length: 1.791,
            bond_angle_deg: 100.9,
            torsion_deg: 0.0,

            relative_chi: None, // χ3
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });

    t.add_dihedral(DihedralDef {
        name: "χ3".to_string(),
        atom_indices: [4, 5, 6, 7],
    });

    t
}

fn cysteine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("CYS");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("SG");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.808,
            bond_angle_deg: 113.8,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t
}

fn cysteine_protonated_template() -> ResidueTemplate {
    // CYH has same geometry as CYS (only protonation state differs)
    cysteine_template()
}

fn cysteine_disulfide_template() -> ResidueTemplate {
    // CYD has same geometry as CYS (only bonding differs in external modeling)
    cysteine_template()
}

fn aspartic_acid_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("ASP");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("OD1");
    t.add_atom("OD2");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.516,
            bond_angle_deg: 113.3,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.249,
            bond_angle_deg: 118.8,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 5,
            bond_length: 1.249,
            bond_angle_deg: 118.8,
            torsion_deg: 180.0,

            relative_chi: Some(1), // χ2 + 180° for second carboxyl oxygen
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });

    t
}

fn asparagine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("ASN");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("OD1");
    t.add_atom("ND2");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.516,
            bond_angle_deg: 116.4,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 5,
            bond_length: 1.328,
            bond_angle_deg: 116.5,
            torsion_deg: 180.0,

            relative_chi: Some(1), // χ2 + 180° for amide
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });

    t
}

fn glutamic_acid_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("GLU");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("CD");
    t.add_atom("OE1");
    t.add_atom("OE2");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.516,
            bond_angle_deg: 114.2,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.516,
            bond_angle_deg: 116.3,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 6,
            bond_length: 1.249,
            bond_angle_deg: 118.8,
            torsion_deg: 0.0,

            relative_chi: None, // χ3
        },
    );

    t.set_bond(
        8,
        BondDef {
            parent_idx: 6,
            bond_length: 1.249,
            bond_angle_deg: 118.8,
            torsion_deg: 180.0,

            relative_chi: Some(2), // χ3 + 180° for second carboxyl oxygen
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });

    t.add_dihedral(DihedralDef {
        name: "χ3".to_string(),
        atom_indices: [4, 5, 6, 7],
    });

    t
}

fn glutamine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("GLN");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("CD");
    t.add_atom("OE1");
    t.add_atom("NE2");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.516,
            bond_angle_deg: 113.8,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.516,
            bond_angle_deg: 112.6,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 6,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 0.0,

            relative_chi: None, // χ3
        },
    );

    t.set_bond(
        8,
        BondDef {
            parent_idx: 6,
            bond_length: 1.328,
            bond_angle_deg: 116.5,
            torsion_deg: 180.0,

            relative_chi: Some(2), // χ3 + 180° for amide
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });

    t.add_dihedral(DihedralDef {
        name: "χ3".to_string(),
        atom_indices: [4, 5, 6, 7],
    });

    t
}

fn phenylalanine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("PHE");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("CD1");
    t.add_atom("CD2");
    t.add_atom("CE1");
    t.add_atom("CE2");
    t.add_atom("CZ");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.502,
            bond_angle_deg: 113.8,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.384,
            bond_angle_deg: 120.7,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 5,
            bond_length: 1.384,
            bond_angle_deg: 120.7,
            torsion_deg: 180.0,

            relative_chi: Some(1), // χ2 + 180° for symmetric ring
        },
    );

    t.set_bond(
        8,
        BondDef {
            parent_idx: 6,
            bond_length: 1.382,
            bond_angle_deg: 120.7,
            torsion_deg: 180.0,

            relative_chi: None, // ring closure
        },
    );

    t.set_bond(
        9,
        BondDef {
            parent_idx: 7,
            bond_length: 1.382,
            bond_angle_deg: 120.7,
            torsion_deg: 180.0,

            relative_chi: None, // ring closure
        },
    );

    t.set_bond(
        10,
        BondDef {
            parent_idx: 8,
            bond_length: 1.382,
            bond_angle_deg: 120.0,
            torsion_deg: 0.0,

            relative_chi: None, // aromatic ring
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });

    t
}

fn tyrosine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("TYR");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("CD1");
    t.add_atom("CD2");
    t.add_atom("CE1");
    t.add_atom("CE2");
    t.add_atom("CZ");
    t.add_atom("OH");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.502,
            bond_angle_deg: 113.8,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.384,
            bond_angle_deg: 120.7,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 5,
            bond_length: 1.384,
            bond_angle_deg: 120.7,
            torsion_deg: 180.0,

            relative_chi: Some(1), // χ2 + 180°
        },
    );

    t.set_bond(
        8,
        BondDef {
            parent_idx: 6,
            bond_length: 1.382,
            bond_angle_deg: 120.7,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        9,
        BondDef {
            parent_idx: 7,
            bond_length: 1.382,
            bond_angle_deg: 120.7,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        10,
        BondDef {
            parent_idx: 8,
            bond_length: 1.382,
            bond_angle_deg: 120.0,
            torsion_deg: 0.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        11,
        BondDef {
            parent_idx: 10,
            bond_length: 1.362,
            bond_angle_deg: 119.8,
            torsion_deg: 0.0,

            relative_chi: None,
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });

    t
}

fn tryptophan_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("TRP");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("CD1");
    t.add_atom("CD2");
    t.add_atom("NE1");
    t.add_atom("CE2");
    t.add_atom("CE3");
    t.add_atom("CZ2");
    t.add_atom("CZ3");
    t.add_atom("CH2");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.498,
            bond_angle_deg: 113.6,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.365,
            bond_angle_deg: 127.1,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 5,
            bond_length: 1.430,
            bond_angle_deg: 126.6,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        8,
        BondDef {
            parent_idx: 6,
            bond_length: 1.372,
            bond_angle_deg: 107.5,
            torsion_deg: 0.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        9,
        BondDef {
            parent_idx: 7,
            bond_length: 1.415,
            bond_angle_deg: 107.1,
            torsion_deg: 0.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        10,
        BondDef {
            parent_idx: 7,
            bond_length: 1.404,
            bond_angle_deg: 133.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        11,
        BondDef {
            parent_idx: 9,
            bond_length: 1.404,
            bond_angle_deg: 120.0,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        12,
        BondDef {
            parent_idx: 10,
            bond_length: 1.404,
            bond_angle_deg: 120.0,
            torsion_deg: 0.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        13,
        BondDef {
            parent_idx: 11,
            bond_length: 1.404,
            bond_angle_deg: 120.0,
            torsion_deg: 0.0,

            relative_chi: None,
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });

    t
}

fn histidine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("HIS");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("ND1");
    t.add_atom("CD2");
    t.add_atom("CE1");
    t.add_atom("NE2");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.497,
            bond_angle_deg: 113.8,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.345,
            bond_angle_deg: 122.0,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 5,
            bond_length: 1.354,
            bond_angle_deg: 131.0,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        8,
        BondDef {
            parent_idx: 6,
            bond_length: 1.343,
            bond_angle_deg: 107.1,
            torsion_deg: 0.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        9,
        BondDef {
            parent_idx: 7,
            bond_length: 1.338,
            bond_angle_deg: 110.5,
            torsion_deg: 0.0,

            relative_chi: None,
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });

    t
}

fn lysine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("LYS");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("CD");
    t.add_atom("CE");
    t.add_atom("NZ");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.530,
            bond_angle_deg: 114.1,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.520,
            bond_angle_deg: 111.3,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 6,
            bond_length: 1.520,
            bond_angle_deg: 111.3,
            torsion_deg: 0.0,

            relative_chi: None, // χ3
        },
    );

    t.set_bond(
        8,
        BondDef {
            parent_idx: 7,
            bond_length: 1.489,
            bond_angle_deg: 111.9,
            torsion_deg: 0.0,

            relative_chi: None, // χ4
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });

    t.add_dihedral(DihedralDef {
        name: "χ3".to_string(),
        atom_indices: [4, 5, 6, 7],
    });

    t.add_dihedral(DihedralDef {
        name: "χ4".to_string(),
        atom_indices: [5, 6, 7, 8],
    });

    t
}

fn arginine_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("ARG");
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("CD");
    t.add_atom("NE");
    t.add_atom("CZ");
    t.add_atom("NH1");
    t.add_atom("NH2");

    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 120.8,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.540,
            bond_angle_deg: 110.5,
            torsion_deg: -119.7,

            relative_chi: None,
        },
    );

    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.530,
            bond_angle_deg: 114.1,
            torsion_deg: 0.0,

            relative_chi: None, // χ1
        },
    );

    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.520,
            bond_angle_deg: 111.3,
            torsion_deg: 0.0,

            relative_chi: None, // χ2
        },
    );

    t.set_bond(
        7,
        BondDef {
            parent_idx: 6,
            bond_length: 1.460,
            bond_angle_deg: 112.0,
            torsion_deg: 0.0,

            relative_chi: None, // χ3
        },
    );

    t.set_bond(
        8,
        BondDef {
            parent_idx: 7,
            bond_length: 1.329,
            bond_angle_deg: 124.2,
            torsion_deg: 0.0,

            relative_chi: None, // χ4
        },
    );

    t.set_bond(
        9,
        BondDef {
            parent_idx: 8,
            bond_length: 1.326,
            bond_angle_deg: 120.0,
            torsion_deg: 0.0,

            relative_chi: None, // χ4
        },
    );

    t.set_bond(
        10,
        BondDef {
            parent_idx: 8,
            bond_length: 1.326,
            bond_angle_deg: 120.0,
            torsion_deg: 180.0,

            relative_chi: None, // guanidinium is planar; NH2 is always 180° from NH1
        },
    );

    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });

    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });

    t.add_dihedral(DihedralDef {
        name: "χ3".to_string(),
        atom_indices: [4, 5, 6, 7],
    });

    t.add_dihedral(DihedralDef {
        name: "χ4".to_string(),
        atom_indices: [5, 6, 7, 8],
    });

    t
}

/// Create the proline residue template with CCD-derived ideal geometry and discrete pucker model.
///
/// # Ring Closure Strategy (P3 v2)
/// The proline ring is built using exact χ angles (from Dunbrack) and a 1-D angle-relaxation
/// ring closure. The CB-CG-CD bond angle is solved (not χ2) to achieve ring closure
/// |CD−N| = 1.487 Å. This preserves both χ1 and χ2 exactly (within tolerance) while
/// only relaxing the ring angle. Pucker is intrinsic to χ1 sign (endo: χ1≥0, exo: χ1<0).
///
/// All bond and angle values are from the public-domain CCD PRO.cif entry.
/// CHARMM #820 is the canonical reference — see synthesis.jsonl for convergence history.
pub fn proline_template() -> ResidueTemplate {
    let mut t = ResidueTemplate::new("PRO");

    // Backbone atoms (N, CA, C, O)
    t.add_atom("N");
    t.add_atom("CA");
    t.add_atom("C");
    t.add_atom("O");

    // Sidechain atoms (CB, CG, CD)
    t.add_atom("CB");
    t.add_atom("CG");
    t.add_atom("CD");

    // Internal coordinates: Placeholder CCD values (will be replaced by converter with CHARMM).
    // Backbone atoms (N, CA, C) have no BondDef.
    // The converter loads CHARMM36 FFXML and applies ideals to this template before building.
    // See geometry/charmm_ic.rs and backlog #820 research doc for sourcing rationale.
    // Atom 3 (O): parent=C (idx 2), C-O bond ~1.23 Å, angle C-C-O ~123°, torsion N-CA-C-O ~180°
    t.set_bond(
        3,
        BondDef {
            parent_idx: 2,
            bond_length: 1.231,
            bond_angle_deg: 123.0,
            torsion_deg: 180.0,

            relative_chi: None,
        },
    );

    // Atom 4 (CB): parent=CA (idx 1) — CCD placeholder; overridden by converter with CHARMM
    t.set_bond(
        4,
        BondDef {
            parent_idx: 1,
            bond_length: 1.543,
            bond_angle_deg: 104.7,
            torsion_deg: -119.6,

            relative_chi: None,
        },
    );

    // Atom 5 (CG): parent=CB (idx 4) — CCD placeholder; overridden by converter with CHARMM
    t.set_bond(
        5,
        BondDef {
            parent_idx: 4,
            bond_length: 1.543,
            bond_angle_deg: 105.1,
            torsion_deg: 0.0,

            relative_chi: None,
        },
    );

    // Atom 6 (CD): parent=CG (idx 5) — CCD placeholder; overridden by converter with CHARMM
    // Ring closure target CD_N_IDEAL is CHARMM 1.455 Å (changed in proline.rs).
    t.set_bond(
        6,
        BondDef {
            parent_idx: 5,
            bond_length: 1.544,
            bond_angle_deg: 105.1,
            torsion_deg: 0.0,

            relative_chi: None,
        },
    );

    // Dihedral definitions.
    // χ1 = N-CA-CB-CG
    t.add_dihedral(DihedralDef {
        name: "χ1".to_string(),
        atom_indices: [0, 1, 4, 5],
    });
    // χ2 = CA-CB-CG-CD
    t.add_dihedral(DihedralDef {
        name: "χ2".to_string(),
        atom_indices: [1, 4, 5, 6],
    });
    // χ3 = CB-CG-CD-N (ring closure)
    t.add_dihedral(DihedralDef {
        name: "χ3".to_string(),
        atom_indices: [4, 5, 6, 0],
    });

    t
}
