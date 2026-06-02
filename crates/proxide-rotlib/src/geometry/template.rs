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
    /// Torsion angle (degrees) — defined by great_grandparent→grandparent→parent→this.
    pub torsion_deg: f32,
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

    // Internal coordinates (Engh-Huber ideal values, placeholder).
    // PLACEHOLDER: Engh-Huber; canonical match is CHARMM — see backlog #820 / synthesis.jsonl
    // Atoms 0-2 (N, CA, C) are backbone and have no BondDef.
    // Atom 3 (O): parent=C (idx 2), C-O bond ~1.23 Å, angle C-C-O ~123°, torsion N-CA-C-O ~180°
    t.set_bond(3, BondDef {
        parent_idx: 2,
        bond_length: 1.231,
        bond_angle_deg: 123.0,
        torsion_deg: 180.0,
    });

    // Atom 4 (CB): parent=CA (idx 1), CA-CB bond = 1.543 Å, angle N-CA-CB = 104.7°
    // torsion is the fixed backbone improper dihedral (C-N-CA-CB) from CCD = -119.6°
    // CCD-derived ideal geometry from crates/proxide-rotlib/tests/data/ccd/PRO.cif (public domain).
    // See synthesis.jsonl for canonical CHARMM #820 convergence history.
    t.set_bond(4, BondDef {
        parent_idx: 1,
        bond_length: 1.543,
        bond_angle_deg: 104.7,
        torsion_deg: -119.6, // Fixed improper dihedral (C-N-CA-CB), CCD value
    });

    // Atom 5 (CG): parent=CB (idx 4), CB-CG bond = 1.543 Å, angle CA-CB-CG = 105.1°
    // torsion will be set by χ1 (N-CA-CB-CG), which is EXACT (not relaxed)
    t.set_bond(5, BondDef {
        parent_idx: 4,
        bond_length: 1.543,
        bond_angle_deg: 105.1,
        torsion_deg: 0.0, // placeholder; set by χ1 (exact)
    });

    // Atom 6 (CD): parent=CG (idx 5), CG-CD bond = 1.544 Å, angle CB-CG-CD = SOLVED
    // torsion will be set by χ2 (CA-CB-CG-CD), which is EXACT (not relaxed)
    // The angle CB-CG-CD is the dependent degree of freedom solved by 1-D root-find
    // to achieve ring closure |CD−N| = 1.487 Å. Expect ~105.1° ±3° per CCD ideal.
    t.set_bond(6, BondDef {
        parent_idx: 5,
        bond_length: 1.544,
        bond_angle_deg: 105.1, // Initial guess; will be solved in ring closure
        torsion_deg: 0.0, // placeholder; set by χ2 (exact)
    });

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
