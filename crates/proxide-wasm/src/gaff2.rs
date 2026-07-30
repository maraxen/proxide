use crate::smiles::WasmMol;
use once_cell::sync::Lazy;
use proxide_core::forcefield::topology::{Bond as TopoBond, Topology};
use proxide_gaff::gaff_generator::GaffTemplateGenerator;
use serde::Serialize;

static GAFF2_XML: &str = include_str!("../../../src/proxide/assets/gaff/ffxml/gaff-2.11.xml");

static GENERATOR: Lazy<GaffTemplateGenerator> = Lazy::new(|| {
    GaffTemplateGenerator::from_xml_content("gaff-2.11", GAFF2_XML)
        .expect("Failed to load embedded GAFF2 XML")
});

/// Record of a single atom with its assigned GAFF2 type
#[derive(Debug, Serialize)]
pub struct AtomRecord {
    pub idx: usize,
    pub atomic_num: u8,
    pub atom_type: String,
}

/// Record of a bond parameter
#[derive(Debug, Serialize)]
pub struct BondRecord {
    pub i: usize,
    pub j: usize,
    pub k: f64,  // force constant, kJ/mol/nm²
    pub r0: f64, // equilibrium length, nm
    pub type_pair: [String; 2],
}

/// Record of an angle parameter
#[derive(Debug, Serialize)]
pub struct AngleRecord {
    pub i: usize,
    pub j: usize,
    pub k_idx: usize, // central atom index
    pub k_angle: f64, // force constant, kJ/mol/rad²
    pub theta0: f64,  // equilibrium angle, radians
    pub type_triple: [String; 3],
}

/// Record of a torsion (proper dihedral) parameter
#[derive(Debug, Serialize)]
pub struct TorsionRecord {
    pub i: usize,
    pub j: usize,
    pub k_idx: usize,
    pub l: usize,
    pub periodicity: u32,
    pub k_torsion: f64, // kcal/mol
    pub phase: f64,     // radians
    pub type_quad: [String; 4],
}

/// Complete parameter set for a molecule
#[derive(Debug, Serialize)]
pub struct ParamSet {
    pub molecule_id: String,
    pub smiles: Option<String>,
    pub atoms: Vec<AtomRecord>,
    pub bonds: Vec<BondRecord>,
    pub angles: Vec<AngleRecord>,
    pub torsions: Vec<TorsionRecord>,
}

/// Assign GAFF2 parameters to a molecule
///
/// Takes a WasmMol (from SMILES or PDB parsing) and returns a complete
/// parameter set with atom types and bond/angle/torsion parameters.
pub fn assign_params(mol: &WasmMol) -> ParamSet {
    let gen = &*GENERATOR;

    // Build element list from atomic numbers
    let elements: Vec<String> = mol
        .atoms
        .iter()
        .map(|a| atomic_num_to_element(a.atomic_num).to_string())
        .collect();

    // Build proxide-core Topology from WasmMol bonds, adding implicit hydrogens
    let topo_bonds: Vec<TopoBond> = mol.bonds.iter().map(|b| TopoBond::new(b.i, b.j)).collect();

    // Add implicit hydrogen atoms and bonds
    let (elements, topo_bonds) = add_implicit_hydrogens(&elements, &topo_bonds, &mol.atoms);

    let topology = Topology::new(topo_bonds, &elements);

    // Assign GAFF2 atom types
    let mut atom_types = gen.assign_atom_types(&elements, &topology);

    // Correct aromatic atoms based on SMILES notation
    // (the Topology's aromaticity detection may miss aromatic atoms marked in SMILES)
    for (idx, atom) in mol.atoms.iter().enumerate() {
        if atom.is_aromatic
            && atom.atomic_num == 6
            && atom_types
                .get(idx)
                .map(|t| t.starts_with('c'))
                .unwrap_or(false)
        {
            // Convert aliphatic carbon to aromatic
            atom_types[idx] = match atom_types[idx].as_str() {
                "c1" => "c1".to_string(),   // Already aromatic
                "c2" => "ca".to_string(),   // sp2 -> ca
                "c3" => "ca".to_string(),   // sp3 -> ca
                "ca" => "ca".to_string(),   // Already aromatic
                "cc" => "cc".to_string(),   // Already aromatic
                other => other.to_string(), // Leave as-is
            };
        }
        if atom.is_aromatic
            && atom.atomic_num == 7
            && atom_types
                .get(idx)
                .map(|t| t.starts_with('n'))
                .unwrap_or(false)
        {
            // Convert aliphatic nitrogen to aromatic
            atom_types[idx] = match atom_types[idx].as_str() {
                "n2" => "na".to_string(),   // sp2 N -> na
                "n3" => "na".to_string(),   // sp3 N -> na
                "na" => "na".to_string(),   // Already aromatic
                "nb" => "nb".to_string(),   // Already aromatic
                other => other.to_string(), // Leave as-is
            };
        }
    }

    // Build output atom records
    let atoms: Vec<AtomRecord> = mol
        .atoms
        .iter()
        .enumerate()
        .map(|(idx, a)| AtomRecord {
            idx,
            atomic_num: a.atomic_num,
            atom_type: atom_types.get(idx).cloned().unwrap_or_default(),
        })
        .collect();

    // Look up bond parameters
    let bonds: Vec<BondRecord> = topology
        .bonds
        .iter()
        .filter_map(|b| {
            let t1 = atom_types.get(b.i)?;
            let t2 = atom_types.get(b.j)?;
            let param = gen.get_bond_parameters(t1, t2)?;
            Some(BondRecord {
                i: b.i,
                j: b.j,
                k: param.k as f64,
                r0: param.length as f64,
                type_pair: [t1.clone(), t2.clone()],
            })
        })
        .collect();

    // Look up angle parameters
    let angles: Vec<AngleRecord> = topology
        .angles
        .iter()
        .filter_map(|a| {
            let t1 = atom_types.get(a.i)?;
            let t2 = atom_types.get(a.j)?;
            let t3 = atom_types.get(a.k)?;
            let param = gen.get_angle_parameters(t1, t2, t3)?;
            Some(AngleRecord {
                i: a.i,
                j: a.j,
                k_idx: a.k,
                k_angle: param.k as f64,
                theta0: param.angle as f64,
                type_triple: [t1.clone(), t2.clone(), t3.clone()],
            })
        })
        .collect();

    // Look up torsion parameters
    let torsions: Vec<TorsionRecord> = topology
        .proper_dihedrals
        .iter()
        .flat_map(|d| {
            let t1 = match atom_types.get(d.i) {
                Some(t) => t,
                None => return vec![],
            };
            let t2 = match atom_types.get(d.j) {
                Some(t) => t,
                None => return vec![],
            };
            let t3 = match atom_types.get(d.k) {
                Some(t) => t,
                None => return vec![],
            };
            let t4 = match atom_types.get(d.l) {
                Some(t) => t,
                None => return vec![],
            };
            let param = match gen.get_proper_torsion_parameters(t1, t2, t3, t4) {
                Some(p) => p,
                None => return vec![],
            };

            // Each ProperTorsionParam has a Vec<TorsionTerm>, we return one record per term
            param
                .terms
                .iter()
                .map(|term| TorsionRecord {
                    i: d.i,
                    j: d.j,
                    k_idx: d.k,
                    l: d.l,
                    periodicity: term.periodicity,
                    k_torsion: term.k as f64,
                    phase: term.phase as f64,
                    type_quad: [t1.clone(), t2.clone(), t3.clone(), t4.clone()],
                })
                .collect()
        })
        .collect();

    ParamSet {
        molecule_id: "mol".to_string(),
        smiles: None,
        atoms,
        bonds,
        angles,
        torsions,
    }
}

/// Add implicit hydrogens to the element list and bonds
/// Returns expanded (elements, bonds) with explicit H atoms and C-H / N-H / O-H / S-H bonds
fn add_implicit_hydrogens(
    elements: &[String],
    bonds: &[TopoBond],
    atoms: &[crate::smiles::WasmAtom],
) -> (Vec<String>, Vec<TopoBond>) {
    let mut expanded_elements = elements.to_vec();
    let mut expanded_bonds = bonds.to_vec();

    // Calculate current degree for each atom (from explicit bonds)
    let mut degrees: Vec<usize> = vec![0; elements.len()];
    for bond in bonds {
        degrees[bond.i] += 1;
        degrees[bond.j] += 1;
    }

    // For each atom, add implicit H atoms
    for (i, atom) in atoms.iter().enumerate() {
        if atom.implicit_h > 0 {
            for _ in 0..atom.implicit_h {
                let h_idx = expanded_elements.len();
                expanded_elements.push("H".to_string());
                expanded_bonds.push(TopoBond::new(i, h_idx));
            }
        }
    }

    (expanded_elements, expanded_bonds)
}

/// Convert atomic number to element symbol
fn atomic_num_to_element(z: u8) -> &'static str {
    match z {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        15 => "P",
        16 => "S",
        17 => "Cl",
        35 => "Br",
        53 => "I",
        _ => "X",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smiles::parse;

    #[test]
    fn methane_atom_types() {
        let mol = parse("C").unwrap();
        let params = assign_params(&mol);
        assert_eq!(
            params.atoms[0].atom_type, "c3",
            "methane carbon should be c3"
        );
    }

    #[test]
    fn ethane_has_cc_bond() {
        let mol = parse("CC").unwrap();
        let params = assign_params(&mol);

        // Should have at least 1 C-C bond (and 6 C-H bonds from implicit hydrogens)
        let cc_bond = params.bonds.iter().find(|b| {
            (b.type_pair[0] == "c3" && b.type_pair[1] == "c3")
                || (b.type_pair[1] == "c3" && b.type_pair[0] == "c3")
        });
        assert!(
            cc_bond.is_some(),
            "ethane should have a c3-c3 bond, bonds present: {:?}",
            params
                .bonds
                .iter()
                .map(|b| &b.type_pair)
                .collect::<Vec<_>>()
        );

        let cc = cc_bond.unwrap();
        assert!(
            cc.r0 > 0.14 && cc.r0 < 0.16,
            "C-C bond length should be ~0.1526 nm, got {}",
            cc.r0
        );
        assert!(cc.k > 100.0, "C-C force constant should be significant");

        // With implicit H expansion, ethane (CC) should have 7 bonds total: 1 C-C + 6 C-H
        assert!(
            params.bonds.len() > 1,
            "ethane with implicit H should have >1 bond (C-C + C-H), got {} bonds",
            params.bonds.len()
        );
    }

    #[test]
    fn benzene_carbons_are_ca_type() {
        let mol = parse("c1ccccc1").unwrap();
        let params = assign_params(&mol);
        assert!(
            params.atoms.iter().all(|a| a.atom_type == "ca"),
            "all benzene carbons should be 'ca' type"
        );
    }

    #[test]
    fn ethanol_has_correct_atom_types() {
        let mol = parse("CCO").unwrap();
        let params = assign_params(&mol);
        let types: Vec<_> = params.atoms.iter().map(|a| a.atom_type.as_str()).collect();
        assert!(types.contains(&"c3"), "should have c3 (sp3 carbon)");
        assert!(
            params.atoms.iter().any(|a| a.atom_type.starts_with('o')),
            "should have an oxygen type"
        );
    }

    #[test]
    fn ethanol_has_bonds_and_angles() {
        let mol = parse("CCO").unwrap();
        let params = assign_params(&mol);
        assert!(params.bonds.len() > 0, "ethanol should have bonds");
        assert!(params.angles.len() > 0, "ethanol should have angles");
    }

    #[test]
    fn json_schema_roundtrip_ethane() {
        let mol = parse("CC").unwrap(); // ethane: has bonds, angles, torsions
        let params = assign_params(&mol);
        let json = serde_json::to_value(&params).unwrap();

        // Top-level keys match prolix §7.1 schema
        assert!(json.get("molecule_id").is_some(), "missing molecule_id");
        assert!(
            json.get("atoms").is_some() && json["atoms"].is_array(),
            "missing/wrong atoms"
        );
        assert!(
            json.get("bonds").is_some() && json["bonds"].is_array(),
            "missing/wrong bonds"
        );
        assert!(
            json.get("angles").is_some() && json["angles"].is_array(),
            "missing/wrong angles"
        );
        assert!(
            json.get("torsions").is_some() && json["torsions"].is_array(),
            "missing/wrong torsions"
        );

        // Bond record shape: i, j, k, r0, type_pair
        if let Some(bond) = json["bonds"].as_array().and_then(|a| a.first()) {
            assert!(
                bond.get("i").and_then(|v| v.as_u64()).is_some(),
                "bond.i missing or wrong type"
            );
            assert!(
                bond.get("j").and_then(|v| v.as_u64()).is_some(),
                "bond.j missing or wrong type"
            );
            assert!(
                bond.get("k").and_then(|v| v.as_f64()).is_some(),
                "bond.k missing or wrong type"
            );
            assert!(
                bond.get("r0").and_then(|v| v.as_f64()).is_some(),
                "bond.r0 missing or wrong type"
            );
            assert!(
                bond.get("type_pair")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len() == 2)
                    .unwrap_or(false),
                "bond.type_pair should be array of 2 strings"
            );
        } else {
            panic!("ethane should have at least one bond");
        }

        // Atom record shape: idx, atomic_num, atom_type
        if let Some(atom) = json["atoms"].as_array().and_then(|a| a.first()) {
            assert!(
                atom.get("idx").and_then(|v| v.as_u64()).is_some(),
                "atom.idx missing"
            );
            assert!(
                atom.get("atomic_num").and_then(|v| v.as_u64()).is_some(),
                "atom.atomic_num missing"
            );
            assert!(
                atom.get("atom_type").and_then(|v| v.as_str()).is_some(),
                "atom.atom_type missing"
            );
        }

        // Angle record shape: i, j, k_idx, k_angle, theta0, type_triple
        if let Some(angle) = json["angles"].as_array().and_then(|a| a.first()) {
            assert!(angle.get("i").is_some(), "angle.i missing");
            assert!(angle.get("j").is_some(), "angle.j missing");
            assert!(angle.get("k_idx").is_some(), "angle.k_idx missing");
            assert!(
                angle.get("k_angle").and_then(|v| v.as_f64()).is_some(),
                "angle.k_angle missing/wrong"
            );
            assert!(
                angle.get("theta0").and_then(|v| v.as_f64()).is_some(),
                "angle.theta0 missing/wrong"
            );
            assert!(
                angle
                    .get("type_triple")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len() == 3)
                    .unwrap_or(false),
                "angle.type_triple should be array of 3"
            );
        }

        // Torsion record shape (if present)
        if let Some(torsion) = json["torsions"].as_array().and_then(|a| a.first()) {
            assert!(torsion.get("i").is_some(), "torsion.i missing");
            assert!(torsion.get("j").is_some(), "torsion.j missing");
            assert!(torsion.get("k_idx").is_some(), "torsion.k_idx missing");
            assert!(torsion.get("l").is_some(), "torsion.l missing");
            assert!(
                torsion
                    .get("periodicity")
                    .and_then(|v| v.as_u64())
                    .is_some(),
                "torsion.periodicity missing"
            );
            assert!(
                torsion.get("k_torsion").and_then(|v| v.as_f64()).is_some(),
                "torsion.k_torsion missing"
            );
            assert!(
                torsion.get("phase").and_then(|v| v.as_f64()).is_some(),
                "torsion.phase missing"
            );
            assert!(
                torsion
                    .get("type_quad")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len() == 4)
                    .unwrap_or(false),
                "torsion.type_quad should be 4 strings"
            );
        }

        // Print the full JSON for manual verification
        println!("{}", serde_json::to_string_pretty(&params).unwrap());
    }
}
