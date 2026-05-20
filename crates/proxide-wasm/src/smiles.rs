use serde::{Deserialize, Serialize};
use purr::graph::Builder;
use purr::read::read;
use purr::feature::{AtomKind, BondKind, Element, Aliphatic, Aromatic, BracketSymbol};

/// Bond order enumeration for WASM serialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BondOrder {
    Single,
    Double,
    Triple,
    Quadruple,
    Aromatic,
}

/// WASM-compatible atom representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmAtom {
    pub idx: usize,
    pub atomic_num: u8,
    pub formal_charge: i8,
    pub implicit_h: u8,
    pub in_ring: bool,
    pub is_aromatic: bool,
    pub atom_type: String, // GAFF2 type — empty until assign_params() is called
}

/// WASM-compatible bond representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmBond {
    pub i: usize,
    pub j: usize,
    pub order: BondOrder,
}

/// WASM-compatible molecular representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmMol {
    pub atoms: Vec<WasmAtom>,
    pub bonds: Vec<WasmBond>,
    pub residue_names: Vec<String>, // empty for SMILES path; populated by PDB path
}

/// Parse a SMILES string and return a WasmMol.
pub fn parse(smiles: &str) -> Result<WasmMol, String> {
    let mut builder = Builder::new();
    read(smiles, &mut builder, None).map_err(|e| format!("SMILES parse error: {:?}", e))?;

    let atoms = builder
        .build()
        .map_err(|e| format!("Graph build error: {:?}", e))?;

    // Detect rings using DFS
    let in_ring = detect_rings(&atoms);

    // Detect aromatic atoms from SMILES notation
    let is_aromatic = detect_aromaticity(&atoms);

    // Convert atoms from purr representation to WasmAtom
    let mut wasm_atoms = Vec::new();
    for (idx, atom) in atoms.iter().enumerate() {
        let (atomic_num, formal_charge) = extract_atomic_info(&atom.kind)?;
        let implicit_h = atom.subvalence();

        wasm_atoms.push(WasmAtom {
            idx,
            atomic_num,
            formal_charge,
            implicit_h,
            in_ring: in_ring[idx],
            is_aromatic: is_aromatic[idx],
            atom_type: String::new(), // To be populated by assign_params()
        });
    }

    // Convert bonds from purr representation to WasmBond
    let mut wasm_bonds = Vec::new();
    let mut visited_bonds = std::collections::HashSet::new();

    for (i, atom) in atoms.iter().enumerate() {
        for bond in &atom.bonds {
            let j = bond.tid;
            // Avoid duplicate bonds (bonds are stored in both directions)
            let edge = if i < j { (i, j) } else { (j, i) };
            if !visited_bonds.contains(&edge) {
                visited_bonds.insert(edge);
                wasm_bonds.push(WasmBond {
                    i: i.min(j),
                    j: i.max(j),
                    order: bond_kind_to_order(&bond.kind),
                });
            }
        }
    }

    Ok(WasmMol {
        atoms: wasm_atoms,
        bonds: wasm_bonds,
        residue_names: Vec::new(),
    })
}

/// Extract atomic number and formal charge from an AtomKind.
fn extract_atomic_info(atom_kind: &AtomKind) -> Result<(u8, i8), String> {
    match atom_kind {
        AtomKind::Star => Err("Star atoms not supported".to_string()),
        AtomKind::Aliphatic(aliphatic) => {
            let atomic_num = aliphatic_to_atomic_num(aliphatic)?;
            Ok((atomic_num, 0))
        }
        AtomKind::Aromatic(aromatic) => {
            let atomic_num = aromatic_to_atomic_num(aromatic)?;
            Ok((atomic_num, 0))
        }
        AtomKind::Bracket {
            symbol,
            charge,
            ..
        } => {
            let atomic_num = bracket_symbol_to_atomic_num(symbol)?;
            let formal_charge = charge.as_ref().map(|c| charge_to_i8(c)).unwrap_or(0);
            Ok((atomic_num, formal_charge))
        }
    }
}

/// Convert Aliphatic to atomic number.
fn aliphatic_to_atomic_num(aliphatic: &Aliphatic) -> Result<u8, String> {
    use purr::feature::Aliphatic::*;
    Ok(match aliphatic {
        B => 5,
        C => 6,
        N => 7,
        O => 8,
        S => 16,
        P => 15,
        F => 9,
        Cl => 17,
        Br => 35,
        I => 53,
        At => 85,
        Ts => 117,
    })
}

/// Convert Aromatic to atomic number.
fn aromatic_to_atomic_num(aromatic: &Aromatic) -> Result<u8, String> {
    use purr::feature::Aromatic::*;
    Ok(match aromatic {
        B => 5,
        C => 6,
        N => 7,
        O => 8,
        S => 16,
        P => 15,
    })
}

/// Convert BracketSymbol to atomic number.
fn bracket_symbol_to_atomic_num(symbol: &BracketSymbol) -> Result<u8, String> {
    match symbol {
        BracketSymbol::Star => Err("Star atoms not supported".to_string()),
        BracketSymbol::Element(element) => element_to_atomic_num(element),
        BracketSymbol::Aromatic(bracket_aromatic) => bracket_aromatic_to_atomic_num(bracket_aromatic),
    }
}

/// Convert BracketAromatic to atomic number.
fn bracket_aromatic_to_atomic_num(aromatic: &purr::feature::BracketAromatic) -> Result<u8, String> {
    use purr::feature::BracketAromatic::*;
    Ok(match aromatic {
        B => 5,
        C => 6,
        N => 7,
        O => 8,
        S => 16,
        P => 15,
        Se => 34,
        As => 33,
    })
}

/// Convert Element to atomic number.
fn element_to_atomic_num(element: &Element) -> Result<u8, String> {
    use purr::feature::Element::*;
    Ok(match element {
        H => 1,
        He => 2,
        Li => 3,
        Be => 4,
        B => 5,
        C => 6,
        N => 7,
        O => 8,
        F => 9,
        Ne => 10,
        Na => 11,
        Mg => 12,
        Al => 13,
        Si => 14,
        P => 15,
        S => 16,
        Cl => 17,
        Ar => 18,
        K => 19,
        Ca => 20,
        Sc => 21,
        Ti => 22,
        V => 23,
        Cr => 24,
        Mn => 25,
        Fe => 26,
        Co => 27,
        Ni => 28,
        Cu => 29,
        Zn => 30,
        Ga => 31,
        Ge => 32,
        As => 33,
        Se => 34,
        Br => 35,
        Kr => 36,
        Rb => 37,
        Sr => 38,
        Y => 39,
        Zr => 40,
        Nb => 41,
        Mo => 42,
        Tc => 43,
        Ru => 44,
        Rh => 45,
        Pd => 46,
        Ag => 47,
        Cd => 48,
        In => 49,
        Sn => 50,
        Sb => 51,
        Te => 52,
        I => 53,
        Xe => 54,
        Cs => 55,
        Ba => 56,
        La => 57,
        Ce => 58,
        Pr => 59,
        Nd => 60,
        Pm => 61,
        Sm => 62,
        Eu => 63,
        Gd => 64,
        Tb => 65,
        Dy => 66,
        Ho => 67,
        Er => 68,
        Tm => 69,
        Yb => 70,
        Lu => 71,
        Hf => 72,
        Ta => 73,
        W => 74,
        Re => 75,
        Os => 76,
        Ir => 77,
        Pt => 78,
        Au => 79,
        Hg => 80,
        Tl => 81,
        Pb => 82,
        Bi => 83,
        Po => 84,
        At => 85,
        Rn => 86,
        Fr => 87,
        Ra => 88,
        Ac => 89,
        Th => 90,
        Pa => 91,
        U => 92,
        Np => 93,
        Pu => 94,
        Am => 95,
        Cm => 96,
        Bk => 97,
        Cf => 98,
        Es => 99,
        Fm => 100,
        Md => 101,
        No => 102,
        Lr => 103,
        Rf => 104,
        Db => 105,
        Sg => 106,
        Bh => 107,
        Hs => 108,
        Mt => 109,
        Ds => 110,
        Rg => 111,
        Cn => 112,
        Nh => 113,
        Fl => 114,
        Mc => 115,
        Lv => 116,
        Ts => 117,
        Og => 118,
    })
}

/// Convert Charge to i8.
fn charge_to_i8(charge: &purr::feature::Charge) -> i8 {
    use purr::feature::Charge::*;
    match charge {
        MinusFifteen => -15,
        MinusFourteen => -14,
        MinusThirteen => -13,
        MinusTwelve => -12,
        MinusEleven => -11,
        MinusTen => -10,
        MinusNine => -9,
        MinusEight => -8,
        MinusSeven => -7,
        MinusSix => -6,
        MinusFive => -5,
        MinusFour => -4,
        MinusThree => -3,
        MinusTwo => -2,
        MinusOne => -1,
        Zero => 0,
        One => 1,
        Two => 2,
        Three => 3,
        Four => 4,
        Five => 5,
        Six => 6,
        Seven => 7,
        Eight => 8,
        Nine => 9,
        Ten => 10,
        Eleven => 11,
        Twelve => 12,
        Thirteen => 13,
        Fourteen => 14,
        Fifteen => 15,
    }
}

/// Convert BondKind to BondOrder.
fn bond_kind_to_order(kind: &BondKind) -> BondOrder {
    match kind {
        BondKind::Elided | BondKind::Single | BondKind::Up | BondKind::Down => BondOrder::Single,
        BondKind::Double => BondOrder::Double,
        BondKind::Triple => BondOrder::Triple,
        BondKind::Quadruple => BondOrder::Quadruple,
        BondKind::Aromatic => BondOrder::Aromatic,
    }
}

/// Detect ring membership via iterative DFS — safe for WASM (no stack overflow).
fn detect_rings(atoms: &[purr::graph::Atom]) -> Vec<bool> {
    let n = atoms.len();
    let mut in_ring = vec![false; n];
    if n == 0 {
        return in_ring;
    }

    // Build adjacency list
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, atom) in atoms.iter().enumerate() {
        for bond in &atom.bonds {
            adj[i].push(bond.tid);
        }
    }

    let mut visited = vec![false; n];
    let mut parent = vec![usize::MAX; n];

    // Iterative DFS — one pass per connected component
    for start in 0..n {
        if visited[start] {
            continue;
        }

        // Stack entries: (node, index_into_adj[node])
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        visited[start] = true;

        while let Some(&mut (node, ref mut ni)) = stack.last_mut() {
            if *ni >= adj[node].len() {
                stack.pop();
                continue;
            }
            let neighbor = adj[node][*ni];
            *ni += 1;

            // Skip the edge back to this node's parent (undirected graph)
            if neighbor == parent[node] {
                continue;
            }

            if !visited[neighbor] {
                visited[neighbor] = true;
                parent[neighbor] = node;
                stack.push((neighbor, 0));
            } else {
                // Back edge found — trace all atoms on the cycle path
                let mut cur = node;
                while cur != neighbor && cur != usize::MAX {
                    in_ring[cur] = true;
                    cur = parent[cur];
                }
                in_ring[neighbor] = true;
            }
        }
    }

    in_ring
}

/// Detect aromatic atoms from SMILES notation (Aromatic vs Aliphatic)
fn detect_aromaticity(atoms: &[purr::graph::Atom]) -> Vec<bool> {
    let n = atoms.len();
    let mut is_aromatic = vec![false; n];

    for (i, atom) in atoms.iter().enumerate() {
        is_aromatic[i] = matches!(atom.kind, AtomKind::Aromatic(_));
    }

    is_aromatic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_methane() {
        let mol = parse("C").unwrap();
        assert_eq!(mol.atoms.len(), 1, "methane should have 1 heavy atom");
        assert_eq!(mol.atoms[0].atomic_num, 6, "carbon atom");
        assert_eq!(mol.atoms[0].implicit_h, 4, "methane: 4 implicit H");
        assert_eq!(mol.atoms[0].formal_charge, 0);
    }

    #[test]
    fn parse_water() {
        let mol = parse("O").unwrap();
        assert_eq!(mol.atoms.len(), 1, "water should have 1 heavy atom");
        assert_eq!(mol.atoms[0].atomic_num, 8, "oxygen atom");
        assert_eq!(mol.atoms[0].implicit_h, 2, "water: 2 implicit H");
    }

    #[test]
    fn parse_ethanol() {
        let mol = parse("CCO").unwrap();
        assert_eq!(mol.atoms.len(), 3, "ethanol: C-C-O");
        assert_eq!(mol.bonds.len(), 2, "ethanol: 2 bonds");
        assert_eq!(mol.atoms[0].atomic_num, 6);
        assert_eq!(mol.atoms[1].atomic_num, 6);
        assert_eq!(mol.atoms[2].atomic_num, 8);
    }

    #[test]
    fn parse_benzene() {
        let mol = parse("c1ccccc1").unwrap();
        assert_eq!(mol.atoms.len(), 6, "benzene: 6 carbons");
        assert!(
            mol.atoms.iter().all(|a| a.in_ring),
            "all benzene carbons should be in ring"
        );
    }

    #[test]
    fn parse_ethane() {
        let mol = parse("CC").unwrap();
        assert_eq!(mol.atoms.len(), 2, "ethane: 2 carbons");
        assert_eq!(mol.bonds.len(), 1, "ethane: 1 C-C bond");
        assert_eq!(mol.bonds[0].order, BondOrder::Single);
    }

    #[test]
    fn parse_invalid_smiles() {
        assert!(parse("not-smiles!!").is_err(), "invalid SMILES should error");
    }

    #[test]
    fn parse_ethene() {
        let mol = parse("C=C").unwrap();
        assert_eq!(mol.atoms.len(), 2);
        assert_eq!(mol.bonds.len(), 1);
        assert_eq!(mol.bonds[0].order, BondOrder::Double);
    }

    #[test]
    fn parse_acetylene() {
        let mol = parse("C#C").unwrap();
        assert_eq!(mol.atoms.len(), 2);
        assert_eq!(mol.bonds.len(), 1);
        assert_eq!(mol.bonds[0].order, BondOrder::Triple);
    }

    #[test]
    fn acyclic_propane_no_ring() {
        let mol = parse("CCC").unwrap();
        assert!(
            mol.atoms.iter().all(|a| !a.in_ring),
            "propane has no ring atoms, got: {:?}",
            mol.atoms.iter().map(|a| a.in_ring).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cyclohexane_all_in_ring() {
        let mol = parse("C1CCCCC1").unwrap();
        assert_eq!(mol.atoms.len(), 6, "cyclohexane: 6 carbons");
        assert!(
            mol.atoms.iter().all(|a| a.in_ring),
            "all cyclohexane atoms should be in ring"
        );
    }
}
