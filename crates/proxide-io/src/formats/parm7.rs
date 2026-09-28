//! AMBER parm7 / prmtop topology reader.
//!
//! A parm7 file is a sequence of `%FLAG <NAME>` sections, each followed by a
//! `%FORMAT(<n><type><width>[.<prec>])` line and fixed-width Fortran records.
//! Fields are packed by WIDTH, not separated by whitespace -- atom names like
//! `HE21HE22` sit in adjacent 4-character cells -- so every section is split by
//! the width its own `%FORMAT` line declares.
//!
//! A topology has no coordinates. [`Parm7Topology::to_raw_atom_data`] combines it
//! with one frame of coordinates (e.g. from an XTC trajectory) into the same
//! [`RawAtomData`] the PDB/mmCIF parsers produce, so it flows through the shared
//! structure pipeline unchanged.
//!
//! Format reference: <https://ambermd.org/prmtop.pdf>.

use proxide_core::structure::{AtomRecord, RawAtomData};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

/// Errors from parm7 parsing.
#[derive(Error, Debug)]
pub enum Parm7Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("missing required %FLAG {0}")]
    MissingFlag(&'static str),
    #[error("%FLAG {flag}: {msg}")]
    Parse { flag: String, msg: String },
    #[error("%FLAG {flag}: expected {expected} values, found {found}")]
    Count {
        flag: String,
        expected: usize,
        found: usize,
    },
    #[error(
        "parm7 has no ATOMIC_NUMBER section; refusing to guess elements from names or \
         masses (AMBER atom names are not element-prefixed reliably, and hydrogen mass \
         repartitioning makes masses ambiguous). Regenerate the topology with a modern \
         tleap/parmed."
    )]
    NoAtomicNumbers,
    #[error("atomic number {z} (atom {atom}) is outside the supported table")]
    UnknownAtomicNumber { z: i64, atom: usize },
    #[error("coordinate length {got} does not match 3 * {n_atoms} atoms")]
    CoordLength { got: usize, n_atoms: usize },
}

/// One `%FLAG` section: field width from its `%FORMAT` line plus raw record lines.
struct Section {
    width: usize,
    lines: Vec<String>,
}

impl Section {
    /// Split every record line into `width`-sized cells (newline already stripped).
    fn cells(&self) -> impl Iterator<Item = &str> {
        let width = self.width;
        self.lines.iter().flat_map(move |line| {
            (0..line.len())
                .step_by(width)
                .map(move |start| &line[start..(start + width).min(line.len())])
        })
    }

    fn ints(&self, flag: &str) -> Result<Vec<i64>, Parm7Error> {
        self.cells()
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .map(|c| {
                c.parse::<i64>().map_err(|_| Parm7Error::Parse {
                    flag: flag.to_string(),
                    msg: format!("invalid integer {c:?}"),
                })
            })
            .collect()
    }

    fn floats(&self, flag: &str) -> Result<Vec<f32>, Parm7Error> {
        self.cells()
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .map(|c| {
                c.parse::<f32>().map_err(|_| Parm7Error::Parse {
                    flag: flag.to_string(),
                    msg: format!("invalid float {c:?}"),
                })
            })
            .collect()
    }

    /// Fixed-width strings. Blank cells are kept (a residue label can never be blank,
    /// but truncating on the first blank would silently shift every later value).
    fn strings(&self, count: usize, flag: &str) -> Result<Vec<String>, Parm7Error> {
        let out: Vec<String> = self.cells().take(count).map(|c| c.trim().to_string()).collect();
        if out.len() != count {
            return Err(Parm7Error::Count {
                flag: flag.to_string(),
                expected: count,
                found: out.len(),
            });
        }
        Ok(out)
    }
}

/// Parse `%FORMAT(20a4)` / `%FORMAT(10I8)` / `%FORMAT(5E16.8)` into the field width.
fn parse_format_width(line: &str) -> Result<usize, String> {
    let inner = line
        .split_once('(')
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(inner, _)| inner.trim())
        .ok_or_else(|| format!("malformed FORMAT line {line:?}"))?;
    let type_pos = inner
        .find(|c: char| c.is_ascii_alphabetic())
        .ok_or_else(|| format!("no type letter in {inner:?}"))?;
    let after_type = &inner[type_pos + 1..];
    let width_str = after_type.split('.').next().unwrap_or("");
    width_str
        .parse::<usize>()
        .ok()
        .filter(|w| *w > 0)
        .ok_or_else(|| format!("no field width in {inner:?}"))
}

fn read_sections<R: BufRead>(reader: R) -> Result<HashMap<String, Section>, Parm7Error> {
    let mut sections: HashMap<String, Section> = HashMap::new();
    let mut current: Option<String> = None;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim_end_matches(['\r', '\n']);
        if let Some(rest) = line.strip_prefix("%FLAG") {
            let flag = rest.trim().to_string();
            sections.insert(
                flag.clone(),
                Section {
                    width: 0,
                    lines: Vec::new(),
                },
            );
            current = Some(flag);
        } else if line.starts_with("%FORMAT") {
            let Some(flag) = current.as_ref() else { continue };
            let width = parse_format_width(line).map_err(|msg| Parm7Error::Parse {
                flag: flag.clone(),
                msg,
            })?;
            if let Some(section) = sections.get_mut(flag) {
                section.width = width;
            }
        } else if line.starts_with('%') {
            // %VERSION / %COMMENT lines carry no data.
        } else if let Some(flag) = current.as_ref() {
            if let Some(section) = sections.get_mut(flag) {
                section.lines.push(line.to_string());
            }
        }
    }

    for (flag, section) in &sections {
        if section.width == 0 && !section.lines.is_empty() {
            return Err(Parm7Error::Parse {
                flag: flag.clone(),
                msg: "data lines without a %FORMAT line".to_string(),
            });
        }
    }
    Ok(sections)
}

/// Element symbols by atomic number (index = Z), uppercase like the PDB element column.
const ELEMENTS: [&str; 87] = [
    "", "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG", "AL", "SI", "P", "S",
    "CL", "AR", "K", "CA", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE",
    "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD",
    "IN", "SN", "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD",
    "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
    "TL", "PB", "BI", "PO", "AT", "RN",
];

/// Element for an extra point / virtual site (TIP4P `EPW`, lone pairs), which AMBER
/// writes with ATOMIC_NUMBER 0 or -1. It is not an element; this label makes that
/// explicit instead of substituting a real one.
pub const EXTRA_POINT_ELEMENT: &str = "EP";

/// Chain id given to every water and ion residue. `~` sorts after every alphanumeric
/// chain label, so solvent never shifts the index of a real chain.
pub const SOLVENT_CHAIN_ID: &str = "~";

/// Amino-acid residue names a parm7 may carry, including AMBER protonation/disulfide
/// variants. Residues NOT in this list are marked HETATM so the structure pipeline
/// classifies them as ligand/solvent/ion rather than as unknown protein residues.
const AMINO_ACIDS: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "HSD", "HSE", "HSP",
    "CYX", "CYM", "ASH", "GLH", "LYN",
];

const WATER_RESIDUES: &[&str] = &["WAT", "HOH", "TIP3", "TP3", "TIP4", "T4P", "TIP5", "SOL", "SPC", "OPC"];

/// A single-atom residue carrying a monatomic ion (AMBER labels: `Na+`, `Cl-`, `K+`,
/// `MG`, `Zn`, ...).
fn is_ion_residue(n_atoms: usize, atomic_number: i64) -> bool {
    n_atoms == 1 && !matches!(atomic_number, 1 | 6 | 7 | 8 | 15 | 16)
}

/// Parsed AMBER topology. Per-residue fields are indexed by residue, per-atom by atom.
#[derive(Debug, Clone)]
pub struct Parm7Topology {
    pub n_atoms: usize,
    pub n_residues: usize,
    /// Per atom.
    pub atom_names: Vec<String>,
    /// Per atom, uppercase symbols (see [`EXTRA_POINT_ELEMENT`]).
    pub elements: Vec<String>,
    /// Per atom, AMBER mass units (amu).
    pub masses: Vec<f32>,
    /// Per atom, elementary charges (the file stores e * 18.2223).
    pub charges: Vec<f32>,
    /// Per residue, verbatim labels (AMBER variants such as `HIE`/`CYX` preserved).
    pub res_names: Vec<String>,
    /// Per residue, 0-based index of the residue's first atom.
    pub res_first_atom: Vec<usize>,
    /// Per residue: `RESIDUE_NUMBER` if present, else 1-based sequential index.
    pub res_ids: Vec<i32>,
    /// Per residue chain label (see [`Parm7Topology::chain_policy`]).
    pub res_chain_ids: Vec<String>,
    /// Per residue: true for waters and monatomic ions.
    pub res_is_solvent: Vec<bool>,
    /// 0-based atom index pairs, from BONDS_INC_HYDROGEN + BONDS_WITHOUT_HYDROGEN.
    pub bonds: Vec<(usize, usize)>,
}

/// AMBER stores charges pre-multiplied by this factor (sqrt of the Coulomb constant in
/// kcal*A/(mol*e^2)), per the prmtop specification.
pub const AMBER_CHARGE_SCALE: f32 = 18.2223;

impl Parm7Topology {
    /// How chains are assigned, for documentation and error messages.
    pub fn chain_policy() -> &'static str {
        "RESIDUE_CHAINID when present; otherwise connected components of the bond graph \
         (disulfide S-S bonds excluded, so a disulfide never fuses two chains) labelled \
         A-Z, a-z, 0-9 in order of first atom; all water/ion residues share chain '~'"
    }

    /// Per-atom residue index.
    pub fn atom_residue_index(&self) -> Vec<usize> {
        let mut out = vec![0usize; self.n_atoms];
        for r in 0..self.n_residues {
            let start = self.res_first_atom[r];
            let end = self
                .res_first_atom
                .get(r + 1)
                .copied()
                .unwrap_or(self.n_atoms);
            for slot in &mut out[start..end] {
                *slot = r;
            }
        }
        out
    }

    /// Combine this topology with one frame of coordinates (flat `x0,y0,z0,...`, Å)
    /// into [`RawAtomData`] in the same shape the PDB parser produces.
    pub fn to_raw_atom_data(&self, coords_angstrom: &[f32]) -> Result<RawAtomData, Parm7Error> {
        let mut raw = RawAtomData::with_capacity(self.n_atoms);
        self.push_frame(&mut raw, coords_angstrom)?;
        Ok(raw)
    }

    /// Several frames as one multi-model [`RawAtomData`] plus per-atom 1-based model
    /// ids -- the same `(raw, model_ids)` pair `parse_pdb_file` returns for an NMR file.
    pub fn to_multi_model_raw_atom_data(
        &self,
        frames: &[Vec<f32>],
    ) -> Result<(RawAtomData, Vec<usize>), Parm7Error> {
        let mut raw = RawAtomData::with_capacity(self.n_atoms * frames.len());
        let mut model_ids = Vec::with_capacity(self.n_atoms * frames.len());
        for (position, coords) in frames.iter().enumerate() {
            self.push_frame(&mut raw, coords)?;
            model_ids.extend(std::iter::repeat(position + 1).take(self.n_atoms));
        }
        Ok((raw, model_ids))
    }

    fn push_frame(&self, raw: &mut RawAtomData, coords_angstrom: &[f32]) -> Result<(), Parm7Error> {
        if coords_angstrom.len() != self.n_atoms * 3 {
            return Err(Parm7Error::CoordLength {
                got: coords_angstrom.len(),
                n_atoms: self.n_atoms,
            });
        }
        let res_of_atom = self.atom_residue_index();
        for i in 0..self.n_atoms {
            let r = res_of_atom[i];
            let res_name = &self.res_names[r];
            raw.add_atom(AtomRecord {
                serial: (i + 1) as i32,
                atom_name: self.atom_names[i].clone(),
                alt_loc: ' ',
                res_name: res_name.clone(),
                chain_id: self.res_chain_ids[r].clone(),
                res_seq: self.res_ids[r],
                i_code: ' ',
                x: coords_angstrom[3 * i],
                y: coords_angstrom[3 * i + 1],
                z: coords_angstrom[3 * i + 2],
                occupancy: 1.0,
                temp_factor: 0.0,
                element: self.elements[i].clone(),
                charge: Some(self.charges[i]),
                radius: None,
                is_hetatm: !AMINO_ACIDS.contains(&res_name.to_uppercase().as_str()),
            });
        }
        Ok(())
    }
}

/// Parse a parm7/prmtop file.
pub fn parse_parm7<P: AsRef<Path>>(path: P) -> Result<Parm7Topology, Parm7Error> {
    parse_parm7_reader(BufReader::new(File::open(path)?))
}

fn require<'a>(
    sections: &'a HashMap<String, Section>,
    flag: &'static str,
) -> Result<&'a Section, Parm7Error> {
    sections.get(flag).ok_or(Parm7Error::MissingFlag(flag))
}

fn check_count(flag: &str, expected: usize, found: usize) -> Result<(), Parm7Error> {
    if expected == found {
        Ok(())
    } else {
        Err(Parm7Error::Count {
            flag: flag.to_string(),
            expected,
            found,
        })
    }
}

/// Parse parm7 content from any reader.
pub fn parse_parm7_reader<R: BufRead>(reader: R) -> Result<Parm7Topology, Parm7Error> {
    let sections = read_sections(reader)?;

    let pointers = require(&sections, "POINTERS")?.ints("POINTERS")?;
    if pointers.len() < 12 {
        return Err(Parm7Error::Count {
            flag: "POINTERS".to_string(),
            expected: 12,
            found: pointers.len(),
        });
    }
    let n_atoms = pointers[0] as usize;
    let n_residues = pointers[11] as usize;

    let atom_names = require(&sections, "ATOM_NAME")?.strings(n_atoms, "ATOM_NAME")?;
    let res_names = require(&sections, "RESIDUE_LABEL")?.strings(n_residues, "RESIDUE_LABEL")?;

    let res_ptr = require(&sections, "RESIDUE_POINTER")?.ints("RESIDUE_POINTER")?;
    check_count("RESIDUE_POINTER", n_residues, res_ptr.len())?;
    let mut res_first_atom = Vec::with_capacity(n_residues);
    for (r, &p) in res_ptr.iter().enumerate() {
        let first = p - 1;
        let prev = res_first_atom.last().copied().map(|x: usize| x as i64);
        if first < 0 || first as usize >= n_atoms.max(1) || prev.is_some_and(|q| first <= q) {
            return Err(Parm7Error::Parse {
                flag: "RESIDUE_POINTER".to_string(),
                msg: format!("residue {r} pointer {p} is not strictly increasing within 1..={n_atoms}"),
            });
        }
        res_first_atom.push(first as usize);
    }

    let masses = require(&sections, "MASS")?.floats("MASS")?;
    check_count("MASS", n_atoms, masses.len())?;
    let charges: Vec<f32> = require(&sections, "CHARGE")?
        .floats("CHARGE")?
        .into_iter()
        .map(|q| q / AMBER_CHARGE_SCALE)
        .collect();
    check_count("CHARGE", n_atoms, charges.len())?;

    let atomic_numbers = sections
        .get("ATOMIC_NUMBER")
        .ok_or(Parm7Error::NoAtomicNumbers)?
        .ints("ATOMIC_NUMBER")?;
    check_count("ATOMIC_NUMBER", n_atoms, atomic_numbers.len())?;
    let elements = atomic_numbers
        .iter()
        .enumerate()
        .map(|(atom, &z)| {
            if z <= 0 {
                Ok(EXTRA_POINT_ELEMENT.to_string())
            } else {
                ELEMENTS
                    .get(z as usize)
                    .map(|s| s.to_string())
                    .ok_or(Parm7Error::UnknownAtomicNumber { z, atom })
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut bonds = Vec::new();
    for flag in ["BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"] {
        let Some(section) = sections.get(flag) else { continue };
        let values = section.ints(flag)?;
        if values.len() % 3 != 0 {
            return Err(Parm7Error::Parse {
                flag: flag.to_string(),
                msg: format!("{} values is not a multiple of 3 (i, j, type)", values.len()),
            });
        }
        for triple in values.chunks_exact(3) {
            // Atom indices are stored as 3 * (0-based index), a legacy of coordinate-array offsets.
            let (a, b) = ((triple[0] / 3) as usize, (triple[1] / 3) as usize);
            if a >= n_atoms || b >= n_atoms {
                return Err(Parm7Error::Parse {
                    flag: flag.to_string(),
                    msg: format!("bond ({a}, {b}) out of range for {n_atoms} atoms"),
                });
            }
            bonds.push((a.min(b), a.max(b)));
        }
    }

    let res_ids: Vec<i32> = match sections.get("RESIDUE_NUMBER") {
        Some(section) => {
            let ids = section.ints("RESIDUE_NUMBER")?;
            check_count("RESIDUE_NUMBER", n_residues, ids.len())?;
            ids.into_iter().map(|v| v as i32).collect()
        }
        None => (1..=n_residues as i32).collect(),
    };

    let mut topo = Parm7Topology {
        n_atoms,
        n_residues,
        atom_names,
        elements,
        masses,
        charges,
        res_names,
        res_first_atom,
        res_ids,
        res_chain_ids: Vec::new(),
        res_is_solvent: Vec::new(),
        bonds,
    };

    let res_of_atom = topo.atom_residue_index();
    let mut res_atom_count = vec![0usize; n_residues];
    for &r in &res_of_atom {
        res_atom_count[r] += 1;
    }
    topo.res_is_solvent = (0..n_residues)
        .map(|r| {
            let name = topo.res_names[r].to_uppercase();
            WATER_RESIDUES.contains(&name.as_str())
                || is_ion_residue(res_atom_count[r], atomic_numbers[topo.res_first_atom[r]])
        })
        .collect();

    topo.res_chain_ids = match sections.get("RESIDUE_CHAINID") {
        Some(section) => section.strings(n_residues, "RESIDUE_CHAINID")?,
        None => chains_from_bond_graph(&topo, &res_of_atom),
    };
    Ok(topo)
}

fn union_find_root(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn chain_label(index: usize) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    if index < ALPHABET.len() {
        (ALPHABET[index] as char).to_string()
    } else {
        // Beyond 62 molecules: two-character labels (still sort before SOLVENT_CHAIN_ID).
        let hi = ALPHABET[(index / ALPHABET.len() - 1) % ALPHABET.len()] as char;
        let lo = ALPHABET[index % ALPHABET.len()] as char;
        format!("{hi}{lo}")
    }
}

fn chains_from_bond_graph(topo: &Parm7Topology, res_of_atom: &[usize]) -> Vec<String> {
    let mut parent: Vec<usize> = (0..topo.n_atoms).collect();
    // A residue is one piece of one molecule by definition: join its atoms first, so a
    // residue whose first atom happens to carry no listed bond still lands in its chain.
    for (i, &r) in res_of_atom.iter().enumerate() {
        let first = topo.res_first_atom[r];
        if i != first {
            let (ra, rb) = (union_find_root(&mut parent, first), union_find_root(&mut parent, i));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    for &(a, b) in &topo.bonds {
        // A disulfide links two cysteines that may belong to different chains.
        if topo.elements[a] == "S" && topo.elements[b] == "S" {
            continue;
        }
        let (ra, rb) = (union_find_root(&mut parent, a), union_find_root(&mut parent, b));
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    }

    let mut label_of_root: HashMap<usize, String> = HashMap::new();
    let mut out = Vec::with_capacity(topo.n_residues);
    for r in 0..topo.n_residues {
        if topo.res_is_solvent[r] {
            out.push(SOLVENT_CHAIN_ID.to_string());
            continue;
        }
        let root = union_find_root(&mut parent, topo.res_first_atom[r]);
        let next = label_of_root.len();
        let label = label_of_root
            .entry(root)
            .or_insert_with(|| chain_label(next))
            .clone();
        out.push(label);
    }
    debug_assert_eq!(res_of_atom.len(), topo.n_atoms);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two chains (a 2-residue peptide A and a 1-residue peptide B joined to A only by
    /// the SG5-SG26 disulfide), one water, one Na+. `HE21`/`HE22` are atoms 19/20, so
    /// the packed pair straddles the first 20a4 line break.
    const FIXTURE: &str = include_str!("tests/parm7_fixture.parm7");

    fn fixture() -> Parm7Topology {
        parse_parm7_reader(FIXTURE.as_bytes()).expect("fixture parses")
    }

    #[test]
    fn format_width_parsing() {
        assert_eq!(parse_format_width("%FORMAT(20a4)").unwrap(), 4);
        assert_eq!(parse_format_width("%FORMAT(10I8)").unwrap(), 8);
        assert_eq!(parse_format_width("%FORMAT(5E16.8)").unwrap(), 16);
        assert_eq!(parse_format_width("%FORMAT(1a80)").unwrap(), 80);
        assert!(parse_format_width("%FORMAT()").is_err());
    }

    #[test]
    fn packed_names_split_by_width_across_lines() {
        let t = fixture();
        assert_eq!(t.n_atoms, 31);
        assert_eq!(t.atom_names.len(), 31);
        // Cells 20 and 21 straddle the first line break; a newline-shifted parser
        // would misalign every name after it.
        assert_eq!(t.atom_names[19], "HE21");
        assert_eq!(t.atom_names[20], "HE22");
        assert_eq!(t.atom_names[21], "N");
        assert_eq!(t.atom_names[30], "Na+");
    }

    #[test]
    fn residues_charges_elements() {
        let t = fixture();
        assert_eq!(t.res_names, vec!["CYX", "GLN", "CYX", "WAT", "Na+"]);
        assert_eq!(t.res_first_atom, vec![0, 6, 21, 27, 30]);
        assert_eq!(t.elements[0], "N");
        assert_eq!(t.elements[5], "S");
        assert_eq!(t.elements[30], "NA");
        assert!((t.charges[30] - 1.0).abs() < 1e-5, "charge divided by 18.2223");
    }

    #[test]
    fn bonds_are_divided_by_three() {
        let t = fixture();
        assert!(t.bonds.contains(&(0, 1)));
        assert!(t.bonds.contains(&(5, 26)), "disulfide S(5)-S(26) present");
    }

    #[test]
    fn chains_split_at_disulfide_and_solvent_shares_one_chain() {
        let t = fixture();
        assert_eq!(t.res_chain_ids, vec!["A", "A", "B", "~", "~"]);
        assert_eq!(t.res_is_solvent, vec![false, false, false, true, true]);
    }

    #[test]
    fn residue_with_unbonded_first_atom_keeps_its_chain() {
        let mut t = fixture();
        t.bonds.retain(|&(a, b)| a != 0 && b != 0);
        let res_of_atom = t.atom_residue_index();
        assert_eq!(chains_from_bond_graph(&t, &res_of_atom), vec!["A", "A", "B", "~", "~"]);
    }

    #[test]
    fn raw_atom_data_is_per_atom() {
        let t = fixture();
        let coords: Vec<f32> = (0..t.n_atoms * 3).map(|v| v as f32).collect();
        let raw = t.to_raw_atom_data(&coords).unwrap();
        assert_eq!(raw.num_atoms, 31);
        assert_eq!(raw.res_names.len(), 31);
        assert_eq!(raw.chain_ids[22], "B");
        assert_eq!(raw.res_ids[22], 3);
        assert!(!raw.is_hetatm[0] && raw.is_hetatm[27] && raw.is_hetatm[30]);
        assert!(t.to_raw_atom_data(&coords[..6]).is_err());

        let (multi, model_ids) = t
            .to_multi_model_raw_atom_data(&[coords.clone(), coords.clone()])
            .unwrap();
        assert_eq!(multi.num_atoms, 62);
        assert_eq!(model_ids.len(), 62);
        assert_eq!((model_ids[30], model_ids[31]), (1, 2));
    }

    #[test]
    fn missing_atomic_numbers_is_an_error() {
        let stripped: String = {
            let mut out = String::new();
            let mut skip = false;
            for line in FIXTURE.lines() {
                if line.starts_with("%FLAG") {
                    skip = line.contains("ATOMIC_NUMBER");
                }
                if !skip {
                    out.push_str(line);
                    out.push('\n');
                }
            }
            out
        };
        assert!(matches!(
            parse_parm7_reader(stripped.as_bytes()),
            Err(Parm7Error::NoAtomicNumbers)
        ));
    }

    #[test]
    fn chain_labels_never_collide_with_solvent() {
        for i in [0, 25, 61, 62, 100, 5000] {
            let label = chain_label(i);
            assert_ne!(label, SOLVENT_CHAIN_ID);
            assert!(label.as_str() < SOLVENT_CHAIN_ID);
        }
    }
}
