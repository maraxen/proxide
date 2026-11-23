//! Test support for the insertion-code and hetero tests (spec 261123, T1/T2).
//!
//! The expectations are derived by reading `tests/data/icode_ligand.pdb` in the
//! test itself: the record type, column 27 and the residue/atom identity of
//! every ATOM/HETATM line. Nothing here hard-codes the atom or residue counts.

use crate::formats::pdb::parse_pdb_file;
use proxide_core::processing::ProcessedStructure;
use std::path::PathBuf;

/// One ATOM or HETATM record of the fixture, read by fixed columns.
#[derive(Debug, Clone, PartialEq)]
pub struct FileAtom {
    /// True for a HETATM record (columns 1-6).
    pub hetatm: bool,
    pub atom_name: String,
    pub res_name: String,
    pub chain: String,
    pub res_seq: i32,
    /// Insertion code, column 27 (`' '` when blank).
    pub icode: char,
}

/// Key that identifies one atom of the fixture across formatters.
pub type AtomKey = (String, i32, String, String);

pub fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/icode_ligand.pdb")
}

/// Every ATOM/HETATM record of the fixture, in file order.
pub fn read_file_atoms() -> Vec<FileAtom> {
    let text = std::fs::read_to_string(fixture_path()).expect("read icode_ligand.pdb");
    text.lines()
        .filter_map(|line| {
            let record = line.get(0..6)?.trim_end();
            if record != "ATOM" && record != "HETATM" {
                return None;
            }
            Some(FileAtom {
                hetatm: record == "HETATM",
                atom_name: line[12..16].trim().to_string(),
                res_name: line[17..20].trim().to_string(),
                chain: line[21..22].to_string(),
                res_seq: line[22..26].trim().parse().expect("resSeq column"),
                icode: line.as_bytes()[26] as char,
            })
        })
        .collect()
}

/// Protein residues in residue order, as `(chain, res_seq, icode)`.
///
/// A residue counts as protein when it has an ATOM record or is MSE, which the
/// pipeline types as a peptide-linking residue (spec F15). Residues are kept in
/// file order and then stable-sorted by `(chain, res_seq)`, the order the
/// residue pipeline produces. Two residues that share a number differ only by
/// their insertion code, so the sort keeps their file order.
pub fn expected_protein_residues(atoms: &[FileAtom]) -> Vec<(String, i32, char)> {
    let mut residues: Vec<((String, i32, char), bool)> = Vec::new();
    for atom in atoms {
        let key = (atom.chain.clone(), atom.res_seq, atom.icode);
        let is_protein = !atom.hetatm || atom.res_name == "MSE";
        match residues.iter_mut().find(|(k, _)| *k == key) {
            Some((_, protein)) => *protein |= is_protein,
            None => residues.push((key, is_protein)),
        }
    }
    let mut protein: Vec<(String, i32, char)> = residues
        .into_iter()
        .filter_map(|(key, is_protein)| is_protein.then_some(key))
        .collect();
    protein.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    protein
}

/// The key of a fixture atom, as the formatters report it.
pub fn file_atom_key(atom: &FileAtom) -> AtomKey {
    (
        atom.chain.clone(),
        atom.res_seq,
        atom.res_name.clone(),
        atom.atom_name.clone(),
    )
}

/// Parse the fixture through the PDB reader and the residue pipeline.
pub fn load_processed() -> ProcessedStructure {
    let (raw, _model_ids) = parse_pdb_file(fixture_path()).expect("parse icode_ligand.pdb");
    ProcessedStructure::from_raw(raw).expect("process icode_ligand.pdb")
}
