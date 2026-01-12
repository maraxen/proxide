// TODO: Review allow attributes at a later point
#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use crate::physics::nerf::Nerf;
use crate::structure::systems::AtomicSystem;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub mod db;

// Constants from foldcomp.h/cpp
const MAGICNUMBER: &[u8] = b"FCMP";
const N_TO_CA_DIST: f32 = 1.4581;
const PRO_N_TO_CA_DIST: f32 = 1.353;
const CA_TO_C_DIST: f32 = 1.5281;
const C_TO_N_DIST: f32 = 1.3311;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CompressedFileHeaderRaw {
    n_residue: u16,
    n_atom: u16,
    idx_residue: u16,
    idx_atom: u16,
    n_anchor: u8,
    chain: u8,
    _pad1: [u8; 2],
    n_side_chain_torsion: u32,
    first_residue: u8,
    last_residue: u8,
    _pad2: [u8; 2],
    len_title: u32,
    mins: [f32; 6],
    cont_fs: [f32; 6],
}

#[derive(Debug, Clone)]
pub struct BackboneChain {
    pub residue: u8,
    pub omega: u16,
    pub psi: u16,
    pub phi: u16,
    pub ca_c_n_angle: u8,
    pub c_n_ca_angle: u8,
    pub n_ca_c_angle: u8,
}

impl BackboneChain {
    pub fn from_bytes(bytes: &[u8; 8]) -> Self {
        let b0 = bytes[0] as u16;
        let b1 = bytes[1] as u16;
        let b2 = bytes[2] as u16;
        let b3 = bytes[3] as u16;
        let b4 = bytes[4] as u16;

        let residue = (bytes[0] & 0xF8) >> 3;
        let omega = ((b0 & 0x07) << 8) | b1;
        let psi = ((b2 & 0xFF) << 4) | ((b3 & 0xF0) >> 4);
        let phi = ((b3 & 0x0F) << 8) | b4;
        let ca_c_n_angle = bytes[5];
        let c_n_ca_angle = bytes[6];
        let n_ca_c_angle = bytes[7];

        BackboneChain {
            residue,
            omega,
            psi,
            phi,
            ca_c_n_angle,
            c_n_ca_angle,
            n_ca_c_angle,
        }
    }
}

pub fn read_foldcomp<P: AsRef<Path>>(path: P) -> std::io::Result<AtomicSystem> {
    let mut file = File::open(path)?;
    read_foldcomp_from_reader(&mut file)
}

pub fn read_foldcomp_from_reader<R: Read + Seek>(reader: &mut R) -> std::io::Result<AtomicSystem> {
    // 1. Read Magic Number
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != MAGICNUMBER {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid FoldComp file magic number",
        ));
    }

    // 2. Read Header
    let mut header_bytes = [0u8; 72];
    reader.read_exact(&mut header_bytes)?;
    let header: CompressedFileHeaderRaw = unsafe { std::mem::transmute(header_bytes) };

    // 3. Read Anchor Indices
    // Count is header.n_anchor (u8 -> usize)
    let n_anchors = header.n_anchor as usize;
    // Skip n_anchors * sizeof(int)
    reader.seek(SeekFrom::Current((n_anchors * 4) as i64))?;

    // 4. Read Title
    // Skip len_title
    reader.seek(SeekFrom::Current(header.len_title as i64))?;

    // 5. Read Start Atoms (prevAtomCoords)
    // 3 atoms * 3 floats * 4 bytes = 36 bytes
    let mut start_atoms_bytes = [0u8; 36];
    reader.read_exact(&mut start_atoms_bytes)?;
    let start_atoms = parse_atoms_from_bytes(&start_atoms_bytes);

    // 6. Skip Inner Anchor Atoms
    // (n_anchors - 2) * 36 bytes
    if n_anchors > 2 {
        let skip = (n_anchors - 2) * 36;
        reader.seek(SeekFrom::Current(skip as i64))?;
    }

    // 7. Skip Last Anchor Atoms
    // 1 * 36 bytes
    // Only if n_anchors > 0? Assuming yes.
    reader.seek(SeekFrom::Current(36))?;

    // 8. Skip OXT
    // char hasOXT (1)
    reader.seek(SeekFrom::Current(1))?;
    // OXT coords (12)
    reader.seek(SeekFrom::Current(12))?;

    // 9. Read Backbone Records
    let mut records: Vec<BackboneChain> = Vec::with_capacity(header.n_residue as usize);
    let mut buf = [0u8; 8];
    for _ in 0..header.n_residue {
        reader.read_exact(&mut buf)?;
        records.push(BackboneChain::from_bytes(&buf));
    }

    // Reconstruct
    reconstruct(&header, &records, start_atoms).map_err(std::io::Error::other)
}

fn parse_atoms_from_bytes(bytes: &[u8; 36]) -> [[f32; 3]; 3] {
    let mut atoms = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let start = (i * 3 + j) * 4;
            let end = start + 4;
            let val = f32::from_le_bytes(bytes[start..end].try_into().unwrap());
            atoms[i][j] = val;
        }
    }
    atoms
}

fn reconstruct(
    header: &CompressedFileHeaderRaw,
    records: &[BackboneChain],
    start_atoms: [[f32; 3]; 3],
) -> Result<AtomicSystem, String> {
    // vectors for AtomicSystem
    // We expect 3 atoms per residue (N, CA, C).
    let total_atoms = (header.n_residue as usize) * 3;
    let mut coords = Vec::with_capacity(total_atoms * 3);
    let mut atom_names = Vec::with_capacity(total_atoms);

    // Currently AtomicSystem takes elements. Foldcomp doesn't store them.
    // We infer from atom names.
    let mut elements = Vec::with_capacity(total_atoms);

    // We reconstruct full residue details for AtomicSystem
    // But currently `push_atom` collected them.
    // To match AtomicSystem which uses SoA.
    // Let's keep using push_atom but adapt the system creation.

    let mut residue_names = Vec::with_capacity(total_atoms);
    let mut residue_indices = Vec::with_capacity(total_atoms);
    let mut chain_ids = Vec::with_capacity(total_atoms);

    let chain_char = header.chain as char;
    let chain_str = chain_char.to_string(); // Single char string

    // Add start atoms (Residue 0)
    // records[0] has residue info for Residue 0.
    let res0_code = convert_int_to_one_letter_code(records[0].residue);
    let res0_name = one_letter_to_three_letter(res0_code);
    let res0_idx = header.idx_residue as i32;

    // N
    push_atom(
        &mut coords,
        &mut atom_names,
        &mut residue_names,
        &mut residue_indices,
        &mut elements,
        &mut chain_ids,
        start_atoms[0],
        "N",
        &res0_name,
        res0_idx,
        "N",
        &chain_str,
    );
    // CA
    push_atom(
        &mut coords,
        &mut atom_names,
        &mut residue_names,
        &mut residue_indices,
        &mut elements,
        &mut chain_ids,
        start_atoms[1],
        "CA",
        &res0_name,
        res0_idx,
        "C",
        &chain_str,
    );
    // C
    push_atom(
        &mut coords,
        &mut atom_names,
        &mut residue_names,
        &mut residue_indices,
        &mut elements,
        &mut chain_ids,
        start_atoms[2],
        "C",
        &res0_name,
        res0_idx,
        "C",
        &chain_str,
    );

    // Track previous 3 atoms for NeRF
    let mut prev_atoms = start_atoms;

    // Iterate to generate residues 1 to N-1
    // Loop i from 0 to N-2
    let n_residue = header.n_residue as usize;
    if n_residue > 1 {
        for i in 0..(n_residue - 1) {
            // Use records[i] angles to generate residue i+1
            let rec = &records[i];

            // Decompress angles
            let psi = _continuize(rec.psi as u32, header.mins[1], header.cont_fs[1]);
            let omega = _continuize(rec.omega as u32, header.mins[2], header.cont_fs[2]);
            // phi is technically phi of next residue, stored in rec.phi?
            let phi = _continuize(rec.phi as u32, header.mins[0], header.cont_fs[0]);

            let n_ca_c = _continuize(rec.n_ca_c_angle as u32, header.mins[3], header.cont_fs[3]);
            let ca_c_n = _continuize(rec.ca_c_n_angle as u32, header.mins[4], header.cont_fs[4]);
            let c_n_ca = _continuize(rec.c_n_ca_angle as u32, header.mins[5], header.cont_fs[5]);

            // Place N (of i+1)
            let next_n = Nerf::place_atom(&prev_atoms, C_TO_N_DIST, ca_c_n, psi);

            // Update prev_atoms: [CA_i, C_i, N_next]
            let mut buf_atoms = [prev_atoms[1], prev_atoms[2], next_n];

            // Place CA (of i+1)
            let next_res_code = convert_int_to_one_letter_code(records[i + 1].residue);
            let dist = if next_res_code == 'P' {
                PRO_N_TO_CA_DIST
            } else {
                N_TO_CA_DIST
            };

            let next_ca = Nerf::place_atom(&buf_atoms, dist, c_n_ca, omega);

            // Update buf
            buf_atoms = [buf_atoms[1], buf_atoms[2], next_ca];

            // Place C (of i+1)
            let next_c = Nerf::place_atom(&buf_atoms, CA_TO_C_DIST, n_ca_c, phi);

            // Update prev_atoms for next iteration to [N, CA, C] of new residue
            prev_atoms = [next_n, next_ca, next_c];

            // Add atoms
            let res_idx = header.idx_residue as i32 + (i as i32) + 1;
            let next_res_name = one_letter_to_three_letter(next_res_code);

            push_atom(
                &mut coords,
                &mut atom_names,
                &mut residue_names,
                &mut residue_indices,
                &mut elements,
                &mut chain_ids,
                next_n,
                "N",
                &next_res_name,
                res_idx,
                "N",
                &chain_str,
            );
            push_atom(
                &mut coords,
                &mut atom_names,
                &mut residue_names,
                &mut residue_indices,
                &mut elements,
                &mut chain_ids,
                next_ca,
                "CA",
                &next_res_name,
                res_idx,
                "C",
                &chain_str,
            );
            push_atom(
                &mut coords,
                &mut atom_names,
                &mut residue_names,
                &mut residue_indices,
                &mut elements,
                &mut chain_ids,
                next_c,
                "C",
                &next_res_name,
                res_idx,
                "C",
                &chain_str,
            );
        }
    }

    let atom_mask = vec![1.0; coords.len() / 3];

    // Create AtomicSystem
    let mut system = AtomicSystem::new(coords, atom_mask, Some(atom_names), Some(elements));

    // Set additional fields
    // residue_index
    system.residue_index = Some(residue_indices);
    // unique_chain_ids
    system.unique_chain_ids = Some(vec![chain_str]);
    // chain_index: all 0 since one chain
    let num_atoms = system.coordinates.len() / 3;
    system.chain_index = Some(vec![0; num_atoms]);

    Ok(system)
}

#[allow(clippy::too_many_arguments)]
fn push_atom(
    coords: &mut Vec<f32>,
    names: &mut Vec<String>,
    res_names: &mut Vec<String>,
    res_indices: &mut Vec<i32>,
    elements: &mut Vec<String>,
    chains: &mut Vec<String>,
    xyz: [f32; 3],
    name: &str,
    res_name: &str,
    res_idx: i32,
    elem: &str,
    chain: &str,
) {
    coords.push(xyz[0]);
    coords.push(xyz[1]);
    coords.push(xyz[2]);
    names.push(name.to_string());
    res_names.push(res_name.to_string());
    res_indices.push(res_idx);
    elements.push(elem.to_string());
    chains.push(chain.to_string());
}

fn _continuize(val: u32, min: f32, cont_f: f32) -> f32 {
    min + (val as f32) * cont_f
}

fn convert_int_to_one_letter_code(i: u8) -> char {
    match i {
        0 => 'A',
        1 => 'R',
        2 => 'N',
        3 => 'D',
        4 => 'C',
        5 => 'Q',
        6 => 'E',
        7 => 'G',
        8 => 'H',
        9 => 'I',
        10 => 'L',
        11 => 'K',
        12 => 'M',
        13 => 'F',
        14 => 'P',
        15 => 'S',
        16 => 'T',
        17 => 'W',
        18 => 'Y',
        19 => 'V',
        _ => 'X',
    }
}

fn one_letter_to_three_letter(c: char) -> String {
    match c {
        'A' => "ALA",
        'R' => "ARG",
        'N' => "ASN",
        'D' => "ASP",
        'C' => "CYS",
        'Q' => "GLN",
        'E' => "GLU",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'L' => "LEU",
        'K' => "LYS",
        'M' => "MET",
        'F' => "PHE",
        'P' => "PRO",
        'S' => "SER",
        'T' => "THR",
        'W' => "TRP",
        'Y' => "TYR",
        'V' => "VAL",
        _ => "UNK",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuize() {
        // float _continuize(unsigned int input, float min, float cont_f) {
        //     float output = min + ((float)input * cont_f);
        //     return output;
        // }
        // Example: min=0.0, cont_f=0.1, input=5 -> 0.5
        assert!((_continuize(5, 0.0, 0.1) - 0.5).abs() < 1e-6);
        // inputs are integer steps.
    }

    #[test]
    fn test_backbone_chain_unpacking() {
        // Create bytes that correspond to known values.
        // struct BackboneChain {
        //     residue: 5 bits
        //     omega: 11 bits
        //     psi: 12 bits
        //     phi: 12 bits
        //     ca_c_n: 8 bits
        //     c_n_ca: 8 bits
        //     n_ca_c: 8 bits
        // }

        // Let's pack some values manually to verify against C++ logic.
        // residue = 5 (00101)
        // omega = 10 (00000001010)
        // psi = 20 (00000010100)
        // phi = 30 (00000011110)
        // angles = 1, 2, 3

        // Byte 0: ((residue & 0xF8) >> 3) | ((omega & 0x7FF) >> 8) -> but wait
        // C++ packing:
        // output[0] = ((res.residue & 0xF8) >> 3) | ((res.omega & 0x07FF) >> 8);
        // Actually residue is 5 bits?
        // foldcomp.h: uint64_t residue: 5;
        // If residue=5 (00101).
        // (residue & 0xF8)? 5 is 00000101.
        // 5 & 0xF8 is 0.
        // Wait, clearly the macro defines NUM_BITS_RESIDUE as 5.
        // But the C++ code likely shifts it differently?
        // Let's re-read foldcomp.cpp convertBackboneChainToBytes logic (from my memory or view).
        // I viewed convertBytesToBackboneChain logic in Step 47.
        // output[0] = ((res.residue & 0xF8) >> 3) ...
        // If residue is 5 bits, say 0..31.
        // If residue=5 (00101). 00101 & 11111000 (0xF8) is 0.
        // This implies residue is STORED in higher bits?
        // Or `res.residue` in the struct is an integer?
        // In Struct, `uint64_t residue : 5`.
        // If I assign `res.residue = 5`.
        // Then `res.residue & 0xF8`. 5 is 00000101.
        // This seems to imply `residue` values are shifted?
        // OR the C code assumes `residue` holds a value that USES those bits?
        // Ah, maybe `residue` is an index into AA table? 0..20.
        // 5 bits fit 0..31.
        // But `(residue & 0xF8) >> 3`?
        // If residue is 5 bits, maximum is 31 (0x1F).
        // 0x1F & 0xF8 is 0x18 (binary 11000).
        // So this logic EXTRACTS bits 3 and 4?
        // And `omega` provides ...

        // Let's look at `from_bytes` implementation I wrote based on `convertBytesToBackboneChain`:
        // let residue = (bytes[0] & 0xF8) >> 3;
        // This extracts top 5 bits of byte 0?
        // 0xF8 is 11111000.
        // Shift right 3 gives 00011111 (5 bits).
        // So `residue` effectively gets the top 5 bits of Byte 0.
        // Wait, `bytes[0]` has 8 bits.
        // `residue` takes bits 7,6,5,4,3.

        // My code: `let residue = (bytes[0] & 0xF8) >> 3;`. Correct.

        // `omega`.
        // C++: `output[0] = ... | ((res.omega & 0x07FF) >> 8)`.
        // `omega` is 11 bits. 0x7FF.
        // `omega >> 8` gives top 3 bits (bits 10, 9, 8).
        // These are put in bottom 3 bits of Byte 0.
        // So Byte 0 = [Residue(5 bits) | Omega_High(3 bits)].

        // My code for omega: `let omega = ((b0 & 0x07) << 8) | b1;`.
        // `b0 & 0x07` extracts bottom 3 bits of b0. (Which are Omega_High).
        // Shift left 8.
        // OR with `b1`.
        // `b1` comes from `output[1] = res.omega & 0x00FF`. (Bottom 8 bits).
        // So Omega = [Omega_High | Omega_Low].
        // Correct.

        // `psi` (12 bits).
        // `output[2] = ((res.psi & 0x0FFF) >> 4)`.
        // Top 8 bits of Psi (bits 11..4) go to Byte 2.
        // `output[3] = ((res.psi & 0x000F) << 4) | ...`.
        // Bottom 4 bits of Psi (bits 3..0) go to top 4 bits of Byte 3.

        // My code for psi: `let psi = ((b2 & 0xFF) << 4) | ((b3 & 0xF0) >> 4);`.
        // `b2` is Top 8 bits. Shift left 4.
        // `b3 & 0xF0` is Top 4 bits of Byte 3. Shift right 4 -> Bottom 4 bits of Psi.
        // Combined -> 12 bits. Correct.

        // `phi` (12 bits).
        // `output[3] = ... | ((res.phi & 0x0FFF) >> 8)`.
        // Top 4 bits of Phi (bits 11..8) go to Bottom 4 bits of Byte 3.
        // `output[4] = res.phi & 0x00FF`.
        // Bottom 8 bits of Phi go to Byte 4.

        // My code for phi: `let phi = ((b3 & 0x0F) << 8) | b4;`.
        // `b3 & 0x0F` extracts Bottom 4 bits of Byte 3 (Phi High). Shift left 8.
        // `b4` is Phi Low.
        // Combined. Correct.

        // Remaining 3 bytes: 5, 6, 7.
        // Just the angles.

        // Test Case Construction:
        // Residue = 10 (01010)
        // Omega = 1000 (01111101000) -> 0x3E8
        // Psi = 2000 (011111010000) -> 0x7D0
        // Phi = 3000 (101110111000) -> 0xBB8
        // A1 = 100, A2 = 200, A3 = 50

        // Byte 0: [Res(5) | Omega_H(3)]
        // Res=10 (01010). Omega=0x3E8 (011 11101000). Omega_H = 011 (3).
        // Byte 0 = 01010 011 = 0x53 (83).

        // Byte 1: Omega_L(8) = 11101000 = 0xE8 (232).

        // Byte 2: Psi_H(8). Psi=0x7D0. 01111101 0000.
        // Psi_H = 01111101 = 0x7D (125).

        // Byte 3: [Psi_L(4) | Phi_H(4)]
        // Psi_L = 0000 (0).
        // Phi=0xBB8 (1011 10111000). Phi_H = 1011 (11/0xB).
        // Byte 3 = 0000 1011 = 0x0B (11).

        // Byte 4: Phi_L(8) = 10111000 = 0xB8 (184).

        // Byte 5: 100
        // Byte 6: 200
        // Byte 7: 50

        let bytes: [u8; 8] = [83, 232, 125, 11, 184, 100, 200, 50];
        let bc = BackboneChain::from_bytes(&bytes);

        assert_eq!(bc.residue, 10);
        assert_eq!(bc.omega, 1000);
        assert_eq!(bc.psi, 2000);
        assert_eq!(bc.phi, 3000);
        assert_eq!(bc.ca_c_n_angle, 100);
        assert_eq!(bc.c_n_ca_angle, 200);
        assert_eq!(bc.n_ca_c_angle, 50);
    }
}
