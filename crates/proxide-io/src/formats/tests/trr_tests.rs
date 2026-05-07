use crate::formats::trr::TrrReader;
use std::io::Cursor;

fn create_mock_header(magic: i32, natoms: i32) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&magic.to_be_bytes()); // magic
    data.extend_from_slice(&4i32.to_be_bytes()); // string len
    data.extend_from_slice(b"v1.0"); // version string
    data.extend_from_slice(&0i32.to_be_bytes()); // ir_size
    data.extend_from_slice(&0i32.to_be_bytes()); // e_size
    data.extend_from_slice(&0i32.to_be_bytes()); // box_size
    data.extend_from_slice(&0i32.to_be_bytes()); // vir_size
    data.extend_from_slice(&0i32.to_be_bytes()); // pres_size
    data.extend_from_slice(&0i32.to_be_bytes()); // top_size
    data.extend_from_slice(&0i32.to_be_bytes()); // sym_size
    data.extend_from_slice(&0i32.to_be_bytes()); // x_size
    data.extend_from_slice(&0i32.to_be_bytes()); // v_size
    data.extend_from_slice(&0i32.to_be_bytes()); // f_size
    data.extend_from_slice(&natoms.to_be_bytes()); // natoms
    data.extend_from_slice(&0i32.to_be_bytes()); // step
    data.extend_from_slice(&0i32.to_be_bytes()); // nre
    data.extend_from_slice(&1.0f32.to_be_bytes()); // t
    data.extend_from_slice(&0.0f32.to_be_bytes()); // lambda
    data
}

fn create_mock_full_frame(natoms: i32) -> Vec<u8> {
    let mut data = Vec::new();
    let x_size: i32 = natoms * 3 * 4;
    let v_size: i32 = natoms * 3 * 4;
    let f_size: i32 = natoms * 3 * 4;
    let box_size: i32 = 9 * 4;

    data.extend_from_slice(&1993i32.to_be_bytes()); // magic
    data.extend_from_slice(&4i32.to_be_bytes()); // string len
    data.extend_from_slice(b"v1.0"); // version string
    data.extend_from_slice(&0i32.to_be_bytes()); // ir_size
    data.extend_from_slice(&0i32.to_be_bytes()); // e_size
    data.extend_from_slice(&box_size.to_be_bytes()); // box_size
    data.extend_from_slice(&0i32.to_be_bytes()); // vir_size
    data.extend_from_slice(&0i32.to_be_bytes()); // pres_size
    data.extend_from_slice(&0i32.to_be_bytes()); // top_size
    data.extend_from_slice(&0i32.to_be_bytes()); // sym_size
    data.extend_from_slice(&x_size.to_be_bytes()); // x_size
    data.extend_from_slice(&v_size.to_be_bytes()); // v_size
    data.extend_from_slice(&f_size.to_be_bytes()); // f_size
    data.extend_from_slice(&natoms.to_be_bytes()); // natoms
    data.extend_from_slice(&123i32.to_be_bytes()); // step
    data.extend_from_slice(&0i32.to_be_bytes()); // nre
    data.extend_from_slice(&5.0f32.to_be_bytes()); // t
    data.extend_from_slice(&0.0f32.to_be_bytes()); // lambda

    // Box: 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 (nm) -> 10.0, ... (A)
    for i in 0..9 {
        let val = if i % 4 == 0 { 1.0f32 } else { 0.0f32 };
        data.extend_from_slice(&val.to_be_bytes());
    }

    // Coordinates: all 1.0 nm -> 10.0 A
    for _ in 0..(natoms * 3) {
        data.extend_from_slice(&1.0f32.to_be_bytes());
    }

    // Velocities: all 2.0 nm/ps -> 2.0 nm/ps (no conversion)
    for _ in 0..(natoms * 3) {
        data.extend_from_slice(&2.0f32.to_be_bytes());
    }

    // Forces: all 3.0 kJ/mol/nm -> 3.0 kJ/mol/nm (no conversion)
    for _ in 0..(natoms * 3) {
        data.extend_from_slice(&3.0f32.to_be_bytes());
    }

    data
}

#[test]
fn test_read_frame_full() {
    let natoms = 2;
    let data = create_mock_full_frame(natoms);
    let mut reader = TrrReader::new(Cursor::new(data));
    let frame = reader.read_frame().unwrap().unwrap();

    assert_eq!(frame.time, 5.0);
    
    // Box
    let box_vecs = frame.box_vectors.unwrap();
    assert_eq!(box_vecs[0][0], 10.0);
    assert_eq!(box_vecs[1][1], 10.0);
    assert_eq!(box_vecs[2][2], 10.0);
    assert_eq!(box_vecs[0][1], 0.0);

    // Coordinates
    let coords = frame.coordinates.unwrap();
    assert_eq!(coords.len(), 6);
    for &c in &coords {
        assert_eq!(c, 10.0);
    }

    // Velocities
    let vels = frame.velocities.unwrap();
    assert_eq!(vels.len(), 6);
    for &v in &vels {
        assert_eq!(v, 2.0);
    }

    // Forces
    let frcs = frame.forces.unwrap();
    assert_eq!(frcs.len(), 6);
    for &f in &frcs {
        assert_eq!(f, 3.0);
    }
}

#[test]
fn test_read_double_precision_frame() {
    let natoms: i32 = 1;
    let mut data = Vec::new();
    let x_size: i32 = natoms * 3 * 8; // double precision
    let box_size: i32 = 9 * 8; // double precision

    data.extend_from_slice(&1993i32.to_be_bytes()); // magic
    data.extend_from_slice(&4i32.to_be_bytes()); // string len
    data.extend_from_slice(b"v1.0"); // version string
    data.extend_from_slice(&0i32.to_be_bytes()); // ir_size
    data.extend_from_slice(&0i32.to_be_bytes()); // e_size
    data.extend_from_slice(&box_size.to_be_bytes()); // box_size
    data.extend_from_slice(&0i32.to_be_bytes()); // vir_size
    data.extend_from_slice(&0i32.to_be_bytes()); // pres_size
    data.extend_from_slice(&0i32.to_be_bytes()); // top_size
    data.extend_from_slice(&0i32.to_be_bytes()); // sym_size
    data.extend_from_slice(&x_size.to_be_bytes()); // x_size
    data.extend_from_slice(&0i32.to_be_bytes()); // v_size
    data.extend_from_slice(&0i32.to_be_bytes()); // f_size
    data.extend_from_slice(&natoms.to_be_bytes()); // natoms
    data.extend_from_slice(&1i32.to_be_bytes()); // step
    data.extend_from_slice(&0i32.to_be_bytes()); // nre
    data.extend_from_slice(&10.0f64.to_be_bytes()); // t (double)
    data.extend_from_slice(&0.0f64.to_be_bytes()); // lambda (double)

    // Box (double)
    for i in 0..9 {
        let val = if i % 4 == 0 { 1.0f64 } else { 0.0f64 };
        data.extend_from_slice(&val.to_be_bytes());
    }

    // Coordinates (double)
    for _ in 0..3 {
        data.extend_from_slice(&1.23456789f64.to_be_bytes());
    }

    let mut reader = TrrReader::new(Cursor::new(data));
    let frame = reader.read_frame().unwrap().unwrap();

    assert_eq!(frame.time, 10.0);
    assert_eq!(frame.box_vectors.unwrap()[0][0], 10.0);
    // 1.23456789 nm -> 12.3456789 A (truncated to f32)
    assert!((frame.coordinates.unwrap()[0] - 12.345679).abs() < 1e-5);
}

#[test]
fn test_trr_truncation() {
    let natoms = 2;
    let data = create_mock_full_frame(natoms);
    
    // Truncate in the middle of the header
    let mut reader = TrrReader::new(Cursor::new(&data[..20]));
    assert!(reader.read_next_header().is_err());

    // Truncate in the middle of coordinates
    // Let's find exactly where coordinates start
    // magic(4) + len(4) + "v1.0"(4) + 13 ints(52) + t(4) + lambda(4) = 72 bytes.
    // Box starts at 72. Box is 36 bytes. Box ends at 108.
    // Coords start at 108.
    let mut reader = TrrReader::new(Cursor::new(&data[..120]));
    assert!(reader.read_frame().is_err());
}

#[test]
fn test_read_header_magic_valid() {
    let data = create_mock_header(1993, 10);
    let mut reader = TrrReader::new(Cursor::new(data));
    let header = reader.read_next_header().unwrap().unwrap();
    assert_eq!(header.natoms, 10);
}

#[test]
fn test_read_header_magic_invalid() {
    let data = create_mock_header(1994, 10);
    let mut reader = TrrReader::new(Cursor::new(data));
    assert!(reader.read_next_header().is_err());
}
