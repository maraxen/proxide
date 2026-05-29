use std::io::Write;
use proxide_rotlib::RotlibError;

#[derive(Clone, Debug)]
pub struct BinSpec {
    pub phi: f32,
    pub psi: f32,
    pub freq: f32,
}

#[derive(Clone, Debug)]
pub struct RotSpec {
    pub prob: f32,
    pub coords: Vec<[f32; 3]>,
}

/// Write a minimal single-AA binary rotamer library for testing.
/// Defaults to na=1, atom_names=["CB"], nc=0 (no chi angles).
pub fn write_minimal_lib(
    aa: &str,
    atom_names: &[&str],
    bins: &[BinSpec],
    rotamers_per_bin: &[RotSpec],
) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();

    let na = atom_names.len() as i32;
    let nc = 0i32; // no chi angles
    let nb = bins.len() as i32;
    let nr = rotamers_per_bin.len() as i32;

    // AA name
    f.write_all(aa.as_bytes()).unwrap();
    f.write_all(&[0u8]).unwrap();

    // nc, na, nb
    f.write_all(&nc.to_le_bytes()).unwrap();
    f.write_all(&na.to_le_bytes()).unwrap();
    f.write_all(&nb.to_le_bytes()).unwrap();

    // sidechain atom names
    for name in atom_names {
        f.write_all(name.as_bytes()).unwrap();
        f.write_all(&[0u8]).unwrap();
    }

    // bin descriptors
    for bin in bins {
        f.write_all(&bin.phi.to_le_bytes()).unwrap();
        f.write_all(&bin.psi.to_le_bytes()).unwrap();
        f.write_all(&bin.freq.to_le_bytes()).unwrap();
    }

    // rotamer data
    for _bin_idx in 0..nb {
        f.write_all(&nr.to_le_bytes()).unwrap();
        for rot in rotamers_per_bin {
            f.write_all(&rot.prob.to_le_bytes()).unwrap();
            // no chi values (nc=0)
            for &xyz in &rot.coords {
                for v in xyz {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
            }
        }
    }

    f.flush().unwrap();
    f
}

/// Load the real rotlib for integration tests.
/// Uses ROTLIB_PATH env var if set, else `/home/marielle/repos/mosaist/testfiles/rotlib.bin`.
pub fn real_rotlib_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("ROTLIB_PATH") {
        std::path::PathBuf::from(p)
    } else {
        std::path::PathBuf::from("/home/marielle/repos/mosaist/testfiles/rotlib.bin")
    }
}
