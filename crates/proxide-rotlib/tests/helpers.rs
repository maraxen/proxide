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

/// Load the small PDB fixture for integration tests.
/// Uses PDB_PATH env var if set, else mosaist's `small.pdb`.
pub fn real_pdb_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("PDB_PATH") {
        std::path::PathBuf::from(p)
    } else {
        std::path::PathBuf::from("/home/marielle/repos/mosaist/testfiles/small.pdb")
    }
}

/// A residue with complete backbone coordinates parsed from a PDB ATOM record.
#[derive(Clone, Debug)]
pub struct BackboneResidue {
    pub chain:   char,
    pub res_seq: i32,
    pub aa:      String,
    pub n:       [f64; 3],
    pub ca:      [f64; 3],
    pub c:       [f64; 3],
}

/// Parse a PDB file and return all residues with complete N/CA/C backbone, in
/// (chain, res_seq) order.
pub fn parse_pdb_backbone(path: &std::path::Path) -> Vec<BackboneResidue> {
    use std::collections::BTreeMap;
    use std::io::{BufRead, BufReader};

    struct Partial {
        aa: String,
        n:  Option<[f64; 3]>,
        ca: Option<[f64; 3]>,
        c:  Option<[f64; 3]>,
    }

    let file = std::fs::File::open(path).expect("open PDB");
    let mut map: BTreeMap<(char, i32), Partial> = BTreeMap::new();

    for line in BufReader::new(file).lines().flatten() {
        if !line.starts_with("ATOM") || line.len() < 54 { continue; }
        let atom = line[12..16].trim();
        if !matches!(atom, "N" | "CA" | "C") { continue; }
        let aa      = line[17..20].trim().to_string();
        let chain   = line.chars().nth(21).unwrap_or(' ');
        let res_seq: i32 = line[22..26].trim().parse().unwrap_or(0);
        let xyz = [
            line[30..38].trim().parse::<f64>().unwrap_or(f64::NAN),
            line[38..46].trim().parse::<f64>().unwrap_or(f64::NAN),
            line[46..54].trim().parse::<f64>().unwrap_or(f64::NAN),
        ];
        let e = map.entry((chain, res_seq)).or_insert(Partial { aa, n: None, ca: None, c: None });
        match atom { "N" => e.n = Some(xyz), "CA" => e.ca = Some(xyz), _ => e.c = Some(xyz) }
    }

    map.into_iter()
        .filter_map(|((chain, res_seq), r)| {
            Some(BackboneResidue { chain, res_seq, aa: r.aa, n: r.n?, ca: r.ca?, c: r.c? })
        })
        .collect()
}
