// Shared test-helper module: cargo compiles this file as its own test binary,
// so items used only by other test binaries (or by #[ignore]'d real-data tests)
// appear unused here. Suppress so `cargo check --all-targets` stays warning-free.
#![allow(dead_code, unused_imports)]

use proxide_rotlib::RotlibError;
use std::io::Write;

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

/// Load the 1DC7 PDB fixture (124-residue protein, contains GLY and PRO).
/// Uses DC7_PDB_PATH env var if set, else mosaist's `1DC7.pdb`.
pub fn real_pdb_path_1dc7() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("DC7_PDB_PATH") {
        std::path::PathBuf::from(p)
    } else {
        std::path::PathBuf::from("/home/marielle/repos/mosaist/testfiles/1DC7.pdb")
    }
}

/// Load the 2ZTA PDB fixture (GCN4 leucine zipper, 31 residues chain A, 14 unique AAs).
/// Uses PDB_2ZTA_PATH env var if set, else mosaist's `2ZTA.pdb`.
pub fn real_pdb_path_2zta() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("PDB_2ZTA_PATH") {
        std::path::PathBuf::from(p)
    } else {
        std::path::PathBuf::from("/home/marielle/repos/mosaist/testfiles/2ZTA.pdb")
    }
}

/// A residue with complete backbone coordinates parsed from a PDB ATOM record.
#[derive(Clone, Debug)]
pub struct BackboneResidue {
    pub chain: char,
    pub res_seq: i32,
    /// Insertion code (decision j, sprint 25 task 260922_autonomous-loop,
    /// track a): part of the residue key alongside `(chain, res_seq)` so two
    /// distinct residues sharing a `res_seq` but differing only by insertion
    /// code (e.g. "52" and "52A") are never silently merged.
    pub i_code: char,
    pub aa: String,
    pub n: [f64; 3],
    pub ca: [f64; 3],
    pub c: [f64; 3],
}

/// Parse a PDB file and return all residues with complete N/CA/C backbone, in
/// (chain, res_seq, i_code) order.
///
/// Sprint 25 (task 260922_autonomous-loop, track a, decision j): this
/// test-only helper previously used `.lines().flatten()` (silently dropping
/// any I/O error mid-file), `res_seq...unwrap_or(0)` (zero-filling a garbage
/// residue number), and `...unwrap_or(f64::NAN)` (a literal NaN-as-unknown
/// sentinel -- ledger A1) for x/y/z. It now panics, naming the source file,
/// line number, and field, on any of those instead -- this is a test helper,
/// not library code, so a panic (not a `Result`) is the appropriate fail-loud
/// signal; every existing caller is `#[ignore]`d (reads real files from
/// `/home/marielle/repos/mosaist`), so this file's own non-ignored tests
/// below are the only thing that actually exercises this function in CI.
pub fn parse_pdb_backbone(path: &std::path::Path) -> Vec<BackboneResidue> {
    use std::collections::BTreeMap;
    use std::io::{BufRead, BufReader};

    struct Partial {
        aa: String,
        n: Option<[f64; 3]>,
        ca: Option<[f64; 3]>,
        c: Option<[f64; 3]>,
    }

    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("parse_pdb_backbone: cannot open {}: {e}", path.display()));
    let mut map: BTreeMap<(char, i32, char), Partial> = BTreeMap::new();

    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line_no = idx + 1;
        let line = line.unwrap_or_else(|e| {
            panic!(
                "parse_pdb_backbone: {}: line {line_no}: I/O error: {e}",
                path.display()
            )
        });
        if !line.starts_with("ATOM") || line.len() < 54 {
            continue;
        }
        let atom = line[12..16].trim();
        if !matches!(atom, "N" | "CA" | "C") {
            continue;
        }
        let aa = line[17..20].trim().to_string();
        let chain = line.chars().nth(21).unwrap_or(' ');
        let res_seq_text = line[22..26].trim();
        let res_seq: i32 = res_seq_text.parse().unwrap_or_else(|_| {
            panic!(
                "parse_pdb_backbone: {}: line {line_no}: field 'res_seq': unparseable value {res_seq_text:?}",
                path.display()
            )
        });
        let i_code = line.chars().nth(26).unwrap_or(' ');
        let field = |start: usize, end: usize, name: &str| -> f64 {
            let text = line[start..end].trim();
            text.parse::<f64>().unwrap_or_else(|_| {
                panic!(
                    "parse_pdb_backbone: {}: line {line_no}: field '{name}': unparseable value {text:?}",
                    path.display()
                )
            })
        };
        let xyz = [field(30, 38, "x"), field(38, 46, "y"), field(46, 54, "z")];
        let e = map.entry((chain, res_seq, i_code)).or_insert(Partial {
            aa,
            n: None,
            ca: None,
            c: None,
        });
        match atom {
            "N" => e.n = Some(xyz),
            "CA" => e.ca = Some(xyz),
            _ => e.c = Some(xyz),
        }
    }

    map.into_iter()
        .filter_map(|((chain, res_seq, i_code), r)| {
            Some(BackboneResidue {
                chain,
                res_seq,
                i_code,
                aa: r.aa,
                n: r.n?,
                ca: r.ca?,
                c: r.c?,
            })
        })
        .collect()
}

#[cfg(test)]
mod parse_pdb_backbone_tests {
    use super::parse_pdb_backbone;
    use std::path::Path;

    fn checked_in_snippet() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/three_residue_backbone.pdb")
    }

    #[test]
    fn parses_three_residue_checked_in_snippet() {
        let residues = parse_pdb_backbone(&checked_in_snippet());
        assert_eq!(residues.len(), 3);
        assert_eq!(residues[0].chain, 'A');
        assert_eq!(residues[0].res_seq, 1);
        assert_eq!(residues[0].i_code, ' ');
        assert_eq!(residues[0].aa, "ALA");
        assert!((residues[0].n[0] - 20.154).abs() < 1e-6);
        assert_eq!(residues[1].aa, "GLY");
        assert_eq!(residues[1].res_seq, 2);
        assert_eq!(residues[2].aa, "SER");
        assert_eq!(residues[2].res_seq, 3);
    }

    fn write_temp_pdb(content: &str) -> tempfile::NamedTempFile {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    #[should_panic(expected = "field 'res_seq'")]
    fn bad_res_seq_panics_naming_field() {
        // Same column widths as the checked-in snippet's first line, with
        // the res_seq field (cols 23-26) replaced by garbage.
        let tmp = write_temp_pdb(
            "ATOM      1  N   ALA AXXXX      20.154  29.699   5.276  1.00 49.05           N  \n",
        );
        parse_pdb_backbone(tmp.path());
    }

    #[test]
    #[should_panic(expected = "field 'x'")]
    fn bad_x_panics_naming_field() {
        // Same column widths, x field (cols 31-38) replaced by garbage.
        let tmp = write_temp_pdb(
            "ATOM      1  N   ALA A   1    xx.xxx    29.699   5.276  1.00 49.05           N  \n",
        );
        parse_pdb_backbone(tmp.path());
    }
}
