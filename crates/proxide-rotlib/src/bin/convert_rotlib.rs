/// Convert rotamer library to proxide rotlib protobuf format.
///
/// Reads a rotamer source (e.g., Dunbrack text file), groups rotamers by (residue, phi, psi),
/// builds sidechain coordinates using template geometry and NeRF, and serializes to
/// compressed protobuf.
use clap::Parser;
use proxide_rotlib::build_library;
use proxide_rotlib::geometry::{ccd_parser::parse_ccd_ic_table, rtf_parser::parse_rtf_ic_table};
use proxide_rotlib::pb::proxide::rotlib::v1::ResidueGeometryTable;
use proxide_rotlib::rotlib_source::{DunbrackSource, RotlibSource};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "convert_rotlib")]
#[command(about = "Convert rotamer library to proxide protobuf format")]
struct Args {
    /// Path to input rotamer library file (legacy)
    #[arg(
        long,
        default_value = "data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib"
    )]
    input: PathBuf,

    /// Path to output protobuf file
    #[arg(long, default_value = "data/rotlibs/proxide-rotlib-bbdep2010.pb.zst")]
    output: PathBuf,

    /// Rotamer library source: "dunbrack:<path>" for Dunbrack BBDEP text file.
    /// Default: uses --input path if present, for backwards compatibility.
    #[arg(long)]
    rotlib_source: Option<String>,

    /// IC geometry source: "rtf:<path>" for CHARMM36 RTF, "ccd:<dir>" for PDB CCD directory.
    /// If absent, Engh-Huber placeholder values in template.rs are used.
    #[arg(long)]
    ic_source: Option<String>,

    /// Add a synthetic ALA entry (backlog #5244): the Dunbrack BBDEP format never
    /// contains ALA (or GLY), so without this flag the output library has no ALA
    /// entry at all. See `proxide_rotlib::add_synthetic_ala` for what "synthetic"
    /// means here (1 bin, 1 rotamer, p=1, num_chi=0) and why it is opt-in rather
    /// than the unconditional default.
    #[arg(long, default_value_t = false)]
    synthesize_ala: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load rotamer source
    let rotlib_source: Box<dyn RotlibSource> = match &args.rotlib_source {
        Some(s) if s.starts_with("dunbrack:") => {
            let path_str = &s["dunbrack:".len()..];
            let path = std::path::Path::new(path_str);
            info!("Loading Dunbrack rotamer source from {}...", path.display());
            let source = DunbrackSource::from_file(path)
                .map_err(|e| format!("Failed to load Dunbrack source: {}", e))?;
            info!(
                "Loaded Dunbrack source: {} residues (license={})",
                source.residue_codes().len(),
                source.data_license()
            );
            Box::new(source)
        }
        Some(other) => {
            return Err(format!(
                "Unknown --rotlib-source format '{}'. Use 'dunbrack:<path>'.",
                other
            )
            .into());
        }
        None => {
            // Backwards compatibility: default to --input if present
            warn!("No --rotlib-source specified; defaulting to --input (backwards compat)");
            warn!("DEPRECATED: please use --rotlib-source dunbrack:<path> in the future");
            let source = DunbrackSource::from_file(&args.input)
                .map_err(|e| format!("Failed to load from --input: {}", e))?;
            info!(
                "Loaded Dunbrack source: {} residues (license={})",
                source.residue_codes().len(),
                source.data_license()
            );
            Box::new(source)
        }
    };

    // Load IC geometry table (optional)
    let ic_table: Option<ResidueGeometryTable> = match &args.ic_source {
        Some(s) if s.starts_with("rtf:") => {
            let path = &s["rtf:".len()..];
            info!("Loading CHARMM36 RTF IC table from {}...", path);
            let table = parse_rtf_ic_table(path)
                .map_err(|e| format!("Failed to parse RTF IC table: {}", e))?;
            info!(
                "Loaded RTF IC table: {} residues (source={}, license={})",
                table.residues.len(),
                table.source,
                table.license
            );
            Some(table)
        }
        Some(s) if s.starts_with("ccd:") => {
            let dir = &s["ccd:".len()..];
            info!("Loading PDB CCD IC table from {}...", dir);
            let table = parse_ccd_ic_table(dir)
                .map_err(|e| format!("Failed to parse CCD IC table: {}", e))?;
            info!(
                "Loaded CCD IC table: {} residues (source={}, license={})",
                table.residues.len(),
                table.source,
                table.license
            );
            Some(table)
        }
        Some(other) => {
            return Err(format!(
                "Unknown --ic-source format '{}'. Use 'rtf:<path>' or 'ccd:<dir>'.",
                other
            )
            .into());
        }
        None => {
            info!("No --ic-source specified; using Engh-Huber template defaults.");
            None
        }
    };

    if args.synthesize_ala {
        info!("--synthesize-ala set: a synthetic ALA entry will be added (backlog #5244).");
    }

    // Build the protobuf
    let lib = build_library(rotlib_source.as_ref(), ic_table.as_ref(), args.synthesize_ala)?;
    info!("Built library with {} residue types", lib.residues.len());

    // Serialize and compress
    info!(
        "Serializing and compressing to {}...",
        args.output.display()
    );
    let encoded = prost::Message::encode_to_vec(&lib);
    let compressed = zstd::encode_all(&encoded[..], 19)?;

    // Write output
    let mut out_file = File::create(&args.output)?;
    out_file.write_all(&compressed)?;
    info!(
        "Success! Wrote {} bytes (compressed from {} bytes)",
        compressed.len(),
        encoded.len()
    );

    Ok(())
}
