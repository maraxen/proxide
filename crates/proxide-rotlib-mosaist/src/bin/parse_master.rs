/// Parse Mosaist MASTER rotlib.bin to proxide rotlib protobuf format.
///
/// Converts an existing Mosaist rotlib.bin (binary rotamer library) to the proxide
/// RotamerLibrary protobuf format in PRECOMPUTED geometry mode.
///
/// ## License note
///
/// Output is CC-BY-NC-SA 4.0 (Grigoryan lab / Mosaist). You may use the output
/// for personal research. Do NOT commit the .pb.zst output to version control or
/// distribute it to others.
///
/// WARNING: Output carries CC-BY-NC-SA license. Do NOT commit to version control.
use clap::Parser;
use prost::Message;
use proxide_rotlib_mosaist::RotamerLibrary;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "parse_master")]
#[command(about = "Convert Mosaist MASTER rotlib.bin to proxide protobuf format")]
#[command(
    long_about = "Parse Mosaist MASTER binary rotamer library and convert to \
    proxide RotamerLibrary protobuf (PRECOMPUTED geometry mode). Output carries \
    CC-BY-NC-SA license (Grigoryan lab, Dartmouth). NOT FOR REDISTRIBUTION."
)]
struct Args {
    /// Path to input Mosaist rotlib.bin file
    #[arg(long)]
    input: PathBuf,

    /// Path to output protobuf file (.pb.zst)
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber for structured logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // License warning — to stderr before any processing
    warn!("output is CC-BY-NC-SA 4.0 (Mosaist/Grigoryan lab). Do NOT commit or redistribute.");

    // Load the binary rotamer library
    info!("Loading MASTER library from {}", args.input.display());
    let master_lib = RotamerLibrary::load(&args.input)?;

    // Convert to protobuf format
    info!(
        "Converting {} residues to protobuf",
        master_lib.residue_codes().len()
    );

    let pb_lib = master_lib.to_protobuf(
        "master_precomputed".to_string(),
        "CC-BY-NC-SA-4.0".to_string(),
    );

    // Serialize to protobuf
    let encoded = pb_lib.encode_to_vec();

    // Compress with zstd
    info!("Compressing with zstd");
    let compressed = zstd::encode_all(encoded.as_slice(), 0)?;

    // Write to output file
    info!(
        "Writing {} bytes to {}",
        compressed.len(),
        args.output.display()
    );
    let mut output_file = File::create(&args.output)?;
    output_file.write_all(&compressed)?;

    info!("Done. Output: {}", args.output.display());

    // License warning — to stderr after processing
    warn!("output is CC-BY-NC-SA 4.0 (Mosaist/Grigoryan lab). Do NOT commit or redistribute.");

    Ok(())
}
