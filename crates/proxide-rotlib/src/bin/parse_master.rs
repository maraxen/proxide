/// Parse Mosaist MASTER rotlib.bin to proxide rotlib protobuf format.
///
/// Converts an existing Mosaist rotlib.bin (binary rotamer library) to the proxide
/// RotamerLibrary protobuf format in PRECOMPUTED geometry mode. This is a developer
/// tool only — output is CC-BY-NC-SA (Grigoryan lab / Mosaist) and must not be
/// committed or redistributed.
///
/// WARNING: Output carries CC-BY-NC-SA license. Do NOT commit to version control.

use clap::Parser;
use prost::Message;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use proxide_rotlib::RotamerLibrary;

#[derive(Parser, Debug)]
#[command(name = "parse_master")]
#[command(about = "Convert Mosaist MASTER rotlib.bin to proxide protobuf format")]
#[command(long_about = "Parse Mosaist MASTER binary rotamer library and convert to \
    proxide RotamerLibrary protobuf (PRECOMPUTED geometry mode). Output carries \
    CC-BY-NC-SA license (Grigoryan lab, Dartmouth). NOT FOR REDISTRIBUTION.")]
struct Args {
    /// Path to input Mosaist rotlib.bin file
    #[arg(long)]
    input: PathBuf,

    /// Path to output protobuf file (.pb.zst)
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // License warning — always to stderr before any processing
    eprintln!("WARNING: output is CC-BY-NC-SA 4.0 (Mosaist/Grigoryan lab). Do NOT commit or redistribute.");

    // Load the binary rotamer library
    eprintln!("Loading MASTER library from {}...", args.input.display());
    let master_lib = RotamerLibrary::load(&args.input)?;

    // Convert to protobuf format
    eprintln!("Converting {} residues to protobuf...", master_lib.residue_codes().len());

    let pb_lib = master_lib.to_protobuf(
        "master_precomputed".to_string(),
        "CC-BY-NC-SA-4.0".to_string(),
    );

    // Serialize to protobuf
    let encoded = pb_lib.encode_to_vec();

    // Compress with zstd
    eprintln!("Compressing with zstd...");
    let compressed = zstd::encode_all(encoded.as_slice(), 0)?;

    // Write to output file
    eprintln!("Writing {} bytes to {}...", compressed.len(), args.output.display());
    let mut output_file = File::create(&args.output)?;
    output_file.write_all(&compressed)?;

    eprintln!("Done. Output: {}", args.output.display());
    eprintln!("WARNING: output is CC-BY-NC-SA 4.0 (Mosaist/Grigoryan lab). Do NOT commit or redistribute.");

    Ok(())
}
