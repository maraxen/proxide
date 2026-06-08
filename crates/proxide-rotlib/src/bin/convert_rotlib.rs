/// Convert rotamer library to proxide rotlib protobuf format.
///
/// Reads a rotamer source (e.g., Dunbrack text file), groups rotamers by (residue, phi, psi),
/// builds sidechain coordinates using template geometry and NeRF, and serializes to
/// compressed protobuf.

use clap::Parser;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tracing::{info, warn};
use proxide_rotlib::pb::rotlib_v1;
use proxide_rotlib::pb::proxide::rotlib::v1::ResidueGeometryTable;
use proxide_rotlib::geometry::{
    standard_residue_template, proline_template, build_standard_sidechain, ProlineBuilder,
    apply_ic_table, rtf_parser::parse_rtf_ic_table, ccd_parser::parse_ccd_ic_table,
};
use proxide_rotlib::rotlib_source::{RotlibSource, DunbrackSource};

#[derive(Parser, Debug)]
#[command(name = "convert_rotlib")]
#[command(about = "Convert rotamer library to proxide protobuf format")]
struct Args {
    /// Path to input rotamer library file (legacy)
    #[arg(long, default_value = "data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib")]
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
            return Err(
                format!(
                    "Unknown --rotlib-source format '{}'. Use 'dunbrack:<path>'.",
                    other
                )
                .into(),
            );
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
            return Err(
                format!(
                    "Unknown --ic-source format '{}'. Use 'rtf:<path>' or 'ccd:<dir>'.",
                    other
                )
                .into(),
            );
        }
        None => {
            info!("No --ic-source specified; using Engh-Huber template defaults.");
            None
        }
    };

    // Build the protobuf
    let lib = build_library(rotlib_source.as_ref(), ic_table.as_ref())?;
    info!("Built library with {} residue types", lib.residues.len());

    // Serialize and compress
    info!("Serializing and compressing to {}...", args.output.display());
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


/// Build the protobuf RotamerLibrary from a RotlibSource.
fn build_library(
    source: &dyn RotlibSource,
    ic_table: Option<&ResidueGeometryTable>,
) -> Result<rotlib_v1::RotamerLibrary, Box<dyn std::error::Error>> {
    let mut residues = Vec::new();
    let mut ic_applied_count = 0;
    let mut ic_proline_skipped = 0;

    // Canonical backbone frame (CA-origin): CA at origin, matches place_rotamer's backbone_frame
    let backbone_n = [-1.458_f32, 0.0, 0.0];
    let backbone_ca = [0.0_f32, 0.0, 0.0];
    let backbone_c = [0.551_f32, 1.420, 0.0];

    for res_code in source.residue_codes().iter() {
        // Get template
        let template = if res_code == "PRO" || res_code == "TPR" || res_code == "CPR" {
            proline_template()
        } else {
            standard_residue_template(res_code)
                .ok_or_else(|| format!("Unknown residue code: {}", res_code))?
        };

        // Apply IC table geometry (RTF or CCD source) to override Engh-Huber placeholders.
        // Proline is skipped: its ring geometry is managed by ProlineBuilder's CCD ring closure.
        let is_proline = matches!(res_code.as_str(), "PRO" | "CPR" | "TPR");
        let mut template = template;
        if let Some(table) = ic_table {
            if is_proline {
                ic_proline_skipped += 1;
            } else {
                apply_ic_table(&mut template, table);
                ic_applied_count += 1;
            }
        } else if is_proline {
            ic_proline_skipped += 1;
        }

        // Determine num_chi from the template's dihedrals
        let num_chi = template.dihedrals.len() as u32;

        // Get bins for this residue from the source
        let bins = source.bins(res_code);
        if bins.is_empty() {
            warn!("No bins for residue {}", res_code);
            continue;
        }

        // Build phi and psi centers (unique values in this residue)
        let mut phi_vals: Vec<f64> = bins.iter().map(|b| b.phi).collect();
        let mut psi_vals: Vec<f64> = bins.iter().map(|b| b.psi).collect();
        phi_vals.sort_by(|a, b| a.total_cmp(b));
        phi_vals.dedup();
        psi_vals.sort_by(|a, b| a.total_cmp(b));
        psi_vals.dedup();

        // Sidechain atom names (skip N, CA, C, O which are backbone)
        let atom_names: Vec<String> = template
            .atom_names
            .iter()
            .skip(4) // Skip N, CA, C, O
            .map(|s| s.clone())
            .collect();

        // Build proto bins
        let mut proto_bins = Vec::new();
        for bin_data in bins.iter() {
            let mut proto_rotamers = Vec::new();

            for entry in bin_data.rotamers.iter() {
                // Build sidechain coordinates
                let chi_vals_arr: [f32; 4] = {
                    let mut arr = [0.0_f32; 4];
                    for (i, &v) in entry.chi_values.iter().enumerate().take(4) {
                        arr[i] = v;
                    }
                    arr
                };

                let coords = if res_code == "PRO" || res_code == "TPR" || res_code == "CPR" {
                    let builder = ProlineBuilder::new(template.clone());
                    let proline_coords = builder.build(
                        &[backbone_n, backbone_ca, backbone_c],
                        [chi_vals_arr[0], chi_vals_arr[1], chi_vals_arr[2]],
                    )?;
                    proline_coords.sidechain
                } else {
                    let coords = build_standard_sidechain(
                        &template,
                        &chi_vals_arr,
                        backbone_n,
                        backbone_ca,
                        backbone_c,
                    );
                    // Skip backbone atoms (N, CA, C, O) — keep only sidechain
                    coords.into_iter().skip(4).collect::<Vec<_>>()
                };

                // Determine which chi values to include (only up to num_chi)
                let chi_to_include = std::cmp::min(num_chi as usize, entry.chi_values.len());
                let chi_vals: Vec<rotlib_v1::ChiValue> = (0..chi_to_include)
                    .map(|i| rotlib_v1::ChiValue {
                        val: entry.chi_values[i],
                        sigma: entry.chi_sigmas[i],
                    })
                    .collect();

                // Convert coordinates to Vec3 messages
                let coord_msgs: Vec<rotlib_v1::Vec3> = coords
                    .iter()
                    .map(|c| rotlib_v1::Vec3 {
                        x: c[0],
                        y: c[1],
                        z: c[2],
                    })
                    .collect();

                proto_rotamers.push(rotlib_v1::Rotamer {
                    prob: entry.probability as f32,
                    chi: chi_vals,
                    coords: coord_msgs,
                });
            }

            // Sort rotamers by probability (descending)
            proto_rotamers.sort_by(|a, b| b.prob.total_cmp(&a.prob));

            proto_bins.push(rotlib_v1::Bin {
                phi: bin_data.phi,
                psi: bin_data.psi,
                freq: bin_data.freq,
                rotamers: proto_rotamers,
            });
        }

        // Find default bin by matching the index from the source
        let default_bin_idx = source.default_bin_index(res_code);
        let default_bin = std::cmp::min(default_bin_idx, proto_bins.len().saturating_sub(1)) as u32;

        residues.push(rotlib_v1::ResidueEntry {
            code: res_code.clone(),
            atom_names,
            num_chi,
            phi_centers: phi_vals,
            psi_centers: psi_vals,
            default_bin,
            bins: proto_bins,
        });
    }

    // Sort residues by code
    residues.sort_by(|a, b| a.code.cmp(&b.code));

    let lib = rotlib_v1::RotamerLibrary {
        version: 1,
        provenance: format!(
            "{} source; convert_rotlib {}; git {}",
            source.source_tag(),
            env!("CARGO_PKG_VERSION"),
            "unknown"
        ),
        attribution: source.attribution().to_string(),
        data_license: source.data_license().to_string(),
        geometry_mode: rotlib_v1::GeometryMode::Precomputed as i32,
        residues,
        geometry_source: ic_table.map(|t| t.source.clone()).unwrap_or_default(),
        geometry_license: ic_table.map(|t| t.license.clone()).unwrap_or_default(),
    };

    // Print coverage summary
    info!(
        "IC geometry: {} residues processed, {} had IC table applied, {} proline skipped (ring closure retains geometry)",
        source.residue_codes().len(),
        ic_applied_count,
        ic_proline_skipped
    );

    Ok(lib)
}
