/// Convert Dunbrack BBDEP2010 text library to proxide rotlib protobuf format.
///
/// Reads the text file, groups rotamers by (residue, phi, psi), builds sidechain
/// coordinates using template geometry and NeRF, and serializes to compressed protobuf.

use clap::Parser;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use proxide_rotlib::pb::rotlib_v1;
use proxide_rotlib::geometry::{
    standard_residue_template, proline_template, build_standard_sidechain, ProlineBuilder,
    charmm_ic::{load_charmm_ideals, apply_charmm_ideals, map_template_to_charmm_name},
};

#[derive(Parser, Debug)]
#[command(name = "convert_rotlib")]
#[command(about = "Convert Dunbrack BBDEP2010 rotamer library to proxide protobuf format")]
struct Args {
    /// Path to input rotamer library file
    #[arg(long, default_value = "data/rotlibs/SimpleOpt1-5/ALL.bbdep.rotamers.lib")]
    input: PathBuf,

    /// Path to output protobuf file
    #[arg(long, default_value = "data/rotlibs/proxide-rotlib-bbdep2010.pb.zst")]
    output: PathBuf,

    /// Path to CHARMM force field XML file
    #[arg(long, default_value = "src/proxide/assets/charmm/charmm36_protein.xml")]
    charmm_xml: PathBuf,

    /// IC source for sidechain geometry: "ccd" keeps CCD template values;
    /// "charmm" applies CHARMM36 force field overrides.
    #[arg(long, default_value = "ccd")]
    ic_source: String,
}

/// Parsed rotamer entry from Dunbrack text file.
#[derive(Debug, Clone)]
struct DunbrackRotamer {
    res_code: String,
    phi: f64,
    psi: f64,
    probability: f64,
    chi_values: [f32; 4],
    chi_sigmas: [f32; 4],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load CHARMM ideals
    let charmm_ideals = if args.ic_source == "charmm" {
        eprintln!("Loading CHARMM force field ideals from {}...", args.charmm_xml.display());
        Some(
            load_charmm_ideals(args.charmm_xml.to_str().unwrap())
                .map_err(|e| format!("Failed to load CHARMM ideals: {}", e))?,
        )
    } else {
        eprintln!("IC source: ccd (using CCD template geometry; CHARMM overrides disabled)");
        None
    };

    // Read and parse the input file
    eprintln!("Reading rotamer library from {}...", args.input.display());
    let rotamers = read_rotamer_library(&args.input)?;
    eprintln!("Parsed {} rotamer entries", rotamers.len());

    // Group by residue code and (phi, psi)
    let grouped = group_rotamers(rotamers);
    eprintln!("Grouped into {} residue entries", grouped.len());

    // Build the protobuf (with optional CHARMM IC application)
    let lib = build_library(&grouped, charmm_ideals.as_ref())?;
    eprintln!("Built library with {} residue types", lib.residues.len());

    // Serialize and compress
    eprintln!("Serializing and compressing to {}...", args.output.display());
    let encoded = prost::Message::encode_to_vec(&lib);
    let compressed = zstd::encode_all(&encoded[..], 19)?;

    // Write output
    let mut out_file = File::create(&args.output)?;
    out_file.write_all(&compressed)?;
    eprintln!(
        "Success! Wrote {} bytes (compressed from {} bytes)",
        compressed.len(),
        encoded.len()
    );

    Ok(())
}

/// Read the Dunbrack rotamer library text file.
fn read_rotamer_library(path: &PathBuf) -> Result<Vec<DunbrackRotamer>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut rotamers = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;

        // Skip comment lines
        if line.starts_with('#') {
            continue;
        }

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Parse the line
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 17 {
            eprintln!("Warning: Line {} has too few fields, skipping", line_num + 1);
            continue;
        }

        let res_code = parts[0].to_string();
        let phi: f64 = parts[1].parse()?;
        let psi: f64 = parts[2].parse()?;
        // parts[3]: count (skip)
        // parts[4-7]: r1..r4 (skip)
        let probability: f64 = parts[8].parse()?;
        let chi1: f32 = parts[9].parse()?;
        let chi2: f32 = parts[10].parse()?;
        let chi3: f32 = parts[11].parse()?;
        let chi4: f32 = parts[12].parse()?;
        let chi1_sig: f32 = parts[13].parse()?;
        let chi2_sig: f32 = parts[14].parse()?;
        let chi3_sig: f32 = parts[15].parse()?;
        let chi4_sig: f32 = parts[16].parse()?;

        rotamers.push(DunbrackRotamer {
            res_code,
            phi,
            psi,
            probability,
            chi_values: [chi1, chi2, chi3, chi4],
            chi_sigmas: [chi1_sig, chi2_sig, chi3_sig, chi4_sig],
        });
    }

    Ok(rotamers)
}

/// Group rotamers by (residue_code, phi, psi).
type GroupedRotamers = BTreeMap<String, BTreeMap<(i32, i32), Vec<DunbrackRotamer>>>;

fn group_rotamers(rotamers: Vec<DunbrackRotamer>) -> GroupedRotamers {
    let mut grouped: GroupedRotamers = BTreeMap::new();

    for rot in rotamers {
        // Discretize phi/psi to nearest degree for grouping
        let phi_bin = (rot.phi.round()) as i32;
        let psi_bin = (rot.psi.round()) as i32;

        grouped
            .entry(rot.res_code.clone())
            .or_insert_with(BTreeMap::new)
            .entry((phi_bin, psi_bin))
            .or_insert_with(Vec::new)
            .push(rot);
    }

    grouped
}

/// Build the protobuf RotamerLibrary from grouped rotamers, applying CHARMM ICs.
fn build_library(
    grouped: &GroupedRotamers,
    charmm_ideals: Option<&proxide_rotlib::geometry::charmm_ic::CharmmIdeals>,
) -> Result<rotlib_v1::RotamerLibrary, Box<dyn std::error::Error>> {
    let mut residues = Vec::new();
    let mut ic_overrides_count = 0;
    let mut ic_misses_count = 0;
    let mut ic_proline_skipped = 0;

    // Canonical backbone frame for coordinate building
    let backbone_n = [0.0_f32, 0.0, 0.0];
    let backbone_ca = [1.458_f32, 0.0, 0.0];
    let backbone_c = [2.009_f32, 1.420, 0.0];

    for (res_code, bins) in grouped.iter() {
        // Get template
        let mut template = if res_code == "PRO" || res_code == "TPR" || res_code == "CPR" {
            proline_template()
        } else {
            standard_residue_template(res_code)
                .ok_or_else(|| format!("Unknown residue code: {}", res_code))?
        };

        // Apply CHARMM ideals to the template for the 19 non-proline residues.
        // Proline keeps its CCD self-consistent ring: CHARMM's unstrained equilibrium ring
        // angles break the single-DOF ring closure (solved CB-CG-CD -> ~85.5°). See #820
        // research doc + proline.rs module note; CHARMM-for-proline is a scoped follow-up.
        let is_proline = matches!(res_code.as_str(), "PRO" | "CPR" | "TPR");
        if is_proline {
            ic_proline_skipped += 1;
        } else if let Some(ideals) = charmm_ideals {
            let charmm_resname = map_template_to_charmm_name(res_code);
            match apply_charmm_ideals(&mut template, ideals, charmm_resname) {
                Ok(applied) => ic_overrides_count += applied,
                Err(e) => {
                    eprintln!("Warning: could not apply CHARMM ICs to {}: {}", res_code, e);
                    ic_misses_count += 1;
                }
            }
        }

        // Determine num_chi from the template's dihedrals
        let num_chi = template.dihedrals.len() as u32;

        // Build phi and psi centers (unique values in this residue)
        let mut phi_vals: Vec<f64> = bins.keys().map(|(p, _)| *p as f64).collect();
        let mut psi_vals: Vec<f64> = bins.keys().map(|(_, ps)| *ps as f64).collect();
        phi_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        phi_vals.dedup();
        psi_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        psi_vals.dedup();

        // Sidechain atom names (skip N, CA, C, O which are backbone)
        let atom_names: Vec<String> = template
            .atom_names
            .iter()
            .skip(4) // Skip N, CA, C, O
            .map(|s| s.clone())
            .collect();

        // Build bins
        let mut proto_bins = Vec::new();
        for (&(phi_bin, psi_bin), rots) in bins.iter() {
            let mut proto_rotamers = Vec::new();

            for rot in rots {
                // Build sidechain coordinates
                let coords = if res_code == "PRO" || res_code == "TPR" || res_code == "CPR" {
                    let builder = ProlineBuilder::new(template.clone());
                    let proline_coords = builder.build(
                        &[backbone_n, backbone_ca, backbone_c],
                        [rot.chi_values[0], rot.chi_values[1], rot.chi_values[2]],
                    )?;
                    proline_coords.sidechain
                } else {
                    let coords = build_standard_sidechain(
                        &template,
                        &rot.chi_values,
                        backbone_n,
                        backbone_ca,
                        backbone_c,
                    );
                    // Skip backbone atoms (N, CA, C, O) — keep only sidechain
                    coords.into_iter().skip(4).collect::<Vec<_>>()
                };

                // Determine which chi values to include (only up to num_chi)
                let chi_to_include = std::cmp::min(num_chi as usize, rot.chi_values.len());
                let chi_vals: Vec<rotlib_v1::ChiValue> = (0..chi_to_include)
                    .map(|i| rotlib_v1::ChiValue {
                        val: rot.chi_values[i],
                        sigma: rot.chi_sigmas[i],
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
                    prob: rot.probability as f32,
                    chi: chi_vals,
                    coords: coord_msgs,
                });
            }

            // Sort rotamers by probability (descending)
            proto_rotamers.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());

            proto_bins.push(rotlib_v1::Bin {
                phi: phi_bin as f64,
                psi: psi_bin as f64,
                freq: 0.0, // Not computed; use 0 placeholder
                rotamers: proto_rotamers,
            });
        }

        // Find the default bin (bin with highest frequency)
        let default_bin = 0u32; // Placeholder

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
            "Dunbrack BBDEP2010 SimpleOpt1-5; convert_rotlib {}; git {}",
            env!("CARGO_PKG_VERSION"),
            "unknown"
        ),
        attribution: "Contains information from the 2010 Backbone-Dependent Rotamer Library \
            (http://dunbrack.fccc.edu/bbdep2010), made available under the ODC Attribution \
            License (http://dunbrack.fccc.edu/bbdep2010/license/bbdep2010_license.txt)."
            .to_string(),
        data_license: "ODC-BY-1.0".to_string(),
        geometry_mode: rotlib_v1::GeometryMode::Precomputed as i32,
        residues,
    };

    // Print coverage summary
    eprintln!(
        "CHARMM IC coverage: {} residues processed, {} IC fields overridden, {} misses, {} proline skipped (CCD ring retained)",
        grouped.len(), ic_overrides_count, ic_misses_count, ic_proline_skipped
    );

    Ok(lib)
}
