//! Rotamer-library construction: build a `RotamerLibrary` protobuf from a [`RotlibSource`],
//! plus the `add_synthetic_ala` pipeline step.
//!
//! Lives in the library crate (not `src/bin/convert_rotlib.rs`) so unit tests can call
//! `build_library` and `add_synthetic_ala` directly on small fixtures, without shelling out
//! to the converter binary against multi-hundred-MB real input files.

use crate::geometry::{
    apply_ic_table, build_standard_sidechain, proline_template, standard_residue_template,
    ProlineBuilder,
};
use crate::pb::proxide::rotlib::v1::ResidueGeometryTable;
use crate::pb::rotlib_v1;
use crate::rotlib_source::RotlibSource;

/// Canonical backbone frame (CA-origin), matching `place_rotamer`'s backbone_frame.
const BACKBONE_N: [f32; 3] = [-1.458, 0.0, 0.0];
const BACKBONE_CA: [f32; 3] = [0.0, 0.0, 0.0];
const BACKBONE_C: [f32; 3] = [0.551, 1.420, 0.0];

/// Build the protobuf `RotamerLibrary` from a `RotlibSource`.
///
/// If `synthesize_ala` is true, `add_synthetic_ala` is run as a named pipeline step after
/// the source's own residues are built (see that function's docs for why this step lives
/// outside `DunbrackSource`).
pub fn build_library(
    source: &dyn RotlibSource,
    ic_table: Option<&ResidueGeometryTable>,
    synthesize_ala: bool,
) -> Result<rotlib_v1::RotamerLibrary, Box<dyn std::error::Error>> {
    let mut residues = Vec::new();
    let mut ic_applied_count = 0;
    let mut ic_proline_skipped = 0;

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
            tracing::warn!("No bins for residue {}", res_code);
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
            .cloned()
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
                        &[BACKBONE_N, BACKBONE_CA, BACKBONE_C],
                        [chi_vals_arr[0], chi_vals_arr[1], chi_vals_arr[2]],
                    )?;
                    proline_coords.sidechain
                } else {
                    let coords = build_standard_sidechain(
                        &template,
                        &chi_vals_arr,
                        BACKBONE_N,
                        BACKBONE_CA,
                        BACKBONE_C,
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

    let mut provenance_suffix = String::new();
    if synthesize_ala {
        add_synthetic_ala(&mut residues, ic_table)?;
        provenance_suffix = "; ALA: synthetic, 1 rotamer, p=1, geometry=Engh-Huber placeholder (unverified provenance, see debt)".to_string();
        if let Some(table) = ic_table {
            provenance_suffix = format!(
                "; ALA: synthetic, 1 rotamer, p=1, geometry={} (see debt on CB constant)",
                table.source
            );
        }
    }

    // Sort residues by code
    residues.sort_by_key(|r| r.code.clone());

    let lib = rotlib_v1::RotamerLibrary {
        version: 1,
        provenance: format!(
            "{} source; convert_rotlib {}; git {}{}",
            source.source_tag(),
            env!("CARGO_PKG_VERSION"),
            "unknown",
            provenance_suffix,
        ),
        attribution: source.attribution().to_string(),
        data_license: source.data_license().to_string(),
        geometry_mode: rotlib_v1::GeometryMode::Precomputed as i32,
        residues,
        geometry_source: ic_table.map(|t| t.source.clone()).unwrap_or_default(),
        geometry_license: ic_table.map(|t| t.license.clone()).unwrap_or_default(),
    };

    // Print coverage summary
    tracing::info!(
        "IC geometry: {} residues processed, {} had IC table applied, {} proline skipped (ring closure retains geometry)",
        source.residue_codes().len(),
        ic_applied_count,
        ic_proline_skipped
    );

    Ok(lib)
}

/// Add a synthetic ALA entry to a built residue list.
///
/// The Dunbrack BBDEP format never contains ALA (or GLY) — it only tabulates chi-bearing
/// ("rotameric") residues — so `DunbrackSource::residue_codes()` never yields "ALA" and
/// `build_library`'s main loop never produces an ALA `ResidueEntry`. ConFind needs ALA (it
/// is one of its 18 amino acids, as in Mosaist), so this step synthesizes one.
///
/// This is intentionally a separate, named pipeline step called from `build_library` —
/// NOT logic added inside `DunbrackSource` — because a synthetic entry is a
/// library-construction concern, not sourced Dunbrack data. Folding it into
/// `DunbrackSource` would mislabel a fabricated entry as ODC-BY Dunbrack data and would
/// break any test asserting `DunbrackSource::residue_codes()` matches the BBDEP file's
/// actual contents (spec-challenger review 260923_loop_sprint23_coherence, objection #10).
///
/// The synthetic entry uses:
/// - One (phi, psi) bin, centered at (0.0, 0.0). These are **structural placeholders**:
///   `RotamerLibrary::load_pb` builds a 1x1 grid from `phi_centers`/`psi_centers`
///   (`n_phi * n_psi == 1` bin, verified in `rotlib.rs`), and `find_closest_angle` against
///   a single-element `centers` slice always returns index 0 regardless of the query angle
///   (`binning.rs`) — so no reader ever compares against these numbers; any query angle,
///   including the `9999.0` "missing backbone" sentinel, resolves to this one bin via
///   `default_bin`. See the non-`#[cfg(test)]` unit test below, which asserts this holds
///   for both an interior angle and the sentinel.
/// - `freq = 1.0` (also unread by any query path — see above).
/// - One rotamer at `prob = 1.0`.
/// - `num_chi = 0`, `chi = []` (ALA truly has no chi angles — see the template fix in
///   `geometry/template.rs`).
/// - CB built through the exact same `standard_residue_template("ALA")` +
///   `build_standard_sidechain` path used for every other non-proline residue in this
///   function, with the same `ic_table` applied (or not) as every other residue —
///   decision (b): the underlying 1.540/110.5/-119.7 CB constant is unchanged and
///   documented as "placeholder geometry of unverified provenance" (see template.rs and
///   the filed debt), not cited as Engh & Huber.
///
/// # Errors
/// Returns `Err` if `residues` already contains an ALA entry (refuses to silently overwrite
/// or duplicate a source-provided entry), or if the ALA template unexpectedly has
/// `num_chi != 0` (defensive: this should be structurally impossible after the
/// `alanine_template()` fix, but a silent regression here would reintroduce backlog
/// #5244's FATAL bug under a different name).
pub fn add_synthetic_ala(
    residues: &mut Vec<rotlib_v1::ResidueEntry>,
    ic_table: Option<&ResidueGeometryTable>,
) -> Result<(), String> {
    if residues.iter().any(|r| r.code == "ALA") {
        return Err(
            "add_synthetic_ala: residue list already contains an ALA entry; refusing to \
             overwrite or duplicate a source-provided entry"
                .to_string(),
        );
    }

    let mut template =
        standard_residue_template("ALA").ok_or_else(|| "ALA template missing".to_string())?;
    if let Some(table) = ic_table {
        apply_ic_table(&mut template, table);
    }

    let num_chi = template.dihedrals.len() as u32;
    if num_chi != 0 {
        return Err(format!(
            "add_synthetic_ala: ALA template has num_chi == {} (expected 0); refusing to \
             synthesize a chi-bearing ALA entry — this would silently reintroduce the \
             backlog #5244 FATAL bug (spurious chi dihedral on a residue with an absolute, \
             non-chi CB torsion)",
            num_chi
        ));
    }

    let coords = build_standard_sidechain(&template, &[], BACKBONE_N, BACKBONE_CA, BACKBONE_C);
    let sidechain_coords: Vec<rotlib_v1::Vec3> = coords
        .into_iter()
        .skip(4) // Skip N, CA, C, O
        .map(|c| rotlib_v1::Vec3 {
            x: c[0],
            y: c[1],
            z: c[2],
        })
        .collect();
    let atom_names: Vec<String> = template.atom_names.iter().skip(4).cloned().collect();

    let rotamer = rotlib_v1::Rotamer {
        prob: 1.0,
        chi: Vec::new(),
        coords: sidechain_coords,
    };
    let bin = rotlib_v1::Bin {
        phi: 0.0,
        psi: 0.0,
        freq: 1.0,
        rotamers: vec![rotamer],
    };

    residues.push(rotlib_v1::ResidueEntry {
        code: "ALA".to_string(),
        atom_names,
        num_chi: 0,
        phi_centers: vec![0.0],
        psi_centers: vec![0.0],
        default_bin: 0,
        bins: vec![bin],
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotlib_source::{BinData, RotamerEntry};

    /// A minimal fixture RotlibSource for testing build_library without a 83MB Dunbrack file.
    struct FixtureSource {
        codes: Vec<String>,
    }

    impl RotlibSource for FixtureSource {
        fn residue_codes(&self) -> Vec<String> {
            self.codes.clone()
        }

        fn bins(&self, code: &str) -> Vec<BinData> {
            if !self.codes.contains(&code.to_string()) {
                return Vec::new();
            }
            // One rotameric bin with one rotamer, chi1 = 60.0 (only meaningful for SER).
            vec![BinData {
                phi: -60.0,
                psi: -40.0,
                freq: 1.0,
                rotamers: vec![RotamerEntry {
                    chi_values: vec![60.0],
                    chi_sigmas: vec![5.0],
                    probability: 0.9,
                    count: 100,
                }],
            }]
        }

        fn default_bin_index(&self, _code: &str) -> usize {
            0
        }

        fn data_license(&self) -> &str {
            "ODC-BY-1.0"
        }

        fn attribution(&self) -> &str {
            "Fixture test source"
        }

        fn source_tag(&self) -> &str {
            "fixture"
        }
    }

    #[test]
    fn test_build_library_with_synthesis_ala_present() {
        let source = FixtureSource {
            codes: vec!["SER".to_string()],
        };
        let lib = build_library(&source, None, true).expect("build_library failed");
        let ala = lib
            .residues
            .iter()
            .find(|r| r.code == "ALA")
            .expect("ALA entry missing with synthesis ON");
        assert_eq!(ala.bins.len(), 1, "ALA must have exactly 1 bin");
        assert_eq!(ala.bins[0].rotamers.len(), 1, "ALA bin must have 1 rotamer");
        assert_eq!(
            ala.bins[0].rotamers[0].prob, 1.0,
            "ALA rotamer prob must be 1.0"
        );
        assert_eq!(ala.num_chi, 0, "ALA num_chi must be 0");
        assert!(
            ala.bins[0].rotamers[0].chi.is_empty(),
            "ALA chi must be empty"
        );
        assert_eq!(ala.default_bin, 0);
        assert_eq!(ala.atom_names, vec!["CB".to_string()]);

        // CB torsion correctness: same check as the geometry-level regression test, via
        // the actually-serialized coordinates this time.
        let cb = &ala.bins[0].rotamers[0].coords[0];
        let c = BACKBONE_C;
        let n = BACKBONE_N;
        let ca = BACKBONE_CA;
        let torsion =
            -proxide_geometry::geometry::angles::dihedral_angle(&c, &n, &ca, &[cb.x, cb.y, cb.z])
                .to_degrees();
        assert!(
            (torsion - (-119.7)).abs() < 0.5,
            "synthetic ALA CB torsion: expected -119.7 +-0.5, got {:.3}",
            torsion
        );
    }

    #[test]
    fn test_build_library_without_synthesis_ala_absent() {
        let source = FixtureSource {
            codes: vec!["SER".to_string()],
        };
        let lib = build_library(&source, None, false).expect("build_library failed");
        assert!(
            !lib.residues.iter().any(|r| r.code == "ALA"),
            "ALA must be absent with synthesis OFF"
        );
    }

    #[test]
    fn test_add_synthetic_ala_errors_if_ala_already_present() {
        let mut residues = vec![rotlib_v1::ResidueEntry {
            code: "ALA".to_string(),
            atom_names: vec!["CB".to_string()],
            num_chi: 0,
            phi_centers: vec![0.0],
            psi_centers: vec![0.0],
            default_bin: 0,
            bins: vec![],
        }];
        let result = add_synthetic_ala(&mut residues, None);
        assert!(
            result.is_err(),
            "add_synthetic_ala must error when ALA already exists"
        );
        assert_eq!(residues.len(), 1, "must not have added a duplicate entry");
    }

    #[test]
    fn test_find_closest_angle_bin_selection_for_synthetic_ala() {
        // The synthetic ALA bin's phi/psi=0.0 placeholders must be selected regardless of
        // query angle (interior angle and the 9999.0 "missing backbone" sentinel alike),
        // since find_closest_angle against a single-element centers slice always returns
        // index 0 (binning.rs), and load_pb's default_bin path is used for 9999.0 queries.
        let source = FixtureSource {
            codes: vec!["SER".to_string()],
        };
        let lib = build_library(&source, None, true).unwrap();
        let ala = lib.residues.iter().find(|r| r.code == "ALA").unwrap();
        assert_eq!(ala.phi_centers, vec![0.0]);
        assert_eq!(ala.psi_centers, vec![0.0]);
        assert_eq!(ala.default_bin, 0);
        assert_eq!(ala.bins.len(), 1);
    }
}
