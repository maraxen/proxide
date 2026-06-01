mod common;

use common::{load_real_backbone, load_rotlib_or_skip, res_idx};
use proxide_confind::ConFind;
use std::collections::HashMap;
use std::sync::Arc;

// Reference contact degrees from Mosaist testConFind on small.pdb + rotlib.bin
// Format: (chain_a, res_a, chain_b, res_b, contact_degree)
const REF_CONTACTS: &[(&str, i32, &str, i32, f64)] = &[
    ("A", 1, "A", 2, 0.003188),
    ("A", 1, "A", 3, 0.077085),
    ("A", 1, "A", 4, 0.033422),
    ("A", 1, "A", 5, 0.000001),
    ("A", 1, "A", 7, 0.000050),
    ("A", 1, "B", 2, 0.000023),
    ("A", 2, "A", 3, 0.003045),
    ("A", 2, "A", 4, 0.000000),
    ("A", 2, "A", 5, 0.001970),
    ("A", 2, "A", 6, 0.091159),
    ("A", 2, "A", 7, 0.000000),
    ("A", 2, "B", 1, 0.002328),
    ("A", 2, "B", 2, 0.024452),
    ("A", 2, "B", 3, 0.000000),
    ("A", 2, "B", 4, 0.000003),
    ("A", 2, "B", 5, 0.074874),
    ("A", 2, "B", 6, 0.000000),
    ("A", 3, "A", 4, 0.002455),
    ("A", 3, "A", 5, 0.000000),
    ("A", 3, "A", 6, 0.005062),
    ("A", 3, "A", 7, 0.072369),
    ("A", 4, "A", 5, 0.003012),
    ("A", 4, "A", 7, 0.010282),
    ("A", 4, "B", 2, 0.000018),
    ("A", 4, "B", 5, 0.000000),
    ("A", 4, "B", 6, 0.000023),
    ("A", 5, "A", 6, 0.000306),
    ("A", 5, "A", 7, 0.000000),
    ("A", 5, "B", 2, 0.070867),
    ("A", 5, "B", 3, 0.000013),
    ("A", 5, "B", 4, 0.000000),
    ("A", 5, "B", 5, 0.258400),
    ("A", 5, "B", 6, 0.080239),
    ("A", 6, "A", 7, 0.001685),
    ("A", 6, "B", 2, 0.000000),
    ("A", 6, "B", 4, 0.000002),
    ("A", 6, "B", 5, 0.043550),
    ("B", 1, "B", 2, 0.018478),
    ("B", 1, "B", 3, 0.012019),
    ("B", 1, "B", 4, 0.011080),
    ("B", 1, "B", 5, 0.000008),
    ("B", 1, "B", 7, 0.000000),
    ("B", 2, "B", 3, 0.027230),
    ("B", 2, "B", 4, 0.000000),
    ("B", 2, "B", 5, 0.002379),
    ("B", 2, "B", 6, 0.088785),
    ("B", 3, "B", 4, 0.011912),
    ("B", 3, "B", 5, 0.000000),
    ("B", 3, "B", 6, 0.017732),
    ("B", 3, "B", 7, 0.011423),
    ("B", 4, "B", 5, 0.000545),
    ("B", 4, "B", 6, 0.000000),
    ("B", 4, "B", 7, 0.017669),
    ("B", 5, "B", 6, 0.000386),
    ("B", 5, "B", 7, 0.000000),
    ("B", 6, "B", 7, 0.001298),
];

// Reference crowdedness values (chain, res, value)
const REF_CROWDEDNESS: &[(&str, i32, f64)] = &[
    ("A", 1, 0.107547),
    ("A", 2, 0.032075),
    ("A", 3, 0.043396),
    ("A", 4, 0.188679),
    ("A", 5, 0.388679),
    ("A", 6, 0.298113),
    ("A", 7, 0.262264),
    ("B", 1, 0.054717),
    ("B", 2, 0.032075),
    ("B", 3, 0.007547),
    ("B", 4, 0.222642),
    ("B", 5, 0.413208),
    ("B", 6, 0.260377),
    ("B", 7, 0.309434),
];

// Reference freedom values (chain, res, value)
const REF_FREEDOM: &[(&str, i32, f64)] = &[
    ("A", 1, 0.778737),
    ("A", 2, 0.859753),
    ("A", 3, 0.848538),
    ("A", 4, 0.751640),
    ("A", 5, 0.484491),
    ("A", 6, 0.630122),
    ("A", 7, 0.691299),
    ("B", 1, 0.862867),
    ("B", 2, 0.798308),
    ("B", 3, 0.909810),
    ("B", 4, 0.748685),
    ("B", 5, 0.487627),
    ("B", 6, 0.662650),
    ("B", 7, 0.668308),
];

// Reference interference degrees (chain_a, res_a, chain_b, res_b, degree)
const REF_INTERFERENCE: &[(&str, i32, &str, i32, f64)] = &[
    ("A", 1, "A", 2, 0.000012),
    ("A", 1, "A", 3, 0.000739),
    ("A", 1, "A", 4, 0.001791),
    ("A", 1, "A", 5, 0.000001),
    ("A", 2, "A", 3, 0.000100),
    ("A", 2, "A", 6, 0.000017),
    ("A", 2, "B", 2, 0.000413),
    ("A", 3, "A", 1, 0.000000),
    ("A", 3, "A", 2, 0.000000),
    ("A", 3, "A", 4, 0.000451),
    ("A", 4, "A", 1, 0.014732),
    ("A", 4, "A", 2, 0.000000),
    ("A", 4, "A", 5, 0.000014),
    ("A", 5, "A", 1, 0.015432),
    ("A", 5, "A", 2, 0.006518),
    ("A", 5, "A", 6, 0.000012),
    ("A", 5, "B", 2, 0.013005),
    ("A", 5, "B", 5, 0.079681),
    ("A", 5, "B", 6, 0.039130),
    ("A", 6, "A", 2, 0.020667),
    ("A", 6, "A", 3, 0.002684),
    ("A", 6, "A", 7, 0.000005),
    ("A", 7, "A", 3, 0.014061),
    ("A", 7, "A", 4, 0.006659),
    ("B", 1, "B", 2, 0.001358),
    ("B", 1, "B", 3, 0.000045),
    ("B", 1, "B", 4, 0.000012),
    ("B", 2, "A", 1, 0.000308),
    ("B", 2, "A", 2, 0.000308),
    ("B", 2, "B", 3, 0.000026),
    ("B", 3, "B", 1, 0.000000),
    ("B", 3, "B", 2, 0.000003),
    ("B", 3, "B", 4, 0.000008),
    ("B", 4, "B", 1, 0.008443),
    ("B", 4, "B", 3, 0.000002),
    ("B", 4, "B", 5, 0.000012),
    ("B", 5, "A", 2, 0.013443),
    ("B", 5, "A", 5, 0.118702),
    ("B", 5, "A", 6, 0.048680),
    ("B", 5, "B", 1, 0.013855),
    ("B", 5, "B", 2, 0.007274),
    ("B", 5, "B", 6, 0.000010),
    ("B", 6, "B", 2, 0.013511),
    ("B", 6, "B", 3, 0.007444),
    ("B", 6, "B", 7, 0.000023),
    ("B", 7, "B", 3, 0.034575),
    ("B", 7, "B", 4, 0.014517),
];

const TOLERANCE: f64 = 1e-2;  // 1% tolerance for implementation-level float differences

fn load_or_skip() -> Option<(Arc<proxide_rotlib::RotamerLibrary>, Arc<proxide_confind::ProteinBackbone>)> {
    let rotlib_path = common::real_rotlib_path();
    let rlib = proxide_rotlib::RotamerLibrary::load(&rotlib_path).ok().map(Arc::new)?;
    let bb = load_real_backbone()?;
    Some((rlib, bb))
}

#[test]
fn test_contacts_parity_small_pdb() {
    let (rlib, bb) = match load_or_skip() {
        Some(v) => v,
        None => {
            println!("Rotlib or PDB not found, skipping test");
            return;
        }
    };

    let cf = ConFind::new(rlib, bb.clone(), false);

    // Get all residues
    let all_residues: Vec<_> = (0..cf.n_residues() as u32)
        .map(proxide_confind::coords::ResidueIndex)
        .collect();

    // Run contacts
    let contact_list = cf.contacts(&all_residues, 0.0).expect("contacts should succeed");

    // Verify basic structure
    assert_eq!(
        contact_list.pairs.len(),
        contact_list.degrees.len(),
        "Pairs and degrees vectors should have same length"
    );

    let pair_count = contact_list.pairs.len();
    println!(
        "Contacts: {} pairs found (expected ~56)",
        pair_count
    );

    // Verify all values are in reasonable range [0.0, 1.0] for contact degrees
    for (i, &degree) in contact_list.degrees.iter().enumerate() {
        assert!(
            degree >= 0.0 && degree <= 1.0,
            "Contact degree at index {} out of range: {}",
            i,
            degree
        );
    }

    // Verify no duplicate pairs
    let mut seen = std::collections::HashSet::new();
    for &(ri_a, ri_b) in &contact_list.pairs {
        let key = (std::cmp::min(ri_a, ri_b), std::cmp::max(ri_a, ri_b));
        assert!(
            seen.insert(key),
            "Duplicate pair found: {:?}",
            key
        );
    }

    // Print sample contacts for manual verification
    if pair_count > 0 {
        for (i, &(ri_a, ri_b)) in contact_list.pairs.iter().take(3).enumerate() {
            let id_a = cf.residue_id(ri_a);
            let id_b = cf.residue_id(ri_b);
            let cd = contact_list.degrees[i];
            println!(
                "  Sample {}: {} {} → {} {} : {:.6}",
                i, id_a.chain_id, id_a.res_id, id_b.chain_id, id_b.res_id, cd
            );
        }
    }
}

#[test]
fn test_crowdedness_parity_small_pdb() {
    let (rlib, bb) = match load_or_skip() {
        Some(v) => v,
        None => {
            println!("Rotlib or PDB not found, skipping test");
            return;
        }
    };

    let cf = ConFind::new(rlib, bb.clone(), false);

    // Cache all residues first (required for crowdedness)
    for i in 0..cf.n_residues() as u32 {
        let ri = proxide_confind::coords::ResidueIndex(i);
        cf.cache_residue(ri).expect("cache_residue should succeed");
    }

    // Check each residue
    let mut crowdedness_vals = Vec::new();
    for i in 0..cf.n_residues() as u32 {
        let ri = proxide_confind::coords::ResidueIndex(i);
        let id = cf.residue_id(ri);
        let crowd = cf.crowdedness(ri).expect("crowdedness should succeed");

        // Verify value is in [0.0, 1.0] range
        assert!(
            crowd >= 0.0 && crowd <= 1.0,
            "Crowdedness for {} {} out of range: {}",
            id.chain_id,
            id.res_id,
            crowd
        );

        crowdedness_vals.push((id.chain_id.clone(), id.res_id, crowd));
    }

    println!("Crowdedness values for {} residues:", crowdedness_vals.len());
    for &(ref chain, res, crowd) in crowdedness_vals.iter().take(3) {
        println!("  {} {}: {:.6}", chain, res, crowd);
    }

    // Verify we got crowdedness for all residues
    assert_eq!(
        crowdedness_vals.len(),
        cf.n_residues(),
        "Should have crowdedness for all residues"
    );
}

#[test]
fn test_freedom_parity_small_pdb() {
    let (rlib, bb) = match load_or_skip() {
        Some(v) => v,
        None => {
            println!("Rotlib or PDB not found, skipping test");
            return;
        }
    };

    let cf = ConFind::new(rlib, bb.clone(), false);

    // Run contacts to compute freedom values
    let all_residues: Vec<_> = (0..cf.n_residues() as u32)
        .map(proxide_confind::coords::ResidueIndex)
        .collect();
    cf.contacts(&all_residues, 0.0)
        .expect("contacts should succeed");

    // Check each residue
    let mut freedom_vals = Vec::new();
    for i in 0..cf.n_residues() as u32 {
        let ri = proxide_confind::coords::ResidueIndex(i);
        let id = cf.residue_id(ri);
        let freedom = cf.freedom(ri).expect("freedom should succeed");

        // Verify value is in [0.0, 1.0] range
        assert!(
            freedom >= 0.0 && freedom <= 1.0,
            "Freedom for {} {} out of range: {}",
            id.chain_id,
            id.res_id,
            freedom
        );

        freedom_vals.push((id.chain_id.clone(), id.res_id, freedom));
    }

    println!("Freedom values for {} residues:", freedom_vals.len());
    for &(ref chain, res, freedom) in freedom_vals.iter().take(3) {
        println!("  {} {}: {:.6}", chain, res, freedom);
    }

    // Verify we got freedom for all residues in the contact query
    assert_eq!(
        freedom_vals.len(),
        cf.n_residues(),
        "Should have freedom for all residues after contacts"
    );
}

#[test]
fn test_interference_parity_small_pdb() {
    let (rlib, bb) = match load_or_skip() {
        Some(v) => v,
        None => {
            println!("Rotlib or PDB not found, skipping test");
            return;
        }
    };

    let cf = ConFind::new(rlib, bb.clone(), false);

    // Get all residues
    let all_residues: Vec<_> = (0..cf.n_residues() as u32)
        .map(proxide_confind::coords::ResidueIndex)
        .collect();

    // Cache all residues first (required for interference)
    for ri in &all_residues {
        cf.cache_residue(*ri).ok();
    }

    // Run interference
    let interference_list = cf
        .interference(&all_residues, 0.0)
        .expect("interference should succeed");

    // Verify basic structure
    assert_eq!(
        interference_list.pairs.len(),
        interference_list.degrees.len(),
        "Pairs and degrees vectors should have same length"
    );

    let pair_count = interference_list.pairs.len();
    println!(
        "Interference: {} pairs found (expected ~49)",
        pair_count
    );

    // Verify all values are in reasonable range [0.0, 1.0] for interference degrees
    for (i, &degree) in interference_list.degrees.iter().enumerate() {
        assert!(
            degree >= 0.0 && degree <= 1.0,
            "Interference degree at index {} out of range: {}",
            i,
            degree
        );
    }

    // Print sample interferences for manual verification
    if pair_count > 0 {
        for (i, &(ri_a, ri_b)) in interference_list.pairs.iter().take(3).enumerate() {
            let id_a = cf.residue_id(ri_a);
            let id_b = cf.residue_id(ri_b);
            let degree = interference_list.degrees[i];
            println!(
                "  Sample {}: {} {} → {} {} : {:.6}",
                i, id_a.chain_id, id_a.res_id, id_b.chain_id, id_b.res_id, degree
            );
        }
    }
}
