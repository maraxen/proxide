mod common;

use common::load_real_backbone;
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
    // A,2 → A,7: 0.000000 omitted — proxide uses cd > cut (strict), Mosaist uses >=
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
    // B,3 → B,5: 0.000000 omitted — proxide uses cd > cut (strict), Mosaist uses >=
    ("B", 3, "B", 6, 0.017732),
    ("B", 3, "B", 7, 0.011423),
    ("B", 4, "B", 5, 0.000545),
    // B,4 → B,6: 0.000000 omitted — proxide uses cd > cut (strict), Mosaist uses >=
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

const TOLERANCE: f64 = 5e-4;

fn load_or_skip() -> Option<(Arc<proxide_rotlib::RotamerLibrary>, Arc<proxide_confind::ProteinBackbone>)> {
    let rotlib_path = common::real_rotlib_path();
    let rlib = proxide_rotlib::RotamerLibrary::load(&rotlib_path).ok().map(Arc::new)?;
    let bb = load_real_backbone()?;
    Some((rlib, bb))
}

fn all_res(cf: &ConFind) -> Vec<proxide_confind::ResidueIndex> {
    (0..cf.n_residues() as u32).map(proxide_confind::ResidueIndex).collect()
}

#[test]
fn test_contacts_parity_small_pdb() {
    let (rlib, bb) = match load_or_skip() { Some(v) => v, None => return };

    let cf = ConFind::new(rlib, bb.clone(), false);
    let contact_list = cf.contacts(&all_res(&cf), 0.0).expect("contacts should succeed");

    // Build expected map: canonical key (chain_a, res_a, chain_b, res_b) → cd
    let expected: HashMap<(String, i32, String, i32), f64> = REF_CONTACTS.iter()
        .map(|&(ca, ra, cb, rb, cd)| ((ca.to_string(), ra, cb.to_string(), rb), cd))
        .collect();

    // Build actual pair set for symmetric lookup
    let actual_pairs: std::collections::HashSet<(String, i32, String, i32)> = contact_list.pairs.iter()
        .map(|&(ri_a, ri_b)| {
            let id_a = cf.residue_id(ri_a);
            let id_b = cf.residue_id(ri_b);
            (id_a.chain_id.clone(), id_a.res_id, id_b.chain_id.clone(), id_b.res_id)
        })
        .collect();

    // Report missing pairs before asserting count
    for &(ca, ra, cb, rb, _) in REF_CONTACTS {
        let key = (ca.to_string(), ra, cb.to_string(), rb);
        if !actual_pairs.contains(&key) {
            eprintln!("MISSING contact: {},{} → {},{}", ca, ra, cb, rb);
        }
    }

    assert_eq!(
        contact_list.pairs.len(), REF_CONTACTS.len(),
        "contact pair count: got {}, expected {}",
        contact_list.pairs.len(), REF_CONTACTS.len()
    );

    for (&(ri_a, ri_b), &actual) in contact_list.pairs.iter().zip(&contact_list.degrees) {
        let id_a = cf.residue_id(ri_a);
        let id_b = cf.residue_id(ri_b);
        let key = (id_a.chain_id.clone(), id_a.res_id, id_b.chain_id.clone(), id_b.res_id);
        // Also report unexpected pairs before asserting
        if !expected.contains_key(&key) {
            eprintln!("UNEXPECTED contact: {},{} → {},{} cd={:.6}", id_a.chain_id, id_a.res_id, id_b.chain_id, id_b.res_id, actual);
            continue;
        }
        let &reference = expected.get(&key).unwrap();
        assert!(
            (actual - reference).abs() < TOLERANCE,
            "contact {},{} → {},{}: got {:.6}, expected {:.6} (diff {:.2e})",
            id_a.chain_id, id_a.res_id, id_b.chain_id, id_b.res_id,
            actual, reference, (actual - reference).abs()
        );
    }
}

#[test]
fn test_crowdedness_parity_small_pdb() {
    let (rlib, bb) = match load_or_skip() { Some(v) => v, None => return };

    let cf = ConFind::new(rlib, bb.clone(), false);
    for ri in all_res(&cf) {
        cf.cache_residue(ri).expect("cache_residue should succeed");
    }

    let expected: HashMap<(String, i32), f64> = REF_CROWDEDNESS.iter()
        .map(|&(chain, res, val)| ((chain.to_string(), res), val))
        .collect();

    assert_eq!(cf.n_residues(), REF_CROWDEDNESS.len(), "residue count mismatch");

    for ri in all_res(&cf) {
        let id = cf.residue_id(ri);
        let actual = cf.crowdedness(ri).expect("crowdedness should succeed after cache");
        let key = (id.chain_id.clone(), id.res_id);
        let &reference = expected.get(&key).unwrap_or_else(|| {
            panic!("unexpected residue {},{}", id.chain_id, id.res_id)
        });
        assert!(
            (actual - reference).abs() < TOLERANCE,
            "crowdedness {},{}: got {:.6}, expected {:.6} (diff {:.2e})",
            id.chain_id, id.res_id, actual, reference, (actual - reference).abs()
        );
    }
}

#[test]
fn test_freedom_parity_small_pdb() {
    let (rlib, bb) = match load_or_skip() { Some(v) => v, None => return };

    let cf = ConFind::new(rlib, bb.clone(), false);
    cf.contacts(&all_res(&cf), 0.0).expect("contacts should succeed");

    let expected: HashMap<(String, i32), f64> = REF_FREEDOM.iter()
        .map(|&(chain, res, val)| ((chain.to_string(), res), val))
        .collect();

    assert_eq!(cf.n_residues(), REF_FREEDOM.len(), "residue count mismatch");

    for ri in all_res(&cf) {
        let id = cf.residue_id(ri);
        let actual = cf.freedom(ri).expect("freedom should succeed after contacts");
        let key = (id.chain_id.clone(), id.res_id);
        let &reference = expected.get(&key).unwrap_or_else(|| {
            panic!("unexpected residue {},{}", id.chain_id, id.res_id)
        });
        assert!(
            (actual - reference).abs() < TOLERANCE,
            "freedom {},{}: got {:.6}, expected {:.6} (diff {:.2e})",
            id.chain_id, id.res_id, actual, reference, (actual - reference).abs()
        );
    }
}

#[test]
fn test_interference_parity_small_pdb() {
    let (rlib, bb) = match load_or_skip() { Some(v) => v, None => return };

    let cf = ConFind::new(rlib, bb.clone(), false);
    // contacts() populates the rotamer cache required by interference()
    cf.contacts(&all_res(&cf), 0.0).expect("contacts should succeed");
    let interference_list = cf.interference(&all_res(&cf), 0.0).expect("interference should succeed");

    // Interference is directional: (A→B) ≠ (B→A). Key is ordered exactly as returned.
    let expected: HashMap<(String, i32, String, i32), f64> = REF_INTERFERENCE.iter()
        .map(|&(ca, ra, cb, rb, deg)| ((ca.to_string(), ra, cb.to_string(), rb), deg))
        .collect();

    assert_eq!(
        interference_list.pairs.len(), REF_INTERFERENCE.len(),
        "interference pair count: got {}, expected {}",
        interference_list.pairs.len(), REF_INTERFERENCE.len()
    );

    for (&(ri_a, ri_b), &actual) in interference_list.pairs.iter().zip(&interference_list.degrees) {
        let id_a = cf.residue_id(ri_a);
        let id_b = cf.residue_id(ri_b);
        let key = (id_a.chain_id.clone(), id_a.res_id, id_b.chain_id.clone(), id_b.res_id);
        let &reference = expected.get(&key).unwrap_or_else(|| {
            panic!("unexpected interference pair {},{} → {},{}", id_a.chain_id, id_a.res_id, id_b.chain_id, id_b.res_id)
        });
        assert!(
            (actual - reference).abs() < TOLERANCE,
            "interference {},{} → {},{}: got {:.6}, expected {:.6} (diff {:.2e})",
            id_a.chain_id, id_a.res_id, id_b.chain_id, id_b.res_id,
            actual, reference, (actual - reference).abs()
        );
    }
}
