mod common;

use common::{load_1dc7_backbone, real_rotlib_path};
use proxide_confind::ConFind;
use std::collections::HashMap;
use std::sync::Arc;

const TOLERANCE: f64 = 5e-4;

// Crowdedness (fraction_pruned) at GLY/PRO positions in 1DC7.pdb chain A
// Reference from Mosaist testConFind on 1DC7.pdb + rotlib.bin
const REF_CROWDEDNESS_1DC7_GLY_PRO: &[(&str, i32, f64)] = &[
    ("A", 4, 0.475472),
    ("A", 25, 0.405660),
    ("A", 27, 0.364151),
    ("A", 36, 0.028302),
    ("A", 48, 0.445283),
    ("A", 58, 0.088679),
    ("A", 59, 0.767925),
    ("A", 62, 0.998113),
    ("A", 74, 0.358491),
    ("A", 77, 0.135849),
    ("A", 97, 0.213208),
    ("A", 103, 0.084906),
    ("A", 105, 0.020755),
];

// Freedom values at GLY/PRO positions in 1DC7.pdb chain A
const REF_FREEDOM_1DC7_GLY_PRO: &[(&str, i32, f64)] = &[
    ("A", 4, 0.273504),
    ("A", 25, 0.508476),
    ("A", 27, 0.609105),
    ("A", 36, 0.660654),
    ("A", 48, 0.385906),
    ("A", 58, 0.860115),
    ("A", 59, 0.161439),
    ("A", 62, 0.001887),
    ("A", 74, 0.585102),
    ("A", 77, 0.606493),
    ("A", 97, 0.753596),
    ("A", 103, 0.745168),
    ("A", 105, 0.918216),
];

// Contact degrees for pairs involving GLY or PRO in 1DC7.pdb chain A
const REF_CONTACTS_1DC7_GLY_PRO: &[(&str, i32, &str, i32, f64)] = &[
    ("A", 1, "A", 27, 0.000008),
    ("A", 2, "A", 27, 0.000000),
    ("A", 2, "A", 48, 0.000000),
    ("A", 3, "A", 4, 0.021174),
    ("A", 3, "A", 27, 0.048303),
    ("A", 4, "A", 5, 0.000001),
    ("A", 4, "A", 6, 0.021007),
    ("A", 4, "A", 19, 0.000002),
    ("A", 4, "A", 22, 0.000000),
    ("A", 4, "A", 23, 0.005765),
    ("A", 4, "A", 26, 0.001143),
    ("A", 4, "A", 27, 0.000000),
    ("A", 4, "A", 28, 0.281679),
    ("A", 4, "A", 30, 0.000433),
    ("A", 4, "A", 49, 0.019970),
    ("A", 4, "A", 50, 0.128061),
    ("A", 4, "A", 52, 0.001691),
    ("A", 4, "A", 77, 0.000019),
    ("A", 4, "A", 79, 0.000002),
    ("A", 4, "A", 111, 0.000000),
    ("A", 4, "A", 112, 0.002157),
    ("A", 4, "A", 115, 0.353909),
    ("A", 4, "A", 116, 0.118496),
    ("A", 4, "A", 118, 0.000211),
    ("A", 4, "A", 119, 0.745743),
    ("A", 4, "A", 120, 0.017231),
    ("A", 4, "A", 122, 0.000392),
    ("A", 5, "A", 48, 0.233958),
    ("A", 6, "A", 77, 0.000004),
    ("A", 7, "A", 36, 0.000003),
    ("A", 7, "A", 48, 0.309937),
    ("A", 9, "A", 36, 0.122329),
    ("A", 9, "A", 48, 0.000152),
    ("A", 10, "A", 36, 0.000000),
    ("A", 10, "A", 105, 0.000145),
    ("A", 11, "A", 36, 0.006498),
    ("A", 11, "A", 58, 0.000001),
    ("A", 12, "A", 105, 0.002280),
    ("A", 14, "A", 105, 0.003031),
    ("A", 15, "A", 105, 0.080171),
    ("A", 17, "A", 25, 0.000007),
    ("A", 18, "A", 25, 0.000305),
    ("A", 18, "A", 105, 0.005300),
    ("A", 19, "A", 105, 0.000001),
    ("A", 20, "A", 25, 0.000000),
    ("A", 21, "A", 25, 0.301153),
    ("A", 22, "A", 25, 0.022825),
    ("A", 23, "A", 25, 0.000000),
    ("A", 24, "A", 25, 0.001853),
    ("A", 24, "A", 27, 0.000000),
    ("A", 25, "A", 26, 0.004467),
    ("A", 25, "A", 27, 0.000003),
    ("A", 25, "A", 28, 0.000000),
    ("A", 25, "A", 107, 0.000000),
    ("A", 25, "A", 108, 0.000428),
    ("A", 25, "A", 109, 0.004993),
    ("A", 25, "A", 111, 0.000000),
    ("A", 25, "A", 112, 0.000482),
    ("A", 25, "A", 113, 0.000000),
    ("A", 25, "A", 116, 0.000000),
    ("A", 26, "A", 27, 0.000006),
    ("A", 27, "A", 28, 0.000007),
    ("A", 27, "A", 29, 0.000000),
    ("A", 27, "A", 116, 0.000154),
    ("A", 27, "A", 119, 0.000000),
    ("A", 27, "A", 120, 0.000000),
    ("A", 28, "A", 77, 0.000000),
    ("A", 29, "A", 48, 0.003345),
    ("A", 31, "A", 48, 0.116557),
    ("A", 33, "A", 36, 0.000020),
    ("A", 33, "A", 48, 0.012071),
    ("A", 34, "A", 36, 0.000000),
    ("A", 35, "A", 36, 0.000142),
    ("A", 36, "A", 37, 0.009070),
    ("A", 36, "A", 39, 0.036738),
    ("A", 36, "A", 40, 0.138150),
    ("A", 36, "A", 41, 0.000000),
    ("A", 36, "A", 43, 0.000385),
    ("A", 36, "A", 51, 0.000079),
    ("A", 36, "A", 53, 0.027916),
    ("A", 36, "A", 54, 0.000036),
    ("A", 36, "A", 56, 0.007075),
    ("A", 36, "A", 57, 0.423527),
    ("A", 36, "A", 58, 0.000117),
    ("A", 36, "A", 64, 0.000005),
    ("A", 36, "A", 65, 0.372174),
    ("A", 36, "A", 66, 0.000002),
    ("A", 36, "A", 67, 0.000010),
    ("A", 36, "A", 68, 0.080620),
    ("A", 36, "A", 69, 0.006815),
    ("A", 36, "A", 71, 0.000022),
    ("A", 36, "A", 72, 0.000368),
    ("A", 36, "A", 78, 0.000003),
    ("A", 36, "A", 80, 0.000007),
    ("A", 38, "A", 48, 0.000002),
    ("A", 39, "A", 48, 0.001151),
    ("A", 42, "A", 48, 0.008976),
    ("A", 43, "A", 48, 0.084361),
    ("A", 45, "A", 48, 0.000005),
    ("A", 46, "A", 48, 0.022620),
    ("A", 47, "A", 48, 0.000459),
    ("A", 47, "A", 74, 0.000000),
    ("A", 47, "A", 77, 0.000000),
    ("A", 48, "A", 49, 0.001655),
    ("A", 48, "A", 50, 0.000000),
    ("A", 48, "A", 51, 0.041566),
    ("A", 48, "A", 53, 0.000012),
    ("A", 48, "A", 69, 0.000004),
    ("A", 48, "A", 72, 0.000003),
    ("A", 48, "A", 73, 0.012757),
    ("A", 48, "A", 75, 0.000000),
    ("A", 48, "A", 76, 0.008501),
    ("A", 48, "A", 78, 0.000203),
    ("A", 48, "A", 119, 0.000000),
    ("A", 49, "A", 77, 0.000001),
    ("A", 50, "A", 77, 0.011515),
    ("A", 51, "A", 77, 0.000000),
    ("A", 52, "A", 77, 0.000004),
    ("A", 52, "A", 105, 0.000000),
    ("A", 54, "A", 105, 0.000080),
    ("A", 55, "A", 105, 0.000000),
    ("A", 56, "A", 58, 0.000000),
    ("A", 57, "A", 58, 0.003033),
    ("A", 57, "A", 59, 0.000676),
    ("A", 57, "A", 62, 0.169225),
    ("A", 58, "A", 59, 0.000016),
    ("A", 58, "A", 60, 0.000011),
    ("A", 58, "A", 64, 0.022984),
    ("A", 58, "A", 65, 0.002603),
    ("A", 58, "A", 67, 0.000002),
    ("A", 58, "A", 68, 0.000000),
    ("A", 59, "A", 60, 0.009098),
    ("A", 59, "A", 61, 0.001559),
    ("A", 59, "A", 63, 0.308017),
    ("A", 59, "A", 64, 0.619066),
    ("A", 59, "A", 66, 0.000284),
    ("A", 59, "A", 67, 0.000026),
    ("A", 59, "A", 86, 0.000000),
    ("A", 59, "A", 88, 0.000000),
    ("A", 59, "A", 89, 0.031443),
    ("A", 59, "A", 90, 0.000108),
    ("A", 59, "A", 93, 0.161548),
    ("A", 61, "A", 62, 0.000001),
    ("A", 62, "A", 65, 0.000050),
    ("A", 66, "A", 74, 0.000003),
    ("A", 66, "A", 97, 0.000002),
    ("A", 67, "A", 74, 0.000016),
    ("A", 70, "A", 74, 0.244158),
    ("A", 70, "A", 97, 0.000003),
    ("A", 71, "A", 74, 0.000923),
    ("A", 73, "A", 74, 0.000000),
    ("A", 73, "A", 77, 0.000000),
    ("A", 74, "A", 75, 0.006321),
    ("A", 74, "A", 76, 0.000000),
    ("A", 74, "A", 78, 0.000000),
    ("A", 74, "A", 94, 0.000144),
    ("A", 74, "A", 95, 0.027996),
    ("A", 74, "A", 96, 0.024981),
    ("A", 74, "A", 98, 0.000003),
    ("A", 75, "A", 77, 0.000298),
    ("A", 76, "A", 77, 0.000236),
    ("A", 77, "A", 78, 0.000000),
    ("A", 77, "A", 79, 0.018416),
    ("A", 77, "A", 96, 0.000005),
    ("A", 77, "A", 98, 0.029896),
    ("A", 77, "A", 99, 0.000002),
    ("A", 77, "A", 100, 0.006497),
    ("A", 77, "A", 102, 0.000000),
    ("A", 77, "A", 111, 0.000000),
    ("A", 77, "A", 114, 0.006684),
    ("A", 77, "A", 115, 0.000328),
    ("A", 77, "A", 117, 0.000013),
    ("A", 77, "A", 118, 0.270296),
    ("A", 77, "A", 119, 0.000500),
    ("A", 77, "A", 121, 0.041609),
    ("A", 77, "A", 122, 0.246378),
    ("A", 77, "A", 124, 0.001021),
    ("A", 78, "A", 97, 0.000000),
    ("A", 79, "A", 103, 0.000000),
    ("A", 81, "A", 103, 0.000000),
    ("A", 81, "A", 105, 0.004308),
    ("A", 82, "A", 103, 0.049250),
    ("A", 83, "A", 103, 0.000000),
    ("A", 83, "A", 105, 0.009497),
    ("A", 84, "A", 103, 0.293916),
    ("A", 84, "A", 105, 0.000100),
    ("A", 85, "A", 103, 0.000009),
    ("A", 86, "A", 105, 0.000000),
    ("A", 87, "A", 97, 0.000145),
    ("A", 87, "A", 103, 0.306268),
    ("A", 88, "A", 103, 0.000005),
    ("A", 90, "A", 97, 0.000001),
    ("A", 90, "A", 103, 0.000065),
    ("A", 91, "A", 97, 0.020079),
    ("A", 91, "A", 103, 0.027312),
    ("A", 92, "A", 97, 0.002349),
    ("A", 94, "A", 97, 0.203848),
    ("A", 95, "A", 97, 0.000000),
    ("A", 96, "A", 97, 0.000000),
    ("A", 97, "A", 98, 0.000045),
    ("A", 97, "A", 99, 0.021159),
    ("A", 97, "A", 100, 0.000009),
    ("A", 97, "A", 101, 0.000057),
    ("A", 100, "A", 103, 0.000111),
    ("A", 101, "A", 103, 0.000882),
    ("A", 102, "A", 103, 0.000516),
    ("A", 102, "A", 105, 0.000001),
    ("A", 103, "A", 104, 0.000000),
    ("A", 103, "A", 105, 0.000000),
    ("A", 103, "A", 106, 0.020356),
    ("A", 103, "A", 107, 0.000001),
    ("A", 103, "A", 110, 0.000002),
    ("A", 103, "A", 111, 0.000000),
    ("A", 103, "A", 114, 0.000000),
    ("A", 104, "A", 105, 0.145533),
    ("A", 105, "A", 106, 0.000003),
    ("A", 105, "A", 107, 0.000008),
    ("A", 105, "A", 108, 0.001073),
    ("A", 105, "A", 111, 0.000003),
];

fn load_or_skip_1dc7() -> Option<(
    Arc<proxide_rotlib::RotamerLibrary>,
    Arc<proxide_confind::ProteinBackbone>,
)> {
    let rotlib_path = real_rotlib_path();
    let rlib = proxide_rotlib::RotamerLibrary::load(&rotlib_path)
        .ok()
        .map(Arc::new)?;
    let bb = load_1dc7_backbone()?;
    Some((rlib, bb))
}

fn all_res(cf: &ConFind) -> Vec<proxide_confind::ResidueIndex> {
    (0..cf.n_residues() as u32)
        .map(proxide_confind::ResidueIndex)
        .collect()
}

#[test]
fn test_crowdedness_parity_1dc7_gly_pro() {
    let (rlib, bb) = match load_or_skip_1dc7() {
        Some(v) => v,
        None => return,
    };

    let cf = ConFind::new(rlib, bb.clone(), false);
    for ri in all_res(&cf) {
        cf.cache_residue(ri).expect("cache_residue should succeed");
    }

    let expected: HashMap<(String, i32), f64> = REF_CROWDEDNESS_1DC7_GLY_PRO
        .iter()
        .map(|&(chain, res, val)| ((chain.to_string(), res), val))
        .collect();

    for ri in all_res(&cf) {
        let id = cf.residue_id(ri);
        let key = (id.chain_id.clone(), id.res_id);
        if let Some(&reference) = expected.get(&key) {
            let actual = cf
                .crowdedness(ri)
                .expect("crowdedness should succeed after cache");
            assert!(
                (actual - reference).abs() < TOLERANCE,
                "crowdedness GLY/PRO {},{}: got {:.6}, expected {:.6} (diff {:.2e})",
                id.chain_id,
                id.res_id,
                actual,
                reference,
                (actual - reference).abs()
            );
        }
    }
}

#[test]
fn test_freedom_parity_1dc7_gly_pro() {
    let (rlib, bb) = match load_or_skip_1dc7() {
        Some(v) => v,
        None => return,
    };

    let cf = ConFind::new(rlib, bb.clone(), false);
    cf.contacts(&all_res(&cf), 0.0)
        .expect("contacts should succeed");

    let expected: HashMap<(String, i32), f64> = REF_FREEDOM_1DC7_GLY_PRO
        .iter()
        .map(|&(chain, res, val)| ((chain.to_string(), res), val))
        .collect();

    for ri in all_res(&cf) {
        let id = cf.residue_id(ri);
        let key = (id.chain_id.clone(), id.res_id);
        if let Some(&reference) = expected.get(&key) {
            let actual = cf
                .freedom(ri)
                .expect("freedom should succeed after contacts");
            assert!(
                (actual - reference).abs() < TOLERANCE,
                "freedom GLY/PRO {},{}: got {:.6}, expected {:.6} (diff {:.2e})",
                id.chain_id,
                id.res_id,
                actual,
                reference,
                (actual - reference).abs()
            );
        }
    }
}

#[test]
fn test_contacts_parity_1dc7_gly_pro() {
    let (rlib, bb) = match load_or_skip_1dc7() {
        Some(v) => v,
        None => return,
    };

    let cf = ConFind::new(rlib, bb.clone(), false);
    let contact_list = cf
        .contacts(&all_res(&cf), 0.0)
        .expect("contacts should succeed");

    let expected: HashMap<(String, i32, String, i32), f64> = REF_CONTACTS_1DC7_GLY_PRO
        .iter()
        .map(|&(ca, ra, cb, rb, cd)| ((ca.to_string(), ra, cb.to_string(), rb), cd))
        .collect();

    // Build actual contact map for lookup
    let actual_map: HashMap<(String, i32, String, i32), f64> = contact_list
        .pairs
        .iter()
        .zip(&contact_list.degrees)
        .map(|(&(ri_a, ri_b), &cd)| {
            let id_a = cf.residue_id(ri_a);
            let id_b = cf.residue_id(ri_b);
            (
                (
                    id_a.chain_id.clone(),
                    id_a.res_id,
                    id_b.chain_id.clone(),
                    id_b.res_id,
                ),
                cd,
            )
        })
        .collect();

    // Verify all expected GLY/PRO pairs are present with correct values
    for &(ca, ra, cb, rb, expected_cd) in REF_CONTACTS_1DC7_GLY_PRO {
        let key = (ca.to_string(), ra, cb.to_string(), rb);
        let &actual_cd = actual_map
            .get(&key)
            .unwrap_or_else(|| panic!("missing contact {},{}->{},{}", ca, ra, cb, rb));
        assert!(
            (actual_cd - expected_cd).abs() < TOLERANCE,
            "contact GLY/PRO {},{}->{},{}: got {:.6}, expected {:.6} (diff {:.2e})",
            ca,
            ra,
            cb,
            rb,
            actual_cd,
            expected_cd,
            (actual_cd - expected_cd).abs()
        );
    }
}
