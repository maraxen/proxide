//! Acceptance tests for the typed configuration (spec §3.9.6): defaults, the policy
//! combination matrix, orthogonality, scope behaviour, validation, cutoff composition and
//! determinism. Library-backed tests use the committed rotamer library and either the
//! synthetic helix generator or the vendored public-domain crambin coordinates.

mod common;

use common::{
    crambin_path, make_helix_backbone, make_synthetic_backbone, require_rotlib,
    truncate_backbone,
};
use proxide_confind::{
    load_pdb_f64, CbClashPolicy, ConFind, ConFindConfig, ConFindError, ConfigError,
    CrowdednessMode, Cutoffs, FreedomMode, ProteinBackbone, ResidueIndex, ResidueScope, AA_NAMES,
};
use proxide_rotlib::RotamerLibrary;
use std::sync::Arc;

const POLICIES: [CbClashPolicy; 2] = [CbClashPolicy::AlaOnly, CbClashPolicy::AllResidues];
const CROWD: [CrowdednessMode; 2] = [
    CrowdednessMode::ProbabilityWeighted,
    CrowdednessMode::PrunedCountFraction,
];
const FREEDOM: [FreedomMode; 2] = [FreedomMode::Rms, FreedomMode::SingleThreshold];

fn all(n: usize) -> Vec<ResidueIndex> {
    (0..n as u32).map(ResidueIndex).collect()
}

fn structures() -> Vec<(&'static str, Arc<ProteinBackbone>)> {
    let crambin = load_pdb_f64(crambin_path()).expect("vendored crambin must load");
    vec![
        ("helix", make_helix_backbone(10)),
        ("crambin[..20]", truncate_backbone(&crambin, 20)),
    ]
}

/// Everything a run reports, for bitwise comparison between configurations.
#[derive(Debug, Clone, PartialEq)]
struct Outputs {
    crowdedness: Vec<u64>,
    freedom: Vec<Option<u64>>,
    contacts: Vec<((u32, u32), u64)>,
    interference: Vec<((u32, u32), u64)>,
}

fn outputs(rotlib: &Arc<RotamerLibrary>, bb: &Arc<ProteinBackbone>, cfg: ConFindConfig) -> Outputs {
    let cf = ConFind::with_config(rotlib.clone(), bb.clone(), false, cfg).unwrap();
    let residues = all(cf.n_residues());
    cf.cache_all().unwrap();
    let contacts = cf.contacts(&residues, 0.0).unwrap();
    let interference = cf.interference(&residues, 0.0).unwrap();
    let bits = |l: &proxide_confind::ContactList| {
        l.pairs
            .iter()
            .zip(&l.degrees)
            .map(|(&(a, b), d)| ((a.0, b.0), d.to_bits()))
            .collect::<Vec<_>>()
    };
    Outputs {
        crowdedness: residues
            .iter()
            .map(|&r| cf.crowdedness(r).unwrap().to_bits())
            .collect(),
        freedom: residues
            .iter()
            .map(|&r| match cf.freedom(r) {
                Ok(f) => Some(f.to_bits()),
                Err(ConFindError::OutOfScope(_)) => None,
                Err(e) => panic!("freedom failed: {e}"),
            })
            .collect(),
        contacts: bits(&contacts),
        interference: bits(&interference),
    }
}

fn proper_subset(n: usize) -> ResidueScope {
    ResidueScope::Subset((0..(n as u32) / 2).map(ResidueIndex).collect())
}

// ---------------------------------------------------------------- 1. defaults

#[test]
fn defaults_match_specification() {
    let c = ConFindConfig::default();
    assert_eq!(c.cb_clash_policy, CbClashPolicy::AllResidues);
    assert_eq!(c.crowdedness_mode, CrowdednessMode::ProbabilityWeighted);
    assert_eq!(c.freedom_mode, FreedomMode::Rms);
    assert_eq!(c.scope, ResidueScope::All);
    assert_eq!(
        c.cutoffs,
        Cutoffs {
            d_clash: 2.0,
            d_cont: 3.0,
            d_cut: 25.0,
            tau_lo: 0.5,
            tau_hi: 2.0
        }
    );
    c.validate().unwrap();
}

#[test]
fn new_equals_with_default_config() {
    let rotlib = require_rotlib();
    let bb = make_helix_backbone(8);
    let a = ConFind::new(rotlib.clone(), bb.clone(), false).unwrap();
    assert_eq!(a.config(), &ConFindConfig::default());
    assert_eq!(
        outputs_of(&a),
        outputs(&rotlib, &bb, ConFindConfig::default())
    );
}

fn outputs_of(cf: &ConFind) -> Outputs {
    outputs(
        &cf.rotlib,
        &cf.backbone,
        cf.config().clone(),
    )
}

#[test]
fn builder_setters_set_fields() {
    let c = ConFindConfig::default()
        .with_cb_clash_policy(CbClashPolicy::AlaOnly)
        .with_crowdedness_mode(CrowdednessMode::PrunedCountFraction)
        .with_freedom_mode(FreedomMode::SingleThreshold)
        .with_scope(ResidueScope::Subset(vec![ResidueIndex(2)]))
        .with_cutoffs(Cutoffs {
            d_clash: 1.5,
            ..Cutoffs::default()
        });
    assert_eq!(c.cb_clash_policy, CbClashPolicy::AlaOnly);
    assert_eq!(c.crowdedness_mode, CrowdednessMode::PrunedCountFraction);
    assert_eq!(c.freedom_mode, FreedomMode::SingleThreshold);
    assert_eq!(c.scope, ResidueScope::Subset(vec![ResidueIndex(2)]));
    assert_eq!(c.cutoffs.d_clash, 1.5);
}

// ------------------------------------------------- 4. policy combination matrix

#[test]
fn policy_matrix_all_combinations_run_and_stay_bounded() {
    let rotlib = require_rotlib();
    for (name, bb) in structures() {
        let n = bb.bb.len();
        for policy in POLICIES {
            for crowd in CROWD {
                for freedom in FREEDOM {
                    for scope in [ResidueScope::All, proper_subset(n)] {
                        let cfg = ConFindConfig::default()
                            .with_cb_clash_policy(policy)
                            .with_crowdedness_mode(crowd)
                            .with_freedom_mode(freedom)
                            .with_scope(scope.clone());
                        let ctx = format!("{name} {policy} {crowd} {freedom} {scope:?}");
                        let cf = ConFind::with_config(rotlib.clone(), bb.clone(), false, cfg)
                            .unwrap_or_else(|e| panic!("{ctx}: construction failed: {e}"));
                        let residues = all(n);
                        cf.cache_all().unwrap();
                        let contacts = cf.contacts(&residues, 0.0).unwrap();
                        let interf = cf.interference(&residues, 0.0).unwrap();
                        for d in &contacts.degrees {
                            assert!(d.is_finite() && (0.0..=1.0).contains(d), "{ctx}: c = {d}");
                        }
                        for &r in &residues {
                            let c = cf.crowdedness(r).unwrap();
                            assert!(c.is_finite() && (0.0..=1.0).contains(&c), "{ctx}: C = {c}");
                            if cf.in_scope(r) {
                                let f = cf.freedom(r).unwrap();
                                assert!(
                                    f.is_finite() && (0.0..=1.0).contains(&f),
                                    "{ctx}: F = {f}"
                                );
                            } else {
                                assert!(matches!(cf.freedom(r), Err(ConFindError::OutOfScope(x)) if x == r));
                            }
                        }
                        // I(i,j) <= Z_i * C_w(i) for every directed pair.
                        for (&(i, _), &v) in interf.pairs.iter().zip(&interf.degrees) {
                            let rc = cf.residue_cache(i).unwrap();
                            let z = rc.omega_lib / 100.0;
                            let cw = rc.crowdedness(CrowdednessMode::ProbabilityWeighted);
                            assert!(v >= 0.0 && v <= z * cw + 1e-12, "{ctx}: I = {v}");
                        }
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------- 5. orthogonality (§3.9.4)

#[test]
fn crowdedness_mode_changes_only_crowdedness() {
    let rotlib = require_rotlib();
    for (name, bb) in structures() {
        let base = outputs(&rotlib, &bb, ConFindConfig::default());
        let alt = outputs(
            &rotlib,
            &bb,
            ConFindConfig::default().with_crowdedness_mode(CrowdednessMode::PrunedCountFraction),
        );
        assert_eq!(base.freedom, alt.freedom, "{name}");
        assert_eq!(base.contacts, alt.contacts, "{name}");
        assert_eq!(base.interference, alt.interference, "{name}");
    }
}

#[test]
fn freedom_mode_changes_only_freedom() {
    let rotlib = require_rotlib();
    for (name, bb) in structures() {
        let base = outputs(&rotlib, &bb, ConFindConfig::default());
        let alt = outputs(
            &rotlib,
            &bb,
            ConFindConfig::default().with_freedom_mode(FreedomMode::SingleThreshold),
        );
        assert_eq!(base.crowdedness, alt.crowdedness, "{name}");
        assert_eq!(base.contacts, alt.contacts, "{name}");
        assert_eq!(base.interference, alt.interference, "{name}");
        assert_ne!(base.freedom, alt.freedom, "{name}: modes should differ somewhere");
    }
}

#[test]
fn scope_changes_only_freedom() {
    let rotlib = require_rotlib();
    for (name, bb) in structures() {
        let n = bb.bb.len();
        let base = outputs(&rotlib, &bb, ConFindConfig::default());
        let alt = outputs(
            &rotlib,
            &bb,
            ConFindConfig::default().with_scope(proper_subset(n)),
        );
        assert_eq!(base.crowdedness, alt.crowdedness, "{name}");
        assert_eq!(base.contacts, alt.contacts, "{name}");
        assert_eq!(base.interference, alt.interference, "{name}");
    }
}

#[test]
fn cb_clash_policy_monotonicity() {
    let rotlib = require_rotlib();
    for (name, bb) in structures() {
        let mk = |p| {
            ConFind::with_config(
                rotlib.clone(),
                bb.clone(),
                false,
                ConFindConfig::default().with_cb_clash_policy(p),
            )
            .unwrap()
        };
        let ala = mk(CbClashPolicy::AlaOnly);
        let allr = mk(CbClashPolicy::AllResidues);
        for r in all(bb.bb.len()) {
            let (a, b) = (ala.residue_cache(r).unwrap(), allr.residue_cache(r).unwrap());
            assert_eq!(a.n_library, b.n_library);
            assert!(b.n_surviving <= a.n_surviving, "{name} {r:?}");
            for mode in CROWD {
                assert!(
                    b.crowdedness(mode) >= a.crowdedness(mode) - 1e-12,
                    "{name} {r:?} {mode}"
                );
            }
        }
    }
}

// ------------------------------------------------------ 6. freedom-mode relation

#[test]
fn single_threshold_never_exceeds_rms() {
    let rotlib = require_rotlib();
    for (name, bb) in structures() {
        let mk = |m| {
            ConFind::with_config(
                rotlib.clone(),
                bb.clone(),
                false,
                ConFindConfig::default().with_freedom_mode(m),
            )
            .unwrap()
        };
        let rms = mk(FreedomMode::Rms);
        let one = mk(FreedomMode::SingleThreshold);
        for r in all(bb.bb.len()) {
            assert!(one.freedom(r).unwrap() <= rms.freedom(r).unwrap() + 1e-12, "{name} {r:?}");
        }
    }
}

#[test]
fn without_neighbours_both_modes_equal_surviving_fraction() {
    let rotlib = require_rotlib();
    // Two residues 60 A apart: neither is within d_cut of the other.
    let bb = make_synthetic_backbone(2, 60.0);
    for m in FREEDOM {
        let cf = ConFind::with_config(
            rotlib.clone(),
            bb.clone(),
            false,
            ConFindConfig::default().with_freedom_mode(m),
        )
        .unwrap();
        for r in all(2) {
            assert!(cf.neighbors(r).is_empty());
            let rc = cf.residue_cache(r).unwrap();
            let expect = rc.n_surviving as f64 / rc.n_library as f64;
            assert!((cf.freedom(r).unwrap() - expect).abs() < 1e-12, "{m}");
        }
    }
}

#[test]
fn isolated_residue_is_uncrowded_and_free() {
    // §5.3 item 5: no spatial obstacles => C = 0 and F = 1.
    let rotlib = require_rotlib();
    let cf = ConFind::new(rotlib, make_synthetic_backbone(1, 3.8), false).unwrap();
    let r = ResidueIndex(0);
    assert_eq!(cf.crowdedness(r).unwrap(), 0.0);
    assert!((cf.freedom(r).unwrap() - 1.0).abs() < 1e-12);
}

// ------------------------------------------------------------------- 8. scope

#[test]
fn scope_equals_all_when_neighbours_inside_and_is_monotone() {
    let rotlib = require_rotlib();
    let crambin = load_pdb_f64(crambin_path()).unwrap();
    let bb = truncate_backbone(&crambin, 24);
    let mk = |scope: ResidueScope| {
        ConFind::with_config(
            rotlib.clone(),
            bb.clone(),
            false,
            ConFindConfig::default().with_scope(scope),
        )
        .unwrap()
    };
    let subset = |k: u32| ResidueScope::Subset((0..k).map(ResidueIndex).collect());
    let (all_cf, s12, s18) = (mk(ResidueScope::All), mk(subset(12)), mk(subset(18)));

    // Equality for residues whose neighbours all lie in the subset.
    let mut checked = 0;
    for r in (0..12u32).map(ResidueIndex) {
        if all_cf.neighbors(r).iter().all(|n| n.0 < 12) {
            assert_eq!(
                s12.freedom(r).unwrap().to_bits(),
                all_cf.freedom(r).unwrap().to_bits()
            );
            checked += 1;
        }
    }
    println!("scope equality checked on {checked} residues");

    // Out of scope.
    assert!(matches!(
        s12.freedom(ResidueIndex(12)),
        Err(ConFindError::OutOfScope(ResidueIndex(12)))
    ));

    // Adding residues to the scope never increases freedom.
    for r in (0..12u32).map(ResidueIndex) {
        let (f12, f18, fall) = (
            s12.freedom(r).unwrap(),
            s18.freedom(r).unwrap(),
            all_cf.freedom(r).unwrap(),
        );
        assert!(f18 <= f12 + 1e-12 && fall <= f18 + 1e-12, "{r:?}: {f12} {f18} {fall}");
    }
}

#[test]
fn scope_duplicates_are_ignored() {
    let rotlib = require_rotlib();
    let bb = make_helix_backbone(6);
    let dup = ResidueScope::Subset(vec![ResidueIndex(1), ResidueIndex(1), ResidueIndex(2)]);
    let single = ResidueScope::Subset(vec![ResidueIndex(1), ResidueIndex(2)]);
    assert_eq!(
        outputs(&rotlib, &bb, ConFindConfig::default().with_scope(dup)),
        outputs(&rotlib, &bb, ConFindConfig::default().with_scope(single))
    );
}

// ------------------------------------------------------------- 9. validation

fn cutoffs_with(field: &str, v: f64) -> Cutoffs {
    let mut c = Cutoffs::default();
    match field {
        "d_clash" => c.d_clash = v,
        "d_cont" => c.d_cont = v,
        "d_cut" => c.d_cut = v,
        "tau_lo" => c.tau_lo = v,
        "tau_hi" => c.tau_hi = v,
        _ => unreachable!(),
    }
    c
}

#[test]
fn v1_every_cutoff_must_be_finite_and_positive() {
    for field in ["d_clash", "d_cont", "d_cut", "tau_lo", "tau_hi"] {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let cfg = ConFindConfig::default()
                .with_freedom_mode(FreedomMode::SingleThreshold)
                .with_cutoffs(cutoffs_with(field, bad));
            match cfg.validate() {
                Err(ConfigError::InvalidCutoff { name, .. }) => assert_eq!(name, field),
                other => panic!("{field}={bad}: expected InvalidCutoff, got {other:?}"),
            }
        }
    }
}

#[test]
fn v2_tau_order_depends_on_freedom_mode() {
    for (lo, hi) in [(2.0, 2.0), (3.0, 2.0)] {
        let cut = Cutoffs {
            tau_lo: lo,
            tau_hi: hi,
            ..Cutoffs::default()
        };
        let rms = ConFindConfig::default().with_cutoffs(cut);
        assert_eq!(
            rms.validate(),
            Err(ConfigError::TauOrder {
                tau_lo: lo,
                tau_hi: hi
            })
        );
        let single = rms.with_freedom_mode(FreedomMode::SingleThreshold);
        single.validate().unwrap();
    }
}

#[test]
fn v3_empty_scope_subset() {
    let cfg = ConFindConfig::default().with_scope(ResidueScope::Subset(vec![]));
    assert_eq!(cfg.validate(), Err(ConfigError::EmptyScope));
    let rotlib = require_rotlib();
    let err = ConFind::with_config(rotlib, make_helix_backbone(4), false, cfg)
        .err()
        .unwrap();
    assert!(matches!(err, ConFindError::InvalidConfig(ConfigError::EmptyScope)));
}

#[test]
fn v4_unknown_variant_names_are_rejected() {
    let e = "bogus".parse::<CbClashPolicy>().unwrap_err();
    assert!(matches!(e, ConfigError::UnknownVariant { option: "cbclash", ref value, .. } if value == "bogus"));
    let e = "AllResidues".parse::<CbClashPolicy>().unwrap_err();
    assert!(matches!(e, ConfigError::UnknownVariant { .. }), "names are kebab-case only");
    let e = "".parse::<CrowdednessMode>().unwrap_err();
    assert!(matches!(e, ConfigError::UnknownVariant { option: "crowdmode", .. }));
    let e = "median".parse::<FreedomMode>().unwrap_err();
    assert!(matches!(e, ConfigError::UnknownVariant { option: "freedommode", .. }));
    assert!(e.to_string().contains("rms, single-threshold"));
}

#[test]
fn variant_names_round_trip() {
    assert_eq!("ala-only".parse::<CbClashPolicy>().unwrap(), CbClashPolicy::AlaOnly);
    assert_eq!("all-residues".parse::<CbClashPolicy>().unwrap(), CbClashPolicy::AllResidues);
    assert_eq!(
        "probability-weighted".parse::<CrowdednessMode>().unwrap(),
        CrowdednessMode::ProbabilityWeighted
    );
    assert_eq!(
        "pruned-count-fraction".parse::<CrowdednessMode>().unwrap(),
        CrowdednessMode::PrunedCountFraction
    );
    assert_eq!("rms".parse::<FreedomMode>().unwrap(), FreedomMode::Rms);
    assert_eq!(
        "single-threshold".parse::<FreedomMode>().unwrap(),
        FreedomMode::SingleThreshold
    );
}

fn required_dcut(rotlib: &RotamerLibrary, d_cont: f64) -> f64 {
    let max_reach = AA_NAMES
        .iter()
        .map(|aa| rotlib.max_sidechain_reach(aa).unwrap())
        .fold(0.0, f64::max);
    2.0 * max_reach + d_cont
}

#[test]
fn v5_dcut_must_cover_library_reach() {
    let rotlib = require_rotlib();
    let bb = make_helix_backbone(4);
    let cfg = ConFindConfig::default().with_cutoffs(Cutoffs {
        d_cut: 10.0,
        ..Cutoffs::default()
    });
    match ConFind::with_config(rotlib.clone(), bb.clone(), false, cfg)
        .err()
        .expect("d_cut = 10 must be refused")
    {
        ConFindError::InvalidConfig(ConfigError::DcutTooSmall { d_cut, required }) => {
            assert_eq!(d_cut, 10.0);
            assert!((required - required_dcut(&rotlib, 3.0)).abs() < 1e-12);
        }
        other => panic!("expected DcutTooSmall, got {other:?}"),
    }

    // A large d_cont raises the requirement: d_cut = 25 fails once 2*R_max + d_cont > 25.
    let d_cont = 25.0 - 2.0 * (required_dcut(&rotlib, 0.0) / 2.0) + 1.0;
    let cfg = ConFindConfig::default().with_cutoffs(Cutoffs {
        d_cont,
        ..Cutoffs::default()
    });
    assert!(matches!(
        ConFind::with_config(rotlib.clone(), bb.clone(), false, cfg).err(),
        Some(ConFindError::InvalidConfig(ConfigError::DcutTooSmall { .. }))
    ));

    // Exactly the required value is accepted (inclusive), and nothing is silently replaced.
    let ok = ConFindConfig::default().with_cutoffs(Cutoffs {
        d_cut: required_dcut(&rotlib, 3.0),
        ..Cutoffs::default()
    });
    let cf = ConFind::with_config(rotlib.clone(), bb, false, ok.clone()).unwrap();
    assert_eq!(cf.config(), &ok);
    assert!((cf.d_cut_min() - required_dcut(&rotlib, 3.0)).abs() < 1e-12);
}

#[test]
fn v6_scope_index_must_be_inside_structure() {
    let rotlib = require_rotlib();
    let cfg = ConFindConfig::default()
        .with_scope(ResidueScope::Subset(vec![ResidueIndex(0), ResidueIndex(7)]));
    let err = ConFind::with_config(rotlib, make_helix_backbone(4), false, cfg)
        .err()
        .unwrap();
    assert!(matches!(
        err,
        ConFindError::InvalidConfig(ConfigError::ScopeOutOfRange {
            index: 7,
            n_residues: 4
        })
    ));
}

#[test]
fn construction_validates_in_documented_order() {
    // V1 before V3: both a bad cutoff and an empty scope present -> InvalidCutoff.
    let rotlib = require_rotlib();
    let cfg = ConFindConfig::default()
        .with_scope(ResidueScope::Subset(vec![]))
        .with_cutoffs(Cutoffs {
            d_clash: -1.0,
            ..Cutoffs::default()
        });
    assert!(matches!(
        ConFind::with_config(rotlib.clone(), make_helix_backbone(3), false, cfg).err(),
        Some(ConFindError::InvalidConfig(ConfigError::InvalidCutoff { name: "d_clash", .. }))
    ));
    // V6 before V5: out-of-range scope and too-small d_cut -> ScopeOutOfRange.
    let cfg = ConFindConfig::default()
        .with_scope(ResidueScope::Subset(vec![ResidueIndex(99)]))
        .with_cutoffs(Cutoffs {
            d_cut: 5.0,
            ..Cutoffs::default()
        });
    assert!(matches!(
        ConFind::with_config(rotlib.clone(), make_helix_backbone(3), false, cfg).err(),
        Some(ConFindError::InvalidConfig(ConfigError::ScopeOutOfRange { .. }))
    ));
    // V5 before strict preconditions: a structure with missing backbone atoms and d_cut = 5.0
    // with strict = true still fails with DcutTooSmall (V5) before PreconditionsFailed.
    let mut bad_bb = (*make_helix_backbone(3)).clone();
    bad_bb.bb[0].n = None;
    let cfg = ConFindConfig::default().with_cutoffs(Cutoffs {
        d_cut: 5.0,
        ..Cutoffs::default()
    });
    assert!(matches!(
        ConFind::with_config(rotlib, Arc::new(bad_bb), true, cfg).err(),
        Some(ConFindError::InvalidConfig(ConfigError::DcutTooSmall { .. }))
    ));
}

// ------------------------------------------------------------------ R_max §3.8.5

#[test]
fn r_max_properties() {
    let rotlib = require_rotlib();
    let cf = ConFind::new(rotlib.clone(), make_helix_backbone(3), false).unwrap();
    let ala = cf.r_max("ALA").unwrap();
    assert!((1.45..=1.60).contains(&ala), "R_max(ALA) = {ala}");
    for aa in AA_NAMES {
        let r = cf.r_max(aa).unwrap();
        assert!(r >= ala - 1e-12, "{aa}: {r}");
    }
    assert!(cf.r_max("ARG").unwrap() >= 5.0);
    // Library-free ceiling along the longest bonded path (Engh & Huber lengths, spec §3.8.5).
    assert!(cf.r_max("ARG").unwrap() <= 8.7);
    assert!(cf.r_max("LYS").unwrap() <= 7.6);
    assert!(cf.r_max("GLY").is_none());
    assert!(cf.d_cut_min() <= 25.0, "shipped library must fit the default d_cut");
}

#[test]
fn contacts_beyond_required_dcut_are_zero_even_without_cutoff() {
    // Exactness: placed pairs farther apart than 2*R_max + d_cont have no contact, so the
    // default d_cut and a huge d_cut give identical (zero) results.
    let rotlib = require_rotlib();
    let probe = ConFind::new(rotlib.clone(), make_helix_backbone(3), false).unwrap();
    let spacing = probe.d_cut_min() + 0.01;
    let bb = make_synthetic_backbone(2, spacing);
    let huge = ConFindConfig::default().with_cutoffs(Cutoffs {
        d_cut: 1.0e9,
        ..Cutoffs::default()
    });
    let cf = ConFind::with_config(rotlib, bb, false, huge).unwrap();
    let list = cf.contacts(&all(2), -1.0).unwrap();
    assert_eq!(list.pairs.len(), 1, "pair is evaluated under a huge d_cut");
    assert_eq!(list.degrees[0], 0.0);
}

#[test]
fn d_cut_does_not_change_results_when_valid() {
    // §3.8.4 test 3: default d_cut and an effectively unbounded one agree exactly.
    let rotlib = require_rotlib();
    let crambin = load_pdb_f64(crambin_path()).unwrap();
    let bb = truncate_backbone(&crambin, 20);
    let base = outputs(&rotlib, &bb, ConFindConfig::default());
    let huge = outputs(
        &rotlib,
        &bb,
        ConFindConfig::default().with_cutoffs(Cutoffs {
            d_cut: 1.0e9,
            ..Cutoffs::default()
        }),
    );
    assert_eq!(base, huge);
}

#[test]
fn d_cut_boundary_is_inclusive() {
    let rotlib = require_rotlib();
    let at = |gap: f64| {
        let bb = make_synthetic_backbone(2, 25.0 + gap);
        let cf = ConFind::new(rotlib.clone(), bb, false).unwrap();
        cf.neighbors(ResidueIndex(0))
    };
    assert_eq!(at(0.0), vec![ResidueIndex(1)], "25.0 A is evaluated");
    assert!(at(1e-9).is_empty(), "25.0 + 1e-9 A is not");
}

// -------------------------------------------------------------- 10. composition

#[test]
fn freedom_is_nondecreasing_in_tau_and_bounded() {
    let rotlib = require_rotlib();
    let crambin = load_pdb_f64(crambin_path()).unwrap();
    let bb = truncate_backbone(&crambin, 20);
    for policy in POLICIES {
        for mode in FREEDOM {
            let mk = |lo: f64, hi: f64| {
                ConFind::with_config(
                    rotlib.clone(),
                    bb.clone(),
                    false,
                    ConFindConfig::default()
                        .with_cb_clash_policy(policy)
                        .with_freedom_mode(mode)
                        .with_cutoffs(Cutoffs {
                            tau_lo: lo,
                            tau_hi: hi,
                            ..Cutoffs::default()
                        }),
                )
                .unwrap()
            };
            let grid = [(0.25, 1.0), (0.5, 2.0), (1.0, 4.0), (2.0, 8.0)];
            let cfs: Vec<_> = grid.iter().map(|&(lo, hi)| mk(lo, hi)).collect();
            for r in all(bb.bb.len()) {
                let fs: Vec<f64> = cfs.iter().map(|c| c.freedom(r).unwrap()).collect();
                for w in fs.windows(2) {
                    assert!(w[0] <= w[1] + 1e-12, "{policy} {mode} {r:?}: {fs:?}");
                }
                assert!(fs.iter().all(|f| (0.0..=1.0).contains(f)));
            }
        }
    }
}

#[test]
fn tiny_clash_cutoff_prunes_nothing() {
    let rotlib = require_rotlib();
    for policy in POLICIES {
        let cf = ConFind::with_config(
            rotlib.clone(),
            make_helix_backbone(8),
            false,
            ConFindConfig::default()
                .with_cb_clash_policy(policy)
                .with_cutoffs(Cutoffs {
                    d_clash: 0.01,
                    ..Cutoffs::default()
                }),
        )
        .unwrap();
        for r in all(8) {
            let rc = cf.residue_cache(r).unwrap();
            assert_eq!(rc.n_surviving, rc.n_library);
            assert_eq!(cf.crowdedness(r).unwrap(), 0.0);
        }
    }
}

// ----------------------------------------------------- §5.3 invariants, determinism

#[test]
fn contact_degree_is_symmetric_and_bounded() {
    let rotlib = require_rotlib();
    let bb = make_helix_backbone(10);
    let cf = ConFind::new(rotlib, bb, false).unwrap();
    let fwd = all(10);
    let rev: Vec<ResidueIndex> = fwd.iter().rev().copied().collect();
    let a = cf.contacts(&fwd, -1.0).unwrap();
    let b = cf.contacts(&rev, -1.0).unwrap();
    for &(i, j) in &a.pairs {
        let (x, y) = (a.degree(i, j).unwrap(), b.degree(j, i).unwrap());
        assert_eq!(x.to_bits(), y.to_bits());
        assert!((0.0..=1.0).contains(&x));
    }
    assert_eq!(a.pairs.len(), b.pairs.len());
}

#[test]
fn identical_runs_are_bitwise_identical() {
    let rotlib = require_rotlib();
    for (_, bb) in structures() {
        let cfg = ConFindConfig::default()
            .with_crowdedness_mode(CrowdednessMode::PrunedCountFraction)
            .with_freedom_mode(FreedomMode::SingleThreshold);
        assert_eq!(outputs(&rotlib, &bb, cfg.clone()), outputs(&rotlib, &bb, cfg));
    }
}

#[test]
fn interference_is_directed() {
    let rotlib = require_rotlib();
    let cf = ConFind::new(rotlib, make_helix_backbone(10), false).unwrap();
    let residues = all(10);
    cf.cache_all().unwrap();
    let list = cf.interference(&residues, 0.0).unwrap();
    assert_eq!(list.pairs.len(), 10 * 9, "both directions of every pair are reported");
    let hit = list.pairs.iter().zip(&list.degrees).filter(|(_, &d)| d > 0.0).count();
    assert!(hit > 0, "a compact helix must produce some interference");
}
