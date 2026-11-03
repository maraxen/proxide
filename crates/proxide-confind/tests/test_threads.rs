//! Finding T1: a requested worker count must reach every parallel phase.
//!
//! Spec §3.9 does not mention worker threads. The count is an
//! `Option<NonZeroUsize>` on [`ConFindConfig`]: `None` is the orx-parallel
//! default, and `Some(n)` is the limit. Numeric results with `n = 1` match
//! the default bitwise.

mod common;

use common::{dummy_rotlib, make_synthetic_backbone};
use proxide_confind::{ConFind, ConFindConfig, ResidueIndex};
use std::num::NonZeroUsize;

#[derive(Debug, PartialEq)]
struct Bits {
    crowdedness: Vec<u64>,
    freedom: Vec<u64>,
    contacts: Vec<((u32, u32), u64)>,
}

fn bits(cfg: ConFindConfig) -> Bits {
    let cf = ConFind::with_config(dummy_rotlib(), make_synthetic_backbone(6, 3.8), false, cfg)
        .expect("synthetic backbone and rotamer library must construct");
    let residues: Vec<ResidueIndex> = (0..cf.n_residues() as u32).map(ResidueIndex).collect();
    cf.cache_all().expect("cache_all");
    let contacts = cf.contacts(&residues, 0.0).expect("contacts");
    Bits {
        crowdedness: residues
            .iter()
            .map(|&r| cf.crowdedness(r).expect("crowdedness").to_bits())
            .collect(),
        freedom: residues
            .iter()
            .map(|&r| cf.freedom(r).expect("freedom").to_bits())
            .collect(),
        contacts: contacts
            .pairs
            .iter()
            .zip(&contacts.degrees)
            .map(|(&(a, b), d)| ((a.0, b.0), d.to_bits()))
            .collect(),
    }
}

#[test]
fn thread_count_is_carried_and_one_matches_default_bitwise() {
    let one = NonZeroUsize::new(1).expect("1 is non-zero");
    assert_eq!(ConFindConfig::default().threads, None);

    let limited = ConFindConfig::default().with_threads(Some(one));
    assert_eq!(limited.threads, Some(one));

    let cf = ConFind::with_config(
        dummy_rotlib(),
        make_synthetic_backbone(4, 3.8),
        false,
        limited.clone(),
    )
    .expect("construct with an explicit thread count");
    assert_eq!(
        cf.config().threads,
        Some(one),
        "with_config must retain the requested thread count"
    );

    assert_eq!(
        bits(ConFindConfig::default()),
        bits(limited),
        "threads = 1 and the library default must be bitwise identical"
    );
}
