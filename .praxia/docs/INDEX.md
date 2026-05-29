# proxide Internal Docs

## Notes
- [260529_confind_model](notes/260529_confind_model.md) — Ground-truth algorithmic reference for ConFind (mstcondeg.cpp): parameter defaults, 4-phase cache pipeline, CD formula, freedom types, interference, bb-interaction, CLI I/O contract, parallelization hazard map

## Specs
- [260529_rotlib](specs/260529_rotlib.md) — `proxide-rotlib` crate: standalone Rust port of MSL RotamerLibrary — binary format, backbone binning, Frame/Transform placement
- [260529_confind](specs/260529_confind.md) — `proxide-confind` crate: rayon ConFind reimplementation; depends on proxide-rotlib (rev 3)

## Superpowers
> Skill outputs live in `.praxia/docs/superpowers/plans/` and `.praxia/docs/superpowers/specs/`.
- [plans](superpowers/plans/) — brainstorming + writing-plans outputs
- [specs](superpowers/specs/) — specification outputs
