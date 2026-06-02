# proxide Internal Docs

## Handoffs
- [260601_proxconfind3-close](handoffs/260601_proxconfind3-close.md) — proxconfind3 session close: constrained-contacts + 1DC7 GLY/PRO parity tests; 52/36 pass/ignored; main at ac8776e
- [260601_confind-parity-merge](handoffs/260601_confind-parity-merge.md) — confind parity tests merged to main; 48/36 pass/ignored; deferred: constrained_contacts + 1DC7 GLY/PRO parity
- [260601_rotlib-fixture-expansion](handoffs/260601_rotlib-fixture-expansion.md) — Expand rotlib parity test fixtures: GLY/PRO coverage, real backbone diversity, distogram tests beyond small.pdb
- [260601_confind-fullprot-distogram](handoffs/260601_confind-fullprot-distogram.md) — Full-protein CB distogram parity test in confind: 1DC7 (all 20 AAs, 124 res); revert 2ZTA rotlib distogram

## Notes
- [260529_confind_model](notes/260529_confind_model.md) — Ground-truth algorithmic reference for ConFind (mstcondeg.cpp): parameter defaults, 4-phase cache pipeline, CD formula, freedom types, interference, bb-interaction, CLI I/O contract, parallelization hazard map

## Research
- [260602_rotlib-notebook-plan](research/260602_rotlib-notebook-plan.md) — NotebookLM research notebook plan: rotamer library + confind sources, prompts, and expected outputs for grounding project direction

## Specs
- [260529_rotlib](specs/260529_rotlib.md) — `proxide-rotlib` crate: standalone Rust port of MSL RotamerLibrary — binary format, backbone binning, Frame/Transform placement
- [260529_confind](specs/260529_confind.md) — `proxide-confind` crate: rayon ConFind reimplementation; depends on proxide-rotlib (rev 3)

## Superpowers
> Skill outputs live in `.praxia/docs/superpowers/plans/` and `.praxia/docs/superpowers/specs/`.
- [plans](superpowers/plans/) — brainstorming + writing-plans outputs
- [specs](superpowers/specs/) — specification outputs
