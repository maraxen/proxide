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
- [260603_master-rotlib-cartesian-derivation](research/260603_master-rotlib-cartesian-derivation.md) — #820: MASTER built rotlib.bin Cartesians from CHARMM ideal ICs (not Engh-Huber); measured ground truth root-causes the load_pb drift; fix = source ICs from proxide's bundled charmm36_protein.xml
- [260603_branch-torsion-offset-diagnostic](research/260603_branch-torsion-offset-diagnostic.md) — #869 diagnostic: measure branch atom torsion offsets in MASTER rotlib.bin; hypothesis = failing residues (GLU/PHE/ASP/LEU/VAL/MET) have offset != template constant; pass = confines error source; fail = error from bond angles or CB placement

## Plans
- [260602_rotlib-confind-actions](plans/260602_rotlib-confind-actions.md) — Post-NLM-synthesis action plan: rotlib and confind code/doc actions with citations

## Decisions
- [260602_contact-threshold-adr](decisions/260602_contact-threshold-adr.md) — ADR: CONTACT_THRESHOLD as public arg with cited const (vs. hard const)

## Specs
- [260529_rotlib](specs/260529_rotlib.md) — `proxide-rotlib` crate: standalone Rust port of MSL RotamerLibrary — binary format, backbone binning, Frame/Transform placement
- [260529_confind](specs/260529_confind.md) — `proxide-confind` crate: rayon ConFind reimplementation; depends on proxide-rotlib (rev 3)
- [260602_dunbrack-rotlib-protobuf-cis-pro](specs/260602_dunbrack-rotlib-protobuf-cis-pro.md) — Dunbrack 2010 → protobuf+zstd rotamer library; adds cis-PRO (CPR); MIT-code/ODC-BY-data; geometry engine (proline-first)
- [260604_rotlib-ic-geometry-schema](specs/260604_rotlib-ic-geometry-schema.md) — Unified residue_geometry.proto (IcRecord build-tree schema, multi-source) + CHARMM36 RTF/CCD importers + parse_master dev tool (#987, #988); supersedes #976
- [260605_browser-wasm-parallel](specs/260605_browser-wasm-parallel.md) — Browser WASM parallelism: orx-parallel + wasm-bindgen-rayon + proxide-parallel-rt; 6 open risks; awaiting oracle critique

## Dynamic Workflows
> Executable Claude Code Workflow scripts (`Workflow({ scriptPath: ... })`). See [dynamic_workflows/INDEX.md](dynamic_workflows/INDEX.md).
- [260602_dunbrack-rotlib-sprint.js](dynamic_workflows/260602_dunbrack-rotlib-sprint.js) — Sprint #12 executor: Dunbrack→protobuf rotlib + cis-PRO (Research ∥ P1→P6). Backlog #814–#820.
- [260604_rotlib-ic-sprint13.js](dynamic_workflows/260604_rotlib-ic-sprint13.js) — Sprint #13 executor: residue_geometry.proto + RTF IC parser + CCD importer + convert_rotlib wiring + parse_master + confind migration. Backlog #987/#988/#869.

## Superpowers
> Skill outputs live in `.praxia/docs/superpowers/plans/` and `.praxia/docs/superpowers/specs/`.
- [plans](superpowers/plans/) — brainstorming + writing-plans outputs
- [specs](superpowers/specs/) — specification outputs
