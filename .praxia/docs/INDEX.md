# gaff2-parity-campaign Internal Docs

## Daily

## Handoffs
- [260601_confind-fullprot-distogram](handoffs/260601_confind-fullprot-distogram.md) — Full-protein CB distogram parity test in confind: 1DC7 (all 20 AAs, 124 res); revert 2ZTA rotlib distogram
- [260601_confind-parity-merge](handoffs/260601_confind-parity-merge.md) — confind parity tests merged to main; 48/36 pass/ignored; deferred: constrained_contacts + 1DC7 GLY/PRO parity
- [260601_proxconfind3-close](handoffs/260601_proxconfind3-close.md) — proxconfind3 session close: constrained-contacts + 1DC7 GLY/PRO parity tests; 52/36 pass/ignored; main at ac8776e
- [260601_rotlib-fixture-expansion](handoffs/260601_rotlib-fixture-expansion.md) — Expand rotlib parity test fixtures: GLY/PRO coverage, real backbone diversity, distogram tests beyond small.pdb

## Plans
- [260616_cargo-publish-new-crates](plans/260616_cargo-publish-new-crates.md)
- [260602_rotlib-confind-actions](plans/260602_rotlib-confind-actions.md) — Implementation plan for post-NLM-synthesis code and doc actions in proxide-rotlib and proxide-confind
- [260529_rotlib_impl_plan](plans/260529_rotlib_impl_plan.md)
- [260417_comprehensive-deployment-plan](plans/260417_comprehensive-deployment-plan.md)
- [260416_deployment-plan](plans/260416_deployment-plan.md)
- [260416_phase2-plan](plans/260416_phase2-plan.md)

## Specs
- [260729_proxide-tmalign-phases-2-5](specs/260729_proxide-tmalign-phases-2-5.md) — Remaining seeds, parity harness, orx-parallel, PyO3 bindings, bathos benchmark for the TM-align port
- [260630_proxide-jaccard](specs/260630_proxide-jaccard.md) — proxide-jaccard crate — pairwise Jaccard distance matrices over scaled-MinHash genome sketches, parallelized with orx-parallel
- [260618_system-prep-scope](specs/260618_system-prep-scope.md) — #977 EPIC RESEARCH: full system-prep audit — 12-component inventory (present/partial/missing), reference-tool matrix, MIT licensing survey, prioritized child DAG (C1–C13)
- [260605_browser-wasm-parallel](specs/260605_browser-wasm-parallel.md) — Browser WASM parallelism: orx-parallel + wasm-bindgen-rayon + proxide-parallel-rt; 6 open risks; awaiting oracle critique
- [260604_rotlib-ic-geometry-schema](specs/260604_rotlib-ic-geometry-schema.md) — Unified residue_geometry.proto (IcRecord build-tree schema, multi-source) + CHARMM36 RTF/CCD importers + parse_master dev tool (#987, #988); supersedes #976
- [260604_rotlib-multi-source-arch](specs/260604_rotlib-multi-source-arch.md)
- [260603_charmm-ic-sourcing](specs/260603_charmm-ic-sourcing.md) — Implementation spec — source rotamer template internal coordinates from bundled CHARMM36 FFXML (all residues) to match MASTER rotlib.bin geometry and reduce confind load_pb drift
- [260602_design-the-proxide-master-crate-backbone](specs/260602_design-the-proxide-master-crate-backbone.md)
- [260602_dunbrack-rotlib-protobuf-cis-pro](specs/260602_dunbrack-rotlib-protobuf-cis-pro.md) — Dunbrack 2010 → protobuf+zstd rotamer library; adds cis-PRO (CPR); MIT-code/ODC-BY-data; geometry engine (proline-first)
- [260602_proxide-master-spec](specs/260602_proxide-master-spec.md)
- [260529_confind](specs/260529_confind.md) — Spec for proxide-confind crate — Rust/rayon reimplementation of ConFind contact-degree algorithm (rev 4 — cis-proline detection, omega field, backbone_bin cis_proline param)
- [260529_rotlib](specs/260529_rotlib.md) — Spec for proxide-rotlib crate — standalone Rust port of MSL RotamerLibrary (mstrotlib.cpp), backbone binning, Frame/Transform rigid-body placement
- [260529_rotlib_testplan](specs/260529_rotlib_testplan.md) — Test plan for proxide-rotlib — 55 unit + 4 integration tests across 6 modules

## Actuation Surfaces

## Audits
- [260820_gaff2-parity-verdict](audits/260820_gaff2-parity-verdict.md) — Phase 5 graded verdict (PARTIAL) for the GAFF2 atom-typing bathos-literature-parity campaign, closing out Phases 3-5

## Research
- [260729_tm-align-phase-2-algorithm-map](research/260729_tm-align-phase-2-algorithm-map.md) — Line-referenced map of USalign's 5 seeding strategies, get_score_fast, DP_iter/NWDP_TM/TMscore8_search, and final multi-TM output
- [260630_arrow-ipc-prototype](research/260630_arrow-ipc-prototype.md) — proxide-jaccard — Arrow IPC + sorted accession index prototype, measured against the real corpus; planus vs flatbuffers vs Arrow IPC tradeoff analysis
- [260603_branch-torsion-offset-diagnostic](research/260603_branch-torsion-offset-diagnostic.md) — #869 diagnostic: measure branch atom torsion offsets in MASTER rotlib.bin; hypothesis = failing residues (GLU/PHE/ASP/LEU/VAL/MET) have offset != template constant; pass = confines error source; fail = error from bond angles or CB placement
- [260603_master-rotlib-cartesian-derivation](research/260603_master-rotlib-cartesian-derivation.md) — Backlog
- [260602_dunbrack-geometry-synthesis](research/260602_dunbrack-geometry-synthesis.jsonl)
- [260602_rotlib-notebook-plan](research/260602_rotlib-notebook-plan.md) — NotebookLM research notebook plan for rotamer library and confind — sources, prompts, and expected outputs to ground project direction

## Decisions
- [260818_gaff2-parity-verdict-policy](decisions/260818_gaff2-parity-verdict-policy.md) — Verdict policy and tolerances for bathos-literature-parity validation of GAFF2 atom typing implementation
- [260623_coords_to_chi_path](decisions/260623_coords_to_chi_path.md) — ADR: pure-Python coords→χ + chi_to_coords chosen; PyO3/Rust deferred (B4 #2654)
- [260602_contact-threshold-adr](decisions/260602_contact-threshold-adr.md) — Architecture decision record for CONTACT_THRESHOLD in proxide-confind — const vs. public arg with cited default

## Preregistration

## Reference

## Roadmaps

## Archive

## Misc
- [260630_jaccard-output-format-debt](misc/260630_jaccard-output-format-debt.md) — Tech debt — proxide-jaccard's dense symmetric .npy output is an MVP choice; revisit for memory/IO efficiency once matrix sizes and JAX consumption patterns are known
- [260623_gaff2-typing-debt](misc/260623_gaff2-typing-debt.md) — Tech debt — deeper audit of GAFF2 atom typing + parity CI tests against AmberTools/OpenFF reference
- [260417_oracle-critique-01-1](misc/260417_oracle-critique-01-1.md)
- [260417_oracle-critique-01](misc/260417_oracle-critique-01.md)
- [260417_oracle-critique-02](misc/260417_oracle-critique-02.md)
- [260417_oracle-critique-03](misc/260417_oracle-critique-03.md)
- [260417_untitled-1](misc/260417_untitled-1.md)
- [260417_untitled-2](misc/260417_untitled-2.md)
- [260417_untitled-3](misc/260417_untitled-3.md)
- [260417_untitled](misc/260417_untitled.md)

## Superpowers
> Skill outputs live in `.praxia/docs/superpowers/plans/` and `.praxia/docs/superpowers/specs/.
- [plans](superpowers/plans/) — brainstorming + writing-plans outputs
- [specs](superpowers/specs/) — specification outputs

