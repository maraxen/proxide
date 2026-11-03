---
title: Mosaist derivation map by module
description: Module-level map of which proxide code derives from Mosaist, adapts it, or is independent, with what was verified against source and published crates.
status: draft
task_id: 261102_proxide-stable-hardening
date: '261102'
verdict: ''
base_sha: '5c1c035'
---
# Mosaist derivation map by module

Compares proxide `origin/main` 5c1c035 with Mosaist 450816a, whose `LICENSE.md` is CC BY-NC-SA 4.0. This is engineering evidence. Whether any of it is a derivative work is a question for counsel or the upstream lab.

`mstcondeg`, `mstrotlib`, `msttypes` and `mstfasst` below mean the matching files in Mosaist `src/`.

## Summary

| Crate | Status | Published on crates.io | Depends on |
|---|---|---|---|
| `proxide-confind` | Core translated | alpha.9 to alpha.18, MIT | rotlib |
| `proxide-rotlib` | Small parts trace upstream; most is independent | alpha.9 to alpha.18, MIT | no affected crate |
| `proxide_fixer` | Own code independent | alpha.10 to alpha.18, MIT | rotlib, plus one constant from confind |
| `proxide-frag` | Independent | alpha.9 to alpha.18, MIT | no proxide crate |

None of these versions is yanked. No 0.1.0 of these crates was published.

The PyPI wheel is not exposed through these crates. `cargo tree` on `proxide_py` with all features resolves none of the four.

## proxide-confind

| Module | Category | Upstream | Evidence |
|---|---|---|---|
| `cache.rs` | Translated | `mstcondeg:100-190, 519-537` | Same aa, rotamer and atom nesting; same interference update; ALA permanent contacts; comments say "matches Mosaist" |
| `parallel.rs` `contact_degree_raw` | Translated | `mstcondeg:207-281` | Same nested loops, clashing map, sum and denominator |
| `parallel.rs` `run_phases_b_c` | Adapted | `mstcondeg:306-342` | Same collision merge; proxide's own parallel phases and canonical pairs |
| `freedom.rs` | Translated | `mstcondeg:550-588` | Same three variants and the same unusual `n2*n2 + n2*n1` |
| `confind.rs` | Adapted | `mstcondeg:87-98, 433-448, 591-604` | Session and grids mirror init; `bb_interaction` differs |
| `params.rs` | Facts; table unclear | `mstcondeg:57-69` | Same cutoffs and aaProp values, including entries that are not used |
| `grid.rs`, `contact_list.rs` | Algorithm only | `msttypes:3119-3134, 3305-3359` | Standard structures with their own layout |
| `coords.rs` | Algorithm only | `msttypes:918-960` | Standard dihedral definitions; own structure |
| `precondition.rs`, `error.rs`, benches | Independent | none | No upstream counterpart |
| `bin/confind.rs` | Output format | `tests/testConFind.cpp:179-221` | Same output tags and flags; own parser |

The translated and adapted modules are about a third of the non-test lines and all of the crate's function. Replacing them is a rewrite of the core, not a patch.

## proxide-rotlib

| Module | Category | Upstream | Evidence |
|---|---|---|---|
| `rotlib.rs` `load`, `validate_grid` | Format reader | `mstrotlib:19-135` | Field order is set by the format; names `nc`, `na`, `nb` are kept |
| `rotlib.rs` query API, `place_rotamer` | Adapted | `mstrotlib:142-233, 388-406` | Parallel API; own cis-PRO routing and no residue mutation |
| `frame.rs` `backbone_frame` | Adapted, small | `mstrotlib:189-191` | Same axis steps; the convention is needed to read `rotlib.bin` |
| `frame.rs` `switch_frames`, `binning.rs` | Algorithm only | `mstrotlib:137-140, 418-438` | Own formulas; comments cite upstream |
| `sidechain.rs` | Adapted, tiny | `mstcondeg:192-196` | Same CB rule |
| `rotamer_id.rs` | Trivial | `include/mstrotlib.h:10-22` | Same triple |
| `bin/parse_master.rs` | Format tool | `mstrotlib` format | Own tool; its output is non-commercial data |
| `load_pb`, `aa_grid_summary`, `to_protobuf` | Independent | none | `to_protobuf` stamps Mosaist licence strings on converted data |
| `geometry/*`, `rotlib_source/*`, `convert.rs`, `pb.rs`, proto files | Independent, partly read | none | Headers and identifier search; most bodies unread |

## proxide-frag and proxide_fixer

`proxide-frag` is independent. `search.rs` is a linear scan, where upstream's FASST is a recursive multi-segment search. `kabsch.rs` uses nalgebra SVD, and `persist.rs` uses bincode. The only link is the phrase "MASTER-style" in its description. Its tests use no upstream data.

`proxide_fixer`'s own code showed no upstream counterpart. `repack.rs` and `mutate.rs` were partly read; the other modules were checked by headers and identifier search.

- **confind dependency:** non-test code uses only `CANONICAL_AA_NAMES` (`precondition.rs:79`). That list includes Amber names upstream lacks, so inlining it removes the dependency. Test use needs a dev-dependency or a move.
- **rotlib dependency:** real. It calls `num_rotamers`, `place_rotamer`, `rotamer_probability` and `sidechain_atom_names`, and uses `RotamerLibrary` and its error type.

## Test data

These tests hold numbers produced by upstream code or taken from `rotlib.bin`, and they ship in the crate tarball:

- confind: `test_parity_small_pdb.rs`, `test_parity_1dc7.rs`, `test_drift_loadpb_small_pdb.rs`, `test_distogram.rs`
- rotlib: `test_distogram.rs`, `test_placement.rs`, `test_load.rs`, `test_parse_master.rs`

## Smallest set to rewrite or license

1. confind: `cache.rs`, `parallel.rs`, `freedom.rs`, the session logic in `confind.rs`, and possibly the `params.rs` table.
2. rotlib: `load()`, `parse_master` and the Mosaist path in `to_protobuf`, unless counsel accepts the reader as interoperability. Also `backbone_frame`, the query API and `counts_as_sidechain` if counsel wants them independent. All are small.
3. The test tables above: regenerate from a free source, or license.

Clearing rotlib clears fixer once fixer inlines the one constant. Removing `load()` still leaves a working path, since `bin/confind.rs:20` already uses `load_pb`. Confind would then be the only crate needing permission or a core rewrite.

## Basis for a rewrite

According to the PubMed full text of [Zheng & Grigoryan 2017](https://doi.org/10.1371/journal.pone.0178272), the paper defines contact degree: discard backbone-clashing rotamers, count non-hydrogen pairs within 3 Å, weight by amino-acid frequency times rotamer probability, use Dunbrack rotamers and a 0.02 threshold. The extracted text lost the equation itself.

The paper does not fully specify the crate as it stands:

- Freedom in the main text is the fraction of rotamers the backbone accommodates, which is not what the code computes from collision probabilities.
- Interference and the 2.0 and 25 Å cutoffs were not found in the main text.
- S1 Text and [Zheng 2015](https://doi.org/10.1016/j.str.2015.03.015) were not read and may resolve these.

`260529_confind.md:11` is the `Grigoryanlab/Mosaist@450816a` source pin. `260529_confind.md:295` tells implementers to match `cache()` lines 158-160, so the project's own records show the code was written against upstream. Numerical parity tests against upstream would also tie any rewrite back to it.

## Other corrections found

- `parallel.rs:37-55` says the code replicates an upstream bug and cites `contactDegree.cpp`. No such branch exists in the Rust, and that file is not in the Mosaist clone.
- `confind.rs:184` documents EITHER endpoint for interference, but the code requires both (`:197`), unlike upstream `:382`.
- `lib.rs:36-37` cites Grigoryan & DeGrado, a [coiled-coil paper](https://doi.org/10.1016/j.jmb.2010.08.058), as a contact-degree source.
- The licence spec (line 270) and master spec (`:44`) name a frag dependency on confind. `cargo tree` shows none.
- Possible bug unrelated to licensing: `load()` assumes a phi-major sorted file, while upstream regrids. An unsorted file would misassign bins without an error.

## Verification

**Checked directly by the orchestrator on 2026-11-02:**

- `freedom.rs` and the `cache.rs` interference update, against `mstcondeg`
- the `nc`, `na`, `nb` reader against `mstrotlib:30-32`
- the `parallel.rs:37-55` comment
- `cargo tree --all-features` for `proxide_py`, `proxide_fixer` and `proxide-frag`, and reverse dependents of confind and rotlib
- fixer's non-test imports of confind and rotlib
- crates.io version lists for all four crates
- the alpha.18 tarballs of confind and rotlib: they contain the translated source and parity tests, and no `rotlib.bin` or `.pb` data file

**From the subagent only, cited but not re-read:** the remaining file:line rows above, and the paper reading.

**Not verified:**

- Bodies of the rotlib and fixer modules marked partly read.
- Absence of paraphrased code: identifier search cannot prove it.
- Earlier alpha tarballs, and the fixer and frag tarballs.
- Git history and author intent beyond the specs.
- Python scripts in `scripts/analysis` and `scripts/data` mention rotlib or MASTER. They are not in the crates checked here and were not compared with upstream.

**Questions for counsel or the lab:** whether translated Rust is a derivative; whether a format reader and frame convention need permission; the aaProp table; whether embedded test numbers carry NC-SA.
