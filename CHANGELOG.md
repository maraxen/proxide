# Changelog

All notable changes to proxide are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries marked
**Changed (behaviour)** alter parsed or written output for input the previous
release accepted.

PyPI wheels are built with the `full` and `xtc` feature sets. The GPL-2.0-or-later
GAFF2 engine (`gaff2-engine`) and the `ligand-frame` feature are not in the wheels;
they exist only in source builds that opt in.

## [Unreleased]

### Added
- `write_pdb` and `write_mmcif` accept Atom14 `Protein` objects; atom slot names come from the per-residue Atom14 table.
- `Protein.atom_chain_ids` carries per-atom chain IDs; `Protein.n_models` reports the size of a model stack.
- `Protein.b_factors` and `Protein.occupancies` carry the source per-atom values (`None` when the source has none) through every format, the cache, truncation, padding, stacking and `write_npz`.
- `Protein.residue_names` keeps each residue's source name (for example `MSE`) alongside the model `aatype`.
- `fetch_rcsb` and `fetch_afdb` honour `PROXIDE_HTTP_TIMEOUT_SECS` and `PROXIDE_HTTP_CONNECT_TIMEOUT_SECS`; `PROXIDE_RCSB_BASE_URL` overrides the download host (for tests).
- `measure_chi` accepts an optional `atom_mask`; with it, an atom is missing exactly when its mask is 0.
- `proxide.__experimental__` lists the top-level exports outside the stable core IO set; the README Stability section names the stable groups. No names were removed.
- `proxide` exports `write_pdb`, `write_mmcif`, `write_npz`, `Protein` and `UnitSystem` at top level; `proxide.io` exports the three writers.
- `parse_mmcif`, `parse_structure`, `load_rust` and `project_to_mpnn_batch` accept `numbering` (`"auth"` or `"label"`).

### Changed
- **Changed (behaviour)**: Full-format output carries per-atom elements, residue names and chains; Python no longer infers elements.
- **Changed (behaviour)**: the Atom14 formatter looks up its layout by residue type (`res_type`). Alias residues (HSD, HSE, HSP, ASH, GLH, LYN, CYX, CYM) are placed instead of emitted empty; residues with no Atom14 layout (for example ligands) raise a `UserWarning` naming them (`unplaced_residues`).
- **Changed (behaviour)**: `write_pdb` and `write_mmcif` write no invented values and reject non-finite coordinates. `write_pdb` also raises on column overflow; `write_mmcif` rejects values that cannot be written as CIF tokens.
- **Changed (behaviour)**: MOL2 `MOLECULE` lines are counted raw, so a blank name line no longer shifts the charge-type line.
- **Changed (behaviour)**: chain IDs are resolved or reported unknown, never defaulted to `"A"`; failed chain filters raise.
- **Changed (behaviour)**: the PQR parser rejects multi-model files. Multi-model Atom37 parses return a model-stacked `Protein`; other formats keep model 1 and warn with the model count.
- **Changed (behaviour)**: the PQR and mmCIF readers raise when a residue reappears after its chain has moved on.
- **Changed (behaviour)**: unknown molecule partial charges are `None`, never a zero fill.
- **Changed (behaviour)**: `parameterize_molecule` omits the `charges` key when no charges exist, omits each bonded key whose term list is empty, and returns `unparameterized_terms` and `unparameterized_atoms`. It used to return fabricated per-term values and an all-zero `charges` array.
- **Changed (behaviour)**: `project_to_openmm_system` and `to_openmm_system` raise `ValueError` when atoms carry no element symbols, instead of treating every atom as carbon. Atom37 and Atom14 Proteins carry no per-atom elements, so these projections raise for them.
- **Changed (behaviour)**: `parameterize_gaff_with_rdkit` and `build_gaff2_ffxml` raise `Gaff2ParameterMissingError` for a missing bond, angle, mass, van der Waals or torsion term. `missing_torsions="omit"` opts out of the torsion case, with a warning. (experimental)
- **Changed (behaviour)**: GAFF2 enumerates every angle and torsion; van der Waals, mass and type stand-ins are removed. (experimental)
- **Changed (behaviour)**: the `gaff-2.2.20.dat` loader reads AMBER `parm.dat` fixed-width fields, so GAFF2 angles, torsion periodicities and impropers differ from earlier releases (720 previously dropped angle types and all impropers now load). (experimental)
- **Changed (behaviour)**: a later AMBER `DIHE` redefinition replaces the earlier one instead of summing.
- **Changed (behaviour)**: element symbols are normalised at parse time. A filled PDB element column or mmCIF `type_symbol` is trimmed, title-cased and checked against the 118 IUPAC symbols plus D, T and X; an unknown symbol raises `PROX-IO-MALFORMED-RECORD`. Sign-last mmCIF charge suffixes (`Fe2+`) are stripped. Writers emit upper-case elements right-justified in PDB columns 77-78.
- **Changed (behaviour)**: `write_pdb` and `write_mmcif` write the source occupancy and B-factor, and blank columns (PDB) or `?` (mmCIF) when they are unknown. They used to write `1.00` and `0.00` for every atom.
- **Changed (behaviour)**: selenomethionine (MSE) is typed as MET (`aatype` 12) with SE in the SD slot in Atom14 and Atom37 (it was UNK and empty in Atom14; Atom37 dropped SE). `write_pdb` and `write_mmcif` restore `HETATM MSE` with atom `SE` from `residue_names`; proteins without `residue_names` are written as MET/SD.
- **Changed (behaviour)**: `fetch_afdb` and `proxide fetch afdb` default to AlphaFold DB model version 6; version 4 is no longer served.
- **Changed (behaviour)**: stacking proteins where only some carry `b_factors`/`occupancies` warns once and drops those fields for the batch.
- Coordinates where `atom_mask` is 0 are 0.0 and carry no meaning; use `atom_mask`, not NaN tests, to find missing atoms. Frame validation and `measure_chi` now follow this.
- **Changed (behaviour)**: the GAFF2 DEF parser rejects malformed `f8`/`f9` bodies, malformed `ATD` rule lines and an explicit DEF with zero parseable rules, instead of dropping them. (experimental)
- **Changed (behaviour)**: mmCIF chains and residues use author numbering (`auth_asym_id`, `auth_seq_id`) by default; `numbering="label"` restores label numbering. The parse cache is keyed by numbering. With auth numbering, a file whose author numbering revisits a residue after its chain has moved on now raises.
- **Changed (behaviour)**: altloc resolution keys atoms on model, chain, residue number, insertion code and atom name, so insertion-coded residues (`17` and `17A`) are no longer merged in mmCIF or PDB input.
- **Changed (behaviour)**: `compute_pairwise_displacements` raises `ValueError` for inputs that are not 2-D (N, D) with matching D. (experimental)
- The jax-md dependency is removed; free-space displacement and distance are plain `jax.numpy` with identical values and gradients. flax is declared directly.
- `alphex` requires 0.1.0 or later, its first stable release.
- Python 3.12+ installs get jax 0.11 with flax 0.12.10 or later; Python 3.11 stays on jax 0.10.

### Fixed
- The parse cache no longer serves a model-selected parse, and cache hits return chain IDs instead of `None`.
- Chains not in sorted file order were written with the other chain's letter.
- `load_rust(chain_id=...)` remaps `chain_index` onto the filtered chain vocabulary (kept in file order), so a single-chain parse no longer writes chain `"A"`.
- Model-stack consumers count one structure or refuse, instead of miscounting.
- The confind CLI propagates crowdedness and freedom errors instead of dropping residues. (experimental)
- The format cache is keyed on file content (path, length, content hash and output options), so a file rewritten at the same path is re-parsed instead of served stale; the key and the parse use the same read.
- `fetch_rcsb` no longer hangs: the Python bindings released the GIL too late, deadlocking the HTTP client's logging thread. Downloads are atomic (temp file + rename), a truncated or empty cached file is re-downloaded, the format fallback runs only on HTTP 404, and requests send a `User-Agent`.
- Element masses for upper-case element columns (`CL`, `ZN`) no longer fall back to carbon.
- Altloc resolution of structures with serial numbers above 99999 no longer fails under auth numbering, and HETATM records, insertion codes and models survive it.

## [0.1.0a17] - 2026-09-29

### Added
- AMBER parm7 topology reader; `parse_amber_trajectory` reads topology plus XTC frames, and returns one `Protein` per requested frame.
- Per-atom GB parameter provenance is exposed to Python.
- `unparameterized_atoms`, `strict_parameterization` and `water_model` expose which atoms fell back to a stand-in and let callers refuse such output.
- rotlib includes a synthetic ALA rotamer entry. (experimental)
- confind computes backbone dihedrals with Mosaist semantics, with configurable chain-break and legacy missing-atom policies. (experimental)

### Changed
- **Changed (behaviour)**: the PDB reader is fail-loud. Malformed records raise instead of being dropped, including non-ASCII lines, too-short ATOM or HETATM records, non-decimal serial or residue numbers, non-finite occupancy or B-factor, and residue reappearance. Blank occupancy and B-factor take documented defaults.
- **Changed (behaviour)**: the mmCIF reader is fail-loud. A non-loop `_atom_site`, an unterminated quote, and an `_atom_site` loop repeated in a second `data_` block raise. Unquoted `?` and `.` are unknown and inapplicable placeholders; a quoted `?` or `.` is a literal value. The misread of the `_atom_sites` and `_atom_site_anisotrop` prefixes is fixed.
- **Changed (behaviour)**: the PQR reader is fail-loud; malformed records raise instead of being dropped.
- **Changed (behaviour)**: `infer_element` strips a leading digit from atom names before inference, and the PDB, mmCIF and PQR readers infer two-letter elements (Cl, Na) instead of truncating to the first letter.
- **Changed (behaviour)**: solvent atoms are parameterized instead of silently zeroed.
- **Changed (behaviour)**: the 1-4 pair list excludes solvent and out-of-range indices.
- **Changed (behaviour)**: mbondi2 radii come from a provenance-carrying table, and every fallback is tagged.
- **Changed (behaviour)**: silicon resolves as Si, not sulfur.
- **Changed (behaviour)**: fixer element inference goes through `infer_element`. (experimental)
- **Changed (behaviour)**: the GAFF2 partial-charge method is explicit, validated and recorded; there is no silent fallback and no zero fill. (experimental)
- **Changed (behaviour)**: a missing, altered or empty `ATOMTYPE_GFF2.DEF` raises `Gaff2DefMissingError` or `Gaff2DefInvalidError` instead of silently loading an empty ruleset. (experimental)
- **Changed (behaviour)**: stub Python bindings that returned empty results are removed (`run_confind`, `search_fragments` and the rotlib stubs). Code that called them will now fail with an import or attribute error.
- confind propagates rotamer-library errors instead of dropping amino acids. (experimental)
- The fixer and confind PDB loaders read through the fail-loud reader. (experimental)

### Fixed
- Out-of-range topology indices return an error instead of panicking in `md_params`.
- rotlib no longer applies a spurious ALA chi1 dihedral. (experimental)

## [0.1.0a16] - 2026-08-27

### Added
- `tm_align` and `tm_scores_fixed_correspondence` are exported from `proxide.tmalign`; `pairwise_tm_scores` is available in Rust only.
- `read_xtc_ca_distogram` computes a C-alpha distogram with the minimum-image convention.
- XTC readers expose `frame_count` and `n_atoms`.
- `build_ligand_reference_frame`, `canonicalize_ligand_topology` and `extract_ligand_frame_coordinates` (experimental; source builds with `--features ligand-frame` only, not in the PyPI wheel).
- GAFF2 Rust engine (DEF-grammar typer, `proxide-gaff2`) with real H-atom typing and AR1-AR5 aromaticity (experimental; built only with `--features gaff2-engine`, not in the PyPI wheel).
- Ensemble completion of partially observed structures from a donor ensemble (Rust).

### Changed
- **Changed (behaviour)**: `assign_gaff_atom_types` is cut over to the validated GAFF2 engine (breaking). This applies only to builds with `--features gaff2-engine`, which is GPL-2.0-or-later and not in the PyPI wheels.
- **Changed (behaviour)**: `write_pdb` and `write_mmcif` reject batched `Protein` input instead of writing corrupted output.
- **Changed (behaviour)**: chain letters resolve through `chain_index`, not a flat atom-slot index.
- **Changed (behaviour)**: RDKit no longer fabricates implicit hydrogens on explicit-H structures.
- The `gaff2` Rust crate is licensed GPL-2.0-or-later, with NOTICE and LICENSE files.
- Alphabet tables in `io/mappings.py` and `chem/conversion.py` come from `alphex`, now a runtime dependency.
- `jax` is pinned below 0.11 until flax supports it.
- `ATOMTYPE_GFF2.DEF` is fetched at build time instead of being vendored.
- The XTC offset-cache version is bumped; existing caches are rebuilt.

### Fixed
- `_stack_padded_proteins` no longer corrupts batched `chain_ids`.
- DCD unit cells decode to `[a, b, c, alpha, beta, gamma]` correctly.
- XTC scanning tolerates EOF in the metadata region; a truncated trailing frame is handled.
- `read_frames_parallel` filters `atom_indices` inside its workers.
- Parsing loads JAX lazily, so fixtures work in torch-only environments.
- `FoldCompDatabase` is cfg-gated behind the `foldcomp` feature.
- GAFF2 typing fixes: ring alternation, `atadjust` reseeding, fused-ring lone-pair donors, DEF `cp`/`ca` mistyping. (experimental)

## [0.1.0a15] - 2026-07-22

### Added
- XTC offset `.npz` sidecars written by MDAnalysis can be imported.

## [0.1.0a14] - 2026-07-22

### Added
- Lazy and parallel XTC readers are exposed to Python.
- Lazy and parallel XTC readers expose `times` and `total_frames_on_disk`.

## [0.1.0a13] - 2026-07-22

### Added
- Random-access XTC reading through the molly offset API.
- `radius_of_gyration`, `weighted_radius_of_gyration` and `kabsch_rmsd` are exposed to Python (Rust `proxide-geometry`). `weighted_radius_of_gyration` propagates NaN for NaN weights instead of returning 0.0.

## [0.1.0a12] - 2026-07-07

### Added
- `read_fasta`, `read_newick` and `jaccard_distance_matrix` are exposed to Python.
- `proxide-jaccard` crate: pairwise Jaccard and containment distance matrices with Parquet pushdown (Rust).

## [0.1.0a11] - 2026-07-06

### Fixed
- Wheel builds vendor `protoc`, which `proxide-rotlib` needs at build time.
