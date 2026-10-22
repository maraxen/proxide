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

### Changed
- **Changed (behaviour)**: Full-format output carries per-atom elements, residue names and chains; Python no longer infers elements.
- **Changed (behaviour)**: the Atom14 formatter looks up its layout by residue type (`res_type`). Alias residues (HSD, HSE, HSP, ASH, GLH, LYN, CYX, CYM) are placed instead of emitted empty; residues with no Atom14 layout (for example MSE) raise a `UserWarning` naming them (`unplaced_residues`).
- **Changed (behaviour)**: `write_pdb` and `write_mmcif` write no invented values, raise on column overflow and reject non-finite coordinates.
- **Changed (behaviour)**: MOL2 `MOLECULE` lines are counted raw, so a blank name line no longer shifts the charge-type line.
- **Changed (behaviour)**: chain IDs are resolved or reported unknown, never defaulted to `"A"`; failed chain filters raise.
- **Changed (behaviour)**: the PQR parser rejects multi-model files. Multi-model Atom37 parses return a model-stacked `Protein`; other formats keep model 1 and warn with the model count.
- **Changed (behaviour)**: the PQR and mmCIF readers raise when a residue reappears after its chain has moved on.
- **Changed (behaviour)**: unknown molecule partial charges are `None`, never a zero fill.
- **Changed (behaviour)**: `parameterize_molecule` returns empty charge, bond, angle, dihedral and improper arrays and lists `unparameterized_terms` and `unparameterized_atoms`; it used to return fabricated per-term values and zero charges.
- **Changed (behaviour)**: `project_to_openmm_system` and `to_openmm_system` raise `ValueError` when atoms carry no element symbols, instead of treating every atom as carbon. Atom37 and Atom14 Proteins carry no per-atom elements, so these projections raise for them.
- **Changed (behaviour)**: `parameterize_gaff_with_rdkit` and `build_gaff2_ffxml` raise `Gaff2ParameterMissingError` for a missing bond, angle, mass, van der Waals or torsion term. `missing_torsions="omit"` opts out of the torsion case, with a warning. (experimental)
- **Changed (behaviour)**: GAFF2 enumerates every angle and torsion; van der Waals, mass and type stand-ins are removed. (experimental)
- **Changed (behaviour)**: the `gaff-2.2.20.dat` loader reads AMBER `parm.dat` fixed-width fields, so GAFF2 angles, torsion periodicities and impropers differ from earlier releases (720 previously dropped angle types and all impropers now load). (experimental)
- **Changed (behaviour)**: a later AMBER `DIHE` redefinition replaces the earlier one instead of summing.
- **Changed (behaviour)**: the GAFF2 DEF parser rejects malformed `f8`/`f9` bodies, malformed `ATD` rule lines and an explicit DEF with zero parseable rules, instead of dropping them. (experimental)

### Fixed
- The parse cache no longer serves a model-selected parse, and cache hits return chain IDs instead of `None`.
- Chains not in sorted file order were written with the other chain's letter.
- `load_rust(chain_id=...)` keeps the original `chain_index` and no longer writes chain `"A"` by default.
- Model-stack consumers count one structure or refuse, instead of miscounting.
- The confind CLI propagates crowdedness and freedom errors instead of dropping residues. (experimental)

## [0.1.0a17] - 2026-09-29

### Added
- AMBER parm7 topology reader; `parse_amber_trajectory` reads topology plus XTC frames, and returns one `Protein` per requested frame.
- Per-atom GB parameter provenance is exposed to Python.
- `unparameterized_atoms`, `strict_parameterization` and `water_model` expose which atoms fell back to a stand-in and let callers refuse such output.
- rotlib includes a synthetic ALA rotamer entry. (experimental)
- confind computes backbone dihedrals with Mosaist semantics, with configurable chain-break and legacy missing-atom policies. (experimental)

### Changed
- **Changed (behaviour)**: the PDB reader is fail-loud. Malformed records raise instead of being dropped, including non-ASCII lines, too-short ATOM or HETATM records, non-decimal serial or residue numbers, non-finite occupancy or B-factor, and residue reappearance. Blank occupancy and B-factor take documented defaults.
- **Changed (behaviour)**: the mmCIF reader is fail-loud. A second `data_` block, a non-loop `_atom_site` and an unterminated quote raise. Unquoted `?` and `.` are unknown and inapplicable placeholders; a quoted `?` or `.` is a literal value. The misread of the `_atom_sites` and `_atom_site_anisotrop` prefixes is fixed.
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
