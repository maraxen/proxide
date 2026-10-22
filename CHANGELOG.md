# Changelog

All notable changes to proxide are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries marked
**Changed (behaviour)** alter parsed or written output for input the previous
release accepted.

## [Unreleased]

### Added
- `write_pdb` and `write_mmcif` accept Atom14 `Protein` objects; atom slot names come from the per-residue Atom14 table.
- GAFF2 reports silent parameter fills (experimental).

### Changed
- **Changed (behaviour)**: `write_pdb` and `write_mmcif` write no invented values and raise on column overflow.
- **Changed (behaviour)**: Full-format output carries per-atom elements, residue names and chains; Python no longer infers elements.
- **Changed (behaviour)**: the Atom14 formatter looks up its layout by residue type (`res_type`), not residue name.
- **Changed (behaviour)**: chain IDs are resolved or reported unknown, never defaulted to `"A"`; failed chain filters raise.
- **Changed (behaviour)**: the PQR parser rejects multi-model files.
- **Changed (behaviour)**: the PQR and mmCIF readers raise when a residue reappears after its chain has moved on.
- **Changed (behaviour)**: multi-model parses either return consistent models or report which models were dropped.
- **Changed (behaviour)**: non-finite coordinates are rejected; MOL2 `MOLECULE` lines are counted raw.
- **Changed (behaviour)**: unknown molecule partial charges are `None`, never a zero fill.
- **Changed (behaviour)**: `project_to_openmm_system` uses each atom's real element; the carbon fallback is removed.
- **Changed (behaviour)**: `parameterize_molecule` reports missing terms instead of fabricating them.
- **Changed (behaviour)**: GAFF2 enumerates every angle and torsion; van der Waals, mass and type stand-ins are removed.
- **Changed (behaviour)**: a later AMBER `DIHE` redefinition replaces the earlier one instead of summing.
- **Changed (behaviour)**: the GAFF2 DEF parser rejects malformed `f8`/`f9` bodies, malformed `ATD` rule lines and an explicit DEF with zero parseable rules, instead of dropping them.
- **Changed (behaviour)**: a missing, altered or empty `ATOMTYPE_GFF2.DEF` raises.
- The `gaff-2.2.20.dat` file is read as an AMBER `parm.dat`.

### Fixed
- Batched `chain_ids` are no longer corrupted by `_stack_padded_proteins`.
- The OpenMM projector registration (`@register_projector("openmm")`) is restored.
- Model-stack consumers count one structure or refuse, instead of miscounting.
- The parse cache no longer serves a model-selected parse; cache hits keep the chain vocabulary.
- Atom37 elements, PDB name alignment and mmCIF token safety are restored in `write_*`.
- Build no longer produces a stray top-level `_proxider` extension module.
- The confind CLI propagates crowdedness and freedom errors instead of dropping residues (experimental).

## [0.1.0a17] - 2026-09-29

### Added
- AMBER parm7 topology reader; `parse_amber_trajectory` reads topology plus XTC frames.
- `parse_amber_trajectory` returns one `Protein` per requested frame.
- rotlib includes a synthetic ALA rotamer entry (experimental).
- Per-atom GB parameter provenance is exposed to Python.
- confind computes backbone dihedrals with Mosaist semantics, with configurable chain-break and legacy missing-atom policies (experimental).

### Changed
- **Changed (behaviour)**: the PDB reader is fail-loud; malformed records raise instead of being dropped.
- **Changed (behaviour)**: the mmCIF reader is fail-loud; malformed records raise instead of being dropped.
- **Changed (behaviour)**: the PQR reader is fail-loud; malformed records raise instead of being dropped.
- **Changed (behaviour)**: `infer_element` strips a leading digit from atom names before inference.
- **Changed (behaviour)**: solvent atoms are parameterized instead of silently zeroed.
- **Changed (behaviour)**: the 1-4 pair list excludes solvent and out-of-range indices.
- **Changed (behaviour)**: mbondi2 radii come from a provenance-carrying table, and every fallback is tagged.
- **Changed (behaviour)**: silicon resolves as Si, not sulfur.
- **Changed (behaviour)**: fixer element inference goes through `infer_element`.
- **Changed (behaviour)**: the GAFF2 partial-charge method is explicit, validated and recorded; there is no silent fallback and no zero fill.
- The GAFF2 loader raises when `ATOMTYPE_GFF2.DEF` is missing.
- confind propagates rotamer-library errors instead of dropping amino acids (experimental).
- The fixer and confind PDB loaders read through the fail-loud reader (experimental).
- Stub Python bindings that returned empty results are removed (`run_confind`, `search_fragments`, rotlib stubs).

### Fixed
- The parm7 reader joins each residue's atoms before labelling chains.
- ALA chi1 dihedral removed from rotlib (experimental).

## [0.1.0a16] - 2026-08-27

### Added
- `tm_align` and `pairwise_tm_scores` are exposed to Python (TM-align, Rust `proxide-tmalign`).
- `read_xtc_ca_distogram` computes a C-alpha distogram with the minimum-image convention.
- XTC readers expose `frame_count` and `n_atoms`.
- `build_ligand_reference_frame`, `canonicalize_ligand_topology` and `extract_ligand_frame_coordinates` (behind the `ligand-frame` Cargo feature).
- GAFF2 Rust engine (DEF-grammar typer, `proxide-gaff2`) with real H-atom typing and AR1-AR5 aromaticity.
- Ensemble completion of partially observed structures from a donor ensemble (Rust).

### Changed
- **Changed (behaviour)**: `assign_gaff_atom_types` is cut over to the validated GAFF2 engine (breaking).
- **Changed (behaviour)**: `write_pdb` and `write_mmcif` reject batched `Protein` input instead of writing corrupted output.
- **Changed (behaviour)**: chain letters resolve through `chain_index`, not a flat atom-slot index.
- **Changed (behaviour)**: RDKit no longer fabricates implicit hydrogens on explicit-H structures.
- Alphabet tables in `io/mappings.py` and `chem/conversion.py` come from `alphex`, now a runtime dependency.
- `jax` is pinned below 0.11 until flax supports it.
- `ATOMTYPE_GFF2.DEF` is fetched at build time instead of being vendored.
- The `gaff2` Rust crate is licensed GPL-2.0-or-later, with NOTICE and LICENSE files.
- The XTC offset-cache version is bumped; existing caches are rebuilt.

### Fixed
- DCD unit cells decode to `[a, b, c, alpha, beta, gamma]` correctly.
- XTC scanning tolerates EOF in the metadata region; a truncated trailing frame is handled.
- `read_frames_parallel` filters `atom_indices` inside its workers.
- Parsing loads JAX lazily, so fixtures work in torch-only environments.
- `FoldCompDatabase` is cfg-gated behind the `foldcomp` feature.
- GAFF2 typing fixes: ring alternation, `atadjust` reseeding, fused-ring lone-pair donors, DEF `cp`/`ca` mistyping.

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
- `radius_of_gyration` and `weighted_radius_of_gyration` (Rust `proxide-geometry`).
- `kabsch_rmsd` for coordinate sets of arbitrary length (Rust).

### Changed
- **Changed (behaviour)**: `weighted_radius_of_gyration` returns NaN for NaN weights instead of 0.0.

## [0.1.0a12] - 2026-07-07

### Added
- `read_fasta`, `read_newick` and `jaccard_distance_matrix` are exposed to Python.
- `proxide-jaccard` crate: pairwise Jaccard and containment distance matrices with Parquet pushdown (Rust).

## [0.1.0a11] - 2026-07-06

### Fixed
- Wheel builds vendor `protoc`, which `proxide-rotlib` needs at build time.
