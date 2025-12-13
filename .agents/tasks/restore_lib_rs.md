You are taking over a critical task in the Phase 5 Rust Migration (Priox).

## Current Situation

We are migrating parsing logic to Rust.

- **PQR Parser**: Done.
- **Mass Assignment**: Done.
- **Problem**: The file `rust_ext/src/lib.rs` was corrupted during an edit to add multi-model support. It currently contains syntax errors, mismatched braces, and incomplete replacements.

## Objective

**IMMEDIATELY Restore `rust_ext/src/lib.rs` to a valid state.**

## Restoration Plan

You basically need to rewrite `lib.rs`. Do not try to patch it. Use `write_to_file` to overwrite it with the complete, correct content.

The `lib.rs` file should contain:

1. **Imports**: Standard `pyo3`, `numpy` imports.
2. **`parse_structure` function**: This is where logic was being added. It must implement:
    - Cache check.
    - Dispatch: `if path.ends_with(".cif") || ... { formats::mmcif::... } else { formats::pdb::... }`.
    - Model Splitting: `processing::models::split_by_model`.
    - Model Filtering: Select models based on `spec.models`.
    - Reference Model Processing: Process the first model (`models_to_process[0]`).
    - Hydrogen Addition: Use `geometry::hydrogens` if `spec.add_hydrogens` is true.
    - **Multi-Model Stacking**:
        - Format reference model using `formatters::Atom37Formatter` (or appropriate formatter based on `spec.coord_format`).
        - Iterate over remaining models.
        - Process, H-add, and Format each model.
        - Validates shape consistency.
        - Stack coordinates into a `(N_models, N_res, N_atoms, 3)` array.
        - Update the output dict's `coordinates` item with this stacked array.
    - **Legacy Feature Logic**: The block handling `compute_rbf`, `infer_bonds` (topology), `molecule_type`, `electrostatics`, `vdw`, `parameterize_md`, `GAFF`. *This logic exists in the broken file (lines ~300+).* You should read the broken file to extract this block if you can, or reconstruct it. **Logic:**
        - `compute_rbf`: `geometry::neighbors` + `geometry::radial_basis`.
        - `infer_bonds`: `forcefield::topology::Topology::from_coords`.
        - `electrostatics`: `physics::electrostatics`.
        - `vdw`: `physics::vdw`.
        - `parameterize_md`: `physics::md_params`.
3. **Other Functions**: `parse_pqr`, `load_forcefield`, `parse_xtc`, `parse_dcd`, `parse_trr`, `assign_masses`, and the GBSA/Water/CMAP functions. These are likely intact in the broken file (lines 600+). *Read the file first to recover them.*

## Next Steps after Fix

1. **Build**: Run `maturin develop` to ensure it compiles.
2. **Verify**: Run `pytest tests/io/parsing/test_dispatch.py`.
    - If it fails on `assert len(protein_list) == 4`, update the test to expect `len(protein_list) == 1` (a single batched Protein object), and check `protein.coordinates.shape[0] == 4`.
3. **Feature Request**: The user wants the `Protein` class (in `src/priox/core/atomic_system.py` or similar) to have a `format` attribute (Literal) tracking its underlying format (e.g., 'Atom37'). Add this.

## Resources

- `rust_ext/src/lib.rs` (Broken, source of code fragments).
- `rust_ext/src/formats/mmcif.rs` (Reference for mmCIF return type).
- `.agents/PHASE5_MIGRATION.md` (Status).
