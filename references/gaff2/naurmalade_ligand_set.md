# Naurmalade Ligand Set

## Investigation Summary

Examined `/home/marielle/projects/naurmalade` for enumerated ligand lists or centralized configuration.

## Findings

**No enumerated ligand list found in naurmalade.**

The project uses a per-structure configuration model:

1. **LigandEntry configuration schema** (`src/naurmalade/ligand/config.py`):
   - Individual ligand specifications are defined in YAML/JSON config files passed at runtime
   - Each entry specifies: `name`, `resname`, `smiles`, `charge_method` (default: espaloma-am1bcc), `bonded_ff` (default: gaff-2.2.20)
   - Example from tests: ethanol (resname OHP, SMILES "CCO")

2. **Per-PDB metadata approach** (`inputs/*/collection.yaml` and `*.meta.yaml` files):
   - Structures are tracked via per-PDB YAML metadata files
   - Metadata includes a `has_ligands` boolean flag (predominantly `false` across the collection)
   - Current collections: hyperTEV_pdb (6 structures from tev_design project), sweet_vft (sucralose cosolvent studies)

3. **Ligand parameterization workflow** (`src/naurmalade/ligand/` module):
   - Inputs: holo PDB + YAML/JSON config
   - Process: GAFF2 bonded parameterization + EspalomaCharge hybrid model
   - No hard-coded benchmark ligand set; configurations are project-specific

## Recommendation

For GAFF2 parity validation:
- **If building a naurmalade-based benchmark:** Consider creating a dedicated `benchmark_ligands.yaml` that enumerates small, well-parameterized molecules (e.g., from PDB's ligand frequency set or Amber geostd)
- **Current naurmalade scope:** Appears focused on protein engineering (TEV, sweet enzymes) rather than general ligand benchmarking
- **Integration point:** The modular LigandEntry config allows easy addition of benchmark molecules without core changes
