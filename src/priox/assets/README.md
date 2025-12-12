# Force Field Assets

This directory contains force field XML files from:
- [openmmforcefields](https://github.com/openmm/openmmforcefields)
- [OpenMM](https://github.com/openmm/openmm) bundled data

## Directory Structure

```
assets/
├── amber/          # Amber protein, lipid, nucleic acid force fields (from openmmforcefields)
├── charmm/         # CHARMM force fields (from openmmforcefields)
├── gaff/           # GAFF small molecule force fields
│   ├── ffxml/      # OpenMM-format XML files
│   └── dat/        # Original GAFF .dat parameter files
├── implicit/       # Implicit solvent (GBSA-OBC) parameters
├── openmm_bundled/ # Force fields bundled with OpenMM itself
│   ├── amber14/    # Amber ff14SB with water models
│   ├── amber19/    # Amber ff19SB with water models
│   └── charmm36/   # CHARMM36 with water models
└── water/          # Water models and ion parameters
```

## Supported Force Fields

### Amber Protein Force Fields (openmmforcefields)
- `ff14SB.xml` - Amber ff14SB (recommended for proteins)
- `protein.ff19SB.xml` - Amber ff19SB (latest)
- `ff99SBildn.xml` - Amber ff99SB-ILDN

### Amber Protein Force Fields (OpenMM bundled)
- `amber14/` - ff14SB with tip3p, tip3pfb, tip4pew, tip4pfb
- `amber19/` - ff19SB with opc, opc3

### GAFF (Small Molecules)
- `gaff-2.11.xml` - GAFF 2.11 (recommended)
- `gaff-2.2.20.xml` - GAFF 2.2.20 (latest)
- `gaff-1.81.xml` - GAFF 1.81 (legacy)

### Water Models
- `tip3p_standard.xml` - TIP3P
- `opc_standard.xml` - OPC (recommended)
- `tip4pew_standard.xml` - TIP4P-Ew

### Implicit Solvent (GBSA-OBC)
- `amber99_obc.xml` - GBSA-OBC parameters for Amber99
- `amber03_obc.xml` - GBSA-OBC parameters for Amber03
- `amber10_obc.xml` - GBSA-OBC parameters for Amber10
- `amber96_obc.xml` - GBSA-OBC parameters for Amber96

### CHARMM
- `charmm36.xml` - CHARMM36 (complete)
- `charmm36_protein.xml` - CHARMM36 proteins only

## Updating Assets

To update these files from the source repositories:

```bash
uv run python scripts/sync_forcefields.py
```

To force a fresh clone:

```bash
uv run python scripts/sync_forcefields.py --force
```

## License

Force field files are distributed under the terms of their original licenses.
See the respective repositories for details.
