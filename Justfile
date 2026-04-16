# Justfile for proxide (Parameterization)

default:
    @just --list

# Prepare PDZ test systems
prepare-pdz:
    uv run scripts/prepare_pdz_systems.py

# Sanity check energy terms for a PDB
check-energy pdb_path:
    uv run scripts/check_energy.py {{pdb_path}}

# --- Quality ---

test:
    uv run pytest

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

check:
    uv run pyright
