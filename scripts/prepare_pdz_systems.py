#!/usr/bin/env python3
"""PDZ Domain System Preparation.

Downloads, extracts, and fixes 5 PDZ domain structures for batched simulation.

For each PDB:
  1. Download from RCSB
  2. Inspect chains and residue ranges using biotite
  3. Extract PDZ domain region (+ peptide for complexes)
  4. Run PDBFixer: removeHeterogens, addMissingAtoms, addMissingHydrogens(pH=7.0)
  5. Save to references/pdb/{PDB_ID}_pdz_fixed.pdb
  6. Report atom counts and bucket assignments

Usage:
    uv run python scripts/prepare_pdz_systems.py
    uv run python scripts/prepare_pdz_systems.py --inspect-only  # Just inspect, no fixing
"""

from __future__ import annotations

import argparse
import logging
import math
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pdz_prep")

# ---------------------------------------------------------------------------
# System definitions
# ---------------------------------------------------------------------------
# Each entry: (pdb_id, type, description, extraction_config)
# extraction_config is a dict with:
#   chains: list of chain IDs to keep (None = all ATOM chains)
#   resrange: optional (start, end) tuple for auth residue IDs to keep
#             Applied AFTER chain filtering. Uses PDB auth_seq_id numbering.
#   notes: human-readable description

PDZ_SYSTEMS: list[dict[str, Any]] = [
    {
        "pdb_id": "2Z9I",
        "type": "apo",
        "description": "HtrA protease PDZ domain (M. tuberculosis)",
        "chains": ["A"],
        "resrange": (227, 314),
        "notes": (
            "Multi-domain protein (324 res). Protease domain = res 6-226, "
            "PDZ domain = res 227-314 (CATH 2.30.42.10 / SCOP d2z9ia1). "
            "Extract PDZ only."
        ),
    },
    {
        "pdb_id": "5HEB",
        "type": "complex",
        "description": "PSD-95 PDZ3 + CRIPT peptide",
        "chains": None,  # Keep all chains (PDZ + peptide)
        "notes": "PDZ3 domain (119 res) + CRIPT peptide (9 res), 2 polymer chains",
    },
    {
        "pdb_id": "2VSV",
        "type": "apo",
        "description": "Rhophilin-2 PDZ domain (human)",
        "chains": ["A"],  # 2 copies in ASU — take chain A only
        "notes": "X-ray, 109 residues, 2 copies in ASU — extract chain A only",
    },
    {
        "pdb_id": "1KEF",
        "type": "complex",
        "description": "PDZ1 of SAP90/PSD-95",
        "chains": None,  # Single entity — PDZ+peptide in one chain
        "notes": "NMR, 93 residues, PDZ domain with peptide modeled in same chain",
    },
    {
        "pdb_id": "3ZRT",
        "type": "complex",
        "description": "PSD-95 PDZ1 (from PDZ1-2 tandem)",
        "chains": ["A"],  # 4 copies in ASU — take chain A only
        "resrange": (1, 95),
        "notes": (
            "PDZ1-2 tandem (199 res, auth -7 to 191). "
            "PDZ1 = CATH res 10-103 (entity), auth 1 to 95 (His-tag -7..0 trimmed). "
            "PDZ2 starts at auth 96. Extract PDZ1 only."
        ),
    },
]

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
OUTPUT_DIR = Path("references/pdb")
ATOM_BUCKETS = (1024, 2048, 4096, 8192, 16384)


def next_power_of_2_bucket(n_atoms: int) -> int:
    """Select smallest bucket that fits n_atoms."""
    for b in ATOM_BUCKETS:
        if n_atoms <= b:
            return b
    return 2 ** math.ceil(math.log2(n_atoms))


# ---------------------------------------------------------------------------
# Step 1: Download
# ---------------------------------------------------------------------------
def download_pdb(pdb_id: str, output_dir: Path) -> Path:
    """Download PDB from RCSB if not already cached."""
    output_path = output_dir / f"{pdb_id}.pdb"
    if output_path.exists():
        logger.info("  %s: already downloaded", pdb_id)
        return output_path

    url = RCSB_URL.format(pdb_id=pdb_id)
    logger.info("  %s: downloading from %s", pdb_id, url)
    urllib.request.urlretrieve(url, output_path)
    logger.info("  %s: saved to %s", pdb_id, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Step 2: Inspect & Extract with biotite
# ---------------------------------------------------------------------------
def inspect_pdb(pdb_path: Path) -> None:
    """Print structural summary using biotite."""
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb

    pdb_file = pdb.PDBFile.read(str(pdb_path))
    # Get all models — use first model for inspection
    structure = pdb_file.get_structure(model=1)

    # Filter to ATOM records only (no HETATM)
    atom_only = structure[structure.hetero == False]  # noqa: E712

    chain_ids = struc.get_chains(atom_only)
    logger.info("  Chains: %s", list(chain_ids))

    for chain_id in chain_ids:
        chain_mask = atom_only.chain_id == chain_id
        chain_atoms = atom_only[chain_mask]
        residues = struc.get_residues(chain_atoms)
        res_ids = residues[0]   # unique residue IDs
        res_names = residues[1]  # unique residue names
        n_atoms = len(chain_atoms)
        n_res = len(res_ids)
        logger.info(
            "  Chain %s: %d residues (%d-%d), %d atoms, seq: %s...%s",
            chain_id,
            n_res,
            int(res_ids[0]) if len(res_ids) > 0 else 0,
            int(res_ids[-1]) if len(res_ids) > 0 else 0,
            n_atoms,
            "".join(res_names[:5]),
            "".join(res_names[-3:]) if len(res_names) > 5 else "",
        )


def extract_domain(
    pdb_path: Path,
    output_path: Path,
    chains: list[str] | None = None,
    resrange: tuple[int, int] | None = None,
) -> Path:
    """Extract specific chains/residues from PDB using biotite.

    If chains is None, keeps all ATOM chains.
    If resrange is (start, end), filters to auth residue IDs in [start, end].
    Always uses model 1 only (for NMR structures).
    Removes HETATM records.
    """
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb

    pdb_file = pdb.PDBFile.read(str(pdb_path))
    structure = pdb_file.get_structure(model=1)

    # Filter to ATOM records only (remove HETATM)
    mask = structure.hetero == False  # noqa: E712

    # Filter chains if specified
    if chains is not None:
        chain_mask = sum(structure.chain_id == c for c in chains) > 0
        mask = mask & chain_mask

    # Apply residue range filter if specified
    if resrange is not None:
        start, end = resrange
        res_mask = (structure.res_id >= start) & (structure.res_id <= end)
        mask = mask & res_mask

    filtered = structure[mask]

    n_atoms = len(filtered)
    unique_chains = struc.get_chains(filtered)

    # Report residue range of result
    residues = struc.get_residues(filtered)
    res_ids = residues[0]
    n_res = len(res_ids)

    logger.info(
        "  Extracted: %d atoms, %d residues (auth %d-%d), chains %s",
        n_atoms,
        n_res,
        int(res_ids[0]) if n_res > 0 else 0,
        int(res_ids[-1]) if n_res > 0 else 0,
        list(unique_chains),
    )

    # Write with biotite PDBFile
    out_file = pdb.PDBFile()
    out_file.set_structure(filtered)
    out_file.write(str(output_path))
    logger.info("  Saved extracted structure to %s", output_path)

    return output_path


# ---------------------------------------------------------------------------
# Step 3: PDBFixer — add missing atoms, protonate
# ---------------------------------------------------------------------------
def fix_pdb(input_path: Path, output_path: Path) -> int:
    """Run PDBFixer: fill missing atoms, add hydrogens at pH 7.0.

    Returns atom count of the fixed structure.
    """
    from pdbfixer import PDBFixer
    from openmm import app

    fixer = PDBFixer(filename=str(input_path))

    # Remove heterogens (crystallization additives, ions, remaining waters)
    fixer.removeHeterogens(keepWater=False)

    # Find and report missing residues (but don't add them — we don't want
    # modeled loops that weren't in the crystal structure)
    fixer.findMissingResidues()
    if fixer.missingResidues:
        logger.info(
            "  Missing residues found (NOT adding): %s",
            dict(fixer.missingResidues),
        )
    fixer.missingResidues = {}  # Clear — don't model missing loops

    # Find and add missing atoms (terminal atoms, etc.)
    fixer.findMissingAtoms()
    if fixer.missingAtoms:
        n_missing = sum(len(v) for v in fixer.missingAtoms.values())
        logger.info("  Adding %d missing atoms", n_missing)
    fixer.addMissingAtoms()

    # Add hydrogens at pH 7.0
    # This handles histidine protonation (HID/HIE/HIP) via pKa estimation
    fixer.addMissingHydrogens(7.0)

    # Count atoms
    n_atoms = fixer.topology.getNumAtoms()
    n_residues = fixer.topology.getNumResidues()
    n_chains = fixer.topology.getNumChains()
    logger.info(
        "  Fixed: %d atoms, %d residues, %d chains",
        n_atoms,
        n_residues,
        n_chains,
    )

    # Report protonation of histidines
    his_states = []
    for residue in fixer.topology.residues():
        if residue.name in ("HIS", "HID", "HIE", "HIP"):
            his_states.append(f"{residue.name}{residue.id}")
    if his_states:
        logger.info("  Histidine states: %s", ", ".join(his_states))

    # Write fixed PDB
    with open(output_path, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    logger.info("  Saved fixed structure to %s", output_path)

    return n_atoms


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare PDZ domain systems")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only download and inspect, don't fix",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="Specific PDB IDs to process (default: all)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter systems if requested
    systems = PDZ_SYSTEMS
    if args.systems:
        requested = {s.upper() for s in args.systems}
        systems = [s for s in systems if s["pdb_id"] in requested]

    results = []

    for sys_info in systems:
        pdb_id = sys_info["pdb_id"]
        logger.info("=" * 60)
        logger.info("Processing %s: %s", pdb_id, sys_info["description"])
        logger.info("  Type: %s | Notes: %s", sys_info["type"], sys_info["notes"])
        logger.info("=" * 60)

        # Step 1: Download
        raw_path = download_pdb(pdb_id, OUTPUT_DIR)

        # Step 2: Inspect
        logger.info("--- Inspecting %s ---", pdb_id)
        inspect_pdb(raw_path)

        if args.inspect_only:
            continue

        # Step 3: Extract domain
        extracted_path = OUTPUT_DIR / f"{pdb_id}_pdz.pdb"
        logger.info("--- Extracting domain ---")
        extract_domain(
            raw_path,
            extracted_path,
            chains=sys_info.get("chains"),
            resrange=sys_info.get("resrange"),
        )

        # Step 4: PDBFixer
        fixed_path = OUTPUT_DIR / f"{pdb_id}_pdz_fixed.pdb"
        logger.info("--- Running PDBFixer ---")
        n_atoms = fix_pdb(extracted_path, fixed_path)

        bucket = next_power_of_2_bucket(n_atoms)
        results.append((pdb_id, n_atoms, bucket, sys_info["type"]))

    # Summary table
    if results:
        logger.info("")
        logger.info("=" * 60)
        logger.info("SUMMARY: Atom Counts & Bucket Assignments")
        logger.info("=" * 60)
        logger.info("%-8s  %6s  %6s  %s", "PDB", "Atoms", "Bucket", "Type")
        logger.info("-" * 40)
        for pdb_id, n_atoms, bucket, sys_type in results:
            logger.info("%-8s  %6d  %6d  %s", pdb_id, n_atoms, bucket, sys_type)
        logger.info("-" * 40)

        # Recommend bucket configuration
        bucket_groups: dict[int, list[str]] = {}
        for pdb_id, n_atoms, bucket, _ in results:
            bucket_groups.setdefault(bucket, []).append(pdb_id)

        logger.info("")
        logger.info("Bucket groups:")
        for bucket, pdb_ids in sorted(bucket_groups.items()):
            logger.info(
                "  Bucket %d: %s (%d systems)", bucket, pdb_ids, len(pdb_ids)
            )

        # Print catalog entries for copy-paste
        logger.info("")
        logger.info("SYSTEM_CATALOG entries:")
        for pdb_id, _, _, _ in results:
            logger.info(
                '    "%s_pdz": "references/pdb/%s_pdz_fixed.pdb",',
                pdb_id,
                pdb_id,
            )


if __name__ == "__main__":
    main()
