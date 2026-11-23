"""Generate byte-level golden files for proxide's PDB/mmCIF writers.

Spec ``.praxia/docs/specs/261123_pdb-writer-icode-hetero.md``, task T0. The
goldens pin what ``write_pdb``/``write_mmcif`` produce on the code this script
runs against, so the writer change (insertion codes, Full-layout hetero flag)
can be checked for byte compatibility in T4:

  (a) 5awl Atom37 -> PDB and mmCIF
  (b) 5awl Atom14 -> PDB
  (c) a hand-built Full Protein with no hetero / insertion codes -> PDB and mmCIF
  (d) 5awl Full -> PDB and mmCIF (its 12 HOH rows are expected to change
      record type after the change; everything else must not)

Safety rails:
  * ``proxide`` and ``proxide._proxider`` must both resolve inside the checkout
    that holds this script. Another checkout's venv (for example the main
    checkout, which is on a different branch) would silently pin the wrong bytes.
  * An existing golden is never overwritten without ``--force``, so a re-run
    after the writer change cannot silently re-pin.
  * mmCIF embeds ``data_<stem>``, so each golden is written under its fixed
    final file name; T4 must write to the same name before comparing.

Usage (from the repository root, with the extension built from this checkout):

    .venv/bin/python scripts/golden/make_writer_goldens.py
    .venv/bin/python scripts/golden/make_writer_goldens.py --out-dir /tmp/x  # determinism check
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("make_writer_goldens")

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PDB = REPO_ROOT / "tests" / "data" / "5awl.pdb"
DEFAULT_OUT = REPO_ROOT / "tests" / "data" / "golden_writer"

GOLDENS: tuple[tuple[str, str, str], ...] = (
  # (file name, source, writer)
  ("5awl_atom37.pdb", "5awl:Atom37", "pdb"),
  ("5awl_atom37.cif", "5awl:Atom37", "mmcif"),
  ("5awl_atom14.pdb", "5awl:Atom14", "pdb"),
  ("handbuilt_full.pdb", "handbuilt:Full", "pdb"),
  ("handbuilt_full.cif", "handbuilt:Full", "mmcif"),
  ("5awl_full.pdb", "5awl:Full", "pdb"),
  ("5awl_full.cif", "5awl:Full", "mmcif"),
)


def _inside(path: str | None, root: Path) -> bool:
  if path is None:
    return False
  try:
    Path(path).resolve().relative_to(root)
  except ValueError:
    return False
  return True


def check_import_location(root: Path) -> tuple[str, str]:
  """Refuse unless proxide and its extension load from ``root``."""
  import proxide
  import proxide._proxider as ext

  where = (proxide.__file__, ext.__file__)
  bad = [p for p in where if not _inside(p, root)]
  if bad:
    msg = (
      f"refusing: proxide resolves outside {root}: {bad}. Build the extension from this "
      "checkout (uv pip install -e . / uv sync) and run with its .venv."
    )
    raise SystemExit(msg)
  return tuple(str(Path(p).resolve().relative_to(root)) for p in where)  # type: ignore[return-value]


def handbuilt_full_protein():
  """A deterministic two-chain Full Protein with no hetero or insertion codes.

  Covers one-letter and two-letter element names (column-14 rule), a table
  HETATM residue (MSE), an ion that is not in the residue table (ZN, written
  ATOM today), and source B-factors / occupancies.
  """
  from proxide.core.containers import Protein

  rows = [
    # chain, resseq, res_name, atom_name, element
    ("A", 1, "ALA", "N", "N"),
    ("A", 1, "ALA", "CA", "C"),
    ("A", 1, "ALA", "C", "C"),
    ("A", 1, "ALA", "O", "O"),
    ("A", 1, "ALA", "CB", "C"),
    ("A", 2, "MSE", "N", "N"),
    ("A", 2, "MSE", "CA", "C"),
    ("A", 2, "MSE", "SE", "SE"),
    ("B", 7, "GLY", "N", "N"),
    ("B", 7, "GLY", "CA", "C"),
    ("B", 101, "ZN", "ZN", "ZN"),
  ]
  n = len(rows)
  chain_vocab = sorted({r[0] for r in rows})
  coords = np.array(
    [[1.0 + 1.5 * i, -2.25 + 0.5 * i, 3.125 - 0.75 * i] for i in range(n)], dtype=np.float32
  )
  return Protein(
    coordinates=np.zeros((n, 37, 3), dtype=np.float32),
    aatype=np.zeros(n, dtype=np.int8),
    residue_index=np.array([r[1] for r in rows], dtype=np.int32),
    chain_index=np.array([chain_vocab.index(r[0]) for r in rows], dtype=np.int32),
    chain_ids=chain_vocab,
    full_coordinates=coords,
    atom_names=[r[3] for r in rows],
    res_names=[r[2] for r in rows],
    elements=[r[4] for r in rows],
    atom_chain_ids=[r[0] for r in rows],
    atom_res_index=np.array([r[1] for r in rows], dtype=np.int32),
    b_factors=np.array([10.0 + i for i in range(n)], dtype=np.float32),
    occupancies=np.array([1.0] * (n - 1) + [0.5], dtype=np.float32),
  )


def load(source: str):
  from proxide import CoordFormat, OutputSpec, parse_structure

  kind, layout = source.split(":")
  if kind == "handbuilt":
    return handbuilt_full_protein()
  fmt = {"Atom37": CoordFormat.Atom37, "Atom14": CoordFormat.Atom14, "Full": CoordFormat.Full}[
    layout
  ]
  return parse_structure(str(SOURCE_PDB), OutputSpec(coord_format=fmt))


def git_state(root: Path) -> tuple[str, bool]:
  head = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
  ).stdout.strip()
  dirty = bool(
    subprocess.run(
      ["git", "status", "--porcelain", "--", "src", "crates"],
      cwd=root,
      capture_output=True,
      text=True,
      check=True,
    ).stdout.strip()
  )
  return head, dirty


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
  parser.add_argument("--force", action="store_true", help="overwrite existing goldens")
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

  pkg_path, ext_path = check_import_location(REPO_ROOT)
  head, dirty = git_state(REPO_ROOT)
  if dirty:
    raise SystemExit("refusing: src/ or crates/ has uncommitted changes; goldens must pin committed code")

  out_dir = args.out_dir.resolve()
  existing = [name for name, _, _ in GOLDENS if (out_dir / name).exists()]
  if existing and not args.force:
    raise SystemExit(f"refusing to overwrite existing goldens without --force: {existing}")
  out_dir.mkdir(parents=True, exist_ok=True)

  from proxide.io.writing import write_mmcif, write_pdb

  writers = {"pdb": write_pdb, "mmcif": write_mmcif}
  cache: dict[str, object] = {}
  digests: list[tuple[str, str, str, str]] = []
  for name, source, writer in GOLDENS:
    protein = cache.setdefault(source, load(source))
    path = writers[writer](protein, out_dir / name)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    digests.append((name, source, writer, digest))
    logger.info("wrote %s (%s, %s) sha256=%s", name, source, writer, digest)

  readme = out_dir / "README.md"
  lines = [
    "# Writer golden files",
    "",
    "Generated by `scripts/golden/make_writer_goldens.py` (spec",
    "`.praxia/docs/specs/261123_pdb-writer-icode-hetero.md`, T0) from unmodified writer code.",
    "Do not regenerate after the writer change: T4 compares the new output against these bytes.",
    "",
    f"- generating commit: `{head}` (src/ and crates/ clean)",
    f"- `proxide` resolved to: `{pkg_path}` (inside the generating checkout)",
    f"- `proxide._proxider` resolved to: `{ext_path}` (inside the generating checkout)",
    f"- source structure: `tests/data/5awl.pdb`; hand-built protein: `handbuilt_full_protein()` in the script",
    "",
    "| file | source | writer | sha256 |",
    "|---|---|---|---|",
    *[f"| `{n}` | {s} | {w} | `{d}` |" for n, s, w, d in digests],
    "",
  ]
  readme.write_text("\n".join(lines))
  logger.info("wrote %s", readme)
  return 0


if __name__ == "__main__":
  sys.exit(main())
