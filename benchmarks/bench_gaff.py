"""Benchmark AmberTools-backed GAFF parameterization.

This benchmark is optional and requires AmberTools plus a built proxide
extension. It writes JSON metrics and a small readable table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

PANEL: dict[str, str] = {
  "methane": "C",
  "ethane": "CC",
  "ethanol": "CCO",
  "benzene": "c1ccccc1",
  "phenol": "c1ccc(cc1)O",
  "acetone": "CC(=O)C",
  "acetic_acid": "CC(=O)O",
  "methylamine": "CN",
  "dimethyl_ether": "COC",
  "chloromethane": "CCl",
  "fluorobenzene": "c1ccc(cc1)F",
  "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
  "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
}


def _write_panel_sdf(directory: Path) -> list[Path]:
  try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
  except ImportError as exc:
    raise SystemExit("RDKit is required to generate the benchmark molecule panel") from exc

  directory.mkdir(parents=True, exist_ok=True)
  paths: list[Path] = []
  for name, smiles in PANEL.items():
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
      raise RuntimeError(f"RDKit embedding failed for {name}")
    AllChem.MMFFOptimizeMolecule(mol)
    path = directory / f"{name}.sdf"
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    paths.append(path)
  return paths


def _summarize(values: list[float]) -> dict[str, float]:
  ordered = sorted(values)
  p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
  total = sum(values)
  return {
    "median_s": statistics.median(values),
    "p95_s": ordered[p95_idx],
    "molecules_per_s": len(values) / total if total else 0.0,
  }


def _run_once(paths: list[Path], version: str, cache: Path | None) -> list[dict[str, Any]]:
  from proxide import parameterize_gaff

  rows: list[dict[str, Any]] = []
  for path in paths:
    start = time.perf_counter()
    template = parameterize_gaff(
      str(path),
      version=version,
      net_charge=0,
      input_format=path.suffix.lstrip("."),
      cache=str(cache / path.stem) if cache is not None else None,
    )
    elapsed = time.perf_counter() - start
    rows.append(
      {
        "name": path.stem,
        "seconds": elapsed,
        "atoms": len(template["atom_types"]),
        "bonds": len(template["bonds"]),
      },
    )
  return rows


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-dir", type=Path, help="Directory containing SDF/MOL2/PDB files")
  parser.add_argument("--version", default="gaff-2.2.20")
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--json-out", type=Path, default=Path("gaff_benchmark.json"))
  args = parser.parse_args()

  with tempfile.TemporaryDirectory(prefix="proxide-gaff-bench-") as tmp:
    tmp_path = Path(tmp)
    paths = (
      sorted(args.input_dir.glob("*")) if args.input_dir else _write_panel_sdf(tmp_path / "panel")
    )
    paths = [p for p in paths if p.suffix.lower() in {".sdf", ".mol2", ".pdb"}]
    if not paths:
      raise SystemExit("No SDF/MOL2/PDB inputs found")

    runs: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
      cache = tmp_path / "cache" / str(repeat)
      cache.mkdir(parents=True, exist_ok=True)
      rows = _run_once(paths, args.version, cache)
      runs.append(
        {
          "repeat": repeat,
          "rows": rows,
          "summary": _summarize([r["seconds"] for r in rows]),
        },
      )

  payload = {
    "version": args.version,
    "molecules": [p.stem for p in paths],
    "runs": runs,
  }
  args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

  latest = runs[-1]
  print("molecule seconds atoms bonds")
  for row in latest["rows"]:
    print(f"{row['name']} {row['seconds']:.4f} {row['atoms']} {row['bonds']}")
  summary = latest["summary"]
  print(
    f"median={summary['median_s']:.4f}s "
    f"p95={summary['p95_s']:.4f}s "
    f"throughput={summary['molecules_per_s']:.2f} molecules/s",
  )
  print(f"wrote {args.json_out}")


if __name__ == "__main__":
  main()
