"""Exercise the stable surface of an installed proxide build.

publish.yml runs this against every built wheel and the sdist, from outside the
checkout, before anything can be uploaded. CI's own tests build from a tree that
has the GPL GAFF2 DEF fetched into it, so they cannot see what a user installs.

Usage:
    python scripts/smoke_wheel.py --xtc tests/data/trajectories/frame0.xtc

Exits non-zero on the first failed check. Run it from a directory other than
the repo root, or the check that the installed package is under test fails.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np

import proxide
from proxide import _proxider
from proxide.chem.gaff2 import Gaff2DefMissingError, load_gaff2_rules

log = logging.getLogger("smoke_wheel")

# One binding per stable feature. A wheel built without the feature lacks it, and
# the README promises these on every platform.
STABLE_FEATURE_BINDINGS = {
  "xtc": ("parse_xtc", "read_xtc_lazy", "frame_count", "n_atoms"),
  "hdf5": ("parse_mdtraj_h5_metadata", "parse_mdtraj_h5_frame"),
  "foldcomp": ("parse_foldcomp",),
  "fetching": ("fetch_rcsb", "fetch_afdb", "fetch_foldcomp_database"),
}

PDB = "".join(
  f"ATOM  {i + 1:>5} {name:<4} ALA A{1:>4}    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{b:6.2f}"
  f"{'':10}{element:>2}\n"
  for i, (name, x, y, z, b, element) in enumerate(
    [
      (" N", 11.104, 6.134, -6.504, 10.0, "N"),
      (" CA", 11.639, 6.071, -5.147, 11.0, "C"),
      (" C", 10.504, 6.071, -4.123, 12.0, "C"),
      (" O", 9.340, 6.100, -4.500, 13.0, "O"),
      (" CB", 12.500, 4.820, -4.980, 14.0, "C"),
    ]
  )
)


def check_installed_package_is_under_test() -> None:
  """Fail if the import resolved to a source tree instead of the installed build."""
  location = Path(proxide.__file__).resolve()
  if "site-packages" not in location.parts:
    raise AssertionError(f"testing {location}, not an installed package")
  log.info("proxide %s from %s", proxide.__version__, location.parent)


def check_stable_features_are_linked() -> None:
  """Fail if a stable feature was compiled out of this build."""
  missing = {
    feature: [name for name in names if not hasattr(_proxider, name)]
    for feature, names in STABLE_FEATURE_BINDINGS.items()
  }
  missing = {feature: names for feature, names in missing.items() if names}
  if missing:
    raise AssertionError(f"stable features missing from this build: {missing}")
  log.info("linked features: %s", sorted(STABLE_FEATURE_BINDINGS))


def check_structure_round_trip(workdir: Path) -> None:
  """Parse a PDB, write PDB and mmCIF, and read both back with the same values."""
  source = workdir / "source.pdb"
  source.write_text(PDB + "END\n")
  spec = proxide.OutputSpec(coord_format=proxide.CoordFormat.Full)
  original = proxide.parse_structure(str(source), spec)

  for suffix, write in ((".pdb", proxide.write_pdb), (".cif", proxide.write_mmcif)):
    written = write(original, workdir / f"written{suffix}")
    again = proxide.parse_structure(str(written), spec)
    np.testing.assert_allclose(
      np.asarray(again.coordinates), np.asarray(original.coordinates), atol=1e-3
    )
    assert list(again.elements) == list(original.elements), suffix
    np.testing.assert_allclose(np.asarray(again.b_factors), np.asarray(original.b_factors))
  log.info("round trip ok: %d atoms", len(original.elements))


def check_malformed_input_raises(workdir: Path) -> None:
  """Fail if a parser accepts a broken record or guesses an element."""
  for name, text in {
    "short": "ATOM      1  N\n",
    "bad_coord": PDB.replace("  11.104", " garbage", 1),
  }.items():
    path = workdir / f"{name}.pdb"
    path.write_text(text)
    try:
      proxide.parse_structure(str(path))
    except ValueError as error:
      assert "PROX-IO-MALFORMED-RECORD" in str(error), error
    else:
      raise AssertionError(f"{name} parsed without an error")
  log.info("malformed input raises")


def check_xtc_read(xtc: Path) -> None:
  """Read a real XTC frame through the bound reader."""
  frames = proxide.frame_count(str(xtc))
  assert frames > 0, frames
  coordinates = np.asarray(proxide.parse_xtc(str(xtc))["coordinates"])
  assert coordinates.shape[0] == frames, (coordinates.shape, frames)
  assert np.isfinite(coordinates).all()
  log.info("xtc ok: %d frames", frames)


def check_gaff2_def_is_not_bundled() -> None:
  """The GPL DEF stays out of MIT artifacts, and its absence fails loudly."""
  try:
    load_gaff2_rules()
  except Gaff2DefMissingError:
    log.info("GAFF2 DEF absent, and it raises")
  else:
    raise AssertionError("ATOMTYPE_GFF2.DEF is present in a distributed build")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--xtc", type=Path, required=True, help="XTC fixture to read")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

  check_installed_package_is_under_test()
  check_stable_features_are_linked()
  with tempfile.TemporaryDirectory() as name:
    workdir = Path(name)
    check_structure_round_trip(workdir)
    check_malformed_input_raises(workdir)
  check_xtc_read(args.xtc.resolve())
  check_gaff2_def_is_not_bundled()
  log.info("all checks passed")


if __name__ == "__main__":
  main()
