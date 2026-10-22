"""The format cache must key on file content, not just the path (debt #2372).

Before the fix, the cache was keyed on (path, format, flags) only, so overwriting
a structure file at the same path made parse_structure return the OLD parse.
"""

import logging
import os
from pathlib import Path

import numpy as np

from proxide import CoordFormat, OutputSpec, parse_structure

DATA = Path("tests/data")
CACHE_HIT = "Cache hit for"


def _cached_spec() -> OutputSpec:
  return OutputSpec(coord_format=CoordFormat.Atom37, enable_caching=True)


def test_overwritten_file_at_same_path_is_reparsed(tmp_path, caplog) -> None:
  path = tmp_path / "structure.pdb"
  path.write_text((DATA / "5awl.pdb").read_text())
  first = parse_structure(str(path), _cached_spec())
  assert np.shape(first.coordinates) == (10, 37, 3)

  path.write_text((DATA / "2ala.pdb").read_text())
  with caplog.at_level(logging.DEBUG):
    second = parse_structure(str(path), _cached_spec())
  assert np.shape(second.coordinates) == (2, 37, 3)
  assert np.shape(second.atom_mask) == (2, 37)
  assert CACHE_HIT not in caplog.text


def test_same_size_overwrite_with_same_mtime_is_reparsed(tmp_path) -> None:
  # Length and mtime alone cannot see this edit: the content hash must.
  src = (DATA / "5awl.pdb").read_text()
  shifted = "".join(
    (x[:30] + f"{float(x[30:38]) + 50.0:8.3f}" + x[38:]) if x.startswith("ATOM") else x
    for x in src.splitlines(keepends=True)
  )
  assert len(shifted) == len(src)
  path = tmp_path / "same_size.pdb"
  path.write_text(src)
  stat = os.stat(path)
  first = parse_structure(str(path), _cached_spec())

  path.write_text(shifted)
  os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
  second = parse_structure(str(path), _cached_spec())
  fresh = parse_structure(str(path), OutputSpec(coord_format=CoordFormat.Atom37))
  np.testing.assert_allclose(np.asarray(second.coordinates), np.asarray(fresh.coordinates))
  assert not np.allclose(np.asarray(second.coordinates), np.asarray(first.coordinates))


def test_unchanged_file_is_still_a_cache_hit(tmp_path, caplog) -> None:
  path = tmp_path / "unchanged.pdb"
  path.write_text((DATA / "2ala.pdb").read_text())
  first = parse_structure(str(path), _cached_spec())
  with caplog.at_level(logging.DEBUG):
    second = parse_structure(str(path), _cached_spec())
  assert CACHE_HIT in caplog.text
  np.testing.assert_allclose(np.asarray(second.coordinates), np.asarray(first.coordinates))


def test_option_change_is_not_served_from_another_options_entry(tmp_path) -> None:
  path = tmp_path / "options.pdb"
  path.write_text((DATA / "5awl.pdb").read_text())
  atom37 = parse_structure(str(path), _cached_spec())
  assert np.shape(atom37.coordinates) == (10, 37, 3)

  full_spec = OutputSpec(coord_format=CoordFormat.Full, enable_caching=True)
  full = parse_structure(str(path), full_spec)
  assert np.shape(full.coordinates) != np.shape(atom37.coordinates)

  again = parse_structure(str(path), _cached_spec())
  assert np.shape(again.coordinates) == (10, 37, 3)
  np.testing.assert_allclose(np.asarray(again.coordinates), np.asarray(atom37.coordinates))
