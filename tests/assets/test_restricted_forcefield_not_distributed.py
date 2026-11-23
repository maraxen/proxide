"""CGenFF is not redistributable, so proxide must not ship it or pretend to find it.

CGenFF needs a signed agreement from the MacKerell lab (decision D5 in
.praxia/docs/specs/261102_proxide-stable-hardening-and-cisternal.md). Asking for
it by name must raise a named error, not a generic "not found", and a user's own
licensed copy passed by path must still load.
"""

import importlib.util
import shutil
from pathlib import Path

import pytest

import proxide
from proxide.physics.force_fields import (
  FullForceField,
  list_available_force_fields,
  load_force_field,
)
from proxide.physics.force_fields.loader import CgenffNotDistributedError

ASSETS = Path(proxide.__file__).parent / "assets"
REPO = Path(__file__).resolve().parents[2]


def test_assets_tree_has_no_cgenff() -> None:
  assert ASSETS.is_dir(), ASSETS
  assert sorted(p.name for p in ASSETS.rglob("*") if "cgenff" in p.name.lower()) == []


def test_cgenff_not_listed() -> None:
  listed = list_available_force_fields()
  assert listed, "no force fields listed at all; the check below would pass vacuously"
  assert [name for name in listed if "cgenff" in name.lower()] == []


@pytest.mark.parametrize(
  "name",
  [
    "charmm36_cgenff",
    "charmm36_cgenff.xml",
    "charmm/charmm36_cgenff",
    "charmm/charmm36_cgenff.xml",
    "CHARMM36_CGenFF.xml",
  ],
)
def test_cgenff_by_name_raises_named_error(name: str) -> None:
  with pytest.raises(CgenffNotDistributedError) as info:
    load_force_field(name)
  message = str(info.value)
  assert "mackerell.umaryland.edu" in message
  assert "path" in message  # tells the user how to bring their own copy


def test_named_error_is_still_a_not_found_error() -> None:
  # Callers that caught the old generic ValueError or a FileNotFoundError keep working.
  assert issubclass(CgenffNotDistributedError, FileNotFoundError)
  assert issubclass(CgenffNotDistributedError, ValueError)


def test_unrelated_missing_name_keeps_generic_error() -> None:
  with pytest.raises(ValueError) as info:
    load_force_field("no_such_forcefield_xyz")
  assert not isinstance(info.value, CgenffNotDistributedError)


def test_user_supplied_copy_by_path_still_loads(tmp_path: Path) -> None:
  # Stand-in for a user's licensed CGenFF file: any valid ffxml under that name.
  # Distribution is restricted, use with a licence is not.
  own_copy = tmp_path / "charmm36_cgenff.xml"
  shutil.copy(ASSETS / "water" / "tip3p_standard.xml", own_copy)
  assert isinstance(load_force_field(str(own_copy)), FullForceField)


def _load_smoke_wheel():
  spec = importlib.util.spec_from_file_location("smoke_wheel", REPO / "scripts" / "smoke_wheel.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_smoke_check_passes_on_clean_tree(tmp_path: Path) -> None:
  (tmp_path / "assets" / "charmm").mkdir(parents=True)
  (tmp_path / "assets" / "charmm" / "charmm36.xml").write_text("<ForceField/>")
  _load_smoke_wheel().check_cgenff_is_not_bundled(tmp_path)


@pytest.mark.parametrize(
  "planted", ["assets/charmm/charmm36_cgenff.xml", "assets/elsewhere/CGenFF_par.prm"]
)
def test_smoke_check_fails_when_cgenff_is_put_back(tmp_path: Path, planted: str) -> None:
  target = tmp_path / planted
  target.parent.mkdir(parents=True)
  target.write_text("<ForceField/>")
  with pytest.raises(AssertionError, match="cgenff"):
    _load_smoke_wheel().check_cgenff_is_not_bundled(tmp_path)
