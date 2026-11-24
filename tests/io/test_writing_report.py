"""Writer report, logging and provenance sidecar (spec 261123, T5; decisions D4 and D5).

Covers, for `write_pdb`/`write_mmcif` and their `_reported` twins:

- a must-fire and a must-not-fire protein for each warning code (HETERO-IGNORED, ICODE-BLANKED,
  ALIAS-ATOMS-MISSING), with every warning attributed to this file through the public entry point;
- the info codes (HETERO-UNKNOWN, ICODE-UNKNOWN) present and never warned;
- ``report.sha256`` equal to the sha256 of the bytes on disk;
- exactly one ``proxide.io.writing`` INFO record per write;
- ``provenance="none"`` writes one file; ``"sidecar"`` writes two, the sidecar parses as TOML with
  a matching ``output_sha256``; a sidecar failure (a directory at ``<path>.prov.toml``) raises and
  leaves the output intact;
- an adversarial ``protein.source`` round-trips through ``tomllib`` exactly.

The fixture is ``tests/data/icode_ligand.pdb`` (T1): residues 52, 52A and 52B, one MSE, one ZN, one
4-atom ligand and two HOH. Its HETATM table is {MSE}, so the seven non-MSE hetero rows (ZN, the
ligand, the waters) are the ones ``record_type="residue_table"`` drops to ATOM.
"""

from __future__ import annotations

import hashlib
import logging
import tomllib
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from proxide import CoordFormat, OutputSpec, parse_structure
from proxide.chem.residues import atom_order, resnames
from proxide.core.containers import Protein
from proxide.io.writing import WriteReport, write_mmcif, write_mmcif_reported, write_pdb, write_pdb_reported

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests" / "data"
FIXTURE = DATA / "icode_ligand.pdb"
ALT_PROBE = DATA / "mse_altloc.pdb"  # ALA, MSE with altlocs on SE, GLY (see test_atom14_variants)

ENTRY_POINTS: dict[str, Callable] = {
  "write_pdb": write_pdb,
  "write_pdb_reported": write_pdb_reported,
  "write_mmcif": write_mmcif,
  "write_mmcif_reported": write_mmcif_reported,
}
ADVERSARIAL_SOURCE = 'a"b\\c\nd é\x01'  # quote, backslash, newline, non-ASCII, a control


def _parse(path: Path, layout: CoordFormat, **kwargs) -> Protein:
  return parse_structure(path, OutputSpec(coord_format=layout, **kwargs), use_jax=False)


@pytest.fixture(scope="module")
def t1_full() -> Protein:
  return _parse(FIXTURE, CoordFormat.Full)


def _full(
  res_names: list[str],
  *,
  hetero=None,
  insertion_codes=None,
  source: str | None = None,
) -> Protein:
  """A hand-built Full Protein with one CA atom per entry of ``res_names``."""
  n = len(res_names)
  return Protein(
    coordinates=np.zeros((n, 37, 3), dtype=np.float32),
    aatype=np.zeros(n, dtype=np.int8),
    residue_index=np.arange(n, dtype=np.int32),
    chain_index=np.zeros(n, dtype=np.int32),
    chain_ids=["A"],
    full_coordinates=np.stack([np.arange(n, dtype=np.float32) * 1.5, np.zeros(n), np.zeros(n)], -1),
    atom_names=["CA"] * n,
    res_names=list(res_names),
    elements=["C"] * n,
    hetero=None if hetero is None else np.asarray(hetero, dtype=np.bool_),
    insertion_codes=insertion_codes,
    source=source,
  )


def _atom37(res_names: list[str]) -> Protein:
  """A hand-built Atom37 Protein, one resolved CA per residue, named by ``res_names``."""
  n = len(res_names)
  mask = np.zeros((n, 37), dtype=np.float32)
  mask[:, atom_order["CA"]] = 1.0
  coords = np.zeros((n, 37, 3), dtype=np.float32)
  coords[:, atom_order["CA"], 0] = np.arange(n) * 1.5
  aatype = np.array([resnames.index("MET" if r == "MSE" else r) for r in res_names], np.int8)
  return Protein(
    coordinates=coords,
    aatype=aatype,
    residue_index=np.arange(1, n + 1, dtype=np.int32),
    chain_index=np.zeros(n, dtype=np.int32),
    chain_ids=["A"],
    atom_mask=mask,
    residue_names=list(res_names),
  )


def _call(fn, protein: Protein, path: Path, **kwargs):
  """Run one writer call. Returns (result, the UserWarnings that call raised)."""
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = fn(protein, path, **kwargs)
  return result, [w for w in caught if issubclass(w.category, UserWarning)]


def _codes(report: WriteReport) -> list[str]:
  return [f.code for f in report.findings]


def _finding(report: WriteReport, code: str):
  matches = [f for f in report.findings if f.code == code]
  assert len(matches) <= 1, report.findings
  return matches[0] if matches else None


# --- Warning codes: must fire and must not fire ----------------------------------------------------


def _hetero_ignored_protein() -> Protein:
  return _full(["LIG", "GLY"], hetero=[True, False])


def _hetero_ignored_quiet_protein() -> Protein:
  # Table residue under residue_table (fine), and a hetero row under "source" (honoured, no warning).
  return _full(["MSE", "GLY"], hetero=[True, False])


def _icode_blanked_protein() -> Protein:
  return _full(["ALA", "GLY"], insertion_codes=["A", " "])


def _alias_probe(typing: str | None) -> Protein:
  """The MSE probe parsed with ``modified_residue_typing=typing``; None keeps the default."""
  options = {} if typing is None else {"modified_residue_typing": typing}
  return parse_structure(
    ALT_PROBE, OutputSpec(coord_format=CoordFormat.Atom37, **options), use_jax=False
  )


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_hetero_ignored_fires_and_is_attributed_to_the_caller(tmp_path: Path, entry: str) -> None:
  fn = ENTRY_POINTS[entry]
  _, warned = _call(fn, _hetero_ignored_protein(), tmp_path / "h.pdb", record_type="residue_table")
  assert len(warned) == 1
  assert "ignores hetero=True" in str(warned[0].message)
  assert warned[0].filename == __file__


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_hetero_ignored_does_not_fire_on_table_residues_or_under_source(tmp_path: Path, entry: str) -> None:
  fn = ENTRY_POINTS[entry]
  _, quiet_table = _call(
    fn, _hetero_ignored_quiet_protein(), tmp_path / "q.pdb", record_type="residue_table"
  )
  _, quiet_source = _call(fn, _hetero_ignored_protein(), tmp_path / "s.pdb", record_type="source")
  assert quiet_table == []
  assert quiet_source == []


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_icode_blanked_fires_and_is_attributed_to_the_caller(tmp_path: Path, entry: str) -> None:
  fn = ENTRY_POINTS[entry]
  _, warned = _call(fn, _icode_blanked_protein(), tmp_path / "b.pdb", insertion_codes="blank")
  assert len(warned) == 1
  assert "insertion-code column blank" in str(warned[0].message)
  assert warned[0].filename == __file__


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_icode_blanked_does_not_fire_when_codes_are_written_or_all_blank(tmp_path: Path, entry: str) -> None:
  fn = ENTRY_POINTS[entry]
  _, written = _call(fn, _icode_blanked_protein(), tmp_path / "w.pdb", insertion_codes="write")
  all_blank = _full(["ALA", "GLY"], insertion_codes=[" ", " "])
  _, blank_only = _call(fn, all_blank, tmp_path / "x.pdb", insertion_codes="blank")
  assert written == []
  assert blank_only == []


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_alias_atoms_missing_fires_and_is_attributed_to_the_caller(tmp_path: Path, entry: str) -> None:
  fn = ENTRY_POINTS[entry]
  _, warned = _call(fn, _alias_probe("unknown"), tmp_path / "a.pdb")
  assert len(warned) == 1
  assert "alias atoms" in str(warned[0].message)
  assert warned[0].filename == __file__


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_alias_atoms_missing_does_not_fire_for_default_typing(tmp_path: Path, entry: str) -> None:
  fn = ENTRY_POINTS[entry]
  _, warned = _call(fn, _alias_probe(None), tmp_path / "d.pdb")
  assert warned == []


# --- Info codes: present, never warned -------------------------------------------------------------


def test_info_codes_are_present_and_never_warn(tmp_path: Path) -> None:
  protein = _full(["ALA", "GLY"])  # hetero None and insertion_codes None: both info codes
  for name, fn in ENTRY_POINTS.items():
    if name.endswith("_reported"):
      report, warned = _call(fn, protein, tmp_path / f"{name}.out")
      assert warned == [], name
      hetero = _finding(report, "PROX-WRITE-HETERO-UNKNOWN")
      icode = _finding(report, "PROX-WRITE-ICODE-UNKNOWN")
      assert hetero is not None and hetero.severity == "info" and hetero.count == 2
      assert icode is not None and icode.severity == "info" and icode.count == 2
    else:
      _, warned = _call(fn, protein, tmp_path / f"{name}.out")
      assert warned == [], name


def test_info_codes_absent_when_the_protein_carries_the_fields(tmp_path: Path) -> None:
  protein = _full(["ALA", "GLY"], hetero=[False, False], insertion_codes=["A", " "])
  report = write_pdb_reported(protein, tmp_path / "clean.pdb")
  assert "PROX-WRITE-HETERO-UNKNOWN" not in _codes(report)
  assert "PROX-WRITE-ICODE-UNKNOWN" not in _codes(report)


def test_warning_findings_carry_their_severity_and_count(tmp_path: Path, t1_full: Protein) -> None:
  report = write_pdb_reported(t1_full, tmp_path / "t1.pdb", record_type="residue_table")
  ignored = _finding(report, "PROX-WRITE-HETERO-IGNORED")
  assert ignored is not None
  assert ignored.severity == "warning"
  # ZN, the four ligand atoms and two waters carry hetero=True but are not in the residue table.
  assert ignored.count == 7


# --- Report contents -------------------------------------------------------------------------------


@pytest.mark.parametrize(("entry", "fmt"), [("write_pdb_reported", "pdb"), ("write_mmcif_reported", "mmcif")])
def test_report_sha256_matches_the_bytes_on_disk(tmp_path: Path, t1_full: Protein, entry: str, fmt: str) -> None:
  path = tmp_path / f"t1.{fmt}"
  report = ENTRY_POINTS[entry](t1_full, path)
  assert isinstance(report, WriteReport)
  assert report.format == fmt
  assert report.path == path
  assert report.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
  assert report.provenance_path is None


@pytest.mark.parametrize("entry", ["write_pdb", "write_mmcif"])
def test_plain_entry_points_return_the_path_only(tmp_path: Path, entry: str) -> None:
  out = ENTRY_POINTS[entry](_full(["ALA"]), tmp_path / "p.out")
  assert out == tmp_path / "p.out"


def test_report_counts_on_the_fixture(tmp_path: Path, t1_full: Protein) -> None:
  report = write_pdb_reported(t1_full, tmp_path / "t1.pdb")
  rows = [ln for ln in (tmp_path / "t1.pdb").read_text().splitlines() if ln.startswith(("ATOM", "HETATM"))]
  assert report.n_atoms == len(rows) == len(t1_full.atom_names)
  assert report.n_hetatm == sum(r.startswith("HETATM") for r in rows) == 15  # 8 MSE + 7 hetero
  assert report.n_hetatm_from_hetero == 7
  assert report.n_insertion_codes == sum(c != " " for c in t1_full.insertion_codes)
  assert report.options == (("insertion_codes", "write"), ("record_type", "source"))
  assert report.source == t1_full.source
  assert report.proxide_version


def test_residue_table_report_counts_no_hetero_rows_as_hetatm(tmp_path: Path, t1_full: Protein) -> None:
  report = write_pdb_reported(t1_full, tmp_path / "t.pdb", record_type="residue_table")
  assert report.n_hetatm == 8  # the MSE rows only
  assert report.n_hetatm_from_hetero == 0


def test_blank_option_counts_the_codes_it_blanked(tmp_path: Path, t1_full: Protein) -> None:
  report = write_mmcif_reported(t1_full, tmp_path / "b.cif", insertion_codes="blank")
  blanked = _finding(report, "PROX-WRITE-ICODE-BLANKED")
  assert blanked is not None and blanked.count == sum(c != " " for c in t1_full.insertion_codes)
  assert report.n_insertion_codes == 0
  assert "pdbx_PDB_ins_code" not in (tmp_path / "b.cif").read_text()


def test_options_are_validated_before_the_file_is_opened(tmp_path: Path) -> None:
  target = tmp_path / "bad.pdb"
  with pytest.raises(ValueError, match="provenance='always' is not accepted"):
    write_pdb_reported(_full(["ALA"]), target, provenance="always")
  assert not target.exists()


# --- Logging ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("entry", "kwargs"),
  [
    ("write_pdb", {}),
    ("write_mmcif", {}),
    ("write_pdb_reported", {"provenance": "sidecar"}),
    ("write_mmcif_reported", {"provenance": "sidecar", "insertion_codes": "blank"}),
  ],
)
def test_one_info_record_per_write(tmp_path: Path, caplog, t1_full: Protein, entry: str, kwargs: dict) -> None:
  caplog.set_level(logging.INFO, logger="proxide.io.writing")
  before = len([r for r in caplog.records if r.name == "proxide.io.writing"])
  ENTRY_POINTS[entry](t1_full, tmp_path / "log.out", **kwargs)
  records = [r for r in caplog.records if r.name == "proxide.io.writing"][before:]
  assert len(records) == 1
  assert records[0].levelno == logging.INFO


# --- Provenance sidecar ----------------------------------------------------------------------------


def test_provenance_none_writes_exactly_one_file(tmp_path: Path, t1_full: Protein) -> None:
  out_dir = tmp_path / "none"
  out_dir.mkdir()
  report = write_pdb_reported(t1_full, out_dir / "t1.pdb", provenance="none")
  assert [p.name for p in out_dir.iterdir()] == ["t1.pdb"]
  assert report.provenance_path is None


def test_provenance_sidecar_writes_two_files_that_parse(tmp_path: Path, t1_full: Protein) -> None:
  out_dir = tmp_path / "side"
  out_dir.mkdir()
  report = write_mmcif_reported(t1_full, out_dir / "t1.cif", provenance="sidecar")
  assert sorted(p.name for p in out_dir.iterdir()) == ["t1.cif", "t1.cif.prov.toml"]
  assert report.provenance_path == out_dir / "t1.cif.prov.toml"

  prov = tomllib.loads((out_dir / "t1.cif.prov.toml").read_text(encoding="utf-8"))
  assert prov["schema"] == "proxide-write-prov/0"
  block = prov["provenance"]
  assert block["output_sha256"] == hashlib.sha256((out_dir / "t1.cif").read_bytes()).hexdigest()
  assert block["output_sha256"] == report.sha256
  assert Path(block["output_path"]) == (out_dir / "t1.cif").resolve()
  assert block["options"] == {"insertion_codes": "write", "record_type": "source"}
  assert block["counts"] == {
    "n_atoms": report.n_atoms,
    "n_hetatm": report.n_hetatm,
    "n_hetatm_from_hetero": 7,
    "n_insertion_codes": report.n_insertion_codes,
  }
  assert block["input_sha256"] == "unavailable"
  assert block["finding_codes"] == []
  assert block["proxide_install"] in {"editable", "wheel", "vcs", "unknown"}
  assert block["proxide_source_commit"]
  if block["proxide_install"] == "editable":
    assert block["proxide_editable_path"]


def test_provenance_sidecar_lists_finding_codes(tmp_path: Path) -> None:
  protein = _full(["ALA", "GLY"])  # two info findings, no warnings
  report = write_pdb_reported(protein, tmp_path / "f.pdb", provenance="sidecar")
  prov = tomllib.loads((tmp_path / "f.pdb.prov.toml").read_text(encoding="utf-8"))
  assert prov["provenance"]["finding_codes"] == [
    "PROX-WRITE-HETERO-UNKNOWN",
    "PROX-WRITE-ICODE-UNKNOWN",
  ]
  assert report.findings[0].code == "PROX-WRITE-HETERO-UNKNOWN"


@pytest.mark.parametrize("entry", ["write_pdb", "write_mmcif"])
def test_sidecar_failure_raises_and_keeps_the_output(tmp_path: Path, t1_full: Protein, entry: str) -> None:
  # The reference bytes come from a provenance="none" write of the same protein and name.
  fmt = "pdb" if entry == "write_pdb" else "cif"
  reference_dir = tmp_path / "reference"
  reference_dir.mkdir()
  reference = ENTRY_POINTS[entry](t1_full, reference_dir / f"t1.{fmt}").read_bytes()

  out_dir = tmp_path / "fail"
  out_dir.mkdir()
  (out_dir / f"t1.{fmt}.prov.toml").mkdir()  # a directory where the sidecar must go
  with pytest.raises(OSError):
    ENTRY_POINTS[entry](t1_full, out_dir / f"t1.{fmt}", provenance="sidecar")
  assert (out_dir / f"t1.{fmt}").read_bytes() == reference


def test_adversarial_source_round_trips_through_tomllib(tmp_path: Path) -> None:
  protein = _full(["ALA", "GLY"], source=ADVERSARIAL_SOURCE)
  report = write_pdb_reported(protein, tmp_path / "adv.pdb", provenance="sidecar")
  assert report.source == ADVERSARIAL_SOURCE
  prov = tomllib.loads((tmp_path / "adv.pdb.prov.toml").read_text(encoding="utf-8"))
  assert prov["provenance"]["input_source"] == ADVERSARIAL_SOURCE


def test_missing_source_is_recorded_as_unknown(tmp_path: Path) -> None:
  write_pdb(_full(["ALA"]), tmp_path / "n.pdb", provenance="sidecar")
  prov = tomllib.loads((tmp_path / "n.pdb.prov.toml").read_text(encoding="utf-8"))
  assert prov["provenance"]["input_source"] == "unknown"
