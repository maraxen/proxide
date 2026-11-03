"""Fail-loud checks for the confind black-box parity harness.

These exercise the harness functions directly. No reference binary is required.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validation" / "confind_blackbox_parity.py"

ENV_REFERENCE_BIN = "CONFIND_PARITY_REFERENCE_BIN"
ENV_REFERENCE_ROTLIB = "CONFIND_PARITY_REFERENCE_ROTLIB"
ENV_CLEAN_BIN = "CONFIND_PARITY_CLEAN_BIN"
ENV_CLEAN_ROTLIB = "CONFIND_PARITY_CLEAN_ROTLIB"
ENV_PDB_DIR = "CONFIND_PARITY_PDB_DIR"
ENV_SCRATCH = "CONFIND_PARITY_SCRATCH"

PDB_NAMES = (
  "1CRN.pdb",
  "ubiquitin_1ubq_A.pdb",
  "f2c006r13.pdb",
  "chain_break.pdb",
)

CONTACT_LINE = "contact\tA,1\tA,2\t0.25\tALA\tALA\n"


def _load():
  spec = importlib.util.spec_from_file_location("confind_blackbox_parity", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


harness = _load()


def _matched_contacts():
  upstream = {("A,1", "A,2"): 0.20, ("A,2", "A,3"): 0.50}
  clean = {("A,1", "A,2"): 0.20, ("A,2", "A,3"): 0.40}
  return upstream, clean


def _quiet_outputs():
  contacts = {("A,1", "A,2"): 0.10}
  metrics = {"A,1": 0.20}
  dihedrals = {"A,1": (1.0, 2.0, 180.0)}
  return contacts, metrics, metrics, dihedrals, contacts, metrics, metrics


def _install_benchmark_stubs(monkeypatch, contacts_clean=None, omega=(4, 0, 0.0), prevented=True):
  base = _quiet_outputs()

  def fake_run(_pdb, _tag, extra_flags=None):
    if contacts_clean is None:
      return base
    contacts_up, metrics, _, dihedrals, _, _, _ = base
    return contacts_up, metrics, metrics, dihedrals, contacts_clean, metrics, metrics

  monkeypatch.setattr(harness, "run_benchmark", fake_run)
  monkeypatch.setattr(harness, "assess_omega_sign", lambda _pdb, _dih: omega)
  monkeypatch.setattr(harness, "assess_chain_semantics", lambda: (prevented, 0.2, 0.0))
  monkeypatch.delenv("BTH_RESULTS_PATH", raising=False)


def _provision_locations(monkeypatch, tmp_path):
  scratch = tmp_path / "scratch"
  scratch.mkdir()
  pdb_dir = tmp_path / "pdbs"
  pdb_dir.mkdir()
  for name in PDB_NAMES:
    (pdb_dir / name).write_text("END\n")
  reference_bin = tmp_path / "testConFind"
  reference_rotlib = tmp_path / "rotlib.bin"
  clean_bin = tmp_path / "confind"
  clean_rotlib = tmp_path / "clean.pb.zst"
  for path in (reference_bin, reference_rotlib, clean_bin, clean_rotlib):
    path.write_text("x")
  monkeypatch.setenv(ENV_REFERENCE_BIN, str(reference_bin))
  monkeypatch.setenv(ENV_REFERENCE_ROTLIB, str(reference_rotlib))
  monkeypatch.setenv(ENV_CLEAN_BIN, str(clean_bin))
  monkeypatch.setenv(ENV_CLEAN_ROTLIB, str(clean_rotlib))
  monkeypatch.setenv(ENV_PDB_DIR, str(pdb_dir))
  monkeypatch.setenv(ENV_SCRATCH, str(scratch))
  return {
    "scratch": scratch,
    "pdb_dir": pdb_dir,
    "reference_bin": reference_bin,
    "reference_rotlib": reference_rotlib,
    "clean_bin": clean_bin,
    "clean_rotlib": clean_rotlib,
  }


def _point_run_at(monkeypatch, locations, pdb):
  monkeypatch.setattr(harness, "SCRATCH", locations["scratch"])
  monkeypatch.setattr(harness, "UPSTREAM_BIN", locations["reference_bin"])
  monkeypatch.setattr(harness, "UPSTREAM_RLIB", locations["reference_rotlib"])
  monkeypatch.setattr(harness, "CLEAN_BIN", locations["clean_bin"])
  monkeypatch.setattr(harness, "CLEAN_RLIB", locations["clean_rotlib"])
  calls = []

  def fake_run(cmd, **_kwargs):
    calls.append(list(cmd))
    out = Path(cmd[cmd.index("--o") + 1])
    out.write_text(CONTACT_LINE)
    return subprocess.CompletedProcess(cmd, 0)

  monkeypatch.setattr(harness.subprocess, "run", fake_run)
  return calls


def test_matched_finite_contacts_keep_the_indexed_median():
  upstream, clean = _matched_contacts()
  median, mean, maximum, pct_1e4, pct_005, n_pairs, n_common = harness.compute_contact_deltas(
    upstream, clean
  )
  # 0.50 - 0.40 is not a binary 0.10; the indexed median is deltas[n // 2].
  assert median == pytest.approx(0.10)
  assert mean == pytest.approx(0.05)
  assert maximum == pytest.approx(0.10)
  assert pct_1e4 == 0.5
  assert pct_005 == 0.5
  assert n_pairs == 2
  assert n_common == 2


def test_empty_contact_sets_are_rejected():
  with pytest.raises(ValueError):
    harness.compute_contact_deltas({}, {})


def test_one_sided_contact_is_not_filled_with_zero():
  with pytest.raises(ValueError):
    harness.compute_contact_deltas(
      {("A,1", "A,2"): 0.20, ("A,3", "A,4"): 0.50},
      {("A,1", "A,2"): 0.20},
    )


def test_non_finite_contact_delta_is_rejected():
  with pytest.raises(ValueError):
    harness.compute_contact_deltas(
      {("A,1", "A,2"): float("nan")},
      {("A,1", "A,2"): 0.10},
    )


def test_parse_rejects_nan_and_inf(tmp_path):
  nan_path = tmp_path / "nan.out"
  nan_path.write_text("contact\tA,1\tA,2\tnan\tALA\tALA\n")
  inf_path = tmp_path / "inf.out"
  inf_path.write_text("freedom\tA,1\tinf\t1\t2\t3\tALA\n")
  with pytest.raises(ValueError):
    harness.parse_confind_output(nan_path)
  with pytest.raises(ValueError):
    harness.parse_confind_output(inf_path, is_upstream=True)


def test_empty_metric_intersection_is_rejected():
  with pytest.raises(ValueError):
    harness.compute_metric_deltas({}, {})


def test_unequal_metric_keys_are_rejected():
  with pytest.raises(ValueError):
    harness.compute_metric_deltas({"A,1": 0.20, "A,2": 0.30}, {"A,1": 0.20})


def test_non_finite_metric_is_rejected():
  with pytest.raises(ValueError):
    harness.compute_metric_deltas({"A,1": float("nan")}, {"A,1": 0.10})


def test_empty_omega_map_is_rejected(tmp_path):
  pdb = tmp_path / "empty.pdb"
  pdb.write_text("")
  with pytest.raises(ValueError, match="omega_same_sign_count"):
    harness.assess_omega_sign(pdb, {})


def test_zero_byte_upstream_cache_is_regenerated(monkeypatch, tmp_path):
  locations = _provision_locations(monkeypatch, tmp_path)
  cache = locations["scratch"] / "tag_upstream.out"
  cache.write_bytes(b"")
  pdb = tmp_path / "in.pdb"
  pdb.write_text("END\n")
  calls = _point_run_at(monkeypatch, locations, pdb)
  harness.run_benchmark(pdb, "tag")
  upstream = [cmd for cmd in calls if cmd[0] == str(locations["reference_bin"])]
  assert upstream


def test_cache_older_than_the_binary_is_regenerated(monkeypatch, tmp_path):
  locations = _provision_locations(monkeypatch, tmp_path)
  cache = locations["scratch"] / "tag_upstream.out"
  cache.write_text(CONTACT_LINE)
  pdb = tmp_path / "in.pdb"
  pdb.write_text("END\n")
  calls = _point_run_at(monkeypatch, locations, pdb)
  binary = locations["reference_bin"]
  os.utime(cache, (50, 50))
  os.utime(binary, (100, 100))
  harness.run_benchmark(pdb, "tag")
  upstream = [cmd for cmd in calls if cmd[0] == str(binary)]
  assert upstream


def test_fresh_parseable_cache_is_reused(monkeypatch, tmp_path):
  locations = _provision_locations(monkeypatch, tmp_path)
  cache = locations["scratch"] / "tag_upstream.out"
  cache.write_text(CONTACT_LINE)
  pdb = tmp_path / "in.pdb"
  pdb.write_text("END\n")
  calls = _point_run_at(monkeypatch, locations, pdb)
  binary = locations["reference_bin"]
  os.utime(binary, (100, 100))
  os.utime(cache, (200, 200))
  harness.run_benchmark(pdb, "tag")
  upstream = [cmd for cmd in calls if cmd[0] == str(binary)]
  assert upstream == []


def test_missing_reference_binary_names_the_argument(monkeypatch, tmp_path):
  locations = _provision_locations(monkeypatch, tmp_path)
  monkeypatch.setattr(harness, "SCRATCH", locations["scratch"])
  monkeypatch.setattr(harness, "UPSTREAM_BIN", tmp_path / "missing-testConFind")
  monkeypatch.setattr(harness, "UPSTREAM_RLIB", locations["reference_rotlib"])
  monkeypatch.setattr(harness, "CLEAN_BIN", locations["clean_bin"])
  monkeypatch.setattr(harness, "CLEAN_RLIB", locations["clean_rotlib"])
  pdb = tmp_path / "in.pdb"
  pdb.write_text("END\n")
  with pytest.raises(FileNotFoundError, match="--reference-bin"):
    harness.run_benchmark(pdb, "tag")


def test_main_returns_nonzero_and_writes_json_on_fail(monkeypatch, tmp_path):
  _provision_locations(monkeypatch, tmp_path)
  _install_benchmark_stubs(
    monkeypatch,
    contacts_clean={("A,1", "A,2"): 0.0},
    omega=(4, 1, 20.0),
    prevented=False,
  )
  out = tmp_path / "results.json"
  monkeypatch.setattr(sys, "argv", ["confind_blackbox_parity", "--out", str(out)])
  code = harness.main()
  assert code == 1
  payload = json.loads(out.read_text())
  assert payload["outcome"] == "fail"
  assert payload["median_contact_delta_1crn"] == pytest.approx(0.1)


def test_main_returns_zero_on_pass(monkeypatch, tmp_path):
  _provision_locations(monkeypatch, tmp_path)
  _install_benchmark_stubs(monkeypatch, omega=(4, 0, 0.0), prevented=True)
  out = tmp_path / "pass.json"
  monkeypatch.setattr(sys, "argv", ["confind_blackbox_parity", "--out", str(out)])
  code = harness.main()
  assert code == 0
  payload = json.loads(out.read_text())
  assert payload["outcome"] == "pass"


def test_main_names_a_missing_required_location(monkeypatch, tmp_path, capsys):
  for name in (
    ENV_REFERENCE_BIN,
    ENV_REFERENCE_ROTLIB,
    ENV_CLEAN_BIN,
    ENV_CLEAN_ROTLIB,
    ENV_PDB_DIR,
    ENV_SCRATCH,
  ):
    monkeypatch.delenv(name, raising=False)
  _install_benchmark_stubs(monkeypatch)
  out = tmp_path / "missing.json"
  monkeypatch.setattr(sys, "argv", ["confind_blackbox_parity", "--out", str(out)])
  with pytest.raises(SystemExit) as caught:
    harness.main()
  assert caught.value.code not in (0, None)
  err = capsys.readouterr().err
  assert "--reference-bin" in err
  assert ENV_REFERENCE_BIN in err
