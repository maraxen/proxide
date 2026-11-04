"""Fail-loud checks for the confind black-box parity harness and v2 pre-registration (#3217).

These exercise the harness functions directly. No reference binary is required.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validation" / "confind_blackbox_parity.py"
SIDECAR_V1 = ROOT / "scripts" / "validation" / "confind_blackbox_parity.bth.toml"
SIDECAR_V2 = ROOT / "scripts" / "validation" / "confind_blackbox_parity_v2.bth.toml"

ENV_OUT = "CONFIND_PARITY_OUT"
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


_CONTACTS = {("A,1", "A,2"): 0.10}
_FREEDOM = {"A,1": 0.20, "A,2": 0.40}
# Crowdedness per arm, chosen so every by-construction relationship holds:
# C(all) >= C(ala) at fixed crowdmode, crowdmodes differ on A,1, and A,2 is an endpoint.
_ARM_CROWD = {
  "ala-count": {"A,1": 0.20, "A,2": 0.0},
  "all-weighted": {"A,1": 0.30, "A,2": 0.0},
  "all-count": {"A,1": 0.25, "A,2": 0.0},
  "ala-weighted": {"A,1": 0.22, "A,2": 0.0},
}


def _arm_for_flags(extra_flags) -> str:
  flags = list(extra_flags) if extra_flags else []
  cb = flags[flags.index("--cbclash") + 1] if "--cbclash" in flags else "all-residues"
  crowd = flags[flags.index("--crowdmode") + 1] if "--crowdmode" in flags else "probability-weighted"
  policy = "all" if cb == "all-residues" else "ala"
  mode = "weighted" if crowd == "probability-weighted" else "count"
  return f"{policy}-{mode}"


def _install_benchmark_stubs(monkeypatch, contacts_clean=None, omega=(4, 0, 0.0), prevented=True):
  dihedrals = {"A,1": (1.0, 2.0, 180.0), "A,2": (1.0, 2.0, 180.0)}
  clean_contacts = _CONTACTS if contacts_clean is None else contacts_clean

  def fake_run(_pdb, _tag, extra_flags=None):
    # The reference matches the parity arm exactly; the default arm differs in crowdedness only.
    return (
      _CONTACTS,
      _ARM_CROWD["ala-count"],
      _FREEDOM,
      dihedrals,
      clean_contacts,
      _ARM_CROWD[_arm_for_flags(extra_flags)],
      _FREEDOM,
    )

  def fake_clean(_pdb, _tag, extra_flags=None):
    return clean_contacts, _ARM_CROWD[_arm_for_flags(extra_flags)], _FREEDOM

  monkeypatch.setattr(harness, "run_benchmark", fake_run)
  monkeypatch.setattr(harness, "run_clean", fake_clean)
  monkeypatch.setattr(harness, "check_arm_headers", lambda _runs: (0, harness.N_CLEAN_RUNS))
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
  """A one-sided pair is excluded from the deltas (no stand-in degree), not raised on."""
  upstream = {("A,1", "A,2"): 0.20, ("A,3", "A,4"): 0.50}
  clean = {("A,1", "A,2"): 0.20}
  median, mean, maximum, _, _, n_union, n_common = harness.compute_contact_deltas(upstream, clean)
  # If the missing pair were filled with 0.0, max would be 0.50 and the median 0.50.
  assert maximum == 0.0
  assert median == 0.0
  assert mean == 0.0
  assert (n_union, n_common) == (2, 1)
  assert harness.compute_contact_p95_delta(upstream, clean) == 0.0


def test_one_sided_contacts_are_counted_and_bounded_by_present_degree():
  upstream = {("A,1", "A,2"): 0.20, ("A,3", "A,4"): 0.50, ("A,5", "A,6"): 0.0}
  clean = {("A,1", "A,2"): 0.20, ("A,7", "A,8"): 0.03}
  stats = harness.compute_one_sided_contacts(upstream, clean)
  assert stats["ref_only_count"] == 2
  assert stats["clean_only_count"] == 1
  assert stats["one_sided_fraction"] == pytest.approx(3 / 4)
  assert stats["max_ref_only_degree"] == pytest.approx(0.50)
  assert stats["max_clean_only_degree"] == pytest.approx(0.03)
  assert stats["max_one_sided_degree"] == pytest.approx(0.50)


def test_no_one_sided_contacts_reports_zero_counts():
  upstream, clean = _matched_contacts()
  stats = harness.compute_one_sided_contacts(upstream, clean)
  assert stats["ref_only_count"] == 0
  assert stats["clean_only_count"] == 0
  assert stats["one_sided_fraction"] == 0.0
  assert stats["max_one_sided_degree"] == 0.0


def test_disjoint_contact_sets_are_rejected():
  with pytest.raises(ValueError, match="present in both"):
    harness.compute_contact_deltas({("A,1", "A,2"): 0.2}, {("A,3", "A,4"): 0.2})


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
  harness._write_completion_stamp(cache, pdb)
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
  harness._write_completion_stamp(cache, pdb)
  calls = _point_run_at(monkeypatch, locations, pdb)
  binary = locations["reference_bin"]
  os.utime(binary, (100, 100))
  os.utime(cache, (200, 200))
  harness.run_benchmark(pdb, "tag")
  upstream = [cmd for cmd in calls if cmd[0] == str(binary)]
  assert upstream == []


def test_cache_without_completion_stamp_is_regenerated(monkeypatch, tmp_path):
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
  assert upstream
  assert harness._stamp_path(cache).is_file()


def test_cache_with_mismatched_input_sha256_is_regenerated(monkeypatch, tmp_path):
  locations = _provision_locations(monkeypatch, tmp_path)
  cache = locations["scratch"] / "tag_upstream.out"
  cache.write_text(CONTACT_LINE)
  pdb = tmp_path / "in.pdb"
  pdb.write_text("END\n")
  harness._write_completion_stamp(cache, pdb)
  # Mutate the input PDB after the completion stamp was written.
  pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n")
  calls = _point_run_at(monkeypatch, locations, pdb)
  binary = locations["reference_bin"]
  os.utime(binary, (100, 100))
  os.utime(cache, (200, 200))
  harness.run_benchmark(pdb, "tag")
  upstream = [cmd for cmd in calls if cmd[0] == str(binary)]
  assert upstream


def test_clean_cache_reused_on_matching_sha256_and_flags(monkeypatch, tmp_path):
  locations = _provision_locations(monkeypatch, tmp_path)
  pdb = tmp_path / "in.pdb"
  pdb.write_text("END\n")
  calls = _point_run_at(monkeypatch, locations, pdb)
  flags = ["--cbclash", "ala-only", "--crowdmode", "pruned-count-fraction"]
  harness.run_benchmark(pdb, "tag", flags)
  assert len(calls) == 2

  # Advance mtimes on cached outputs so they are strictly newer than the binaries.
  for out_file in locations["scratch"].glob("tag*.out"):
    os.utime(out_file, (500, 500))
  os.utime(locations["reference_bin"], (100, 100))
  os.utime(locations["clean_bin"], (100, 100))

  calls.clear()
  harness.run_benchmark(pdb, "tag", flags)
  assert calls == []
  struct_res = json.loads(
    (
      locations["scratch"]
      / "tag_cbclash_ala-only_crowdmode_pruned-count-fraction_structure_result.json"
    ).read_text()
  )
  assert struct_res["completed"] is True
  assert struct_res["input_sha256"] == harness._file_sha256(pdb)


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


def test_out_path_requires_flag_or_env_and_rejects_bare_tmp(monkeypatch, tmp_path, capsys):
  _provision_locations(monkeypatch, tmp_path)
  _install_benchmark_stubs(monkeypatch)
  monkeypatch.delenv(ENV_OUT, raising=False)

  # Neither --out nor CONFIND_PARITY_OUT provided -> error naming both
  monkeypatch.setattr(sys, "argv", ["confind_blackbox_parity"])
  with pytest.raises(SystemExit) as caught:
    harness.main()
  assert caught.value.code not in (0, None)
  err = capsys.readouterr().err
  assert "--out" in err
  assert ENV_OUT in err

  # Bare /tmp path rejected
  monkeypatch.setattr(
    sys, "argv", ["confind_blackbox_parity", "--out", "/tmp/confind_parity_results.json"]
  )
  with pytest.raises(SystemExit) as caught_tmp:
    harness.main()
  assert caught_tmp.value.code not in (0, None)
  err_tmp = capsys.readouterr().err
  assert "/tmp" in err_tmp

  # Env var fallback works and explicit --out overrides CONFIND_PARITY_OUT
  env_out = tmp_path / "from_env.json"
  arg_out = tmp_path / "from_arg.json"
  monkeypatch.setenv(ENV_OUT, str(env_out))
  monkeypatch.setattr(sys, "argv", ["confind_blackbox_parity"])
  assert harness.main() == 0
  assert env_out.is_file()

  monkeypatch.setattr(sys, "argv", ["confind_blackbox_parity", "--out", str(arg_out)])
  assert harness.main() == 0
  assert arg_out.is_file()


def _schema() -> dict:
  return tomllib.loads(SIDECAR_V2.read_text(encoding="utf-8"))["result_schema"]


def _base_passing_result() -> dict:
  """A result satisfying every term, built from the sidecar schema and the term lists."""
  result = {}
  for field, kind in _schema().items():
    if field == "outcome":
      continue
    if kind == "string":
      result[field] = "label"
    elif kind == "bool":
      result[field] = True
    elif kind == "int":
      result[field] = 1
    else:
      result[field] = 0.0
  for field, op, value in harness.INVARIANT_TERMS:
    if op == "=":
      result[field] = value
  return result


def test_v2_sidecar_conditions_match_harness_terms_exactly():
  parsed = tomllib.loads(SIDECAR_V2.read_text(encoding="utf-8"))
  conditions = harness.render_conditions()
  assert parsed["outcomes"]["pass"]["condition"] == conditions["pass"]
  assert parsed["outcomes"]["marginal"]["condition"] == conditions["marginal"]
  assert parsed["outcomes"]["fail"]["condition"] == conditions["fail"]
  assert parsed["outcomes"]["fail"]["is_residual"] is True


def test_v2_sidecar_records_prior_knowledge_and_drops_vacuous_gates():
  raw = SIDECAR_V2.read_text(encoding="utf-8")
  for needle in ("0.14-0.18", "0.019-0.025", "ea7e91ff", "#3223", "#3224", "PRIOR KNOWLEDGE"):
    assert needle in raw
  schema = _schema()
  assert "default_arm_crowdedness_expected_to_differ_by_design" not in schema
  assert "default_options_valid" not in schema
  conditions = " ".join(harness.render_conditions().values())
  assert "_crowdedness_delta_1crn_default" not in conditions
  assert "median_crowdedness_delta_1ubq_default" not in conditions


def test_every_gate_field_is_in_the_schema():
  schema = _schema()
  terms = harness.INVARIANT_TERMS + harness.TIGHT_TERMS + harness.MARGINAL_TERMS
  assert [t[0] for t in terms if t[0] not in schema] == []


def test_main_payload_matches_the_sidecar_schema(monkeypatch, tmp_path):
  _provision_locations(monkeypatch, tmp_path)
  _install_benchmark_stubs(monkeypatch)
  out = tmp_path / "schema.json"
  monkeypatch.setattr(sys, "argv", ["confind_blackbox_parity", "--out", str(out)])
  assert harness.main() == 0
  payload = json.loads(out.read_text())
  assert set(payload) == set(_schema())


def test_tight_bounds_are_never_looser_than_marginal():
  tight = harness.TIGHT_TERMS
  marginal = harness.MARGINAL_TERMS
  assert [t[0] for t in tight] == [t[0] for t in marginal]
  assert all(a[2] <= b[2] for a, b in zip(tight, marginal, strict=True))


def test_partition_pass_marginal_fail():
  assert harness.evaluate_outcome(_base_passing_result()) == "pass"
  for field, value, expected in (
    ("median_contact_delta_1crn", 0.00015, "marginal"),
    ("median_crowdedness_delta_1crn", 0.045, "marginal"),
    ("p95_freedom_delta_1ubq", 0.20, "marginal"),
    ("max_one_sided_contact_degree_1crn", 0.30, "marginal"),
    ("max_one_sided_contact_degree_multichain", 0.40, "marginal"),
    ("max_crowdedness_delta_1crn", 0.65, "fail"),
    ("max_one_sided_contact_degree_1crn", 0.45, "fail"),
    ("median_contact_delta_1crn_default", 0.00015, "marginal"),
    ("median_contact_delta_1crn_default", 0.00025, "fail"),
  ):
    assert harness.evaluate_outcome({**_base_passing_result(), field: value}) == expected, field


def test_default_arm_crowdedness_is_reported_not_gated():
  result = {**_base_passing_result(), "median_crowdedness_delta_1crn_default": 0.99}
  assert harness.evaluate_outcome(result) == "pass"


@pytest.mark.parametrize("term", harness.INVARIANT_TERMS, ids=lambda t: t[0])
def test_every_invariant_term_can_force_fail(term):
  """No invariant may be vacuous: violating any single term fails, even in the marginal band."""
  field, op, value = term
  if isinstance(value, bool):
    bad = not value
  elif op == "=":
    bad = value + 1
  elif op == "<=":
    bad = value + 1
  else:
    bad = value - 1
  base = {**_base_passing_result(), field: bad}
  assert harness.evaluate_outcome(base) == "fail"
  in_marginal = {**base, "median_contact_delta_1crn": 0.00015}
  assert harness.evaluate_outcome(in_marginal) == "fail"


def test_cbclash_monotonicity_catches_swapped_policy():
  c_all = {"A,1": 0.5, "A,2": 0.1}
  c_ala = {"A,1": 0.4, "A,2": 0.2}
  assert harness.cbclash_monotonicity(c_all, c_ala) == (1, 2)
  assert harness.cbclash_effect(c_all, c_ala) == 1


def test_comparisons_reject_unequal_or_empty_key_sets():
  with pytest.raises(ValueError, match="key sets differ"):
    harness.cbclash_monotonicity({"A,1": 0.5}, {"A,2": 0.5})
  with pytest.raises(ValueError, match="empty comparison"):
    harness.crowdmode_effect({}, {})


def test_max_micro_diff_tolerates_one_printed_unit_only():
  assert harness.max_micro_diff({"a": 0.123457}, {"a": 0.123456}, "x") == (1, 1)
  assert harness.max_micro_diff({"a": 0.123458}, {"a": 0.123456}, "x") == (2, 1)


def test_endpoint_agreement_flags_mismatched_endpoints():
  count = {"A,1": 0.0, "A,2": 1.0, "A,3": 0.5}
  good = {"A,1": 0.0, "A,2": 1.0, "A,3": 0.9}
  bad = {"A,1": 0.1, "A,2": 1.0, "A,3": 0.9}
  assert harness.endpoint_agreement(count, good) == (0, 2)
  assert harness.endpoint_agreement(count, bad) == (1, 2)


def test_dclash_monotonicity_counts_decreases():
  series = [{"A,1": 0.1, "A,2": 0.5}, {"A,1": 0.2, "A,2": 0.4}, {"A,1": 0.2, "A,2": 0.6}]
  assert harness.dclash_monotonicity(series) == (1, 4)
  with pytest.raises(ValueError):
    harness.dclash_monotonicity(series[:1])


def _factorial(**overrides):
  crowd = {**_ARM_CROWD, **overrides}
  arms = {
    "A": (_CONTACTS, crowd["ala-count"], _FREEDOM),
    "B": (_CONTACTS, crowd["all-weighted"], _FREEDOM),
    "C": (_CONTACTS, crowd["all-count"], _FREEDOM),
    "D": (_CONTACTS, crowd["ala-weighted"], _FREEDOM),
  }
  return {"1crn": arms, "1ubq": arms}


def _sweep(all_values=None):
  c_all = all_values or {"A,1": 0.25, "A,2": 0.0}
  c_ala = {"A,1": 0.20, "A,2": 0.0}
  n = len(harness.DCLASH_SWEEP)
  return {"all-residues": [c_all] * n, "ala-only": [c_ala] * n}


def test_clean_invariants_on_consistent_arms_are_clean():
  out = harness.compute_clean_invariants(_factorial(), _sweep())
  assert out["cbclash_mono_violations_1crn_count"] == 0
  assert out["crowdmode_effect_residues_1crn"] == 1
  assert out["endpoint_violations_1crn"] == 0
  # A,2 is the only count-mode endpoint, in each of the two policy comparisons.
  assert out["endpoint_n_eligible_1crn"] == 2
  assert out["dclash_mono_violations_1crn"] == 0
  assert out["dclash_mono_n_compared_1crn"] == 2 * 3 * 2
  assert out["cbclash_effect_max_over_sweep_1crn"] == 1


def test_ignored_flags_fail_the_positive_controls():
  """An ignored --crowdmode or --cbclash passes monotonicity but must fail the effect gates."""
  same = _ARM_CROWD["ala-count"]
  out = harness.compute_clean_invariants(
    _factorial(**{"all-weighted": same, "all-count": same, "ala-weighted": same}),
    _sweep(all_values=same),
  )
  assert out["cbclash_mono_violations_1crn_count"] == 0
  assert out["crowdmode_effect_residues_1crn"] == 0
  assert out["cbclash_effect_max_over_sweep_1crn"] == 0


def test_check_arm_headers_detects_a_changed_default(tmp_path):
  def write(name, cb, crowd, dclash):
    path = tmp_path / name
    path.write_text(f"# cbclash\t{cb}\n# crowdmode\t{crowd}\n# dclash\t{dclash}\n")
    return str(path)

  good_default = write("a.out", "all-residues", "probability-weighted", "2")
  good_parity = write("b.out", "ala-only", "pruned-count-fraction", "2")
  good_sweep = write("c.out", "ala-only", "pruned-count-fraction", "3.5")
  drifted_default = write("d.out", "ala-only", "pruned-count-fraction", "2")
  runs = [
    (good_default, []),
    (good_parity, list(harness.PARITY_FLAGS)),
    (good_sweep, [*harness.PARITY_FLAGS, "--dclash", "3.5"]),
  ]
  assert harness.check_arm_headers(runs) == (0, 3)
  assert harness.check_arm_headers([*runs, (drifted_default, [])]) == (1, 4)


def test_clean_runs_are_registered_once_for_the_header_check(monkeypatch, tmp_path):
  locations = _provision_locations(monkeypatch, tmp_path)
  _point_run_at(monkeypatch, locations, None)
  harness._CLEAN_RUNS.clear()
  pdb = locations["pdb_dir"] / "1CRN.pdb"
  harness.run_clean(pdb, "1crn", harness.ALL_COUNT_FLAGS)
  harness.run_clean(pdb, "1crn", harness.ALL_COUNT_FLAGS)
  assert len(harness._CLEAN_RUNS) == 1
  harness._CLEAN_RUNS.clear()


def test_v2_entry_point_binds_the_v2_sidecar_and_reports_the_harness_usage(tmp_path):
  """bth binds <stem>.bth.toml, so the v2 sidecar needs a script with its stem."""
  entry = SCRIPT.with_name("confind_blackbox_parity_v2.py")
  assert entry.with_suffix(".bth.toml") == SIDECAR_V2
  assert SIDECAR_V2.is_file()
  env = {k: v for k, v in os.environ.items() if not k.startswith("CONFIND_PARITY_")}
  done = subprocess.run(
    [sys.executable, str(entry), "--out", str(tmp_path / "r.json")],
    capture_output=True,
    text=True,
    env=env,
    check=False,
  )
  # Delegation reached the harness: its argument resolution names the missing location.
  assert done.returncode not in (0, None)
  assert "--reference-bin" in done.stderr
