"""Verification and defect reproduction runner for stale debts #2368, #2363, #1905.

Pre-registered in scripts/validation/verify_stale_debts.bth.toml.
Task ID: 261102_proxide-followups (Workstream B1).

Proves defect reproduction against pre-fix git revisions:
- Debt #2368: pre-fix commit 2b73958 (AMBER parm.dat whitespace split
  dropping angles/impropers, wrong IDIVF)
- Debt #2363 (Python): pre-fix commit 8d74a64 (silent acceptance of
  malformed f8/f9 DEF bodies)
- Debt #2363 (Rust): pre-fix commit 48b1c5c (silent acceptance of
  malformed f8/f9 DEF bodies)
- Debt #1905 (GAFF2 half): pre-fix commit 1d4c925 (silent 0.0 fills
  into missing_params instead of raising)

And verifies genuine resolution and pass on origin/main.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Drop unrelated env site-packages before any third-party import.
sys.path = [p for p in sys.path if "jaxlint_ci312_env" not in p and "prolix_py312_site" not in p]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("verify_stale_debts")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEF_REAL_PATH = Path(
  "/home/marielle/projects/proxide/src/proxide/assets/gaff/dat/ATOMTYPE_GFF2.DEF"
)
if not DEF_REAL_PATH.exists():
  DEF_REAL_PATH = REPO_ROOT / "src" / "proxide" / "assets" / "gaff" / "dat" / "ATOMTYPE_GFF2.DEF"

SO_REAL_PATH = REPO_ROOT / "src" / "proxide" / "_proxider.abi3.so"
if not SO_REAL_PATH.exists():
  SO_REAL_PATH = Path(
    "/home/marielle/projects/proxide/.worktrees/wt-ws-b-debt-1899/src/proxide/_proxider.abi3.so"
  )

os.environ["PROXIDE_GAFF2_DEF"] = str(DEF_REAL_PATH)


def _extract_git_tree(commit: str, dest_dir: Path) -> None:
  """Extract src/ from a git commit into dest_dir."""
  tar_proc = subprocess.Popen(
    ["git", "archive", commit, "src"],
    cwd=REPO_ROOT,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )
  untar_proc = subprocess.Popen(
    ["tar", "-x", "-C", str(dest_dir)],
    stdin=tar_proc.stdout,
    stderr=subprocess.PIPE,
  )
  if tar_proc.stdout:
    tar_proc.stdout.close()
  untar_proc.communicate()
  if untar_proc.returncode != 0:
    raise RuntimeError(f"Failed to extract git archive for {commit}")

  # Symlink compiled _proxider
  if SO_REAL_PATH.exists():
    target = dest_dir / "src" / "proxide" / "_proxider.abi3.so"
    if not target.exists():
      os.symlink(str(SO_REAL_PATH), str(target))


def verify_debt_2368_reproduction() -> bool:
  """Debt #2368: verify whitespace loader defect reproduces at 2b73958."""
  with tempfile.TemporaryDirectory() as td:
    dest = Path(td)
    _extract_git_tree("2b73958", dest)

    gaff2_file = dest / "src" / "proxide" / "chem" / "gaff2.py"
    spec = importlib.util.spec_from_file_location("gaff2_prefix_2368", str(gaff2_file))
    if spec is None or spec.loader is None:
      raise RuntimeError("Could not load gaff2 spec from 2b73958")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gaff2_prefix_2368"] = mod
    spec.loader.exec_module(mod)

    dat_path = dest / "src" / "proxide" / "assets" / "gaff" / "dat" / "gaff-2.2.20.dat"
    params = mod.load_gaff2_parameters(str(dat_path))

    # Check 1: ('c3', 'c', 'c3') was dropped due to space padding 'c3-c -c3'
    padded_angle_missing = ("c3", "c", "c3") not in params.get("angles", {})

    # Check 2: 720 angle rows were dropped (8992 loaded instead of 9712)
    angles_count_deficient = len(params.get("angles", {})) == 8992

    # Check 3: specific torsion c3-c3-c3-c3 periodicity was parsed as 1 instead of 3
    c3_torsion = params.get("torsions", {}).get(("c3", "c3", "c3", "c3"))
    periodicity_wrong = c3_torsion is not None and c3_torsion[0][0] == 1

    # Check 4: all 38 impropers were dropped
    impropers_missing = len(params.get("impropers", {})) == 0

    reproduced = (
      padded_angle_missing and angles_count_deficient and periodicity_wrong and impropers_missing
    )
    log.info(
      "Debt #2368 defect reproduction at 2b73958: padded_angle_missing=%s, "
      "angles_count=%d (missing 720), periodicity_wrong=%s, impropers=%d (missing all 38)",
      padded_angle_missing,
      len(params.get("angles", {})),
      periodicity_wrong,
      len(params.get("impropers", {})),
    )
    return bool(reproduced)


def verify_debt_2368_main() -> bool:
  """Debt #2368: verify fixed-width AMBER loader correctness on current main."""
  from proxide.chem.gaff2 import load_gaff2_parameters

  params = load_gaff2_parameters()

  # Check 1: padded angles are loaded accurately
  c3_c_c3 = params["angles"].get(("c3", "c", "c3"))
  if not c3_c_c3 or abs(c3_c_c3[0] - 59.15) > 1e-4 or abs(c3_c_c3[1] - 116.68) > 1e-4:
    log.error("Angle (c3, c, c3) incorrect on main: %s", c3_c_c3)
    return False

  c3_c_o = params["angles"].get(("c3", "c", "o"))
  if not c3_c_o or abs(c3_c_o[0] - 76.45) > 1e-4 or abs(c3_c_o[1] - 122.90) > 1e-4:
    log.error("Angle (c3, c, o) incorrect on main: %s", c3_c_o)
    return False

  # Check 2: generic torsion periodicity and barrier PK/IDIVF
  x_c_c_x = params["torsions"].get(("X", "c", "c", "X"))
  if not x_c_c_x or x_c_c_x[0][0] != 2 or abs(x_c_c_x[0][1] - 0.3) > 1e-4:
    log.error("Generic torsion (X, c, c, X) incorrect on main: %s", x_c_c_x)
    return False

  # Check 3: specific torsion periodicity
  c3_4 = params["torsions"].get(("c3", "c3", "c3", "c3"))
  if not c3_4 or c3_4[0][0] != 3 or abs(c3_4[0][1] - 0.52) > 1e-4:
    log.error("Torsion c3-c3-c3-c3 incorrect on main: %s", c3_4)
    return False

  # Check 4: impropers loaded
  imp = params["impropers"].get(("X", "X", "c", "o"))
  if not imp or abs(imp[0] - 10.5) > 1e-4 or abs(imp[1] - 180.0) > 1e-4:
    log.error("Improper (X, X, c, o) incorrect on main: %s", imp)
    return False

  # Check 5: full table sizes
  if len(params["angles"]) != 9712:
    log.error("Angles count != 9712 on main: %d", len(params["angles"]))
    return False
  if len(params["torsions"]) != 1341:
    log.error("Torsions count != 1341 on main: %d", len(params["torsions"]))
    return False
  if len(params["impropers"]) != 38:
    log.error("Impropers count != 38 on main: %d", len(params["impropers"]))
    return False

  log.info(
    "Debt #2368 on main verified: 9712 angles, 1341 torsions, 38 impropers loaded correctly."
  )
  return True


def verify_debt_2363_py_reproduction() -> bool:
  """Debt #2363: verify Python DEF parser loose defect reproduces at 8d74a64."""
  with tempfile.TemporaryDirectory() as td:
    dest = Path(td)
    _extract_git_tree("8d74a64", dest)

    gaff2_file = dest / "src" / "proxide" / "chem" / "gaff2.py"
    spec = importlib.util.spec_from_file_location("gaff2_prefix_2363", str(gaff2_file))
    if spec is None or spec.loader is None:
      raise RuntimeError("Could not load gaff2 spec from 8d74a64")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gaff2_prefix_2363"] = mod
    spec.loader.exec_module(mod)

    header = "Defination begin\n------------------\n"
    bad_f8_file = dest / "bad_f8.DEF"
    bad_f8_file.write_text(f"{header}ATD  c3 * 6 4 * [unclosed &\n")

    bad_f9_file = dest / "bad_f9.DEF"
    bad_f9_file.write_text(f"{header}ATD  c3 * 6 4 * * (unclosed &\n")

    try:
      rules_f8, _ = mod.parse_gaff2_rules(str(bad_f8_file))
      f8_silently_accepted = len(rules_f8) == 1
    except Exception:
      f8_silently_accepted = False

    try:
      rules_f9, _ = mod.parse_gaff2_rules(str(bad_f9_file))
      f9_silently_accepted = len(rules_f9) == 1
    except Exception:
      f9_silently_accepted = False

    reproduced = f8_silently_accepted and f9_silently_accepted
    log.info(
      "Debt #2363 (Python) defect reproduction at 8d74a64: f8_accepted=%s, f9_accepted=%s",
      f8_silently_accepted,
      f9_silently_accepted,
    )
    return bool(reproduced)


def verify_debt_2363_py_main() -> bool:
  """Debt #2363: verify Python DEF parser strictness on current main."""
  from proxide.chem.gaff2 import Gaff2DefInvalidError, parse_gaff2_rules

  header = "Defination begin\n------------------\n"
  bad_f8_variants = ["[unclosed", "[invalid-token]", "[2]"]
  bad_f9_variants = ["(unclosed", "(N3)extra", "(XX[AR1,junk!])", "JUNK"]

  with tempfile.TemporaryDirectory() as td:
    dest = Path(td)
    for i, bad_f8 in enumerate(bad_f8_variants):
      fpath = dest / f"bad_f8_{i}.DEF"
      fpath.write_text(f"{header}ATD  c3 * 6 4 * {bad_f8} &\n")
      try:
        parse_gaff2_rules(fpath)
        log.error("Failed to reject malformed f8 on main: %s", bad_f8)
        return False
      except Gaff2DefInvalidError as e:
        if "1 malformed ATD rule line" not in str(e):
          log.error("Unexpected error message on main: %s", e)
          return False

    for i, bad_f9 in enumerate(bad_f9_variants):
      fpath = dest / f"bad_f9_{i}.DEF"
      fpath.write_text(f"{header}ATD  c3 * 6 4 * * {bad_f9} &\n")
      try:
        parse_gaff2_rules(fpath)
        log.error("Failed to reject malformed f9 on main: %s", bad_f9)
        return False
      except Gaff2DefInvalidError as e:
        if "1 malformed ATD rule line" not in str(e):
          log.error("Unexpected error message on main: %s", e)
          return False

  log.info("Debt #2363 (Python) on main verified: malformed f8 and f9 variants raise.")
  return True


# Lines the pre-fix Rust parser kept as rules (debt #2363). Each must parse
# as Ok with that rule retained. An error on any line means the defect is gone.
_DEBT_2363_RS_COMMIT = "48b1c5c"
_DEBT_2363_RS_LINES = (
  "ATD  c3 * 6 4 * [unclosed &",
  "ATD  c3 * 6 4 * * (N3)extra &",
  "ATD  c3 * 6 4 * * JUNK &",
)
_DEBT_2363_RS_TREE = Path(tempfile.gettempdir()) / "proxide-pr86-debt2363-48b1c5c"
_DEBT_2363_RS_PROBE = """\
fn main() {
    let path = std::env::args().nth(1).expect("def path");
    let content = std::fs::read_to_string(&path).expect("read");
    let docs = content.split("\\n---DOC---\\n");
    for (i, doc) in docs.enumerate() {
        let status = match proxide_gaff2::def_parser::parse_gaff2_rules(doc) {
            Ok((rules, _)) if rules.len() == 1 => "RETAINED",
            Ok(_) => "EMPTY",
            Err(_) => "ERR",
        };
        println!("DOC {i} {status}");
    }
}
"""


def _ensure_debt_2363_rs_tree() -> Path:
  """Historical proxide-gaff2 sources, outside this worktree."""
  dest = _DEBT_2363_RS_TREE
  if (dest / "Cargo.toml").is_file():
    return dest
  if dest.exists():
    shutil.rmtree(dest)
  dest.mkdir(parents=True)
  tar_proc = subprocess.Popen(
    ["git", "archive", _DEBT_2363_RS_COMMIT],
    cwd=REPO_ROOT,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )
  untar_proc = subprocess.Popen(
    ["tar", "-x", "-C", str(dest)],
    stdin=tar_proc.stdout,
    stderr=subprocess.PIPE,
  )
  if tar_proc.stdout:
    tar_proc.stdout.close()
  untar_err = untar_proc.communicate()[1]
  tar_err = tar_proc.stderr.read() if tar_proc.stderr else b""
  tar_proc.wait()
  if tar_proc.returncode != 0 or untar_proc.returncode != 0:
    raise RuntimeError(f"Failed to extract {_DEBT_2363_RS_COMMIT}: {tar_err!r} {untar_err!r}")
  def_dest = dest / "src" / "proxide" / "assets" / "gaff" / "dat" / "ATOMTYPE_GFF2.DEF"
  if not def_dest.exists():
    if not DEF_REAL_PATH.exists():
      raise RuntimeError(f"ATOMTYPE_GFF2.DEF missing at {DEF_REAL_PATH}")
    def_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEF_REAL_PATH, def_dest)
  return dest


def _debt_2363_rs_parse_outcomes(parser_source: str) -> list[str]:
  """Compile parser_source as 48b1c5c's def_parser and parse each bad line."""
  tree = _ensure_debt_2363_rs_tree()
  parser_path = tree / "crates" / "proxide-gaff2" / "src" / "def_parser.rs"
  parser_path.write_text(parser_source)
  example = tree / "crates" / "proxide-gaff2" / "examples" / "debt2363_probe.rs"
  example.parent.mkdir(parents=True, exist_ok=True)
  example.write_text(_DEBT_2363_RS_PROBE)
  header = "Defination begin\n------------------\n"
  docs = "\n---DOC---\n".join(f"{header}{line}\n" for line in _DEBT_2363_RS_LINES)
  docs_path = tree / "debt2363_docs.txt"
  docs_path.write_text(docs)
  res = subprocess.run(
    [
      "cargo",
      "run",
      "-q",
      "-p",
      "proxide-gaff2",
      "--offline",
      "--locked",
      "--example",
      "debt2363_probe",
      "--",
      str(docs_path),
    ],
    cwd=tree,
    capture_output=True,
    text=True,
  )
  if res.returncode != 0:
    raise RuntimeError("debt #2363 Rust parser probe failed\n" + res.stdout + res.stderr)
  outcomes = [line for line in res.stdout.splitlines() if line.startswith("DOC ")]
  if len(outcomes) != len(_DEBT_2363_RS_LINES):
    raise RuntimeError(f"probe returned {outcomes!r}, stderr={res.stderr!r}")
  return outcomes


def _debt_2363_rs_rule_retained(outcome: str) -> bool:
  parts = outcome.split()
  return len(parts) >= 3 and parts[0] == "DOC" and parts[2] == "RETAINED"


def verify_debt_2363_rs_reproduction() -> bool:
  """Debt #2363: malformed f8/f9 lines parse as Ok and keep the rule at 48b1c5c."""
  proc = subprocess.run(
    ["git", "show", f"{_DEBT_2363_RS_COMMIT}:crates/proxide-gaff2/src/def_parser.rs"],
    cwd=REPO_ROOT,
    capture_output=True,
    text=True,
    check=True,
  )
  outcomes = _debt_2363_rs_parse_outcomes(proc.stdout)
  reproduced = all(_debt_2363_rs_rule_retained(line) for line in outcomes)
  log.info(
    "Debt #2363 (Rust) defect reproduction at %s: %s",
    _DEBT_2363_RS_COMMIT,
    outcomes,
  )
  return bool(reproduced)


def verify_debt_2363_rs_main() -> bool:
  """Debt #2363: verify Rust DEF parser passes test on main."""
  cmd = [
    "cargo",
    "test",
    "-p",
    "proxide-gaff2",
    "test_malformed_f8_f9_bodies_are_errors_like_python",
  ]
  env = os.environ.copy()
  env["RUSTC_WRAPPER"] = ""
  res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
  if res.returncode != 0:
    log.error("Rust test failed on main: %s\n%s", res.stdout, res.stderr)
    return False
  if (
    "test def_parser::tests::test_malformed_f8_f9_bodies_are_errors_like_python ... ok"
    not in res.stdout
  ):
    log.error("Expected test output not found: %s", res.stdout)
    return False

  log.info(
    "Debt #2363 (Rust) on main verified: test_malformed_f8_f9_bodies_are_errors_like_python passed."
  )
  return True


def _c3_c_c3_angle_zero_filled(res: dict) -> bool:
  """True when c3-c-c3 is listed missing and its angle constants are 0.0.

  Any other missing angle that merely contains c3, or a c3-c-c3 angle
  carrying other constants, is not this defect.
  """
  missing = res.get("missing_params") or []
  listed = any(
    item.get("term") == "angle" and list(item.get("types") or []) == ["c3", "c", "c3"]
    for item in missing
  )
  if not listed:
    return False
  matched = [
    angle
    for angle in (res.get("angles") or [])
    if list(angle.get("types") or []) == ["c3", "c", "c3"]
  ]
  if not matched:
    return False
  return all(angle.get("kt") == 0.0 and angle.get("t0") == 0.0 for angle in matched)


def verify_debt_1905_reproduction() -> bool:
  """Debt #1905 (GAFF2 half): verify silent 0.0 parameter fills at 1d4c925."""
  with tempfile.TemporaryDirectory() as td:
    dest = Path(td)
    _extract_git_tree("1d4c925", dest)

    # Ensure PROXIDE_GAFF2_DEF points to valid DEF
    os.environ["PROXIDE_GAFF2_DEF"] = str(DEF_REAL_PATH)

    gaff2_file = dest / "src" / "proxide" / "chem" / "gaff2.py"
    spec = importlib.util.spec_from_file_location("gaff2_prefix_1905", str(gaff2_file))
    if spec is None or spec.loader is None:
      raise RuntimeError("Could not load gaff2 spec from 1d4c925")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gaff2_prefix_1905"] = mod
    spec.loader.exec_module(mod)

    # Clear cached rules in the loaded module so it reads from PROXIDE_GAFF2_DEF
    mod._default_rules = None
    mod._default_wildatom = None

    import rdkit.Chem as Chem

    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)C"))

    real = mod.load_gaff2_parameters()
    for key in [("c3", "c", "c3")]:
      real["angles"].pop(key, None)
      real["angles"].pop(key[::-1], None)

    mod.load_gaff2_parameters = lambda *a, **k: real

    try:
      res = mod.parameterize_gaff_with_rdkit(mol, charge_method="gasteiger")
      reproduced = _c3_c_c3_angle_zero_filled(res)
    except Exception as e:
      log.warning("Unexpected exception at 1d4c925: %s", e)
      reproduced = False

    log.info(
      "Debt #1905 defect reproduction at 1d4c925: c3_c_c3_zero_filled=%s",
      reproduced,
    )
    return bool(reproduced)


def verify_debt_1905_main() -> bool:
  """Debt #1905 (GAFF2 half): verify fail-fast Gaff2ParameterMissingError on current main."""
  import rdkit.Chem as Chem

  from proxide.chem import gaff2

  os.environ["PROXIDE_GAFF2_DEF"] = str(DEF_REAL_PATH)
  gaff2._default_rules = None
  gaff2._default_wildatom = None

  mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)C"))
  real = gaff2.load_gaff2_parameters()
  for key in [("c3", "c", "c3")]:
    real["angles"].pop(key, None)
    real["angles"].pop(key[::-1], None)

  orig_loader = gaff2.load_gaff2_parameters
  try:
    gaff2.load_gaff2_parameters = lambda *a, **k: real
    try:
      gaff2.parameterize_gaff_with_rdkit(mol, charge_method="gasteiger")
      log.error("Failed to raise on missing parameter on main!")
      return False
    except gaff2.Gaff2ParameterMissingError as e:
      if "angle c3-c-c3" not in str(e):
        log.error("Gaff2ParameterMissingError did not name missing angle: %s", e)
        return False
  finally:
    gaff2.load_gaff2_parameters = orig_loader

  log.info(
    "Debt #1905 on main verified: missing angle raises Gaff2ParameterMissingError('angle c3-c-c3')."
  )
  return True


def main() -> int:
  parser = argparse.ArgumentParser(description="Verify stale debts #2368, #2363, #1905")
  parser.add_argument("--out", type=Path, default=None, help="Output JSON results path")
  args = parser.parse_args()

  checks = {
    "debt_2368_reproduced": verify_debt_2368_reproduction,
    "debt_2368_main_passed": verify_debt_2368_main,
    "debt_2363_py_reproduced": verify_debt_2363_py_reproduction,
    "debt_2363_py_main_passed": verify_debt_2363_py_main,
    "debt_2363_rs_reproduced": verify_debt_2363_rs_reproduction,
    "debt_2363_rs_main_passed": verify_debt_2363_rs_main,
    "debt_1905_reproduced": verify_debt_1905_reproduction,
    "debt_1905_main_passed": verify_debt_1905_main,
  }

  results: dict[str, bool | int] = {}
  passed_count = 0
  for name, fn in checks.items():
    try:
      passed = fn()
    except Exception as e:
      log.exception("Check %s failed with unexpected exception: %s", name, e)
      passed = False
    results[name] = passed
    if passed:
      passed_count += 1

  all_repro = bool(
    results["debt_2368_reproduced"]
    and results["debt_2363_py_reproduced"]
    and results["debt_2363_rs_reproduced"]
    and results["debt_1905_reproduced"]
  )
  all_main = bool(
    results["debt_2368_main_passed"]
    and results["debt_2363_py_main_passed"]
    and results["debt_2363_rs_main_passed"]
    and results["debt_1905_main_passed"]
  )
  overall = all_repro and all_main

  payload = {
    "all_reproductions_confirmed": all_repro,
    "all_main_verifications_passed": all_main,
    "overall_pass": overall,
    "debt_2368_reproduced": bool(results["debt_2368_reproduced"]),
    "debt_2368_main_passed": bool(results["debt_2368_main_passed"]),
    "debt_2363_py_reproduced": bool(results["debt_2363_py_reproduced"]),
    "debt_2363_py_main_passed": bool(results["debt_2363_py_main_passed"]),
    "debt_2363_rs_reproduced": bool(results["debt_2363_rs_reproduced"]),
    "debt_2363_rs_main_passed": bool(results["debt_2363_rs_main_passed"]),
    "debt_1905_reproduced": bool(results["debt_1905_reproduced"]),
    "debt_1905_main_passed": bool(results["debt_1905_main_passed"]),
    "n_checks_passed": passed_count,
    "n_checks_total": len(checks),
  }

  log.info("Verification payload: %s", json.dumps(payload, indent=2))

  # Write results according to bathos contract
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    log.info("Wrote result to --out: %s", args.out)

  bth_results_path = os.environ.get("BTH_RESULTS_PATH")
  if bth_results_path:
    Path(bth_results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(bth_results_path).write_text(json.dumps(payload, indent=2))
    log.info("Wrote result to $BTH_RESULTS_PATH: %s", bth_results_path)

  script_stem_results = Path(__file__).with_name(f"{Path(__file__).stem}.bth-results.json")
  script_stem_results.write_text(json.dumps(payload, indent=2))

  return 0 if overall else 1


if __name__ == "__main__":
  sys.exit(main())
