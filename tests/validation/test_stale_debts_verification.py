"""Regression and verification tests for resolved stale debts: #2368, #2363, #1905.

Task ID: 261102_proxide-followups (Workstream B1).

Covers:
- Debt #2368: Fixed-width AMBER parm.dat loader (angle padding, PK/IDIVF, impropers).
- Debt #2363: Strict DEF parser error handling for malformed f8 and f9 bodies.
- Debt #1905 (GAFF2 half): Fail-fast Gaff2ParameterMissingError instead of 0.0 fills.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from proxide.chem import gaff2
from proxide.chem.gaff2 import (
    Gaff2DefInvalidError,
    Gaff2ParameterMissingError,
    load_gaff2_parameters,
    parse_gaff2_rules,
)

DEF_REAL_PATH = Path("/home/marielle/projects/proxide/src/proxide/assets/gaff/dat/ATOMTYPE_GFF2.DEF")
if not DEF_REAL_PATH.exists():
    DEF_REAL_PATH = Path(__file__).resolve().parents[2] / "src" / "proxide" / "assets" / "gaff" / "dat" / "ATOMTYPE_GFF2.DEF"

if DEF_REAL_PATH.exists():
    os.environ["PROXIDE_GAFF2_DEF"] = str(DEF_REAL_PATH)

_DEF_HEADER = "Defination begin\n------------------\n"


def test_debt_2368_amber_parm_dat_fixed_width_loader():
    """Debt #2368: verify load_gaff2_parameters reads fixed-width fields properly."""
    params = load_gaff2_parameters()

    # Angles with padded middle type (previously dropped by whitespace split)
    assert params["angles"][("c3", "c", "c3")] == pytest.approx((59.15, 116.68))
    assert params["angles"][("c3", "c", "o")] == pytest.approx((76.45, 122.90))
    assert params["angles"][("ca", "ca", "nb")] == pytest.approx((68.17, 122.94))

    # Generic torsion: IDIVF=4, PK=1.2 -> barrier 0.3; periodicity PN=2
    assert params["torsions"][("X", "c", "c", "X")] == [pytest.approx((2, 0.3, 180.0))]

    # Specific torsion: IDIVF=1, PN=3 -> periodicity 3 (previously misread as 1)
    assert params["torsions"][("c3", "c3", "c3", "c3")] == [pytest.approx((3, 0.52, 0.0))]

    # Impropers: previously dropped entirely
    assert params["impropers"][("X", "X", "c", "o")] == pytest.approx((10.5, 180.0))

    # Complete table counts
    assert len(params["angles"]) == 9712
    assert len(params["torsions"]) == 1341
    assert len(params["impropers"]) == 38


@pytest.mark.parametrize(
    "bad_f8",
    [
        "[unclosed",
        "[invalid-token]",
        "[2]",
    ],
)
def test_debt_2363_strict_def_parser_malformed_f8(tmp_path, bad_f8):
    """Debt #2363: verify malformed f8 bodies raise Gaff2DefInvalidError."""
    def_path = tmp_path / "bad_f8.DEF"
    def_path.write_text(f"{_DEF_HEADER}ATD  c3 * 6 4 * {bad_f8} &\n")
    with pytest.raises(Gaff2DefInvalidError, match="1 malformed ATD rule line"):
        parse_gaff2_rules(def_path)


@pytest.mark.parametrize(
    "bad_f9",
    [
        "(unclosed",
        "(N3)extra",
        "(XX[AR1,junk!])",
    ],
)
def test_debt_2363_strict_def_parser_malformed_f9(tmp_path, bad_f9):
    """Debt #2363: verify malformed f9 patterns raise Gaff2DefInvalidError."""
    def_path = tmp_path / "bad_f9.DEF"
    def_path.write_text(f"{_DEF_HEADER}ATD  c3 * 6 4 * * {bad_f9} &\n")
    with pytest.raises(Gaff2DefInvalidError, match="1 malformed ATD rule line"):
        parse_gaff2_rules(def_path)


def test_debt_1905_gaff2_missing_parameter_raises(monkeypatch):
    """Debt #1905 (GAFF2 half): verify missing parameter raises instead of 0.0 fill."""
    Chem = pytest.importorskip("rdkit.Chem")

    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)C"))
    real = gaff2.load_gaff2_parameters()
    for key in [("c3", "c", "c3")]:
        real["angles"].pop(key, None)
        real["angles"].pop(key[::-1], None)

    monkeypatch.setattr(gaff2, "load_gaff2_parameters", lambda *a, **k: real)
    with pytest.raises(Gaff2ParameterMissingError, match="angle c3-c-c3"):
        gaff2.parameterize_gaff_with_rdkit(mol, charge_method="gasteiger")
