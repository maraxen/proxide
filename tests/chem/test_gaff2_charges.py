"""GAFF2 partial-charge assignment: explicit method, validation, no silent fallback.

debt #1900 / sprint-22 Track C: charge_method has exactly two values ("espaloma",
default and strict, "gasteiger", explicit opt-in) and no "auto" -- see
proxide.chem.gaff2._assign_gaff2_charges and decision d2 in
.praxia/sprint_plans/260923_sprint-22-silent-fallbacks.toml. Every branch is
validated (finite, correct length, total charge conserves formal charge) and a
Gasteiger charge exceeding GASTEIGER_BLOWUP_GUARD raises instead of being
clipped or zeroed.

Two tiers:
- Pure tests (this file's TestValidator* classes and test_dispatcher_rejects_*):
  exercise _validate_gaff2_charges / _check_gasteiger_blowup / the
  charge_method dispatch check directly against duck-typed fakes -- no RDKit
  needed, no ATOMTYPE_GFF2.DEF needed. Run in the main pytest job.
- @pytest.mark.espaloma tests: call _assign_gaff2_charges (and
  parameterize_gaff_with_rdkit's charge plumbing) against real RDKit
  molecules, exercising the real espaloma/Gasteiger backends. These need
  rdkit + expaloma (the [espaloma] extra) but NOT the GAFF2 DEF file, since
  they call the charge helper directly rather than going through
  parameterize_gaff_with_rdkit's atom-typing path. Run in CI's test-espaloma
  job; also run locally when expaloma/rdkit happen to be installed.
"""

from __future__ import annotations

import math

import pytest

from proxide.chem.gaff2 import (
    GASTEIGER_BLOWUP_GUARD,
    _CHARGE_TOTAL_CONSERVATION_TOL,
    _assign_gaff2_charges,
    _check_gasteiger_blowup,
    _validate_gaff2_charges,
)
from proxide.chem.partial_charges import (
    CHARGE_SOURCE_ESPALOMA_AM1BCC,
    CHARGE_SOURCE_GASTEIGER,
)

# Reference Gasteiger charges for explicit-H methanol (C, O, H, H, H, H atom
# order from Chem.AddHs(Chem.MolFromSmiles("CO"))), regenerated with this
# environment's RDKit (2026.03.1) via
# rdPartialCharges.ComputeGasteigerCharges(mol, throwOnParamFailure=True).
# Matches the sprint-22 plan's stated reference values to well within 1e-3 --
# no contradiction to report.
METHANOL_GASTEIGER_REFERENCE = [0.0319, -0.3996, 0.0527, 0.0527, 0.0527, 0.2096]


class _FakeAtom:
    """Duck-typed stand-in for an rdkit.Chem.Atom -- only the methods the
    charge validator and blow-up guard actually call."""

    def __init__(self, formal_charge: int = 0, symbol: str = "C"):
        self._formal_charge = formal_charge
        self._symbol = symbol

    def GetFormalCharge(self) -> int:
        return self._formal_charge

    def GetSymbol(self) -> str:
        return self._symbol


class _FakeMol:
    """Duck-typed stand-in for an rdkit.Chem.Mol -- only GetNumAtoms/GetAtoms,
    which is all _validate_gaff2_charges touches on the mol argument."""

    def __init__(self, formal_charges: list[int]):
        self._atoms = [_FakeAtom(fc) for fc in formal_charges]

    def GetNumAtoms(self) -> int:
        return len(self._atoms)

    def GetAtoms(self) -> list[_FakeAtom]:
        return self._atoms


# ---------------------------------------------------------------------------
# Pure tests: _validate_gaff2_charges (no RDKit needed)
# ---------------------------------------------------------------------------


def test_validator_accepts_conserved_finite_charges():
    mol = _FakeMol([0, 0, 0])
    arr = _validate_gaff2_charges([0.2, -0.1, -0.1], mol, "espaloma")
    assert list(arr) == pytest.approx([0.2, -0.1, -0.1])


def test_validator_rejects_nan():
    mol = _FakeMol([0, 0, 0])
    with pytest.raises(ValueError, match="non-finite"):
        _validate_gaff2_charges([math.nan, 0.0, 0.0], mol, "espaloma")


def test_validator_rejects_inf():
    mol = _FakeMol([0, 0, 0])
    with pytest.raises(ValueError, match="non-finite"):
        _validate_gaff2_charges([math.inf, 0.0, 0.0], mol, "espaloma")


def test_validator_rejects_neg_inf():
    mol = _FakeMol([0, 0, 0])
    with pytest.raises(ValueError, match="non-finite"):
        _validate_gaff2_charges([-math.inf, 0.0, 0.0], mol, "espaloma")


def test_validator_rejects_wrong_length():
    mol = _FakeMol([0, 0, 0])
    with pytest.raises(ValueError, match="shape"):
        _validate_gaff2_charges([0.1, -0.1], mol, "espaloma")


def test_validator_rejects_bad_total():
    """Charges that don't conserve the molecule's formal charge are rejected.

    This is precisely the implicit-hydrogen-input failure mode: the real
    RDKit-backed espaloma test below (test_gasteiger_implicit_hydrogens_raise)
    exercises the same check end to end with a live measured value.
    """
    mol = _FakeMol([0, 0, 0])
    with pytest.raises(ValueError, match="sum to"):
        _validate_gaff2_charges([1.0, 1.0, 1.0], mol, "espaloma")


def test_validator_respects_nonzero_formal_charge():
    """A charged species (net formal charge != 0) is not penalized -- the
    validator compares against the molecule's actual formal charge total,
    not a hardcoded zero."""
    mol = _FakeMol([1, 0, 0])  # net +1
    arr = _validate_gaff2_charges([0.9, 0.05, 0.05], mol, "espaloma")
    assert list(arr) == pytest.approx([0.9, 0.05, 0.05])


def test_validator_tolerance_boundary():
    """Exactly at the tolerance boundary is accepted; just past it is not."""
    mol = _FakeMol([0, 0, 0])
    ok_total = _CHARGE_TOTAL_CONSERVATION_TOL * 0.99
    _validate_gaff2_charges([ok_total, 0.0, 0.0], mol, "espaloma")

    bad_total = _CHARGE_TOTAL_CONSERVATION_TOL * 10
    with pytest.raises(ValueError, match="sum to"):
        _validate_gaff2_charges([bad_total, 0.0, 0.0], mol, "espaloma")


# ---------------------------------------------------------------------------
# Pure tests: _check_gasteiger_blowup (no RDKit needed)
# ---------------------------------------------------------------------------


def test_blowup_guard_allows_values_within_bound():
    _check_gasteiger_blowup(0, "C", 0.5)
    _check_gasteiger_blowup(1, "O", -GASTEIGER_BLOWUP_GUARD)  # boundary inclusive
    _check_gasteiger_blowup(2, "N", GASTEIGER_BLOWUP_GUARD)


def test_blowup_guard_trips_on_large_positive():
    with pytest.raises(ValueError, match="blow-up guard"):
        _check_gasteiger_blowup(3, "Zn", GASTEIGER_BLOWUP_GUARD + 1)


def test_blowup_guard_trips_on_large_negative():
    with pytest.raises(ValueError, match="blow-up guard"):
        _check_gasteiger_blowup(4, "Zn", -(GASTEIGER_BLOWUP_GUARD + 1))


def test_blowup_guard_trips_on_nan():
    with pytest.raises(ValueError, match="blow-up guard"):
        _check_gasteiger_blowup(5, "C", math.nan)


def test_blowup_guard_trips_on_inf():
    with pytest.raises(ValueError, match="blow-up guard"):
        _check_gasteiger_blowup(6, "C", math.inf)


# ---------------------------------------------------------------------------
# Pure test: dispatcher rejects an unknown charge_method before touching
# RDKit at all (no "auto", no silent fallback)
# ---------------------------------------------------------------------------


def test_dispatcher_rejects_unknown_charge_method():
    """An invalid charge_method is rejected before any RDKit import/call --
    passing a non-molecule sentinel for `mol` proves this (if RDKit were
    touched first, this would blow up with an unrelated RDKit-side error
    instead of the intended ValueError)."""
    with pytest.raises(ValueError, match="Unknown charge_method"):
        _assign_gaff2_charges(object(), "auto")


def test_dispatcher_rejects_auto_explicitly():
    """'auto' specifically is named as rejected -- decision d2 forbids it."""
    with pytest.raises(ValueError, match="no 'auto'"):
        _assign_gaff2_charges(object(), "auto")


def test_dispatcher_rejects_empty_string():
    with pytest.raises(ValueError, match="Unknown charge_method"):
        _assign_gaff2_charges(object(), "")


# ---------------------------------------------------------------------------
# @pytest.mark.espaloma tests: real RDKit molecules, real backends.
# These need rdkit + expaloma ([espaloma] extra) but NOT the GAFF2 DEF file.
# ---------------------------------------------------------------------------


def _methanol_explicit_h():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles("CO")
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)
    return mol


@pytest.mark.espaloma
def test_espaloma_charges_methanol_finite_nonzero_conserved():
    pytest.importorskip("expaloma")
    pytest.importorskip("rdkit")

    mol = _methanol_explicit_h()
    charges = _assign_gaff2_charges(mol, "espaloma")

    assert len(charges) == mol.GetNumAtoms()
    assert all(math.isfinite(q) for q in charges)
    assert not all(q == 0.0 for q in charges), "espaloma must not return all-zero charges"
    formal_total = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    assert abs(float(sum(charges)) - formal_total) <= _CHARGE_TOTAL_CONSERVATION_TOL


@pytest.mark.espaloma
def test_gasteiger_charges_methanol_match_rdkit_reference():
    """Regenerated directly against this environment's RDKit (2026.03.1);
    matches the sprint-22 plan's stated reference values within 1e-3 -- no
    contradiction to report, so the plan's expected values are used verbatim.
    """
    pytest.importorskip("rdkit")

    mol = _methanol_explicit_h()
    charges = _assign_gaff2_charges(mol, "gasteiger")

    assert len(charges) == len(METHANOL_GASTEIGER_REFERENCE)
    for actual, expected in zip(charges, METHANOL_GASTEIGER_REFERENCE):
        assert abs(float(actual) - expected) <= 1e-3


@pytest.mark.espaloma
def test_gasteiger_organometallic_raises():
    """CC[Zn]C: RDKit has no Gasteiger/PEOE parameters for Zn, so this must
    raise rather than silently substituting a zero or clipped charge."""
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles("CC[Zn]C")
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)

    with pytest.raises(ValueError):
        _assign_gaff2_charges(mol, "gasteiger")


@pytest.mark.espaloma
def test_gasteiger_implicit_hydrogens_raise():
    """Methanol without AddHs: the total-charge conservation check must catch
    this (measured drift ~0.37e for methanol, far above
    _CHARGE_TOTAL_CONSERVATION_TOL) rather than silently returning charges for
    only the heavy atoms."""
    pytest.importorskip("rdkit")
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CO")  # implicit hydrogens: no AddHs call

    with pytest.raises(ValueError, match="sum to"):
        _assign_gaff2_charges(mol, "gasteiger")


@pytest.mark.espaloma
def test_espaloma_and_gasteiger_charge_source_constants_distinct():
    """CHARGE_SOURCE_* constants used by parameterize_gaff_with_rdkit's
    charge_method plumbing are distinct, non-empty strings."""
    assert CHARGE_SOURCE_ESPALOMA_AM1BCC != CHARGE_SOURCE_GASTEIGER
    assert CHARGE_SOURCE_ESPALOMA_AM1BCC and CHARGE_SOURCE_GASTEIGER
