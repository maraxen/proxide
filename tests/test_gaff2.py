"""Test dataset for GAFF2 parameterization.

Validation strategy:
1. Masses/bonds/angles - compare against parsed dat file entries
2. Charges - verify charge conservation (sum ≈ 0)
3. Parameter lookups - verify substitution logic works for torsions

Reference values from gaff-2.2.20.dat. Atom-type-assignment correctness is
covered by tests/test_gaff2_golden.py (real `assert`-based tests against the
ATOMTYPE_GFF2.DEF rule table) -- this file's own former ATOM_TYPE_TESTS/
test_atom_types() were removed because (a) their expected values were wrong
relative to the actual DEF file (e.g. methane -> "cx", which requires a
3-membered ring methane doesn't have; ethanol O -> "op", a 3-membered-ring
oxygen type, not "oh") and (b) the test itself used `return failed == 0`
instead of `assert`, which pytest does not treat as a failure -- it could
never actually fail regardless of content.

debt #1900 (sprint-22 Track C): the remaining tests below had the exact same
`return failed == 0` bug -- fixed here to use real `assert`s so pytest
actually fails them on a regression. `parameterize_gaff_with_rdkit` needs the
ATOMTYPE_GFF2.DEF rule table (via assign_gaff2_atom_types) for every test in
this file; per this repo's rules that file is never fetched or stubbed by an
agent, so locally (without a fetched DEF) every test here raises
Gaff2DefMissingError -- expected, not a regression. CI's `tests` job fetches
the DEF before running (see .github/workflows/ci.yml).
"""

import pytest
import numpy as np

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem

from proxide.chem.gaff2 import (
    Gaff2ParameterMissingError,
    parameterize_gaff_with_rdkit,
    load_gaff2_parameters,
)
from proxide.chem.partial_charges import CHARGE_SOURCE_ESPALOMA_AM1BCC


# Test molecules for parameter validation
PARAM_TESTS = [
    "C",  # methane
    "CC",  # ethane
    "CCC",  # propane
    "CCO",  # ethanol
    "CCCO",  # propanol
    "c1ccccc1",  # benzene
    "c1ccc(O)cc1",  # phenol
    "c1ccncc1",  # pyridine
    "CC(=O)C",  # acetone
    "CC(=O)O",  # acetic acid
]


def prepare_mol(smiles: str) -> Chem.Mol:
    """Prepare RDKit molecule with hydrogens."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.SanitizeMol(mol)
    return mol


def test_charge_conservation():
    """Charges sum to approximately zero for these (net-neutral) test molecules.

    charge_method is passed explicitly (rather than relying on the default)
    since this test cares about the numeric result, not about
    parameterize_gaff_with_rdkit's default-argument behavior.

    Note: under the pre-debt-#1900 silent-fallback implementation, this test
    would spuriously pass even when charge assignment failed entirely, because
    a failure there returned [0.0] * n_atoms -- whose sum is trivially 0.0,
    comfortably inside the tolerance below. The not-all-zero assertion is what
    catches that failure mode now.
    """
    for smiles in PARAM_TESTS:
        mol = prepare_mol(smiles)
        result = parameterize_gaff_with_rdkit(mol, charge_method="espaloma")

        assert result["charge_method"] == CHARGE_SOURCE_ESPALOMA_AM1BCC

        charges = result["charges"]
        assert not all(q == 0.0 for q in charges), (
            f"{smiles}: all-zero charges (this is the exact silent-fallback "
            "failure mode debt #1900 removed)"
        )

        charge_sum = sum(charges)
        assert abs(charge_sum) < 1e-5, f"{smiles}: charge sum {charge_sum:.2e} not ~0"


def test_parameter_lookups():
    """Parameters are looked up correctly for known types/pairs/triples/quads."""
    params = load_gaff2_parameters()

    # Mass lookups
    for atom_type in ["c3", "cp", "cx", "oh", "op", "n3", "ni", "hc"]:
        assert atom_type in params["masses"], f"missing mass lookup for {atom_type}"

    # Bond lookups
    for pair in [("c3", "c3"), ("cx", "op"), ("c3", "oh"), ("c3", "hc")]:
        key = tuple(sorted(pair))
        assert key in params["bonds"], f"missing bond lookup for {pair}"

    # Angle lookups
    for triple in [("c3", "c3", "c3"), ("c3", "c3", "oh"), ("c3", "c3", "hc")]:
        assert triple in params["angles"], f"missing angle lookup for {triple}"

    # Torsion lookups with our substitution logic
    # We substitute cx -> c3, x -> hc when looking up
    type_sub = lambda t: "c3" if t == "cx" else ("hc" if t == "x" else t)  # noqa: E731

    test_torsions = [
        (("cx", "cx", "cx", "x"), type_sub),  # propane H torsions become c3-c3-c3-hc
        (("c3", "c3", "c3", "c3"), lambda t: t),  # direct lookup
    ]

    for quad, subst in test_torsions:
        key = tuple(subst(t) for t in quad)
        assert key in params["torsions"] and params["torsions"][key], (
            f"missing torsion lookup for {quad}"
        )


def test_dat_loader_reads_amber_parm_dat_format():
    """Debt #2368: load_gaff2_parameters reads gaff-2.2.20.dat as AMBER parm.dat.

    Expected values are the same rows as read by ParmEd 4.3.1's
    AmberParameterSet on the identical file (the full-table comparison is the
    bathos-tracked scripts/validation/gaff2_dat_parmed_parity.py). Every
    assertion here failed on the previous whitespace-splitting loader:
    padded single-letter types ("c3-c -c3", "X -c -c -X") were misread as
    bonds and dropped; DIHE's IDIVF column was read as the periodicity, PK was
    never divided by IDIVF, and every IMPROPER row was dropped.
    """
    p = load_gaff2_parameters()
    # Angles with a padded middle type.
    assert p["angles"][("c3", "c", "c3")] == pytest.approx((59.15, 116.68))
    assert p["angles"][("c3", "c", "o")] == pytest.approx((76.45, 122.90))
    assert p["angles"][("ca", "ca", "nb")] == pytest.approx((68.17, 122.94))
    # Generic torsion: IDIVF=4, PK=1.2 -> barrier 0.3; periodicity from PN=2.
    assert p["torsions"][("X", "c", "c", "X")] == [pytest.approx((2, 0.3, 180.0))]
    # Specific torsion: IDIVF=1, PN=3 -> periodicity 3 (was read as 1).
    assert p["torsions"][("c3", "c3", "c3", "c3")] == [pytest.approx((3, 0.52, 0.0))]
    # Multi-term: a negative PN means another term for the same quartet follows.
    assert p["torsions"][("X", "c", "na", "X")] == [
        pytest.approx((2, 1.45, 180.0)), pytest.approx((4, 0.35, 180.0)),
    ]
    # Impropers (no IDIVF column): used to be dropped entirely.
    assert p["impropers"][("X", "X", "c", "o")] == pytest.approx((10.5, 180.0))
    # Table sizes: unique angle and torsion keys equal ParmEd's.
    assert len(p["angles"]) == 9712
    assert len(p["torsions"]) == 1341
    assert len(p["impropers"]) == 38  # 35 in ParmEd's permutation-canonical form
    assert len(p["vdw"]) == 97


def test_dat_loader_later_torsion_redefinition_replaces_not_appends(caplog):
    """gaff-2.2.20.dat defines c3-c3-n-c twice (two independent fits, rows
    12147-12149 and 12150-12152, each ended by a positive PN). The later
    definition replaces the earlier -- ParmEd's reading; appending summed
    both fits into six terms (found by the pre-registered parity run
    9e8d6de2, which failed on exactly this key). The redefinition is logged.
    """
    import logging

    with caplog.at_level(logging.INFO, logger="proxide.chem.gaff2"):
        p = load_gaff2_parameters()
    assert sorted(p["torsions"][("c3", "c3", "n", "c")]) == [
        pytest.approx((1, 1.02, 180.0)), pytest.approx((3, 0.17, 0.0)),
        pytest.approx((4, 0.10, 180.0)),
    ]
    assert "c3-c3-n-c" in caplog.text


def test_dat_loader_raises_on_a_malformed_row(tmp_path):
    """A row that does not fit its section raises instead of being skipped."""
    from pathlib import Path

    from proxide.chem import gaff2

    src = Path(gaff2.__file__).parent.parent / "assets" / "gaff" / "dat" / "gaff-2.2.20.dat"
    lines = src.read_text().split("\n")
    target = next(i for i, ln in enumerate(lines) if ln.startswith("c3-c -c3"))
    lines[target] = "c3-c -c3   not-a-number"
    bad = tmp_path / "bad.dat"
    bad.write_text("\n".join(lines))
    with pytest.raises(ValueError, match=r"bad\.dat:\d+: malformed ANGLE record"):
        load_gaff2_parameters(bad)


def test_parameter_values():
    """Parameter values match expected values from the dat file."""
    params = load_gaff2_parameters()

    # Known values from gaff-2.2.20.dat (what we actually parse).
    #
    # FIX (debt #1900, converting this file's return-failed==0 tests to real
    # asserts): the previous reference dict held VDW rmin_half values
    # (1.9069, 1.8606, 1.82, 1.7713 -- these are exactly params["vdw"]["c3"],
    # ["cp"], ["oh"], ["op"][0], measured 2026-09-23) under a dict literally
    # named `known_masses`, checked against `params["masses"]`. That
    # mismatch -- real atomic masses are 12.01 (c3, cp) / 16.0 (oh, op) amu,
    # measured against this repo's own parsed gaff-2.2.20.dat -- was
    # invisible under the old `return failed == 0` contract (pytest never
    # treats a returned False as a failure), so it silently "passed" for as
    # long as that bug existed. Corrected here to the real atomic masses;
    # this is a test-fixture correctness fix, not a re-fit to whatever the
    # code currently outputs -- amu values for C and O are settled physical
    # constants, independent of anything Track C touched.
    known_masses = {
        "c3": (12.01, 0.001),
        "cp": (12.01, 0.001),
        "oh": (16.0, 0.001),
        "op": (16.0, 0.001),
    }

    for atom_type, (expected, tolerance) in known_masses.items():
        assert atom_type in params["masses"], f"missing mass lookup for {atom_type}"
        actual = params["masses"][atom_type]
        assert abs(actual - expected) <= tolerance, (
            f"{atom_type} mass = {actual}, expected {expected}"
        )

    # Known bond values from what we actually load.
    #
    # FIX (same pass as above): the key was ("c3", "op"), which does not
    # exist in params["bonds"] (measured: params["bonds"][("c3","op")] is
    # absent; verified via `sorted(("c3","op"))`). The value pair
    # (273.64, 1.4368) is real and correct -- it's exactly
    # params["bonds"][("cx","op")], the same pair test_parameter_lookups
    # above already exercises (presence-only, no value check) -- so this was
    # a copy-paste key typo (c3 vs cx), not a wrong value. Same
    # invisible-under-return-False history as the masses fix.
    known_bonds = {
        ("cx", "op"): (273.64, 1.4368),  # alcohol O-H
    }

    for pair, (expected_kb, expected_r0) in known_bonds.items():
        key = tuple(sorted(pair))
        assert key in params["bonds"], f"missing bond lookup for {pair}"
        actual_kb, actual_r0 = params["bonds"][key]
        assert abs(actual_kb - expected_kb) <= 0.1 and abs(actual_r0 - expected_r0) <= 0.001, (
            f"{pair} bond = ({actual_kb}, {actual_r0}), expected ({expected_kb}, {expected_r0})"
        )

    # Verify we have substantial data loaded
    assert len(params["masses"]) >= 90, f"only {len(params['masses'])} masses loaded"
    assert len(params["bonds"]) >= 1000, f"only {len(params['bonds'])} bonds loaded"
    assert len(params["angles"]) >= 7000, f"only {len(params['angles'])} angles loaded"


def test_full_parameterization():
    """Full parameterization produces valid output, including charges/charge_method."""
    for smiles in PARAM_TESTS:
        mol = prepare_mol(smiles)
        result = parameterize_gaff_with_rdkit(mol, charge_method="espaloma")

        # Required fields -- charges and charge_method must always be present
        # (debt #1900: never silently omitted or left to a caller to infer).
        for field in (
            "atom_types",
            "charges",
            "charge_method",
            "masses",
            "bonds",
            "angles",
            "torsions",
        ):
            assert field in result, f"{smiles}: missing {field}"

        assert result["charge_method"] == CHARGE_SOURCE_ESPALOMA_AM1BCC, (
            f"{smiles}: charge_method = {result['charge_method']!r}, "
            f"expected {CHARGE_SOURCE_ESPALOMA_AM1BCC!r}"
        )
        assert not all(q == 0.0 for q in result["charges"]), (
            f"{smiles}: all-zero charges"
        )

        # Atom types length matches heavy atoms
        heavy_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
        assert len(result["atom_types"]) >= heavy_atoms, (
            f"{smiles}: atom_types length mismatch: "
            f"{len(result['atom_types'])} < {heavy_atoms}"
        )


# Debt #1905: Observability of silent parameter fills


def test_acetone_central_angle_has_its_dat_parameters():
    """Acetone's c3-c-c3 angle carries gaff-2.2.20.dat's 59.15 kcal/mol/rad^2,
    116.68 deg (debt #2368).

    The old loader dropped this row ("c3-c -c3": padded single-letter type),
    so the angle was filled with 0.0 on every call -- this test's previous
    version asserted exactly that, as if it were correct.
    """
    result = parameterize_gaff_with_rdkit(prepare_mol("CC(=O)C"))
    central = [a for a in result["angles"] if sorted((a["types"][0], a["types"][2])) == ["c3", "c3"]
               and a["types"][1] == "c"]
    assert len(central) == 1, [a["types"] for a in result["angles"]]
    assert central[0]["kt"] == pytest.approx(59.15)
    assert central[0]["t0"] == pytest.approx(116.68)


def test_every_angle_and_torsion_is_enumerated():
    """Review #6: angles and torsions used to be generated only along
    index-increasing paths (i<j<k, i<j<k<l), so most were never produced --
    and never checked for missing parameters. Counts are the graph's own.
    """
    acetone = parameterize_gaff_with_rdkit(prepare_mol("CC(=O)C"))
    # 2 methyl C (6 H-C-H/H-C-C each... 4 neighbours -> 6 angles) + the
    # carbonyl C (3 neighbours -> 3 angles, incl. both O-C-C) = 15.
    assert len(acetone["angles"]) == 15
    occ = [a for a in acetone["angles"] if a["types"][1] == "c" and "o" in (a["types"][0], a["types"][2])]
    assert len(occ) == 2, [a["types"] for a in acetone["angles"]]

    ethane = parameterize_gaff_with_rdkit(prepare_mol("CC"))
    hcch = [t for t in ethane["torsions"] if t["types"][0].startswith("h") and t["types"][3].startswith("h")]
    assert len(hcch) == 9  # 3 x 3 across the one C-C bond; there were 0
    benzene = parameterize_gaff_with_rdkit(prepare_mol("c1ccccc1"))
    assert len(benzene["angles"]) == 18 and len(benzene["torsions"]) == 24


def test_torsion_without_parameters_raises_or_warns():
    """Thioacetone's hc-c3-cs-s has no gaff-2.2.20.dat row (specific or
    generic). It was never even generated before review #6; now it raises by
    default, or is omitted with a warning naming it."""
    mol = prepare_mol("CC(=S)C")
    with pytest.raises(Gaff2ParameterMissingError, match="hc-c3-cs-s"):
        parameterize_gaff_with_rdkit(mol)
    with pytest.warns(UserWarning, match="hc-c3-cs-s"):
        result = parameterize_gaff_with_rdkit(mol, missing_torsions="omit")
    assert all(t["params"] for t in result["torsions"])


def test_a_term_missing_from_the_table_raises(monkeypatch):
    """No term is ever filled with 0.0: a missing one raises, naming it (debt #1905)."""
    from proxide.chem import gaff2

    real = gaff2.load_gaff2_parameters()
    for key in [("c3", "c", "c3")]:
        real["angles"].pop(key, None)
        real["angles"].pop(key[::-1], None)
    monkeypatch.setattr(gaff2, "load_gaff2_parameters", lambda *a, **k: real)
    with pytest.raises(gaff2.Gaff2ParameterMissingError, match="angle c3-c-c3"):
        gaff2.parameterize_gaff_with_rdkit(prepare_mol("CC(=O)C"))


def test_substitutions_tracking():
    """substitutions tracks torsion type substitutions applied (debt #1905)."""
    # Any molecule where a torsion substitution actually applies
    # In the golden tests, no substitutions happen (all types are available)
    # But we can verify the key exists and is a list
    for smiles in PARAM_TESTS:
        mol = prepare_mol(smiles)
        result = parameterize_gaff_with_rdkit(mol)

        # substitutions should be present
        assert "substitutions" in result, f"{smiles}: missing substitutions key"
        assert isinstance(result["substitutions"], list), (
            f"{smiles}: substitutions is not a list"
        )

        # Each entry should have term="torsion", "from", and "to" keys
        for subst in result["substitutions"]:
            assert subst.get("term") == "torsion", (
                f"{smiles}: substitution term is not 'torsion': {subst}"
            )
            assert "from" in subst and isinstance(subst["from"], list), (
                f"{smiles}: substitution missing 'from' key: {subst}"
            )
            assert "to" in subst and isinstance(subst["to"], list), (
                f"{smiles}: substitution missing 'to' key: {subst}"
            )
            assert len(subst["from"]) == 4, (
                f"{smiles}: substitution 'from' should be 4-element torsion: {subst}"
            )
            assert len(subst["to"]) == 4, (
                f"{smiles}: substitution 'to' should be 4-element torsion: {subst}"
            )


def test_substitutions_key_always_present():
    """The substitutions record is always present, even when empty (debt #1905)."""
    for smiles in PARAM_TESTS:
        result = parameterize_gaff_with_rdkit(prepare_mol(smiles))
        assert isinstance(result.get("substitutions"), list), smiles
        assert "missing_params" not in result, "missing terms raise; there is no fill to record"


def test_debt_2375_regression_impropers_and_bond_sub_visibility():
    """Regression test for Debt #2375:
    1. build_gaff2_ffxml emits <Improper> records for planar centers (e.g. acetone).
    2. parameterize_gaff_with_rdkit returns 'impropers' key with non-empty list for planar centers.
    3. _BOND_TYPE_SUB substitutions in bond and angle lookups are recorded and exposed in substitutions list.
    """
    import io
    from proxide.chem.gaff2 import (
        build_gaff2_ffxml,
        _lookup_bond_params,
        _lookup_angle_params,
    )

    # Molecule with planar carbonyl center (acetone)
    mol = prepare_mol("CC(=O)C")
    charges = [0.0] * mol.GetNumAtoms()

    # 1. Improper torsions in build_gaff2_ffxml
    xml = build_gaff2_ffxml(mol, "ACE", charges)
    assert "<Improper" in xml, "build_gaff2_ffxml must emit <Improper> records for planar centers"
    assert 'type1="ACE_1"' in xml, "Central atom (C2, index 1) must be type1 in OpenMM Improper tag"
    assert 'periodicity1="2"' in xml, "Improper periodicity must be 2"

    # OpenMM can parse the FFXML with the improper tag
    openmm_app = pytest.importorskip("openmm.app")
    ff = openmm_app.ForceField(io.StringIO(xml))
    assert ff is not None

    # 2. parameterize_gaff_with_rdkit returns 'impropers'
    res = parameterize_gaff_with_rdkit(mol, charge_method="gasteiger")
    assert "impropers" in res, "parameterize_gaff_with_rdkit must return 'impropers' key"
    assert len(res["impropers"]) > 0, "impropers list must not be empty for acetone"
    impr = res["impropers"][0]
    assert impr["center"] == 1, "Carbonyl carbon (atom 1) must be improper center"
    assert impr["pk"] == pytest.approx(10.5)
    assert impr["phase"] == pytest.approx(180.0)

    # 3. _BOND_TYPE_SUB substitutions in bond and angle lookups are recorded
    params = load_gaff2_parameters()
    bond_subs = []
    kb, r0 = _lookup_bond_params("cx", "o", params, substitutions=bond_subs)
    assert len(bond_subs) == 1, "Expected bond substitution to be recorded"
    assert bond_subs[0] == {"term": "bond", "from": ["cx", "o"], "to": ["c3", "o"]}
    assert kb > 0.0 and r0 > 0.0

    angle_subs = []
    kt, t0 = _lookup_angle_params("cx", "c2", "hc", params, substitutions=angle_subs)
    assert len(angle_subs) == 1, "Expected angle substitution to be recorded"
    assert angle_subs[0] == {
        "term": "angle",
        "from": ["cx", "c2", "hc"],
        "to": ["c3", "c2", "hc"],
    }
    assert kt > 0.0 and t0 > 0.0


def test_improper_torsions_multiple_planar_centers():
    """Verify improper torsions for benzene, acetic acid, and non-planar alkanes."""
    # Benzene: all 6 aromatic carbons are planar centers (rule X-X-ca-ha, pk=1.1, phase=180.0)
    mol_benzene = prepare_mol("c1ccccc1")
    res_b = parameterize_gaff_with_rdkit(mol_benzene)
    assert "impropers" in res_b
    assert len(res_b["impropers"]) == 6
    for impr in res_b["impropers"]:
        assert impr["pk"] == pytest.approx(1.1)
        assert impr["phase"] == pytest.approx(180.0)

    # Acetic acid: carbonyl carbon matches exact rule c3-o-c-oh (pk=1.1, phase=180.0)
    mol_ac = prepare_mol("CC(=O)O")
    res_ac = parameterize_gaff_with_rdkit(mol_ac)
    assert len(res_ac["impropers"]) == 1
    impr_ac = res_ac["impropers"][0]
    assert impr_ac["pk"] == pytest.approx(1.1)
    assert impr_ac["phase"] == pytest.approx(180.0)

    # Alkanes: methane and ethane have 0 planar centers, so 0 impropers
    for smiles in ("C", "CC"):
        res = parameterize_gaff_with_rdkit(prepare_mol(smiles))
        assert "impropers" in res
        assert len(res["impropers"]) == 0


def test_build_gaff2_ffxml_substitutions_argument():
    """Verify build_gaff2_ffxml records substitutions when caller provides substitutions list."""
    from proxide.chem.gaff2 import build_gaff2_ffxml

    mol = prepare_mol("CC(=O)C")
    charges = [0.0] * mol.GetNumAtoms()
    subs = []
    xml = build_gaff2_ffxml(mol, "ACE", charges, substitutions=subs)
    assert "<Improper" in xml
    assert isinstance(subs, list)


def _improper_line(xml: str) -> str:
    lines = [line.strip() for line in xml.splitlines() if "<Improper" in line]
    assert len(lines) == 1, lines
    return lines[0]


def test_acetone_improper_ffxml_type4_is_carbonyl_oxygen():
    """OpenMM default improper ordering keeps the central atom in type1 and the
    parm.dat r4 atom in type4. GAFF's X-X-c-o row puts the oxygen in r4.
    CC(=O)C numbers that oxygen before the second methyl, so index-sorting the
    three neighbors writes a methyl as type4 and the two acetone SMILES do not
    apply the planarity term to the same atom.
    """
    from proxide.chem.gaff2 import build_gaff2_ffxml

    for smiles in ("CC(=O)C", "CC(C)=O"):
        mol = prepare_mol(smiles)
        oxygen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
        carbonyl = next(
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetSymbol() == "C" and any(n.GetSymbol() == "O" for n in a.GetNeighbors())
        )
        xml = build_gaff2_ffxml(mol, "ACE", [0.0] * mol.GetNumAtoms())
        line = _improper_line(xml)
        assert f'type1="ACE_{carbonyl}"' in line
        assert f'type4="ACE_{oxygen}"' in line, (smiles, line)


def test_acetone_improper_quartet_central_atom_is_k_and_oxygen_is_l():
    """parameterize stores proper torsions as i-j-k-l along the dihedral.
    An improper passed to PeriodicTorsionForce.addTorsion the same way must
    use k = central atom and l = the parm.dat r4 atom (the carbonyl oxygen
    for X-X-c-o), not the central atom in i and the highest neighbor index in l.
    """
    for smiles in ("CC(=O)C", "CC(C)=O"):
        mol = prepare_mol(smiles)
        oxygen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
        carbonyl = next(
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetSymbol() == "C" and any(n.GetSymbol() == "O" for n in a.GetNeighbors())
        )
        res = parameterize_gaff_with_rdkit(mol, charge_method="gasteiger")
        assert len(res["impropers"]) == 1
        impr = res["impropers"][0]
        assert impr["center"] == carbonyl
        assert impr["k"] == carbonyl
        assert impr["l"] == oxygen
        assert {impr["i"], impr["j"]} == {n.GetIdx() for n in mol.GetAtomWithIdx(carbonyl).GetNeighbors()} - {oxygen}


def test_improper_miss_does_not_borrow_bond_type_substitution(monkeypatch):
    """gaff-2.2.20.dat has no equivalencing block (the only line between the
    IMPROPER blank and MOD4 RE is the HBOND water flag). tleap assigns an
    improper only when a row matches. ca with neighbors ca, n2, c3 is the
    real row c3-ca-ca-n2 and must stay. A center the table does not list
    must not be rewritten through _BOND_TYPE_SUB: ca with neighbors c2, c3, n2
    matches no ca row, and c2 with neighbors c2, cx, hc matches no c2 row,
    but cx→c3 / ca→c2 borrows c3-c2-c2-n2 or c2-c3-c2-hc (both pk=1.1).
    """
    from pathlib import Path

    from proxide.chem.gaff2 import _lookup_improper_params, build_gaff2_ffxml
    from proxide.chem import gaff2

    dat_path = Path(gaff2.__file__).resolve().parents[1] / "assets" / "gaff" / "dat" / "gaff-2.2.20.dat"
    lines = dat_path.read_text().splitlines()
    mod4 = next(i for i, line in enumerate(lines) if line.startswith("MOD4"))
    between = [line.strip() for line in lines[mod4 - 6 : mod4] if line.strip()]
    assert between == ["hw  ow  0000.     0000.                                4.  flag for fast water"] or any(
        line.startswith("hw") and "ow" in line for line in between
    )
    assert not any(line.startswith(("EQUI", "equi")) or "equivalen" in line.lower() for line in lines)
    assert any(line.startswith("c3-c2-c2-n2") and line.split()[1] == "1.1" for line in lines)
    assert any(line.startswith("c3-ca-ca-n2") and line.split()[1] == "1.1" for line in lines)
    assert any(line.startswith("c2-c3-c2-hc") and line.split()[1] == "1.1" for line in lines)

    params = load_gaff2_parameters()
    # The review's quartet is a direct ca-centered row, not a borrowed c2 row.
    direct = _lookup_improper_params("ca", "ca", "n2", "c3", params)
    assert direct is not None and direct[0] == pytest.approx(1.1)

    borrowed = _lookup_improper_params("ca", "c2", "c3", "n2", params)
    assert borrowed is None

    mol = prepare_mol("C=CC1CC1")
    base = gaff2.assign_gaff2_atom_types(mol)
    mapped = list(base)
    for atom in mol.GetAtoms():
        if base[atom.GetIdx()] == "c2" and any(base[n.GetIdx()] == "cx" for n in atom.GetNeighbors()):
            for nbr in atom.GetNeighbors():
                if base[nbr.GetIdx()] == "ha":
                    mapped[nbr.GetIdx()] = "hc"
    monkeypatch.setattr(gaff2, "assign_gaff2_atom_types", lambda m, *a, **k: list(mapped))

    res = parameterize_gaff_with_rdkit(mol, charge_method="gasteiger", missing_torsions="omit")
    assert res["impropers"] == []
    assert all(entry["term"] != "improper" for entry in res["substitutions"])
    xml = build_gaff2_ffxml(
        mol, "VCY", [0.0] * mol.GetNumAtoms(), missing_torsions="omit"
    )
    assert "<Improper" not in xml


def test_public_api_records_bond_angle_and_torsion_substitutions(monkeypatch):
    """build_gaff2_ffxml and parameterize_gaff_with_rdkit must record the
    from/to of every bond, angle, and torsion substitution. An empty list
    fails. cx-o is a bond miss (c3-o), cx-c2-hc is an angle miss (c3-c2-hc),
    and methylcyclopropane's hc-c3-cx-* torsions miss until cx→c3.
    """
    from proxide.chem.gaff2 import build_gaff2_ffxml
    from proxide.chem import gaff2

    def _records(subs: list[dict]) -> set[tuple]:
        return {(s["term"], tuple(s["from"]), tuple(s["to"])) for s in subs}

    # Torsion fallback on a molecule the typer already labels with cx.
    mol = prepare_mol("CC1CC1")
    charges = [0.0] * mol.GetNumAtoms()
    ff_subs: list[dict] = []
    build_gaff2_ffxml(mol, "MCP", charges, substitutions=ff_subs)
    param = parameterize_gaff_with_rdkit(mol, charge_method="gasteiger")
    torsion_rows = {
        ("torsion", ("hc", "c3", "cx", "hc"), ("hc", "c3", "c3", "hc")),
        ("torsion", ("hc", "c3", "cx", "cx"), ("hc", "c3", "c3", "c3")),
    }
    assert torsion_rows <= _records(ff_subs)
    assert torsion_rows <= _records(param["substitutions"])
    assert ff_subs and param["substitutions"]

    # Bond cx-o: cyclopropanone's carbonyl carbon relabeled c→cx. The o-cx
    # bond is not in the table; c3-o is.
    mol = prepare_mol("O=C1CC1")
    base = gaff2.assign_gaff2_atom_types(mol)
    mapped = list(base)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C" and any(n.GetSymbol() == "O" for n in atom.GetNeighbors()):
            mapped[atom.GetIdx()] = "cx"
    monkeypatch.setattr(gaff2, "assign_gaff2_atom_types", lambda m, *a, **k: list(mapped))
    bond_row = ("bond", ("o", "cx"), ("o", "c3"))
    ff_subs = []
    build_gaff2_ffxml(mol, "CPN", [0.0] * mol.GetNumAtoms(), substitutions=ff_subs)
    param = parameterize_gaff_with_rdkit(mol, charge_method="gasteiger")
    assert bond_row in _records(ff_subs)
    assert bond_row in _records(param["substitutions"])

    # Angle cx-c2-hc: vinylcyclopropane's ring-attached alkene hydrogen
    # relabeled ha→hc. That angle is not in the table; c3-c2-hc is.
    monkeypatch.undo()
    mol = prepare_mol("C=CC1CC1")
    base = gaff2.assign_gaff2_atom_types(mol)
    mapped = list(base)
    for atom in mol.GetAtoms():
        if base[atom.GetIdx()] == "c2" and any(base[n.GetIdx()] == "cx" for n in atom.GetNeighbors()):
            for nbr in atom.GetNeighbors():
                if base[nbr.GetIdx()] == "ha":
                    mapped[nbr.GetIdx()] = "hc"
    monkeypatch.setattr(gaff2, "assign_gaff2_atom_types", lambda m, *a, **k: list(mapped))
    angle_row = ("angle", ("cx", "c2", "hc"), ("c3", "c2", "hc"))
    ff_subs = []
    build_gaff2_ffxml(
        mol, "VCY", [0.0] * mol.GetNumAtoms(), substitutions=ff_subs, missing_torsions="omit"
    )
    param = parameterize_gaff_with_rdkit(mol, charge_method="gasteiger", missing_torsions="omit")
    assert angle_row in _records(ff_subs)
    assert angle_row in _records(param["substitutions"])


def test_parameterize_docstring_lists_bond_angle_and_torsion_substitutions():
    """The substitutions list records bond, angle, and torsion from/to pairs,
    not torsion rows alone.
    """
    doc = parameterize_gaff_with_rdkit.__doc__ or ""
    bullet = "\n".join(line for line in doc.splitlines() if "substitution" in line.lower())
    assert "bond" in bullet
    assert "angle" in bullet
    assert "torsion" in bullet
    assert '{"term": "torsion"' not in bullet