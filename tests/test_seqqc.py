"""Unit tests for proxide.seqqc (positive, negative, and batching equivalence)."""

from __future__ import annotations

import inspect
import math
import subprocess
import sys

import numpy as np
import pytest

from proxide.seqqc import (
    AA,
    SWISSPROT_BG,
    amino_acid_frequencies,
    aromatic_fraction,
    gravy,
    isoelectric_point,
    net_charge,
    screen,
    sequence_entropy,
)


def test_positive_selftest_single_residue() -> None:
    """Positive selftest: single residue EEEE produces freqs['E'] == 1.0."""
    f = amino_acid_frequencies("EEEE")
    assert np.isclose(f["E"], 1.0)
    assert np.isclose(sum(f.values()), 1.0)
    for a in AA:
        if a != "E":
            assert f[a] == 0.0


def test_positive_selftest_swissprot_background_sums_to_one() -> None:
    """Positive selftest: SwissProt background frequencies sum to 1.0 +- 0.01."""
    bg_sum = sum(SWISSPROT_BG.values())
    assert abs(bg_sum - 1.0) < 0.01


def test_positive_selftest_standard_sequence_passes_screen() -> None:
    """Positive selftest: balanced sequence passes screening complexity filter."""
    # Balanced sequence with all 20 amino acids
    seq = "ACDEFGHIKLMNPQRSTVWY"
    res = screen([seq])
    assert res["controls_pass"] is True
    assert res["screen_pass"] is True
    assert res["flag_reasons"] == []
    assert res["n_sequences"] == 1
    assert np.isclose(res["mean_ek"], 2 / 20)
    assert np.isclose(res["mean_aromatic_fraction"], 3 / 20)


def test_net_charge_models() -> None:
    """Test heuristic and Bjellqvist net charge calculations."""
    seq = "KKRRDD"  # 2 K (+2), 2 R (+2), 2 D (-2) -> net +2
    q_heur = net_charge(seq, ph=7.4, model="heuristic")
    assert np.isclose(q_heur, 2.0)

    # Heuristic at non-7.4 raises ValueError
    with pytest.raises(ValueError, match="only defined at pH 7.4"):
        net_charge(seq, ph=6.0, model="heuristic")

    # Bjellqvist Henderson-Hasselbalch charge
    q_bj_low = net_charge("EEEE", ph=1.0, model="bjellqvist")
    q_bj_high = net_charge("EEEE", ph=12.0, model="bjellqvist")
    assert q_bj_low > q_bj_high  # Monotonically decreasing with pH


def test_isoelectric_point() -> None:
    """Test pI root finding within 1e-4 tolerance."""
    pi_e = isoelectric_point("EEEE")
    # At pI, net charge under Bjellqvist model should be approximately 0
    q_at_pi = net_charge("EEEE", ph=pi_e, model="bjellqvist")
    assert abs(q_at_pi) < 0.01
    assert 3.0 <= pi_e <= 4.0

    pi_k = isoelectric_point("KKKK")
    assert 10.0 <= pi_k <= 11.0


def test_gravy_and_aromatic() -> None:
    """Test GRAVY and aromatic fraction."""
    # Kyte-Doolittle: A=1.8, I=4.5, V=4.2 -> mean 3.5
    assert np.isclose(gravy("AIV"), 3.5)
    # Aromatic: F, W, Y
    assert np.isclose(aromatic_fraction("AFWYA"), 3 / 5)
    assert aromatic_fraction("AAAA") == 0.0


def test_sequence_entropy() -> None:
    """Test sequence entropy calculation."""
    # Homopolymer has 0 entropy
    assert sequence_entropy("AAAA") == 0.0
    # Equal distribution of 2 amino acids has entropy 1.0 (base 2)
    assert np.isclose(sequence_entropy("AKAK"), 1.0)
    # Normalized entropy of 20 distinct residues is 1.0
    assert np.isclose(sequence_entropy("ACDEFGHIKLMNPQRSTVWY", normalized=True), 1.0)


def test_negative_selftest_empty_sequence_raises() -> None:
    """Negative selftest: empty string raises ValueError."""
    with pytest.raises(ValueError, match="Sequence cannot be empty"):
        amino_acid_frequencies("")

    with pytest.raises(ValueError, match="Sequence cannot be empty"):
        net_charge("")

    with pytest.raises(ValueError, match="Sequence cannot be empty"):
        gravy("")

    with pytest.raises(ValueError, match="Sequence cannot be empty"):
        aromatic_fraction("")

    with pytest.raises(ValueError, match="Sequence cannot be empty"):
        sequence_entropy("")

    with pytest.raises(ValueError, match="Sequence cannot be empty"):
        isoelectric_point("")


def test_negative_selftest_invalid_characters_raise() -> None:
    """Negative selftest: invalid characters raise ValueError."""
    with pytest.raises(ValueError, match="Invalid amino acid character"):
        amino_acid_frequencies("ACDEF123")

    with pytest.raises(ValueError, match="Invalid amino acid character"):
        net_charge("ACDEF_X")


def test_negative_selftest_empty_screen_inputs_raise() -> None:
    """Negative selftest: empty sequence collections raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        screen([])

    with pytest.raises(ValueError, match="cannot be empty"):
        screen({})


def test_negative_selftest_skewed_sequence_fails_thresholds() -> None:
    """Negative selftest: skewed sequence 'E'*50 fails E+K and aromatic thresholds."""
    skewed = "E" * 50
    res = screen([skewed])
    assert res["controls_pass"] is True
    assert res["screen_pass"] is False
    assert "exceeds_ek_threshold" in res["flag_reasons"]
    assert "low_aromatic_fraction" in res["flag_reasons"]
    assert np.isclose(res["mean_ek"], 1.0)
    assert np.isclose(res["mean_aromatic_fraction"], 0.0)


def test_negative_selftest_frequency_separation() -> None:
    """Negative selftest from tfr1_composition.py:53: EEKK frequency separation."""
    f = amino_acid_frequencies("EEKK")
    assert not np.isclose(f["E"], 1.0)
    assert np.isclose(f["E"], 0.5)
    assert np.isclose(f["K"], 0.5)


def test_batched_equals_looped_equivalence() -> None:
    """Equivalence assertion per spec §4.3.1: batched == looped."""
    seqs = [
        "SEEEIEEQMLELYHKIEELLKVEGEDKKKALEEALELLDEIEKLAEKLPESKKEIEEEIEKHRKEIEEKLKELE",
        "SELRERAIELLKELKKLAETDKEKAKELAEEAQRLAEELLKAGDDFLVNALGEVAHLVLEGKFEEAVELAEKYLKELEKR",
        "KKSAEEVLEEFTEKLLNDERLSEELKEEIKKRVEEALKDKENFYELALGLAREVLTMAGHDTELRELAEKALKELEELKKEEE",
        "SEVEKLEEQIKKFEEAVKKVKEKYKKGEDLGHAVEELEKAAGKKAEKLVEILEKGDLKGFEELAKELLEEMKKELEEAKKKEEA",
        "VEELRRKNAELREKLGETPLTRNTEEGIAAFEKGDYKKAAQYFEFILENREGFLEALKEAAKKRGVSLEGLEEEFEEFMELVKKALEEAKK",
    ]
    batched = screen(seqs)
    looped = [screen([s]) for s in seqs]

    assert np.isclose(batched["mean_ek"], np.mean([r["mean_ek"] for r in looped]), atol=1e-6)
    assert np.isclose(batched["mean_net_charge"], np.mean([r["mean_net_charge"] for r in looped]), atol=1e-6)
    assert np.isclose(batched["mean_gravy"], np.mean([r["mean_gravy"] for r in looped]), atol=1e-6)


def test_2d_array_with_pad_mask_equivalence() -> None:
    """2D array with pad_mask produces identical results to ragged sequence list."""
    seq1 = "ACDEF"
    seq2 = "GHIKLMNP"
    seqs = [seq1, seq2]

    # Convert to 2D token array with padding
    l_max = max(len(s) for s in seqs)
    arr = np.zeros((len(seqs), l_max), dtype=int)
    mask = np.zeros((len(seqs), l_max), dtype=bool)

    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            arr[i, j] = AA.index(c)
            mask[i, j] = True

    res_arr = screen(arr, pad_mask=mask)
    res_seqs = screen(seqs)

    assert np.isclose(res_arr["mean_ek"], res_seqs["mean_ek"])
    assert np.isclose(res_arr["mean_net_charge"], res_seqs["mean_net_charge"])
    assert np.isclose(res_arr["mean_gravy"], res_seqs["mean_gravy"])


def test_empty_2d_ndarray_and_ndim_validation() -> None:
    """Empty 2D ndarray and non-2D ndarray raise descriptive ValueErrors."""
    empty_2d = np.empty((0, 10), dtype=int)
    mask = np.empty((0, 10), dtype=bool)
    with pytest.raises(ValueError, match="Sequences array cannot be empty"):
        screen(empty_2d, pad_mask=mask)

    dim1 = np.array([0, 1, 2], dtype=int)
    with pytest.raises(ValueError, match="Expected 2D sequences array"):
        screen(dim1, pad_mask=dim1 > 0)


def test_isoelectric_point_unbracketed_boundary() -> None:
    """Same-sign charge at pH 0 and pH 14 is undefined, not a pH-boundary sentinel."""
    seq = "R" * 1000
    q0 = net_charge(seq, ph=0.0, model="bjellqvist")
    q14 = net_charge(seq, ph=14.0, model="bjellqvist")
    assert q0 > 0.0
    assert q14 > 0.0
    with pytest.raises(ValueError, match=r"q\(0\)=") as excinfo:
        isoelectric_point(seq)
    message = str(excinfo.value)
    assert str(q0) in message
    assert str(q14) in message


def test_out_of_range_integer_tokens_raise() -> None:
    """Integer tokens outside [0, 20) are rejected by name, not wrapped or IndexError."""
    mask = np.array([[True]])
    for tok in (-1, 20):
        arr = np.array([[tok]], dtype=int)
        with pytest.raises(ValueError, match=str(tok)):
            screen(arr, pad_mask=mask)


def test_non_boolean_pad_mask_raises() -> None:
    """A 0/1 integer pad_mask must not be used as an index array."""
    arr = np.array([[AA.index("A"), AA.index("C")]], dtype=int)
    mask = np.array([[1, 0]], dtype=int)
    with pytest.raises(TypeError, match="bool"):
        screen(arr, pad_mask=mask)


def test_multicharacter_tokens_raise() -> None:
    """Multi-character residue tokens are not concatenated into a fake sequence."""
    arr = np.array([["ALA", "GLY", "SER"]])
    mask = np.ones((1, 3), dtype=bool)
    with pytest.raises(ValueError, match="ALA"):
        screen(arr, pad_mask=mask)
    single = np.array([["a", "c"]])
    single_mask = np.ones((1, 2), dtype=bool)
    res = screen(single, pad_mask=single_mask)
    assert np.isclose(res["mean_gravy"], gravy("AC"))


def test_background_dict_must_be_complete_distribution() -> None:
    """Only background=None selects SWISSPROT_BG; a passed dict is validated."""
    with pytest.raises(ValueError):
        screen(["ACDEFGHIKLMNPQRSTVWY"], background={})
    missing = dict(SWISSPROT_BG)
    del missing["Y"]
    with pytest.raises(ValueError):
        screen(["A"], background=missing)
    extra = dict(SWISSPROT_BG)
    extra["Z"] = 0.0
    with pytest.raises(ValueError):
        screen(["A"], background=extra)
    skewed = dict(SWISSPROT_BG)
    skewed["A"] = skewed["A"] + 0.02
    with pytest.raises(ValueError):
        screen(["A"], background=skewed)
    uniform = {a: 0.05 for a in AA}
    res_uniform = screen(["A"], background=uniform)
    res_default = screen(["A"], background=None)
    assert res_uniform["background_ek"] == 0.1
    assert res_default["background_ek"] == round(SWISSPROT_BG["E"] + SWISSPROT_BG["K"], 4)


def test_nonfinite_threshold_ek_and_ph_raise() -> None:
    """NaN and inf must not silently disable the E+K gate or return a NaN charge."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            screen(["E" * 50], threshold_ek=bad)
        with pytest.raises(ValueError):
            net_charge("EEEE", ph=bad, model="bjellqvist")
        with pytest.raises(ValueError):
            net_charge("EEEE", ph=bad, model="heuristic")


def test_dict_fold_over_background_uses_unrounded_frequencies() -> None:
    """Fold enrichment is rounded once from unrounded frequencies for every input shape."""
    seqs = ["AAACCCC"]
    freq_a = seqs[0].count("A") / len(seqs[0])
    expected = round(freq_a / SWISSPROT_BG["A"], 2)
    from_rounded_freq = round(round(freq_a, 4) / SWISSPROT_BG["A"], 2)
    assert expected != from_rounded_freq
    listed = screen(seqs)["fold_over_background"]["A"]
    grouped = screen({"g": seqs})["fold_over_background"]["g"]["A"]
    assert listed == expected
    assert grouped == expected


def test_composition_tables_cite_sources() -> None:
    """pKa, hydropathy, and background tables carry source notes and no invented release."""
    import proxide.seqqc as seqqc

    src = inspect.getsource(seqqc)
    assert "Kyte & Doolittle" in src
    assert "J. Mol. Biol. 157:105-132 (1982)" in src
    assert "Bjellqvist et al." in src
    assert "Electrophoresis 14:1023-1031 (1993)" in src
    assert "simplified" in src
    assert "unverified" in src


def test_import_proxide_does_not_eagerly_import_scipy() -> None:
    """import proxide must not load scipy, geometry, or seqqc until those names are used."""
    script = (
        "import sys\n"
        "import proxide\n"
        "loaded = [m for m in ('scipy', 'proxide.geometry', 'proxide.seqqc') if m in sys.modules]\n"
        "raise SystemExit(0 if not loaded else 1)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout

