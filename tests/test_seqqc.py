"""Unit tests for proxide.seqqc (positive, negative, and batching equivalence)."""

from __future__ import annotations

import math
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
    """Extreme sequences outside standard titration window resolve to boundaries without drift."""
    # Poly-arginine has net positive charge at pH 14.0
    assert isoelectric_point("R" * 1000) == 14.0

