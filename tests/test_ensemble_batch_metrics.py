"""Contract tests for the one-vs-many donor-selection metrics.

These back `proxide.ensemble`'s frame selection. Both functions take the residue
correspondence to be the identity, which is what makes them O(n_frames) instead of
n_frames alignment searches -- so the tests that matter are the ones pinning that
assumption: a frame must be its own nearest neighbour, rigid motion must not change
the answer, and a topology mismatch must raise rather than quietly compare residue i
to a different residue.
"""

from __future__ import annotations

import numpy as np
import pytest

from proxide import kabsch_rmsd_batch
from proxide.tmalign import tm_scores_fixed_correspondence


def helix(n: int, phase: float = 0.0) -> np.ndarray:
    """A non-collinear Ca-like trace. Varying `phase` is a pure rotation about z."""
    i = np.arange(n, dtype=np.float32)
    ang = np.deg2rad(i * 100.0 + phase)
    return np.stack(
        [2.3 * np.cos(ang), 2.3 * np.sin(ang), i * 1.5], axis=1
    ).astype(np.float32)


def bent(base: np.ndarray, amplitude: float) -> np.ndarray:
    """Deform the tail half outward -- a real conformational change, not a rotation."""
    out = base.copy()
    n = len(base)
    half = n // 2
    t = np.linspace(0.0, 1.0, n - half, dtype=np.float32)
    out[half:, 0] += amplitude * t
    return out


# --------------------------------------------------------------------------
# tm_scores_fixed_correspondence
# --------------------------------------------------------------------------


def test_tm_self_score_is_one():
    q = helix(40)
    s = tm_scores_fixed_correspondence(q, q[None, :, :])
    assert s.shape == (1,)
    assert s[0] == pytest.approx(1.0, abs=1e-4)


def test_tm_invariant_to_rigid_motion():
    q = helix(40)
    moved = (q + np.array([13.0, -7.0, 21.0], dtype=np.float32))[None, :, :]
    assert tm_scores_fixed_correspondence(q, moved)[0] == pytest.approx(1.0, abs=1e-4)


def test_tm_decreases_monotonically_with_deformation():
    q = helix(40)
    cands = np.stack([bent(q, a) for a in (0.0, 1.0, 3.0, 6.0, 12.0)])
    s = tm_scores_fixed_correspondence(q, cands)
    assert np.all(np.diff(s) <= 1e-6), f"not monotone: {s}"
    assert s[0] == pytest.approx(1.0, abs=1e-4)


def test_tm_argmax_recovers_the_planted_nearest_frame():
    """The actual donor-selection operation, on a trajectory with a known answer."""
    q = helix(40)
    rng = np.random.default_rng(0)
    cands = np.stack([bent(q, a) for a in rng.uniform(2.0, 15.0, size=25)])
    planted = 11
    cands[planted] = bent(q, 0.2)
    assert int(np.argmax(tm_scores_fixed_correspondence(q, cands))) == planted


def test_tm_rejects_topology_mismatch():
    with pytest.raises(ValueError, match="residues to match the query"):
        tm_scores_fixed_correspondence(helix(40), helix(39)[None, :, :])


def test_tm_rejects_nan():
    q = helix(40)
    bad = q[None, :, :].copy()
    bad[0, 5, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tm_scores_fixed_correspondence(q, bad)


def test_tm_empty_candidate_stack_returns_empty():
    q = helix(40)
    s = tm_scores_fixed_correspondence(q, np.zeros((0, 40, 3), dtype=np.float32))
    assert s.shape == (0,)


# --------------------------------------------------------------------------
# kabsch_rmsd_batch
# --------------------------------------------------------------------------


# Kabsch runs an SVD in float32, which leaves a ~1e-4 A floor on a trace spanning tens
# of Angstroms even for an exactly-identical pair. That is four orders of magnitude
# below any conformational difference this is used to rank, so the tolerance is set to
# the numerical floor rather than to zero. (test_rmsd_matches_scalar_kabsch_rmsd
# confirms the batch form is not itself adding error.)
_RMSD_FLOOR_A = 1e-3


def test_rmsd_self_is_zero():
    q = helix(40)
    assert kabsch_rmsd_batch(q, q[None, :, :])[0] == pytest.approx(
        0.0, abs=_RMSD_FLOOR_A
    )


def test_rmsd_invariant_to_rigid_motion():
    q = helix(40)
    moved = (q + np.array([100.0, 5.0, -3.0], dtype=np.float32))[None, :, :]
    assert kabsch_rmsd_batch(q, moved)[0] == pytest.approx(0.0, abs=_RMSD_FLOOR_A)


def test_rmsd_increases_monotonically_with_deformation():
    q = helix(40)
    cands = np.stack([bent(q, a) for a in (0.0, 1.0, 3.0, 6.0, 12.0)])
    r = kabsch_rmsd_batch(q, cands)
    assert np.all(np.diff(r) >= -1e-6), f"not monotone: {r}"


def test_rmsd_and_tm_agree_on_the_nearest_frame():
    """The two metrics are configurable alternatives, so they must not disagree on
    an unambiguous case -- if they did, 'nearest' would depend on the setting."""
    q = helix(40)
    cands = np.stack([bent(q, a) for a in (0.5, 4.0, 9.0, 14.0)])
    assert int(np.argmin(kabsch_rmsd_batch(q, cands))) == int(
        np.argmax(tm_scores_fixed_correspondence(q, cands))
    )


def test_rmsd_matches_scalar_kabsch_rmsd():
    """The batch form must not be a second implementation that can drift."""
    from proxide import kabsch_rmsd

    q = helix(30)
    c = bent(q, 5.0)
    batch = float(kabsch_rmsd_batch(q, c[None, :, :])[0])
    scalar = float(kabsch_rmsd(q, c)["rmsd"])
    assert batch == pytest.approx(scalar, rel=1e-5)


def test_rmsd_rejects_topology_mismatch():
    with pytest.raises(ValueError, match="residues to match the query"):
        kabsch_rmsd_batch(helix(40), helix(39)[None, :, :])


def test_rmsd_rejects_nan():
    q = helix(40)
    bad = q[None, :, :].copy()
    bad[0, 2, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        kabsch_rmsd_batch(q, bad)
