"""Invariants for nearest-frame structure completion.

The failure mode this module has to defend against is not a crash. A wrong graft
produces a complete, finite, plausible-looking coordinate array that features cleanly
and plots. So the tests here are mostly ground-truth reconstructions: hide residues
whose real positions are known, impute them, and check the answer against what was
hidden -- which is the only way to catch a transform applied in the wrong direction or
about the wrong centroid.
"""

from __future__ import annotations

import numpy as np
import pytest

from proxide.ensemble import ImputeConfig, impute_frame, select_donors, superpose


def helix(n: int, phase: float = 0.0, radius: float = 2.3) -> np.ndarray:
  i = np.arange(n, dtype=np.float32)
  ang = np.deg2rad(i * 100.0 + phase)
  return np.stack(
    [radius * np.cos(ang), radius * np.sin(ang), i * 1.5], axis=1
  ).astype(np.float32)


def bent(base: np.ndarray, amplitude: float) -> np.ndarray:
  out = base.copy()
  n = len(base)
  half = n // 2
  t = np.linspace(0.0, 1.0, n - half, dtype=np.float32)
  out[half:, 0] += amplitude * t
  return out


def rigid(coords: np.ndarray, *, axis_angle_deg: float, shift) -> np.ndarray:
  """Rotate about z by an angle and translate. Ground truth for superpose()."""
  a = np.deg2rad(axis_angle_deg)
  r = np.array(
    [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
  )
  return (coords.astype(np.float64) @ r.T + np.asarray(shift, dtype=np.float64)).astype(
    np.float32
  )


def gapped(n: int, gaps: list[tuple[int, int]]) -> np.ndarray:
  obs = np.ones(n, dtype=bool)
  for lo, hi in gaps:
    obs[lo:hi] = False
  return obs


# --------------------------------------------------------------------------
# superpose -- the transform convention
# --------------------------------------------------------------------------


def test_superpose_recovers_a_known_rigid_transform():
  """The convention check.

  kabsch_rmsd centres both inputs internally, so its rotation relates the centred
  sets. Reapplying the centroids wrongly displaces every grafted residue by the
  distance between them while leaving the reported RMSD at zero -- a wrong answer
  that reports itself as a perfect fit.
  """
  base = helix(50)
  moved = rigid(base, axis_angle_deg=37.0, shift=[12.0, -5.0, 8.0])

  rot, trans, rmsd = superpose(base, moved)

  assert rmsd == pytest.approx(0.0, abs=1e-3)
  back = base.astype(np.float64) @ rot.T + trans
  assert np.allclose(back, moved, atol=1e-2), (
    "superpose's transform does not reproduce the known rigid motion; the "
    "centroid handling is inverted"
  )


def test_superpose_transform_extrapolates_beyond_the_fitted_subset():
  """Fitting on a subset must still place points outside it correctly.

  This is exactly what grafting does: fit on shared residues, apply to the ones
  only the donor has. If the transform were only valid on the fitted subset the
  grafted coordinates would be silently wrong.
  """
  base = helix(60)
  moved = rigid(base, axis_angle_deg=-64.0, shift=[3.0, 40.0, -11.0])
  subset = np.arange(0, 30)

  rot, trans, _ = superpose(base, moved, subset)
  back = base.astype(np.float64) @ rot.T + trans

  assert np.allclose(back[30:], moved[30:], atol=1e-2)


def test_superpose_needs_three_points():
  with pytest.raises(ValueError, match="at least 3 points"):
    superpose(helix(10), helix(10), np.array([0, 1]))


# --------------------------------------------------------------------------
# Ground-truth reconstruction
# --------------------------------------------------------------------------


def test_self_donor_reconstructs_hidden_residues_exactly():
  """Hide residues, put the true structure in the donor pool, get them back.

  The strongest available invariant: the answer is known exactly. If this drifts,
  every softer comparison downstream is unanchored.
  """
  truth = helix(80)
  obs = gapped(80, [(20, 32), (55, 60)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan

  result = impute_frame(
    acceptor, obs, truth[None, :, :], ImputeConfig(method="nearest_frame")
  )

  assert result.n_imputed == 17
  assert np.isfinite(result.coordinates).all()
  assert np.allclose(result.coordinates, truth, atol=1e-2)
  assert result.drift is not None
  assert result.drift.max == pytest.approx(0.0, abs=1e-2)


def test_reconstruction_survives_a_rigidly_moved_donor():
  """The donor arrives in an arbitrary frame; superposition has to undo that."""
  truth = helix(80)
  obs = gapped(80, [(30, 45)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan
  donor = rigid(truth, axis_angle_deg=123.0, shift=[-30.0, 17.0, 55.0])

  result = impute_frame(
    acceptor, obs, donor[None, :, :], ImputeConfig(method="nearest_frame")
  )

  assert np.allclose(result.coordinates, truth, atol=1e-2)


def test_observed_positions_are_never_modified():
  """Grafting must not perturb coordinates the acceptor actually has."""
  truth = helix(80)
  obs = gapped(80, [(10, 20)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan
  donors = np.stack([bent(truth, a) for a in (1.0, 4.0, 9.0)])

  result = impute_frame(
    acceptor, obs, donors, ImputeConfig(method="nearest_frame")
  )

  assert np.array_equal(result.coordinates[obs], truth[obs])
  assert np.array_equal(result.imputed_mask, ~obs)


def test_nearest_donor_is_chosen_over_a_worse_one():
  """The planted near-identical donor must win, and its graft must be accurate."""
  truth = helix(80)
  obs = gapped(80, [(40, 52)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan
  donors = np.stack([bent(truth, 14.0), truth.copy(), bent(truth, 9.0)])

  result = impute_frame(
    acceptor, obs, donors, ImputeConfig(method="nearest_frame")
  )

  assert result.selection is not None
  assert result.selection.frame_index == 1
  assert np.allclose(result.coordinates, truth, atol=1e-2)


@pytest.mark.parametrize("metric", ["rmsd", "tm_score"])
def test_both_metrics_pick_the_same_unambiguous_donor(metric):
  truth = helix(80)
  obs = gapped(80, [(25, 35)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan
  donors = np.stack([bent(truth, 12.0), bent(truth, 0.1), bent(truth, 6.0)])

  result = impute_frame(
    acceptor, obs, donors, ImputeConfig(method="nearest_frame", metric=metric)
  )

  assert result.selection is not None
  assert result.selection.frame_index == 1


# --------------------------------------------------------------------------
# Degeneracy reporting
# --------------------------------------------------------------------------


def test_identical_donors_are_reported_as_degenerate():
  """A tie among donors must be visible, not hidden behind a confident index."""
  truth = helix(80)
  obs = gapped(80, [(30, 40)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan
  donors = np.stack([truth.copy() for _ in range(12)])

  result = impute_frame(
    acceptor, obs, donors, ImputeConfig(method="nearest_frame")
  )

  assert result.selection is not None
  assert result.selection.n_within_epsilon == 12
  assert result.selection.selection_margin == pytest.approx(0.0, abs=1e-3)


def test_a_clearly_distinct_donor_is_not_reported_as_degenerate():
  truth = helix(80)
  obs = gapped(80, [(30, 40)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan
  donors = np.stack([truth.copy(), bent(truth, 25.0), bent(truth, 40.0)])

  result = impute_frame(
    acceptor, obs, donors, ImputeConfig(method="nearest_frame", degeneracy_epsilon=0.05)
  )

  assert result.selection is not None
  assert result.selection.n_within_epsilon == 1
  assert result.selection.selection_margin > 0.05


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------


def test_reject_is_the_default_and_refuses_a_gap():
  truth = helix(80)
  obs = gapped(80, [(30, 40)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan

  with pytest.raises(ValueError, match="method='reject'"):
    impute_frame(acceptor, obs, truth[None, :, :])


def test_full_coverage_needs_no_method_and_returns_a_record():
  truth = helix(80)
  obs = np.ones(80, dtype=bool)

  result = impute_frame(truth, obs, truth[None, :, :])

  assert result.method == "none_needed"
  assert result.n_imputed == 0
  assert result.provenance["coverage"] == 1.0


def test_mask_invents_nothing_and_keeps_the_nan():
  """mask must not quietly become an imputation with a different name."""
  truth = helix(80)
  obs = gapped(80, [(30, 40)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan

  result = impute_frame(acceptor, obs, truth[None, :, :], ImputeConfig(method="mask"))

  assert result.n_imputed == 0
  assert np.isnan(result.coordinates[~obs]).all()
  assert np.array_equal(result.coordinates[obs], truth[obs])


def test_ensemble_reports_spread_and_agrees_when_donors_agree():
  truth = helix(80)
  obs = gapped(80, [(30, 40)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan
  donors = np.stack([truth.copy() for _ in range(4)])

  result = impute_frame(
    acceptor,
    obs,
    donors,
    ImputeConfig(method="nearest_frame_ensemble", k=3),
  )

  assert result.ensemble_spread is not None
  assert np.nanmax(result.ensemble_spread) == pytest.approx(0.0, abs=1e-2)
  assert np.allclose(result.coordinates, truth, atol=1e-2)
  assert np.isnan(result.ensemble_spread[obs]).all()


def test_ensemble_spread_grows_when_donors_disagree():
  """The spread has to actually track donor disagreement, or it is decoration."""
  truth = helix(80)
  obs = gapped(80, [(30, 40)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan

  agree = np.stack([bent(truth, a) for a in (3.0, 3.05, 3.1, 3.15)])
  disagree = np.stack([bent(truth, a) for a in (1.0, 8.0, 16.0, 26.0)])

  cfg = ImputeConfig(method="nearest_frame_ensemble", k=4)
  s_agree = impute_frame(acceptor, obs, agree, cfg).ensemble_spread
  s_disagree = impute_frame(acceptor, obs, disagree, cfg).ensemble_spread

  assert np.nanmax(s_disagree) > np.nanmax(s_agree)


def test_mean_frame_runs_and_labels_itself_a_control():
  truth = helix(80)
  obs = gapped(80, [(30, 40)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan
  donors = np.stack([truth.copy() for _ in range(3)])

  result = impute_frame(acceptor, obs, donors, ImputeConfig(method="mean_frame"))

  assert result.method == "mean_frame"
  assert np.isfinite(result.coordinates).all()
  assert "not itself a physical structure" in result.provenance["note"]


def test_nearest_frame_beats_mean_frame_when_the_ensemble_is_spread():
  """The claim that conformational matching earns its cost, tested rather than assumed.

  If this ever fails, nearest_frame is not buying anything over the cheap control
  and the extra machinery should be reconsidered.
  """
  truth = bent(helix(80), 6.0)
  obs = gapped(80, [(45, 60)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan
  donors = np.stack([bent(helix(80), a) for a in (0.0, 6.0, 14.0, 22.0, 30.0)])

  near = impute_frame(acceptor, obs, donors, ImputeConfig(method="nearest_frame"))
  mean = impute_frame(acceptor, obs, donors, ImputeConfig(method="mean_frame"))

  err_near = np.linalg.norm(near.coordinates[~obs] - truth[~obs], axis=1).mean()
  err_mean = np.linalg.norm(mean.coordinates[~obs] - truth[~obs], axis=1).mean()
  assert err_near < err_mean


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_coverage_floor_refuses_a_mostly_missing_structure():
  truth = helix(100)
  obs = gapped(100, [(10, 80)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan

  with pytest.raises(ValueError, match="below coverage_floor"):
    impute_frame(
      acceptor, obs, truth[None, :, :], ImputeConfig(method="nearest_frame")
    )


def test_min_shared_residues_refuses_a_small_reference():
  """A high coverage fraction of very few residues is still too few to fit on."""
  truth = helix(20)
  obs = gapped(20, [(0, 1)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan

  with pytest.raises(ValueError, match="below min_shared_residues"):
    impute_frame(
      acceptor, obs, truth[None, :, :], ImputeConfig(method="nearest_frame")
    )


def test_non_finite_donor_is_rejected():
  truth = helix(80)
  obs = gapped(80, [(30, 40)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan
  donors = truth[None, :, :].copy()
  donors[0, 5, 0] = np.nan

  with pytest.raises(ValueError, match="donor pool must be fully observed"):
    impute_frame(acceptor, obs, donors, ImputeConfig(method="nearest_frame"))


def test_mask_and_coordinates_disagreeing_is_rejected():
  """A NaN at a position claimed observed means the caller's bookkeeping is wrong."""
  truth = helix(80)
  obs = np.ones(80, dtype=bool)
  acceptor = truth.copy()
  acceptor[7, 1] = np.nan

  with pytest.raises(ValueError, match="observed mask and the coordinate array"):
    impute_frame(acceptor, obs, truth[None, :, :], ImputeConfig(method="nearest_frame"))


def test_shape_mismatch_is_rejected():
  with pytest.raises(ValueError, match="donors must be"):
    impute_frame(
      helix(80),
      np.ones(80, dtype=bool),
      helix(79)[None, :, :],
      ImputeConfig(method="mask"),
    )


def test_config_rejects_ensemble_with_k_one():
  with pytest.raises(ValueError, match="nearest_frame_ensemble with k=1"):
    ImputeConfig(method="nearest_frame_ensemble", k=1)


def test_config_rejects_unknown_method():
  with pytest.raises(ValueError, match="method must be one of"):
    ImputeConfig(method="interpolate")


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_provenance_is_serialisable_and_names_the_donor():
  import json

  truth = helix(80)
  obs = gapped(80, [(30, 40)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan
  donors = np.stack([bent(truth, a) for a in (0.2, 5.0, 11.0)])

  result = impute_frame(
    acceptor, obs, donors, ImputeConfig(method="nearest_frame", metric="rmsd")
  )

  json.dumps(result.provenance)  # must not raise
  p = result.provenance
  assert p["method_applied"] == "nearest_frame"
  assert p["frame_index"] == result.selection.frame_index
  assert p["n_missing_residues"] == 10
  assert p["config"]["metric"] == "rmsd"
  for key in ("rmsd_shared_a", "drift_median_a", "drift_max_a", "n_within_epsilon"):
    assert key in p


def test_flanking_superposition_uses_only_the_flanks():
  """superpose_on='flanking' must actually narrow the fit, not silently do nothing."""
  truth = helix(80)
  obs = gapped(80, [(30, 40)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan

  cfg = ImputeConfig(method="nearest_frame", superpose_on="flanking", flank_width=6)
  result = impute_frame(acceptor, obs, truth[None, :, :], cfg)

  assert result.provenance["superpose_fallback_to_shared"] is False
  assert result.provenance["n_superposition_positions"] < int(obs.sum())
  assert np.allclose(result.coordinates, truth, atol=1e-2)


def test_flanking_superposition_falls_back_visibly():
  """A silent fallback to a different alignment would be exactly the wrong behaviour.

  One isolated missing residue with flank_width=1 leaves only two flanking observed
  positions -- too few for a rotation -- so the fit widens to all shared residues.
  That is the right recovery, but the caller asked for something else and has to be
  able to tell it did not happen.
  """
  truth = helix(80)
  obs = gapped(80, [(40, 41)])
  acceptor = truth.copy()
  acceptor[~obs] = np.nan

  cfg = ImputeConfig(method="nearest_frame", superpose_on="flanking", flank_width=1)
  result = impute_frame(acceptor, obs, truth[None, :, :], cfg)

  assert result.provenance["superpose_fallback_to_shared"] is True
  assert result.provenance["n_superposition_positions"] == int(obs.sum())


def test_config_rejects_min_shared_below_three():
  with pytest.raises(ValueError, match="min_shared_residues must be >= 3"):
    ImputeConfig(min_shared_residues=2)
