"""Complete a partially-observed structure from a conformational ensemble.

The problem: a structure observes only part of a reference residue set -- a cryo-EM
model with unresolved loops, a construct built from such a model and inheriting its
gaps -- and something downstream needs the full set. Pairwise-distance features are
the motivating case, where one missing residue removes every pair column that touches
it, so a 15% residue gap can cost nearly 30% of the features.

**Imputation happens at the residue-coordinate level, not the feature level.** Filling
missing *distances* directly produces n_res - 1 independent scalars per missing
residue with no guarantee they describe any arrangement of points in space -- they
need not satisfy the triangle inequality. Filling a missing *position* is three
numbers that produce a mutually consistent set of distances by construction. The same
reasoning applies to any derived feature, which is why this module returns
coordinates and leaves featurisation to the caller.

Nothing here can verify a grafted residue is where the real one would be; the acceptor
has no coordinate there to check against. What it does instead is measure how well the
donor matched at the positions both structures *do* have, and record which donor was
used and how forced that choice was, so a reader can judge the result rather than
trust it.
"""

from __future__ import annotations

import numpy as np

from proxide import kabsch_rmsd, kabsch_rmsd_batch
from proxide.tmalign import tm_scores_fixed_correspondence

from ._config import AlignmentDrift, DonorSelection, ImputeConfig, ImputeResult

__all__ = ["impute_frame", "select_donors", "superpose"]


def _validate(
  acceptor: np.ndarray, observed: np.ndarray, donors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  acceptor = np.asarray(acceptor, dtype=np.float32)
  observed = np.asarray(observed, dtype=bool)
  donors = np.asarray(donors, dtype=np.float32)

  if acceptor.ndim != 2 or acceptor.shape[1] != 3:
    raise ValueError(f"acceptor must be (n_res, 3), got {acceptor.shape}")
  n_res = acceptor.shape[0]
  if observed.shape != (n_res,):
    raise ValueError(f"observed must be ({n_res},), got {observed.shape}")
  if donors.ndim != 3 or donors.shape[1:] != (n_res, 3):
    raise ValueError(f"donors must be (n_donors, {n_res}, 3), got {donors.shape}")
  if not np.isfinite(donors).all():
    raise ValueError(
      "donors contain non-finite coordinates; a donor pool must be fully observed "
      "at every reference position, since those are exactly the positions it is "
      "being asked to supply"
    )
  if not np.isfinite(acceptor[observed]).all():
    raise ValueError(
      "acceptor has non-finite coordinates at positions marked observed; the "
      "observed mask and the coordinate array disagree about what is present"
    )
  return acceptor, observed, donors


def superpose(
  mobile: np.ndarray, target: np.ndarray, subset: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
  """Kabsch-superpose ``mobile`` onto ``target``, fitting over ``subset``.

  Returns ``(rotation, translation, rmsd_over_subset)`` such that
  ``mobile @ rotation.T + translation`` places every mobile point -- including
  those outside ``subset`` -- in the target's frame. That extrapolation beyond the
  fitted subset is the whole point: the fit uses shared residues, and the transform
  is then applied to the residues only the donor has.

  ``proxide.kabsch_rmsd`` centres both inputs internally and returns a rotation
  relating the *centred* sets, so the centroids are reapplied here. Getting that
  wrong displaces every grafted residue by the distance between the two centroids
  while leaving the reported RMSD untouched, which is why
  ``test_superpose_recovers_a_known_rigid_transform`` exists.
  """
  mobile = np.asarray(mobile, dtype=np.float32)
  target = np.asarray(target, dtype=np.float32)
  if subset is None:
    m_fit, t_fit = mobile, target
  else:
    m_fit, t_fit = mobile[subset], target[subset]

  if m_fit.shape[0] < 3:
    raise ValueError(
      f"need at least 3 points to superpose, got {m_fit.shape[0]}; a rotation is "
      "not determined by fewer"
    )

  res = kabsch_rmsd(m_fit, t_fit)
  rotation = np.asarray(res["rotation"], dtype=np.float64)
  m_centroid = m_fit.mean(axis=0, dtype=np.float64)
  t_centroid = t_fit.mean(axis=0, dtype=np.float64)
  # kabsch_rmsd's rotation maps centred-mobile onto centred-target (R @ a ~= b),
  # so the affine form is R @ (p - centroid_m) + centroid_t.
  translation = t_centroid - rotation @ m_centroid
  return rotation, translation, float(res["rmsd"])


def _apply(coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
  return (np.asarray(coords, dtype=np.float64) @ rotation.T + translation).astype(np.float32)


def _fit_positions(observed: np.ndarray, config: ImputeConfig) -> np.ndarray:
  """Positions to superpose over: all shared residues, or those flanking a gap."""
  shared = np.flatnonzero(observed)
  if config.superpose_on == "shared":
    return shared

  missing = np.flatnonzero(~observed)
  if missing.size == 0:
    return shared
  keep = np.zeros(observed.shape[0], dtype=bool)
  for m in missing:
    lo = max(0, m - config.flank_width)
    hi = min(observed.shape[0], m + config.flank_width + 1)
    keep[lo:hi] = True
  flanking = np.flatnonzero(keep & observed)
  # Falling back to a global fit is better than superposing on too few points, but
  # it means the result is not what was asked for, so the caller is told via
  # provenance rather than silently given a different alignment.
  if flanking.size < 3:
    return shared
  return flanking


def select_donors(
  acceptor: np.ndarray,
  observed: np.ndarray,
  donors: np.ndarray,
  config: ImputeConfig,
) -> tuple[np.ndarray, DonorSelection, dict]:
  """Rank donors by agreement with the acceptor over shared residues only.

  Returns ``(ranked_indices, selection, info)``, best first.

  Scoring is restricted to shared residues because those are the only positions
  where the two structures can be compared at all. Both metrics reject non-finite
  input, so this restriction has to be explicit rather than implicit in NaN
  propagation -- which is the safer arrangement anyway.
  """
  shared = np.flatnonzero(observed)
  acc = np.ascontiguousarray(acceptor[shared], dtype=np.float32)
  cand = np.ascontiguousarray(donors[:, shared, :], dtype=np.float32)
  n_donors = donors.shape[0]

  info: dict = {"prefiltered": False}

  if config.metric == "rmsd":
    scores = np.asarray(kabsch_rmsd_batch(acc, cand), dtype=np.float64)
    better_is_lower = True
    n_scored = n_donors
  else:
    subset = np.arange(n_donors)
    if 0 < config.prefilter_keep < n_donors:
      # TM-align's refinement is far more expensive per candidate than Kabsch, so
      # rank cheaply first. This can change the answer if the metrics disagree
      # about which donors are plausible at all, hence the provenance flag.
      cheap = np.asarray(kabsch_rmsd_batch(acc, cand), dtype=np.float64)
      subset = np.argsort(cheap, kind="stable")[: config.prefilter_keep]
      info["prefiltered"] = True
      info["prefilter_keep"] = int(config.prefilter_keep)
      cand = np.ascontiguousarray(cand[subset])
    tm = np.asarray(tm_scores_fixed_correspondence(acc, cand), dtype=np.float64)
    scores = np.full(n_donors, -np.inf)
    scores[subset] = tm
    better_is_lower = False
    n_scored = int(subset.size)

  ranked_by = scores if better_is_lower else -scores
  order = np.argsort(ranked_by, kind="stable")
  best = int(order[0])
  best_val = float(scores[best])
  runner_up = float(scores[order[1]]) if n_donors > 1 else float("nan")

  if better_is_lower:
    tied = scores <= best_val + config.degeneracy_epsilon
    margin = runner_up - best_val if n_donors > 1 else float("nan")
  else:
    tied = scores >= best_val - config.degeneracy_epsilon
    margin = best_val - runner_up if n_donors > 1 else float("nan")

  selection = DonorSelection(
    frame_index=best,
    metric=config.metric,
    metric_value=best_val,
    runner_up_value=runner_up,
    selection_margin=float(margin),
    n_within_epsilon=int(tied.sum()),
    n_candidates=n_donors,
    n_scored_exactly=n_scored,
  )
  return order, selection, info


def _drift(
  acceptor: np.ndarray,
  donor_moved: np.ndarray,
  fit_positions: np.ndarray,
  rmsd_shared: float,
) -> AlignmentDrift:
  d = np.linalg.norm(
    donor_moved[fit_positions].astype(np.float64)
    - acceptor[fit_positions].astype(np.float64),
    axis=1,
  )
  worst = int(fit_positions[int(np.argmax(d))]) if d.size else -1
  return AlignmentDrift(
    rmsd_shared=float(rmsd_shared),
    per_residue=d.astype(np.float32),
    shared_positions=fit_positions.astype(np.int32),
    median=float(np.median(d)) if d.size else float("nan"),
    p95=float(np.percentile(d, 95)) if d.size else float("nan"),
    max=float(d.max()) if d.size else float("nan"),
    max_position=worst,
  )


def impute_frame(
  acceptor: np.ndarray,
  observed: np.ndarray,
  donors: np.ndarray,
  config: ImputeConfig | None = None,
) -> ImputeResult:
  """Complete ``acceptor`` at the positions ``observed`` marks as missing.

  Args:
    acceptor: ``(n_res, 3)`` reference-length coordinates. Values at unobserved
      positions are ignored and may be NaN.
    observed: ``(n_res,)`` bool. True where the acceptor has a real coordinate.
    donors: ``(n_donors, n_res, 3)``, fully observed at every reference position.
    config: see :class:`ImputeConfig`. Defaults to ``method="reject"``.

  Returns:
    An :class:`ImputeResult`. ``provenance`` is a plain dict, safe to serialise
    alongside whatever the completed structure feeds.

  Raises:
    ValueError: on shape or finiteness violations, when coverage falls below
      ``coverage_floor`` or ``min_shared_residues``, or when ``method="reject"``
      meets anything short of full coverage.
  """
  config = config or ImputeConfig()
  acceptor, observed, donors = _validate(acceptor, observed, donors)

  n_res = acceptor.shape[0]
  n_obs = int(observed.sum())
  coverage = n_obs / n_res if n_res else 0.0
  missing = ~observed

  base_prov: dict = {
    "config": config.as_dict(),
    "n_reference_residues": n_res,
    "n_observed_residues": n_obs,
    "n_missing_residues": int(missing.sum()),
    "coverage": coverage,
    "n_donors": int(donors.shape[0]),
  }

  if missing.sum() == 0:
    # Nothing to do, and every method agrees on that. Returned as a real result
    # with an explicit method label rather than short-circuiting, so a provenance
    # record exists for the "we checked and there was no gap" case too.
    return ImputeResult(
      coordinates=acceptor.copy(),
      imputed_mask=np.zeros(n_res, dtype=bool),
      method="none_needed",
      provenance={**base_prov, "method_applied": "none_needed"},
    )

  if config.method == "reject":
    raise ValueError(
      f"coverage is {coverage:.4f} ({n_obs}/{n_res} residues) and method='reject'. "
      "Completing the structure is a deliberate choice: pass method='mask' to "
      "project on the observed subset without inventing coordinates, or "
      "method='nearest_frame' to graft them from a donor."
    )

  if coverage < config.coverage_floor:
    raise ValueError(
      f"coverage {coverage:.4f} is below coverage_floor {config.coverage_floor}; "
      f"{int(missing.sum())} of {n_res} residues would be grafted"
    )
  if n_obs < config.min_shared_residues:
    raise ValueError(
      f"only {n_obs} shared residues, below min_shared_residues "
      f"{config.min_shared_residues}; too few to superpose a donor on"
    )

  if config.method == "mask":
    # Deliberately returns the acceptor untouched, NaN and all. See ImputeResult.
    return ImputeResult(
      coordinates=acceptor.copy(),
      imputed_mask=np.zeros(n_res, dtype=bool),
      method="mask",
      provenance={**base_prov, "method_applied": "mask"},
    )

  if donors.shape[0] == 0:
    raise ValueError(f"method={config.method!r} needs at least one donor, got 0")

  fit_pos = _fit_positions(observed, config)
  prov = {
    **base_prov,
    "n_superposition_positions": int(fit_pos.size),
    "superpose_fallback_to_shared": bool(
      config.superpose_on == "flanking" and fit_pos.size == n_obs
    ),
  }

  if config.method == "mean_frame":
    moved = np.empty_like(donors)
    for i in range(donors.shape[0]):
      rot, trans, _ = superpose(donors[i], acceptor, fit_pos)
      moved[i] = _apply(donors[i], rot, trans)
    out = acceptor.copy()
    out[missing] = moved[:, missing, :].mean(axis=0)
    rot, trans, rmsd_shared = superpose(moved.mean(axis=0), acceptor, fit_pos)
    drift = _drift(acceptor, moved.mean(axis=0), fit_pos, rmsd_shared)
    return ImputeResult(
      coordinates=out,
      imputed_mask=missing.copy(),
      method="mean_frame",
      drift=drift,
      provenance={
        **prov,
        "method_applied": "mean_frame",
        **drift.as_dict(),
        "note": (
          "A mean of superposed structures is not itself a physical structure; "
          "averaging shortens bond lengths. Control, not recommendation."
        ),
      },
    )

  order, selection, sel_info = select_donors(acceptor, observed, donors, config)
  prov.update(sel_info)

  if config.method == "nearest_frame":
    idx = selection.frame_index
    rot, trans, rmsd_shared = superpose(donors[idx], acceptor, fit_pos)
    moved = _apply(donors[idx], rot, trans)
    out = acceptor.copy()
    out[missing] = moved[missing]
    drift = _drift(acceptor, moved, fit_pos, rmsd_shared)
    return ImputeResult(
      coordinates=out,
      imputed_mask=missing.copy(),
      method="nearest_frame",
      selection=selection,
      drift=drift,
      provenance={
        **prov,
        "method_applied": "nearest_frame",
        **selection.as_dict(),
        **drift.as_dict(),
      },
    )

  # nearest_frame_ensemble
  k = min(config.k, donors.shape[0])
  chosen = order[:k]
  grafts = np.empty((k, int(missing.sum()), 3), dtype=np.float32)
  best_moved = None
  best_rmsd = float("nan")
  for slot, idx in enumerate(chosen):
    rot, trans, rmsd_shared = superpose(donors[idx], acceptor, fit_pos)
    moved = _apply(donors[idx], rot, trans)
    grafts[slot] = moved[missing]
    if slot == 0:
      best_moved, best_rmsd = moved, rmsd_shared

  out = acceptor.copy()
  out[missing] = grafts.mean(axis=0)
  # Per-residue spread across the k grafts: how much the imputed position depends
  # on which donor was chosen. Large spread means the ensemble does not agree and
  # a single-donor graft would have been an arbitrary pick among alternatives.
  spread = np.full(n_res, np.nan, dtype=np.float32)
  spread[missing] = np.linalg.norm(grafts - grafts.mean(axis=0), axis=2).std(axis=0)
  assert best_moved is not None
  drift = _drift(acceptor, best_moved, fit_pos, best_rmsd)

  return ImputeResult(
    coordinates=out,
    imputed_mask=missing.copy(),
    method="nearest_frame_ensemble",
    selection=selection,
    drift=drift,
    ensemble_spread=spread,
    provenance={
      **prov,
      "method_applied": "nearest_frame_ensemble",
      "k_used": int(k),
      "donor_frame_indices": [int(i) for i in chosen],
      **selection.as_dict(),
      **drift.as_dict(),
      "ensemble_spread_median_a": float(np.nanmedian(spread)),
      "ensemble_spread_max_a": float(np.nanmax(spread)),
    },
  )
