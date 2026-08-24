"""Configuration and result types for partially-observed structure completion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Method = Literal["reject", "mask", "mean_frame", "nearest_frame", "nearest_frame_ensemble"]
Metric = Literal["rmsd", "tm_score"]

METHODS: tuple[str, ...] = (
  "reject",
  "mask",
  "mean_frame",
  "nearest_frame",
  "nearest_frame_ensemble",
)
METRICS: tuple[str, ...] = ("rmsd", "tm_score")

# Metric-appropriate "these donors are tied" windows. RMSD is an unbounded distance in
# Angstroms, so 0.05 A is a genuinely tight window. TM-score is bounded above by 1.0 and
# saturates for frames of one protein, so the same 0.05 spans a large fraction of any
# realistic pool -- 0.002 is the comparable window there.
DEFAULT_DEGENERACY_EPSILON: dict[str, float] = {"rmsd": 0.05, "tm_score": 0.002}


@dataclass(frozen=True, kw_only=True)
class ImputeConfig:
  """How to complete a structure that observes only part of a reference residue set.

  ``reject`` is the default on purpose. Completing a structure is a choice with
  consequences for anything measured downstream, so it has to be asked for; a
  coverage gate that a default silently satisfies is not a gate.

  Attributes:
    method:
      ``reject`` -- refuse anything below full coverage. The default.
      ``mask`` -- do not impute. Leaves the structure as-is for a caller that will
        project onto the observed sub-basis by least squares. The honest baseline:
        no invented coordinates, at the cost of a less determined projection.
      ``mean_frame`` -- fill from the donor pool's mean position, after superposing
        each donor on the acceptor. A control, not a recommendation: an average of
        superposed structures is not itself a physical structure (averaging shrinks
        bond lengths), so it exists to show what conformational matching buys.
      ``nearest_frame`` -- superpose the single best-matching donor and graft its
        coordinates for the missing residues.
      ``nearest_frame_ensemble`` -- the same for the ``k`` best donors, returning
        their mean and the spread between them. The spread is the point: it
        measures how much the answer depends on which donor was picked.
    metric:
      ``rmsd`` -- Kabsch RMSD over shared residues. The default, and the right
        choice when donors share the acceptor's topology, since the residue
        correspondence is then the identity.
      ``tm_score`` -- length-normalised TM-score. Prefer it when the acceptor
        observes a small or unevenly-distributed fraction of the reference, where
        RMSD is dominated by a few outlying residues.
    k: donor count for ``nearest_frame_ensemble``. Ignored otherwise.
    superpose_on:
      ``shared`` -- superpose over every shared residue. Best global agreement.
      ``flanking`` -- superpose over shared residues within ``flank_width`` of a
        gap. A graft's quality depends on the local frame at its anchor points,
        which a global fit will trade away to improve agreement elsewhere.
    flank_width: residues either side of each gap when ``superpose_on="flanking"``.
    min_shared_residues: refuse below this many shared residues, whatever the
      coverage fraction says. A high fraction of a small reference is still too
      little to superpose on.
    coverage_floor: refuse below this observed fraction regardless of method. Not
      the caller's coverage gate -- a backstop against grafting most of a molecule.
    prefilter_keep: when ``metric="tm_score"``, rank all donors by the much cheaper
      RMSD first and score only this many with TM-score. ``0`` disables it. This is
      an optimisation and can change the answer if the two metrics disagree about
      which donors are plausible, so it is recorded in the provenance.
    degeneracy_epsilon: donors scoring within this of the best count as tied. In
      metric units, and the two metrics have completely different natural scales,
      so ``None`` (the default) resolves per metric via
      :data:`DEFAULT_DEGENERACY_EPSILON`. Setting one number for both is the trap:
      TM-scores between frames of the same protein cluster just under 1.0, so an
      Angstrom-appropriate 0.05 window swallows most of the pool and reports near-
      total degeneracy no matter how well-determined the choice actually was.
      Measured on 280 donors: median 2 tied under ``rmsd``, 241.5 under
      ``tm_score``, for the same frames.
    random_seed: reserved for deterministic tie-breaking. Selection is currently
      first-index-wins among ties, which is already deterministic.
  """

  method: Method = "reject"
  metric: Metric = "rmsd"
  k: int = 1
  superpose_on: Literal["shared", "flanking"] = "shared"
  flank_width: int = 8
  min_shared_residues: int = 50
  coverage_floor: float = 0.60
  prefilter_keep: int = 256
  degeneracy_epsilon: float | None = None
  random_seed: int = 0

  def __post_init__(self) -> None:
    if self.method not in METHODS:
      raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
    if self.metric not in METRICS:
      raise ValueError(f"metric must be one of {METRICS}, got {self.metric!r}")
    if self.k < 1:
      raise ValueError(f"k must be >= 1, got {self.k}")
    if self.method == "nearest_frame_ensemble" and self.k < 2:
      raise ValueError(
        "nearest_frame_ensemble with k=1 is nearest_frame without the spread "
        "estimate that justifies it; use method='nearest_frame' or raise k"
      )
    if not 0.0 <= self.coverage_floor <= 1.0:
      raise ValueError(f"coverage_floor must be in [0, 1], got {self.coverage_floor}")
    # A rotation is not determined by fewer than 3 points, so a lower setting cannot
    # be honoured -- it just defers the same refusal to superposition time with a
    # less informative message.
    if self.min_shared_residues < 3:
      raise ValueError(
        f"min_shared_residues must be >= 3, got {self.min_shared_residues}; "
        "a rotation is not determined by fewer points"
      )
    if self.degeneracy_epsilon is not None and self.degeneracy_epsilon < 0.0:
      raise ValueError("degeneracy_epsilon must be >= 0")
    if self.flank_width < 1:
      raise ValueError("flank_width must be >= 1")

  @property
  def epsilon(self) -> float:
    """The tied-donor window actually in force, resolved for this metric."""
    if self.degeneracy_epsilon is not None:
      return self.degeneracy_epsilon
    return DEFAULT_DEGENERACY_EPSILON[self.metric]

  def as_dict(self) -> dict:
    return {
      "method": self.method,
      "metric": self.metric,
      "k": self.k,
      "superpose_on": self.superpose_on,
      "flank_width": self.flank_width,
      "min_shared_residues": self.min_shared_residues,
      "coverage_floor": self.coverage_floor,
      "prefilter_keep": self.prefilter_keep,
      "degeneracy_epsilon": self.epsilon,
      "degeneracy_epsilon_was_default": self.degeneracy_epsilon is None,
      "random_seed": self.random_seed,
    }


@dataclass(frozen=True, kw_only=True)
class DonorSelection:
  """Which donor was chosen, and how much the choice was forced by the data.

  ``n_within_epsilon`` is the field to read before trusting a result. If hundreds
  of donors score within noise of the winner, the specific frame grafted is
  arbitrary; the imputation may still be reasonable, but attributing anything to
  *that* frame is not.
  """

  frame_index: int
  metric: str
  metric_value: float
  runner_up_value: float
  selection_margin: float
  n_within_epsilon: int
  n_candidates: int
  n_scored_exactly: int

  def as_dict(self) -> dict:
    return {
      "frame_index": self.frame_index,
      "metric": self.metric,
      "metric_value": self.metric_value,
      "runner_up_value": self.runner_up_value,
      "selection_margin": self.selection_margin,
      "n_within_epsilon": self.n_within_epsilon,
      "n_candidates": self.n_candidates,
      "n_scored_exactly": self.n_scored_exactly,
    }


@dataclass(frozen=True, kw_only=True)
class AlignmentDrift:
  """How well the donor actually superposed on the residues both structures have.

  This is the trustworthiness signal for the grafted coordinates. Nothing directly
  measures whether a grafted residue is in the right place -- the acceptor has no
  coordinate there to compare against. What can be measured is how far the donor
  had to be from the acceptor at the positions they share, and a donor that fits
  the shared residues badly has no claim to be right about the missing ones.

  ``per_residue`` is indexed against the shared positions used for superposition,
  in ascending residue order; ``shared_positions`` records which those were.
  """

  rmsd_shared: float
  per_residue: np.ndarray = field(repr=False)
  shared_positions: np.ndarray = field(repr=False)
  median: float
  p95: float
  max: float
  max_position: int

  def as_dict(self, *, include_per_residue: bool = False) -> dict:
    out = {
      "rmsd_shared_a": self.rmsd_shared,
      "drift_median_a": self.median,
      "drift_p95_a": self.p95,
      "drift_max_a": self.max,
      "drift_max_position": self.max_position,
      "n_shared_residues": int(self.shared_positions.size),
    }
    if include_per_residue:
      out["per_residue_drift_a"] = self.per_residue.tolist()
      out["shared_positions"] = self.shared_positions.tolist()
    return out


@dataclass(frozen=True, kw_only=True)
class ImputeResult:
  """A completed structure plus everything needed to judge it.

  ``coordinates`` is always the full reference-length array. For ``mask`` it is the
  acceptor unchanged, still carrying NaN at unobserved positions -- that method
  deliberately invents nothing, and a caller that ignores ``imputed_mask`` and
  feeds this straight into a distance calculation should get NaN rather than a
  plausible number.
  """

  coordinates: np.ndarray = field(repr=False)
  imputed_mask: np.ndarray = field(repr=False)
  method: str
  selection: DonorSelection | None = None
  drift: AlignmentDrift | None = None
  ensemble_spread: np.ndarray | None = field(default=None, repr=False)
  provenance: dict = field(default_factory=dict, repr=False)

  @property
  def n_imputed(self) -> int:
    return int(self.imputed_mask.sum())
