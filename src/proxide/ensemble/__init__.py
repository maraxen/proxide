"""Complete partially-observed structures from a conformational ensemble.

A structure that observes only part of a reference residue set -- a cryo-EM model with
unresolved loops, or a construct built from one and inheriting its gaps -- cannot be
featurised against that full reference. This module fills the missing positions from a
donor ensemble (typically MD frames), records which donor supplied them, and measures
how well that donor agreed at the positions both structures share.

    from proxide.ensemble import ImputeConfig, impute_frame

    result = impute_frame(
        acceptor,                 # (n_res, 3), NaN where unobserved
        observed,                 # (n_res,) bool
        donors,                   # (n_donors, n_res, 3), fully observed
        ImputeConfig(method="nearest_frame", metric="rmsd"),
    )
    result.coordinates            # (n_res, 3), complete
    result.imputed_mask           # which positions were grafted
    result.provenance             # serialisable record of how

The default method is ``reject``. Completing a structure changes what everything
downstream measures, so it has to be asked for explicitly -- a coverage gate that a
default silently satisfies is not a gate.

Before relying on a result, read two numbers from the provenance:

``n_within_epsilon`` -- how many donors tied with the winner. If hundreds did, the
particular frame grafted is arbitrary, and while the imputation may still be
reasonable, attributing anything to that frame is not.

``drift_max_a`` -- the largest distance between donor and acceptor at a shared
residue after superposition. Nothing can directly verify a grafted coordinate, since
the acceptor has no value there to compare against; a donor that fits the shared
residues badly has no claim to be right about the missing ones.
"""

from ._config import (
  METHODS,
  METRICS,
  AlignmentDrift,
  DonorSelection,
  ImputeConfig,
  ImputeResult,
)
from ._impute import impute_frame, select_donors, superpose

__all__ = [
  "METHODS",
  "METRICS",
  "AlignmentDrift",
  "DonorSelection",
  "ImputeConfig",
  "ImputeResult",
  "impute_frame",
  "select_donors",
  "superpose",
]
