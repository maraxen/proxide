"""MD integration module.

NOTE: Most MD parameterization has been moved to Rust.

Use `priox.io.parsing.rust.parse_structure(..., spec)` with
`spec.parameterize_md = True` for Rust-based MD parameterization.

For GBSA, water models, and CMAP, use the Rust functions directly via:
  - oxidize.assign_mbondi2_radii()
  - oxidize.get_water_model()
  - oxidize.compute_bicubic_params()
"""

from proxide.md.bridge.types import SystemParams
from proxide.md.bridge.utils import assign_masses


def parameterize_system(*args, **kwargs):  # noqa: ANN002, ANN003
    """DEPRECATED: Use Rust-based parameterization via parse_structure."""
    msg = (
        "parameterize_system has been removed. "
        "Use priox.io.parsing.rust.parse_structure with spec.parameterize_md=True."
    )
    raise NotImplementedError(msg)


__all__ = [
    "SystemParams",
    "assign_masses",
    "parameterize_system",
]
