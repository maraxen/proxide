"""Bridge between PrxteinMPNN data structures and JAX MD arrays.

Note: This module has been refactored. Core logic is now in `priox.md.bridge`.
"""

from priox.md.bridge.core import parameterize_system
from priox.md.bridge.types import SystemParams

__all__ = ["parameterize_system", "SystemParams"]
