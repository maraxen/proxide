"""Geometry utilities and transformations.

Experimental: not covered by proxide's stability guarantee; may change in a minor release.
"""

from proxide.geometry import contacts
from proxide.geometry.contacts import (
    InterfaceContactResult,
    compute_interface_contacts,
    compute_interface_contacts_batch,
)

__all__ = [
    "contacts",
    "InterfaceContactResult",
    "compute_interface_contacts",
    "compute_interface_contacts_batch",
]
