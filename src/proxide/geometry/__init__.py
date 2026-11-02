"""Geometry utilities and transformations."""

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
