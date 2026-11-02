"""Geometry utilities and transformations.

Experimental: not covered by proxide's stability guarantee; may change in a minor release.
"""

from proxide.geometry import contacts
from proxide.geometry.contacts import (
  InterfaceContactResult,
  aligned_contact_overlap,
  associates,
  compute_buried_sasa,
  compute_interface_contacts,
  compute_interface_contacts_batch,
  compute_shrake_rupley,
  contact_jaccard,
  contacts_and_dock,
  contacts_and_dock_batch,
  interface_counts,
  load_aligned_pairs,
  screen_self_association,
)

__all__ = [
  "contacts",
  "InterfaceContactResult",
  "aligned_contact_overlap",
  "associates",
  "compute_buried_sasa",
  "compute_interface_contacts",
  "compute_interface_contacts_batch",
  "compute_shrake_rupley",
  "contact_jaccard",
  "contacts_and_dock",
  "contacts_and_dock_batch",
  "interface_counts",
  "load_aligned_pairs",
  "screen_self_association",
]
