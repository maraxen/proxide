"""Chemical properties utilities."""

from proxide import _oxidize


def assign_masses(atom_names: list[str]) -> list[float]:
  """Assign atomic masses based on element type.

  Uses the Rust implementation for performance.

  Args:
      atom_names: List of atom names.

  Returns:
      List of masses in amu.

  """
  return _oxidize.assign_masses(atom_names)
