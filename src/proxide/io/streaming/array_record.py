"""Grain data source for loading preprocessed array_record files with physics features."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, SupportsIndex

import grain.python as grain
import jax.numpy as jnp
import msgpack
import msgpack_numpy as m
from array_record.python.array_record_module import (  # type: ignore[unresolved-import]
  ArrayRecordReader,
)

from proxide.chem.residues import atom_order
from proxide.core.containers import Protein

m.patch()

logger = logging.getLogger(__name__)


def _normalize_index_entry(
  entry: int | dict[str, Any], default_split: str = "all"
) -> dict[str, Any]:
  """Normalize an index entry to the modern format.

  Supports multiple index formats:
  - Legacy integer: `42` -> `{"idx": [42], "set": "all"}`
  - Modern dict: `{"idx": [42], "set": "train"}` -> unchanged
  - Extended dict: `{"idx": [42], "tag": "high_quality", ...}` -> unchanged

  Args:
      entry: The index entry (integer or dict)
      default_split: Default split to assign if entry is an integer

  Returns:
      Normalized dictionary with at least "idx" key

  """
  if isinstance(entry, int):
    return {"idx": [entry], "set": default_split}

  if isinstance(entry, dict):
    # Ensure idx is a list
    if "idx" not in entry:
      # Maybe it's a direct index without idx key
      return {"idx": [0], "set": default_split, **entry}
    if isinstance(entry["idx"], int):
      entry = {**entry, "idx": [entry["idx"]]}
    return entry

  msg = f"Unsupported index entry type: {type(entry)}"
  raise TypeError(msg)


def _detect_index_format(full_index: dict[str, Any]) -> str:
  """Detect the format of an index file.

  Returns:
      "legacy" if all values are integers
      "modern" if all values are dicts with "idx" and "set" keys
      "extended" if all values are dicts with "idx" but not necessarily "set"

  """
  if not full_index:
    return "modern"

  first_value = next(iter(full_index.values()))

  if isinstance(first_value, int):
    return "legacy"
  if isinstance(first_value, dict):
    if "set" in first_value:
      return "modern"
    return "extended"

  return "unknown"


class ArrayRecordDataSource(grain.RandomAccessDataSource):
  """Grain data source for preprocessed protein structures in array_record format.

  This source reads from array_record files created by the PQR preprocessing pipeline,
  which include precomputed physics features (electrostatic forces projected onto
  backbone frame).

  Supports multiple index formats:
  - Legacy: `{"protein_id": 0, ...}` (integers)
  - Modern: `{"protein_id": {"idx": [0], "set": "train"}, ...}`
  - Extended: `{"protein_id": {"idx": [0], "tag": "value", ...}, ...}`

  Attributes:
      array_record_path: Path to the .array_record file
      index: Dictionary mapping protein_id to record index
      reader: ArrayRecordReader instance
      _length: Total number of records

  Example:
      >>> source = ArrayRecordDataSource(
      ...     "data/preprocessed/train.array_record",
      ...     "data/preprocessed/train.index.json"
      ... )
      >>> protein = source[0]  # Returns Protein with physics_features
      >>> print(protein.physics_features.shape)  # (n_residues, 5)

      >>> # With custom filter function
      >>> source = ArrayRecordDataSource(
      ...     "data/preprocessed/data.array_record",
      ...     "data/preprocessed/data.index.json",
      ...     filter_fn=lambda pid, entry: entry.get("quality") == "high",
      ... )

  """

  def __init__(
    self,
    array_record_path: str | Path,
    index_path: str | Path,
    split: str = "train",
    filter_fn: Callable[[str, dict[str, Any]], bool] | None = None,
    features: Sequence[str] | None = None,
  ) -> None:
    """Initialize the array_record data source.

    Args:
        array_record_path: Path to the array_record file
        index_path: Path to the JSON index file mapping protein_id -> index
        split: Data split to load ("train", "valid", "test", "inference", "all")
        filter_fn: Optional custom filter function. Takes (protein_id, entry_dict)
            and returns True if the entry should be included. If provided, this
            takes precedence over split-based filtering.
        features: Optional list of features to load. If None, loads all available features.
            Standard features like 'coordinates', 'aatype', 'mask', etc are always loaded
            if they exist, unless this list is provided and they are excluded
            (though core features are usually required).
            If provided, only these keys will be extracted from the record.

    Raises:
        FileNotFoundError: If array_record or index file doesn't exist
        ValueError: If index file is malformed

    """
    super().__init__()
    self.array_record_path = Path(array_record_path)
    self.index_path = Path(index_path)
    self.features = set(features) if features is not None else None

    if not self.array_record_path.exists():
      msg = f"Array record file not found: {self.array_record_path}"
      raise FileNotFoundError(msg)

    # Initialize reader early to get num_records if needed for "all" or "inference" split
    self.reader = ArrayRecordReader(str(self.array_record_path))
    self._length = self.reader.num_records()

    if not self.index_path.exists():
      if split in ["inference", "all"]:
        logger.info(
          "Index file not found at %s, but split is '%s'. "
          "Assuming all records in array_record file for this split.",
          self.index_path,
          split,
        )
        full_index = {f"record_{i}": {"idx": [i], "set": split} for i in range(self._length)}
      else:
        msg = f"Index file not found: {self.index_path}"
        raise FileNotFoundError(msg)
    else:
      with self.index_path.open("r") as f:
        full_index = json.load(f)

    # Detect index format and normalize entries
    index_format = _detect_index_format(full_index)
    if index_format == "legacy":
      logger.info(
        "Detected legacy index format (integer values). "
        "All records will be included for split '%s'.",
        split,
      )
      # For legacy format, include all records regardless of split
      # (since there's no split information in the index)
      full_index = {
        pid: _normalize_index_entry(entry, default_split=split) for pid, entry in full_index.items()
      }

    # Filter index by split or custom filter_fn
    self.index = {}
    for pid, entry in full_index.items():
      normalized = _normalize_index_entry(entry)

      if filter_fn is not None:
        # Use custom filter function
        if filter_fn(pid, normalized):
          self.index[pid] = normalized
      elif split in ["inference", "all"]:
        # Include all records for inference/all splits
        self.index[pid] = normalized
      elif normalized.get("set") == split:
        # Filter by split field
        self.index[pid] = normalized

    self._record_indices = []
    for entry in self.index.values():
      self._record_indices.extend(entry["idx"])

    if len(self._record_indices) == 0 and split in ["inference", "all"]:
      logger.info(
        "No records found for split '%s' in index, but split is '%s'. "
        "Falling back to using all records in array_record file.",
        split,
        split,
      )
      self._record_indices = list(range(self._length))

    logger.info(
      "Loaded %d records for split '%s' from %s",
      len(self._record_indices),
      split,
      self.index_path,
    )

    self.reader = ArrayRecordReader(str(self.array_record_path))
    self._length = self.reader.num_records()

    if len(self._record_indices) == 0:
      logger.warning("No records found for split '%s'", split)

    logger.info(
      "Loaded array_record data source with %d proteins from %s",
      self._length,
      self.array_record_path,
    )

  def __len__(self) -> int:
    """Return the total number of records."""
    return len(self._record_indices)

  def __getitem__(self, index: SupportsIndex) -> Protein:  # type: ignore[override]
    """Load and deserialize a protein structure.

    Args:
        index: Integer index of the record to load

    Returns:
        Protein with all fields including physics_features

    Raises:
        IndexError: If index is out of range
        RuntimeError: If deserialization fails

    """
    idx = int(index)
    if not 0 <= idx < len(self):
      msg = f"Index {idx} out of range [0, {len(self)})"
      raise IndexError(msg)

    try:
      # Read record
      record_idx = self._record_indices[idx]
      record_bytes = self.reader.read(record_idx, record_idx + 1)[0]

      # Deserialize
      record_data = msgpack.unpackb(record_bytes, raw=False)

      # Convert to Protein
      return self._record_to_protein(record_data)

    except Exception as e:
      msg = f"Failed to load record at index {idx}"
      logger.exception(msg)
      raise RuntimeError(msg) from e

  def _should_load(self, feature_name: str) -> bool:
    """Check if a feature should be loaded."""
    if self.features is None:
      return True
    return feature_name in self.features

  def _record_to_protein(self, record: dict[str, Any]) -> Protein:
    """Convert deserialized record to Protein.

    Args:
        record: Dictionary with protein data

    Returns:
        Protein instance

    """
    # Extract all fields, converting to appropriate types
    # Use .get() for optional fields to ensure robustness

    # Basic structure (always required usually, but can be checked if needed)
    # We assume coordinates/aatype are core.
    if "coordinates" in record:
      coordinates = jnp.array(record["coordinates"], dtype=jnp.float32)
      n_residues = coordinates.shape[0]
    else:
      # Should not happen for valid records
      n_residues = 0
      coordinates = jnp.zeros((0, 3), dtype=jnp.float32)

    if "aatype" in record:
      aatype = jnp.array(record["aatype"], dtype=jnp.int8)
    else:
      aatype = jnp.zeros((n_residues,), dtype=jnp.int8)

    # Defaults for physics features if missing
    default_physics = jnp.zeros((n_residues, 5), dtype=jnp.float32)

    # Defaults for full atomic data if missing
    full_coords = None
    if self._should_load("full_coordinates"):
      full_coords = record.get("full_coordinates")
      if full_coords is not None:
        full_coords = jnp.array(full_coords, dtype=jnp.float32)

    # Need to handle charges/radii only if full_coordinates implied or requested?
    # Usually charges/radii go with full_coords.
    charges = None
    if self._should_load("charges"):
      raw_charges = record.get("charges")
      if raw_charges is not None:
        charges = jnp.array(raw_charges, dtype=jnp.float32)

    radii = None
    if self._should_load("radii"):
      raw_radii = record.get("radii")
      if raw_radii is not None:
        radii = jnp.array(raw_radii, dtype=jnp.float32)

    atom_mask_2d = jnp.array(
      record.get("atom_mask", jnp.ones((n_residues, 37), dtype=jnp.float32)),
      dtype=jnp.float32,
    )

    physics_features = None
    if self._should_load("physics_features"):
      raw_phys = record.get("physics_features")
      if raw_phys is not None:
        physics_features = jnp.array(raw_phys, dtype=jnp.float32)
      elif self.features is None:  # Legacy behavior: default to zeros
        physics_features = default_physics

    return Protein(
      coordinates=coordinates,
      aatype=aatype,
      atom_mask=atom_mask_2d,
      mask=atom_mask_2d[:, atom_order["CA"]],
      one_hot_sequence=jnp.eye(21)[aatype],
      residue_index=jnp.array(
        record.get("residue_index", jnp.arange(n_residues)),
        dtype=jnp.int32,
      ),
      chain_index=jnp.array(
        record.get("chain_index", jnp.zeros(n_residues)),
        dtype=jnp.int32,
      ),
      dihedrals=None,
      # Full atomic data
      full_coordinates=full_coords,
      charges=charges,
      radii=radii,
      # Physics features
      physics_features=physics_features,
      elements=None,
      atom_names=None,
    )

  def close(self) -> None:
    """Close the ArrayRecordReader."""
    if hasattr(self, "reader"):
      self.reader.close()

  def __del__(self) -> None:
    """Cleanup when object is destroyed."""
    self.close()


def get_protein_by_id(
  source: ArrayRecordDataSource,
  protein_id: str,
) -> Protein | None:
  """Retrieve a protein by its ID.

  Args:
      source: ArrayRecordDataSource instance
      protein_id: Protein identifier

  Returns:
      Protein if found, None otherwise

  Example:
      >>> source = ArrayRecordDataSource("train.array_record", "train.index.json")
      >>> protein = get_protein_by_id(source, "1UBQ")

  """
  if protein_id not in source.index:
    logger.warning("Protein ID '%s' not found in index", protein_id)
    return None

  if protein_id not in source.index:
    logger.warning("Protein ID '%s' not found in index for this split", protein_id)
    return None

  # Get the first record index for this protein
  # Note: This assumes we want the first conformation if multiple exist
  file_record_index = source.index[protein_id]["idx"][0]

  # We need to find where this file_record_index is in our filtered _record_indices list
  # This is O(N) which is slow, but get_protein_by_id is mainly for debugging
  try:
    dataset_index = source._record_indices.index(file_record_index)  # noqa: SLF001
    return source[dataset_index]
  except ValueError:
    logger.warning(
      "Protein ID '%s' found in index but its records are not in this split",
      protein_id,
    )
    return None
