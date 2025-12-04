"""Data loading utilities for protein structure data."""

import logging
import pathlib
from collections.abc import Sequence
from typing import IO, Any, SupportsIndex

import grain.python as grain
import numpy as np

from functools import partial
from priox.core.containers import ProteinTuple
from priox.io.parsing.foldcomp import (
  FoldCompDatabase,
)
from priox.io.streaming.array_record import ArrayRecordDataSource
from priox.ops import prefetch as prefetch_autotune
from priox.ops import transforms as operations

from .processing import frame_iterator_from_inputs

logger = logging.getLogger(__name__)


def _is_frame_valid(frame: ProteinTuple) -> tuple[bool, str]:
  """Check if a protein frame is valid."""
  if len(frame.aatype) == 0:
    return False, "Empty structure"
  if frame.coordinates.shape[0] != len(frame.aatype):
    return False, "Shape mismatch between coordinates and aatype"
  if np.isnan(frame.coordinates).any():
    return False, "NaN values in coordinates"
  return True, ""


class ProteinDataSource(grain.RandomAccessDataSource):
  """Implements a Grain DataSource for streaming protein structure frames."""

  def __init__(
    self,
    inputs: Sequence[str | pathlib.Path | IO[str]],
    parse_kwargs: dict[str, Any] | None = None,
    foldcomp_database: FoldCompDatabase | None = None,
  ) -> None:
    """Initialize the data source by preparing the frame iterator.

    Args:
        inputs: A sequence of input sources (file paths, file-like objects, etc.).
        parse_kwargs: Optional keyword arguments to pass to the parsing function.
        foldcomp_database: An optional FoldCompDatabase for resolving FoldComp IDs.

    """
    super().__init__()
    self.inputs = inputs
    self.parse_kwargs = parse_kwargs or {}
    self.foldcomp_database = foldcomp_database
    self.frames = []
    self.skipped_frames = []
    for frame in frame_iterator_from_inputs(
      self.inputs,
      self.parse_kwargs,
      self.foldcomp_database,
    ):
      is_valid, reason = _is_frame_valid(frame)
      if is_valid:
        self.frames.append(frame)
      else:
        logger.warning("Skipping invalid frame from %s: %s", frame.source, reason)
        self.skipped_frames.append({"source": frame.source, "reason": reason})

    self._length = len(self.frames)

  def __len__(self) -> int:
    """Return the total number of frames available."""
    return self._length

  def __getitem__(self, index: SupportsIndex) -> ProteinTuple:  # type: ignore[override]
    """Return the ProteinTuple at the specified index.

    Args:
        index (SupportsIndex): The index of the item to retrieve.

    Returns:
        ProteinTuple: The protein structure frame at the specified index.

    Raises:
        IndexError: If the index is out of range.

    """
    idx = int(index)
    if not 0 <= idx < len(self):
      msg = f"Attempted to access index {idx}, but valid indices are 0 to {len(self) - 1}."
      raise IndexError(msg)
    return self.frames[idx]


def create_protein_dataset(  # noqa: PLR0913
  inputs: str | pathlib.Path | Sequence[str | pathlib.Path | IO[str]],
  batch_size: int,
  parse_kwargs: dict[str, Any] | None = None,
  foldcomp_database: FoldCompDatabase | None = None,
  pass_mode: str = "intra",  # noqa: S107
  *,
  use_electrostatics: bool = False,
  use_vdw: bool = False,
  estat_noise: Sequence[float] | float | None = None,
  estat_noise_mode: str = "direct",
  vdw_noise: Sequence[float] | float | None = None,
  vdw_noise_mode: str = "direct",
  use_preprocessed: bool = False,
  preprocessed_index_path: str | pathlib.Path | None = None,
  split: str = "train",
  max_length: int | None = 512,
  truncation_strategy: str = "none",
  ram_budget_mb: int = 1024,
  max_workers: int | None = None,
  max_buffer_size: int | None = None,
) -> grain.IterDataset:
  """Construct a high-performance protein data pipeline using Grain.

  Args:
      inputs: A single input (file, PDB ID, etc.) or a sequence of such inputs.
              When use_preprocessed=True, this should be the path to the array_record file.
      batch_size: The number of protein structures to include in each batch.
      parse_kwargs: Optional dictionary of keyword arguments for parsing.
      foldcomp_database: Optional path to a FoldComp database.
      pass_mode: "intra" (default) for normal batching, "inter" for concatenation.
      use_electrostatics: Whether to compute and include electrostatic features.
      use_vdw: Whether to compute and include van der Waals features.
      estat_noise: Noise level(s) for electrostatics.
      estat_noise_mode: Mode for electrostatic noise.
      vdw_noise: Noise level(s) for vdW.
      vdw_noise_mode: Mode for vdW noise.
      use_preprocessed: If True, load from preprocessed array_record files
      preprocessed_index_path: Path to index file (required if use_preprocessed=True)
      split: Data split to load ("train", "valid", "test")
      max_length: Maximum length for truncation/padding.
      truncation_strategy: Strategy for truncation ("none", "random_crop", "center_crop").
      ram_budget_mb: RAM budget in MB for prefetching.
      max_workers: Maximum number of workers for prefetching.
      max_buffer_size: Maximum buffer size for prefetching.

  Returns:
      A Grain IterDataset that yields batches of padded `Protein` objects.

  Example:
      >>> # File-based loading (original)
      >>> ds = create_protein_dataset(
      ...     inputs="data/train/",
      ...     batch_size=8,
      ... )

      >>> # Preprocessed loading (new, faster)
      >>> ds = create_protein_dataset(
      ...     inputs="data/preprocessed/train.array_record",
      ...     batch_size=8,
      ...     use_preprocessed=True,
      ...     preprocessed_index_path="data/preprocessed/train.index.json",
      ... )

  """
  parse_kwargs = parse_kwargs or {}
  # Only add hydrogens/relax if we need physics features
  if "add_hydrogens" not in parse_kwargs:
    parse_kwargs["add_hydrogens"] = use_electrostatics or use_vdw

  if use_preprocessed:
    if preprocessed_index_path is None:
      msg = "preprocessed_index_path is required when use_preprocessed=True"
      raise ValueError(msg)

    if not isinstance(inputs, (str | pathlib.Path)):
      msg = "When use_preprocessed=True, inputs must be a single path to array_record file"
      raise ValueError(msg)

    source = ArrayRecordDataSource(
      array_record_path=inputs,
      index_path=preprocessed_index_path,
      split=split,
    )
    ds = grain.MapDataset.source(source)

  else:
    if not isinstance(inputs, Sequence) or isinstance(inputs, str | pathlib.Path):
      inputs = (inputs,)

    source = ProteinDataSource(
      inputs=inputs,
      parse_kwargs=parse_kwargs,
      foldcomp_database=foldcomp_database,
    )
    ds = grain.MapDataset.source(source)

  if max_length is not None and truncation_strategy != "none":
    ds = ds.map(
      partial(
        operations.truncate_protein,
        max_length=max_length,
        strategy=truncation_strategy,
      ),
    )

  performance_config = prefetch_autotune.pick_performance_config(
    ds=ds,
    ram_budget_mb=ram_budget_mb,
    max_workers=max_workers,
    max_buffer_size=max_buffer_size,
  )

  batch_fn = (
    operations.concatenate_proteins_for_inter_mode
    if pass_mode == "inter"  # noqa: S105
    else partial(
      operations.pad_and_collate_proteins,
      use_electrostatics=use_electrostatics,
      use_vdw=use_vdw,
      estat_noise=estat_noise,
      estat_noise_mode=estat_noise_mode,
      vdw_noise=vdw_noise,
      vdw_noise_mode=vdw_noise_mode,
      max_length=max_length,
    )
  )

  iter_ds = ds.to_iter_dataset(read_options=performance_config.read_options).batch(
    batch_size,
    batch_fn=batch_fn,
  )

  if hasattr(source, "skipped_frames"):
    iter_ds.skipped_frames = source.skipped_frames  # type: ignore[attr-defined]

  return iter_ds
