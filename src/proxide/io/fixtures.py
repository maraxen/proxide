"""Generic nested tensor-bundle I/O for parity fixtures (npz + optional JSON meta).

Used by domain libraries (e.g. plegadx RF3 E2E) to freeze / reload nested dicts of
arrays without vendor-specific schema. Schema asserts stay in the consumer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ArrayLike = np.ndarray | Any


def _to_numpy(value: ArrayLike) -> np.ndarray:
  if isinstance(value, np.ndarray):
    return value
  if hasattr(value, "detach") and hasattr(value, "cpu"):
    return value.detach().cpu().numpy()
  if hasattr(value, "__array__"):
    return np.asarray(value)
  raise TypeError(f"Cannot convert {type(value)!r} to numpy array")


def flatten_tensor_dict(
  nested: Mapping[str, Any],
  *,
  prefix: str = "",
  sep: str = "/",
) -> dict[str, np.ndarray]:
  """Flatten a nested dict of arrays into ``prefix/key/...`` leaf keys."""
  out: dict[str, np.ndarray] = {}
  for key, value in nested.items():
    path = f"{prefix}{sep}{key}" if prefix else str(key)
    if isinstance(value, Mapping):
      out.update(flatten_tensor_dict(value, prefix=path, sep=sep))
    else:
      out[path] = _to_numpy(value)
  return out


def unflatten_tensor_dict(
  flat: Mapping[str, np.ndarray],
  *,
  prefix: str = "",
  sep: str = "/",
) -> dict[str, Any]:
  """Rebuild a nested dict from flattened ``prefix/key/...`` leaf keys.

  If ``prefix`` is set, only keys starting with ``prefix + sep`` (or equal to
  ``prefix``) are included; the prefix is stripped from the result.
  """
  root: dict[str, Any] = {}
  prefix_with_sep = f"{prefix}{sep}" if prefix else ""

  for key, value in flat.items():
    if prefix:
      if key == prefix:
        # Single leaf stored under exact prefix — unusual; treat as root value.
        return {"_": value}
      if not key.startswith(prefix_with_sep):
        continue
      rel = key[len(prefix_with_sep) :]
    else:
      rel = key

    parts = rel.split(sep)
    cursor: dict[str, Any] = root
    for part in parts[:-1]:
      nxt = cursor.get(part)
      if not isinstance(nxt, dict):
        nxt = {}
        cursor[part] = nxt
      cursor = nxt
    cursor[parts[-1]] = value
  return root


def save_tensor_bundle(
  npz_path: Path | str,
  tensors: Mapping[str, Any],
  *,
  meta_path: Path | str | None = None,
  meta: Mapping[str, Any] | None = None,
  flatten: bool = True,
  prefix: str = "",
) -> Path:
  """Write a compressed npz of arrays; optionally write JSON metadata beside it."""
  path = Path(npz_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  flat = flatten_tensor_dict(tensors, prefix=prefix) if flatten else {
    k: _to_numpy(v) for k, v in tensors.items()
  }
  np.savez_compressed(path, **flat)  # ty: ignore[invalid-argument-type]

  if meta is not None:
    mpath = Path(meta_path) if meta_path is not None else path.with_suffix(".meta.json")
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(dict(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return path


def load_tensor_bundle(
  npz_path: Path | str,
  *,
  meta_path: Path | str | None = None,
  unflatten: bool = False,
  prefix: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Load npz arrays and optional JSON meta. Optionally unflatten nested keys."""
  path = Path(npz_path)
  data = np.load(path, allow_pickle=False)
  flat = {k: data[k] for k in data.files}
  tensors: dict[str, Any] = (
    unflatten_tensor_dict(flat, prefix=prefix) if unflatten else flat
  )

  meta: dict[str, Any] = {}
  mpath = Path(meta_path) if meta_path is not None else path.with_suffix(".meta.json")
  if mpath.is_file():
    meta = json.loads(mpath.read_text(encoding="utf-8"))
  return tensors, meta


def assert_bundle_keys(
  bundle: Mapping[str, Any],
  required: Sequence[str],
  *,
  flat: bool = True,
) -> None:
  """Raise ``KeyError`` if any required key is missing from a (flat) bundle."""
  if flat:
    keys = set(bundle.keys())
  else:
    keys = set(flatten_tensor_dict(bundle).keys())
  missing = [k for k in required if k not in keys]
  if missing:
    raise KeyError(f"tensor bundle missing required keys: {missing}")
