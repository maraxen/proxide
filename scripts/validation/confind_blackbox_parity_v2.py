#!/usr/bin/env python3
"""Entry point for the v2 parity pre-registration (debt #3217).

`bth run` binds the sidecar adjacent to the script by name, and has no flag to name
another. This file exists so that a tracked run binds
confind_blackbox_parity_v2.bth.toml. Running confind_blackbox_parity.py directly binds
the historical v1 sidecar, whose result schema the current harness no longer emits.
All logic lives in confind_blackbox_parity.py; arguments pass through unchanged.

  bth run --project-slug proxide --output-paths <out.json> -- \\
    uv run --no-sync python scripts/validation/confind_blackbox_parity_v2.py --out <out.json> ...
"""

import importlib.util
import sys
from pathlib import Path


def _load_harness():
  path = Path(__file__).resolve().with_name("confind_blackbox_parity.py")
  spec = importlib.util.spec_from_file_location("confind_blackbox_parity", path)
  if spec is None or spec.loader is None:
    raise ImportError(f"cannot load parity harness from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


if __name__ == "__main__":
  sys.exit(_load_harness().main())
