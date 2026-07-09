"""Parsing utilities for various protein structure formats."""

from __future__ import annotations

from typing import Any

from proxide.io.parsing.dispatch import load_structure, parse_input
from proxide.io.parsing.registry import (
  FormatNotSupportedError,
  ParserFunc,
  ParsingError,
  ProxideError,
  register_parser,
)

__all__ = [
  "parse_structure",
  "load_structure",
  "parse_input",
  "register_parser",
  "ProxideError",
  "ParsingError",
  "FormatNotSupportedError",
  "ParserFunc",
]


def __getattr__(name: str) -> Any:
  if name == "parse_structure":
    from proxide.io.parsing.backend import parse_structure

    return parse_structure
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
