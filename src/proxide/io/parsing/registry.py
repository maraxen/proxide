"""Parser registry and custom exceptions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from proxide.core.containers import ProteinStream

# --- Exception Hierarchy ---


class ProxideError(Exception):
  """Base class for all Proxide exceptions."""


class ParsingError(ProxideError):
  """Error raised when parsing fails."""


class FormatNotSupportedError(ProxideError):
  """Error raised when file format is not supported."""


# --- Registry ---

# Parser function signature:
# Parser function signature:
# (file_path: str | pathlib.Path | IO[str], chain_id: str | Sequence[str] | None, **kwargs)
#   -> ProteinStream
ParserFunc = Callable[..., "ProteinStream"]

_PARSER_REGISTRY: dict[str, ParserFunc] = {}


def register_parser(formats: list[str]) -> Callable[[ParserFunc], ParserFunc]:
  """Decorator to register a parser function for specific file formats."""

  def decorator(fn: ParserFunc) -> ParserFunc:
    for fmt in formats:
      _PARSER_REGISTRY[fmt] = fn
    return fn

  return decorator


def get_parser(fmt: str) -> ParserFunc | None:
  """Get a parser function for a specific format."""
  return _PARSER_REGISTRY.get(fmt)


def list_supported_formats() -> list[str]:
  """List all supported formats."""
  return sorted(_PARSER_REGISTRY.keys())
