"""Parser registry and custom exceptions."""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Sequence
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from priox.core.containers import ProteinStream

# --- Exception Hierarchy ---

class PrioxError(Exception):
    """Base class for all Priox exceptions."""


class ParsingError(PrioxError):
    """Raised when an error occurs during parsing."""


class FormatNotSupportedError(PrioxError):
    """Raised when a file format is not supported."""


# --- Registry ---

# Parser function signature:
# (file_path: str | pathlib.Path | IO[str], chain_id: str | Sequence[str] | None, **kwargs) -> ProteinStream
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
