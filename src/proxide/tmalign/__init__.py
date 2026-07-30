"""TM-align pairwise structural alignment.

Thin re-export of the Rust `_proxider.tm_align` binding (`proxide-tmalign`
crate). See that function's docstring for the input array shapes and the
returned dict shape.
"""

from proxide._proxider import tm_align  # type: ignore[unresolved-import]

__all__ = ["tm_align"]
