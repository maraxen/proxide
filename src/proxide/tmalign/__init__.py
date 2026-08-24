"""TM-align pairwise structural alignment.

Thin re-export of the Rust `_proxider` bindings (`proxide-tmalign` crate). See each
function's docstring for the input array shapes and the returned shapes.

Two entry points, and the choice between them is about whether the residue
correspondence is known:

`tm_align` derives the correspondence via the full five-strategy seeding search. Use it
for genuinely unrelated inputs -- different constructs, different lengths, unknown
residue ordering.

`tm_scores_fixed_correspondence` takes the correspondence to be the identity and scores
one query against a whole stack of candidates. Use it for frames of a single MD
trajectory, where the correspondence is already known and rediscovering it per frame
costs far more than the scoring it feeds.
"""

from proxide._proxider import (  # ty: ignore[unresolved-import]
    tm_align,
    tm_scores_fixed_correspondence,
)

__all__ = ["tm_align", "tm_scores_fixed_correspondence"]
