"""MinHash Jaccard/containment distance matrices over sourmash-style sketch parquets.

Thin re-export of the Rust `_proxider.jaccard_distance_matrix` binding
(`proxide-jaccard` crate). See that function's docstring for the parquet schema,
the `metric` options, and the returned dict shape.
"""

from proxide._proxider import jaccard_distance_matrix  # type: ignore[unresolved-import]

__all__ = ["jaccard_distance_matrix"]
