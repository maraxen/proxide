---
name: 260630-jaccard-output-format-debt
description: Tech debt — proxide-jaccard's dense symmetric .npy output is an MVP choice; revisit for memory/IO efficiency once matrix sizes and JAX consumption patterns are known
metadata:
  type: project
---

`proxide-jaccard` (see [[proxide-jaccard-spec]]) currently writes its n×n distance matrix as a dense, fully-mirrored symmetric f32 `.npy` file. This was an explicit MVP tradeoff made during the 260630 design conversation: simplest possible `jnp.load`/`np.load` ergonomics on the consuming side, at the cost of 2x the memory/IO of a condensed upper-triangle representation, and no exploitation of sparsity if the requested accession set ever includes near-duplicate or highly divergent clusters.

**Why:** Rust owns the full read→compute→write pipeline (`io.rs`-equivalent is internal to the crate), so the on-disk/in-memory format is free to change later without touching the compute kernel (`distance.rs`/`matrix.rs`) or the parquet ingestion path (`sketch.rs`) at all — only `output.rs` and the JAX-side loader need to move together.

**How to apply:** Before scaling this beyond MVP usage (large n, e.g. hundreds-to-thousands of accessions per matrix, or repeated batch runs feeding a training pipeline), reconsider:

1. **Condensed upper-triangle vector** (scipy `squareform` convention, length n·(n-1)/2) — halves memory/IO; needs a `squareform` step (numpy has one; trivial in JAX too) before most consumers can use it directly.
2. **Chunked/memory-mapped output** for n large enough that the dense matrix doesn't comfortably fit in memory on the JAX-consuming side.
3. **Sparse representation** if the actual access pattern only ever needs a threshold-filtered neighbor graph rather than the full dense matrix (e.g. "within distance 0.2 of accession X") — would change the API shape more substantially (a graph/edge-list output mode alongside, not instead of, the dense path).

Don't undertake any of this speculatively — wait until there's a real matrix-size or pipeline-latency complaint, since the format is now isolated to `output.rs` and cheap to swap.
