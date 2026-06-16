---
session_id: 98a30224
topic: Design the proxide-frag crate: backbone RMSD substructure search (MASTER-style) — data structures, search algorithm, API surface, and integration with proxide
task_type: constrained-technical
winner: Pre-centering at build time + full 3D coordinates stored + phantom-typed Fragment&lt;Centered&gt;/Fragment&lt;Raw&gt; to encode centroid state in the type system. Enables the fast inner-product RMSD form (RMSD² = (||A||² + ||B||²)/N − 2·max_σ(SVD(AᵀB))/N), preserves full geometry for downstream use (alignment transform recoverable), and eliminates the centering footgun via compile-time enforcement. Database stores pre-centered coords + centroid vector per fragment so the original geometry can be reconstructed.
created_at: 2026-06-02T14:58:34.071912+00:00
---

# Brainstorm: Design the proxide-frag crate: backbone RMSD substructure search (MASTER-style) — data structures, search algorithm, API surface, and integration with proxide

## Problem Frame
Fixed constraints:
1. Rust crate in the proxide Cargo workspace — no FFI to C MASTER, pure Rust
2. Backbone atoms are N, CA, C, O as already defined in proxide-confind/coords.rs — we build on that representation, not replace it
3. Must be a library crate — no CLI requirement at this stage
4. RMSD must be exact (or exact within floating-point tolerance) — approximate search is only acceptable as a pre-filter, not the final answer
5. Performance is load-bearing: the search will be called over databases of 10k–100k+ fragments in production use

Negotiable:
1. Database format: flat in-memory array vs. disk-indexed (mmap'd binary) vs. hybrid
2. Whether fragment length is fixed or variable per query
3. Pre-filter strategy before full RMSD computation (distance bounds, Gram matrix fingerprints, etc.)
4. Threading model: rayon parallel search vs. caller-controlled parallelism
5. Whether database build and search are separate structs/steps or unified
6. Exact API surface: what the query type looks like, what a match result contains

## Idea Pool
- [user] **Processes:**
- [user] Fragment extraction: from a protein structure, extract a contiguous window of N residues, collecting N/CA/C/O coordinates → 12 floats per residue
- [user] Superposition: Kabsch optimal superposition of query fragment onto database fragment → RMSD scalar
- [user] Database build: ingest protein structures, extract all fragments of length N, store with source labels (PDB ID + chain + residue range)
- [user] Query: given query fragment + RMSD threshold ε, return all database fragments with RMSD ≤ ε after superposition
- [user] Pre-filtering: fast elimination before Kabsch — reject candidates where RMSD lower bound > ε
- [user] **Events:**
- [user] Fragment extracted (coordinates ready for storage or query)
- [user] Pre-filter pass/fail (candidate survives to Kabsch or is dropped)
- [user] Kabsch RMSD computed
- [user] Match emitted (RMSD ≤ ε, source label attached)
- [user] Database serialized to/from disk
- [user] **Goals:**
- [user] Correct RMSD (exact Kabsch, no false negatives)
- [user] High throughput over 10^5–10^6 database fragments
- [user] Ergonomic build/query API
- [user] Reuse proxide-confind backbone coordinate types
- [user] Database survives process restart (disk persistence)
- [user] **States:**
- [user] `Fragment`: N×4×3 float array, optionally pre-centered
- [user] `FragmentDb`: collection of fragments with source labels, optionally indexed
- [user] `SearchResult`: RMSD value + source label (+ optional alignment transform)
- [user] Search in progress: current position, accumulated matches
- [user] **Assumptions under tension:**
- [user] Fixed-length database (simplest, enables vectorized search) vs. variable-length (more flexible, harder to batch)
- [user] Pre-centering fragments at build time vs. at query time (build-time is free, enables faster inner product tricks)
- [user] Storing full 3D coordinates vs. derived invariants (distance matrices, Gram matrices) — invariants are rotation-free but can't reconstruct geometry

## Decision Log
- [ACCEPT] Fixed-length database (all fragments same N): Enables contiguous float arrays and vectorized RMSD batches. Variable-length search can be handled by maintaining multiple fixed-length databases (one per fragment size), which is how MASTER itself works. This is not a fundamental limitation.
- [ACCEPT] Separate FragmentDbBuilder and FragmentDb structs (build phase separate from search phase): Builder pattern ensures the database is immutable after build — no accidental fragment insertions during a running search. Also allows build-time pre-processing (centering, norms, sorting) as a distinct step.
- [ACCEPT] Rayon-parallel search over database fragments: Search is embarrassingly parallel — each database fragment is independent. Rayon par_iter maps directly onto this. Caller-controlled parallelism is more complex and offers no benefit when the database is large enough that fragment-level parallelism dominates.
- [DEFER] Disk persistence / mmap'd binary database format: Phase 2 feature. In-memory first (Vec of pre-centered arrays). Serialization to disk via bincode or a flat binary format can be added once the search correctness and API are validated. Don't design the serialization format before the in-memory layout is proven.
- [DEFER] Pre-filter before Kabsch (distance bounds, RMSD lower bound): Optimization pass only. Implement correct brute-force Kabsch first; profile against real databases; add pre-filter (e.g., distance geometry lower bound) only if profiling shows it's needed. Premature pre-filter adds code complexity without confirmed benefit at target database sizes.

## Assumptions

## TBDs

## Pre-mortem Record
**User:** _not recorded_
**AI:** _not recorded_

## Acceptance Criteria
**Given** Fixed constraints:
1. Rust crate in the proxide Cargo workspace — no FFI to C MASTER, pure Rust
2. Backbone atoms are N, CA, C, O as already defined in proxide-confind/coords.rs — we build on that representation, not replace it
3. Must be a library crate — no CLI requirement at this stage
4. RMSD must be exact (or exact within floating-point tolerance) — approximate search is only acceptable as a pre-filter, not the final answer
5. Performance is load-bearing: the search will be called over databases of 10k–100k+ fragments in production use

Negotiable:
1. Database format: flat in-memory array vs. disk-indexed (mmap'd binary) vs. hybrid
2. Whether fragment length is fixed or variable per query
3. Pre-filter strategy before full RMSD computation (distance bounds, Gram matrix fingerprints, etc.)
4. Threading model: rayon parallel search vs. caller-controlled parallelism
5. Whether database build and search are separate structs/steps or unified
6. Exact API surface: what the query type looks like, what a match result contains
**When** implementing Pre-centering at build time + full 3D coordinates stored + phantom-typed Fragment&lt;Centered&gt;/Fragment&lt;Raw&gt; to encode centroid state in the type system. Enables the fast inner-product RMSD form (RMSD² = (||A||² + ||B||²)/N − 2·max_σ(SVD(AᵀB))/N), preserves full geometry for downstream use (alignment transform recoverable), and eliminates the centering footgun via compile-time enforcement. Database stores pre-centered coords + centroid vector per fragment so the original geometry can be reconstructed.
**Then**
  - [ ] _add specific measurable criteria_
