---
name: arrow-ipc-prototype
description: proxide-jaccard — Arrow IPC + sorted accession index prototype, measured against the real corpus; planus vs flatbuffers vs Arrow IPC tradeoff analysis
metadata:
  type: research
---

Prototype for `proxide-jaccard`'s storage layer, evaluating whether an Arrow IPC file + sorted in-memory accession index beats the current parquet+RowFilter path for point-lookup-heavy queries (the access pattern this crate actually has: an arbitrary, possibly-scattered subset of accessions, not a full-corpus scan).

**Why:** parquet's row-group/page granularity means a scattered point-query still pays to decode every touched group regardless of how few rows in it actually matched (measured earlier in [[proxide-jaccard-spec]]: 63.82s/205MB RSS for 300 accessions scattered across all 20 row groups of the real `signature_index_k31.parquet`, vs 0.16s/57MB for the same count clustered in one group). Arrow's on-disk IPC format *is* its in-memory layout — no decode/parse step — so once an accession's (batch, row) is known, fetching it costs one seek + one (small, tunable-size) batch read, independent of how scattered the overall query is.

**How to apply:** Code lives at `crates/proxide-jaccard/src/ipc_index.rs`, marked experimental/prototype, not wired into the CLI. Run the real-data validation with `cargo run -p proxide-jaccard --release --example ipc_prototype_bench -- <parquet> <out.arrow>` (needs a local copy of `signature_index_k31.parquet` or equivalent schema, e.g. `gdrive:00shared/phylogeny/datasets/signature_index_k31.parquet`).

---

## Measured results (real data: signature_index_k31.parquet, 982,265 rows, 2 row groups / 100,000-row slice converted for this prototype)

| | parquet + RowFilter + row-group pruning (full 982k-row corpus) | Arrow IPC + sorted index (100k-row slice, batch_size=2048) |
|---|---|---|
| Clustered query (n=300, same row group / batch) | 0.16s, 57MB RSS | **0.035s lookup + 0.11s matrix = 0.15s** |
| Scattered query (n=300, spread across whole file) | 63.82s, 205MB RSS, 37% CPU | **1.01s lookup + 0.07s matrix = 1.07s** (**~60x faster**) |
| Conversion (one-time, 100k rows, 2 row groups) | n/a | 5.6-8.6s |
| Disk (same logical data) | ~19.9GB (whole corpus, zstd) | ~27.3GB extrapolated for whole corpus (no compression — see below) |

Correctness: cross-checked against `SketchStore::load_parquet` on the same real accessions — distance matrices and accession ordering match exactly for both query patterns.

**Caveat on the comparison:** the IPC numbers are against a 100k-row slice, not the full 982k-row corpus (kept the prototype memory-safe — this machine has 29GB RAM, much of it already in use, and the full corpus's raw hash data is ~27GB; converting it all into a single process would risk OOM/swap). This doesn't bias the *clustered* number (cost is independent of corpus size — one batch, decoded once, regardless of how many other rows exist elsewhere in the file). It likely *understates* the full-corpus scattered cost somewhat, since 300 accessions scattered across only 49 batches (100k rows / 2048 batch_size) hit fewer distinct batches than the same 300 would across ~480 batches at the same batch_size over the full corpus — worth re-measuring against the full corpus before treating "~60x" as the production number, but directionally the architecture's win is real and large.

**One implementation pitfall worth flagging for posterity:** the first version of `IpcSketchReader::get()` had no batch caching — every single accession lookup called `set_index()` + decoded a fresh batch from scratch, even when consecutive lookups landed in the *same* batch (the common case for clustered queries, where all 300 wanted rows can share one batch). That version measured **8.83s clustered / 6.05s scattered** — *worse* than parquet's optimized path, and counter to the whole point of the prototype. Caching the most-recently-decoded batch (`IpcSketchReader::cached_batch`) fixed this. Documented here because it's the kind of regression that's invisible without a real before/after measurement — the "obviously correct" architecture had a real bug that inverted the expected result until it was actually run.

## Disk: why this format has no compression headroom to give up

Pulled exact compressed/uncompressed sizes from the real file's parquet footer metadata (no data download needed — just column statistics already in the footer):

| Column | Compressed | Uncompressed | Ratio |
|---|---|---|---|
| `accession` (Utf8) | 4.0 MB | 20.6 MB | 0.195 (compresses well) |
| `hashes_list` (LargeList<Int64>) | 19.88 GB | 27.3 GB | 0.729 (barely compresses) |

3.67 billion int64 hash values (avg ~3,734 hashes/accession), only 27% smaller under zstd — consistent with minhash values being close to high-entropy/uniform-random over their range, so there isn't much redundancy for any general-purpose compressor to exploit. This means: an uncompressed Arrow IPC file (or any other zero-copy/mmap-friendly format, FlatBuffers included) costs roughly **+37% disk** (27.3GB vs 19.9GB) relative to the current parquet file, for the *same logical data* — there's no clever encoding trick being left on the table by skipping compression; the data just doesn't compress much in the first place.

## planus vs. official `flatbuffers` vs. Arrow IPC

Asked to consider as a FlatBuffers alternative. Researched via WebSearch (2026-06-30): `planus` (planus-org/planus on GitHub) is a pure-Rust FlatBuffers compiler/runtime, v1.3.0 (Jan 2026), 126 stars / 22 forks / 307 commits — small but active. MSRV 1.88.0 (this workspace runs rustc 1.94.0, no compatibility issue).

**If choosing between FlatBuffers implementations:** planus over the official `flatbuffers` crate. The official crate exposes `_unchecked` accessor methods that skip verification and can read out-of-bounds memory on malformed/adversarial input; planus's stated design principle is that *any* undefined behavior in generated code is a critical bug, and it doesn't provide validation-free APIs at all. Tradeoffs: planus only targets Rust codegen (official flatbuffers supports many languages — relevant if a non-Rust reader is ever needed), and it's a much smaller project (126 stars vs. an established, heavily-used official library) — less battle-tested at scale, smaller community to have already hit edge cases.

**But neither beats Arrow IPC for this specific data shape.** Reasoning:
- FlatBuffers has no built-in compression (same +37% disk cost as Arrow IPC uncompressed — measured above, not a FlatBuffers-specific penalty, just inherent to storing this particular data uncompressed).
- FlatBuffers' main selling point — avoid a deserialize/parse step — doesn't address where our time actually goes. `SketchStore` already does a cheap flat copy into a contiguous `Vec<i64>`, not an expensive parse; the measured bottleneck this whole investigation has been decompression + I/O granularity, not deserialization.
- FlatBuffers' real strength (schema evolution, optional fields, unions, nested heterogeneous documents) is built for a problem we don't have — our actual schema is "string key → vector of int64", about as simple as a record gets. Arrow's columnar/array model is a more natural fit for that than a general-purpose table-with-optional-fields schema, and it's purpose-built for exactly the bulk-numeric-array + parallel-compute domain this crate already lives in (`ndarray`, `orx-parallel`, ultimately `numpy`/`jax.numpy` on the consuming side).
- Zero new dependencies: we already depend on `arrow`/`parquet`. FlatBuffers (either implementation) would add a new schema language (`.fbs`), a separate codegen toolchain, and a translation layer between FlatBuffers table accessors and the existing `&[i64]`-based Jaccard kernel, for no clear benefit over what Arrow IPC already gives us directly.

**Conclusion:** if a future need specifically requires cross-language schema sharing or deeply nested/evolving record structures, planus is the right choice over the official flatbuffers crate (safety stance fits this codebase's general preference for safe Rust). For the minhash storage problem specifically, Arrow IPC remains the better fit — this prototype validates that the architecture works and measures a real, large win for the access pattern this crate actually has.

## Forward-looking: isolating future metadata columns (taxonomy, UniProt, etc.) from hash I/O

Raised as a design consideration: if/when taxonomy (`ncbi_taxonomy_vocab.pkl`/`.pb`, already sitting alongside the minhash parquet in the same Drive folder) or UniProt cross-reference columns get added, they should **not** live in the same physical file/column-family as the hash data. Reasoning, grounded in the compression numbers above:

- Taxonomy/accession-style string data compresses *well* (accession column measured at 0.195 ratio — would expect similar or better for repeated taxon-rank strings, which have far more redundancy than accession numbers do).
- Hash data compresses *poorly* (0.729 ratio, as measured) and dominates total size (19.88GB of 19.9GB total — metadata is a rounding error by comparison).
- Mixing them in one columnar file means a pure "give me the taxonomy for these accessions" query pays to scan past gigabytes of incompressible hash data it doesn't need (or vice versa: a pure distance-matrix computation touching metadata I/O it doesn't need) — exactly the kind of unnecessary I/O isolation this whole investigation has been about avoiding for the existing two columns.
- **Recommendation:** keep hash data and metadata as physically separate stores, joined by accession (a classic vertical-partitioning / column-family split). Hash data: Arrow IPC (this prototype) or parquet+pushdown, optimized for the point-lookup access pattern. Metadata: parquet, where its strong dictionary/RLE encoding on repeated categorical strings (taxon names, ranks) actually pays off — a format choice that would be *wasted* on the hash column but is exactly right for metadata. A distance-matrix-only query then never touches metadata I/O at all, and a taxonomy/lineage query never has to wade through the hash data.

This is a recommendation for when those columns actually get added, not something to build speculatively now — the relevant thing today is that the hash-store format decision (this prototype) doesn't preclude it; keeping hash and metadata physically separate from the start avoids ever needing to *un-mix* them later.
