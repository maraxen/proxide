# proxide-parallel-rt

> **Alpha — experimental, no API stability guarantees.** This crate is part of an active research project; APIs may change without notice between versions. Not recommended for production use.

Minimal parallel runtime: thread-count registry for wasm32 builds

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

proxide-parallel-rt is a tiny shim that provides a global thread-count registry via a process-wide atomic variable. It exists so that code which needs to know how many worker threads are available (e.g. rayon-like parallelism) can query a single canonical source without pulling in a full threading runtime, making it compatible with wasm32 targets where real thread pools are unavailable.

## Usage

```rust
// At program startup, register the number of worker threads:
proxide_parallel_rt::set_num_threads(rayon::current_num_threads());

// Anywhere in the crate tree, retrieve the configured parallelism count:
let n = proxide_parallel_rt::num_threads();
```
