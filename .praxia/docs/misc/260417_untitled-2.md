# Reconnaissance: 26-04-15

## Context Analysis
- The project is transitioning from a standalone `_proxider` Rust extension to a unified `proxide-jax` (Python) and `proxide-core` (Rust) workspace.
- The `_proxider` codebase already has many of the components (parsing, force fields, fetching) implemented.
- `fetch_rcsb`, `fetch_md_cath`, and `fetch_afdb` have a path traversal vulnerability and lack input sanitization.
- `HDF5` is a potential point of failure for platform-wide `pip install` due to C library dependencies.

## Strategic Understanding
- The split into `core` and `bindings` crates is strategically sound for portability but requires careful `ndarray` <-> `numpy` zero-copy management.
- `no_std` is viable for the core math/geometry but not for I/O/parsers.

## Risks Identified
- **Critical:** Path traversal in `fetch`.
- **Warning:** `maturin` failures on Linux/macOS without HDF5 dev headers.
- **Warning:** Ownership issues across FFI boundary in a split-crate architecture.
