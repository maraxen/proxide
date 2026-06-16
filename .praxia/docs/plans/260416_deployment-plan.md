# Comprehensive Deployment Plan: Proxide Monorepo (Refined)

## 1. Executive Summary
This plan details the final steps to transition the `proxide` project into a robust, high-performance monorepo supporting dual-ecosystem deployment (PyPI + Crates.io) and a Typer-powered CLI.

## 2. Core Implementation Strategy
### A. Workspace Consolidation
- **Repository Layout**:
  - `crates/proxider`: Pure Rust Core (`rlib`, `no_std` compatible where possible).
  - `crates/proxider-py`: PyO3 Bindings (`cdylib`).
- **Versioning**: Enforced through `[workspace.package] version`.

### B. High-Performance Refactor
- **Parallelism**: Implement `rayon` in `proxider` core using spatial hashing (Cell Lists) for neighbor searching. Expose thread-control in the Python extension.
- **I/O Streaming**: Convert all trajectory parsing to `Iterator`-based streaming.

### C. CLI Development (`proxide` CLI)
- **Framework**: `typer` + `rich` + `tqdm`.
- **Commands**: `fetch` (with path sanitization), `info`, `convert`, `validate`, `parameterize`, `bench`.

## 3. Oracle-Critique Roadmap (Iterative Refinement)

### Cycle 1: Architecture & Naming
- **Focus**: Validate the `proxide`/`proxider` split and crate architecture.
- **Decision Point**: Finalize the `PyAtomicSystem` wrapper pattern to maintain zero-copy efficiency.
- **Action**: Assess memory management overhead of wrappers vs. alternative approaches.

### Cycle 2: Parallelism & Scalability
- **Focus**: HPC-readiness.
- **Decision Point**: Thread pool management for Python-called Rust functions.
- **Action**: Benchmarking strategy for Cell Lists vs. legacy O(N²).

### Cycle 3: Deployment & CI/CD
- **Focus**: Automated workflows (`publish.yml`).
- **Decision Point**: How to handle multi-platform wheel building vs. standalone crate publishing.
- **Action**: Security audit of `fetch` utilities and `typer` CLI.

## 4. Operational Requirements (Rules of Engagement)
- **Git Safety**: Do not commit without user authorization.
- **Validation**: Every refactor must include a functional test case.
- **Audit**: Persist all operations in `.agent/recon/`.
- **Transparency**: Explain before acting; brief summaries post-action.
