# proxide Dynamic Workflows

Executable Claude Code **Workflow** scripts (plain JS, run via the Workflow tool:
`Workflow({ scriptPath: ".praxia/docs/dynamic_workflows/<file>.js" })`). Each uses
`agent()/parallel()/pipeline()/phase()/log()` to orchestrate subagents
deterministically. Run from the proxide repo root.

- [260602_dunbrack-rotlib-sprint.js](260602_dunbrack-rotlib-sprint.js) — **Sprint #12**: Dunbrack 2010 → protobuf+zstd rotamer library + cis-PRO (`CPR`). One read-only **Research** agent (parallel, non-blocking) + 6 **sequential** build phases (P1 Extract → P2 Schema → P3 Geometry → P4 Converter → P5 Loader → P6 Verify). Locked decisions baked in: A=precompute coords, 5% stepdown, reuse proxide-geometry NeRF, ODC-BY data / MIT code, never copy MASTER's CC BY-NC-SA coords. Spec: [../specs/260602_dunbrack-rotlib-protobuf-cis-pro.md](../specs/260602_dunbrack-rotlib-protobuf-cis-pro.md). Backlog #814–#820.
