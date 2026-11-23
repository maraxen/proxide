---
title: 'proxide writers: insertion codes, per-atom hetero, configurable and reported'
description: 'Debt #3583 rescoped: extend main''s write_pdb/write_mmcif/write_npz with insertion codes and a per-atom hetero flag, behind per-call options, with an additive *_reported WriteReport (counts, coded findings, sha256), phase-boundary telemetry and an opt-in provenance sidecar; ports the archived aminx 095a49e0 PDB tests as acceptance tests'
status: draft
task_id: 261123_proxide-pdb-writer
date: '261123'
backlog_ids: '3583'
adversarial_review: ''
---
# proxide writers: insertion codes, per-atom hetero, configurable and reported

## 0. Why this spec exists, and what changed from the debt text

Debt #3583 asked for "a configurable full-atom PDB writer" and listed defects in
`src/proxide/io/writing.py:87-160` (hard-coded B=0.00/occ=1.00, no HETATM, no icode,
misaligned 1-letter names, silent CA/UNK fallback). **Those line numbers and defects are from the
stale branch `fix/parameterize-solvent-atoms`, not `main`.** On `main` (6d6f111) most were
already fixed by #1927, #1928, #2252, #2362, #2366, #2367 and backlog #5684/#5687.

The aminx decision (`aminx .praxia/docs/decisions/261004_laser-pdb-output-scope.md`, option C,
PR #304) said "port aminx `095a49e0:src/aminx/io/laser_pdb.py`, don't reinvent". On 2026-11-23
the user re-decided, given the recon below: **extend main's writer instead of adding a second
writer, and make the new behaviour configurable, with provenance and telemetry recorded and
reported.** The archived writer's *tests* are still ported, as acceptance tests (T6).

## 1. Verified state of `main` (6d6f111)

Tags: **V** = read at file:line on `origin/main`; **U** = unverified, probe in T0.

| # | Fact | Tag |
|---|---|---|
| F1 | `write_pdb(protein, path) -> Path` (`writing.py:413`), `write_mmcif` (`:515`), `write_npz` (`:605`); no options argument. All three, and `Protein`, are in `_STABLE` (`src/proxide/__init__.py:164`) and README "Stability" (`README.md:166-171`). | V |
| F2 | Already correct on main: column-14 names for one-letter elements (`writing.py:494-500`); source B-factor/occupancy, never fabricated (`:376-400`, #2367); overflow raises (`:443-485`); non-finite coordinates raise (`:104-117`); Atom37/14 write only `atom_mask` rows; per-atom (Full) path refuses to invent names/res_names/ids (`:192-249`). | V |
| F3 | Column 27 (insertion code) is always blank: the format string writes `{res_seq:>4}` then four spaces (`writing.py:501`). | V |
| F4 | Record type is `HETATM` iff `res_name in _HETATM_RESIDUES` (`writing.py:467`, mmCIF `:553`); `_HETATM_RESIDUES` = `["MSE"]` (`crates/proxide-io/src/formatters/residue_policy.rs:41-43`). **A parsed non-MSE ligand is written as `ATOM`** even when the source said `HETATM`. | V |
| F5 | Rust `RawAtomData` carries `insertion_codes` and `is_hetatm` (`crates/proxide-core/src/structure/mod.rs:40,67`); the raw dict emits them (`crates/proxide_py/src/bindings/conversion.rs:40-49`). The formatted Atom37/Atom14/Full/Backbone dicts **drop both** (`conversion.rs:109-216`). | V (recon, spot-checked :40-45) |
| F6 | `Protein` has no insertion-code or hetero field (`src/proxide/core/containers.py:64-176`; grep for `insertion`/`hetat` returns nothing). `from_rust_dict` reads `residue_names` per residue for Atom37/14 (`:383`) and `res_names`/`atom_chain_ids` per atom for Full (`:522-523`). | V |
| F7 | A caller can build a Full `Protein` directly: flat `coordinates`/`full_coordinates`, per-atom `atom_names`, `res_names`, `elements`, `atom_chain_ids`, `atom_res_index`, `b_factors`, `occupancies` are all plain fields (`containers.py:115-176`); `atom_residue_ids` returns `atom_res_index` when set (`:215-226`). | V |
| F8 | The per-atom path does **not** consult `full_atom_mask`; every row is written, and NaN rows raise (F2). A caller with NaN-marked absent atoms (aminx LASEr) must drop those rows itself. | V |
| F9 | Per-residue icode exists in Rust on `ProcessedStructure` residues (`crates/proxide-core/src/processing/residues.rs:22-37`, set at `:164-176`, `:265`). That the Atom37/14 formatters can read it the way they read `residue_names` is **INFERRED**. | U |
| F10 | Private writers already exist: `backend._raw_atom_data_to_pdb` (`src/proxide/io/parsing/backend.py:355-396`, ATOM only, no icode) and `_raw_atom_data_to_mmcif` (`:438-502`, writes `group_PDB` from `is_hetatm` and `pdbx_PDB_ins_code`). Used only for altloc re-parse. | V |
| F11 | Diagnostics: no `Report`/`DiagPolicy`/`*_reported`/`diagnostics/registry.toml` exists in code; spec `260910_proxide-silent-substitution-observability.md` (D1-D3, D9) is a **draft**. Telemetry in Python is `warnings.warn` (`writing.py:337`) and module loggers; Rust uses `log` bridged by `pyo3_log`. | V |
| F12 | Spec `261102_proxide-stable-hardening-and-cisternal.md` T6 (draft): **library calls stay silent; only CLI/MCP entry points record to cisternal**; storage location is layered config; absent provenance reports `{"source": "unavailable", "reason": ...}`. | V |
| F13 | No fixture under `tests/` has a non-blank PDB column 27 (`tests/data/mse_altloc.pdb` checked: none). `tests/data/5awl.pdb` has 12 HETATM HOH. Round trips exist for B/occ (`tests/io/test_writing.py:700`) and Full (`:594`); none for icode or record type. | V |
| F14 | Tests of writers need the extension built from this worktree (`writing.py:127` imports `_proxider`); `pyproject.toml:115-127` says build from the repo root with `maturin develop --release` or `uv pip install -e .`. | V |

## 2. Design

### D1. Two new optional `Protein` fields, layout copied from existing fields

- `insertion_codes`: per-residue `(N_res,)` one-character strings for Atom37/Atom14 (same layout
  as `residue_names`); per-atom `(N_atoms,)` for Full (same layout as `res_names` and
  `atom_chain_ids`). Blank is `" "`; `None` means "the source did not say".
- `hetero`: per-atom bool, **layout follows the coordinates** exactly as `b_factors` does
  (`containers.py:125-130`): `(N_res, 37)`, `(N_res, 14)` or `(N_atoms,)`. `None` means unknown.

Both default to `None`, so existing constructors and pickles are unaffected (additive to a stable
type). Docstring the layouts next to `b_factors`.

Rejected: a per-residue `hetero`. A residue is hetero in PDB terms, but Full proteins mix residues
and ligands per atom, and copying `b_factors`' layout keeps one rule for per-atom metadata.

### D2. Plumbing

Rust formatters (Atom37, Atom14, Full; Backbone may stay without) fill the two fields and
`conversion.rs` emits them as `insertion_codes` and `hetero`. Atom37/14 take the per-residue
icode from the processed residues (F9). `from_rust_dict` reads them in both branches. `write_npz`
saves them when present, like `b_factors` (`writing.py:633-637`). `Protein.replace` and any
slicing/chain-filter helpers that already carry `b_factors` (`TestChainFilterKeepsSourceBAndOccupancy`,
`test_writing.py:881`) must carry the new fields the same way.

### D3. Writer options (configurable), keyword-only, validated

```python
def write_pdb(protein, path, *, record_type="source", insertion_codes="write",
              provenance="none") -> Path
def write_mmcif(...same keywords...) -> Path
```

| option | values | meaning |
|---|---|---|
| `record_type` | `"source"` (default) | `HETATM` where `protein.hetero` is true; where `hetero is None`, fall back to the `_HETATM_RESIDUES` rule (today's behaviour). MSE written as MET (no `residue_names`) stays `ATOM` because its res_name is MET. |
| | `"residue_table"` | Exactly today's rule, ignoring `hetero`. |
| `insertion_codes` | `"write"` (default) | Write the codes; blank when `None`. |
| | `"blank"` | Write column 27 blank (and `?` in mmCIF). |
| `provenance` | `"none"` (default), `"sidecar"` | See D5. |

Validation follows `ModifiedResidueTyping::from_name` (`crates/proxide-core/src/processing/residues.rs:103-111`):
an unknown value raises `ValueError` naming the accepted values, never a silent default. Options
are **per call only**: no env var or pyproject layer, because writer output must not depend on the
environment. A value that loses information (`"residue_table"` when `hetero` marks a non-table
residue, `"blank"` when any code is non-blank) is allowed but produces a coded finding and a
`UserWarning` (D4).

Rejected: an options dataclass. The only precedent for options on these functions is keyword
strings on `load_structure` (`backend.py:84-136`). An `OutputSpec`-style object would be new
vocabulary for three keys.

Rejected: changing the default to `"residue_table"` to keep byte-identical output. Writing a
parsed ligand as `ATOM` silently drops what the source said (F4). This is a correctness fix to
a stable function; changelog it under "Fixed", and show the before and after in the PR.

### D4. Reported: additive `*_reported` functions and a `WriteReport`

Following the shape of spec 260910 D3 (additive `_reported` function, existing signature unchanged),
without building that spec's generic `Report<K>` machinery, which is still a draft:

```python
@dataclass(frozen=True)
class WriteFinding:
    code: str          # stable, e.g. "PROX-WRITE-HETERO-IGNORED"
    severity: Literal["info", "coercion", "warning"]
    count: int         # number of atoms or residues affected
    message: str

@dataclass(frozen=True)
class WriteReport:
    path: Path
    format: Literal["pdb", "mmcif"]
    sha256: str                      # of the written file
    n_atoms: int
    n_hetatm: int
    hetatm_from: dict[str, int]      # {"hetero_flag": n, "residue_table": n}
    n_insertion_codes: int           # non-blank codes written
    options: dict[str, str]          # the effective options
    source: str | None               # protein.source
    proxide_version: str
    findings: tuple[WriteFinding, ...]
    provenance_path: Path | None     # D5, None unless provenance="sidecar"

def write_pdb_reported(protein, path, **options) -> WriteReport
def write_mmcif_reported(protein, path, **options) -> WriteReport
```

`write_pdb` becomes `write_pdb_reported(...).path`. The function bodies stay in one place.

Codes, all new; list them in a new `src/proxide/io/write_codes.py` table with one-line meanings
(a seed for 260910 D2's registry, not a substitute for it):

- `PROX-WRITE-HETERO-IGNORED` (warning): `record_type="residue_table"` wrote `ATOM` for atoms whose `hetero` is true.
- `PROX-WRITE-HETERO-UNKNOWN` (info): `hetero is None`; the residue-table rule decided every record.
- `PROX-WRITE-ICODE-BLANKED` (warning): `insertion_codes="blank"` dropped N non-blank codes.
- `PROX-WRITE-ICODE-UNKNOWN` (info): `insertion_codes is None`; column 27 written blank.

Warning-severity findings are also emitted as `warnings.warn(..., UserWarning, stacklevel=3)`,
the existing pattern (`writing.py:337`), so `write_pdb` callers see them without the report.

The report lives in a new module `src/proxide/io/write_report.py`. `write_pdb_reported`,
`write_mmcif_reported`, `WriteReport` and `WriteFinding` are **experimental** (not added to
`_STABLE`) until spec 260910's `Report` lands and the two are reconciled.

### D5. Telemetry and provenance, inside spec 261102 T6's boundary

- **Telemetry (library, silent by default).** `writing.py` gets `logger = logging.getLogger(__name__)`
  and logs **one** `INFO` line per write, at the phase boundary, with the report's counts, options
  and code list. There is no per-atom logging (260910 D8). Python `logging` control is the switch, the same as for the Rust `log`→`pyo3_log` path.
- **Recording to cisternal** happens only at CLI/MCP entry points (261102 T6) and is **out of
  scope here**. `WriteReport` is designed as that record's payload: it carries the operation,
  output path and sha256, version, options, counts and codes. T6 adds the input path and sha256
  and the duration.
- **Provenance sidecar (opt-in).** `provenance="sidecar"` writes `<path>.prov.toml` next to
  the output, shaped like 260910 D9 (`schema = "bathos/0.3"`, a `[provenance]` table with
  `proxide_version`, `proxide_git_sha` or `"nogit"`, `output_sha256`, `source` if known, the
  options, the counts and the finding codes). Write the output and the sidecar atomically
  (temporary file, then rename), with the sidecar written after the output and its sha256 matching the
  file on disk. With `provenance="none"`, the library writes nothing but the requested file.
- Rejected: `REMARK` lines in the PDB. Downstream readers (prody, biotite, aminx oracles) vary in
  REMARK tolerance; and a sidecar matches 260910 D9 and the `.offsets`/`.manifest` precedent.

### D6. What this spec does not do

- No backbone N-H imputation or titratable-H cleanup (aminx option B epic).
- No change to `backend._raw_atom_data_to_pdb` (F10); note in a comment that it remains a
  private altloc helper and must not be exported.
- No `Report<K>`/`DiagPolicy` (260910), no cisternal integration (261102 T6), no config-file resolver.
- No array-level convenience writer. aminx #3565 builds a Full `Protein` (F7) and drops NaN rows (F8).

## 3. Tasks

Each task ends with its acceptance command. Build the extension once per Rust change, from the repo
root of the worktree (`uv pip install -e .`, F14). **Do not run the whole suite locally**:
run named test files only, and rely on CI for the full matrix.

- **T0. Probes (throwaway, record results in this doc's section 1).**
  (a) F9: can `formatters/atom37.rs` and `atom14.rs` reach per-residue insertion codes? Name the access path.
  (b) Parse a PDB fixture with a ligand through `parse_structure(format=Full)` and confirm the
  raw `is_hetatm` is true for ligand atoms and false for protein atoms (F5 shows emission in the raw dict only).
  (c) Confirm the 5awl HOH atoms are present in the Full output.
  Accept: section 1 rows F9 and a new F15 (hetero parse) tagged V with file:line, or the spec is revised.
- **T1. Fixture.** Add `tests/data/icode_ligand.pdb`: two chains, at least two residues with
  insertion codes (`52A`, `52B`), one ligand with `HETATM` records (2+ atoms, two-letter element,
  e.g. a `ZN` or a small `HETATM` organic), one MSE. Hand-written or trimmed from a real wwPDB entry;
  if trimmed, record the source in `tests/data/PROVENANCE.md` if one exists, else in the test docstring.
  Accept: `awk` shows non-blank column 27 rows and HETATM rows.
- **T2. Rust plumbing (D2).** Formatters and `conversion.rs` emit `insertion_codes` and `hetero`.
  Accept: `cargo test --workspace` for the touched crates; `cargo clippy -D warnings` clean; a new
  Rust unit test per formatter asserting both fields on the T1 fixture.
- **T3. Python `Protein` and `from_rust_dict` (D1, D2), plus `write_npz`.**
  Accept: `uv run pytest tests/io/test_writing.py -k "npz or chain_filter"` plus a new test that
  `parse_structure(T1)` yields the expected per-residue codes for Atom37 and per-atom codes and hetero for Full.
- **T4. Writer options and record logic (D3).** Accept: new tests in `tests/io/test_writing.py`:
  parse→write→parse round-trip of T1 for Atom37 and Full preserves icodes and record types;
  every option value has a test; an unknown option value raises `ValueError` naming the accepted
  values; **must-fire and must-not-fire controls** for each finding code (a protein that should
  produce `PROX-WRITE-HETERO-IGNORED` and one that must not; same for `ICODE-BLANKED`).
- **T5. Report, telemetry, provenance (D4, D5).** Accept: tests that `write_pdb` and
  `write_pdb_reported(...).path` produce byte-identical files; the report's `sha256` equals the
  file's; `caplog` sees exactly one INFO record per write; `pytest.warns` for warning codes only;
  `provenance="none"` leaves exactly one new file in `tmp_path`; `"sidecar"` leaves two, the
  sidecar parses as TOML and its `output_sha256` matches.
- **T6. Port the archived aminx tests as acceptance tests.** Source:
  `/home/marielle/scratch/261123_proxide-pdb-writer/test_laser_pdb.py.at_095a49e0` (aminx commit
  `095a49e0`, local tag only). Port `test_heavy_layout_roundtrip`,
  `test_protonated_layout_writes_non_nan_hydrogens`, `test_identifiers_and_bfactors_use_pdb_columns`
  and `test_ligand_passes_through_as_hetatm` as `tests/io/test_writing_full_atom.py`, building a Full
  `Protein` (F7) instead of calling `write_laser_pdb`; replace prody with a small pure column parser
  (columns 1-6, 13-16, 18-20, 22, 23-26, 27, 31-54, 55-60, 61-66, 77-78) plus `parse_structure`
  for the round trip. Drop `test_docstring_leaves_hydrogen_cleanup_open` and
  `test_featurizer_identifiers_reach_the_sample_helper` (aminx-only). Per-residue B-factors are
  broadcast to atoms by the test; that broadcast is the aminx caller's job in #3565.
  Accept: the new file passes, and each ported test names its archived original in a docstring.
- **T7. Docs and changelog.** CHANGELOG "Fixed": parsed ligands are written as HETATM; "Added":
  insertion codes, `hetero`, writer options, `*_reported` (experimental). README Stability: the
  new keyword arguments are stable, and the `*_reported` functions and report types are experimental.

Order: T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7. T6 can start after T3 because it uses only
the Full path and per-atom fields.

## 4. Verification rules

- CI on the draft PR is the full-suite gate. Locally, run only named files.
- Green CI alone is not evidence for the new codes; the must-fire and must-not-fire controls in T4 are.
- Review the PR with one Cursor review (praxia surface dispatch) and spot-check its claims against the diff.

## 5. Risks

| Risk | Mitigation |
|---|---|
| F9 is wrong: Atom37/14 formatters cannot reach per-residue icodes cheaply | T0(a) decides first; fallback is to reduce the per-atom raw codes per residue in the formatter, asserting that all atoms of a residue agree |
| Default `record_type="source"` changes bytes for users who write parsed ligands | Changelog "Fixed"; `"residue_table"` restores the old bytes exactly; T4 has a test proving it |
| Waters (HOH) become HETATM in Full output | Correct per PDB; T0(c) confirms what Full carries today; call it out in the changelog |
| The report types diverge from 260910's `Report` when that lands | Experimental status; codes are strings that the registry can adopt unchanged |
| Branch drift: proxide's main checkout is on `fix/parameterize-solvent-atoms`, with a different `writing.py` | Work only in this worktree off `origin/main`; never cite line numbers from the main checkout |
