---
title: 'proxide writers: insertion codes, per-atom hetero, configurable and reported'
description: 'Debt #3583 rescoped: extend main''s write_pdb/write_mmcif/write_npz with insertion codes and a per-atom hetero flag, behind per-call options, with an additive *_reported WriteReport (counts, coded findings, sha256), phase-boundary telemetry and an opt-in provenance sidecar; ports the archived aminx 095a49e0 PDB tests as acceptance tests'
status: draft
task_id: 261123_proxide-pdb-writer
date: '261123'
backlog_ids: '3583'
adversarial_review: "R1 dsp_efe6460553135a87 not_ready -> R2"
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
| F9 | Per-residue icode exists on processed residues: `ResidueInfo.insertion_code` (`crates/proxide-core/src/processing/residues.rs:37`, set at `:265`, `ResidueId` built from the raw icode at `:164`); the formatters iterate `residue_info` (`crates/proxide-io/src/formatters/atom37.rs:69-72`), where they already read `residue_names`. | V (review 1) |
| F15 | A residue's hetero status is taken from its FIRST atom (`residues.rs:194`). Only residues classified as protein reach `residue_info` and the Atom37/14 arrays (alias path, or CCD peptide-linking such as MSE, `residues.rs:207-235`); ligands and solvent never appear in Atom37/14 (`tests/io/test_writing.py:718`: 5awl's 12 HETATM HOH are not written). MSE written as MSE is HETATM by test (`tests/io/test_atom14_variants.py:88,105`). | V |
| F16 | Paths that produce `Protein` fields besides the formatters: the parse cache (`crates/proxide_py/src/py_parsers.rs:901-916`, `:976-998`; `conversion.rs:62-107`) and the Atom37 multi-model stacker (`py_parsers.rs:~798-896`, stacks `b_factors`/`occupancies` per model). Python sites that copy or reshape `Protein` fields: `src/proxide/ops/transforms.py` (`truncate_protein` `:120`, `concatenate_proteins_for_inter_mode` `:194`, pad, stack `:568`) and the chain filter that rebuilds `Protein` (`src/proxide/io/parsing/backend.py:~250-285`). | V (review 1, sites spot-checked) |
| F17 | The mmCIF writer has no `pdbx_PDB_ins_code` column (`writing.py:572-583`); the mmCIF reader reads `pdbx_PDB_ins_code` and keys residues on (chain, seq, icode) (`crates/proxide-io/src/formats/mmcif.rs:1838-1866`, its tests), so without the column 52 and 52A merge on re-read. | V |
| F18 | No git sha is embedded anywhere: `crates/proxide-core/build.rs` and `crates/proxide-rotlib/build.rs` exist but only compile protobufs (review 1's "no build.rs" was wrong; corrected by reading). `proxide.__version__` comes from `importlib.metadata` (`src/proxide/__init__.py:15`). | V |
| F10 | Private writers already exist: `backend._raw_atom_data_to_pdb` (`src/proxide/io/parsing/backend.py:355-396`, ATOM only, no icode) and `_raw_atom_data_to_mmcif` (`:438-502`, writes `group_PDB` from `is_hetatm` and `pdbx_PDB_ins_code`). Used only for altloc re-parse. | V |
| F11 | Diagnostics: no `Report`/`DiagPolicy`/`*_reported`/`diagnostics/registry.toml` exists in code; spec `260910_proxide-silent-substitution-observability.md` (D1-D3, D9) is a **draft**. Telemetry in Python is `warnings.warn` (`writing.py:337`) and module loggers; Rust uses `log` bridged by `pyo3_log`. | V |
| F12 | Spec `261102_proxide-stable-hardening-and-cisternal.md` T6 (draft): **library calls stay silent; only CLI/MCP entry points record to cisternal**; storage location is layered config; absent provenance reports `{"source": "unavailable", "reason": ...}`. | V |
| F13 | No fixture under `tests/` has a non-blank PDB column 27 (`tests/data/mse_altloc.pdb` checked: none). `tests/data/5awl.pdb` has 12 HETATM HOH. Round trips exist for B/occ (`tests/io/test_writing.py:700`) and Full (`:594`); none for icode or record type. | V |
| F14 | Tests of writers need the extension built from this worktree (`writing.py:127` imports `_proxider`); `pyproject.toml:115-127` says build from the repo root with `maturin develop --release` or `uv pip install -e .`. | V |

## 2. Design

Revision 2 (after adversarial review 1, `dsp_efe6460553135a87`, verdict not_ready): `hetero` is now
**Full-only**, the record rule never downgrades an existing HETATM, plumbing and propagation sites
are enumerated, the mmCIF column policy is decided, byte-compatibility is pinned by golden files, the
atomicity claim is dropped, and the provenance vocabulary follows bathos `track_packages`. See section 6.

### D1. New optional `Protein` fields

- `insertion_codes`: one-character strings. Per residue, `(N_res,)`, for Atom37/Atom14 (the same layout and
  handling as `residue_names`). Per atom, `(N_atoms,)`, for Full (the same as `res_names`/`atom_chain_ids`).
  Blank is `" "`. `None` means the source gave none. Validation: every entry is exactly one printable
  ASCII character, else `ValueError`. Length must equal n_res or n_atoms, else `ValueError` (the existing
  "refusing to pad or truncate" helpers, `writing.py:143-154`, `:357-373`).
- `hetero`: **per-atom `(N_atoms,)` numpy bool, Full layout only.** It has the same name as the raw-dict
  key (`conversion.rs:41`). `from_rust_dict` converts it to `np.bool_` explicitly and never routes it
  through `_per_slot`, which casts to float32. Setting `hetero` on an Atom37/Atom14 `Protein` raises
  `ValueError` at write time. Those layouts can only hold protein residues (F15), so a hetero flag there
  carries no ligand information. Their record type is decided as today (D3).

Both default to `None`, which is additive to a stable type. Document both next to `b_factors`
(`containers.py:125-130`).

### D2. Plumbing: every producer and every propagation site

Producers, all in T2:
1. The Atom37, Atom14 and Full formatters, and the `to_py_dict` for each (`conversion.rs:109-216`).
   Atom37/14 read per-residue icodes from `residue_info` (F9). Full reads per-atom raw icodes and `is_hetatm`.
   Backbone is unchanged.
2. The parse cache: `CachedStructure` and `FullPerAtom` (`conversion.rs:62-107`, `py_parsers.rs:901-916`, `:976-998`).
   A cache hit must return the same fields as a cold parse.
3. The Atom37 multi-model stacker (`py_parsers.rs:~798-896`). `insertion_codes` is per residue and shared
   across models. Take it from the reference model, and raise if any model disagrees.
4. `from_rust_dict`, both branches (`containers.py:361-400`, `:514-615`).
5. `write_npz` saves both when present, like `b_factors` (`writing.py:633-637`). `src/proxide/io/fixtures.py:129`
   (the only `np.load` of writer output) loads them back if present.

Python propagation sites. Each gets a test in T3.

| site | `insertion_codes` | `hetero` (Full) |
|---|---|---|
| `ops/transforms.py:120` `truncate_protein` | crop like `residue_names` | crop the atom rows the truncation keeps, or raise if truncate does not support Full |
| `ops/transforms.py:194` concat | concatenate like `residue_names` (special-cased, not tree_map) | concatenate |
| pad | pad with `" "` | pad with `False` |
| `ops/transforms.py:568` stack | strip from tree_map like `residue_names`; keep a per-row list | stack |
| chain filter (`backend.py:~250-285`) | Atom37/14: index with the residue mask, as `b_factors` is | see below |

The chain filter already drops `residue_names`, `atom_names`, `res_names`, `elements` and `atom_chain_ids` on
Full, and applies a per-residue mask to per-atom arrays there. That is a **pre-existing defect and out of
scope**. T3 files it as a proxide debt and adds a test that a chain-filtered Full `Protein` fails loudly on
write, not silently.

Grep `_reject_if_batched` (`writing.py:17-42`) and extend it: a nested `insertion_codes` (list of lists) is batched.

Not changed: `proxide-wasm` (it consumes `RawAtomData` only and has no writer), and `crates/proxide_rs/oxidize.pyi`
(it does not describe these dicts). `ty` is advisory in proxide's CI for the stub surface, so there is no stub task.

### D3. Writer options and the record rule

```python
def write_pdb(protein, path, *, record_type="source", insertion_codes="write",
              provenance="none") -> Path
def write_mmcif(protein, path, *, ...same keywords...) -> Path
```

Values are validated like `ModifiedResidueTyping::from_name` (`residues.rs:103-111`). An unknown value raises
`ValueError` and names the accepted values. Options are **per call only**: there is no env or pyproject layer,
because writer output must not depend on the environment.

**Record rule, per written row.** `table` means the written res_name is in `_HETATM_RESIDUES`.

| layout | `record_type` | `hetero[row]` | record |
|---|---|---|---|
| Atom37/14 | any | (must be None; else ValueError) | HETATM iff `table` (exactly today's rule, `writing.py:467`) |
| Full | `"source"` (default) | None | HETATM iff `table` (today) |
| Full | `"source"` | True | HETATM |
| Full | `"source"` | False | HETATM iff `table` (never downgrades a table residue) |
| Full | `"residue_table"` | any | HETATM iff `table` (today, ignoring `hetero`) |

Consequences:
- Atom37/14 output is unchanged byte for byte. That covers MSE→MET (ATOM MET), HSD→HIS (ATOM HIS) and
  MSE as MSE (HETATM).
- The only default-output change in PDB is a Full `Protein` whose `hetero` marks a non-table residue.
  That means parsed ligands, ions and waters, such as 5awl's 12 HOH. It is a correctness fix under
  CHANGELOG "Fixed".

`insertion_codes="write"` (default) writes the code in PDB column 27 and in mmCIF `pdbx_PDB_ins_code`, blank
or `?` when absent. `"blank"` writes column 27 blank and `?`.

**mmCIF column policy (owner decision 2026-11-23: only when needed):** `write_mmcif` emits
`_atom_site.pdbx_PDB_ins_code` (after `label_seq_id`) **only when at least one written code is non-blank**
after applying the `insertion_codes` option; then blank rows get `?`. An icode-free `Protein`, or
`insertion_codes="blank"`, gives mmCIF byte-identical to today. `group_PDB` follows the record rule. Without the
column, 52/52A merge on re-read (F17), so a file with codes always carries it.

### D4. `WriteReport`: additive, experimental

Shape follows spec 260910 D3 (an additive `_reported` function; existing signature unchanged). The return
type diverges: 260910's `Reported<T, K>` is value plus report. Here the value is the path, and it is
carried inside the report (`report.path`). Reconcile when 260910 lands.

```python
@dataclass(frozen=True)
class WriteFinding:
    code: str                       # stable string, see the code table
    severity: Literal["info", "warning"]
    count: int
    message: str

@dataclass(frozen=True)
class WriteReport:
    path: Path
    format: Literal["pdb", "mmcif"]
    sha256: str                     # of the exact bytes written
    n_atoms: int
    n_hetatm: int                   # HETATM records written, in either mode
    n_hetatm_from_hetero: int       # rows HETATM because hetero was True and table was False
    n_insertion_codes: int          # non-blank codes written
    options: tuple[tuple[str, str], ...]      # effective options, sorted; immutable
    source: str | None              # protein.source
    proxide_version: str
    findings: tuple[WriteFinding, ...]
    provenance_path: Path | None
```

Codes live in `src/proxide/io/write_codes.py` as a literal table (code, severity, meaning). No code is built
dynamically (260910 OBS-110). When 260910's `diagnostics/registry.toml` lands, it imports this table and
does not duplicate it.

| code | severity | when |
|---|---|---|
| `PROX-WRITE-HETERO-IGNORED` | warning | `record_type="residue_table"` and some `hetero` True row is not `table` |
| `PROX-WRITE-HETERO-UNKNOWN` | info | Full, `hetero is None` |
| `PROX-WRITE-ICODE-BLANKED` | warning | `insertion_codes="blank"` and some code is non-blank |
| `PROX-WRITE-ICODE-UNKNOWN` | info | `insertion_codes is None` |
| `PROX-WRITE-ALIAS-ATOMS-MISSING` | warning | the existing unknown-typed MSE warning (`writing.py:337`), now also a finding |

**One emitter.** `write_pdb` and `write_pdb_reported` both call a private `_write_pdb(..., _warn_stacklevel)`,
so a warning resolves to the *caller's* frame from either entry point. Every warning-severity finding goes
through `_emit_warnings(findings, stacklevel)`; info findings never warn. T5 asserts each warning's
`filename` is the test file. The functions are experimental and not added to `_STABLE`.

### D5. Telemetry and provenance

- **Logging.** `logger = logging.getLogger("proxide.io.writing")`. One `logger.info` per write, with
  %-style lazy arguments (counts, options, codes) and nothing per atom. proxide installs no handler, so
  the library stays silent unless the caller configures `logging`. Warnings are as in D4. This adds a
  `write-file` phase boundary to 260910 D8's list. Python `logging` is the mechanism, and Rust `tracing` is untouched.
- **Cost.** The writer already builds the full text in memory before writing (`writing.py:506-510`). The new
  code encodes it once, writes those bytes, and hashes the same bytes, with no read-back. That is one sha256
  over the file size per call.
- **No atomicity change.** The output file is written exactly as today: validated first, then `open(path, "w")`.
  It is not temp-then-rename. A rejected structure still never touches the file.
- **Provenance sidecar, opt-in, experimental.** `provenance="sidecar"` writes `<path>.prov.toml` **after**
  the output. If that write fails, the `OSError` propagates and the output stays. A missing sidecar means
  only "provenance not recorded". Keys follow the bathos `track_packages` vocabulary, so bathos can
  ingest them:
  ```toml
  schema = "proxide-write-prov/0"     # experimental; 260910 OBS-306 keys pending the bathos contract
  [provenance]
  proxide_version = "..."             # importlib.metadata
  proxide_source_commit = "..."       # PEP 610 direct_url.json vcs_info.commit_id, else "unknown"
  output_path = "..."
  output_sha256 = "..."
  input_source = "..."                # protein.source, or "unknown"
  input_sha256 = "unavailable"        # the writer never sees input bytes; T6 of 261102 adds it at the CLI
  options = { record_type = "...", insertion_codes = "..." }
  counts = { n_atoms = 0, n_hetatm = 0, n_hetatm_from_hetero = 0, n_insertion_codes = 0 }
  finding_codes = ["..."]
  ```
  `"unknown"` is the single spelling for a value that cannot be established. This follows bathos, not
  260910's "unknown"/261102's "nogit" split. **Owner-approved exception (2026-11-23)** to 261102 T6's "library calls
  never write". This writes only on explicit `provenance="sidecar"` from the caller, beside a file the
  caller asked for. Amend T6's text when that spec is next revised.
- **Out of scope:** cisternal recording (261102 T6). T6 can build its per-operation record from
  `WriteReport` plus input path/sha, duration and error code, which the writer does not have.

### D6. Not in this spec

- Backbone N-H imputation, titratable-H cleanup (aminx option B epic).
- `backend._raw_atom_data_to_pdb`/`_raw_atom_data_to_mmcif` stay private altloc helpers. T4 adds a comment
  saying so.
- `Report<K>`/`DiagPolicy`, cisternal integration, config-file resolver.
- An array-level convenience writer.
- **The aminx #3565 contract (updated):** build a Full `Protein` (F7) and drop NaN-marked absent atoms (F8).
  Set `hetero=True` for ligand atoms and `insertion_codes` per atom from the input identifiers. Broadcast the
  per-residue sampled-token probability to `b_factors`. Pass `provenance="sidecar"` if the run wants a
  record.
- Version (owner decision 2026-11-23): the next release is **0.3.0**, because a stable writer's default output
  changes. The change is Full HETATM from the parsed flag; mmCIF changes only for inputs with insertion codes,
  which were written wrong before. The PR records both under CHANGELOG "Fixed"/"Changed".

## 3. Tasks

Build the extension from the worktree root (`uv pip install -e .`, F14) after each Rust change. Run only
named test files locally. CI runs the full matrix. Rust tests: `cargo test -p proxide-io -p proxide-core`
(not `--workspace`). Clippy: `cargo clippy -p proxide-io -p proxide-core -p proxide_py -- -D warnings`.

- **T0. Golden files and probes, BEFORE any source change.** On unmodified `origin/main` code, write
  and commit goldens under `tests/data/golden_writer/`:
  (a) 5awl Atom37 → PDB and mmCIF;
  (b) 5awl Atom14 → PDB;
  (c) a hand-built Full `Protein` with no hetero/icode → PDB and mmCIF;
  (d) 5awl Full → PDB.
  Generate them with a tracked script, `scripts/golden/make_writer_goldens.py` (argparse + logging). The
  script first asserts that `proxide.__file__` and `proxide._proxider.__file__` resolve inside this
  worktree, and refuses otherwise: the main checkout's venv is on another branch and would silently pin the
  wrong bytes. Record the generating commit and both resolved paths in `tests/data/golden_writer/README.md`.
  Probes, with results written into
  section 1:
  - a residue whose atoms carry mixed HETATM/ATOM flags (F15, first-atom rule);
  - which parse path keeps 5awl's HOH in Full;
  - `OutputSpec.include_hetatm`/`remove_solvent` (`spec.rs:97-98`), which a grep suggests are read nowhere.
  Accept: goldens committed in their own commit; section 1 updated.
- **T1. Fixture.** `tests/data/icode_ligand.pdb`, hand-written, with its provenance in a header REMARK and the test
  docstring. Contents: two chains; residues 52, 52A, 52B in chain A; one MSE; one ZN ion and one 3+-atom organic
  HETATM ligand with a two-letter element; two HOH. Accept: `awk` shows non-blank column 27 and HETATM rows.
- **T2. Rust producers (D2 items 1-3).** Accept: Rust unit tests per formatter on T1 (codes per residue for
  Atom37/14; per-atom codes and `is_hetatm` for Full); a multi-model test; cold parse == cache-hit parse
  for both fields (a Python test in T3 if the cache is only reachable from Python).
- **T3. Python fields, `from_rust_dict`, npz, propagation sites (D1, D2).** Accept: new tests for
  `parse_structure(T1)` per layout; each row of the D2 site table; the multi-model and cache tests; the
  chain-filtered-Full-fails-loudly test; npz save/load round trip of both fields. File the chain-filter debt
  and quote its number in the PR.
- **T4. Writer options and record rule (D3).** Accept:
  - a parametrised test covering every row of the record-rule table;
  - PDB round trip of T1 in Full (record types, icodes, ligand atoms);
  - PDB round trip of T1 in Atom37 (icodes; MSE HETATM unchanged);
  - mmCIF round trip of T1 in Full and Atom37 (52/52A/52B stay distinct);
  - every option value, and an unknown value raises;
  - `hetero` on an Atom37 `Protein` raises;
  - an invalid icode (two characters, non-ASCII) raises;
  - goldens: (a)-(c) PDB and mmCIF byte-identical (no goldens carry insertion codes);
    (d) differs from its golden **only** in the record type of the HOH rows (assert the exact
    line set that changed).
- **T5. Report, logging, provenance (D4, D5).** Accept:
  - a must-fire and a must-not-fire protein for each warning code;
  - info codes present and never warned;
  - warning `filename` is the test file, via both entry points;
  - `report.sha256 == sha256(path.read_bytes())`;
  - `caplog.set_level(logging.INFO, logger="proxide.io.writing")` sees exactly one record per write;
  - `provenance="none"` creates exactly one file;
  - `"sidecar"` creates two, the sidecar parses as TOML, and its `output_sha256` matches;
  - a sidecar write forced to fail (monkeypatched `open` for the `.prov.toml` path) leaves the output intact
    and raises.
- **T6. Ported aminx acceptance tests.** Create `tests/io/test_writing_full_atom.py`, self-contained in the PR.
  Port the logic of four tests from aminx commit `095a49e0`
  (`tests/io/test_laser_pdb.py`; the source is a local-only tag, so quote each original assertion in
  the new test's docstring; a local copy is at
  `/home/marielle/scratch/261123_proxide-pdb-writer/test_laser_pdb.py.at_095a49e0` for the implementer's
  convenience only): heavy-layout round trip, protonated layout writes non-NaN hydrogens, identifiers and
  B-factors in PDB columns, ligand passes through as HETATM.
  - Each test builds a Full `Protein` and **sets `hetero` and `insertion_codes` explicitly**. With
    `hetero=None` the ligand test would rightly see ATOM, so the test also asserts that case.
  - prody is replaced by `_columns(line) -> dict` returning `record` (1-6), `name` (13-16, stripped),
    `resname` (18-20), `chain` (22), `resseq` (23-26, int), `icode` (27), `xyz` (31-54, 3 floats),
    `occ` (55-60), `b` (61-66), `element` (77-78, stripped), plus `parse_structure` for the round trip.
  - Drop the two aminx-only tests, and say so in the module docstring.
- **T7. Docs.** CHANGELOG:
  - "Fixed": Full HETATM from the parsed flag.
  - "Changed": mmCIF gains `pdbx_PDB_ins_code` when the structure has insertion codes; Full outputs of
    parsed ligands, ions and waters change record type. The release is 0.3.0.
  - "Added": the fields, the options, and `*_reported` (experimental).
  README Stability: the new keyword arguments are stable; `*_reported`, `WriteReport`, `WriteFinding` and
  the provenance sidecar are experimental. Tag F-rows in this spec V after T0.

Order: T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7. T6 may start after T3.

## 4. Verification rules

- The full suite runs in CI on the draft PR. Locally, run only named files.
- Green CI is not evidence for the codes or the record rule. The must-fire and must-not-fire controls, and
  the golden diffs in T4, are.
- One Cursor review via praxia surface dispatch. Spot-check its claims against the diff before acting on them.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Users of Full output see HOH/ligands change to HETATM | Changelog; `record_type="residue_table"` restores the old bytes, proven by the golden (d) test with that option |
| mmCIF schema varies with input (column present only with codes) | Owner decision; readers tolerate the optional column; T4 tests both the with-codes and the without-codes shape |
| A cache path or transform drops the new fields silently | D2 enumerates them; T2/T3 test cold == cached and each transform |
| `WriteReport` diverges from 260910's `Report` | Experimental; codes are a literal table the registry can import |
| Provenance sidecar conflicts with 261102 T6 "library never writes" | Opt-in only; flagged to the owner in D5 |
| Branch drift: proxide's main checkout is on `fix/parameterize-solvent-atoms` | Work only in this worktree off `origin/main`; never cite line numbers from the main checkout |

## 6. Revision log

- **R1 (261123).** First draft.
- **R2 (261123), after adversarial review `dsp_efe6460553135a87` (Sonnet spec-challenger, verdict not_ready).**
  The reviewer's `agent_result` failed, so its claims were spot-checked against code before acting: F15
  (`residues.rs:190-236`), MSE HETATM tests, cache fields (`py_parsers.rs:904-908, 979-983`), and the
  transforms sites (the path is `src/proxide/ops/transforms.py`, not `core/`).
  - Blocking 1 (MSE/HSD contradiction) → D1 Full-only `hetero` + D3 table.
  - Blocking 2 (cache, multi-model, Atom14) → D2 producers.
  - Blocking 3 (transforms, chain filter) → D2 site table + filed debt.
  - Majors:
    - mmCIF column → D3 policy.
    - Vacuous byte test → T0 goldens.
    - Atomicity → dropped.
    - git sha → F18, D5 vocabulary.
    - stacklevel → D4 emitter.
    - Atom37 ligands → D1/D3.
    - T6 hetero contract → T6, D6.
    - F9 → tagged V; T0 probes refocused.
  - Minors:
    - Validation, dtype, batching → D1/D2.
    - Report design → D4.
    - Spec conflicts → D4/D5.
    - Missing tasks → D2/T2/T7.
- **R3 (261123), owner decisions via AskUserQuestion:**
  - The mmCIF `pdbx_PDB_ins_code` column is emitted only when needed, so icode-free mmCIF stays byte-identical.
  - The opt-in provenance sidecar is approved as an exception to 261102 T6.
  - The default is fixed to HETATM-from-flag, and the next release is 0.3.0.
  - praxia runtime files are ignored in proxide's `.gitignore`.
  - F17 and F18 were re-verified by reading. F18's "no build.rs" was wrong: two exist, but neither
    embeds a git sha.
  - T0 goldens come from a tracked script that refuses a shadowing install.
