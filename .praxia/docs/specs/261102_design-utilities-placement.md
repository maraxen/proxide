---
title: 'Shared design-screening utilities and lesson enforcement: placement across proxide, alynxr, aminx and project specs'
description: 'Engineering specification for Slice 1 and Slice 2 design-screening utilities in proxide: proxide.seqqc and proxide.geometry.contacts (contact analysis, Shrake-Rupley SASA, self-association screening, and alignment overlap), with exact API signatures, batching contracts, selftest gates, and bathos parity verification across 4 runs'
status: hardened_round_1_slice2
task_id: 261102_design-lessons-systemization
date: '261102'
backlog_ids: ''
adversarial_review: 'Round 1 Defender Concessions Applied'
---
# Shared design-screening utilities and lesson enforcement

Source of the lessons: `.praxia/docs/audits/261102_tfr1-epic-close-out.md` (L1-L13). Tags: **[V]** verified by reading the source this session, **[R]** from a Haiku recon report, not re-read, **[U]** unverified.

## 1. Principles and Scope

### 1.1 Architectural Principle
Lessons that are mechanical become library functions; policy stays in specs. Stateless sequence and structure utilities go to the lowest shared layer (`proxide`); anything that runs a predictor or manages campaigns stays in `alynxr`; policy stays in specs. Scripts in project repos become thin wrappers.

### 1.2 Scope Boundary: Slice 1 and Slice 2
- **In Scope for Slice 1**:
  1. `proxide.seqqc`: Pure-Python sequence quality control submodule (composition vs background, net charge, pI, GRAVY, aromatic fraction, sequence entropy, and a `screen()` function returning a dict shaped for bathos `[result_schema]`).
  2. `proxide.geometry.contacts`: Generic interface contact utility between two chains (heavy-atom cutoff, contact sets, residue counts, and ion-pair / like-charge / polar / nonpolar classification).
  3. Batched execution support (`batched == looped`) for both sequence and contact utilities.
  4. Positive and negative selftests for each utility.
  5. Parity verification tests in `tests/validation/test_bathos_parity.py` against registered bathos runs `dda7de75` (composition) and `bba33e3a` (interface audit, bounded to contact/charge contracts).
- **In Scope for Slice 2**:
  1. **2a: Interface Jaccard, Alignment-Mapped Overlap, and Epitope Docking** (`proxide.geometry.contacts`):
     - `contact_jaccard(a, b)`: set intersection over union (`len(a & b) / len(a | b)` if union non-empty else `0.0`), distinct from MinHash `proxide.jaccard`.
     - `aligned_contact_overlap(contacts_a, contacts_b, aligned_pairs)`: projects contacts onto aligned positions in ortholog A numbering, strictly ignoring unaligned residues.
     - `load_aligned_pairs(csv_text_or_path, ...)`: parses alignment numbering tables (`tfr1_numbering_table.csv`, 194 aligned pairs in crop, human 212 unaligned).
     - `contacts_and_dock(...)`: heavy-atom interface contacts, sequence number offset, epitope contact counting, and binary docking verdict (`docked = epitope_contacts >= min_epitope_contacts and n_contacted >= min_interface_residues`), supporting single and batched execution (`batched == looped`).
  2. **2b: Self-Association (SA) Screen** (`proxide.geometry.contacts`):
     - `interface_counts(a, b, cutoff=5.0)`: cross-chain heavy-atom interface residue counts (`n_if_a`, `n_if_b`).
     - `associates(n_a, n_b, ipsae, min_if_residues=12, min_ipsae=0.4)`: binary association test.
     - `screen_self_association(runs, ...)`: aggregates per-run interface metrics across predictors/seeds into bathos `[result_schema]` (`controls_ok`, `design_rate`, `shuffle_rate`, `outcome` in `{"monomeric", "self_associates", "inconclusive"}`).
  3. **2c: Buried Binder Surface Area (`buried_binder_a2` / SASA)** (`proxide.geometry.contacts`):
     - Pure-NumPy Shrake-Rupley (`compute_shrake_rupley`, `compute_buried_sasa`) with standard Bondi atomic radii, probe radius $1.40$ Å, and $N=100$ Fibonacci sphere points (zero external runtime dependencies).
     - Physical invariant S1: non-increasing binder SASA on binding (`complex_sasa <= free_sasa + 1e-5`).
     - Integer rounding vs float precision: exact unrounded dSASA matches Biopython to $< 10^{-6}$ Å², integer rounded values match `tfr1_interface.tsv` exactly (`[764, 537, 467, 515, 485]`).
     - Wired into `InterfaceContactResult` as `buried_binder_a2: int | None = None` and populated in `compute_interface_contacts` when SASA computation is requested or executed.
  4. **2d: Bathos Parity Verification across 4 Runs**:
     - Parity verification in `tests/validation/test_bathos_parity.py` across 4 bathos runs:
       - `dda7de75`: composition QC on `tfr1_final_slate.csv` and `composition_result.json`.
       - `bba33e3a`: interface contact and charge audit on `tfr1_interface.tsv`, un-excluding `buried_binder_A2` (`[764, 537, 467, 515, 485]`).
       - `20aeafaf`: SA screen on `tfr1_sa_runs.tsv` and `sa_result.json` (`outcome: "monomeric"`, `design_rate: 0.2`, `controls_ok: True`, synthetic dimer/monomer 3D tests).
       - `5d906f67`: P3 mouse overlap on `tfr1_p3_mouse.tsv`, `tfr1_p3_mouse_runs.tsv`, `p3_result.json` (194 aligned pairs, human 212 unaligned, `outcome: "pass"`, `median_overlap: 0.0`).
     - Explicit documentation that ephemeral Modal scratch directories (`sa_modal_pull`, `p3_modal_pull`) no longer exist on disk.
- **Deferred to Slice 3+ / Future Milestones**:
  - Design campaign spec skeletons and `spec_lint` rule extensions (L1, L8, L9, L10).
  - Predictor lineage, effective votes, and arm unit readers in `alynxr` (L4, L7).
  - Fixed-token MPNN recipes in `aminx`.

---

## 2. Inventory: proxide vs TfR1 Functions

| Capability (TfR1 source) | In proxide today | Verdict |
|---|---|---|
| Composition vs background, net charge, aromatic fraction, entropy, GRAVY, pI (`tfr1_composition.py`: `freqs`, `charge`, `summarize`; `tfr1_mpnn_redesign.py`: `metrics`) | None [V] | **Slice 1: In-Scope (`proxide.seqqc`)** |
| Interface contacts between two chains, per-residue contact sets (`tfr1_interface_charge.py: pair_counts`) | `geometry/metrics.py` functions raise `NotImplementedError` (6 stubs) [V]; `run_confind` was deleted in commit `47c397d5b` (debt #1901) [V] | **Slice 1: In-Scope (`proxide.geometry.contacts`)** |
| Ion-pair and like-charge contact classification (`tfr1_interface_charge.py: pair_counts`) | None [V] | **Slice 1: In-Scope (`proxide.geometry.contacts`)** |
| Contact-set Jaccard with alignment mapping (`tfr1_p3_mouse.py: jaccard, contact_overlap, load_aligned_pairs, contacts_and_dock`) | `jaccard` module is MinHash-sketch distance over sourmash-style sketch Parquets, NOT contact sets [V] | **Slice 2: In-Scope (`proxide.geometry.contacts`: `contact_jaccard`, `aligned_contact_overlap`, `load_aligned_pairs`, `contacts_and_dock`)** |
| Self-association homodimer screen (`tfr1_sa_screen.py: interface_counts, associates, screen_self_association`) | None [V] | **Slice 2: In-Scope (`proxide.geometry.contacts`)** |
| Buried Binder Surface Area / SASA (`tfr1_interface_charge.py: buried_binder_A2`, `tfr1_his_inspect.py: free vs complex SASA`) | None [V] | **Slice 2: In-Scope (`proxide.geometry.contacts`: `compute_shrake_rupley`, `compute_buried_sasa`, `buried_binder_a2`)** |
| TM-align, structure novelty | `tm_align, tm_scores_fixed_correspondence` exist [V] | **Exists**; use instead of foldseek-side reimplementation |
| AF vs MPNN alphabet conversion (`af_to_mpnn`) | `chem/conversion.py` (JAX) and `io/parsing/mappings.py` (NumPy) derive permutation tables via runtime dependency `alphex.perm`, pinned by `tests/test_alphabet_conformance.py` [V] | **Exists**; aminx uses its own `aminx.utils.aa_convert` |
| Vote logic with predictor family (`tfr1_p5b_check.py: vote, member_verdict`) | N/A | **Not proxide**: belongs to `alynxr` arm metadata |

---

## 3. Placement per Lesson

### 3.1 Slice 1 Placement
- **L2 (Sequence screen)** → `proxide.seqqc`: Pure-Python sequence QC functions, plus a `screen(sequences, background)` that returns a result dict matching the bathos `[result_schema]`.
- **L1 / L11 (Interface contacts and charge audit)** → `proxide.geometry.contacts`: Generic two-chain heavy-atom contacts, residue sets, and charge pair classifications.

### 3.2 Slice 2 Placement
- **L3 / L11 / P3 (Cross-reactivity, alignment overlap, contact Jaccard)** → `proxide.geometry.contacts`: `contact_jaccard`, `aligned_contact_overlap`, `load_aligned_pairs`, `contacts_and_dock`.
- **L6 (Self-association / homodimer screen)** → `proxide.geometry.contacts`: `interface_counts`, `associates`, `screen_self_association`.
- **L11 (Buried binder surface area / SASA)** → `proxide.geometry.contacts`: `compute_shrake_rupley`, `compute_buried_sasa`, wired into `InterfaceContactResult` as `buried_binder_a2`.

### 3.3 Deferred Placement (Slice 3+)
- **L4 (Predictor lineage, effective votes), L7 (Pin tool commit, reader contract)** → `alynxr`: Add `family` and `lineage` to arm definitions, command reporting effective independent votes, and campaign reader contracts.
- **MPNN redesign with fixed target chain** → `aminx`: Recipe/CLI feature for fixed-token specs.
- **L1, L8, L9, L10 (Gate bypass, validated submission artifact, slate freeze date, amendment budget, null rounds)** → Design-campaign spec skeleton and `spec_lint` rules.
- **L3, L5, L6, L12 (Generator diff, organizer extensions marked, speculative arm date, mechanism requirements)** → Spec skeleton checklist items.

---

## 4. Detailed Specification

### 4.1 Sequence QC Submodule: `proxide.seqqc`

#### 4.1.1 Location and Dependencies
- Located at `src/proxide/seqqc/`.
- Pure-Python / NumPy implementation; no heavy C-extensions or external network dependencies.
- Standard 20 amino acid alphabet: `AA = "ACDEFGHIKLMNPQRSTVWY"`.
- Canonical background: SwissProt Release 2021 composition:
  ```python
  SWISSPROT_BG: dict[str, float] = {
      "A": 0.0825, "C": 0.0137, "D": 0.0545, "E": 0.0675, "F": 0.0386,
      "G": 0.0707, "H": 0.0227, "I": 0.0596, "K": 0.0584, "L": 0.0966,
      "M": 0.0242, "N": 0.0406, "P": 0.0470, "Q": 0.0393, "R": 0.0553,
      "S": 0.0656, "T": 0.0534, "V": 0.0687, "W": 0.0108, "Y": 0.0292,
  }
  ```

#### 4.1.2 Function Signatures and Mathematical Definitions
1. `def amino_acid_frequencies(sequence: str) -> dict[str, float]:`
   Normalized frequency $f_a = \frac{\text{count}(a)}{L}$ for each $a \in \text{AA}$.

2. `def net_charge(sequence: str, ph: float = 7.4, model: str = "heuristic") -> float:`
   - When `model == "heuristic"` and `np.isclose(ph, 7.4)`:
     $$q = \text{count}(\text{K}) + \text{count}(\text{R}) + 0.1 \times \text{count}(\text{H}) - \text{count}(\text{D}) - \text{count}(\text{E})$$
     (matches `tfr1_composition.py:33` to guarantee parity with `dda7de75`).
   - When `model == "bjellqvist"`: Henderson-Hasselbalch charge using Bjellqvist pKa values (N-term: 7.50, C-term: 3.55, K: 10.00, R: 12.00, H: 5.98, D: 4.05, E: 4.45, C: 9.00, Y: 10.00).

3. `def isoelectric_point(sequence: str, model: str = "bjellqvist") -> float:`
   Calculates pI via bisection root-finding on `net_charge(sequence, ph=pI, model=model) == 0.0` within tolerance $10^{-4}$ in $[0.0, 14.0]$.

4. `def gravy(sequence: str) -> float:`
   Grand Average of Hydropathy using the Kyte-Doolittle scale:
   $$\text{GRAVY} = \frac{1}{L} \sum_{i=1}^L \text{KD}[s_i]$$
   Scale: `A: 1.8, C: 2.5, D: -3.5, E: -3.5, F: 2.8, G: -0.4, H: -3.2, I: 4.5, K: -3.9, L: 3.8, M: 1.9, N: -3.5, P: -1.6, Q: -3.5, R: -4.5, S: -0.8, T: -0.7, V: 4.2, W: -0.9, Y: -1.3`.

5. `def aromatic_fraction(sequence: str) -> float:`
   $$\text{aromatic} = \frac{\text{count}(\text{F}) + \text{count}(\text{W}) + \text{count}(\text{Y})}{L}$$

6. `def sequence_entropy(sequence: str, base: float = 2.0, normalized: bool = False) -> float:`
   Shannon entropy:
   $$H = -\sum_{a \in \text{AA}, f_a > 0} f_a \log_{\text{base}} f_a$$
   If `normalized=True`, $H_{\text{norm}} = \frac{H}{\log_{\text{base}}(20)}$.

7. `def screen(sequences: list[str] | dict[str, list[str]], background: dict[str, float] | None = None, threshold_ek: float = 0.30) -> dict[str, Any]:`
   Aggregates sequence QC metrics across a list or dictionary of sequence groups.
   Returns a dictionary structured for bathos `[result_schema]`:
   ```json
   {
     "controls_pass": true,
     "n_sequences": 5,
     "background_ek": 0.1259,
     "mean_ek": 0.4517,
     "frac_ek_gt_30pct": 1.0,
     "mean_net_charge": -8.5,
     "frac_net_neg": 1.0,
     "mean_gravy": -1.25,
     "mean_aromatic_fraction": 0.042,
     "mean_entropy": 2.85,
     "pool_ek_enrichment": 3.59,
     "mean_freq": {"A": 0.0935, ..., "Y": 0.0144},
     "fold_over_background": {"A": 1.13, ..., "Y": 0.49},
     "groups": {
       "final_five": {
         "n": 5,
         "mean_len": 82.4,
         "mean_ek": 0.4517,
         "frac_ek_gt_30pct": 1.0,
         "mean_net_charge": -8.5,
         "frac_net_neg": 1.0,
         "mean_freq": {"A": 0.0935, ..., "Y": 0.0144}
       }
     }
   }
   ```

---

### 4.2 Interface Contact Utility: `proxide.geometry.contacts`

#### 4.2.1 Location and Signatures
- Located at `src/proxide/geometry/contacts.py` and exported from `proxide.geometry`.
- Primary function:
  ```python
  def compute_interface_contacts(
      structure: str | Path | list[dict[str, Any]],
      chain_a: str = "A",
      chain_b: str = "B",
      cutoff_heavy: float = 5.0,
      cutoff_ion_pair: float = 4.0,
      cutoff_like_charge: float = 4.5,
      cutoff_polar: float = 3.5,
      cutoff_nonpolar: float = 4.5,
      compute_sasa: bool = False,
      probe_radius: float = 1.40,
      n_points: int = 100,
  ) -> InterfaceContactResult:
  ```

#### 4.2.2 Input Representation
Accepts:
1. `Path` or `str` filepath: Loads and parses the PDB structure.
2. `str` multi-line string: Parses PDB-formatted text directly.
3. `list[dict[str, Any]]`: Pre-parsed atom records with keys:
   - `"chain"`: `str` (e.g. `"A"`, `"B"`)
   - `"name"`: `str` (atom name, e.g. `"CA"`, `"NZ"`, `"OE1"`)
   - `"res"`: `str` (3-letter residue name, e.g. `"LYS"`, `"GLU"`)
   - `"num"`: `int` or `tuple[int, str]` (residue sequence number and optional insertion code)
   - `"xyz"`: `tuple[float, float, float]` or `np.ndarray` of shape `(3,)`
   - optional `"element"`: `str` (chemical element, e.g. `"C"`, `"N"`, `"O"`)

- **Element Preservation and Inference Rule**:
  - When parsing PDB text (`_parse_pdb_text`), columns 76–78 are read for `element`. If empty or whitespace, `element` is inferred from `name[0].upper()` for amino acid heavy atoms (`C, N, O, S`). Backbone alpha carbons named `"CA"` must strictly be treated as Carbon (`"C"`), never assigned Calcium (`"Ca"`).
  - For atom dict inputs, `element` is read from `atom.get("element")` or inferred from `atom["name"][0].upper()`.
  - Element lookup in atomic radii tables is case-insensitive (`element.upper()`).

#### 4.2.3 Chemical Groups and Cutoff Definitions
- **Interface Residues**: Any residue with a heavy atom within `cutoff_heavy` (default 5.0 Å) of any heavy atom in the opposite chain.
- **Charge Sign Mapping**:
  - Cations (+1): `("LYS", "NZ")`, `("ARG", "NE")`, `("ARG", "NH1")`, `("ARG", "NH2")`.
  - Anions (-1): `("ASP", "OD1")`, `("ASP", "OD2")`, `("GLU", "OE1")`, `("GLU", "OE2")`.
  - Histidine counted neutral at pH 7.4 interface default.
- **Ion Pairs**: Cross-chain opposite-sign charge group atoms within `cutoff_ion_pair` (default 4.0 Å).
- **Like-Charge Close (Repulsive)**: Cross-chain same-sign charge group atoms within `cutoff_like_charge` (default 4.5 Å).
- **Polar Pairs**: Cross-chain $N \dots O$ pairs within `cutoff_polar` (default 3.5 Å).
- **Nonpolar Pairs**: Cross-chain sidechain $C \dots C$ pairs within `cutoff_nonpolar` (default 4.5 Å), strictly excluding backbone atoms `{"N", "CA", "C", "O", "OXT"}`.

#### 4.2.4 Result Dataclass
```python
@dataclass(frozen=True)
class InterfaceContactResult:
    n_binder_interface: int
    n_target_interface: int
    interface_residues_a: list[int]
    interface_residues_b: list[int]
    ion_pairs: int
    like_close: int
    polar_pairs: int
    nonpolar_pairs: int
    binder_if_charge: int
    target_if_charge: int
    binder_if_frac_ek: float
    binder_if_frac_hydrophobic: float
    details: dict[str, Any]
    buried_binder_a2: int | None = None  # Slice 2: Shrake-Rupley buried binder surface area in Å²

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

- **SASA Field and Precision Contract**:
  - When `compute_sasa=False` (default), `buried_binder_a2` is `None`.
  - When `compute_sasa=True`, `buried_binder_a2` holds the rounded integer ($\text{round}(\Delta \text{SASA}_A)$ in Å²), matching registered golden tables (e.g. `tfr1_interface.tsv`).
  - When `compute_sasa=True`, `details["buried_binder_a2_float"]` holds the unrounded float value $\Delta \text{SASA}_A$ in Å².
  - When `compute_sasa=True`, `details["sasa_details"]` holds the complete per-residue SASA breakdown dictionary returned by `compute_buried_sasa`.

---

### 4.3 Batching Contract: `batched == looped`

#### 4.3.1 Ragged Sequence Batching
- `screen(sequences: list[str] | dict[str, list[str]])` accepts ragged string lists natively.
- When 2D array representation `(B, L_max)` is provided, an accompanying `pad_mask: np.ndarray` (boolean mask where True indicates valid amino acid) is required. Unmasked tokens are guaranteed to produce identical metrics to unpadded string inputs.
- Equivalence Assertion:
  For any sequence list `seqs = [s_1, ..., s_N]`:
  ```python
  batched = screen(seqs)
  looped = [screen([s]) for s in seqs]
  assert np.isclose(batched["mean_ek"], np.mean([r["mean_ek"] for r in looped]), atol=1e-6)
  assert np.isclose(batched["mean_net_charge"], np.mean([r["mean_net_charge"] for r in looped]), atol=1e-6)
  assert np.isclose(batched["mean_gravy"], np.mean([r["mean_gravy"] for r in looped]), atol=1e-6)
  ```

#### 4.3.2 Ragged Structure Batching
- Batch utility:
  ```python
  def compute_interface_contacts_batch(
      structures: list[str | Path | list[dict[str, Any]]],
      chain_a: str = "A",
      chain_b: str = "B",
      cutoff_heavy: float = 5.0,
      cutoff_ion_pair: float = 4.0,
      cutoff_like_charge: float = 4.5,
      cutoff_polar: float = 3.5,
      cutoff_nonpolar: float = 4.5,
      compute_sasa: bool = False,
      probe_radius: float = 1.40,
      n_points: int = 100,
  ) -> list[InterfaceContactResult]:
  ```
- Equivalence Assertion:
  ```python
  batched = compute_interface_contacts_batch(structures, compute_sasa=compute_sasa)
  looped = [compute_interface_contacts(s, compute_sasa=compute_sasa) for s in structures]
  assert len(batched) == len(looped)
  for b, l in zip(batched, looped):
      assert b == l
  ```

---

### 4.4 Error Handling and Boundary Conditions
- **Empty Sequence**: `""` raises `ValueError("Sequence cannot be empty")`.
- **Case Normalization**: Sequences are automatically normalized to uppercase (`seq.upper()`).
- **Non-Canonical Amino Acids**: Symbols outside `ACDEFGHIKLMNPQRSTVWY` raise `ValueError(f"Invalid amino acid character: '{c}'")` by default.
- **Missing Chains**: Requesting a non-existent chain in `compute_interface_contacts` raises `KeyError(f"Chain '{chain}' not found in structure")`.
- **CA-only Structures**: Heavy-atom interface residue detection functions normally; sidechain-specific contacts (`ion_pairs`, `nonpolar_pairs`) return 0 without error.

---

### 4.5 Positive and Negative Selftest Gates

#### 4.5.1 `proxide.seqqc` Selftests
1. **Positive Selftest**:
   - Single-residue sequence `"EEEE"` produces `freqs["E"] == 1.0`.
   - SwissProt background frequencies sum to $1.0 \pm 0.01$.
   - Standard balanced sequence passes screening complexity filter (`controls_pass == True`, `screen_pass == True`).
2. **Negative Selftest (Validation Failure)**:
   - Input `""` raises `ValueError`.
   - Input `"ACDEF123"` raises `ValueError`.
3. **Negative Selftest (Design Threshold Failure)**:
   - Skewed sequence `"E" * 50` exceeds E+K threshold ($1.0 > 0.30$) and fails aromatic threshold ($0.0 < 0.02$), resulting in `screen_pass == False` with flag reasons populated.
   - Reference negative check from `tfr1_composition.py:53`: `not np.isclose(freqs("EEKK")[AA.index("E")], 1.0)` verifies frequency separation.

#### 4.5.2 `proxide.geometry.contacts` Selftests
1. **Positive Selftests**:
   - Cation `LYS:NZ` at $(0, 0, 0)$ and anion `GLU:OE1` at $(3.0, 0, 0)$ registers `ion_pairs == 1`.
   - Polar pair `SER:OG` at $(0, 0, 0)$ and `LYS:NZ` at $(3.0, 0, 0)$ registers `polar_pairs == 1`.
   - Nonpolar pair `LEU:CD1` at $(0, 0, 0)$ and `ILE:CG2` at $(4.0, 0, 0)$ registers `nonpolar_pairs == 1`.
2. **Negative Selftests**:
   - Cation `LYS:NZ` at $(0, 0, 0)$ and anion `GLU:OE1` at $(4.5, 0, 0)$ registers `ion_pairs == 0` (distance gate).
   - Same-sign pair `GLU:OE1` at $(0, 0, 0)$ and `ASP:OD1` at $(4.0, 0, 0)$ registers `like_close == 1` and `ion_pairs == 0` (repulsive gate).
   - Backbone carbon `LEU:CA` at $(0, 0, 0)$ and `ILE:CB` at $(4.0, 0, 0)$ registers `nonpolar_pairs == 0` (backbone exclusion gate).
   - Chains separated by $500$ Å register `n_binder_interface == 0`, `n_target_interface == 0`, `ion_pairs == 0`.

---

### 4.6 Interface Jaccard & Alignment Overlap

#### 4.6.1 Mathematical Formulation
1. **`contact_jaccard`**:
   - Signature:
     ```python
     def contact_jaccard(
         a: set[Any] | frozenset[Any],
         b: set[Any] | frozenset[Any],
     ) -> float:
     ```
   - Mathematical definition:
     $$\text{Jaccard}(A, B) = \begin{cases} \frac{|A \cap B|}{|A \cup B|}, & |A \cup B| > 0 \\ 0.0, & |A \cup B| = 0 \end{cases}$$
   - Naming rationale: Distinct from `proxide.jaccard` (which computes MinHash signature distances over sketch Parquets).
   - Invariants and boundary conditions:
     - Disjoint sets ($A \cap B = \emptyset, A \cup B \neq \emptyset$) return `0.0`.
     - Identical non-empty sets ($A = B \neq \emptyset$) return `1.0`.
     - Both sets empty ($A = \emptyset, B = \emptyset$) return `0.0`.
     - Output is bounded in $[0.0, 1.0]$.
     - Supports residue sets composed of integers or `(int, str)` tuples with insertion codes.

2. **`aligned_contact_overlap`**:
   - Signature:
     ```python
     def aligned_contact_overlap(
         contacts_a: set[Any] | frozenset[Any],
         contacts_b: set[Any] | frozenset[Any],
         aligned_pairs: Sequence[tuple[int, int]],
     ) -> tuple[float, int, int]:
     ```
   - Projects contacted residue sets from two species/orthologs (e.g. human ortholog A and mouse ortholog B) onto aligned reference positions in ortholog A numbering:
     $$H = \{a \mid (a, b) \in \text{aligned\_pairs} \land a \in \text{contacts\_a}\}$$
     $$M = \{a \mid (a, b) \in \text{aligned\_pairs} \land b \in \text{contacts\_b}\}$$
   - Returns:
     $$\big(\text{contact\_jaccard}(H, M),\, |H|,\, |M|\big)$$
   - Unaligned residues (such as human residue 212 / `L212`, which exists in the human crop but has no counterpart in mouse) are strictly dropped from both projection sets and do not participate in Jaccard calculation.
   - Accepts sets of integers or `(int, str)` residue tuples.

#### 4.6.2 Alignment Numbering Table Parser
- Signature:
  ```python
  def load_aligned_pairs(
      csv_text_or_path: str | Path,
      col_a: str = "human_resnum",
      col_b: str = "mouse_resnum",
      in_crop_col: str | None = "in_crop",
      aligned_col: str | None = None,
  ) -> list[tuple[int, int]]:
  ```
- Input Type Disambiguation:
  - If `isinstance(csv_text_or_path, str)` and (`"\n"` in `csv_text_or_path` or not Path(csv_text_or_path).exists()), parses directly as CSV text via `io.StringIO`.
  - Otherwise, opens `csv_text_or_path` as a filesystem path (`Path(csv_text_or_path).open()`).
- Header Validation:
  - Inspects the CSV header. If `col_a`, `col_b`, `in_crop_col` (when not `None`), or `aligned_col` (when not `None`) is missing from the header, raises `KeyError(f"Required column '{col}' not found in CSV header")`.
- Filtering and Row Parsing Rules:
  - Both `col_a` and `col_b` must be non-empty and non-whitespace.
  - If `in_crop_col` is provided, row must satisfy `str(row[in_crop_col]).strip() in ("1", "True", "true")`.
  - If `aligned_col` is provided, row must satisfy `str(row[aligned_col]).strip() in ("1", "True", "true")`.
  - Passing rows are validated for integer residue numbers: if `int(row[col_a])` or `int(row[col_b])` fails, raises `ValueError(f"Could not parse integer residue number in row: {row}")`.
- Empty CSV Contract:
  - An empty file, whitespace-only string, or header-only CSV returns an empty list `[]`.
- Authoritative fixture contract: on `tfr1_numbering_table.csv`, returns exactly 194 tuples in crop; human residue 212 has an empty `mouse_resnum` and is strictly omitted.

#### 4.6.3 Contact and Epitope Docking
- Signature:
  ```python
  def contacts_and_dock(
      target_coords: np.ndarray,
      target_resids: np.ndarray | Sequence[int],
      binder_coords: np.ndarray,
      *,
      target_offset: int = 0,
      cutoff: float = 5.0,
      epitope_residues: set[int] | frozenset[int] | None = None,
      min_epitope_contacts: int = 4,
      min_interface_residues: int = 10,
  ) -> dict[str, Any]:
  ```
- Shape and Length Invariants:
  - `target_resids` must have shape `(N,)` where $N = \text{target\_coords.shape}[0]$ (exactly one residue identifier per target atom coordinate).
  - Validation: if `len(target_resids) != target_coords.shape[0]`, raises:
    `ValueError(f"Length mismatch: target_coords has {target_coords.shape[0]} atoms, but target_resids has {len(target_resids)}")`.
- Boundary Conditions:
  - If `target_coords` has 0 atoms (`shape[0] == 0`), raises `ValueError("target_coords cannot be empty")`.
  - If `binder_coords` has 0 atoms (`shape[0] == 0`), returns `{"contacted": [], "n_contacted": 0, "epitope_contacts": 0, "docked": False}` without error.
- Computes heavy-atom minimum Euclidean distances between binder atoms and target atoms.
- Identifies contacted target residue sequence numbers having $\ge 1$ heavy atom within `cutoff` (default 5.0 Å) of any binder heavy atom.
- Applies `target_offset` to all contacted residue numbers:
  $$\text{contacted} = \text{sorted}\left(\{r + \text{target\_offset} \mid r \in \text{raw\_contacted}\}\right)$$
- Computes `n_contacted = len(contacted)`.
- If `epitope_residues` is provided, computes:
  $$\text{epitope\_contacts} = |\text{set}(\text{contacted}) \cap \text{epitope\_residues}|$$
  (defaults to `0` if `epitope_residues is None`).
- Evaluates binary docking verdict:
  $$\text{docked} = \text{bool}(\text{epitope\_contacts} \ge \text{min\_epitope\_contacts} \land \text{n\_contacted} \ge \text{min\_interface\_residues})$$
- Returns dictionary:
  ```python
  {
      "contacted": contacted,
      "n_contacted": n_contacted,
      "epitope_contacts": epitope_contacts,
      "docked": docked,
  }
  ```

#### 4.6.4 Batching Contract: `batched == looped`
- Single and batched execution:
  ```python
  def contacts_and_dock_batch(
      target_coords: np.ndarray,
      target_resids: np.ndarray | Sequence[int],
      binder_coords_list: Sequence[np.ndarray],
      **kwargs: Any,
  ) -> list[dict[str, Any]]:
  ```
- Equivalence Assertion:
  For any sequence of binder coordinate sets `[b_1, ..., b_M]`:
  ```python
  batched = contacts_and_dock_batch(target_coords, target_resids, binder_coords_list, **kwargs)
  looped = [contacts_and_dock(target_coords, target_resids, b, **kwargs) for b in binder_coords_list]
  assert batched == looped
  ```
- Implementation Optimization Note:
  `contacts_and_dock_batch` SHOULD construct `cKDTree(target_coords)` once and query against each binder coordinate array, guaranteeing mathematical equivalence to looping while avoiding redundant KDTree rebuilds across binders.

---

### 4.7 Self-Association (SA) Homodimer Screen

#### 4.7.1 Interface Residue Counting
- Signature:
  ```python
  def interface_counts(
      atoms_or_coords_a: Any,
      atoms_or_coords_b: Any,
      cutoff: float = 5.0,
  ) -> tuple[int, int]:
  ```
- Accepts atom record lists, coordinate arrays with residue mapping, or parsed PDB models for chains A and B.
- Computes pairwise heavy-atom Euclidean distances.
- Returns `(n_if_a, n_if_b)`: count of unique residues on chain A within `cutoff` of chain B, and unique residues on chain B within `cutoff` of chain A.

#### 4.7.2 Association Criterion
- Signature:
  ```python
  def associates(
      n_a: int,
      n_b: int,
      ipsae: float | None,
      min_if_residues: int = 12,
      min_ipsae: float = 0.4,
  ) -> bool:
  ```
- Mathematical criterion:
  $$\text{associates} = \text{bool}\left(n_a \ge \text{min\_if\_residues} \land n_b \ge \text{min\_if\_residues} \land (\text{ipsae} \neq \text{None} \land \neg\text{isnan}(\text{ipsae}) \land \text{ipsae} \ge \text{min\_ipsae})\right)$$
- If `ipsae` is `None` or `math.isnan(ipsae)`, returns `False`.

#### 4.7.3 Run Aggregation and Bathos Schema
- Signature:
  ```python
  def screen_self_association(
      runs: Sequence[Mapping[str, Any]],
      *,
      min_if_residues: int = 12,
      min_ipsae: float = 0.4,
      monomeric_max_rate: float = 0.20,
      self_associates_min_rate: float = 0.50,
  ) -> dict[str, Any]:
  ```
- Input `runs` is a sequence of run records (e.g. parsed from `tfr1_sa_runs.tsv`).
  Each run contains:
  - `group` (primary column name, matching `tfr1_sa_runs.tsv`) or `role` (fallback): `"pos_ctl"` (e.g. GCN4-p1), `"neg_ctl"` (e.g. ubiquitin), `"design"` (candidate binders), or `"shuffle"` (scrambled controls).
  - `design_id`: design identifier string.
  - `valid`: optional validity flag (`int`, `str`, or `bool`). Default is 1 (valid).
  - `n_if_a`: interface residue count on chain A.
  - `n_if_b`: interface residue count on chain B.
  - `ipsae`: predicted interface ipSAE score (`float` or `None`).
  - optional `assoc`: existing boolean/integer flag (if omitted, evaluated via `associates(n_if_a, n_if_b, ipsae, min_if_residues, min_ipsae)`).
  - optional `problem`: error or status description string.
- Run Validity Contract:
  - A run record is considered valid iff `int(row.get("valid", 1)) != 0` and no unhandled error occurred.
  - Invalid runs must be counted in `n_problems` and appended to `problems` (e.g. `f"{row.get('design_id', 'unknown')}: {row.get('problem', 'invalid run')}"`).
  - Invalid runs are strictly excluded from the denominators and numerators of `design_rate` and `shuffle_rate`.
  - Numeric parsing of `n_if_a`, `n_if_b`, and `ipsae` must be safely guarded against empty strings (`""`) or `None` on invalid rows.
- Control validation rules:
  - Positive control: at least 1 valid run of positive control must associate (`n_pos_ctl_assoc >= 1`).
  - Negative control: zero valid runs of negative control may associate (`n_neg_ctl_assoc == 0`).
  - `controls_ok = bool(n_pos_ctl_assoc >= 1 and n_neg_ctl_assoc == 0)`.
- Candidate and shuffle rates:
  - `design_rate = n_design_assoc / max(1, n_valid_design_runs)`
  - `shuffle_rate = n_shuffle_assoc / max(1, n_valid_shuffle_runs)`
- Control Key Conventions:
  - Emits generic keys `"n_pos_ctl_assoc"` and `"n_neg_ctl_assoc"`.
  - Retains domain-specific aliases `"n_gcn4_assoc"` and `"n_ubiquitin_assoc"` for full backward compatibility with `sa_result.json`.
- Bathos `[result_schema]` summary return:
  ```json
  {
    "controls_ok": true,
    "design_rate": 0.2,
    "n_design_assoc": 2,
    "n_pos_ctl_assoc": 2,
    "n_neg_ctl_assoc": 0,
    "n_gcn4_assoc": 2,
    "n_ubiquitin_assoc": 0,
    "n_problems": 0,
    "n_runs": 24,
    "n_shuffle_assoc": 0,
    "problems": [],
    "rows_complete": true,
    "selftest_all_pass": true,
    "shuffle_rate": 0.0,
    "outcome": "monomeric"
  }
  ```
- Bathos Outcome Decoupling:
  `"outcome"` is emitted by `screen_self_association` for bathos outcome evaluation (matching registered bathos run `20aeafaf` in `scripts/design/tfr1_sa_screen.bth.toml`). When asserting exact equality against the historical golden fixture `sa_result.json` (which omitted `"outcome"`), `"outcome"` is either excluded or asserted against the bathos run record outcome `"monomeric"`.

#### 4.7.4 Decision Logic and Outcomes
- Categorical 3-tier `outcome` determination matching registered Bathos sidecar:
  1. If `not controls_ok` or `not rows_complete`: `outcome = "inconclusive"` (experiment rejected due to control failure or incomplete rows).
  2. Else if `design_rate <= monomeric_max_rate` (default $0.20$): `outcome = "monomeric"` (binder population qualifies as predominantly monomeric).
  3. Else if `design_rate >= self_associates_min_rate` (default $0.50$): `outcome = "self_associates"` (binder population exceeds tolerable self-association frequency).
  4. Else (`monomeric_max_rate < design_rate < self_associates_min_rate`): `outcome = "inconclusive"` (intermediate association frequency; ambiguous or indeterminate outcome).

---

### 4.8 Buried Binder Surface Area (`buried_binder_a2` / SASA)

#### 4.8.1 Algorithm: Zero-Dependency Pure-NumPy Shrake-Rupley
- To eliminate heavy external dependencies (Biopython, Biotite, FreeSASA) while maintaining machine precision against registered goldens, `proxide.geometry.contacts` implements a pure-NumPy Shrake-Rupley (1973) numerical surface integration.

#### 4.8.2 Atomic Radii and Sphere Generation
1. **Bondi Atomic Radii**:
   Standard Bondi (1964) van der Waals radii in Ångströms:
   ```python
   BONDI_RADII: dict[str, float] = {
       "H": 1.20,
       "C": 1.70,
       "N": 1.55,
       "O": 1.52,
       "S": 1.80,
       "P": 1.80,
       "F": 1.47,
       "CL": 1.75,
       "BR": 1.85,
       "I": 1.98,
   }
   DEFAULT_RADIUS: float = 2.00
   ```
   - **Element Resolution Rules**:
     - When reading PDB files, columns 76–78 are read for `element`. If empty or whitespace, infer element from `name[0].upper()` for amino acid heavy atoms (`C, N, O, S`). Backbone alpha carbons named `"CA"` must strictly be treated as Carbon (`"C"`), never assigned Calcium (`"Ca"`).
     - For atom dict inputs, `element` is read from `atom.get("element")` or inferred from `atom["name"][0].upper()`.
     - Element lookup in `BONDI_RADII` is case-insensitive (`element.upper()`). If not found, defaults to `DEFAULT_RADIUS = 2.00` Å.
2. **Water Probe Radius**: $r_{\text{probe}} = 1.40$ Å.
3. **Effective Atomic Sphere Radius**: $R_i = r_{\text{vdw}, i} + r_{\text{probe}}$.
4. **Golden Spiral Sphere Points ($N = 100$)**:
   Points distributed uniformly on the unit sphere via Fibonacci golden section:
   $$\Delta \lambda = \pi (3 - \sqrt{5}), \quad \Delta z = \frac{2}{N}$$
   $$z_k = 1 - \frac{\Delta z}{2} - k \cdot \Delta z, \quad r_k = \sqrt{1 - z_k^2}, \quad \lambda_k = k \cdot \Delta \lambda$$
   $$u_k = (r_k \cos \lambda_k,\, r_k \sin \lambda_k,\, z_k) \quad \text{for } k = 0, \dots, N-1$$
5. **Accessibility Test**:
   For atom $i$ at coordinate $C_i$, the $k$-th test point is $P_{i,k} = C_i + R_i u_k$.
   Point $P_{i,k}$ is solvent-accessible if and only if for all neighbor atoms $j \neq i$ with $\|C_i - C_j\| < R_i + R_j$:
   $$\|P_{i,k} - C_j\|^2 \ge R_j^2$$
6. **Atomic Solvent Accessible Surface Area**:
   $$\text{SASA}_i = 4 \pi R_i^2 \times \frac{\sum_{k=0}^{N-1} \mathbb{I}(P_{i,k} \text{ accessible})}{N}$$

#### 4.8.3 Binder Free vs Complex SASA Difference ($\Delta \text{SASA}_A$)
- Primary functions:
  ```python
  def compute_shrake_rupley(
      coords: np.ndarray,
      elements: Sequence[str],
      probe_radius: float = 1.40,
      n_points: int = 100,
  ) -> np.ndarray:
      ...

  def compute_buried_sasa(
      structure: str | Path | list[dict[str, Any]],
      chain_a: str = "A",
      chain_b: str = "B",
      probe_radius: float = 1.40,
      n_points: int = 100,
  ) -> tuple[float, int, dict[str, Any]]:
  ```
- Evaluates binder chain A in isolation: $\text{SASA}_A^{\text{free}} = \sum_{i \in \text{Chain A}} \text{SASA}_i^{\text{free}}$.
- Evaluates binder chain A in the presence of target chain B: $\text{SASA}_A^{\text{complex}} = \sum_{i \in \text{Chain A}} \text{SASA}_i^{\text{complex}}$.
- Buried Binder Surface Area ($\Delta \text{SASA}_A$):
  $$\Delta \text{SASA}_A = \text{SASA}_A^{\text{free}} - \text{SASA}_A^{\text{complex}}$$
- Exact Return Schema:
  `compute_buried_sasa` returns `(delta_sasa_float, delta_sasa_int, details_dict)` where `details_dict` has the exact schema:
  ```python
  {
      "free_sasa": float,
      "complex_sasa": float,
      "delta_sasa": float,
      "residue_sasa_free": dict[Any, float],
      "residue_sasa_complex": dict[Any, float],
      "residue_sasa_buried": dict[Any, float],
  }
  ```
  where `residue_sasa_free[r]` is the free SASA of residue $r$, `residue_sasa_complex[r]` is the complex SASA of residue $r$, and `residue_sasa_buried[r] = residue_sasa_free[r] - residue_sasa_complex[r]`.

#### 4.8.4 Physical Invariant S1: Negative SASA Gain Check
- **Conservation of Solvent Accessibility**: When a binder associates with a target, steric occlusion can only shield surface points; no residue on chain A may become more solvent-accessible upon forming the complex.
- **Invariant Assertion**:
  $$\text{SASA}_{A, r}^{\text{complex}} \le \text{SASA}_{A, r}^{\text{free}} + 10^{-5} \quad \forall r \in \text{Residues}(A)$$
- If any binder residue violates this condition ($\text{SASA}_{A, r}^{\text{complex}} > \text{SASA}_{A, r}^{\text{free}} + 10^{-5}$), the calculation must fail immediately with:
  `ValueError(f"complex SASA exceeds free SASA for {len(bad)} binder residues {bad[:5]}; SASA physical control failed")`.
  Here `bad` contains the identifiers of the violating residues formatted as `num` or `(num, ins_code)`.

#### 4.8.5 Precision Contract: Integer Rounding vs Unrounded Float
- **Unrounded Numerical Parity**: Pure-NumPy Shrake-Rupley unrounded float output matches Biopython 1.88 `Bio.PDB.SASA.ShrakeRupley` across all atoms to within $< 10^{-6}$ Å².
- **Integer Storage Contract**: In registered bathos run `bba33e3a` (`tfr1_interface.tsv`), `buried_binder_A2` is stored as an integer:
  $$\text{buried\_binder\_a2} = \text{round}(\Delta \text{SASA}_A)$$
  Exact values across the 5 complex PDBs: `[764, 537, 467, 515, 485]`.
- Both float precision (`delta_sasa_float`) and integer rounded value (`round(delta_sasa_float)`) are returned.

#### 4.8.6 Wiring into `InterfaceContactResult` and `compute_interface_contacts`
- Field `buried_binder_a2: int | None = None` in `InterfaceContactResult`.
- Parameter `compute_sasa: bool = False` in `compute_interface_contacts(...)`.
- When `compute_sasa=True`:
  - `InterfaceContactResult.buried_binder_a2` holds the rounded integer ($\text{round}(\Delta \text{SASA}_A)$ in Å²).
  - `InterfaceContactResult.details["buried_binder_a2_float"]` holds the exact unrounded float value $\Delta \text{SASA}_A$ in Å².
  - `InterfaceContactResult.details["sasa_details"]` holds the complete `details_dict` from `compute_buried_sasa`.

---

### 4.9 Positive and Negative Selftest Gates for Slice 2 Utilities

#### 4.9.1 Interface Jaccard & Alignment Overlap Selftests
1. **Positive Selftests**:
   - `contact_jaccard({1, 2, 3}, {2, 3, 4}) == 2 / 4 == 0.5`.
   - Identical sets `{1, 2, 3}` and `{1, 2, 3}` yield `contact_jaccard == 1.0`.
   - Aligned overlap with human contacts `{200, 212}` and mouse contacts `{200}` with aligned pairs `[(200, 200)]` yields `(1.0, 1, 1)` (verifying unaligned residue 212 is excluded from both ortholog projections).
   - `contacts_and_dock` on synthetic target/binder coordinates:
     - Target: 15 collinear atoms along x-axis at 4.0 Å spacing: `coords = np.array([[i * 4.0, 0.0, 0.0] for i in range(15)])`, `resids = np.array(list(range(1, 16)))`, `epitope_residues = {1, 2, 3, 4, 5}`.
     - Positive binder: 12 atoms placed at $y = 3.0$ Å near target atoms 0..11: `binder_coords = np.array([[i * 4.0, 3.0, 0.0] for i in range(12)])`.
     - Result yields `len(contacted) == 12`, `epitope_contacts == 5`, and `docked == True`.
2. **Negative Selftests**:
   - Disjoint sets `{1, 2}` and `{3, 4}` yield `contact_jaccard == 0.0`.
   - Both sets empty `set()` and `set()` yield `contact_jaccard == 0.0`.
   - Far binder: 12 atoms placed at $y = 50.0$ Å (`np.array([[i * 4.0, 50.0, 0.0] for i in range(12)])`) yields 0 contacts, `docked == False`.
   - Epitope shortfall: 7 atoms placed near target atoms 8..14 (`np.array([[i * 4.0, 3.0, 0.0] for i in range(8, 15)])`) yields 7 contacts but 0 epitope contacts (< 4), `docked == False`.
   - Interface shortfall: 3 atoms placed near target atoms 0..2 (`np.array([[i * 4.0, 3.0, 0.0] for i in range(3)])`) yields 3 contacts (< 10), `docked == False`.

#### 4.9.2 Self-Association Screen Selftests
1. **Positive Selftests**:
   - Synthetic dimer test: two identical peptide helical backbones placed at contact distance ($3.5$ Å separation) yield $N_A \ge 12, N_B \ge 12$; with `ipsae = 0.65`, `associates == True`.
   - Positive control run GCN4-p1 (`n_if_a=18, n_if_b=18, ipsae=0.4234`) yields `assoc == True`.
   - Standard pass slate yields `controls_ok == True`, `design_rate == 0.2 <= 0.20`, and `outcome == "monomeric"`.
2. **Negative Selftests**:
   - Synthetic monomer test: two chains placed at non-interacting distance ($40.0$ Å separation) yield $N_A = 0, N_B = 0$; yields `associates == False`.
   - Negative control run ubiquitin (`n_if_a=16, n_if_b=16, ipsae=0.0000`) yields `assoc == False` (guarded by `ipsae >= min_ipsae`).
   - If positive control fails (`n_pos_ctl_assoc == 0`), `screen_self_association` returns `controls_ok == False` and `outcome == "inconclusive"`.
   - If candidate association rate exceeds threshold (`design_rate = 0.60 >= 0.50`), returns `outcome == "self_associates"`.
   - If candidate association rate is intermediate (`design_rate = 0.35` where $0.20 < 0.35 < 0.50$), returns `outcome == "inconclusive"`.

#### 4.9.3 Buried Binder SASA Selftests
1. **Positive Selftests**:
   - Single isolated carbon atom ($r_{\text{vdw}} = 1.70$ Å) yields area $4 \pi (1.70 + 1.40)^2 \approx 120.76$ Å².
   - Bound complex structures yield positive buried area: $\Delta \text{SASA}_A > 0$.
   - S1 physical check passes without error on all standard test complexes (`complex_sasa <= free_sasa + 1e-5`).
2. **Negative Selftests**:
   - Infinitely separated chains (offset $+500$ Å) yield $\Delta \text{SASA}_A == 0.0$ Å².
   - Artificial negative buried area (complex SASA artificially inflated over free SASA by $> 10^{-5}$ Å²) raises `ValueError`.

---

## 5. Parity Verification and Test Fixtures

### 5.1 Test Fixture Manifest (`tests/validation/fixtures/tfr1/`)
All test fixtures are stored under `tests/validation/fixtures/tfr1/` in the `proxide` repository.

| Fixture Path | SHA256 Hash | Purpose |
|---|---|---|
| `tests/validation/fixtures/tfr1/tfr1_final_slate.csv` | `346c70e21e626c33fa803f18346a1b5b7caebb3615766e36b58670ff45671b5f` | 5 delivered TfR1 designs for `dda7de75` parity |
| `tests/validation/fixtures/tfr1/tfr1_f2_candidates.tsv` | `fc4c470cf19b6ca7ea692c99dc6d2a0f758fce317c176d1557f135b464a5fa88` | Full TfR1 F2 candidate pool (594 designs) for `dda7de75` parity |
| `tests/validation/fixtures/tfr1/tfr1_final_order.tsv` | `bc5b1570564b2847c457b9cb056d70192ad065755a8d25d7dd5758f41d4b2003` | Ordering and rank metadata for `bba33e3a` audit |
| `tests/validation/fixtures/tfr1/composition_result.json` | `66377549e3564c7287eb599bee4464523b0cdf796b9a48b00c6fbf6b608f91a6` | Golden registered output for `dda7de75` |
| `tests/validation/fixtures/tfr1/tfr1_composition.tsv` | `681506b5bdafe439fe24705726bd270671bd2d289c3c30203deca14657805692` | Golden per-group TSV output for `dda7de75` |
| `tests/validation/fixtures/tfr1/interface_result.json` | `109e0a6a65897612be337180ba9027891357c10eb81745625eaaa53e7a893b79` | Golden registered summary for `bba33e3a` |
| `tests/validation/fixtures/tfr1/tfr1_interface.tsv` | `f90040b53322d346a4a7847a1ee025618877288a924e5b779526e7585199ba6c` | Golden per-complex interface audit table |
| `tests/validation/fixtures/tfr1/tfr1_numbering_table.csv` | `a7cd98283012c7e0ebe8e67946796c5c7ab3ccf4ed801c398b66b4b18ce54e1c` | Human vs mouse apical domain alignment numbering table (194 aligned pairs) |
| `tests/validation/fixtures/tfr1/sa_result.json` | `10573f96e26c4970bf8deb8ba7277047a8f6694260a26640dba0850f593bb088` | Golden registered summary for bathos run `20aeafaf` (SA screen) |
| `tests/validation/fixtures/tfr1/tfr1_sa_runs.tsv` | `b6c61fa896c710ef3ec9ac47bccf6e8d90ab06320756ec2289c2eb41678e6f5b` | Golden 24-run homodimer interface and ipSAE table for `20aeafaf` |
| `tests/validation/fixtures/tfr1/tfr1_sa_designs.tsv` | `0af7f574eb0d68051617d19a83188a67d95a02c66bff14d6d8739e9eba165806` | Golden per-design association table for `20aeafaf` |
| `tests/validation/fixtures/tfr1/p3_result.json` | `0f08eecaccdcfa22e9b2d45343d06d1a31acdfc6fb57e6c11daf24ed593c27cb` | Golden registered summary for bathos run `5d906f67` (P3 mouse overlap) |
| `tests/validation/fixtures/tfr1/tfr1_p3_mouse.tsv` | `e484e9305b2ad84e77ee1b4fda673f99a249270fc763de3a4645d487211a200e` | Golden 14-candidate mouse cross-reactivity and overlap summary |
| `tests/validation/fixtures/tfr1/tfr1_p3_mouse_runs.tsv` | `2eb426825e9afb4e8b16cdbc2b6999e86d44360551f7dd74665df0b61b9bd7c2` | Golden 84-run predictor contact overlap table |
| `tests/validation/fixtures/tfr1/structures/f2c006r13.pdb` | `35b33d416edc6b6991765a6b74082c9d1593a2cce68b42c4412e9b89d0a10ab9` | F3 Complex PDB (Rank 5) |
| `tests/validation/fixtures/tfr1/structures/f2c008r18.pdb` | `cc6c71cbd338809d5def07d78e9a609b418e422852cafad714c0b76dfea0e001` | F3 Complex PDB (Rank 2) |
| `tests/validation/fixtures/tfr1/structures/f2c016r5_H32.pdb` | `816fd52830793b5b600dacc831ba0ab6fc455911032fa0dd4f9e9820680cc073` | F3 Complex PDB (Rank 4) |
| `tests/validation/fixtures/tfr1/structures/f2c020r4_H62.pdb` | `0802ec32057fa83af2302a9b03fbaf087a45b34e8edd375524a64bfbb4a0e1b2` | F3 Complex PDB (Rank 1 parent) |
| `tests/validation/fixtures/tfr1/structures/f2c025r15_H60.pdb` | `c6004dd2bf964c98f30880e4513c44c1ec38d8acd6875b13c1afa2c1de78d5fd` | F3 Complex PDB (Rank 3) |

### 5.2 Parity Scope for `dda7de75` (Composition)
- Evaluated against `tfr1_final_slate.csv`.
- Must match the `"final_five"` group of `composition_result.json` and row 6 of `tfr1_composition.tsv`:
  - `n == 5`
  - `mean_len == 82.4`
  - `mean_ek == 0.4517`
  - `frac_ek_gt_30pct == 1.0`
  - `mean_net_charge == -8.5`
  - `frac_net_neg == 1.0`
  - Per-amino acid frequencies matching `mean_freq` within tolerance.

### 5.3 Parity Scope for `bba33e3a` (Interface Audit and Buried SASA)
- Evaluated against `tfr1_final_order.tsv`, `tfr1_interface.tsv`, `interface_result.json`, and the 5 complex PDB structures in `structures/`.
- Summary parity with `interface_result.json`:
  - `selftest_all_pass == true`
  - `total_ion_pairs == 21`
  - `total_like_charge_close == 2`
- Per-complex parity with `tfr1_interface.tsv`:
  - Contact counts: `n_binder_interface`, `n_target_interface`, `ion_pairs`, `like_close`, `polar_pairs`, `nonpolar_pairs`.
  - Charges and fractions: `binder_if_charge`, `target_if_charge`, `binder_if_frac_EK`, `binder_if_frac_hydrophobic`.
  - Detailed fields: `target_charge_within_6A`, `motif_contacted`, `target_if_residues`.
  - **Buried Binder Surface Area (`buried_binder_A2`)**: Fully un-excluded in Slice 2. Evaluated via pure-NumPy Shrake-Rupley ($\Delta \text{SASA}_A = \text{SASA}_A^{\text{free}} - \text{SASA}_A^{\text{complex}}$) with Bondi radii, probe radius $1.40$ Å, and $N=100$ Fibonacci sphere points, rounded to the nearest integer. Matches registered golden values exactly:
    - `f2c020r4_H62.pdb` (Rank 1): $763.70 \to \mathbf{764}$ Å²
    - `f2c008r18.pdb` (Rank 2): $536.80 \to \mathbf{537}$ Å²
    - `f2c025r15_H60.pdb` (Rank 3): $467.01 \to \mathbf{467}$ Å²
    - `f2c016r5_H32.pdb` (Rank 4): $515.42 \to \mathbf{515}$ Å²
    - `f2c006r13.pdb` (Rank 5): $485.03 \to \mathbf{485}$ Å²
- Decoupling rule: Human residue offset `+188` and epitope set `MOTIF = {202..215, 348, 371}` reside inside `tests/validation/test_bathos_parity.py`.

### 5.4 Parity Scope for `20aeafaf` (Self-Association Screen)
- Evaluated against `tfr1_sa_runs.tsv` (24 runs), `tfr1_sa_designs.tsv`, and `sa_result.json`.
- Per-run association parity:
  - For each of the 24 rows in `tfr1_sa_runs.tsv`, `associates(row["n_if_a"], row["n_if_b"], row["ipsae"]) == int(row["assoc"])`.
  - Positive control (GCN4-p1): 2/2 associate (`n_pos_ctl_assoc == 2` / `n_gcn4_assoc == 2`, condition `pos_ctl >= 1` satisfied).
  - Negative control (ubiquitin): 0/2 associate (`n_neg_ctl_assoc == 0` / `n_ubiquitin_assoc == 0`, condition `neg_ctl == 0` satisfied).
  - Controls valid: `controls_ok == True`.
  - Candidate designs: 2/10 associate (`n_design_assoc == 2`, `design_rate == 0.2`).
  - Shuffles: 0/10 associate (`n_shuffle_assoc == 0`, `shuffle_rate == 0.0`).
- Summary parity with `sa_result.json`:
  - Emits generic control keys `"n_pos_ctl_assoc"` and `"n_neg_ctl_assoc"` along with backwards-compatible aliases `"n_gcn4_assoc"` and `"n_ubiquitin_assoc"`.
  - `controls_ok == true`
  - `design_rate == 0.2`
  - `n_design_assoc == 2`
  - `n_pos_ctl_assoc == 2`
  - `n_neg_ctl_assoc == 0`
  - `n_gcn4_assoc == 2`
  - `n_ubiquitin_assoc == 0`
  - `n_shuffle_assoc == 0`
  - `shuffle_rate == 0.0`
  - `selftest_all_pass == true`
  - `rows_complete == true`
  - Outcome contract: `screen_self_association` emits `outcome == "monomeric"`, matching the registered Bathos run record `20aeafaf`. When validating against historical fixture `sa_result.json` (which did not serialize an `"outcome"` key), `"outcome"` is either asserted against the Bathos run record or excluded from exact fixture dictionary comparisons.
- Synthetic 3D coordinate checks:
  - Synthetic dimer (separation $3.5$ Å, `ipsae=0.65`): `n_if_a >= 12, n_if_b >= 12, associates == True`.
  - Synthetic monomer (separation $40.0$ Å): `n_if_a == 0, n_if_b == 0, associates == False`.

### 5.5 Parity Scope for `5d906f67` (P3 Mouse Cross-Reactivity & Overlap)
- Evaluated against `tfr1_numbering_table.csv`, `tfr1_p3_mouse.tsv` (14 candidates), `tfr1_p3_mouse_runs.tsv` (84 runs), `p3_result.json`, and the 5 complex PDBs.
- Numbering table alignment parity:
  - `load_aligned_pairs` parses exactly 194 aligned pairs in crop.
  - Verifies human residue 212 (`L212`) has no mouse counterpart and is strictly unaligned.
- Human complex heavy-atom contacts:
  - `contacts_and_dock` on the 5 complex PDBs at 5.0 Å heavy-atom cutoff yields total human contacted residues matching `n_human_contacted`:
    `[11, 12, 12, 18, 14]` for `[f2c006r13, f2c008r18, f2c016r5_H32, f2c020r4_H62, f2c025r15_H60]`.
  - Human residue 212 is contacted in all 5 complexes.
  - Aligned contact count `n_h_aligned_contacts = n_human_contacted - 1` matches `[10, 11, 11, 17, 13]` exactly.
- Per-run and per-candidate overlap:
  - Evaluates `aligned_contact_overlap` across all 84 runs in `tfr1_p3_mouse_runs.tsv` and 14 candidates in `tfr1_p3_mouse.tsv`.
  - Zero designs dock against mouse apical crop (`n_designs_mouse_docked_any == 0`) due to high sequence divergence at key epitope positions.
  - Median and maximum mouse overlap Jaccard equal $0.0$.
- Summary parity with `p3_result.json`:
  - `n_designs == 14`
  - `n_designs_complete == 14`
  - `n_mouse_runs_valid == 84`
  - `n_designs_mouse_docked_any == 0`
  - `median_overlap == 0.0`
  - `max_overlap == 0.0`
  - `n_overlap_at_or_above_floor == 0`
  - `selftest_all_pass == true`
  - `rows_complete == true`
- Bathos registered outcome: `"pass"`.

### 5.6 Ephemeral Scratch Path Documentation
- Historical exploratory logs and sidecars from the initial campaign referenced scratch download paths:
  - `/home/marielle/scratch/sa_modal_pull` (ephemeral CIF pulls for self-association folds)
  - `/home/marielle/scratch/p3_modal_pull` (ephemeral CIF pulls for mouse cross-reactivity folds)
- **Status on Disk**: These scratch paths were temporary Modal volume pulls located on ephemeral scratch storage and **no longer exist on disk**.
- **Reproducibility Guarantee**: All production artifacts, run tables, summary metrics, and structural models required for full parity verification across runs `dda7de75`, `bba33e3a`, `20aeafaf`, and `5d906f67` are fully committed as version-controlled test fixtures in `tests/validation/fixtures/tfr1/` and tracked by SHA256 hashes in Table 5.1. No external network access or scratch paths are required.

### 5.7 Numerical Tolerances
- Integer counts (`n`, `ion_pairs`, `like_close`, `polar_pairs`, `nonpolar_pairs`, residue counts, `buried_binder_a2` integer): exact match (`atol = 0`).
- Unrounded float SASA: `atol = 0.5` Å² (Biopython exact match $< 10^{-6}$ Å²).
- Frequencies and fractions (`mean_ek`, `frac_ek_gt_30pct`, amino acid frequencies, Jaccard): `atol = 1e-4`.
- Net charge (`mean_net_charge`): `atol = 0.05`.
- Averages and lengths: `atol = 1e-3`.

---

## 6. Open Decisions and Constraints

1. **Rust Extension Build**: Accepted by user (2026-11-02). `proxide` builds with maturin via `cargo` and `uv`.
2. **Local Test Execution Limit**: Run only narrow test commands (`pytest tests/validation/test_bathos_parity.py` or `-k` selector) with threads capped at 4.
3. **No Predictor or Cluster Calls**: Purely deterministic, stateless mathematical and geometric algorithms.
4. **Registered Runs Untouched**: Registered TfR1 scripts and results remain immutable.
