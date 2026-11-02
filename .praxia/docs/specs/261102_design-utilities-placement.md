---
title: 'Shared design-screening utilities and lesson enforcement: placement across proxide, alynxr, aminx and project specs'
description: 'Engineering specification for Slice 1 design-screening utilities in proxide: proxide.seqqc and proxide.geometry.contacts, with exact API signatures, batching contracts, selftest gates, and bathos parity verification'
status: hardened_round_1
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

### 1.2 Slice 1 Scope Boundary
- **In Scope for Slice 1**:
  1. `proxide.seqqc`: Pure-Python sequence quality control submodule (composition vs background, net charge, pI, GRAVY, aromatic fraction, sequence entropy, and a `screen()` function returning a dict shaped for bathos `[result_schema]`).
  2. `proxide.geometry.contacts`: Generic interface contact utility between two chains (heavy-atom cutoff, contact sets, residue counts, and ion-pair / like-charge / polar / nonpolar classification).
  3. Batched execution support (`batched == looped`) for both sequence and contact utilities.
  4. Positive and negative selftests for each utility.
  5. Parity verification tests in `tests/validation/test_bathos_parity.py` against registered bathos runs `dda7de75` (composition) and `bba33e3a` (interface audit, bounded to contact/charge contracts).
- **Deferred to Slice 2+ / Future Milestones**:
  - Design campaign spec skeletons and `spec_lint` rule extensions (L1, L8, L9, L10).
  - Predictor lineage, effective votes, and arm unit readers in `alynxr` (L4, L7).
  - Fixed-token MPNN recipes in `aminx`.
  - Contact-set Jaccard alignment mapping (`proxide.geometry.jaccard_contacts`).

---

## 2. Inventory: proxide vs TfR1 Functions

| Capability (TfR1 source) | In proxide today | Verdict |
|---|---|---|
| Composition vs background, net charge, aromatic fraction, entropy, GRAVY, pI (`tfr1_composition.py`: `freqs`, `charge`, `summarize`; `tfr1_mpnn_redesign.py`: `metrics`) | None [V] | **Slice 1: New submodule `proxide.seqqc`** |
| Interface contacts between two chains, per-residue contact sets (`tfr1_sa_screen.py: interface_counts`, `tfr1_interface_charge.py: pair_counts`, `tfr1_p3_mouse.py: contacts_and_dock`) | `geometry/metrics.py` functions raise `NotImplementedError` (6 stubs) [V]; `run_confind` was deleted in commit `47c397d5b` (debt #1901) [V] | **Slice 1: New utility `proxide.geometry.contacts`** (single shared implementation) |
| Ion-pair and like-charge contact classification (`tfr1_interface_charge.py: pair_counts`) | None [V] | **Slice 1: Included in `proxide.geometry.contacts`** |
| Contact-set Jaccard with alignment mapping (`tfr1_p3_mouse.py: jaccard, contact_overlap`) | `jaccard` module is MinHash-sketch distance over sourmash-style sketch Parquets, NOT contact sets [V] | **Deferred: Slice 2** (`proxide.geometry.contact_jaccard`) |
| TM-align, structure novelty | `tm_align, tm_scores_fixed_correspondence` exist [V] | **Exists**; use instead of foldseek-side reimplementation |
| AF vs MPNN alphabet conversion (`af_to_mpnn`) | `chem/conversion.py` (JAX) and `io/parsing/mappings.py` (NumPy) derive permutation tables via runtime dependency `alphex.perm`, pinned by `tests/test_alphabet_conformance.py` [V] | **Exists**; aminx uses its own `aminx.utils.aa_convert` |
| Vote logic with predictor family (`tfr1_p5b_check.py: vote, member_verdict`) | N/A | **Not proxide**: belongs to `alynxr` arm metadata |

---

## 3. Placement per Lesson

### 3.1 Slice 1 Placement
- **L2 (Sequence screen)** → `proxide.seqqc`: Pure-Python sequence QC functions, plus a `screen(sequences, background)` that returns a result dict matching the bathos `[result_schema]`.
- **L1 / L11 (Interface contacts and charge audit)** → `proxide.geometry.contacts`: Generic two-chain heavy-atom contacts, residue sets, and charge pair classifications.

### 3.2 Deferred Placement (Slice 2+)
- **L4 (Predictor lineage, effective votes), L7 (Pin tool commit, reader contract)** → `alynxr`: Add `family` and `lineage` to arm definitions, command reporting effective independent votes, and campaign reader contracts.
- **MPNN redesign with fixed target chain** → `aminx`: Recipe/CLI feature for fixed-token specs.
- **L1, L8, L9, L10 (Gate bypass, validated submission artifact, slate freeze date, amendment budget, null rounds)** → Design-campaign spec skeleton and `spec_lint` rules.
- **L3, L5, L6, L12 (Generator diff, organizer extensions marked, speculative arm date, mechanism requirements)** → Spec skeleton checklist items.

---

## 4. Slice 1 Detailed Specification

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
   - `"num"`: `int` (residue sequence number)
   - `"xyz"`: `tuple[float, float, float]` or `np.ndarray` of shape `(3,)`

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

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
- Batch utility: `compute_interface_contacts_batch(structures: list[...], ...) -> list[InterfaceContactResult]`.
- Equivalence Assertion:
  ```python
  batched = compute_interface_contacts_batch(structures)
  looped = [compute_interface_contacts(s) for s in structures]
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

## 5. Parity Verification and Test Fixtures

### 5.1 Test Fixture Manifest (`tests/fixtures/tfr1/`)
All test fixtures are stored under `tests/fixtures/tfr1/` in the `proxide` repository.

| Fixture Path | SHA256 Hash | Purpose |
|---|---|---|
| `tests/fixtures/tfr1/tfr1_final_slate.csv` | `346c70e21e626c33fa803f18346a1b5b7caebb3615766e36b58670ff45671b5f` | 5 delivered TfR1 designs for `dda7de75` parity |
| `tests/fixtures/tfr1/tfr1_final_order.tsv` | `bc5b1570564b2847c457b9cb056d70192ad065755a8d25d7dd5758f41d4b2003` | Ordering and rank metadata for `bba33e3a` audit |
| `tests/fixtures/tfr1/composition_result.json` | `66377549e3564c7287eb599bee4464523b0cdf796b9a48b00c6fbf6b608f91a6` | Golden registered output for `dda7de75` |
| `tests/fixtures/tfr1/tfr1_composition.tsv` | `681506b5bdafe439fe24705726bd270671bd2d289c3c30203deca14657805692` | Golden per-group TSV output for `dda7de75` |
| `tests/fixtures/tfr1/interface_result.json` | `109e0a6a65897612be337180ba9027891357c10eb81745625eaaa53e7a893b79` | Golden registered summary for `bba33e3a` |
| `tests/fixtures/tfr1/tfr1_interface.tsv` | `f90040b53322d346a4a7847a1ee025618877288a924e5b779526e7585199ba6c` | Golden per-complex interface audit table |

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

### 5.3 Parity Scope for `bba33e3a` (Interface Audit)
- Bounded strictly to interface contact and charge metrics:
  - `interface_result.json`: `selftest_all_pass == true`, `total_ion_pairs == 21`, `total_like_charge_close == 2`.
  - `tfr1_interface.tsv`: `n_binder_interface`, `n_target_interface`, `binder_if_charge`, `target_if_charge`, `binder_if_frac_EK`, `binder_if_frac_hydrophobic`, `target_charge_within_6A`, `motif_contacted`, `ion_pairs`, `like_close`, `polar_pairs`, `nonpolar_pairs`, and `target_if_residues`.
  - **Explicit Exclusion**: `buried_binder_A2` is excluded from Slice 1 parity verification.
- Decoupling rule: Human residue offset `+188` and epitope set `MOTIF = {202..215, 348, 371}` reside inside `tests/validation/test_bathos_parity.py`.

### 5.4 Numerical Tolerances
- Integer counts (`n`, `ion_pairs`, `like_close`, `polar_pairs`, `nonpolar_pairs`, residue counts): exact match (`atol = 0`).
- Frequencies and fractions (`mean_ek`, `frac_ek_gt_30pct`, amino acid frequencies): `atol = 1e-4`.
- Net charge (`mean_net_charge`): `atol = 0.05`.
- Averages and lengths: `atol = 1e-3`.

---

## 6. Open Decisions and Constraints

1. **Rust Extension Build**: Accepted by user (2026-11-02). `proxide` builds with maturin via `cargo` and `uv`.
2. **Local Test Execution Limit**: Run only narrow test commands (`pytest tests/validation/test_bathos_parity.py` or `-k` selector) with threads capped at 4.
3. **No Predictor or Cluster Calls**: Purely deterministic, stateless mathematical and geometric algorithms.
4. **Registered Runs Untouched**: Registered TfR1 scripts and results remain immutable.
