# proxide-confind

Sequence-independent contact degree, crowdedness, and structural freedom algorithm for protein backbones.

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

`proxide-confind` implements the sequence-independent contact degree metric for protein residue pairs, following the biophysical models of Zheng, Zhang, & Grigoryan (2015) and Zheng & Grigoryan (2017). Given a protein backbone and a backbone-dependent rotamer library, it runs a three-phase pipeline:

- **Phase A — Cache:** for each residue, enumerates rotamers across the 18 canonical amino acids with discrete rotameric freedom, prunes rotamers clashing with the backbone of any other residue (distance <= `d_clash`, default 2.0 Å), and evaluates crowdedness $C(i)$ and the directed interference $I(i,j)$, summed over the unpruned library.
- **Phase B — Contacts:** for candidate pairs within the `d_cut` (default 25.0 Å) Cα–Cα cutoff, enumerates cross-rotamer contacts (distance <= `d_cont`, default 3.0 Å) in parallel (`orx-parallel`) weighted by $P(a) P(b) P(r_a) P(r_b)$ and normalized by $\Omega_i \Omega_j$.
- **Phase C — Freedom:** aggregates collision masses of the residues in the configured scope into a scalar residual conformational freedom value $F(i)$ per residue.

Contact-degree values are used downstream (e.g. by dTERMen) to identify residue pairs that are "poised to interact" ($c(i, j) \ge 0.02$).

## Key Types

- `ConFind` — main pipeline orchestrator; holds the rotamer library, protein backbone, spatial acceleration, and caches
- `ResidueIndex` — strongly-typed 0-based index into the backbone residue array
- `ResidueBackbone` — per-residue backbone geometry (`Option<f64>` dihedrals, standard IUPAC omega)
- `ProteinBackbone` — full-protein backbone container; constructed from PDB input with preserved file order
- `ContactList` — collection of residue-pair contact-degree values with filtering and lookup utilities
- `ConFindConfig`, `CbClashPolicy`, `CrowdednessMode`, `FreedomMode`, `ResidueScope`, `Cutoffs`, `ConfigError` — typed options and their validation errors
- `ConFindError` — error type covering preconditions, missing atoms, configuration, scope, and I/O failures

## Usage

```rust
use std::path::Path;
use std::sync::Arc;
use proxide_rotlib::RotamerLibrary;
use proxide_confind::{ConFind, ResidueIndex, load_pdb_f64, CONTACT_THRESHOLD};

let rotlib = Arc::new(RotamerLibrary::load_pb_zst(Path::new("rotlib.pb.zst")).unwrap());
let backbone = Arc::new(load_pdb_f64(Path::new("protein.pdb")).unwrap());
let cf = ConFind::new(rotlib, backbone, false).unwrap();

let residues = vec![ResidueIndex(0), ResidueIndex(1)];
let contacts = cf.contacts(&residues, 0.0).unwrap();
let filtered = contacts.filter(CONTACT_THRESHOLD);
```

## Configuration

Every behaviour-selecting parameter lives in one typed `ConFindConfig`, passed to `ConFind::with_config`
(`ConFind::new` uses `ConFindConfig::default()`). Options are orthogonal: any combination of valid values is legal.

| Option | Variants (kebab-case) | Default | Affects |
|---|---|---|---|
| `CbClashPolicy` | `ala-only`, `all-residues` | `all-residues` | which atoms are tested against backbone: Cβ for alanine only, or for every amino acid; changes the surviving rotamer sets and so every output |
| `CrowdednessMode` | `probability-weighted`, `pruned-count-fraction` | `probability-weighted` | `crowdedness` only: $C = 1 - W_\text{surv}/W_\text{lib}$ or $C = 1 - N_\text{surv}/N_\text{lib}$ |
| `FreedomMode` | `rms`, `single-threshold` | `rms` | `freedom` only: $F = \sqrt{(N_\text{lo}^2+N_\text{hi}^2)/2}/N_\text{lib}$ or $F = N_\text{lo}/N_\text{lib}$ |
| `ResidueScope` | `All`, `Subset(Vec<ResidueIndex>)` | `All` | `freedom` only: which residues count as neighbours; `freedom(i)` for `i` outside the scope is `ConFindError::OutOfScope` |
| `Cutoffs` | `d_clash`, `d_cont`, `d_cut`, `tau_lo`, `tau_hi` | 2.0 Å, 3.0 Å, 25.0 Å, 0.5, 2.0 | clash/contact/neighbour distances (inclusive) and the strict collision-mass thresholds of freedom |

```rust
use proxide_confind::{CbClashPolicy, ConFind, ConFindConfig, FreedomMode};

let config = ConFindConfig::default()
    .with_cb_clash_policy(CbClashPolicy::AlaOnly)
    .with_freedom_mode(FreedomMode::SingleThreshold);
let cf = ConFind::with_config(rotlib, backbone, false, config).unwrap();
let f = cf.freedom(ResidueIndex(3)).unwrap();
```

Validation never corrects silently; failures are `ConFindError::InvalidConfig(ConfigError)`:

- **V1** every cutoff finite and `> 0` (`InvalidCutoff`).
- **V2** `tau_lo < tau_hi` when the freedom mode is `rms` (`TauOrder`); `single-threshold` ignores `tau_hi` beyond V1.
- **V3** a `Subset` scope is non-empty (`EmptyScope`).
- **V4** parsing an unknown variant name (`UnknownVariant`).
- **V5** at construction, `d_cut >= 2 * max_a R_max(a) + d_cont`, with `R_max(a)` computed from the loaded rotamer library over every bin and rotamer (`DcutTooSmall`).
- **V6** every scope index is below the residue count (`ScopeOutOfRange`).

Definitions: interference $I(i,j)$ is the prior mass of the unpruned library rotamers of `i` that hit the backbone of `j`
(directed; every hit residue is credited); collision mass is the unnormalised $\sum P(b)/100 \cdot P(s)$ over contacting
surviving rotamers of neighbours; freedom is count-based and a rotamer with collision mass exactly `tau` is not counted for that `tau`.
An empty rotamer library for a residue (glycine) gives crowdedness and freedom `0`; there are no sentinel values.

## `confind` binary

```
confind --p <pdb> --rLib <library> [--o <file>] [--sel <spec>] [--cdcut <f64>] [--incut <f64>] [--threads <n>]
        [--cbclash ala-only|all-residues] [--crowdmode probability-weighted|pruned-count-fraction]
        [--freedommode rms|single-threshold] [--scope all|<residue list or file>]
        [--dclash <Å>] [--dcont <Å>] [--dcut <Å>] [--taulo <f64>] [--tauhi <f64>]
```

Flags may be given in any order. Defaults are shown in the table above. `--scope` takes `all`, a list such as `A:10-50,B`
(`chain:start-end`, `chain:number` or `chain`), or the path of a file with such tokens (`#` lines ignored); freedom lines are
printed for selected residues inside the scope. An invalid value or configuration exits non-zero with a message naming the option,
the value and the allowed values. The output begins with `# option<TAB>value` header lines recording every effective option, so a
run is reproducible from its output; a command line without the new flags produces the same output as one that sets each to its default.

## References

- Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
- Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs reflect protein stability." *PLoS ONE*, 12(5), e0178272.

## Licence and provenance

`proxide-confind` is an independent implementation of the contact-degree method in the papers cited above (Zheng, Zhang, & Grigoryan, 2015; Zheng & Grigoryan, 2017). It was written from the functional specification `.praxia/docs/specs/261102_clean-room-functional-specification-proxide-confind-proxide-rotlib-and-proxide_fixer.md`. The workspace `Cargo.toml` `license` field is `MIT`. The earlier Mosaist-derived port and its NOTICE were removed.
