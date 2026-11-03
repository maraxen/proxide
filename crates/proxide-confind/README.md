# proxide-confind

Sequence-independent contact degree, crowdedness, and structural freedom algorithm for protein backbones.

Part of the [proxide](https://github.com/maraxen/proxide) workspace.

## Overview

`proxide-confind` implements the sequence-independent contact degree metric for protein residue pairs, following the biophysical models of Zheng, Zhang, & Grigoryan (2015) and Zheng & Grigoryan (2017). Given a protein backbone and a backbone-dependent rotamer library, it runs a three-phase pipeline:

- **Phase A — Cache:** for each residue, enumerates rotamers across the 18 canonical amino acids with discrete rotameric freedom, prunes rotamers clashing with the backbone (distance <= 2.0 Å), and evaluates crowdedness $C(i) = 1.0 - (\Omega_i / \Omega_{\text{lib}})$.
- **Phase B — Contacts:** for candidate pairs within the 25.0 Å Cα–Cα cutoff, enumerates cross-rotamer contacts (distance <= 3.0 Å) in parallel (`orx-parallel`) weighted by $P(a) P(b) P(r_a) P(r_b)$ and normalized by $\Omega_i \Omega_j$.
- **Phase C — Freedom:** aggregates collision probabilities into a scalar residual conformational freedom value $F(i)$ per residue.

Contact-degree values are used downstream (e.g. by dTERMen) to identify residue pairs that are "poised to interact" ($c(i, j) \ge 0.02$).

## Key Types

- `ConFind` — main pipeline orchestrator; holds the rotamer library, protein backbone, spatial acceleration, and caches
- `ResidueIndex` — strongly-typed 0-based index into the backbone residue array
- `ResidueBackbone` — per-residue backbone geometry (`Option<f64>` dihedrals, standard IUPAC omega)
- `ProteinBackbone` — full-protein backbone container; constructed from PDB input with preserved file order
- `ContactList` — collection of residue-pair contact-degree values with filtering and lookup utilities
- `ConFindError` — error type covering preconditions, missing atoms, and I/O failures

## Usage

```rust
use std::path::Path;
use std::sync::Arc;
use proxide_rotlib::RotamerLibrary;
use proxide_confind::{ConFind, ResidueIndex, load_pdb_f64, CONTACT_THRESHOLD};

let rotlib = Arc::new(RotamerLibrary::load_pb_zst(Path::new("rotlib.pb.zst")).unwrap());
let backbone = Arc::new(load_pdb_f64(Path::new("protein.pdb")).unwrap());
let cf = ConFind::new(rotlib, backbone, false);

let residues = vec![ResidueIndex(0), ResidueIndex(1)];
let contacts = cf.contacts(&residues, 0.0).unwrap();
let filtered = contacts.filter(CONTACT_THRESHOLD);
```

## References

- Zheng, F., Zhang, J., & Grigoryan, G. (2015). "Tertiary Structural Propensities Reveal Fundamental Sequence/Structure Relationships in Proteins." *Structure*, 23(5), 961–971.
- Zheng, W., & Grigoryan, G. (2017). "Sequence statistics of tertiary structural motifs reflect protein stability." *PLoS ONE*, 12(5), e0178272.
