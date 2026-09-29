# Real wwPDB fixture provenance

Sprint 25 (task 260922_autonomous-loop, track a, decision i / debt #1919-#1920).
These three files are real, unmodified wwPDB depositions fetched directly from
RCSB so the fail-loud PQR/mmCIF rewrite has at least one real-world file per
format to snapshot and independently atom-count against, instead of only
hand-written synthetic fixtures.

All three files are released by wwPDB under the **CC0 1.0 Universal** public
domain dedication (https://www.rcsb.org/pages/policies) -- no license
restriction on redistribution.

| File | Source URL | Retrieved (UTC) | sha256 |
|---|---|---|---|
| `1CRN.cif` | https://files.rcsb.org/download/1CRN.cif | 2026-09-25T20:20:28Z | `23787562c427d7c1abe5420e86d5f1d0a6c7007dec1e8ce85645a6d69c32e8ba` |
| `1CRN.pdb` | https://files.rcsb.org/download/1CRN.pdb | 2026-09-25T20:20:28Z | `42199a30a0701864a2a5cc76cd7f35cc544cd0e65fbcf63e03c166543249b811` |
| `1UBQ.cif` | https://files.rcsb.org/download/1UBQ.cif | 2026-09-25T20:20:28Z | `056f98710cb2b36f633c45e41902a02eb446e82871da21ff2dd44f74a56ca0f6` |

Notes:
- `1CRN.cif`'s `_entry.id` is `1CRN`; `1UBQ.cif`'s `_entry.id` is `1UBQ` --
  verified to match the filename (guards against a stale/renamed download).
- `1CRN.pdb` and `1CRN.cif` are the same deposition in both legacy PDB and
  mmCIF format (327 ATOM/HETATM rows each, verified independently by a plain
  line/row count -- see `pqr_cif_snapshot.rs`'s `independent_counts_match_*`
  tests) and are used for a cross-format parity test (decision i).
- `1UBQ.cif` (76-residue ubiquitin, 660 ATOM/HETATM rows) was chosen
  specifically because it has ordered water (`HOH`) HETATM records with
  `label_seq_id` = `.` (Inapplicable) and `auth_seq_id` present -- the
  fallback path decision f documents.
- No altloc (alternate-location) records are present in `1CRN.pdb`/`1CRN.cif`;
  `1UBQ.cif`'s waters carry fractional (<1.0) occupancy but are not altloc
  duplicates (checked: no repeated (chain, res_id, atom_name) triples).
