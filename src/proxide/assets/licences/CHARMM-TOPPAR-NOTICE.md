# CHARMM-derived force-field XMLs: licence status

This notice covers every XML that `FORCEFIELD-LICENCES.tsv` points here: the
CHARMM36 protein and nucleic-acid force field, CGenFF, the CHARMM water and
ion models, and the CHARMM Drude polarizable force field.

## What these files are

Each file is byte-identical to a file that OpenMM's maintainers distribute.
`FORCEFIELD-LICENCES.tsv` names the exact upstream path and commit for each
one.

- `charmm/*.xml` come from `openmm/openmmforcefields` (`openmmforcefields/ffxml/charmm/`).
- `openmm_bundled/charmm36*.xml`, `openmm_bundled/charmm36*/` and
  `openmm_bundled/charmm_polar_*.xml` come from `openmm/openmm`
  (`wrappers/python/openmm/app/data/`).

Both projects generated these XMLs by converting the CHARMM topology and
parameter ("toppar") files that the MacKerell lab publishes. The `<Source>`
elements in each XML name the original files: `toppar_c36_jul24.tgz`,
`toppar_c36_aug15.tgz`, `drude_toppar_2019.tgz` and `drude_toppar_2023.tgz`.

## Licence status

- **The conversion.** openmmforcefields distributes these XMLs under its MIT
  licence (`openmmforcefields-LICENSE`). OpenMM distributes them as part of
  its Python package, under section 1 (MIT) of `openmm-Licenses.txt`.
- **The underlying parameter data.** The MacKerell lab states no licence and
  no redistribution terms for the toppar files. The lab's pages
  (`mackerell.umaryland.edu/charmm_ff.shtml` and `warning.shtml`, fetched on
  2026-11-23) publish licence terms only for the CGenFF *program*: a free
  not-for-profit binary licence, or SilcsBio for for-profit users. The toppar
  files themselves, CGenFF's included, are offered as open downloads.

No stated licence is not a grant. proxide ships these files as
openmmforcefields and OpenMM do, by an owner decision of 2026-11-24 (spec
261102, decision D5a). Explicit written clearance from the MacKerell lab is
tracked as a follow-up that does not block releases (debt #3258).

## If you redistribute proxide

Treat these files as you would the same files from openmmforcefields or
OpenMM. If you need certainty about the parameter data, contact the MacKerell
lab (<https://mackerell.umaryland.edu/>).
