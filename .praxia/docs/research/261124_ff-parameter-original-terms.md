---
title: Original terms of the AMOEBA and GLYCAM parameters, and a clearance request for CHARMM toppar
description: 'Debt #3599 findings (Tinker licence for AMOEBA, no stated GLYCAM terms) and a draft clearance email to the MacKerell lab for debt #3258'
status: draft
task_id: 261124_ff-licence-followups
date: '261124'
confidence: 'medium: primary sources quoted; the scope of the Tinker licence over parameter data is unresolved'
sources: 'TinkerTools/tinker LICENSE.pdf v26.2; openmm/openmm; openmm/openmmforcefields; glycam.org; ambermd.org'
---
# Original terms of the AMOEBA and GLYCAM parameters, and a clearance request for CHARMM toppar

proxide records every vendored force-field XML under the licence of the project
that redistributes it, OpenMM or openmmforcefields (both MIT; `FORCEFIELD-LICENCES.tsv`,
PR #106). Debt #3599 asks what terms the *original* parameter authors attach to the
AMOEBA and GLYCAM files. This is not legal advice.

Tags: **[V]** quoted from a source that was fetched, **[U]** unverified. The AMOEBA
licence quotes were checked twice, once by the research pass and again by
extracting `LICENSE.pdf` with `pdftotext`.

## 1. AMOEBA

**Files.** `openmm_bundled/amoeba{2009,2013,2018}{,_gk}.xml` and `iamoeba.xml`.
Their `<Source>` elements name Tinker parameter files: `amoebabio09.prm`,
`amoebapro13.prm` and `amoebabio18.prm` [V]. OpenMM converts them with
`processTinkerForceField.py` (openmm/openmm commit eaf56f96bc, #5086) [V].
`iamoeba.xml` has no `<Info>` block and its origin is unknown; it may come from the
SimTK iAMOEBA project [U].

**Stated terms.** The only licence text in TinkerTools/tinker is `LICENSE.pdf`
(Tinker v26.2, April 2026) [V]:

- Scope: "This Tinker license covers the source codes and executables comprising
  the three Tinker codes: Tinker, Tinker-HP and Tinker-OpenMM."
- Clause 1 grants "a non-exclusive, non-sublicensable, non-transferable, revocable,
  non-commercial license to use the object and source code versions of the
  software known as 'Tinker Software' (the 'Software'), for internal academic
  purposes only".
- Clause 2: the Licensee may not "make it available to third parties, without the
  prior written consent of at least one TCDC-Member."
- Clause 6: "The Licensee agrees, in perpetuity, not to disclose the Software to
  any third party ... unless the TCDC gives prior written consent."
- Clause 9: "Commercial use of the Software, or derivative works based thereon,
  REQUIRES A COMMERCIAL LICENSE." Commercial licences go through Qubit
  Pharmaceuticals; the contact is in the licence.

**Data or code?** Unclear. The licence speaks of "source codes and executables" and
never mentions parameter files. The `.prm` files carry no header, copyright or
licence line, and `params/0README` has no licence statement [V]. They ship inside
the licensed distribution, though, and nothing says they fall outside it.

**Redistribution in converted form.** Unclear, leaning restrictive. If the `.prm`
data count as "the Software", public MIT redistribution, which permits commercial
use, conflicts with clauses 1, 2, 6 and 9. If they are only data, the licence may
not reach them, but no source says so.

**Upstream practice.** OpenMM ships the AMOEBA XMLs in its MIT-licensed package.
Their headers give only the source and references, and OpenMM's `Licenses.txt` does
not mention Tinker or AMOEBA [V]. No OpenMM or Tinker issue about licensing these
parameters was found [V]. openmmforcefields does not ship AMOEBA.

**Bottom line.** #3599 is not closable for AMOEBA. Clearance would come from the
Tinker Core Development Consortium (Jay W. Ponder, WashU; Pengyu Ren, UT Austin;
Jean-Philip Piquemal, Sorbonne). Asking OpenMM (Peter Eastman) whether it holds
permission is the cheaper first step. Of the files proxide ships, this set has the
most restrictive text attached to its source distribution.

## 2. GLYCAM

**Files.** `amber/GLYCAM_06j-1.xml` (openmmforcefields), `openmm_bundled/amber14/GLYCAM_06j-1.xml`
and `openmm_bundled/glycam-hydrogens.xml` (OpenMM, added in openmm/openmm#3303) [V].
The XML header names `glycam/leaprc.GLYCAM_06j-1` from AmberTools 24.8 and cites
Woods-lab papers. It has no licence text [V].

**Stated terms.** None found [V for each source below].

- The GLYCAM leaprc and `.lib` files have no licence, copyright or permission text.
- `glycam.org/docs/forcefield/` has no licence or redistribution statement.
- ambermd.org says AmberTools components are "mostly released under the GNU General
  Public License (GPL)". It says nothing specific about GLYCAM.
- No openmm or openmmforcefields issue or PR discusses GLYCAM licensing.

Not read: the GLYCAM-Web terms of use, and the AmberTools per-file licence for
`dat/leap/*GLYCAM*` [U]. If the GPL covered those files, MIT redistribution of
derived data would raise a separate question; this is unconfirmed.

**Bottom line.** Not closable on evidence. No stated licence is not a grant.
Clearance would come from the Woods lab (Robert J. Woods, UGA CCRC) and/or the
Amber project. The risk is lower than for AMOEBA, because no restrictive text was
found. `glycam-hydrogens.xml` may be OpenMM's own work (hydrogen naming for
Modeller) [U].

## 3. Draft clearance request to the MacKerell lab (debt #3258)

For the owner to send. Nothing has been sent.

> **To:** MacKerell lab (contact via https://mackerell.umaryland.edu/)
> **Subject:** Permission to redistribute CHARMM36/CGenFF/Drude parameters (OpenMM XML) in an MIT-licensed Python package
>
> Dear Professor MacKerell,
>
> I maintain proxide (https://github.com/maraxen/proxide), an open-source
> Python/Rust library for protein structure I/O. It's published on PyPI under the
> MIT licence. To support simulation setup, proxide redistributes force-field files
> in OpenMM XML format. These include 28 files converted from your lab's CHARMM
> topology and parameter ("toppar") releases:
>
> - CHARMM36 protein and nucleic acids, including the CGenFF parameters
>   (`charmm36.xml`, `charmm36_cgenff.xml` and related files)
> - the CHARMM water and ion models (TIP3P/TIP4P/TIP5P/SPC/E variants)
> - the Drude polarizable force field (`charmm_polar_2013/2019/2023.xml`) and the
>   SWM4-NDP water model
>
> We did not convert these files ourselves. Each is a byte-identical copy of a file
> that the OpenMM project distributes in `openmm/openmmforcefields` (MIT) or
> `openmm/openmm` (MIT). Their headers name the source archives:
> `toppar_c36_jul24.tgz`, `toppar_c36_aug15.tgz`, `drude_toppar_2019.tgz` and
> `drude_toppar_2023.tgz`.
>
> Your website gives licence terms for the CGenFF program, but I could not find
> terms that cover redistributing the toppar parameter files themselves. We
> currently ship them as OpenMM does, with a notice that credits your lab and states
> that no licence is given for the underlying data. We would rather have your
> explicit permission.
>
> Could you tell me:
>
> 1. whether we may redistribute these parameter files, in this converted form, in
>    an MIT-licensed package that anyone (including commercial users) can install
>    from PyPI;
> 2. whether any part, such as the CGenFF parameters, needs different handling,
>    for example omitting it or putting it under separate terms; and
> 3. what attribution or citation you would like us to include.
>
> If you would prefer that we not distribute some or all of these files, we will
> remove them from the next release.
>
> Thank you for making these parameters available to the community.
>
> Best regards,
> [Your name]
> proxide maintainer

## Open

- AMOEBA: what to do while clearance is pending is an owner decision (keep as
  OpenMM does, or stop shipping).
- GLYCAM: an optional clearance request to the Woods lab, at lower priority.
- `iamoeba.xml` and `glycam-hydrogens.xml`: provenance is unverified.
