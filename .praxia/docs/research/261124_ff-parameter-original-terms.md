---
title: Original terms of the AMOEBA and GLYCAM parameters
description: 'Debt #3599 findings (Tinker licence for AMOEBA, no stated GLYCAM terms) and the status of the MacKerell clearance request (debt #3258)'
status: final
task_id: 261124_ff-licence-followups
date: '261124'
confidence: 'medium: primary sources quoted; the scope of the Tinker licence over parameter data is unresolved'
sources: 'TinkerTools/tinker LICENSE.pdf v26.2; openmm/openmm; openmm/openmmforcefields; glycam.org; ambermd.org'
---
# Original terms of the AMOEBA and GLYCAM parameters

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
`iamoeba.xml` has no `<Info>` block. Its author added it to OpenMM (see the
OpenMM review below) [V].

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

**OpenMM documentation and history (checked 2026-11-24, owner request: review
before any outreach).** All [V], from openmm/openmm on GitHub.

- *What OpenMM claims.* The user guide (`docs-source/usersguide/library/01_introduction.rst`)
  says: "The public API, the low level API, the reference platform, the CPU
  platform, and the application layer are all distributed under the MIT license."
  The XMLs live in the application layer (`openmm/app/data`). The guide adds that
  third-party code is "covered by other licenses ... similar in their terms to the
  MIT license". Neither the guide nor `Licenses.txt` (nine sections: API/platforms,
  CUDA/OpenCL, SFMT, Hilbert curve, AsmJit, PdbxReader, irrXML, L-BFGS, VkFFT)
  makes an exception for force-field data, Tinker or AMOEBA. OpenMM therefore
  presents the AMOEBA XMLs as MIT. Nowhere does it say it holds Tinker's
  permission to do so.
- *Who converted the files.* Peter Eastman (OpenMM lead) generated `amoeba2013.xml`
  from Tinker files in 2014–2015 (commit 1cbc2f3829 and later: "Regenerated
  AMOEBA2013 based on Tinker 7.0.1"). He added AMOEBA 2018 in PR #3299 (2021),
  which closed issues #2816 (David Case) and #3123. João Morado refreshed all of
  them in PR #5086 (2025). The 2009 file goes back to 2011 (Eastman, Mark
  Friedrichs).
- *iamoeba.xml.* Lee-Ping Wang added it ("Added iAMOEBA water parameters",
  fda320e65f, 2014-02-18). He is iAMOEBA's lead author; the 2013 paper has Ponder
  as a co-author. This resolves the open provenance question: the force field's
  own author contributed this file.
- *Collaboration with the Ponder lab.* The AMOEBA plugin chapter
  (`08_amoeba_plugin.rst`) describes "the AMOEBA polarizable atomic multipole
  force field from Jay Ponder's lab" and TINKER-OpenMM, "a modified version of
  TINKER ... that uses OpenMM", built from files on the OpenMM home page. This
  shows a long-running working relationship. It is not a licence statement.
- *Raw Tinker files in OpenMM's MIT repo.* At the pin, openmm/openmm carries
  `amoebabio18.prm` and `amoebapro13.prm` under `wrappers/python/tests/systems/`
  and `examples/`. It also carries raw CHARMM `par_all36_*.prm`. OpenMM
  redistributes the unconverted parameter files as well as the XMLs.
- *Tinker's licence when OpenMM converted.* The 2014 licence (Tinker 6.3,
  `doc/license.pdf`, commit b572ca1c50) is worded more broadly than today's: "The
  package and portions thereof may not be sold nor may copies be distributed to
  third parties without the express permission of the author and Washington
  University." The `.prm` files ship inside "the package". If express permission
  was given to OpenMM, no public record of it was found.
- *Discussion.* No openmm/openmm issue or PR discusses licensing AMOEBA, Tinker or
  any force-field data (GitHub issue search for license/licence with
  amoeba/tinker/"force field"/charmm).

**Reading.** The evidence is consistent with OpenMM distributing these files with
the Tinker developers' knowledge and involvement:

- OpenMM's lead generated the 2013 and 2018 files, and iAMOEBA's own author
  contributed his file.
- The Ponder lab co-maintains TINKER-OpenMM.
- AMOEBA has been in OpenMM since 2011, with no recorded objection.

None of this is a written grant. On the text, Tinker's licence restricts
redistribution of "the package and portions thereof" (2014) and of "the Software"
(2026). proxide's position is the same as for CHARMM under D5a: it ships what
OpenMM ships, under OpenMM's stated terms. The upstream ambiguity is documented,
not resolved.

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

## 3. Clearance request to the MacKerell lab (debt #3258)

A draft request exists, for the owner to send. It is deliberately kept out of
this public repository. Nothing has been sent.

## Open

- AMOEBA: owner decision 2026-11-24, no outreach for now. Keep shipping as OpenMM
  does, after the OpenMM due-diligence review above. Revisit if OpenMM's
  terms change, or if anyone asks.
- GLYCAM: an optional clearance request to the Woods lab, at lower priority.
- `glycam-hydrogens.xml`: provenance is unverified. (`iamoeba.xml` is resolved
  above.)
