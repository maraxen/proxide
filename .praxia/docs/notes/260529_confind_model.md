---
name: 260529_confind_model
description: Comprehensive algorithmic model of ConFind (Mosaist mstcondeg) — contact degree, freedom, interference, parallelization hazards
metadata:
  type: reference
---

# ConFind Algorithmic Model

**Source**: `Grigoryanlab/Mosaist` @ `450816a` — `src/mstcondeg.cpp` (634 lines), `include/mstcondeg.h` (212 lines), `tests/testConFind.cpp` (280 lines, also the main CLI).

ConFind computes three families of per-residue and per-pair quantities for a protein structure:

| Family | Quantities |
|--------|-----------|
| Contact degree | Pairwise sidechain-sidechain coupling probability (CD) |
| Freedom / Crowdedness | Per-residue rotamer freedom score; backbone crowdedness |
| Interference / BB-interaction | Directional SC→BB clash; BB→BB minimum distance |

---

## 1. Parameters (`setParams`, lines 56–70)

```
dcut         = 25.0 Å      CA–CA cutoff beyond which pairs are ignored
clashDist    = 2.0 Å       atom–atom distance defining a backbone clash
contDist     = 3.0 Å       atom–atom distance defining a sidechain contact
doNotCountCB = true        CB excluded from sidechain (except ALA)
aaNames      = 18 AAs      all standard AAs minus GLY and PRO
loCollProbCut = 0.5        freedom type 2/3 low threshold
hiCollProbCut = 2.0        freedom type 2/3 high threshold
freedomType   = 2
```

**Amino-acid propensities** (percent background frequency, lines 63–66):

| AA | % | AA | % | AA | % |
|----|---|----|---|----|---|
| ALA | 7.73 | ILE | 5.66 | SER | 6.13 |
| ARG | 5.03 | LEU | 8.83 | THR | 5.53 |
| ASN | 4.50 | LYS | 6.27 | TRP | 1.51 |
| ASP | 5.82 | MET | 2.08 | TYR | 3.54 |
| CYS | 1.84 | PHE | 4.05 | VAL | 6.91 |
| GLN | 3.94 | PRO | 4.52 | (GLY 7.11) |
| GLU | 6.61 | HIS | 2.35 | |

(GLY and PRO are in `aaProp` but not in `aaNames`; they are never placed as rotamers.)

---

## 2. Initialization (`init`, lines 87–98)

```
S.getAtoms()  →  filter non-H backbone atoms  →  backbone (AtomPointerVector)
                                               →  ca subset
bbNN = ProximitySearch(backbone, clashDist/2)  // radius 1.0 Å grid
caNN = ProximitySearch(ca, dcut/2)             // radius 12.5 Å grid
```

Two spatial grids are built once for the whole structure. Both are read-only after construction — safe to share across threads.

---

## 3. Per-Residue Cache (`cache(Residue*)`, lines 100–190)

Early-return if already cached (line 102). Pipeline per residue:

### 3a. Rotamer placement and backbone-clash pruning

```
for aa in aaNames:                              // 18 amino acids
  rotamerHeavySC[res][aa] = NULL               // sentinel
  nr = rotLib->numberOfRotamers(aa, phi, psi)  // from BB-dep library
  pointCloud, pointCloudTags = [], []          // atom coords + rotamerID* per aa
  for ri in 0..nr:
    rID = rotLib->placeRotamer(*res, aa, ri, &rot)
    rotP = rotLib->rotamerProbability(aa, ri, phi, psi)
    prune = false
    for atom in rot.sidechain_heavy_atoms():   // countsAsSidechain filter
      closeOnes = bbNN->pointsWithin(atom.getCoor(), 0.0, clashDist)
      for ci in closeOnes:
        if backbone[ci].getResidue() != res:
          prune = true
          if aa == "ALA": permanentContacts[res].insert(ci)  // record all
          else: break
        // accumulate interference (regardless of prune):
        resB = backbone[ci].getResidue()
        if resB != res and resB not in seen:
          interference[res][resB][aa] += aaP * rotP / 100.0
      if prune: break
    if prune: continue
    survivingRotamers[res].push_back(new rotamerID(rID))
    for atom in rot.sidechain_heavy_atoms():
      pointCloud.push_back(new Atom(atom))
      pointCloudTags.push_back(rotTag)
    numRemRotsInPosition++
  if pointCloud.size() > 0:
    rotamerHeavySC[res][aa] = new DecoratedProximitySearch<rotamerID*>(
        pointCloud, contDist/2, pointCloudTags)  // grid spacing 1.5 Å
  pointCloud.deletePointers(); pointCloudTags.clear()
  totNumRotsInPosition += nr
fractionPruned[res] = (tot - rem) / tot    // line 188
numLibraryRotamers[res] = tot              // line 189
```

### 3b. `countsAsSidechain` (lines 192–196)
Excludes: hydrogens, backbone atoms (N/CA/C/O/H), CB (unless residue is ALA when `doNotCountCB=true`).

### 3c. Parallelizability of `cache`
**Fully independent across residues.** Each residue reads from `bbNN`/`caNN` (read-only), `rotLib` (read-only), and writes only into maps keyed by its own `Residue*`. No cross-residue writes during caching. → **rayon `par_iter` safe after restructuring into `Arc<RwLock<HashMap>>` or split-phase patterns.**

---

## 4. Contact Degree (`contactDegree`, lines 207–281)

### 4a. Cache-check and setup (lines 208–220)
```
no_aa_restriction = aaAllowedA.empty() && aaAllowedB.empty()   // BUG: L209 checks A twice
if no_aa_restriction && degrees[resA][resB] exists && !updateA && !updateB:
  return cached value
cache(resA) if cacheA; cache(resB) if cacheB
if checkNeighbors && CA_distance(resA,resB) > dcut: return 0.0
```

### 4b. Rotamer-pair collision detection (lines 222–253)
```
clashing: Map<rotamerID*, Map<rotamerID*, bool>>
for resA_aa in aaAllowedA:
  cloudA = rotamerHeavySC[resA][resA_aa]    // DecoratedProximitySearch
  for resB_aa in aaAllowedB:
    cloudB = rotamerHeavySC[resB][resB_aa]
    if not cloudA->overlaps(*cloudB, contDist): continue   // bbox prefilter
    for ai in 0..cloudA->pointSize():
      p = cloudB->getPointsWithin(cloudA->getPoint(ai), 0, contDist)
      rID = cloudA->getPointTag(ai)
      for rotB in p: clashing[rID][rotB] = true
```

### 4c. Contact degree formula (lines 255–274)

```
cd = 0.0
for (rotA, rotBset) in clashing:
  rotProbA = rotLib->rotamerProbability(rotA)
  aaPropA  = aaProp[rotA->aminoAcid()]
  for rotB in rotBset:
    rotProbB = rotLib->rotamerProbability(rotB)
    aaPropB  = aaProp[rotB->aminoAcid()]
    cd += aaPropA * aaPropB * rotProbA * rotProbB
    if updateA: collProb[resA][rotA] += aaPropB * rotProbB   // ← MUTABLE SHARED STATE
    if updateB: collProb[resB][rotB] += aaPropA * rotProbA   // ← MUTABLE SHARED STATE

denom = weightOfAvailableRotamers(resA, aaAllowedA) * weightOfAvailableRotamers(resB, aaAllowedB)
cd = (denom == 0) ? 0 : cd / denom

// symmetric caching when no_aa_restriction:
degrees[resA][resB] = degrees[resB][resA] = cd               // ← MUTABLE SHARED STATE
```

**`weightOfAvailableRotamers`** (lines 519–529):
```
sum over survivingRotamers[res] where aa in available_aa:
  aaProp[aa] * rotLib->rotamerProbability(rot)
```

**`weightOfAvailableAminoAcids`** (lines 531–537):
```
sum(aaProp[aa] for aa in available_aa) / 100.0
```

---

## 5. Contact Collection and Freedom (`getContacts(vector<Residue*>)`, lines 306–342)

This is the **serial bottleneck**. The sequential structure is load-bearing:

```
cache(all residues)
for i, resi in residues:
  collProbUpdateOn(resi)                  // flag: update collProb[resi] during CD calls
  neighborhood = caNN->getPointsWithin(resi.CA, 0, dcut)
  for resj in neighborhood:
    if resi != resj and not checked[resi][resj]:
      if resj in ofInterest: collProbUpdateOn(resj)
      checked[resj][resi] = true
      cd = contactDegree(resi, resj, false, true, false)  // no extra cache/neighbor check
      if cd > cdcut: list->addContact(resi, resj, cd)
      if resj in ofInterest: collProbUpdateOff(resj)
  collProbUpdateOff(resi)
  if collProb[resi] missing: collProb[resi] = {}
  computeFreedom(resi)                    // requires collProb[resi] COMPLETE
```

**Why this must currently run serially:** `collProb[resi][rotA]` is accumulated across all of resi's neighbor contacts before `computeFreedom(resi)` is called. If contacts were computed in parallel, multiple threads would race on `collProb[resA][rotA]` and `collProb[resB][rotB]` simultaneously.

---

## 6. Freedom (`computeFreedom`, lines 550–588)

Operates on `collProb[res]`, `survivingRotamers[res]`, `numLibraryRotamers[res]`.

```
cp = collProb[res]     // rotamers that have ANY collision probability mass
n_uncontested = survivingRotamers[res].size() - cp.size()
```

| freedomType | Formula |
|-------------|---------|
| 1 | `(n_uncontested + count(cp where val/100 < 0.5)) / numLibraryRotamers` |
| 2 (default) | `sqrt((n1² + n2²) / 2) / numLibraryRotamers` |
| 3 | `sqrt((n2² + n2*n1) / 2)` (already normalized internally) |

where `n1 = n_uncontested + count(val/100 < loCollProbCut=0.5)`,
      `n2 = n_uncontested + count(val/100 < hiCollProbCut=2.0)`.

**Crowdedness** = `fractionPruned[res]` (already computed during `cache`; no dependency on collProb).

---

## 7. Interference (`interferenceValue`, lines 433–448)

Populated during `cache` (line 158): `interference[resA][resB][aa] += aaP * rotP / 100.0`

Query:
```
in = sum(interference[resA][resB][aa] for aa in aaAllowed)
in /= weightOfAvailableAminoAcids(aaAllowed)   // = sum(aaProp[aa])/100
```

Directional: `getInterference` scans all `(resA, resB)` pairs in both directions; `getInterfering` scans only where resA is the source.

---

## 8. Backbone Interaction (`bbInteraction`, lines 450–466)

```
min over all (N,CA,C,O) atom pairs from resA × resB:
  Euclidean distance
```

`getBBInteraction` skips pairs within `ignoreFlanking=1` positions on the same chain (adjacent residues).

---

## 9. Constrained Contacts (`getConstrainedContacts`, lines 344–367)

For each `(resi, resj)` in neighborhood, iterates over all 18 `aaNames` at `resi` with `resj` unconstrained, producing up to 18 (not 324 — only one direction implemented) CD values per pair.

---

## 10. CLI I/O Contract (from `testConFind.cpp`)

**Inputs:**
- `--p <pdb>` or `--pL <list>` — protein structure(s)
- `--rLib <path>` — MSL rotamer library (backbone-dependent if directory, independent if file)
- Optional: `--sel`, `--o`, `--oL`, `--opdb`, `--rout`, `--verb`, `--pp`, `--omg`, `--ren`, `--seq_const`, `--freeB`

**Output format** (tab-delimited, one record per line):
```
contact       chain,resnum  chain,resnum  degree    resA_name  resB_name
crwdnes       chain,resnum  crowdedness   resA_name
freedom       chain,resnum  freedom       resA_name
interference  chain,resnum  chain,resnum  degree    resA_name  resB_name
seq_const_contact  chain,resnum  chain,resnum  degree  resA_name  resB_name  aaA  XXX
SEQUENCE: ALA GLY ...
```

**Amino acid filter**: only standard 20 + HSD/HSE/HSC/HSP/MSE/CSO/HIP/SEC/SEP/TPO/PTR residues are kept; heteroatoms/ligands dropped.

---

## 11. Parallelization Hazard Map

| Data structure | Written by | Thread-safety | Strategy for Rust |
|----------------|-----------|---------------|-------------------|
| `bbNN`, `caNN` | `init` once | Read-only after init | `Arc<ProximitySearch>` |
| `rotamerHeavySC[res]` | `cache(res)` | Per-residue independent | rayon par_iter, write to distinct slots |
| `survivingRotamers[res]` | `cache(res)` | Per-residue independent | same |
| `fractionPruned[res]` | `cache(res)` | Per-residue independent | same |
| `interference[res][…]` | `cache(res)` | Per-residue independent | same |
| `degrees[resA][resB]` | `contactDegree` | **RACE** — symmetric write | Use `DashMap` or post-join merge |
| `collProb[res][rot]` | `contactDegree` (when update flag set) | **RACE** — accumulated across neighbor loop | Phase-split: accumulate thread-locally, merge after parallel CD pass |
| `freedom[res]` | `computeFreedom` | Per-residue, but depends on complete `collProb[res]` | Compute after `collProb` fully merged |

### Recommended Rust parallelization phases

1. **Phase A — cache** (`rayon::par_iter` over residues): fully independent.
2. **Phase B — contact degree** (`rayon::par_iter` over pairs): each pair produces `(rotA, rotProbA, aaPropA, rotB, rotProbB, aaPropB)` tuples; accumulate `collProb` in per-residue thread-local maps, merge after phase.
3. **Phase C — freedom** (sequential or per-residue parallel, after merge): `computeFreedom` over all residues.
4. **Phase D — output** (parallel over residues): crowdedness, interference lookups all read-only.

---

## 12. Dependency Substrate Required in Rust

| Module | Size estimate | Must port |
|--------|--------------|-----------|
| Structure + Residue + Atom + Chain | ~1200 LOC | Yes |
| CartesianPoint (3D vector) | ~200 LOC | Yes |
| ProximitySearch (3D grid) | ~600 LOC | Yes |
| DecoratedProximitySearch\<T\> | ~300 LOC | Yes — generic over rotamerID ref |
| RotamerLibrary (load, place, prob) | ~1500 LOC | Yes |
| rotamerID (AA + bin + rot index) | ~100 LOC | Yes |
| Frame + Transform (4×4 matrix) | ~500 LOC | Yes (rotamer placement only) |
| PDB reader | — | Reuse existing `proxide-io` |

**Total new substrate** if not already in proxide: ~4400 LOC. Much of the structure/geometry substrate likely already exists in `proxide-core` and `proxide-geometry`.
