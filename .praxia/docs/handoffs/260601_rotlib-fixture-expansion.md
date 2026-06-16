---
title: proxide-rotlib — expand parity test fixtures beyond small.pdb
date: 260601
project: proxide
branch: main
head: 1e8b15c
suite: 77 passed, 1 ignored (cargo test -p proxide-rotlib)
---

## Session summary

Three commits landed on `worktree-mosaist-parity`, merged to `main` via fast-forward.

**`aa1b2c7`** — promoted `test_place_parity_mosaist` from `#[ignore]` stub to real
coordinate parity: ALA default-bin CB on synthetic backbone N=[0,0,0] CA=[1.458,0,0]
C=[1.980,1.418,0], reference `[1.977, -0.670, -1.262]` derived from rotlib.bin
f32→f64 + frame math, tolerance 1e-5.

**`1e8b15c`** — four changes bundled:
1. `test_load.rs`: added GLY + PRO to `AA_NAMES` (were missing), un-ignored
   `test_load_all_aa_names_present` and `test_num_rotamers_sentinel_positive`.
2. `test_placement.rs`: added `test_place_all_aa_smoke` — parametric over all 20 AAs:
   atom count == library `na`, finite coords, CB–CA in [1.2, 1.8] Å for non-GLY.
3. `tests/test_distogram.rs` (new): `test_distogram_chain_a_small_pdb` — loads
   `small.pdb` chain A backbone (ARG MET LYS GLN LEU GLU ASP, 7 residues), places
   rot_index=0 default-bin CB via real rotlib.bin, asserts 7×7 CB–CB distance matrix
   within 1e-6 Å of hardcoded reference.
4. `helpers.rs`: added `real_pdb_path()`, `BackboneResidue` struct,
   `parse_pdb_backbone()` — minimal fixed-width PDB ATOM parser (N/CA/C only).

Tech debt **#67** logged: Dunbrack / multi-rotlib / PRO cis-trans support.

---

## Open task for this session

**Goal:** expand parity test fixtures so the frame transform, backbone_bin lookup, and
placement pipeline are exercised on diverse real crystallographic backbone geometry.

### What's missing

| Gap | Why it matters |
|-----|----------------|
| No GLY in distogram | `na=0` zero-atom path untested under real backbone coords |
| No PRO in distogram | Ring geometry; first atom is CD not CB (`atoms=['CD','CB','CG']`) |
| Only one tiny PDB | 7 residues, no secondary structure diversity |
| Only sentinel phi/psi | `backbone_bin` grid lookup never exercised with real angles |
| Single chain only | No inter-chain distance check |

### Available fixtures (no download needed)

All in `/home/marielle/repos/mosaist/testfiles/`:

| File | Size | Notes |
|------|------|-------|
| `small.pdb` | 9.8 K | Already used. Chain A+B, 7 residues each |
| `heptad.0388_0001.pdb` | 8.9 K | Heptad repeat — inspect for GLY/PRO |
| `heptad.0388_0007.pdb` | 8.9 K | Second heptad |
| `heptad.0388_0014.pdb` | 2.5 K | Short heptad |
| `2ZTA.pdb` | 45.4 K | Leucine zipper — helical, backbone diversity |
| `fuserinput.pdb` | 14.5 K | Unknown content — inspect |
| `1DC7.pdb` | 164.7 K | Larger structure, broad AA coverage |
| `1DC8.pdb` | 168.4 K | Related to 1DC7 |

If none contain GLY or PRO with full backbone, fetch from RCSB. Keep ≤50 residues;
commit to `crates/proxide-rotlib/tests/fixtures/` and wire via env var.

### Reference computation — Python snippet (mirrors frame.rs exactly)

```python
import struct, math

def rstr(f):
    buf = b""
    while True:
        b = f.read(1)
        if not b or b == b'\x00': return buf.decode('ascii')
        buf += b

def ri32(f): return struct.unpack('<i', f.read(4))[0]
def rf32(f): return struct.unpack('<f', f.read(4))[0]

def load_rotlib(path):
    entries = {}
    with open(path, 'rb') as f:
        while True:
            aa = rstr(f)
            if not aa: break
            nc = ri32(f); na = ri32(f); nb = ri32(f)
            for _ in range(nc):
                for _ in range(4): rstr(f)
            atom_names = [rstr(f) for _ in range(na)]
            bins = [(rf32(f), rf32(f), rf32(f)) for _ in range(nb)]
            default_bin = max(range(nb), key=lambda i: bins[i][2])
            rotamers = []
            for _ in range(nb):
                nr = ri32(f)
                rots = []
                for _ in range(nr):
                    prob = rf32(f)
                    for _ in range(nc): rf32(f); rf32(f)
                    coords = [[rf32(f), rf32(f), rf32(f)] for _ in range(na)]
                    rots.append(coords)
                rotamers.append(rots)
            entries[aa] = {'atom_names': atom_names, 'nb': nb, 'na': na,
                           'default_bin': default_bin, 'rotamers': rotamers}
    return entries

def normalize(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v]
def cross(a, b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def sub(a, b): return [a[i]-b[i] for i in range(3)]

def backbone_frame(N, CA, C):
    x_raw = sub(CA, N)
    z_raw = cross(x_raw, sub(C, CA))
    y_raw = cross(z_raw, x_raw)
    return {'origin': CA, 'x': normalize(x_raw), 'y': normalize(y_raw), 'z': normalize(z_raw)}

def switch_frames(frm, to_frm):
    t2t = [to_frm['x'], to_frm['y'], to_frm['z']]
    t1  = [[frm['x'][r], frm['y'][r], frm['z'][r]] for r in range(3)]
    R   = [[sum(t2t[i][k]*t1[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    ori = sub(frm['origin'], to_frm['origin'])
    t   = [sum(t2t[i][k]*ori[k] for k in range(3)) for i in range(3)]
    return R, t

def apply_xform(R, t, p):
    return [sum(R[i][j]*p[j] for j in range(3)) + t[i] for i in range(3)]

def parse_pdb_backbone(path):
    residues = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM") or len(line) < 54: continue
            atom  = line[12:16].strip()
            if atom not in ('N','CA','C'): continue
            aa    = line[17:20].strip()
            chain = line[21]
            resq  = int(line[22:26])
            xyz   = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            key = (chain, resq)
            if key not in residues: residues[key] = {'aa': aa}
            residues[key][atom] = xyz
    return residues

def place_first_atom(entry, N, CA, C):
    """Place rot_index=0, default_bin. Returns None for GLY (na=0)."""
    if entry['na'] == 0: return None
    db = entry['default_bin']
    canonical = entry['rotamers'][db][0][0]  # first atom, rot 0
    lab = {'origin': [0.,0.,0.], 'x': [1.,0.,0.], 'y': [0.,1.,0.], 'z': [0.,0.,1.]}
    R, t = switch_frames(backbone_frame(N, CA, C), lab)
    return apply_xform(R, t, [float(x) for x in canonical])
```

Run against each PDB, print 49-value Rust const array, paste into test_distogram.rs.

### Test structure

```rust
// test_distogram.rs — add after existing test
const REF_DISTOGRAM_HEPTAD_0001: [f64; N*N] = [...];

#[test]
fn test_distogram_heptad_0001() {
    let lib = RotamerLibrary::load(&real_rotlib_path()).unwrap();
    let all = parse_pdb_backbone(&Path::new("...heptad.0388_0001.pdb"));
    let chain_a: Vec<_> = all.iter().filter(|r| r.chain == 'A').collect();
    // Build CB list — skip GLY (assert placed.atoms.is_empty()), find CB by name for PRO
    // Assert distogram within 1e-6 Å
}
```

### Priority order

1. Inter-chain small.pdb: chain B distogram (identical to chain A → same reference, free sanity check)
2. Pick whichever local fixture has the most AA diversity and secondary structure
3. Get GLY coverage — check fuserinput.pdb or fetch from RCSB
4. Get PRO coverage — check 2ZTA or fetch
5. (Stretch) pass real phi/psi angles to exercise backbone_bin grid lookup

### Do NOT touch

- Debt #67 (Dunbrack / multi-rotlib / PRO cis-trans) — logged, defer
- `test_place_parity_mosaist` ALA coordinate test — correct, leave it
- ConFind crate — out of scope

---

## Key files

| Path | Role |
|------|------|
| `crates/proxide-rotlib/tests/test_distogram.rs` | Add new `#[test]` functions here |
| `crates/proxide-rotlib/tests/helpers.rs` | `parse_pdb_backbone`, `real_pdb_path` already present |
| `/home/marielle/repos/mosaist/testfiles/` | Fixture PDB source |

## Env vars (CI portability)

- `ROTLIB_PATH` — path to rotlib.bin (default: mosaist testfiles)
- `PDB_PATH` — path to small.pdb (default: mosaist testfiles)
- Add per-fixture vars if committing PDBs to repo
