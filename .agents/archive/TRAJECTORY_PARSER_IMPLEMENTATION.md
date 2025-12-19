# Pure-Rust Trajectory Parser Implementation Plan

**Status:** In Progress (DCD & TRR Completed)  
**Priority:** Medium (blocked by chemfiles SIGFPE crashes)  
**Target:** Replace chemfiles-based DCD/TRR parsers with pure-Rust implementations

---

## Background

The current DCD and TRR trajectory parsers use the `chemfiles` crate, which crashes with SIGFPE on some Linux systems. The XTC parser was successfully migrated to pure-Rust via the `molly` crate. This document outlines the implementation plan for DCD and TRR.

---

## Format Specifications

### DCD (CHARMM/NAMD Trajectory)

**Structure:** Binary FORTRAN unformatted file

```text
┌──────────────────────────────────────────────────────────────┐
│ Header Block 1: File signature and metadata                  │
│ ├─ 4 bytes: Record length marker                             │
│ ├─ 4 bytes: "CORD" magic                                     │
│ ├─ 9 i32:   NFILE, NPRIV, NSAVC, NSTEP, etc.                │
│ ├─ f64:    DELTA (timestep)                                  │
│ ├─ 1 i32:   Unit cell flag                                   │
│ ├─ 8 i32:   Padding                                          │
│ ├─ 1 i32:   CHARMM version (24 for new format)              │
│ └─ 4 bytes: Record length marker                             │
├──────────────────────────────────────────────────────────────┤
│ Header Block 2: Title strings                                │
│ ├─ 4 bytes: Record length                                    │
│ ├─ i32:     NTITLE (number of 80-char title lines)          │
│ ├─ 80*N:    Title strings                                    │
│ └─ 4 bytes: Record length                                    │
├──────────────────────────────────────────────────────────────┤
│ Header Block 3: Atom count                                   │
│ ├─ 4 bytes: Record length                                    │
│ ├─ i32:     NATOM                                            │
│ └─ 4 bytes: Record length                                    │
├──────────────────────────────────────────────────────────────┤
│ Frame N (repeated for each timestep):                        │
│ ├─ [Optional] Unit cell (6 f64 values)                       │
│ ├─ X coordinates: NATOM * f32                                │
│ ├─ Y coordinates: NATOM * f32                                │
│ └─ Z coordinates: NATOM * f32                                │
└──────────────────────────────────────────────────────────────┘
```

**Key Challenges:**

- Multiple variants (CHARMM, NAMD, X-PLOR) with subtle differences
- Endianness detection required (Fortran can write either)
- Unit cell format varies between implementations
- Single precision (f32) coordinates in Angstroms

### TRR (GROMACS Trajectory)

**Structure:** XDR (eXternal Data Representation) encoded binary

```text
┌──────────────────────────────────────────────────────────────┐
│ Frame Header:                                                │
│ ├─ i32:     Magic number (1993)                              │
│ ├─ string:  Version string                                   │
│ ├─ i32:     ir_size (0 for trajectory)                       │
│ ├─ i32:     e_size (energy size)                             │
│ ├─ i32:     box_size (unit cell)                             │
│ ├─ i32:     vir_size (virial)                                │
│ ├─ i32:     pres_size (pressure)                             │
│ ├─ i32:     top_size (topology)                              │
│ ├─ i32:     sym_size (symmetry)                              │
│ ├─ i32:     x_size (coordinates size)                        │
│ ├─ i32:     v_size (velocities size)                         │
│ ├─ i32:     f_size (forces size)                             │
│ ├─ i32:     natoms                                           │
│ ├─ i32:     step                                             │
│ ├─ i32:     nre (number of energy terms)                     │
│ ├─ f32/f64: t (time in ps)                                   │
│ └─ f32/f64: lambda                                           │
├──────────────────────────────────────────────────────────────┤
│ Frame Data:                                                  │
│ ├─ [If box_size > 0] 3x3 f32 box matrix                      │
│ ├─ [If x_size > 0]   natoms * 3 * f32 coordinates (nm)       │
│ ├─ [If v_size > 0]   natoms * 3 * f32 velocities (nm/ps)     │
│ └─ [If f_size > 0]   natoms * 3 * f32 forces (kJ/mol/nm)     │
└──────────────────────────────────────────────────────────────┘
```

**Key Challenges:**

- Requires XDR decoding (big-endian, padded to 4 bytes)
- Optional velocities and forces
- Units are in nm (convert to Angstroms for proxide)
- Variable precision (f32 or f64 depending on flags)

---

## Implementation Strategy

### Phase 1: DCD Parser (Simpler)

DCD is a simpler format and should be implemented first.

#### New Files

```
oxidize/src/formats/dcd_pure.rs    [NEW]
```

#### API Design

```rust
pub struct DcdHeader {
    pub n_atoms: usize,
    pub n_frames: usize,
    pub timestep: f64,
    pub has_unit_cell: bool,
    pub is_charmm: bool,
}

pub struct DcdFrame {
    pub coordinates: Vec<[f32; 3]>,  // Angstroms
    pub box_vectors: Option<[[f64; 3]; 3]>,
    pub step: i32,
}

/// Parse DCD file header
pub fn parse_dcd_header(path: &str) -> Result<DcdHeader, DcdError>;

/// Parse all frames
pub fn parse_dcd_frames(path: &str) -> Result<Vec<DcdFrame>, DcdError>;

/// Streaming iterator for large files
pub fn dcd_frame_iterator(path: &str) -> Result<DcdFrameIterator, DcdError>;
```

### Phase 2: TRR Parser

TRR requires XDR decoding, which can be implemented using:

- **Option A:** Pure-Rust XDR implementation (recommended)
- **Option B:** `groan_rs` crate (complex API, but tested)

#### New Files

```
oxidize/src/formats/xdr.rs         [NEW] - XDR decoder utilities
oxidize/src/formats/trr_pure.rs    [NEW] - TRR parser
```

#### API Design

```rust
pub struct TrrHeader {
    pub n_atoms: usize,
    pub has_velocities: bool,
    pub has_forces: bool,
}

pub struct TrrFrame {
    pub coordinates: Vec<[f32; 3]>,      // Angstroms (converted from nm)
    pub velocities: Option<Vec<[f32; 3]>>,
    pub forces: Option<Vec<[f32; 3]>>,
    pub box_vectors: Option<[[f32; 3]; 3]>,
    pub time: f32,                        // ps
    pub step: i32,
}

/// Parse TRR file
pub fn parse_trr(path: &str) -> Result<TrrResult, TrrError>;
```

---

## Validation Strategy

### Reference Implementation: MDTraj

MDTraj is a well-tested Python library for trajectory analysis. It will serve as the reference implementation for validation.

**Why MDTraj:**

- Already a dev dependency
- Battle-tested across many trajectory files
- Easy to use Python API
- Lightweight compared to alternatives

### Test Data Strategy

Instead of generating synthetic data, we will use real trajectory files from the MDTraj repository to ensure we handle "in-the-wild" format quirks (like different DCD headers).

**Source:** [MDTraj GitHub Repository (tests/data)](https://github.com/mdtraj/mdtraj/tree/master/tests/data)

**Files to Download:**

- `frame0.pdb` (Topology)
- `frame0.dcd` (Standard DCD)
- `frame0.trr` (Standard TRR)
- `frame0.xtc` (Standard XTC)

**Setup Script:**

```python
# scripts/fetch_test_data.py
import urllib.request
import shutil
from pathlib import Path

DATA_DIR = Path("tests/data/trajectories")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/mdtraj/mdtraj/master/tests/data"

FILES = [
    "frame0.pdb",
    "frame0.dcd", 
    "frame0.trr",
    "frame0.xtc"
]

def fetch_data():
    for filename in FILES:
        target = DATA_DIR / filename
        if not target.exists():
            print(f"Downloading {filename}...")
            url = f"{BASE_URL}/{filename}"
            # Use a proper user agent to avoid 403s
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            try:
                with urllib.request.urlopen(req) as response, open(target, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                print(f"Downloaded {target}")
            except urllib.error.HTTPError as e:
                print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    fetch_data()
```

### Parity Test Structure

```python
# tests/validation/test_trajectory_parity.py

@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj required")
def test_dcd_frame_coordinates_vs_mdtraj():
    """Compare DCD frame coordinates against MDTraj."""
    from proxide import _oxidize
    
    dcd_file = Path("tests/data/trajectories/test.dcd")
    pdb_file = Path("tests/data/1crn.pdb")
    
    if not dcd_file.exists():
        pytest.skip("Test DCD file not available")
    
    # Parse with Rust
    result = _oxidize.parse_dcd(str(dcd_file))
    rust_coords = result["coordinates"]  # Angstroms
    
    # Parse with MDTraj
    traj = mdtraj.load(str(dcd_file), top=str(pdb_file))
    mdtraj_coords = traj.xyz * 10.0  # nm -> Angstroms
    
    # Compare
    assert rust_coords.shape == mdtraj_coords.shape
    np.testing.assert_allclose(rust_coords, mdtraj_coords, atol=1e-3)


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj required")
def test_trr_frame_coordinates_vs_mdtraj():
    """Compare TRR frame coordinates against MDTraj."""
    # Similar structure to DCD test
    ...


@pytest.mark.skipif(not MDTRAJ_AVAILABLE, reason="MDTraj required") 
def test_trr_velocities_vs_mdtraj():
    """Verify TRR velocities match MDTraj."""
    # TRR-specific: also check velocities if present
    ...
```

### Test Cases

| Test | Format | Validates |
|------|--------|-----------|
| `test_dcd_header_parsing` | DCD | Header metadata (n_atoms, n_frames, timestep) |
| `test_dcd_frame_coordinates_vs_mdtraj` | DCD | Coordinate parity with MDTraj (1e-3 Å) |
| `test_dcd_unit_cell_vs_mdtraj` | DCD | Box vector parity |
| `test_dcd_endianness_detection` | DCD | Auto-detect little/big endian |
| `test_trr_header_parsing` | TRR | Header metadata (n_atoms, has_velocities, etc.) |
| `test_trr_frame_coordinates_vs_mdtraj` | TRR | Coordinate parity (1e-3 Å) |
| `test_trr_velocities_vs_mdtraj` | TRR | Velocity parity if present |
| `test_trr_forces_vs_mdtraj` | TRR | Force parity if present |

---

## CI Integration

### GitHub Actions Workflow

Update `.github/workflows/trajectory_parity.yml`:

```yaml
name: Trajectory Parity Tests

on: [push, pull_request]

jobs:
  trajectory-parity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install uv
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "$HOME/.local/bin" >> $GITHUB_PATH
          
      - name: Install dependencies
        run: |
          uv pip install --system mdtraj numpy pytest maturin
          
      - name: Build Rust extension
        run: |
          cd oxidize
          # Build with new features enabled (when they exist)
          maturin develop --release --features dcd-pure,trr-pure
          
      - name: Fetch test data
        run: |
          uv run python scripts/fetch_test_data.py
          
      - name: Run parity tests
        run: |
          uv run pytest tests/validation/test_trajectory_parity.py -v
```

### Feature Flags

Add new Cargo features for the pure-Rust parsers:

```toml
# oxidize/Cargo.toml
[features]
default = []
dcd-pure = []           # Pure-Rust DCD parser
trr-pure = []           # Pure-Rust TRR parser  
xtc-pure = ["molly"]    # Existing pure-Rust XTC
full-pure = ["xtc-pure", "dcd-pure", "trr-pure"]
```

---

## Implementation Checklist

### Phase 1: DCD Parser

- [x] Research FORTRAN unformatted record structure
- [x] Implement endianness detection
- [x] Parse DCD header (signature, metadata, atom count)
- [x] Parse coordinate frames (X, Y, Z arrays)
- [x] Handle optional unit cell data
- [x] Add `parse_dcd()` PyO3 function
- [x] Write parity tests against MDTraj
- [x] Add `dcd-pure` feature flag

### Phase 2: TRR Parser

- [x] Implement XDR decoder utilities
- [x] Parse TRR frame headers
- [x] Parse coordinate data (convert nm → Å)
- [x] Parse optional velocities
- [x] Parse optional forces
- [x] Handle box matrix
- [x] Add `parse_trr()` PyO3 function
- [x] Write parity tests against MDTraj
- [x] Add `trr-pure` feature flag

### Phase 3: Integration

- [x] Update `tests/validation/test_trajectory_parity.py`
- [x] Add test data generation script
- [x] Update GitHub Actions workflow
- [ ] Update documentation
- [ ] Remove chemfiles dependency when complete

---

## Estimated Effort

| Phase | Complexity | Estimated Time |
|-------|------------|----------------|
| DCD Parser | Medium | 4-6 hours |
| TRR Parser | Medium-High | 6-8 hours |
| Tests & CI | Low | 2-3 hours |
| **Total** | | **12-17 hours** |

---

## References

- [MDAnalysis DCD Format](https://docs.mdanalysis.org/stable/documentation_pages/coordinates/DCD.html)
- [GROMACS File Formats](https://manual.gromacs.org/current/reference-manual/file-formats.html)
- [XDR Protocol Specification (RFC 4506)](https://datatracker.ietf.org/doc/html/rfc4506)
- [molly crate (XTC reference)](https://github.com/chemfiles/molly)
- [groan_rs crate (TRR reference)](https://github.com/Ladme/groan_rs)
