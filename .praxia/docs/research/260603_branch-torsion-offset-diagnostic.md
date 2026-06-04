# Branch Torsion Offset Diagnostic for #869

**Date**: 2026-06-03  
**Task ID**: 260603_loadpb_frame_fix  
**Backlog**: #869

## Summary

Diagnostic script `scripts/analysis/diagnose_branch_torsion_offsets.py` measures branch atom torsion offsets in MASTER's `rotlib.bin` to isolate the root cause of 2-5° per-residue Kabsch rotation errors in 6 residues.

## Problem

Six residues fail frame verification when comparing proxide's rebuilt `.pb.zst` to MASTER `rotlib.bin`:
- **Failing**: GLU, PHE, ASP, LEU, VAL, MET (2-5° rotation per residue)
- **Passing**: ILE, THR, ASN, SER, CYS (<0.5° rotation)

The paradox: both groups use the same formulas in `template.rs`:
```rust
VAL CG2: torsion = chi1 + 120°  (same formula as ILE CG2, ILE PASSES)
ASP OD2: torsion = chi2 + 180°  (same formula as ASN ND2, ASN PASSES)
```

## Hypothesis

Branch atom torsion offsets or CB placement errors in the template calculations account for the observed rotation. If the measured branch dihedral in MASTER differs from the template constant (e.g., chi1 + 120°), the coordinate placement will be incorrect.

## Measurement Strategy

For each failing + passing residue, load all rotamers from `rotlib.bin` and measure:

### Part A: Chi Angle Consistency
- Compute chi dihedrals from MASTER coordinates using canonical backbone atoms
- Compare to stored chi values
- Delta should be ~0° if coordinates are self-consistent

### Part B: Branch Atom Torsion Offset
- For each residue's branch atom (CG2, OD2, CD2, OE2, etc.):
  - Compute actual dihedral from 4-atom definition
  - Subtract the reference chi value: `offset = branch_dihedral - chi[relative_chi_idx]`
  - Compare to template constant (120° or 180°)
  - Report `delta = measured_offset - expected_offset`

### Part C: Passing vs. Failing Pattern
- If **passing residues** show `|delta| < 2°` and **failing residues** show `|delta| > 2°`, the offset error is the root cause
- If all residues show similar delta patterns, error is elsewhere (bond angles, CB placement, rotamer-set rotation)

## Expected Outcomes

**PASS**: Failing residues show large offset errors; passing residues match template.  
**Marginal**: Offset exists but is 1-2° (smaller than full rotation error).  
**FAIL**: All residues show matched offsets → error is elsewhere.

## Reference Geometry

Branch atoms and dihedral definitions per residue (from `template.rs`):
```python
VAL CG2:   ["N", "CA", "CB", "CG2"],     chi_idx=0, offset=120°
LEU CD2:   ["CA", "CB", "CG", "CD2"],    chi_idx=1, offset=120°
ASP OD2:   ["CA", "CB", "CG", "OD2"],    chi_idx=1, offset=180°
GLU OE2:   ["CB", "CG", "CD", "OE2"],    chi_idx=2, offset=180°
PHE CD2:   ["CA", "CB", "CG", "CD2"],    chi_idx=1, offset=180°
```

Canonical backbone atoms (fixed frame):
- N = [-1.458, 0, 0]
- CA = [0, 0, 0]
- C = [0.551, 1.420, 0]

## Deliverables

1. **Script**: `/scripts/analysis/diagnose_branch_torsion_offsets.py`
   - Standalone, imports `parse_rotlib` from sibling script
   - CLI: `--rotlib-bin`, `--residues`, `--output`, `--log-level`, `--dry-run`
   - Default paths configured; JSON + stdout summary output

2. **Bathos Sidecar**: `/scripts/analysis/diagnose_branch_torsion_offsets.py.bth.toml`
   - Pre-registered hypothesis, outcomes, result schema
   - Provenance tied to MASTER rotlib.bin input and dihedral formula

## Next Steps

Run script with default arguments to measure all 10 residues across all rotamers in `rotlib.bin`. Results determine whether offset or bond/CB placement is the root cause.
