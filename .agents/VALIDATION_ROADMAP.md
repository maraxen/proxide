# Validation Roadmap

**Purpose:** Verify parity between Rust implementations and reference implementations (original proxide Python, Biotite, OpenMM, MDTraj)

---

## 1. Structure Parsing Parity

### 1.1 PDB Parser

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Atom coordinates | `parse_pdb_file()` | Original proxide `parse_pdb()` | 1e-3 Å | ✅ |
| Residue grouping | `ProcessedStructure::from_raw()` | Original proxide processing | Exact | ✅ |
| Chain mapping | `chain_indices` | Original proxide chains | Exact | ✅ |
| HETATM classification | `molecule_type` | Original proxide HETATM | Exact | ✅ |

---

## 5.7 Hydrogen Addition Parity

> **Implementation Status:** ✅ COMPLETE & VALIDATED (2025-12)
>
> - Fixed fragment library loading.
> - Implemented energy relaxation using OpenMM.
> - Validated successful energy relaxation on 1uao.pdb.

| Test | Rust Function | Reference | Tolerance | Status |
|------|--------------|-----------|-----------|--------|
| Hydrogen count | `add_hydrogens()` | Reference | Exact | ✅ |
| Energy relaxation | `relax_hydrogens()` | OpenMM | 0.1 Å | ✅ |

---

## 8. Execution Status

### Phase 1: Core Parsing & Structure (✅ Completed)

### Phase 2: Hydrogen & Force Fields (✅ Completed)

### Phase 3: Extended Parity & Format Validation (✅ Completed)

### Phase 4: Optimization & Polish (Active)
