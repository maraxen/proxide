# Force Field XML Parser - Rust Integration Plan

**Purpose:** Parse OpenMM XML force field files directly into `FullForceField` dataclass structures using Rust for speed.

## Implementation Status (2024-12-12)

✅ **Implemented & Validated:**

- **XML Parsing:** Fully implemented in Rust using `roxmltree`.
- **Supported Formats:** OpenMM XML format (Amber, CHARMM, and others).
- **GAFF Integration:** Special handling for GAFF atom typing (`force_field="gaff"`).
- **Topology Generation:** Bond, angle, and dihedral lists generated from residue templates.
- **Exclusions:** 1-2, 1-3, and 1-4 exclusion lists generated compatible with OpenMM.
- **Python Integration:** Exposed via `oxidize.parse_structure` with `OutputSpec`.

## Data Flow

```
XML File / "gaff" string → Rust Parser (`load_forcefield`) → `ForceField` Struct → `ProcessedStructure` → `AtomicSystem` (Python)
```

## oxidize/src/forcefield/ Structure

```rust
pub struct ForceField {
    // Atom types
    pub atom_types: Vec<AtomType>,
    
    // Residue templates  
    pub residue_templates: Vec<ResidueTemplate>,
    
    // Parameter maps
    pub nonbonded_params: Vec<NonbondedParam>,
    pub bonds: Vec<BondParam>,
    pub angles: Vec<AngleParam>,
    pub dihedrals: Vec<PeriodicTorsionParam>,
    pub impropers: Vec<PeriodicTorsionParam>,
    
    // Helper function
    pub fn get_parameters(&self, atom_type: &str) -> Option<&NonbondedParam>
}
```

## Validated features

1. **Protein FF Parsing**: Validated with `protein.ff14SB.xml` on `1crn.pdb`, `5awl.pdb`, `1uao.pdb`.
2. **Hydrogen Addition**: Validated correct placement and N-H bond lengths (approx 1.01 Å).
3. **Energy Relaxation**: Validated minimization using OpenMM via `to_openmm_system()` export.
4. **GAFF Atom Typing**: Validated on Benzene, correctly assigning aromatic `ca` types.

## Next Steps

- [ ] Expand GAFF testing to more complex molecules.
- [ ] Optimize 1-4 pair generation for very large systems.
- [ ] Improve error messages for missing atom types in fragments.
