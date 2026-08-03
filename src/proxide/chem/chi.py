"""Public API for chi angle atom tables and symmetry classification."""



from proxide.chem import residues as rc

# ==============================================================================
# CHI_ANGLES_ATOMS: Public dict mirror of chi_angles_atoms (wraps residues.py)
# ==============================================================================
CHI_ANGLES_ATOMS: dict[str, list[list[str]]] = rc.chi_angles_atoms


# ==============================================================================
# CHI_SYMMETRY_CLASS: Per-(restype, chi-index) symmetry classification
# ==============================================================================
# Tags:
#   'none'              — No symmetry (most χ's)
#   'pi_periodic_180'   — χ ≡ χ±180° (Asp χ2, Glu χ3)
#   'aromatic_180'      — Ring 2-fold symmetry (Phe χ2, Tyr χ2)
#   'non_rotameric_sp2' — Continuous, not discretized (Asn χ2, Gln χ3, His χ2)

CHI_SYMMETRY_CLASS: dict[tuple[str, int], str] = {
    # ALA, GLY: no chi angles

    # ARG: chi1-4 all have no symmetry
    ("ARG", 0): "none",
    ("ARG", 1): "none",
    ("ARG", 2): "none",
    ("ARG", 3): "none",

    # ASN: chi1 no symmetry, chi2 is non-rotameric sp2 (amide)
    ("ASN", 0): "none",
    ("ASN", 1): "non_rotameric_sp2",

    # ASP: chi1 no symmetry, chi2 is pi_periodic (carboxyl, 180° rotation)
    ("ASP", 0): "none",
    ("ASP", 1): "pi_periodic_180",

    # CYS: only chi1, no symmetry
    ("CYS", 0): "none",

    # GLN: chi1 no symmetry, chi2 no symmetry, chi3 is non-rotameric sp2 (amide)
    ("GLN", 0): "none",
    ("GLN", 1): "none",
    ("GLN", 2): "non_rotameric_sp2",

    # GLU: chi1 no symmetry, chi2 no symmetry, chi3 is pi_periodic (carboxyl)
    ("GLU", 0): "none",
    ("GLU", 1): "none",
    ("GLU", 2): "pi_periodic_180",

    # HIS: chi1 no symmetry, chi2 is non-rotameric sp2 (imidazole)
    ("HIS", 0): "none",
    ("HIS", 1): "non_rotameric_sp2",

    # ILE: chi1 no symmetry, chi2 no symmetry
    ("ILE", 0): "none",
    ("ILE", 1): "none",

    # LEU: chi1 no symmetry, chi2 no symmetry
    ("LEU", 0): "none",
    ("LEU", 1): "none",

    # LYS: chi1-4 all have no symmetry
    ("LYS", 0): "none",
    ("LYS", 1): "none",
    ("LYS", 2): "none",
    ("LYS", 3): "none",

    # MET: chi1-3 all have no symmetry
    ("MET", 0): "none",
    ("MET", 1): "none",
    ("MET", 2): "none",

    # PHE: chi1 no symmetry, chi2 is aromatic 2-fold (benzene ring)
    ("PHE", 0): "none",
    ("PHE", 1): "aromatic_180",

    # PRO: chi1 no symmetry, chi2 no symmetry
    ("PRO", 0): "none",
    ("PRO", 1): "none",

    # SER: only chi1, no symmetry
    ("SER", 0): "none",

    # THR: only chi1, no symmetry
    ("THR", 0): "none",

    # TRP: chi1 no symmetry, chi2 no symmetry
    ("TRP", 0): "none",
    ("TRP", 1): "none",

    # TYR: chi1 no symmetry, chi2 is aromatic 2-fold (benzene ring)
    ("TYR", 0): "none",
    ("TYR", 1): "aromatic_180",

    # VAL: only chi1, no symmetry
    ("VAL", 0): "none",
}


# ==============================================================================
# Utility function to retrieve symmetry class
# ==============================================================================

def get_chi_symmetry_class(restype_3: str, chi_index: int) -> str:
    """
    Get the symmetry class for a given residue type and chi index.

    Args:
        restype_3: 3-letter residue name (e.g., "ASP", "PHE")
        chi_index: 0-based chi index (0=chi1, 1=chi2, etc.)

    Returns:
        Symmetry class string: 'none', 'pi_periodic_180', 'aromatic_180',
        'non_rotameric_sp2', or raises KeyError if not found
    """
    return CHI_SYMMETRY_CLASS[(restype_3, chi_index)]


# ==============================================================================
# Utility function to verify 4-distinct-atoms property
# ==============================================================================

def verify_four_distinct_atoms(
    chi_angles_atoms_dict: dict[str, list[list[str]]]
) -> tuple[bool, dict[str, list[tuple[int, int, str]]]]:
    """
    Verify that every chi angle has exactly 4 distinct atom names.

    Args:
        chi_angles_atoms_dict: Dictionary mapping restype_3 to list of chi atom quads

    Returns:
        (all_valid, violations_dict) where:
          - all_valid: True if all residue types pass the check
          - violations_dict: Maps residue names to list of (chi_idx, num_atoms, resname)
                            for violations (empty dict = all pass)
    """
    violations: dict[str, list[tuple[int, int, str]]] = {}

    for resname, chi_atoms_list in chi_angles_atoms_dict.items():
        for chi_idx, atom_quad in enumerate(chi_atoms_list):
            if len(atom_quad) != 4:
                if resname not in violations:
                    violations[resname] = []
                violations[resname].append((chi_idx, len(atom_quad), resname))

            # Also check that all 4 atoms are distinct
            if len(set(atom_quad)) != 4:
                if resname not in violations:
                    violations[resname] = []
                violations[resname].append((chi_idx, len(set(atom_quad)), resname))

    return len(violations) == 0, violations
