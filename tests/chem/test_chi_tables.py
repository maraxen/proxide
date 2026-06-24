"""Unit tests for chi angle atom tables and symmetry classification."""

import pytest

from proxide.chem import chi, residues as rc


# All 18 chi-bearing residue types (exclude ALA, GLY which have no chi)
CHI_BEARING_RESTYPES = [
    "ARG",  # 4 chi
    "ASN",  # 2 chi
    "ASP",  # 2 chi
    "CYS",  # 1 chi
    "GLN",  # 3 chi
    "GLU",  # 3 chi
    "HIS",  # 2 chi
    "ILE",  # 2 chi
    "LEU",  # 2 chi
    "LYS",  # 4 chi
    "MET",  # 3 chi
    "PHE",  # 2 chi
    "PRO",  # 2 chi
    "SER",  # 1 chi
    "THR",  # 1 chi
    "TRP",  # 2 chi
    "TYR",  # 2 chi
    "VAL",  # 1 chi
]


@pytest.mark.smoke
class TestChiAnglesAtomsPublicAPI:
    """Test the public CHI_ANGLES_ATOMS API."""

    def test_chi_angles_atoms_is_dict(self):
        """CHI_ANGLES_ATOMS should be a dict."""
        assert isinstance(chi.CHI_ANGLES_ATOMS, dict)

    def test_chi_angles_atoms_wraps_residues(self):
        """CHI_ANGLES_ATOMS should match chi_angles_atoms from residues."""
        assert chi.CHI_ANGLES_ATOMS is rc.chi_angles_atoms

    def test_chi_angles_atoms_contains_all_20_types(self):
        """CHI_ANGLES_ATOMS should contain all 20 standard amino acid types."""
        expected_types = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLE", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        }
        # Note: GLU is spelled "GLU" not "GLE" — fix expected
        expected_types = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        }
        actual_types = set(chi.CHI_ANGLES_ATOMS.keys())
        assert actual_types == expected_types


@pytest.mark.smoke
class TestFourDistinctAtomsProperty:
    """Test that every chi angle in every chi-bearing residue has 4 distinct atoms."""

    @pytest.mark.parametrize("restype_3", CHI_BEARING_RESTYPES)
    def test_chi_bearing_restype_has_chi_list(self, restype_3):
        """Every chi-bearing residue should have a non-empty chi list."""
        chi_list = chi.CHI_ANGLES_ATOMS.get(restype_3, [])
        assert len(chi_list) > 0, f"{restype_3} should have at least one chi angle"

    @pytest.mark.parametrize("restype_3", CHI_BEARING_RESTYPES)
    def test_every_chi_has_four_atoms(self, restype_3):
        """Every chi angle should define exactly 4 atoms."""
        chi_list = chi.CHI_ANGLES_ATOMS[restype_3]
        for chi_idx, atom_quad in enumerate(chi_list):
            assert len(atom_quad) == 4, (
                f"{restype_3} chi{chi_idx + 1}: expected 4 atoms, got {len(atom_quad)}: {atom_quad}"
            )

    @pytest.mark.parametrize("restype_3", CHI_BEARING_RESTYPES)
    def test_every_chi_has_four_distinct_atoms(self, restype_3):
        """Every chi angle should have 4 distinct atom names (no duplicates)."""
        chi_list = chi.CHI_ANGLES_ATOMS[restype_3]
        for chi_idx, atom_quad in enumerate(chi_list):
            unique_atoms = set(atom_quad)
            assert len(unique_atoms) == 4, (
                f"{restype_3} chi{chi_idx + 1}: atoms not distinct: {atom_quad}"
            )

    def test_all_18_chi_bearing_types_pass_4distinct_atoms(self):
        """All 18 chi-bearing residue types should pass the 4-distinct-atoms check."""
        all_valid, violations = chi.verify_four_distinct_atoms(chi.CHI_ANGLES_ATOMS)
        assert all_valid, f"4-distinct-atoms check failed for: {violations}"
        assert not violations


@pytest.mark.smoke
class TestSymmetryClassTags:
    """Test that symmetry class tags are present and correct."""

    def test_chi_symmetry_class_is_dict(self):
        """CHI_SYMMETRY_CLASS should be a dict."""
        assert isinstance(chi.CHI_SYMMETRY_CLASS, dict)

    def test_symmetry_class_has_valid_values(self):
        """All symmetry class values should be from the defined set."""
        valid_values = {"none", "pi_periodic_180", "aromatic_180", "non_rotameric_sp2"}
        for value in chi.CHI_SYMMETRY_CLASS.values():
            assert value in valid_values, f"Invalid symmetry class value: {value}"

    def test_asp_chi2_is_pi_periodic_180(self):
        """ASP chi2 should be tagged as pi_periodic_180."""
        sym_class = chi.CHI_SYMMETRY_CLASS.get(("ASP", 1))
        assert sym_class == "pi_periodic_180", f"ASP chi2 symmetry: {sym_class}"

    def test_glu_chi3_is_pi_periodic_180(self):
        """GLU chi3 should be tagged as pi_periodic_180."""
        sym_class = chi.CHI_SYMMETRY_CLASS.get(("GLU", 2))
        assert sym_class == "pi_periodic_180", f"GLU chi3 symmetry: {sym_class}"

    def test_phe_chi2_is_aromatic_180(self):
        """PHE chi2 should be tagged as aromatic_180."""
        sym_class = chi.CHI_SYMMETRY_CLASS.get(("PHE", 1))
        assert sym_class == "aromatic_180", f"PHE chi2 symmetry: {sym_class}"

    def test_tyr_chi2_is_aromatic_180(self):
        """TYR chi2 should be tagged as aromatic_180."""
        sym_class = chi.CHI_SYMMETRY_CLASS.get(("TYR", 1))
        assert sym_class == "aromatic_180", f"TYR chi2 symmetry: {sym_class}"

    def test_asn_chi2_is_non_rotameric_sp2(self):
        """ASN chi2 (amide) should be tagged as non_rotameric_sp2."""
        sym_class = chi.CHI_SYMMETRY_CLASS.get(("ASN", 1))
        assert sym_class == "non_rotameric_sp2", f"ASN chi2 symmetry: {sym_class}"

    def test_gln_chi3_is_non_rotameric_sp2(self):
        """GLN chi3 (amide) should be tagged as non_rotameric_sp2."""
        sym_class = chi.CHI_SYMMETRY_CLASS.get(("GLN", 2))
        assert sym_class == "non_rotameric_sp2", f"GLN chi3 symmetry: {sym_class}"

    def test_his_chi2_is_non_rotameric_sp2(self):
        """HIS chi2 (imidazole) should be tagged as non_rotameric_sp2."""
        sym_class = chi.CHI_SYMMETRY_CLASS.get(("HIS", 1))
        assert sym_class == "non_rotameric_sp2", f"HIS chi2 symmetry: {sym_class}"

    @pytest.mark.parametrize("restype_3", CHI_BEARING_RESTYPES)
    def test_symmetry_class_defined_for_all_chi_indices(self, restype_3):
        """Every chi angle should have a symmetry class entry."""
        chi_list = chi.CHI_ANGLES_ATOMS[restype_3]
        for chi_idx in range(len(chi_list)):
            key = (restype_3, chi_idx)
            assert key in chi.CHI_SYMMETRY_CLASS, (
                f"Missing symmetry class for {restype_3} chi{chi_idx + 1}"
            )


@pytest.mark.smoke
class TestDictMirrorAgreement:
    """Test that the dict mirror (CHI_ANGLES_ATOMS) agrees with chi_angles_atom_indices."""

    def test_chi_angles_atoms_dict_vs_indices_array(self):
        """CHI_ANGLES_ATOMS dict should match chi_angles_atom_indices array structure."""
        # Verify the dict names match what the indices array represents
        for restype_letter in rc.restypes:
            restype_3 = rc.restype_1to3[restype_letter]
            chi_list_dict = chi.CHI_ANGLES_ATOMS[restype_3]
            chi_indices_array = rc.chi_angles_atom_indices[rc.restype_order[restype_letter]]

            # Verify each chi's atom indices map back to names correctly
            # chi_indices_array is padded to 4 rows, so only check up to len(chi_list_dict)
            for chi_idx, atom_names in enumerate(chi_list_dict):
                atom_indices = chi_indices_array[chi_idx]
                for atom_idx, atom_name in enumerate(atom_names):
                    expected_idx = rc.atom_order[atom_name]
                    assert atom_indices[atom_idx] == expected_idx, (
                        f"{restype_3} chi{chi_idx + 1} atom{atom_idx}: "
                        f"name={atom_name} (idx={expected_idx}) != array idx {atom_indices[atom_idx]}"
                    )

    def test_ala_and_gly_have_no_chi(self):
        """ALA and GLY should have empty chi lists (no side chain)."""
        assert chi.CHI_ANGLES_ATOMS["ALA"] == []
        assert chi.CHI_ANGLES_ATOMS["GLY"] == []


@pytest.mark.smoke
class TestGetChiSymmetryClassFunction:
    """Test the get_chi_symmetry_class utility function."""

    def test_get_chi_symmetry_class_asp_chi2(self):
        """Retrieve symmetry class for ASP chi2."""
        result = chi.get_chi_symmetry_class("ASP", 1)
        assert result == "pi_periodic_180"

    def test_get_chi_symmetry_class_phe_chi2(self):
        """Retrieve symmetry class for PHE chi2."""
        result = chi.get_chi_symmetry_class("PHE", 1)
        assert result == "aromatic_180"

    def test_get_chi_symmetry_class_asp_chi1(self):
        """Retrieve symmetry class for ASP chi1 (no symmetry)."""
        result = chi.get_chi_symmetry_class("ASP", 0)
        assert result == "none"

    def test_get_chi_symmetry_class_invalid_restype_raises(self):
        """Requesting symmetry class for invalid residue should raise KeyError."""
        with pytest.raises(KeyError):
            chi.get_chi_symmetry_class("INVALID", 0)

    def test_get_chi_symmetry_class_invalid_chi_index_raises(self):
        """Requesting symmetry class for invalid chi index should raise KeyError."""
        with pytest.raises(KeyError):
            chi.get_chi_symmetry_class("ASP", 99)


@pytest.mark.smoke
class TestVerifyFourDistinctAtomsFunction:
    """Test the verify_four_distinct_atoms utility function."""

    def test_verify_valid_dict(self):
        """verify_four_distinct_atoms should return (True, {}) for valid dict."""
        all_valid, violations = chi.verify_four_distinct_atoms(chi.CHI_ANGLES_ATOMS)
        assert all_valid is True
        assert violations == {}

    def test_verify_detects_wrong_length(self):
        """verify_four_distinct_atoms should detect quads with wrong length."""
        bad_dict = {
            "TEST": [["N", "CA", "CB", "CG", "CD"]],  # 5 atoms instead of 4
        }
        all_valid, violations = chi.verify_four_distinct_atoms(bad_dict)
        assert all_valid is False
        assert "TEST" in violations

    def test_verify_detects_duplicate_atoms(self):
        """verify_four_distinct_atoms should detect non-distinct atoms."""
        bad_dict = {
            "TEST": [["N", "CA", "N", "CG"]],  # "N" repeated, only 3 distinct
        }
        all_valid, violations = chi.verify_four_distinct_atoms(bad_dict)
        assert all_valid is False
        assert "TEST" in violations

    def test_verify_empty_dict(self):
        """verify_four_distinct_atoms should handle empty dict."""
        all_valid, violations = chi.verify_four_distinct_atoms({})
        assert all_valid is True
        assert violations == {}
