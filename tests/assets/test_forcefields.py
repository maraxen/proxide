"""Tests for force field loading and component access."""

import jax.numpy as jnp
import pytest

from proxide.physics.force_fields import (
    FullForceField,
    load_force_field,
    list_available_force_fields,
)
from proxide.physics.force_fields.components import (
    AtomTypeParams,
    BondPotentialParams,
    AnglePotentialParams,
    DihedralPotentialParams,
    CMAPParams,
    UreyBradleyParams,
    VirtualSiteParams,
    NonbondedGlobalParams,
)


def create_minimal_force_field() -> FullForceField:
    """Create a minimal FullForceField for testing."""
    atom_params = AtomTypeParams(
        charges=jnp.array([0.0, -0.5, 0.5], dtype=jnp.float32),
        sigmas=jnp.array([3.5, 3.0, 3.2], dtype=jnp.float32),
        epsilons=jnp.array([0.1, 0.15, 0.12], dtype=jnp.float32),
        radii=jnp.zeros(3, dtype=jnp.float32),
        scales=jnp.zeros(3, dtype=jnp.float32),
        atom_key_to_id={("ALA", "N"): 0, ("ALA", "CA"): 1, ("ALA", "C"): 2},
        id_to_atom_key=[("ALA", "N"), ("ALA", "CA"), ("ALA", "C")],
        atom_class_map={"ALA_N": "N", "ALA_CA": "CT1", "ALA_C": "C"},
        atom_type_map={},
    )

    return FullForceField(
        atom_params=atom_params,
        bond_params=BondPotentialParams(params=[]),
        angle_params=AnglePotentialParams(params=[]),
        dihedral_params=DihedralPotentialParams(propers=[], impropers=[]),
        cmap_params=CMAPParams(energy_grids=jnp.zeros((0, 24, 24)), torsions=[]),
        urey_bradley_params=UreyBradleyParams(params=[]),
        virtual_site_params=VirtualSiteParams(definitions={}),
        global_params=NonbondedGlobalParams(),
        source_files=["test.xml"],
        residue_templates={},
    )


class TestForceFieldComponents:
    """Tests for force field component structure."""

    def test_force_field_creation(self) -> None:
        """Test creating a FullForceField object with the new modular structure."""
        ff = create_minimal_force_field()

        assert len(ff.charges_by_id) == 3
        assert len(ff.atom_params.id_to_atom_key) == 3

    def test_force_field_get_charge(self) -> None:
        """Test getting charge for specific atom."""
        ff = create_minimal_force_field()

        charge = ff.get_charge("ALA", "CA")
        assert charge == -0.5

    def test_force_field_get_charge_unknown_atom(self) -> None:
        """Test that unknown atom returns zero charge."""
        ff = create_minimal_force_field()

        charge = ff.get_charge("GLY", "CA")  # Not in force field
        assert charge == 0.0

    def test_force_field_get_lj_params(self) -> None:
        """Test getting LJ parameters for specific atom."""
        ff = create_minimal_force_field()

        sigma, epsilon = ff.get_lj_params("ALA", "N")
        assert sigma == pytest.approx(3.5, rel=1e-5)
        assert epsilon == pytest.approx(0.1, rel=1e-5)

    def test_backward_compatibility_properties(self) -> None:
        """Test that backward-compatibility properties work."""
        ff = create_minimal_force_field()

        # These should delegate to atom_params
        assert len(ff.charges_by_id) == 3
        assert len(ff.sigmas_by_id) == 3
        assert len(ff.epsilons_by_id) == 3
        assert len(ff.radii_by_id) == 3
        assert len(ff.scales_by_id) == 3
        assert ff.atom_key_to_id == ff.atom_params.atom_key_to_id


class TestForceFieldLoading:
    """Tests for loading force fields from assets."""

    def test_list_available_force_fields(self) -> None:
        """Test that we can list available force fields."""
        available = list_available_force_fields()

        assert isinstance(available, list)
        assert len(available) >= 2  # ff14SB and ff19SB
        assert "protein.ff14SB" in available
        assert "protein.ff19SB" in available

    def test_load_ff14sb(self) -> None:
        """Test loading ff14SB force field."""
        ff = load_force_field("protein.ff14SB")

        # Should have loaded atom types
        assert len(ff.atom_params.id_to_atom_key) > 0

        # Should have bonds, angles, dihedrals
        assert len(ff.bonds) > 0
        assert len(ff.angles) > 0
        assert len(ff.propers) > 0

    def test_load_ff19sb(self) -> None:
        """Test loading ff19SB force field."""
        ff = load_force_field("protein.ff19SB")

        assert len(ff.atom_params.id_to_atom_key) > 0
        assert len(ff.bonds) > 0

    def test_load_with_extension(self) -> None:
        """Test loading with explicit .xml extension."""
        ff = load_force_field("protein.ff14SB.xml")
        assert len(ff.atom_params.id_to_atom_key) > 0

    def test_load_nonexistent_raises(self) -> None:
        """Test that loading nonexistent force field raises."""
        with pytest.raises(ValueError, match="not found"):
            load_force_field("nonexistent_forcefield")

    def test_proper_torsion_structure(self) -> None:
        """Test that proper torsions have the expected structure."""
        ff = load_force_field("protein.ff14SB")

        assert len(ff.propers) > 0
        proper = ff.propers[0]

        # Should have class atoms and terms
        assert "terms" in proper
        assert len(proper["terms"]) > 0

        term = proper["terms"][0]
        assert "periodicity" in term
        assert "phase" in term
        assert "k" in term
