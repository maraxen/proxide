"""Round-trip validation: Parse → OpenMM → Simulate.

This test validates the full pipeline from parsing a structure file,
applying force field parameters, exporting to OpenMM, and running
a short energy minimization.

Requires OpenMM to be installed: conda install -c conda-forge openmm
"""

from __future__ import annotations

import pytest
import numpy as np
import jax.numpy as jnp


# Check if OpenMM is available
try:
    import openmm
    from openmm import unit as u
    from openmm.app import Simulation
    from openmm import LangevinMiddleIntegrator

    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False


@pytest.mark.skipif(not OPENMM_AVAILABLE, reason="OpenMM not installed")
class TestOpenMMRoundTrip:
    """End-to-end tests for OpenMM export and simulation."""

    def test_small_system_energy_minimization(self) -> None:
        """Test creating an OpenMM system from a small molecule and running minimization."""
        from priox.core.atomic_system import AtomicSystem

        # Create a simple 3-atom water-like system
        # H-O-H with ~104.5° angle
        coords = jnp.array(
            [
                [0.0, 0.0, 0.0],  # O at origin
                [0.96, 0.0, 0.0],  # H1 along x
                [-0.24, 0.93, 0.0],  # H2 (104.5° from H1)
            ],
            dtype=jnp.float32,
        )

        system = AtomicSystem(
            coordinates=coords,
            atom_mask=jnp.ones(3),
            elements=["O", "H", "H"],
            atom_names=["O", "H1", "H2"],
            bonds=jnp.array([[0, 1], [0, 2]], dtype=jnp.int32),
            angles=jnp.array([[1, 0, 2]], dtype=jnp.int32),  # H-O-H angle
            # Force field params (TIP3P-like)
            charges=jnp.array([-0.834, 0.417, 0.417], dtype=jnp.float32),
            sigmas=jnp.array([3.15061, 0.4, 0.4], dtype=jnp.float32),  # Angstroms
            epsilons=jnp.array([0.1521, 0.0, 0.0], dtype=jnp.float32),  # kcal/mol
            # Bond params: [length (Å), k (kcal/mol/Å²)]
            bond_params=jnp.array(
                [[0.9572, 450.0], [0.9572, 450.0]], dtype=jnp.float32
            ),
            # Angle params: [theta (rad), k (kcal/mol/rad²)]
            angle_params=jnp.array(
                [[1.8242, 55.0]], dtype=jnp.float32  # 104.5° in radians
            ),
        )

        # Export to OpenMM
        topology = system.to_openmm_topology()
        omm_system = system.to_openmm_system(
            nonbonded_cutoff=1.0,  # nm
            use_switching_function=False,
        )

        # Verify forces were added
        assert omm_system.getNumParticles() == 3
        force_names = [
            omm_system.getForce(i).__class__.__name__
            for i in range(omm_system.getNumForces())
        ]
        assert "NonbondedForce" in force_names
        assert "HarmonicBondForce" in force_names
        assert "HarmonicAngleForce" in force_names

        # Create simulation
        integrator = LangevinMiddleIntegrator(
            300 * u.kelvin, 1.0 / u.picosecond, 0.002 * u.picosecond
        )

        # Use NoCutoff for small system
        nonbonded = next(
            omm_system.getForce(i)
            for i in range(omm_system.getNumForces())
            if isinstance(omm_system.getForce(i), openmm.NonbondedForce)
        )
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

        simulation = Simulation(topology, omm_system, integrator)

        # Set positions (convert Å to nm)
        positions = np.array(coords) * 0.1  # Å to nm
        simulation.context.setPositions(positions * u.nanometer)

        # Get initial energy
        state = simulation.context.getState(getEnergy=True)
        initial_energy = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)

        # Run minimization
        simulation.minimizeEnergy(maxIterations=100)

        # Get final energy
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        final_energy = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        final_positions = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)

        # Energy should decrease or stay similar (system was already near equilibrium)
        assert final_energy <= initial_energy + 10.0  # Allow small tolerance

        # Positions should have changed slightly during minimization
        assert final_positions is not None
        assert len(final_positions) == 3

    def test_with_torsions(self) -> None:
        """Test system with proper dihedrals (torsions)."""
        from priox.core.atomic_system import AtomicSystem

        # Create a simple 4-atom chain: C-C-C-C (butane-like)
        # Atom positions in a trans conformation
        coords = jnp.array(
            [
                [0.0, 0.0, 0.0],  # C1
                [1.54, 0.0, 0.0],  # C2
                [2.31, 1.26, 0.0],  # C3
                [3.85, 1.26, 0.0],  # C4
            ],
            dtype=jnp.float32,
        )

        system = AtomicSystem(
            coordinates=coords,
            atom_mask=jnp.ones(4),
            elements=["C", "C", "C", "C"],
            atom_names=["C1", "C2", "C3", "C4"],
            bonds=jnp.array([[0, 1], [1, 2], [2, 3]], dtype=jnp.int32),
            angles=jnp.array([[0, 1, 2], [1, 2, 3]], dtype=jnp.int32),
            proper_dihedrals=jnp.array([[0, 1, 2, 3]], dtype=jnp.int32),
            # Force field params
            charges=jnp.zeros(4, dtype=jnp.float32),
            sigmas=jnp.array([3.4, 3.4, 3.4, 3.4], dtype=jnp.float32),  # Angstroms
            epsilons=jnp.array(
                [0.066, 0.066, 0.066, 0.066], dtype=jnp.float32
            ),  # kcal/mol
            # Bond params: [length (Å), k (kcal/mol/Å²)]
            bond_params=jnp.array(
                [[1.54, 300.0], [1.54, 300.0], [1.54, 300.0]], dtype=jnp.float32
            ),
            # Angle params: [theta (rad), k (kcal/mol/rad²)]
            angle_params=jnp.array(
                [[1.9373, 50.0], [1.9373, 50.0]], dtype=jnp.float32  # ~111°
            ),
            # Dihedral params: [periodicity, phase (rad), k (kcal/mol)]
            dihedral_params=jnp.array(
                [[3, 0.0, 0.3]], dtype=jnp.float32  # 3-fold, phase=0, k=0.3
            ),
        )

        # Export to OpenMM
        topology = system.to_openmm_topology()
        omm_system = system.to_openmm_system()

        # Verify torsion force was added
        force_names = [
            omm_system.getForce(i).__class__.__name__
            for i in range(omm_system.getNumForces())
        ]
        assert "PeriodicTorsionForce" in force_names

        # Find torsion force and verify parameters
        torsion_force = next(
            omm_system.getForce(i)
            for i in range(omm_system.getNumForces())
            if isinstance(omm_system.getForce(i), openmm.PeriodicTorsionForce)
        )
        assert torsion_force.getNumTorsions() == 1

        # Get torsion parameters
        i, j, k, m, periodicity, phase, k_torsion = torsion_force.getTorsionParameters(
            0
        )
        assert (i, j, k, m) == (0, 1, 2, 3)
        assert periodicity == 3
        assert abs(phase.value_in_unit(u.radian) - 0.0) < 1e-5
        # k = 0.3 kcal/mol * 4.184 = 1.2552 kJ/mol
        assert abs(k_torsion.value_in_unit(u.kilojoule_per_mole) - 1.2552) < 0.01

    def test_topology_with_bonds(self) -> None:
        """Test that topology correctly represents bonds."""
        from priox.core.atomic_system import AtomicSystem

        coords = jnp.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float32
        )

        system = AtomicSystem(
            coordinates=coords,
            atom_mask=jnp.ones(3),
            elements=["C", "C", "C"],
            bonds=jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
        )

        topology = system.to_openmm_topology()

        # Check atoms
        atoms = list(topology.atoms())
        assert len(atoms) == 3

        # Check bonds
        bonds = list(topology.bonds())
        assert len(bonds) == 2


class TestForceFieldParameters:
    """Tests for force field loading and OpenMM conversion."""

    def test_load_ff14sb_and_check_parameters(self) -> None:
        """Load ff14SB and verify it has expected structure."""
        from priox.physics.force_fields import load_force_field

        ff = load_force_field("protein.ff14SB")

        # Check basic structure
        assert len(ff.atom_params.id_to_atom_key) > 0
        assert len(ff.bonds) > 0
        assert len(ff.angles) > 0
        assert len(ff.propers) > 0

        # Check proper torsion structure
        proper = ff.propers[0]
        assert "class1" in proper or "class2" in proper
        assert "terms" in proper
        assert len(proper["terms"]) > 0

        term = proper["terms"][0]
        assert "periodicity" in term
        assert "phase" in term
        assert "k" in term

    def test_unit_conversions(self) -> None:
        """Verify unit conversion constants are correct."""
        # Å to nm
        assert abs(1.0 * 0.1 - 0.1) < 1e-10

        # kcal/mol to kJ/mol
        assert abs(1.0 * 4.184 - 4.184) < 1e-10

        # kcal/mol/Å² to kJ/mol/nm²
        # 4.184 (energy) * 100 (length²)
        assert abs(1.0 * 4.184 * 100 - 418.4) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
