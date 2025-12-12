
import jax.numpy as jnp
import pytest
import numpy as np

from priox.core.atomic_system import AtomicSystem

try:
    from openmm import NonbondedForce
    from openmm import unit as u
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False


@pytest.mark.skipif(not OPENMM_AVAILABLE, reason="OpenMM not installed")
def test_openmm_exclusions_and_scaling():
    """Test verification of 1-3 and 1-4 exclusions/scaling in OpenMM export."""
    
    # 4 atoms: 0-1-2-3
    n_atoms = 4
    # Charges: +1, -1, +1, -1
    charges = jnp.array([1.0, -1.0, 1.0, -1.0])
    # Sigmas (Å): 2.0, 2.0, 2.0, 2.0
    sigmas = jnp.array([2.0, 2.0, 2.0, 2.0])
    # Epsilons (kcal/mol): 0.1, 0.1, 0.1, 0.1
    epsilons = jnp.array([0.1, 0.1, 0.1, 0.1])
    
    # Coordinates (dummy)
    coords = jnp.zeros((n_atoms, 3))
    mask = jnp.ones(n_atoms)
    
    # Topology
    bonds = jnp.array([[0, 1], [1, 2], [2, 3]]) # 1-2 pairs
    angles = jnp.array([[0, 1, 2], [1, 2, 3]])  # 1-3 pairs: (0,2), (1,3)
    dihedrals = jnp.array([[0, 1, 2, 3]])       # 1-4 pairs: (0,3)
    
    system = AtomicSystem(
        coordinates=coords,
        atom_mask=mask,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        bonds=bonds,
        angles=angles,
        proper_dihedrals=dihedrals,
    )
    
    # Use standard AMBER scaling
    c_scale = 0.8333
    lj_scale = 0.5
    
    omm_system = system.to_openmm_system(
        coulomb14scale=c_scale,
        lj14scale=lj_scale
    )
    
    # Get NonbondedForce
    nonbonded = None
    for force in omm_system.getForces():
        if isinstance(force, NonbondedForce):
            nonbonded = force
            break
            
    assert nonbonded is not None
    
    # Check number of exceptions
    # 1-2 pairs: (0,1), (1,2), (2,3) -> 3
    # 1-3 pairs: (0,2), (1,3) -> 2
    # 1-4 pairs: (0,3) -> 1
    # Total = 6
    num_expect = nonbonded.getNumExceptions()
    assert num_expect == 6
    
    exceptions = {}
    for i in range(num_expect):
        p1, p2, q, sig, eps = nonbonded.getExceptionParameters(i)
        pair = tuple(sorted((p1, p2)))
        exceptions[pair] = (q, sig, eps)
        
    # Check 1-2 (Exclusions)
    for pair in [(0,1), (1,2), (2,3)]:
        assert pair in exceptions
        q, sig, eps = exceptions[pair]
        assert q._value == 0.0
        assert eps._value == 0.0
        
    # Check 1-3 (Exclusions)
    for pair in [(0,2), (1,3)]:
        assert pair in exceptions
        q, sig, eps = exceptions[pair]
        assert q._value == 0.0
        assert eps._value == 0.0
        
    # Check 1-4 (Scaled)
    pair = (0, 3)
    assert pair in exceptions
    q, sig, eps = exceptions[pair]
    
    # Expected values
    # q1=1.0, q2=-1.0 -> prod = -1.0 * 0.8333 = -0.8333
    expected_q = 1.0 * -1.0 * c_scale
    
    # sig1=2.0A=0.2nm. sig_mix = (0.2+0.2)/2 = 0.2
    expected_sig = 0.2
    
    # eps1=0.1 kcal/mol. eps_mix = sqrt(0.1*0.1) * 0.5 = 0.05 kcal/mol
    # Convert to kJ/mol: 0.05 * 4.184 = 0.2092
    expected_eps = 0.1 * lj_scale * 4.184
    
    assert pytest.approx(q._value, rel=1e-4) == expected_q
    assert pytest.approx(sig._value, rel=1e-4) == expected_sig
    assert pytest.approx(eps._value, rel=1e-4) == expected_eps

