"""Water model definitions and parameterization logic (Explicit Solvent)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

# Conversion factors
DEG2RAD = math.pi / 180.0

@dataclass
class WaterModelParams:
    """Parameters for a specific water model."""
    name: str
    atoms: List[str]
    charges: Dict[str, float]
    sigmas: Dict[str, float]  # Angstroms
    epsilons: Dict[str, float] # kcal/mol
    bonds: List[Tuple[str, str, float, float]] # (atom1, atom2, length, k_kcal_mol_A2)
    angles: List[Tuple[str, str, str, float, float]] # (a1, a2, a3, theta_rad, k_kcal_mol_rad2)
    # Constraints for rigid water (SETTLE/RATTLE)
    # Usually we constrain OH bonds and HH distance (or HOH angle)
    # Stored as (atom1, atom2, length)
    constraints: List[Tuple[str, str, float]] 
    
    # Virtual Sites (for 4/5-site models)
    # List of dicts matching structure in ForceField.virtual_sites
    # {"type": "Twocenter", "siteName": "M", "atoms": ["O", "H1", "H2"], ... parameters ...}
    virtual_sites: Optional[List[dict]] = None

# -------------------------------------------------------------------------
# TIP3P (Standard)
# -------------------------------------------------------------------------
# r(OH) = 0.9572 A, theta(HOH) = 104.52 deg
# HH dist = 2 * 0.9572 * sin(104.52/2) = 1.51390065
TIP3P_DEFS = WaterModelParams(
    name="TIP3P",
    atoms=["O", "H1", "H2"],
    charges={"O": -0.834, "H1": 0.417, "H2": 0.417},
    sigmas={"O": 3.15061, "H1": 0.0001, "H2": 0.0001},
    epsilons={"O": 0.1521, "H1": 0.0, "H2": 0.0},
    bonds=[
        ("O", "H1", 0.9572, 450.0),
        ("O", "H2", 0.9572, 450.0)
    ],
    angles=[
        ("H1", "O", "H2", 104.52 * DEG2RAD, 100.0)
    ],
    constraints=[
        ("O", "H1", 0.9572),
        ("O", "H2", 0.9572),
        ("H1", "H2", 1.51390065)
    ],
    virtual_sites=None
)

# -------------------------------------------------------------------------
# SPC/E (Extended Simple Point Charge)
# -------------------------------------------------------------------------
# r(OH) = 1.0 A, theta(HOH) = 109.47 deg
# HH dist = 2 * 1.0 * sin(109.47/2) = 1.63298
SPCE_DEFS = WaterModelParams(
    name="SPCE",
    atoms=["O", "H1", "H2"],
    charges={"O": -0.8476, "H1": 0.4238, "H2": 0.4238},
    sigmas={"O": 3.166, "H1": 0.0001, "H2": 0.0001},
    epsilons={"O": 0.1553, "H1": 0.0, "H2": 0.0},
    bonds=[
        ("O", "H1", 1.0, 450.0),
        ("O", "H2", 1.0, 450.0)
    ],
    angles=[
        ("H1", "O", "H2", 109.47 * DEG2RAD, 100.0)
    ],
    constraints=[
        ("O", "H1", 1.0),
        ("O", "H2", 1.0),
        ("H1", "H2", 1.63298086)
    ],
    virtual_sites=None
)

# -------------------------------------------------------------------------
# TIP4P-Ew (Ewald optimized)
# -------------------------------------------------------------------------
# 4-site model with virtual site M
# r(OH) = 0.9572, theta = 104.52
# M is on bisector of HOH, distance d_OM = 0.125 A from O towards H's?
# Charges: H = 0.52422, M = -1.04844, O = 0.0
# Warning: O has charge 0, but M carries the negative charge.
# Sigma/Eps on O.
TIP4PEW_DEFS = WaterModelParams(
    name="TIP4PEW",
    atoms=["O", "H1", "H2", "M"], # M is virtual
    charges={"O": 0.0, "H1": 0.52422, "H2": 0.52422, "M": -1.04844},
    sigmas={"O": 3.16435, "H1": 0.0001, "H2": 0.0001, "M": 0.0001},
    epsilons={"O": 0.16275, "H1": 0.0, "H2": 0.0, "M": 0.0},
    bonds=[
        ("O", "H1", 0.9572, 450.0),
        ("O", "H2", 0.9572, 450.0)
    ],
    angles=[
        ("H1", "O", "H2", 104.52 * DEG2RAD, 100.0)
    ],
    constraints=[
        ("O", "H1", 0.9572),
        ("O", "H2", 0.9572),
        ("H1", "H2", 1.51390065)
    ],
    virtual_sites=[
        # Type depends on implementation.
        # "ThreeAtomAverage" / "LocalCoordinates" / "TiP4P" specific
        # Prolix system.py has: 'siteName', 'atoms', parameters p, wo, wx, wy...
        # We need to map to what system.py expects from FF.
        # For now, placeholder. TIP4P usually handled by specialized VS logic.
    ]
)


def get_water_params(model_name: str = "TIP3P", rigid: bool = True) -> WaterModelParams:
    """Get parameters for specified water model name (case-insensitive).
    
    Args:
        model_name: Water model name (TIP3P, SPCE, TIP4PEW, etc.)
        rigid: If True, sets bond and angle force constants to 0.0, matching
               OpenMM's rigidWater=True behavior. Constraints still define
               geometry for SHAKE/SETTLE. Default True for standard MD.
    
    Returns:
        WaterModelParams with appropriate force constants.
    """
    name = model_name.upper()
    if name == "TIP3P":
        base = TIP3P_DEFS
    elif name == "SPCE" or name == "SPC/E":
        base = SPCE_DEFS
    elif name == "TIP4PEW" or name == "TIP4P-EW":
        base = TIP4PEW_DEFS
    elif name == "HOH" or name == "WAT" or name == "SOL":
        # Default fallback
        base = TIP3P_DEFS
    else:
        raise ValueError(f"Unknown water model: {model_name}")
    
    if rigid:
        # Zero out bond and angle force constants
        # Keep equilibrium values for constraint geometry
        rigid_bonds = [(a1, a2, length, 0.0) for a1, a2, length, _ in base.bonds]
        rigid_angles = [(a1, a2, a3, theta, 0.0) for a1, a2, a3, theta, _ in base.angles]
        
        return dataclasses.replace(
            base,
            bonds=rigid_bonds,
            angles=rigid_angles
        )
    
    return base
