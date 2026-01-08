"""Type definitions for proxide."""

from __future__ import annotations

import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

ArrayLike = Array | np.ndarray

# Scalar Types
Scalar = Int[ArrayLike, ""]
ScalarFloat = Float[ArrayLike, ""]

# Structural Types
Coordinates = Float[ArrayLike, "num_atoms 3"]
Velocities = Float[ArrayLike, "num_atoms 3"]
StructureAtomicCoordinates = Float[ArrayLike, "num_residues 37 3"]
ResidueAtomsCoordinates = Float[ArrayLike, "num_residues num_atoms 3"]
DistanceMatrix = Float[ArrayLike, "n_atoms n_atoms"]
BackboneCoordinates = Float[ArrayLike, "num_residues 5 3"]
BackboneNoise = Float[ArrayLike, ""]  # Scalar noise level
AtomicCoordinate = Float[ArrayLike, "3"]
AlphaCarbonDistance = Float[ArrayLike, "num_residues num_residues"]
AtomIndexPair = Int[ArrayLike, "2"]
ScoreMatrix = Float[ArrayLike, "N M"]
AtomsMask = Bool[ArrayLike, "num_atoms"]
ResidueAtomsMask = Bool[ArrayLike, "num_residues num_atoms"]
AlphaCarbonMask = Float[ArrayLike, "num_residues"]  # 1.0 or 0.0
ResidueMask = Float[ArrayLike, "num_residues"]  # 1.0 or 0.0
AtomMask = Bool[ArrayLike, "num_residues num_atoms"]
Elements = list[str]
AtomTypes = list[str]

# Protein Types
ProteinSequence = Int[ArrayLike, "num_residues"]
OneHotProteinSequence = Float[ArrayLike, "num_residues num_classes"]
Logits = Float[ArrayLike, "... n_classes"]
ResidueIndex = Int[ArrayLike, "num_residues"]
ChainIndex = Int[ArrayLike, "num_residues"]
PerAtomResidueIndex = Int[ArrayLike, "num_atoms"]
PerAtomChainIndex = Int[ArrayLike, "num_atoms"]
BackboneDihedrals = Float[ArrayLike, "num_residues 3"]
MoleculeType = Int[ArrayLike, "num_atoms"]

# Feature Types
NodeFeatures = Float[ArrayLike, "num_atoms num_features"]
EdgeFeatures = Float[ArrayLike, "num_atoms num_neighbors num_features"]
NodeEdgeFeatures = Float[ArrayLike, "num_atoms num_neighbors num_features"]
RBFFeatures = Float[ArrayLike, "num_residues num_neighbors num_features"]
PhysicsFeatures = Float[ArrayLike, "num_residues num_features"]
Parameters = Float[ArrayLike, "num_parameters"]
ModelParameters = PyTree[str, "P"]

# Physics Types
Charges = Float[ArrayLike, "num_atoms"]
Masses = Float[ArrayLike, "num_atoms"]
Sigmas = Float[ArrayLike, "num_atoms"]
Epsilons = Float[ArrayLike, "num_atoms"]
Radii = Float[ArrayLike, "num_atoms"]
Scales = Float[ArrayLike, "num_atoms"]
EnergyGrids = Float[ArrayLike, "n_maps grid_size grid_size"]
CmapGrid = Float[ArrayLike, "grid_size grid_size"]

# Topology Types
Bonds = Int[ArrayLike, "num_bonds 2"]
Angles = Int[ArrayLike, "num_angles 3"]
Dihedrals = Int[ArrayLike, "num_dihedrals 4"]
ProperDihedrals = Int[ArrayLike, "num_dihedrals 4"]
Impropers = Int[ArrayLike, "num_impropers 4"]
BondParams = Float[ArrayLike, "num_bonds 2"]
AngleParams = Float[ArrayLike, "num_angles 2"]
DihedralParams = Float[ArrayLike, "num_dihedrals 3"]
ImproperParams = Float[ArrayLike, "num_impropers 3"]
CmapCoeffs = Float[ArrayLike, "num_maps grid_size grid_size 4"]
BoxVectors = Float[ArrayLike, "3 3"]
ConstrainedBonds = Int[ArrayLike, "num_constraints 2"]
ConstrainedBondLengths = Float[ArrayLike, "num_constraints"]
VirtualSiteDef = Int[ArrayLike, "num_vs 4"]
VirtualSiteParams = Float[ArrayLike, "num_vs 12"]
NeighborIndices = Int[ArrayLike, "num_atoms num_neighbors"]
BackboneIndices = Int[ArrayLike, "num_residues 4"]
ExclusionMask = Bool[ArrayLike, "N N"]
ScaleMatrix = Float[ArrayLike, "N N"]
CmapIndices = Int[ArrayLike, "num_torsions"]
CmapTorsions = Int[ArrayLike, "num_torsions 5"]
UreyBradleyBonds = Int[ArrayLike, "num_ub 2"]
UreyBradleyParams = Float[ArrayLike, "num_ub 2"]

# Alignment Types
InterproteinMapping = Int[ArrayLike, "num_pairs max_length 2"]

# aliases
PRNGKey = PRNGKeyArray
