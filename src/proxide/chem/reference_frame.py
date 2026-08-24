"""Canonical ligand reference frame: topology-derived atom ordering plus
per-frame torsion/pucker/geometry extraction.

Wraps proxide's native `proxide-ligand-frame` Rust crate (built with
`--features ligand-frame`). This is proxide's ligand-analog of what
`ReplicaSet` loading does for proteins today (`replicas.py`, producing
`(angles, feature_mask, names)`): demistify's ligand pipeline calls
`build_ligand_reference_frame`, then flattens `LigandFrameCoordinates`
into the `(N_res, N_frames, D)` / `(N_res, D)` / `names` triple
`run_demistify_pipeline` already accepts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from proxide.io.parsing.molecule import Molecule


@dataclass
class LigandTopology:
  """Frame-invariant ligand topology: canonical atom order, gaff2 types,
  partial charges, aromaticity/rings, torsion/pucker definitions. Computed
  once per unique molecular graph + reference geometry."""

  ligand_id: str
  canonical_order: np.ndarray
  elements: list[str]
  atom_names: list[str]
  gaff2_types: list[str]
  formal_charges: np.ndarray
  partial_charges: np.ndarray
  aromaticity: np.ndarray
  ring_membership: list[list[int]]
  bonds: list[tuple[int, int, int, bool, bool]]
  torsion_definitions: np.ndarray
  pucker_definitions: list[tuple[list[int], int]]
  unrepresented_ring_dof: list[list[int]]


@dataclass
class LigandFrameCoordinates:
  """Per-frame extraction: positions, torsions, pucker phase, bond
  lengths/angles, frame validity."""

  positions: np.ndarray
  torsions: np.ndarray
  feature_mask: np.ndarray
  frame_validity: np.ndarray
  pucker_phase: np.ndarray
  bond_lengths: np.ndarray
  bond_angles: np.ndarray


def _assert_explicit_hydrogens(molecule: Molecule) -> None:
  """Reject a `molecule` that appears to be missing explicit hydrogens.

  `build_ligand_reference_frame` passes `molecule.elements`/`molecule.bonds`
  straight through to gaff2 typing, whose contract (`assign_gaff_atom_types`
  in `py_chemistry.rs`) requires H atoms to already be explicit and
  index-aligned with the caller's `trajectory_positions`. `Molecule._to_rdkit`
  sets `NoImplicit=True` on every atom (deliberately -- see that method's
  docstring), which means a silently H-deficient input sails through typing
  as a wrong, lower-valence atom type (e.g. a benzene ring carbon typed
  `c1`, sp1, instead of `ca`, aromatic) rather than failing loudly. Because
  `Chem.AddHs` is a no-op on a `NoImplicit=True` atom (it only fills the
  *implicit* valence gap, which is forced to zero), that check must run
  against a fresh graph built *without* `NoImplicit`.

  Raises:
      ValueError: if RDKit's standard (neutral) valence model expects more
          atoms, once hydrogens are added, than `molecule.elements` has --
          i.e. some heavy atom has an unfilled valence that a real
          hydrogen should occupy. A structure that is correctly charged
          with intentionally no H (e.g. a fully-deprotonated ion) would
          need a formal charge on the relevant atom to avoid a false
          positive here; `Molecule` does not currently track formal
          charges, so this check assumes neutral valence, consistent with
          the rest of this module (`formal_charges` is likewise derived
          downstream as all-zero unless RDKit's sanitizer perceives
          otherwise).
  """
  from rdkit import Chem

  check_mol = Chem.RWMol()
  for elem in molecule.elements:
    check_mol.AddAtom(Chem.Atom(elem))

  bond_type_map = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
  }
  aromatic_atoms: set[int] = set()
  for k, ((i, j), order) in enumerate(zip(molecule.bonds, molecule.bond_orders, strict=True)):
    is_aromatic = k < len(molecule.bond_aromatic) and molecule.bond_aromatic[k]
    if is_aromatic:
      bond_type = Chem.BondType.AROMATIC
      aromatic_atoms.add(i)
      aromatic_atoms.add(j)
    else:
      bond_type = bond_type_map.get(order, Chem.BondType.SINGLE)
    check_mol.AddBond(i, j, bond_type)
  for idx in aromatic_atoms:
    check_mol.GetAtomWithIdx(idx).SetIsAromatic(True)

  check_mol = check_mol.GetMol()
  Chem.SanitizeMol(check_mol)

  expected_n_atoms = Chem.AddHs(check_mol).GetNumAtoms()
  actual_n_atoms = len(molecule.elements)
  if expected_n_atoms != actual_n_atoms:
    raise ValueError(
      f"molecule '{molecule.name}' appears to be missing explicit hydrogens: "
      f"standard valence for its heavy-atom graph expects {expected_n_atoms} "
      f"total atoms once hydrogens are filled in, but `molecule.elements` "
      f"has {actual_n_atoms}. `build_ligand_reference_frame` requires the "
      "caller-supplied `molecule` to already have explicit H atoms whose "
      "count and order are consistent with `trajectory_positions` -- add "
      "explicit hydrogens on the caller's side (e.g. via `Chem.AddHs(mol)` "
      "with a 3D embedding) before constructing/parsing `molecule`, rather "
      "than passing a heavy-atom-only structure here."
    )


def _espaloma_graph_features(mol) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  """RDKit-based Espaloma featurization -- the same `expaloma.featurize`
  path `assign_espaloma_charges_rdkit` (`partial_charges.py`) already uses.
  No native-Rust featurizer exists; only the message-passing inference step
  is ported (`proxide_core::chem::inference`)."""
  from expaloma.featurize import from_rdkit_mol

  g = from_rdkit_mol(mol)
  h0 = np.ascontiguousarray(g.h0, dtype=np.float32)
  senders = np.ascontiguousarray(g.senders, dtype=np.uint32)
  receivers = np.ascontiguousarray(g.receivers, dtype=np.uint32)
  total_charge = float(np.ascontiguousarray(g.q_ref, dtype=np.float32).sum())
  return h0, senders, receivers, total_charge


def build_ligand_reference_frame(
  molecule: Molecule,
  ligand_id: str,
  trajectory_positions: np.ndarray,
  ref_frame_index: int = 0,
) -> tuple[LigandTopology, LigandFrameCoordinates]:
  """Build the canonical topology and per-frame coordinates for a ligand.

  Args:
      molecule: Parsed ligand (e.g. ``Molecule.from_mol2``), input atom order.
      ligand_id: Caller-supplied identity, used in downstream torsion names
          (``f"{ligand_id}:torsion_{i}"``).
      trajectory_positions: ``(n_frames, n_atoms, 3)``, input atom order
          (same order as ``molecule.elements``/``molecule.bonds``).
      ref_frame_index: Which frame's coordinates seed Espaloma charge
          inference (frame 0 by default).
  """
  from proxide._proxider import canonicalize_ligand_topology as _canonicalize
  from proxide._proxider import extract_ligand_frame_coordinates as _extract

  _assert_explicit_hydrogens(molecule)

  mol = molecule._to_rdkit()
  ring_info = mol.GetRingInfo()
  rings = [list(r) for r in ring_info.AtomRings()]
  bond_is_aromatic = [b.GetIsAromatic() for b in mol.GetBonds()]
  formal_charges = [a.GetFormalCharge() for a in mol.GetAtoms()]

  # NOTE: `canonicalize_ligand_topology`'s `ref_positions` argument goes
  # through the PyO3 `extract_coords` helper, which only accepts a list of
  # lists or a float32 Nx3 numpy array (shared contract with
  # `assign_gaff_atom_types`/`parameterize_molecule` -- see
  # `test_canonicalize_ligand_topology_rejects_disconnected_graph`'s
  # float32 ref_positions). This is intentionally float32, unlike
  # `trajectory_positions`/`extract_ligand_frame_coordinates`'s `positions`
  # argument below, which requires float64.
  ref_positions = np.asarray(trajectory_positions[ref_frame_index], dtype=np.float32)
  h0, senders, receivers, total_charge = _espaloma_graph_features(mol)

  bonds = [
    (i, j, order, bond_is_aromatic[k])
    for k, ((i, j), order) in enumerate(zip(molecule.bonds, molecule.bond_orders, strict=True))
  ]

  topology_handle = _canonicalize(
    ligand_id,
    molecule.elements,
    molecule.atom_names,
    bonds,
    bond_is_aromatic,
    rings,
    formal_charges,
    ref_positions,
    h0,
    senders,
    receivers,
    total_charge,
  )

  topology = LigandTopology(
    ligand_id=topology_handle.ligand_id,
    canonical_order=np.asarray(topology_handle.canonical_order),
    elements=topology_handle.elements,
    atom_names=topology_handle.atom_names,
    gaff2_types=topology_handle.gaff2_types,
    formal_charges=np.asarray(topology_handle.formal_charges, dtype=np.int8),
    partial_charges=np.asarray(topology_handle.partial_charges, dtype=np.float64),
    aromaticity=np.asarray(topology_handle.aromaticity, dtype=bool),
    ring_membership=topology_handle.ring_membership,
    bonds=topology_handle.bonds,
    torsion_definitions=np.asarray(
      topology_handle.torsion_definitions, dtype=np.int64
    ).reshape(-1, 4),
    pucker_definitions=topology_handle.pucker_definitions,
    unrepresented_ring_dof=topology_handle.unrepresented_ring_dof,
  )

  raw = _extract(
    topology_handle,
    np.asarray(trajectory_positions, dtype=np.float64),
    molecule.elements,
  )

  coordinates = LigandFrameCoordinates(
    positions=np.asarray(raw["positions"], dtype=np.float64),
    torsions=np.asarray(raw["torsions"], dtype=np.float64),
    feature_mask=np.asarray(raw["feature_mask"], dtype=bool),
    frame_validity=np.asarray(raw["frame_validity"], dtype=bool),
    pucker_phase=np.asarray(raw["pucker_phase"], dtype=np.float64),
    bond_lengths=np.asarray(raw["bond_lengths"], dtype=np.float64),
    bond_angles=np.asarray(raw["bond_angles"], dtype=np.float64),
  )

  return topology, coordinates
