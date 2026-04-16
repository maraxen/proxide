"""Espaloma partial charges (optional dependency)."""

from pathlib import Path

import numpy as np
import pytest

from proxide.chem.partial_charges import (  # noqa: E402
  CHARGE_SOURCE_ESPALOMA_AM1BCC,
  assign_espaloma_charges_from_proxide_molecule,
  assign_espaloma_charges_rdkit,
)

_GOLDEN_DIR = Path(__file__).resolve().parent.parent / "data" / "espaloma_golden"

# Diversified Ground Truth Battery for Rust vs JAX Parity
ESPALOMA_TEST_SMILES = [
    "N#N",                   # Simple diatomic
    "C",                     # Methane
    "CO",                    # Methanol
    "C1=CC=CC=C1",           # Benzene
    "CC(=O)Oc1ccccc1C(=O)O", # Aspirin
    "C[C@H](N)C(=O)O",       # Alanine (zwitterion is handled by RDKit)
    "CC(C)CC(C(=O)O)N",      # Leucine
    "c1ccccc1",              # Aromaticity check
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
    "CCCC",                  # Linear alkane
    "C1CCCCC1",              # Cyclohexane
    "C=C",                   # Double bond
    "C#C",                   # Triple bond
    "O=C1NC(=O)NC(=O)N1",    # Cyanuric acid
]


@pytest.mark.espaloma
@pytest.mark.parametrize("smiles", ESPALOMA_TEST_SMILES)
def test_espaloma_rust_vs_jax_equivalence_battery(smiles):
    """Rigorous parity battery: Rust Native vs JAX baseline."""
    pytest.importorskip("expaloma")
    pytest.importorskip("rdkit")
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    # Add hydrogens as per typical MM/ML featurization protocol
    mol = Chem.AddHs(mol)

    # 1. Compute via JAX backend
    q_jax = assign_espaloma_charges_rdkit(mol, backend="jax")

    # 2. Compute via Rust backend
    q_rust = assign_espaloma_charges_rdkit(mol, backend="rust")

    # 3. Assert parity within 1e-5 relative tolerance
    # XLA (JAX) vs nalgebra (Rust) accumulation drift is checked here.
    np.testing.assert_allclose(
        q_rust, 
        q_jax, 
        rtol=1e-5, 
        atol=1e-6,
        err_msg=f"Rust/JAX parity failed for SMILES: {smiles}"
    )


@pytest.mark.espaloma
def test_espaloma_nitrogen_diatomic():
  pytest.importorskip("expaloma")
  pytest.importorskip("rdkit")
  from rdkit import Chem

  mol = Chem.MolFromSmiles("N#N")
  assert Chem.GetFormalCharge(mol) == 0
  
  # Default backend (rust)
  q = assign_espaloma_charges_rdkit(mol)
  assert q.shape == (mol.GetNumAtoms(),)
  assert np.isfinite(q).all()
  assert np.isclose(q.sum(), 0.0, atol=1e-5)
  assert CHARGE_SOURCE_ESPALOMA_AM1BCC == "espaloma-am1bcc"


@pytest.mark.espaloma
@pytest.mark.parametrize("backend", ["jax", "rust"])
def test_espaloma_golden_n2_regression(backend):
  """Golden file matches RDKit path used in docs."""
  pytest.importorskip("expaloma")
  pytest.importorskip("rdkit")
  from rdkit import Chem

  mol = Chem.MolFromSmiles("N#N")
  q = assign_espaloma_charges_rdkit(mol, backend=backend)
  golden = np.load(_GOLDEN_DIR / "n2_charges.npy")
  np.testing.assert_allclose(q, golden, rtol=1e-5, atol=1e-6)


@pytest.mark.espaloma
@pytest.mark.parametrize("backend", ["jax", "rust"])
def test_atom_ordering_invariance_aspirin(backend):
  """Verify that charges are invariant to atom permutation in Molecule."""
  pytest.importorskip("expaloma")
  pytest.importorskip("rdkit")
  from proxide.io.parsing.molecule import Molecule

  aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
  mol_orig = Molecule.from_smiles(aspirin_smiles, name="aspirin")
  q_orig = assign_espaloma_charges_from_proxide_molecule(mol_orig, backend=backend)

  # Create a permuted version
  n = mol_orig.n_atoms
  perm = np.random.permutation(n)
  
  # Re-build molecule with permuted indices
  p_names = [mol_orig.atom_names[i] for i in perm]
  p_types = [mol_orig.atom_types[i] for i in perm]
  p_elements = [mol_orig.elements[i] for i in perm]
  p_pos = mol_orig.positions[perm]
  p_charges = mol_orig.charges[perm]

  old_to_new = {old: new for new, old in enumerate(perm)}
  p_bonds = [(old_to_new[b[0]], old_to_new[b[1]]) for b in mol_orig.bonds]

  mol_perm = Molecule(
    name="aspirin_permuted",
    atom_names=p_names,
    atom_types=p_types,
    elements=p_elements,
    positions=p_pos,
    charges=p_charges,
    bonds=p_bonds,
    bond_orders=mol_orig.bond_orders,
  )

  q_perm = assign_espaloma_charges_from_proxide_molecule(mol_perm, backend=backend)

  # q_perm[j] should match q_orig[perm[j]]
  np.testing.assert_allclose(q_perm, q_orig[perm], rtol=1e-5, atol=1e-6)


@pytest.mark.espaloma
def test_espaloma_golden_aspirin_regression():
  """Regression test for Aspirin charges against pre-computed golden data."""
  pytest.importorskip("expaloma")
  pytest.importorskip("rdkit")
  from proxide.io.parsing.molecule import Molecule

  aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
  mol = Molecule.from_smiles(aspirin_smiles, name="aspirin")
  q = assign_espaloma_charges_from_proxide_molecule(mol)

  golden_path = _GOLDEN_DIR / "aspirin_charges.npy"
  assert golden_path.exists(), f"Missing golden file: {golden_path}"

  golden = np.load(golden_path)
  np.testing.assert_allclose(q, golden, rtol=1e-5, atol=1e-6)
