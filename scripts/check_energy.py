"""Quick diagnostic: decompose energy of 6XHB at initial crystal structure."""

import os

os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space
from prolix.batched_energy import (
    _angle_energy_masked,
    _bond_energy_masked,
    _build_dense_exclusion_scales,
    _cmap_energy_masked,
    _coulomb_energy_masked,
    _dihedral_energy_masked,
    _lj_energy_masked,
    single_padded_energy,
    single_padded_force,
)
from prolix.padding import ATOM_BUCKETS, pad_protein, select_bucket
from prolix.physics.generalized_born import (
    compute_ace_nonpolar_energy,
    compute_gb_energy,
)

from proxide import CoordFormat, OutputSpec
from proxide.io.parsing.backend import parse_structure

# Load and parameterize
spec = OutputSpec(
    coord_format=CoordFormat.Full,
    parameterize_md=True,
    force_field="proxide/src/proxide/assets/protein.ff19SB.xml",
    add_hydrogens=False,
    remove_solvent=True,
    remove_hetatm=True,
)
protein = parse_structure("references/pdb/6XHB_chainA_fixed.pdb", spec)

# Assign GB radii
if protein.radii is None:
    from proxide import assign_mbondi2_radii, assign_obc2_scaling_factors
    _radii = assign_mbondi2_radii(list(protein.atom_names), protein.bonds)
    _scaled = assign_obc2_scaling_factors(list(protein.atom_names))
    protein = dataclasses.replace(
        protein,
        radii=jnp.asarray(_radii),
        scaled_radii=jnp.asarray(_scaled),
    )

n_atoms = np.asarray(protein.coordinates).reshape(-1, 3).shape[0]
bucket = select_bucket(n_atoms, ATOM_BUCKETS)
print(f"Real atoms: {n_atoms}, bucket: {bucket}")

# Pad
padded = pad_protein(protein, bucket)
print(f"Padded size: {padded.positions.shape[0]}")

displacement_fn, _ = space.free()

# Total energy
e_total = single_padded_energy(padded, displacement_fn, soft_core_lambda=jnp.float32(1.0))
print(f"\nTotal energy: {float(e_total):.2f} kcal/mol")

# Decompose
r = padded.positions
N = len(padded.atom_mask)

e_bond = _bond_energy_masked(r, padded.bonds, padded.bond_params, padded.bond_mask, displacement_fn)
e_angle = _angle_energy_masked(r, padded.angles, padded.angle_params, padded.angle_mask, displacement_fn)
e_dih = _dihedral_energy_masked(r, padded.dihedrals, padded.dihedral_params, padded.dihedral_mask, displacement_fn)
e_imp = _dihedral_energy_masked(r, padded.impropers, padded.improper_params, padded.improper_mask, displacement_fn)
e_cmap = _cmap_energy_masked(r, padded.cmap_torsions, padded.cmap_mask, padded.cmap_coeffs, displacement_fn)

excl_vdw = jax.lax.stop_gradient(_build_dense_exclusion_scales(padded.excl_indices, padded.excl_scales_vdw, N))
excl_elec = jax.lax.stop_gradient(_build_dense_exclusion_scales(padded.excl_indices, padded.excl_scales_elec, N))

e_lj = _lj_energy_masked(r, padded.sigmas, padded.epsilons, padded.atom_mask, displacement_fn,
                          soft_core_lambda=jnp.float32(1.0), excl_scale_vdw=excl_vdw)
e_elec = _coulomb_energy_masked(r, padded.charges, padded.atom_mask, displacement_fn, excl_scale_elec=excl_elec)

# GB
mask_ij = padded.atom_mask[:, None] & padded.atom_mask[None, :]
energy_mask = mask_ij * (1.0 - jnp.eye(N))
e_gb, born_radii = compute_gb_energy(
    positions=r, charges=padded.charges, radii=padded.radii,
    scaled_radii=padded.scaled_radii, mask=padded.atom_mask,
    energy_mask=energy_mask, dielectric_offset=0.09,
)
e_np = jnp.sum(compute_ace_nonpolar_energy(padded.radii, born_radii) * padded.atom_mask)

print("\n=== Energy Decomposition ===")
print(f"  Bonds:      {float(e_bond):>15.2f}")
print(f"  Angles:     {float(e_angle):>15.2f}")
print(f"  Dihedrals:  {float(e_dih):>15.2f}")
print(f"  Impropers:  {float(e_imp):>15.2f}")
print(f"  CMAP:       {float(e_cmap):>15.2f}")
print(f"  LJ:         {float(e_lj):>15.2f}")
print(f"  Coulomb:    {float(e_elec):>15.2f}")
print(f"  GB:         {float(e_gb):>15.2f}")
print(f"  ACE np:     {float(e_np):>15.2f}")
print("  --------------------")
total = float(e_bond + e_angle + e_dih + e_imp + e_cmap + e_lj + e_elec + e_gb + e_np)
print(f"  SUM:        {total:>15.2f}")
print(f"  single_padded_energy: {float(e_total):>15.2f}")

# Forces
forces = single_padded_force(padded, displacement_fn, soft_core_lambda=jnp.float32(1.0))
f_mag = jnp.sqrt(jnp.sum(forces**2, axis=-1))
rms = jnp.sqrt(jnp.sum(f_mag**2 * padded.atom_mask) / jnp.sum(padded.atom_mask))
max_f = jnp.max(f_mag * padded.atom_mask)

# Which atom has max force?
max_idx = int(jnp.argmax(f_mag * padded.atom_mask))
print("\n=== Force Analysis ===")
print(f"  RMS force:  {float(rms):.4f} kcal/mol/Å")
print(f"  Max force:  {float(max_f):.4f} kcal/mol/Å  (atom {max_idx})")

# Top 10 highest-force atoms
sorted_idx = jnp.argsort(-(f_mag * padded.atom_mask))[:10]
print("\n  Top 10 force atoms:")
for idx in sorted_idx:
    i = int(idx)
    print(f"    atom {i}: |F| = {float(f_mag[i]):.2f}")
