# Espaloma golden charge vectors

Reference arrays for regression tests (`pytest -m espaloma`).

- **`n2_charges.npy`**: partial charges for `Chem.MolFromSmiles("N#N")` (2 atoms), **no** explicit 3D embed step — matches the upstream espaloma-charge README example.
- **`aspirin_charges.npy`**: charges from :meth:`proxide.chem.partial_charges.assign_espaloma_charges_from_proxide_molecule` for aspirin built via :meth:`~proxide.io.parsing.molecule.Molecule.from_smiles` (explicit hydrogens + MMFF coordinates).

**Bump policy:** When the `expaloma` dependency or charge protocol changes, regenerate:

```bash
uv pip install -e ".[espaloma]"
python -c "
from pathlib import Path
import numpy as np
from rdkit import Chem
from expaloma.infer import charges_for_rdkit_mol

mol = Chem.MolFromSmiles('N#N')
Chem.SanitizeMol(mol)
q = charges_for_rdkit_mol(mol)
p = Path('tests/data/espaloma_golden/n2_charges.npy')
np.save(p, np.asarray(q, dtype=np.float64))
print('wrote', p)
"
```

For aspirin, run the same assignment path as the test (Molecule → `_to_rdkit` → sanitize → `charges_for_rdkit_mol` + AtomMapNum remap) or copy from a trusted `expaloma` regression run.
