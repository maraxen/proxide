GAFF and MD Parameterization
============================

Proxide supports assigning GAFF (General Amber Force Field) atom types and full Molecular Dynamics (MD) parameterization via its Rust backend.

This functionality is accessible through the :class:`~proxide.io.parsing.rust.OutputSpec` class.

Assigning GAFF Atom Types
-------------------------

To assign GAFF atom types to a small molecule or protein, you need to:

1. Enable **infer_bonds** (required for ring perception and aromaticity detection).
2. Set **force_field** to ``"gaff"``.

.. code-block:: python

    from proxide.io.parsing import rust

    # Create configuration
    spec = rust.OutputSpec()
    spec.force_field = "gaff"
    spec.infer_bonds = True
    spec.include_hetatm = True  # If loading a ligand
    
    # helper for debugging
    spec.error_mode = rust.ErrorMode.Warn

    # Load structure
    system = rust.parse_structure("ligand.pdb", spec=spec)

    # Access atom types
    print("Atom Types:", system.atom_types)
    # ['ca', 'ca', 'ca', ... 'ha', 'ha']

    # Access topology
    print("Bonds shape:", system.bonds.shape)
    print("Proper Dihedrals:", system.proper_dihedrals.shape)

Full MD Parameterization
------------------------

You can also parameterize a structure using an OpenMM-style XML force field file. This will assign partial charges, VdW parameters (sigma/epsilon), and GBSA radii/scales.

.. code-block:: python

    from proxide.io.parsing import rust

    spec = rust.OutputSpec()
    spec.parameterize_md = True
    spec.force_field = "amber14-all.xml"  # Path to XML file
    spec.add_hydrogens = True  # Often required for parameterization

    protein = rust.parse_structure("1crn.pdb", spec=spec)

    # Access MD parameters
    print("Charges:", protein.charges)
    print("Sigmas:", protein.sigmas)
    print("Epsilons:", protein.epsilons)

Data Structure Integration
--------------------------

The returned :class:`~proxide.core.containers.Protein` object (which inherits from :class:`~proxide.core.atomic_system.AtomicSystem`) is populated with:

- ``atom_types``: Sequence of string atom types (GAFF or Force Field types).
- ``proper_dihedrals``: (N, 4) array of torsion indices.
- ``impropers``: (N, 4) array of improper torsion indices.
- ``charges``, ``sigmas``, ``epsilons``: Arrays of physical parameters (if parameterize_md=True).

Topology Graph
^^^^^^^^^^^^^^

You can access the sparse adjacency matrix for graph neural networks:

.. code-block:: python

    adj_matrix = protein.topology
    # returns scipy.sparse.csr_matrix
