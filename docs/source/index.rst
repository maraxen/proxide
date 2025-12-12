Priox Documentation
===================

.. image:: https://img.shields.io/badge/status-work--in--progress-yellow.svg
  :alt: Status

**Priox** is a specialized library for Protein I/O and Physics bridging in JAX. It provides efficient tools for loading, processing, and converting protein structure data in the JAX ecosystem, as well as bridging with MD engines like JAX MD.

.. note::
   This is a work-in-progress library and is not yet ready for production use. It is currently in active development and subject to change.

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  api/modules
  gaff_integration

Key Features
------------

* **JAX-Native I/O:** Load and process protein structures directly into JAX arrays.
* **MD Engine Bridging:** Seamless integration with JAX MD for physics-based simulations.
* **Efficient Parsing:** High-performance parsing of PDB, CIF, and other structure formats.
* **Type Safety:** Fully typed codebase for robust development.

Installation
------------

.. code-block:: bash

  pip install priox

For development:

.. code-block:: bash

  pip install -e ".[dev]"
