//! Throwaway pyo3 validation entrypoint.
//!
//! **Not the Cutover integration.** `crates/proxide_py` (the real,
//! long-lived Python binding for this whole workspace) is untouched by this
//! module and by this port task -- wiring GAFF2 into `proxide_py` is a
//! separate, later Cutover step. This module exists solely so a local
//! validation script can build this crate's Rust GAFF2 engine as an
//! importable Python extension and diff its output, atom-by-atom, against
//! `proxide.chem.gaff2.assign_gaff2_atom_types` (the Python reference this
//! crate is a behavior-preserving port of).
//!
//! Feature-gated behind `python-validation` (off by default -- neither
//! `cargo build -p proxide-gaff2` nor `cargo test -p proxide-gaff2` compile
//! this module or pull in `pyo3` at all). Build and use it with:
//!
//! ```text
//! cargo build -p proxide-gaff2 --features python-validation
//! cp target/debug/libproxide_gaff2.so ./proxide_gaff2.so   # macOS: .dylib -> .so
//! python3 -c "
//! import proxide_gaff2
//! print(proxide_gaff2.assign_gaff2_atom_types_rs(['C', 'H', 'H', 'H', 'H'],
//!     [(0, 1, 1, False), (0, 2, 1, False), (0, 3, 1, False), (0, 4, 1, False)]))
//! "
//! ```
//!
//! `libproxide_gaff2.so` is this crate's own cdylib output (Cargo always
//! names a `cdylib` artifact `lib<crate_name>.so` on Linux / `.dylib` on
//! macOS / `<crate_name>.dll` on Windows, regardless of the `#[pymodule]`
//! function name below) -- copying/renaming it to `proxide_gaff2.so` is
//! what makes `import proxide_gaff2` resolve it, since CPython's import
//! machinery matches on filename, not on any embedded module metadata. No
//! `maturin`/`pip install` step is needed for this throwaway use.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::mol::{Bond, BondOrder, MolGraph};
use crate::orchestrate::assign_gaff2_atom_types;

/// `1`/`2`/`3` -> Kekule bond order, matching the magnitude convention a
/// validation script would already have on hand from RDKit (e.g.
/// `int(bond.GetBondTypeAsDouble())` after `Chem.Kekulize`, or a plain
/// `{SINGLE: 1, DOUBLE: 2, TRIPLE: 3}` map over `bond.GetBondType()`).
fn bond_order_from_u8(order: u8) -> Result<BondOrder, String> {
    match order {
        1 => Ok(BondOrder::Single),
        2 => Ok(BondOrder::Double),
        3 => Ok(BondOrder::Triple),
        other => Err(format!(
            "invalid bond order {other}: expected 1 (single), 2 (double), or 3 (triple) \
             -- pass the true Kekule bond identity, not an aromatic/1.5 order"
        )),
    }
}

/// Assign GAFF2 atom types to a molecule described in plain, RDKit-free
/// terms, for diffing this Rust engine against the Python reference from a
/// validation script.
///
/// # Python signature
/// `assign_gaff2_atom_types_rs(elements, bonds, formal_charges=None, rings=None) -> list[str]`
///
/// - `elements: list[str]` -- element symbols, index-aligned atom order.
///   H atoms must be explicit (same contract as the Python
///   `assign_gaff2_atom_types`: call `Chem.AddHs(mol)` before extracting
///   these from an RDKit molecule).
/// - `bonds: list[tuple[int, int, int, bool]]` -- `(atom_i, atom_j,
///   bond_order, is_aromatic)` per bond, already Kekulized. `bond_order` is
///   `1`/`2`/`3` (see [`bond_order_from_u8`]); `is_aromatic` should mirror
///   RDKit's `GetIsAromatic()` as preserved by
///   `Chem.Kekulize(mol, clearAromaticFlags=False)` -- see
///   `orchestrate.rs`'s module doc for why Kekulization itself must happen
///   on the Python/RDKit side before calling this function.
/// - `formal_charges: list[int] | None` -- one entry per atom; defaults to
///   all-zero when omitted. Not read by GAFF2 atom typing itself (kept for
///   `MolGraph` signature completeness / future callers).
/// - `rings: list[list[int]] | None` -- SSSR atom-index rings, e.g. from
///   RDKit's `mol.GetRingInfo().AtomRings()`. Omitting this on a molecule
///   that actually has rings will silently under-type every ring
///   atom/aromaticity-dependent rule (see `MolGraph::rings`'s doc) -- it is
///   optional only in the sense that a genuinely ring-free molecule doesn't
///   need it, not because it is safe to skip in general.
///
/// # Returns
/// `list[str]` -- one GAFF2 atom type per atom, in `elements` order
/// (including H atoms) -- the same index-aligned contract
/// `assign_gaff2_atom_types` documents on the Rust/Python side alike.
///
/// # Errors
/// Raises `ValueError` (not a panic) for an invalid bond order, a
/// structurally invalid `MolGraph` (out-of-range bond/ring atom indices, a
/// self-loop bond, a `formal_charges` length mismatch -- see
/// `MolGraph::new`), or a failure loading the bundled `ATOMTYPE_GFF2.DEF`.
#[pyfunction]
#[pyo3(signature = (elements, bonds, formal_charges=None, rings=None))]
fn assign_gaff2_atom_types_rs(
    elements: Vec<String>,
    bonds: Vec<(usize, usize, u8, bool)>,
    formal_charges: Option<Vec<i8>>,
    rings: Option<Vec<Vec<usize>>>,
) -> PyResult<Vec<String>> {
    let bonds: Vec<Bond> = bonds
        .into_iter()
        .map(|(i, j, order, aromatic)| {
            bond_order_from_u8(order).map(|order| Bond {
                i,
                j,
                order,
                aromatic,
            })
        })
        .collect::<Result<_, String>>()
        .map_err(PyValueError::new_err)?;

    let mol = MolGraph::new(elements, bonds, formal_charges, None, rings)
        .map_err(PyValueError::new_err)?;

    assign_gaff2_atom_types(&mol).map_err(PyValueError::new_err)
}

/// Python extension module -- `import proxide_gaff2` once built and copied
/// per this module's doc comment. Exposes exactly one function,
/// [`assign_gaff2_atom_types_rs`]; nothing else from this crate is bound.
#[pymodule]
fn proxide_gaff2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(assign_gaff2_atom_types_rs, m)?)?;
    Ok(())
}
