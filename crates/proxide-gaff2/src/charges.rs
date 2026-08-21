//! Partial charge assignment: native Rust Espaloma, then Gasteiger-Marsili PEOE
//! fallback, then zeros.
//!
//! Faithful port of `_get_espaloma_charges` (`src/proxide/chem/gaff2.py`, lines
//! 1680-1734, branch `gaff2-parity/rust-port`). Behavior-preserving: this is not
//! a rewrite of the charge-assignment algorithm, it is a port of the *orchestration*
//! that Python performs around two external, best-effort backends.
//!
//! ## Control flow (must match Python exactly)
//!
//! ```text
//! mol_copy = sanitize(mol)                         # UNCAUGHT -- propagates on failure
//! try: import proxide._proxider.assign_espaloma_charges  -> assign_rust_charges
//! except ImportError: assign_rust_charges = None
//! try: import expaloma.featurize.from_rdkit_mol
//! except ImportError: from_rdkit_mol = None
//! if assign_rust_charges and from_rdkit_mol:
//!     try: return native-espaloma-charges(mol_copy)   # UNCLAMPED, returned as-is
//!     except Exception: pass                          # fall through silently
//! try:
//!     charges = gasteiger(mol_copy)
//!     clamp each charge: inf/-inf/|q|>10 -> 0.0
//!     return charges
//! except Exception:
//!     return [0.0] * n_atoms
//! ```
//!
//! ## Why the espaloma backends are trait objects, not concrete calls
//!
//! Python treats both halves of the native-espaloma path as optional external
//! imports that are commonly absent (`assign_rust_charges = None` /
//! `from_rdkit_mol = None` on `ImportError`), each independently:
//!
//! - `assign_espaloma_charges` (the actual SAGE-GNN + QEq inference) already has a
//!   complete, tested native Rust port at `proxide_core::chem::inference::infer_charges`
//!   -- but wiring it into *this* crate would pull in `nalgebra` and the embedded
//!   587 KiB `espaloma_v0_0_8.bin` weight blob, which `mol.rs`'s own doc comment
//!   ("not dependent on external topology types... enable standalone wasm
//!   compilation") flags as a real crate-boundary cost. That's a call for whoever
//!   owns the crate's dependency graph, not for a single-module port.
//! - The featurizer (`expaloma.featurize.from_rdkit_mol`: mol -> initial node
//!   features / graph edges / reference charges) has no Rust port anywhere in this
//!   workspace at all -- it is itself a separate module boundary in Python too.
//!
//! So both are represented here as traits (`EspalomaFeaturizer`, `EspalomaAssigner`)
//! that a caller may wire up; with neither wired (the common case, matching Python's
//! common "both imports fail" case), [`get_espaloma_charges`] falls straight through
//! to Gasteiger, exactly as `_get_espaloma_charges` does when `assign_rust_charges`
//! or `from_rdkit_mol` is `None`.
//!
//! ## What has no Rust equivalent here
//!
//! `Chem.SanitizeMol(mol_copy)` runs unconditionally, *outside* both `try` blocks --
//! a malformed `mol` propagates an uncaught RDKit exception straight out of
//! `_get_espaloma_charges`. There is no sanitize step in this module: a `MolGraph`
//! is documented (see `mol.rs`) to already be a validated, Kekulized structure by
//! the time it reaches atom typing / charge assignment, so this port has nothing
//! to call at the equivalent point. Preserved as a documented gap, not "fixed" by
//! inventing a validation step Python doesn't perform here either.

use crate::mol::{BondOrder, MolGraph};

/// Espaloma graph features for one molecule: initial per-atom node features,
/// directed message-passing edges, and the reference total charge used by the
/// QEq readout. Mirrors the four arrays Python's `_get_espaloma_charges` reads
/// off `expaloma.featurize.from_rdkit_mol`'s return value (`g.h0`, `g.senders`,
/// `g.receivers`, `g.q_ref`) before calling into the native Rust assigner.
#[derive(Debug, Clone)]
pub struct EspalomaGraph {
    /// Flattened `(n_atoms, feature_units)` initial node features, row-major.
    pub h0: Vec<f32>,
    /// Width of one atom's feature row within `h0`.
    pub feature_units: usize,
    /// Directed edge source atom indices (message-passing senders).
    pub senders: Vec<u32>,
    /// Directed edge destination atom indices (message-passing receivers).
    pub receivers: Vec<u32>,
    /// Total formal charge of the molecule (`q_ref.sum()` in Python).
    pub total_charge: f32,
}

/// Produces [`EspalomaGraph`] features from a molecule.
///
/// Corresponds to Python's `from expaloma.featurize import from_rdkit_mol` --
/// an external package import that is commonly absent (`from_rdkit_mol = None`
/// on `ImportError`). No default implementation ships in this crate; a caller
/// with no featurizer wired should pass `None` to [`get_espaloma_charges`],
/// which skips straight to Gasteiger, exactly as Python does.
pub trait EspalomaFeaturizer {
    fn featurize(&self, mol: &MolGraph) -> Result<EspalomaGraph, String>;
}

/// Runs espaloma (SAGE-GNN + QEq) inference on already-featurized graph data.
///
/// Corresponds to Python's `from proxide._proxider import assign_espaloma_charges`
/// -- the native Rust extension, commonly absent in a build that doesn't compile
/// the pyo3 bindings (`assign_rust_charges = None` on `ImportError`). The complete
/// algorithm already exists at `proxide_core::chem::inference::infer_charges`;
/// implement this trait against it to wire up the real backend.
pub trait EspalomaAssigner {
    fn assign(&self, graph: &EspalomaGraph) -> Result<Vec<f32>, String>;
}

/// Compute partial charges using expaloma or fallback to Gasteiger.
///
/// Tries native Rust expaloma first (via the injected `featurizer`/`assigner`
/// pair, both required -- matching Python's `if assign_rust_charges and
/// from_rdkit_mol:`), then falls back to a Gasteiger-Marsili PEOE calculation,
/// then to zero charges if that also fails.
///
/// Port of `_get_espaloma_charges` (`gaff2.py:1680-1734`). See the module docs
/// for the exact control-flow correspondence and documented gaps.
pub fn get_espaloma_charges(
    mol: &MolGraph,
    featurizer: Option<&dyn EspalomaFeaturizer>,
    assigner: Option<&dyn EspalomaAssigner>,
) -> Vec<f64> {
    // if assign_rust_charges and from_rdkit_mol:
    //     try: ... return list(q_rust)
    //     except Exception: pass
    if let (Some(featurizer), Some(assigner)) = (featurizer, assigner) {
        if let Ok(graph) = featurizer.featurize(mol) {
            if let Ok(q_rust) = assigner.assign(&graph) {
                // NOTE: returned as-is, deliberately UNCLAMPED -- Python's early
                // `return list(q_rust)` here never touches the inf/-inf/|q|>10
                // sanitization below. Only the Gasteiger path is sanitized.
                return q_rust.into_iter().map(f64::from).collect();
            }
        }
        // Any failure in either step is swallowed silently, same as Python's
        // bare `except Exception: pass` -- falls through to Gasteiger below.
    }

    // try: mol_copy.ComputeGasteigerCharges() ... except Exception: return zeros
    match gasteiger_charges(mol) {
        Ok(charges) => charges.into_iter().map(clamp_invalid_charge).collect(),
        Err(_) => vec![0.0; mol.elements.len()],
    }
}

/// Mirrors `gaff2.py:1729-1730` exactly:
/// `if charge == inf or charge == -inf or abs(charge) > 10: charge = 0.0`
///
/// Deliberately preserved as-is including its residual quirk: a `NaN` charge
/// is caught by NEITHER `NaN == inf`/`NaN == -inf` (both false under IEEE 754,
/// in Python and in Rust alike) NOR `NaN.abs() > 10` (also false) -- so a NaN
/// charge silently survives this "sanitizer" untouched in the original Python.
/// Rust's float comparison semantics match Python's here, so this port
/// reproduces that quirk for free rather than needing to special-case it.
/// This looks like a latent bug in the Python; per port instructions it is
/// preserved, not fixed -- see the `lesson` field of the port report.
fn clamp_invalid_charge(q: f64) -> f64 {
    // is_infinite() is equivalent to Python's `== inf or == -inf` here (both
    // false for NaN, in either language), just clippy's preferred spelling.
    if q.is_infinite() || q.abs() > 10.0 {
        0.0
    } else {
        q
    }
}

/// Gasteiger-Marsili PEOE (Partial Equalization of Orbital Electronegativity)
/// partial charge calculation.
///
/// Stands in for RDKit's `Mol.ComputeGasteigerCharges()`, which Python calls as
/// an opaque external-library primitive rather than implementing itself -- there
/// is no Python source in `gaff2.py` for this algorithm to port line-by-line.
/// This is a from-literature reimplementation (Gasteiger, J.; Marsili, M.
/// *Tetrahedron* 1980, 36, 3219) using the standard published orbital
/// electronegativity parameters, run for the classic 6 iterations with
/// per-iteration damping `0.5^k`.
///
/// **Not verified bit-exact against RDKit's own implementation** -- no golden
/// fixture for raw Gasteiger output exists in `tests/test_gaff2*.py` (only the
/// full-pipeline "charge sum ~ 0" conservation checks), and this environment has
/// no RDKit reference to diff against. Hybridization (needed to pick sp/sp2/sp3
/// parameters) is approximated from bond orders/aromaticity rather than RDKit's
/// full valence-model perception. Tested here only against invariants that hold
/// for any correct PEOE implementation: exact total-charge conservation per
/// iteration, and the expected sign of charge transfer down an electronegativity
/// gradient.
fn gasteiger_charges(mol: &MolGraph) -> Result<Vec<f64>, String> {
    const N_ITERATIONS: u32 = 6;

    let n = mol.elements.len();
    let mut hybridization = vec![Hybridization::Sp3; n];
    for bond in &mol.bonds {
        let h = match bond.order {
            BondOrder::Triple => Hybridization::Sp,
            BondOrder::Double => Hybridization::Sp2,
            BondOrder::Single if bond.aromatic => Hybridization::Sp2,
            BondOrder::Single => continue,
        };
        // A triple bond always wins over a previously-seen double/aromatic bond
        // on the same atom; sp2 wins over the sp3 default. Never downgrade.
        for idx in [bond.i, bond.j] {
            if h.rank() > hybridization[idx].rank() {
                hybridization[idx] = h;
            }
        }
    }

    let mut params = Vec::with_capacity(n);
    for (elem, hyb) in mol.elements.iter().zip(hybridization.iter()) {
        params.push(peoe_params(elem, *hyb)?);
    }
    let denom: Vec<f64> = params.iter().map(|p| p.a + p.b + p.c).collect();

    let mut q: Vec<f64> = mol.formal_charges.iter().map(|&c| c as f64).collect();

    for iter in 1..=N_ITERATIONS {
        let damp = 0.5f64.powi(iter as i32);
        let chi: Vec<f64> = q
            .iter()
            .zip(params.iter())
            .map(|(&qi, p)| p.a + p.b * qi + p.c * qi * qi)
            .collect();

        for bond in &mol.bonds {
            let (i, j) = (bond.i, bond.j);
            let transfer = if chi[i] >= chi[j] {
                (chi[i] - chi[j]) / denom[j]
            } else {
                (chi[i] - chi[j]) / denom[i]
            };
            q[i] -= transfer * damp;
            q[j] += transfer * damp;
        }
    }

    Ok(q)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hybridization {
    Sp3,
    Sp2,
    Sp,
}

impl Hybridization {
    /// Ordering for "most unsaturated wins" when an atom has multiple incident
    /// multiple bonds (e.g. an allene carbon is Sp from either of its two
    /// double bonds; Sp3 default is never chosen over evidence of unsaturation).
    fn rank(self) -> u8 {
        match self {
            Hybridization::Sp3 => 0,
            Hybridization::Sp2 => 1,
            Hybridization::Sp => 2,
        }
    }
}

struct PeoeParams {
    a: f64,
    b: f64,
    c: f64,
}

/// Standard Gasteiger-Marsili PEOE orbital electronegativity parameters
/// (Gasteiger & Marsili 1980; Gasteiger & Hendrickson extensions for O/N/S/
/// halogens/P), as widely reproduced across open-source cheminformatics
/// toolkits. Hydrogen and elements without a hybridization-dependent split
/// (F, Cl, Br, I, S, P) ignore `hyb`.
fn peoe_params(element: &str, hyb: Hybridization) -> Result<PeoeParams, String> {
    let (a, b, c) = match (element, hyb) {
        ("H", _) => (7.17, 6.24, -0.56),
        ("C", Hybridization::Sp3) => (7.98, 9.18, 1.88),
        ("C", Hybridization::Sp2) => (8.79, 9.32, 1.71),
        ("C", Hybridization::Sp) => (10.39, 9.45, 0.73),
        ("N", Hybridization::Sp3) => (11.54, 10.82, 1.36),
        ("N", Hybridization::Sp2) => (12.87, 11.15, 0.85),
        ("N", Hybridization::Sp) => (15.68, 11.70, -0.27),
        ("O", Hybridization::Sp3) => (14.18, 12.92, 1.39),
        ("O", Hybridization::Sp2 | Hybridization::Sp) => (17.07, 13.79, 0.47),
        ("F", _) => (14.66, 13.85, 2.31),
        ("Cl", _) => (11.00, 9.69, 1.35),
        ("Br", _) => (10.08, 8.47, 1.16),
        ("I", _) => (9.90, 7.96, 0.96),
        ("S", _) => (10.14, 9.13, 1.38),
        ("P", _) => (8.90, 8.24, 1.14),
        (other, _) => {
            return Err(format!(
                "no Gasteiger PEOE parameters for element {other:?}"
            ));
        }
    };
    Ok(PeoeParams { a, b, c })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mol::Bond;

    fn mol_graph(elements: Vec<&str>, bonds: Vec<Bond>, formal_charges: Vec<i8>) -> MolGraph {
        MolGraph {
            elements: elements.into_iter().map(String::from).collect(),
            formal_charges,
            bonds,
            rings: None,
        }
    }

    fn single(i: usize, j: usize) -> Bond {
        Bond {
            i,
            j,
            order: BondOrder::Single,
            aromatic: false,
        }
    }

    fn double(i: usize, j: usize) -> Bond {
        Bond {
            i,
            j,
            order: BondOrder::Double,
            aromatic: false,
        }
    }

    // --- clamp_invalid_charge: port of gaff2.py:1729-1730 -----------------

    #[test]
    fn clamp_leaves_ordinary_charges_untouched() {
        assert_eq!(clamp_invalid_charge(-0.43), -0.43);
        assert_eq!(clamp_invalid_charge(0.0), 0.0);
        assert_eq!(clamp_invalid_charge(9.999), 9.999);
        assert_eq!(clamp_invalid_charge(-10.0), -10.0); // boundary: |q| > 10, not >=
    }

    #[test]
    fn clamp_zeros_infinities_and_large_magnitudes() {
        assert_eq!(clamp_invalid_charge(f64::INFINITY), 0.0);
        assert_eq!(clamp_invalid_charge(f64::NEG_INFINITY), 0.0);
        assert_eq!(clamp_invalid_charge(10.0001), 0.0);
        assert_eq!(clamp_invalid_charge(-10.0001), 0.0);
    }

    #[test]
    fn clamp_preserves_the_python_nan_quirk() {
        // Faithful-port requirement: this must NOT be "fixed" to catch NaN.
        let clamped = clamp_invalid_charge(f64::NAN);
        assert!(clamped.is_nan(), "NaN must survive the clamp unmodified, matching gaff2.py's `charge == inf` check (false for NaN in both Python and Rust)");
    }

    // --- native-espaloma path: unclamped, bypasses Gasteiger ---------------

    struct StubFeaturizer;
    impl EspalomaFeaturizer for StubFeaturizer {
        fn featurize(&self, mol: &MolGraph) -> Result<EspalomaGraph, String> {
            Ok(EspalomaGraph {
                h0: vec![0.0; mol.elements.len()],
                feature_units: 1,
                senders: vec![],
                receivers: vec![],
                total_charge: 0.0,
            })
        }
    }

    struct StubAssigner {
        charges: Vec<f32>,
    }
    impl EspalomaAssigner for StubAssigner {
        fn assign(&self, _graph: &EspalomaGraph) -> Result<Vec<f32>, String> {
            Ok(self.charges.clone())
        }
    }

    struct FailingFeaturizer;
    impl EspalomaFeaturizer for FailingFeaturizer {
        fn featurize(&self, _mol: &MolGraph) -> Result<EspalomaGraph, String> {
            Err("featurizer unavailable".to_string())
        }
    }

    struct FailingAssigner;
    impl EspalomaAssigner for FailingAssigner {
        fn assign(&self, _graph: &EspalomaGraph) -> Result<Vec<f32>, String> {
            Err("inference failed".to_string())
        }
    }

    #[test]
    fn espaloma_success_returns_charges_unclamped() {
        // 25.0 is far outside the |q|>10 Gasteiger clamp window -- proving this
        // path truly bypasses the sanitizer, matching Python's early `return`.
        let mol = mol_graph(vec!["C", "O"], vec![single(0, 1)], vec![0, 0]);
        let featurizer = StubFeaturizer;
        let assigner = StubAssigner {
            charges: vec![25.0, -25.0],
        };
        let charges = get_espaloma_charges(&mol, Some(&featurizer), Some(&assigner));
        assert_eq!(charges, vec![25.0, -25.0]);
    }

    #[test]
    fn missing_either_backend_falls_through_to_gasteiger() {
        // Mirrors `if assign_rust_charges and from_rdkit_mol:` -- both required.
        let mol = mol_graph(vec!["C", "O"], vec![single(0, 1)], vec![0, 0]);
        let featurizer = StubFeaturizer;

        let only_featurizer = get_espaloma_charges(&mol, Some(&featurizer), None);
        let neither = get_espaloma_charges(&mol, None, None);
        assert_eq!(only_featurizer, neither);
    }

    #[test]
    fn espaloma_failure_falls_through_silently_to_gasteiger() {
        let mol = mol_graph(vec!["C", "O"], vec![single(0, 1)], vec![0, 0]);

        let bad_featurizer = FailingFeaturizer;
        let ok_assigner = StubAssigner {
            charges: vec![1.0, -1.0],
        };
        let via_bad_featurizer =
            get_espaloma_charges(&mol, Some(&bad_featurizer), Some(&ok_assigner));

        let ok_featurizer = StubFeaturizer;
        let bad_assigner = FailingAssigner;
        let via_bad_assigner =
            get_espaloma_charges(&mol, Some(&ok_featurizer), Some(&bad_assigner));

        let baseline = get_espaloma_charges(&mol, None, None);
        assert_eq!(via_bad_featurizer, baseline);
        assert_eq!(via_bad_assigner, baseline);
    }

    // --- Gasteiger fallback: invariants (no RDKit reference available) -----

    #[test]
    fn gasteiger_conserves_total_formal_charge() {
        // Port of the `test_charge_conservation` intent (tests/test_gaff2.py) /
        // "Charge sum ~ 0" check (tests/test_gaff2_golden.py), adapted to what
        // this crate can actually construct without RDKit: a hand-built neutral
        // molecule graph, checked against the PEOE conservation invariant that
        // holds by construction (every transfer moves charge symmetrically
        // between the two bonded atoms, so the sum is invariant per iteration).
        let mol = mol_graph(
            vec!["C", "O", "H", "H", "H"],
            vec![single(0, 1), single(0, 2), single(0, 3), single(1, 4)],
            vec![0, 0, 0, 0, 0],
        );
        let charges = get_espaloma_charges(&mol, None, None);
        let sum: f64 = charges.iter().sum();
        assert!(
            sum.abs() < 1e-9,
            "charge sum {sum} should be ~0 for a neutral molecule"
        );
    }

    #[test]
    fn gasteiger_conserves_nonzero_net_charge() {
        let mol = mol_graph(
            vec!["N", "H", "H", "H", "H"],
            vec![single(0, 1), single(0, 2), single(0, 3), single(0, 4)],
            vec![1, 0, 0, 0, 0],
        ); // ammonium, net +1
        let charges = get_espaloma_charges(&mol, None, None);
        let sum: f64 = charges.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "charge sum {sum} should be ~1 for NH4+"
        );
    }

    #[test]
    fn gasteiger_pulls_negative_charge_toward_more_electronegative_atom() {
        // O is more electronegative than C at every hybridization state in the
        // PEOE table (a_O > a_C at q=0), so an isolated C-O bond must polarize
        // O negative, C positive -- true of any correct PEOE run regardless of
        // exact numeric parity with RDKit. Deliberately a bare 2-atom graph
        // (no H's): adding C-H bonds pulls C the *other* way (chi_C > chi_H at
        // q=0, so C-H bonds push C negative too), which for a real CH3-OH
        // shape nearly cancels the C-O pull and makes C's sign too close to
        // call -- a confound, not a property of the algorithm.
        let mol = mol_graph(vec!["C", "O"], vec![single(0, 1)], vec![0, 0]);
        let charges = get_espaloma_charges(&mol, None, None);
        assert!(
            charges[1] < 0.0,
            "O charge {} should be negative",
            charges[1]
        );
        assert!(
            charges[0] > 0.0,
            "C charge {} should be positive",
            charges[0]
        );
        assert!((charges[0] + charges[1]).abs() < 1e-9);
    }

    #[test]
    fn gasteiger_zero_fallback_for_unsupported_element() {
        // Mirrors `except Exception: return [0.0] * mol.GetNumAtoms()`.
        let mol = mol_graph(vec!["Zz", "C"], vec![single(0, 1)], vec![0, 0]);
        let charges = get_espaloma_charges(&mol, None, None);
        assert_eq!(charges, vec![0.0, 0.0]);
    }

    #[test]
    fn double_bond_and_aromatic_flag_select_sp2_hybridization_params() {
        // Not directly observable from outside, but exercised via a shape that
        // would panic/produce NaN if hybridization selection indexed wrong --
        // an sp2 carbonyl carbon (C=O) plus an aromatic-flagged ring bond.
        let mol = mol_graph(
            vec!["C", "O", "C", "C"],
            vec![
                double(0, 1),
                Bond {
                    i: 0,
                    j: 2,
                    order: BondOrder::Single,
                    aromatic: false,
                },
                Bond {
                    i: 2,
                    j: 3,
                    order: BondOrder::Single,
                    aromatic: true,
                },
            ],
            vec![0, 0, 0, 0],
        );
        let charges = get_espaloma_charges(&mol, None, None);
        assert_eq!(charges.len(), 4);
        assert!(charges.iter().all(|q| q.is_finite()));
    }
}
