//! Idealized C-beta placement from backbone geometry.
//!
//! Provides an empirical closed-form estimate of the C-beta atom position
//! from a residue's N, CA, C backbone coordinates. The formula is general —
//! it does not special-case any residue type — but its canonical use is to
//! impute a C-beta position for glycine, which has no real side chain and
//! therefore no CB atom to read from a structure.
//!
//! Ported from `aminx.utils.coordinates.compute_c_beta`
//! (`aminx/src/aminx/utils/coordinates.py`), which is itself the standard
//! empirical idealized-C-beta-placement formula used broadly in structural
//! bioinformatics: a linear combination of the N->CA and CA->C backbone bond
//! vectors (plus their cross product) with fixed, empirically-fit
//! coefficients. This supersedes an earlier ad hoc bisector construction
//! (`_pseudo_cb_gly`, previously duplicated in sweetprots) — this is the
//! single shared implementation all consumers (aminx, plegadx, sweetprots)
//! should use.

/// Compute an idealized C-beta position from pre-differenced backbone bond
/// vectors.
///
/// # Arguments
/// * `n_to_ca` - Bond vector from N to CA, i.e. `ca - n`.
/// * `ca_to_c` - Bond vector from CA to C, i.e. `c - ca`.
/// * `ca` - Coordinates of the alpha carbon (the CB estimate is anchored
///   here).
///
/// # Returns
/// Estimated C-beta coordinates.
///
/// # Sign convention (verified against aminx's actual runtime behavior, not
/// its param names)
///
/// aminx's `compute_c_beta` takes parameters literally named
/// `alpha_to_nitrogen` and `carbon_to_alpha`, which — read naively — would
/// suggest CA->N and C->CA. But its one call site
/// (`compute_backbone_coordinates`, `aminx/src/aminx/utils/coordinates.py:88-90`)
/// actually passes `alpha_carbon - nitrogen` (= `ca - n`, i.e. N->CA) and
/// `carbon - alpha_carbon` (= `c - ca`, i.e. CA->C) — the reverse of what
/// the names imply. The function's own docstring example uses variable
/// names `n_to_ca` / `ca_to_c` for these same two positional arguments,
/// confirming this is the intended (if confusingly named) convention. This
/// function uses the same convention as the actual runtime behavior:
/// `n_to_ca` first, `ca_to_c` second.
pub fn idealized_cbeta(n_to_ca: [f32; 3], ca_to_c: [f32; 3], ca: [f32; 3]) -> [f32; 3] {
    const F1: f32 = -0.582_734_3;
    const F2: f32 = 0.568_028_3;
    const F3: f32 = -0.540_674_66;

    let cross = [
        n_to_ca[1] * ca_to_c[2] - n_to_ca[2] * ca_to_c[1],
        n_to_ca[2] * ca_to_c[0] - n_to_ca[0] * ca_to_c[2],
        n_to_ca[0] * ca_to_c[1] - n_to_ca[1] * ca_to_c[0],
    ];

    [
        F1 * cross[0] + F2 * n_to_ca[0] + F3 * ca_to_c[0] + ca[0],
        F1 * cross[1] + F2 * n_to_ca[1] + F3 * ca_to_c[1] + ca[1],
        F1 * cross[2] + F2 * n_to_ca[2] + F3 * ca_to_c[2] + ca[2],
    ]
}

/// Convenience wrapper computing the idealized C-beta position directly
/// from raw N, CA, C atom coordinates rather than pre-differenced bond
/// vectors. See [`idealized_cbeta`] for the underlying formula and sign
/// convention.
pub fn idealized_cbeta_from_atoms(n: [f32; 3], ca: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let n_to_ca = [ca[0] - n[0], ca[1] - n[1], ca[2] - n[2]];
    let ca_to_c = [c[0] - ca[0], c[1] - ca[1], c[2] - ca[2]];
    idealized_cbeta(n_to_ca, ca_to_c, ca)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-check against aminx's own docstring / unit-test example
    /// (`aminx/tests/utils/test_coordinates.py::test_compute_c_beta`):
    /// `n_to_ca = [1, 0, 0]`, `ca_to_c = [0, 1, 0]`, `ca = [0, 0, 0]`.
    ///
    /// Hand-derivation of aminx's formula for this input:
    /// `cross(n_to_ca, ca_to_c) = cross([1,0,0], [0,1,0]) = [0, 0, 1]`
    /// `term1 = f1 * [0,0,1] = [0, 0, -0.58273431]`
    /// `term2 = f2 * [1,0,0] = [0.56802827, 0, 0]`
    /// `term3 = f3 * [0,1,0] = [0, -0.54067466, 0]`
    /// `cb = term1 + term2 + term3 + [0,0,0]`
    ///    `= [0.56802827, -0.54067466, -0.58273431]`
    #[test]
    fn test_idealized_cbeta_matches_aminx_reference_example() {
        let n_to_ca = [1.0, 0.0, 0.0];
        let ca_to_c = [0.0, 1.0, 0.0];
        let ca = [0.0, 0.0, 0.0];

        let cb = idealized_cbeta(n_to_ca, ca_to_c, ca);
        let expected = [0.568_028_27, -0.540_674_66, -0.582_734_31];

        for i in 0..3 {
            assert!(
                (cb[i] - expected[i]).abs() < 1e-6,
                "component {}: got {} expected {}",
                i,
                cb[i],
                expected[i]
            );
        }
    }

    /// Same case, routed through the raw-atom-coordinate convenience
    /// wrapper: n=[-1,0,0], ca=[0,0,0], c=[0,1,0] differences to the same
    /// bond vectors as the reference example above (n_to_ca = ca-n =
    /// [1,0,0], ca_to_c = c-ca = [0,1,0]).
    #[test]
    fn test_idealized_cbeta_from_atoms_matches_bond_vector_form() {
        let n = [-1.0, 0.0, 0.0];
        let ca = [0.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];

        let cb = idealized_cbeta_from_atoms(n, ca, c);
        let expected = idealized_cbeta([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]);

        for i in 0..3 {
            assert!((cb[i] - expected[i]).abs() < 1e-6);
        }
    }

    /// CB should sit near CA, anchored by CA, not translated arbitrarily far
    /// — sanity bound using a realistic-scale bond-vector input (~1.5 A
    /// bonds), checking the result stays within a few Angstroms of CA.
    #[test]
    fn test_idealized_cbeta_realistic_scale_stays_near_ca() {
        let n_to_ca = [1.46, 0.0, 0.0];
        let ca_to_c = [0.0, 1.52, 0.0];
        let ca = [10.0, 20.0, 30.0];

        let cb = idealized_cbeta(n_to_ca, ca_to_c, ca);
        let dist = ((cb[0] - ca[0]).powi(2) + (cb[1] - ca[1]).powi(2) + (cb[2] - ca[2]).powi(2))
            .sqrt();
        assert!(
            dist > 0.5 && dist < 3.0,
            "expected CB within a few Angstroms of CA, got distance {}",
            dist
        );
    }
}
