//! `RingPucker` definition assembly (traversal order, spec §2c) and the
//! Cremer-Pople dominant (m=2) phase-angle extraction used per-frame.

#[derive(Debug, Clone)]
pub struct RingPucker {
    pub ring_atoms: Vec<usize>,
    pub ring_size: usize,
}

/// Below this amplitude (Angstrom), a ring is treated as transiently
/// planar and its phase is defined as `0` by convention, never `NaN`
/// (spec §2b).
const PUCKER_AMPLITUDE_EPSILON: f64 = 1e-3;

/// Fixed, deterministic traversal convention (spec §2c): start at the
/// lowest canonical index, walk toward whichever of its two ring
/// neighbors has the lower canonical index.
pub fn ring_traversal_order(ring: &[usize], adjacency: &[Vec<usize>]) -> Vec<usize> {
    let ring_set: std::collections::HashSet<usize> = ring.iter().copied().collect();
    let start = *ring.iter().min().expect("ring must be non-empty");

    let mut ring_neighbors: Vec<usize> = adjacency[start]
        .iter()
        .copied()
        .filter(|nb| ring_set.contains(nb))
        .collect();
    ring_neighbors.sort_unstable();

    let mut prev = start;
    let mut current = ring_neighbors[0];
    let mut order = vec![start, current];

    while order.len() < ring.len() {
        let next = adjacency[current]
            .iter()
            .copied()
            .find(|nb| ring_set.contains(nb) && *nb != prev)
            .expect("ring traversal: no unvisited ring neighbor found");
        order.push(next);
        prev = current;
        current = next;
    }
    order
}

/// Rings 5-8: get a `RingPucker`. Rings >=9 (macrocycles): flagged in the
/// returned `unrepresented_ring_dof` list instead (spec §2c). Rings <5
/// (e.g. epoxides) get neither -- out of scope, no representable pucker DOF.
pub fn build_ring_puckers(
    rings: &[Vec<usize>],
    adjacency: &[Vec<usize>],
) -> (Vec<RingPucker>, Vec<Vec<usize>>) {
    let mut puckers = Vec::new();
    let mut unrepresented = Vec::new();
    for ring in rings {
        let size = ring.len();
        if (5..=8).contains(&size) {
            puckers.push(RingPucker {
                ring_atoms: ring_traversal_order(ring, adjacency),
                ring_size: size,
            });
        } else if size >= 9 {
            let mut atoms = ring.clone();
            atoms.sort_unstable();
            unrepresented.push(atoms);
        }
    }
    (puckers, unrepresented)
}

/// Dominant (m=2) Cremer-Pople phase angle, radians, for a ring given its
/// atoms' positions IN THE FIXED TRAVERSAL ORDER from
/// [`ring_traversal_order`]. Amplitude is computed but not returned (spec
/// §2b/§5 item 2: amplitude is punted for v1).
pub fn cremer_pople_phase(ring_positions: &[[f64; 3]]) -> f64 {
    let n = ring_positions.len();
    let mean = {
        let mut m = [0.0f64; 3];
        for p in ring_positions {
            m[0] += p[0];
            m[1] += p[1];
            m[2] += p[2];
        }
        [m[0] / n as f64, m[1] / n as f64, m[2] / n as f64]
    };
    let centered: Vec<[f64; 3]> = ring_positions
        .iter()
        .map(|p| [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]])
        .collect();

    let mut r_prime = [0.0f64; 3];
    let mut r_double_prime = [0.0f64; 3];
    for (j, c) in centered.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * j as f64 / n as f64;
        for k in 0..3 {
            r_prime[k] += c[k] * angle.sin();
            r_double_prime[k] += c[k] * angle.cos();
        }
    }
    let normal_unnorm = [
        r_prime[1] * r_double_prime[2] - r_prime[2] * r_double_prime[1],
        r_prime[2] * r_double_prime[0] - r_prime[0] * r_double_prime[2],
        r_prime[0] * r_double_prime[1] - r_prime[1] * r_double_prime[0],
    ];
    let norm_mag =
        (normal_unnorm[0].powi(2) + normal_unnorm[1].powi(2) + normal_unnorm[2].powi(2)).sqrt();
    let normal = [
        normal_unnorm[0] / norm_mag,
        normal_unnorm[1] / norm_mag,
        normal_unnorm[2] / norm_mag,
    ];

    let z: Vec<f64> = centered
        .iter()
        .map(|c| c[0] * normal[0] + c[1] * normal[1] + c[2] * normal[2])
        .collect();

    let mut a = 0.0;
    let mut b = 0.0;
    for (j, &zj) in z.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * 2.0 * j as f64 / n as f64;
        a += zj * angle.cos();
        b -= zj * angle.sin();
    }
    let amplitude = (2.0 / n as f64).sqrt() * (a * a + b * b).sqrt();

    if amplitude < PUCKER_AMPLITUDE_EPSILON {
        0.0
    } else {
        b.atan2(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traversal_starts_at_lowest_canonical_index() {
        // Hexagon ring 0-1-2-3-4-5-0.
        let adjacency = vec![
            vec![1, 5],
            vec![0, 2],
            vec![1, 3],
            vec![2, 4],
            vec![3, 5],
            vec![4, 0],
        ];
        let ring = vec![3, 0, 5, 1, 4, 2];
        let order = ring_traversal_order(&ring, &adjacency);
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 1); // lower of {1, 5}
    }

    #[test]
    fn planar_ring_phase_is_zero_by_convention() {
        let n = 6;
        let radius = 1.4;
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|j| {
                let angle = 2.0 * std::f64::consts::PI * j as f64 / n as f64;
                [radius * angle.cos(), radius * angle.sin(), 0.0]
            })
            .collect();
        assert_eq!(cremer_pople_phase(&positions), 0.0);
    }

    #[test]
    fn puckered_ring_gives_deterministic_finite_phase() {
        let n = 6;
        let radius = 1.4;
        // NOTE: a symmetric alternating +/-z "chair-like" displacement is a
        // pure m=3 (chair) Fourier mode, which is exactly orthogonal to the
        // m=2 mode this function extracts -- it gives a=b=0 (verified: the
        // dominant-m=2 amplitude for that fixture is 0 to float precision),
        // so it would trivially satisfy the "planar by convention" branch
        // rather than exercising the puckered branch. Use an asymmetric
        // (linear-ramp) out-of-plane displacement instead, which has a
        // nonzero m=2 component by construction.
        let positions: Vec<[f64; 3]> = (0..n)
            .map(|j| {
                let angle = 2.0 * std::f64::consts::PI * j as f64 / n as f64;
                let z = -0.3 + 0.1 * j as f64;
                [radius * angle.cos(), radius * angle.sin(), z]
            })
            .collect();
        let phase_a = cremer_pople_phase(&positions);
        let phase_b = cremer_pople_phase(&positions);
        assert!(phase_a.is_finite());
        assert_ne!(phase_a, 0.0);
        assert_eq!(phase_a, phase_b); // deterministic
    }

    #[test]
    fn rings_ge_9_atoms_flagged_as_unrepresented() {
        let ring: Vec<usize> = (0..9).collect();
        let adjacency: Vec<Vec<usize>> = (0..9).map(|i| vec![(i + 8) % 9, (i + 1) % 9]).collect();
        let (puckers, unrepresented) = build_ring_puckers(std::slice::from_ref(&ring), &adjacency);
        assert!(puckers.is_empty());
        assert_eq!(unrepresented, vec![ring]);
    }

    #[test]
    fn rings_5_to_8_atoms_get_pucker_definitions() {
        for size in 5..=8 {
            let ring: Vec<usize> = (0..size).collect();
            let adjacency: Vec<Vec<usize>> = (0..size)
                .map(|i| vec![(i + size - 1) % size, (i + 1) % size])
                .collect();
            let (puckers, unrepresented) = build_ring_puckers(&[ring], &adjacency);
            assert_eq!(puckers.len(), 1);
            assert_eq!(puckers[0].ring_size, size);
            assert!(unrepresented.is_empty());
        }
    }
}
