use proxide_confind::grid::ProximityGrid;

fn brute_force_within(
    points: &[[f64; 3]],
    tags: &[usize],
    center: [f64; 3],
    dmin: f64,
    dmax: f64,
) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let d = ((p[0] - center[0]).powi(2)
            + (p[1] - center[1]).powi(2)
            + (p[2] - center[2]).powi(2))
        .sqrt();
        if d >= dmin && d <= dmax {
            out.push(tags[i]);
        }
    }
    out.sort_unstable();
    out
}

#[test]
fn empty_grid_returns_empty() {
    let g = ProximityGrid::<usize>::build(&[], vec![], 3.0);
    let result = g.points_within([0.0, 0.0, 0.0], 0.0, 10.0);
    assert!(result.is_empty());
}

#[test]
fn single_point_at_origin_query_at_origin() {
    let pts = [[1.0, 2.0, 3.0]];
    let tags = vec![42usize];
    let g = ProximityGrid::build(&pts, tags, 3.0);

    // Query at exact same position with dmax=0
    let res = g.points_within([1.0, 2.0, 3.0], 0.0, 0.0);
    assert_eq!(res, vec![42]);
}

#[test]
fn single_point_query_far_away() {
    let pts = [[1.0, 2.0, 3.0]];
    let tags = vec![42usize];
    let g = ProximityGrid::build(&pts, tags, 3.0);

    // Query far away from the point
    let res = g.points_within([10.0, 0.0, 0.0], 0.0, 1.0);
    assert!(res.is_empty());
}

#[test]
fn annular_query_with_dmin_greater_than_zero() {
    let pts = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ];
    let tags: Vec<usize> = vec![0, 1, 2, 3, 4];
    let g = ProximityGrid::build(&pts, tags, 3.0);

    // Query with dmin=1.5, dmax=2.5 should find point at [2.0, 0, 0]
    // and possibly others in that annular shell
    let res = g.points_within([0.0, 0.0, 0.0], 1.5, 2.5);
    assert!(res.contains(&2));
}

#[test]
fn grid_vs_brute_force_deterministic() {
    // Generate deterministic pseudo-random points using LCG.
    let mut pts: Vec<[f64; 3]> = Vec::new();
    let mut seed: u64 = 12345;

    for _ in 0..100 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = (seed >> 33) as f64 / u32::MAX as f64 * 20.0 - 10.0;

        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let y = (seed >> 33) as f64 / u32::MAX as f64 * 20.0 - 10.0;

        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let z = (seed >> 33) as f64 / u32::MAX as f64 * 20.0 - 10.0;

        pts.push([x, y, z]);
    }

    let tags: Vec<usize> = (0..pts.len()).collect();
    let g = ProximityGrid::build(&pts, tags.clone(), 3.0);

    let center = [1.5, -2.0, 0.3];
    let dmin = 0.0;
    let dmax = 4.0;

    let mut grid_res = g.points_within(center, dmin, dmax);
    let mut brute_res = brute_force_within(&pts, &tags, center, dmin, dmax);

    grid_res.sort_unstable();
    brute_res.sort_unstable();

    assert_eq!(grid_res, brute_res, "grid and brute-force results differ");
}

#[test]
fn grid_vs_brute_force_annular() {
    // Generate same deterministic points but test annular query.
    let mut pts: Vec<[f64; 3]> = Vec::new();
    let mut seed: u64 = 12345;

    for _ in 0..100 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = (seed >> 33) as f64 / u32::MAX as f64 * 20.0 - 10.0;

        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let y = (seed >> 33) as f64 / u32::MAX as f64 * 20.0 - 10.0;

        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let z = (seed >> 33) as f64 / u32::MAX as f64 * 20.0 - 10.0;

        pts.push([x, y, z]);
    }

    let tags: Vec<usize> = (0..pts.len()).collect();
    let g = ProximityGrid::build(&pts, tags.clone(), 3.0);

    let center = [1.5, -2.0, 0.3];
    let dmin = 1.0;
    let dmax = 3.5;

    let mut grid_res = g.points_within(center, dmin, dmax);
    let mut brute_res = brute_force_within(&pts, &tags, center, dmin, dmax);

    grid_res.sort_unstable();
    brute_res.sort_unstable();

    assert_eq!(grid_res, brute_res, "annular query: grid and brute-force differ");
}
