/// Generic 3D bucketed spatial index in f64.
/// Replaces CellList (f32-only) for ConFind's geometry needs.
pub struct ProximityGrid<T: Clone> {
    /// Flat cell storage: cell_at(ix, iy, iz) = cells[ix * dims[1]*dims[2] + iy*dims[2] + iz].
    cells: Vec<Vec<usize>>,
    /// All points stored flat; `cells` holds indices into these.
    points: Vec<[f64; 3]>,
    tags: Vec<T>,
    origin: [f64; 3],
    cell_size: f64,
    dims: [usize; 3],
}

impl<T: Clone + Send + Sync> ProximityGrid<T> {
    /// Build a grid from a list of points and their associated tags.
    /// `char_dist` is the characteristic distance (cell size = char_dist / 2).
    pub fn build(points: &[[f64; 3]], tags: Vec<T>, char_dist: f64) -> Self {
        assert_eq!(points.len(), tags.len());
        let cell_size = char_dist / 2.0;

        if points.is_empty() {
            return Self {
                cells: Vec::new(),
                points: Vec::new(),
                tags,
                origin: [0.0; 3],
                cell_size,
                dims: [0; 3],
            };
        }

        // Bounding box.
        let mut min = points[0];
        let mut max = points[0];
        for p in points {
            for d in 0..3 {
                if p[d] < min[d] {
                    min[d] = p[d];
                }
                if p[d] > max[d] {
                    max[d] = p[d];
                }
            }
        }

        // Pad and compute cell dims.
        let pad = cell_size;
        let origin = [min[0] - pad, min[1] - pad, min[2] - pad];
        let dims = [
            ((max[0] - origin[0]) / cell_size).ceil() as usize + 2,
            ((max[1] - origin[1]) / cell_size).ceil() as usize + 2,
            ((max[2] - origin[2]) / cell_size).ceil() as usize + 2,
        ];

        let total = dims[0] * dims[1] * dims[2];
        let mut cells: Vec<Vec<usize>> = vec![Vec::new(); total];

        for (i, p) in points.iter().enumerate() {
            let ci = cell_coord(p, &origin, cell_size, &dims);
            cells[flat_idx(&ci, &dims)].push(i);
        }

        Self {
            cells,
            points: points.to_vec(),
            tags,
            origin,
            cell_size,
            dims,
        }
    }

    /// Return tags of all points with distance in `[dmin, dmax]` from `center`.
    pub fn points_within(&self, center: [f64; 3], dmin: f64, dmax: f64) -> Vec<T> {
        if self.points.is_empty() {
            return Vec::new();
        }
        let r = dmax;
        let steps = (r / self.cell_size).ceil() as i64 + 1;

        let cx = ((center[0] - self.origin[0]) / self.cell_size) as i64;
        let cy = ((center[1] - self.origin[1]) / self.cell_size) as i64;
        let cz = ((center[2] - self.origin[2]) / self.cell_size) as i64;

        let dmin2 = dmin * dmin;
        let dmax2 = dmax * dmax;

        let mut result = Vec::new();
        for dx in -steps..=steps {
            for dy in -steps..=steps {
                for dz in -steps..=steps {
                    let ix = cx + dx;
                    let iy = cy + dy;
                    let iz = cz + dz;
                    if ix < 0
                        || iy < 0
                        || iz < 0
                        || ix >= self.dims[0] as i64
                        || iy >= self.dims[1] as i64
                        || iz >= self.dims[2] as i64
                    {
                        continue;
                    }
                    let fi = flat_idx(&[ix as usize, iy as usize, iz as usize], &self.dims);
                    for &pi in &self.cells[fi] {
                        let p = &self.points[pi];
                        let d2 = (p[0] - center[0]).powi(2)
                            + (p[1] - center[1]).powi(2)
                            + (p[2] - center[2]).powi(2);
                        if d2 >= dmin2 && d2 <= dmax2 {
                            result.push(self.tags[pi].clone());
                        }
                    }
                }
            }
        }
        result
    }

    pub fn point_size(&self) -> usize {
        self.points.len()
    }

    pub fn get_point(&self, i: usize) -> [f64; 3] {
        self.points[i]
    }

    pub fn get_tag(&self, i: usize) -> &T {
        &self.tags[i]
    }
}

fn cell_coord(p: &[f64; 3], origin: &[f64; 3], cell_size: f64, dims: &[usize; 3]) -> [usize; 3] {
    [
        ((p[0] - origin[0]) / cell_size)
            .floor()
            .max(0.0)
            .min((dims[0] - 1) as f64) as usize,
        ((p[1] - origin[1]) / cell_size)
            .floor()
            .max(0.0)
            .min((dims[1] - 1) as f64) as usize,
        ((p[2] - origin[2]) / cell_size)
            .floor()
            .max(0.0)
            .min((dims[2] - 1) as f64) as usize,
    ]
}

fn flat_idx(ci: &[usize; 3], dims: &[usize; 3]) -> usize {
    ci[0] * dims[1] * dims[2] + ci[1] * dims[2] + ci[2]
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn empty_grid() {
        let g = ProximityGrid::<usize>::build(&[], vec![], 3.0);
        assert!(g.points_within([0.0, 0.0, 0.0], 0.0, 10.0).is_empty());
    }

    #[test]
    fn single_point() {
        let pts = [[1.0, 2.0, 3.0]];
        let tags = vec![42usize];
        let g = ProximityGrid::build(&pts, tags, 3.0);
        let res = g.points_within([1.0, 2.0, 3.0], 0.0, 0.0);
        assert_eq!(res, vec![42]);
        let res2 = g.points_within([10.0, 0.0, 0.0], 0.0, 1.0);
        assert!(res2.is_empty());
    }

    #[test]
    fn grid_vs_brute_force_random() {
        // Generate deterministic pseudo-random points.
        let mut pts: Vec<[f64; 3]> = Vec::new();
        let mut seed: u64 = 12345;
        for _ in 0..200 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (seed >> 33) as f64 / u32::MAX as f64 * 20.0 - 10.0;
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let y = (seed >> 33) as f64 / u32::MAX as f64 * 20.0 - 10.0;
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
        assert_eq!(grid_res, brute_res);

        // Also test annular query (dmin > 0).
        let dmin2 = 1.0;
        let dmax2 = 3.5;
        let mut grid_res2 = g.points_within(center, dmin2, dmax2);
        let mut brute_res2 = brute_force_within(&pts, &tags, center, dmin2, dmax2);
        grid_res2.sort_unstable();
        brute_res2.sort_unstable();
        assert_eq!(grid_res2, brute_res2);
    }
}
