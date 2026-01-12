//! Cell list algorithm for O(N) neighbor search
//!
//! Implements spatial hashing for efficient neighbor queries.
//! Particularly useful for bond inference and distance-based computations
//! on large protein structures.

#![allow(dead_code)]

use std::collections::HashMap;

/// Cell index (grid position in 3D space)
type CellIndex = (i32, i32, i32);

/// Cell list for spatial hashing
pub struct CellList {
    /// Cell size (should be >= cutoff distance)
    cell_size: f32,
    /// Maps cell index to list of atom indices in that cell
    cells: HashMap<CellIndex, Vec<usize>>,
    /// Bounding box minimum
    min_coords: [f32; 3],
}

impl CellList {
    /// Build a cell list from coordinates
    ///
    /// # Arguments
    /// * `coords` - Array of (x, y, z) coordinates
    /// * `cell_size` - Size of each cell (should be >= cutoff for queries)
    pub fn new(coords: &[[f32; 3]], cell_size: f32) -> Self {
        let mut cells: HashMap<CellIndex, Vec<usize>> = HashMap::new();

        if coords.is_empty() {
            return Self {
                cell_size,
                cells,
                min_coords: [0.0, 0.0, 0.0],
            };
        }

        // Find bounding box
        let mut min_coords = [f32::MAX; 3];
        let mut _max_coords = [f32::MIN; 3];

        for coord in coords {
            for i in 0..3 {
                min_coords[i] = min_coords[i].min(coord[i]);
                _max_coords[i] = _max_coords[i].max(coord[i]);
            }
        }

        // Assign atoms to cells
        for (idx, coord) in coords.iter().enumerate() {
            let cell_idx = Self::coord_to_cell(coord, &min_coords, cell_size);
            cells.entry(cell_idx).or_default().push(idx);
        }

        Self {
            cell_size,
            cells,
            min_coords,
        }
    }

    /// Convert coordinate to cell index
    fn coord_to_cell(coord: &[f32; 3], min_coords: &[f32; 3], cell_size: f32) -> CellIndex {
        let x = ((coord[0] - min_coords[0]) / cell_size).floor() as i32;
        let y = ((coord[1] - min_coords[1]) / cell_size).floor() as i32;
        let z = ((coord[2] - min_coords[2]) / cell_size).floor() as i32;
        (x, y, z)
    }

    /// Get cell index for a coordinate
    pub fn get_cell(&self, coord: &[f32; 3]) -> CellIndex {
        Self::coord_to_cell(coord, &self.min_coords, self.cell_size)
    }

    /// Get all neighboring cell indices (including the cell itself)
    /// Returns the 27 cells in a 3x3x3 neighborhood
    fn get_neighbor_cells(cell: CellIndex) -> Vec<CellIndex> {
        let mut neighbors = Vec::with_capacity(27);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    neighbors.push((cell.0 + dx, cell.1 + dy, cell.2 + dz));
                }
            }
        }
        neighbors
    }

    /// Find all atoms within cutoff distance of a query point
    ///
    /// Returns indices of atoms within `cutoff` distance.
    /// Assumes `cutoff <= cell_size`.
    pub fn query_neighbors(
        &self,
        query: &[f32; 3],
        coords: &[[f32; 3]],
        cutoff: f32,
    ) -> Vec<usize> {
        let cutoff_sq = cutoff * cutoff;
        let query_cell = self.get_cell(query);
        let neighbor_cells = Self::get_neighbor_cells(query_cell);

        let mut neighbors = Vec::new();

        for cell_idx in neighbor_cells {
            if let Some(atom_indices) = self.cells.get(&cell_idx) {
                for &idx in atom_indices {
                    let dx = coords[idx][0] - query[0];
                    let dy = coords[idx][1] - query[1];
                    let dz = coords[idx][2] - query[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;

                    if dist_sq <= cutoff_sq {
                        neighbors.push(idx);
                    }
                }
            }
        }

        neighbors
    }

    /// Find all atom pairs within cutoff distance (for bond inference)
    ///
    /// Returns pairs `(i, j)` where `i < j` and distance < cutoff.
    pub fn find_pairs_within_cutoff(
        &self,
        coords: &[[f32; 3]],
        cutoff: f32,
    ) -> Vec<(usize, usize)> {
        let cutoff_sq = cutoff * cutoff;
        let mut pairs = Vec::new();

        // For each cell, check pairs within the cell and with neighbor cells
        for (&cell_idx, atom_indices) in &self.cells {
            // Pairs within the same cell
            for (i, &idx_a) in atom_indices.iter().enumerate() {
                for &idx_b in atom_indices.iter().skip(i + 1) {
                    let dist_sq = distance_squared(&coords[idx_a], &coords[idx_b]);
                    if dist_sq <= cutoff_sq {
                        let (min_idx, max_idx) = if idx_a < idx_b {
                            (idx_a, idx_b)
                        } else {
                            (idx_b, idx_a)
                        };
                        pairs.push((min_idx, max_idx));
                    }
                }

                // Pairs with neighboring cells (only check "forward" neighbors to avoid duplicates)
                let neighbor_cells = Self::get_neighbor_cells(cell_idx);
                for neighbor_cell in neighbor_cells {
                    // Only check cells that are "greater than" current cell lexicographically
                    if neighbor_cell <= cell_idx {
                        continue;
                    }

                    if let Some(neighbor_atoms) = self.cells.get(&neighbor_cell) {
                        for &idx_b in neighbor_atoms {
                            let dist_sq = distance_squared(&coords[idx_a], &coords[idx_b]);
                            if dist_sq <= cutoff_sq {
                                let (min_idx, max_idx) = if idx_a < idx_b {
                                    (idx_a, idx_b)
                                } else {
                                    (idx_b, idx_a)
                                };
                                pairs.push((min_idx, max_idx));
                            }
                        }
                    }
                }
            }
        }

        pairs
    }
}

/// Squared distance between two 3D points
#[inline]
fn distance_squared(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// O(N) neighbor search using cell lists
///
/// Returns indices of all neighbors within `cutoff` for each point.
pub fn find_neighbors_within_cutoff_fast(coords: &[[f32; 3]], cutoff: f32) -> Vec<Vec<usize>> {
    if coords.is_empty() {
        return Vec::new();
    }

    let cell_list = CellList::new(coords, cutoff);

    coords
        .iter()
        .enumerate()
        .map(|(i, coord)| {
            cell_list
                .query_neighbors(coord, coords, cutoff)
                .into_iter()
                .filter(|&j| j != i) // Exclude self
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_list_basic() {
        let coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 10.0, 10.0], // Far away
        ];

        let cell_list = CellList::new(&coords, 2.0);

        // Query neighbors of origin within 1.5
        let neighbors = cell_list.query_neighbors(&[0.0, 0.0, 0.0], &coords, 1.5);

        // Should include indices 0, 1, 2 (all within 1.5 of origin)
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
        assert!(!neighbors.contains(&3)); // Too far
    }

    #[test]
    fn test_find_pairs() {
        let coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [10.0, 0.0, 0.0], // Far away
        ];

        let cell_list = CellList::new(&coords, 2.0);
        let pairs = cell_list.find_pairs_within_cutoff(&coords, 1.5);

        // Should find (0,1) since distance is 1.0
        assert!(pairs.contains(&(0, 1)));
        // (0,2) has distance 2.0 > 1.5, should not be included
        assert!(!pairs.contains(&(0, 2)));
    }

    #[test]
    fn test_fast_neighbor_search() {
        let coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 5.0],
        ];

        let neighbors = find_neighbors_within_cutoff_fast(&coords, 2.5);

        // Origin should have 2 neighbors within 2.5 (indices 1 and 2)
        assert_eq!(neighbors[0].len(), 2);
        assert!(neighbors[0].contains(&1));
        assert!(neighbors[0].contains(&2));
        assert!(!neighbors[0].contains(&3)); // Distance 5 > 2.5
    }
}
