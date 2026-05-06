#[cfg(any(test, feature = "testing"))]
pub fn normalize_float(f: f32) -> f32 {
    (f * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(any(test, feature = "testing"))]
pub fn normalize_coords(coords: &[[f32; 3]]) -> Vec<[f32; 3]> {
    coords
        .iter()
        .map(|c| [
            normalize_float(c[0]),
            normalize_float(c[1]),
            normalize_float(c[2]),
        ])
        .collect()
}
