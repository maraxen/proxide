/// Normalize angle to [-180.0, 180.0).
/// Matches MSL `angleToStandardRange` (mstrotlib.cpp:137–139).
pub(crate) fn angle_to_standard(a: f64) -> f64 {
    let mut v = a.rem_euclid(360.0);
    if v >= 180.0 { v -= 360.0; }
    v
}

/// Unsigned circular distance between two angles (degrees).
/// Returns the shorter arc, in [0.0, 180.0].
pub(crate) fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).abs() % 360.0;
    if d > 180.0 { 360.0 - d } else { d }
}

/// Index of the bin center closest to `query` (circular). First minimum wins on tie.
/// Matches MSL `findClosestAngle` strict-less-than update.
pub(crate) fn find_closest_angle(centers: &[f64], query: f64) -> usize {
    centers.iter().enumerate()
        .min_by(|(_, &a), (_, &b)|
            angle_diff(a, query).total_cmp(&angle_diff(b, query)))
        .unwrap()
        .0
}
