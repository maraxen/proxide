/// Local coordinate frame: an origin and three orthonormal axes.
#[derive(Clone, Debug)]
pub struct Frame {
    pub origin: [f64; 3],
    pub x: [f64; 3],
    pub y: [f64; 3],
    pub z: [f64; 3],
}

impl Frame {
    /// Construct from raw (unnormalized) axis vectors. Normalizes all three.
    pub fn new(origin: [f64; 3], x_raw: [f64; 3], y_raw: [f64; 3], z_raw: [f64; 3]) -> Self {
        todo!("implement in Phase 4")
    }

    pub fn identity() -> Self {
        todo!("implement in Phase 4")
    }
}

/// Row-major homogeneous 4×4 rigid-body transform.
/// Upper-left 3×3 = rotation R; right column rows 0–2 = translation t.
#[derive(Clone, Debug)]
pub struct Transform {
    pub(crate) m: [[f64; 4]; 4],
}

impl Transform {
    pub fn identity() -> Self {
        todo!("implement in Phase 4")
    }

    pub fn apply(&self, _p: [f64; 3]) -> [f64; 3] {
        todo!("implement in Phase 4")
    }

    pub fn switch_frames(_from: &Frame, _to: &Frame) -> Self {
        todo!("implement in Phase 4")
    }
}

/// Build the backbone coordinate frame from N, CA, C positions (all in lab frame).
pub fn backbone_frame(_n: [f64; 3], _ca: [f64; 3], _c: [f64; 3]) -> Frame {
    todo!("implement in Phase 4")
}
