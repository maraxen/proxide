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
        fn normalize(v: [f64; 3]) -> [f64; 3] {
            let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
            [v[0]/len, v[1]/len, v[2]/len]
        }
        Self {
            origin,
            x: normalize(x_raw),
            y: normalize(y_raw),
            z: normalize(z_raw),
        }
    }

    pub fn identity() -> Self {
        Self {
            origin: [0.0, 0.0, 0.0],
            x: [1.0, 0.0, 0.0],
            y: [0.0, 1.0, 0.0],
            z: [0.0, 0.0, 1.0],
        }
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
        Self { m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]}
    }

    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        [0, 1, 2].map(|i|
            self.m[i][0]*p[0] + self.m[i][1]*p[1] + self.m[i][2]*p[2] + self.m[i][3])
    }

    /// Build the transform that maps points from `from`-frame to `to`-frame.
    /// Algorithm (MSL TransformFactory::switchFrames, msttransforms.cpp:530–546):
    ///   T1 = 3×3 with columns [from.x | from.y | from.z]
    ///   T2 = 3×3 with columns [to.x   | to.y   | to.z  ]
    ///   For orthonormal T2: T2⁻¹ = T2ᵀ
    ///   R = T2ᵀ · T1
    ///   t = T2ᵀ · (from.origin - to.origin)
    pub fn switch_frames(from: &Frame, to: &Frame) -> Self {
        // T1[row][col] where cols are from.x, from.y, from.z
        let t1 = [
            [from.x[0], from.y[0], from.z[0]],
            [from.x[1], from.y[1], from.z[1]],
            [from.x[2], from.y[2], from.z[2]],
        ];
        // T2ᵀ[row][col] = T2[col][row]; T2 cols are to.x, to.y, to.z
        // T2ᵀ rows are to.x, to.y, to.z
        let t2t = [
            [to.x[0], to.x[1], to.x[2]],
            [to.y[0], to.y[1], to.y[2]],
            [to.z[0], to.z[1], to.z[2]],
        ];
        // R = T2ᵀ · T1 (3×3 matrix multiply)
        let mut r = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    r[i][j] += t2t[i][k] * t1[k][j];
                }
            }
        }
        // ori = from.origin - to.origin
        let ori = [
            from.origin[0] - to.origin[0],
            from.origin[1] - to.origin[1],
            from.origin[2] - to.origin[2],
        ];
        // t = T2ᵀ · ori
        let t = [
            t2t[0][0]*ori[0] + t2t[0][1]*ori[1] + t2t[0][2]*ori[2],
            t2t[1][0]*ori[0] + t2t[1][1]*ori[1] + t2t[1][2]*ori[2],
            t2t[2][0]*ori[0] + t2t[2][1]*ori[1] + t2t[2][2]*ori[2],
        ];
        // Assemble 4×4
        Self { m: [
            [r[0][0], r[0][1], r[0][2], t[0]],
            [r[1][0], r[1][1], r[1][2], t[1]],
            [r[2][0], r[2][1], r[2][2], t[2]],
            [0.0,     0.0,     0.0,     1.0 ],
        ]}
    }
}

/// Construct the backbone coordinate frame from N, CA, C positions (lab frame).
/// Matches MSL placeRotamer lines 182–194.
///   x = CA − N  (N→CA direction)
///   z = x × (C − CA)
///   y = z × x
///   origin = CA
pub fn backbone_frame(n: [f64; 3], ca: [f64; 3], c: [f64; 3]) -> Frame {
    let sub = |a: [f64;3], b: [f64;3]| [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
    let cross = |a: [f64;3], b: [f64;3]| [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ];
    let x_raw = sub(ca, n);
    let z_raw = cross(x_raw, sub(c, ca));
    let y_raw = cross(z_raw, x_raw);
    Frame::new(ca, x_raw, y_raw, z_raw)
}
