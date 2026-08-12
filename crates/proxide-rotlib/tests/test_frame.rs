use proxide_rotlib::frame::{backbone_frame, Frame, Transform};

fn assert_near(a: [f64; 3], b: [f64; 3], tol: f64, msg: &str) {
    for i in 0..3 {
        assert!(
            (a[i] - b[i]).abs() < tol,
            "{msg}: dim {i}: {:.10} vs {:.10} (diff {:.2e})",
            a[i],
            b[i],
            (a[i] - b[i]).abs()
        );
    }
}

#[test]
fn test_frame_identity_switch_is_identity() {
    let xform = Transform::switch_frames(&Frame::identity(), &Frame::identity());
    let result = xform.apply([1.0, 2.0, 3.0]);
    assert_near(result, [1.0, 2.0, 3.0], 1e-12, "identity switch");
}

#[test]
fn test_frame_new_normalizes_x() {
    let f = Frame::new(
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    );
    assert_near(f.x, [1.0, 0.0, 0.0], 1e-12, "x normalized");
}

#[test]
fn test_frame_new_normalizes_all_axes() {
    let f = Frame::new(
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 7.0],
    );
    assert_near(f.x, [1.0, 0.0, 0.0], 1e-12, "x");
    assert_near(f.y, [0.0, 1.0, 0.0], 1e-12, "y");
    assert_near(f.z, [0.0, 0.0, 1.0], 1e-12, "z");
}

#[test]
fn test_transform_identity_apply() {
    let result = Transform::identity().apply([3.14, -2.71, 0.0]);
    assert_near(result, [3.14, -2.71, 0.0], 1e-15, "identity apply");
}

#[test]
fn test_switch_frames_known_backbone() {
    // Backbone N=[0,0,0], CA=[1.458,0,0], C=[1.980,1.418,0]
    // x_raw = [1.458,0,0] → x=[1,0,0]
    // z_raw=cross([1.458,0,0],[0.522,1.418,0])=[0,0,2.067] → z=[0,0,1]
    // y_raw=cross(z,x)=[0,1,0]; origin=CA=[1.458,0,0]
    // switch_frames(backbone, identity): t = T2ᵀ·ori = identity·[1.458,0,0] = [1.458,0,0]
    // apply [1,0,0] → R·[1,0,0] + t = [1,0,0] + [1.458,0,0] = [2.458,0,0]
    let n = [0.000, 0.000, 0.000];
    let ca = [1.458, 0.000, 0.000];
    let c = [1.980, 1.418, 0.000];
    let bf = backbone_frame(n, ca, c);
    let xform = Transform::switch_frames(&bf, &Frame::identity());
    let result = xform.apply([1.0, 0.0, 0.0]);
    assert_near(
        result,
        [2.458, 0.0, 0.0],
        1e-9,
        "known backbone switch_frames",
    );
}

#[test]
fn test_switch_frames_roundtrip() {
    // Compose switch_frames(F, I) then switch_frames(I, F) → identity on any point
    let n = [0.000, 0.000, 0.000];
    let ca = [1.458, 0.000, 0.000];
    let c = [1.980, 1.418, 0.000];
    let bf = backbone_frame(n, ca, c);
    let to_lab = Transform::switch_frames(&bf, &Frame::identity());
    let to_bb = Transform::switch_frames(&Frame::identity(), &bf);
    let p = [3.0, 5.0, 7.0];
    let roundtrip = to_bb.apply(to_lab.apply(p));
    assert_near(roundtrip, p, 1e-9, "roundtrip");
}

#[test]
fn test_switch_frames_translated_origin() {
    // from = Frame at [5,0,0] with identity axes; to = Frame::identity()
    // apply [0,0,0] (in from-frame) → should give [5,0,0]
    let from = Frame {
        origin: [5.0, 0.0, 0.0],
        x: [1.0, 0.0, 0.0],
        y: [0.0, 1.0, 0.0],
        z: [0.0, 0.0, 1.0],
    };
    let xform = Transform::switch_frames(&from, &Frame::identity());
    let result = xform.apply([0.0, 0.0, 0.0]);
    assert_near(result, [5.0, 0.0, 0.0], 1e-10, "translated origin");
}
