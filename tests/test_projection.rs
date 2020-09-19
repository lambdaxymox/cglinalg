pub extern crate cglinalg;


use cglinalg::{
    OrthographicSpec,
    OrthographicProjection3D,
    PerspectiveFovSpec,
    PerspectiveSpec,
    PerspectiveProjection3D,
    Matrix4x4,
};

    
#[test]
fn test_perspective_projection_matrix() {
    let left = -4.0;
    let right = 4.0;
    let bottom = -2.0;
    let top = 3.0;
    let near = 1.0;
    let far = 100.0;
    let spec = PerspectiveSpec::new(left, right, bottom, top, near, far);
    let expected = Matrix4x4::new(
        1.0 / 4.0,  0.0,        0.0,           0.0,
        0.0,        2.0 / 5.0,  0.0,           0.0,
        0.0,        1.0 / 5.0, -101.0 / 99.0, -1.0,
        0.0,        0.0,       -200.0 / 99.0,  0.0
    );
    let result = Matrix4x4::from(spec);

    assert_eq!(result, expected);
}

#[test]
fn test_perspective_projection_transformation() {
    let left = -4.0;
    let right = 4.0;
    let bottom = -2.0;
    let top = 3.0;
    let near = 1.0;
    let far = 100.0;
    let spec = PerspectiveSpec::new(left, right, bottom, top, near, far);
    let expected = Matrix4x4::new(
        1.0 / 4.0,  0.0,        0.0,           0.0,
        0.0,        2.0 / 5.0,  0.0,           0.0,
        0.0,        1.0 / 5.0, -101.0 / 99.0, -1.0,
        0.0,        0.0,       -200.0 / 99.0,  0.0
    );
    let result = PerspectiveProjection3D::new(spec);

    assert_eq!(result.to_matrix(), &expected);
}

#[test]
fn test_orthographic_projection_matrix() {
    let left = -4.0;
    let right = 4.0;
    let bottom = -2.0;
    let top = 2.0;
    let near = 1.0;
    let far = 100.0;
    let spec = OrthographicSpec::new(left, right, bottom, top, near, far);
    let expected = Matrix4x4::new(
        1.0 / 4.0,  0.0,        0.0,          0.0,
        0.0,        1.0 / 2.0,  0.0,          0.0,
        0.0,        0.0,       -2.0 / 99.0,   0.0,
        0.0,        0.0,       -101.0 / 99.0, 1.0
    );
    let result = Matrix4x4::from(spec);

    assert_eq!(result, expected);
}

#[test]
fn test_orthographic_projection_transformation() {
    let left = -4.0;
    let right = 4.0;
    let bottom = -2.0;
    let top = 2.0;
    let near = 1.0;
    let far = 100.0;
    let spec = OrthographicSpec::new(left, right, bottom, top, near, far);
    let expected = Matrix4x4::new(
        1.0 / 4.0,  0.0,        0.0,          0.0,
        0.0,        1.0 / 2.0,  0.0,          0.0,
        0.0,        0.0,       -2.0 / 99.0,   0.0,
        0.0,        0.0,       -101.0 / 99.0, 1.0
    );
    let result = OrthographicProjection3D::new(spec);

    assert_eq!(result.to_matrix(), &expected);
}
