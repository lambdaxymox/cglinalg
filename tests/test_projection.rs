extern crate cglinalg;


use cglinalg::cglinalg_core::{
    Matrix4x4,
    Angle,
    Degrees,
    Radians,
    Point3,
    Vector3,
};
use cglinalg::{
    Orthographic3,
    OrthographicFov3,
    Perspective3,
    PerspectiveFov3,
};
use approx::{
    assert_relative_eq,
};


#[test]
fn test_perspective_projection_matrix() {
    let left = -4.0;
    let right = 4.0;
    let bottom = -2.0;
    let top = 3.0;
    let near = 1.0;
    let far = 100.0;
    let expected = Matrix4x4::new(
        1.0 / 4.0,  0.0,        0.0,           0.0,
        0.0,        2.0 / 5.0,  0.0,           0.0,
        0.0,        1.0 / 5.0, -101.0 / 99.0, -1.0,
        0.0,        0.0,       -200.0 / 99.0,  0.0
    );
    let result = Matrix4x4::from_perspective(left, right, bottom, top, near, far);

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
    let expected = Matrix4x4::new(
        1.0 / 4.0,  0.0,        0.0,           0.0,
        0.0,        2.0 / 5.0,  0.0,           0.0,
        0.0,        1.0 / 5.0, -101.0 / 99.0, -1.0,
        0.0,        0.0,       -200.0 / 99.0,  0.0
    );
    let result = Perspective3::new(left, right, bottom, top, near, far);

    assert_eq!(result.matrix(), &expected);
}

#[test]
fn test_perspective_projection_fov_matrix() {
    let vfov = Degrees(72.0);
    let aspect = 800 as f32 / 600 as f32;
    let near = 0.1;
    let far = 100.0;
    let expected = Matrix4x4::new(
        1.0322863, 0.0,        0.0,       0.0, 
        0.0,       1.3763818,  0.0,       0.0, 
        0.0,       0.0,       -1.002002, -1.0, 
        0.0,       0.0,       -0.2002002, 0.0
    );
    let result = Matrix4x4::from_perspective_fov(vfov, aspect, near, far);

    assert_relative_eq!(result, expected);
}

#[test]
fn test_perspective_projection_fov_transformation() {
    let vfov = Degrees(72.0);
    let aspect = 800 as f32 / 600 as f32;
    let near = 0.1;
    let far = 100.0;
    let expected = Matrix4x4::new(
        1.0322863, 0.0,        0.0,       0.0, 
        0.0,       1.3763818,  0.0,       0.0, 
        0.0,       0.0,       -1.002002, -1.0, 
        0.0,       0.0,       -0.2002002, 0.0
    );
    let result = PerspectiveFov3::new(vfov, aspect, near, far);

    assert_relative_eq!(result.matrix(), &expected, epsilon = 1e-10);
}

#[test]
fn test_perspective_projection_unproject_point1() {
    let vfov = Degrees(72.0);
    let aspect = 800 as f64 / 600 as f64;
    let near = 0.1;
    let far = 100.0;
    let point = Point3::new(-2.0, 2.0, -50.0);
    let projection = PerspectiveFov3::new(vfov, aspect, near, far);
    let expected = point;
    let projected_point = projection.project_point(&expected);
    let result = projection.unproject_point(&projected_point);

    assert_relative_eq!(result, expected, epsilon = 1e-8);
}

#[test]
fn test_perspective_projection_unproject_vector1() {
    let vfov = Degrees(72.0);
    let aspect = 800 as f64 / 600 as f64;
    let near = 0.1;
    let far = 100.0;
    let vector = Vector3::new(-2.0, 2.0, -50.0);
    let projection = PerspectiveFov3::new(vfov, aspect, near, far);
    let expected = vector;
    let projected_vector = projection.project_vector(&expected);
    let result = projection.unproject_vector(&projected_vector);

    assert_relative_eq!(result, expected, epsilon = 1e-8);
}

#[test]
fn test_perspective_projection_unproject_point2() {
    let left = -4.0;
    let right = 4.0;
    let bottom = -2.0;
    let top = 2.0;
    let near = 1.0;
    let far = 100.0;
    let projection = Perspective3::new(left, right, bottom, top, near, far);
    let expected = Point3::new(-2.0, 2.0, -50.0);
    let projected_point = projection.project_point(&expected);
    let result = projection.unproject_point(&projected_point);

    assert_relative_eq!(result, expected, epsilon = 1e-8);
}

#[test]
fn test_perspective_projection_unproject_vector2() {
    let left = -4.0;
    let right = 4.0;
    let bottom = -2.0;
    let top = 2.0;
    let near = 1.0;
    let far = 100.0;
    let projection = Perspective3::new(left, right, bottom, top, near, far);
    let expected = Vector3::new(-2.0, 2.0, -50.0);
    let projected_vector = projection.project_vector(&expected);
    let result = projection.unproject_vector(&projected_vector);

    assert_relative_eq!(result, expected, epsilon = 1e-8);
}

#[test]
fn test_orthographic_projection_matrix() {
    let left = -4.0;
    let right = 4.0;
    let bottom = -2.0;
    let top = 2.0;
    let near = 1.0;
    let far = 100.0;
    let expected = Matrix4x4::new(
        1.0 / 4.0,  0.0,        0.0,          0.0,
        0.0,        1.0 / 2.0,  0.0,          0.0,
        0.0,        0.0,       -2.0 / 99.0,   0.0,
        0.0,        0.0,       -101.0 / 99.0, 1.0
    );
    let result = Matrix4x4::from_orthographic(left, right, bottom, top, near, far);

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
    let expected = Matrix4x4::new(
        1.0 / 4.0,  0.0,        0.0,          0.0,
        0.0,        1.0 / 2.0,  0.0,          0.0,
        0.0,        0.0,       -2.0 / 99.0,   0.0,
        0.0,        0.0,       -101.0 / 99.0, 1.0
    );
    let result = Orthographic3::new(left, right, bottom, top, near, far);

    assert_eq!(result.matrix(), &expected);
}

#[test]
fn test_orthographic_projection_unproject_point() {
    let left = -4.0;
    let right = 4.0;
    let bottom = -2.0;
    let top = 2.0;
    let near = 1.0;
    let far = 100.0;
    let projection = Orthographic3::new(left, right, bottom, top, near, far);
    let expected = Point3::new(1.0, 1.0, 50.0);
    let projected_point = projection.project_point(&expected);
    let result = projection.unproject_point(&projected_point);

    assert_eq!(result, expected);
}

#[test]
fn test_orthographic_projection_unproject_vector() {
    let left = -4.0;
    let right = 4.0;
    let bottom = -2.0;
    let top = 2.0;
    let near = 1.0;
    let far = 100.0;
    let projection = Orthographic3::new(left, right, bottom, top, near, far);
    let expected = Vector3::new(1.0, 1.0, 50.0);
    let projected_vector = projection.project_vector(&expected);
    let result = projection.unproject_vector(&projected_vector);

    assert_eq!(result, expected);
}

#[test]
fn test_orthographic_fov_projection_matrix() {
    let aspect = 2.0;
    // 9.1478425198 Degrees.
    let vfov = Degrees::from(Radians::atan2(8.0, 100.0) * 2.0);
    let near = 1.0;
    let far = 100.0;
    let expected = Matrix4x4::new(
        1.0 / 4.0,  0.0,        0.0,          0.0,
        0.0,        1.0 / 2.0,  0.0,          0.0,
        0.0,        0.0,       -2.0 / 99.0,   0.0,
        0.0,        0.0,       -101.0 / 99.0, 1.0
    );
    let result = Matrix4x4::from_orthographic_fov(vfov, aspect, near, far);

    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

#[test]
fn test_orthographic_fov_projecton_transformation() {
    let aspect = 2.0;
    // 9.1478425198 Degrees.
    let vfov = Degrees::from(Radians::atan2(8.0, 100.0) * 2.0);
    let near = 1.0;
    let far = 100.0;
    let expected = Matrix4x4::new(
        1.0 / 4.0,  0.0,        0.0,          0.0,
        0.0,        1.0 / 2.0,  0.0,          0.0,
        0.0,        0.0,       -2.0 / 99.0,   0.0,
        0.0,        0.0,       -101.0 / 99.0, 1.0
    );
    let result = OrthographicFov3::new(vfov, aspect, near, far);

    assert_relative_eq!(result.matrix(), &expected, epsilon = 1e-10);
}

#[test]
fn test_orthographic_fov_projection_unproject_point() {
    let aspect = 2.0;
    // 9.1478425198 Degrees.
    let vfov = Degrees::from(Radians::atan2(8.0, 100.0) * 2.0);
    let near = 1.0;
    let far = 100.0;
    let projection = OrthographicFov3::new(vfov, aspect, near, far);
    let expected = Point3::new(1.0, 1.0, 50.0);
    let projected_point = projection.project_point(&expected);
    let result = projection.unproject_point(&projected_point);

    assert_eq!(result, expected);
}

#[test]
fn test_orthographic_fov_projection_unproject_vector() {
    let aspect = 2.0;
    // 9.1478425198 Degrees.
    let vfov = Degrees::from(Radians::atan2(8.0, 100.0) * 2.0);
    let near = 1.0;
    let far = 100.0;
    let projection = OrthographicFov3::new(vfov, aspect, near, far);
    let expected = Vector3::new(1.0, 1.0, 50.0);
    let projected_vector = projection.project_vector(&expected);
    let result = projection.unproject_vector(&projected_vector);

    assert_eq!(result, expected);
}

