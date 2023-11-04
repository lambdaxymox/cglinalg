extern crate cglinalg_trigonometry;
extern crate cglinalg_transform;


use cglinalg_trigonometry::{
    Degrees,
};
use cglinalg_core::{
    Matrix4x4,
    Point3,
    Vector3,
};
use cglinalg_transform::{
    Orthographic3,
    Perspective3,
    PerspectiveFov3,
};
use approx::{
    assert_relative_eq,
};


#[rustfmt::skip]
#[test]
fn test_perspective_projection_matrix() {
    let left = -4_f64;
    let right = 4_f64;
    let bottom = -2_f64;
    let top = 3_f64;
    let near = 1_f64;
    let far = 100_f64;
    let expected = Matrix4x4::new(
        1_f64 / 4_f64, 0_f64,          0_f64,             0_f64,
        0_f64,         2_f64 / 5_f64,  0_f64,             0_f64,
        0_f64,         1_f64 / 5_f64, -101_f64 / 99_f64, -1_f64,
        0_f64,         0_f64,         -200_f64 / 99_f64,  0_f64,
    );
    let result = Matrix4x4::from_perspective(left, right, bottom, top, near, far);

    assert_eq!(result, expected);
}

#[rustfmt::skip]
#[test]
fn test_perspective_projection_transformation() {
    let left = -4_f64;
    let right = 4_f64;
    let bottom = -2_f64;
    let top = 3_f64;
    let near = 1_f64;
    let far = 100_f64;
    let expected = Matrix4x4::new(
        1_f64 / 4_f64, 0_f64,          0_f64,             0_f64,
        0_f64,         2_f64 / 5_f64,  0_f64,             0_f64,
        0_f64,         1_f64 / 5_f64, -101_f64 / 99_f64, -1_f64,
        0_f64,         0_f64,         -200_f64 / 99_f64,  0_f64,
    );
    let perspective = Perspective3::new(left, right, bottom, top, near, far);
    let result = perspective.matrix();

    assert_eq!(result, &expected);
}

#[rustfmt::skip]
#[test]
fn test_perspective_projection_rectangular_parameters() {
    let left = -4_f64;
    let right = 4_f64;
    let bottom = -2_f64;
    let top = 3_f64;
    let near = 1_f64;
    let far = 100_f64;
    let perspective = Perspective3::new(left, right, bottom, top, near, far);

    assert_relative_eq!(perspective.left(),   left,   epsilon = 1e-10);
    assert_relative_eq!(perspective.right(),  right,  epsilon = 1e-10);
    assert_relative_eq!(perspective.bottom(), bottom, epsilon = 1e-10);
    assert_relative_eq!(perspective.top(),    top,    epsilon = 1e-10);
    assert_relative_eq!(perspective.near(),   near,   epsilon = 1e-10);
    assert_relative_eq!(perspective.far(),    far,    epsilon = 1e-10);
}

#[rustfmt::skip]
#[test]
fn test_perspective_projection_fov_matrix() {
    let vfov = Degrees(72_f64);
    let aspect_ratio = 800_f64 / 600_f64;
    let near = 0.1_f64;
    let far = 100_f64;
    let tan_vfov_over_two = f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64));
    let c0r0 = 3_f64 / (4_f64 * tan_vfov_over_two);
    let c1r1 = 1_f64 / tan_vfov_over_two;
    let c2r2 = -100.1_f64 / 99.9_f64;
    let c3r2 = -2_f64 * (100_f64 * (1_f64 / 10_f64)) / (100_f64 - (1_f64 / 10_f64));
    let expected = Matrix4x4::new(
        c0r0,  0_f64, 0_f64,  0_f64,
        0_f64, c1r1,  0_f64,  0_f64,
        0_f64, 0_f64, c2r2,  -1_f64,
        0_f64, 0_f64, c3r2,   0_f64,
    );
    let result = Matrix4x4::from_perspective_fov(vfov, aspect_ratio, near, far);

    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

#[rustfmt::skip]
#[test]
fn test_perspective_projection_fov_transformation() {
    let vfov = Degrees(72_f64);
    let aspect_ratio = 800_f64 / 600_f64;
    let near = 0.1_f64;
    let far = 100_f64;
    let tan_vfov_over_two = f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64));
    let c0r0 = 3_f64 / (4_f64 * tan_vfov_over_two);
    let c1r1 = 1_f64 / tan_vfov_over_two;
    let c2r2 = -100.1_f64 / 99.9_f64;
    let c3r2 = -2_f64 * (100_f64 * (1_f64 / 10_f64)) / (100_f64 - (1_f64 / 10_f64));
    let expected = Matrix4x4::new(
        c0r0,  0_f64, 0_f64,  0_f64,
        0_f64, c1r1,  0_f64,  0_f64,
        0_f64, 0_f64, c2r2,  -1_f64,
        0_f64, 0_f64, c3r2,   0_f64,
    );
    let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    let result = perspective.matrix();

    assert_relative_eq!(result, &expected, epsilon = 1e-10);
}

#[rustfmt::skip]
#[test]
fn test_perspective_projection_fov_rectangular_parameters() {
    let vfov = Degrees(72_f64);
    let aspect_ratio = 800_f64 / 600_f64;
    let near = 0.1_f64;
    let far = 100_f64;
    let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    let expected_left = -(1_f64 / 10_f64) * (4_f64 / 3_f64) * f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64));
    let expected_right = (1_f64 / 10_f64) * (4_f64 / 3_f64) * f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64));
    let expected_bottom = -(1_f64 / 10_f64) * (f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64)));
    let expected_top = (1_f64 / 10_f64) * (f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64)));

    assert_relative_eq!(perspective.left(),   expected_left,   epsilon = 1e-10);
    assert_relative_eq!(perspective.right(),  expected_right,  epsilon = 1e-10);
    assert_relative_eq!(perspective.bottom(), expected_bottom, epsilon = 1e-10);
    assert_relative_eq!(perspective.top(),    expected_top,    epsilon = 1e-10);
    assert_relative_eq!(perspective.near(),   near,            epsilon = 1e-10);
    assert_relative_eq!(perspective.far(),    far,             epsilon = 1e-10);
}

#[rustfmt::skip]
#[test]
fn test_perspective_projection_fov_fov_parameters() {
    let vfov = Degrees(72_f64);
    let aspect_ratio = 800_f64 / 600_f64;
    let near = 0.1_f64;
    let far = 100_f64;
    let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    let expected_vfov = vfov.into();

    assert_relative_eq!(perspective.vfov(),         expected_vfov, epsilon = 1e-10);
    assert_relative_eq!(perspective.aspect_ratio(), aspect_ratio,  epsilon = 1e-10);
    assert_relative_eq!(perspective.near(),         near,          epsilon = 1e-10);
    assert_relative_eq!(perspective.far(),          far,           epsilon = 1e-10);
}

#[test]
fn test_perspective_projection_fov_unproject_point() {
    let vfov = Degrees(72_f64);
    let aspect_ratio = 800_f64 / 600_f64;
    let near = 0.1_f64;
    let far = 100_f64;
    let point = Point3::new(-2_f64, 2_f64, -50_f64);
    let projection = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    let expected = point;
    let projected_point = projection.project_point(&expected);
    let result = projection.unproject_point(&projected_point);

    assert_relative_eq!(result, expected, epsilon = 1e-8);
}

#[test]
fn test_perspective_projection_fov_unproject_vector() {
    let vfov = Degrees(72_f64);
    let aspect_ratio = 800_f64 / 600_f64;
    let near = 0.1_f64;
    let far = 100_f64;
    let vector = Vector3::new(-2_f64, 2_f64, -50_f64);
    let projection = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    let expected = vector;
    let projected_vector = projection.project_vector(&expected);
    let result = projection.unproject_vector(&projected_vector);

    assert_relative_eq!(result, expected, epsilon = 1e-8);
}

#[test]
fn test_perspective_projection_unproject_point() {
    let left = -4_f64;
    let right = 4_f64;
    let bottom = -2_f64;
    let top = 2_f64;
    let near = 1_f64;
    let far = 100_f64;
    let projection = Perspective3::new(left, right, bottom, top, near, far);
    let expected = Point3::new(-2_f64, 2_f64, -50_f64);
    let projected_point = projection.project_point(&expected);
    let result = projection.unproject_point(&projected_point);

    assert_relative_eq!(result, expected, epsilon = 1e-8);
}

#[test]
fn test_perspective_projection_unproject_vector() {
    let left = -4_f64;
    let right = 4_f64;
    let bottom = -2_f64;
    let top = 2_f64;
    let near = 1_f64;
    let far = 100_f64;
    let projection = Perspective3::new(left, right, bottom, top, near, far);
    let expected = Vector3::new(-2_f64, 2_f64, -50_f64);
    let projected_vector = projection.project_vector(&expected);
    let result = projection.unproject_vector(&projected_vector);

    assert_relative_eq!(result, expected, epsilon = 1e-8);
}

#[rustfmt::skip]
#[test]
fn test_orthographic_projection_matrix() {
    let left = -4_f64;
    let right = 4_f64;
    let bottom = -2_f64;
    let top = 2_f64;
    let near = 1_f64;
    let far = 100_f64;
    let expected = Matrix4x4::new(
        1_f64 / 4_f64, 0_f64,          0_f64,            0_f64,
        0_f64,         1_f64 / 2_f64,  0_f64,            0_f64,
        0_f64,         0_f64,         -2_f64 / 99_f64,   0_f64,
        0_f64,         0_f64,         -101_f64 / 99_f64, 1_f64,
    );
    let result = Matrix4x4::from_orthographic(left, right, bottom, top, near, far);

    assert_eq!(result, expected);
}

#[rustfmt::skip]
#[test]
fn test_orthographic_projection_transformation() {
    let left = -4_f64;
    let right = 4_f64;
    let bottom = -2_f64;
    let top = 2_f64;
    let near = 1_f64;
    let far = 100_f64;
    let expected = Matrix4x4::new(
        1_f64 / 4_f64, 0_f64,          0_f64,            0_f64,
        0_f64,         1_f64 / 2_f64,  0_f64,            0_f64,
        0_f64,         0_f64,         -2_f64 / 99_f64,   0_f64,
        0_f64,         0_f64,         -101_f64 / 99_f64, 1_f64,
    );
    let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    let result = orthographic.matrix();

    assert_eq!(result, &expected);
}

#[rustfmt::skip]
#[test]
fn test_orthographic_projection_rectangular_parameters() {
    let left = -4_f64;
    let right = 4_f64;
    let bottom = -2_f64;
    let top = 2_f64;
    let near = 1_f64;
    let far = 100_f64;
    let orthographic = Orthographic3::new(left, right, bottom, top, near, far);

    assert_relative_eq!(orthographic.left(),   left,   epsilon = 1e-10);
    assert_relative_eq!(orthographic.right(),  right,  epsilon = 1e-10);
    assert_relative_eq!(orthographic.bottom(), bottom, epsilon = 1e-10);
    assert_relative_eq!(orthographic.top(),    top,    epsilon = 1e-10);
    assert_relative_eq!(orthographic.near(),   near,   epsilon = 1e-10);
    assert_relative_eq!(orthographic.far(),    far,    epsilon = 1e-10);
}

#[test]
fn test_orthographic_projection_unproject_point() {
    let left = -4_f64;
    let right = 4_f64;
    let bottom = -2_f64;
    let top = 2_f64;
    let near = 1_f64;
    let far = 100_f64;
    let projection = Orthographic3::new(left, right, bottom, top, near, far);
    let expected = Point3::new(1_f64, 1_f64, 50_f64);
    let projected_point = projection.project_point(&expected);
    let result = projection.unproject_point(&projected_point);

    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

#[test]
fn test_orthographic_projection_unproject_vector() {
    let left = -4_f64;
    let right = 4_f64;
    let bottom = -2_f64;
    let top = 2_f64;
    let near = 1_f64;
    let far = 100_f64;
    let projection = Orthographic3::new(left, right, bottom, top, near, far);
    let expected = Vector3::new(1_f64, 1_f64, 50_f64);
    let projected_vector = projection.project_vector(&expected);
    let result = projection.unproject_vector(&projected_vector);

    assert_relative_eq!(result, expected, epsilon = 1e-10);
}
