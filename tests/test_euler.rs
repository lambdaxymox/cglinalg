extern crate cglinalg;


use cglinalg::{
    EulerAngles,
    Radians,
    Angle,
    Matrix3x3,
    Matrix4x4,
    AdditiveIdentity,
    Identity,
};
use cglinalg::approx::{
    relative_eq,
};


/// Test with the following Euler angles:
/// ```text
/// roll_yz = pi / 2
/// yaw_zx = pi / 4
/// pitch_xy = pi / 4
/// ```
#[test]
fn test_to_matrix() {
    let roll_yz = Radians::full_turn_div_4();
    let yaw_zx = Radians::full_turn_div_8();
    let pitch_xy = Radians::full_turn_div_8();
    let euler = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    
    let c0r0 =  1.0 / 2.0;
    let c0r1 =  1.0 / 2.0;
    let c0r2 =  1.0 / f64::sqrt(2.0);

    let c1r0 = -1.0 / 2.0;
    let c1r1 = -1.0 / 2.0;
    let c1r2 =  1.0 / f64::sqrt(2.0);

    let c2r0 =  1.0 / f64::sqrt(2.0);
    let c2r1 = -1.0 / f64::sqrt(2.0);
    let c2r2 =  0.0;

    let expected = Matrix3x3::new(
        c0r0, c0r1, c0r2,
        c1r0, c1r1, c1r2,
        c2r0, c2r1, c2r2
    );
    let result = euler.to_matrix();

    assert!(relative_eq!(result, expected, epsilon = 1e-8));
}

/// An Euler rotation that's all zeros should be the identity.
#[test]
fn test_to_matrix_identity() {
    let euler: EulerAngles<Radians<f64>> = EulerAngles::zero();
    let expected:Matrix4x4<f64> = Matrix4x4::identity();
    let result = euler.to_affine_matrix();

    assert_eq!(result, expected);
}

/// Test with the following Euler angles:
/// ```text
/// roll_yz = pi / 2
/// yaw_zx = pi / 4
/// pitch_xy = pi / 4
/// ```
#[test]
fn test_to_affine_matrix() {
    let roll_yz = Radians::full_turn_div_4();
    let yaw_zx = Radians::full_turn_div_8();
    let pitch_xy = Radians::full_turn_div_8();
    let euler = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    
    let c0r0 =  1.0 / 2.0;
    let c0r1 =  1.0 / 2.0;
    let c0r2 =  1.0 / f64::sqrt(2.0);
    let c0r3 =  0.0;

    let c1r0 = -1.0 / 2.0;
    let c1r1 = -1.0 / 2.0;
    let c1r2 =  1.0 / f64::sqrt(2.0);
    let c1r3 =  0.0;

    let c2r0 =  1.0 / f64::sqrt(2.0);
    let c2r1 = -1.0 / f64::sqrt(2.0);
    let c2r2 =  0.0;
    let c2r3 = 0.0;

    let c3r0 = 0.0;
    let c3r1 = 0.0;
    let c3r2 = 0.0;
    let c3r3 = 1.0;

    let expected = Matrix4x4::new(
        c0r0, c0r1, c0r2, c0r3,
        c1r0, c1r1, c1r2, c1r3,
        c2r0, c2r1, c2r2, c2r3,
        c3r0, c3r1, c3r2, c3r3
    );
    let result = euler.to_affine_matrix();

    assert!(relative_eq!(result, expected, epsilon = 1e-8));  
}

/// An Euler rotation that's all zeros should be the identity.
#[test]
fn test_to_affine_matrix_identity() {
    let euler: EulerAngles<Radians<f64>> = EulerAngles::zero();
    let expected:Matrix4x4<f64> = Matrix4x4::identity();
    let result = euler.to_affine_matrix();

    assert_eq!(result, expected);
}

/// A set of Euler angles with only one nonzero entry should act like the 
/// corresponding rotation matrix.
#[test]
fn test_euler_rotation_roll_yz() {
    let roll_yz: Radians<f64> = Radians::full_turn_div_6();
    let yaw_zx: Radians<f64> = Radians::zero();
    let pitch_xy: Radians<f64> = Radians::zero();
    let euler = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    let expected = Matrix3x3::from_angle_x(roll_yz);
    let result = euler.to_matrix();
    
    assert_eq!(result, expected);
}

/// A set of Euler angles with only one nonzero entry should act like the 
/// corresponding rotation matrix.
#[test]
fn test_euler_rotation_yaw_zx() {
    let roll_yz: Radians<f64> = Radians::zero();
    let yaw_zx: Radians<f64> = Radians::full_turn_div_6();
    let pitch_xy: Radians<f64> = Radians::zero();
    let euler = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    let expected = Matrix3x3::from_angle_y(yaw_zx);
    let result = euler.to_matrix();

    assert_eq!(result, expected);
}

/// A set of Euler angles with only one nonzero entry should act like the 
/// corresponding rotation matrix.
#[test]
fn test_euler_rotation_pitch_xy() {
    let roll_yz: Radians<f64> = Radians::zero();
    let yaw_zx: Radians<f64> = Radians::zero();
    let pitch_xy: Radians<f64> = Radians::full_turn_div_6();
    let euler = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    let expected = Matrix3x3::from_angle_z(pitch_xy);
    let result = euler.to_matrix();

    assert_eq!(result, expected);
}

#[test]
fn test_euler_angles_from_matrix_roll_yz() {
    let roll_yz: Radians<f64> = Radians::full_turn_div_6();
    let yaw_zx: Radians<f64> = Radians::zero();
    let pitch_xy: Radians<f64> = Radians::zero();
    let matrix = Matrix3x3::from_angle_x(roll_yz);
    let expected = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    let result = EulerAngles::from_matrix(&matrix);

    assert_eq!(result, expected);
}

#[test]
fn test_euler_angles_from_matrix_yaw_zx() {
    let roll_yz: Radians<f64> = Radians::zero();
    let yaw_zx: Radians<f64> = Radians::full_turn_div_6();
    let pitch_xy: Radians<f64> = Radians::zero();
    let matrix = Matrix3x3::from_angle_y(yaw_zx);
    let expected = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    let result = EulerAngles::from_matrix(&matrix);

    assert_eq!(result, expected);
}

#[test]
fn test_euler_angles_from_matrix_pitch_xy() {
    let roll_yz: Radians<f64> = Radians::zero();
    let yaw_zx: Radians<f64> = Radians::zero();
    let pitch_xy: Radians<f64> = Radians::full_turn_div_6();
    let matrix = Matrix3x3::from_angle_z(pitch_xy);
    let expected = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    let result = EulerAngles::from_matrix(&matrix);

    assert_eq!(result, expected);
}

#[test]
fn test_euler_angles_from_matrix_rotation_matrix1() {
    let roll_yz: Radians<f64> = Radians::full_turn_div_2();
    let yaw_zx: Radians<f64> = Radians::full_turn_div_8();
    let pitch_xy: Radians<f64> = Radians::full_turn_div_6();
    let matrix_yz = Matrix3x3::from_angle_x(roll_yz);
    let matrix_zx = Matrix3x3::from_angle_y(yaw_zx);
    let matrix_xy = Matrix3x3::from_angle_z(pitch_xy);
    let matrix = matrix_yz * matrix_zx * matrix_xy;
    let expected = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    let result = EulerAngles::from_matrix(&matrix);

    assert_eq!(result, expected);
}

#[test]
fn test_euler_angles_from_matrix_rotation_matrix2() {
    let roll_yz: Radians<f64> = Radians::full_turn_div_2();
    let yaw_zx: Radians<f64> = Radians::full_turn_div_4();
    let pitch_xy: Radians<f64> = Radians::full_turn_div_6();
    let matrix_yz = Matrix3x3::from_angle_x(roll_yz);
    let matrix_zx = Matrix3x3::from_angle_y(yaw_zx);
    let matrix_xy = Matrix3x3::from_angle_z(pitch_xy);
    let matrix = matrix_yz * matrix_zx * matrix_xy;
    let expected = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    let result = EulerAngles::from_matrix(&matrix);

    assert_eq!(result, expected);
}

