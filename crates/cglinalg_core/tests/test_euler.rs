extern crate cglinalg_trigonometry;
extern crate cglinalg_core;


use cglinalg_trigonometry::{
    Angle,
    Radians,
};
use cglinalg_core::{
    EulerAngles,
    Matrix3x3,
    Matrix4x4,
};
use approx::{
    assert_relative_eq,
};


#[rustfmt::skip]
#[test]
fn test_to_matrix() {
    let roll_yz = Radians::full_turn_div_4();
    let yaw_zx = Radians::full_turn_div_8();
    let pitch_xy = Radians::full_turn_div_8();
    let euler = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    
    let c0r0 =  1_f64 / 2_f64;
    let c0r1 =  1_f64 / 2_f64;
    let c0r2 =  1_f64 / f64::sqrt(2_f64);

    let c1r0 = -1_f64 / 2_f64;
    let c1r1 = -1_f64 / 2_f64;
    let c1r2 =  1_f64 / f64::sqrt(2_f64);

    let c2r0 =  1_f64 / f64::sqrt(2_f64);
    let c2r1 = -1_f64 / f64::sqrt(2_f64);
    let c2r2 =  0_f64;

    let expected = Matrix3x3::new(
        c0r0, c0r1, c0r2,
        c1r0, c1r1, c1r2,
        c2r0, c2r1, c2r2
    );
    let result = euler.to_matrix();

    assert_relative_eq!(result, expected, epsilon = 1e-8);
}


#[test]
fn test_to_matrix_zero_euler_angles_is_identity() {
    let euler: EulerAngles<Radians<f64>> = EulerAngles::zero();
    let expected:Matrix4x4<f64> = Matrix4x4::identity();
    let result = euler.to_affine_matrix();

    assert_eq!(result, expected);
}


#[rustfmt::skip]
#[test]
fn test_to_affine_matrix() {
    let roll_yz = Radians::full_turn_div_4();
    let yaw_zx = Radians::full_turn_div_8();
    let pitch_xy = Radians::full_turn_div_8();
    let euler = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    
    let c0r0 =  1_f64 / 2_f64;
    let c0r1 =  1_f64 / 2_f64;
    let c0r2 =  1_f64 / f64::sqrt(2_f64);
    let c0r3 =  0_f64;

    let c1r0 = -1_f64 / 2_f64;
    let c1r1 = -1_f64 / 2_f64;
    let c1r2 =  1_f64 / f64::sqrt(2_f64);
    let c1r3 =  0_f64;

    let c2r0 =  1_f64 / f64::sqrt(2_f64);
    let c2r1 = -1_f64 / f64::sqrt(2_f64);
    let c2r2 =  0_f64;
    let c2r3 = 0_f64;

    let c3r0 = 0_f64;
    let c3r1 = 0_f64;
    let c3r2 = 0_f64;
    let c3r3 = 1_f64;

    let expected = Matrix4x4::new(
        c0r0, c0r1, c0r2, c0r3,
        c1r0, c1r1, c1r2, c1r3,
        c2r0, c2r1, c2r2, c2r3,
        c3r0, c3r1, c3r2, c3r3
    );
    let result = euler.to_affine_matrix();

    assert_relative_eq!(result, expected, epsilon = 1e-8);  
}


#[test]
fn test_to_affine_matrix_zero_euler_angles_is_identity() {
    let euler: EulerAngles<Radians<f64>> = EulerAngles::zero();
    let expected:Matrix4x4<f64> = Matrix4x4::identity();
    let result = euler.to_affine_matrix();

    assert_eq!(result, expected);
}

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
fn test_euler_angles_to_matrix_rotation_matrix1() {
    let roll_yz: Radians<f64> = Radians::full_turn_div_2();
    let yaw_zx: Radians<f64> = Radians::full_turn_div_8();
    let pitch_xy: Radians<f64> = Radians::full_turn_div_6();
    let matrix_yz = Matrix3x3::from_angle_x(roll_yz);
    let matrix_zx = Matrix3x3::from_angle_y(yaw_zx);
    let matrix_xy = Matrix3x3::from_angle_z(pitch_xy);
    let matrix = matrix_yz * matrix_zx * matrix_xy;
    let euler_angles = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    let expected = matrix;
    let result = euler_angles.to_matrix();

    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

#[test]
fn test_euler_angles_to_matrix_rotation_matrix2() {
    let roll_yz: Radians<f64> = Radians::full_turn_div_2();
    let yaw_zx: Radians<f64> = Radians::full_turn_div_4();
    let pitch_xy: Radians<f64> = Radians::full_turn_div_6();
    let matrix_yz = Matrix3x3::from_angle_x(roll_yz);
    let matrix_zx = Matrix3x3::from_angle_y(yaw_zx);
    let matrix_xy = Matrix3x3::from_angle_z(pitch_xy);
    let matrix = matrix_yz * matrix_zx * matrix_xy;
    let euler_angles = EulerAngles::new(roll_yz, yaw_zx, pitch_xy);
    let expected = matrix;
    let result = euler_angles.to_matrix();

    assert_eq!(result, expected);
}