extern crate cglinalg_core;


use approx::assert_relative_eq;
use cglinalg_core::{
    Quaternion,
    Unit,
    Vector1,
    Vector2,
    Vector3,
    Vector4,
};


#[test]
fn test_unit_vector1_unit_x() {
    let unit_x: Vector2<f64> = Vector2::unit_x();
    let (unit_unit_x, norm_unit_x) = Unit::from_value_with_norm(unit_x);

    assert_eq!(norm_unit_x, 1_f64);
    assert_eq!(unit_unit_x.as_ref(), &unit_x);
}

#[test]
fn test_unit_vector2_unit_x() {
    let unit_x: Vector2<f64> = Vector2::unit_x();
    let (unit_unit_x, norm_unit_x) = Unit::from_value_with_norm(unit_x);

    assert_eq!(norm_unit_x, 1_f64);
    assert_eq!(unit_unit_x.as_ref(), &unit_x);
}

#[test]
fn test_unit_vector2_unit_y() {
    let unit_y: Vector2<f64> = Vector2::unit_y();
    let (unit_unit_y, norm_unit_y) = Unit::from_value_with_norm(unit_y);

    assert_eq!(norm_unit_y, 1_f64);
    assert_eq!(unit_unit_y.as_ref(), &unit_y);
}

#[test]
fn test_unit_vector3_unit_x() {
    let unit_x: Vector3<f64> = Vector3::unit_x();
    let (unit_unit_x, norm_unit_x) = Unit::from_value_with_norm(unit_x);

    assert_eq!(norm_unit_x, 1_f64);
    assert_eq!(unit_unit_x.as_ref(), &unit_x);
}

#[test]
fn test_unit_vector3_unit_y() {
    let unit_y: Vector3<f64> = Vector3::unit_y();
    let (unit_unit_y, norm_unit_y) = Unit::from_value_with_norm(unit_y);

    assert_eq!(norm_unit_y, 1_f64);
    assert_eq!(unit_unit_y.as_ref(), &unit_y);
}

#[test]
fn test_unit_vector3_unit_z() {
    let unit_z: Vector3<f64> = Vector3::unit_z();
    let (unit_unit_z, norm_unit_z) = Unit::from_value_with_norm(unit_z);

    assert_eq!(norm_unit_z, 1_f64);
    assert_eq!(unit_unit_z.as_ref(), &unit_z);
}

#[test]
fn test_unit_vector4_unit_x() {
    let unit_x: Vector4<f64> = Vector4::unit_x();
    let (unit_unit_x, norm_unit_x) = Unit::from_value_with_norm(unit_x);

    assert_eq!(norm_unit_x, 1_f64);
    assert_eq!(unit_unit_x.as_ref(), &unit_x);
}

#[test]
fn test_unit_vector4_unit_y() {
    let unit_y: Vector4<f64> = Vector4::unit_y();
    let (unit_unit_y, norm_unit_y) = Unit::from_value_with_norm(unit_y);

    assert_eq!(norm_unit_y, 1_f64);
    assert_eq!(unit_unit_y.as_ref(), &unit_y);
}

#[test]
fn test_unit_vector4_unit_z() {
    let unit_z: Vector4<f64> = Vector4::unit_z();
    let (unit_unit_z, norm_unit_z) = Unit::from_value_with_norm(unit_z);

    assert_eq!(norm_unit_z, 1_f64);
    assert_eq!(unit_unit_z.as_ref(), &unit_z);
}

#[test]
fn test_unit_vector4_unit_w() {
    let unit_w: Vector4<f64> = Vector4::unit_w();
    let (unit_unit_w, norm_unit_w) = Unit::from_value_with_norm(unit_w);

    assert_eq!(norm_unit_w, 1_f64);
    assert_eq!(unit_unit_w.as_ref(), &unit_w);
}

#[test]
fn test_unit_vector1() {
    let vector = Vector1::new(3_f64);
    let (wrapped, norm) = Unit::from_value_with_norm(vector);
    let unit_vector = wrapped.as_ref();

    assert_relative_eq!(unit_vector.norm(), 1_f64, epsilon = 1e-10);
    assert_eq!(vector.norm(), norm);
}

#[test]
fn test_unit_vector2() {
    let vector = Vector2::new(3_f64, 4_f64);
    let (wrapped, norm) = Unit::from_value_with_norm(vector);
    let unit_vector = wrapped.as_ref();

    assert_relative_eq!(unit_vector.norm(), 1_f64, epsilon = 1e-10);
    assert_eq!(vector.norm(), norm);
}

#[test]
fn test_unit_vector3() {
    let vector = Vector3::new(3_f64, 4_f64, 5_f64);
    let (wrapped, norm) = Unit::from_value_with_norm(vector);
    let unit_vector = wrapped.as_ref();

    assert_relative_eq!(unit_vector.norm(), 1_f64, epsilon = 1e-10);
    assert_eq!(vector.norm(), norm);
}

#[test]
fn test_unit_vector4() {
    let vector = Vector4::new(3_f64, 4_f64, 5_f64, 6_f64);
    let (wrapped, norm) = Unit::from_value_with_norm(vector);
    let unit_vector = wrapped.as_ref();

    assert_relative_eq!(unit_vector.norm(), 1_f64, epsilon = 1e-10);
    assert_eq!(vector.norm(), norm);
}

#[test]
fn test_unit_vector1_close_to_zero_returns_none() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Vector1::new(f64::EPSILON);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_vector2_close_to_zero_returns_none1() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Vector2::new(f64::EPSILON, 0_f64);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_vector2_close_to_zero_returns_none2() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Vector2::new(0_f64, f64::EPSILON);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_vector3_close_to_zero_returns_none1() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Vector3::new(f64::EPSILON, 0_f64, 0_f64);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_vector3_close_to_zero_returns_none2() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Vector3::new(0_f64, f64::EPSILON, 0_f64);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_vector3_close_to_zero_returns_none3() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Vector3::new(0_f64, 0_f64, f64::EPSILON);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_vector4_close_to_zero_returns_none1() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Vector4::new(f64::EPSILON, 0_f64, 0_f64, 0_f64);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_vector4_close_to_zero_returns_none2() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Vector4::new(0_f64, f64::EPSILON, 0_f64, 0_f64);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_vector4_close_to_zero_returns_none3() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Vector4::new(0_f64, 0_f64, f64::EPSILON, 0_f64);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_vector4_close_to_zero_returns_none4() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Vector4::new(0_f64, 0_f64, 0_f64, f64::EPSILON);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_quaternion_unit_s() {
    let unit_s: Quaternion<f64> = Quaternion::unit_s();
    let (unit_unit_s, norm_unit_s) = Unit::from_value_with_norm(unit_s);

    assert_eq!(norm_unit_s, 1_f64);
    assert_eq!(unit_unit_s.as_ref(), &unit_s);
}

#[test]
fn test_unit_quaternion_unit_x() {
    let unit_x: Vector4<f64> = Vector4::unit_x();
    let (unit_unit_x, norm_unit_x) = Unit::from_value_with_norm(unit_x);

    assert_eq!(norm_unit_x, 1_f64);
    assert_eq!(unit_unit_x.as_ref(), &unit_x);
}

#[test]
fn test_unit_quaternion_unit_y() {
    let unit_y: Vector4<f64> = Vector4::unit_y();
    let (unit_unit_y, norm_unit_y) = Unit::from_value_with_norm(unit_y);

    assert_eq!(norm_unit_y, 1_f64);
    assert_eq!(unit_unit_y.as_ref(), &unit_y);
}

#[test]
fn test_unit_quaternion_unit_z() {
    let unit_z: Vector4<f64> = Vector4::unit_z();
    let (unit_unit_z, norm_unit_z) = Unit::from_value_with_norm(unit_z);

    assert_eq!(norm_unit_z, 1_f64);
    assert_eq!(unit_unit_z.as_ref(), &unit_z);
}

#[test]
fn test_unit_quaternion_close_to_zero_returns_none1() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Quaternion::new(f64::EPSILON, 0_f64, 0_f64, 0_f64);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_quaternion_close_to_zero_returns_none2() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Quaternion::new(0_f64, f64::EPSILON, 0_f64, 0_f64);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_quaternion_close_to_zero_returns_none3() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Quaternion::new(0_f64, 0_f64, f64::EPSILON, 0_f64);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}

#[test]
fn test_unit_quaternion_close_to_zero_returns_none4() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Quaternion::new(0_f64, 0_f64, 0_f64, f64::EPSILON);
    let result = Unit::try_from_value(vector, threshold);

    assert!(result.is_none());
}
