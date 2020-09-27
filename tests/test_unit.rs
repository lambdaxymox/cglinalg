extern crate cglinalg;
extern crate proptest;


use cglinalg::{
    Magnitude,
    Unit,
    Vector3,
};


/// Pass a unit vector into a `Unit` constructor should not affect the unit vector.
/// A unit vector should have a unit norm.
#[test]
fn test_vector3_unit_vectors() {
    let unit_x: Vector3<f64> = Vector3::unit_x();
    let unit_y: Vector3<f64> = Vector3::unit_y();
    let unit_z: Vector3<f64> = Vector3::unit_z();

    let (unit_unit_x, norm_unit_x) = Unit::from_value_with_magnitude(unit_x); 
    let (unit_unit_y, norm_unit_y) = Unit::from_value_with_magnitude(unit_y);
    let (unit_unit_z, norm_unit_z) = Unit::from_value_with_magnitude(unit_z);

    assert_eq!(norm_unit_x, 1_f64);
    assert_eq!(norm_unit_y, 1_f64);
    assert_eq!(norm_unit_z, 1_f64);

    assert_eq!(unit_unit_x.as_ref(), &unit_x);
    assert_eq!(unit_unit_y.as_ref(), &unit_y);
    assert_eq!(unit_unit_z.as_ref(), &unit_z);
}

/// A `Unit` constructor should correctly normalize the resulting vector.
#[test]
fn test_vector() {
    let vector = Vector3::new(2.0, 2.0, 2.0);
    let (wrapped, norm) = Unit::from_value_with_magnitude(vector);
    let unit_vector = wrapped.as_ref();

    assert_eq!(unit_vector.magnitude(), 1.0);
    assert_eq!(vector.magnitude(), norm);
}


/// The `try_new` function should return `None` when the
/// vector is smaller than the threshold.
#[test]
fn test_vector_close_to_zero() {
    let threshold = 3_f64 * f64::EPSILON;
    let vector = Vector3::new(f64::EPSILON, 0.0, 0.0);
    let expected = None;
    let result = Unit::try_from_value(vector, threshold);

    assert_eq!(result, expected);
}

