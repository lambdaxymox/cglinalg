extern crate cglinalg;


use cglinalg::{
    Translation2,
    Translation3,
    Point2,
    Point3,
    Vector2,
    Vector3,
    Zero,
};


#[test]
fn test_translation2_point() {
    let point = Point2::new(1_f64, 2_f64);
    let distance = Vector2::new(4_f64, 4_f64);
    let translation = Translation2::from_vector(&distance);
    let expected = Point2::new(1_f64 + 4_f64, 2_f64 + 4_f64);
    let result = translation.translate_point(&point);

    assert_eq!(result, expected);
}

/// A translation should not translate a vector.
#[test]
fn test_translation2_vector() {
    let vector = Vector2::new(1_f64, 2_f64);
    let distance = Vector2::new(4_f64, 4_f64);
    let translation = Translation2::from_vector(&distance);
    let expected = vector;
    let result = translation.translate_vector(&vector);

    assert_eq!(result, expected);
}

#[test]
fn test_translation2_inverse_point() {
    let point = Point2::new(1_f64, 2_f64);
    let distance = Vector2::new(4_f64, 4_f64);
    let translation = Translation2::from_vector(&distance);
    let expected = Point2::new(1_f64 - 4_f64, 2_f64 - 4_f64);
    let result = translation.inverse_translate_point(&point);

    assert_eq!(result, expected);
}

/// A translation should not translate a vector.
#[test]
fn test_translation2_inverse_vector() {
    let vector = Vector2::new(1_f64, 2_f64);
    let distance = Vector2::new(4_f64, 4_f64);
    let translation = Translation2::from_vector(&distance);
    let expected = vector;
    let result = translation.inverse_translate_vector(&vector);

    assert_eq!(result, expected);
}

#[test]
fn test_translation2_inverse() {
    let vector = Vector2::new(4_f64, 5_f64);
    let inverse_vector = -vector;
    let translation = Translation2::from_vector(&vector);
    let expected = Translation2::from_vector(&inverse_vector);
    let result = translation.inverse();

    assert_eq!(result, expected);
}

#[test]
fn test_translation2_identity() {
    let zero: Vector2<f64> = Vector2::zero();
    let result = Translation2::from_vector(&zero);
    let expected = Translation2::identity();

    assert_eq!(result, expected);
}

#[test]
fn test_translation2_translation_between_points() {
    let point1 = Point2::new(1_f64, 2_f64);
    let point2 = Point2::new(5_f64, 6_f64);
    let diff = Vector2::new(4_f64, 4_f64);
    let expected = Translation2::from_vector(&diff);
    let result = Translation2::translation_between_points(&point1, &point2);

    assert_eq!(result, expected);
}

#[test]
fn test_translation2_translation_between_vectors() {
    let vector1 = Vector2::new(1_f64, 2_f64);
    let vector2 = Vector2::new(5_f64, 6_f64);
    let diff = Vector2::new(4_f64, 4_f64);
    let expected = Translation2::from_vector(&diff);
    let result = Translation2::translation_between_vectors(&vector1, &vector2);

    assert_eq!(result, expected);    
}

#[test]
fn test_translation2_multiplication_point() {
    let point = Point2::new(1_f64, 2_f64);
    let distance = Vector2::new(4_f64, 4_f64);
    let translation = Translation2::from_vector(&distance);
    let expected = Point2::new(1_f64 + 4_f64, 2_f64 + 4_f64);
    let result = translation * point;

    assert_eq!(result, expected);
}

#[test]
fn test_translation3_point() {
    let point = Point3::new(1_f64, 2_f64, 3_f64);
    let distance = Vector3::new(4_f64, 4_f64, 4_f64);
    let translation = Translation3::from_vector(&distance);
    let expected = Point3::new(1_f64 + 4_f64, 2_f64 + 4_f64, 3_f64 + 4_f64);
    let result = translation.translate_point(&point);

    assert_eq!(result, expected);
}

/// A translation should not translate a vector.
#[test]
fn test_translation3_vector() {
    let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    let distance = Vector3::new(4_f64, 4_f64, 4_f64);
    let translation = Translation3::from_vector(&distance);
    let expected = vector;
    let result = translation.translate_vector(&vector);

    assert_eq!(result, expected);
}

#[test]
fn test_translation3_inverse_point() {
    let point = Point3::new(1_f64, 2_f64, 3_f64);
    let distance = Vector3::new(4_f64, 4_f64, 4_f64);
    let translation = Translation3::from_vector(&distance);
    let expected = Point3::new(1_f64 - 4_f64, 2_f64 - 4_f64, 3_f64 - 4_f64);
    let result = translation.inverse_translate_point(&point);

    assert_eq!(result, expected);
}

/// A translation should not translate a vector.
#[test]
fn test_translation3_inverse_vector() {
    let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    let distance = Vector3::new(4_f64, 4_f64, 4_f64);
    let translation = Translation3::from_vector(&distance);
    let expected = vector;
    let result = translation.inverse_translate_vector(&vector);

    assert_eq!(result, expected);
}

#[test]
fn test_translation3_inverse() {
    let vector = Vector3::new(4_f64, 5_f64, 6_f64);
    let inverse_vector = -vector;
    let translation = Translation3::from_vector(&vector);
    let expected = Translation3::from_vector(&inverse_vector);
    let result = translation.inverse();

    assert_eq!(result, expected);
}

#[test]
fn test_translation3_identity() {
    let zero: Vector3<f64> = Vector3::zero();
    let result = Translation3::from_vector(&zero);
    let expected = Translation3::identity();

    assert_eq!(result, expected);
}

#[test]
fn test_translation3_translation_between_points() {
    let point1 = Point3::new(1_f64, 2_f64, 3_f64);
    let point2 = Point3::new(5_f64, 6_f64, 7_f64);
    let diff = Vector3::new(4_f64, 4_f64, 4_f64);
    let expected = Translation3::from_vector(&diff);
    let result = Translation3::translation_between_points(&point1, &point2);

    assert_eq!(result, expected);
}

#[test]
fn test_translation3_translation_between_vectors() {
    let vector1 = Vector3::new(1_f64, 2_f64, 3_f64);
    let vector2 = Vector3::new(5_f64, 6_f64, 7_f64);
    let diff = Vector3::new(4_f64, 4_f64, 4_f64);
    let expected = Translation3::from_vector(&diff);
    let result = Translation3::translation_between_vectors(&vector1, &vector2);

    assert_eq!(result, expected);    
}

#[test]
fn test_translation3_multiplication_point() {
    let point = Point3::new(1_f64, 2_f64, 3_f64);
    let distance = Vector3::new(4_f64, 4_f64, 4_f64);
    let translation = Translation3::from_vector(&distance);
    let expected = Point3::new(1_f64 + 4_f64, 2_f64 + 4_f64, 3_f64 + 4_f64);
    let result = translation * point;

    assert_eq!(result, expected);
}

