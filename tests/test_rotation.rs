extern crate cglinalg;


#[cfg(test)]
mod rotation2_tests {
    use cglinalg::{
        Rotation2,
        Degrees,
        Radians,
        Point2,
        Vector2,
        Unit,
    };
    use cglinalg::approx::{
        relative_eq,
    };


    #[test]
    fn test_rotate_point1() {
        let point = Point2::new(3_f64, 0_f64);
        let angle = Degrees(90_f64);
        let rotation = Rotation2::from_angle(angle);
        let expected = Point2::new(0_f64, 3_f64);
        let result = rotation.rotate_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_rotate_vector1() {
        let vector = 3_f64 * Vector2::unit_x();
        let angle = Degrees(90_f64);
        let rotation = Rotation2::from_angle(angle);
        let expected = 3_f64 * Vector2::unit_y();
        let result = rotation.rotate_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_rotate_point2() {
        let point = 3_f64 * Point2::new(f64::sqrt(3_f64) / 2_f64, -1_f64 / 2_f64);
        let angle = Degrees(-90_f64);
        let rotation = Rotation2::from_angle(angle);
        let expected = 3_f64 * Point2::new(-1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
        let result = rotation.rotate_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_rotate_vector2() {
        let vector = 3_f64 * Vector2::new(f64::sqrt(3_f64) / 2_f64, -1_f64 / 2_f64);
        let angle = Degrees(-90_f64);
        let rotation = Rotation2::from_angle(angle);
        let expected = 3_f64 * Vector2::new(-1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
        let result = rotation.rotate_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_rotation_between_axis_vectors_point1() {
        let point = Point2::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64);
        let vector1 = Unit::from_value(Vector2::unit_y());
        let vector2 = Unit::from_value(Vector2::unit_x());
        let rotation = Rotation2::rotation_between_axis(&vector1, &vector2);
        let expected = Point2::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
        let result = rotation.rotate_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_rotation_between_axis_vectors_point2() {
        let point = Point2::new(0_f64, 1_f64);
        let vector1 = Unit::from_value(Vector2::unit_y());
        let vector2 = Unit::from_value(Vector2::unit_x());
        let rotation = Rotation2::rotation_between_axis(&vector1, &vector2);
        let expected = Point2::new(1_f64, 0_f64);
        let result = rotation.rotate_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_rotation_between_axis_vectors_vector1() {
        let vector = Vector2::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64);
        let vector1 = Unit::from_value(Vector2::unit_y());
        let vector2 = Unit::from_value(Vector2::unit_x());
        let rotation = Rotation2::rotation_between_axis(&vector1, &vector2);
        let expected = Vector2::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
        let result = rotation.rotate_vector(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_rotation_between_axis_vectors_vector2() {
        let vector1: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_y());
        let vector2: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
        let rotation = Rotation2::rotation_between_axis(&vector1, &vector2);
        let vector = Vector2::unit_y();
        let expected = Vector2::unit_x();
        let result = rotation.rotate_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_rotation_between_vectors_point() {
        let point = 3_f64 * Point2::new(0_f64, 1_f64);
        let vector1 = 5_f64 * Vector2::unit_y();
        let vector2 = 12_f64 * Vector2::unit_x();
        let rotation = Rotation2::rotation_between(&vector1, &vector2);
        let expected = 3_f64 * Point2::new(1_f64, 0_f64);
        let result = rotation.rotate_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_rotation_between_vectors_vector() {
        let vector = 3_f64 * Vector2::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64);
        let vector1 = 5_f64 * Vector2::unit_y();
        let vector2 = 12_f64 * Vector2::unit_x();
        let rotation = Rotation2::rotation_between(&vector1, &vector2);
        let expected = 3_f64 * Vector2::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
        let result = rotation.rotate_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }
}


#[cfg(test)]
mod rotation3_tests {
    use cglinalg::{
        Matrix3x3,
        Angle,
        Degrees,
        Radians,
        Point3,
        Vector3,
    };
    use cglinalg::approx::{
        relative_eq,
    };
}
