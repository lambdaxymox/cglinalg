extern crate cglinalg;


#[cfg(test)]
mod isometry2_tests {
    use cglinalg::{
        Isometry2,
        Rotation2,
        Translation2,
        Degrees,
        Point2,
        Vector2,
        Unit,
    };
    use cglinalg::approx::{
        relative_eq,
    };


    #[test]
    fn test_isometry_transform_point() {
        let vector = Vector2::new(1_f64, 2_f64);
        let translation = Translation2::from_vector(&vector);
        let rotation = Rotation2::from_angle(Degrees(90_f64));
        let isometry = Isometry2::from_parts(translation, rotation);
        let point = Point2::new(4_f64, 5_f64);
        let expected = Point2::new(-4_f64, 6_f64);
        let result = isometry.transform_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_isometry_transform_vector() {
        let distance = Vector2::new(1_f64, 2_f64);
        let translation = Translation2::from_vector(&distance);
        let rotation = Rotation2::from_angle(Degrees(90_f64));
        let isometry = Isometry2::from_parts(translation, rotation);
        let vector = Vector2::new(1_f64, 2_f64);
        let expected = Vector2::new(-2_f64, 1_f64);
        let result = isometry.transform_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_from_rotation() {
        let isometry = Isometry2::from_angle(Degrees(90_f64));
        let vector = Vector2::unit_x();
        let expected = Vector2::unit_y();
        let result = isometry.transform_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_from_translation() {
        let distance = Vector2::new(4_f64, 5_f64);
        let translation = Translation2::from_vector(&distance);
        let isometry = Isometry2::from_translation(translation);
        
        assert_eq!(isometry.translation(), &translation);
    }

    #[test]
    fn test_from_angle_translation() {
        let angle = Degrees(70_f64);
        let distance = Vector2::new(12_f64, 5_f64);
        let rotation = Rotation2::from_angle(angle);
        let translation = Translation2::from_vector(&distance);
        let expected = Isometry2::from_parts(translation, rotation);
        let result = Isometry2::from_angle_translation(angle, &distance);

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod isometry3_tests {
    use cglinalg::{
        Isometry3,
        Angle,
        Degrees,
        Radians,
        Point3,
        Vector3,
        Unit,
    };
    use cglinalg::approx::{
        relative_eq,
    };

}