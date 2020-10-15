extern crate cglinalg;


#[cfg(test)]
mod isometry2_tests {
    use cglinalg::{
        Isometry2,
        Rotation2,
        Translation2,
        Angle,
        Degrees,
        Point2,
        Vector2,
        Matrix3x3,
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

    #[test]
    fn test_rotation_between_axis() {
        let unit_x: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
        let unit_y: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_y());
        let isometry = Isometry2::rotation_between_axis(&unit_x, &unit_y);
        let expected = unit_y.into_inner();
        let result = isometry.transform_vector(&unit_x.into_inner());

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_rotation_between_vectors() {
        let vector1: Vector2<f64> = 17_f64 * Vector2::unit_x();
        let vector2: Vector2<f64> = 3_f64 * Vector2::unit_y();
        let isometry = Isometry2::rotation_between(&vector1, &vector2);
        let point = Point2::new(203_f64, 0_f64);
        let expected = Point2::new(0_f64, 203_f64);
        let result = isometry.transform_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_to_affine_matrix() {
        let angle = Degrees(60_f64);
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        let distance = Vector2::new(5_f64, 18_f64);
        let isometry = Isometry2::from_angle_translation(angle, &distance);
        let expected = Matrix3x3::new(
             cos_angle,  sin_angle,  0_f64,
            -sin_angle,  cos_angle,  0_f64,
             distance.x, distance.y, 1_f64
        );
        let result = isometry.to_affine_matrix();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_isometry_inverse() {
        let angle = Degrees(72_f64);
        let distance = Vector2::new(-567_f64, 23_f64);
        let isometry = Isometry2::from_angle_translation(angle, &distance);
        let isometry_inv = isometry.inverse();
        let point = Point2::new(34_f64, 139_f64);
        let expected = point;
        let result = isometry_inv * (isometry * point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));

        let result = isometry * (isometry_inv * point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_inverse_transform_point() {
        let angle = Degrees(72_f64);
        let cos_neg_angle = (-angle).cos();
        let sin_neg_angle = (-angle).sin();
        let distance = Vector2::new(-567_f64, 23_f64);
        let isometry = Isometry2::from_angle_translation(angle, &distance);        
        let point = Point2::new(1_f64, 2_f64);
        let diff: Point2<f64> = point - distance;
        let expected = Point2::new(
            cos_neg_angle * diff.x - sin_neg_angle * diff.y,
            sin_neg_angle * diff.x + cos_neg_angle * diff.y
        );
        let result = isometry.inverse_transform_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_transform_vector() {
        let angle = Degrees(72_f64);
        let cos_neg_angle = (-angle).cos();
        let sin_neg_angle = (-angle).sin();
        let distance = Vector2::new(-567_f64, 23_f64);
        let isometry = Isometry2::from_angle_translation(angle, &distance);        
        let vector = Vector2::new(1_f64, 2_f64);
        let expected = Vector2::new(
            cos_neg_angle * vector.x - sin_neg_angle * vector.y,
            sin_neg_angle * vector.x + cos_neg_angle * vector.y
        );
        let result = isometry.inverse_transform_vector(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_identity() {
        let isometry = Isometry2::identity();
        let point = Point2::new(1_f64, 2_f64);
        let expected = point;
        let result = isometry * point;

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod isometry3_tests {
    use cglinalg::{
        Matrix4x4,
        Isometry3,
        Rotation3,
        Translation3,
        Angle,
        Degrees,
        Point3,
        Vector3,
        Unit,
    };
    use cglinalg::approx::{
        relative_eq,
    };

    #[test]
    fn test_isometry_transform_point() {
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let translation = Translation3::from_vector(&vector);
        let angle = Degrees(90_f64);
        let axis = Unit::from_value(Vector3::unit_z());
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let isometry = Isometry3::from_parts(translation, rotation);
        let point = Point3::new(4_f64, 5_f64, 6_f64);
        let expected = Point3::new(-4_f64, 6_f64, 9_f64);
        let result = isometry.transform_point(&point);
    
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_isometry_transform_vector() {
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let translation = Translation3::from_vector(&vector);
        let angle = Degrees(90_f64);
        let axis = Unit::from_value(Vector3::unit_z());
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let isometry = Isometry3::from_parts(translation, rotation);
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector3::new(-2_f64, 1_f64, 3_f64);
        let result = isometry.transform_vector(&vector);
    
        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_from_rotation() {
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Degrees(90_f64);
        let isometry = Isometry3::from_axis_angle(&axis, angle);
        let vector = Vector3::unit_x();
        let expected = Vector3::unit_y();
        let result = isometry.transform_vector(&vector);
    
        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }
    
    #[test]
    fn test_from_translation() {
        let distance = Vector3::new(4_f64, 5_f64, 6_f64);
        let translation = Translation3::from_vector(&distance);
        let isometry = Isometry3::from_translation(translation);
        
        assert_eq!(isometry.translation(), &translation);
    }
    
    #[test]
    fn test_from_angle_translation() {
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Degrees(70_f64);
        let distance = Vector3::new(12_f64, 5_f64, 77_f64);
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let translation = Translation3::from_vector(&distance);
        let expected = Isometry3::from_parts(translation, rotation);
        let result = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
    
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_rotation_between_axis() {
        let unit_x: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x());
        let unit_y: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y());
        let isometry = Isometry3::rotation_between_axis(&unit_x, &unit_y).unwrap();
        let expected = unit_y.into_inner();
        let result = isometry.transform_vector(&unit_x.into_inner());
    
        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_rotation_between_vectors() {
        let vector1: Vector3<f64> = 17_f64 * Vector3::unit_x();
        let vector2: Vector3<f64> = 3_f64 * Vector3::unit_y();
        let isometry = Isometry3::rotation_between(&vector1, &vector2).unwrap();
        let point = Point3::new(203_f64, 0_f64, 0_f64);
        let expected = Point3::new(0_f64, 203_f64, 0_f64);
        let result = isometry.transform_point(&point);
    
        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }
    
    #[test]
    fn test_to_affine_matrix() {
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Degrees(60_f64);
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        let distance = Vector3::new(5_f64, 18_f64, 12_f64);
        let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
        let expected = Matrix4x4::new(
             cos_angle,  sin_angle,  0_f64,      0_f64,
            -sin_angle,  cos_angle,  0_f64,      0_f64,
             0_f64,      0_f64,      1_f64,      0_f64,
             distance.x, distance.y, distance.z, 1_f64
        );
        let result = isometry.to_affine_matrix();
    
        assert_eq!(result, expected);
    }

    #[test]
    fn test_isometry_inverse() {
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Degrees(72_f64);
        let distance = Vector3::new(-567_f64, 23_f64, 201_f64);
        let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
        let isometry_inv = isometry.inverse();
        let point = Point3::new(34_f64, 139_f64, 91_f64);
        let expected = point;
        let result = isometry_inv * (isometry * point);
    
        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    
        let result = isometry * (isometry_inv * point);
    
        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    #[test]
    fn test_inverse_transform_point() {
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Degrees(72_f64);
        let cos_neg_angle = (-angle).cos();
        let sin_neg_angle = (-angle).sin();
        let distance = Vector3::new(-567_f64, 23_f64, 201_f64);
        let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);        
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let diff: Point3<f64> = point - distance;
        let expected = Point3::new(
            cos_neg_angle * diff.x - sin_neg_angle * diff.y,
            sin_neg_angle * diff.x + cos_neg_angle * diff.y,
            diff.z
        );
        let result = isometry.inverse_transform_point(&point);
    
        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_transform_vector() {
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Degrees(72_f64);
        let cos_neg_angle = (-angle).cos();
        let sin_neg_angle = (-angle).sin();
        let distance = Vector3::new(-567_f64, 23_f64, 201_f64);
        let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);        
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector3::new(
            cos_neg_angle * vector.x - sin_neg_angle * vector.y,
            sin_neg_angle * vector.x + cos_neg_angle * vector.y,
            vector.z
        );
        let result = isometry.inverse_transform_vector(&vector);
    
        assert_eq!(result, expected);
    }

    #[test]
    fn test_identity() {
        let isometry = Isometry3::identity();
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = point;
        let result = isometry * point;

        assert_eq!(result, expected);
    }
}