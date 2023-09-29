extern crate cglinalg_trigonometry;
extern crate cglinalg_transform;


#[cfg(test)]
mod rotation2_tests {
    use cglinalg_trigonometry::{
        Degrees,
    };
    use cglinalg_core::{
        Point2,
        Vector2,
        Unit,
    };
    use cglinalg_transform::{
        Rotation2,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_rotate_point1() {
        let point = Point2::new(3_f64, 0_f64);
        let angle = Degrees(90_f64);
        let rotation = Rotation2::from_angle(angle);
        let expected = Point2::new(0_f64, 3_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotate_vector1() {
        let vector = 3_f64 * Vector2::unit_x();
        let angle = Degrees(90_f64);
        let rotation = Rotation2::from_angle(angle);
        let expected = 3_f64 * Vector2::unit_y();
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotate_point2() {
        let point = 3_f64 * Point2::new(f64::sqrt(3_f64) / 2_f64, -1_f64 / 2_f64);
        let angle = Degrees(-90_f64);
        let rotation = Rotation2::from_angle(angle);
        let expected = 3_f64 * Point2::new(-1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotate_vector2() {
        let vector = 3_f64 * Vector2::new(f64::sqrt(3_f64) / 2_f64, -1_f64 / 2_f64);
        let angle = Degrees(-90_f64);
        let rotation = Rotation2::from_angle(angle);
        let expected = 3_f64 * Vector2::new(-1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_between_axis_vectors_point1() {
        let point = Point2::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64);
        let vector1 = Unit::from_value(Vector2::unit_y());
        let vector2 = Unit::from_value(Vector2::unit_x());
        let rotation = Rotation2::rotation_between_axis(&vector1, &vector2);
        let expected = Point2::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
        let result = rotation.apply_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_rotation_between_axis_vectors_point2() {
        let point = Point2::new(0_f64, 1_f64);
        let vector1 = Unit::from_value(Vector2::unit_y());
        let vector2 = Unit::from_value(Vector2::unit_x());
        let rotation = Rotation2::rotation_between_axis(&vector1, &vector2);
        let expected = Point2::new(1_f64, 0_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_between_axis_vectors_vector1() {
        let vector = Vector2::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64);
        let vector1 = Unit::from_value(Vector2::unit_y());
        let vector2 = Unit::from_value(Vector2::unit_x());
        let rotation = Rotation2::rotation_between_axis(&vector1, &vector2);
        let expected = Vector2::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
        let result = rotation.apply_vector(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_rotation_between_axis_vectors_vector2() {
        let vector1: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_y());
        let vector2: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
        let rotation = Rotation2::rotation_between_axis(&vector1, &vector2);
        let vector = Vector2::unit_y();
        let expected = Vector2::unit_x();
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_between_vectors_point() {
        let point = 3_f64 * Point2::new(0_f64, 1_f64);
        let vector1 = 5_f64 * Vector2::unit_y();
        let vector2 = 12_f64 * Vector2::unit_x();
        let rotation = Rotation2::rotation_between(&vector1, &vector2);
        let expected = 3_f64 * Point2::new(1_f64, 0_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_between_vectors_vector() {
        let vector = 3_f64 * Vector2::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64);
        let vector1 = 5_f64 * Vector2::unit_y();
        let vector2 = 12_f64 * Vector2::unit_x();
        let rotation = Rotation2::rotation_between(&vector1, &vector2);
        let expected = 3_f64 * Vector2::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_vector3() {
        let angle = Degrees(135_f64);
        let rotation = Rotation2::from_angle(angle);
        let expected = Vector2::new(-1_f64, 1_f64);
        let vector =  Vector2::new(f64::sqrt(2_f64), 0_f64);
        let result = rotation.apply_vector(&vector);
        
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_inverse_vector() {
        let angle = Degrees(135_f64);
        let rotation = Rotation2::from_angle(angle);
        let vector = Vector2::new(-1_f64, 1_f64);
        let expected = Vector2::new(f64::sqrt(2_f64), 0_f64);
        let rotation_inv = rotation.inverse();
        let result = rotation_inv.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_inverse_rotate_vector() {
        let angle = Degrees(135_f64);
        let rotation = Rotation2::from_angle(angle);
        let  vector = Vector2::new(-1_f64, 1_f64);
        let expected =  Vector2::new(f64::sqrt(2_f64), 0_f64);
        let result = rotation.inverse_apply_vector(&vector);
        
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_inverse_rotate_point() {
        let angle = Degrees(135_f64);
        let rotation = Rotation2::from_angle(angle);
        let point = Point2::new(-1_f64, 1_f64);
        let expected = Point2::new(f64::sqrt(2_f64), 0_f64);
        let result = rotation.inverse_apply_point(&point);
        
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
}


#[cfg(test)]
mod rotation3_tests {
    use cglinalg_trigonometry::{
        Angle,
        Degrees,
        Radians,
    };
    use cglinalg_core::{
        Point3,
        Vector3,
        Unit,
    };
    use cglinalg_transform::{
        Rotation3,
    };
    use approx::{
        assert_relative_eq,
    };

    
    #[test]
    fn test_from_angle_x_rotation_should_not_rotate_x_axis() {
        let rotation = Rotation3::from_angle_x(Degrees(70_f64));
        let vector = Vector3::unit_x();
        let expected = vector;
        let result = rotation.apply_vector(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_angle_y_rotation_should_not_rotate_y_axis() {
        let rotation = Rotation3::from_angle_y(Degrees(70_f64));
        let vector = Vector3::unit_y();
        let expected = vector;
        let result = rotation.apply_vector(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_angle_z_rotation_should_not_rotate_z_axis() {
        let rotation = Rotation3::from_angle_z(Degrees(70_f64));
        let vector = Vector3::unit_z();
        let expected = vector;
        let result = rotation.apply_vector(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_angle_x_rotate_point1() {
        let rotation = Rotation3::from_angle_x(Degrees(90_f64));
        let point = Point3::new(0_f64, 1_f64, 0_f64);
        let expected = Point3::new(0_f64, 0_f64, 1_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_x_rotate_vector1() {
        let rotation = Rotation3::from_angle_x(Degrees(90_f64));
        let vector = Vector3::unit_y();
        let expected = Vector3::unit_z();
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_x_rotate_point2() {
        let rotation = Rotation3::from_angle_x(Degrees(90_f64));
        let point = Point3::new(0_f64, 1_f64, 1_f64);
        let expected = Point3::new(0_f64, -1_f64, 1_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_x_rotate_vector2() {
        let rotation = Rotation3::from_angle_x(Degrees(90_f64));
        let vector = Vector3::new(1_f64, 1_f64, 1_f64);
        let expected = Vector3::new(1_f64, -1_f64, 1_f64);
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_x_rotate_point3() {
        let rotation = Rotation3::from_angle_x(Degrees(90_f64));
        let point = Point3::new(3_f64, 1_f64, 1_f64);
        let expected = Point3::new(3_f64, -1_f64, 1_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_x_rotate_vector3() {
        let rotation = Rotation3::from_angle_x(Degrees(90_f64));
        let vector = Vector3::new(3_f64, 1_f64, 1_f64);
        let expected = Vector3::new(3_f64, -1_f64, 1_f64);
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_y_rotate_point1() {
        let rotation = Rotation3::from_angle_y(Degrees(90_f64));
        let point = Point3::new(0_f64, 0_f64, 1_f64);
        let expected = Point3::new(1_f64, 0_f64, 0_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_y_rotate_vector1() {
        let rotation = Rotation3::from_angle_y(Degrees(90_f64));
        let vector = Vector3::unit_z();
        let expected = Vector3::unit_x();
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_y_rotate_point2() {
        let rotation = Rotation3::from_angle_y(Degrees(90_f64));
        let point = Point3::new(-1_f64, 0_f64, 1_f64);
        let expected = Point3::new(1_f64, 0_f64, 1_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_y_rotate_vector2() {
        let rotation = Rotation3::from_angle_y(Degrees(90_f64));
        let vector = Vector3::new(-1_f64, 0_f64, 1_f64);
        let expected = Vector3::new(1_f64, 0_f64, 1_f64);
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_y_rotate_point3() {
        let rotation = Rotation3::from_angle_y(Degrees(90_f64));
        let point = Point3::new(-1_f64, 1_f64, 1_f64);
        let expected = Point3::new(1_f64, 1_f64, 1_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_y_rotate_vector3() {
        let rotation = Rotation3::from_angle_y(Degrees(90_f64));
        let vector = Vector3::new(-1_f64, 1_f64, 1_f64);
        let expected = Vector3::new(1_f64, 1_f64, 1_f64);
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_z_rotate_point1() {
        let rotation = Rotation3::from_angle_z(Degrees(90_f64));
        let point = Point3::new(1_f64, 0_f64, 0_f64);
        let expected = Point3::new(0_f64, 1_f64, 0_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_z_rotate_vector1() {
        let rotation = Rotation3::from_angle_z(Degrees(90_f64));
        let vector = Vector3::unit_x();
        let expected = Vector3::unit_y();
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_z_rotate_point2() {
        let rotation = Rotation3::from_angle_z(Degrees(90_f64));
        let point = Point3::new(1_f64, 1_f64, 0_f64);
        let expected = Point3::new(-1_f64, 1_f64, 0_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_z_rotate_vector2() {
        let rotation = Rotation3::from_angle_z(Degrees(90_f64));
        let vector = Vector3::new(1_f64, 1_f64, 0_f64);
        let expected = Vector3::new(-1_f64, 1_f64, 0_f64);
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_z_rotate_point3() {
        let rotation = Rotation3::from_angle_z(Degrees(90_f64));
        let point = Point3::new(1_f64, 1_f64, -1_f64);
        let expected = Point3::new(-1_f64, 1_f64, -1_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_z_rotate_vector3() {
        let rotation = Rotation3::from_angle_z(Degrees(90_f64));
        let vector = Vector3::new(1_f64, 1_f64, -1_f64);
        let expected = Vector3::new(-1_f64, 1_f64, -1_f64);
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle_x_inverse_rotate_point1() {
        let rotation = Rotation3::from_angle_x(Degrees(90_f64));
        let point = Point3::new(0_f64, 0_f64, 1_f64);
        let expected = Point3::new(0_f64, 1_f64, 0_f64);
        let result = rotation.inverse_apply_point(&point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_x_inverse_rotate_vector1() {
        let rotation = Rotation3::from_angle_x(Degrees(90_f64));
        let vector = Vector3::unit_z();
        let expected = Vector3::unit_y();
        let result = rotation.inverse_apply_vector(&vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_x_inverse_rotate_point2() {
        let rotation = Rotation3::from_angle_x(Degrees(90_f64));
        let point = Point3::new(0_f64, -1_f64, 1_f64);
        let expected = Point3::new(0_f64, 1_f64, 1_f64);
        let result = rotation.inverse_apply_point(&point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_x_inverse_rotate_vector2() {
        let rotation = Rotation3::from_angle_x(Degrees(90_f64));
        let vector = Vector3::new(1_f64, -1_f64, 1_f64);
        let expected = Vector3::new(1_f64, 1_f64, 1_f64);
        let result = rotation.inverse_apply_vector(&vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_x_inverse_rotate_point3() {
        let rotation = Rotation3::from_angle_x(Degrees(90_f64));
        let point = Point3::new(3_f64, -1_f64, 1_f64);
        let expected = Point3::new(3_f64, 1_f64, 1_f64);
        let result = rotation.inverse_apply_point(&point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_x_inverse_rotate_vector3() {
        let rotation = Rotation3::from_angle_x(Degrees(90_f64));
        let vector = Vector3::new(3_f64, -1_f64, 1_f64);
        let expected = Vector3::new(3_f64, 1_f64, 1_f64);
        let result = rotation.inverse_apply_vector(&vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_y_inverse_rotate_point1() {
        let rotation = Rotation3::from_angle_y(Degrees(90_f64));
        let point = Point3::new(1_f64, 0_f64, 0_f64);
        let expected = Point3::new(0_f64, 0_f64, 1_f64);
        let result = rotation.inverse_apply_point(&point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_y_inverse_rotate_vector1() {
        let rotation = Rotation3::from_angle_y(Degrees(90_f64));
        let vector = Vector3::unit_x();
        let expected = Vector3::unit_z();
        let result = rotation.inverse_apply_vector(&vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_y_inverse_rotate_point2() {
        let rotation = Rotation3::from_angle_y(Degrees(90_f64));
        let point = Point3::new(1_f64, 0_f64, 1_f64);
        let expected = Point3::new(-1_f64, 0_f64, 1_f64);
        let result = rotation.inverse_apply_point(&point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_y_inverse_rotate_vector2() {
        let rotation = Rotation3::from_angle_y(Degrees(90_f64));
        let vector = Vector3::new(1_f64, 0_f64, 1_f64);
        let expected = Vector3::new(-1_f64, 0_f64, 1_f64);
        let result = rotation.inverse_apply_vector(&vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_y_inverse_rotate_point3() {
        let rotation = Rotation3::from_angle_y(Degrees(90_f64));
        let point = Point3::new(1_f64, 1_f64, 1_f64);
        let expected = Point3::new(-1_f64, 1_f64, 1_f64);
        let result = rotation.inverse_apply_point(&point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_y_inverse_rotate_vector3() {
        let rotation = Rotation3::from_angle_y(Degrees(90_f64));
        let vector = Vector3::new(1_f64, 1_f64, 1_f64);
        let expected = Vector3::new(-1_f64, 1_f64, 1_f64);
        let result = rotation.inverse_apply_vector(&vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_z_inverse_rotate_point1() {
        let rotation = Rotation3::from_angle_z(Degrees(90_f64));
        let point = Point3::new(0_f64, 1_f64, 0_f64);
        let expected = Point3::new(1_f64, 0_f64, 0_f64);
        let result = rotation.inverse_apply_point(&point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_z_inverse_rotate_vector1() {
        let rotation = Rotation3::from_angle_z(Degrees(90_f64));
        let vector = Vector3::unit_y();
        let expected = Vector3::unit_x();
        let result = rotation.inverse_apply_vector(&vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_z_inverse_rotate_point2() {
        let rotation = Rotation3::from_angle_z(Degrees(90_f64));
        let point = Point3::new(-1_f64, 1_f64, 0_f64);
        let expected = Point3::new(1_f64, 1_f64, 0_f64);
        let result = rotation.inverse_apply_point(&point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_z_inverse_rotate_vector2() {
        let rotation = Rotation3::from_angle_z(Degrees(90_f64));
        let vector = Vector3::new(-1_f64, 1_f64, 0_f64);
        let expected = Vector3::new(1_f64, 1_f64, 0_f64);
        let result = rotation.inverse_apply_vector(&vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_z_inverse_rotate_point3() {
        let rotation = Rotation3::from_angle_z(Degrees(90_f64));
        let point = Point3::new(-1_f64, 1_f64, -1_f64);
        let expected = Point3::new(1_f64, 1_f64, -1_f64);
        let result = rotation.inverse_apply_point(&point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
    
    #[test]
    fn test_from_angle_z_inverse_rotate_vector3() {
        let rotation = Rotation3::from_angle_z(Degrees(90_f64));
        let vector = Vector3::new(-1_f64, 1_f64, -1_f64);
        let expected = Vector3::new(1_f64, 1_f64, -1_f64);
        let result = rotation.inverse_apply_vector(&vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_axis_angle_rotate_point() {
        let axis = Unit::from_value(Vector3::new(-1_f64, -1_f64, 1_f64));
        let angle = Degrees(60_f64);
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let point = Point3::new(-1_f64, -1_f64, 0_f64);
        let expected = Point3::new(-2_f64 / 6_f64,  -8_f64 / 6_f64, 2_f64 / 6_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_axis_angle_rotate_vector() {
        let axis = Unit::from_value(Vector3::new(-1_f64, -1_f64, 1_f64));
        let angle = Degrees(60_f64);
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let point = Point3::new(-1_f64, -1_f64, 0_f64);
        let expected = Point3::new(-2_f64 / 6_f64,  -8_f64 / 6_f64, 2_f64 / 6_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_inverse_of_rotation_is_rotation_by_negative_angle() {
        let axis = Unit::from_value(Vector3::new(-1_f64, -1_f64, 1_f64));
        let angle = Radians::full_turn_div_6();
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let expected = Rotation3::from_axis_angle(&axis, -angle);
        let result = rotation.inverse();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_rotation_between_axis_vectors_point1() {
        let point = Point3::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64, 0_f64);
        let vector1 = Unit::from_value(Vector3::unit_y());
        let vector2 = Unit::from_value(Vector3::unit_x());
        let rotation = Rotation3::rotation_between_axis(&vector1, &vector2).unwrap();
        let expected = Point3::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64, 0_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_between_axis_vectors_point2() {
        let point = Point3::new(0_f64, 1_f64, 0_f64);
        let vector1 = Unit::from_value(Vector3::unit_y());
        let vector2 = Unit::from_value(Vector3::unit_x());
        let rotation = Rotation3::rotation_between_axis(&vector1, &vector2).unwrap();
        let expected = Point3::new(1_f64, 0_f64, 0_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_between_axis_vectors_vector1() {
        let vector = Vector3::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64, 0_f64);
        let vector1 = Unit::from_value(Vector3::unit_y());
        let vector2 = Unit::from_value(Vector3::unit_x());
        let rotation = Rotation3::rotation_between_axis(&vector1, &vector2).unwrap();
        let expected = Vector3::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64, 0_f64);
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_between_axis_vectors_vector2() {
        let vector1: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y());
        let vector2: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x());
        let rotation = Rotation3::rotation_between_axis(&vector1, &vector2).unwrap();
        let vector = Vector3::unit_y();
        let expected = Vector3::unit_x();
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_between_vectors_point() {
        let point = 3_f64 * Point3::new(0_f64, 1_f64, 0_f64);
        let vector1 = 5_f64 * Vector3::unit_y();
        let vector2 = 12_f64 * Vector3::unit_x();
        let rotation = Rotation3::rotation_between(&vector1, &vector2).unwrap();
        let expected = 3_f64 * Point3::new(1_f64, 0_f64, 0_f64);
        let result = rotation.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_between_vectors_vector() {
        let vector = 3_f64 * Vector3::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64, 0_f64);
        let vector1 = 5_f64 * Vector3::unit_y();
        let vector2 = 12_f64 * Vector3::unit_x();
        let rotation = Rotation3::rotation_between(&vector1, &vector2).unwrap();
        let expected = 3_f64 * Vector3::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64, 0_f64);
        let result = rotation.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_look_at_rh_x_axis() {
        let eye = Point3::new(-1_f64, 0_f64, 0_f64);
        let target = Point3::origin();
        let up: Vector3<f64> = Vector3::unit_y();
        let angle = Degrees(90_f64);
        let expected = Rotation3::from_angle_y(angle);
        let result = Rotation3::look_at_rh(&eye, &target, &up);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_look_at_rh_y_axis() {
        let eye = Point3::new(0_f64, -1_f64, 0_f64);
        let target = Point3::origin();
        let direction: Vector3<f64> = Vector3::unit_y();
        let up: Vector3<f64> = Vector3::unit_x();
        let rotation = Rotation3::look_at_rh(&eye, &target, &up);
        let result = rotation.apply_vector(&direction);
        let expected = -Vector3::unit_z();
        
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_look_at_lh_x_axis() {
        let eye = Point3::new(-1_f64, 0_f64, 0_f64);
        let target = Point3::origin();
        let up: Vector3<f64> = Vector3::unit_y();
        let angle = Degrees(-90_f64);
        let expected = Rotation3::from_angle_y(angle);
        let result = Rotation3::look_at_lh(&eye, &target, &up);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_rotation_look_at_lh_y_axis() {
        let eye = Point3::new(0_f64, -1_f64, 0_f64);
        let target = Point3::origin();
        let direction: Vector3<f64> = Vector3::unit_y();
        let up: Vector3<f64> = Vector3::unit_x();
        let rotation = Rotation3::look_at_lh(&eye, &target, &up);
        let result = rotation.apply_vector(&direction);
        let expected = Vector3::unit_z();

        assert_eq!(result, expected);
    }
}

