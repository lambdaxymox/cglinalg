extern crate cglinalg_trigonometry;
extern crate cglinalg_transform;


#[cfg(test)]
mod similarity2_tests {
    use cglinalg_trigonometry::{
        Angle,
        Degrees,
        Radians,
    };
    use cglinalg_core::{
        Vector2,
        Point2,
        Matrix3x3,
    };
    use cglinalg_transform::{
        Similarity2,
        Isometry2,
        Rotation2,
        Translation2,
    };
    use approx::{
        assert_relative_eq,
    };
    use core::f64;


    #[test]
    fn test_from_translation() {
        let translation = Translation2::new(1_f64, 2_f64);
        let similarity = Similarity2::from_translation(&translation);
        let point = Point2::new(5_f64, 5_f64);
        let expected = Point2::new(6_f64, 7_f64);
        let result = similarity.apply_point(&point);
    
        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_rotation() {
        let angle = Radians(f64::consts::FRAC_PI_4);
        let rotation = Rotation2::from_angle(angle);
        let similarity = Similarity2::from_rotation(&rotation);
        let vector = Vector2::new(2_f64, 0_f64);
        let expected = Vector2::new(f64::sqrt(2_f64), f64::sqrt(2_f64));
        let result = similarity.apply_vector(&vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_angle() {
        let angle = Degrees(90_f64);
        let similarity = Similarity2::from_angle(angle);
        let unit_x = Vector2::unit_x();
        let unit_y = Vector2::unit_y();
        let expected = unit_y;
        let result = similarity.apply_vector(&unit_x);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_isometry() {
        let angle = Radians(f64::consts::FRAC_PI_3);
        let distance = Vector2::new(5_f64, 5_f64);
        let isometry = Isometry2::from_angle_translation(angle, &distance);
        let similarity = Similarity2::from_isometry(&isometry);
        let point = Point2::new(2_f64, 0_f64);
        let expected = Point2::new(6_f64, f64::sqrt(3_f64) + 5_f64);
        let result = similarity.apply_point(&point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[rustfmt::skip]
    #[test]
    fn test_to_affine_matrix() {
        let scale = 2_f64;
        let angle = Degrees(72_f64);
        let rotation = Rotation2::from_angle(angle);
        let translation = Translation2::new(2_f64, 3_f64);
        let similarity = Similarity2::from_parts(&translation, &rotation, scale);
        let expected = Matrix3x3::new(
             scale * angle.cos(), scale * angle.sin(), 0_f64,
            -scale * angle.sin(), scale * angle.cos(), 0_f64,
             2_f64,               3_f64,               1_f64
        );
        let result = similarity.to_affine_matrix();
    
        assert_eq!(result, expected);
    }

    #[test]
    fn test_identity() {
        let similarity = Similarity2::identity();
        let point = Point2::new(1_f64, 2_f64);

        assert_eq!(similarity * point, point);
    }

    #[test]
    fn test_inverse() {
        let scale = 5_f64;
        let angle = Degrees(72_f64);
        let distance = Vector2::new(1_f64, 2_f64);
        let translation = Translation2::from_vector(&distance);
        let rotation = Rotation2::from_angle(angle);
        let similarity = Similarity2::from_parts(&translation, &rotation, scale);
        let similarity_inv = similarity.inverse();
        let point = Point2::new(1_f64, 2_f64);
        let expected = point;
        let transformed_point = similarity.apply_point(&point);
        let result = similarity_inv.apply_point(&transformed_point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_inverse_mut() {
        let scale = 5_f64;
        let angle = Degrees(72_f64);
        let distance = Vector2::new(1_f64, 2_f64);
        let translation = Translation2::from_vector(&distance);
        let rotation = Rotation2::from_angle(angle);
        let similarity = Similarity2::from_parts(&translation, &rotation, scale);
        let mut similarity_mut = similarity;
        similarity_mut.inverse_mut();
        let point = Point2::new(1_f64, 2_f64);
        let expected = point;
        let transformed_point = similarity.apply_point(&point);
        let result = similarity_mut.apply_point(&transformed_point);
        
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_inverse_apply_point() {
        let scale = 12_f64;
        let angle = Radians(f64::consts::FRAC_PI_2);
        let distance = Vector2::new(2_f64, 2_f64);
        let translation = Translation2::from_vector(&distance);
        let rotation = Rotation2::from_angle(angle);
        let similarity = Similarity2::from_parts(&translation, &rotation, scale);
        let point = Point2::new(1_f64, 2_f64);
        let expected = point;
        let transformed_point = similarity.apply_point(&point);
        let result = similarity.inverse_apply_point(&transformed_point);
        
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_inverse_apply_vector() {
        let scale = 12_f64;
        let angle = Radians(f64::consts::FRAC_PI_2);
        let distance = Vector2::new(1_f64, 1_f64);
        let translation = Translation2::from_vector(&distance);
        let rotation = Rotation2::from_angle(angle);
        let similarity = Similarity2::from_parts(&translation, &rotation, scale);
        let vector = Vector2::unit_x();
        let expected = vector;
        let transformed_vector = similarity.apply_vector(&vector);
        let result = similarity.inverse_apply_vector(&transformed_vector);
        
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_apply_point() {
        let scale = 12_f64;
        let angle = Radians(f64::consts::FRAC_PI_2);
        let distance = Vector2::new(2_f64, 2_f64);
        let translation = Translation2::from_vector(&distance);
        let rotation = Rotation2::from_angle(angle);
        let similarity = Similarity2::from_parts(&translation, &rotation, scale);
        let point = Point2::new(1_f64, 2_f64);
        let expected = Point2::new(-22_f64, 14_f64);
        let result = similarity.apply_point(&point);
        
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_apply_vector() {
        let scale = 12_f64;
        let angle = Radians(f64::consts::FRAC_PI_2);
        let distance = Vector2::new(1_f64, 1_f64);
        let translation = Translation2::from_vector(&distance);
        let rotation = Rotation2::from_angle(angle);
        let similarity = Similarity2::from_parts(&translation, &rotation, scale);
        let vector = Vector2::unit_x();
        let expected = scale * Vector2::unit_y();
        let result = similarity.apply_vector(&vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_scale_multiplication() {
        let scale1 = Similarity2::from_scale(12_f64);
        let scale2 = Similarity2::from_scale(34_f64);
        let expected = Similarity2::from_scale(12_f64 * 34_f64);
        let result = scale1 * scale2;

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod similarity3_tests {
    use cglinalg_trigonometry::{
        Degrees,
        Radians,
    };
    use cglinalg_core::{
        Vector3,
        Point3,
        Normed,
        Matrix4x4,
        Unit,
    };
    use cglinalg_transform::{
        Similarity3,
        Isometry3,
        Rotation3,
        Translation3,
    };
    use approx::{
        assert_relative_eq,
    };
    use core::f64;


    #[test]
    fn test_from_rotation() {
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Radians(f64::consts::FRAC_PI_4);
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let similarity = Similarity3::from_rotation(&rotation);
        let vector = Vector3::new(2_f64, 0_f64, 5_f64);
        let expected = Vector3::new(f64::sqrt(2_f64), f64::sqrt(2_f64), 5_f64);
        let result = similarity.apply_vector(&vector);
        
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_scale() {
        let scale = 15_f64;
        let similarity = Similarity3::from_scale(scale);
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector3::new(15_f64, 30_f64, 45_f64);
        let result = similarity.apply_vector(&vector);
    
        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_translation() {
        let distance = Vector3::new(5_f64, 5_f64, 5_f64);
        let translation = Translation3::from_vector(&distance);
        let similarity = Similarity3::from_translation(&translation);
        let point = Point3::new(1_f64, 2_f64, 3_f64);
    
        assert_eq!(similarity * point, point + distance);
    }

    #[test]
    fn test_from_isometry() {
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Radians(f64::consts::FRAC_PI_3);
        let distance = Vector3::new(5_f64, 5_f64, 0_f64);
        let isometry = Isometry3::from_axis_angle_translation(&axis, angle, &distance);
        let similarity = Similarity3::from_isometry(&isometry);
        let point = Point3::new(2_f64, 0_f64, 13_f64);
        let expected = Point3::new(6_f64, f64::sqrt(3_f64) + 5_f64, 13_f64);
        let result = similarity.apply_point(&point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_axis_angle() {
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Radians(f64::consts::FRAC_PI_4);
        let similarity = Similarity3::from_axis_angle(&axis, angle);
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector3::new(-1_f64 / f64::sqrt(2_f64), 3_f64 / f64::sqrt(2_f64), 3_f64);
        let result = similarity.apply_vector(&vector);
        
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_look_to_lh() {
        let eye = Point3::new(1_f64, 2_f64, 3_f64);
        let target = Point3::new(1_f64, -1_f64, 1_f64);
        let direction = (target - eye).normalize();
        let up = Vector3::new(2_f64, 2_f64, 0_f64);
        let isometry = Similarity3::look_to_lh(&eye, &direction, &up);
        let expected = Vector3::unit_z();
        let result = isometry.apply_vector(&direction);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_look_to_rh() {
        let eye = Point3::new(1_f64, 2_f64, 3_f64);
        let target = Point3::new(1_f64, -1_f64, 1_f64);
        let direction = (target - eye).normalize();
        let up = Vector3::new(2_f64, 2_f64, 0_f64);
        let isometry = Similarity3::look_to_rh(&eye, &direction, &up);
        let expected = -Vector3::unit_z();
        let result = isometry.apply_vector(&direction);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_look_at_lh() {
        let target = Point3::new(0_f64, 6_f64, 0_f64);
        let up: Vector3<f64> = Vector3::unit_x();
        let eye = Point3::new(1_f64, 2_f64, 3_f64);
        let similarity = Similarity3::look_at_lh(&eye, &target, &up);
        let expected = Vector3::unit_z();
        let result = similarity.apply_vector(&(target - eye).normalize());
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
        assert_eq!(similarity.apply_point(&eye), Point3::origin());
    }

    #[test]
    fn test_look_at_rh() {
        let target = Point3::new(0_f64, 6_f64, 0_f64);
        let up: Vector3<f64> = Vector3::unit_x();
        let eye = Point3::new(1_f64, 2_f64, 3_f64);
        let similarity = Similarity3::look_at_rh(&eye, &target, &up);
        let expected = -Vector3::unit_z();
        let result = similarity.apply_vector(&(target - eye).normalize());
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
        assert_eq!(similarity.apply_point(&eye), Point3::origin());
    }

    #[rustfmt::skip]
    #[test]
    fn test_to_affine_matrix() {
        let scale = 2_f64;
        let axis = Unit::from_value(Vector3::new(1_f64, 1_f64, 0_f64));
        let angle = Degrees(60_f64);
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let translation = Translation3::new(2_f64, 3_f64, 4_f64);
        let similarity = Similarity3::from_parts(&translation, &rotation, scale);
        let sq_3_8 = f64::sqrt(3_f64 / 8_f64);
        let expected = Matrix4x4::new(
             scale * 3_f64 / 4_f64, scale * 1_f64 / 4_f64, scale * -sq_3_8,       0_f64,
             scale * 1_f64 / 4_f64, scale * 3_f64 / 4_f64, scale *  sq_3_8,       0_f64,
             scale * sq_3_8,        scale * -sq_3_8,       scale * 1_f64 / 2_f64, 0_f64,
             2_f64,                 3_f64,                 4_f64,                 1_f64
        );
        let result = similarity.to_affine_matrix();
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_identity() {
        let similarity = Similarity3::identity();
        let point = Point3::new(1_f64, 2_f64, 3_f64);
    
        assert_eq!(similarity * point, point);
    }

    #[test]
    fn test_inverse() {
        let scale = 5_f64;
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Degrees(72_f64);
        let distance = Vector3::new(6_f64, 7_f64, 8_f64);
        let translation = Translation3::from_vector(&distance);
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let similarity = Similarity3::from_parts(&translation, &rotation, scale);
        let similarity_inv = similarity.inverse();
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = point;
        let transformed_point = similarity.apply_point(&point);
        let result = similarity_inv.apply_point(&transformed_point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_inverse_mut() {
        let scale = 5_f64;
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Degrees(72_f64);
        let distance = Vector3::new(6_f64, 7_f64, 8_f64);
        let translation = Translation3::from_vector(&distance);
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let similarity = Similarity3::from_parts(&translation, &rotation, scale);
        let mut similarity_mut = similarity;
        similarity_mut.inverse_mut();
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = point;
        let transformed_point = similarity.apply_point(&point);
        let result = similarity_mut.apply_point(&transformed_point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_inverse_apply_point() {
        let scale = 12_f64;
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Radians(f64::consts::FRAC_PI_2);
        let distance = Vector3::new(2_f64, 2_f64, 2_f64);
        let translation = Translation3::from_vector(&distance);
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let similarity = Similarity3::from_parts(&translation, &rotation, scale);
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = point;
        let transformed_point = similarity.apply_point(&point);
        let result = similarity.inverse_apply_point(&transformed_point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_inverse_apply_vector() {
        let scale = 12_f64;
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Radians(f64::consts::FRAC_PI_2);
        let distance = Vector3::new(1_f64, 1_f64, 1_f64);
        let translation = Translation3::from_vector(&distance);
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let similarity = Similarity3::from_parts(&translation, &rotation, scale);
        let vector = Vector3::unit_x();
        let expected = vector;
        let transformed_vector = similarity.apply_vector(&vector);
        let result = similarity.inverse_apply_vector(&transformed_vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_apply_point() {
        let scale = 12_f64;
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Radians(f64::consts::FRAC_PI_2);
        let distance = Vector3::new(2_f64, 2_f64, 2_f64);
        let translation = Translation3::from_vector(&distance);
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let similarity = Similarity3::from_parts(&translation, &rotation, scale);
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = Point3::new(-22_f64, 14_f64, 38_f64);
        let result = similarity.apply_point(&point);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_apply_vector() {
        let scale = 12_f64;
        let axis = Unit::from_value(Vector3::unit_z());
        let angle = Radians(f64::consts::FRAC_PI_2);
        let distance = Vector3::new(1_f64, 1_f64, 1_f64);
        let translation = Translation3::from_vector(&distance);
        let rotation = Rotation3::from_axis_angle(&axis, angle);
        let similarity = Similarity3::from_parts(&translation, &rotation, scale);
        let vector = Vector3::unit_x();
        let expected = scale * Vector3::unit_y();
        let result = similarity.apply_vector(&vector);
    
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_scale_multiplication() {
        let scale1 = Similarity3::from_scale(12_f64);
        let scale2 = Similarity3::from_scale(34_f64);
        let expected = Similarity3::from_scale(12_f64 * 34_f64);
        let result = scale1 * scale2;

        assert_eq!(result, expected);
    }
}

