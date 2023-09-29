extern crate cglinalg_transform;


#[cfg(test)]
mod shear2_tests {
    use cglinalg_core::{
        Point2,
        Vector2,
    };
    use cglinalg_transform::{
        Shear2,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_from_shear_x_point() {
        let shear_x_with_y = 2_f64;
        let shear = Shear2::from_shear_x(shear_x_with_y);
        let point = Point2::new(1_f64, 2_f64);
        let expected = Point2::new(5_f64, 2_f64);
        let result = shear.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_x_vector() {
        let shear_x_with_y = 2_f64;
        let shear = Shear2::from_shear_x(shear_x_with_y);
        let vector = Vector2::new(1_f64, 2_f64);
        let expected = Vector2::new(5_f64, 2_f64);
        let result = shear.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_y_point() {
        let shear_y_with_x = 5_f64;
        let shear = Shear2::from_shear_y(shear_y_with_x);
        let point = Point2::new(1_f64, 2_f64);
        let expected = Point2::new(1_f64, 7_f64);
        let result = shear.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_y_vector() {
        let shear_y_with_x = 5_f64;
        let shear = Shear2::from_shear_y(shear_y_with_x);
        let vector = Vector2::new(1_f64, 2_f64);
        let expected = Vector2::new(1_f64, 7_f64);
        let result = shear.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_point() {
        let shear_x_with_y = 10_f64;
        let shear_y_with_x = 5_f64;
        let shear = Shear2::from_shear(shear_x_with_y, shear_y_with_x);
        let point = Point2::new(1_f64, 2_f64);
        let expected = Point2::new(21_f64, 7_f64);
        let result = shear.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_vector() {
        let shear_x_with_y = 10_f64;
        let shear_y_with_x = 5_f64;
        let shear = Shear2::from_shear(shear_x_with_y, shear_y_with_x);
        let vector = Vector2::new(1_f64, 2_f64);
        let expected = Vector2::new(21_f64, 7_f64);
        let result = shear.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_x_inverse_point() {
        let shear_x_with_y = 2_f64;
        let shear = Shear2::from_shear_x(shear_x_with_y);
        let point = Point2::new(1_f64, 2_f64);
        let expected = Point2::new(-3_f64, 2_f64);
        let result = shear.inverse_apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_x_inverse_vector() {
        let shear_x_with_y = 2_f64;
        let shear = Shear2::from_shear_x(shear_x_with_y);
        let vector = Vector2::new(1_f64, 2_f64);
        let expected = Vector2::new(-3_f64, 2_f64);
        let result = shear.inverse_apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_y_inverse_point() {
        let shear_y_with_x = 5_f64;
        let shear = Shear2::from_shear_y(shear_y_with_x);
        let point = Point2::new(1_f64, 2_f64);
        let expected = Point2::new(1_f64, -3_f64);
        let result = shear.inverse_apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_y_inverse_vector() {
        let shear_y_with_x = 5_f64;
        let shear = Shear2::from_shear_y(shear_y_with_x);
        let vector = Vector2::new(1_f64, 2_f64);
        let expected = Vector2::new(1_f64, -3_f64);
        let result = shear.inverse_apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_inverse_point() {
        let shear_x_with_y = 10_f64;
        let shear_y_with_x = 5_f64;
        let shear = Shear2::from_shear(shear_x_with_y, shear_y_with_x);
        let point = Point2::new(1_f64, 2_f64);
        let expected = Point2::new(19_f64 / 49_f64, 3_f64 / 49_f64);
        let result = shear.inverse_apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_shear_inverse_vector() {
        let shear_x_with_y = 10_f64;
        let shear_y_with_x = 5_f64;
        let shear = Shear2::from_shear(shear_x_with_y, shear_y_with_x);
        let vector = Vector2::new(1_f64, 2_f64);
        let expected = Vector2::new(19_f64 / 49_f64, 3_f64 / 49_f64);
        let result = shear.inverse_apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_multiplication_point() {
        let shear_x_with_y = 10_f64;
        let shear_y_with_x = 5_f64;
        let shear = Shear2::from_shear(shear_x_with_y, shear_y_with_x);
        let point = Point2::new(1_f64, 2_f64);
        let expected = Point2::new(21_f64, 7_f64);
        let result = shear * point;

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_zero_shear_point() {
        let shear = Shear2::identity();
        let point = Point2::new(1_f64, 2_f64);
        let expected = point;
        let result = shear.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_zero_shear_vector() {
        let shear = Shear2::identity();
        let vector = Vector2::new(1_f64, 2_f64);
        let expected = vector;
        let result = shear.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }
}

#[cfg(test)]
mod shear3_tests {
    use cglinalg_core::{
        Point3,
        Vector3,
    };
    use cglinalg_transform::{
        Shear3,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_from_shear_x_point() {
        let shear_x_with_y = 2_f64;
        let shear_x_with_z = 5_f64;
        let shear = Shear3::from_shear_x(shear_x_with_y, shear_x_with_z);
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = Point3::new(20_f64, 2_f64, 3_f64);
        let result = shear.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_x_vector() {
        let shear_x_with_y = 2_f64;
        let shear_x_with_z = 5_f64;
        let shear = Shear3::from_shear_x(shear_x_with_y, shear_x_with_z);
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector3::new(20_f64, 2_f64, 3_f64);
        let result = shear.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_y_point() {
        let shear_y_with_x = 5_f64;
        let shear_y_with_z = 15_f64;
        let shear = Shear3::from_shear_y(shear_y_with_x, shear_y_with_z);
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = Point3::new(1_f64, 52_f64, 3_f64);
        let result = shear.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_y_vector() {
        let shear_y_with_x = 5_f64;
        let shear_y_with_z = 15_f64;
        let shear = Shear3::from_shear_y(shear_y_with_x, shear_y_with_z);
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector3::new(1_f64, 52_f64, 3_f64);
        let result = shear.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_z_point() {
        let shear_z_with_x = 5_f64;
        let shear_z_with_y = 15_f64;
        let shear = Shear3::from_shear_z(shear_z_with_x, shear_z_with_y);
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = Point3::new(1_f64, 2_f64, 38_f64);
        let result = shear.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_z_vector() {
        let shear_z_with_x = 5_f64;
        let shear_z_with_y = 15_f64;
        let shear = Shear3::from_shear_z(shear_z_with_x, shear_z_with_y);
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector3::new(1_f64, 2_f64, 38_f64);
        let result = shear.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_point() {
        let shear_x_with_y = 1_f64;
        let shear_x_with_z = 1_f64;
        let shear_y_with_x = 2_f64;
        let shear_y_with_z = 2_f64;
        let shear_z_with_x = 3_f64;
        let shear_z_with_y = 3_f64;
        let shear = Shear3::from_shear(
            shear_x_with_y, 
            shear_x_with_z, 
            shear_y_with_x, 
            shear_y_with_z, 
            shear_z_with_x, 
            shear_z_with_y
        );
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = Point3::new(6_f64, 10_f64, 12_f64);
        let result = shear.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_vector() {
        let shear_x_with_y = 1_f64;
        let shear_x_with_z = 1_f64;
        let shear_y_with_x = 2_f64;
        let shear_y_with_z = 2_f64;
        let shear_z_with_x = 3_f64;
        let shear_z_with_y = 3_f64;
        let shear = Shear3::from_shear(
            shear_x_with_y, 
            shear_x_with_z, 
            shear_y_with_x, 
            shear_y_with_z, 
            shear_z_with_x, 
            shear_z_with_y
        );
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector3::new(6_f64, 10_f64, 12_f64);
        let result = shear.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_x_inverse_point() {
        let shear_x_with_y = 2_f64;
        let shear_x_with_z = 5_f64;
        let shear = Shear3::from_shear_x(shear_x_with_y, shear_x_with_z);
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = Point3::new(-18_f64, 2_f64, 3_f64);
        let result = shear.inverse_apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_x_inverse_vector() {
        let shear_x_with_y = 2_f64;
        let shear_x_with_z = 5_f64;
        let shear = Shear3::from_shear_x(shear_x_with_y, shear_x_with_z);
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector3::new(-18_f64, 2_f64, 3_f64);
        let result = shear.inverse_apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_y_inverse_point() {
        let shear_y_with_x = 3_f64;
        let shear_y_with_z = 5_f64;
        let shear = Shear3::from_shear_y(shear_y_with_x, shear_y_with_z);
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = Point3::new(1_f64, -16_f64, 3_f64);
        let result = shear.inverse_apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_y_inverse_vector() {
        let shear_y_with_x = 3_f64;
        let shear_y_with_z = 5_f64;
        let shear = Shear3::from_shear_y(shear_y_with_x, shear_y_with_z);
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector3::new(1_f64, -16_f64, 3_f64);
        let result = shear.inverse_apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_z_inverse_point() {
        let shear_z_with_x = 3_f64;
        let shear_z_with_y = 5_f64;
        let shear = Shear3::from_shear_z(shear_z_with_x, shear_z_with_y);
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = Point3::new(1_f64, 2_f64, -10_f64);
        let result = shear.inverse_apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_z_inverse_vector() {
        let shear_z_with_x = 3_f64;
        let shear_z_with_y = 5_f64;
        let shear = Shear3::from_shear_z(shear_z_with_x, shear_z_with_y);
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector3::new(1_f64, 2_f64, -10_f64);
        let result = shear.inverse_apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_from_shear_inverse_point() {
        let shear_x_with_y = 1_f64;
        let shear_x_with_z = 2_f64;
        let shear_y_with_x = 3_f64;
        let shear_y_with_z = 4_f64;
        let shear_z_with_x = 5_f64;
        let shear_z_with_y = 6_f64;
        let shear = Shear3::from_shear(
            shear_x_with_y, 
            shear_x_with_z, 
            shear_y_with_x, 
            shear_y_with_z, 
            shear_z_with_x, 
            shear_z_with_y
        );
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = Point3::new(1_f64 / 4_f64, 1_f64 / 4_f64, 1_f64 / 4_f64);
        let result = shear.inverse_apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_from_shear_inverse_vector() {
        let shear_x_with_y = 1_f64;
        let shear_x_with_z = 2_f64;
        let shear_y_with_x = 3_f64;
        let shear_y_with_z = 4_f64;
        let shear_z_with_x = 5_f64;
        let shear_z_with_y = 6_f64;
        let shear = Shear3::from_shear(
            shear_x_with_y, 
            shear_x_with_z, 
            shear_y_with_x, 
            shear_y_with_z, 
            shear_z_with_x, 
            shear_z_with_y
        );
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector3::new(1_f64 / 4_f64, 1_f64 / 4_f64, 1_f64 / 4_f64);
        let result = shear.inverse_apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_multiplication_point() {
        let shear_x_with_y = 1_f64;
        let shear_x_with_z = 2_f64;
        let shear_y_with_x = 3_f64;
        let shear_y_with_z = 4_f64;
        let shear_z_with_x = 5_f64;
        let shear_z_with_y = 6_f64;
        let shear = Shear3::from_shear(
            shear_x_with_y, 
            shear_x_with_z, 
            shear_y_with_x, 
            shear_y_with_z, 
            shear_z_with_x, 
            shear_z_with_y
        );
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = Point3::new(9_f64, 17_f64, 20_f64);
        let result = shear * point;

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_zero_shear_point() {
        let shear = Shear3::identity();
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let expected = point;
        let result = shear.apply_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }

    #[test]
    fn test_zero_shear_vector() {
        let shear = Shear3::identity();
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = vector;
        let result = shear.apply_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10)
    }
}

