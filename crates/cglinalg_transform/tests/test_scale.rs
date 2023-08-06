extern crate cglinalg_transform;


#[cfg(test)]
mod scale2_tests {
    use cglinalg_core::{
        Point2,
        Vector2,
    };
    use cglinalg_transform::{
        Scale2,
    };
    use approx::{
        assert_relative_eq,
    };
    


    #[test]
    fn test_scale_point() {
        let point = Point2::new(1_f64, 2_f64);
        let scale = Scale2::from_nonuniform_scale(10_f64, 20_f64);
        let expected = Point2::new(10_f64, 40_f64);
        let result = scale.scale_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_scale_vector() {
        let vector = Vector2::new(1_f64, 2_f64);
        let scale = Scale2::from_nonuniform_scale(10_f64, 20_f64);
        let expected = Vector2::new(10_f64, 40_f64);
        let result = scale.scale_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_scale_point() {
        let point = Point2::new(10_f64, 40_f64);
        let scale = Scale2::from_nonuniform_scale(10_f64, 20_f64);
        let expected = Point2::new(1_f64, 2_f64);
        let result = scale.inverse_scale_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_scale_vector() {
        let vector = Vector2::new(10_f64, 40_f64);
        let scale = Scale2::from_nonuniform_scale(10_f64, 20_f64);
        let expected = Vector2::new(1_f64, 2_f64);
        let result = scale.inverse_scale_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod scale3_tests {
    use cglinalg_core::{
        Point3,
        Vector3,
    };
    use cglinalg_transform::{
        Scale3,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_scale_point() {
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let scale = Scale3::from_nonuniform_scale(10_f64, 20_f64, 30_f64);
        let expected = Point3::new(10_f64, 40_f64, 90_f64);
        let result = scale.scale_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_scale_vector() {
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let scale = Scale3::from_nonuniform_scale(10_f64, 20_f64, 30_f64);
        let expected = Vector3::new(10_f64, 40_f64, 90_f64);
        let result = scale.scale_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_scale_point() {
        let point = Point3::new(10_f64, 40_f64, 90_f64);
        let scale = Scale3::from_nonuniform_scale(10_f64, 20_f64, 30_f64);
        let expected = Point3::new(1_f64, 2_f64, 3_f64);
        let result = scale.inverse_scale_point(&point);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_scale_vector() {
        let vector = Vector3::new(10_f64, 40_f64, 90_f64);
        let scale = Scale3::from_nonuniform_scale(10_f64, 20_f64, 30_f64);
        let expected = Vector3::new(1_f64, 2_f64, 3_f64);
        let result = scale.inverse_scale_vector(&vector);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod scale2_composition_tests {
    use cglinalg_transform::{
        Scale2,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_scale_multiplication_nonuniform() {
        let scale1 = Scale2::from_nonuniform_scale(1_f64, 2_f64);
        let scale2 = Scale2::from_nonuniform_scale(3_f64, 4_f64);
        let expected = Scale2::from_nonuniform_scale(3_f64, 8_f64);
        let result = scale1 * scale2;

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod scale3_composition_tests {
    use cglinalg_transform::{
        Scale3,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_scale_multiplication_nonuniform() {
        let scale1 = Scale3::from_nonuniform_scale(1_f64, 2_f64, 10_f64);
        let scale2 = Scale3::from_nonuniform_scale(3_f64, 4_f64, 32_f64);
        let expected = Scale3::from_nonuniform_scale(3_f64, 8_f64, 320_f64);
        let result = scale1 * scale2;

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }
}

