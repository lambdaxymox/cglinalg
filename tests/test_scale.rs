extern crate cglinalg;


#[cfg(test)]
mod scale2_tests {
    use cglinalg::{
        Scale2,
        Point2,
        Vector2,
        Zero,
    };


    #[test]
    fn test_scale_point() {
        let point = Point2::new(1_f64, 2_f64);
        let scale = Scale2::from_nonuniform_scale(10_f64, 20_f64);
        let expected = Point2::new(10_f64, 40_f64);
        let result = scale.scale_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scale_vector() {
        let vector = Vector2::new(1_f64, 2_f64);
        let scale = Scale2::from_nonuniform_scale(10_f64, 20_f64);
        let expected = Vector2::new(10_f64, 40_f64);
        let result = scale.scale_vector(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_scale_point() {
        let point = Point2::new(10_f64, 40_f64);
        let scale = Scale2::from_nonuniform_scale(10_f64, 20_f64);
        let expected = Point2::new(1_f64, 2_f64);
        let result = scale.inverse_scale_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_scale_vector() {
        let vector = Vector2::new(10_f64, 40_f64);
        let scale = Scale2::from_nonuniform_scale(10_f64, 20_f64);
        let expected = Vector2::new(1_f64, 2_f64);
        let result = scale.inverse_scale_vector(&vector);

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod scale3_tests {
    use cglinalg::{
        Scale3,
        Point3,
        Vector3,
        Zero,
    };


    #[test]
    fn test_scale_point() {
        let point = Point3::new(1_f64, 2_f64, 3_f64);
        let scale = Scale3::from_nonuniform_scale(10_f64, 20_f64, 30_f64);
        let expected = Point3::new(10_f64, 40_f64, 90_f64);
        let result = scale.scale_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scale_vector() {
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let scale = Scale3::from_nonuniform_scale(10_f64, 20_f64, 30_f64);
        let expected = Vector3::new(10_f64, 40_f64, 90_f64);
        let result = scale.scale_vector(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_scale_point() {
        let point = Point3::new(10_f64, 40_f64, 90_f64);
        let scale = Scale3::from_nonuniform_scale(10_f64, 20_f64, 30_f64);
        let expected = Point3::new(1_f64, 2_f64, 3_f64);
        let result = scale.inverse_scale_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_scale_vector() {
        let vector = Vector3::new(10_f64, 40_f64, 90_f64);
        let scale = Scale3::from_nonuniform_scale(10_f64, 20_f64, 30_f64);
        let expected = Vector3::new(1_f64, 2_f64, 3_f64);
        let result = scale.inverse_scale_vector(&vector);

        assert_eq!(result, expected);
    }
}

