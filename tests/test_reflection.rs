extern crate cglinalg;


#[cfg(test)]
mod reflection2_tests {
    use cglinalg::{
        Reflection2,
        Point2,
        Vector2,
        Unit,
        Zero,
    };


    #[test]
    fn test_reflection_x_line_point1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_y());
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(1_f64, 1_f64);
        let expected = Point2::new(1_f64, -1_f64);
        let result = reflection.reflect_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_reflection_x_line_vector1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_y());
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(1_f64, 1_f64);
        let expected = Vector2::new(1_f64, -1_f64);
        let result = reflection.reflect_vector(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_reflection_x_line_point2() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(1_f64, 1_f64);
        let expected = Point2::new(-1_f64, 1_f64);
        let result = reflection.reflect_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_reflection_x_line_vector2() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(1_f64, 1_f64);
        let expected = Vector2::new(-1_f64, 1_f64);
        let result = reflection.reflect_vector(&vector);

        assert_eq!(result, expected);
    }
}