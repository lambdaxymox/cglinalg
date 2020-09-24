extern crate cglinalg;


#[cfg(test)]
mod point1_tests {
    use cglinalg::{
        Point1,
        Vector1,
        Magnitude,
        Zero,
    };
    use core::slice::Iter;


    #[test]
    fn test_addition() {
        let p = Point1::new(27.6189);
        let v = Vector1::new(258.083);
        let expected = Point1::new(p.x + v.x);
        let result = p + v;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction_point_vector() {
        let p = Point1::new(-23.43);
        let v = Vector1::new(426.1);
        let expected = Point1::new(p.x - v.x);
        let result = p - v;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction_point_point() {
        let p1 = Point1::new(-23.43);
        let p2 = Point1::new(426.1);
        let expected = Vector1::new(p1.x - p2.x);
        let result = p1 - p2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication() {
        let c = 33.249539; 
        let p = Point1::from(27.6189);
        let expected = Point1::from(p.x * c);
        let result = p * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division() {
        let c = 33.249539; 
        let p = Point1::from(27.6189);
        let expected = Point1::from(p.x / c);
        let result = p / c;

        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_out_of_bounds_array_access() {
        let p = Point1::new(1_f32);

        assert_eq!(p[1], p[1]);
    }

    #[test]
    fn test_vector_times_zero_equals_zero() {
        let p = Point1::new(1_f32);

        assert_eq!(p * 0_f32, Point1::new(0_f32));
    }

    #[test]
    fn test_zero_times_vector_equals_zero() {
        let p = Point1::new(1_f32);

        assert_eq!(0_f32 * p, Point1::new(0_f32));
    }

    #[test]
    fn test_as_ref() {
        let p: Point1<i32> = Point1::new(1);
        let p_ref: &[i32; 1] = p.as_ref();

        assert_eq!(p_ref, &[1]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let p = Point1::new(1);

        assert_eq!(p[0], p.x);
    }

    #[test]
    fn test_as_mut() {
        let mut p: Point1<i32> = Point1::new(1);
        let p_ref: &mut [i32; 1] = p.as_mut();
        p_ref[0] = 5;

        assert_eq!(p.x, 5);
    }

    #[test]
    fn test_zero_vector_zero_magnitude() {
        let zero: Point1<f32> = Point1::new(0_f32);

        assert_eq!(zero.magnitude(), 0_f32);
    }

    #[test]
    fn test_vector_index_matches_component() {
        let p = Point1::new(1);

        assert_eq!(p.x, p[0]);
    }
}

