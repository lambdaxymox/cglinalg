extern crate cglinalg_core;


#[cfg(test)]
mod point1_tests {
    use cglinalg_core::{
        Point1,
        Point2,
        Vector1,
        Vector2,
    };


    #[test]
    fn test_components1() {
        let point = Point1::new(1_i32);

        assert_eq!(point[0], 1_i32);
    }

    #[test]
    fn test_components2() {
        let point = Point1::new(1_i32);

        assert_eq!(point.x, point[0]);
    }

    #[test]
    #[should_panic]
    fn test_point_components_out_of_bounds1() {
        let point = Point1::new(1_i32);

        assert_eq!(point[1], point[1]);
    }

    #[test]
    #[should_panic]
    fn test_point_components_out_of_bounds2() {
        let point = Point1::new(1_i32);

        assert_eq!(point[usize::MAX], point[usize::MAX]);
    }

    #[test]
    fn test_addition() {
        let p = Point1::new(27.6189_f64);
        let v = Vector1::new(258.083_f64);
        let expected = Point1::new(p.x + v.x);
        let result = p + v;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction_point_vector() {
        let p = Point1::new(-23.43_f64);
        let v = Vector1::new(426.1_f64);
        let expected = Point1::new(p.x - v.x);
        let result = p - v;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction_point_point() {
        let p1 = Point1::new(-23.43_f64);
        let p2 = Point1::new(426.1_f64);
        let expected = Vector1::new(p1.x - p2.x);
        let result = p1 - p2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication() {
        let c = 33.249539_f64;
        let p = Point1::from(27.6189_f64);
        let expected = Point1::from(p.x * c);
        let result = p * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division() {
        let c = 33.249539_f64;
        let p = Point1::from(27.6189_f64);
        let expected = Point1::from(p.x / c);
        let result = p / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_point_times_zero_equals_zero() {
        let p = Point1::new(1_f32);

        assert_eq!(p * 0_f32, Point1::new(0_f32));
    }

    #[test]
    fn test_zero_times_point_equals_zero() {
        let p = Point1::new(1_f32);

        assert_eq!(0_f32 * p, Point1::new(0_f32));
    }

    #[test]
    fn test_as_ref() {
        let p = Point1::new(1_i32);
        let p_ref: &[i32; 1] = p.as_ref();

        assert_eq!(p_ref, &[1_i32]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let p = Point1::new(1_i32);

        assert_eq!(p[0], p.x);
    }

    #[test]
    fn test_as_mut() {
        let mut p = Point1::new(1_i32);
        let p_ref: &mut [i32; 1] = p.as_mut();
        p_ref[0] = 5_i32;

        assert_eq!(p.x, 5_i32);
    }

    #[test]
    fn test_zero_point_zero_norm() {
        let zero = Point1::new(0_f32);

        assert_eq!(zero.norm(), 0_f32);
    }

    #[test]
    fn test_point_index_matches_component() {
        let p = Point1::new(1_i32);

        assert_eq!(p.x, p[0]);
    }

    #[test]
    fn test_extend() {
        let point = Point1::new(1_i32);
        let expected = Point2::new(1_i32, 2_i32);
        let result = point.extend(2_i32);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_to_homogeneous() {
        let point = Point1::new(1_f64);
        let expected = Vector2::new(1_f64, 1_f64);
        let result = point.to_homogeneous();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_homogeneous1() {
        let vector = Vector2::new(4_f64, 2_f64);
        let expected = Some(Point1::new(4_f64 / 2_f64));
        let result = Point1::from_homogeneous(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_homogeneous2() {
        let vector = Vector2::new(4_f64, 0_f64);
        let result = Point1::from_homogeneous(&vector);

        assert!(result.is_none());
    }
}


#[cfg(test)]
mod point2_tests {
    use cglinalg_core::{
        Point1,
        Point2,
        Point3,
        Vector2,
        Vector3,
    };


    #[test]
    fn test_components1() {
        let point = Point2::new(1_i32, 2_i32);

        assert_eq!(point[0], 1_i32);
        assert_eq!(point[1], 2_i32);
    }

    #[test]
    fn test_components2() {
        let point = Point2::new(1_i32, 2_i32);

        assert_eq!(point.x, point[0]);
        assert_eq!(point.y, point[1]);
    }

    #[test]
    #[should_panic]
    fn test_point_components_out_of_bounds1() {
        let point = Point2::new(1_i32, 2_i32);

        assert_eq!(point[2], point[2]);
    }

    #[test]
    #[should_panic]
    fn test_point_components_out_of_bounds2() {
        let point = Point2::new(1_i32, 2_i32);

        assert_eq!(point[usize::MAX], point[usize::MAX]);
    }

    #[test]
    fn test_addition() {
        let p = Point2::new(6.741_f64, 23.5724_f64);
        let v = Vector2::new(80_f64, 43.569_f64);
        let expected = Point2::new(p.x + v.x, p.y + v.y);
        let result = p + v;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction_point_vector() {
        let p = Point2::new(6.741_f64, 23.5724_f64);
        let v = Vector2::new(80_f64, 43.569_f64);
        let expected = Point2::new(p.x - v.x, p.y - v.y);
        let result = p - v;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction_point_point() {
        let p1 = Point2::new(6.741_f64, 23.5724_f64);
        let p2 = Point2::new(80_f64, 43.569_f64);
        let expected = Vector2::new(p1.x - p2.x, p1.y - p2.y);
        let result = p1 - p2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication() {
        let c = 7.04217_f64;
        let p = Point2::new(70_f64, 49_f64);
        let expected = Point2::new(p.x * c, p.y * c);
        let result = p * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division() {
        let c = 61.891390_f64;
        let p = Point2::new(89_f64, 936.5_f64);
        let expected = Point2::new(p.x / c, p.y / c);
        let result = p / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_point_times_zero_equals_zero() {
        let p = Point2::new(1_f32, 2_f32);

        assert_eq!(p * 0_f32, Point2::new(0_f32, 0_f32));
    }

    #[test]
    fn test_zero_times_point_equals_zero() {
        let p = Point2::new(1_f32, 2_f32);

        assert_eq!(0_f32 * p, Point2::new(0_f32, 0_f32));
    }

    #[test]
    fn test_as_ref() {
        let p = Point2::new(1_i32, 2_i32);
        let p_ref: &[i32; 2] = p.as_ref();

        assert_eq!(p_ref, &[1_i32, 2_i32]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let p = Point2::new(1_i32, 2_i32);

        assert_eq!(p[0], p.x);
        assert_eq!(p[1], p.y);
    }

    #[test]
    fn test_as_mut() {
        let mut p = Point2::new(1_i32, 2_i32);
        let p_ref: &mut [i32; 2] = p.as_mut();
        p_ref[0] = 5_i32;

        assert_eq!(p.x, 5_i32);
    }

    #[test]
    fn test_zero_point_zero_norm() {
        let zero = Point2::new(0_f32, 0_f32);

        assert_eq!(zero.norm(), 0_f32);
    }

    #[test]
    fn test_point_index_matches_component() {
        let p = Point2::new(1_i32, 2_i32);

        assert_eq!(p.x, p[0]);
        assert_eq!(p.y, p[1]);
    }

    #[test]
    fn test_extend() {
        let point = Point2::new(1_i32, 2_i32);
        let expected = Point3::new(1_i32, 2_i32, 3_i32);
        let result = point.extend(3_i32);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_contract() {
        let point = Point2::new(1_i32, 2_i32);
        let expected = Point1::new(1_i32);
        let result = point.contract();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_to_homogeneous() {
        let point = Point2::new(1_f64, 2_f64);
        let expected = Vector3::new(1_f64, 2_f64, 1_f64);
        let result = point.to_homogeneous();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_homogeneous1() {
        let vector = Vector3::new(4_f64, 6_f64, 2_f64);
        let expected = Some(Point2::new(4_f64 / 2_f64, 6_f64 / 2_f64));
        let result = Point2::from_homogeneous(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_homogeneous2() {
        let vector = Vector3::new(4_f64, 6_f64, 0_f64);
        let result = Point2::from_homogeneous(&vector);

        assert!(result.is_none());
    }
}


#[cfg(test)]
mod point3_tests {
    use cglinalg_core::{
        Point2,
        Point3,
        Vector3,
        Vector4,
    };


    #[test]
    fn test_components1() {
        let point = Point3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(point[0], 1_i32);
        assert_eq!(point[1], 2_i32);
        assert_eq!(point[2], 3_i32);
    }

    #[test]
    fn test_components2() {
        let point = Point3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(point.x, point[0]);
        assert_eq!(point.y, point[1]);
        assert_eq!(point.z, point[2]);
    }

    #[test]
    #[should_panic]
    fn test_point_components_out_of_bounds1() {
        let point = Point3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(point[3], point[3]);
    }

    #[test]
    #[should_panic]
    fn test_point_components_out_of_bounds2() {
        let point = Point3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(point[usize::MAX], point[usize::MAX]);
    }

    #[test]
    fn test_addition() {
        let p = Point3::new(27.6189_f64, 13.90_f64, 4.2219_f64);
        let v = Vector3::new(258.083_f64, 31.70_f64, 42.17_f64);
        let expected = Point3::new(p.x + v.x, p.y + v.y, p.z + v.z);
        let result = p + v;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction_point_vector() {
        let p = Point3::new(70_f64, 49_f64, 95_f64);
        let v = Vector3::new(89.9138_f64, 36.84_f64, 427.46894_f64);
        let expected = Point3::new(p.x - v.x, p.y - v.y, p.z - v.z);
        let result = p - v;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction_point_point() {
        let p1 = Point3::new(8827.1983_f64, 89.5049494_f64, 56.31_f64);
        let p2 = Point3::new(89_f64, 72_f64, 936.5_f64);
        let expected = Vector3::new(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
        let result = p1 - p2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication() {
        let c = 7.04217_f64;
        let p = Point3::new(70_f64, 49_f64, 95_f64);
        let expected = Point3::new(p.x * c, p.y * c, p.z * c);
        let result = p * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division() {
        let c = 802.3435169_f64;
        let p = Point3::new(80_f64, 23.43_f64, 43.569_f64);
        let expected = Point3::new(p.x / c, p.y / c, p.z / c);
        let result = p / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_point_times_zero_equals_zero() {
        let p = Point3::new(1_f32, 2_f32, 3_f32);

        assert_eq!(p * 0_f32, Point3::new(0_f32, 0_f32, 0_f32));
    }

    #[test]
    fn test_zero_times_point_equals_zero() {
        let p = Point3::new(1_f32, 2_f32, 3_f32);

        assert_eq!(0_f32 * p, Point3::new(0_f32, 0_f32, 0_f32));
    }

    #[test]
    fn test_as_ref() {
        let p = Point3::new(1_i32, 2_i32, 3_i32);
        let p_ref: &[i32; 3] = p.as_ref();

        assert_eq!(p_ref, &[1_i32, 2_i32, 3_i32]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let p = Point3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(p[0], p.x);
        assert_eq!(p[1], p.y);
        assert_eq!(p[2], p.z);
    }

    #[test]
    fn test_as_mut() {
        let mut p = Point3::new(1_i32, 2_i32, 3_i32);
        let p_ref: &mut [i32; 3] = p.as_mut();
        p_ref[2] = 5_i32;

        assert_eq!(p.z, 5_i32);
    }

    #[test]
    fn test_zero_point_zero_norm() {
        let zero = Point3::new(0_f32, 0_f32, 0_f32);

        assert_eq!(zero.norm(), 0_f32);
    }

    #[test]
    fn test_point_index_matches_component() {
        let p = Point3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(p.x, p[0]);
        assert_eq!(p.y, p[1]);
        assert_eq!(p.z, p[2]);
    }

    #[test]
    fn test_contract() {
        let point = Point3::new(1_i32, 2_i32, 3_i32);
        let expected = Point2::new(1_i32, 2_i32);
        let result = point.contract();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_homogeneous1() {
        let vector = Vector4::new(4_f64, 6_f64, 8_f64, 2_f64);
        let expected = Some(Point3::new(4_f64 / 2_f64, 6_f64 / 2_f64, 8_f64 / 2_f64));
        let result = Point3::from_homogeneous(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_homogeneous2() {
        let vector = Vector4::new(4_f64, 6_f64, 8_f64, 0_f64);
        let result = Point3::from_homogeneous(&vector);

        assert!(result.is_none());
    }
}
