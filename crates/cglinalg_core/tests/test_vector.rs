#[cfg(test)]
mod vector1_tests {
    use cglinalg_core::{
        Vector1,
        Vector2,
    };

    #[test]
    fn test_components1() {
        let vector = Vector1::new(1_i32);

        assert_eq!(vector[0], 1_i32);
    }

    #[test]
    fn test_components2() {
        let vector = Vector1::new(1_i32);

        assert_eq!(vector.x, vector[0]);
    }

    #[test]
    #[should_panic]
    fn test_vector_components_out_of_bounds1() {
        let vector = Vector1::new(1_f32);

        assert_eq!(vector[1], vector[1]);
    }

    #[test]
    #[should_panic]
    fn test_vector_components_out_of_bounds2() {
        let vector = Vector1::new(1_f32);

        assert_eq!(vector[usize::MAX], vector[usize::MAX]);
    }

    #[test]
    fn test_addition1() {
        // let c = 802.3435169_f32;
        let vector1 = Vector1::from(-23.43_f32);
        let vector2 = Vector1::from(426.1_f32);
        let expected = Vector1::from(vector1.x + vector2.x);
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction1() {
        // let c = 802.3435169_f32;
        let vector1 = Vector1::from(-23.43_f32);
        let vector2 = Vector1::from(426.1_f32);
        let expected = Vector1::from(vector1.x - vector2.x);
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication1() {
        let c = 802.3435169_f32;
        let vector1 = Vector1::from(-23.43_f32);
        // let vector2 = Vector1::from(426.1_f32);
        let expected = Vector1::from(vector1.x * c);
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division1() {
        let c = 802.3435169_f32;
        let vector1 = Vector1::from(-23.43_f32);
        // let vector2 = Vector1::from(426.1_f32);
        let expected = Vector1::from(vector1.x / c);
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition2() {
        // let c = 33.249539_f32;
        let vector1 = Vector1::from(27.6189_f32);
        let vector2 = Vector1::from(258.083_f32);
        let expected = Vector1::from(vector1.x + vector2.x);
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction2() {
        // let c = 33.249539_f32;
        let vector1 = Vector1::from(27.6189_f32);
        let vector2 = Vector1::from(258.083_f32);
        let expected = Vector1::from(vector1.x - vector2.x);
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication2() {
        let c = 33.249539_f32;
        let vector1 = Vector1::from(27.6189_f32);
        // let vector2 = Vector1::from(258.083_f32);
        let expected = Vector1::from(vector1.x * c);
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division2() {
        let c = 33.249539_f32;
        let vector1 = Vector1::from(27.6189_f32);
        // let vector2 = Vector1::from(258.083_f32);
        let expected = Vector1::from(vector1.x / c);
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition3() {
        // let c = 7.04217_f32;
        let vector1 = Vector1::from(0_f32);
        let vector2 = Vector1::from(0_f32);
        let expected = Vector1::from(vector1.x + vector2.x);
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction3() {
        // let c = 7.04217_f32;
        let vector1 = Vector1::from(0_f32);
        let vector2 = Vector1::from(0_f32);
        let expected = Vector1::from(vector1.x - vector2.x);
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication3() {
        let c = 7.04217_f32;
        let vector1 = Vector1::from(0_f32);
        // let vector2 = Vector1::from(0_f32);
        let expected = Vector1::from(vector1.x * c);
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division3() {
        let c = 7.04217_f32;
        let vector1 = Vector1::from(0_f32);
        // let vector2 = Vector1::from(0_f32);
        let expected = Vector1::from(vector1.x / c);
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition4() {
        // let c = 61.891390_f32;
        let vector1 = Vector1::from(8827.1983_f32);
        let vector2 = Vector1::from(89_f32);
        let expected = Vector1::from(vector1.x + vector2.x);
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction4() {
        // let c = 61.891390_f32;
        let vector1 = Vector1::from(8827.1983_f32);
        let vector2 = Vector1::from(89_f32);
        let expected = Vector1::from(vector1.x - vector2.x);
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication4() {
        let c = 61.891390_f32;
        let vector1 = Vector1::from(8827.1983_f32);
        // let vector2 = Vector1::from(89_f32);
        let expected = Vector1::from(c * vector1.x);
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division4() {
        let c = 61.891390_f32;
        let vector1 = Vector1::from(8827.1983_f32);
        // let vector2 = Vector1::from(89_f32);
        let expected = Vector1::from(vector1.x / c);
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_vector_times_zero_equals_zero() {
        let vector = Vector1::new(1_f32);

        assert_eq!(vector * 0_f32, Vector1::zero());
    }

    #[test]
    fn test_zero_times_vector_equals_zero() {
        let vector = Vector1::new(1_f32);

        assert_eq!(0_f32 * vector, Vector1::zero());
    }

    #[test]
    fn test_as_ref() {
        let vector = Vector1::new(1_i32);
        let vector_ref: &[i32; 1] = vector.as_ref();

        assert_eq!(vector_ref, &[1]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let vector = Vector1::new(1_i32);

        assert_eq!(vector[0], vector.x);
    }

    #[test]
    fn test_as_mut() {
        let mut vector = Vector1::new(1_i32);
        let vector_ref: &mut [i32; 1] = vector.as_mut();
        vector_ref[0] = 5_i32;

        assert_eq!(vector.x, 5);
    }

    #[test]
    fn test_vector_addition_over_integers_commutative() {
        let vector1 = Vector1::new(2_i32);
        let vector2 = Vector1::new(3_i32);

        assert_eq!(vector1 + vector2, vector2 + vector1);
    }

    #[test]
    fn test_negative_zero_equals_positive_zero() {
        let zero: Vector1<f32> = Vector1::zero();

        assert_eq!(zero, -zero);
    }

    #[test]
    fn test_zero_vector_zero_norm() {
        let zero: Vector1<f32> = Vector1::zero();

        assert_eq!(zero.norm(), 0_f32);
    }

    #[test]
    fn test_vector_index_matches_component() {
        let vector = Vector1::new(1_i32);

        assert_eq!(vector.x, vector[0]);
    }

    #[test]
    fn test_norm() {
        let vector = Vector1::new(4_f64);
        let expected = 4_f64;
        let result = vector.norm();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_norm_unit_vectors() {
        let unit_x: Vector1<f64> = Vector1::unit_x();

        assert_eq!(unit_x.norm_squared(), 1_f64);
        assert_eq!(unit_x.norm(), 1_f64);
    }

    #[test]
    fn test_extend() {
        let vector = Vector1::new(1_i32);
        let expected = Vector2::new(1_i32, 2_i32);
        let result = vector.extend(2_i32);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_to_homogeneous() {
        let vector = Vector1::new(1_f64);
        let expected = Vector2::new(1_f64, 0_f64);
        let result = vector.to_homogeneous();

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod vector2_tests {
    use cglinalg_core::{
        Vector1,
        Vector2,
        Vector3,
    };

    #[test]
    fn test_components1() {
        let vector = Vector2::new(1_i32, 2_i32);

        assert_eq!(vector[0], 1_i32);
        assert_eq!(vector[1], 2_i32);
    }

    #[test]
    fn test_components2() {
        let vector = Vector2::new(1_i32, 2_i32);

        assert_eq!(vector.x, vector[0]);
        assert_eq!(vector.y, vector[1]);
    }

    #[test]
    #[should_panic]
    fn test_vector_components_out_of_bounds1() {
        let vector = Vector2::new(1_f32, 2_f32);

        assert_eq!(vector[2], vector[2]);
    }

    #[test]
    #[should_panic]
    fn test_vector_components_out_of_bounds2() {
        let vector = Vector2::new(1_f32, 2_f32);

        assert_eq!(vector[usize::MAX], vector[usize::MAX]);
    }

    #[test]
    fn test_addition1() {
        // let c = 802.3435169_f32;
        let vector1 = Vector2::from((80_f32, 43.569_f32));
        let vector2 = Vector2::from((6.741_f32, 23.5724_f32));
        let expected = Vector2::from((vector1.x + vector2.x, vector1.y + vector2.y));
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction1() {
        // let c = 802.3435169_f32;
        let vector1 = Vector2::from((80_f32, 43.569_f32));
        let vector2 = Vector2::from((6.741_f32, 23.5724_f32));
        let expected = Vector2::from((vector1.x - vector2.x, vector1.y - vector2.y));
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication1() {
        let c = 802.3435169_f32;
        let vector1 = Vector2::from((80_f32, 43.569_f32));
        // let vector2 = Vector2::from((6.741_f32, 23.5724_f32));
        let expected = Vector2::from((c * vector1.x, c * vector1.y));
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division1() {
        let c = 802.3435169_f32;
        let vector1 = Vector2::from((80_f32, 43.569_f32));
        // let vector2 = Vector2::from((6.741_f32, 23.5724_f32));
        let expected = Vector2::from((vector1.x / c, vector1.y / c));
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition2() {
        // let c = 33.249539_f32;
        let vector1 = Vector2::from((27.6189_f32, 4.2219_f32));
        let vector2 = Vector2::from((258.083_f32, 42.17_f32));
        let expected = Vector2::from((vector1.x + vector2.x, vector1.y + vector2.y));
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction2() {
        // let c = 33.249539_f32;
        let vector1 = Vector2::from((27.6189_f32, 4.2219_f32));
        let vector2 = Vector2::from((258.083_f32, 42.17_f32));
        let expected = Vector2::from((vector1.x - vector2.x, vector1.y - vector2.y));
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication2() {
        let c = 33.249539_f32;
        let vector1 = Vector2::from((27.6189_f32, 4.2219_f32));
        // let vector2 = Vector2::from((258.083_f32, 42.17_f32));
        let expected = Vector2::from((c * vector1.x, c * vector1.y));
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division2() {
        let c = 33.249539_f32;
        let vector1 = Vector2::from((27.6189_f32, 4.2219_f32));
        // let vector2 = Vector2::from((258.083_f32, 42.17_f32));
        let expected = Vector2::from((vector1.x / c, vector1.y / c));
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition3() {
        // let c = 7.04217_f32;
        let vector1 = Vector2::from((70_f32, 49_f32));
        let vector2 = Vector2::from((89.9138_f32, 427.46894_f32));
        let expected = Vector2::from((vector1.x + vector2.x, vector1.y + vector2.y));
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction3() {
        // let c = 7.04217_f32;
        let vector1 = Vector2::from((70_f32, 49_f32));
        let vector2 = Vector2::from((89.9138_f32, 427.46894_f32));
        let expected = Vector2::from((vector1.x - vector2.x, vector1.y - vector2.y));
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication3() {
        let c = 7.04217_f32;
        let vector1 = Vector2::from((70_f32, 49_f32));
        // let vector2 = Vector2::from((89.9138_f32, 427.46894_f32));
        let expected = Vector2::from((c * vector1.x, c * vector1.y));
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division3() {
        let c = 7.04217_f32;
        let vector1 = Vector2::from((70_f32, 49_f32));
        // let vector2 = Vector2::from((89.9138_f32, 427.46894_f32));
        let expected = Vector2::from((vector1.x / c, vector1.y / c));
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition4() {
        // let c = 61.891390_f32;
        let vector1 = Vector2::from((8827.1983_f32, 56.31_f32));
        let vector2 = Vector2::from((89_f32, 936.5_f32));
        let expected = Vector2::from((vector1.x + vector2.x, vector1.y + vector2.y));
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction4() {
        // let c = 61.891390_f32;
        let vector1 = Vector2::from((8827.1983_f32, 56.31_f32));
        let vector2 = Vector2::from((89_f32, 936.5_f32));
        let expected = Vector2::from((vector1.x - vector2.x, vector1.y - vector2.y));
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication4() {
        let c = 61.891390_f32;
        let vector1 = Vector2::from((8827.1983_f32, 56.31_f32));
        // let vector2 = Vector2::from((89_f32, 936.5_f32));
        let expected = Vector2::from((c * vector1.x, c * vector1.y));
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division4() {
        let c = 61.891390_f32;
        let vector1 = Vector2::from((8827.1983_f32, 56.31_f32));
        // let vector2 = Vector2::from((89_f32, 936.5_f32));
        let expected = Vector2::from((vector1.x / c, vector1.y / c));
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_vector_times_zero_equals_zero() {
        let vector = Vector2::new(1_f32, 2_f32);

        assert_eq!(vector * 0_f32, Vector2::zero());
    }

    #[test]
    fn test_zero_times_vector_equals_zero() {
        let vector = Vector2::new(1_f32, 2_f32);

        assert_eq!(0_f32 * vector, Vector2::zero());
    }

    #[test]
    fn test_as_ref() {
        let vector = Vector2::new(1_i32, 2_i32);
        let vector_ref: &[i32; 2] = vector.as_ref();

        assert_eq!(vector_ref, &[1, 2]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let vector = Vector2::new(1_i32, 2_i32);

        assert_eq!(vector[0], vector.x);
        assert_eq!(vector[1], vector.y);
    }

    #[test]
    fn test_as_mut() {
        let mut vector = Vector2::new(1_i32, 2_i32);
        let vector_ref: &mut [i32; 2] = vector.as_mut();
        vector_ref[0] = 5;

        assert_eq!(vector.x, 5);
    }

    #[test]
    fn test_vector_addition_over_integers_commutative() {
        let vector1 = Vector2::new(2_i32, 3_i32);
        let vector2 = Vector2::new(4_i32, 5_i32);

        assert_eq!(vector1 + vector2, vector2 + vector1);
    }

    #[test]
    fn test_negative_zero_equals_positive_zero() {
        let zero: Vector2<f32> = Vector2::zero();

        assert_eq!(zero, -zero);
    }

    #[test]
    fn test_zero_vector_zero_norm() {
        let zero: Vector2<f32> = Vector2::zero();

        assert_eq!(zero.norm(), 0_f32);
    }

    #[test]
    fn test_vector_index_matches_component() {
        let vector = Vector2::new(1_i32, 2_i32);

        assert_eq!(vector.x, vector[0]);
        assert_eq!(vector.y, vector[1]);
    }

    #[test]
    fn test_norm1() {
        let vector = Vector2::new(4_f64, 0_f64);
        let expected = 4_f64;
        let result = vector.norm();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_norm2() {
        let vector = Vector2::new(0_f64, 4_f64);
        let expected = 4_f64;
        let result = vector.norm();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_norm_unit_vectors() {
        let unit_x: Vector2<f64> = Vector2::unit_x();
        let unit_y: Vector2<f64> = Vector2::unit_y();

        assert_eq!(unit_x.norm_squared(), 1_f64);
        assert_eq!(unit_x.norm(), 1_f64);
        assert_eq!(unit_y.norm_squared(), 1_f64);
        assert_eq!(unit_y.norm(), 1_f64);
    }

    #[test]
    fn test_extend() {
        let vector = Vector2::new(1_i32, 2_i32);
        let expected = Vector3::new(1_i32, 2_i32, 3_i32);
        let result = vector.extend(3_i32);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_contract() {
        let vector = Vector2::new(1_i32, 2_i32);
        let expected = Vector1::new(1_i32);
        let result = vector.contract();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_to_homogeneous() {
        let vector = Vector2::new(1_f64, 2_f64);
        let expected = Vector3::new(1_f64, 2_f64, 0_f64);
        let result = vector.to_homogeneous();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_homogeneous1() {
        let vector = Vector2::new(1_f64, 0_f64);
        let expected = Some(Vector1::new(1_f64));
        let result = vector.from_homogeneous();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_homogeneous2() {
        let vector = Vector2::new(1_f64, 2_f64);
        let result = vector.from_homogeneous();

        assert!(result.is_none());
    }
}

#[cfg(test)]
mod vector3_tests {
    use cglinalg_core::{
        Vector2,
        Vector3,
        Vector4,
    };

    #[test]
    fn test_components1() {
        let vector = Vector3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(vector[0], 1_i32);
        assert_eq!(vector[1], 2_i32);
        assert_eq!(vector[2], 3_i32);
    }

    #[test]
    fn test_components2() {
        let vector = Vector3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(vector.x, vector[0]);
        assert_eq!(vector.y, vector[1]);
        assert_eq!(vector.z, vector[2]);
    }

    #[test]
    #[should_panic]
    fn test_vector_components_out_of_bounds1() {
        let vector = Vector3::new(1_f32, 2_f32, 3_f32);

        assert_eq!(vector[3], vector[3]);
    }

    #[test]
    #[should_panic]
    fn test_vector_components_out_of_bounds2() {
        let vector = Vector3::new(1_f32, 2_f32, 3_f32);

        assert_eq!(vector[usize::MAX], vector[usize::MAX]);
    }

    #[test]
    fn test_addition1() {
        // let c = 802.3435169_f32;
        let vector1 = Vector3::from((80_f32, 23.43_f32, 43.569_f32));
        let vector2 = Vector3::from((6.741_f32, 426.1_f32, 23.5724_f32));
        let expected = Vector3::from((vector1.x + vector2.x, vector1.y + vector2.y, vector1.z + vector2.z));
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction1() {
        // let c = 802.3435169_f32;
        let vector1 = Vector3::from((80_f32, 23.43_f32, 43.569_f32));
        let vector2 = Vector3::from((6.741_f32, 426.1_f32, 23.5724_f32));
        let expected = Vector3::from((vector1.x - vector2.x, vector1.y - vector2.y, vector1.z - vector2.z));
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication1() {
        let c = 802.3435169_f32;
        let vector1 = Vector3::from((80_f32, 23.43_f32, 43.569_f32));
        // let vector2 = Vector3::from((6.741_f32, 426.1_f32, 23.5724_f32));
        let expected = Vector3::from((c * vector1.x, c * vector1.y, c * vector1.z));
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division1() {
        let c = 802.3435169_f32;
        let vector1 = Vector3::from((80_f32, 23.43_f32, 43.569_f32));
        // let vector2 = Vector3::from((6.741_f32, 426.1_f32, 23.5724_f32));
        let expected = Vector3::from((vector1.x / c, vector1.y / c, vector1.z / c));
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition2() {
        // let c = 33.249539_f32;
        let vector1 = Vector3::from((27.6189_f32, 13.90_f32, 4.2219_f32));
        let vector2 = Vector3::from((258.083_f32, 31.70_f32, 42.17_f32));
        let expected = Vector3::from((vector1.x + vector2.x, vector1.y + vector2.y, vector1.z + vector2.z));
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction2() {
        // let c = 33.249539_f32;
        let vector1 = Vector3::from((27.6189_f32, 13.90_f32, 4.2219_f32));
        let vector2 = Vector3::from((258.083_f32, 31.70_f32, 42.17_f32));
        let expected = Vector3::from((vector1.x - vector2.x, vector1.y - vector2.y, vector1.z - vector2.z));
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication2() {
        let c = 33.249539_f32;
        let vector1 = Vector3::from((27.6189_f32, 13.90_f32, 4.2219_f32));
        // let vector2 = Vector3::from((258.083_f32, 31.70_f32, 42.17_f32));
        let expected = Vector3::from((c * vector1.x, c * vector1.y, c * vector1.z));
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division2() {
        let c = 33.249539_f32;
        let vector1 = Vector3::from((27.6189_f32, 13.90_f32, 4.2219_f32));
        // let vector2 = Vector3::from((258.083_f32, 31.70_f32, 42.17_f32));
        let expected = Vector3::from((vector1.x / c, vector1.y / c, vector1.z / c));
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition3() {
        // let c = 7.04217_f32;
        let vector1 = Vector3::from((70_f32, 49_f32, 95_f32));
        let vector2 = Vector3::from((89.9138_f32, 36.84_f32, 427.46894_f32));
        let expected = Vector3::from((vector1.x + vector2.x, vector1.y + vector2.y, vector1.z + vector2.z));
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction3() {
        // let c = 7.04217_f32;
        let vector1 = Vector3::from((70_f32, 49_f32, 95_f32));
        let vector2 = Vector3::from((89.9138_f32, 36.84_f32, 427.46894_f32));
        let expected = Vector3::from((vector1.x - vector2.x, vector1.y - vector2.y, vector1.z - vector2.z));
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication3() {
        let c = 7.04217_f32;
        let vector1 = Vector3::from((70_f32, 49_f32, 95_f32));
        // let vector2 = Vector3::from((89.9138_f32, 36.84_f32, 427.46894_f32));
        let expected = Vector3::from((c * vector1.x, c * vector1.y, c * vector1.z));
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division3() {
        let c = 7.04217_f32;
        let vector1 = Vector3::from((70_f32, 49_f32, 95_f32));
        // let vector2 = Vector3::from((89.9138_f32, 36.84_f32, 427.46894_f32));
        let expected = Vector3::from((vector1.x / c, vector1.y / c, vector1.z / c));
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition4() {
        // let c = 61.891390_f32;
        let vector1 = Vector3::from((8827.1983_f32, 89.5049494_f32, 56.31_f32));
        let vector2 = Vector3::from((89_f32, 72_f32, 936.5_f32));
        let expected = Vector3::from((vector1.x + vector2.x, vector1.y + vector2.y, vector1.z + vector2.z));
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction4() {
        // let c = 61.891390_f32;
        let vector1 = Vector3::from((8827.1983_f32, 89.5049494_f32, 56.31_f32));
        let vector2 = Vector3::from((89_f32, 72_f32, 936.5_f32));
        let expected = Vector3::from((vector1.x - vector2.x, vector1.y - vector2.y, vector1.z - vector2.z));
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication4() {
        let c = 61.891390_f32;
        let vector1 = Vector3::from((8827.1983_f32, 89.5049494_f32, 56.31_f32));
        // let vector2 = Vector3::from((89_f32, 72_f32, 936.5_f32));
        let expected = Vector3::from((c * vector1.x, c * vector1.y, c * vector1.z));
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division4() {
        let c = 61.891390_f32;
        let vector1 = Vector3::from((8827.1983_f32, 89.5049494_f32, 56.31_f32));
        // let vector2 = Vector3::from((89_f32, 72_f32, 936.5_f32));
        let expected = Vector3::from((vector1.x / c, vector1.y / c, vector1.z / c));
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_vector_times_zero_equals_zero() {
        let vector = Vector3::new(1_f32, 2_f32, 3_f32);

        assert_eq!(vector * 0_f32, Vector3::zero());
    }

    #[test]
    fn test_zero_times_vector_equals_zero() {
        let vector = Vector3::new(1_f32, 2_f32, 3_f32);

        assert_eq!(0_f32 * vector, Vector3::zero());
    }

    #[test]
    fn test_as_ref() {
        let vector = Vector3::new(1_i32, 2_i32, 3_i32);
        let vector_ref: &[i32; 3] = vector.as_ref();

        assert_eq!(vector_ref, &[1_i32, 2_i32, 3_i32]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let vector = Vector3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(vector[0], vector.x);
        assert_eq!(vector[1], vector.y);
        assert_eq!(vector[2], vector.z);
    }

    #[test]
    fn test_as_mut() {
        let mut vector = Vector3::new(1_i32, 2_i32, 3_i32);
        let vector_ref: &mut [i32; 3] = vector.as_mut();
        vector_ref[2] = 5_i32;

        assert_eq!(vector.z, 5_i32);
    }

    #[test]
    fn test_vector_addition_over_integers_commutative() {
        let vector1 = Vector3::new(1_i32, 2_i32, 3_i32);
        let vector2 = Vector3::new(4_i32, 5_i32, 6_i32);

        assert_eq!(vector1 + vector2, vector2 + vector1);
    }

    #[test]
    fn test_negative_zero_equals_positive_zero() {
        let zero: Vector3<f32> = Vector3::zero();

        assert_eq!(zero, -zero);
    }

    #[test]
    fn test_zero_vector_zero_norm() {
        let zero: Vector3<f32> = Vector3::zero();

        assert_eq!(zero.norm(), 0_f32);
    }

    #[test]
    fn test_vector_index_matches_component() {
        let vector = Vector3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(vector.x, vector[0]);
        assert_eq!(vector.y, vector[1]);
        assert_eq!(vector.z, vector[2]);
    }

    #[test]
    fn test_norm1() {
        let vector = Vector3::new(4_f64, 0_f64, 0_f64);
        let expected = 4_f64;
        let result = vector.norm();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_norm2() {
        let vector = Vector3::new(0_f64, 4_f64, 0_f64);
        let expected = 4_f64;
        let result = vector.norm();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_norm3() {
        let vector = Vector3::new(0_f64, 0_f64, 4_f64);
        let expected = 4_f64;
        let result = vector.norm();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_norm_unit_vectors() {
        let unit_x: Vector3<f64> = Vector3::unit_x();
        let unit_y: Vector3<f64> = Vector3::unit_y();
        let unit_z: Vector3<f64> = Vector3::unit_z();

        assert_eq!(unit_x.norm_squared(), 1_f64);
        assert_eq!(unit_x.norm(), 1_f64);
        assert_eq!(unit_y.norm_squared(), 1_f64);
        assert_eq!(unit_y.norm(), 1_f64);
        assert_eq!(unit_z.norm_squared(), 1_f64);
        assert_eq!(unit_z.norm(), 1_f64);
    }

    #[test]
    fn test_extend() {
        let vector = Vector3::new(1_i32, 2_i32, 3_i32);
        let expected = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
        let result = vector.extend(4_i32);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_contract() {
        let vector = Vector3::new(1_i32, 2_i32, 3_i32);
        let expected = Vector2::new(1_i32, 2_i32);
        let result = vector.contract();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_to_homogeneous() {
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let expected = Vector4::new(1_f64, 2_f64, 3_f64, 0_f64);
        let result = vector.to_homogeneous();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_homogeneous1() {
        let vector = Vector3::new(1_f64, 2_f64, 0_f64);
        let expected = Some(Vector2::new(1_f64, 2_f64));
        let result = vector.from_homogeneous();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_homogeneous2() {
        let vector = Vector3::new(1_f64, 2_f64, 3_f64);
        let result = vector.from_homogeneous();

        assert!(result.is_none());
    }
}

#[cfg(test)]
mod vector4_tests {
    use cglinalg_core::{
        Vector3,
        Vector4,
    };

    #[test]
    fn test_components1() {
        let vector = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(vector[0], 1_i32);
        assert_eq!(vector[1], 2_i32);
        assert_eq!(vector[2], 3_i32);
        assert_eq!(vector[3], 4_i32);
    }

    #[test]
    fn test_components2() {
        let vector = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(vector.x, vector[0]);
        assert_eq!(vector.y, vector[1]);
        assert_eq!(vector.z, vector[2]);
        assert_eq!(vector.w, vector[3]);
    }

    #[test]
    #[should_panic]
    fn test_vector_components_out_of_bounds1() {
        let vector = Vector4::new(1_f32, 2_f32, 3_f32, 4_f32);

        assert_eq!(vector[4], vector[4]);
    }

    #[test]
    #[should_panic]
    fn test_vector_components_out_of_bounds2() {
        let vector = Vector4::new(1_f32, 2_f32, 3_f32, 4_f32);

        assert_eq!(vector[usize::MAX], vector[usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    fn test_addition1() {
        // let c = 802.3435169_f32;
        let vector1 = Vector4::from((80_f32, 23.43_f32, 43.569_f32, 69.9093_f32));
        let vector2 = Vector4::from((6.741_f32, 426.1_f32, 23.5724_f32, 85567.75976_f32));
        let expected = Vector4::from((
            vector1.x + vector2.x,
            vector1.y + vector2.y,
            vector1.z + vector2.z,
            vector1.w + vector2.w,
        ));
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_subtraction1() {
        // let c = 802.3435169_f32;
        let vector1 = Vector4::from((80_f32, 23.43_f32, 43.569_f32, 69.9093_f32));
        let vector2 = Vector4::from((6.741_f32, 426.1_f32, 23.5724_f32, 85567.75976_f32));
        let expected = Vector4::from((
            vector1.x - vector2.x,
            vector1.y - vector2.y,
            vector1.z - vector2.z,
            vector1.w - vector2.w,
        ));
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_multiplication1() {
        let c = 802.3435169_f32;
        let vector1 = Vector4::from((80_f32, 23.43_f32, 43.569_f32, 69.9093_f32));
        // let vector2 = Vector4::from((6.741_f32, 426.1_f32, 23.5724_f32, 85567.75976_f32));
        let expected = Vector4::from((
            c * vector1.x,
            c * vector1.y,
            c * vector1.z,
            c * vector1.w,
        ));
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_division1() {
        let c = 802.3435169_f32;
        let vector1 = Vector4::from((80_f32, 23.43_f32, 43.569_f32, 69.9093_f32));
        // let vector2 = Vector4::from((6.741_f32, 426.1_f32, 23.5724_f32, 85567.75976_f32));
        let expected = Vector4::from((
            vector1.x / c,
            vector1.y / c,
            vector1.z / c,
            vector1.w / c,
        ));
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_addition2() {
        // let c = 33.249539_f32;
        let vector1 = Vector4::from((27.6189_f32, 13.90_f32, 4.2219_f32, 91.11955_f32));
        let vector2 = Vector4::from((258.083_f32, 31.70_f32, 42.17_f32, 8438.2376_f32));
        let expected = Vector4::from((
            vector1.x + vector2.x,
            vector1.y + vector2.y,
            vector1.z + vector2.z,
            vector1.w + vector2.w,
        ));
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_subtraction2() {
        // let c = 33.249539_f32;
        let vector1 = Vector4::from((27.6189_f32, 13.90_f32, 4.2219_f32, 91.11955_f32));
        let vector2 = Vector4::from((258.083_f32, 31.70_f32, 42.17_f32, 8438.2376_f32));
        let expected = Vector4::from((
            vector1.x - vector2.x,
            vector1.y - vector2.y,
            vector1.z - vector2.z,
            vector1.w - vector2.w,
        ));
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_multiplication2() {
        let c = 33.249539_f32;
        let vector1 = Vector4::from((27.6189_f32, 13.90_f32, 4.2219_f32, 91.11955_f32));
        // let vector2 = Vector4::from((258.083_f32, 31.70_f32, 42.17_f32, 8438.2376_f32));
        let expected = Vector4::from((
            c * vector1.x,
            c * vector1.y,
            c * vector1.z,
            c * vector1.w,
        ));
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_division2() {
        let c = 33.249539_f32;
        let vector1 = Vector4::from((27.6189_f32, 13.90_f32, 4.2219_f32, 91.11955_f32));
        // let vector2 = Vector4::from((258.083_f32, 31.70_f32, 42.17_f32, 8438.2376_f32));
        let expected = Vector4::from((
            vector1.x / c,
            vector1.y / c,
            vector1.z / c,
            vector1.w / c,
        ));
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_addition3() {
        // let c = 7.04217_f32;
        let vector1 = Vector4::from((70_f32, 49_f32, 95_f32, 508.5602759_f32));
        let vector2 = Vector4::from((89.9138_f32, 36.84_f32, 427.46894_f32, 0.5796180917_f32));
        let expected = Vector4::from((
            vector1.x + vector2.x,
            vector1.y + vector2.y,
            vector1.z + vector2.z,
            vector1.w + vector2.w,
        ));
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_subtraction3() {
        // let c = 7.04217_f32;
        let vector1 = Vector4::from((70_f32, 49_f32, 95_f32, 508.5602759_f32));
        let vector2 = Vector4::from((89.9138_f32, 36.84_f32, 427.46894_f32, 0.5796180917_f32));
        let expected = Vector4::from((
            vector1.x - vector2.x,
            vector1.y - vector2.y,
            vector1.z - vector2.z,
            vector1.w - vector2.w,
        ));
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_multiplication3() {
        let c = 7.04217_f32;
        let vector1 = Vector4::from((70_f32, 49_f32, 95_f32, 508.5602759_f32));
        // let vector2 = Vector4::from((89.9138_f32, 36.84_f32, 427.46894_f32, 0.5796180917_f32));
        let expected = Vector4::from((
            c * vector1.x,
            c * vector1.y,
            c * vector1.z,
            c * vector1.w,
        ));
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_division3() {
        let c = 7.04217_f32;
        let vector1 = Vector4::from((70_f32, 49_f32, 95_f32, 508.5602759_f32));
        // let vector2 = Vector4::from((89.9138_f32, 36.84_f32, 427.46894_f32, 0.5796180917_f32));
        let expected = Vector4::from((
            vector1.x / c,
            vector1.y / c,
            vector1.z / c,
            vector1.w / c,
        ));
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_addition4() {
        // let c = 61.891390_f32;
        let vector1 = Vector4::from((8827.1983_f32, 89.5049494_f32, 56.31_f32, 0.2888633714_f32));
        let vector2 = Vector4::from((89_f32, 72_f32, 936.5_f32, 0.2888633714_f32));
        let expected = Vector4::from((
            vector1.x + vector2.x,
            vector1.y + vector2.y,
            vector1.z + vector2.z,
            vector1.w + vector2.w,
        ));
        let result = vector1 + vector2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_subtraction4() {
        // let c = 61.891390_f32;
        let vector1 = Vector4::from((8827.1983_f32, 89.5049494_f32, 56.31_f32, 0.2888633714_f32));
        let vector2 = Vector4::from((89_f32, 72_f32, 936.5_f32, 0.2888633714_f32));
        let expected = Vector4::from((
            vector1.x - vector2.x,
            vector1.y - vector2.y,
            vector1.z - vector2.z,
            vector1.w - vector2.w,
        ));
        let result = vector1 - vector2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_multiplication4() {
        let c = 61.891390_f32;
        let vector1 = Vector4::from((8827.1983_f32, 89.5049494_f32, 56.31_f32, 0.2888633714_f32));
        // let vector2 = Vector4::from((89_f32, 72_f32, 936.5_f32, 0.2888633714_f32));
        let expected = Vector4::from((
            c * vector1.x,
            c * vector1.y,
            c * vector1.z,
            c * vector1.w,
        ));
        let result = vector1 * c;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_division4() {
        let c = 61.891390_f32;
        let vector1 = Vector4::from((8827.1983_f32, 89.5049494_f32, 56.31_f32, 0.2888633714_f32));
        // let vector2 = Vector4::from((89_f32, 72_f32, 936.5_f32, 0.2888633714_f32));
        let expected = Vector4::from((
            vector1.x / c,
            vector1.y / c,
            vector1.z / c,
            vector1.w / c,
        ));
        let result = vector1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_vector_times_zero_equals_zero() {
        let vector = Vector4::new(1_f32, 2_f32, 3_f32, 4_f32);

        assert_eq!(vector * 0_f32, Vector4::zero());
    }

    #[test]
    fn test_zero_times_vector_equals_zero() {
        let vector = Vector4::new(1_f32, 2_f32, 3_f32, 4_f32);

        assert_eq!(0_f32 * vector, Vector4::zero());
    }

    #[test]
    fn test_as_mut() {
        let mut vector = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
        let vector_ref: &mut [i32; 4] = vector.as_mut();
        vector_ref[2] = 5_i32;

        assert_eq!(vector.z, 5_i32);
    }

    #[test]
    fn test_vector_addition_over_integers_commutative() {
        let vector1 = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
        let vector2 = Vector4::new(5_i32, 6_i32, 7_i32, 8_i32);

        assert_eq!(vector1 + vector2, vector2 + vector1);
    }

    #[test]
    fn test_negative_zero_equals_positive_zero() {
        let zero: Vector4<f32> = Vector4::zero();

        assert_eq!(zero, -zero);
    }

    #[test]
    fn test_zero_vector_zero_norm() {
        let zero: Vector4<f32> = Vector4::zero();

        assert_eq!(zero.norm(), 0_f32);
    }

    #[test]
    fn test_as_ref() {
        let vector = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
        let vector_ref: &[i32; 4] = vector.as_ref();

        assert_eq!(vector_ref, &[1_i32, 2_i32, 3_i32, 4_i32]);
    }

    #[test]
    fn test_vector_indices_matches_components() {
        let vector = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(vector.x, vector[0]);
        assert_eq!(vector.y, vector[1]);
        assert_eq!(vector.z, vector[2]);
        assert_eq!(vector.w, vector[3]);
    }

    #[test]
    fn test_norm1() {
        let vector = Vector4::new(4_f64, 0_f64, 0_f64, 0_f64);
        let expected = 4_f64;
        let result = vector.norm();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_norm2() {
        let vector = Vector4::new(0_f64, 4_f64, 0_f64, 0_f64);
        let expected = 4_f64;
        let result = vector.norm();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_norm3() {
        let vector = Vector4::new(0_f64, 0_f64, 4_f64, 0_f64);
        let expected = 4_f64;
        let result = vector.norm();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_norm4() {
        let vector = Vector4::new(0_f64, 0_f64, 0_f64, 4_f64);
        let expected = 4_f64;
        let result = vector.norm();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_norm_unit_vectors() {
        let unit_x: Vector4<f64> = Vector4::unit_x();
        let unit_y: Vector4<f64> = Vector4::unit_y();
        let unit_z: Vector4<f64> = Vector4::unit_z();
        let unit_w: Vector4<f64> = Vector4::unit_w();

        assert_eq!(unit_x.norm_squared(), 1_f64);
        assert_eq!(unit_x.norm(),         1_f64);
        assert_eq!(unit_y.norm_squared(), 1_f64);
        assert_eq!(unit_y.norm(),         1_f64);
        assert_eq!(unit_z.norm_squared(), 1_f64);
        assert_eq!(unit_z.norm(),         1_f64);
        assert_eq!(unit_w.norm_squared(), 1_f64);
        assert_eq!(unit_w.norm(),         1_f64);
    }

    #[test]
    fn test_contract() {
        let vector = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
        let expected = Vector3::new(1_i32, 2_i32, 3_i32);
        let result = vector.contract();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_homogeneous1() {
        let vector = Vector4::new(1_f64, 2_f64, 3_f64, 0_f64);
        let expected = Some(Vector3::new(1_f64, 2_f64, 3_f64));
        let result = vector.from_homogeneous();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_homogeneous2() {
        let vector = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
        let result = vector.from_homogeneous();

        assert!(result.is_none());
    }
}
