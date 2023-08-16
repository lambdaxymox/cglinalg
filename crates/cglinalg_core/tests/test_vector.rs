extern crate cglinalg_core;


#[cfg(test)]
mod vector1_tests {
    use cglinalg_core::{
        Vector1,
    };
    use core::slice::Iter;


    struct TestCase {
        c: f32,
        v1: Vector1<f32>,
        v2: Vector1<f32>,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase { c: 802.3435169_f32, v1: Vector1::from(-23.43_f32),  v2: Vector1::from(426.1_f32),   },
                TestCase { c: 33.249539_f32,   v1: Vector1::from(27.6189_f32), v2: Vector1::from(258.083_f32), },
                TestCase { c: 7.04217_f32,     v1: Vector1::from(0_f32),       v2: Vector1::from(0_f32),       },
            ]
        }
    }


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
        let v = Vector1::new(1_f32);

        assert_eq!(v[1], v[1]);
    }

    #[test]
    #[should_panic]
    fn test_vector_components_out_of_bounds2() {
        let v = Vector1::new(1_f32);

        assert_eq!(v[usize::MAX], v[usize::MAX]);
    }

    #[test]
    fn test_addition1() {
        // let c = 802.3435169_f32;
        let v1 = Vector1::from(-23.43_f32);
        let v2 = Vector1::from(426.1_f32);
        let expected = Vector1::from(v1.x + v2.x);
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction1() {
        // let c = 802.3435169_f32;
        let v1 = Vector1::from(-23.43_f32);
        let v2 = Vector1::from(426.1_f32);
        let expected = Vector1::from(v1.x - v2.x);
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication1() {
        let c = 802.3435169_f32;
        let v1 = Vector1::from(-23.43_f32);
        // let v2 = Vector1::from(426.1_f32);
        let expected = Vector1::from(v1.x * c);
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division1() {
        let c = 802.3435169_f32;
        let v1 = Vector1::from(-23.43_f32);
        // let v2 = Vector1::from(426.1_f32);
        let expected = Vector1::from(v1.x / c);
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition2() {
        // let c = 33.249539_f32;
        let v1 = Vector1::from(27.6189_f32);
        let v2 = Vector1::from(258.083_f32);
        let expected = Vector1::from(v1.x + v2.x);
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction2() {
        // let c = 33.249539_f32;
        let v1 = Vector1::from(27.6189_f32);
        let v2 = Vector1::from(258.083_f32);
        let expected = Vector1::from(v1.x - v2.x);
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication2() {
        let c = 33.249539_f32;
        let v1 = Vector1::from(27.6189_f32);
        // let v2 = Vector1::from(258.083_f32);
        let expected = Vector1::from(v1.x * c);
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division2() {
        let c = 33.249539_f32;
        let v1 = Vector1::from(27.6189_f32);
        // let v2 = Vector1::from(258.083_f32);
        let expected = Vector1::from(v1.x / c);
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition3() {
        // let c = 7.04217_f32;
        let v1 = Vector1::from(0_f32);
        let v2 = Vector1::from(0_f32);
        let expected = Vector1::from(v1.x + v2.x);
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction3() {
        // let c = 7.04217_f32;
        let v1 = Vector1::from(0_f32);
        let v2 = Vector1::from(0_f32);
        let expected = Vector1::from(v1.x - v2.x);
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication3() {
        let c = 7.04217_f32;
        let v1 = Vector1::from(0_f32);
        // let v2 = Vector1::from(0_f32);
        let expected = Vector1::from(v1.x * c);
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division3() {
        let c = 7.04217_f32;
        let v1 = Vector1::from(0_f32);
        // let v2 = Vector1::from(0_f32);
        let expected = Vector1::from(v1.x / c);
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition4() {
        // let c = 61.891390_f32;
        let v1 = Vector1::from(8827.1983_f32);
        let v2 = Vector1::from(89_f32);
        let expected = Vector1::from(v1.x + v2.x);
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction4() {
        // let c = 61.891390_f32;
        let v1 = Vector1::from(8827.1983_f32);
        let v2 = Vector1::from(89_f32);
        let expected = Vector1::from(v1.x - v2.x);
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication4() {
        let c = 61.891390_f32;
        let v1 = Vector1::from(8827.1983_f32);
        // let v2 = Vector1::from(89_f32);
        let expected = Vector1::from(c * v1.x);
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division4() {
        let c = 61.891390_f32;
        let v1 = Vector1::from(8827.1983_f32);
        // let v2 = Vector1::from(89_f32);
        let expected = Vector1::from(v1.x / c);
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_vector_times_zero_equals_zero() {
        let v = Vector1::new(1_f32);

        assert_eq!(v * 0_f32, Vector1::zero());
    }

    #[test]
    fn test_zero_times_vector_equals_zero() {
        let v = Vector1::new(1_f32);

        assert_eq!(0_f32 * v, Vector1::zero());
    }

    #[test]
    fn test_as_ref() {
        let v = Vector1::new(1_i32);
        let v_ref: &[i32; 1] = v.as_ref();

        assert_eq!(v_ref, &[1]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let v = Vector1::new(1_i32);

        assert_eq!(v[0], v.x);
    }

    #[test]
    fn test_as_mut() {
        let mut v = Vector1::new(1_i32);
        let v_ref: &mut [i32; 1] = v.as_mut();
        v_ref[0] = 5_i32;

        assert_eq!(v.x, 5);
    }

    #[test]
    fn test_vector_addition_over_integers_commutative() {
        let v = Vector1::new(2_i32);
        let w = Vector1::new(3_i32);

        assert_eq!(v + w, w + v);
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
        let v = Vector1::new(1_i32);

        assert_eq!(v.x, v[0]);
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
}


#[cfg(test)]
mod vector2_tests {
    use cglinalg_core::{
        Vector2,
    };
    use core::slice::Iter;


    struct TestCase {
        c: f32,
        v1: Vector2<f32>,
        v2: Vector2<f32>,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase { c: 802.3435169_f32, v1: Vector2::from((80_f32,  43.569_f32)),      v2: Vector2::from((6.741_f32, 23.5724_f32)),     },
                TestCase { c: 33.249539_f32,   v1: Vector2::from((27.6189_f32, 4.2219_f32)),  v2: Vector2::from((258.083_f32, 42.17_f32))      },
                TestCase { c: 7.04217_f32,     v1: Vector2::from((70_f32,  49_f32)),          v2: Vector2::from((89.9138_f32, 427.46894_f32)), },
                TestCase { c: 61.891390_f32,   v1: Vector2::from((8827.1983_f32, 56.31_f32)), v2: Vector2::from((89_f32, 936.5_f32)),          }
            ]
        }
    }

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
        let v = Vector2::new(1_f32, 2_f32);

        assert_eq!(v[2], v[2]);
    }

    #[test]
    #[should_panic]
    fn test_vector_components_out_of_bounds2() {
        let v = Vector2::new(1_f32, 2_f32);

        assert_eq!(v[usize::MAX], v[usize::MAX]);
    }

    #[test]
    fn test_addition1() {
        // let c = 802.3435169_f32;
        let v1 = Vector2::from((80_f32, 43.569_f32));
        let v2 = Vector2::from((6.741_f32, 23.5724_f32));
        let expected = Vector2::from((v1.x + v2.x, v1.y + v2.y));
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction1() {
        // let c = 802.3435169_f32;
        let v1 = Vector2::from((80_f32, 43.569_f32));
        let v2 = Vector2::from((6.741_f32, 23.5724_f32));
        let expected = Vector2::from((v1.x - v2.x, v1.y - v2.y));
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication1() {
        let c = 802.3435169_f32;
        let v1 = Vector2::from((80_f32, 43.569_f32));
        // let v2 = Vector2::from((6.741_f32, 23.5724_f32));
        let expected = Vector2::from((c * v1.x, c * v1.y));
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division1() {
        let c = 802.3435169_f32;
        let v1 = Vector2::from((80_f32, 43.569_f32));
        // let v2 = Vector2::from((6.741_f32, 23.5724_f32));
        let expected = Vector2::from((v1.x / c, v1.y / c));
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition2() {
        // let c = 33.249539_f32;
        let v1 = Vector2::from((27.6189_f32, 4.2219_f32));
        let v2 = Vector2::from((258.083_f32, 42.17_f32));
        let expected = Vector2::from((v1.x + v2.x, v1.y + v2.y));
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction2() {
        // let c = 33.249539_f32;
        let v1 = Vector2::from((27.6189_f32, 4.2219_f32));
        let v2 = Vector2::from((258.083_f32, 42.17_f32));
        let expected = Vector2::from((v1.x - v2.x, v1.y - v2.y));
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication2() {
        let c = 33.249539_f32;
        let v1 = Vector2::from((27.6189_f32, 4.2219_f32));
        // let v2 = Vector2::from((258.083_f32, 42.17_f32));
        let expected = Vector2::from((c * v1.x, c * v1.y));
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division2() {
        let c = 33.249539_f32;
        let v1 = Vector2::from((27.6189_f32, 4.2219_f32));
        // let v2 = Vector2::from((258.083_f32, 42.17_f32));
        let expected = Vector2::from((v1.x / c, v1.y / c));
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition3() {
        // let c = 7.04217_f32;
        let v1 = Vector2::from((70_f32, 49_f32));
        let v2 = Vector2::from((89.9138_f32, 427.46894_f32));
        let expected = Vector2::from((v1.x + v2.x, v1.y + v2.y));
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction3() {
        // let c = 7.04217_f32;
        let v1 = Vector2::from((70_f32, 49_f32));
        let v2 = Vector2::from((89.9138_f32, 427.46894_f32));
        let expected = Vector2::from((v1.x - v2.x, v1.y - v2.y));
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication3() {
        let c = 7.04217_f32;
        let v1 = Vector2::from((70_f32, 49_f32));
        // let v2 = Vector2::from((89.9138_f32, 427.46894_f32));
        let expected = Vector2::from((c * v1.x, c * v1.y));
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division3() {
        let c = 7.04217_f32;
        let v1 = Vector2::from((70_f32, 49_f32));
        // let v2 = Vector2::from((89.9138_f32, 427.46894_f32));
        let expected = Vector2::from((v1.x / c, v1.y / c));
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition4() {
        // let c = 61.891390_f32;
        let v1 = Vector2::from((8827.1983_f32, 56.31_f32));
        let v2 = Vector2::from((89_f32, 936.5_f32));
        let expected = Vector2::from((v1.x + v2.x, v1.y + v2.y));
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction4() {
        // let c = 61.891390_f32;
        let v1 = Vector2::from((8827.1983_f32, 56.31_f32));
        let v2 = Vector2::from((89_f32, 936.5_f32));
        let expected = Vector2::from((v1.x - v2.x, v1.y - v2.y));
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication4() {
        let c = 61.891390_f32;
        let v1 = Vector2::from((8827.1983_f32, 56.31_f32));
        // let v2 = Vector2::from((89_f32, 936.5_f32));
        let expected = Vector2::from((c * v1.x, c * v1.y));
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division4() {
        let c = 61.891390_f32;
        let v1 = Vector2::from((8827.1983_f32, 56.31_f32));
        // let v2 = Vector2::from((89_f32, 936.5_f32));
        let expected = Vector2::from((v1.x / c, v1.y / c));
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_vector_times_zero_equals_zero() {
        let v = Vector2::new(1_f32, 2_f32);

        assert_eq!(v * 0_f32, Vector2::zero());
    }

    #[test]
    fn test_zero_times_vector_equals_zero() {
        let v = Vector2::new(1_f32, 2_f32);

        assert_eq!(0_f32 * v, Vector2::zero());
    }

    #[test]
    fn test_as_ref() {
        let v = Vector2::new(1_i32, 2_i32);
        let v_ref: &[i32; 2] = v.as_ref();

        assert_eq!(v_ref, &[1, 2]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let v = Vector2::new(1_i32, 2_i32);

        assert_eq!(v[0], v.x);
        assert_eq!(v[1], v.y);
    }

    #[test]
    fn test_as_mut() {
        let mut v = Vector2::new(1_i32, 2_i32);
        let v_ref: &mut [i32; 2] = v.as_mut();
        v_ref[0] = 5;

        assert_eq!(v.x, 5);
    }

    #[test]
    fn test_vector_addition_over_integers_commutative() {
        let v = Vector2::new(2_i32, 3_i32);
        let w = Vector2::new(4_i32, 5_i32);

        assert_eq!(v + w, w + v);
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
        let v = Vector2::new(1_i32, 2_i32);

        assert_eq!(v.x, v[0]);
        assert_eq!(v.y, v[1]);
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
}


#[cfg(test)]
mod vector3_tests {
    use cglinalg_core::{
        Vector3,
    };
    use core::slice::Iter;


    struct TestCase {
        c: f32,
        x: Vector3<f32>,
        y: Vector3<f32>,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase {
                    c: 802.3435169_f32,
                    x: Vector3::from((80_f32,  23.43_f32, 43.569_f32)),
                    y: Vector3::from((6.741_f32, 426.1_f32, 23.5724_f32)),
                },
                TestCase {
                    c: 33.249539_f32,
                    x: Vector3::from((27.6189_f32, 13.90_f32, 4.2219_f32)),
                    y: Vector3::from((258.083_f32, 31.70_f32, 42.17_f32))
                },
                TestCase {
                    c: 7.04217_f32,
                    x: Vector3::from((70_f32,  49_f32,  95_f32)),
                    y: Vector3::from((89.9138_f32, 36.84_f32, 427.46894_f32)),
                },
                TestCase {
                    c: 61.891390_f32,
                    x: Vector3::from((8827.1983_f32, 89.5049494_f32, 56.31_f32)),
                    y: Vector3::from((89_f32, 72_f32, 936.5_f32)),
                }
            ]
        }
    }

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
        let v = Vector3::new(1_f32, 2_f32, 3_f32);

        assert_eq!(v[3], v[3]);
    }

    #[test]
    #[should_panic]
    fn test_vector_components_out_of_bounds2() {
        let v = Vector3::new(1_f32, 2_f32, 3_f32);

        assert_eq!(v[usize::MAX], v[usize::MAX]);
    }

    #[test]
    fn test_addition1() {
        // let c = 802.3435169_f32;
        let v1 = Vector3::from((80_f32, 23.43_f32, 43.569_f32));
        let v2 = Vector3::from((6.741_f32, 426.1_f32, 23.5724_f32));
        let expected = Vector3::from((v1.x + v2.x, v1.y + v2.y, v1.z + v2.z));
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction1() {
        // let c = 802.3435169_f32;
        let v1 = Vector3::from((80_f32, 23.43_f32, 43.569_f32));
        let v2 = Vector3::from((6.741_f32, 426.1_f32, 23.5724_f32));
        let expected = Vector3::from((v1.x - v2.x, v1.y - v2.y, v1.z - v2.z));
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication1() {
        let c = 802.3435169_f32;
        let v1 = Vector3::from((80_f32, 23.43_f32, 43.569_f32));
        // let v2 = Vector3::from((6.741_f32, 426.1_f32, 23.5724_f32));
        let expected = Vector3::from((c * v1.x, c * v1.y, c * v1.z));
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division1() {
        let c = 802.3435169_f32;
        let v1 = Vector3::from((80_f32, 23.43_f32, 43.569_f32));
        // let v2 = Vector3::from((6.741_f32, 426.1_f32, 23.5724_f32));
        let expected = Vector3::from((v1.x / c, v1.y / c, v1.z / c));
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition2() {
        // let c = 33.249539_f32;
        let v1 = Vector3::from((27.6189_f32, 13.90_f32, 4.2219_f32));
        let v2 = Vector3::from((258.083_f32, 31.70_f32, 42.17_f32));
        let expected = Vector3::from((v1.x + v2.x, v1.y + v2.y, v1.z + v2.z));
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction2() {
        // let c = 33.249539_f32;
        let v1 = Vector3::from((27.6189_f32, 13.90_f32, 4.2219_f32));
        let v2 = Vector3::from((258.083_f32, 31.70_f32, 42.17_f32));
        let expected = Vector3::from((v1.x - v2.x, v1.y - v2.y, v1.z - v2.z));
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication2() {
        let c = 33.249539_f32;
        let v1 = Vector3::from((27.6189_f32, 13.90_f32, 4.2219_f32));
        // let v2 = Vector3::from((258.083_f32, 31.70_f32, 42.17_f32));
        let expected = Vector3::from((c * v1.x, c * v1.y, c * v1.z));
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division2() {
        let c = 33.249539_f32;
        let v1 = Vector3::from((27.6189_f32, 13.90_f32, 4.2219_f32));
        // let v2 = Vector3::from((258.083_f32, 31.70_f32, 42.17_f32));
        let expected = Vector3::from((v1.x / c, v1.y / c, v1.z / c));
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition3() {
        // let c = 7.04217_f32;
        let v1 = Vector3::from((70_f32, 49_f32, 95_f32));
        let v2 = Vector3::from((89.9138_f32, 36.84_f32, 427.46894_f32));
        let expected = Vector3::from((v1.x + v2.x, v1.y + v2.y, v1.z + v2.z));
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction3() {
        // let c = 7.04217_f32;
        let v1 = Vector3::from((70_f32, 49_f32, 95_f32));
        let v2 = Vector3::from((89.9138_f32, 36.84_f32, 427.46894_f32));
        let expected = Vector3::from((v1.x - v2.x, v1.y - v2.y, v1.z - v2.z));
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication3() {
        let c = 7.04217_f32;
        let v1 = Vector3::from((70_f32, 49_f32, 95_f32));
        // let v2 = Vector3::from((89.9138_f32, 36.84_f32, 427.46894_f32));
        let expected = Vector3::from((c * v1.x, c * v1.y, c * v1.z));
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division3() {
        let c = 7.04217_f32;
        let v1 = Vector3::from((70_f32, 49_f32, 95_f32));
        // let v2 = Vector3::from((89.9138_f32, 36.84_f32, 427.46894_f32));
        let expected = Vector3::from((v1.x / c, v1.y / c, v1.z / c));
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition4() {
        // let c = 61.891390_f32;
        let v1 = Vector3::from((8827.1983_f32, 89.5049494_f32, 56.31_f32));
        let v2 = Vector3::from((89_f32, 72_f32, 936.5_f32));
        let expected = Vector3::from((v1.x + v2.x, v1.y + v2.y, v1.z + v2.z));
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction4() {
        // let c = 61.891390_f32;
        let v1 = Vector3::from((8827.1983_f32, 89.5049494_f32, 56.31_f32));
        let v2 = Vector3::from((89_f32, 72_f32, 936.5_f32));
        let expected = Vector3::from((v1.x - v2.x, v1.y - v2.y, v1.z - v2.z));
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication4() {
        let c = 61.891390_f32;
        let v1 = Vector3::from((8827.1983_f32, 89.5049494_f32, 56.31_f32));
        // let v2 = Vector3::from((89_f32, 72_f32, 936.5_f32));
        let expected = Vector3::from((c * v1.x, c * v1.y, c * v1.z));
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division4() {
        let c = 61.891390_f32;
        let v1 = Vector3::from((8827.1983_f32, 89.5049494_f32, 56.31_f32));
        // let v2 = Vector3::from((89_f32, 72_f32, 936.5_f32));
        let expected = Vector3::from((v1.x / c, v1.y / c, v1.z / c));
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_vector_times_zero_equals_zero() {
        let v = Vector3::new(1_f32, 2_f32, 3_f32);

        assert_eq!(v * 0_f32, Vector3::zero());
    }

    #[test]
    fn test_zero_times_vector_equals_zero() {
        let v = Vector3::new(1_f32, 2_f32, 3_f32);

        assert_eq!(0_f32 * v, Vector3::zero());
    }

    #[test]
    fn test_as_ref() {
        let v = Vector3::new(1_i32, 2_i32, 3_i32);
        let v_ref: &[i32; 3] = v.as_ref();

        assert_eq!(v_ref, &[1_i32, 2_i32, 3_i32]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let v = Vector3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(v[0], v.x);
        assert_eq!(v[1], v.y);
        assert_eq!(v[2], v.z);
    }

    #[test]
    fn test_as_mut() {
        let mut v = Vector3::new(1_i32, 2_i32, 3_i32);
        let v_ref: &mut [i32; 3] = v.as_mut();
        v_ref[2] = 5_i32;

        assert_eq!(v.z, 5_i32);
    }

    #[test]
    fn test_vector_addition_over_integers_commutative() {
        let v = Vector3::new(1_i32, 2_i32, 3_i32);
        let w = Vector3::new(4_i32, 5_i32, 6_i32);

        assert_eq!(v + w, w + v);
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
        let v = Vector3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(v.x, v[0]);
        assert_eq!(v.y, v[1]);
        assert_eq!(v.z, v[2]);
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
}


#[cfg(test)]
mod vector4_tests {
    use cglinalg_core::{
        Vector4,
    };
    use core::slice::Iter;

    
    struct TestCase {
        c: f32,
        v1: Vector4<f32>,
        v2: Vector4<f32>,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase {
                    c: 802.3435169_f32,
                    v1: Vector4::from((80_f32,  23.43_f32, 43.569_f32, 69.9093_f32)),
                    v2: Vector4::from((6.741_f32, 426.1_f32, 23.5724_f32, 85567.75976_f32)),
                },
                TestCase {
                    c: 33.249539_f32,
                    v1: Vector4::from((27.6189_f32, 13.90_f32, 4.2219_f32, 91.11955_f32)),
                    v2: Vector4::from((258.083_f32, 31.70_f32, 42.17_f32, 8438.2376_f32))
                },
                TestCase {
                    c: 7.04217_f32,
                    v1: Vector4::from((70_f32, 49_f32, 95_f32, 508.5602759_f32)),
                    v2: Vector4::from((89.9138_f32, 36.84_f32, 427.46894_f32, 0.5796180917_f32)),
                },
                TestCase {
                    c: 61.891390_f32,
                    v1: Vector4::from((8827.1983_f32, 89.5049494_f32, 56.31_f32, 0.2888633714_f32)),
                    v2: Vector4::from((89_f32, 72_f32, 936.5_f32, 0.2888633714_f32)),
                }
            ]
        }
    }


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
    fn  test_vector_components_out_of_bounds1() {
        let v = Vector4::new(1_f32, 2_f32, 3_f32, 4_f32);

        assert_eq!(v[4], v[4]);
    }

    #[test]
    #[should_panic]
    fn  test_vector_components_out_of_bounds2() {
        let v = Vector4::new(1_f32, 2_f32, 3_f32, 4_f32);

        assert_eq!(v[usize::MAX], v[usize::MAX]);
    }

    #[test]
    fn test_addition1() {
        // let c = 802.3435169_f32;
        let v1 = Vector4::from((80_f32,  23.43_f32, 43.569_f32, 69.9093_f32));
        let v2 = Vector4::from((6.741_f32, 426.1_f32, 23.5724_f32, 85567.75976_f32));
        let expected = Vector4::from((v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w));
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction1() {
        // let c = 802.3435169_f32;
        let v1 = Vector4::from((80_f32,  23.43_f32, 43.569_f32, 69.9093_f32));
        let v2 = Vector4::from((6.741_f32, 426.1_f32, 23.5724_f32, 85567.75976_f32));
        let expected = Vector4::from((v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w));
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication1() {
        let c = 802.3435169_f32;
        let v1 = Vector4::from((80_f32,  23.43_f32, 43.569_f32, 69.9093_f32));
        // let v2 = Vector4::from((6.741_f32, 426.1_f32, 23.5724_f32, 85567.75976_f32));
        let expected = Vector4::from((c * v1.x, c * v1.y, c * v1.z, c * v1.w));
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division1() {
        let c = 802.3435169_f32;
        let v1 = Vector4::from((80_f32,  23.43_f32, 43.569_f32, 69.9093_f32));
        // let v2 = Vector4::from((6.741_f32, 426.1_f32, 23.5724_f32, 85567.75976_f32));
        let expected = Vector4::from((v1.x / c, v1.y / c, v1.z / c, v1.w / c));
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition2() {
        // let c = 33.249539_f32;
        let v1 = Vector4::from((27.6189_f32, 13.90_f32, 4.2219_f32, 91.11955_f32));
        let v2 = Vector4::from((258.083_f32, 31.70_f32, 42.17_f32, 8438.2376_f32));
        let expected = Vector4::from((v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w));
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction2() {
        // let c = 33.249539_f32;
        let v1 = Vector4::from((27.6189_f32, 13.90_f32, 4.2219_f32, 91.11955_f32));
        let v2 = Vector4::from((258.083_f32, 31.70_f32, 42.17_f32, 8438.2376_f32));
        let expected = Vector4::from((v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w));
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication2() {
        let c = 33.249539_f32;
        let v1 = Vector4::from((27.6189_f32, 13.90_f32, 4.2219_f32, 91.11955_f32));
        // let v2 = Vector4::from((258.083_f32, 31.70_f32, 42.17_f32, 8438.2376_f32));
        let expected = Vector4::from((c * v1.x, c * v1.y, c * v1.z, c * v1.w));
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division2() {
        let c = 33.249539_f32;
        let v1 = Vector4::from((27.6189_f32, 13.90_f32, 4.2219_f32, 91.11955_f32));
        // let v2 = Vector4::from((258.083_f32, 31.70_f32, 42.17_f32, 8438.2376_f32));
        let expected = Vector4::from((v1.x / c, v1.y / c, v1.z / c, v1.w / c));
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition3() {
        // let c = 7.04217_f32;
        let v1 = Vector4::from((70_f32, 49_f32, 95_f32, 508.5602759_f32));
        let v2 = Vector4::from((89.9138_f32, 36.84_f32, 427.46894_f32, 0.5796180917_f32));
        let expected = Vector4::from((v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w));
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction3() {
        // let c = 7.04217_f32;
        let v1 = Vector4::from((70_f32, 49_f32, 95_f32, 508.5602759_f32));
        let v2 = Vector4::from((89.9138_f32, 36.84_f32, 427.46894_f32, 0.5796180917_f32));
        let expected = Vector4::from((v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w));
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication3() {
        let c = 7.04217_f32;
        let v1 = Vector4::from((70_f32, 49_f32, 95_f32, 508.5602759_f32));
        // let v2 = Vector4::from((89.9138_f32, 36.84_f32, 427.46894_f32, 0.5796180917_f32));
        let expected = Vector4::from((c * v1.x, c * v1.y, c * v1.z, c * v1.w));
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division3() {
        let c = 7.04217_f32;
        let v1 = Vector4::from((70_f32, 49_f32, 95_f32, 508.5602759_f32));
        // let v2 = Vector4::from((89.9138_f32, 36.84_f32, 427.46894_f32, 0.5796180917_f32));
        let expected = Vector4::from((v1.x / c, v1.y / c, v1.z / c, v1.w / c));
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition4() {
        // let c = 61.891390_f32;
        let v1 = Vector4::from((8827.1983_f32, 89.5049494_f32, 56.31_f32, 0.2888633714_f32));
        let v2 = Vector4::from((89_f32, 72_f32, 936.5_f32, 0.2888633714_f32));
        let expected = Vector4::from((v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w));
        let result = v1 + v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction4() {
        // let c = 61.891390_f32;
        let v1 = Vector4::from((8827.1983_f32, 89.5049494_f32, 56.31_f32, 0.2888633714_f32));
        let v2 = Vector4::from((89_f32, 72_f32, 936.5_f32, 0.2888633714_f32));
        let expected = Vector4::from((v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w));
        let result = v1 - v2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiplication4() {
        let c = 61.891390_f32;
        let v1 = Vector4::from((8827.1983_f32, 89.5049494_f32, 56.31_f32, 0.2888633714_f32));
        // let v2 = Vector4::from((89_f32, 72_f32, 936.5_f32, 0.2888633714_f32));
        let expected = Vector4::from((c * v1.x, c * v1.y, c * v1.z, c * v1.w));
        let result = v1 * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_division4() {
        let c = 61.891390_f32;
        let v1 = Vector4::from((8827.1983_f32, 89.5049494_f32, 56.31_f32, 0.2888633714_f32));
        // let v2 = Vector4::from((89_f32, 72_f32, 936.5_f32, 0.2888633714_f32));
        let expected = Vector4::from((v1.x / c, v1.y / c, v1.z / c, v1.w / c));
        let result = v1 / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_vector_times_zero_equals_zero() {
        let v = Vector4::new(1_f32, 2_f32, 3_f32, 4_f32);

        assert_eq!(v * 0_f32, Vector4::zero());
    }

    #[test]
    fn test_zero_times_vector_equals_zero() {
        let v = Vector4::new(1_f32, 2_f32, 3_f32, 4_f32);

        assert_eq!(0_f32 * v, Vector4::zero());
    }

    #[test]
    fn test_as_mut() {
        let mut v = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
        let v_ref: &mut [i32; 4] = v.as_mut();
        v_ref[2] = 5_i32;

        assert_eq!(v.z, 5_i32);
    }

    #[test]
    fn test_vector_addition_over_integers_commutative() {
        let v = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
        let w = Vector4::new(5_i32, 6_i32, 7_i32, 8_i32);

        assert_eq!(v + w, w + v);
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
        let v = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
        let v_ref: &[i32; 4] = v.as_ref();

        assert_eq!(v_ref, &[1_i32, 2_i32, 3_i32, 4_i32]);
    }

    #[test]
    fn test_vector_indices_matches_components() {
        let v = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
        
        assert_eq!(v.x, v[0]);
        assert_eq!(v.y, v[1]);
        assert_eq!(v.z, v[2]);
        assert_eq!(v.w, v[3]);
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

    #[test]
    fn test_norm_unit_vectors() {
        let unit_x: Vector4<f64> = Vector4::unit_x();
        let unit_y: Vector4<f64> = Vector4::unit_y();
        let unit_z: Vector4<f64> = Vector4::unit_z();
        let unit_w: Vector4<f64> = Vector4::unit_w();

        assert_eq!(unit_x.norm_squared(), 1_f64);
        assert_eq!(unit_x.norm(), 1_f64);
        assert_eq!(unit_y.norm_squared(), 1_f64);
        assert_eq!(unit_y.norm(), 1_f64);
        assert_eq!(unit_z.norm_squared(), 1_f64);
        assert_eq!(unit_z.norm(), 1_f64);
        assert_eq!(unit_w.norm_squared(), 1_f64);
        assert_eq!(unit_w.norm(), 1_f64);
    }
}

