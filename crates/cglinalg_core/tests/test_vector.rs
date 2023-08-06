extern crate cglinalg_core;


#[cfg(test)]
mod vector1_tests {
    use cglinalg_core::{
        Vector1,
        Magnitude,
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
                TestCase { c: 802.3435169, v1: Vector1::from(-23.43),  v2: Vector1::from(426.1),   },
                TestCase { c: 33.249539,   v1: Vector1::from(27.6189), v2: Vector1::from(258.083), },
                TestCase { c: 7.04217,     v1: Vector1::from(0.0),     v2: Vector1::from(0.0),     },
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
    fn test_addition() {
        test_cases().iter().for_each(|test| {
            let expected = Vector1::from(test.v1.x + test.v2.x);
            let result = test.v1 + test.v2;

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_subtraction() {
        test_cases().iter().for_each(|test| {
            let expected = Vector1::from(test.v1.x - test.v2.x);
            let result = test.v1 - test.v2;

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_scalar_multiplication() {
        test_cases().iter().for_each(|test| {
            let expected = Vector1::from(test.c * test.v1.x);
            let result = test.v1 * test.c;

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_scalar_division() {
        test_cases().iter().for_each(|test| {
            let expected = Vector1::from(test.v1.x / test.c);
            let result = test.v1 / test.c;

            assert_eq!(result, expected);
        });
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
        let v: Vector1<i32> = Vector1::new(1);
        let v_ref: &[i32; 1] = v.as_ref();

        assert_eq!(v_ref, &[1]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let v = Vector1::new(1);

        assert_eq!(v[0], v.x);
    }

    #[test]
    fn test_as_mut() {
        let mut v: Vector1<i32> = Vector1::new(1);
        let v_ref: &mut [i32; 1] = v.as_mut();
        v_ref[0] = 5;

        assert_eq!(v.x, 5);
    }

    #[test]
    fn test_vector_addition_over_integers_commutative() {
        let v = Vector1::new(2);
        let w = Vector1::new(3);

        assert_eq!(v + w, w + v);
    }

    #[test]
    fn test_negative_zero_equals_positive_zero() {
        let zero: Vector1<f32> = Vector1::zero();

        assert_eq!(zero, -zero);
    }

    #[test]
    fn test_zero_vector_zero_magnitude() {
        let zero: Vector1<f32> = Vector1::zero();

        assert_eq!(zero.magnitude(), 0_f32);
    }

    #[test]
    fn test_vector_index_matches_component() {
        let v = Vector1::new(1);

        assert_eq!(v.x, v[0]);
    }

    #[test]
    fn test_magnitude() {
        let vector = Vector1::new(4.0);
        let expected = 4.0;
        let result = vector.magnitude();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_magnitude_unit_vectors() {
        let unit_x: Vector1<f64> = Vector1::unit_x();

        assert_eq!(unit_x.magnitude_squared(), 1.0);
        assert_eq!(unit_x.magnitude(), 1.0);
    }
}


#[cfg(test)]
mod vector2_tests {
    use cglinalg_core::{
        Vector2,
        Magnitude,   
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
                TestCase { c: 802.3435169, v1: Vector2::from((80.0,  43.569)),    v2: Vector2::from((6.741, 23.5724)),     },
                TestCase { c: 33.249539,   v1: Vector2::from((27.6189, 4.2219)),  v2: Vector2::from((258.083, 42.17))      },
                TestCase { c: 7.04217,     v1: Vector2::from((70.0,  49.0)),      v2: Vector2::from((89.9138, 427.46894)), },
                TestCase { c: 61.891390,   v1: Vector2::from((8827.1983, 56.31)), v2: Vector2::from((89.0, 936.5)),        }
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
    fn test_addition() {
        test_cases().iter().for_each(|test| {
            let expected = Vector2::from((test.v1.x + test.v2.x, test.v1.y + test.v2.y));
            let result = test.v1 + test.v2;

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_subtraction() {
        test_cases().iter().for_each(|test| {
            let expected = Vector2::from((test.v1.x - test.v2.x, test.v1.y - test.v2.y));
            let result = test.v1 - test.v2;

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_scalar_multiplication() {
        test_cases().iter().for_each(|test| {
            let expected = Vector2::from((test.c * test.v1.x, test.c * test.v1.y));
            let result = test.v1 * test.c;

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_scalar_division() {
        test_cases().iter().for_each(|test| {
            let expected = Vector2::from((test.v1.x / test.c, test.v1.y / test.c));
            let result = test.v1 / test.c;

            assert_eq!(result, expected);
        });
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
        let v: Vector2<i32> = Vector2::new(1, 2);
        let v_ref: &[i32; 2] = v.as_ref();

        assert_eq!(v_ref, &[1, 2]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let v = Vector2::new(1, 2);

        assert_eq!(v[0], v.x);
        assert_eq!(v[1], v.y);
    }

    #[test]
    fn test_as_mut() {
        let mut v: Vector2<i32> = Vector2::new(1, 2);
        let v_ref: &mut [i32; 2] = v.as_mut();
        v_ref[0] = 5;

        assert_eq!(v.x, 5);
    }

    #[test]
    fn test_vector_addition_over_integers_commutative() {
        let v = Vector2::new(2, 3);
        let w = Vector2::new(4, 5);

        assert_eq!(v + w, w + v);
    }

    #[test]
    fn test_negative_zero_equals_positive_zero() {
        let zero: Vector2<f32> = Vector2::zero();

        assert_eq!(zero, -zero);
    }

    #[test]
    fn test_zero_vector_zero_magnitude() {
        let zero: Vector2<f32> = Vector2::zero();

        assert_eq!(zero.magnitude(), 0_f32);
    }

    #[test]
    fn test_vector_index_matches_component() {
        let v = Vector2::new(1, 2);

        assert_eq!(v.x, v[0]);
        assert_eq!(v.y, v[1]);
    }

    #[test]
    fn test_magnitude1() {
        let vector = Vector2::new(4.0, 0.0);
        let expected = 4.0;
        let result = vector.magnitude();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_magnitude2() {
        let vector = Vector2::new(0.0, 4.0);
        let expected = 4.0;
        let result = vector.magnitude();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_magnitude_unit_vectors() {
        let unit_x: Vector2<f64> = Vector2::unit_x();
        let unit_y: Vector2<f64> = Vector2::unit_y();

        assert_eq!(unit_x.magnitude_squared(), 1.0);
        assert_eq!(unit_x.magnitude(), 1.0);
        assert_eq!(unit_y.magnitude_squared(), 1.0);
        assert_eq!(unit_y.magnitude(), 1.0);
    }
}


#[cfg(test)]
mod vector3_tests {
    use cglinalg_core::{
        Vector3,
        Magnitude,
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
                    c: 802.3435169,
                    x: Vector3::from((80.0,  23.43, 43.569)),
                    y: Vector3::from((6.741, 426.1, 23.5724)),
                },
                TestCase {
                    c: 33.249539,
                    x: Vector3::from((27.6189, 13.90, 4.2219)),
                    y: Vector3::from((258.083, 31.70, 42.17))
                },
                TestCase {
                    c: 7.04217,
                    x: Vector3::from((70.0,  49.0,  95.0)),
                    y: Vector3::from((89.9138, 36.84, 427.46894)),
                },
                TestCase {
                    c: 61.891390,
                    x: Vector3::from((8827.1983, 89.5049494, 56.31)),
                    y: Vector3::from((89.0, 72.0, 936.5)),
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
    fn test_addition() {
        test_cases().iter().for_each(|test| {
            let expected = Vector3::from((test.x.x + test.y.x, test.x.y + test.y.y, test.x.z + test.y.z));
            let result = test.x + test.y;

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_subtraction() {
        test_cases().iter().for_each(|test| {
            let expected = Vector3::from((test.x.x - test.y.x, test.x.y - test.y.y, test.x.z - test.y.z));
            let result = test.x - test.y;

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_scalar_multiplication() {
        test_cases().iter().for_each(|test| {
            let expected = Vector3::from((test.c * test.x.x, test.c * test.x.y, test.c * test.x.z));
            let result = test.x * test.c;

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_scalar_division() {
        test_cases().iter().for_each(|test| {
            let expected = Vector3::from((test.x.x / test.c, test.x.y / test.c, test.x.z / test.c));
            let result = test.x / test.c;

            assert_eq!(result, expected);
        });
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
        let v: Vector3<i32> = Vector3::new(1, 2, 3);
        let v_ref: &[i32; 3] = v.as_ref();

        assert_eq!(v_ref, &[1, 2, 3]);
    }

    #[test]
    fn test_indexes_and_variables() {
        let v = Vector3::new(1, 2, 3);

        assert_eq!(v[0], v.x);
        assert_eq!(v[1], v.y);
        assert_eq!(v[2], v.z);
    }

    #[test]
    fn test_as_mut() {
        let mut v: Vector3<i32> = Vector3::new(1, 2, 3);
        let v_ref: &mut [i32; 3] = v.as_mut();
        v_ref[2] = 5;

        assert_eq!(v.z, 5);
    }

    #[test]
    fn test_vector_addition_over_integers_commutative() {
        let v = Vector3::new(1, 2, 3);
        let w = Vector3::new(4, 5, 6);

        assert_eq!(v + w, w + v);
    }

    #[test]
    fn test_negative_zero_equals_positive_zero() {
        let zero: Vector3<f32> = Vector3::zero();

        assert_eq!(zero, -zero);
    }

    #[test]
    fn test_zero_vector_zero_magnitude() {
        let zero: Vector3<f32> = Vector3::zero();

        assert_eq!(zero.magnitude(), 0_f32);
    }

    #[test]
    fn test_vector_index_matches_component() {
        let v = Vector3::new(1, 2, 3);

        assert_eq!(v.x, v[0]);
        assert_eq!(v.y, v[1]);
        assert_eq!(v.z, v[2]);
    }

    #[test]
    fn test_magnitude1() {
        let vector = Vector3::new(4.0, 0.0, 0.0);
        let expected = 4.0;
        let result = vector.magnitude();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_magnitude2() {
        let vector = Vector3::new(0.0, 4.0, 0.0);
        let expected = 4.0;
        let result = vector.magnitude();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_magnitude3() {
        let vector = Vector3::new(0.0, 0.0, 4.0);
        let expected = 4.0;
        let result = vector.magnitude();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_magnitude_unit_vectors() {
        let unit_x: Vector3<f64> = Vector3::unit_x();
        let unit_y: Vector3<f64> = Vector3::unit_y();
        let unit_z: Vector3<f64> = Vector3::unit_z();

        assert_eq!(unit_x.magnitude_squared(), 1.0);
        assert_eq!(unit_x.magnitude(), 1.0);
        assert_eq!(unit_y.magnitude_squared(), 1.0);
        assert_eq!(unit_y.magnitude(), 1.0);
        assert_eq!(unit_z.magnitude_squared(), 1.0);
        assert_eq!(unit_z.magnitude(), 1.0);
    }
}


#[cfg(test)]
mod vector4_tests {
    use cglinalg_core::{
        Vector4,
        Magnitude,
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
                    c: 802.3435169,
                    v1: Vector4::from((80.0,  23.43, 43.569, 69.9093)),
                    v2: Vector4::from((6.741, 426.1, 23.5724, 85567.75976)),
                },
                TestCase {
                    c: 33.249539,
                    v1: Vector4::from((27.6189, 13.90, 4.2219, 91.11955)),
                    v2: Vector4::from((258.083, 31.70, 42.17, 8438.2376))
                },
                TestCase {
                    c: 7.04217,
                    v1: Vector4::from((70.0, 49.0, 95.0, 508.5602759)),
                    v2: Vector4::from((89.9138, 36.84, 427.46894, 0.5796180917)),
                },
                TestCase {
                    c: 61.891390,
                    v1: Vector4::from((8827.1983, 89.5049494, 56.31, 0.2888633714)),
                    v2: Vector4::from((89.0, 72.0, 936.5, 0.2888633714)),
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
    fn test_addition() {
        test_cases().iter().for_each(|test| {
            let expected = Vector4::from((
                test.v1.x + test.v2.x, test.v1.y + test.v2.y,
                test.v1.z + test.v2.z, test.v1.w + test.v2.w
            ));
            let result = test.v1 + test.v2;

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_subtraction() {
        test_cases().iter().for_each(|test| {
            let expected = Vector4::from((
                test.v1.x - test.v2.x, test.v1.y - test.v2.y,
                test.v1.z - test.v2.z, test.v1.w - test.v2.w
            ));
            let result = test.v1 - test.v2;

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_scalar_multiplication() {
        test_cases().iter().for_each(|test| {
            let expected = Vector4::from((
                test.c * test.v1.x, test.c * test.v1.y, test.c * test.v1.z, test.c * test.v1.w
            ));
            let result = test.v1 * test.c;

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_scalar_division() {
        test_cases().iter().for_each(|test| {
            let expected = Vector4::from((
                test.v1.x / test.c, test.v1.y / test.c, test.v1.z / test.c, test.v1.w / test.c
            ));
            let result = test.v1 / test.c;

            assert_eq!(result, expected);
        });
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
        let mut v: Vector4<i32> = Vector4::new(1, 2, 3, 4);
        let v_ref: &mut [i32; 4] = v.as_mut();
        v_ref[2] = 5;

        assert_eq!(v.z, 5);
    }

    #[test]
    fn test_vector_addition_over_integers_commutative() {
        let v = Vector4::new(1, 2, 3, 4);
        let w = Vector4::new(5, 6, 7, 8);

        assert_eq!(v + w, w + v);
    }

    #[test]
    fn test_negative_zero_equals_positive_zero() {
        let zero: Vector4<f32> = Vector4::zero();

        assert_eq!(zero, -zero);
    }

    #[test]
    fn test_zero_vector_zero_magnitude() {
        let zero: Vector4<f32> = Vector4::zero();

        assert_eq!(zero.magnitude(), 0_f32);
    }

    #[test]
    fn test_as_ref() {
        let v: Vector4<i32> = Vector4::new(1, 2, 3, 4);
        let v_ref: &[i32; 4] = v.as_ref();

        assert_eq!(v_ref, &[1, 2, 3, 4]);
    }

    #[test]
    fn test_vector_indices_matches_components() {
        let v = Vector4::new(1, 2, 3, 4);
        
        assert_eq!(v.x, v[0]);
        assert_eq!(v.y, v[1]);
        assert_eq!(v.z, v[2]);
        assert_eq!(v.w, v[3]);
    }

    #[test]
    fn test_magnitude1() {
        let vector = Vector4::new(4.0, 0.0, 0.0, 0.0);
        let expected = 4.0;
        let result = vector.magnitude();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_magnitude2() {
        let vector = Vector4::new(0.0, 4.0, 0.0, 0.0);
        let expected = 4.0;
        let result = vector.magnitude();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_magnitude3() {
        let vector = Vector4::new(0.0, 0.0, 4.0, 0.0);
        let expected = 4.0;
        let result = vector.magnitude();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_magnitude4() {
        let vector = Vector4::new(0.0, 0.0, 0.0, 4.0);
        let expected = 4.0;
        let result = vector.magnitude();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_magnitude_unit_vectors() {
        let unit_x: Vector4<f64> = Vector4::unit_x();
        let unit_y: Vector4<f64> = Vector4::unit_y();
        let unit_z: Vector4<f64> = Vector4::unit_z();
        let unit_w: Vector4<f64> = Vector4::unit_w();

        assert_eq!(unit_x.magnitude_squared(), 1.0);
        assert_eq!(unit_x.magnitude(), 1.0);
        assert_eq!(unit_y.magnitude_squared(), 1.0);
        assert_eq!(unit_y.magnitude(), 1.0);
        assert_eq!(unit_z.magnitude_squared(), 1.0);
        assert_eq!(unit_z.magnitude(), 1.0);
        assert_eq!(unit_w.magnitude_squared(), 1.0);
        assert_eq!(unit_w.magnitude(), 1.0);
    }
}
