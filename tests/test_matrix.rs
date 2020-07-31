extern crate gdmath;
extern crate num_traits;
extern crate proptest;


#[cfg(test)]
mod matrix2_tests {
    use gdmath::{
        Vector2,
        Matrix2,
        One, 
        Zero, 
        Matrix,
    };
    use gdmath::approx::relative_eq;
    use std::slice::Iter;


    struct TestCase {
        a_mat: Matrix2<f32>,
        b_mat: Matrix2<f32>,
        expected: Matrix2<f32>,
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
                    a_mat: Matrix2::new(80.0,  23.43,     426.1,   23.5724),
                    b_mat: Matrix2::new(36.84, 427.46894, 7.04217, 61.891390),
                    expected: Matrix2::new(185091.72, 10939.63, 26935.295, 1623.9266),
                },
                TestCase {
                    a_mat: Matrix2::one(),
                    b_mat: Matrix2::one(),
                    expected: Matrix2::one(),
                },
                TestCase {
                    a_mat: Matrix2::zero(),
                    b_mat: Matrix2::zero(),
                    expected: Matrix2::zero(),
                },
                TestCase {
                    a_mat: Matrix2::new(68.32, 0.0, 0.0, 37.397),
                    b_mat: Matrix2::new(57.72, 0.0, 0.0, 9.5433127),
                    expected: Matrix2::new(3943.4304, 0.0, 0.0, 356.89127),
                },
            ]
        }
    }

    #[test]
    fn test_mat_times_identity_equals_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_identity = test.a_mat * Matrix2::one();
            let b_mat_times_identity = test.b_mat * Matrix2::one();

            assert_eq!(a_mat_times_identity, test.a_mat);
            assert_eq!(b_mat_times_identity, test.b_mat);
        })
    }

    #[test]
    fn test_mat_times_zero_equals_zero() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_zero = test.a_mat * Matrix2::zero();
            let b_mat_times_zero = test.b_mat * Matrix2::zero();

            assert_eq!(a_mat_times_zero, Matrix2::zero());
            assert_eq!(b_mat_times_zero, Matrix2::zero());
        })
    }

    #[test]
    fn test_zero_times_mat_equals_zero() {
        test_cases().iter().for_each(|test| {
            let zero_times_a_mat = Matrix2::zero() * test.a_mat;
            let zero_times_b_mat = Matrix2::zero() * test.b_mat;

            assert_eq!(zero_times_a_mat, Matrix2::zero());
            assert_eq!(zero_times_b_mat, Matrix2::zero());
        })
    }

    #[test]
    fn test_mat_times_identity_equals_identity_times_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_identity = test.a_mat * Matrix2::one();
            let identity_times_a_mat = Matrix2::one() * test.a_mat;
            let b_mat_times_identity = test.b_mat * Matrix2::one();
            let identity_times_b_mat = Matrix2::one() * test.b_mat;

            assert_eq!(a_mat_times_identity, identity_times_a_mat);
            assert_eq!(b_mat_times_identity, identity_times_b_mat);
        })
    }

    #[test]
    fn test_mat_transpose_transpose_equals_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_tr_tr = test.a_mat.transpose().transpose();
            let b_mat_tr_tr = test.b_mat.transpose().transpose();
            
            assert_eq!(a_mat_tr_tr, test.a_mat);
            assert_eq!(b_mat_tr_tr, test.b_mat);
        })
    }

    #[test]
    fn test_identity_transpose_equals_identity() {
        let identity = Matrix2::<f32>::one();
        let identity_tr = identity.transpose();
            
        assert_eq!(identity, identity_tr);
    }

    #[test]
    fn test_matrix_multiplication() {
        test_cases().iter().for_each(|test| {
            let result = test.a_mat * test.b_mat;
            let expected = test.expected;

            assert_eq!(result, expected);
        })
    }

    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector2::new(1.0, 2.0);
        let c1 = Vector2::new(3.0, 4.0);
        let expected = Matrix2::new(
            1.0, 2.0, 
            3.0, 4.0
        );
        let result = Matrix2::from_cols(c0, c1);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_constant_times_identity_is_constant_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix2::one();
        let expected = Matrix2::new(
            c, 0.0, 
            0.0, c
        );

        assert_eq!(id * c, expected);
    }

    #[test]
    fn test_identity_divide_constant_is_constant_inverse_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix2::one();
        let expected = Matrix2::new(
            1.0 / c, 0.0, 
            0.0,     1.0 / c
        );

        assert_eq!(id / c, expected);
    }

    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero = Matrix2::zero();
        let matrix = Matrix2::new(
            36.84, 427.46, 
            7.47,  61.89
        );

        assert_eq!(matrix + zero, matrix);
    }

    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero = Matrix2::zero();
        let matrix = Matrix2::new(
            36.84, 427.46, 
            7.47,  61.89
        );

        assert_eq!(zero + matrix, matrix);
    }

    #[test]
    fn test_matrix_with_zero_determinant() {
        let matrix: Matrix2<f64> = Matrix2::new(
            1_f64, 2_f64, 
            4_f64, 8_f64
        );
        
        assert_eq!(matrix.determinant(), 0.0);
    }

    #[test]
    fn test_lower_triangular_matrix_determinant() {
        let matrix: Matrix2<f64> = Matrix2::new(
            2_f64,  0_f64,
            5_f64,  3_f64
        );

        assert_eq!(matrix.determinant(), 2_f64 * 3_f64);
    }

    #[test]
    fn test_upper_triangular_matrix_determinant() {
        let matrix: Matrix2<f64> = Matrix2::new(
            2_f64,  5_f64,
            0_f64,  3_f64
        );

        assert_eq!(matrix.determinant(), 2_f64 * 3_f64);
    }

    #[test]
    fn test_matrix_inverse() {
        let matrix: Matrix2<f64> = Matrix2::new(
            5_f64, 1_f64, 
            1_f64, 5_f64
        );
        let expected: Matrix2<f64> = (1_f64 / 24_f64) * Matrix2::new(
             5_f64, -1_f64,
            -1_f64,  5_f64
        );
        let result = matrix.inverse().unwrap();
        let epsilon = 1e-7;

        assert!(relative_eq!(result, expected, epsilon = epsilon));
    }

    #[test]
    fn test_identity_is_invertible() {
        assert!(Matrix2::<f64>::one().is_invertible());
    }

    #[test]
    fn test_identity_inverse_is_identity() {
        let result: Matrix2<f64> = Matrix2::one().inverse().unwrap();
        let expected: Matrix2<f64> = Matrix2::one();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_diagonal_matrix() {
        let matrix: Matrix2<f64> = 4_f64 * Matrix2::one();
        let expected: Matrix2<f64> = (1_f64 / 4_f64) * Matrix2::one();
        let result = matrix.inverse().unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_with_nonzero_determinant_is_invertible() {
        let matrix = Matrix2::new(1f32, 2f32, 3f32, 4f32);
        
        assert!(matrix.is_invertible());
    }

    #[test]
    fn test_matrix_with_zero_determinant_is_not_invertible() {
        let matrix = Matrix2::new(1f32, 2f32, 4f32, 8f32);
        
        assert!(!matrix.is_invertible());
    }

    #[test]
    fn test_noninvertible_matrix_returns_none() {
        let matrix = Matrix2::new(1f32, 2f32, 4f32, 8f32);
        
        assert!(matrix.inverse().is_none());
    }


    #[test]
    fn test_matrix_times_inverse_is_identity() {
        let matrix = Matrix2::new(36.84, 427.46, 7.47, 61.89);
        let matrix_inv = matrix.inverse().unwrap();
        let one = Matrix2::one();

        assert!(relative_eq!(matrix * matrix_inv, one, epsilon = 1e-7));
    }

    #[test]
    fn test_inverse_times_matrix_is_identity() {
        let matrix = Matrix2::new(36.84, 427.46, 7.47, 61.89);
        let matrix_inv = matrix.inverse().unwrap();
        let one = Matrix2::one();

        assert!(relative_eq!(matrix_inv * matrix, one, epsilon = 1e-7));        
    }

    #[test]
    fn test_constant_times_matrix_inverse_equals_constant_inverse_times_matrix_inverse() {
        let matrix: Matrix2<f64> = Matrix2::new(
            80.0,   426.1,
            23.43,  23.5724
        );
        let constant: f64 = 4_f64;
        let constant_times_matrix_inverse = (constant * matrix).inverse().unwrap();
        let constant_inverse_times_matrix_inverse = (1_f64 / constant) * matrix.inverse().unwrap();

        assert_eq!(constant_times_matrix_inverse, constant_inverse_times_matrix_inverse);
    }

    #[test]
    fn test_matrix_transpose_inverse_equals_matrix_inverse_transpose() {
        let matrix: Matrix2<f64> = Matrix2::new(
            80.0,   426.1, 
            23.43,  23.5724
        );
        let matrix_transpose_inverse = matrix.transpose().inverse().unwrap();
        let matrix_inverse_transpose = matrix.inverse().unwrap().transpose();

        assert_eq!(matrix_transpose_inverse, matrix_inverse_transpose);
    }

    #[test]
    fn test_matrix_inverse_inverse_equals_matrix() {
        let matrix: Matrix2<f64> = Matrix2::new(
            80.0,   426.1, 
            23.43,  23.5724
        );
        let result = matrix.inverse().unwrap().inverse().unwrap();
        let expected = matrix;
        let epsilon = 1e-7;

        assert!(relative_eq!(result, expected, epsilon = epsilon));
    }

    #[test]
    fn test_matrix_elements_should_be_column_major_order() {
        let matrix = Matrix2::new(1, 2, 3, 4);
        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c0r1, matrix[0][1]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c1r1, matrix[1][1]);
    }

    #[test]
    fn test_matrix_swap_columns() {
        let mut result = Matrix2::new(1, 2, 3, 4);
        result.swap_columns(0, 1);
        let expected = Matrix2::new(3, 4, 1, 2);
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_swap_rows() {
        let mut result = Matrix2::new(1, 2, 3, 4);
        result.swap_rows(0, 1);
        let expected = Matrix2::new(2, 1, 4, 3);
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_swap_elements() {
        let mut result = Matrix2::new(1, 2, 3, 4);
        result.swap_elements((0, 0), (1, 1));
        let expected = Matrix2::new(4, 2, 3, 1);

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix3_tests {
    use gdmath::{
        Vector3,
        Matrix3,
        One, 
        Zero, 
        Matrix,
    };
    use gdmath::approx::relative_eq;
    use std::slice::Iter;


    struct TestCase {
        a_mat: Matrix3<f32>,
        b_mat: Matrix3<f32>,
        expected: Matrix3<f32>,
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
                    a_mat: Matrix3::new(
                        80.0,   426.1,   43.393, 
                        23.43,  23.5724, 1.27, 
                        81.439, 12.19,   43.36
                    ),
                    b_mat: Matrix3::new(
                        36.84,     7.04217,  5.74, 
                        427.46894, 61.89139, 96.27, 
                        152.66,    86.333,   26.71
                    ),
                    expected: Matrix3::new(
                        3579.6579,  15933.496,   1856.4281, 
                        43487.7660, 184776.9752, 22802.0289, 
                        16410.8178, 67409.1000,  7892.1646
                    ),
                },
                TestCase {
                    a_mat: Matrix3::one(),
                    b_mat: Matrix3::one(),
                    expected: Matrix3::one(),
                },
                TestCase {
                    a_mat: Matrix3::zero(),
                    b_mat: Matrix3::zero(),
                    expected: Matrix3::zero(),
                },
                TestCase {
                    a_mat: Matrix3::new(
                        68.32, 0.0,    0.0, 
                        0.0,   37.397, 0.0, 
                        0.0,   0.0,    43.393
                    ),
                    b_mat: Matrix3::new(
                        57.72, 0.0,       0.0, 
                        0.0,   9.5433127, 0.0, 
                        0.0,   0.0,       12.19
                    ),
                    expected: Matrix3::new(
                        3943.4304, 0.0,       0.0, 
                        0.0,       356.89127, 0.0, 
                        0.0,       0.0,       528.96067
                    ),
                },
            ]
        }
    }

    #[test]
    fn test_mat_times_identity_equals_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_identity = test.a_mat * Matrix3::one();
            let b_mat_times_identity = test.b_mat * Matrix3::one();

            assert_eq!(a_mat_times_identity, test.a_mat);
            assert_eq!(b_mat_times_identity, test.b_mat);
        })
    }

    #[test]
    fn test_mat_times_zero_equals_zero() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_zero = test.a_mat * Matrix3::zero();
            let b_mat_times_zero = test.b_mat * Matrix3::zero();

            assert_eq!(a_mat_times_zero, Matrix3::zero());
            assert_eq!(b_mat_times_zero, Matrix3::zero());
        })
    }

    #[test]
    fn test_zero_times_mat_equals_zero() {
        test_cases().iter().for_each(|test| {
            let zero_times_a_mat = Matrix3::zero() * test.a_mat;
            let zero_times_b_mat = Matrix3::zero() * test.b_mat;

            assert_eq!(zero_times_a_mat, Matrix3::zero());
            assert_eq!(zero_times_b_mat, Matrix3::zero());
        })
    }

    #[test]
    fn test_mat_times_identity_equals_identity_times_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_identity = test.a_mat * Matrix3::one();
            let identity_times_a_mat = Matrix3::one() * test.a_mat;
            let b_mat_times_identity = test.b_mat * Matrix3::one();
            let identity_times_b_mat = Matrix3::one() * test.b_mat;

            assert_eq!(a_mat_times_identity, identity_times_a_mat);
            assert_eq!(b_mat_times_identity, identity_times_b_mat);
        })
    }

    #[test]
    fn test_mat_transpose_transpose_equals_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_tr_tr = test.a_mat.transpose().transpose();
            let b_mat_tr_tr = test.b_mat.transpose().transpose();
            
            assert_eq!(a_mat_tr_tr, test.a_mat);
            assert_eq!(b_mat_tr_tr, test.b_mat);
        })
    }

    #[test]
    fn test_identity_transpose_equals_identity() {
        let identity = Matrix3::<f32>::one();
        let identity_tr = identity.transpose();
            
        assert_eq!(identity, identity_tr);
    }

    #[test]
    fn test_matrix_multiplication() {
        test_cases().iter().for_each(|test| {
            let result = test.a_mat * test.b_mat;
            let expected = test.expected;

            assert_eq!(result, expected);
        })
    }

    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector3::new(1.0, 2.0, 3.0);
        let c1 = Vector3::new(4.0, 5.0, 6.0);
        let c2 = Vector3::new(7.0, 8.0, 9.0);
        let expected = Matrix3::new(
            1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0, 
            7.0, 8.0, 9.0
        );
        let result = Matrix3::from_cols(c0, c1, c2);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_constant_times_identity_is_constant_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix3::one();
        let expected = Matrix3::new(
            c,   0.0, 0.0, 
            0.0, c,   0.0, 
            0.0, 0.0, c
        );

        assert_eq!(id * c, expected);
    }

    #[test]
    fn test_identity_divide_constant_is_constant_inverse_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix3::one();
        let expected = Matrix3::new(
            1.0/c, 0.0,   0.0, 
            0.0,   1.0/c, 0.0, 
            0.0,   0.0,   1.0/c
        );

        assert_eq!(id / c, expected);
    }

    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero = Matrix3::zero();
        let matrix = Matrix3::new(
            80.0,   426.1,   43.393, 
            23.43,  23.5724, 1.27, 
            81.439, 12.19,   43.36
        );

        assert_eq!(matrix + zero, matrix);
    }

    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero = Matrix3::zero();
        let matrix = Matrix3::new(
            80.0,   426.1,   43.393, 
            23.43,  23.5724, 1.27, 
            81.439, 12.19,   43.36
        );

        assert_eq!(zero + matrix, matrix);
    }

    #[test]
    fn test_matrix_with_zero_determinant() {
        let matrix = Matrix3::new(
            1f32, 2f32, 3f32, 
            4f32, 5f32, 6f32, 
            4f32, 5f32, 6f32
        );
        
        assert_eq!(matrix.determinant(), 0.0);
    }

    #[test]
    fn test_lower_triangular_matrix_determinant() {
        let matrix: Matrix3<f64> = Matrix3::new(
            1_f64,  0_f64,  0_f64,
            5_f64,  2_f64,  0_f64,
            5_f64,  5_f64,  3_f64
        );

        assert_eq!(matrix.determinant(), 1_f64 * 2_f64 * 3_f64);
    }

    #[test]
    fn test_upper_triangular_matrix_determinant() {
        let matrix: Matrix3<f64> = Matrix3::new(
            1_f64,  5_f64,  5_f64,
            0_f64,  2_f64,  5_f64,
            0_f64,  0_f64,  3_f64
        );

        assert_eq!(matrix.determinant(), 1_f64 * 2_f64 * 3_f64);
    }

    #[test]
    fn test_matrix_inverse() {
        let matrix: Matrix3<f64> = Matrix3::new(
            5_f64, 1_f64, 1_f64,
            1_f64, 5_f64, 1_f64,
            1_f64, 1_f64, 5_f64
        );
        let expected: Matrix3<f64> = (1_f64 / 28_f64) * Matrix3::new(
             6_f64, -1_f64, -1_f64, 
            -1_f64,  6_f64, -1_f64, 
            -1_f64, -1_f64,  6_f64,
        );
        let result = matrix.inverse().unwrap();
        let epsilon = 1e-7;

        assert!(relative_eq!(result, expected, epsilon = epsilon));
    }

    #[test]
    fn test_identity_is_invertible() {
        assert!(Matrix3::<f64>::one().is_invertible());
    }

    #[test]
    fn test_identity_inverse_is_identity() {
        let result: Matrix3<f64> = Matrix3::one().inverse().unwrap();
        let expected: Matrix3<f64> = Matrix3::one();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_diagonal_matrix() {
        let matrix: Matrix3<f64> = 4_f64 * Matrix3::one();
        let expected: Matrix3<f64> = (1_f64 / 4_f64) * Matrix3::one();
        let result = matrix.inverse().unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_with_nonzero_determinant_is_invertible() {
        let matrix = Matrix3::new(
            1f32, 2f32, 3f32, 
            0f32, 4f32, 5f32, 
            0f32, 0f32, 6f32
        );
        
        assert!(matrix.is_invertible());
    }

    #[test]
    fn test_matrix_with_zero_determinant_is_not_invertible() {
        let matrix = Matrix3::new(
            1f32, 2f32, 3f32, 
            4f32, 5f32, 6f32, 
            4f32, 5f32, 6f32
        );
        
        assert!(!matrix.is_invertible());
    }

    #[test]
    fn test_noninvertible_matrix_returns_none() {
        let matrix = Matrix3::new(
            1f32, 2f32, 3f32, 
            4f32, 5f32, 6f32, 
            4f32, 5f32, 6f32
        );
        
        assert!(matrix.inverse().is_none());
    }

    #[test]
    fn test_matrix_times_inverse_is_identity() {
        let matrix = Matrix3::new(
            80.0,   426.1,   43.393, 
            23.43,  23.5724, 1.27, 
            81.439, 12.19,   43.36
        );
        let matrix_inv = matrix.inverse().unwrap();
        let one = Matrix3::one();

        assert!(relative_eq!(matrix * matrix_inv, one, epsilon = 1e-7));
    }

    #[test]
    fn test_constant_times_matrix_inverse_equals_constant_inverse_times_matrix_inverse() {
        let matrix: Matrix3<f64> = Matrix3::new(
            80.0,   426.1,   43.393, 
            23.43,  23.5724, 1.27, 
            81.439, 12.19,   43.36
        );
        let constant: f64 = 4_f64;
        let constant_times_matrix_inverse = (constant * matrix).inverse().unwrap();
        let constant_inverse_times_matrix_inverse = (1_f64 / constant) * matrix.inverse().unwrap();

        assert_eq!(constant_times_matrix_inverse, constant_inverse_times_matrix_inverse);
    }

    #[test]
    fn test_matrix_transpose_inverse_equals_matrix_inverse_transpose() {
        let matrix: Matrix3<f64> = Matrix3::new(
            80.0,   426.1,   43.393, 
            23.43,  23.5724, 1.27, 
            81.439, 12.19,   43.36
        );
        let matrix_transpose_inverse = matrix.transpose().inverse().unwrap();
        let matrix_inverse_transpose = matrix.inverse().unwrap().transpose();

        assert_eq!(matrix_transpose_inverse, matrix_inverse_transpose);
    }

    #[test]
    fn test_inverse_times_matrix_is_identity() {
        let matrix = Matrix3::new(
            80.0,   426.1,   43.393, 
            23.43,  23.5724, 1.27, 
            81.439, 12.19,   43.36
        );
        let matrix_inv = matrix.inverse().unwrap();
        let one = Matrix3::one();

        assert!(relative_eq!(matrix_inv * matrix, one, epsilon = 1e-7));
    }

    #[test]
    fn test_matrix_inverse_inverse_equals_matrix() {
        let matrix: Matrix3<f64> = Matrix3::new(
            80.0,   426.1,   43.393, 
            23.43,  23.5724, 1.27, 
            81.439, 12.19,   43.36
        );
        let result = matrix.inverse().unwrap().inverse().unwrap();
        let expected = matrix;
        let epsilon = 1e-7;

        assert!(relative_eq!(result, expected, epsilon = epsilon));
    }

    #[test]
    fn test_matrix_elements_should_be_column_major_order() {
        let matrix = Matrix3::new(
            1, 2, 3, 
            4, 5, 6, 
            7, 8, 9
        );

        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c0r1, matrix[0][1]);
        assert_eq!(matrix.c0r2, matrix[0][2]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c1r1, matrix[1][1]);
        assert_eq!(matrix.c1r2, matrix[1][2]);
        assert_eq!(matrix.c2r0, matrix[2][0]);
        assert_eq!(matrix.c2r1, matrix[2][1]);
        assert_eq!(matrix.c2r2, matrix[2][2]);
    }

    #[test]
    fn test_matrix_swap_columns() {
        let mut result = Matrix3::new(1, 2, 3, 4, 5, 6, 7, 8, 9);
        result.swap_columns(0, 1);
        let expected = Matrix3::new(4, 5, 6, 1, 2, 3, 7, 8, 9);
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_swap_rows() {
        let mut result = Matrix3::new(1, 2, 3, 4, 5, 6, 7, 8, 9);
        result.swap_rows(0, 1);
        let expected = Matrix3::new(2, 1, 3, 5, 4, 6, 8, 7, 9);
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_swap_elements() {
        let mut result = Matrix3::new(1, 2, 3, 4, 5, 6, 7, 8, 9);
        result.swap_elements((0, 0), (2, 1));
        let expected = Matrix3::new(8, 2, 3, 4, 5, 6, 7, 1, 9);

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod matrix4_tests {
    use gdmath::{
        Vector3,
        Vector4,
        Matrix4,
        One, 
        Zero, 
        Matrix,
    };
    use gdmath::approx::relative_eq;
    use std::slice::Iter;


    struct TestCase {
        a_mat: Matrix4<f64>,
        b_mat: Matrix4<f64>,
        expected: Matrix4<f64>,
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
                    a_mat: Matrix4::new(
                        80.0,  23.43,  43.56, 6.74, 
                        426.1, 23.57,  27.61, 13.90,
                        4.22,  258.08, 31.70, 42.17, 
                        70.0,  49.0,   95.0,  89.91
                    ),
                    b_mat: Matrix4::new(
                        36.84, 427.46, 882.19, 89.50, 
                        7.04,  61.89,  56.31,  89.0, 
                        72.0,  936.5,  413.80, 50.31,  
                        37.69, 311.8,  60.81,  73.83
                    ),
                    expected: Matrix4::new(
                        195075.7478, 242999.4886, 49874.8440, 51438.8929,
                        33402.1572,  20517.1793,  12255.4723, 11284.3033,
                        410070.5860, 133018.9590, 46889.9950, 35475.9481,
                        141297.8982, 27543.7175,  19192.1014, 13790.4636
                    ),
                },
                TestCase {
                    a_mat: Matrix4::one(),
                    b_mat: Matrix4::one(),
                    expected: Matrix4::one(),
                },
                TestCase {
                    a_mat: Matrix4::zero(),
                    b_mat: Matrix4::zero(),
                    expected: Matrix4::zero(),
                },
                TestCase {
                    a_mat: Matrix4::new(
                        68.32, 0.0,    0.0,   0.0,
                        0.0,   37.397, 0.0,   0.0,
                        0.0,   0.0,    9.483, 0.0,
                        0.0,   0.0,    0.0,   887.710
                    ),
                    b_mat: Matrix4::new(
                        57.72, 0.0,    0.0,     0.0, 
                        0.0,   9.5433, 0.0,     0.0, 
                        0.0,   0.0,    86.7312, 0.0,
                        0.0,   0.0,    0.0,     269.1134
                    ),
                    expected: Matrix4::new(
                        3943.4304, 0.0,         0.0,         0.0,
                        0.0,       356.8907901, 0.0,         0.0,
                        0.0,       0.0,         822.4719696, 0.0,
                        0.0,       0.0,         0.0,         238894.656314
                    ),
                },
            ]
        }
    }

    #[test]
    fn test_mat_times_identity_equals_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_identity = test.a_mat * Matrix4::one();
            let b_mat_times_identity = test.b_mat * Matrix4::one();

            assert_eq!(a_mat_times_identity, test.a_mat);
            assert_eq!(b_mat_times_identity, test.b_mat);
        })
    }

    #[test]
    fn test_mat_times_zero_equals_zero() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_zero = test.a_mat * Matrix4::zero();
            let b_mat_times_zero = test.b_mat * Matrix4::zero();

            assert_eq!(a_mat_times_zero, Matrix4::zero());
            assert_eq!(b_mat_times_zero, Matrix4::zero());
        })
    }

    #[test]
    fn test_zero_times_mat_equals_zero() {
        test_cases().iter().for_each(|test| {
            let zero_times_a_mat = Matrix4::zero() * test.a_mat;
            let zero_times_b_mat = Matrix4::zero() * test.b_mat;

            assert_eq!(zero_times_a_mat, Matrix4::zero());
            assert_eq!(zero_times_b_mat, Matrix4::zero());
        })
    }

    #[test]
    fn test_mat_times_identity_equals_identity_times_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_identity = test.a_mat * Matrix4::one();
            let identity_times_a_mat = Matrix4::one() * test.a_mat;
            let b_mat_times_identity = test.b_mat * Matrix4::one();
            let identity_times_b_mat = Matrix4::one() * test.b_mat;

            assert_eq!(a_mat_times_identity, identity_times_a_mat);
            assert_eq!(b_mat_times_identity, identity_times_b_mat);
        })
    }

    #[test]
    fn test_mat_transpose_transpose_equals_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_tr_tr = test.a_mat.transpose().transpose();
            let b_mat_tr_tr = test.b_mat.transpose().transpose();
            
            assert_eq!(a_mat_tr_tr, test.a_mat);
            assert_eq!(b_mat_tr_tr, test.b_mat);
        })
    }

    #[test]
    fn test_identity_transpose_equals_identity() {
        let identity = Matrix4::<f32>::one();
        let identity_tr = identity.transpose();
            
        assert_eq!(identity, identity_tr);
    }

    #[test]
    fn test_matrix_multiplication() {
        test_cases().iter().for_each(|test| {
            let result = test.a_mat * test.b_mat;
            let expected = test.expected;
            let epsilon = 1e-7;

            assert!(relative_eq!(result, expected, epsilon = epsilon));
        })
    }

    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector4::new(1, 2, 3, 4);
        let c1 = Vector4::new(5, 6, 7, 8);
        let c2 = Vector4::new(9, 10, 11, 12);
        let c3 = Vector4::new(13, 14, 15, 16);
        let expected = Matrix4::new(
            1,  2,  3,  4, 
            5,  6,  7,  8, 
            9,  10, 11, 12, 
            13, 14 ,15, 16
        );
        let result = Matrix4::from_cols(c0, c1, c2, c3);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_identity_mat4_translates_vector_along_vector() {
        let v = Vector3::from((2.0, 2.0, 2.0));
        let trans_mat = Matrix4::from_translation(v);
        let zero_vec4 = Vector4::from((0.0, 0.0, 0.0, 1.0));
        let zero_vec3 = Vector3::from((0.0, 0.0, 0.0));

        let result = trans_mat * zero_vec4;
        assert_eq!(result, Vector4::from((zero_vec3 + v, 1.0)));
    }

    #[test]
    fn test_constant_times_identity_is_constant_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix4::one();
        let expected = Matrix4::new(
            c,   0.0, 0.0, 0.0, 
            0.0, c,   0.0, 0.0, 
            0.0, 0.0, c,   0.0, 
            0.0, 0.0, 0.0, c
        );

        assert_eq!(id * c, expected);
    }

    #[test]
    fn test_identity_divide_constant_is_constant_inverse_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix4::one();
        let expected = Matrix4::new(
            1.0 / c, 0.0,     0.0,     0.0, 
            0.0,     1.0 / c, 0.0,     0.0, 
            0.0,     0.0,     1.0 / c, 0.0, 
            0.0,     0.0,     0.0,     1.0 / c
        );

        assert_eq!(id / c, expected);
    }

    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero = Matrix4::zero();
        let matrix = Matrix4::new(
            36.84,   427.46894, 8827.1983, 89.5049494, 
            7.04217, 61.891390, 56.31,     89.0, 
            72.0,    936.5,     413.80,    50.311160,  
            37.6985,  311.8,    60.81,     73.8393
        );

        assert_eq!(matrix + zero, matrix);
    }

    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero = Matrix4::zero();
        let matrix = Matrix4::new(
            36.84,   427.46894, 8827.1983, 89.5049494, 
            7.04217, 61.891390, 56.31,     89.0, 
            72.0,    936.5,     413.80,    50.311160,  
            37.6985,  311.8,    60.81,     73.8393
        );

        assert_eq!(zero + matrix, matrix);
    }

    #[test]
    fn test_matrix_with_zero_determinant() {
        // This matrix should have a zero determinant since it has two repeating columns.
        use num_traits::Zero;
        let matrix: Matrix4<f64> = Matrix4::new(
            1_f64,  2_f64,  3_f64,  4_f64, 
            5_f64,  6_f64,  7_f64,  8_f64,
            5_f64,  6_f64,  7_f64,  8_f64, 
            9_f64,  10_f64, 11_f64, 12_f64
        );
        
        assert!(matrix.determinant().is_zero());
    }

    #[test]
    fn test_lower_triangular_matrix_determinant() {
        let matrix: Matrix4<f64> = Matrix4::new(
            1_f64,  0_f64,  0_f64,  0_f64, 
            5_f64,  2_f64,  0_f64,  0_f64,
            5_f64,  5_f64,  3_f64,  0_f64, 
            5_f64,  5_f64,  5_f64,  4_f64
        );

        assert_eq!(matrix.determinant(), 1_f64 * 2_f64 * 3_f64 * 4_f64);
    }

    #[test]
    fn test_upper_triangular_matrix_determinant() {
        let matrix: Matrix4<f64> = Matrix4::new(
            1_f64,  5_f64,  5_f64,  5_f64, 
            0_f64,  2_f64,  5_f64,  5_f64,
            0_f64,  0_f64,  3_f64,  5_f64, 
            0_f64,  0_f64,  0_f64,  4_f64
        );

        assert_eq!(matrix.determinant(), 1_f64 * 2_f64 * 3_f64 * 4_f64);
    }

    #[test]
    fn test_scalar_multiplication() {
        let result: Matrix4<f64> = (1_f64 / 32_f64) * Matrix4::new(
            7_f64, -1_f64, -1_f64, -1_f64,
           -1_f64,  7_f64, -1_f64, -1_f64,
           -1_f64, -1_f64,  7_f64, -1_f64,
           -1_f64, -1_f64, -1_f64,  7_f64
       );
       let expected: Matrix4<f64> = Matrix4::new(
        (1_f64 / 32_f64) *  7_f64, (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) * -1_f64,
        (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) *  7_f64, (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) * -1_f64,
        (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) *  7_f64, (1_f64 / 32_f64) * -1_f64,
        (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) *  7_f64
       );

       assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_inverse() {
        let matrix: Matrix4<f64> = Matrix4::new(
            5_f64, 1_f64, 1_f64, 1_f64, 
            1_f64, 5_f64, 1_f64, 1_f64,
            1_f64, 1_f64, 5_f64, 1_f64,
            1_f64, 1_f64, 1_f64, 5_f64, 
        );
        let expected: Matrix4<f64> = (1_f64 / 32_f64) * Matrix4::new(
             7_f64, -1_f64, -1_f64, -1_f64,
            -1_f64,  7_f64, -1_f64, -1_f64,
            -1_f64, -1_f64,  7_f64, -1_f64,
            -1_f64, -1_f64, -1_f64,  7_f64
        );
        let result = matrix.inverse().unwrap();
        let epsilon = 1e-7;

        assert!(relative_eq!(result, expected, epsilon = epsilon),
            "\nmatrix = {:?}\nmatrix_inv = {:?}\nmatrix * matrix_inv = {:?}\nexpected = {:?}\nepsilon = {:?}\n",
            matrix, result, matrix * result, expected, epsilon
        );
    }

    #[test]
    fn test_identity_is_invertible() {
        assert!(Matrix4::<f64>::one().is_invertible());
    }

    #[test]
    fn test_identity_inverse_is_identity() {
        let result: Matrix4<f64> = Matrix4::one().inverse().unwrap();
        let expected: Matrix4<f64> = Matrix4::one();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_diagonal_matrix() {
        let matrix: Matrix4<f64> = 4_f64 * Matrix4::one();
        let expected: Matrix4<f64> = (1_f64 / 4_f64) * Matrix4::one();
        let result = matrix.inverse().unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_with_nonzero_determinant_is_invertible() {
        let matrix = Matrix4::new(
            1_f64,  2_f64,  3_f64,   4_f64,
            5_f64,  60_f64, 7_f64,   8_f64,
            9_f64,  10_f64, 11_f64,  12_f64,
            13_f64, 14_f64, 150_f64, 16_f64
        );
        
        assert!(matrix.is_invertible());
    }

    #[test]
    fn test_matrix_with_zero_determinant_is_not_invertible() {
        // This matrix should not be invertible since it has two identical columns.
        let matrix: Matrix4<f64> = Matrix4::new(
            1_f64,  2_f64,   3_f64,  4_f64, 
            5_f64,  6_f64,   7_f64,  8_f64,
            5_f64,  6_f64,   7_f64,  8_f64, 
            9_f64,  10_f64,  11_f64, 12_f64
        );
        
        assert!(!matrix.is_invertible());
    }

    #[test]
    fn test_noninvertible_matrix_returns_none() {
        let matrix: Matrix4<f64> = Matrix4::new(
            1_f64,  2_f64,  3_f64,  4_f64, 
            5_f64,  6_f64,  7_f64,  8_f64,
            5_f64,  6_f64,  7_f64,  8_f64, 
            9_f64,  10_f64, 11_f64, 12_f64
        );
        
        assert!(matrix.inverse().is_none());
    }

    #[test]
    fn test_matrix_inversion2() {
        let matrix: Matrix4<f64> = Matrix4::new(
            36.84,   427.468, 882.198,  89.504, 
            7.042,   61.891,  56.31,    89.0, 
            72.0,    936.5,   413.80,   50.311,  
            37.698,  311.8,   60.81,    73.839
        );
        let result = matrix.inverse().unwrap();
        let expected: Matrix4<f64> = Matrix4::new(
             0.01146093272878252,  -0.06212100841992658, -0.02771783718075694,    0.07986947998777854,
            -0.00148039611514755,   0.004464130960444646, 0.003417891441120325,  -0.005915083057511776,
             0.001453087396607042, -0.0009538600348427,  -0.0005129477357421059, -0.0002621470728476185,
            -0.0007967195911958656, 0.01365031989418242,  0.0001408581712825875, -0.002040325515611523
        );
        let epsilon = 1e-7;

        assert!(relative_eq!(result, expected, epsilon = epsilon));
    }

    #[test]
    fn test_matrix_times_inverse_is_identity() {
        let matrix: Matrix4<f64> = Matrix4::new(
            36.84,  427.468, 882.198, 89.504, 
            7.042 , 61.891,  56.31,   89.0, 
            72.0,   936.5,   413.80,  50.311,  
            37.698, 311.8,   60.81,   73.839
        );
        let matrix_inv = matrix.inverse().unwrap();
        let one = Matrix4::one();
        let epsilon = 1e-7;

        assert!(relative_eq!(matrix * matrix_inv, one, epsilon = epsilon),
            "\nmatrix = {:?}\nmatrix_inv = {:?}\nmmatrix * matrix_inv = {:?}\nepsilon = {:?}\n",
            matrix, matrix_inv, matrix * matrix_inv, epsilon
        );
    }

    #[test]
    fn test_constant_times_matrix_inverse_equals_constant_inverse_times_matrix_inverse() {
        let matrix: Matrix4<f64> = Matrix4::new(
            36.84,  427.468, 882.198, 89.504, 
            7.042 , 61.891,  56.31,   89.0, 
            72.0,   936.5,   413.80,  50.311,  
            37.698, 311.8,   60.81,   73.839
        );
        let constant: f64 = 4_f64;
        let constant_times_matrix_inverse = (constant * matrix).inverse().unwrap();
        let constant_inverse_times_matrix_inverse = (1_f64 / constant) * matrix.inverse().unwrap();

        assert_eq!(constant_times_matrix_inverse, constant_inverse_times_matrix_inverse);
    }

    /// Test whether the inverse of the transpose of a matrix is approximately equal to the 
    /// transpose of the inverse of a matrix. when the matrices are defined over the real numbers,
    /// we have the equality
    /// ```
    /// Inverse(Transpose(M)) == Transpose(Inverse(M)).
    /// ```
    /// The equality does not hold over a set of floating point numbers because floating point arithmetic
    /// is not commutative, so we cannot guarantee exact equality even though transposing a matrix does not
    /// cause a loss of precesion in the matrix entries. We can only guarantee approximate equality.
    #[test]
    fn test_matrix_transpose_inverse_equals_matrix_inverse_transpose() {
        let matrix: Matrix4<f64> = Matrix4::new(
            36.84,  427.468, 882.198, 89.504, 
            7.042 , 61.891,  56.31,   89.0, 
            72.0,   936.5,   413.80,  50.311,  
            37.698, 311.8,   60.81,   73.839
        );
        let matrix_transpose_inverse = matrix.transpose().inverse().unwrap();
        let matrix_inverse_transpose = matrix.inverse().unwrap().transpose();
        let epsilon = 1e-7;

        assert!(relative_eq!(matrix_transpose_inverse, matrix_inverse_transpose, epsilon = epsilon));
    }

    #[test]
    fn test_inverse_times_matrix_is_identity() {
        let matrix: Matrix4<f64> = Matrix4::new(
            36.84,   427.468, 882.198,  89.504, 
            7.042,   61.891,  56.31,    89.0, 
            72.0,    936.5,   413.80,   50.311,  
            37.698,  311.8,   60.81,    73.839
        );
        let matrix_inv = matrix.inverse().unwrap();
        let one = Matrix4::one();
        let epsilon = 1e-7;
        
        assert!(relative_eq!(matrix_inv * matrix, one, epsilon = epsilon),
            "\nmatrix = {:?}\nmatrix_inv = {:?}\nmatrix_inv * matrix = {:?}\nepsilon = {:?}\n",
            matrix, matrix_inv, matrix_inv * matrix, epsilon
        );
    }

    #[test]
    fn test_matrix_inverse_inverse_equals_matrix() {
        let matrix: Matrix4<f64> = Matrix4::new(
            36.84,  427.468, 882.198, 89.504, 
            7.042 , 61.891,  56.31,   89.0, 
            72.0,   936.5,   413.80,  50.311,  
            37.698, 311.8,   60.81,   73.839
        );
        let result = matrix.inverse().unwrap().inverse().unwrap();
        let expected = matrix;
        let epsilon = 1e-7;

        assert!(relative_eq!(result, expected, epsilon = epsilon));
    }

    #[test]
    fn test_matrix_elements_should_be_column_major_order() {
        let matrix = Matrix4::new(
            1,  2,  3,  4, 
            5,  6,  7,  8, 
            9,  10, 11, 12, 
            13, 14, 15, 16
        );
        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c0r1, matrix[0][1]);
        assert_eq!(matrix.c0r2, matrix[0][2]);
        assert_eq!(matrix.c0r3, matrix[0][3]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c1r1, matrix[1][1]);
        assert_eq!(matrix.c1r2, matrix[1][2]);
        assert_eq!(matrix.c1r3, matrix[1][3]);
        assert_eq!(matrix.c2r0, matrix[2][0]);
        assert_eq!(matrix.c2r1, matrix[2][1]);
        assert_eq!(matrix.c2r2, matrix[2][2]);
        assert_eq!(matrix.c2r3, matrix[2][3]);
        assert_eq!(matrix.c3r0, matrix[3][0]);
        assert_eq!(matrix.c3r1, matrix[3][1]);
        assert_eq!(matrix.c3r2, matrix[3][2]);
        assert_eq!(matrix.c3r3, matrix[3][3]);
    }

    #[test]
    fn test_matrix_swap_columns() {
        let mut result = Matrix4::new(
            1,  2,  3,   4, 
            5,  6,  7,   8, 
            9,  10, 11,  12,
            13, 14, 15,  16
        );
        result.swap_columns(3, 1);
        let expected = Matrix4::new(
            1,  2,  3,  4,
            13, 14, 15, 16,
            9,  10, 11, 12,
            5,  6,  7,  8 
        );
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_swap_rows() {
        let mut result = Matrix4::new(
            1,  2,  3,  4, 
            5,  6,  7,  8, 
            9,  10, 11, 12, 
            13, 14, 15, 16
        );
        result.swap_rows(3, 1);
        let expected = Matrix4::new(
            1,  4,  3,  2,
            5,  8,  7,  6,
            9,  12, 11, 10,
            13, 16, 15, 14
        );
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_swap_elements() {
        let mut result = Matrix4::new(
            1,  2,  3,  4, 
            5,  6,  7,  8, 
            9,  10, 11, 12,
            13, 14, 15, 16
        );
        result.swap_elements((2, 0), (1, 3));
        let expected = Matrix4::new(
            1,  2,  3,  4,
            5,  6,  7,  9,
            8,  10, 11, 12,
            13, 14, 15, 16
        );

        assert_eq!(result, expected);
    }
}
