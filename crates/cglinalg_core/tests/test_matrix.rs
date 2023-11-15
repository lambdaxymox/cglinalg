#[cfg(test)]
mod matrix2x2_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix2x2,
        Vector2,
    };


    #[rustfmt::skip]
    #[test]
    fn test_matrix_components1() {
        let matrix = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );

        assert_eq!(matrix[0][0], 1_i32);
        assert_eq!(matrix[0][1], 2_i32);
        assert_eq!(matrix[1][0], 3_i32);
        assert_eq!(matrix[1][1], 4_i32);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_components2() {
        let matrix = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );

        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c0r1, matrix[0][1]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c1r1, matrix[1][1]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds1() {
        let matrix = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );

        assert_eq!(matrix[2][0], matrix[2][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds2() {
        let matrix = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );

        assert_eq!(matrix[0][2], matrix[0][2]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds3() {
        let matrix = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );

        assert_eq!(matrix[2][2], matrix[2][2]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds4() {
        let matrix = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );

        assert_eq!(matrix[0][usize::MAX], matrix[0][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds5() {
        let matrix = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );

        assert_eq!(matrix[usize::MAX][0], matrix[usize::MAX][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds6() {
        let matrix = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );

        assert_eq!(matrix[usize::MAX][usize::MAX], matrix[usize::MAX][usize::MAX]);
    }

    #[test]
    fn test_identity_matrix_times_identity_matrix_equals_identity_matrix() {
        let identity_matrix: Matrix2x2<f32> = Matrix2x2::identity();

        assert_eq!(identity_matrix * identity_matrix, identity_matrix);
    }

    #[test]
    fn test_zero_matrix_times_zero_matrix_equals_zero_matrix() {
        let zero_matrix: Matrix2x2<f32> = Matrix2x2::zero();

        assert_eq!(zero_matrix * zero_matrix, zero_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_matrix1() {
        let a_matrix = Matrix2x2::new(
            80_f32,      23.43_f32,
            426.1_f32,   23.5724_f32,
        );
        let b_matrix = Matrix2x2::new(
            36.84_f32,   427.46894_f32,
            7.04217_f32, 61.891390_f32,
        );
        // let expected = Matrix2x2::new(
        //     185091.72_f32, 10939.63_f32,
        //     26935.295_f32, 1623.9266_f32,
        // );
        let a_matrix_times_identity = a_matrix * Matrix2x2::identity();
        let b_matrix_times_identity = b_matrix * Matrix2x2::identity();

        assert_eq!(a_matrix_times_identity, a_matrix);
        assert_eq!(b_matrix_times_identity, b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_zero_equals_zero1() {
        let a_matrix = Matrix2x2::new(
            80_f32,      23.43_f32,
            426.1_f32,   23.5724_f32,
        );
        let b_matrix = Matrix2x2::new(
            36.84_f32,   427.46894_f32,
            7.04217_f32, 61.891390_f32,
        );
        // let expected = Matrix2x2::new(
        //     185091.72_f32, 10939.63_f32,
        //     26935.295_f32, 1623.9266_f32,
        // );
        let a_matrix_times_zero_matrix = a_matrix * Matrix2x2::zero();
        let b_matrix_times_zero_matrix = b_matrix * Matrix2x2::zero();

        assert_eq!(a_matrix_times_zero_matrix, Matrix2x2::zero());
        assert_eq!(b_matrix_times_zero_matrix, Matrix2x2::zero());
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_times_matrix_equals_zero1() {
        let a_matrix = Matrix2x2::new(
            80_f32,      23.43_f32,
            426.1_f32,   23.5724_f32,
        );
        let b_matrix = Matrix2x2::new(
            36.84_f32,   427.46894_f32,
            7.04217_f32, 61.891390_f32,
        );
        // let expected = Matrix2x2::new(
        //     185091.72_f32, 10939.63_f32,
        //     26935.295_f32, 1623.9266_f32,
        // );
        let zero_times_a_matrix = Matrix2x2::zero() * a_matrix;
        let zero_times_b_matrix = Matrix2x2::zero() * b_matrix;

        assert_eq!(zero_times_a_matrix, Matrix2x2::zero());
        assert_eq!(zero_times_b_matrix, Matrix2x2::zero());
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_identity_times_matrix1() {
        let a_matrix = Matrix2x2::new(
            80_f32,      23.43_f32,
            426.1_f32,   23.5724_f32,
        );
        let b_matrix = Matrix2x2::new(
            36.84_f32,   427.46894_f32,
            7.04217_f32, 61.891390_f32,
        );
        // let expected = Matrix2x2::new(
        //     185091.72_f32, 10939.63_f32,
        //     26935.295_f32, 1623.9266_f32,
        // );
        let a_matrix_times_identity = a_matrix * Matrix2x2::identity();
        let identity_times_a_matrix = Matrix2x2::identity() * a_matrix;
        let b_matrix_times_identity = b_matrix * Matrix2x2::identity();
        let identity_times_b_matrix = Matrix2x2::identity() * b_matrix;

        assert_eq!(a_matrix_times_identity, identity_times_a_matrix);
        assert_eq!(b_matrix_times_identity, identity_times_b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose_transpose_equals_matrix1() {
        let a_matrix = Matrix2x2::new(
            80_f32,      23.43_f32,
            426.1_f32,   23.5724_f32,
        );
        let b_matrix = Matrix2x2::new(
            36.84_f32,   427.46894_f32,
            7.04217_f32, 61.891390_f32,
        );
        // let expected = Matrix2x2::new(
        //     185091.72_f32, 10939.63_f32,
        //     26935.295_f32, 1623.9266_f32,
        // );
        let a_matrix_transpose_transpose = a_matrix.transpose().transpose();
        let b_matrix_transpose_transpose = b_matrix.transpose().transpose();

        assert_eq!(a_matrix_transpose_transpose, a_matrix);
        assert_eq!(b_matrix_transpose_transpose, b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication1() {
        let a_matrix = Matrix2x2::new(
            80_f32,      23.43_f32,
            426.1_f32,   23.5724_f32,
        );
        let b_matrix = Matrix2x2::new(
            36.84_f32,   427.46894_f32,
            7.04217_f32, 61.891390_f32,
        );
        let expected = Matrix2x2::new(
            185091.72_f32, 10939.63_f32,
            26935.295_f32, 1623.9266_f32,
        );
        let result = a_matrix * b_matrix;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f32::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_matrix2() {
        let a_matrix = Matrix2x2::new(
            68.32_f32, 0_f32,
            0_f32,     37.397_f32,
        );
        let b_matrix = Matrix2x2::new(
            57.72_f32, 0_f32,
            0_f32,     9.5433127_f32,
        );
        // let expected = Matrix2x2::new(
        //     3943.4304_f32, 0_f32,
        //     0_f32,         356.89127_f32,
        // );
        let a_matrix_times_identity = a_matrix * Matrix2x2::identity();
        let b_matrix_times_identity = b_matrix * Matrix2x2::identity();

        assert_eq!(a_matrix_times_identity, a_matrix);
        assert_eq!(b_matrix_times_identity, b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_zero_equals_zero2() {
        let a_matrix = Matrix2x2::new(
            68.32_f32, 0_f32,
            0_f32,     37.397_f32,
        );
        let b_matrix = Matrix2x2::new(
            57.72_f32, 0_f32,
            0_f32,     9.5433127_f32,
        );
        // let expected = Matrix2x2::new(
        //     3943.4304_f32, 0_f32,
        //     0_f32,         356.89127_f32,
        // );
        let a_matrix_times_zero_matrix = a_matrix * Matrix2x2::zero();
        let b_matrix_times_zero_matrix = b_matrix * Matrix2x2::zero();

        assert_eq!(a_matrix_times_zero_matrix, Matrix2x2::zero());
        assert_eq!(b_matrix_times_zero_matrix, Matrix2x2::zero());
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_times_matrix_equals_zero2() {
        let a_matrix = Matrix2x2::new(
            68.32_f32, 0_f32,
            0_f32,     37.397_f32,
        );
        let b_matrix = Matrix2x2::new(
            57.72_f32, 0_f32,
            0_f32,     9.5433127_f32,
        );
        // let expected = Matrix2x2::new(
        //     3943.4304_f32, 0_f32,
        //     0_f32,         356.89127_f32,
        // );
        let zero_times_a_matrix = Matrix2x2::zero() * a_matrix;
        let zero_times_b_matrix = Matrix2x2::zero() * b_matrix;

        assert_eq!(zero_times_a_matrix, Matrix2x2::zero());
        assert_eq!(zero_times_b_matrix, Matrix2x2::zero());
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_identity_times_matrix2() {
        let a_matrix = Matrix2x2::new(
            68.32_f32, 0_f32,
            0_f32,     37.397_f32,
        );
        let b_matrix = Matrix2x2::new(
            57.72_f32, 0_f32,
            0_f32,     9.5433127_f32,
        );
        // let expected = Matrix2x2::new(
        //     3943.4304_f32, 0_f32,
        //     0_f32,         356.89127_f32,
        // );
        let a_matrix_times_identity = a_matrix * Matrix2x2::identity();
        let identity_times_a_matrix = Matrix2x2::identity() * a_matrix;
        let b_matrix_times_identity = b_matrix * Matrix2x2::identity();
        let identity_times_b_matrix = Matrix2x2::identity() * b_matrix;

        assert_eq!(a_matrix_times_identity, identity_times_a_matrix);
        assert_eq!(b_matrix_times_identity, identity_times_b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose_transpose_equals_matrix2() {
        let a_matrix = Matrix2x2::new(
            68.32_f32, 0_f32,
            0_f32,     37.397_f32,
        );
        let b_matrix = Matrix2x2::new(
            57.72_f32, 0_f32,
            0_f32,     9.5433127_f32,
        );
        // let expected = Matrix2x2::new(
        //     3943.4304_f32, 0_f32,
        //     0_f32,         356.89127_f32,
        // );
        let a_matrix_transpose_transpose = a_matrix.transpose().transpose();
        let b_matrix_transpose_transpose = b_matrix.transpose().transpose();

        assert_eq!(a_matrix_transpose_transpose, a_matrix);
        assert_eq!(b_matrix_transpose_transpose, b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication2() {
        let a_matrix = Matrix2x2::new(
            68.32_f32, 0_f32,
            0_f32,     37.397_f32,
        );
        let b_matrix = Matrix2x2::new(
            57.72_f32, 0_f32,
            0_f32,     9.5433127_f32,
        );
        let expected = Matrix2x2::new(
            3943.4304_f32, 0_f32,
            0_f32,         356.89127_f32,
        );
        let result = a_matrix * b_matrix;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f32::EPSILON);
    }

    #[test]
    fn test_identity_transpose_equals_identity() {
        let identity = Matrix2x2::<f32>::identity();
        let identity_transpose = identity.transpose();

        assert_eq!(identity, identity_transpose);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector2::new(1_i32, 2_i32);
        let c1 = Vector2::new(3_i32, 4_i32);
        let columns = [c0, c1];
        let expected = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );
        let result = Matrix2x2::from_columns(&columns);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_rows() {
        let r0 = Vector2::new(1_i32, 2_i32);
        let r1 = Vector2::new(3_i32, 4_i32);
        let rows = [r0, r1];
        let expected = Matrix2x2::new(
            1_i32, 3_i32,
            2_i32, 4_i32,
        );
        let result = Matrix2x2::from_rows(&rows);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_constant_times_identity_is_constant_along_diagonal() {
        let c = 802.3435169_f64;
        let identity = Matrix2x2::identity();
        let expected = Matrix2x2::new(
            c,     0_f64,
            0_f64, c,
        );

        assert_eq!(identity * c, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_identity_divide_constant_is_constant_inverse_along_diagonal() {
        let c = 802.3435169_f64;
        let identity = Matrix2x2::identity();
        let expected = Matrix2x2::new(
            1_f64 / c, 0_f64,
            0_f64,     1_f64 / c,
        );

        assert_eq!(identity / c, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero_matrix = Matrix2x2::zero();
        let matrix = Matrix2x2::new(
            36.84_f64, 427.46_f64,
            7.47_f64,  61.89_f64,
        );

        assert_eq!(matrix + zero_matrix, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero_matrix = Matrix2x2::zero();
        let matrix = Matrix2x2::new(
            36.84_f64, 427.46_f64,
            7.47_f64,  61.89_f64,
        );

        assert_eq!(zero_matrix + matrix, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_with_zero_determinant() {
        let matrix = Matrix2x2::new(
            1_f64, 2_f64,
            4_f64, 8_f64,
        );

        assert_eq!(matrix.determinant(), 0_f64);
    }

    #[rustfmt::skip]
    #[test]
    fn test_lower_triangular_matrix_determinant() {
        let matrix = Matrix2x2::new(
            2_f64,  0_f64,
            5_f64,  3_f64,
        );

        assert_eq!(matrix.determinant(), 2_f64 * 3_f64);
    }

    #[rustfmt::skip]
    #[test]
    fn test_upper_triangular_matrix_determinant() {
        let matrix = Matrix2x2::new(
            2_f64,  5_f64,
            0_f64,  3_f64,
        );

        assert_eq!(matrix.determinant(), 2_f64 * 3_f64);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_inverse() {
        let matrix = Matrix2x2::new(
            5_f64, 1_f64,
            1_f64, 5_f64,
        );
        let expected = (1_f64 / 24_f64) * Matrix2x2::new(
             5_f64, -1_f64,
            -1_f64,  5_f64,
        );
        let result = matrix.try_inverse().unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_identity_is_invertible() {
        let identity: Matrix2x2<f64> = Matrix2x2::identity();

        assert!(identity.is_invertible());
    }

    #[test]
    fn test_identity_inverse_is_identity() {
        let result: Matrix2x2<f64> = Matrix2x2::identity().try_inverse().unwrap();
        let expected: Matrix2x2<f64> = Matrix2x2::identity();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_diagonal_matrix() {
        let matrix = 4_f64 * Matrix2x2::identity();
        let expected = (1_f64 / 4_f64) * Matrix2x2::identity();
        let result = matrix.try_inverse().unwrap();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_with_nonzero_determinant_is_invertible() {
        let matrix = Matrix2x2::new(
            1_f32, 2_f32,
            3_f32, 4_f32,
        );

        assert!(matrix.is_invertible());
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_with_zero_determinant_is_not_invertible() {
        let matrix = Matrix2x2::new(
            1_f32, 2_f32,
            4_f32, 8_f32,
        );

        assert!(!matrix.is_invertible());
    }

    #[rustfmt::skip]
    #[test]
    fn test_noninvertible_matrix_returns_none() {
        let matrix = Matrix2x2::new(
            1_f32, 2_f32,
            4_f32, 8_f32,
        );

        assert!(matrix.try_inverse().is_none());
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_inverse_is_identity() {
        let matrix = Matrix2x2::new(
            36.84_f64, 427.46_f64,
            7.47_f64,  61.89_f64,
        );
        let matrix_inverse = matrix.try_inverse().unwrap();
        let identity = Matrix2x2::identity();

        assert_relative_eq!(matrix * matrix_inverse, identity, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_inverse_times_matrix_is_identity() {
        let matrix = Matrix2x2::new(
            36.84_f64, 427.46_f64,
            7.47_f64,  61.89_f64,
        );
        let matrix_inverse = matrix.try_inverse().unwrap();
        let identity = Matrix2x2::identity();

        assert_relative_eq!(matrix_inverse * matrix, identity, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);        
    }

    #[rustfmt::skip]
    #[test]
    fn test_constant_times_matrix_inverse_equals_constant_inverse_times_matrix_inverse() {
        let matrix = Matrix2x2::new(
            80_f64,    426.1_f64,
            23.43_f64, 23.5724_f64,
        );
        let constant: f64 = 4_f64;
        let constant_times_matrix_inverse = (constant * matrix).try_inverse().unwrap();
        let constant_inverse_times_matrix_inverse = (1_f64 / constant) * matrix.try_inverse().unwrap();

        assert_eq!(constant_times_matrix_inverse, constant_inverse_times_matrix_inverse);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose_inverse_equals_matrix_inverse_transpose() {
        let matrix = Matrix2x2::new(
            80_f64,    426.1_f64,
            23.43_f64, 23.5724_f64,
        );
        let matrix_transpose_inverse = matrix.transpose().try_inverse().unwrap();
        let matrix_inverse_transpose = matrix.try_inverse().unwrap().transpose();

        assert_eq!(matrix_transpose_inverse, matrix_inverse_transpose);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_inverse_inverse_equals_matrix() {
        let matrix = Matrix2x2::new(
            80_f64,    426.1_f64,
            23.43_f64, 23.5724_f64,
        );
        let result = matrix.try_inverse().unwrap().try_inverse().unwrap();
        let expected = matrix;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_elements_should_be_column_major_order() {
        let matrix = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );

        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c0r1, matrix[0][1]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c1r1, matrix[1][1]);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_swap_columns() {
        let mut result = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );
        result.swap_columns(0, 1);
        let expected = Matrix2x2::new(
            3_i32, 4_i32,
            1_i32, 2_i32,
        );

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_swap_rows() {
        let mut result = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );
        result.swap_rows(0, 1);
        let expected = Matrix2x2::new(
            2_i32, 1_i32,
            4_i32, 3_i32,
        );

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_swap_elements() {
        let mut result = Matrix2x2::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
        );
        result.swap((0, 0), (1, 1));
        let expected = Matrix2x2::new(
            4_i32, 2_i32,
            3_i32, 1_i32,
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_scale() {
        let matrix = Matrix2x2::from_scale(3_i32);
        let unit_x = Vector2::unit_x();
        let unit_y = Vector2::unit_y();
        let expected = unit_x * 3_i32 + unit_y * 3_i32;
        let result = matrix * Vector2::new(1_i32, 1_i32);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_nonuniform_scale() {
        let matrix = Matrix2x2::from_nonuniform_scale(&Vector2::new(3_i32, 7_i32));
        let unit_x = Vector2::unit_x();
        let unit_y = Vector2::unit_y();
        let expected = unit_x * 3_i32 + unit_y * 7_i32;
        let result = matrix * Vector2::new(1_i32, 1_i32);

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix2x2_rotation_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix2x2,
        Unit,
        Vector2,
    };
    use cglinalg_trigonometry::{
        Angle,
        Radians,
    };


    #[test]
    fn test_from_angle() {
        let matrix: Matrix2x2<f64> = Matrix2x2::from_angle(Radians::full_turn_div_4());
        let unit_x = Vector2::unit_x();
        let unit_y = Vector2::unit_y();
        let expected = unit_y;
        let result = matrix * unit_x;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);

        let expected = -unit_x;
        let result = matrix * unit_y;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_rotation_between() {
        let unit_x: Vector2<f64> = Vector2::unit_x();
        let unit_y: Vector2<f64> = Vector2::unit_y();
        let expected = Matrix2x2::new(
             0_f64, 1_f64,
            -1_f64, 0_f64,
        );
        let result = Matrix2x2::rotation_between(&unit_x, &unit_y);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_rotation_between_axis() {
        let unit_x: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
        let unit_y: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_y());
        let expected = Matrix2x2::new(
             0_f64, 1_f64,
            -1_f64, 0_f64,
        );
        let result = Matrix2x2::rotation_between_axis(&unit_x, &unit_y);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }
}


#[cfg(test)]
mod matrix2x2_reflection_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix2x2,
        Unit,
        Vector2,
    };


    /// Construct a reflection matrix test case for reflection about the **x-axis**.
    /// In two dimensions there is an ambiguity in the orientation of the line 
    /// segment; there are two possible normal vectors for the line.
    #[rustfmt::skip]
    #[test]
    fn test_from_reflection_x_axis1() {
        // The y-axis is the normal vector to the plane of the x-axis.
        let normal = Unit::from_value(Vector2::unit_y());
        let expected = Matrix2x2::new(
            1_f64,  0_f64,
            0_f64, -1_f64,
        );
        let result = Matrix2x2::from_reflection(&normal);

        assert_eq!(result, expected);
    }

    /// Construct a reflection matrix test case for reflection about the **x-axis**.
    /// In two dimensions there is an ambiguity in the orientation of the line 
    /// segment; there are two possible normal vectors for the line.
    #[rustfmt::skip]
    #[test]
    fn test_from_reflection_x_axis2() {
        // The y-axis is the normal vector to the plane of the x-axis.
        let normal = Unit::from_value(-Vector2::unit_y());
        let expected = Matrix2x2::new(
            1_f64,  0_f64,
            0_f64, -1_f64,
        );
        let result = Matrix2x2::from_reflection(&normal);

        assert_eq!(result, expected);
    }

    /// Construct a reflection matrix test case for reflection about the **x-axis**.
    /// In two dimensions there is an ambiguity in the orientation of the line 
    /// segment; there are two possible normal vectors for the line.
    #[rustfmt::skip]
    #[test]
    fn test_from_reflection_y_axis1() {
        // The y-axis is the normal vector to the plane of the y-axis.
        let normal = Unit::from_value(Vector2::unit_x());
        let expected = Matrix2x2::new(
            -1_f64, 0_f64,
             0_f64, 1_f64,
        );
        let result = Matrix2x2::from_reflection(&normal);

        assert_eq!(result, expected);
    }

    /// Construct a reflection matrix test case for reflection about the **x-axis**.
    /// In two dimensions there is an ambiguity in the orientation of the line 
    /// segment; there are two possible normal vectors for the line.
    #[rustfmt::skip]
    #[test]
    fn test_from_reflection_y_axis2() {
        // The y-axis is the normal vector to the plane of the y-axis.
        let normal = Unit::from_value(-Vector2::unit_x());
        let expected = Matrix2x2::new(
            -1_f64, 0_f64,
             0_f64, 1_f64,
        );
        let result = Matrix2x2::from_reflection(&normal);

        assert_eq!(result, expected);
    }

    /// Construct a reflection matrix test case for reflection about the 
    /// line `y - x = 0`. In two dimensions there is an ambiguity in the orientation 
    /// of the line segment; there are two possible normal vectors for the line.
    #[rustfmt::skip]
    #[test]
    fn test_from_reflection_from_plane1() {
        let normal = Unit::from_value(
            Vector2::new(f64::sqrt(2_f64)/ 2_f64, -f64::sqrt(2_f64) / 2_f64)
        );
        let expected = Matrix2x2::new(
            0_f64, 1_f64,
            1_f64, 0_f64,
        );
        let result = Matrix2x2::from_reflection(&normal);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Construct a reflection matrix test case for reflection about the 
    /// line `y - x = 0`. In two dimensions there is an ambiguity in the orientation 
    /// of the line segment; there are two possible normal vectors for the line.
    #[rustfmt::skip]
    #[test]
    fn test_from_reflection_from_plane2() {
        let normal = Unit::from_value(
            Vector2::new(-f64::sqrt(2_f64)/ 2_f64, f64::sqrt(2_f64) / 2_f64)
        );
        let expected = Matrix2x2::new(
            0_f64, 1_f64,
            1_f64, 0_f64,
        );
        let result = Matrix2x2::from_reflection(&normal);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }
}


#[cfg(test)]
mod matrix2x2_shear_tests {
    use cglinalg_core::{
        Matrix2x2,
        Unit,
        Vector2,
    };


    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy() {
        let shear_factor = 5_i32;
        let matrix = Matrix2x2::from_shear_xy(shear_factor);
        let vertices = [
            Vector2::new( 1_i32,  1_i32),
            Vector2::new(-1_i32,  1_i32),
            Vector2::new(-1_i32, -1_i32),
            Vector2::new( 1_i32, -1_i32),
        ];
        let expected = [
            Vector2::new( 1_i32 + shear_factor,  1_i32),
            Vector2::new(-1_i32 + shear_factor,  1_i32),
            Vector2::new(-1_i32 - shear_factor, -1_i32),
            Vector2::new( 1_i32 - shear_factor, -1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_shearing_plane() {
        let shear_factor = 5_i32;
        let matrix = Matrix2x2::from_shear_xy(shear_factor);
        let vertices = [
            Vector2::new( 1_i32, 0_i32),
            Vector2::new( 0_i32, 0_i32),
            Vector2::new(-1_i32, 0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx() {
        let shear_factor = 5_i32;
        let matrix = Matrix2x2::from_shear_yx(shear_factor);
        let vertices = [
            Vector2::new( 1_i32,  1_i32),
            Vector2::new(-1_i32,  1_i32),
            Vector2::new(-1_i32, -1_i32),
            Vector2::new( 1_i32, -1_i32),
        ];
        let expected = [
            Vector2::new( 1_i32,  1_i32 + shear_factor),
            Vector2::new(-1_i32,  1_i32 - shear_factor),
            Vector2::new(-1_i32, -1_i32 - shear_factor),
            Vector2::new( 1_i32, -1_i32 + shear_factor),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_shearing_plane() {
        let shear_factor = 5_i32;
        let matrix = Matrix2x2::from_shear_yx(shear_factor);
        let vertices = [
            Vector2::new(0_i32,  1_i32),
            Vector2::new(0_i32,  0_i32),
            Vector2::new(0_i32, -1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_xy() {
        let shear_factor = 7_f64;
        let direction = Unit::from_value(Vector2::unit_x());
        let normal = Unit::from_value(Vector2::unit_y());
        let expected = Matrix2x2::from_shear_xy(shear_factor);
        let result = Matrix2x2::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_yx() {
        let shear_factor = 7_f64;
        let direction = Unit::from_value(Vector2::unit_y());
        let normal = Unit::from_value(Vector2::unit_x());
        let expected = Matrix2x2::from_shear_yx(shear_factor);
        let result = Matrix2x2::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }
}


/// Shearing about the line `y == (1 / 2) * x`.
#[cfg(test)]
mod matrix2x2_shear_noncoordinate_plane_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix2x2,
        Unit,
        Vector2,
    };
    use cglinalg_trigonometry::{
        Angle,
        Radians,
    };


    fn shear_factor() -> f64 {
        5_f64
    }

    fn rotation_angle() -> Radians<f64> {
        Radians(f64::atan2(1_f64, 2_f64))
    }

    fn direction() -> Unit<Vector2<f64>> {
        Unit::from_value(Vector2::new(2_f64, 1_f64))
    }

    fn normal() -> Unit<Vector2<f64>> {
        Unit::from_value(Vector2::new(-1_f64, 2_f64))
    }

    #[rustfmt::skip]
    fn rotation() -> Matrix2x2<f64> {
        Matrix2x2::new(
             2_f64 / f64::sqrt(5_f64),
             1_f64 / f64::sqrt(5_f64),
            -1_f64 / f64::sqrt(5_f64),
             2_f64 / f64::sqrt(5_f64),
        )
    }

    #[rustfmt::skip]
    fn rotation_inv() -> Matrix2x2<f64> {
        Matrix2x2::new(
             2_f64 / f64::sqrt(5_f64),
            -1_f64 / f64::sqrt(5_f64),
             1_f64 / f64::sqrt(5_f64),
             2_f64 / f64::sqrt(5_f64),
        )
    }

    fn shear_matrix_xy() -> Matrix2x2<f64> {
        let shear_factor = shear_factor();

        Matrix2x2::new(1_f64, 0_f64, shear_factor, 1_f64)
    }


    #[test]
    fn test_from_shear_rotation_angle() {
        let rotation_angle = rotation_angle();

        assert_relative_eq!(
            rotation_angle.cos(),
            2_f64 / f64::sqrt(5_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
        assert_relative_eq!(
            rotation_angle.sin(),
            1_f64 / f64::sqrt(5_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
    }

    #[test]
    fn test_from_shear_rotation_matrix() {
        let rotation_angle = rotation_angle();
        let expected = rotation();
        let result = Matrix2x2::from_angle(rotation_angle);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_shear_rotation_matrix_inv() {
        let rotation_angle = rotation_angle();
        let computed_rotation = Matrix2x2::from_angle(rotation_angle);
        let expected = rotation_inv();
        let result = computed_rotation.try_inverse().unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_vertices_xy() {
        let vertices = [
            Vector2::new( 1_f64 / f64::sqrt(5_f64),  3_f64 / f64::sqrt(5_f64)),
            Vector2::new(-3_f64 / f64::sqrt(5_f64),  1_f64 / f64::sqrt(5_f64)),
            Vector2::new(-1_f64 / f64::sqrt(5_f64), -3_f64 / f64::sqrt(5_f64)),
            Vector2::new( 3_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64)),
        ];
        let rotation = rotation();
        let vertices_xy = [
            Vector2::new( 1_f64,  1_f64),
            Vector2::new(-1_f64,  1_f64),
            Vector2::new(-1_f64, -1_f64),
            Vector2::new( 1_f64, -1_f64),
        ];
        let expected = vertices;
        let result = vertices_xy.map(|v| rotation * v);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_shear_direction_xy() {
        let direction = direction();
        let rotation_inv = rotation_inv();
        let expected = Vector2::unit_x();
        let result = rotation_inv * direction.into_inner();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_shear_normal_xy() {
        let normal = normal();
        let rotation_inv = rotation_inv();
        let expected = Vector2::unit_y();
        let result = rotation_inv * normal.into_inner();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_vertices() {
        let shear_factor = shear_factor();
        let direction = direction();
        let normal = normal();
        let matrix = Matrix2x2::from_shear(shear_factor, &direction, &normal);
        let vertices = [
            Vector2::new( 1_f64 / f64::sqrt(5_f64),  3_f64 / f64::sqrt(5_f64)),
            Vector2::new(-3_f64 / f64::sqrt(5_f64),  1_f64 / f64::sqrt(5_f64)),
            Vector2::new(-1_f64 / f64::sqrt(5_f64), -3_f64 / f64::sqrt(5_f64)),
            Vector2::new( 3_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64)),
        ];
        let expected = [
            Vector2::new(
                (2_f64 / f64::sqrt(5_f64)) * (1_f64 + shear_factor) - 1_f64 / f64::sqrt(5_f64),
                (1_f64 / f64::sqrt(5_f64)) * (1_f64 + shear_factor) + 2_f64 / f64::sqrt(5_f64),
            ),
            Vector2::new(
                (2_f64 / f64::sqrt(5_f64)) * (-1_f64 + shear_factor) - 1_f64 / f64::sqrt(5_f64),
                (1_f64 / f64::sqrt(5_f64)) * (-1_f64 + shear_factor) + 2_f64 / f64::sqrt(5_f64),
            ),
            Vector2::new(
                (2_f64 / f64::sqrt(5_f64)) * (-1_f64 - shear_factor) + 1_f64 / f64::sqrt(5_f64),
                (1_f64 / f64::sqrt(5_f64)) * (-1_f64 - shear_factor) - 2_f64 / f64::sqrt(5_f64),
            ),
            Vector2::new(
                (2_f64 / f64::sqrt(5_f64)) * (1_f64 - shear_factor) + 1_f64 / f64::sqrt(5_f64),
                (1_f64 / f64::sqrt(5_f64)) * (1_f64 - shear_factor) - 2_f64 / f64::sqrt(5_f64),
            ),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_matrix() {
        let shear_factor = shear_factor();
        let direction = direction();
        let normal = normal();
        let expected = Matrix2x2::new(
            1_f64 - (2_f64 / 5_f64) * shear_factor, -(1_f64 / 5_f64) * shear_factor,
            (4_f64 / 5_f64) * shear_factor,          1_f64 + (2_f64 / 5_f64) * shear_factor,
        );
        let result = Matrix2x2::from_shear(shear_factor, &direction, &normal);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_shear_matrix_alternative_method() {
        let shear_factor = shear_factor();
        let direction = direction();
        let normal = normal();
        let rotation = rotation();
        let rotation_inv = rotation_inv();
        let shear_matrix_xy = shear_matrix_xy();
        let expected = Matrix2x2::from_shear(shear_factor, &direction, &normal);
        let result = rotation * shear_matrix_xy * rotation_inv;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_shearing_plane() {
        let shear_factor = shear_factor();
        let direction = direction();
        let normal = normal();
        let matrix = Matrix2x2::from_shear(shear_factor, &direction, &normal);
        let vertices = [
            Vector2::new( 1_f64 / f64::sqrt(5_f64),  1_f64 / (2_f64 * f64::sqrt(5_f64))),
            Vector2::new(-3_f64 / f64::sqrt(5_f64), -3_f64 / (2_f64 * f64::sqrt(5_f64))),
            Vector2::new(-1_f64 / f64::sqrt(5_f64), -1_f64 / (2_f64 * f64::sqrt(5_f64))),
            Vector2::new( 3_f64 / f64::sqrt(5_f64),  3_f64 / (2_f64 * f64::sqrt(5_f64))),
            Vector2::new( 0_f64, 0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}


#[cfg(test)]
mod matrix3x3_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix3x3,
        Vector2,
        Vector3,
    };


    #[rustfmt::skip]
    #[test]
    fn test_matrix_components1() {
        let matrix = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );

        assert_eq!(matrix[0][0], 1_i32);
        assert_eq!(matrix[0][1], 2_i32);
        assert_eq!(matrix[0][2], 3_i32);
        assert_eq!(matrix[1][0], 4_i32);
        assert_eq!(matrix[1][1], 5_i32);
        assert_eq!(matrix[1][2], 6_i32);
        assert_eq!(matrix[2][0], 7_i32);
        assert_eq!(matrix[2][1], 8_i32);
        assert_eq!(matrix[2][2], 9_i32);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_components2() {
        let matrix = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
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

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds1() {
        let matrix = Matrix3x3::new(
            1_i32, 2_i32, 3_i32, 
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );

        assert_eq!(matrix[3][0], matrix[3][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds2() {
        let matrix = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );

        assert_eq!(matrix[0][3], matrix[0][3]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds3() {
        let matrix = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );

        assert_eq!(matrix[3][3], matrix[3][3]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds4() {
        let matrix = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );

        assert_eq!(matrix[0][usize::MAX], matrix[0][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds5() {
        let matrix = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );

        assert_eq!(matrix[usize::MAX][0], matrix[usize::MAX][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds6() {
        let matrix = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );

        assert_eq!(matrix[usize::MAX][usize::MAX], matrix[usize::MAX][usize::MAX]);
    }

    #[test]
    fn test_identity_matrix_times_identity_matrix_equals_identity_matrix() {
        let identity_matrix: Matrix3x3<f32> = Matrix3x3::identity();

        assert_eq!(identity_matrix * identity_matrix, identity_matrix);
    }

    #[test]
    fn test_zero_matrix_times_zero_matrix_equals_zero_matrix() {
        let zero_matrix: Matrix3x3<f32> = Matrix3x3::zero();

        assert_eq!(zero_matrix * zero_matrix, zero_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_matrix1() {
        let a_matrix = Matrix3x3::new(
            80_f32,     426.1_f32,   43.393_f32,
            23.43_f32,  23.5724_f32, 1.27_f32,
            81.439_f32, 12.19_f32,   43.36_f32,
        );
        let b_matrix = Matrix3x3::new(
            36.84_f32,     7.04217_f32,  5.74_f32,
            427.46894_f32, 61.89139_f32, 96.27_f32,
            152.66_f32,    86.333_f32,   26.71_f32,
        );
        // let expected = Matrix3x3::new(
        //     3579.6579_f32,  15933.496_f32,   1856.4281_f32,
        //     43487.7660_f32, 184776.9752_f32, 22802.0289_f32,
        //     16410.8178_f32, 67409.1000_f32,  7892.1646_f32,
        // );
        let a_matrix_times_identity = a_matrix * Matrix3x3::identity();
        let b_matrix_times_identity = b_matrix * Matrix3x3::identity();

        assert_eq!(a_matrix_times_identity, a_matrix);
        assert_eq!(b_matrix_times_identity, b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_zero_equals_zero1() {
        let a_matrix = Matrix3x3::new(
            80_f32,     426.1_f32,   43.393_f32,
            23.43_f32,  23.5724_f32, 1.27_f32,
            81.439_f32, 12.19_f32,   43.36_f32
        );
        let b_matrix = Matrix3x3::new(
            36.84_f32,     7.04217_f32,  5.74_f32,
            427.46894_f32, 61.89139_f32, 96.27_f32,
            152.66_f32,    86.333_f32,   26.71_f32,
        );
        // let expected = Matrix3x3::new(
        //     3579.6579_f32,  15933.496_f32,   1856.4281_f32,
        //     43487.7660_f32, 184776.9752_f32, 22802.0289_f32,
        //     16410.8178_f32, 67409.1000_f32,  7892.1646_f32,
        // );
        let a_matrix_times_zero_matrix = a_matrix * Matrix3x3::zero();
        let b_matrix_times_zero_matrix = b_matrix * Matrix3x3::zero();

        assert_eq!(a_matrix_times_zero_matrix, Matrix3x3::zero());
        assert_eq!(b_matrix_times_zero_matrix, Matrix3x3::zero());
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_times_matrix_equals_zero1() {
        let a_matrix = Matrix3x3::new(
            80_f32,     426.1_f32,   43.393_f32,
            23.43_f32,  23.5724_f32, 1.27_f32,
            81.439_f32, 12.19_f32,   43.36_f32,
        );
        let b_matrix = Matrix3x3::new(
            36.84_f32,     7.04217_f32,  5.74_f32,
            427.46894_f32, 61.89139_f32, 96.27_f32,
            152.66_f32,    86.333_f32,   26.71_f32,
        );
        // let expected = Matrix3x3::new(
        //     3579.6579_f32,  15933.496_f32,   1856.4281_f32,
        //     43487.7660_f32, 184776.9752_f32, 22802.0289_f32,
        //     16410.8178_f32, 67409.1000_f32,  7892.1646_f32,
        // );
        let zero_times_a_matrix = Matrix3x3::zero() * a_matrix;
        let zero_times_b_matrix = Matrix3x3::zero() * b_matrix;

        assert_eq!(zero_times_a_matrix, Matrix3x3::zero());
        assert_eq!(zero_times_b_matrix, Matrix3x3::zero());
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_identity_times_matrix1() {
        let a_matrix = Matrix3x3::new(
            80_f32,     426.1_f32,   43.393_f32,
            23.43_f32,  23.5724_f32, 1.27_f32,
            81.439_f32, 12.19_f32,   43.36_f32,
        );
        let b_matrix = Matrix3x3::new(
            36.84_f32,     7.04217_f32,  5.74_f32,
            427.46894_f32, 61.89139_f32, 96.27_f32,
            152.66_f32,    86.333_f32,   26.71_f32,
        );
        // let expected = Matrix3x3::new(
        //     3579.6579_f32,  15933.496_f32,   1856.4281_f32,
        //     43487.7660_f32, 184776.9752_f32, 22802.0289_f32,
        //     16410.8178_f32, 67409.1000_f32,  7892.1646_f32,
        // );
        let a_matrix_times_identity = a_matrix * Matrix3x3::identity();
        let identity_times_a_matrix = Matrix3x3::identity() * a_matrix;
        let b_matrix_times_identity = b_matrix * Matrix3x3::identity();
        let identity_times_b_matrix = Matrix3x3::identity() * b_matrix;

        assert_eq!(a_matrix_times_identity, identity_times_a_matrix);
        assert_eq!(b_matrix_times_identity, identity_times_b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose_transpose_equals_matrix1() {
        let a_matrix = Matrix3x3::new(
            80_f32,     426.1_f32,   43.393_f32,
            23.43_f32,  23.5724_f32, 1.27_f32,
            81.439_f32, 12.19_f32,   43.36_f32,
        );
        let b_matrix = Matrix3x3::new(
            36.84_f32,     7.04217_f32,  5.74_f32,
            427.46894_f32, 61.89139_f32, 96.27_f32,
            152.66_f32,    86.333_f32,   26.71_f32,
        );
        // let expected = Matrix3x3::new(
        //     3579.6579_f32,  15933.496_f32,   1856.4281_f32,
        //     43487.7660_f32, 184776.9752_f32, 22802.0289_f32,
        //     16410.8178_f32, 67409.1000_f32,  7892.1646_f32,
        // );
        let a_matrix_transpose_transpose = a_matrix.transpose().transpose();
        let b_matrix_transpose_transpose = b_matrix.transpose().transpose();

        assert_eq!(a_matrix_transpose_transpose, a_matrix);
        assert_eq!(b_matrix_transpose_transpose, b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication1() {
        let a_matrix = Matrix3x3::new(
            80_f32,     426.1_f32,   43.393_f32,
            23.43_f32,  23.5724_f32, 1.27_f32,
            81.439_f32, 12.19_f32,   43.36_f32,
        );
        let b_matrix = Matrix3x3::new(
            36.84_f32,     7.04217_f32,  5.74_f32,
            427.46894_f32, 61.89139_f32, 96.27_f32,
            152.66_f32,    86.333_f32,   26.71_f32,
        );
        let expected = Matrix3x3::new(
            3579.6579_f32,  15933.496_f32,   1856.4281_f32,
            43487.7660_f32, 184776.9752_f32, 22802.0289_f32,
            16410.8178_f32, 67409.1000_f32,  7892.1646_f32,
        );
        let result = a_matrix * b_matrix;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f32::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_matrix2() {
        let a_matrix = Matrix3x3::new(
            68.32_f32, 0_f32,      0_f32,
            0_f32,     37.397_f32, 0_f32,
            0_f32,     0_f32,      43.393_f32,
        );
        let b_matrix = Matrix3x3::new(
            57.72_f32, 0_f32,         0_f32,
            0_f32,     9.5433127_f32, 0_f32,
            0_f32,     0_f32,         12.19_f32,
        );
        // let expected = Matrix3x3::new(
        //     3943.4304_f32, 0_f32,         0_f32,
        //     0_f32,         356.89127_f32, 0_f32,
        //     0_f32,         0_f32,         528.96067_f32,
        // );
        let a_matrix_times_identity = a_matrix * Matrix3x3::identity();
        let b_matrix_times_identity = b_matrix * Matrix3x3::identity();

        assert_eq!(a_matrix_times_identity, a_matrix);
        assert_eq!(b_matrix_times_identity, b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_zero_equals_zero2() {
        let a_matrix = Matrix3x3::new(
            68.32_f32, 0_f32,      0_f32,
            0_f32,     37.397_f32, 0_f32,
            0_f32,     0_f32,      43.393_f32,
        );
        let b_matrix = Matrix3x3::new(
            57.72_f32, 0_f32,         0_f32,
            0_f32,     9.5433127_f32, 0_f32,
            0_f32,     0_f32,         12.19_f32,
        );
        // let expected = Matrix3x3::new(
        //     3943.4304_f32, 0_f32,         0_f32,
        //     0_f32,         356.89127_f32, 0_f32,
        //     0_f32,         0_f32,         528.96067_f32,
        // );
        let a_matrix_times_zero_matrix = a_matrix * Matrix3x3::zero();
        let b_matrix_times_zero_matrix = b_matrix * Matrix3x3::zero();

        assert_eq!(a_matrix_times_zero_matrix, Matrix3x3::zero());
        assert_eq!(b_matrix_times_zero_matrix, Matrix3x3::zero());
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_times_matrix_equals_zero2() {
        let a_matrix = Matrix3x3::new(
            68.32_f32, 0_f32,      0_f32,
            0_f32,     37.397_f32, 0_f32,
            0_f32,     0_f32,      43.393_f32,
        );
        let b_matrix = Matrix3x3::new(
            57.72_f32, 0_f32,         0_f32,
            0_f32,     9.5433127_f32, 0_f32,
            0_f32,     0_f32,         12.19_f32,
        );
        // let expected = Matrix3x3::new(
        //     3943.4304_f32, 0_f32,         0_f32,
        //     0_f32,         356.89127_f32, 0_f32,
        //     0_f32,         0_f32,         528.96067_f32,
        // );
        let zero_times_a_matrix = Matrix3x3::zero() * a_matrix;
        let zero_times_b_matrix = Matrix3x3::zero() * b_matrix;

        assert_eq!(zero_times_a_matrix, Matrix3x3::zero());
        assert_eq!(zero_times_b_matrix, Matrix3x3::zero());
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_identity_times_matrix2() {
        let a_matrix = Matrix3x3::new(
            68.32_f32, 0_f32,      0_f32,
            0_f32,     37.397_f32, 0_f32,
            0_f32,     0_f32,      43.393_f32,
        );
        let b_matrix = Matrix3x3::new(
            57.72_f32, 0_f32,         0_f32,
            0_f32,     9.5433127_f32, 0_f32,
            0_f32,     0_f32,         12.19_f32,
        );
        // let expected = Matrix3x3::new(
        //     3943.4304_f32, 0_f32,         0_f32,
        //     0_f32,         356.89127_f32, 0_f32,
        //     0_f32,         0_f32,         528.96067_f32,
        // );
        let a_matrix_times_identity = a_matrix * Matrix3x3::identity();
        let identity_times_a_matrix = Matrix3x3::identity() * a_matrix;
        let b_matrix_times_identity = b_matrix * Matrix3x3::identity();
        let identity_times_b_matrix = Matrix3x3::identity() * b_matrix;

        assert_eq!(a_matrix_times_identity, identity_times_a_matrix);
        assert_eq!(b_matrix_times_identity, identity_times_b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose_transpose_equals_matrix2() {
        let a_matrix = Matrix3x3::new(
            68.32_f32, 0_f32,      0_f32,
            0_f32,     37.397_f32, 0_f32,
            0_f32,     0_f32,      43.393_f32,
        );
        let b_matrix = Matrix3x3::new(
            57.72_f32, 0_f32,         0_f32,
            0_f32,     9.5433127_f32, 0_f32,
            0_f32,     0_f32,         12.19_f32,
        );
        // let expected = Matrix3x3::new(
        //     3943.4304_f32, 0_f32,         0_f32,
        //     0_f32,         356.89127_f32, 0_f32,
        //     0_f32,         0_f32,         528.96067_f32,
        // );
        let a_matrix_transpose_transpose = a_matrix.transpose().transpose();
        let b_matrix_transpose_transpose = b_matrix.transpose().transpose();

        assert_eq!(a_matrix_transpose_transpose, a_matrix);
        assert_eq!(b_matrix_transpose_transpose, b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication2() {
        let a_matrix = Matrix3x3::new(
            68.32_f32, 0_f32,      0_f32,
            0_f32,     37.397_f32, 0_f32,
            0_f32,     0_f32,      43.393_f32,
        );
        let b_matrix = Matrix3x3::new(
            57.72_f32, 0_f32,         0_f32,
            0_f32,     9.5433127_f32, 0_f32,
            0_f32,     0_f32,         12.19_f32,
        );
        let expected = Matrix3x3::new(
            3943.4304_f32, 0_f32,         0_f32,
            0_f32,         356.89127_f32, 0_f32,
            0_f32,         0_f32,         528.96067_f32,
        );
        let result = a_matrix * b_matrix;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f32::EPSILON);
    }

    #[test]
    fn test_identity_transpose_equals_identity() {
        let identity = Matrix3x3::<f32>::identity();
        let identity_transpose = identity.transpose();

        assert_eq!(identity, identity_transpose);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector3::new(1_i32, 2_i32, 3_i32);
        let c1 = Vector3::new(4_i32, 5_i32, 6_i32);
        let c2 = Vector3::new(7_i32, 8_i32, 9_i32);
        let columns = [c0, c1, c2];
        let expected = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );
        let result = Matrix3x3::from_columns(&columns);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_rows() {
        let r0 = Vector3::new(1_i32, 2_i32, 3_i32);
        let r1 = Vector3::new(4_i32, 5_i32, 6_i32);
        let r2 = Vector3::new(7_i32, 8_i32, 9_i32);
        let rows = [r0, r1, r2];
        let expected = Matrix3x3::new(
            1_i32, 4_i32, 7_i32,
            2_i32, 5_i32, 8_i32,
            3_i32, 6_i32, 9_i32
        );
        let result = Matrix3x3::from_rows(&rows);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_constant_times_identity_is_constant_along_diagonal() {
        let c = 802.3435169_f64;
        let identity = Matrix3x3::identity();
        let expected = Matrix3x3::new(
            c,     0_f64, 0_f64,
            0_f64, c,     0_f64,
            0_f64, 0_f64, c,
        );

        assert_eq!(identity * c, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_identity_divide_constant_is_constant_inverse_along_diagonal() {
        let c = 802.3435169;
        let identity = Matrix3x3::identity();
        let expected = Matrix3x3::new(
            1_f64 / c, 0_f64,     0_f64,
            0_f64,     1_f64 / c, 0_f64,
            0_f64,     0_f64,     1_f64 / c,
        );

        assert_eq!(identity / c, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero_matrix = Matrix3x3::zero();
        let matrix = Matrix3x3::new(
            80_f64,     426.1_f64,   43.393_f64,
            23.43_f64,  23.5724_f64, 1.27_f64,
            81.439_f64, 12.19_f64,   43.36_f64,
        );

        assert_eq!(matrix + zero_matrix, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero_matrix = Matrix3x3::zero();
        let matrix = Matrix3x3::new(
            80_f64,     426.1_f64,   43.393_f64,
            23.43_f64,  23.5724_f64, 1.27_f64,
            81.439_f64, 12.19_f64,   43.36_f64,
        );

        assert_eq!(zero_matrix + matrix, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_with_zero_determinant() {
        let matrix = Matrix3x3::new(
            1_f32, 2_f32, 3_f32,
            4_f32, 5_f32, 6_f32,
            4_f32, 5_f32, 6_f32,
        );

        assert_eq!(matrix.determinant(), 0_f32);
    }

    #[rustfmt::skip]
    #[test]
    fn test_lower_triangular_matrix_determinant() {
        let matrix = Matrix3x3::new(
            1_f64,  0_f64,  0_f64,
            5_f64,  2_f64,  0_f64,
            5_f64,  5_f64,  3_f64,
        );

        assert_eq!(matrix.determinant(), 1_f64 * 2_f64 * 3_f64);
    }

    #[rustfmt::skip]
    #[test]
    fn test_upper_triangular_matrix_determinant() {
        let matrix = Matrix3x3::new(
            1_f64,  5_f64,  5_f64,
            0_f64,  2_f64,  5_f64,
            0_f64,  0_f64,  3_f64,
        );

        assert_eq!(matrix.determinant(), 1_f64 * 2_f64 * 3_f64);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_inverse() {
        let matrix = Matrix3x3::new(
            5_f64, 1_f64, 1_f64,
            1_f64, 5_f64, 1_f64,
            1_f64, 1_f64, 5_f64,
        );
        let expected = (1_f64 / 28_f64) * Matrix3x3::new(
             6_f64, -1_f64, -1_f64,
            -1_f64,  6_f64, -1_f64,
            -1_f64, -1_f64,  6_f64,
        );
        let result = matrix.try_inverse().unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_identity_is_invertible() {
        assert!(Matrix3x3::<f64>::identity().is_invertible());
    }

    #[test]
    fn test_identity_inverse_is_identity() {
        let result: Matrix3x3<f64> = Matrix3x3::identity().try_inverse().unwrap();
        let expected: Matrix3x3<f64> = Matrix3x3::identity();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_diagonal_matrix() {
        let matrix = 4_f64 * Matrix3x3::identity();
        let expected = (1_f64 / 4_f64) * Matrix3x3::identity();
        let result = matrix.try_inverse().unwrap();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_with_nonzero_determinant_is_invertible() {
        let matrix = Matrix3x3::new(
            1_f32, 2_f32, 3_f32,
            0_f32, 4_f32, 5_f32,
            0_f32, 0_f32, 6_f32,
        );

        assert!(matrix.is_invertible());
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_with_zero_determinant_is_not_invertible() {
        let matrix = Matrix3x3::new(
            1_f32, 2_f32, 3_f32,
            4_f32, 5_f32, 6_f32,
            4_f32, 5_f32, 6_f32,
        );

        assert!(!matrix.is_invertible());
    }

    #[rustfmt::skip]
    #[test]
    fn test_noninvertible_matrix_returns_none() {
        let matrix = Matrix3x3::new(
            1_f32, 2_f32, 3_f32,
            4_f32, 5_f32, 6_f32,
            4_f32, 5_f32, 6_f32,
        );

        assert!(matrix.try_inverse().is_none());
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_inverse_is_identity() {
        let matrix = Matrix3x3::new(
            80_f64,     426.1_f64,   43.393_f64,
            23.43_f64,  23.5724_f64, 1.27_f64,
            81.439_f64, 12.19_f64,   43.36_f64,
        );
        let matrix_inverse = matrix.try_inverse().unwrap();
        let identity = Matrix3x3::identity();

        assert_relative_eq!(matrix * matrix_inverse, identity, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_constant_times_matrix_inverse_equals_constant_inverse_times_matrix_inverse() {
        let matrix = Matrix3x3::new(
            80_f64,     426.1_f64,   43.393_f64,
            23.43_f64,  23.5724_f64, 1.27_f64,
            81.439_f64, 12.19_f64,   43.36_f64,
        );
        let constant = 4_f64;
        let constant_times_matrix_inverse = (constant * matrix).try_inverse().unwrap();
        let constant_inverse_times_matrix_inverse = (1_f64 / constant) * matrix.try_inverse().unwrap();

        assert_eq!(constant_times_matrix_inverse, constant_inverse_times_matrix_inverse);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose_inverse_equals_matrix_inverse_transpose() {
        let matrix = Matrix3x3::new(
            80_f64,     426.1_f64,   43.393_f64,
            23.43_f64,  23.5724_f64, 1.27_f64,
            81.439_f64, 12.19_f64,   43.36_f64,
        );
        let matrix_transpose_inverse = matrix.transpose().try_inverse().unwrap();
        let matrix_inverse_transpose = matrix.try_inverse().unwrap().transpose();

        assert_eq!(matrix_transpose_inverse, matrix_inverse_transpose);
    }

    #[rustfmt::skip]
    #[test]
    fn test_inverse_times_matrix_is_identity() {
        let matrix = Matrix3x3::new(
            80_f64,     426.1_f64,   43.393_f64,
            23.43_f64,  23.5724_f64, 1.27_f64,
            81.439_f64, 12.19_f64,   43.36_f64,
        );
        let matrix_inverse = matrix.try_inverse().unwrap();
        let identity = Matrix3x3::identity();

        assert_relative_eq!(matrix_inverse * matrix, identity, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_inverse_inverse_equals_matrix() {
        let matrix = Matrix3x3::new(
            80_f64,     426.1_f64,   43.393_f64,
            23.43_f64,  23.5724_f64, 1.27_f64,
            81.439_f64, 12.19_f64,   43.36_f64,
        );
        let result = matrix.try_inverse().unwrap().try_inverse().unwrap();
        let expected = matrix;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_elements_should_be_column_major_order() {
        let matrix = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
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

    #[rustfmt::skip]
    #[test]
    fn test_matrix_swap_columns() {
        let mut result = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );
        result.swap_columns(0, 1);
        let expected = Matrix3x3::new(
            4_i32, 5_i32, 6_i32,
            1_i32, 2_i32, 3_i32,
            7_i32, 8_i32, 9_i32,
        );

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_swap_rows() {
        let mut result = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );
        result.swap_rows(0, 1);
        let expected = Matrix3x3::new(
            2_i32, 1_i32, 3_i32,
            5_i32, 4_i32, 6_i32,
            8_i32, 7_i32, 9_i32,
        );

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_swap_elements() {
        let mut result = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );
        result.swap((0, 0), (2, 1));
        let expected = Matrix3x3::new(
            8_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 1_i32, 9_i32,
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_scale() {
        let matrix = Matrix3x3::from_scale(3_i32);
        let unit_x = Vector3::unit_x();
        let unit_y = Vector3::unit_y();
        let unit_z = Vector3::unit_z();
        let expected = unit_x * 3 + unit_y * 3_i32 + unit_z * 3_i32;
        let result = matrix * Vector3::new(1_i32, 1_i32, 1_i32);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_nonuniform_scale() {
        let matrix = Matrix3x3::from_nonuniform_scale(&Vector3::new(3_i32, 5_i32, 7_i32));
        let unit_x = Vector3::unit_x();
        let unit_y = Vector3::unit_y();
        let unit_z = Vector3::unit_z();
        let expected = unit_x * 3_i32 + unit_y * 5_i32 + unit_z * 7_i32;
        let result = matrix * Vector3::new(1_i32, 1_i32, 1_i32);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_scale_does_not_change_homogeneous_coordinate() {
        let matrix = Matrix3x3::from_affine_scale(5_i32);
        let unit_z = Vector3::unit_z();
        let expected = unit_z;
        let result = matrix * unit_z;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_nonuniform_scale() {
        let matrix = Matrix3x3::from_affine_nonuniform_scale(&Vector2::new(7_i32, 11_i32));
        let expected = Vector3::new(7_i32, 11_i32, 1_i32);
        let result = matrix * Vector3::new(1_i32, 1_i32, 1_i32);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_nonuniform_scale_does_not_change_homogeneous_coordinate() {
        let matrix = Matrix3x3::from_affine_nonuniform_scale(&Vector2::new(7_i32, 11_i32));
        let unit_z = Vector3::unit_z();
        let expected = unit_z;
        let result = matrix * unit_z;

        assert_eq!(result, expected);
    }

    /// An affine translation should only displace points and not vectors. We
    /// distinguish points by using a `1` in the last coordinate, and vectors
    /// by using a `0` in the last coordinate.
    #[test]
    fn test_from_affine_translation_point() {
        let distance = Vector2::new(3_i32, 7_i32);
        let matrix = Matrix3x3::from_affine_translation(&distance);
        let point = Vector3::new(0_i32, 0_i32, 1_i32);
        let expected = Vector3::new(3_i32, 7_i32, 1_i32);
        let result = matrix * point;

        assert_eq!(result, expected);
    }

    /// An affine translation should only displace points and not vectors. We
    /// distinguish points by using a `1` in the last coordinate, and vectors
    /// by using a `0` in the last coordinate.
    #[test]
    fn test_from_affine_translation_vector() {
        let distance = Vector2::new(3_i32, 7_i32);
        let matrix = Matrix3x3::from_affine_translation(&distance);
        let vector = Vector3::zero();
        let expected = vector;
        let result = matrix * vector;

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix3x3_rotation_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix3x3,
        Normed,
        Point3,
        Unit,
        Vector2,
        Vector3,
    };
    use cglinalg_trigonometry::{
        Angle,
        Radians,
    };


    #[test]
    fn test_from_angle_x() {
        let angle: Radians<f64> = Radians::full_turn_div_4();
        let unit_y = Vector3::unit_y();
        let unit_z = Vector3::unit_z();
        let matrix = Matrix3x3::from_angle_x(angle);
        let expected = unit_z;
        let result = matrix * unit_y;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_angle_y() {
        let angle: Radians<f64> = Radians::full_turn_div_4();
        let unit_z = Vector3::unit_z();
        let unit_x = Vector3::unit_x();
        let matrix = Matrix3x3::from_angle_y(angle);
        let expected = unit_x;
        let result = matrix * unit_z;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_angle_z() {
        let angle: Radians<f64> = Radians::full_turn_div_4();
        let unit_x = Vector3::unit_x();
        let unit_y = Vector3::unit_y();
        let matrix = Matrix3x3::from_angle_z(angle);
        let expected = unit_y;
        let result = matrix * unit_x;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_axis_angle() {
        let angle: Radians<f64> = Radians::full_turn_div_2();
        let axis = Unit::from_value((1_f64 / f64::sqrt(2_f64)) * Vector3::new(1_f64, 1_f64, 0_f64));
        let vector = Vector3::new(1_f64, 1_f64, -1_f64);
        let matrix = Matrix3x3::from_axis_angle(&axis, angle);
        let expected = Vector3::new(1_f64, 1_f64, 1_f64);
        let result = matrix * vector;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_angle() {
        let matrix: Matrix3x3<f64> = Matrix3x3::from_affine_angle(Radians::full_turn_div_4());
        let unit_x = Vector2::unit_x();
        let unit_y = Vector2::unit_y();
        let expected = unit_y.extend(0_f64);
        let result = matrix * unit_x.extend(0_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);

        let expected = -unit_x.extend(0_f64);
        let result = matrix * unit_y.extend(0_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_look_to_lh() {
        let direction = Vector3::new(1_f64, 1_f64, 1_f64);
        let up = Vector3::unit_y();
        let unit_z = Vector3::unit_z();
        let look_to = Matrix3x3::look_to_lh(&direction, &up);
        let expected = unit_z;
        let result = look_to * direction.normalize();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_look_to_rh() {
        let direction = Vector3::new(1_f64, 1_f64, 1_f64).normalize();
        let up = Vector3::unit_y();
        let minus_unit_z = -Vector3::unit_z();
        let look_to = Matrix3x3::look_to_rh(&direction, &up);
        let expected = minus_unit_z;
        let result = look_to * direction;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_look_at_lh() {
        let eye = Point3::new(-1_f64, -1_f64, -1_f64);
        let target = Point3::origin();
        let direction = target - eye;
        let up = Vector3::unit_y();
        let unit_z = Vector3::unit_z();
        let look_at = Matrix3x3::look_at_lh(&eye, &target, &up);
        let expected = unit_z;
        let result = look_at * direction.normalize();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_look_at_rh() {
        let eye = Point3::new(-1_f64, -1_f64, -1_f64);
        let target = Point3::origin();
        let direction = target - eye;
        let up = Vector3::unit_y();
        let minus_unit_z = -Vector3::unit_z();
        let look_at = Matrix3x3::look_at_rh(&eye, &target, &up);
        let expected = minus_unit_z;
        let result = look_at * direction.normalize();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_rotation_between() {
        let unit_x: Vector3<f64> = Vector3::unit_x();
        let unit_y: Vector3<f64> = Vector3::unit_y();
        let expected = Matrix3x3::new(
             0_f64, 1_f64, 0_f64,
            -1_f64, 0_f64, 0_f64,
             0_f64, 0_f64, 1_f64,
        );
        let result = Matrix3x3::rotation_between(&unit_x, &unit_y).unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }
}


#[cfg(test)]
mod matrix3x3_reflection_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix3x3,
        Point2,
        Unit,
        Vector2,
        Vector3,
    };


    /// Construct a reflection matrix test case for reflection about the **x-axis**.
    /// In two dimensions there is an ambiguity in the orientation of the line 
    /// segment; there are two possible normal vectors for the line.
    #[rustfmt::skip]
    #[test]
    fn test_from_affine_reflection_x_axis1() {
        // The y-axis is the normal vector to the plane of the x-axis.
        let bias = Point2::origin();
        let normal = Unit::from_value(Vector2::unit_y());
        let expected = Matrix3x3::new(
            1_f64,  0_f64, 0_f64,
            0_f64, -1_f64, 0_f64,
            0_f64,  0_f64, 1_f64,
        );
        let result = Matrix3x3::from_affine_reflection(&normal, &bias);

        assert_eq!(result, expected);
    }

    /// Construct a reflection matrix test case for reflection about the **x-axis**.
    /// In two dimensions there is an ambiguity in the orientation of the line 
    /// segment; there are two possible normal vectors for the line.
    #[rustfmt::skip]
    #[test]
    fn test_from_affine_reflection_x_axis2() {
        // The y-axis is the normal vector to the plane of the x-axis.
        let bias = Point2::origin();
        let normal = Unit::from_value(-Vector2::unit_y());
        let expected = Matrix3x3::new(
            1_f64,  0_f64, 0_f64,
            0_f64, -1_f64, 0_f64,
            0_f64,  0_f64, 1_f64,
        );
        let result = Matrix3x3::from_affine_reflection(&normal, &bias);

        assert_eq!(result, expected);
    }

    /// Construct a reflection matrix test case for reflection about the **x-axis**.
    /// In two dimensions there is an ambiguity in the orientation of the line 
    /// segment; there are two possible normal vectors for the line.
    #[rustfmt::skip]
    #[test]
    fn test_from_affine_reflection_y_axis1() {
        // The y-axis is the normal vector to the plane of the y-axis.
        let bias = Point2::origin();
        let normal = Unit::from_value(Vector2::unit_x());
        let expected = Matrix3x3::new(
            -1_f64, 0_f64, 0_f64,
             0_f64, 1_f64, 0_f64,
             0_f64, 0_f64, 1_f64,
        );
        let result = Matrix3x3::from_affine_reflection(&normal, &bias);

        assert_eq!(result, expected);
    }

    /// Construct a reflection matrix test case for reflection about the **x-axis**.
    /// In two dimensions there is an ambiguity in the orientation of the line 
    /// segment; there are two possible normal vectors for the line.
    #[rustfmt::skip]
    #[test]
    fn test_from_affine_reflection_y_axis2() {
        // The y-axis is the normal vector to the plane of the y-axis.
        let bias = Point2::origin();
        let normal = Unit::from_value(-Vector2::unit_x());
        let expected = Matrix3x3::new(
            -1_f64, 0_f64, 0_f64,
             0_f64, 1_f64, 0_f64,
             0_f64, 0_f64, 1_f64,
        );
        let result = Matrix3x3::from_affine_reflection(&normal, &bias);

        assert_eq!(result, expected);
    }

    /// Construct a reflection matrix test case for reflection about the 
    /// line `y - x = 0`. In two dimensions there is an ambiguity in the 
    /// orientation of the line segment; there are two possible normal vectors 
    /// for the line.
    #[rustfmt::skip]
    #[test]
    fn test_from_affine_reflection_from_plane1() {
        let bias = Point2::origin();
        let normal = Unit::from_value(
            Vector2::new(f64::sqrt(2_f64)/ 2_f64, -f64::sqrt(2_f64) / 2_f64)
        );
        let expected = Matrix3x3::new(
            0_f64, 1_f64, 0_f64,
            1_f64, 0_f64, 0_f64,
            0_f64, 0_f64, 1_f64,
        );
        let result = Matrix3x3::from_affine_reflection(&normal, &bias);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Construct a reflection matrix test case for reflection about the 
    /// line `y - x = 0`. In two dimensions there is an ambiguity in the 
    /// orientation of the line segment; there are two possible normal vectors 
    /// for the line.
    #[rustfmt::skip]
    #[test]
    fn test_from_affine_reflection_from_plane2() {
        let bias = Point2::origin();
        let normal = Unit::from_value(
            Vector2::new(-f64::sqrt(2_f64)/ 2_f64, f64::sqrt(2_f64) / 2_f64)
        );
        let expected = Matrix3x3::new(
            0_f64, 1_f64, 0_f64,
            1_f64, 0_f64, 0_f64,
            0_f64, 0_f64, 1_f64,
        );
        let result = Matrix3x3::from_affine_reflection(&normal, &bias);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Construct an affine reflection matrix about the line `y = (1/2)x + 2`.
    /// This line does not cross the origin.
    #[test]
    fn test_from_affine_reflection_from_line_that_does_not_cross_origin1() {
        // We can always choose the y-intercept as the known point.
        let bias = Point2::new(0_f64, 2_f64);
        let normal = Unit::from_value(Vector2::new(-1_f64 / f64::sqrt(5_f64), 2_f64 / f64::sqrt(5_f64)));
        let matrix = Matrix3x3::from_affine_reflection(&normal, &bias);
        let vector = Vector3::new(1_f64, 0_f64, 1_f64);
        let expected = Vector3::new(-1_f64, 4_f64, 1_f64);
        let result = matrix * vector;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Construct an affine reflection matrix about the line `y = (1/2)x + 2`.
    /// This line does not cross the origin.
    #[test]
    fn test_from_affine_reflection_from_line_that_does_not_cross_origin2() {
        // We can always choose the y-intercept as the known point.
        let bias = Point2::new(0_f64, 2_f64);
        let normal = Unit::from_value(Vector2::new(1_f64 / f64::sqrt(5_f64), -2_f64 / f64::sqrt(5_f64)));
        let matrix = Matrix3x3::from_affine_reflection(&normal, &bias);
        let vector = Vector3::new(1_f64, 0_f64, 1_f64);
        let expected = Vector3::new(-1_f64, 4_f64, 1_f64);
        let result = matrix * vector;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_reflection_xy_plane() {
        let normal = Unit::from_value(Vector3::unit_z());
        let expected = Matrix3x3::new(
            1_f64, 0_f64,  0_f64,
            0_f64, 1_f64,  0_f64,
            0_f64, 0_f64, -1_f64,
        );
        let result = Matrix3x3::from_reflection(&normal);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_reflection_zx_plane() {
        let normal = Unit::from_value(-Vector3::unit_y());
        let expected = Matrix3x3::new(
            1_f64,  0_f64, 0_f64,
            0_f64, -1_f64, 0_f64,
            0_f64,  0_f64, 1_f64,
        );
        let result = Matrix3x3::from_reflection(&normal);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_reflection_yz_plane() {
        let normal = Unit::from_value(Vector3::unit_x());
        let expected = Matrix3x3::new(
            -1_f64,  0_f64, 0_f64,
             0_f64, 1_f64,  0_f64,
             0_f64,  0_f64, 1_f64,
        );
        let result = Matrix3x3::from_reflection(&normal);

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix3x3_shear_tests {
    use cglinalg_core::{
        Matrix3x3,
        Unit,
        Vector3,
    };


    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_shear_xy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32 + shear_factor,  1_i32,  1_i32),
            Vector3::new(-1_i32 + shear_factor,  1_i32,  1_i32),
            Vector3::new(-1_i32 - shear_factor, -1_i32,  1_i32),
            Vector3::new( 1_i32 - shear_factor, -1_i32,  1_i32),
            Vector3::new( 1_i32 + shear_factor,  1_i32, -1_i32),
            Vector3::new(-1_i32 + shear_factor,  1_i32, -1_i32),
            Vector3::new(-1_i32 - shear_factor, -1_i32, -1_i32),
            Vector3::new( 1_i32 - shear_factor, -1_i32, -1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_shearing_plane() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_shear_xy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32, 0_i32,  1_i32),
            Vector3::new(-1_i32, 0_i32,  1_i32),
            Vector3::new(-1_i32, 0_i32, -1_i32),
            Vector3::new( 1_i32, 0_i32, -1_i32),
            Vector3::new( 0_i32, 0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xz() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_shear_xz(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32 + shear_factor,  1_i32,  1_i32),
            Vector3::new(-1_i32 + shear_factor,  1_i32,  1_i32),
            Vector3::new(-1_i32 + shear_factor, -1_i32,  1_i32),
            Vector3::new( 1_i32 + shear_factor, -1_i32,  1_i32),
            Vector3::new( 1_i32 - shear_factor,  1_i32, -1_i32),
            Vector3::new(-1_i32 - shear_factor,  1_i32, -1_i32),
            Vector3::new(-1_i32 - shear_factor, -1_i32, -1_i32),
            Vector3::new( 1_i32 - shear_factor, -1_i32, -1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xz_shearing_plane() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_shear_xz(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32, 0_i32),
            Vector3::new(-1_i32,  1_i32, 0_i32),
            Vector3::new(-1_i32, -1_i32, 0_i32),
            Vector3::new( 1_i32, -1_i32, 0_i32),
            Vector3::new( 0_i32,  0_i32, 0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_shear_yx(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32 + shear_factor,  1_i32),
            Vector3::new(-1_i32,  1_i32 - shear_factor,  1_i32),
            Vector3::new(-1_i32, -1_i32 - shear_factor,  1_i32),
            Vector3::new( 1_i32, -1_i32 + shear_factor,  1_i32),
            Vector3::new( 1_i32,  1_i32 + shear_factor, -1_i32),
            Vector3::new(-1_i32,  1_i32 - shear_factor, -1_i32),
            Vector3::new(-1_i32, -1_i32 - shear_factor, -1_i32),
            Vector3::new( 1_i32, -1_i32 + shear_factor, -1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_shearing_plane() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_shear_yx(shear_factor);
        let vertices = [
            Vector3::new(0_i32,  1_i32,  1_i32),
            Vector3::new(0_i32, -1_i32,  1_i32),
            Vector3::new(0_i32,  1_i32, -1_i32),
            Vector3::new(0_i32, -1_i32, -1_i32),
            Vector3::new(0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yz() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_shear_yz(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32 + shear_factor,  1_i32),
            Vector3::new(-1_i32,  1_i32 + shear_factor,  1_i32),
            Vector3::new(-1_i32, -1_i32 + shear_factor,  1_i32),
            Vector3::new( 1_i32, -1_i32 + shear_factor,  1_i32),
            Vector3::new( 1_i32,  1_i32 - shear_factor, -1_i32),
            Vector3::new(-1_i32,  1_i32 - shear_factor, -1_i32),
            Vector3::new(-1_i32, -1_i32 - shear_factor, -1_i32),
            Vector3::new( 1_i32, -1_i32 - shear_factor, -1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yz_shearing_plane() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_shear_yz(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32, 0_i32),
            Vector3::new(-1_i32,  1_i32, 0_i32),
            Vector3::new(-1_i32, -1_i32, 0_i32),
            Vector3::new( 1_i32, -1_i32, 0_i32),
            Vector3::new( 0_i32,  0_i32, 0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zx() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_shear_zx(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32,  1_i32 + shear_factor),
            Vector3::new(-1_i32,  1_i32,  1_i32 - shear_factor),
            Vector3::new(-1_i32, -1_i32,  1_i32 - shear_factor),
            Vector3::new( 1_i32, -1_i32,  1_i32 + shear_factor),
            Vector3::new( 1_i32,  1_i32, -1_i32 + shear_factor),
            Vector3::new(-1_i32,  1_i32, -1_i32 - shear_factor),
            Vector3::new(-1_i32, -1_i32, -1_i32 - shear_factor),
            Vector3::new( 1_i32, -1_i32, -1_i32 + shear_factor),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zx_shearing_plane() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_shear_zx(shear_factor);
        let vertices = [
            Vector3::new(0_i32,  1_i32,  1_i32),
            Vector3::new(0_i32, -1_i32,  1_i32),
            Vector3::new(0_i32, -1_i32, -1_i32),
            Vector3::new(0_i32,  1_i32, -1_i32),
            Vector3::new(0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zy() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_shear_zy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32,  1_i32 + shear_factor),
            Vector3::new(-1_i32,  1_i32,  1_i32 + shear_factor),
            Vector3::new(-1_i32, -1_i32,  1_i32 - shear_factor),
            Vector3::new( 1_i32, -1_i32,  1_i32 - shear_factor),
            Vector3::new( 1_i32,  1_i32, -1_i32 + shear_factor),
            Vector3::new(-1_i32,  1_i32, -1_i32 + shear_factor),
            Vector3::new(-1_i32, -1_i32, -1_i32 - shear_factor),
            Vector3::new( 1_i32, -1_i32, -1_i32 - shear_factor),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zy_shearing_plane() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_shear_zy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32, 0_i32,  1_i32),
            Vector3::new(-1_i32, 0_i32,  1_i32),
            Vector3::new(-1_i32, 0_i32, -1_i32),
            Vector3::new( 1_i32, 0_i32, -1_i32),
            Vector3::new( 0_i32, 0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_xy() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let expected = Matrix3x3::from_shear_xy(shear_factor);
        let result = Matrix3x3::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_xz() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let expected = Matrix3x3::from_shear_xz(shear_factor);
        let result = Matrix3x3::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_yx() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let expected = Matrix3x3::from_shear_yx(shear_factor);
        let result = Matrix3x3::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_yz() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let expected = Matrix3x3::from_shear_yz(shear_factor);
        let result = Matrix3x3::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_zx() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_x());
        let expected = Matrix3x3::from_shear_zx(shear_factor);
        let result = Matrix3x3::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_zy() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let expected = Matrix3x3::from_shear_zy(shear_factor);
        let result = Matrix3x3::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }
}


/// Shearing along the plane `(1 / sqrt(3)) * x + (1 / sqrt(3)) * y - z == 0`
/// with direction `[sqrt(3 / 10), sqrt(3 / 10), sqrt(4 / 10)]` and normal
/// `[-sqrt(2 / 10), -sqrt(2 / 10), sqrt(6 / 10)]`.
#[cfg(test)]
mod matrix3x3_shear_noncoordinate_plane_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix3x3,
        Unit,
        Vector3,
    };
    use cglinalg_trigonometry::{
        Angle,
        Radians,
    };


    fn shear_factor() -> f64 {
        15_f64
    }

    fn rotation_angle_z_xy() -> Radians<f64> {
        let tan_rotation_angle_z_xy = 1_f64;

        Radians::atan2(tan_rotation_angle_z_xy, 1_f64)
    }

    fn rotation_angle_y_zx() -> Radians<f64> {
        let tan_rotation_y_zx = -f64::sqrt(2_f64 / 3_f64);

        Radians::atan2(tan_rotation_y_zx, 1_f64)
    }

    #[rustfmt::skip]
    fn direction() -> Unit<Vector3<f64>> {
        Unit::from_value(Vector3::new(
            f64::sqrt(3_f64 / 10_f64),
            f64::sqrt(3_f64 / 10_f64),
            f64::sqrt(4_f64 / 10_f64),
        ))
    }

    #[rustfmt::skip]
    fn normal() -> Unit<Vector3<f64>> {
        Unit::from_value(Vector3::new(
            -f64::sqrt(2_f64 / 10_f64),
            -f64::sqrt(2_f64 / 10_f64),
             f64::sqrt(6_f64 / 10_f64),
        ))
    }

    #[rustfmt::skip]
    fn rotation_z_xy() -> Matrix3x3<f64> {
        Matrix3x3::new(
            1_f64 / f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64), 0_f64,
           -1_f64 / f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64), 0_f64,
            0_f64,                    0_f64,                    1_f64,
        )
    }

    #[rustfmt::skip]
    fn rotation_z_xy_inv() -> Matrix3x3<f64> {
        Matrix3x3::new(
            1_f64 / f64::sqrt(2_f64), -1_f64 / f64::sqrt(2_f64), 0_f64,
            1_f64 / f64::sqrt(2_f64),  1_f64 / f64::sqrt(2_f64), 0_f64,
            0_f64,                     0_f64,                    1_f64,
        )
    }

    #[rustfmt::skip]
    fn rotation_y_zx() -> Matrix3x3<f64> {
        Matrix3x3::new(
             f64::sqrt(3_f64 / 5_f64), 0_f64, f64::sqrt(2_f64 / 5_f64),
             0_f64,                    1_f64, 0_f64,
            -f64::sqrt(2_f64 / 5_f64), 0_f64, f64::sqrt(3_f64 / 5_f64),
        )
    }

    #[rustfmt::skip]
    fn rotation_y_zx_inv() -> Matrix3x3<f64> {
        Matrix3x3::new(
            f64::sqrt(3_f64 / 5_f64), 0_f64, -f64::sqrt(2_f64 / 5_f64),
            0_f64,                    1_f64,  0_f64,
            f64::sqrt(2_f64 / 5_f64), 0_f64,  f64::sqrt(3_f64 / 5_f64),
        )
    }

    #[rustfmt::skip]
    fn rotation() -> Matrix3x3<f64> {
        Matrix3x3::new(
             f64::sqrt(3_f64 / 10_f64), f64::sqrt(3_f64 / 10_f64), f64::sqrt(2_f64 / 5_f64),
            -1_f64 / f64::sqrt(2_f64),  1_f64 / f64::sqrt(2_f64),  0_f64,
            -1_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64),  f64::sqrt(3_f64 / 5_f64),

        )
    }

    #[rustfmt::skip]
    fn rotation_inv() -> Matrix3x3<f64> {
        Matrix3x3::new(
            f64::sqrt(3_f64 / 10_f64), -1_f64 / f64::sqrt(2_f64), -1_f64 / f64::sqrt(5_f64),
            f64::sqrt(3_f64/ 10_f64),   1_f64 / f64::sqrt(2_f64), -1_f64 / f64::sqrt(5_f64),
            f64::sqrt(2_f64 / 5_f64),   0_f64,                     f64::sqrt(3_f64 / 5_f64),
        )
    }

    #[rustfmt::skip]
    fn shear_matrix_xz() -> Matrix3x3<f64> {
        let shear_factor = shear_factor();

        Matrix3x3::new(
            1_f64,        0_f64, 0_f64,
            0_f64,        1_f64, 0_f64,
            shear_factor, 0_f64, 1_f64,
        )
    }


    #[test]
    fn test_from_shear_rotation_angle_z_xy() {
        let tan_rotation_angle_z_xy = 1_f64;
        let rotation_angle_z_xy = Radians::atan2(tan_rotation_angle_z_xy, 1_f64);

        assert_relative_eq!(
            rotation_angle_z_xy.cos(),
            1_f64 / f64::sqrt(2_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
        assert_relative_eq!(
            rotation_angle_z_xy.sin(),
            1_f64 / f64::sqrt(2_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
    }

    #[test]
    fn test_from_shear_rotation_angle_y_zx() {
        let tan_rotation_y_zx = -f64::sqrt(2_f64 / 3_f64);
        let rotation_angle_y_zx = Radians::atan2(tan_rotation_y_zx, 1_f64);

        assert_relative_eq!(
            rotation_angle_y_zx.cos(),
            f64::sqrt(3_f64 / 5_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
        assert_relative_eq!(
            rotation_angle_y_zx.sin(),
            -f64::sqrt(2_f64 / 5_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
    }

    #[test]
    fn test_from_shear_rotation_z_xy() {
        let rotation_angle_z_xy = rotation_angle_z_xy();
        let expected = rotation_z_xy();
        let result = Matrix3x3::from_angle_z(rotation_angle_z_xy);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_shear_rotation_z_xy_inv() {
        let rotation_angle_z_xy = rotation_angle_z_xy();
        let expected = rotation_z_xy_inv();
        let result_inv = Matrix3x3::from_angle_z(rotation_angle_z_xy);
        let result = result_inv.try_inverse().unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_shear_rotation_y_zx() {
        let rotation_angle_y_zx = rotation_angle_y_zx();
        let expected = rotation_y_zx();
        let result = Matrix3x3::from_angle_y(rotation_angle_y_zx);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_shear_rotation_y_zx_inv() {
        let rotation_angle_y_zx = rotation_angle_y_zx();
        let expected = rotation_y_zx_inv();
        let result_inv = Matrix3x3::from_angle_y(rotation_angle_y_zx);
        let result = result_inv.try_inverse().unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_shear_rotation() {
        let rotation_z_xy = rotation_z_xy();
        let rotation_y_zx = rotation_y_zx();
        let expected = rotation();
        let result = rotation_z_xy * rotation_y_zx;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_shear_rotation_inv() {
        let rotation_z_xy_inv = rotation_z_xy_inv();
        let rotation_y_zx_inv = rotation_y_zx_inv();
        let expected = rotation_inv();
        let result = rotation_y_zx_inv * rotation_z_xy_inv;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_shear_direction_xz() {
        let direction = direction();
        let rotation_inv = rotation_inv();
        let expected = Vector3::unit_x();
        let result = rotation_inv * direction.into_inner();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_shear_normal_xz() {
        let normal = normal();
        let rotation_inv = rotation_inv();
        let expected = Vector3::unit_z();
        let result = rotation_inv * normal.into_inner();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_vertices_xz() {
        let rotation = rotation();
        let vertices = [
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(2_f64 / 5_f64)  + f64::sqrt(3_f64 / 5_f64)
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                 f64::sqrt(3_f64 / 5_f64)  - f64::sqrt(2_f64 / 5_f64)
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                 f64::sqrt(3_f64 / 5_f64)  - f64::sqrt(2_f64 / 5_f64)
            ),
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(2_f64 / 5_f64)  + f64::sqrt(3_f64 / 5_f64)
            ),
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(2_f64 / 5_f64)  - f64::sqrt(3_f64 / 5_f64)
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(2_f64 / 5_f64)  - f64::sqrt(3_f64 / 5_f64)
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(2_f64 / 5_f64)  - f64::sqrt(3_f64 / 5_f64)
            ),
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(2_f64 / 5_f64)  - f64::sqrt(3_f64 / 5_f64)
            ),
        ];
        let rotated_vertices = [
            Vector3::new( 1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64, -1_f64, -1_f64),
            Vector3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = vertices;
        let result = rotated_vertices.map(|v| rotation * v);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_vertices() {
        let shear_factor = shear_factor();
        let direction = direction();
        let normal = normal();
        let matrix = Matrix3x3::from_shear(shear_factor, &direction, &normal);
        let vertices = [
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(2_f64 / 5_f64)  + f64::sqrt(3_f64 / 5_f64)
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                 f64::sqrt(3_f64 / 5_f64)  - f64::sqrt(2_f64 / 5_f64)
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                 f64::sqrt(3_f64 / 5_f64)  - f64::sqrt(2_f64 / 5_f64)
            ),
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(2_f64 / 5_f64)  + f64::sqrt(3_f64 / 5_f64)
            ),
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(2_f64 / 5_f64)  - f64::sqrt(3_f64 / 5_f64)
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(2_f64 / 5_f64)  - f64::sqrt(3_f64 / 5_f64)
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(2_f64 / 5_f64)  - f64::sqrt(3_f64 / 5_f64)
            ),
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(2_f64 / 5_f64)  - f64::sqrt(3_f64 / 5_f64)
            ),
        ];
        let expected = [
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64) + f64::sqrt(3_f64 / 10_f64) * shear_factor,
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64) + f64::sqrt(3_f64 / 10_f64) * shear_factor,
                f64::sqrt(2_f64 / 5_f64)  + f64::sqrt(3_f64 / 5_f64) + f64::sqrt(2_f64 / 5_f64) * shear_factor
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64) + f64::sqrt(3_f64 / 10_f64) * shear_factor,
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64) + f64::sqrt(3_f64 / 10_f64) * shear_factor,
                 f64::sqrt(3_f64 / 5_f64)  - f64::sqrt(2_f64 / 5_f64) + f64::sqrt(2_f64 / 5_f64) * shear_factor
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64) + f64::sqrt(3_f64 / 10_f64) * shear_factor,
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64) + f64::sqrt(3_f64 / 10_f64) * shear_factor,
                 f64::sqrt(3_f64 / 5_f64)  - f64::sqrt(2_f64 / 5_f64) + f64::sqrt(2_f64 / 5_f64) * shear_factor
            ),
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64) + f64::sqrt(3_f64 / 10_f64) * shear_factor,
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64) + f64::sqrt(3_f64 / 10_f64) * shear_factor,
                f64::sqrt(2_f64 / 5_f64)  + f64::sqrt(3_f64 / 5_f64) + f64::sqrt(2_f64 / 5_f64) * shear_factor
            ),
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64) - f64::sqrt(3_f64 / 10_f64) * shear_factor,
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64) - f64::sqrt(3_f64 / 10_f64) * shear_factor,
                f64::sqrt(2_f64 / 5_f64)  - f64::sqrt(3_f64 / 5_f64) - f64::sqrt(2_f64 / 5_f64) * shear_factor
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64) - f64::sqrt(3_f64 / 10_f64) * shear_factor,
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64) - f64::sqrt(3_f64 / 10_f64) * shear_factor,
                -f64::sqrt(2_f64 / 5_f64)  - f64::sqrt(3_f64 / 5_f64) - f64::sqrt(2_f64 / 5_f64) * shear_factor
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64) - f64::sqrt(3_f64 / 10_f64) * shear_factor,
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64) - f64::sqrt(3_f64 / 10_f64) * shear_factor,
                -f64::sqrt(2_f64 / 5_f64)  - f64::sqrt(3_f64 / 5_f64) - f64::sqrt(2_f64 / 5_f64) * shear_factor
            ),
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64) - f64::sqrt(3_f64 / 10_f64) * shear_factor,
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64) - f64::sqrt(3_f64 / 10_f64) * shear_factor,
                f64::sqrt(2_f64 / 5_f64)  - f64::sqrt(3_f64 / 5_f64) - f64::sqrt(2_f64 / 5_f64) * shear_factor
            ),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_matrix() {
        let shear_factor = shear_factor();
        let direction = direction();
        let normal = normal();
        let expected = {
            let c0r0 = 1_f64 - (1_f64 / 5_f64) * f64::sqrt(3_f64 / 2_f64) * shear_factor;
            let c0r1 = -(1_f64 / 5_f64) * f64::sqrt(3_f64 / 2_f64) * shear_factor;
            let c0r2 = -(f64::sqrt(2_f64) / 5_f64) * shear_factor;

            let c1r0 = -(1_f64 / 5_f64) * f64::sqrt(3_f64 / 2_f64) * shear_factor;
            let c1r1 = 1_f64 - (1_f64 / 5_f64) * f64::sqrt(3_f64 / 2_f64) * shear_factor;
            let c1r2 = -(f64::sqrt(2_f64) / 5_f64) * shear_factor;

            let c2r0 = (3_f64 / (5_f64 * f64::sqrt(2_f64))) * shear_factor;
            let c2r1 = (3_f64 / (5_f64 * f64::sqrt(2_f64))) * shear_factor;
            let c2r2 = 1_f64 + (f64::sqrt(6_f64) / 5_f64) * shear_factor;
            
            Matrix3x3::new(
                c0r0, c0r1, c0r2,
                c1r0, c1r1, c1r2,
                c2r0, c2r1, c2r2,
            )
        };
        let result = Matrix3x3::from_shear(shear_factor, &direction, &normal);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_shear_matrix_alternative_path() {
        let shear_factor = shear_factor();
        let direction = direction();
        let normal = normal();
        let rotation = rotation();
        let rotation_inv = rotation_inv();
        let shear_matrix_xz = shear_matrix_xz();
        let expected = Matrix3x3::from_shear(shear_factor, &direction, &normal);
        let result = rotation * shear_matrix_xz * rotation_inv;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_shearing_plane() {
        let shear_factor = shear_factor();
        let direction = direction();
        let normal = normal();
        let matrix = Matrix3x3::from_shear(shear_factor, &direction, &normal);
        let vertices = [
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(2_f64 / 5_f64)  - 2_f64 / f64::sqrt(15_f64)
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(2_f64 / 5_f64)  - 2_f64 / f64::sqrt(15_f64)
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(2_f64 / 5_f64)  - 2_f64 / f64::sqrt(15_f64)
            ),
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) - 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(2_f64 / 5_f64)  - 2_f64 / f64::sqrt(15_f64)
            ),
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(2_f64 / 5_f64)  + 2_f64 / f64::sqrt(15_f64)
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(2_f64 / 5_f64)  + 2_f64 / f64::sqrt(15_f64)
            ),
            Vector3::new(
                -f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                -f64::sqrt(2_f64 / 5_f64)  + 2_f64 / f64::sqrt(15_f64)
            ),
            Vector3::new(
                f64::sqrt(3_f64 / 10_f64) + 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(3_f64 / 10_f64) - 1_f64 / f64::sqrt(2_f64) + 1_f64 / f64::sqrt(5_f64),
                f64::sqrt(2_f64 / 5_f64)  + 2_f64 / f64::sqrt(15_f64)
            ),
            Vector3::new(
                0_f64, 
                0_f64, 
                0_f64
            ),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}


#[cfg(test)]
mod matrix3x3_affine_shear_tests {
    use cglinalg_core::{
        Matrix3x3,
        Point2,
        Unit,
        Vector2,
        Vector3,
    };


    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy1() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_affine_shear_xy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32, 1_i32),
            Vector3::new(-1_i32,  1_i32, 1_i32),
            Vector3::new(-1_i32, -1_i32, 1_i32),
            Vector3::new( 1_i32, -1_i32, 1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32 + shear_factor,  1_i32, 1_i32),
            Vector3::new(-1_i32 + shear_factor,  1_i32, 1_i32),
            Vector3::new(-1_i32 - shear_factor, -1_i32, 1_i32),
            Vector3::new( 1_i32 - shear_factor, -1_i32, 1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy2() {
        let shear_factor = 5_i32;
        let expected = Matrix3x3::new(
            1_i32,        0_i32, 0_i32,
            shear_factor, 1_i32, 0_i32,
            0_i32,        0_i32, 1_i32,
        );
        let result = Matrix3x3::from_affine_shear_xy(shear_factor);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy_shearing_plane() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_affine_shear_xy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32, 0_i32, 1_i32),
            Vector3::new(-1_i32, 0_i32, 1_i32),
            Vector3::new( 0_i32, 0_i32, 1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_xy_does_not_change_homogeneous_coordinate() {
        let shear_factor = 5_i32;
        let matrix = Matrix3x3::from_affine_shear_xy(shear_factor);
        let unit_z = Vector3::unit_z();
        let expected = unit_z;
        let result = matrix * unit_z;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx1() {
        let shear_factor = 3_i32;
        let matrix = Matrix3x3::from_affine_shear_yx(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32, 1_i32),
            Vector3::new(-1_i32,  1_i32, 1_i32),
            Vector3::new(-1_i32, -1_i32, 1_i32),
            Vector3::new( 1_i32, -1_i32, 1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32 + shear_factor, 1_i32),
            Vector3::new(-1_i32,  1_i32 - shear_factor, 1_i32),
            Vector3::new(-1_i32, -1_i32 - shear_factor, 1_i32),
            Vector3::new( 1_i32, -1_i32 + shear_factor, 1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx2() {
        let shear_factor = 5_i32;
        let expected = Matrix3x3::new(
            1_i32, shear_factor, 0_i32,
            0_i32, 1_i32,        0_i32,
            0_i32, 0_i32,        1_i32
        );
        let result = Matrix3x3::from_affine_shear_yx(shear_factor);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx_shearing_plane() {
        let shear_factor = 3_i32;
        let matrix = Matrix3x3::from_affine_shear_yx(shear_factor);
        let vertices = [
            Vector3::new(0_i32,  1_i32, 1_i32),
            Vector3::new(0_i32, -1_i32, 1_i32),
            Vector3::new(0_i32,  0_i32, 1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_yx_does_not_change_homogeneous_coordinate() {
        let shear_factor = 3_i32;
        let matrix = Matrix3x3::from_affine_shear_yx(shear_factor);
        let unit_z = Vector3::unit_z();
        let expected = unit_z;
        let result = matrix * unit_z;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_from_affine_shear_xy() {
        let shear_factor = 5_f64;
        let origin = Point2::new(0_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_x());
        let normal = Unit::from_value(Vector2::unit_y());
        let expected = Matrix3x3::from_affine_shear_xy(shear_factor);
        let result = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_from_affine_shear_yx() {
        let shear_factor = 5_f64;
        let origin = Point2::new(0_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_y());
        let normal = Unit::from_value(Vector2::unit_x());
        let expected = Matrix3x3::from_affine_shear_yx(shear_factor);
        let result = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix3x3_affine_shear_coordinate_plane_tests {
    use cglinalg_core::{
        Matrix3x3,
        Point2,
        Unit,
        Vector2,
        Vector3,
    };


    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy() {
        let shear_factor = 7_f64;
        let origin = Point2::new(-2_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_x());
        let normal = Unit::from_value(Vector2::unit_y());
        let matrix = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64,  1_f64, 1_f64),
            Vector3::new(-1_f64,  1_f64, 1_f64),
            Vector3::new(-1_f64, -1_f64, 1_f64),
            Vector3::new( 1_f64, -1_f64, 1_f64),
        ];
        let expected = [
            Vector3::new( 1_f64 + shear_factor,  1_f64, 1_f64),
            Vector3::new(-1_f64 + shear_factor,  1_f64, 1_f64),
            Vector3::new(-1_f64 - shear_factor, -1_f64, 1_f64),
            Vector3::new( 1_f64 - shear_factor, -1_f64, 1_f64),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy_matrix() {
        let shear_factor = 7_f64;
        let origin = Point2::new(-2_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_x());
        let normal = Unit::from_value(Vector2::unit_y());
        let expected = Matrix3x3::new(
            1_f64,        0_f64, 0_f64,
            shear_factor, 1_f64, 0_f64,
            0_f64,        0_f64, 1_f64
        );
        let result = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy_shearing_plane() {
        let shear_factor = 7_f64;
        let origin = Point2::new(-2_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_x());
        let normal = Unit::from_value(Vector2::unit_y());
        let matrix = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64, 0_f64, 1_f64),
            Vector3::new(-1_f64, 0_f64, 1_f64),
            Vector3::new( 0_f64, 0_f64, 0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx() {
        let shear_factor = 7_f64;
        let origin = Point2::new(-2_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_y());
        let normal = Unit::from_value(Vector2::unit_x());
        let matrix = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64,  1_f64, 1_f64),
            Vector3::new(-1_f64,  1_f64, 1_f64),
            Vector3::new(-1_f64, -1_f64, 1_f64),
            Vector3::new( 1_f64, -1_f64, 1_f64),
        ];
        let expected = [
            Vector3::new( 1_f64,  1_f64 + 3_f64 * shear_factor, 1_f64),
            Vector3::new(-1_f64,  1_f64 + shear_factor,         1_f64),
            Vector3::new(-1_f64, -1_f64 + shear_factor,         1_f64),
            Vector3::new( 1_f64, -1_f64 + 3_f64 * shear_factor, 1_f64),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx_matrix() {
        let shear_factor = 7_f64;
        let origin = Point2::new(-2_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_y());
        let normal = Unit::from_value(Vector2::unit_x());
        let expected = Matrix3x3::new(
            1_f64,  shear_factor,             0_f64,
            0_f64,  1_f64,                    0_f64,
            0_f64, -origin[0] * shear_factor, 1_f64
        );
        let result = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx_shearing_plane() {
        let shear_factor = 7_f64;
        let origin = Point2::new(-2_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_y());
        let normal = Unit::from_value(Vector2::unit_x());
        let matrix = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new(-2_f64,  1_f64, 1_f64),
            Vector3::new(-2_f64, -1_f64, 1_f64),
            Vector3::new(-2_f64,  0_f64, 1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }
}


/// Shearing along the plane `(1 / 2) * x + 1 - y == 0`
/// with origin `[2, 2]`, direction `[2 / sqrt(5), 1 / sqrt(5)]`, and
/// normal `[-1 / sqrt(5), 2 / sqrt(5)]`.
#[cfg(test)]
mod matrix3x3_affine_shear_noncoordinate_plane_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix3x3,
        Point2,
        Unit,
        Vector2,
        Vector3,
    };
    use cglinalg_trigonometry::{
        Angle,
        Radians,
    };


    fn shear_factor() -> f64 {
        7_f64
    }

    fn rotation_angle() -> Radians<f64> {
        Radians(f64::atan2(1_f64, 2_f64))
    }

    fn origin() -> Point2<f64> {
        Point2::new(2_f64, 2_f64)
    }

    fn direction() -> Unit<Vector2<f64>> {
        Unit::from_value(Vector2::new(2_f64 / f64::sqrt(5_f64), 1_f64 / f64::sqrt(5_f64)))
    }

    fn normal() -> Unit<Vector2<f64>> {
        Unit::from_value(Vector2::new(-1_f64 / f64::sqrt(5_f64), 2_f64 / f64::sqrt(5_f64)))
    }

    fn translation() -> Matrix3x3<f64> {
        Matrix3x3::from_affine_translation(&Vector2::new(0_f64, 1_f64))
    }

    fn translation_inv() -> Matrix3x3<f64> {
        Matrix3x3::from_affine_translation(&Vector2::new(0_f64, -1_f64))
    }

    #[rustfmt::skip]
    fn rotation() -> Matrix3x3<f64> {
        Matrix3x3::new(
            2_f64 / f64::sqrt(5_f64), 1_f64 / f64::sqrt(5_f64), 0_f64,
           -1_f64 / f64::sqrt(5_f64), 2_f64 / f64::sqrt(5_f64), 0_f64,
            0_f64,                          0_f64,                         1_f64,
        )
    }

    #[rustfmt::skip]
    fn rotation_inv() -> Matrix3x3<f64> {
        Matrix3x3::new(
            2_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64), 0_f64,
            1_f64 / f64::sqrt(5_f64),  2_f64 / f64::sqrt(5_f64), 0_f64,
            0_f64,                           0_f64,                         1_f64,
        )
    }

    #[rustfmt::skip]
    fn shear_matrix_xy() -> Matrix3x3<f64> {
        let shear_factor = shear_factor();

        Matrix3x3::new(
            1_f64,        0_f64, 0_f64,
            shear_factor, 1_f64, 0_f64,
            0_f64,        0_f64, 1_f64
        )
    }


    #[test]
    fn test_from_affine_shear_rotation_angle() {
        let rotation_angle = rotation_angle();

        assert_relative_eq!(
            rotation_angle.cos(),
            2_f64 / f64::sqrt(5_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
        assert_relative_eq!(
            rotation_angle.sin(),
            1_f64 / f64::sqrt(5_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
    }

    #[test]
    fn test_from_affine_rotation_matrix() {
        let rotation_angle = rotation_angle();
        let expected = rotation();
        let result = Matrix3x3::from_affine_angle(rotation_angle);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn tests_from_affine_shear_coordinates() {
        let translation = translation();
        let rotation = rotation();
        let vertices = [
            Vector3::new( 1_f64 / f64::sqrt(5_f64),  3_f64 / f64::sqrt(5_f64) + 1_f64, 1_f64),
            Vector3::new(-3_f64 / f64::sqrt(5_f64),  1_f64 / f64::sqrt(5_f64) + 1_f64, 1_f64),
            Vector3::new(-1_f64 / f64::sqrt(5_f64), -3_f64 / f64::sqrt(5_f64) + 1_f64, 1_f64),
            Vector3::new( 3_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64) + 1_f64, 1_f64),
        ];
        let rotated_vertices = [
            Vector3::new( 1_f64,  1_f64, 1_f64),
            Vector3::new(-1_f64,  1_f64, 1_f64),
            Vector3::new(-1_f64, -1_f64, 1_f64),
            Vector3::new( 1_f64, -1_f64, 1_f64),
        ];
        let expected = vertices;
        let result = rotated_vertices.map(|v| translation * rotation * v);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_origin() {
        let origin = origin();
        let translation = translation();
        let rotation = rotation();
        let rotated_origin = Vector3::new(f64::sqrt(5_f64), 0_f64, 1_f64);
        let result_rotated_translated_origin = translation * rotation * rotated_origin;

        assert_relative_eq!(
            result_rotated_translated_origin[0],
            origin[0],
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
        assert_relative_eq!(
            result_rotated_translated_origin[1],
            origin[1],
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
        assert_relative_eq!(
            result_rotated_translated_origin[2],
            1_f64,
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
    }

    #[test]
    fn test_from_affine_shear_direction() {
        let direction = direction();
        let translation_inv = translation_inv();
        let rotation_inv = rotation_inv();
        let expected = Vector2::unit_x().extend(0_f64);
        let result = {
            let _direction = direction.into_inner().extend(0_f64);
            translation_inv * rotation_inv * _direction
        };

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_normal() {
        let normal = normal();
        let translation_inv = translation_inv();
        let rotation_inv = rotation_inv();
        let expected = Vector2::unit_y().extend(0_f64);
        let result = {
            let _normal = normal.into_inner().extend(0_f64);
            translation_inv * rotation_inv * _normal
        };

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_vertices() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let matrix = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64 / f64::sqrt(5_f64),  3_f64 / f64::sqrt(5_f64) + 1_f64, 1_f64),
            Vector3::new(-3_f64 / f64::sqrt(5_f64),  1_f64 / f64::sqrt(5_f64) + 1_f64, 1_f64),
            Vector3::new(-1_f64 / f64::sqrt(5_f64), -3_f64 / f64::sqrt(5_f64) + 1_f64, 1_f64),
            Vector3::new( 3_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64) + 1_f64, 1_f64),
        ];
        let expected = [
            Vector3::new(
                 (1_f64 / f64::sqrt(5_f64)) + (2_f64 / f64::sqrt(5_f64)) * shear_factor,
                 (3_f64 / f64::sqrt(5_f64)) + (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
                 1_f64,
            ),
            Vector3::new(
                -(3_f64 / f64::sqrt(5_f64)) + (2_f64 / f64::sqrt(5_f64)) * shear_factor,
                 (1_f64 / f64::sqrt(5_f64))  + (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
                 1_f64,
            ),
            Vector3::new(
                -(1_f64 / f64::sqrt(5_f64)) - (2_f64 / f64::sqrt(5_f64)) * shear_factor,
                -(3_f64 / f64::sqrt(5_f64)) - (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
                 1_f64,
            ),
            Vector3::new(
                 (3_f64 / f64::sqrt(5_f64)) - (2_f64 / f64::sqrt(5_f64)) * shear_factor,
                -(1_f64 / f64::sqrt(5_f64)) - (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
                 1_f64,
            ),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_matrix() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let expected = Matrix3x3::new(
             1_f64 - (2_f64 / 5_f64) * shear_factor, -(1_f64 / 5_f64) * shear_factor,         0_f64,
             (4_f64 / 5_f64) * shear_factor,          1_f64 + (2_f64 / 5_f64) * shear_factor, 0_f64,
            -(4_f64 / 5_f64) * shear_factor,         -(2_f64 / 5_f64) * shear_factor,         1_f64,
        );
        let result = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_matrix_alternative_path() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let translation = translation();
        let translation_inv = translation_inv();
        let rotation = rotation();
        let rotation_inv = rotation_inv();
        let shear_matrix_xy = shear_matrix_xy();
        let expected = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let result = (translation * rotation) * shear_matrix_xy * (rotation_inv * translation_inv);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_shearing_plane() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let matrix = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64 / f64::sqrt(5_f64),  1_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64, 1_f64),
            Vector3::new(-3_f64 / f64::sqrt(5_f64), -3_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64, 1_f64),
            Vector3::new(-1_f64 / f64::sqrt(5_f64), -1_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64, 1_f64),
            Vector3::new( 3_f64 / f64::sqrt(5_f64),  3_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64, 1_f64),
            Vector3::new( 0_f64, 1_f64, 1_f64),

        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);
    
        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}


#[cfg(test)]
mod matrix4x4_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix4x4,
        Vector3,
        Vector4,
    };


    #[rustfmt::skip]
    #[test]
    fn test_matrix_components1() {
        let matrix = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );

        assert_eq!(matrix[0][0], 1_i32);
        assert_eq!(matrix[0][1], 2_i32);
        assert_eq!(matrix[0][2], 3_i32);
        assert_eq!(matrix[0][3], 4_i32);
        assert_eq!(matrix[1][0], 5_i32);
        assert_eq!(matrix[1][1], 6_i32);
        assert_eq!(matrix[1][2], 7_i32);
        assert_eq!(matrix[1][3], 8_i32);
        assert_eq!(matrix[2][0], 9_i32);
        assert_eq!(matrix[2][1], 10_i32);
        assert_eq!(matrix[2][2], 11_i32);
        assert_eq!(matrix[2][3], 12_i32);
        assert_eq!(matrix[3][0], 13_i32);
        assert_eq!(matrix[3][1], 14_i32);
        assert_eq!(matrix[3][2], 15_i32);
        assert_eq!(matrix[3][3], 16_i32);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_components2() {
        let matrix = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
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

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds1() {
        let matrix = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );

        assert_eq!(matrix[4][0], matrix[4][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds2() {
        let matrix = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );

        assert_eq!(matrix[0][4], matrix[0][4]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds3() {
        let matrix = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );

        assert_eq!(matrix[4][4], matrix[4][4]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds4() {
        let matrix = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );

        assert_eq!(matrix[0][usize::MAX], matrix[0][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds5() {
        let matrix = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );

        assert_eq!(matrix[usize::MAX][0], matrix[usize::MAX][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds6() {
        let matrix = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );

        assert_eq!(matrix[usize::MAX][usize::MAX], matrix[usize::MAX][usize::MAX]);
    }

    #[test]
    fn test_identity_matrix_times_identity_matrix_equals_identity_matrix() {
        let identity_matrix: Matrix4x4<f32> = Matrix4x4::identity();

        assert_eq!(identity_matrix * identity_matrix, identity_matrix);
    }

    #[test]
    fn test_zero_matrix_times_zero_matrix_equals_zero_matrix() {
        let zero_matrix: Matrix4x4<f32> = Matrix4x4::zero();

        assert_eq!(zero_matrix * zero_matrix, zero_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_matrix1() {
        let a_matrix = Matrix4x4::new(
            80_f64,    23.43_f64,  43.56_f64, 6.74_f64,
            426.1_f64, 23.57_f64,  27.61_f64, 13.90_f64,
            4.22_f64,  258.08_f64, 31.70_f64, 42.17_f64,
            70_f64,    49_f64,     95_f64,    89.91_f64,
        );
        let b_matrix = Matrix4x4::new(
            36.84_f64, 427.46_f64, 882.19_f64, 89.50_f64,
            7.04_f64,  61.89_f64,  56.31_f64,  89_f64,
            72_f64,    936.5_f64,  413.80_f64, 50.31_f64,
            37.69_f64, 311.8_f64,  60.81_f64,  73.83_f64,
        );
        // let expected = Matrix4x4::new(
        //     195075.7478_f64, 242999.4886_f64, 49874.8440_f64, 51438.8929_f64,
        //     33402.1572_f64,  20517.1793_f64,  12255.4723_f64, 11284.3033_f64,
        //     410070.5860_f64, 133018.9590_f64, 46889.9950_f64, 35475.9481_f64,
        //     141297.8982_f64, 27543.7175_f64,  19192.1014_f64, 13790.4636_f64,
        // );
        let a_matrix_times_identity = a_matrix * Matrix4x4::identity();
        let b_matrix_times_identity = b_matrix * Matrix4x4::identity();

        assert_eq!(a_matrix_times_identity, a_matrix);
        assert_eq!(b_matrix_times_identity, b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_zero_equals_zero1() {
        let a_matrix = Matrix4x4::new(
            80_f64,    23.43_f64,  43.56_f64, 6.74_f64,
            426.1_f64, 23.57_f64,  27.61_f64, 13.90_f64,
            4.22_f64,  258.08_f64, 31.70_f64, 42.17_f64,
            70_f64,    49_f64,     95_f64,    89.91_f64,
        );
        let b_matrix = Matrix4x4::new(
            36.84_f64, 427.46_f64, 882.19_f64, 89.50_f64,
            7.04_f64,  61.89_f64,  56.31_f64,  89_f64,
            72_f64,    936.5_f64,  413.80_f64, 50.31_f64,
            37.69_f64, 311.8_f64,  60.81_f64,  73.83_f64,
        );
        // let expected = Matrix4x4::new(
        //     195075.7478_f64, 242999.4886_f64, 49874.8440_f64, 51438.8929_f64,
        //     33402.1572_f64,  20517.1793_f64,  12255.4723_f64, 11284.3033_f64,
        //     410070.5860_f64, 133018.9590_f64, 46889.9950_f64, 35475.9481_f64,
        //     141297.8982_f64, 27543.7175_f64,  19192.1014_f64, 13790.4636_f64,
        // );
        let a_matrix_times_zero_matrix = a_matrix * Matrix4x4::zero();
        let b_matrix_times_zero_matrix = b_matrix * Matrix4x4::zero();

        assert_eq!(a_matrix_times_zero_matrix, Matrix4x4::zero());
        assert_eq!(b_matrix_times_zero_matrix, Matrix4x4::zero());
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_times_matrix_equals_zero1() {
        let a_matrix = Matrix4x4::new(
            80_f64,    23.43_f64,  43.56_f64, 6.74_f64,
            426.1_f64, 23.57_f64,  27.61_f64, 13.90_f64,
            4.22_f64,  258.08_f64, 31.70_f64, 42.17_f64,
            70_f64,    49_f64,     95_f64,    89.91_f64,
        );
        let b_matrix = Matrix4x4::new(
            36.84_f64, 427.46_f64, 882.19_f64, 89.50_f64,
            7.04_f64,  61.89_f64,  56.31_f64,  89_f64,
            72_f64,    936.5_f64,  413.80_f64, 50.31_f64,
            37.69_f64, 311.8_f64,  60.81_f64,  73.83_f64,
        );
        // let expected = Matrix4x4::new(
        //     195075.7478_f64, 242999.4886_f64, 49874.8440_f64, 51438.8929_f64,
        //     33402.1572_f64,  20517.1793_f64,  12255.4723_f64, 11284.3033_f64,
        //     410070.5860_f64, 133018.9590_f64, 46889.9950_f64, 35475.9481_f64,
        //     141297.8982_f64, 27543.7175_f64,  19192.1014_f64, 13790.4636_f64,
        // );
        let zero_times_a_matrix = Matrix4x4::zero() * a_matrix;
        let zero_times_b_matrix = Matrix4x4::zero() * b_matrix;

        assert_eq!(zero_times_a_matrix, Matrix4x4::zero());
        assert_eq!(zero_times_b_matrix, Matrix4x4::zero());
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_identity_times_matrix1() {
        let a_matrix = Matrix4x4::new(
            80_f64,    23.43_f64,  43.56_f64, 6.74_f64,
            426.1_f64, 23.57_f64,  27.61_f64, 13.90_f64,
            4.22_f64,  258.08_f64, 31.70_f64, 42.17_f64,
            70_f64,    49_f64,     95_f64,    89.91_f64
        );
        let b_matrix = Matrix4x4::new(
            36.84_f64, 427.46_f64, 882.19_f64, 89.50_f64,
            7.04_f64,  61.89_f64,  56.31_f64,  89_f64,
            72_f64,    936.5_f64,  413.80_f64, 50.31_f64,
            37.69_f64, 311.8_f64,  60.81_f64,  73.83_f64,
        );
        // let expected = Matrix4x4::new(
        //     195075.7478_f64, 242999.4886_f64, 49874.8440_f64, 51438.8929_f64,
        //     33402.1572_f64,  20517.1793_f64,  12255.4723_f64, 11284.3033_f64,
        //     410070.5860_f64, 133018.9590_f64, 46889.9950_f64, 35475.9481_f64,
        //     141297.8982_f64, 27543.7175_f64,  19192.1014_f64, 13790.4636_f64,
        // );
        let a_matrix_times_identity = a_matrix * Matrix4x4::identity();
        let identity_times_a_matrix = Matrix4x4::identity() * a_matrix;
        let b_matrix_times_identity = b_matrix * Matrix4x4::identity();
        let identity_times_b_matrix = Matrix4x4::identity() * b_matrix;

        assert_eq!(a_matrix_times_identity, identity_times_a_matrix);
        assert_eq!(b_matrix_times_identity, identity_times_b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose_transpose_equals_matrix1() {
        let a_matrix = Matrix4x4::new(
            80_f64,    23.43_f64,  43.56_f64, 6.74_f64,
            426.1_f64, 23.57_f64,  27.61_f64, 13.90_f64,
            4.22_f64,  258.08_f64, 31.70_f64, 42.17_f64,
            70_f64,    49_f64,     95_f64,    89.91_f64,
        );
        let b_matrix = Matrix4x4::new(
            36.84_f64, 427.46_f64, 882.19_f64, 89.50_f64,
            7.04_f64,  61.89_f64,  56.31_f64,  89_f64,
            72_f64,    936.5_f64,  413.80_f64, 50.31_f64,
            37.69_f64, 311.8_f64,  60.81_f64,  73.83_f64,
        );
        // let expected = Matrix4x4::new(
        //     195075.7478_f64, 242999.4886_f64, 49874.8440_f64, 51438.8929_f64,
        //     33402.1572_f64,  20517.1793_f64,  12255.4723_f64, 11284.3033_f64,
        //     410070.5860_f64, 133018.9590_f64, 46889.9950_f64, 35475.9481_f64,
        //     141297.8982_f64, 27543.7175_f64,  19192.1014_f64, 13790.4636_f64,
        // );
        let a_matrix_transpose_transpose = a_matrix.transpose().transpose();
        let b_matrix_transpose_transpose = b_matrix.transpose().transpose();

        assert_eq!(a_matrix_transpose_transpose, a_matrix);
        assert_eq!(b_matrix_transpose_transpose, b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication1() {
        let a_matrix = Matrix4x4::new(
            80_f64,    23.43_f64,  43.56_f64, 6.74_f64,
            426.1_f64, 23.57_f64,  27.61_f64, 13.90_f64,
            4.22_f64,  258.08_f64, 31.70_f64, 42.17_f64,
            70_f64,    49_f64,     95_f64,    89.91_f64,
        );
        let b_matrix = Matrix4x4::new(
            36.84_f64, 427.46_f64, 882.19_f64, 89.50_f64,
            7.04_f64,  61.89_f64,  56.31_f64,  89_f64,
            72_f64,    936.5_f64,  413.80_f64, 50.31_f64,
            37.69_f64, 311.8_f64,  60.81_f64,  73.83_f64,
        );
        let expected = Matrix4x4::new(
            195075.7478_f64, 242999.4886_f64, 49874.8440_f64, 51438.8929_f64,
            33402.1572_f64,  20517.1793_f64,  12255.4723_f64, 11284.3033_f64,
            410070.5860_f64, 133018.9590_f64, 46889.9950_f64, 35475.9481_f64,
            141297.8982_f64, 27543.7175_f64,  19192.1014_f64, 13790.4636_f64,
        );
        let result = a_matrix * b_matrix;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_matrix2() {
        let a_matrix = Matrix4x4::new(
            68.32_f64, 0_f64,      0_f64,     0_f64,
            0_f64,     37.397_f64, 0_f64,     0_f64,
            0_f64,     0_f64,      9.483_f64, 0_f64,
            0_f64,     0_f64,      0_f64,     887.710_f64,
        );
        let b_matrix = Matrix4x4::new(
            57.72_f64, 0_f64,      0_f64,       0_f64,
            0_f64,     9.5433_f64, 0_f64,       0_f64,
            0_f64,     0_f64,      86.7312_f64, 0_f64,
            0_f64,     0_f64,      0_f64,       269.1134_f64,
        );
        // let expected = Matrix4x4::new(
        //     3943.4304_f64, 0_f64,           0_f64,           0_f64,
        //     0_f64,         356.8907901_f64, 0_f64,           0_f64,
        //     0_f64,         0_f64,           822.4719696_f64, 0_f64,
        //     0_f64,         0_f64,           0_f64,           238894.656314_f64,
        // );
        let a_matrix_times_identity = a_matrix * Matrix4x4::identity();
        let b_matrix_times_identity = b_matrix * Matrix4x4::identity();

        assert_eq!(a_matrix_times_identity, a_matrix);
        assert_eq!(b_matrix_times_identity, b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_zero_equals_zero2() {
        let a_matrix = Matrix4x4::new(
            68.32_f64, 0_f64,      0_f64,     0_f64,
            0_f64,     37.397_f64, 0_f64,     0_f64,
            0_f64,     0_f64,      9.483_f64, 0_f64,
            0_f64,     0_f64,      0_f64,     887.710_f64,
        );
        let b_matrix = Matrix4x4::new(
            57.72_f64, 0_f64,      0_f64,       0_f64,
            0_f64,     9.5433_f64, 0_f64,       0_f64,
            0_f64,     0_f64,      86.7312_f64, 0_f64,
            0_f64,     0_f64,      0_f64,       269.1134_f64,
        );
        // let expected = Matrix4x4::new(
        //     3943.4304_f64, 0_f64,           0_f64,           0_f64,
        //     0_f64,         356.8907901_f64, 0_f64,           0_f64,
        //     0_f64,         0_f64,           822.4719696_f64, 0_f64,
        //     0_f64,         0_f64,           0_f64,           238894.656314_f64,
        // );
        let a_matrix_times_zero_matrix = a_matrix * Matrix4x4::zero();
        let b_matrix_times_zero_matrix = b_matrix * Matrix4x4::zero();

        assert_eq!(a_matrix_times_zero_matrix, Matrix4x4::zero());
        assert_eq!(b_matrix_times_zero_matrix, Matrix4x4::zero());
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_times_matrix_equals_zero2() {
        let a_matrix = Matrix4x4::new(
            68.32_f64, 0_f64,      0_f64,     0_f64,
            0_f64,     37.397_f64, 0_f64,     0_f64,
            0_f64,     0_f64,      9.483_f64, 0_f64,
            0_f64,     0_f64,      0_f64,     887.710_f64,
        );
        let b_matrix = Matrix4x4::new(
            57.72_f64, 0_f64,      0_f64,       0_f64,
            0_f64,     9.5433_f64, 0_f64,       0_f64,
            0_f64,     0_f64,      86.7312_f64, 0_f64,
            0_f64,     0_f64,      0_f64,       269.1134_f64,
        );
        // let expected = Matrix4x4::new(
        //     3943.4304_f64, 0_f64,           0_f64,           0_f64,
        //     0_f64,         356.8907901_f64, 0_f64,           0_f64,
        //     0_f64,         0_f64,           822.4719696_f64, 0_f64,
        //     0_f64,         0_f64,           0_f64,           238894.656314_f64,
        // );
        let zero_times_a_matrix = Matrix4x4::zero() * a_matrix;
        let zero_times_b_matrix = Matrix4x4::zero() * b_matrix;

        assert_eq!(zero_times_a_matrix, Matrix4x4::zero());
        assert_eq!(zero_times_b_matrix, Matrix4x4::zero());
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_identity_times_matrix2() {
        let a_matrix = Matrix4x4::new(
            68.32_f64, 0_f64,      0_f64,     0_f64,
            0_f64,     37.397_f64, 0_f64,     0_f64,
            0_f64,     0_f64,      9.483_f64, 0_f64,
            0_f64,     0_f64,      0_f64,     887.710_f64,
        );
        let b_matrix = Matrix4x4::new(
            57.72_f64, 0_f64,      0_f64,       0_f64,
            0_f64,     9.5433_f64, 0_f64,       0_f64,
            0_f64,     0_f64,      86.7312_f64, 0_f64,
            0_f64,     0_f64,      0_f64,       269.1134_f64,
        );
        // let expected = Matrix4x4::new(
        //     3943.4304_f64, 0_f64,           0_f64,           0_f64,
        //     0_f64,         356.8907901_f64, 0_f64,           0_f64,
        //     0_f64,         0_f64,           822.4719696_f64, 0_f64,
        //     0_f64,         0_f64,           0_f64,           238894.656314_f64,
        // );
        let a_matrix_times_identity = a_matrix * Matrix4x4::identity();
        let identity_times_a_matrix = Matrix4x4::identity() * a_matrix;
        let b_matrix_times_identity = b_matrix * Matrix4x4::identity();
        let identity_times_b_matrix = Matrix4x4::identity() * b_matrix;

        assert_eq!(a_matrix_times_identity, identity_times_a_matrix);
        assert_eq!(b_matrix_times_identity, identity_times_b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose_transpose_equals_matrix2() {
        let a_matrix = Matrix4x4::new(
            68.32_f64, 0_f64,      0_f64,     0_f64,
            0_f64,     37.397_f64, 0_f64,     0_f64,
            0_f64,     0_f64,      9.483_f64, 0_f64,
            0_f64,     0_f64,      0_f64,     887.710_f64
        );
        let b_matrix = Matrix4x4::new(
            57.72_f64, 0_f64,      0_f64,       0_f64,
            0_f64,     9.5433_f64, 0_f64,       0_f64,
            0_f64,     0_f64,      86.7312_f64, 0_f64,
            0_f64,     0_f64,      0_f64,       269.1134_f64
        );
        // let expected = Matrix4x4::new(
        //     3943.4304_f64, 0_f64,           0_f64,           0_f64,
        //     0_f64,         356.8907901_f64, 0_f64,           0_f64,
        //     0_f64,         0_f64,           822.4719696_f64, 0_f64,
        //     0_f64,         0_f64,           0_f64,           238894.656314_f64,
        // );
        let a_matrix_transpose_transpose = a_matrix.transpose().transpose();
        let b_matrix_transpose_transpose = b_matrix.transpose().transpose();

        assert_eq!(a_matrix_transpose_transpose, a_matrix);
        assert_eq!(b_matrix_transpose_transpose, b_matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication2() {
        let a_matrix = Matrix4x4::new(
            68.32_f64, 0_f64,      0_f64,     0_f64,
            0_f64,     37.397_f64, 0_f64,     0_f64,
            0_f64,     0_f64,      9.483_f64, 0_f64,
            0_f64,     0_f64,      0_f64,     887.710_f64
        );
        let b_matrix = Matrix4x4::new(
            57.72_f64, 0_f64,      0_f64,       0_f64,
            0_f64,     9.5433_f64, 0_f64,       0_f64,
            0_f64,     0_f64,      86.7312_f64, 0_f64,
            0_f64,     0_f64,      0_f64,       269.1134_f64
        );
        let expected = Matrix4x4::new(
            3943.4304_f64, 0_f64,           0_f64,           0_f64,
            0_f64,         356.8907901_f64, 0_f64,           0_f64,
            0_f64,         0_f64,           822.4719696_f64, 0_f64,
            0_f64,         0_f64,           0_f64,           238894.656314_f64,
        );
        let result = a_matrix * b_matrix;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_identity_transpose_equals_identity() {
        let identity = Matrix4x4::<f32>::identity();
        let identity_transpose = identity.transpose();

        assert_eq!(identity, identity_transpose);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector4::new(1_i32,  2_i32,  3_i32,  4_i32);
        let c1 = Vector4::new(5_i32,  6_i32,  7_i32,  8_i32);
        let c2 = Vector4::new(9_i32,  10_i32, 11_i32, 12_i32);
        let c3 = Vector4::new(13_i32, 14_i32, 15_i32, 16_i32);
        let columns = [c0, c1, c2, c3];
        let expected = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );
        let result = Matrix4x4::from_columns(&columns);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_rows() {
        let r0 = Vector4::new(1_i32,  2_i32,  3_i32,  4_i32);
        let r1 = Vector4::new(5_i32,  6_i32,  7_i32,  8_i32);
        let r2 = Vector4::new(9_i32,  10_i32, 11_i32, 12_i32);
        let r3 = Vector4::new(13_i32, 14_i32, 15_i32, 16_i32);
        let rows = [r0, r1, r2, r3];
        let expected = Matrix4x4::new(
            1_i32, 5_i32, 9_i32,  13_i32,
            2_i32, 6_i32, 10_i32, 14_i32,
            3_i32, 7_i32, 11_i32, 15_i32,
            4_i32, 8_i32, 12_i32, 16_i32,
        );
        let result = Matrix4x4::from_rows(&rows);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_identity_matrix4x4_translates_vector_along_vector() {
        let vector = Vector3::from((2_f64, 2_f64, 2_f64));
        let trans_matrix = Matrix4x4::from_affine_translation(&vector);
        let zero_vector4 = Vector4::from((0_f64, 0_f64, 0_f64, 1_f64));
        let zero_vector3 = Vector3::from((0_f64, 0_f64, 0_f64));

        let result = trans_matrix * zero_vector4;
        assert_eq!(result, (zero_vector3 + vector).extend(1_f64));
    }

    #[rustfmt::skip]
    #[test]
    fn test_constant_times_identity_is_constant_along_diagonal() {
        let c = 802.3435169_f64;
        let identity = Matrix4x4::identity();
        let expected = Matrix4x4::new(
            c,     0_f64, 0_f64, 0_f64,
            0_f64, c,     0_f64, 0_f64,
            0_f64, 0_f64, c,     0_f64,
            0_f64, 0_f64, 0_f64, c,
        );

        assert_eq!(identity * c, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_identity_divide_constant_is_constant_inverse_along_diagonal() {
        let c = 802.3435169_f64;
        let identity = Matrix4x4::identity();
        let expected = Matrix4x4::new(
            1_f64 / c, 0_f64,     0_f64,     0_f64,
            0_f64,     1_f64 / c, 0_f64,     0_f64,
            0_f64,     0_f64,     1_f64 / c, 0_f64,
            0_f64,     0_f64,     0_f64,     1_f64 / c,
        );

        assert_eq!(identity / c, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero_matrix = Matrix4x4::zero();
        let matrix = Matrix4x4::new(
            36.84_f64,   427.46894_f64, 8827.1983_f64, 89.5049494_f64,
            7.04217_f64, 61.891390_f64, 56.31_f64,     89_f64,
            72_f64,      936.5_f64,     413.80_f64,    50.311160_f64,
            37.6985_f64, 311.8_f64,     60.81_f64,     73.8393_f64,
        );

        assert_eq!(matrix + zero_matrix, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero_matrix = Matrix4x4::zero();
        let matrix = Matrix4x4::new(
            36.84_f64,   427.46894_f64, 8827.1983_f64, 89.5049494_f64,
            7.04217_f64, 61.891390_f64, 56.31_f64,     89_f64,
            72_f64,      936.5_f64,     413.80_f64,    50.311160_f64,
            37.6985_f64, 311.8_f64,     60.81_f64,     73.8393_f64,
        );

        assert_eq!(zero_matrix + matrix, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_with_zero_determinant() {
        // This matrix should have a zero determinant since it has two repeating columns.
        let matrix = Matrix4x4::new(
            1_f64,  2_f64,  3_f64,  4_f64,
            5_f64,  6_f64,  7_f64,  8_f64,
            5_f64,  6_f64,  7_f64,  8_f64,
            9_f64,  10_f64, 11_f64, 12_f64,
        );

        assert_eq!(matrix.determinant(), 0_f64);
    }

    #[rustfmt::skip]
    #[test]
    fn test_lower_triangular_matrix_determinant() {
        let matrix = Matrix4x4::new(
            1_f64,  0_f64,  0_f64,  0_f64,
            5_f64,  2_f64,  0_f64,  0_f64,
            5_f64,  5_f64,  3_f64,  0_f64,
            5_f64,  5_f64,  5_f64,  4_f64,
        );

        assert_eq!(matrix.determinant(), 1_f64 * 2_f64 * 3_f64 * 4_f64);
    }

    #[rustfmt::skip]
    #[test]
    fn test_upper_triangular_matrix_determinant() {
        let matrix = Matrix4x4::new(
            1_f64,  5_f64,  5_f64,  5_f64,
            0_f64,  2_f64,  5_f64,  5_f64,
            0_f64,  0_f64,  3_f64,  5_f64,
            0_f64,  0_f64,  0_f64,  4_f64,
        );

        assert_eq!(matrix.determinant(), 1_f64 * 2_f64 * 3_f64 * 4_f64);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_multiplication() {
        let result = (1_f64 / 32_f64) * Matrix4x4::new(
            7_f64, -1_f64, -1_f64, -1_f64,
           -1_f64,  7_f64, -1_f64, -1_f64,
           -1_f64, -1_f64,  7_f64, -1_f64,
           -1_f64, -1_f64, -1_f64,  7_f64,
       );
       let expected = Matrix4x4::new(
           (1_f64 / 32_f64) *  7_f64, (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) * -1_f64,
           (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) *  7_f64, (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) * -1_f64,
           (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) *  7_f64, (1_f64 / 32_f64) * -1_f64,
           (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) * -1_f64, (1_f64 / 32_f64) *  7_f64,
       );

       assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_inverse() {
        let matrix = Matrix4x4::new(
            5_f64, 1_f64, 1_f64, 1_f64,
            1_f64, 5_f64, 1_f64, 1_f64,
            1_f64, 1_f64, 5_f64, 1_f64,
            1_f64, 1_f64, 1_f64, 5_f64,
        );
        let expected = (1_f64 / 32_f64) * Matrix4x4::new(
             7_f64, -1_f64, -1_f64, -1_f64,
            -1_f64,  7_f64, -1_f64, -1_f64,
            -1_f64, -1_f64,  7_f64, -1_f64,
            -1_f64, -1_f64, -1_f64,  7_f64,
        );
        let result = matrix.try_inverse().unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_identity_is_invertible() {
        assert!(Matrix4x4::<f64>::identity().is_invertible());
    }

    #[test]
    fn test_identity_inverse_is_identity() {
        let result: Matrix4x4<f64> = Matrix4x4::identity().try_inverse().unwrap();
        let expected: Matrix4x4<f64> = Matrix4x4::identity();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_diagonal_matrix() {
        let matrix = 4_f64 * Matrix4x4::identity();
        let expected = (1_f64 / 4_f64) * Matrix4x4::identity();
        let result = matrix.try_inverse().unwrap();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_with_nonzero_determinant_is_invertible() {
        let matrix = Matrix4x4::new(
            1_f64,  2_f64,  3_f64,   4_f64,
            5_f64,  60_f64, 7_f64,   8_f64,
            9_f64,  10_f64, 11_f64,  12_f64,
            13_f64, 14_f64, 150_f64, 16_f64,
        );

        assert!(matrix.is_invertible());
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_with_zero_determinant_is_not_invertible() {
        // This matrix should not be invertible since it has two identical columns.
        let matrix = Matrix4x4::new(
            1_f64,  2_f64,   3_f64,  4_f64,
            5_f64,  6_f64,   7_f64,  8_f64,
            5_f64,  6_f64,   7_f64,  8_f64,
            9_f64,  10_f64,  11_f64, 12_f64,
        );

        assert!(!matrix.is_invertible());
    }

    #[rustfmt::skip]
    #[test]
    fn test_noninvertible_matrix_returns_none() {
        let matrix = Matrix4x4::new(
            1_f64,  2_f64,  3_f64,  4_f64,
            5_f64,  6_f64,  7_f64,  8_f64,
            5_f64,  6_f64,  7_f64,  8_f64,
            9_f64,  10_f64, 11_f64, 12_f64,
        );

        assert!(matrix.try_inverse().is_none());
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_inversion2() {
        let matrix = Matrix4x4::new(
            36.84_f64,  427.468_f64, 882.198_f64, 89.504_f64,
            7.042_f64,  61.891_f64,  56.31_f64,   89_f64,
            72_f64,     936.5_f64,   413.80_f64,  50.311_f64,
            37.698_f64, 311.8_f64,   60.81_f64,   73.839_f64,
        );
        let result = matrix.try_inverse().unwrap();
        let expected = Matrix4x4::new(
             0.01146093272878252_f64,  -0.06212100841992658_f64, -0.02771783718075694_f64,    0.07986947998777854_f64,
            -0.00148039611514755_f64,   0.004464130960444646_f64, 0.003417891441120325_f64,  -0.005915083057511776_f64,
             0.001453087396607042_f64, -0.0009538600348427_f64,  -0.0005129477357421059_f64, -0.0002621470728476185_f64,
            -0.0007967195911958656_f64, 0.01365031989418242_f64,  0.0001408581712825875_f64, -0.002040325515611523_f64,
        );

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_inverse_is_identity() {
        let matrix = Matrix4x4::new(
            36.84_f64,  427.468_f64, 882.198_f64, 89.504_f64,
            7.042_f64,  61.891_f64,  56.31_f64,   89_f64,
            72_f64,     936.5_f64,   413.80_f64,  50.311_f64,
            37.698_f64, 311.8_f64,   60.81_f64,   73.839_f64,
        );
        let matrix_inverse = matrix.try_inverse().unwrap();
        let identity = Matrix4x4::identity();

        assert_relative_eq!(matrix * matrix_inverse, identity, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_constant_times_matrix_inverse_equals_constant_inverse_times_matrix_inverse() {
        let matrix = Matrix4x4::new(
            36.84_f64,  427.468_f64, 882.198_f64, 89.504_f64,
            7.042_f64,  61.891_f64,  56.31_f64,   89_f64,
            72_f64,     936.5_f64,   413.80_f64,  50.311_f64,
            37.698_f64, 311.8_f64,   60.81_f64,   73.839_f64,
        );
        let constant: f64 = 4_f64;
        let constant_times_matrix_inverse = (constant * matrix).try_inverse().unwrap();
        let constant_inverse_times_matrix_inverse = (1_f64 / constant) * matrix.try_inverse().unwrap();

        assert_eq!(constant_times_matrix_inverse, constant_inverse_times_matrix_inverse);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose_inverse_equals_matrix_inverse_transpose() {
        let matrix = Matrix4x4::new(
            36.84_f64,  427.468_f64, 882.198_f64, 89.504_f64,
            7.042_f64,  61.891_f64,  56.31_f64,   89_f64,
            72_f64,     936.5_f64,   413.80_f64,  50.311_f64,
            37.698_f64, 311.8_f64,   60.81_f64,   73.839_f64,
        );
        let matrix_transpose_inverse = matrix.transpose().try_inverse().unwrap();
        let matrix_inverse_transpose = matrix.try_inverse().unwrap().transpose();

        assert_relative_eq!(matrix_transpose_inverse, matrix_inverse_transpose, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_inverse_times_matrix_is_identity() {
        let matrix = Matrix4x4::new(
            36.84_f64,  427.468_f64, 882.198_f64, 89.504_f64,
            7.042_f64,  61.891_f64,  56.31_f64,   89_f64,
            72_f64,     936.5_f64,   413.80_f64,  50.311_f64,
            37.698_f64, 311.8_f64,   60.81_f64,   73.839_f64,
        );
        let matrix_inverse = matrix.try_inverse().unwrap();
        let identity = Matrix4x4::identity();

        assert_relative_eq!(matrix_inverse * matrix, identity, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_inverse_inverse_equals_matrix() {
        let matrix = Matrix4x4::new(
            36.84_f64,  427.468_f64, 882.198_f64, 89.504_f64,
            7.042_f64,  61.891_f64,  56.31_f64,   89_f64,
            72_f64,     936.5_f64,   413.80_f64,  50.311_f64,
            37.698_f64, 311.8_f64,   60.81_f64,   73.839_f64,
        );
        let result = matrix.try_inverse().unwrap().try_inverse().unwrap();
        let expected = matrix;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_elements_should_be_column_major_order() {
        let matrix = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
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

    #[rustfmt::skip]
    #[test]
    fn test_matrix_swap_columns() {
        let mut result = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,   4_i32,
            5_i32,  6_i32,  7_i32,   8_i32,
            9_i32,  10_i32, 11_i32,  12_i32,
            13_i32, 14_i32, 15_i32,  16_i32,
        );
        result.swap_columns(3, 1);
        let expected = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
        );

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_swap_rows() {
        let mut result = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );
        result.swap_rows(3, 1);
        let expected = Matrix4x4::new(
            1_i32,  4_i32,  3_i32,  2_i32,
            5_i32,  8_i32,  7_i32,  6_i32,
            9_i32,  12_i32, 11_i32, 10_i32,
            13_i32, 16_i32, 15_i32, 14_i32,
        );

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_swap_elements() {
        let mut result = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );
        result.swap((2, 0), (1, 3));
        let expected = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  9_i32,
            8_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_scale() {
        let matrix = Matrix4x4::from_affine_scale(5_i32);
        let unit_w = Vector4::unit_w();
        let expected = Vector4::new(5_i32, 5_i32, 5_i32, 1_i32);
        let result = matrix * Vector4::new(1_i32, 1_i32, 1_i32, 1_i32);

        assert_eq!(result, expected);
        assert_eq!(matrix * unit_w, unit_w);
    }

    #[test]
    fn test_from_affine_nonuniform_scale() {
        let matrix = Matrix4x4::from_affine_nonuniform_scale(&Vector3::new(5_i32, 7_i32, 11_i32));
        let unit_w = Vector4::unit_w();
        let expected = Vector4::new(5_i32, 7_i32, 11_i32, 1_i32);
        let result = matrix * Vector4::new(1_i32, 1_i32, 1_i32, 1_i32);

        assert_eq!(result, expected);
        assert_eq!(matrix * unit_w, unit_w);
    }

    /// An affine translation should only displace points and not vectors. We
    /// distinguish points by using a `1` in the last coordinate, and vectors
    /// by using a `0` in the last coordinate.
    #[test]
    fn test_from_affine_translation_point() {
        let distance = Vector3::new(3_i32, 7_i32, 11_i32);
        let matrix = Matrix4x4::from_affine_translation(&distance);
        let point = Vector4::new(0_i32, 0_i32, 0_i32, 1_i32);
        let expected = Vector4::new(3_i32, 7_i32, 11_i32, 1_i32);
        let result = matrix * point;

        assert_eq!(result, expected);
    }

    /// An affine translation should only displace points and not vectors. We
    /// distinguish points by using a `1` in the last coordinate, and vectors
    /// by using a `0` in the last coordinate.
    #[test]
    fn test_from_affine_translation_vector() {
        let distance = Vector3::new(3_i32, 7_i32, 11_i32);
        let matrix = Matrix4x4::from_affine_translation(&distance);
        let vector = Vector4::new(0_i32, 0_i32, 0_i32, 0_i32);
        let expected = vector;
        let result = matrix * vector;

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix4x4_projection_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::Matrix4x4;
    use cglinalg_trigonometry::Degrees;


    #[rustfmt::skip]
    #[test]
    fn test_from_orthographic() {
        let left = -4_f64;
        let right = 4_f64;
        let bottom = -2_f64;
        let top = 2_f64;
        let near = 1_f64;
        let far = 100_f64;
        let expected = Matrix4x4::new(
            1_f64 / 4_f64,  0_f64,          0_f64,            0_f64,
            0_f64,          1_f64 / 2_f64,  0_f64,            0_f64,
            0_f64,          0_f64,         -2_f64 / 99_f64,   0_f64,
            0_f64,          0_f64,         -101_f64 / 99_f64, 1_f64,
        );
        let result = Matrix4x4::from_orthographic(left, right, bottom, top, near, far);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_perspective_fov() {
        let vfov = Degrees(72_f32);
        let aspect_ratio = 800_f32 / 600_f32;
        let near = 0.1_f32;
        let far = 100_f32;
        let expected = Matrix4x4::new(
            1.0322863_f32, 0_f32,          0_f32,         0_f32,
            0_f32,         1.3763818_f32,  0_f32,         0_f32,
            0_f32,         0_f32,         -1.002002_f32, -1_f32,
            0_f32,         0_f32,         -0.2002002_f32, 0_f32,
        );
        let result = Matrix4x4::from_perspective_fov(vfov, aspect_ratio, near, far);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f32::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_perspective() {
        let left = -4_f64;
        let right = 4_f64;
        let bottom = -2_f64;
        let top = 3_f64;
        let near = 1_f64;
        let far = 100_f64;
        let expected = Matrix4x4::new(
            1_f64 / 4_f64, 0_f64,          0_f64,             0_f64,
            0_f64,         2_f64 / 5_f64,  0_f64,             0_f64,
            0_f64,         1_f64 / 5_f64, -101_f64 / 99_f64, -1_f64,
            0_f64,         0_f64,         -200_f64 / 99_f64,  0_f64,
        );
        let result = Matrix4x4::from_perspective(left, right, bottom, top, near, far);

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix4x4_rotation_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix4x4,
        Normed,
        Point3,
        Unit,
        Vector3,
        Vector4,
    };
    use cglinalg_trigonometry::{
        Angle,
        Radians,
    };


    #[test]
    fn test_from_angle_x() {
        let angle: Radians<f64> = Radians::full_turn_div_4();
        let unit_y = Vector3::unit_y();
        let unit_z = Vector3::unit_z();
        let matrix = Matrix4x4::from_affine_angle_x(angle);
        let expected = unit_z.extend(0_f64);
        let result = matrix * unit_y.extend(0_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_angle_y() {
        let angle: Radians<f64> = Radians::full_turn_div_4();
        let unit_z = Vector3::unit_z();
        let unit_x = Vector3::unit_x();
        let matrix = Matrix4x4::from_affine_angle_y(angle);
        let expected = unit_x.extend(0_f64);
        let result = matrix * unit_z.extend(0_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_angle_z() {
        let angle: Radians<f64> = Radians::full_turn_div_4();
        let unit_x = Vector3::unit_x();
        let unit_y = Vector3::unit_y();
        let matrix = Matrix4x4::from_affine_angle_z(angle);
        let expected = unit_y.extend(0_f64);
        let result = matrix * unit_x.extend(0_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_angle_x() {
        let angle: Radians<f64> = Radians::full_turn_div_4();
        let unit_y = Vector3::unit_y();
        let unit_z = Vector3::unit_z();
        let matrix = Matrix4x4::from_affine_angle_x(angle);
        let expected = unit_z.extend(0_f64);
        let result = matrix * unit_y.extend(0_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_angle_y() {
        let angle: Radians<f64> = Radians::full_turn_div_4();
        let unit_z = Vector3::unit_z();
        let unit_x = Vector3::unit_x();
        let matrix = Matrix4x4::from_affine_angle_y(angle);
        let expected = unit_x.extend(0_f64);
        let result = matrix * unit_z.extend(0_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_angle_z() {
        let angle: Radians<f64> = Radians::full_turn_div_4();
        let unit_x = Vector3::unit_x();
        let unit_y = Vector3::unit_y();
        let matrix = Matrix4x4::from_affine_angle_z(angle);
        let expected = unit_y.extend(0_f64);
        let result = matrix * unit_x.extend(0_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_axis_angle() {
        let angle: Radians<f64> = Radians::full_turn_div_2();
        let axis = Unit::from_value((1_f64 / f64::sqrt(2_f64)) * Vector3::new(1_f64, 1_f64, 0_f64));
        let vector = Vector4::new(1_f64, 1_f64, -1_f64, 0_f64);
        let matrix = Matrix4x4::from_affine_axis_angle(&axis, angle);
        let expected = Vector4::new(1_f64, 1_f64, 1_f64, 0_f64);
        let result = matrix * vector;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_look_at_rh_at_origin() {
        let eye = Point3::new(0_f64, 0_f64, 0_f64);
        let target = Point3::new(1_f64, 1_f64, 1_f64);
        let up = Vector3::unit_y();
        let minus_unit_z = -Vector3::unit_z();
        let look_at = Matrix4x4::look_at_rh(&eye, &target, &up);
        let direction = target - Point3::origin();
        let expected = minus_unit_z.extend(0_f64);
        let result = look_at * direction.normalize().extend(0_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_look_at_lh_at_origin() {
        let eye = Point3::new(0_f64, 0_f64, 0_f64);
        let target = Point3::new(1_f64, 1_f64, 1_f64);
        let up = Vector3::unit_y();
        let unit_z = Vector3::unit_z();
        let look_at = Matrix4x4::look_at_lh(&eye, &target, &up);
        let direction = target - Point3::origin();
        let expected = unit_z.extend(0_f64);
        let result = look_at * direction.normalize().extend(0_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_look_at_lh_no_displacement_or_rotation() {
        let eye = Point3::new(0_f64, 0_f64, 0_f64);
        let target = Point3::new(0_f64, 0_f64, 1_f64);
        let up = Vector3::unit_y();
        let look_at = Matrix4x4::look_at_lh(&eye, &target, &up);
        let direction = target - Point3::origin();
        let expected = Vector4::new(0_f64, 0_f64, 1_f64, 0_f64);
        let result = look_at * direction.normalize().extend(0_f64);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_look_at_rh_no_displacement_or_rotation() {
        let eye = Point3::new(0_f64, 0_f64, 0_f64);
        let target = Point3::new(0_f64, 0_f64, 1_f64);
        let up = Vector3::unit_y();
        let look_at = Matrix4x4::look_at_rh(&eye, &target, &up);
        let expected = Vector4::new(0_f64, 0_f64, 0_f64, 1_f64);
        let result = look_at * eye.to_homogeneous();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_look_at_lh_eye_to_origin() {
        let eye = Point3::new(-1_f64, -1_f64, -1_f64);
        let target = Point3::new(1_f64, 1_f64, 1_f64);
        let up = Vector3::unit_y();
        let look_at = Matrix4x4::look_at_lh(&eye, &target, &up);
        let expected = Vector4::unit_w();
        let result = look_at * eye.to_homogeneous();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_look_at_rh_eye_to_origin() {
        let eye = Point3::new(-1_f64, -1_f64, -1_f64);
        let target = Point3::new(1_f64, 1_f64, 1_f64);
        let up = Vector3::unit_y();
        let look_at = Matrix4x4::look_at_rh(&eye, &target, &up);
        let expected = Vector4::unit_w();
        let result = look_at * eye.to_homogeneous();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_look_to_lh1() {
        let eye = Point3::new(1_f64, 1_f64, 1_f64);
        let direction = (eye - Point3::origin()).normalize();
        let up = Vector3::unit_y();
        let unit_z = Vector3::unit_z().to_homogeneous();
        let look_to = Matrix4x4::look_to_lh(&eye, &direction, &up);
        let expected = unit_z;
        let result = look_to * direction.normalize().to_homogeneous();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_look_to_lh2() {
        let eye = Point3::new(1_f64, 1_f64, 1_f64);
        let direction = (eye - Point3::origin()).normalize();
        let up = Vector3::unit_y();
        let minus_unit_z = (-Vector3::unit_z()).to_homogeneous();
        let look_to = Matrix4x4::look_to_lh(&eye, &direction, &up);
        let expected = minus_unit_z;
        let result = look_to * (-direction).normalize().to_homogeneous();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }
}


#[cfg(test)]
mod matrix4x4_reflection_tests {
    use cglinalg_core::{
        Matrix4x4,
        Point3,
        Unit,
        Vector3,
        Vector4,
    };


    #[rustfmt::skip]
    #[test]
    fn test_from_affine_reflection_xy_plane() {
        let bias = Point3::origin();
        let normal = Unit::from_value(Vector3::unit_z());
        let expected = Matrix4x4::new(
            1_f64, 0_f64,  0_f64, 0_f64,
            0_f64, 1_f64,  0_f64, 0_f64,
            0_f64, 0_f64, -1_f64, 0_f64,
            0_f64, 0_f64,  0_f64, 1_f64,
        );
        let result = Matrix4x4::from_affine_reflection(&normal, &bias);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_reflection_zx_plane() {
        let bias = Point3::origin();
        let normal = Unit::from_value(-Vector3::unit_y());
        let expected = Matrix4x4::new(
            1_f64,  0_f64, 0_f64, 0_f64,
            0_f64, -1_f64, 0_f64, 0_f64,
            0_f64,  0_f64, 1_f64, 0_f64,
            0_f64,  0_f64, 0_f64, 1_f64,
        );
        let result = Matrix4x4::from_affine_reflection(&normal, &bias);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_reflection_yz_plane() {
        let bias = Point3::origin();
        let normal = Unit::from_value(Vector3::unit_x());
        let expected = Matrix4x4::new(
            -1_f64,  0_f64, 0_f64,  0_f64,
             0_f64,  1_f64, 0_f64,  0_f64,
             0_f64,  0_f64, 1_f64,  0_f64,
             0_f64,  0_f64, 0_f64,  1_f64,
        );
        let result = Matrix4x4::from_affine_reflection(&normal, &bias);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_reflection_plane1() {
        // The plane `z = 1`.
        let bias = Point3::new(0_f64, 0_f64, 1_f64);
        let normal = Unit::from_value(Vector3::new(0_f64, 0_f64, 1_f64));
        let matrix = Matrix4x4::from_affine_reflection(&normal, &bias);
        let vector = Vector4::new(1_f64, 1_f64, 0.5_f64, 1_f64);
        let expected = Vector4::new(1_f64, 1_f64, 1.5_f64, 1_f64);
        let result = matrix * vector;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_reflection_plane2() {
        // The plane `x = -1`.
        let bias = Point3::new(-1_f64, 0_f64, 0_f64);
        let normal = Unit::from_value(Vector3::new(1_f64, 0_f64, 0_f64));
        let matrix = Matrix4x4::from_affine_reflection(&normal, &bias);
        let vector = Vector4::new(-2_f64, 1_f64, 1_f64, 1_f64);
        let expected = Vector4::new(0_f64, 1_f64, 1_f64, 1_f64);
        let result = matrix * vector;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_reflection_plane3() {
        // The plane `y = 1`.
        let bias = Point3::new(0_f64, 1_f64, 0_f64);
        let normal = Unit::from_value(Vector3::new(0_f64, 1_f64, 0_f64));
        let matrix = Matrix4x4::from_affine_reflection(&normal, &bias);
        let vector = Vector4::new(0_f64, 0_f64, 0_f64, 1_f64);
        let expected = Vector4::new(0_f64, 2_f64, 0_f64, 1_f64);
        let result = matrix * vector;

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix4x4_affine_shear_tests {
    use cglinalg_core::{
        Matrix4x4,
        Point3,
        Unit,
        Vector3,
        Vector4,
    };


    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_xy(shear_factor);
        let vertices = [
            Vector4::new( 1_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32, -1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32, -1_i32, 1_i32),
        ];
        let expected = [
            Vector4::new( 1_i32 + shear_factor,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32 + shear_factor,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32 - shear_factor, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32 - shear_factor, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32 + shear_factor,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32 + shear_factor,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32 - shear_factor, -1_i32, -1_i32, 1_i32),
            Vector4::new( 1_i32 - shear_factor, -1_i32, -1_i32, 1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy_shearing_plane() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_xy(shear_factor);
        let vertices = [
            Vector4::new( 1_i32, 0_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32, 0_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32, 0_i32, -1_i32, 1_i32),
            Vector4::new( 1_i32, 0_i32, -1_i32, 1_i32),
            Vector4::new( 0_i32, 0_i32,  0_i32, 1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_xy_does_not_change_homogeneous_coordinate() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_xy(shear_factor);
        let unit_w = Vector4::unit_w();
        let expected = unit_w;
        let result = matrix * unit_w;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xz() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_xz(shear_factor);
        let vertices = [
            Vector4::new( 1_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32, -1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32, -1_i32, 1_i32),
        ];
        let expected = [
            Vector4::new( 1_i32 + shear_factor,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32 + shear_factor,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32 + shear_factor, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32 + shear_factor, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32 - shear_factor,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32 - shear_factor,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32 - shear_factor, -1_i32, -1_i32, 1_i32),
            Vector4::new( 1_i32 - shear_factor, -1_i32, -1_i32, 1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xz_shearing_plane() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_xz(shear_factor);
        let vertices = [
            Vector4::new( 1_i32,  1_i32, 0_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32, 0_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32, 0_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32, 0_i32, 1_i32),
            Vector4::new( 0_i32,  0_i32, 0_i32, 1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_xz_does_not_change_homogeneous_coordinate() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_xz(shear_factor);
        let unit_w = Vector4::unit_w();
        let expected = unit_w;
        let result = matrix * unit_w;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_yx(shear_factor);
        let vertices = [
            Vector4::new( 1_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32, -1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32, -1_i32, 1_i32),
        ];
        let expected = [
            Vector4::new( 1_i32,  1_i32 + shear_factor,  1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32 - shear_factor,  1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32 - shear_factor,  1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32 + shear_factor,  1_i32, 1_i32),
            Vector4::new( 1_i32,  1_i32 + shear_factor, -1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32 - shear_factor, -1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32 - shear_factor, -1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32 + shear_factor, -1_i32, 1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx_shearing_plane() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_yx(shear_factor);
        let vertices = [
            Vector4::new(0_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(0_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new(0_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(0_i32, -1_i32, -1_i32, 1_i32),
            Vector4::new(0_i32,  0_i32,  0_i32, 1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_yx_does_not_change_homogeneous_coordinate() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_yx(shear_factor);
        let unit_w = Vector4::unit_w();
        let expected = unit_w;
        let result = matrix * unit_w;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yz() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_yz(shear_factor);
        let vertices = [
            Vector4::new( 1_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32, -1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32, -1_i32, 1_i32),
        ];
        let expected = [
            Vector4::new( 1_i32,  1_i32 + shear_factor,  1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32 + shear_factor,  1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32 + shear_factor,  1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32 + shear_factor,  1_i32, 1_i32),
            Vector4::new( 1_i32,  1_i32 - shear_factor, -1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32 - shear_factor, -1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32 - shear_factor, -1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32 - shear_factor, -1_i32, 1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yz_shearing_plane() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_yz(shear_factor);
        let vertices = [
            Vector4::new( 1_i32,  1_i32, 0_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32, 0_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32, 0_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32, 0_i32, 1_i32),
            Vector4::new( 0_i32,  0_i32, 0_i32, 1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_yz_does_not_change_homogeneous_coordinate() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_yz(shear_factor);
        let unit_w = Vector4::unit_w();
        let expected = unit_w;
        let result = matrix * unit_w;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zx() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_zx(shear_factor);
        let vertices = [
            Vector4::new( 1_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32, -1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32, -1_i32, 1_i32),
        ];
        let expected = [
            Vector4::new( 1_i32,  1_i32,  1_i32 + shear_factor, 1_i32),
            Vector4::new(-1_i32,  1_i32,  1_i32 - shear_factor, 1_i32),
            Vector4::new(-1_i32, -1_i32,  1_i32 - shear_factor, 1_i32),
            Vector4::new( 1_i32, -1_i32,  1_i32 + shear_factor, 1_i32),
            Vector4::new( 1_i32,  1_i32, -1_i32 + shear_factor, 1_i32),
            Vector4::new(-1_i32,  1_i32, -1_i32 - shear_factor, 1_i32),
            Vector4::new(-1_i32, -1_i32, -1_i32 - shear_factor, 1_i32),
            Vector4::new( 1_i32, -1_i32, -1_i32 + shear_factor, 1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zx_shearing_plane() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_zx(shear_factor);
        let vertices = [
            Vector4::new(0_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(0_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new(0_i32, -1_i32, -1_i32, 1_i32),
            Vector4::new(0_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(0_i32,  0_i32,  0_i32, 1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_zx_does_not_change_homogeneous_coordinate() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_zx(shear_factor);
        let unit_w = Vector4::unit_w();
        let expected = unit_w;
        let result = matrix * unit_w;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zy() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_zy(shear_factor);
        let vertices = [
            Vector4::new( 1_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32,  1_i32, 1_i32),
            Vector4::new( 1_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32,  1_i32, -1_i32, 1_i32),
            Vector4::new(-1_i32, -1_i32, -1_i32, 1_i32),
            Vector4::new( 1_i32, -1_i32, -1_i32, 1_i32),
        ];
        let expected = [
            Vector4::new( 1_i32,  1_i32,  1_i32 + shear_factor, 1_i32),
            Vector4::new(-1_i32,  1_i32,  1_i32 + shear_factor, 1_i32),
            Vector4::new(-1_i32, -1_i32,  1_i32 - shear_factor, 1_i32),
            Vector4::new( 1_i32, -1_i32,  1_i32 - shear_factor, 1_i32),
            Vector4::new( 1_i32,  1_i32, -1_i32 + shear_factor, 1_i32),
            Vector4::new(-1_i32,  1_i32, -1_i32 + shear_factor, 1_i32),
            Vector4::new(-1_i32, -1_i32, -1_i32 - shear_factor, 1_i32),
            Vector4::new( 1_i32, -1_i32, -1_i32 - shear_factor, 1_i32),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zy_shearing_plane() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_zy(shear_factor);
        let vertices = [
            Vector4::new( 1_i32, 0_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32, 0_i32,  1_i32, 1_i32),
            Vector4::new(-1_i32, 0_i32, -1_i32, 1_i32),
            Vector4::new( 1_i32, 0_i32, -1_i32, 1_i32),
            Vector4::new( 0_i32, 0_i32,  0_i32, 1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_zy_does_not_change_homogeneous_coordinate() {
        let shear_factor = 11_i32;
        let matrix = Matrix4x4::from_affine_shear_zy(shear_factor);
        let unit_w = Vector4::unit_w();
        let expected = unit_w;
        let result = matrix * unit_w;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_from_affine_shear_xy() {
        let shear_factor = 19_f64;
        let origin = Point3::origin();
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let expected = Matrix4x4::from_affine_shear_xy(shear_factor);
        let result = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_from_affine_shear_xz() {
        let shear_factor = 19_f64;
        let origin = Point3::origin();
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let expected = Matrix4x4::from_affine_shear_xz(shear_factor);
        let result = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_from_affine_shear_yx() {
        let shear_factor = 19_f64;
        let origin = Point3::origin();
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let expected = Matrix4x4::from_affine_shear_yx(shear_factor);
        let result = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_from_affine_shear_yz() {
        let shear_factor = 19_f64;
        let origin = Point3::origin();
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let expected = Matrix4x4::from_affine_shear_yz(shear_factor);
        let result = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_from_affine_shear_zx() {
        let shear_factor = 19_f64;
        let origin = Point3::origin();
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_x());
        let expected = Matrix4x4::from_affine_shear_zx(shear_factor);
        let result = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_from_affine_shear_zy() {
        let shear_factor = 19_f64;
        let origin = Point3::origin();
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let expected = Matrix4x4::from_affine_shear_zy(shear_factor);
        let result = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix4x4_affine_shear_coordinate_plane_tests {
    use cglinalg_core::{
        Matrix4x4,
        Point3,
        Unit,
        Vector3,
        Vector4,
    };


    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new( 1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64, -1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64, -1_f64, 1_f64),
        ];
        let expected = [
            Vector4::new( 1_f64 + shear_factor,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64 + shear_factor,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64 - shear_factor, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64 - shear_factor, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64 + shear_factor,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64 + shear_factor,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64 - shear_factor, -1_f64, -1_f64, 1_f64),
            Vector4::new( 1_f64 - shear_factor, -1_f64, -1_f64, 1_f64),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy_shearing_plane() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new( 1_f64, 0_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64, 0_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64, 0_f64, -1_f64, 1_f64),
            Vector4::new( 1_f64, 0_f64, -1_f64, 1_f64),
            Vector4::new( 0_f64, 0_f64,  0_f64, 1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_xy_does_not_change_homogeneous_coordinate() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let unit_w = Vector4::unit_w();
        let expected = unit_w;
        let result = matrix * unit_w;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xz() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new( 1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64, -1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64, -1_f64, 1_f64),
        ];
        let expected = [
            Vector4::new( 1_f64 + shear_factor,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64 + shear_factor,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64 + shear_factor, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64 + shear_factor, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64 - shear_factor,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64 - shear_factor,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64 - shear_factor, -1_f64, -1_f64, 1_f64),
            Vector4::new( 1_f64 - shear_factor, -1_f64, -1_f64, 1_f64),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xz_shearing_plane() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new( 1_f64,  1_f64, 0_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64, 0_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64, 0_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64, 0_f64, 1_f64),
            Vector4::new( 0_f64,  0_f64, 0_f64, 1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_xz_does_not_change_homogeneous_coordinate() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let unit_w = Vector4::unit_w();
        let expected = unit_w;
        let result = matrix * unit_w;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new( 1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64, -1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64, -1_f64, 1_f64),
        ];
        let expected = [
            Vector4::new( 1_f64,  1_f64 + shear_factor,  1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64 - shear_factor,  1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64 - shear_factor,  1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64 + shear_factor,  1_f64, 1_f64),
            Vector4::new( 1_f64,  1_f64 + shear_factor, -1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64 - shear_factor, -1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64 - shear_factor, -1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64 + shear_factor, -1_f64, 1_f64),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx_shearing_plane() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new(0_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(0_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new(0_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(0_f64, -1_f64, -1_f64, 1_f64),
            Vector4::new(0_f64,  0_f64,  0_f64, 1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_yx_does_not_change_homogeneous_coordinate() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let unit_w = Vector4::unit_w();
        let expected = unit_w;
        let result = matrix * unit_w;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yz() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new( 1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64, -1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64, -1_f64, 1_f64),
        ];
        let expected = [
            Vector4::new( 1_f64,  1_f64 + shear_factor,  1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64 + shear_factor,  1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64 + shear_factor,  1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64 + shear_factor,  1_f64, 1_f64),
            Vector4::new( 1_f64,  1_f64 - shear_factor, -1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64 - shear_factor, -1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64 - shear_factor, -1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64 - shear_factor, -1_f64, 1_f64),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yz_shearing_plane() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new( 1_f64,  1_f64, 0_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64, 0_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64, 0_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64, 0_f64, 1_f64),
            Vector4::new( 0_f64,  0_f64, 0_f64, 1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_yz_does_not_change_homogeneous_coordinate() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let unit_w = Vector4::unit_w();
        let expected = unit_w;
        let result = matrix * unit_w;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zx() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_x());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new( 1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64, -1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64, -1_f64, 1_f64),
        ];
        let expected = [
            Vector4::new( 1_f64,  1_f64,  1_f64 + shear_factor, 1_f64),
            Vector4::new(-1_f64,  1_f64,  1_f64 - shear_factor, 1_f64),
            Vector4::new(-1_f64, -1_f64,  1_f64 - shear_factor, 1_f64),
            Vector4::new( 1_f64, -1_f64,  1_f64 + shear_factor, 1_f64),
            Vector4::new( 1_f64,  1_f64, -1_f64 + shear_factor, 1_f64),
            Vector4::new(-1_f64,  1_f64, -1_f64 - shear_factor, 1_f64),
            Vector4::new(-1_f64, -1_f64, -1_f64 - shear_factor, 1_f64),
            Vector4::new( 1_f64, -1_f64, -1_f64 + shear_factor, 1_f64),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zx_shearing_plane() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_x());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new(0_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(0_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new(0_f64, -1_f64, -1_f64, 1_f64),
            Vector4::new(0_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(0_f64,  0_f64,  0_f64, 1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_zx_does_not_change_homogeneous_coordinate() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_x());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let unit_w = Vector4::unit_w();
        let expected = unit_w;
        let result = matrix * unit_w;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zy() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new( 1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64, -1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64, -1_f64, 1_f64),
        ];
        let expected = [
            Vector4::new( 1_f64,  1_f64,  1_f64 + shear_factor, 1_f64),
            Vector4::new(-1_f64,  1_f64,  1_f64 + shear_factor, 1_f64),
            Vector4::new(-1_f64, -1_f64,  1_f64 - shear_factor, 1_f64),
            Vector4::new( 1_f64, -1_f64,  1_f64 - shear_factor, 1_f64),
            Vector4::new( 1_f64,  1_f64, -1_f64 + shear_factor, 1_f64),
            Vector4::new(-1_f64,  1_f64, -1_f64 + shear_factor, 1_f64),
            Vector4::new(-1_f64, -1_f64, -1_f64 - shear_factor, 1_f64),
            Vector4::new( 1_f64, -1_f64, -1_f64 - shear_factor, 1_f64),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zy_shearing_plane() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new( 1_f64, 0_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64, 0_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64, 0_f64, -1_f64, 1_f64),
            Vector4::new( 1_f64, 0_f64, -1_f64, 1_f64),
            Vector4::new( 0_f64, 0_f64,  0_f64, 1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| matrix * v);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_zy_does_not_change_homogeneous_coordinate() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let unit_w = Vector4::unit_w();
        let expected = unit_w;
        let result = matrix * unit_w;

        assert_eq!(result, expected);
    }
}


/// Shearing along the plane `(1 / 2) * x + (1 / 3) * y - z + 1 == 0`
/// with origin `[2, 3, 3]`, direction `[2 / sqrt(17), 3 / sqrt(17), 2 / sqrt(17)]`,
/// and normal `[0, -2 / sqrt(13), 3 / sqrt(13)]`.
#[cfg(test)]
mod matrix4x4_affine_shear_noncoordinate_plane_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix4x4,
        Point3,
        Unit,
        Vector3,
        Vector4,
    };
    use cglinalg_trigonometry::{
        Angle,
        Radians,
    };


    fn shear_factor() -> f64 {
        11_f64
    }

    fn origin() -> Point3<f64> {
        Point3::new(2_f64, 3_f64, 3_f64)
    }

    #[rustfmt::skip]
    fn direction() -> Unit<Vector3<f64>> {
        Unit::from_value(Vector3::new(
            2_f64 / f64::sqrt(17_f64), 
            3_f64 / f64::sqrt(17_f64),
            2_f64 / f64::sqrt(17_f64),
        ))
    }

    #[rustfmt::skip]
    fn normal() -> Unit<Vector3<f64>> {
        Unit::from_value(Vector3::new(
            0_f64, 
           -2_f64 / f64::sqrt(13_f64),
            3_f64 / f64::sqrt(13_f64),
       ))
    }

    fn rotation_angle_x_yz() -> Radians<f64> {
        Radians(f64::atan2(2_f64 / 3_f64, 1_f64))
    }

    fn rotation_angle_z_xy() -> Radians<f64> {
        Radians(f64::atan2(13_f64 / (2_f64 * f64::sqrt(13_f64)), 1_f64))
    }

    fn translation() -> Matrix4x4<f64> {
        Matrix4x4::from_affine_translation(&Vector3::new(0_f64, 0_f64, 1_f64))
    }

    fn translation_inv() -> Matrix4x4<f64> {
        Matrix4x4::from_affine_translation(&Vector3::new(0_f64, 0_f64, -1_f64))
    }

    #[rustfmt::skip]
    fn rotation_x_yz() -> Matrix4x4<f64> {
        Matrix4x4::new(
            1_f64,  0_f64,                     0_f64,                     0_f64,
            0_f64,  f64::sqrt(9_f64 / 13_f64), f64::sqrt(4_f64 / 13_f64), 0_f64,
            0_f64, -f64::sqrt(4_f64 / 13_f64), f64::sqrt(9_f64 / 13_f64), 0_f64,
            0_f64,  0_f64,                     0_f64,                     1_f64,
        )
    }

    #[rustfmt::skip]
    fn rotation_x_yz_inv() -> Matrix4x4<f64> {
        Matrix4x4::new(
            1_f64, 0_f64,                      0_f64,                     0_f64,
            0_f64, f64::sqrt(9_f64 / 13_f64), -f64::sqrt(4_f64 / 13_f64), 0_f64,
            0_f64, f64::sqrt(4_f64 / 13_f64),  f64::sqrt(9_f64 / 13_f64), 0_f64,
            0_f64, 0_f64,                      0_f64,                     1_f64,
        )
    }

    #[rustfmt::skip]
    fn rotation_z_xy() -> Matrix4x4<f64> {
        Matrix4x4::new(
            f64::sqrt(4_f64 / 17_f64),  f64::sqrt(13_f64 / 17_f64), 0_f64, 0_f64,
           -f64::sqrt(13_f64 / 17_f64), f64::sqrt(4_f64 / 17_f64),  0_f64, 0_f64,
            0_f64,                      0_f64,                      1_f64, 0_f64,
            0_f64,                      0_f64,                      0_f64, 1_f64,
       )
    }

    #[rustfmt::skip]
    fn rotation_z_xy_inv() -> Matrix4x4<f64> {
        Matrix4x4::new(
            f64::sqrt(4_f64 / 17_f64),  -f64::sqrt(13_f64 / 17_f64), 0_f64, 0_f64,
            f64::sqrt(13_f64 / 17_f64),  f64::sqrt(4_f64 / 17_f64),  0_f64, 0_f64,
            0_f64,                       0_f64,                      1_f64, 0_f64,
            0_f64,                       0_f64,                      0_f64, 1_f64,
        )
    }


    #[rustfmt::skip]
    fn rotation() -> Matrix4x4<f64> {
        let c0r0 = f64::sqrt(4_f64 / 17_f64);
        let c0r1 = f64::sqrt(9_f64 / 17_f64);
        let c0r2 = f64::sqrt(4_f64 / 17_f64);
        let c0r3 = 0_f64;

        let c1r0 = -f64::sqrt(13_f64 / 17_f64);
        let c1r1 = 6_f64 / f64::sqrt(221_f64);
        let c1r2 = 4_f64 / f64::sqrt(221_f64);
        let c1r3 = 0_f64;

        let c2r0 = 0_f64;
        let c2r1 = -f64::sqrt(4_f64 / 13_f64);
        let c2r2 = f64::sqrt(9_f64 / 13_f64);
        let c2r3 = 0_f64;

        let c3r0 = 0_f64;
        let c3r1 = 0_f64;
        let c3r2 = 0_f64;
        let c3r3 = 1_f64;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3,
        )
    }

    #[rustfmt::skip]
    fn rotation_inv() -> Matrix4x4<f64> {
        let c0r0 = f64::sqrt(4_f64 / 17_f64);
        let c0r1 = -f64::sqrt(13_f64 / 17_f64);
        let c0r2 = 0_f64;
        let c0r3 = 0_f64;

        let c1r0 = f64::sqrt(9_f64 / 17_f64);
        let c1r1 = 6_f64 / f64::sqrt(221_f64);
        let c1r2 = -f64::sqrt(4_f64 / 13_f64);
        let c1r3 = 0_f64;

        let c2r0 = f64::sqrt(4_f64 / 17_f64);
        let c2r1 = 4_f64 / f64::sqrt(221_f64);
        let c2r2 = f64::sqrt(9_f64 / 13_f64);
        let c2r3 = 0_f64;

        let c3r0 = 0_f64;
        let c3r1 = 0_f64;
        let c3r2 = 0_f64;
        let c3r3 = 1_f64;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3,
        )
    }

    #[rustfmt::skip]
    fn shear_matrix_xz() -> Matrix4x4<f64> {
        let shear_factor = shear_factor();

        Matrix4x4::new(
            1_f64,        0_f64, 0_f64, 0_f64,
            0_f64,        1_f64, 0_f64, 0_f64,
            shear_factor, 0_f64, 1_f64, 0_f64,
            0_f64,        0_f64, 0_f64, 1_f64,
        )
    }


    #[test]
    fn test_rotation_angle_x_yz() {
        let rotation_angle_x_yz = rotation_angle_x_yz();

        assert_relative_eq!(
            rotation_angle_x_yz.cos(),
            3_f64 / f64::sqrt(13_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
        assert_relative_eq!(
            rotation_angle_x_yz.sin(),
            2_f64 / f64::sqrt(13_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
    }

    #[test]
    fn test_rotation_angle_z_xy() {
        let rotation_angle_z_xy = rotation_angle_z_xy();

        assert_relative_eq!(
            rotation_angle_z_xy.cos(),
            f64::sqrt(4_f64 / 17_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
        assert_relative_eq!(
            rotation_angle_z_xy.sin(),
            f64::sqrt(13_f64 / 17_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
    }

    #[test]
    fn test_from_affine_shear_translation_inv() {
        let translation = translation();
        let expected = translation_inv();
        let result = translation.try_inverse().unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_rotation_x_yz() {
        let rotation_angle_x_yz = rotation_angle_x_yz();
        let expected = rotation_x_yz();
        let result = Matrix4x4::from_affine_angle_x(rotation_angle_x_yz);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation_x_yz_inv() {
        let rotation_angle_x_yz = rotation_angle_x_yz();
        let expected = rotation_x_yz_inv();
        let result = Matrix4x4::from_affine_angle_x(-rotation_angle_x_yz);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation_z_xy() {
        let rotation_angle_z_xy = rotation_angle_z_xy();
        let expected = rotation_z_xy();
        let result = Matrix4x4::from_affine_angle_z(rotation_angle_z_xy);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation_z_xy_inv() {
        let rotation_angle_z_xy = rotation_angle_z_xy();
        let expected = rotation_z_xy_inv();
        let result = Matrix4x4::from_affine_angle_z(-rotation_angle_z_xy);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation() {
        let expected = rotation();
        let rotation_x_yz = rotation_x_yz();
        let rotation_z_xy = rotation_z_xy();
        let result = rotation_x_yz * rotation_z_xy;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation_inv() {
        let expected = rotation_inv();
        let rotation = rotation();
        let result = rotation.try_inverse().unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_origin_xz() {
        let translation = translation();
        let rotation = rotation();
        let origin = origin();
        let origin_xz = Vector4::new(f64::sqrt(17_f64), 0_f64, 0_f64, 1_f64);
        let expected = origin;
        let result = {
            let _origin = origin.to_vector().extend(1_f64);
            let _origin_xz = translation * rotation * origin_xz;
            Point3::new(_origin_xz[0], _origin_xz[1], _origin_xz[2])
        };

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_direction_xz() {
        let direction = direction();
        let rotation_inv = rotation_inv();
        let expected = Vector4::unit_x();
        let result = {
            let _direction = direction.into_inner().extend(0_f64);
            rotation_inv * _direction
        };

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_normal_xz() {
        let normal = normal();
        let rotation_inv = rotation_inv();
        let expected = Vector4::unit_z();
        let result = {
            let _normal = normal.into_inner().extend(0_f64);
            rotation_inv * _normal
        };

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_coordinates_vertices() {
        let vertices = [
            Vector4::new(
                -f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64)  + f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64)  + f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                -f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64)  - f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64)  - f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                 f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64)  - f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64)  - f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                 f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64)  + f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64)  + f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                -f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64)  + f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64)  + f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                -f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64)  - f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64)  - f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                 f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64)  - f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64)  - f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                 f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64)  + f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64)  + f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
        ];
        let translation = translation();
        let rotation = rotation();
        let vertices_xz = [
            Vector4::new( 1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64,  1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64,  1_f64, 1_f64),
            Vector4::new( 1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64,  1_f64, -1_f64, 1_f64),
            Vector4::new(-1_f64, -1_f64, -1_f64, 1_f64),
            Vector4::new( 1_f64, -1_f64, -1_f64, 1_f64),
        ];
        let result = vertices_xz.map(|v| translation * rotation * v);

        assert_relative_eq!(result, vertices, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_vertices() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector4::new(
                -f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64)  + f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64)  + f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                -f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64)  - f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64)  - f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                 f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64)  - f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64)  - f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                 f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64)  + f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64)  + f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                -f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64)  + f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64)  + f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                -f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64)  - f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64)  - f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                 f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64)  - f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64)  - f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
            Vector4::new(
                 f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64)  + f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64)  + f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
                 1_f64,
            ),
        ];
        let expected = [
            Vector4::new(
                 f64::sqrt(4_f64 / 17_f64) - f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64) * shear_factor,
                -f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64)  + 6_f64 / f64::sqrt(221_f64) + f64::sqrt(9_f64 / 17_f64) * shear_factor,
                 f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64)  + 4_f64 / f64::sqrt(221_f64) + 1_f64 + f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 1_f64,
            ),
            Vector4::new(
                -f64::sqrt(4_f64 / 17_f64) - f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64) * shear_factor,
                -f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64)  + 6_f64 / f64::sqrt(221_f64) + f64::sqrt(9_f64 / 17_f64) * shear_factor,
                 f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64)  + 4_f64 / f64::sqrt(221_f64) + 1_f64 + f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 1_f64,
            ),
            Vector4::new(
                -f64::sqrt(4_f64 / 17_f64) + f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64) * shear_factor,
                -f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64)  - 6_f64 / f64::sqrt(221_f64) + f64::sqrt(9_f64 / 17_f64) * shear_factor,
                 f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64)  - 4_f64 / f64::sqrt(221_f64) + 1_f64 + f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 1_f64,
            ),
            Vector4::new(
                 f64::sqrt(4_f64 / 17_f64) + f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64) * shear_factor,
                -f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64)  - 6_f64 / f64::sqrt(221_f64) + f64::sqrt(9_f64 / 17_f64) * shear_factor,
                 f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64)  - 4_f64 / f64::sqrt(221_f64) + 1_f64 + f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 1_f64,
            ),
            Vector4::new(
                 f64::sqrt(4_f64 / 17_f64) - f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64)  + 6_f64 / f64::sqrt(221_f64) - f64::sqrt(9_f64 / 17_f64) * shear_factor,
                -f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64)  + 4_f64 / f64::sqrt(221_f64) + 1_f64 - f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 1_f64,
            ),
            Vector4::new(
                -f64::sqrt(4_f64 / 17_f64) - f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64)  + 6_f64 / f64::sqrt(221_f64) - f64::sqrt(9_f64 / 17_f64) * shear_factor,
                -f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64)  + 4_f64 / f64::sqrt(221_f64) + 1_f64 - f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 1_f64,
            ),
            Vector4::new(
                -f64::sqrt(4_f64 / 17_f64) + f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64) - f64::sqrt(9_f64 / 17_f64) * shear_factor,
                -f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64 - f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 1_f64,
            ),
            Vector4::new(
                 f64::sqrt(4_f64 / 17_f64) + f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64) - f64::sqrt(9_f64 / 17_f64) * shear_factor,
                -f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64 - f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 1_f64,
            ),
        ];
        let result = vertices.map(|v| matrix * v);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_matrix() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let expected = {
            let c0r0 = 1_f64;
            let c0r1 = 0_f64;
            let c0r2 = 0_f64;
            let c0r3 = 0_f64;

            let c1r0 = (-4_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c1r1 = 1_f64 - (6_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c1r2 = (-4_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c1r3 = 0_f64;

            let c2r0 = (6_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c2r1 = (9_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c2r2 = 1_f64 + (6_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c2r3 = 0_f64;

            let c3r0 = (-6_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c3r1 = (-9_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c3r2 = (-6_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c3r3 = 1_f64;

            Matrix4x4::new(
                c0r0, c0r1, c0r2, c0r3,
                c1r0, c1r1, c1r2, c1r3,
                c2r0, c2r1, c2r2, c2r3,
                c3r0, c3r1, c3r2, c3r3,
            )
        };
        let result = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_matrix_alternative_path() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let translation = translation();
        let translation_inv = translation_inv();
        let rotation = rotation();
        let rotation_inv = rotation_inv();
        let shear_matrix_xz = shear_matrix_xz();
        let expected = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let result = (translation * rotation) * shear_matrix_xz * (rotation_inv * translation_inv);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_does_not_change_homogeneous_coordinate() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vector_in_plane = Vector4::new(0_f64, 0_f64, 1_f64, 1_f64);
        let expected = vector_in_plane;
        let result = matrix * vector_in_plane;

        assert_eq!(result[3], expected[3]);
    }
}


#[cfg(test)]
mod matrix4x4_trace_determinant_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix4x4,
        Point3,
        Unit,
        Vector3,
    };

    fn shear_factor() -> f64 {
        -372203_f64
    }

    #[rustfmt::skip]
    fn direction() -> Unit<Vector3<f64>> {
        Unit::from_value(Vector3::new(
            f64::sqrt(1_f64 / 3_f64), 
            f64::sqrt(1_f64 / 3_f64), 
            f64::sqrt(1_f64 / 3_f64),
        ))
    }

    #[rustfmt::skip]
    fn normal() -> Unit<Vector3<f64>> {
        Unit::from_value(Vector3::new(
             f64::sqrt(1_f64 / 6_f64),
             f64::sqrt(1_f64 / 6_f64),
            -f64::sqrt(4_f64 / 6_f64),
        ))
    }

    fn shear_matrix() -> Matrix4x4<f64> {
        let shear_factor = shear_factor();
        let origin = Point3::new(1_f64, 1_f64, 1_f64);
        let direction = direction();
        let normal = normal();

        Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal)
    }

    #[rustfmt::skip]
    fn expected_matrix() -> Matrix4x4<f64> {
        let shear_factor = shear_factor();

        let c0r0 = 1_f64 + f64::sqrt(1_f64 / 18_f64) * shear_factor;
        let c0r1 = f64::sqrt(1_f64 / 18_f64) * shear_factor;
        let c0r2 = f64::sqrt(1_f64 / 18_f64) * shear_factor;
        let c0r3 = 0_f64;

        let c1r0 = f64::sqrt(1_f64 / 18_f64) * shear_factor;
        let c1r1 = 1_f64 + f64::sqrt(1_f64 / 18_f64) * shear_factor;
        let c1r2 = f64::sqrt(1_f64 / 18_f64) * shear_factor;
        let c1r3 = 0_f64;

        let c2r0 = -f64::sqrt(4_f64 / 18_f64) * shear_factor;
        let c2r1 = -f64::sqrt(4_f64 / 18_f64) * shear_factor;
        let c2r2 = 1_f64 - f64::sqrt(4_f64 / 18_f64) * shear_factor;
        let c2r3 = 0_f64;

        let c3r0 = 0_f64;
        let c3r1 = 0_f64;
        let c3r2 = 0_f64;
        let c3r3 = 1_f64;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3,
        )
    }

    #[test]
    fn test_shear_matrix_direction_normal() {
        let direction = direction();
        let normal = normal();

        assert_relative_eq!(direction.dot(&normal), 0_f64, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_shear_matrix() {
        let expected = expected_matrix();
        let result = shear_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_shear_expected_matrix_trace() {
        let matrix = expected_matrix();
        let expected = 4_f64;
        let result = matrix.trace();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_shear_shear_matrix_trace() {
        let matrix = shear_matrix();
        let expected = 4_f64;
        let result = matrix.trace();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    /// Mathematically, the shearing transformation has a determinant of `1`,
    /// but numerically the shearing transformation is not always well behaved.
    #[test]
    fn test_shear_shear_matrix_determinant() {
        let matrix = shear_matrix();
        let expected = 1.25_f64;
        let result = matrix.determinant();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    /// Mathematically, the shearing transformation has a determinant of `1`,
    /// but numerically the shearing transformation is not always well behaved.
    #[test]
    fn test_shear_expected_matrix_determinant() {
        let matrix = expected_matrix();
        let expected = 0.75_f64;
        let result = matrix.determinant();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}


#[cfg(test)]
mod matrix1x2_tests {
    use cglinalg_core::{
        Matrix1x2,
        Matrix2x2,
        Vector1,
        Vector2,
    };


    #[test]
    fn test_matrix_components1() {
        let matrix = Matrix1x2::new(1_i32, 2_i32);

        assert_eq!(matrix[0][0], 1_i32);
        assert_eq!(matrix[1][0], 2_i32);
    }

    #[test]
    fn test_matrix_components2() {
        let matrix = Matrix1x2::new(1_i32, 2_i32);

        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds1() {
        let matrix = Matrix1x2::new(1_i32, 2_i32);

        assert_eq!(matrix[2][0], matrix[2][0]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds2() {
        let matrix = Matrix1x2::new(1_i32, 2_i32);

        assert_eq!(matrix[0][1], matrix[0][1]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds3() {
        let matrix = Matrix1x2::new(1_i32, 2_i32);

        assert_eq!(matrix[2][1], matrix[2][1]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds4() {
        let matrix = Matrix1x2::new(1_i32, 2_i32);

        assert_eq!(matrix[0][usize::MAX], matrix[0][usize::MAX]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds5() {
        let matrix = Matrix1x2::new(1_i32, 2_i32);

        assert_eq!(matrix[usize::MAX][0], matrix[usize::MAX][0]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds6() {
        let matrix = Matrix1x2::new(1_i32, 2_i32);

        assert_eq!(matrix[usize::MAX][usize::MAX], matrix[usize::MAX][usize::MAX]);
    }

    #[test]
    fn test_matrix_times_identity_equals_matrix() {
        let matrix = Matrix1x2::new(2_i32, 3_i32);
        let identity = Matrix2x2::identity();

        assert_eq!(matrix * identity, matrix);
    }

    #[test]
    fn test_matrix_times_zero_equals_zero() {
        let matrix = Matrix1x2::new(33_i32, 54_i32);
        let zero_matrix2x2 = Matrix2x2::zero();
        let zero_matrix1x2 = Matrix1x2::zero();

        assert_eq!(matrix * zero_matrix2x2, zero_matrix1x2);
    }

    #[test]
    fn test_zero_times_matrix_equals_zero() {
        let matrix = Matrix1x2::new(33_i32, 54_i32);
        let zero = 0_i32;
        let zero_matrix1x2 = Matrix1x2::zero();

        assert_eq!(zero * matrix, zero_matrix1x2);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication1() {
        let matrix1x2 = Matrix1x2::new(2_i32, 3_i32);
        let matrix2x2 = Matrix2x2::new(
            1_i32, 2_i32, 
            3_i32, 4_i32
        );
        let expected = Matrix1x2::new(8_i32, 18_i32);
        let result = matrix1x2 * matrix2x2;
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_multiplication2() {
        let matrix1x2 = Matrix1x2::new(4_i32, 5_i32);
        let vector = Vector2::new(9_i32, 6_i32);
        let expected = Vector1::new(66_i32);
        let result = matrix1x2 * vector;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_scalar_multiplication() {
        let matrix1x2 = Matrix1x2::new(1_i32, 2_i32);
        let scalar = 13_i32;
        let expected = Matrix1x2::new(13_i32, 26_i32);
        let result = matrix1x2 * scalar;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_matrix_multiplication() {
        let matrix1x2 = Matrix1x2::new(1_i32, 2_i32);
        let scalar = 13_i32;
        let expected = Matrix1x2::new(13_i32, 26_i32);
        let result = scalar * matrix1x2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero_matrix1x2 = Matrix1x2::zero();
        let matrix = Matrix1x2::new(3684_i32, 42746_i32);

        assert_eq!(matrix + zero_matrix1x2, matrix);
    }

    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero_matrix1x2 = Matrix1x2::zero();
        let matrix = Matrix1x2::new(3684_i32, 42746_i32);

        assert_eq!(zero_matrix1x2 + matrix, matrix);
    }

    #[test]
    fn test_addition() {
        let matrix1 = Matrix1x2::new(23_i32, 76_i32);
        let matrix2 = Matrix1x2::new(1_i32, 5_i32);
        let expected = Matrix1x2::new(24_i32, 81_i32);
        let result = matrix1 + matrix2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction() {
        let matrix1 = Matrix1x2::new(3_i32, 6_i32);
        let matrix2 = Matrix1x2::new(1_i32, 15_i32);
        let expected = Matrix1x2::new(2_i32, -9_i32);
        let result = matrix1 - matrix2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_minus_matrix_is_zero() {
        let matrix = Matrix1x2::new(3_i32, 6_i32);
        let zero_matrix1x2 = Matrix1x2::zero();

        assert_eq!(matrix - matrix, zero_matrix1x2);
    }
}

#[cfg(test)]
mod matrix1x3_tests {
    use cglinalg_core::{
        Matrix1x3,
        Matrix3x3,
        Vector1,
        Vector3,
    };


    #[test]
    fn test_matrix_components1() {
        let matrix = Matrix1x3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(matrix[0][0], 1_i32);
        assert_eq!(matrix[1][0], 2_i32);
        assert_eq!(matrix[2][0], 3_i32);
    }

    #[test]
    fn test_matrix_components2() {
        let matrix = Matrix1x3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c2r0, matrix[2][0]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds1() {
        let matrix = Matrix1x3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(matrix[3][0], matrix[3][0]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds2() {
        let matrix = Matrix1x3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(matrix[0][1], matrix[0][1]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds3() {
        let matrix = Matrix1x3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(matrix[3][1], matrix[3][1]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds4() {
        let matrix = Matrix1x3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(matrix[0][usize::MAX], matrix[0][usize::MAX]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds5() {
        let matrix = Matrix1x3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(matrix[usize::MAX][0], matrix[usize::MAX][0]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds6() {
        let matrix = Matrix1x3::new(1_i32, 2_i32, 3_i32);

        assert_eq!(matrix[usize::MAX][usize::MAX], matrix[usize::MAX][usize::MAX]);
    }

    #[test]
    fn test_matrix_times_identity_equals_matrix() {
        let matrix = Matrix1x3::new(2_i32, 3_i32, 4_i32);
        let identity = Matrix3x3::identity();

        assert_eq!(matrix * identity, matrix);
    }

    #[test]
    fn test_matrix_times_zero_equals_zero() {
        let matrix = Matrix1x3::new(33_i32, 54_i32, 19_i32);
        let zero_matrix3x3 = Matrix3x3::zero();
        let zero_matrix1x3 = Matrix1x3::zero();

        assert_eq!(matrix * zero_matrix3x3, zero_matrix1x3);
    }

    #[test]
    fn test_zero_times_matrix_equals_zero() {
        let matrix = Matrix1x3::new(33_i32, 54_i32, 19_i32);
        let zero = 0_i32;
        let zero_matrix1x3 = Matrix1x3::zero();

        assert_eq!(zero * matrix, zero_matrix1x3);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication1() {
        let matrix1x3 = Matrix1x3::new(2_i32, 3_i32, 4_i32);
        let matrix3x3 = Matrix3x3::new(
            1_i32, 2_i32, 3_i32, 
            4_i32, 5_i32, 6_i32, 
            7_i32, 8_i32, 9_i32
        );
        let expected = Matrix1x3::new(20_i32, 47_i32, 74_i32);
        let result = matrix1x3 * matrix3x3;
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_multiplication2() {
        let matrix1x3 = Matrix1x3::new(4_i32, 5_i32, 6_i32);
        let vector = Vector3::new(9_i32, 6_i32, -12_i32);
        let expected = Vector1::new(-6_i32);
        let result = matrix1x3 * vector;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_scalar_multiplication() {
        let matrix1x3 = Matrix1x3::new(1_i32, 2_i32, 3_i32);
        let scalar = 13_i32;
        let expected = Matrix1x3::new(13_i32, 26_i32, 39_i32);
        let result = matrix1x3 * scalar;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_matrix_multiplication() {
        let matrix1x3 = Matrix1x3::new(1_i32, 2_i32, 3_i32);
        let scalar = 13_i32;
        let expected = Matrix1x3::new(13_i32, 26_i32, 39_i32);
        let result = scalar * matrix1x3;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero_matrix1x3 = Matrix1x3::zero();
        let matrix = Matrix1x3::new(3684_i32, 42746_i32, 345_i32);

        assert_eq!(matrix + zero_matrix1x3, matrix);
    }

    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero_matrix1x3 = Matrix1x3::zero();
        let matrix = Matrix1x3::new(3684_i32, 42746_i32, 345_i32);

        assert_eq!(zero_matrix1x3 + matrix, matrix);
    }

    #[test]
    fn test_addition() {
        let matrix1 = Matrix1x3::new(23_i32, 76_i32, 89_i32);
        let matrix2 = Matrix1x3::new(1_i32, 5_i32, 9_i32);
        let expected = Matrix1x3::new(24_i32, 81_i32, 98_i32);
        let result = matrix1 + matrix2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction() {
        let matrix1 = Matrix1x3::new(3_i32, 6_i32, 9_i32);
        let matrix2 = Matrix1x3::new(1_i32, 15_i32, 29_i32);
        let expected = Matrix1x3::new(2_i32, -9_i32, -20_i32);
        let result = matrix1 - matrix2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_minus_matrix_is_zero() {
        let matrix = Matrix1x3::new(3_i32, 6_i32, 9_i32);
        let zero_matrix1x3 = Matrix1x3::zero();

        assert_eq!(matrix - matrix, zero_matrix1x3);
    }
}

#[cfg(test)]
mod matrix1x4_tests {
    use cglinalg_core::{
        Matrix1x4,
        Matrix4x4,
        Vector1,
        Vector4,
    };


    #[test]
    fn test_matrix_components1() {
        let matrix = Matrix1x4::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(matrix[0][0], 1_i32);
        assert_eq!(matrix[1][0], 2_i32);
        assert_eq!(matrix[2][0], 3_i32);
        assert_eq!(matrix[3][0], 4_i32);
    }

    #[test]
    fn test_matrix_components2() {
        let matrix = Matrix1x4::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c2r0, matrix[2][0]);
        assert_eq!(matrix.c3r0, matrix[3][0]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds1() {
        let matrix = Matrix1x4::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(matrix[4][0], matrix[4][0]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds2() {
        let matrix = Matrix1x4::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(matrix[0][1], matrix[0][1]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds3() {
        let matrix = Matrix1x4::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(matrix[4][1], matrix[4][1]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds4() {
        let matrix = Matrix1x4::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(matrix[0][usize::MAX], matrix[0][usize::MAX]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds5() {
        let matrix = Matrix1x4::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(matrix[usize::MAX][0], matrix[usize::MAX][0]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds6() {
        let matrix = Matrix1x4::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(matrix[usize::MAX][usize::MAX], matrix[usize::MAX][usize::MAX]);
    }

    #[test]
    fn test_matrix_times_identity_equals_matrix() {
        let matrix = Matrix1x4::new(2_i32, 3_i32, 4_i32, 5_i32);
        let identity = Matrix4x4::identity();

        assert_eq!(matrix * identity, matrix);
    }

    #[test]
    fn test_matrix_times_zero_equals_zero() {
        let matrix = Matrix1x4::new(33_i32, 54_i32, 19_i32, 5_i32);
        let zero_matrix4x4 = Matrix4x4::zero();
        let zero_matrix1x4 = Matrix1x4::zero();

        assert_eq!(matrix * zero_matrix4x4, zero_matrix1x4);
    }

    #[test]
    fn test_zero_times_matrix_equals_zero() {
        let matrix = Matrix1x4::new(33_i32, 54_i32, 19_i32, 5_i32);
        let zero = 0_i32;
        let zero_matrix1x4 = Matrix1x4::zero();

        assert_eq!(zero * matrix, zero_matrix1x4);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication1() {
        let matrix1x4 = Matrix1x4::new(2_i32, 3_i32, 4_i32, 5_i32);
        let matrix4x4 = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32, 
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32
        );
        let expected = Matrix1x4::new(40_i32, 96_i32, 152_i32, 208_i32);
        let result = matrix1x4 * matrix4x4;
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_multiplication2() {
        let matrix1x4 = Matrix1x4::new(4_i32, 5_i32, 6_i32, 7_i32);
        let vector = Vector4::new(9_i32, 6_i32, -12_i32, -72_i32);
        let expected = Vector1::new(-510_i32);
        let result = matrix1x4 * vector;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_scalar_multiplication() {
        let matrix1x4 = Matrix1x4::new(1_i32, 2_i32, 3_i32, 4_i32);
        let scalar = 13_i32;
        let expected = Matrix1x4::new(13_i32, 26_i32, 39_i32, 52_i32);
        let result = matrix1x4 * scalar;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_matrix_multiplication() {
        let matrix1x4 = Matrix1x4::new(1_i32, 2_i32, 3_i32, 4_i32);
        let scalar = 13_i32;
        let expected = Matrix1x4::new(13_i32, 26_i32, 39_i32, 52_i32);
        let result = scalar * matrix1x4;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero_matrix1x4 = Matrix1x4::zero();
        let matrix = Matrix1x4::new(3684_i32, 42746_i32, 345_i32, 546_i32);

        assert_eq!(matrix + zero_matrix1x4, matrix);
    }

    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero_matrix1x4 = Matrix1x4::zero();
        let matrix = Matrix1x4::new(3684_i32, 42746_i32, 345_i32, 546_i32);

        assert_eq!(zero_matrix1x4 + matrix, matrix);
    }

    #[test]
    fn test_addition() {
        let matrix1 = Matrix1x4::new(23_i32, 76_i32, 89_i32, 34_i32);
        let matrix2 = Matrix1x4::new(1_i32, 5_i32, 9_i32, 13_i32);
        let expected = Matrix1x4::new(24_i32, 81_i32, 98_i32, 47_i32);
        let result = matrix1 + matrix2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction() {
        let matrix1 = Matrix1x4::new(3_i32, 6_i32, 9_i32, 12_i32);
        let matrix2 = Matrix1x4::new(1_i32, 15_i32, 29_i32, 6_i32);
        let expected = Matrix1x4::new(2_i32, -9_i32, -20_i32, 6_i32);
        let result = matrix1 - matrix2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_minus_matrix_is_zero() {
        let matrix = Matrix1x4::new(3_i32, 6_i32, 9_i32, 12_i32);
        let zero_matrix1x4 = Matrix1x4::zero();

        assert_eq!(matrix - matrix, zero_matrix1x4);
    }
}


#[cfg(test)]
mod matrix2x3_tests {
    use cglinalg_core::{
        Matrix2x2,
        Matrix2x3,
        Matrix3x2,
        Matrix3x3,
        Vector2,
        Vector3,
    };


    #[rustfmt::skip]
    #[test]
    fn test_matrix_components1() {
        let matrix = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
        );

        assert_eq!(matrix[0][0], 1_i32);
        assert_eq!(matrix[0][1], 2_i32);
        assert_eq!(matrix[1][0], 3_i32);
        assert_eq!(matrix[1][1], 4_i32);
        assert_eq!(matrix[2][0], 5_i32);
        assert_eq!(matrix[2][1], 6_i32);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_components2() {
        let matrix = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
        );

        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c0r1, matrix[0][1]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c1r1, matrix[1][1]);
        assert_eq!(matrix.c2r0, matrix[2][0]);
        assert_eq!(matrix.c2r1, matrix[2][1]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds1() {
        let matrix = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
        );

        assert_eq!(matrix[3][0], matrix[3][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds2() {
        let matrix = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
        );

        assert_eq!(matrix[0][2], matrix[0][2]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds3() {
        let matrix = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
        );

        assert_eq!(matrix[3][2], matrix[3][2]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds4() {
        let matrix = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
        );

        assert_eq!(matrix[0][usize::MAX], matrix[0][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds5() {
        let matrix = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
        );

        assert_eq!(matrix[usize::MAX][0], matrix[usize::MAX][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds6() {
        let matrix = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
        );

        assert_eq!(matrix[usize::MAX][usize::MAX], matrix[usize::MAX][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_matrix() {
        let matrix = Matrix2x3::new(
            2_i32, 3_i32,
            4_i32, 5_i32,
            6_i32, 7_i32,
        );
        let identity = Matrix3x3::identity();

        assert_eq!(matrix * identity, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_zero_equals_zero() {
        let matrix = Matrix2x3::new(
            33_i32,  54_i32,
            19_i32,  5_i32,
            793_i32, 23_i32,
        );
        let zero_matrix3x3 = Matrix3x3::zero();
        let zero_matrix2x3 = Matrix2x3::zero();

        assert_eq!(matrix * zero_matrix3x3, zero_matrix2x3);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_times_matrix_equals_zero() {
        let matrix = Matrix2x3::new(
            33_i32,  54_i32,
            19_i32,  5_i32,
            234_i32, 98_i32,
        );
        let zero = 0_i32;
        let zero_matrix2x3 = Matrix2x3::zero();

        assert_eq!(zero * matrix, zero_matrix2x3);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication1() {
        let matrix2x3 = Matrix2x3::new(
            2_i32, 3_i32,
            4_i32, 5_i32,
            6_i32, 7_i32,
        );
        let matrix3x3 = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 8_i32,
            7_i32, 8_i32, 9_i32,
        );
        let expected = Matrix2x3::new(
            28_i32,  34_i32,
            76_i32,  93_i32,
            100_i32, 124_i32,
        );
        let result = matrix2x3 * matrix3x3;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication2() {
        let matrix2x3 = Matrix2x3::new(
            2_i32, 3_i32,
            4_i32, 5_i32,
            6_i32, 7_i32,
        );
        let matrix3x2 = Matrix3x2::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 8_i32,
        );
        let expected = Matrix2x2::new(
            28_i32,  34_i32,
            76_i32,  93_i32,
        );
        let result = matrix2x3 * matrix3x2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication3() {
        let matrix2x3 = Matrix2x3::new(
            4_i32, 5_i32,
            6_i32, 7_i32,
            8_i32, 9_i32,
        );
        let vector = Vector3::new(9_i32, 6_i32, -12_i32);
        let expected = Vector2::new(-24_i32, -21_i32);
        let result = matrix2x3 * vector;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_scalar_multiplication() {
        let matrix2x3 = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 7_i32,
        );
        let scalar = 13_i32;
        let expected = Matrix2x3::new(
            13_i32, 26_i32,
            39_i32, 52_i32,
            65_i32, 91_i32,
        );
        let result = matrix2x3 * scalar;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_matrix_multiplication() {
        let matrix2x3 = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 7_i32,
        );
        let scalar = 13_i32;
        let expected = Matrix2x3::new(
            13_i32, 26_i32,
            39_i32, 52_i32,
            65_i32, 91_i32,
        );
        let result = scalar * matrix2x3;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero_matrix2x3 = Matrix2x3::zero();
        let matrix = Matrix2x3::new(
            3684_i32, 42746_i32,
            345_i32,  546_i32,
            76_i32,   167_i32,
        );

        assert_eq!(matrix + zero_matrix2x3, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero_matrix2x3 = Matrix2x3::zero();
        let matrix = Matrix2x3::new(
            3684_i32, 42746_i32,
            345_i32,  546_i32,
            76_i32,   167_i32,
        );

        assert_eq!(zero_matrix2x3 + matrix, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_addition() {
        let matrix1 = Matrix2x3::new(
            23_i32,  76_i32,
            89_i32,  34_i32,
            324_i32, 75_i32,
        );
        let matrix2 = Matrix2x3::new(
            1_i32,  5_i32,
            9_i32,  13_i32,
            17_i32, 21_i32,
        );
        let expected = Matrix2x3::new(
            24_i32,  81_i32,
            98_i32,  47_i32,
            341_i32, 96_i32,
        );
        let result = matrix1 + matrix2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_subtraction() {
        let matrix1 = Matrix2x3::new(
            3_i32,  6_i32,
            9_i32,  12_i32,
            15_i32, 18_i32,
        );
        let matrix2 = Matrix2x3::new(
            1_i32,   15_i32,
            29_i32,  6_i32,
            234_i32, 93_i32,
        );
        let expected = Matrix2x3::new(
             2_i32,   -9_i32,
            -20_i32,   6_i32,
            -219_i32, -75_i32,
        );
        let result = matrix1 - matrix2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_minus_matrix_is_zero() {
        let matrix = Matrix2x3::new(
            3_i32,  6_i32,
            9_i32,  12_i32,
            15_i32, 18_i32,
        );
        let zero_matrix2x3 = Matrix2x3::zero();

        assert_eq!(matrix - matrix, zero_matrix2x3);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose() {
        let matrix = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
        );
        let expected = Matrix3x2::new(
            1_i32, 3_i32, 5_i32,
            2_i32, 4_i32, 6_i32,
        );
        let result = matrix.transpose();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector2::new(1_i32, 2_i32);
        let c1 = Vector2::new(3_i32, 4_i32);
        let c2 = Vector2::new(5_i32, 6_i32);
        let columns = [c0, c1, c2];
        let expected = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
        );
        let result = Matrix2x3::from_columns(&columns);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_rows() {
        let r0 = Vector3::new(1_i32, 2_i32, 3_i32);
        let r1 = Vector3::new(4_i32, 5_i32, 6_i32);
        let rows = [r0, r1];
        let expected = Matrix2x3::new(
            1_i32, 4_i32,
            2_i32, 5_i32,
            3_i32, 6_i32,
        );
        let result = Matrix2x3::from_rows(&rows);

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix3x2_tests {
    use cglinalg_core::{
        Matrix2x2,
        Matrix2x3,
        Matrix3x2,
        Matrix3x3,
        Vector2,
        Vector3,
    };


    #[rustfmt::skip]
    #[test]
    fn test_matrix_components1() {
        let matrix = Matrix3x2::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
        );

        assert_eq!(matrix[0][0], 1_i32);
        assert_eq!(matrix[0][1], 2_i32);
        assert_eq!(matrix[0][2], 3_i32);
        assert_eq!(matrix[1][0], 4_i32);
        assert_eq!(matrix[1][1], 5_i32);
        assert_eq!(matrix[1][2], 6_i32);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_components2() {
        let matrix = Matrix3x2::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
        );

        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c0r1, matrix[0][1]);
        assert_eq!(matrix.c0r2, matrix[0][2]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c1r1, matrix[1][1]);
        assert_eq!(matrix.c1r2, matrix[1][2]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds1() {
        let matrix = Matrix3x2::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
        );

        assert_eq!(matrix[2][0], matrix[2][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds2() {
        let matrix = Matrix3x2::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
        );

        assert_eq!(matrix[0][3], matrix[0][3]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds3() {
        let matrix = Matrix3x2::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
        );

        assert_eq!(matrix[2][3], matrix[2][3]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds4() {
        let matrix = Matrix3x2::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
        );

        assert_eq!(matrix[0][usize::MAX], matrix[0][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds5() {
        let matrix = Matrix3x2::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
        );

        assert_eq!(matrix[usize::MAX][0], matrix[usize::MAX][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds6() {
        let matrix = Matrix3x2::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
        );

        assert_eq!(matrix[usize::MAX][usize::MAX], matrix[usize::MAX][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_matrix() {
        let matrix = Matrix3x2::new(
            2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32,
        );
        let identity = Matrix2x2::identity();

        assert_eq!(matrix * identity, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_zero_equals_zero() {
        let matrix = Matrix3x2::new(
            33_i32, 54_i32,  19_i32,
            5_i32,  793_i32, 23_i32,
        );
        let zero_matrix2x2 = Matrix2x2::zero();
        let zero_matrix3x2 = Matrix3x2::zero();

        assert_eq!(matrix * zero_matrix2x2, zero_matrix3x2);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_times_matrix_equals_zero() {
        let matrix = Matrix3x2::new(
            33_i32, 54_i32,  19_i32,
            5_i32,  234_i32, 98_i32,
        );
        let zero = 0_i32;
        let zero_matrix3x2 = Matrix3x2::zero();

        assert_eq!(zero * matrix, zero_matrix3x2);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication1() {
        let matrix3x2 = Matrix3x2::new(
            2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32,
        );
        let matrix2x2 = Matrix2x2::new(
            1_i32, 2_i32,
            4_i32, 5_i32,
        );
        let expected = Matrix3x2::new(
            12_i32, 15_i32, 18_i32,
            33_i32, 42_i32, 51_i32,
        );
        let result = matrix3x2 * matrix2x2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication2() {
        let matrix3x2 = Matrix3x2::new(
            2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32,
        );
        let matrix2x3 = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 8_i32,
        );
        let expected = Matrix3x3::new(
            12_i32, 15_i32, 18_i32,
            26_i32, 33_i32, 40_i32,
            50_i32, 63_i32, 76_i32,
        );
        let result = matrix3x2 * matrix2x3;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication3() {
        let matrix3x2 = Matrix3x2::new(
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );
        let vector = Vector2::new(9_i32, -6_i32);
        let expected = Vector3::new(-6_i32, -3_i32, 0_i32);
        let result = matrix3x2 * vector;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_scalar_multiplication() {
        let matrix3x2 = Matrix3x2::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 7_i32,
        );
        let scalar = 13_i32;
        let expected = Matrix3x2::new(
            13_i32, 26_i32, 39_i32,
            52_i32, 65_i32, 91_i32,
        );
        let result = matrix3x2 * scalar;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_matrix_multiplication() {
        let matrix3x2 = Matrix3x2::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 7_i32,
        );
        let scalar = 13_i32;
        let expected = Matrix3x2::new(
            13_i32, 26_i32, 39_i32,
            52_i32, 65_i32, 91_i32,
        );
        let result = scalar * matrix3x2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero_matrix3x2 = Matrix3x2::zero();
        let matrix = Matrix3x2::new(
            3684_i32, 42746_i32, 345_i32,
            546_i32,  76_i32,    167_i32,
        );

        assert_eq!(matrix + zero_matrix3x2, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero_matrix3x2 = Matrix3x2::zero();
        let matrix = Matrix3x2::new(
            3684_i32, 42746_i32, 345_i32,
            546_i32,  76_i32,    167_i32,
        );

        assert_eq!(zero_matrix3x2 + matrix, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_addition() {
        let matrix1 = Matrix3x2::new(
            23_i32, 76_i32,  89_i32,
            34_i32, 324_i32, 75_i32,
        );
        let matrix2 = Matrix3x2::new(
            1_i32,  5_i32,  9_i32,
            13_i32, 17_i32, 21_i32,
        );
        let expected = Matrix3x2::new(
            24_i32, 81_i32, 98_i32,
            47_i32, 341_i32, 96_i32,
        );
        let result = matrix1 + matrix2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_subtraction() {
        let matrix1 = Matrix3x2::new(
            3_i32,  6_i32,  9_i32,
            12_i32, 15_i32, 18_i32,
        );
        let matrix2 = Matrix3x2::new(
            1_i32, 15_i32,  29_i32,
            6_i32, 234_i32, 93_i32,
        );
        let expected = Matrix3x2::new(
             2_i32, -9_i32,   -20_i32,
             6_i32, -219_i32, -75_i32,
        );
        let result = matrix1 - matrix2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_minus_matrix_is_zero() {
        let matrix = Matrix3x2::new(
            3_i32,  6_i32,  9_i32,
            12_i32, 15_i32, 18_i32,
        );
        let zero_matrix3x2 = Matrix3x2::zero();

        assert_eq!(matrix - matrix, zero_matrix3x2);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose() {
        let matrix = Matrix3x2::new(
            1_i32, 3_i32, 5_i32,
            2_i32, 4_i32, 6_i32,
        );
        let expected = Matrix2x3::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
        );
        let result = matrix.transpose();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector3::new(1_i32, 2_i32, 3_i32);
        let c1 = Vector3::new(4_i32, 5_i32, 6_i32);
        let columns = [c0, c1];
        let expected = Matrix3x2::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
        );
        let result = Matrix3x2::from_columns(&columns);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_rows() {
        let r0 = Vector2::new(1_i32, 2_i32);
        let r1 = Vector2::new(3_i32, 4_i32);
        let r2 = Vector2::new(5_i32, 6_i32);
        let rows = [r0, r1, r2];
        let expected = Matrix3x2::new(
            1_i32, 3_i32, 5_i32,
            2_i32, 4_i32, 6_i32,
        );
        let result = Matrix3x2::from_rows(&rows);

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix2x4_tests {
    use cglinalg_core::{
        Matrix2x2,
        Matrix2x4,
        Matrix4x2,
        Matrix4x4,
        Vector2,
        Vector4,
    };


    #[rustfmt::skip]
    #[test]
    fn test_matrix_components1() {
        let matrix = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
            7_i32, 8_i32,
        );

        assert_eq!(matrix[0][0], 1_i32);
        assert_eq!(matrix[0][1], 2_i32);
        assert_eq!(matrix[1][0], 3_i32);
        assert_eq!(matrix[1][1], 4_i32);
        assert_eq!(matrix[2][0], 5_i32);
        assert_eq!(matrix[2][1], 6_i32);
        assert_eq!(matrix[3][0], 7_i32);
        assert_eq!(matrix[3][1], 8_i32);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_components2() {
        let matrix = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
            7_i32, 8_i32,
        );

        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c0r1, matrix[0][1]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c1r1, matrix[1][1]);
        assert_eq!(matrix.c2r0, matrix[2][0]);
        assert_eq!(matrix.c2r1, matrix[2][1]);
        assert_eq!(matrix.c3r0, matrix[3][0]);
        assert_eq!(matrix.c3r1, matrix[3][1]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds1() {
        let matrix = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
            7_i32, 8_i32,
        );

        assert_eq!(matrix[4][0], matrix[4][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds2() {
        let matrix = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
            7_i32, 8_i32,
        );

        assert_eq!(matrix[0][2], matrix[0][2]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds3() {
        let matrix = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
            7_i32, 8_i32,
        );

        assert_eq!(matrix[4][2], matrix[4][2]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds4() {
        let matrix = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
            7_i32, 8_i32,
        );

        assert_eq!(matrix[0][usize::MAX], matrix[0][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds5() {
        let matrix = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
            7_i32, 8_i32,
        );

        assert_eq!(matrix[usize::MAX][0], matrix[usize::MAX][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds6() {
        let matrix = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
            7_i32, 8_i32,
        );

        assert_eq!(matrix[usize::MAX][usize::MAX], matrix[usize::MAX][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_matrix() {
        let matrix = Matrix2x4::new(
            2_i32, 3_i32,
            4_i32, 5_i32,
            6_i32, 7_i32,
            8_i32, 9_i32,
        );
        let identity = Matrix4x4::identity();

        assert_eq!(matrix * identity, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_zero_equals_zero() {
        let matrix = Matrix2x4::new(
            33_i32,  54_i32,
            19_i32,  5_i32,
            793_i32, 23_i32,
            49_i32,  11_i32,
        );
        let zero_matrix4x4 = Matrix4x4::zero();
        let zero_matrix2x4 = Matrix2x4::zero();

        assert_eq!(matrix * zero_matrix4x4, zero_matrix2x4);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_matrix_times_matrix_equals_zero() {
        let matrix = Matrix2x4::new(
            33_i32,  54_i32,
            19_i32,  5_i32,
            793_i32, 23_i32,
            49_i32,  11_i32,
        );
        let zero_matrix2x2 = Matrix2x2::zero();
        let zero_matrix2x4 = Matrix2x4::zero();

        assert_eq!(zero_matrix2x2 * matrix, zero_matrix2x4);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_times_matrix_equals_zero() {
        let matrix = Matrix2x4::new(
            33_i32,  54_i32,
            19_i32,  5_i32,
            234_i32, 98_i32,
            64_i32,  28_i32,
        );
        let zero = 0_i32;
        let zero_matrix2x4 = Matrix2x4::zero();

        assert_eq!(zero * matrix, zero_matrix2x4);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication1() {
        let matrix2x4 = Matrix2x4::new(
            2_i32, 3_i32,
            4_i32, 5_i32,
            6_i32, 7_i32,
            8_i32, 9_i32,
        );
        let matrix4x4 = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );
        let expected = Matrix2x4::new(
            60_i32,  70_i32,
            140_i32, 166_i32,
            220_i32, 262_i32,
            300_i32, 358_i32,
        );
        let result = matrix2x4 * matrix4x4;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication2() {
        let matrix2x4 = Matrix2x4::new(
            4_i32,  5_i32,
            6_i32,  7_i32,
            8_i32,  9_i32,
            10_i32, 11_i32,
        );
        let vector = Vector4::new(9_i32, 6_i32, -12_i32, -24_i32);
        let expected = Vector2::new(-264_i32, -285_i32);
        let result = matrix2x4 * vector;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication3() {
        let matrix2x4 = Matrix2x4::new(
            2_i32, 3_i32,
            4_i32, 5_i32,
            6_i32, 7_i32,
            8_i32, 9_i32,
        );
        let vector = Vector4::new(9_i32, -6_i32, 12_i32, 4_i32);
        let expected = Vector2::new(98_i32, 117_i32);
        let result = matrix2x4 * vector;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_scalar_multiplication() {
        let matrix2x4 = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 7_i32,
            8_i32, 9_i32,
        );
        let scalar = 13_i32;
        let expected = Matrix2x4::new(
            13_i32,  26_i32,
            39_i32,  52_i32,
            65_i32,  91_i32,
            104_i32, 117_i32,
        );
        let result = matrix2x4 * scalar;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_matrix_multiplication() {
        let matrix2x4 = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 7_i32,
            8_i32, 9_i32,
        );
        let scalar = 13_i32;
        let expected = Matrix2x4::new(
            13_i32,  26_i32,
            39_i32,  52_i32,
            65_i32,  91_i32,
            104_i32, 117_i32,
        );
        let result = scalar * matrix2x4;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero_matrix2x4 = Matrix2x4::zero();
        let matrix = Matrix2x4::new(
            3684_i32, 42746_i32,
            345_i32,  546_i32,
            76_i32,   167_i32,
            415_i32,  251_i32,
        );

        assert_eq!(matrix + zero_matrix2x4, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero_matrix2x4 = Matrix2x4::zero();
        let matrix = Matrix2x4::new(
            3684_i32, 42746_i32,
            345_i32,  546_i32,
            76_i32,   167_i32,
            415_i32,  251_i32,
        );

        assert_eq!(zero_matrix2x4 + matrix, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_addition() {
        let matrix1 = Matrix2x4::new(
            23_i32,  76_i32,
            89_i32,  34_i32,
            324_i32, 75_i32,
            614_i32, 15_i32,
        );
        let matrix2 = Matrix2x4::new(
            1_i32,  5_i32,
            9_i32,  13_i32,
            17_i32, 21_i32,
            87_i32, 41_i32,
        );
        let expected = Matrix2x4::new(
            24_i32,  81_i32,
            98_i32,  47_i32,
            341_i32, 96_i32,
            701_i32, 56_i32,
        );
        let result = matrix1 + matrix2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_subtraction() {
        let matrix1 = Matrix2x4::new(
            3_i32,  6_i32,
            9_i32,  12_i32,
            15_i32, 18_i32,
            21_i32, 24_i32,
        );
        let matrix2 = Matrix2x4::new(
            1_i32,   15_i32,
            29_i32,  6_i32,
            234_i32, 93_i32,
            93_i32,  7_i32,
        );
        let expected = Matrix2x4::new(
             2_i32,   -9_i32,
            -20_i32,   6_i32,
            -219_i32, -75_i32,
            -72_i32,   17_i32,
        );
        let result = matrix1 - matrix2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_minus_matrix_is_zero() {
        let matrix = Matrix2x4::new(
            3_i32,  6_i32,
            9_i32,  12_i32,
            15_i32, 18_i32,
            21_i32, 24_i32,
        );
        let zero_matrix2x4 = Matrix2x4::zero();

        assert_eq!(matrix - matrix, zero_matrix2x4);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose() {
        let matrix = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
            7_i32, 8_i32,
        );
        let expected = Matrix4x2::new(
            1_i32, 3_i32, 5_i32, 7_i32,
            2_i32, 4_i32, 6_i32, 8_i32,
        );
        let result = matrix.transpose();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector2::new(1_i32, 2_i32);
        let c1 = Vector2::new(3_i32, 4_i32);
        let c2 = Vector2::new(5_i32, 6_i32);
        let c3 = Vector2::new(7_i32, 8_i32);
        let columns = [c0, c1, c2, c3];
        let expected = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 6_i32,
            7_i32, 8_i32,
        );
        let result = Matrix2x4::from_columns(&columns);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_rows() {
        let r0 = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
        let r1 = Vector4::new(5_i32, 6_i32, 7_i32, 8_i32);
        let rows = [r0, r1];
        let expected = Matrix2x4::new(
            1_i32, 5_i32,
            2_i32, 6_i32,
            3_i32, 7_i32,
            4_i32, 8_i32,
        );
        let result = Matrix2x4::from_rows(&rows);

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix4x2_tests {
    use cglinalg_core::{
        Matrix2x2,
        Matrix2x4,
        Matrix4x2,
        Matrix4x4,
        Vector2,
        Vector4,
    };


    #[rustfmt::skip]
    #[test]
    fn test_matrix_components1() {
        let matrix = Matrix4x2::new(
            1_i32, 2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32, 8_i32,
        );

        assert_eq!(matrix[0][0], 1_i32);
        assert_eq!(matrix[0][1], 2_i32);
        assert_eq!(matrix[0][2], 3_i32);
        assert_eq!(matrix[0][3], 4_i32);
        assert_eq!(matrix[1][0], 5_i32);
        assert_eq!(matrix[1][1], 6_i32);
        assert_eq!(matrix[1][2], 7_i32);
        assert_eq!(matrix[1][3], 8_i32);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_components2() {
        let matrix = Matrix4x2::new(
            1_i32, 2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32, 8_i32,
        );

        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c0r1, matrix[0][1]);
        assert_eq!(matrix.c0r2, matrix[0][2]);
        assert_eq!(matrix.c0r3, matrix[0][3]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c1r1, matrix[1][1]);
        assert_eq!(matrix.c1r2, matrix[1][2]);
        assert_eq!(matrix.c1r3, matrix[1][3]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds1() {
        let matrix = Matrix4x2::new(
            1_i32, 2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32, 8_i32,
        );

        assert_eq!(matrix[0][4], matrix[0][4]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds2() {
        let matrix = Matrix4x2::new(
            1_i32, 2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32, 8_i32,
        );

        assert_eq!(matrix[2][0], matrix[2][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds3() {
        let matrix = Matrix4x2::new(
            1_i32, 2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32, 8_i32,
        );

        assert_eq!(matrix[2][4], matrix[2][4]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds4() {
        let matrix = Matrix4x2::new(
            1_i32, 2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32, 8_i32,
        );

        assert_eq!(matrix[0][usize::MAX], matrix[0][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds5() {
        let matrix = Matrix4x2::new(
            1_i32, 2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32, 8_i32,
        );

        assert_eq!(matrix[usize::MAX][0], matrix[usize::MAX][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds6() {
        let matrix = Matrix4x2::new(
            1_i32, 2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32, 8_i32,
        );

        assert_eq!(matrix[usize::MAX][usize::MAX], matrix[usize::MAX][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_matrix() {
        let matrix = Matrix4x2::new(
            2_i32, 3_i32, 4_i32, 5_i32,
            6_i32, 7_i32, 8_i32, 9_i32,
        );
        let identity = Matrix2x2::identity();

        assert_eq!(matrix * identity, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_zero_equals_zero() {
        let matrix = Matrix4x2::new(
            33_i32, 54_i32,  19_i32, 345_i32,
            5_i32,  793_i32, 23_i32, 324_i32,
        );
        let zero_matrix2x2 = Matrix2x2::zero();
        let zero_matrix3x2 = Matrix4x2::zero();

        assert_eq!(matrix * zero_matrix2x2, zero_matrix3x2);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_times_matrix_equals_zero() {
        let matrix = Matrix4x2::new(
            33_i32, 54_i32,  19_i32, 29_i32,
            5_i32,  234_i32, 98_i32, 7_i32,
        );
        let zero = 0_i32;
        let zero_matrix3x2 = Matrix4x2::zero();

        assert_eq!(zero * matrix, zero_matrix3x2);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication1() {
        let matrix4x2 = Matrix4x2::new(
            2_i32, 3_i32, 4_i32, 5_i32,
            5_i32, 6_i32, 7_i32, 8_i32,
        );
        let matrix2x2 = Matrix2x2::new(
            1_i32, 2_i32,
            4_i32, 5_i32,
        );
        let expected = Matrix4x2::new(
            12_i32, 15_i32, 18_i32, 21_i32,
            33_i32, 42_i32, 51_i32, 60_i32,
        );
        let result = matrix4x2 * matrix2x2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication2() {
        let matrix4x2 = Matrix4x2::new(
            2_i32, 3_i32, 4_i32, 5_i32,
            5_i32, 6_i32, 7_i32, 8_i32,
        );
        let matrix2x4 = Matrix2x4::new(
            1_i32, 2_i32,
            3_i32, 4_i32,
            5_i32, 8_i32,
            7_i32, 10_i32,
        );
        let expected = Matrix4x4::new(
            12_i32, 15_i32, 18_i32, 21_i32,
            26_i32, 33_i32, 40_i32, 47_i32,
            50_i32, 63_i32, 76_i32, 89_i32,
            64_i32, 81_i32, 98_i32, 115_i32,
        );
        let result = matrix4x2 * matrix2x4;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication3() {
        let matrix4x2 = Matrix4x2::new(
            4_i32, 5_i32, 6_i32,  7_i32,
            8_i32, 9_i32, 10_i32, 11_i32,
        );
        let vector = Vector2::new(9_i32, -6_i32);
        let expected = Vector4::new(-12_i32, -9_i32, -6_i32, -3_i32);
        let result = matrix4x2 * vector;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_scalar_multiplication() {
        let matrix4x2 = Matrix4x2::new(
            1_i32, 2_i32, 3_i32, 4_i32,
            4_i32, 5_i32, 7_i32, 8_i32,
        );
        let scalar = 13_i32;
        let expected = Matrix4x2::new(
            13_i32, 26_i32, 39_i32, 52_i32,
            52_i32, 65_i32, 91_i32, 104_i32,
        );
        let result = matrix4x2 * scalar;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_matrix_multiplication() {
        let matrix4x2 = Matrix4x2::new(
            1_i32, 2_i32, 3_i32, 4_i32,
            4_i32, 5_i32, 7_i32, 8_i32,
        );
        let scalar = 13_i32;
        let expected = Matrix4x2::new(
            13_i32, 26_i32, 39_i32, 52_i32,
            52_i32, 65_i32, 91_i32, 104_i32,
        );
        let result = scalar * matrix4x2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero_matrix4x2 = Matrix4x2::zero();
        let matrix = Matrix4x2::new(
            3684_i32, 42746_i32, 345_i32, 456_i32,
            546_i32,  76_i32,    167_i32, 915_i32,
        );

        assert_eq!(matrix + zero_matrix4x2, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero_matrix4x2 = Matrix4x2::zero();
        let matrix = Matrix4x2::new(
            3684_i32, 42746_i32, 345_i32, 456_i32,
            546_i32,  76_i32,    167_i32, 915_i32,
        );

        assert_eq!(zero_matrix4x2 + matrix, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_addition() {
        let matrix1 = Matrix4x2::new(
            23_i32, 76_i32,  89_i32, 11_i32,
            34_i32, 324_i32, 75_i32, 62_i32,
        );
        let matrix2 = Matrix4x2::new(
            1_i32,  5_i32,  9_i32,  82_i32,
            13_i32, 17_i32, 21_i32, 6_i32,
        );
        let expected = Matrix4x2::new(
            24_i32, 81_i32, 98_i32,  93_i32,
            47_i32, 341_i32, 96_i32, 68_i32,
        );
        let result = matrix1 + matrix2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_subtraction() {
        let matrix1 = Matrix4x2::new(
            3_i32,  6_i32,  9_i32,  65_i32,
            12_i32, 15_i32, 18_i32, 333_i32,
        );
        let matrix2 = Matrix4x2::new(
            1_i32, 15_i32,  29_i32, 27_i32,
            6_i32, 234_i32, 93_i32, 38_i32,
        );
        let expected = Matrix4x2::new(
            2_i32, -9_i32,   -20_i32, 38_i32,
            6_i32, -219_i32, -75_i32, 295_i32,
        );
        let result = matrix1 - matrix2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_minus_matrix_is_zero() {
        let matrix = Matrix4x2::new(
            3_i32,  6_i32,  9_i32,  12_i32,
            12_i32, 15_i32, 18_i32, 21_i32,
        );
        let zero_matrix3x2 = Matrix4x2::zero();

        assert_eq!(matrix - matrix, zero_matrix3x2);
    }

    #[rustfmt::skip]
    #[test]
    fn test_transpose() {
        let matrix = Matrix4x2::new(
            1_i32, 2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32, 8_i32,
        );
        let expected = Matrix2x4::new(
            1_i32, 5_i32,
            2_i32, 6_i32,
            3_i32, 7_i32,
            4_i32, 8_i32,
        );
        let result = matrix.transpose();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
        let c1 = Vector4::new(5_i32, 6_i32, 7_i32, 8_i32);
        let columns = [c0, c1];
        let expected = Matrix4x2::new(
            1_i32, 2_i32, 3_i32, 4_i32,
            5_i32, 6_i32, 7_i32, 8_i32,
        );
        let result = Matrix4x2::from_columns(&columns);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_rows() {
        let r0 = Vector2::new(1_i32, 2_i32);
        let r1 = Vector2::new(3_i32, 4_i32);
        let r2 = Vector2::new(5_i32, 6_i32);
        let r3 = Vector2::new(7_i32, 8_i32);
        let rows = [r0, r1, r2, r3];
        let expected = Matrix4x2::new(
            1_i32, 3_i32, 5_i32, 7_i32,
            2_i32, 4_i32, 6_i32, 8_i32,
        );
        let result = Matrix4x2::from_rows(&rows);

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix3x4_tests {
    use cglinalg_core::{
        Matrix3x3,
        Matrix3x4,
        Matrix4x3,
        Matrix4x4,
        Vector3,
        Vector4,
    };


    #[rustfmt::skip]
    #[test]
    fn test_matrix_components1() {
        let matrix = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
        );

        assert_eq!(matrix[0][0], 1_i32);
        assert_eq!(matrix[0][1], 2_i32);
        assert_eq!(matrix[0][2], 3_i32);
        assert_eq!(matrix[1][0], 4_i32);
        assert_eq!(matrix[1][1], 5_i32);
        assert_eq!(matrix[1][2], 6_i32);
        assert_eq!(matrix[2][0], 7_i32);
        assert_eq!(matrix[2][1], 8_i32);
        assert_eq!(matrix[2][2], 9_i32);
        assert_eq!(matrix[3][0], 10_i32);
        assert_eq!(matrix[3][1], 11_i32);
        assert_eq!(matrix[3][2], 12_i32);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_components2() {
        let matrix = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
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
        assert_eq!(matrix.c3r0, matrix[3][0]);
        assert_eq!(matrix.c3r1, matrix[3][1]);
        assert_eq!(matrix.c3r2, matrix[3][2]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds1() {
        let matrix = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
        );

        assert_eq!(matrix[4][0], matrix[4][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds2() {
        let matrix = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
        );

        assert_eq!(matrix[0][3], matrix[0][3]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds3() {
        let matrix = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
        );

        assert_eq!(matrix[4][3], matrix[4][3]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds4() {
        let matrix = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
        );

        assert_eq!(matrix[0][usize::MAX], matrix[0][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds5() {
        let matrix = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
        );

        assert_eq!(matrix[usize::MAX][0], matrix[usize::MAX][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds6() {
        let matrix = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
        );

        assert_eq!(matrix[usize::MAX][usize::MAX], matrix[usize::MAX][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_matrix() {
        let matrix = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
        );
        let identity = Matrix4x4::identity();

        assert_eq!(matrix * identity, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_zero_equals_zero() {
        let matrix = Matrix3x4::new(
            33_i32,  54_i32, 234_i32,
            19_i32,  5_i32,  308_i32,
            793_i32, 23_i32, 8_i32,
            49_i32,  11_i32, 27_i32,
        );
        let zero_matrix4x4 = Matrix4x4::zero();
        let zero_matrix3x4 = Matrix3x4::zero();

        assert_eq!(matrix * zero_matrix4x4, zero_matrix3x4);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_matrix_times_matrix_equals_zero() {
        let matrix = Matrix3x4::new(
            33_i32,  54_i32, 234_i32,
            19_i32,  5_i32,  308_i32,
            793_i32, 23_i32, 8_i32,
            49_i32,  11_i32, 27_i32,
        );
        let zero_matrix3x3: Matrix3x3<i32> = Matrix3x3::zero();
        let zero_matrix3x4: Matrix3x4<i32> = Matrix3x4::zero();

        assert_eq!(zero_matrix3x3 * matrix, zero_matrix3x4);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_times_matrix_equals_zero() {
        let matrix = Matrix3x4::new(
            33_i32,  54_i32, 234_i32,
            19_i32,  5_i32,  308_i32,
            793_i32, 23_i32, 8_i32,
            49_i32,  11_i32, 27_i32,
        );
        let zero = 0_i32;
        let zero_matrix3x4 = Matrix3x4::zero();

        assert_eq!(zero * matrix, zero_matrix3x4);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication1() {
        let matrix3x4 = Matrix3x4::new(
            2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,
            8_i32,  9_i32,  10_i32,
            11_i32, 12_i32, 13_i32,
        );
        let matrix4x4 = Matrix4x4::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            9_i32,  10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        );
        let expected = Matrix3x4::new(
            80_i32,  90_i32,  100_i32,
            184_i32, 210_i32, 236_i32,
            288_i32, 330_i32, 372_i32,
            392_i32, 450_i32, 508_i32,
        );
        let result = matrix3x4 * matrix4x4;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication2() {
        let matrix3x4 = Matrix3x4::new(
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32,
        );
        let vector = Vector4::new(9_i32, 6_i32, -12_i32, -24_i32);
        let expected = Vector3::new(-354_i32, -375_i32, -396_i32);
        let result = matrix3x4 * vector;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication3() {
        let matrix3x4 = Matrix3x4::new(
            2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,
            8_i32,  9_i32,  10_i32,
            11_i32, 12_i32, 13_i32,
        );
        let matrix4x3 = Matrix4x3::new(
            9_i32,  -6_i32,  12_i32, 4_i32,
            35_i32,  96_i32, 27_i32, 4_i32,
            87_i32,  8_i32,  80_i32, 70_i32,
        );
        let expected = Matrix3x3::new(
            128_i32,  147_i32,  166_i32,
            810_i32,  972_i32,  1134_i32,
            1624_i32, 1869_i32, 2114_i32,
        );
        let result = matrix3x4 * matrix4x3;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_scalar_multiplication() {
        let matrix3x4 = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
        );
        let scalar = 13_i32;
        let expected = Matrix3x4::new(
            13_i32,  26_i32,  39_i32,
            52_i32,  65_i32,  78_i32,
            91_i32,  104_i32, 117_i32,
            130_i32, 143_i32, 156_i32,
        );
        let result = matrix3x4 * scalar;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_matrix_multiplication() {
        let matrix3x4 = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
        );
        let scalar = 13_i32;
        let expected = Matrix3x4::new(
            13_i32,  26_i32,  39_i32,
            52_i32,  65_i32,  78_i32,
            91_i32,  104_i32, 117_i32,
            130_i32, 143_i32, 156_i32,
        );
        let result = scalar * matrix3x4;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero_matrix3x4 = Matrix3x4::zero();
        let matrix = Matrix3x4::new(
            3684_i32, 42746_i32, 2389_i32,
            345_i32,  546_i32,   234_i32,
            76_i32,   167_i32,   890_i32,
            415_i32,  251_i32,   2340_i32,
        );

        assert_eq!(matrix + zero_matrix3x4, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero_matrix3x4 = Matrix3x4::zero();
        let matrix = Matrix3x4::new(
            3684_i32, 42746_i32, 2389_i32,
            345_i32,  546_i32,   234_i32,
            76_i32,   167_i32,   890_i32,
            415_i32,  251_i32,   2340_i32,
        );

        assert_eq!(zero_matrix3x4 + matrix, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_addition() {
        let matrix1 = Matrix3x4::new(
            23_i32,  76_i32,  45_i32,
            89_i32,  34_i32, -21_i32,
            324_i32, 75_i32, -204_i32,
            614_i32, 15_i32,  98_i32,
        );
        let matrix2 = Matrix3x4::new(
            1_i32,  5_i32,  23_i32,
            9_i32,  13_i32, 80_i32,
            17_i32, 21_i32, 3_i32,
            87_i32, 41_i32, 34_i32,
        );
        let expected = Matrix3x4::new(
            24_i32,  81_i32,  68_i32,
            98_i32,  47_i32,  59_i32,
            341_i32, 96_i32, -201_i32,
            701_i32, 56_i32,  132_i32,
        );
        let result = matrix1 + matrix2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_subtraction() {
        let matrix1 = Matrix3x4::new(
            3_i32,  6_i32,  9_i32,
            9_i32,  12_i32, 12_i32,
            15_i32, 18_i32, 15_i32,
            21_i32, 24_i32, 18_i32,
        );
        let matrix2 = Matrix3x4::new(
            1_i32,   15_i32, 10_i32,
            29_i32,  6_i32,  71_i32,
            234_i32, 93_i32, 67_i32,
            93_i32,  7_i32,  91_i32,
        );
        let expected = Matrix3x4::new(
             2_i32,   -9_i32,  -1_i32,
            -20_i32,   6_i32,  -59_i32,
            -219_i32, -75_i32, -52_i32,
            -72_i32,   17_i32, -73_i32,
        );
        let result = matrix1 - matrix2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_minus_matrix_is_zero() {
        let matrix = Matrix3x4::new(
            3_i32,  6_i32,  9_i32,
            9_i32,  12_i32, 15_i32,
            15_i32, 18_i32, 21_i32,
            21_i32, 24_i32, 27_i32,
        );
        let zero_matrix3x4 = Matrix3x4::zero();

        assert_eq!(matrix - matrix, zero_matrix3x4);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_transpose() {
        let matrix = Matrix3x4::new(
            1_i32, 2_i32, 3_i32,
            3_i32, 4_i32, 6_i32,
            5_i32, 6_i32, 9_i32,
            7_i32, 8_i32, 12_i32,
        );
        let expected = Matrix4x3::new(
            1_i32, 3_i32, 5_i32, 7_i32,
            2_i32, 4_i32, 6_i32, 8_i32,
            3_i32, 6_i32, 9_i32, 12_i32,
        );
        let result = matrix.transpose();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector3::new(1_i32,  2_i32,  3_i32);
        let c1 = Vector3::new(4_i32,  5_i32,  6_i32);
        let c2 = Vector3::new(7_i32,  8_i32,  9_i32);
        let c3 = Vector3::new(10_i32, 11_i32, 12_i32);
        let columns = [c0, c1, c2, c3];
        let expected = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
        );
        let result = Matrix3x4::from_columns(&columns);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_rows() {
        let r0 = Vector4::new(1_i32, 2_i32,  3_i32,  4_i32);
        let r1 = Vector4::new(5_i32, 6_i32,  7_i32,  8_i32);
        let r2 = Vector4::new(9_i32, 10_i32, 11_i32, 12_i32);
        let rows = [r0, r1, r2];
        let expected = Matrix3x4::new(
            1_i32, 5_i32, 9_i32,
            2_i32, 6_i32, 10_i32,
            3_i32, 7_i32, 11_i32,
            4_i32, 8_i32, 12_i32,
        );
        let result = Matrix3x4::from_rows(&rows);

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod matrix4x3_tests {
    use cglinalg_core::{
        Matrix3x3,
        Matrix3x4,
        Matrix4x3,
        Matrix4x4,
        Vector3,
        Vector4,
    };


    #[rustfmt::skip]
    #[test]
    fn test_matrix_components1() {
        let matrix = Matrix4x3::new(
            1_i32, 2_i32,  3_i32,  4_i32,
            5_i32, 6_i32,  7_i32,  8_i32,
            9_i32, 10_i32, 11_i32, 12_i32,
        );

        assert_eq!(matrix[0][0], 1_i32);
        assert_eq!(matrix[0][1], 2_i32);
        assert_eq!(matrix[0][2], 3_i32);
        assert_eq!(matrix[0][3], 4_i32);
        assert_eq!(matrix[1][0], 5_i32);
        assert_eq!(matrix[1][1], 6_i32);
        assert_eq!(matrix[1][2], 7_i32);
        assert_eq!(matrix[1][3], 8_i32);
        assert_eq!(matrix[2][0], 9_i32);
        assert_eq!(matrix[2][1], 10_i32);
        assert_eq!(matrix[2][2], 11_i32);
        assert_eq!(matrix[2][3], 12_i32);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_components2() {
        let matrix = Matrix4x3::new(
            1_i32, 2_i32,  3_i32,  4_i32,
            5_i32, 6_i32,  7_i32,  8_i32,
            9_i32, 10_i32, 11_i32, 12_i32,
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
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds1() {
        let matrix = Matrix4x3::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            10_i32, 11_i32, 12_i32, 13_i32,
        );

        assert_eq!(matrix[0][4], matrix[0][4]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds2() {
        let matrix = Matrix4x3::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            10_i32, 11_i32, 12_i32, 13_i32,
        );

        assert_eq!(matrix[3][0], matrix[3][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds3() {
        let matrix = Matrix4x3::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            10_i32, 11_i32, 12_i32, 13_i32,
        );

        assert_eq!(matrix[3][4], matrix[3][4]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds4() {
        let matrix = Matrix4x3::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            10_i32, 11_i32, 12_i32, 13_i32,
        );

        assert_eq!(matrix[0][usize::MAX], matrix[0][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds5() {
        let matrix = Matrix4x3::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            10_i32, 11_i32, 12_i32, 13_i32,
        );

        assert_eq!(matrix[usize::MAX][0], matrix[usize::MAX][0]);
    }

    #[rustfmt::skip]
    #[test]
    #[should_panic]
    fn test_matrix_components_out_of_bounds6() {
        let matrix = Matrix4x3::new(
            1_i32,  2_i32,  3_i32,  4_i32,
            5_i32,  6_i32,  7_i32,  8_i32,
            10_i32, 11_i32, 12_i32, 13_i32,
        );

        assert_eq!(matrix[usize::MAX][usize::MAX], matrix[usize::MAX][usize::MAX]);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_identity_equals_matrix() {
        let matrix = Matrix4x3::new(
            2_i32,  3_i32,  4_i32,  5_i32,
            6_i32,  7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32, 13_i32,
        );
        let identity = Matrix3x3::identity();

        assert_eq!(matrix * identity, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_times_zero_equals_zero() {
        let matrix = Matrix4x3::new(
            33_i32, 54_i32,  19_i32, 345_i32,
            5_i32,  793_i32, 23_i32, 324_i32,
            23_i32, 98_i32,  84_i32, 89_i32,
        );
        let zero_matrix3x3 = Matrix3x3::zero();
        let zero_matrix4x3 = Matrix4x3::zero();

        assert_eq!(matrix * zero_matrix3x3, zero_matrix4x3);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_times_matrix_equals_zero() {
        let matrix = Matrix4x3::new(
            33_i32, 54_i32,  19_i32, 29_i32,
            5_i32,  234_i32, 98_i32, 7_i32,
            23_i32, 98_i32,  84_i32, 89_i32,
        );
        let zero = 0_i32;
        let zero_matrix4x3 = Matrix4x3::zero();

        assert_eq!(zero * matrix, zero_matrix4x3);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication1() {
        let matrix4x3 = Matrix4x3::new(
            2_i32, 3_i32,  4_i32,  5_i32,
            5_i32, 6_i32,  7_i32,  8_i32,
            9_i32, 10_i32, 11_i32, 12_i32,
        );
        let matrix3x3 = Matrix3x3::new(
            1_i32, 2_i32, 3_i32,
            4_i32, 5_i32, 6_i32,
            7_i32, 8_i32, 9_i32,
        );
        let expected = Matrix4x3::new(
            39_i32,  45_i32,  51_i32,  57_i32,
            87_i32,  102_i32, 117_i32, 132_i32,
            135_i32, 159_i32, 183_i32, 207_i32,
        );
        let result = matrix4x3 * matrix3x3;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication2() {
        let matrix4x3 = Matrix4x3::new(
            2_i32, 3_i32, 4_i32,  5_i32,
            5_i32, 6_i32, 7_i32,  8_i32,
            8_i32, 9_i32, 10_i32, 11_i32,
        );
        let matrix3x4 = Matrix3x4::new(
            1_i32,  2_i32,  3_i32,
            4_i32,  5_i32,  6_i32,
            7_i32,  8_i32,  9_i32,
            10_i32, 11_i32, 12_i32,
        );
        let expected = Matrix4x4::new(
            36_i32,  42_i32,  48_i32,  54_i32,
            81_i32,  96_i32,  111_i32, 126_i32,
            126_i32, 150_i32, 174_i32, 198_i32,
            171_i32, 204_i32, 237_i32, 270_i32,
        );
        let result = matrix4x3 * matrix3x4;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_multiplication3() {
        let matrix4x3 = Matrix4x3::new(
            4_i32,  5_i32,  6_i32,  7_i32,
            8_i32,  9_i32,  10_i32, 11_i32,
            12_i32, 13_i32, 14_i32, 15_i32,
        );
        let vector = Vector3::new(9_i32, -6_i32, 34_i32);
        let expected = Vector4::new(396_i32, 433_i32, 470_i32, 507_i32);
        let result = matrix4x3 * vector;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_scalar_multiplication() {
        let matrix4x3 = Matrix4x3::new(
            1_i32, 2_i32,  3_i32,  4_i32,
            4_i32, 5_i32,  7_i32,  8_i32,
            9_i32, 10_i32, 11_i32, 12_i32,
        );
        let scalar = 13_i32;
        let expected = Matrix4x3::new(
            13_i32,  26_i32,  39_i32,  52_i32,
            52_i32,  65_i32,  91_i32,  104_i32,
            117_i32, 130_i32, 143_i32, 156_i32,
        );
        let result = matrix4x3 * scalar;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_scalar_matrix_multiplication() {
        let matrix4x3 = Matrix4x3::new(
            1_i32, 2_i32,  3_i32,  4_i32,
            4_i32, 5_i32,  7_i32,  8_i32,
            9_i32, 10_i32, 11_i32, 12_i32,
        );
        let scalar = 13_i32;
        let expected = Matrix4x3::new(
            13_i32,  26_i32,  39_i32,  52_i32,
            52_i32,  65_i32,  91_i32,  104_i32,
            117_i32, 130_i32, 143_i32, 156_i32,
        );
        let result = scalar * matrix4x3;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero_matrix4x3 = Matrix4x3::zero();
        let matrix = Matrix4x3::new(
            3684_i32, 42746_i32, 345_i32, 456_i32,
            546_i32,  76_i32,    167_i32, 915_i32,
            320_i32,  2430_i32,  894_i32, 324_i32,
        );

        assert_eq!(matrix + zero_matrix4x3, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero_matrix4x3 = Matrix4x3::zero();
        let matrix = Matrix4x3::new(
            3684_i32, 42746_i32, 345_i32, 456_i32,
            546_i32,  76_i32,    167_i32, 915_i32,
            320_i32,  2430_i32,  894_i32, 324_i32,
        );

        assert_eq!(zero_matrix4x3 + matrix, matrix);
    }

    #[rustfmt::skip]
    #[test]
    fn test_addition() {
        let matrix1 = Matrix4x3::new(
            23_i32, 76_i32,  89_i32, 11_i32,
            34_i32, 324_i32, 75_i32, 62_i32,
            88_i32, 61_i32,  45_i32, 16_i32,
        );
        let matrix2 = Matrix4x3::new(
            1_i32,  5_i32,  9_i32,  82_i32,
            13_i32, 17_i32, 21_i32, 6_i32,
            29_i32, 91_i32, 64_i32, 43_i32,
        );
        let expected = Matrix4x3::new(
            24_i32,  81_i32,  98_i32,  93_i32,
            47_i32,  341_i32, 96_i32,  68_i32,
            117_i32, 152_i32, 109_i32, 59_i32,
        );
        let result = matrix1 + matrix2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_subtraction() {
        let matrix1 = Matrix4x3::new(
            3_i32,  6_i32,  9_i32,  65_i32,
            12_i32, 15_i32, 18_i32, 333_i32,
            28_i32, 71_i32, 4_i32,  92_i32,
        );
        let matrix2 = Matrix4x3::new(
            1_i32, 15_i32,  29_i32, 27_i32,
            6_i32, 234_i32, 93_i32, 38_i32,
            74_i32, 97_i32, 10_i32, 100_i32,
        );
        let expected = Matrix4x3::new(
             2_i32,  -9_i32,   -20_i32,  38_i32,
             6_i32,  -219_i32, -75_i32,  295_i32,
            -46_i32, -26_i32,  -6_i32,  -8_i32,
        );
        let result = matrix1 - matrix2;

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_matrix_minus_matrix_is_zero() {
        let matrix = Matrix4x3::new(
            3_i32,  6_i32,  9_i32,  12_i32,
            12_i32, 15_i32, 18_i32, 21_i32,
            34_i32, 17_i32, 8_i32,  84_i32,
        );
        let zero_matrix4x3 = Matrix4x3::zero();

        assert_eq!(matrix - matrix, zero_matrix4x3);
    }

    #[rustfmt::skip]
    #[test]
    fn test_transpose() {
        let matrix = Matrix4x3::new(
            1_i32, 2_i32,  3_i32,  4_i32,
            5_i32, 6_i32,  7_i32,  8_i32,
            9_i32, 10_i32, 11_i32, 12_i32,
        );
        let expected = Matrix3x4::new(
            1_i32, 5_i32, 9_i32,
            2_i32, 6_i32, 10_i32,
            3_i32, 7_i32, 11_i32,
            4_i32, 8_i32, 12_i32,
        );
        let result = matrix.transpose();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector4::new(1_i32, 2_i32,  3_i32,  4_i32);
        let c1 = Vector4::new(5_i32, 6_i32,  7_i32,  8_i32);
        let c2 = Vector4::new(9_i32, 10_i32, 11_i32, 12_i32);
        let columns = [c0, c1, c2];
        let expected = Matrix4x3::new(
            1_i32, 2_i32,  3_i32,  4_i32,
            5_i32, 6_i32,  7_i32,  8_i32,
            9_i32, 10_i32, 11_i32, 12_i32,
        );
        let result = Matrix4x3::from_columns(&columns);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_construction_from_rows() {
        let r0 = Vector3::new(1_i32,  2_i32,  3_i32);
        let r1 = Vector3::new(4_i32,  5_i32,  6_i32);
        let r2 = Vector3::new(7_i32,  8_i32,  9_i32);
        let r3 = Vector3::new(10_i32, 11_i32, 12_i32);
        let rows = [r0, r1, r2, r3];
        let expected = Matrix4x3::new(
            1_i32, 4_i32, 7_i32, 10_i32,
            2_i32, 5_i32, 8_i32, 11_i32,
            3_i32, 6_i32, 9_i32, 12_i32,
        );
        let result = Matrix4x3::from_rows(&rows);

        assert_eq!(result, expected);
    }
}
