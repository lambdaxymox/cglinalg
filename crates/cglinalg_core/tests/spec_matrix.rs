extern crate cglinalg_core;
extern crate num_traits;
extern crate proptest;


use proptest::prelude::*;
use cglinalg_core::{
    Matrix,
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
    SimdScalar,
    SimdScalarSigned,
    SimdScalarFloat,
};
use approx::{
    relative_eq,
};



fn strategy_scalar_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = S> 
where
    S: SimdScalarSigned + Arbitrary
{
    fn rescale<S: SimdScalarSigned>(value: S, min_value: S, max_value: S) -> S {
        min_value + (value % (max_value - min_value))
    }

    any::<S>().prop_map(move |value| {
        let sign_value = value.signum();
        let abs_value = value.abs();
        
        sign_value * rescale(abs_value, min_value, max_value)
    })
    .no_shrink()
}

fn any_matrix2<S>() -> impl Strategy<Value = Matrix2x2<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S, S, S)>()
        .prop_map(|(c0r0, c0r1, c1r0, c1r1)| {
            let modulus = num_traits::cast(100_000_000).unwrap();
            let matrix = Matrix2x2::new(c0r0, c0r1, c1r0, c1r1);

            matrix % modulus
        })
}

fn any_matrix3<S>() -> impl Strategy<Value = Matrix3x3<S>>
where 
    S: SimdScalar + Arbitrary
{
    any::<((S, S, S), (S, S, S), (S, S, S))>()
        .prop_map(|((c0r0, c0r1, c0r2), (c1r0, c1r1, c1r2), (c2r0, c2r1, c2r2))| {
            let modulus = num_traits::cast(100_000_000).unwrap();
            let matrix = Matrix3x3::new(
                c0r0, c0r1, c0r2, 
                c1r0, c1r1, c1r2, 
                c2r0, c2r1, c2r2
            );

            matrix % modulus
        })
}

fn any_matrix4<S>() -> impl Strategy<Value = Matrix4x4<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<((S, S, S, S), (S, S, S, S), (S, S, S, S), (S, S, S, S))>().prop_map(
        |((c0r0, c0r1, c0r2, c0r3), (c1r0, c1r1, c1r2, c1r3), (c2r0, c2r1, c2r2, c2r3), (c3r0, c3r1, c3r2, c3r3))| {
            let modulus = num_traits::cast(100_000_000).unwrap();
            let matrix = Matrix4x4::new(
                c0r0, c0r1, c0r2, c0r3, 
                c1r0, c1r1, c1r2, c1r3, 
                c2r0, c2r1, c2r2, c2r3, 
                c3r0, c3r1, c3r2, c3r3
            );

            matrix % modulus
        }
    )
}

fn strategy_scalar_f64_any() -> impl Strategy<Value = f64> {
    let min_value = f64::sqrt(f64::EPSILON);
    let max_value = f64::sqrt(f64::MAX) / f64::sqrt(2_f64);

    strategy_scalar_signed_from_abs_range(min_value, max_value)
}

fn strategy_scalar_i32_any() -> impl Strategy<Value = i32> {
    let min_value = 0_i32;
    // let max_value = f64::floor(f64::sqrt(i32::MAX as f64 / 2_f64)) as i32;
    let max_value = 32767_i32;

    strategy_scalar_signed_from_abs_range(min_value, max_value)
}


/// A zero matrix should act as the additive unit element for matrices
/// over their underlying scalars. 
///
/// Given a matrix `m` and a zero matrix `0`
/// ```text
/// 0 + m = m
/// ```
fn prop_matrix_additive_identity<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let zero_matrix = Matrix::zero();

    prop_assert_eq!(zero_matrix + m, m);

    Ok(())
}
        
/// A zero matrix should act as the additive unit element for matrices 
/// over their underlying scalars. 
///
/// Given a matrix `m` and a zero matrix `0`
/// ```text
/// m + 0 = m
/// ```
fn prop_matrix_plus_zero_equals_zero<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let zero_matrix = Matrix::zero();

    prop_assert_eq!(m + zero_matrix, m);

    Ok(())
}

/// Matrix addition over exact scalars is commutative.
///
/// Given matrices `m1` and `m2`
/// ```text
/// m1 + m2 = m2 + m1
/// ```
fn prop_matrix_addition_commutative<S, const R: usize, const C: usize, const RC: usize>(
    m1: Matrix<S, R, C, RC>, 
    m2: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(m1 + m2, m2 + m1);

    Ok(())
}

/// Matrix addition over exact scalars is associative.
///
/// Given matrices `m1`, `m2`, and `m3`
/// ```text
/// (m1 + m2) + m3 ~= m1 + (m2 + m3)
/// ```
fn prop_matrix_addition_approx_associative<S, const R: usize, const C: usize, const RC: usize>(
    m1: Matrix<S, R, C, RC>, 
    m2: Matrix<S, R, C, RC>, 
    m3: Matrix<S, R, C, RC>,
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat + Arbitrary
{
    prop_assert!(relative_eq!((m1 + m2) + m3, m1 + (m2 + m3), epsilon = tolerance));

    Ok(())
}

/// The sum of a matrix and it's additive inverse is the same as 
/// subtracting the two matrices from each other.
///
/// Given matrices `m1` and `m2`
/// ```text
/// m1 + (-m2) = m1 - m2
/// ```
fn prop_matrix_subtraction<S, const R: usize, const C: usize, const RC: usize>(
    m1: Matrix<S, R, C, RC>, 
    m2: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + Arbitrary
{
    prop_assert_eq!(m1 + (-m2), m1 - m2);

    Ok(())
}

/// Matrix addition over exact scalars is associative.
///
/// Given matrices `m1`, `m2`, and `m3`
/// ```text
/// (m1 + m2) + m3 = m1 + (m2 + m3)
/// ```
fn prop_matrix_addition_associative<S, const R: usize, const C: usize, const RC: usize>(
    m1: Matrix<S, R, C, RC>, 
    m2: Matrix<S, R, C, RC>, 
    m3: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!((m1 + m2) + m3, m1 + (m2 + m3));

    Ok(())
}

/// Multiplication of a matrix by a scalar zero is the zero matrix.
///
/// Given a matrix `m` and a zero scalar `0`
/// ```text
/// 0 * m = m * 0 = 0
/// ```
/// Note that we diverge from traditional formalisms of matrix arithmetic 
/// in that we allow multiplication of matrices by scalars on the right-hand 
/// side as well as left-hand side. 
fn prop_zero_times_matrix_equals_zero_matrix<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let zero: S = num_traits::zero();
    let zero_matrix = Matrix::zero();

    prop_assert_eq!(m * zero, zero_matrix);

    Ok(())
}

/// Multiplication of a matrix by a scalar one is the original matrix.
///
/// Given a matrix `m` and a unit scalar `1`
/// ```text
/// 1 * m = m * 1 = m
/// ```
/// Note that we diverge from traditional formalisms of matrix arithmetic 
/// in that we allow multiplication of matrices by scalars on the right-hand 
/// side as well as left-hand side. 
fn prop_one_times_matrix_equals_matrix<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let one: S = num_traits::one();

    prop_assert_eq!(m * one, m);

    Ok(())
}

/// Multiplication of a matrix by a scalar negative one is the additive 
/// inverse of the original matrix.
///
/// Given a matrix `m` and a negative unit scalar `-1`
/// ```text
/// (-1) * m = = m * (-1) = -m
/// ```
/// Note that we diverge from traditional formalisms of matrix arithmetic 
/// in that we allow multiplication of matrices by scalars on the right-hand 
/// side as well as left-hand side. 
fn prop_negative_one_times_matrix_equals_negative_matrix<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + Arbitrary
{
    let one: S = num_traits::one();
    let minus_one = -one;

    prop_assert_eq!(m * minus_one, -m);

    Ok(())
}

/*
/// Multiplication of a matrix by a scalar commutes with scalars.
///
/// Given a matrix `m` and a scalar `c`
/// ```text
/// c * m ~= m * c
/// ```
/// Note that we diverse from traditional formalisms of matrix arithmetic 
/// in that we allow multiplication of matrices by scalars on the left-hand 
/// side as well as the right-hand side.
fn prop_scalar_matrix_multiplication_commutative<S, const R: usize, const C: usize, const RC: usize>(
    c: S, 
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(c * m, m * c);

    Ok(())
}
*/

/// Multiplication of matrices by scalars is compatible with matrix 
/// addition.
///
/// Given matrices `m1` and `m2`, and a scalar `c`
/// ```text
/// c * (m1 + m2) = c * m1 + c * m2
/// ```
fn prop_scalar_matrix_multiplication_compatible_addition<S, const R: usize, const C: usize, const RC: usize>(
    c: S, 
    m1: Matrix<S, R, C, RC>, 
    m2: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!((m1 + m2) * c, m1 * c + m2 * c);

    Ok(())
}

/// Multiplication of matrices by scalars is compatible with matrix 
/// subtraction.
///
/// Given matrices `m1` and `m2`, and a scalar `c`
/// ```text
/// c * (m1 - m2) = c * m1 - c * m2
/// ```
fn prop_scalar_matrix_multiplication_compatible_subtraction<S, const R: usize, const C: usize, const RC: usize>(
    c: S, 
    m1: Matrix<S, R, C, RC>, 
    m2: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!((m1 - m2) * c, m1 * c - m2 * c);

    Ok(())
}

/*
/// Multiplication of a matrix by a scalar commutes with scalars.
///
/// Given a matrix `m` and a scalar `c`
/// ```text
/// c * m = m * c
/// ```
/// Note that we diverse from traditional formalisms of matrix arithmetic 
/// in that we allow multiplication of matrices by scalars on the left-hand 
/// side as well as the right-hand side.
fn prop_scalar_matrix_multiplication_commutative<S, const R: usize, const C: usize, const RC: usize>(
    c: S, 
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(c * m, m * c);

    Ok(())
}
*/

/// Scalar multiplication of a matrix by scalars is compatible.
///
/// Given a matrix `m` and scalars `a` and `b`
/// ```text
/// (a * b) * m = a * (b * m)
/// ```
fn prop_scalar_matrix_multiplication_compatible<S, const R: usize, const C: usize, const RC: usize>(
    a: S, 
    b: S, 
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(m * (a * b), (m * a) * b);

    Ok(())
}

/*
/// Multiplication of a matrix by a scalar commutes with scalars.
///
/// Given a matrix `m` and a scalar `c`
/// ```text
/// c * m = m * c
/// ```
/// Note that we diverse from traditional formalisms of matrix arithmetic 
/// in that we allow multiplication of matrices by scalars on the left-hand 
/// side as well as the right-hand side.
fn prop_scalar_matrix_multiplication_commutative<S, const R: usize, const C: usize, const RC: usize>(
    c: S, 
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(c * m, m * c);

    Ok(())
}
*/
/*
/// Matrices over a set of floating point scalars have a 
/// multiplicative identity.
/// 
/// Given a matrix `m` there is a matrix `identity` such that
/// ```text
/// m * identity = identity * m = m
/// ```
fn prop_matrix_multiplication_identity<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let identity = Matrix::identity();

    prop_assert_eq!(m * identity, m);
    prop_assert_eq!(identity * m, m);

    Ok(())
}
*/
/*
/// Multiplication of a matrix by a scalar zero is the zero matrix.
///
/// Given a matrix `m` and a zero scalar `0`
/// ```text
/// 0 * m = m * 0 = 0
/// ```
/// Note that we diverge from tradition formalisms of matrix arithmetic 
/// in that we allow multiplication of matrices by scalars on the right-hand 
/// side as well as left-hand side. 
fn prop_zero_times_matrix_equals_zero_matrix<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let zero = num_traits::zero();
    let zero_matrix = Matrix::zero();

    prop_assert_eq!(zero * m, zero_matrix);
    prop_assert_eq!(m * zero, zero_matrix);

    Ok(())
}
*/
/*
/// Multiplication of a matrix by the zero matrix is the zero matrix.
///
/// Given a matrix `m`, and the zero matrix `0`
/// ```text
/// 0 * m = m * 0 = 0
/// ```
fn prop_zero_matrix_times_matrix_equals_zero_matrix<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let zero_matrix = Matrix::zero();

    prop_assert_eq!(zero_matrix * m, zero_matrix);
    prop_assert_eq!(m * zero_matrix, zero_matrix);

    Ok(())
}
*/
/*
/// Matrix multiplication is associative.
///
/// Given matrices `m1`, `m2`, and `m3`
/// ```text
/// (m1 * m2) * m3 = m1 * (m2 * m3)
/// ```
fn prop_matrix_multiplication_associative<S, const R: usize, const C: usize, const RC: usize>(
    m1: Matrix<S, R, C, RC>, 
    m2: Matrix<S, R, C, RC>, 
    m3: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!((m1 * m2) * m3, m1* (m2 * m3));

    Ok(())
}
*/
/*
/// Matrix multiplication is distributive over matrix addition.
///
/// Given matrices `m1`, `m2`, and `m3`
/// ```text
/// m1 * (m2 + m3) = m1 * m2 + m1 * m3
/// ```
fn prop_matrix_multiplication_distributive<S, const R: usize, const C: usize, const RC: usize>(
    m1: Matrix<S, R, C, RC>, 
    m2: Matrix<S, R, C, RC>, 
    m3: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(m1 * (m2 + m3), m1 * m2 + m1 * m3);

    Ok(())
}
*/
/*
/// Matrix multiplication is compatible with scalar multiplication.
///
/// Given matrices `m1` and `m2` and a scalar `c`
/// ```text
/// c * (m1 * m2) = (c * m1) * m2 = m1 * (c * m2)
/// ```
fn prop_matrix_multiplication_compatible_with_scalar_multiplication<S, const R: usize, const C: usize, const RC: usize>(
    c: S, 
    m1: Matrix<S, R, C, RC>, 
    m2: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!((m1 * m2) * c, m1 * (m2 * c));

    Ok(())
}
*/

/// Matrix multiplication is compatible with scalar multiplication.
///
/// Given a matrix `m`, scalars `c1` and `c2`
/// ```text
/// (c1 * c2) * m = c1 * (c2 * m)
/// ```
fn prop_matrix_multiplication_compatible_with_scalar_multiplication1<S, const R: usize, const C: usize, const RC: usize>(
    c1: S, 
    c2: S, 
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(m * (c1 * c2), (m * c1) * c2);

    Ok(())
}

/*
/// Matrices over a set of floating point scalars have a 
/// multiplicative identity.
/// 
/// Given a matrix `m` there is a matrix `identity` such that
/// ```text
/// m * identity = identity * m = m
/// ```
fn prop_matrix_multiplication_identity<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let identity = Matrix::identity();

    prop_assert_eq!(m * identity, m);
    prop_assert_eq!(identity * m, m);

    Ok(())
}
*/

/// The double transpose of a matrix is the original matrix.
///
/// Given a matrix `m`
/// ```text
/// transpose(transpose(m)) = m
/// ```
fn prop_matrix_transpose_transpose_equals_matrix<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(m.transpose().transpose(), m);

    Ok(())
}

/// The transposition operation is linear.
/// 
/// Given matrices `m1` and `m2`
/// ```text
/// transpose(m1 + m2) = transpose(m1) + transpose(m2)
/// ```
fn prop_transpose_linear<S, const R: usize, const C: usize, const RC: usize>(
    m1: Matrix<S, R, C, RC>, 
    m2: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!((m1 + m2).transpose(), m1.transpose() + m2.transpose());

    Ok(())
}

/// Scalar multiplication of a matrix and a scalar commutes with 
/// transposition.
/// 
/// Given a matrix `m` and a scalar `c`
/// ```text
/// transpose(c * m) = c * transpose(m)
/// ```
fn prop_transpose_scalar_multiplication<S, const R: usize, const C: usize, const RC: usize>(
    c: S, 
    m: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!((m * c).transpose(), m.transpose() * c);

    Ok(())
}

/*
/// The transpose of the product of two matrices equals the product 
/// of the transposes of the two matrices swapped.
/// 
/// Given matrices `m1` and `m2`
/// ```text
/// transpose(m1 * m2) = transpose(m2) * transpose(m1)
/// ```
fn prop_transpose_product<S, const R: usize, const C: usize, const RC: usize>(
    m1: Matrix<S, R, C, RC>, 
    m2: Matrix<S, R, C, RC>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!((m1 * m2).transpose(), m2.transpose() * m1.transpose());

    Ok(())
}
*/

/// Swapping rows is commutative in the row arguments.
///
/// Given a matrix `m`, and rows `row1` and `row2`
/// ```text
/// m.swap_rows(row1, row2) = m.swap_rows(row2, row1)
/// ```
fn prop_swap_rows_commutative<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>, 
    row1: usize, 
    row2: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let mut m1 = m;
    let mut m2 = m;
    m1.swap_rows(row1, row2);
    m2.swap_rows(row2, row1);

    prop_assert_eq!(m1, m2);

    Ok(())
}

/// Swapping the same row in both arguments is the identity map.
///
/// Given a matrix `m`, and a row `row`
/// ```text
/// m.swap_rows(row, row) = m
/// ```
fn prop_swap_identical_rows_identity<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>, 
    row: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let mut m1 = m;
    m1.swap_rows(row, row);

    prop_assert_eq!(m1, m);

    Ok(())
}

/// Swapping the same two rows twice in succession yields the original 
/// matrix.
///
/// Given a matrix `m`, and rows `row1` and `row2`
/// ```text
/// m.swap_rows(row1, row2).swap_rows(row1, row2) = m
/// ```
fn prop_swap_rows_twice_is_identity<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>, 
    row1: usize, 
    row2: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let mut m1 = m;
    m1.swap_rows(row1, row2);
    m1.swap_rows(row1, row2);

    prop_assert_eq!(m1, m);

    Ok(())
}

/// Swapping columns is commutative in the column arguments.
///
/// Given a matrix `m`, and columns `col1` and `col2`
/// ```text
/// m.swap_columns(col1, col2) = m.swap_columns(col2, col1)
/// ```
fn prop_swap_columns_commutative<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>, 
    col1: usize, 
    col2: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let mut m1 = m;
    let mut m2 = m;
    m1.swap_columns(col1, col2);
    m2.swap_columns(col2, col1);

    prop_assert_eq!(m1, m2);

    Ok(())
}

/// Swapping the same column in both arguments is the identity map.
///
/// Given a matrix `m`, and a column `col`
/// ```text
/// m.swap_columns(col, col) = m
/// ```
fn prop_swap_identical_columns_is_identity<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>, 
    col: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let mut m1 = m;
    m1.swap_columns(col, col);

    prop_assert_eq!(m1, m);

    Ok(())
}

/// Swapping the same two columns twice in succession yields the 
/// original matrix.
///
/// Given a matrix `m`, and columns `col1` and `col2`
/// ```text
/// m.swap_columns(col1, col2).swap_columns(col1, col2) = m
/// ```
fn prop_swap_columns_twice_is_identity<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>, 
    col1: usize, 
    col2: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let mut m1 = m;
    m1.swap_columns(col1, col2);
    m1.swap_columns(col1, col2);

    prop_assert_eq!(m1, m);

    Ok(())
}

/// Swapping elements is commutative in the arguments.
///
/// Given a matrix `m`, and elements `(col1, row1)` and `(col2, row2)`
/// ```text
/// m.swap_elements((col1, row1), (col2, row2)) = m.swap_elements((col2, row2), (col1, row1))
/// ```
fn prop_swap_elements_commutative<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>, 
    col1: usize, 
    row1: usize, 
    col2: usize, 
    row2: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let mut m1 = m;
    let mut m2 = m;
    m1.swap((col1, row1), (col2, row2));
    m2.swap((col2, row2), (col1, row1));

    prop_assert_eq!(m1, m2);

    Ok(())
}

/// Swapping the same element in both arguments is the identity map.
///
/// Given a matrix `m`, and an element index `(col, row)`
/// ```text
/// m.swap_elements((col, row), (col, row)) = m
/// ```
fn prop_swap_identical_elements_is_identity<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>, 
    col: usize, 
    row: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let mut m1 = m;
    m1.swap((col, row), (col, row));

    prop_assert_eq!(m1, m);

    Ok(())
}

/// Swapping the same two elements twice in succession yields the 
/// original matrix.
///
/// Given a matrix `m`, and elements `(col1, row1)` and `(col2, row2)`
/// ```text
/// m.swap_elements((col1, row1), (col2, row2)).swap_elements((col1, row1), (col2, row2)) = m
/// ```
fn prop_swap_elements_twice_is_identity<S, const R: usize, const C: usize, const RC: usize>(
    m: Matrix<S, R, C, RC>, 
    col1: usize, 
    row1: usize, 
    col2: usize, 
    row2: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let mut m1 = m;
    m1.swap((col1, row1), (col2, row2));
    m1.swap((col1, row1), (col2, row2));

    prop_assert_eq!(m1, m);

    Ok(())
}


macro_rules! approx_addition_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use approx::{
            relative_eq
        };
        use cglinalg_core::{
            $MatrixN,
        };
        use super::{
            $Generator,
        };


        proptest! {
            #[test]
            fn prop_matrix_additive_identity(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_additive_identity(m)?
            }
        
            #[test]
            fn prop_matrix_plus_zero_equals_zero(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_plus_zero_equals_zero(m)?
            }

            #[test]
            fn prop_matrix_addition_commutative(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_addition_commutative(m1, m2)?
            }

            #[test]
            fn prop_matrix_addition_approx_associative(m1 in super::$Generator(), m2 in super::$Generator(), m3 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                let m3: super::$MatrixN<$ScalarType> = m3;
                super::prop_matrix_addition_approx_associative(m1, m2, m3, $tolerance)?
            }

            #[test]
            fn prop_matrix_subtraction(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_subtraction(m1, m2)?
            }
        }
    }
    }
}

approx_addition_props!(matrix2_f64_addition_props, Matrix2x2, f64, any_matrix2, 1e-7);
approx_addition_props!(matrix3_f64_addition_props, Matrix3x3, f64, any_matrix3, 1e-7);
approx_addition_props!(matrix4_f64_addition_props, Matrix4x4, f64, any_matrix4, 1e-7);


macro_rules! exact_addition_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            $MatrixN,
        };
        use super::{
            $Generator,
        };


        proptest! {
            #[test]
            fn prop_matrix_additive_identity(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_additive_identity(m)?
            }
        
            #[test]
            fn prop_matrix_plus_zero_equals_zero(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_plus_zero_equals_zero(m)?
            }

            #[test]
            fn prop_matrix_addition_commutative(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_addition_commutative(m1, m2)?
            }

            #[test]
            fn prop_matrix_addition_associative(m1 in super::$Generator(), m2 in super::$Generator(), m3 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                let m3: super::$MatrixN<$ScalarType> = m3;
                super::prop_matrix_addition_associative(m1, m2, m3)?
            }
        }
    }
    }
}

exact_addition_props!(matrix2_i32_addition_props, Matrix2x2, i32, any_matrix2);
exact_addition_props!(matrix3_i32_addition_props, Matrix3x3, i32, any_matrix3);
exact_addition_props!(matrix4_i32_addition_props, Matrix4x4, i32, any_matrix4);


macro_rules! approx_scalar_multiplication_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            $MatrixN,
        };
        use super::{
            $Generator,
            $ScalarGen,
        };


        proptest! {
            #[test]
            fn prop_zero_times_matrix_equals_zero_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_zero_times_matrix_equals_zero_matrix(m)?
            }

            #[test]
            fn prop_one_times_matrix_equals_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_one_times_matrix_equals_matrix(m)?
            }

            #[test]
            fn prop_negative_one_times_matrix_equals_negative_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_negative_one_times_matrix_equals_negative_matrix(m)?
            }

            /*
            #[test]
            fn prop_scalar_matrix_multiplication_commutative(c in super::$ScalarGen(), m in super::$Generator()) {
                let c: $ScalarType = c;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_scalar_matrix_multiplication_commutative(c, m)?
            }
            */
        }
    }
    }
}

approx_scalar_multiplication_props!(
    matrix2_f64_scalar_multiplication_props, 
    Matrix2x2, f64, 
    any_matrix2, 
    strategy_scalar_f64_any, 
    1e-7
);
approx_scalar_multiplication_props!(
    matrix3_f64_scalar_multiplication_props, 
    Matrix3x3, f64, 
    any_matrix3, 
    strategy_scalar_f64_any, 
    1e-7
);
approx_scalar_multiplication_props!(
    matrix4_f64_scalar_multiplication_props, 
    Matrix4x4, 
    f64, 
    any_matrix4, 
    strategy_scalar_f64_any, 
    1e-7
);


macro_rules! exact_scalar_multiplication_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            $MatrixN, 
        };
        use super::{
            $Generator,
        };


        proptest! {
            #[test]
            fn prop_scalar_matrix_multiplication_compatible_addition(
                c in super::$ScalarGen(), 
                m1 in super::$Generator(), 
                m2 in super::$Generator()
            ) {
                let c: $ScalarType = c;
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_scalar_matrix_multiplication_compatible_addition(c, m1, m2)?
            }

            #[test]
            fn prop_scalar_matrix_multiplication_compatible_subtraction(
                c in super::$ScalarGen(), 
                m1 in super::$Generator(), 
                m2 in super::$Generator()
            ) {
                let c: $ScalarType = c;
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_scalar_matrix_multiplication_compatible_subtraction(c, m1, m2)?
            }

            #[test]
            fn prop_zero_times_matrix_equals_zero_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_zero_times_matrix_equals_zero_matrix(m)?
            }

            #[test]
            fn prop_one_times_matrix_equals_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_one_times_matrix_equals_matrix(m)?
            }

            /*
            #[test]
            fn prop_scalar_matrix_multiplication_commutative(c in super::$ScalarGen(), m in super::$Generator()) {
                let c: $ScalarType = c;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_scalar_matrix_multiplication_commutative(c, m)?
            }
            */

            #[test]
            fn prop_scalar_matrix_multiplication_compatible(
                a in super::$ScalarGen(), 
                b in super::$ScalarGen(), 
                m in super::$Generator()
            ) {
                let a: $ScalarType = a;
                let b: $ScalarType = b;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_scalar_matrix_multiplication_compatible(a, b, m)?
            }
        }
    }
    }
}

exact_scalar_multiplication_props!(matrix2_i32_scalar_multiplication_props, Matrix2x2, i32, any_matrix2, strategy_scalar_i32_any);
exact_scalar_multiplication_props!(matrix3_i32_scalar_multiplication_props, Matrix3x3, i32, any_matrix3, strategy_scalar_i32_any);
exact_scalar_multiplication_props!(matrix4_i32_scalar_multiplication_props, Matrix4x4, i32, any_matrix4, strategy_scalar_i32_any);


macro_rules! approx_multiplication_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            $MatrixN,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /*
            #[test]
            fn prop_scalar_matrix_multiplication_commutative(c in super::$ScalarGen(), m in super::$Generator()) {
                let c: $ScalarType = c;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_scalar_matrix_multiplication_commutative(c, m)?
            }
            */
            /*
            #[test]
            fn prop_matrix_multiplication_identity(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_multiplication_identity(m)?
            }
            */
            /*
            #[test]
            fn prop_zero_matrix_times_matrix_equals_zero_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_zero_matrix_times_matrix_equals_zero_matrix(m)?
            }
            */

            #[test]
            fn prop_zero_times_matrix_equals_zero_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_zero_times_matrix_equals_zero_matrix(m)?
            }
        }
    }
    }
}

approx_multiplication_props!(matrix2_f64_matrix_multiplication_props, Matrix2x2, f64, any_matrix2, strategy_scalar_f64_any);
approx_multiplication_props!(matrix3_f64_matrix_multiplication_props, Matrix3x3, f64, any_matrix3, strategy_scalar_f64_any);
approx_multiplication_props!(matrix4_f64_matrix_multiplication_props, Matrix4x4, f64, any_matrix4, strategy_scalar_f64_any);


macro_rules! exact_multiplication_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            $MatrixN,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /*
            #[test]
            fn prop_matrix_multiplication_associative(
                m1 in super::$Generator(), 
                m2 in super::$Generator(), 
                m3 in super::$Generator()
            ) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_multiplication_associative(m1, m2)?
            }
            */
            /*
            #[test]
            fn prop_matrix_multiplication_distributive(
                m1 in super::$Generator(), 
                m2 in super::$Generator(), 
                m3 in super::$Generator()
            ) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_multiplication_distributive(m1, m2, m3)?
            }
            */
            /*
            #[test]
            fn prop_matrix_multiplication_compatible_with_scalar_multiplication(
                c in super::$ScalarGen(), 
                m1 in super::$Generator(), 
                m2 in super::$Generator()
            ) {
                let c: $ScalarType = c;
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_multiplication_compatible_with_scalar_multiplication(c, m1, m2)?
            }
            */

            #[test]
            fn prop_matrix_multiplication_compatible_with_scalar_multiplication1(
                c1 in super::$ScalarGen(), 
                c2 in super::$ScalarGen(), 
                m in super::$Generator()
            ) {
                let c1: $ScalarType = c1;
                let c2: $ScalarType = c2;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_multiplication_compatible_with_scalar_multiplication1(c1, c2, m)?
            }
            /*
            #[test]
            fn prop_zero_matrix_times_matrix_equals_zero_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_zero_matrix_times_matrix_equals_zero_matrix(m)?
            }
            */
            /*
            #[test]
            fn prop_matrix_multiplication_identity(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_multiplication_identity(m)?
            }
            */
        }
    }
    }
}

exact_multiplication_props!(matrix2_i32_matrix_multiplication_props, Matrix2x2, i32, any_matrix2, strategy_scalar_i32_any);
exact_multiplication_props!(matrix3_i32_matrix_multiplication_props, Matrix3x3, i32, any_matrix3, strategy_scalar_i32_any);
exact_multiplication_props!(matrix4_i32_matrix_multiplication_props, Matrix4x4, i32, any_matrix4, strategy_scalar_i32_any);


macro_rules! approx_transposition_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            #[test]
            fn prop_matrix_transpose_transpose_equals_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_transpose_transpose_equals_matrix(m)?
            }

            #[test]
            fn prop_transpose_linear(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_transpose_linear(m1, m2)?
            }

            #[test]
            fn prop_transpose_scalar_multiplication(c in super::$ScalarGen(), m in super::$Generator()) {
                let c: $ScalarType = c;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_transpose_scalar_multiplication(c, m)?
            }

            /*
            #[test]
            fn prop_transpose_product(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_transpose_product(m1, m2)?
            }
            */
        }
    }
    }
}

approx_transposition_props!(matrix2_f64_transposition_props, Matrix2x2, f64, any_matrix2, strategy_scalar_f64_any, 1e-7);
approx_transposition_props!(matrix3_f64_transposition_props, Matrix3x3, f64, any_matrix3, strategy_scalar_f64_any, 1e-7);
approx_transposition_props!(matrix4_f64_transposition_props, Matrix4x4, f64, any_matrix4, strategy_scalar_f64_any, 1e-7);


macro_rules! exact_transposition_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            #[test]
            fn prop_matrix_transpose_transpose_equals_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_transpose_transpose_equals_matrix(m)?
            }

            #[test]
            fn prop_transpose_linear(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_transpose_linear(m1, m2)?
            }

            #[test]
            fn prop_transpose_scalar_multiplication(c in super::$ScalarGen(), m in super::$Generator()) {
                let c: $ScalarType = c;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_transpose_scalar_multiplication(c, m)?
            }

            /*
            #[test]
            fn prop_transpose_product(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_transpose_product(m1, m2)?
            }
            */
        }
    }
    }
}

exact_transposition_props!(matrix2_i32_transposition_props, Matrix2x2, i32, any_matrix2, strategy_scalar_i32_any);
exact_transposition_props!(matrix3_i32_transposition_props, Matrix3x3, i32, any_matrix3, strategy_scalar_i32_any);
exact_transposition_props!(matrix4_i32_transposition_props, Matrix4x4, i32, any_matrix4, strategy_scalar_i32_any);


macro_rules! swap_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $UpperBound:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            #[test]
            fn prop_swap_rows_commutative(
                m in super::$Generator(), 
                row1 in 0..$UpperBound as usize, row2 in 0..$UpperBound as usize
            ) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_swap_rows_commutative(m, row1, row2)?
            }

            #[test]
            fn prop_swap_identical_rows_identity(m in super::$Generator(), row in 0..$UpperBound as usize) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_swap_identical_rows_identity(m, row)?
            }

            #[test]
            fn prop_swap_rows_twice_is_identity(
                m in super::$Generator(), 
                row1 in 0..$UpperBound as usize, row2 in 0..$UpperBound as usize
            ) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_swap_rows_twice_is_identity(m, row1, row2)?
            }

            #[test]
            fn prop_swap_columns_commutative(
                m in super::$Generator(), 
                col1 in 0..$UpperBound as usize, col2 in 0..$UpperBound as usize
            ) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_swap_columns_commutative(m, col1, col2)?
            }

            #[test]
            fn prop_swap_identical_columns_is_identity(m in super::$Generator(), col in 0..$UpperBound as usize) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_swap_identical_columns_is_identity(m, col)?
            }

            #[test]
            fn prop_swap_columns_twice_is_identity(
                m in super::$Generator(), 
                col1 in 0..$UpperBound as usize, col2 in 0..$UpperBound as usize
            ) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_swap_columns_twice_is_identity(m, col1, col2)?
            }

            #[test]
            fn prop_swap_elements_commutative(
                m in super::$Generator(), 
                col1 in 0..$UpperBound as usize, row1 in 0..$UpperBound as usize,
                col2 in 0..$UpperBound as usize, row2 in 0..$UpperBound as usize
            ) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_swap_elements_commutative(m, col1, row1, col2, row2)?
            }

            #[test]
            fn prop_swap_identical_elements_is_identity(
                m in super::$Generator(), 
                col in 0..$UpperBound as usize, row in 0..$UpperBound as usize
            ) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_swap_identical_elements_is_identity(m, col, row)?
            }

            #[test]
            fn prop_swap_elements_twice_is_identity(
                m in super::$Generator(), 
                col1 in 0..$UpperBound as usize, row1 in 0..$UpperBound as usize, 
                col2 in 0..$UpperBound as usize, row2 in 0..$UpperBound as usize
            ) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_swap_elements_twice_is_identity(m, col1, row1, col2, row2)?
            }
        }
    }
    }
}

swap_props!(matrix2_swap_props, Matrix2x2, i32, any_matrix2, 2);
swap_props!(matrix3_swap_props, Matrix3x3, i32, any_matrix3, 3);
swap_props!(matrix4_swap_props, Matrix4x4, i32, any_matrix4, 4);

