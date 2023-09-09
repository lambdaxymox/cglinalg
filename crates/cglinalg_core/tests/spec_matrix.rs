extern crate cglinalg_numeric;
extern crate cglinalg_core;
extern crate proptest;


use cglinalg_numeric::{
    SimdScalar,
    SimdScalarOrd,
    SimdScalarSigned,
    SimdScalarFloat,
};
use cglinalg_core::{
    Matrix,
    Matrix1x1,
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
    Matrix2x3,
    Matrix3x2,
    Matrix2x4,
    Matrix4x2,
    Matrix3x4,
    Matrix4x3,
};
use cglinalg_core::{
    Const,
    CanMultiply,
    DimMul,
    ShapeConstraint,
};
use approx::{
    relative_eq,
    relative_ne,
};

use proptest::prelude::*;


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

fn strategy_matrix_signed_from_abs_range<S, const R: usize, const C: usize>(min_value: S, max_value: S) -> impl Strategy<Value = Matrix<S, R, C>>
where
    S: SimdScalarSigned + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarSigned
    {
        min_value + (value % (max_value - min_value))
    }

    fn rescale_matrix<S, const R: usize, const C: usize>(value: Matrix<S, R, C>, min_value: S, max_value: S) -> Matrix<S, R, C> 
    where
        S: SimdScalarSigned
    {
        value.map(|element| rescale(element, min_value, max_value))
    }

    any::<[[S; R]; C]>().prop_map(move |array| {
        let vector = Matrix::from(array);
        
        rescale_matrix(vector, min_value, max_value)
    })
    .no_shrink()
}

fn strategy_matrix_f64_any<const R: usize, const C: usize>() -> impl Strategy<Value = Matrix<f64, R, C>> {
    let min_value = f64::sqrt(f64::sqrt(f64::EPSILON));
    let max_value = f64::sqrt(f64::sqrt(f64::MAX)) / 2_f64;

    strategy_matrix_signed_from_abs_range(min_value, max_value)
}

fn strategy_matrix_i32_any<const R: usize, const C: usize>() -> impl Strategy<Value = Matrix<i32, R, C>> {
    let min_value = 0_i32;
    let max_value = 1_000_000_000_i32;

    strategy_matrix_signed_from_abs_range(min_value, max_value)
}

fn strategy_matrix_i32_norm<const R: usize, const C: usize>() -> impl Strategy<Value = Matrix<i32, R, C>> {
    let min_value = 0_i32;
    // let max_value = (f64::floor(f64::sqrt(i32::MAX as f64) / 4_f64)) as i32;
    let max_value = 11585_i32;

    strategy_matrix_signed_from_abs_range(min_value, max_value)
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
/// 0 + m == m
/// ```
fn prop_matrix_additive_identity<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalar
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
/// m + 0 == m
/// ```
fn prop_matrix_plus_zero_equals_zero<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let zero_matrix = Matrix::zero();

    prop_assert_eq!(m + zero_matrix, m);

    Ok(())
}

/// Matrix addition is commutative.
///
/// Given matrices `m1` and `m2`
/// ```text
/// m1 + m2 == m2 + m1
/// ```
fn prop_matrix_addition_commutative<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(m1 + m2, m2 + m1);

    Ok(())
}

/// Matrix addition over exact scalars is associative.
///
/// Given matrices `m1`, `m2`, and `m3`
/// ```text
/// (m1 + m2) + m3 == m1 + (m2 + m3)
/// ```
fn prop_matrix_addition_associative<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>, 
    m3: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!((m1 + m2) + m3, m1 + (m2 + m3));

    Ok(())
}

/// The sum of a matrix and it's additive inverse is the same as 
/// subtracting the two matrices from each other.
///
/// Given matrices `m1` and `m2`
/// ```text
/// m1 + (-m2) == m1 - m2
/// ```
fn prop_matrix_subtraction<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    prop_assert_eq!(m1 + (-m2), m1 - m2);

    Ok(())
}

/// Multiplication of a matrix by a scalar zero is the zero matrix.
///
/// Given a matrix `m` and a zero scalar `0`
/// ```text
/// 0 * m == m * 0 == 0
/// ```
/// Note that we diverge from traditional formalisms of matrix arithmetic 
/// in that we allow multiplication of matrices by scalars on the right-hand 
/// side as well as left-hand side. 
fn prop_zero_times_matrix_equals_zero_matrix<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let zero = S::zero();
    let zero_matrix = Matrix::zero();

    prop_assert_eq!(m * zero, zero_matrix);

    Ok(())
}

/// Multiplication of a matrix by a scalar one is the original matrix.
///
/// Given a matrix `m` and a unit scalar `1`
/// ```text
/// 1 * m == m * 1 == m
/// ```
/// Note that we diverge from traditional formalisms of matrix arithmetic 
/// in that we allow multiplication of matrices by scalars on the right-hand 
/// side as well as left-hand side. 
fn prop_one_times_matrix_equals_matrix<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let one = S::one();

    prop_assert_eq!(m * one, m);

    Ok(())
}

/// Multiplication of a matrix by a scalar negative one is the additive 
/// inverse of the original matrix.
///
/// Given a matrix `m` and a negative unit scalar `-1`
/// ```text
/// (-1) * m == m * (-1) == -m
/// ```
/// Note that we diverge from traditional formalisms of matrix arithmetic 
/// in that we allow multiplication of matrices by scalars on the right-hand 
/// side as well as left-hand side. 
fn prop_negative_one_times_matrix_equals_negative_matrix<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let one = S::one();
    let minus_one = -one;

    prop_assert_eq!(m * minus_one, -m);

    Ok(())
}

/// Multiplication of matrices by scalars is compatible with matrix 
/// addition.
///
/// Given matrices `m1` and `m2`, and a scalar `c`
/// ```text
/// (m1 + m2) * c == m1 * c + m2 * c
/// ```
fn prop_scalar_matrix_multiplication_compatible_addition<S, const R: usize, const C: usize>(
    c: S, 
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!((m1 + m2) * c, m1 * c + m2 * c);

    Ok(())
}

/// Multiplication of matrices by scalars is compatible with matrix 
/// subtraction.
///
/// Given matrices `m1` and `m2`, and a scalar `c`
/// ```text
/// (m1 - m2) * c == m1 * c - m2 * c
/// ```
fn prop_scalar_matrix_multiplication_compatible_subtraction<S, const R: usize, const C: usize>(
    c: S, 
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!((m1 - m2) * c, m1 * c - m2 * c);

    Ok(())
}

/// Multiplication of a matrix by a scalar matrix commutes.
///
/// Given a matrix `m`, a scalar `c`, and a matrix `c_matrix` that is `c` on the
/// diagonal and zero elsewhere
/// ```text
/// c_matrix * m == m * c_matrix
/// ```
/// Note that we diverse from traditional formalisms of matrix arithmetic 
/// in that we allow multiplication of matrices by scalars on the left-hand 
/// side as well as the right-hand side.
fn prop_scalar_matrix_multiplication_commutative<S, const N: usize, const NN: usize>(
    c: S, 
    m: Matrix<S, N, N>
) -> Result<(), TestCaseError>
where
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<N>, Const<N>, Const<N>, Const<N>>,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    let c_matrix = Matrix::identity() * c;

    prop_assert_eq!(c_matrix * m, m * c_matrix);

    Ok(())
}

/// Scalar multiplication of a matrix by scalars is compatible.
///
/// Given a matrix `m` and scalars `a` and `b`
/// ```text
/// m * (a * b) == (m * a) * b
/// ```
fn prop_scalar_matrix_multiplication_compatible<S, const R: usize, const C: usize>(
    a: S, 
    b: S, 
    m: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(m * (a * b), (m * a) * b);

    Ok(())
}

/// Matrices over a set of floating point scalars have a 
/// multiplicative identity.
/// 
/// Given a matrix `m` there is a matrix `identity` such that
/// ```text
/// m * identity == identity * m == m
/// ```
fn prop_matrix_multiplication_identity<S, const N: usize, const NN: usize>(m: Matrix<S, N, N>) -> Result<(), TestCaseError>
where
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<N>, Const<N>, Const<N>, Const<N>>,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    let identity = Matrix::identity();

    prop_assert_eq!(m * identity, m);
    prop_assert_eq!(identity * m, m);

    Ok(())
}

/// Multiplication of a matrix by the zero matrix is the zero matrix.
///
/// Given a matrix `m`, and the zero matrix `0`
/// ```text
/// 0 * m == m * 0 == 0
/// ```
fn prop_zero_matrix_times_matrix_equals_zero_matrix<S, const N: usize, const NN: usize>(m: Matrix<S, N, N>) -> Result<(), TestCaseError>
where
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<N>, Const<N>, Const<N>, Const<N>>,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    let zero_matrix = Matrix::zero();

    prop_assert_eq!(zero_matrix * m, zero_matrix);
    prop_assert_eq!(m * zero_matrix, zero_matrix);

    Ok(())
}


/// Matrix multiplication over exact scalars is associative.
///
/// Given matrices `m1`, `m2`, and `m3`
/// ```text
/// (m1 * m2) * m3 == m1 * (m2 * m3)
/// ```
fn prop_matrix_multiplication_associative<S, const N: usize, const NN: usize>(
    m1: Matrix<S, N, N>, 
    m2: Matrix<S, N, N>, 
    m3: Matrix<S, N, N>
) -> Result<(), TestCaseError>
where
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<N>, Const<N>, Const<N>, Const<N>>,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    prop_assert_eq!((m1 * m2) * m3, m1* (m2 * m3));

    Ok(())
}


/// Matrix multiplication is distributive over matrix addition.
///
/// Given matrices `m1`, `m2`, and `m3`
/// ```text
/// m1 * (m2 + m3) == m1 * m2 + m1 * m3
/// ```
fn prop_matrix_multiplication_distributive<S, const N: usize, const NN: usize>(
    m1: Matrix<S, N, N>, 
    m2: Matrix<S, N, N>, 
    m3: Matrix<S, N, N>
) -> Result<(), TestCaseError>
where
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<N>, Const<N>, Const<N>, Const<N>>,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    prop_assert_eq!(m1 * (m2 + m3), m1 * m2 + m1 * m3);

    Ok(())
}

/// Matrix multiplication is compatible with scalar multiplication.
///
/// Given matrices `m1` and `m2` and a scalar `c`
/// ```text
/// (m1 * m2) * c == m1 * (m2 * c)
/// ```
fn prop_matrix_multiplication_compatible_with_scalar_multiplication<S, const N: usize, const NN: usize>(
    c: S, 
    m1: Matrix<S, N, N>, 
    m2: Matrix<S, N, N>
) -> Result<(), TestCaseError>
where
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<N>, Const<N>, Const<N>, Const<N>>,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    prop_assert_eq!((m1 * m2) * c, m1 * (m2 * c));

    Ok(())
}


/// Matrix multiplication is compatible with scalar multiplication.
///
/// Given a matrix `m`, scalars `c1` and `c2`
/// ```text
/// m * (c1 * c2) == (m * c1) * c2
/// ```
fn prop_matrix_multiplication_compatible_with_scalar_multiplication1<S, const R: usize, const C: usize>(
    c1: S, 
    c2: S, 
    m: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(m * (c1 * c2), (m * c1) * c2);

    Ok(())
}

/// The double transpose of a matrix is the original matrix.
///
/// Given a matrix `m`
/// ```text
/// transpose(transpose(m)) == m
/// ```
fn prop_matrix_transpose_transpose_equals_matrix<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(m.transpose().transpose(), m);

    Ok(())
}

/// The transposition operation is linear.
/// 
/// Given matrices `m1` and `m2`
/// ```text
/// transpose(m1 + m2) == transpose(m1) + transpose(m2)
/// ```
fn prop_transpose_linear<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!((m1 + m2).transpose(), m1.transpose() + m2.transpose());

    Ok(())
}

/// Scalar multiplication of a matrix and a scalar commutes with 
/// transposition.
/// 
/// Given a matrix `m` and a scalar `c`
/// ```text
/// transpose(m * c) == transpose(m) * c
/// ```
fn prop_transpose_scalar_multiplication<S, const R: usize, const C: usize>(
    c: S, 
    m: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!((m * c).transpose(), m.transpose() * c);

    Ok(())
}


/// The transpose of the product of two matrices equals the product 
/// of the transposes of the two matrices swapped.
/// 
/// Given matrices `m1` and `m2`
/// ```text
/// transpose(m1 * m2) == transpose(m2) * transpose(m1)
/// ```
fn prop_transpose_product<S, const N: usize, const NN: usize>(
    m1: Matrix<S, N, N>, 
    m2: Matrix<S, N, N>
) -> Result<(), TestCaseError>
where
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<N>, Const<N>, Const<N>, Const<N>>,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    prop_assert_eq!((m1 * m2).transpose(), m2.transpose() * m1.transpose());

    Ok(())
}


/// Swapping rows is commutative in the row arguments.
///
/// Given a matrix `m`, and rows `row1` and `row2`
/// ```text
/// m.swap_rows(row1, row2) == m.swap_rows(row2, row1)
/// ```
fn prop_swap_rows_commutative<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>, 
    row1: usize, 
    row2: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar
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
/// m.swap_rows(row, row) == m
/// ```
fn prop_swap_identical_rows_identity<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>, 
    row: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar
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
/// m.swap_rows(row1, row2).swap_rows(row1, row2) == m
/// ```
fn prop_swap_rows_twice_is_identity<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>, 
    row1: usize, 
    row2: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar
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
/// m.swap_columns(col1, col2) == m.swap_columns(col2, col1)
/// ```
fn prop_swap_columns_commutative<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>, 
    col1: usize, 
    col2: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar
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
/// m.swap_columns(col, col) == m
/// ```
fn prop_swap_identical_columns_is_identity<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>, 
    col: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar
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
/// m.swap_columns(col1, col2).swap_columns(col1, col2) == m
/// ```
fn prop_swap_columns_twice_is_identity<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>, 
    col1: usize, 
    col2: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar
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
/// m.swap_elements((col1, row1), (col2, row2)) == m.swap_elements((col2, row2), (col1, row1))
/// ```
fn prop_swap_elements_commutative<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>, 
    col1: usize, 
    row1: usize, 
    col2: usize, 
    row2: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar
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
/// m.swap_elements((col, row), (col, row)) == m
/// ```
fn prop_swap_identical_elements_is_identity<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>, 
    col: usize, 
    row: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar
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
/// m.swap_elements((col1, row1), (col2, row2)).swap_elements((col1, row1), (col2, row2)) == m
/// ```
fn prop_swap_elements_twice_is_identity<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>, 
    col1: usize, 
    row1: usize, 
    col2: usize, 
    row2: usize
) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let mut m1 = m;
    m1.swap((col1, row1), (col2, row2));
    m1.swap((col1, row1), (col2, row2));

    prop_assert_eq!(m1, m);

    Ok(())
}

/// The matrix dot product is nonnegative.
/// 
/// Given a matrix `m` the dot product of `m` satisfies
/// ```text
/// dot(m, m) >= 0
/// ```
fn prop_matrix_dot_product_nonnegative<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let zero = S::zero();

    prop_assert!(m.dot(&m) >= zero);

    Ok(())
}

/// The matrix dot product is point separating from zero.
/// 
/// Given a matrix `m`, the dot product of `m` with itself satisfies the property
/// ```text
/// dot(m, m) != 0 ==> m != 0
/// ```
/// Equivalently, the matrix dot product satisfies
/// ```text
/// m != 0 ==> dot(m, m) != 0
/// ```
/// which is the relation the property uses for testability reasons.
fn prop_matrix_dot_product_nonzero<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let zero = S::zero();
    let zero_matrix = Matrix::zero();

    prop_assume!(m != zero_matrix);
    prop_assert_ne!(m.dot(&m), zero);

    Ok(())
}

/// The matrix dot product is left bilinear.
/// 
/// Given matrices `m1`, `m2`, and `m3`, the matrix dot product satisfies
/// ```text
/// dot(m1 + m2, m3) == dot(m1, m3) + dot(m2, m3)
/// ```
fn prop_matrix_dot_product_left_bilinear<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>,
    m2: Matrix<S, R, C>,
    m3: Matrix<S, R, C>
) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let lhs = (m1 + m2).dot(&m3);
    let rhs = m1.dot(&m3) + m2.dot(&m3);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The matrix dot product is right bilinear
/// 
/// Given matrices `m1`, `m2`, and `m3`, the matrix dot product satisfies
/// ```text
/// dot(m1, m2 + m3) == dot(m1, m2) + dot(m1, m3)
/// ```
fn prop_matrix_dot_product_right_bilinear<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>, 
    m3: Matrix<S, R, C>
) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let lhs = m1.dot(&(m2 + m3));
    let rhs = m1.dot(&m2) + m1.dot(&m3);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The matrix dot product is homogeneous.
/// 
/// Given constants `c1` and `c2`, and matrices `m1` and `m2`, the matrix dot
/// product satisfies
/// ```text
/// dot(m1 * c1, m2 * c2) == dot(m1, m2) * (c1 * c2)
/// ```
fn prop_matrix_dot_product_homogeneous<S, const R: usize, const C: usize>(
    c1: S, 
    c2: S, 
    m1: Matrix<S, R, C>,
    m2: Matrix<S, R, C>,
) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let lhs = (m1 * c1).dot(&(m2 * c2));
    let rhs = (m1.dot(&m2)) * (c1 * c2);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The matrix **L1** norm is nonnegative.
/// 
/// Given a matrix `m`
/// ```text
/// l1_norm(m) >= 0
/// ```
fn prop_matrix_l1_norm_nonnegative<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let zero = S::zero();

    prop_assert!(m.l1_norm() >= zero);

    Ok(())
}

/// The matrix **L1** norm is point separating from zero.
/// 
/// Given a matrix `m`
/// ```text
/// l1_norm(m) == 0 ==> m == 0
/// ```
/// Equivalently, if `m` is not zero
/// ```text
/// m != 0 ==> l1_norm(m) != 0
/// ```
/// For the sake of testability, we use the second form.
fn prop_matrix_l1_norm_point_separating1<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let zero = S::zero();
    let zero_matrix = Matrix::zero();

    prop_assume!(m != zero_matrix);
    prop_assert_ne!(m.l1_norm(), zero);

    Ok(())
}

/// The matrix **L1** norm is point separating.
/// 
/// Given matrices `m1` and `m2`
/// ```text
/// l1_norm(m1) == l1_norm(m2) ==> m1 == m2
/// ```
/// Equivalently
/// ```text
/// m1 != m2 ==> l1_norm(m1) != l1_norm(m2)
/// ```
/// For the sake of testability, we use the second form.
fn prop_matrix_l1_norm_point_separating2<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let zero = S::zero();

    prop_assume!(m1 != m2);
    prop_assert_ne!((m1 - m2).l1_norm(), zero);

    Ok(())
}

/// The matrix **L1** norm is homogeneous.
/// 
/// Given a constant `c` and a matrix `m`
/// ```text
/// l1_norm(m * c) == l1_norm(m) * abs(c)
/// ```
fn prop_matrix_l1_norm_homogeneous<S, const R: usize, const C: usize>(c: S, m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let lhs = (m * c).l1_norm();
    let rhs = m.l1_norm() * c.abs();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The matrix **L1** norm satisfies the triangle inequality.
/// 
/// Given matrices `m1` and `m2`
/// ```text
/// l1_norm(m1 + m2) <= l1_norm(m1) + l1_norm(m2)
/// ```
fn prop_matrix_l1_norm_triangle_inequality<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>,
    m2: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let lhs = (m1 + m2).l1_norm();
    let rhs = m1.l1_norm() + m2.l1_norm();

    prop_assert!(lhs <= rhs);

    Ok(())
}

/// The matrix **L1** norm is point separating from zero.
/// 
/// Given a matrix `m`
/// ```text
/// l1_norm(m) == 0 ==> m == 0
/// ```
/// Equivalently, if `m` is not zero
/// ```text
/// m != 0 ==> l1_norm(m) != 0
/// ```
/// For the sake of testability, we use the second form.
fn prop_approx_matrix_l1_norm_point_separating1<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>,
    tolerance: S
) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let zero_matrix = Matrix::zero();

    prop_assume!(relative_ne!(m, zero_matrix, epsilon = tolerance));
    prop_assert!(m.l1_norm() > tolerance);

    Ok(())
}

/// The matrix **L1** norm is point separating.
/// 
/// Given matrices `m1` and `m2`
/// ```text
/// l1_norm(m1) == l1_norm(m2) ==> m1 == m2
/// ```
/// Equivalently
/// ```text
/// m1 != m2 ==> l1_norm(m1) != l1_norm(m2)
/// ```
/// For the sake of testability, we use the second form.
fn prop_approx_matrix_l1_norm_point_separating2<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>, 
    tolerance: S
) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(m1, m2, epsilon = tolerance));
    prop_assert!((m1 - m2).l1_norm() > tolerance);

    Ok(())
}

/// The matrix **L-infinity** norm is nonnegative.
/// 
/// Given a matrix `m`
/// ```text
/// linf_norm(m) >= 0
/// ```
fn prop_matrix_linf_norm_nonnegative<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let zero = S::zero();

    prop_assert!(m.linf_norm() >= zero);

    Ok(())
}

/// The matrix **L-infinity** norm is point separating from zero.
/// 
/// Given a matrix `m`
/// ```text
/// linf_norm(m) == 0 ==> m == 0
/// ```
/// Equivalently, if `m` is not zero
/// ```text
/// m != 0 ==> linf_norm(m) != 0
/// ```
/// For the sake of testability, we use the second form.
fn prop_matrix_linf_norm_point_separating1<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let zero = S::zero();
    let zero_matrix = Matrix::zero();

    prop_assume!(m != zero_matrix);
    prop_assert_ne!(m.linf_norm(), zero);

    Ok(())
}

/// The matrix **L-infinity** norm is point separating.
/// 
/// Given matrices `m1` and `m2`
/// ```text
/// linf_norm(m1) == linf_norm(m2) ==> m1 == m2
/// ```
/// Equivalently
/// ```text
/// m1 != m2 ==> linf_norm(m1) != linf_norm(m2)
/// ```
/// For the sake of testability, we use the second form.
fn prop_matrix_linf_norm_point_separating2<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let zero = S::zero();

    prop_assume!(m1 != m2);
    prop_assert_ne!((m1 - m2).linf_norm(), zero);

    Ok(())
}

/// The matrix **L-infinity** norm is homogeneous.
/// 
/// Given a constant `c` and a matrix `m`
/// ```text
/// linf_norm(m * c) == linf_norm(m) * abs(c)
/// ```
fn prop_matrix_linf_norm_homogeneous<S, const R: usize, const C: usize>(c: S, m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let lhs = (m * c).linf_norm();
    let rhs = m.linf_norm() * c.abs();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The matrix **L-infinity** norm is point separating from zero.
/// 
/// Given a matrix `m`
/// ```text
/// linf_norm(m) == 0 ==> m == 0
/// ```
/// Equivalently, if `m` is not zero
/// ```text
/// m != 0 ==> linf_norm(m) != 0
/// ```
/// For the sake of testability, we use the second form.
fn prop_approx_matrix_linf_norm_point_separating1<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>, 
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let zero_matrix = Matrix::zero();

    prop_assume!(relative_ne!(m, zero_matrix, epsilon = tolerance));
    prop_assert!(m.linf_norm() > tolerance);

    Ok(())
}

/// The matrix **L-infinity** norm is point separating.
/// 
/// Given matrices `m1` and `m2`
/// ```text
/// linf_norm(m1) == linf_norm(m2) ==> m1 == m2
/// ```
/// Equivalently
/// ```text
/// m1 != m2 ==> linf_norm(m1) != linf_norm(m2)
/// ```
/// For the sake of testability, we use the second form.
fn prop_approx_matrix_linf_norm_point_separating2<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>,
    m2: Matrix<S, R, C>,
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(m1, m2, epsilon = tolerance));
    prop_assert!((m1 - m2).linf_norm() > tolerance);

    Ok(())
}

/// The matrix **L-infinity** norm satisfies the triangle inequality.
/// 
/// Given matrices `m1` and `m2`
/// ```text
/// linf_norm(m1 + m2) <= linf_norm(m1) + linf_norm(m2)
/// ```
fn prop_matrix_linf_norm_triangle_inequality<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>,
    m2: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let lhs = (m1 + m2).linf_norm();
    let rhs = m1.linf_norm() + m2.linf_norm();

    prop_assert!(lhs <= rhs);

    Ok(())
}

/// The squared matrix **Frobenius** norm is nonnegative.
/// 
/// Given a matrix `m`
/// ```text
/// norm_squared(m) >= 0
/// ```
fn prop_matrix_norm_squared_nonnegative<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let zero = S::zero();

    prop_assert!(m.norm_squared() >= zero);

    Ok(())
}

/// The squared matrix **Frobenius** norm is point separating from zero.
/// 
/// Given a matrix `m`
/// ```text
/// norm_squared(m) == 0 ==> m == 0
/// ```
/// Equivalently, if `m` is not zero
/// ```text
/// m != 0 ==> norm_squared(m) != 0
/// ```
/// For the sake of testability, we use the second form.
fn prop_matrix_norm_squared_point_separating1<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let zero = S::zero();
    let zero_matrix = Matrix::zero();

    prop_assume!(m != zero_matrix);
    prop_assert_ne!(m.norm_squared(), zero);

    Ok(())
}

/// The squared matrix **Frobenius** norm is point separating.
/// 
/// Given matrices `m1` and `m2`
/// ```text
/// norm_squared(m1) == norm_squared(m2) ==> m1 == m2
/// ```
/// Equivalently
/// ```text
/// m1 != m2 ==> norm_squared(m1) != norm_squared(m2)
/// ```
/// For the sake of testability, we use the second form.
fn prop_matrix_norm_squared_point_separating2<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let zero = S::zero();

    prop_assume!(m1 != m2);
    prop_assert_ne!((m1 - m2).norm_squared(), zero);

    Ok(())
}

/// The squared matrix **Frobenius** norm is homogeneous.
/// 
/// Given a constant `c` and a matrix `m`
/// ```text
/// norm_squared(m * c) == norm_squared(m) * abs(c) * abs(c)
/// ```
fn prop_matrix_norm_squared_homogeneous_squared<S, const R: usize, const C: usize>(c: S, m: Matrix<S, R, C>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let lhs = (m * c).norm_squared();
    let rhs = m.norm_squared() * c.abs() * c.abs();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The [`Matrix::magnitude`] function and [`Matrix::norm`] function are synonyms.
/// 
/// Given a matrix `m`
/// ```text
/// magnitude(m) == norm(m)
/// ```
fn prop_matrix_magnitude_norm_synonyms<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert_eq!(m.magnitude(), m.norm());

    Ok(())
}

/// The [`Matrix::magnitude_squared`] function and [`Matrix::norm_squared`] function
/// are synonyms.
/// 
/// Given a matrix `m`
/// ```text
/// magnitude_squared(m) == norm_squared(m)
/// ```
fn prop_matrix_magnitude_squared_norm_squared_synonyms<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    prop_assert_eq!(m.magnitude_squared(), m.norm_squared());

    Ok(())
}

/// The matrix **Frobenius** norm is nonnegative.
/// 
/// Given a matrix `m`
/// ```text
/// norm(m) >= 0
/// ```
fn prop_matrix_norm_nonnegative<S, const R: usize, const C: usize>(m: Matrix<S, R, C>) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let zero = S::zero();

    prop_assert!(m.norm() >= zero);

    Ok(())
}

/// The matrix **Frobenius** norm is point separating from zero.
/// 
/// Given a matrix `m`
/// ```text
/// norm(m) == 0 ==> m == 0
/// ```
/// Equivalently, if `m` is not zero
/// ```text
/// m != 0 ==> norm(m) != 0
/// ```
/// For the sake of testability, we use the second form.
fn prop_approx_matrix_norm_point_separating1<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>, 
    tolerance: S
) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let zero = S::zero();
    let zero_matrix = Matrix::zero();

    prop_assume!(relative_ne!(m, zero_matrix, epsilon = tolerance));
    prop_assert!(relative_ne!(m.norm(), zero, epsilon = tolerance));

    Ok(())
}

/// The matrix **Frobenius** norm is point separating.
/// 
/// Given matrices `m1` and `m2`
/// ```text
/// norm(m1) == norm(m2) ==> m1 == m2
/// ```
/// Equivalently
/// ```text
/// m1 != m2 ==> norm(m1) != norm(m2)
/// ```
/// For the sake of testability, we use the second form.
fn prop_approx_matrix_norm_point_separating2<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>, 
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(m1, m2, epsilon = tolerance));
    prop_assert!(relative_ne!(m1.norm(), m2.norm(), epsilon = tolerance));

    Ok(())
}


fn prop_matrix_trace_linear<S, const N: usize>(m1: Matrix<S, N, N>, m2: Matrix<S, N, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let lhs = (m1 + m2).trace();
    let rhs = m1.trace() + m2.trace();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}


fn prop_matrix_trace_transpose<S, const N: usize>(m: Matrix<S, N, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let lhs = m.transpose().trace();
    let rhs = m.trace();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}


fn prop_matrix_trace_product<S, const N: usize, const NN: usize>(m1: Matrix<S, N, N,>, m2: Matrix<S, N, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    let lhs = (m1 * m2).trace();
    let rhs = (m2 * m1).trace();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}


fn prop_matrix_trace_scalar_product<S, const N: usize>(c: S, m: Matrix<S, N, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let lhs = (m * c).trace();
    let rhs = m.trace() * c;

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

fn prop_approx_matrix_trace_linear<S, const N: usize>(
    m1: Matrix<S, N, N>, 
    m2: Matrix<S, N, N>,
    tolerance: S,
    max_relative: S
) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let lhs = (m1 + m2).trace();
    let rhs = m1.trace() + m2.trace();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}


fn prop_approx_matrix_trace_product<S, const N: usize, const NN: usize>(
    m1: Matrix<S, N, N,>, 
    m2: Matrix<S, N, N>,
    tolerance: S,
    max_relative: S
) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    let lhs = (m1 * m2).trace();
    let rhs = (m2 * m1).trace();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}


fn prop_approx_matrix_trace_scalar_product<S, const N: usize>(
    c: S, 
    m: Matrix<S, N, N>,
    tolerance: S,
    max_relative: S,
) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let lhs = (m * c).trace();
    let rhs = m.trace() * c;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}


macro_rules! approx_arithmetic_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
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
            fn prop_matrix_subtraction(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_subtraction(m1, m2)?
            }
        }
    }
    }
}

approx_arithmetic_props!(matrix1x1_f64_arithmetic_props, Matrix2x2, f64, strategy_matrix_f64_any);
approx_arithmetic_props!(matrix2x2_f64_arithmetic_props, Matrix2x2, f64, strategy_matrix_f64_any);
approx_arithmetic_props!(matrix3x3_f64_arithmetic_props, Matrix3x3, f64, strategy_matrix_f64_any);
approx_arithmetic_props!(matrix4x4_f64_arithmetic_props, Matrix4x4, f64, strategy_matrix_f64_any);


macro_rules! exact_arithmetic_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
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
            fn prop_matrix_subtraction(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_subtraction(m1, m2)?
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

exact_arithmetic_props!(matrix1x1_i32_arithmetic_props, Matrix1x1, i32, strategy_matrix_i32_any);
exact_arithmetic_props!(matrix2x2_i32_arithmetic_props, Matrix2x2, i32, strategy_matrix_i32_any);
exact_arithmetic_props!(matrix3x3_i32_arithmetic_props, Matrix3x3, i32, strategy_matrix_i32_any);
exact_arithmetic_props!(matrix4x4_i32_arithmetic_props, Matrix4x4, i32, strategy_matrix_i32_any);

exact_arithmetic_props!(matrix2x3_i32_arithmetic_props, Matrix2x3, i32, strategy_matrix_i32_any);
exact_arithmetic_props!(matrix3x2_i32_arithmetic_props, Matrix3x2, i32, strategy_matrix_i32_any);
exact_arithmetic_props!(matrix2x4_i32_arithmetic_props, Matrix2x4, i32, strategy_matrix_i32_any);
exact_arithmetic_props!(matrix4x2_i32_arithmetic_props, Matrix4x2, i32, strategy_matrix_i32_any);
exact_arithmetic_props!(matrix3x4_i32_arithmetic_props, Matrix3x4, i32, strategy_matrix_i32_any);
exact_arithmetic_props!(matrix4x3_i32_arithmetic_props, Matrix4x3, i32, strategy_matrix_i32_any);


macro_rules! approx_scalar_multiplication_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
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

            #[test]
            fn prop_scalar_matrix_multiplication_commutative(c in super::$ScalarGen(), m in super::$Generator()) {
                let c: $ScalarType = c;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_scalar_matrix_multiplication_commutative(c, m)?
            }
        }
    }
    }
}

approx_scalar_multiplication_props!(
    matrix1x1_f64_scalar_multiplication_props,
    Matrix1x1,
    f64,
    strategy_matrix_f64_any,
    strategy_scalar_f64_any
);
approx_scalar_multiplication_props!(
    matrix2x2_f64_scalar_multiplication_props,
    Matrix2x2,
    f64,
    strategy_matrix_f64_any,
    strategy_scalar_f64_any
);
approx_scalar_multiplication_props!(
    matrix3x3_f64_scalar_multiplication_props,
    Matrix3x3,
    f64,
    strategy_matrix_f64_any,
    strategy_scalar_f64_any
);
approx_scalar_multiplication_props!(
    matrix4x4_f64_scalar_multiplication_props,
    Matrix4x4,
    f64,
    strategy_matrix_f64_any,
    strategy_scalar_f64_any
);


macro_rules! exact_scalar_multiplication_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
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

            #[test]
            fn prop_scalar_matrix_multiplication_commutative(c in super::$ScalarGen(), m in super::$Generator()) {
                let c: $ScalarType = c;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_scalar_matrix_multiplication_commutative(c, m)?
            }

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

exact_scalar_multiplication_props!(matrix1x1_i32_scalar_multiplication_props, Matrix1x1, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);
exact_scalar_multiplication_props!(matrix2x2_i32_scalar_multiplication_props, Matrix2x2, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);
exact_scalar_multiplication_props!(matrix3x3_i32_scalar_multiplication_props, Matrix3x3, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);
exact_scalar_multiplication_props!(matrix4x4_i32_scalar_multiplication_props, Matrix4x4, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);


macro_rules! exact_multiplication_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_multiplication_associative(
                m1 in super::$Generator(), 
                m2 in super::$Generator(), 
                m3 in super::$Generator()
            ) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                let m3: super::$MatrixN<$ScalarType> = m3;
                super::prop_matrix_multiplication_associative(m1, m2, m3)?
            }

            #[test]
            fn prop_matrix_multiplication_distributive(
                m1 in super::$Generator(), 
                m2 in super::$Generator(), 
                m3 in super::$Generator()
            ) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                let m3: super::$MatrixN<$ScalarType> = m3;
                super::prop_matrix_multiplication_distributive(m1, m2, m3)?
            }

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

            #[test]
            fn prop_zero_matrix_times_matrix_equals_zero_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_zero_matrix_times_matrix_equals_zero_matrix(m)?
            }

            #[test]
            fn prop_matrix_multiplication_identity(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_multiplication_identity(m)?
            }
        }
    }
    }
}

exact_multiplication_props!(matrix1x1_i32_matrix_multiplication_props, Matrix1x1, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);
exact_multiplication_props!(matrix2x2_i32_matrix_multiplication_props, Matrix2x2, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);
exact_multiplication_props!(matrix3x3_i32_matrix_multiplication_props, Matrix3x3, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);
exact_multiplication_props!(matrix4x4_i32_matrix_multiplication_props, Matrix4x4, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);


macro_rules! approx_multiplication_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_multiplication_identity(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_multiplication_identity(m)?
            }

            #[test]
            fn prop_zero_matrix_times_matrix_equals_zero_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_zero_matrix_times_matrix_equals_zero_matrix(m)?
            }

            #[test]
            fn prop_zero_times_matrix_equals_zero_matrix(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_zero_times_matrix_equals_zero_matrix(m)?
            }
        }
    }
    }
}

approx_multiplication_props!(matrix1x1_f64_matrix_multiplication_props, Matrix1x1, f64, strategy_matrix_f64_any);
approx_multiplication_props!(matrix2x2_f64_matrix_multiplication_props, Matrix2x2, f64, strategy_matrix_f64_any);
approx_multiplication_props!(matrix3x3_f64_matrix_multiplication_props, Matrix3x3, f64, strategy_matrix_f64_any);
approx_multiplication_props!(matrix4x4_f64_matrix_multiplication_props, Matrix4x4, f64, strategy_matrix_f64_any);


macro_rules! exact_transposition_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
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

            #[test]
            fn prop_transpose_product(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_transpose_product(m1, m2)?
            }
        }
    }
    }
}

exact_transposition_props!(matrix1x1_i32_transposition_props, Matrix1x1, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);
exact_transposition_props!(matrix2x2_i32_transposition_props, Matrix2x2, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);
exact_transposition_props!(matrix3x3_i32_transposition_props, Matrix3x3, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);
exact_transposition_props!(matrix4x4_i32_transposition_props, Matrix4x4, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);


macro_rules! approx_transposition_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
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

            #[test]
            fn prop_transpose_product(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_transpose_product(m1, m2)?
            }
        }
    }
    }
}

approx_transposition_props!(matrix1x1_f64_transposition_props, Matrix1x1, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_transposition_props!(matrix2x2_f64_transposition_props, Matrix2x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_transposition_props!(matrix3x3_f64_transposition_props, Matrix3x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_transposition_props!(matrix4x4_f64_transposition_props, Matrix4x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);


macro_rules! swap_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $UpperBound:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
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

swap_props!(matrix1x1_swap_props, Matrix1x1, i32, strategy_matrix_i32_any, 1);
swap_props!(matrix2x2_swap_props, Matrix2x2, i32, strategy_matrix_i32_any, 2);
swap_props!(matrix3x3_swap_props, Matrix3x3, i32, strategy_matrix_i32_any, 3);
swap_props!(matrix4x4_swap_props, Matrix4x4, i32, strategy_matrix_i32_any, 4);


macro_rules! exact_dot_product_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_dot_product_nonnegative(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_dot_product_nonnegative(m)?
            }

            #[test]
            fn prop_matrix_dot_product_nonzero(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_dot_product_nonzero(m)?
            }

            #[test]
            fn prop_matrix_dot_product_left_bilinear(
                m1 in super::$Generator(),
                m2 in super::$Generator(),
                m3 in super::$Generator()
            ) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                let m3: super::$MatrixN<$ScalarType> = m3;
                super::prop_matrix_dot_product_left_bilinear(m1, m2, m3)?
            }

            #[test]
            fn prop_matrix_dot_product_right_bilinear(
                m1 in super::$Generator(),
                m2 in super::$Generator(),
                m3 in super::$Generator()
            ) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                let m3: super::$MatrixN<$ScalarType> = m3;
                super::prop_matrix_dot_product_right_bilinear(m1, m2, m3)?
            }

            #[test]
            fn prop_matrix_dot_product_homogeneous(
                c1 in super::$ScalarGen(), 
                c2 in super::$ScalarGen(), 
                m1 in super::$Generator(),
                m2 in super::$Generator(),
            ) {
                let c1: $ScalarType = c1;
                let c2: $ScalarType = c2;
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_dot_product_homogeneous(c1, c2, m1, m2)?
            }
        }
    }
    }
}

exact_dot_product_props!(matrix1x1_i32_dot_product_props, Matrix1x1, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_dot_product_props!(matrix2x2_i32_dot_product_props, Matrix2x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_dot_product_props!(matrix3x3_i32_dot_product_props, Matrix3x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_dot_product_props!(matrix4x4_i32_dot_product_props, Matrix4x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);

exact_dot_product_props!(matrix2x3_i32_dot_product_props, Matrix2x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_dot_product_props!(matrix3x2_i32_dot_product_props, Matrix3x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_dot_product_props!(matrix2x4_i32_dot_product_props, Matrix2x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_dot_product_props!(matrix4x2_i32_dot_product_props, Matrix4x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_dot_product_props!(matrix3x4_i32_dot_product_props, Matrix3x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_dot_product_props!(matrix4x3_i32_dot_product_props, Matrix4x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);


macro_rules! exact_l1_norm_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_l1_norm_nonnegative(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_l1_norm_nonnegative(m)?
            }

            #[test]
            fn prop_matrix_l1_norm_point_separating1(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_l1_norm_point_separating1(m)?
            }

            #[test]
            fn prop_matrix_l1_norm_point_separating2(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_l1_norm_point_separating2(m1, m2)?
            }

            #[test]
            fn prop_matrix_l1_norm_homogeneous(c in super::$ScalarGen(), m in super::$Generator()) {
                let c: $ScalarType = c;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_l1_norm_homogeneous(c, m)?
            }

            #[test]
            fn prop_matrix_l1_norm_triangle_inequality(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_l1_norm_triangle_inequality(m1, m2)?
            }
        }
    }
    }
}

exact_l1_norm_props!(matrix1x1_i32_l1_norm_props, Matrix1x1, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_l1_norm_props!(matrix2x2_i32_l1_norm_props, Matrix2x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_l1_norm_props!(matrix3x3_i32_l1_norm_props, Matrix3x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_l1_norm_props!(matrix4x4_i32_l1_norm_props, Matrix4x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);

exact_l1_norm_props!(matrix2x3_i32_l1_norm_props, Matrix2x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_l1_norm_props!(matrix3x2_i32_l1_norm_props, Matrix3x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_l1_norm_props!(matrix2x4_i32_l1_norm_props, Matrix2x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_l1_norm_props!(matrix4x2_i32_l1_norm_props, Matrix4x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_l1_norm_props!(matrix3x4_i32_l1_norm_props, Matrix3x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_l1_norm_props!(matrix4x3_i32_l1_norm_props, Matrix4x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);


macro_rules! approx_l1_norm_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_l1_norm_nonnegative(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_l1_norm_nonnegative(m)?
            }

            #[test]
            fn prop_approx_matrix_l1_norm_point_separating1(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_approx_matrix_l1_norm_point_separating1(m, 1e-10)?
            }

            #[test]
            fn prop_approx_matrix_l1_norm_point_separating2(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_approx_matrix_l1_norm_point_separating2(m1, m2, 1e-10)?
            }
        }
    }
    }
}

approx_l1_norm_props!(matrix1x1_f64_l1_norm_props, Matrix1x1, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_l1_norm_props!(matrix2x2_f64_l1_norm_props, Matrix2x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_l1_norm_props!(matrix3x3_f64_l1_norm_props, Matrix3x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_l1_norm_props!(matrix4x4_f64_l1_norm_props, Matrix4x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);

approx_l1_norm_props!(matrix2x3_f64_l1_norm_props, Matrix2x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_l1_norm_props!(matrix3x2_f64_l1_norm_props, Matrix3x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_l1_norm_props!(matrix2x4_f64_l1_norm_props, Matrix2x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_l1_norm_props!(matrix4x2_f64_l1_norm_props, Matrix4x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_l1_norm_props!(matrix3x4_f64_l1_norm_props, Matrix3x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_l1_norm_props!(matrix4x3_f64_l1_norm_props, Matrix4x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);


macro_rules! exact_linf_norm_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_linf_norm_nonnegative(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_linf_norm_nonnegative(m)?
            }

            #[test]
            fn prop_matrix_linf_norm_point_separating1(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_linf_norm_point_separating1(m)?
            }

            #[test]
            fn prop_matrix_linf_norm_point_separating2(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_linf_norm_point_separating2(m1, m2)?
            }

            #[test]
            fn prop_matrix_linf_norm_homogeneous(c in super::$ScalarGen(), m in super::$Generator()) {
                let c: $ScalarType = c;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_linf_norm_homogeneous(c, m)?
            }

            #[test]
            fn prop_matrix_linf_norm_triangle_inequality(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_linf_norm_triangle_inequality(m1, m2)?
            }
        }
    }
    }
}

exact_linf_norm_props!(matrix1x1_i32_linf_norm_props, Matrix1x1, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_linf_norm_props!(matrix2x2_i32_linf_norm_props, Matrix2x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_linf_norm_props!(matrix3x3_i32_linf_norm_props, Matrix3x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_linf_norm_props!(matrix4x4_i32_linf_norm_props, Matrix4x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);

exact_linf_norm_props!(matrix2x3_i32_linf_norm_props, Matrix2x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_linf_norm_props!(matrix3x2_i32_linf_norm_props, Matrix3x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_linf_norm_props!(matrix2x4_i32_linf_norm_props, Matrix2x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_linf_norm_props!(matrix4x2_i32_linf_norm_props, Matrix4x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_linf_norm_props!(matrix3x4_i32_linf_norm_props, Matrix3x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_linf_norm_props!(matrix4x3_i32_linf_norm_props, Matrix4x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);


macro_rules! approx_linf_norm_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_linf_norm_nonnegative(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_linf_norm_nonnegative(m)?
            }

            #[test]
            fn prop_approx_matrix_linf_norm_point_separating1(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_approx_matrix_linf_norm_point_separating1(m, 1e-10)?
            }

            #[test]
            fn prop_approx_matrix_linf_norm_point_separating2(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_approx_matrix_linf_norm_point_separating2(m1, m2, 1e-10)?
            }
        }
    }
    }
}

approx_linf_norm_props!(matrix1x1_f64_linf_norm_props, Matrix1x1, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_linf_norm_props!(matrix2x2_f64_linf_norm_props, Matrix2x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_linf_norm_props!(matrix3x3_f64_linf_norm_props, Matrix3x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_linf_norm_props!(matrix4x4_f64_linf_norm_props, Matrix4x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);

approx_linf_norm_props!(matrix2x3_f64_linf_norm_props, Matrix2x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_linf_norm_props!(matrix3x2_f64_linf_norm_props, Matrix3x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_linf_norm_props!(matrix2x4_f64_linf_norm_props, Matrix2x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_linf_norm_props!(matrix4x2_f64_linf_norm_props, Matrix4x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_linf_norm_props!(matrix3x4_f64_linf_norm_props, Matrix3x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_linf_norm_props!(matrix4x3_f64_linf_norm_props, Matrix4x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);


macro_rules! exact_norm_squared_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_norm_squared_nonnegative(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_norm_squared_nonnegative(m)?
            }

            #[test]
            fn prop_matrix_norm_squared_point_separating1(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_norm_squared_point_separating1(m)?
            }

            #[test]
            fn prop_matrix_norm_squared_point_separating2(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_norm_squared_point_separating2(m1, m2)?
            }

            #[test]
            fn prop_matrix_norm_squared_homogeneous_squared(c in super::$ScalarGen(), m in super::$Generator()) {
                let c: $ScalarType = c;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_norm_squared_homogeneous_squared(c, m)?
            }
        }
    }
    }
}

exact_norm_squared_props!(matrix1x1_i32_norm_squared_props, Matrix1x1, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_norm_squared_props!(matrix2x2_i32_norm_squared_props, Matrix2x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_norm_squared_props!(matrix3x3_i32_norm_squared_props, Matrix3x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_norm_squared_props!(matrix4x4_i32_norm_squared_props, Matrix4x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);

exact_norm_squared_props!(matrix2x3_i32_norm_squared_props, Matrix2x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_norm_squared_props!(matrix3x2_i32_norm_squared_props, Matrix3x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_norm_squared_props!(matrix2x4_i32_norm_squared_props, Matrix2x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_norm_squared_props!(matrix4x2_i32_norm_squared_props, Matrix4x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_norm_squared_props!(matrix3x4_i32_norm_squared_props, Matrix3x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
exact_norm_squared_props!(matrix4x3_i32_norm_squared_props, Matrix4x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);


macro_rules! norm_synonym_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_magnitude_norm_synonyms(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_magnitude_norm_synonyms(m)?
            }
        }
    }
    }
}

norm_synonym_props!(matrix1x1_f64_norm_synonym_props, Matrix1x1, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
norm_synonym_props!(matrix2x2_f64_norm_synonym_props, Matrix2x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
norm_synonym_props!(matrix3x3_f64_norm_synonym_props, Matrix3x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
norm_synonym_props!(matrix4x4_f64_norm_synonym_props, Matrix4x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);

norm_synonym_props!(matrix2x3_f64_norm_synonym_props, Matrix2x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
norm_synonym_props!(matrix3x2_f64_norm_synonym_props, Matrix3x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
norm_synonym_props!(matrix2x4_f64_norm_synonym_props, Matrix2x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
norm_synonym_props!(matrix4x2_f64_norm_synonym_props, Matrix4x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
norm_synonym_props!(matrix3x4_f64_norm_synonym_props, Matrix3x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
norm_synonym_props!(matrix4x3_f64_norm_synonym_props, Matrix4x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);


macro_rules! norm_squared_synonym_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_magnitude_squared_norm_squared_synonyms(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_magnitude_squared_norm_squared_synonyms(m)?
            }
        }
    }
    }
}

norm_squared_synonym_props!(matrix1x1_i32_norm_squared_synonym_props, Matrix1x1, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
norm_squared_synonym_props!(matrix2x2_i32_norm_squared_synonym_props, Matrix2x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
norm_squared_synonym_props!(matrix3x3_i32_norm_squared_synonym_props, Matrix3x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
norm_squared_synonym_props!(matrix4x4_i32_norm_squared_synonym_props, Matrix4x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);

norm_squared_synonym_props!(matrix2x3_i32_norm_squared_synonym_props, Matrix2x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
norm_squared_synonym_props!(matrix3x2_i32_norm_squared_synonym_props, Matrix3x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
norm_squared_synonym_props!(matrix2x4_i32_norm_squared_synonym_props, Matrix2x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
norm_squared_synonym_props!(matrix4x2_i32_norm_squared_synonym_props, Matrix4x2, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
norm_squared_synonym_props!(matrix3x4_i32_norm_squared_synonym_props, Matrix3x4, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);
norm_squared_synonym_props!(matrix4x3_i32_norm_squared_synonym_props, Matrix4x3, i32, strategy_matrix_i32_norm, strategy_scalar_i32_any);


macro_rules! approx_norm_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_norm_nonnegative(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_norm_nonnegative(m)?
            }

            #[test]
            fn prop_approx_matrix_norm_point_separating1(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_approx_matrix_norm_point_separating1(m, 1e-10)?
            }

            #[test]
            fn prop_approx_matrix_norm_point_separating2(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_approx_matrix_norm_point_separating2(m1, m2, 1e-10)?
            }
        }
    }
    }
}

approx_norm_props!(matrix1x1_f64_norm_props, Matrix1x1, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_norm_props!(matrix2x2_f64_norm_props, Matrix2x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_norm_props!(matrix3x3_f64_norm_props, Matrix3x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_norm_props!(matrix4x4_f64_norm_props, Matrix4x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);

approx_norm_props!(matrix2x3_f64_norm_props, Matrix2x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_norm_props!(matrix3x2_f64_norm_props, Matrix3x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_norm_props!(matrix2x4_f64_norm_props, Matrix2x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_norm_props!(matrix4x2_f64_norm_props, Matrix4x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_norm_props!(matrix3x4_f64_norm_props, Matrix3x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_norm_props!(matrix4x3_f64_norm_props, Matrix4x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);


macro_rules! exact_trace_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_trace_linear(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_trace_linear(m1, m2)?
            }

            #[test]
            fn prop_matrix_trace_transpose(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_trace_transpose(m)?
            }

            #[test]
            fn prop_matrix_trace_product(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_matrix_trace_product(m1, m2)?
            }

            #[test]
            fn prop_matrix_trace_scalar_product(c in super::$ScalarGen(), m in super::$Generator()) {
                let c: $ScalarType = c;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_trace_scalar_product(c, m)?
            }
        }
    }
    }
}

exact_trace_props!(matrix1x1_i32_trace_props, Matrix1x1, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);
exact_trace_props!(matrix2x2_i32_trace_props, Matrix2x2, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);
exact_trace_props!(matrix3x3_i32_trace_props, Matrix3x3, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);
exact_trace_props!(matrix4x4_i32_trace_props, Matrix4x4, i32, strategy_matrix_i32_any, strategy_scalar_i32_any);


macro_rules! approx_trace_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_approx_matrix_trace_linear(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_approx_matrix_trace_linear(m1, m2, 1e-10, 1e-10)?
            }

            #[test]
            fn prop_matrix_trace_transpose(m in super::$Generator()) {
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_matrix_trace_transpose(m)?
            }

            #[test]
            fn prop_approx_matrix_trace_product(m1 in super::$Generator(), m2 in super::$Generator()) {
                let m1: super::$MatrixN<$ScalarType> = m1;
                let m2: super::$MatrixN<$ScalarType> = m2;
                super::prop_approx_matrix_trace_product(m1, m2, 1e-10, 1e-10)?
            }

            #[test]
            fn prop_approx_matrix_trace_scalar_product(c in super::$ScalarGen(), m in super::$Generator()) {
                let c: $ScalarType = c;
                let m: super::$MatrixN<$ScalarType> = m;
                super::prop_approx_matrix_trace_scalar_product(c, m, 1e-10, 1e-10)?
            }
        }
    }
    }
}

approx_trace_props!(matrix1x1_f64_trace_props, Matrix1x1, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_trace_props!(matrix2x2_f64_trace_props, Matrix2x2, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_trace_props!(matrix3x3_f64_trace_props, Matrix3x3, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);
approx_trace_props!(matrix4x4_f64_trace_props, Matrix4x4, f64, strategy_matrix_f64_any, strategy_scalar_f64_any);


