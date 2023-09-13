extern crate cglinalg_numeric;
extern crate cglinalg_core;
extern crate proptest;


use cglinalg_numeric::{
    SimdScalarSigned,
};
use cglinalg_core::{
    Matrix,
    Matrix1x1,
    Matrix1x2,
    Matrix1x3,
    Matrix1x4,
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
    Matrix2x3,
    Matrix3x2,
    Matrix2x4,
    Matrix4x2,
    Matrix3x4,
    Matrix4x3,
    Vector,
    Vector1,
    Vector2,
    Vector3,
    Vector4,
};

use proptest::prelude::*;


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

fn strategy_vector_signed_from_abs_range<S, const N: usize>(min_value: S, max_value: S) -> impl Strategy<Value = Vector<S, N>>
where
    S: SimdScalarSigned + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarSigned
    {
        min_value + (value % (max_value - min_value))
    }

    fn rescale_vector<S, const N: usize>(value: Vector<S, N>, min_value: S, max_value: S) -> Vector<S, N>
    where
        S: SimdScalarSigned
    {
        value.map(|element| rescale(element, min_value, max_value))
    }

    any::<[S; N]>().prop_map(move |array| {
        let vector = Vector::from(array);
        
        rescale_vector(vector, min_value, max_value)
    })
    .no_shrink()
}

fn strategy_matrix_i32_any<const R: usize, const C: usize>() -> impl Strategy<Value = Matrix<i32, R, C>> {
    let min_value = 0_i32;
    let max_value = 1_000_000_000_i32;

    strategy_matrix_signed_from_abs_range(min_value, max_value)
}

fn strategy_vector_i32_any<const N: usize>() -> impl Strategy<Value = Vector<i32, N>> {
    let min_value = 0_i32;
    let max_value = 1_000_000_000_i32;

    strategy_vector_signed_from_abs_range(min_value, max_value)
}

/// Matrix/vector multiplication is left distributive.
/// 
/// Given matrices `m1` and `m2`, and a vector `v`
/// ```text
/// (m1 + m2) * v == m1 * v + m2 * v
/// ```
fn prop_matrix_times_vector_left_distributive<S, const R: usize, const C: usize>(
    m1: Matrix<S, R, C>, 
    m2: Matrix<S, R, C>, 
    v: Vector<S, C>
) -> Result<(), TestCaseError> 
where
    S: SimdScalarSigned
{
    let lhs = (m1 + m2) * v;
    let rhs = m1 * v + m2 * v;

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// Matrix/vector multiplication is right distributive.
/// 
/// Given a matrix `m` and vectors `v1` and `v2`
/// ```text
/// m * (v1 + v2) == m * v1 + m * v2
/// ```
fn prop_matrix_times_vector_right_distributive<S, const R: usize, const C: usize>(
    m: Matrix<S, R, C>, 
    v1: Vector<S, C>, 
    v2: Vector<S, C>
) -> Result<(), TestCaseError> 
where
    S: SimdScalarSigned
{
    let lhs = m * (v1 + v2);
    let rhs = m * v1 + m * v2;

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// A row vector times a column vector equals the dot product.
/// 
/// Given a row vector `m` and a column vector `v`
/// ```text
/// m * v == dot(transpose(m), v)
/// ```
fn prop_matrix_times_vector_dot_product<S, const C: usize>(m: Matrix<S, 1, C>, v: Vector<S, C>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let m_times_v = m * v;
    let m_tr = m.transpose();
    let lhs = m_times_v[0];
    let rhs = m_tr[0].dot(&v);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}


macro_rules! exact_multiplication_props {
    ($TestModuleName:ident, $MatrixType:ident, $VectorType:ident, $ScalarType:ty, $MatrixGen:ident, $VectorGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_times_vector_left_distributive(
                m1 in super::$MatrixGen(), 
                m2 in super::$MatrixGen(), 
                v in super::$VectorGen()
            ) {
                let m1: super::$MatrixType<$ScalarType> = m1;
                let m2: super::$MatrixType<$ScalarType> = m2;
                let v: super::$VectorType<$ScalarType> = v;
                super::prop_matrix_times_vector_left_distributive(m1, m2, v)?
            }

            #[test]
            fn prop_matrix_times_vector_right_distributive(
                m in super::$MatrixGen(),
                v1 in super::$VectorGen(),
                v2 in super::$VectorGen()
            ) {
                let m: super::$MatrixType<$ScalarType> = m;
                let v1: super::$VectorType<$ScalarType> = v1;
                let v2: super::$VectorType<$ScalarType> = v2;
                super::prop_matrix_times_vector_right_distributive(m, v1, v2)?
            }
        }
    }
    }
}

exact_multiplication_props!(matrix1x1_i32_vector1_i32_props, Matrix1x1, Vector1, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_multiplication_props!(matrix2x2_i32_vector2_i32_props, Matrix2x2, Vector2, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_multiplication_props!(matrix3x3_i32_vector3_i32_props, Matrix3x3, Vector3, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_multiplication_props!(matrix4x4_i32_vector4_i32_props, Matrix4x4, Vector4, i32, strategy_matrix_i32_any, strategy_vector_i32_any);

exact_multiplication_props!(matrix1x2_i32_vector2_i32_props, Matrix1x2, Vector2, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_multiplication_props!(matrix3x2_i32_vector2_i32_props, Matrix3x2, Vector2, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_multiplication_props!(matrix4x2_i32_vector2_i32_props, Matrix4x2, Vector2, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_multiplication_props!(matrix1x3_i32_vector3_i32_props, Matrix1x3, Vector3, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_multiplication_props!(matrix2x3_i32_vector3_i32_props, Matrix2x3, Vector3, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_multiplication_props!(matrix4x3_i32_vector3_i32_props, Matrix4x3, Vector3, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_multiplication_props!(matrix1x4_i32_vector4_i32_props, Matrix1x4, Vector4, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_multiplication_props!(matrix2x4_i32_vector4_i32_props, Matrix2x4, Vector4, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_multiplication_props!(matrix3x4_i32_vector4_i32_props, Matrix3x4, Vector4, i32, strategy_matrix_i32_any, strategy_vector_i32_any);


macro_rules! exact_row_vector_vector_dot_product_props {
    ($TestModuleName:ident, $MatrixType:ident, $VectorType:ident, $ScalarType:ty, $MatrixGen:ident, $VectorGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_matrix_times_vector_dot_product(m in super::$MatrixGen(), v in super::$VectorGen()) {
                let m: super::$MatrixType<$ScalarType> = m;
                let v: super::$VectorType<$ScalarType> = v;
                super::prop_matrix_times_vector_dot_product(m, v)?
            }
        }
    }
    }
}

exact_row_vector_vector_dot_product_props!(matrix1x1_i32_vector1_i32_dot_props, Matrix1x1, Vector1, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_row_vector_vector_dot_product_props!(matrix1x2_i32_vector2_i32_dot_props, Matrix1x2, Vector2, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_row_vector_vector_dot_product_props!(matrix1x3_i32_vector2_i32_dot_props, Matrix1x3, Vector3, i32, strategy_matrix_i32_any, strategy_vector_i32_any);
exact_row_vector_vector_dot_product_props!(matrix1x4_i32_vector2_i32_dot_props, Matrix1x4, Vector4, i32, strategy_matrix_i32_any, strategy_vector_i32_any);

