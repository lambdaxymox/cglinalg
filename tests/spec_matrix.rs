extern crate cglinalg;
extern crate num_traits;
extern crate proptest;


use proptest::prelude::*;
use cglinalg::{
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
    Scalar,
};


fn any_scalar<S>() -> impl Strategy<Value = S>
    where S: Scalar + Arbitrary
{
    any::<S>().prop_map(|scalar| {
        let modulus = num_traits::cast(100_000_000).unwrap();

        scalar % modulus
    })
}

fn any_matrix2<S>() -> impl Strategy<Value = Matrix2x2<S>> 
    where S: Scalar + Arbitrary
{
    any::<(S, S, S, S)>()
        .prop_map(|(c0r0, c0r1, c1r0, c1r1)| {
            let modulus = num_traits::cast(100_000_000).unwrap();
            let matrix = Matrix2x2::new(c0r0, c0r1, c1r0, c1r1);

            matrix % modulus
        })
}

fn any_matrix3<S>() -> impl Strategy<Value = Matrix3x3<S>>
    where S: Scalar + Arbitrary
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
    where S: Scalar + Arbitrary
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


/// Generate property tests for matrix addition over floating point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$MatrixN` denotes the name of the matrix type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of matrices.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_addition_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use approx::{
            relative_eq
        };
        use cglinalg::{
            $MatrixN,
            Zero,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// A zero matrix should act as the additive unit element for matrices
            /// over their underlying scalars. 
            ///
            /// Given a matrix `m` and a zero matrix `0`
            /// ```text
            /// 0 + m = m
            /// ```
            #[test]
            fn prop_matrix_additive_identity(m in $Generator::<$ScalarType>()) {
                let zero_mat = $MatrixN::zero();

                prop_assert_eq!(zero_mat + m, m);
            }
        
            /// A zero matrix should act as the additive unit element for matrices 
            /// over their underlying scalars. 
            ///
            /// Given a matrix `m` and a zero matrix `0`
            /// ```text
            /// m + 0 = m
            /// ```
            #[test]
            fn prop_vector_times_zero_equals_zero(m in $Generator::<$ScalarType>()) {
                let zero_mat = $MatrixN::zero();

                prop_assert_eq!(m + zero_mat, m);
            }

            /// Matrix addition over exact scalars is commutative.
            ///
            /// Given matrices `m1` and `m2`
            /// ```text
            /// m1 + m2 ~= m2 + m1
            /// ```
            #[test]
            fn prop_matrix_addition_approx_commutative(m1 in $Generator(), m2 in $Generator::<$ScalarType>()) {
                prop_assert!(relative_eq!(m1 + m2, m2 + m1, epsilon = $tolerance));
            }

            /// Matrix addition over exact scalars is associative.
            ///
            /// Given matrices `m1`, `m2`, and `m3`
            /// ```text
            /// (m1 + m2) + m3 ~= m1 + (m2 + m3)
            /// ```
            #[test]
            fn prop_matrix_addition_approx_associative(
                m1 in $Generator::<$ScalarType>(), 
                m2 in $Generator::<$ScalarType>(), m3 in $Generator::<$ScalarType>()) {

                prop_assert!(relative_eq!((m1 + m2) + m3, m1 + (m2 + m3), epsilon = $tolerance));
            }

            /// The sum of a matrix and it's additive inverse is the same as 
            /// subtracting the two matrices from each other.
            ///
            /// Given matrices `m1` and `m2`
            /// ```text
            /// m1 + (-m2) = m1 - m2
            /// ```
            #[test]
            fn prop_matrix_subtraction(
                m1 in $Generator::<$ScalarType>(), m2 in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!(m1 + (-m2), m1 - m2);
            }
        }
    }
    }
}

approx_addition_props!(matrix2_f64_addition_props, Matrix2x2, f64, any_matrix2, 1e-7);
approx_addition_props!(matrix3_f64_addition_props, Matrix3x3, f64, any_matrix3, 1e-7);
approx_addition_props!(matrix4_f64_addition_props, Matrix4x4, f64, any_matrix4, 1e-7);


/// Generate property tests for matrix addition over exact scalars. We define 
/// an exact scalar type as a type where scalar arithmetic is 
/// exact (e.g. integers).
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$MatrixN` denotes the name of the matrix type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of matrices.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_addition_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::{
            $MatrixN,
            Zero,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// A zero matrix should act as the additive unit element for matrices 
            /// over their underlying scalars. 
            ///
            /// Given a matrix `m`
            /// ```text
            /// 0 + m = m
            /// ```
            #[test]
            fn prop_matrix_additive_identity(m in $Generator::<$ScalarType>()) {
                let zero_mat = $MatrixN::zero();

                prop_assert_eq!(zero_mat + m, m);
            }
        
            /// A zero matrix should act as the additive unit element for matrices 
            /// over their underlying scalars. 
            ///
            /// Given a matrix `m`
            /// ```text
            /// m + 0 = m
            /// ```
            #[test]
            fn prop_vector_times_zero_equals_zero(m in $Generator::<$ScalarType>()) {
                let zero_mat = $MatrixN::zero();

                prop_assert_eq!(m + zero_mat, m);
            }

            /// Matrix addition over exact scalars is commutative.
            ///
            /// Given matrices `m1` and `m2`
            /// ```text
            /// m1 + m2 = m2 + m1
            /// ```
            #[test]
            fn prop_matrix_addition_commutative(m1 in $Generator(), m2 in $Generator::<$ScalarType>()) {
                prop_assert_eq!(m1 + m2, m2 + m1);
            }

            /// Matrix addition over exact scalars is associative.
            ///
            /// Given matrices `m1`, `m2`, and `m3`
            /// ```text
            /// (m1 + m2) + m3 = m1 + (m2 + m3)
            /// ```
            #[test]
            fn prop_matrix_addition_associative(
                m1 in $Generator(), m2 in $Generator(), m3 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((m1 + m2) + m3, m1 + (m2 + m3));
            }
        }
    }
    }
}

exact_addition_props!(matrix2_u32_addition_props, Matrix2x2, u32, any_matrix2);
exact_addition_props!(matrix2_i32_addition_props, Matrix2x2, i32, any_matrix2);
exact_addition_props!(matrix3_u32_addition_props, Matrix3x3, u32, any_matrix3);
exact_addition_props!(matrix3_i32_addition_props, Matrix3x3, i32, any_matrix3);
exact_addition_props!(matrix4_u32_addition_props, Matrix4x4, u32, any_matrix4);
exact_addition_props!(matrix4_i32_addition_props, Matrix4x4, i32, any_matrix4);


/// Generate property tests for the multiplication of matrices of floating point 
/// scalars by floating point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$MatrixN` denotes the name of the matrix type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of matrices.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_scalar_multiplication_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::{
            $MatrixN,
            Zero,
        };
        use super::{
            $Generator,
            $ScalarGen,
        };


        proptest! {
            /// Multiplication of a matrix by a scalar zero is the zero matrix.
            ///
            /// Given a matrix `m` and a zero scalar `0`
            /// ```text
            /// 0 * m = m * 0 = 0
            /// ```
            /// Note that we diverge from traditional formalisms of matrix arithmetic 
            /// in that we allow multiplication of matrices by scalars on the right-hand 
            /// side as well as left-hand side. 
            #[test]
            fn prop_zero_times_matrix_equals_zero_matrix(m in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();
                let zero_mat = $MatrixN::zero();

                prop_assert_eq!(zero * m, zero_mat);
                prop_assert_eq!(m * zero, zero_mat);
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
            #[test]
            fn prop_one_times_matrix_equals_matrix(m in $Generator::<$ScalarType>()) {
                let one: $ScalarType = num_traits::one();

                prop_assert_eq!(one * m, m);
                prop_assert_eq!(m * one, m);
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
            #[test]
            fn prop_negative_one_times_matrix_equals_negative_matrix(m in $Generator::<$ScalarType>()) {
                let one: $ScalarType = num_traits::one();
                let minus_one = -one;

                prop_assert_eq!(minus_one * m, -m);
            }

            /// Multiplication of a matrix by a scalar commutes with scalars.
            ///
            /// Given a matrix `m` and a scalar `c`
            /// ```text
            /// c * m ~= m * c
            /// ```
            /// Note that we diverse from traditional formalisms of matrix arithmetic 
            /// in that we allow multiplication of matrices by scalars on the left-hand 
            /// side as well as the right-hand side.
            #[test]
            fn prop_scalar_matrix_multiplication_commutative(
                c in $ScalarGen::<$ScalarType>(), m in $Generator::<$ScalarType>()) {

                prop_assert_eq!(c * m, m * c);
            }
        }
    }
    }
}

approx_scalar_multiplication_props!(
    matrix2_f64_scalar_multiplication_props, 
    Matrix2x2, f64, 
    any_matrix2, 
    any_scalar, 
    1e-7
);
approx_scalar_multiplication_props!(
    matrix3_f64_scalar_multiplication_props, 
    Matrix3x3, f64, 
    any_matrix3, 
    any_scalar, 
    1e-7
);
approx_scalar_multiplication_props!(
    matrix4_f64_scalar_multiplication_props, 
    Matrix4x4, 
    f64, 
    any_matrix4, 
    any_scalar, 
    1e-7
);


/// Generate property tests for the multiplication of matrices of integer scalars 
/// by integers.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$MatrixN` denotes the name of the matrix type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of matrices.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_scalar_multiplication_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::{
            $MatrixN, 
            Zero,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// Multiplication of matrices by scalars is compatible with matrix 
            /// addition.
            ///
            /// Given matrices `m1` and `m2`, and a scalar `c`
            /// ```text
            /// c * (m1 + m2) = c * m1 + c * m2
            /// ```
            #[test]
            fn prop_scalar_matrix_multiplication_compatible_addition(
                c in any::<$ScalarType>(),
                m1 in $Generator::<$ScalarType>(), m2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!(c * (m1 + m2), c * m1 + c * m2);
            }

            /// Multiplication of matrices by scalars is compatible with matrix 
            /// subtraction.
            ///
            /// Given matrices `m1` and `m2`, and a scalar `c`
            /// ```text
            /// c * (m1 - m2) = c * m1 - c * m2
            /// ```
            #[test]
            fn prop_scalar_matrix_multiplication_compatible_subtraction(
                c in any::<$ScalarType>(),
                m1 in $Generator::<$ScalarType>(), m2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!(c * (m1 - m2), c * m1 - c * m2);
            }

            /// Multiplication of a matrix by a scalar zero is the zero matrix.
            ///
            /// Given a matrix `m` and a zero scalar `0`
            /// ```text
            /// 0 * m = m * 0 = 0
            /// ```
            /// Note that we diverge from tradition formalisms of matrix arithmetic 
            /// in that we allow multiplication of matrices by scalars on the right-hand 
            /// side as well as left-hand side. 
            #[test]
            fn prop_zero_times_matrix_equals_zero_matrix(m in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();
                let zero_mat = $MatrixN::zero();

                prop_assert_eq!(zero * m, zero_mat);
                prop_assert_eq!(m * zero, zero_mat);
            }

            /// Multiplication of a matrix by a scalar one is the original matrix.
            ///
            /// Given a matrix `m` and a unit scalar `1`
            /// ```text
            /// 1 * m = m * 1 = m
            /// ```
            /// Note that we diverge from tradition formalisms of matrix arithmetic 
            /// in that we allow multiplication of matrices by scalars on the right-hand 
            /// side as well as left-hand side. 
            #[test]
            fn prop_one_times_matrix_equals_matrix(m in $Generator::<$ScalarType>()) {
                let one: $ScalarType = num_traits::one();

                prop_assert_eq!(one * m, m);
                prop_assert_eq!(m * one, m);
            }

            /// Multiplication of a matrix by a scalar commutes with scalars.
            ///
            /// Given a matrix `m` and a scalar `c`
            /// ```text
            /// c * m = m * c
            /// ```
            /// Note that we diverse from traditional formalisms of matrix arithmetic 
            /// in that we allow multiplication of matrices by scalars on the left-hand 
            /// side as well as the right-hand side.
            #[test]
            fn prop_scalar_matrix_multiplication_commutative(
                c in any::<$ScalarType>(), m in $Generator::<$ScalarType>()) {

                prop_assert_eq!(c * m, m * c);
            }

            /// Scalar multiplication of a matrix by scalars is compatible.
            ///
            /// Given a matrix `m` and scalars `a` and `b`
            /// ```text
            /// (a * b) * m = a * (b * m)
            /// ```
            #[test]
            fn prop_scalar_matrix_multiplication_compatible(
                a in any::<$ScalarType>(), 
                b in any::<$ScalarType>(),
                m in $Generator::<$ScalarType>()
            ) {
                prop_assert_eq!((a * b) * m, a * (b * m));
            }
        }
    }
    }
}

exact_scalar_multiplication_props!(matrix2_u32_scalar_multiplication_props, Matrix2x2, u32, any_matrix2);
exact_scalar_multiplication_props!(matrix2_i32_scalar_multiplication_props, Matrix2x2, i32, any_matrix2);
exact_scalar_multiplication_props!(matrix3_u32_scalar_multiplication_props, Matrix3x3, u32, any_matrix3);
exact_scalar_multiplication_props!(matrix3_i32_scalar_multiplication_props, Matrix3x3, i32, any_matrix3);
exact_scalar_multiplication_props!(matrix4_u32_scalar_multiplication_props, Matrix4x4, u32, any_matrix4);
exact_scalar_multiplication_props!(matrix4_i32_scalar_multiplication_props, Matrix4x4, i32, any_matrix4);


/// Generate property tests for the multiplication of matrices of floating 
/// point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$MatrixN` denotes the name of the matrix type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of matrices.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! approx_multiplication_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::{
            $MatrixN,
            Identity,
            Zero,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// Multiplication of a matrix by a scalar commutes with scalars.
            ///
            /// Given a matrix `m` and a scalar `c`
            /// ```text
            /// c * m = m * c
            /// ```
            /// Note that we diverse from traditional formalisms of matrix arithmetic 
            /// in that we allow multiplication of matrices by scalars on the left-hand 
            /// side as well as the right-hand side.
            #[test]
            fn prop_scalar_matrix_multiplication_commutative(
                c in any::<$ScalarType>(), m in $Generator::<$ScalarType>()) {

                prop_assert_eq!(c * m, m * c);
            }

            /// Matrices over a set of floating point scalars have a 
            /// multiplicative identity.
            /// 
            /// Given a matrix `m` there is a matrix `identity` such that
            /// ```text
            /// m * identity = identity * m = m
            /// ```
            #[test]
            fn prop_matrix_multiplication_identity(m in $Generator::<$ScalarType>()) {
                let identity = $MatrixN::identity();

                prop_assert_eq!(m * identity, m);
                prop_assert_eq!(identity * m, m);
            }

            /// Multiplication of a matrix by a scalar zero is the zero matrix.
            ///
            /// Given a matrix `m` and a zero scalar `0`
            /// ```text
            /// 0 * m = m * 0 = 0
            /// ```
            /// Note that we diverge from tradition formalisms of matrix arithmetic 
            /// in that we allow multiplication of matrices by scalars on the right-hand 
            /// side as well as left-hand side. 
            #[test]
            fn prop_zero_times_matrix_equals_zero_matrix(m in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();
                let zero_mat = $MatrixN::zero();

                prop_assert_eq!(zero * m, zero_mat);
                prop_assert_eq!(m * zero, zero_mat);
            }
        }
    }
    }
}

approx_multiplication_props!(matrix2_f64_matrix_multiplication_props, Matrix2x2, f64, any_matrix2);
approx_multiplication_props!(matrix3_f64_matrix_multiplication_props, Matrix3x3, f64, any_matrix3);
approx_multiplication_props!(matrix4_f64_matrix_multiplication_props, Matrix4x4, f64, any_matrix4);


/// Generate property tests for the multiplication of matrices of floating 
/// point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$MatrixN` denotes the name of the matrix type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of matrices.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_multiplication_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::{
            $MatrixN,
            Identity
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// Matrix multiplication is associative.
            ///
            /// Given matrices `m1`, `m2`, and `m3`
            /// ```text
            /// (m1 * m2) * m3 = m1 * (m2 * m3)
            /// ```
            #[test]
            fn prop_matrix_multiplication_associative(
                m1 in $Generator::<$ScalarType>(),
                m2 in $Generator::<$ScalarType>(), m3 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((m1 * m2) * m3, m1* (m2 * m3));
            }

            /// Matrix multiplication is distributive over matrix addition.
            ///
            /// Given matrices `m1`, `m2`, and `m3`
            /// ```text
            /// m1 * (m2 + m3) = m1 * m2 + m1 * m3
            /// ```
            #[test]
            fn prop_matrix_multiplication_distributive(                
                m1 in $Generator::<$ScalarType>(),
                m2 in $Generator::<$ScalarType>(), m3 in $Generator::<$ScalarType>()) {

                prop_assert_eq!(m1 * (m2 + m3), m1 * m2 + m1 * m3);
            }

            /// Matrix multiplication is compatible with scalar multiplication.
            ///
            /// Given matrices `m1` and `m2` and a scalar `c`
            /// ```text
            /// c * (m1 * m2) = (c * m1) * m2 = m1 * (c * m2)
            /// ```
            #[test]
            fn prop_matrix_multiplication_compatible_with_scalar_multiplication(
                c in any::<$ScalarType>(),
                m1 in $Generator::<$ScalarType>(), m2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!(c * (m1 * m2), (c * m1) * m2);
                prop_assert_eq!((c * m1) * m2, m1 * (c * m2));
            }

            /// Matrix multiplication is compatible with scalar multiplication.
            ///
            /// Given a matrix `m`, scalars `c1` and `c2`
            /// ```text
            /// (c1 * c2) * m = c1 * (c2 * m)
            /// ```
            #[test]
            fn prop_matrix_multiplication_compatible_with_scalar_multiplication1(
                c1 in any::<$ScalarType>(), c2 in any::<$ScalarType>(), m in $Generator::<$ScalarType>()) {

                prop_assert_eq!((c1 * c2) * m, c1 * (c2 * m));
            }

            /// Matrices over a set of floating point scalars have a 
            /// multiplicative identity.
            /// 
            /// Given a matrix `m` there is a matrix `identity` such that
            /// ```text
            /// m * identity = identity * m = m
            /// ```
            #[test]
            fn prop_matrix_multiplication_identity(m in $Generator::<$ScalarType>()) {
                let identity = $MatrixN::identity();

                prop_assert_eq!(m * identity, m);
                prop_assert_eq!(identity * m, m);
            }
        }
    }
    }
}

exact_multiplication_props!(matrix2_u32_matrix_multiplication_props, Matrix2x2, u32, any_matrix2);
exact_multiplication_props!(matrix2_i32_matrix_multiplication_props, Matrix2x2, i32, any_matrix2);
exact_multiplication_props!(matrix3_u32_matrix_multiplication_props, Matrix3x3, u32, any_matrix3);
exact_multiplication_props!(matrix3_i32_matrix_multiplication_props, Matrix3x3, i32, any_matrix3);
exact_multiplication_props!(matrix4_u32_matrix_multiplication_props, Matrix4x4, u32, any_matrix4);
exact_multiplication_props!(matrix4_i32_matrix_multiplication_props, Matrix4x4, i32, any_matrix4);


/// Generate property tests for the transposition of matrices over floating 
/// point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$MatrixN` denotes the name of the matrix type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of matrices.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_transposition_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::{
            Matrix,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The double transpose of a matrix is the original matrix.
            ///
            /// Given a matrix `m`
            /// ```text
            /// transpose(transpose(m)) = m
            /// ```
            #[test]
            fn prop_matrix_transpose_transpose_equals_matrix(m in $Generator::<$ScalarType>()) {
                prop_assert_eq!(m.transpose().transpose(), m);
            }

            /// The transposition operation is linear.
            /// 
            /// Given matrices `m1` and `m2`
            /// ```text
            /// transpose(m1 + m2) = transpose(m1) + transpose(m2)
            /// ```
            #[test]
            fn prop_transpose_linear(
                m1 in $Generator::<$ScalarType>(), m2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((m1 + m2).transpose(), m1.transpose() + m2.transpose());
            }

            /// Scalar multiplication of a matrix and a scalar commutes with 
            /// transposition.
            /// 
            /// Given a matrix `m` and a scalar `c`
            /// ```text
            /// transpose(c * m) = c * transpose(m)
            /// ```
            #[test]
            fn prop_transpose_scalar_multiplication(
                c in any::<$ScalarType>(), m in $Generator::<$ScalarType>()) {

                prop_assert_eq!((c * m).transpose(), c * m.transpose());
            }

            /// The transpose of the product of two matrices equals the product 
            /// of the transposes of the two matrices swapped.
            /// 
            /// Given matrices `m1` and `m2`
            /// ```text
            /// transpose(m1 * m2) = transpose(m2) * transpose(m1)
            /// ```
            #[test]
            fn prop_transpose_product(
                m1 in $Generator::<$ScalarType>(), m2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((m1 * m2).transpose(), m2.transpose() * m1.transpose());
            }
        }
    }
    }
}

approx_transposition_props!(matrix2_f64_transposition_props, Matrix2x2, f64, any_matrix2, 1e-7);
approx_transposition_props!(matrix3_f64_transposition_props, Matrix3x3, f64, any_matrix3, 1e-7);
approx_transposition_props!(matrix4_f64_transposition_props, Matrix4x4, f64, any_matrix4, 1e-7);


/// Generate property tests for the transposition of matrices over floating 
/// point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$MatrixN` denotes the name of the matrix type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of matrices.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_transposition_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::{
            Matrix,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The double transpose of a matrix is the original matrix.
            ///
            /// Given a matrix `m`
            /// ```text
            /// transpose(transpose(m)) = m
            /// ```
            #[test]
            fn prop_matrix_transpose_transpose_equals_matrix(m in $Generator::<$ScalarType>()) {
                prop_assert_eq!(m.transpose().transpose(), m);
            }

            /// The transposition operation is linear.
            /// 
            /// Given matrices `m1` and `m2`
            /// ```text
            /// transpose(m1 + m2) = transpose(m1) + transpose(m2)
            /// ```
            #[test]
            fn prop_transpose_linear(
                m1 in $Generator::<$ScalarType>(), m2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((m1 + m2).transpose(), m1.transpose() + m2.transpose());
            }

            /// Scalar multiplication of a matrix and a scalar commutes with transposition.
            /// 
            /// Given a matrix `m` and a scalar `c`
            /// ```text
            /// transpose(c * m) = c * transpose(m)
            /// ```
            #[test]
            fn prop_transpose_scalar_multiplication(
                c in any::<$ScalarType>(), m in $Generator::<$ScalarType>()) {

                prop_assert_eq!((c * m).transpose(), c * m.transpose());
            }

            /// The transpose of the product of two matrices equals the product of the transposes
            /// of the two matrices swapped.
            /// 
            /// Given matrices `m1` and `m2`
            /// ```text
            /// transpose(m1 * m2) = transpose(m2) * transpose(m1)
            /// ```
            #[test]
            fn prop_transpose_product(
                m1 in $Generator::<$ScalarType>(), m2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((m1 * m2).transpose(), m2.transpose() * m1.transpose());
            }
        }
    }
    }
}

exact_transposition_props!(matrix2_u32_transposition_props, Matrix2x2, u32, any_matrix2);
exact_transposition_props!(matrix2_i32_transposition_props, Matrix2x2, i32, any_matrix2);
exact_transposition_props!(matrix3_u32_transposition_props, Matrix3x3, u32, any_matrix3);
exact_transposition_props!(matrix3_i32_transposition_props, Matrix3x3, i32, any_matrix3);
exact_transposition_props!(matrix4_u32_transposition_props, Matrix4x4, u32, any_matrix4);
exact_transposition_props!(matrix4_i32_transposition_props, Matrix4x4, i32, any_matrix4);


/// Generate property tests for the swap operations on matrices.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$MatrixN` denotes the name of the matrix type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of matrices.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! swap_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $UpperBound:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::{
            Matrix,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// Swapping rows is commutative in the row arguments.
            ///
            /// Given a matrix `m`, and rows `row1` and `row2`
            /// ```text
            /// m.swap_rows(row1, row2) = m.swap_rows(row2, row1)
            /// ```
            #[test]
            fn prop_swap_rows_commutative(
                m in $Generator::<$ScalarType>(), 
                row1 in 0..$UpperBound as usize, row2 in 0..$UpperBound as usize) {
                
                let mut m1 = m;
                let mut m2 = m;
                m1.swap_rows(row1, row2);
                m2.swap_rows(row2, row1);

                prop_assert_eq!(m1, m2);
            }

            /// Swapping the same row in both arguments is the identity map.
            ///
            /// Given a matrix `m`, and a row `row`
            /// ```text
            /// m.swap_rows(row, row) = m
            /// ```
            #[test]
            fn prop_swap_identical_rows_identity(
                m in $Generator::<$ScalarType>(), row in 0..$UpperBound as usize) {

                let mut m1 = m;
                m1.swap_rows(row, row);

                prop_assert_eq!(m1, m);
            }

            /// Swapping the same two rows twice in succession yields the original 
            /// matrix.
            ///
            /// Given a matrix `m`, and rows `row1` and `row2`
            /// ```text
            /// m.swap_rows(row1, row2).swap_rows(row1, row2) = m
            /// ```
            #[test]
            fn prop_swap_rows_twice_is_identity(
                m in $Generator::<$ScalarType>(), 
                row1 in 0..$UpperBound as usize, row2 in 0..$UpperBound as usize) {
                
                let mut m1 = m;
                m1.swap_rows(row1, row2);
                m1.swap_rows(row1, row2);

                prop_assert_eq!(m1, m);
            }

            /// Swapping columns is commutative in the column arguments.
            ///
            /// Given a matrix `m`, and columns `col1` and `col2`
            /// ```text
            /// m.swap_columns(col1, col2) = m.swap_columns(col2, col1)
            /// ```
            #[test]
            fn prop_swap_columns_commutative(
                m in $Generator::<$ScalarType>(), 
                col1 in 0..$UpperBound as usize, col2 in 0..$UpperBound as usize) {
                
                let mut m1 = m;
                let mut m2 = m;
                m1.swap_columns(col1, col2);
                m2.swap_columns(col2, col1);

                prop_assert_eq!(m1, m2);
            }

            /// Swapping the same column in both arguments is the identity map.
            ///
            /// Given a matrix `m`, and a column `col`
            /// ```text
            /// m.swap_columns(col, col) = m
            /// ```
            #[test]
            fn prop_swap_identical_columns_is_identity(
                m in $Generator::<$ScalarType>(), col in 0..$UpperBound as usize) {

                let mut m1 = m;
                m1.swap_columns(col, col);

                prop_assert_eq!(m1, m);
            }

            /// Swapping the same two columns twice in succession yields the 
            /// original matrix.
            ///
            /// Given a matrix `m`, and columns `col1` and `col2`
            /// ```text
            /// m.swap_columns(col1, col2).swap_columns(col1, col2) = m
            /// ```
            #[test]
            fn prop_swap_columns_twice_is_identity(
                m in $Generator::<$ScalarType>(), 
                col1 in 0..$UpperBound as usize, col2 in 0..$UpperBound as usize) {
                
                let mut m1 = m;
                m1.swap_columns(col1, col2);
                m1.swap_columns(col1, col2);

                prop_assert_eq!(m1, m);
            }

            /// Swapping elements is commutative in the arguments.
            ///
            /// Given a matrix `m`, and elements `(col1, row1)` and `(col2, row2)`
            /// ```text
            /// m.swap_elements((col1, row1), (col2, row2)) = m.swap_elements((col2, row2), (col1, row1))
            /// ```
            #[test]
            fn prop_swap_elements_commutative(
                m in $Generator::<$ScalarType>(), 
                col1 in 0..$UpperBound as usize, row1 in 0..$UpperBound as usize,
                col2 in 0..$UpperBound as usize, row2 in 0..$UpperBound as usize) {
                
                let mut m1 = m;
                let mut m2 = m;
                m1.swap_elements((col1, row1), (col2, row2));
                m2.swap_elements((col2, row2), (col1, row1));

                prop_assert_eq!(m1, m2);
            }

            /// Swapping the same element in both arguments is the identity map.
            ///
            /// Given a matrix `m`, and an element index `(col, row)`
            /// ```text
            /// m.swap_elements((col, row), (col, row)) = m
            /// ```
            #[test]
            fn prop_swap_identical_elements_is_identity(
                m in $Generator::<$ScalarType>(), 
                col in 0..$UpperBound as usize, row in 0..$UpperBound as usize) {

                let mut m1 = m;
                m1.swap_elements((col, row), (col, row));

                prop_assert_eq!(m1, m);
            }

            /// Swapping the same two elements twice in succession yields the 
            /// original matrix.
            ///
            /// Given a matrix `m`, and elements `(col1, row1)` and `(col2, row2)`
            /// ```text
            /// m.swap_elements((col1, row1), (col2, row2)).swap_elements((col1, row1), (col2, row2)) = m
            /// ```
            #[test]
            fn prop_swap_elements_twice_is_identity(
                m in $Generator::<$ScalarType>(), 
                col1 in 0..$UpperBound as usize, row1 in 0..$UpperBound as usize, 
                col2 in 0..$UpperBound as usize, row2 in 0..$UpperBound as usize) {
                
                let mut m1 = m;
                m1.swap_elements((col1, row1), (col2, row2));
                m1.swap_elements((col1, row1), (col2, row2));

                prop_assert_eq!(m1, m);
            }
        }
    }
    }
}

swap_props!(matrix2_swap_props, Matrix2x2, isize, any_matrix2, 2);
swap_props!(matrix3_swap_props, Matrix3x3, isize, any_matrix3, 3);
swap_props!(matrix4_swap_props, Matrix4x4, isize, any_matrix4, 4);

