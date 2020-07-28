extern crate gdmath;
extern crate num_traits;
extern crate proptest;

use proptest::prelude::*;
use gdmath::{
    Matrix2,
    Matrix3,
    Matrix4,
    Matrix, 
    Scalar,
    ScalarFloat,
};

fn any_matrix2<S>() -> impl Strategy<Value = Matrix2<S>> where S: Scalar + Arbitrary {
    any::<(S, S, S, S)>().prop_map(
    |(c0r0, c0r1, c1r0, c1r1)| Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    )
}

fn any_matrix3<S>() -> impl Strategy<Value = Matrix3<S>> where S: Scalar + Arbitrary {
    any::<((S, S, S), (S, S, S), (S, S, S))>().prop_map(
        |((c0r0, c0r1, c0r2), (c1r0, c1r1, c1r2), (c2r0, c2r1, c2r2))| {
            Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
        }
    )
}

fn any_matrix4<S>() -> impl Strategy<Value = Matrix4<S>> where S: Scalar + Arbitrary {
    any::<((S, S, S, S), (S, S, S, S), (S, S, S, S), (S, S, S, S))>().prop_map(
        |((c0r0, c0r1, c0r2, c0r3), (c1r0, c1r1, c1r2, c1r3), (c2r0, c2r1, c2r2, c2r3), (c3r0, c3r1, c3r2, c3r3))| {
            Matrix4::new(c0r0,c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2,  c2r3, c3r0, c3r1, c3r2, c3r3)
        }
    )
}


/// Generate the properties for matrix addition over floating point scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$MatrixN` denotes the name of the matrix type.
/// `$ScalarType` denotes the underlying system of numbers that compose the matrices.
/// `$Generator` is the name of a function or closure for generating examples.
macro_rules! approx_addition_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::approx::relative_eq;
        use gdmath::{$MatrixN, Zero};

        proptest! {
            /// A zero matrix should act as the additive unit element for matrices over 
            /// their underlying scalars. 
            ///
            /// Given a matrix `m`
            /// ```
            /// 0 + m = m
            /// ```
            #[test]
            fn prop_matrix_additive_identity(m in super::$Generator::<$ScalarType>()) {
                let zero_mat = $MatrixN::zero();
                prop_assert_eq!(zero_mat + m, m);
            }
        
            /// A zero matrix should act as the additive unit element for matrices over 
            /// their underlying scalars. 
            ///
            /// Given a matrix `m`
            /// ```
            /// m + 0 = m
            /// ```
            #[test]
            fn prop_vector_times_zero_equals_zero(m in super::$Generator::<$ScalarType>()) {
                let zero_mat = $MatrixN::zero();
                prop_assert_eq!(m + zero_mat, m);
            }

            /// Matrix addition over exact scalars is commutative.
            ///
            /// Given matrices `m1` and `m2`
            /// ```
            /// m1 + m2 ~= m2 + m1
            /// ```
            #[test]
            fn prop_matrix_addition_approx_commutative(m1 in super::$Generator(), m2 in super::$Generator::<$ScalarType>()) {
                prop_assert!(relative_eq!(m1 + m2, m2 + m1, epsilon = $tolerance));
            }

            /// Matrix addition over exact scalars is associative.
            ///
            /// Given matrices `m1`, `m2`, and `m3`
            /// ```
            /// (m1 + m2) + m3 ~= m1 + (m2 + m3)
            /// ```
            #[test]
            fn prop_matrix_addition_approx_associative(
                m1 in super::$Generator(), m2 in super::$Generator(), m3 in super::$Generator::<$ScalarType>()) {

                prop_assert!(relative_eq!((m1 + m2) + m3, m1 + (m2 + m3), epsilon = $tolerance));
            }
        }
    }
    }
}

approx_addition_props!(matrix2_f64_addition_props, Matrix2, f64, any_matrix2, 1e-7);
approx_addition_props!(matrix3_f64_addition_props, Matrix3, f64, any_matrix3, 1e-7);
approx_addition_props!(matrix4_f64_addition_props, Matrix4, f64, any_matrix4, 1e-7);


/// Generate the properties for matrix addition over exact scalars. We define an exact
/// scalar type as a type where scalar arithmetic is exact (e.g. integers).
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$MatrixN` denotes the name of the matrix type.
/// `$ScalarType` denotes the underlying system of numbers that compose the matrices.
/// `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_addition_props {
    ($TestModuleName:ident, $MatrixN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{$MatrixN, Zero};

        proptest! {
            /// A zero matrix should act as the additive unit element for matrices over 
            /// their underlying scalars. 
            ///
            /// Given a matrix `m`
            /// ```
            /// 0 + m = m
            /// ```
            #[test]
            fn prop_matrix_additive_identity(m in super::$Generator::<$ScalarType>()) {
                let zero_mat = $MatrixN::zero();
                prop_assert_eq!(zero_mat + m, m);
            }
        
            /// A zero matrix should act as the additive unit element for matrices over 
            /// their underlying scalars. 
            ///
            /// Given a matrix `m`
            /// ```
            /// m + 0 = m
            /// ```
            #[test]
            fn prop_vector_times_zero_equals_zero(m in super::$Generator::<$ScalarType>()) {
                let zero_mat = $MatrixN::zero();
                prop_assert_eq!(m + zero_mat, m);
            }

            /// Matrix addition over exact scalars is commutative.
            ///
            /// Given matrices `m1` and `m2`
            /// ```
            /// m1 + m2 = m2 + m1
            /// ```
            #[test]
            fn prop_matrix_addition_commutative(m1 in super::$Generator(), m2 in super::$Generator::<$ScalarType>()) {
                prop_assert_eq!(m1 + m2, m2 + m1);
            }

            /// Matrix addition over exact scalars is associative.
            ///
            /// Given matrices `m1`, `m2`, and `m3`
            /// ```
            /// (m1 + m2) + m3 = m1 + (m2 + m3)
            /// ```
            #[test]
            fn prop_matrix_addition_associative(
                m1 in super::$Generator(), m2 in super::$Generator(), m3 in super::$Generator::<$ScalarType>()) {

                prop_assert_eq!((m1 + m2) + m3, m1 + (m2 + m3));
            }
        }
    }
    }
}

exact_addition_props!(matrix2_u32_addition_props, Matrix2, u32, any_matrix2);
exact_addition_props!(matrix2_i32_addition_props, Matrix2, i32, any_matrix2);
exact_addition_props!(matrix3_u32_addition_props, Matrix3, u32, any_matrix3);
exact_addition_props!(matrix3_i32_addition_props, Matrix3, i32, any_matrix3);
exact_addition_props!(matrix4_u32_addition_props, Matrix4, u32, any_matrix4);
exact_addition_props!(matrix4_i32_addition_props, Matrix4, i32, any_matrix4);

