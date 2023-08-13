extern crate cglinalg_core;
extern crate num_traits;
extern crate proptest;


use proptest::prelude::*;
use cglinalg_core::{
    Quaternion, 
    SimdScalar,
    SimdScalarFloat,
};


fn any_scalar<S>() -> impl Strategy<Value = S>
where
    S: SimdScalar + Arbitrary
{
    any::<S>().prop_map(|scalar| {
        let modulus = num_traits::cast(100_000_000).unwrap();

        scalar % modulus
    })
}

fn any_quaternion<S>() -> impl Strategy<Value = Quaternion<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S, S, S)>().prop_map(|(s, x, y, z)| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let quaternion = Quaternion::new(s, x, y, z);

        quaternion % modulus
    })
    .no_shrink()
}

fn any_quaternion_squared<S>() -> impl Strategy<Value = Quaternion<S>>
where
    S: SimdScalarFloat + Arbitrary
{
    any::<(S, S, S, S)>().prop_map(|(s, x, y, z)| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let quaternion = Quaternion::new(S::abs(s), S::abs(x), S::abs(y), S::abs(z));

        quaternion % modulus
    })
    .no_shrink()
}

/// Generate property tests for quaternion arithmetic over exact scalars. We 
/// define an exact scalar type as a type where scalar arithmetic is 
/// exact (e.g. integers).
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_arithmetic_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Quaternion,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// A scalar `0` times a quaternion should be a zero quaternion.
            ///
            /// Given a quaternion `q`, it satisfies
            /// ```text
            /// 0 * q = 0.
            /// ```
            #[test]
            fn prop_zero_times_quaternion_equals_zero(q in $Generator()) {
                let zero: $ScalarType = num_traits::zero();
                let zero_quat = Quaternion::zero();

                prop_assert_eq!(zero * q, zero_quat);
            }
        
            /// A scalar `0` times a quaternion should be zero.
            ///
            /// Given a quaternion `q`, it satisfies
            /// ```text
            /// q * 0 = 0
            /// ```
            #[test]
            fn prop_quaternion_times_zero_equals_zero(q in $Generator()) {
                let zero: $ScalarType = num_traits::zero();
                let zero_quat = Quaternion::zero();

                prop_assert_eq!(q * zero, zero_quat);
            }

            /// A zero quaternion should act as the additive unit element of a set 
            /// of quaternions.
            ///
            /// Given a quaternion `q`
            /// ```text
            /// q + 0 = q
            /// ```
            #[test]
            fn prop_quaternion_plus_zero_equals_quaternion(q in $Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();

                prop_assert_eq!(q + zero_quat, q);
            }

            /// A zero quaternion should act as the additive unit element of a set 
            /// of quaternions.
            ///
            /// Given a quaternion `q`
            /// ```text
            /// 0 + q = q
            /// ```
            #[test]
            fn prop_zero_plus_quaternion_equals_quaternion(q in $Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();

                prop_assert_eq!(zero_quat + q, q);
            }

            /// Multiplying a quaternion by a scalar `1` should give the original 
            /// quaternion.
            ///
            /// Given a quaternion `q`
            /// ```text
            /// 1 * q = q
            /// ```
            #[test]
            fn prop_one_times_quaternion_equal_quaternion(q in $Generator()) {
                let one: $ScalarType = num_traits::one();

                prop_assert_eq!(one * q, q);
            }

            /// Multiplying a quaternion by a scalar `1` should give the original 
            /// quaternion.
            ///
            /// Given a quaternion `q`
            /// ```text
            /// q * 1 = q.
            /// ```
            #[test]
            fn prop_quaternion_times_one_equals_quaternion(q in $Generator()) {
                let one: $ScalarType = num_traits::one();

                prop_assert_eq!(one * q, q);
            }
        }
    }
    }
}

exact_arithmetic_props!(quaternion_f64_arithmetic_props, f64, any_quaternion);
exact_arithmetic_props!(quaternion_i32_arithmetic_props, i32, any_quaternion);
exact_arithmetic_props!(quaternion_u32_arithmetic_props, u32, any_quaternion);


/// Generate property tests for quaternion arithmetic over floating point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_add_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Quaternion, 
        };
        use approx::{
            relative_eq
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// A quaternion plus a zero quaternion equals the same quaternion. 
            ///
            /// Given a quaternion `q`
            /// ```text
            /// q + 0 = q
            /// ```
            #[test]
            fn prop_quaternion_plus_zero_equals_quaternion(q in $Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();

                prop_assert_eq!(q + zero_quat, q);
            }

            /// A quaternion plus a zero quaternion equals the same quaternion.
            /// 
            /// Given a quaternion `q`
            /// ```text
            /// 0 + q = q
            /// ```
            #[test]
            fn prop_zero_plus_quaternion_equals_quaternion(q in $Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();

                prop_assert_eq!(zero_quat + q, q);
            }

            /// Given quaternions `q1` and `q2`, we should be able to use `q1` 
            /// and `q2` interchangeably with their references `&q1` and `&q2` in 
            /// arithmetic expressions involving quaternions.
            ///
            /// Given quaternions `q1` and `q2`, and their references `&q1` 
            /// and `&q2`, they should satisfy
            /// ```text
            ///  q1 +  q2 = &q1 +  q2
            ///  q1 +  q2 =  q1 + &q2
            ///  q1 +  q2 = &q1 + &q2
            ///  q1 + &q2 = &q1 +  q2
            /// &q1 +  q2 =  q1 + &q2
            /// &q1 +  q2 = &q1 + &q2
            ///  q1 + &q2 = &q1 + &q2
            /// ```
            #[test]
            fn prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!( q1 +  q2, &q1 +  q2);
                prop_assert_eq!( q1 +  q2,  q1 + &q2);
                prop_assert_eq!( q1 +  q2, &q1 + &q2);
                prop_assert_eq!( q1 + &q2, &q1 +  q2);
                prop_assert_eq!(&q1 +  q2,  q1 + &q2);
                prop_assert_eq!(&q1 +  q2, &q1 + &q2);
                prop_assert_eq!( q1 + &q2, &q1 + &q2);
            }

            /// Quaternion addition over floating point scalars should be 
            /// approximately commutative.
            /// 
            /// Given quaternions `q1` and `q2`, we have
            /// ```text
            /// q1 + q2 ~= q2 + q1
            /// ```
            /// Note that floating point quaternion addition cannot be exactly 
            /// commutative because arithmetic with floating point numbers is 
            /// not commutative.
            #[test]
            fn prop_quaternion_addition_almost_commutative(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {

                prop_assert!(relative_eq!(q1 + q2, q2 + q1, epsilon = $tolerance));
            }

            /// Quaternion addition over floating point scalars should be 
            /// approximately associative. 
            ///
            /// Given quaternions `q1`, `q2`, and `q3` we have
            /// ```text
            /// (q1 + q2) + q3 ~= q1 + (q2 + q3).
            /// ```
            /// Note that floating point quaternion addition cannot be exactly 
            /// associative because arithmetic with floating point numbers is 
            /// not associative.
            #[test]
            fn prop_quaternion_addition_almost_associative(
                q1 in $Generator::<$ScalarType>(), 
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()) {

                prop_assert!(relative_eq!((q1 + q2) + q3, q1 + (q2 + q3), epsilon = $tolerance));
            }
        }
    }
    }
}

approx_add_props!(quaternion_f64_add_props, f64, any_quaternion, 1e-8);


/// Generate property tests for quaternion arithmetic over exact scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_add_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Quaternion, 
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// A quaternion plus a zero quaternion equals the same quaternion.
            ///
            /// Given a quaternion `q`, it should satisfy
            /// ```text
            /// q + 0 = q
            /// ```
            #[test]
            fn prop_quaternion_plus_zero_equals_quaternion(q in $Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();

                prop_assert_eq!(q + zero_quat, q);
            }

            /// A zero quaternion plus a quaternion equals the same quaternion.
            ///
            /// Given a quaternion `q`, it should satisfy
            /// ```text
            /// 0 + q = q
            /// ```
            #[test]
            fn prop_zero_plus_quaternion_equals_quaternion(q in $Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();

                prop_assert_eq!(zero_quat + q, q);
            }

            /// Given quaternions `q1` and `q2`, we should be able to use `q1` 
            /// and `q2` interchangeably with their references `&q1` and `&q2` in 
            /// arithmetic expressions involving quaternions.
            ///
            /// Given quaternions `q1` and `q2`, and their references `&q1` and `&q2`, they 
            /// should satisfy
            /// ```text
            ///  q1 +  q2 = &q1 +  q2
            ///  q1 +  q2 =  q1 + &q2
            ///  q1 +  q2 = &q1 + &q2
            ///  q1 + &q2 = &q1 +  q2
            /// &q1 +  q2 =  q1 + &q2
            /// &q1 +  q2 = &q1 + &q2
            ///  q1 + &q2 = &q1 + &q2
            /// ```
            #[test]
            fn prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!( q1 +  q2, &q1 +  q2);
                prop_assert_eq!( q1 +  q2,  q1 + &q2);
                prop_assert_eq!( q1 +  q2, &q1 + &q2);
                prop_assert_eq!( q1 + &q2, &q1 +  q2);
                prop_assert_eq!(&q1 +  q2,  q1 + &q2);
                prop_assert_eq!(&q1 +  q2, &q1 + &q2);
                prop_assert_eq!( q1 + &q2, &q1 + &q2);
            }

            /// Quaternion addition over integer scalars should be commutative.
            ///
            /// Given quaternions `q1` and `q2`, we have
            /// ```text
            /// q1 + q2 = q2 + q1.
            /// ```
            #[test]
            fn prop_quaternion_addition_commutative(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                let zero = Quaternion::<$ScalarType>::zero();

                prop_assert_eq!((q1 + q2) - (q2 + q1), zero);
            }

            /// Given three quaternions of integer scalars, quaternion addition 
            /// should be associative.
            ///
            /// Given quaternions `q1`, `q2`, and `q3`, we have
            /// ```text
            /// (q1 + q2) + q3 = q1 + (q2 + q3)
            /// ```
            #[test]
            fn prop_quaternion_addition_associative(
                q1 in $Generator::<$ScalarType>(), 
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((q1 + q2) + q3, q1 + (q2 + q3));
            }
        }
    }
    }
}

exact_add_props!(quaternion_i32_add_props, i32, any_quaternion);
exact_add_props!(quaternion_u32_add_props, u32, any_quaternion);


/// Generate property tests for quaternion subtraction over floating point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_sub_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Quaternion,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The zero quaternion over floating point scalars should act as an 
            /// additive unit.
            ///
            /// Given a quaternion `q`, we have
            /// ```text
            /// q - 0 = q
            /// ```
            #[test]
            fn prop_quaternion_minus_zero_equals_quaternion(q in $Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();

                prop_assert_eq!(q - zero_quat, q);
            }

            /// Every quaternion should have an additive inverse.
            ///
            /// Given a quaternion `q`, there is a quaternion `-q` such that
            /// ```text
            /// q - q = q + (-q) = (-q) + q = 0
            /// ```
            #[test]
            fn prop_quaternion_minus_quaternion_equals_zero(q in $Generator::<$ScalarType>()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();

                prop_assert_eq!(q - q, zero_quat);
                prop_assert_eq!((-q) + q, zero_quat);
                prop_assert_eq!(q + (-q), zero_quat);
            }

            /// Given quaternions `q1` and `q2`, we should be able to use `q1` and 
            /// `q2` interchangeably with their references `&q1` and `&q2` in 
            /// arithmetic expressions involving quaternions.
            ///
            /// Given quaternions `q1` and `q2`, and their references `&q1` and 
            /// `&q2`, they should satisfy
            /// ```text
            ///  q1 -  q2 = &q1 -  q2
            ///  q1 -  q2 =  q1 - &q2
            ///  q1 -  q2 = &q1 - &q2
            ///  q1 - &q2 = &q1 -  q2
            /// &q1 -  q2 =  q1 - &q2
            /// &q1 -  q2 = &q1 - &q2
            ///  q1 - &q2 = &q1 - &q2
            /// ```
            #[test]
            fn prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!( q1 -  q2, &q1 -  q2);
                prop_assert_eq!( q1 -  q2,  q1 - &q2);
                prop_assert_eq!( q1 -  q2, &q1 - &q2);
                prop_assert_eq!( q1 - &q2, &q1 -  q2);
                prop_assert_eq!(&q1 -  q2,  q1 - &q2);
                prop_assert_eq!(&q1 -  q2, &q1 - &q2);
                prop_assert_eq!( q1 - &q2, &q1 - &q2);
            }
        }
    }
    }
}

approx_sub_props!(quaternion_f64_sub_props, f64, any_quaternion, 1e-8);


/// Generate property tests for quaternion arithmetic over exact scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_sub_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Quaternion,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The zero quaternion should act as an additive unit. 
            ///
            /// Given a quaternion `q`, we have
            /// ```text
            /// q - 0 = q
            /// ```
            #[test]
            fn prop_quaternion_minus_zero_equals_quaternion(q in $Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();

                prop_assert_eq!(q - zero_quat, q);
            }

            /// Every quaternion should have an additive inverse. 
            ///
            /// Given a quaternion `q`, there is a quaternion `-q` such that
            /// ```text
            /// q - q = q + (-q) = (-q) + q = 0
            /// ```
            #[test]
            fn prop_quaternion_minus_quaternion_equals_zero(q in $Generator::<$ScalarType>()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();

                prop_assert_eq!(q - q, zero_quat);
            }

            /// Given quaternions `q1` and `q2`, we should be able to use `q1` 
            /// and `q2` interchangeably with their references `&q1` and `&q2` 
            /// in arithmetic expressions involving quaternions.
            ///
            /// Given quaternions `q1` and `q2`, and their references `&q1` 
            /// and `&q2`, they should satisfy
            /// ```text
            ///  q1 -  q2 = &q1 -  q2
            ///  q1 -  q2 =  q1 - &q2
            ///  q1 -  q2 = &q1 - &q2
            ///  q1 - &q2 = &q1 -  q2
            /// &q1 -  q2 =  q1 - &q2
            /// &q1 -  q2 = &q1 - &q2
            ///  q1 - &q2 = &q1 - &q2
            /// ```
            #[test]
            fn prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!( q1 -  q2, &q1 -  q2);
                prop_assert_eq!( q1 -  q2,  q1 - &q2);
                prop_assert_eq!( q1 -  q2, &q1 - &q2);
                prop_assert_eq!( q1 - &q2, &q1 -  q2);
                prop_assert_eq!(&q1 -  q2,  q1 - &q2);
                prop_assert_eq!(&q1 -  q2, &q1 - &q2);
                prop_assert_eq!( q1 - &q2, &q1 - &q2);
            }
        }
    }
    }
}

exact_sub_props!(quaternion_i32_sub_props, i32, any_quaternion);
exact_sub_props!(quaternion_u32_sub_props, u32, any_quaternion);


/// Generate property tests for quaternion multiplication over floating point 
/// scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_mul_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use approx::relative_eq;
        use cglinalg_core::{
            Quaternion,
        };
        use super::{
            $Generator,
            $ScalarGen,
        };


        proptest! {
            /// Multiplication of a scalar and a quaternion should be approximately 
            /// commutative.
            ///
            /// Given a constant `c` and a quaternion `q`
            /// ```text
            /// c * q ~= q * c
            /// ```
            /// Note that floating point quaternion multiplication cannot be commutative 
            /// because multiplication in the underlying floating point scalars is not 
            /// commutative.
            #[test]
            fn prop_scalar_times_quaternion_equals_quaternion_times_scalar(
                c in $ScalarGen::<$ScalarType>(), q in $Generator::<$ScalarType>()) {
                
                prop_assume!(c.is_finite());
                prop_assume!(q.is_finite());
                prop_assert_eq!(c * q, q * c);
            }

            /// Quaternions have a multiplicative unit element.
            ///
            /// Given a quaternion `q`, and the unit quaternion `1`, we have
            /// ```text
            /// q * 1 = 1 * q = q
            /// ```
            #[test]
            fn prop_quaternion_multiplicative_unit(q in $Generator::<$ScalarType>()) {
                let one = Quaternion::identity();

                prop_assert_eq!(q * one, q);
                prop_assert_eq!(one * q, q);
                prop_assert_eq!(q * one, one * q);
            }

            /// Every nonzero quaternion over floating point scalars has an 
            /// approximate multiplicative inverse.
            ///
            /// Given a quaternion `q` and its inverse `q_inv`, we have
            /// ```text
            /// q * q_inv ~= q_inv * q ~= 1
            /// ```
            /// Note that quaternion algebra over floating point scalars is not 
            /// commutative because multiplication of the underlying scalars is 
            /// not commutative.
            #[test]
            fn prop_quaternion_multiplicative_inverse(q in $Generator::<$ScalarType>()) {
                prop_assume!(q.is_finite());
                prop_assume!(q.is_invertible());

                let one = Quaternion::identity();
                let q_inv = q.inverse().unwrap();

                prop_assert!(relative_eq!(q * q_inv, one, epsilon = $tolerance));
                prop_assert!(relative_eq!(q_inv * q, one, epsilon = $tolerance));
            }
        }
    }
    }
}

approx_mul_props!(quaternion_f64_mul_props, f64, any_quaternion, any_scalar, 1e-8);


/// Generate property tests for quaternion multiplication over exact scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_mul_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Quaternion,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// Multiplication of an integer scalar and a quaternion over integer 
            /// scalars should be commutative.
            ///
            /// Given a constant `c` and a quaternion `q`
            /// ```text
            /// c * q = q * c
            /// ```
            #[test]
            fn prop_scalar_times_quaternion_equals_quaternion_times_scalar(
                c in any::<$ScalarType>(), q in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!(c * q, q * c);
            }

            /// Exact multiplication of two scalars and a quaternion should be 
            /// compatible with multiplication of all scalars. 
            ///
            /// In other words, scalar multiplication of two scalars with a 
            /// quaternion should act associatively just like the multiplication 
            /// of three scalars. 
            ///
            /// Given scalars `a` and `b`, and a quaternion `q`, we have
            /// ```text
            /// (a * b) * q = a * (b * q)
            /// ```
            #[test]
            fn prop_scalar_multiplication_compatibility(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), q in $Generator::<$ScalarType>()) {

                prop_assert_eq!(a * (b * q), (a * b) * q);
            }

            /// Quaternion multiplication over integer scalars is exactly associative.
            ///
            /// Given quaternions `q1`, `q2`, and `q3`, we have
            /// ```text
            /// (q1 * q2) * q3 = q1 * (q2 * q3)
            /// ```
            #[test]
            fn prop_quaternion_multiplication_associative(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>(), 
                q3 in $Generator::<$ScalarType>()
            ) {
                prop_assert_eq!(q1 * (q2 * q3), (q1 * q2) * q3);
            }

            /// Quaternions have a multiplicative unit element.
            ///
            /// Given a quaternion `q`, and the unit quaternion `1`, we have
            /// ```text
            /// q * 1 = 1 * q = q
            /// ```
            #[test]
            fn prop_quaternion_multiplicative_unit(q in $Generator::<$ScalarType>()) {
                let one = Quaternion::identity();
                prop_assert_eq!(q * one, q);
                prop_assert_eq!(one * q, q);
                prop_assert_eq!(q * one, one * q);
            }
        }
    }
    }
}

exact_mul_props!(quaternion_i32_mul_props, i32, any_quaternion);
exact_mul_props!(quaternion_u32_mul_props, u32, any_quaternion);


/// Generate property tests for quaternion distribution over exact scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_distributive_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// Scalar multiplication should distribute over quaternion addition.
            ///
            /// Given a scalar `a` and quaternions `q1` and `q2`
            /// ```text
            /// a * (q1 + q2) = a * q1 + a * q2
            /// ```
            #[test]
            fn prop_distribution_over_quaternion_addition(
                a in any::<$ScalarType>(), 
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!(a * (q1 + q2), a * q1 + a * q2);
                prop_assert_eq!((q1 + q2) * a,  q1 * a + q2 * a);
            }

            /// Multiplication of a sum of scalars should distribute over a 
            /// quaternion.
            ///
            /// Given scalars `a` and `b` and a quaternion `q`, we have
            /// ```text
            /// (a + b) * q = a * q + b * q
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                q in $Generator::<$ScalarType>()) {
    
                prop_assert_eq!((a + b) * q, a * q + b * q);
                prop_assert_eq!(q * (a + b), q * a + q * b);
            }

            /// Multiplication of two quaternions by a scalar on the right 
            /// should distribute.
            ///
            /// Given quaternions `q1` and `q2`, and a scalar `a`
            /// ```text
            /// (q1 + q2) * a = q1 * a + q2 * a
            /// ```
            #[test]
            fn prop_distribution_over_quaternion_addition1(
                a in any::<$ScalarType>(), 
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                    
                prop_assert_eq!((q1 + q2) * a,  q1 * a + q2 * a);
            }

            /// Multiplication of a quaternion on the right by the sum of two 
            /// scalars should distribute over the two scalars. 
            ///
            /// Given a quaternion `q` and scalars `a` and `b`
            /// ```text
            /// q * (a + b) = q * a + q * b
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition1(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                q in $Generator::<$ScalarType>()) {
    
                prop_assert_eq!(q * (a + b), q * a + q * b);
            }

            /// Quaternion multiplication should be distributive on the right.
            ///
            /// Given three quaternions `q1`, `q2`, and `q3`
            /// ```text
            /// (q1 + q2) * q3 = q1 * q3 + q2 * q3
            /// ```
            #[test]
            fn prop_quaternion_multiplication_right_distributive(
                q1 in $Generator::<$ScalarType>(), 
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()
            ) {
                prop_assert_eq!((q1 + q2) * q3, q1 * q3 + q2 * q3);
            }

            /// Quaternion multiplication should be distributive on the left.
            ///
            /// Given three quaternions `q1`, `q2`, and `q3`
            /// ```text
            /// q1 * (q2 + q3) = q1 * q2 + q1 * q3
            /// ```
            #[test]
            fn prop_quaternion_multiplication_left_distributive(
                q1 in $Generator::<$ScalarType>(), 
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()
            ) {
                prop_assert_eq!((q1 + q2) * q3, q1 * q3 + q2 * q3);
            }
        }
    }
    }    
}

exact_distributive_props!(quaternion_i32_distributive_props, i32, any_quaternion);
exact_distributive_props!(quaternion_u32_distributive_props, u32, any_quaternion);


/// Generate property tests for quaternion dot products over integer scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_dot_product_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };

    
        proptest! {
            /// The dot product of quaternions over integer scalars is commutative.
            ///
            /// Given quaternions `q1` and `q2`
            /// ```text
            /// dot(q1, q2) = dot(q2, q1)
            /// ```
            #[test]
            fn prop_quaternion_dot_product_commutative(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!(q1.dot(&q2), q2.dot(&q1));

            }

            /// The dot product of quaternions over integer scalars is right 
            /// distributive.
            ///
            /// Given quaternions `q1`, `q2`, and `q3`
            /// ```text
            /// dot(q1, q2 + q3) = dot(q1, q2) + dot(q1, q3)
            /// ```
            #[test]
            fn prop_quaternion_dot_product_right_distributive(
                q1 in $Generator::<$ScalarType>(),
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()) {
            
                prop_assert_eq!(q1.dot(&(q2 + q3)), q1.dot(&q2) + q1.dot(&q3));
            }

            /// The dot product of quaternions over integer scalars is left 
            /// distributive.
            ///
            /// Given quaternions `q1`, `q2`, and `q3`
            /// ```text
            /// dot(q1 + q2,  q3) = dot(q1, q3) + dot(q2, q3)
            /// ```
            #[test]
            fn prop_quaternion_dot_product_left_distributive(
                q1 in $Generator::<$ScalarType>(),
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()) {
            
                prop_assert_eq!((q1 + q2).dot(&q3), q1.dot(&q3) + q2.dot(&q3));
            }

            /// The dot product of quaternions over integer scalars is 
            /// commutative with scalars.
            ///
            /// Given quaternions `q1` and `q2`, and scalars `a` and `b`
            /// ```text
            /// dot(a * q1, b * q2) = a * b * dot(q1, q2)
            /// ```
            #[test]
            fn prop_quaternion_dot_product_times_scalars_commutative(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(),
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((a * q1).dot(&(b * q2)), a * b * q1.dot(&q2));
            }

            /// The dot product of quaternions over integer scalars is right
            /// bilinear.
            ///
            /// Given quaternions `q1`, `q2` and `q3`, and scalars `a` and `b`
            /// ```text
            /// dot(q1, a * q2 + b * q3) = a * dot(q1, q2) + b * dot(q1, q3)
            /// ```
            #[test]
            fn prop_quaternion_dot_product_right_bilinear(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(),
                q1 in $Generator::<$ScalarType>(),
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()) {

                prop_assert_eq!(q1.dot(&(a * q2 + b * q3)), a * q1.dot(&q2) + b * q1.dot(&q3));
            }

            /// The dot product of quaternions over integer scalars is left
            /// bilinear.
            ///
            /// Given quaternions `q1`, `q2` and `q3`, and scalars `a` and `b`
            /// ```text
            /// dot(a * q1 + b * q2, q3) = a * dot(q1, q3) + b * dot(q2, q3)
            /// ```
            #[test]
            fn prop_quaternion_dot_product_left_bilinear(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(),
                q1 in $Generator::<$ScalarType>(),
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((a * q1 + b * q2).dot(&q3), a * q1.dot(&q3) + b * q2.dot(&q3));
            }
        }
    }
    }
}

exact_dot_product_props!(quaternion_i32_dot_product_props, i32, any_quaternion);
exact_dot_product_props!(quaternion_u32_dot_product_props, u32, any_quaternion);


/// Generate property tests for quaternion conjugation over floating point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! approx_conjugation_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };

    
        proptest! {
            /// Conjugating a quaternion twice should give the original quaternion.
            ///
            /// Given a quaternion `q`
            /// ```text
            /// q** = conjugate(conjugate(q)) = q
            /// ```
            #[test]
            fn prop_quaternion_conjugate_conjugate_equals_quaternion(q in $Generator::<$ScalarType>()) {
                prop_assert_eq!(q.conjugate().conjugate(), q);
            }

            /// Quaternion conjugation is linear.
            ///
            /// Given quaternions `q1` and `q2`, quaternion conjugation satisfies
            /// ```text
            /// conjugate(q1 + q2) = conjugate(q1) + conjugate(q2)
            /// ```
            #[test]
            fn prop_quaternion_conjugation_linear(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((q1 + q2).conjugate(), q1.conjugate() + q2.conjugate());
            }
        }
    }
    }
}

approx_conjugation_props!(quaternion_f64_conjugation_props, f64, any_quaternion);


/// Generate property tests for quaternion conjugation over exact scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_conjugation_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };

    
        proptest! {
            /// Conjugating a quaternion twice should give the original quaternion.
            ///
            /// Given a quaternion `q`
            /// ```text
            /// q** = conjugate(conjugate(q)) = q
            /// ```
            #[test]
            fn prop_quaternion_conjugate_conjugate_equals_quaternion(q in $Generator::<$ScalarType>()) {
                prop_assert_eq!(q.conjugate().conjugate(), q);
            }

            /// Quaternion conjugation is linear.
            ///
            /// Given quaternions `q1` and `q2`, quaternion conjugation satisfies
            /// ```text
            /// conjugate(q1 + q2) = conjugate(q1) + conjugate(q2)
            /// ```
            #[test]
            fn prop_quaternion_conjugation_linear(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((q1 + q2).conjugate(), q1.conjugate() + q2.conjugate());
            }

            /// Quaternion multiplication transposes under conjugation.
            ///
            /// Given quaternions `q1` and `q2`
            /// ```text
            /// conjugate(q1 * q2) = conjugate(q2) * conjugate(q1)
            /// ```
            #[test]
            fn prop_quaternion_conjugation_transposes_products(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((q1 * q2).conjugate(), q2.conjugate() * q1.conjugate());
            }
        }
    }
    }
}

exact_conjugation_props!(quaternion_i32_conjugation_props, i32, any_quaternion);
exact_conjugation_props!(quaternion_i64_conjugation_props, i64, any_quaternion);


/// Generate property tests for the quaternion squared **L2** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_norm_squared_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use approx::{
            relative_ne,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The squared norm of a quaternion is nonnegative. 
            ///
            /// Given a quaternion `q`
            /// ```text
            /// norm_squared(q) >= 0
            /// ```
            #[test]
            fn prop_norm_squared_nonnegative(q in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();

                prop_assert!(q.norm_squared() >= zero);
            }

            /// The squared norm function is point separating. In particular, if 
            /// the squared distance between two quaternions `q1` and `q2` is 
            /// zero, then `q1 = q2`.
            ///
            /// Given quaternions `q1` and `q2`
            /// ```text
            /// norm_squared(q1 - q2) = 0 => q1 = q2 
            /// ```
            /// Equivalently, if `q1` is not equal to `q2`, then their squared distance is 
            /// nonzero
            /// ```text
            /// q1 != q2 => norm_squared(q1 - q2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the 
            /// norm function.
            #[test]
            fn prop_norm_squared_approx_point_separating(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                let zero: $ScalarType = num_traits::zero();

                prop_assume!(relative_ne!(q1, q2, epsilon = $tolerance));
                prop_assert!(
                    relative_ne!((q1 - q2).norm_squared(), zero, epsilon = $tolerance),
                    "\n|q1 - q2|^2 = {}\n",
                    (q1 - q2).norm_squared()
                );
            }
        }
    }
    }
}

approx_norm_squared_props!(quaternion_f64_norm_squared_props, f64, any_quaternion, any_scalar, 1e-10);


/// Generate property tests for quaternion squared **L2** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! approx_norm_squared_synonym_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The [`Quaternion::magnitude_squared`] function and the [`Quaternion::norm_squared`] 
            /// function are synonyms. In particular, given a quaternion `q`
            /// ```text
            /// magnitude_squared(q) = norm_squared(q)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_magnitude_squared_norm_squared(q in $Generator::<$ScalarType>()) {
                prop_assert_eq!(q.magnitude_squared(), q.norm_squared());
            }
        }
    }
    }
}

approx_norm_squared_synonym_props!(quaternion_f64_norm_squared_synonym_props, f64, any_quaternion);


/// Generate property tests for the quaternion squared **L2** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
macro_rules! exact_norm_squared_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The squared norm of a quaternion is nonnegative. 
            ///
            /// Given a quaternion `q`
            /// ```text
            /// norm_squared(q) >= 0
            /// ```
            #[test]
            fn prop_norm_squared_nonnegative(q in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();

                prop_assert!(q.norm_squared() >= zero);
            }

            /// The squared norm function is point separating. In particular, if 
            /// the squared distance between two quaternions `q1` and `q2` is 
            /// zero, then `q1 = q2`.
            ///
            /// Given quaternions `q1` and `q2`
            /// ```text
            /// norm_squared(q1 - q2) = 0 => q1 = q2 
            /// ```
            /// Equivalently, if `q1` is not equal to `q2`, then their squared distance is 
            /// nonzero
            /// ```text
            /// q1 != q2 => norm_squared(q1 - q2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the 
            /// norm function.
            #[test]
            fn prop_norm_squared_approx_point_separating(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                let zero: $ScalarType = num_traits::zero();

                prop_assume!(q1 != q2);
                prop_assert_ne!(
                    (q1 - q2).norm_squared(), zero,
                    "\n|q1 - q2|^2 = {}\n",
                    (q1 - q2).norm_squared()
                );
            }
        }
    }
    }
}

exact_norm_squared_props!(quaternion_i32_norm_squared_props, i32, any_quaternion, any_scalar);
exact_norm_squared_props!(quaternion_u32_norm_squared_props, u32, any_quaternion, any_scalar);


/// Generate property tests for quaternion squared **L2** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_norm_squared_synonym_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The [`Quaternion::magnitude_squared`] function and the [`Quaternion::norm_squared`] 
            /// function are synonyms. In particular, given a quaternion `q`
            /// ```text
            /// magnitude_squared(q) = norm_squared(q)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_magnitude_squared_norm_squared(q in $Generator::<$ScalarType>()) {
                prop_assert_eq!(q.magnitude_squared(), q.norm_squared());
            }
        }
    }
    }
}

exact_norm_squared_synonym_props!(quaternion_i32_norm_squared_synonym_props, i32, any_quaternion);
exact_norm_squared_synonym_props!(quaternion_u32_norm_squared_synonym_props, u32, any_quaternion);


/// Generate property tests for the quaternion **L2** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! norm_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use approx::{
            relative_ne,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The norm of a quaternion is nonnegative. 
            ///
            /// Given a quaternion `q`
            /// ```text
            /// norm(q) >= 0
            /// ```
            #[test]
            fn prop_norm_nonnegative(q in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();

                prop_assert!(q.norm() >= zero);
            }

            /// The norm function is point separating. In particular, if 
            /// the distance between two quaternions `q1` and `q2` is 
            /// zero, then `q1 = q2`.
            ///
            /// Given quaternions `q1` and `q2`
            /// ```text
            /// norm(q1 - q2) = 0 => q1 = q2 
            /// ```
            /// Equivalently, if `q1` is not equal to `q2`, then their distance is 
            /// nonzero
            /// ```text
            /// q1 != q2 => norm(q1 - q2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the 
            /// norm function.
            #[test]
            fn prop_norm_approx_point_separating(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                let zero: $ScalarType = num_traits::zero();

                prop_assume!(relative_ne!(q1, q2, epsilon = $tolerance));
                prop_assert!(
                    relative_ne!((q1 - q2).norm(), zero, epsilon = $tolerance),
                    "\n|q1 - q2| = {}\n",
                    (q1 - q2).norm()
                );
            }
        }
    }
    }
}

norm_props!(quaternion_f64_norm_props, f64, any_quaternion, any_scalar, 1e-10);


/// Generate property tests for the quaternion **L1** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! l1_norm_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use approx::{
            relative_ne,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The **L1** norm of a quaternion is nonnegative. 
            ///
            /// Given a quaternion `q`
            /// ```text
            /// l1_norm(q) >= 0
            /// ```
            #[test]
            fn prop_l1_norm_nonnegative(q in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();

                prop_assert!(q.l1_norm() >= zero);
            }

            /// The **L1** norm function is point separating. In particular, if 
            /// the distance between two quaternions `q1` and `q2` is 
            /// zero, then `q1 = q2`.
            ///
            /// Given quaternions `q1` and `q2`
            /// ```text
            /// l1_norm(q1 - q2) = 0 => q1 = q2 
            /// ```
            /// Equivalently, if `q1` is not equal to `q2`, then their distance is 
            /// nonzero
            /// ```text
            /// q1 != q2 => l1_norm(q1 - q2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the 
            /// norm function.
            #[test]
            fn prop_l1_norm_approx_point_separating(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                let zero: $ScalarType = num_traits::zero();

                prop_assume!(relative_ne!(q1, q2, epsilon = $tolerance));
                prop_assert!(
                    relative_ne!((q1 - q2).l1_norm(), zero, epsilon = $tolerance),
                    "\nl1_norm(q1 - q2) = {}\n",
                    (q1 - q2).l1_norm()
                );
            }
        }
    }
    }
}

l1_norm_props!(quaternion_f64_l1_norm_props, f64, any_quaternion, any_scalar, 1e-10);


/// Generate property tests for quaternion norms.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! norm_synonym_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The [`Quaternion::magnitude`] function and the [`Quaternion::norm`] function
            /// are synonyms. In particular, given a quaternion `q`
            /// ```text
            /// magnitude(q) = norm(q)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_magnitude_norm_synonyms(q in $Generator::<$ScalarType>()) {
                prop_assert_eq!(q.magnitude(), q.norm());
            }

            /// The [`Quaternion::l2_norm`] function and the [`Quaternion::norm`] function
            /// are synonyms. In particular, given a quaternion `q`
            /// ```text
            /// l2_norm(q) = norm(q)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_l2_norm_norm_synonyms(q in $Generator::<$ScalarType>()) {
                prop_assert_eq!(q.l2_norm(), q.norm());
            }

            /// The [`Quaternion::magnitude_squared`] function and the [`Quaternion::norm_squared`] 
            /// function are synonyms. In particular, given a quaternion `q`
            /// ```text
            /// magnitude_squared(q) = norm_squared(q)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_magnitude_squared_norm_squared(q in $Generator::<$ScalarType>()) {
                prop_assert_eq!(q.magnitude_squared(), q.norm_squared());
            }
        }
    }
    }
}

norm_synonym_props!(quaternion_f64_norm_synonym_props, f64, any_quaternion, any_scalar, 1e-10);


/// Generate property tests for quaternion square roots.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    quaternions.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! sqrt_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use approx::{
            relative_eq,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The square of the positive square root of a quaternion is the original
            /// quaternion.
            /// 
            /// Given a quaternion `q`
            /// ```text
            /// sqrt(q) * sqrt(q) == q
            /// ```
            #[test]
            fn prop_positive_square_root_squared(q in $Generator::<$ScalarType>()) {
                let sqrt_q = q.sqrt();

                prop_assert!(
                    relative_eq!(sqrt_q * sqrt_q, q, epsilon = $tolerance),
                    "q = {:?}\nsqrt_q = {:?}\nsqrt_q * sqrt_q = {:?}",
                    q, sqrt_q, sqrt_q * sqrt_q
                );
            }

            /// The square of the negative square root of a quaternion is the original
            /// quaternion.
            /// 
            /// Given a quaternion `q`
            /// ```text
            /// -sqrt(q) * -sqrt(q) == q
            /// ```
            #[test]
            fn prop_negative_square_root_squared(q in $Generator::<$ScalarType>()) {
                let minus_sqrt_q = -q.sqrt();

                prop_assert!(
                    relative_eq!(minus_sqrt_q * minus_sqrt_q, q, epsilon = $tolerance),
                    "q = {:?}\nminus_sqrt_q = {:?}\nminus_sqrt_q * minus_sqrt_q = {:?}",
                    q, minus_sqrt_q, minus_sqrt_q * minus_sqrt_q
                );
            }
        }
    }
    }
}

sqrt_props!(quaternion_f64_sqrt_props, f64, any_quaternion_squared, any_scalar, 1e-7);

