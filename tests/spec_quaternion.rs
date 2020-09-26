extern crate cglinalg;
extern crate num_traits;
extern crate proptest;

use proptest::prelude::*;
use cglinalg::{
    Quaternion, 
    Scalar,
    ScalarFloat,
};


fn any_scalar<S>() -> impl Strategy<Value = S>
    where S: Scalar + Arbitrary
{
    any::<S>().prop_map(|scalar| {
        let modulus = num_traits::cast(1_000_000).unwrap();

        scalar % modulus
    })
}

fn any_quaternion<S>() -> impl Strategy<Value = Quaternion<S>> 
    where S: Scalar + Arbitrary
{
    any::<(S, S, S, S)>().prop_map(|(x, y, z, w)| {
        let modulus = num_traits::cast(1_000_000).unwrap();
        let quaternion = Quaternion::new(x, y, z, w);

        quaternion % modulus
    })
}

fn any_unit_quaternion<S>() -> impl Strategy<Value = Quaternion<S>> 
    where S: ScalarFloat + Arbitrary
{
    use cglinalg::Magnitude;
    any::<(S, S, S, S)>()
        .prop_map(|(x, y, z, w)| Quaternion::new(x, y, z, w))
        .prop_filter("Zero length quaternions are not invertible.", |q| !q.magnitude().is_zero())
        .prop_map(|q| q.normalize())
}


/// Generate property tests for quaternion indexing.
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
/// * `$UpperBound` denotes the upperbound on the range of acceptable indices.
macro_rules! index_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $UpperBound:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// When a quaternion is treated like an array, it should accept all indices
            /// below the length of the array.
            ///
            /// Given a quaternion `q`, it should return the element at position 
            /// `index` in the underlying storage of the quaternion when the given 
            /// index is inbounds.
            #[test]
            fn prop_accepts_all_indices_in_of_bounds(
                q in $Generator::<$ScalarType>(), index in 0..$UpperBound as usize) {

                prop_assert_eq!(q[index], q[index]);
            }
    
            /// When a quaternion is treated like an array, it should reject any 
            /// input index outside the length of the array.
            ///
            /// Given a quaternion `q`, when the element index `index` is out of 
            /// bounds, it should generate a panic just like an array indexed 
            /// out of bounds.
            #[test]
            #[should_panic]
            fn prop_panics_when_index_out_of_bounds(
                q in $Generator::<$ScalarType>(), index in $UpperBound..usize::MAX) {
                
                prop_assert_eq!(q[index], q[index]);
            }
        }
    }
    }
}

index_props!(quaternion_index_props, f64, any_quaternion, 4);


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
        use cglinalg::{
            Quaternion,
            Zero
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
            fn prop_quaternion_times_one_equals_quaternion(v in $Generator()) {
                let one: $ScalarType = num_traits::one();

                prop_assert_eq!(one * v, v);
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
        use cglinalg::{
            Quaternion, 
            Zero
        };
        use cglinalg::approx::{
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
            ///  q1 + &q2 = &q1 + &vq
            /// ```
            #[test]
            fn prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!(q1 + q2, &q1 + q2);
                prop_assert_eq!(q1 + q2, q1 + &q2);
                prop_assert_eq!(q1 + q2, &q1 + &q2);
                prop_assert_eq!(q1 + &q2, &q1 + q2);
                prop_assert_eq!(&q1 + q2, q1 + &q2);
                prop_assert_eq!(&q1 + q2, &q1 + &q2);
                prop_assert_eq!(q1 + &q2, &q1 + &q2);
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

approx_add_props!(quaternion_f64_add_props, f64, any_quaternion, 1e-7);


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
        use cglinalg::{
            Quaternion, 
            Zero
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
        use cglinalg::{
            Quaternion,
            Zero
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

approx_sub_props!(quaternion_f64_sub_props, f64, any_quaternion, 1e-7);


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
        use cglinalg::{
            Quaternion,
            Zero
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
        use cglinalg::approx::relative_eq;
        use cglinalg::{
            Quaternion,
            Identity,
            Finite
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
                prop_assert!(
                    relative_eq!(c * q, q * c, epsilon = $tolerance)
                );
            }

            /// Multiplication of two scalars and a quaternion should be compatible 
            /// with multiplication of all scalars. 
            ///
            /// In other words, scalar multiplication of two scalar with a quaternion 
            /// should act associatively, just like the multiplication of three scalars. 
            /// Given scalars `a` and `b`, and a quaternion `q`, we have
            /// ```text
            /// (a * b) * q ~= a * (b * q)
            /// ```
            /// Note that the compatability of scalars with quaternions can only be 
            /// approximate and not exact because multiplication of the underlying 
            /// scalars is not associative. 
            #[test]
            fn prop_scalar_multiplication_compatibility(
                a in $ScalarGen::<$ScalarType>(), b in $ScalarGen::<$ScalarType>(), q in $Generator::<$ScalarType>()) {

                prop_assert!(
                    relative_eq!(a * (b * q), (a * b) * q, epsilon = $tolerance),
                    "a * (b * q) = {}\n(a * b) * q = {}",
                    a * (b * q), (a * b) * q
                );
            }

            /// Quaternion multiplication over floating point numbers is 
            /// approximately associative.
            ///
            /// Given quaternions `q1`, `q2`, and `q3`, we have
            /// ```text
            /// (q1 * q2) * q3 ~= q1 * (q2 * q3)
            /// ```
            /// Note that the quaternion multiplication can only be approximately 
            /// associative and not exactly associative because multiplication of 
            /// the underlying scalars is not associative. 
            #[test]
            fn prop_quaternion_multiplication_associative(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>(), 
                q3 in $Generator::<$ScalarType>()) {

                prop_assume!((q1 * (q2 * q3)).is_finite());
                prop_assume!(((q1 * q2) * q3).is_finite());
                prop_assert!(relative_eq!(q1 * (q2 * q3), (q1 * q2) * q3, epsilon = $tolerance));
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

            /// Quaternion multiplication transposes under inversion.
            ///
            /// Given two invertible quaternions `q1` and `q2`
            /// ```text
            /// inverse(q1 * q2) = inverse(q2) * inverse(q1)
            /// ```
            /// Note that quaternion multiplication is noncommutative.
            #[test]
            fn prop_quaternion_inversion_involutive(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {

                prop_assume!(q1.is_finite());
                prop_assume!(q1.is_invertible());
                prop_assume!(q2.is_finite());
                prop_assume!(q2.is_invertible());
                prop_assume!((q1 * q2).is_finite());
                prop_assume!((q1 * q2).is_invertible());

                let q1_inv = q1.inverse().unwrap();
                let q2_inv = q2.inverse().unwrap();
                let q1_times_q2_inv = (q1 * q2).inverse().unwrap();

                prop_assert!(relative_eq!(q1_times_q2_inv, q2_inv * q1_inv, epsilon = $tolerance));
            }
        }
    }
    }
}

approx_mul_props!(quaternion_f64_mul_props, f64, any_quaternion, any_scalar, 1e-7);


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
        use cglinalg::{
            Quaternion, 
            Identity,
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


/// Generate property tests for quaternion distribution over floating point 
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
macro_rules! approx_distributive_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::{
            Finite
        };
        use cglinalg::approx::{
            relative_eq
        };
        use super::{
            $Generator,
            $ScalarGen,
        };

    
        proptest! {
            /// Scalar multiplication should approximately distribute over 
            /// quaternion addition.
            ///
            /// Given a scalar `a` and quaternions `q1` and `q2`
            /// ```text
            /// a * (q1 + q2) ~= a * q1 + a * q2
            /// ```
            #[test]
            fn prop_distribution_over_quaternion_addition(
                a in $ScalarGen::<$ScalarType>(), 
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                prop_assume!((a * (q1 + q2)).is_finite());
                prop_assume!((a * q1 + a * q2).is_finite());
                prop_assert!(relative_eq!(a * (q1 + q2), a * q1 + a * q2, epsilon = $tolerance));
            }
    
            /// Multiplication of a sum of scalars should approximately distribute 
            /// over a quaternion.
            ///
            /// Given scalars `a` and `b` and a quaternion `q`, we have
            /// ```text
            /// (a + b) * q ~= a * q + b * q
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition(
                a in $ScalarGen::<$ScalarType>(), b in $ScalarGen::<$ScalarType>(), 
                q in $Generator::<$ScalarType>()) {
    
                prop_assume!(((a + b) * q).is_finite());
                prop_assume!((a * q + b * q).is_finite());
                prop_assert!(relative_eq!((a + b) * q, a * q + b * q, epsilon = $tolerance));
            }

            /// Multiplication of two quaternions by a scalar on the right should 
            /// approximately distribute.
            ///
            /// Given quaternions `q1` and `q2` and a scalar `a`
            /// ```text
            /// (q1 + q2) * a ~= q1 * a + q2 * a
            /// ```
            #[test]
            fn prop_distribution_over_quaternion_addition1(
                a in $ScalarGen::<$ScalarType>(), 
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                    
                prop_assume!(((q1 + q2) * a).is_finite());
                prop_assume!((q1 * a + q2 * a).is_finite());
                prop_assert!(relative_eq!((q1 + q2) * a,  q1 * a + q2 * a, epsilon = $tolerance));
            }

            /// Multiplication of a quaternion on the right by the sum of two 
            /// scalars should approximately distribute over the two scalars.
            ///
            /// Given a quaternion `q` and scalars `a` and `b`
            /// ```text
            /// q * (a + b) ~= q * a + q * b
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition1(
                a in $ScalarGen::<$ScalarType>(), b in $ScalarGen::<$ScalarType>(), 
                q in $Generator::<$ScalarType>()) {
    
                prop_assume!((q * (a + b)).is_finite());
                prop_assume!((q * a + q * b).is_finite());
                prop_assert!(relative_eq!(q * (a + b), q * a + q * b, epsilon = $tolerance));
            }

            /// Quaternion multiplication over floating point numbers should be 
            /// approximately distributive on the right.
            ///
            /// Given three quaternions `q1`, `q2`, and `q3`
            /// ```text
            /// (q1 + q2) * q3 ~= q1 * q3 + q2 * q3
            /// ```
            #[test]
            fn prop_quaternion_multiplication_right_distributive(
                q1 in $Generator::<$ScalarType>(), 
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()) {
    
                prop_assume!(((q1 + q2) * q3).is_finite());
                prop_assume!((q1 * q3 + q2 * q3).is_finite());
                prop_assert!(relative_eq!((q1 + q2) * q3, q1 * q3 + q2 * q3, epsilon = $tolerance));
            }

            /// Quaternion multiplication over floating point numbers should be 
            /// approximately distributive on the left.
            ///
            /// Given three quaternions `q1`, `q2`, and `q3`
            /// ```text
            /// q1 * (q2 + q3) ~= q1 * q2 + q1 * q3
            /// ```
            #[test]
            fn prop_quaternion_multiplication_left_distributive(
                q1 in $Generator::<$ScalarType>(), 
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()) {

                prop_assume!(((q1 * (q2 + q3)).is_finite()));
                prop_assume!((q1 * q2 + q1 * q3).is_finite());
                prop_assert!(relative_eq!(q1 * (q2 + q3), q1 * q2 + q1 * q3, epsilon = $tolerance));
            }
        }
    }
    }    
}

approx_distributive_props!(quaternion_f64_distributive_props, f64, any_quaternion, any_scalar, 1e-7);


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


/// Generate property tests for quaternion dot products over floating point 
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
macro_rules! approx_dot_product_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::{
            DotProduct
        };
        use cglinalg::approx::{
            relative_eq
        };
        use super::{
            $Generator,
            $ScalarGen,
        };

    
        proptest! {
            /// The dot product of quaternions over floating point scalars is 
            /// approximately commutative.
            ///
            /// Given quaternions `q1` and `q2`
            /// ```text
            /// dot(q1, q2) ~= dot(q2, q1)
            /// ```
            #[test]
            fn prop_quaternion_dot_product_commutative(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {

                prop_assume!(q1.dot(q2).is_finite());
                prop_assume!(q2.dot(q1).is_finite());
                prop_assert!(relative_eq!(q1.dot(q2), q2.dot(q1), epsilon = $tolerance));
            }

            /// The dot product of quaternions over floating point scalars is 
            /// approximately right distributive.
            ///
            /// Given quaternions `q1`, `q2`, and `q3`
            /// ```text
            /// dot(q1, q2 + q3) ~= dot(q1, q2) + dot(q1, q3)
            /// ```
            #[test]
            fn prop_quaternion_dot_product_right_distributive(
                q1 in $Generator::<$ScalarType>(),
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()) {
            
                prop_assume!(q1.dot(q2 + q3).is_finite());
                prop_assume!((q1.dot(q2) + q1.dot(q3)).is_finite());
                prop_assert!(
                    relative_eq!(q1.dot(q2 + q3), q1.dot(q2) + q1.dot(q3), epsilon = $tolerance)
                );
            }

            /// The dot product of quaternions over floating point scalars is 
            /// approximately left distributive.
            ///
            /// Given quaternions `q1`, `q2`, and `q3`
            /// ```text
            /// dot(q1 + q2,  q3) ~= dot(q1, q3) + dot(q2, q3)
            /// ```
            #[test]
            fn prop_quaternion_dot_product_left_distributive(
                q1 in $Generator::<$ScalarType>(),
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()) {
            
                prop_assume!((q1 + q2).dot(q3).is_finite());
                prop_assume!((q1.dot(q3) + q2.dot(q3)).is_finite());
                prop_assert!(
                    relative_eq!((q1 + q2).dot(q3), q1.dot(q3) + q2.dot(q3), epsilon = $tolerance)
                );
            }

            /// The dot product of quaternions over floating point scalars is s
            /// approximately commutative with scalars.
            ///
            /// Given quaternions `q1` and `q2`, and scalars `a` and `b`
            /// ```text
            /// dot(a * q1, b * q2) ~= a * b * dot(q1, q2)
            /// ```
            #[test]
            fn prop_quaternion_dot_product_times_scalars_commutative(
                a in $ScalarGen::<$ScalarType>(), b in $ScalarGen::<$ScalarType>(),
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {

                prop_assume!((a * q1).dot(b * q2).is_finite());
                prop_assume!((a * b * q1.dot(q2)).is_finite());
                prop_assert!(relative_eq!((a * q1).dot(b * q2), a * b * q1.dot(q2), epsilon = $tolerance));
            }

            /// The dot product of quaternions over floating point scalars is 
            /// approximately right bilinear.
            ///
            /// Given quaternions `q1`, `q2` and `q3`, and scalars `a` and `b`
            /// ```text
            /// dot(q1, a * q2 + b * q3) ~= a * dot(q1, q2) + b * dot(q1, q3)
            /// ```
            #[test]
            fn prop_quaternion_dot_product_right_bilinear(
                a in $ScalarGen::<$ScalarType>(), b in $ScalarGen::<$ScalarType>(),
                q1 in $Generator::<$ScalarType>(),
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()) {

                prop_assume!((q1.dot(a * q2 + b * q3)).is_finite());
                prop_assume!((a * q1.dot(q2) + b * q1.dot(q3)).is_finite());
                prop_assert!(
                    relative_eq!(q1.dot(a * q2 + b * q3), a * q1.dot(q2) + b * q1.dot(q3), epsilon = $tolerance)
                );
            }

            /// The dot product of quaternions over floating point scalars is 
            /// approximately left bilinear.
            ///
            /// Given quaternions `q1`, `q2` and `q3`, and scalars `a` and `b`
            /// ```text
            /// dot(a * q1 + b * q2, q3) ~= a * dot(q1, q3) + b * dot(q2, q3)
            /// ```
            #[test]
            fn prop_quaternion_dot_product_left_bilinear(
                a in $ScalarGen::<$ScalarType>(), b in $ScalarGen::<$ScalarType>(),
                q1 in $Generator::<$ScalarType>(),
                q2 in $Generator::<$ScalarType>(), q3 in $Generator::<$ScalarType>()) {

                prop_assume!(((a * q1 + b * q2).dot(q3)).is_finite());
                prop_assume!((a * q1.dot(q3) + b * q2.dot(q3)).is_finite());
                prop_assert!(relative_eq!(
                    (a * q1 + b * q2).dot(q3), a * q1.dot(q3) + b * q2.dot(q3), epsilon = $tolerance
                ));
            }
        }
    }
    }
}

approx_dot_product_props!(quaternion_f64_dot_product_props, f64, any_quaternion, any_scalar, 1e-7);


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
        use cglinalg::DotProduct;
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

                prop_assert_eq!(q1.dot(q2), q2.dot(q1));

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
            
                prop_assert_eq!(q1.dot(q2 + q3), q1.dot(q2) + q1.dot(q3));
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
            
                prop_assert_eq!((q1 + q2).dot(q3), q1.dot(q3) + q2.dot(q3));
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

                prop_assert_eq!((a * q1).dot(b * q2), a * b * q1.dot(q2));
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

                prop_assert_eq!(q1.dot(a * q2 + b * q3), a * q1.dot(q2) + b * q1.dot(q3));
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

                prop_assert_eq!((a * q1 + b * q2).dot(q3), a * q1.dot(q3) + b * q2.dot(q3));
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
macro_rules! conjugation_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::approx::{
            relative_eq,
        };
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

                prop_assert!(
                    relative_eq!((q1 + q2).conjugate(), q1.conjugate() + q2.conjugate(), epsilon = $tolerance)
                );
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

                prop_assert!(
                    relative_eq!((q1 * q2).conjugate(), q2.conjugate() * q1.conjugate(), epsilon = $tolerance)
                );
            } 
        }
    }
    }
}

conjugation_props!(quaternion_f64_conjugation_props, f64, any_quaternion, 1e-7);


/// Generate property tests for quaternion magnitudes.
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
macro_rules! magnitude_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::Magnitude;
        use cglinalg::approx::{
            relative_eq,
            relative_ne
        };
        use super::{
            $Generator,
            $ScalarGen,
        };


        proptest! {
            #[test]
            /// The magnitude of a quaternion preserves scales. 
            /// 
            /// Given a scalar constant `c`, and a quaternion `q` of scalars, the 
            /// magnitude function satisfies
            /// ```text
            /// magnitude(c * q) = abs(c) * magnitude(q)
            /// ```
            fn prop_magnitude_preserves_scale(
                q in $Generator::<$ScalarType>(), c in $ScalarGen::<$ScalarType>()) {
                
                let abs_c = <$ScalarType as num_traits::Float>::abs(c);   

                prop_assert!(
                    relative_eq!( (c * q).magnitude(), abs_c * q.magnitude(), epsilon = $tolerance),
                    "\n||c * q|| = {}\n|c| * ||q|| = {}\n", (c * q).magnitude(), abs_c * q.magnitude(),
                );
            }

            /// The magnitude of a quaternion is nonnegative. 
            ///
            /// Given a quaternion `q`
            /// ```text
            /// magnitude(q) >= 0
            /// ```
            #[test]
            fn prop_magnitude_nonnegative(q in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();

                prop_assert!(q.magnitude() >= zero);
            }

            /// The magnitude of a quaternion satisfies the triangle inequality. 
            ///
            /// Given a quaternions `q1` and `q2`, the magnitude function satisfies
            /// ```text
            /// magnitude(q1 + q2) <= magnitude(q1) + magnitude(q2)
            /// ```
            #[test]
            fn prop_magnitude_satisfies_triangle_inequality(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
            
                prop_assume!((q1 + q2).magnitude().is_finite());
                prop_assume!((q1.magnitude() + q2.magnitude()).is_finite());
                prop_assert!((q1 + q2).magnitude() <= q1.magnitude() + q2.magnitude(), 
                    "\n|q1 + q2| = {}\n|q1| = {}\n|q2| = {}\n|q1| + |q2| = {}\n",
                    (q1 + q2).magnitude(), q1.magnitude(), q2.magnitude(), q1.magnitude() + q2.magnitude()
                );
            }

            /// The magnitude function is point separating. In particular, if 
            /// the distance between two quaternions `q1` and `q2` is 
            /// zero, then q1 = q2.
            ///
            /// Given quaternions `q1` and `q2`
            /// ```text
            /// magnitude(q1 - q2) = 0 => q1 = q2 
            /// ```
            /// Equivalently, if `q1` is not equal to `q2`, then their distance is 
            /// nonzero
            /// ```text
            /// q1 != q2 => magnitude(q1 - q2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the 
            /// magnitude function.
            #[test]
            fn prop_magnitude_approx_point_separating(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {
                
                let zero: $ScalarType = num_traits::zero();

                prop_assume!(relative_ne!(q1, q2, epsilon = $tolerance));
                prop_assert!(relative_ne!((q1 - q2).magnitude(), zero, epsilon = $tolerance),
                    "\n|q1 - q2| = {}\n", (q1 - q2).magnitude()
                );
            }
        }
    }
    }
}

magnitude_props!(quaternion_f64_magnitude_props, f64, any_quaternion, any_scalar, 1e-7);


/// Generate property tests for quaternion spherical linear interpolation (slerp).
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
macro_rules! slerp_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::Slerp;
        use super::{
            $Generator,
        };


        proptest! {
            /// Quaternion spherical linear interpolation should act like a 
            /// quaternion rotor between two quaternions.
            ///
            /// Given quaternions `q1` and `q2`
            /// ```text
            /// slerp(q1, q2, t) = q1 * (inverse(q1) * q2) ^ t
            /// ```
            #[test]
            fn prop_quaternion_slerp_as_quaternion_rotor(
                q1 in $Generator::<$ScalarType>(), q2 in $Generator::<$ScalarType>()) {

                prop_assume!(q1.is_invertible());
                prop_assume!(q2.is_invertible());
                prop_assert!(false);
            }

            /// Quaternion spherical linear interpolation should yield the 
            /// respective interpolants at the endpoints.
            ///
            /// Given quaternions `q0` and `q1`
            /// ```text
            /// slerp(q0, q1, 0) = q0
            /// slerp(q0, q1, 1) = q1
            /// ```
            #[test]
            fn prop_quaternion_slerp_endpoints0(
                q0 in $Generator::<$ScalarType>(), q1 in $Generator::<$ScalarType>()) {

                prop_assert_eq!(q0.slerp(q1, 0.0), q0);
            }

            /// Quaternion spherical linear interpolation should yield the 
            /// respective interpolants at the endpoints.
            ///
            /// Given quaternions `q0` and `q1`
            /// ```text
            /// slerp(q0, q1, 0) = q0
            /// slerp(q0, q1, 1) = q1
            /// ```
            #[test]
            fn prop_quaternion_slerp_endpoints1(
                q0 in $Generator::<$ScalarType>(), q1 in $Generator::<$ScalarType>()) {

                prop_assert_eq!(q0.slerp(q1, 1.0), q1);
            }
        }
    }
    }
}

slerp_props!(quaternion_f64_slerp_props, f64, any_unit_quaternion, 1e-7);



/// Generate property tests for quaternion exponentiation and natural logarithms.
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
macro_rules! exp_log_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::{
            Quaternion,
            Zero,
            Identity
        };
        use cglinalg::approx::{
            relative_eq,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// Quaternion exponentiation commutes with quaternion conjugation.
            ///
            /// Given a quaternion `q`
            /// ```text
            /// conjugate(exp(q)) = exp(conjugate(q))
            /// ```
            #[test]
            fn prop_quaternion_conjugation_exp_commutes(q in $Generator::<$ScalarType>()) {
                prop_assert_eq!(q.exp().conjugate(), q.conjugate().exp());
            }

            /// The exponential of a quaternion is never zero.
            ///
            /// Given a quaternion `q` and the zero quaternion `0`
            /// ```text
            /// exp(q) != 0
            /// ```
            #[test]
            fn prop_quaternion_exp_nonzero(q in $Generator::<$ScalarType>()) {
                prop_assert!(q.exp() != Quaternion::zero());
            }

            /// Every quaternion exponential has a multiplicative inverse.
            ///
            /// Given a quaternion `q` and the unit quaternion `1`
            /// ```text
            /// exp(-q) * exp(q) = exp(q) * exp(-q) = 1
            /// ```
            #[test]
            fn prop_quaternion_exp_inverse(q in $Generator::<$ScalarType>()) {
                prop_assert!(
                    relative_eq!((-q).exp() * q.exp(), Quaternion::identity(), epsilon = $tolerance)
                );
                prop_assert!(
                    relative_eq!(q.exp() * (-q).exp(), Quaternion::identity(), epsilon = $tolerance)
                );
            }

            /// The quaternion logarithm is the inverse of the quaternion exponential.
            ///
            /// Given a quaternion `q`
            /// ```text
            /// exp(ln(q)) = ln(exp(q)) = q
            /// ```
            #[test]
            fn prop_quaternion_exp_log_inverses(q in $Generator::<$ScalarType>()) {
                prop_assert!(relative_eq!(q.ln().exp(), q, epsilon = $tolerance));
                prop_assert!(relative_eq!(q.exp().ln(), q, epsilon = $tolerance));
            }
        }
    }
    }
}

exp_log_props!(quaternion_f64_exp_props, f64, any_quaternion, 1e-7);

