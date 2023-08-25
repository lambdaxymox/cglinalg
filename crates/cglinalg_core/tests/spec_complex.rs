extern crate cglinalg_core;
extern crate num_traits;
extern crate proptest;


use std::ops::RangeInclusive;

use proptest::prelude::*;
use cglinalg_core::{
    Complex, 
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

fn any_complex<S>() -> impl Strategy<Value = Complex<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(re, im)| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let complex = Complex::new(re, im);

        complex % modulus
    })
    .no_shrink()
}


/// Generate property tests for complex number arithmetic over exact scalars. We 
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
///    set of complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_arithmetic_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Complex,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// A scalar `0` times a complex number should be a zero complex number.
            ///
            /// Given a complex number `z`, it satisfies
            /// ```text
            /// 0 * z = 0.
            /// ```
            #[test]
            fn prop_zero_times_complex_equals_zero(z in $Generator()) {
                let zero: $ScalarType = num_traits::zero();
                let zero_complex = Complex::zero();

                prop_assert_eq!(zero * z, zero_complex);
            }
        
            /// A scalar `0` times a complex number should be zero.
            ///
            /// Given a complex number `z`, it satisfies
            /// ```text
            /// z * 0 = 0
            /// ```
            #[test]
            fn prop_complex_times_zero_equals_zero(z in $Generator()) {
                let zero: $ScalarType = num_traits::zero();
                let zero_complex = Complex::zero();

                prop_assert_eq!(z * zero, zero_complex);
            }

            /// A zero complex number should act as the additive unit element of a set 
            /// of complex numbers.
            ///
            /// Given a complex number `z`
            /// ```text
            /// z + 0 = z
            /// ```
            #[test]
            fn prop_complex_plus_zero_equals_complex(z in $Generator()) {
                let zero_complex = Complex::<$ScalarType>::zero();

                prop_assert_eq!(z + zero_complex, z);
            }

            /// A zero complex number should act as the additive unit element of a set 
            /// of complex numbers.
            ///
            /// Given a complex number `z`
            /// ```text
            /// 0 + z = z
            /// ```
            #[test]
            fn prop_zero_plus_complex_equals_complex(z in $Generator()) {
                let zero_complex = Complex::<$ScalarType>::zero();

                prop_assert_eq!(zero_complex + z, z);
            }

            /// Multiplying a complex number by a scalar `1` should give the original 
            /// complex number.
            ///
            /// Given a complex number `z`
            /// ```text
            /// 1 * z = z
            /// ```
            #[test]
            fn prop_one_times_complex_equal_complex(z in $Generator()) {
                let one: $ScalarType = num_traits::one();

                prop_assert_eq!(one * z, z);
            }

            /// Multiplying a complex number by a scalar `1` should give the original 
            /// complex number.
            ///
            /// Given a complex number `z`
            /// ```text
            /// z * 1 = z.
            /// ```
            #[test]
            fn prop_complex_times_one_equals_complex(z in $Generator()) {
                let one: $ScalarType = num_traits::one();

                prop_assert_eq!(one * z, z);
            }
        }
    }
    }
}

exact_arithmetic_props!(complex_f64_arithmetic_props, f64, any_complex);
exact_arithmetic_props!(complex_i32_arithmetic_props, i32, any_complex);
exact_arithmetic_props!(complex_u32_arithmetic_props, u32, any_complex);


/// Generate property tests for complex number arithmetic over floating point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_add_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Complex, 
        };
        use approx::{
            relative_eq,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// A complex number plus a zero complex number equals the same complex number. 
            ///
            /// Given a complex number `z`
            /// ```text
            /// z + 0 = z
            /// ```
            #[test]
            fn prop_complex_plus_zero_equals_complex(z in $Generator()) {
                let zero_complex = Complex::<$ScalarType>::zero();

                prop_assert_eq!(z + zero_complex, z);
            }

            /// A complex number plus a zero complex number equals the same complex number.
            /// 
            /// Given a complex number `z`
            /// ```text
            /// 0 + z = z
            /// ```
            #[test]
            fn prop_zero_plus_complex_equals_complex(z in $Generator()) {
                let zero_complex = Complex::<$ScalarType>::zero();

                prop_assert_eq!(zero_complex + z, z);
            }

            /// Given complex numbers `z1` and `z2`, we should be able to use `z1` 
            /// and `z2` interchangeably with their references `&z1` and `&z2` in 
            /// arithmetic expressions involving complex numbers.
            ///
            /// Given complex numbers `z1` and `z2`, and their references `&z1` 
            /// and `&z2`, they should satisfy
            /// ```text
            ///  z1 +  z2 = &z1 +  z2
            ///  z1 +  z2 =  z1 + &z2
            ///  z1 +  z2 = &z1 + &z2
            ///  z1 + &z2 = &z1 +  z2
            /// &z1 +  z2 =  z1 + &z2
            /// &z1 +  z2 = &z1 + &z2
            ///  z1 + &z2 = &z1 + &z2
            /// ```
            #[test]
            fn prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!( z1 +  z2, &z1 +  z2);
                prop_assert_eq!( z1 +  z2,  z1 + &z2);
                prop_assert_eq!( z1 +  z2, &z1 + &z2);
                prop_assert_eq!( z1 + &z2, &z1 +  z2);
                prop_assert_eq!(&z1 +  z2,  z1 + &z2);
                prop_assert_eq!(&z1 +  z2, &z1 + &z2);
                prop_assert_eq!( z1 + &z2, &z1 + &z2);
            }

            /// Complex number addition over floating point scalars should be commutative.
            /// 
            /// Given complex numbers `z1` and `z2`, we have
            /// ```text
            /// z1 + z2 = z2 + z1
            /// ```
            #[test]
            fn prop_complex_addition_commutative(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!(z1 + z2, z2 + z1);
            }

            /// Complex number addition over floating point scalars should be 
            /// approximately associative. 
            ///
            /// Given complex numbers `z1`, `z2`, and `z3` we have
            /// ```text
            /// (z1 + z2) + z3 ~= z1 + (z2 + z3).
            /// ```
            #[test]
            fn prop_complex_addition_almost_associative(
                z1 in $Generator::<$ScalarType>(), 
                z2 in $Generator::<$ScalarType>(), z3 in $Generator::<$ScalarType>()) {

                prop_assert!(relative_eq!((z1 + z2) + z3, z1 + (z2 + z3), epsilon = $tolerance));
            }
        }
    }
    }
}

approx_add_props!(complex_f64_add_props, f64, any_complex, 1e-7);


/// Generate property tests for complex number arithmetic over exact scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_add_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Complex, 
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// A complex number plus a zero complex number equals the same complex.
            ///
            /// Given a complex number `z`, it should satisfy
            /// ```text
            /// z + 0 = z
            /// ```
            #[test]
            fn prop_complex_plus_zero_equals_complex(z in $Generator()) {
                let zero_complex = Complex::<$ScalarType>::zero();

                prop_assert_eq!(z + zero_complex, z);
            }

            /// A zero complex number plus a complex number equals the same complex number.
            ///
            /// Given a complex number `z`, it should satisfy
            /// ```text
            /// 0 + z = z
            /// ```
            #[test]
            fn prop_zero_plus_complex_equals_complex(z in $Generator()) {
                let zero_complex = Complex::<$ScalarType>::zero();

                prop_assert_eq!(zero_complex + z, z);
            }

            /// Given complex numbers `z1` and `z2`, we should be able to use `z1` 
            /// and `z2` interchangeably with their references `&z1` and `&z2` in 
            /// arithmetic expressions involving complex numbers.
            ///
            /// Given complex numbers `z1` and `z2`, and their references `&z1` and `&z2`, they 
            /// should satisfy
            /// ```text
            ///  z1 +  z2 = &z1 +  z2
            ///  z1 +  z2 =  z1 + &z2
            ///  z1 +  z2 = &z1 + &z2
            ///  z1 + &z2 = &z1 +  z2
            /// &z1 +  z2 =  z1 + &z2
            /// &z1 +  z2 = &z1 + &z2
            ///  z1 + &z2 = &z1 + &z2
            /// ```
            #[test]
            fn prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex2(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!( z1 +  z2, &z1 +  z2);
                prop_assert_eq!( z1 +  z2,  z1 + &z2);
                prop_assert_eq!( z1 +  z2, &z1 + &z2);
                prop_assert_eq!( z1 + &z2, &z1 +  z2);
                prop_assert_eq!(&z1 +  z2,  z1 + &z2);
                prop_assert_eq!(&z1 +  z2, &z1 + &z2);
                prop_assert_eq!( z1 + &z2, &z1 + &z2);
            }

            /// Complex number addition over integer scalars should be commutative.
            ///
            /// Given complex numbers `z1` and `z2`, we have
            /// ```text
            /// z1 + z2 = z2 + z1.
            /// ```
            #[test]
            fn prop_complex_addition_commutative(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!(z1 + z2, z2 + z1);
            }

            /// Given three complex numbers of integer scalars, complex number addition 
            /// should be associative.
            ///
            /// Given complex numbers `z1`, `z2`, and `z3`, we have
            /// ```text
            /// (z1 + z2) + z3 = z1 + (z2 + z3)
            /// ```
            #[test]
            fn prop_complex_addition_associative(
                z1 in $Generator::<$ScalarType>(), 
                z2 in $Generator::<$ScalarType>(), z3 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((z1 + z2) + z3, z1 + (z2 + z3));
            }
        }
    }
    }
}

exact_add_props!(complex_i32_add_props, i32, any_complex);
exact_add_props!(complex_u32_add_props, u32, any_complex);


/// Generate property tests for complex number subtraction over floating point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_sub_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Complex,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The zero complex number over floating point scalars should act as an 
            /// additive unit.
            ///
            /// Given a complex number `z`, we have
            /// ```text
            /// z - 0 = z
            /// ```
            #[test]
            fn prop_complex_minus_zero_equals_complex(q in $Generator()) {
                let zero_quat = Complex::<$ScalarType>::zero();

                prop_assert_eq!(q - zero_quat, q);
            }

            /// Every complex number should have an additive inverse.
            ///
            /// Given a complex number `z`, there is a complex number `-z` such that
            /// ```text
            /// z - z = z + (-z) = (-z) + z = 0
            /// ```
            #[test]
            fn prop_complex_minus_complex_equals_zero(q in $Generator::<$ScalarType>()) {
                let zero_quat = Complex::<$ScalarType>::zero();

                prop_assert_eq!(q - q, zero_quat);
                prop_assert_eq!((-q) + q, zero_quat);
                prop_assert_eq!(q + (-q), zero_quat);
            }

            /// Given complex numbers `z1` and `z2`, we should be able to use `z1` and 
            /// `z2` interchangeably with their references `&z1` and `&z2` in 
            /// arithmetic expressions involving complex numbers.
            ///
            /// Given complex numbers `z1` and `z2`, and their references `&z1` and 
            /// `&z2`, they should satisfy
            /// ```text
            ///  z1 -  z2 = &z1 -  z2
            ///  z1 -  z2 =  z1 - &z2
            ///  z1 -  z2 = &z1 - &z2
            ///  z1 - &z2 = &z1 -  z2
            /// &z1 -  z2 =  z1 - &z2
            /// &z1 -  z2 = &z1 - &z2
            ///  z1 - &z2 = &z1 - &z2
            /// ```
            #[test]
            fn prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex2(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!( z1 -  z2, &z1 -  z2);
                prop_assert_eq!( z1 -  z2,  z1 - &z2);
                prop_assert_eq!( z1 -  z2, &z1 - &z2);
                prop_assert_eq!( z1 - &z2, &z1 -  z2);
                prop_assert_eq!(&z1 -  z2,  z1 - &z2);
                prop_assert_eq!(&z1 -  z2, &z1 - &z2);
                prop_assert_eq!( z1 - &z2, &z1 - &z2);
            }
        }
    }
    }
}

approx_sub_props!(complex_f64_sub_props, f64, any_complex, 1e-7);


/// Generate property tests for complex number arithmetic over exact scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_sub_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Complex,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The zero complex number should act as an additive unit. 
            ///
            /// Given a complex number `z`, we have
            /// ```text
            /// z - 0 = z
            /// ```
            #[test]
            fn prop_complex_minus_zero_equals_complex(z in $Generator()) {
                let zero_complex = Complex::<$ScalarType>::zero();

                prop_assert_eq!(z - zero_complex, z);
            }

            /// Every complex number should have an additive inverse. 
            ///
            /// Given a complex number `z`, there is a complex number `-z` such that
            /// ```text
            /// z - z = z + (-z) = (-z) + z = 0
            /// ```
            #[test]
            fn prop_complex_minus_complex_equals_zero(z in $Generator::<$ScalarType>()) {
                let zero_quat = Complex::<$ScalarType>::zero();

                prop_assert_eq!(z - z, zero_quat);
            }

            /// Given complex numbers `z1` and `z2`, we should be able to use `z1` 
            /// and `z2` interchangeably with their references `&z1` and `&z2` 
            /// in arithmetic expressions involving complex numbers.
            ///
            /// Given complex numbers `z1` and `z2`, and their references `&z1` 
            /// and `&z2`, they should satisfy
            /// ```text
            ///  z1 -  z2 = &z1 -  z2
            ///  z1 -  z2 =  z1 - &z2
            ///  z1 -  z2 = &z1 - &z2
            ///  z1 - &z2 = &z1 -  z2
            /// &z1 -  z2 =  z1 - &z2
            /// &z1 -  z2 = &z1 - &z2
            ///  z1 - &z2 = &z1 - &z2
            /// ```
            #[test]
            fn prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex2(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!( z1 -  z2, &z1 -  z2);
                prop_assert_eq!( z1 -  z2,  z1 - &z2);
                prop_assert_eq!( z1 -  z2, &z1 - &z2);
                prop_assert_eq!( z1 - &z2, &z1 -  z2);
                prop_assert_eq!(&z1 -  z2,  z1 - &z2);
                prop_assert_eq!(&z1 -  z2, &z1 - &z2);
                prop_assert_eq!( z1 - &z2, &z1 - &z2);
            }
        }
    }
    }
}

exact_sub_props!(complex_i32_sub_props, i32, any_complex);
exact_sub_props!(complex_u32_sub_props, u32, any_complex);


/// Generate property tests for complex number multiplication over floating point 
/// scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of complex numbers.
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
            Complex,
        };
        use super::{
            $Generator,
            $ScalarGen,
        };


        proptest! {
            /// Multiplication of a scalar and a complex number should be commutative.
            ///
            /// Given a constant `c` and a complex number `z`
            /// ```text
            /// c * z = z * c
            /// ```
            #[test]
            fn prop_scalar_times_complex_equals_complex_times_scalar(
                c in $ScalarGen::<$ScalarType>(), z in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!(c * z, z * c);
            }

            /// Complexs have a multiplicative unit element.
            ///
            /// Given a complex number `z`, and the unit complex number `1`, we have
            /// ```text
            /// z * 1 = 1 * z = z
            /// ```
            #[test]
            fn prop_complex_multiplicative_unit(z in $Generator::<$ScalarType>()) {
                let one = Complex::identity();

                prop_assert_eq!(z * one, z);
                prop_assert_eq!(one * z, z);
                prop_assert_eq!(z * one, one * z);
            }

            /// Every nonzero complex number over floating point scalars has an 
            /// approximate multiplicative inverse.
            ///
            /// Given a complex number `z` and its inverse `z_inv`, we have
            /// ```text
            /// z * z_inv ~= z_inv * z ~= 1
            /// ```
            #[test]
            fn prop_complex_multiplicative_inverse(z in $Generator::<$ScalarType>()) {
                prop_assume!(z.is_finite());
                prop_assume!(z.is_invertible());

                let one = Complex::identity();
                let z_inv = z.inverse().unwrap();

                prop_assert!(relative_eq!(z * z_inv, one, epsilon = $tolerance));
                prop_assert!(relative_eq!(z_inv * z, one, epsilon = $tolerance));
            }

            /// Complex multiplication over floating point scalars is approximately
            /// commutative.
            /// 
            /// Given a complex number `z1`, and another complex number `z2`, we
            /// have
            /// ```text
            /// z1 * z2 ~= z2 * z1
            /// ```
            #[test]
            fn prop_complex_multiplication_commutative(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {

                prop_assert!(relative_eq!(z1 * z2, z2 * z1, epsilon = $tolerance));
            }
        }
    }
    }
}

approx_mul_props!(complex_f64_mul_props, f64, any_complex, any_scalar, 1e-7);


/// Generate property tests for complex number multiplication over exact scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_mul_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Complex,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// Multiplication of an integer scalar and a complex number over integer 
            /// scalars should be commutative.
            ///
            /// Given a constant `c` and a complex number `z`
            /// ```text
            /// c * z = z * c
            /// ```
            #[test]
            fn prop_scalar_times_complex_equals_complex_times_scalar(
                c in any::<$ScalarType>(), z in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!(c * z, z * c);
            }

            /// Exact multiplication of two scalars and a complex number should be 
            /// compatible with multiplication of all scalars. 
            ///
            /// In other words, scalar multiplication of two scalars with a 
            /// complex number should act associatively just like the multiplication 
            /// of three scalars. 
            ///
            /// Given scalars `a` and `b`, and a complex number `z`, we have
            /// ```text
            /// (a * b) * z = a * (b * z)
            /// ```
            #[test]
            fn prop_scalar_multiplication_compatibility(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), z in $Generator::<$ScalarType>()) {

                prop_assert_eq!(a * (b * z), (a * b) * z);
            }

            /// Complex number multiplication over integer scalars is exactly associative.
            ///
            /// Given complex numbers `z1`, `z2`, and `z3`, we have
            /// ```text
            /// (z1 * z2) * z3 = z1 * (z2 * z3)
            /// ```
            #[test]
            fn prop_complex_multiplication_associative(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>(), 
                z3 in $Generator::<$ScalarType>()
            ) {
                prop_assert_eq!(z1 * (z2 * z3), (z1 * z2) * z3);
            }

            /// Complex numbers have a multiplicative unit element.
            ///
            /// Given a complex number `z`, and the unit complex number `1`, we have
            /// ```text
            /// z * 1 = 1 * z = z
            /// ```
            #[test]
            fn prop_complex_multiplicative_unit(z in $Generator::<$ScalarType>()) {
                let one = Complex::identity();
                prop_assert_eq!(z * one, z);
                prop_assert_eq!(one * z, z);
                prop_assert_eq!(z * one, one * z);
            }

            /// Multiplication of complex numbers over integer scalars is commutative.
            /// 
            /// Given a complex number `z1` and another complex number `z2`, we have
            /// ```text
            /// z1 * z2 = z2 * z1
            /// ```
            #[test]
            fn prop_complex_multiplication_commutative(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!(z1 * z2, z2 * z1);
            }
        }
    }
    }
}

exact_mul_props!(complex_i32_mul_props, i32, any_complex);
exact_mul_props!(complex_u32_mul_props, u32, any_complex);


/// Generate property tests for complex number distribution over exact scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of complex numbers.
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
            /// Scalar multiplication should distribute over complex number addition.
            ///
            /// Given a scalar `a` and complex numbers `z1` and `z2`
            /// ```text
            /// a * (z1 + z2) = a * z1 + a * z2
            /// ```
            #[test]
            fn prop_distribution_over_complex_addition(
                a in any::<$ScalarType>(), 
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!(a * (z1 + z2), a * z1 + a * z2);
                prop_assert_eq!((z1 + z2) * a,  z1 * a + z2 * a);
            }

            /// Multiplication of a sum of scalars should distribute over a 
            /// complex number.
            ///
            /// Given scalars `a` and `b` and a complex number `z`, we have
            /// ```text
            /// (a + b) * z = a * z + b * z
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                z in $Generator::<$ScalarType>()) {
    
                prop_assert_eq!((a + b) * z, a * z + b * z);
                prop_assert_eq!(z * (a + b), z * a + z * b);
            }

            /// Multiplication of two complex numbers by a scalar on the right 
            /// should distribute.
            ///
            /// Given complex numbers `z1` and `z2`, and a scalar `a`
            /// ```text
            /// (z1 + z2) * a = z1 * a + z2 * a
            /// ```
            #[test]
            fn prop_distribution_over_complex_addition1(
                a in any::<$ScalarType>(), 
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {
                    
                prop_assert_eq!((z1 + z2) * a,  z1 * a + z2 * a);
            }

            /// Multiplication of a complex number on the right by the sum of two 
            /// scalars should distribute over the two scalars. 
            ///
            /// Given a complex number `z` and scalars `a` and `b`
            /// ```text
            /// z * (a + b) = z * a + z * b
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition1(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                z in $Generator::<$ScalarType>()) {
    
                prop_assert_eq!(z * (a + b), z * a + z * b);
            }

            /// Complex number multiplication should be distributive on the right.
            ///
            /// Given three complex numbers `z1`, `z2`, and `z3`
            /// ```text
            /// (z1 + z2) * z3 = z1 * z3 + z2 * z3
            /// ```
            #[test]
            fn prop_complex_multiplication_right_distributive(
                z1 in $Generator::<$ScalarType>(), 
                z2 in $Generator::<$ScalarType>(), z3 in $Generator::<$ScalarType>()
            ) {
                prop_assert_eq!((z1 + z2) * z3, z1 * z3 + z2 * z3);
            }

            /// Complex Number multiplication should be distributive on the left.
            ///
            /// Given three complex numbers `z1`, `z2`, and `z3`
            /// ```text
            /// z1 * (z2 + z3) = z1 * z2 + z1 * z3
            /// ```
            #[test]
            fn prop_complex_multiplication_left_distributive(
                z1 in $Generator::<$ScalarType>(), 
                z2 in $Generator::<$ScalarType>(), z3 in $Generator::<$ScalarType>()
            ) {
                prop_assert_eq!((z1 + z2) * z3, z1 * z3 + z2 * z3);
            }
        }
    }
    }    
}

exact_distributive_props!(complex_i32_distributive_props, i32, any_complex);
exact_distributive_props!(complex_u32_distributive_props, u32, any_complex);


/// Generate property tests for complex number conjugation over floating point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of complex numbers.
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
            /// Conjugating a complex number twice should give the original complex number.
            ///
            /// Given a complex number `z`
            /// ```text
            /// z** = conjugate(conjugate(z)) = z
            /// ```
            #[test]
            fn prop_complex_conjugate_conjugate_equals_complex(z in $Generator::<$ScalarType>()) {
                prop_assert_eq!(z.conjugate().conjugate(), z);
            }

            /// Quaternion conjugation is linear.
            ///
            /// Given complex numbers `z1` and `z2`, complex number conjugation satisfies
            /// ```text
            /// conjugate(z1 + z2) = conjugate(z1) + conjugate(z2)
            /// ```
            #[test]
            fn prop_complex_conjugation_linear(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((z1 + z2).conjugate(), z1.conjugate() + z2.conjugate());
            }
        }
    }
    }
}

approx_conjugation_props!(complex_f64_conjugation_props, f64, any_complex);


/// Generate property tests for complex number conjugation over exact scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of complex numbers.
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
            /// Conjugating a complex number twice should give the original complex number.
            ///
            /// Given a complex number `z`
            /// ```text
            /// z** = conjugate(conjugate(z)) = z
            /// ```
            #[test]
            fn prop_complex_conjugate_conjugate_equals_complex(z in $Generator::<$ScalarType>()) {
                prop_assert_eq!(z.conjugate().conjugate(), z);
            }

            /// Complex conjugation is linear.
            ///
            /// Given complex numbers `z1` and `z2`, complex number conjugation satisfies
            /// ```text
            /// conjugate(z1 + z2) = conjugate(z1) + conjugate(z2)
            /// ```
            #[test]
            fn prop_complex_conjugation_linear(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((z1 + z2).conjugate(), z1.conjugate() + z2.conjugate());
            }

            /// Complex multiplication transposes under conjugation.
            ///
            /// Given complex numbers `z1` and `z2`
            /// ```text
            /// conjugate(z1 * z2) = conjugate(z2) * conjugate(z1)
            /// ```
            #[test]
            fn prop_complex_conjugation_transposes_products(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {

                prop_assert_eq!((z1 * z2).conjugate(), z2.conjugate() * z1.conjugate());
            }
        }
    }
    }
}

exact_conjugation_props!(complex_i32_conjugation_props, i32, any_complex);
exact_conjugation_props!(complex_i64_conjugation_props, i64, any_complex);


fn any_complex_modulus_squared_f64<S>() -> impl Strategy<Value = Complex<f64>> {
    use cglinalg_core::Radians;

    any::<(f64, f64)>().prop_map(|(_scale, _angle)| {
        let min_scale = f64::sqrt(f64::EPSILON);
        let max_scale = f64::sqrt(f64::MAX);
        let scale = min_scale + (_scale % (max_scale - min_scale));
        let angle = Radians(_angle % core::f64::consts::FRAC_PI_2);

        Complex::from_polar_decomposition(scale, angle)
    })
    .no_shrink()
}

/// Generate property tests for the complex number squared modulus.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$input_tolerance` specifies the amount of acceptable error in the input for 
///    a correct operation with floating point scalars.
/// * `$output_tolerance` specifies the amount of acceptable error in the input for 
///    a correct operation with floating point scalars.
macro_rules! approx_modulus_squared_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $input_tolerance:expr, $output_tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use approx::{
            relative_ne,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The squared modulus of a complex number is nonnegative. 
            ///
            /// Given a complex number `z`
            /// ```text
            /// modulus_squared(z) >= 0
            /// ```
            #[test]
            fn prop_modulus_squared_nonnegative(z in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();

                prop_assert!(z.modulus_squared() >= zero);
            }

            /// The squared modulus function is point separating. In particular, if 
            /// the squared distance between two complex numbers `z1` and `z2` is 
            /// zero, then `z1 = z2`.
            ///
            /// Given complex numbers `z1` and `z2`
            /// ```text
            /// modulus_squared(z1 - z2) = 0 => z1 = z2 
            /// ```
            /// Equivalently, if `z1` is not equal to `z2`, then their squared distance is 
            /// nonzero
            /// ```text
            /// z1 != z2 => modulus_squared(z1 - z2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the 
            /// norm function.
            #[test]
            fn prop_modulus_squared_approx_point_separating(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {

                prop_assume!(relative_ne!(z1, z2, epsilon = $input_tolerance));
                prop_assert!(
                    // relative_ne!((z1 - z2).modulus_squared(), zero, epsilon = $output_tolerance),
                    (z1 - z2).modulus_squared() > $output_tolerance,
                    "\n|z1 - z2|^2 = {:e}\n",
                    (z1 - z2).modulus_squared()
                );
            }
        }
    }
    }
}

approx_modulus_squared_props!(complex_f64_modulus_squared_props, f64, any_complex_modulus_squared_f64, any_scalar, 1e-10, 1e-20);


/// Generate property tests for complex number squared modulus.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! approx_modulus_squared_synonym_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The [`Complex::magnitude_squared`] function and the [`Complex::modulus_squared`]
            /// function are synonyms. In particular, given a complex number `z`
            /// ```text
            /// magnitude_squared(z) = modulus_squared(z)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_magnitude_squared_modulus_squared_synonyms(z in $Generator::<$ScalarType>()) {
                prop_assert_eq!(z.magnitude_squared(), z.modulus_squared());
            }

            /// The [`Complex::norm_squared`] function and the [`Complex::modulus_squared`]
            /// functions are synonyms. In particular, given a complex number `z`
            /// ```text
            /// norm_squared(z) = modulus_squared(z)
            /// ```
            #[test]
            fn prop_norm_squared_modulus_squared_synonyms(z in $Generator::<$ScalarType>()) {
                prop_assert_eq!(z.norm_squared(), z.modulus_squared());
            }
        }
    }
    }
}

approx_modulus_squared_synonym_props!(complex_f64_modulus_squared_synonym_props, f64, any_complex);


fn any_complex_modulus_squared_i32<S>() -> impl Strategy<Value = Complex<i32>> {
    any::<(i32, i32)>().prop_map(|(_re, _im)| {
        let min_value = 0;
        // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
        let max_square_root = 46340;
        let max_value = max_square_root / 2;
        let re = min_value + (_re % (max_value - min_value + 1));
        let im = min_value + (_im % (max_value - min_value + 1));
        
        Complex::new(re, im)
    })
    .no_shrink()
}

fn any_complex_modulus_squared_u32<S>() -> impl Strategy<Value = Complex<u32>> {
    any::<(u32, u32)>().prop_map(|(_re, _im)| {
        let min_value = 0;
        // let max_square_root = f64::floor(f64::sqrt(u32::MAX as f64)) as u32;
        let max_square_root = 46340;
        let max_value = max_square_root / 2;
        let re = min_value + (_re % (max_value - min_value + 1));
        let im = min_value + (_im % (max_value - min_value + 1));

        Complex::new(re, im)
    })
    .no_shrink()
}

/// Generate property tests for the complex number squared modulus.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
macro_rules! exact_modulus_squared_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The squared modulus of a complex number is nonnegative. 
            ///
            /// Given a complex number `z`
            /// ```text
            /// modulus_squared(z) >= 0
            /// ```
            #[test]
            fn prop_modulus_squared_nonnegative(z in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();

                prop_assert!(z.modulus_squared() >= zero);
            }

            /// The squared modulus function is point separating. In particular, if 
            /// the squared distance between two complex numbers `z1` and `z2` is 
            /// zero, then `z1 = z2`.
            ///
            /// Given complex numbers `z1` and `z2`
            /// ```text
            /// modulus_squared(z1 - z2) = 0 => z1 = z2 
            /// ```
            /// Equivalently, if `z1` is not equal to `z2`, then their squared distance is 
            /// nonzero
            /// ```text
            /// z1 != z2 => modulus_squared(z1 - z2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the 
            /// norm function.
            #[test]
            fn prop_modulus_squared_point_separating(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {
                
                let zero: $ScalarType = num_traits::zero();

                prop_assume!(z1 != z2);
                prop_assert_ne!(
                    (z1 - z2).modulus_squared(), zero,
                    "\n|z1 - z2|^2 = {}\n",
                    (z1 - z2).modulus_squared()
                );
            }
        }
    }
    }
}

exact_modulus_squared_props!(complex_i32_modulus_squared_props, i32, any_complex_modulus_squared_i32, any_scalar);
exact_modulus_squared_props!(complex_u32_modulus_squared_props, u32, any_complex_modulus_squared_u32, any_scalar);


/// Generate property tests for complex number squared modulus.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_modulus_squared_synonym_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The [`Complex::magnitude_squared`] function and the [`Complex::modulus_squared`]
            /// function are synonyms. In particular, given a complex number `z`
            /// ```text
            /// magnitude_squared(z) = modulus_squared(z)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_magnitude_squared_modulus_squared_synonyms(z in $Generator::<$ScalarType>()) {
                prop_assert_eq!(z.magnitude_squared(), z.modulus_squared());
            }

            /// The [`Complex::norm_squared`] function and the [`Complex::modulus_squared`]
            /// functions are synonyms. In particular, given a complex number `z`
            /// ```text
            /// norm_squared(z) = modulus_squared(z)
            /// ```
            #[test]
            fn prop_norm_squared_modulus_squared_synonyms(z in $Generator::<$ScalarType>()) {
                prop_assert_eq!(z.norm_squared(), z.modulus_squared());
            }
        }
    }
    }
}

exact_modulus_squared_synonym_props!(complex_i32_modulus_squared_synonym_props, i32, any_complex);
exact_modulus_squared_synonym_props!(complex_u32_modulus_squared_synonym_props, u32, any_complex);


/// Generate property tests for the complex number modulus.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! modulus_props {
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
            /// The modulus of a complex number is nonnegative. 
            ///
            /// Given a complex number `z`
            /// ```text
            /// modulus(z) >= 0
            /// ```
            #[test]
            fn prop_modulus_nonnegative(z in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();

                prop_assert!(z.modulus() >= zero);
            }

            /// The modulus function is point separating. In particular, if 
            /// the distance between two complex numbers `z1` and `z2` is 
            /// zero, then `z1 = z2`.
            ///
            /// Given complex numbers `z1` and `z2`
            /// ```text
            /// modulus(z1 - z2) = 0 => z1 = z2 
            /// ```
            /// Equivalently, if `z1` is not equal to `z2`, then their distance is 
            /// nonzero
            /// ```text
            /// z1 != z2 => modulus(z1 - z2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the 
            /// norm function.
            #[test]
            fn prop_modulus_approx_point_separating(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {
                
                let zero: $ScalarType = num_traits::zero();

                prop_assume!(relative_ne!(z1, z2, epsilon = $tolerance));
                prop_assert!(
                    relative_ne!((z1 - z2).modulus(), zero, epsilon = $tolerance),
                    "\n|z1 - z2| = {}\n",
                    (z1 - z2).modulus()
                );
            }
        }
    }
    }
}

modulus_props!(complex_f64_modulus_props, f64, any_complex, any_scalar, 1e-8);


/// Generate property tests for complex number norms.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! modulus_synonym_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The [`Complex::magnitude`] function and the [`Complex::modulus`] function 
            /// are synonyms. In particular, given a complex number `z`
            /// ```text
            /// magnitude(z) = norm(z)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_magnitude_modulus_synonyms(z in $Generator::<$ScalarType>()) {
                prop_assert_eq!(z.magnitude(), z.modulus());
            }

            /// The [`Complex::norm`] function and the [`Complex::modulus`] function
            /// are synonyms. In particular, given a complex number `z`
            /// ```text
            /// norm(z) = modulus(z)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_norm_modulus_synonyms(z in $Generator::<$ScalarType>()) {
                prop_assert_eq!(z.norm(), z.modulus());
            }

            /// The [`Complex::l2_norm`] function and the [`Complex::modulus`] function
            /// are synonyms. In particular, given a complex number `z`
            /// ```text
            /// l2_norm(z) = modulus(z)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_l2_norm_modulus_synonyms(z in $Generator::<$ScalarType>()) {
                prop_assert_eq!(z.l2_norm(), z.modulus());
            }
        }
    }
    }
}

modulus_synonym_props!(complex_f64_modulus_synonym_props, f64, any_complex);


/// Generate property tests for the complex number **L1** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_l1_norm_props {
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
            /// The **L1** norm of a complex number is nonnegative. 
            ///
            /// Given a complex number `z`
            /// ```text
            /// l1_norm(z) >= 0
            /// ```
            #[test]
            fn prop_l1_norm_nonnegative(z in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();

                prop_assert!(z.l1_norm() >= zero);
            }

            /// The **L1** norm function is point separating. In particular, if 
            /// the distance between two complex numbers `z1` and `z2` is 
            /// zero, then `z1 = z2`.
            ///
            /// Given complex numbers `z1` and `z2`
            /// ```text
            /// l1_norm(z1 - z2) = 0 => z1 = z2 
            /// ```
            /// Equivalently, if `z1` is not equal to `z2`, then their distance is 
            /// nonzero
            /// ```text
            /// z1 != z2 => l1_norm(z1 - z2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the 
            /// norm function.
            #[test]
            fn prop_l1_norm_approx_point_separating(
                z1 in $Generator::<$ScalarType>(), 
                z2 in $Generator::<$ScalarType>()
            ) {
                prop_assume!(relative_ne!(z1, z2, epsilon = $tolerance));
                prop_assert!(
                    (z1 - z2).l1_norm() > $tolerance,
                    "\nl1_norm(z1 - z2) = {}\n",
                    (z1 - z2).l1_norm()
                );
            }
        }
    }
    }
}

approx_l1_norm_props!(complex_f64_l1_norm_props, f64, any_complex, any_scalar, 1e-8);


/// Generate property tests for the complex number **L1** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
macro_rules! exact_l1_norm_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The **L1** norm of a complex number is nonnegative. 
            ///
            /// Given a complex number `z`
            /// ```text
            /// l1_norm(z) >= 0
            /// ```
            #[test]
            fn prop_l1_norm_nonnegative(z in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();

                prop_assert!(z.l1_norm() >= zero);
            }

            /// The **L1** norm function is point separating. In particular, if 
            /// the distance between two complex numbers `z1` and `z2` is 
            /// zero, then `z1 = z2`.
            ///
            /// Given complex numbers `z1` and `z2`
            /// ```text
            /// l1_norm(z1 - z2) = 0 => z1 = z2 
            /// ```
            /// Equivalently, if `z1` is not equal to `z2`, then their distance is 
            /// nonzero
            /// ```text
            /// z1 != z2 => l1_norm(z1 - z2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the 
            /// norm function.
            #[test]
            fn prop_l1_norm_point_separating(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {
                
                let zero: $ScalarType = num_traits::zero();

                prop_assume!(z1 != z2);
                prop_assert_ne!(
                    (z1 - z2).l1_norm(), zero,
                    "\nl1_norm(z1 - z2) = {}\n",
                    (z1 - z2).l1_norm()
                );
            }
        }
    }
    }
}

exact_l1_norm_props!(complex_i32_l1_norm_props, i32, any_complex, any_scalar);


/*
fn imaginary_from_range<S>(min_value: S, max_value: S) -> Box<dyn Fn() -> proptest::strategy::NoShrink<proptest::strategy::Map<RangeInclusive<S>, Box<dyn Fn(S) -> Complex<S>>>>>
where 
    S: SimdScalarFloat + Arbitrary + 'static,
    RangeInclusive<S>: Strategy<Value = S>
{
    Box::new(move || { 
        let complex_fn: Box<dyn Fn(S) -> Complex<S>> = Box::new(Complex::from_imaginary);
        
        (min_value..=max_value).prop_map(complex_fn).no_shrink()
    })
}
*/

fn imaginary_from_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Complex<S>>
where 
    S: SimdScalarFloat + Arbitrary,
    RangeInclusive<S>: Strategy<Value = S>
{
    (min_value..=max_value)
        .prop_map(Complex::from_imaginary)
        .no_shrink()
}

fn real_from_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Complex<S>>
where 
    S: SimdScalarFloat + Arbitrary,
    RangeInclusive<S>: Strategy<Value = S>
{
    (min_value..=max_value)
        .prop_map(Complex::from_real)
        .no_shrink()
}

fn complex_from_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Complex<S>>
where 
    S: SimdScalarFloat + Arbitrary,
    RangeInclusive<S>: Strategy<Value = S>
{
    let generator = (min_value..=max_value, min_value..=max_value);
    generator
        .prop_map(|(re, im)| Complex::new(re, im))
        .no_shrink()
}


/// Generate property tests for complex number square roots.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
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
            /// The square of the positive square root of a complex number is the original
            /// complex number.
            /// 
            /// Given a complex number `z`
            /// ```text
            /// sqrt(z) * sqrt(z) == z
            /// ```
            #[test]
            fn prop_positive_square_root_squared(z in $Generator()) {
                let sqrt_z = z.sqrt();

                prop_assert!(
                    relative_eq!(sqrt_z * sqrt_z, z, epsilon = $tolerance, max_relative = $tolerance),
                    "z = {:?}\nsqrt_z = {:?}\nsqrt_z * sqrt_z = {:?}",
                    z, sqrt_z, sqrt_z * sqrt_z
                );
            }

            /// The square of the negative square root of a complex number is the original
            /// complex number.
            /// 
            /// Given a complex number `z`
            /// ```text
            /// -sqrt(z) * -sqrt(z) == z
            /// ```
            #[test]
            fn prop_negative_square_root_squared(z in $Generator()) {
                let minus_sqrt_z = -z.sqrt();

                prop_assert!(
                    relative_eq!(minus_sqrt_z * minus_sqrt_z, z, epsilon = $tolerance, max_relative = $tolerance),
                    "z = {:?}\nminus_sqrt_z = {:?}\nminus_sqrt_z * minus_sqrt_z = {:?}",
                    z, minus_sqrt_z, minus_sqrt_z * minus_sqrt_z
                );
            }
        }
    }
    }
}

fn sqrt_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, f64::sqrt(f64::MAX) / f64::sqrt(2_f64))
}

sqrt_props!(complex_f64_sqrt_props, f64, sqrt_strategy_f64, any_scalar, 1e-10);


/// Generate property tests for complex number trigonometry.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! complex_cos_sin_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Complex,
        };
        use approx::{
            relative_eq,
        };
        use super::{
            $Generator,
        };


        proptest! {
            #[test]
            fn prop_cos_real_equals_cos_real(z in $Generator()) {
                let re_z = z.real();
                let z_re = Complex::from_real(re_z);

                prop_assert!(
                    relative_eq!(z_re.cos().real(), re_z.cos(), epsilon = $tolerance),
                    "z = {}; re(z) = {}; cos(re(z)) = {}; cos(z_re) = {}",
                    z, re_z, re_z.cos(), z_re.cos()
                );
            }

            #[test]
            fn prop_sin_real_equals_sin_real(z in $Generator()) {
                let re_z = z.real();
                let z_re = Complex::from_real(re_z);

                prop_assert!(
                    relative_eq!(z_re.sin().real(), re_z.sin(), epsilon = $tolerance),
                    "z = {}; re(z) = {}; sin(re(z)) = {}; sin(z_re) = {}",
                    z, re_z, re_z.cos(), z_re.cos()
                );
            }

            #[test]
            fn prop_cos_imaginary_equals_imaginary_cosh(z in $Generator()) {
                let zero: $ScalarType = num_traits::zero();
                let i = Complex::unit_im();
                let im_z = z.imaginary();

                prop_assert!(
                    relative_eq!((i * im_z).cos(), (im_z + i * zero).cosh(), epsilon = $tolerance),
                    "z = {}; im_z = {}; cos(i * im_z) = {}, i * cosh(im_z) = {}",
                    z, im_z, (i * im_z).cos(), (im_z + i * zero).cosh()
                );
            }

            #[test]
            fn prop_sin_imaginary_equals_imaginary_sinh(z in $Generator()) {
                let i = Complex::unit_im();
                let im_z = z.imaginary();

                prop_assert!(
                    relative_eq!((i * im_z).sin(), i * im_z.sinh(), epsilon = $tolerance),
                    "z = {}; im_z = {}; sin(i * im_z) = {}, i * sinh(im_z) = {}",
                    z, im_z, (i * im_z).sin(), i * im_z.sinh()
                );
            }
        }
    }
    }
}

fn cos_sin_strategy_f64() -> impl Strategy<Value = Complex<f64>>{
    imaginary_from_range(f64::EPSILON, f64::ln(f64::MAX))
}

complex_cos_sin_props!(complex_f64_sin_cos_props, f64, cos_sin_strategy_f64, any_scalar, 1e-10);


/// Generate property tests for complex number trigonometry.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! real_tan_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Complex,
        };
        use approx::{
            relative_eq,
        };
        use crate::{
            $Generator,
        };


        proptest! {
            #[test]
            fn prop_tan_real_equals_real_tan(z in $Generator()) {
                prop_assert!(
                    relative_eq!(z.tan().real(), z.real().tan(), epsilon = $tolerance),
                    "z = {}; z.tan() = {}; z.tan().real() = {}; z.real().tan() = {}",
                    z, z.tan(), z.tan().real(), z.real().tan()
                );
            }
        }
    }
    }
}

fn tan_strategy_real_f64() -> impl Strategy<Value = Complex<f64>> {
    real_from_range(f64::EPSILON, 200_f64)
}

real_tan_props!(complex_f64_tan_real_props, f64, tan_strategy_real_f64, any_scalar, 1e-7);


/// Generate property tests for complex number trigonometry.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! imaginary_tan_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Complex,
        };
        use approx::{
            relative_eq,
        };
        use crate::{
            $Generator,
        };


        proptest! {
            #[test]
            fn prop_tan_imaginary_equals_imaginary_tanh(z in $Generator()) {
                let i = Complex::unit_im();
                let im_z = z.imaginary();

                prop_assert!(
                    relative_eq!((i * im_z).tan(), i * im_z.tanh(), epsilon = $tolerance),
                    "z = {}; (i * im_z).tan() = {}; i * im_z.tanh() = {}",
                    z, (i * im_z).tan(), i * im_z.tanh()
                );
            }
        }
    }
    }
}

fn tan_strategy_imaginary_f64() -> impl Strategy<Value = Complex<f64>> {
    imaginary_from_range(f64::EPSILON, 200_f64)
}

imaginary_tan_props!(complex_f64_tan_imaginary_props, f64, tan_strategy_imaginary_f64, any_scalar, 1e-8);


/// Generate property tests for complex number hyperbolic trigonometry.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    complex numbers.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! tanh_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Complex,
        };
        use approx::{
            relative_eq,
        };
        use crate::{
            $Generator,
        };


        proptest! {
            #[test]
            fn prop_tanh_conjugate_z_equals_conjugate_tanh_z(z in $Generator()) {
                prop_assert_eq!(z.conjugate().tanh(), z.tanh().conjugate());
            }

            #[test]
            fn prop_tanh_negative_z_equals_negative_tanh_z(z in $Generator()) {
                prop_assert_eq!(
                    (-z).tanh(), -z.tanh(),
                    "z = {}; z.tanh() = {}; (-z).tanh() = {}; -z.tanh() = {}",
                    z, z.tanh(), (-z).tanh(), -z.tanh()
                );
            }
        }
    }
    }
}

fn tanh_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 200_f64)
}

tanh_props!(complex_f64_tanh_props, f64, tanh_strategy_f64, any_scalar, 1e-7);

