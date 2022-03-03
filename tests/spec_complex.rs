extern crate cglinalg;
extern crate num_traits;
extern crate proptest;


use proptest::prelude::*;
use cglinalg::{
    Complex, 
    Scalar,
};


fn any_scalar<S>() -> impl Strategy<Value = S>
where
    S: Scalar + Arbitrary
{
    any::<S>().prop_map(|scalar| {
        let modulus = num_traits::cast(100_000_000).unwrap();

        scalar % modulus
    })
}

fn any_complex<S>() -> impl Strategy<Value = Complex<S>> 
where 
    S: Scalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let complex = Complex::new(x, y);

        complex % modulus
    })
    .no_shrink()
}

/// Generate property tests for complex number indexing.
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
/// * `$UpperBound` denotes the upper bound on the range of acceptable indices.
macro_rules! index_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $UpperBound:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// When a complex number is treated like an array, it should accept all indices
            /// below the length of the array.
            ///
            /// Given a complex number `z`, it should return the element at position 
            /// `index` in the underlying storage of the complex number when the given 
            /// index is in bounds.
            #[test]
            fn prop_accepts_all_indices_in_of_bounds(
                z in $Generator::<$ScalarType>(), index in 0..$UpperBound as usize) {

                prop_assert_eq!(z[index], z[index]);
            }
    
            /// When a complex number is treated like an array, it should reject any 
            /// input index outside the length of the array.
            ///
            /// Given a complex number `z`, when the element index `index` is out of 
            /// bounds, it should generate a panic just like an array indexed 
            /// out of bounds.
            #[test]
            #[should_panic]
            fn prop_panics_when_index_out_of_bounds(
                z in $Generator::<$ScalarType>(), index in $UpperBound..usize::MAX) {
                
                prop_assert_eq!(z[index], z[index]);
            }
        }
    }
    }
}

index_props!(complex_index_props, f64, any_complex, 2);


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
        use cglinalg::{
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
        use cglinalg::{
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
                
                prop_assert_eq!(z1 + z2, &z1 + z2);
                prop_assert_eq!(z1 + z2, z1 + &z2);
                prop_assert_eq!(z1 + z2, &z1 + &z2);
                prop_assert_eq!(z1 + &z2, &z1 + z2);
                prop_assert_eq!(&z1 + z2, z1 + &z2);
                prop_assert_eq!(&z1 + z2, &z1 + &z2);
                prop_assert_eq!(z1 + &z2, &z1 + &z2);
            }

            /// Complex number addition over floating point scalars should be 
            /// approximately commutative.
            /// 
            /// Given complex numbers `z1` and `z2`, we have
            /// ```text
            /// z1 + z2 ~= z2 + z1
            /// ```
            /// Note that floating point complex number addition cannot be exactly 
            /// commutative because arithmetic with floating point numbers is 
            /// not commutative.
            #[test]
            fn prop_complex_addition_almost_commutative(
                z1 in $Generator::<$ScalarType>(), z2 in $Generator::<$ScalarType>()) {

                prop_assert!(relative_eq!(z1 + z2, z2 + z1, epsilon = $tolerance));
            }

            /// Complex number addition over floating point scalars should be 
            /// approximately associative. 
            ///
            /// Given complex numbers `z1`, `z2`, and `z3` we have
            /// ```text
            /// (z1 + z2) + z3 ~= z1 + (z2 + z3).
            /// ```
            /// Note that floating point complex number addition cannot be exactly 
            /// associative because arithmetic with floating point numbers is 
            /// not associative.
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
        use cglinalg::{
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
                
                let zero = Complex::<$ScalarType>::zero();

                prop_assert_eq!((z1 + z2) - (z2 + z1), zero);
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
        use cglinalg::{
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

