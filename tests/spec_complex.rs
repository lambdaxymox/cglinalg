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

