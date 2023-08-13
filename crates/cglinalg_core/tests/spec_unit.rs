extern crate cglinalg_core;
extern crate num_traits;
extern crate proptest;


use cglinalg_core::{
    Complex,
    Quaternion,
    Vector1, 
    Vector2, 
    Vector3, 
    Vector4,
    SimdScalar,
};

use proptest::prelude::*;


fn any_vector1<S>() -> impl Strategy<Value = Vector1<S>> 
where 
    S: SimdScalar + Arbitrary 
{
    any::<S>().prop_map(|x| {
        let offset = num_traits::cast::<f64, S>(1e-6_f64).unwrap();
        let vector_offset = Vector1::from_fill(offset);
        let modulus = num_traits::cast(100_000_000).unwrap();
        let vector = Vector1::new(x);

        vector_offset + vector % modulus
    })
}

fn any_vector2<S>() -> impl Strategy<Value = Vector2<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let offset = num_traits::cast::<f64, S>(1e-6_f64).unwrap();
        let vector_offset = Vector2::from_fill(offset);
        let modulus = num_traits::cast(100_000_000).unwrap();
        let vector = Vector2::new(x, y);

        vector_offset + vector % modulus
    })
}

fn any_vector3<S>() -> impl Strategy<Value = Vector3<S>>
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S, S)>().prop_map(|(x, y, z)| {
        let offset = num_traits::cast::<f64, S>(1e-6_f64).unwrap();
        let vector_offset = Vector3::from_fill(offset);
        let modulus = num_traits::cast(100_000_000).unwrap();
        let vector = Vector3::new(x, y, z);

        vector_offset + vector % modulus
    })
}

fn any_vector4<S>() -> impl Strategy<Value = Vector4<S>>
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S, S, S)>().prop_map(|(x, y, z, w)| {
        let offset = num_traits::cast::<f64, S>(1e-6_f64).unwrap();
        let vector_offset = Vector4::from_fill(offset);
        let modulus = num_traits::cast(100_000_000).unwrap();
        let vector = Vector4::new(x, y, z, w);

        vector_offset + vector % modulus
    })
}

fn any_quaternion<S>() -> impl Strategy<Value = Quaternion<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S, S, S)>().prop_map(|(x, y, z, w)| {
        let offset = num_traits::cast::<f64, S>(1e-6_f64).unwrap();
        let quaternion_offset = Quaternion::from_fill(offset);
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let quaternion = Quaternion::new(x, y, z, w);

        quaternion_offset + quaternion % modulus
    })
    .no_shrink()
}

fn any_complex<S>() -> impl Strategy<Value = Complex<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let offset = num_traits::cast::<f64, S>(1e-6_f64).unwrap();
        let complex_offset = Complex::new(offset, offset);
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let complex = Complex::new(x, y);

        complex_offset + complex % modulus
    })
    .no_shrink()
}


/// Generate property tests for normed data types.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property tests 
///    in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$VectorN` denotes the name of the normed type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the normed type.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! unit_props {
    ($TestModuleName:ident, $UnitType:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            Normed,
            Unit,
        };
        use super::{
            $Generator,
        };


        proptest! {
            #[test]
            fn prop_from_value_normalized(value in $Generator::<$ScalarType>()) {
                let expected = value.normalize();
                let unit_value = Unit::from_value(value);
                let result = unit_value.into_inner();

                prop_assert_eq!(result, expected);
            }

            #[test]
            fn prop_from_value_with_norm_normalized(value in $Generator::<$ScalarType>()) {
                let expected = value.normalize();
                let (unit_value, _) = Unit::from_value_with_norm(value);
                let result = unit_value.into_inner();

                prop_assert_eq!(result, expected);
            }

            #[test]
            fn prop_from_value_with_norm_correct_norm(value in $Generator::<$ScalarType>()) {
                let expected = value.norm();
                let (_, result) = Unit::from_value_with_norm(value);

                prop_assert_eq!(result, expected);
            }

            #[test]
            fn prop_from_value_unchecked_into_inner(value in $Generator::<$ScalarType>()) {
                let expected = value;
                let unit_value = Unit::from_value_unchecked(value);
                let result = unit_value.into_inner();

                prop_assert_eq!(result, expected);
            }

            #[test]
            fn prop_try_from_value_with_norm_above_threshold_is_some(value in $Generator::<$ScalarType>()) {
                let threshold = num_traits::cast::<f64, $ScalarType>(1e-8).unwrap();
                let result = Unit::try_from_value_with_norm(value, threshold);

                prop_assert!(result.is_some());
            }

            #[test]
            fn prop_try_from_value_above_threshold_is_some(value in $Generator::<$ScalarType>()) {
                let threshold = num_traits::cast::<f64, $ScalarType>(1e-8).unwrap();
                let result = Unit::try_from_value(value, threshold);

                prop_assert!(result.is_some());
            }
        }
    }
    }
}

unit_props!(unit_vector1_f64_props, Vector1, f64, any_vector1);
unit_props!(unit_vector2_f64_props, Vector2, f64, any_vector2);
unit_props!(unit_vector3_f64_props, Vector3, f64, any_vector3);
unit_props!(unit_vector4_f64_props, Vector4, f64, any_vector4);

unit_props!(unit_quaternion_f64_props, Quaternion, f64, any_quaternion);

unit_props!(unit_complex_f64_props, Complex, f64, any_complex);

