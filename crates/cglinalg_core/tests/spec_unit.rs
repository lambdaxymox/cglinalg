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
    SimdScalarSigned,
    SimdScalarFloat,
    Unit,
    Normed,
};

use proptest::prelude::*;
use core::fmt;


fn any_vector1<S>() -> impl Strategy<Value = Vector1<S>> 
where 
    S: SimdScalarSigned + Arbitrary 
{
    any::<S>().prop_map(|x| {
        let offset = num_traits::cast(1e-6_f64).unwrap();
        let signum_x = x.signum();
        let vector_offset = Vector1::new(
            signum_x * offset
        );
        let modulus = num_traits::cast(100_000_000).unwrap();
        let vector = Vector1::new(x) % modulus;

        vector_offset + vector
    })
}

fn any_vector2<S>() -> impl Strategy<Value = Vector2<S>> 
where 
    S: SimdScalarSigned + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let offset = num_traits::cast(1e-6_f64).unwrap();
        let signum_x = x.signum();
        let signum_y = y.signum();
        let vector_offset = Vector2::new(
            signum_x * offset,
            signum_y * offset
        );
        let modulus = num_traits::cast(100_000_000).unwrap();
        let vector = Vector2::new(x, y) % modulus;

        vector_offset + vector
    })
}

fn any_vector3<S>() -> impl Strategy<Value = Vector3<S>>
where 
    S: SimdScalarSigned + Arbitrary
{
    any::<(S, S, S)>().prop_map(|(x, y, z)| {
        let offset = num_traits::cast(1e-6_f64).unwrap();
        let signum_x = x.signum();
        let signum_y = y.signum();
        let signum_z = z.signum();
        let vector_offset = Vector3::new(
            signum_x * offset,
            signum_y * offset,
            signum_z * offset
        );
        let modulus = num_traits::cast(100_000_000).unwrap();
        let vector = Vector3::new(x, y, z) % modulus;

        vector_offset + vector
    })
}

fn any_vector4<S>() -> impl Strategy<Value = Vector4<S>>
where 
    S: SimdScalarSigned + Arbitrary
{
    any::<(S, S, S, S)>().prop_map(|(x, y, z, w)| {
        let offset = num_traits::cast(1e-6_f64).unwrap();
        let signum_x = x.signum();
        let signum_y = y.signum();
        let signum_z = z.signum();
        let signum_w = w.signum();
        let vector_offset = Vector4::new(
            signum_x * offset,
            signum_y * offset,
            signum_z * offset,
            signum_w * offset
        );
        let modulus = num_traits::cast(100_000_000).unwrap();
        let vector = Vector4::new(x, y, z, w) % modulus;

        vector_offset + vector
    })
}

fn any_quaternion<S>() -> impl Strategy<Value = Quaternion<S>> 
where 
    S: SimdScalarSigned + Arbitrary
{
    any::<(S, S, S, S)>().prop_map(|(s, x, y, z)| {
        let offset = num_traits::cast(1e-6_f64).unwrap();
        let signum_s = s.signum();
        let signum_x = x.signum();
        let signum_y = y.signum();
        let signum_z = z.signum();
        let quaternion_offset = Quaternion::new(
            signum_s * offset,
            signum_x * offset,
            signum_y * offset,
            signum_z * offset
        );
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let quaternion = Quaternion::new(s, x, y, z) % modulus;

        quaternion_offset + quaternion
    })
    .no_shrink()
}

fn any_complex<S>() -> impl Strategy<Value = Complex<S>> 
where 
    S: SimdScalarSigned + Arbitrary
{
    any::<(S, S)>().prop_map(|(re, im)| {
        let offset = num_traits::cast(1e-6_f64).unwrap();
        let signum_re = re.signum();
        let signum_im = im.signum();
        let complex_offset = Complex::new(
            signum_re * offset, 
            signum_im * offset
        );
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let complex = Complex::new(re, im) % modulus;

        complex_offset + complex
    })
    .no_shrink()
}


fn prop_from_value_normalized<S, T>(value: T) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat,
    T: Normed<Output = S> + PartialEq + fmt::Debug
{
    let expected = value.normalize();
    let unit_value = Unit::from_value(value);
    let result = unit_value.into_inner();

    prop_assert_eq!(result, expected);

    Ok(())
}


fn prop_from_value_with_norm_normalized<S, T>(value: T) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat,
    T: Normed<Output = S> + PartialEq + fmt::Debug
{
    let expected = value.normalize();
    let (unit_value, _) = Unit::from_value_with_norm(value);
    let result = unit_value.into_inner();

    prop_assert_eq!(result, expected);

    Ok(())
}


fn prop_from_value_with_norm_correct_norm<S, T>(value: T) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat,
    T: Normed<Output = S> + PartialEq + fmt::Debug
{
    let expected = value.norm();
    let (_, result) = Unit::from_value_with_norm(value);

    prop_assert_eq!(result, expected);

    Ok(())
}


fn prop_from_value_unchecked_into_inner<S, T>(value: T) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat,
    T: Normed<Output = S> + PartialEq + fmt::Debug + Clone
{
    let expected = value.clone();
    let unit_value = Unit::from_value_unchecked(value);
    let result = unit_value.into_inner();

    prop_assert_eq!(result, expected);

    Ok(())
}


fn prop_try_from_value_with_norm_above_threshold_is_some<S, T>(value: T) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat,
    T: Normed<Output = S> + PartialEq + fmt::Debug
{
    let threshold = num_traits::cast(1e-8).unwrap();
    let result = Unit::try_from_value_with_norm(value, threshold);

    prop_assert!(result.is_some());

    Ok(())
}


fn prop_try_from_value_above_threshold_is_some<S, T>(value: T) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat,
    T: Normed<Output = S> + PartialEq + fmt::Debug
{
    let threshold = num_traits::cast(1e-8).unwrap();
    let result = Unit::try_from_value(value, threshold);

    prop_assert!(result.is_some());

    Ok(())
}


macro_rules! unit_props {
    ($TestModuleName:ident, $UnitType:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_from_value_normalized(value in super::$Generator()) {
                let value: super::$UnitType<$ScalarType> = value;
                super::prop_from_value_normalized(value)?
            }

            #[test]
            fn prop_from_value_with_norm_normalized(value in super::$Generator()) {
                let value: super::$UnitType<$ScalarType> = value;
                super::prop_from_value_with_norm_normalized(value)?
            }

            #[test]
            fn prop_from_value_with_norm_correct_norm(value in super::$Generator()) {
                let value: super::$UnitType<$ScalarType> = value;
                super::prop_from_value_with_norm_correct_norm(value)?
            }

            #[test]
            fn prop_from_value_unchecked_into_inner(value in super::$Generator()) {
                let value: super::$UnitType<$ScalarType> = value;
                super::prop_from_value_unchecked_into_inner(value)?
            }

            #[test]
            fn prop_try_from_value_with_norm_above_threshold_is_some(value in super::$Generator()) {
                let value: super::$UnitType<$ScalarType> = value;
                super::prop_try_from_value_with_norm_above_threshold_is_some(value)?
            }

            #[test]
            fn prop_try_from_value_above_threshold_is_some(value in super::$Generator()) {
                let value: super::$UnitType<$ScalarType> = value;
                super::prop_try_from_value_above_threshold_is_some(value)?
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

