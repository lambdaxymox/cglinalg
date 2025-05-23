use cglinalg_core::{
    Complex,
    Normed,
    Quaternion,
    Unit,
    Vector,
    Vector1,
    Vector2,
    Vector3,
    Vector4,
};
use cglinalg_numeric::{
    SimdScalarFloat,
    SimdScalarSigned,
};

use core::fmt;
use proptest::prelude::*;

fn strategy_vector_any<S, const N: usize>(min_value: S, max_value: S) -> impl Strategy<Value = Vector<S, N>>
where
    S: SimdScalarSigned + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarSigned,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<[S; N]>().prop_map(move |array| {
        let abs_offset = cglinalg_numeric::cast(1e-6_f64);
        let mut vector = Vector::zero();
        for i in 0..N {
            let signum = array[i].signum();
            let offset = signum * abs_offset;
            let value = signum * rescale(array[i].abs(), min_value, max_value);
            vector[i] = offset + value;
        }

        vector
    })
}

fn strategy_vector_f64_any<const N: usize>() -> impl Strategy<Value = Vector<f64, N>> {
    let min_value = f64::sqrt(f64::EPSILON);
    let max_value = f64::sqrt(f64::MAX) / f64::sqrt(2_f64);

    strategy_vector_any(min_value, max_value)
}

fn strategy_quaternion_f64_any() -> impl Strategy<Value = Quaternion<f64>> {
    strategy_vector_f64_any().prop_map(|vector| Quaternion::from(vector))
}

fn strategy_complex_f64_any() -> impl Strategy<Value = Complex<f64>> {
    strategy_vector_f64_any::<2>().prop_map(|vector| Complex::new(vector[0], vector[1]))
}

/// The unit data type normalizes the input value `value`.
///
/// Given a value `value` of type [`T`] which has a notion of vector length
/// ```text
/// norm(into_inner(from_value(value))) == 1
/// ```
fn prop_from_value_normalized<S, T>(value: T) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    T: Normed<Output = S> + PartialEq + fmt::Debug,
{
    let expected = value.normalize();
    let unit_value = Unit::from_value(value);
    let result = unit_value.into_inner();

    prop_assert_eq!(result, expected);

    Ok(())
}

/// The unit data type normalizes the input value `value`.
///
/// Given a value `value` of type [`T`] which has a notion of vector length
/// ```text
/// norm(into_inner(fst(from_value_with_norm(value)))) == 1
/// ```
fn prop_from_value_with_norm_normalized<S, T>(value: T) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    T: Normed<Output = S> + PartialEq + fmt::Debug,
{
    let expected = value.normalize();
    let (unit_value, _) = Unit::from_value_with_norm(value);
    let result = unit_value.into_inner();

    prop_assert_eq!(result, expected);

    Ok(())
}

/// The unit data type normalizes an input value to a specified norm.
///
/// Given a value `value` of type [`T`] which has norm `norm`
/// ```text
/// norm(from_value_with_norm(value)) == norm
/// ```
fn prop_from_value_with_norm_correct_norm<S, T>(value: T) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    T: Normed<Output = S> + PartialEq + fmt::Debug,
{
    let expected = value.norm();
    let (_, result) = Unit::from_value_with_norm(value);

    prop_assert_eq!(result, expected);

    Ok(())
}

/// The unit data type when constructed unchecked does not change the
/// inner value.
///
/// Given a value `value` of type [`T`]
/// ```text
/// into_inner(from_value_unchecked(value)) == value
/// ```
fn prop_from_value_unchecked_into_inner<S, T>(value: T) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    T: Normed<Output = S> + PartialEq + fmt::Debug + Clone,
{
    let expected = value.clone();
    let unit_value = Unit::from_value_unchecked(value);
    let result = unit_value.into_inner();

    prop_assert_eq!(result, expected);

    Ok(())
}

/// When the norm of a vector value is above a specified input threshold,
/// [`Unit::try_from_value_with_norm`] returns a normalized value. Otherwise, it
/// returns `None`.
fn prop_try_from_value_with_norm_above_threshold_is_some<S, T>(value: T) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    T: Normed<Output = S> + PartialEq + fmt::Debug,
{
    let threshold = cglinalg_numeric::cast(1e-8);
    let result = Unit::try_from_value_with_norm(value, threshold);

    prop_assert!(result.is_some());

    Ok(())
}

/// When the norm of a vector value is above a specified input threshold,
/// [`Unit::try_from_value`] returns a normalized value. Otherwise, it
/// returns `None`.
fn prop_try_from_value_above_threshold_is_some<S, T>(value: T) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    T: Normed<Output = S> + PartialEq + fmt::Debug,
{
    let threshold = cglinalg_numeric::cast(1e-8);
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
    };
}

unit_props!(unit_vector1_f64_props, Vector1, f64, strategy_vector_f64_any);
unit_props!(unit_vector2_f64_props, Vector2, f64, strategy_vector_f64_any);
unit_props!(unit_vector3_f64_props, Vector3, f64, strategy_vector_f64_any);
unit_props!(unit_vector4_f64_props, Vector4, f64, strategy_vector_f64_any);

unit_props!(unit_quaternion_f64_props, Quaternion, f64, strategy_quaternion_f64_any);

unit_props!(unit_complex_f64_props, Complex, f64, strategy_complex_f64_any);
