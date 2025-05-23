use approx_cmp::relative_eq;
use cglinalg_numeric::SimdScalarFloat;
use cglinalg_trigonometry::{
    Angle,
    Degrees,
    Radians,
};

use proptest::prelude::*;

fn strategy_radians_any<S>() -> impl Strategy<Value = Radians<S>>
where
    S: SimdScalarFloat + Arbitrary,
{
    any::<S>().prop_map(|dimensionless| {
        let two_pi = S::two_pi();
        let one_hundred_million: S = cglinalg_numeric::cast(100_000_000);

        Radians(dimensionless % (one_hundred_million * two_pi))
    })
}

fn strategy_degrees_any<S>() -> impl Strategy<Value = Degrees<S>>
where
    S: SimdScalarFloat + Arbitrary,
{
    any::<S>().prop_map(|dimensionless| {
        let two_pi: S = cglinalg_numeric::cast(360_f64);
        let one_hundred_million: S = cglinalg_numeric::cast(100_000_000);

        Degrees(dimensionless % (one_hundred_million * two_pi))
    })
}

/// Typed angles have an additive unit element.
///
/// Given a typed angle `angle`
/// ```text
/// angle + 0 == angle
/// ```
fn prop_angle_additive_zero<S, A>(angle: A) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let zero = A::zero();

    prop_assert_eq!(angle + zero, angle);

    Ok(())
}

/// Typed angles have additive inverses.
///
/// Given a typed angle `angle`, there is a typed angle `-angle` satisfying
/// ```text
/// angle - angle == angle + (-angle) = (-angle) + angle == 0
/// ```
fn prop_angle_additive_identity<S, A>(angle: A) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let zero = A::zero();

    prop_assert_eq!(angle - angle, zero);
    prop_assert_eq!(angle + (-angle), zero);
    prop_assert_eq!((-angle) + angle, zero);

    Ok(())
}

/// Typed angles are compatible with dimensionless multiplicative unit element.
///
/// Given a typed angle `angle`, and the dimensionless constant `1`
/// ```text
/// angle * 1 == angle
/// ```
fn prop_angle_multiplication_dimensionless_unit_element<S, A>(angle: A) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let one = S::one();

    prop_assert_eq!(angle * one, angle);

    Ok(())
}

/// The sine and arc sine functions should be inverses to each other.
///
/// Let `angle` be an angle and `recovered_angle = acos(cos(angle))` be an
/// angle recovered from a call to the sine and then the arc sine. Then they
/// should have matching sines.
///
/// Given a typed angle `angle`
/// ```text
/// recovered_angle := asin(sin(angle))
/// sin(recovered_angle) == sin(angle)
/// ```
fn prop_approx_sine_and_arcsine_inverses<S, A>(angle: A, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let sin_angle = angle.sin();
    let recovered_angle = A::asin(sin_angle);
    let sin_recovered_angle = recovered_angle.sin();

    prop_assert!(relative_eq!(
        sin_recovered_angle,
        sin_angle,
        abs_diff <= tolerance,
        relative <= tolerance
    ));

    Ok(())
}

/// The cosine and arc cosine functions should be inverses to each other.
///
/// Let `angle` be an angle and `recovered_angle = acos(cos(angle))` be an
/// angle recovered from a call to the cosine and then the arc cosine. Then they
/// should have matching cosines.
///
/// Given a typed angle `angle`
/// ```text
/// recovered_angle := acos(cos(angle))
/// cos(recovered_angle) == cos(angle)
/// ```
fn prop_approx_cosine_and_arccosine_inverses<S, A>(angle: A, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let cos_angle = angle.cos();
    let recovered_angle = A::acos(cos_angle);
    let cos_recovered_angle = recovered_angle.cos();

    prop_assert!(relative_eq!(
        cos_recovered_angle,
        cos_angle,
        abs_diff <= tolerance,
        relative <= tolerance
    ));

    Ok(())
}

/// The tangent and arc tangent functions should be inverses to each other.
///
/// Let `angle` be an angle and `recovered_angle = atan(tan(angle))` be an
/// angle recovered from a call to tangent and then arc tangent. The recovered
/// angle `recovered_angle` is congruent to `angle`, `angle + pi` or `angle - pi`
/// modulo `2 * pi`. There are the three angles in the interval `[0, 2pi)` that
/// have the same tangent.
///
/// Given a typed angle `angle`
/// ```text
/// recovered_angle := atan(tan(angle))
/// tan(recovered_angle) == tan(angle)
/// ```
fn prop_approx_tangent_and_arctangent_inverses<S, A>(angle: A, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let tan_angle = angle.tan();
    let recovered_angle = A::atan(tan_angle);
    let tan_recovered_angle = recovered_angle.tan();

    prop_assert!(
        relative_eq!(tan_recovered_angle, tan_angle, abs_diff <= tolerance, relative <= tolerance),
        "angle = {}\nrecovered_angle = {}\ntan_angle = {}\ntan_recovered_angle = {}",
        angle,
        recovered_angle,
        tan_angle,
        tan_recovered_angle
    );

    Ok(())
}

/// A typed angle and its congruent typed angles modulo `full_turn` should
/// give the same trigonometric outputs.
///
/// Given a typed angle `angle` and an integer `k`
/// ```text
/// sin(angle) == sin(angle + k * full_turn())
/// cos(angle) == cos(angle + k * full_turn())
/// tan(angle) == tan(angle + k * full_turn())
/// ```
fn prop_approx_congruent_angles<S, A>(angle: A, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let angle_plus_full_turn = angle + A::full_turn();

    prop_assert!(relative_eq!(
        angle.sin(),
        angle_plus_full_turn.sin(),
        abs_diff <= tolerance,
        relative <= tolerance
    ));
    prop_assert!(relative_eq!(
        angle.cos(),
        angle_plus_full_turn.cos(),
        abs_diff <= tolerance,
        relative <= tolerance
    ));

    Ok(())
}

/// Typed angle trigonometry satisfies the Pythagorean identity.
///
/// Given a typed angle `angle`
/// ```text
/// sin(angle)^2 + cos(angle)^2 == 1
/// ```
fn prop_approx_pythagorean_identity<S, A>(angle: A, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let one = S::one();
    let lhs = angle.cos() * angle.cos() + angle.sin() * angle.sin();
    let rhs = one;

    prop_assert!(relative_eq!(lhs, rhs, abs_diff <= tolerance, relative <= tolerance));

    Ok(())
}

/// A normalized angle correctly falls into the range `[0, full_turn)`.
///
/// Given an angle `angle`, the normalized angle satisfies
/// ```text
/// 0 =< normalize(angle) < full_turn
/// ```
fn prop_normalize_normalizes_to_interval<S, A>(angle: A) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let full_turn = A::full_turn();
    let zero = A::zero();
    let normalized_angle = angle.normalize();

    prop_assert!(
        (normalized_angle >= zero) && (normalized_angle <= full_turn),
        "angle = {:?}; normalized_angle = {:?}",
        angle,
        normalized_angle
    );

    Ok(())
}

/// A signed normalized angle correctly falls into the range `[-full_turn / 2, full_turn / 2)`.
///
/// Given an angle `angle`, the signed normalized angle satisfies
/// ```text
/// -full_turn / 2 =< normalize_signed(angle) < full_turn / 2
/// ```
fn prop_normalize_signed_normalizes_to_interval<S, A>(angle: A) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let full_turn_over_2 = A::full_turn_div_2();
    let normalized_angle = angle.normalize_signed();

    prop_assert!(
        (normalized_angle >= -full_turn_over_2) && (normalized_angle <= full_turn_over_2),
        "angle = {:?}; normalized_angle = {:?}",
        angle,
        normalized_angle
    );

    Ok(())
}

#[cfg(test)]
mod radians_f64_arithmetic_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_angle_additive_zero(angle in super::strategy_radians_any()) {
            let angle: super::Radians<f64> = angle;
            super::prop_angle_additive_zero(angle)?
        }

        #[test]
        fn prop_angle_additive_identity(angle in super::strategy_radians_any()) {
            let angle: super::Radians<f64> = angle;
            super::prop_angle_additive_identity(angle)?
        }

        #[test]
        fn prop_angle_multiplication_dimensionless_unit_element(angle in super::strategy_radians_any()) {
            let angle: super::Radians<f64> = angle;
            super::prop_angle_multiplication_dimensionless_unit_element(angle)?
        }
    }
}

#[cfg(test)]
mod radians_f64_trigonometry_prop {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_sine_and_arcsine_inverses(angle in super::strategy_radians_any()) {
            let angle: super::Radians<f64> = angle;
            super::prop_approx_sine_and_arcsine_inverses(angle, 1e-6)?
        }

        #[test]
        fn prop_approx_cosine_and_arccosine_inverses(angle in super::strategy_radians_any()) {
            let angle: super::Radians<f64> = angle;
            super::prop_approx_cosine_and_arccosine_inverses(angle, 1e-6)?
        }

        #[test]
        fn prop_approx_tangent_and_arctangent_inverses(angle in super::strategy_radians_any()) {
            let angle: super::Radians<f64> = angle;
            super::prop_approx_tangent_and_arctangent_inverses(angle, 1e-6)?
        }

        #[test]
        fn prop_approx_congruent_angles(angle in super::strategy_radians_any()) {
            let angle: super::Radians<f64> = angle;
            super::prop_approx_congruent_angles(angle, 1e-6)?
        }

        #[test]
        fn prop_approx_pythagorean_identity(angle in super::strategy_radians_any()) {
            let angle: super::Radians<f64> = angle;
            super::prop_approx_pythagorean_identity(angle, 1e-6)?
        }
    }
}

#[cfg(test)]
mod radians_f64_normalize_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_normalize_normalizes_to_interval(angle in super::strategy_radians_any()) {
            let angle: super::Radians<f64> = angle;
            super::prop_normalize_normalizes_to_interval(angle)?
        }

        #[test]
        fn prop_normalize_signed_normalizes_to_interval(angle in super::strategy_radians_any()) {
            let angle: super::Radians<f64> = angle;
            super::prop_normalize_signed_normalizes_to_interval(angle)?
        }
    }
}

#[cfg(test)]
mod degrees_f64_arithmetic_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_angle_additive_zero(angle in super::strategy_degrees_any()) {
            let angle: super::Degrees<f64> = angle;
            super::prop_angle_additive_zero(angle)?
        }

        #[test]
        fn prop_angle_additive_identity(angle in super::strategy_degrees_any()) {
            let angle: super::Degrees<f64> = angle;
            super::prop_angle_additive_identity(angle)?
        }

        #[test]
        fn prop_angle_multiplication_dimensionless_unit_element(angle in super::strategy_degrees_any()) {
            let angle: super::Degrees<f64> = angle;
            super::prop_angle_multiplication_dimensionless_unit_element(angle)?
        }
    }
}

#[cfg(test)]
mod degrees_f64_trigonometry_prop {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_sine_and_arcsine_inverses(angle in super::strategy_degrees_any()) {
            let angle: super::Degrees<f64> = angle;
            super::prop_approx_sine_and_arcsine_inverses(angle, 1e-6)?
        }

        #[test]
        fn prop_approx_cosine_and_arccosine_inverses(angle in super::strategy_degrees_any()) {
            let angle: super::Degrees<f64> = angle;
            super::prop_approx_cosine_and_arccosine_inverses(angle, 1e-6)?
        }

        #[test]
        fn prop_approx_tangent_and_arctangent_inverses(angle in super::strategy_degrees_any()) {
            let angle: super::Degrees<f64> = angle;
            super::prop_approx_tangent_and_arctangent_inverses(angle, 1e-6)?
        }

        #[test]
        fn prop_approx_congruent_angles(angle in super::strategy_degrees_any()) {
            let angle: super::Degrees<f64> = angle;
            super::prop_approx_congruent_angles(angle, 1e-6)?
        }

        #[test]
        fn prop_approx_pythagorean_identity(angle in super::strategy_degrees_any()) {
            let angle: super::Degrees<f64> = angle;
            super::prop_approx_pythagorean_identity(angle, 1e-6)?
        }
    }
}

#[cfg(test)]
mod degrees_f64_normalize_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_normalize_normalizes_to_interval(angle in super::strategy_degrees_any()) {
            let angle: super::Degrees<f64> = angle;
            super::prop_normalize_normalizes_to_interval(angle)?
        }

        #[test]
        fn prop_normalize_signed_normalizes_to_interval(angle in super::strategy_degrees_any()) {
            let angle: super::Degrees<f64> = angle;
            super::prop_normalize_signed_normalizes_to_interval(angle)?
        }
    }
}
