extern crate cglinalg_core;
extern crate num_traits;
extern crate proptest;


use proptest::prelude::*;
use cglinalg_core::{
    Degrees,
    Radians,
    SimdScalar,
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

fn any_radians<S>() -> impl Strategy<Value = Radians<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<S>()
        .prop_map(|dimensionless| {
            let two_pi: S = num_traits::cast(2_f64 * core::f64::consts::PI).unwrap();
            let one_million: S = num_traits::cast(100_000_000).unwrap();
            Radians(dimensionless % (one_million * two_pi))
        })
        .no_shrink()
}

fn any_degrees<S>() -> impl Strategy<Value = Degrees<S>>
where 
    S: SimdScalar + Arbitrary
{
    any::<S>()
        .prop_map(|dimensionless| {
            let two_pi: S = num_traits::cast(360_f64).unwrap();
            let one_million: S = num_traits::cast(100_000_000).unwrap();
            Degrees(dimensionless % (one_million * two_pi)) 
        })
        .no_shrink()
}


/// Generate property tests for typed angle arithmetic over floating point 
/// scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$AngleType` is the name of the angle type, e.g. Radians or Degrees.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of typed angles.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_arithmetic_props {
    ($TestModuleName:ident, $AngleType:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use approx::relative_eq;
        use cglinalg_core::{
            $AngleType,
        };
        use super::{
            $Generator,
            $ScalarGen,
        };

    
        proptest! {
            /// Multiplication of typed angles is compatible with dimensionless 
            /// constants.
            ///
            /// Given a typed angle `angle`, and dimensionless constants `a`, and `b`
            /// ```text
            /// (a * b) * angle ~= a * (b * angle)
            /// ```
            #[test]
            fn prop_angle_multiplication_compatible(
                a in $ScalarGen::<$ScalarType>(), 
                b in $ScalarGen::<$ScalarType>(), angle in $Generator::<$ScalarType>()) {
            
                prop_assume!((angle * (a * b)).is_finite());
                prop_assume!(((angle * a) * b).is_finite());
                prop_assert!(
                    relative_eq!(angle * (a * b), (angle * a) * b, epsilon = $tolerance)
                );
            }

            /// Typed angles have an additive unit element.
            ///
            /// Given a typed angle `angle`
            /// ```text
            /// angle + 0 = angle
            /// ```
            #[test]
            fn prop_angle_additive_zero(angle in $Generator::<$ScalarType>()) {
                let zero = $AngleType::zero();

                prop_assert_eq!(angle + zero, angle);
            }

            /// Typed angles have additive inverses.
            ///
            /// Given a typed angle `angle`, there is a typed angle `-angle` satisfying
            /// ```text
            /// angle - angle = angle + (-angle) = (-angle) + angle = 0
            /// ```
            #[test]
            fn prop_angle_additive_identity(angle in $Generator::<$ScalarType>()) {
                let zero = $AngleType::zero();

                prop_assert_eq!(angle - angle, zero);
                prop_assert_eq!(angle + (-angle), zero);
                prop_assert_eq!((-angle) + angle, zero);
            }

            /// Typed angles are compatible with dimensionless multiplicative 
            /// unit element.
            ///
            /// Given a typed angle `angle`, and the dimensionless constant `1`
            /// ```text
            /// angle * 1 = angle
            /// ```
            #[test]
            fn prop_angle_multiplication_dimensionless_unit_element(angle in $Generator::<$ScalarType>()) {
                let one: $ScalarType = num_traits::one();

                prop_assert_eq!(angle * one, angle);
            }
        }
    }
    }
}

approx_arithmetic_props!(radians_f64_arithmetic_props, Radians, f64, any_radians, any_scalar, 1e-10);
approx_arithmetic_props!(degrees_f64_arithmetic_props, Degrees, f64, any_degrees, any_scalar, 1e-10);


/// Generate property tests for typed angle trigonometry over floating point 
/// scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$AngleType` is the name of the angle type, e.g. Radians or Degrees.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of typed angles.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_trigonometry_props {
    ($TestModuleName:ident, $AngleType:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use approx::relative_eq;
        use cglinalg_core::{
            $AngleType,
            Angle,
        };
        use super::$Generator;

    
        proptest! {
            /// The sine and arc sine functions should be inverses to each other.
            ///
            /// Let `angle` be an angle and `recovered_angle = acos(cos(angle))` be an 
            /// angle recovered from a call to the sine and then the arc sine. Then they
            /// should have matching sines. 
            ///
            /// Given a typed angle `angle`
            /// ```text
            /// recovered_angle := asin(sin(angle))
            /// sin(recovered_angle) = sin(angle)
            /// ```
            #[test]
            fn prop_sine_and_arcsine_inverses(angle in $Generator::<$ScalarType>()) {
                let sin_angle = angle.sin();
                let recovered_angle = <$AngleType<$ScalarType> as Angle>::asin(sin_angle);
                let sin_recovered_angle = recovered_angle.sin();

                prop_assert!(relative_eq!(sin_recovered_angle, sin_angle, epsilon = $tolerance));
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
            /// cos(recoved_angle) = cos(angle)
            /// ```
            #[test]
            fn prop_cosine_and_arccosine_inverses(angle in $Generator::<$ScalarType>()) {
                let cos_angle = angle.cos();
                let recovered_angle = <$AngleType<$ScalarType> as Angle>::acos(cos_angle);
                let cos_recovered_angle = recovered_angle.cos();

                prop_assert!(relative_eq!(cos_recovered_angle, cos_angle, epsilon = $tolerance));
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
            #[test]
            fn prop_tangent_and_arctangent_inverses(angle in $Generator::<$ScalarType>()) {
                let tan_angle = angle.tan();
                let recovered_angle = <$AngleType<$ScalarType> as Angle>::atan(tan_angle);
                let tan_recovered_angle = recovered_angle.tan();

                prop_assert!(
                    relative_eq!(tan_recovered_angle, tan_angle, epsilon = $tolerance),
                    "angle = {}\nrecovered_angle = {}\ntan_angle = {}\ntan_recovered_angle = {}",
                    angle, recovered_angle, tan_angle, tan_recovered_angle
                );
            }

            /// A typed angle and its congruent typed angles modulo `full_turn` should 
            /// give the same trigonometric outputs.
            ///
            /// Given a typed angle `angle` and an integer `k`
            /// ```text
            /// sin(angle) = sin(angle + k * full_turn())
            /// cos(angle) = cos(angle + k * full_turn())
            /// tan(angle) = tan(angle + k * full_turn())
            /// ```
            #[test]
            fn prop_congruent_angles(angle in $Generator::<$ScalarType>()) {
                let angle_plus_full_turn = angle + <$AngleType<$ScalarType> as Angle>::full_turn();

                prop_assert!(relative_eq!(angle.sin(), angle_plus_full_turn.sin(), epsilon = $tolerance));
                prop_assert!(relative_eq!(angle.cos(), angle_plus_full_turn.cos(), epsilon = $tolerance));
            }

            /// Typed angle trigonometry satisfies the Pythagorean identity.
            ///
            /// Given a typed angle `angle`
            /// ```text
            /// sin(angle)^2 + cos(angle)^2 = 1
            /// ```
            #[test]
            fn prop_pythagorean_identity(angle in $Generator::<$ScalarType>()) {
                let one: $ScalarType = num_traits::one();

                prop_assert!(relative_eq!(
                    angle.cos() * angle.cos() + angle.sin() * angle.sin(), one, epsilon = $tolerance
                ));
            }
        }
    }
    }
}

approx_trigonometry_props!(radians_f64_trigonometry_props, Radians, f64, any_radians, 1e-6);
approx_trigonometry_props!(degrees_f64_trigonometry_props, Degrees, f64, any_degrees, 1e-6);

