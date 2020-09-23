extern crate cglinalg;
extern crate num_traits;
extern crate proptest;

use proptest::prelude::*;
use cglinalg::{
    Degrees,
    Radians,
    Scalar,
};


fn any_radians<S>() -> impl Strategy<Value = Radians<S>> 
    where S: Scalar + Arbitrary
{
    any::<S>().prop_map(|unitless| Radians(unitless))
}

fn any_degrees<S>() -> impl Strategy<Value = Degrees<S>>
    where S: Scalar + Arbitrary
{
    any::<S>().prop_map(|unitless| Degrees(unitless))
}

/// Generate property tests for typed angle arithmetic over floating point 
/// scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property 
///    tests in to separate them from each other for each field type to prevent 
///    namespace collisions.
/// * `$AngleType` is the name of the angle type, e.g. Radians or Degrees.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    typed angles.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$tolerance` specifies the highest amount of acceptable error in the 
///    floating point computations that defines a correct computation.
///
/// We use approximate comparisons because arithmetic is not exact over finite 
/// precision floating point scalar types.
macro_rules! approx_arithmetic_props {
    ($TestModuleName:ident, $AngleType:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::approx::relative_eq;
        use cglinalg::{$AngleType, Zero};
    
        proptest! {
            /// Angle addition should be approximately commutative.
            ///
            /// Given typed angles `angle1` and `angle2`
            /// ```
            /// angle1 + angle2 ~= angle2 + angle1
            /// ```
            #[test]
            fn prop_angle_addition_commutative(
                angle1 in super::$Generator::<$ScalarType>(), angle2 in super::$Generator::<$ScalarType>()) {

                prop_assert!(relative_eq!(angle1 + angle2, angle2 + angle1, epsilon = $tolerance));
            }

            /// Angle addition is approximately associative.
            /// 
            /// Given typed angles `angle1`, `angle2, and `angle3`
            /// ```
            /// (angle1 + angle2) + angle3 ~= angle1 + (angle2 + angle3)
            /// ```
            #[test]
            fn prop_angle_addition_associative(
                angle1 in super::$Generator::<$ScalarType>(), 
                angle2 in super::$Generator::<$ScalarType>(), angle3 in super::$Generator::<$ScalarType>()) {
            
                prop_assert!(
                    relative_eq!((angle1 + angle2) + angle3, angle1 + (angle2 + angle3), epsilon = $tolerance)
                );
            }

            /// Multiplication of typed angles is compatible with unitless constants.
            ///
            /// Given a typed angle `angle`, and unitless constants `a`, and `b`
            /// ```
            /// (a * b) * angle ~= a * (b * angle3)
            /// ```
            #[test]
            fn prop_angle_multiplication_compatible(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), angle in super::$Generator::<$ScalarType>()) {
            
                prop_assert!(
                    relative_eq!(angle * (a * b), (angle * a) * b, epsilon = $tolerance)
                );
            }

            /// Typed angles have an additive unit element.
            ///
            /// Given a typed angle `angle`
            /// ```
            /// angle + 0 = angle
            /// ```
            #[test]
            fn prop_angle_additive_zero(angle in super::$Generator::<$ScalarType>()) {
                let zero = $AngleType::zero();
                prop_assert_eq!(angle + zero, angle);
            }

            /// Typed angles have additive inverses.
            ///
            /// Given a typed angle `angle`, there is a typed angle `-angle` satisfying
            /// ```
            /// angle - angle = angle + (-angle) = (-angle) + angle = 0
            /// ```
            #[test]
            fn prop_angle_additive_identity(angle in super::$Generator::<$ScalarType>()) {
                let zero = $AngleType::zero();
                prop_assert_eq!(angle - angle, zero);
                prop_assert_eq!(angle + (-angle), zero);
                prop_assert_eq!((-angle) + angle, zero);
            }

            /// Typed angles are compatible with unitless multiplicative unit element.
            ///
            /// Given a typed angle `angle`, and the unitless constant `1`
            /// ```
            /// angle * 1 = angle
            /// ```
            #[test]
            fn prop_angle_multiplication_unitless_unit_element(angle in super::$Generator::<$ScalarType>()) {
                let one: $ScalarType = num_traits::one();
                prop_assert_eq!(angle * one, angle);
            }
        }
    }
    }
}

approx_arithmetic_props!(radians_f64_arithmetic_props, Radians, f64, any_radians, 1e-7);
approx_arithmetic_props!(degrees_f64_arithmetic_props, Degrees, f64, any_degrees, 1e-7);

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
///    typed angles.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$tolerance` specifies the highest amount of acceptable error in the 
///    floating point computations that defines a correct computation.
///
/// We use approximate comparisons because arithmetic is not exact over finite 
/// precision floating point scalar types.
macro_rules! approx_trigonometry_props {
    ($TestModuleName:ident, $AngleType:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::approx::relative_eq;
        use cglinalg::{$AngleType, Angle};
    
        proptest! {
            /// The sine and arcsine functions should be inverses to each other.
            ///
            /// Given a typed angle `angle`
            /// ```
            /// asin(sin(angle)) = angle
            /// ```
            #[test]
            fn prop_sine_and_arcsine_inverses(angle in super::$Generator::<$ScalarType>()) {
                let recovered_angle = <$AngleType<$ScalarType> as Angle>::asin(angle.sin());
                prop_assert!(relative_eq!(recovered_angle, angle, epsilon = $tolerance));
            }

            /// The cosine and arccosine functions should be inverses to each other.
            ///
            /// Given a typed angle `angle`
            /// ```
            /// acos(cos(angle)) = angle
            /// ```
            #[test]
            fn prop_cosine_and_arccosine_inverses(angle in super::$Generator::<$ScalarType>()) {
                let recovered_angle = <$AngleType<$ScalarType> as Angle>::acos(angle.cos());
                prop_assert!(relative_eq!(recovered_angle, angle, epsilon = $tolerance));
            }

            /// The tangent and arctangent functions should be inverses to each other.
            ///
            /// Given a typed angle `angle`
            /// ```
            /// atan(tan(angle)) = angle
            /// ```
            #[test]
            fn prop_tangent_and_arctangent_inverses(angle in super::$Generator::<$ScalarType>()) {
                let recovered_angle = <$AngleType<$ScalarType> as Angle>::atan(angle.tan());
                prop_assert!(relative_eq!(recovered_angle, angle, epsilon = $tolerance));
            }

            /// A typed angle and its congruent typed angles modulo `full_turn` should 
            /// give the same trigonometric outputs.
            ///
            /// Given a typed angle `angle` and an integer `k`
            /// ```
            /// sin(angle) = sin(angle + k * full_turn())
            /// cos(angle) = cos(angle + k * full_turn())
            /// tan(angle) = tan(angle + k * full_turn())
            /// ```
            #[test]
            fn prop_congruent_angles(angle in super::$Generator::<$ScalarType>()) {
                let angle_plus_full_turn = angle + <$AngleType<$ScalarType> as Angle>::full_turn();
                prop_assert!(relative_eq!(angle.sin(), angle_plus_full_turn.sin(), epsilon = $tolerance));
                prop_assert!(relative_eq!(angle.cos(), angle_plus_full_turn.cos(), epsilon = $tolerance));
                prop_assert!(relative_eq!(angle.tan(), angle_plus_full_turn.tan(), epsilon = $tolerance));
            }

            /// Typed angle trigonometry satisfies the pythagorean identity.
            ///
            /// Given a typed angle `angle`
            /// ```
            /// sin(angle)^2 + cos(angle)^2 = 1
            /// ```
            #[test]
            fn prop_pythagorean_identity(angle in super::$Generator::<$ScalarType>()) {
                let one: $ScalarType = num_traits::one();
                prop_assert!(relative_eq!(
                    angle.cos() * angle.cos() + angle.sin() * angle.sin(), one, epsilon = $tolerance
                ));
            }
        }
    }
    }
}

approx_trigonometry_props!(radians_f64_trigonometry_props, Radians, f64, any_radians, 1e-7);
approx_trigonometry_props!(degrees_f64_trigonometry_props, Degrees, f64, any_degrees, 1e-7);

