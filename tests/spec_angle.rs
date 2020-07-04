extern crate gdmath;
extern crate num_traits;
extern crate proptest;

use proptest::prelude::*;
use gdmath::{
    Degrees,
    Radians,
    Angle, 
    Scalar,
    ScalarFloat,
};


fn any_radians<S>() -> impl Strategy<Value = Radians<S>> where S: Scalar + Arbitrary {
    any::<S>().prop_map(|unitless| Radians(unitless))
}

fn any_degrees<S>() -> impl Strategy<Value = Degrees<S>> where S: Scalar + Arbitrary {
    any::<S>().prop_map(|unitless| Degrees(unitless))
}

/// Generate the properties for typed angle arithmetic over floating point scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$AngleType` is the name of the typed angle type, e.g. Radians or Degrees.
/// `$ScalarType` denotes the underlying system of numbers that compose the typed angles.
/// `$Generator` is the name of a function or closure for generating examples.
/// `$tolerance` specifies the highest amount of acceptable error in the floating point computations
///  that still defines a correct computation. We cannot guarantee floating point computations
///  will be exact since the underlying floating point arithmetic is not exact.
///
/// We use approximate comparisons because arithmetic is not exact over finite precision floating point
/// scalar types.
macro_rules! approx_arithmetic_props {
    ($TestModuleName:ident, $AngleType:ty, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::approx::relative_eq;
    
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

            /// Angle multiplication is approximately commutative.
            ///
            /// Given typed angles `angle1` and `angle2`
            /// ```
            /// angle1 * angle2 ~= angle2 * angle1
            /// ```
            #[test]
            fn prop_angle_multiplication_commutative(
                angle1 in super::$Generator::<$ScalarType>(), angle2 in super::$Generator::<$ScalarType>()) {

                prop_assert!(relative_eq!(angle1 * angle2, angle2 * angle1, epsilon = $tolerance));
            }
        }
    }
    }
}

approx_arithmetic_props!(radians_f64_arithmetic_props, Radians, f64, any_radians, 1e-7);
approx_arithmetic_props!(degrees_f64_arithmetic_props, degrees, f64, any_degrees, 1e-7);
