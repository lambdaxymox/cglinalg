extern crate cglinalg;
extern crate num_traits;
extern crate proptest;


use cglinalg::{
    Point1,
    Point2,
    Point3,
    Vector1, 
    Vector2, 
    Vector3, 
    Scalar,
};

use proptest::prelude::*;


fn any_scalar<S>() -> impl Strategy<Value = S>
    where S: Scalar + Arbitrary
{
    any::<S>().prop_map(|scalar| {
        let modulus = num_traits::cast(1_000_000).unwrap();

        scalar % modulus
    })
}

fn any_vector1<S>() -> impl Strategy<Value = Vector1<S>> 
    where S: Scalar + Arbitrary 
{
    any::<S>().prop_map(|x| {
        let modulus: S = num_traits::cast(1_000_000).unwrap();
        let vector = Vector1::new(x);

        vector % modulus
    })
    .no_shrink()
}

fn any_vector2<S>() -> impl Strategy<Value = Vector2<S>> 
    where S: Scalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let modulus: S = num_traits::cast(1_000_000).unwrap();
        let vector = Vector2::new(x, y);
    
        vector % modulus
    })
    .no_shrink()
}

fn any_vector3<S>() -> impl Strategy<Value = Vector3<S>>
    where S: Scalar + Arbitrary
{
    any::<(S, S, S)>().prop_map(|(x, y, z)| { 
        let modulus: S = num_traits::cast(1_000_000).unwrap();
        let vector = Vector3::new(x, y, z);

        vector % modulus
    })
    .no_shrink()
}

fn any_point1<S>() -> impl Strategy<Value = Point1<S>> 
    where S: Scalar + Arbitrary 
{
    any::<S>().prop_map(|x| {
        let modulus: S = num_traits::cast(1_000_000).unwrap();
        let point = Point1::new(x);

        point % modulus
    })
    .no_shrink()
}

fn any_point2<S>() -> impl Strategy<Value = Point2<S>> 
    where S: Scalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let modulus: S = num_traits::cast(1_000_000).unwrap();
        let point = Point2::new(x, y);

        point % modulus
    })
    .no_shrink()
}

fn any_point3<S>() -> impl Strategy<Value = Point3<S>>
    where S: Scalar + Arbitrary
{
    any::<(S, S, S)>().prop_map(|(x, y, z)| {
        let modulus = num_traits::cast(1_000_000).unwrap();
        let point = Point3::new(x, y, z);

        point % modulus
    })
    .no_shrink()
}


/// Generate property tests for point multiplication over floating point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property tests 
///    in to separate them from each other for each field type to prevent 
///    namespace collisions.
/// * `$PointN` denotes the name of the point type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of points.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_mul_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg::approx::{
            relative_eq,
        };
        use super::{
            $Generator,
            $ScalarGen,
        };


        proptest! {
            /// Multiplication of a scalar and a point should be approximately 
            /// commutative.
            ///
            /// Given a constant `c` and a point `p`
            /// ```text
            /// c * p ~= p * c
            /// ```
            /// Note that floating point scalar/point multiplication cannot be commutative 
            /// because multiplication in the underlying floating point scalars is 
            /// not commutative.
            #[test]
            fn prop_scalar_times_point_equals_point_times_scalar(
                c in $ScalarGen::<$ScalarType>(), p in $Generator::<$ScalarType>()) {
                
                prop_assert!(
                    relative_eq!(c * p, p * c, epsilon = $tolerance)
                );
            }

            /// Multiplication of two scalars and a point should be compatible with 
            /// multiplication of all scalars. In other words, scalar multiplication 
            /// of two scalar with a point should act associatively, just like the 
            /// multiplication of three scalars.
            ///
            /// Given scalars `a` and `b`, and a point `p`, we have
            /// ```text
            /// (a * b) * p ~= a * (b * p)
            /// ```
            /// Note that the compatability of scalars with points can only be 
            /// approximate and not exact because multiplication of the underlying 
            /// scalars is not associative. 
            #[test]
            fn prop_scalar_multiplication_compatibility(
                a in $ScalarGen::<$ScalarType>(), b in $ScalarGen::<$ScalarType>(), p in $Generator::<$ScalarType>()) {

                prop_assert!(relative_eq!(a * (b * p), (a * b) * p, epsilon = $tolerance));
            }

            /// A scalar `1` acts like a multiplicative identity element.
            ///
            /// Given a vector `p`
            /// ```text
            /// 1 * p = p * 1 = p
            /// ```
            #[test]
            fn prop_one_times_vector_equals_vector(p in $Generator::<$ScalarType>()) {
                let one = num_traits::one();

                prop_assert_eq!(one * p, p);
                prop_assert_eq!(p * one, p);
            }
        }
    }
    }
}

approx_mul_props!(point1_f64_mul_props, Point1, f64, any_point1, any_scalar, 1e-7);
approx_mul_props!(point2_f64_mul_props, Point2, f64, any_point2, any_scalar, 1e-7);
approx_mul_props!(point3_f64_mul_props, Point3, f64, any_point3, any_scalar, 1e-7);

