extern crate cglinalg_core;
extern crate num_traits;
extern crate proptest;


use cglinalg_core::{
    Point1,
    Point2,
    Point3,
    Vector1, 
    Vector2, 
    Vector3, 
    SimdScalar,
};

use proptest::prelude::*;


fn any_scalar<S>() -> impl Strategy<Value = S>
where 
    S: SimdScalar + Arbitrary
{
    any::<S>().prop_map(|scalar| {
        let modulus = num_traits::cast(100_000_000).unwrap();

        scalar % modulus
    })
}

fn any_vector1<S>() -> impl Strategy<Value = Vector1<S>> 
where 
    S: SimdScalar + Arbitrary 
{
    any::<S>().prop_map(|x| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let vector = Vector1::new(x);

        vector % modulus
    })
}

fn any_vector2<S>() -> impl Strategy<Value = Vector2<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let vector = Vector2::new(x, y);
    
        vector % modulus
    })
}

fn any_vector3<S>() -> impl Strategy<Value = Vector3<S>>
where
    S: SimdScalar + Arbitrary
{
    any::<(S, S, S)>().prop_map(|(x, y, z)| { 
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let vector = Vector3::new(x, y, z);

        vector % modulus
    })
}

fn any_point1<S>() -> impl Strategy<Value = Point1<S>> 
where 
    S: SimdScalar + Arbitrary 
{
    any::<S>().prop_map(|x| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let point = Point1::new(x);

        point % modulus
    })
}

fn any_point2<S>() -> impl Strategy<Value = Point2<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let point = Point2::new(x, y);

        point % modulus
    })
}

fn any_point3<S>() -> impl Strategy<Value = Point3<S>>
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S, S)>().prop_map(|(x, y, z)| {
        let modulus = num_traits::cast(100_000_000).unwrap();
        let point = Point3::new(x, y, z);

        point % modulus
    })
}


/// Generate property tests for point multiplication over floating point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property tests 
///    in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$PointN` denotes the name of the point type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of points.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_mul_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
            $ScalarGen,
        };


        proptest! {
            /// Multiplication of a scalar and a point should be commutative.
            ///
            /// Given a constant `c` and a point `p`
            /// ```text
            /// c * p = p * c
            /// ```
            #[test]
            fn prop_scalar_times_point_equals_point_times_scalar(
                c in $ScalarGen::<$ScalarType>(), p in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!(c * p, p * c);
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


/// Generate property tests for point multiplication over floating point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property tests 
///    in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$PointN` denotes the name of the point type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of points.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
macro_rules! exact_mul_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
            $ScalarGen,
        };


        proptest! {
            /// Multiplication of a scalar and a point should be commutative.
            ///
            /// Given a constant `c` and a point `p`
            /// ```text
            /// c * p = p * c
            /// ```
            #[test]
            fn prop_scalar_times_point_equals_point_times_scalar(
                c in $ScalarGen::<$ScalarType>(), p in $Generator::<$ScalarType>()) {
                
                prop_assert_eq!(c * p, p * c);
            }

            /// Multiplication of two scalars and a point should be compatible with 
            /// multiplication of all scalars. In other words, scalar multiplication 
            /// of two scalar with a point should act associatively, just like the 
            /// multiplication of three scalars.
            ///
            /// Given scalars `a` and `b`, and a point `p`, we have
            /// ```text
            /// (a * b) * p = a * (b * p)
            /// ```
            #[test]
            fn prop_scalar_multiplication_compatibility(
                a in $ScalarGen::<$ScalarType>(), 
                b in $ScalarGen::<$ScalarType>(), 
                p in $Generator::<$ScalarType>()) {

                prop_assert_eq!(a * (b * p), (a * b) * p);
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

exact_mul_props!(point1_i32_mul_props, Point1, i32, any_point1, any_scalar);
exact_mul_props!(point2_i32_mul_props, Point2, i32, any_point2, any_scalar);
exact_mul_props!(point3_i32_mul_props, Point3, i32, any_point3, any_scalar);


/// Generate property tests for point/vector arithmetic over floating point scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property tests
///    in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `PointN` denotes the name of the point type.
/// * `$VectorN` denotes the name of the vector type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of vectors.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! approx_arithmetic_props {
    ($TestModuleName:ident, $PointN:ident, $VectorN:ident, $ScalarType:ty, $PointGen:ident, $VectorGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            $VectorN,
        };
        use approx::{
            relative_eq,
        };
        use super::{
            $PointGen,
            $VectorGen,
        };
    

        proptest! {
            /// A point plus a zero vector equals the same point.
            ///
            /// Given a point `p` and a vector `0`
            /// ```text
            /// p + 0 = p
            /// ```
            #[test]
            fn prop_point_plus_zero_equals_vector(p in $PointGen()) {
                let zero_vector = $VectorN::<$ScalarType>::zero();

                prop_assert_eq!(p + zero_vector, p);
            }

            /// Given a point `p` and a vector `v`, we should be able to use `p` and 
            /// `v` interchangeably with their references `&p` and `&v` in 
            /// arithmetic expressions involving points and vectors.
            ///
            /// Given a point `p` and a vector `v`, and their references `&p` and 
            /// `&v`, they should satisfy
            /// ```text
            ///  p +  v = &p +  v
            ///  p +  v =  p + &v
            ///  p +  v = &p + &v
            ///  p + &v = &p +  v
            /// &p +  v =  p + &v
            /// &p +  v = &p + &v
            ///  p + &v = &p + &v
            /// ```
            #[test]
            fn prop_point_plus_vector_equals_refpoint_plus_refvector(
                p in $PointGen::<$ScalarType>(), v in $VectorGen::<$ScalarType>()) {
                
                prop_assert_eq!( p +  v, &p +  v);
                prop_assert_eq!( p +  v,  p + &v);
                prop_assert_eq!( p +  v, &p + &v);
                prop_assert_eq!( p + &v, &p +  v);
                prop_assert_eq!(&p +  v,  p + &v);
                prop_assert_eq!(&p +  v, &p + &v);
                prop_assert_eq!( p + &v, &p + &v);
            }


            /// Point and vector addition should be approximately compatible. 
            /// 
            /// Given a point `p` and vectors `v1` and `v2`, we have
            /// ```text
            /// (p + v1) + v2 ~= p + (v1 + v2)
            /// ```
            #[test]
            fn prop_point_vector_addition_compatible(
                p in $PointGen::<$ScalarType>(), 
                v1 in $VectorGen::<$ScalarType>(), v2 in $VectorGen::<$ScalarType>()) {

                prop_assert!(relative_eq!((p + v1) + v2, p + (v1 + v2), epsilon = $tolerance));
            }

            /// A point minus a zero vector equals the same point.
            ///
            /// Given a point `p` and a vector `0`
            /// ```text
            /// p - 0 = p
            /// ```
            #[test]
            fn prop_point_minus_zero_equals_vector(p in $PointGen()) {
                let zero_vector = $VectorN::<$ScalarType>::zero();

                prop_assert_eq!(p - zero_vector, p);
            }

            /// A point minus itself equals the zero vector.
            ///
            /// Given a point `p` and a vector `0`
            /// ```text
            /// p - p = 0
            /// ```
            #[test]
            fn prop_point_minus_point_equals_zero_vector(p in $PointGen()) {
                let zero_vector = $VectorN::<$ScalarType>::zero();

                prop_assert_eq!(p - p, zero_vector);
            }

            /// Given a point `p` and a vector `v`, we should be able to use `p` and 
            /// `v` interchangeably with their references `&p` and `&v` in 
            /// arithmetic expressions involving points and vectors.
            ///
            /// Given a point `p` and a vector `v`, and their references `&p` and 
            /// `&v`, they should satisfy
            /// ```text
            ///  p -  v = &p -  v
            ///  p -  v =  p - &v
            ///  p -  v = &p - &v
            ///  p - &v = &p -  v
            /// &p -  v =  p - &v
            /// &p -  v = &p - &v
            ///  p - &v = &p - &v
            /// ```
            #[test]
            fn prop_point_minus_vector_equals_refpoint_plus_refvector(
                p in $PointGen::<$ScalarType>(), v in $VectorGen::<$ScalarType>()) {
                
                prop_assert_eq!( p -  v, &p -  v);
                prop_assert_eq!( p -  v,  p - &v);
                prop_assert_eq!( p -  v, &p - &v);
                prop_assert_eq!( p - &v, &p -  v);
                prop_assert_eq!(&p -  v,  p - &v);
                prop_assert_eq!(&p -  v, &p - &v);
                prop_assert_eq!( p - &v, &p - &v);
            }
        }
    }
    }
}

approx_arithmetic_props!(
    point1_f64_arithmetic_props, 
    Point1, 
    Vector1, 
    f64, 
    any_point1, 
    any_vector1, 
    1e-7
);
approx_arithmetic_props!(
    point2_f64_arithmetic_props, 
    Point2, 
    Vector2, 
    f64, 
    any_point2, 
    any_vector2, 
    1e-7
);
approx_arithmetic_props!(
    point3_f64_arithmetic_props, 
    Point3, 
    Vector3, 
    f64, 
    any_point3, 
    any_vector3, 
    1e-7
);


/// Generate property tests for point/vector arithmetic over integer scalars.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property tests
///    in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `PointN` denotes the name of the point type.
/// * `$VectorN` denotes the name of the vector type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of vectors.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_arithmetic_props {
    ($TestModuleName:ident, $PointN:ident, $VectorN:ident, $ScalarType:ty, $PointGen:ident, $VectorGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cglinalg_core::{
            $VectorN,
        };
        use super::{
            $PointGen,
            $VectorGen,
        };
    

        proptest! {
            /// A point plus a zero vector equals the same point.
            ///
            /// Given a point `p` and a vector `0`
            /// ```text
            /// p + 0 = p
            /// ```
            #[test]
            fn prop_point_plus_zero_equals_vector(p in $PointGen()) {
                let zero_vector = $VectorN::<$ScalarType>::zero();

                prop_assert_eq!(p + zero_vector, p);
            }

            /// Given a point `p` and a vector `v`, we should be able to use `p` and 
            /// `v` interchangeably with their references `&p` and `&v` in 
            /// arithmetic expressions involving points and vectors.
            ///
            /// Given a point `p` and a vector `v`, and their references `&p` and 
            /// `&v`, they should satisfy
            /// ```text
            ///  p +  v = &p +  v
            ///  p +  v =  p + &v
            ///  p +  v = &p + &v
            ///  p + &v = &p +  v
            /// &p +  v =  p + &v
            /// &p +  v = &p + &v
            ///  p + &v = &p + &v
            /// ```
            #[test]
            fn prop_point_plus_vector_equals_refpoint_plus_refvector(
                p in $PointGen::<$ScalarType>(), v in $VectorGen::<$ScalarType>()) {
                
                prop_assert_eq!( p +  v, &p +  v);
                prop_assert_eq!( p +  v,  p + &v);
                prop_assert_eq!( p +  v, &p + &v);
                prop_assert_eq!( p + &v, &p +  v);
                prop_assert_eq!(&p +  v,  p + &v);
                prop_assert_eq!(&p +  v, &p + &v);
                prop_assert_eq!( p + &v, &p + &v);
            }


            /// Point and vector addition should be approximately compatible. 
            /// 
            /// Given a point `p` and vectors `v1` and `v2`, we have
            /// ```text
            /// (p + v1) + v2 ~= p + (v1 + v2)
            /// ```
            #[test]
            fn prop_point_vector_addition_compatible(
                p in $PointGen::<$ScalarType>(), 
                v1 in $VectorGen::<$ScalarType>(), v2 in $VectorGen::<$ScalarType>()) {

                prop_assert_eq!((p + v1) + v2, p + (v1 + v2));
            }

            /// A point minus a zero vector equals the same point.
            ///
            /// Given a point `p` and a vector `0`
            /// ```text
            /// p - 0 = p
            /// ```
            #[test]
            fn prop_point_minus_zero_equals_vector(p in $PointGen()) {
                let zero_vector = $VectorN::<$ScalarType>::zero();

                prop_assert_eq!(p - zero_vector, p);
            }

            /// A point minus itself equals the zero vector.
            ///
            /// Given a point `p` and a vector `0`
            /// ```text
            /// p - p = 0
            /// ```
            #[test]
            fn prop_point_minus_point_equals_zero_vector(p in $PointGen()) {
                let zero_vector = $VectorN::<$ScalarType>::zero();

                prop_assert_eq!(p - p, zero_vector);
            }

            /// Given a point `p` and a vector `v`, we should be able to use `p` and 
            /// `v` interchangeably with their references `&p` and `&v` in 
            /// arithmetic expressions involving points and vectors.
            ///
            /// Given a point `p` and a vector `v`, and their references `&p` and 
            /// `&v`, they should satisfy
            /// ```text
            ///  p -  v = &p -  v
            ///  p -  v =  p - &v
            ///  p -  v = &p - &v
            ///  p - &v = &p -  v
            /// &p -  v =  p - &v
            /// &p -  v = &p - &v
            ///  p - &v = &p - &v
            /// ```
            #[test]
            fn prop_point_minus_vector_equals_refpoint_plus_refvector(
                p in $PointGen::<$ScalarType>(), v in $VectorGen::<$ScalarType>()) {
                
                prop_assert_eq!( p -  v, &p -  v);
                prop_assert_eq!( p -  v,  p - &v);
                prop_assert_eq!( p -  v, &p - &v);
                prop_assert_eq!( p - &v, &p -  v);
                prop_assert_eq!(&p -  v,  p - &v);
                prop_assert_eq!(&p -  v, &p - &v);
                prop_assert_eq!( p - &v, &p - &v);
            }
        }
    }
    }
}

exact_arithmetic_props!(point1_i64_arithmetic_props, Point1, Vector1, i64, any_point1, any_vector1);
exact_arithmetic_props!(point2_i64_arithmetic_props, Point2, Vector2, i64, any_point2, any_vector2);
exact_arithmetic_props!(point3_i64_arithmetic_props, Point3, Vector3, i64, any_point3, any_vector3);


/// Generate properties for the point squared **L2** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property tests 
///    in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$PointN` denotes the name of the point type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of points.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! approx_norm_squared_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use approx::{
            relative_ne,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The squared **L2** norm of a point is nonnegative.
            ///
            /// Given a point `p`
            /// ```text
            /// norm_squared(p) >= 0
            /// ```
            #[test]
            fn prop_norm_squared_onnegative(p in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();
                
                prop_assert!(p.norm_squared() >= zero);
            }

            /// The squared **L2** norm function is point separating. In particular, if the 
            /// squared distance between two points `p1` and `p2` is zero, then `p1 = p2`.
            ///
            /// Given vectors `p1` and `p2`
            /// ```text
            /// norm_squared(p1 - p2) = 0 => p1 = p2 
            /// ```
            /// Equivalently, if `p1` is not equal to `p2`, then their squared distance is nonzero
            /// ```text
            /// p1 != p2 => norm_squared(p1 - p2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the norm
            /// function.
            #[test]
            fn prop_norm_squared_approx_point_separating(
                p1 in $Generator::<$ScalarType>(), p2 in $Generator::<$ScalarType>()) {
                
                let zero: $ScalarType = num_traits::zero();

                prop_assume!(relative_ne!(p1, p2, epsilon = $tolerance));
                prop_assert!(
                    relative_ne!((p1 - p2).norm_squared(), zero, epsilon = $tolerance),
                    "\n|p1 - p2|^2 = {}\n",
                    (p1 - p2).norm_squared()
                );
            }
        }
    }
    }
}

approx_norm_squared_props!(point1_f64_norm_squared_props, Point1, f64, any_point1, any_scalar, 1e-8);
approx_norm_squared_props!(point2_f64_norm_squared_props, Point2, f64, any_point2, any_scalar, 1e-8);
approx_norm_squared_props!(point3_f64_norm_squared_props, Point3, f64, any_point3, any_scalar, 1e-8);


/// Generate properties for the point squared **L2** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property tests 
///    in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$PointN` denotes the name of the point type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of vectors.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! approx_norm_squared_synonym_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The [`Point::magnitude_squared`] function and the [`Point::norm_squared`] 
            /// function are synonyms. In particular, given a point `p`
            /// ```text
            /// magnitude_squared(p) = norm_squared(p)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_magnitude_squared_norm_squared_synonyms(v in $Generator::<$ScalarType>()) {
                prop_assert_eq!(v.magnitude_squared(), v.norm_squared());
            }
        }
    }
    }
}

approx_norm_squared_synonym_props!(point1_f64_norm_squared_synonym_props, Point1, f64, any_point1);
approx_norm_squared_synonym_props!(point2_f64_norm_squared_synonym_props, Point2, f64, any_point2);
approx_norm_squared_synonym_props!(point3_f64_norm_squared_synonym_props, Point3, f64, any_point3);


/// Generate properties for the point squared **L2** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property tests 
///    in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$PointN` denotes the name of the point type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of points.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
macro_rules! exact_norm_squared_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The squared **L2** norm of a point is nonnegative.
            ///
            /// Given a point `p`
            /// ```text
            /// norm_squared(p) >= 0
            /// ```
            #[test]
            fn prop_norm_squared_onnegative(p in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();
                
                prop_assert!(p.norm_squared() >= zero);
            }

            /// The squared **L2** norm function is point separating. In particular, if the 
            /// squared distance between two points `p1` and `p2` is zero, then `p1 = p2`.
            ///
            /// Given vectors `p1` and `p2`
            /// ```text
            /// norm_squared(p1 - p2) = 0 => p1 = p2 
            /// ```
            /// Equivalently, if `p1` is not equal to `p2`, then their squared distance is nonzero
            /// ```text
            /// p1 != p2 => norm_squared(p1 - p2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the norm
            /// function.
            #[test]
            fn prop_norm_squared_approx_point_separating(
                p1 in $Generator::<$ScalarType>(), p2 in $Generator::<$ScalarType>()) {
                
                let zero: $ScalarType = num_traits::zero();

                prop_assume!(p1 != p2);
                prop_assert_eq!(
                    (p1 - p2).norm_squared(), zero,
                    "\n|p1 - p2|^2 = {}\n",
                    (p1 - p2).norm_squared()
                );
            }
        }
    }
    }
}

exact_norm_squared_props!(point1_i32_norm_squared_props, Point1, i32, any_point1, any_scalar);
exact_norm_squared_props!(point2_i32_norm_squared_props, Point2, i32, any_point2, any_scalar);
exact_norm_squared_props!(point3_i32_norm_squared_props, Point3, i32, any_point3, any_scalar);

exact_norm_squared_props!(point1_u32_norm_squared_props, Point1, u32, any_point1, any_scalar);
exact_norm_squared_props!(point2_u32_norm_squared_props, Point2, u32, any_point2, any_scalar);
exact_norm_squared_props!(point3_u32_norm_squared_props, Point3, u32, any_point3, any_scalar);


/// Generate properties for the point squared **L2** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property tests 
///    in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$PointN` denotes the name of the point type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of vectors.
/// * `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_norm_squared_synonym_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The [`Point::magnitude_squared`] function and the [`Point::norm_squared`] 
            /// function are synonyms. In particular, given a point `p`
            /// ```text
            /// magnitude_squared(p) = norm_squared(p)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_magnitude_squared_norm_squared_synonyms(v in $Generator::<$ScalarType>()) {
                prop_assert_eq!(v.magnitude_squared(), v.norm_squared());
            }
        }
    }
    }
}

exact_norm_squared_synonym_props!(point1_i32_norm_squared_synonym_props, Point1, i32, any_point1);
exact_norm_squared_synonym_props!(point2_i32_norm_squared_synonym_props, Point2, i32, any_point2);
exact_norm_squared_synonym_props!(point3_i32_norm_squared_synonym_props, Point3, i32, any_point3);

exact_norm_squared_synonym_props!(point1_u32_norm_squared_synonym_props, Point1, u32, any_point1);
exact_norm_squared_synonym_props!(point2_u32_norm_squared_synonym_props, Point2, u32, any_point2);
exact_norm_squared_synonym_props!(point3_u32_norm_squared_synonym_props, Point3, u32, any_point3);


/// Generate properties for the point **L2** norm.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property tests 
///    in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$PointN` denotes the name of the point type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of points.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! norm_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use approx::{
            relative_ne,
        };
        use super::{
            $Generator,
        };


        proptest! {
            /// The **L2** norm of a point is nonnegative.
            ///
            /// Given a point `p`
            /// ```text
            /// norm(p) >= 0
            /// ```
            #[test]
            fn prop_norm_nonnegative(p in $Generator::<$ScalarType>()) {
                let zero: $ScalarType = num_traits::zero();
                
                prop_assert!(p.norm() >= zero);
            }

            /// The **L2** norm function is point separating. In particular, if the 
            /// distance between two points `p1` and `p2` is zero, then `p1 = p2`.
            ///
            /// Given vectors `p1` and `p2`
            /// ```text
            /// norm(p1 - p2) = 0 => p1 = p2 
            /// ```
            /// Equivalently, if `p1` is not equal to `p2`, then their distance is nonzero
            /// ```text
            /// p1 != p2 => norm(p1 - p2) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the norm
            /// function.
            #[test]
            fn prop_norm_approx_point_separating(
                p1 in $Generator::<$ScalarType>(), p2 in $Generator::<$ScalarType>()) {
                
                let zero: $ScalarType = num_traits::zero();

                prop_assume!(relative_ne!(p1, p2, epsilon = $tolerance));
                prop_assert!(
                    relative_ne!((p1 - p2).norm(), zero, epsilon = $tolerance),
                    "\n|p1 - p2| = {}\n",
                    (p1 - p2).norm()
                );
            }
        }
    }
    }
}

norm_props!(point1_f64_norm_props, Point1, f64, any_point1, any_scalar, 1e-8);
norm_props!(point2_f64_norm_props, Point2, f64, any_point2, any_scalar, 1e-8);
norm_props!(point3_f64_norm_props, Point3, f64, any_point3, any_scalar, 1e-8);


/// Generate properties for point norms.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the property tests 
///    in to separate them from each other for each scalar type to prevent 
///    namespace collisions.
/// * `$PointN` denotes the name of the point type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the 
///    set of vectors.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$ScalarGen` is the name of a function or closure for generating scalars.
/// * `$tolerance` specifies the amount of acceptable error for a correct operation 
///    with floating point scalars.
macro_rules! norm_synonym_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// The [`Point::magnitude`] function and the [`Point::norm`] function 
            /// are synonyms. In particular, given a point `p`
            /// ```text
            /// magnitude(p) = norm(p)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_magnitude_norm_synonyms(v in $Generator::<$ScalarType>()) {
                prop_assert_eq!(v.magnitude(), v.norm());
            }

            /// The [`Point::l2_norm`] function and the [`Point::norm`] function
            /// are synonyms. In particular, given a point `p`
            /// ```text
            /// l2_norm(p) = norm(p)
            /// ```
            /// where equality is exact.
            #[test]
            fn prop_l2_norm_norm_synonyms(v in $Generator::<$ScalarType>()) {
                prop_assert_eq!(v.l2_norm(), v.norm());
            }
        }
    }
    }
}

norm_synonym_props!(point1_f64_norm_synonym_props, Point1, f64, any_point1, any_scalar, 1e-8);
norm_synonym_props!(point2_f64_norm_synonym_props, Point2, f64, any_point2, any_scalar, 1e-8);
norm_synonym_props!(point3_f64_norm_synonym_props, Point3, f64, any_point3, any_scalar, 1e-8);

