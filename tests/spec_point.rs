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
}

fn any_vector2<S>() -> impl Strategy<Value = Vector2<S>> 
    where S: Scalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let modulus: S = num_traits::cast(1_000_000).unwrap();
        let vector = Vector2::new(x, y);
    
        vector % modulus
    })
}

fn any_vector3<S>() -> impl Strategy<Value = Vector3<S>>
    where S: Scalar + Arbitrary
{
    any::<(S, S, S)>().prop_map(|(x, y, z)| { 
        let modulus: S = num_traits::cast(1_000_000).unwrap();
        let vector = Vector3::new(x, y, z);

        vector % modulus
    })
}

fn any_point1<S>() -> impl Strategy<Value = Point1<S>> 
    where S: Scalar + Arbitrary 
{
    any::<S>().prop_map(|x| {
        let modulus: S = num_traits::cast(1_000_000).unwrap();
        let point = Point1::new(x);

        point % modulus
    })
}

fn any_point2<S>() -> impl Strategy<Value = Point2<S>> 
    where S: Scalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let modulus: S = num_traits::cast(1_000_000).unwrap();
        let point = Point2::new(x, y);

        point % modulus
    })
}

fn any_point3<S>() -> impl Strategy<Value = Point3<S>>
    where S: Scalar + Arbitrary
{
    any::<(S, S, S)>().prop_map(|(x, y, z)| {
        let modulus = num_traits::cast(1_000_000).unwrap();
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
            /// Multiplication of a scalar and a point should be approximately 
            /// commutative.
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



/// Generate property tests for point indexing.
///
/// ### Macro Parameters
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name we give to the module we place the tests in 
///    to separate them from each other for each scalar type to prevent 
////   namespace collisions.
/// * `$PointN` denotes the name of the vector type.
/// * `$ScalarType` denotes the underlying system of numbers that compose the
///    set of points.
/// * `$Generator` is the name of a function or closure for generating examples.
/// * `$UpperBound` denotes the upper bound on the range of acceptable indices.
macro_rules! index_props {
    ($TestModuleName:ident, $Point:ident, $ScalarType:ty, $Generator:ident, $UpperBound:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use super::{
            $Generator,
        };


        proptest! {
            /// When a point is treated like an array, it should accept all indices
            /// below the length of the array.
            ///
            /// Given a point `p`, it should return the entry at position `index` in the 
            /// underlying storage of the point when the given index is in bounds.
            #[test]
            fn prop_accepts_all_indices_in_of_bounds(
                p in $Generator::<$ScalarType>(), index in 0..$UpperBound as usize) {

                prop_assert_eq!(p[index], p[index]);
            }
    
            /// When a point is treated like an array, it should reject any input index outside
            /// the length of the array.
            ///
            /// Given a point `v`, when the element index `index` is out of bounds, it should 
            /// generate a panic just like an array indexed out of bounds.
            #[test]
            #[should_panic]
            fn prop_panics_when_index_out_of_bounds(
                p in $Generator::<$ScalarType>(), index in $UpperBound..usize::MAX) {
                
                prop_assert_eq!(p[index], p[index]);
            }
        }
    }
    }
}

index_props!(vector1_f64_index_props, Vector1, f64, any_point1, 1);
index_props!(vector2_f64_index_props, Vector2, f64, any_point2, 2);
index_props!(vector3_f64_index_props, Vector3, f64, any_point3, 3);


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
        use cglinalg::{
            $VectorN,
            Zero,
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
                let zero_vec = $VectorN::<$ScalarType>::zero();

                prop_assert_eq!(p + zero_vec, p);
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
            /// Note that floating point vector addition cannot be exactly 
            /// associative because arithmetic with floating point numbers 
            /// is not associative.
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
                let zero_vec = $VectorN::<$ScalarType>::zero();

                prop_assert_eq!(p - zero_vec, p);
            }

            /// A point minus itself equals the zero vector.
            ///
            /// Given a point `p` and a vector `0`
            /// ```text
            /// p - p = 0
            /// ```
            #[test]
            fn prop_point_minus_point_equals_zero_vector(p in $PointGen()) {
                let zero_vec = $VectorN::<$ScalarType>::zero();

                prop_assert_eq!(p - p, zero_vec);
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
        use cglinalg::{
            $VectorN,
            Zero,
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
                let zero_vec = $VectorN::<$ScalarType>::zero();

                prop_assert_eq!(p + zero_vec, p);
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
            /// (p + v1) + v2 = p + (v1 + v2)
            /// ```
            /// Note that floating point vector addition cannot be exactly 
            /// associative because arithmetic with floating point numbers 
            /// is not associative.
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
                let zero_vec = $VectorN::<$ScalarType>::zero();

                prop_assert_eq!(p - zero_vec, p);
            }

            /// A point minus itself equals the zero vector.
            ///
            /// Given a point `p` and a vector `0`
            /// ```text
            /// p - p = 0
            /// ```
            #[test]
            fn prop_point_minus_point_equals_zero_vector(p in $PointGen()) {
                let zero_vec = $VectorN::<$ScalarType>::zero();

                prop_assert_eq!(p - p, zero_vec);
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

