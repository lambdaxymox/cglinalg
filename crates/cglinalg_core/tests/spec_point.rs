extern crate cglinalg_core;
extern crate num_traits;
extern crate proptest;


use cglinalg_core::{
    Point,
    Point1,
    Point2,
    Point3,
    Vector,
    Vector1, 
    Vector2, 
    Vector3, 
    SimdScalar,
    SimdScalarFloat,
};
use approx::{
    relative_eq,
    relative_ne,
};

use proptest::prelude::*;


fn strategy_scalar_any<S>() -> impl Strategy<Value = S>
where 
    S: SimdScalar + Arbitrary
{
    any::<S>().prop_map(|scalar| {
        let modulus = num_traits::cast(100_000_000).unwrap();

        scalar % modulus
    })
}

fn strategy_vector1_any<S>() -> impl Strategy<Value = Vector1<S>> 
where 
    S: SimdScalar + Arbitrary 
{
    any::<S>().prop_map(|x| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let vector = Vector1::new(x);

        vector % modulus
    })
}

fn strategy_vector2_any<S>() -> impl Strategy<Value = Vector2<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let vector = Vector2::new(x, y);
    
        vector % modulus
    })
}

fn strategy_vector3_any<S>() -> impl Strategy<Value = Vector3<S>>
where
    S: SimdScalar + Arbitrary
{
    any::<(S, S, S)>().prop_map(|(x, y, z)| { 
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let vector = Vector3::new(x, y, z);

        vector % modulus
    })
}

fn strategy_point1_any<S>() -> impl Strategy<Value = Point1<S>> 
where 
    S: SimdScalar + Arbitrary 
{
    any::<S>().prop_map(|x| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let point = Point1::new(x);

        point % modulus
    })
}

fn strategy_point2_any<S>() -> impl Strategy<Value = Point2<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let point = Point2::new(x, y);

        point % modulus
    })
}

fn strategy_point3_any<S>() -> impl Strategy<Value = Point3<S>>
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S, S)>().prop_map(|(x, y, z)| {
        let modulus = num_traits::cast(100_000_000).unwrap();
        let point = Point3::new(x, y, z);

        point % modulus
    })
}

fn strategy_point1_f64_norm_squared() -> impl Strategy<Value = Point1<f64>> {
    fn rescale(value: f64, min_value: f64, max_value: f64) -> f64 {
        min_value + (value % (max_value - min_value))
    }

    any::<f64>().prop_map(|_x| {
        let min_value = f64::sqrt(f64::EPSILON);
        let max_value = f64::sqrt(f64::MAX);
        let x = rescale(_x, min_value, max_value);

        Point1::new(x)
    })
    .no_shrink()
}

fn strategy_point2_f64_norm_squared() -> impl Strategy<Value = Point2<f64>> {
    fn rescale(value: f64, min_value: f64, max_value: f64) -> f64 {
        min_value + (value % (max_value - min_value))
    }

    any::<(f64, f64)>().prop_map(|(_x, _y)| {
        let min_value = f64::sqrt(f64::EPSILON);
        let max_value = f64::sqrt(f64::MAX);
        let x = rescale(_x, min_value, max_value);
        let y = rescale(_y, min_value, max_value);

        Point2::new(x, y)
    })
    .no_shrink()
}

fn strategy_point3_f64_norm_squared() -> impl Strategy<Value = Point3<f64>> {
    fn rescale(value: f64, min_value: f64, max_value: f64) -> f64 {
        min_value + (value % (max_value - min_value))
    }

    any::<(f64, f64, f64)>().prop_map(|(_x, _y, _z)| {
        let min_value = f64::sqrt(f64::EPSILON);
        let max_value = f64::sqrt(f64::MAX);
        let x = rescale(_x, min_value, max_value);
        let y = rescale(_y, min_value, max_value);
        let z = rescale(_z, min_value, max_value);

        Point3::new(x, y, z)
    })
    .no_shrink()
}

fn strategy_point1_i32_norm_squared() -> impl Strategy<Value = Point1<i32>> {
    any::<i32>().prop_map(|_x| {
        let min_value = 0;
        // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
        let max_square_root = 46340;
        let max_value = max_square_root;
        let x = min_value + (_x % (max_value - min_value + 1));

        Point1::new(x)
    })
}

fn strategy_point2_i32_norm_squared() -> impl Strategy<Value = Point2<i32>> {
    any::<(i32, i32)>().prop_map(|(_x, _y)| {
        let min_value = 0;
        // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
        let max_square_root = 46340;
        let max_value = max_square_root / 2;
        let x = min_value + (_x % (max_value - min_value + 1));
        let y = min_value + (_y % (max_value - min_value + 1));

        Point2::new(x, y)
    })
}

fn strategy_point3_i32_norm_squared() -> impl Strategy<Value = Point3<i32>> {
    any::<(i32, i32, i32)>().prop_map(|(_x, _y, _z)| {
        let min_value = 0;
        // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
        let max_square_root = 46340;
        let max_value = max_square_root / 3;
        let x = min_value + (_x % (max_value - min_value + 1));
        let y = min_value + (_y % (max_value - min_value + 1));
        let z = min_value + (_z % (max_value - min_value + 1));

        Point3::new(x, y, z)
    })
}


/// A scalar `1` acts like a multiplicative identity element.
///
/// Given a vector `p`
/// ```text
/// 1 * p = p * 1 = p
/// ```
fn prop_one_times_vector_equals_vector<S, const N: usize>(p: Point<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar + Arbitrary
{
    let one = num_traits::one();

    prop_assert_eq!(p * one, p);

    Ok(())
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
fn prop_scalar_multiplication_compatibility<S, const N: usize>(a: S, b: S, p: Point<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar + Arbitrary
{
    let lhs = (p * a) * b;
    let rhs = p * (a * b);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// A point plus a zero vector equals the same point.
///
/// Given a point `p` and a vector `0`
/// ```text
/// p + 0 = p
/// ```
fn prop_point_plus_zero_equals_vector<S, const N: usize>(p: Point<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar + Arbitrary
{
    let zero_vector = Vector::zero();

    prop_assert_eq!(p + zero_vector, p);

    Ok(())
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
fn prop_point_plus_vector_equals_refpoint_plus_refvector<S, const N: usize>(p: Point<S, N>, v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar + Arbitrary
{   
    prop_assert_eq!( p +  v, &p +  v);
    prop_assert_eq!( p +  v,  p + &v);
    prop_assert_eq!( p +  v, &p + &v);
    prop_assert_eq!( p + &v, &p +  v);
    prop_assert_eq!(&p +  v,  p + &v);
    prop_assert_eq!(&p +  v, &p + &v);
    prop_assert_eq!( p + &v, &p + &v);

    Ok(())
}


/// Point and vector addition should be compatible. 
/// 
/// Given a point `p` and vectors `v1` and `v2`, we have
/// ```text
/// (p + v1) + v2 = p + (v1 + v2)
/// ```
fn prop_point_vector_addition_compatible<S, const N: usize>(p: Point<S, N>, v1: Vector<S, N>, v2: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar + Arbitrary
{
    let lhs = (p + v1) + v2;
    let rhs = p + (v1 + v2);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// Point and vector addition should be compatible. 
/// 
/// Given a point `p` and vectors `v1` and `v2`, we have
/// ```text
/// (p + v1) + v2 = p + (v1 + v2)
/// ```
fn prop_approx_point_vector_addition_compatible<S, const N: usize>(p: Point<S, N>, v1: Vector<S, N>, v2: Vector<S, N>, tolerance: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat + Arbitrary
{
    let lhs = (p + v1) + v2;
    let rhs = p + (v1 + v2);

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// A point minus a zero vector equals the same point.
///
/// Given a point `p` and a vector `0`
/// ```text
/// p - 0 = p
/// ```
fn prop_point_minus_zero_equals_vector<S, const N: usize>(p: Point<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar + Arbitrary
{
    let zero_vector = Vector::zero();

    prop_assert_eq!(p - zero_vector, p);

    Ok(())
}

/// A point minus itself equals the zero vector.
///
/// Given a point `p` and a vector `0`
/// ```text
/// p - p = 0
/// ```
fn prop_point_minus_point_equals_zero_vector<S, const N: usize>(p: Point<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar + Arbitrary
{
    let zero_vector = Vector::zero();

    prop_assert_eq!(p - p, zero_vector);

    Ok(())
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
fn prop_point_minus_vector_equals_refpoint_plus_refvector<S, const N: usize>(p: Point<S, N>, v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar + Arbitrary
{   
    prop_assert_eq!( p -  v, &p -  v);
    prop_assert_eq!( p -  v,  p - &v);
    prop_assert_eq!( p -  v, &p - &v);
    prop_assert_eq!( p - &v, &p -  v);
    prop_assert_eq!(&p -  v,  p - &v);
    prop_assert_eq!(&p -  v, &p - &v);
    prop_assert_eq!( p - &v, &p - &v);

    Ok(())
}

/// The squared **L2** norm of a point is nonnegative.
///
/// Given a point `p`
/// ```text
/// norm_squared(p) >= 0
/// ```
fn prop_norm_squared_nonnegative<S, const N: usize>(p: Point<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar + Arbitrary
{
    let zero = num_traits::zero();
    
    prop_assert!(p.norm_squared() >= zero);

    Ok(())
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
fn prop_approx_norm_squared_point_separating<S, const N: usize>(p1: Point<S, N>, p2: Point<S, N>, input_tolerance: S, output_tolerance: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat + Arbitrary
{
    prop_assume!(relative_ne!(p1, p2, epsilon = input_tolerance));
    prop_assert!(
        (p1 - p2).norm_squared() > output_tolerance,
        "\n|p1 - p2|^2 = {}\n",
        (p1 - p2).norm_squared()
    );

    Ok(())
}

/// The [`Point::magnitude_squared`] function and the [`Point::norm_squared`] 
/// function are synonyms. In particular, given a point `p`
/// ```text
/// magnitude_squared(p) = norm_squared(p)
/// ```
/// where equality is exact.
fn prop_magnitude_squared_norm_squared_synonyms<S, const N: usize>(v: Point<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(v.magnitude_squared(), v.norm_squared());

    Ok(())
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
fn prop_norm_squared_point_separating<S, const N: usize>(p1: Point<S, N>, p2: Point<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar + Arbitrary
{   
    let zero = num_traits::zero();

    prop_assume!(p1 != p2);
    prop_assert_ne!(
        (p1 - p2).norm_squared(), zero,
        "\n|p1 - p2|^2 = {}\n",
        (p1 - p2).norm_squared()
    );

    Ok(())
}

/// The **L2** norm of a point is nonnegative.
///
/// Given a point `p`
/// ```text
/// norm(p) >= 0
/// ```
fn prop_norm_nonnegative<S, const N: usize>(p: Point<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat + Arbitrary
{
    let zero = num_traits::zero();
    
    prop_assert!(p.norm() >= zero);

    Ok(())
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
fn prop_norm_approx_point_separating<S, const N: usize>(p1: Point<S, N>, p2: Point<S, N>, input_tolerance: S, output_tolerance: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat + Arbitrary
{   
    let zero = num_traits::zero();

    prop_assume!(relative_ne!(p1, p2, epsilon = input_tolerance));
    prop_assert!(
        relative_ne!((p1 - p2).norm(), zero, epsilon = output_tolerance),
        "\n|p1 - p2| = {}\n",
        (p1 - p2).norm()
    );

    Ok(())
}

/// The [`Point::magnitude`] function and the [`Point::norm`] function 
/// are synonyms. In particular, given a point `p`
/// ```text
/// magnitude(p) = norm(p)
/// ```
/// where equality is exact.
fn prop_magnitude_norm_synonyms<S, const N: usize>(v: Point<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat + Arbitrary
{
    prop_assert_eq!(v.magnitude(), v.norm());

    Ok(())
}

/// The [`Point::l2_norm`] function and the [`Point::norm`] function
/// are synonyms. In particular, given a point `p`
/// ```text
/// l2_norm(p) = norm(p)
/// ```
/// where equality is exact.
fn prop_l2_norm_norm_synonyms<S, const N: usize>(v: Point<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat + Arbitrary
{
    prop_assert_eq!(v.l2_norm(), v.norm());

    Ok(())
}


macro_rules! approx_mul_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_one_times_vector_equals_vector(p in super::$Generator()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_one_times_vector_equals_vector(p)?
            }
        }
    }
    }
}

approx_mul_props!(point1_f64_mul_props, Point1, f64, strategy_point1_any, strategy_scalar_any, 1e-7);
approx_mul_props!(point2_f64_mul_props, Point2, f64, strategy_point2_any, strategy_scalar_any, 1e-7);
approx_mul_props!(point3_f64_mul_props, Point3, f64, strategy_point3_any, strategy_scalar_any, 1e-7);


macro_rules! exact_mul_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_scalar_multiplication_compatibility(a in super::$ScalarGen(), b in super::$ScalarGen(), p in super::$Generator()) {
                let a: $ScalarType = a;
                let b: $ScalarType = b;
                let p: super::$PointN<$ScalarType> = p;
                super::prop_scalar_multiplication_compatibility(a, b, p)?
            }

            #[test]
            fn prop_one_times_vector_equals_vector(p in super::$Generator()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_one_times_vector_equals_vector(p)?
            }
        }
    }
    }
}

exact_mul_props!(point1_i32_mul_props, Point1, i32, strategy_point1_any, strategy_scalar_any);
exact_mul_props!(point2_i32_mul_props, Point2, i32, strategy_point2_any, strategy_scalar_any);
exact_mul_props!(point3_i32_mul_props, Point3, i32, strategy_point3_any, strategy_scalar_any);


macro_rules! approx_arithmetic_props {
    ($TestModuleName:ident, $PointN:ident, $VectorN:ident, $ScalarType:ty, $PointGen:ident, $VectorGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_point_plus_zero_equals_vector(p in super::$PointGen()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_point_plus_zero_equals_vector(p)?
            }

            #[test]
            fn prop_point_plus_vector_equals_refpoint_plus_refvector(p in super::$PointGen(), v in super::$VectorGen()) {
                let p: super::$PointN<$ScalarType> = p;
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_point_plus_vector_equals_refpoint_plus_refvector(p, v)?
            }

            #[test]
            fn prop_approx_point_vector_addition_compatible(p in super::$PointGen(), v1 in super::$VectorGen(), v2 in super::$VectorGen()) {
                let p: super::$PointN<$ScalarType> = p;
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_approx_point_vector_addition_compatible(p, v1, v2, 1e-7)?
            }

            #[test]
            fn prop_point_minus_zero_equals_vector(p in super::$PointGen()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_point_minus_zero_equals_vector(p)?
            }

            #[test]
            fn prop_point_minus_point_equals_zero_vector(p in super::$PointGen()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_point_minus_point_equals_zero_vector(p)?
            }

            #[test]
            fn prop_point_minus_vector_equals_refpoint_plus_refvector(p in super::$PointGen(), v in super::$VectorGen()) {
                let p: super::$PointN<$ScalarType> = p;
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_point_minus_vector_equals_refpoint_plus_refvector(p, v)?
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
    strategy_point1_any, 
    strategy_vector1_any, 
    1e-7
);
approx_arithmetic_props!(
    point2_f64_arithmetic_props, 
    Point2, 
    Vector2, 
    f64, 
    strategy_point2_any, 
    strategy_vector2_any, 
    1e-7
);
approx_arithmetic_props!(
    point3_f64_arithmetic_props, 
    Point3, 
    Vector3, 
    f64, 
    strategy_point3_any, 
    strategy_vector3_any, 
    1e-7
);


macro_rules! exact_arithmetic_props {
    ($TestModuleName:ident, $PointN:ident, $VectorN:ident, $ScalarType:ty, $PointGen:ident, $VectorGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_point_plus_zero_equals_vector(p in super::$PointGen()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_point_plus_zero_equals_vector(p)?
            }

            #[test]
            fn prop_point_plus_vector_equals_refpoint_plus_refvector(p in super::$PointGen(), v in super::$VectorGen()) {
                let p: super::$PointN<$ScalarType> = p;
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_point_plus_vector_equals_refpoint_plus_refvector(p, v)?
            }

            #[test]
            fn prop_point_vector_addition_compatible(p in super::$PointGen(), v1 in super::$VectorGen(), v2 in super::$VectorGen()) {
                let p: super::$PointN<$ScalarType> = p;
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_point_vector_addition_compatible(p, v1, v2)?
            }

            #[test]
            fn prop_point_minus_zero_equals_vector(p in super::$PointGen()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_point_minus_zero_equals_vector(p)?
            }

            #[test]
            fn prop_point_minus_point_equals_zero_vector(p in super::$PointGen()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_point_minus_point_equals_zero_vector(p)?
            }

            #[test]
            fn prop_point_minus_vector_equals_refpoint_plus_refvector(p in super::$PointGen(), v in super::$VectorGen()) {
                let p: super::$PointN<$ScalarType> = p;
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_point_minus_vector_equals_refpoint_plus_refvector(p, v)?
            }
        }
    }
    }
}

exact_arithmetic_props!(point1_i64_arithmetic_props, Point1, Vector1, i64, strategy_point1_any, strategy_vector1_any);
exact_arithmetic_props!(point2_i64_arithmetic_props, Point2, Vector2, i64, strategy_point2_any, strategy_vector2_any);
exact_arithmetic_props!(point3_i64_arithmetic_props, Point3, Vector3, i64, strategy_point3_any, strategy_vector3_any);


macro_rules! approx_norm_squared_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $input_tolerance:expr, $output_tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_norm_squared_nonnegative(p in super::$Generator()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_norm_squared_nonnegative(p)?
            }

            #[test]
            fn prop_norm_squared_approx_point_separating(p1 in super::$Generator(), p2 in super::$Generator()) {
                let p1: super::$PointN<$ScalarType> = p1;
                let p2: super::$PointN<$ScalarType> = p2;
                super::prop_approx_norm_squared_point_separating(p1, p2, $input_tolerance, $output_tolerance)?
            }
        }
    }
    }
}

approx_norm_squared_props!(point1_f64_norm_squared_props, Point1, f64, strategy_point1_f64_norm_squared, any_scalar, 1e-10, 1e-20);
approx_norm_squared_props!(point2_f64_norm_squared_props, Point2, f64, strategy_point2_f64_norm_squared, any_scalar, 1e-10, 1e-20);
approx_norm_squared_props!(point3_f64_norm_squared_props, Point3, f64, strategy_point3_f64_norm_squared, any_scalar, 1e-10, 1e-20);


macro_rules! approx_norm_squared_synonym_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_magnitude_squared_norm_squared_synonyms(v in super::$Generator()) {
                let v: super::$PointN<$ScalarType> = v;
                super::prop_magnitude_squared_norm_squared_synonyms(v)?
            }
        }
    }
    }
}

approx_norm_squared_synonym_props!(point1_f64_norm_squared_synonym_props, Point1, f64, strategy_point1_any);
approx_norm_squared_synonym_props!(point2_f64_norm_squared_synonym_props, Point2, f64, strategy_point2_any);
approx_norm_squared_synonym_props!(point3_f64_norm_squared_synonym_props, Point3, f64, strategy_point3_any);


macro_rules! exact_norm_squared_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_norm_squared_nonnegative(p in super::$Generator()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_norm_squared_nonnegative(p)?
            }

            #[test]
            fn prop_norm_squared_point_separating(p1 in super::$Generator(), p2 in super::$Generator()) {
                let p1: super::$PointN<$ScalarType> = p1;
                let p2: super::$PointN<$ScalarType> = p2;
                super::prop_norm_squared_point_separating(p1, p2)?
            }
        }
    }
    }
}

exact_norm_squared_props!(point1_i32_norm_squared_props, Point1, i32, strategy_point1_i32_norm_squared, any_scalar);
exact_norm_squared_props!(point2_i32_norm_squared_props, Point2, i32, strategy_point2_i32_norm_squared, any_scalar);
exact_norm_squared_props!(point3_i32_norm_squared_props, Point3, i32, strategy_point3_i32_norm_squared, any_scalar);


macro_rules! exact_norm_squared_synonym_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_magnitude_squared_norm_squared_synonyms(v in super::$Generator()) {
                let v: super::$PointN<$ScalarType> = v;
                super::prop_magnitude_squared_norm_squared_synonyms(v)?
            }
        }
    }
    }
}

exact_norm_squared_synonym_props!(point1_i32_norm_squared_synonym_props, Point1, i32, strategy_point1_any);
exact_norm_squared_synonym_props!(point2_i32_norm_squared_synonym_props, Point2, i32, strategy_point2_any);
exact_norm_squared_synonym_props!(point3_i32_norm_squared_synonym_props, Point3, i32, strategy_point3_any);


macro_rules! norm_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_norm_nonnegative(p in super::$Generator()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_norm_nonnegative(p)?
            }

            #[test]
            fn prop_norm_approx_point_separating(p1 in super::$Generator(), p2 in super::$Generator()) {
                let p1: super::$PointN<$ScalarType> = p1;
                let p2: super::$PointN<$ScalarType> = p2;
                super::prop_norm_approx_point_separating(p1, p2, 1e-8, 1e-8)?
            }
        }
    }
    }
}

norm_props!(point1_f64_norm_props, Point1, f64, strategy_point1_any, any_scalar, 1e-8);
norm_props!(point2_f64_norm_props, Point2, f64, strategy_point2_any, any_scalar, 1e-8);
norm_props!(point3_f64_norm_props, Point3, f64, strategy_point3_any, any_scalar, 1e-8);


macro_rules! norm_synonym_props {
    ($TestModuleName:ident, $PointN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_magnitude_norm_synonyms(p in super::$Generator()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_magnitude_norm_synonyms(p)?
            }

            #[test]
            fn prop_l2_norm_norm_synonyms(p in super::$Generator()) {
                let p: super::$PointN<$ScalarType> = p;
                super::prop_l2_norm_norm_synonyms(p)?
            }
        }
    }
    }
}

norm_synonym_props!(point1_f64_norm_synonym_props, Point1, f64, strategy_point1_any, any_scalar, 1e-8);
norm_synonym_props!(point2_f64_norm_synonym_props, Point2, f64, strategy_point2_any, any_scalar, 1e-8);
norm_synonym_props!(point3_f64_norm_synonym_props, Point3, f64, strategy_point3_any, any_scalar, 1e-8);

