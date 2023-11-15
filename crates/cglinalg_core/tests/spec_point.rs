extern crate cglinalg_core;
extern crate cglinalg_numeric;
extern crate proptest;


use approx_cmp::{
    relative_eq,
    relative_ne,
};
use cglinalg_core::{
    Point,
    Point1,
    Point2,
    Point3,
    Vector,
    Vector1,
    Vector2,
    Vector3,
};
use cglinalg_numeric::{
    SimdScalar,
    SimdScalarFloat,
    SimdScalarSigned,
};

use proptest::prelude::*;


fn strategy_point_signed_from_abs_range<S, const N: usize>(min_value: S, max_value: S) -> impl Strategy<Value = Point<S, N>>
where
    S: SimdScalarSigned + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarSigned,
    {
        min_value + (value % (max_value - min_value))
    }

    fn rescale_point<S, const N: usize>(value: Point<S, N>, min_value: S, max_value: S) -> Point<S, N>
    where
        S: SimdScalarSigned,
    {
        value.map(|element| rescale(element, min_value, max_value))
    }

    any::<[S; N]>().prop_map(move |array| {
        let point = Point::from(array);

        rescale_point(point, min_value, max_value)
    })
}

fn strategy_scalar_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = S>
where
    S: SimdScalarSigned + Arbitrary,
{
    fn rescale<S: SimdScalarSigned>(value: S, min_value: S, max_value: S) -> S {
        min_value + (value % (max_value - min_value))
    }

    any::<S>().prop_map(move |value| {
        let sign_value = value.signum();
        let abs_value = value.abs();

        sign_value * rescale(abs_value, min_value, max_value)
    })
}

fn strategy_point_any<S, const N: usize>() -> impl Strategy<Value = Point<S, N>>
where
    S: SimdScalarSigned + Arbitrary,
{
    any::<[S; N]>().prop_map(|array| {
        let mut result = Point::origin();
        for i in 0..N {
            let sign_value = array[i].signum();
            let abs_value = array[i].abs();
            result[i] = sign_value * abs_value;
        }

        result
    })
}

fn strategy_vector_any<S, const N: usize>() -> impl Strategy<Value = Vector<S, N>>
where
    S: SimdScalarSigned + Arbitrary,
{
    any::<[S; N]>().prop_map(|array| {
        let mut result = Vector::zero();
        for i in 0..N {
            let sign_value = array[i].signum();
            let abs_value = array[i].abs();
            result[i] = sign_value * abs_value;
        }

        result
    })
}

fn strategy_scalar_i32_any() -> impl Strategy<Value = i32> {
    let min_value = 0_i32;
    // let max_value = f64::floor(f64::sqrt(i32::MAX as f64 / 2_f64)) as i32;
    let max_value = 32767_i32;

    strategy_scalar_signed_from_abs_range(min_value, max_value)
}

fn strategy_point_f64_max_safe_square_root<const N: usize>() -> impl Strategy<Value = Point<f64, N>> {
    let min_value = f64::sqrt(f64::EPSILON);
    let max_value = f64::sqrt(f64::MAX);

    strategy_point_signed_from_abs_range(min_value, max_value)
}

fn strategy_point_i32_max_safe_square_root<const N: usize>() -> impl Strategy<Value = Point<i32, N>> {
    let min_value = 0;
    // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
    let max_square_root = 46340;
    let max_value = max_square_root / (N as i32);

    strategy_point_signed_from_abs_range(min_value, max_value)
}


/// A scalar `1` acts like a multiplicative identity element.
///
/// Given a vector `p`
/// ```text
/// 1 * p == p * 1 == p
/// ```
fn prop_one_times_point_equals_point<S, const N: usize>(p: Point<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalar,
{
    let one = S::one();

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
/// (a * b) * p == a * (b * p)
/// ```
fn prop_scalar_multiplication_compatibility<S, const N: usize>(a: S, b: S, p: Point<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalar,
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
/// p + 0 == p
/// ```
fn prop_point_plus_zero_equals_vector<S, const N: usize>(p: Point<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalar,
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
///  p +  v == &p +  v
///  p +  v ==  p + &v
///  p +  v == &p + &v
///  p + &v == &p +  v
/// &p +  v ==  p + &v
/// &p +  v == &p + &v
///  p + &v == &p + &v
/// ```
#[rustfmt::skip]
fn prop_point_plus_vector_equals_refpoint_plus_refvector<S, const N: usize>(p: Point<S, N>, v: Vector<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalar,
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
/// (p + v1) + v2 == p + (v1 + v2)
/// ```
fn prop_point_vector_addition_compatible<S, const N: usize>(p: Point<S, N>, v1: Vector<S, N>, v2: Vector<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalar,
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
/// (p + v1) + v2 == p + (v1 + v2)
/// ```
fn prop_approx_point_vector_addition_compatible<S, const N: usize>(
    p: Point<S, N>,
    v1: Vector<S, N>,
    v2: Vector<S, N>,
    tolerance: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = (p + v1) + v2;
    let rhs = p + (v1 + v2);

    prop_assert!(relative_eq!(lhs, rhs, abs_diff_all <= tolerance, relative_all <= S::default_epsilon()));

    Ok(())
}

/// A point minus a zero vector equals the same point.
///
/// Given a point `p` and a vector `0`
/// ```text
/// p - 0 == p
/// ```
fn prop_point_minus_zero_equals_vector<S, const N: usize>(p: Point<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalar,
{
    let zero_vector = Vector::zero();

    prop_assert_eq!(p - zero_vector, p);

    Ok(())
}

/// A point minus itself equals the zero vector.
///
/// Given a point `p` and a vector `0`
/// ```text
/// p - p == 0
/// ```
fn prop_point_minus_point_equals_zero_vector<S, const N: usize>(p: Point<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalar,
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
///  p -  v == &p -  v
///  p -  v ==  p - &v
///  p -  v == &p - &v
///  p - &v == &p -  v
/// &p -  v ==  p - &v
/// &p -  v == &p - &v
///  p - &v == &p - &v
/// ```
#[rustfmt::skip]
fn prop_point_minus_vector_equals_refpoint_plus_refvector<S, const N: usize>(p: Point<S, N>, v: Vector<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalar,
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
    S: SimdScalar,
{
    let zero = S::zero();

    prop_assert!(p.norm_squared() >= zero);

    Ok(())
}

/// The squared **L2** norm function is point separating. In particular, if the
/// squared distance between two points `p1` and `p2` is zero, then `p1 == p2`.
///
/// Given vectors `p1` and `p2`
/// ```text
/// norm_squared(p1 - p2) == 0 => p1 == p2
/// ```
/// Equivalently, if `p1` is not equal to `p2`, then their squared distance is nonzero
/// ```text
/// p1 != p2 => norm_squared(p1 - p2) != 0
/// ```
/// For the sake of testability, we use the second form to test the norm
/// function.
fn prop_approx_norm_squared_point_separating<S, const N: usize>(
    p1: Point<S, N>,
    p2: Point<S, N>,
    input_tolerance: S,
    output_tolerance: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    prop_assume!(relative_ne!(p1, p2, abs_diff_all <= input_tolerance, relative_all <= S::default_epsilon()));
    prop_assert!((p1 - p2).norm_squared() > output_tolerance);

    Ok(())
}

/// The squared **L2** norm function is point separating. In particular, if the
/// squared distance between two points `p1` and `p2` is zero, then `p1 == p2`.
///
/// Given vectors `p1` and `p2`
/// ```text
/// norm_squared(p1 - p2) == 0 => p1 == p2
/// ```
/// Equivalently, if `p1` is not equal to `p2`, then their squared distance is nonzero
/// ```text
/// p1 != p2 => norm_squared(p1 - p2) != 0
/// ```
/// For the sake of testability, we use the second form to test the norm
/// function.
fn prop_norm_squared_point_separating<S, const N: usize>(p1: Point<S, N>, p2: Point<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalar,
{
    let zero = S::zero();

    prop_assume!(p1 != p2);
    prop_assert_ne!((p1 - p2).norm_squared(), zero);

    Ok(())
}

/// The squared **L2** norm function is squared homogeneous.
///
/// Given a point `p` and a scalar `c`, the **L2** norm satisfies
/// ```text
/// norm(p * c) == norm(p) * abs(c)
/// ```
/// and the squared **L2** norm function satisfies
/// ```text
/// norm(p * c)^2 == norm(p)^2 * abs(c)^2
/// ```
fn prop_norm_squared_homogeneous_squared<S, const N: usize>(v: Point<S, N>, c: S) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned,
{
    let lhs = (v * c).norm_squared();
    let rhs = v.norm_squared() * c.abs() * c.abs();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The [`Point::magnitude_squared`] function and the [`Point::norm_squared`]
/// function are synonyms. In particular, given a point `p`
/// ```text
/// magnitude_squared(p) == norm_squared(p)
/// ```
/// where equality is exact.
fn prop_magnitude_squared_norm_squared_synonyms<S, const N: usize>(v: Point<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalar,
{
    prop_assert_eq!(v.magnitude_squared(), v.norm_squared());

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
    S: SimdScalarFloat,
{
    let zero = S::zero();

    prop_assert!(p.norm() >= zero);

    Ok(())
}

/// The **L2** norm function is point separating. In particular, if the
/// distance between two points `p1` and `p2` is zero, then `p1 == p2`.
///
/// Given vectors `p1` and `p2`
/// ```text
/// norm(p1 - p2) == 0 => p1 == p2
/// ```
/// Equivalently, if `p1` is not equal to `p2`, then their distance is nonzero
/// ```text
/// p1 != p2 => norm(p1 - p2) != 0
/// ```
/// For the sake of testability, we use the second form to test the norm
/// function.
fn prop_approx_norm_point_separating<S, const N: usize>(
    p1: Point<S, N>,
    p2: Point<S, N>,
    input_tolerance: S,
    output_tolerance: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let zero = S::zero();

    prop_assume!(relative_ne!(p1, p2, abs_diff_all <= input_tolerance, relative_all <= S::default_epsilon()));
    prop_assert!(relative_ne!((p1 - p2).norm(), zero, abs_diff_all <= output_tolerance, relative_all <= S::default_epsilon()));

    Ok(())
}

/// The [`Point::magnitude`] function and the [`Point::norm`] function
/// are synonyms. In particular, given a point `p`
/// ```text
/// magnitude(p) == norm(p)
/// ```
/// where equality is exact.
fn prop_magnitude_norm_synonyms<S, const N: usize>(v: Point<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    prop_assert_eq!(v.magnitude(), v.norm());

    Ok(())
}

/// The [`Point::l2_norm`] function and the [`Point::norm`] function
/// are synonyms. In particular, given a point `p`
/// ```text
/// l2_norm(p) == norm(p)
/// ```
/// where equality is exact.
fn prop_l2_norm_norm_synonyms<S, const N: usize>(v: Point<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    prop_assert_eq!(v.l2_norm(), v.norm());

    Ok(())
}


macro_rules! exact_mul_props {
    ($TestModuleName:ident, $PointType:ident, $ScalarType:ty, $PointGen:ident, $ScalarGen:ident) => {
        #[cfg(test)]
        mod $TestModuleName {
            use proptest::prelude::*;
            proptest! {
                #[test]
                fn prop_scalar_multiplication_compatibility(a in super::$ScalarGen(), b in super::$ScalarGen(), p in super::$PointGen()) {
                    let a: $ScalarType = a;
                    let b: $ScalarType = b;
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_scalar_multiplication_compatibility(a, b, p)?
                }

                #[test]
                fn prop_one_times_point_equals_point(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_one_times_point_equals_point(p)?
                }
            }
        }
    };
}

exact_mul_props!(point1_i32_mul_props, Point1, i32, strategy_point_any, strategy_scalar_i32_any);
exact_mul_props!(point2_i32_mul_props, Point2, i32, strategy_point_any, strategy_scalar_i32_any);
exact_mul_props!(point3_i32_mul_props, Point3, i32, strategy_point_any, strategy_scalar_i32_any);


macro_rules! approx_mul_props {
    ($TestModuleName:ident, $PointType:ident, $ScalarType:ty, $PointGen:ident) => {
        #[cfg(test)]
        mod $TestModuleName {
            use proptest::prelude::*;
            proptest! {
                #[test]
                fn prop_one_times_point_equals_point(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_one_times_point_equals_point(p)?
                }
            }
        }
    };
}

approx_mul_props!(point1_f64_mul_props, Point1, f64, strategy_point_any);
approx_mul_props!(point2_f64_mul_props, Point2, f64, strategy_point_any);
approx_mul_props!(point3_f64_mul_props, Point3, f64, strategy_point_any);


macro_rules! exact_arithmetic_props {
    ($TestModuleName:ident, $PointType:ident, $VectorType:ident, $ScalarType:ty, $PointGen:ident, $VectorGen:ident) => {
        #[cfg(test)]
        mod $TestModuleName {
            use proptest::prelude::*;
            proptest! {
                #[test]
                fn prop_point_plus_zero_equals_vector(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_point_plus_zero_equals_vector(p)?
                }

                #[test]
                fn prop_point_plus_vector_equals_refpoint_plus_refvector(p in super::$PointGen(), v in super::$VectorGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    let v: super::$VectorType<$ScalarType> = v;
                    super::prop_point_plus_vector_equals_refpoint_plus_refvector(p, v)?
                }

                #[test]
                fn prop_point_vector_addition_compatible(p in super::$PointGen(), v1 in super::$VectorGen(), v2 in super::$VectorGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    let v1: super::$VectorType<$ScalarType> = v1;
                    let v2: super::$VectorType<$ScalarType> = v2;
                    super::prop_point_vector_addition_compatible(p, v1, v2)?
                }

                #[test]
                fn prop_point_minus_zero_equals_vector(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_point_minus_zero_equals_vector(p)?
                }

                #[test]
                fn prop_point_minus_point_equals_zero_vector(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_point_minus_point_equals_zero_vector(p)?
                }

                #[test]
                fn prop_point_minus_vector_equals_refpoint_plus_refvector(p in super::$PointGen(), v in super::$VectorGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    let v: super::$VectorType<$ScalarType> = v;
                    super::prop_point_minus_vector_equals_refpoint_plus_refvector(p, v)?
                }
            }
        }
    };
}

exact_arithmetic_props!(
    point1_i64_arithmetic_props,
    Point1,
    Vector1,
    i64,
    strategy_point_any,
    strategy_vector_any
);
exact_arithmetic_props!(
    point2_i64_arithmetic_props,
    Point2,
    Vector2,
    i64,
    strategy_point_any,
    strategy_vector_any
);
exact_arithmetic_props!(
    point3_i64_arithmetic_props,
    Point3,
    Vector3,
    i64,
    strategy_point_any,
    strategy_vector_any
);


macro_rules! approx_arithmetic_props {
    ($TestModuleName:ident, $PointType:ident, $VectorType:ident, $ScalarType:ty, $PointGen:ident, $VectorGen:ident, $tolerance:expr) => {
        #[cfg(test)]
        mod $TestModuleName {
            use proptest::prelude::*;
            proptest! {
                #[test]
                fn prop_point_plus_zero_equals_vector(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_point_plus_zero_equals_vector(p)?
                }

                #[test]
                fn prop_point_plus_vector_equals_refpoint_plus_refvector(p in super::$PointGen(), v in super::$VectorGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    let v: super::$VectorType<$ScalarType> = v;
                    super::prop_point_plus_vector_equals_refpoint_plus_refvector(p, v)?
                }

                #[test]
                fn prop_approx_point_vector_addition_compatible(p in super::$PointGen(), v1 in super::$VectorGen(), v2 in super::$VectorGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    let v1: super::$VectorType<$ScalarType> = v1;
                    let v2: super::$VectorType<$ScalarType> = v2;
                    super::prop_approx_point_vector_addition_compatible(p, v1, v2, $tolerance)?
                }

                #[test]
                fn prop_point_minus_zero_equals_vector(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_point_minus_zero_equals_vector(p)?
                }

                #[test]
                fn prop_point_minus_point_equals_zero_vector(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_point_minus_point_equals_zero_vector(p)?
                }

                #[test]
                fn prop_point_minus_vector_equals_refpoint_plus_refvector(p in super::$PointGen(), v in super::$VectorGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    let v: super::$VectorType<$ScalarType> = v;
                    super::prop_point_minus_vector_equals_refpoint_plus_refvector(p, v)?
                }
            }
        }
    };
}

approx_arithmetic_props!(
    point1_f64_arithmetic_props,
    Point1,
    Vector1,
    f64,
    strategy_point_any,
    strategy_vector_any,
    1e-7
);
approx_arithmetic_props!(
    point2_f64_arithmetic_props,
    Point2,
    Vector2,
    f64,
    strategy_point_any,
    strategy_vector_any,
    1e-7
);
approx_arithmetic_props!(
    point3_f64_arithmetic_props,
    Point3,
    Vector3,
    f64,
    strategy_point_any,
    strategy_vector_any,
    1e-7
);


macro_rules! exact_norm_squared_props {
    ($TestModuleName:ident, $PointType:ident, $ScalarType:ty, $PointGen:ident, $ScalarGen:ident) => {
        #[cfg(test)]
        mod $TestModuleName {
            use proptest::prelude::*;
            proptest! {
                #[test]
                fn prop_norm_squared_nonnegative(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_norm_squared_nonnegative(p)?
                }

                #[test]
                fn prop_norm_squared_point_separating(p1 in super::$PointGen(), p2 in super::$PointGen()) {
                    let p1: super::$PointType<$ScalarType> = p1;
                    let p2: super::$PointType<$ScalarType> = p2;
                    super::prop_norm_squared_point_separating(p1, p2)?
                }

                #[test]
                fn prop_norm_squared_homogeneous_squared(v in super::$PointGen(), c in super::$ScalarGen()) {
                    let v: super::$PointType<$ScalarType> = v;
                    let c: $ScalarType = c;
                    super::prop_norm_squared_homogeneous_squared(v, c)?
                }
            }
        }
    };
}

exact_norm_squared_props!(
    point1_i32_norm_squared_props,
    Point1,
    i32,
    strategy_point_i32_max_safe_square_root,
    strategy_scalar_i32_any
);
exact_norm_squared_props!(
    point2_i32_norm_squared_props,
    Point2,
    i32,
    strategy_point_i32_max_safe_square_root,
    strategy_scalar_i32_any
);
exact_norm_squared_props!(
    point3_i32_norm_squared_props,
    Point3,
    i32,
    strategy_point_i32_max_safe_square_root,
    strategy_scalar_i32_any
);


macro_rules! approx_norm_squared_props {
    ($TestModuleName:ident, $PointType:ident, $ScalarType:ty, $PointGen:ident, $input_tolerance:expr, $output_tolerance:expr) => {
        #[cfg(test)]
        mod $TestModuleName {
            use proptest::prelude::*;
            proptest! {
                #[test]
                fn prop_norm_squared_nonnegative(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_norm_squared_nonnegative(p)?
                }

                #[test]
                fn prop_approx_norm_squared_point_separating(p1 in super::$PointGen(), p2 in super::$PointGen()) {
                    let p1: super::$PointType<$ScalarType> = p1;
                    let p2: super::$PointType<$ScalarType> = p2;
                    super::prop_approx_norm_squared_point_separating(p1, p2, $input_tolerance, $output_tolerance)?
                }
            }
        }
    };
}

approx_norm_squared_props!(
    point1_f64_norm_squared_props,
    Point1,
    f64,
    strategy_point_f64_max_safe_square_root,
    1e-10,
    1e-20
);
approx_norm_squared_props!(
    point2_f64_norm_squared_props,
    Point2,
    f64,
    strategy_point_f64_max_safe_square_root,
    1e-10,
    1e-20
);
approx_norm_squared_props!(
    point3_f64_norm_squared_props,
    Point3,
    f64,
    strategy_point_f64_max_safe_square_root,
    1e-10,
    1e-20
);


macro_rules! approx_norm_squared_synonym_props {
    ($TestModuleName:ident, $PointType:ident, $ScalarType:ty, $PointGen:ident) => {
        #[cfg(test)]
        mod $TestModuleName {
            use proptest::prelude::*;
            proptest! {
                #[test]
                fn prop_magnitude_squared_norm_squared_synonyms(v in super::$PointGen()) {
                    let v: super::$PointType<$ScalarType> = v;
                    super::prop_magnitude_squared_norm_squared_synonyms(v)?
                }
            }
        }
    };
}

approx_norm_squared_synonym_props!(point1_f64_norm_squared_synonym_props, Point1, f64, strategy_point_any);
approx_norm_squared_synonym_props!(point2_f64_norm_squared_synonym_props, Point2, f64, strategy_point_any);
approx_norm_squared_synonym_props!(point3_f64_norm_squared_synonym_props, Point3, f64, strategy_point_any);


macro_rules! exact_norm_squared_synonym_props {
    ($TestModuleName:ident, $PointType:ident, $ScalarType:ty, $PointGen:ident) => {
        #[cfg(test)]
        mod $TestModuleName {
            use proptest::prelude::*;
            proptest! {
                #[test]
                fn prop_magnitude_squared_norm_squared_synonyms(v in super::$PointGen()) {
                    let v: super::$PointType<$ScalarType> = v;
                    super::prop_magnitude_squared_norm_squared_synonyms(v)?
                }
            }
        }
    };
}

exact_norm_squared_synonym_props!(point1_i32_norm_squared_synonym_props, Point1, i32, strategy_point_any);
exact_norm_squared_synonym_props!(point2_i32_norm_squared_synonym_props, Point2, i32, strategy_point_any);
exact_norm_squared_synonym_props!(point3_i32_norm_squared_synonym_props, Point3, i32, strategy_point_any);


macro_rules! approx_norm_props {
    ($TestModuleName:ident, $PointType:ident, $ScalarType:ty, $PointGen:ident, $tolerance:expr) => {
        #[cfg(test)]
        mod $TestModuleName {
            use proptest::prelude::*;
            proptest! {
                #[test]
                fn prop_norm_nonnegative(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_norm_nonnegative(p)?
                }

                #[test]
                fn prop_approx_norm_point_separating(p1 in super::$PointGen(), p2 in super::$PointGen()) {
                    let p1: super::$PointType<$ScalarType> = p1;
                    let p2: super::$PointType<$ScalarType> = p2;
                    super::prop_approx_norm_point_separating(p1, p2, $tolerance, $tolerance)?
                }
            }
        }
    };
}

approx_norm_props!(point1_f64_norm_props, Point1, f64, strategy_point_any, 1e-8);
approx_norm_props!(point2_f64_norm_props, Point2, f64, strategy_point_any, 1e-8);
approx_norm_props!(point3_f64_norm_props, Point3, f64, strategy_point_any, 1e-8);


macro_rules! norm_synonym_props {
    ($TestModuleName:ident, $PointType:ident, $ScalarType:ty, $PointGen:ident) => {
        #[cfg(test)]
        mod $TestModuleName {
            use proptest::prelude::*;
            proptest! {
                #[test]
                fn prop_magnitude_norm_synonyms(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_magnitude_norm_synonyms(p)?
                }

                #[test]
                fn prop_l2_norm_norm_synonyms(p in super::$PointGen()) {
                    let p: super::$PointType<$ScalarType> = p;
                    super::prop_l2_norm_norm_synonyms(p)?
                }
            }
        }
    };
}

norm_synonym_props!(point1_f64_norm_synonym_props, Point1, f64, strategy_point_any);
norm_synonym_props!(point2_f64_norm_synonym_props, Point2, f64, strategy_point_any);
norm_synonym_props!(point3_f64_norm_synonym_props, Point3, f64, strategy_point_any);
