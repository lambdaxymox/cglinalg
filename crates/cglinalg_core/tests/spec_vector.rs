extern crate cglinalg_core;
extern crate num_traits;
extern crate proptest;


use cglinalg_core::{
    Vector,
    Vector1,
    Vector2,
    Vector3,
    Vector4, 
    SimdScalar,
    SimdScalarSigned,
    SimdScalarOrd,
    SimdScalarFloat,
};
use approx::{
    relative_ne,
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
        let modulus = num_traits::cast(100_000_000).unwrap();
        let vector = Vector1::new(x);

        vector % modulus
    })
}

fn any_vector2<S>() -> impl Strategy<Value = Vector2<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(x, y)| {
        let modulus = num_traits::cast(100_000_000).unwrap();
        let vector = Vector2::new(x, y);

        vector % modulus
    })
}

fn any_vector3<S>() -> impl Strategy<Value = Vector3<S>>
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S, S)>().prop_map(|(x, y, z)| {
        let modulus = num_traits::cast(100_000_000).unwrap();
        let vector = Vector3::new(x, y, z);

        vector % modulus
    })
}

fn any_vector4<S>() -> impl Strategy<Value = Vector4<S>>
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S, S, S)>().prop_map(|(x, y, z, w)| {
        let modulus = num_traits::cast(100_000_000).unwrap();
        let vector = Vector4::new(x, y, z, w);

        vector % modulus
    })
}

fn any_u32() -> impl Strategy<Value = u32> {
    any::<u32>().prop_map(|i| {
        let modulus = 50;

        i % modulus
    })
}

fn any_vector1_norm_squared_f64() -> impl Strategy<Value = Vector1<f64>> {
    fn rescale(value: f64, min_value: f64, max_value: f64) -> f64 {
        min_value + (value % (max_value - min_value))
    }

    any::<f64>().prop_map(|_x| {
        let min_value = f64::sqrt(f64::EPSILON);
        let max_value = f64::sqrt(f64::MAX);
        let x = rescale(_x, min_value, max_value);

        Vector1::new(x)
    })
    .no_shrink()
}

fn any_vector2_norm_squared_f64() -> impl Strategy<Value = Vector2<f64>> {
    fn rescale(value: f64, min_value: f64, max_value: f64) -> f64 {
        min_value + (value % (max_value - min_value))
    }

    any::<(f64, f64)>().prop_map(|(_x, _y)| {
        let min_value = f64::sqrt(f64::EPSILON);
        let max_value = f64::sqrt(f64::MAX);
        let x = rescale(_x, min_value, max_value);
        let y = rescale(_y, min_value, max_value);

        Vector2::new(x, y)
    })
    .no_shrink()
}

fn any_vector3_norm_squared_f64() -> impl Strategy<Value = Vector3<f64>> {
    fn rescale(value: f64, min_value: f64, max_value: f64) -> f64 {
        min_value + (value % (max_value - min_value))
    }

    any::<(f64, f64, f64)>().prop_map(|(_x, _y, _z)| {
        let min_value = f64::sqrt(f64::EPSILON);
        let max_value = f64::sqrt(f64::MAX);
        let x = rescale(_x, min_value, max_value);
        let y = rescale(_y, min_value, max_value);
        let z = rescale(_z, min_value, max_value);

        Vector3::new(x, y, z)
    })
    .no_shrink()
}

fn any_vector4_norm_squared_f64() -> impl Strategy<Value = Vector4<f64>> {
    fn rescale(value: f64, min_value: f64, max_value: f64) -> f64 {
        min_value + (value % (max_value - min_value))
    }

    any::<(f64, f64, f64, f64)>().prop_map(|(_x, _y, _z, _w)| {
        let min_value = f64::sqrt(f64::EPSILON);
        let max_value = f64::sqrt(f64::MAX);
        let x = rescale(_x, min_value, max_value);
        let y = rescale(_y, min_value, max_value);
        let z = rescale(_z, min_value, max_value);
        let w = rescale(_w, min_value, max_value);

        Vector4::new(x, y, z, w)
    })
    .no_shrink()
}

fn any_vector1_norm_squared_i32() -> impl Strategy<Value = Vector1<i32>> {
    any::<i32>().prop_map(|_x| {
        let min_value = 0;
        // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
        let max_square_root = 46340;
        let max_value = max_square_root;
        let x = min_value + (_x % (max_value - min_value + 1));

        Vector1::new(x)
    })
}

fn any_vector2_norm_squared_i32() -> impl Strategy<Value = Vector2<i32>> {
    any::<(i32, i32)>().prop_map(|(_x, _y)| {
        let min_value = 0;
        // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
        let max_square_root = 46340;
        let max_value = max_square_root / 2;
        let x = min_value + (_x % (max_value - min_value + 1));
        let y = min_value + (_y % (max_value - min_value + 1));

        Vector2::new(x, y)
    })
}

fn any_vector3_norm_squared_i32() -> impl Strategy<Value = Vector3<i32>> {
    any::<(i32, i32, i32)>().prop_map(|(_x, _y, _z)| {
        let min_value = 0;
        // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
        let max_square_root = 46340;
        let max_value = max_square_root / 3;
        let x = min_value + (_x % (max_value - min_value + 1));
        let y = min_value + (_y % (max_value - min_value + 1));
        let z = min_value + (_z % (max_value - min_value + 1));

        Vector3::new(x, y, z)
    })
}

fn any_vector4_norm_squared_i32() -> impl Strategy<Value = Vector4<i32>> {
    any::<(i32, i32, i32, i32)>().prop_map(|(_x, _y, _z, _w)| {
        let min_value = 0;
        // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
        let max_square_root = 46340;
        let max_value = max_square_root / 4;
        let x = min_value + (_x % (max_value - min_value + 1));
        let y = min_value + (_y % (max_value - min_value + 1));
        let z = min_value + (_z % (max_value - min_value + 1));
        let w = min_value + (_w % (max_value - min_value + 1));

        Vector4::new(x, y, z, w)
    })
}

        
/// A vector times a scalar zero should be a zero vector.
///
/// Given a vector `v`
/// ```text
/// v * 0 = 0
/// ```
/// Note that we deviate from the usual formalisms of vector algebra in that we 
/// allow the ability to multiply scalars from the right of a vector.
fn prop_vector_times_zero_equals_zero<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let zero = num_traits::zero();
    let zero_vector = Vector::zero();

    prop_assert_eq!(v * zero, zero_vector);

    Ok(())
}

/// A zero vector should act as the additive unit element of a vector space.
///
/// Given a vector `v`
/// ```text
/// v + 0 = v
/// ```
fn prop_vector_plus_zero_equals_vector<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let zero_vector = Vector::zero();

    prop_assert_eq!(v + zero_vector, v);

    Ok(())
}

/// A zero vector should act as the additive unit element of a vector space.
///
/// Given a vector `v`
/// ```text
/// 0 + v = v
/// ```
fn prop_zero_plus_vector_equals_vector<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let zero_vector = Vector::zero();

    prop_assert_eq!(zero_vector + v, v);

    Ok(())
}

/// Multiplying a vector by one should give the original vector.
///
/// Given a vector `v`
/// ```text
/// v * 1 = v
/// ```
/// Note that we deviate from the usual formalisms of vector algebra in that we 
/// allow the ability to multiply scalars to the right of a vector.
fn prop_vector_times_one_equals_vector<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let one = num_traits::one();

    prop_assert_eq!(v * one, v);

    Ok(())
}

/// Given vectors `v1` and `v2`, we should be able to use `v1` and 
/// `v2` interchangeably with their references `&v1` and `&v2` in 
/// arithmetic expressions involving vectors. 
///
/// Given vectors `v1` and `v2`, and their references `&v1` and 
/// `&v2`, they should satisfy
/// ```text
///  v1 +  v2 = &v1 +  v2
///  v1 +  v2 =  v1 + &v2
///  v1 +  v2 = &v1 + &v2
///  v1 + &v2 = &v1 +  v2
/// &v1 +  v2 =  v1 + &v2
/// &v1 +  v2 = &v1 + &v2
///  v1 + &v2 = &v1 + &v2
/// ```
fn prop_vector1_plus_vector2_equals_refvector1_plus_refvector2<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{   
    prop_assert_eq!( v1 +  v2, &v1 +  v2);
    prop_assert_eq!( v1 +  v2,  v1 + &v2);
    prop_assert_eq!( v1 +  v2, &v1 + &v2);
    prop_assert_eq!( v1 + &v2, &v1 +  v2);
    prop_assert_eq!(&v1 +  v2,  v1 + &v2);
    prop_assert_eq!(&v1 +  v2, &v1 + &v2);
    prop_assert_eq!( v1 + &v2, &v1 + &v2);

    Ok(())
}

/// Vector addition over floating point scalars should be commutative.
///
/// Given vectors `v1` and `v2`, we have
/// ```text
/// v1 + v2 = v2 + v1
/// ```
fn prop_vector_addition_commutative<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!(v1 + v2, v2 + v1);

    Ok(())
}

/// Vector addition over integer scalars should be associative.
///
/// Given three vectors `v1`, `v2`, and `v3`, we have
/// ```text
/// (v1 + v2) + v3 = v1 + (v2 + v3)
/// ```
fn prop_vector_addition_associative<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>, v3: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!((v1 + v2) + v3, v1 + (v2 + v3));

    Ok(())
}

/// The zero vector over vectors of floating point scalars should act as an 
/// additive unit. 
///
/// Given a vector `v`, we have
/// ```text
/// v - 0 = v
/// ```
fn prop_vector_minus_zero_equals_vector<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let zero_vector = Vector::zero();

    prop_assert_eq!(v - zero_vector, v);

    Ok(())
}

/// Every vector of floating point scalars should have an additive inverse.
///
/// Given a vector `v`, there is a vector `-v` such that
/// ```text
/// v - v = v + (-v) = (-v) + v = 0
/// ```
fn prop_vector_minus_vector_equals_zero<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let zero_vector = Vector::zero();

    prop_assert_eq!(v - v, zero_vector);

    Ok(())
}

/// Given vectors `v1` and `v2`, we should be able to use `v1` and `v2` 
/// interchangeably with their references `&v1` and `&v2` in arithmetic 
/// expressions involving vectors. 
///
/// Given vectors `v1` and `v2`, and their references `&v1` and `&v2`, 
/// they should satisfy
/// ```text
///  v1 -  v2 = &v1 -  v2
///  v1 -  v2 =  v1 - &v2
///  v1 -  v2 = &v1 - &v2
///  v1 - &v2 = &v1 -  v2
/// &v1 -  v2 =  v1 - &v2
/// &v1 -  v2 = &v1 - &v2
///  v1 - &v2 = &v1 - &v2
/// ```
fn prop_vector1_minus_vector2_equals_refvector1_minus_refvector2<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!( v1 +  v2, &v1 +  v2);
    prop_assert_eq!( v1 +  v2,  v1 + &v2);
    prop_assert_eq!( v1 +  v2, &v1 + &v2);
    prop_assert_eq!( v1 + &v2, &v1 +  v2);
    prop_assert_eq!(&v1 +  v2,  v1 + &v2);
    prop_assert_eq!(&v1 +  v2, &v1 + &v2);
    prop_assert_eq!( v1 + &v2, &v1 + &v2);

    Ok(())
}

/// A scalar `1` acts like a multiplicative identity element.
///
/// Given a vector `v`
/// ```text
/// 1 * v = v * 1 = v
/// ```
fn prop_one_times_vector_equals_vector<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let one = num_traits::one();

    prop_assert_eq!(v * one, v);

    Ok(())
}

/// Exact multiplication of two scalars and a vector should be compatible 
/// with multiplication of all scalars. In other words, scalar multiplication 
/// of two scalars with a vector should act associatively just like the 
/// multiplication of three scalars. 
///
/// Given scalars `a` and `b`, and a vector `v`, we have
/// ```text
/// v * (a * b) = (v * a) * b
/// ```
fn prop_scalar_multiplication_compatibility<S, const N: usize>(a: S, b: S, v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!((v * a) * b, v * (a * b));

    Ok(())
}

/// Scalar multiplication should distribute over vector addition.
///
/// Given a scalar `a` and vectors `v1` and `v2`
/// ```text
/// (v1 + v2) * a = v1 * a + v2 * a
/// ```
fn prop_scalar_vector_addition_right_distributive<S, const N: usize>(a: S, v1: Vector<S, N>, v2: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!((v1 + v2) * a,  v1 * a + v2 * a);

    Ok(())
}

/// Multiplication of a sum of scalars should distribute over a vector.
///
/// Given scalars `a` and `b` and a vector `v`, we have
/// ```text
/// v * (a + b) = v * a + v * b
/// ```
fn prop_vector_scalar_addition_left_distributive<S, const N: usize>(a: S, b: S, v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!(v * (a + b), v * a + v * b);

    Ok(())
}

/// Multiplication of two vectors by a scalar on the right should be 
/// right distributive.
///
/// Given vectors `v1` and `v2` and a scalar `a`
/// ```text
/// (v1 + v2) * a = v1 * a + v2 * a
/// ```
/// We deviate from the usual formalisms of vector algebra in that we 
/// allow the ability to multiply scalars from the left, or from the 
/// right of a vector.
fn prop_scalar_vector_addition_left_distributive<S, const N: usize>(a: S, v1: Vector<S, N>, v2: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!((v1 + v2) * a,  v1 * a + v2 * a);

    Ok(())
}

/// Multiplication of a vector on the right by the sum of two scalars should 
/// distribute over the two scalars.
/// 
/// Given a vector `v` and scalars `a` and `b`
/// ```text
/// v * (a + b) = v * a + v * b
/// ```
/// We deviate from the usual formalisms of vector algebra in that we 
/// allow the ability to multiply scalars from the left, or from the 
/// right of a vector.
fn prop_vector_scalar_addition_right_distributive<S, const N: usize>(a: S, b: S, v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!(v * (a + b), v * a + v * b);

    Ok(())
}

/// The dot product of vectors over integer scalars is commutative.
///
/// Given vectors `v1` and `v2`
/// ```text
/// dot(v1, v2) = dot(v2, v1)
/// ```
fn prop_vector_dot_product_commutative<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!(v1.dot(&v2), v2.dot(&v1));

    Ok(())
}

/// The dot product of vectors over integer scalars is right distributive.
///
/// Given vectors `v1`, `v2`, and `v3`
/// ```text
/// dot(v1, v2 + v3) = dot(v1, v2) + dot(v1, v3)
/// ```
fn prop_vector_dot_product_right_distributive<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>, v3: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let lhs = v1.dot(&(v2 + v3));
    let rhs = v1.dot(&v2) + v1.dot(&v3);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The dot product of vectors over integer scalars is left distributive.
///
/// Given vectors `v1`, `v2`, and `v3`
/// ```text
/// dot(v1 + v2, v3) = dot(v1, v3) + dot(v2, v3)
/// ```
fn prop_vector_dot_product_left_distributive<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>, v3: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let lhs = (v1 + v2).dot(&v3);
    let rhs = v1.dot(&v3) + v2.dot(&v3);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The dot product of vectors over integer scalars is commutative with scalars.
///
/// Given vectors `v1` and `v2`, and scalars `a` and `b`
/// ```text
/// dot(v1 * a, v2 * b) = dot(v1, v2) * (a * b)
/// ```
fn prop_vector_dot_product_times_scalars_commutative<S, const N: usize>(a: S, b: S, v1: Vector<S, N>, v2: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let lhs = (v1 * a).dot(&(v2 * b));
    let rhs = v1.dot(&v2) * (a * b);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The dot product of vectors over integer scalars is right bilinear.
///
/// Given vectors `v1`, `v2` and `v3`, and scalars `a` and `b`
/// ```text
/// dot(v1, v2 * a + v3 * b) = dot(v1, v2) * a + dot(v2, v3) * b
/// ```
fn prop_vector_dot_product_right_bilinear<S, const N: usize>(a: S, b: S, v1: Vector<S, N>, v2: Vector<S, N>, v3: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let lhs = v1.dot(&(v2 * a + v3 * b));
    let rhs = v1.dot(&v2) * a + v1.dot(&v3) * b;

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The dot product of vectors over integer scalars is left bilinear.
///
/// Given vectors `v1`, `v2` and `v3`, and scalars `a` and `b`
/// ```text
/// dot(v1 * a + v2 * b, v3) = dot(v1, v3) * a + dot(v2, v3) * b
/// ```
fn prop_vector_dot_product_left_bilinear<S, const N: usize>(a: S, b: S, v1: Vector<S, N>, v2: Vector<S, N>, v3: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let lhs = (v1 * a + v2 * b).dot(&v3);
    let rhs = v1.dot(&v3) * a + v2.dot(&v3) * b;

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The three-dimensional vector cross product of a vector with
/// itself is zero.
///
/// Given a vector `v`
/// ```text
/// v x v = 0
/// ```
fn prop_vector_cross_itself_is_zero<S>(v: Vector3<S>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let zero_vector = Vector3::zero();

    prop_assert_eq!(v.cross(&v), zero_vector);

    Ok(())
}

/// The three-dimensional cross product should commute with 
/// multiplication by a scalar.
///
/// Given vectors `v1` and `v2` and a scalar constant `c`
/// ```text
/// (v1 * c) x v2 = (v1 x v2) * c = v2 x (v2 * c)
/// ```
fn prop_vector_cross_product_multiplication_by_scalars<S>(c: S, v1: Vector3<S>, v2: Vector3<S>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!((v1 * c).cross(&v2), v1.cross(&v2) * c);
    prop_assert_eq!(v1.cross(&(v2 * c)), v1.cross(&v2) * c);

    Ok(())
}

/// The three-dimensional vector cross product is distributive.
///
/// Given vectors `v1`, `v2`, and `v3`
/// ```text
/// v1 x (v2 + v3) = v1 x v2 + v1 x v3
/// ```
fn prop_vector_cross_product_distribute<S>(v1: Vector3<S>, v2: Vector3<S>, v3: Vector3<S>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!(v1.cross(&(v2 + v3)), v1.cross(&v2) + v1.cross(&v3));

    Ok(())
}

/// The three-dimensional vector cross product satisfies the scalar 
/// triple product.
///
/// Given vectors `v1`, `v2`, and `v3`
/// ```text
/// v1 . (v2 x v3) = (v1 x v2) . v3
/// ```
fn prop_vector_cross_product_scalar_triple_product<S>(v1: Vector3<S>, v2: Vector3<S>, v3: Vector3<S>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!(v1.dot(&(v2.cross(&v3))), v1.cross(&v2).dot(&v3));

    Ok(())
}

/// The three-dimensional vector cross product is anti-commutative.
///
/// Given vectors `v1` and `v2`
/// ```text
/// v1 x v2 = - v2 x v1
/// ```
fn prop_vector_cross_product_anticommutative<S>(v1: Vector3<S>, v2: Vector3<S>) -> Result<(), TestCaseError> 
where
    S: SimdScalarSigned
{
    prop_assert_eq!(v1.cross(&v2), -v2.cross(&v1));

    Ok(())
}

/// The three-dimensional vector cross product satisfies the vector 
/// triple product.
///
/// Given vectors `v1`, `v2`, and `v3`
/// ```text
/// v1 x (v2 x v3) = (v1 . v3) x v2 - (v1 . v2) x v3
/// ```
fn prop_vector_cross_product_satisfies_vector_triple_product<S>(v1: Vector3<S>, v2: Vector3<S>, v3: Vector3<S>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let lhs = v1.cross(&v2.cross(&v3));
    let rhs = v2 * v1.dot(&v3) - v3 * v1.dot(&v2);
    
    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The squared **L2** norm of a vector is nonnegative.
///
/// Given a vector `v`
/// ```text
/// norm_squared(v) >= 0
/// ```
fn prop_norm_squared_nonnegative<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let zero = num_traits::zero();
    
    prop_assert!(v.norm_squared() >= zero);

    Ok(())
}

/// The squared **L2** norm function is point separating. In particular, 
/// if the squared distance between two vectors `v1` and `v2` is zero, then `v1 = v2`.
///
/// Given vectors `v1` and `v2`
/// ```text
/// norm_squared(v1 - v2) = 0 => v1 = v2 
/// ```
/// Equivalently, if `v1` is not equal to `v2`, then their squared distance is nonzero
/// ```text
/// v1 != v2 => norm_squared(v1 - v2) != 0
/// ```
/// For the sake of testability, we use the second form to test the norm
/// function.
fn prop_approx_norm_squared_point_separating<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>, input_tolerance: S, output_tolerance: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(v1, v2, epsilon = input_tolerance));
    prop_assert!(
        (v1 - v2).norm_squared() > output_tolerance,
        "\n|v - w|^2 = {:?}\n", 
        (v1 - v2).norm_squared()
    );

    Ok(())
}

/// The [`Vector::magnitude_squared`] function and the [`Vector::norm_squared`] 
/// function are synonyms. In particular, given a vector `v`
/// ```text
/// magnitude_squared(v) = norm_squared(v)
/// ```
/// where equality is exact.
fn prop_magnitude_squared_norm_squared_synonyms<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    prop_assert_eq!(v.magnitude_squared(), v.norm_squared());

    Ok(())
}

/// The squared **L2** norm function is point separating. In particular, 
/// if the squared distance between two vectors `v1` and `v2` is zero, then `v1 = v2`.
///
/// Given vectors `v1` and `v2`
/// ```text
/// norm_squared(v1 - v2) = 0 => v1 = v2 
/// ```
/// Equivalently, if `v1` is not equal to `v2`, then their squared distance is nonzero
/// ```text
/// v1 != v2 => norm_squared(v1 - v2) != 0
/// ```
/// For the sake of testability, we use the second form to test the norm
/// function.
fn prop_norm_squared_point_separating<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalar
{
    let zero = num_traits::zero();

    prop_assume!(v1 != v2);
    prop_assert_ne!(
        (v1 - v2).norm_squared(), zero,
        "\n|v - w|^2 = {}\n", 
        (v1 - v2).norm_squared()
    );

    Ok(())
}

/// The **L2** norm of a vector is nonnegative.
///
/// Given a vector `v`
/// ```text
/// norm(v) >= 0
/// ```
fn prop_norm_nonnegative<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let zero = num_traits::zero();
    
    prop_assert!(v.norm() >= zero);

    Ok(())
}

/// The **L2** norm function is point separating. In particular, if the 
/// distance between two vectors `v1` and `v2` is zero, then `v1 = v2`.
///
/// Given vectors `v1` and `v2`
/// ```text
/// norm(v1 - v2) = 0 => v1 = v2 
/// ```
/// Equivalently, if `v1` is not equal to `v2`, then their distance is nonzero
/// ```text
/// v1 != v2 => norm(v1 - v2) != 0
/// ```
/// For the sake of testability, we use the second form to test the norm
/// function.
fn prop_approx_norm_point_separating<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>, tolerance: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(v1, v2, epsilon = tolerance));
    prop_assert!(
        (v1 - v2).norm() > tolerance,
        "\n|v - w| = {:?}\n",
        (v1 - v2).norm()
    );

    Ok(())
}

/// The **L1** norm of a vector is nonnegative.
///
/// Given a vector `v`
/// ```text
/// l1_norm(v) >= 0
/// ```
fn prop_l1_norm_nonnegative<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalarSigned
{
    let zero = num_traits::zero();
    
    prop_assert!(v.l1_norm() >= zero);

    Ok(())
}

/// The **L1** norm function is point separating. In particular, if the 
/// distance between two vectors `v1` and `v2` is zero, then `v1 = v2`.
///
/// Given vectors `v1` and `v2`
/// ```text
/// l1_norm(v1 - v2) = 0 => v1 = v2 
/// ```
/// Equivalently, if `v1` is not equal to `v2`, then their distance is nonzero
/// ```text
/// v1 != v2 => l1_norm(v1 - v2) != 0
/// ```
/// For the sake of testability, we use the second form to test the norm
/// function.
fn prop_approx_l1_norm_point_separating<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>, tolerance: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(v1, v2, epsilon = tolerance));
    prop_assert!(
        (v1 - v2).l1_norm() > tolerance,
        "\nl1_norm(v - w) = {:?}\n", 
        (v1 - v2).l1_norm()
    );

    Ok(())
}

/// The **L1** norm function is point separating. In particular, if the 
/// distance between two vectors `v1` and `v2` is zero, then `v1 = v2`.
///
/// Given vectors `v1` and `v2`
/// ```text
/// l1_norm(v1 - v2) = 0 => v1 = v2 
/// ```
/// Equivalently, if `v1` is not equal to `v2`, then their distance is nonzero
/// ```text
/// v1 != v2 => l1_norm(v1 - v2) != 0
/// ```
/// For the sake of testability, we use the second form to test the norm
/// function.
fn prop_l1_norm_point_separating<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalarSigned
{
    let zero = num_traits::zero();

    prop_assume!(v1 != v2);
    prop_assert_ne!(
        (v1 - v2).l1_norm(), zero,
        "\nl1_norm(v - w) = {:?}\n", 
        (v1 - v2).l1_norm()
    );

    Ok(())
}

/// The **Lp** norm of a vector is nonnegative.
///
/// Given a vector `v`
/// ```text
/// lp_norm(v) >= 0
/// ```
fn prop_lp_norm_nonnegative<S, const N: usize>(v: Vector<S, N>, p: u32) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let zero = num_traits::zero();
    
    prop_assert!(v.lp_norm(p) >= zero);

    Ok(())
}

/// The **Lp** norm function is point separating. In particular, if the 
/// distance between two vectors `v1` and `v2` is zero, then `v1 = v2`.
///
/// Given vectors `v1` and `v2`
/// ```text
/// lp_norm(v1 - v2) = 0 => v1 = v2 
/// ```
/// Equivalently, if `v1` is not equal to `v2`, then their distance is nonzero
/// ```text
/// v1 != v2 => lp_norm(v1 - v2) != 0
/// ```
/// For the sake of testability, we use the second form to test the norm
/// function.
fn prop_approx_lp_norm_point_separating<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>, p: u32, tolerance: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(v1, v2, epsilon = tolerance));
    prop_assert!(
        (v1 - v2).lp_norm(p) > tolerance,
        "\nlp_norm(v - w, p) = {}\n",
        (v1 - v2).lp_norm(p)
    );

    Ok(())
}

/// The **L-infinity** norm of a vector is nonnegative.
///
/// Given a vector `v`
/// ```text
/// linf_norm(v) >= 0
/// ```
fn prop_linf_norm_nonnegative<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let zero = num_traits::zero();
    
    prop_assert!(v.linf_norm() >= zero);

    Ok(())
}

/// The **L-infinity** norm function is point separating. In particular, if the 
/// distance between two vectors `v1` and `v2` is zero, then `v1 = v2`.
///
/// Given vectors `v1` and `v2`
/// ```text
/// linf_norm(v1 - v2) = 0 => v1 = v2 
/// ```
/// Equivalently, if `v1` is not equal to `v2`, then their distance is nonzero
/// ```text
/// v1 != v2 => linf_norm(v1 - v2) != 0
/// ```
/// For the sake of testability, we use the second form to test the norm
/// function.
fn prop_approx_linf_norm_point_separating<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>, tolerance: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(v1, v2, epsilon = tolerance));
    prop_assert!(
        (v1 - v2).linf_norm() > tolerance,
        "\nlinf_norm(v - w) = {}\n", 
        (v1 - v2).linf_norm()
    );

    Ok(())
}

/// The **L-infinity** norm function is point separating. In particular, if the 
/// distance between two vectors `v1` and `v2` is zero, then `v1 = v2`.
///
/// Given vectors `v1` and `v2`
/// ```text
/// linf_norm(v1 - v2) = 0 => v1 = v2 
/// ```
/// Equivalently, if `v1` is not equal to `v2`, then their distance is nonzero
/// ```text
/// v1 != v2 => linf_norm(v1 - v2) != 0
/// ```
/// For the sake of testability, we use the second form to test the norm
/// function.
fn prop_linf_norm_point_separating<S, const N: usize>(v1: Vector<S, N>, v2: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalarSigned + SimdScalarOrd
{
    let zero = num_traits::zero();

    prop_assume!(v1 != v2);
    prop_assert_ne!(
        (v1 - v2).linf_norm(), zero,
        "\nlinf_norm(v - w) = {}\n", 
        (v1 - v2).linf_norm()
    );

    Ok(())
}

/// The [`Vector::magnitude`] function and the [`Vector::norm`] function 
/// are synonyms. In particular, given a vector `v`
/// ```text
/// magnitude(v) = norm(v)
/// ```
/// where equality is exact.
fn prop_magnitude_norm_synonyms<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    prop_assert_eq!(v.magnitude(), v.norm());

    Ok(())
}

/// The [`Vector::l2_norm`] function and the [`Vector::norm`] function
/// are synonyms. In particular, given a vector `v`
/// ```text
/// l2_norm(v) = norm(v)
/// ```
/// where equality is exact.
fn prop_l2_norm_norm_synonyms<S, const N: usize>(v: Vector<S, N>) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    prop_assert_eq!(v.l2_norm(), v.norm());

    Ok(())
}


macro_rules! exact_arithmetic_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_vector_times_zero_equals_zero(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_vector_times_zero_equals_zero(v)?
            }

            #[test]
            fn prop_vector_plus_zero_equals_vector(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_vector_plus_zero_equals_vector(v)?
            }

            #[test]
            fn prop_zero_plus_vector_equals_vector(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_zero_plus_vector_equals_vector(v)?
            }

            #[test]
            fn prop_vector_times_one_equals_vector(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_vector_times_one_equals_vector(v)?
            }
        }
    }
    }
}

exact_arithmetic_props!(vector1_f64_arithmetic_props, Vector1, f64, any_vector1);
exact_arithmetic_props!(vector2_f64_arithmetic_props, Vector2, f64, any_vector2);
exact_arithmetic_props!(vector3_f64_arithmetic_props, Vector3, f64, any_vector3);
exact_arithmetic_props!(vector4_f64_arithmetic_props, Vector4, f64, any_vector4);

exact_arithmetic_props!(vector1_i32_arithmetic_props, Vector1, i32, any_vector1);
exact_arithmetic_props!(vector2_i32_arithmetic_props, Vector2, i32, any_vector2);
exact_arithmetic_props!(vector3_i32_arithmetic_props, Vector3, i32, any_vector3);
exact_arithmetic_props!(vector4_i32_arithmetic_props, Vector4, i32, any_vector4);


macro_rules! approx_add_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_vector_plus_zero_equals_vector(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_vector_plus_zero_equals_vector(v)?
            }

            #[test]
            fn prop_zero_plus_vector_equals_vector(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_zero_plus_vector_equals_vector(v)?
            }

            #[test]
            fn prop_vector1_plus_vector2_equals_refvector1_plus_refvector2(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_vector1_plus_vector2_equals_refvector1_plus_refvector2(v1, v2)?
            }

            #[test]
            fn prop_vector_addition_commutative(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_vector_addition_commutative(v1, v2)?
            }
        }
    }
    }
}

approx_add_props!(vector1_f64_add_props, Vector1, f64, any_vector1, 1e-8);
approx_add_props!(vector2_f64_add_props, Vector2, f64, any_vector2, 1e-8);
approx_add_props!(vector3_f64_add_props, Vector3, f64, any_vector3, 1e-8);
approx_add_props!(vector4_f64_add_props, Vector4, f64, any_vector4, 1e-8);


macro_rules! exact_add_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_vector_plus_zero_equals_vector(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_vector_plus_zero_equals_vector(v)?
            }

            #[test]
            fn prop_zero_plus_vector_equals_vector(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_zero_plus_vector_equals_vector(v)?
            }

            #[test]
            fn prop_vector1_plus_vector2_equals_refvector1_plus_refvector2(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_vector1_plus_vector2_equals_refvector1_plus_refvector2(v1, v2)?
            }

            #[test]
            fn prop_vector_addition_commutative(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_vector_addition_commutative(v1, v2)?
            }

            #[test]
            fn prop_vector_addition_associative(v1 in super::$Generator(), v2 in super::$Generator(), v3 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                let v3: super::$VectorN<$ScalarType> = v3;
                super::prop_vector_addition_associative(v1, v2, v3)?
            }
        }
    }
    }
}

exact_add_props!(vector1_i32_add_props, Vector1, i32, any_vector1);
exact_add_props!(vector2_i32_add_props, Vector2, i32, any_vector2);
exact_add_props!(vector3_i32_add_props, Vector3, i32, any_vector3);
exact_add_props!(vector4_i32_add_props, Vector4, i32, any_vector4);


macro_rules! approx_sub_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_vector_minus_zero_equals_vector(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_vector_minus_zero_equals_vector(v)?
            }

            #[test]
            fn prop_vector_minus_vector_equals_zero(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_vector_minus_vector_equals_zero(v)?
            }

            #[test]
            fn prop_vector1_minus_vector2_equals_refvector1_minus_refvector2(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_vector1_minus_vector2_equals_refvector1_minus_refvector2(v1, v2)?
            }
        }
    }
    }
}

approx_sub_props!(vector1_f64_sub_props, Vector1, f64, any_vector1);
approx_sub_props!(vector2_f64_sub_props, Vector2, f64, any_vector2);
approx_sub_props!(vector3_f64_sub_props, Vector3, f64, any_vector3);
approx_sub_props!(vector4_f64_sub_props, Vector4, f64, any_vector4);


macro_rules! exact_sub_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_vector_minus_zero_equals_vector(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_vector_minus_zero_equals_vector(v)?
            }

            #[test]
            fn prop_vector_minus_vector_equals_zero(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_vector_minus_vector_equals_zero(v)?
            }

            #[test]
            fn prop_vector1_minus_vector2_equals_refvector1_minus_refvector2(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_vector1_minus_vector2_equals_refvector1_minus_refvector2(v1, v2)?
            }
        }
    }
    }
}

exact_sub_props!(vector1_i32_sub_props, Vector1, i32, any_vector1);
exact_sub_props!(vector2_i32_sub_props, Vector2, i32, any_vector2);
exact_sub_props!(vector3_i32_sub_props, Vector3, i32, any_vector3);
exact_sub_props!(vector4_i32_sub_props, Vector4, i32, any_vector4);


macro_rules! approx_mul_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_one_times_vector_equals_vector(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_one_times_vector_equals_vector(v)?
            }
        }
    }
    }
}

approx_mul_props!(vector1_f64_mul_props, Vector1, f64, any_vector1, any_scalar, 1e-8);
approx_mul_props!(vector2_f64_mul_props, Vector2, f64, any_vector2, any_scalar, 1e-8);
approx_mul_props!(vector3_f64_mul_props, Vector3, f64, any_vector3, any_scalar, 1e-8);
approx_mul_props!(vector4_f64_mul_props, Vector4, f64, any_vector4, any_scalar, 1e-8);


macro_rules! exact_mul_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_scalar_multiplication_compatibility(a in super::$ScalarGen(), b in super::$ScalarGen(), v in super::$Generator()) {
                let a: $ScalarType = a;
                let b: $ScalarType = b;
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_scalar_multiplication_compatibility(a, b, v)?
            }

            #[test]
            fn prop_one_times_vector_equals_vector(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_one_times_vector_equals_vector(v)?
            }
        }
    }
    }
}

exact_mul_props!(vector1_i32_mul_props, Vector1, i32, any_vector1, any_scalar);
exact_mul_props!(vector2_i32_mul_props, Vector2, i32, any_vector2, any_scalar);
exact_mul_props!(vector3_i32_mul_props, Vector3, i32, any_vector3, any_scalar);
exact_mul_props!(vector4_i32_mul_props, Vector4, i32, any_vector4, any_scalar);


macro_rules! exact_distributive_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_scalar_vector_addition_right_distributive(a in super::$ScalarGen(), v1 in super::$Generator(), v2 in super::$Generator()) {
                let a: $ScalarType = a;
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_scalar_vector_addition_right_distributive(a, v1, v2)?
            }

            #[test]
            fn prop_vector_scalar_addition_left_distributive(a in super::$ScalarGen(), b in super::$ScalarGen(), v in super::$Generator()) {
                let a: $ScalarType = a;
                let b: $ScalarType = b;
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_vector_scalar_addition_left_distributive(a, b, v)?
            }

            #[test]
            fn prop_scalar_vector_addition_left_distributive(a in super::$ScalarGen(), v1 in super::$Generator(), v2 in super::$Generator()) {
                let a: $ScalarType = a;
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_scalar_vector_addition_left_distributive(a, v1, v2)?
            }

            #[test]
            fn prop_vector_scalar_addition_right_distributive(a in super::$ScalarGen(), b in super::$ScalarGen(), v in super::$Generator()) {
                let a: $ScalarType = a;
                let b: $ScalarType = b;
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_vector_scalar_addition_right_distributive(a, b, v)?
            }
        }
    }
    }    
}

exact_distributive_props!(vector1_i32_distributive_props, Vector1, i32, any_vector1, any_scalar);
exact_distributive_props!(vector2_i32_distributive_props, Vector2, i32, any_vector2, any_scalar);
exact_distributive_props!(vector3_i32_distributive_props, Vector3, i32, any_vector3, any_scalar);
exact_distributive_props!(vector4_i32_distributive_props, Vector4, i32, any_vector4, any_scalar);


macro_rules! exact_dot_product_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_vector_dot_product_commutative(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_vector_dot_product_commutative(v1, v2)?
            }

            #[test]
            fn prop_vector_dot_product_right_distributive(v1 in super::$Generator(), v2 in super::$Generator(), v3 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                let v3: super::$VectorN<$ScalarType> = v3;
                super::prop_vector_dot_product_right_distributive(v1, v2, v3)?
            }

            #[test]
            fn prop_vector_dot_product_left_distributive(v1 in super::$Generator(), v2 in super::$Generator(), v3 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                let v3: super::$VectorN<$ScalarType> = v3;
                super::prop_vector_dot_product_left_distributive(v1, v2, v3)?
            }

            #[test]
            fn prop_vector_dot_product_times_scalars_commutative(a in super::$ScalarGen(), b in super::$ScalarGen(), v1 in super::$Generator(), v2 in super::$Generator()) {
                let a: $ScalarType = a;
                let b: $ScalarType = b;
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_vector_dot_product_times_scalars_commutative(a, b, v1, v2)?
            }

            #[test]
            fn prop_vector_dot_product_right_bilinear(a in super::$ScalarGen(), b in super::$ScalarGen(), v1 in super::$Generator(), v2 in super::$Generator(), v3 in super::$Generator()) {
                let a: $ScalarType = a;
                let b: $ScalarType = b;
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                let v3: super::$VectorN<$ScalarType> = v3;
                super::prop_vector_dot_product_right_bilinear(a, b, v1, v2, v3)?
            }

            #[test]
            fn prop_vector_dot_product_left_bilinear(a in super::$ScalarGen(), b in super::$ScalarGen(), v1 in super::$Generator(), v2 in super::$Generator(), v3 in super::$Generator()) {
                let a: $ScalarType = a;
                let b: $ScalarType = b;
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                let v3: super::$VectorN<$ScalarType> = v3;
                super::prop_vector_dot_product_left_bilinear(a, b, v1, v2, v3)?
            }
        }
    }
    }
}

exact_dot_product_props!(vector1_i32_dot_product_props, Vector1, i32, any_vector1, any_scalar);
exact_dot_product_props!(vector2_i32_dot_product_props, Vector2, i32, any_vector2, any_scalar);
exact_dot_product_props!(vector3_i32_dot_product_props, Vector3, i32, any_vector3, any_scalar);
exact_dot_product_props!(vector4_i32_dot_product_props, Vector4, i32, any_vector4, any_scalar);


macro_rules! exact_cross_product_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_vector_cross_itself_is_zero(v in super::$Generator()) {
                let v: super::Vector3<$ScalarType> = v;
                super::prop_vector_cross_itself_is_zero(v)?
            }

            #[test]
            fn prop_vector_cross_product_multiplication_by_scalars(c in super::$ScalarGen(), v1 in super::$Generator(), v2 in super::$Generator()) {
                let c: $ScalarType = c;
                let v1: super::Vector3<$ScalarType> = v1;
                let v2: super::Vector3<$ScalarType> = v2;
                super::prop_vector_cross_product_multiplication_by_scalars(c, v1, v2)?
            }

            #[test]
            fn prop_vector_cross_product_distribute(v1 in super::$Generator(), v2 in super::$Generator(), v3 in super::$Generator()) {
                let v1: super::Vector3<$ScalarType> = v1;
                let v2: super::Vector3<$ScalarType> = v2;
                let v3: super::Vector3<$ScalarType> = v3;
                super::prop_vector_cross_product_distribute(v1, v2, v3)?
            }

            #[test]
            fn prop_vector_cross_product_scalar_triple_product(v1 in super::$Generator(), v2 in super::$Generator(), v3 in super::$Generator()) {
                let v1: super::Vector3<$ScalarType> = v1;
                let v2: super::Vector3<$ScalarType> = v2;
                let v3: super::Vector3<$ScalarType> = v3;
                super::prop_vector_cross_product_scalar_triple_product(v1, v2, v3)?
            }

            #[test]
            fn prop_vector_cross_product_anticommutative(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::Vector3<$ScalarType> = v1;
                let v2: super::Vector3<$ScalarType> = v2;
                super::prop_vector_cross_product_anticommutative(v1, v2)?
            }

            #[test]
            fn prop_vector_cross_product_satisfies_vector_triple_product(v1 in super::$Generator(), v2 in super::$Generator(), v3 in super::$Generator()) {
                let v1: super::Vector3<$ScalarType> = v1;
                let v2: super::Vector3<$ScalarType> = v2;
                let v3: super::Vector3<$ScalarType> = v3;
                super::prop_vector_cross_product_satisfies_vector_triple_product(v1, v2, v3)?
            }
        }
    }
    }
}

exact_cross_product_props!(vector3_i32_cross_product_props, i32, any_vector3, any_scalar);


macro_rules! approx_norm_squared_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $input_tolerance:expr, $output_tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_norm_squared_nonnegative(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_norm_squared_nonnegative(v)?
            }

            #[test]
            fn prop_approx_norm_squared_point_separating(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_approx_norm_squared_point_separating(v1, v2, $input_tolerance, $output_tolerance)?
            }
        }
    }
    }
}

approx_norm_squared_props!(vector1_f64_norm_squared_props, Vector1, f64, any_vector1_norm_squared_f64, 1e-10, 1e-20);
approx_norm_squared_props!(vector2_f64_norm_squared_props, Vector2, f64, any_vector2_norm_squared_f64, 1e-10, 1e-20);
approx_norm_squared_props!(vector3_f64_norm_squared_props, Vector3, f64, any_vector3_norm_squared_f64, 1e-10, 1e-20);
approx_norm_squared_props!(vector4_f64_norm_squared_props, Vector4, f64, any_vector4_norm_squared_f64, 1e-10, 1e-20);


macro_rules! approx_norm_squared_synonym_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_magnitude_squared_norm_squared_synonyms(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_magnitude_squared_norm_squared_synonyms(v)?
            }
        }
    }
    }
}

approx_norm_squared_synonym_props!(vector1_f64_norm_squared_synonym_props, Vector1, f64, any_vector1);
approx_norm_squared_synonym_props!(vector2_f64_norm_squared_synonym_props, Vector2, f64, any_vector2);
approx_norm_squared_synonym_props!(vector3_f64_norm_squared_synonym_props, Vector3, f64, any_vector3);
approx_norm_squared_synonym_props!(vector4_f64_norm_squared_synonym_props, Vector4, f64, any_vector4);


macro_rules! exact_norm_squared_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_norm_squared_nonnegative(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_norm_squared_nonnegative(v)?
            }

            #[test]
            fn prop_norm_squared_point_separating(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_norm_squared_point_separating(v1, v2)?
            }
        }
    }
    }
}

exact_norm_squared_props!(vector1_i32_norm_squared_props, Vector1, i32, any_vector1_norm_squared_i32, any_scalar);
exact_norm_squared_props!(vector2_i32_norm_squared_props, Vector2, i32, any_vector2_norm_squared_i32, any_scalar);
exact_norm_squared_props!(vector3_i32_norm_squared_props, Vector3, i32, any_vector3_norm_squared_i32, any_scalar);
exact_norm_squared_props!(vector4_i32_norm_squared_props, Vector4, i32, any_vector4_norm_squared_i32, any_scalar);


macro_rules! exact_norm_squared_synonym_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_magnitude_squared_norm_squared_synonyms(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_magnitude_squared_norm_squared_synonyms(v)?
            }
        }
    }
    }
}

exact_norm_squared_synonym_props!(vector1_i32_norm_squared_synonym_props, Vector1, i32, any_vector1);
exact_norm_squared_synonym_props!(vector2_i32_norm_squared_synonym_props, Vector2, i32, any_vector2);
exact_norm_squared_synonym_props!(vector3_i32_norm_squared_synonym_props, Vector3, i32, any_vector3);
exact_norm_squared_synonym_props!(vector4_i32_norm_squared_synonym_props, Vector4, i32, any_vector4);


macro_rules! norm_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_norm_nonnegative(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_norm_nonnegative(v)?
            }

            #[test]
            fn prop_approx_norm_point_separating(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_approx_norm_point_separating(v1, v2, $tolerance)?
            }
        }
    }
    }
}

norm_props!(vector1_f64_norm_props, Vector1, f64, any_vector1, any_scalar, 1e-8);
norm_props!(vector2_f64_norm_props, Vector2, f64, any_vector2, any_scalar, 1e-8);
norm_props!(vector3_f64_norm_props, Vector3, f64, any_vector3, any_scalar, 1e-8);
norm_props!(vector4_f64_norm_props, Vector4, f64, any_vector4, any_scalar, 1e-8);


macro_rules! approx_l1_norm_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_l1_norm_nonnegative(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_l1_norm_nonnegative(v)?
            }

            #[test]
            fn prop_approx_l1_norm_point_separating(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_approx_l1_norm_point_separating(v1, v2, $tolerance)?
            }
        }
    }
    }
}

approx_l1_norm_props!(vector1_f64_l1_norm_props, Vector1, f64, any_vector1, any_scalar, 1e-8);
approx_l1_norm_props!(vector2_f64_l1_norm_props, Vector2, f64, any_vector2, any_scalar, 1e-8);
approx_l1_norm_props!(vector3_f64_l1_norm_props, Vector3, f64, any_vector3, any_scalar, 1e-8);
approx_l1_norm_props!(vector4_f64_l1_norm_props, Vector4, f64, any_vector4, any_scalar, 1e-8);


macro_rules! exact_l1_norm_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_l1_norm_nonnegative(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_l1_norm_nonnegative(v)?
            }

            #[test]
            fn prop_l1_norm_point_separating(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_l1_norm_point_separating(v1, v2)?
            }
        }
    }
    }
}

exact_l1_norm_props!(vector1_i32_l1_norm_props, Vector1, i32, any_vector1, any_scalar);
exact_l1_norm_props!(vector2_i32_l1_norm_props, Vector2, i32, any_vector2, any_scalar);
exact_l1_norm_props!(vector3_i32_l1_norm_props, Vector3, i32, any_vector3, any_scalar);
exact_l1_norm_props!(vector4_i32_l1_norm_props, Vector4, i32, any_vector4, any_scalar);


macro_rules! lp_norm_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $DegreeGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_lp_norm_nonnegative(v in super::$Generator(), p in super::$DegreeGen()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_lp_norm_nonnegative(v, p)?
            }

            #[test]
            fn prop_approx_lp_norm_point_separating(v1 in super::$Generator(), v2 in super::$Generator(), p in super::$DegreeGen()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_approx_lp_norm_point_separating(v1, v2, p, $tolerance)?
            }
        }
    }
    }
}

lp_norm_props!(vector1_f64_lp_norm_props, Vector1, f64, any_vector1, any_scalar, any_u32, 1e-6);
lp_norm_props!(vector2_f64_lp_norm_props, Vector2, f64, any_vector2, any_scalar, any_u32, 1e-6);
lp_norm_props!(vector3_f64_lp_norm_props, Vector3, f64, any_vector3, any_scalar, any_u32, 1e-6);
lp_norm_props!(vector4_f64_lp_norm_props, Vector4, f64, any_vector4, any_scalar, any_u32, 1e-6);


macro_rules! approx_linf_norm_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_linf_norm_nonnegative(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_linf_norm_nonnegative(v)?
            }

            #[test]
            fn prop_approx_linf_norm_point_separating(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_approx_linf_norm_point_separating(v1, v2, $tolerance)?
            }
        }
    }
    }
}

approx_linf_norm_props!(vector1_f64_linf_norm_props, Vector1, f64, any_vector1, any_scalar, 1e-8);
approx_linf_norm_props!(vector2_f64_linf_norm_props, Vector2, f64, any_vector2, any_scalar, 1e-8);
approx_linf_norm_props!(vector3_f64_linf_norm_props, Vector3, f64, any_vector3, any_scalar, 1e-8);
approx_linf_norm_props!(vector4_f64_linf_norm_props, Vector4, f64, any_vector4, any_scalar, 1e-8);


macro_rules! exact_linf_norm_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $ScalarGen:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_linf_norm_nonnegative(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_linf_norm_nonnegative(v)?
            }

            #[test]
            fn prop_linf_norm_point_separating(v1 in super::$Generator(), v2 in super::$Generator()) {
                let v1: super::$VectorN<$ScalarType> = v1;
                let v2: super::$VectorN<$ScalarType> = v2;
                super::prop_linf_norm_point_separating(v1, v2)?
            }
        }
    }
    }
}

exact_linf_norm_props!(vector1_i32_linf_norm_props, Vector1, i32, any_vector1, any_scalar);
exact_linf_norm_props!(vector2_i32_linf_norm_props, Vector2, i32, any_vector2, any_scalar);
exact_linf_norm_props!(vector3_i32_linf_norm_props, Vector3, i32, any_vector3, any_scalar);
exact_linf_norm_props!(vector4_i32_linf_norm_props, Vector4, i32, any_vector4, any_scalar);


macro_rules! norm_synonym_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_magnitude_norm_synonyms(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_magnitude_norm_synonyms(v)?
            }

            #[test]
            fn prop_l2_norm_norm_synonyms(v in super::$Generator()) {
                let v: super::$VectorN<$ScalarType> = v;
                super::prop_l2_norm_norm_synonyms(v)?
            }
        }
    }
    }
}

norm_synonym_props!(vector1_f64_norm_synonym_props, Vector1, f64, any_vector1);
norm_synonym_props!(vector2_f64_norm_synonym_props, Vector2, f64, any_vector2);
norm_synonym_props!(vector3_f64_norm_synonym_props, Vector3, f64, any_vector3);
norm_synonym_props!(vector4_f64_norm_synonym_props, Vector4, f64, any_vector4);

