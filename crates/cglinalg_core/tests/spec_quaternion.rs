extern crate cglinalg_core;
extern crate num_traits;
extern crate proptest;


use proptest::prelude::*;
use cglinalg_core::{
    Quaternion, 
    SimdScalar,
    SimdScalarSigned,
    SimdScalarFloat,
};
use approx::{
    relative_eq,
    relative_ne,
};


fn strategy_scalar_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = S>
where
    S: SimdScalarSigned + Arbitrary
{
    fn rescale<S: SimdScalarSigned>(value: S, min_value: S, max_value: S) -> S {
        min_value + (value % (max_value - min_value))
    }

    any::<S>().prop_map(move |value| {
        let sign_value = value.signum();
        let abs_value = value.abs();
        
        sign_value * rescale(abs_value, min_value, max_value)
    })
    .no_shrink()
}

fn strategy_scalar_f64_any() -> impl Strategy<Value = f64> {
    let min_value = f64::sqrt(f64::EPSILON);
    let max_value = f64::sqrt(f64::MAX) / f64::sqrt(2_f64);

    strategy_scalar_signed_from_abs_range(min_value, max_value)
}

fn strategy_scalar_i32_any() -> impl Strategy<Value = i32> {
    let min_value = 0_i32;
    // let max_value = f64::floor(f64::sqrt(i32::MAX as f64 / 2_f64)) as i32;
    let max_value = 46340_i32;

    strategy_scalar_signed_from_abs_range(min_value, max_value)
}

fn strategy_quaternion_any<S>() -> impl Strategy<Value = Quaternion<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S, S, S)>().prop_map(|(s, x, y, z)| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let quaternion = Quaternion::new(s, x, y, z);

        quaternion % modulus
    })
    .no_shrink()
}

fn strategy_quaternion_f64_norm_squared() -> impl Strategy<Value = Quaternion<f64>> {
    use cglinalg_core::{
        Radians,
        Vector3,
        Unit,
    };

    fn rescale(value: f64, min_value: f64, max_value: f64) -> f64 {
        min_value + (value % (max_value - min_value))
    }

    any::<(f64, f64, f64, f64, f64)>().prop_map(|(_scale, _angle, _x, _y, _z)| {
        let min_scale = f64::sqrt(f64::EPSILON);
        let max_scale = f64::sqrt(f64::MAX);
        let scale = rescale(_scale, min_scale, max_scale);
        let angle = Radians(_angle % core::f64::consts::FRAC_PI_2);
        let unnormalized_axis = {
            let min_value = f64::sqrt(f64::EPSILON);
            let max_value = f64::sqrt(f64::MAX) / f64::sqrt(3_f64);
            let x = rescale(_x, min_value, max_value);
            let y = rescale(_y, min_value, max_value);
            let z = rescale(_z, min_value, max_value);

            Vector3::new(x, y, z)
        };
        let axis = Unit::from_value(unnormalized_axis);
        
        Quaternion::from_polar_decomposition(scale, angle, &axis)
    })
    .no_shrink()
}

fn strategy_quaternion_i32_norm_squared() -> impl Strategy<Value = Quaternion<i32>> {
    any::<(i32, i32, i32, i32)>().prop_map(|(_s, _x, _y, _z)| {
        let min_value = 0;
        // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
        let max_square_root = 46340;
        let max_value = max_square_root / 4;
        let qs = min_value + (_s % (max_value - min_value + 1));
        let qx = min_value + (_x % (max_value - min_value + 1));
        let qy = min_value + (_y % (max_value - min_value + 1));
        let qz = min_value + (_z % (max_value - min_value + 1));
        
        Quaternion::new(qs, qx, qy, qz)
    })
    .no_shrink()
}

fn strategy_quaternion_squared_any<S>() -> impl Strategy<Value = Quaternion<S>>
where
    S: SimdScalarFloat + Arbitrary
{
    any::<(S, S, S, S)>().prop_map(|(s, x, y, z)| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let quaternion = Quaternion::new(s.abs(), x.abs(), y.abs(), z.abs());

        quaternion % modulus
    })
    .no_shrink()
}


/// A scalar `0` times a quaternion should be a zero quaternion.
///
/// Given a quaternion `q`, it satisfies
/// ```text
/// 0 * q == 0.
/// ```
fn prop_zero_times_quaternion_equals_zero<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let zero_quat = Quaternion::zero();

    prop_assert_eq!(zero_quat * q, zero_quat);

    Ok(())
}

        
/// A scalar `0` times a quaternion should be zero.
///
/// Given a quaternion `q`, it satisfies
/// ```text
/// q * 0 == 0
/// ```
fn prop_quaternion_times_zero_equals_zero<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let zero = S::zero();
    let zero_quat = Quaternion::zero();

    prop_assert_eq!(q * zero, zero_quat);

    Ok(())
}

/// A zero quaternion should act as the additive unit element of a set 
/// of quaternions.
///
/// Given a quaternion `q`
/// ```text
/// q + 0 == q
/// ```
fn prop_quaternion_plus_zero_equals_quaternion<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let zero_quat = Quaternion::zero();

    prop_assert_eq!(q + zero_quat, q);

    Ok(())
}

/// A zero quaternion should act as the additive unit element of a set 
/// of quaternions.
///
/// Given a quaternion `q`
/// ```text
/// 0 + q == q
/// ```
fn prop_zero_plus_quaternion_equals_quaternion<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let zero_quat = Quaternion::zero();

    prop_assert_eq!(zero_quat + q, q);

    Ok(())
}


/// Multiplying a quaternion by a scalar `1` should give the original 
/// quaternion.
///
/// Given a quaternion `q`
/// ```text
/// 1 * q == q
/// ```
fn prop_one_times_quaternion_equals_quaternion<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let one_quat: Quaternion<S> = Quaternion::one();

    prop_assert_eq!(one_quat * q, q);

    Ok(())
}


/// Multiplying a quaternion by a scalar `1` should give the original 
/// quaternion.
///
/// Given a quaternion `q`
/// ```text
/// q * 1 == q.
/// ```
fn prop_quaternion_times_one_equals_quaternion<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let one_quat = Quaternion::one();

    prop_assert_eq!(q * one_quat, q);

    Ok(())
}

/// Given quaternions `q1` and `q2`, we should be able to use `q1` 
/// and `q2` interchangeably with their references `&q1` and `&q2` in 
/// arithmetic expressions involving quaternions.
///
/// Given quaternions `q1` and `q2`, and their references `&q1` 
/// and `&q2`, they should satisfy
/// ```text
///  q1 +  q2 == &q1 +  q2
///  q1 +  q2 ==  q1 + &q2
///  q1 +  q2 == &q1 + &q2
///  q1 + &q2 == &q1 +  q2
/// &q1 +  q2 ==  q1 + &q2
/// &q1 +  q2 == &q1 + &q2
///  q1 + &q2 == &q1 + &q2
/// ```
fn prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2<S>(
    q1: Quaternion<S>, 
    q2: Quaternion<S>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!( q1 +  q2, &q1 +  q2);
    prop_assert_eq!( q1 +  q2,  q1 + &q2);
    prop_assert_eq!( q1 +  q2, &q1 + &q2);
    prop_assert_eq!( q1 + &q2, &q1 +  q2);
    prop_assert_eq!(&q1 +  q2,  q1 + &q2);
    prop_assert_eq!(&q1 +  q2, &q1 + &q2);
    prop_assert_eq!( q1 + &q2, &q1 + &q2);

    Ok(())
}

/// Quaternion addition should be commutative.
/// 
/// Given quaternions `q1` and `q2`, we have
/// ```text
/// q1 + q2 == q2 + q1
/// ```
fn prop_quaternion_addition_commutative<S>(q1: Quaternion<S>, q2: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(q1 + q2, q2 + q1);

    Ok(())
}

/// Given three quaternions of integer scalars, quaternion addition 
/// should be associative.
///
/// Given quaternions `q1`, `q2`, and `q3`, we have
/// ```text
/// (q1 + q2) + q3 == q1 + (q2 + q3)
/// ```
fn prop_quaternion_addition_associative<S>(
    q1: Quaternion<S>, 
    q2: Quaternion<S>, 
    q3: Quaternion<S>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!((q1 + q2) + q3, q1 + (q2 + q3));

    Ok(())
}

/// The zero quaternion should act as an additive unit.
///
/// Given a quaternion `q`, we have
/// ```text
/// q - 0 == q
/// ```
fn prop_quaternion_minus_zero_equals_quaternion<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let zero_quat = Quaternion::zero();

    prop_assert_eq!(q - zero_quat, q);

    Ok(())
}

/// Every quaternion should have an additive inverse.
///
/// Given a quaternion `q`, there is a quaternion `-q` such that
/// ```text
/// q - q == q + (-q) = (-q) + q == 0
/// ```
fn prop_quaternion_minus_quaternion_equals_zero<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + Arbitrary
{
    let zero_quat = Quaternion::zero();

    prop_assert_eq!(q - q, zero_quat);
    prop_assert_eq!((-q) + q, zero_quat);
    prop_assert_eq!(q + (-q), zero_quat);

    Ok(())
}

/// Given quaternions `q1` and `q2`, we should be able to use `q1` and 
/// `q2` interchangeably with their references `&q1` and `&q2` in 
/// arithmetic expressions involving quaternions.
///
/// Given quaternions `q1` and `q2`, and their references `&q1` and 
/// `&q2`, they should satisfy
/// ```text
///  q1 -  q2 == &q1 -  q2
///  q1 -  q2 ==  q1 - &q2
///  q1 -  q2 == &q1 - &q2
///  q1 - &q2 == &q1 -  q2
/// &q1 -  q2 ==  q1 - &q2
/// &q1 -  q2 == &q1 - &q2
///  q1 - &q2 == &q1 - &q2
/// ```
fn prop_quaternion1_minus_quaternion2_equals_refquaternion1_minus_refquaternion2<S>(
    q1: Quaternion<S>, 
    q2: Quaternion<S>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{    
    prop_assert_eq!( q1 -  q2, &q1 -  q2);
    prop_assert_eq!( q1 -  q2,  q1 - &q2);
    prop_assert_eq!( q1 -  q2, &q1 - &q2);
    prop_assert_eq!( q1 - &q2, &q1 -  q2);
    prop_assert_eq!(&q1 -  q2,  q1 - &q2);
    prop_assert_eq!(&q1 -  q2, &q1 - &q2);
    prop_assert_eq!( q1 - &q2, &q1 - &q2);

    Ok(())
}

/// Multiplication of a scalar and a quaternion should be commutative.
///
/// Given a constant `c` and a quaternion `q`
/// ```text
/// c * q == q * c
/// ```
fn prop_scalar_times_quaternion_equals_quaternion_times_scalar<S>(c: S, q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let c_quaternion = Quaternion::from_real(c);

    prop_assert_eq!(c_quaternion * q, q * c_quaternion);

    Ok(())
}

/// Quaternions have a multiplicative unit element.
///
/// Given a quaternion `q`, and the unit quaternion `1`, we have
/// ```text
/// q * 1 == 1 * q == q
/// ```
fn prop_quaternion_multiplicative_unit<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let one = Quaternion::identity();

    prop_assert_eq!(q * one, q);
    prop_assert_eq!(one * q, q);
    prop_assert_eq!(q * one, one * q);

    Ok(())
}

/// Every nonzero quaternion over floating point scalars has an 
/// approximate multiplicative inverse.
///
/// Given a quaternion `q` and its inverse `q_inv`, we have
/// ```text
/// q * q_inv == q_inv * q == 1
/// ```
fn prop_approx_quaternion_multiplicative_inverse<S>(q: Quaternion<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat + Arbitrary
{
    prop_assume!(q.is_finite());
    prop_assume!(q.is_invertible());

    let one = Quaternion::identity();
    let q_inv = q.inverse().unwrap();

    prop_assert!(relative_eq!(q * q_inv, one, epsilon = tolerance));
    prop_assert!(relative_eq!(q_inv * q, one, epsilon = tolerance));

    Ok(())
}

/// Exact multiplication of two scalars and a quaternion should be 
/// compatible with multiplication of all scalars. 
///
/// In other words, scalar multiplication of two scalars with a 
/// quaternion should act associatively just like the multiplication 
/// of three scalars. 
///
/// Given scalars `a` and `b`, and a quaternion `q`, we have
/// ```text
/// q * (a * b) == (q * a) * b
/// ```
fn prop_scalar_multiplication_compatibility<S>(a: S, b: S, q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let lhs = q * (a * b);
    let rhs = (q * a) * b;

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// Quaternion multiplication over integer scalars is associative.
///
/// Given quaternions `q1`, `q2`, and `q3`, we have
/// ```text
/// (q1 * q2) * q3 == q1 * (q2 * q3)
/// ```
fn prop_quaternion_multiplication_associative<S>(
    q1: Quaternion<S>, 
    q2: Quaternion<S>, 
    q3: Quaternion<S>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(q1 * (q2 * q3), (q1 * q2) * q3);

    Ok(())
}

/// Scalar multiplication should distribute over quaternion addition.
///
/// Given a scalar `a` and quaternions `q1` and `q2`
/// ```text
/// (q1 + q2) * a == q1 * a + q2 * a
/// ```
fn prop_distribution_over_quaternion_addition<S>(
    a: S, 
    q1: Quaternion<S>, 
    q2: Quaternion<S>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{   
    prop_assert_eq!((q1 + q2) * a,  q1 * a + q2 * a);

    Ok(())
}

/// Multiplication of a sum of scalars should distribute over a quaternion.
///
/// Given scalars `a` and `b` and a quaternion `q`, we have
/// ```text
/// q * (a + b) == q * a + q * b
/// ```
fn prop_distribution_over_scalar_addition<S>(a: S, b: S, q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{   
    prop_assert_eq!(q * (a + b), q * a + q * b);

    Ok(())
}

/// Multiplication of two quaternions by a scalar on the right 
/// should distribute.
///
/// Given quaternions `q1` and `q2`, and a scalar `a`
/// ```text
/// (q1 + q2) * a == q1 * a + q2 * a
/// ```
fn prop_distribution_over_quaternion_addition1<S>(a: S, q1: Quaternion<S>, q2: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{       
    prop_assert_eq!((q1 + q2) * a,  q1 * a + q2 * a);

    Ok(())
}

/// Quaternion multiplication should be distributive on the right.
///
/// Given three quaternions `q1`, `q2`, and `q3`
/// ```text
/// (q1 + q2) * q3 == q1 * q3 + q2 * q3
/// ```
fn prop_quaternion_multiplication_right_distributive<S>(
    q1: Quaternion<S>, 
    q2: Quaternion<S>, 
    q3: Quaternion<S>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!((q1 + q2) * q3, q1 * q3 + q2 * q3);

    Ok(())
}

/// Quaternion multiplication should be distributive on the left.
///
/// Given three quaternions `q1`, `q2`, and `q3`
/// ```text
/// q1 * (q2 + q3) == q1 * q2 + q1 * q3
/// ```
fn prop_quaternion_multiplication_left_distributive<S>(
    q1: Quaternion<S>, 
    q2: Quaternion<S>, 
    q3: Quaternion<S>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!((q1 + q2) * q3, q1 * q3 + q2 * q3);

    Ok(())
}

/// The dot product of quaternions over integer scalars is commutative.
///
/// Given quaternions `q1` and `q2`
/// ```text
/// dot(q1, q2) == dot(q2, q1)
/// ```
fn prop_quaternion_dot_product_commutative<S>(q1: Quaternion<S>, q2: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(q1.dot(&q2), q2.dot(&q1));

    Ok(())
}

/// The dot product of quaternions over integer scalars is right distributive.
///
/// Given quaternions `q1`, `q2`, and `q3`
/// ```text
/// dot(q1, q2 + q3) == dot(q1, q2) + dot(q1, q3)
/// ```
fn prop_quaternion_dot_product_right_distributive<S>(
    q1: Quaternion<S>,
    q2: Quaternion<S>, 
    q3: Quaternion<S>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(q1.dot(&(q2 + q3)), q1.dot(&q2) + q1.dot(&q3));

    Ok(())
}

/// The dot product of quaternions over integer scalars is left 
/// distributive.
///
/// Given quaternions `q1`, `q2`, and `q3`
/// ```text
/// dot(q1 + q2,  q3) == dot(q1, q3) + dot(q2, q3)
/// ```
fn prop_quaternion_dot_product_left_distributive<S>(
    q1: Quaternion<S>,
    q2: Quaternion<S>, 
    q3: Quaternion<S>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!((q1 + q2).dot(&q3), q1.dot(&q3) + q2.dot(&q3));

    Ok(())
}

/// The dot product of quaternions over integer scalars is commutative with 
/// scalars.
///
/// Given quaternions `q1` and `q2`, and scalars `a` and `b`
/// ```text
/// dot(q1 * a, q2 * b) == dot(q1, q2) * (a * b)
/// ```
fn prop_quaternion_dot_product_times_scalars_commutative<S>(
    a: S, 
    b: S,
    q1: Quaternion<S>, 
    q2: Quaternion<S>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let lhs = (q1 * a).dot(&(q2 * b));
    let rhs = q1.dot(&q2) * (a * b);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The dot product of quaternions over integer scalars is right
/// bilinear.
///
/// Given quaternions `q1`, `q2` and `q3`, and scalars `a` and `b`
/// ```text
/// dot(q1, q2 * a + q3 * b) == dot(q1, q2) * a + dot(q1, q3) * b
/// ```
fn prop_quaternion_dot_product_right_bilinear<S>(
    a: S, b: S,
    q1: Quaternion<S>,
    q2: Quaternion<S>, 
    q3: Quaternion<S>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let lhs = q1.dot(&(q2 * a + q3 * b));
    let rhs = q1.dot(&q2) * a + q1.dot(&q3) * b;

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The dot product of quaternions over integer scalars is left
/// bilinear.
///
/// Given quaternions `q1`, `q2` and `q3`, and scalars `a` and `b`
/// ```text
/// dot(q1 * a + q2 * b, q3) == dot(q1, q3) * a + dot(q2, q3) * b
/// ```
fn prop_quaternion_dot_product_left_bilinear<S>(
    a: S, 
    b: S,
    q1: Quaternion<S>,
    q2: Quaternion<S>, 
    q3: Quaternion<S>
) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let lhs = (q1 * a + q2 * b).dot(&q3);
    let rhs = q1.dot(&q3) * a + q2.dot(&q3) * b;

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// Conjugating a quaternion twice should give the original quaternion.
///
/// Given a quaternion `q`
/// ```text
/// conjugate(conjugate(q)) == q
/// ```
fn prop_quaternion_conjugate_conjugate_equals_quaternion<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + Arbitrary
{
    prop_assert_eq!(q.conjugate().conjugate(), q);

    Ok(())
}

/// Quaternion conjugation is linear.
///
/// Given quaternions `q1` and `q2`, quaternion conjugation satisfies
/// ```text
/// conjugate(q1 + q2) == conjugate(q1) + conjugate(q2)
/// ```
fn prop_quaternion_conjugation_linear<S>(q1: Quaternion<S>, q2: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + Arbitrary
{
    prop_assert_eq!((q1 + q2).conjugate(), q1.conjugate() + q2.conjugate());

    Ok(())
}

/// Quaternion multiplication transposes under conjugation.
///
/// Given quaternions `q1` and `q2`
/// ```text
/// conjugate(q1 * q2) == conjugate(q2) * conjugate(q1)
/// ```
fn prop_quaternion_conjugation_transposes_products<S>(q1: Quaternion<S>, q2: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + Arbitrary
{
    prop_assert_eq!((q1 * q2).conjugate(), q2.conjugate() * q1.conjugate());

    Ok(())
}

/// The squared norm of a quaternion is nonnegative. 
///
/// Given a quaternion `q`
/// ```text
/// norm_squared(q) >= 0
/// ```
fn prop_norm_squared_nonnegative<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    let zero = S::zero();

    prop_assert!(q.norm_squared() >= zero);

    Ok(())
}

/// The squared norm function is point separating. In particular, if 
/// the squared distance between two quaternions `q1` and `q2` is 
/// zero, then `q1 == q2`.
///
/// Given quaternions `q1` and `q2`
/// ```text
/// norm_squared(q1 - q2) == 0 => q1 == q2 
/// ```
/// Equivalently, if `q1` is not equal to `q2`, then their squared distance is 
/// nonzero
/// ```text
/// q1 != q2 => norm_squared(q1 - q2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_norm_squared_point_separating<S>(q1: Quaternion<S>, q2: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + Arbitrary
{   
    let zero = S::zero();

    prop_assume!(q1 != q2);
    prop_assert_ne!(
        (q1 - q2).norm_squared(), zero,
        "\n|q1 - q2|^2 = {}\n",
        (q1 - q2).norm_squared()
    );

    Ok(())
}

/// The squared norm function is point separating. In particular, if 
/// the squared distance between two quaternions `q1` and `q2` is 
/// zero, then `q1 == q2`.
///
/// Given quaternions `q1` and `q2`
/// ```text
/// norm_squared(q1 - q2) == 0 => q1 == q2 
/// ```
/// Equivalently, if `q1` is not equal to `q2`, then their squared distance is 
/// nonzero
/// ```text
/// q1 != q2 => norm_squared(q1 - q2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_approx_norm_squared_point_separating<S>(
    q1: Quaternion<S>, 
    q2: Quaternion<S>,
    input_tolerance: S,
    output_tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat + Arbitrary
{
    prop_assume!(relative_ne!(q1, q2, epsilon = input_tolerance));
    prop_assert!(
        (q1 - q2).norm_squared() > output_tolerance,
        "\n|q1 - q2|^2 = {}\n",
        (q1 - q2).norm_squared()
    );

    Ok(())
}

/// The [`Quaternion::magnitude_squared`] function and the [`Quaternion::norm_squared`] 
/// function are synonyms. In particular, given a quaternion `q`
/// ```text
/// magnitude_squared(q) == norm_squared(q)
/// ```
/// where equality is exact.
fn prop_magnitude_squared_norm_squared<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar + Arbitrary
{
    prop_assert_eq!(q.magnitude_squared(), q.norm_squared());

    Ok(())
}

/// The norm of a quaternion is nonnegative. 
///
/// Given a quaternion `q`
/// ```text
/// norm(q) >= 0
/// ```
fn prop_norm_nonnegative<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat + Arbitrary
{
    let zero = S::zero();

    prop_assert!(q.norm() >= zero);

    Ok(())
}

/// The norm function is point separating. In particular, if 
/// the distance between two quaternions `q1` and `q2` is 
/// zero, then `q1 == q2`.
///
/// Given quaternions `q1` and `q2`
/// ```text
/// norm(q1 - q2) == 0 => q1 == q2 
/// ```
/// Equivalently, if `q1` is not equal to `q2`, then their distance is 
/// nonzero
/// ```text
/// q1 != q2 => norm(q1 - q2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_approx_norm_point_separating<S>(q1: Quaternion<S>, q2: Quaternion<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat + Arbitrary
{
    prop_assume!(relative_ne!(q1, q2, epsilon = tolerance));
    prop_assert!(
        (q1 - q2).norm() > tolerance,
        "\n|q1 - q2| = {}\n",
        (q1 - q2).norm()
    );

    Ok(())
}

/// The **L1** norm of a quaternion is nonnegative. 
///
/// Given a quaternion `q`
/// ```text
/// l1_norm(q) >= 0
/// ```
fn prop_l1_norm_nonnegative<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + Arbitrary
{
    let zero = S::zero();

    prop_assert!(q.l1_norm() >= zero);

    Ok(())
}

/// The **L1** norm function is point separating. In particular, if 
/// the distance between two quaternions `q1` and `q2` is 
/// zero, then `q1 = q2`.
///
/// Given quaternions `q1` and `q2`
/// ```text
/// l1_norm(q1 - q2) = 0 => q1 = q2 
/// ```
/// Equivalently, if `q1` is not equal to `q2`, then their distance is 
/// nonzero
/// ```text
/// q1 != q2 => l1_norm(q1 - q2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_l1_norm_point_separating<S>(q1: Quaternion<S>, q2: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned + Arbitrary
{
    let zero = S::zero();

    prop_assume!(q1 != q2);
    prop_assert_ne!(
        (q1 - q2).l1_norm(), zero,
        "\nl1_norm(q1 - q2) = {}\n",
        (q1 - q2).l1_norm()
    );

    Ok(())
}

/// The **L1** norm function is point separating. In particular, if 
/// the distance between two quaternions `q1` and `q2` is 
/// zero, then `q1 == q2`.
///
/// Given quaternions `q1` and `q2`
/// ```text
/// l1_norm(q1 - q2) == 0 => q1 == q2 
/// ```
/// Equivalently, if `q1` is not equal to `q2`, then their distance is 
/// nonzero
/// ```text
/// q1 != q2 => l1_norm(q1 - q2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_approx_l1_norm_point_separating<S>(q1: Quaternion<S>, q2: Quaternion<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat + Arbitrary
{
    prop_assume!(relative_ne!(q1, q2, epsilon = tolerance));
    prop_assert!(
        (q1 - q2).l1_norm() > tolerance,
        "\nl1_norm(q1 - q2) = {}\n",
        (q1 - q2).l1_norm()
    );

    Ok(())
}

/// The [`Quaternion::magnitude`] function and the [`Quaternion::norm`] function
/// are synonyms. In particular, given a quaternion `q`
/// ```text
/// magnitude(q) == norm(q)
/// ```
/// where equality is exact.
fn prop_magnitude_norm_synonyms<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat + Arbitrary
{
    prop_assert_eq!(q.magnitude(), q.norm());

    Ok(())
}

/// The [`Quaternion::l2_norm`] function and the [`Quaternion::norm`] function
/// are synonyms. In particular, given a quaternion `q`
/// ```text
/// l2_norm(q) == norm(q)
/// ```
/// where equality is exact.
fn prop_l2_norm_norm_synonyms<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat + Arbitrary
{
    prop_assert_eq!(q.l2_norm(), q.norm());

    Ok(())
}

/// The square of the positive square root of a quaternion is the original
/// quaternion.
/// 
/// Given a quaternion `q`
/// ```text
/// sqrt(q) * sqrt(q) == q
/// ```
fn prop_approx_positive_square_root_squared<S>(q: Quaternion<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat + Arbitrary
{
    let sqrt_q = q.sqrt();

    prop_assert!(
        relative_eq!(sqrt_q * sqrt_q, q, epsilon = tolerance),
        "q = {:?}\nsqrt_q = {:?}\nsqrt_q * sqrt_q = {:?}",
        q, sqrt_q, sqrt_q * sqrt_q
    );

    Ok(())
}

/// The square of the negative square root of a quaternion is the original
/// quaternion.
/// 
/// Given a quaternion `q`
/// ```text
/// -sqrt(q) * -sqrt(q) == q
/// ```
fn prop_approx_negative_square_root_squared<S>(q: Quaternion<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat + Arbitrary
{
    let minus_sqrt_q = -q.sqrt();

    prop_assert!(
        relative_eq!(minus_sqrt_q * minus_sqrt_q, q, epsilon = tolerance),
        "q = {:?}\nminus_sqrt_q = {:?}\nminus_sqrt_q * minus_sqrt_q = {:?}",
        q, minus_sqrt_q, minus_sqrt_q * minus_sqrt_q
    );

    Ok(())
}


#[cfg(test)]
mod quaternion_f64_arithmetic_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_zero_times_quaternion_equals_zero(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_zero_times_quaternion_equals_zero(q)?
        }
        
        #[test]
        fn prop_quaternion_times_zero_equals_zero(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_times_zero_equals_zero(q)?
        }

        #[test]
        fn prop_quaternion_plus_zero_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_plus_zero_equals_quaternion(q)?
        }

        #[test]
        fn prop_zero_plus_quaternion_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_zero_plus_quaternion_equals_quaternion(q)?
        }

        #[test]
        fn prop_one_times_quaternion_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_one_times_quaternion_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion_times_one_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_times_one_equals_quaternion(q)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_arithmetic_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_zero_times_quaternion_equals_zero(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_zero_times_quaternion_equals_zero(q)?
        }
        
        #[test]
        fn prop_quaternion_times_zero_equals_zero(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_times_zero_equals_zero(q)?
        }

        #[test]
        fn prop_quaternion_plus_zero_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_plus_zero_equals_quaternion(q)?
        }

        #[test]
        fn prop_zero_plus_quaternion_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_zero_plus_quaternion_equals_quaternion(q)?
        }

        #[test]
        fn prop_one_times_quaternion_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_one_times_quaternion_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion_times_one_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_times_one_equals_quaternion(q)?
        }
    }
}


#[cfg(test)]
mod complex_f64_add_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_quaternion_plus_zero_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_plus_zero_equals_quaternion(q)?
        }

        #[test]
        fn prop_zero_plus_quaternion_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_zero_plus_quaternion_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(q1, q2)?
        }

        #[test]
        fn prop_quaternion_addition_commutative(
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_quaternion_addition_commutative(q1, q2)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_add_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_quaternion_plus_zero_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_plus_zero_equals_quaternion(q)?
        }

        #[test]
        fn prop_zero_plus_quaternion_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_zero_plus_quaternion_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(q1, q2)?
        }

        #[test]
        fn prop_quaternion_addition_commutative(
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_quaternion_addition_commutative(q1, q2)?
        }

        #[test]
        fn prop_quaternion_addition_associative(
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any(), 
            q3 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            let q3: super::Quaternion<i32> = q3;
            super::prop_quaternion_addition_associative(q1, q2, q3)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_sub_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_quaternion_minus_zero_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_minus_zero_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion_minus_quaternion_equals_zero(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_minus_quaternion_equals_zero(q)?
        }

        #[test]
        fn prop_quaternion1_minus_quaternion2_equals_refquaternion1_minus_refquaternion2(
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_quaternion1_minus_quaternion2_equals_refquaternion1_minus_refquaternion2(q1, q2)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_sub_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_quaternion_minus_zero_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_minus_zero_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion_minus_quaternion_equals_zero(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_minus_quaternion_equals_zero(q)?
        }

        #[test]
        fn prop_quaternion1_minus_quaternion2_equals_refquaternion1_minus_refquaternion2(
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_quaternion1_minus_quaternion2_equals_refquaternion1_minus_refquaternion2(q1, q2)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_mul_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_scalar_times_quaternion_equals_quaternion_times_scalar(
            c in super::strategy_scalar_f64_any(), 
            q in super::strategy_quaternion_any()
        ) {
            let c: f64 = c;
            let q: super::Quaternion<f64> = q;
            super::prop_scalar_times_quaternion_equals_quaternion_times_scalar(c, q)?
        }

        #[test]
        fn prop_quaternion_multiplicative_unit(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_multiplicative_unit(q)?
        }

        #[test]
        fn prop_approx_quaternion_multiplicative_inverse(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_approx_quaternion_multiplicative_inverse(q, 1e-8)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_mul_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_scalar_times_quaternion_equals_quaternion_times_scalar(
            c in super::strategy_scalar_i32_any(), 
            q in super::strategy_quaternion_any()
        ) {
            let c: i32 = c;
            let q: super::Quaternion<i32> = q;
            super::prop_scalar_times_quaternion_equals_quaternion_times_scalar(c, q)?
        }

        #[test]
        fn prop_scalar_multiplication_compatibility(
            a in super::strategy_scalar_i32_any(), 
            b in super::strategy_scalar_i32_any(), 
            q in super::strategy_quaternion_any()
        ) {
            let a: i32 = a;
            let b: i32 = b;
            let q: super::Quaternion<i32> = q;
            super::prop_scalar_multiplication_compatibility(a, b, q)?
        }

        #[test]
        fn prop_quaternion_multiplication_associative(
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any(), 
            q3 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            let q3: super::Quaternion<i32> = q3;
            super::prop_quaternion_multiplication_associative(q1, q2, q3)?
        }

        #[test]
        fn prop_quaternion_multiplicative_unit(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_multiplicative_unit(q)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_distributive_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_distribution_over_quaternion_addition(
            a in super::strategy_scalar_i32_any(), 
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any()
        ) {
            let a: i32 = a;
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_distribution_over_quaternion_addition(a, q1, q2)?
        }

        #[test]
        fn prop_distribution_over_scalar_addition(
            a in super::strategy_scalar_i32_any(), 
            b in super::strategy_scalar_i32_any(), 
            q in super::strategy_quaternion_any()
        ) {
            let a: i32 = a;
            let b: i32 = b;
            let q: super::Quaternion<i32> = q;
            super::prop_distribution_over_scalar_addition(a, b, q)?
        }

        #[test]
        fn prop_distribution_over_quaternion_addition1(
            a in super::strategy_scalar_i32_any(), 
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any()
        ) {
            let a: i32 = a;
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_distribution_over_quaternion_addition1(a, q1, q2)?
        }

        #[test]
        fn prop_quaternion_multiplication_right_distributive(
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any(), 
            q3 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            let q3: super::Quaternion<i32> = q3;
            super::prop_quaternion_multiplication_right_distributive(q1, q2, q3)?
        }

        #[test]
        fn prop_quaternion_multiplication_left_distributive(
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any(), 
            q3 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            let q3: super::Quaternion<i32> = q3;
            super::prop_quaternion_multiplication_left_distributive(q1, q2, q3)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_dot_product_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_quaternion_dot_product_commutative(q1 in super::strategy_quaternion_any(), q2 in super::strategy_quaternion_any()) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_quaternion_dot_product_commutative(q1, q2)?
        }

        #[test]
        fn prop_quaternion_dot_product_right_distributive(
            q1 in super::strategy_quaternion_any(),
            q2 in super::strategy_quaternion_any(), 
            q3 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            let q3: super::Quaternion<i32> = q3;
            super::prop_quaternion_dot_product_right_distributive(q1, q2, q3)?
        }

        #[test]
        fn prop_quaternion_dot_product_left_distributive(
            q1 in super::strategy_quaternion_any(),
            q2 in super::strategy_quaternion_any(), 
            q3 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            let q3: super::Quaternion<i32> = q3;
            super::prop_quaternion_dot_product_left_distributive(q1, q2, q3)?
        }

        #[test]
        fn prop_quaternion_dot_product_times_scalars_commutative(
            a in super::strategy_scalar_i32_any(), 
            b in super::strategy_scalar_i32_any(),
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any()
        ) {
            let a: i32 = a;
            let b: i32 = b;
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_quaternion_dot_product_times_scalars_commutative(a, b, q1, q2)?
        }

        #[test]
        fn prop_quaternion_dot_product_right_bilinear(
            a in super::strategy_scalar_i32_any(), 
            b in super::strategy_scalar_i32_any(),
            q1 in super::strategy_quaternion_any(),
            q2 in super::strategy_quaternion_any(), 
            q3 in super::strategy_quaternion_any()
        ) {
            let a: i32 = a;
            let b: i32 = b;
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            let q3: super::Quaternion<i32> = q3;
            super::prop_quaternion_dot_product_right_bilinear(a, b, q1, q2, q3)?
        }

        #[test]
        fn prop_quaternion_dot_product_left_bilinear(
            a in super::strategy_scalar_i32_any(), 
            b in super::strategy_scalar_i32_any(),
            q1 in super::strategy_quaternion_any(),
            q2 in super::strategy_quaternion_any(), 
            q3 in super::strategy_quaternion_any()
        ) {
            let a: i32 = a;
            let b: i32 = b;
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            let q3: super::Quaternion<i32> = q3;
            super::prop_quaternion_dot_product_left_bilinear(a, b, q1, q2, q3)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_conjugation_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_quaternion_conjugate_conjugate_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_conjugate_conjugate_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion_conjugation_linear(q1 in super::strategy_quaternion_any(), q2 in super::strategy_quaternion_any()) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_quaternion_conjugation_linear(q1, q2)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_conjugation_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_quaternion_conjugate_conjugate_equals_quaternion(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_conjugate_conjugate_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion_conjugation_linear(q1 in super::strategy_quaternion_any(), q2 in super::strategy_quaternion_any()) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_quaternion_conjugation_linear(q1, q2)?
        }

        #[test]
        fn prop_quaternion_conjugation_transposes_products(
            q1 in super::strategy_quaternion_any(), 
            q2 in super::strategy_quaternion_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_quaternion_conjugation_transposes_products(q1, q2)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_norm_squared_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_norm_squared_nonnegative(q in super::strategy_quaternion_f64_norm_squared()) {
            let q: super::Quaternion<f64> = q;
            super::prop_norm_squared_nonnegative(q)?
        }

        #[test]
        fn prop_approx_norm_squared_point_separating(
            q1 in super::strategy_quaternion_f64_norm_squared(), 
            q2 in super::strategy_quaternion_f64_norm_squared()
        ) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_approx_norm_squared_point_separating(q1, q2, 1e-10, 1e-20)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_norm_squared_synonym_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_magnitude_squared_norm_squared(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_magnitude_squared_norm_squared(q)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_norm_squared_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_norm_squared_nonnegative(q in super::strategy_quaternion_i32_norm_squared()) {
            let q: super::Quaternion<i32> = q;
            super::prop_norm_squared_nonnegative(q)?
        }

        #[test]
        fn prop_norm_squared_point_separating(
            q1 in super::strategy_quaternion_i32_norm_squared(),
            q2 in super::strategy_quaternion_i32_norm_squared()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_norm_squared_point_separating(q1, q2)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_norm_squared_synonym_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_magnitude_squared_norm_squared(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_magnitude_squared_norm_squared(q)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_norm_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_norm_nonnegative(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_norm_nonnegative(q)?
        }

        #[test]
        fn prop_approx_norm_point_separating(q1 in super::strategy_quaternion_any(), q2 in super::strategy_quaternion_any()) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_approx_norm_point_separating(q1, q2, 1e-10)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_l1_norm_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_l1_norm_nonnegative(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_l1_norm_nonnegative(q)?
        }

        #[test]
        fn prop_approx_l1_norm_point_separating(q1 in super::strategy_quaternion_any(), q2 in super::strategy_quaternion_any()) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_approx_l1_norm_point_separating(q1, q2, 1e-10)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_l1_norm_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_l1_norm_nonnegative(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_l1_norm_nonnegative(q)?
        }

        #[test]
        fn prop_l1_norm_point_separating(q1 in super::strategy_quaternion_any(), q2 in super::strategy_quaternion_any()) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_l1_norm_point_separating(q1, q2)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_norm_synonym_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_magnitude_norm_synonyms(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_magnitude_norm_synonyms(q)?
        }

        #[test]
        fn prop_l2_norm_norm_synonyms(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_l2_norm_norm_synonyms(q)?
        }

        #[test]
        fn prop_magnitude_squared_norm_squared(q in super::strategy_quaternion_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_magnitude_squared_norm_squared(q)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_sqrt_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_positive_square_root_squared(q in super::strategy_quaternion_squared_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_approx_positive_square_root_squared(q, 1e-7)?
        }

        #[test]
        fn prop_approx_negative_square_root_squared(q in super::strategy_quaternion_squared_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_approx_negative_square_root_squared(q, 1e-7)?
        }
    }
}

