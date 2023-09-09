extern crate cglinalg_core;
extern crate num_traits;
extern crate proptest;


use proptest::prelude::*;
use cglinalg_core::{
    Quaternion,
    Vector3,
    SimdScalar,
    SimdScalarSigned,
    SimdScalarFloat,
};
use approx::{
    relative_eq,
    relative_ne,
    abs_diff_ne,
};


fn strategy_quaternion_polar_from_range<S>(min_scale: S, max_scale: S, min_angle: S, max_angle: S) -> impl Strategy<Value = Quaternion<S>>
where
    S: SimdScalarFloat + Arbitrary
{
    use cglinalg_core::{
        Radians,
        Unit,
    };

    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarFloat
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, S, S, S, S)>().prop_map(move |(_scale, _angle, _axis_x, _axis_y, _axis_z)| {
        let scale = SimdScalarSigned::abs(rescale(_scale, min_scale, max_scale));
        let angle = Radians(SimdScalarSigned::abs(rescale(_angle, min_angle, max_angle)));
        let unnormalized_axis = {
            let axis_x = rescale(_axis_x, S::machine_epsilon(), S::one());
            let axis_y = rescale(_axis_y, S::machine_epsilon(), S::one());
            let axis_z = rescale(_axis_z, S::machine_epsilon(), S::one());

            Vector3::new(axis_x, axis_y, axis_z)
        };
        let axis = Unit::from_value(unnormalized_axis);

        Quaternion::from_polar_decomposition(scale, angle, &axis)
    })
    .no_shrink()
}

/*
fn strategy_quaternion_polar_from_range_z_axis<S>(min_scale: S, max_scale: S, min_angle: S, max_angle: S) -> impl Strategy<Value = Quaternion<S>>
where
    S: SimdScalarFloat + Arbitrary
{
    use cglinalg_core::{
        Radians,
        Unit,
    };

    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarFloat
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, S)>().prop_map(move |(_scale, _angle)| {
        let scale = SimdScalarSigned::abs(rescale(_scale, min_scale, max_scale));
        let angle = Radians(SimdScalarSigned::abs(rescale(_angle, min_angle, max_angle)));
        let axis = Unit::from_value(Vector3::unit_z());

        Quaternion::from_polar_decomposition(scale, angle, &axis)
    })
    .no_shrink()
}
*/

fn strategy_quaternion_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Quaternion<S>>
where
    S: SimdScalarSigned + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarSigned
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, S, S, S)>().prop_map(move |(_qs, _qx, _qy, _qz)| {
        let sign_qs = _qs.signum();
        let sign_qx = _qx.signum();
        let sign_qy = _qy.signum();
        let sign_qz = _qz.signum();
        let abs_qs = _qs.abs();
        let abs_qx = _qx.abs();
        let abs_qy = _qy.abs();
        let abs_qz = _qz.abs();
        let qs = sign_qs * rescale(abs_qs, min_value, max_value);
        let qx = sign_qx * rescale(abs_qx, min_value, max_value);
        let qy = sign_qy * rescale(abs_qy, min_value, max_value);
        let qz = sign_qz * rescale(abs_qz, min_value, max_value);
        
        Quaternion::new(qs, qx, qy, qz)
    })
    .no_shrink()
}

fn strategy_scalar_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = S>
where
    S: SimdScalarSigned + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarSigned
    {
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
    let min_value = f64::sqrt(f64::EPSILON) / 2_f64;
    let max_value = f64::sqrt(f64::MAX) / 2_f64;

    strategy_scalar_signed_from_abs_range(min_value, max_value)
}

fn strategy_scalar_i32_any() -> impl Strategy<Value = i32> {
    let min_value = 0_i32;
    // let max_value = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
    let max_value = 46340_i32;

    strategy_scalar_signed_from_abs_range(min_value, max_value)
}

fn strategy_quaternion_f64_any() -> impl Strategy<Value = Quaternion<f64>> {
    let min_value = f64::sqrt(f64::EPSILON) / 2_f64;
    let max_value = f64::sqrt(f64::MAX) / 2_f64;

    strategy_quaternion_signed_from_abs_range(min_value, max_value)
}

fn strategy_quaternion_i32_any() -> impl Strategy<Value = Quaternion<i32>> {
    let min_value = 0_i32;
    // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
    let max_square_root = 46340_i32;
    let max_value = max_square_root / 4;

    strategy_quaternion_signed_from_abs_range(min_value, max_value)
}

fn strategy_quaternion_f64_norm_squared() -> impl Strategy<Value = Quaternion<f64>> {
    let min_scale = f64::sqrt(f64::EPSILON) / 2_f64;
    let max_scale = f64::sqrt(f64::MAX) / 2_f64;
    let min_angle = 0_f64;
    let max_angle = f64::two_pi();

    strategy_quaternion_polar_from_range(min_scale, max_scale, min_angle, max_angle)
}

fn strategy_quaternion_i32_norm_squared() -> impl Strategy<Value = Quaternion<i32>> {
    let min_value = 0_i32;
    // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
    let max_square_root = 46340_i32;
    let max_value = max_square_root / 4;

    strategy_quaternion_signed_from_abs_range(min_value, max_value)
}

fn strategy_quaternion_squared_any() -> impl Strategy<Value = Quaternion<f64>> {
    let min_scale = f64::sqrt(f64::sqrt(f64::EPSILON)) / 4_f64;
    let max_scale = f64::sqrt(f64::sqrt(f64::MAX)) / 4_f64;
    let min_angle = 0_f64;
    let max_angle = f64::two_pi();

    strategy_quaternion_polar_from_range(min_scale, max_scale, min_angle, max_angle)
}

/*
fn strategy_quaternion_squared_z_axis() -> impl Strategy<Value = Quaternion<f64>> {
    let min_scale = 1e-4;
    let max_scale = 100_f64;
    let min_angle = 0_f64;
    let max_angle = f64::two_pi();

    strategy_quaternion_polar_from_range_z_axis(min_scale, max_scale, min_angle, max_angle)
}
*/

fn strategy_quaternion_f64_exp() -> impl Strategy<Value = Quaternion<f64>> {
    let min_scale = f64::ln(f64::EPSILON) / 4_f64;
    let max_scale = f64::ln(f64::MAX) / 4_f64;
    let min_angle = 0_f64;
    let max_angle = f64::two_pi();

    strategy_quaternion_polar_from_range(min_scale, max_scale, min_angle, max_angle)
}

fn strategy_quaternion_f64_sqrt() -> impl Strategy<Value = Quaternion<f64>> {
    let min_scale = f64::sqrt(f64::EPSILON) / 2_f64;
    let max_scale = f64::sqrt(f64::MAX) / 2_f64;
    let min_angle = 0_f64;
    let max_angle = f64::two_pi();

    strategy_quaternion_polar_from_range(min_scale, max_scale, min_angle, max_angle)
}

fn strategy_quaternion_f64_sqrt_product() -> impl Strategy<Value = Quaternion<f64>> {
    let min_scale = f64::sqrt(f64::sqrt(f64::EPSILON)) / 4_f64;
    let max_scale = f64::sqrt(f64::sqrt(f64::MAX)) / 4_f64;
    let min_angle = 0_f64;
    let max_angle = f64::two_pi();

    strategy_quaternion_polar_from_range(min_scale, max_scale, min_angle, max_angle)
}


/// A scalar `0` times a quaternion should be a zero quaternion.
///
/// Given a quaternion `q`, it satisfies
/// ```text
/// 0 * q == 0.
/// ```
fn prop_zero_times_quaternion_equals_zero<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalarSigned
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalarFloat
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalarSigned
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
    S: SimdScalarSigned
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
    S: SimdScalarSigned
{
    prop_assert_eq!((q1 * q2).conjugate(), q2.conjugate() * q1.conjugate());

    Ok(())
}

/// The squared modulus of a quaternion is nonnegative. 
///
/// Given a quaternion `q`
/// ```text
/// norm_squared(q) >= 0
/// ```
fn prop_modulus_squared_nonnegative<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let zero = S::zero();

    prop_assert!(q.modulus_squared() >= zero);

    Ok(())
}

/// The squared modulus function is point separating. In particular, if 
/// the squared distance between two quaternions `q1` and `q2` is 
/// zero, then `q1 == q2`.
///
/// Given quaternions `q1` and `q2`
/// ```text
/// modulus_squared(q1 - q2) == 0 => q1 == q2 
/// ```
/// Equivalently, if `q1` is not equal to `q2`, then their squared distance is 
/// nonzero
/// ```text
/// q1 != q2 => modulus_squared(q1 - q2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_modulus_squared_point_separating<S>(q1: Quaternion<S>, q2: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{   
    let zero = S::zero();

    prop_assume!(q1 != q2);
    prop_assert_ne!(
        (q1 - q2).modulus_squared(), zero,
        "\n|q1 - q2|^2 = {}\n",
        (q1 - q2).modulus_squared()
    );

    Ok(())
}

/// The squared modulus function is homogeneous.
/// 
/// Given a quaternion `q` and a scalar `c`
/// ```text
/// modulus_squared(q * c) == modulus_squared(q) * abs(c) * abs(c)
/// ```
fn prop_modulus_squared_homogeneous_squared<S>(q: Quaternion<S>, c: S) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = (q * c).modulus_squared();
    let rhs = q.modulus_squared() * c.abs() * c.abs();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The squared modulus function is point separating. In particular, if 
/// the squared distance between two quaternions `q1` and `q2` is 
/// zero, then `q1 == q2`.
///
/// Given quaternions `q1` and `q2`
/// ```text
/// modulus_squared(q1 - q2) == 0 => q1 == q2 
/// ```
/// Equivalently, if `q1` is not equal to `q2`, then their squared distance is 
/// nonzero
/// ```text
/// q1 != q2 => modulus_squared(q1 - q2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_approx_modulus_squared_point_separating<S>(
    q1: Quaternion<S>, 
    q2: Quaternion<S>,
    input_tolerance: S,
    output_tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(q1, q2, epsilon = input_tolerance));
    prop_assert!(
        (q1 - q2).modulus_squared() > output_tolerance,
        "\n|q1 - q2|^2 = {}\n",
        (q1 - q2).modulus_squared()
    );

    Ok(())
}

/// The [`Quaternion::norm_squared`] function and the [`Quaternion::modulus_squared`]
/// function are synonyms. In particular, given a quaternion `q`
/// ```text
/// norm_squared(q) == modulus_squared(q)
/// ```
/// where equality is exact.
fn prop_norm_squared_modulus_squared<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(q.norm_squared(), q.modulus_squared());

    Ok(())
}

/// The [`Quaternion::magnitude_squared`] function and the [`Quaternion::modulus_squared`] 
/// function are synonyms. In particular, given a quaternion `q`
/// ```text
/// magnitude_squared(q) == modulus_squared(q)
/// ```
/// where equality is exact.
fn prop_magnitude_squared_modulus_squared<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(q.magnitude_squared(), q.modulus_squared());

    Ok(())
}

/// The modulus of a quaternion is nonnegative. 
///
/// Given a quaternion `q`
/// ```text
/// modulus(q) >= 0
/// ```
fn prop_modulus_nonnegative<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let zero = S::zero();

    prop_assert!(q.modulus() >= zero);

    Ok(())
}

/// The norm function is point separating. In particular, if 
/// the distance between two quaternions `q1` and `q2` is 
/// zero, then `q1 == q2`.
///
/// Given quaternions `q1` and `q2`
/// ```text
/// modulus(q1 - q2) == 0 => q1 == q2 
/// ```
/// Equivalently, if `q1` is not equal to `q2`, then their distance is 
/// nonzero
/// ```text
/// q1 != q2 => modulus(q1 - q2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_approx_modulus_point_separating<S>(q1: Quaternion<S>, q2: Quaternion<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(q1, q2, epsilon = tolerance));
    prop_assert!(
        (q1 - q2).modulus() > tolerance,
        "\n|q1 - q2| = {}\n",
        (q1 - q2).modulus()
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
    S: SimdScalarSigned
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
    S: SimdScalarSigned
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

/// The quaternion **L1** norm is homogeneous.
/// 
/// Given a quaternion `q` and a constant scalar `c`
/// ```text
/// l1_norm(q * c) == l1_norm(q) * abs(c)
/// ```
fn prop_l1_norm_homogeneous<S>(q: Quaternion<S>, c: S) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = (q * c).l1_norm();
    let rhs = q.l1_norm() * c.abs();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The quaternion **L1** norm satisfies the triangle inequality.
/// 
/// Given quaternions `q1` and `q2`
/// ```text
/// l1_norm(q1 + q1) <= l1_norm(q1) + l1_norm(q2)
/// ```
fn prop_l1_norm_triangle_inequality<S>(q1: Quaternion<S>, q2: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = (q1 + q2).l1_norm();
    let rhs = q1.l1_norm() + q2.l1_norm();

    prop_assert!(lhs <= rhs);

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
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(q1, q2, epsilon = tolerance));
    prop_assert!(
        (q1 - q2).l1_norm() > tolerance,
        "\nl1_norm(q1 - q2) = {}\n",
        (q1 - q2).l1_norm()
    );

    Ok(())
}

/// The [`Quaternion::norm`] function and the [`Quaternion::modulus`] function
/// are synonyms. In particular, given a quaternion `q`
/// ```text
/// norm(q) == modulus(q)
/// ```
/// where equality is exact.
fn prop_norm_modulus_synonyms<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert_eq!(q.norm(), q.modulus());

    Ok(())
}

/// The [`Quaternion::magnitude`] function and the [`Quaternion::modulus`] function
/// are synonyms. In particular, given a quaternion `q`
/// ```text
/// magnitude(q) == modulus(q)
/// ```
/// where equality is exact.
fn prop_magnitude_modulus_synonyms<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert_eq!(q.magnitude(), q.modulus());

    Ok(())
}

/// The [`Quaternion::l2_norm`] function and the [`Quaternion::modulus`] function
/// are synonyms. In particular, given a quaternion `q`
/// ```text
/// l2_norm(q) == modulus(q)
/// ```
/// where equality is exact.
fn prop_l2_norm_modulus_synonyms<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert_eq!(q.l2_norm(), q.modulus());

    Ok(())
}

/// The quaternion exponential satisfies the following relationship.
/// 
/// Given a quaternion `q`, let `s := scalar(q)` be the scalar part of `q`, and
/// let `v := vector(q)` be the vector part of `q`. Then
/// ```text
/// exp(q) == exp(s + v) == exp(s) * exp(v)
/// ```
fn prop_approx_exp_scalar_vector_sum<S>(q: Quaternion<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let q_scalar = Quaternion::from_real(q.scalar());
    let q_vector = Quaternion::from_pure(q.vector());
    let lhs = (q_scalar + q_vector).exp();
    let rhs = q_scalar.exp() * q_vector.exp();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The exponential of a quaternion is non-zero.
/// 
/// Given a quaternion `q`
/// ```text
/// exp(q) != 0
/// ```
fn prop_exp_quaternion_nonzero<S>(q: Quaternion<S>) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let zero_quaternion = Quaternion::zero();

    prop_assert_ne!(q.exp(), zero_quaternion);

    Ok(())
}

/// The quaternion exponential satisfies the following relation.
/// 
/// Given a quaternion `q`
/// ```text
/// exp(q) * exp(-q) == exp(-q) * exp(q) == 1
/// ```
fn prop_approx_exp_quaternion_exp_negative_quaternion<S>(q: Quaternion<S>, tolerance: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let unit_s = Quaternion::unit_s();
    let exp_q = q.exp();
    let exp_negative_q = (-q).exp();

    prop_assert!(relative_eq!(exp_negative_q * exp_q, unit_s, epsilon = tolerance));
    prop_assert!(relative_eq!(exp_q * exp_negative_q, unit_s, epsilon = tolerance));

    Ok(())
}

/// The scalar part of the principal value of a quaternion satisfies the following
/// relation.
/// 
/// Given a quaternion `q`
/// ```text
/// scalar(ln(q)) == ln(norm(q))
/// ```
fn prop_approx_quaternion_ln_scalar_part<S>(q: Quaternion<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = q.ln().scalar();
    let rhs = q.norm().ln();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The quaternion exponential and principal value of the quaternion logarithm 
/// satisfy the following relation.
/// 
/// Given a quaternion `q`
/// ```text
/// exp(ln(q)) == q
/// ```
fn prop_approx_exp_ln_identity<S>(q: Quaternion<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = q.ln().exp();
    let rhs = q;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative), "lhs = {}; rhs = {}", lhs, rhs);

    Ok(())
}

/// The principal argument of two quaternions that differ only by a phase factor
/// of `2 * pi * k` for some integer `k` have the same argument up to a sign factor.
/// 
/// Given quaternions `q1` and `q2` such that `q1 := r * exp(v * angle)` and 
/// `q2 := r * exp(v * (angle + 2 * pi * k))` where `r` is a floating point number 
/// and `k` is an integer
/// ```text
/// arg(q1) == arg(q2)
/// ```
/// Moreover, this indicates that the `arg` function correctly implements the fact
/// that the principal argument is unique on the interval `[0, pi]`.
fn prop_approx_arg_congruent<S>(q: Quaternion<S>, k: i32, tolerance: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    use cglinalg_core::Radians;

    let (norm_q, _, axis_q) = q.polar_decomposition();
    let arg_q = q.arg();
    let _k = cglinalg_core::cast(k);
    let arg_new_q = arg_q + S::two_pi() * _k;
    let angle_new_q = {
        // NOTE: The principal argument of the quaternion is half of the angle 
        // of rotation, not the full angle of rotation.
        let two = S::one() + S::one();
        Radians(two * arg_new_q)
    };
    let new_q = Quaternion::from_polar_decomposition(norm_q, angle_new_q, &axis_q.unwrap());

    let lhs = q.arg();
    let rhs = new_q.arg();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The principal argument of a quaternion is in the closed interval `[0, pi]`.
/// 
/// Given a quaternion `q`
/// ```text
/// 0 =< arg(q) <= pi
/// ```
fn prop_approx_arg_range<S>(q: Quaternion<S>) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let arg_q = q.arg();

    prop_assert!(arg_q >= S::zero());
    prop_assert!(arg_q <= S::pi());

    Ok(())
}

/// The square of the positive square root of a quaternion is the original
/// quaternion.
/// 
/// Given a quaternion `q` such that `vector(q) != 0`
/// ```text
/// sqrt(q) * sqrt(q) == q
/// ```
/// When `vector(q) == 0`, the quaterion square root is not well-defined.
fn prop_approx_square_root_quaternion_squared<S>(q: Quaternion<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    // Ensure that the vector part is sufficiently far from zero for the square 
    // root to be well-defined for `q`.
    prop_assume!(abs_diff_ne!(q.vector(), Vector3::zero(), epsilon = cglinalg_core::cast(1e-6)));
    
    let sqrt_q = q.sqrt();
    let lhs = sqrt_q * sqrt_q;
    let rhs = q;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/*
/// The square of the square root of the negation of a quaternion is the 
/// negation of the original quaternion.
/// 
/// Given a quaternion `q` such that `vector(q) != 0`
/// ```text
/// sqrt(-q) * sqrt(-q) == -q
/// ```
/// When `vector(q) == 0`, the quaterion square root is not well-defined.
fn prop_approx_square_root_negative_quaternion_squared<S>(q: Quaternion<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    // Ensure that the vector part is sufficiently far from zero for the square 
    // root to be well-defined for `q`.
    prop_assume!(abs_diff_ne!(q.vector(), Vector3::zero(), epsilon = cglinalg_core::cast(1e-4)));
    
    let negative_q = -q;
    let sqrt_negative_q = negative_q.sqrt();
    let sqrt_negative_q_squared = sqrt_negative_q * sqrt_negative_q;
    
    let lhs = sqrt_negative_q_squared;
    let rhs = negative_q;
    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative),
        "q = {:?};\n-q = {:?};\nsqrt(-q) = {:?};\nsqrt(-q) * sqrt(-q) = {:?}",
        q, negative_q, sqrt_negative_q, lhs
    );

    Ok(())
}
*/

/// The norm of the square root of the product of two quaternions is the product 
/// of the norms of the square roots of the two quaternions separately.
/// 
/// Given quaternions `q1` and `q2`
/// ```text
/// norm(sqrt(q1 * q2)) == norm(sqrt(q2)) * norm(sqrt(q2))
/// ```
fn prop_approx_square_root_product_norm<S>(q1: Quaternion<S>, q2: Quaternion<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{   
    let lhs = (q1 * q2).sqrt().norm();
    let rhs = q1.sqrt().norm() * q2.sqrt().norm();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The principal argument of a quaternion is in the range `[-pi, pi]`.
/// 
/// Given a quaternion `q`
/// ```text
/// -pi =< arg(q) <= pi
/// ```
fn prop_square_root_arg_range<S>(q: Quaternion<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let arg_q = q.arg();

    prop_assert!(arg_q >= -S::pi());
    prop_assert!(arg_q <= S::pi());

    Ok(())
}


#[cfg(test)]
mod quaternion_f64_arithmetic_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_zero_times_quaternion_equals_zero(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_zero_times_quaternion_equals_zero(q)?
        }
        
        #[test]
        fn prop_quaternion_times_zero_equals_zero(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_times_zero_equals_zero(q)?
        }

        #[test]
        fn prop_quaternion_plus_zero_equals_quaternion(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_plus_zero_equals_quaternion(q)?
        }

        #[test]
        fn prop_zero_plus_quaternion_equals_quaternion(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_zero_plus_quaternion_equals_quaternion(q)?
        }

        #[test]
        fn prop_one_times_quaternion_equals_quaternion(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_one_times_quaternion_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion_times_one_equals_quaternion(q in super::strategy_quaternion_f64_any()) {
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
        fn prop_zero_times_quaternion_equals_zero(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_zero_times_quaternion_equals_zero(q)?
        }
        
        #[test]
        fn prop_quaternion_times_zero_equals_zero(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_times_zero_equals_zero(q)?
        }

        #[test]
        fn prop_quaternion_plus_zero_equals_quaternion(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_plus_zero_equals_quaternion(q)?
        }

        #[test]
        fn prop_zero_plus_quaternion_equals_quaternion(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_zero_plus_quaternion_equals_quaternion(q)?
        }

        #[test]
        fn prop_one_times_quaternion_equals_quaternion(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_one_times_quaternion_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion_times_one_equals_quaternion(q in super::strategy_quaternion_i32_any()) {
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
        fn prop_quaternion_plus_zero_equals_quaternion(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_plus_zero_equals_quaternion(q)?
        }

        #[test]
        fn prop_zero_plus_quaternion_equals_quaternion(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_zero_plus_quaternion_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(
            q1 in super::strategy_quaternion_f64_any(), 
            q2 in super::strategy_quaternion_f64_any()
        ) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(q1, q2)?
        }

        #[test]
        fn prop_quaternion_addition_commutative(
            q1 in super::strategy_quaternion_f64_any(), 
            q2 in super::strategy_quaternion_f64_any()
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
        fn prop_quaternion_plus_zero_equals_quaternion(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_plus_zero_equals_quaternion(q)?
        }

        #[test]
        fn prop_zero_plus_quaternion_equals_quaternion(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_zero_plus_quaternion_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(
            q1 in super::strategy_quaternion_i32_any(), 
            q2 in super::strategy_quaternion_i32_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(q1, q2)?
        }

        #[test]
        fn prop_quaternion_addition_commutative(
            q1 in super::strategy_quaternion_i32_any(), 
            q2 in super::strategy_quaternion_i32_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_quaternion_addition_commutative(q1, q2)?
        }

        #[test]
        fn prop_quaternion_addition_associative(
            q1 in super::strategy_quaternion_i32_any(), 
            q2 in super::strategy_quaternion_i32_any(), 
            q3 in super::strategy_quaternion_i32_any()
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
        fn prop_quaternion_minus_zero_equals_quaternion(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_minus_zero_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion_minus_quaternion_equals_zero(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_minus_quaternion_equals_zero(q)?
        }

        #[test]
        fn prop_quaternion1_minus_quaternion2_equals_refquaternion1_minus_refquaternion2(
            q1 in super::strategy_quaternion_f64_any(), 
            q2 in super::strategy_quaternion_f64_any()
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
        fn prop_quaternion_minus_zero_equals_quaternion(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_minus_zero_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion_minus_quaternion_equals_zero(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_minus_quaternion_equals_zero(q)?
        }

        #[test]
        fn prop_quaternion1_minus_quaternion2_equals_refquaternion1_minus_refquaternion2(
            q1 in super::strategy_quaternion_i32_any(), 
            q2 in super::strategy_quaternion_i32_any()
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
            q in super::strategy_quaternion_f64_any()
        ) {
            let c: f64 = c;
            let q: super::Quaternion<f64> = q;
            super::prop_scalar_times_quaternion_equals_quaternion_times_scalar(c, q)?
        }

        #[test]
        fn prop_quaternion_multiplicative_unit(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_multiplicative_unit(q)?
        }

        #[test]
        fn prop_approx_quaternion_multiplicative_inverse(q in super::strategy_quaternion_f64_any()) {
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
            q in super::strategy_quaternion_i32_any()
        ) {
            let c: i32 = c;
            let q: super::Quaternion<i32> = q;
            super::prop_scalar_times_quaternion_equals_quaternion_times_scalar(c, q)?
        }

        #[test]
        fn prop_scalar_multiplication_compatibility(
            a in super::strategy_scalar_i32_any(), 
            b in super::strategy_scalar_i32_any(), 
            q in super::strategy_quaternion_i32_any()
        ) {
            let a: i32 = a;
            let b: i32 = b;
            let q: super::Quaternion<i32> = q;
            super::prop_scalar_multiplication_compatibility(a, b, q)?
        }

        #[test]
        fn prop_quaternion_multiplication_associative(
            q1 in super::strategy_quaternion_i32_any(), 
            q2 in super::strategy_quaternion_i32_any(), 
            q3 in super::strategy_quaternion_i32_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            let q3: super::Quaternion<i32> = q3;
            super::prop_quaternion_multiplication_associative(q1, q2, q3)?
        }

        #[test]
        fn prop_quaternion_multiplicative_unit(q in super::strategy_quaternion_i32_any()) {
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
            q1 in super::strategy_quaternion_i32_any(), 
            q2 in super::strategy_quaternion_i32_any()
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
            q in super::strategy_quaternion_i32_any()
        ) {
            let a: i32 = a;
            let b: i32 = b;
            let q: super::Quaternion<i32> = q;
            super::prop_distribution_over_scalar_addition(a, b, q)?
        }

        #[test]
        fn prop_distribution_over_quaternion_addition1(
            a in super::strategy_scalar_i32_any(), 
            q1 in super::strategy_quaternion_i32_any(), 
            q2 in super::strategy_quaternion_i32_any()
        ) {
            let a: i32 = a;
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_distribution_over_quaternion_addition1(a, q1, q2)?
        }

        #[test]
        fn prop_quaternion_multiplication_right_distributive(
            q1 in super::strategy_quaternion_i32_any(), 
            q2 in super::strategy_quaternion_i32_any(), 
            q3 in super::strategy_quaternion_i32_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            let q3: super::Quaternion<i32> = q3;
            super::prop_quaternion_multiplication_right_distributive(q1, q2, q3)?
        }

        #[test]
        fn prop_quaternion_multiplication_left_distributive(
            q1 in super::strategy_quaternion_i32_any(), 
            q2 in super::strategy_quaternion_i32_any(), 
            q3 in super::strategy_quaternion_i32_any()
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
        fn prop_quaternion_dot_product_commutative(
            q1 in super::strategy_quaternion_i32_any(), 
            q2 in super::strategy_quaternion_i32_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_quaternion_dot_product_commutative(q1, q2)?
        }

        #[test]
        fn prop_quaternion_dot_product_right_distributive(
            q1 in super::strategy_quaternion_i32_any(),
            q2 in super::strategy_quaternion_i32_any(), 
            q3 in super::strategy_quaternion_i32_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            let q3: super::Quaternion<i32> = q3;
            super::prop_quaternion_dot_product_right_distributive(q1, q2, q3)?
        }

        #[test]
        fn prop_quaternion_dot_product_left_distributive(
            q1 in super::strategy_quaternion_i32_any(),
            q2 in super::strategy_quaternion_i32_any(), 
            q3 in super::strategy_quaternion_i32_any()
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
            q1 in super::strategy_quaternion_i32_any(), 
            q2 in super::strategy_quaternion_i32_any()
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
            q1 in super::strategy_quaternion_i32_any(),
            q2 in super::strategy_quaternion_i32_any(), 
            q3 in super::strategy_quaternion_i32_any()
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
            q1 in super::strategy_quaternion_i32_any(),
            q2 in super::strategy_quaternion_i32_any(), 
            q3 in super::strategy_quaternion_i32_any()
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
        fn prop_quaternion_conjugate_conjugate_equals_quaternion(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_quaternion_conjugate_conjugate_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion_conjugation_linear(q1 in super::strategy_quaternion_f64_any(), q2 in super::strategy_quaternion_f64_any()) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_quaternion_conjugation_linear(q1, q2)?
        }

        #[test]
        fn prop_quaternion_conjugation_transposes_products(
            q1 in super::strategy_quaternion_f64_any(), 
            q2 in super::strategy_quaternion_f64_any()
        ) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_quaternion_conjugation_transposes_products(q1, q2)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_conjugation_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_quaternion_conjugate_conjugate_equals_quaternion(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_quaternion_conjugate_conjugate_equals_quaternion(q)?
        }

        #[test]
        fn prop_quaternion_conjugation_linear(q1 in super::strategy_quaternion_i32_any(), q2 in super::strategy_quaternion_i32_any()) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_quaternion_conjugation_linear(q1, q2)?
        }

        #[test]
        fn prop_quaternion_conjugation_transposes_products(
            q1 in super::strategy_quaternion_i32_any(), 
            q2 in super::strategy_quaternion_i32_any()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_quaternion_conjugation_transposes_products(q1, q2)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_modulus_squared_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_modulus_squared_nonnegative(q in super::strategy_quaternion_f64_norm_squared()) {
            let q: super::Quaternion<f64> = q;
            super::prop_modulus_squared_nonnegative(q)?
        }

        #[test]
        fn prop_approx_modulus_squared_point_separating(
            q1 in super::strategy_quaternion_f64_norm_squared(), 
            q2 in super::strategy_quaternion_f64_norm_squared()
        ) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_approx_modulus_squared_point_separating(q1, q2, 1e-10, 1e-20)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_modulus_squared_synonym_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_norm_squared_modulus_squared(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_norm_squared_modulus_squared(q)?
        }
        #[test]
        fn prop_magnitude_squared_modulus_squared(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_magnitude_squared_modulus_squared(q)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_modulus_squared_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_modulus_squared_nonnegative(q in super::strategy_quaternion_i32_norm_squared()) {
            let q: super::Quaternion<i32> = q;
            super::prop_modulus_squared_nonnegative(q)?
        }

        #[test]
        fn prop_modulus_squared_point_separating(
            q1 in super::strategy_quaternion_i32_norm_squared(),
            q2 in super::strategy_quaternion_i32_norm_squared()
        ) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_modulus_squared_point_separating(q1, q2)?
        }

        #[test]
        fn prop_modulus_squared_homogeneous_squared(
            q in super::strategy_quaternion_i32_norm_squared(),
            c in super::strategy_scalar_i32_any()
        ) {
            let q: super::Quaternion<i32> = q;
            let c: i32 = c;
            super::prop_modulus_squared_homogeneous_squared(q, c)?
        }
    }
}


#[cfg(test)]
mod quaternion_i32_modulus_squared_synonym_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_norm_squared_modulus_squared(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_norm_squared_modulus_squared(q)?
        }
        
        #[test]
        fn prop_magnitude_squared_modulus_squared(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_magnitude_squared_modulus_squared(q)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_modulus_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_modulus_nonnegative(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_modulus_nonnegative(q)?
        }

        #[test]
        fn prop_approx_modulus_point_separating(q1 in super::strategy_quaternion_f64_any(), q2 in super::strategy_quaternion_f64_any()) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_approx_modulus_point_separating(q1, q2, 1e-10)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_l1_norm_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_l1_norm_nonnegative(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_l1_norm_nonnegative(q)?
        }

        #[test]
        fn prop_approx_l1_norm_point_separating(q1 in super::strategy_quaternion_f64_any(), q2 in super::strategy_quaternion_f64_any()) {
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
        fn prop_l1_norm_nonnegative(q in super::strategy_quaternion_i32_any()) {
            let q: super::Quaternion<i32> = q;
            super::prop_l1_norm_nonnegative(q)?
        }

        #[test]
        fn prop_l1_norm_point_separating(q1 in super::strategy_quaternion_i32_any(), q2 in super::strategy_quaternion_i32_any()) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_l1_norm_point_separating(q1, q2)?
        }

        #[test]
        fn prop_l1_norm_homogeneous(q in super::strategy_quaternion_i32_any(), c in super::strategy_scalar_i32_any()) {
            let q: super::Quaternion<i32> = q;
            let c: i32 = c;
            super::prop_l1_norm_homogeneous(q, c)?
        }

        #[test]
        fn prop_l1_norm_triangle_inequality(q1 in super::strategy_quaternion_i32_any(), q2 in super::strategy_quaternion_i32_any()) {
            let q1: super::Quaternion<i32> = q1;
            let q2: super::Quaternion<i32> = q2;
            super::prop_l1_norm_triangle_inequality(q1, q2)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_modulus_synonym_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_norm_modulus_synonyms(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_norm_modulus_synonyms(q)?
        }
        #[test]
        fn prop_magnitude_modulus_synonyms(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_magnitude_modulus_synonyms(q)?
        }

        #[test]
        fn prop_l2_norm_modulus_synonyms(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_l2_norm_modulus_synonyms(q)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_exp_prop {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_exp_scalar_vector_sum(q in super::strategy_quaternion_f64_exp()) {
            let q: super::Quaternion<f64> = q;
            super::prop_approx_exp_scalar_vector_sum(q, 1e-10, 1e-12)?
        }

        #[test]
        fn prop_exp_quaternion_nonzero(q in super::strategy_quaternion_f64_exp()) {
            let q: super::Quaternion<f64> = q;
            super::prop_exp_quaternion_nonzero(q)?
        }

        #[test]
        fn prop_approx_exp_quaternion_exp_negative_quaternion(q in super::strategy_quaternion_f64_exp()) {
            let q: super::Quaternion<f64> = q;
            super::prop_approx_exp_quaternion_exp_negative_quaternion(q, 1e-10)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_ln_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_quaternion_ln_scalar_part(q in super::strategy_quaternion_f64_exp()) {
            let q: super::Quaternion<f64> = q;
            super::prop_approx_quaternion_ln_scalar_part(q, 1e-10)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_exp_ln_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_exp_ln_identity(q in super::strategy_quaternion_f64_exp()) {
            let q: super::Quaternion<f64> = q;
            super::prop_approx_exp_ln_identity(q, 1e-4, 1e-4)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_arg_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_arg_congruent(q in super::strategy_quaternion_f64_any(), k in super::strategy_scalar_i32_any()) {
            let q: super::Quaternion<f64> = q;
            let k: i32 = k;
            super::prop_approx_arg_congruent(q, k, 1e-10)?
        }

        #[test]
        fn prop_approx_arg_range(q in super::strategy_quaternion_f64_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_approx_arg_range(q)?
        }
    }
}


#[cfg(test)]
mod quaternion_f64_sqrt_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_square_root_quaternion_squared(q in super::strategy_quaternion_squared_any()) {
            let q: super::Quaternion<f64> = q;
            super::prop_approx_square_root_quaternion_squared(q, 1e-8, 1e-8)?
        }

        /*
        #[test]
        fn prop_approx_square_root_negative_quaternion_squared(q in super::strategy_quaternion_squared_z_axis()) {
            let q: super::Quaternion<f64> = q;
            super::prop_approx_square_root_negative_quaternion_squared(q, 1e-6, 1e-6)?
        }
        */

        #[test]
        fn prop_approx_square_root_product_norm(q1 in super::strategy_quaternion_f64_sqrt_product(), q2 in super::strategy_quaternion_f64_sqrt_product()) {
            let q1: super::Quaternion<f64> = q1;
            let q2: super::Quaternion<f64> = q2;
            super::prop_approx_square_root_product_norm(q1, q2, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_square_root_arg_range(q in super::strategy_quaternion_f64_sqrt()) {
            let q: super::Quaternion<f64> = q;
            super::prop_square_root_arg_range(q)?
        }
    }
}

