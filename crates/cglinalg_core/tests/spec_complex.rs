extern crate cglinalg_core;
extern crate num_traits;
extern crate proptest;


use proptest::prelude::*;
use cglinalg_core::{
    Complex, 
    SimdScalar,
    SimdScalarSigned,
    SimdScalarFloat,
};
use approx::{
    relative_eq,
    relative_ne,
};


fn strategy_complex_polar_from_range<S>(min_scale: S, max_scale: S, min_angle: S, max_angle: S) -> impl Strategy<Value = Complex<S>>
where
    S: SimdScalarFloat + Arbitrary
{
    use cglinalg_core::Radians;

    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarFloat
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, S)>().prop_map(move |(_scale, _angle)| {
        let scale = SimdScalarSigned::abs(rescale(_scale, min_scale, max_scale));
        let angle = Radians(SimdScalarSigned::abs(rescale(_angle, min_angle, max_angle)));

        Complex::from_polar_decomposition(scale, angle)
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

fn strategy_complex_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Complex<S>>
where
    S: SimdScalarSigned + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarSigned
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, S)>().prop_map(move |(_re, _im)| {
        let sign_re = _re.signum();
        let sign_im = _im.signum();
        let abs_re = _re.abs();
        let abs_im = _im.abs();
        let re = sign_re * rescale(abs_re, min_value, max_value);
        let im = sign_im * rescale(abs_im, min_value, max_value);
        
        Complex::new(re, im)
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
    // let max_value = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
    let max_value = 46340_i32;

    strategy_scalar_signed_from_abs_range(min_value, max_value)
}

fn strategy_scalar_i32_power() -> impl Strategy<Value = i32> {
    let min_value = 0_i32;
    // let max_value = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
    let max_value = 100_i32;

    strategy_scalar_signed_from_abs_range(min_value, max_value)
}

fn strategy_complex_f64_any() -> impl Strategy<Value = Complex<f64>> {
    let min_value = f64::sqrt(f64::EPSILON);
    let max_value = f64::sqrt(f64::MAX) / f64::sqrt(2_f64);

    strategy_complex_signed_from_abs_range(min_value, max_value)
}

fn strategy_complex_i32_any() -> impl Strategy<Value = Complex<i32>> {
    let min_value = 0_i32;
    // let max_value = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
    let max_value = 46340_i32;

    strategy_complex_signed_from_abs_range(min_value, max_value)
}

fn strategy_complex_f64_modulus_squared() -> impl Strategy<Value = Complex<f64>> {
    let min_scale = f64::sqrt(f64::EPSILON);
    let max_scale = f64::sqrt(f64::MAX) / f64::sqrt(2_f64);
    let min_angle = 0_f64;
    let max_angle = f64::two_pi();

    strategy_complex_polar_from_range(min_scale, max_scale, min_angle, max_angle)
}

fn strategy_complex_i32_modulus_squared() -> impl Strategy<Value = Complex<i32>> {
    let min_value = 0_i32;
    // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
    let max_square_root = 46340_i32;
    let max_value = max_square_root / 2;

    strategy_complex_signed_from_abs_range(min_value, max_value)
}

fn strategy_imaginary_from_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Complex<S>>
where 
    S: SimdScalarFloat + Arbitrary
{
    fn rescale<S: SimdScalarFloat>(value: S, min_value: S, max_value: S) -> S {
        min_value + (value % (max_value - min_value))
    }

    any::<S>().prop_map(move |im| {
        Complex::from_imaginary(rescale(im, min_value, max_value))
    })
    .no_shrink()
}

fn strategy_real_from_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Complex<S>>
where 
    S: SimdScalarFloat + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarFloat
    {
        min_value + (value % (max_value - min_value))
    }

    any::<S>().prop_map(move |re| {
        Complex::from_real(rescale(re, min_value, max_value))
    })
    .no_shrink()
}

fn strategy_complex_f64_exp() -> impl Strategy<Value = Complex<f64>> {
    let min_scale = f64::ln(f64::EPSILON);
    let max_scale = f64::ln(f64::MAX) / 4_f64;
    let min_angle = 0_f64;
    let max_angle = f64::two_pi();

    strategy_complex_polar_from_range(min_scale, max_scale, min_angle, max_angle)
}

fn strategy_complex_f64_ln() -> impl Strategy<Value = Complex<f64>> {
    let min_scale = f64::ln(f64::EPSILON);
    let max_scale = f64::ln(f64::MAX) / 4_f64;
    let min_angle = 0_f64;
    let max_angle = f64::two_pi();

    strategy_complex_polar_from_range(min_scale, max_scale, min_angle, max_angle)
}

fn strategy_complex_f64_sqrt() -> impl Strategy<Value = Complex<f64>> {
    let min_scale = f64::EPSILON;
    let max_scale = f64::sqrt(f64::MAX) / f64::sqrt(2_f64);
    let min_angle = 0_f64;
    let max_angle = f64::two_pi();

    strategy_complex_polar_from_range(min_scale, max_scale, min_angle, max_angle)
}

fn strategy_complex_f64_sqrt_product() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, f64::sqrt(f64::sqrt(f64::MAX)) / f64::sqrt(2_f64), 0_f64, f64::two_pi())
}

fn strategy_complex_f64_cbrt() -> impl Strategy<Value = Complex<f64>> {
    let min_scale = f64::EPSILON;
    let max_scale = f64::cbrt(f64::MAX) / f64::cbrt(2_f64);
    let min_angle = 0_f64;
    let max_angle = f64::two_pi();

    strategy_complex_polar_from_range(min_scale, max_scale, min_angle, max_angle)
}

fn strategy_imaginary_f64_cos() -> impl Strategy<Value = Complex<f64>>{
    strategy_imaginary_from_range(f64::EPSILON, f64::ln(f64::MAX))
}

fn strategy_imaginary_f64_sin() -> impl Strategy<Value = Complex<f64>>{
    strategy_imaginary_from_range(f64::EPSILON, f64::ln(f64::MAX))
}

fn strategy_complex_f64_tan() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::two_pi())
}

fn strategy_real_f64_tan() -> impl Strategy<Value = Complex<f64>> {
    strategy_real_from_range(f64::EPSILON, 100_f64)
}

fn strategy_imaginary_f64_tan() -> impl Strategy<Value = Complex<f64>> {
    strategy_imaginary_from_range(f64::EPSILON, 100_f64)
}

fn strategy_complex_f64_cos_double_angle() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::two_pi())
}

fn strategy_complex_f64_sin_double_angle() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::two_pi())
}

fn strategy_complex_f64_tan_double_angle() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::two_pi())
}

fn strategy_complex_f64_cos_angle_sum() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::pi())
}

fn strategy_complex_f64_sin_angle_sum() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::pi())
}

fn strategy_complex_f64_tan_angle_sum() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::two_pi())
}

fn strategy_complex_f64_tan_angle_difference() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::two_pi())
}

fn strategy_complex_f64_cosh() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, f64::ln(f64::MAX), 0_f64, f64::pi())
}

fn strategy_complex_f64_sinh() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, f64::ln(f64::MAX), 0_f64, f64::pi())
}

fn strategy_complex_f64_tanh() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::two_pi())
}

fn strategy_complex_f64_cosh_double_angle() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::two_pi())
}

fn strategy_complex_f64_sinh_double_angle() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::two_pi())
}

fn strategy_complex_f64_tanh_double_angle() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::two_pi())
}

fn strategy_complex_f64_cosh_angle_sum() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::frac_pi_2())
}

fn strategy_complex_f64_sinh_angle_sum() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::frac_pi_2())
}

fn strategy_complex_f64_tanh_angle_sum() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::pi())
}

fn strategy_complex_f64_tanh_angle_difference() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::two_pi())
}

fn strategy_complex_f64_acosh() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, f64::ln(f64::MAX), 0_f64, f64::frac_pi_2())
}

fn strategy_complex_f64_asinh() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, f64::ln(f64::MAX), 0_f64, f64::frac_pi_2())
}

fn strategy_complex_f64_atanh() -> impl Strategy<Value = Complex<f64>> {
    strategy_complex_polar_from_range(f64::EPSILON, 100_f64, 0_f64, f64::frac_pi_2())
}


/// A scalar `0` times a complex number should be a zero complex number.
///
/// Given a complex number `z`, it satisfies
/// ```text
/// 0 * z == 0.
/// ```
fn prop_zero_times_complex_equals_zero<S>(z: Complex<S>) -> Result<(), TestCaseError>
where 
    S: SimdScalar
{
    let zero_complex = Complex::zero();

    prop_assert_eq!(zero_complex * z, zero_complex);
    
    Ok(())
}

/// A scalar `0` times a complex number should be zero.
///
/// Given a complex number `z`, it satisfies
/// ```text
/// z * 0 == 0
/// ```
fn prop_complex_times_zero_equals_zero<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let zero = S::zero();
    let zero_complex = Complex::zero();

    prop_assert_eq!(z * zero, zero_complex);
    
    Ok(())
}

/// A zero complex number should act as the additive unit element of a set 
/// of complex numbers.
///
/// Given a complex number `z`
/// ```text
/// z + 0 == z
/// ```
fn prop_complex_plus_zero_equals_complex<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let zero_complex = Complex::zero();

    prop_assert_eq!(z + zero_complex, z);
    
    Ok(())
}

/// A zero complex number should act as the additive unit element of a set 
/// of complex numbers.
///
/// Given a complex number `z`
/// ```text
/// 0 + z == z
/// ```
fn prop_zero_plus_complex_equals_complex<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let zero_complex = Complex::zero();

    prop_assert_eq!(zero_complex + z, z);
    
    Ok(())
}

/// Multiplying a complex number by a scalar `1` should give the original 
/// complex number.
///
/// Given a complex number `z`
/// ```text
/// 1 * z == z
/// ```
fn prop_one_times_complex_equal_complex<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let one = Complex::one();

    prop_assert_eq!(one * z, z);
    
    Ok(())
}

/// Multiplying a complex number by a scalar `1` should give the original 
/// complex number.
///
/// Given a complex number `z`
/// ```text
/// z * 1 == z.
/// ```
fn prop_complex_times_one_equals_complex<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let one = S::one();

    prop_assert_eq!(z * one, z);
    
    Ok(())
}

/// Given complex numbers `z1` and `z2`, we should be able to use `z1` 
/// and `z2` interchangeably with their references `&z1` and `&z2` in 
/// arithmetic expressions involving complex numbers.
///
/// Given complex numbers `z1` and `z2`, and their references `&z1` 
/// and `&z2`, they should satisfy
/// ```text
///  z1 +  z2 == &z1 +  z2
///  z1 +  z2 ==  z1 + &z2
///  z1 +  z2 == &z1 + &z2
///  z1 + &z2 == &z1 +  z2
/// &z1 +  z2 ==  z1 + &z2
/// &z1 +  z2 == &z1 + &z2
///  z1 + &z2 == &z1 + &z2
/// ```
fn prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex2<S>(z1: Complex<S>, z2: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!( z1 +  z2, &z1 +  z2);
    prop_assert_eq!( z1 +  z2,  z1 + &z2);
    prop_assert_eq!( z1 +  z2, &z1 + &z2);
    prop_assert_eq!( z1 + &z2, &z1 +  z2);
    prop_assert_eq!(&z1 +  z2,  z1 + &z2);
    prop_assert_eq!(&z1 +  z2, &z1 + &z2);
    prop_assert_eq!( z1 + &z2, &z1 + &z2);
    
    Ok(())
}

/// Complex number addition over floating point scalars should be commutative.
/// 
/// Given complex numbers `z1` and `z2`, we have
/// ```text
/// z1 + z2 == z2 + z1
/// ```
fn prop_complex_addition_commutative<S>(z1: Complex<S>, z2: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(z1 + z2, z2 + z1);

    Ok(())
}

/// Given three complex numbers of integer scalars, complex number addition 
/// should be associative.
///
/// Given complex numbers `z1`, `z2`, and `z3`, we have
/// ```text
/// (z1 + z2) + z3 == z1 + (z2 + z3)
/// ```
fn prop_complex_addition_associative<S>(z1: Complex<S>, z2: Complex<S>, z3: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!((z1 + z2) + z3, z1 + (z2 + z3));

    Ok(())
}

/// The zero complex number should act as an additive unit. 
///
/// Given a complex number `z`, we have
/// ```text
/// z - 0 == z
/// ```
fn prop_complex_minus_zero_equals_complex<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let zero_complex = Complex::zero();

    prop_assert_eq!(z - zero_complex, z);

    Ok(())
}

/// Every complex number should have an additive inverse. 
///
/// Given a complex number `z`, there is a complex number `-z` such that
/// ```text
/// z - z == z + (-z) == (-z) + z == 0
/// ```
fn prop_complex_minus_complex_equals_zero<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let zero_complex = Complex::zero();

    prop_assert_eq!(z - z, zero_complex);

    Ok(())
}

/// Given complex numbers `z1` and `z2`, we should be able to use `z1` 
/// and `z2` interchangeably with their references `&z1` and `&z2` 
/// in arithmetic expressions involving complex numbers.
///
/// Given complex numbers `z1` and `z2`, and their references `&z1` 
/// and `&z2`, they should satisfy
/// ```text
///  z1 -  z2 == &z1 -  z2
///  z1 -  z2 ==  z1 - &z2
///  z1 -  z2 == &z1 - &z2
///  z1 - &z2 == &z1 -  z2
/// &z1 -  z2 ==  z1 - &z2
/// &z1 -  z2 == &z1 - &z2
///  z1 - &z2 == &z1 - &z2
/// ```
fn prop_complex1_minus_complex2_equals_refcomplex1_minus_refcomplex2<S>(z1: Complex<S>, z2: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!( z1 -  z2, &z1 -  z2);
    prop_assert_eq!( z1 -  z2,  z1 - &z2);
    prop_assert_eq!( z1 -  z2, &z1 - &z2);
    prop_assert_eq!( z1 - &z2, &z1 -  z2);
    prop_assert_eq!(&z1 -  z2,  z1 - &z2);
    prop_assert_eq!(&z1 -  z2, &z1 - &z2);
    prop_assert_eq!( z1 - &z2, &z1 - &z2);

    Ok(())
}

/// Multiplication of a scalar and a complex number should be commutative.
///
/// Given a constant `c` and a complex number `z`
/// ```text
/// c * z == z * c
/// ```
fn prop_scalar_times_complex_equals_complex_times_scalar<S>(c: S, z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{   
    let c_complex = Complex::from_real(c);

    prop_assert_eq!(c_complex * z, z * c_complex);

    Ok(())
}

/// Complexs have a multiplicative unit element.
///
/// Given a complex number `z`, and the unit complex number `1`, we have
/// ```text
/// z * 1 == 1 * z == z
/// ```
fn prop_complex_multiplicative_unit<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let one = Complex::one();

    prop_assert_eq!(z * one, z); 
    prop_assert_eq!(one * z, z);
    prop_assert_eq!(z * one, one * z);

    Ok(())
}

/// Every nonzero complex number over floating point scalars has an 
/// approximate multiplicative inverse.
///
/// Given a complex number `z` and its inverse `z_inv`, we have
/// ```text
/// z * z_inv == z_inv * z == 1
/// ```
fn prop_approx_complex_multiplicative_inverse<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assume!(z.is_finite());
    prop_assume!(z.is_invertible());
    let one = Complex::one();
    let z_inv = z.inverse().unwrap();

    prop_assert!(relative_eq!(z * z_inv, one, epsilon = tolerance));
    prop_assert!(relative_eq!(z_inv * z, one, epsilon = tolerance));

    Ok(())
}

/// Exact multiplication of two scalars and a complex number should be 
/// compatible with multiplication of all scalars. 
///
/// In other words, scalar multiplication of two scalars with a 
/// complex number should act associatively just like the multiplication 
/// of three scalars. 
///
/// Given scalars `a` and `b`, and a complex number `z`, observe that `a == a + i0` 
/// and `b == b + i0`. We have
/// ```text
/// (a * b) * z == a * (b * z)
/// z * (a * b) == (z * a) * b
/// ```
fn prop_scalar_multiplication_compatibility1<S>(a: S, b: S, z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let a_complex = Complex::from_real(a);
    let b_complex = Complex::from_real(b);

    prop_assert_eq!(a_complex * (b_complex * z), (a_complex * b_complex) * z);
    prop_assert_eq!(z * (a_complex * b_complex), (z * a_complex) * b_complex);

    Ok(())
}

/// Exact multiplication of two scalars and a complex number should be 
/// compatible with multiplication of all scalars. 
///
/// In other words, scalar multiplication of two scalars with a 
/// complex number should act associatively just like the multiplication 
/// of three scalars. 
///
/// Given scalars `a` and `b`, and a complex number `z`, we have
/// ```text
/// z * (a * b) == (z * a) * b
/// ```
fn prop_scalar_multiplication_compatibility2<S>(a: S, b: S, z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(z * (a * b), (z * a) * b);

    Ok(())
}

/// Complex number multiplication over integer scalars is exactly associative.
///
/// Given complex numbers `z1`, `z2`, and `z3`, we have
/// ```text
/// (z1 * z2) * z3 == z1 * (z2 * z3)
/// ```
fn prop_complex_multiplication_associative<S>(z1: Complex<S>, z2: Complex<S>, z3: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(z1 * (z2 * z3), (z1 * z2) * z3);

    Ok(())
}

/// Multiplication of complex numbers over integer scalars is commutative.
/// 
/// Given a complex number `z1` and another complex number `z2`, we have
/// ```text
/// z1 * z2 == z2 * z1
/// ```
fn prop_complex_multiplication_commutative<S>(z1: Complex<S>, z2: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(z1 * z2, z2 * z1);

    Ok(())
}

/// Scalar multiplication should distribute over complex number addition.
///
/// Given a scalar `a` and complex numbers `z1` and `z2`
/// ```text
/// a * (z1 + z2) == a * z1 + a * z2
/// ```
fn prop_distribution_over_complex_addition<S>(a: S, z1: Complex<S>, z2: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let a_complex = Complex::from_real(a);

    prop_assert_eq!(a_complex * (z1 + z2), a_complex * z1 + a_complex * z2);
    prop_assert_eq!((z1 + z2) * a_complex, z1 * a_complex + z2 * a_complex);

    Ok(())
}

/// Multiplication of a sum of scalars should distribute over a 
/// complex number.
///
/// Given scalars `a` and `b` and a complex number `z`, we have
/// ```text
/// (a + b) * z == a * z + b * z
/// ```
fn prop_distribution_over_scalar_addition<S>(a: S, b: S, z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let a_complex = Complex::from_real(a);
    let b_complex = Complex::from_real(b);

    prop_assert_eq!((a_complex + b_complex) * z, a_complex * z + b_complex * z);
    prop_assert_eq!(z * (a_complex + b_complex), z * a_complex + z * b_complex);

    Ok(())
}

/// Multiplication of two complex numbers by a scalar on the right 
/// should distribute.
///
/// Given complex numbers `z1` and `z2`, and a scalar `a`
/// ```text
/// (z1 + z2) * a == z1 * a + z2 * a
/// ```
fn prop_distribution_over_complex_addition1<S>(a: S, z1: Complex<S>, z2: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!((z1 + z2) * a, z1 * a + z2 * a);

    Ok(())
}

/// Multiplication of a complex number on the right by the sum of two 
/// scalars should distribute over the two scalars. 
///
/// Given a complex number `z` and scalars `a` and `b`
/// ```text
/// z * (a + b) == z * a + z * b
/// ```
fn prop_distribution_over_scalar_addition1<S>(a: S, b: S, z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(z * (a + b), z * a + z * b);

    Ok(())
}

/// Complex number multiplication should be distributive on the right.
///
/// Given three complex numbers `z1`, `z2`, and `z3`
/// ```text
/// (z1 + z2) * z3 == z1 * z3 + z2 * z3
/// ```
fn prop_complex_multiplication_right_distributive<S>(z1: Complex<S>, z2: Complex<S>, z3: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!((z1 + z2) * z3, z1 * z3 + z2 * z3);

    Ok(())
}

/// Complex Number multiplication should be distributive on the left.
///
/// Given three complex numbers `z1`, `z2`, and `z3`
/// ```text
/// z1 * (z2 + z3) == z1 * z2 + z1 * z3
/// ```
fn prop_complex_multiplication_left_distributive<S>(z1: Complex<S>, z2: Complex<S>, z3: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!((z1 + z2) * z3, z1 * z3 + z2 * z3);

    Ok(())
}

/// Conjugating a complex number twice should give the original complex number.
///
/// Given a complex number `z`
/// ```text
/// conjugate(conjugate(z)) == z
/// ```
fn prop_complex_conjugate_conjugate_equals_complex<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    prop_assert_eq!(z.conjugate().conjugate(), z);

    Ok(())
}

/// Complex conjugation is linear.
///
/// Given complex numbers `z1` and `z2`, complex number conjugation satisfies
/// ```text
/// conjugate(z1 + z2) == conjugate(z1) + conjugate(z2)
/// ```
fn prop_complex_conjugation_linear<S>(z1: Complex<S>, z2: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    prop_assert_eq!((z1 + z2).conjugate(), z1.conjugate() + z2.conjugate());

    Ok(())
}

/// Complex multiplication transposes under conjugation.
///
/// Given complex numbers `z1` and `z2`
/// ```text
/// conjugate(z1 * z2) == conjugate(z2) * conjugate(z1)
/// ```
fn prop_complex_conjugation_transposes_products<S>(z1: Complex<S>, z2: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    prop_assert_eq!((z1 * z2).conjugate(), z2.conjugate() * z1.conjugate());

    Ok(())
}

/// The squared modulus of a complex number is nonnegative. 
///
/// Given a complex number `z`
/// ```text
/// modulus_squared(z) >= 0
/// ```
fn prop_modulus_squared_nonnegative<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let zero = S::zero();

    prop_assert!(z.modulus_squared() >= zero);
    
    Ok(())
}

/// The squared modulus function is point separating. In particular, if 
/// the squared distance between two complex numbers `z1` and `z2` is 
/// zero, then `z1 == z2`.
///
/// Given complex numbers `z1` and `z2`
/// ```text
/// modulus_squared(z1 - z2) == 0 => z1 == z2 
/// ```
/// Equivalently, if `z1` is not equal to `z2`, then their squared distance is 
/// nonzero
/// ```text
/// z1 != z2 => modulus_squared(z1 - z2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_approx_modulus_squared_point_separating<S>(
    z1: Complex<S>, 
    z2: Complex<S>, 
    input_tolerance: S, 
    output_tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(z1, z2, epsilon = input_tolerance));
    prop_assert!((z1 - z2).modulus_squared() > output_tolerance);

    Ok(())
}

/// The [`Complex::magnitude_squared`] function and the [`Complex::modulus_squared`]
/// function are synonyms. In particular, given a complex number `z`
/// ```text
/// magnitude_squared(z) == modulus_squared(z)
/// ```
/// where equality is exact.
fn prop_magnitude_squared_modulus_squared_synonyms<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(z.magnitude_squared(), z.modulus_squared());

    Ok(())
}

/// The [`Complex::norm_squared`] function and the [`Complex::modulus_squared`]
/// functions are synonyms. In particular, given a complex number `z`
/// ```text
/// norm_squared(z) == modulus_squared(z)
/// ```
fn prop_norm_squared_modulus_squared_synonyms<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!(z.norm_squared(), z.modulus_squared());

    Ok(())
}

/// The squared modulus function is point separating. In particular, if 
/// the squared distance between two complex numbers `z1` and `z2` is 
/// zero, then `z1 == z2`.
///
/// Given complex numbers `z1` and `z2`
/// ```text
/// modulus_squared(z1 - z2) == 0 => z1 == z2 
/// ```
/// Equivalently, if `z1` is not equal to `z2`, then their squared distance is 
/// nonzero
/// ```text
/// z1 != z2 => modulus_squared(z1 - z2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_modulus_squared_point_separating<S>(z1: Complex<S>, z2: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let zero = S::zero();

    // prop_assume!(z1 != z2);
    prop_assert_ne!((z1 - z2).modulus_squared(), zero);

    Ok(())
}

/// The squared modulus function is homogeneous.
/// 
/// Given a complex number `z` and a scalar `c`
/// ```text
/// modulus_squared(z * c) == modulus_squared(z) * abs(c) * abs(c)
/// ```
fn prop_modulus_squared_homogeneous_squared<S>(z: Complex<S>, c: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarSigned
{
    let lhs = (z * c).modulus_squared();
    let rhs = z.modulus_squared() * c.abs() * c.abs();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The modulus of a complex number is nonnegative. 
///
/// Given a complex number `z`
/// ```text
/// modulus(z) >= 0
/// ```
fn prop_modulus_nonnegative<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let zero = S::zero();

    prop_assert!(z.modulus() >= zero);

    Ok(())
}

/// The modulus function is point separating. In particular, if 
/// the distance between two complex numbers `z1` and `z2` is 
/// zero, then `z1 == z2`.
///
/// Given complex numbers `z1` and `z2`
/// ```text
/// modulus(z1 - z2) == 0 => z1 == z2 
/// ```
/// Equivalently, if `z1` is not equal to `z2`, then their distance is 
/// nonzero
/// ```text
/// z1 != z2 => modulus(z1 - z2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_approx_modulus_point_separating<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let zero = S::zero();

    prop_assume!(relative_ne!(z1, z2, epsilon = tolerance));
    prop_assert!(relative_ne!((z1 - z2).modulus(), zero, epsilon = tolerance));

    Ok(())
}

/// The [`Complex::magnitude`] function and the [`Complex::modulus`] function 
/// are synonyms. In particular, given a complex number `z`
/// ```text
/// magnitude(z) == norm(z)
/// ```
/// where equality is exact.
fn prop_magnitude_modulus_synonyms<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert_eq!(z.magnitude(), z.modulus());

    Ok(())
}

/// The [`Complex::norm`] function and the [`Complex::modulus`] function
/// are synonyms. In particular, given a complex number `z`
/// ```text
/// norm(z) == modulus(z)
/// ```
/// where equality is exact.
fn prop_norm_modulus_synonyms<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert_eq!(z.norm(), z.modulus());

    Ok(())
}

/// The [`Complex::l2_norm`] function and the [`Complex::modulus`] function
/// are synonyms. In particular, given a complex number `z`
/// ```text
/// l2_norm(z) == modulus(z)
/// ```
/// where equality is exact.
fn prop_l2_norm_modulus_synonyms<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert_eq!(z.l2_norm(), z.modulus());

    Ok(())
}

/// The **L1** norm of a complex number is nonnegative. 
///
/// Given a complex number `z`
/// ```text
/// l1_norm(z) >= 0
/// ```
fn prop_l1_norm_nonnegative<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let zero = S::zero();

    prop_assert!(z.l1_norm() >= zero);

    Ok(())
}

/// The **L1** norm function is homogeneous. 
/// 
/// Given a complex number `z` and a scalar `c`
/// ```text
/// l1_norm(z * c) == l1_norm(z) * abs(c)
/// ```
fn prop_l1_norm_homogeneous<S>(z: Complex<S>, c: S) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = (z * c).l1_norm();
    let rhs = z.l1_norm() * c.abs();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The **L1** norm satisfies the triangle inequality.
/// 
/// Given complex numbers `z1` and `z2`
/// ```text
/// l1_norm(z1 + z2) <= l1_norm(z1) + l1_norm(z2)
/// ```
fn prop_l1_norm_triangle_inequality<S>(z1: Complex<S>, z2: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = (z1 + z2).l1_norm();
    let rhs = z1.l1_norm() + z2.l1_norm();

    prop_assert!(lhs <= rhs);

    Ok(())
}

/// The **L1** norm function is point separating. In particular, if 
/// the distance between two complex numbers `z1` and `z2` is 
/// zero, then `z1 == z2`.
///
/// Given complex numbers `z1` and `z2`
/// ```text
/// l1_norm(z1 - z2) == 0 => z1 == z2 
/// ```
/// Equivalently, if `z1` is not equal to `z2`, then their distance is 
/// nonzero
/// ```text
/// z1 != z2 => l1_norm(z1 - z2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_approx_l1_norm_point_separating<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(z1, z2, epsilon = tolerance));
    prop_assert!((z1 - z2).l1_norm() > tolerance);

    Ok(())
}

/// The **L1** norm function is point separating. In particular, if 
/// the distance between two complex numbers `z1` and `z2` is 
/// zero, then `z1 == z2`.
///
/// Given complex numbers `z1` and `z2`
/// ```text
/// l1_norm(z1 - z2) == 0 => z1 == z2 
/// ```
/// Equivalently, if `z1` is not equal to `z2`, then their distance is 
/// nonzero
/// ```text
/// z1 != z2 => l1_norm(z1 - z2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_l1_norm_point_separating<S>(z1: Complex<S>, z2: Complex<S>) -> Result<(), TestCaseError>
where 
    S: SimdScalarSigned
{    
    let zero = S::zero();

    prop_assume!(z1 != z2);
    prop_assert_ne!((z1 - z2).l1_norm(), zero);

    Ok(())
}

/// The exponential of the sum of two complex numbers is the product of the 
/// exponentials of the two complex numbers.
/// 
/// Given complex numbers `z1` and `z2`
/// ```text
/// exp(z1 + z2) == exp(z1) * exp(z2)
/// ```
fn prop_approx_exp_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = (z1 + z2).exp();
    let rhs = z1.exp() * z2.exp();

    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative), 
        "z1 = {}; z1 = {}; exp(z1 + z2) = {}; exp(z1) * exp(z2) = {}",
        z1, z2, lhs, rhs
    );

    Ok(())
}

/// The complex exponential of a complex number is nonzero.
/// 
/// Given a complex number `z`
/// ```text
/// exp(z) != 0
/// ```
fn prop_exp_complex_nonzero<S>(z: Complex<S>) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let zero_complex = Complex::zero();

    prop_assert_ne!(z.exp(), zero_complex);

    Ok(())
}

/// The complex exponential satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// exp(-z) * exp(z) == exp(z) * exp(-z) == 1
/// ```
fn prop_approx_exp_complex_exp_negative_complex<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let unit_re = Complex::unit_re();
    let exp_z = z.exp();
    let exp_negative_z = (-z).exp();

    let lhs1 = exp_negative_z * exp_z;
    let rhs1 = unit_re;

    prop_assert!(relative_eq!(lhs1, rhs1, epsilon = tolerance));

    let lhs2 = exp_z * exp_negative_z;
    let rhs2 = unit_re;

    prop_assert!(relative_eq!(lhs2, rhs2, epsilon = tolerance));

    Ok(())
}

/// The complex logarithm satisfiess the following relation.
/// 
/// Given non-zero complex numbers `z1` and `z2`, there is an integer `k` such that
/// ```text
/// ln(z1 * z2) - (ln(z1) + ln(z2)) == 2 * pi * k * i
/// ```
fn prop_approx_ln_product<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assume!(!z1.is_zero());
    prop_assume!(!z2.is_zero());

    let ln_z1_times_z2 = (z1 * z2).ln();
    let ln_z1_plus_ln_z2 = z1.ln() + z2.ln();
    let lhs = (ln_z1_times_z2 - ln_z1_plus_ln_z2) / S::two_pi();
    let rhs = Complex::new(lhs.real().round(), lhs.imaginary().round());

    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance),
        "z1 = {}; z2 = {}; ln(z1 * z2) = {}; ln(z1) + ln(z2) = {}; (ln(z1 * z2) - (ln(z1) + ln(z2))) / (2 * pi) = {}",
        z1, z2, ln_z1_times_z2, ln_z1_plus_ln_z2, lhs
    );

    Ok(())
}

/// The real part of the complex logarithm is the logarithm of the complex modulus.
/// 
/// Given a complex number `z`
/// ```text
/// re(ln(z)) == ln(modulus(z))
/// ```
fn prop_approx_complex_ln_real_part<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = z.ln().real();
    let rhs = z.modulus().ln();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The complex exponential and the principal value of the complex logarithm
/// satisfy the folowing relation.
/// 
/// Given a complex number `z`
/// ```text
/// ln(exp(z)) == z
/// ```
fn prop_approx_exp_ln_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = z.ln().exp();
    let rhs = z;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The complex exponential and the principal value of the complex logarithm
/// satisfy the following relation.
/// 
/// Given a complex number `z`, there is an integer `k` such that
/// ```text
/// ln(exp(z)) - z == 2 * pi * k * i
/// ```
fn prop_approx_ln_exp_identity_up_to_phase<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let ln_exp_z = z.exp().ln();
    let lhs = (ln_exp_z - z) / (S::two_pi());
    let rhs = Complex::new(lhs.real().round(), lhs.imaginary().round());

    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance),
        "z = {}; ln_exp_z = {}; lhs = {}; rhs = {}",
        z, ln_exp_z, lhs, rhs
    );

    Ok(())
}


/// The principal argument of two complex numbers that differ only by a phase factor
/// of `2 * pi * k` for some integer `k` have the same argument up to a sign factor.
/// 
/// Given complex numbers `z1` and `z2` such that `z1 := r * exp(i * angle)` and 
/// `z2 := r * exp(i * (angle + 2 * pi * k))` where `r` is a floating point number 
/// and `k` is an integer
/// ```text
/// arg(z1) == arg(z2)
/// ```
/// Moreover, this indicates that the `arg` function correctly implements the fact
/// that the principal argument is unique on the interval `[-pi, pi]`.
fn prop_approx_arg_congruent<S>(z: Complex<S>, k: i32, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    use cglinalg_core::Radians;

    let modulus_z = z.modulus();
    let principal_arg_z = z.arg();
    let _k = num_traits::cast(k).unwrap();
    let arg_new_z = principal_arg_z + S::two_pi() * _k;
    let angle_new_z = Radians(arg_new_z);
    let new_z = Complex::from_polar_decomposition(modulus_z, angle_new_z);

    let lhs = z.arg();
    let rhs = new_z.arg();

    prop_assert!(relative_eq!(lhs.abs(), rhs.abs(), epsilon = tolerance));

    Ok(())
}

/// The argument of the quotient of two non-zero complex numbers satisfies the following
/// relation.
/// 
/// Given two non-zero complex numbers `z1` and `z2`, there exists an integer `k` such 
/// that
/// ```text
/// arg(z1 * z2) - (arg(z1) + arg(z2)) == 2 * pi * k
/// ```
fn prop_approx_arg_complex_times_complex_equals_arg_complex_plus_arg_complex<S>(
    z1: Complex<S>, 
    z2: Complex<S>, 
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assume!(!z1.is_zero());
    prop_assume!(!z2.is_zero());

    let arg_z1_times_z2 = (z1 * z2).arg();
    let arg_z1_plus_arg_z2 = z1.arg() + z2.arg();
    let lhs = (arg_z1_times_z2 - arg_z1_plus_arg_z2) / S::two_pi();
    let rhs = lhs.round();

    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance),
        "z1 = {}; z2 = {}; arg(z1 * z2) = {}; arg(z1) + arg(z2) = {}; lhs = {}, rhs = {}",
        z1, z2, arg_z1_times_z2, arg_z1_plus_arg_z2, lhs, rhs
    );

    Ok(())
}

/// The argument of the quotient of two non-zero complex numbers satisfies the following
/// relation.
/// 
/// Given two non-zero complex numbers `z1` and `z2`, there exists an integer `k` such 
/// that
/// ```text
/// arg(z1 / z2) - (arg(z1) - arg(z2)) == 2 * pi * k
/// ```
fn prop_approx_arg_complex_div_complex_equals_arg_complex_minus_arg_complex<S>(
    z1: Complex<S>, 
    z2: Complex<S>, 
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assume!(!z1.is_zero());
    prop_assume!(!z2.is_zero());

    let arg_z1_div_z2 = (z1 / z2).arg();
    let arg_z1_minus_arg_z2 = z1.arg() - z2.arg();
    let lhs = (arg_z1_div_z2 - arg_z1_minus_arg_z2) / S::two_pi();
    let rhs = lhs.round();

    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance),
        "z1 = {}; z2 = {}; arg(z1 / z2) = {}; arg(z1) - arg(z2) = {}, lhs = {}; rhs = {}",
        z1, z2, arg_z1_div_z2, arg_z1_minus_arg_z2, lhs, rhs
    );

    Ok(())
}

/// The principal argument of a complex number is in the range `[-pi, pi]`.
/// 
/// Given a complex number `z`
/// ```text
/// -pi =< arg(z) <= pi
/// ```
fn prop_arg_range<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let arg_z = z.arg();

    prop_assert!(arg_z >= -S::pi());
    prop_assert!(arg_z <= S::pi());

    Ok(())
}

/// The square of the positive square root of a complex number is the original
/// complex number.
/// 
/// Given a complex number `z`
/// ```text
/// sqrt(z) * sqrt(z) == z
/// ```
fn prop_approx_square_root_complex_squared<S>(z: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let sqrt_z = z.sqrt();
    let lhs = sqrt_z * sqrt_z;
    let rhs = z;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The square of the principal value of the square root of a the negation of a 
/// complex number is negation of the original complex number.
/// 
/// Given a complex number `z`
/// ```text
/// sqrt(-z) * sqrt(-z) == -z
/// ```
/// Indeed, we can write
/// ```text
/// sqrt(-z) * sqrt(-z) == (i * sqrt(z)) * (i * sqrt(z))
///                     == i * i * sqrt(z) * sqrt(z)
///                     == -(sqrt(z) * sqrt(z)
///                     == -z
/// ```
/// as desired.
fn prop_approx_square_root_negative_complex_squared<S>(z: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let sqrt_negative_z = (-z).sqrt();
    let sqrt_negative_z_squared = sqrt_negative_z * sqrt_negative_z;
    let negative_z = -z;
    let lhs = sqrt_negative_z_squared.polar_decomposition();
    let rhs = negative_z.polar_decomposition();

    prop_assert!(relative_eq!(lhs.0, rhs.0, epsilon = tolerance, max_relative = max_relative));
    prop_assert!(relative_eq!(lhs.1, rhs.1, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The square of the principal value of the conjugate of a complex number is
/// the conjugate of the original complex number.
/// 
/// Given a complex number `z`
/// ```text
/// sqrt(conjugate(z)) * sqrt(conjugate(z)) == conjugate(z)
/// ```
fn prop_approx_square_root_complex_conjugate_squared<S>(z: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let sqrt_conjugate_z = z.conjugate().sqrt();
    let lhs = sqrt_conjugate_z * sqrt_conjugate_z;
    let rhs = z.conjugate();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative), "lhs = {}; rhs = {}", lhs, rhs);

    Ok(())
}

/// The modulus of the square root of the product of two complex numbers is the 
/// product of the moduli of the square roots of the individual complex numbers.
/// 
/// Given complex numbers `z1` and `z2`
/// ```text
/// modulus(sqrt(z1 * z2)) == modulus(sqrt(z1)) * modulus(zqrt(z2))
/// ```
fn prop_approx_square_root_product_modulus<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = (z1 * z2).sqrt().modulus();
    let rhs = z1.sqrt().modulus() * z2.sqrt().modulus();
    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative),
        "z1 = {}; z2 = {}; modulus(sqrt(z1 * z2)) = {}; modulus(sqrt(z1)) * modulus(sqrt(z2)) = {}",
        z1, z2, lhs, rhs
    );

    Ok(())
}

/// The argument of the square root of a complex number should satisfy the
/// following property.
/// 
/// Given a complex number `z`
/// ```text
/// pi / 2 =< arg(sqrt(z)) <= pi / 2
/// ```
fn prop_square_root_arg_range<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let pi_over_2 = S::frac_pi_2();
    let arg_sqrt_z = z.sqrt().arg();

    prop_assert!((arg_sqrt_z >= -pi_over_2) && (arg_sqrt_z <= pi_over_2));

    Ok(())
}

/// The cube of the cubed root of a complex number is the original complex number.
/// 
/// Given a complex number `z`
/// ```text
/// cubed(cbrt(z)) == z
/// ```
fn prop_approx_cubed_root_complex_cubed<S>(z: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let cbrt_z = z.cbrt();
    let lhs = cbrt_z.cubed();
    let rhs = z;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The cosine of a complex number with imaginary part zero equals the 
/// cosine of the real part.
/// 
/// Given a complex number `z` with imaginary part `im(z) == 0`
/// ```text
/// cos(z) == cos(re(z))
/// ```
fn prop_approx_cos_real_equals_cos_real<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where 
    S: SimdScalarFloat
{
    let re_z = z.real();
    let z_re = Complex::from_real(re_z);

    prop_assert!(relative_eq!(z_re.cos().real(), re_z.cos(), epsilon = tolerance));

    Ok(())
}

/// The cosine of the negation of a complex number equals the cosine of the
/// complex number.
/// 
/// Given a complex number `z`
/// ```text
/// cos(-z) == cos(z)
/// ```
fn prop_approx_cos_negative_z_equals_cos_z<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = (-z).cos();
    let rhs = z.cos();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The cosine of a complex number with real part zero equals i times the 
/// hyperbolic cosine of the imaginary part.
/// 
/// Given a complex number `z` with real part `re(z) == 0`
/// ```text
/// cos(z) == i * cosh(im(z))
/// ```
fn prop_approx_cos_imaginary_equals_imaginary_cosh<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let im_z = z.imaginary(); 
    let lhs = Complex::cos(Complex::from_imaginary(im_z));
    let rhs = Complex::cosh(Complex::from_real(im_z));

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The sine of a complex number with imaginary part zero equals the sine
/// of the real part.
/// 
/// Given a complex number `z` such that `im(z) == 0`
/// ```text
/// sin(z) == sin(re(z))
/// ```
fn prop_approx_sin_real_equals_sin_real<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let re_z = z.real();
    let z_re = Complex::from_real(re_z);

    prop_assert!(relative_eq!(z_re.sin().real(), re_z.sin(), epsilon = tolerance));

    Ok(())
}

/// The sine of the negation of a complex number equals the negation of the sine
/// of the complex number.
/// 
/// Given a complex number `z`
/// ```text
/// sin(-z) == -sin(z)
/// ```
fn prop_approx_sin_negative_z_equals_negative_sin_z<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = (-z).sin();
    let rhs = -(z.sin());

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The sine of a complex number with real part zero equals i times the 
/// hyperbolic sine of the imaginary part.
/// 
/// Given a complex number `z` with real part `re(z) == 0`
/// ```text
/// sin(z) == i * sinh(im(z))
/// ```
fn prop_approx_sin_imaginary_equals_imaginary_sinh<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let i = Complex::unit_im();
    let im_z = z.imaginary();

    prop_assert!(relative_eq!((i * im_z).sin(), i * im_z.sinh(), epsilon = tolerance));

    Ok(())
}

/// The tangent of a complex number with imaginary part zero equals the tangent
/// of the real part.
/// 
/// Given a complex number `z` such that `im(z) == 0`
/// ```text
/// tan(z) == tan(re(z))
/// ```
fn prop_approx_tan_real_equals_real_tan<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let tan_z = z.tan();
    let lhs = tan_z.real();
    let rhs = z.real().tan();

    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance),
        "z = {}; tan(z) = {}; re(tan(z)) = {}; tan(re(z)) = {}",
        z, tan_z, lhs, rhs
    );

    Ok(())
}

/// The tangent of the negation of a complex number equals the negation of the
/// tangent of a complex number.
/// 
/// Given a complex number `z`
/// ```text
/// tan(-z) == -tan(z)
/// ```
fn prop_approx_tan_negative_z_equals_negative_tan_z<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = (-z).tan();
    let rhs = -(z.tan());

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}


/// The tangent of a complex number with real part zero equals i times the
/// hyperbolic tangent of the imaginary part.
/// 
/// Given a complex number `z` such that `re(z) == 0`
/// ```text
/// tan(z) == i * tanh(im(z))
/// ```
fn prop_approx_tan_imaginary_equals_imaginary_tanh<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let i = Complex::unit_im();
    let im_z = z.imaginary();

    prop_assert!(relative_eq!((i * im_z).tan(), i * im_z.tanh(), epsilon = tolerance));

    Ok(())
}

/// The complex cosine function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// cos(2 * z) == cos(z) * cos(z) - sin(z) * sin(z)
/// ```
fn prop_approx_cos_two_times_angle_equals_two_times_cos_angle_squared_minus_sin_angle_squared<S>(
    z: Complex<S>, 
    tolerance: S,
    max_relative: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let one = Complex::one();
    let two = one + one;
    let lhs = (two * z).cos();
    let cos_z_squared = z.cos().squared();
    let sin_z_squared = z.sin().squared();
    let rhs = cos_z_squared - sin_z_squared;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The complex sine function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// sin(2 * z) == 2 * sin(z) * cos(z)
/// ```
fn prop_approx_sin_two_times_angle_equals_two_times_sin_angle_times_cos_angle<S>(
    z: Complex<S>, 
    tolerance: S, 
    max_relative: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let one = Complex::one();
    let two = one + one;
    let lhs = (two * z).sin();
    let rhs = two * z.sin() * z.cos();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The complex tangent function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// tan(z) * (1 - tan(z) * tan(z)) == 2 * tan(z)
/// ```
fn prop_approx_tan_two_times_angle<S>(z: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let one = Complex::one();
    let two = one + one;
    let tan_two_z = (two * z).tan();
    let tan_z_squared = z.tan().squared();
    let lhs = tan_two_z * (one - tan_z_squared);
    let rhs = two * z.tan();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The complex cosine function satisfies the following relation.
/// 
/// Given two complex numbers `z1` and `z2`
/// ```text
/// cos(z1 + z2) == cos(z1) * cos(z2) - sin(z1) * cos(z2)
/// ```
fn prop_approx_cos_angle_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = (z1 + z2).cos();
    let rhs = z1.cos() * z2.cos() - z1.sin() * z2.sin();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The complex sine function satisfies the following relation.
/// 
/// Given complex numbers `z1` and `z2`
/// ```text
/// sin(z1 + z2) == sin(z1) * cos(z2) + cos(z1) * sin(z2)
/// ```
fn prop_approx_sin_angle_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = (z1 + z2).sin();
    let rhs = z1.sin() * z2.cos() + z1.cos() * z2.sin();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The complex tangent function satisfies the following relation.
/// 
/// Given complex numbers `z1` and `z2`
/// ```text
/// tan(z1 + z2) * (1 - tan(z1) * tan(z2)) == tan(z1) + tan(z2)
/// ```
fn prop_approx_tan_angle_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let one = Complex::one();
    let lhs = (z1 + z2).tan() * (one - z1.tan() * z2.tan());
    let rhs = z1.tan() + z2.tan();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The complex tangent function satisfies the following relation.
/// 
/// Given complex numbers `z1` and `z2`
/// ```text
/// tan(z1 - z2) * (1 - tan(z1) * tan(z2)) == tan(z1) - tan(z2)
/// ```
fn prop_approx_tan_angle_difference<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let one = Complex::one();
    let lhs = (z1 - z2).tan() * (one + z1.tan() * z2.tan());
    let rhs = z1.tan() - z2.tan();

    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative),
        "z1 = {}; z2 = {}; tan(z1 - z2) * (1 - tan(z1) * tan(z2)) = {}; tan(z1) - tan(z2) = {}",
        z1, z2, lhs, rhs
    );

    Ok(())
}

/// The complex arccosine function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// acos(conjugate(z)) == conjugate(acos(z))
/// ```
fn prop_approx_acos_conjugate_z_equals_conjugate_acos_z<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = z.conjugate().acos();
    let rhs = z.acos().conjugate();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The complex arcsine function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// asin(conjugate(z)) == conjugate(asin(z))
/// ```
fn prop_approx_asin_conjugate_z_equals_conjugate_asin_z<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = z.conjugate().asin();
    let rhs = z.asin().conjugate();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The complex arctangent function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// atan(conjugate(z)) == conjugate(atan(z))
/// ```
fn prop_approx_atan_conjugate_z_equals_conjugate_atan_z<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = z.conjugate().atan();
    let rhs = z.atan().conjugate();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The cosine and arccosine functions satisfy the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// cos(acos(z)) == z
/// ```
fn prop_approx_cos_acos_equals_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = z.acos().cos();
    let rhs = z;
    
    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The sine and arcsine functions satisfy the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// sin(asin(z)) == z
/// ```
fn prop_approx_sin_asin_equals_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = z.asin().sin();
    let rhs = z;
    
    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance),
        "z = {}; asin(z) = {}; sin(asin(z)) = {}",
        z, z.asin(), lhs
    );

    Ok(())
}

/// The tangent and arctangent functions satisfy the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// tan(atan(z)) == z
/// ```
fn prop_approx_tan_atan_equals_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = z.atan().tan();
    let rhs = z;
    
    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The complex hyperbolic cosine function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// cosh(conjugate(z)) == conjugate(cosh(z))
/// ```
fn prop_approx_cosh_conjugate_z_equals_conjugate_cosh_z<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert_eq!(z.conjugate().cosh(), z.cosh().conjugate());

    Ok(())
}

/// The complex hyperbolic cosine function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// cosh(-z) == cosh(z)
/// ```
fn prop_approx_cosh_negative_z_equals_negative_cosh_z<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert!(relative_eq!((-z).cosh(), z.cosh(), epsilon = tolerance));

    Ok(())
}

/// The complex hyperbolic sine function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// sinh(conjugate(z)) == conjugate(sinh(z))
/// ```
fn prop_approx_sinh_conjugate_z_equals_conjugate_sinh_z<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert_eq!(z.conjugate().sinh(), z.sinh().conjugate());

    Ok(())
}

/// The complex hyperbolic sine function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// sinh(-z) == -sinh(z)
/// ```
fn prop_approx_sinh_negative_z_equals_negative_sinh_z<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert!(relative_eq!((-z).sinh(), -z.sinh(), epsilon = tolerance));

    Ok(())
}

/// The complex hyperbolic tangent function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// tanh(conjugate(z)) == conjugate(tanh(z))
/// ```
fn prop_approx_tanh_conjugate_z_equals_conjugate_tanh_z<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert_eq!(z.conjugate().tanh(), z.tanh().conjugate());

    Ok(())
}

/// The complex hyperbolic tangent function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// tanh(-z) == -tanh(z)
/// ```
fn prop_approx_tanh_negative_z_equals_negative_tanh_z<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert!(relative_eq!((-z).tanh(), -z.tanh(), epsilon = tolerance));

    Ok(())
}

/// The complex hyperbolic cosine satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// cosh(2 * z) == cosh(z) * cosh(z) + sinh(z) * sinh(z)
/// ```
fn prop_approx_cosh_two_times_angle_equals_cosh_squared_plus_sinh_squared<S>(
    z: Complex<S>, 
    tolerance: S, 
    max_relative: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let one = Complex::one();
    let two = one + one;
    let lhs = (two * z).cosh();
    let rhs = z.cosh().squared() + z.sinh().squared();

    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative),
        "z = {}; cosh(2 * z) = {}; cosh(z) * cosh(z) + sinh(z) * sinh(z) = {}",
        z, lhs, rhs
    );

    Ok(())
}

/// The complex hyperbolic sine function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// sinh(2 * z) == 2 * sinh(z) * cosh(z)
/// ```
fn prop_approx_sinh_two_times_angle_equals_two_times_sinh_cosh<S>(
    z: Complex<S>, 
    tolerance: S, 
    max_relative: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let one = Complex::one();
    let two = one + one;
    let lhs = (two * z).sinh();
    let rhs = two * z.sinh() * z.cosh();

    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative),
        "z = {}; sinh(2 * z) = {}; 2 * sinh(z) * cosh(z) = {}",
        z, lhs, rhs
    );

    Ok(())
}

/// The complex hyperbolic tangent function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// tanh(2 * z) * (1 + tanh(z) * tanh(z)) == 2 * tanh(z)
/// ```
fn prop_approx_tanh_two_times_angle<S>(z: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let one = Complex::one();
    let two = one + one;
    let tanh_two_z = (two * z).tanh();
    let tanh_z_squared = z.tanh().squared(); 
    let lhs = tanh_two_z * (one + tanh_z_squared);
    let rhs = two * z.tanh();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The complex hyperbolic cosine function satisfies the following relation.
/// 
/// Given complex numbers `z1` and `z2`
/// ```text
/// cosh(z1 + z2) == cosh(z1) * cosh(z2) + sinh(z1) * sinh(z2)
/// ```
fn prop_approx_cosh_angle_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = (z1 + z2).cosh();
    let rhs = z1.cosh() * z2.cosh() + z1.sinh() * z2.sinh();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The complex hyperbolic sine function satisfies the following relation.
/// 
/// Given complex numbers `z1` and `z1`
/// ```text
/// sinh(z1 + z2) == sinh(z1) * cosh(z2) + cosh(z1) * sinh(z2)
/// ```
fn prop_approx_sinh_angle_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = (z1 + z2).sinh();
    let rhs = z1.sinh() * z2.cosh() + z1.cosh() * z2.sinh();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The hyperbolic tangent function satisfies the following relation.
/// 
/// Given complex numbers `z1` and `z2`
/// ```text
/// tanh(z1 + z2) * (1 + tanh(z1) * tanh(z2)) == tanh(z1) + tanh(z2)
/// ```
fn prop_approx_tanh_angle_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let one = Complex::one();
    let lhs = (z1 + z2).tanh() * (one + z1.tanh() * z2.tanh());
    let rhs = z1.tanh() + z2.tanh();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The hyperbolic tangent function satisfies the following relation.
/// 
/// Given complex numbers `z1` and `z2`
/// ```text
/// tanh(z1 - z2) * (1 - tanh(z1) * tanh(z2)) == tanh(z1) - tanh(z2)
/// ```
fn prop_approx_tanh_angle_difference<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let one = Complex::one();
    let lhs = (z1 - z2).tanh() * (one - z1.tanh() * z2.tanh());
    let rhs = z1.tanh() - z2.tanh();

    prop_assert!(
        relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative),
        "z1 = {}; z2 = {}; tanh(z1 - z2) * (1 - tanh(z1) * tanh(z2)) = {}; tanh(z1) - tanh(z2) = {}",
        z1, z2, lhs, rhs
    );

    Ok(())
}


/// The hyperbolic cosine and hyperbolic arccosine functions satisfy the 
/// following relation.
/// 
/// Given a complex number `z`
/// ```text
/// cosh(acosh(z)) == z
/// ```
fn prop_approx_cosh_acosh_equals_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = z.acosh().cosh();
    let rhs = z;
    
    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The hyperbolic sine and hyperbolic arcsine functions satisfy the 
/// following relation.
/// 
/// Given a complex number `z`
/// ```text
/// sinh(asinh(z)) == z
/// ```
fn prop_approx_sinh_asinh_equals_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = z.asinh().sinh();
    let rhs = z;
    
    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance), "lhs = {}; rhs = {}", lhs, rhs);

    Ok(())
}

/// The hyperbolic tangent and hyperbolic arctangent functions satisfy the 
/// following relation.
/// 
/// Given a complex number `z`
/// ```text
/// tanh(atanh(z)) == z
/// ```
fn prop_approx_tanh_atanh_equals_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = z.atanh().tanh();
    let rhs = z;
    
    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}


#[cfg(test)]
mod complex_f64_arithmetic_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_zero_times_complex_equals_zero(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_zero_times_complex_equals_zero(z)?
        }
    
        #[test]
        fn prop_complex_times_zero_equals_zero(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_times_zero_equals_zero(z)?
        }

        #[test]
        fn prop_complex_plus_zero_equals_complex(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_plus_zero_equals_complex(z)?
        }

        #[test]
        fn prop_zero_plus_complex_equals_complex(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_zero_plus_complex_equals_complex(z)?
        }

        #[test]
        fn prop_one_times_complex_equal_complex(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_one_times_complex_equal_complex(z)?
        }

        #[test]
        fn prop_complex_times_one_equals_complex(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_times_one_equals_complex(z)?
        }

        #[test]
        fn prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex(z1 in super::strategy_complex_f64_any(), z2 in super::strategy_complex_f64_any()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex2(z1, z2)?
        }

        #[test]
        fn prop_complex_addition_commutative(z1 in super::strategy_complex_f64_any(), z2 in super::strategy_complex_f64_any()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_complex_addition_commutative(z1, z2)?
        }

        #[test]
        fn prop_complex_minus_zero_equals_complex(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_minus_zero_equals_complex(z)?
        }

        #[test]
        fn prop_complex_minus_complex_equals_zero(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_minus_complex_equals_zero(z)?
        }

        #[test]
        fn prop_complex1_minus_complex2_equals_refcomplex1_minus_refcomplex2(
            z1 in super::strategy_complex_f64_any(), 
            z2 in super::strategy_complex_f64_any()
        ) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_complex1_minus_complex2_equals_refcomplex1_minus_refcomplex2(z1, z2)?
        }

        #[test]
        fn prop_scalar_times_complex_equals_complex_times_scalar(
            c in super::strategy_scalar_f64_any(), 
            z in super::strategy_complex_f64_any()
        ) {
            let c: f64 = c;
            let z: super::Complex<f64> = z;
            super::prop_scalar_times_complex_equals_complex_times_scalar(c, z)?
        }

        #[test]
        fn prop_complex_multiplicative_unit(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_multiplicative_unit(z)?
        }

        #[test]
        fn prop_approx_complex_multiplicative_inverse(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_complex_multiplicative_inverse(z, 1e-7)?
        }

        #[test]
        fn prop_complex_multiplication_commutative(z1 in super::strategy_complex_f64_any(), z2 in super::strategy_complex_f64_any()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_complex_multiplication_commutative(z1, z2)?
        }
    }
}

#[cfg(test)]
mod complex_i32_arithmetic_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_zero_times_complex_equals_zero(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_zero_times_complex_equals_zero(z)?
        }
    
        #[test]
        fn prop_complex_times_zero_equals_zero(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_times_zero_equals_zero(z)?
        }

        #[test]
        fn prop_complex_plus_zero_equals_complex(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_plus_zero_equals_complex(z)?
        }

        #[test]
        fn prop_zero_plus_complex_equals_complex(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_zero_plus_complex_equals_complex(z)?
        }

        #[test]
        fn prop_one_times_complex_equal_complex(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_one_times_complex_equal_complex(z)?
        }

        #[test]
        fn prop_complex_times_one_equals_complex(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_times_one_equals_complex(z)?
        }

        #[test]
        fn prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex2(
            z1 in super::strategy_complex_i32_any(), 
            z2 in super::strategy_complex_i32_any()
        ) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex2(z1, z2)?
        }

        #[test]
        fn prop_complex_addition_commutative(z1 in super::strategy_complex_i32_any(), z2 in super::strategy_complex_i32_any()) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_complex_addition_commutative(z1, z2)?
        }

        #[test]
        fn prop_complex_addition_associative(
            z1 in super::strategy_complex_i32_any(), 
            z2 in super::strategy_complex_i32_any(), 
            z3 in super::strategy_complex_i32_any()
        ) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            let z3: super::Complex<i32> = z3;
            super::prop_complex_addition_associative(z1, z2, z3)?
        }

        #[test]
        fn prop_complex_minus_zero_equals_complex(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_minus_zero_equals_complex(z)?
        }

        #[test]
        fn prop_complex_minus_complex_equals_zero(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_minus_complex_equals_zero(z)?
        }

        #[test]
        fn prop_complex1_minus_complex2_equals_refcomplex1_minus_refcomplex2(
            z1 in super::strategy_complex_i32_any(), 
            z2 in super::strategy_complex_i32_any()
        ) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_complex1_minus_complex2_equals_refcomplex1_minus_refcomplex2(z1, z2)?
        }

        #[test]
        fn prop_scalar_times_complex_equals_complex_times_scalar(
            c in super::strategy_scalar_i32_any(), 
            z in super::strategy_complex_i32_any()
        ) {
            let c: i32 = c;
            let z: super::Complex<i32> = z;
            super::prop_scalar_times_complex_equals_complex_times_scalar(c, z)?
        }

        #[test]
        fn prop_scalar_multiplication_compatibility1(
            a in super::strategy_scalar_i32_any(), 
            b in super::strategy_scalar_i32_any(), 
            z in super::strategy_complex_i32_any()
        ) {
            let a: i32 = a;
            let b: i32 = b;
            let z: super::Complex<i32> = z;
            super::prop_scalar_multiplication_compatibility1(a, b, z)?
        }

        #[test]
        fn prop_scalar_multiplication_compatibility2(
            a in super::strategy_scalar_i32_any(), 
            b in super::strategy_scalar_i32_any(), 
            z in super::strategy_complex_i32_any()
        ) {
            let a: i32 = a;
            let b: i32 = b;
            let z: super::Complex<i32> = z;
            super::prop_scalar_multiplication_compatibility2(a, b, z)?
        }

        #[test]
        fn prop_complex_multiplication_associative(
            z1 in super::strategy_complex_i32_any(), 
            z2 in super::strategy_complex_i32_any(), 
            z3 in super::strategy_complex_i32_any()
        ) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            let z3: super::Complex<i32> = z3;
            super::prop_complex_multiplication_associative(z1, z2, z3)?
        }

        #[test]
        fn prop_complex_multiplicative_unit(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_multiplicative_unit(z)?
        }

        #[test]
        fn prop_complex_multiplication_commutative(
            z1 in super::strategy_complex_i32_any(), 
            z2 in super::strategy_complex_i32_any()
        ) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_complex_multiplication_commutative(z1, z2)?
        }
    }
}

#[cfg(test)]
mod complex_i32_distributive_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_distribution_over_complex_addition(
            a in super::strategy_scalar_i32_any(), 
            z1 in super::strategy_complex_i32_any(), 
            z2 in super::strategy_complex_i32_any()
        ) {
            let a: i32 = a;
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_distribution_over_complex_addition(a, z1, z2)?
        }

        #[test]
        fn prop_distribution_over_scalar_addition(
            a in super::strategy_scalar_i32_any(),
            b in super::strategy_scalar_i32_any(), 
            z in super::strategy_complex_i32_any()
        ) {
            let a: i32 = a;
            let b: i32 = b;
            let z: super::Complex<i32> = z;
            super::prop_distribution_over_scalar_addition(a, b, z)?
        }

        #[test]
        fn prop_distribution_over_complex_addition1(
            a in super::strategy_scalar_i32_any(), 
            z1 in super::strategy_complex_i32_any(), 
            z2 in super::strategy_complex_i32_any()
        ) {
            let a: i32 = a;
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_distribution_over_complex_addition1(a, z1, z2)?
        }

        #[test]
        fn prop_distribution_over_scalar_addition1(
            a in super::strategy_scalar_i32_any(), 
            b in super::strategy_scalar_i32_any(), 
            z in super::strategy_complex_i32_any()
        ) {
            let a: i32 = a;
            let b: i32 = b;
            let z: super::Complex<i32> = z;
            super::prop_distribution_over_scalar_addition1(a, b, z)?
        }

        #[test]
        fn prop_complex_multiplication_right_distributive(
            z1 in super::strategy_complex_i32_any(), 
            z2 in super::strategy_complex_i32_any(), 
            z3 in super::strategy_complex_i32_any()
        ) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            let z3: super::Complex<i32> = z3;
            super::prop_complex_multiplication_right_distributive(z1, z2, z3)?
        }

        #[test]
        fn prop_complex_multiplication_left_distributive(
            z1 in super::strategy_complex_i32_any(), 
            z2 in super::strategy_complex_i32_any(), 
            z3 in super::strategy_complex_i32_any()
        ) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            let z3: super::Complex<i32> = z3;
            super::prop_complex_multiplication_left_distributive(z1, z2, z3)?
        }
    }
}

#[cfg(test)]
mod complex_f64_conjugation_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_complex_conjugate_conjugate_equals_complex(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_conjugate_conjugate_equals_complex(z)?
        }

        #[test]
        fn prop_complex_conjugation_linear(z1 in super::strategy_complex_f64_any(), z2 in super::strategy_complex_f64_any()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_complex_conjugation_linear(z1, z2)?
        }

        #[test]
        fn prop_complex_conjugation_transposes_products(
            z1 in super::strategy_complex_f64_any(), 
            z2 in super::strategy_complex_f64_any()
        ) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_complex_conjugation_transposes_products(z1, z2)?
        }
    }
}


#[cfg(test)]
mod complex_i32_conjugation_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_complex_conjugate_conjugate_equals_complex(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_conjugate_conjugate_equals_complex(z)?
        }

        #[test]
        fn prop_complex_conjugation_linear(z1 in super::strategy_complex_i32_any(), z2 in super::strategy_complex_i32_any()) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_complex_conjugation_linear(z1, z2)?
        }

        #[test]
        fn prop_complex_conjugation_transposes_products(z1 in super::strategy_complex_i32_any(), z2 in super::strategy_complex_i32_any()) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_complex_conjugation_transposes_products(z1, z2)?
        }
    }
}

#[cfg(test)]
mod complex_f64_modulus_squared_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_modulus_squared_nonnegative(z in super::strategy_complex_f64_modulus_squared()) {
            let z: super::Complex<f64> = z;
            super::prop_modulus_squared_nonnegative(z)?
        }

        #[test]
        fn prop_approx_modulus_squared_point_separating(
            z1 in super::strategy_complex_f64_modulus_squared(), 
            z2 in super::strategy_complex_f64_modulus_squared()
        ) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_modulus_squared_point_separating(z1, z2, 1e-10, 1e-20)?
        }
    }
}

#[cfg(test)]
mod complex_f64_modulus_squared_synonym_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_magnitude_squared_modulus_squared_synonyms(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_magnitude_squared_modulus_squared_synonyms(z)?
        }

        #[test]
        fn prop_norm_squared_modulus_squared_synonyms(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_norm_squared_modulus_squared_synonyms(z)?
        }
    }
}

#[cfg(test)]
mod complex_i32_modulus_squared_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_modulus_squared_nonnegative(z in super::strategy_complex_i32_modulus_squared()) {
            let z: super::Complex<i32> = z;
            super::prop_modulus_squared_nonnegative(z)?
        }

        #[test]
        fn prop_modulus_squared_point_separating(
            z1 in super::strategy_complex_i32_modulus_squared(), 
            z2 in super::strategy_complex_i32_modulus_squared()
        ) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_modulus_squared_point_separating(z1, z2)?
        }

        #[test]
        fn prop_modulus_squared_homogeneous_squared(z in super::strategy_complex_i32_modulus_squared(), c in super::strategy_scalar_i32_any()) {
            let z: super::Complex<i32> = z;
            let c: i32 = c;
            super::prop_modulus_squared_homogeneous_squared(z, c)?
        }
    }
}

#[cfg(test)]
mod complex_i32_modulus_squared_synonym_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_magnitude_squared_modulus_squared_synonyms(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_magnitude_squared_modulus_squared_synonyms(z)?
        }

        #[test]
        fn prop_norm_squared_modulus_squared_synonyms(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_norm_squared_modulus_squared_synonyms(z)?
        }
    }
}

#[cfg(test)]
mod complex_f64_modulus_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_modulus_nonnegative(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_modulus_nonnegative(z)?
        }

        #[test]
        fn prop_approx_modulus_point_separating(
            z1 in super::strategy_complex_f64_any(),
            z2 in super::strategy_complex_f64_any()
        ) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_modulus_point_separating(z1, z2, 1e-8)?
        }
    }
}

#[cfg(test)]
mod complex_f64_modulus_synonym_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_magnitude_modulus_synonyms(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_magnitude_modulus_synonyms(z)?
        }

        #[test]
        fn prop_norm_modulus_synonyms(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_norm_modulus_synonyms(z)?
        }

        #[test]
        fn prop_l2_norm_modulus_synonyms(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_l2_norm_modulus_synonyms(z)?
        }
    }
}

#[cfg(test)]
mod complex_f64_l1_norm_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_l1_norm_nonnegative(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_l1_norm_nonnegative(z)?
        }

        #[test]
        fn prop_approx_l1_norm_point_separating(z1 in super::strategy_complex_f64_any(), z2 in super::strategy_complex_f64_any()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_l1_norm_point_separating(z1, z2, 1e-8)?
        }
    }
}

#[cfg(test)]
mod complex_i32_l1_norm_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_l1_norm_nonnegative(z in super::strategy_complex_i32_any()) {
            let z: super::Complex<i32> = z;
            super::prop_l1_norm_nonnegative(z)?
        }

        #[test]
        fn prop_l1_norm_point_separating(z1 in super::strategy_complex_i32_any(), z2 in super::strategy_complex_i32_any()) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_l1_norm_point_separating(z1, z2)?
        }

        #[test]
        fn prop_l1_norm_homogeneous(z in super::strategy_complex_i32_any(), c in super::strategy_scalar_i32_any()) {
            let z: super::Complex<i32> = z;
            let c: i32 = c;
            super::prop_l1_norm_homogeneous(z, c)?
        }

        #[test]
        fn prop_l1_norm_triangle_inequality(z1 in super::strategy_complex_i32_any(), z2 in super::strategy_complex_i32_any()) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_l1_norm_triangle_inequality(z1, z2)?
        }
    }
}

#[cfg(test)]
mod complex_f64_exp_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_exp_sum(z1 in super::strategy_complex_f64_exp(), z2 in super::strategy_complex_f64_exp()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_exp_sum(z1, z2, 1e-10, 1e-10)?
        }

        #[test]
        fn prop_exp_complex_nonzero(z in super::strategy_complex_f64_exp()) {
            let z: super::Complex<f64> = z;
            super::prop_exp_complex_nonzero(z)?
        }

        #[test]
        fn prop_approx_exp_complex_exp_negative_complex(z in super::strategy_complex_f64_exp()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_exp_complex_exp_negative_complex(z, 1e-10)?
        }
    }
}

#[cfg(test)]
mod complex_f64_ln_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_ln_product(z1 in super::strategy_complex_f64_ln(), z2 in super::strategy_complex_f64_ln()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_ln_product(z1, z2, 1e-10)?
        }

        #[test]
        fn prop_approx_complex_ln_real_part(z in super::strategy_complex_f64_ln()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_complex_ln_real_part(z, 1e-10)?
        }
    }
}

#[cfg(test)]
mod complex_f64_exp_ln_props {
    use proptest::prelude::*;
    proptest!{
        #[test]
        fn prop_approx_exp_ln_identity(z in super::strategy_complex_f64_ln()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_exp_ln_identity(z, 1e-10)?
        }

        #[test]
        fn prop_approx_ln_exp_identity_up_to_phase(z in super::strategy_complex_f64_ln()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_ln_exp_identity_up_to_phase(z, 1e-10)?
        }
    }
}

#[cfg(test)]
mod complex_f64_arg_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_arg_congruent(z in super::strategy_complex_f64_any(), k in super::strategy_scalar_i32_any()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_arg_congruent(z, k, 1e-10)?
        }

        #[test]
        fn prop_approx_arg_complex_times_complex_equals_arg_complex_plus_arg_complex(
            z1 in super::strategy_complex_f64_any(),
            z2 in super::strategy_complex_f64_any()
        ) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_arg_complex_times_complex_equals_arg_complex_plus_arg_complex(z1, z2, 1e-12)?
        }

        #[test]
        fn prop_approx_arg_complex_div_complex_equals_arg_complex_minus_arg_complex(
            z1 in super::strategy_complex_f64_any(), 
            z2 in super::strategy_complex_f64_any()
        ) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_arg_complex_div_complex_equals_arg_complex_minus_arg_complex(z1, z2, 1e-12)?
        }

        #[test]
        fn prop_approx_arg_range(z in super::strategy_complex_f64_any()) {
            let z: super::Complex<f64> = z;
            super::prop_arg_range(z)?
        }
    }
}

#[cfg(test)]
mod complex_f64_sqrt_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_square_root_complex_squared(z in super::strategy_complex_f64_sqrt()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_square_root_complex_squared(z, 1e-10, 1e-10)?
        }

        #[test]
        fn prop_approx_square_root_negative_complex_squared(z in super::strategy_complex_f64_sqrt()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_square_root_negative_complex_squared(z, 1e-10, 1e-10)?
        }

        #[test]
        fn prop_approx_square_root_complex_conjugate_squared(z in super::strategy_complex_f64_sqrt()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_square_root_complex_conjugate_squared(z, 1e-10, 1e-10)?
        }

        #[test]
        fn prop_approx_square_root_product_modulus(z1 in super::strategy_complex_f64_sqrt_product(), z2 in super::strategy_complex_f64_sqrt_product()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_square_root_product_modulus(z1, z2, 1e-10, 1e-10)?
        }

        #[test]
        fn prop_square_root_arg_range(z in super::strategy_complex_f64_sqrt()) {
            let z: super::Complex<f64> = z;
            super::prop_square_root_arg_range(z)?
        }
    }
}

#[cfg(test)]
mod complex_f64_cbrt_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_cubed_root_complex_cubed(z in super::strategy_complex_f64_cbrt()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_cubed_root_complex_cubed(z, 1e-10, 1e-10)?
        }
    }
}

#[cfg(test)]
mod complex_f64_trigonometry_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_cos_real_equals_cos_real(z in super::strategy_imaginary_f64_cos()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_cos_real_equals_cos_real(z, 1e-10)?
        }

        #[test]
        fn prop_approx_cos_negative_z_equals_negative_cos_z(z in super::strategy_imaginary_f64_cos()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_cos_negative_z_equals_cos_z(z, 1e-8)?
        }

        #[test]
        fn prop_approx_sin_real_equals_sin_real(z in super::strategy_imaginary_f64_sin()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_sin_real_equals_sin_real(z, 1e-10)?
        }

        #[test]
        fn prop_approx_sin_negative_z_equals_negative_sin_z(z in super::strategy_imaginary_f64_sin()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_sin_negative_z_equals_negative_sin_z(z, 1e-8)?
        }

        #[test]
        fn prop_approx_tan_real_equals_real_tan(z in super::strategy_real_f64_tan()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_tan_real_equals_real_tan(z, 1e-4)?
        }

        #[test]
        fn prop_approx_tan_negative_z_equals_negative_tan_z(z in super::strategy_complex_f64_tan()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_tan_negative_z_equals_negative_tan_z(z, 1e-8)?
        }

        #[test]
        fn prop_approx_cos_imaginary_equals_imaginary_cosh(z in super::strategy_imaginary_f64_cos()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_cos_imaginary_equals_imaginary_cosh(z, 1e-8)?
        }

        #[test]
        fn prop_approx_sin_imaginary_equals_imaginary_sinh(z in super::strategy_imaginary_f64_sin()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_sin_imaginary_equals_imaginary_sinh(z, 1e-8)?
        }

        #[test]
        fn prop_approx_tan_imaginary_equals_imaginary_tanh(z in super::strategy_imaginary_f64_tan()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_tan_imaginary_equals_imaginary_tanh(z, 1e-8)?
        }

        #[test]
        fn prop_approx_cos_two_times_angle_equals_two_times_cos_angle_squared_minus_sin_angle_squared(z in super::strategy_complex_f64_cos_double_angle()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_cos_two_times_angle_equals_two_times_cos_angle_squared_minus_sin_angle_squared(z, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_approx_sin_two_times_angle_equals_two_times_sin_angle_times_cos_angle(z in super::strategy_complex_f64_sin_double_angle()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_sin_two_times_angle_equals_two_times_sin_angle_times_cos_angle(z, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_approx_tan_two_times_angle(z in super::strategy_complex_f64_tan_double_angle()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_tan_two_times_angle(z, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_approx_cos_angle_sum(z1 in super::strategy_complex_f64_cos_angle_sum(), z2 in super::strategy_complex_f64_cos_angle_sum()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_cos_angle_sum(z1, z2, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_approx_sin_angle_sum(z1 in super::strategy_complex_f64_sin_angle_sum(), z2 in super::strategy_complex_f64_sin_angle_sum()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_sin_angle_sum(z1, z2, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_approx_tan_angle_sum(z1 in super::strategy_complex_f64_tan_angle_sum(), z2 in super::strategy_complex_f64_tan_angle_sum()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_tan_angle_sum(z1, z2, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_tan_angle_difference(
            z1 in super::strategy_complex_f64_tan_angle_difference(), 
            z2 in super::strategy_complex_f64_tan_angle_difference()
        ) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_tan_angle_difference(z1, z2, 1e-8, 1e-8)?
        }
    }
}

#[cfg(test)]
mod complex_f64_trigonometry_inverse_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_acos_conjugate_z_equals_conjugate_acos_z(z in super::strategy_imaginary_f64_cos()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_acos_conjugate_z_equals_conjugate_acos_z(z, 1e-8)?
        }

        #[test]
        fn prop_approx_asin_conjugate_z_equals_conjugate_asin_z(z in super::strategy_imaginary_f64_sin()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_asin_conjugate_z_equals_conjugate_asin_z(z, 1e-8)?
        }

        #[test]
        fn prop_approx_atan_conjugate_z_equals_conjugate_atan_z(z in super::strategy_complex_f64_tan()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_atan_conjugate_z_equals_conjugate_atan_z(z, 1e-8)?
        }

        #[test]
        fn prop_approx_cos_acos_equals_identity(z in super::strategy_imaginary_f64_cos()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_cos_acos_equals_identity(z, 1e-7)?
        }

        #[test]
        fn prop_approx_sin_asin_equals_identity(z in super::strategy_imaginary_f64_sin()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_sin_asin_equals_identity(z, 1e-7)?
        }

        #[test]
        fn prop_approx_tan_atan_equals_identity(z in super::strategy_complex_f64_tan()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_tan_atan_equals_identity(z, 1e-7)?
        }
    }
}

#[cfg(test)]
mod complex_f64_hyperbolic_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_cosh_conjugate_z_equals_conjugate_cosh_z(z in super::strategy_complex_f64_cosh()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_cosh_conjugate_z_equals_conjugate_cosh_z(z)?
        }

        #[test]
        fn prop_approx_cosh_negative_z_equals_negative_cosh_z(z in super::strategy_complex_f64_cosh()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_cosh_negative_z_equals_negative_cosh_z(z, 1e-8)?
        }

        #[test]
        fn prop_approx_sinh_conjugate_z_equals_conjugate_sinh_z(z in super::strategy_complex_f64_sinh()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_sinh_conjugate_z_equals_conjugate_sinh_z(z)?
        }

        #[test]
        fn prop_approx_sinh_negative_z_equals_negative_sinh_z(z in super::strategy_complex_f64_sinh()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_sinh_negative_z_equals_negative_sinh_z(z, 1e-8)?
        }

        #[test]
        fn prop_approx_tanh_conjugate_z_equals_conjugate_tanh_z(z in super::strategy_complex_f64_tanh()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_tanh_conjugate_z_equals_conjugate_tanh_z(z)?
        }

        #[test]
        fn prop_approx_tanh_negative_z_equals_negative_tanh_z(z in super::strategy_complex_f64_tanh()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_tanh_negative_z_equals_negative_tanh_z(z, 1e-8)?
        }

        #[test]
        fn prop_approx_cosh_two_times_angle_equals_cosh_squared_plus_sinh_squared(z in super::strategy_complex_f64_cosh_double_angle()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_cosh_two_times_angle_equals_cosh_squared_plus_sinh_squared(z, 1e-8, 1e-10)?
        }

        #[test]
        fn prop_approx_sinh_two_times_angle_equals_two_times_sinh_cosh(z in super::strategy_complex_f64_sinh_double_angle()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_sinh_two_times_angle_equals_two_times_sinh_cosh(z, 1e-8, 1e-10)?
        }

        #[test]
        fn prop_approx_tanh_two_times_angle(z in super::strategy_complex_f64_tanh_double_angle()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_tanh_two_times_angle(z, 1e-8, 1e-10)?
        }

        #[test]
        fn prop_approx_cosh_angle_sum(z1 in super::strategy_complex_f64_cosh_angle_sum(), z2 in super::strategy_complex_f64_cosh_angle_sum()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_cosh_angle_sum(z1, z2, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_approx_sinh_angle_sum(z1 in super::strategy_complex_f64_sinh_angle_sum(), z2 in super::strategy_complex_f64_sinh_angle_sum()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_sinh_angle_sum(z1, z2, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_approx_tanh_angle_sum(z1 in super::strategy_complex_f64_tanh_angle_sum(), z2 in super::strategy_complex_f64_tanh_angle_sum()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_tanh_angle_sum(z1, z2, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_approx_tanh_angle_difference(
            z1 in super::strategy_complex_f64_tanh_angle_difference(), 
            z2 in super::strategy_complex_f64_tanh_angle_difference()
        ) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_approx_tanh_angle_difference(z1, z2, 1e-8, 1e-8)?
        }
    }
}

#[cfg(test)]
mod complex_f64_hyperbolic_inverse_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_cosh_acosh_equals_identity(z in super::strategy_complex_f64_acosh()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_cosh_acosh_equals_identity(z, 1e-8)?
        }

        #[test]
        fn prop_approx_sinh_asinh_equals_identity(z in super::strategy_complex_f64_asinh()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_sinh_asinh_equals_identity(z, 1e-8)?
        }

        #[test]
        fn prop_approx_tanh_atanh_equals_identity(z in super::strategy_complex_f64_atanh()) {
            let z: super::Complex<f64> = z;
            super::prop_approx_tanh_atanh_equals_identity(z, 1e-6)?
        }
    }
}

