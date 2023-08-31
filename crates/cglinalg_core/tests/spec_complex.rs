extern crate cglinalg_core;
extern crate num_traits;
extern crate proptest;


use std::ops::RangeInclusive;

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


fn any_scalar<S>() -> impl Strategy<Value = S>
where
    S: SimdScalar + Arbitrary
{
    any::<S>().prop_map(|scalar| {
        let modulus = num_traits::cast(100_000_000).unwrap();

        scalar % modulus
    })
}

fn any_complex<S>() -> impl Strategy<Value = Complex<S>> 
where 
    S: SimdScalar + Arbitrary
{
    any::<(S, S)>().prop_map(|(re, im)| {
        let modulus: S = num_traits::cast(100_000_000).unwrap();
        let complex = Complex::new(re, im);

        complex % modulus
    })
    .no_shrink()
}

fn any_complex_modulus_squared_f64() -> impl Strategy<Value = Complex<f64>> {
    use cglinalg_core::Radians;

    any::<(f64, f64)>().prop_map(|(_scale, _angle)| {
        let min_scale = f64::sqrt(f64::EPSILON);
        let max_scale = f64::sqrt(f64::MAX);
        let scale = min_scale + (_scale % (max_scale - min_scale));
        let angle = Radians(_angle % core::f64::consts::FRAC_PI_2);

        Complex::from_polar_decomposition(scale, angle)
    })
    .no_shrink()
}

fn any_complex_modulus_squared_i32() -> impl Strategy<Value = Complex<i32>> {
    any::<(i32, i32)>().prop_map(|(_re, _im)| {
        let min_value = 0;
        // let max_square_root = f64::floor(f64::sqrt(i32::MAX as f64)) as i32;
        let max_square_root = 46340;
        let max_value = max_square_root / 2;
        let re = min_value + (_re % (max_value - min_value + 1));
        let im = min_value + (_im % (max_value - min_value + 1));
        
        Complex::new(re, im)
    })
    .no_shrink()
}

fn imaginary_from_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Complex<S>>
where 
    S: SimdScalarFloat + Arbitrary,
    RangeInclusive<S>: Strategy<Value = S>
{
    (min_value..=max_value)
        .prop_map(Complex::from_imaginary)
        .no_shrink()
}

fn real_from_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Complex<S>>
where 
    S: SimdScalarFloat + Arbitrary,
    RangeInclusive<S>: Strategy<Value = S>
{
    (min_value..=max_value)
        .prop_map(Complex::from_real)
        .no_shrink()
}

fn complex_from_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Complex<S>>
where 
    S: SimdScalarFloat + Arbitrary,
    RangeInclusive<S>: Strategy<Value = S>
{
    let generator = (min_value..=max_value, min_value..=max_value);
    generator
        .prop_map(|(re, im)| Complex::new(re, im))
        .no_shrink()
}

fn sqrt_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, f64::sqrt(f64::MAX) / f64::sqrt(2_f64))
}

fn cos_strategy_f64() -> impl Strategy<Value = Complex<f64>>{
    imaginary_from_range(f64::EPSILON, f64::ln(f64::MAX))
}

fn sin_strategy_f64() -> impl Strategy<Value = Complex<f64>>{
    imaginary_from_range(f64::EPSILON, f64::ln(f64::MAX))
}

fn tan_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 100_f64)
}

fn tan_strategy_real_f64() -> impl Strategy<Value = Complex<f64>> {
    real_from_range(f64::EPSILON, 100_f64)
}

fn tan_strategy_imaginary_f64() -> impl Strategy<Value = Complex<f64>> {
    imaginary_from_range(f64::EPSILON, 200_f64)
}

fn cos_double_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 100_f64)
}

fn sin_double_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 100_f64)
}

fn tan_double_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 100_f64)
}

fn cos_angle_sum_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 100_f64)
}

fn sin_angle_sum_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 100_f64)
}

fn tan_angle_sum_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 100_f64)
}

fn cosh_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, f64::ln(f64::MAX))
}

fn sinh_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, f64::ln(f64::MAX))
}

fn tanh_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 200_f64)
}

fn cosh_double_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 200_f64)
}

fn sinh_double_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 200_f64)
}

fn tanh_double_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 100_f64)
}

fn cosh_angle_sum_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 100_f64)
}

fn sinh_angle_sum_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 100_f64)
}

fn tanh_angle_sum_strategy_f64() -> impl Strategy<Value = Complex<f64>> {
    complex_from_range(f64::EPSILON, 100_f64)
}


/// A scalar `0` times a complex number should be a zero complex number.
///
/// Given a complex number `z`, it satisfies
/// ```text
/// 0 * z = 0.
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
/// z * 0 = 0
/// ```
fn prop_complex_times_zero_equals_zero<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let zero: S = num_traits::zero();
    let zero_complex = Complex::zero();

    prop_assert_eq!(z * zero, zero_complex);
    
    Ok(())
}

/// A zero complex number should act as the additive unit element of a set 
/// of complex numbers.
///
/// Given a complex number `z`
/// ```text
/// z + 0 = z
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
/// 0 + z = z
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
/// 1 * z = z
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
/// z * 1 = z.
/// ```
fn prop_complex_times_one_equals_complex<S>(z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let one: S = num_traits::one();

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
///  z1 +  z2 = &z1 +  z2
///  z1 +  z2 =  z1 + &z2
///  z1 +  z2 = &z1 + &z2
///  z1 + &z2 = &z1 +  z2
/// &z1 +  z2 =  z1 + &z2
/// &z1 +  z2 = &z1 + &z2
///  z1 + &z2 = &z1 + &z2
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
/// z1 + z2 = z2 + z1
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
/// (z1 + z2) + z3 = z1 + (z2 + z3)
/// ```
fn prop_complex_addition_associative<S>(z1: Complex<S>, z2: Complex<S>, z3: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    prop_assert_eq!((z1 + z2) + z3, z1 + (z2 + z3));

    Ok(())
}

/// Complex number addition over floating point scalars should be 
/// approximately associative. 
///
/// Given complex numbers `z1`, `z2`, and `z3` we have
/// ```text
/// (z1 + z2) + z3 ~= z1 + (z2 + z3).
/// ```
fn prop_complex_addition_almost_associative<S>(z1: Complex<S>, z2: Complex<S>, z3: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert!(
        relative_eq!((z1 + z2) + z3, z1 + (z2 + z3), epsilon = tolerance)
    );

    Ok(())
}

/// The zero complex number should act as an additive unit. 
///
/// Given a complex number `z`, we have
/// ```text
/// z - 0 = z
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
/// z - z = z + (-z) = (-z) + z = 0
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
///  z1 -  z2 = &z1 -  z2
///  z1 -  z2 =  z1 - &z2
///  z1 -  z2 = &z1 - &z2
///  z1 - &z2 = &z1 -  z2
/// &z1 -  z2 =  z1 - &z2
/// &z1 -  z2 = &z1 - &z2
///  z1 - &z2 = &z1 - &z2
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
/// c * z = z * c
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
/// z * 1 = 1 * z = z
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
/// z * z_inv ~= z_inv * z ~= 1
/// ```
fn prop_complex_approx_multiplicative_inverse<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
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
/// Given scalars `a` and `b`, and a complex number `z`, we have
/// ```text
/// (a * b) * z = a * (b * z)
/// ```
fn prop_scalar_multiplication_compatibility<S>(a: S, b: S, z: Complex<S>) -> Result<(), TestCaseError>
where
    S: SimdScalar
{
    let a_complex = Complex::from_real(a);
    let b_complex = Complex::from_real(b);

    prop_assert_eq!(a_complex * (b_complex * z), (a_complex * b_complex) * z);

    Ok(())
}

/// Complex number multiplication over integer scalars is exactly associative.
///
/// Given complex numbers `z1`, `z2`, and `z3`, we have
/// ```text
/// (z1 * z2) * z3 = z1 * (z2 * z3)
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
/// z1 * z2 = z2 * z1
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
/// a * (z1 + z2) = a * z1 + a * z2
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
/// (a + b) * z = a * z + b * z
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
/// (z1 + z2) * a = z1 * a + z2 * a
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
/// z * (a + b) = z * a + z * b
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
/// (z1 + z2) * z3 = z1 * z3 + z2 * z3
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
/// z1 * (z2 + z3) = z1 * z2 + z1 * z3
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
/// z** = conjugate(conjugate(z)) = z
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
/// conjugate(z1 + z2) = conjugate(z1) + conjugate(z2)
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
/// conjugate(z1 * z2) = conjugate(z2) * conjugate(z1)
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
    let zero = num_traits::zero();

    prop_assert!(z.modulus_squared() >= zero);
    
    Ok(())
}

/// The squared modulus function is point separating. In particular, if 
/// the squared distance between two complex numbers `z1` and `z2` is 
/// zero, then `z1 = z2`.
///
/// Given complex numbers `z1` and `z2`
/// ```text
/// modulus_squared(z1 - z2) = 0 => z1 = z2 
/// ```
/// Equivalently, if `z1` is not equal to `z2`, then their squared distance is 
/// nonzero
/// ```text
/// z1 != z2 => modulus_squared(z1 - z2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_modulus_squared_approx_point_separating<S>(z1: Complex<S>, z2: Complex<S>, input_tolerance: S, output_tolerance: S) -> Result<(), TestCaseError>
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
/// magnitude_squared(z) = modulus_squared(z)
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
/// norm_squared(z) = modulus_squared(z)
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
/// zero, then `z1 = z2`.
///
/// Given complex numbers `z1` and `z2`
/// ```text
/// modulus_squared(z1 - z2) = 0 => z1 = z2 
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
    let zero = num_traits::zero();

    // prop_assume!(z1 != z2);
    prop_assert_ne!((z1 - z2).modulus_squared(), zero);

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
    let zero = num_traits::zero();

    prop_assert!(z.modulus() >= zero);

    Ok(())
}

/// The modulus function is point separating. In particular, if 
/// the distance between two complex numbers `z1` and `z2` is 
/// zero, then `z1 = z2`.
///
/// Given complex numbers `z1` and `z2`
/// ```text
/// modulus(z1 - z2) = 0 => z1 = z2 
/// ```
/// Equivalently, if `z1` is not equal to `z2`, then their distance is 
/// nonzero
/// ```text
/// z1 != z2 => modulus(z1 - z2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_modulus_approx_point_separating<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let zero = num_traits::zero();

    prop_assume!(relative_ne!(z1, z2, epsilon = tolerance));
    prop_assert!(relative_ne!((z1 - z2).modulus(), zero, epsilon = tolerance));

    Ok(())
}

/// The [`Complex::magnitude`] function and the [`Complex::modulus`] function 
/// are synonyms. In particular, given a complex number `z`
/// ```text
/// magnitude(z) = norm(z)
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
/// norm(z) = modulus(z)
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
/// l2_norm(z) = modulus(z)
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
    let zero = num_traits::zero();

    prop_assert!(z.l1_norm() >= zero);

    Ok(())
}

/// The **L1** norm function is point separating. In particular, if 
/// the distance between two complex numbers `z1` and `z2` is 
/// zero, then `z1 = z2`.
///
/// Given complex numbers `z1` and `z2`
/// ```text
/// l1_norm(z1 - z2) = 0 => z1 = z2 
/// ```
/// Equivalently, if `z1` is not equal to `z2`, then their distance is 
/// nonzero
/// ```text
/// z1 != z2 => l1_norm(z1 - z2) != 0
/// ```
/// For the sake of testability, we use the second form to test the 
/// norm function.
fn prop_l1_norm_approx_point_separating<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assume!(relative_ne!(z1, z2, epsilon = tolerance));
    prop_assert!((z1 - z2).l1_norm() > tolerance);

    Ok(())
}

/// The **L1** norm function is point separating. In particular, if 
/// the distance between two complex numbers `z1` and `z2` is 
/// zero, then `z1 = z2`.
///
/// Given complex numbers `z1` and `z2`
/// ```text
/// l1_norm(z1 - z2) = 0 => z1 = z2 
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
    let zero = num_traits::zero();

    prop_assume!(z1 != z2);
    prop_assert_ne!((z1 - z2).l1_norm(), zero);

    Ok(())
}

/// The square of the positive square root of a complex number is the original
/// complex number.
/// 
/// Given a complex number `z`
/// ```text
/// sqrt(z) * sqrt(z) == z
/// ```
fn prop_positive_square_root_squared<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let sqrt_z = z.sqrt();

    prop_assert!(relative_eq!(sqrt_z * sqrt_z, z, epsilon = tolerance, max_relative = tolerance));

    Ok(())
}

/// The square of the negative square root of a complex number is the original
/// complex number.
/// 
/// Given a complex number `z`
/// ```text
/// -sqrt(z) * -sqrt(z) == z
/// ```
fn prop_negative_square_root_squared<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let minus_sqrt_z = -z.sqrt();

    prop_assert!(relative_eq!(minus_sqrt_z * minus_sqrt_z, z, epsilon = tolerance, max_relative = tolerance));

    Ok(())
}

/// The cosine of a complex number with imaginary part zero equals the 
/// cosine of the real part.
/// 
/// Given a complex number `z` with imaginary part `im(z) == 0`
/// ```text
/// cos(z) = cos(re(z))
/// ```
fn prop_cos_real_equals_cos_real<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where 
    S: SimdScalarFloat
{
    let re_z = z.real();
    let z_re = Complex::from_real(re_z);

    prop_assert!(relative_eq!(z_re.cos().real(), re_z.cos(), epsilon = tolerance));

    Ok(())
}

/// The cosine of a complex number with real part zero equals i times the 
/// hyperbolic cosine of the imaginary part.
/// 
/// Given a complex number `z` with real part `re(z) == 0`
/// ```text
/// cos(z) = i * cosh(im(z))
/// ```
fn prop_cos_imaginary_equals_imaginary_cosh<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
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
/// sin(z) = sin(re(z))
/// ```
fn prop_sin_real_equals_sin_real<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let re_z = z.real();
    let z_re = Complex::from_real(re_z);

    prop_assert!(relative_eq!(z_re.sin().real(), re_z.sin(), epsilon = tolerance));

    Ok(())
}

/// The sine of a complex number with real part zero equals i times the 
/// hyperbolic sine of the imaginary part.
/// 
/// Given a complex number `z` with real part `re(z) == 0`
/// ```text
/// sin(z) = i * sinh(im(z))
/// ```
fn prop_sin_imaginary_equals_imaginary_sinh<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
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
/// tan(z) = tan(re(z))
/// ```
fn prop_tan_real_equals_real_tan<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
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

/// The tangent of a complex number with real part zero equals i times the
/// hyperbolic tangent of the imaginary part.
/// 
/// Given a complex number `z` such that `re(z) == 0`
/// ```text
/// tan(z) = i * tanh(im(z))
/// ```
fn prop_tan_imaginary_equals_imaginary_tanh<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
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
/// cos(2 * z) = cos(z) * cos(z) - sin(z) * sin(z)
/// ```
fn prop_cos_two_times_angle_equals_two_times_cos_angle_squared_minus_sin_angle_squared<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let two = Complex::from_real(num_traits::cast(2).unwrap());
    let lhs = (two * z).cos();
    let cos_z_squared = z.cos().squared();
    let sin_z_squared = z.sin().squared();
    let rhs = cos_z_squared - sin_z_squared;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = tolerance));

    Ok(())
}

/// The complex sine function satisfies the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// sin(2 * z) = 2 * sin(z) * cos(z)
/// ```
fn prop_sin_two_times_angle_equals_two_times_sin_angle_times_cos_angle<S>(z: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
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
/// tan(z) * (1 - tan(z) * tan(z)) = 2 * tan(z)
/// ```
fn prop_tan_two_times_angle<S>(z: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
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
/// cos(z1 + z2) = cos(z1) * cos(z2) - sin(z1) * cos(z2)
/// ```
fn prop_cos_angle_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
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
/// sin(z1 + z2) = sin(z1) * cos(z2) + cos(z1) * sin(z2)
/// ```
fn prop_sin_angle_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
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
/// tan(z1 + z2) * (1 - tan(z1) * tan(z2)) = tan(z1) + tan(z2)
/// ```
fn prop_tan_angle_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let one = Complex::one();
    let lhs = (z1 + z2).tan() * (one - z1.tan() * z2.tan());
    let rhs = z1.tan() + z2.tan();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The cosine and arccosine functions satisfy the following relation.
/// 
/// Given a complex number `z`
/// ```text
/// cos(acos(z)) == z
/// ```
fn prop_cos_acos_equals_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError> 
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
fn prop_sin_asin_equals_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError> 
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
fn prop_tan_atan_equals_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError> 
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
/// cosh(conjugate(z)) = conjugate(cosh(z))
/// ```
fn prop_cosh_conjugate_z_equals_conjugate_cosh_z<S>(z: Complex<S>) -> Result<(), TestCaseError>
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
/// cosh(-z) = cosh(z)
/// ```
fn prop_cosh_negative_z_equals_negative_cosh_z<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
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
/// sinh(conjugate(z)) = conjugate(sinh(z))
/// ```
fn prop_sinh_conjugate_z_equals_conjugate_sinh_z<S>(z: Complex<S>) -> Result<(), TestCaseError>
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
/// sinh(-z) = -sinh(z)
/// ```
fn prop_sinh_negative_z_equals_negative_sinh_z<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
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
/// tanh(conjugate(z)) = conjugate(tanh(z))
/// ```
fn prop_tanh_conjugate_z_equals_conjugate_tanh_z<S>(z: Complex<S>) -> Result<(), TestCaseError>
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
/// tanh(-z) = -tanh(z)
/// ```
fn prop_tanh_negative_z_equals_negative_tanh_z<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError>
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
/// cosh(2 * z) = cosh(z) * cosh(z) + sinh(z) * sinh(z)
/// ```
fn prop_cosh_two_times_angle_equals_cosh_squared_plus_sinh_squared<S>(z: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
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
/// sinh(2 * z) = 2 * sinh(z) * cosh(z)
/// ```
fn prop_sinh_two_times_angle_equals_two_times_sinh_cosh<S>(z: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
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
/// tanh(2 * z) * (1 + tanh(z) * tanh(z)) = 2 * tanh(z)
/// ```
fn prop_tanh_two_times_angle<S>(z: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
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
/// cosh(z1 + z2) = cosh(z1) * cosh(z2) + sinh(z1) * sinh(z2)
/// ```
fn prop_cosh_angle_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
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
/// sinh(z1 + z2) = sinh(z1) * cosh(z2) + cosh(z1) * sinh(z2)
/// ```
fn prop_sinh_angle_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
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
/// tanh(z1 + z2) * (1 + tanh(z1) * tanh(z2)) = tanh(z1) + tanh(z2)
/// ```
fn prop_tanh_angle_sum<S>(z1: Complex<S>, z2: Complex<S>, tolerance: S, max_relative: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let one = Complex::one();
    let lhs = (z1 + z2).tanh() * (one + z1.tanh() * z2.tanh());
    let rhs = z1.tanh() + z2.tanh();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

/// The hyperbolic cosine and hyperbolic arccosine functions satisfy the 
/// following relation.
/// 
/// Given a complex number `z`
/// ```text
/// cosh(acosh(z)) == z
/// ```
fn prop_cosh_acosh_equals_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError> 
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
fn prop_sinh_asinh_equals_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError> 
where
    S: SimdScalarFloat
{
    let lhs = z.asinh().sinh();
    let rhs = z;
    
    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// The hyperbolic tangent and hyperbolic arctangent functions satisfy the 
/// following relation.
/// 
/// Given a complex number `z`
/// ```text
/// tanh(atanh(z)) == z
/// ```
fn prop_tanh_atanh_equals_identity<S>(z: Complex<S>, tolerance: S) -> Result<(), TestCaseError> 
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
        fn prop_zero_times_complex_equals_zero(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_zero_times_complex_equals_zero(z)?
        }
    
        #[test]
        fn prop_complex_times_zero_equals_zero(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_times_zero_equals_zero(z)?
        }

        #[test]
        fn prop_complex_plus_zero_equals_complex(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_plus_zero_equals_complex(z)?
        }

        #[test]
        fn prop_zero_plus_complex_equals_complex(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_zero_plus_complex_equals_complex(z)?
        }

        #[test]
        fn prop_one_times_complex_equal_complex(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_one_times_complex_equal_complex(z)?
        }

        #[test]
        fn prop_complex_times_one_equals_complex(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_times_one_equals_complex(z)?
        }

        #[test]
        fn prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex(z1 in super::any_complex(), z2 in super::any_complex()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex2(z1, z2)?
        }

        #[test]
        fn prop_complex_addition_commutative(z1 in super::any_complex(), z2 in super::any_complex()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_complex_addition_commutative(z1, z2)?
        }

        #[test]
        fn prop_complex_addition_almost_associative(z1 in super::any_complex(), z2 in super::any_complex(), z3 in super::any_complex()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            let z3: super::Complex<f64> = z3;
            super::prop_complex_addition_almost_associative(z1, z2, z3, 1e-7)?
        }

        #[test]
        fn prop_complex_minus_zero_equals_complex(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_minus_zero_equals_complex(z)?
        }

        #[test]
        fn prop_complex_minus_complex_equals_zero(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_minus_complex_equals_zero(z)?
        }

        #[test]
        fn prop_complex1_minus_complex2_equals_refcomplex1_minus_refcomplex2(z1 in super::any_complex(), z2 in super::any_complex()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_complex1_minus_complex2_equals_refcomplex1_minus_refcomplex2(z1, z2)?
        }

        #[test]
        fn prop_scalar_times_complex_equals_complex_times_scalar(c in super::any_scalar(), z in super::any_complex()) {
            let c: f64 = c;
            let z: super::Complex<f64> = z;
            super::prop_scalar_times_complex_equals_complex_times_scalar(c, z)?
        }

        #[test]
        fn prop_complex_multiplicative_unit(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_multiplicative_unit(z)?
        }

        #[test]
        fn prop_complex_approx_multiplicative_inverse(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_approx_multiplicative_inverse(z, 1e-7)?
        }

        #[test]
        fn prop_complex_multiplication_commutative(z1 in super::any_complex(), z2 in super::any_complex()) {
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
        fn prop_zero_times_complex_equals_zero(z in super::any_complex()) {
            let z: super::Complex<i32> = z;
            super::prop_zero_times_complex_equals_zero(z)?
        }
    
        #[test]
        fn prop_complex_times_zero_equals_zero(z in super::any_complex()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_times_zero_equals_zero(z)?
        }

        #[test]
        fn prop_complex_plus_zero_equals_complex(z in super::any_complex()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_plus_zero_equals_complex(z)?
        }

        #[test]
        fn prop_zero_plus_complex_equals_complex(z in super::any_complex()) {
            let z: super::Complex<i32> = z;
            super::prop_zero_plus_complex_equals_complex(z)?
        }

        #[test]
        fn prop_one_times_complex_equal_complex(z in super::any_complex()) {
            let z: super::Complex<i32> = z;
            super::prop_one_times_complex_equal_complex(z)?
        }

        #[test]
        fn prop_complex_times_one_equals_complex(z in super::any_complex()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_times_one_equals_complex(z)?
        }

        #[test]
        fn prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex2(z1 in super::any_complex(), z2 in super::any_complex()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_complex1_plus_complex2_equals_refcomplex1_plus_refcomplex2(z1, z2)?
        }

        #[test]
        fn prop_complex_addition_commutative(z1 in super::any_complex(), z2 in super::any_complex()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_complex_addition_commutative(z1, z2)?
        }

        #[test]
        fn prop_complex_addition_associative(z1 in super::any_complex(), z2 in super::any_complex(), z3 in super::any_complex()) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            let z3: super::Complex<i32> = z3;
            super::prop_complex_addition_associative(z1, z2, z3)?
        }

        #[test]
        fn prop_complex_minus_zero_equals_complex(z in super::any_complex()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_minus_zero_equals_complex(z)?
        }

        #[test]
        fn prop_complex_minus_complex_equals_zero(z in super::any_complex()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_minus_complex_equals_zero(z)?
        }

        #[test]
        fn prop_complex1_minus_complex2_equals_refcomplex1_minus_refcomplex2(z1 in super::any_complex(), z2 in super::any_complex()) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_complex1_minus_complex2_equals_refcomplex1_minus_refcomplex2(z1, z2)?
        }

        #[test]
        fn prop_scalar_times_complex_equals_complex_times_scalar(c in super::any_scalar(), z in super::any_complex()) {
            let c: i32 = c;
            let z: super::Complex<i32> = z;
            super::prop_scalar_times_complex_equals_complex_times_scalar(c, z)?
        }

        #[test]
        fn prop_scalar_multiplication_compatibility(a in super::any_scalar(), b in super::any_scalar(), z in super::any_complex()) {
            let a: i32 = a;
            let b: i32 = b;
            let z: super::Complex<i32> = z;
            super::prop_scalar_multiplication_compatibility(a, b, z)?
        }

        #[test]
        fn prop_complex_multiplication_associative(z1 in super::any_complex(), z2 in super::any_complex(), z3 in super::any_complex()) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            let z3: super::Complex<i32> = z3;
            super::prop_complex_multiplication_associative(z1, z2, z3)?
        }

        #[test]
        fn prop_complex_multiplicative_unit(z in super::any_complex()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_multiplicative_unit(z)?
        }

        #[test]
        fn prop_complex_multiplication_commutative(z1 in super::any_complex(), z2 in super::any_complex()) {
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
        fn prop_distribution_over_complex_addition(a in super::any_scalar(), z1 in super::any_complex(), z2 in super::any_complex()) {
            let a: i32 = a;
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_distribution_over_complex_addition(a, z1, z2)?
        }

        #[test]
        fn prop_distribution_over_scalar_addition(a in super::any_scalar(), b in super::any_scalar(), z in super::any_complex()) {
            let a: i32 = a;
            let b: i32 = b;
            let z: super::Complex<i32> = z;
            super::prop_distribution_over_scalar_addition(a, b, z)?
        }

        #[test]
        fn prop_distribution_over_complex_addition1(a in super::any_scalar(), z1 in super::any_complex(), z2 in super::any_complex()) {
            let a: i32 = a;
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_distribution_over_complex_addition1(a, z1, z2)?
        }

        #[test]
        fn prop_distribution_over_scalar_addition1(a in super::any_scalar(), b in super::any_scalar(), z in super::any_complex()) {
            let a: i32 = a;
            let b: i32 = b;
            let z: super::Complex<i32> = z;
            super::prop_distribution_over_scalar_addition1(a, b, z)?
        }

        #[test]
        fn prop_complex_multiplication_right_distributive(z1 in super::any_complex(), z2 in super::any_complex(), z3 in super::any_complex()) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            let z3: super::Complex<i32> = z3;
            super::prop_complex_multiplication_right_distributive(z1, z2, z3)?
        }

        #[test]
        fn prop_complex_multiplication_left_distributive(z1 in super::any_complex(), z2 in super::any_complex(), z3 in super::any_complex()) {
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
        fn prop_complex_conjugate_conjugate_equals_complex(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_complex_conjugate_conjugate_equals_complex(z)?
        }

        #[test]
        fn prop_complex_conjugation_linear(z1 in super::any_complex(), z2 in super::any_complex()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_complex_conjugation_linear(z1, z2)?
        }
    }
}


#[cfg(test)]
mod complex_i32_conjugation_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_complex_conjugate_conjugate_equals_complex(z in super::any_complex()) {
            let z: super::Complex<i32> = z;
            super::prop_complex_conjugate_conjugate_equals_complex(z)?
        }

        #[test]
        fn prop_complex_conjugation_linear(z1 in super::any_complex(), z2 in super::any_complex()) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_complex_conjugation_linear(z1, z2)?
        }

        #[test]
        fn prop_complex_conjugation_transposes_products(z1 in super::any_complex(), z2 in super::any_complex()) {
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
        fn prop_modulus_squared_nonnegative(z in super::any_complex_modulus_squared_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_modulus_squared_nonnegative(z)?
        }

        #[test]
        fn prop_modulus_squared_approx_point_separating(z1 in super::any_complex_modulus_squared_f64(), z2 in super::any_complex_modulus_squared_f64()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_modulus_squared_approx_point_separating(z1, z2, 1e-10, 1e-20)?
        }
    }
}

#[cfg(test)]
mod complex_f64_modulus_squared_synonym_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_magnitude_squared_modulus_squared_synonyms(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_magnitude_squared_modulus_squared_synonyms(z)?
        }

        #[test]
        fn prop_norm_squared_modulus_squared_synonyms(z in super::any_complex()) {
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
        fn prop_modulus_squared_nonnegative(z in super::any_complex_modulus_squared_i32()) {
            let z: super::Complex<i32> = z;
            super::prop_modulus_squared_nonnegative(z)?
        }

        #[test]
        fn prop_modulus_squared_point_separating(z1 in super::any_complex_modulus_squared_i32(), z2 in super::any_complex_modulus_squared_i32()) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_modulus_squared_point_separating(z1, z2)?
        }
    }
}

#[cfg(test)]
mod complex_i32_modulus_squared_synonym_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_magnitude_squared_modulus_squared_synonyms(z in super::any_complex()) {
            let z: super::Complex<i32> = z;
            super::prop_magnitude_squared_modulus_squared_synonyms(z)?
        }

        #[test]
        fn prop_norm_squared_modulus_squared_synonyms(z in super::any_complex()) {
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
        fn prop_modulus_nonnegative(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_modulus_nonnegative(z)?
        }

        #[test]
        fn prop_modulus_approx_point_separating(z1 in super::any_complex(), z2 in super::any_complex()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_modulus_approx_point_separating(z1, z2, 1e-8)?
        }
    }
}

#[cfg(test)]
mod complex_f64_modulus_synonym_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_magnitude_modulus_synonyms(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_magnitude_modulus_synonyms(z)?
        }

        #[test]
        fn prop_norm_modulus_synonyms(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_norm_modulus_synonyms(z)?
        }

        #[test]
        fn prop_l2_norm_modulus_synonyms(z in super::any_complex()) {
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
        fn prop_l1_norm_nonnegative(z in super::any_complex()) {
            let z: super::Complex<f64> = z;
            super::prop_l1_norm_nonnegative(z)?
        }

        #[test]
        fn prop_l1_norm_approx_point_separating(z1 in super::any_complex(), z2 in super::any_complex()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_l1_norm_approx_point_separating(z1, z2, 1e-8)?
        }
    }
}

#[cfg(test)]
mod complex_i32_l1_norm_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_l1_norm_nonnegative(z in super::any_complex()) {
            let z: super::Complex<i32> = z;
            super::prop_l1_norm_nonnegative(z)?
        }

        #[test]
        fn prop_l1_norm_point_separating(z1 in super::any_complex(), z2 in super::any_complex()) {
            let z1: super::Complex<i32> = z1;
            let z2: super::Complex<i32> = z2;
            super::prop_l1_norm_point_separating(z1, z2)?
        }
    }
}

#[cfg(test)]
mod complex_f64_sqrt_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_positive_square_root_squared(z in super::sqrt_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_positive_square_root_squared(z, 1e-10)?
        }

        #[test]
        fn prop_negative_square_root_squared(z in super::sqrt_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_negative_square_root_squared(z, 1e-10)?
        }
    }
}

#[cfg(test)]
mod complex_f64_trigonometry_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_cos_real_equals_cos_real(z in super::cos_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_cos_real_equals_cos_real(z, 1e-10)?
        }

        #[test]
        fn prop_sin_real_equals_sin_real(z in super::sin_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_sin_real_equals_sin_real(z, 1e-10)?
        }

        #[test]
        fn prop_tan_real_equals_real_tan(z in super::tan_strategy_real_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_tan_real_equals_real_tan(z, 1e-4)?
        }

        #[test]
        fn prop_cos_imaginary_equals_imaginary_cosh(z in super::cos_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_cos_imaginary_equals_imaginary_cosh(z, 1e-8)?
        }

        #[test]
        fn prop_sin_imaginary_equals_imaginary_sinh(z in super::sin_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_sin_imaginary_equals_imaginary_sinh(z, 1e-8)?
        }

        #[test]
        fn prop_tan_imaginary_equals_imaginary_tanh(z in super::tan_strategy_imaginary_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_tan_imaginary_equals_imaginary_tanh(z, 1e-8)?
        }

        #[test]
        fn prop_cos_two_times_angle_equals_two_times_cos_angle_squared_minus_sin_angle_squared(z in super::cos_double_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_cos_two_times_angle_equals_two_times_cos_angle_squared_minus_sin_angle_squared(z, 1e-8)?
        }

        #[test]
        fn prop_sin_two_times_angle_equals_two_times_sin_angle_times_cos_angle(z in super::sin_double_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_sin_two_times_angle_equals_two_times_sin_angle_times_cos_angle(z, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_tan_two_times_angle(z in super::tan_double_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_tan_two_times_angle(z, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_cos_angle_sum(z1 in super::cos_angle_sum_strategy_f64(), z2 in super::cos_angle_sum_strategy_f64()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_cos_angle_sum(z1, z2, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_sin_angle_sum(z1 in super::sin_angle_sum_strategy_f64(), z2 in super::sin_angle_sum_strategy_f64()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_sin_angle_sum(z1, z2, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_tan_angle_sum(z1 in super::tan_angle_sum_strategy_f64(), z2 in super::tan_angle_sum_strategy_f64()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_tan_angle_sum(z1, z2, 1e-8, 1e-8)?
        }
    }
}

#[cfg(test)]
mod complex_f64_trigonometry_inverse_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_cos_acos_equals_identity(z in super::cos_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_cos_acos_equals_identity(z, 1e-7)?
        }

        #[test]
        fn prop_sin_asin_equals_identity(z in super::sin_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_sin_asin_equals_identity(z, 1e-7)?
        }

        #[test]
        fn prop_tan_atan_equals_identity(z in super::tan_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_tan_atan_equals_identity(z, 1e-7)?
        }
    }
}

#[cfg(test)]
mod complex_f64_hyperbolic_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_cosh_conjugate_z_equals_conjugate_cosh_z(z in super::cosh_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_cosh_conjugate_z_equals_conjugate_cosh_z(z)?
        }

        #[test]
        fn prop_cosh_negative_z_equals_negative_cosh_z(z in super::cosh_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_cosh_negative_z_equals_negative_cosh_z(z, 1e-8)?
        }

        #[test]
        fn prop_sinh_conjugate_z_equals_conjugate_sinh_z(z in super::sinh_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_sinh_conjugate_z_equals_conjugate_sinh_z(z)?
        }

        #[test]
        fn prop_sinh_negative_z_equals_negative_sinh_z(z in super::sinh_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_sinh_negative_z_equals_negative_sinh_z(z, 1e-8)?
        }

        #[test]
        fn prop_tanh_conjugate_z_equals_conjugate_tanh_z(z in super::tanh_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_tanh_conjugate_z_equals_conjugate_tanh_z(z)?
        }

        #[test]
        fn prop_tanh_negative_z_equals_negative_tanh_z(z in super::tanh_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_tanh_negative_z_equals_negative_tanh_z(z, 1e-8)?
        }

        #[test]
        fn prop_cosh_two_times_angle_equals_cosh_squared_plus_sinh_squared(z in super::cosh_double_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_cosh_two_times_angle_equals_cosh_squared_plus_sinh_squared(z, 1e-8, 1e-10)?
        }

        #[test]
        fn prop_sinh_two_times_angle_equals_two_times_sinh_cosh(z in super::sinh_double_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_sinh_two_times_angle_equals_two_times_sinh_cosh(z, 1e-8, 1e-10)?
        }

        #[test]
        fn prop_tanh_two_times_angle(z in super::tanh_double_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_tanh_two_times_angle(z, 1e-8, 1e-10)?
        }

        #[test]
        fn prop_cosh_angle_sum(z1 in super::cosh_angle_sum_strategy_f64(), z2 in super::cosh_angle_sum_strategy_f64()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_cosh_angle_sum(z1, z2, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_sinh_angle_sum(z1 in super::sinh_angle_sum_strategy_f64(), z2 in super::sinh_angle_sum_strategy_f64()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_sinh_angle_sum(z1, z2, 1e-8, 1e-8)?
        }

        #[test]
        fn prop_tanh_angle_sum(z1 in super::tanh_angle_sum_strategy_f64(), z2 in super::tanh_angle_sum_strategy_f64()) {
            let z1: super::Complex<f64> = z1;
            let z2: super::Complex<f64> = z2;
            super::prop_tanh_angle_sum(z1, z2, 1e-8, 1e-8)?
        }
    }
}

#[cfg(test)]
mod complex_f64_hyperbolic_inverse_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_cosh_acosh_equals_identity(z in super::cos_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_cosh_acosh_equals_identity(z, 1e-8)?
        }

        #[test]
        fn prop_sinh_asinh_equals_identity(z in super::sin_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_sinh_asinh_equals_identity(z, 1e-8)?
        }

        #[test]
        fn prop_tanh_atanh_equals_identity(z in super::tan_strategy_f64()) {
            let z: super::Complex<f64> = z;
            super::prop_tanh_atanh_equals_identity(z, 1e-8)?
        }
    }
}