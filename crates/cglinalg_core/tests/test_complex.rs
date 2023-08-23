extern crate cglinalg_core;


#[cfg(test)]
mod index_tests {
    use cglinalg_core::{
        Complex,
    };


    #[test]
    fn test_as_ref() {
        let z = Complex::new(1_i32, 2_i32);
        let z_ref: &[i32; 2] = z.as_ref();

        assert_eq!(z_ref, &[1_i32, 2_i32]);
    }

    #[test]
    fn test_indices_match_components() {
        let z = Complex::new(1_i32, 2_i32);

        assert_eq!(z[0], z.re);
        assert_eq!(z[1], z.im);
    }

    #[test]
    #[should_panic]
    fn test_complex_components_out_of_bounds1() {
        let z = Complex::new(1_i32, 2_i32);

        assert_eq!(z[2], z[2]);
    }

    #[test]
    #[should_panic]
    fn test_complex_components_out_of_bounds2() {
        let z = Complex::new(1_i32, 2_i32);

        assert_eq!(z[usize::MAX], z[usize::MAX]);
    }
}


#[cfg(test)]
mod constructor_tests {
    use cglinalg_core::{
        Complex,
        Radians,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_from_angle1() {
        let angle = Radians(0_f64);
        let expected = Complex::new(1_f64, 0_f64);
        let result = Complex::from_angle(angle);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_from_angle2() {
        let angle = Radians(core::f64::consts::FRAC_PI_2);
        let expected = Complex::new(0_f64, 1_f64);
        let result = Complex::from_angle(angle);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_from_angle3() {
        let angle = Radians(core::f64::consts::PI);
        let expected = Complex::new(-1_f64, 0_f64);
        let result = Complex::from_angle(angle);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_from_angle4() {
        let angle = Radians(3_f64 * core::f64::consts::FRAC_PI_2);
        let expected = Complex::new(0_f64, -1_f64);
        let result = Complex::from_angle(angle);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_from_angle5() {
        let angle = Radians(2_f64 * core::f64::consts::PI);
        let expected = Complex::new(1_f64, 0_f64);
        let result = Complex::from_angle(angle);

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_from_angle_norm() {
        let angle = Radians(493_f64);
        let z = Complex::from_angle(angle);

        assert_relative_eq!(z.norm(), 1_f64);
    }

    #[test]
    fn test_from_polar_decomposition_norm() {
        let angle = Radians(493_f64);
        let radius = 5_f64;
        let z = Complex::from_polar_decomposition(radius, angle);

        assert_eq!(z.norm(), radius);
    }

    #[test]
    fn test_from_polar_decomposition1() {
        let angle = Radians(0_f64);
        let radius = 5_f64;
        let expected = Complex::new(5_f64, 0_f64);
        let result = Complex::from_polar_decomposition(radius, angle);
        
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_from_polar_decomposition2() {
        let angle = Radians(core::f64::consts::FRAC_PI_2);
        let radius = 5_f64;
        let expected = Complex::new(0_f64, 5_f64);
        let result = Complex::from_polar_decomposition(radius, angle);
        
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_from_polar_decomposition3() {
        let angle = Radians(core::f64::consts::PI);
        let radius = 5_f64;
        let expected = Complex::new(-5_f64, 0_f64);
        let result = Complex::from_polar_decomposition(radius, angle);
        
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_from_polar_decomposition4() {
        let angle = Radians(3_f64 * core::f64::consts::FRAC_PI_2);
        let radius = 5_f64;
        let expected = Complex::new(0_f64, -5_f64);
        let result = Complex::from_polar_decomposition(radius, angle);
        
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_from_polar_decomposition5() {
        let angle = Radians(2_f64 * core::f64::consts::PI);
        let radius = 5_f64;
        let expected = Complex::new(5_f64, 0_f64);
        let result = Complex::from_polar_decomposition(radius, angle);
        
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }
}


#[cfg(test)]
mod arithmetic_tests {
    use cglinalg_core::{
        Complex,
    };


    #[test]
    fn test_addition_complex_complex() {
        let z1 = Complex::new(1_i32, 3_i32);
        let z2 = Complex::new(5_i32, 17_i32);
        let expected = Complex::new(6_i32, 20_i32);
        let result = z1 + z2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition_complex_zero() {
        let zero: Complex<i32> = Complex::zero();
        let z = Complex::new(74_i32, 12_i32);

        assert_eq!(z + zero, z);
    }

    #[test]
    fn test_addition_zero_complex() {
        let zero: Complex<i32> = Complex::zero();
        let z = Complex::new(74_i32, 12_i32);

        assert_eq!(zero + z, z);
    }

    #[test]
    fn test_addition_zero_zero() {
        let zero: Complex<i32> = Complex::zero();

        assert_eq!(zero + zero, zero);
    }

    #[test]
    fn test_addition_scalar_complex() {
        let c = 8_i32;
        let z = Complex::new(9_i32, 4_i32);
        let expected = Complex::new(17_i32, 4_i32);
        let result = c + z;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_addition_complex_scalar() {
        let c = 8_i32;
        let z = Complex::new(9_i32, 4_i32);
        let expected = Complex::new(17_i32, 4_i32);
        let result = z + c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction_complex_complex() {
        let z1 = Complex::new(45_i32, 7_i32);
        let z2 = Complex::new(74_i32, 10_i32);
        let expected = Complex::new(-29_i32, -3_i32);
        let result = z1 - z2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction_complex_zero() {
        let zero: Complex<i32> = Complex::zero();
        let z = Complex::new(74_i32, 12_i32);

        assert_eq!(z - zero, z);
    }

    #[test]
    fn test_subtraction_zero_complex() {
        let zero: Complex<i32> = Complex::zero();
        let z = Complex::new(74_i32, 12_i32);

        assert_eq!(zero - z, -z);
    }

    #[test]
    fn test_subtraction_zero_zero() {
        let zero: Complex<i32> = Complex::zero();

        assert_eq!(zero - zero, zero);
    }

    #[test]
    fn test_subtraction_scalar_complex() {
        let c = 7_i32;
        let z = Complex::new(1_i32, 3_i32);
        let expected = Complex::new(6_i32, 3_i32);
        let result = c - z;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction_complex_scalar() {
        let c = 7_i32;
        let z = Complex::new(1_i32, 3_i32);
        let expected = Complex::new(-6_i32, 3_i32);
        let result = z - c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiplication_unit_re_complex() {
        let one: Complex<i32> = Complex::unit_re();
        let z = Complex::new(3_i32, 4_i32);

        assert_eq!(one * z, z);
    }

    #[test]
    fn test_multiplication_complex_unit_re() {
        let one: Complex<i32> = Complex::unit_re();
        let z = Complex::new(3_i32, 4_i32);

        assert_eq!(z * one, z);
    }

    #[test]
    fn test_unit_im_times_unit_im() {
        let i: Complex<i32> = Complex::unit_im();
        let one: Complex<i32> = Complex::unit_re();

        assert_eq!(i * i, -one);
    }

    #[test]
    fn test_multiplication_zero_zero() {
        let zero: Complex<i32> = Complex::zero();

        assert_eq!(zero * zero, zero);
    }

    #[test]
    fn test_multiplication_zero_complex() {
        let zero: Complex<i32> = Complex::zero();
        let z = Complex::new(2_i32, 3_i32);

        assert_eq!(zero * z, zero);
    }

    #[test]
    fn test_multiplication_complex_zero() {
        let zero: Complex<i32> = Complex::zero();
        let z = Complex::new(2_i32, 3_i32);

        assert_eq!(z * zero, zero);
    }

    #[test]
    fn test_multiplication_complex_complex() {
        let z1 = Complex::new(2_i32, 3_i32);
        let z2 = Complex::new(5_i32, 8_i32);
        let expected = Complex::new(-14_i32, 31_i32);
        let result = z1 * z2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiplication_scalar_complex() {
        let c = 3_i32;
        let z = Complex::new(2_i32, 5_i32);
        let expected = Complex::new(6_i32, 15_i32);
        let result = c * z;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiplication_complex_scalar() {
        let c = 3_i32;
        let z = Complex::new(2_i32, 5_i32);
        let expected = Complex::new(6_i32, 15_i32);
        let result = z * c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_division_complex_complex() {
        let z1 = Complex::new(1_f64, 3_f64);
        let z2 = Complex::new(4_f64, 8_f64);
        let expected = Complex::new(7_f64 / 20_f64, 1_f64 / 20_f64);
        let result = z1 / z2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_division_complex_scalar() {
        let c = 5_f64;
        let z = Complex::new(3_f64, 7_f64);
        let expected = Complex::new(3_f64 / 5_f64, 7_f64 / 5_f64);
        let result = z / c;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_division_scalar_complex() {
        let c = 7_f64;
        let z = Complex::new(24_f64, 69_f64);
        let expected = Complex::new(56_f64 / 1779_f64, -161_f64 / 1779_f64);
        let result = c / z;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_division_unit_re_complex() {
        let one: Complex<f64> = Complex::unit_re();
        let z = Complex::new(57_f64, 92_f64);
        let expected = Complex::new(57_f64 / 11713_f64, -92_f64 / 11713_f64);
        let result = one / z;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_division_complex_unit_re() {
        let one: Complex<f64> = Complex::unit_re();
        let z = Complex::new(57_f64, 92_f64);
        let expected = z;
        let result = z / one;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_division_zero_complex() {
        let zero: Complex<i32> = Complex::zero();
        let z = Complex::new(1_i32, 2_i32);

        assert_eq!(zero / z, zero);
    }

    #[test]
    #[should_panic]
    fn test_division_complex_zero() {
        let zero: Complex<i32> = Complex::zero();
        let z = Complex::new(1_i32, 2_i32);

        assert_eq!(z / zero, z / zero);
    }
}


#[cfg(test)]
mod norm_tests {
    use cglinalg_core::{
        Complex,
    };


    #[test]
    fn test_unit_re_should_have_unit_norm() {
        let one: Complex<f64> = Complex::unit_re();

        assert_eq!(one.norm(), 1_f64);
    }

    #[test]
    fn test_unit_im_should_have_unit_norm() {
        let i: Complex<f64> = Complex::unit_im();

        assert_eq!(i.norm(), 1_f64);
    }
}


#[cfg(test)]
mod conjugate_tests {
    use cglinalg_core::{
        Complex,
    };


    #[test]
    fn test_conjugate() {
        let z = Complex::new(1_i32, 2_i32);
        let expected = Complex::new(1_i32, -2_i32);
        let result = z.conjugate();

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod arg_tests {
    use cglinalg_core::{
        Complex,
    };


    #[test]
    fn test_arg_unit_im() {
        let i: Complex<f64> = Complex::unit_im();
        let expected = core::f64::consts::FRAC_PI_2;
        let result = i.arg();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_arg_unit_re() {
        let one: Complex<f64> = Complex::unit_re();
        let expected = 0_f64;
        let result = one.arg();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_arg_minus_unit_im() {
        let minus_i: Complex<f64> = -Complex::unit_im();
        let expected = -core::f64::consts::FRAC_PI_2;
        let result = minus_i.arg();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_arg_minus_unit_re() {
        let minus_one: Complex<f64> = -Complex::unit_re();
        let expected = -core::f64::consts::PI;
        let result = minus_one.arg();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_arg_complex1() {
        let z = Complex::new(1_f64, 1_f64);
        let expected = core::f64::consts::FRAC_PI_4;
        let result = z.arg();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_arg_complex2() {
        let z = Complex::new(1_f64, -1_f64);
        let expected = -core::f64::consts::FRAC_PI_4;
        let result = z.arg();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_arg_complex3() {
        let z = Complex::new(-1_f64, 1_f64);
        let expected = 3_f64 * core::f64::consts::FRAC_PI_4;
        let result = z.arg();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_arg_complex4() {
        let z = Complex::new(-1_f64, -1_f64);
        let expected = -3_f64 * core::f64::consts::FRAC_PI_4;
        let result = z.arg();

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod inverse_tests {
    use cglinalg_core::{
        Complex,
    };


    #[test]
    fn test_inverse_zero() {
        let zero: Complex<f64> = Complex::zero();

        assert!(zero.inverse().is_none());
    }

    #[test]
    fn test_inverse_unit_re() {
        let one: Complex<f64> = Complex::unit_re();

        assert_eq!(one.inverse(), Some(one));
    }

    #[test]
    fn test_inverse_unit_im() {
        let i: Complex<f64> = Complex::unit_im();

        assert_eq!(i.inverse(), Some(-i));
    }

    #[test]
    fn test_inverse_real() {
        let z = Complex::from_real(2_f64);
        let expected = Some(Complex::from_real(1_f64 / 2_f64));
        let result = z.inverse();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_imaginary() {
        let z = Complex::from_imaginary(4_f64);
        let expected = Some(Complex::from_imaginary(-1_f64 / 4_f64));
        let result = z.inverse();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_complex() {
        let z = Complex::new(1_f64, 2_f64);
        let expected = Some(Complex::new(1_f64 / 5_f64, -2_f64 / 5_f64));
        let result = z.inverse();

        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod exp_tests {
    use cglinalg_core::{
        Complex,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_exp_zero() {
        let zero: Complex<f64> = Complex::zero();
        let expected: Complex<f64> = Complex::unit_re();
        let result = zero.exp();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_exp_one() {
        let one: Complex<f64> = Complex::unit_re();
        let expected: Complex<f64> = Complex::from_real(f64::exp(1_f64));
        let result = one.exp();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_exp_i_pi() {
        let i: Complex<f64> = Complex::unit_im();
        let expected: Complex<f64> = Complex::from_real(-1_f64);
        let result = (i * core::f64::consts::PI).exp();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }
}


#[cfg(test)]
mod logarithm_tests {
    use cglinalg_core::{
        Complex,
    };


    #[test]
    fn test_natural_logarithm1() {
        let one: Complex<f64> = Complex::one();
        let zero: Complex<f64> = Complex::zero();
        let expected = zero;
        let result = one.ln();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_natural_logarithm2() {
        let i: Complex<f64> = Complex::unit_im();
        let pi_over_two = core::f64::consts::FRAC_PI_2;
        let expected = -i * pi_over_two;
        let result = (-i).ln();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_natural_logarithm3() {
        let i: Complex<f64> = Complex::unit_im();
        let pi_over_two = core::f64::consts::FRAC_PI_2;
        let expected = i * pi_over_two;
        let result = (i).ln();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_natural_logarithm4() {
        let z = Complex::from_real(7_f64);
        let expected = Complex::new(f64::ln(7_f64), 0_f64);
        let result = z.ln();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_natural_logarithm5() {
        let z = Complex::from_imaginary(7_f64);
        let pi_over_two = core::f64::consts::FRAC_PI_2;
        let expected = Complex::new(f64::ln(7_f64), pi_over_two);
        let result = z.ln();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_natural_logarithm6() {
        let z = Complex::new(1_f64, 3_f64);
        let expected = Complex::new(1.151292546497023_f64, 1.2490457723982544_f64);
        let result = z.ln();
        
        assert_eq!(result, expected);
    }
}


#[cfg(test)]
mod square_root_tests {
    use cglinalg_core::{
        Complex,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_square_root_zero() {
        let zero: Complex<f64> = Complex::zero();

        assert_eq!(zero.sqrt(), zero);
    }

    #[test]
    fn test_square_root_one() {
        let one: Complex<f64> = Complex::one();
        
        assert_eq!(one.sqrt(), one);
    }

    #[test]
    fn test_square_root_real() {
        let z = Complex::from_real(2_f64);
        let expected = Complex::from_real(f64::sqrt(2_f64));
        let result = z.sqrt();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_square_root_i() {
        let i: Complex<f64> = Complex::unit_im();
        let expected = (1_f64 / f64::sqrt(2_f64)) * Complex::new(1_f64, 1_f64);
        let result = i.sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_square_root_minus_i() {
        let i: Complex<f64> = Complex::unit_im();
        let expected = (1_f64 / f64::sqrt(2_f64)) * Complex::new(1_f64, -1_f64);
        let result = (-i).sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_square_root_complex() {
        let z = Complex::new(2_f64, 5_f64);
        let expected = Complex::new(1.9216093264675973_f64, 1.3009928530039094_f64);
        let result = z.sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }
}


#[cfg(test)]
mod trigonometry_tests {
    use cglinalg_core::{
        Complex,
    };
    use approx::{
        assert_relative_eq,
    };
    use core::f64;


    const _0_I0: Complex<f64> = Complex::new(0_f64, 0_f64);
    const _1_I0: Complex<f64> = Complex::new(1_f64, 0_f64);
    const _0_I1: Complex<f64> = Complex::new(0_f64, 1_f64);
    const _NEG1_I0: Complex<f64> = Complex::new(-1_f64, 0_f64);
    const _0_I_NEG1: Complex<f64> = Complex::new(0_f64, -1_f64);
    const _PI_I0: Complex<f64> = Complex::new(f64::consts::PI, 0_f64);
    const _0_IPI: Complex<f64> = Complex::new(0_f64, f64::consts::PI);
    const _0_I_INF: Complex<f64> = Complex::new(0_f64, f64::INFINITY);
    const _0_I_NEGINF: Complex<f64> = Complex::new(0_f64, f64::NEG_INFINITY);
    const _0_I_FRAC_PI_8: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_8);
    const _0_I_FRAC_PI_6: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_6);
    const _0_I_FRAC_PI_4: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_4);
    const _0_I_FRAC_PI_3: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_3);
    const _0_I_FRAC_PI_2: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_2);
    const _0_I_PI: Complex<f64> = Complex::new(0_f64, f64::consts::PI);
    const _0_I_2_PI: Complex<f64> = Complex::new(0_f64, 2_f64 * f64::consts::PI);


    fn test_fn<F>(func: F, z: Complex<f64>, expected: Complex<f64>) 
    where
        F: Fn(Complex<f64>) -> Complex<f64>
    {
        let result = func(z);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }


    #[test]
    fn test_complex_cos() {
        test_fn(Complex::cos, _0_I0, Complex::new(1_f64, 0_f64));
        test_fn(Complex::cos, _1_I0, Complex::new(0.540302305868_f64, 0_f64));
        test_fn(Complex::cos, _NEG1_I0, Complex::new(0.540302305868_f64, 0_f64));
        test_fn(Complex::cos, _0_I1, Complex::new(1.54308063482_f64, 0_f64));
        test_fn(Complex::cos, _0_I_NEG1, Complex::new(1.54308063482_f64, 0_f64));

        test_fn(Complex::cos, _0_I_FRAC_PI_8, Complex::new(1.07810228857_f64, 0_f64));
        test_fn(Complex::cos, _0_I_FRAC_PI_6, Complex::new(1.14023832108_f64, 0_f64));
        test_fn(Complex::cos, _0_I_FRAC_PI_4, Complex::new(1.32460908925_f64, 0_f64));
        test_fn(Complex::cos, _0_I_FRAC_PI_3, Complex::new(1.60028685770_f64, 0_f64));
        test_fn(Complex::cos, _0_I_FRAC_PI_2, Complex::new(2.50917847866_f64, 0_f64));
        test_fn(Complex::cos, _0_I_PI, Complex::new(11.5919532755_f64, 0_f64));
        test_fn(Complex::cos, _0_I_2_PI, Complex::new(267.746761484_f64, 0_f64));
    }

    #[test]
    fn test_complex_acos() {
        test_fn(Complex::acos, _1_I0, Complex::new(0_f64, 0_f64));
        test_fn(Complex::acos, _NEG1_I0, _PI_I0);
        test_fn(Complex::acos, _0_I1, Complex::new(1.57079632679_f64, -0.881373587020_f64));
        test_fn(Complex::acos, _0_I_NEG1, Complex::new(1.57079632679_f64, 0.881373587020_f64));
    }

    #[test]
    fn test_complex_sin() {
        test_fn(Complex::sin, _0_I0, Complex::new(0_f64, 0_f64));
        test_fn(Complex::sin, _1_I0,  Complex::new(0.841470984808_f64, 0_f64));
        test_fn(Complex::sin, _NEG1_I0, Complex::new(-0.841470984808_f64, 0_f64));
        test_fn(Complex::sin, _0_I1, Complex::new(0_f64, 1.17520119364_f64));
        test_fn(Complex::sin, _0_I_NEG1, Complex::new(0_f64, -1.17520119364_f64));

        test_fn(Complex::sin, _0_I_FRAC_PI_8, Complex::new(0_f64, 0.402870381917_f64));
        test_fn(Complex::sin, _0_I_FRAC_PI_6, Complex::new(0_f64, 0.547853473888_f64));
        test_fn(Complex::sin, _0_I_FRAC_PI_4, Complex::new(0_f64, 0.868670961486_f64));
        test_fn(Complex::sin, _0_I_FRAC_PI_3, Complex::new(0_f64, 1.24936705052_f64));
        test_fn(Complex::sin, _0_I_FRAC_PI_2, Complex::new(0_f64, 2.30129890231_f64));
        test_fn(Complex::sin, _0_I_PI, Complex::new(0_f64, 11.5487393573_f64));
        test_fn(Complex::sin, _0_I_2_PI, Complex::new(0_f64, 267.744894041_f64));
    }

    #[test]
    fn test_complex_asin() {
        test_fn(Complex::asin, _1_I0, Complex::new(1.57079632679_f64, 0_f64));
        test_fn(Complex::asin, _NEG1_I0, Complex::new(-1.57079632679_f64, 0_f64));
        test_fn(Complex::asin, _0_I1, Complex::new(0_f64, 0.881373587020_f64));
        test_fn(Complex::asin, _0_I_NEG1, Complex::new(0_f64, -0.881373587020_f64));
    }

    #[test]
    fn test_complex_tan() {
        test_fn(Complex::tan, _0_I0, Complex::new(0_f64, 0_f64));
        test_fn(Complex::tan, _1_I0, Complex::new(1.55740772465_f64, 0_f64));
        test_fn(Complex::tan, _NEG1_I0, Complex::new(-1.55740772465_f64, 0_f64));
        test_fn(Complex::tan, _0_I1, Complex::new(0_f64, 0.761594155956_f64));
        test_fn(Complex::tan, _0_I_NEG1, Complex::new(0_f64, -0.761594155956_f64));

        test_fn(Complex::tan, _0_I_FRAC_PI_8, Complex::new(0_f64, 0.373684747901_f64));
        test_fn(Complex::tan, _0_I_FRAC_PI_6, Complex::new(0_f64, 0.480472778156_f64));
        test_fn(Complex::tan, _0_I_FRAC_PI_4, Complex::new(0_f64, 0.655794202633_f64));
        test_fn(Complex::tan, _0_I_FRAC_PI_3, Complex::new(0_f64, 0.780714435359_f64));
        test_fn(Complex::tan, _0_I_FRAC_PI_2, Complex::new(0_f64, 0.917152335667_f64));
        test_fn(Complex::tan, _0_I_PI, Complex::new(0_f64, 0.996272076221_f64));
        test_fn(Complex::tan, _0_I_2_PI, Complex::new(0_f64, 0.999993025340_f64));
    }

    #[test]
    fn test_complex_atan() {
        test_fn(Complex::atan, _1_I0, Complex::new(0.785398163397_f64, 0_f64));
        test_fn(Complex::atan, _NEG1_I0, Complex::new(-0.785398163397_f64, 0_f64));
        test_fn(Complex::atan, _0_I1, Complex::new(0_f64, f64::INFINITY));
        test_fn(Complex::atan, _0_I_NEG1, Complex::new(0_f64, f64::NEG_INFINITY));
    }

    #[test]
    fn test_complex_sec() {
        test_fn(Complex::sec, _1_I0, Complex::new(1.85081571768_f64, 0_f64));
        test_fn(Complex::sec, _NEG1_I0, Complex::new(1.85081571768_f64, 0_f64));
        test_fn(Complex::sec, _0_I1, Complex::new(0.648054273664_f64, 0_f64));
        test_fn(Complex::sec, _0_I_NEG1, Complex::new(0.648054273664_f64, 0_f64));
    }

    #[test]
    fn test_complex_asec() {
        test_fn(Complex::asec, _1_I0, Complex::new(0_f64, 0_f64));
        test_fn(Complex::asec, _NEG1_I0, Complex::new(3.14159265359_f64, 0_f64));
        test_fn(Complex::asec, _0_I1, Complex::new(1.57079632679_f64, 0.881373587020_f64));
        test_fn(Complex::asec, _0_I_NEG1, Complex::new(1.57079632679_f64, -0.881373587020_f64));
    }

    #[test]
    fn test_complex_csc() {
        test_fn(Complex::csc, _1_I0, Complex::new(1.18839510578_f64, 0_f64));
        test_fn(Complex::csc, _NEG1_I0, Complex::new(-1.18839510578_f64, 0_f64));
        test_fn(Complex::csc, _0_I1, Complex::new(0_f64, -0.850918128239_f64));
        test_fn(Complex::csc, _0_I_NEG1, Complex::new(0_f64, 0.850918128239_f64));
    }

    #[test]
    fn test_complex_acsc() {
        test_fn(Complex::acsc, _1_I0, Complex::new(1.57079632679_f64, 0_f64));
        test_fn(Complex::acsc, _NEG1_I0, Complex::new(-1.57079632679_f64, 0_f64));
        test_fn(Complex::acsc, _0_I1, Complex::new(0_f64, -0.881373587020_f64));
        test_fn(Complex::acsc, _0_I_NEG1, Complex::new(0_f64, 0.881373587020_f64));
    }

    #[test]
    fn test_complex_cot() {
        test_fn(Complex::cot, _1_I0, Complex::new(0.642092615934_f64, 0_f64));
        test_fn(Complex::cot, _NEG1_I0, Complex::new(-0.642092615934_f64, 0_f64));
        test_fn(Complex::cot, _0_I1, Complex::new(0_f64, -1.31303528550_f64));
        test_fn(Complex::cot, _0_I_NEG1, Complex::new(0_f64, 1.31303528550_f64));
    }

    #[test]
    fn test_complex_acot() {
        test_fn(Complex::acot, _1_I0, Complex::new(0.785398163397, 0_f64));
        test_fn(Complex::acot, _NEG1_I0, Complex::new(-0.785398163397, 0_f64));
        test_fn(Complex::acot, _0_I1, Complex::new(0_f64, f64::NEG_INFINITY));
        test_fn(Complex::acot, _0_I_NEG1, Complex::new(0_f64, f64::INFINITY));
    }
}


#[cfg(test)]
mod hyperbolic_trigonometry_tests {
    use cglinalg_core::{
        Complex,
    };
    use approx::{
        assert_relative_eq,
    };
    use core::f64;


    #[test]
    fn test_cosh_unit_re() {
        let z: Complex<f64> = Complex::unit_re();
        let expected = Complex::new(1.54308063482_f64, 0_f64);
        let result = z.cosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_cosh_minus_unit_re() {
        let z: Complex<f64> = -Complex::unit_re();
        let expected = Complex::new(1.54308063482_f64, 0_f64);
        let result = z.cosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_cosh_unit_im() {
        let z: Complex<f64> = Complex::unit_im();
        let expected = Complex::new(0.540302305868_f64, 0_f64);
        let result = z.cosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_cosh_minus_unit_im() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(0.540302305868_f64, 0_f64);
        let result = z.cosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_cosh_unit_re_times_pi() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(0.540302305868_f64, 0_f64);
        let result = z.cosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_acosh_unit_re() {
        let z: Complex<f64> = Complex::unit_re();
        let expected = Complex::new(0_f64, 0_f64);
        let result = z.acosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_acosh_minus_unit_re() {
        let z: Complex<f64> = -Complex::unit_re();
        let expected = Complex::new(0_f64, 3.14159265359_f64);
        let result = z.acosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_acosh_unit_im() {
        let z: Complex<f64> = Complex::unit_im();
        let expected = Complex::new(0.881373587020_f64, 1.57079632679_f64);
        let result = z.acosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_acosh_minus_unit_im() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(0.881373587020_f64, -1.57079632679_f64);
        let result = z.acosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sinh_unit_re() {
        let z: Complex<f64> = Complex::unit_re();
        let expected = Complex::new(1.175201193643_f64, 0_f64);
        let result = z.sinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sinh_minus_unit_re() {
        let z: Complex<f64> = -Complex::unit_re();
        let expected = Complex::new(-1.175201193643_f64, 0_f64);
        let result = z.sinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sinh_unit_im() {
        let z: Complex<f64> = Complex::unit_im();
        let expected = Complex::new(0_f64, 0.841470984808_f64);
        let result = z.sinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sinh_minus_unit_im() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(0_f64, -0.841470984808_f64);
        let result = z.sinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_asinh_unit_re() {
        let z: Complex<f64> = Complex::unit_re();
        let expected = Complex::new(0.881373587020_f64, 0_f64);
        let result = z.asinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_asinh_minus_unit_re() {
        let z: Complex<f64> = -Complex::unit_re();
        let expected = Complex::new(-0.881373587020_f64, 0_f64);
        let result = z.asinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_asinh_unit_im() {
        let z: Complex<f64> = Complex::unit_im();
        let expected = Complex::new(0_f64, 1.57079632679_f64);
        let result = z.asinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_asinh_minus_unit_im() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(0_f64, -1.57079632679_f64);
        let result = z.asinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_tanh_unit_re() {
        let z: Complex<f64> = Complex::unit_re();
        let expected = Complex::new(0.761594155956_f64, 0_f64);
        let result = z.tanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_tanh_minus_unit_re() {
        let z: Complex<f64> = -Complex::unit_re();
        let expected = Complex::new(-0.761594155956_f64, 0_f64);
        let result = z.tanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_tanh_unit_im() {
        let z: Complex<f64> = Complex::unit_im();
        let expected = Complex::new(0_f64, 1.55740772465_f64);
        let result = z.tanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_tanh_minus_unit_im() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(0_f64, -1.55740772465_f64);
        let result = z.tanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_atanh_unit_re() {
        let z: Complex<f64> = Complex::unit_re();
        let expected = Complex::new(f64::INFINITY, 0_f64);
        let result = z.atanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_atanh_minus_unit_re() {
        let z: Complex<f64> = -Complex::unit_re();
        let expected = Complex::new(f64::NEG_INFINITY, 0_f64);
        let result = z.atanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_atanh_unit_im() {
        let z: Complex<f64> = Complex::unit_im();
        let expected = Complex::new(0_f64, 0.785398163397_f64);
        let result = z.atanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_atanh_minus_unit_im() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(0_f64, -0.785398163397_f64);
        let result = z.atanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sech_unit_re() {
        let z: Complex<f64> = Complex::unit_re();
        let expected = Complex::new(0.648054273664_f64, 0_f64);
        let result = z.sech();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sech_minus_unit_re() {
        let z: Complex<f64> = -Complex::unit_re();
        let expected = Complex::new(0.648054273664_f64, 0_f64);
        let result = z.sech();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sech_unit_im() {
        let z: Complex<f64> = Complex::unit_im();
        let expected = Complex::new(1.85081571768_f64, 0_f64);
        let result = z.sech();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sech_minus_unit_im() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(1.85081571768_f64, 0_f64);
        let result = z.sech();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_asech_unit_re() {
        let z: Complex<f64> = Complex::unit_re();
        let expected = Complex::new(0_f64, 0_f64);
        let result = z.asech();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_asech_minus_unit_re() {
        let z: Complex<f64> = -Complex::unit_re();
        let expected = Complex::new(0_f64, 3.14159265359_f64);
        let result = z.asech();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_asech_unit_im() {
        let z: Complex<f64> = Complex::unit_im();
        let expected = Complex::new(0.881373587020_f64, -1.57079632679_f64);
        let result = z.asech();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_asech_minus_unit_im() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(0.881373587020_f64, 1.57079632679_f64);
        let result = z.asech();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_csch_unit_re() {
        let z: Complex<f64> = Complex::unit_re();
        let expected = Complex::new(0.850918128239_f64, 0_f64);
        let result = z.csch();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_csch_minus_unit_re() {
        let z: Complex<f64> = -Complex::unit_re();
        let expected = Complex::new(-0.850918128239_f64, 0_f64);
        let result = z.csch();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_csch_unit_im() {
        let z: Complex<f64> = Complex::unit_im();
        let expected = Complex::new(0_f64, -1.18839510578_f64);
        let result = z.csch();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_csch_minus_unit_im() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(0_f64, 1.18839510578_f64);
        let result = z.csch();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_acsch_unit_re() {
        let z: Complex<f64> = Complex::unit_re();
        let expected = Complex::new(0.881373587020_f64, 0_f64);
        let result = z.acsch();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_acsch_minus_unit_re() {
        let z: Complex<f64> = -Complex::unit_re();
        let expected = Complex::new(-0.881373587020_f64, 0_f64);
        let result = z.acsch();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_acsch_unit_im() {
        let z: Complex<f64> = Complex::unit_im();
        let expected = Complex::new(0_f64, -1.57079632679_f64);
        let result = z.acsch();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_acsch_minus_unit_im() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(0_f64, 1.57079632679_f64);
        let result = z.acsch();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_coth_unit_re() {
        let z: Complex<f64> = Complex::unit_re();
        let expected = Complex::new(1.31303528550_f64, 0_f64);
        let result = z.coth();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_coth_minus_unit_re() {
        let z: Complex<f64> = -Complex::unit_re();
        let expected = Complex::new(-1.31303528550_f64, 0_f64);
        let result = z.coth();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_coth_unit_im() {
        let z: Complex<f64> = Complex::unit_im();
        let expected = Complex::new(0_f64, -0.642092615934_f64);
        let result = z.coth();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_coth_minus_unit_im() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(0_f64, 0.642092615934_f64);
        let result = z.coth();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_acoth_unit_re() {
        let z: Complex<f64> = Complex::unit_re();
        let expected = Complex::new(f64::INFINITY, 0_f64);
        let result = z.acoth();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_acoth_minus_unit_re() {
        let z: Complex<f64> = -Complex::unit_re();
        let expected = Complex::new(f64::NEG_INFINITY, 0_f64);
        let result = z.acoth();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_acoth_unit_im() {
        let z: Complex<f64> = Complex::unit_im();
        let expected = Complex::new(0_f64, -0.785398163397_f64);
        let result = z.acoth();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_acoth_minus_unit_im() {
        let z: Complex<f64> = -Complex::unit_im();
        let expected = Complex::new(0_f64, 0.785398163397_f64);
        let result = z.acoth();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_cosh_unit_im_times_pi_over_8() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_8;
        let expected_re = f64::sqrt(2_f64 + f64::sqrt(2_f64)) / 2_f64;
        let expected = Complex::new(expected_re, 0_f64);
        let result = z.cosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_cosh_unit_im_times_pi_over_6() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_6;
        let expected_re = f64::sqrt(3_f64) / 2_f64;
        let expected = Complex::new(expected_re, 0_f64);
        let result = z.cosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_cosh_unit_im_times_pi_over_4() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_4;
        let expected_re = 1_f64 / f64::sqrt(2_f64);
        let expected = Complex::new(expected_re, 0_f64);
        let result = z.cosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_cosh_unit_im_times_pi_over_3() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_3;
        let expected_re = 1_f64 / 2_f64;
        let expected = Complex::new(expected_re, 0_f64);
        let result = z.cosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_cosh_unit_im_times_pi_over_2() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_2;
        let expected_re = 0_f64;
        let expected = Complex::new(expected_re, 0_f64);
        let result = z.cosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_cosh_unit_im_times_pi() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::PI;
        let expected_re = -1_f64;
        let expected = Complex::new(expected_re, 0_f64);
        let result = z.cosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_cosh_unit_im_times_two_pi() {
        let z: Complex<f64> = Complex::unit_im() * 2_f64 * f64::consts::PI;
        let expected_re = 1_f64;
        let expected = Complex::new(expected_re, 0_f64);
        let result = z.cosh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sinh_unit_im_times_pi_over_8() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_8;
        let expected_im = f64::sqrt(2_f64 - f64::sqrt(2_f64)) / 2_f64;
        let expected = Complex::new(0_f64, expected_im);
        let result = z.sinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sinh_unit_im_times_pi_over_6() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_6;
        let expected_im = 1_f64 / 2_f64;
        let expected = Complex::new(0_f64, expected_im);
        let result = z.sinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sinh_unit_im_times_pi_over_4() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_4;
        let expected_im = 1_f64 / f64::sqrt(2_f64);
        let expected = Complex::new(0_f64, expected_im);
        let result = z.sinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sinh_unit_im_times_pi_over_3() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_3;
        let expected_im = f64::sqrt(3_f64) / 2_f64;
        let expected = Complex::new(0_f64, expected_im);
        let result = z.sinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sinh_unit_im_times_pi_over_2() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_2;
        let expected_im = 1_f64;
        let expected = Complex::new(0_f64, expected_im);
        let result = z.sinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sinh_unit_im_times_pi() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::PI;
        let expected_im = 0_f64;
        let expected = Complex::new(0_f64, expected_im);
        let result = z.sinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_sinh_unit_im_times_two_pi() {
        let z: Complex<f64> = Complex::unit_im() * 2_f64 * f64::consts::PI;
        let expected_im = 0_f64;
        let expected = Complex::new(0_f64, expected_im);
        let result = z.sinh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_tanh_unit_im_times_pi_over_8() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_8;
        let expected_im = f64::sqrt(3_f64 - 2_f64 * f64::sqrt(2_f64));
        let expected = Complex::new(0_f64, expected_im);
        let result = z.tanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_tanh_unit_im_times_pi_over_6() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_6;
        let expected_im = 1_f64 / f64::sqrt(3_f64);
        let expected = Complex::new(0_f64, expected_im);
        let result = z.tanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_tanh_unit_im_times_pi_over_4() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_4;
        let expected_im = 1_f64;
        let expected = Complex::new(0_f64, expected_im);
        let result = z.tanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_tanh_unit_im_times_pi_over_3() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_3;
        let expected_im = f64::sqrt(3_f64);
        let expected = Complex::new(0_f64, expected_im);
        let result = z.tanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_tanh_unit_im_times_pi_over_2() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::FRAC_PI_2;
        let expected_im = f64::INFINITY;
        let expected = Complex::new(0_f64, expected_im);
        let result = z.tanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_tanh_unit_im_times_pi() {
        let z: Complex<f64> = Complex::unit_im() * f64::consts::PI;
        let expected_im = 0_f64;
        let expected = Complex::new(0_f64, expected_im);
        let result = z.tanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_tanh_unit_im_times_two_pi() {
        let z: Complex<f64> = Complex::unit_im() * 2_f64 * f64::consts::PI;
        let expected_im = 0_f64;
        let expected = Complex::new(0_f64, expected_im);
        let result = z.tanh();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
}

