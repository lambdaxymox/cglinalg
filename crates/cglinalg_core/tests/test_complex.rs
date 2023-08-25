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
        assert_relative_eq!(result * result, z, epsilon = 1e-10);
    }

    #[test]
    fn test_square_root_i() {
        let i: Complex<f64> = Complex::unit_im();
        let expected = (1_f64 / f64::sqrt(2_f64)) * Complex::new(1_f64, 1_f64);
        let result = i.sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
        assert_relative_eq!(result * result, i, epsilon = 1e-10);
    }

    #[test]
    fn test_square_root_minus_i() {
        let i: Complex<f64> = Complex::unit_im();
        let expected = (1_f64 / f64::sqrt(2_f64)) * Complex::new(1_f64, -1_f64);
        let result = (-i).sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
        assert_relative_eq!(result * result, -i, epsilon = 1e-10);
    }

    #[test]
    fn test_square_root_complex() {
        let z = Complex::new(2_f64, 5_f64);
        let expected = Complex::new(1.9216093264675973_f64, 1.3009928530039094_f64);
        let result = z.sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
        assert_relative_eq!(result * result, z, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_square_root_large_input1() {
        // value = 9.480751908109174e153_f64
        let value = f64::sqrt(f64::MAX) / f64::sqrt(2_f64);
        let z = Complex::new(value, value);
        let expected = Complex::new(1.06977941273076882e77, 4.43117141500609302e76);
        let sqrt_z = z.sqrt();
        let result = sqrt_z;

        assert_relative_eq!(result, expected, epsilon = 1e-16);
        assert_relative_eq!(sqrt_z * sqrt_z, z, epsilon = 1e-16);
    }

    #[test]
    fn test_complex_square_root_large_input2() {
        // value = 9.480751908109174e153_f64
        let value = f64::sqrt(f64::MAX) / f64::sqrt(2_f64);
        let z = Complex::new(-value, -value);
        let expected = Complex::new(4.43117141500609302e76, -1.06977941273076882e77);
        let sqrt_z = z.sqrt();
        let result = sqrt_z;

        assert_relative_eq!(result, expected, epsilon = 1e-16);
        assert_relative_eq!(sqrt_z * sqrt_z, z, epsilon = 1e-16);
    }

    #[test]
    fn test_complex_square_root_large_input3() {
        let z = Complex::new(7.059507001456394e153, 1.1468385853229968e151);
        let expected = Complex::new(8.40209001328809051e76, 6.82472208408411659e73);
        let sqrt_z = z.sqrt();
        let result = sqrt_z;

        assert_relative_eq!(result, expected, epsilon = 1e-16);
        assert_relative_eq!((sqrt_z * sqrt_z).imaginary(), z.imaginary(), epsilon = 1e-10, max_relative = 1e-15);
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


    const _0_I_0: Complex<f64> = Complex::new(0_f64, 0_f64);
    const _1_I_0: Complex<f64> = Complex::new(1_f64, 0_f64);
    const _0_I_1: Complex<f64> = Complex::new(0_f64, 1_f64);
    const _NEG1_I_0: Complex<f64> = Complex::new(-1_f64, 0_f64);
    const _0_I_NEG1: Complex<f64> = Complex::new(0_f64, -1_f64);
    const _0_I_INF: Complex<f64> = Complex::new(0_f64, f64::INFINITY);
    const _0_I_NEGINF: Complex<f64> = Complex::new(0_f64, f64::NEG_INFINITY);
    const _0_I_FRAC_PI_8: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_8);
    const _0_I_FRAC_PI_6: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_6);
    const _0_I_FRAC_PI_4: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_4);
    const _0_I_FRAC_PI_3: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_3);
    const _0_I_FRAC_PI_2: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_2);
    const _0_I_PI: Complex<f64> = Complex::new(0_f64, f64::consts::PI);
    const _0_I_2_PI: Complex<f64> = Complex::new(0_f64, 2_f64 * f64::consts::PI);
    const _FRAC_PI_8_I_0: Complex<f64> = Complex::new(f64::consts::FRAC_PI_8, 0_f64);
    const _FRAC_PI_6_I_0: Complex<f64> = Complex::new(f64::consts::FRAC_PI_6, 0_f64);
    const _FRAC_PI_4_I_0: Complex<f64> = Complex::new(f64::consts::FRAC_PI_4, 0_f64);
    const _FRAC_PI_3_I_0: Complex<f64> = Complex::new(f64::consts::FRAC_PI_3, 0_f64);
    const _FRAC_PI_2_I_0: Complex<f64> = Complex::new(f64::consts::FRAC_PI_2, 0_f64);
    const _PI_I_0: Complex<f64> = Complex::new(f64::consts::PI, 0_f64);
    const _2_PI_I_0: Complex<f64> = Complex::new(2_f64 * f64::consts::PI, 0_f64);


    #[test]
    fn test_complex_cos() {
        assert_relative_eq!(Complex::cos(_0_I_0),         Complex::new(1_f64, 0_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_1_I_0),         Complex::new(0.540302305868140_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_NEG1_I_0),      Complex::new(0.540302305868140_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_0_I_1),         Complex::new(1.54308063481524_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_0_I_NEG1),      Complex::new(1.54308063481524_f64, 0_f64),  epsilon = 1e-8);

        assert_relative_eq!(Complex::cos(_0_I_FRAC_PI_8), Complex::new(1.07810228857284_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_0_I_FRAC_PI_6), Complex::new(1.14023832107643_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_0_I_FRAC_PI_4), Complex::new(1.32460908925201_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_0_I_FRAC_PI_3), Complex::new(1.60028685770239_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_0_I_FRAC_PI_2), Complex::new(2.50917847865806_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_0_I_PI),        Complex::new(11.5919532755215_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_0_I_2_PI),      Complex::new(267.746761483748_f64, 0_f64),  epsilon = 1e-8);

        assert_relative_eq!(Complex::cos(_FRAC_PI_8_I_0), Complex::new(0.923879532511287_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_FRAC_PI_6_I_0), Complex::new(0.866025403784439_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_FRAC_PI_4_I_0), Complex::new(0.707106781186548_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_FRAC_PI_3_I_0), Complex::new(0.500000000000000_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_FRAC_PI_2_I_0), Complex::new(0_f64, 0_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_PI_I_0),        Complex::new(-1_f64, 0_f64),                epsilon = 1e-8);
        assert_relative_eq!(Complex::cos(_2_PI_I_0),      Complex::new(1_f64, 0_f64),                 epsilon = 1e-8);
    }

    #[test]
    fn test_complex_acos() {
        assert_relative_eq!(Complex::acos(_1_I_0),    Complex::new(0_f64, 0_f64),                                 epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(_NEG1_I_0), _PI_I_0,                                                    epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(_0_I_1),    Complex::new(1.57079632679490_f64, -0.881373587019543_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(_0_I_NEG1), Complex::new(1.57079632679490_f64, 0.881373587019543_f64),  epsilon = 1e-8);

        assert_relative_eq!(Complex::acos(Complex::new(1.07810228857284_f64, 0_f64)), _0_I_FRAC_PI_8, epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(Complex::new(1.14023832107643_f64, 0_f64)), _0_I_FRAC_PI_6, epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(Complex::new(1.32460908925201_f64, 0_f64)), _0_I_FRAC_PI_4, epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(Complex::new(1.60028685770239_f64, 0_f64)), _0_I_FRAC_PI_3, epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(Complex::new(2.50917847865806_f64, 0_f64)), _0_I_FRAC_PI_2, epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(Complex::new(11.5919532755215_f64, 0_f64)), _0_I_PI,        epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(Complex::new(267.746761483748_f64, 0_f64)), _0_I_2_PI,      epsilon = 1e-8);

        assert_relative_eq!(Complex::acos(Complex::new(0.923879532511287_f64, 0_f64)), _FRAC_PI_8_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(Complex::new(0.866025403784439_f64, 0_f64)), _FRAC_PI_6_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(Complex::new(0.707106781186548_f64, 0_f64)), _FRAC_PI_4_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(Complex::new(0.500000000000000_f64, 0_f64)), _FRAC_PI_3_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(Complex::new(0_f64, 0_f64)),              _FRAC_PI_2_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acos(Complex::new(-1_f64, 0_f64)),             _PI_I_0,        epsilon = 1e-8);
        // assert_relative_eq!(Complex::acos(Complex::new(1_f64, 0_f64)),              _2_PI_I_0,      epsilon = 1e-8);
    }

    #[test]
    fn test_complex_sin() {
        assert_relative_eq!(Complex::sin(_0_I_0),    Complex::new(0_f64, 0_f64),                  epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_1_I_0),    Complex::new(0.841470984807897_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_NEG1_I_0), Complex::new(-0.841470984807897_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_0_I_1),    Complex::new(0_f64, 1.17520119364380_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_0_I_NEG1), Complex::new(0_f64, -1.17520119364380_f64),  epsilon = 1e-8);

        assert_relative_eq!(Complex::sin(_0_I_FRAC_PI_8), Complex::new(0_f64, 0.402870381917066_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_0_I_FRAC_PI_6), Complex::new(0_f64, 0.547853473888040_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_0_I_FRAC_PI_4), Complex::new(0_f64, 0.868670961486010_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_0_I_FRAC_PI_3), Complex::new(0_f64, 1.24936705052398_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_0_I_FRAC_PI_2), Complex::new(0_f64, 2.30129890230729_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_0_I_PI),        Complex::new(0_f64, 11.5487393572577_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_0_I_2_PI),      Complex::new(0_f64, 267.744894041017_f64),  epsilon = 1e-8);

        assert_relative_eq!(Complex::sin(_FRAC_PI_8_I_0), Complex::new(0.382683432365090_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_FRAC_PI_6_I_0), Complex::new(0.500000000000000_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_FRAC_PI_4_I_0), Complex::new(0.707106781186548_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_FRAC_PI_3_I_0), Complex::new(0.866025403784439_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_FRAC_PI_2_I_0), Complex::new(1_f64, 0_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_PI_I_0),        Complex::new(0_f64, 0_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::sin(_2_PI_I_0),      Complex::new(0_f64, 0_f64),                 epsilon = 1e-8);
    }

    #[test]
    fn test_complex_asin() {
        assert_relative_eq!(Complex::asin(Complex::new(0_f64, 0_f64)), _0_I_0,                                      epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(_1_I_0),                     Complex::new(1.57079632679490_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(_NEG1_I_0),                  Complex::new(-1.57079632679490_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(_0_I_1),                     Complex::new(0_f64, 0.881373587019543_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(_0_I_NEG1),                  Complex::new(0_f64, -0.881373587019543_f64), epsilon = 1e-8);

        assert_relative_eq!(Complex::asin(Complex::new(0_f64, 0.402870381917066_f64)), _0_I_FRAC_PI_8, epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(Complex::new(0_f64, 0.547853473888040_f64)), _0_I_FRAC_PI_6, epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(Complex::new(0_f64, 0.868670961486010_f64)), _0_I_FRAC_PI_4, epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(Complex::new(0_f64, 1.24936705052398_f64)),  _0_I_FRAC_PI_3, epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(Complex::new(0_f64, 2.30129890230729_f64)),  _0_I_FRAC_PI_2, epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(Complex::new(0_f64, 11.5487393572577_f64)),  _0_I_PI,        epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(Complex::new(0_f64, 267.744894041017_f64)),  _0_I_2_PI,      epsilon = 1e-8);

        assert_relative_eq!(Complex::asin(Complex::new(0.382683432365090_f64, 0_f64)), _FRAC_PI_8_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(Complex::new(0.500000000000000_f64, 0_f64)), _FRAC_PI_6_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(Complex::new(0.707106781186548_f64, 0_f64)), _FRAC_PI_4_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(Complex::new(0.866025403784439_f64, 0_f64)), _FRAC_PI_3_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asin(Complex::new(1_f64, 0_f64)),                 _FRAC_PI_2_I_0, epsilon = 1e-8);
        // assert_relative_eq!(Complex::asin(Complex::new(0_f64, 0_f64)),                 _PI_I_0,        epsilon = 1e-8);
        // assert_relative_eq!(Complex::asin(Complex::new(0_f64, 0_f64)),                 _2_PI_I_0,      epsilon = 1e-8);
    }

    #[test]
    fn test_complex_tan() {
        assert_relative_eq!(Complex::tan(_0_I_0),    Complex::new(0_f64, 0_f64),                  epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_1_I_0),    Complex::new(1.55740772465490_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_NEG1_I_0), Complex::new(-1.55740772465490_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_0_I_1),    Complex::new(0_f64, 0.761594155955765_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_0_I_NEG1), Complex::new(0_f64, -0.761594155955765_f64), epsilon = 1e-8);

        assert_relative_eq!(Complex::tan(_0_I_FRAC_PI_8), Complex::new(0_f64, 0.373684747901215_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_0_I_FRAC_PI_6), Complex::new(0_f64, 0.480472778156452_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_0_I_FRAC_PI_4), Complex::new(0_f64, 0.655794202632672_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_0_I_FRAC_PI_3), Complex::new(0_f64, 0.780714435359268_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_0_I_FRAC_PI_2), Complex::new(0_f64, 0.917152335667274_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_0_I_PI),        Complex::new(0_f64, 0.996272076220750_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_0_I_2_PI),      Complex::new(0_f64, 0.999993025339611_f64), epsilon = 1e-8);

        assert_relative_eq!(Complex::tan(_FRAC_PI_8_I_0), Complex::new(0.414213562373095_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_FRAC_PI_6_I_0), Complex::new(0.577350269189626_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_FRAC_PI_4_I_0), Complex::new(1_f64, 0_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_FRAC_PI_3_I_0), Complex::new(1.73205080756888_f64, 0_f64),  epsilon = 1e-8);
        // assert_relative_eq!(Complex::tan(_FRAC_PI_2_I_0), Complex::new(f64::INFINITY, 0_f64),         epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_PI_I_0),        Complex::new(0_f64, 0_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::tan(_2_PI_I_0),      Complex::new(0_f64, 0_f64),                 epsilon = 1e-8);
    }

    #[test]
    fn test_complex_atan() {
        assert_relative_eq!(Complex::atan(_0_I_0),    Complex::new(0_f64, 0_f64),                  epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(_1_I_0),    Complex::new(0.785398163397448_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(_NEG1_I_0), Complex::new(-0.785398163397448_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(_0_I_1),    Complex::new(0_f64, f64::INFINITY),          epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(_0_I_NEG1), Complex::new(0_f64, f64::NEG_INFINITY),      epsilon = 1e-8);

        assert_relative_eq!(Complex::atan(Complex::new(0_f64, 0.373684747901215_f64)), _0_I_FRAC_PI_8, epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(Complex::new(0_f64, 0.480472778156452_f64)), _0_I_FRAC_PI_6, epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(Complex::new(0_f64, 0.655794202632672_f64)), _0_I_FRAC_PI_4, epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(Complex::new(0_f64, 0.780714435359268_f64)), _0_I_FRAC_PI_3, epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(Complex::new(0_f64, 0.917152335667274_f64)), _0_I_FRAC_PI_2, epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(Complex::new(0_f64, 0.996272076220750_f64)), _0_I_PI,        epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(Complex::new(0_f64, 0.999993025339611_f64)), _0_I_2_PI,      epsilon = 1e-7);

        assert_relative_eq!(Complex::atan(Complex::new(0.414213562373095_f64, 0_f64)), _FRAC_PI_8_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(Complex::new(0.577350269189626_f64, 0_f64)), _FRAC_PI_6_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(Complex::new(1_f64, 0_f64)),                 _FRAC_PI_4_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::atan(Complex::new(1.73205080756888_f64, 0_f64)),  _FRAC_PI_3_I_0, epsilon = 1e-8);
        // assert_relative_eq!(Complex::atan(Complex::new(f64::INFINITY, 0_f64)),  _FRAC_PI_2_I_0 epsilon = 1e-8);
        // assert_relative_eq!(Complex::atan(Complex::new(0_f64, 0_f64)),              _PI_I_0,        epsilon = 1e-8);
        // assert_relative_eq!(Complex::atan(Complex::new(0_f64, 0_f64)),              _2_PI_I_0,      epsilon = 1e-8);
    }

    #[test]
    fn test_complex_sec() {
        assert_relative_eq!(Complex::sec(_0_I_0),    Complex::new(1_f64, 0_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_1_I_0),    Complex::new(1.85081571768093_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_NEG1_I_0), Complex::new(1.85081571768093_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_0_I_1),    Complex::new(0.648054273663885_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_0_I_NEG1), Complex::new(0.648054273663885_f64, 0_f64), epsilon = 1e-8);

        assert_relative_eq!(Complex::sec(_0_I_FRAC_PI_8), Complex::new(0.927555771469298_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_0_I_FRAC_PI_6), Complex::new(0.877009640454779_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_0_I_FRAC_PI_4), Complex::new(0.754939708714131_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_0_I_FRAC_PI_3), Complex::new(0.624887966296087_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_0_I_FRAC_PI_2), Complex::new(0.398536815338387_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_0_I_PI),        Complex::new(0.0862667383340544_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_0_I_2_PI),      Complex::new(0.00373487243863713_f64, 0_f64), epsilon = 1e-8);

        assert_relative_eq!(Complex::sec(_FRAC_PI_8_I_0), Complex::new(1.08239220029239_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_FRAC_PI_6_I_0), Complex::new(1.15470053837925_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_FRAC_PI_4_I_0), Complex::new(1.41421356237310_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_FRAC_PI_3_I_0), Complex::new(2_f64, 0_f64),                epsilon = 1e-8);
        // assert_relative_eq!(Complex::sec(_FRAC_PI_2_I_0), Complex::new(f64::INFINITY, 0_f64),     epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_PI_I_0),        Complex::new(-1_f64, 0_f64),               epsilon = 1e-8);
        assert_relative_eq!(Complex::sec(_2_PI_I_0),      Complex::new(1_f64, 0_f64),                epsilon = 1e-8);
    }

    #[test]
    fn test_complex_asec() {
        // assert_relative_eq!(Complex::asec(_0_I0),     Complex::new(f64::INFINITY, f64::INFINITY),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(_1_I_0),    Complex::new(0_f64, 0_f64),                                 epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(_NEG1_I_0), Complex::new(3.14159265358979_f64, 0_f64),                  epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(_0_I_1),    Complex::new(1.57079632679490_f64, 0.881373587019543_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(_0_I_NEG1), Complex::new(1.57079632679490_f64, -0.881373587019543_f64), epsilon = 1e-8);

        assert_relative_eq!(Complex::asec(Complex::new(0.927555771469298_f64, 0_f64)),   _0_I_FRAC_PI_8,  epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(Complex::new(0.877009640454779_f64, 0_f64)),   _0_I_FRAC_PI_6,  epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(Complex::new(0.754939708714131_f64, 0_f64)),   _0_I_FRAC_PI_4,  epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(Complex::new(0.624887966296087_f64, 0_f64)),   _0_I_FRAC_PI_3,  epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(Complex::new(0.398536815338387_f64, 0_f64)),   _0_I_FRAC_PI_2,  epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(Complex::new(0.0862667383340544_f64, 0_f64)),  _0_I_PI,         epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(Complex::new(0.00373487243863713_f64, 0_f64)), _0_I_2_PI,       epsilon = 1e-8);

        assert_relative_eq!(Complex::asec(Complex::new(1.08239220029239_f64, 0_f64)), _FRAC_PI_8_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(Complex::new(1.15470053837925_f64, 0_f64)), _FRAC_PI_6_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(Complex::new(1.41421356237310_f64, 0_f64)), _FRAC_PI_4_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(Complex::new(2_f64, 0_f64)),                _FRAC_PI_3_I_0, epsilon = 1e-8);
        // assert_relative_eq!(Complex::asec(Complex::new(f64::INFINITY, 0_f64)),        _FRAC_PI_2_I_0,   epsilon = 1e-8);
        assert_relative_eq!(Complex::asec(Complex::new(-1_f64, 0_f64)),               _PI_I_0, epsilon = 1e-8);
        // assert_relative_eq!(Complex::asec(Complex::new(1_f64, 0_f64)),                _2_PI_I_0, epsilon = 1e-8);
    }

    #[test]
    fn test_complex_csc() {
        // assert_relative_eq!(Complex::csc(_0_I0),     Complex::new(), epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_1_I_0),    Complex::new(1.18839510577812_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_NEG1_I_0), Complex::new(-1.18839510577812_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_0_I_1),    Complex::new(0_f64, -0.850918128239322_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_0_I_NEG1), Complex::new(0_f64, 0.850918128239322_f64),  epsilon = 1e-8);

        assert_relative_eq!(Complex::csc(_0_I_FRAC_PI_8), Complex::new(0_f64, -2.48218793161582_f64),    epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_0_I_FRAC_PI_6), Complex::new(0_f64, -1.82530557468795_f64),    epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_0_I_FRAC_PI_4), Complex::new(0_f64, -1.15118387092085_f64),    epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_0_I_FRAC_PI_3), Complex::new(0_f64, -0.800405292888593_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_0_I_FRAC_PI_2), Complex::new(0_f64, -0.434537208094696_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_0_I_PI),        Complex::new(0_f64, -0.0865895375300469_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_0_I_2_PI),      Complex::new(0_f64, -0.00373489848828567_f64), epsilon = 1e-8);

        assert_relative_eq!(Complex::csc(_FRAC_PI_8_I_0), Complex::new(2.61312592975275_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_FRAC_PI_6_I_0), Complex::new(2_f64, 0_f64),                epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_FRAC_PI_4_I_0), Complex::new(1.41421356237310_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_FRAC_PI_3_I_0), Complex::new(1.15470053837925_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::csc(_FRAC_PI_2_I_0), Complex::new(1_f64, 0_f64),                epsilon = 1e-8);
        // assert_relative_eq!(Complex::csc(Complex::new(f64::INFINITY, 0_f64)),   _PI_I_0,  epsilon = 1e-8);
        // assert_relative_eq!(Complex::csc(Complex::new(f64::INFINITY, 0_f64)),  _2_PI_I_0,   epsilon = 1e-8);
    }

    #[test]
    fn test_complex_acsc() {
        // assert_relative_eq!(Complex::acsc(_0_I0),     Complex::new(), epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(_1_I_0),    Complex::new(1.57079632679_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(_NEG1_I_0), Complex::new(-1.57079632679_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(_0_I_1),    Complex::new(0_f64, -0.881373587020_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(_0_I_NEG1), Complex::new(0_f64, 0.881373587020_f64),  epsilon = 1e-8);

        assert_relative_eq!(Complex::acsc(Complex::new(0_f64, -2.48218793162_f64)),    _0_I_FRAC_PI_8,  epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(Complex::new(0_f64, -1.82530557469_f64)),    _0_I_FRAC_PI_6,  epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(Complex::new(0_f64, -1.15118387092_f64)),    _0_I_FRAC_PI_4,  epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(Complex::new(0_f64, -0.800405292889_f64)),   _0_I_FRAC_PI_3,  epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(Complex::new(0_f64, -0.434537208095_f64)),   _0_I_FRAC_PI_2,  epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(Complex::new(0_f64, -0.0865895375300_f64)),  _0_I_PI,         epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(Complex::new(0_f64, -0.00373489848829_f64)), _0_I_2_PI,       epsilon = 1e-8);

        assert_relative_eq!(Complex::acsc(Complex::new(2.61312592975_f64, 0_f64)), _FRAC_PI_8_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(Complex::new(2_f64, 0_f64)),             _FRAC_PI_6_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(Complex::new(1.41421356237_f64, 0_f64)), _FRAC_PI_4_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(Complex::new(1.15470053838_f64, 0_f64)), _FRAC_PI_3_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsc(Complex::new(1_f64, 0_f64)),             _FRAC_PI_2_I_0, epsilon = 1e-8);
        // assert_relative_eq!(Complex::acsc(Complex::new(f64::INFINITY, 0_f64)),     _PI_I_0, epsilon = 1e-8);
        // assert_relative_eq!(Complex::acsc(Complex::new(f64::INFINITY, 0_f64)),     _2_PI_I_0, epsilon = 1e-8);
    }

    #[test]
    fn test_complex_cot() {
        // assert_relative_eq!(Complex::cot(_0_I_0),     Complex::new(f64::INFINITY, 0_f64),         epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_1_I_0),    Complex::new(0.642092615934331_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_NEG1_I_0), Complex::new(-0.642092615934331_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_0_I_1),    Complex::new(0_f64, -1.31303528549933_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_0_I_NEG1), Complex::new(0_f64, 1.31303528549933_f64),   epsilon = 1e-8);

        assert_relative_eq!(Complex::cot(_0_I_FRAC_PI_8), Complex::new(0_f64, -2.67605248974291_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_0_I_FRAC_PI_6), Complex::new(0_f64, -2.08128336393364_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_0_I_FRAC_PI_4), Complex::new(0_f64, -1.52486861882206_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_0_I_FRAC_PI_3), Complex::new(0_f64, -1.28087807104504_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_0_I_FRAC_PI_2), Complex::new(0_f64, -1.09033141072737_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_0_I_PI),        Complex::new(0_f64, -1.00374187319732_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_0_I_2_PI),      Complex::new(0_f64, -1.00000697470904_f64), epsilon = 1e-8);

        assert_relative_eq!(Complex::cot(_FRAC_PI_8_I_0), Complex::new(2.41421356237310_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_FRAC_PI_6_I_0), Complex::new(1.73205080756888_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_FRAC_PI_4_I_0), Complex::new(1_f64, 0_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_FRAC_PI_3_I_0), Complex::new(0.577350269189626_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cot(_FRAC_PI_2_I_0), Complex::new(0_f64, 0_f64),                 epsilon = 1e-8);
        // assert_relative_eq!(Complex::cot(_PI_I_0),        Complex::new(f64::INFINITY, 0_f64),         epsilon = 1e-8);
        // assert_relative_eq!(Complex::cot(_2_PI_I_0),      Complex::new(f64::INFINITY, 0_f64),         epsilon = 1e-8);
    }

    #[test]
    fn test_complex_acot() {
        assert_relative_eq!(Complex::acot(_0_I_0),    Complex::new(f64::consts::FRAC_PI_2, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(_1_I_0),    Complex::new(0.785398163397448_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(_NEG1_I_0), Complex::new(-0.785398163397448_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(_0_I_1),    Complex::new(0_f64, f64::NEG_INFINITY),      epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(_0_I_NEG1), Complex::new(0_f64, f64::INFINITY),          epsilon = 1e-8);

        assert_relative_eq!(Complex::acot(Complex::new(0_f64, -2.67605248974291_f64)), _0_I_FRAC_PI_8, epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(Complex::new(0_f64, -2.08128336393364_f64)), _0_I_FRAC_PI_6, epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(Complex::new(0_f64, -1.52486861882206_f64)), _0_I_FRAC_PI_4, epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(Complex::new(0_f64, -1.28087807104504_f64)), _0_I_FRAC_PI_3, epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(Complex::new(0_f64, -1.09033141072737_f64)), _0_I_FRAC_PI_2, epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(Complex::new(0_f64, -1.00374187319732_f64)), _0_I_PI,        epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(Complex::new(0_f64, -1.00000697470904_f64)), _0_I_2_PI,      epsilon = 1e-7);

        assert_relative_eq!(Complex::acot(Complex::new(2.41421356237310_f64, 0_f64)),  _FRAC_PI_8_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(Complex::new(1.73205080756888_f64, 0_f64)),  _FRAC_PI_6_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(Complex::new(1_f64, 0_f64)),                 _FRAC_PI_4_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(Complex::new(0.577350269189626_f64, 0_f64)), _FRAC_PI_3_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acot(Complex::new(0_f64, 0_f64)),                 _FRAC_PI_2_I_0, epsilon = 1e-8);
        // assert_relative_eq!(Complex::acot(Complex::new(f64::INFINITY, 0_f64)),      _PI_I_0,           epsilon = 1e-8);
        // assert_relative_eq!(Complex::acot(Complex::new(f64::INFINITY, 0_f64)),      _2_PI_I_0,         epsilon = 1e-8);
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


    const _0_I_0: Complex<f64> = Complex::new(0_f64, 0_f64);
    const _1_I_0: Complex<f64> = Complex::new(1_f64, 0_f64);
    const _0_I_1: Complex<f64> = Complex::new(0_f64, 1_f64);
    const _NEG1_I_0: Complex<f64> = Complex::new(-1_f64, 0_f64);
    const _0_I_NEG1: Complex<f64> = Complex::new(0_f64, -1_f64);

    const _0_I_FRAC_PI_8: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_8);
    const _0_I_FRAC_PI_6: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_6);
    const _0_I_FRAC_PI_4: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_4);
    const _0_I_FRAC_PI_3: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_3);
    const _0_I_FRAC_PI_2: Complex<f64> = Complex::new(0_f64, f64::consts::FRAC_PI_2);
    const _0_I_PI: Complex<f64> = Complex::new(0_f64, f64::consts::PI);
    const _0_I_2_PI: Complex<f64> = Complex::new(0_f64, 2_f64 * f64::consts::PI);

    const _FRAC_PI_8_I_0: Complex<f64> = Complex::new(f64::consts::FRAC_PI_8, 0_f64);
    const _FRAC_PI_6_I_0: Complex<f64> = Complex::new(f64::consts::FRAC_PI_6, 0_f64);
    const _FRAC_PI_4_I_0: Complex<f64> = Complex::new(f64::consts::FRAC_PI_4, 0_f64);
    const _FRAC_PI_3_I_0: Complex<f64> = Complex::new(f64::consts::FRAC_PI_3, 0_f64);
    const _FRAC_PI_2_I_0: Complex<f64> = Complex::new(f64::consts::FRAC_PI_2, 0_f64);
    const _PI_I_0: Complex<f64> = Complex::new(f64::consts::PI, 0_f64);
    const _2_PI_I_0: Complex<f64> = Complex::new(2_f64 * f64::consts::PI, 0_f64);

    const _INF_I_INF: Complex<f64> = Complex::new(f64::INFINITY, f64::INFINITY);


    #[test]
    fn test_complex_cosh() {
        assert_relative_eq!(Complex::cosh(_0_I_0),    _1_I_0,                                     epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_1_I_0),    Complex::new(1.54308063481524_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_0_I_1),    Complex::new(0.540302305868140_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_NEG1_I_0), Complex::new(1.54308063481524_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_0_I_NEG1), Complex::new(0.540302305868140_f64, 0_f64), epsilon = 1e-8);

        assert_relative_eq!(Complex::cosh(_0_I_FRAC_PI_8), Complex::new(0.923879532511287_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_0_I_FRAC_PI_6), Complex::new(0.866025403784439_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_0_I_FRAC_PI_4), Complex::new(0.707106781186548_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_0_I_FRAC_PI_3), Complex::new(0.500000000000000_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_0_I_FRAC_PI_2), Complex::new(0_f64, 0_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_0_I_PI),        Complex::new(-1_f64, 0_f64),                epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_0_I_2_PI),      Complex::new(1_f64, 0_f64),                 epsilon = 1e-8);
    
        assert_relative_eq!(Complex::cosh(_FRAC_PI_8_I_0), Complex::new(1.07810228857284_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_FRAC_PI_6_I_0), Complex::new(1.14023832107643_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_FRAC_PI_4_I_0), Complex::new(1.32460908925201_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_FRAC_PI_3_I_0), Complex::new(1.60028685770239_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_FRAC_PI_2_I_0), Complex::new(2.50917847865806_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_PI_I_0),        Complex::new(11.5919532755215_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::cosh(_2_PI_I_0),      Complex::new(267.746761483748_f64, 0_f64), epsilon = 1e-8);
    }

    #[test]
    fn test_complex_cosh_special_values() {
        assert_eq!(Complex::cosh(Complex::new(0_f64, 0_f64)), Complex::new(1_f64, 0_f64));
        
        let z_nan0 = Complex::cosh(Complex::new(0_f64, f64::INFINITY));
        assert!(z_nan0.real().is_nan());
        assert_eq!(z_nan0.imaginary(), 0_f64);

        let z_nan1 = Complex::cosh(Complex::new(0_f64, f64::NAN));
        assert!(z_nan1.real().is_nan());
        assert_eq!(z_nan1.imaginary(), 0_f64);

        let z_nan2 = Complex::cosh(Complex::new(-f64::MIN_POSITIVE, f64::INFINITY));
        assert!(z_nan2.real().is_nan());
        assert!(z_nan2.imaginary().is_nan());

        let z_nan3 = Complex::cosh(Complex::new(f64::MIN_POSITIVE, f64::INFINITY));
        assert!(z_nan3.real().is_nan());
        assert!(z_nan3.imaginary().is_nan());

        let z_nan4 = Complex::cosh(Complex::new(-f64::MIN_POSITIVE, f64::NAN));
        assert!(z_nan4.real().is_nan());
        assert!(z_nan4.imaginary().is_nan());

        let z_nan5 = Complex::cosh(Complex::new(f64::MIN_POSITIVE, f64::NAN));
        assert!(z_nan5.real().is_nan());
        assert!(z_nan5.imaginary().is_nan());

        assert_eq!(Complex::cosh(Complex::new(f64::INFINITY, 0_f64)), Complex::new(f64::INFINITY, 0_f64));
        assert_eq!(Complex::cosh(Complex::new(f64::INFINITY, -f64::MIN_POSITIVE)), Complex::new(f64::INFINITY, -f64::INFINITY));
        assert_eq!(Complex::cosh(Complex::new(f64::INFINITY, f64::MIN_POSITIVE)), Complex::new(f64::INFINITY, f64::INFINITY));

        let z_nan6 = Complex::cosh(Complex::new(f64::INFINITY, f64::INFINITY));
        assert!(z_nan6.real().is_infinite());
        assert!(z_nan6.imaginary().is_nan());

        let z_nan7 = Complex::cosh(Complex::new(f64::INFINITY, f64::NAN));
        assert!(z_nan7.real().is_infinite());
        assert!(z_nan7.imaginary().is_nan());

        let z_nan7 = Complex::cosh(Complex::new(f64::NAN, 0_f64));
        assert!(z_nan7.real().is_nan());
        assert_eq!(z_nan7.imaginary(), 0_f64);

        let z_nan8 = Complex::cosh(Complex::new(f64::NAN, -f64::MIN_POSITIVE));
        assert!(z_nan8.real().is_nan());
        assert!(z_nan8.imaginary().is_nan());

        let z_nan9 = Complex::cosh(Complex::new(f64::NAN, f64::MIN_POSITIVE));
        assert!(z_nan9.real().is_nan());
        assert!(z_nan9.imaginary().is_nan());

        let z_nan10 = Complex::cosh(Complex::new(f64::NAN, f64::NAN));
        assert!(z_nan10.is_nan());
        assert!(z_nan10.is_nan());
    }

    #[test]
    fn test_complex_acosh() {
        assert_relative_eq!(Complex::acosh(_0_I_0),    Complex::new(0_f64, 1.57079632679490_f64),                  epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(_1_I_0),    Complex::new(0_f64, 0_f64),                                 epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(_0_I_1),    Complex::new(0.881373587019543_f64, 1.57079632679490_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(_NEG1_I_0), Complex::new(0_f64, 3.14159265358979_f64),                  epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(_0_I_NEG1), Complex::new(0.881373587019543_f64, -1.57079632679490_f64), epsilon = 1e-8);

        assert_relative_eq!(Complex::acosh(Complex::new(0.923879532511287_f64, 0_f64)), _0_I_FRAC_PI_8, epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(Complex::new(0.866025403784439_f64, 0_f64)), _0_I_FRAC_PI_6, epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(Complex::new(0.707106781186548_f64, 0_f64)), _0_I_FRAC_PI_4, epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(Complex::new(0.500000000000000_f64, 0_f64)), _0_I_FRAC_PI_3, epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(Complex::new(0_f64, 0_f64)),                 _0_I_FRAC_PI_2, epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(Complex::new(-1_f64, 0_f64)),                _0_I_PI,        epsilon = 1e-8);
        // assert_relative_eq!(Complex::acosh(Complex::new(1_f64, 0_f64)),                 _0_I_2_PI,      epsilon = 1e-8);
    
        assert_relative_eq!(Complex::acosh(Complex::new(1.07810228857284_f64, 0_f64)),  _FRAC_PI_8_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(Complex::new(1.14023832107643_f64, 0_f64)),  _FRAC_PI_6_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(Complex::new(1.32460908925201_f64, 0_f64)),  _FRAC_PI_4_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(Complex::new(1.60028685770239_f64, 0_f64)),  _FRAC_PI_3_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(Complex::new(2.50917847865806_f64, 0_f64)),  _FRAC_PI_2_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acosh(Complex::new(11.5919532755215_f64, 0_f64)),  _PI_I_0,        epsilon = 1e-8);
        // assert_relative_eq!(Complex::acosh(Complex::new(267.746761483748_f64, 0_f64)),  _2_PI_I_0,      epsilon = 1e-8);
    }

    #[test]
    fn test_complex_acosh_special_values() {
        assert_eq!(Complex::acosh(Complex::new(0_f64, 0_f64)), Complex::new(0_f64, f64::consts::FRAC_PI_2));
        
        assert_eq!(Complex::acosh(Complex::new(0_f64, f64::INFINITY)), Complex::new(f64::INFINITY, f64::consts::FRAC_PI_2));
        assert_eq!(Complex::acosh(Complex::new(-f64::MIN_POSITIVE, f64::INFINITY)), Complex::new(f64::INFINITY, f64::consts::FRAC_PI_2));
        assert_eq!(Complex::acosh(Complex::new(f64::MIN_POSITIVE, f64::INFINITY)), Complex::new(f64::INFINITY, f64::consts::FRAC_PI_2));

        let z_nan0 = Complex::acosh(Complex::new(0_f64, f64::NAN));
        assert!(z_nan0.real().is_nan());
        assert!(z_nan0.imaginary().is_nan());

        let z_nan1 = Complex::acosh(Complex::new(-f64::MIN_POSITIVE, f64::NAN));
        assert!(z_nan1.real().is_nan());
        assert!(z_nan1.imaginary().is_nan());
        
        let z_nan2 = Complex::acosh(Complex::new(f64::MIN_POSITIVE, f64::NAN));
        assert!(z_nan2.real().is_nan());
        assert!(z_nan2.imaginary().is_nan());

        assert_eq!(Complex::acosh(Complex::new(f64::NEG_INFINITY, f64::MIN_POSITIVE)), Complex::new(f64::INFINITY, f64::consts::PI));
        assert_eq!(Complex::acosh(Complex::new(f64::INFINITY, f64::MIN_POSITIVE)), Complex::new(f64::INFINITY, 0_f64));
        assert_eq!(Complex::acosh(Complex::new(f64::NEG_INFINITY, f64::INFINITY)), Complex::new(f64::INFINITY, 3_f64 * f64::consts::FRAC_PI_4));

        let z_nan3 = Complex::acosh(Complex::new(f64::INFINITY, f64::NAN));
        assert!(z_nan3.real().is_infinite());
        assert!(z_nan3.imaginary().is_nan());

        let z_nan4 = Complex::acosh(Complex::new(f64::NEG_INFINITY, f64::NAN));
        assert!(z_nan4.real().is_infinite());
        assert!(z_nan4.imaginary().is_nan());

        let z_nan5 = Complex::acosh(Complex::new(f64::NAN, 0_f64));
        assert!(z_nan5.real().is_nan());
        assert!(z_nan5.imaginary().is_nan());

        let z_nan6 = Complex::acosh(Complex::new(f64::NAN, -f64::MIN_POSITIVE));
        assert!(z_nan6.real().is_nan());
        assert!(z_nan6.imaginary().is_nan());

        let z_nan7 = Complex::acosh(Complex::new(f64::NAN, f64::MIN_POSITIVE));
        assert!(z_nan7.real().is_nan());
        assert!(z_nan7.imaginary().is_nan());

        let z_nan8 = Complex::acosh(Complex::new(f64::NAN, f64::INFINITY));
        assert!(z_nan8.real().is_infinite());
        assert!(z_nan8.imaginary().is_nan());

        let z_nan9 = Complex::acosh(Complex::new(f64::NAN, f64::NAN));
        assert!(z_nan9.real().is_nan());
        assert!(z_nan9.imaginary().is_nan());
    }

    #[test]
    fn test_complex_sinh() {
        assert_relative_eq!(Complex::sinh(_0_I_0),    _0_I_0,                                      epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_1_I_0),    Complex::new(1.17520119364380_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_0_I_1),    Complex::new(0_f64, 0.841470984807897_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_NEG1_I_0), Complex::new(-1.17520119364380_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_0_I_NEG1), Complex::new(0_f64, -0.841470984807897_f64), epsilon = 1e-8);

        assert_relative_eq!(Complex::sinh(_0_I_FRAC_PI_8), Complex::new(0_f64, 0.382683432365090_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_0_I_FRAC_PI_6), Complex::new(0_f64, 0.500000000000000_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_0_I_FRAC_PI_4), Complex::new(0_f64, 0.707106781186548_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_0_I_FRAC_PI_3), Complex::new(0_f64, 0.866025403784439_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_0_I_FRAC_PI_2), Complex::new(0_f64, 1_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_0_I_PI),        Complex::new(0_f64, 0_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_0_I_2_PI),      Complex::new(0_f64, 0_f64),                 epsilon = 1e-8);
    
        assert_relative_eq!(Complex::sinh(_FRAC_PI_8_I_0), Complex::new(0.402870381917066_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_FRAC_PI_6_I_0), Complex::new(0.547853473888040_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_FRAC_PI_4_I_0), Complex::new(0.868670961486010_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_FRAC_PI_3_I_0), Complex::new(1.24936705052398_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_FRAC_PI_2_I_0), Complex::new(2.30129890230729_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_PI_I_0),        Complex::new(11.5487393572577_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sinh(_2_PI_I_0),      Complex::new(267.744894041017_f64, 0_f64),  epsilon = 1e-8);
    }

    #[test]
    fn test_complex_sinh_special_values() {
        assert_eq!(Complex::sinh(Complex::new(0_f64, 0_f64)), Complex::new(0_f64, 0_f64));

        let z_nan0 = Complex::sinh(Complex::new(0_f64, f64::INFINITY));
        assert_eq!(z_nan0.real(), 0_f64);
        assert!(z_nan0.imaginary().is_nan());

        let z_nan1 = Complex::sinh(Complex::new(0_f64, f64::NAN));
        assert_eq!(z_nan1.real(), 0_f64);
        assert!(z_nan1.imaginary().is_nan());

        let z_nan2 = Complex::sinh(Complex::new(f64::NAN, f64::NAN));
        assert!(z_nan2.real().is_nan());
        assert!(z_nan2.imaginary().is_nan());

        let z_nan3 = Complex::sinh(Complex::new(f64::MIN_POSITIVE, f64::NAN));
        assert!(z_nan3.real().is_nan());
        assert!(z_nan3.imaginary().is_nan());
        
        assert_eq!(Complex::sinh(Complex::new(f64::INFINITY, 0_f64)), Complex::new(f64::INFINITY, 0_f64));
        
        assert_eq!(Complex::sinh(Complex::new(f64::INFINITY, 0_f64)), Complex::new(f64::INFINITY, 0_f64));
        assert_eq!(Complex::sinh(Complex::new(f64::INFINITY, -f64::MIN_POSITIVE)), Complex::new(f64::INFINITY, -f64::INFINITY));
        assert_eq!(Complex::sinh(Complex::new(f64::INFINITY, f64::MIN_POSITIVE)), Complex::new(f64::INFINITY, f64::INFINITY));
        
        let z_nan4 = Complex::sinh(Complex::new(f64::INFINITY, f64::INFINITY));
        assert!(z_nan4.real().is_infinite());
        assert!(z_nan4.imaginary().is_nan());

        let z_nan5 = Complex::sinh(Complex::new(f64::INFINITY, f64::NAN));
        assert!(z_nan5.real().is_infinite());
        assert!(z_nan5.imaginary().is_nan());

        let z_nan6 = Complex::sinh(Complex::new(f64::NAN, 0_f64));
        assert!(z_nan6.real().is_nan());
        assert_eq!(z_nan6.imaginary(), 0_f64);

        let z_nan7 = Complex::sinh(Complex::new(f64::NAN, -f64::MIN_POSITIVE));
        assert!(z_nan7.real().is_nan());
        assert!(z_nan7.imaginary().is_nan());

        let z_nan8 = Complex::sinh(Complex::new(f64::NAN, f64::MIN_POSITIVE));
        assert!(z_nan8.real().is_nan());
        assert!(z_nan8.imaginary().is_nan());

        let z_nan9 = Complex::sinh(Complex::new(f64::NAN, f64::NAN));
        assert!(z_nan9.real().is_nan());
        assert!(z_nan9.imaginary().is_nan());
    }

    #[test]
    fn test_complex_asinh() {
        assert_relative_eq!(Complex::asinh(_0_I_0),    _0_I_0,                           epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(1.17520119364380_f64, 0_f64)),   _1_I_0,  epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(0_f64, 0.841470984807897_f64)),  _0_I_1, epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(-1.17520119364380_f64, 0_f64)),  _NEG1_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(0_f64, -0.841470984807897_f64)), _0_I_NEG1, epsilon = 1e-8);

        assert_relative_eq!(Complex::asinh(Complex::new(0_f64, 0.382683432365090_f64)), _0_I_FRAC_PI_8, epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(0_f64, 0.500000000000000_f64)), _0_I_FRAC_PI_6, epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(0_f64, 0.707106781186548_f64)), _0_I_FRAC_PI_4, epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(0_f64, 0.866025403784439_f64)), _0_I_FRAC_PI_3, epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(0_f64, 1_f64)),                 _0_I_FRAC_PI_2, epsilon = 1e-8);
        // assert_relative_eq!(Complex::asinh(Complex::new(0_f64, 0_f64)),                 _0_I_PI, epsilon = 1e-8);
        // assert_relative_eq!(Complex::asinh(Complex::new(0_f64, 0_f64)),                 _0_I_2_PI, epsilon = 1e-8);
    
        assert_relative_eq!(Complex::asinh(Complex::new(0.402870381917066_f64, 0_f64)), _FRAC_PI_8_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(0.547853473888040_f64, 0_f64)), _FRAC_PI_6_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(0.868670961486010_f64, 0_f64)), _FRAC_PI_4_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(1.24936705052398_f64, 0_f64)),  _FRAC_PI_3_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(2.30129890230729_f64, 0_f64)),  _FRAC_PI_2_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(11.5487393572577_f64, 0_f64)),  _PI_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asinh(Complex::new(267.744894041017_f64, 0_f64)),  _2_PI_I_0, epsilon = 1e-8);
    }

    #[test]
    fn test_complex_asinh_special_values() {
        assert_eq!(Complex::asinh(Complex::new(0_f64, 0_f64)), Complex::new(0_f64, 0_f64));

        assert_eq!(Complex::asinh(Complex::new(f64::MIN_POSITIVE, f64::INFINITY)), Complex::new(f64::INFINITY, f64::consts::FRAC_PI_2));

        let z_nan0 = Complex::asinh(Complex::new(-f64::MIN_POSITIVE, f64::NAN));
        assert!(z_nan0.real().is_nan());
        assert!(z_nan0.imaginary().is_nan());

        assert_eq!(Complex::asinh(Complex::new(f64::INFINITY, f64::MIN_POSITIVE)), Complex::new(f64::INFINITY, 0_f64));

        assert_eq!(Complex::asinh(Complex::new(f64::INFINITY, f64::INFINITY)), Complex::new(f64::INFINITY, f64::consts::FRAC_PI_4));

        let z_nan1 = Complex::asinh(Complex::new(f64::INFINITY, f64::NAN));
        assert!(z_nan1.real().is_infinite());
        assert!(z_nan1.imaginary().is_nan());

        let z_nan2 = Complex::asinh(Complex::new(f64::NAN, 0_f64));
        assert!(z_nan2.real().is_nan());
        assert_eq!(z_nan2.imaginary(), 0_f64);

        let z_nan3 = Complex::asinh(Complex::new(f64::NAN, -f64::MIN_POSITIVE));
        assert!(z_nan3.real().is_nan());
        assert!(z_nan3.imaginary().is_nan());

        let z_nan4 = Complex::asinh(Complex::new(f64::NAN, f64::MIN_POSITIVE));
        assert!(z_nan4.real().is_nan());
        assert!(z_nan4.imaginary().is_nan());

        let z_nan5 = Complex::asinh(Complex::new(f64::NAN, f64::INFINITY));
        assert!(z_nan5.real().is_infinite());
        assert!(z_nan5.imaginary().is_nan());

        let z_nan6 = Complex::asinh(Complex::new(f64::NAN, f64::NAN));
        assert!(z_nan6.real().is_nan());
        assert!(z_nan6.imaginary().is_nan());
    }

    #[test]
    fn test_complex_tanh() {
        assert_relative_eq!(Complex::tanh(_0_I_0),    _0_I_0,                                      epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_1_I_0),    Complex::new(0.761594155955765_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_0_I_1),    Complex::new(0_f64, 1.55740772465490_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_NEG1_I_0), Complex::new(-0.761594155955765_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_0_I_NEG1), Complex::new(0_f64, -1.55740772465490_f64),  epsilon = 1e-8);

        assert_relative_eq!(Complex::tanh(_0_I_FRAC_PI_8), Complex::new(0_f64, 0.414213562373095_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_0_I_FRAC_PI_6), Complex::new(0_f64, 0.577350269189626_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_0_I_FRAC_PI_4), Complex::new(0_f64, 1_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_0_I_FRAC_PI_3), Complex::new(0_f64, 1.73205080756888_f64),  epsilon = 1e-8);
        // assert_relative_eq!(Complex::tanh(_0_I_FRAC_PI_2), Complex::new(0_64, f64::INFINITY),          epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_0_I_PI),        Complex::new(0_f64, 0_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_0_I_2_PI),      Complex::new(0_f64, 0_f64),                 epsilon = 1e-8);
    
        assert_relative_eq!(Complex::tanh(_FRAC_PI_8_I_0), Complex::new(0.373684747901215_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_FRAC_PI_6_I_0), Complex::new(0.480472778156452_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_FRAC_PI_4_I_0), Complex::new(0.655794202632672_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_FRAC_PI_3_I_0), Complex::new(0.780714435359268_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_FRAC_PI_2_I_0), Complex::new(0.917152335667274_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_PI_I_0),        Complex::new(0.996272076220750_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::tanh(_2_PI_I_0),      Complex::new(0.999993025339611_f64, 0_f64), epsilon = 1e-8);
    }

    #[test]
    fn test_complex_tanh_imaginary_pi_over_two_periodic() {
        for k in 0..100 {
            let z = Complex::new(0_f64, f64::consts::FRAC_PI_2 + (k as f64) * f64::consts::PI);
            let result = z.tanh();

            assert!(result.is_nan());
        }

        for k in 0..100 {
            let z = Complex::new(0_f64, f64::consts::FRAC_PI_2 - (k as f64) * f64::consts::PI);
            let result = z.tanh();

            assert!(result.is_nan());
        }
    }

    #[test]
    fn test_complex_tanh_special_values() {
        assert_eq!(Complex::tanh(_0_I_0),    _0_I_0);
        assert_eq!(Complex::tanh(Complex::new(f64::INFINITY, -f64::MIN_POSITIVE)), _1_I_0);
        assert_eq!(Complex::tanh(Complex::new(f64::INFINITY, f64::MIN_POSITIVE)), _1_I_0);
        assert_eq!(Complex::tanh(Complex::new(f64::INFINITY, f64::MIN_POSITIVE)), Complex::new(1_f64, 0_f64));
        assert_eq!(Complex::tanh(Complex::new(f64::INFINITY, 0_f64)), Complex::new(1_f64, 0_f64));
        assert_eq!(Complex::tanh(Complex::new(f64::INFINITY, f64::INFINITY)), _1_I_0);
        assert_eq!(Complex::tanh(Complex::new(f64::INFINITY, f64::NAN)), Complex::new(1_f64, 0_f64));
        
        let z_nan0 = Complex::tanh(Complex::new(f64::NAN, 0_f64));
        assert!(z_nan0.real().is_nan());
        assert_eq!(z_nan0.imaginary(), 0_f64);

        let z_nan1 = Complex::tanh(Complex::new(-f64::MIN_POSITIVE, f64::INFINITY));
        assert!(z_nan1.real().is_nan());
        assert!(z_nan1.imaginary().is_nan());

        let z_nan2 = Complex::tanh(Complex::new(0_f64, f64::INFINITY));
        assert!(z_nan2.real().is_nan());
        assert!(z_nan2.imaginary().is_nan());

        let z_nan3 = Complex::tanh(Complex::new(f64::MIN_POSITIVE, f64::INFINITY));
        assert!(z_nan3.real().is_nan());
        assert!(z_nan3.imaginary().is_nan());

        let z_nan4 = Complex::tanh(Complex::new(-f64::MIN_POSITIVE, f64::NAN));
        assert!(z_nan4.real().is_nan());
        assert!(z_nan4.imaginary().is_nan());

        let z_nan5 = Complex::tanh(Complex::new(0_f64, f64::NAN));
        assert!(z_nan5.real().is_nan());
        assert!(z_nan5.imaginary().is_nan());

        let z_nan6 = Complex::tanh(Complex::new(f64::MIN_POSITIVE, f64::NAN));
        assert!(z_nan6.real().is_nan());
        assert!(z_nan6.imaginary().is_nan());

        let z_nan7 = Complex::tanh(Complex::new(f64::NAN, f64::MIN_POSITIVE));
        assert!(z_nan7.real().is_nan());
        assert!(z_nan7.imaginary().is_nan());

        let z_nan8 = Complex::tanh(Complex::new(f64::NAN, f64::MIN_POSITIVE));
        assert!(z_nan8.real().is_nan());
        assert!(z_nan8.imaginary().is_nan());

        let z_nan9 = Complex::tanh(Complex::new(f64::NAN, -f64::MIN_POSITIVE));
        assert!(z_nan9.real().is_nan());
        assert!(z_nan9.imaginary().is_nan());

        let z_nan10 = Complex::tanh(Complex::new(f64::NAN, f64::NAN));
        assert!(z_nan10.real().is_nan());
        assert!(z_nan10.imaginary().is_nan());
    }

    #[test]
    fn test_complex_atanh() {
        assert_relative_eq!(Complex::atanh(_0_I_0),    _0_I_0,                                      epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(_1_I_0),    Complex::new(f64::INFINITY, 0_f64),          epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(_0_I_1),    Complex::new(0_f64, 0.785398163397448_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(_NEG1_I_0), Complex::new(f64::NEG_INFINITY, 0_f64),      epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(_0_I_NEG1), Complex::new(0_f64, -0.785398163397448_f64), epsilon = 1e-8);

        assert_relative_eq!(Complex::atanh(Complex::new(0_f64, 0.414213562373095_f64)), _0_I_FRAC_PI_8, epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(Complex::new(0_f64, 0.577350269189626_f64)), _0_I_FRAC_PI_6, epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(Complex::new(0_f64, 1_f64)),                 _0_I_FRAC_PI_4, epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(Complex::new(0_f64, 1.73205080756888_f64)),  _0_I_FRAC_PI_3, epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(Complex::new(f64::INFINITY, 0_f64)),         _0_I_FRAC_PI_2, epsilon = 1e-8);
        // assert_relative_eq!(Complex::atanh(Complex::new(0_f64, 0_f64)),                 _0_I_PI,        epsilon = 1e-8);
        // assert_relative_eq!(Complex::atanh(Complex::new(0_f64, 0_f64)),                 _0_I_2_PI,      epsilon = 1e-8);
    
        assert_relative_eq!(Complex::atanh(Complex::new(0.373684747901215_f64, 0_f64)), _FRAC_PI_8_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(Complex::new(0.480472778156452_f64, 0_f64)), _FRAC_PI_6_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(Complex::new(0.655794202632672_f64, 0_f64)), _FRAC_PI_4_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(Complex::new(0.780714435359268_f64, 0_f64)), _FRAC_PI_3_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(Complex::new(0.917152335667274_f64, 0_f64)), _FRAC_PI_2_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(Complex::new(0.996272076220750_f64, 0_f64)), _PI_I_0,        epsilon = 1e-8);
        assert_relative_eq!(Complex::atanh(Complex::new(0.999993025339611_f64, 0_f64)), _2_PI_I_0,      epsilon = 1e-8);
    }

    #[test]
    fn test_complex_atanh_special_values() {
        assert_eq!(Complex::atanh(_0_I_0), _0_I_0);
        
        let z_nan0 = Complex::atanh(Complex::new(0_f64, f64::NAN));
        assert_eq!(z_nan0.real(), 0_f64);
        assert!(z_nan0.imaginary().is_nan());

        assert_eq!(Complex::atanh(Complex::new(1_f64, 0_f64)), Complex::new(f64::INFINITY, 0_f64));
        assert_eq!(Complex::atanh(Complex::new(f64::MIN_POSITIVE, f64::INFINITY)), Complex::new(0_f64, f64::consts::FRAC_PI_2));

        let z_nan1 = Complex::atanh(Complex::new(-f64::MIN_POSITIVE, f64::NAN));
        assert!(z_nan1.real().is_nan());
        assert!(z_nan1.imaginary().is_nan());

        assert_eq!(Complex::atanh(Complex::new(f64::INFINITY, f64::MIN_POSITIVE)), Complex::new(0_f64, f64::consts::FRAC_PI_2));
        assert_eq!(Complex::atanh(Complex::new(f64::INFINITY, f64::INFINITY)), Complex::new(0_f64, f64::consts::FRAC_PI_2));

        let z_nan2 = Complex::atanh(Complex::new(f64::INFINITY, f64::NAN));
        assert_eq!(z_nan2.real(), 0_f64);
        assert!(z_nan2.imaginary().is_nan());

        let z_nan3 = Complex::atanh(Complex::new(f64::NAN, 0_f64));
        assert!(z_nan3.real().is_nan());
        assert!(z_nan3.imaginary().is_nan());

        let z_nan4 = Complex::atanh(Complex::new(f64::NAN, -f64::MIN_POSITIVE));
        assert!(z_nan4.real().is_nan());
        assert!(z_nan4.imaginary().is_nan());

        let z_nan5 = Complex::atanh(Complex::new(f64::NAN, f64::MIN_POSITIVE));
        assert!(z_nan5.real().is_nan());
        assert!(z_nan5.imaginary().is_nan());

        assert_eq!(Complex::atanh(Complex::new(f64::NAN, f64::INFINITY)), Complex::new(0_f64, f64::consts::FRAC_PI_2));

        let z_nan6 = Complex::atanh(Complex::new(f64::NAN, f64::NAN));
        assert!(z_nan6.real().is_nan());
        assert!(z_nan6.imaginary().is_nan());
    }

    #[test]
    fn test_complex_sech() {
        assert_relative_eq!(Complex::sech(_0_I_0),    _1_I_0,                                     epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_1_I_0),    Complex::new(0.648054273663885_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_0_I_1),    Complex::new(1.85081571768093_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_NEG1_I_0), Complex::new(0.648054273663885_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_0_I_NEG1), Complex::new(1.85081571768093_f64, 0_f64),  epsilon = 1e-8);

        assert_relative_eq!(Complex::sech(_0_I_FRAC_PI_8), Complex::new(1.08239220029239_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_0_I_FRAC_PI_6), Complex::new(1.15470053837925_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_0_I_FRAC_PI_4), Complex::new(1.41421356237310_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_0_I_FRAC_PI_3), Complex::new(2.00000000000000_f64, 0_f64),   epsilon = 1e-8);
        // assert_relative_eq!(Complex::sech(_0_I_FRAC_PI_2), Complex::new(f64::INFINITY, f64::INFINITY),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_0_I_PI),        Complex::new(-1_f64, 0_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_0_I_2_PI),      Complex::new(1_f64, 0_f64),                  epsilon = 1e-8);
    
        assert_relative_eq!(Complex::sech(_FRAC_PI_8_I_0), Complex::new(0.927555771469298_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_FRAC_PI_6_I_0), Complex::new(0.877009640454779_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_FRAC_PI_4_I_0), Complex::new(0.754939708714131_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_FRAC_PI_3_I_0), Complex::new(0.624887966296087_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_FRAC_PI_2_I_0), Complex::new(0.398536815338387_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_PI_I_0),        Complex::new(0.0862667383340544_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::sech(_2_PI_I_0),      Complex::new(0.00373487243863713_f64, 0_f64), epsilon = 1e-8);
    }

    #[test]
    fn test_complex_asech() {
        // assert_relative_eq!(Complex::asech(_0_I_0),    Complex::new(f64::INFINITY, 0_f64),                         epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(_1_I_0),    Complex::new(0_f64, 0_f64),                                 epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(_0_I_1),    Complex::new(0.881373587019543_f64, -1.57079632679490_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(_NEG1_I_0), Complex::new(0_f64, 3.14159265358979_f64),                  epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(_0_I_NEG1), Complex::new(0.881373587019543_f64, 1.57079632679490_f64),  epsilon = 1e-8);

        assert_relative_eq!(Complex::asech(Complex::new(1.08239220029239_f64, 0_f64)), _0_I_FRAC_PI_8,  epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(Complex::new(1.15470053837925_f64, 0_f64)), _0_I_FRAC_PI_6,  epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(Complex::new(1.41421356237310_f64, 0_f64)), _0_I_FRAC_PI_4,  epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(Complex::new(2.00000000000000_f64, 0_f64)), _0_I_FRAC_PI_3,  epsilon = 1e-8);
        // assert_relative_eq!(Complex::asech(Complex::new(f64::INFINITY, f64::INFINITY)), _0_I_FRAC_PI_2, epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(Complex::new(-1_f64, 0_f64)),               _0_I_PI,         epsilon = 1e-8);
        // assert_relative_eq!(Complex::asech(Complex::new(1_f64, 0_f64)),                _0_I_2_PI,       epsilon = 1e-8);
    
        assert_relative_eq!(Complex::asech(Complex::new(0.927555771469298_f64, 0_f64)),   _FRAC_PI_8_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(Complex::new(0.877009640454779_f64, 0_f64)),   _FRAC_PI_6_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(Complex::new(0.754939708714131_f64, 0_f64)),   _FRAC_PI_4_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(Complex::new(0.624887966296087_f64, 0_f64)),   _FRAC_PI_3_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(Complex::new(0.398536815338387_f64, 0_f64)),   _FRAC_PI_2_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(Complex::new(0.0862667383340544_f64, 0_f64)),  _PI_I_0,        epsilon = 1e-8);
        assert_relative_eq!(Complex::asech(Complex::new(0.00373487243863713_f64, 0_f64)), _2_PI_I_0,      epsilon = 1e-8);
    }

    #[test]
    fn test_complex_csch() {
        // assert_relative_eq!(Complex::csch(_0_I_0),    Complex::new(f64::INFINITY, f64::INFINITY), epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_1_I_0),    Complex::new(0.850918128239322_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_0_I_1),    Complex::new(0_f64, -1.18839510577812_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_NEG1_I_0), Complex::new(-0.850918128239322_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_0_I_NEG1), Complex::new(0_f64, 1.18839510577812_f64),   epsilon = 1e-8);

        assert_relative_eq!(Complex::csch(_0_I_FRAC_PI_8), Complex::new(0_f64, -2.61312592975275_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_0_I_FRAC_PI_6), Complex::new(0_f64, -2.00000000000000_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_0_I_FRAC_PI_4), Complex::new(0_f64, -1.41421356237310_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_0_I_FRAC_PI_3), Complex::new(0_f64, -1.15470053837925_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_0_I_FRAC_PI_2), Complex::new(0_f64, -1_f64),                 epsilon = 1e-8);
        // assert_relative_eq!(Complex::csch(_0_I_PI),        Complex::new(f64::INFINITY, f64::INFINITY),  epsilon = 1e-8);
        // assert_relative_eq!(Complex::csch(_0_I_2_PI),      Complex::new(f64::INFINITY, f64::INFINITY),  epsilon = 1e-8);
    
        assert_relative_eq!(Complex::csch(_FRAC_PI_8_I_0), Complex::new(2.48218793161582_f64, 0_f64),    epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_FRAC_PI_6_I_0), Complex::new(1.82530557468795_f64, 0_f64),    epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_FRAC_PI_4_I_0), Complex::new(1.15118387092085_f64, 0_f64),    epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_FRAC_PI_3_I_0), Complex::new(0.800405292888593_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_FRAC_PI_2_I_0), Complex::new(0.434537208094696_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_PI_I_0),        Complex::new(0.0865895375300469_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::csch(_2_PI_I_0),      Complex::new(0.00373489848828567_f64, 0_f64), epsilon = 1e-8);
    }

    #[test]
    fn test_complex_acsch() {
        // assert_relative_eq!(Complex::acsch(_0_I_0),    Complex::new(f64::INFINITY, f64::INFINITY), epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(_1_I_0),    Complex::new(0.881373587019543_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(_0_I_1),    Complex::new(0_f64, -1.57079632679490_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(_NEG1_I_0), Complex::new(-0.881373587019543_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(_0_I_NEG1), Complex::new(0_f64, 1.57079632679490_f64),   epsilon = 1e-8);

        assert_relative_eq!(Complex::acsch(Complex::new(0_f64, -2.61312592975275_f64)), _0_I_FRAC_PI_8, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(Complex::new(0_f64, -2.00000000000000_f64)), _0_I_FRAC_PI_6, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(Complex::new(0_f64, -1.41421356237310_f64)), _0_I_FRAC_PI_4, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(Complex::new(0_f64, -1.15470053837925_f64)), _0_I_FRAC_PI_3, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(Complex::new(0_f64, -1_f64)),                _0_I_FRAC_PI_2, epsilon = 1e-8);
        // assert_relative_eq!(Complex::acsch(Complex::new(f64::INFINITY, f64::INFINITY)), _0_I_PI, epsilon = 1e-8);
        // assert_relative_eq!(Complex::acsch(Complex::new(f64::INFINITY, f64::INFINITY)), _0_I_2_PI, epsilon = 1e-8);
    
        assert_relative_eq!(Complex::acsch(Complex::new(2.48218793161582_f64, 0_f64)),    _FRAC_PI_8_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(Complex::new(1.82530557468795_f64, 0_f64)),    _FRAC_PI_6_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(Complex::new(1.15118387092085_f64, 0_f64)),    _FRAC_PI_4_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(Complex::new(0.800405292888593_f64, 0_f64)),   _FRAC_PI_3_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(Complex::new(0.434537208094696_f64, 0_f64)),   _FRAC_PI_2_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(Complex::new(0.0865895375300469_f64, 0_f64)),  _PI_I_0,        epsilon = 1e-8);
        assert_relative_eq!(Complex::acsch(Complex::new(0.00373489848828567_f64, 0_f64)), _2_PI_I_0,      epsilon = 1e-8);
    }

    #[test]
    fn test_complex_coth() {
        // assert_relative_eq!(Complex::coth(_0_I_0),    Complex::new(f64::INFINITY, f64::INFINITY), epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_1_I_0),    Complex::new(1.31303528549933_f64, 0_f64),   epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_0_I_1),    Complex::new(0_f64, -0.642092615934331_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_NEG1_I_0), Complex::new(-1.31303528549933_f64, 0_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_0_I_NEG1), Complex::new(0_f64, 0.642092615934331_f64),  epsilon = 1e-8);

        assert_relative_eq!(Complex::coth(_0_I_FRAC_PI_8), Complex::new(0_f64, -2.41421356237310_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_0_I_FRAC_PI_6), Complex::new(0_f64, -1.73205080756888_f64),  epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_0_I_FRAC_PI_4), Complex::new(0_f64, -1_f64),                 epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_0_I_FRAC_PI_3), Complex::new(0_f64, -0.577350269189626_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_0_I_FRAC_PI_2), Complex::new(0_f64, 0_f64), epsilon = 1e-8);
        // assert_relative_eq!(Complex::coth(_0_I_PI),        Complex::new(f64::INFINITY, f64::INFINITY), epsilon = 1e-8);
        // assert_relative_eq!(Complex::coth(_0_I_2_PI),      Complex::new(f64::INFINITY, f64::INFINITY), epsilon = 1e-8);
    
        assert_relative_eq!(Complex::coth(_FRAC_PI_8_I_0), Complex::new(2.67605248974291_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_FRAC_PI_6_I_0), Complex::new(2.08128336393364_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_FRAC_PI_4_I_0), Complex::new(1.52486861882206_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_FRAC_PI_3_I_0), Complex::new(1.28087807104504_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_FRAC_PI_2_I_0), Complex::new(1.09033141072737_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_PI_I_0),        Complex::new(1.00374187319732_f64, 0_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::coth(_2_PI_I_0),      Complex::new(1.00000697470904_f64, 0_f64), epsilon = 1e-8);
    }

    #[test]
    fn test_complex_acoth() {
        // assert_relative_eq!(Complex::acoth(_0_I_0),    Complex::new(0_f64, 1.57079632679490_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(_1_I_0),    Complex::new(f64::INFINITY, 0_f64),          epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(_0_I_1),    Complex::new(0_f64, -0.785398163397448_f64), epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(_NEG1_I_0), Complex::new(f64::NEG_INFINITY, 0_f64),      epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(_0_I_NEG1), Complex::new(0_f64, 0.785398163397448_f64),  epsilon = 1e-8);

        assert_relative_eq!(Complex::acoth(Complex::new(0_f64, -2.41421356237310_f64)),  _0_I_FRAC_PI_8, epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(Complex::new(0_f64, -1.73205080756888_f64)),  _0_I_FRAC_PI_6, epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(Complex::new(0_f64, -1_f64)),                 _0_I_FRAC_PI_4, epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(Complex::new(0_f64, -0.577350269189626_f64)), _0_I_FRAC_PI_3, epsilon = 1e-8);
        // assert_relative_eq!(Complex::acoth(Complex::new(0_f64, 0_f64)),                  _0_I_FRAC_PI_2, epsilon = 1e-8);
        // assert_relative_eq!(Complex::acoth(Complex::new(f64::INFINITY, f64::INFINITY)), _0_I_PI,         epsilon = 1e-8);
        // assert_relative_eq!(Complex::acoth(Complex::new(f64::INFINITY, f64::INFINITY)), _0_I_2_PI,       epsilon = 1e-8);
    
        assert_relative_eq!(Complex::acoth(Complex::new(2.67605248974291_f64, 0_f64)), _FRAC_PI_8_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(Complex::new(2.08128336393364_f64, 0_f64)), _FRAC_PI_6_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(Complex::new(1.52486861882206_f64, 0_f64)), _FRAC_PI_4_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(Complex::new(1.28087807104504_f64, 0_f64)), _FRAC_PI_3_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(Complex::new(1.09033141072737_f64, 0_f64)), _FRAC_PI_2_I_0, epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(Complex::new(1.00374187319732_f64, 0_f64)), _PI_I_0,        epsilon = 1e-8);
        assert_relative_eq!(Complex::acoth(Complex::new(1.00000697470904_f64, 0_f64)), _2_PI_I_0,      epsilon = 1e-8);
    }
}

