extern crate cglinalg_core;


#[cfg(test)]
mod storage_tests {
    use cglinalg_core::Quaternion;

    #[test]
    fn test_as_ref() {
        let q = Quaternion::new(1_i32, 2_i32, 3_i32, 4_i32);
        let q_ref: &[i32; 4] = q.as_ref();

        assert_eq!(q_ref, &[1_i32, 2_i32, 3_i32, 4_i32]);
    }

    #[test]
    fn test_indices_match_components() {
        let q = Quaternion::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(q[0], q.s);
        assert_eq!(q[1], q.v.x);
        assert_eq!(q[2], q.v.y);
        assert_eq!(q[3], q.v.z);
    }

    #[test]
    #[should_panic]
    fn test_quaternion_components_out_of_bounds1() {
        let q = Quaternion::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(q[4], q[4]);
    }

    #[test]
    #[should_panic]
    fn test_quaternion_components_out_of_bounds2() {
        let q = Quaternion::new(1_i32, 2_i32, 3_i32, 4_i32);

        assert_eq!(q[usize::MAX], q[usize::MAX]);
    }
}

#[cfg(test)]
mod lerp_tests {
    use cglinalg_core::{
        Quaternion,
        Normed
    };


    #[test]
    fn test_nlerp() {
        let q1 = Quaternion::new(0_f64, 0_f64, 0_f64, 0_f64);
        let q2 = Quaternion::new(1_f64, 1_f64, 1_f64, 1_f64);
        let amount = 0.5_f64;
        let result = q1.nlerp(&q2, amount);
        let expected = Quaternion::new(0.5_f64, 0.5_f64, 0.5_f64, 0.5_f64);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_nlerp_should_interpolate_to_endoints_normalized() {
        let q1 = Quaternion::new(1_f64, 1_f64, 1_f64, 1_f64);
        let q2 = Quaternion::new(2_f64, 2_f64, 2_f64, 2_f64);
        
        let result = q1.nlerp(&q2, 0_f64);
        let expected = q1.normalize();

        assert_eq!(result, expected);

        let result = q1.nlerp(&q2, 1_f64);
        let expected = q2.normalize();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_lerp_should_interpolate_to_endoints() {
        let q1 = Quaternion::new(1_f64, 1_f64, 1_f64, 1_f64);
        let q2 = Quaternion::new(2_f64, 2_f64, 2_f64, 2_f64);
        
        let result = q1.lerp(&q2, 0_f64);

        assert_eq!(result, q1);

        let result = q1.lerp(&q2, 1_f64);
        
        assert_eq!(result, q2);
    }
}


#[cfg(test)]
mod arithmetic_tests {
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };


    #[test]
    fn test_unit_axis_quaternions() {
        let i = Quaternion::unit_x();
        let j = Quaternion::unit_y();
        let k = Quaternion::unit_z();

        let result_i = 4_f64 * i;
        let expected_i = Quaternion::from_parts(0_f64, Vector3::new(4_f64, 0_f64, 0_f64));
        let result_j = 4_f64 * j;
        let expected_j = Quaternion::from_parts(0_f64, Vector3::new(0_f64, 4_f64, 0_f64));
        let result_k = 4_f64 * k;
        let expected_k = Quaternion::from_parts(0_f64, Vector3::new(0_f64, 0_f64, 4_f64));

        assert_eq!(result_i, expected_i);
        assert_eq!(result_j, expected_j);
        assert_eq!(result_k, expected_k);
    }

    #[test]
    fn test_quaternion_addition() {
        let q1 = Quaternion::new(1_i32, 2_i32, 3_i32, 4_i32);
        let q2 = Quaternion::new(5_i32, 6_i32, 7_i32, 8_i32);
        let expected = Quaternion::new(6_i32, 8_i32, 10_i32, 12_i32);
        let result = q1 + q2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_subtraction() {
        let q1 = Quaternion::new(1_i32, 2_i32, 3_i32, 4_i32);
        let q2 = Quaternion::new(5_i32, 6_i32, 7_i32, 8_i32);
        let expected = Quaternion::new(-4_i32, -4_i32, -4_i32, -4_i32);
        let result = q1 - q2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_unit_squares() {
        let i = Quaternion::<f64>::unit_x();
        let j = Quaternion::<f64>::unit_y();
        let k = Quaternion::<f64>::unit_z();
        let minus_one = -Quaternion::identity();

        assert_eq!(i * i, minus_one);
        assert_eq!(j * j, minus_one);
        assert_eq!(k * k, minus_one);
    }

    #[test]
    fn test_quaternion_product_of_all_unit_axis_quaternions() {
        let i = Quaternion::<f64>::unit_x();
        let j = Quaternion::<f64>::unit_y();
        let k = Quaternion::<f64>::unit_z();
        let minus_one = -Quaternion::identity();

        assert_eq!(i * j * k, minus_one);
    }

    #[test]
    fn test_quaternion_unit_products() {
        let i = Quaternion::<f64>::unit_x();
        let j = Quaternion::<f64>::unit_y();
        let k = Quaternion::<f64>::unit_z();

        assert_eq!(i * j, k);
        assert_eq!(j * i, -k);
        assert_eq!(j * k, i);
        assert_eq!(k * j, -i);
        assert_eq!(k * i, j);
        assert_eq!(i * k, -j);
    }
}

#[cfg(test)]
mod modulus_tests {
    use cglinalg_core::{
        Quaternion,
        Normed,
        Vector3,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_unit_axis_quaternions_should_have_unit_norms() {
        let i = Quaternion::<f64>::unit_x();
        let j = Quaternion::<f64>::unit_y();
        let k = Quaternion::<f64>::unit_z();
    
        assert_eq!(i.norm(), 1_f64);
        assert_eq!(j.norm(), 1_f64);
        assert_eq!(k.norm(), 1_f64);
    }

    #[test]
    fn test_quaternion_modulus() {
        let q = Quaternion::from_parts(3_f64, Vector3::new(34.8_f64, 75.1939_f64, 1.0366_f64));
        let result = q.modulus_squared();
        let expected = 6875.23713677_f64;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_normalized() {
        let q = Quaternion::from_parts(3_f64, Vector3::new(34.8_f64, 75.1939_f64, 1.0366_f64));
        let result = q.normalize().modulus();
        let expected = 1_f64;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_normalized_to() {
        let q = Quaternion::from_parts(3_f64, Vector3::new(34.8_f64, 75.1939_f64, 1.0366_f64));
        let norm = 12_f64;
        let result = q.scale(norm).modulus();
        let expected = norm;

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
}

#[cfg(test)]
mod slerp_tests {
    use cglinalg_core::{
        Quaternion,
        Angle,
        Degrees,
        Vector3,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_slerp_upper_right_quadrant() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(60_f64);
        let unit_z = Vector3::unit_z();
        let q1 = Quaternion::from_parts(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q2 = Quaternion::from_parts(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );
        let angle_expected = Degrees(45_f64);
        let expected = Quaternion::from_parts(
            Angle::cos(angle_expected / 2_f64), 
            Angle::sin(angle_expected / 2_f64) * unit_z
        );
        let result = q1.slerp(&q2, 0.5_f64);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_slerp_upper_right_quadrant1() {
        let angle1 = Degrees(20_f64);
        let angle2 = Degrees(70_f64);
        let unit_z = Vector3::unit_z();
        let q1 = Quaternion::from_parts(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q2 = Quaternion::from_parts(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );
        let angle_expected = Degrees(30_f64);
        let expected = Quaternion::from_parts(
            Angle::cos(angle_expected / 2_f64), 
            Angle::sin(angle_expected / 2_f64) * unit_z
        );
        let result = q1.slerp(&q2, 0.2_f64);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_slerp_upper_half_plane() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(150_f64);
        let unit_z = Vector3::unit_z();
        let q1 = Quaternion::from_parts(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q2 = Quaternion::from_parts(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );
        let angle_expected = Degrees(90_f64);
        let expected = Quaternion::from_parts(
            Angle::cos(angle_expected / 2_f64), 
            Angle::sin(angle_expected / 2_f64) * unit_z
        );
        let result = q1.slerp(&q2, 0.5_f64);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_slerp_negative_dot_product() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(240_f64);
        let unit_z = Vector3::unit_z();
        let q1 = Quaternion::from_parts(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q2 = Quaternion::from_parts(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );
        let angle_expected = Degrees(315_f64);
        let expected = Quaternion::from_parts(
            Angle::cos(angle_expected / 2_f64), 
            Angle::sin(angle_expected / 2_f64) * unit_z
        );
        let result = q1.slerp(&q2, 0.5_f64);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_slerp_endpoints0() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(240_f64);
        let unit_z = Vector3::unit_z();
        let q0 = Quaternion::from_parts(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q1 = Quaternion::from_parts(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );

        // The slerp function can produce either the starting quaternion
        // or its negation at 0. Both quaternions produce the same rotation.
        let expected1 = q0;
        let expected2 = -q0;
        let result = q0.slerp(&q1, 0_f64);

        assert!(result == expected1 || result == expected2);
    }

    #[test]
    fn test_slerp_endpoints1() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(240_f64);
        let unit_z = Vector3::unit_z();
        let q0 = Quaternion::from_parts(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q1 = Quaternion::from_parts(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );

        let expected = q1;
        let result = q0.slerp(&q1, 1_f64);

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod arg_tests {
    use cglinalg_core::{
        Radians,
        Quaternion,
        Unit,
        Vector3,
    };
    use approx::{
        assert_relative_eq,
    };

    use core::f64;

    
    #[test]
    fn test_quaternion_arg() {
        let q = Quaternion::new(0_f64, 1_f64, 1_f64, 1_f64);
        let expected = core::f64::consts::FRAC_PI_2;
        let result = q.arg();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_arg1() {
        let q = Quaternion::new(1_f64, 2_f64, -1_f64, 0_f64);
        let expected = f64::acos(f64::sqrt(6_f64) / 6_f64);
        let result = q.arg();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_arg2() {
        let angle = Radians(0_f64);
        let axis = Unit::from_value(Vector3::unit_z());
        let quaternion = Quaternion::from_axis_angle(&axis, angle);
        let expected = 0_f64;
        let result = quaternion.arg();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_arg3() {
        let angle = Radians(f64::consts::FRAC_PI_2);
        let axis = Unit::from_value(Vector3::unit_z());
        let quaternion = Quaternion::from_axis_angle(&axis, angle);
        let expected = f64::consts::FRAC_PI_4;
        let result = quaternion.arg();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_arg4() {
        let angle = Radians(f64::consts::PI);
        let axis = Unit::from_value(Vector3::unit_z());
        let quaternion = Quaternion::from_axis_angle(&axis, angle);
        let expected = f64::consts::FRAC_PI_2;
        let result = quaternion.arg();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_quaternion_arg5() {
        let angle = -Radians(f64::consts::FRAC_PI_2);
        let axis = Unit::from_value(Vector3::unit_z());
        let quaternion = Quaternion::from_axis_angle(&axis, angle);
        let expected = f64::consts::FRAC_PI_4;
        let result = quaternion.arg();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_arg6() {
        let angle = -Radians(f64::consts::PI);
        let axis = Unit::from_value(Vector3::unit_z());
        let quaternion = Quaternion::from_axis_angle(&axis, angle);
        let expected = f64::consts::FRAC_PI_2;
        let result = quaternion.arg();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_quaternion_arg_branches() {
        let angle = Radians(f64::consts::FRAC_PI_4);
        let axis = Unit::from_value(Vector3::unit_z());
        let q = Quaternion::from_axis_angle(&axis, angle);

        // The principal argument is half of the angle of rotation.
        let principal_arg_q = q.arg();

        assert_relative_eq!(principal_arg_q, f64::consts::FRAC_PI_8, epsilon = 1e-10);
        
        for k in 0..100 {
            let _k = k as f64;
            let arg_plus_2k_pi = principal_arg_q + 2_f64 * f64::consts::PI * _k;
            let angle_new_q = Radians(2_f64 * arg_plus_2k_pi);
            let new_q = Quaternion::from_axis_angle(&axis, angle_new_q);
            let expected = principal_arg_q;
            let result = new_q.arg();

            assert_relative_eq!(result, expected, epsilon = 1e-10);
        }
    }
}

#[cfg(test)]
mod exp_tests {
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };
    use approx::{
        assert_relative_eq,
    };

    use core::f64;


    #[test]
    fn test_quaternion_exp_zero() {
        let zero_quat: Quaternion<f64> = Quaternion::zero();
        let one_quat: Quaternion<f64> = Quaternion::identity();
        let result = zero_quat.exp();

        assert_eq!(result, one_quat);
    }

    #[test]
    fn test_quaternion_exp_power_times_pi() {
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let sgn_qv = Quaternion::from_parts(0_f64, q.v / q.v.norm());
        let pi = core::f64::consts::PI;
        let expected = -Quaternion::identity();
        let result = (sgn_qv * pi).exp();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_exp_power_unit_x_times_pi_over_2() {
        let unit_x = Quaternion::unit_x();
        let pi_over_two = f64::consts::PI / 2_f64;
        let result = (unit_x * pi_over_two).exp();
        let expected = unit_x;
     
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_exp_power_unit_y_times_pi_over_2() {
        let unit_y = Quaternion::unit_y();
        let pi_over_two = f64::consts::PI / 2_f64;
        let result = (unit_y * pi_over_two).exp();
        let expected = unit_y;
     
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_exp_power_unit_z_times_pi_over_2() {
        let unit_z = Quaternion::unit_z();
        let pi_over_two = f64::consts::PI / 2_f64;
        let result = (unit_z * pi_over_two).exp();
        let expected = unit_z;
     
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_exp_power_times_pi_unit_scalar() {
        let qs = 3_f64;
        let q = Quaternion::new(qs, 0_f64, 0_f64, 0_f64);
        let expected = Quaternion::from_parts(qs.exp(), Vector3::zero());
        let result = q.exp();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_exp_power_times_pi_unit_x() {
        let zero_vector = Vector3::zero();
        let unit_x = Quaternion::unit_x();
        let pi = f64::consts::PI;
        let result = (unit_x * pi).exp();
        let expected = Quaternion::from_parts(-1_f64, zero_vector);
     
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_exp_power_times_pi_unit_y() {
        let zero_vector = Vector3::zero();
        let unit_y = Quaternion::unit_y();
        let pi = f64::consts::PI;
        let result = (unit_y * pi).exp();
        let expected = Quaternion::from_parts(-1_f64, zero_vector);
     
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_exp_power_times_pi_unit_z() {
        let zero_vector = Vector3::zero();
        let unit_z = Quaternion::unit_z();
        let pi = f64::consts::PI;
        let result = (unit_z * pi).exp();
        let expected = Quaternion::from_parts(-1_f64, zero_vector);
     
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_exp_inverse() {
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let expected = Quaternion::identity();
        let result = (-q).exp() * q.exp();

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_quaternion_conjugate_commutes_with_exp() {
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let result = q.exp().conjugate();
        let expected = q.conjugate().exp();

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod logarithm_tests {
    use cglinalg_core::{
        Vector3,
        Quaternion,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_quaternion_logarithm_log_one_should_be_zero() {
        let q = Quaternion::<f64>::identity();
        let expected = Quaternion::<f64>::zero();
        let result = q.ln();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_logarithm() {
        let i = Quaternion::<f64>::unit_x();
        let j = Quaternion::<f64>::unit_y();
        let k = Quaternion::<f64>::unit_z();
        let exp_i = i.exp();
        let exp_j = j.exp();
        let exp_k = k.exp();

        assert_relative_eq!(exp_i.ln(), i, epsilon = 1e-8);
        assert_relative_eq!(exp_j.ln(), j, epsilon = 1e-8);
        assert_relative_eq!(exp_k.ln(), k, epsilon = 1e-8);
    }

    #[test]
    fn test_quaternion_logarithm1() {
        let i = Quaternion::<f64>::unit_x();
        let j = Quaternion::<f64>::unit_y();
        let k = Quaternion::<f64>::unit_z();
        let pi_over_2 = core::f64::consts::FRAC_PI_2;

        assert_eq!(i.ln(), i * pi_over_2);
        assert_eq!(j.ln(), j * pi_over_2);
        assert_eq!(k.ln(), k * pi_over_2);
    }

    #[test]
    fn test_quaternion_logarithm2() {
        let q = 2_f64 * Quaternion::unit_y() - 5_f64 * Quaternion::unit_z();
        let sqrt_29 = f64::sqrt(29_f64);
        let pi_over_2 = core::f64::consts::FRAC_PI_2;
        let expected_s = sqrt_29.ln();
        let expected_v = (2_f64 * Vector3::unit_y() - 5_f64 * Vector3::unit_z()) * pi_over_2 / sqrt_29;
        let expected = Quaternion::from_parts(expected_s, expected_v);
        let result = q.ln();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_log_negative_one() {
        let q = -Quaternion::<f64>::identity();
        let expected = Quaternion::<f64>::zero();
        let result = q.ln();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_negative_logarithm() {
        let i = Quaternion::<f64>::unit_x();
        let j = Quaternion::<f64>::unit_y();
        let k = Quaternion::<f64>::unit_z();
        let pi_over_2 = core::f64::consts::FRAC_PI_2;

        assert_eq!((-i).ln(), -i * pi_over_2);
        assert_eq!((-j).ln(), -j * pi_over_2);
        assert_eq!((-k).ln(), -k * pi_over_2);
    }
}

#[cfg(test)]
mod exp_ln_tests {
    use cglinalg_core::{
        Radians,
        Quaternion,
        Unit,
        Vector3,
    };
    use approx::{
        assert_relative_eq,
    };

    use core::f64;



    #[test]
    fn test_quaternion_ln_exp_pi_i() {
        let pi = f64::consts::PI;
        let quaternion = Quaternion::unit_x() * pi;
        let expected = Quaternion::zero();
        let result = quaternion.exp().ln();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_ln_exp_pi_j() {
        let pi = f64::consts::PI;
        let quaternion = Quaternion::unit_y() * pi;
        let expected = Quaternion::zero();
        let result = quaternion.exp().ln();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_ln_exp_pi_k() {
        let pi = f64::consts::PI;
        let quaternion = Quaternion::unit_z() * pi;
        let expected = Quaternion::zero();
        let result = quaternion.exp().ln();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }


    #[test]
    fn test_quaternion_ln_exp_inside_principal_branch_xy() {
        let scalar = 100_f64;
        let norm_vector = f64::consts::PI - 2_f64 * f64::EPSILON;
        let axis = Unit::from_value(Vector3::unit_z());
        for i in 0..400 {
            let angle = Radians(f64::consts::FRAC_PI_8 * (i as f64));
            let vector = (Quaternion::from_axis_angle(&axis, angle) * norm_vector).vector();
            let quaternion = Quaternion::from_parts(scalar, vector);
            let expected = quaternion;
            let result = quaternion.exp().ln();

            assert_relative_eq!(result, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_quaternion_ln_exp_inside_principal_branch_yz() {
        let scalar = 100_f64;
        let norm_vector = f64::consts::PI - 2_f64 * f64::EPSILON;
        let axis = Unit::from_value(Vector3::unit_x());
        for i in 0..400 {
            let angle = Radians(f64::consts::FRAC_PI_8 * (i as f64));
            let vector = (Quaternion::from_axis_angle(&axis, angle) * norm_vector).vector();
            let quaternion = Quaternion::from_parts(scalar, vector);
            let expected = quaternion;
            let result = quaternion.exp().ln();

            assert_relative_eq!(result, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_quaternion_ln_exp_inside_principal_branch_zx() {
        let scalar = 100_f64;
        let norm_vector = f64::consts::PI - 2_f64 * f64::EPSILON;
        let axis = Unit::from_value(Vector3::unit_y());
        for i in 0..400 {
            let angle = Radians(f64::consts::FRAC_PI_8 * (i as f64));
            let vector = (Quaternion::from_axis_angle(&axis, angle) * norm_vector).vector();
            let quaternion = Quaternion::from_parts(scalar, vector);
            let expected = quaternion;
            let result = quaternion.exp().ln();

            assert_relative_eq!(result, expected, epsilon = 1e-10);
        }
    }
}

#[cfg(test)]
mod power_tests {
    use cglinalg_core::{
        Quaternion,
    };
    use approx::{
        assert_relative_eq,
    };
    use core::f64;


    #[test]
    fn test_quaternion_with_zero_exponent_is_one() {
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let exponent = 0_f64;
        let expected = Quaternion::<f64>::identity();
        let result = q.powf(exponent);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_with_one_exponent_is_quaternion() {
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let exponent = 1_f64;
        let expected = q;
        let result = q.powf(exponent);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_quaternion_power1() {
        let i = Quaternion::<f64>::unit_x();
        let exponent = 2_f64;
        let expected = Quaternion::new(-1_f64, 0_f64, 0_f64, 0_f64);
        let result = i.powf(exponent);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_quaternion_power2() {
        let j = Quaternion::<f64>::unit_y();
        let exponent = 2_f64;
        let expected = Quaternion::new(-1_f64, 0_f64, 0_f64, 0_f64);
        let result = j.powf(exponent);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_quaternion_power3() {
        let k = Quaternion::<f64>::unit_z();
        let exponent = 2_f64;
        let expected = Quaternion::new(-1_f64, 0_f64, 0_f64, 0_f64);
        let result = k.powf(exponent);

        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
}

#[cfg(test)]
mod rotation_tests {
    use cglinalg_core::{
        Angle,
        Radians,
        Unit,
        Vector3,
        Quaternion,
    };
    use approx::{
        assert_relative_eq,
    };

    
    #[test]
    fn test_rotation_between_unit_vectors() {
        let unit_x: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x());
        let unit_y: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y());
        let unit_z: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
        let expected = Quaternion::from_axis_angle(
            &unit_z, 
            Radians::full_turn_div_4()
        );
        let result = Quaternion::rotation_between_axis(&unit_x, &unit_y).unwrap();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_between_same_unit_vectors() {
        let unit_v1: Unit<Vector3<f64>> = Unit::from_value(
            Vector3::new(1_f64, 1_f64, 0_f64)
        );
        let unit_v2: Unit<Vector3<f64>> = Unit::from_value(
            Vector3::new(1_f64, 1_f64, 0_f64)
        );
        let unit_z: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
        let expected = Quaternion::from_axis_angle(
            &unit_z, 
            Radians(0_f64)
        );
        let result = Quaternion::rotation_between_axis(&unit_v1, &unit_v2).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_rotation_between_vectors() {
        let unit_x: Vector3<f64> = Vector3::unit_x();
        let unit_y: Vector3<f64> = Vector3::unit_y();
        let unit_z: Vector3<f64> = Vector3::unit_z();
        let expected = Quaternion::from_axis_angle(
            &Unit::from_value(unit_z), 
            Radians::full_turn_div_4()
        );
        let result = Quaternion::rotation_between(&unit_x, &unit_y).unwrap();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_between_same_vectors() {
        let v1 = Vector3::new(1_f64, 1_f64, 0_f64) * 3_f64;
        let v2 = Vector3::new(1_f64, 1_f64, 0_f64) * 3_f64;
        let unit_z = Vector3::unit_z();
        let expected = Quaternion::from_axis_angle(
            &Unit::from_value(unit_z), 
            Radians(0_f64)
        );
        let result = Quaternion::rotation_between(&v1, &v2).unwrap();

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod inverse_tests {
    use cglinalg_core::{
        Quaternion,
    };


    #[test]
    fn test_inverse() {
        let quaternion = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let norm = 30_f64;
        let expected = Quaternion::new(1_f64, -2_f64, -3_f64, -4_f64) / norm;
        let result = quaternion.inverse().unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_is_invertible() {
        let quaternion = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);

        assert!(quaternion.is_invertible());
    }

    #[test]
    fn test_univertible_quaternion() {
        let quaternion: Quaternion<f64> = Quaternion::zero();
        let expected = None;
        let result = quaternion.inverse();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_univertible_quaternion_is_not_invertible() {
        let quaternion: Quaternion<f64> = Quaternion::zero();

        assert!(!quaternion.is_invertible());
    }
}

#[cfg(test)]
mod division_tests {
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };
    use approx::{
        assert_relative_eq,
    };


    #[rustfmt::skip]
    #[test]
    fn test_div_left() {
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let p = Quaternion::new(5_f64, 7_f64, 11_f64, 13_f64);
        let expected = Quaternion::new(
             104_f64 / 364_f64, 
            -2_f64 / 364_f64, 6_f64 / 364_f64, 8_f64 / 364_f64
        );
        let result = q.div_left(&p);
     
        assert!(result.is_some());
     
        let result = result.unwrap();
     
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[rustfmt::skip]
    #[test]
    fn test_div_right() {
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let p = Quaternion::new(5_f64, 7_f64, 11_f64, 13_f64);
        let expected = Quaternion::new(
            104_f64 / 364_f64, 
            8_f64 / 364_f64, 2_f64 / 364_f64, 6_f64 / 364_f64);
        let result = q.div_right(&p);
     
        assert!(result.is_some());
     
        let result = result.unwrap();
     
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[rustfmt::skip]
    #[test]
    fn test_div_left_scalar() {
        let scalar_part = 3_f64;
        let scalar = Quaternion::from_real(scalar_part);
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let expected = Quaternion::new(
            1_f64 / scalar_part,
            2_f64 / scalar_part, 3_f64 / scalar_part, 4_f64 / scalar_part
        );
        let result = q.div_left(&scalar).unwrap();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[rustfmt::skip]
    #[test]
    fn test_div_right_scalar() {
        let scalar_part = 3_f64;
        let scalar = Quaternion::from_real(scalar_part);
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let expected = Quaternion::new(
            1_f64 / scalar_part,
            2_f64 / scalar_part, 3_f64 / scalar_part, 4_f64 / scalar_part
        );
        let result = q.div_right(&scalar).unwrap();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[rustfmt::skip]
    #[test]
    fn test_div_left_vector() {
        let v = Quaternion::from_pure(Vector3::new(2_f64, 5_f64, 3_f64));
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let expected = Quaternion::new(
            31_f64 / 38_f64, 
            -13_f64 / 38_f64, -3_f64 / 38_f64, 1_f64 / 38_f64
        );
        let result = q.div_left(&v).unwrap();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[rustfmt::skip]
    #[test]
    fn test_div_right_vector() {
        let v = Quaternion::from_pure(Vector3::new(2_f64, 5_f64, 3_f64));
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let expected = Quaternion::new(
            31_f64 / 38_f64, 
            9_f64 / 38_f64, -7_f64 / 38_f64, -7_f64 / 38_f64
        );
        let result = q.div_right(&v).unwrap();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }
}

#[cfg(test)]
mod square_tests {
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };
    use approx::{
        assert_relative_eq,
    };

    
    #[test]
    fn test_square_pure_quaternion() {
        let modulus = f64::sqrt(14_f64);
        let vector = Vector3::new(2_f64, -1_f64, 3_f64) / modulus;
        let q = Quaternion::from_pure(vector);
        let expected = Quaternion::from_real(-1_f64);
        let result = q.squared();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_square_real_quaternion() {
        let q = Quaternion::from_real(-1_f64);
        let unit_scalar: Quaternion<f64> = Quaternion::unit_s();
        let unit_x: Quaternion<f64>  = Quaternion::unit_x();
        let unit_y: Quaternion<f64> = Quaternion::unit_y();
        let unit_z: Quaternion<f64> = Quaternion::unit_z();
     
        assert_eq!(unit_x * unit_x, q);
        assert_eq!(unit_y * unit_y, q);
        assert_eq!(unit_z * unit_z, q);
        assert_ne!(unit_scalar * unit_scalar, q);
    }

    /// There are infinitely many solutions to the quaternion polynomial equation
    /// ```text
    /// q^2 + 1 = 0
    /// ```
    /// whose solutions are the square roots of a quaternion `q` whose square is `-1`.
    /// In particular, the solution set is a unit two-sphere centered at the origin
    /// in the pure vector subspace of the space of quaternions. This solution set 
    /// includes the poles of the imaginary part of the complex plane `i` and `-i`.
    #[test]
    fn test_square_unit_x() {
        let i: Quaternion<f64> = Quaternion::unit_x();
        let minus_one= Quaternion::from_real(-1_f64);
        
        assert_eq!(i.squared(), minus_one);
        assert_eq!((-i).squared(), minus_one);
    }

    #[test]
    fn test_square_quaternions_one_unit_two_sphere_xy_plane() {
        let q = Quaternion::from_real(-1_f64);
        let pi_over_eight = core::f64::consts::FRAC_PI_8;
        for i in 0..64 {
            let angle = (i as f64) * pi_over_eight;
            let cos_angle = angle.cos();
            let sin_angle = angle.sin();
            let vector = Vector3::new(cos_angle, sin_angle, 0_f64);
            let sqrt_q = Quaternion::from_pure(vector);

            assert_relative_eq!(sqrt_q.squared(), q, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_square_quaternions_one_unit_two_sphere_yz_plane() {
        let q = Quaternion::from_real(-1_f64);
        let pi_over_eight = core::f64::consts::FRAC_PI_8;
        for i in 0..64 {
            let angle = (i as f64) * pi_over_eight;
            let cos_angle = angle.cos();
            let sin_angle = angle.sin();
            let vector = Vector3::new(0_f64, cos_angle, sin_angle);
            let sqrt_q = Quaternion::from_pure(vector);

            assert_relative_eq!(sqrt_q.squared(), q, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_square_quaternions_one_unit_two_sphere_zx_plane() {
        let q = Quaternion::from_real(-1_f64);
        let pi_over_eight = core::f64::consts::FRAC_PI_8;
        for i in 0..64 {
            let angle = (i as f64) * pi_over_eight;
            let cos_angle = angle.cos();
            let sin_angle = angle.sin();
            let vector = Vector3::new(cos_angle, 0_f64, sin_angle);
            let sqrt_q = Quaternion::from_pure(vector);

            assert_relative_eq!(sqrt_q.squared(), q, epsilon = 1e-10);
        }
    }
}

#[cfg(test)]
mod square_root_tests {
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };
    use approx::{
        assert_relative_eq,
    };


    #[test]
    fn test_square_root_zero() {
        let zero: Quaternion<f64> = Quaternion::zero();
        
        assert_eq!(zero.sqrt(), zero);
    }

    #[test]
    fn test_square_root_real_quaternion1() {
        let scalar_part = 2_f64;
        let scalar = Quaternion::from_real(scalar_part);
        let expected = Quaternion::from_real(f64::sqrt(scalar_part));
        let result = scalar.sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_square_root_real_quaternion2() {
        let scalar_part = -2_f64;
        let scalar = Quaternion::from_real(scalar_part);
        let sqrt_scalar = scalar.sqrt();

        assert_relative_eq!(sqrt_scalar * sqrt_scalar, scalar, epsilon = 1e-10);
    }

    #[test]
    fn test_square_root_pure_quaternion() {
        let qv = Vector3::new(2_f64, -2_f64, 1_f64);
        let q = Quaternion::from_parts(0_f64, qv);
        let expected = Quaternion::from_parts(3_f64, qv) * (1_f64 / f64::sqrt(6_f64));
        let result = q.sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_square_root_quaternion1() {
        let q = Quaternion::new(2_f64, 3_f64, 0_f64, 0_f64);
        let cos_angle_over_two_squared = (2_f64 + f64::sqrt(13_f64)) / (2_f64 * f64::sqrt(13_f64));
        let cos_angle_over_two = f64::sqrt(cos_angle_over_two_squared);
        let sin_angle_over_two_squared = 1_f64 - cos_angle_over_two_squared;
        let sin_angle_over_two = f64::sqrt(sin_angle_over_two_squared);
        let sqrt_norm_q = f64::sqrt(q.norm());
        let expected_s = sqrt_norm_q * cos_angle_over_two;
        let expected_v = sqrt_norm_q * sin_angle_over_two * Vector3::unit_x();
        let expected = Quaternion::from_parts(expected_s, expected_v);
        let result = q.sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_square_root_quaternion2() {
        let q = Quaternion::new(2_f64, 0_f64, 3_f64, 0_f64);
        let cos_angle_over_two_squared = (2_f64 + f64::sqrt(13_f64)) / (2_f64 * f64::sqrt(13_f64));
        let cos_angle_over_two = f64::sqrt(cos_angle_over_two_squared);
        let sin_angle_over_two_squared = 1_f64 - cos_angle_over_two_squared;
        let sin_angle_over_two = f64::sqrt(sin_angle_over_two_squared);
        let sqrt_norm_q = f64::sqrt(q.norm());
        let expected_s = sqrt_norm_q * cos_angle_over_two;
        let expected_v = sqrt_norm_q * sin_angle_over_two * Vector3::unit_y();
        let expected = Quaternion::from_parts(expected_s, expected_v);
        let result = q.sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_square_root_quaternion3() {
        let quaternion = Quaternion::new(2_f64, 0_f64, 0_f64, 3_f64);
        let cos_angle_over_two_squared = (2_f64 + f64::sqrt(13_f64)) / (2_f64 * f64::sqrt(13_f64));
        let cos_angle_over_two = f64::sqrt(cos_angle_over_two_squared);
        let sin_angle_over_two_squared = 1_f64 - cos_angle_over_two_squared;
        let sin_angle_over_two = f64::sqrt(sin_angle_over_two_squared);
        let sqrt_norm_q = f64::sqrt(quaternion.norm());
        let expected_s = sqrt_norm_q * cos_angle_over_two;
        let expected_v = sqrt_norm_q * sin_angle_over_two * Vector3::unit_z();
        let expected = Quaternion::from_parts(expected_s, expected_v);
        let result = quaternion.sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_square_root_quaternion4() {
        let scalar = 1_f64;
        let vector = Vector3::new(2_f64, 3_f64, 4_f64);
        let quaternion = Quaternion::from_parts(scalar, vector);
        let norm_q = quaternion.norm();
        let norm_qv = vector.norm();
        let cos_angle_over_two_squared = (1_f64 / 2_f64) * (1_f64 + (1_f64 / norm_q));
        let cos_angle_over_two = f64::sqrt(cos_angle_over_two_squared);
        let sin_angle_over_two_squared = 1_f64 - cos_angle_over_two_squared;
        let sin_angle_over_two = f64::sqrt(sin_angle_over_two_squared);
        let sqrt_norm_q = f64::sqrt(norm_q);
        let expected_s = sqrt_norm_q * cos_angle_over_two;
        let expected_v = sqrt_norm_q * sin_angle_over_two * (1_f64 / norm_qv) * vector;
        let expected = Quaternion::from_parts(expected_s, expected_v);
        let result = quaternion.sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }
}

