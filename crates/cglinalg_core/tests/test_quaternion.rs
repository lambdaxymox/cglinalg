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
        Normed,
        Quaternion,
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
    fn test_nlerp_should_interpolate_to_end_points_normalized() {
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
    fn test_lerp_should_interpolate_to_end_points() {
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

    #[rustfmt::skip]
    #[test]
    fn test_unit_axis_quaternions() {
        let i = Quaternion::unit_x();
        let j = Quaternion::unit_y();
        let k = Quaternion::unit_z();

        let result_i = 4_f64 * i;
        let expected_i = Quaternion::from_parts(
            0_f64,
            Vector3::new(4_f64, 0_f64, 0_f64),
        );
        let result_j = 4_f64 * j;
        let expected_j = Quaternion::from_parts(
            0_f64,
            Vector3::new(0_f64, 4_f64, 0_f64),
        );
        let result_k = 4_f64 * k;
        let expected_k = Quaternion::from_parts(
            0_f64,
            Vector3::new(0_f64, 0_f64, 4_f64),
        );

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
        Normed,
        Quaternion,
        Vector3,
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
}

#[cfg(test)]
mod slerp_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };
    use cglinalg_trigonometry::{
        Angle,
        Degrees,
    };

    #[rustfmt::skip]
    #[test]
    fn test_slerp_upper_right_quadrant() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(60_f64);
        let unit_z = Vector3::unit_z();
        let q1 = Quaternion::from_parts(
            Angle::cos(angle1 / 2_f64),
            Angle::sin(angle1 / 2_f64) * unit_z,
        );
        let q2 = Quaternion::from_parts(
            Angle::cos(angle2 / 2_f64),
            Angle::sin(angle2 / 2_f64) * unit_z,
        );
        let angle_expected = Degrees(45_f64);
        let expected = Quaternion::from_parts(
            Angle::cos(angle_expected / 2_f64),
            Angle::sin(angle_expected / 2_f64) * unit_z,
        );
        let result = q1.slerp(&q2, 0.5_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_slerp_upper_right_quadrant1() {
        let angle1 = Degrees(20_f64);
        let angle2 = Degrees(70_f64);
        let unit_z = Vector3::unit_z();
        let q1 = Quaternion::from_parts(
            Angle::cos(angle1 / 2_f64),
            Angle::sin(angle1 / 2_f64) * unit_z,
        );
        let q2 = Quaternion::from_parts(
            Angle::cos(angle2 / 2_f64),
            Angle::sin(angle2 / 2_f64) * unit_z,
        );
        let angle_expected = Degrees(30_f64);
        let expected = Quaternion::from_parts(
            Angle::cos(angle_expected / 2_f64),
            Angle::sin(angle_expected / 2_f64) * unit_z,
        );
        let result = q1.slerp(&q2, 0.2_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_slerp_upper_half_plane() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(150_f64);
        let unit_z = Vector3::unit_z();
        let q1 = Quaternion::from_parts(
            Angle::cos(angle1 / 2_f64),
            Angle::sin(angle1 / 2_f64) * unit_z,
        );
        let q2 = Quaternion::from_parts(
            Angle::cos(angle2 / 2_f64),
            Angle::sin(angle2 / 2_f64) * unit_z,
        );
        let angle_expected = Degrees(90_f64);
        let expected = Quaternion::from_parts(
            Angle::cos(angle_expected / 2_f64),
            Angle::sin(angle_expected / 2_f64) * unit_z,
        );
        let result = q1.slerp(&q2, 0.5_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_slerp_negative_dot_product() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(240_f64);
        let unit_z = Vector3::unit_z();
        let q1 = Quaternion::from_parts(
            Angle::cos(angle1 / 2_f64),
            Angle::sin(angle1 / 2_f64) * unit_z,
        );
        let q2 = Quaternion::from_parts(
            Angle::cos(angle2 / 2_f64),
            Angle::sin(angle2 / 2_f64) * unit_z,
        );
        let angle_expected = Degrees(315_f64);
        let expected = Quaternion::from_parts(
            Angle::cos(angle_expected / 2_f64),
            Angle::sin(angle_expected / 2_f64) * unit_z,
        );
        let result = q1.slerp(&q2, 0.5_f64);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_slerp_endpoints0() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(240_f64);
        let unit_z = Vector3::unit_z();
        let q0 = Quaternion::from_parts(
            Angle::cos(angle1 / 2_f64),
            Angle::sin(angle1 / 2_f64) * unit_z,
        );
        let q1 = Quaternion::from_parts(
            Angle::cos(angle2 / 2_f64),
            Angle::sin(angle2 / 2_f64) * unit_z,
        );

        // The slerp function can produce either the starting quaternion
        // or its negation at 0. Both quaternions produce the same rotation.
        let expected1 = q0;
        let expected2 = -q0;
        let result = q0.slerp(&q1, 0_f64);

        assert!(result == expected1 || result == expected2);
    }

    #[rustfmt::skip]
    #[test]
    fn test_slerp_endpoints1() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(240_f64);
        let unit_z = Vector3::unit_z();
        let q0 = Quaternion::from_parts(
            Angle::cos(angle1 / 2_f64),
            Angle::sin(angle1 / 2_f64) * unit_z,
        );
        let q1 = Quaternion::from_parts(
            Angle::cos(angle2 / 2_f64),
            Angle::sin(angle2 / 2_f64) * unit_z,
        );

        let expected = q1;
        let result = q0.slerp(&q1, 1_f64);

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod arg_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Unit,
        Vector3,
    };
    use cglinalg_trigonometry::Radians;

    use core::f64;

    #[test]
    fn test_quaternion_arg() {
        let q = Quaternion::new(0_f64, 1_f64, 1_f64, 1_f64);
        let expected = f64::consts::FRAC_PI_2;
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

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_arg4() {
        let angle = Radians(f64::consts::PI);
        let axis = Unit::from_value(Vector3::unit_z());
        let quaternion = Quaternion::from_axis_angle(&axis, angle);
        let expected = f64::consts::FRAC_PI_2;
        let result = quaternion.arg();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_arg5() {
        let angle = -Radians(f64::consts::FRAC_PI_2);
        let axis = Unit::from_value(Vector3::unit_z());
        let quaternion = Quaternion::from_axis_angle(&axis, angle);
        let expected = f64::consts::FRAC_PI_4;
        let result = quaternion.arg();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_arg6() {
        let angle = -Radians(f64::consts::PI);
        let axis = Unit::from_value(Vector3::unit_z());
        let quaternion = Quaternion::from_axis_angle(&axis, angle);
        let expected = f64::consts::FRAC_PI_2;
        let result = quaternion.arg();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_arg_branches() {
        let angle = Radians(f64::consts::FRAC_PI_4);
        let axis = Unit::from_value(Vector3::unit_z());
        let q = Quaternion::from_axis_angle(&axis, angle);

        // The principal argument is half of the angle of rotation.
        let principal_arg_q = q.arg();

        assert_relative_eq!(
            principal_arg_q,
            f64::consts::FRAC_PI_8,
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );

        for k in 0..100 {
            let _k = k as f64;
            let arg_plus_2k_pi = principal_arg_q + 2_f64 * f64::consts::PI * _k;
            let angle_new_q = Radians(2_f64 * arg_plus_2k_pi);
            let new_q = Quaternion::from_axis_angle(&axis, angle_new_q);
            let expected = principal_arg_q;
            let result = new_q.arg();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
        }
    }
}

#[cfg(test)]
mod exp_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
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
        let pi = f64::consts::PI;
        let expected = -Quaternion::identity();
        let result = (sgn_qv * pi).exp();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_exp_power_unit_x_times_pi_over_2() {
        let unit_x = Quaternion::unit_x();
        let pi_over_two = f64::consts::PI / 2_f64;
        let result = (unit_x * pi_over_two).exp();
        let expected = unit_x;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_exp_power_unit_y_times_pi_over_2() {
        let unit_y = Quaternion::unit_y();
        let pi_over_two = f64::consts::PI / 2_f64;
        let result = (unit_y * pi_over_two).exp();
        let expected = unit_y;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_exp_power_unit_z_times_pi_over_2() {
        let unit_z = Quaternion::unit_z();
        let pi_over_two = f64::consts::PI / 2_f64;
        let result = (unit_z * pi_over_two).exp();
        let expected = unit_z;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_exp_power_times_pi_unit_y() {
        let zero_vector = Vector3::zero();
        let unit_y = Quaternion::unit_y();
        let pi = f64::consts::PI;
        let result = (unit_y * pi).exp();
        let expected = Quaternion::from_parts(-1_f64, zero_vector);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_exp_power_times_pi_unit_z() {
        let zero_vector = Vector3::zero();
        let unit_z = Quaternion::unit_z();
        let pi = f64::consts::PI;
        let result = (unit_z * pi).exp();
        let expected = Quaternion::from_parts(-1_f64, zero_vector);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_exp_inverse() {
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let expected = Quaternion::identity();
        let result = (-q).exp() * q.exp();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
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
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };
    use core::f64;

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

        assert_relative_eq!(exp_i.ln(), i, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
        assert_relative_eq!(exp_j.ln(), j, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
        assert_relative_eq!(exp_k.ln(), k, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_logarithm1() {
        let i = Quaternion::<f64>::unit_x();
        let j = Quaternion::<f64>::unit_y();
        let k = Quaternion::<f64>::unit_z();
        let pi_over_2 = f64::consts::FRAC_PI_2;

        assert_eq!(i.ln(), i * pi_over_2);
        assert_eq!(j.ln(), j * pi_over_2);
        assert_eq!(k.ln(), k * pi_over_2);
    }

    #[test]
    fn test_quaternion_logarithm2() {
        let q = 2_f64 * Quaternion::unit_y() - 5_f64 * Quaternion::unit_z();
        let sqrt_29 = f64::sqrt(29_f64);
        let pi_over_2 = f64::consts::FRAC_PI_2;
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
        let pi_over_2 = f64::consts::FRAC_PI_2;

        assert_eq!((-i).ln(), -i * pi_over_2);
        assert_eq!((-j).ln(), -j * pi_over_2);
        assert_eq!((-k).ln(), -k * pi_over_2);
    }
}

#[cfg(test)]
mod exp_ln_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Unit,
        Vector3,
    };
    use cglinalg_trigonometry::Radians;

    use core::f64;

    #[test]
    fn test_quaternion_ln_exp_pi_i() {
        let pi = f64::consts::PI;
        let quaternion = Quaternion::unit_x() * pi;
        let expected = Quaternion::zero();
        let result = quaternion.exp().ln();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_ln_exp_pi_j() {
        let pi = f64::consts::PI;
        let quaternion = Quaternion::unit_y() * pi;
        let expected = Quaternion::zero();
        let result = quaternion.exp().ln();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_ln_exp_pi_k() {
        let pi = f64::consts::PI;
        let quaternion = Quaternion::unit_z() * pi;
        let expected = Quaternion::zero();
        let result = quaternion.exp().ln();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
        }
    }
}

#[cfg(test)]
mod power_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::Quaternion;
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

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_power1() {
        let i = Quaternion::<f64>::unit_x();
        let exponent = 2_f64;
        let expected = Quaternion::new(-1_f64, 0_f64, 0_f64, 0_f64);
        let result = i.powf(exponent);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_power2() {
        let j = Quaternion::<f64>::unit_y();
        let exponent = 2_f64;
        let expected = Quaternion::new(-1_f64, 0_f64, 0_f64, 0_f64);
        let result = j.powf(exponent);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_power3() {
        let k = Quaternion::<f64>::unit_z();
        let exponent = 2_f64;
        let expected = Quaternion::new(-1_f64, 0_f64, 0_f64, 0_f64);
        let result = k.powf(exponent);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod rotation_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Unit,
        Vector3,
    };
    use cglinalg_trigonometry::{
        Angle,
        Radians,
    };

    #[test]
    fn test_rotation_between_unit_vectors() {
        let unit_x: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x());
        let unit_y: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y());
        let unit_z: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
        let expected = Quaternion::from_axis_angle(&unit_z, Radians::full_turn_div_4());
        let result = Quaternion::rotation_between_axis(&unit_x, &unit_y).unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_rotation_between_same_unit_vectors() {
        let unit_v1: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(1_f64, 1_f64, 0_f64));
        let unit_v2: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(1_f64, 1_f64, 0_f64));
        let unit_z: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
        let expected = Quaternion::from_axis_angle(&unit_z, Radians(0_f64));
        let result = Quaternion::rotation_between_axis(&unit_v1, &unit_v2).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_rotation_between_vectors() {
        let unit_x: Vector3<f64> = Vector3::unit_x();
        let unit_y: Vector3<f64> = Vector3::unit_y();
        let unit_z: Vector3<f64> = Vector3::unit_z();
        let expected = Quaternion::from_axis_angle(&Unit::from_value(unit_z), Radians::full_turn_div_4());
        let result = Quaternion::rotation_between(&unit_x, &unit_y).unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_rotation_between_same_vectors() {
        let v1 = Vector3::new(1_f64, 1_f64, 0_f64) * 3_f64;
        let v2 = Vector3::new(1_f64, 1_f64, 0_f64) * 3_f64;
        let unit_z = Vector3::unit_z();
        let expected = Quaternion::from_axis_angle(&Unit::from_value(unit_z), Radians(0_f64));
        let result = Quaternion::rotation_between(&v1, &v2).unwrap();

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod inverse_tests {
    use cglinalg_core::Quaternion;

    #[test]
    fn test_inverse() {
        let quaternion = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let norm = 30_f64;
        let expected = Quaternion::new(1_f64, -2_f64, -3_f64, -4_f64) / norm;
        let result = quaternion.try_inverse().unwrap();

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
        let result = quaternion.try_inverse();

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
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    #[rustfmt::skip]
    #[test]
    fn test_div_left() {
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let p = Quaternion::new(5_f64, 7_f64, 11_f64, 13_f64);
        let expected = Quaternion::new(
             104_f64 / 364_f64,
            -2_f64 / 364_f64, 6_f64 / 364_f64, 8_f64 / 364_f64,
        );
        let result = q.div_left(&p);
     
        assert!(result.is_some());
     
        let result = result.unwrap();
     
        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_div_right() {
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let p = Quaternion::new(5_f64, 7_f64, 11_f64, 13_f64);
        let expected = Quaternion::new(
            104_f64 / 364_f64,
            8_f64 / 364_f64, 2_f64 / 364_f64, 6_f64 / 364_f64,
        );
        let result = q.div_right(&p);
     
        assert!(result.is_some());
     
        let result = result.unwrap();
     
        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_div_left_scalar() {
        let scalar_part = 3_f64;
        let scalar = Quaternion::from_real(scalar_part);
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let expected = Quaternion::new(
            1_f64 / scalar_part,
            2_f64 / scalar_part, 3_f64 / scalar_part, 4_f64 / scalar_part,
        );
        let result = q.div_left(&scalar).unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_div_right_scalar() {
        let scalar_part = 3_f64;
        let scalar = Quaternion::from_real(scalar_part);
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let expected = Quaternion::new(
            1_f64 / scalar_part,
            2_f64 / scalar_part, 3_f64 / scalar_part, 4_f64 / scalar_part,
        );
        let result = q.div_right(&scalar).unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_div_left_vector() {
        let v = Quaternion::from_pure(Vector3::new(2_f64, 5_f64, 3_f64));
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let expected = Quaternion::new(
             31_f64 / 38_f64,
            -13_f64 / 38_f64, -3_f64 / 38_f64, 1_f64 / 38_f64,
        );
        let result = q.div_left(&v).unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_div_right_vector() {
        let v = Quaternion::from_pure(Vector3::new(2_f64, 5_f64, 3_f64));
        let q = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let expected = Quaternion::new(
            31_f64 / 38_f64,
            9_f64 / 38_f64, -7_f64 / 38_f64, -7_f64 / 38_f64,
        );
        let result = q.div_right(&v).unwrap();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod square_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };
    use core::f64;

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
        let unit_x: Quaternion<f64> = Quaternion::unit_x();
        let unit_y: Quaternion<f64> = Quaternion::unit_y();
        let unit_z: Quaternion<f64> = Quaternion::unit_z();

        assert_eq!(unit_x * unit_x, q);
        assert_eq!(unit_y * unit_y, q);
        assert_eq!(unit_z * unit_z, q);
        assert_ne!(unit_scalar * unit_scalar, q);
    }

    #[test]
    fn test_square_unit_x() {
        let i: Quaternion<f64> = Quaternion::unit_x();
        let minus_one = Quaternion::from_real(-1_f64);

        assert_eq!(i.squared(), minus_one);
        assert_eq!((-i).squared(), minus_one);
    }

    #[test]
    fn test_square_quaternions_one_unit_two_sphere_xy_plane() {
        let q = Quaternion::from_real(-1_f64);
        let pi_over_eight = f64::consts::FRAC_PI_8;
        for i in 0..64 {
            let angle = (i as f64) * pi_over_eight;
            let cos_angle = angle.cos();
            let sin_angle = angle.sin();
            let vector = Vector3::new(cos_angle, sin_angle, 0_f64);
            let sqrt_q = Quaternion::from_pure(vector);

            assert_relative_eq!(sqrt_q.squared(), q, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_square_quaternions_one_unit_two_sphere_yz_plane() {
        let q = Quaternion::from_real(-1_f64);
        let pi_over_eight = f64::consts::FRAC_PI_8;
        for i in 0..64 {
            let angle = (i as f64) * pi_over_eight;
            let cos_angle = angle.cos();
            let sin_angle = angle.sin();
            let vector = Vector3::new(0_f64, cos_angle, sin_angle);
            let sqrt_q = Quaternion::from_pure(vector);

            assert_relative_eq!(sqrt_q.squared(), q, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_square_quaternions_one_unit_two_sphere_zx_plane() {
        let q = Quaternion::from_real(-1_f64);
        let pi_over_eight = f64::consts::FRAC_PI_8;
        for i in 0..64 {
            let angle = (i as f64) * pi_over_eight;
            let cos_angle = angle.cos();
            let sin_angle = angle.sin();
            let vector = Vector3::new(cos_angle, 0_f64, sin_angle);
            let sqrt_q = Quaternion::from_pure(vector);

            assert_relative_eq!(sqrt_q.squared(), q, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
        }
    }
}

#[cfg(test)]
mod square_root_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
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

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_square_root_real_quaternion2() {
        let scalar_part = -2_f64;
        let scalar = Quaternion::from_real(scalar_part);
        let sqrt_scalar = scalar.sqrt();

        assert_relative_eq!(
            sqrt_scalar * sqrt_scalar,
            scalar,
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
    }

    #[test]
    fn test_square_root_pure_quaternion() {
        let qv = Vector3::new(2_f64, -2_f64, 1_f64);
        let q = Quaternion::from_parts(0_f64, qv);
        let expected = Quaternion::from_parts(3_f64, qv) * (1_f64 / f64::sqrt(6_f64));
        let result = q.sqrt();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod trigonometry_cos_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    use core::f64;

    // const SQRT_2: f64 = 1.41421356237309505_f64;
    const SQRT_3: f64 = 1.73205080756887729_f64;
    const COS_1: f64 = 0.54030230586813972_f64;
    const SIN_1: f64 = 0.84147098480789651_f64;
    // const TAN_1: f64 = 1.55740772465490223_f64;
    const COSH_1: f64 = 1.54308063481524378_f64;
    const SINH_1: f64 = 1.17520119364380146_f64;
    // const TANH_1: f64 = 0.7615941559557647_f64;
    // const COS_SQRT_2: f64 = 0.15594369476537447_f64;
    // const SIN_SQRT_2: f64 = 0.98776594599273553_f64;
    // const TAN_SQRT_2: f64 = 6.33411916704219155_f64;
    const COSH_SQRT_3: f64 = 2.9145774401759282_f64;
    const SINH_SQRT_3: f64 = 2.7376562338581640_f64;
    // const COS_SQRT_3: f64 = -0.16055653857469063_f64;
    // const SIN_SQRT_3: f64 = 0.98702664499035378_f64;
    const COSH_SQRT_2: f64 = 2.17818355660857086_f64;
    // const SINH_SQRT_2: f64 = 1.93506682217435665_f64;
    // const TANH_SQRT_2: f64 = 0.88838556158566054_f64;

    #[test]
    fn test_quaternion_cos_real() {
        let base_angle = 0_f64;
        let angle_multiple = f64::consts::FRAC_PI_8;
        for i in 0..400 {
            let angle = base_angle + angle_multiple * (i as f64);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(angle.cos());
            let result = quaternion.cos();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_quaternion_cos_n_plus_one_half_times_pi_over_two() {
        for i in 0..400 {
            let angle = (i as f64 + 1_f64 / 2_f64) * f64::consts::PI;
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::zero();
            let result = quaternion.cos();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-12, relative_all <= f64::EPSILON);
        }

        for i in 0..400 {
            let angle = -(i as f64 + 1_f64 / 2_f64) * f64::consts::PI;
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::zero();
            let result = quaternion.cos();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-12, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_quaternion_cos_pi_plus_n_times_two_pi() {
        for i in 0..400 {
            let angle = f64::consts::PI + (i as f64) * (2_f64 * f64::consts::PI);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(-1_f64);
            let result = quaternion.cos();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-12, relative_all <= f64::EPSILON);
        }

        for i in 0..400 {
            let angle = f64::consts::PI - (i as f64) * (2_f64 * f64::consts::PI);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(-1_f64);
            let result = quaternion.cos();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-12, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_quaternion_cos_n_times_two_pi() {
        for i in 0..400 {
            let angle = (i as f64) * (2_f64 * f64::consts::PI);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(1_f64);
            let result = quaternion.cos();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-12, relative_all <= f64::EPSILON);
        }

        for i in 0..400 {
            let angle = -(i as f64) * (2_f64 * f64::consts::PI);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(1_f64);
            let result = quaternion.cos();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-12, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_quaternion_cos1() {
        // quaternion := 0 + 0i + 0k + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos2() {
        // quaternion := 1 + 1i + 1j + 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_1 * COSH_SQRT_3;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64) * (-SIN_1 * SINH_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos3() {
        // quaternion := 1 - 1i - 1j - 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_1 * COSH_SQRT_3;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64) * (-SIN_1 * SINH_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos4() {
        // quaternion := 1 + 0i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos5() {
        // quaternion := 0 + 1i + 0j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos6() {
        // quaternion := 0 + 0i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos7() {
        // quaternion := 0 + 0i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos8() {
        // quaternion := 1 + 1i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_1 * COSH_1;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * (-SIN_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos9() {
        // quaternion := 1 - 1i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_1 * COSH_1;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64) * (-SIN_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos10() {
        // quaternion := 1 + 0i + 1j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_1 * COSH_1;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * (-SIN_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos11() {
        // quaternion := 1 + 0i - 1j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_1 * COSH_1;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64) * (-SIN_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos12() {
        // quaternion := 1 + 0i + 0j + 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_1 * COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * (-SIN_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos13() {
        // quaternion := 1 + 0i + 0j - 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_1 * COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64) * (-SIN_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos14() {
        // quaternion := 0 + 1i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos15() {
        // quaternion := 0 + 1i - 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos16() {
        // quaternion := 0 + 0i + 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos17() {
        // quaternion := 0 + 0i + 1j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos18() {
        // quaternion := 0 + 1i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos19() {
        // quaternion := 0 + 1i + 0j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos20() {
        // quaternion := 0 - 1i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos21() {
        // quaternion := 0 - 1i - 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos22() {
        // quaternion := 0 + 0i - 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos23() {
        // quaternion := 0 + 0i - 1j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos24() {
        // quaternion := 0 - 1i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cos25() {
        // quaternion := 0 - 1i + 0j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod trigonometry_acos_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    use core::f64;

    // const SQRT_2: f64 = 1.41421356237309505_f64;
    const SQRT_3: f64 = 1.73205080756887729_f64;
    const COS_1: f64 = 0.54030230586813972_f64;
    const SIN_1: f64 = 0.84147098480789651_f64;
    // const TAN_1: f64 = 1.55740772465490223_f64;
    const COSH_1: f64 = 1.54308063481524378_f64;
    const SINH_1: f64 = 1.17520119364380146_f64;
    // const TANH_1: f64 = 0.7615941559557647_f64;
    // const COS_SQRT_2: f64 = 0.15594369476537447_f64;
    // const SIN_SQRT_2: f64 = 0.98776594599273553_f64;
    // const TAN_SQRT_2: f64 = 6.33411916704219155_f64;
    const COSH_SQRT_3: f64 = 2.9145774401759282_f64;
    const SINH_SQRT_3: f64 = 2.7376562338581640_f64;
    // const COS_SQRT_3: f64 = -0.16055653857469063_f64;
    // const SIN_SQRT_3: f64 = 0.98702664499035378_f64;
    // const COSH_SQRT_2: f64 = 2.17818355660857086_f64;
    // const SINH_SQRT_2: f64 = 1.93506682217435665_f64;
    // const TANH_SQRT_2: f64 = 0.88838556158566054_f64;

    #[test]
    fn test_quaternion_acos1() {
        let quaternion = {
            let scalar = COS_1 * COSH_SQRT_3;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64) * (-SIN_1 * SINH_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 1i + 1j + 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acos2() {
        let quaternion = {
            let scalar = COS_1 * COSH_SQRT_3;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64) * (-SIN_1 * SINH_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 - 1i - 1j - 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acos3() {
        let quaternion = {
            let scalar = COS_1 * COSH_1;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * (-SIN_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 1i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acos4() {
        let quaternion = {
            let scalar = COS_1 * COSH_1;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64) * (-SIN_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 - 1i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acos5() {
        let quaternion = {
            let scalar = COS_1 * COSH_1;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * (-SIN_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 1j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acos6() {
        let quaternion = {
            let scalar = COS_1 * COSH_1;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64) * (-SIN_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i - 1j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acos7() {
        let quaternion = {
            let scalar = COS_1 * COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * (-SIN_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j + 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acos8() {
        let quaternion = {
            let scalar = COS_1 * COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64) * (-SIN_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j - 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acos();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod trigonometry_sin_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    use core::f64;

    const SQRT_2: f64 = 1.41421356237309505_f64;
    const SQRT_3: f64 = 1.73205080756887729_f64;
    const COS_1: f64 = 0.54030230586813972_f64;
    const SIN_1: f64 = 0.84147098480789651_f64;
    // const TAN_1: f64 = 1.55740772465490223_f64;
    const COSH_1: f64 = 1.54308063481524378_f64;
    const SINH_1: f64 = 1.17520119364380146_f64;
    // const TANH_1: f64 = 0.7615941559557647_f64;
    // const COS_SQRT_2: f64 = 0.15594369476537447_f64;
    // const SIN_SQRT_2: f64 = 0.98776594599273553_f64;
    // const TAN_SQRT_2: f64 = 6.33411916704219155_f64;
    const COSH_SQRT_3: f64 = 2.9145774401759282_f64;
    const SINH_SQRT_3: f64 = 2.7376562338581640_f64;
    // const COS_SQRT_3: f64 = -0.16055653857469063_f64;
    // const SIN_SQRT_3: f64 = 0.98702664499035378_f64;
    // const COSH_SQRT_2: f64 = 2.17818355660857086_f64;
    const SINH_SQRT_2: f64 = 1.93506682217435665_f64;
    // const TANH_SQRT_2: f64 = 0.88838556158566054_f64;

    #[test]
    fn test_quaternion_sin_real() {
        let base_angle = 0_f64;
        let angle_multiple = f64::consts::FRAC_PI_8;
        for i in 0..400 {
            let angle = base_angle + angle_multiple * (i as f64);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(angle.sin());
            let result = quaternion.sin();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_quaternion_sin_n_times_pi() {
        for i in 0..400 {
            let angle = (i as f64) * f64::consts::PI;
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::zero();
            let result = quaternion.sin();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-12, relative_all <= f64::EPSILON);
        }

        for i in 0..400 {
            let angle = -(i as f64) * f64::consts::PI;
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::zero();
            let result = quaternion.sin();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-12, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_quaternion_sin_pi_over_two_plus_n_times_two_pi() {
        for i in 0..400 {
            let angle = f64::consts::FRAC_PI_2 + (i as f64) * (2_f64 * f64::consts::PI);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(1_f64);
            let result = quaternion.sin();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-12, relative_all <= f64::EPSILON);
        }

        for i in 0..400 {
            let angle = -(f64::consts::FRAC_PI_2 + (i as f64) * (2_f64 * f64::consts::PI));
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(-1_f64);
            let result = quaternion.sin();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-12, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_quaternion_cos_n_times_two_pi() {
        for i in 0..400 {
            let angle = (i as f64) * (2_f64 * f64::consts::PI);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(1_f64);
            let result = quaternion.cos();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-12, relative_all <= f64::EPSILON);
        }

        for i in 0..400 {
            let angle = -(i as f64) * (2_f64 * f64::consts::PI);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(1_f64);
            let result = quaternion.cos();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-12, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_quaternion_sin1() {
        // quaternion := 0 + 0i + 0k + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin2() {
        // quaternion := 1 + 1i + 1j + 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SIN_1 * COSH_SQRT_3;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64) * (COS_1 * SINH_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin3() {
        // quaternion := 1 - 1i - 1j - 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SIN_1 * COSH_SQRT_3;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64) * (COS_1 * SINH_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin4() {
        // quaternion := 1 + 0i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SIN_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin5() {
        // quaternion := 0 + 1i + 0j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * SINH_1;
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin6() {
        // quaternion := 0 + 0i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * SINH_1;
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin7() {
        // quaternion := 0 + 0i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * SINH_1;
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin8() {
        // quaternion := 1 + 1i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SIN_1 * COSH_1;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * (COS_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin9() {
        // quaternion := 1 - 1i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SIN_1 * COSH_1;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64) * (COS_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin10() {
        // quaternion := 1 + 0i + 1j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SIN_1 * COSH_1;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * (COS_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin11() {
        // quaternion := 1 + 0i - 1j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SIN_1 * COSH_1;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64) * (COS_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin12() {
        // quaternion := 1 + 0i + 0j + 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SIN_1 * COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * (COS_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin13() {
        // quaternion := 1 + 0i + 0j - 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SIN_1 * COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64) * (COS_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin14() {
        // quaternion := 0 + 1i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin15() {
        // quaternion := 0 + 1i - 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin16() {
        // quaternion := 0 + 0i + 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin17() {
        // quaternion := 0 + 0i + 1j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin18() {
        // quaternion := 0 + 1i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin19() {
        // quaternion := 0 + 1i + 0j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin20() {
        // quaternion := 0 - 1i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin21() {
        // quaternion := 0 - 1i - 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin22() {
        // quaternion := 0 + 0i - 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin23() {
        // quaternion := 0 + 0i - 1j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin24() {
        // quaternion := 0 - 1i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sin25() {
        // quaternion := 0 - 1i + 0j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod trigonometry_asin_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    use core::f64;

    const SQRT_2: f64 = 1.41421356237309505_f64;
    const SQRT_3: f64 = 1.73205080756887729_f64;
    const COS_1: f64 = 0.54030230586813972_f64;
    const SIN_1: f64 = 0.84147098480789651_f64;
    // const TAN_1: f64 = 1.55740772465490223_f64;
    const COSH_1: f64 = 1.54308063481524378_f64;
    const SINH_1: f64 = 1.17520119364380146_f64;
    // const TANH_1: f64 = 0.7615941559557647_f64;
    // const COS_SQRT_2: f64 = 0.15594369476537447_f64;
    // const SIN_SQRT_2: f64 = 0.98776594599273553_f64;
    // const TAN_SQRT_2: f64 = 6.33411916704219155_f64;
    const COSH_SQRT_3: f64 = 2.9145774401759282_f64;
    const SINH_SQRT_3: f64 = 2.7376562338581640_f64;
    // const COS_SQRT_3: f64 = -0.16055653857469063_f64;
    // const SIN_SQRT_3: f64 = 0.98702664499035378_f64;
    // const COSH_SQRT_2: f64 = 2.17818355660857086_f64;
    const SINH_SQRT_2: f64 = 1.93506682217435665_f64;
    // const TANH_SQRT_2: f64 = 0.88838556158566054_f64;

    #[test]
    fn test_quaternion_asin1() {
        let quaternion = {
            let scalar = SIN_1 * COSH_SQRT_3;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64) * (COS_1 * SINH_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 1i + 1j + 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin2() {
        let quaternion = {
            let scalar = SIN_1 * COSH_SQRT_3;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64) * (COS_1 * SINH_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 - 1i - 1j - 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin3() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * SINH_1;
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin4() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * SINH_1;
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin5() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * SINH_1;
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 0j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin6() {
        let quaternion = {
            let scalar = SIN_1 * COSH_1;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * (COS_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 1i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin7() {
        let quaternion = {
            let scalar = SIN_1 * COSH_1;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64) * (COS_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 - 1i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin8() {
        let quaternion = {
            let scalar = SIN_1 * COSH_1;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * (COS_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 1j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin9() {
        let quaternion = {
            let scalar = SIN_1 * COSH_1;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64) * (COS_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i - 1j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin10() {
        let quaternion = {
            let scalar = SIN_1 * COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * (COS_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j + 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin11() {
        let quaternion = {
            let scalar = SIN_1 * COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64) * (COS_1 * SINH_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j - 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin12() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin13() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i - 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin14() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 1j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin15() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 1j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin16() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin17() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin18() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i + 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin19() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i - 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin20() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i - 1j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin21() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i - 1j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin22() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i + 0j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asin23() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64) * (SINH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i + 0j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asin();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod trigonometry_tan_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    use core::f64;

    const SQRT_2: f64 = 1.41421356237309505_f64;
    const SQRT_3: f64 = 1.73205080756887729_f64;
    const COS_1: f64 = 0.54030230586813972_f64;
    const SIN_1: f64 = 0.84147098480789651_f64;
    const TAN_1: f64 = 1.55740772465490223_f64;
    const COSH_1: f64 = 1.54308063481524378_f64;
    const SINH_1: f64 = 1.17520119364380146_f64;
    const TANH_1: f64 = 0.7615941559557647_f64;
    // const COS_SQRT_2: f64 = 0.15594369476537447_f64;
    // const SIN_SQRT_2: f64 = 0.98776594599273553_f64;
    // const TAN_SQRT_2: f64 = 6.33411916704219155_f64;
    const COSH_SQRT_3: f64 = 2.9145774401759282_f64;
    const SINH_SQRT_3: f64 = 2.7376562338581640_f64;
    // const COS_SQRT_3: f64 = -0.16055653857469063_f64;
    // const SIN_SQRT_3: f64 = 0.98702664499035378_f64;
    // const COSH_SQRT_2: f64 = 2.17818355660857086_f64;
    // const SINH_SQRT_2: f64 = 1.93506682217435665_f64;
    const TANH_SQRT_2: f64 = 0.88838556158566054_f64;

    #[test]
    fn test_quaternion_tan_real() {
        let base_angle = 0_f64;
        let angle_multiple = f64::consts::FRAC_PI_8;
        for i in 0..400 {
            let angle = base_angle + angle_multiple * (i as f64);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(angle.tan());
            let result = quaternion.tan();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_quaternion_tan1() {
        // quaternion := 0 + 0i + 0k + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan2() {
        // quaternion := 1 + 1i + 1j + 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_SQRT_3 * SINH_SQRT_3;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_SQRT_3 * SINH_SQRT_3;
                let denominator = (COS_1 * COS_1 + SINH_SQRT_3 * SINH_SQRT_3) * SQRT_3;
                let scale = numerator / denominator;
                Vector3::new(1_f64, 1_f64, 1_f64) * scale
            };

            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan3() {
        // quaternion := 1 - 1i - 1j - 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_SQRT_3 * SINH_SQRT_3;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_SQRT_3 * SINH_SQRT_3;
                let denominator = (COS_1 * COS_1 + SINH_SQRT_3 * SINH_SQRT_3) * SQRT_3;
                let scale = numerator / denominator;
                Vector3::new(-1_f64, -1_f64, -1_f64) * scale
            };

            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan4() {
        // quaternion := 1 + 0i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = TAN_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan5() {
        // quaternion := 0 + 1i + 0j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * TANH_1;
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan6() {
        // quaternion := 0 + 0i + 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * TANH_1;
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan7() {
        // quaternion := 0 + 0i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * TANH_1;
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan8() {
        // quaternion := 1 + 1i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_1 * SINH_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                let scale = numerator / denominator;
                Vector3::new(1_f64, 0_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan9() {
        // quaternion := 1 - 1i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_1 * SINH_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                let scale = numerator / denominator;
                Vector3::new(-1_f64, 0_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan10() {
        // quaternion := 1 + 0i + 1j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_1 * SINH_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, 1_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan11() {
        // quaternion := 1 + 0i - 1j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_1 * SINH_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, -1_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan12() {
        // quaternion := 1 + 0i + 0j + 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_1 * SINH_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, 0_f64, 1_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan13() {
        // quaternion := 1 + 0i + 0j - 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_1 * SINH_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, 0_f64, -1_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan14() {
        // quaternion := 0 + 1i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan15() {
        // quaternion := 0 + 1i - 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan16() {
        // quaternion := 0 + 0i + 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan17() {
        // quaternion := 0 + 0i + 1j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan18() {
        // quaternion := 0 + 1i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan19() {
        // quaternion := 0 + 1i + 0j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan20() {
        // quaternion := 0 - 1i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan21() {
        // quaternion := 0 - 1i - 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan22() {
        // quaternion := 0 + 0i - 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan23() {
        // quaternion := 0 + 0i - 1j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan24() {
        // quaternion := 0 - 1i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tan25() {
        // quaternion := 0 - 1i + 0j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod trigonometry_atan_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    use core::f64;

    const SQRT_2: f64 = 1.41421356237309505_f64;
    const SQRT_3: f64 = 1.73205080756887729_f64;
    const COS_1: f64 = 0.54030230586813972_f64;
    const SIN_1: f64 = 0.84147098480789651_f64;
    // const TAN_1: f64 = 1.55740772465490223_f64;
    const COSH_1: f64 = 1.54308063481524378_f64;
    const SINH_1: f64 = 1.17520119364380146_f64;
    const TANH_1: f64 = 0.7615941559557647_f64;
    // const COS_SQRT_2: f64 = 0.15594369476537447_f64;
    // const SIN_SQRT_2: f64 = 0.98776594599273553_f64;
    // const TAN_SQRT_2: f64 = 6.33411916704219155_f64;
    const COSH_SQRT_3: f64 = 2.9145774401759282_f64;
    const SINH_SQRT_3: f64 = 2.7376562338581640_f64;
    // const COS_SQRT_3: f64 = -0.16055653857469063_f64;
    // const SIN_SQRT_3: f64 = 0.98702664499035378_f64;
    // const COSH_SQRT_2: f64 = 2.17818355660857086_f64;
    // const SINH_SQRT_2: f64 = 1.93506682217435665_f64;
    const TANH_SQRT_2: f64 = 0.88838556158566054_f64;

    #[test]
    fn test_quaternion_atan1() {
        let quaternion = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_SQRT_3 * SINH_SQRT_3;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_SQRT_3 * SINH_SQRT_3;
                let denominator = (COS_1 * COS_1 + SINH_SQRT_3 * SINH_SQRT_3) * SQRT_3;
                let scale = numerator / denominator;
                Vector3::new(1_f64, 1_f64, 1_f64) * scale
            };

            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 1i + 1j + 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan2() {
        let quaternion = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_SQRT_3 * SINH_SQRT_3;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_SQRT_3 * SINH_SQRT_3;
                let denominator = (COS_1 * COS_1 + SINH_SQRT_3 * SINH_SQRT_3) * SQRT_3;
                let scale = numerator / denominator;
                Vector3::new(-1_f64, -1_f64, -1_f64) * scale
            };

            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 - 1i - 1j - 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan3() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * TANH_1;
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan4() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * TANH_1;
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 1j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan5() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * TANH_1;
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 0j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan6() {
        let quaternion = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_1 * SINH_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                let scale = numerator / denominator;
                Vector3::new(1_f64, 0_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 1i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan7() {
        let quaternion = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_1 * SINH_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                let scale = numerator / denominator;
                Vector3::new(-1_f64, 0_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 - 1i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan8() {
        let quaternion = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_1 * SINH_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, 1_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 1j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan9() {
        let quaternion = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_1 * SINH_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, -1_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i - 1j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan10() {
        let quaternion = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_1 * SINH_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, 0_f64, 1_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j + 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan11() {
        let quaternion = {
            let scalar = {
                let numerator = SIN_1 * COS_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COSH_1 * SINH_1;
                let denominator = COS_1 * COS_1 + SINH_1 * SINH_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, 0_f64, -1_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j - 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan12() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan13() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i - 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan14() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 1j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan15() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 1j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan16() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan17() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan18() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i + 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan19() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i - 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan20() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i - 1j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan21() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i - 1j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan22() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i + 0j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atan23() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64) * (TANH_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i + 0j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atan();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod trigonometry_cosh_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    use core::f64;

    // const SQRT_2: f64 = 1.41421356237309505_f64;
    const SQRT_3: f64 = 1.73205080756887729_f64;
    const COS_1: f64 = 0.54030230586813972_f64;
    const SIN_1: f64 = 0.84147098480789651_f64;
    // const TAN_1: f64 = 1.55740772465490223_f64;
    const COSH_1: f64 = 1.54308063481524378_f64;
    const SINH_1: f64 = 1.17520119364380146_f64;
    // const TANH_1: f64 = 0.7615941559557647_f64;
    const COS_SQRT_2: f64 = 0.15594369476537447_f64;
    // const SIN_SQRT_2: f64 = 0.98776594599273553_f64;
    // const TAN_SQRT_2: f64 = 6.33411916704219155_f64;
    // const COSH_SQRT_3: f64 = 2.9145774401759282_f64;
    // const SINH_SQRT_3: f64 = 2.7376562338581640_f64;
    const COS_SQRT_3: f64 = -0.16055653857469063_f64;
    const SIN_SQRT_3: f64 = 0.98702664499035378_f64;
    // const COSH_SQRT_2: f64 = 2.17818355660857086_f64;
    // const SINH_SQRT_2: f64 = 1.93506682217435665_f64;
    // const TANH_SQRT_2: f64 = 0.88838556158566054_f64;

    #[test]
    fn test_quaternion_cosh_real() {
        let base_angle = 0_f64;
        let angle_multiple = f64::consts::FRAC_PI_8;
        for i in 0..400 {
            let angle = base_angle + angle_multiple * (i as f64);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(angle.cosh());
            let result = quaternion.cosh();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_quaternion_cosh1() {
        // quaternion := 0 + 0i + 0k + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh2() {
        // quaternion := 1 + 1i + 1j + 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_1 * COS_SQRT_3;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64) * (SINH_1 * SIN_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh3() {
        // quaternion := 1 - 1i - 1j - 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_1 * COS_SQRT_3;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64) * (SINH_1 * SIN_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh4() {
        // quaternion := 1 + 0i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh5() {
        // quaternion := 0 + 1i + 0j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh6() {
        // quaternion := 0 + 0i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh7() {
        // quaternion := 0 + 0i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh8() {
        // quaternion := 1 + 1i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_1 * COS_1;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * (SINH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh9() {
        // quaternion := 1 - 1i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_1 * COS_1;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64) * (SINH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh10() {
        // quaternion := 1 + 0i + 1j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_1 * COS_1;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * (SINH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh11() {
        // quaternion := 1 + 0i - 1j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_1 * COS_1;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64) * (SINH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh12() {
        // quaternion := 1 + 0i + 0j + 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_1 * COS_1;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * (SINH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh13() {
        // quaternion := 1 + 0i + 0j - 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COSH_1 * COS_1;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64) * (SINH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh14() {
        // quaternion := 0 + 1i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh15() {
        // quaternion := 0 + 1i - 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh16() {
        // quaternion := 0 + 0i + 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh17() {
        // quaternion := 0 + 0i + 1j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh18() {
        // quaternion := 0 + 1i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh19() {
        // quaternion := 0 + 1i + 0j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh20() {
        // quaternion := 0 - 1i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh21() {
        // quaternion := 0 - 1i - 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh22() {
        // quaternion := 0 + 0i - 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh23() {
        // quaternion := 0 + 0i - 1j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh24() {
        // quaternion := 0 - 1i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_cosh25() {
        // quaternion := 0 - 1i + 0j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.cosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod trigonometry_acosh_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    use core::f64;

    const SQRT_2: f64 = 1.41421356237309505_f64;
    const SQRT_3: f64 = 1.73205080756887729_f64;
    const COS_1: f64 = 0.54030230586813972_f64;
    const SIN_1: f64 = 0.84147098480789651_f64;
    // const TAN_1: f64 = 1.55740772465490223_f64;
    const COSH_1: f64 = 1.54308063481524378_f64;
    const SINH_1: f64 = 1.17520119364380146_f64;
    // const TANH_1: f64 = 0.7615941559557647_f64;
    const COS_SQRT_2: f64 = 0.15594369476537447_f64;
    // const SIN_SQRT_2: f64 = 0.98776594599273553_f64;
    // const TAN_SQRT_2: f64 = 6.33411916704219155_f64;
    // const COSH_SQRT_3: f64 = 2.9145774401759282_f64;
    // const SINH_SQRT_3: f64 = 2.7376562338581640_f64;
    const COS_SQRT_3: f64 = -0.16055653857469063_f64;
    const SIN_SQRT_3: f64 = 0.98702664499035378_f64;
    // const COSH_SQRT_2: f64 = 2.17818355660857086_f64;
    // const SINH_SQRT_2: f64 = 1.93506682217435665_f64;
    // const TANH_SQRT_2: f64 = 0.88838556158566054_f64;

    #[test]
    fn test_quaternion_acosh1() {
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 0k + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh2() {
        let quaternion = {
            let scalar = COSH_1 * COS_SQRT_3;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64) * (SINH_1 * SIN_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 1i + 1j + 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh3() {
        let quaternion = {
            let scalar = COSH_1 * COS_SQRT_3;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64) * (SINH_1 * SIN_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 - 1i - 1j - 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh4() {
        let quaternion = {
            let scalar = COSH_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh5() {
        let quaternion = {
            let scalar = COS_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh6() {
        // quaternion = cosh(0 + 0i + 1j + 0k)
        let quaternion = {
            let scalar = COS_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh7() {
        // quaternion = cosh(0 + 0i + 0j + 1k)
        let quaternion = {
            let scalar = COS_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh8() {
        let quaternion = {
            let scalar = COSH_1 * COS_1;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * (SINH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 1i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh9() {
        let quaternion = {
            let scalar = COSH_1 * COS_1;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64) * (SINH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 - 1i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh10() {
        let quaternion = {
            let scalar = COSH_1 * COS_1;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * (SINH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 1j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh11() {
        let quaternion = {
            let scalar = COSH_1 * COS_1;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64) * (SINH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i - 1j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh12() {
        let quaternion = {
            let scalar = COSH_1 * COS_1;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * (SINH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j + 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh13() {
        let quaternion = {
            let scalar = COSH_1 * COS_1;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64) * (SINH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j - 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh14() {
        // quaternion := cosh(0 + 1i + 1j + 0k)
        let quaternion = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + sqrt(2) * i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            let scale = SQRT_2;
            Quaternion::from_parts(scalar, vector * scale)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh15() {
        // quaternion := cosh(0 + 1i - 1j + 0k)
        let quaternion = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i - 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            let scale = SQRT_2;
            Quaternion::from_parts(scalar, vector * scale)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh16() {
        // quaternion := cosh(0 + 0i + 1j + 1k)
        let quaternion = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + sqrt(2) * i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            let scale = SQRT_2;
            Quaternion::from_parts(scalar, vector * scale)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh17() {
        // quaternion := cosh(0 + 0i + 1j - 1k)
        let quaternion = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + sqrt(2) * i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            let scale = SQRT_2;
            Quaternion::from_parts(scalar, vector * scale)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh18() {
        // quaternion := cosh(0 + 1i +0j + 1k)
        let quaternion = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + sqrt(2) * i + 0 + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            let scale = SQRT_2;
            Quaternion::from_parts(scalar, vector * scale)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh19() {
        // quaternion := cosh(0 + 1i + 0j - 1k)
        let quaternion = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + sqrt(2) * i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            let scale = SQRT_2;
            Quaternion::from_parts(scalar, vector * scale)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh20() {
        // quaternion := cosh(0 - 1i + 1j + 0k)
        let quaternion = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + sqrt(2) * i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            let scale = SQRT_2;
            Quaternion::from_parts(scalar, vector * scale)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh21() {
        // quaternion := cosh(0 - 1i - 1j + 0k)
        let quaternion = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + sqrt(2) * i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            let scale = SQRT_2;
            Quaternion::from_parts(scalar, vector * scale)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh22() {
        // quaternion := cosh(0 + 0i - 1j + 1k)
        let quaternion = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + sqrt(2) * i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            let scale = SQRT_2;
            Quaternion::from_parts(scalar, vector * scale)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh23() {
        // quaternion := cosh(0 + 0i - 1j - 1k)
        let quaternion = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + sqrt(2) * i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            let scale = SQRT_2;
            Quaternion::from_parts(scalar, vector * scale)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh24() {
        // quaternion := cosh(0 - 1i + 0j + 1k)
        let quaternion = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + sqrt(2) * i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            let scale = SQRT_2;
            Quaternion::from_parts(scalar, vector * scale)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_acosh25() {
        // quaternion := cosh(0 - 1i + 0j - 1k)
        let quaternion = {
            let scalar = COS_SQRT_2;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + sqrt(2) * i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            let scale = SQRT_2;
            Quaternion::from_parts(scalar, vector * scale)
        };
        let result = quaternion.acosh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod trigonometry_sinh_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    use core::f64;

    const SQRT_2: f64 = 1.41421356237309505_f64;
    const SQRT_3: f64 = 1.73205080756887729_f64;
    const COS_1: f64 = 0.54030230586813972_f64;
    const SIN_1: f64 = 0.84147098480789651_f64;
    // const TAN_1: f64 = 1.55740772465490223_f64;
    const COSH_1: f64 = 1.54308063481524378_f64;
    const SINH_1: f64 = 1.17520119364380146_f64;
    // const TANH_1: f64 = 0.7615941559557647_f64;
    // const COS_SQRT_2: f64 = 0.15594369476537447_f64;
    const SIN_SQRT_2: f64 = 0.98776594599273553_f64;
    // const TAN_SQRT_2: f64 = 6.33411916704219155_f64;
    // const COSH_SQRT_3: f64 = 2.9145774401759282_f64;
    // const SINH_SQRT_3: f64 = 2.7376562338581640_f64;
    const COS_SQRT_3: f64 = -0.16055653857469063_f64;
    const SIN_SQRT_3: f64 = 0.98702664499035378_f64;
    // const COSH_SQRT_2: f64 = 2.17818355660857086_f64;
    // const SINH_SQRT_2: f64 = 1.93506682217435665_f64;
    // const TANH_SQRT_2: f64 = 0.88838556158566054_f64;

    #[test]
    fn test_quaternion_sinh_real() {
        let base_angle = 0_f64;
        let angle_multiple = f64::consts::FRAC_PI_8;
        for i in 0..400 {
            let angle = base_angle + angle_multiple * (i as f64);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(angle.sinh());
            let result = quaternion.sinh();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_quaternion_sinh1() {
        // quaternion := 0 + 0i + 0k + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh2() {
        // quaternion := 1 + 1i + 1j + 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SINH_1 * COS_SQRT_3;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64) * (COSH_1 * SIN_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh3() {
        // quaternion := 1 - 1i - 1j - 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SINH_1 * COS_SQRT_3;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64) * (COSH_1 * SIN_SQRT_3 / SQRT_3);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh4() {
        // quaternion := 1 + 0i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SINH_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh5() {
        // quaternion := 0 + 1i + 0j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * SIN_1;
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh6() {
        // quaternion := 0 + 0i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * SIN_1;
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh7() {
        // quaternion := 0 + 0i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * SIN_1;
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh8() {
        // quaternion := 1 + 1i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SINH_1 * COS_1;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * (COSH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh9() {
        // quaternion := 1 - 1i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SINH_1 * COS_1;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64) * (COSH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh10() {
        // quaternion := 1 + 0i + 1j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SINH_1 * COS_1;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * (COSH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh11() {
        // quaternion := 1 + 0i - 1j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SINH_1 * COS_1;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64) * (COSH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh12() {
        // quaternion := 1 + 0i + 0j + 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SINH_1 * COS_1;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * (COSH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh13() {
        // quaternion := 1 + 0i + 0j - 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = SINH_1 * COS_1;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64) * (COSH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh14() {
        // quaternion := 0 + 1i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh15() {
        // quaternion := 0 + 1i - 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh16() {
        // quaternion := 0 + 0i + 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh17() {
        // quaternion := 0 + 0i + 1j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh18() {
        // quaternion := 0 + 1i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh19() {
        // quaternion := 0 + 1i + 0j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh20() {
        // quaternion := 0 - 1i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh21() {
        // quaternion := 0 - 1i - 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh22() {
        // quaternion := 0 + 0i - 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh23() {
        // quaternion := 0 + 0i - 1j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh24() {
        // quaternion := 0 - 1i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_sinh25() {
        // quaternion := 0 - 1i + 0j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.sinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod trigonometry_asinh_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    use core::f64;

    const SQRT_2: f64 = 1.41421356237309505_f64;
    // const SQRT_3: f64 = 1.73205080756887729_f64;
    const COS_1: f64 = 0.54030230586813972_f64;
    const SIN_1: f64 = 0.84147098480789651_f64;
    // const TAN_1: f64 = 1.55740772465490223_f64;
    const COSH_1: f64 = 1.54308063481524378_f64;
    const SINH_1: f64 = 1.17520119364380146_f64;
    // const TANH_1: f64 = 0.7615941559557647_f64;
    // const COS_SQRT_2: f64 = 0.15594369476537447_f64;
    const SIN_SQRT_2: f64 = 0.98776594599273553_f64;
    // const TAN_SQRT_2: f64 = 6.33411916704219155_f64;
    // const COSH_SQRT_3: f64 = 2.9145774401759282_f64;
    // const SINH_SQRT_3: f64 = 2.7376562338581640_f64;
    // const COS_SQRT_3: f64 = -0.16055653857469063_f64;
    // const SIN_SQRT_3: f64 = 0.98702664499035378_f64;
    // const COSH_SQRT_2: f64 = 2.17818355660857086_f64;
    // const SINH_SQRT_2: f64 = 1.93506682217435665_f64;
    // const TANH_SQRT_2: f64 = 0.88838556158566054_f64;

    #[test]
    fn test_quaternion_asinh1() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 0k + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh2() {
        let quaternion = {
            let scalar = SINH_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh3() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * SIN_1;
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh4() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * SIN_1;
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh5() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * SIN_1;
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 0j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh6() {
        let quaternion = {
            let scalar = SINH_1 * COS_1;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * (COSH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 1i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh7() {
        let quaternion = {
            let scalar = SINH_1 * COS_1;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64) * (COSH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 - 1i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh8() {
        let quaternion = {
            let scalar = SINH_1 * COS_1;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * (COSH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 1j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh9() {
        let quaternion = {
            let scalar = SINH_1 * COS_1;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64) * (COSH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i - 1j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh10() {
        let quaternion = {
            let scalar = SINH_1 * COS_1;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * (COSH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j + 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh11() {
        let quaternion = {
            let scalar = SINH_1 * COS_1;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64) * (COSH_1 * SIN_1);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j - 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh12() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh13() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i - 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh14() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 1j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh15() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 1j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh16() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh17() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh18() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i + 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh19() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i - 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh20() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i - 1j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh21() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i - 1j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh22() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i + 0j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_asinh23() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64) * (SIN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i + 0j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.asinh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod trigonometry_tanh_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    use core::f64;

    const SQRT_2: f64 = 1.41421356237309505_f64;
    const SQRT_3: f64 = 1.73205080756887729_f64;
    const COS_1: f64 = 0.54030230586813972_f64;
    const SIN_1: f64 = 0.84147098480789651_f64;
    const TAN_1: f64 = 1.55740772465490223_f64;
    const COSH_1: f64 = 1.54308063481524378_f64;
    const SINH_1: f64 = 1.17520119364380146_f64;
    const TANH_1: f64 = 0.7615941559557647_f64;
    // const COS_SQRT_2: f64 = 0.15594369476537447_f64;
    // const SIN_SQRT_2: f64 = 0.98776594599273553_f64;
    const TAN_SQRT_2: f64 = 6.33411916704219155_f64;
    // const COSH_SQRT_3: f64 = 2.9145774401759282_f64;
    // const SINH_SQRT_3: f64 = 2.7376562338581640_f64;
    const COS_SQRT_3: f64 = -0.16055653857469063_f64;
    const SIN_SQRT_3: f64 = 0.98702664499035378_f64;
    // const COSH_SQRT_2: f64 = 2.17818355660857086_f64;
    // const SINH_SQRT_2: f64 = 1.93506682217435665_f64;
    // const TANH_SQRT_2: f64 = 0.88838556158566054_f64;

    #[test]
    fn test_quaternion_tanh_real() {
        let base_angle = 0_f64;
        let angle_multiple = f64::consts::FRAC_PI_8;
        for i in 0..400 {
            let angle = base_angle + angle_multiple * (i as f64);
            let quaternion = Quaternion::from_real(angle);
            let expected = Quaternion::from_real(angle.tanh());
            let result = quaternion.tanh();

            assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
        }
    }

    #[test]
    fn test_quaternion_tanh1() {
        // quaternion := 0 + 0i + 0k + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh2() {
        // quaternion := 1 + 1i + 1j + 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_SQRT_3 * SIN_SQRT_3;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_SQRT_3 * SIN_SQRT_3;
                let denominator = (COSH_1 * COSH_1 - SIN_SQRT_3 * SIN_SQRT_3) * SQRT_3;
                let scale = numerator / denominator;
                Vector3::new(1_f64, 1_f64, 1_f64) * scale
            };

            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh3() {
        // quaternion := 1 - 1i - 1j - 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_SQRT_3 * SIN_SQRT_3;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_SQRT_3 * SIN_SQRT_3;
                let denominator = (COSH_1 * COSH_1 - SIN_SQRT_3 * SIN_SQRT_3) * SQRT_3;
                let scale = numerator / denominator;
                Vector3::new(-1_f64, -1_f64, -1_f64) * scale
            };

            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh4() {
        // quaternion := 1 + 0i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = TANH_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh5() {
        // quaternion := 0 + 1i + 0j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * TAN_1;
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh6() {
        // quaternion := 0 + 0i + 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * TAN_1;
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh7() {
        // quaternion := 0 + 0i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * TAN_1;
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh8() {
        // quaternion := 1 + 1i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_1 * SIN_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                let scale = numerator / denominator;
                Vector3::new(1_f64, 0_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh9() {
        // quaternion := 1 - 1i + 0j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_1 * SIN_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                let scale = numerator / denominator;
                Vector3::new(-1_f64, 0_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh10() {
        // quaternion := 1 + 0i + 1j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_1 * SIN_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, 1_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh11() {
        // quaternion := 1 + 0i - 1j + 0k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_1 * SIN_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, -1_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh12() {
        // quaternion := 1 + 0i + 0j + 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_1 * SIN_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, 0_f64, 1_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh13() {
        // quaternion := 1 + 0i + 0j - 1k
        let quaternion = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_1 * SIN_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, 0_f64, -1_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh14() {
        // quaternion := 0 + 1i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh15() {
        // quaternion := 0 + 1i - 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh16() {
        // quaternion := 0 + 0i + 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh17() {
        // quaternion := 0 + 0i + 1j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh18() {
        // quaternion := 0 + 1i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh19() {
        // quaternion := 0 + 1i + 0j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh20() {
        // quaternion := 0 - 1i + 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh21() {
        // quaternion := 0 - 1i - 1j + 0k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh22() {
        // quaternion := 0 + 0i - 1j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh23() {
        // quaternion := 0 + 0i - 1j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh24() {
        // quaternion := 0 - 1i + 0j + 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_tanh25() {
        // quaternion := 0 - 1i + 0j - 1k
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.tanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod trigonometry_atanh_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Quaternion,
        Vector3,
    };

    use core::f64;

    const SQRT_2: f64 = 1.41421356237309505_f64;
    // const SQRT_3: f64 = 1.73205080756887729_f64;
    const COS_1: f64 = 0.54030230586813972_f64;
    const SIN_1: f64 = 0.84147098480789651_f64;
    const TAN_1: f64 = 1.55740772465490223_f64;
    const COSH_1: f64 = 1.54308063481524378_f64;
    const SINH_1: f64 = 1.17520119364380146_f64;
    const TANH_1: f64 = 0.7615941559557647_f64;
    // const COS_SQRT_2: f64 = 0.15594369476537447_f64;
    // const SIN_SQRT_2: f64 = 0.98776594599273553_f64;
    const TAN_SQRT_2: f64 = 6.33411916704219155_f64;
    // const COSH_SQRT_3: f64 = 2.9145774401759282_f64;
    // const SINH_SQRT_3: f64 = 2.7376562338581640_f64;
    // const COS_SQRT_3: f64 = -0.16055653857469063_f64;
    // const SIN_SQRT_3: f64 = 0.98702664499035378_f64;
    // const COSH_SQRT_2: f64 = 2.17818355660857086_f64;
    // const SINH_SQRT_2: f64 = 1.93506682217435665_f64;
    // const TANH_SQRT_2: f64 = 0.88838556158566054_f64;

    #[test]
    fn test_quaternion_atanh1() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 0k + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh4() {
        let quaternion = {
            let scalar = TANH_1;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh5() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64) * TAN_1;
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh6() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64) * TAN_1;
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 1j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh7() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64) * TAN_1;
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 0j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh8() {
        let quaternion = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_1 * SIN_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                let scale = numerator / denominator;
                Vector3::new(1_f64, 0_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 1i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh9() {
        let quaternion = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_1 * SIN_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                let scale = numerator / denominator;
                Vector3::new(-1_f64, 0_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 - 1i + 0j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh10() {
        let quaternion = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_1 * SIN_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, 1_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 1j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh11() {
        let quaternion = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_1 * SIN_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, -1_f64, 0_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i - 1j + 0k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh12() {
        let quaternion = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_1 * SIN_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, 0_f64, 1_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j + 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh13() {
        let quaternion = {
            let scalar = {
                let numerator = SINH_1 * COSH_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                numerator / denominator
            };
            let vector = {
                let numerator = COS_1 * SIN_1;
                let denominator = COSH_1 * COSH_1 - SIN_1 * SIN_1;
                let scale = numerator / denominator;
                Vector3::new(0_f64, 0_f64, -1_f64) * scale
            };
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 1 + 0i + 0j - 1k
        let expected = {
            let scalar = 1_f64;
            let vector = Vector3::new(0_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh14() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh15() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i - 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh16() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 1j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh17() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i + 1j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, 1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh18() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh19() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 1i + 0j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh20() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i + 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh21() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i - 1j + 0k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, -1_f64, 0_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh22() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i - 1j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh23() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 + 0i - 1j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(0_f64, -1_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh24() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i + 0j + 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, 1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_quaternion_atanh25() {
        let quaternion = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64) * (TAN_SQRT_2 / SQRT_2);
            Quaternion::from_parts(scalar, vector)
        };
        // expected := 0 - 1i + 0j - 1k
        let expected = {
            let scalar = 0_f64;
            let vector = Vector3::new(-1_f64, 0_f64, -1_f64);
            Quaternion::from_parts(scalar, vector)
        };
        let result = quaternion.atanh();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-13, relative_all <= f64::EPSILON);
    }
}
