extern crate cglinalg;
extern crate num_traits;
extern crate proptest;


#[cfg(test)]
mod storage_tests {
    use cglinalg::Quaternion;

    #[test]
    fn test_as_ref() {
        let v: Quaternion<i32> = Quaternion::new(1, 2, 3, 4);
        let v_ref: &[i32; 4] = v.as_ref();

        assert_eq!(v_ref, &[1, 2, 3, 4]);
    }

    #[test]
    fn test_indices_match_components() {
        let q = Quaternion::new(1, 2, 3, 4);

        assert_eq!(q[0], q.s);
        assert_eq!(q[1], q.v.x);
        assert_eq!(q[2], q.v.y);
        assert_eq!(q[3], q.v.z);
    }

}

#[cfg(test)]
mod lerp_tests {
    use cglinalg::{
        Quaternion,
        Magnitude
    };


    #[test]
    fn test_nlerp() {
        let q1 = Quaternion::new(0_f64, 0_f64, 0_f64, 0_f64);
        let q2 = Quaternion::new(1_f64, 1_f64, 1_f64, 1_f64);
        let amount = 0.5;
        let result = q1.nlerp(&q2, amount);
        let expected = Quaternion::new(0.5, 0.5, 0.5, 0.5);

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
    use cglinalg::{
        Quaternion,
        Vector3,
        Identity
    };


    #[test]
    fn test_unit_axis_quaternions() {
        let i = Quaternion::unit_x();
        let j = Quaternion::unit_y();
        let k = Quaternion::unit_z();

        let result_i = 4_f64 * i;
        let expected_i = Quaternion::from_sv(0_f64, Vector3::new(4_f64, 0_f64, 0_f64));
        let result_j = 4_f64 * j;
        let expected_j = Quaternion::from_sv(0_f64, Vector3::new(0_f64, 4_f64, 0_f64));
        let result_k = 4_f64 * k;
        let expected_k = Quaternion::from_sv(0_f64, Vector3::new(0_f64, 0_f64, 4_f64));

        assert_eq!(result_i, expected_i);
        assert_eq!(result_j, expected_j);
        assert_eq!(result_k, expected_k);
    }

    #[test]
    fn test_quaternion_addition() {
        let q1 = Quaternion::new(1, 2, 3, 4);
        let q2 = Quaternion::new(5, 6, 7, 8);
        let expected = Quaternion::new(6, 8, 10, 12);
        let result = q1 + q2;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_subtraction() {
        let q1 = Quaternion::new(1, 2, 3, 4);
        let q2 = Quaternion::new(5, 6, 7, 8);
        let expected = Quaternion::new(-4, -4, -4, -4);
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
mod magnitude_tests {
    use cglinalg::{
        Quaternion,
        Magnitude,
        Vector3,
    };
    use cglinalg::approx::relative_eq;


    #[test]
    fn test_unit_axis_quaternions_should_have_unit_norms() {
        let i = Quaternion::<f64>::unit_x();
        let j = Quaternion::<f64>::unit_y();
        let k = Quaternion::<f64>::unit_z();
    
        assert_eq!(i.magnitude(), 1_f64);
        assert_eq!(j.magnitude(), 1_f64);
        assert_eq!(k.magnitude(), 1_f64);
    }

    #[test]
    fn test_quaternion_magnitude() {
        let q = Quaternion::from_sv(3_f64, Vector3::new(34.8, 75.1939, 1.0366));
        let result = q.magnitude_squared();
        let expected = 6875.23713677;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_normalized() {
        let q = Quaternion::from_sv(3_f64, Vector3::new(34.8, 75.1939, 1.0366));
        let result = q.normalize().magnitude();
        let expected = 1_f64;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_quaternion_normalized_to() {
        let q = Quaternion::from_sv(3_f64, Vector3::new(34.8, 75.1939, 1.0366));
        let magnitude = 12_f64;
        let result = q.normalize_to(magnitude).magnitude();
        let expected = magnitude;
        let tolerance = 1e-7;

        assert!(relative_eq!(result, expected, epsilon = tolerance));
    }
}

#[cfg(test)]
mod slerp_tests {
    use cglinalg::{
        Quaternion,
        Angle,
        Degrees,
        Vector3,
    };
    use cglinalg::approx::{
        relative_eq,
    };


    #[test]
    fn test_slerp_upper_right_quadrant() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(60_f64);
        let unit_z = Vector3::unit_z();
        let q1 = Quaternion::from_sv(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q2 = Quaternion::from_sv(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );
        let angle_expected = Degrees(45_f64);
        let expected = Quaternion::from_sv(
            Angle::cos(angle_expected / 2_f64), 
            Angle::sin(angle_expected / 2_f64) * unit_z
        );
        let result = q1.slerp(&q2, 0.5);

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }

    #[test]
    fn test_slerp_upper_right_quadrant1() {
        let angle1 = Degrees(20_f64);
        let angle2 = Degrees(70_f64);
        let unit_z = Vector3::unit_z();
        let q1 = Quaternion::from_sv(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q2 = Quaternion::from_sv(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );
        let angle_expected = Degrees(30_f64);
        let expected = Quaternion::from_sv(
            Angle::cos(angle_expected / 2_f64), 
            Angle::sin(angle_expected / 2_f64) * unit_z
        );
        let result = q1.slerp(&q2, 0.2);

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }

    #[test]
    fn test_slerp_upper_half_plane() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(150_f64);
        let unit_z = Vector3::unit_z();
        let q1 = Quaternion::from_sv(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q2 = Quaternion::from_sv(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );
        let angle_expected = Degrees(90_f64);
        let expected = Quaternion::from_sv(
            Angle::cos(angle_expected / 2_f64), 
            Angle::sin(angle_expected / 2_f64) * unit_z
        );
        let result = q1.slerp(&q2, 0.5);

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }

    #[test]
    fn test_slerp_negative_dot_product() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(240_f64);
        let unit_z = Vector3::unit_z();
        let q1 = Quaternion::from_sv(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q2 = Quaternion::from_sv(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );
        let angle_expected = Degrees(315_f64);
        let expected = Quaternion::from_sv(
            Angle::cos(angle_expected / 2_f64), 
            Angle::sin(angle_expected / 2_f64) * unit_z
        );
        let result = q1.slerp(&q2, 0.5);

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }

    #[test]
    fn test_slerp_endpoints0() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(240_f64);
        let unit_z = Vector3::unit_z();
        let q0 = Quaternion::from_sv(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q1 = Quaternion::from_sv(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );

        // The slerp function can produce either the starting quaternion
        // or its negation at 0.0. Both quaternions produce the same rotation.
        let expected1 = q0;
        let expected2 = -q0;
        let result = q0.slerp(&q1, 0.0);

        assert!(result == expected1 || result == expected2);
    }

    #[test]
    fn test_slerp_endpoints1() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(240_f64);
        let unit_z = Vector3::unit_z();
        let q0 = Quaternion::from_sv(
            Angle::cos(angle1 / 2_f64), 
            Angle::sin(angle1 / 2_f64) * unit_z
        );
        let q1 = Quaternion::from_sv(
            Angle::cos(angle2 / 2_f64), 
            Angle::sin(angle2 / 2_f64) * unit_z
        );

        let expected = q1;
        let result = q0.slerp(&q1, 1.0);

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod arg_tests {
    use cglinalg::Quaternion;

    
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
}

#[cfg(test)]
mod exp_tests {
    use cglinalg::{
        Magnitude,
        Quaternion,
        Identity,
        AdditiveIdentity,
    };
    use cglinalg::approx::relative_eq;


    #[test]
    fn test_quaternion_exp_zero() {
        let zero_quat: Quaternion<f64> = Quaternion::zero();
        let one_quat: Quaternion<f64> = Quaternion::identity();
        let result = zero_quat.exp();

        assert_eq!(result, one_quat);
    }

    #[test]
    fn test_quaternion_exp_power_times_pi() {
        let q: Quaternion<f64> = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let sgn_qv = Quaternion::from_sv(0_f64, q.v / q.v.magnitude());
        let pi = core::f64::consts::PI;
        let expected = -Quaternion::identity();
        let result = (sgn_qv * pi).exp();

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }

    #[test]
    fn test_quaternion_exp_inverse() {
        let q: Quaternion<f64> = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let expected = Quaternion::identity();
        let result = (-q).exp() * q.exp();

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }

    #[test]
    fn test_quaternion_conjugate_commutes_with_exp() {
        let q: Quaternion<f64> = Quaternion::new(1_f64, 2_f64, 3_f64, 4_f64);
        let result = q.exp().conjugate();
        let expected = q.conjugate().exp();

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod logarithm_tests {
    use cglinalg::{
        Vector3,
        Quaternion,
        AdditiveIdentity,
        Identity,
    };
    use cglinalg::approx::relative_eq;


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

        assert!(relative_eq!(exp_i.ln(), i, epsilon = 1e-7));
        assert!(relative_eq!(exp_j.ln(), j, epsilon = 1e-7));
        assert!(relative_eq!(exp_k.ln(), k, epsilon = 1e-7));
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
        let expected = Quaternion::from_sv(expected_s, expected_v);
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
mod power_tests {
    use cglinalg::{
        Quaternion,
        Identity,
    };
    use cglinalg::approx::{
        relative_eq
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

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }

    #[test]
    fn test_quaternion_power1() {
        let i = Quaternion::<f64>::unit_x();
        let exponent = 2_f64;
        let expected = Quaternion::new(-1_f64, 0_f64, 0_f64, 0_f64);
        let result = i.powf(exponent);

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }

    #[test]
    fn test_quaternion_power2() {
        let j = Quaternion::<f64>::unit_y();
        let exponent = 2_f64;
        let expected = Quaternion::new(-1_f64, 0_f64, 0_f64, 0_f64);
        let result = j.powf(exponent);

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }

    #[test]
    fn test_quaternion_power3() {
        let k = Quaternion::<f64>::unit_z();
        let exponent = 2_f64;
        let expected = Quaternion::new(-1_f64, 0_f64, 0_f64, 0_f64);
        let result = k.powf(exponent);

        assert!(relative_eq!(result, expected, epsilon = 1e-7));
    }
}

#[cfg(test)]
mod rotation_tests {
    use cglinalg::{
        Angle,
        Radians,
        Unit,
        Vector3,
        Quaternion,
    };
    use cglinalg::approx::{
        relative_eq,
    };

    
    #[test]
    fn test_rotation_between_unit_vectors() {
        let unit_x: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x());
        let unit_y: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y());
        let unit_z: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
        let expected = Quaternion::from_axis_angle(
            unit_z, 
            Radians::full_turn_div_4()
        );
        let result = Quaternion::rotation_between_axis(&unit_x, &unit_y).unwrap();

        assert!(relative_eq!(result, expected));
    }

    #[test]
    fn test_rotation_between_same_unit_vectors() {
        let unit_v1: Unit<Vector3<f64>> = Unit::from_value(
            Vector3::new(1.0, 1.0, 0.0)
        );
        let unit_v2: Unit<Vector3<f64>> = Unit::from_value(
            Vector3::new(1.0, 1.0, 0.0)
        );
        let unit_z: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
        let expected = Quaternion::from_axis_angle(
            unit_z, 
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
            Unit::from_value(unit_z), 
            Radians::full_turn_div_4()
        );
        let result = Quaternion::rotation_between(&unit_x, &unit_y).unwrap();

        assert!(relative_eq!(result, expected));
    }

    #[test]
    fn test_rotation_between_same_vectors() {
        let v1 = Vector3::new(1.0, 1.0, 0.0) * 3.0;
        let v2 = Vector3::new(1.0, 1.0, 0.0) * 3.0;
        let unit_z = Vector3::unit_z();
        let expected = Quaternion::from_axis_angle(
            Unit::from_value(unit_z), 
            Radians(0_f64)
        );
        let result = Quaternion::rotation_between(&v1, &v2).unwrap();

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod inverse_tests {
    use cglinalg::{
        Quaternion,
        AdditiveIdentity,
    };


    #[test]
    fn test_inverse() {
        let quaternion = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let magnitude = 30_f64;
        let expected = Quaternion::new(1.0, -2.0, -3.0, -4.0) / magnitude;
        let result = quaternion.inverse().unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_is_invertible() {
        let quaternion = Quaternion::new(1.0, 2.0, 3.0, 4.0);

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