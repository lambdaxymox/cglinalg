extern crate cglinalg;
extern crate num_traits;
extern crate proptest;


#[cfg(test)]
mod conversion_tests {
    use cglinalg::{
        Radians,
        Degrees,
    };
    use std::f64;


    struct Test<A, B> {
        input: A,
        expected: B, 
    }

    fn radians_to_degrees_tests() -> Vec<Test<Radians<f64>, Degrees<f64>>> {
        let pi = f64::consts::PI;
        vec![
            Test { input: Radians(0_f64),              expected: Degrees(0_f64)   },
            Test { input: Radians(pi / 4_f64),         expected: Degrees(45_f64)  },
            Test { input: Radians(pi / 2_f64),         expected: Degrees(90_f64)  },
            Test { input: Radians(3_f64 * pi / 4_f64), expected: Degrees(135_f64) },
            Test { input: Radians(pi),                 expected: Degrees(180_f64) },
            Test { input: Radians(5_f64 * pi / 4_f64), expected: Degrees(225_f64) },
            Test { input: Radians(3_f64 * pi / 2_f64), expected: Degrees(270_f64) },
            Test { input: Radians(7_f64 * pi / 4_f64), expected: Degrees(315_f64) },
            Test { input: Radians(2_f64 * pi),         expected: Degrees(360_f64) },
        ]
    }

    fn degrees_to_radians_tests() -> Vec<Test<Degrees<f64>, Radians<f64>>> {
        let pi = f64::consts::PI;
        vec![
            Test { input: Degrees(0_f64),   expected: Radians(0_f64)              },
            Test { input: Degrees(45_f64),  expected: Radians(pi / 4_f64)         },
            Test { input: Degrees(90_f64),  expected: Radians(pi / 2_f64)         },
            Test { input: Degrees(135_f64), expected: Radians(3_f64 * pi / 4_f64) },
            Test { input: Degrees(180_f64), expected: Radians(pi)                 },
            Test { input: Degrees(225_f64), expected: Radians(5_f64 * pi / 4_f64) },
            Test { input: Degrees(270_f64), expected: Radians(3_f64 * pi / 2_f64) },
            Test { input: Degrees(315_f64), expected: Radians(7_f64 * pi / 4_f64) },
            Test { input: Degrees(360_f64), expected: Radians(2_f64 * pi)         },
        ]
    }

    /// Converting an angle from radians to degrees should yield the correct angle in
    /// in degrees.
    #[test]
    fn convert_radians_to_degrees() {
        radians_to_degrees_tests().iter().for_each(|test| {
            let result = Degrees::from(test.input);
            let expected = test.expected;

            assert_eq!(result, expected);
        })
    }

    /// Converting an angle from degrees to radians should yield the correct angle
    /// in radians.
    #[test]
    fn convert_degrees_to_radians() {
        degrees_to_radians_tests().iter().for_each(|test| {
            let result = Radians::from(test.input);
            let expected = test.expected;

            assert_eq!(result, expected);
        })
    }
}

#[cfg(test)]
mod degrees_arithmetic_tests {
    use cglinalg::Degrees;


    #[test]
    fn test_addition() {
        let angle1 = Degrees(30_f64); 
        let angle2 = Degrees(45_f64);
        let expected = Degrees(75_f64);
        
        let result = angle1 + angle2;
        assert_eq!(result, expected);

        let result = &angle1 + angle2;
        assert_eq!(result, expected);

        let result = angle1 + &angle2;
        assert_eq!(result, expected);

        let result = &angle1 + &angle2;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(45_f64);
        let expected = -Degrees(15_f64);
        
        let result = angle1 - angle2;
        assert_eq!(result, expected);

        let result = &angle1 - angle2;
        assert_eq!(result, expected);

        let result = angle1 - &angle2;
        assert_eq!(result, expected);

        let result = &angle1 - &angle2;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiplication() {
        let angle = Degrees(30_f64);
        let c = 45_f64;
        let expected = Degrees(30_f64 * 45_f64);
        
        let result = angle * c;
        assert_eq!(result, expected);

        let result = &angle * c;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_division() {
        let angle1 = Degrees(30_f64);
        let angle2 = Degrees(45_f64);
        let expected = 30_f64 / 45_f64;
        
        let result = angle1 / angle2;
        assert_eq!(result, expected);

        let result = &angle1 / angle2;
        assert_eq!(result, expected);

        let result = angle1 / &angle2;
        assert_eq!(result, expected);

        let result = &angle1 / &angle2;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_negation() {
        let angle = Degrees(30_f64);
        let expected = Degrees(-30_f64);
        
        let result = -angle;
        assert_eq!(result, expected);

        let result = -&angle;
        assert_eq!(result, expected)
    }

    /// The remainder of an angle by a modulus smaller than the modulus should be
    /// the same as the original angle. That is, angles satisfy
    /// ```
    /// For each angle smaller than modulus angle is congruent to itself modulo modulus.
    /// That is,
    /// For all angle < modulus, angle = angle (mod modulus).
    /// ``` 
    #[test]
    fn test_remainder_less_than_modulus() {
        let angle = Degrees(45_f64);
        let modulus = Degrees(360_f64);
        let expected = angle;

        let result = angle % modulus;
        assert_eq!(result, expected);

        let result = &angle % modulus;
        assert_eq!(result, expected);

        let result = angle % &modulus;
        assert_eq!(result, expected);

        let result = &angle % &modulus;
        assert_eq!(result, expected);
    }

    /// The remainder of an angle by a modulus larger than the modulus should be
    /// smaller then the modulus. That is, angles satisfy
    /// ```
    /// For each angle larger than modulus, angle mod modulus < modulus.
    /// ``` 
    #[test]
    fn test_remainder_greater_than_modulus() {
        let angle = Degrees(405_f64);
        let modulus = Degrees(360_f64);
        let expected = Degrees(45_f64);

        let result = angle % modulus;
        assert_eq!(result, expected);

        let result = &angle % modulus;
        assert_eq!(result, expected);

        let result = angle % &modulus;
        assert_eq!(result, expected);

        let result = &angle % &modulus;
        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod radians_arithmetic_tests {
    use cglinalg::Radians;
    use cglinalg::approx::relative_eq;
    use std::f64;

    const PI: Radians<f64> = Radians(f64::consts::PI);


    #[test]
    fn test_addition() {
        let angle1 = PI / 6_f64; 
        let angle2 = PI / 4_f64;
        let expected = PI * 10_f64 / 24_f64;

        let result = angle1 + angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 + angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = angle1 + &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 + &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_subtraction() {
        let angle1 = PI / 6_f64;
        let angle2 = PI / 4_f64;
        let expected = -PI / 12_f64;

        let result = angle1 - angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 - angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = angle1 - &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 - &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_multiplication() {
        let angle = PI / 6_f64;
        let c = 4_f64;
        let expected = Radians(f64::consts::PI * 4_f64 / 6_f64);

        let result = angle * c;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle * c;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_division() {
        let angle1 = PI / 6_f64;
        let angle2 = PI / 4_f64;
        let expected = 4_f64 / 6_f64;

        let result = angle1 / angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 / angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = angle1 / &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = &angle1 / &angle2;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_negation() {
        let angle = PI / 6_f64;
        let expected = -PI / 6_f64;
        
        let result = -angle;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));

        let result = -&angle;
        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    /// The remainder of an angle by a modulus smaller than the modulus should be
    /// the same as the original angle.
    ///
    /// That is, given an angle `angle` smaller than modulus `modulus`, `angle` is congruent to itself
    /// modulo `modulus`
    /// ```
    /// angle = angle (mod modulus).
    /// ``` 
    #[test]
    fn test_remainder_less_than_modulus() {
        let angle = PI / 4_f64;
        let modulus = PI * 2_f64;
        let expected = angle;

        let result = angle % modulus;
        assert_eq!(result, expected);

        let result = &angle % modulus;
        assert_eq!(result, expected);

        let result = angle % &modulus;
        assert_eq!(result, expected);

        let result = &angle % &modulus;
        assert_eq!(result, expected);
    }

    /// The remainder of an angle by a modulus larger than the modulus should be
    /// smaller then the modulus. That is, angles satisfy
    /// ```
    /// For each angle larger than modulus, angle mod modulus < modulus.
    /// ``` 
    #[test]
    fn test_remainder_greater_than_modulus() {
        let angle = PI * 2_f64 + PI / 4_f64;
        let modulus = PI * 2_f64;
        let expected = PI / 4_f64;

        let result = angle % modulus;
        assert_eq!(result, expected);

        let result = &angle % modulus;
        assert_eq!(result, expected);

        let result = angle % &modulus;
        assert_eq!(result, expected);

        let result = &angle % &modulus;
        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod radian_angle_tests {
    use cglinalg::{
        Radians,
        Angle
    };
    use cglinalg::approx::relative_eq;
    use std::f64;


    #[test]
    fn test_full_turn() {
        let expected = Radians(2_f64 * f64::consts::PI);
        let result = Radians::full_turn();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sin() {
        let expected = 1_f64 / 2_f64;
        let result = Radians(f64::consts::PI / 6_f64).sin();

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_cos() {
        let expected = f64::sqrt(3_f64) / 2_f64;
        let result = Radians(f64::consts::PI / 6_f64).cos();

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_tan() {
        let expected = f64::sqrt(3_f64) / 3_f64;
        let result = Radians(f64::consts::PI / 6_f64).tan();

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[ignore]
    #[test]
    fn test_tangent_of_vertical_angle_should_be_infinite() {
        let expected = f64::INFINITY;
        let result = Radians(f64::consts::FRAC_PI_2).tan();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_asin() {
        let expected = Radians(f64::consts::PI / 6_f64);
        let result = Radians::asin(1_f64 / 2_f64);

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_acos() {
        let expected = Radians(f64::consts::PI / 6_f64);
        let result = Radians::acos(f64::sqrt(3_f64) / 2_f64);

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_atan() {
        let expected = Radians(f64::consts::PI / 6_f64);
        let result = Radians::atan(f64::sqrt(3_f64) / 3_f64);

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_atan_of_infinity_should_be_pi_over_two() {
        let expected = Radians(f64::consts::FRAC_PI_2);
        let result = Radians::atan(f64::INFINITY);

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_sin_cos() {
        let angle = Radians(f64::consts::PI / 6_f64);
        let expected = (angle.sin(), angle.cos());
        let result = angle.sin_cos();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_full_turn_div_2() {
        let expected = Radians(2_f64 * f64::consts::PI / 2_f64);
        let result = Radians::full_turn_div_2();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_full_turn_div_4() {
        let expected = Radians(2_f64 * f64::consts::PI / 4_f64);
        let result = Radians::full_turn_div_4();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_full_turn_div_6() {
        let expected = Radians(2_f64 * f64::consts::PI / 6_f64);
        let result = Radians::full_turn_div_6();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_full_turn_div_8() {
        let expected = Radians(2_f64 * f64::consts::PI / 8_f64);
        let result = Radians::full_turn_div_8();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_normalize() {
        let angle = Radians::full_turn() + Radians(f64::consts::PI / 4_f64);
        let expected = Radians(f64::consts::PI / 4_f64);
        let result = angle.normalize();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_normalize_signed() {
        let angle = Radians::full_turn() + Radians(f64::consts::PI / 4_f64);
        let expected = Radians(f64::consts::PI / 4_f64);
        let result = angle.normalize_signed();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_opposite() {
        let angle = Radians(f64::consts::PI / 4_f64);
        let expected = Radians(5_f64 * f64::consts::PI / 4_f64);
        let result = angle.opposite();

        assert_eq!(result, expected);    
    }

    #[test]
    fn test_bisect() {
        let angle1 = Radians(0_f64);
        let angle2 = Radians(f64::consts::PI / 2_f64);
        let expected = Radians(f64::consts::PI / 4_f64);
        let result = angle1.bisect(angle2);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_csc() {
        let expected = 2_f64;
        let result = Radians(f64::consts::PI / 6_f64).csc();

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_cot() {
        let expected = 3_f64 / f64::sqrt(3_f64);
        let result = Radians(f64::consts::PI / 6_f64).cot();

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_sec() {
        let expected = 2_f64 / f64::sqrt(3_f64);
        let result = Radians(f64::consts::PI / 6_f64).sec();

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }
}

#[cfg(test)]
mod degree_angle_tests {
    use cglinalg::{
        Degrees,
        Angle,
    };
    use cglinalg::approx::relative_eq;
    use std::f64;


    #[test]
    fn test_full_turn() {
        let expected = Degrees(360_f64);
        let result = Degrees::full_turn();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sin() {
        let expected = 1_f64 / 2_f64;
        let result = Degrees(30_f64).sin();

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_cos() {
        let expected = f64::sqrt(3_f64) / 2_f64;
        let result = Degrees(30_f64).cos();

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_tan() {
        let expected = f64::sqrt(3_f64) / 3_f64;
        let result = Degrees(30_f64).tan();

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[ignore]
    #[test]
    fn test_tangent_of_vertical_angle_should_be_infinite() {
        let expected = f64::INFINITY;
        let result = Degrees(f64::consts::FRAC_PI_2).tan();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_asin() {
        let expected = Degrees(30_f64);
        let result = Degrees::asin(1_f64 / 2_f64);

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_acos() {
        let expected = Degrees(30_f64);
        let result = Degrees::acos(f64::sqrt(3_f64) / 2_f64);

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_atan() {
        let expected = Degrees(30_f64);
        let result = Degrees::atan(f64::sqrt(3_f64) / 3_f64);

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_atan_of_infinity_should_be_pi_over_two() {
        let expected = Degrees(90_f64);
        let result = Degrees::atan(f64::INFINITY);

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_sin_cos() {
        let angle = Degrees(30_f64);
        let expected = (angle.sin(), angle.cos());
        let result = angle.sin_cos();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_full_turn_div_2() {
        let expected = Degrees(180_f64);
        let result = Degrees::full_turn_div_2();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_full_turn_div_4() {
        let expected = Degrees(90_f64);
        let result = Degrees::full_turn_div_4();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_full_turn_div_6() {
        let expected = Degrees(60_f64);
        let result = Degrees::full_turn_div_6();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_full_turn_div_8() {
        let expected = Degrees(45_f64);
        let result = Degrees::full_turn_div_8();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_normalize() {
        let angle = Degrees::full_turn() + Degrees(45_f64);
        let expected = Degrees(45_f64);
        let result = angle.normalize();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_normalize_signed() {
        let angle = Degrees::full_turn() + Degrees(45_f64);
        let expected = Degrees(45_f64);
        let result = angle.normalize_signed();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_opposite() {
        let angle = Degrees(45_f64);
        let expected = Degrees(225_f64);
        let result = angle.opposite();

        assert_eq!(result, expected);    
    }

    #[test]
    fn test_bisect() {
        let angle1 = Degrees(0_f64);
        let angle2 = Degrees(90_f64);
        let expected = Degrees(45_f64);
        let result = angle1.bisect(angle2);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_csc() {
        let expected = 2_f64;
        let result = Degrees(30_f64).csc();

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_cot() {
        let expected = 3_f64 / f64::sqrt(3_f64);
        let result = Degrees(30_f64).cot();

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }

    #[test]
    fn test_sec() {
        let expected = 2_f64 / f64::sqrt(3_f64);
        let result = Degrees(30_f64).sec();

        assert!(relative_eq!(result, expected, epsilon = 1e-10));
    }
}
