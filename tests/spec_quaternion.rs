extern crate gdmath;
extern crate num_traits;
extern crate proptest;

use proptest::prelude::*;
use gdmath::{
    Quaternion, 
    Scalar,
};


fn any_quaternion<S>() -> impl Strategy<Value = Quaternion<S>> where S: Scalar + Arbitrary {
    any::<(S, S, S, S)>().prop_map(|(x, y, z, w)| Quaternion::new(x, y, z, w))
}


/// Generates the properties tests for quaternion indexing.
///
/// `$TestModuleName` is a name we give to the module we place the tests in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers that compose the quaternions.
/// `$Generator` is the name of a function or closure for generating examples.
/// `$UpperBound` denotes the upperbound on the range of acceptable indices.
macro_rules! index_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $UpperBound:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;

        proptest! {
            /// When a quaternion is treated like an array, it should accept all indices
            /// below the length of the array.
            ///
            /// Given a quaternion `q`, it should return the element at position `index` in the 
            /// underlying storage of the quaternion when the given index is inbounds.
            #[test]
            fn prop_accepts_all_indices_in_of_bounds(
                q in super::$Generator::<$ScalarType>(), index in 0..$UpperBound as usize) {

                prop_assert_eq!(q[index], q[index]);
            }
    
            /// When a quaternion is treated like an array, it should reject any input index outside
            /// the length of the array.
            ///
            /// Given a quaternion `q`, when the element index `index` is out of bounds, it should 
            /// generate a panic just like an array indexed out of bounds.
            #[test]
            #[should_panic]
            fn prop_panics_when_index_out_of_bounds(
                q in super::$Generator::<$ScalarType>(), index in $UpperBound..usize::MAX) {
                
                prop_assert_eq!(q[index], q[index]);
            }
        }
    }
    }
}

index_props!(quaternion_index_props, f64, any_quaternion, 4);


/// Generate the properties for quaternion arithmetic over exact scalars. We define an exact
/// scalar type as a type where scalar arithmetic is exact (e.g. integers).
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers that compose the quaternions.
/// `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_arithmetic_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{Quaternion, Zero};

        proptest! {
            /// A scalar `0` times a quaternion should be a zero quaternion.
            ///
            /// Given a quaternion `q`, it satisfies
            /// ```
            /// 0 * q = 0.
            /// ```
            #[test]
            fn prop_zero_times_quaternion_equals_zero(q in super::$Generator()) {
                let zero: $ScalarType = num_traits::Zero::zero();
                let zero_quat = Quaternion::zero();
                prop_assert_eq!(zero * q, zero_quat);
            }
        
            /// A scalar `0` times a quaternion should be zero.
            ///
            /// Given a quaternion `q`, it satisfies
            /// ```
            /// q * 0 = 0
            /// ```
            #[test]
            fn prop_quaternion_times_zero_equals_zero(q in super::$Generator()) {
                let zero: $ScalarType = num_traits::Zero::zero();
                let zero_quat = Quaternion::zero();
                prop_assert_eq!(q * zero, zero_quat);
            }

            /// A zero quaternion should act as the additive unit element of a set of quaternions.
            ///
            /// Given a quaternion `q`
            /// ```
            /// q + 0 = q
            /// ```
            #[test]
            fn prop_quaternion_plus_zero_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(q + zero_quat, q);
            }

            /// A zero quaternion should act as the additive unit element of a set of quaternions.
            ///
            /// Given a quaternion `q`
            /// ```
            /// 0 + q = q
            /// ```
            #[test]
            fn prop_zero_plus_quaternion_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(zero_quat + q, q);
            }

            /// Multiplying a quaternion by a scalar `1` should give the original quaternion.
            ///
            /// Given a quaternion `q`
            /// ```
            /// 1 * q = q
            /// ```
            #[test]
            fn prop_one_times_quaternion_equal_quaternion(q in super::$Generator()) {
                let one: $ScalarType = num_traits::One::one();
                prop_assert_eq!(one * q, q);
            }

            /// Multiplying a quaternion by a scalar `1` should give the original quaternion.
            ///
            /// Given a quaternion `q`
            /// ```
            /// q * 1 = q.
            /// ```
            #[test]
            fn prop_quaternion_times_one_equals_quaternion(v in super::$Generator()) {
                let one: $ScalarType = num_traits::One::one();
                prop_assert_eq!(one * v, v);
            }
        }
    }
    }
}

exact_arithmetic_props!(quaternion_f64_arithmetic_props, f64, any_quaternion);
exact_arithmetic_props!(quaternion_i32_arithmetic_props, i32, any_quaternion);
exact_arithmetic_props!(quaternion_u32_arithmetic_props, u32, any_quaternion);


/// Generate the properties for quaternion arithmetic over floating point scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers that compose the quaternions.
/// `$Generator` is the name of a function or closure for generating examples.
/// `$tolerance` specifies the highest amount of acceptable error in the floating point computations
///  that still defines a correct computation. We cannot guarantee floating point computations
///  will be exact since the underlying floating point arithmetic is not exact.
///
/// We use approximate comparisons because arithmetic is not exact over finite precision floating point
/// scalar types.
macro_rules! approx_add_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{Quaternion, Zero};
        use gdmath::approx::relative_eq;

        proptest! {
            /// A quaternion plus a zero quaternion equals the same quaternion. 
            ///
            /// Given a quaternion `q`
            /// ```
            /// q + 0 = q
            /// ```
            #[test]
            fn prop_quaternion_plus_zero_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(q + zero_quat, q);
            }

            /// A quaternion plus a zero quaternion equals the same quaternion.
            /// 
            /// Given a quaternion `q`
            /// ```
            /// 0 + q = q
            /// ```
            #[test]
            fn prop_zero_plus_quaternion_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(zero_quat + q, q);
            }

            /// Given quaternions `q1` and `q2`, we should be able to use `q1` and `q2` interchangeably 
            /// with their references `&q1` and `&q2` in arithmetic expressions involving quaternions.
            ///
            /// Given quaternions `q1` and `q2`, and their references `&q1` and `&q2`, they 
            /// should satisfy
            /// ```
            ///  q1 +  q2 = &q1 +  q2
            ///  q1 +  q2 =  q1 + &q2
            ///  q1 +  q2 = &q1 + &q2
            ///  q1 + &q2 = &q1 +  q2
            /// &q1 +  q2 =  q1 + &q2
            /// &q1 +  q2 = &q1 + &q2
            ///  q1 + &q2 = &q1 + &vq
            /// ```
            #[test]
            fn prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>()) {
                
                prop_assert_eq!(q1 + q2, &q1 + q2);
                prop_assert_eq!(q1 + q2, q1 + &q2);
                prop_assert_eq!(q1 + q2, &q1 + &q2);
                prop_assert_eq!(q1 + &q2, &q1 + q2);
                prop_assert_eq!(&q1 + q2, q1 + &q2);
                prop_assert_eq!(&q1 + q2, &q1 + &q2);
                prop_assert_eq!(q1 + &q2, &q1 + &q2);
            }

            /// Quaternion addition over floating point scalars should  be approximately commutative.
            /// 
            /// Given quaternions `q1` and `q2`, we have
            /// ```
            /// q1 + q2 ~= q2 + q1
            /// ```
            /// Note that floating point quaternion addition cannot be exactly commutative because arithmetic
            /// with floating point numbers is not commutative.
            #[test]
            fn prop_quaternion_addition_almost_commutative(
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>()) {
                
                let zero: Quaternion<$ScalarType> = Zero::zero();
                prop_assert_eq!((q1 + q2) - (q2 + q1), zero);
            }

            /// Quaternion addition over floating point scalars should be approximately associative. 
            ///
            /// Given quaternions `q1`, `q2`, and `q3` we have
            /// ```
            /// (q1 + q2) + q3 ~= q1 + (q2 + q3).
            /// ```
            /// Note that floating point quaternion addition cannot be exactly associative because arithmetic
            /// with floating point numbers is not associative.
            #[test]
            fn prop_quaternion_addition_almost_associative(
                q1 in super::$Generator::<$ScalarType>(), 
                q2 in super::$Generator::<$ScalarType>(), q3 in super::$Generator::<$ScalarType>()) {

                prop_assert!(relative_eq!((q1 + q2) + q3, q1 + (q2 + q3), epsilon = $tolerance));
            }
        }
    }
    }
}

approx_add_props!(quaternion_f64_add_props, f64, any_quaternion, 1e-7);


/// Generate the properties for quaternion arithmetic over exact scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers that compose the quaternions.
/// `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_add_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{Quaternion, Zero};

        proptest! {
            /// A quaternion plus a zero quaternion equals the same quaternion. 
            ///
            /// Given a quaternion `q`, it should satisfy
            /// ```
            /// q + 0 = q
            /// ```
            #[test]
            fn prop_quaternion_plus_zero_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(q + zero_quat, q);
            }

            /// A zero quaternion plus a quaternion equals the same quaternion.
            ///
            /// Given a quaternion `q`, it should satisfy
            /// ```
            /// 0 + q = q
            /// ```
            #[test]
            fn prop_zero_plus_quaternion_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(zero_quat + q, q);
            }

            /// Given quaternions `q1` and `q2`, we should be able to use `q1` and `q2` interchangeably 
            /// with their references `&q1` and `&q2` in arithmetic expressions involving quaternions.
            ///
            /// Given quaternions `q1` and `q2`, and their references `&q1` and `&q2`, they 
            /// should satisfy
            /// ```
            ///  q1 +  q2 = &q1 +  q2
            ///  q1 +  q2 =  q1 + &q2
            ///  q1 +  q2 = &q1 + &q2
            ///  q1 + &q2 = &q1 +  q2
            /// &q1 +  q2 =  q1 + &q2
            /// &q1 +  q2 = &q1 + &q2
            ///  q1 + &q2 = &q1 + &q2
            /// ```
            #[test]
            fn prop_quaternion1_plus_quaternion2_equals_refquaternion1_plus_refquaternion2(
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>()) {
                
                prop_assert_eq!( q1 +  q2, &q1 +  q2);
                prop_assert_eq!( q1 +  q2,  q1 + &q2);
                prop_assert_eq!( q1 +  q2, &q1 + &q2);
                prop_assert_eq!( q1 + &q2, &q1 +  q2);
                prop_assert_eq!(&q1 +  q2,  q1 + &q2);
                prop_assert_eq!(&q1 +  q2, &q1 + &q2);
                prop_assert_eq!( q1 + &q2, &q1 + &q2);
            }

            /// Quaternion addition over integer scalars should be commutative.
            ///
            /// Given quaternions `q1` and `q2`, we have
            /// ```
            /// q1 + q2 = q2 + q1.
            /// ```
            #[test]
            fn prop_quaternion_addition_commutative(
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>()) {
                
                let zero = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!((q1 + q2) - (q2 + q1), zero);
            }

            /// Given three quaternions of integer scalars, quaternion addition should be associative.
            ///
            /// Given quaternions `q1`, `q2`, and `q3`, we have
            /// ```
            /// (q1 + q2) + q3 = q1 + (q2 + q3)
            /// ```
            #[test]
            fn prop_quaternion_addition_associative(
                q1 in super::$Generator::<$ScalarType>(), 
                q2 in super::$Generator::<$ScalarType>(), q3 in super::$Generator::<$ScalarType>()) {

                prop_assert_eq!((q1 + q2) + q3, q1 + (q2 + q3));
            }
        }
    }
    }
}

exact_add_props!(quaternion_i32_add_props, i32, any_quaternion);
exact_add_props!(quaternion_u32_add_props, u32, any_quaternion);


/// Generate the properties for quaternion subtraction over floating point scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers that compose the quaternion.
/// `$Generator` is the name of a function or closure for generating examples.
/// `$tolerance` specifies the highest amount of acceptable error in the floating point computations
///  that still defines a correct computation. We cannot guarantee floating point computations
///  will be exact since the underlying floating point arithmetic is not exact.
///
/// We use approximate comparisons because arithmetic is not exact over finite precision floating point
/// scalar types.
macro_rules! approx_sub_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{Quaternion, Zero};

        proptest! {
            /// The zero quaternion over floating point scalars should act as an additive unit.
            ///
            /// Given a quaternion `q`, we have
            /// ```
            /// q - 0 = q
            /// ```
            #[test]
            fn prop_quaternion_minus_zero_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(q - zero_quat, q);
            }

            /// Every quaternion should have an additive inverse.
            ///
            /// Given a quaternion `q`, there is a quaternion `-q` such that
            /// ```
            /// q - q = q + (-q) = (-q) + q = 0
            /// ```
            #[test]
            fn prop_quaternion_minus_quaternion_equals_zero(q in super::$Generator::<$ScalarType>()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(q - q, zero_quat);
                prop_assert_eq!((-q) + q, zero_quat);
                prop_assert_eq!(q + (-q), zero_quat);
            }
        }
    }
    }
}

approx_sub_props!(quaternion_f64_sub_props, f64, any_quaternion, 1e-7);


/// Generate the properties for quaternion arithmetic over exact scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers that compose the quaternions.
/// `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_sub_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{Quaternion, Zero};

        proptest! {
            /// The zero quaternion should act as an additive unit. 
            ///
            /// Given a quaternion `q`, we have
            /// ```
            /// q - 0 = q
            /// ```
            #[test]
            fn prop_quaternion_minus_zero_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(q - zero_quat, q);
            }

            /// Every quaternion should have an additive inverse. 
            ///
            /// Given a quaternion `q`, there is a quaternion `-q` such that
            /// ```
            /// q - q = q + (-q) = (-q) + q = 0
            /// ```
            #[test]
            fn prop_quaternion_minus_quaternion_equals_zero(q in super::$Generator::<$ScalarType>()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(q - q, zero_quat);
            }
        }
    }
    }
}

exact_sub_props!(quaternion_i32_sub_props, i32, any_quaternion);
exact_sub_props!(quaternion_u32_sub_props, u32, any_quaternion);


/// Generate the properties for quaternion multiplication over floating point scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers that compose the quaternions.
/// `$Generator` is the name of a function or closure for generating examples.
/// `$tolerance` specifies the highest amount of acceptable error in the floating point computations
///  that still defines a correct computation. We cannot guarantee floating point computations
///  will be exact since the underlying floating point arithmetic is not exact.
///
/// We use approximate comparisons because arithmetic is not exact over finite precision floating point
/// scalar types.
macro_rules! approx_mul_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::approx::relative_eq;
        use gdmath::{Quaternion, One, Zero, Finite};

        proptest! {
            /// Multiplication of a scalar and a quaternion should be approximately commutative.
            ///
            /// Given a constant `c` and a quaternion `q`
            /// ```
            /// c * q ~= q * c
            /// ```
            /// Note that floating point quaternion multiplication cannot be commutative because 
            /// multiplication in the underlying floating point scalars is not commutative.
            #[test]
            fn prop_scalar_times_quaternion_equals_quaternion_times_scalar(
                c in any::<$ScalarType>(), q in super::$Generator::<$ScalarType>()) {
                
                prop_assume!(c.is_finite());
                prop_assume!(q.is_finite());
                prop_assert!(
                    relative_eq!(c * q, q * c, epsilon = $tolerance)
                );
            }

            /// Multiplication of two scalars and a quaternion should be compatible with multiplication of 
            /// all scalars. 
            ///
            /// In other words, scalar multiplication of two scalar with a quaternion should 
            /// act associatively, just like the multiplication of three scalars. 
            /// Given scalars `a` and `b`, and a quaternion `q`, we have
            /// ```
            /// (a * b) * q ~= a * (b * q)
            /// ```
            /// Note that the compatability of scalars with quaternions can only be approximate and not 
            /// exact because multiplication of the underlying scalars is not associative. 
            #[test]
            fn prop_scalar_multiplication_compatability(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), q in super::$Generator::<$ScalarType>()) {

                prop_assume!((a * (b * q)).is_finite());
                prop_assume!(((a * b) * q).is_finite());
                prop_assert!(relative_eq!(a * (b * q), (a * b) * q, epsilon = $tolerance));
            }

            /// Quaternion multiplication over floating point numbers is approximately associative.
            ///
            /// Given quaternions `q1`, `q2`, and `q3`, we have
            /// ```
            /// (q1 * q2) * q3 ~= q1 * (q2 * q3)
            /// ```
            /// Note that the quaternion multiplication can only be approximately associative and not 
            /// exactly associative because multiplication of the underlying scalars is not associative. 
            #[test]
            fn prop_quaternion_multiplication_associative(
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>(), 
                q3 in super::$Generator::<$ScalarType>()
            ) {
                prop_assume!((q1 * (q2 * q3)).is_finite());
                prop_assume!(((q1 * q2) * q3).is_finite());
                prop_assert!(relative_eq!(q1 * (q2 * q3), (q1 * q2) * q3, epsilon = $tolerance));
            }

            /// Quaternions have a multiplicative unit element.
            ///
            /// Given a quaternion `q`, and the unit quaternion `1`, we have
            /// ```
            /// q * 1 = 1 * q = q
            /// ```
            #[test]
            fn prop_quaternion_multiplicative_unit(q in super::$Generator::<$ScalarType>()) {
                let one = Quaternion::one();
                prop_assert_eq!(q * one, q);
                prop_assert_eq!(one * q, q);
                prop_assert_eq!(q * one, one * q);
            }

            /// Every nonzero quaternion over floating point scalars has an approximate multiplicative inverse.
            ///
            /// Given a quaternion `q` and its inverse `q_inv`, we have
            /// ```
            /// q * q_inv ~= q_inv * q ~= 1
            /// ```
            /// Note that quaternion algebra over floating point scalars is not commutative because
            /// multiplication of the underlying scalars is not commutative. As a result, we can only
            /// guarantee an appoximate equality.
            #[test]
            fn prop_quaternion_multiplicative_inverse(q in super::$Generator::<$ScalarType>()) {
                prop_assume!(q.is_finite());
                prop_assume!(q.is_invertible());

                let one = Quaternion::one();
                let q_inv = q.inverse().unwrap();
                prop_assert!(relative_eq!(q * q_inv, one, epsilon = $tolerance));
                prop_assert!(relative_eq!(q_inv * q, one, epsilon = $tolerance));
            }

            /// Quaternion multiplication transposes under inverion.
            ///
            /// Given two invertible quaternions `q1` and `q2`
            /// ```
            /// inverse(q1 * q2) = inverse(q2) * inverse(q1)
            /// ```
            /// Note that quaternion multiplication is noncommutative.
            #[test]
            fn prop_quaternion_inversion_involutive(
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>()) {

                prop_assume!(q1.is_finite());
                prop_assume!(q1.is_invertible());
                prop_assume!(q2.is_finite());
                prop_assume!(q2.is_invertible());
                prop_assume!((q1 * q2).is_finite());
                prop_assume!((q1 * q2).is_invertible());

                let q1_inv = q1.inverse().unwrap();
                let q2_inv = q2.inverse().unwrap();
                let q1_times_q2_inv = (q1 * q2).inverse().unwrap();

                prop_assert!(relative_eq!(q1_times_q2_inv, q2_inv * q1_inv, epsilon = $tolerance));
            }
        }
    }
    }
}

approx_mul_props!(quaternion_f64_mul_props, f64, any_quaternion, 1e-7);


/// Generate the properties for quaternion multiplication over exact scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers that compose the quaternions.
/// `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_mul_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{Quaternion, One};

        proptest! {
            /// Multiplication of an integer scalar and a quaternion over integer scalars should be commutative.
            ///
            /// Given a constant `c` and a quaternion `q`
            /// ```
            /// c * q = q * c
            /// ```
            #[test]
            fn prop_scalar_times_quaternion_equals_quaternion_times_scalar(
                c in any::<$ScalarType>(), q in super::$Generator::<$ScalarType>()) {
                
                prop_assert_eq!(c * q, q * c);
            }

            /// Exact multiplication of two scalars and a quaternion should be compatible with multiplication of 
            /// all scalars. 
            ///
            /// In other words, scalar multiplication of two scalars with a quaternion should 
            /// act associatively just like the multiplication of three scalars. 
            /// Given scalars `a` and `b`, and a quaternion `q`, we have
            /// ```
            /// (a * b) * q = a * (b * q)
            /// ```
            #[test]
            fn prop_scalar_multiplication_compatability(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), q in super::$Generator::<$ScalarType>()) {

                prop_assert_eq!(a * (b * q), (a * b) * q);
            }

            /// Quaternion multiplication over integer scalars is exactly associative.
            ///
            /// Given quaternions `q1`, `q2`, and `q3`, we have
            /// ```
            /// (q1 * q2) * q3 = q1 * (q2 * q3)
            /// ```
            #[test]
            fn prop_quaternion_multiplication_associative(
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>(), 
                q3 in super::$Generator::<$ScalarType>()
            ) {
                prop_assert_eq!(q1 * (q2 * q3), (q1 * q2) * q3);
            }

            /// Quaternions have a multiplicative unit element.
            ///
            /// Given a quaternion `q`, and the unit quaternion `1`, we have
            /// ```
            /// q * 1 = 1 * q = q
            /// ```
            #[test]
            fn prop_quaternion_multiplicative_unit(q in super::$Generator::<$ScalarType>()) {
                let one = Quaternion::one();
                prop_assert_eq!(q * one, q);
                prop_assert_eq!(one * q, q);
                prop_assert_eq!(q * one, one * q);
            }
        }
    }
    }
}

exact_mul_props!(quaternion_i32_mul_props, i32, any_quaternion);
exact_mul_props!(quaternion_u32_mul_props, u32, any_quaternion);


/// Generate the properties for quaternion distribution over floating point scalars.
///
/// Here are what the different macro parameters mean:
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers that compose the quaternions.
/// `$Generator` is the name of a function or closure for generating examples.
/// `$tolerance` specifies the highest amount of acceptable error in the floating point computations
///  that still defines a correct computation. We cannot guarantee floating point computations
///  will be exact since the underlying floating point arithmetic is not exact.
///
/// We use approximate comparisons because arithmetic is not exact over finite precision floating point
/// scalar types.
macro_rules! approx_distributive_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::Finite;
        use gdmath::approx::relative_eq;
    
        proptest! {
            /// Scalar multiplication should approximately distribute over quaternion addition.
            ///
            /// Given a scalar `a` and quaternions `q1` and `q2`
            /// ```
            /// a * (q1 + q2) ~= a * q1 + a * q2
            /// ```
            #[test]
            fn prop_distribution_over_quaternion_addition(
                a in any::<$ScalarType>(), 
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>()) {
                
                prop_assume!((a * (q1 + q2)).is_finite());
                prop_assume!((a * q1 + a * q2).is_finite());
                prop_assert!(relative_eq!(a * (q1 + q2), a * q1 + a * q2, epsilon = $tolerance));
            }
    
            /// Multiplication of a sum of scalars should approximately distribute over a quaternion.
            ///
            /// Given scalars `a` and `b` and a quaternion `q`, we have
            /// ```
            /// (a + b) * q ~= a * q + b * q
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                q in super::$Generator::<$ScalarType>()) {
    
                prop_assume!(((a + b) * q).is_finite());
                prop_assume!((a * q + b * q).is_finite());
                prop_assert!(relative_eq!((a + b) * q, a * q + b * q, epsilon = $tolerance));
            }

            /// Multiplication of two quaternions by a scalar on the right should approximately distribute.
            ///
            /// Given quaternions `q1` and `q2` and a scalar `a`
            /// ```
            /// (q1 + q2) * a ~= q1 * a + q2 * a
            /// ```
            #[test]
            fn prop_distribution_over_quaternion_addition1(
                a in any::<$ScalarType>(), 
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>()) {
                    
                prop_assume!(((q1 + q2) * a).is_finite());
                prop_assume!((q1 * a + q2 * a).is_finite());
                prop_assert!(relative_eq!((q1 + q2) * a,  q1 * a + q2 * a, epsilon = $tolerance));
            }

            /// Multiplication of a quaternion on the right by the sum of two scalars should approximately 
            /// distribute over the two scalars.
            ///
            /// Given a quaternion `q` and scalars `a` and `b`
            /// ```
            /// q * (a + b) ~= q * a + q * b
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition1(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                q in super::$Generator::<$ScalarType>()) {
    
                prop_assume!((q * (a + b)).is_finite());
                prop_assume!((q * a + q * b).is_finite());
                prop_assert!(relative_eq!(q * (a + b), q * a + q * b, epsilon = $tolerance));
            }

            /// Quaternion multiplication over floating point numbers should be 
            /// approximately distributive on the right.
            ///
            /// Given three quaternions `q1`, `q2`, and `q3`
            /// ```
            /// (q1 + q2) * q3 ~= q1 * q3 + q2 * q3
            /// ```
            #[test]
            fn prop_quaternion_multiplication_right_distributive(
                q1 in super::$Generator::<$ScalarType>(), 
                q2 in super::$Generator::<$ScalarType>(), q3 in super::$Generator::<$ScalarType>()
            ) {
                prop_assume!(((q1 + q2) * q3).is_finite());
                prop_assume!((q1 * q3 + q2 * q3).is_finite());
                prop_assert!(relative_eq!((q1 + q2) * q3, q1 * q3 + q2 * q3, epsilon = $tolerance));
            }

            /// Quaternion multiplication over floating point numbers should be approximately 
            /// distributive on the left.
            ///
            /// Given three quaternions `q1`, `q2`, and `q3`
            /// ```
            /// q1 * (q2 + q3) ~= q1 * q2 + q1 * q3
            /// ```
            #[test]
            fn prop_quaternion_multiplication_left_distributive(
                q1 in super::$Generator::<$ScalarType>(), 
                q2 in super::$Generator::<$ScalarType>(), q3 in super::$Generator::<$ScalarType>()
            ) {
                prop_assume!(((q1 * (q2 + q3)).is_finite()));
                prop_assume!((q1 * q2 + q1 * q3).is_finite());
                prop_assert!(relative_eq!(q1 * (q2 + q3), q1 * q2 + q1 * q3, epsilon = $tolerance));
            }
        }
    }
    }    
}

approx_distributive_props!(quaternion_f64_distributive_props, f64, any_quaternion, 1e-7);


/// Generate the properties for quaternion distribution over exact scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers that compose the quaternions.
/// `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_distributive_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;

        proptest! {
            /// Scalar multiplication should distribute over quaternion addition.
            ///
            /// Given a scalar `a` and quaternions `q1` and `q2`
            /// ```
            /// a * (q1 + q2) = a * q1 + a * q2
            /// ```
            #[test]
            fn prop_distribution_over_quaternion_addition(
                a in any::<$ScalarType>(), 
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>()) {
                
                prop_assert_eq!(a * (q1 + q2), a * q1 + a * q2);
                prop_assert_eq!((q1 + q2) * a,  q1 * a + q2 * a);
            }

            /// Multiplication of a sum of scalars should distribute over a quaternion.
            ///
            /// Given scalars `a` and `b` and a quaternion `q`, we have
            /// ```
            /// (a + b) * q = a * q + b * q
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                q in super::$Generator::<$ScalarType>()) {
    
                prop_assert_eq!((a + b) * q, a * q + b * q);
                prop_assert_eq!(q * (a + b), q * a + q * b);
            }

            /// Multiplication of two quaternions by a scalar on the right should distribute.
            ///
            /// Given quaternions `q1` and `q2` and a scalar `a`
            /// ```
            /// (q1 + q2) * a = q1 * a + q2 * a
            /// ```
            #[test]
            fn prop_distribution_over_quaternion_addition1(
                a in any::<$ScalarType>(), 
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>()) {
                    
                prop_assert_eq!((q1 + q2) * a,  q1 * a + q2 * a);
            }

            /// Multiplication of a quaternion on the right by the sum of two scalars should
            /// distribute over the two scalars. 
            ///
            /// Given a quaternion `q` and scalars `a` and `b`
            /// ```
            /// q * (a + b) = q * a + q * b
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition1(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                q in super::$Generator::<$ScalarType>()) {
    
                prop_assert_eq!(q * (a + b), q * a + q * b);
            }

            /// Quaternion multiplication should be distributive on the right.
            ///
            /// Given three quaternions `q1`, `q2`, and `q3`
            /// ```
            /// (q1 + q2) * q3 = q1 * q3 + q2 * q3
            /// ```
            #[test]
            fn prop_quaternion_multiplication_right_distributive(
                q1 in super::$Generator::<$ScalarType>(), 
                q2 in super::$Generator::<$ScalarType>(), q3 in super::$Generator::<$ScalarType>()
            ) {
                prop_assert_eq!((q1 + q2) * q3, q1 * q3 + q2 * q3);
            }

            /// Quaternion multiplication should be distributive on the left.
            ///
            /// Given three quaternions `q1`, `q2`, and `q3`
            /// ```
            /// q1 * (q2 + q3) = q1 * q2 + q1 * q3
            /// ```
            #[test]
            fn prop_quaternion_multiplication_left_distributive(
                q1 in super::$Generator::<$ScalarType>(), 
                q2 in super::$Generator::<$ScalarType>(), q3 in super::$Generator::<$ScalarType>()
            ) {
                prop_assert_eq!((q1 + q2) * q3, q1 * q3 + q2 * q3);
            }
        }
    }
    }    
}

exact_distributive_props!(quaternion_i32_distributive_props, i32, any_quaternion);
exact_distributive_props!(quaternion_u32_distributive_props, u32, any_quaternion);

