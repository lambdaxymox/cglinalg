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
            /// Given a quaternion `q`, it should return the element at position `index` in the quaternion 
            /// when the given index is inbounds.
            #[test]
            fn prop_accepts_all_indices_in_of_bounds(
                q in super::$Generator::<$ScalarType>(), index in 0..$UpperBound as usize) {

                prop_assert_eq!(q[index], q[index]);
            }
    
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
            /// A scalar zero times a quaternion should be zero. That is, quaternion algebra satisfies
            /// ```
            /// For each quaternion q, 0 * q = 0.
            /// ```
            #[test]
            fn prop_zero_times_quaternion_equals_zero(q in super::$Generator()) {
                let zero: $ScalarType = num_traits::Zero::zero();
                let zero_quat = Quaternion::zero();
                prop_assert_eq!(zero * q, zero_quat);
            }
        
            /// A scalar zero times a quaternion should be zero. That is, quaternion algebra satisfies
            /// ```
            /// For each quaternion q, q * 0 = 0.
            /// ```
            #[test]
            fn prop_quaternion_times_zero_equals_zero(q in super::$Generator()) {
                let zero: $ScalarType = num_traits::Zero::zero();
                let zero_quat = Quaternion::zero();
                prop_assert_eq!(q * zero, zero_quat);
            }

            /// A zero quaternion should act as the additive unit element of a quaternion algebra.
            /// In particular, we have
            /// ```
            /// For every quaternion q, q + 0 = q.
            /// ```
            #[test]
            fn prop_quaternion_plus_zero_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(q + zero_quat, q);
            }

            /// A zero quaternion should act as the additive unit element of a set of quaternions.
            /// In particular, we have
            /// ```
            /// For every quaternion q, 0 + q = q.
            /// ```
            #[test]
            fn prop_zero_plus_quaternion_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(zero_quat + q, q);
            }

            /// Multiplying a quaternion by one should give the original quaternion.
            /// In particular, we have
            /// ```
            /// For every quaternion q, 1 * q = q.
            /// ```
            #[test]
            fn prop_one_times_quaternion_equal_quaternion(q in super::$Generator()) {
                let one: $ScalarType = num_traits::One::one();
                prop_assert_eq!(one * q, q);
            }

            /// Multiplying a quaternion by one should give the original quaternion.
            /// In particular, we have
            /// ```
            /// For every quaternion q, q * 1 = q.
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
            /// A quaternion plus a zero quaternion equals the same quaternion. The quaternion algebra satisfies
            /// the following: given a quaternion `q`
            /// ```
            /// q + 0 = q
            /// ```
            #[test]
            fn prop_quaternion_plus_zero_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(q + zero_quat, q);
            }

            /// A quaternion plus a zero quaternion equals the same quaternion. The quaternion algebra satisfies
            /// the following: Given a quaternion `q`
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
            /// In the case of quaternion addition, the quaternions should satisfy
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

            /// Given two quaternions of floating point scalars, quaternion addition should  be approximately
            /// commutative. Given quaternions `q1` and `q2`, we have
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

            /// Given three quaternions of floating point scalars, quaternion addition should be approximately
            /// associative. Given quaternions `q1`, `q2`, and `q3` we have
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
            /// A quaternion plus a zero quaternion equals the same quaternion. The quaternion algebra satisfies
            /// the following: Given a quaternion `q`
            /// ```
            /// q + 0 = q
            /// ```
            #[test]
            fn prop_quaternion_plus_zero_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(q + zero_quat, q);
            }

            /// A zero quaternion plus a quaternion equals the same quaternion. The quaternion algebra satisfies
            /// the following: Given a quaternion `q`
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
            /// In the case of quaternion addition, the quaternions should satisfy
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

            /// Given two quaternions of integer scalars, quaternion addition should be
            /// commutative. Given quaternions `q1` and `q2`, we have
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
            /// The zero quaternion over of floating point scalars should act as an additive unit. 
            /// That is, given a quaternion `q`, we have
            /// ```
            /// q - 0 = q
            /// ```
            #[test]
            fn prop_quaternion_minus_zero_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(q - zero_quat, q);
            }

            /// Every quaternion should have an additive inverse. That is, given a quaternion `q`,
            /// there is a quaternion `-q` such that
            /// ```
            /// q - q = q + (-q) = 0
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
            /// The zero quaternion should act as an additive unit. That is, given a quaternion `q`,
            /// we have
            /// ```
            /// q - 0 = q
            /// ```
            #[test]
            fn prop_quaternion_minus_zero_equals_quaternion(q in super::$Generator()) {
                let zero_quat = Quaternion::<$ScalarType>::zero();
                prop_assert_eq!(q - zero_quat, q);
            }

            /// Every quaternion should have an additive inverse. That is, given a quaternion `q`,
            /// there is a quaternion `-q` such that
            /// we have
            /// ```
            /// q - q = q + (-q) = 0
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
        use gdmath::Magnitude;
        use gdmath::approx::relative_eq;

        proptest! {
            /// Multiplication of a scalar and a quaternion should be approximately commutative.
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
                prop_assume!(q.magnitude().is_finite());
                prop_assert!(
                    relative_eq!(c * q, q * c, epsilon = $tolerance)
                );
            }

            /// Multiplication of two scalars and a quaternion should be compatible with multiplication of 
            /// all scalars. In other words, scalar multiplication of two scalar with a quaternion should 
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

                prop_assert!(relative_eq!(a * (b * q), (a * b) * q, epsilon = $tolerance));
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

        proptest! {
            /// Exact multiplication of a scalar and a quaternion should be commutative.
            /// Given a constant `c` and a quaternion `q`
            /// ```
            /// c * q = q * c
            /// ```
            /// We deviate from the usual formalisms of quaternion algebra in that we 
            /// allow the ability to multiply scalars from the left, or from the right of a quaternion.
            #[test]
            fn prop_scalar_times_quaternion_equals_quaternion_times_scalar(
                c in any::<$ScalarType>(), q in super::$Generator::<$ScalarType>()) {
                
                prop_assert_eq!(c * q, q * c);
            }

            /// Exact multiplication of two scalars and a quaternion should be compatible with multiplication of 
            /// all scalars. In other words, scalar multiplication of two scalars with a quaternion should 
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
        }
    }
    }
}

exact_mul_props!(quaternion_i32_mul_props, i32, any_quaternion);
exact_mul_props!(quaternion_u32_mul_props, u32, any_quaternion);


/// Generate the properties for quaternion distribution over floating point scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers that compose the quaternions.
/// `$Generator` is the name of a function or closure for generating examples.
///
/// We use approximate comparisons because arithmetic is not exact over finite precision floating point
/// scalar types.
macro_rules! approx_distributive_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::Magnitude;
    
        proptest! {
            /// Scalar multiplication should approximately distribute over quaternion addition.
            /// Given a scalar `a` and quaternions `q1` and `q2`
            /// ```
            /// a * (q1 + q2) ~= a * q1 + a * q2
            /// ```
            #[test]
            fn prop_distribution_over_quaternion_addition(
                a in any::<$ScalarType>(), 
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>()) {
                
                prop_assume!((a * (q1 + q2)).magnitude().is_finite());
                prop_assume!((a * q1 + a * q2).magnitude().is_finite());
                prop_assert_eq!(a * (q1 + q2), a * q1 + a * q2);
            }
    
            /// Multiplication of a sum of scalars should approximately distribute over a quaternion.
            /// Given scalars `a` and `b` and a quaternion `q`, we have
            /// ```
            /// (a + b) * q ~= a * q + b * q
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                q in super::$Generator::<$ScalarType>()) {
    
                prop_assume!(((a + b) * q).magnitude().is_finite());
                prop_assume!((a * q + b * q).magnitude().is_finite());
                prop_assert_eq!((a + b) * q, a * q + b * q);
            }

            /// Multiplication of two quaternions by a scalar on the right should approximately distribute.
            /// Given quaternions `q1` and `q2` and a scalar `a`
            /// ```
            /// (q1 + q2) * a ~= q1 * a + q2 * a
            /// ```
            #[test]
            fn prop_distribution_over_quaternion_addition1(
                a in any::<$ScalarType>(), 
                q1 in super::$Generator::<$ScalarType>(), q2 in super::$Generator::<$ScalarType>()) {
                    
                prop_assume!(((q1 + q2) * a).magnitude().is_finite());
                prop_assume!((q1 * a + q2 * a).magnitude().is_finite());
                prop_assert_eq!((q1 + q2) * a,  q1 * a + q2 * a);
            }

            /// Multiplication of a quaternion on the right by the sum of two scalars should approximately 
            /// distribute over the two scalars. 
            /// Given a quaternion `q` and scalars `a` and `b`
            /// ```
            /// q * (a + b) ~= q * a + q * b
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition1(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                q in super::$Generator::<$ScalarType>()) {
    
                prop_assume!((q * (a + b)).magnitude().is_finite());
                prop_assume!((q * a + q * b).magnitude().is_finite());
                prop_assert_eq!(q * (a + b), q * a + q * b);
            }
        }
    }
    }    
}

approx_distributive_props!(quaternion_f64_distributive_props, f64, any_quaternion);


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
            /// distribute over the two scalars. Given a quaternion `q` and scalars `a` and `b`
            /// ```
            /// q * (a + b) = q * a + q * b
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition1(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                q in super::$Generator::<$ScalarType>()) {
    
                prop_assert_eq!(q * (a + b), q * a + q * b);
            }
        }
    }
    }    
}

exact_distributive_props!(quaternion_i32_distributive_props, i32, any_quaternion);
exact_distributive_props!(quaternion_u32_distributive_props, u32, any_quaternion);

