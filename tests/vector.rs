extern crate cgmath;
extern crate num_traits;
extern crate proptest;

use proptest::prelude::*;
use cgmath::{
    Vector1, 
    Vector2, 
    Vector3, 
    Vector4, 
    Scalar,
    ScalarFloat,
};



/// A macro that generates the property tests for vector indexing.
/// `$VectorN` denotes the name of the vector type.
/// `$FieldType` denotes the underlying system of numbers that we access using indexing.
/// `$UpperBound` denotes the upperbound on the range of acceptable indexes.
/// `$TestModuleName` is a name we give to the module we place the tests in to separate them
/// from each other for each field type to prevent namespace collisions.
macro_rules! index_props {
    ($VectorN:ident, $FieldType:ty, $UpperBound:expr, $TestModuleName:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cgmath::{$VectorN, Zero};

        proptest! {
            /// Given a vector of type `$VectorN`, it should return the entry at position `index` in the vector 
            /// when the given index is inbounds.
            #[test]
            fn prop_accepts_all_indices_in_of_bounds(index in 0..$UpperBound as usize) {
                let v: $VectorN<$FieldType> = $VectorN::zero();
                prop_assert_eq!(v[index], v[index]);
            }
    
            /// Given a vector of type `$VectorN`, when the entry position is out of bounds, it should 
            /// generate a panic just like an array or vector indexed out of bounds.
            #[test]
            #[should_panic]
            fn prop_panics_when_index_out_of_bounds(index in $UpperBound..usize::MAX) {
                let v: $VectorN<$FieldType> = $VectorN::zero();
                prop_assert_eq!(v[index], v[index]);
            }
        }
    }
    }
}

index_props!(Vector1, f64, 1, vector1_f64_index_props);
index_props!(Vector2, f64, 2, vector2_f64_index_props);
index_props!(Vector3, f64, 3, vector3_f64_index_props);
index_props!(Vector4, f64, 4, vector4_f64_index_props);


fn any_vector1<S>() -> impl Strategy<Value = Vector1<S>> where S: Scalar + Arbitrary {
    any::<S>().prop_map(|x| Vector1::new(x))
}

fn any_vector1_no_overflow<S>() -> impl Strategy<Value = Vector1<S>> where S: Scalar + num_traits::One + Arbitrary {
    any::<S>().prop_map(|x| { 
        let two = <S as num_traits::One>::one() + <S as num_traits::One>::one();
        Vector1::new(x / two)
    })
}

fn any_vector2<S>() -> impl Strategy<Value = Vector2<S>> where S: Scalar + Arbitrary {
    any::<(S, S)>().prop_map(|(x, y)| Vector2::new(x, y))
}

fn any_vector2_no_overflow<S>() -> impl Strategy<Value = Vector2<S>> where S: Scalar + num_traits::One + Arbitrary {
    any::<(S, S)>().prop_map(|(x, y)| { 
        let two = <S as num_traits::One>::one() + <S as num_traits::One>::one();
        Vector2::new(x / two, y / two) 
    })
}

fn any_vector3<S>() -> impl Strategy<Value = Vector3<S>> where S: Scalar + Arbitrary {
    any::<(S, S, S)>().prop_map(|(x, y, z)| Vector3::new(x, y, z))
}

fn any_vector3_no_overflow<S>() -> impl Strategy<Value = Vector3<S>> where S: Scalar + num_traits::One + Arbitrary {
    any::<(S, S, S)>().prop_map(|(x, y, z)| { 
        let two = <S as num_traits::One>::one() + <S as num_traits::One>::one();
        Vector3::new(x / two, y / two, z / two)
    })
}

fn any_vector4<S>() -> impl Strategy<Value = Vector4<S>> where S: Scalar + Arbitrary {
    any::<(S, S, S, S)>().prop_map(|(x, y, z, w)| Vector4::new(x, y, z, w))
}

fn any_vector4_no_overflow<S>() -> impl Strategy<Value = Vector4<S>> where S: Scalar + num_traits::One + Arbitrary {
    any::<(S, S, S, S)>().prop_map(|(x, y, z, w)| {
        let two = <S as num_traits::One>::one() + <S as num_traits::One>::one();
        Vector4::new(x / two, y / two, z / two, w / two)
    })
}

macro_rules! vector_arithmetic_props {
    ($VectorN:ident, $FieldType:ty, $Generator:ident, $TestModuleName:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cgmath::{$VectorN, Zero};

        proptest! {
            #[test]
            fn prop_zero_times_vector_equals_zero(v in super::$Generator()) {
                let zero: $FieldType = num_traits::Zero::zero();
                let zero_vec = $VectorN::zero();
                prop_assert_eq!(zero * v, zero_vec);
            }
        
            #[test]
            fn prop_vector_times_zero_equals_zero(v in super::$Generator()) {
                let zero: $FieldType = num_traits::Zero::zero();
                let zero_vec = $VectorN::zero();
                prop_assert_eq!(v * zero, zero_vec);
            }

            #[test]
            fn prop_vector_plus_zero_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$FieldType>::zero();
                prop_assert_eq!(v + zero_vec, v);
            }

            #[test]
            fn prop_zero_plus_vector_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$FieldType>::zero();
                prop_assert_eq!(zero_vec + v, v);
            }

            #[test]
            fn prop_one_times_vector_equal_vector(v in super::$Generator()) {
                let one: $FieldType = num_traits::One::one();
                prop_assert_eq!(one * v, v);
            }
        }
    }
    }
}

macro_rules! vector_add_props {
    ($VectorN:ident, $FieldType:ty, $Generator:ident, $TestModuleName:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cgmath::{$VectorN, Zero};

        proptest! {
            #[test]
            fn prop_vector_plus_zero_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$FieldType>::zero();
                prop_assert_eq!(v + zero_vec, v);
            }

            #[test]
            fn prop_zero_plus_vector_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$FieldType>::zero();
                prop_assert_eq!(zero_vec + v, v);
            }

            #[test]
            fn prop_vector1_plus_vector2_equals_refvector1_plus_refvector2(
                v1 in super::$Generator::<$FieldType>(), v2 in super::$Generator::<$FieldType>()) {
                
                prop_assert_eq!(v1 + v2, &v1 + v2);
                prop_assert_eq!(v1 + v2, v1 + &v2);
                prop_assert_eq!(v1 + v2, &v1 + &v2);
                prop_assert_eq!(v1 + &v2, &v1 + v2);
                prop_assert_eq!(&v1 + v2, v1 + &v2);
                prop_assert_eq!(&v1 + v2, &v1 + &v2);
                prop_assert_eq!(v1 + &v2, &v1 + &v2);
            }

            #[test]
            fn prop_vector_addition_almost_commutative(
                v1 in super::$Generator::<$FieldType>(), v2 in super::$Generator::<$FieldType>()) {
                
                let zero: $VectorN<$FieldType> = Zero::zero();
                prop_assert_eq!((v1 + v2) - (v2 + v1), zero);
            }
        }
    }
    }
}

macro_rules! vector_sub_props {
    ($VectorN:ident, $FieldType:ty, $Generator:ident, $TestModuleName:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cgmath::{$VectorN, Zero};

        proptest! {
            #[test]
            fn prop_vector_minus_zero_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$FieldType>::zero();
                prop_assert_eq!(v - zero_vec, v);
            }

            #[test]
            fn prop_vector_minus_vector_equals_zero(v in super::$Generator::<$FieldType>()) {
                let zero_vec = $VectorN::<$FieldType>::zero();
                prop_assert_eq!(v - v, zero_vec);
            }
        }
    }
    }
}

vector_arithmetic_props!(Vector1, f64, any_vector1, vector1_f64_arithmetic_props);
vector_arithmetic_props!(Vector1, u32, any_vector1, vector1_u32_arithmetic_props);
vector_arithmetic_props!(Vector1, i32, any_vector1, vector1_i32_arithmetic_props);

vector_arithmetic_props!(Vector2, f64, any_vector2, vector2_f64_arithmetic_props);
vector_arithmetic_props!(Vector2, u32, any_vector2, vector2_u32_arithmetic_props);
vector_arithmetic_props!(Vector2, i32, any_vector2, vector2_i32_arithmetic_props);

vector_arithmetic_props!(Vector3, f64, any_vector3, vector3_f64_arithmetic_props);
vector_arithmetic_props!(Vector3, u32, any_vector3, vector3_u32_arithmetic_props);
vector_arithmetic_props!(Vector3, i32, any_vector3, vector3_i32_arithmetic_props);

vector_arithmetic_props!(Vector4, f64, any_vector4, vector4_f64_arithmetic_props);
vector_arithmetic_props!(Vector4, u32, any_vector4, vector4_u32_arithmetic_props);
vector_arithmetic_props!(Vector4, i32, any_vector4, vector4_i32_arithmetic_props);



vector_add_props!(Vector1, f64, any_vector1_no_overflow, vector1_f64_add_props);
vector_add_props!(Vector1, u32, any_vector1_no_overflow, vector1_u32_add_props);
vector_add_props!(Vector1, i32, any_vector1_no_overflow, vector1_i32_add_props);

vector_add_props!(Vector2, f64, any_vector2_no_overflow, vector2_f64_add_props);
vector_add_props!(Vector2, u32, any_vector2_no_overflow, vector2_u32_add_props);
vector_add_props!(Vector2, i32, any_vector2_no_overflow, vector2_i32_add_props);

vector_add_props!(Vector3, f64, any_vector3_no_overflow, vector3_f64_add_props);
vector_add_props!(Vector3, u32, any_vector3_no_overflow, vector3_u32_add_props);
vector_add_props!(Vector3, i32, any_vector3_no_overflow, vector3_i32_add_props);

vector_add_props!(Vector4, f64, any_vector4_no_overflow, vector4_f64_add_props);
vector_add_props!(Vector4, u32, any_vector4_no_overflow, vector4_u32_add_props);
vector_add_props!(Vector4, i32, any_vector4_no_overflow, vector4_i32_add_props);


vector_sub_props!(Vector1, f64, any_vector1_no_overflow, vector1_f64_sub_props);
vector_sub_props!(Vector1, u32, any_vector1_no_overflow, vector1_u32_sub_props);
vector_sub_props!(Vector1, i32, any_vector1_no_overflow, vector1_i32_sub_props);

vector_sub_props!(Vector2, f64, any_vector2_no_overflow, vector2_f64_sub_props);
vector_sub_props!(Vector2, u32, any_vector2_no_overflow, vector2_u32_sub_props);
vector_sub_props!(Vector2, i32, any_vector2_no_overflow, vector2_i32_sub_props);

vector_sub_props!(Vector3, f64, any_vector3_no_overflow, vector3_f64_sub_props);
vector_sub_props!(Vector3, u32, any_vector3_no_overflow, vector3_u32_sub_props);
vector_sub_props!(Vector3, i32, any_vector3_no_overflow, vector3_i32_sub_props);

vector_sub_props!(Vector4, f64, any_vector4_no_overflow, vector4_f64_sub_props);
vector_sub_props!(Vector4, u32, any_vector4_no_overflow, vector4_u32_sub_props);
vector_sub_props!(Vector4, i32, any_vector4_no_overflow, vector4_i32_sub_props);


macro_rules! vector_magnitude_props {
    ($TestModuleName:ident, $VectorN:ident, $FieldType:ty, $Generator:ident, $epsilon:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn prop_magnitude_preserves_scale(
                v in super::$Generator::<$FieldType>(), c in any::<$FieldType>()) {
                use cgmath::approx::relative_eq;
                use cgmath::{$VectorN, Magnitude, Zero};
                
                let abs_c = <$FieldType as num_traits::Float>::abs(c);                
                prop_assume!((abs_c * v.magnitude()).is_finite());
                prop_assume!((c * v).magnitude().is_finite());
                prop_assert!(
                    relative_eq!( (c * v).magnitude(), abs_c * v.magnitude(), epsilon = $epsilon),
                    "\n||c * v|| = {}\n|c| * ||v|| = {}\n", (c * v).magnitude(), abs_c * v.magnitude(),
                );
            }

            #[test]
            fn prop_magnitude_nonnegative(v in super::$Generator::<$FieldType>()) {
                use cgmath::{$VectorN, Magnitude, Zero};
                let zero = <$FieldType as num_traits::Zero>::zero();
                prop_assert!(v.magnitude() >= zero);
            }

            #[test]
            fn prop_magnitude_satisfies_triangle_inequality(
                v in super::$Generator::<$FieldType>(), w in super::$Generator::<$FieldType>()) {
            
                use cgmath::{$VectorN, Magnitude, Zero};
                prop_assume!((v + w).magnitude().is_finite());
                prop_assume!(v.magnitude().is_finite());
                prop_assume!(w.magnitude().is_finite());
                prop_assume!((v.magnitude() + w.magnitude()).is_finite());
                prop_assert!((v + w).magnitude() <= v.magnitude() + w.magnitude(), 
                    "\n|v + w| = {}\n|v| = {}\n|w| = {}\n|v| + |w| = {}\n",
                    (v + w).magnitude(), v.magnitude(), w.magnitude(), v.magnitude() + w.magnitude()
                );
            }
        }
    }
    }
}


vector_magnitude_props!(vector1_f64_magnitude_props, Vector1, f64, any_vector1, 1e-7);
vector_magnitude_props!(vector2_f64_magnitude_props, Vector2, f64, any_vector2, 1e-7);
vector_magnitude_props!(vector3_f64_magnitude_props, Vector3, f64, any_vector3, 1e-7);
vector_magnitude_props!(vector4_f64_magnitude_props, Vector4, f64, any_vector4, 1e-7);


macro_rules! vector_mul_props {
    ($VectorN:ident, $FieldType:ty, $Generator:ident, $TestModuleName:ident, $epsilon:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cgmath::{$VectorN, Magnitude, Zero};

        proptest! {
            #[test]
            fn prop_scalar_times_vector_equals_vector_times_scalar(
                c in any::<$FieldType>(), v in super::$Generator::<$FieldType>()) {
                
                use cgmath::approx::relative_eq;
                prop_assume!(c.is_finite());
                prop_assume!(v.magnitude().is_finite());
                prop_assert!(
                    relative_eq!(c * v, v * c, epsilon = $epsilon)
                );
            }
        }
    }
    }
}

vector_mul_props!(Vector1, f64, any_vector1, vector1_f64_mul_props, 1e-7);
vector_mul_props!(Vector2, f64, any_vector2, vector2_f64_mul_props, 1e-7);
vector_mul_props!(Vector3, f64, any_vector3, vector3_f64_mul_props, 1e-7);
vector_mul_props!(Vector4, f64, any_vector4, vector4_f64_mul_props, 1e-7);
