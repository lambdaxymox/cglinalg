extern crate cgmath;
extern crate num_traits;
extern crate proptest;

use proptest::prelude::*;
use cgmath::{Vector1, Vector2, Vector3, Vector4, Zero, Scalar};

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

index_props!(Vector1, f32, 1, vector1_f32_index_props);
index_props!(Vector1, f64, 1, vector1_f64_index_props);
index_props!(Vector2, f32, 2, vector2_f32_index_props);
index_props!(Vector2, f64, 2, vector2_f64_index_props);
index_props!(Vector3, f32, 3, vector3_f32_index_props);
index_props!(Vector3, f64, 3, vector3_f64_index_props);
index_props!(Vector4, f32, 4, vector4_f32_index_props);
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
        }
    }
    }
}

vector_arithmetic_props!(Vector1, f32, any_vector1, vector1_f32_arithmetic_props);
vector_arithmetic_props!(Vector1, f64, any_vector1, vector1_f64_arithmetic_props);
vector_arithmetic_props!(Vector1, u8, any_vector1, vector1_u8_arithmetic_props);
vector_arithmetic_props!(Vector1, u16, any_vector1, vector1_u16_arithmetic_props);
vector_arithmetic_props!(Vector1, u32, any_vector1, vector1_u32_arithmetic_props);
vector_arithmetic_props!(Vector1, u64, any_vector1, vector1_u64_arithmetic_props);
vector_arithmetic_props!(Vector1, u128, any_vector1, vector1_u128_arithmetic_props);
vector_arithmetic_props!(Vector1, usize, any_vector1, vector1_usize_arithmetic_props);
vector_arithmetic_props!(Vector1, i8, any_vector1, vector1_i8_arithmetic_props);
vector_arithmetic_props!(Vector1, i16, any_vector1, vector1_i16_arithmetic_props);
vector_arithmetic_props!(Vector1, i32, any_vector1, vector1_i32_arithmetic_props);
vector_arithmetic_props!(Vector1, i64, any_vector1, vector1_i64_arithmetic_props);
vector_arithmetic_props!(Vector1, i128, any_vector1, vector1_i128_arithmetic_props);
vector_arithmetic_props!(Vector1, isize, any_vector1, vector1_isize_arithmetic_props);

vector_arithmetic_props!(Vector2, f32, any_vector2, vector2_f32_arithmetic_props);
vector_arithmetic_props!(Vector2, f64, any_vector2, vector2_f64_arithmetic_props);
vector_arithmetic_props!(Vector2, u8, any_vector2, vector2_u8_arithmetic_props);
vector_arithmetic_props!(Vector2, u16, any_vector2, vector2_u16_arithmetic_props);
vector_arithmetic_props!(Vector2, u32, any_vector2, vector2_u32_arithmetic_props);
vector_arithmetic_props!(Vector2, u64, any_vector2, vector2_u64_arithmetic_props);
vector_arithmetic_props!(Vector2, u128, any_vector2, vector2_u128_arithmetic_props);
vector_arithmetic_props!(Vector2, usize, any_vector2, vector2_usize_arithmetic_props);
vector_arithmetic_props!(Vector2, i8, any_vector2, vector2_i8_arithmetic_props);
vector_arithmetic_props!(Vector2, i16, any_vector2, vector2_i16_arithmetic_props);
vector_arithmetic_props!(Vector2, i32, any_vector2, vector2_i32_arithmetic_props);
vector_arithmetic_props!(Vector2, i64, any_vector2, vector2_i64_arithmetic_props);
vector_arithmetic_props!(Vector2, i128, any_vector2, vector2_i128_arithmetic_props);
vector_arithmetic_props!(Vector2, isize, any_vector2, vector2_isize_arithmetic_props);

vector_arithmetic_props!(Vector3, f32, any_vector3, vector3_f32_arithmetic_props);
vector_arithmetic_props!(Vector3, f64, any_vector3, vector3_f64_arithmetic_props);
vector_arithmetic_props!(Vector3, u8, any_vector3, vector3_u8_arithmetic_props);
vector_arithmetic_props!(Vector3, u16, any_vector3, vector3_u16_arithmetic_props);
vector_arithmetic_props!(Vector3, u32, any_vector3, vector3_u32_arithmetic_props);
vector_arithmetic_props!(Vector3, u64, any_vector3, vector3_u64_arithmetic_props);
vector_arithmetic_props!(Vector3, u128, any_vector3, vector3_u128_arithmetic_props);
vector_arithmetic_props!(Vector3, usize, any_vector3, vector3_usize_arithmetic_props);
vector_arithmetic_props!(Vector3, i8, any_vector3, vector3_i8_arithmetic_props);
vector_arithmetic_props!(Vector3, i16, any_vector3, vector3_i16_arithmetic_props);
vector_arithmetic_props!(Vector3, i32, any_vector3, vector3_i32_arithmetic_props);
vector_arithmetic_props!(Vector3, i64, any_vector3, vector3_i64_arithmetic_props);
vector_arithmetic_props!(Vector3, i128, any_vector3, vector3_i128_arithmetic_props);
vector_arithmetic_props!(Vector3, isize, any_vector3, vector3_isize_arithmetic_props);

vector_arithmetic_props!(Vector4, f32, any_vector4, vector4_f32_arithmetic_props);
vector_arithmetic_props!(Vector4, f64, any_vector4, vector4_f64_arithmetic_props);
vector_arithmetic_props!(Vector4, u8, any_vector4, vector4_u8_arithmetic_props);
vector_arithmetic_props!(Vector4, u16, any_vector4, vector4_u16_arithmetic_props);
vector_arithmetic_props!(Vector4, u32, any_vector4, vector4_u32_arithmetic_props);
vector_arithmetic_props!(Vector4, u64, any_vector4, vector4_u64_arithmetic_props);
vector_arithmetic_props!(Vector4, u128, any_vector4, vector4_u128_arithmetic_props);
vector_arithmetic_props!(Vector4, usize, any_vector4, vector4_usize_arithmetic_props);
vector_arithmetic_props!(Vector4, i8, any_vector4, vector4_i8_arithmetic_props);
vector_arithmetic_props!(Vector4, i16, any_vector4, vector4_i16_arithmetic_props);
vector_arithmetic_props!(Vector4, i32, any_vector4, vector4_i32_arithmetic_props);
vector_arithmetic_props!(Vector4, i64, any_vector4, vector4_i64_arithmetic_props);
vector_arithmetic_props!(Vector4, i128, any_vector4, vector4_i128_arithmetic_props);
vector_arithmetic_props!(Vector4, isize, any_vector4, vector4_isize_arithmetic_props);

vector_add_props!(Vector1, f32, any_vector1_no_overflow, vector1_f32_add_props);
vector_add_props!(Vector1, f64, any_vector1_no_overflow, vector1_f64_add_props);
vector_add_props!(Vector1, u8, any_vector1_no_overflow, vector1_u8_add_props);
vector_add_props!(Vector1, u16, any_vector1_no_overflow, vector1_u16_add_props);
vector_add_props!(Vector1, u32, any_vector1_no_overflow, vector1_u32_add_props);
vector_add_props!(Vector1, u64, any_vector1_no_overflow, vector1_u64_add_props);
vector_add_props!(Vector1, u128, any_vector1_no_overflow, vector1_u128_add_props);
vector_add_props!(Vector1, usize, any_vector1_no_overflow, vector1_usize_add_props);
vector_add_props!(Vector1, i8, any_vector1_no_overflow, vector1_i8_add_props);
vector_add_props!(Vector1, i16, any_vector1_no_overflow, vector1_i16_add_props);
vector_add_props!(Vector1, i32, any_vector1_no_overflow, vector1_i32_add_props);
vector_add_props!(Vector1, i64, any_vector1_no_overflow, vector1_i64_add_props);
vector_add_props!(Vector1, i128, any_vector1_no_overflow, vector1_i128_add_props);
vector_add_props!(Vector1, isize, any_vector1_no_overflow, vector1_isize_add_props);

vector_add_props!(Vector2, f32, any_vector2_no_overflow, vector2_f32_add_props);
vector_add_props!(Vector2, f64, any_vector2_no_overflow, vector2_f64_add_props);
vector_add_props!(Vector2, u8, any_vector2_no_overflow, vector2_u8_add_props);
vector_add_props!(Vector2, u16, any_vector2_no_overflow, vector2_u16_add_props);
vector_add_props!(Vector2, u32, any_vector2_no_overflow, vector2_u32_add_props);
vector_add_props!(Vector2, u64, any_vector2_no_overflow, vector2_u64_add_props);
vector_add_props!(Vector2, u128, any_vector2_no_overflow, vector2_u128_add_props);
vector_add_props!(Vector2, usize, any_vector2_no_overflow, vector2_usize_add_props);
vector_add_props!(Vector2, i8, any_vector2_no_overflow, vector2_i8_add_props);
vector_add_props!(Vector2, i16, any_vector2_no_overflow, vector2_i16_add_props);
vector_add_props!(Vector2, i32, any_vector2_no_overflow, vector2_i32_add_props);
vector_add_props!(Vector2, i64, any_vector2_no_overflow, vector2_i64_add_props);
vector_add_props!(Vector2, i128, any_vector2_no_overflow, vector2_i128_add_props);
vector_add_props!(Vector2, isize, any_vector2_no_overflow, vector2_isize_add_props);

vector_add_props!(Vector3, f32, any_vector3_no_overflow, vector3_f32_add_props);
vector_add_props!(Vector3, f64, any_vector3_no_overflow, vector3_f64_add_props);
vector_add_props!(Vector3, u8, any_vector3_no_overflow, vector3_u8_add_props);
vector_add_props!(Vector3, u16, any_vector3_no_overflow, vector3_u16_add_props);
vector_add_props!(Vector3, u32, any_vector3_no_overflow, vector3_u32_add_props);
vector_add_props!(Vector3, u64, any_vector3_no_overflow, vector3_u64_add_props);
vector_add_props!(Vector3, u128, any_vector3_no_overflow, vector3_u128_add_props);
vector_add_props!(Vector3, usize, any_vector3_no_overflow, vector3_usize_add_props);
vector_add_props!(Vector3, i8, any_vector3_no_overflow, vector3_i8_add_props);
vector_add_props!(Vector3, i16, any_vector3_no_overflow, vector3_i16_add_props);
vector_add_props!(Vector3, i32, any_vector3_no_overflow, vector3_i32_add_props);
vector_add_props!(Vector3, i64, any_vector3_no_overflow, vector3_i64_add_props);
vector_add_props!(Vector3, i128, any_vector3_no_overflow, vector3_i128_add_props);
vector_add_props!(Vector3, isize, any_vector3_no_overflow, vector3_isize_add_props);

vector_add_props!(Vector4, f32, any_vector4_no_overflow, vector4_f32_add_props);
vector_add_props!(Vector4, f64, any_vector4_no_overflow, vector4_f64_add_props);
vector_add_props!(Vector4, u8, any_vector4_no_overflow, vector4_u8_add_props);
vector_add_props!(Vector4, u16, any_vector4_no_overflow, vector4_u16_add_props);
vector_add_props!(Vector4, u32, any_vector4_no_overflow, vector4_u32_add_props);
vector_add_props!(Vector4, u64, any_vector4_no_overflow, vector4_u64_add_props);
vector_add_props!(Vector4, u128, any_vector4_no_overflow, vector4_u128_add_props);
vector_add_props!(Vector4, usize, any_vector4_no_overflow, vector4_usize_add_props);
vector_add_props!(Vector4, i8, any_vector4_no_overflow, vector4_i8_add_props);
vector_add_props!(Vector4, i16, any_vector4_no_overflow, vector4_i16_add_props);
vector_add_props!(Vector4, i32, any_vector4_no_overflow, vector4_i32_add_props);
vector_add_props!(Vector4, i64, any_vector4_no_overflow, vector4_i64_add_props);
vector_add_props!(Vector4, i128, any_vector4_no_overflow, vector4_i128_add_props);
vector_add_props!(Vector4, isize, any_vector4_no_overflow, vector4_isize_add_props);

