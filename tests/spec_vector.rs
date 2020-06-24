extern crate gdmath;
extern crate num_traits;
extern crate proptest;

use proptest::prelude::*;
use gdmath::{
    Vector1, 
    Vector2, 
    Vector3, 
    Vector4, 
    Scalar,
};


fn any_vector1<S>() -> impl Strategy<Value = Vector1<S>> where S: Scalar + Arbitrary {
    any::<S>().prop_map(|x| Vector1::new(x))
}

fn any_vector2<S>() -> impl Strategy<Value = Vector2<S>> where S: Scalar + Arbitrary {
    any::<(S, S)>().prop_map(|(x, y)| Vector2::new(x, y))
}

fn any_vector3<S>() -> impl Strategy<Value = Vector3<S>> where S: Scalar + Arbitrary {
    any::<(S, S, S)>().prop_map(|(x, y, z)| Vector3::new(x, y, z))
}

fn any_vector4<S>() -> impl Strategy<Value = Vector4<S>> where S: Scalar + Arbitrary {
    any::<(S, S, S, S)>().prop_map(|(x, y, z, w)| Vector4::new(x, y, z, w))
}


/// Generates the properties tests for vector indexing.
///
/// `$TestModuleName` is a name we give to the module we place the tests in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$VectorN` denotes the name of the vector type.
/// `$ScalarType` denotes the underlying system of numbers that compose `$VectorN`.
/// `$Generator` is the name of a function or closure for generating examples.
/// `$UpperBound` denotes the upperbound on the range of acceptable indices.
macro_rules! index_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $UpperBound:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;

        proptest! {
            /// Given a vector `v`, it should return the entry at position `index` in the vector 
            /// when the given index is inbounds.
            #[test]
            fn prop_accepts_all_indices_in_of_bounds(
                v in super::$Generator::<$ScalarType>(), index in 0..$UpperBound as usize) {

                prop_assert_eq!(v[index], v[index]);
            }
    
            /// Given a vector `v`, when the entry position is out of bounds, it should 
            /// generate a panic just like an array or vector indexed out of bounds.
            #[test]
            #[should_panic]
            fn prop_panics_when_index_out_of_bounds(
                v in super::$Generator::<$ScalarType>(), index in $UpperBound..usize::MAX) {
                
                prop_assert_eq!(v[index], v[index]);
            }
        }
    }
    }
}

index_props!(vector1_f64_index_props, Vector1, f64, any_vector1, 1);
index_props!(vector2_f64_index_props, Vector2, f64, any_vector2, 2);
index_props!(vector3_f64_index_props, Vector3, f64, any_vector3, 3);
index_props!(vector4_f64_index_props, Vector4, f64, any_vector4, 4);


/// Generate the properties for vector arithmetic over exact scalars. We define an exact
/// scalar type as a type where scalar arithmetic is exact (e.g. integers).
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$VectorN` denotes the name of the vector type.
/// `$ScalarType` denotes the underlying system of numbers that compose `$VectorN`.
/// `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_arithmetic_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{$VectorN, Zero};

        proptest! {
            /// A scalar zero times a vector should be zero. That is, vector algebra satisfies
            /// ```
            /// For each vector v, 0 * v = 0.
            /// ```
            #[test]
            fn prop_zero_times_vector_equals_zero(v in super::$Generator()) {
                let zero: $ScalarType = num_traits::Zero::zero();
                let zero_vec = $VectorN::zero();
                prop_assert_eq!(zero * v, zero_vec);
            }
        
            /// A scalar zero times a vector should be zero. That is, vector algebra satisfies
            /// ```
            /// For each vector v, v * 0 = 0.
            /// ```
            /// Note that we deviate from the usual formalisms of vector algebra in that we 
            /// allow the ability to multiply scalars from the left, or from the right of a vector.
            /// In each case the result should be the same.
            #[test]
            fn prop_vector_times_zero_equals_zero(v in super::$Generator()) {
                let zero: $ScalarType = num_traits::Zero::zero();
                let zero_vec = $VectorN::zero();
                prop_assert_eq!(v * zero, zero_vec);
            }

            /// A zero vector should act as the additive unit element of a vector space.
            /// In particular, we have
            /// ```
            /// For every vector v, v + 0 = v.
            /// ```
            #[test]
            fn prop_vector_plus_zero_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$ScalarType>::zero();
                prop_assert_eq!(v + zero_vec, v);
            }

            /// A zero vector should act as the additive unit element of a vector space.
            /// In particular, we have
            /// ```
            /// For every vector v, 0 + v = v.
            /// ```
            #[test]
            fn prop_zero_plus_vector_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$ScalarType>::zero();
                prop_assert_eq!(zero_vec + v, v);
            }

            /// Multiplying a vector by one should give the original vector.
            /// In particular, we have
            /// ```
            /// For every vector v, 1 * v = v.
            /// ```
            #[test]
            fn prop_one_times_vector_equal_vector(v in super::$Generator()) {
                let one: $ScalarType = num_traits::One::one();
                prop_assert_eq!(one * v, v);
            }

            /// Multiplying a vector by one should give the original vector.
            /// In particular, we have
            /// ```
            /// For every vector v, v * 1 = v.
            /// ```
            /// Note that we deviate from the usual formalisms of vector algebra in that we 
            /// allow the ability to multiply scalars from the left, or from the right of a vector.
            /// In each case the result should be the same.
            #[test]
            fn prop_vector_times_one_equals_vector(v in super::$Generator()) {
                let one: $ScalarType = num_traits::One::one();
                prop_assert_eq!(one * v, v);
            }
        }
    }
    }
}

exact_arithmetic_props!(vector1_f64_arithmetic_props, Vector1, f64, any_vector1);
exact_arithmetic_props!(vector2_f64_arithmetic_props, Vector2, f64, any_vector2);
exact_arithmetic_props!(vector3_f64_arithmetic_props, Vector3, f64, any_vector3);
exact_arithmetic_props!(vector4_f64_arithmetic_props, Vector4, f64, any_vector4);

exact_arithmetic_props!(vector1_i32_arithmetic_props, Vector1, i32, any_vector1);
exact_arithmetic_props!(vector2_i32_arithmetic_props, Vector2, i32, any_vector2);
exact_arithmetic_props!(vector3_i32_arithmetic_props, Vector3, i32, any_vector3);
exact_arithmetic_props!(vector4_i32_arithmetic_props, Vector4, i32, any_vector4);

exact_arithmetic_props!(vector1_u32_arithmetic_props, Vector1, u32, any_vector1);
exact_arithmetic_props!(vector2_u32_arithmetic_props, Vector2, u32, any_vector2);
exact_arithmetic_props!(vector3_u32_arithmetic_props, Vector3, u32, any_vector3);
exact_arithmetic_props!(vector4_u32_arithmetic_props, Vector4, u32, any_vector4);


/// Generate the properties for vector arithmetic over floating point scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$VectorN` denotes the name of the vector type.
/// `$ScalarType` denotes the underlying system of numbers that compose `$VectorN`.
/// `$Generator` is the name of a function or closure for generating examples.
///
/// We use approximate comparisons because arithmetic is not exact over finite precision floating point
/// scalar types.
macro_rules! approx_add_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{$VectorN, Zero};

        proptest! {
            /// A vector plus a zero vector equals the same vector. The vector algebra satisfies
            /// the following: given a vector `v`
            /// ```
            /// v + 0 = v.
            /// ```
            #[test]
            fn prop_vector_plus_zero_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$ScalarType>::zero();
                prop_assert_eq!(v + zero_vec, v);
            }

            /// A vector plus a zero vector equals the same vector. The vector algebra satisfies
            /// the following: Given a vector `v`
            /// ```
            /// 0 + v = v.
            /// ```
            #[test]
            fn prop_zero_plus_vector_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$ScalarType>::zero();
                prop_assert_eq!(zero_vec + v, v);
            }

            /// Given vectors `v1` and `v2`, we should be able to use `v1` and `v2` interchangeably 
            /// with their references `&v1` and `&v2` in arithmetic expressions involving vectors. 
            /// In the case of vector addition, the vectors should satisfy
            /// ```
            ///  v1 +  v2 = &v1 +  v2
            ///  v1 +  v2 =  v1 + &v2
            ///  v1 +  v2 = &v1 + &v2
            ///  v1 + &v2 = &v1 +  v2
            /// &v1 +  v2 =  v1 + &v2
            /// &v1 +  v2 = &v1 + &v2
            ///  v1 + &v2 = &v1 + &v2
            /// ```
            #[test]
            fn prop_vector1_plus_vector2_equals_refvector1_plus_refvector2(
                v1 in super::$Generator::<$ScalarType>(), v2 in super::$Generator::<$ScalarType>()) {
                
                prop_assert_eq!(v1 + v2, &v1 + v2);
                prop_assert_eq!(v1 + v2, v1 + &v2);
                prop_assert_eq!(v1 + v2, &v1 + &v2);
                prop_assert_eq!(v1 + &v2, &v1 + v2);
                prop_assert_eq!(&v1 + v2, v1 + &v2);
                prop_assert_eq!(&v1 + v2, &v1 + &v2);
                prop_assert_eq!(v1 + &v2, &v1 + &v2);
            }

            /// Given two vectors of floating point scalars, vector addition should  be approximately
            /// commutative. Given vectors `v1` and `v2`, we have
            /// ```
            /// v1 + v2 ~= v2 + v1.
            /// ```
            /// Note that floating point vector addition cannot be exactly commutative because arithmetic
            /// with floating point numbers is not commutative.
            #[test]
            fn prop_vector_addition_almost_commutative(
                v1 in super::$Generator::<$ScalarType>(), v2 in super::$Generator::<$ScalarType>()) {
                
                let zero: $VectorN<$ScalarType> = Zero::zero();
                prop_assert_eq!((v1 + v2) - (v2 + v1), zero);
            }

            /// Given three vectors of floating point scalars, vector addition should  be approximately
            /// associative. Given vectors `v1`, `v2`, and `v3` we have
            /// ```
            /// (v1 + v2) + v3 ~= v1 + (v2 + v3).
            /// ```
            /// Note that floating point vector addition cannot be exactly associative because arithmetic
            /// with floating point numbers is not associative.
            #[test]
            fn prop_vector_addition_associative(
                u in super::$Generator::<$ScalarType>(), 
                v in super::$Generator::<$ScalarType>(), w in super::$Generator::<$ScalarType>()) {

                prop_assert_eq!((u + v) + w, u + (v + w));
            }
        }
    }
    }
}

approx_add_props!(vector1_f64_add_props, Vector1, f64, any_vector1);
approx_add_props!(vector2_f64_add_props, Vector2, f64, any_vector2);
approx_add_props!(vector3_f64_add_props, Vector3, f64, any_vector3);
approx_add_props!(vector4_f64_add_props, Vector4, f64, any_vector4);


/// Generate the properties for vector arithmetic over exact scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$VectorN` denotes the name of the vector type.
/// `$ScalarType` denotes the underlying system of numbers that compose `$VectorN`.
/// `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_add_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{$VectorN, Zero};

        proptest! {
            /// A vector plus a zero vector equals the same vector. The vector algebra satisfies
            /// the following: Given a vector `v`
            /// ```
            /// v + 0 = v.
            /// ```
            #[test]
            fn prop_vector_plus_zero_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$ScalarType>::zero();
                prop_assert_eq!(v + zero_vec, v);
            }

            /// A zero vector plus a vector equals the same vector. The vector algebra satisfies
            /// the following: Given a vector `v`
            /// ```
            /// 0 + v = v.
            /// ```
            #[test]
            fn prop_zero_plus_vector_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$ScalarType>::zero();
                prop_assert_eq!(zero_vec + v, v);
            }

            /// Given vectors `v1` and `v2`, we should be able to use `v1` and `v2` interchangeably 
            /// with their references `&v1` and `&v2` in arithmetic expressions involving vectors. 
            /// In the case of vector addition, the vectors should satisfy
            /// ```
            ///  v1 +  v2 = &v1 +  v2
            ///  v1 +  v2 =  v1 + &v2
            ///  v1 +  v2 = &v1 + &v2
            ///  v1 + &v2 = &v1 +  v2
            /// &v1 +  v2 =  v1 + &v2
            /// &v1 +  v2 = &v1 + &v2
            ///  v1 + &v2 = &v1 + &v2
            /// ```
            #[test]
            fn prop_vector1_plus_vector2_equals_refvector1_plus_refvector2(
                v1 in super::$Generator::<$ScalarType>(), v2 in super::$Generator::<$ScalarType>()) {
                
                prop_assert_eq!(v1 + v2, &v1 + v2);
                prop_assert_eq!(v1 + v2, v1 + &v2);
                prop_assert_eq!(v1 + v2, &v1 + &v2);
                prop_assert_eq!(v1 + &v2, &v1 + v2);
                prop_assert_eq!(&v1 + v2, v1 + &v2);
                prop_assert_eq!(&v1 + v2, &v1 + &v2);
                prop_assert_eq!(v1 + &v2, &v1 + &v2);
            }

            /// Given two vectors of integer scalars, vector addition should be
            /// commutative. Given vectors `v1` and `v2`, we have
            /// ```
            /// v1 + v2 = v2 + v1.
            /// ```
            #[test]
            fn prop_vector_addition_commutative(
                v1 in super::$Generator::<$ScalarType>(), v2 in super::$Generator::<$ScalarType>()) {
                
                let zero: $VectorN<$ScalarType> = Zero::zero();
                prop_assert_eq!((v1 + v2) - (v2 + v1), zero);
            }

            /// Given three vectors of integer scalars, vector addition should be associative.
            /// Given vectors `v1`, `v2`, and `v3`, we have
            /// ```
            /// (v1 + v2) + v3 = v1 + (v2 + v3)
            /// ```
            #[test]
            fn prop_vector_addition_associative(
                u in super::$Generator::<$ScalarType>(), 
                v in super::$Generator::<$ScalarType>(), w in super::$Generator::<$ScalarType>()) {

                prop_assert_eq!((u + v) + w, u + (v + w));
            }
        }
    }
    }
}

exact_add_props!(vector1_i32_add_props, Vector1, i32, any_vector1);
exact_add_props!(vector2_i32_add_props, Vector2, i32, any_vector2);
exact_add_props!(vector3_i32_add_props, Vector3, i32, any_vector3);
exact_add_props!(vector4_i32_add_props, Vector4, i32, any_vector4);

exact_add_props!(vector1_u32_add_props, Vector1, u32, any_vector1);
exact_add_props!(vector2_u32_add_props, Vector2, u32, any_vector2);
exact_add_props!(vector3_u32_add_props, Vector3, u32, any_vector3);
exact_add_props!(vector4_u32_add_props, Vector4, u32, any_vector4);


/// Generate the properties for vector subtraction over floating point scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$VectorN` denotes the name of the vector type.
/// `$ScalarType` denotes the underlying system of numbers that compose `$VectorN`.
/// `$Generator` is the name of a function or closure for generating examples.
///
/// We use approximate comparisons because arithmetic is not exact over finite precision floating point
/// scalar types.
macro_rules! approx_sub_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{$VectorN, Zero};

        proptest! {
            /// The zero vector over of floating point scalars should act as an additive unit. 
            /// That is, given a vector `v`, we have
            /// ```
            /// v - 0 = v
            /// ```
            #[test]
            fn prop_vector_minus_zero_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$ScalarType>::zero();
                prop_assert_eq!(v - zero_vec, v);
            }

            /// Every vector should have an additive inverse. That is, given a vector `v`,
            /// there is a vector `-v` such that
            /// we have
            /// ```
            /// v - v = 0
            /// ```
            #[test]
            fn prop_vector_minus_vector_equals_zero(v in super::$Generator::<$ScalarType>()) {
                let zero_vec = $VectorN::<$ScalarType>::zero();
                prop_assert_eq!(v - v, zero_vec);
            }
        }
    }
    }
}

approx_sub_props!(vector1_f64_sub_props, Vector1, f64, any_vector1);
approx_sub_props!(vector2_f64_sub_props, Vector2, f64, any_vector2);
approx_sub_props!(vector3_f64_sub_props, Vector3, f64, any_vector3);
approx_sub_props!(vector4_f64_sub_props, Vector4, f64, any_vector4);


/// Generate the properties for vector arithmetic over exact scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$VectorN` denotes the name of the vector type.
/// `$ScalarType` denotes the underlying system of numbers that compose `$VectorN`.
/// `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_sub_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{$VectorN, Zero};

        proptest! {
            /// The zero vector should act as an additive unit. That is, given a vector `v`,
            /// we have
            /// ```
            /// v - 0 = v
            /// ```
            #[test]
            fn prop_vector_minus_zero_equals_vector(v in super::$Generator()) {
                let zero_vec = $VectorN::<$ScalarType>::zero();
                prop_assert_eq!(v - zero_vec, v);
            }

            /// Every vector should have an additive inverse. That is, given a vector `v`,
            /// there is a vector `-v` such that
            /// we have
            /// ```
            /// v - v = 0
            /// ```
            #[test]
            fn prop_vector_minus_vector_equals_zero(v in super::$Generator::<$ScalarType>()) {
                let zero_vec = $VectorN::<$ScalarType>::zero();
                prop_assert_eq!(v - v, zero_vec);
            }
        }
    }
    }
}

exact_sub_props!(vector1_i32_sub_props, Vector1, i32, any_vector1);
exact_sub_props!(vector2_i32_sub_props, Vector2, i32, any_vector2);
exact_sub_props!(vector3_i32_sub_props, Vector3, i32, any_vector3);
exact_sub_props!(vector4_i32_sub_props, Vector4, i32, any_vector4);

exact_sub_props!(vector1_u32_sub_props, Vector1, u32, any_vector1);
exact_sub_props!(vector2_u32_sub_props, Vector2, u32, any_vector2);
exact_sub_props!(vector3_u32_sub_props, Vector3, u32, any_vector3);
exact_sub_props!(vector4_u32_sub_props, Vector4, u32, any_vector4);


/// Generate the properties for vector magnitudes.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$VectorN` denotes the name of the vector type.
/// `$ScalarType` denotes the underlying system of numbers that compose `$VectorN`.
/// `$Generator` is the name of a function or closure for generating examples.
/// `$tolerance` specifies the highest amount of acceptable error in the floating point computations
///  that still defines a correct computation. We cannot guarantee floating point computations
///  will be exact since the underlying floating point arithmetic is not exact.
macro_rules! magnitude_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::{$VectorN, Magnitude};
        use gdmath::approx::{relative_eq, relative_ne};

        proptest! {
            #[test]
            /// The magnitude of a vector preserves scales. Given a scalar constant `c`, and a 
            /// vector `v` of scalars, the magnitude function satisfies
            /// ```
            /// magnitude(c * v) = abs(c) * magnitude(v)
            /// ```
            fn prop_magnitude_preserves_scale(
                v in super::$Generator::<$ScalarType>(), c in any::<$ScalarType>()) {
                
                let abs_c = <$ScalarType as num_traits::Float>::abs(c);                
                prop_assume!((abs_c * v.magnitude()).is_finite());
                prop_assume!((c * v).magnitude().is_finite());
                
                prop_assert!(
                    relative_eq!( (c * v).magnitude(), abs_c * v.magnitude(), epsilon = $tolerance),
                    "\n||c * v|| = {}\n|c| * ||v|| = {}\n", (c * v).magnitude(), abs_c * v.magnitude(),
                );
            }

            /// The magnitude of a vector is nonnegative. Given a vector `v`
            /// ```
            /// magnitude(v) >= 0
            /// ```
            #[test]
            fn prop_magnitude_nonnegative(v in super::$Generator::<$ScalarType>()) {
                let zero = <$ScalarType as num_traits::Zero>::zero();
                prop_assert!(v.magnitude() >= zero);
            }

            /// The magnitude of a vector satisfies the triangle inequality. Given a vector `v` and a 
            /// vector `w`, the magnitude function satisfies
            /// ```
            /// magnitude(v + w) <= magnitude(v) + magnitude(w)
            /// ```
            #[test]
            fn prop_magnitude_satisfies_triangle_inequality(
                v in super::$Generator::<$ScalarType>(), w in super::$Generator::<$ScalarType>()) {
            
                prop_assume!((v + w).magnitude().is_finite());
                prop_assume!((v.magnitude() + w.magnitude()).is_finite());
                prop_assert!((v + w).magnitude() <= v.magnitude() + w.magnitude(), 
                    "\n|v + w| = {}\n|v| = {}\n|w| = {}\n|v| + |w| = {}\n",
                    (v + w).magnitude(), v.magnitude(), w.magnitude(), v.magnitude() + w.magnitude()
                );
            }

            /// The magnitude function is point separating. In particular, if the distance between two 
            /// vectors `v` and `w` is zero, then v = w:
            /// ```
            /// magnitude(v - w) = 0 => v = w 
            /// ```
            /// Equivalently, if `v` is not equal to `w`, then their distance is nonzero
            /// ```
            /// v != w => magnitude(v - w) != 0
            /// ```
            /// For the sake of testability, we use the second form to test the magnitude function.
            #[test]
            fn prop_magnitude_approx_point_separating(v in super::$Generator::<$ScalarType>()) {
                let zero_vec = <$VectorN<$ScalarType> as gdmath::Zero>::zero();

                prop_assume!(relative_ne!(v, zero_vec, epsilon = $tolerance));
                prop_assert!(relative_ne!(v.magnitude(), zero_vec.magnitude(), epsilon = $tolerance),
                    "\n|v| = {}\n|zero_vec| = {}\n", v.magnitude(), zero_vec.magnitude()
                );
            }
        }
    }
    }
}

magnitude_props!(vector1_f64_magnitude_props, Vector1, f64, any_vector1, 1e-7);
magnitude_props!(vector2_f64_magnitude_props, Vector2, f64, any_vector2, 1e-7);
magnitude_props!(vector3_f64_magnitude_props, Vector3, f64, any_vector3, 1e-7);
magnitude_props!(vector4_f64_magnitude_props, Vector4, f64, any_vector4, 1e-7);


/// Generate the properties for vector multiplication over floating point scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$VectorN` denotes the name of the vector type.
/// `$ScalarType` denotes the underlying system of numbers that compose `$VectorN`.
/// `$Generator` is the name of a function or closure for generating examples.
/// `$tolerance` specifies the highest amount of acceptable error in the floating point computations
///  that still defines a correct computation. We cannot guarantee floating point computations
///  will be exact since the underlying floating point arithmetic is not exact.
///
/// We use approximate comparisons because arithmetic is not exact over finite precision floating point
/// scalar types.
macro_rules! approx_mul_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident, $tolerance:expr) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::Magnitude;
        use gdmath::approx::relative_eq;

        proptest! {
            /// Multiplication of a scalar and a vector should be approximately commutative.
            /// Given a constant `c` and a vector `v`
            /// ```
            /// c * v ~= v * c
            /// ```
            /// We deviate from the usual formalisms of vector algebra in that we 
            /// allow the ability to multiply scalars from the left, or from the right of a vector.
            /// Note that floating point vector multiplication cannot be commutative because 
            /// multiplication in the underlying floating point scalars is not commutative.
            #[test]
            fn prop_scalar_times_vector_equals_vector_times_scalar(
                c in any::<$ScalarType>(), v in super::$Generator::<$ScalarType>()) {
                
                prop_assume!(c.is_finite());
                prop_assume!(v.magnitude().is_finite());
                prop_assert!(
                    relative_eq!(c * v, v * c, epsilon = $tolerance)
                );
            }

            /// Multiplication of two scalars and a vector should be compatible with multiplication of 
            /// all scalars. In other words, scalar multiplication of two scalar with a vector should 
            /// act associatively, just like the multiplication of three scalars. 
            /// Given scalars `a` and `b`, and a vector `v`, we have
            /// ```
            /// (a * b) * v ~= a * (b * v)
            /// ```
            /// Note that the compatability of scalars with vectors can only be approximate and not 
            /// exact because multiplication of the underlying scalars is not associative. 
            #[test]
            fn prop_scalar_multiplication_compatability(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), v in super::$Generator::<$ScalarType>()) {

                prop_assert!(relative_eq!(a * (b * v), (a * b) * v, epsilon = $tolerance));
            }
        }
    }
    }
}

approx_mul_props!(vector1_f64_mul_props, Vector1, f64, any_vector1, 1e-7);
approx_mul_props!(vector2_f64_mul_props, Vector2, f64, any_vector2, 1e-7);
approx_mul_props!(vector3_f64_mul_props, Vector3, f64, any_vector3, 1e-7);
approx_mul_props!(vector4_f64_mul_props, Vector4, f64, any_vector4, 1e-7);


/// Generate the properties for vector multiplication over exact scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$VectorN` denotes the name of the vector type.
/// `$ScalarType` denotes the underlying system of numbers that compose `$VectorN`.
/// `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_mul_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;

        proptest! {
            /// Exact multiplication of a scalar and a vector should be commutative.
            /// Given a constant `c` and a vector `v`
            /// ```
            /// c * v = v * c
            /// ```
            /// We deviate from the usual formalisms of vector algebra in that we 
            /// allow the ability to multiply scalars from the left, or from the right of a vector.
            #[test]
            fn prop_scalar_times_vector_equals_vector_times_scalar(
                c in any::<$ScalarType>(), v in super::$Generator::<$ScalarType>()) {
                
                prop_assert_eq!(c * v, v * c);
            }

            /// Exact multiplication of two scalars and a vector should be compatible with multiplication of 
            /// all scalars. In other words, scalar multiplication of two scalars with a vector should 
            /// act associatively just like the multiplication of three scalars. 
            /// Given scalars `a` and `b`, and a vector `v`, we have
            /// ```
            /// (a * b) * v = a * (b * v)
            /// ```
            #[test]
            fn prop_scalar_multiplication_compatability(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), v in super::$Generator::<$ScalarType>()) {

                prop_assert_eq!(a * (b * v), (a * b) * v);
            }
        }
    }
    }
}

exact_mul_props!(vector1_i32_mul_props, Vector1, i32, any_vector1);
exact_mul_props!(vector2_i32_mul_props, Vector2, i32, any_vector2);
exact_mul_props!(vector3_i32_mul_props, Vector3, i32, any_vector3);
exact_mul_props!(vector4_i32_mul_props, Vector4, i32, any_vector4);

exact_mul_props!(vector1_u32_mul_props, Vector1, u32, any_vector1);
exact_mul_props!(vector2_u32_mul_props, Vector2, u32, any_vector2);
exact_mul_props!(vector3_u32_mul_props, Vector3, u32, any_vector3);
exact_mul_props!(vector4_u32_mul_props, Vector4, u32, any_vector4);


/// Generate the properties for vector distribution over floating point scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$VectorN` denotes the name of the vector type.
/// `$ScalarType` denotes the underlying system of numbers that compose `$VectorN`.
/// `$Generator` is the name of a function or closure for generating examples.
///
/// We use approximate comparisons because arithmetic is not exact over finite precision floating point
/// scalar types.
macro_rules! approx_distributive_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use gdmath::Magnitude;
    
        proptest! {
            /// Scalar multiplication should approximately distribute over vector addition.
            /// Given a scalar `a` and vectors `v` and `w`
            /// ```
            /// a * (v + w) ~= a * v + a * w
            /// ```
            #[test]
            fn prop_distribution_over_vector_addition(
                a in any::<$ScalarType>(), 
                v in super::$Generator::<$ScalarType>(), w in super::$Generator::<$ScalarType>()) {
                
                prop_assume!((a * (v + w)).magnitude().is_finite());
                prop_assume!((a * v + a * w).magnitude().is_finite());
                prop_assert_eq!(a * (v + w), a * v + a * w);
            }
    
            /// Multiplication of a sum of scalars should approximately distribute over a vector.
            /// Given scalars `a` and `b` and a vector `v`, we have
            /// ```
            /// (a + b) * v ~= a * v + b * v
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                v in super::$Generator::<$ScalarType>()) {
    
                prop_assume!(((a + b) * v).magnitude().is_finite());
                prop_assume!((a * v + b * v).magnitude().is_finite());
                prop_assert_eq!((a + b) * v, a * v + b * v);
            }

            /// Multiplication of two vectors by a scalar on the right should approximately distribute.
            /// Given vectors `v` and `w` and a scalar `a`
            /// ```
            /// (v + w) * a ~= v * a + w * a
            /// ```
            /// We deviate from the usual formalisms of vector algebra in that we 
            /// allow the ability to multiply scalars from the left, or from the right of a vector.
            #[test]
            fn prop_distribution_over_vector_addition1(
                a in any::<$ScalarType>(), 
                v in super::$Generator::<$ScalarType>(), w in super::$Generator::<$ScalarType>()) {
                    
                prop_assume!(((v + w) * a).magnitude().is_finite());
                prop_assume!((v * a + w * a).magnitude().is_finite());
                prop_assert_eq!((v + w) * a,  v * a + w * a);
            }

            /// Multiplication of a vector on the right by the sum of two scalars should approximately 
            /// distribute over the two scalars. 
            /// Given a vector `v` and scalars `a` and `b`
            /// ```
            /// v * (a + b) ~= v * a + v * b
            /// ```
            /// We deviate from the usual formalisms of vector algebra in that we 
            /// allow the ability to multiply scalars from the left, or from the right of a vector.
            #[test]
            fn prop_distribution_over_scalar_addition1(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                v in super::$Generator::<$ScalarType>()) {
    
                prop_assume!((v * (a + b)).magnitude().is_finite());
                prop_assume!((v * a + v * b).magnitude().is_finite());
                prop_assert_eq!(v * (a + b), v * a + v * b);
            }
        }
    }
    }    
}

approx_distributive_props!(vector1_f64_distributive_props, Vector1, f64, any_vector1);
approx_distributive_props!(vector2_f64_distributive_props, Vector2, f64, any_vector2);
approx_distributive_props!(vector3_f64_distributive_props, Vector3, f64, any_vector3);
approx_distributive_props!(vector4_f64_distributive_props, Vector4, f64, any_vector4);


/// Generate the properties for vector distribution over exact scalars.
///
/// `$TestModuleName` is a name we give to the module we place the properties in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$VectorN` denotes the name of the vector type.
/// `$ScalarType` denotes the underlying system of numbers that compose `$VectorN`.
/// `$Generator` is the name of a function or closure for generating examples.
macro_rules! exact_distributive_props {
    ($TestModuleName:ident, $VectorN:ident, $ScalarType:ty, $Generator:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;

        proptest! {
            /// Scalar multiplication should distribute over vector addition.
            /// Given a scalar `a` and vectors `v` and `w`
            /// ```
            /// a * (v + w) = a * v + a * w
            /// ```
            #[test]
            fn prop_distribution_over_vector_addition(
                a in any::<$ScalarType>(), 
                v in super::$Generator::<$ScalarType>(), w in super::$Generator::<$ScalarType>()) {
                
                prop_assert_eq!(a * (v + w), a * v + a * w);
                prop_assert_eq!((v + w) * a,  v * a + w * a);
            }

            /// Multiplication of a sum of scalars should distribute over a vector.
            /// Given scalars `a` and `b` and a vector `v`, we have
            /// ```
            /// (a + b) * v = a * v + b * v
            /// ```
            #[test]
            fn prop_distribution_over_scalar_addition(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                v in super::$Generator::<$ScalarType>()) {
    
                prop_assert_eq!((a + b) * v, a * v + b * v);
                prop_assert_eq!(v * (a + b), v * a + v * b);
            }

            /// Multiplication of two vectors by a scalar on the right should distribute.
            /// Given vectors `v` and `w` and a scalar `a`
            /// ```
            /// (v + w) * a = v * a + w * a
            /// ```
            /// We deviate from the usual formalisms of vector algebra in that we 
            /// allow the ability to multiply scalars from the left, or from the right of a vector.
            #[test]
            fn prop_distribution_over_vector_addition1(
                a in any::<$ScalarType>(), 
                v in super::$Generator::<$ScalarType>(), w in super::$Generator::<$ScalarType>()) {
                    
                prop_assert_eq!((v + w) * a,  v * a + w * a);
            }

            /// Multiplication of a vector on the right by the sum of two scalars should
            /// distribute over the two scalars. 
            /// Given a vector `v` and scalars `a` and `b`
            /// ```
            /// v * (a + b) = v * a + v * b
            /// ```
            /// We deviate from the usual formalisms of vector algebra in that we 
            /// allow the ability to multiply scalars from the left, or from the right of a vector.
            #[test]
            fn prop_distribution_over_scalar_addition1(
                a in any::<$ScalarType>(), b in any::<$ScalarType>(), 
                v in super::$Generator::<$ScalarType>()) {
    
                prop_assert_eq!(v * (a + b), v * a + v * b);
            }
        }
    }
    }    
}

exact_distributive_props!(vector1_i32_distributive_props, Vector1, i32, any_vector1);
exact_distributive_props!(vector2_i32_distributive_props, Vector2, i32, any_vector2);
exact_distributive_props!(vector3_i32_distributive_props, Vector3, i32, any_vector3);
exact_distributive_props!(vector4_i32_distributive_props, Vector4, i32, any_vector4);

exact_distributive_props!(vector1_u32_distributive_props, Vector1, u32, any_vector1);
exact_distributive_props!(vector2_u32_distributive_props, Vector2, u32, any_vector2);
exact_distributive_props!(vector3_u32_distributive_props, Vector3, u32, any_vector3);
exact_distributive_props!(vector4_u32_distributive_props, Vector4, u32, any_vector4);
