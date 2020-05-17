extern crate cgmath;
extern crate proptest;

use proptest::prelude::*;
use cgmath::{Vector1, Zero};

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
            /// Given a vector, it should return the entry at position `index` in the vector 
            /// when the given index is inbounds.
            #[test]
            fn prop_accepts_all_indices_in_of_bounds(index in (0 as usize..$UpperBound as usize)) {
                let v: $VectorN<$FieldType> = $VectorN::zero();
                prop_assert_eq!(v[index], v[index]);
            }
    
            /// Given a vector, when the entry position is out of bounds, it should 
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

index_props!(Vector1, f32, 1, vector1_f32_props);
index_props!(Vector1, f64, 1, vector1_f64_props);
index_props!(Vector2, f32, 2, vector2_f32_props);
index_props!(Vector2, f64, 2, vector2_f64_props);
index_props!(Vector3, f32, 3, vector3_f32_props);
index_props!(Vector3, f64, 3, vector3_f64_props);
index_props!(Vector4, f32, 4, vector4_f32_props);
index_props!(Vector4, f64, 4, vector4_f64_props);

/*
proptest! {
    #[test]
    fn 
}
*/