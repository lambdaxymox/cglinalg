extern crate gdmath;
extern crate num_traits;
extern crate proptest;

use proptest::prelude::*;
use gdmath::{
    Quaternion, 
    ScalarFloat,
};

fn any_quaternion<S>() -> impl Strategy<Value = Quaternion<S>> where S: ScalarFloat + Arbitrary {
    any::<(S, S, S, S)>().prop_map(|(x, y, z, w)| Quaternion::new(x, y, z, w))
}


/// Generates the properties tests for quaternion indexing.
///
/// `$TestModuleName` is a name we give to the module we place the tests in to separate them
///  from each other for each field type to prevent namespace collisions.
/// `$ScalarType` denotes the underlying system of numbers that compose `$VectorN`.
/// `$UpperBound` denotes the upperbound on the range of acceptable indices.
macro_rules! index_props {
    ($TestModuleName:ident, $ScalarType:ty, $Generator:ident, $UpperBound:expr) => {
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

index_props!(quaternion_index_props, f64, any_quaternion, 4);
