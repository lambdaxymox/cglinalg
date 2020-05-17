extern crate cgmath;
extern crate proptest;

use proptest::prelude::*;
use cgmath::{Vector1, Zero};

macro_rules! index_props {
    ($VectorN:ident, $FieldType:ty, $UpperBound:expr, $TestModuleName:ident) => {
    #[cfg(test)]
    mod $TestModuleName {
        use proptest::prelude::*;
        use cgmath::{$VectorN, Zero};

        proptest! {
            #[test]
            fn accepts_all_indices_in_of_bounds(index in (0 as usize..$UpperBound as usize)) {
                let v: $VectorN<$FieldType> = $VectorN::zero();
                prop_assert_eq!(v[index], v[index]);
            }
    
            #[test]
            #[should_panic]
            fn panics_when_index_out_of_bounds(index in $UpperBound..usize::MAX) {
                let v: $VectorN<$FieldType> = $VectorN::zero();
                prop_assert_eq!(v[index], v[index]);
            }
        }
    }
    }
}

index_props!(Vector1, f32, 1, vector1_f32_props);
index_props!(Vector1, f64, 1, vector1_f64_props);

/*
proptest! {
    #[test]
    fn 
}
*/