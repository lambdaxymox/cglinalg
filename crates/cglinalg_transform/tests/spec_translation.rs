extern crate cglinalg_numeric;
extern crate cglinalg_core;
extern crate cglinalg_transform;
extern crate proptest;


use cglinalg_numeric::{
    SimdScalarSigned,
};
use cglinalg_core::{
    Point,
    Point2,
    Point3,
    Vector,
};
use cglinalg_transform::{
    Translation,
    Translation2,
    Translation3,
};

use proptest::prelude::*;


fn strategy_vector_any<S, const N: usize>() -> impl Strategy<Value = Vector<S, N>>
where 
    S: SimdScalarSigned + Arbitrary 
{
    any::<[S; N]>().prop_map(|array| {        
        let mut result = Vector::zero();
        for i in 0..N {
            let sign_value = array[i].signum();
            let abs_value = array[i].abs();
            result[i] = sign_value * abs_value;
        }

        result
    })
}

fn strategy_point_any<S, const N: usize>() -> impl Strategy<Value = Point<S, N>>
where 
    S: SimdScalarSigned + Arbitrary 
{
    any::<[S; N]>().prop_map(|array| {        
        let mut result = Point::origin();
        for i in 0..N {
            let sign_value = array[i].signum();
            let abs_value = array[i].abs();
            result[i] = sign_value * abs_value;
        }

        result
    })
}

fn strategy_translation_any<S, const N: usize>() -> impl Strategy<Value = Translation<S, N>>
where
    S: SimdScalarSigned + Arbitrary
{
    strategy_vector_any::<S, N>().prop_map(|vector| { 
        Translation::from_vector(&vector)
    })
}


/// Translations preserve the differences between points.
/// 
/// Given a translation `T`, and points `p1` and `p2`
/// ```text
/// T(p1 - p2) == p1 - p2
/// ```
fn prop_translation_preserves_point_differences<S, const N: usize>(
    t: Translation<S, N>,
    p1: Point<S, N>,
    p2: Point<S, N>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = t * (p1 - p2);
    let rhs = p1 - p2;

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// Every translation has an inverse such that when applied to a point, the 
/// original point remains.
/// 
/// Given a translation `T` and a point `p`
/// ```text
/// (T * inverse(T)) * p == p
/// (inverse(T) * T) * p == p
/// ```
/// This property is meant to test the interaction of translations with points.
fn prop_translation_translation_inverse_is_original_point<S, const N: usize>(
    t: Translation<S, N>, 
    p: Point<S, N>
) -> Result<(), TestCaseError> 
where
    S: SimdScalarSigned
{
    prop_assert_eq!((t * t.inverse()) * p, p);
    prop_assert_eq!((t.inverse() * t) * p, p);

    Ok(())
}

/// Every translation has a unique inverse, and a translation commutes 
/// with its inverse.
/// 
/// Given a translation `T`
/// ```text
/// T * inverse(T) == inverse(T) * T == Identity
/// ```
fn prop_translation_translation_inverse<S, const N: usize>(t: Translation<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = t * t.inverse();
    let rhs = t.inverse() * t;
    let identity = Translation::identity();

    prop_assert_eq!(lhs, rhs);
    prop_assert_eq!(lhs, identity);
    prop_assert_eq!(rhs, identity);

    Ok(())
}

/// Translation composition is associative.
/// 
/// Given translations `T1`, `T2`, and `T3`
/// ```text
/// (T1 * T2) * T3 == T1 * (T2 * T3)
/// ```
fn prop_translation_composition_associative<S, const N: usize>(
    t1: Translation<S, N>, 
    t2: Translation<S, N>, 
    t3: Translation<S, N>,
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = (t1 * t2) * t3;
    let rhs = t1 * (t2 * t3);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}


#[cfg(test)]
mod translation2_i32_point_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_translation_preserves_point_differences(
            t in super::strategy_translation_any(),
            p1 in super::strategy_point_any(),
            p2 in super::strategy_point_any() 
        ) {
            let t: super::Translation2<i32> = t;
            let p1: super::Point2<i32> = p1;
            let p2: super::Point2<i32> = p2;
            super::prop_translation_preserves_point_differences(t, p1, p2)?
        }

        #[test]
        fn prop_translation_translation_inverse_is_original_point(
            t in super::strategy_translation_any(), 
            p in super::strategy_point_any()
        ) {
            let t: super::Translation2<i32> = t;
            let p: super::Point2<i32> = p;
            super::prop_translation_translation_inverse_is_original_point(t, p)?
        }
    }
}

#[cfg(test)]
mod translation3_i32_point_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_translation_preserves_point_differences(
            t in super::strategy_translation_any(),
            p1 in super::strategy_point_any(),
            p2 in super::strategy_point_any() 
        ) {
            let t: super::Translation3<i32> = t;
            let p1: super::Point3<i32> = p1;
            let p2: super::Point3<i32> = p2;
            super::prop_translation_preserves_point_differences(t, p1, p2)?
        }

        #[test]
        fn prop_translation_translation_inverse_is_original_point(
            t in super::strategy_translation_any(), 
            p in super::strategy_point_any()
        ) {
            let t: super::Translation3<i32> = t;
            let p: super::Point3<i32> = p;
            super::prop_translation_translation_inverse_is_original_point(t, p)?
        }
    }
}


#[cfg(test)]
mod translation2_i32_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_translation_translation_inverse(t in super::strategy_translation_any()) {
            let t: super::Translation2<i32> = t;
            super::prop_translation_translation_inverse(t)?
        }

        #[test]
        fn prop_translation_composition_associative(
            t1 in super::strategy_translation_any(), 
            t2 in super::strategy_translation_any(), 
            t3 in super::strategy_translation_any()
        ) {
            let t1: super::Translation2<i32> = t1;
            let t2: super::Translation2<i32> = t2;
            let t3: super::Translation2<i32> = t3;
            super::prop_translation_composition_associative(t1, t2, t3)?
        }
    }
}

#[cfg(test)]
mod translation3_i32_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_translation_translation_inverse(t in super::strategy_translation_any()) {
            let t: super::Translation3<i32> = t;
            super::prop_translation_translation_inverse(t)?
        }

        #[test]
        fn prop_translation_composition_associative(
            t1 in super::strategy_translation_any(), 
            t2 in super::strategy_translation_any(), 
            t3 in super::strategy_translation_any()
        ) {
            let t1: super::Translation3<i32> = t1;
            let t2: super::Translation3<i32> = t2;
            let t3: super::Translation3<i32> = t3;
            super::prop_translation_composition_associative(t1, t2, t3)?
        }
    }
}

