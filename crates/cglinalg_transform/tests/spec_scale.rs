extern crate cglinalg_numeric;
extern crate cglinalg_core;
extern crate cglinalg_transform;
extern crate proptest;


use cglinalg_numeric::{
    SimdScalarSigned,
    SimdScalarFloat,
};
use cglinalg_core::{
    Point,
    Point2,
    Point3,
    Vector,
    Vector2,
    Vector3,
};
use cglinalg_transform::{
    Scale,
    Scale2,
    Scale3,
};
use approx::{
    relative_eq,
};

use proptest::prelude::*;


fn strategy_vector_signed_from_abs_range<S, const N: usize>(min_value: S, max_value: S) -> impl Strategy<Value = Vector<S, N>>
where
    S: SimdScalarSigned + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarSigned
    {
        min_value + (value % (max_value - min_value))
    }

    fn rescale_vector<S, const N: usize>(value: Vector<S, N>, min_value: S, max_value: S) -> Vector<S, N>
    where
        S: SimdScalarSigned
    {
        value.map(|element| rescale(element, min_value, max_value))
    }

    any::<[S; N]>().prop_map(move |array| {
        let vector = Vector::from(array);
        
        rescale_vector(vector, min_value, max_value)
    })
}

fn strategy_point_signed_from_abs_range<S, const N: usize>(min_value: S, max_value: S) -> impl Strategy<Value = Point<S, N>>
where
    S: SimdScalarSigned + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarSigned
    {
        min_value + (value % (max_value - min_value))
    }

    fn rescale_point<S, const N: usize>(value: Point<S, N>, min_value: S, max_value: S) -> Point<S, N>
    where
        S: SimdScalarSigned
    {
        value.map(|element| rescale(element, min_value, max_value))
    }

    any::<[S; N]>().prop_map(move |array| {
        let point = Point::from(array);
        
        rescale_point(point, min_value, max_value)
    })
}

fn strategy_vector_i32_any<const N: usize>() -> impl Strategy<Value = Vector<i32, N>> {
    let min_value = 0_i32;
    let max_value = 1_000_000_i32;

    strategy_vector_signed_from_abs_range(min_value, max_value)
}

fn strategy_point_i32_any<const N: usize>() -> impl Strategy<Value = Point<i32, N>> {
    let min_value = 0_i32;
    let max_value = 1_000_000_i32;

    strategy_point_signed_from_abs_range(min_value, max_value)
}

fn strategy_vector_f64_any<const N: usize>() -> impl Strategy<Value = Vector<f64, N>> {
    let min_value = 0_f64;
    let max_value = 1_000_000_f64;

    strategy_vector_signed_from_abs_range(min_value, max_value)
}

fn strategy_point_f64_any<const N: usize>() -> impl Strategy<Value = Point<f64, N>> {
    let min_value = 0_f64;
    let max_value = 1_000_000_f64;

    strategy_point_signed_from_abs_range(min_value, max_value)
}

fn strategy_scale_signed_from_abs_range<S, const N: usize>(min_value: S, max_value: S) -> impl Strategy<Value = Scale<S, N>>
where
    S: SimdScalarSigned + Arbitrary
{
    strategy_vector_signed_from_abs_range(min_value, max_value).prop_map(|vector| { 
        Scale::from_nonuniform_scale(&vector)
    })
}

fn strategy_scale_i32_any<const N: usize>() -> impl Strategy<Value = Scale<i32, N>> {
    let min_value = 1_i32;
    let max_value = 1_000_000_i32;
    
    strategy_scale_signed_from_abs_range(min_value, max_value)
}

fn strategy_scale_f64_any<const N: usize>() -> impl Strategy<Value = Scale<f64, N>> {
    let min_value = 1_f64;
    let max_value = 1_000_000_f64;

    strategy_scale_signed_from_abs_range(min_value, max_value)
}


fn prop_scale_composition_commutative<S, const N: usize>(s1: Scale<S, N>, s2: Scale<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    prop_assert_eq!(s1 * s2, s2 * s1);

    Ok(())
}

fn prop_scale_scale_inverse_commutative<S, const N: usize>(s: Scale<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert_eq!(s * s.inverse(), s.inverse() * s);

    Ok(())
}

fn prop_approx_scale_scale_inverse_identity<S, const N: usize>(s: Scale<S, N>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let identity = Scale::identity();

    prop_assert!(relative_eq!(s * s.inverse(), identity, epsilon = tolerance));
    prop_assert!(relative_eq!(s.inverse() * s, identity, epsilon = tolerance));

    Ok(())
}

fn prop_scale_composition_commutative_pointwise_point<S, const N: usize>(
    s1: Scale<S, N>,
    s2: Scale<S, N>,
    p: Point<S, N>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = (s1 * s2) * p;
    let rhs = (s2 * s1) * p;

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

fn prop_scale_composition_commutative_pointwise_vector<S, const N: usize>(
    s1: Scale<S, N>,
    s2: Scale<S, N>,
    v: Vector<S, N>
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = s1 * (s2 * v);
    let rhs = s2 * (s1 * v);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

fn prop_approx_scale_composition_commutative_pointwise_point<S, const N: usize>(
    s1: Scale<S, N>,
    s2: Scale<S, N>,
    p: Point<S, N>,
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = (s1 * s2) * p;
    let rhs = (s2 * s1) * p;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

fn prop_approx_scale_composition_commutative_pointwise_vector<S, const N: usize>(
    s1: Scale<S, N>,
    s2: Scale<S, N>,
    v: Vector<S, N>,
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = (s1 * s2) * v;
    let rhs = (s2 * s1) * v;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

fn prop_approx_scale_scale_inverse_identity_pointwise_point<S, const N: usize>(
    s: Scale<S, N>,
    p: Point<S, N>,
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = s * (s.inverse() * p);
    let rhs = s.inverse() * (s * p);

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

fn prop_approx_scale_scale_inverse_identity_pointwise_vector<S, const N: usize>(
    s: Scale<S, N>,
    v: Vector<S, N>,
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = s * (s.inverse() * v);
    let rhs = s.inverse() * (s * v);

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}


#[cfg(test)]
mod scale2_i32_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_scale_composition_commutative(s1 in super::strategy_scale_i32_any(), s2 in super::strategy_scale_i32_any()) {
            let s1: super::Scale2<i32> = s1;
            let s2: super::Scale2<i32> = s2;
            super::prop_scale_composition_commutative(s1, s2)?
        }
    }
}

#[cfg(test)]
mod scale2_f64_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_scale_composition_commutative(s1 in super::strategy_scale_f64_any(), s2 in super::strategy_scale_f64_any()) {
            let s1: super::Scale2<f64> = s1;
            let s2: super::Scale2<f64> = s2;
            super::prop_scale_composition_commutative(s1, s2)?
        }

        #[test]
        fn prop_scale_scale_inverse_identity(s in super::strategy_scale_f64_any()) {
            let s: super::Scale2<f64> = s;
            super::prop_scale_scale_inverse_commutative(s)?
        }
        
        #[test]
        fn prop_approx_scale_scale_inverse_identity(s in super::strategy_scale_f64_any()) {
            let s: super::Scale2<f64> = s;
            super::prop_approx_scale_scale_inverse_identity(s, 1e-10)?
        }
    }
}

#[cfg(test)]
mod scale2_i32_composition_pointwise_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_scale_composition_commutative_pointwise_point(
            s1 in super::strategy_scale_i32_any(), 
            s2 in super::strategy_scale_i32_any(),
            p in super::strategy_point_i32_any()
        ) {
            let s1: super::Scale2<i32> = s1;
            let s2: super::Scale2<i32> = s2;
            let p: super::Point2<i32> = p;
            super::prop_scale_composition_commutative_pointwise_point(s1, s2, p)?
        }

        #[test]
        fn prop_scale_composition_commutative_pointwise_vector(
            s1 in super::strategy_scale_i32_any(), 
            s2 in super::strategy_scale_i32_any(),
            v in super::strategy_vector_i32_any()
        ) {
            let s1: super::Scale2<i32> = s1;
            let s2: super::Scale2<i32> = s2;
            let v: super::Vector2<i32> = v;
            super::prop_scale_composition_commutative_pointwise_vector(s1, s2, v)?
        }
    }
}

#[cfg(test)]
mod scale2_f64_composition_pointwise_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_scale_composition_commutative_pointwise_point(
            s1 in super::strategy_scale_f64_any(), 
            s2 in super::strategy_scale_f64_any(),
            p in super::strategy_point_f64_any()
        ) {
            let s1: super::Scale2<f64> = s1;
            let s2: super::Scale2<f64> = s2;
            let p: super::Point2<f64> = p;
            super::prop_approx_scale_composition_commutative_pointwise_point(s1, s2, p, 1e-10)?
        }

        #[test]
        fn prop_approx_scale_composition_commutative_pointwise_vector(
            s1 in super::strategy_scale_f64_any(), 
            s2 in super::strategy_scale_f64_any(),
            v in super::strategy_vector_f64_any()
        ) {
            let s1: super::Scale2<f64> = s1;
            let s2: super::Scale2<f64> = s2;
            let v: super::Vector2<f64> = v;
            super::prop_approx_scale_composition_commutative_pointwise_vector(s1, s2, v, 1e-10)?
        }

        #[test]
        fn prop_approx_scale_scale_inverse_identity_pointwise_point(
            s in super::strategy_scale_f64_any(), 
            p in super::strategy_point_f64_any()
        ) {
            let s: super::Scale2<f64> = s;
            let p: super::Point2<f64> = p;
            super::prop_approx_scale_scale_inverse_identity_pointwise_point(s, p, 1e-10)?
        }

        #[test]
        fn prop_approx_scale_scale_inverse_identity_pointwise_vector(
            s in super::strategy_scale_f64_any(), 
            v in super::strategy_vector_f64_any()
        ) {
            let s: super::Scale2<f64> = s;
            let v: super::Vector2<f64> = v;
            super::prop_approx_scale_scale_inverse_identity_pointwise_vector(s, v, 1e-10)?
        }
    }
}

#[cfg(test)]
mod scale3_i32_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_scale_composition_commutative(s1 in super::strategy_scale_i32_any(), s2 in super::strategy_scale_i32_any()) {
            let s1: super::Scale3<i32> = s1;
            let s2: super::Scale3<i32> = s2;
            super::prop_scale_composition_commutative(s1, s2)?
        }
    }
}

#[cfg(test)]
mod scale3_f64_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_scale_composition_commutative(s1 in super::strategy_scale_f64_any(), s2 in super::strategy_scale_f64_any()) {
            let s1: super::Scale3<f64> = s1;
            let s2: super::Scale3<f64> = s2;
            super::prop_scale_composition_commutative(s1, s2)?
        }

        #[test]
        fn prop_scale_scale_inverse_identity(s in super::strategy_scale_f64_any()) {
            let s: super::Scale3<f64> = s;
            super::prop_scale_scale_inverse_commutative(s)?
        }

        #[test]
        fn prop_approx_scale_scale_inverse_identity(s in super::strategy_scale_f64_any()) {
            let s: super::Scale2<f64> = s;
            super::prop_approx_scale_scale_inverse_identity(s, 1e-10)?
        }
    }
}


#[cfg(test)]
mod scale3_i32_composition_pointwise_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_scale_composition_commutative_pointwise_point(
            s1 in super::strategy_scale_i32_any(), 
            s2 in super::strategy_scale_i32_any(),
            p in super::strategy_point_i32_any()
        ) {
            let s1: super::Scale3<i32> = s1;
            let s2: super::Scale3<i32> = s2;
            let p: super::Point3<i32> = p;
            super::prop_scale_composition_commutative_pointwise_point(s1, s2, p)?
        }

        #[test]
        fn prop_scale_composition_commutative_pointwise_vector(
            s1 in super::strategy_scale_i32_any(), 
            s2 in super::strategy_scale_i32_any(),
            v in super::strategy_vector_i32_any()
        ) {
            let s1: super::Scale3<i32> = s1;
            let s2: super::Scale3<i32> = s2;
            let v: super::Vector3<i32> = v;
            super::prop_scale_composition_commutative_pointwise_vector(s1, s2, v)?
        }
    }
}

#[cfg(test)]
mod scale3_f64_composition_pointwise_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_scale_composition_commutative_pointwise_point(
            s1 in super::strategy_scale_f64_any(), 
            s2 in super::strategy_scale_f64_any(),
            p in super::strategy_point_f64_any()
        ) {
            let s1: super::Scale3<f64> = s1;
            let s2: super::Scale3<f64> = s2;
            let p: super::Point3<f64> = p;
            super::prop_approx_scale_composition_commutative_pointwise_point(s1, s2, p, 1e-10)?
        }

        #[test]
        fn prop_approx_scale_composition_commutative_pointwise_vector(
            s1 in super::strategy_scale_f64_any(), 
            s2 in super::strategy_scale_f64_any(),
            v in super::strategy_vector_f64_any()
        ) {
            let s1: super::Scale3<f64> = s1;
            let s2: super::Scale3<f64> = s2;
            let v: super::Vector3<f64> = v;
            super::prop_approx_scale_composition_commutative_pointwise_vector(s1, s2, v, 1e-10)?
        }

        #[test]
        fn prop_approx_scale_scale_inverse_identity_pointwise_point(
            s in super::strategy_scale_f64_any(), 
            p in super::strategy_point_f64_any()
        ) {
            let s: super::Scale3<f64> = s;
            let p: super::Point3<f64> = p;
            super::prop_approx_scale_scale_inverse_identity_pointwise_point(s, p, 1e-10)?
        }

        #[test]
        fn prop_approx_scale_scale_inverse_identity_pointwise_vector(
            s in super::strategy_scale_f64_any(), 
            v in super::strategy_vector_f64_any()
        ) {
            let s: super::Scale3<f64> = s;
            let v: super::Vector3<f64> = v;
            super::prop_approx_scale_scale_inverse_identity_pointwise_vector(s, v, 1e-10)?
        }
    }
}

