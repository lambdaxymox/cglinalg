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
    Shear,
    Shear2,
    Shear3,
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

fn strategy_shear2_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Shear2<S>>
where
    S: SimdScalarSigned + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarSigned
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, S)>().prop_map(move |(_shear_x_with_y, _shear_y_with_x)| {
        let shear_x_with_y = rescale(_shear_x_with_y, min_value, max_value);
        let shear_y_with_x = rescale(_shear_y_with_x, min_value, max_value);

        Shear2::from_shear(shear_x_with_y, shear_y_with_x)
    })
}

fn strategy_shear3_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Shear3<S>>
where
    S: SimdScalarSigned + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarSigned
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, S, S, S, S, S)>().prop_map(move |(
        _shear_x_with_y, _shear_x_with_z,
        _shear_y_with_x, _shear_y_with_z, 
        _shear_z_with_x, _shear_z_with_y
    )| {
        let shear_x_with_y = rescale(_shear_x_with_y, min_value, max_value);
        let shear_x_with_z = rescale(_shear_x_with_z, min_value, max_value);
        let shear_y_with_x = rescale(_shear_y_with_x, min_value, max_value);
        let shear_y_with_z = rescale(_shear_y_with_z, min_value, max_value);
        let shear_z_with_x = rescale(_shear_z_with_x, min_value, max_value);
        let shear_z_with_y = rescale(_shear_z_with_y, min_value, max_value);

        Shear3::from_shear(
            shear_x_with_y, shear_x_with_z, 
            shear_y_with_x, shear_y_with_z,
            shear_z_with_x, shear_z_with_y
        )
    })
}

fn strategy_shear2_i32_any() -> impl Strategy<Value = Shear2<i32>> {
    let min_value = 1_i32;
    let max_value = 1_000_000_i32;
    
    strategy_shear2_signed_from_abs_range(min_value, max_value)
}

fn strategy_shear2_f64_any() -> impl Strategy<Value = Shear2<f64>> {
    let min_value = 1_f64;
    let max_value = 1_000_000_f64;

    strategy_shear2_signed_from_abs_range(min_value, max_value)
}

fn strategy_shear3_i32_any() -> impl Strategy<Value = Shear3<i32>> {
    let min_value = 1_i32;
    let max_value = 1_000_000_i32;
    
    strategy_shear3_signed_from_abs_range(min_value, max_value)
}

fn strategy_shear3_f64_any() -> impl Strategy<Value = Shear3<f64>> {
    let min_value = 1_f64;
    let max_value = 1_000_000_f64;

    strategy_shear3_signed_from_abs_range(min_value, max_value)
}


/// The trace of a shear matrix is always `N` where `N` is the dimensionality
/// of the shear matrix.
/// 
/// Given a shear matrix `S` in `N` dimensions.
/// ```
/// trace(S) == N
/// ```
fn prop_shear_trace<S, const N: usize>(s: Shear<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = s.to_matrix().trace();
    let rhs = cglinalg_numeric::cast(N);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}


#[cfg(test)]
mod shear2_i32_trace_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_shear_trace(s in super::strategy_shear2_i32_any()) {
            let s: super::Shear2<i32> = s;
            super::prop_shear_trace(s)?
        }
    }
}

#[cfg(test)]
mod shear2_f64_trace_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_shear_trace(s in super::strategy_shear2_f64_any()) {
            let s: super::Shear2<f64> = s;
            super::prop_shear_trace(s)?
        }
    }
}

#[cfg(test)]
mod shear3_i32_trace_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_shear_trace(s in super::strategy_shear3_i32_any()) {
            let s: super::Shear3<i32> = s;
            super::prop_shear_trace(s)?
        }
    }
}

#[cfg(test)]
mod shear3_f64_trace_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_shear_trace(s in super::strategy_shear3_f64_any()) {
            let s: super::Shear3<f64> = s;
            super::prop_shear_trace(s)?
        }
    }
}

