extern crate approx_cmp;
extern crate cglinalg_core;
extern crate cglinalg_numeric;
extern crate cglinalg_transform;
extern crate proptest;


use approx_cmp::relative_eq;
use cglinalg_core::{
    Point,
    Point2,
    Point3,
    Unit,
    Vector,
    Vector2,
    Vector3,
};
use cglinalg_numeric::SimdScalarFloat;
use cglinalg_transform::{
    Reflection,
    Reflection2,
    Reflection3,
};

use proptest::prelude::*;


fn strategy_reflection2_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Reflection2<S>>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<([S; 2], [S; 2])>().prop_map(move |(_normal, _bias)| {
        let normal = {
            let sign_normal_0 = _normal[0].signum();
            let sign_normal_1 = _normal[1].signum();
            let abs_normal_0 = _normal[0].abs();
            let abs_normal_1 = _normal[1].abs();
            let normal_0 = sign_normal_0 * rescale(abs_normal_0, S::default_epsilon(), S::one());
            let normal_1 = sign_normal_1 * rescale(abs_normal_1, S::default_epsilon(), S::one());

            Unit::from_value(Vector2::new(normal_0, normal_1))
        };
        let bias = {
            let sign_bias_0 = _bias[0].signum();
            let sign_bias_1 = _bias[1].signum();
            let abs_bias_0 = _bias[0].abs();
            let abs_bias_1 = _bias[1].abs();
            let bias_0 = sign_bias_0 * rescale(abs_bias_0, min_value, max_value);
            let bias_1 = sign_bias_1 * rescale(abs_bias_1, min_value, max_value);

            Point2::new(bias_0, bias_1)
        };

        Reflection2::from_normal_bias(&normal, &bias)
    })
}

fn strategy_reflection3_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Reflection3<S>>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<([S; 3], [S; 3])>().prop_map(move |(_normal, _bias)| {
        let normal = {
            let sign_normal_0 = _normal[0].signum();
            let sign_normal_1 = _normal[1].signum();
            let sign_normal_2 = _normal[2].signum();
            let abs_normal_0 = _normal[0].abs();
            let abs_normal_1 = _normal[1].abs();
            let abs_normal_2 = _normal[2].abs();
            let normal_0 = sign_normal_0 * rescale(abs_normal_0, S::default_epsilon(), S::one());
            let normal_1 = sign_normal_1 * rescale(abs_normal_1, S::default_epsilon(), S::one());
            let normal_2 = sign_normal_2 * rescale(abs_normal_2, S::default_epsilon(), S::one());

            Unit::from_value(Vector3::new(normal_0, normal_1, normal_2))
        };
        let bias = {
            let sign_bias_0 = _bias[0].signum();
            let sign_bias_1 = _bias[1].signum();
            let sign_bias_2 = _bias[2].signum();
            let abs_bias_0 = _bias[0].abs();
            let abs_bias_1 = _bias[1].abs();
            let abs_bias_2 = _bias[2].abs();
            let bias_0 = sign_bias_0 * rescale(abs_bias_0, min_value, max_value);
            let bias_1 = sign_bias_1 * rescale(abs_bias_1, min_value, max_value);
            let bias_2 = sign_bias_2 * rescale(abs_bias_2, min_value, max_value);

            Point3::new(bias_0, bias_1, bias_2)
        };

        Reflection3::from_normal_bias(&normal, &bias)
    })
}

fn strategy_vector_signed_from_abs_range<S, const N: usize>(min_value: S, max_value: S) -> impl Strategy<Value = Vector<S, N>>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    fn rescale_vector<S, const N: usize>(value: Vector<S, N>, min_value: S, max_value: S) -> Vector<S, N>
    where
        S: SimdScalarFloat,
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
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    fn rescale_point<S, const N: usize>(value: Point<S, N>, min_value: S, max_value: S) -> Point<S, N>
    where
        S: SimdScalarFloat,
    {
        value.map(|element| rescale(element, min_value, max_value))
    }

    any::<[S; N]>().prop_map(move |array| {
        let point = Point::from(array);

        rescale_point(point, min_value, max_value)
    })
}

fn strategy_reflection2_f64_any() -> impl Strategy<Value = Reflection2<f64>> {
    let min_value = f64::sqrt(f64::sqrt(f64::EPSILON));
    let max_value = 1_000_000_f64;

    strategy_reflection2_signed_from_abs_range(min_value, max_value)
}

fn strategy_reflection3_f64_any() -> impl Strategy<Value = Reflection3<f64>> {
    let min_value = f64::sqrt(f64::sqrt(f64::EPSILON));
    let max_value = 1_000_000_f64;

    strategy_reflection3_signed_from_abs_range(min_value, max_value)
}

fn strategy_vector_f64_any<const N: usize>() -> impl Strategy<Value = Vector<f64, N>> {
    let min_value = f64::sqrt(f64::sqrt(f64::EPSILON));
    let max_value = 1_000_000_f64;

    strategy_vector_signed_from_abs_range(min_value, max_value)
}

fn strategy_point_f64_any<const N: usize>() -> impl Strategy<Value = Point<f64, N>> {
    let min_value = f64::sqrt(f64::sqrt(f64::EPSILON));
    let max_value = 1_000_000_f64;

    strategy_point_signed_from_abs_range(min_value, max_value)
}


/// The determinant of a reflection matrix is negative one.
///
/// Given a reflection `R`
/// ```text
/// det(matrix(R)) == -1
/// ```
fn prop_approx_reflection2_determinant_minus_one<S>(r: Reflection2<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = r.to_affine_matrix().determinant();
    let rhs = -S::one();

    prop_assert!(relative_eq!(lhs, rhs, abs_diff_all <= tolerance, relative_all <= S::default_epsilon()));

    Ok(())
}

/// The determinant of a reflection matrix is negative one.
///
/// Given a reflection `R`
/// ```text
/// det(matrix(R)) == -1
/// ```
fn prop_approx_reflection3_determinant_minus_one<S>(r: Reflection3<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = r.to_affine_matrix().determinant();
    let rhs = -S::one();

    prop_assert!(relative_eq!(lhs, rhs, abs_diff_all <= tolerance, relative_all <= S::default_epsilon()));

    Ok(())
}

/// Reflections preserve vector norms, i.e. every reflection is an isometry.
///
/// Given a reflection `r` and a vector `v`
/// ```text
/// norm(r * v) == norm(v)
/// ```
fn prop_approx_reflection_preserves_norm<S, const N: usize>(r: Reflection<S, N>, v: Vector<S, N>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = (r * v).norm();
    let rhs = v.norm();

    prop_assert!(relative_eq!(lhs, rhs, abs_diff_all <= tolerance, relative_all <= S::default_epsilon()));

    Ok(())
}

/// Every reflection is its own inverse, i.e. reflecting a point twice with
/// the same reflection matrix yields the original point.
///
/// Given a reflection `R` and a point `p`
/// ```text
/// R * (R * p) == p
/// ```
fn prop_approx_reflection_times_reflection_identity_pointwise_point<S, const N: usize>(
    r: Reflection<S, N>,
    p: Point<S, N>,
    tolerance: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = r * (r * p);
    let rhs = p;

    prop_assert!(relative_eq!(lhs, rhs, abs_diff_all <= tolerance, relative_all <= S::default_epsilon()));

    Ok(())
}

/// Every reflection is its own inverse, i.e. reflecting a vector twice with
/// the same reflection matrix yields the original vector.
///
/// Given a reflection `R` and a vector `v`
/// ```text
/// R * (R * v) == v
/// ```
fn prop_approx_reflection_times_reflection_identity_pointwise_vector<S, const N: usize>(
    r: Reflection<S, N>,
    v: Vector<S, N>,
    tolerance: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = r * (r * v);
    let rhs = v;

    prop_assert!(relative_eq!(lhs, rhs, abs_diff_all <= tolerance, relative_all <= S::default_epsilon()));

    Ok(())
}


#[cfg(test)]
mod reflection2_invariant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_reflection2_determinant_minus_one(r in super::strategy_reflection2_f64_any()) {
            let r: super::Reflection2<f64> = r;
            super::prop_approx_reflection2_determinant_minus_one(r, 1e-10)?
        }

        #[test]
        fn prop_approx_reflection_preserves_norm(
            r in super::strategy_reflection2_f64_any(),
            v in super::strategy_vector_f64_any(),
        ) {
            let r: super::Reflection2<f64> = r;
            let v: super::Vector2<f64> = v;
            super::prop_approx_reflection_preserves_norm(r, v, 1e-8)?
        }
    }
}

#[cfg(test)]
mod reflection2_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_reflection_times_reflection_identity_pointwise_point(
            r in super::strategy_reflection2_f64_any(),
            p in super::strategy_point_f64_any(),
        ) {
            let r: super::Reflection2<f64> = r;
            let p: super::Point2<f64> = p;
            super::prop_approx_reflection_times_reflection_identity_pointwise_point(r, p, 1e-8)?
        }

        #[test]
        fn prop_approx_reflection_times_reflection_identity_pointwise_vector(
            r in super::strategy_reflection2_f64_any(),
            v in super::strategy_vector_f64_any(),
        ) {
            let r: super::Reflection2<f64> = r;
            let v: super::Vector2<f64> = v;
            super::prop_approx_reflection_times_reflection_identity_pointwise_vector(r, v, 1e-8)?
        }
    }
}

#[cfg(test)]
mod reflection3_invariant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_reflection3_determinant_minus_one(r in super::strategy_reflection3_f64_any()) {
            let r: super::Reflection3<f64> = r;
            super::prop_approx_reflection3_determinant_minus_one(r, 1e-10)?
        }

        #[test]
        fn prop_approx_reflection_preserves_norm(
            r in super::strategy_reflection3_f64_any(),
            v in super::strategy_vector_f64_any(),
        ) {
            let r: super::Reflection3<f64> = r;
            let v: super::Vector3<f64> = v;
            super::prop_approx_reflection_preserves_norm(r, v, 1e-8)?
        }
    }
}

#[cfg(test)]
mod reflection3_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_reflection_times_reflection_identity_pointwise_point(
            r in super::strategy_reflection3_f64_any(),
            p in super::strategy_point_f64_any(),
        ) {
            let r: super::Reflection3<f64> = r;
            let p: super::Point3<f64> = p;
            super::prop_approx_reflection_times_reflection_identity_pointwise_point(r, p, 1e-8)?
        }

        #[test]
        fn prop_approx_reflection_times_reflection_identity_pointwise_vector(
            r in super::strategy_reflection3_f64_any(),
            v in super::strategy_vector_f64_any(),
        ) {
            let r: super::Reflection3<f64> = r;
            let v: super::Vector3<f64> = v;
            super::prop_approx_reflection_times_reflection_identity_pointwise_vector(r, v, 1e-8)?
        }
    }
}
