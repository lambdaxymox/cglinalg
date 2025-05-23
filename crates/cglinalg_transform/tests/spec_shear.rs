use approx_cmp::{
    abs_diff_eq,
    relative_eq,
};
use cglinalg_core::{
    Point,
    Point2,
    Point3,
    Unit,
    Vector,
    Vector2,
    Vector3,
};
use cglinalg_numeric::{
    SimdScalarFloat,
    SimdScalarSigned,
};
use cglinalg_transform::{
    Shear,
    Shear2,
    Shear3,
};

use proptest::prelude::*;

fn strategy_vector_signed_from_abs_range<S, const N: usize>(min_value: S, max_value: S) -> impl Strategy<Value = Vector<S, N>>
where
    S: SimdScalarSigned + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarSigned,
    {
        min_value + (value % (max_value - min_value))
    }

    fn rescale_vector<S, const N: usize>(value: Vector<S, N>, min_value: S, max_value: S) -> Vector<S, N>
    where
        S: SimdScalarSigned,
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
    S: SimdScalarSigned + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarSigned,
    {
        min_value + (value % (max_value - min_value))
    }

    fn rescale_point<S, const N: usize>(value: Point<S, N>, min_value: S, max_value: S) -> Point<S, N>
    where
        S: SimdScalarSigned,
    {
        value.map(|element| rescale(element, min_value, max_value))
    }

    any::<[S; N]>().prop_map(move |array| {
        let point = Point::from(array);

        rescale_point(point, min_value, max_value)
    })
}

#[rustfmt::skip]
fn strategy_shear2_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Shear2<S>>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, [S; 2], [S; 2])>().prop_map(move |(_shear_factor, _origin, _direction)| {
        let shear_factor = rescale(_shear_factor, min_value, max_value);
        let origin = Point2::new(
            rescale(_origin[0], min_value, max_value),
            rescale(_origin[1], min_value, max_value),
        );
        let direction = Unit::from_value(Vector2::new(
            rescale(_direction[0], min_value, max_value),
            rescale(_direction[1], min_value, max_value),
        ));
        let normal = Unit::from_value(Vector2::new(-direction[1], direction[0]));

        assert!(relative_eq!(
            direction.norm(),
            S::one(),
            abs_diff_all <= cglinalg_numeric::cast(1e-14),
            relative_all <= S::default_epsilon(),
        ));
        assert!(relative_eq!(
            normal.norm(),
            S::one(),
            abs_diff_all <= cglinalg_numeric::cast(1e-14),
            relative_all <= S::default_epsilon(),
        ));
        assert!(abs_diff_eq!(direction.dot(&normal), S::zero(), abs_diff_all <= cglinalg_numeric::cast(1e-15)));

        Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal)
    })
}

fn strategy_shear3_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Shear3<S>>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, [S; 3], [S; 3], [S; 3])>().prop_map(move |(_shear_factor, _origin, _direction, _normal)| {
        let shear_factor = rescale(_shear_factor, min_value, max_value);
        let origin = Point3::new(
            rescale(_origin[0], min_value, max_value),
            rescale(_origin[1], min_value, max_value),
            rescale(_origin[2], min_value, max_value),
        );
        let direction = Unit::from_value(Vector3::new(
            rescale(_direction[0], min_value, max_value),
            rescale(_direction[1], min_value, max_value),
            rescale(_direction[2], min_value, max_value),
        ));
        let normal = {
            let _new_normal = if abs_diff_eq!(direction[2], S::zero(), abs_diff_all <= cglinalg_numeric::cast(1e-15)) {
                Unit::from_value(Vector3::new(S::zero(), S::zero(), _normal[2].signum() * S::one()))
            } else {
                let _new_normal_0 = rescale(_normal[0], min_value, max_value);
                let _new_normal_1 = rescale(_normal[1], min_value, max_value);
                let _new_normal_2 = -(direction[0] * _new_normal_0 + direction[1] * _new_normal_1) / direction[2];

                Unit::from_value(Vector3::new(_new_normal_0, _new_normal_1, _new_normal_2))
            };

            _new_normal
        };

        assert!(relative_eq!(
            direction.norm(),
            S::one(),
            abs_diff_all <= cglinalg_numeric::cast(1e-14),
            relative_all <= S::default_epsilon(),
        ));
        assert!(relative_eq!(
            normal.norm(),
            S::one(),
            abs_diff_all <= cglinalg_numeric::cast(1e-14),
            relative_all <= S::default_epsilon(),
        ));
        assert!(abs_diff_eq!(
            direction.dot(&normal),
            S::zero(),
            abs_diff_all <= cglinalg_numeric::cast(1e-15)
        ));

        Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal)
    })
}

fn strategy_shear_axis_signed_from_abs_range<F, S, const N: usize>(
    constructor: F,
    min_value: S,
    max_value: S,
) -> impl Strategy<Value = Shear<S, N>>
where
    S: SimdScalarSigned + Arbitrary,
    F: Fn(S) -> Shear<S, N>,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarSigned,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<S>().prop_map(move |_shear_factor| {
        let shear_factor = rescale(_shear_factor, min_value, max_value);

        constructor(shear_factor)
    })
}

fn strategy_shear2_f64_any() -> impl Strategy<Value = Shear2<f64>> {
    let min_value = 1_f64;
    let max_value = 1_000_000_f64;

    strategy_shear2_signed_from_abs_range(min_value, max_value)
}

fn strategy_shear3_f64_any() -> impl Strategy<Value = Shear3<f64>> {
    let min_value = 1_f64;
    let max_value = 1_000_000_f64;

    strategy_shear3_signed_from_abs_range(min_value, max_value)
}

fn strategy_shear_i32_any<F, const N: usize>(constructor: F) -> impl Strategy<Value = Shear<i32, N>>
where
    F: Fn(i32) -> Shear<i32, N>,
{
    let min_value = -100_000_i32;
    let max_value = 100_000_i32;

    strategy_shear_axis_signed_from_abs_range(constructor, min_value, max_value)
}

fn strategy_shear2_xy_i32_any() -> impl Strategy<Value = Shear2<i32>> {
    strategy_shear_i32_any(Shear2::from_shear_xy)
}

fn strategy_shear2_yx_i32_any() -> impl Strategy<Value = Shear2<i32>> {
    strategy_shear_i32_any(Shear2::from_shear_yx)
}

fn strategy_shear3_xy_i32_any() -> impl Strategy<Value = Shear3<i32>> {
    strategy_shear_i32_any(Shear3::from_shear_xy)
}

fn strategy_shear3_xz_i32_any() -> impl Strategy<Value = Shear3<i32>> {
    strategy_shear_i32_any(Shear3::from_shear_xz)
}

fn strategy_shear3_yx_i32_any() -> impl Strategy<Value = Shear3<i32>> {
    strategy_shear_i32_any(Shear3::from_shear_yx)
}

fn strategy_shear3_yz_i32_any() -> impl Strategy<Value = Shear3<i32>> {
    strategy_shear_i32_any(Shear3::from_shear_yz)
}

fn strategy_shear3_zx_i32_any() -> impl Strategy<Value = Shear3<i32>> {
    strategy_shear_i32_any(Shear3::from_shear_zx)
}

fn strategy_shear3_zy_i32_any() -> impl Strategy<Value = Shear3<i32>> {
    strategy_shear_i32_any(Shear3::from_shear_zy)
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

/// The trace of an affine shear matrix is always `N + 1` where `N` is the dimensionality
/// of the shearing transformation.
///
/// Given a shear matrix `s` in `N` dimensions.
/// ```text
/// trace(to_affine_matrix(s)) == N + 1
/// ```
fn prop_approx_shear2_trace<S>(s: Shear2<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = s.to_affine_matrix().trace();
    let rhs = cglinalg_numeric::cast(3_f64);

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/// The trace of an affine shear matrix is always `N + 1` where `N` is the dimensionality
/// of the shearing transformation.
///
/// Given a shear matrix `s` in `N` dimensions.
/// ```text
/// trace(to_affine_matrix(s)) == N + 1
/// ```
fn prop_approx_shear3_trace<S>(s: Shear3<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = s.to_affine_matrix().trace();
    let rhs = cglinalg_numeric::cast(4_f64);

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/// The determinant of a shearing transformation matrix is one.
///
/// Given a shear matrix `s`
/// ```text
/// determinant(to_affine_matrix(s)) == 1
/// ```
fn prop_approx_shear2_determinant<S>(s: Shear2<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = s.to_affine_matrix().determinant();
    let rhs = S::one();

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/*
/// The determinant of a shearing transformation matrix is one.
///
/// Given a shear matrix `s`
/// ```text
/// determinant(to_affine_matrix(s)) == 1
/// ```
fn prop_approx_shear3_determinant<S>(s: Shear3<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = s.to_affine_matrix().determinant();
    let rhs = S::one();

    prop_assert!(relative_eq!(lhs, rhs, abs_diff_all <= tolerance, relative_all <= S::default_epsilon()));

    Ok(())
}
*/

/// The matrix of a shearing transformation is asymmetric.
///
/// Given a shearing transformation `s`, of dimension `N`, let `m` be the matrix
/// of `s`. Then the matrix of `s` satisfies
/// ```text
/// exists i in 0..N. exists j in 0..N. m[i][j] != m[j][i].
/// ```
fn prop_shear2_matrix_asymmetric<S>(s: Shear2<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let matrix = s.to_affine_matrix();

    prop_assert!(!matrix.is_symmetric());

    Ok(())
}

/// The matrix of a shearing transformation is asymmetric.
///
/// Given a shearing transformation `s`, of dimension `N`, let `m` be the matrix
/// of `s`. Then the matrix of `s` satisfies
/// ```text
/// exists i in 0..N. exists j in 0..N. m[i][j] != m[j][i].
/// ```
fn prop_shear3_matrix_asymmetric<S>(s: Shear3<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let matrix = s.to_affine_matrix();

    prop_assert!(!matrix.is_symmetric());

    Ok(())
}

/// The shearing transformation satisfies the following property.
///
/// Given a shearing transformation `s` and a point `p`
/// ```text
/// inverse_apply_point(apply_point(p)) == p
/// ```
fn prop_shear_apply_inverse_apply_identity_point<S, const N: usize>(s: Shear<S, N>, p: Point<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned,
{
    let sheared_point = s.apply_point(&p);
    let lhs = s.inverse_apply_point(&sheared_point);
    let rhs = p;

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The shearing transformation satisfies the following property.
///
/// Given a shearing transformation `s` and a point `v`
/// ```text
/// inverse_apply_vector(apply_vector(v)) == v
/// ```
fn prop_shear_apply_inverse_apply_identity_vector<S, const N: usize>(s: Shear<S, N>, v: Vector<S, N>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned,
{
    let sheared_vector = s.apply_vector(&v);
    let lhs = s.inverse_apply_vector(&sheared_vector);
    let rhs = v;

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

#[cfg(test)]
mod shear2_f64_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_shear2_trace(s in super::strategy_shear2_f64_any()) {
            let s: super::Shear2<f64> = s;
            super::prop_approx_shear2_trace(s, 1e-8)?
        }

        #[test]
        fn prop_approx_shear2_determinant(s in super::strategy_shear2_f64_any()) {
            let s: super::Shear2<f64> = s;
            super::prop_approx_shear2_determinant(s, 1e-4)?
        }

        #[test]
        fn prop_shear2_matrix_asymmetric(s in super::strategy_shear2_f64_any()) {
            let s: super::Shear2<f64> = s;
            super::prop_shear2_matrix_asymmetric(s)?
        }
    }
}

#[cfg(test)]
mod shear3_f64_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_shear3_trace(s in super::strategy_shear3_f64_any()) {
            let s: super::Shear3<f64> = s;
            super::prop_approx_shear3_trace(s, 1e-8)?
        }

        #[test]
        fn prop_shear3_matrix_asymmetric(s in super::strategy_shear3_f64_any()) {
            let s: super::Shear3<f64> = s;
            super::prop_shear3_matrix_asymmetric(s)?
        }
    }
}

#[cfg(test)]
mod shear2_i32_apply_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_shear_xy_apply_inverse_apply_identity_point(
            s in super::strategy_shear2_xy_i32_any(),
            p in super::strategy_point_i32_any(),
        ) {
            let s: super::Shear2<i32> = s;
            let p: super::Point2<i32> = p;
            super::prop_shear_apply_inverse_apply_identity_point(s, p)?
        }

        #[test]
        fn prop_shear_xy_apply_inverse_apply_identity_vector(
            s in super::strategy_shear2_xy_i32_any(),
            v in super::strategy_vector_i32_any(),
        ) {
            let s: super::Shear2<i32> = s;
            let v: super::Vector2<i32> = v;
            super::prop_shear_apply_inverse_apply_identity_vector(s, v)?
        }

        #[test]
        fn prop_shear_yx_apply_inverse_apply_identity_point(
            s in super::strategy_shear2_yx_i32_any(),
            p in super::strategy_point_i32_any(),
        ) {
            let s: super::Shear2<i32> = s;
            let p: super::Point2<i32> = p;
            super::prop_shear_apply_inverse_apply_identity_point(s, p)?
        }

        #[test]
        fn prop_shear_yx_apply_inverse_apply_identity_vector(
            s in super::strategy_shear2_yx_i32_any(),
            v in super::strategy_vector_i32_any(),
        ) {
            let s: super::Shear2<i32> = s;
            let v: super::Vector2<i32> = v;
            super::prop_shear_apply_inverse_apply_identity_vector(s, v)?
        }
    }
}

#[cfg(test)]
mod shear3_i32_apply_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_shear_xy_apply_inverse_apply_identity_point(
            s in super::strategy_shear3_xy_i32_any(),
            p in super::strategy_point_i32_any(),
        ) {
            let s: super::Shear3<i32> = s;
            let p: super::Point3<i32> = p;
            super::prop_shear_apply_inverse_apply_identity_point(s, p)?
        }

        #[test]
        fn prop_shear_xy_apply_inverse_apply_identity_vector(
            s in super::strategy_shear3_xy_i32_any(),
            v in super::strategy_vector_i32_any(),
        ) {
            let s: super::Shear3<i32> = s;
            let v: super::Vector3<i32> = v;
            super::prop_shear_apply_inverse_apply_identity_vector(s, v)?
        }

        #[test]
        fn prop_shear_xz_apply_inverse_apply_identity_point(
            s in super::strategy_shear3_xz_i32_any(),
            p in super::strategy_point_i32_any(),
        ) {
            let s: super::Shear3<i32> = s;
            let p: super::Point3<i32> = p;
            super::prop_shear_apply_inverse_apply_identity_point(s, p)?
        }

        #[test]
        fn prop_shear_xz_apply_inverse_apply_identity_vector(
            s in super::strategy_shear3_xz_i32_any(),
            v in super::strategy_vector_i32_any(),
        ) {
            let s: super::Shear3<i32> = s;
            let v: super::Vector3<i32> = v;
            super::prop_shear_apply_inverse_apply_identity_vector(s, v)?
        }

        #[test]
        fn prop_shear_yx_apply_inverse_apply_identity_point(
            s in super::strategy_shear3_yx_i32_any(),
            p in super::strategy_point_i32_any(),
        ) {
            let s: super::Shear3<i32> = s;
            let p: super::Point3<i32> = p;
            super::prop_shear_apply_inverse_apply_identity_point(s, p)?
        }

        #[test]
        fn prop_shear_yx_apply_inverse_apply_identity_vector(
            s in super::strategy_shear3_yx_i32_any(),
            v in super::strategy_vector_i32_any(),
        ) {
            let s: super::Shear3<i32> = s;
            let v: super::Vector3<i32> = v;
            super::prop_shear_apply_inverse_apply_identity_vector(s, v)?
        }

        #[test]
        fn prop_shear_yz_apply_inverse_apply_identity_point(
            s in super::strategy_shear3_yz_i32_any(),
            p in super::strategy_point_i32_any(),
        ) {
            let s: super::Shear3<i32> = s;
            let p: super::Point3<i32> = p;
            super::prop_shear_apply_inverse_apply_identity_point(s, p)?
        }

        #[test]
        fn prop_shear_yz_apply_inverse_apply_identity_vector(
            s in super::strategy_shear3_yz_i32_any(),
            v in super::strategy_vector_i32_any(),
        ) {
            let s: super::Shear3<i32> = s;
            let v: super::Vector3<i32> = v;
            super::prop_shear_apply_inverse_apply_identity_vector(s, v)?
        }

        #[test]
        fn prop_shear_zx_apply_inverse_apply_identity_point(
            s in super::strategy_shear3_zx_i32_any(),
            p in super::strategy_point_i32_any(),
        ) {
            let s: super::Shear3<i32> = s;
            let p: super::Point3<i32> = p;
            super::prop_shear_apply_inverse_apply_identity_point(s, p)?
        }

        #[test]
        fn prop_shear_zx_apply_inverse_apply_identity_vector(
            s in super::strategy_shear3_zx_i32_any(),
            v in super::strategy_vector_i32_any(),
        ) {
            let s: super::Shear3<i32> = s;
            let v: super::Vector3<i32> = v;
            super::prop_shear_apply_inverse_apply_identity_vector(s, v)?
        }

        #[test]
        fn prop_shear_zy_apply_inverse_apply_identity_point(
            s in super::strategy_shear3_zy_i32_any(),
            p in super::strategy_point_i32_any(),
        ) {
            let s: super::Shear3<i32> = s;
            let p: super::Point3<i32> = p;
            super::prop_shear_apply_inverse_apply_identity_point(s, p)?
        }

        #[test]
        fn prop_shear_zy_apply_inverse_apply_identity_vector(
            s in super::strategy_shear3_zy_i32_any(),
            v in super::strategy_vector_i32_any(),
        ) {
            let s: super::Shear3<i32> = s;
            let v: super::Vector3<i32> = v;
            super::prop_shear_apply_inverse_apply_identity_vector(s, v)?
        }
    }
}
