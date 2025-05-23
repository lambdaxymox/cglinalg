use approx_cmp::relative_eq;
use cglinalg_core::{
    Const,
    DimMul,
    Point,
    Point2,
    Point3,
    ShapeConstraint,
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
    Rotation2,
    Rotation3,
    Similarity,
    Similarity2,
    Similarity3,
    Translation2,
    Translation3,
};
use cglinalg_trigonometry::Radians;

use proptest::prelude::*;

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

fn strategy_angle_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Radians<S>>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<S>().prop_map(move |value| {
        let sign_value = value.signum();
        let abs_value = value.abs();
        let value = sign_value * rescale(abs_value, min_value, max_value);

        Radians(value)
    })
}

fn strategy_similarity2_from_range<S>(
    min_scale: S,
    max_scale: S,
    min_angle: S,
    max_angle: S,
    min_distance: S,
    max_distance: S,
) -> impl Strategy<Value = Similarity2<S>>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, S, [S; 2])>().prop_map(move |(_scale, _angle, _vector)| {
        let scale = rescale(_scale, min_scale, max_scale);
        let translation = {
            let vector = Vector2::new(
                rescale(_vector[0], min_distance, max_distance),
                rescale(_vector[1], min_distance, max_distance),
            );

            Translation2::from_vector(&vector)
        };
        let rotation = {
            let angle = Radians(SimdScalarSigned::abs(rescale(_angle, min_angle, max_angle)));

            Rotation2::from_angle(angle)
        };

        Similarity2::from_parts(&translation, &rotation, scale)
    })
}

fn strategy_similarity3_from_range<S>(
    min_scale: S,
    max_scale: S,
    min_angle: S,
    max_angle: S,
    min_distance: S,
    max_distance: S,
) -> impl Strategy<Value = Similarity3<S>>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, S, [S; 3], [S; 3])>().prop_map(move |(_scale, _angle, _axis, _vector)| {
        let scale = rescale(_scale, min_scale, max_scale);
        let translation = {
            let vector = Vector3::new(
                rescale(_vector[0], min_distance, max_distance),
                rescale(_vector[1], min_distance, max_distance),
                rescale(_vector[2], min_distance, max_distance),
            );

            Translation3::from_vector(&vector)
        };
        let rotation = {
            let angle = Radians(SimdScalarSigned::abs(rescale(_angle, min_angle, max_angle)));
            let unnormalized_axis = {
                let axis_x = rescale(_axis[0], S::default_epsilon(), S::one());
                let axis_y = rescale(_axis[1], S::default_epsilon(), S::one());
                let axis_z = rescale(_axis[2], S::default_epsilon(), S::one());

                Vector3::new(axis_x, axis_y, axis_z)
            };
            let axis = Unit::from_value(unnormalized_axis);

            Rotation3::from_axis_angle(&axis, angle)
        };

        Similarity3::from_parts(&translation, &rotation, scale)
    })
}

fn strategy_vector_f64_any<const N: usize>() -> impl Strategy<Value = Vector<f64, N>> {
    let min_value = f64::sqrt(f64::EPSILON);
    let max_value = 1_000_000_f64;

    strategy_vector_signed_from_abs_range(min_value, max_value)
}

fn strategy_unit_vector_f64_any<const N: usize>() -> impl Strategy<Value = Unit<Vector<f64, N>>> {
    let min_value = f64::sqrt(f64::EPSILON);
    let max_value = 1_f64 + min_value;

    strategy_vector_signed_from_abs_range(min_value, max_value).prop_map(Unit::from_value)
}

fn strategy_point_f64_any<const N: usize>() -> impl Strategy<Value = Point<f64, N>> {
    let min_value = f64::sqrt(f64::EPSILON);
    let max_value = 1_000_000_f64;

    strategy_point_signed_from_abs_range(min_value, max_value)
}

fn strategy_similarity2_f64_any() -> impl Strategy<Value = Similarity2<f64>> {
    let min_scale = f64::EPSILON;
    let max_scale = 1_000_000_f64;
    let min_angle = 0_f64;
    let max_angle = 50_f64 * f64::two_pi();
    let min_distance = -1_000_000_f64;
    let max_distance = 1_000_000_f64;

    strategy_similarity2_from_range(min_scale, max_scale, min_angle, max_angle, min_distance, max_distance)
}

fn strategy_similarity3_f64_any() -> impl Strategy<Value = Similarity3<f64>> {
    let min_scale = f64::EPSILON;
    let max_scale = 1_000_000_f64;
    let min_angle = 0_f64;
    let max_angle = 50_f64 * f64::two_pi();
    let min_distance = -1_000_000_f64;
    let max_distance = 1_000_000_f64;

    strategy_similarity3_from_range(min_scale, max_scale, min_angle, max_angle, min_distance, max_distance)
}

fn strategy_angle_f64_any() -> impl Strategy<Value = Radians<f64>> {
    let min_value = 0_f64;
    let max_value = 50_f64 * f64::two_pi();

    strategy_angle_signed_from_abs_range(min_value, max_value)
}

/// Similarity transformations scale vector norms.
///
/// Given a similarity `M` and a vector `v`
/// ```text
/// norm(M * v) == abs(scale(M)) * norm(v)
/// ```
fn prop_approx_similarity_vector_scales_norm<S, const N: usize>(
    m: Similarity<S, N>,
    v: Vector<S, N>,
    tolerance: S,
    max_relative: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = (m * v).norm();
    let rhs = m.scale().abs() * v.norm();

    prop_assert!(relative_eq!(lhs, rhs, abs_diff_all <= tolerance, relative_all <= max_relative));

    Ok(())
}

/// All similarity transformations are invertible. Similarity transformations
/// commute with their inverses.
///
/// Given a similarity `M`
/// ```text
/// M * inverse(M) == inverse(M) * M == 1
/// ```
fn prop_approx_similarity_similarity_inverse<S, const N: usize, const NN: usize>(
    m: Similarity<S, N>,
    tolerance: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>,
{
    let identity = Similarity::identity();
    let lhs = m * m.inverse();
    let rhs = m.inverse() * m;

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));
    prop_assert!(relative_eq!(
        lhs,
        identity,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));
    prop_assert!(relative_eq!(
        rhs,
        identity,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/// The product of a similarity transformation with its inverse always acts as
/// the identity pointwise.
///
/// Given a similarity `M` and a point `p`
/// ```text
/// M * (inverse(M) * p) == p
/// inverse(M) * (M * p) == p
/// ```
fn prop_approx_similarity_similarity_inverse_pointwise_point<S, const N: usize>(
    m: Similarity<S, N>,
    p: Point<S, N>,
    tolerance: S,
    max_relative: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = m * (m.inverse() * p);
    let rhs = p;

    prop_assert!(relative_eq!(lhs, rhs, abs_diff_all <= tolerance, relative_all <= max_relative));

    /*
    let lhs = m.inverse() * (m * p);
    let rhs = p;

    prop_assert!(relative_eq!(lhs, rhs, abs_diff_all <= tolerance, relative_all <= max_relative));
    */
    Ok(())
}

/// The product of a similarity transformation with its inverse always acts as
/// the identity pointwise.
///
/// Given a similarity `M` and a vector `v`
/// ```text
/// M * (inverse(M) * v) == v
/// inverse(M) * (M * v) == v
/// ```
fn prop_approx_similarity_similarity_inverse_pointwise_vector<S, const N: usize>(
    m: Similarity<S, N>,
    v: Vector<S, N>,
    tolerance: S,
    max_relative: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let lhs = m * (m.inverse() * v);
    let rhs = v;

    prop_assert!(relative_eq!(lhs, rhs, abs_diff_all <= tolerance, relative_all <= max_relative));

    let lhs = m.inverse() * (m * v);
    let rhs = v;

    prop_assert!(relative_eq!(lhs, rhs, abs_diff_all <= tolerance, relative_all <= max_relative));

    Ok(())
}

/// In two dimensions, the composition of rotations is the same as
/// one rotation with the angles added up.
///
/// Given a similarity `M` that rotates the **xy-plane**, and angles `angle1` and `angle2`
/// ```text
/// M(angle1) * M(angle2) == M(angle1 + angle2)
/// ```
fn prop_approx_similarity2_composition_same_axis_equals_addition_of_angles<S>(
    angle1: Radians<S>,
    angle2: Radians<S>,
    tolerance: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let m1 = Similarity2::from_angle(angle1);
    let m2 = Similarity2::from_angle(angle2);
    let m3 = Similarity2::from_angle(angle1 + angle2);
    let lhs = m1 * m2;
    let rhs = m3;

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/// In three dimensions, the composition of rotations about the same axis is the
/// same as one rotation about the same axis with the angles added up.
///
/// Given a similarity `M` that rotates the plane perpendicular to the unit vector `axis`, and
/// angles `angle1` and `angle2`
/// ```text
/// M(axis, angle1) * M(axis, angle2) == M(axis, angle1 + angle2)
/// ```
fn prop_approx_similarity3_composition_same_axis_equals_addition_of_angles<S>(
    axis: Unit<Vector3<S>>,
    angle1: Radians<S>,
    angle2: Radians<S>,
    tolerance: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let m1 = Similarity3::from_axis_angle(&axis, angle1);
    let m2 = Similarity3::from_axis_angle(&axis, angle2);
    let m3 = Similarity3::from_axis_angle(&axis, angle1 + angle2);
    let lhs = m1 * m2;
    let rhs = m3;

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

#[cfg(test)]
mod similarity2_f64_invariant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_similarity_vector_scales_norm(m in super::strategy_similarity2_f64_any(), v in super::strategy_vector_f64_any()) {
            let m: super::Similarity2<f64> = m;
            let v: super::Vector2<f64> = v;
            super::prop_approx_similarity_vector_scales_norm(m, v, 1e-10, 1e-10)?
        }
    }
}

#[cfg(test)]
mod similarity2_f64_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_similarity_similarity_inverse(m in super::strategy_similarity2_f64_any()) {
            let m: super::Similarity2<f64> = m;
            super::prop_approx_similarity_similarity_inverse(m, 1e-8)?
        }

        #[test]
        fn prop_approx_similarity_similarity_inverse_pointwise_point(
            m in super::strategy_similarity2_f64_any(),
            p in super::strategy_point_f64_any()
        ) {
            let m: super::Similarity2<f64> = m;
            let p: super::Point2<f64> = p;
            super::prop_approx_similarity_similarity_inverse_pointwise_point(m, p, 1e-6, 1e-6)?
        }

        #[test]
        fn prop_approx_similarity_similarity_inverse_pointwise_vector(
            m in super::strategy_similarity2_f64_any(),
            v in super::strategy_vector_f64_any()
        ) {
            let r: super::Similarity2<f64> = m;
            let v: super::Vector2<f64> = v;
            super::prop_approx_similarity_similarity_inverse_pointwise_vector(r, v, 1e-6, 1e-6)?
        }

        #[test]
        fn prop_approx_similarity2_composition_same_axis_equals_addition_of_angles(
            angle1 in super::strategy_angle_f64_any(),
            angle2 in super::strategy_angle_f64_any(),
        ) {
            let angle1: super::Radians<f64> = angle1;
            let angle2: super::Radians<f64> = angle2;
            super::prop_approx_similarity2_composition_same_axis_equals_addition_of_angles(angle1, angle2, 1e-10)?
        }
    }
}

#[cfg(test)]
mod similarity3_f64_invariant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_similarity_vector_preserves_norm(m in super::strategy_similarity3_f64_any(), v in super::strategy_vector_f64_any()) {
            let m: super::Similarity3<f64> = m;
            let v: super::Vector3<f64> = v;
            super::prop_approx_similarity_vector_scales_norm(m, v, 1e-10, 1e-10)?
        }
    }
}

#[cfg(test)]
mod similarity3_f64_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_similarity_similarity_inverse(m in super::strategy_similarity3_f64_any()) {
            let m: super::Similarity3<f64> = m;
            super::prop_approx_similarity_similarity_inverse(m, 1e-8)?
        }

        #[test]
        fn prop_approx_similarity_similarity_inverse_pointwise_point(
            m in super::strategy_similarity3_f64_any(),
            p in super::strategy_point_f64_any()
        ) {
            let m: super::Similarity3<f64> = m;
            let p: super::Point3<f64> = p;
            super::prop_approx_similarity_similarity_inverse_pointwise_point(m, p, 1e-6, 1e-6)?
        }

        #[test]
        fn prop_approx_similarity_similarity_inverse_pointwise_vector(
            m in super::strategy_similarity3_f64_any(),
            v in super::strategy_vector_f64_any()
        ) {
            let r: super::Similarity3<f64> = m;
            let v: super::Vector3<f64> = v;
            super::prop_approx_similarity_similarity_inverse_pointwise_vector(r, v, 1e-6, 1e-6)?
        }

        #[test]
        fn prop_approx_similarity3_composition_same_axis_equals_addition_of_angles(
            axis in super::strategy_unit_vector_f64_any(),
            angle1 in super::strategy_angle_f64_any(),
            angle2 in super::strategy_angle_f64_any(),
        ) {
            let angle1: super::Radians<f64> = angle1;
            let angle2: super::Radians<f64> = angle2;
            super::prop_approx_similarity3_composition_same_axis_equals_addition_of_angles(axis, angle1, angle2, 1e-10)?
        }
    }
}
