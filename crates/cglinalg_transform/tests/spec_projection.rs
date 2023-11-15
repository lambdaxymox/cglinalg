use approx_cmp::relative_eq;
use cglinalg_core::{
    Point2,
    Point3,
    Vector2,
    Vector3,
};
use cglinalg_numeric::SimdScalarFloat;
use cglinalg_transform::PerspectiveFov3;
use cglinalg_trigonometry::{
    Angle,
    Radians,
};

use proptest::prelude::*;

use core::f64;


#[derive(Copy, Clone, Debug, PartialEq)]
struct PointLine<S> {
    start: Point3<S>,
    end: Point3<S>,
}

impl<S> PointLine<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    const fn new(start: Point3<S>, end: Point3<S>) -> Self {
        Self { start, end }
    }

    #[inline]
    const fn start(&self) -> Point3<S> {
        self.start
    }

    #[inline]
    const fn end(&self) -> Point3<S> {
        self.end
    }

    #[inline]
    fn interpolate(&self, t: S) -> Point3<S> {
        self.start + (self.end - self.start) * t
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct VectorLine<S> {
    start: Vector3<S>,
    end: Vector3<S>,
}

impl<S> VectorLine<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    const fn new(start: Vector3<S>, end: Vector3<S>) -> Self {
        Self { start, end }
    }

    #[inline]
    const fn start(&self) -> Vector3<S> {
        self.start
    }

    #[inline]
    const fn end(&self) -> Vector3<S> {
        self.end
    }

    #[inline]
    fn interpolate(&self, t: S) -> Vector3<S> {
        self.start + (self.end - self.start) * t
    }
}

fn strategy_perspectivefov3_from_range<S>(near: S, far: S) -> impl Strategy<Value = PerspectiveFov3<S>>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    fn choose_aspect_ratio<S>(i: usize) -> S
    where
        S: SimdScalarFloat,
    {
        let three: S = cglinalg_numeric::cast(3);
        let four: S = cglinalg_numeric::cast(4);
        let nine: S = cglinalg_numeric::cast(9);
        let ten: S = cglinalg_numeric::cast(10);
        let sixteen: S = cglinalg_numeric::cast(16);
        let twenty_one: S = cglinalg_numeric::cast(21);
        let aspects = [four / three, sixteen / nine, sixteen / ten, twenty_one / nine];

        aspects[i % aspects.len()]
    }

    any::<(S, usize)>().prop_map(move |(_vfov, _aspect_ratio_i)| {
        let vfov = Radians(rescale(_vfov, S::frac_pi_6(), S::pi()));
        let aspect_ratio = choose_aspect_ratio(_aspect_ratio_i);

        PerspectiveFov3::new(vfov, aspect_ratio, near, far)
    })
}

fn strategy_perspectivefov3_line_point<S>(near: S, far: S) -> impl Strategy<Value = (PerspectiveFov3<S>, PointLine<S>)>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    fn choose_aspect_ratio<S>(i: usize) -> S
    where
        S: SimdScalarFloat,
    {
        let three: S = cglinalg_numeric::cast(3);
        let four: S = cglinalg_numeric::cast(4);
        let nine: S = cglinalg_numeric::cast(9);
        let ten: S = cglinalg_numeric::cast(10);
        let sixteen: S = cglinalg_numeric::cast(16);
        let twenty_one: S = cglinalg_numeric::cast(21);
        let aspects = [four / three, sixteen / nine, sixteen / ten, twenty_one / nine];

        aspects[i % aspects.len()]
    }

    any::<(S, usize, [S; 2], [S; 2])>().prop_map(move |(_vfov, _aspect_ratio_i, _point_near, _point_far)| {
        let vfov = Radians(rescale(_vfov, S::frac_pi_6(), S::pi()));
        let aspect_ratio = choose_aspect_ratio(_aspect_ratio_i);
        let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
        let one = S::one();
        let two = one + one;
        let top_near = near.abs() * Radians::tan(vfov / two);
        let right_near = aspect_ratio * top_near;
        let top_far = far.abs() * Radians::tan(vfov / two);
        let right_far = aspect_ratio * top_far;
        let start = Point3::new(
            rescale(_point_near[0], -right_near, right_near),
            rescale(_point_near[1], -top_near, top_near),
            -near.abs(),
        );
        let end = Point3::new(
            rescale(_point_far[0], -right_far, right_far),
            rescale(_point_far[1], -top_far, top_far),
            -far.abs(),
        );
        let line = PointLine::new(start, end);

        (perspective, line)
    })
}

fn strategy_perspectivefov3_line_vector<S>(near: S, far: S) -> impl Strategy<Value = (PerspectiveFov3<S>, VectorLine<S>)>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    fn choose_aspect_ratio<S>(i: usize) -> S
    where
        S: SimdScalarFloat,
    {
        let three: S = cglinalg_numeric::cast(3);
        let four: S = cglinalg_numeric::cast(4);
        let nine: S = cglinalg_numeric::cast(9);
        let ten: S = cglinalg_numeric::cast(10);
        let sixteen: S = cglinalg_numeric::cast(16);
        let twenty_one: S = cglinalg_numeric::cast(21);
        let aspects = [four / three, sixteen / nine, sixteen / ten, twenty_one / nine];

        aspects[i % aspects.len()]
    }

    any::<(S, usize, [S; 2], [S; 2])>().prop_map(move |(_vfov, _aspect_ratio_i, _point_near, _point_far)| {
        let vfov = Radians(rescale(_vfov, S::frac_pi_6(), S::pi()));
        let aspect_ratio = choose_aspect_ratio(_aspect_ratio_i);
        let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
        let one = S::one();
        let two = one + one;
        let top_near = near.abs() * Radians::tan(vfov / two);
        let right_near = aspect_ratio * top_near;
        let top_far = far.abs() * Radians::tan(vfov / two);
        let right_far = aspect_ratio * top_far;
        let start = Vector3::new(
            rescale(_point_near[0], -right_near, right_near),
            rescale(_point_near[1], -top_near, top_near),
            -near.abs(),
        );
        let end = Vector3::new(
            rescale(_point_far[0], -right_far, right_far),
            rescale(_point_far[1], -top_far, top_far),
            -far.abs(),
        );
        let line = VectorLine::new(start, end);

        (perspective, line)
    })
}

fn strategy_perspectivefov3_lines_eye_point<S>(near: S, far: S) -> impl Strategy<Value = (PerspectiveFov3<S>, PointLine<S>)>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    fn choose_aspect_ratio<S>(i: usize) -> S
    where
        S: SimdScalarFloat,
    {
        let three: S = cglinalg_numeric::cast(3);
        let four: S = cglinalg_numeric::cast(4);
        let nine: S = cglinalg_numeric::cast(9);
        let ten: S = cglinalg_numeric::cast(10);
        let sixteen: S = cglinalg_numeric::cast(16);
        let twenty_one: S = cglinalg_numeric::cast(21);
        let aspects = [four / three, sixteen / nine, sixteen / ten, twenty_one / nine];

        aspects[i % aspects.len()]
    }

    any::<(S, usize, [S; 2], [S; 2])>().prop_map(move |(_vfov, _aspect_ratio_i, _point_near, _point_far)| {
        let vfov = Radians(rescale(_vfov, S::frac_pi_6(), S::pi()));
        let aspect_ratio = choose_aspect_ratio(_aspect_ratio_i);
        let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
        let one = S::one();
        let two = one + one;
        let top_far = far.abs() * Radians::tan(vfov / two);
        let right_far = aspect_ratio * top_far;
        let end = Point3::new(
            rescale(_point_far[0], -right_far, right_far),
            rescale(_point_far[1], -top_far, top_far),
            -far.abs(),
        );
        let start = end * (near.abs() / far.abs());
        let line = PointLine::new(start, end);

        (perspective, line)
    })
}

fn strategy_perspectivefov3_lines_eye_vector<S>(near: S, far: S) -> impl Strategy<Value = (PerspectiveFov3<S>, VectorLine<S>)>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    fn choose_aspect_ratio<S>(i: usize) -> S
    where
        S: SimdScalarFloat,
    {
        let three: S = cglinalg_numeric::cast(3);
        let four: S = cglinalg_numeric::cast(4);
        let nine: S = cglinalg_numeric::cast(9);
        let ten: S = cglinalg_numeric::cast(10);
        let sixteen: S = cglinalg_numeric::cast(16);
        let twenty_one: S = cglinalg_numeric::cast(21);
        let aspects = [four / three, sixteen / nine, sixteen / ten, twenty_one / nine];

        aspects[i % aspects.len()]
    }

    any::<(S, usize, [S; 2], [S; 2])>().prop_map(move |(_vfov, _aspect_ratio_i, _point_near, _point_far)| {
        let vfov = Radians(rescale(_vfov, S::frac_pi_6(), S::pi()));
        let aspect_ratio = choose_aspect_ratio(_aspect_ratio_i);
        let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
        let one = S::one();
        let two = one + one;
        let top_far = far.abs() * Radians::tan(vfov / two);
        let right_far = aspect_ratio * top_far;
        let end = Vector3::new(
            rescale(_point_far[0], -right_far, right_far),
            rescale(_point_far[1], -top_far, top_far),
            -far.abs(),
        );
        let start = end * (near.abs() / far.abs());
        let line = VectorLine::new(start, end);

        (perspective, line)
    })
}

fn strategy_point2_from_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Point2<S>>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<[S; 2]>().prop_map(move |array| Point2::new(rescale(array[0], min_value, max_value), rescale(array[1], min_value, max_value)))
}

fn strategy_vector2_from_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Vector2<S>>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<[S; 2]>().prop_map(move |array| Vector2::new(rescale(array[0], min_value, max_value), rescale(array[1], min_value, max_value)))
}

fn strategy_point2_any() -> impl Strategy<Value = Point2<f64>> {
    let min_value = -10_f64;
    let max_value = 10_f64;

    strategy_point2_from_range(min_value, max_value)
}

fn strategy_vector2_any() -> impl Strategy<Value = Vector2<f64>> {
    let min_value = -10_f64;
    let max_value = 10_f64;

    strategy_vector2_from_range(min_value, max_value)
}

fn strategy_interval<S>(min_value: S, max_value: S) -> impl Strategy<Value = S>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<S>().prop_map(move |value| rescale(value, min_value, max_value))
}


/// The perspective projection matrix maps every point on the near plane
/// to a `z` position of `-1` in normalized device coordinates.
///
/// Given a perspective projection `m` and a point `p` on the **xy-plane**
/// at `-near`
/// ```text
/// (m * p).z == -1
/// ```
fn prop_approx_perspective_projection_near_plane_point<S>(m: PerspectiveFov3<S>, p: Point2<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let p3 = p.extend(-m.near());
    let lhs = (m * p3).z;
    let rhs = -S::one();

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/// The perspective projection matrix maps every vector on the near plane
/// to a `z` position of `-1` in normalized device coordinates.
///
/// Given a perspective projection `m` and a vector `v` on the **xy-plane**
/// at `-near`
/// ```text
/// (m * v).z == -1
/// ```
fn prop_approx_perspective_projection_near_plane_vector<S>(m: PerspectiveFov3<S>, v: Vector2<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let v3 = v.extend(-m.near());
    let lhs = (m * v3).z;
    let rhs = -S::one();

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/// The perspective projection matrix maps every point on the far plane
/// to a `z` position of `1` in normalized device coordinates.
///
/// Given a perspective projection `m` and a point `p` on the **xy-plane**
/// at `-far`
/// ```text
/// (m * p).z == 1
/// ```
fn prop_approx_perspective_projection_far_plane_point<S>(m: PerspectiveFov3<S>, p: Point2<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let p3 = p.extend(-m.far());
    let lhs = (m * p3).z;
    let rhs = S::one();

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/// The perspective projection matrix maps every vector on the far plane
/// to a `z` position of `1` in normalized device coordinates.
///
/// Given a perspective projection `m` and a vector `v` on the **xy-plane**
/// at `-far`
/// ```text
/// (m * v).z == 1
/// ```
fn prop_approx_perspective_projection_far_plane_vector<S>(m: PerspectiveFov3<S>, v: Vector2<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let v3 = v.extend(-m.far());
    let lhs = (m * v3).z;
    let rhs = S::one();

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/// The perspective projection matrix preserves depth ordering of points.
///
/// Given a perspective projection `m` and points `p1`, `p2` such that
/// `p1.z < p2.z` inside the viewing frustum
/// ```text
/// (m * p1).z < (m * p2).z
/// ```
fn prop_perspective_projection_preserves_z_depth_ordering_point<S>(
    m: PerspectiveFov3<S>,
    l: PointLine<S>,
    t1: S,
    t2: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    prop_assume!(t1 >= S::zero() && t1 <= S::one());
    prop_assume!(t2 >= S::zero() && t2 <= S::one());

    let t_nearest = S::min(t1, t2);
    let t_farthest = S::max(t1, t2);

    let nearest = l.interpolate(t_nearest);
    let farthest = l.interpolate(t_farthest);

    let projected_nearest = m * nearest;
    let projected_farthest = m * farthest;

    prop_assert!(projected_nearest.z <= projected_farthest.z);

    Ok(())
}

/// The perspective projection matrix preserves depth ordering of vectors.
///
/// Given a perspective projection `m` and vectors `v1`, `v2` such that
/// `v1.z < v2.z` inside the viewing frustum
/// ```text
/// (m * v1).z < (m * v2).z
/// ```
fn prop_perspective_projection_preserves_z_depth_ordering_vector<S>(
    m: PerspectiveFov3<S>,
    l: VectorLine<S>,
    t1: S,
    t2: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    prop_assume!(t1 >= S::zero() && t1 <= S::one());
    prop_assume!(t2 >= S::zero() && t2 <= S::one());

    let t_nearest = S::min(t1, t2);
    let t_farthest = S::max(t1, t2);

    let nearest = l.interpolate(t_nearest);
    let farthest = l.interpolate(t_farthest);

    let projected_nearest = m * nearest;
    let projected_farthest = m * farthest;

    prop_assert!(projected_nearest.z <= projected_farthest.z);

    Ok(())
}

/// The perspective projection matrix maps lines through the origin in eye
/// space to lines parallel to the **z-axis** in normalized device coordinates.
fn prop_approx_perspective_projection_lines_through_eye_parallel_to_z_axis_point<S>(
    m: PerspectiveFov3<S>,
    l: PointLine<S>,
    tolerance: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let start = l.start();
    let end = l.end();
    let projected_start = m * start;
    let projected_end = m * end;
    let diff_start = projected_start.contract();
    let diff_end = projected_end.contract();

    prop_assert!(relative_eq!(
        diff_start,
        diff_end,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/// The perspective projection matrix maps lines through the origin in eye
/// space to lines parallel to the **z-axis** in normalized device coordinates.
fn prop_approx_perspective_projection_lines_through_eye_parallel_to_z_axis_vector<S>(
    m: PerspectiveFov3<S>,
    l: VectorLine<S>,
    tolerance: S,
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
{
    let start = l.start();
    let end = l.end();
    let projected_start = m * start;
    let projected_end = m * end;
    let diff_start = projected_start.contract();
    let diff_end = projected_end.contract();

    prop_assert!(relative_eq!(
        diff_start,
        diff_end,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}


#[cfg(test)]
mod projectionfov3_f64_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_perspective_projection_near_plane_point(
            m in super::strategy_perspectivefov3_from_range(0.1_f64, 100_f64),
            p in super::strategy_point2_any(),
        ) {
            let m: super::PerspectiveFov3<f64> = m;
            let p: super::Point2<f64> = p;
            super::prop_approx_perspective_projection_near_plane_point(m, p, 1e-10)?
        }

        #[test]
        fn prop_approx_perspective_projection_near_plane_vector(
            m in super::strategy_perspectivefov3_from_range(0.1_f64, 100_f64),
            v in super::strategy_vector2_any(),
        ) {
            let m: super::PerspectiveFov3<f64> = m;
            let v: super::Vector2<f64> = v;
            super::prop_approx_perspective_projection_near_plane_vector(m, v, 1e-10)?
        }

        #[test]
        fn prop_approx_perspective_projection_far_plane_point(
            m in super::strategy_perspectivefov3_from_range(0.1_f64, 100_f64),
            p in super::strategy_point2_any(),
        ) {
            let m: super::PerspectiveFov3<f64> = m;
            let p: super::Point2<f64> = p;
            super::prop_approx_perspective_projection_far_plane_point(m, p, 1e-10)?
        }

        #[test]
        fn prop_approx_perspective_projection_far_plane_vector(
            m in super::strategy_perspectivefov3_from_range(0.1_f64, 100_f64),
            v in super::strategy_vector2_any(),
        ) {
            let m: super::PerspectiveFov3<f64> = m;
            let v: super::Vector2<f64> = v;
            super::prop_approx_perspective_projection_far_plane_vector(m, v, 1e-10)?
        }

        #[test]
        fn prop_perspective_projection_preserves_z_depth_ordering_point(
            (m, l) in super::strategy_perspectivefov3_line_point(0.1_f64, 100_f64),
            t1 in super::strategy_interval(0_f64, 1_f64),
            t2 in super::strategy_interval(0_f64, 1_f64),
        ) {
            let m: super::PerspectiveFov3<f64> = m;
            let l: super::PointLine<f64> = l;
            let t1: f64 = t1;
            let t2: f64 = t2;
            super::prop_perspective_projection_preserves_z_depth_ordering_point(m, l, t1, t2)?
        }

        #[test]
        fn prop_perspective_projection_preserves_z_depth_ordering_vector(
            (m, l) in super::strategy_perspectivefov3_line_vector(0.1_f64, 100_f64),
            t1 in super::strategy_interval(0_f64, 1_f64),
            t2 in super::strategy_interval(0_f64, 1_f64),
        ) {
            let m: super::PerspectiveFov3<f64> = m;
            let t1: f64 = t1;
            let t2: f64 = t2;
            super::prop_perspective_projection_preserves_z_depth_ordering_vector(m, l, t1, t2)?
        }

        #[test]
        fn prop_approx_perspective_projection_lines_through_eye_parallel_to_z_axis_point(
            (m, l) in super::strategy_perspectivefov3_lines_eye_point(0.1_f64, 100_f64),
        ) {
            let m: super::PerspectiveFov3<f64> = m;
            let l: super::PointLine<f64> = l;
            super::prop_approx_perspective_projection_lines_through_eye_parallel_to_z_axis_point(m, l, 1e-10)?
        }

        #[test]
        fn prop_approx_perspective_projection_lines_through_eye_parallel_to_z_axis_vector(
            (m, l) in super::strategy_perspectivefov3_lines_eye_vector(0.1_f64, 100_f64),
        ) {
            let m: super::PerspectiveFov3<f64> = m;
            let l: super::VectorLine<f64> = l;
            super::prop_approx_perspective_projection_lines_through_eye_parallel_to_z_axis_vector(m, l, 1e-10)?
        }
    }
}
