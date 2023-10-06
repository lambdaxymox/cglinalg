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


#[derive(Copy, Clone, Debug, PartialEq)]
struct Triangle2<S> {
    vertices: [Vector2<S>; 3],
}

impl<S> Triangle2<S>
where
    S: SimdScalarFloat
{
    #[inline]
    const fn new(vertex_0: Vector2<S>, vertex_1: Vector2<S>, vertex_2: Vector2<S>) -> Self {
        Self {
            vertices: [vertex_0, vertex_1, vertex_2]
        }
    }

    #[inline]
    fn area(&self) -> S {
        let one = S::one();
        let two = one + one;
        let segment_0 = Vector2::extend(&(self.vertices[0] - self.vertices[1]), S::zero());
        let segment_1 = Vector2::extend(&(self.vertices[0] - self.vertices[2]), S::zero());
        let cross_area = Vector3::cross(&segment_0, &segment_1);

        cross_area.norm() / two
    }

    #[inline]
    fn map<T, F>(&self, mut op: F) -> Triangle2<T> 
    where 
        F: FnMut(Vector2<S>) -> Vector2<T>
    {
        Triangle2 {
            vertices: self.vertices.map(op),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Triangle3<S> {
    vertices: [Vector3<S>; 3],
}

impl<S> Triangle3<S>
where
    S: SimdScalarFloat
{
    #[inline]
    const fn new(vertex_0: Vector3<S>, vertex_1: Vector3<S>, vertex_2: Vector3<S>) -> Self {
        Self {
            vertices: [vertex_0, vertex_1, vertex_2]
        }
    }

    #[inline]
    fn area(&self) -> S {
        let one = S::one();
        let two = one + one;
        let cross_area = Vector3::cross(
            &(self.vertices[0] - self.vertices[1]),
            &(self.vertices[0] - self.vertices[2])
        );

        eprintln!("self = {:?}; cross_area = {:?}", self, cross_area);

        cross_area.norm() / two
    }

    #[inline]
    fn map<T, F>(&self, mut op: F) -> Triangle3<T> 
    where 
        F: FnMut(Vector3<S>) -> Vector3<T>
    {
        Triangle3 {
            vertices: self.vertices.map(op),
        }
    }
}


fn strategy_triangle2_from_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Triangle2<S>>
where
    S: SimdScalarFloat + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarSigned
    {
        min_value + (value % (max_value - min_value))
    }

    any::<[[S; 2]; 3]>().prop_map(move |_vertices| {
        let vertex_0 = Vector2::new(
            rescale(_vertices[0][0], min_value, max_value),
            rescale(_vertices[0][1], min_value, max_value)
        );
        let vertex_1 = Vector2::new(
            rescale(_vertices[1][0], min_value, max_value),
            rescale(_vertices[1][1], min_value, max_value),
        );
        let vertex_2 = Vector2::new(
            rescale(_vertices[2][0], min_value, max_value),
            rescale(_vertices[2][1], min_value, max_value)
        );

        Triangle2::new(vertex_0, vertex_1, vertex_2)
    })
}

fn strategy_triangle3_from_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Triangle3<S>>
where
    S: SimdScalarFloat + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarSigned
    {
        min_value + (value % (max_value - min_value))
    }

    any::<[[S; 3]; 3]>().prop_map(move |_vertices| {
        let vertex_0 = Vector3::new(
            rescale(_vertices[0][0], min_value, max_value),
            rescale(_vertices[0][1], min_value, max_value),
            rescale(_vertices[0][2], min_value, max_value)
        );
        let vertex_1 = Vector3::new(
            rescale(_vertices[1][0], min_value, max_value),
            rescale(_vertices[1][1], min_value, max_value),
            rescale(_vertices[1][2], min_value, max_value)
        );
        let vertex_2 = Vector3::new(
            rescale(_vertices[2][0], min_value, max_value),
            rescale(_vertices[2][1], min_value, max_value),
            rescale(_vertices[2][2], min_value, max_value)
        );

        Triangle3::new(vertex_0, vertex_1, vertex_2)
    })
}

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

fn strategy_triangle2_f64_any() -> impl Strategy<Value = Triangle2<f64>> {
    let min_value = 0_f64;
    let max_value = 1_000_000_f64;

    strategy_triangle2_from_range(min_value, max_value)
}

fn strategy_triangle3_f64_any() -> impl Strategy<Value = Triangle3<f64>> {
    let min_value = 0_f64;
    let max_value = 1_000_000_f64;

    strategy_triangle3_from_range(min_value, max_value)
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

/// The determinant of a shear matrix is one.
/// 
/// Given a shear matrix `S`
/// ```text
/// determinant(to_matrix(S)) == 1
/// ```
fn prop_shear2_determinant<S>(s: Shear2<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = s.to_matrix().determinant();
    let rhs = S::one();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The determinant of a shear matrix is one.
/// 
/// Given a shear matrix `S`
/// ```text
/// determinant(to_matrix(S)) == 1
/// ```
fn prop_shear3_determinant<S>(s: Shear3<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = s.to_matrix().determinant();
    let rhs = S::one();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The determinant of the composite of two shear matrices is one.
/// 
/// Given shear matrices `S1` and `S2`
/// ```text
/// determinant(to_matrix(S1 * S2)) == 1
/// ```
fn prop_shear2_composition_determinant<S>(s1: Shear2<S>, s2: Shear2<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = (s1.to_matrix() * s2.to_matrix()).determinant();
    let rhs = S::one();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The determinant of the composite of two shear matrices is one.
/// 
/// Given shear matrices `S1` and `S2`
/// ```text
/// determinant(to_matrix(S1 * S2)) == 1
/// ```
fn prop_shear3_composition_determinant<S>(s1: Shear3<S>, s2: Shear3<S>) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned
{
    let lhs = (s1.to_matrix() * s2.to_matrix()).determinant();
    let rhs = S::one();

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// Shearing transformations preserve the areas of polytopes. In particular,
/// they preserve the areas of triangles.
/// 
/// Given a shearing transformation `s` and a triangle `triangle`, let 
/// `sheared_triangle` be the result of applying `s` to the vertices of `triangle`.
/// Then
/// ```text
/// area(sheared_triangle) == area(triangle)
/// ```
fn prop_approx_shear2_preserves_triangle_areas<S>(
    s: Shear2<S>, 
    triangle: Triangle2<S>,
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let sheared_triangle = triangle.map(|v| s.apply_vector(&v));
    let lhs = sheared_triangle.area();
    let rhs = triangle.area();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

/// Shearing transformations preserve the areas of polytopes. In particular,
/// they preserve the areas of triangles.
/// 
/// Given a shearing transformation `s` and a triangle `triangle`, let 
/// `sheared_triangle` be the result of applying `s` to the vertices of `triangle`.
/// Then
/// ```text
/// area(sheared_triangle) == area(triangle)
/// ```
fn prop_approx_shear3_preserves_triangle_areas<S>(
    s: Shear3<S>, 
    triangle: Triangle3<S>,
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let sheared_triangle = triangle.map(|v| s.apply_vector(&v));
    let lhs = sheared_triangle.area();
    let rhs = triangle.area();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

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
mod shear2_i32_determinant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_shear2_determinant(s in super::strategy_shear2_i32_any()) {
            let s: super::Shear2<i32> = s;
            super::prop_shear2_determinant(s)?
        }

        #[test]
        fn prop_shear2_composition_determinant(
            s1 in super::strategy_shear2_i32_any(),
            s2 in super::strategy_shear2_i32_any()
        ) {
            let s1: super::Shear2<i32> = s1;
            let s2: super::Shear2<i32> = s2;
            super::prop_shear2_composition_determinant(s1, s2)?
        }
    }
}

#[cfg(test)]
mod shear2_f64_determinant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_shear2_determinant(s in super::strategy_shear2_f64_any()) {
            let s: super::Shear2<f64> = s;
            super::prop_shear2_determinant(s)?
        }

        #[test]
        fn prop_shear2_composition_determinant(
            s1 in super::strategy_shear2_f64_any(),
            s2 in super::strategy_shear2_f64_any()
        ) {
            let s1: super::Shear2<f64> = s1;
            let s2: super::Shear2<f64> = s2;
            super::prop_shear2_composition_determinant(s1, s2)?
        }
    }
}

#[cfg(test)]
mod shear2_f64_invariant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_shear2_preserves_triangle_areas(
            s in super::strategy_shear2_f64_any(),
            triangle in super::strategy_triangle2_f64_any()
        ) {
            let s: super::Shear2<f64> = s;
            let triangle: super::Triangle2<f64> = triangle;
            super::prop_approx_shear2_preserves_triangle_areas(s, triangle, 1e-10)?
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

#[cfg(test)]
mod shear3_i32_determinant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_shear3_determinant(s in super::strategy_shear3_i32_any()) {
            let s: super::Shear3<i32> = s;
            super::prop_shear3_determinant(s)?
        }

        #[test]
        fn prop_shear3_composition_determinant(
            s1 in super::strategy_shear3_i32_any(),
            s2 in super::strategy_shear3_i32_any()
        ) {
            let s1: super::Shear3<i32> = s1;
            let s2: super::Shear3<i32> = s2;
            super::prop_shear3_composition_determinant(s1, s2)?
        }
    }
}

#[cfg(test)]
mod shear3_f64_determinant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_shear3_determinant(s in super::strategy_shear3_f64_any()) {
            let s: super::Shear3<f64> = s;
            super::prop_shear3_determinant(s)?
        }

        #[test]
        fn prop_shear3_composition_determinant(
            s1 in super::strategy_shear3_f64_any(),
            s2 in super::strategy_shear3_f64_any()
        ) {
            let s1: super::Shear3<f64> = s1;
            let s2: super::Shear3<f64> = s2;
            super::prop_shear3_composition_determinant(s1, s2)?
        }
    }
}

#[cfg(test)]
mod shear3_f64_invariant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_shear3_preserves_triangle_areas(
            s in super::strategy_shear3_f64_any(),
            triangle in super::strategy_triangle3_f64_any()
        ) {
            let s: super::Shear3<f64> = s;
            let triangle: super::Triangle3<f64> = triangle;
            super::prop_approx_shear3_preserves_triangle_areas(s, triangle, 1e-10)?
        }
    }
}

