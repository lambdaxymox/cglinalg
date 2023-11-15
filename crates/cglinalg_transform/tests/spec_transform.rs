use cglinalg_core::{
    CanContract,
    CanExtend,
    CanMultiply,
    Const,
    DimAdd,
    DimMul,
    Matrix3x3,
    Matrix4x4,
    Point,
    Point2,
    Point3,
    ShapeConstraint,
    Vector,
    Vector2,
    Vector3,
};
use cglinalg_numeric::SimdScalarSigned;
use cglinalg_transform::{
    Transform,
    Transform2,
    Transform3,
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

fn strategy_transform2_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Transform2<S>>
where
    S: SimdScalarSigned + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarSigned,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<([[S; 2]; 2], [S; 2])>().prop_map(move |(matrix_array, translation_array)| {
        let mut result = Matrix3x3::identity();
        result[0][0] = rescale(matrix_array[0][0], min_value, max_value);
        result[0][1] = rescale(matrix_array[0][1], min_value, max_value);
        result[1][0] = rescale(matrix_array[1][0], min_value, max_value);
        result[1][1] = rescale(matrix_array[1][1], min_value, max_value);

        result[2][0] = rescale(translation_array[0], min_value, max_value);
        result[2][1] = rescale(translation_array[1], min_value, max_value);

        Transform2::from_matrix_unchecked(result)
    })
}

fn strategy_transform3_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Transform3<S>>
where
    S: SimdScalarSigned + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarSigned,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<([[S; 3]; 3], [S; 3])>().prop_map(move |(matrix_array, translation_array)| {
        let mut result = Matrix4x4::identity();
        result[0][0] = rescale(matrix_array[0][0], min_value, max_value);
        result[0][1] = rescale(matrix_array[0][1], min_value, max_value);
        result[0][2] = rescale(matrix_array[0][2], min_value, max_value);
        result[1][0] = rescale(matrix_array[1][0], min_value, max_value);
        result[1][1] = rescale(matrix_array[1][1], min_value, max_value);
        result[1][2] = rescale(matrix_array[1][2], min_value, max_value);
        result[2][0] = rescale(matrix_array[2][0], min_value, max_value);
        result[2][1] = rescale(matrix_array[2][1], min_value, max_value);
        result[2][2] = rescale(matrix_array[2][2], min_value, max_value);

        result[3][0] = rescale(translation_array[0], min_value, max_value);
        result[3][1] = rescale(translation_array[1], min_value, max_value);
        result[3][2] = rescale(translation_array[2], min_value, max_value);

        Transform3::from_matrix_unchecked(result)
    })
}

fn strategy_transform2_i32_any() -> impl Strategy<Value = Transform2<i32>> {
    let min_value = 1_i32;
    let max_value = 1_000_000_i32;

    strategy_transform2_signed_from_abs_range(min_value, max_value)
}

fn strategy_transform3_i32_any() -> impl Strategy<Value = Transform3<i32>> {
    let min_value = 1_i32;
    let max_value = 1_000_000_i32;

    strategy_transform3_signed_from_abs_range(min_value, max_value)
}


/// The composition of homogeneous transformations is associative over exact
/// scalars.
///
/// Given transformations `t1`, `t2`, and `t3`
/// ```text
/// (t1 * t2) * t3 == t1 * (t2 * t3)
/// ```
fn prop_transform_composition_associative<S, const N: usize, const NPLUS1: usize, const NP1NP1: usize>(
    t1: Transform<S, N, NPLUS1>,
    t2: Transform<S, N, NPLUS1>,
    t3: Transform<S, N, NPLUS1>,
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: CanMultiply<Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>,
{
    let lhs = (t1 * t2) * t3;
    let rhs = t1 * (t2 * t3);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The composition of transformations satisfies the following.
///
/// Given transformations `t1` and `t2`, and a point `p`
/// ```text
/// (t1 * t2) * p == t1 * (t2 * p)
/// ```
fn prop_transform_composition_pointwise_point<S, const N: usize, const NPLUS1: usize, const NP1NP1: usize>(
    t1: Transform<S, N, NPLUS1>,
    t2: Transform<S, N, NPLUS1>,
    p: Point<S, N>,
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: CanMultiply<Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>,
    ShapeConstraint: CanContract<Const<NPLUS1>, Const<N>>,
    ShapeConstraint: CanExtend<Const<N>, Const<NPLUS1>>,
{
    let lhs = (t1 * t2) * p;
    let rhs = t1 * (t2 * p);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}

/// The composition of transformations satisfies the following.
///
/// Given transformations `t1` and `t2`, and a vector `v`
/// ```text
/// (t1 * t2) * v == t1 * (t2 * v)
/// ```
fn prop_transform_composition_pointwise_vector<S, const N: usize, const NPLUS1: usize, const NP1NP1: usize>(
    t1: Transform<S, N, NPLUS1>,
    t2: Transform<S, N, NPLUS1>,
    v: Vector<S, N>,
) -> Result<(), TestCaseError>
where
    S: SimdScalarSigned,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<1>, Const<N>, Output = Const<NPLUS1>>,
    ShapeConstraint: CanMultiply<Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>, Const<NPLUS1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>,
    ShapeConstraint: DimMul<Const<NPLUS1>, Const<NPLUS1>, Output = Const<NP1NP1>>,
    ShapeConstraint: CanContract<Const<NPLUS1>, Const<N>>,
    ShapeConstraint: CanExtend<Const<N>, Const<NPLUS1>>,
{
    let lhs = (t1 * t2) * v;
    let rhs = t1 * (t2 * v);

    prop_assert_eq!(lhs, rhs);

    Ok(())
}


#[cfg(test)]
mod transform2_i32_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_transform_composition_associative(
            t1 in super::strategy_transform2_i32_any(),
            t2 in super::strategy_transform2_i32_any(),
            t3 in super::strategy_transform2_i32_any()
        ) {
            let t1: super::Transform2<i32> = t1;
            let t2: super::Transform2<i32> = t2;
            let t3: super::Transform2<i32> = t3;
            super::prop_transform_composition_associative(t1, t2, t3)?
        }
    }
}

#[cfg(test)]
mod transform2_i32_composition_pointwise_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_transform_composition_pointwise_point(
            t1 in super::strategy_transform2_i32_any(),
            t2 in super::strategy_transform2_i32_any(),
            p in super::strategy_point_i32_any()
        ) {
            let t1: super::Transform2<i32> = t1;
            let t2: super::Transform2<i32> = t2;
            let p: super::Point2<i32> = p;
            super::prop_transform_composition_pointwise_point(t1, t2, p)?
        }

        #[test]
        fn prop_transform_composition_pointwise_vector(
            t1 in super::strategy_transform2_i32_any(),
            t2 in super::strategy_transform2_i32_any(),
            v in super::strategy_vector_i32_any()
        ) {
            let t1: super::Transform2<i32> = t1;
            let t2: super::Transform2<i32> = t2;
            let v: super::Vector2<i32> = v;
            super::prop_transform_composition_pointwise_vector(t1, t2, v)?
        }
    }
}

#[cfg(test)]
mod transform3_i32_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_transform_composition_associative(
            t1 in super::strategy_transform3_i32_any(),
            t2 in super::strategy_transform3_i32_any(),
            t3 in super::strategy_transform3_i32_any()
        ) {
            let t1: super::Transform3<i32> = t1;
            let t2: super::Transform3<i32> = t2;
            let t3: super::Transform3<i32> = t3;
            super::prop_transform_composition_associative(t1, t2, t3)?
        }
    }
}

#[cfg(test)]
mod transform3_i32_composition_pointwise_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_transform_composition_pointwise_point(
            t1 in super::strategy_transform3_i32_any(),
            t2 in super::strategy_transform3_i32_any(),
            p in super::strategy_point_i32_any()
        ) {
            let t1: super::Transform3<i32> = t1;
            let t2: super::Transform3<i32> = t2;
            let p: super::Point3<i32> = p;
            super::prop_transform_composition_pointwise_point(t1, t2, p)?
        }

        #[test]
        fn prop_transform_composition_pointwise_vector(
            t1 in super::strategy_transform3_i32_any(),
            t2 in super::strategy_transform3_i32_any(),
            v in super::strategy_vector_i32_any()
        ) {
            let t1: super::Transform3<i32> = t1;
            let t2: super::Transform3<i32> = t2;
            let v: super::Vector3<i32> = v;
            super::prop_transform_composition_pointwise_vector(t1, t2, v)?
        }
    }
}
