extern crate cglinalg_numeric;
extern crate cglinalg_trigonometry;
extern crate cglinalg_core;
extern crate cglinalg_transform;
extern crate approx;
extern crate proptest;


use cglinalg_numeric::{
    SimdScalarSigned,
    SimdScalarFloat,
};
use cglinalg_trigonometry::{
    Radians,
};
use cglinalg_core::{
    ShapeConstraint,
    Const,
    DimMul,
    Point,
    Point2,
    Point3,
    Vector,
    Vector2,
    Vector3,
    Unit,
};
use cglinalg_transform::{
    Rotation,
    Rotation2,
    Rotation3,
};
use approx::{
    relative_eq,
};

use proptest::prelude::*;


fn strategy_rotation2_from_range<S>(min_angle: S, max_angle: S) -> impl Strategy<Value = Rotation2<S>>
where
    S: SimdScalarFloat + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarFloat
    {
        min_value + (value % (max_value - min_value))
    }

    any::<S>().prop_map(move |_angle| {
        let angle = Radians(SimdScalarSigned::abs(rescale(_angle, min_angle, max_angle)));

        Rotation2::from_angle(angle)
    })
}

fn strategy_rotation3_from_range<S>(min_angle: S, max_angle: S) -> impl Strategy<Value = Rotation3<S>>
where
    S: SimdScalarFloat + Arbitrary
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S 
    where
        S: SimdScalarFloat
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, S, S, S)>().prop_map(move |(_angle, _axis_x, _axis_y, _axis_z)| {
        let angle = Radians(SimdScalarSigned::abs(rescale(_angle, min_angle, max_angle)));
        let unnormalized_axis = {
            let axis_x = rescale(_axis_x, S::machine_epsilon(), S::one());
            let axis_y = rescale(_axis_y, S::machine_epsilon(), S::one());
            let axis_z = rescale(_axis_z, S::machine_epsilon(), S::one());

            Vector3::new(axis_x, axis_y, axis_z)
        };
        let axis = Unit::from_value(unnormalized_axis);

        Rotation3::from_axis_angle(&axis, angle)
    })
}

fn strategy_angle_signed_from_abs_range<S>(min_value: S, max_value: S) -> impl Strategy<Value = Radians<S>>
where
    S: SimdScalarFloat + Arbitrary
{
    fn rescale<S: SimdScalarFloat>(value: S, min_value: S, max_value: S) -> S {
        min_value + (value % (max_value - min_value))
    }

    any::<S>().prop_map(move |value| {
        let sign_value = value.signum();
        let abs_value = value.abs();
        let value = sign_value * rescale(abs_value, min_value, max_value);

        Radians(value)
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

fn strategy_vector_f64_any<const N: usize>() -> impl Strategy<Value = Vector<f64, N>> {
    let min_value = 0_f64;
    let max_value = 1_000_000_000_f64;

    strategy_vector_signed_from_abs_range(min_value, max_value)
}

fn strategy_unit_vector_f64_any<const N: usize>() -> impl Strategy<Value = Unit<Vector<f64, N>>> {
    let min_value = f64::sqrt(f64::EPSILON);
    let max_value = 1_f64 + min_value;

    strategy_vector_signed_from_abs_range(min_value, max_value).prop_map(Unit::from_value)
}

fn strategy_point_f64_any<const N: usize>() -> impl Strategy<Value = Point<f64, N>> {
    let min_value = 0_f64;
    let max_value = 1_000_000_000_f64;

    strategy_point_signed_from_abs_range(min_value, max_value)
}

fn strategy_rotation2_any() -> impl Strategy<Value = Rotation2<f64>> {
    let min_angle = 0_f64;
    let max_angle = f64::two_pi();

    strategy_rotation2_from_range(min_angle, max_angle)
}

fn strategy_rotation3_any() -> impl Strategy<Value = Rotation3<f64>> {
    let min_angle = 0_f64;
    let max_angle = 50_f64 * f64::two_pi();

    strategy_rotation3_from_range(min_angle, max_angle)
}

fn strategy_angle_any() -> impl Strategy<Value = Radians<f64>> {
    let min_value = 0_f64;
    let max_value = 50_f64 * f64::two_pi();

    strategy_angle_signed_from_abs_range(min_value, max_value)
}


fn prop_approx_rotation2_matrix_determinant_one<S>(r: Rotation2<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert!(relative_eq!(r.matrix().determinant(), S::one(), epsilon = tolerance));

    Ok(())
}

fn prop_approx_rotation3_matrix_determinant_one<S>(r: Rotation3<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    prop_assert!(relative_eq!(r.matrix().determinant(), S::one(), epsilon = tolerance));

    Ok(())
}

fn prop_approx_rotation_vector_preserves_norm<S, const N: usize>(
    r: Rotation<S, N>, 
    v: Vector<S, N>, 
    tolerance: S, 
    max_relative: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let lhs = (r * v).norm();
    let rhs = v.norm();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

fn prop_approx_rotation_rotation_inverse<S, const N: usize, const NN: usize>(r: Rotation<S, N>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    let lhs = r * r.inverse();
    let rhs = r.inverse() * r;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

fn prop_approx_rotation_rotation_inverse_pointwise_point<S, const N: usize, const NN: usize>(
    r: Rotation<S, N>, 
    p: Point<S, N>,
    tolerance: S,
    max_relative: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    let lhs = r * (r.inverse() * p);
    let rhs = p;
    
    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));
    
    let lhs = r.inverse() * (r * p);
    let rhs = p;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

fn prop_approx_rotation_rotation_inverse_pointwise_vector<S, const N: usize, const NN: usize>(
    r: Rotation<S, N>, 
    v: Vector<S, N>,
    tolerance: S,
    max_relative: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    let lhs = r * (r.inverse() * v);
    let rhs = v;
    
    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));
    
    let lhs = r.inverse() * (r * v);
    let rhs = v;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance, max_relative = max_relative));

    Ok(())
}

fn prop_approx_rotation2_composition_same_axis_equals_addition_of_angles<S>(
    angle1: Radians<S>, 
    angle2: Radians<S>,
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let r1 = Rotation2::from_angle(angle1);
    let r2 = Rotation2::from_angle(angle2);
    let r3 = Rotation2::from_angle(angle1 + angle2);
    let lhs = r1 * r2;
    let rhs = r3;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

fn prop_approx_rotation3_composition_same_axis_equals_addition_of_angles<S>(
    axis: Unit<Vector3<S>>,
    angle1: Radians<S>,
    angle2: Radians<S>,
    tolerance: S
) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let r1 = Rotation3::from_axis_angle(&axis, angle1);
    let r2 = Rotation3::from_axis_angle(&axis, angle2);
    let r3 = Rotation3::from_axis_angle(&axis, angle1 + angle2);
    let lhs = r1 * r2;
    let rhs = r3;

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}

fn prop_approx_rotation2_from_angle_angle<S>(angle: Radians<S>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat
{
    let r = Rotation2::from_angle(angle);
    let lhs = (r.angle() - angle) / Radians(S::two_pi());
    let rhs = lhs.round();

    prop_assert!(relative_eq!(lhs, rhs, epsilon = tolerance));

    Ok(())
}


#[cfg(test)]
mod rotation2_determinant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_rotation2_matrix_determinant_one(r in super::strategy_rotation2_any()) {
            let r: super::Rotation2<f64> = r;
            super::prop_approx_rotation2_matrix_determinant_one(r, 1e-10)?
        }
    }
}

#[cfg(test)]
mod rotation3_determinant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_rotation3_matrix_determinant_one(r in super::strategy_rotation3_any()) {
            let r: super::Rotation3<f64> = r;
            super::prop_approx_rotation3_matrix_determinant_one(r, 1e-10)?
        }
    }
}

#[cfg(test)]
mod rotation2_invariant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_rotation_vector_preserves_norm(r in super::strategy_rotation2_any(), v in super::strategy_vector_f64_any()) {
            let r: super::Rotation2<f64> = r;
            let v: super::Vector2<f64> = v;
            super::prop_approx_rotation_vector_preserves_norm(r, v, 1e-10, 1e-10)?
        }
    }
}

#[cfg(test)]
mod rotation3_invariant_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_rotation_vector_preserves_norm(r in super::strategy_rotation3_any(), v in super::strategy_vector_f64_any()) {
            let r: super::Rotation3<f64> = r;
            let v: super::Vector3<f64> = v;
            super::prop_approx_rotation_vector_preserves_norm(r, v, 1e-10, 1e-10)?
        }
    }
}

#[cfg(test)]
mod rotation2_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_rotation_rotation_inverse(r in super::strategy_rotation2_any()) {
            let r: super::Rotation2<f64> = r;
            super::prop_approx_rotation_rotation_inverse(r, 1e-10)?
        }

        #[test]
        fn prop_approx_rotation_rotation_inverse_pointwise_point(
            r in super::strategy_rotation2_any(),
            p in super::strategy_point_f64_any()
        ) {
            let r: super::Rotation2<f64> = r;
            let p: super::Point2<f64> = p;
            super::prop_approx_rotation_rotation_inverse_pointwise_point(r, p, 1e-6, 1e-6)?
        }

        #[test]
        fn prop_approx_rotation_rotation_inverse_pointwise_vector(
            r in super::strategy_rotation2_any(),
            v in super::strategy_vector_f64_any()
        ) {
            let r: super::Rotation2<f64> = r;
            let v: super::Vector2<f64> = v;
            super::prop_approx_rotation_rotation_inverse_pointwise_vector(r, v, 1e-6, 1e-6)?
        }

        #[test]
        fn prop_approx_rotation2_composition_same_axis_equals_addition_of_angles(
            angle1 in super::strategy_angle_any(),
            angle2 in super::strategy_angle_any(),
        ) {
            let angle1: super::Radians<f64> = angle1;
            let angle2: super::Radians<f64> = angle2;
            super::prop_approx_rotation2_composition_same_axis_equals_addition_of_angles(angle1, angle2, 1e-10)?
        }
    }
}

#[cfg(test)]
mod rotation3_composition_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_rotation_rotation_inverse(r in super::strategy_rotation3_any()) {
            let r: super::Rotation3<f64> = r;
            super::prop_approx_rotation_rotation_inverse(r, 1e-10)?
        }

        #[test]
        fn prop_approx_rotation_rotation_inverse_pointwise_point(
            r in super::strategy_rotation3_any(),
            p in super::strategy_point_f64_any()
        ) {
            let r: super::Rotation3<f64> = r;
            let p: super::Point3<f64> = p;
            super::prop_approx_rotation_rotation_inverse_pointwise_point(r, p, 1e-6, 1e-6)?
        }

        #[test]
        fn prop_approx_rotation_rotation_inverse_pointwise_vector(
            r in super::strategy_rotation3_any(),
            v in super::strategy_vector_f64_any()
        ) {
            let r: super::Rotation3<f64> = r;
            let v: super::Vector3<f64> = v;
            super::prop_approx_rotation_rotation_inverse_pointwise_vector(r, v, 1e-6, 1e-6)?
        }

        #[test]
        fn prop_approx_rotation3_composition_same_axis_equals_addition_of_angles(
            axis in super::strategy_unit_vector_f64_any(),
            angle1 in super::strategy_angle_any(),
            angle2 in super::strategy_angle_any(),
        ) {
            let axis: super::Unit<super::Vector3<f64>> = axis;
            let angle1: super::Radians<f64> = angle1;
            let angle2: super::Radians<f64> = angle2;
            super::prop_approx_rotation3_composition_same_axis_equals_addition_of_angles(axis, angle1, angle2, 1e-10)?
        }
    }
}

#[cfg(test)]
mod rotation2_constructor_props {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_rotation2_from_angle_angle(angle in super::strategy_angle_any()) {
            let angle: super::Radians<f64> = angle;
            super::prop_approx_rotation2_from_angle_angle(angle, 1e-10)?
        }
    }
}

