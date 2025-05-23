use approx_cmp::relative_eq;
use cglinalg_core::Euler;
use cglinalg_numeric::SimdScalarFloat;
use cglinalg_trigonometry::{
    Angle,
    Radians,
};

use proptest::prelude::*;

fn strategy_euler_angles_radians_from_range<S>(
    min_roll: S,
    max_roll: S,
    min_yaw: S,
    max_yaw: S,
    min_pitch: S,
    max_pitch: S,
) -> impl Strategy<Value = Euler<Radians<S>>>
where
    S: SimdScalarFloat + Arbitrary,
{
    fn rescale<S>(value: S, min_value: S, max_value: S) -> S
    where
        S: SimdScalarFloat,
    {
        min_value + (value % (max_value - min_value))
    }

    any::<(S, S, S)>().prop_map(move |(_roll, _yaw, _pitch)| {
        let roll = Radians(rescale(_roll, min_roll, max_roll));
        let yaw = Radians(rescale(_yaw, min_yaw, max_yaw));
        let pitch = Radians(rescale(_pitch, min_pitch, max_pitch));

        Euler::new(roll, yaw, pitch)
    })
}

fn strategy_euler_angles_radians_f64_any() -> impl Strategy<Value = Euler<Radians<f64>>> {
    let min_roll = -f64::pi();
    let max_roll = f64::pi();
    let min_yaw = -f64::frac_pi_2();
    let max_yaw = f64::frac_pi_2();
    let min_pitch = -f64::pi();
    let max_pitch = f64::pi();

    strategy_euler_angles_radians_from_range(min_roll, max_roll, min_yaw, max_yaw, min_pitch, max_pitch)
}

/// The matrix generated by a set of Euler angles should be an orthogonal rotation
/// matrix.
///
/// Given a set of Euler angles `euler`, we have
/// ```text
/// inverse(to_matrix(euler)) == transpose(to_matrix(euler))
/// ```
fn prop_approx_euler_matrix_inverse_equals_transpose<S, A>(euler_angles: Euler<A>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let matrix = euler_angles.to_matrix();
    let lhs = matrix.try_inverse().unwrap();
    let rhs = matrix.transpose();

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/// The matrix generated by a set of Euler angles should be an orthogonal rotation
/// matrix.
///
/// Given a set of Euler angles `euler`, we have
/// ```text
/// inverse(to_affine_matrix(euler)) == transpose(to_affine_matrix(euler))
/// ```
fn prop_approx_euler_affine_matrix_inverse_equals_transpose<S, A>(euler_angles: Euler<A>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let matrix = euler_angles.to_affine_matrix();
    let lhs = matrix.try_inverse().unwrap();
    let rhs = matrix.transpose();

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/// The rotation matrix generated by a set of Euler angles should have a
/// determinant of one.
///
/// Given a set of Euler angles `euler`, we have
/// ```text
/// determinant(to_matrix(euler)) == 1
/// ```
fn prop_approx_euler_matrix_determinant_equals_one<S, A>(euler_angles: Euler<A>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let matrix = euler_angles.to_matrix();
    let lhs = matrix.determinant();
    let rhs = S::one();

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

/// The affine rotation matrix generated by a set of Euler angles should have a
/// determinant of one.
///
/// Given a set of Euler angles `euler`, we have
/// ```text
/// determinant(to_affine_matrix(euler)) == 1
/// ```
fn prop_approx_euler_affine_matrix_determinant_equals_one<S, A>(euler_angles: Euler<A>, tolerance: S) -> Result<(), TestCaseError>
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    let matrix = euler_angles.to_affine_matrix();
    let lhs = matrix.determinant();
    let rhs = S::one();

    prop_assert!(relative_eq!(
        lhs,
        rhs,
        abs_diff_all <= tolerance,
        relative_all <= S::default_epsilon()
    ));

    Ok(())
}

#[cfg(test)]
mod euler_angles_f64_tests {
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_approx_euler_matrix_inverse_equals_transpose(
            euler_angles in super::strategy_euler_angles_radians_f64_any()
        ) {
            let euler_angles: super::Euler<super::Radians<f64>> = euler_angles;
            super::prop_approx_euler_matrix_inverse_equals_transpose(euler_angles, 1e-10)?
        }

        #[test]
        fn prop_approx_euler_affine_matrix_inverse_equals_transpose(
            euler_angles in super::strategy_euler_angles_radians_f64_any()
        ) {
            let euler_angles: super::Euler<super::Radians<f64>> = euler_angles;
            super::prop_approx_euler_affine_matrix_inverse_equals_transpose(euler_angles, 1e-10)?
        }

        #[test]
        fn prop_approx_euler_matrix_determinant_equals_one(
            euler_angles in super::strategy_euler_angles_radians_f64_any()
        ) {
            let euler_angles: super::Euler<super::Radians<f64>> = euler_angles;
            super::prop_approx_euler_matrix_determinant_equals_one(euler_angles, 1e-10)?
        }

        #[test]
        fn prop_approx_euler_affine_matrix_determinant_equals_one(
            euler_angles in super::strategy_euler_angles_radians_f64_any()
        ) {
            let euler_angles: super::Euler<super::Radians<f64>> = euler_angles;
            super::prop_approx_euler_affine_matrix_determinant_equals_one(euler_angles, 1e-10)?
        }
    }
}
