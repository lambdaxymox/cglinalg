use crate::euler::Euler;
use crate::matrix::{
    Matrix3x3,
    Matrix4x4,
};
use cglinalg_numeric::SimdScalarFloat;
use cglinalg_trigonometry::{
    Angle,
    Radians,
};

impl<S> Matrix3x3<S>
where
    S: SimdScalarFloat,
{
    /// Construct a rotation matrix from a set of Euler angles.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Euler,
    /// # };
    /// # use cglinalg_trigonometry::Radians;
    /// # use core::f64;
    /// #
    /// let euler_angles = {
    ///     let roll = Radians(f64::consts::FRAC_PI_6);
    ///     let yaw = Radians(f64::consts::FRAC_PI_4);
    ///     let pitch = Radians(f64::consts::FRAC_PI_3);
    ///
    ///     Euler::new(roll, yaw, pitch)
    /// };
    /// let expected = {
    ///     let frac_1_sqrt_2 = 1_f64 / f64::sqrt(2_f64);
    ///     let frac_1_2 = 1_f64 / 2_f64;
    ///     let frac_sqrt_3_2 = f64::sqrt(3_f64) / 2_f64;
    ///
    ///     Matrix3x3::new(
    ///          frac_1_sqrt_2 * frac_1_2,
    ///          frac_sqrt_3_2 * frac_sqrt_3_2 + frac_1_2 * frac_1_sqrt_2 * frac_1_2,
    ///          frac_1_2 * frac_sqrt_3_2 - frac_sqrt_3_2 * frac_1_sqrt_2 * frac_1_2,
    ///
    ///         -frac_1_sqrt_2 * frac_sqrt_3_2,
    ///          frac_sqrt_3_2 * frac_1_2 - frac_1_2 * frac_1_sqrt_2 * frac_sqrt_3_2,
    ///          frac_1_2 * frac_1_2 + frac_sqrt_3_2 * frac_1_sqrt_2 * frac_sqrt_3_2,
    ///
    ///          frac_1_sqrt_2,
    ///         -frac_1_2 * frac_1_sqrt_2,
    ///          frac_sqrt_3_2 * frac_1_sqrt_2,
    ///     )
    /// };
    /// let result = Matrix3x3::from_euler_angles(&euler_angles);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_euler_angles<A>(euler_angles: &Euler<A>) -> Self
    where
        A: Angle + Into<Radians<S>>,
    {
        let euler_radians: Euler<Radians<S>> = Euler::new(
            euler_angles.x.into(),
            euler_angles.y.into(),
            euler_angles.z.into(),
        );

        euler_radians.to_matrix()
    }
}

impl<S> Matrix4x4<S>
where
    S: SimdScalarFloat,
{
    /// Construct an affine rotation matrix from a set of Euler angles.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Euler,
    /// #     Matrix4x4,
    /// # };
    /// # use cglinalg_trigonometry::Radians;
    /// # use core::f64;
    /// #
    /// let euler_angles = {
    ///     let roll = Radians(f64::consts::FRAC_PI_6);
    ///     let yaw = Radians(f64::consts::FRAC_PI_4);
    ///     let pitch = Radians(f64::consts::FRAC_PI_3);
    ///
    ///     Euler::new(roll, yaw, pitch)
    /// };
    /// let expected = {
    ///     let frac_1_sqrt_2 = 1_f64 / f64::sqrt(2_f64);
    ///     let frac_1_2 = 1_f64 / 2_f64;
    ///     let frac_sqrt_3_2 = f64::sqrt(3_f64) / 2_f64;
    ///
    ///     Matrix4x4::new(
    ///          frac_1_sqrt_2 * frac_1_2,
    ///          frac_sqrt_3_2 * frac_sqrt_3_2 + frac_1_2 * frac_1_sqrt_2 * frac_1_2,
    ///          frac_1_2 * frac_sqrt_3_2 - frac_sqrt_3_2 * frac_1_sqrt_2 * frac_1_2,
    ///          0_f64,
    ///
    ///         -frac_1_sqrt_2 * frac_sqrt_3_2,
    ///          frac_sqrt_3_2 * frac_1_2 - frac_1_2 * frac_1_sqrt_2 * frac_sqrt_3_2,
    ///          frac_1_2 * frac_1_2 + frac_sqrt_3_2 * frac_1_sqrt_2 * frac_sqrt_3_2,
    ///          0_f64,
    ///
    ///          frac_1_sqrt_2,
    ///         -frac_1_2 * frac_1_sqrt_2,
    ///          frac_sqrt_3_2 * frac_1_sqrt_2,
    ///          0_f64,
    ///
    ///          0_f64,
    ///          0_f64,
    ///          0_f64,
    ///          1_f64,
    ///     )
    /// };
    /// let result = euler_angles.to_affine_matrix();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_euler_angles<A>(euler_angles: &Euler<A>) -> Self
    where
        A: Angle + Into<Radians<S>>,
    {
        let euler_radians: Euler<Radians<S>> = Euler::new(
            euler_angles.x.into(),
            euler_angles.y.into(),
            euler_angles.z.into(),
        );

        euler_radians.to_affine_matrix()
    }
}
