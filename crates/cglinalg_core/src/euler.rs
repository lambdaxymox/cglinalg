use cglinalg_numeric::{
    SimdScalarFloat,
};
use cglinalg_trigonometry::{
    Angle,
    Radians,
};
use crate::matrix::{
    Matrix3x3,
    Matrix4x4,
};
use crate::quaternion::{
    Quaternion,
};

use core::fmt;


/// A data type storing a set of Euler angles for representing a rotation about
/// an arbitrary axis in three dimensions.
///
/// The rotations are defined in the **ZYX** rotation order. That is, the Euler 
/// rotation applies a rotation to the **z-axis**, followed by the **y-axis**, and 
/// lastly the **x-axis**. The ranges of each axis are 
/// ```text
/// x in [-pi, pi]
/// y in [-pi/2, pi/2]
/// z in [-pi, pi]
/// ```
/// where each interval includes its endpoints. There is no one correct way to 
/// apply Euler angles, but the ZYX rotation order is one of the most commonly
/// use rotation orders used in computer graphics.
///
/// ## Note
/// 
/// Euler angles are prone to gimbal lock. Gimbal lock is the loss of one 
/// degree of freedom when rotations about two axes come into parallel alignment. 
/// In particular, when an object rotates on one axis and enters into parallel 
/// alignment with another rotation axis, the gimbal can no longer distinguish 
/// two of the rotation axes: when one tries to Euler rotate along one gimbal, the 
/// other one rotates by the same amount; one degree of freedom is lost.
/// Let's give a couple examples of Euler angles.
///
/// # Example (No Gimbal Lock)
/// 
/// The following example is a rotation without gimbal lock.
/// ```
/// # use cglinalg_trigonometry::{
/// #     Degrees,
/// # };
/// # use cglinalg_core::{
/// #     EulerAngles,
/// #     Matrix3x3,
/// # };
/// # use approx::{
/// #     assert_relative_eq,
/// # };
///
/// let roll = Degrees(45_f64);
/// let yaw = Degrees(30_f64);
/// let pitch = Degrees(15_f64);
/// let euler = EulerAngles::new(roll, yaw, pitch);
///
/// let c0r0 =  (1_f64 / 4_f64) * f64::sqrt(3_f64 / 2_f64) * (1_f64 + f64::sqrt(3_f64));
/// let c0r1 =  (1_f64 / 8_f64) * (3_f64 * f64::sqrt(3_f64) - 1_f64);
/// let c0r2 =  (1_f64 / 8_f64) * (f64::sqrt(3_f64) - 3_f64);
/// let c1r0 = -(1_f64 / 4_f64) * f64::sqrt(3_f64 / 2_f64) * (f64::sqrt(3_f64) - 1_f64);
/// let c1r1 =  (1_f64 / 8_f64) * (3_f64 + f64::sqrt(3_f64));
/// let c1r2 =  (1_f64 / 8_f64) * (3_f64 * f64::sqrt(3_f64) + 1_f64);
/// let c2r0 =  1_f64 / 2_f64;
/// let c2r1 = -(1_f64 / 2_f64) * f64::sqrt(3_f64 / 2_f64);
/// let c2r2 =  (1_f64 / 2_f64) * f64::sqrt(3_f64 / 2_f64);
/// let expected = Matrix3x3::new(
///     c0r0, c0r1, c0r2,
///     c1r0, c1r1, c1r2,
///     c2r0, c2r1, c2r2
/// );
/// let result = Matrix3x3::from(euler); 
///
/// assert_relative_eq!(result, expected, epsilon = 1e-8);
/// ```
/// 
/// # Example (Gimbal Lock)
/// 
/// An Euler rotation can be represented as a product three rotations. We are using the ZYX
/// rotation application order. 
/// ```text
/// R(roll, yaw, pitch) == R_x(roll) * R_y(yaw) * R_z(pitch)
/// ```
/// The corresponding rotation matrices are
/// ```text
///               | 1   0            0         |
/// R_x(roll)  := | 0   cos(roll)   -sin(roll) |
///               | 0   sin(rol)     cos(roll) |
/// 
///               |  cos(yaw)   0   sin(yaw) |
/// R_y(yaw)   := |  0          1   0        |
///               | -sin(yaw)   0   cos(yaw) |
///
///               | cos(pitch)   -sin(pitch)   0 |
/// R_z(pitch) := | sin(pitch)    cos(pitch)   0 |
///               | 0             0            1 |
/// ```
/// Let's examine what happens when the yaw angle is `pi / 2`. The cosine of 
/// and sine of this angle are `cos(pi / 2) == 0` and `sin(pi / 2) == 1`. 
/// Plugging this into the rotation equations gives the rotation matrix
/// ```text
///      | 1   0            0         |   |  0   0   1 |   | cos(pitch)   -sin(pitch)   0 |
/// R == | 0   cos(roll)   -sin(roll) | * |  0   1   0 | * | sin(pitch)    cos(pitch)   0 |
///      | 0   sin(roll)    cos(roll) |   | -1   0   0 |   | 0             0            1 |
///
///      | 1    0            0         |   | cos(pitch)   -sin(pitch)   0 |
///   == | 0    sin(roll)    cos(roll) | * | sin(pitch)    cos(pitch)   0 |
///      | 0   -cos(roll)    sin(roll) |   | 0             0            1 |
///
///      |  0                                              0                                           1 |
///   == |  sin(roll)*cos(pitch) + cos(roll)*sin(pitch)   -sin(roll)*sin(pitch) + cos(roll)*cos(pitch) 0 |
///      | -cos(roll)*cos(pitch) + sin(roll)*sin(pitch)    cos(roll)*sin(pitch) + sin(roll)*cos(pitch) 0 |
///
///      |  0                   0                 1 |
///   == |  sin(roll + pitch)   cos(roll + pitch) 0 |
///      | -cos(roll + pitch)   sin(roll + pitch) 0 |
/// ```
/// Changing either the values of the `pitch` or the `roll` has the same 
/// effect: it rotates an object about the **z-axis**. We have lost the ability 
/// to roll about the **x-axis**. Let's illustrate this effect with some code.
/// ```
/// # use cglinalg_trigonometry::{
/// #     Degrees,
/// # };
/// # use cglinalg_core::{
/// #    EulerAngles,
/// #    Matrix3x3,
/// # };
/// # use approx::{
/// #    assert_ulps_eq,
/// # };
/// #
/// // Gimbal lock the x-axis.
/// let roll = Degrees(45_f64);
/// let yaw = Degrees(90_f64);
/// let pitch = Degrees(15_f64);
/// let euler = EulerAngles::new(roll, yaw, pitch);
/// let matrix_z_locked = Matrix3x3::from(euler);
/// 
/// assert_ulps_eq!(matrix_z_locked.c0r0, 0_f64);
/// assert_ulps_eq!(matrix_z_locked.c1r0, 0_f64);
/// assert_ulps_eq!(matrix_z_locked.c2r0, 1_f64);
/// assert_ulps_eq!(matrix_z_locked.c2r1, 0_f64);
/// assert_ulps_eq!(matrix_z_locked.c2r2, 0_f64);
/// 
/// // Attempt to roll in the gimbal locked state.
/// let euler_roll = EulerAngles::new(Degrees(15_f64), Degrees(0_f64), Degrees(0_f64));
/// let matrix_roll = Matrix3x3::from(euler_roll);
/// let matrix = matrix_roll * matrix_z_locked;
///
/// // But the matrix is still gimbal locked.
/// assert_ulps_eq!(matrix.c0r0, 0_f64);
/// assert_ulps_eq!(matrix.c1r0, 0_f64);
/// assert_ulps_eq!(matrix.c2r0, 1_f64);
/// assert_ulps_eq!(matrix.c2r1, 0_f64);
/// assert_ulps_eq!(matrix.c2r2, 0_f64);
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EulerAngles<A> {
    /// The rotation angle about the **x-axis** in the **yz-plane**. This is also 
    /// known as the **roll** angle.
    pub x: A,
    /// The rotation angle about the **y-axis** in the **zx-plane**. This is also 
    /// known as the **yaw** angle.
    pub y: A,
    /// The rotation angle about the **z-axis** in the **xy-plane**. This is also
    /// called the **pitch** angle.
    pub z: A,
}

impl<A> EulerAngles<A> {
    /// Construct a new set of Euler angles.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     EulerAngles,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// #
    /// let euler_angles = EulerAngles::new(
    ///     Radians(1_f64), 
    ///     Radians(2_f64), 
    ///     Radians(3_f64)
    /// );
    /// 
    /// assert_eq!(euler_angles.x, Radians(1_f64));
    /// assert_eq!(euler_angles.y, Radians(2_f64));
    /// assert_eq!(euler_angles.z, Radians(3_f64));
    /// ```
    #[inline]
    pub const fn new(x: A, y: A, z: A) -> Self {
        Self { x, y, z }
    }
}

impl<S, A> EulerAngles<A> 
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    /// Construct a zero element of the set of Euler angles.
    ///
    /// The zero element is the element where each Euler angle is zero.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     EulerAngles,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// #
    /// let euler_angles: EulerAngles<Radians<f64>> = EulerAngles::zero();
    /// 
    /// assert!(euler_angles.is_zero());
    /// assert!(euler_angles.x.is_zero());
    /// assert!(euler_angles.y.is_zero());
    /// assert!(euler_angles.z.is_zero());
    /// ```
    #[inline]
    pub fn zero() -> Self {
        EulerAngles::new(A::zero(), A::zero(), A::zero())
    }
    
    /// Test whether an Euler angle is self.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     EulerAngles,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// #
    /// let euler_angles = EulerAngles::new(
    ///     Radians(0.1_f64), 
    ///     Radians(0.2_f64), 
    ///     Radians(0.3_f64)
    /// );
    /// 
    /// assert!(!euler_angles.is_zero());
    /// 
    /// let euler_angles: EulerAngles<Radians<f64>> = EulerAngles::zero();
    /// 
    /// assert!(euler_angles.is_zero());
    /// ```
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
    }

    /// Construct a rotation matrix from a set of Euler angles.
    ///
    /// A set of Euler angles describes an arbitrary rotation as a sequence
    /// of three axial rotations: one for each axis in thee dimensions 
    /// (`x`, `y`, `z`). The rotation matrix described by Euler angles can be
    /// decomposed into a product of rotation matrices about each axis: let 
    /// `R_x(roll)`, `R_y(yaw)`, and `R_z(pitch)` denote the rotations about 
    /// the **x-axis**, **y-axis**, and **z-axis**, respectively. The Euler rotation
    /// is decomposed as follows:
    /// ```text
    /// R(roll, yaw, pitch) == R_x(roll) * R_y(yaw) * R_z(pitch)
    /// ```
    /// The corresponding rotation matrices are
    /// ```text
    ///               | 1   0            0         |
    /// R_x(roll)  := | 0   cos(roll)   -sin(roll) |
    ///               | 0   sin(rol)     cos(roll) |
    /// 
    ///               |  cos(yaw)   0   sin(yaw) |
    /// R_y(yaw)   := |  0          1   0        |
    ///               | -sin(yaw)   0   cos(yaw) |
    ///
    ///               | cos(pitch)   -sin(pitch)   0 |
    /// R_z(pitch) := | sin(pitch)    cos(pitch)   0 |
    ///               | 0             0            1 |
    /// ```
    /// Multiplying out the axial rotations yields the following rotation matrix.
    /// ```text
    ///                        | m[0, 0]   m[1, 0]   m[2, 0] |
    /// R(roll, yaw, pitch) == | m[0, 1]   m[1, 1]   m[2, 1] |
    ///                        | m[0, 2]   m[1, 2]   m[2, 2] |
    /// where (indexing from zero in column-major order `m[column, row]`)
    /// m[0, 0] :=  cos(yaw) * cos(pitch)
    /// m[0, 1] :=  cos(roll) * sin(pitch) + cos(pitch) * sin(yaw) * sin(roll)
    /// m[0, 2] :=  sin(pitch) * sin(roll) - cos(pitch) * cos(roll) * sin(yaw)
    /// m[1, 0] := -cos(yaw) * cos(pitch)
    /// m[1, 1] :=  cos(pitch) * cos(roll) - sin(yaw) * sin(pitch) * sin(roll)
    /// m[1, 2] :=  cos(pitch) * sin(roll) + cos(roll) * sin(yaw) * sin(pitch)
    /// m[2, 0] :=  sin(yaw)
    /// m[2, 1] := -cos(yaw) * sin(roll)
    /// m[2, 2] :=  cos(yaw) * cos(roll)
    /// ```
    /// This yields the entries in the rotation matrix.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     EulerAngles,
    /// #     Matrix3x3,
    /// # }; 
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let euler_angles = {
    ///     let roll = Radians(f64::consts::FRAC_PI_6);
    ///     let yaw = Radians(f64::consts::FRAC_PI_4);
    ///     let pitch = Radians(f64::consts::FRAC_PI_3);
    /// 
    ///     EulerAngles::new(roll, yaw, pitch)
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
    ///          frac_sqrt_3_2 * frac_1_sqrt_2
    ///     )
    /// };
    /// let result = euler_angles.to_matrix();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn to_matrix(&self) -> Matrix3x3<S> {
        let (sin_roll, cos_roll) = self.x.sin_cos();
        let (sin_yaw, cos_yaw) = self.y.sin_cos();
        let (sin_pitch, cos_pitch) = self.z.sin_cos();
        
        let c0r0 =  cos_yaw * cos_pitch;
        let c0r1 =  cos_roll * sin_pitch + cos_pitch * sin_yaw * sin_roll;
        let c0r2 =  sin_pitch * sin_roll - cos_pitch * cos_roll * sin_yaw;

        let c1r0 = -cos_yaw * sin_pitch;
        let c1r1 =  cos_pitch * cos_roll - sin_yaw * sin_pitch * sin_roll;
        let c1r2 =  cos_pitch * sin_roll + cos_roll * sin_yaw * sin_pitch;

        let c2r0 =  sin_yaw;
        let c2r1 = -cos_yaw * sin_roll;
        let c2r2 =  cos_yaw * cos_roll;

        Matrix3x3::new(
            c0r0, c0r1, c0r2,
            c1r0, c1r1, c1r2,
            c2r0, c2r1, c2r2
        )
    }

    /// Construct an affine rotation matrix from a set of Euler angles.
    ///
    /// A set of Euler angles describes an arbitrary rotation as a sequence
    /// of three axial rotations: one for each axis in thee dimensions **(x, y, z)**.
    /// The rotation matrix described by Euler angles can be decomposed into a 
    /// product of rotation matrices about each axis: let `R_x(roll)`, `R_y(yaw)`, 
    /// and `R_z(pitch)` denote the rotations about the 
    /// **x-axis**, **y-axis**, and **z-axis**, respectively. The Euler rotation
    /// is decomposed as follows
    /// ```text
    /// R(roll, yaw, pitch) == R_x(roll) * R_y(yaw) * R_z(pitch)
    /// ```
    /// The corresponding rotation matrices are
    /// ```text
    ///               | 1   0            0         |
    /// R_x(roll)  := | 0   cos(roll)   -sin(roll) |
    ///               | 0   sin(rol)     cos(roll) |
    /// 
    ///               |  cos(yaw)   0   sin(yaw) |
    /// R_y(yaw)   := |  0          1   0        |
    ///               | -sin(yaw)   0   cos(yaw) |
    ///
    ///               | cos(pitch)   -sin(pitch)   0 |
    /// R_z(pitch) := | sin(pitch)    cos(pitch)   0 |
    ///               | 0             0            1 |
    /// ```
    /// Multiplying out the axial rotations yields the following rotation matrix.
    /// ```text
    ///                        | m[0, 0]   m[1, 0]   m[2, 0] |
    /// R(roll, yaw, pitch) == | m[0, 1]   m[1, 1]   m[2, 1] |
    ///                        | m[0, 2]   m[1, 2]   m[2, 2] |
    /// where (indexing from zero in column-major order `m[column, row]`)
    /// m[0, 0] :=  cos(yaw) * cos(pitch)
    /// m[0, 1] :=  cos(roll) * sin(pitch) + cos(pitch) * sin(yaw) * sin(roll)
    /// m[0, 2] :=  sin(pitch) * sin(roll) - cos(pitch) * cos(roll) * sin(yaw)
    /// m[1, 0] := -cos(yaw) * cos(pitch)
    /// m[1, 1] :=  cos(pitch) * cos(roll) - sin(yaw) * sin(pitch) * sin(roll)
    /// m[1, 2] :=  cos(pitch) * sin(roll) + cos(roll) * sin(yaw) * sin(pitch)
    /// m[2, 0] :=  sin(yaw)
    /// m[2, 1] := -cos(yaw) * sin(roll)
    /// m[2, 2] :=  cos(yaw) * cos(roll)
    /// ```
    /// This yields the entries in the rotation matrix. Since an affine rotation 
    /// matrix has no translation terms in it, the final matrix has the form
    /// ```text
    ///                        | m[0, 0]   m[1, 0]   m[2, 0]   0 |
    /// R(roll, yaw, pitch) == | m[0, 1]   m[1, 1]   m[2, 1]   0 |
    ///                        | m[0, 2]   m[1, 2]   m[2, 2]   0 |
    ///                        | 0         0         0         1 |
    /// ```
    /// as desired.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     EulerAngles,
    /// #     Matrix4x4,
    /// # }; 
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # use core::f64;
    /// #
    /// let euler_angles = { 
    ///     let roll = Radians(f64::consts::FRAC_PI_6);
    ///     let yaw = Radians(f64::consts::FRAC_PI_4);
    ///     let pitch = Radians(f64::consts::FRAC_PI_3);
    /// 
    ///     EulerAngles::new(roll, yaw, pitch)
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
    ///          1_f64
    ///     )
    /// };
    /// let result = euler_angles.to_affine_matrix();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix4x4<S> {
        let (sin_roll, cos_roll) = self.x.sin_cos();
        let (sin_yaw, cos_yaw) = self.y.sin_cos();
        let (sin_pitch, cos_pitch) = self.z.sin_cos();
        let zero = S::zero();
        let one = S::one();

        let c0r0 = cos_yaw * cos_pitch;
        let c0r1 = cos_roll * sin_pitch + cos_pitch * sin_yaw * sin_roll;
        let c0r2 = sin_pitch * sin_roll - cos_pitch * cos_roll * sin_yaw;
        let c0r3 = zero;

        let c1r0 = -cos_yaw * sin_pitch;
        let c1r1 =  cos_pitch * cos_roll - sin_yaw * sin_pitch * sin_roll;
        let c1r2 =  cos_pitch * sin_roll + cos_roll * sin_yaw * sin_pitch;
        let c1r3 =  zero;

        let c2r0 =  sin_yaw;
        let c2r1 = -cos_yaw * sin_roll;
        let c2r2 =  cos_yaw * cos_roll;
        let c2r3 =  zero;

        let c3r0 = zero;
        let c3r1 = zero;
        let c3r2 = zero;
        let c3r3 = one;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        )
    }

    #[inline]
    fn from_matrix(matrix: &Matrix3x3<S>) -> EulerAngles<Radians<S>> {
        let yaw = Radians::asin(matrix[2][0]);
        let cos_yaw = Radians::cos(yaw);
        let (pitch, roll) = if cos_yaw.abs().is_zero() {
            let _pitch = Radians::zero();
            let _roll = Radians::atan2(matrix[1][2], matrix[1][1]);

            (_pitch, _roll)
        } else {
            let _pitch = Radians::atan2(-matrix[1][0], matrix[0][0]);
            let _roll = Radians::atan2(-matrix[2][1], matrix[2][2]);
            
            (_pitch, _roll)
        };

        EulerAngles::new(roll, yaw, pitch)
    }
}

impl<A> fmt::Display for EulerAngles<A> 
where
    A: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "EulerAngles [roll={}, yaw={}, pitch={}]",
            self.x, self.y, self.z
        )
    }
}

impl<S, A> From<EulerAngles<A>> for Matrix3x3<S> 
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    #[inline]
    fn from(euler: EulerAngles<A>) -> Matrix3x3<S> {
        euler.to_matrix()
    }
}

impl<S, A> From<&EulerAngles<A>> for Matrix3x3<S> 
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    #[inline]
    fn from(euler: &EulerAngles<A>) -> Matrix3x3<S> {
        euler.to_matrix()
    }
}

impl<A, S> From<EulerAngles<A>> for Matrix4x4<S> 
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    #[inline]
    fn from(euler: EulerAngles<A>) -> Matrix4x4<S> {
        euler.to_affine_matrix()
    }
}

impl<A, S> From<&EulerAngles<A>> for Matrix4x4<S> 
where
    S: SimdScalarFloat,
    A: Angle<Dimensionless = S>,
{
    #[inline]
    fn from(euler: &EulerAngles<A>) -> Matrix4x4<S> {
        euler.to_affine_matrix()
    }
}

impl<S> From<Quaternion<S>> for EulerAngles<Radians<S>> 
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(quaternion: Quaternion<S>) -> EulerAngles<Radians<S>> {
        let rotation_matrix = quaternion.to_matrix();
        Self::from_matrix(&rotation_matrix)
    }
}

impl<S> From<&Quaternion<S>> for EulerAngles<Radians<S>> 
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(quaternion: &Quaternion<S>) -> EulerAngles<Radians<S>> {
        let rotation_matrix = quaternion.to_matrix();
        Self::from_matrix(&rotation_matrix)
    }
}

impl<A> approx::AbsDiffEq for EulerAngles<A> 
where
    A: Angle,
{
    type Epsilon = A::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        A::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        A::abs_diff_eq(&self.x, &other.x, epsilon) && 
        A::abs_diff_eq(&self.y, &other.y, epsilon) && 
        A::abs_diff_eq(&self.z, &other.z, epsilon)
    }
}

impl<A> approx::RelativeEq for EulerAngles<A> 
where
    A: Angle,
{
    #[inline]
    fn default_max_relative() -> A::Epsilon {
        A::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        A::relative_eq(&self.x, &other.x, epsilon, max_relative) && 
        A::relative_eq(&self.y, &other.y, epsilon, max_relative) &&
        A::relative_eq(&self.z, &other.z, epsilon, max_relative)
    }
}

impl<A> approx::UlpsEq for EulerAngles<A> 
where
    A: Angle,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        A::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        A::ulps_eq(&self.x, &other.x, epsilon, max_ulps) && 
        A::ulps_eq(&self.y, &other.y, epsilon, max_ulps) && 
        A::ulps_eq(&self.z, &other.z, epsilon, max_ulps)
    }
}

