use crate::angle::{
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
use crate::scalar::{
    ScalarFloat,
};
use crate::traits::{
    Zero,
};
use approx::{
    ulps_eq,
};

use core::fmt;
use core::ops;


/// A data type storing a set of Euler angles for representing a rotation about
/// an arbitrary axis in three dimensions.
///
/// The rotations are defined in the ZYX rotation order. That is, the Euler 
/// rotation applies a rotation to the **z-axis**, followed by the **y-axis**, and 
/// lastly the **x-axis**. The ranges of each axis are 
/// ```text
/// x in [-pi, pi]
/// y in [-pi/2, pi/2]
/// z in [-pi, pi]
/// ```
/// where each interval includes its endpoints.
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
/// ## Example (No Gimbal Lock)
/// 
/// The following example is a rotation without gimbal lock.
/// ```
/// # use cglinalg::{
/// #     Degrees,
/// #     EulerAngles,
/// #     Matrix3x3,
/// # };
/// # use cglinalg::approx::{
/// #     relative_eq,
/// # };
///
/// let roll = Degrees(45.0);
/// let yaw = Degrees(30.0);
/// let pitch = Degrees(15.0);
/// let euler = EulerAngles::new(roll, yaw, pitch);
///
/// let c0r0 =  (1.0 / 4.0) * f64::sqrt(3.0 / 2.0) * (1.0 + f64::sqrt(3.0));
/// let c0r1 =  (1.0 / 8.0) * (3.0 * f64::sqrt(3.0) - 1.0);
/// let c0r2 =  (1.0 / 8.0) * (f64::sqrt(3.0) - 3.0);
/// let c1r0 = -(1.0 / 4.0) * f64::sqrt(3.0 / 2.0) * (f64::sqrt(3.0) - 1.0);
/// let c1r1 =  (1.0 / 8.0) * (3.0 + f64::sqrt(3.0));
/// let c1r2 =  (1.0 / 8.0) * (3.0 * f64::sqrt(3.0) + 1.0);
/// let c2r0 =  1.0 / 2.0;
/// let c2r1 = -(1.0 / 2.0) * f64::sqrt(3.0 / 2.0);
/// let c2r2 =  (1.0 / 2.0) * f64::sqrt(3.0 / 2.0);
/// let expected = Matrix3x3::new(
///     c0r0, c0r1, c0r2,
///     c1r0, c1r1, c1r2,
///     c2r0, c2r1, c2r2
/// );
/// let result = Matrix3x3::from(euler); 
///
/// assert!(relative_eq!(result, expected, epsilon = 1e-8));
/// ```
/// 
/// ## Example (Gimbal Lock)
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
/// # use cglinalg::{
/// #    Degrees,
/// #    EulerAngles,
/// #    Matrix3x3,
/// # };
/// # use cglinalg::approx::{
/// #    ulps_eq,
/// # };
///
/// // Gimbal lock the x-axis.
/// let roll = Degrees(45.0);
/// let yaw = Degrees(90.0);
/// let pitch = Degrees(15.0);
/// let euler = EulerAngles::new(roll, yaw, pitch);
/// let matrix_z_locked = Matrix3x3::from(euler);
/// 
/// assert!(ulps_eq!(matrix_z_locked.c0r0, 0.0));
/// assert!(ulps_eq!(matrix_z_locked.c1r0, 0.0));
/// assert!(ulps_eq!(matrix_z_locked.c2r0, 1.0));
/// assert!(ulps_eq!(matrix_z_locked.c2r1, 0.0));
/// assert!(ulps_eq!(matrix_z_locked.c2r2, 0.0));
/// 
/// // Attempt to roll in the gimbal locked state.
/// let euler_roll = EulerAngles::new(Degrees(15.0), Degrees(0.0), Degrees(0.0));
/// let matrix_roll = Matrix3x3::from(euler_roll);
/// let matrix = matrix_roll * matrix_z_locked;
///
/// // But the matrix is still gimbal locked.
/// assert!(ulps_eq!(matrix.c0r0, 0.0));
/// assert!(ulps_eq!(matrix.c1r0, 0.0));
/// assert!(ulps_eq!(matrix.c2r0, 1.0));
/// assert!(ulps_eq!(matrix.c2r1, 0.0));
/// assert!(ulps_eq!(matrix.c2r2, 0.0));
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
    #[inline]
    pub const fn new(x: A, y: A, z: A) -> EulerAngles<A> {
        EulerAngles { 
            x: x,
            y: y, 
            z: z,
        }
    }
}

impl<S> EulerAngles<Radians<S>> where S: ScalarFloat {
    /// Construct an rotation matrix about an axis and an angle from a set 
    /// of Euler angles.
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
    #[inline]
    pub fn to_matrix(&self) -> Matrix3x3<S> {
        let (sin_roll, cos_roll) = Radians::sin_cos(self.x);
        let (sin_yaw, cos_yaw) = Radians::sin_cos(self.y);
        let (sin_pitch, cos_pitch) = Radians::sin_cos(self.z);
        
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

    /// Construct an affine rotation matrix about an axis and an angle from a set 
    /// of Euler angles.
    ///
    /// A set of Euler angles describes an arbitrary rotation as a sequence
    /// of three axial rotations: one for each axis in thee dimensions (x, y, z).
    /// The rotation matrix described by Euler angles can be decomposed into a 
    /// product of rotation matrices about each axis: let `R_x(roll)`, `R_y(yaw)`, 
    /// and `R_z(pitch)` denote the rotations about the 
    /// `x`-axis, `y`-axis, and `z`-axis, respectively. The Euler rotation
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
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix4x4<S> {
        let (sin_roll, cos_roll) = Radians::sin_cos(self.x);
        let (sin_yaw, cos_yaw) = Radians::sin_cos(self.y);
        let (sin_pitch, cos_pitch) = Radians::sin_cos(self.z);
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

    /// Extract Euler angles from a rotation matrix, in units of radians.
    ///
    /// We explain the method because the formulas are not exactly obvious. 
    /// The method is based on the method derived by Ken Shoemake in [1]
    ///
    /// ## The Setup For Extracting Euler Angles
    ///
    /// A set of Euler angles describes an arbitrary rotation as a sequence
    /// of three axial rotations: one for each axis in thee dimensions (x, y, z).
    /// The rotation matrix described by Euler angles can be decomposed into a 
    /// product of rotation matrices about each axis: let `R_x(roll)`, 
    /// `R_y(yaw)`, and `R_z(pitch)` denote the rotations about the 
    /// `x`-axis, `y`-axis, and `z`-axis, respectively. The Euler rotation
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
    /// from which the angles can be extracted.
    /// 
    /// ## The Method For Extracting Euler Angles
    ///
    /// We can now extract Euler angles from the matrix. Consider the 
    /// entries `m[2, 1]` and `m[2, 2]`. If we negate `m[2, 1]` and divide 
    /// by `m[2, 2]` we obtain
    /// ```text
    ///     -m[2, 1]        -(-cos(yaw) * sin(roll))        cos(yaw) * sin(roll)  
    ///   ------------ == ---------------------------- == ------------------------ 
    ///      m[2, 2]          cos(yaw) * cos(roll)          cos(yaw) * cos(roll)   
    ///
    ///                     sin(roll)
    ///                == ------------- =: tan(roll)
    ///                     cos(roll)
    /// ```
    /// We now derive the formula for the `roll` angle
    /// ```text
    /// roll = atan2(-m[2, 1], m[2, 2])
    /// ```
    /// To derive the formula for the `yaw` angle, observe the entries `m[0, 0]` 
    /// and `m[1, 0]` from the matrix.
    /// ```text
    /// m[0, 0]^2 + m[1, 0]^2 == [cos(yaw) * cos(pitch)]^2 + [cos(yaw) * sin(pitch)]^2
    ///                       == cos(yaw)^2  * cos(pitch)^2 + cos(yaw)^2 * sin(pitch)^2
    ///                       == cos(yaw)^2 * [cos(pitch)^2 + sin(pitch)^2]
    ///                       == cos(yaw)^2
    /// ``` 
    /// which implies that
    /// ```text
    /// cos(yaw) == sqrt(m[0, 0] * m[0, 0] + m[1, 0] * m[1, 0])
    /// ```
    /// Also note that we get the sine of the `yaw` angle directly from the 
    /// `m[2, 0]` entry
    /// ```text
    /// sin(yaw) == m[2, 0]
    /// ```
    /// and the ratio of these gives us
    /// ```text
    ///               sin(yaw)                        m[2, 0]
    /// tan(yaw) == ------------ == -----------------------------------------------
    ///               cos(yaw)        sqrt(m[0, 0] * m[0, 0] + m[1, 0] * m[1, 0])
    /// ```
    /// which yield the formula for the `yaw` angle. 
    /// ```text
    /// yaw == atan2(m[2, 0], sqrt(m[0, 0] * m[0, 0] + m[1, 0] * m[1, 0]))
    /// ```
    /// To extract the `pitch` angle
    /// consider the lower left 2x2 square of the matrix. We have the following equations
    /// ```text
    /// m[0, 1] :=  cos(roll) * sin(pitch) + cos(pitch) * sin(yaw) * sin(roll)
    /// m[0, 2] :=  sin(pitch) * sin(roll) - cos(pitch) * cos(roll) * sin(yaw)
    /// m[1, 1] :=  cos(pitch) * cos(roll) - sin(yaw) * sin(pitch) * sin(roll)
    /// m[1, 2] :=  cos(pitch) * sin(roll) + cos(roll) * sin(yaw) * sin(pitch)
    /// ```
    /// Define the following values
    /// ```text
    /// s1 := sin(roll)
    /// c1 := cos(roll)
    /// c2 := cos(yaw)
    /// ```
    /// which are all values we can obtain immediately since we can calculate both the
    /// `roll` and the `yaw` angles. Let's multiply entry entry `m[0, 1]` by `c1`, 
    /// entry `m[0, 2]` by `s1`, entry `m[1, 1]` by `c1`, and entry `m[1, 2]` by `s1`. 
    //// We obtain the relations
    /// ```text
    /// c1 * m[0, 1] == c1 * c1 * sin(pitch) - s1 * c1 * m[2, 0] * cos(pitch)
    /// s1 * m[0, 2] == s1 * s1 * sin(pitch) + s1 * c1 * m[2, 0] * cos(pitch)
    /// c1 * m[1, 1] == c1 * c1 * cos(pitch) - c1 * s1 * m[2, 0] * sin(pitch) 
    /// s1 * m[1, 2] == s1 * s1 * cos(pitch) + c1 * s1 * m[2, 0] * sin(pitch)
    /// ```
    /// Adding the first two equations together and the last two equations together,
    /// respectively, reduces to the following relations
    /// ```text
    /// sin(pitch) == s1 * m[0, 2] + c1 * m[0, 1]
    /// cos(pitch) == c1 * m[1, 1] + s1 * m[1, 2]
    /// ```
    /// and we obtain a relation for the `pitch` angle
    /// ```text
    ///                 sin(pitch)        s1 * m[0, 2] + c1 * m[0, 1]
    /// tan(pitch) == -------------- == -------------------------------
    ///                 cos(pitch)        c1 * m[1, 1] + s1 * m[1, 2]
    /// ```
    /// and the formula for the pitch angle
    /// ```text
    /// y := sin(roll) * m[0, 2] + cos(roll) * m[0, 1]
    /// x := cos(roll) * m[1, 1] + sin(roll) * m[1, 2]
    /// pitch == atan2(y, x)
    /// ```
    /// this gives us the Euler angles for the rotation matrix.
    /// 
    /// ### Note
    /// The method here is just one method of extracting Euler angles. More than one 
    /// set of Euler angles can generate the same axis and rotation.
    /// 
    /// [1] _Paul S. Heckbert (Ed.). 1994. Graphics Gems IV. 
    ///     The Graphics Gems Series, Vol. 4. Academic Press. DOI:10.5555/180895. 
    ///     pp. 222-229_
    #[inline]
    pub fn from_matrix(matrix: &Matrix3x3<S>) -> EulerAngles<Radians<S>> {
        let x = Radians::atan2(-matrix.c2r1, matrix.c2r2);
        let cos_y = S::sqrt(matrix.c0r0 * matrix.c0r0 + matrix.c1r0 * matrix.c1r0);
        let y = Radians::atan2(matrix.c2r0, cos_y);
        let sin_x = Radians::sin(x);
        let cos_roll_zx = Radians::cos(x);

        let numerator = sin_x * matrix.c0r2 + cos_roll_zx * matrix.c0r1;
        let denominator = cos_roll_zx * matrix.c1r1 + sin_x * matrix.c1r2;
        let z = Radians::atan2(numerator, denominator);

        EulerAngles::new(x, y, z)
    }
}

impl<A> fmt::Display for EulerAngles<A> where A: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "EulerAngles [{}, {}, {}]",
            self.x, self.y, self.z
        )
    }
}

impl<A, S> From<EulerAngles<A>> for Matrix3x3<S> where 
    A: Angle + Into<Radians<S>>,
    S: ScalarFloat,
{
    #[inline]
    fn from(euler: EulerAngles<A>) -> Matrix3x3<S> {
        let euler_radians: EulerAngles<Radians<S>> = EulerAngles {
            x: euler.x.into(),
            y: euler.y.into(),
            z: euler.z.into(),
        };
        euler_radians.to_matrix()
    }
}

impl<A, S> From<EulerAngles<A>> for Matrix4x4<S> where 
    A: Angle + Into<Radians<S>>,
    S: ScalarFloat,
{
    #[inline]
    fn from(euler: EulerAngles<A>) -> Matrix4x4<S> {
        let euler_radians: EulerAngles<Radians<S>> = EulerAngles {
            x: euler.x.into(),
            y: euler.y.into(),
            z: euler.z.into(),
        };
        euler_radians.to_affine_matrix()
    }
}

impl<S> From<Quaternion<S>> for EulerAngles<Radians<S>> where S: ScalarFloat {
    #[inline]
    fn from(src: Quaternion<S>) -> EulerAngles<Radians<S>> {
        let sig: S = num_traits::cast(0.499).unwrap();
        let two: S = num_traits::cast(2).unwrap();
        let one: S = num_traits::cast(1).unwrap();

        let (qw, qx, qy, qz) = (src.s, src.v.x, src.v.y, src.v.z);
        let (sqw, sqx, sqy, sqz) = (qw * qw, qx * qx, qy * qy, qz * qz);

        let unit = sqx + sqz + sqy + sqw;
        let test = qx * qz + qy * qw;

        // We set x to zero and z to the value, but the other way would work too.
        if test > sig * unit {
            // x + z = 2 * atan(x / w)
            EulerAngles::new(
                Radians::zero(),
                Radians::full_turn_div_4(),
                Radians::atan2(qx, qw) * two,
            )
        } else if test < -sig * unit {
            // x - z = 2 * atan(x / w)
            EulerAngles::new(
                 Radians::zero(),
                -Radians::full_turn_div_4(),
                -Radians::atan2(qx, qw) * two,
            )
        } else {
            // Using the quat-to-matrix equation from either
            // http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
            // or equation 15 on page 7 of
            // http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf
            // to fill in the equations on page A-2 of the NASA document gives the below.
            EulerAngles::new(
                Radians::atan2(two * (-qy * qz + qx * qw), one - two * (sqx + sqy)),
                Radians::asin(two * (qx * qz + qy * qw)),
                Radians::atan2(two * (-qx * qy + qz * qw), one - two * (sqy + sqz)),
            )
        }
    }
}

impl<A> ops::Add<EulerAngles<A>> for EulerAngles<A> where
    A: Copy + Zero + ops::Add<A> 
{
    type Output = EulerAngles<A>;

    #[inline]
    fn add(self, other: EulerAngles<A>) -> EulerAngles<A> {
        EulerAngles {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<A> ops::Add<&EulerAngles<A>> for EulerAngles<A> where 
    A: Copy + Zero + ops::Add<A> 
{
    type Output = EulerAngles<A>;

    #[inline]
    fn add(self, other: &EulerAngles<A>) -> EulerAngles<A> {
        EulerAngles {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<A> ops::Add<EulerAngles<A>> for &EulerAngles<A> where 
    A: Copy + Zero + ops::Add<A>
{
    type Output = EulerAngles<A>;

    #[inline]
    fn add(self, other: EulerAngles<A>) -> EulerAngles<A> {
        EulerAngles {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<'a, 'b, A> ops::Add<&'a EulerAngles<A>> for &'b EulerAngles<A> where 
    A: Copy + Zero + ops::Add<A>
{
    type Output = EulerAngles<A>;

    #[inline]
    fn add(self, other: &'a EulerAngles<A>) -> EulerAngles<A> {
        EulerAngles {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<A> Zero for EulerAngles<A> where A: Angle {
    #[inline]
    fn zero() -> EulerAngles<A> {
        EulerAngles::new(A::zero(), A::zero(), A::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        ulps_eq!(self, &Self::zero())
    }
}

impl<A: Angle> approx::AbsDiffEq for EulerAngles<A> {
    type Epsilon = A::Epsilon;

    #[inline]
    fn default_epsilon() -> A::Epsilon {
        A::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: A::Epsilon) -> bool {
        A::abs_diff_eq(&self.x, &other.x, epsilon) && 
        A::abs_diff_eq(&self.y, &other.y, epsilon) && 
        A::abs_diff_eq(&self.z, &other.z, epsilon)
    }
}

impl<A: Angle> approx::RelativeEq for EulerAngles<A> {
    #[inline]
    fn default_max_relative() -> A::Epsilon {
        A::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: A::Epsilon, max_relative: A::Epsilon) -> bool {
        A::relative_eq(&self.x, &other.x, epsilon, max_relative) && 
        A::relative_eq(&self.y, &other.y, epsilon, max_relative) &&
        A::relative_eq(&self.z, &other.z, epsilon, max_relative)
    }
}

impl<A: Angle> approx::UlpsEq for EulerAngles<A> {
    #[inline]
    fn default_max_ulps() -> u32 {
        A::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: A::Epsilon, max_ulps: u32) -> bool {
        A::ulps_eq(&self.x, &other.x, epsilon, max_ulps) && 
        A::ulps_eq(&self.y, &other.y, epsilon, max_ulps) && 
        A::ulps_eq(&self.z, &other.z, epsilon, max_ulps)
    }
}
