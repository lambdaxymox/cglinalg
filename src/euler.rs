use crate::angle::{
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
    Angle,
    Zero,
};
use approx::{
    ulps_eq,
};

use std::fmt;
use std::ops;


#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct EulerAngles<A> {
    pub roll_yz: A,
    pub yaw_zx: A,
    pub pitch_xy: A,
}

impl<A> EulerAngles<A> {
    pub const fn new(roll_yz: A, yaw_zx: A, pitch_xy: A) -> EulerAngles<A> {
        EulerAngles { 
            roll_yz: roll_yz,
            yaw_zx: yaw_zx, 
            pitch_xy: pitch_xy,
        }
    }
}

impl<S> EulerAngles<Radians<S>> where S: ScalarFloat {
    /// Construct an rotation matrix about an axis and an angle from
    /// a set of Euler angles.
    pub fn to_matrix(&self) -> Matrix3x3<S> {
        let (sin_roll, cos_roll) = Radians::sin_cos(self.roll_yz);
        let (sin_yaw, cos_yaw) = Radians::sin_cos(self.yaw_zx);
        let (sin_pitch, cos_pitch) = Radians::sin_cos(self.pitch_xy);
        
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

    /// Construct an affine rotation matrix about an axis and an angle
    /// from a set of Euler angles.
    pub fn to_affine_matrix(&self) -> Matrix4x4<S> {
        let (sin_roll, cos_roll) = Radians::sin_cos(self.roll_yz);
        let (sin_yaw, cos_yaw) = Radians::sin_cos(self.yaw_zx);
        let (sin_pitch, cos_pitch) = Radians::sin_cos(self.pitch_xy);
        let zero = S::zero();
        let one = S::one();

        let c0r0 =  cos_yaw * cos_pitch;
        let c0r1 =  cos_roll * sin_pitch + cos_pitch * sin_yaw * sin_roll;
        let c0r2 =  sin_pitch * sin_roll - cos_pitch * cos_roll * sin_yaw;
        let c0r3 = zero;

        let c1r0 = -cos_yaw * sin_pitch;
        let c1r1 =  cos_pitch * cos_roll - sin_yaw * sin_pitch * sin_roll;
        let c1r2 =  cos_pitch * sin_roll + cos_roll * sin_yaw * sin_pitch;
        let c1r3 = zero;

        let c2r0 =  sin_yaw;
        let c2r1 = -cos_yaw * sin_roll;
        let c2r2 =  cos_yaw * cos_roll;
        let c2r3 = zero;

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
    /// We explain the method because the formulas are not exactly obvious. The method
    /// is based on the method derived by Ken Shoemake in [1]
    ///
    /// ## The Setup For Extracting Euler Angles
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
    /// and adding the first two equations together, and the last two equations together,
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
    /// this gives us the euler angles for the rotation matrix.
    /// 
    /// ### Note
    /// The method here is just one method of extracting Euler angles. More than one 
    /// set of Euler angles can generate the same axis and rotation.
    /// 
    /// [1] _Paul S. Heckbert (Ed.). 1994. Graphics Gems IV. The Graphics Gems Series, Vol. 4. Academic Press. 
    ///     DOI:10.5555/180895. pp. 222-229_
    pub fn from_matrix(matrix: &Matrix3x3<S>) -> EulerAngles<Radians<S>> {
        let roll_yz = Radians::atan2(-matrix.c2r1, matrix.c2r2);
        let cos_yaw_zx = S::sqrt(matrix.c0r0 * matrix.c0r0 + matrix.c1r0 * matrix.c1r0);
        let yaw_zx = Radians::atan2(matrix.c2r0, cos_yaw_zx);
        let sin_roll_yz = Radians::sin(roll_yz);
        let cos_roll_zx = Radians::cos(roll_yz);
        let a = sin_roll_yz * matrix.c0r2 + cos_roll_zx * matrix.c0r1;
        let b = cos_roll_zx * matrix.c1r1 + sin_roll_yz * matrix.c1r2;
        let pitch_xy = Radians::atan2(a, b);

        EulerAngles::new(roll_yz, yaw_zx, pitch_xy)
    }
}

impl<A> fmt::Display for EulerAngles<A> where A: fmt::Display + fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<A, S> From<EulerAngles<A>> for Matrix3x3<S> where 
    A: Angle + Into<Radians<S>>,
    S: ScalarFloat,
{
    #[inline]
    fn from(euler: EulerAngles<A>) -> Matrix3x3<S> {
        let euler_radians: EulerAngles<Radians<S>> = EulerAngles {
            roll_yz: euler.roll_yz.into(),
            yaw_zx: euler.yaw_zx.into(),
            pitch_xy: euler.pitch_xy.into(),
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
            roll_yz: euler.roll_yz.into(),
            yaw_zx: euler.yaw_zx.into(),
            pitch_xy: euler.pitch_xy.into(),
        };
        euler_radians.to_affine_matrix()
    }
}

impl<A> ops::Add<EulerAngles<A>> for EulerAngles<A> where
    A: Copy + Zero + ops::Add<A> 
{
    type Output = EulerAngles<A>;

    fn add(self, other: EulerAngles<A>) -> EulerAngles<A> {
        EulerAngles {
            roll_yz: self.roll_yz + other.roll_yz,
            yaw_zx: self.yaw_zx + other.yaw_zx,
            pitch_xy: self.pitch_xy + other.pitch_xy,
        }
    }
}

impl<A> ops::Add<&EulerAngles<A>> for EulerAngles<A> where 
    A: Copy + Zero + ops::Add<A> 
{
    type Output = EulerAngles<A>;

    fn add(self, other: &EulerAngles<A>) -> EulerAngles<A> {
        EulerAngles {
            roll_yz: self.roll_yz + other.roll_yz,
            yaw_zx: self.yaw_zx + other.yaw_zx,
            pitch_xy: self.pitch_xy + other.pitch_xy,
        }
    }
}

impl<A> ops::Add<EulerAngles<A>> for &EulerAngles<A> where 
    A: Copy + Zero + ops::Add<A>
{
    type Output = EulerAngles<A>;

    fn add(self, other: EulerAngles<A>) -> EulerAngles<A> {
        EulerAngles {
            roll_yz: self.roll_yz + other.roll_yz,
            yaw_zx: self.yaw_zx + other.yaw_zx,
            pitch_xy: self.pitch_xy + other.pitch_xy,
        }
    }
}

impl<'a, 'b, A> ops::Add<&'a EulerAngles<A>> for &'b EulerAngles<A> where 
    A: Copy + Zero + ops::Add<A>
{
    type Output = EulerAngles<A>;

    fn add(self, other: &'a EulerAngles<A>) -> EulerAngles<A> {
        EulerAngles {
            roll_yz: self.roll_yz + other.roll_yz,
            yaw_zx: self.yaw_zx + other.yaw_zx,
            pitch_xy: self.pitch_xy + other.pitch_xy,
        }
    }
}

impl<A> Zero for EulerAngles<A> where A: Angle {
    #[inline]
    fn zero() -> EulerAngles<A> {
        EulerAngles::new(A::zero(), A::zero(), A::zero())
    }

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
        A::abs_diff_eq(&self.roll_yz,  &other.roll_yz,  epsilon) && 
        A::abs_diff_eq(&self.yaw_zx,   &other.yaw_zx,   epsilon) && 
        A::abs_diff_eq(&self.pitch_xy, &other.pitch_xy, epsilon)
    }
}

impl<A: Angle> approx::RelativeEq for EulerAngles<A> {
    #[inline]
    fn default_max_relative() -> A::Epsilon {
        A::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: A::Epsilon, max_relative: A::Epsilon) -> bool {
        A::relative_eq(&self.roll_yz,  &other.roll_yz,  epsilon, max_relative) && 
        A::relative_eq(&self.yaw_zx,   &other.yaw_zx,   epsilon, max_relative) &&
        A::relative_eq(&self.pitch_xy, &other.pitch_xy, epsilon, max_relative)
    }
}

impl<A: Angle> approx::UlpsEq for EulerAngles<A> {
    #[inline]
    fn default_max_ulps() -> u32 {
        A::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: A::Epsilon, max_ulps: u32) -> bool {
        A::ulps_eq(&self.roll_yz,  &other.roll_yz,  epsilon, max_ulps) && 
        A::ulps_eq(&self.yaw_zx,   &other.yaw_zx,   epsilon, max_ulps) && 
        A::ulps_eq(&self.pitch_xy, &other.pitch_xy, epsilon, max_ulps)
    }
}
