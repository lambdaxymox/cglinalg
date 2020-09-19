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

    pub fn from_matrix(matrix: &Matrix3x3<S>) -> EulerAngles<Radians<S>> {
        let yaw_zx = Radians::asin(matrix.c2r0);
        let cos_yaw = yaw_zx.cos();
        if ulps_eq!(S::abs(cos_yaw), S::zero()) {
            let roll_yz = Radians::atan2(matrix.c1r2, matrix.c1r1);
            let pitch_xy = Radians(S::zero());

            EulerAngles::new(roll_yz, yaw_zx, pitch_xy)
        } else {
            let roll_yz = Radians::atan2(-matrix.c2r1, matrix.c2r2);
            let pitch_xy = Radians::atan2(-matrix.c1r0, matrix.c0r0);

            EulerAngles::new(roll_yz, yaw_zx, pitch_xy)
        }
    }
}

impl<A> fmt::Display for EulerAngles<A> where A: fmt::Display + fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
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
