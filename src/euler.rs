use crate::angle::{
    Radians,
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

use num_traits;


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
