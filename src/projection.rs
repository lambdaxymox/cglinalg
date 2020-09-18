use crate::scalar::{
    ScalarFloat,
};
use crate::angle::{
    Degrees, 
    Radians,
};
use crate::matrix::{
    Matrix4x4,
};
use crate::point::{
    Point3,
};
use crate::vector::{
    Vector3,
};
use crate::traits::{
    Angle,
};


/// An orthographic projection with arbitrary left, right, top, bottom,
/// near, and far planes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OrthographicSpec<S> {
    left: S,
    right: S,
    bottom: S,
    top: S,
    near: S,
    far: S,
}

impl<S> OrthographicSpec<S> {
    pub const fn new(left: S, right: S, bottom: S, top: S, near: S, far: S) -> OrthographicSpec<S> {
        OrthographicSpec {
            left: left,
            right: right,
            bottom: bottom,
            top: top,
            near: near,
            far: far,
        }
    }
}

impl<S> Into<OrthographicSpec<S>> for (S, S, S, S, S, S) {
    #[inline]
    fn into(self) -> OrthographicSpec<S> {
        match self {
            (left, right, bottom, top, near, far) => {
                OrthographicSpec::new(left, right, bottom, top, near, far)
            }
        }
    }
}

impl<S> Into<OrthographicSpec<S>> for &(S, S, S, S, S, S) where S: Copy {
    #[inline]
    fn into(self) -> OrthographicSpec<S> {
        match *self {
            (left, right, bottom, top, near, far) => {
                OrthographicSpec::new(left, right, bottom, top, near, far)
            }
        }
    }
}

impl<S> Into<OrthographicSpec<S>> for [S; 6] {
    #[inline]
    fn into(self) -> OrthographicSpec<S> {
        match self {
            [left, right, bottom, top, near, far] => {
                OrthographicSpec::new(left, right, bottom, top, near, far)
            }
        }
    }
}

impl<S> Into<OrthographicSpec<S>> for &[S; 6] where S: Copy {
    #[inline]
    fn into(self) -> OrthographicSpec<S> {
        match *self {
            [left, right, bottom, top, near, far] => {
                OrthographicSpec::new(left, right, bottom, top, near, far)
            }
        }
    }
}

/// A perspective projection based on arbitarary left, right, bottom,
/// top, near, and far planes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveSpec<S> {
    left: S,
    right: S,
    bottom: S,
    top: S,
    near: S,
    far: S,
}

impl<S> PerspectiveSpec<S> {
    pub const fn new(left: S, right: S, bottom: S, top: S, near: S, far: S) -> PerspectiveSpec<S> {
        PerspectiveSpec {
            left: left,
            right: right,
            bottom: bottom,
            top: top,
            near: near,
            far: far,
        }
    }
}

impl<S> Into<PerspectiveSpec<S>> for (S, S, S, S, S, S) {
    #[inline]
    fn into(self) -> PerspectiveSpec<S> {
        match self {
            (left, right, bottom, top, near, far) => {
                PerspectiveSpec::new(left, right, bottom, top, near, far)
            }
        }
    }
}

impl<S> Into<PerspectiveSpec<S>> for &(S, S, S, S, S, S) where S: Copy {
    #[inline]
    fn into(self) -> PerspectiveSpec<S> {
        match *self {
            (left, right, bottom, top, near, far) => {
                PerspectiveSpec::new(left, right, bottom, top, near, far)
            }
        }
    }
}

impl<S> Into<PerspectiveSpec<S>> for [S; 6] {
    #[inline]
    fn into(self) -> PerspectiveSpec<S> {
        match self {
            [left, right, bottom, top, near, far] => {
                PerspectiveSpec::new(left, right, bottom, top, near, far)
            }
        }
    }
}

impl<S> Into<PerspectiveSpec<S>> for &[S; 6] where S: Copy {
    #[inline]
    fn into(self) -> PerspectiveSpec<S> {
        match *self {
            [left, right, bottom, top, near, far] => {
                PerspectiveSpec::new(left, right, bottom, top, near, far)
            }
        }
    }
}

/// A perspective projection based on the near and farplanes and the vertical
/// field-of-view angle and the horizontal/vertical aspect ratio.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveFovSpec<S> {
    fovy: Radians<S>,
    aspect: S,
    near: S,
    far: S,
}

impl<S> PerspectiveFovSpec<S> {
    pub fn new<A: Into<Radians<S>>>(fovy: A, aspect: S, near: S, far: S) -> PerspectiveFovSpec<S> {
        PerspectiveFovSpec {
            fovy: fovy.into(),
            aspect: aspect,
            near: near,
            far: far,
        }
    }
}

impl<S> Into<PerspectiveFovSpec<S>> for (Radians<S>, S, S, S) {
    #[inline]
    fn into(self) -> PerspectiveFovSpec<S> {
        match self {
            (fovy, aspect, near, far) => {
                PerspectiveFovSpec::new(fovy, aspect, near, far)
            }
        }
    }
}

impl<S> Into<PerspectiveFovSpec<S>> for &(Radians<S>, S, S, S) where S: Copy {
    #[inline]
    fn into(self) -> PerspectiveFovSpec<S> {
        match *self {
            (fovy, aspect, near, far) => {
                PerspectiveFovSpec::new(fovy, aspect, near, far)
            }
        }
    }
}

impl<S> Into<PerspectiveFovSpec<S>> for (Degrees<S>, S, S, S) where S: ScalarFloat {
    #[inline]
    fn into(self) -> PerspectiveFovSpec<S> {
        match self {
            (fovy, aspect, near, far) => {
                PerspectiveFovSpec::new(fovy, aspect, near, far)
            }
        }
    }
}

impl<S> Into<PerspectiveFovSpec<S>> for &(Degrees<S>, S, S, S) where S: ScalarFloat {
    #[inline]
    fn into(self) -> PerspectiveFovSpec<S> {
        match *self {
            (fovy, aspect, near, far) => {
                PerspectiveFovSpec::new(fovy, aspect, near, far)
            }
        }
    }
}

impl<S> From<PerspectiveFovSpec<S>> for Matrix4x4<S> where S: ScalarFloat {
    fn from(spec: PerspectiveFovSpec<S>) -> Matrix4x4<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let fovy_rad = Radians::from(spec.fovy);
        let range = Angle::tan(fovy_rad / two) * spec.near;
        let sx = (two * spec.near) / (range * spec.aspect + range * spec.aspect);
        let sy = spec.near / range;
        let sz = (spec.far + spec.near) / (spec.near - spec.far);
        let pz = (two * spec.far * spec.near) / (spec.near - spec.far);
        
        Matrix4x4::new(
            sx,    zero,  zero,  zero,
            zero,  sy,    zero,  zero,
            zero,  zero,  sz,   -one,
            zero,  zero,  pz,    zero
        )
    }
}

impl<S> From<PerspectiveSpec<S>> for Matrix4x4<S> where S: ScalarFloat {
    fn from(spec: PerspectiveSpec<S>) -> Matrix4x4<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;

        let c0r0 = (two * spec.near) / (spec.right - spec.left);
        let c0r1 = zero;
        let c0r2 = zero;
        let c0r3 = zero;

        let c1r0 = zero;
        let c1r1 = (two * spec.near) / (spec.top - spec.bottom);
        let c1r2 = zero;
        let c1r3 = zero;

        let c2r0 =  (spec.right + spec.left)   / (spec.right - spec.left);
        let c2r1 =  (spec.top   + spec.bottom) / (spec.top   - spec.bottom);
        let c2r2 = -(spec.far   + spec.near)   / (spec.far   - spec.near);
        let c2r3 = -one;

        let c3r0 = zero;
        let c3r1 = zero;
        let c3r2 = -(two * spec.far * spec.near) / (spec.far - spec.near);
        let c3r3 = zero;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3,
        )
    }
}

impl<S> From<OrthographicSpec<S>> for Matrix4x4<S> where S: ScalarFloat {
    fn from(spec: OrthographicSpec<S>) -> Matrix4x4<S> {
        let zero = S::zero();
        let one  = S::one();
        let two = one + one;
        let sx = two / (spec.right - spec.left);
        let sy = two / (spec.top - spec.bottom);
        let sz = two / (spec.far - spec.near);
        let tx = (spec.right + spec.left) / (spec.right - spec.left);
        let ty = (spec.top + spec.bottom) / (spec.top - spec.bottom);
        let tz = (spec.far + spec.near) / (spec.far - spec.near);

        Matrix4x4::new(
             sx,    zero,  zero, zero,
             zero,  sy,    zero, zero,
             zero,  zero,  sz,   zero,
            -tx,   -ty,   -tz,   one
        )
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct PerspectiveProjection3D<S, Spec> {
    spec: Spec,
    matrix: Matrix4x4<S>,
}

impl<S, Spec> PerspectiveProjection3D<S, Spec> where 
    S: ScalarFloat,
    Spec: Copy + Clone + PartialEq + Into<Matrix4x4<S>>,
{
    pub fn new(spec: Spec) -> PerspectiveProjection3D<S, Spec> {
        PerspectiveProjection3D {
            spec: spec,
            matrix: spec.into(),
        }
    }

    pub fn to_spec(&self) -> Spec {
        self.spec
    }

    pub fn to_matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    pub fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    pub fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }
}

impl<S, Spec> AsRef<Matrix4x4<S>> for PerspectiveProjection3D<S, Spec> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S, Spec> approx::AbsDiffEq for PerspectiveProjection3D<S, Spec> where 
    S: ScalarFloat,
    Spec: Copy + Clone + PartialEq,    
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Matrix4x4::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S, Spec> approx::RelativeEq for PerspectiveProjection3D<S, Spec> where 
    S: ScalarFloat,
    Spec: Copy + Clone + PartialEq,    
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S, Spec> approx::UlpsEq for PerspectiveProjection3D<S, Spec> where 
    S: ScalarFloat,
    Spec: Copy + Clone + PartialEq,    
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct OrthographicProjection3D<S> {
    spec: OrthographicSpec<S>,
    matrix: Matrix4x4<S>,
}

impl<S> OrthographicProjection3D<S> where S: ScalarFloat {
    pub fn new(spec: OrthographicSpec<S>) -> OrthographicProjection3D<S> {
        OrthographicProjection3D {
            spec: spec,
            matrix: spec.into(),
        }
    }

    pub fn to_spec(&self) -> OrthographicSpec<S> {
        self.spec
    }

    pub fn to_matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    pub fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    pub fn transform_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }
}

impl<S> AsRef<Matrix4x4<S>> for OrthographicProjection3D<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> approx::AbsDiffEq for OrthographicProjection3D<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Matrix4x4::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S> approx::RelativeEq for OrthographicProjection3D<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for OrthographicProjection3D<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

