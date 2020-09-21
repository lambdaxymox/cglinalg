use crate::scalar::{
    ScalarFloat,
};
use crate::angle::{
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

use core::fmt;


/// A description of an orthographic projection with arbitrary `left`, `right`, 
/// `top`, `bottom`, `near`, and `far` planes.
///
/// We assume the following constraints to construct a useful orthographic 
/// projection
/// ```text
/// left   < right
/// bottom < top
/// near   < far   (along the negative z-axis).
/// ```
/// Each parameter in the specification is a description of the position along 
/// an axis of a plane that the axis is perpendicular to.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OrthographicSpec<S> {
    /// The horizontal position of the left-hand plane in camera space.
    /// The left-hand plane is a plane parallel to the yz-plane at
    /// the origin.
    left: S,
    /// The horizontal position of the right-hand plane in camera space.
    /// The right-hand plane is a plane parallel to the yz-plane at
    /// the origin.
    right: S,
    /// The vertical position of the bottom plane in camera space.
    /// The bottom plane is a plane parallel to the xz-plane at the origin.
    bottom: S,
    /// The vertical position of the top plane in camera space.
    /// the top plane is a plane parallel to the xz plane at the origin.
    top: S,
    /// The distance along the (-z)-direction of the near plane from the eye.
    /// The near plane is a plane parallel to the xy-plane at the origin.
    near: S,
    /// the distance along the (-z)-direction of the far plane from the eye.
    /// The far plane is a plane parallel to the xy-plane at the origin.
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

impl<S> fmt::Display for OrthographicSpec<S> where S: fmt::Debug + fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

/// A perspective projection based on arbitarary `left`, `right`, `bottom`,
/// `top`, `near`, and `far` planes.
///
/// We assume the following constraints to construct a useful perspective 
/// projection
/// ```text
/// left   < right
/// bottom < top
/// near   < far   (along the negative z-axis)
/// ```
/// Each parameter in the specification is a description of the position along
/// an axis of a plane that the axis is perpendicular to.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveSpec<S> {
    /// The horizontal position of the left-hand plane in camera space.
    /// The left-hand plane is a plane parallel to the yz-plane at
    /// the origin.
    left: S,
    /// The horizontal position of the right-hand plane in camera space.
    /// The right-hand plane is a plane parallel to the yz-plane at
    /// the origin.
    right: S,
    /// The vertical position of the bottom plane in camera space.
    /// The bottom plane is a plane parallel to the xz-plane at the origin.
    bottom: S,
    /// The vertical position of the top plane in camera space.
    /// the top plane is a plane parallel to the xz plane at the origin.
    top: S,
    /// The distance along the (-z)-direction of the near plane from the eye.
    /// The near plane is a plane parallel to the xy-plane at the origin.
    near: S,
    /// the distance along the (-z)-direction of the far plane from the eye.
    /// The far plane is a plane parallel to the xy-plane at the origin.
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

impl<S> fmt::Display for PerspectiveSpec<S> where S: fmt::Debug + fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

/// A perspective projection based on the `near` plane, the `far` plane and 
/// the vertical field of view angle `fovy` and the horizontal/vertical aspect 
/// ratio `aspect`.
///
/// We assume the following constraints to make a useful perspective projection 
/// transformation.
/// ```text
/// 0 radians < fovy < pi radians
/// aspect > 0
/// near < far (along the negative z-axis)
/// ```
/// This perspective projection model imposes some constraints on the more 
/// general perspective specification based on the arbitrary planes. The `fovy` 
/// parameter combined with the aspect ratio `aspect` ensures that the top and 
/// bottom planes are the same distance from the eye position along the vertical 
/// axis on opposite side. They ensure that the `left` and `right` planes are 
/// equidistant from the eye on opposite sides along the horizontal axis. 
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveFovSpec<S> {
    /// The vertical field of view angle of the perspective transformation
    /// viewport.
    fovy: Radians<S>,
    /// The ratio of the horizontal width to the vertical height.
    aspect: S,
    /// The position of the near plane along the negative z-axis.
    near: S,
    /// The position of the far plane along the negative z-axis.
    far: S,
}

impl<S> PerspectiveFovSpec<S> {
    /// Construct a new perspective projection operation specification
    /// based on the vertical field of view angle `fovy`, the `near` plane, the 
    /// `far` plane, and aspect ratio `aspect`. 
    pub fn new<A: Into<Radians<S>>>(fovy: A, aspect: S, near: S, far: S) -> PerspectiveFovSpec<S> {
        PerspectiveFovSpec {
            fovy: fovy.into(),
            aspect: aspect,
            near: near,
            far: far,
        }
    }
}

impl<S> fmt::Display for PerspectiveFovSpec<S> where S: fmt::Debug + fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
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
        
        // We use the same perspective projection matrix that OpenGL uses.
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

        // We use the same perspective projection matrix that OpenGL uses.
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
        let sx =  two / (spec.right - spec.left);
        let sy =  two / (spec.top - spec.bottom);
        let sz = -two / (spec.far - spec.near);
        let tx = -(spec.right + spec.left) / (spec.right - spec.left);
        let ty = -(spec.top + spec.bottom) / (spec.top - spec.bottom);
        let tz = -(spec.far + spec.near) / (spec.far - spec.near);

        // We use the same orthographic projection matrix that OpenGL uses.
        Matrix4x4::new(
            sx,   zero, zero, zero,
            zero, sy,   zero, zero,
            zero, zero, sz,   zero,
            tx,   ty,   tz,   one
        )
    }
}

/// A perspective projection tranformation for converting from camera space to
/// normalized device coordinates.
///
/// Orthographic projections differ from perspective projections in that 
/// orthographic projections keeps parallel lines parallel, whereas perspective 
/// projections preserve the perception of distance. That is, perspective 
/// projections preserve the spatial ordering of points in the distance they 
/// are located from the viewing plane. This property of perspective projection 
/// transformations is important for operations such as z-buffering and 
/// occlusion detection.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct PerspectiveProjection3D<S, Spec> {
    /// The parameters of the perspective projection.
    spec: Spec,
    /// The underlying matrix implementing the perspective projection.
    matrix: Matrix4x4<S>,
}

impl<S, Spec> PerspectiveProjection3D<S, Spec> where 
    S: ScalarFloat,
    Spec: Copy + Clone + PartialEq + Into<Matrix4x4<S>>,
{
    /// Construct a new perspective projection transformation.
    pub fn new(spec: Spec) -> PerspectiveProjection3D<S, Spec> {
        PerspectiveProjection3D {
            spec: spec,
            matrix: spec.into(),
        }
    }

    /// Get the specification describing the perspective projection.
    pub fn to_spec(&self) -> Spec {
        self.spec
    }

    /// Get the matrix that implements the perspective projection transformation.
    pub fn to_matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the transformation to a point.
    pub fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    /// Apply the transformation to a vector.
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

/// An orthographic projection transformation for converting from camera space to
/// normalized device coordinates. 
///
/// Orthographic projections differ from perspective projections in that 
/// orthographic projections keeps parallel lines parallel, whereas perspective 
/// projections preserve the perception of distance. That is, perspective 
/// projections preserve the spatial ordering in the distance that points are 
/// located from the viewing plane.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct OrthographicProjection3D<S> {
    /// The parameters for the orthographic projection.
    spec: OrthographicSpec<S>,
    /// The underlying matrix that implements the orthographic projection.
    matrix: Matrix4x4<S>,
}

impl<S> OrthographicProjection3D<S> where S: ScalarFloat {
    /// Construct a new orthographic projection.
    pub fn new(spec: OrthographicSpec<S>) -> OrthographicProjection3D<S> {
        OrthographicProjection3D {
            spec: spec,
            matrix: spec.into(),
        }
    }

    /// Get the parameters defining the orthographic specification.
    pub fn to_spec(&self) -> OrthographicSpec<S> {
        self.spec
    }

    /// Get the underlying matrix implementing the orthographic tranformation.
    pub fn to_matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the tranformation to a point.
    pub fn transform_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    /// Apply the transformation to a vector.
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

