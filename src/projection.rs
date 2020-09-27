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
    /// The left-hand plane is a plane parallel to the _yz-plane_ at
    /// the origin.
    left: S,
    /// The horizontal position of the right-hand plane in camera space.
    /// The right-hand plane is a plane parallel to the _yz-plane_ at
    /// the origin.
    right: S,
    /// The vertical position of the bottom plane in camera space.
    /// The bottom plane is a plane parallel to the _xz-plane_ at the origin.
    bottom: S,
    /// The vertical position of the top plane in camera space.
    /// the top plane is a plane parallel to the _xz-plane_ at the origin.
    top: S,
    /// The distance along the _(-z)-axis_ of the near plane from the eye.
    /// The near plane is a plane parallel to the _xy-plane_ at the origin.
    near: S,
    /// the distance along the _(-z)-axis_ of the far plane from the eye.
    /// The far plane is a plane parallel to the _xy-plane_ at the origin.
    far: S,
}

impl<S> PerspectiveSpec<S> {
    /// Construct a new perspective specification.
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
    /// The position of the near plane along the negative _z-axis_.
    near: S,
    /// The position of the far plane along the negative _z-axis_.
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
    #[inline]
    fn from(spec: PerspectiveFovSpec<S>) -> Matrix4x4<S> {
        Matrix4x4::from_perspective_fov(spec.fovy, spec.aspect, spec.near, spec.far)
    }
}

impl<S> From<&PerspectiveFovSpec<S>> for Matrix4x4<S> where S: ScalarFloat {
    #[inline]
    fn from(spec: &PerspectiveFovSpec<S>) -> Matrix4x4<S> {
        Matrix4x4::from_perspective_fov(spec.fovy, spec.aspect, spec.near, spec.far)
    }
}

impl<S> From<PerspectiveSpec<S>> for Matrix4x4<S> where S: ScalarFloat {
    #[inline]
    fn from(spec: PerspectiveSpec<S>) -> Matrix4x4<S> {
        Matrix4x4::from_perspective(
            spec.left, spec.right, spec.bottom, spec.top, spec.near, spec.far
        )
    }
}

impl<S> From<&PerspectiveSpec<S>> for Matrix4x4<S> where S: ScalarFloat {
    #[inline]
    fn from(spec: &PerspectiveSpec<S>) -> Matrix4x4<S> {
        Matrix4x4::from_perspective(
            spec.left, spec.right, spec.bottom, spec.top, spec.near, spec.far
        )
    }
}

impl<S> From<OrthographicSpec<S>> for Matrix4x4<S> where S: ScalarFloat {
    #[inline]
    fn from(spec: OrthographicSpec<S>) -> Matrix4x4<S> {
        Matrix4x4::from_orthographic(
            spec.left, spec.right, spec.bottom, spec.top, spec.near, spec.far
        )
    }
}

impl<S> From<&OrthographicSpec<S>> for Matrix4x4<S> where S: ScalarFloat {
    #[inline]
    fn from(spec: &OrthographicSpec<S>) -> Matrix4x4<S> {
        Matrix4x4::from_orthographic(
            spec.left, spec.right, spec.bottom, spec.top, spec.near, spec.far
        )
    }
}

impl<S> From<PerspectiveFovSpec<S>> for PerspectiveSpec<S> where S: ScalarFloat {
    #[inline]
    fn from(spec: PerspectiveFovSpec<S>) -> PerspectiveSpec<S> {
        let two = S::one() + S::one();
        let tan_fovy_div_2 = Radians::tan(spec.fovy / two); 
        let top = spec.near * tan_fovy_div_2;
        let bottom = -top;
        let right = spec.aspect * top;
        let left = -right;
        let near = spec.near;
        let far = spec.far;

        PerspectiveSpec::new(left, right, bottom, top, near, far)
    }
}

impl<S> From<&PerspectiveFovSpec<S>> for PerspectiveSpec<S> where S: ScalarFloat {
    #[inline]
    fn from(spec: &PerspectiveFovSpec<S>) -> PerspectiveSpec<S> {
        let two = S::one() + S::one();
        let tan_fovy_div_2 = Radians::tan(spec.fovy / two); 
        let top = spec.near * tan_fovy_div_2;
        let bottom = -top;
        let right = spec.aspect * top;
        let left = -right;
        let near = spec.near;
        let far = spec.far;

        PerspectiveSpec::new(left, right, bottom, top, near, far)
    }
}


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
    /// The left-hand plane is a plane parallel to the _yz-plane_ at
    /// the origin.
    left: S,
    /// The horizontal position of the right-hand plane in camera space.
    /// The right-hand plane is a plane parallel to the _yz-plane_ at
    /// the origin.
    right: S,
    /// The vertical position of the _bottom_ plane in camera space.
    /// The bottom plane is a plane parallel to the _xz-plane_ at the origin.
    bottom: S,
    /// The vertical position of the _top_ plane in camera space.
    /// the top plane is a plane parallel to the _xz-plane_ at the origin.
    top: S,
    /// The distance along the _(-z)-axis_ of the _near_ plane from the eye.
    /// The near plane is a plane parallel to the _xy-plane_ at the origin.
    near: S,
    /// the distance along the _(-z)-axis_ of the _far_ plane from the eye.
    /// The far plane is a plane parallel to the _xy-plane_ at the origin.
    far: S,
}

impl<S> OrthographicSpec<S> {
    /// Construct a new orthographic specification.
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


/// A perspective projection tranformation for converting from camera space to
/// normalized device coordinates.
///
/// Orthographic projections differ from perspective projections because
/// orthographic projections keeps parallel lines parallel, whereas perspective 
/// projections preserve the perception of distance. Perspective 
/// projections preserve the spatial ordering of points in the distance they 
/// are located from the viewing plane. This property of perspective projection 
/// transformations is important for operations such as z-buffering and 
/// occlusion detection.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct PerspectiveProjection3<S> {
    /// The parameters of the perspective projection.
    spec: PerspectiveSpec<S>,
    /// The underlying matrix implementing the perspective projection.
    matrix: Matrix4x4<S>,
}

impl<S> PerspectiveProjection3<S> 
    where S: ScalarFloat
{
    /// Construct a new perspective projection transformation.
    pub fn new(spec: PerspectiveSpec<S>) -> PerspectiveProjection3<S> {
        PerspectiveProjection3 {
            spec: spec,
            matrix: spec.into(),
        }
    }

    /// Get the specification describing the perspective projection.
    pub fn spec(&self) -> PerspectiveSpec<S> {
        self.spec
    }

    /// Get the matrix that implements the perspective projection transformation.
    pub fn to_matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the transformation to a point.
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    /// Apply the transformation to a vector.
    pub fn project_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let projected_vector = self.matrix * vector.expand(S::one());
        let one_div_w = S::one() / projected_vector.w;
        
        (projected_vector * one_div_w).contract()
    }

    /// Unproject a point from normalized device coordinates back to camera
    /// view space. 
    /// 
    /// This is the inverse operation of `project_point`.
    pub fn unproject_point(&self, point: &Point3<S>) -> Point3<S> {
        let spec = self.spec;
        let zero = S::zero();
        let one  = S::one();
        let two = one + one;
        
        let c0r0 =  (spec.right - spec.left) / (two * spec.near);
        let c0r1 =  zero;
        let c0r2 =  zero;
        let c0r3 =  zero;

        let c1r0 =  zero;
        let c1r1 =  (spec.top - spec.bottom) / (two * spec.near);
        let c1r2 =  zero;
        let c1r3 =  zero;

        let c2r0 =  zero;
        let c2r1 =  zero;
        let c2r2 =  zero;
        let c2r3 =  (spec.near - spec.far) / (two * spec.far * spec.near);
        
        let c3r0 =  (spec.left + spec.right) / (two * spec.near);
        let c3r1 =  (spec.bottom + spec.top) / (two * spec.near);
        let c3r2 = -one;
        let c3r3 =  (spec.far + spec.near) / (two * spec.far * spec.near);
        
        let matrix_inverse = Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        );

        Point3::from_homogeneous(matrix_inverse * point.to_homogeneous())
    }

    /// Unproject a vector from normalized device coordinates back to
    /// camera view space. 
    ///
    /// This is the inverse operation of `project_vector`.
    pub fn unproject_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let spec = self.spec;
        let zero = S::zero();
        let one  = S::one();
        let two = one + one;
        
        let c0r0 =  (spec.right - spec.left) / (two * spec.near);
        let c0r1 =  zero;
        let c0r2 =  zero;
        let c0r3 =  zero;

        let c1r0 =  zero;
        let c1r1 =  (spec.top - spec.bottom) / (two * spec.near);
        let c1r2 =  zero;
        let c1r3 =  zero;

        let c2r0 =  zero;
        let c2r1 =  zero;
        let c2r2 =  zero;
        let c2r3 =  (spec.near - spec.far) / (two * spec.far * spec.near);
        
        let c3r0 =  (spec.left + spec.right) / (two * spec.near);
        let c3r1 =  (spec.bottom + spec.top) / (two * spec.near);
        let c3r2 = -one;
        let c3r3 =  (spec.far + spec.near) / (two * spec.far * spec.near);
        
        let matrix_inverse = Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        );
        
        let projected_vector = vector.expand(S::one());
        let unprojected_vector = matrix_inverse * projected_vector;
        
        unprojected_vector.contract() * (S::one() / unprojected_vector.w)
    }
}

impl<S> AsRef<Matrix4x4<S>> for PerspectiveProjection3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for PerspectiveProjection3<S> 
    where S: fmt::Debug + fmt::Display 
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> approx::AbsDiffEq for PerspectiveProjection3<S> 
    where S: ScalarFloat 
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

impl<S> approx::RelativeEq for PerspectiveProjection3<S> 
    where S: ScalarFloat,
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

impl<S> approx::UlpsEq for PerspectiveProjection3<S> 
    where S: ScalarFloat   
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


/// A perspective projection tranformation for converting from camera space to
/// normalized device coordinates based on the perspective fov model.
///
/// Orthographic projections differ from perspective projections because
/// orthographic projections keeps parallel lines parallel, whereas perspective 
/// projections preserve the perception of distance. Perspective 
/// projections preserve the spatial ordering of points in the distance they 
/// are located from the viewing plane. This property of perspective projection 
/// transformations is important for operations such as z-buffering and 
/// occlusion detection.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct PerspectiveFovProjection3<S> {
    /// The parameters of the perspective projection.
    spec: PerspectiveFovSpec<S>,
    /// The underlying matrix implementing the perspective projection.
    matrix: Matrix4x4<S>,
}

impl<S> PerspectiveFovProjection3<S> 
    where S: ScalarFloat
{
    /// Construct a new perspective projection transformation.
    pub fn new(spec: PerspectiveFovSpec<S>) -> PerspectiveFovProjection3<S> {
        PerspectiveFovProjection3 {
            spec: spec,
            matrix: spec.into(),
        }
    }

    /// Get the specification describing the perspective projection.
    pub fn spec(&self) -> PerspectiveFovSpec<S> {
        self.spec
    }

    /// Get the matrix that implements the perspective projection transformation.
    pub fn to_matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the transformation to a point.
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    /// Apply the transformation to a vector.
    pub fn project_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let projected_vector = self.matrix * vector.expand(S::one());
        let one_div_w = S::one() / projected_vector.w;
        
        (projected_vector * one_div_w).contract()
    }

    /// Unproject a point from normalized device coordinates back to camera
    /// view space. 
    /// 
    /// This is the inverse operation of `project_point`.
    pub fn unproject_point(&self, point: &Point3<S>) -> Point3<S> {
        let spec: PerspectiveSpec<S> = self.spec.into();
        let zero = S::zero();
        let one  = S::one();
        let two = one + one;
        
        let c0r0 =  (spec.right - spec.left) / (two * spec.near);
        let c0r1 =  zero;
        let c0r2 =  zero;
        let c0r3 =  zero;

        let c1r0 =  zero;
        let c1r1 =  (spec.top - spec.bottom) / (two * spec.near);
        let c1r2 =  zero;
        let c1r3 =  zero;

        let c2r0 =  zero;
        let c2r1 =  zero;
        let c2r2 =  zero;
        let c2r3 =  (spec.near - spec.far) / (two * spec.far * spec.near);
        
        let c3r0 =  (spec.left + spec.right) / (two * spec.near);
        let c3r1 =  (spec.bottom + spec.top) / (two * spec.near);
        let c3r2 = -one;
        let c3r3 =  (spec.far + spec.near) / (two * spec.far * spec.near);
        
        let matrix_inverse = Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        );

        Point3::from_homogeneous(matrix_inverse * point.to_homogeneous())
    }

    /// Unproject a vector from normalized device coordinates back to
    /// camera view space. 
    ///
    /// This is the inverse operation of `project_vector`.
    pub fn unproject_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let spec: PerspectiveSpec<S> = self.spec.into();
        let zero = S::zero();
        let one  = S::one();
        let two = one + one;
        
        let c0r0 =  (spec.right - spec.left) / (two * spec.near);
        let c0r1 =  zero;
        let c0r2 =  zero;
        let c0r3 =  zero;

        let c1r0 =  zero;
        let c1r1 =  (spec.top - spec.bottom) / (two * spec.near);
        let c1r2 =  zero;
        let c1r3 =  zero;

        let c2r0 =  zero;
        let c2r1 =  zero;
        let c2r2 =  zero;
        let c2r3 =  (spec.near - spec.far) / (two * spec.far * spec.near);
        
        let c3r0 =  (spec.left + spec.right) / (two * spec.near);
        let c3r1 =  (spec.bottom + spec.top) / (two * spec.near);
        let c3r2 = -one;
        let c3r3 =  (spec.far + spec.near) / (two * spec.far * spec.near);
        
        let matrix_inverse = Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        );
        
        let projected_vector = vector.expand(S::one());
        let unprojected_vector = matrix_inverse * projected_vector;
        
        unprojected_vector.contract() * (S::one() / unprojected_vector.w)
    }
}

impl<S> AsRef<Matrix4x4<S>> for PerspectiveFovProjection3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for PerspectiveFovProjection3<S> 
    where S: fmt::Debug + fmt::Display 
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> approx::AbsDiffEq for PerspectiveFovProjection3<S> 
    where S: ScalarFloat
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

impl<S> approx::RelativeEq for PerspectiveFovProjection3<S> where 
    S: ScalarFloat  
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

impl<S> approx::UlpsEq for PerspectiveFovProjection3<S> where 
    S: ScalarFloat
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
/// projections preserve the perception of distance. Perspective 
/// projections preserve the spatial ordering in the distance that points are 
/// located from the viewing plane.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct OrthographicProjection3<S> {
    /// The parameters for the orthographic projection.
    spec: OrthographicSpec<S>,
    /// The underlying matrix that implements the orthographic projection.
    matrix: Matrix4x4<S>,
}

impl<S> OrthographicProjection3<S> where S: ScalarFloat {
    /// Construct a new orthographic projection.
    pub fn new(spec: OrthographicSpec<S>) -> OrthographicProjection3<S> {
        OrthographicProjection3 {
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
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous())
    }

    /// Apply the transformation to a vector.
    pub fn project_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Unproject a point from normalized devices coordinates back to camera
    /// view space. 
    ///
    /// This is the inverse operation of `project_point`.
    pub fn unproject_point(&self, point: &Point3<S>) -> Point3<S> {
        let zero = S::zero();
        let one  = S::one();
        let one_half: S = num_traits::cast(0.5_f64).unwrap();
        
        let c0r0 =  one_half * (self.spec.right - self.spec.left);
        let c0r1 =  zero;
        let c0r2 =  zero;
        let c0r3 =  zero;

        let c1r0 =  zero;
        let c1r1 =  one_half * (self.spec.top - self.spec.bottom);
        let c1r2 =  zero;
        let c1r3 =  zero;

        let c2r0 =  zero;
        let c2r1 =  zero;
        let c2r2 = -one_half * (self.spec.far - self.spec.near);
        let c2r3 =  zero;
        
        let c3r0 =  one_half * (self.spec.left + self.spec.right);
        let c3r1 =  one_half * (self.spec.bottom + self.spec.top);
        let c3r2 = -one_half * (self.spec.far + self.spec.near);
        let c3r3 =  one;
        
        let matrix_inverse = Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        );

        Point3::from_homogeneous(matrix_inverse * point.to_homogeneous())
    }

    /// Unproject a vector from normalized device coordinates back to
    /// camera view space. 
    ///
    /// This is the inverse operation of `project_vector`.
    pub fn unproject_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let zero = S::zero();
        let one  = S::one();
        let one_half: S = num_traits::cast(0.5_f64).unwrap();
        
        let c0r0 =  one_half * (self.spec.right - self.spec.left);
        let c0r1 =  zero;
        let c0r2 =  zero;
        let c0r3 =  zero;

        let c1r0 =  zero;
        let c1r1 =  one_half * (self.spec.top - self.spec.bottom);
        let c1r2 =  zero;
        let c1r3 =  zero;

        let c2r0 =  zero;
        let c2r1 =  zero;
        let c2r2 = -one_half * (self.spec.far - self.spec.near);
        let c2r3 =  zero;
        
        let c3r0 =  one_half * (self.spec.left + self.spec.right);
        let c3r1 =  one_half * (self.spec.bottom + self.spec.top);
        let c3r2 = -one_half * (self.spec.far + self.spec.near);
        let c3r3 =  one;
        
        let matrix_inverse = Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        );

        (matrix_inverse * vector.expand(S::zero())).contract()
    }
}

impl<S> AsRef<Matrix4x4<S>> for OrthographicProjection3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for OrthographicProjection3<S> 
    where S: fmt::Debug + fmt::Display 
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Debug>::fmt(&self, f)
    }
}

impl<S> approx::AbsDiffEq for OrthographicProjection3<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for OrthographicProjection3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for OrthographicProjection3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

