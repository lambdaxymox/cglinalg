use crate::scalar::{
    ScalarFloat,
};
use crate::angle::{
    Angle,
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

use core::fmt;


/// A perspective projection based on arbitrary `left`, `right`, `bottom`,
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PerspectiveSpec<S> {
    /// The horizontal position of the left-hand plane in camera space.
    /// The left-hand plane is a plane parallel to the **yz-plane** at
    /// the origin.
    left: S,
    /// The horizontal position of the right-hand plane in camera space.
    /// The right-hand plane is a plane parallel to the **yz-plane** at
    /// the origin.
    right: S,
    /// The vertical position of the bottom plane in camera space.
    /// The bottom plane is a plane parallel to the **xz-plane** at the origin.
    bottom: S,
    /// The vertical position of the top plane in camera space.
    /// the top plane is a plane parallel to the **xz-plane** at the origin.
    top: S,
    /// The distance along the **negative z-axis** of the near plane from the eye.
    /// The near plane is a plane parallel to the **xy-plane** at the origin.
    near: S,
    /// the distance along the **negative z-axis** of the far plane from the eye.
    /// The far plane is a plane parallel to the **xy-plane** at the origin.
    far: S,
}

impl<S> PerspectiveSpec<S> {
    /// Construct a new perspective specification.
    #[inline]
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

impl<S> fmt::Display for PerspectiveSpec<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "PerspectiveSpec [left={}, right={}, bottom={}, top={}, near={}, far={}]",
            self.left, self.right, self.bottom, self.top, self.near, self.far
        )
    }
}

/*
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PerspectiveFovSpec<S> {
    /// The vertical field of view angle of the perspective transformation
    /// viewport.
    fovy: Radians<S>,
    /// The ratio of the horizontal width to the vertical height.
    aspect: S,
    /// The position of the near plane along the **negative z-axis**.
    near: S,
    /// The position of the far plane along the **negative z-axis**.
    far: S,
}

impl<S> PerspectiveFovSpec<S> {
    /// Construct a new perspective projection operation specification
    /// based on the vertical field of view angle `fovy`, the `near` plane, the 
    /// `far` plane, and aspect ratio `aspect`.
    #[inline]
    pub fn new<A: Into<Radians<S>>>(fovy: A, aspect: S, near: S, far: S) -> PerspectiveFovSpec<S> {
        PerspectiveFovSpec {
            fovy: fovy.into(),
            aspect: aspect,
            near: near,
            far: far,
        }
    }
}

impl<S> fmt::Display for PerspectiveFovSpec<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
       write!(
           formatter,
           "PerspectiveFovSpec [fovy={}, aspect={}, near={}, far={}]",
           self.fovy, self.aspect, self.near, self.far
       )
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
*/
/// A perspective projection transformation for converting from camera space to
/// normalized device coordinates.
///
/// Orthographic projections differ from perspective projections because
/// orthographic projections keeps parallel lines parallel, whereas perspective 
/// projections preserve the perception of distance. Perspective 
/// projections preserve the spatial ordering of points in the distance they 
/// are located from the viewing plane. This property of perspective projection 
/// transformations is important for operations such as z-buffering and 
/// occlusion detection.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Perspective3<S> {
    /// The parameters of the perspective projection.
    spec: PerspectiveSpec<S>,
    /// The underlying matrix implementing the perspective projection.
    matrix: Matrix4x4<S>,
}

impl<S> Perspective3<S> 
    where S: ScalarFloat
{
    /// Construct a new perspective projection transformation.
    pub fn new(spec: PerspectiveSpec<S>) -> Perspective3<S> {
        Perspective3 {
            spec: spec,
            matrix: Matrix4x4::from_perspective(
                spec.left, 
                spec.right, 
                spec.bottom, 
                spec.top, 
                spec.near, 
                spec.far
            ),
        }
    }

    /// Get the specification describing the perspective projection.
    #[inline]
    pub fn spec(&self) -> PerspectiveSpec<S> {
        self.spec
    }

    /// Get the matrix that implements the perspective projection transformation.
    #[inline]
    pub fn to_matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the transformation to a point.
    #[inline]
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous()).unwrap()
    }

    /// Apply the transformation to a vector.
    #[inline]
    pub fn project_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let projected_vector = self.matrix * vector.expand(S::one());
        let one_div_w = S::one() / projected_vector.w;
        
        (projected_vector * one_div_w).contract()
    }

    /// Unproject a point from normalized device coordinates back to camera
    /// view space. 
    /// 
    /// This is the inverse operation of `project_point`.
    #[inline]
    pub fn unproject_point(&self, point: &Point3<S>) -> Point3<S> {
        // The perspective projection matrix has the form
        // ```text
        // | c0r0 c1r0 c2r0 c3r0 |    | 2*n/(r - l)   0               (r + l)/(r - l)    0             |
        // | c0r1 c1r1 c2r1 c3r1 | == | 0             2*n/(t - b)     (t + b)/(t - b)    0             |
        // | c0r2 c1r2 c2r2 c3r2 |    | 0             0              -(f + n)/(f - n)   -2*f*n/(f - n) |
        // | c0r3 c1r3 c2r3 c3r3 |    | 0             0              -1                  0             |
        // ```
        //
        // The inverse matrix of the perspective projection matrix has the form
        // ```text
        // | c0r0 c1r0 c2r0 c3r0 |    | (r - l)/(2*n)   0               0                   (r + l)/(2*n)   |
        // | c0r1 c1r1 c2r1 c3r1 | == | 0               (t - b)/(2*n)   0                   (t + b)/(2*n)   |
        // | c0r2 c1r2 c2r2 c3r2 |    | 0               0               0                  -1               |
        // | c0r3 c1r3 c2r3 c3r3 |    | 0               0               (f - n)/(-2*f*n)    (f + n)/(2*f*n) |
        // ```
        // 
        // This leads to optimizated unprojection equivalent to the original
        // calculation via matrix calclulation.
        // We can save nine multiplications, nine additions, and one matrix 
        // construction by only applying the nonzero elements
        // c0r0, c1r1, c2r3, c3r0, c3r1, c3r2, and c3r3 to the input vector.
        //
        // let spec = self.spec;
        // let zero = S::zero();
        // let one  = S::one();
        // let two = one + one;
        //
        // let c0r0 =  (spec.right - spec.left) / (two * spec.near);
        // let c0r1 =  zero;
        // let c0r2 =  zero;
        // let c0r3 =  zero;
        //
        // let c1r0 =  zero;
        // let c1r1 =  (spec.top - spec.bottom) / (two * spec.near);
        // let c1r2 =  zero;
        // let c1r3 =  zero;
        // 
        // let c2r0 =  zero;
        // let c2r1 =  zero;
        // let c2r2 =  zero;
        // let c2r3 =  (spec.near - spec.far) / (two * spec.far * spec.near);
        //
        // let c3r0 =  (spec.left + spec.right) / (two * spec.near);
        // let c3r1 =  (spec.bottom + spec.top) / (two * spec.near);
        // let c3r2 = -one;
        // let c3r3 =  (spec.far + spec.near) / (two * spec.far * spec.near);
        //
        // let matrix_inverse = Matrix4x4::new(
        //    c0r0, c0r1, c0r2, c0r3,
        //    c1r0, c1r1, c1r2, c1r3,
        //    c2r0, c2r1, c2r2, c2r3,
        //    c3r0, c3r1, c3r2, c3r3
        // );
        let spec = self.spec;
        let one = S::one();
        let two = one + one;
        let c0r0 = (spec.right - spec.left) / (two * spec.near);
        let c1r1 = (spec.top - spec.bottom) / (two * spec.near);
        let c2r3 =  (spec.near - spec.far) / (two * spec.far * spec.near);
        let c3r0 =  (spec.left + spec.right) / (two * spec.near);
        let c3r1 =  (spec.bottom + spec.top) / (two * spec.near);
        let c3r2 = -one;
        let c3r3 =  (spec.far + spec.near) / (two * spec.far * spec.near);
        let w = c2r3 * point.z + c3r3;
        let inverse_w = one / w;

        Point3::new(
            (c0r0 * point.x + c3r0) * inverse_w,
            (c1r1 * point.y + c3r1) * inverse_w,
            c3r2 * inverse_w,
        )
    }

    /// Unproject a vector from normalized device coordinates back to
    /// camera view space. 
    ///
    /// This is the inverse operation of `project_vector`.
    #[inline]
    pub fn unproject_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        // The perspective projection matrix has the form
        // ```text
        // | c0r0 c1r0 c2r0 c3r0 |    | 2*n/(r - l)   0               (r + l)/(r - l)    0             |
        // | c0r1 c1r1 c2r1 c3r1 | == | 0             2*n/(t - b)     (t + b)/(t - b)    0             |
        // | c0r2 c1r2 c2r2 c3r2 |    | 0             0              -(f + n)/(f - n)   -2*f*n/(f - n) |
        // | c0r3 c1r3 c2r3 c3r3 |    | 0             0              -1                  0             |
        // ```
        //
        // The inverse matrix of the perspective projection matrix has the form
        // ```text
        // | c0r0 c1r0 c2r0 c3r0 |    | (r - l)/(2*n)   0               0                   (r + l)/(2*n)   |
        // | c0r1 c1r1 c2r1 c3r1 | == | 0               (t - b)/(2*n)   0                   (t + b)/(2*n)   |
        // | c0r2 c1r2 c2r2 c3r2 |    | 0               0               0                  -1               |
        // | c0r3 c1r3 c2r3 c3r3 |    | 0               0               (f - n)/(-2*f*n)    (f + n)/(2*f*n) |
        // ```
        // 
        // This leads to optimizated unprojection equivalent to the original
        // calculation via matrix calclulation.
        // We can save nine multiplications, nine additions, and one matrix 
        // construction by only applying the nonzero elements
        // c0r0, c1r1, c2r3, c3r0, c3r1, c3r2, and c3r3 to the input vector.
        //
        // let spec = self.spec;
        // let zero = S::zero();
        // let one  = S::one();
        // let two = one + one;
        //
        // let c0r0 =  (spec.right - spec.left) / (two * spec.near);
        // let c0r1 =  zero;
        // let c0r2 =  zero;
        // let c0r3 =  zero;
        //
        // let c1r0 =  zero;
        // let c1r1 =  (spec.top - spec.bottom) / (two * spec.near);
        // let c1r2 =  zero;
        // let c1r3 =  zero;
        // 
        // let c2r0 =  zero;
        // let c2r1 =  zero;
        // let c2r2 =  zero;
        // let c2r3 =  (spec.near - spec.far) / (two * spec.far * spec.near);
        //
        // let c3r0 =  (spec.left + spec.right) / (two * spec.near);
        // let c3r1 =  (spec.bottom + spec.top) / (two * spec.near);
        // let c3r2 = -one;
        // let c3r3 =  (spec.far + spec.near) / (two * spec.far * spec.near);
        //
        // let matrix_inverse = Matrix4x4::new(
        //    c0r0, c0r1, c0r2, c0r3,
        //    c1r0, c1r1, c1r2, c1r3,
        //    c2r0, c2r1, c2r2, c2r3,
        //    c3r0, c3r1, c3r2, c3r3
        // );
        let spec = self.spec;
        let one = S::one();
        let two = one + one;
        let c0r0 = (spec.right - spec.left) / (two * spec.near);
        let c1r1 = (spec.top - spec.bottom) / (two * spec.near);
        let c2r3 =  (spec.near - spec.far) / (two * spec.far * spec.near);
        let c3r0 =  (spec.left + spec.right) / (two * spec.near);
        let c3r1 =  (spec.bottom + spec.top) / (two * spec.near);
        let c3r2 = -one;
        let c3r3 =  (spec.far + spec.near) / (two * spec.far * spec.near);
        let w = c2r3 * vector.z + c3r3;
        let inverse_w = one / w;

        Vector3::new(
            (c0r0 * vector.x + c3r0) * inverse_w,
            (c1r1 * vector.y + c3r1) * inverse_w,
            c3r2 * inverse_w,
        )
    }
}

impl<S> AsRef<Matrix4x4<S>> for Perspective3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Perspective3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Perspective3 [{}]",
            self.matrix
        )
    }
}

impl<S> approx::AbsDiffEq for Perspective3<S> 
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

impl<S> approx::RelativeEq for Perspective3<S> 
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

impl<S> approx::UlpsEq for Perspective3<S> 
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


/// A perspective projection transformation for converting from camera space to
/// normalized device coordinates based on the perspective field of view model.
///
/// Orthographic projections differ from perspective projections because
/// orthographic projections keeps parallel lines parallel, whereas perspective 
/// projections preserve the perception of distance. Perspective 
/// projections preserve the spatial ordering of points in the distance they 
/// are located from the viewing plane. This property of perspective projection 
/// transformations is important for operations such as z-buffering and 
/// occlusion detection.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveFov3<S> {
    /// The vertical field of view angle of the perspective transformation
    /// viewport.
    fovy: Radians<S>,
    /// The ratio of the horizontal width to the vertical height.
    aspect: S,
    /// The position of the near plane along the **negative z-axis**.
    near: S,
    /// The position of the far plane along the **negative z-axis**.
    far: S,
    /// The underlying matrix implementing the perspective projection.
    matrix: Matrix4x4<S>,
}

impl<S> PerspectiveFov3<S> 
    where S: ScalarFloat
{
    /// Construct a new perspective projection transformation.
    pub fn new<A: Into<Radians<S>>>(fovy: A, aspect: S, near: S, far: S) -> PerspectiveFov3<S> {
        let spec_fovy = fovy.into();

        PerspectiveFov3 {
            fovy: spec_fovy,
            aspect: aspect,
            near: near,
            far: far,
            matrix: Matrix4x4::from_perspective_fov(spec_fovy, aspect, near, far),
        }
    }

    /// Get the matrix that implements the perspective projection transformation.
    #[inline]
    pub fn to_matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the transformation to a point.
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous()).unwrap()
    }

    /// Apply the transformation to a vector.
    #[inline]
    pub fn project_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let projected_vector = self.matrix * vector.expand(S::one());
        let one_div_w = S::one() / projected_vector.w;
        
        (projected_vector * one_div_w).contract()
    }

    /// Unproject a point from normalized device coordinates back to camera
    /// view space. 
    /// 
    /// This is the inverse operation of `project_point`.
    #[inline]
    pub fn unproject_point(&self, point: &Point3<S>) -> Point3<S> {
        // The perspective projection matrix has the form
        // ```text
        // | c0r0 c1r0 c2r0 c3r0 |    | 2*n/(r - l)   0               (r + l)/(r - l)    0             |
        // | c0r1 c1r1 c2r1 c3r1 | == | 0             2*n/(t - b)     (t + b)/(t - b)    0             |
        // | c0r2 c1r2 c2r2 c3r2 |    | 0             0              -(f + n)/(f - n)   -2*f*n/(f - n) |
        // | c0r3 c1r3 c2r3 c3r3 |    | 0             0              -1                  0             |
        // ```
        //
        // The inverse matrix of the perspective projection matrix has the form
        // ```text
        // | c0r0 c1r0 c2r0 c3r0 |    | (r - l)/(2*n)   0               0                   (r + l)/(2*n)   |
        // | c0r1 c1r1 c2r1 c3r1 | == | 0               (t - b)/(2*n)   0                   (t + b)/(2*n)   |
        // | c0r2 c1r2 c2r2 c3r2 |    | 0               0               0                  -1               |
        // | c0r3 c1r3 c2r3 c3r3 |    | 0               0               (f - n)/(-2*f*n)    (f + n)/(2*f*n) |
        // ```
        // 
        // This leads to optimizated unprojection equivalent to the original
        // calculation via matrix calclulation.
        // We can save nine multiplications, nine additions, and one matrix 
        // construction by only applying the nonzero elements
        // c0r0, c1r1, c2r3, c3r0, c3r1, c3r2, and c3r3 to the input vector.
        //
        // let spec = self.spec;
        // let zero = S::zero();
        // let one  = S::one();
        // let two = one + one;
        //
        // let c0r0 =  (spec.right - spec.left) / (two * spec.near);
        // let c0r1 =  zero;
        // let c0r2 =  zero;
        // let c0r3 =  zero;
        //
        // let c1r0 =  zero;
        // let c1r1 =  (spec.top - spec.bottom) / (two * spec.near);
        // let c1r2 =  zero;
        // let c1r3 =  zero;
        // 
        // let c2r0 =  zero;
        // let c2r1 =  zero;
        // let c2r2 =  zero;
        // let c2r3 =  (spec.near - spec.far) / (two * spec.far * spec.near);
        //
        // let c3r0 =  (spec.left + spec.right) / (two * spec.near);
        // let c3r1 =  (spec.bottom + spec.top) / (two * spec.near);
        // let c3r2 = -one;
        // let c3r3 =  (spec.far + spec.near) / (two * spec.far * spec.near);
        //
        // let matrix_inverse = Matrix4x4::new(
        //    c0r0, c0r1, c0r2, c0r3,
        //    c1r0, c1r1, c1r2, c1r3,
        //    c2r0, c2r1, c2r2, c2r3,
        //    c3r0, c3r1, c3r2, c3r3
        // );
        let one = S::one();
        let two = one + one;
        let near = self.near;
        let far = self.far;
        let tan_fovy_div_2 = Radians::tan(self.fovy / two); 
        let top = self.near * tan_fovy_div_2;
        let bottom = -top;
        let right = self.aspect * top;
        let left = -right;

        let c0r0 = (right - left) / (two * near);
        let c1r1 = (top - bottom) / (two * near);
        let c2r3 =  (near - far) / (two * far * near);
        let c3r0 =  (left + right) / (two * near);
        let c3r1 =  (bottom + top) / (two * near);
        let c3r2 = -one;
        let c3r3 =  (far + near) / (two * far * near);
        let w = c2r3 * point.z + c3r3;
        let inverse_w = one / w;

        Point3::new(
            (c0r0 * point.x + c3r0) * inverse_w,
            (c1r1 * point.y + c3r1) * inverse_w,
            c3r2 * inverse_w,
        )
    }

    /// Unproject a vector from normalized device coordinates back to
    /// camera view space. 
    ///
    /// This is the inverse operation of `project_vector`.
    #[inline]
    pub fn unproject_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        // The perspective projection matrix has the form
        // ```text
        // | c0r0 c1r0 c2r0 c3r0 |    | 2*n/(r - l)   0               (r + l)/(r - l)    0             |
        // | c0r1 c1r1 c2r1 c3r1 | == | 0             2*n/(t - b)     (t + b)/(t - b)    0             |
        // | c0r2 c1r2 c2r2 c3r2 |    | 0             0              -(f + n)/(f - n)   -2*f*n/(f - n) |
        // | c0r3 c1r3 c2r3 c3r3 |    | 0             0              -1                  0             |
        // ```
        //
        // The inverse matrix of the perspective projection matrix has the form
        // ```text
        // | c0r0 c1r0 c2r0 c3r0 |    | (r - l)/(2*n)   0               0                   (r + l)/(2*n)   |
        // | c0r1 c1r1 c2r1 c3r1 | == | 0               (t - b)/(2*n)   0                   (t + b)/(2*n)   |
        // | c0r2 c1r2 c2r2 c3r2 |    | 0               0               0                  -1               |
        // | c0r3 c1r3 c2r3 c3r3 |    | 0               0               (f - n)/(-2*f*n)    (f + n)/(2*f*n) |
        // ```
        // 
        // This leads to optimizated unprojection equivalent to the original
        // calculation via matrix calclulation.
        // We can save nine multiplications, nine additions, and one matrix 
        // construction by only applying the nonzero elements
        // c0r0, c1r1, c2r3, c3r0, c3r1, c3r2, and c3r3 to the input vector.
        //
        // let spec = self.spec;
        // let zero = S::zero();
        // let one  = S::one();
        // let two = one + one;
        //
        // let c0r0 =  (spec.right - spec.left) / (two * spec.near);
        // let c0r1 =  zero;
        // let c0r2 =  zero;
        // let c0r3 =  zero;
        //
        // let c1r0 =  zero;
        // let c1r1 =  (spec.top - spec.bottom) / (two * spec.near);
        // let c1r2 =  zero;
        // let c1r3 =  zero;
        // 
        // let c2r0 =  zero;
        // let c2r1 =  zero;
        // let c2r2 =  zero;
        // let c2r3 =  (spec.near - spec.far) / (two * spec.far * spec.near);
        //
        // let c3r0 =  (spec.left + spec.right) / (two * spec.near);
        // let c3r1 =  (spec.bottom + spec.top) / (two * spec.near);
        // let c3r2 = -one;
        // let c3r3 =  (spec.far + spec.near) / (two * spec.far * spec.near);
        //
        // let matrix_inverse = Matrix4x4::new(
        //    c0r0, c0r1, c0r2, c0r3,
        //    c1r0, c1r1, c1r2, c1r3,
        //    c2r0, c2r1, c2r2, c2r3,
        //    c3r0, c3r1, c3r2, c3r3
        // );
        let one = S::one();
        let two = one + one;
        let near = self.near;
        let far = self.far;
        let tan_fovy_div_2 = Radians::tan(self.fovy / two); 
        let top = self.near * tan_fovy_div_2;
        let bottom = -top;
        let right = self.aspect * top;
        let left = -right;

        let c0r0 = (right - left) / (two * near);
        let c1r1 = (top - bottom) / (two * near);
        let c2r3 =  (near - far) / (two * far * near);
        let c3r0 =  (left + right) / (two * near);
        let c3r1 =  (bottom + top) / (two * near);
        let c3r2 = -one;
        let c3r3 =  (far + near) / (two * far * near);
        let w = c2r3 * vector.z + c3r3;
        let inverse_w = one / w;

        Vector3::new(
            (c0r0 * vector.x + c3r0) * inverse_w,
            (c1r1 * vector.y + c3r1) * inverse_w,
            c3r2 * inverse_w,
        )
    }
}

impl<S> AsRef<Matrix4x4<S>> for PerspectiveFov3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for PerspectiveFov3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "PerspectiveFov3 [{}]",
            self.matrix
        )
    }
}

impl<S> approx::AbsDiffEq for PerspectiveFov3<S> 
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

impl<S> approx::RelativeEq for PerspectiveFov3<S> where 
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

impl<S> approx::UlpsEq for PerspectiveFov3<S> where 
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


/// An orthographic projection with arbitrary `left`, `right`, 
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
///
/// Orthographic projections differ from perspective projections in that 
/// orthographic projections keeps parallel lines parallel, whereas perspective 
/// projections preserve the perception of distance. Perspective 
/// projections preserve the spatial ordering in the distance that points are 
/// located from the viewing plane.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Orthographic3<S> {
    /// The horizontal position of the left-hand plane in camera space.
    /// The left-hand plane is a plane parallel to the **yz-plane** at
    /// the origin.
    left: S,
    /// The horizontal position of the right-hand plane in camera space.
    /// The right-hand plane is a plane parallel to the **yz-plane** at
    /// the origin.
    right: S,
    /// The vertical position of the **bottom plane** in camera space.
    /// The bottom plane is a plane parallel to the **zx-plane** at the origin.
    bottom: S,
    /// The vertical position of the **top plane** in camera space.
    /// the top plane is a plane parallel to the **zx-plane** at the origin.
    top: S,
    /// The distance along the **negative z-axis** of the **near plane** from the eye.
    /// The near plane is a plane parallel to the **xy-plane** at the origin.
    near: S,
    /// the distance along the **negative z-axis** of the **far plane** from the eye.
    /// The far plane is a plane parallel to the **xy-plane** at the origin.
    far: S,
    /// The underlying matrix that implements the orthographic projection.
    matrix: Matrix4x4<S>,
}

impl<S> Orthographic3<S> where S: ScalarFloat {
    /// Construct a new orthographic projection.
    pub fn new(left: S, right: S, bottom: S, top: S, near: S, far: S) -> Orthographic3<S> {
        Orthographic3 {
            left: left,
            right: right,
            bottom: bottom,
            top: top,
            near: near,
            far: far,
            matrix: Matrix4x4::from_orthographic(left, right, bottom, top, near, far),
        }
    }

    /// Get the near plane along the **negative z-axis**.
    #[inline]
    pub fn znear(&self) -> S {
        self.near
    }

    /// Get the far plane along the **negative z-axis**.
    #[inline]
    pub fn zfar(&self) -> S {
        self.far
    }

    /// Get the left plane along the **negative x-axis**.
    #[inline]
    pub fn xleft(&self)-> S {
        self.near
    }

    /// Get the right plane along the **positive x-axis**.
    #[inline]
    pub fn xright(&self) -> S {
        self.far
    }

    /// Get the bottom plane along the **negative y-axis**.
    #[inline]
    pub fn ybottom(&self) -> S {
        self.bottom
    }

    /// Get the top plane along the **positive y-axis**.
    #[inline]
    pub fn ytop(&self) -> S {
        self.top
    }

    /// Get the underlying matrix implementing the orthographic transformation.
    #[inline]
    pub fn to_matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the transformation to a point.
    #[inline]
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous()).unwrap()
    }

    /// Apply the transformation to a vector.
    #[inline]
    pub fn project_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Unproject a point from normalized devices coordinates back to camera
    /// view space. 
    ///
    /// This is the inverse operation of `project_point`.
    #[inline]
    pub fn unproject_point(&self, point: &Point3<S>) -> Point3<S> {
        let one_half: S = num_traits::cast(0.5_f64).unwrap();
        let c0r0 =  one_half * (self.right - self.left);
        let c1r1 =  one_half * (self.top - self.bottom);
        let c2r2 = -one_half * (self.far - self.near);
        let c3r0 =  one_half * (self.left + self.right);
        let c3r1 =  one_half * (self.bottom + self.top);
        let c3r2 = -one_half * (self.far + self.near);

        Point3::new(
            c0r0 * point.x + c3r0,
            c1r1 * point.y + c3r1,
            c2r2 * point.z + c3r2
        )
    }

    /// Unproject a vector from normalized device coordinates back to
    /// camera view space. 
    ///
    /// This is the inverse operation of `project_vector`.
    #[inline]
    pub fn unproject_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let one_half: S = num_traits::cast(0.5_f64).unwrap();
        let c0r0 =  one_half * (self.right - self.left);
        let c1r1 =  one_half * (self.top - self.bottom);
        let c2r2 = -one_half * (self.far - self.near);

        Vector3::new(c0r0 * vector.x, c1r1 * vector.y, c2r2 * vector.z)
    }
}

impl<S> AsRef<Matrix4x4<S>> for Orthographic3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for Orthographic3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Orthographic3 [{}]",
            self.matrix
        )
    }
}

impl<S> approx::AbsDiffEq for Orthographic3<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Orthographic3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Orthographic3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}


/// An orthographic projection based on the `near` plane, the `far` plane and 
/// the vertical field of view angle `fovy` and the horizontal/vertical aspect 
/// ratio `aspect`.
///
/// We assume the following constraints to make a useful orthographic projection 
/// transformation.
/// ```text
/// 0 radians < fovy < pi radians
/// aspect > 0
/// near < far (along the negative z-axis)
/// ```
/// This orthographic projection model imposes some constraints on the more 
/// general orthographic specification based on the arbitrary planes. The `fovy` 
/// parameter combined with the aspect ratio `aspect` ensures that the top and 
/// bottom planes are the same distance from the eye position along the vertical 
/// axis on opposite side. They ensure that the `left` and `right` planes are 
/// equidistant from the eye on opposite sides along the horizontal axis.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OrthographicFov3<S> {
    /// The vertical field of view angle of the orthographic transformation
    /// viewport.
    fovy: Radians<S>, 
    /// The ratio of the horizontal width to the vertical height.
    aspect: S, 
    /// The position of the near plane along the **negative z-axis**.
    near: S, 
    /// The position of the far plane along the **negative z-axis**.
    far: S,
    /// The underlying matrix that implements the orthographic projection.
    matrix: Matrix4x4<S>,
}

impl<S> OrthographicFov3<S> where S: ScalarFloat {
    /// Construct a new orthographic projection.
    pub fn new<A: Into<Radians<S>>>(fovy: A, aspect: S, near: S, far: S) -> OrthographicFov3<S> {
        let fovy_rad = fovy.into();
        OrthographicFov3 {
            fovy: fovy_rad,
            aspect: aspect,
            near: near,
            far: far,
            matrix: Matrix4x4::from_orthographic_fov(fovy_rad, aspect, near, far),
        }
    }

    /// Get the vertical field of view angle.
    #[inline]
    pub fn vfov(&self) -> Radians<S> {
        self.fovy
    }

    /// Get the near plane along the **negative z-axis**.
    #[inline]
    pub fn znear(&self) -> S {
        self.near
    }

    /// Get the far plane along the **negative z-axis**.
    #[inline]
    pub fn zfar(&self) -> S {
        self.far
    }

    /// Get the aspect ratio.
    #[inline]
    pub fn aspect(&self) -> S {
        self.aspect
    }

    /// Get the underlying matrix implementing the orthographic transformation.
    #[inline]
    pub fn to_matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the transformation to a point.
    #[inline]
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::from_homogeneous(self.matrix * point.to_homogeneous()).unwrap()
    }

    /// Apply the transformation to a vector.
    #[inline]
    pub fn project_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        (self.matrix * vector.expand(S::zero())).contract()
    }

    /// Unproject a point from normalized devices coordinates back to camera
    /// view space. 
    ///
    /// This is the inverse operation of `project_point`.
    #[inline]
    pub fn unproject_point(&self, point: &Point3<S>) -> Point3<S> {
        let one_half: S = num_traits::cast(0.5_f64).unwrap();
        let width = self.far * Angle::tan(self.fovy * one_half);
        let height = width / self.aspect;
        let left = -width * one_half;
        let right = width * one_half;
        let bottom = -height * one_half;
        let top = height * one_half;
        let near = self.near;
        let far = self.far;
        let c0r0 =  one_half * (right - left);
        let c1r1 =  one_half * (top - bottom);
        let c2r2 = -one_half * (far - near);
        let c3r0 =  one_half * (left + right);
        let c3r1 =  one_half * (bottom + top);
        let c3r2 = -one_half * (far + near);

        Point3::new(
            c0r0 * point.x + c3r0,
            c1r1 * point.y + c3r1,
            c2r2 * point.z + c3r2
        )
    }

    /// Unproject a vector from normalized device coordinates back to
    /// camera view space. 
    ///
    /// This is the inverse operation of `project_vector`.
    #[inline]
    pub fn unproject_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let one_half: S = num_traits::cast(0.5_f64).unwrap();
        let width = self.far * Angle::tan(self.fovy * one_half);
        let height = width / self.aspect;
        let left = -width * one_half;
        let right = width * one_half;
        let bottom = -height * one_half;
        let top = height * one_half;
        let near = self.near;
        let far = self.far;
        let c0r0 =  one_half * (right - left);
        let c1r1 =  one_half * (top - bottom);
        let c2r2 = -one_half * (far - near);

        Vector3::new(c0r0 * vector.x, c1r1 * vector.y, c2r2 * vector.z)
    }
}

impl<S> AsRef<Matrix4x4<S>> for OrthographicFov3<S> {
    #[inline]
    fn as_ref(&self) -> &Matrix4x4<S> {
        &self.matrix
    }
}

impl<S> fmt::Display for OrthographicFov3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "OrthographicFov3 [{}]",
            self.matrix
        )
    }
}

impl<S> approx::AbsDiffEq for OrthographicFov3<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for OrthographicFov3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for OrthographicFov3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

