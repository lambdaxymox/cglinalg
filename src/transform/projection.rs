use crate::base_numeric::{
    SimdScalarFloat,
};
use crate::base::{
    Angle,
    Radians,
    Matrix4x4,
    Point3,
    Vector3,
};

use core::fmt;
use core::ops;


/// A perspective projection transformation based on arbitrary `left`, `right`, 
/// `bottom`, `top`, `near`, and `far` planes.
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
    /// The horizontal position of the left-hand plane in camera space.
    /// The left-hand plane is a plane parallel to the **yz-plane** at
    /// the origin.
    left: S,
    /// The horizontal position of the right-hand plane in camera space.
    /// The right-hand plane is a plane parallel to the **yz-plane** at
    /// the origin.
    right: S,
    /// The vertical position of the bottom plane in camera space.
    /// The bottom plane is a plane parallel to the **zx-plane** at the origin.
    bottom: S,
    /// The vertical position of the top plane in camera space.
    /// the top plane is a plane parallel to the **zx-plane** at the origin.
    top: S,
    /// The distance along the **negative z-axis** of the near plane from the eye.
    /// The near plane is a plane parallel to the **xy-plane** at the origin.
    near: S,
    /// the distance along the **negative z-axis** of the far plane from the eye.
    /// The far plane is a plane parallel to the **xy-plane** at the origin.
    far: S,
    /// The underlying matrix implementing the perspective projection.
    matrix: Matrix4x4<S>,
}

impl<S> Perspective3<S> 
where 
    S: SimdScalarFloat 
{
    /// Construct a new perspective projection transformation.
    ///
    /// The perspective projection transformation uses a right-handed 
    /// coordinate system where the **negative z-axis** is the depth direction.
    pub fn new(left: S, right: S, bottom: S, top: S, near: S, far: S) -> Self {
        Self {
            left,
            right,
            bottom,
            top,
            near,
            far,
            matrix: Matrix4x4::from_perspective(left, right, bottom, top, near, far),
        }
    }

    /// Get the near plane along the **negative z-axis**.
    #[inline]
    pub const fn near_z(&self) -> S {
        self.near
    }

    /// Get the far plane along the **negative z-axis**.
    #[inline]
    pub const fn far_z(&self) -> S {
        self.far
    }

    /// Get the left plane along the **negative x-axis**.
    #[inline]
    pub const fn left_x(&self)-> S {
        self.left
    }

    /// Get the right plane along the **positive x-axis**.
    #[inline]
    pub const fn right_x(&self) -> S {
        self.right
    }

    /// Get the bottom plane along the **negative y-axis**.
    #[inline]
    pub const fn bottom_y(&self) -> S {
        self.bottom
    }

    /// Get the top plane along the **positive y-axis**.
    #[inline]
    pub const fn top_y(&self) -> S {
        self.top
    }

    /// Get the matrix that implements the perspective projection transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Perspective3,
    /// #     Matrix4x4, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let result = perspective.matrix();
    /// let expected = Matrix4x4::new(
    ///     1_f64 / 3_f64, 0_f64,          0_f64,             0_f64,
    ///     0_f64,         1_f64 / 2_f64,  0_f64,             0_f64,
    ///     0_f64,         0_f64,         -101_f64 / 99_f64, -1_f64,
    ///     0_f64,         0_f64,         -200_f64 / 99_f64,  0_f64
    /// );
    ///
    /// assert_relative_eq!(result, &expected, epsilon = 1e-10);
    /// 
    #[inline]
    pub const fn matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the projective projection transformation to a point.
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Perspective3,
    /// #     Point3, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let point = Point3::new(-1_f64, -1_f64, 4_f64);
    /// let expected = Point3::new(1_f64 / 12_f64, 1_f64 / 8_f64, 604_f64 / 396_f64);
    /// let result = perspective.project_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        let inverse_w = -S::one() / point.z;
        
        Point3::new(
            (self.matrix.c0r0 * point.x + self.matrix.c2r0 * point.z) * inverse_w,
            (self.matrix.c1r1 * point.y + self.matrix.c3r1 * point.z) * inverse_w,
            (self.matrix.c2r2 * point.z + self.matrix.c3r2) * inverse_w
        )
    }

    /// Apply the perspective projection transformation to a vector.
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Perspective3,
    /// #     Vector3, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let vector = Vector3::new(-1_f64, -1_f64, 4_f64);
    /// let expected = Vector3::new(1_f64 / 12_f64, 1_f64 / 8_f64, 604_f64 / 396_f64);
    /// let result = perspective.project_vector(&vector);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn project_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let projected_vector = self.matrix * vector.extend(S::one());
        let one_div_w = S::one() / projected_vector.w;
        
        (projected_vector * one_div_w).contract()
    }

    /// Unproject a point from normalized device coordinates back to camera
    /// view space. 
    /// 
    /// This is the inverse operation of `project_point`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Perspective3,
    /// #     Point3, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let point = Point3::new(-1_f64, -1_f64, 4_f64);
    /// let expected = point;
    /// let projected_point = perspective.project_point(&point);
    /// let result = perspective.unproject_point(&projected_point);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
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
        let one = S::one();
        let two = one + one;
        let c0r0 = (self.right - self.left) / (two * self.near);
        let c1r1 = (self.top - self.bottom) / (two * self.near);
        let c2r3 =  (self.near - self.far) / (two * self.far * self.near);
        let c3r0 =  (self.left + self.right) / (two * self.near);
        let c3r1 =  (self.bottom + self.top) / (two * self.near);
        let c3r2 = -one;
        let c3r3 =  (self.far + self.near) / (two * self.far * self.near);
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
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Perspective3,
    /// #     Vector3, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let vector = Vector3::new(-1_f64, -1_f64, 4_f64);
    /// let expected = vector;
    /// let projected_vector = perspective.project_vector(&vector);
    /// let result = perspective.unproject_vector(&projected_vector);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
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
        let one = S::one();
        let two = one + one;
        let c0r0 = (self.right - self.left) / (two * self.near);
        let c1r1 = (self.top - self.bottom) / (two * self.near);
        let c2r3 =  (self.near - self.far) / (two * self.far * self.near);
        let c3r0 =  (self.left + self.right) / (two * self.near);
        let c3r1 =  (self.bottom + self.top) / (two * self.near);
        let c3r2 = -one;
        let c3r3 =  (self.far + self.near) / (two * self.far * self.near);
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

impl<S> fmt::Display for Perspective3<S> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Perspective3 [{}]",
            self.matrix
        )
    }
}

impl<S> approx::AbsDiffEq for Perspective3<S> 
where 
    S: SimdScalarFloat 
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Matrix4x4::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
            && S::abs_diff_eq(&self.left, &other.left, epsilon)
            && S::abs_diff_eq(&self.right, &other.right, epsilon)
            && S::abs_diff_eq(&self.bottom, &other.bottom, epsilon)
            && S::abs_diff_eq(&self.top, &other.top, epsilon)
            && S::abs_diff_eq(&self.near, &other.near, epsilon)
            && S::abs_diff_eq(&self.far, &other.far, epsilon)
    }
}

impl<S> approx::RelativeEq for Perspective3<S> 
where 
    S: SimdScalarFloat,
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
            && S::relative_eq(&self.left, &other.left, epsilon, max_relative)
            && S::relative_eq(&self.right, &other.right, epsilon, max_relative)
            && S::relative_eq(&self.bottom, &other.bottom, epsilon, max_relative)
            && S::relative_eq(&self.top, &other.top, epsilon, max_relative)
            && S::relative_eq(&self.near, &other.near, epsilon, max_relative)
            && S::relative_eq(&self.far, &other.far, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Perspective3<S> 
where 
    S: SimdScalarFloat   
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
            && S::ulps_eq(&self.left, &other.left, epsilon, max_ulps)
            && S::ulps_eq(&self.right, &other.right, epsilon, max_ulps)
            && S::ulps_eq(&self.bottom, &other.bottom, epsilon, max_ulps)
            && S::ulps_eq(&self.top, &other.top, epsilon, max_ulps)
            && S::ulps_eq(&self.near, &other.near, epsilon, max_ulps)
            && S::ulps_eq(&self.far, &other.far, epsilon, max_ulps)
    }
}

impl<S> ops::Mul<Point3<S>> for Perspective3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for Perspective3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.project_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &Perspective3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b Perspective3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.project_point(other)
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
    vfov: Radians<S>,
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
where 
    S: SimdScalarFloat
{
    /// Construct a new perspective projection transformation.
    pub fn new<A: Into<Radians<S>>>(vfov: A, aspect: S, near: S, far: S) -> Self {
        let spec_vfov = vfov.into();

        Self {
            vfov: spec_vfov,
            aspect,
            near,
            far,
            matrix: Matrix4x4::from_perspective_fov(spec_vfov, aspect, near, far),
        }
    }

    /// Get the vertical field of view angle.
    #[inline]
    pub const fn vfov(&self) -> Radians<S> {
        self.vfov
    }

    /// Get the near plane along the **negative z-axis**.
    #[inline]
    pub const fn near_z(&self) -> S {
        self.near
    }

    /// Get the far plane along the **negative z-axis**.
    #[inline]
    pub const fn far_z(&self) -> S {
        self.far
    }

    /// Get the aspect ratio. The aspect ratio is the ratio of the 
    /// width of the viewing plane of the viewing frustum to the height of the 
    /// viewing plane of the viewing frustum.
    #[inline]
    pub const fn aspect(&self) -> S {
        self.aspect
    }

    /// Get the matrix that implements the perspective projection transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     PerspectiveFov3,
    /// #     Matrix4x4,
    /// #     Degrees,
    /// #     Angle,
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let tan_half_vfov = (vfov / 2_f64).tan();
    /// let aspect = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect, near, far);
    /// let c0r0 = 1_f64 / (aspect * tan_half_vfov);
    /// let c1r1 = 1_f64 / (tan_half_vfov);
    /// let c2r2 = -(far + near) / (far - near);
    /// let c3r2 = (-2_f64 * far * near) / (far - near);
    /// let expected = Matrix4x4::new(
    ///     c0r0,  0_f64, 0_f64,  0_f64,
    ///     0_f64, c1r1,  0_f64,  0_f64,
    ///     0_f64, 0_f64, c2r2,  -1_f64,
    ///     0_f64, 0_f64, c3r2,   0_f64
    /// );
    /// let result = perspective.matrix();
    ///
    /// assert_eq!(result, &expected);
    /// ```
    #[inline]
    pub const fn matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the perspective projection transformation to a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     PerspectiveFov3,
    /// #     Matrix4x4,
    /// #     Degrees,
    /// #     Angle,
    /// #     Point3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let tan_half_vfov = (vfov / 2_f64).tan();
    /// let aspect = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect, near, far);
    /// let point = Point3::new(-1_f64, -1_f64, 30_f64);
    /// let expected = Point3::new(3_f64 / 120_f64, 1_f64 / 30_f64, 3230_f64 / 2970_f64);
    /// let result = perspective.project_point(&point);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        let inverse_w = -S::one() / point.z;

        Point3::new(
            (self.matrix.c0r0 * point.x) * inverse_w,
            (self.matrix.c1r1 * point.y) * inverse_w,
            (self.matrix.c2r2 * point.z + self.matrix.c3r2) * inverse_w
        )
    }

    /// Apply the perspective projection transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     PerspectiveFov3,
    /// #     Matrix4x4,
    /// #     Degrees,
    /// #     Angle,
    /// #     Vector3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let tan_half_vfov = (vfov / 2_f64).tan();
    /// let aspect = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect, near, far);
    /// let vector = Vector3::new(-1_f64, -1_f64, 30_f64);
    /// let expected = Vector3::new(3_f64 / 120_f64, 1_f64 / 30_f64, 3230_f64 / 2970_f64);
    /// let result = perspective.project_vector(&vector);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn project_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let projected_vector = self.matrix * vector.extend(S::one());
        let one_div_w = S::one() / projected_vector.w;
        
        (projected_vector * one_div_w).contract()
    }

    /// Unproject a point from normalized device coordinates back to camera
    /// view space. 
    /// 
    /// This is the inverse operation of `project_point`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     PerspectiveFov3,
    /// #     Matrix4x4,
    /// #     Degrees,
    /// #     Angle,
    /// #     Point3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let tan_half_vfov = (vfov / 2_f64).tan();
    /// let aspect = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect, near, far);
    /// let point = Point3::new(-1_f64, -1_f64, 30_f64);
    /// let expected = point;
    /// let projected_point = perspective.project_point(&point);
    /// let result = perspective.unproject_point(&projected_point);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
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
        let one = S::one();
        let two = one + one;
        let near = self.near;
        let far = self.far;
        let tan_vfov_div_2 = Radians::tan(self.vfov / two); 
        let top = self.near * tan_vfov_div_2;
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
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     PerspectiveFov3,
    /// #     Matrix4x4,
    /// #     Degrees,
    /// #     Angle,
    /// #     Vector3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let tan_half_vfov = (vfov / 2_f64).tan();
    /// let aspect = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect, near, far);
    /// let vector = Vector3::new(-1_f64, -1_f64, 30_f64);
    /// let expected = vector;
    /// let projected_vector = perspective.project_vector(&vector);
    /// let result = perspective.unproject_vector(&projected_vector); 
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
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
        let one = S::one();
        let two = one + one;
        let near = self.near;
        let far = self.far;
        let tan_vfov_div_2 = Radians::tan(self.vfov / two); 
        let top = self.near * tan_vfov_div_2;
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

impl<S> fmt::Display for PerspectiveFov3<S> 
where 
    S: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "PerspectiveFov3 [{}]",
            self.matrix
        )
    }
}

impl<S> approx::AbsDiffEq for PerspectiveFov3<S> 
where 
    S: SimdScalarFloat
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Matrix4x4::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
            && Radians::abs_diff_eq(&self.vfov, &other.vfov, epsilon)
            && S::abs_diff_eq(&self.aspect, &other.aspect, epsilon)
            && S::abs_diff_eq(&self.near, &other.near, epsilon)
            && S::abs_diff_eq(&self.far, &other.far, epsilon)
    }
}

impl<S> approx::RelativeEq for PerspectiveFov3<S> 
where 
    S: SimdScalarFloat  
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
            && Radians::relative_eq(&self.vfov, &other.vfov, epsilon, max_relative)
            && S::relative_eq(&self.aspect, &other.aspect, epsilon, max_relative)
            && S::relative_eq(&self.near, &other.near, epsilon, max_relative)
            && S::relative_eq(&self.far, &other.far, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for PerspectiveFov3<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
            && Radians::ulps_eq(&self.vfov, &other.vfov, epsilon, max_ulps)
            && S::ulps_eq(&self.aspect, &other.aspect, epsilon, max_ulps)
            && S::ulps_eq(&self.near, &other.near, epsilon, max_ulps)
            && S::ulps_eq(&self.far, &other.far, epsilon, max_ulps)
    }
}

impl<S> ops::Mul<Point3<S>> for PerspectiveFov3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for PerspectiveFov3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.project_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &PerspectiveFov3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b PerspectiveFov3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.project_point(other)
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

impl<S> Orthographic3<S> 
where 
    S: SimdScalarFloat
{
    /// Construct a new orthographic projection.
    pub fn new(left: S, right: S, bottom: S, top: S, near: S, far: S) -> Self {
        Self {
            left,
            right,
            bottom,
            top,
            near,
            far,
            matrix: Matrix4x4::from_orthographic(left, right, bottom, top, near, far),
        }
    }

    /// Get the near plane along the **negative z-axis**.
    #[inline]
    pub const fn near_z(&self) -> S {
        self.near
    }

    /// Get the far plane along the **negative z-axis**.
    #[inline]
    pub const fn far_z(&self) -> S {
        self.far
    }

    /// Get the left plane along the **negative x-axis**.
    #[inline]
    pub const fn left_x(&self)-> S {
        self.left
    }

    /// Get the right plane along the **positive x-axis**.
    #[inline]
    pub const fn right_x(&self) -> S {
        self.right
    }

    /// Get the bottom plane along the **negative y-axis**.
    #[inline]
    pub const fn bottom_y(&self) -> S {
        self.bottom
    }

    /// Get the top plane along the **positive y-axis**.
    #[inline]
    pub const fn top_y(&self) -> S {
        self.top
    }

    /// Get the underlying matrix implementing the orthographic transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Orthographic3,
    /// #     Matrix4x4,
    /// # };
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let expected = Matrix4x4::new(
    ///     1_f64 / 6_f64, 0_f64,          0_f64,           0_f64,
    ///     0_f64,         1_f64 / 4_f64,  0_f64,           0_f64, 
    ///     0_f64,         0_f64,         -1_f64 / 50_f64,  0_f64,
    ///     0_f64,         0_f64,         -51_f64 / 50_f64, 1_f64
    /// );
    /// let result = orthographic.matrix();
    ///
    /// assert_eq!(result, &expected);
    /// ```
    #[inline]
    pub const fn matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the orthographic projection transformation to a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Orthographic3,
    /// #     Point3,
    /// # };
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let point = Point3::new(2_f64, 3_f64, 30_f64);
    /// let expected = Point3::new(1_f64 / 3_f64, 3_f64 / 4_f64, -81_f64 / 50_f64);
    /// let result = orthographic.project_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::new(
            self.matrix.c0r0 * point.x + self.matrix.c3r0,
            self.matrix.c1r1 * point.y + self.matrix.c3r1,
            self.matrix.c2r2 * point.z + self.matrix.c3r2
        )
    }

    /// Apply the orthographic projection transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Orthographic3,
    /// #     Vector3,
    /// # };
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let vector = Vector3::new(2_f64, 3_f64, 30_f64);
    /// let expected = Vector3::new(1_f64 / 3_f64, 3_f64 / 4_f64, -3_f64 / 5_f64);
    /// let result = orthographic.project_vector(&vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn project_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        Vector3::new(
            self.matrix.c0r0 * vector.x,
            self.matrix.c1r1 * vector.y,
            self.matrix.c2r2 * vector.z
        )
    }

    /// Unproject a point from normalized devices coordinates back to camera
    /// view space. 
    ///
    /// This is the inverse operation of `project_point`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Orthographic3,
    /// #     Point3,
    /// # };
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let point = Point3::new(2_f64, 3_f64, 30_f64);
    /// let expected = point;
    /// let projected_point = orthographic.project_point(&point);
    /// let result = orthographic.unproject_point(&projected_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Orthographic3,
    /// #     Vector3,
    /// # };
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let vector = Vector3::new(2_f64, 3_f64, 30_f64);
    /// let expected = vector;
    /// let projected_vector = orthographic.project_vector(&vector);
    /// let result = orthographic.unproject_vector(&projected_vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
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

impl<S> fmt::Display for Orthographic3<S> 
where 
    S: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "Orthographic3 [{}]",
            self.matrix
        )
    }
}

impl<S> approx::AbsDiffEq for Orthographic3<S> 
where 
    S: SimdScalarFloat
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Matrix4x4::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
            && S::abs_diff_eq(&self.left, &other.left, epsilon)
            && S::abs_diff_eq(&self.right, &other.right, epsilon)
            && S::abs_diff_eq(&self.bottom, &other.bottom, epsilon)
            && S::abs_diff_eq(&self.top, &other.top, epsilon)
            && S::abs_diff_eq(&self.near, &other.near, epsilon)
            && S::abs_diff_eq(&self.far, &other.far, epsilon)
    }
}

impl<S> approx::RelativeEq for Orthographic3<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
            && S::relative_eq(&self.left, &other.left, epsilon, max_relative)
            && S::relative_eq(&self.right, &other.right, epsilon, max_relative)
            && S::relative_eq(&self.bottom, &other.bottom, epsilon, max_relative)
            && S::relative_eq(&self.top, &other.top, epsilon, max_relative)
            && S::relative_eq(&self.near, &other.near, epsilon, max_relative)
            && S::relative_eq(&self.far, &other.far, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Orthographic3<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
            && S::ulps_eq(&self.left, &other.left, epsilon, max_ulps)
            && S::ulps_eq(&self.right, &other.right, epsilon, max_ulps)
            && S::ulps_eq(&self.bottom, &other.bottom, epsilon, max_ulps)
            && S::ulps_eq(&self.top, &other.top, epsilon, max_ulps)
            && S::ulps_eq(&self.near, &other.near, epsilon, max_ulps)
            && S::ulps_eq(&self.far, &other.far, epsilon, max_ulps)
    }
}

impl<S> ops::Mul<Point3<S>> for Orthographic3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for Orthographic3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.project_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &Orthographic3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b Orthographic3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.project_point(other)
    }
}



/// An orthographic projection based on the `near` plane, the `far` plane and 
/// the vertical field of view angle `vfov` and the horizontal/vertical aspect 
/// ratio `aspect`.
///
/// We assume the following constraints to make a useful orthographic projection 
/// transformation.
/// ```text
/// 0 radians < vfov < pi radians
/// aspect > 0
/// near < far (along the negative z-axis)
/// ```
/// This orthographic projection model imposes some constraints on the more 
/// general orthographic specification based on the arbitrary planes. The `vfov` 
/// parameter combined with the aspect ratio `aspect` ensures that the top and 
/// bottom planes are the same distance from the eye position along the vertical 
/// axis on opposite side. They ensure that the `left` and `right` planes are 
/// equidistant from the eye on opposite sides along the horizontal axis.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OrthographicFov3<S> {
    /// The vertical field of view angle of the orthographic transformation
    /// viewport.
    vfov: Radians<S>, 
    /// The ratio of the horizontal width to the vertical height.
    aspect: S, 
    /// The position of the near plane along the **negative z-axis**.
    near: S, 
    /// The position of the far plane along the **negative z-axis**.
    far: S,
    /// The underlying matrix that implements the orthographic projection.
    matrix: Matrix4x4<S>,
}

impl<S> OrthographicFov3<S> 
where 
    S: SimdScalarFloat
{
    /// Construct a new orthographic projection.
    pub fn new<A: Into<Radians<S>>>(vfov: A, aspect: S, near: S, far: S) -> Self {
        let vfov_rad = vfov.into();
        Self {
            vfov: vfov_rad,
            aspect,
            near,
            far,
            matrix: Matrix4x4::from_orthographic_fov(vfov_rad, aspect, near, far),
        }
    }

    /// Get the vertical field of view angle.
    #[inline]
    pub const fn vfov(&self) -> Radians<S> {
        self.vfov
    }

    /// Get the near plane along the **negative z-axis**.
    #[inline]
    pub const fn near_z(&self) -> S {
        self.near
    }

    /// Get the far plane along the **negative z-axis**.
    #[inline]
    pub const fn far_z(&self) -> S {
        self.far
    }

    /// Get the aspect ratio.
    #[inline]
    pub const fn aspect(&self) -> S {
        self.aspect
    }

    /// Get the underlying matrix implementing the orthographic transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     OrthographicFov3,
    /// #     Matrix4x4,
    /// #     Degrees,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let aspect = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = OrthographicFov3::new(vfov, aspect, near, far);
    /// let expected = Matrix4x4::new(
    ///     2_f64 / 101_f64, 0_f64,            0_f64,             0_f64, 
    ///     0_f64,           8_f64 / 303_f64,  0_f64,             0_f64, 
    ///     0_f64,           0_f64,           -2_f64 / 100_f64,   0_f64, 
    ///     0_f64,           0_f64,           -102_f64 / 100_f64, 1_f64
    /// );
    /// let result = orthographic.matrix();
    ///
    /// assert_relative_eq!(result, &expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub const fn matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the orthographic projection transformation to a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     OrthographicFov3,
    /// #     Point3,
    /// #     Degrees,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let aspect = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = OrthographicFov3::new(vfov, aspect, near, far);
    /// let point = Point3::new(2_f64, 3_f64, 50_f64);
    /// let expected = Point3::new(4_f64 / 101_f64, 8_f64 / 101_f64, -101_f64 / 50_f64);
    /// let result = orthographic.project_point(&point);
    /// 
    /// assert_relative_eq!(result, &expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        Point3::new(
            self.matrix.c0r0 * point.x + self.matrix.c3r0,
            self.matrix.c1r1 * point.y + self.matrix.c3r1,
            self.matrix.c2r2 * point.z + self.matrix.c3r2
        )
    }

    /// Apply the orthographic projection transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     OrthographicFov3,
    /// #     Vector3,
    /// #     Degrees,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let aspect = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = OrthographicFov3::new(vfov, aspect, near, far);
    /// let vector = Vector3::new(2_f64, 3_f64, 50_f64);
    /// let expected = Vector3::new(4_f64 / 101_f64, 8_f64 / 101_f64, -1_f64);
    /// let result = orthographic.project_vector(&vector);
    /// 
    /// assert_relative_eq!(result, &expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn project_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        Vector3::new(
            self.matrix.c0r0 * vector.x,
            self.matrix.c1r1 * vector.y,
            self.matrix.c2r2 * vector.z
        )
    }

    /// Unproject a point from normalized devices coordinates back to camera
    /// view space. 
    ///
    /// This is the inverse operation of `project_point`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     OrthographicFov3,
    /// #     Point3,
    /// #     Degrees,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let aspect = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = OrthographicFov3::new(vfov, aspect, near, far);
    /// let point = Point3::new(2_f64, 3_f64, 50_f64);
    /// let expected = point;
    /// let projected_point = orthographic.project_point(&point);
    /// let result = orthographic.unproject_point(&projected_point);
    /// 
    /// assert_relative_eq!(result, &expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn unproject_point(&self, point: &Point3<S>) -> Point3<S> {
        let one_half: S = num_traits::cast(0.5_f64).unwrap();
        let width = self.far * Angle::tan(self.vfov * one_half);
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
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     OrthographicFov3,
    /// #     Vector3,
    /// #     Degrees,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let aspect = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = OrthographicFov3::new(vfov, aspect, near, far);
    /// let vector = Vector3::new(2_f64, 3_f64, 50_f64);
    /// let expected = vector;
    /// let projected_vector = orthographic.project_vector(&vector);
    /// let result = orthographic.unproject_vector(&projected_vector);
    /// 
    /// assert_relative_eq!(result, &expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn unproject_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let one_half: S = num_traits::cast(0.5_f64).unwrap();
        let width = self.far * Angle::tan(self.vfov * one_half);
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

impl<S> fmt::Display for OrthographicFov3<S> 
where 
    S: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "OrthographicFov3 [{}]",
            self.matrix
        )
    }
}

impl<S> approx::AbsDiffEq for OrthographicFov3<S> 
where 
    S: SimdScalarFloat
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        Matrix4x4::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
            && Radians::abs_diff_eq(&self.vfov, &other.vfov, epsilon)
            && S::abs_diff_eq(&self.aspect, &other.aspect, epsilon)
            && S::abs_diff_eq(&self.near, &other.near, epsilon)
            && S::abs_diff_eq(&self.far, &other.far, epsilon)
    }
}

impl<S> approx::RelativeEq for OrthographicFov3<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix4x4::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
            && Radians::relative_eq(&self.vfov, &other.vfov, epsilon, max_relative)
            && S::relative_eq(&self.aspect, &other.aspect, epsilon, max_relative)
            && S::relative_eq(&self.near, &other.near, epsilon, max_relative)
            && S::relative_eq(&self.far, &other.far, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for OrthographicFov3<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix4x4::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
            && Radians::ulps_eq(&self.vfov, &other.vfov, epsilon, max_ulps)
            && S::ulps_eq(&self.aspect, &other.aspect, epsilon, max_ulps)
            && S::ulps_eq(&self.near, &other.near, epsilon, max_ulps)
            && S::ulps_eq(&self.far, &other.far, epsilon, max_ulps)
    }
}

impl<S> ops::Mul<Point3<S>> for OrthographicFov3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for OrthographicFov3<S> 
where 
    S: SimdScalarFloat 
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.project_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &OrthographicFov3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b OrthographicFov3<S> 
where 
    S: SimdScalarFloat
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.project_point(other)
    }
}

