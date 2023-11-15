use crate::transform::Transform3;
use cglinalg_core::{
    Matrix4x4,
    Point3,
    Vector3,
};
use cglinalg_numeric::SimdScalarFloat;
use cglinalg_trigonometry::{
    Angle,
    Radians,
};

use core::fmt;
use core::ops;


/// A perspective projection transformation based on arbitrary `left`, `right`,
/// `bottom`, `top`, `near`, and `far` planes.
///
/// The `near` and `far` parameters are the absolute values of the positions
/// of the **near plane** and the **far** plane, respectively, along the
/// **negative z-axis**. In particular, the position of the **near plane** is
/// `z == -near` and the position of the **far plane** is `z == -far`.
///
/// This data type represents a homogeneous matrix representing a perspective
/// projection transformation with a right-handed coordinate system where the
/// perspective camera faces the **negative z-axis** with the **positive x-axis**
/// going to the right, and the **positive y-axis** going up. The perspective view
/// volume is the frustum contained in
/// `[left, right] x [bottom, top] x [-near, -far]`. The normalized device
/// coordinates this transformation maps to are `[-1, 1] x [-1, 1] x [-1, 1]`.
///
/// The underlying matrix is identical to the one used by OpenGL, provided here
/// for reference
/// ```text
/// | m[0, 0]  0         m[2, 0]  0       |
/// | 0        m[1, 1]   m[2, 1]  0       |
/// | 0        0         m[2, 2]  m[3, 2] |
/// | 0        0        -1        0       |
/// where
/// m[0, 0] == 2 * n / (r - l)
/// m[2, 0] == (r + l) / (r - l)
/// m[1, 1] == 2 * n / (t - b)
/// m[2, 1] == (t + b) / (t - b)
/// m[2, 2] == -(f + n) / (f - n)
/// m[3, 2] == - 2 * f * n / (f - n)
/// ```
/// where the matrix entries are indexed in column-major order.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Perspective3<S> {
    matrix: Matrix4x4<S>,
}

impl<S> Perspective3<S>
where
    S: SimdScalarFloat,
{
    /// Construct a new perspective projection transformation.
    ///
    /// The perspective projection transformation uses a right-handed
    /// coordinate system where the **negative z-axis** is the camera view direction.
    ///
    /// # Parameters
    ///
    /// The parameters must satisfy
    /// ```text
    /// left < right
    /// bottom < top
    /// 0 < near < far
    /// ```
    /// to construct a useful perspective projection. In particular, `near` and
    /// `far` are the respective absolute values of the placement of the near and
    /// far planes along the **z-axis**.
    ///
    /// `left` is the horizontal position of the left plane in camera space.
    /// The left plane is a plane parallel to the **yz-plane** along the **x-axis**.
    ///
    /// `right` is the horizontal position of the right plane in camera space.
    /// The right plane is a plane parallel to the **yz-plane** along the **x-axis**.
    ///
    /// `bottom` is the vertical position of the bottom plane in camera space.
    /// The bottom plane is a plane parallel to the **zx-plane** along the **y-axis**.
    ///
    /// `top` is the vertical position of the top plane in camera space.
    /// the top plane is a plane parallel to the **zx-plane** along the **y-axis**.
    ///
    /// `near` is the distance along the **negative z-axis** of the near plane from the
    /// eye in camera space. The near plane is a plane parallel to the **xy-plane** along
    /// the **negative z-axis**.
    ///
    /// `far` the distance along the **negative z-axis** of the far plane from the
    /// eye in camera space. The far plane is a plane parallel to the **xy-plane** along
    /// the **negative z-axis**.
    ///
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    ///
    /// assert_relative_eq!(perspective.left(),   left,   abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(perspective.right(),  right,  abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(perspective.bottom(), bottom, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(perspective.top(),    top,    abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(perspective.near(),   near,   abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(perspective.far(),    far,    abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    pub fn new(left: S, right: S, bottom: S, top: S, near: S, far: S) -> Self {
        Self {
            matrix: Matrix4x4::from_perspective(left, right, bottom, top, near, far),
        }
    }

    /// Get the position of the near plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **xy-plane** positioned along the **negative z-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = near;
    /// let result = perspective.near();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn near(&self) -> S {
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[2][2] + one) / (-self.matrix[2][2] - one);

        ((one - ratio) / (two * ratio)) * self.matrix[3][2]
    }

    /// Get the position of the far plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **xy-plane** positioned along the **negative z-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = far;
    /// let result = perspective.far();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn far(&self) -> S {
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[2][2] + one) / (-self.matrix[2][2] - one);

        ((one - ratio) / two) * self.matrix[3][2]
    }

    /// Get the position of the right plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **yz-plane** positioned along the **positive x-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = right;
    /// let result = perspective.right();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn right(&self) -> S {
        let one = S::one();
        let two = one + one;
        let ratio = (self.matrix[2][0] + one) / (self.matrix[2][0] - one);
        let near = self.near();

        (two * near * (ratio / (ratio - one))) * (one / self.matrix[0][0])
    }

    /// Get the position of the left plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **yz-plane** positioned along the **positive x-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = left;
    /// let result = perspective.left();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn left(&self) -> S {
        let one = S::one();
        let two = one + one;
        let ratio = (self.matrix[2][0] + one) / (self.matrix[2][0] - one);
        let near = self.near();

        (two * near * (one / (ratio - one))) * (one / self.matrix[0][0])
    }

    /// Get the position of the top plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **zx-plane** positioned along the **positive y-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = top;
    /// let result = perspective.top();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn top(&self) -> S {
        let one = S::one();
        let two = one + one;
        let ratio = (self.matrix[2][1] + one) / (self.matrix[2][1] - one);
        let near = self.near();

        (two * near * (ratio / (ratio - one))) * (one / self.matrix[1][1])
    }

    /// Get the position of the bottom plane of the viewing
    /// frustum descibed by the perspective projection of the plane
    /// parallel to the **zx-plane** positioned along the **positive y-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = bottom;
    /// let result = perspective.bottom();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn bottom(&self) -> S {
        let one = S::one();
        let two = one + one;
        let ratio = (self.matrix[2][1] + one) / (self.matrix[2][1] - one);
        let near = self.near();

        (two * near * (one / (ratio - one))) * (one / self.matrix[1][1])
    }

    /// Get the matrix that implements the perspective projection transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = Matrix4x4::new(
    ///     1_f64 / 3_f64, 0_f64,          0_f64,             0_f64,
    ///     0_f64,         1_f64 / 2_f64,  0_f64,             0_f64,
    ///     0_f64,         0_f64,         -101_f64 / 99_f64, -1_f64,
    ///     0_f64,         0_f64,         -200_f64 / 99_f64,  0_f64,
    /// );
    /// let result = perspective.matrix();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub const fn matrix(&self) -> &Matrix4x4<S> {
        &self.matrix
    }

    /// Apply the projective projection transformation to a point.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Perspective3;
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
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        let inverse_w = -S::one() / point.z;

        Point3::new(
            (self.matrix.c0r0 * point.x + self.matrix.c2r0 * point.z) * inverse_w,
            (self.matrix.c1r1 * point.y + self.matrix.c3r1 * point.z) * inverse_w,
            (self.matrix.c2r2 * point.z + self.matrix.c3r2) * inverse_w,
        )
    }

    /// Apply the perspective projection transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Perspective3;
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
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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
    /// This is the inverse operation of [`project_point`].
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Perspective3;
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
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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
        let c0r0 = (self.right() - self.left()) / (two * self.near());
        let c1r1 = (self.top() - self.bottom()) / (two * self.near());
        let c2r3 = (self.near() - self.far()) / (two * self.far() * self.near());
        let c3r0 = (self.left() + self.right()) / (two * self.near());
        let c3r1 = (self.bottom() + self.top()) / (two * self.near());
        let c3r2 = -one;
        let c3r3 = (self.far() + self.near()) / (two * self.far() * self.near());
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
    /// This is the inverse operation of [`project_vector`].
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Perspective3;
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
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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
        let c0r0 = (self.right() - self.left()) / (two * self.near());
        let c1r1 = (self.top() - self.bottom()) / (two * self.near());
        let c2r3 = (self.near() - self.far()) / (two * self.far() * self.near());
        let c3r0 = (self.left() + self.right()) / (two * self.near());
        let c3r1 = (self.bottom() + self.top()) / (two * self.near());
        let c3r2 = -one;
        let c3r3 = (self.far() + self.near()) / (two * self.far() * self.near());
        let w = c2r3 * vector.z + c3r3;
        let inverse_w = one / w;

        Vector3::new(
            (c0r0 * vector.x + c3r0) * inverse_w,
            (c1r1 * vector.y + c3r1) * inverse_w,
            c3r2 * inverse_w,
        )
    }

    /// Convert a perspective projection to a projective matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = Matrix4x4::new(
    ///     1_f64 / 3_f64, 0_f64,          0_f64,             0_f64,
    ///     0_f64,         1_f64 / 2_f64,  0_f64,             0_f64,
    ///     0_f64,         0_f64,         -101_f64 / 99_f64, -1_f64,
    ///     0_f64,         0_f64,         -200_f64 / 99_f64,  0_f64,
    /// );
    /// let result = perspective.to_projective_matrix();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn to_projective_matrix(&self) -> Matrix4x4<S> {
        self.matrix
    }

    /// Convert a perspective projection to a generic transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::{
    /// #     Perspective3,
    /// #     Transform3,
    /// # };
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let left = -3_f64;
    /// let right = 3_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = Transform3::from_matrix_unchecked(Matrix4x4::new(
    ///     1_f64 / 3_f64, 0_f64,          0_f64,             0_f64,
    ///     0_f64,         1_f64 / 2_f64,  0_f64,             0_f64,
    ///     0_f64,         0_f64,         -101_f64 / 99_f64, -1_f64,
    ///     0_f64,         0_f64,         -200_f64 / 99_f64,  0_f64,
    /// ));
    /// let result = perspective.to_transform();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn to_transform(&self) -> Transform3<S> {
        Transform3::from_matrix_unchecked(self.matrix)
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
    S: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Perspective3 [{}]", self.matrix)
    }
}

impl<S> From<Perspective3<S>> for Matrix4x4<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(perspective: Perspective3<S>) -> Matrix4x4<S> {
        perspective.to_projective_matrix()
    }
}

impl<S> ops::Mul<Point3<S>> for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.project_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &Perspective3<S>
where
    S: SimdScalarFloat,
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b Perspective3<S>
where
    S: SimdScalarFloat,
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.project_point(other)
    }
}

impl<S> ops::Mul<Vector3<S>> for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: Vector3<S>) -> Self::Output {
        self.project_vector(&other)
    }
}

impl<S> ops::Mul<&Vector3<S>> for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: &Vector3<S>) -> Self::Output {
        self.project_vector(other)
    }
}

impl<S> ops::Mul<Vector3<S>> for &Perspective3<S>
where
    S: SimdScalarFloat,
{
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: Vector3<S>) -> Self::Output {
        self.project_vector(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Vector3<S>> for &'b Perspective3<S>
where
    S: SimdScalarFloat,
{
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: &'a Vector3<S>) -> Self::Output {
        self.project_vector(other)
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Perspective3Tol<S> {
    matrix: Matrix4x4<S>,
}

impl<S> From<Matrix4x4<S>> for Perspective3Tol<S> {
    #[inline]
    fn from(matrix: Matrix4x4<S>) -> Self {
        Self { matrix }
    }
}

impl<S> From<&Matrix4x4<S>> for Perspective3Tol<S>
where
    S: Copy,
{
    #[inline]
    fn from(matrix: &Matrix4x4<S>) -> Self {
        Self { matrix: *matrix }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Perspective3Diff<S> {
    matrix: Matrix4x4<S>,
}

impl<S> Perspective3Diff<S> {
    #[inline]
    const fn from(matrix: Matrix4x4<S>) -> Self {
        Self { matrix }
    }
}

impl<S> approx_cmp::AbsDiffEq for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type Tolerance = Perspective3Tol<<S as approx_cmp::AbsDiffEq>::Tolerance>;

    #[inline]
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> bool {
        approx_cmp::AbsDiffEq::abs_diff_eq(&self.matrix, &other.matrix, &max_abs_diff.matrix)
    }
}

impl<S> approx_cmp::AbsDiffAllEq for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::AbsDiffAllEq>::AllTolerance;

    #[inline]
    fn abs_diff_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> bool {
        approx_cmp::AbsDiffAllEq::abs_diff_all_eq(&self.matrix, &other.matrix, max_abs_diff)
    }
}

impl<S> approx_cmp::AssertAbsDiffEq for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = Perspective3Diff<<S as approx_cmp::AssertAbsDiffEq>::DebugAbsDiff>;
    type DebugTolerance = Perspective3Tol<<S as approx_cmp::AssertAbsDiffEq>::DebugTolerance>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let matrix = approx_cmp::AssertAbsDiffEq::debug_abs_diff(&self.matrix, &other.matrix);

        Perspective3Diff::from(matrix)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let matrix = approx_cmp::AssertAbsDiffEq::debug_abs_diff_tolerance(&self.matrix, &other.matrix, &max_abs_diff.matrix);

        Perspective3Tol::from(matrix)
    }
}

impl<S> approx_cmp::AssertAbsDiffAllEq for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = Perspective3Tol<<S as approx_cmp::AssertAbsDiffAllEq>::AllDebugTolerance>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let matrix = approx_cmp::AssertAbsDiffAllEq::debug_abs_diff_all_tolerance(&self.matrix, &other.matrix, max_abs_diff);

        Perspective3Tol::from(matrix)
    }
}

impl<S> approx_cmp::RelativeEq for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type Tolerance = Perspective3Tol<<S as approx_cmp::RelativeEq>::Tolerance>;

    #[inline]
    fn relative_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_relative: &Self::Tolerance) -> bool {
        approx_cmp::RelativeEq::relative_eq(&self.matrix, &other.matrix, &max_abs_diff.matrix, &max_relative.matrix)
    }
}

impl<S> approx_cmp::RelativeAllEq for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::RelativeAllEq>::AllTolerance;

    #[inline]
    fn relative_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_relative: &Self::AllTolerance) -> bool {
        approx_cmp::RelativeAllEq::relative_all_eq(&self.matrix, &other.matrix, max_abs_diff, max_relative)
    }
}

impl<S> approx_cmp::AssertRelativeEq for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = Perspective3Diff<<S as approx_cmp::AssertRelativeEq>::DebugAbsDiff>;
    type DebugTolerance = Perspective3Tol<<S as approx_cmp::AssertRelativeEq>::DebugTolerance>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let matrix = approx_cmp::AssertRelativeEq::debug_abs_diff(&self.matrix, &other.matrix);

        Perspective3Diff::from(matrix)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let matrix = approx_cmp::AssertRelativeEq::debug_abs_diff_tolerance(&self.matrix, &other.matrix, &max_abs_diff.matrix);

        Perspective3Tol::from(matrix)
    }

    #[inline]
    fn debug_relative_tolerance(&self, other: &Self, max_relative: &Self::Tolerance) -> Self::DebugTolerance {
        let matrix = approx_cmp::AssertRelativeEq::debug_relative_tolerance(&self.matrix, &other.matrix, &max_relative.matrix);

        Perspective3Tol::from(matrix)
    }
}

impl<S> approx_cmp::AssertRelativeAllEq for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = Perspective3Tol<<S as approx_cmp::AssertRelativeAllEq>::AllDebugTolerance>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let matrix = approx_cmp::AssertRelativeAllEq::debug_abs_diff_all_tolerance(&self.matrix, &other.matrix, max_abs_diff);

        Perspective3Tol::from(matrix)
    }

    #[inline]
    fn debug_relative_all_tolerance(&self, other: &Self, max_relative: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let matrix = approx_cmp::AssertRelativeAllEq::debug_relative_all_tolerance(&self.matrix, &other.matrix, max_relative);

        Perspective3Tol::from(matrix)
    }
}

impl<S> approx_cmp::UlpsEq for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type Tolerance = Perspective3Tol<<S as approx_cmp::UlpsEq>::Tolerance>;
    type UlpsTolerance = Perspective3Tol<<S as approx_cmp::UlpsEq>::UlpsTolerance>;

    fn ulps_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_ulps: &Self::UlpsTolerance) -> bool {
        approx_cmp::UlpsEq::ulps_eq(&self.matrix, &other.matrix, &max_abs_diff.matrix, &max_ulps.matrix)
    }
}

impl<S> approx_cmp::UlpsAllEq for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::UlpsAllEq>::AllTolerance;
    type AllUlpsTolerance = <S as approx_cmp::UlpsAllEq>::AllUlpsTolerance;

    #[inline]
    fn ulps_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_ulps: &Self::AllUlpsTolerance) -> bool {
        approx_cmp::UlpsAllEq::ulps_all_eq(&self.matrix, &other.matrix, max_abs_diff, max_ulps)
    }
}

impl<S> approx_cmp::AssertUlpsEq for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = Perspective3Diff<<S as approx_cmp::AssertUlpsEq>::DebugAbsDiff>;
    type DebugUlpsDiff = Perspective3Diff<<S as approx_cmp::AssertUlpsEq>::DebugUlpsDiff>;
    type DebugTolerance = Perspective3Tol<<S as approx_cmp::AssertUlpsEq>::DebugTolerance>;
    type DebugUlpsTolerance = Perspective3Tol<<S as approx_cmp::AssertUlpsEq>::DebugUlpsTolerance>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let matrix = approx_cmp::AssertUlpsEq::debug_abs_diff(&self.matrix, &other.matrix);

        Perspective3Diff::from(matrix)
    }

    #[inline]
    fn debug_ulps_diff(&self, other: &Self) -> Self::DebugUlpsDiff {
        let matrix = approx_cmp::AssertUlpsEq::debug_ulps_diff(&self.matrix, &other.matrix);

        Perspective3Diff::from(matrix)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let matrix = approx_cmp::AssertUlpsEq::debug_abs_diff_tolerance(&self.matrix, &other.matrix, &max_abs_diff.matrix);

        Perspective3Tol::from(matrix)
    }

    #[inline]
    fn debug_ulps_tolerance(&self, other: &Self, max_ulps: &Self::UlpsTolerance) -> Self::DebugUlpsTolerance {
        let matrix = approx_cmp::AssertUlpsEq::debug_ulps_tolerance(&self.matrix, &other.matrix, &max_ulps.matrix);

        Perspective3Tol::from(matrix)
    }
}

impl<S> approx_cmp::AssertUlpsAllEq for Perspective3<S>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = Perspective3Tol<<S as approx_cmp::AssertUlpsAllEq>::AllDebugTolerance>;
    type AllDebugUlpsTolerance = Perspective3Tol<<S as approx_cmp::AssertUlpsAllEq>::AllDebugUlpsTolerance>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let matrix = approx_cmp::AssertUlpsAllEq::debug_abs_diff_all_tolerance(&self.matrix, &other.matrix, max_abs_diff);

        Perspective3Tol::from(matrix)
    }

    #[inline]
    fn debug_ulps_all_tolerance(&self, other: &Self, max_ulps: &Self::AllUlpsTolerance) -> Self::AllDebugUlpsTolerance {
        let matrix = approx_cmp::AssertUlpsAllEq::debug_ulps_all_tolerance(&self.matrix, &other.matrix, max_ulps);

        Perspective3Tol::from(matrix)
    }
}



/// A perspective projection transformation for converting from camera space to
/// normalized device coordinates based on the perspective field of view model.
///
/// The `near` and `far` parameters are the absolute values of the positions
/// of the **near plane** and the **far** plane, respectively, along the
/// **negative z-axis**. In particular, the position of the **near plane** is
/// `z == -near` and the position of the **far plane** is `z == -far`. The
/// parameter `aspect_ratio` is the ratio of the width of the viewport to the
/// height of the viewport.
///
/// This data type represents a homogeneous matrix representing a perspective
/// projection transformation with a right-handed coordinate system where the
/// perspective camera faces the **negative z-axis** with the **positive x-axis**
/// going to the right, and the **positive y-axis** going up. The perspective view
/// volume is the symmetric frustum contained in
/// `[-right, right] x [-top, top] x [-near, -far]`, where
/// ```text
/// tan(vfov / 2) == top / near
/// right == aspect_ratio * top == aspect_ratio * n * tan(vfov / 2)
/// top == near * tan(vfov / 2)
/// ```
/// The normalized device coordinates this transformation maps to are
/// `[-1, 1] x [-1, 1] x [-1, 1]`.
///
/// The underlying matrix is identical to the one used by OpenGL, provided here
/// for reference
/// ```text
/// | m[0, 0] 0         0        0       |
/// | 0       m[1, 1]   0        0       |
/// | 0       0         m[2, 2]  m[3, 2] |
/// | 0       0        -1        0       |
/// where
/// m[0, 0] == 1 / (aspect_ratio * tan(vfov / 2))
/// m[1, 1] == 1 / tan(vfov / 2)
/// m[2, 2] == -(f + n) / (f - n)
/// m[3, 2] == -2 * f * n / (f - n)
/// ```
/// where the matrix entries are indexed in column-major order.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveFov3<S> {
    matrix: Matrix4x4<S>,
}

impl<S> PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    /// Construct a new perspective projection transformation.
    ///
    /// # Parameters
    ///
    /// The parameters must satisfy
    /// ```text
    /// 0 < near < far (along the negative z-axis)
    /// ```
    /// to construct a useful perspective projection. In particular, `near` and
    /// `far` are the respective absolute values of the placement of the near and
    /// far planes along the **z-axis**.
    ///
    /// `vfov` is the angle of the field of view of the perspective projection.
    ///
    /// `aspect` is the ratio of the width of the horizontal span of the viewport to
    /// height of the vertical span of the viewport.
    ///
    /// `near` is the distance along the **negative z-axis** of the near plane from the
    /// eye in camera space. The near plane is a plane parallel to the **xy-plane** along
    /// the **negative z-axis**.
    ///
    /// `far` the distance along the **negative z-axis** of the far plane from the
    /// eye in camera space. The far plane is a plane parallel to the **xy-plane** along
    /// the **negative z-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    ///
    /// assert_relative_eq!(perspective.vfov(),         vfov.into(),  abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(perspective.aspect_ratio(), aspect_ratio, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(perspective.near(),         near,         abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(perspective.far(),          far,          abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    pub fn new<A>(vfov: A, aspect_ratio: S, near: S, far: S) -> Self
    where
        A: Into<Radians<S>>,
    {
        let spec_vfov = vfov.into();

        Self {
            matrix: Matrix4x4::from_perspective_fov(spec_vfov, aspect_ratio, near, far),
        }
    }

    /// Get the vertical field of view angle.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let expected = vfov.into();
    /// let result = perspective.vfov();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn vfov(&self) -> Radians<S> {
        // The perspective projection field of view matrix has the form
        // ```text
        // | m[0, 0]  0         0        0       |
        // | 0        m[1, 1]   0        0       |
        // | 0        0         m[2, 2]  m[3, 2] |
        // | 0        0        -1        0       |
        // ```
        // where
        // ```text
        // m[0, 0] := 1 / (aspect * tan(vfov / 2))
        // m[1, 1] := 1 / (tan(vfov / 2))
        // m[2, 2] := -(far + near) / (far - near)
        // m[3, 2] := - 2 * far * near / (far - near)
        // ```
        // We can reconstruct the vertical field of view component from the
        // `m[1, 1]` component follows.
        // ```text
        // m[1, 1] := 1 / tan(vfov / 2)
        //     <==> tan(vfov / 2) == 1 / m[1, 1]
        //     <==> vfov / 2 == atan(1 / m[1, 1])
        //     <==> vfov == 2 * atan(1 / m[1, 1])
        // ```
        // so that `vfov == 2 * atan(1 / m[1, 1])`.
        let one = S::one();
        let two = one + one;
        let vfov = (one / self.matrix[1][1]).atan() * two;

        Radians(vfov)
    }

    /// Get the position of the near plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **xy-plane** positioned along the **negative z-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let expected = near;
    /// let result = perspective.near();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn near(&self) -> S {
        // The perspective projection field of view matrix has the form
        // ```text
        // | m[0, 0]  0         0        0       |
        // | 0        m[1, 1]   0        0       |
        // | 0        0         m[2, 2]  m[3, 2] |
        // | 0        0        -1        0       |
        // ```
        // where
        // ```text
        // m[0, 0] := 1 / (aspect * tan(vfov / 2))
        // m[1, 1] := 1 / (tan(vfov / 2))
        // m[2, 2] := -(far + near) / (far - near)
        // m[3, 2] := - 2 * far * near / (far - near)
        // ```
        // We can reconstruct the `near` parameter from the `m[2, 2]` and `m[3, 2]`
        // components as follows. Define the `ratio` parameter
        // ```text
        // numerator := -m[2, 2] + 1
        //           == ((far + near) / (far - near)) + 1
        //           == (far + near) / (far - near) + (far - near) / (far - near)
        //           == ((far + near) + (far - near)) / (far - near)
        //           == 2 * far / (far - near)
        // denominator := -m[2, 2] - 1
        //             == ((far + near) / (far - near)) - 1
        //             == ((far + near) / (far - near)) - ((far - near) / (far - near))
        //             == ((far + near) - (far - near)) / (far - near)
        //             == 2 * near / (far - near)
        // ratio := numerator / denominator
        //       == ((2 * far) / (far - near)) / ((2 * near) / (far - near))
        //       == (2 * far) / (2 * near)
        //       == far / near
        // ```
        // hence the name `ratio`. This uses the `m[2, 2]` component. To derive
        // `near` we now use the `m[3, 2]` component. Observe that
        // `far == ratio * near`. We then have
        // ```text
        // m[3, 2] := (-2 * far * near) / (far - near)
        //         == (-2 * ratio * near * near) / (ratio * near - near)
        //         == (-(2 * ratio) * near * near) / ((ratio - 1) * near)
        //         == (-2 * ratio * near) / (ratio - 1)
        //         == (2 * ratio * near) / (1 - ratio)
        //         == ((2 * ratio) / (1 - ratio)) * near
        // ```
        // From this we derive the formula for `near`
        // ```text
        // near == ((1 - ratio) / (2 * ratio)) * m[3, 2]
        // ```
        // which is in deed the formula we use.
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[2][2] + one) / (-self.matrix[2][2] - one);

        ((one - ratio) / (two * ratio)) * self.matrix[3][2]
    }

    /// Get the position of the far plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **xy-plane** positioned along the **negative z-axis**.
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let expected = far;
    /// let result = perspective.far();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn far(&self) -> S {
        // The perspective projection field of view matrix has the form
        // ```text
        // | m[0, 0]  0         0        0       |
        // | 0        m[1, 1]   0        0       |
        // | 0        0         m[2, 2]  m[3, 2] |
        // | 0        0        -1        0       |
        // ```
        // where
        // ```text
        // m[0, 0] := 1 / (aspect * tan(vfov / 2))
        // m[1, 1] := 1 / (tan(vfov / 2))
        // m[2, 2] := -(far + near) / (far - near)
        // m[3, 2] := - 2 * far * near / (far - near)
        // ```
        // We can reconstruct the `far` parameter from the `m[2, 2]` and `m[3, 2]`
        // components as follows. Define the `ratio` parameter
        // ```text
        // numerator := -m[2, 2] + 1
        //           == ((far + near) / (far - near)) + 1
        //           == (far + near) / (far - near) + (far - near) / (far - near)
        //           == ((far + near) + (far - near)) / (far - near)
        //           == 2 * far / (far - near)
        // denominator := -m[2, 2] - 1
        //             == ((far + near) / (far - near)) - 1
        //             == ((far + near) / (far - near)) - ((far - near) / (far - near))
        //             == ((far + near) - (far - near)) / (far - near)
        //             == 2 * near / (far - near)
        // ratio := numerator / denominator
        //       == ((2 * far) / (far - near)) / ((2 * near) / (far - near))
        //       == (2 * far) / (2 * near)
        //       == far / near
        // ```
        // hence the name `ratio`. This uses the `m[2, 2]` component. To derive
        // `far` we now use the `m[3, 2]` component. Observe that
        // `near == far / ratio`. We then have
        // ```text
        // m[3, 2] := (-2 * far * near) / (far - near)
        //         == (-2 * far * (far / ratio)) / (far - (far / ratio))
        //         == ((-2 * (1 / ratio)) * far * far) / ((1 - (1 / ratio)) * far)
        //         == ((-2 * (1 / ratio)) * far) / (1 - (1 / ratio))
        //         == (-2 * (1 / ratio) * far) / ((1 / ratio) * (ratio - 1))
        //         == (-2 / (ratio - 1)) * far
        //         == (2 / (1 - ratio)) * far
        // ```
        // From this we derive the formula for `far`
        // ```text
        // far == ((1 - ratio) / 2) * m[3, 2]
        // ```
        // which is in deed the formula we use.
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[2][2] + one) / (-self.matrix[2][2] - one);

        ((one - ratio) / two) * self.matrix[3][2]
    }

    /// Get the aspect ratio. The aspect ratio is the ratio of the
    /// width of the viewing plane of the view volume to the height of the
    /// viewing plane of the view volume.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let expected = aspect_ratio;
    /// let result = perspective.aspect_ratio();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn aspect_ratio(&self) -> S {
        // The perspective projection field of view matrix has the form
        // ```text
        // | m[0, 0]  0         0        0       |
        // | 0        m[1, 1]   0        0       |
        // | 0        0         m[2, 2]  m[3, 2] |
        // | 0        0        -1        0       |
        // ```
        // where
        // ```text
        // m[0, 0] := 1 / (aspect_ratio * tan(vfov / 2))
        // m[1, 1] := 1 / (tan(vfov / 2))
        // m[2, 2] := -(far + near) / (far - near)
        // m[3, 2] := - 2 * far * near / (far - near)
        // ```
        // We can reconstruct the `aspect_ratio` parameter from the `m[0, 0]` and `m[1, 1]`
        // components as follows. Observe that
        // ```text
        // m[1, 1] / m[0, 0] == (1 / tan(vfov / 2)) / (1 / (aspect_ratio * (1 / tan(vfov / 2))))
        //                   == aspect_ratio * ((1 / tan(vfov / 2)) / (1 / tan(vfov / 2))
        //                   == aspect_ratio
        // ```
        // which is the desired formula.
        self.matrix[1][1] / self.matrix[0][0]
    }

    /// Get the position of the right plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **yz-plane** positioned along the **positive x-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::Degrees;
    /// # use core::f64;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let expected = (1_f64 / 10_f64) * (4_f64 / 3_f64) * f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64));
    /// let result = perspective.right();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn right(&self) -> S {
        let abs_near = self.near().abs();

        abs_near / self.matrix[0][0]
    }

    /// Get the position of the left plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **yz-plane** positioned along the **positive x-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::Degrees;
    /// # use core::f64;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let expected = -(1_f64 / 10_f64) * (4_f64 / 3_f64) * f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64));
    /// let result = perspective.left();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn left(&self) -> S {
        -self.right()
    }

    /// Get the position of the top plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **zx-plane** positioned along the **positive y-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::Degrees;
    /// # use core::f64;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let expected = (1_f64 / 10_f64) * (f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64)));
    /// let result = perspective.top();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn top(&self) -> S {
        let abs_near = self.near().abs();

        abs_near / self.matrix[1][1]
    }

    /// Get the position of the bottom plane of the viewing
    /// frustum descibed by the perspective projection of the plane
    /// parallel to the **zx-plane** positioned along the **positive y-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::Degrees;
    /// # use core::f64;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let expected = -(1_f64 / 10_f64) * (f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64)));
    /// let result = perspective.bottom();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn bottom(&self) -> S {
        -self.top()
    }

    /// Get the matrix that implements the perspective projection transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let tan_half_vfov = (vfov / 2_f64).tan();
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let c0r0 = 1_f64 / (aspect_ratio * tan_half_vfov);
    /// let c1r1 = 1_f64 / (tan_half_vfov);
    /// let c2r2 = -(far + near) / (far - near);
    /// let c3r2 = (-2_f64 * far * near) / (far - near);
    /// let expected = Matrix4x4::new(
    ///     c0r0,  0_f64, 0_f64,  0_f64,
    ///     0_f64, c1r1,  0_f64,  0_f64,
    ///     0_f64, 0_f64, c2r2,  -1_f64,
    ///     0_f64, 0_f64, c3r2,   0_f64,
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
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Point3,
    /// # };
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let tan_half_vfov = (vfov / 2_f64).tan();
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let point = Point3::new(-1_f64, -1_f64, 30_f64);
    /// let expected = Point3::new(3_f64 / 120_f64, 1_f64 / 30_f64, 3230_f64 / 2970_f64);
    /// let result = perspective.project_point(&point);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    pub fn project_point(&self, point: &Point3<S>) -> Point3<S> {
        let inverse_w = -S::one() / point.z;

        Point3::new(
            (self.matrix.c0r0 * point.x) * inverse_w,
            (self.matrix.c1r1 * point.y) * inverse_w,
            (self.matrix.c2r2 * point.z + self.matrix.c3r2) * inverse_w,
        )
    }

    /// Apply the perspective projection transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let tan_half_vfov = (vfov / 2_f64).tan();
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let vector = Vector3::new(-1_f64, -1_f64, 30_f64);
    /// let expected = Vector3::new(3_f64 / 120_f64, 1_f64 / 30_f64, 3230_f64 / 2970_f64);
    /// let result = perspective.project_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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
    /// This is the inverse operation of [`project_point`].
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Point3,
    /// # };
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let tan_half_vfov = (vfov / 2_f64).tan();
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let point = Point3::new(-1_f64, -1_f64, 30_f64);
    /// let projected_point = perspective.project_point(&point);
    /// let expected = point;
    /// let result = perspective.unproject_point(&projected_point);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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
        let near = self.near();
        let far = self.far();
        let tan_vfov_div_2 = Radians::tan(self.vfov() / two);
        let top = self.near() * tan_vfov_div_2;
        let bottom = -top;
        let right = self.aspect_ratio() * top;
        let left = -right;

        let c0r0 = (right - left) / (two * near);
        let c1r1 = (top - bottom) / (two * near);
        let c2r3 = (near - far) / (two * far * near);
        let c3r0 = (left + right) / (two * near);
        let c3r1 = (bottom + top) / (two * near);
        let c3r2 = -one;
        let c3r3 = (far + near) / (two * far * near);
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
    /// This is the inverse operation of [`project_vector`].
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let tan_half_vfov = (vfov / 2_f64).tan();
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let vector = Vector3::new(-1_f64, -1_f64, 30_f64);
    /// let projected_vector = perspective.project_vector(&vector);
    /// let expected = vector;
    /// let result = perspective.unproject_vector(&projected_vector);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
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
        let near = self.near();
        let far = self.far();
        let tan_vfov_div_2 = Radians::tan(self.vfov() / two);
        let top = self.near() * tan_vfov_div_2;
        let bottom = -top;
        let right = self.aspect_ratio() * top;
        let left = -right;

        let c0r0 = (right - left) / (two * near);
        let c1r1 = (top - bottom) / (two * near);
        let c2r3 = (near - far) / (two * far * near);
        let c3r0 = (left + right) / (two * near);
        let c3r1 = (bottom + top) / (two * near);
        let c3r2 = -one;
        let c3r3 = (far + near) / (two * far * near);
        let w = c2r3 * vector.z + c3r3;
        let inverse_w = one / w;

        Vector3::new(
            (c0r0 * vector.x + c3r0) * inverse_w,
            (c1r1 * vector.y + c3r1) * inverse_w,
            c3r2 * inverse_w,
        )
    }

    /// Convert a perspective projection to a projective matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::PerspectiveFov3;
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let tan_half_vfov = (vfov / 2_f64).tan();
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let c0r0 = 1_f64 / (aspect_ratio * tan_half_vfov);
    /// let c1r1 = 1_f64 / (tan_half_vfov);
    /// let c2r2 = -(far + near) / (far - near);
    /// let c3r2 = (-2_f64 * far * near) / (far - near);
    /// let expected = Matrix4x4::new(
    ///     c0r0,  0_f64, 0_f64,  0_f64,
    ///     0_f64, c1r1,  0_f64,  0_f64,
    ///     0_f64, 0_f64, c2r2,  -1_f64,
    ///     0_f64, 0_f64, c3r2,   0_f64,
    /// );
    /// let result = perspective.to_projective_matrix();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_projective_matrix(&self) -> Matrix4x4<S> {
        self.matrix
    }

    /// Convert a perspective projection to a generic transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::{
    /// #     PerspectiveFov3,
    /// #     Transform3,
    /// # };
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Degrees,
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let tan_half_vfov = (vfov / 2_f64).tan();
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = PerspectiveFov3::new(vfov, aspect_ratio, near, far);
    /// let c0r0 = 1_f64 / (aspect_ratio * tan_half_vfov);
    /// let c1r1 = 1_f64 / (tan_half_vfov);
    /// let c2r2 = -(far + near) / (far - near);
    /// let c3r2 = (-2_f64 * far * near) / (far - near);
    /// let expected = Transform3::from_matrix_unchecked(Matrix4x4::new(
    ///     c0r0,  0_f64, 0_f64,  0_f64,
    ///     0_f64, c1r1,  0_f64,  0_f64,
    ///     0_f64, 0_f64, c2r2,  -1_f64,
    ///     0_f64, 0_f64, c3r2,   0_f64,
    /// ));
    /// let result = perspective.to_transform();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_transform(&self) -> Transform3<S> {
        Transform3::from_matrix_unchecked(self.matrix)
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
    S: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "PerspectiveFov3 [{}]", self.matrix)
    }
}

impl<S> From<PerspectiveFov3<S>> for Matrix4x4<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(perspective: PerspectiveFov3<S>) -> Matrix4x4<S> {
        perspective.to_projective_matrix()
    }
}

impl<S> ops::Mul<Point3<S>> for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.project_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.project_point(other)
    }
}

impl<S> ops::Mul<Vector3<S>> for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: Vector3<S>) -> Self::Output {
        self.project_vector(&other)
    }
}

impl<S> ops::Mul<&Vector3<S>> for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: &Vector3<S>) -> Self::Output {
        self.project_vector(other)
    }
}

impl<S> ops::Mul<Vector3<S>> for &PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: Vector3<S>) -> Self::Output {
        self.project_vector(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Vector3<S>> for &'b PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: &'a Vector3<S>) -> Self::Output {
        self.project_vector(other)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveFov3Tol<S> {
    matrix: Matrix4x4<S>,
}

impl<S> From<Matrix4x4<S>> for PerspectiveFov3Tol<S> {
    #[inline]
    fn from(matrix: Matrix4x4<S>) -> Self {
        Self { matrix }
    }
}

impl<S> From<&Matrix4x4<S>> for PerspectiveFov3Tol<S>
where
    S: Copy,
{
    #[inline]
    fn from(matrix: &Matrix4x4<S>) -> Self {
        Self { matrix: *matrix }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerspectiveFov3Diff<S> {
    matrix: Matrix4x4<S>,
}

impl<S> PerspectiveFov3Diff<S> {
    #[inline]
    const fn from(matrix: Matrix4x4<S>) -> Self {
        Self { matrix }
    }
}

impl<S> approx_cmp::AbsDiffEq for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type Tolerance = PerspectiveFov3Tol<<S as approx_cmp::AbsDiffEq>::Tolerance>;

    #[inline]
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> bool {
        approx_cmp::AbsDiffEq::abs_diff_eq(&self.matrix, &other.matrix, &max_abs_diff.matrix)
    }
}

impl<S> approx_cmp::AbsDiffAllEq for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::AbsDiffAllEq>::AllTolerance;

    #[inline]
    fn abs_diff_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> bool {
        approx_cmp::AbsDiffAllEq::abs_diff_all_eq(&self.matrix, &other.matrix, max_abs_diff)
    }
}

impl<S> approx_cmp::AssertAbsDiffEq for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = PerspectiveFov3Diff<<S as approx_cmp::AssertAbsDiffEq>::DebugAbsDiff>;
    type DebugTolerance = PerspectiveFov3Tol<<S as approx_cmp::AssertAbsDiffEq>::DebugTolerance>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let matrix = approx_cmp::AssertAbsDiffEq::debug_abs_diff(&self.matrix, &other.matrix);

        PerspectiveFov3Diff::from(matrix)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let matrix = approx_cmp::AssertAbsDiffEq::debug_abs_diff_tolerance(&self.matrix, &other.matrix, &max_abs_diff.matrix);

        PerspectiveFov3Tol::from(matrix)
    }
}

impl<S> approx_cmp::AssertAbsDiffAllEq for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = PerspectiveFov3Tol<<S as approx_cmp::AssertAbsDiffAllEq>::AllDebugTolerance>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let matrix = approx_cmp::AssertAbsDiffAllEq::debug_abs_diff_all_tolerance(&self.matrix, &other.matrix, max_abs_diff);

        PerspectiveFov3Tol::from(matrix)
    }
}

impl<S> approx_cmp::RelativeEq for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type Tolerance = PerspectiveFov3Tol<<S as approx_cmp::RelativeEq>::Tolerance>;

    #[inline]
    fn relative_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_relative: &Self::Tolerance) -> bool {
        approx_cmp::RelativeEq::relative_eq(&self.matrix, &other.matrix, &max_abs_diff.matrix, &max_relative.matrix)
    }
}

impl<S> approx_cmp::RelativeAllEq for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::RelativeAllEq>::AllTolerance;

    #[inline]
    fn relative_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_relative: &Self::AllTolerance) -> bool {
        approx_cmp::RelativeAllEq::relative_all_eq(&self.matrix, &other.matrix, max_abs_diff, max_relative)
    }
}

impl<S> approx_cmp::AssertRelativeEq for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = PerspectiveFov3Diff<<S as approx_cmp::AssertRelativeEq>::DebugAbsDiff>;
    type DebugTolerance = PerspectiveFov3Tol<<S as approx_cmp::AssertRelativeEq>::DebugTolerance>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let matrix = approx_cmp::AssertRelativeEq::debug_abs_diff(&self.matrix, &other.matrix);

        PerspectiveFov3Diff::from(matrix)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let matrix = approx_cmp::AssertRelativeEq::debug_abs_diff_tolerance(&self.matrix, &other.matrix, &max_abs_diff.matrix);

        PerspectiveFov3Tol::from(matrix)
    }

    #[inline]
    fn debug_relative_tolerance(&self, other: &Self, max_relative: &Self::Tolerance) -> Self::DebugTolerance {
        let matrix = approx_cmp::AssertRelativeEq::debug_relative_tolerance(&self.matrix, &other.matrix, &max_relative.matrix);

        PerspectiveFov3Tol::from(matrix)
    }
}

impl<S> approx_cmp::AssertRelativeAllEq for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = PerspectiveFov3Tol<<S as approx_cmp::AssertRelativeAllEq>::AllDebugTolerance>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let matrix = approx_cmp::AssertRelativeAllEq::debug_abs_diff_all_tolerance(&self.matrix, &other.matrix, max_abs_diff);

        PerspectiveFov3Tol::from(matrix)
    }

    #[inline]
    fn debug_relative_all_tolerance(&self, other: &Self, max_relative: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let matrix = approx_cmp::AssertRelativeAllEq::debug_relative_all_tolerance(&self.matrix, &other.matrix, max_relative);

        PerspectiveFov3Tol::from(matrix)
    }
}

impl<S> approx_cmp::UlpsEq for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type Tolerance = PerspectiveFov3Tol<<S as approx_cmp::UlpsEq>::Tolerance>;
    type UlpsTolerance = PerspectiveFov3Tol<<S as approx_cmp::UlpsEq>::UlpsTolerance>;

    fn ulps_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_ulps: &Self::UlpsTolerance) -> bool {
        approx_cmp::UlpsEq::ulps_eq(&self.matrix, &other.matrix, &max_abs_diff.matrix, &max_ulps.matrix)
    }
}

impl<S> approx_cmp::UlpsAllEq for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::UlpsAllEq>::AllTolerance;
    type AllUlpsTolerance = <S as approx_cmp::UlpsAllEq>::AllUlpsTolerance;

    #[inline]
    fn ulps_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_ulps: &Self::AllUlpsTolerance) -> bool {
        approx_cmp::UlpsAllEq::ulps_all_eq(&self.matrix, &other.matrix, max_abs_diff, max_ulps)
    }
}

impl<S> approx_cmp::AssertUlpsEq for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = PerspectiveFov3Diff<<S as approx_cmp::AssertUlpsEq>::DebugAbsDiff>;
    type DebugUlpsDiff = PerspectiveFov3Diff<<S as approx_cmp::AssertUlpsEq>::DebugUlpsDiff>;
    type DebugTolerance = PerspectiveFov3Tol<<S as approx_cmp::AssertUlpsEq>::DebugTolerance>;
    type DebugUlpsTolerance = PerspectiveFov3Tol<<S as approx_cmp::AssertUlpsEq>::DebugUlpsTolerance>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let matrix = approx_cmp::AssertUlpsEq::debug_abs_diff(&self.matrix, &other.matrix);

        PerspectiveFov3Diff::from(matrix)
    }

    #[inline]
    fn debug_ulps_diff(&self, other: &Self) -> Self::DebugUlpsDiff {
        let matrix = approx_cmp::AssertUlpsEq::debug_ulps_diff(&self.matrix, &other.matrix);

        PerspectiveFov3Diff::from(matrix)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let matrix = approx_cmp::AssertUlpsEq::debug_abs_diff_tolerance(&self.matrix, &other.matrix, &max_abs_diff.matrix);

        PerspectiveFov3Tol::from(matrix)
    }

    #[inline]
    fn debug_ulps_tolerance(&self, other: &Self, max_ulps: &Self::UlpsTolerance) -> Self::DebugUlpsTolerance {
        let matrix = approx_cmp::AssertUlpsEq::debug_ulps_tolerance(&self.matrix, &other.matrix, &max_ulps.matrix);

        PerspectiveFov3Tol::from(matrix)
    }
}

impl<S> approx_cmp::AssertUlpsAllEq for PerspectiveFov3<S>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = PerspectiveFov3Tol<<S as approx_cmp::AssertUlpsAllEq>::AllDebugTolerance>;
    type AllDebugUlpsTolerance = PerspectiveFov3Tol<<S as approx_cmp::AssertUlpsAllEq>::AllDebugUlpsTolerance>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let matrix = approx_cmp::AssertUlpsAllEq::debug_abs_diff_all_tolerance(&self.matrix, &other.matrix, max_abs_diff);

        PerspectiveFov3Tol::from(matrix)
    }

    #[inline]
    fn debug_ulps_all_tolerance(&self, other: &Self, max_ulps: &Self::AllUlpsTolerance) -> Self::AllDebugUlpsTolerance {
        let matrix = approx_cmp::AssertUlpsAllEq::debug_ulps_all_tolerance(&self.matrix, &other.matrix, max_ulps);

        PerspectiveFov3Tol::from(matrix)
    }
}



/// An orthographic projection with arbitrary `left`, `right`,
/// `top`, `bottom`, `near`, and `far` planes.
///
/// The `near` and `far` parameters are the absolute values of the positions
/// of the **near plane** and the **far** plane, respectively, along the
/// **negative z-axis**. In particular, the position of the **near plane** is
/// `z == -near` and the position of the **far plane** is `z == -far`.
///
/// This data type represents a homogeneous matrix representing an orthographic
/// projection transformation with a right-handed coordinate system where the
/// orthographic camera faces the **negative z-axis** with the **positive x-axis**
/// going to the right, and the **positive y-axis** going up. The orthographic view
/// volume is the box `[left, right] x [bottom, top] x [-near, -far]`. The
/// normalized device coordinates this transformation maps to are
/// `[-1, 1] x [-1, 1] x [-1, 1]`.
///
/// The underlying matrix is identical to the one used by OpenGL, provided here
/// for reference
/// ```text
/// | m[0, 0]  0        0        m[3, 0] |
/// | 0        m[1, 1]  0        m[3, 1] |
/// | 0        0        m[2, 2]  m[3, 2] |
/// | 0        0        0        1       |
/// where
/// m[0, 0] == 2 / (r - l)
/// m[3, 0] == -(r + l) / (r - l)
/// m[1, 1] == 2 / (t - b)
/// m[3, 1] == -(t + b) / (t - b)
/// m[2, 2] == -2 / (f - n)
/// m[3, 2] == -(f + n) / (f - n)
/// ```
/// where the matrix entries are indexed in column-major order.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Orthographic3<S> {
    matrix: Matrix4x4<S>,
}

impl<S> Orthographic3<S>
where
    S: SimdScalarFloat,
{
    /// Construct a new orthographic projection.
    ///
    /// # Parameters
    ///
    /// The parameters must satisfy
    /// ```text
    /// left < right
    /// bottom < top
    /// 0 < near < far
    /// ```
    /// to construct a useful orthographic projection. In particular, `near` and
    /// `far` are the respective absolute values of the placement of the near and
    /// far planes along the **z-axis**.
    ///
    /// `left` is the horizontal position of the left plane in camera space.
    /// The left plane is a plane parallel to the **yz-plane** along the **x-axis**.
    ///
    /// `right` is the horizontal position of the right plane in camera space.
    /// The right plane is a plane parallel to the **yz-plane** along the **x-axis**.
    ///
    /// `bottom` is the vertical position of the bottom plane in camera space.
    /// The bottom plane is a plane parallel to the **zx-plane** along the **y-axis**.
    ///
    /// `top` is the vertical position of the top plane in camera space.
    /// the top plane is a plane parallel to the **zx-plane** along the **y-axis**.
    ///
    /// `near` is the distance along the **negative z-axis** of the near plane from the
    /// eye in camera space. The near plane is a plane parallel to the **xy-plane** along
    /// the **negative z-axis**.
    ///
    /// `far` the distance along the **negative z-axis** of the far plane from the
    /// eye in camera space. The far plane is a plane parallel to the **xy-plane** along
    /// the **negative z-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Orthographic3;
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let vector = Vector3::new(2_f64, 3_f64, 30_f64);
    ///
    /// assert_relative_eq!(orthographic.left(),   left,   abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(orthographic.right(),  right,  abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(orthographic.bottom(), bottom, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(orthographic.top(),    top,    abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(orthographic.near(),   near,   abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(orthographic.far(),    far,    abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    pub fn new(left: S, right: S, bottom: S, top: S, near: S, far: S) -> Self {
        Self {
            matrix: Matrix4x4::from_orthographic(left, right, bottom, top, near, far),
        }
    }

    /// Get the position of the near plane of the viewing
    /// volume described by the orthographic projection of the plane
    /// parallel to the **xy-plane** positioned along the **negative z-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Orthographic3;
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let vector = Vector3::new(2_f64, 3_f64, 30_f64);
    /// let expected = near;
    /// let result = orthographic.near();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn near(&self) -> S {
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[3][2] + one) / (-self.matrix[3][2] - one);

        (-two / (ratio - one)) * (one / self.matrix[2][2])
    }

    /// Get the position of the far plane of the viewing
    /// volume described by the orthographic projection of the plane
    /// parallel to the **xy-plane** positioned along the **negative z-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Orthographic3;
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let vector = Vector3::new(2_f64, 3_f64, 30_f64);
    /// let expected = far;
    /// let result = orthographic.far();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn far(&self) -> S {
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[3][2] + one) / (-self.matrix[3][2] - one);

        ((-two * ratio) / (ratio - one)) * (one / self.matrix[2][2])
    }

    /// Get the position of the right plane of the viewing
    /// volume described by the orthographic projection of the plane
    /// parallel to the **yz-plane** positioned along the **positive x-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Orthographic3;
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let vector = Vector3::new(2_f64, 3_f64, 30_f64);
    /// let expected = right;
    /// let result = orthographic.right();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn right(&self) -> S {
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[3][0] + one) / (-self.matrix[3][0] - one);

        ((two * ratio) / (ratio - one)) * (one / self.matrix[0][0])
    }

    /// Get the position of the left plane of the viewing
    /// volume described by the orthographic projection of the plane
    /// parallel to the **yz-plane** positioned along the **positive x-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Orthographic3;
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let vector = Vector3::new(2_f64, 3_f64, 30_f64);
    /// let expected = left;
    /// let result = orthographic.left();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn left(&self) -> S {
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[3][0] + one) / (-self.matrix[3][0] - one);

        (two / (ratio - one)) * (one / self.matrix[0][0])
    }

    /// Get the position of the top plane of the viewing
    /// volume described by the orthographic projection of the plane
    /// parallel to the **zx-plane** positioned along the **positive y-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Orthographic3;
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let vector = Vector3::new(2_f64, 3_f64, 30_f64);
    /// let expected = top;
    /// let result = orthographic.top();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn top(&self) -> S {
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[3][1] + one) / (-self.matrix[3][1] - one);

        ((two * ratio) / (ratio - one)) * (one / self.matrix[1][1])
    }

    /// Get the position of the bottom plane of the viewing
    /// volume descibed by the orthographic projection of the plane
    /// parallel to the **zx-plane** positioned along the **positive y-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Orthographic3;
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let vector = Vector3::new(2_f64, 3_f64, 30_f64);
    /// let expected = bottom;
    /// let result = orthographic.bottom();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn bottom(&self) -> S {
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[3][1] + one) / (-self.matrix[3][1] - one);

        (two / (ratio - one)) * (one / self.matrix[1][1])
    }

    /// Get the underlying matrix implementing the orthographic transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Orthographic3;
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
    ///     0_f64,         0_f64,         -51_f64 / 50_f64, 1_f64,
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
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Orthographic3;
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
            self.matrix.c2r2 * point.z + self.matrix.c3r2,
        )
    }

    /// Apply the orthographic projection transformation to a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Orthographic3;
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
            self.matrix.c2r2 * vector.z,
        )
    }

    /// Unproject a point from normalized devices coordinates back to camera
    /// view space.
    ///
    /// This is the inverse operation of [`project_point`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Orthographic3;
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let point = Point3::new(2_f64, 3_f64, 30_f64);
    /// let projected_point = orthographic.project_point(&point);
    /// let expected = point;
    /// let result = orthographic.unproject_point(&projected_point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn unproject_point(&self, point: &Point3<S>) -> Point3<S> {
        let one = S::one();
        let one_half = one / (one + one);
        let c0r0 = one_half * (self.right() - self.left());
        let c1r1 = one_half * (self.top() - self.bottom());
        let c2r2 = -one_half * (self.far() - self.near());
        let c3r0 = one_half * (self.left() + self.right());
        let c3r1 = one_half * (self.bottom() + self.top());
        let c3r2 = -one_half * (self.far() + self.near());

        Point3::new(c0r0 * point.x + c3r0, c1r1 * point.y + c3r1, c2r2 * point.z + c3r2)
    }

    /// Unproject a vector from normalized device coordinates back to
    /// camera view space.
    ///
    /// This is the inverse operation of [`project_vector`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Orthographic3;
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let vector = Vector3::new(2_f64, 3_f64, 30_f64);
    /// let projected_vector = orthographic.project_vector(&vector);
    /// let expected = vector;
    /// let result = orthographic.unproject_vector(&projected_vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn unproject_vector(&self, vector: &Vector3<S>) -> Vector3<S> {
        let one = S::one();
        let one_half = one / (one + one);
        let c0r0 = one_half * (self.right() - self.left());
        let c1r1 = one_half * (self.top() - self.bottom());
        let c2r2 = -one_half * (self.far() - self.near());

        Vector3::new(c0r0 * vector.x, c1r1 * vector.y, c2r2 * vector.z)
    }

    /// Convert an orthographic projection to a projective matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Orthographic3;
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
    ///     0_f64,         0_f64,         -51_f64 / 50_f64, 1_f64,
    /// );
    /// let result = orthographic.to_projective_matrix();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_projective_matrix(&self) -> Matrix4x4<S> {
        self.matrix
    }

    /// Convert a orthographic projection to a generic transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::{
    /// #     Orthographic3,
    /// #     Transform3,
    /// # };
    /// #
    /// let left = -6_f64;
    /// let right = 6_f64;
    /// let bottom = -4_f64;
    /// let top = 4_f64;
    /// let near = 1_f64;
    /// let far = 101_f64;
    /// let orthographic = Orthographic3::new(left, right, bottom, top, near, far);
    /// let expected = Transform3::from_matrix_unchecked(Matrix4x4::new(
    ///     1_f64 / 6_f64, 0_f64,          0_f64,           0_f64,
    ///     0_f64,         1_f64 / 4_f64,  0_f64,           0_f64,
    ///     0_f64,         0_f64,         -1_f64 / 50_f64,  0_f64,
    ///     0_f64,         0_f64,         -51_f64 / 50_f64, 1_f64,
    /// ));
    /// let result = orthographic.to_transform();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_transform(&self) -> Transform3<S> {
        Transform3::from_matrix_unchecked(self.matrix)
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
    S: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Orthographic3 [{}]", self.matrix)
    }
}

impl<S> From<Orthographic3<S>> for Matrix4x4<S>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn from(orthographic: Orthographic3<S>) -> Matrix4x4<S> {
        orthographic.to_projective_matrix()
    }
}

impl<S> ops::Mul<Point3<S>> for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<S> ops::Mul<&Point3<S>> for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &Point3<S>) -> Self::Output {
        self.project_point(other)
    }
}

impl<S> ops::Mul<Point3<S>> for &Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: Point3<S>) -> Self::Output {
        self.project_point(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Point3<S>> for &'b Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type Output = Point3<S>;

    #[inline]
    fn mul(self, other: &'a Point3<S>) -> Self::Output {
        self.project_point(other)
    }
}

impl<S> ops::Mul<Vector3<S>> for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: Vector3<S>) -> Self::Output {
        self.project_vector(&other)
    }
}

impl<S> ops::Mul<&Vector3<S>> for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: &Vector3<S>) -> Self::Output {
        self.project_vector(other)
    }
}

impl<S> ops::Mul<Vector3<S>> for &Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: Vector3<S>) -> Self::Output {
        self.project_vector(&other)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Vector3<S>> for &'b Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: &'a Vector3<S>) -> Self::Output {
        self.project_vector(other)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Orthographic3Tol<S> {
    matrix: Matrix4x4<S>,
}

impl<S> From<Matrix4x4<S>> for Orthographic3Tol<S> {
    #[inline]
    fn from(matrix: Matrix4x4<S>) -> Self {
        Self { matrix }
    }
}

impl<S> From<&Matrix4x4<S>> for Orthographic3Tol<S>
where
    S: Copy,
{
    #[inline]
    fn from(matrix: &Matrix4x4<S>) -> Self {
        Self { matrix: *matrix }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Orthographic3Diff<S> {
    matrix: Matrix4x4<S>,
}

impl<S> Orthographic3Diff<S> {
    #[inline]
    const fn from(matrix: Matrix4x4<S>) -> Self {
        Self { matrix }
    }
}

impl<S> approx_cmp::AbsDiffEq for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type Tolerance = Orthographic3Tol<<S as approx_cmp::AbsDiffEq>::Tolerance>;

    #[inline]
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> bool {
        approx_cmp::AbsDiffEq::abs_diff_eq(&self.matrix, &other.matrix, &max_abs_diff.matrix)
    }
}

impl<S> approx_cmp::AbsDiffAllEq for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::AbsDiffAllEq>::AllTolerance;

    #[inline]
    fn abs_diff_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> bool {
        approx_cmp::AbsDiffAllEq::abs_diff_all_eq(&self.matrix, &other.matrix, max_abs_diff)
    }
}

impl<S> approx_cmp::AssertAbsDiffEq for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = Orthographic3Diff<<S as approx_cmp::AssertAbsDiffEq>::DebugAbsDiff>;
    type DebugTolerance = Orthographic3Tol<<S as approx_cmp::AssertAbsDiffEq>::DebugTolerance>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let matrix = approx_cmp::AssertAbsDiffEq::debug_abs_diff(&self.matrix, &other.matrix);

        Orthographic3Diff::from(matrix)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let matrix = approx_cmp::AssertAbsDiffEq::debug_abs_diff_tolerance(&self.matrix, &other.matrix, &max_abs_diff.matrix);

        Orthographic3Tol::from(matrix)
    }
}

impl<S> approx_cmp::AssertAbsDiffAllEq for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = Orthographic3Tol<<S as approx_cmp::AssertAbsDiffAllEq>::AllDebugTolerance>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let matrix = approx_cmp::AssertAbsDiffAllEq::debug_abs_diff_all_tolerance(&self.matrix, &other.matrix, max_abs_diff);

        Orthographic3Tol::from(matrix)
    }
}

impl<S> approx_cmp::RelativeEq for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type Tolerance = Orthographic3Tol<<S as approx_cmp::RelativeEq>::Tolerance>;

    #[inline]
    fn relative_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_relative: &Self::Tolerance) -> bool {
        approx_cmp::RelativeEq::relative_eq(&self.matrix, &other.matrix, &max_abs_diff.matrix, &max_relative.matrix)
    }
}

impl<S> approx_cmp::RelativeAllEq for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::RelativeAllEq>::AllTolerance;

    #[inline]
    fn relative_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_relative: &Self::AllTolerance) -> bool {
        approx_cmp::RelativeAllEq::relative_all_eq(&self.matrix, &other.matrix, max_abs_diff, max_relative)
    }
}

impl<S> approx_cmp::AssertRelativeEq for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = Orthographic3Diff<<S as approx_cmp::AssertRelativeEq>::DebugAbsDiff>;
    type DebugTolerance = Orthographic3Tol<<S as approx_cmp::AssertRelativeEq>::DebugTolerance>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let matrix = approx_cmp::AssertRelativeEq::debug_abs_diff(&self.matrix, &other.matrix);

        Orthographic3Diff::from(matrix)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let matrix = approx_cmp::AssertRelativeEq::debug_abs_diff_tolerance(&self.matrix, &other.matrix, &max_abs_diff.matrix);

        Orthographic3Tol::from(matrix)
    }

    #[inline]
    fn debug_relative_tolerance(&self, other: &Self, max_relative: &Self::Tolerance) -> Self::DebugTolerance {
        let matrix = approx_cmp::AssertRelativeEq::debug_relative_tolerance(&self.matrix, &other.matrix, &max_relative.matrix);

        Orthographic3Tol::from(matrix)
    }
}

impl<S> approx_cmp::AssertRelativeAllEq for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = Orthographic3Tol<<S as approx_cmp::AssertRelativeAllEq>::AllDebugTolerance>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let matrix = approx_cmp::AssertRelativeAllEq::debug_abs_diff_all_tolerance(&self.matrix, &other.matrix, max_abs_diff);

        Orthographic3Tol::from(matrix)
    }

    #[inline]
    fn debug_relative_all_tolerance(&self, other: &Self, max_relative: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let matrix = approx_cmp::AssertRelativeAllEq::debug_relative_all_tolerance(&self.matrix, &other.matrix, max_relative);

        Orthographic3Tol::from(matrix)
    }
}

impl<S> approx_cmp::UlpsEq for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type Tolerance = Orthographic3Tol<<S as approx_cmp::UlpsEq>::Tolerance>;
    type UlpsTolerance = Orthographic3Tol<<S as approx_cmp::UlpsEq>::UlpsTolerance>;

    fn ulps_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_ulps: &Self::UlpsTolerance) -> bool {
        approx_cmp::UlpsEq::ulps_eq(&self.matrix, &other.matrix, &max_abs_diff.matrix, &max_ulps.matrix)
    }
}

impl<S> approx_cmp::UlpsAllEq for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::UlpsAllEq>::AllTolerance;
    type AllUlpsTolerance = <S as approx_cmp::UlpsAllEq>::AllUlpsTolerance;

    #[inline]
    fn ulps_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_ulps: &Self::AllUlpsTolerance) -> bool {
        approx_cmp::UlpsAllEq::ulps_all_eq(&self.matrix, &other.matrix, max_abs_diff, max_ulps)
    }
}

impl<S> approx_cmp::AssertUlpsEq for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = Orthographic3Diff<<S as approx_cmp::AssertUlpsEq>::DebugAbsDiff>;
    type DebugUlpsDiff = Orthographic3Diff<<S as approx_cmp::AssertUlpsEq>::DebugUlpsDiff>;
    type DebugTolerance = Orthographic3Tol<<S as approx_cmp::AssertUlpsEq>::DebugTolerance>;
    type DebugUlpsTolerance = Orthographic3Tol<<S as approx_cmp::AssertUlpsEq>::DebugUlpsTolerance>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let matrix = approx_cmp::AssertUlpsEq::debug_abs_diff(&self.matrix, &other.matrix);

        Orthographic3Diff::from(matrix)
    }

    #[inline]
    fn debug_ulps_diff(&self, other: &Self) -> Self::DebugUlpsDiff {
        let matrix = approx_cmp::AssertUlpsEq::debug_ulps_diff(&self.matrix, &other.matrix);

        Orthographic3Diff::from(matrix)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let matrix = approx_cmp::AssertUlpsEq::debug_abs_diff_tolerance(&self.matrix, &other.matrix, &max_abs_diff.matrix);

        Orthographic3Tol::from(matrix)
    }

    #[inline]
    fn debug_ulps_tolerance(&self, other: &Self, max_ulps: &Self::UlpsTolerance) -> Self::DebugUlpsTolerance {
        let matrix = approx_cmp::AssertUlpsEq::debug_ulps_tolerance(&self.matrix, &other.matrix, &max_ulps.matrix);

        Orthographic3Tol::from(matrix)
    }
}

impl<S> approx_cmp::AssertUlpsAllEq for Orthographic3<S>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = Orthographic3Tol<<S as approx_cmp::AssertUlpsAllEq>::AllDebugTolerance>;
    type AllDebugUlpsTolerance = Orthographic3Tol<<S as approx_cmp::AssertUlpsAllEq>::AllDebugUlpsTolerance>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let matrix = approx_cmp::AssertUlpsAllEq::debug_abs_diff_all_tolerance(&self.matrix, &other.matrix, max_abs_diff);

        Orthographic3Tol::from(matrix)
    }

    #[inline]
    fn debug_ulps_all_tolerance(&self, other: &Self, max_ulps: &Self::AllUlpsTolerance) -> Self::AllDebugUlpsTolerance {
        let matrix = approx_cmp::AssertUlpsAllEq::debug_ulps_all_tolerance(&self.matrix, &other.matrix, max_ulps);

        Orthographic3Tol::from(matrix)
    }
}
