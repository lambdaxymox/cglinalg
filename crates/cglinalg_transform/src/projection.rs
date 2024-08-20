use crate::transform::Transform3;
use cglinalg_core::{
    Matrix4x4,
    Point3,
    Vector3,
};
use cglinalg_numeric::SimdScalarFloat;
use cglinalg_trigonometry::Radians;

use core::fmt;
use core::ops;


/// A perspective projection transformation.
///
/// The data type represents a perspective projection transformation that follows
/// OpenGL's mathematical characteristics. We state these precisely below.
///
/// This parametrization is different from the standard frustum specification that
/// defines the location of the frustum planes directly. Instead, this parametrization
/// defines the frustum parameters as displacements along the relevant directions in
/// the view space orthonormal frame. This defines a coordinate-independent frustum
/// specification. The final matrix is the same.
///
/// # Vector Space Details
///
/// The matrix transforms from OpenGL's view space to OpenGL's clip space that maps to
/// OpenGL's canonical view volume after depth normalization.
///
/// ## A Visual Description Of The Vector Spaces.
///
/// The **view space** is a vector space with a right-handed orthonormal frame defined
/// as follows.
///
/// * The origin of the coordinate system is `[0, 0, 0]^T`.
/// * The **positive x-axis** is the horizontal direction and points right.
/// * The **positive y-axis** is the vertical direction and points up.
/// * The **positive z-axis** is the depth direction and points away from the
///   viewing frustum.
/// * The **negative z-axis** is the viewing direction and points into the viewing
/// frustum away from the viewer.
///
/// The **clip space** is a vector space with a left-handed orthonormal frame defined
/// as follows.
///
/// * The origin of the coordinate system is `[0, 0, 0]^T`.
/// * The **positive x-axis** is the horizontal direction and points to the right.
/// * The **positive y-axis** is the vertical direction and points up.
/// * The **positive z-axis** is the depth direction and points into the viewing volume.
/// * The **negative z-axis** points away from the viewing volume towards the viewer.
///
/// The **canonical view volume** is a vector space with a left-handed orthonormal
/// frame identical to the clip space with bounds `[-1, 1] x [-1, 1] x [-1, 1]`.
///
/// ## A Mathematically Precise Description Of The Vector Spaces.
///
/// The **view space** is the vector space `V_v := (R^3, O_v, B_v)` where
/// * The underlying vector space is `R^3`.
/// * The **origin** is `O_v := [0, 0, 0]^T`.
/// * The **basis** is `B_v := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
/// * The orthonormal frame `(O_v, B_v)` has a right-handed orientation.
///
/// The **clip space** is the vector space `V_c := (R^3, O_c, B_c)` where
/// * The underlying vector space is `R^3`.
/// * The **origin** is `O_c := [0, 0, 0]^T`.
/// * The **basis** is `B_c := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
/// * The orthonormal frame `(O_c, B_c)` has a left-handed orientation.
/// * The view volume is parametrized by `[-left, right] x [-bottom, top] x [near, far]`.
///
/// The **canonical view volume** is the vector space `V_cvv := (R^3, O_cvv, B_cvv)` where
/// * The underlying vector space is `R^3`.
/// * The **origin** is `O_cvv := [0, 0, 0]^T`.
/// * The **basis** is `B_cvv := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
/// * The orthonormal frame `(O_cvv, B_cvv)` has a left-handed orientation.
/// * The Canonical View Volume is parametrized by `[-1, 1] x [-1, 1] x [-1, 1]`.
///
/// # Parameter Specification
///
/// The fundamental parametrization of the perspective projection transformation
/// is the specification based on defining the placement of the frustum bounds.
/// We represent the frustum bounds by defining the placements with respect to the
/// **view space** orthonormal frame vectors. More precisely, the fundamental
/// parametrization is given by the parameters `left`, `right`, `bottom`, `top`,
/// `near`, and `far` such that
///
/// ```text
/// left   > 0
/// right  > 0
/// bottom > 0
/// top    > 0
/// far    > near > 0
/// ```
///
/// where the parameters define the placement of the planes. The plane placement
/// definitions follow.
///
/// * `left` defines the location of the **left plane** by its distance along
///   the **negative x-axis** from the origin of the coordinate frame.
///   The **left plane** is a plane parallel to the **yz-plane**.
/// * `right` defines the location of the **right plane** by its distance along
///   the **positive x-axis** from the origin of the coordinate frame.
///   The **right plane** is a plane parallel to the **yz-plane**.
/// * `bottom` defines the location of the **bottom plane** by its distance along
///   the **negative y-axis** from the origin of the coordiante frame.
///   The **bottom plane** is a plane parallel to the **zx-plane**.
/// * `top` defines the location of the **top plane** by its distance along
///   the **positive y-axis** from the origin of the coordinate frame.
///   The **top plane** is a plane parallel to the **zx-plane**.
/// * `near` defines the location of the **near plane** by its distance along
///   the **negative z-axis** from the origin of the coordinate frame.
///   The **near plane** is a plane parallel to the **xy-plane**.
/// * `far` defines the location of the **far plane** by its distance along
///   the **negative z-axis** from the origin of the coordinate frame.
///   The **far plane** is a plane parallel to the **xy-plane**.
///
/// # Matrix Representation Of The Perspective Projection Transformation
///
/// The underlying matrix is a homogeneous projective matrix with the following form
///
/// ```text
/// [ m[0, 0]  0         m[2, 0]  0       ]
/// [ 0        m[1, 1]   m[2, 1]  0       ]
/// [ 0        0         m[2, 2]  m[3, 2] ]
/// [ 0        0        -1        0       ]
/// ```
///
/// where
///
/// ```text
/// m[0, 0] :=  (2 * near) / (right - (-left))        == (2 * near) / (right + left)
/// m[2, 0] :=  (right + (-left)) / (right - (-left)) == (right - left) / (right + left)
/// m[1, 1] :=  (2 * near) / (top - (-bottom))        == (2 * near) / (top + bottom)
/// m[2, 1] :=  (top + (-bottom)) / (top - (-bottom)) == (top - bottom) / (top + bottom)
/// m[2, 2] := -(far + near) / (far - near)
/// m[3, 2] := -(2 * far * near) / (far - near)
/// ```
///
/// where the matrix entries are indexed in column-major order.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Perspective3<S> {
    matrix: Matrix4x4<S>,
}

impl<S> Perspective3<S>
where
    S: SimdScalarFloat,
{
    /// Construct a new possibly asymmetric perspective projection transformation based
    /// on the location of the  **left plane**, **right plane**, **bottom plane**,
    /// **top plane**, **near plane**, and **far plane**.
    ///
    /// The resulting transformation represents a perspective projection transformation that
    /// follows OpenGL's mathematical characteristics. We state these precisely below.
    ///
    /// This parametrization is different from the standard frustum specification that
    /// defines the location of the frustum planes directly. Instead, this parametrization
    /// defines the frustum parameters as displacements along the relevant directions in
    /// the view space orthonormal frame. This defines a coordinate-independent frustum
    /// specification. The final matrix is the same.
    ///
    /// # Vector Space Details
    ///
    /// The matrix transforms from OpenGL's view space to OpenGL's clip space that maps to
    /// OpenGL's canonical view volume after depth normalization.
    ///
    /// ## A Visual Description Of The Vector Spaces.
    ///
    /// The **view space** is a vector space with a right-handed orthonormal frame defined
    /// as follows.
    ///
    /// * The origin of the coordinate system is `[0, 0, 0]^T`.
    /// * The **positive x-axis** is the horizontal direction and points right.
    /// * The **positive y-axis** is the vertical direction and points up.
    /// * The **positive z-axis** is the depth direction and points away from the
    ///   viewing frustum.
    /// * The **negative z-axis** is the viewing direction and points into the viewing
    /// frustum away from the viewer.
    ///
    /// The **clip space** is a vector space with a left-handed orthonormal frame defined
    /// as follows.
    ///
    /// * The origin of the coordinate system is `[0, 0, 0]^T`.
    /// * The **positive x-axis** is the horizontal direction and points to the right.
    /// * The **positive y-axis** is the vertical direction and points up.
    /// * The **positive z-axis** is the depth direction and points into the viewing volume.
    /// * The **negative z-axis** points away from the viewing volume towards the viewer.
    ///
    /// The **canonical view volume** is a vector space with a left-handed orthonormal
    /// frame identical to the clip space with bounds `[-1, 1] x [-1, 1] x [-1, 1]`.
    ///
    /// ## A Mathematically Precise Description Of The Vector Spaces.
    ///
    /// The **view space** is the vector space `V_v := (R^3, O_v, B_v)` where
    /// * The underlying vector space is `R^3`.
    /// * The **origin** is `O_v := [0, 0, 0]^T`.
    /// * The **basis** is `B_v := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
    ///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
    /// * The orthonormal frame `(O_v, B_v)` has a right-handed orientation.
    ///
    /// The **clip space** is the vector space `V_c := (R^3, O_c, B_c)` where
    /// * The underlying vector space is `R^3`.
    /// * The **origin** is `O_c := [0, 0, 0]^T`.
    /// * The **basis** is `B_c := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
    ///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
    /// * The orthonormal frame `(O_c, B_c)` has a left-handed orientation.
    /// * The view volume is parametrized by `[-left, right] x [-bottom, top] x [near, far]`.
    ///
    /// The **canonical view volume** is the vector space `V_cvv := (R^3, O_cvv, B_cvv)` where
    /// * The underlying vector space is `R^3`.
    /// * The **origin** is `O_cvv := [0, 0, 0]^T`.
    /// * The **basis** is `B_cvv := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
    ///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
    /// * The orthonormal frame `(O_cvv, B_cvv)` has a left-handed orientation.
    /// * The Canonical View Volume is parametrized by `[-1, 1] x [-1, 1] x [-1, 1]`.
    ///
    /// # Parameter Specification
    ///
    /// The fundamental parametrization of the perspective projection transformation
    /// is the specification based on defining the placement of the frustum bounds.
    /// We represent the frustum bounds by defining the placements with respect to the
    /// **view space** orthonormal frame vectors. More precisely, the fundamental
    /// parametrization is given by the parameters `left`, `right`, `bottom`, `top`,
    /// `near`, and `far` such that
    ///
    /// ```text
    /// left   > 0
    /// right  > 0
    /// bottom > 0
    /// top    > 0
    /// far    > near > 0
    /// ```
    ///
    /// where the parameters define the placement of the planes. The plane placement
    /// definitions follow.
    ///
    /// * `left` defines the location of the **left plane** by its distance along
    ///   the **negative x-axis** from the origin of the coordinate frame.
    ///   The **left plane** is a plane parallel to the **yz-plane**.
    /// * `right` defines the location of the **right plane** by its distance along
    ///   the **positive x-axis** from the origin of the coordinate frame.
    ///   The **right plane** is a plane parallel to the **yz-plane**.
    /// * `bottom` defines the location of the **bottom plane** by its distance along
    ///   the **negative y-axis** from the origin of the coordiante frame.
    ///   The **bottom plane** is a plane parallel to the **zx-plane**.
    /// * `top` defines the location of the **top plane** by its distance along
    ///   the **positive y-axis** from the origin of the coordinate frame.
    ///   The **top plane** is a plane parallel to the **zx-plane**.
    /// * `near` defines the location of the **near plane** by its distance along
    ///   the **negative z-axis** from the origin of the coordinate frame.
    ///   The **near plane** is a plane parallel to the **xy-plane**.
    /// * `far` defines the location of the **far plane** by its distance along
    ///   the **negative z-axis** from the origin of the coordinate frame.
    ///   The **far plane** is a plane parallel to the **xy-plane**.
    ///
    /// # Matrix Representation Of The Perspective Projection Transformation
    ///
    /// The underlying matrix is a homogeneous projective matrix with the following form
    ///
    /// ```text
    /// [ m[0, 0]  0         m[2, 0]  0       ]
    /// [ 0        m[1, 1]   m[2, 1]  0       ]
    /// [ 0        0         m[2, 2]  m[3, 2] ]
    /// [ 0        0        -1        0       ]
    /// ```
    ///
    /// where
    ///
    /// ```text
    /// m[0, 0] :=  (2 * near) / (right - (-left))        == (2 * near) / (right + left)
    /// m[2, 0] :=  (right + (-left)) / (right - (-left)) == (right - left) / (right + left)
    /// m[1, 1] :=  (2 * near) / (top - (-bottom))        == (2 * near) / (top + bottom)
    /// m[2, 1] :=  (top + (-bottom)) / (top - (-bottom)) == (top - bottom) / (top + bottom)
    /// m[2, 2] := -(far + near) / (far - near)
    /// m[3, 2] := -(2 * far * near) / (far - near)
    /// ```
    ///
    /// where the matrix entries are indexed in column-major order.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
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

    /// Construct a perspective projection matrix based on the `near`
    /// plane, the `far` plane and the vertical field of view angle `vfov` and
    /// the horizontal/vertical aspect ratio `aspect_ratio`.
    ///
    /// The `near` and `far` parameters are the absolute values of the positions
    /// of the **near plane** and the **far** plane, respectively, along the
    /// **negative z-axis**. In particular, the position of the **near plane** is
    /// `z == -near` and the position of the **far plane** is `z == -far`. The
    /// parameter `aspect_ratio` is the ratio of the width of the viewport to the
    /// height of the viewport.
    ///
    /// The resulting matrix represents a perspective projection transformation that
    /// follows OpenGL's mathematical characteristics. We state these precisely below.
    ///
    /// This parametrization is different from the standard frustum specification that
    /// defines the location of the frustum planes directly. Instead, this parametrization
    /// defines the frustum parameters as displacements along the relevant directions in
    /// the view space orthonormal frame. This defines a coordinate-independent frustum
    /// specification. The final matrix is the same.
    ///
    /// # Vector Space Details
    ///
    /// The matrix transforms from OpenGL's view space to OpenGL's clip space that maps to
    /// OpenGL's canonical view volume after depth normalization.
    ///
    /// ## A Visual Description Of The Vector Spaces.
    ///
    /// The **view space** is a vector space with a right-handed orthonormal frame defined
    /// as follows.
    ///
    /// * The origin of the coordinate system is `[0, 0, 0]^T`.
    /// * The **positive x-axis** is the horizontal direction and points right.
    /// * The **positive y-axis** is the vertical direction and points up.
    /// * The **positive z-axis** is the depth direction and points away from the
    ///   viewing frustum.
    /// * The **negative z-axis** is the viewing direction and points into the viewing
    /// frustum away from the viewer.
    ///
    /// The **clip space** is a vector space with a left-handed orthonormal frame defined
    /// as follows.
    ///
    /// * The origin of the coordinate system is `[0, 0, 0]^T`.
    /// * The **positive x-axis** is the horizontal direction and points to the right.
    /// * The **positive y-axis** is the vertical direction and points up.
    /// * The **positive z-axis** is the depth direction and points into the viewing volume.
    /// * The **negative z-axis** points away from the viewing volume towards the viewer.
    ///
    /// The **canonical view volume** is a vector space with a left-handed orthonormal
    /// frame identical to the clip space with bounds `[-1, 1] x [-1, 1] x [-1, 1]`.
    ///
    /// ## A Mathematically Precise Description Of The Vector Spaces.
    ///
    /// The **view space** is the vector space `V_v := (R^3, O_v, B_v)` where
    /// * The underlying vector space is `R^3`.
    /// * The **origin** is `O_v := [0, 0, 0]^T`.
    /// * The **basis** is `B_v := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
    ///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
    /// * The orthonormal frame `(O_v, B_v)` has a right-handed orientation.
    ///
    /// The **clip space** is the vector space `V_c := (R^3, O_c, B_c)` where
    /// * The underlying vector space is `R^3`.
    /// * The **origin** is `O_c := [0, 0, 0]^T`.
    /// * The **basis** is `B_c := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
    ///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
    /// * The orthonormal frame `(O_c, B_c)` has a left-handed orientation.
    /// * The view volume is parametrized by `[-left, right] x [-bottom, top] x [near, far]`.
    ///
    /// The **canonical view volume** is the vector space `V_cvv := (R^3, O_cvv, B_cvv)` where
    /// * The underlying vector space is `R^3`.
    /// * The **origin** is `O_cvv := [0, 0, 0]^T`.
    /// * The **basis** is `B_cvv := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
    ///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
    /// * The orthonormal frame `(O_cvv, B_cvv)` has a left-handed orientation.
    /// * The Canonical View Volume is parametrized by `[-1, 1] x [-1, 1] x [-1, 1]`.
    ///
    /// # Parameter Specification
    ///
    /// The fundamental parametrization of the perspective projection transformation
    /// is the specification based on defining the placement of the frustum bounds.
    /// We represent the frustum bounds by defining the placements with respect to the
    /// **view space** orthonormal frame vectors. More precisely, the fundamental
    /// parametrization is given by the parameters `left`, `right`, `bottom`, `top`,
    /// `near`, and `far` such that
    ///
    /// ```text
    /// left   > 0
    /// right  > 0
    /// bottom > 0
    /// top    > 0
    /// far    > near > 0
    /// ```
    ///
    /// where the parameters define the placement of the planes. The plane placement
    /// definitions follow.
    ///
    /// * `left` defines the location of the **left plane** by its distance along
    ///   the **negative x-axis** from the origin of the coordinate frame.
    ///   The **left plane** is a plane parallel to the **yz-plane**.
    /// * `right` defines the location of the **right plane** by its distance along
    ///   the **positive x-axis** from the origin of the coordinate frame.
    ///   The **right plane** is a plane parallel to the **yz-plane**.
    /// * `bottom` defines the location of the **bottom plane** by its distance along
    ///   the **negative y-axis** from the origin of the coordiante frame.
    ///   The **bottom plane** is a plane parallel to the **zx-plane**.
    /// * `top` defines the location of the **top plane** by its distance along
    ///   the **positive y-axis** from the origin of the coordinate frame.
    ///   The **top plane** is a plane parallel to the **zx-plane**.
    /// * `near` defines the location of the **near plane** by its distance along
    ///   the **negative z-axis** from the origin of the coordinate frame.
    ///   The **near plane** is a plane parallel to the **xy-plane**.
    /// * `far` defines the location of the **far plane** by its distance along
    ///   the **negative z-axis** from the origin of the coordinate frame.
    ///   The **far plane** is a plane parallel to the **xy-plane**.
    ///
    /// # Matrix Representation Of The Perspective Projection Transformation
    ///
    /// The underlying matrix is a homogeneous projective matrix with the following form
    ///
    /// ```text
    /// [ m[0, 0]  0         m[2, 0]  0       ]
    /// [ 0        m[1, 1]   m[2, 1]  0       ]
    /// [ 0        0         m[2, 2]  m[3, 2] ]
    /// [ 0        0        -1        0       ]
    /// ```
    ///
    /// where
    ///
    /// ```text
    /// m[0, 0] :=  (2 * near) / (right - (-left))        == (2 * near) / (right + left)
    /// m[2, 0] :=  (right + (-left)) / (right - (-left)) == (right - left) / (right + left)
    /// m[1, 1] :=  (2 * near) / (top - (-bottom))        == (2 * near) / (top + bottom)
    /// m[2, 1] :=  (top + (-bottom)) / (top - (-bottom)) == (top - bottom) / (top + bottom)
    /// m[2, 2] := -(far + near) / (far - near)
    /// m[3, 2] := -(2 * far * near) / (far - near)
    /// ```
    ///
    /// where the matrix entries are indexed in column-major order.
    ///
    /// # Symmetric Vertical Field Of View Parameter Specification
    ///
    /// The field of view parameters are given by
    ///
    /// * the aspect ratio `aspect_ratio`: the ratio of the width of the viewport
    ///   to the height of the viewport.
    /// * The vertical field of view angle `vfov`: the angle subtended by the
    ///   vertical part of the viewport from the origin.
    /// * The near plane placement `near`: the distance of the front clipping plane
    ///   from the origin along the **negative z-axis**.
    /// * The far plane placement `far`: the distance of the rear clipping plane
    ///   from the origin along the **negative z-axis**.
    ///
    /// and they satisfy the following properties
    ///
    /// ```text
    /// aspect_ratio > 0
    /// vfov         > 0
    /// vfov         < pi
    /// far          > near > 0
    /// ```
    ///
    /// The symmetric field of view parameters define a special case of the general
    /// perspective projection matrix. The perspective view volume is contained in
    /// `[-right, right] x [-top, top] x [-far, -near]`.
    ///
    /// # Matrix Representation Of The Symmetric Field Of View Perspective Projection
    ///
    /// The underlying matrix is a homogeneous projective matrix of the form
    ///
    /// ```text
    /// [ m[0, 0] 0         0        0       ]
    /// [ 0       m[1, 1]   0        0       ]
    /// [ 0       0         m[2, 2]  m[3, 2] ]
    /// [ 0       0        -1        0       ]
    /// ```
    ///
    /// where
    ///
    /// ```text
    /// m[0, 0] ==  1 / (aspect_ratio * tan(vfov / 2))
    /// m[1, 1] ==  1 / tan(vfov / 2)
    /// m[2, 2] == -(far + near) / (far - near)
    /// m[3, 2] == -(2 * far * near) / (far - near)
    /// ```
    ///
    /// where the matrix entries are indexed in column-major order.
    ///
    /// # Symmetric Field Of View Perspective Projection Matrix Representation
    ///
    /// We derive the elements of the perspective vertical field of view matrix.
    /// We are going to relative the elements of the general frustum specification
    /// to the tangent field of view elements. From this we can derive the special
    /// form of the resulting matrix.
    ///
    /// The aspect ratio `aspect_ratio` satisfies
    ///
    /// ```text
    /// aspect_ratio := width / height
    ///              == (right + left) / (top + bottom)
    ///              == (right + right) / (top + top)
    ///              == (2 * right) / (2 * top)
    ///              == right / top
    /// ```
    ///
    /// The tangent of the field of view angle `vfov` satisfies
    ///
    /// ```text
    /// tan(vfov / 2) == top / near
    /// ```
    ///
    /// We can now define the general frustum parameters in terms of the symmetric vertical
    /// field of view parameters
    ///
    /// ```text
    /// right  == aspect * near * tan(vfov / 2)
    /// left   == aspect * near * tan(vfov / 2)
    /// top    == near * tan(vfov / 2)
    /// bottom == near * tan(vfov / 2)
    /// ```
    ///
    /// and substituting these into the general perspective projection matrix
    ///
    /// ```text
    /// m[0, 0] == (2 * near) / (right + left)
    ///         == (2 * near) / (2 * right)
    ///         == near / right
    ///         == near / (near * aspect * tan(vfov / 2))
    ///         == 1 / (aspect * tan(vfov / 2))
    /// m[1, 1] == (2 * near) / (top + bottom)
    ///         == (2 * near) / (2 * top)
    ///         == near / top
    ///         == near / (near * tan(vfov / 2))
    ///         == 1 / tan(vfov / 2)
    /// m[2, 2] == -(far + near) / (far - near)
    /// m[3, 2] == -(2 * far * near) / (far - near)
    /// m[2, 0] == (right - left) / (right + left)
    ///         == (right - right) / (2 * right)
    ///         == 0
    /// m[2, 1] == (top - left) / (top + left)
    ///         == (top - top) / (2 * top)
    ///         == 0
    /// ```
    ///
    /// which yields the final matrix
    ///
    /// ```text
    /// [ m[0, 0] 0         0        0       ]
    /// [ 0       m[1, 1]   0        0       ]
    /// [ 0       0         m[2, 2]  m[3, 2] ]
    /// [ 0       0        -1        0       ]
    /// ```
    ///
    /// where
    ///
    /// ```text
    /// m[0, 0] ==  1 / (aspect_ratio * tan(vfov / 2))
    /// m[1, 1] ==  1 / tan(vfov / 2)
    /// m[2, 2] == -(far + near) / (far - near)
    /// m[3, 2] == -(2 * far * near) / (far - near)
    /// ```
    ///
    /// where the matrix entries are indexed in column-major order.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::Perspective3;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
    ///
    /// assert_relative_eq!(perspective.vfov(),         vfov.into(),  abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(perspective.aspect_ratio(), aspect_ratio, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(perspective.near(),         near,         abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(perspective.far(),          far,          abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    pub fn from_vfov<A>(vfov: A, aspect_ratio: S, near: S, far: S) -> Self
    where
        A: Into<Radians<S>>,
    {
        let spec_vfov = vfov.into();

        Self {
            matrix: Matrix4x4::from_perspective_vfov(spec_vfov, aspect_ratio, near, far),
        }
    }

    /// Get the aspect ratio. The aspect ratio is the ratio of the
    /// width of the viewing plane of the view volume to the height of the
    /// viewing plane of the view volume.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_transform::Perspective3;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
    /// let expected = aspect_ratio;
    /// let result = perspective.aspect_ratio();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn aspect_ratio(&self) -> S {
        // The perspective projection matrix has the form
        // ```text
        // [ m[0, 0]  0         m[2, 0]  0       ]
        // [ 0        m[1, 1]   m[2, 1]  0       ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0        -1        0       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  (2 * near) / (right - (-left))
        // m[2, 0] :=  (right + (-left)) / (right - (-left))
        // m[1, 1] :=  (2 * near) / (top - (-bottom))
        // m[2, 1] :=  (top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -(far + near) / (far - near)
        // m[3, 2] := -(2 * far * near) / (far - near)
        // ```
        // Recall that `aspect_ratio` is defined as
        // ```text
        // aspect_ratio := width / height
        //              == (right - (-left)) / (top - (-bottom))
        //              == (right + left) / (top + bottom)
        // ```
        // We can reconstruct the `aspect_ratio` parameter from the `m[0, 0]` and `m[1, 1]`
        // components as follows. Observe that
        // ```text
        // aspect_ratio == (right + left) / (top + bottom)
        //              == (1 / (top + bottom)) / (1 / (right + left))
        //              == ((2 * near) / (top + bottom)) / ((2 * near) / (right + left))
        //              == m[1, 1] / m[0, 0]
        // ```
        // which is the desired formula.
        //
        self.matrix[1][1] / self.matrix[0][0]
    }

    /// Get the vertical field of view angle.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::Perspective3;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
    /// let expected = vfov.into();
    /// let result = perspective.vfov();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn vfov(&self) -> Radians<S> {
        // The perspective projection matrix has the form
        // ```text
        // [ m[0, 0]  0         m[2, 0]  0       ]
        // [ 0        m[1, 1]   m[2, 1]  0       ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0        -1        0       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  (2 * near) / (right - (-left))
        // m[2, 0] :=  (right + (-left)) / (right - (-left))
        // m[1, 1] :=  (2 * near) / (top - (-bottom))
        // m[2, 1] :=  (top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -(far + near) / (far - near)
        // m[3, 2] := -(2 * far * near) / (far - near)
        // ```
        // To reconstruct the vertical field of view angle, we proceed as follows.
        // First, define the vectors `top_v` and `bottom_v`
        // ```text
        // top_v := top * y_hat + near * (-z_hat)
        //       == top * y_hat + (-near) * z_hat
        // bottom_v := bottom * (-y_hat) + near * (-z_hat)
        //          == (-bottom) * y_hat + (-near) * z_hat
        // ```
        // where `y_hat` is the view space up direction, and `z_hat` is the
        // direction of the viewer in view space. In particular, the direction
        // of the gaze direction `-z_hat` is the opposite the direction of the
        // viewer. The vector `top_v` is the position of the top plane parallel
        // to the **yz-plane**. The vector `bottom_v` is the position of the bottom
        // plane parallel to the **yz-plane**.
        //
        // Next, we derive the formulae for the tangents of the angles `angle_vfov_t`
        // and `angle_vfov_b` in terms of frustum parameters. The cosine and sine
        // of `angle_vfov_t` are defined by
        // ```text
        // cos(angle_vfov_t) := dot(top_v, -z_hat) / |top_v| == near / |top_v|
        // sin(angle_vfov_t) := dot(top_v, y_hat) / |top_v| == top / |top_v|
        // ```
        // The cosine and since of `angle_vfov_b` are defined by
        // ```text
        // cos(angle_vfov_b) := dot(bottom_v, -z_hat) / |bottom_v| == near / |bottom_v|
        // sin(angle_vfov_b) := dot(bottom_v, -y_hat) / |bottom_v| == bottom / |bottom_v|
        // ```
        // The tangent of the angle `angle_vfov_t` is given by
        // ```text
        // tan(angle_vfov_t) := sin(angle_vfov_t) / cos(angle_vfov_t)
        //                   == (top / | top_v |) / (near / |top_v|)
        //                   == top / near
        // ```
        // The tangent of the angle `angle_vfov_b` is given by
        // ```text
        // tan(angle_vfov_b) := sin(angle_vfov_b) / cos(angle_vfov_b)
        //                   == (bottom / | bottom_v |) / (near / |bottom_v|)
        //                   == bottom / near
        // ```
        // Therefore
        // ```text
        // tan(angle_vfov_t) == top / near
        // tan(angle_vfov_b) == bottom / near
        // ```
        // The vertical field of view angle of the viewport is given by
        // ```text
        // angle_vfov == angle_vfov_b + angle_vfov_t
        // ```
        // Using the summation formula for tangents of angles
        // ```text
        // tan(angle_vfov) == tan(angle_vfov_b + angle_vfov_t)
        //                 == [tan(angle_vfov_b) + tan(angle_vfov_t)] / [1 - tan(angle_vfov_b) * tan(angle_vfov_t)]
        // ```
        // This completes the derivation of the tangent of the vertical field of
        // view in terms of frustum parameters.
        //
        // Next, we derive the tangent formulae in terms of elements of the projection
        // matrix. Consider the relation for `angle_vfov_t`
        // ```text
        // (top - bottom) / (top + bottom) == m[2, 1]
        // (top - bottom) / (top + bottom) + 1 == [(top - bottom) + (top + bottom)] / (top + bottom)
        //                                     == (2 * top) / (top + bottom)
        //                                     == m[2, 1] + 1
        // ```
        // so that
        // ```text
        // (2 * top) / (top + bottom) == m[2, 1] + 1
        // ```
        // Now consider the relation for `angle_vfov_b`
        // ```text
        // (top - bottom) / (top + bottom) == m[2, 1]
        // (top - bottom) / (top + bottom) - 1 == [(top - bottom) - (top + bottom)] / (top + bottom)
        //                                     == -(2 * bottom) / (top + bottom)
        //                                     == m[2, 1] - 1
        // ```
        // or equivalently
        // ```text
        // (2 * bottom) / (top + bottom) == -m[2, 1] + 1
        // ```
        // From the perspective projection matrix
        // ```text
        // (2 * near) / (top + bottom) == m[1, 1]
        // ```
        // Consider the tangent formula for `angle_vfov_t`
        // ```text
        // tan(angle_vfov_t) == top / near
        //                   == (2 * top) / (2 * near)
        //                   == [(2 * top) / (top + bottom)] / [(2 * near) / (top + bottom)]
        //                   == (m[2, 1] + 1) / m[1, 1]
        // ```
        // and now consider the tangent formula for `angle_vfov_b`
        // ```text
        // tan(angle_vfov_b) == bottom / near
        //                   == (2 * bottom) / (2 * near)
        //                   == [(2 * bottom) / (top + bottom)] / [(2 * near) / (top + bottom)]
        //                   == (-m[2, 1] + 1) / m[1, 1]
        // ```
        // This gives us the component angles in terms of matrix elements.
        //
        // Finally, we substitute the tangent formulas into the tangent summation formula
        // ```text
        // tan(angle_vfov_t) + tan(angle_vfov_b)
        //     == [(m[2, 1] + 1) / m[1, 1]] + [(-m[2, 1] + 1) / m[1, 1]]
        //     == 2 / m[1, 1]
        //
        // 1 - tan(angle_vfov_t) * tan(angle_vfov_b)
        //     == 1 - [(-m[2, 1] + 1) / m[1, 1]] * [(m[2, 1] + 1) / m[1, 1]]
        //     == (m[1, 1] * m[1, 1] - (1 + m[2, 1] * m[2, 1])) / (m[1, 1] * m[1, 1])
        //     == (m[2, 1] * m[2, 1] + m[1, 1] * m[1, 1] - 1) / (m[1, 1] * m[1, 1])
        // ```
        // Finally, substituting the above into the tangent formula
        // ```text
        // tan(angle_vfov)
        //     == [tan(angle_vfov_t + angle_vfov_b)] / [1 - tan(angle_vfov_t) * tan(angle_vfov_b)]
        //     == (2 * m[1, 1]) / (m[2, 1] * m[2, 1] + m[1, 1] * m[1, 1] - 1)
        //     == numerator / denominator
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let numerator = two * self.matrix[1][1];
        let denominator = self.matrix[2][1] * self.matrix[2][1] + self.matrix[1][1] * self.matrix[1][1] - one;
        let tan_vfov = numerator / denominator;
        let vfov = tan_vfov.atan();

        Radians(vfov)
    }

    /// Get the position of the near plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **xy-plane** positioned along the **negative z-axis**.
    ///
    /// # Example (Frustum Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = near;
    /// let result = perspective.near();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Field Of View Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::Perspective3;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
    /// let expected = near;
    /// let result = perspective.near();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn near(&self) -> S {
        // The perspective projection matrix has the form
        // ```text
        // [ m[0, 0]  0         m[2, 0]  0       ]
        // [ 0        m[1, 1]   m[2, 1]  0       ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0        -1        0       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  (2 * near) / (right - (-left))
        // m[2, 0] :=  (right + (-left)) / (right - (-left))
        // m[1, 1] :=  (2 * near) / (top - (-bottom))
        // m[2, 1] :=  (top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -(far + near) / (far - near)
        // m[3, 2] := -(2 * far * near) / (far - near)
        // ```
        // Observe that
        // ```text
        // m[2, 2] + 1 == -[(far + near) / (far - near)] + 1
        //             == (-far - near + far - near) / (far - near)
        //             == -(2 * near) / (far - near)
        // ```
        // or equivalently
        // ```text
        // (2 * near) / (far - near) == -m[2, 2] - 1
        // ```
        // and also
        // ```text
        // m[2, 2] - 1 == -[(far + near) / (far - near)] - 1
        //             == [-far - near - (far - near)] / (far - near)
        //             == -(2 * far) / (far - near)
        // ```
        // or equivalently
        // ```text
        // (2 * far) / (far - near) == -m[2, 2] + 1
        // ```
        // Define
        // ```text
        // ratio := far / near
        // ```
        // the ratio of `far` to `near` so that
        // ```text
        // ratio == far / near
        //       == (2 * far) / (2 * near)
        //       == [(2 * far) / (far - near)] / [(2 * near) / (far - near)]
        //       == (-m[2, 2] + 1) / (-m[2, 2] - 1)
        // ```
        // so that
        // ```text
        // far == ratio * near
        // ```
        // Using the matrix element `m[3, 2]`
        // ```text
        // -(2 * far * near) / (far - near) == m[3, 2]
        // ```
        // or equivalently
        // ```text
        // -m[3, 2] == (2 * far * near) / (far - near)
        //          == (2 * (ratio * near) * near) / (ratio * near - near)
        //          == (2 * ratio * near * near) / ((ratio - 1) * near)
        //          == ((2 * ratio) / (ratio - 1)) * near
        // ```
        // we obtain
        // ```text
        // near == [(1 - ratio) / (2 * ratio)] * m[3, 2]
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[2][2] + one) / (-self.matrix[2][2] - one);

        ((one - ratio) / (two * ratio)) * self.matrix[3][2]
    }

    /// Get the position of the far plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **xy-plane** positioned along the **negative z-axis**.
    ///
    /// # Example (Frustum Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = far;
    /// let result = perspective.far();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Field Of View Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::Perspective3;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
    /// let expected = far;
    /// let result = perspective.far();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn far(&self) -> S {
        // The perspective projection matrix has the form
        // ```text
        // [ m[0, 0]  0         m[2, 0]  0       ]
        // [ 0        m[1, 1]   m[2, 1]  0       ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0        -1        0       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  (2 * near) / (right - (-left))
        // m[2, 0] :=  (right + (-left)) / (right - (-left))
        // m[1, 1] :=  (2 * near) / (top - (-bottom))
        // m[2, 1] :=  (top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -(far + near) / (far - near)
        // m[3, 2] := -(2 * far * near) / (far - near)
        // ```
        // Observe that
        // ```text
        // m[2, 2] + 1 == -[(far + near) / (far - near)] + 1
        //             == (-far - near + far - near) / (far - near)
        //             == -(2 * near) / (far - near)
        // ```
        // or equivalently
        // ```text
        // (2 * near) / (far - near) == -m[2, 2] - 1
        // ```
        // and also
        // ```text
        // m[2, 2] - 1 == -[(far + near) / (far - near)] - 1
        //             == [-far - near - (far - near)] / (far - near)
        //             == -(2 * far) / (far - near)
        // ```
        // or equivalently
        // ```text
        // (2 * far) / (far - near) == -m[2, 2] + 1
        // ```
        // Define
        // ```text
        // ratio := far / near
        // ```
        // the ratio of `far` to `near` so that
        // ```text
        // ratio == far / near
        //       == (2 * far) / (2 * near)
        //       == [(2 * far) / (far - near)] / [(2 * near) / (far - near)]
        //       == (-m[2, 2] + 1) / (-m[2, 2] - 1)
        // ```
        // so that
        // ```text
        // near == far / ratio
        // ```
        // Using the matrix element `m[3, 2]`
        // ```text
        // -(2 * far * near) / (far - near) == m[3, 2]
        // ```
        // or equivalently
        // ```text
        // -m[3, 2] == (2 * far * near) / (far - near)
        //          == (2 * far * (far / ratio)) / (far - (far / ratio))
        //          == [(2 * far * far) / ratio] / [(1 - (1 / ratio)) * far]
        //          == [(2 * far) / ratio] / [(ratio - 1) / ratio]
        //          == 2 * far / (ratio - 1)
        //          == [2 / (ratio - 1)] * far
        // ```
        // we obtain
        // ```text
        // far == [(1 - ratio) / 2] * m[3, 2]
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[2][2] + one) / (-self.matrix[2][2] - one);

        ((one - ratio) / two) * self.matrix[3][2]
    }

    /// Get the position of the right plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **yz-plane** positioned along the **positive x-axis**.
    ///
    /// # Example (Frustum Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = right;
    /// let result = perspective.right();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Field Of View Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::Perspective3;
    /// # use cglinalg_trigonometry::Degrees;
    /// # use core::f64;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
    /// let expected = (1_f64 / 10_f64) * (4_f64 / 3_f64) * f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64));
    /// let result = perspective.right();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn right(&self) -> S {
        // The perspective projection matrix has the form
        // ```text
        // [ m[0, 0]  0         m[2, 0]  0       ]
        // [ 0        m[1, 1]   m[2, 1]  0       ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0        -1        0       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  (2 * near) / (right - (-left))
        // m[2, 0] :=  (right + (-left)) / (right - (-left))
        // m[1, 1] :=  (2 * near) / (top - (-bottom))
        // m[2, 1] :=  (top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -(far + near) / (far - near)
        // m[3, 2] := -(2 * far * near) / (far - near)
        // ```
        // Observe that
        // ```text
        // m[2, 0] + 1 == [(right - left) / (right + left)] + 1
        //             == [(right - left) + (right + left)] / (right + left)
        //             == (2 * right) / (right + left)
        // ```
        // and
        // ```text
        // m[2, 0] - 1 == [(right - left) / (right + left)] - 1
        //             == [(right - left) - (right + left))] / (right + left)
        //             == -(2 * left) / (right + left)
        // ```
        // so that
        // ```text
        // (2 * right) / (right + left) == m[2, 0] + 1
        // (2 * left) / (right + left) == -m[2, 0] + 1
        // ```
        // Define
        // ```text
        // ratio := right / left
        // ```
        // so that
        // ```text
        // ratio == right / left
        //       == (2 * right) / (2 * left)
        //       == [(2 * right) / (right + left)] / [(2 * left) / (right + left)]
        //       == (m[2, 0] + 1) / (-m[2, 0] + 1)
        // ```
        // and
        // ```text
        // left == right / ratio
        // ```
        // Using matrix element `m[0, 0]`
        // ```text
        // m[0, 0] == (2 * near) / (right + left)
        // ```
        // or equivalently
        // ```text
        // 2 * near / m[0, 0] == (2 * near) * (1 / m[0, 0])
        //                    == right + left
        //                    == right + right / ratio
        //                    == (1 + (1 / ratio)) * right
        //                    == ((ratio + 1) / ratio) * right
        // ```
        // We obtain
        // ```text
        // right == [(2 * near) * (ratio / (ratio + 1))] * (1 / m[0, 0])
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let ratio = (self.matrix[2][0] + one) / (-self.matrix[2][0] + one);
        let near = self.near();

        ((two * near) * (ratio / (ratio + one))) * (one / self.matrix[0][0])
    }

    /// Get the position of the left plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **yz-plane** positioned along the **positive x-axis**.
    ///
    /// # Example (Frustum Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = left;
    /// let result = perspective.left();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Field Of View Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::Perspective3;
    /// # use cglinalg_trigonometry::Degrees;
    /// # use core::f64;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
    /// let expected = (1_f64 / 10_f64) * (4_f64 / 3_f64) * f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64));
    /// let result = perspective.left();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn left(&self) -> S {
        // The perspective projection matrix has the form
        // ```text
        // [ m[0, 0]  0         m[2, 0]  0       ]
        // [ 0        m[1, 1]   m[2, 1]  0       ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0        -1        0       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  (2 * near) / (right - (-left))
        // m[2, 0] :=  (right + (-left)) / (right - (-left))
        // m[1, 1] :=  (2 * near) / (top - (-bottom))
        // m[2, 1] :=  (top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -(far + near) / (far - near)
        // m[3, 2] := -(2 * far * near) / (far - near)
        // ```
        // Observe that
        // ```text
        // m[2, 0] + 1 == [(right - left) / (right + left)] + 1
        //             == [(right - left) + (right + left)] / (right + left)
        //             == (2 * right) / (right + left)
        // ```
        // and
        // ```text
        // m[2, 0] - 1 == [(right - left) / (right + left)] - 1
        //             == [(right - left) - (right + left))] / (right + left)
        //             == -(2 * left) / (right + left)
        // ```
        // so that
        // ```text
        // (2 * right) / (right + left) == m[2, 0] + 1
        // (2 * left) / (right + left) == -m[2, 0] + 1
        // ```
        // Define
        // ```text
        // ratio := right / left
        // ```
        // so that
        // ```text
        // ratio == right / left
        //       == (2 * right) / (2 * left)
        //       == [(2 * right) / (right + left)] / [(2 * left) / (right + left)]
        //       == (m[2, 0] + 1) / (-m[2, 0] + 1)
        // ```
        // and
        // ```text
        // right == ratio * left
        // ```
        // Using matrix element `m[0, 0]`
        // ```text
        // m[0, 0] == (2 * near) / (right + left)
        // ```
        // or equivalently
        // ```text
        // 2 * near / m[0, 0] == (2 * near) * (1 / m[0, 0])
        //                    == right + left
        //                    == ratio * left + left
        //                    == (ratio + 1) * left
        // ```
        // We obtain
        // ```text
        // left == [(2 * near) * (1 / (ratio + 1))] * (1 / m[0, 0])
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let ratio = (self.matrix[2][0] + one) / (-self.matrix[2][0] + one);
        let near = self.near();

        ((two * near) * (one / (ratio + one))) * (one / self.matrix[0][0])
    }

    /// Get the position of the top plane of the viewing
    /// frustum described by the perspective projection of the plane
    /// parallel to the **zx-plane** positioned along the **positive y-axis**.
    ///
    /// # Example (Frustum Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = top;
    /// let result = perspective.top();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::Perspective3;
    /// # use cglinalg_trigonometry::Degrees;
    /// # use core::f64;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
    /// let expected = (1_f64 / 10_f64) * (f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64)));
    /// let result = perspective.top();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn top(&self) -> S {
        // The perspective projection matrix has the form
        // ```text
        // [ m[0, 0]  0         m[2, 0]  0       ]
        // [ 0        m[1, 1]   m[2, 1]  0       ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0        -1        0       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  (2 * near) / (right - (-left))
        // m[2, 0] :=  (right + (-left)) / (right - (-left))
        // m[1, 1] :=  (2 * near) / (top - (-bottom))
        // m[2, 1] :=  (top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -(far + near) / (far - near)
        // m[3, 2] := -(2 * far * near) / (far - near)
        // ```
        // Observe that
        // ```text
        // m[2, 1] + 1 == [(top - bottom) / (top + bottom)] + 1
        //             == [(top - bottom) + (top + bottom)] / (top + bottom)
        //             == (2 * top) / (top + bottom)
        // ```
        // and
        // ```text
        // m[2, 1] - 1 == [(top - bottom) / (top + bottom)] - 1
        //             == [(top - bottom) - (top + bottom)] / (top + bottom)
        //             == -(2 * bottom) / (top + bottom)
        // ```
        // so that
        // ```text
        // (2 * top) / (top + bottom) == m[2, 1] + 1
        // (2 * bottom) / (top + bottom) == -m[2, 1] + 1
        // ```
        // Define
        // ```text
        // ratio := top / bottom
        // ```
        // so that
        // ```text
        // ratio == top / bottom
        //       == (2 * top) / (2 * bottom)
        //       == [(2 * top) / (top + bottom)] / [(2 * bottom) / (top + bottom)]
        //       == (m[2, 1] + 1) / (-m[2, 1] + 1)
        // ```
        // and
        // ```text
        // bottom = top / ratio
        // ```
        // Using the matrix element `m[1, 1]`
        // ```text
        // m[1, 1] == (2 * near) / (top + bottom)
        // ```
        // or equivalently
        // ```text
        // (2 * near) / m[1, 1] == (2 * near) * (1 / m[1, 1])
        //                      == top + bottom
        //                      == top + top / ratio
        //                      == (1 + (1 / ratio)) * top
        //                      == ((ratio + 1) / ratio) * top
        // ```
        // We obtain
        // ```text
        // top == [(2 * near) * (ratio / (ratio + 1))] * (1 / m[1, 1])
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let ratio = (self.matrix[2][1] + one) / (-self.matrix[2][1] + one);
        let near = self.near();

        ((two * near) * (ratio / (ratio + one))) * (one / self.matrix[1][1])
    }

    /// Get the position of the bottom plane of the viewing
    /// frustum descibed by the perspective projection of the plane
    /// parallel to the **zx-plane** positioned along the **positive y-axis**.
    ///
    /// # Example (Frustum Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::new(left, right, bottom, top, near, far);
    /// let expected = bottom;
    /// let result = perspective.bottom();
    ///
    /// assert_relative_eq!(result, &expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// # Example (Field Of View Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::Perspective3;
    /// # use cglinalg_trigonometry::Degrees;
    /// # use core::f64;
    /// #
    /// let vfov = Degrees(72_f64);
    /// let aspect_ratio = 800_f64 / 600_f64;
    /// let near = 0.1_f64;
    /// let far = 100_f64;
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
    /// let expected = (1_f64 / 10_f64) * (f64::sqrt(5_f64 - 2_f64 * f64::sqrt(5_f64)));
    /// let result = perspective.bottom();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn bottom(&self) -> S {
        // The perspective projection matrix has the form
        // ```text
        // [ m[0, 0]  0         m[2, 0]  0       ]
        // [ 0        m[1, 1]   m[2, 1]  0       ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0        -1        0       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  (2 * near) / (right - (-left))
        // m[2, 0] :=  (right + (-left)) / (right - (-left))
        // m[1, 1] :=  (2 * near) / (top - (-bottom))
        // m[2, 1] :=  (top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -(far + near) / (far - near)
        // m[3, 2] := -(2 * far * near) / (far - near)
        // ```
        // Observe that
        // ```text
        // m[2, 1] + 1 == [(top - bottom) / (top + bottom)] + 1
        //             == [(top - bottom) + (top + bottom)] / (top + bottom)
        //             == (2 * top) / (top + bottom)
        // ```
        // and
        // ```text
        // m[2, 1] - 1 == [(top - bottom) / (top + bottom)] - 1
        //             == [(top - bottom) - (top + bottom)] / (top + bottom)
        //             == -(2 * bottom) / (top + bottom)
        // ```
        // so that
        // ```text
        // (2 * top) / (top + bottom) == m[2, 1] + 1
        // (2 * bottom) / (top + bottom) == -m[2, 1] + 1
        // ```
        // Define
        // ```text
        // ratio := top / bottom
        // ```
        // so that
        // ```text
        // ratio == top / bottom
        //       == (2 * top) / (2 * bottom)
        //       == [(2 * top) / (top + bottom)] / [(2 * bottom) / (top + bottom)]
        //       == (m[2, 1] + 1) / (-m[2, 1] + 1)
        // ```
        // and
        // ```text
        // top = ratio * bottom
        // ```
        // Using the matrix element `m[1, 1]`
        // ```text
        // m[1, 1] == (2 * near) / (top + bottom)
        // ```
        // or equivalently
        // ```text
        // (2 * near) / m[1, 1] == (2 * near) * (1 / m[1, 1])
        //                      == top + bottom
        //                      == ratio * bottom + bottom
        //                      == (ratio + 1) * bottom
        // ```
        // We obtain
        // ```text
        // bottom == [(2 * near) * (1 / (ratio + 1))] * (1 / m[1, 1])
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let ratio = (self.matrix[2][1] + one) / (-self.matrix[2][1] + one);
        let near = self.near();

        ((two * near) * (one / (ratio + one))) * (one / self.matrix[1][1])
    }

    /// Get the matrix that implements the perspective projection transformation.
    ///
    /// # Example (Frustum Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
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
    ///
    /// # Example (Field Of View Parametrization)
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
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
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
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

    /// Apply the projective projection transformation to a point.
    ///
    /// # Example (Frustum Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
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
    ///
    /// # Example (Field Of View Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Point3,
    /// # };
    /// # use cglinalg_transform::Perspective3;
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
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
    /// let point = Point3::new(-1_f64, -1_f64, 30_f64);
    /// let expected = Point3::new(3_f64 / 120_f64, 1_f64 / 30_f64, 3230_f64 / 2970_f64);
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
    /// # Example (Frustum Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
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
    ///
    /// # Example (Field Of View Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Perspective3;
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
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
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
    /// # Example (Frustum Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Point3;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
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
    ///
    /// # Example (Field Of View Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Point3,
    /// # };
    /// # use cglinalg_transform::Perspective3;
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
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
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
        // [ c0r0 c1r0 c2r0 c3r0 ]    [ (2*n)/(r + l)   0              (r - l)/(r + l)    0               ]
        // [ c0r1 c1r1 c2r1 c3r1 ] == [ 0              (2*n)/(t + b)   (t - b)/(t + b)    0               ]
        // [ c0r2 c1r2 c2r2 c3r2 ]    [ 0              0              -(f + n)/(f - n)   -(2*f*n)/(f - n) ]
        // [ c0r3 c1r3 c2r3 c3r3 ]    [ 0              0              -1                  0               ]
        // ```
        //
        // The inverse matrix of the perspective projection matrix has the form
        // ```text
        // [ c0r0 c1r0 c2r0 c3r0 ]    [ (r + l)/(2*n)   0               0                   (r - l)/(2*n)   ]
        // [ c0r1 c1r1 c2r1 c3r1 ] == [ 0               (t + b)/(2*n)   0                   (t - b)/(2*n)   ]
        // [ c0r2 c1r2 c2r2 c3r2 ]    [ 0               0               0                  -1               ]
        // [ c0r3 c1r3 c2r3 c3r3 ]    [ 0               0               (f - n)/(-2*f*n)   -(f - n)/(2*f*n) ]
        // ```
        //
        // We can save nine multiplications, nine additions, and one matrix
        // construction by only applying the nonzero elements
        // c0r0, c1r1, c2r3, c3r0, c3r1, c3r2, and c3r3 to the input point.
        //
        let one = S::one();
        let two = one + one;
        let c0r0 = (self.right() + self.left()) / (two * self.near());
        let c1r1 = (self.top() + self.bottom()) / (two * self.near());
        let c2r3 = (self.near() - self.far()) / (two * self.far() * self.near());
        let c3r0 = (self.right() - self.left()) / (two * self.near());
        let c3r1 = (self.top() - self.bottom()) / (two * self.near());
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
    /// # Example (Frustum Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Perspective3;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
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
    ///
    /// # Example (Field Of View Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::Perspective3;
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
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
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
        // [ c0r0 c1r0 c2r0 c3r0 ]    [ (2*n)/(r + l)   0              (r - l)/(r + l)    0               ]
        // [ c0r1 c1r1 c2r1 c3r1 ] == [ 0              (2*n)/(t + b)   (t - b)/(t + b)    0               ]
        // [ c0r2 c1r2 c2r2 c3r2 ]    [ 0              0              -(f + n)/(f - n)   -(2*f*n)/(f - n) ]
        // [ c0r3 c1r3 c2r3 c3r3 ]    [ 0              0              -1                  0               ]
        // ```
        //
        // The inverse matrix of the perspective projection matrix has the form
        // ```text
        // [ c0r0 c1r0 c2r0 c3r0 ]    [ (r + l)/(2*n)   0               0                   (r - l)/(2*n)   ]
        // [ c0r1 c1r1 c2r1 c3r1 ] == [ 0               (t + b)/(2*n)   0                   (t - b)/(2*n)   ]
        // [ c0r2 c1r2 c2r2 c3r2 ]    [ 0               0               0                  -1               ]
        // [ c0r3 c1r3 c2r3 c3r3 ]    [ 0               0               (f - n)/(-2*f*n)   -(f - n)/(2*f*n) ]
        // ```
        //
        // We can save nine multiplications, nine additions, and one matrix
        // construction by only applying the nonzero elements
        // c0r0, c1r1, c2r3, c3r0, c3r1, c3r2, and c3r3 to the input vector.
        //
        let one = S::one();
        let two = one + one;
        let c0r0 = (self.right() + self.left()) / (two * self.near());
        let c1r1 = (self.top() + self.bottom()) / (two * self.near());
        let c2r3 = (self.near() - self.far()) / (two * self.far() * self.near());
        let c3r0 = (self.right() - self.left()) / (two * self.near());
        let c3r1 = (self.top() - self.bottom()) / (two * self.near());
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
    /// # Example (Frustum Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
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
    ///
    /// # Example (Field Of View Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Perspective3;
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
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
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
    /// # Example (Frustum Parametrization)
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
    /// let left = 3_f64;
    /// let right = 3_f64;
    /// let bottom = 2_f64;
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
    ///
    /// # Example (Field Of View Parametrization)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_transform::{
    /// #     Perspective3,
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
    /// let perspective = Perspective3::from_vfov(vfov, aspect_ratio, near, far);
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


/// An orthographic projection transformation.
///
/// The data type represents an orthographic projection transformation that follows
/// OpenGL's mathematical characteristics. We state these precisely below.
///
/// This parametrization is different from the standard frustum specification that
/// defines the location of the frustum planes directly. Instead, this parametrization
/// defines the frustum parameters as displacements along the relevant directions in
/// the view space orthonormal frame. This defines a coordinate-independent frustum
/// specification. The final matrix is the same.
///
/// # Vector Space Details
///
/// The matrix transforms from OpenGL's view space to OpenGL's clip space that maps to
/// OpenGL's canonical view volume after depth normalization.
///
/// ## A Visual Description Of The Vector Spaces.
///
/// The **view space** is a vector space with a right-handed orthonormal frame defined
/// as follows.
///
/// * The origin of the coordinate system is `[0, 0, 0]^T`.
/// * The **positive x-axis** is the horizontal direction and points right.
/// * The **positive y-axis** is the vertical direction and points up.
/// * The **positive z-axis** is the depth direction and points away from the
///   viewing frustum.
/// * The **negative z-axis** is the viewing direction and points into the viewing
/// frustum away from the viewer.
///
/// The **clip space** is a vector space with a left-handed orthonormal frame defined
/// as follows.
///
/// * The origin of the coordinate system is `[0, 0, 0]^T`.
/// * The **positive x-axis** is the horizontal direction and points to the right.
/// * The **positive y-axis** is the vertical direction and points up.
/// * The **positive z-axis** is the depth direction and points into the viewing volume.
/// * The **negative z-axis** points away from the viewing volume towards the viewer.
///
/// The **canonical view volume** is a vector space with a left-handed orthonormal
/// frame identical to the clip space with bounds `[-1, 1] x [-1, 1] x [-1, 1]`.
///
/// ## A Mathematically Precise Description Of The Vector Spaces.
///
/// The **view space** is the vector space `V_v := (R^3, O_v, B_v)` where
/// * The underlying vector space is `R^3`.
/// * The **origin** is `O_v := [0, 0, 0]^T`.
/// * The **basis** is `B_v := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
/// * The orthonormal frame `(O_v, B_v)` has a right-handed orientation.
///
/// The **clip space** is the vector space `V_c := (R^3, O_c, B_c)` where
/// * The underlying vector space is `R^3`.
/// * The **origin** is `O_c := [0, 0, 0]^T`.
/// * The **basis** is `B_c := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
/// * The orthonormal frame `(O_c, B_c)` has a left-handed orientation.
/// * The view volume is parametrized by `[-left, right] x [-bottom, top] x [near, far]`.
///
/// The **canonical view volume** is the vector space `V_cvv := (R^3, O_cvv, B_cvv)` where
/// * The underlying vector space is `R^3`.
/// * The **origin** is `O_cvv := [0, 0, 0]^T`.
/// * The **basis** is `B_cvv := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
/// * The orthonormal frame `(O_cvv, B_cvv)` has a left-handed orientation.
/// * The Canonical View Volume is parametrized by `[-1, 1] x [-1, 1] x [-1, 1]`.
///
/// # Parameter Specification
///
/// The fundamental parametrization of the orthographic projection transformation
/// is the specification based on defining the placement of the frustum bounds.
/// We represent the frustum bounds by defining the placements with respect to the
/// **view space** orthonormal frame vectors. More precisely, the fundamental
/// parametrization is given by the parameters `left`, `right`, `bottom`, `top`,
/// `near`, and `far` such that
///
/// ```text
/// left   > 0
/// right  > 0
/// bottom > 0
/// top    > 0
/// far    > near > 0
/// ```
///
/// where the parameters define the placement of the planes. The plane placement
/// definitions follow.
///
/// * `left` defines the location of the **left plane** by its distance along
///   the **negative x-axis** from the origin of the coordinate frame.
///   The **left plane** is a plane parallel to the **yz-plane**.
/// * `right` defines the location of the **right plane** by its distance along
///   the **positive x-axis** from the origin of the coordinate frame.
///   The **right plane** is a plane parallel to the **yz-plane**.
/// * `bottom` defines the location of the **bottom plane** by its distance along
///   the **negative y-axis** from the origin of the coordiante frame.
///   The **bottom plane** is a plane parallel to the **zx-plane**.
/// * `top` defines the location of the **top plane** by its distance along
///   the **positive y-axis** from the origin of the coordinate frame.
///   The **top plane** is a plane parallel to the **zx-plane**.
/// * `near` defines the location of the **near plane** by its distance along
///   the **negative z-axis** from the origin of the coordinate frame.
///   The **near plane** is a plane parallel to the **xy-plane**.
/// * `far` defines the location of the **far plane** by its distance along
///   the **negative z-axis** from the origin of the coordinate frame.
///   The **far plane** is a plane parallel to the **xy-plane**.
///
/// # Matrix Representation Of The Orthographic Projection Transformation
///
/// The underlying matrix is a homogeneous affine matrix with the following form
///
/// ```text
/// [ m[0, 0]  0        0        m[3, 0] ]
/// [ 0        m[1, 1]  0        m[3, 1] ]
/// [ 0        0        m[2, 2]  m[3, 2] ]
/// [ 0        0        0        1       ]
/// ```
///
/// where
///
/// ```text
/// m[0, 0] ==  2 / (right - (-left))                 == 2 / (right + left)
/// m[3, 0] == -(right + (-left)) / (right - (-left)) == -(right - left) / (right + left)
/// m[1, 1] ==  2 / (top - (-bottom))                 == 2 / (top + bottom)
/// m[3, 1] == -(top + (-bottom)) / (top - (-bottom)) == -(top - bottom) / (top + bottom)
/// m[2, 2] == -2 / (far - near)
/// m[3, 2] == -(far + near) / (far - near)
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
    /// Construct a new orthographic projection matrix.
    ///
    /// The data type represents an orthographic projection transformation that follows
    /// OpenGL's mathematical characteristics. We state these precisely below.
    ///
    /// This parametrization is different from the standard frustum specification that
    /// defines the location of the frustum planes directly. Instead, this parametrization
    /// defines the frustum parameters as displacements along the relevant directions in
    /// the view space orthonormal frame. This defines a coordinate-independent frustum
    /// specification. The final matrix is the same.
    ///
    /// # Vector Space Details
    ///
    /// The matrix transforms from OpenGL's view space to OpenGL's clip space that maps to
    /// OpenGL's canonical view volume after depth normalization.
    ///
    /// ## A Visual Description Of The Vector Spaces.
    ///
    /// The **view space** is a vector space with a right-handed orthonormal frame defined
    /// as follows.
    ///
    /// * The origin of the coordinate system is `[0, 0, 0]^T`.
    /// * The **positive x-axis** is the horizontal direction and points right.
    /// * The **positive y-axis** is the vertical direction and points up.
    /// * The **positive z-axis** is the depth direction and points away from the
    ///   viewing frustum.
    /// * The **negative z-axis** is the viewing direction and points into the viewing
    /// frustum away from the viewer.
    ///
    /// The **clip space** is a vector space with a left-handed orthonormal frame defined
    /// as follows.
    ///
    /// * The origin of the coordinate system is `[0, 0, 0]^T`.
    /// * The **positive x-axis** is the horizontal direction and points to the right.
    /// * The **positive y-axis** is the vertical direction and points up.
    /// * The **positive z-axis** is the depth direction and points into the viewing volume.
    /// * The **negative z-axis** points away from the viewing volume towards the viewer.
    ///
    /// The **canonical view volume** is a vector space with a left-handed orthonormal
    /// frame identical to the clip space with bounds `[-1, 1] x [-1, 1] x [-1, 1]`.
    ///
    /// ## A Mathematically Precise Description Of The Vector Spaces.
    ///
    /// The **view space** is the vector space `V_v := (R^3, O_v, B_v)` where
    /// * The underlying vector space is `R^3`.
    /// * The **origin** is `O_v := [0, 0, 0]^T`.
    /// * The **basis** is `B_v := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
    ///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
    /// * The orthonormal frame `(O_v, B_v)` has a right-handed orientation.
    ///
    /// The **clip space** is the vector space `V_c := (R^3, O_c, B_c)` where
    /// * The underlying vector space is `R^3`.
    /// * The **origin** is `O_c := [0, 0, 0]^T`.
    /// * The **basis** is `B_c := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
    ///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
    /// * The orthonormal frame `(O_c, B_c)` has a left-handed orientation.
    /// * The view volume is parametrized by `[-left, right] x [-bottom, top] x [near, far]`.
    ///
    /// The **canonical view volume** is the vector space `V_cvv := (R^3, O_cvv, B_cvv)` where
    /// * The underlying vector space is `R^3`.
    /// * The **origin** is `O_cvv := [0, 0, 0]^T`.
    /// * The **basis** is `B_cvv := { [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T }` where
    ///   `x_hat := [1, 0, 0]^T`, `y_hat := [0, 1, 0]^T`, and `z_hat := [0, 0, 1]^T`.
    /// * The orthonormal frame `(O_cvv, B_cvv)` has a left-handed orientation.
    /// * The Canonical View Volume is parametrized by `[-1, 1] x [-1, 1] x [-1, 1]`.
    ///
    /// # Parameter Specification
    ///
    /// The fundamental parametrization of the orthographic projection transformation
    /// is the specification based on defining the placement of the frustum bounds.
    /// We represent the frustum bounds by defining the placements with respect to the
    /// **view space** orthonormal frame vectors. More precisely, the fundamental
    /// parametrization is given by the parameters `left`, `right`, `bottom`, `top`,
    /// `near`, and `far` such that
    ///
    /// ```text
    /// left   > 0
    /// right  > 0
    /// bottom > 0
    /// top    > 0
    /// far    > near > 0
    /// ```
    ///
    /// where the parameters define the placement of the planes. The plane placement
    /// definitions follow.
    ///
    /// * `left` defines the location of the **left plane** by its distance along
    ///   the **negative x-axis** from the origin of the coordinate frame.
    ///   The **left plane** is a plane parallel to the **yz-plane**.
    /// * `right` defines the location of the **right plane** by its distance along
    ///   the **positive x-axis** from the origin of the coordinate frame.
    ///   The **right plane** is a plane parallel to the **yz-plane**.
    /// * `bottom` defines the location of the **bottom plane** by its distance along
    ///   the **negative y-axis** from the origin of the coordiante frame.
    ///   The **bottom plane** is a plane parallel to the **zx-plane**.
    /// * `top` defines the location of the **top plane** by its distance along
    ///   the **positive y-axis** from the origin of the coordinate frame.
    ///   The **top plane** is a plane parallel to the **zx-plane**.
    /// * `near` defines the location of the **near plane** by its distance along
    ///   the **negative z-axis** from the origin of the coordinate frame.
    ///   The **near plane** is a plane parallel to the **xy-plane**.
    /// * `far` defines the location of the **far plane** by its distance along
    ///   the **negative z-axis** from the origin of the coordinate frame.
    ///   The **far plane** is a plane parallel to the **xy-plane**.
    ///
    /// # Matrix Representation Of The Orthographic Projection Transformation
    ///
    /// The underlying matrix is a homogeneous affine matrix with the following form
    ///
    /// ```text
    /// [ m[0, 0]  0        0        m[3, 0] ]
    /// [ 0        m[1, 1]  0        m[3, 1] ]
    /// [ 0        0        m[2, 2]  m[3, 2] ]
    /// [ 0        0        0        1       ]
    /// ```
    ///
    /// where
    ///
    /// ```text
    /// m[0, 0] ==  2 / (right - (-left))                 == 2 / (right + left)
    /// m[3, 0] == -(right + (-left)) / (right - (-left)) == -(right - left) / (right + left)
    /// m[1, 1] ==  2 / (top - (-bottom))                 == 2 / (top + bottom)
    /// m[3, 1] == -(top + (-bottom)) / (top - (-bottom)) == -(top - bottom) / (top + bottom)
    /// m[2, 2] == -2 / (far - near)
    /// m[3, 2] == -(far + near) / (far - near)
    /// ```
    /// where the matrix entries are indexed in column-major order.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Vector3;
    /// # use cglinalg_transform::Orthographic3;
    /// #
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
        // The orthographic projection matrix has the form
        // ```text
        // [ m[0, 0]  0         0        m[3, 0] ]
        // [ 0        m[1, 1]   0        m[3, 1] ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0         0        1       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  2 / (right - (-left))
        // m[3, 0] := -(right + (-left)) / (right - (-left))
        // m[1, 1] :=  2 / (top - (-bottom))
        // m[3, 1] := -(top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -2 / (far - near)
        // m[3, 2] := -(far + near) / (far - near)
        // ```
        // Observe that
        // ```text
        // m[3, 2] + 1 == -[(far + near) / (far - near)] + 1
        //             == (-(far + near) + (far - near)) / (far - near)
        //             == -(2 * near) / (far - near)
        // ```
        // or equivalently
        // ```text
        // (2 * near) / (far - near) == -m[3, 2] - 1
        // ```
        // and also
        // ```text
        // m[3, 2] - 1 == -[(far + near) / (far - near)] - 1
        //             == [-(far + near) - (far - near)] / (far - near)
        //             == -(2 * far) / (far - near)
        // ```
        // or equivalently
        // ```text
        // (2 * far) / (far - near) == -m[3, 2] + 1
        // ```
        // Define
        // ```text
        // ratio := far / near
        // ```
        // the ratio of `far` to `near` so that
        // ```text
        // ratio == far / near
        //       == (2 * far) / (2 * near)
        //       == [(2 * far) / (far - near)] / [(2 * near) / (far - near)]
        //       == (-m[3, 2] + 1) / (-m[3, 2] - 1)
        // ```
        // so that
        // ```text
        // far == ratio * near
        // ```
        // Using the matrix element `m[2, 2]`
        // ```text
        // m[2, 2] == -2 / (far - near)
        // ```
        // or equivalently
        // ```text
        // -2 / m[2, 2] == far - near
        //              == ratio * near - near
        //              == (ratio - 1) * near
        // ```
        // we obtain
        // ```text
        // near == -2 * (1 / (ratio - 1)) * (1 / m[2, 2])
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[3][2] + one) / (-self.matrix[3][2] - one);

        (-two * (one / (ratio - one))) * (one / self.matrix[2][2])
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
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
        // The orthographic projection matrix has the form
        // ```text
        // [ m[0, 0]  0         0        m[3, 0] ]
        // [ 0        m[1, 1]   0        m[3, 1] ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0         0        1       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  2 / (right - (-left))
        // m[3, 0] := -(right + (-left)) / (right - (-left))
        // m[1, 1] :=  2 / (top - (-bottom))
        // m[3, 1] := -(top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -2 / (far - near)
        // m[3, 2] := -(far + near) / (far - near)
        // ```
        // Observe that
        // ```text
        // m[3, 2] + 1 == -[(far + near) / (far - near)] + 1
        //             == (-(far + near) + (far - near)) / (far - near)
        //             == -(2 * near) / (far - near)
        // ```
        // or equivalently
        // ```text
        // (2 * near) / (far - near) == -m[3, 2] - 1
        // ```
        // and also
        // ```text
        // m[3, 2] - 1 == -[(far + near) / (far - near)] - 1
        //             == [-(far + near) - (far - near)] / (far - near)
        //             == -(2 * far) / (far - near)
        // ```
        // or equivalently
        // ```text
        // (2 * far) / (far - near) == -m[3, 2] + 1
        // ```
        // Define
        // ```text
        // ratio := far / near
        // ```
        // the ratio of `far` to `near` so that
        // ```text
        // ratio == far / near
        //       == (2 * far) / (2 * near)
        //       == [(2 * far) / (far - near)] / [(2 * near) / (far - near)]
        //       == (-m[3, 2] + 1) / (-m[3, 2] - 1)
        // ```
        // so that
        // ```text
        // near == far / ratio
        // ```
        // Using the matrix element `m[2, 2]`
        // ```text
        // m[2, 2] == -2 / (far - near)
        // ```
        // or equivalently
        // ```text
        // -2 / m[2, 2] == far - near
        //              == far - far / ratio
        //              == (1 - (1 / ratio)) * far
        //              == ((ratio - 1) / ratio) * far
        // ```
        // we obtain
        // ```text
        // far == -2 * (ratio / (ratio - 1)) * (1 / m[2, 2])
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[3][2] + one) / (-self.matrix[3][2] - one);

        (-two * (ratio / (ratio - one))) * (one / self.matrix[2][2])
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
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
        // The orthographic projection matrix has the form
        // ```text
        // [ m[0, 0]  0         0        m[3, 0] ]
        // [ 0        m[1, 1]   0        m[3, 1] ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0         0        1       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  2 / (right - (-left))
        // m[3, 0] := -(right + (-left)) / (right - (-left))
        // m[1, 1] :=  2 / (top - (-bottom))
        // m[3, 1] := -(top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -2 / (far - near)
        // m[3, 2] := -(far + near) / (far - near)
        // ```
        // Observe that
        // ```text
        // m[3, 0] - 1 == -[(right - left) / (right + left)] - 1
        //             == [-(right - left) - (right + left)] / (right + left)
        //             == [-right + left - right - left] / (right + left)
        //             == -(2 * right) / (right + left)
        // ```
        // and
        // ```text
        // m[3, 0] + 1 == -[(right - left) / (right + left)] + 1
        //             == [-(right - left) + (right + left)] / (right + left)
        //             == [-right + left + right + left] / (right + left)
        //             == (2 * left) / (right + left)
        // ```
        // so that
        // ```text
        // (2 * right) / (right + left) == -m[3, 0] + 1
        // (2 * left) / (right + left) == m[3, 0] + 1
        // ```
        // Define
        // ```text
        // ratio := right / left
        // ```
        // so that
        // ```text
        // ratio == right / left
        //       == (2 * right) / (2 * left)
        //       == [(2 * right) / (right + left)] / [(2 * left) / (right + left)]
        //       == (-m[3, 0] + 1) / (m[3, 0] + 1)
        // ```
        // and
        // ```text
        // left == right / ratio
        // ```
        // Using the matrix element `m[0, 0]`
        // ```text
        // m[0, 0] == 2 / (right + left)
        // ```
        // or equivalently
        // ```text
        // 2 * (1 / m[0, 0]) == right + left
        //                   == right + right / ratio
        //                   == (1 + (1 / ratio)) * right
        //                   == ((ratio + 1) / ratio) * right
        // ```
        // We obtain
        // ```text
        // right == 2 * (ratio / (ratio + 1)) * (1 / m[0, 0])
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[3][0] + one) / (self.matrix[3][0] + one);

        (two * (ratio / (ratio + one))) * (one / self.matrix[0][0])
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
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
        // The orthographic projection matrix has the form
        // ```text
        // [ m[0, 0]  0         0        m[3, 0] ]
        // [ 0        m[1, 1]   0        m[3, 1] ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0         0        1       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  2 / (right - (-left))
        // m[3, 0] := -(right + (-left)) / (right - (-left))
        // m[1, 1] :=  2 / (top - (-bottom))
        // m[3, 1] := -(top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -2 / (far - near)
        // m[3, 2] := -(far + near) / (far - near)
        // ```
        // Observe that
        // ```text
        // m[3, 0] - 1 == -[(right - left) / (right + left)] - 1
        //             == [-(right - left) - (right + left)] / (right + left)
        //             == [-right + left - right - left] / (right + left)
        //             == -(2 * right) / (right + left)
        // ```
        // and
        // ```text
        // m[3, 0] + 1 == -[(right - left) / (right + left)] + 1
        //             == [-(right - left) + (right + left)] / (right + left)
        //             == [-right + left + right + left] / (right + left)
        //             == (2 * left) / (right + left)
        // ```
        // so that
        // ```text
        // (2 * right) / (right + left) == -m[3, 0] + 1
        // (2 * left) / (right + left) == m[3, 0] + 1
        // ```
        // Define
        // ```text
        // ratio := right / left
        // ```
        // so that
        // ```text
        // ratio == right / left
        //       == (2 * right) / (2 * left)
        //       == [(2 * right) / (right + left)] / [(2 * left) / (right + left)]
        //       == (-m[3, 0] + 1) / (m[3, 0] + 1)
        // ```
        // and
        // ```text
        // right == ratio * left
        // ```
        // Using the matrix element `m[0, 0]`
        // ```text
        // m[0, 0] == 2 / (right + left)
        // ```
        // or equivalently
        // ```text
        // 2 * (1 / m[0, 0]) == right + left
        //                   == ratio * left + left
        //                   == (ratio + 1) * left
        // ```
        // We obtain
        // ```text
        // left == 2 * (1 / (ratio + 1)) * (1 / m[0, 0])
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[3][0] + one) / (self.matrix[3][0] + one);

        (two * (one / (ratio + one))) * (one / self.matrix[0][0])
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
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
        // The orthographic projection matrix has the form
        // ```text
        // [ m[0, 0]  0         0        m[3, 0] ]
        // [ 0        m[1, 1]   0        m[3, 1] ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0         0        1       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  2 / (right - (-left))
        // m[3, 0] := -(right + (-left)) / (right - (-left))
        // m[1, 1] :=  2 / (top - (-bottom))
        // m[3, 1] := -(top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -2 / (far - near)
        // m[3, 2] := -(far + near) / (far - near)
        // ```
        // Observe that
        // ```text
        // m[3, 1] - 1 == -[(top - bottom) / (top + bottom)] - 1
        //             == [-(top - bottom) - (top + bottom)] / (top + bottom)
        //             == [-top + bottom - top - bottom] / (top + bottom)
        //             == -(2 * top) / (top + bottom)
        // ```
        // and
        // ```text
        // m[3, 1] + 1 == -[(top - bottom) / (top + bottom)] + 1
        //             == [-(top - bottom) + (top + bottom)] / (top + bottom)
        //             == [-top + bottom + top + bottom] / (top + bottom)
        //             == (2 * bottom) / (top + bottom)
        // ```
        // so that
        // ```text
        // (2 * top) / (top + bottom) == -m[3, 1] + 1
        // (2 * bottom) / (top + bottom) == m[3, 1] + 1
        // ```
        // Define
        // ```text
        // ratio := top / bottom
        // ```
        // so that
        // ```text
        // ratio == top / bottom
        //       == (2 * top) / (2 * bottom)
        //       == [(2 * top) / (top + bottom)] / [(2 * bottom) / (top + bottom)]
        //       == (-m[3, 1] + 1) / (m[3, 1] + 1)
        // ```
        // and
        // ```text
        // bottom == top / ratio
        // ```
        // Using the matrix element `m[1, 1]`
        // ```text
        // m[1, 1] == 2 / (top + bottom)
        // ```
        // or equivalently
        // ```text
        // 2 * (1 / m[1, 1]) == top + bottom
        //                   == top + top / ratio
        //                   == ((ratio + 1) / ratio) * top
        // ```
        // We obtain
        // ```text
        // top == 2 * (ratio / (ratio + 1)) * (1 / m[1, 1])
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[3][1] + one) / (self.matrix[3][1] + one);

        (two * (ratio / (ratio + one))) * (one / self.matrix[1][1])
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
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
        // The orthographic projection matrix has the form
        // ```text
        // [ m[0, 0]  0         0        m[3, 0] ]
        // [ 0        m[1, 1]   0        m[3, 1] ]
        // [ 0        0         m[2, 2]  m[3, 2] ]
        // [ 0        0         0        1       ]
        // ```
        // where
        // ```text
        // m[0, 0] :=  2 / (right - (-left))
        // m[3, 0] := -(right + (-left)) / (right - (-left))
        // m[1, 1] :=  2 / (top - (-bottom))
        // m[3, 1] := -(top + (-bottom)) / (top - (-bottom))
        // m[2, 2] := -2 / (far - near)
        // m[3, 2] := -(far + near) / (far - near)
        // ```
        // Observe that
        // ```text
        // m[3, 1] - 1 == -[(top - bottom) / (top + bottom)] - 1
        //             == [-(top - bottom) - (top + bottom)] / (top + bottom)
        //             == [-top + bottom - top - bottom] / (top + bottom)
        //             == -(2 * top) / (top + bottom)
        // ```
        // and
        // ```text
        // m[3, 1] + 1 == -[(top - bottom) / (top + bottom)] + 1
        //             == [-(top - bottom) + (top + bottom)] / (top + bottom)
        //             == [-top + bottom + top + bottom] / (top + bottom)
        //             == (2 * bottom) / (top + bottom)
        // ```
        // so that
        // ```text
        // (2 * top) / (top + bottom) == -m[3, 1] + 1
        // (2 * bottom) / (top + bottom) == m[3, 1] + 1
        // ```
        // Define
        // ```text
        // ratio := top / bottom
        // ```
        // so that
        // ```text
        // ratio == top / bottom
        //       == (2 * top) / (2 * bottom)
        //       == [(2 * top) / (top + bottom)] / [(2 * bottom) / (top + bottom)]
        //       == (-m[3, 1] + 1) / (m[3, 1] + 1)
        // ```
        // and
        // ```text
        // top == ratio * bottom
        // ```
        // Using the matrix element `m[1, 1]`
        // ```text
        // m[1, 1] == 2 / (top + bottom)
        // ```
        // or equivalently
        // ```text
        // 2 * (1 / m[1, 1]) == top + bottom
        //                   == ratio * bottom + bottom
        //                   == (ratio + 1) * bottom
        // ```
        // We obtain
        // ```text
        // bottom == 2 * (1 / (ratio + 1)) * (1 / m[1, 1])
        // ```
        // which is the desired formula.
        //
        let one = S::one();
        let two = one + one;
        let ratio = (-self.matrix[3][1] + one) / (self.matrix[3][1] + one);

        (two * (one / (ratio + one))) * (one / self.matrix[1][1])
    }

    /// Get the underlying matrix implementing the orthographic transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_transform::Orthographic3;
    /// #
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
        // The perspective projection matrix has the form
        // ```text
        // [ c0r0 c1r0 c2r0 c3r0 ]    [ 2/(r + l)   0           0          -(r - l)/(r + l) ]
        // [ c0r1 c1r1 c2r1 c3r1 ] == [ 0           2/(t + b)   0          -(t - b)/(t + b) ]
        // [ c0r2 c1r2 c2r2 c3r2 ]    [ 0           0          -2/(f - n)  -(f + n)/(f - n) ]
        // [ c0r3 c1r3 c2r3 c3r3 ]    [ 0           0           0           1               ]
        // ```
        //
        // The inverse matrix of the perspective projection matrix has the form
        // ```text
        // [ c0r0 c1r0 c2r0 c3r0 ]    [ (r + l)/2   0           0           (r - l)/2 ]
        // [ c0r1 c1r1 c2r1 c3r1 ] == [ 0           (t + b)/2   0           (t - b)/2 ]
        // [ c0r2 c1r2 c2r2 c3r2 ]    [ 0           0          -(f - n)/2  -(f + n)/2 ]
        // [ c0r3 c1r3 c2r3 c3r3 ]    [ 0           0           0           1         ]
        // ```
        //
        // We can optimize the inverse calculation applying only
        // c0r0, c1r1, c2r2, c3r0, c3r1, and c3r2 to the input point.
        //
        let one = S::one();
        let one_half = one / (one + one);
        let c0r0 = one_half * (self.right() + self.left());
        let c1r1 = one_half * (self.top() + self.bottom());
        let c2r2 = -one_half * (self.far() - self.near());
        let c3r0 = one_half * (self.right() - self.left());
        let c3r1 = one_half * (self.top() - self.bottom());
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
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
        // The perspective projection matrix has the form
        // ```text
        // [ c0r0 c1r0 c2r0 c3r0 ]    [ 2/(r + l)   0           0          -(r - l)/(r + l) ]
        // [ c0r1 c1r1 c2r1 c3r1 ] == [ 0           2/(t + b)   0          -(t - b)/(t + b) ]
        // [ c0r2 c1r2 c2r2 c3r2 ]    [ 0           0          -2/(f - n)  -(f + n)/(f - n) ]
        // [ c0r3 c1r3 c2r3 c3r3 ]    [ 0           0           0           1               ]
        // ```
        //
        // The inverse matrix of the perspective projection matrix has the form
        // ```text
        // [ c0r0 c1r0 c2r0 c3r0 ]    [ (r + l)/2   0           0           (r - l)/2 ]
        // [ c0r1 c1r1 c2r1 c3r1 ] == [ 0           (t + b)/2   0           (t - b)/2 ]
        // [ c0r2 c1r2 c2r2 c3r2 ]    [ 0           0          -(f - n)/2  -(f + n)/2 ]
        // [ c0r3 c1r3 c2r3 c3r3 ]    [ 0           0           0           1         ]
        // ```
        //
        // We can optimize the inverse calculation applying only
        // c0r0, c1r1, and c2r2 to the input vector.
        //
        let one = S::one();
        let one_half = one / (one + one);
        let c0r0 = one_half * (self.right() + self.left());
        let c1r1 = one_half * (self.top() + self.bottom());
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
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
    /// let left = 6_f64;
    /// let right = 6_f64;
    /// let bottom = 4_f64;
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
