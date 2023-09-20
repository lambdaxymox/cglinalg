use cglinalg_numeric::{
    SimdScalarFloat,
};
use cglinalg_trigonometry::{
    Angle,
    Radians,
};
use cglinalg_core::{
    Const,
    ShapeConstraint,
    DimMul,
    Matrix,
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
    Point,
    Point3,
    Vector,
    Vector2,
    Vector3,
    Quaternion,
    Unit,
};
use crate::transform::{
    Transform2,
    Transform3,
};

use core::fmt;
use core::ops;


/// A rotation matrix in two dimensions.
/// 
/// Two-dimensional rotations are different than three-dimensional rotations 
/// because mathematically we cannot define an axis of rotation in two 
/// dimensions. Instead we have to talk about rotating in the **xy-plane** by an 
/// angle. In low-dimensional settings, the notion of rotation axis is 
/// only well-defined in three dimensions because dimension three is the 
/// only dimension where every plane is guaranteed to have a normal vector. 
/// 
/// If one wants to talk about rotating a vector in the the **xy-plane** about a 
/// normal vector, we are implicitly rotating about the **z-axis** in 
/// three dimensions.
pub type Rotation2<S> = Rotation<S, 2>;

/// A rotation matrix in three dimensions.
///
/// A rotation is an operation that creates circular motions and 
/// preserves at least one point. Rotations preserve the length of vectors and 
/// therefore act as a class of rigid body transformations.
pub type Rotation3<S> = Rotation<S, 3>;


/// A rotation matrix.
///
/// A rotation is an operation that creates circular motions and 
/// preserves at least one point. Rotations preserve the length of vectors and 
/// therefore act as a class of rigid body transformations.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rotation<S, const N: usize> {
    matrix: Matrix<S, N, N>,
}

impl<S, const N: usize> Rotation<S, N> 
where 
    S: SimdScalarFloat 
{
    /// Get a reference to the underlying matrix that represents the 
    /// rotation.
    #[inline]
    pub const fn matrix(&self) -> &Matrix<S, N, N> {
        &self.matrix
    }

    /// Construct a rotation that rotates a vector in the opposite direction 
    /// of `self`. 
    ///
    /// Given a rotation operator that rotates a vector about a normal vector 
    /// `axis` by an angle `theta`, construct a rotation that rotates a 
    /// vector about the same axis by an angle `-theta`.
    /// 
    /// # Examples
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,    
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation2,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_3);
    /// let rotation = Rotation2::from_angle(angle);
    /// let rotation_inv = rotation.inverse();
    /// let expected = Radians(-f64::consts::FRAC_PI_3);
    /// let result = rotation_inv.angle();
    ///
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use core::f64;
    /// # 
    /// let angle = Radians(f64::consts::FRAC_PI_3);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let expected = Rotation3::from_axis_angle(&axis, -angle);
    /// let result = rotation.inverse();
    ///
    /// assert_eq!(result, expected);
    /// assert_eq!(result.angle(), expected.angle());
    /// assert_eq!(result.axis(), expected.axis());
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        Self {
            matrix: self.matrix.transpose(),
        }
    }

    /// Mutably invert a rotation in place.
    ///
    /// # Examples
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation2,
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let mut result = Rotation2::from_angle(angle);
    /// let expected = Rotation2::from_angle(-angle);
    /// result.inverse_mut();
    ///
    /// assert_eq!(result, expected);
    /// ``` 
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use core::f64;
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let mut result = Rotation3::from_axis_angle(&axis, angle);
    /// let expected = Rotation3::from_axis_angle(&axis, -angle);
    /// result.inverse_mut();
    ///
    /// assert_eq!(result, expected);
    /// ``` 
    #[inline]
    pub fn inverse_mut(&mut self) {
        self.matrix.transpose_mut()
    }

    /// Apply the rotation operation to a vector.
    ///
    /// # Examples
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(-f64::consts::FRAC_PI_4);
    /// let rotation = Rotation2::from_angle(angle);
    /// let vector = Vector2::unit_x();
    /// let expected = Vector2::new(1_f64 / f64::sqrt(2_f64), -1_f64 / f64::sqrt(2_f64));
    /// let result = rotation.rotate_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(-f64::consts::FRAC_PI_2);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let vector = Vector3::unit_x();
    /// let expected = -Vector3::unit_y();
    /// let result = rotation.rotate_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn rotate_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        self.matrix * vector
    }

    /// Apply the rotation operation to a point.
    ///
    /// # Example
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(-f64::consts::FRAC_PI_4);
    /// let rotation = Rotation2::from_angle(angle);
    /// let point = Point2::new(1_f64, 0_f64);
    /// let expected = Point2::new(1_f64 / f64::sqrt(2_f64), -1_f64 / f64::sqrt(2_f64));
    /// let result = rotation.rotate_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(-f64::consts::FRAC_PI_2);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let point = Point3::new(1_f64, 0_f64, 0_f64);
    /// let expected = Point3::new(0_f64, -1_f64, 0_f64);
    /// let result = rotation.rotate_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn rotate_point(&self, point: &Point<S, N>) -> Point<S, N> { 
        let vector = point.to_vector();
        let result = self.matrix * vector;

        Point::from_vector(&result)
    }

    /// Apply the inverse of the rotation to a vector.
    ///
    /// # Example
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let rotation = Rotation2::from_angle(angle);
    /// let vector = Vector2::unit_x();
    /// let expected = Vector2::new(1_f64 / f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64));
    /// let result = rotation.rotate_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    ///
    /// let expected = Vector2::new(1_f64 / f64::sqrt(2_f64), -1_f64 / f64::sqrt(2_f64));
    /// let result = rotation.inverse_rotate_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(-f64::consts::FRAC_PI_2);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let vector = Vector3::unit_x();
    /// let expected = vector;
    /// let rotated_vector = rotation.rotate_vector(&vector);
    /// let result = rotation.inverse_rotate_vector(&rotated_vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```   
    #[inline]
    pub fn inverse_rotate_vector(&self, vector: &Vector<S, N>) -> Vector<S, N> {
        let inverse = self.inverse();
        
        inverse.matrix * vector
    }

    /// Apply the inverse of the rotation to a point.
    ///
    /// # Example
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Point2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let rotation = Rotation2::from_angle(angle);
    /// let point = Point2::new(1_f64, 0_f64);
    /// let expected = Point2::new(1_f64 / f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64));
    /// let result = rotation.rotate_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    ///
    /// let expected = Point2::new(1_f64 / f64::sqrt(2_f64), -1_f64 / f64::sqrt(2_f64));
    /// let result = rotation.inverse_rotate_point(&point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(-f64::consts::FRAC_PI_2);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let point = Point3::new(1_f64, 0_f64, 0_f64);
    /// let expected = point;
    /// let rotated_point = rotation.rotate_point(&point);
    /// let result = rotation.inverse_rotate_point(&rotated_point);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn inverse_rotate_point(&self, point: &Point<S, N>) -> Point<S, N> {
        let inverse = self.inverse();
        let vector = point.to_vector();
        let result = inverse.matrix * vector;

        Point::from_vector(&result)
    }

    /// Construct the identity rotation transformation.
    ///
    /// The identity rotation transformation is a rotation that rotates
    /// a vector or point by and angle of zero radians. The inverse operation
    /// will also rotate by zero radians.
    ///
    /// # Example
    ///
    /// An example in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation2,
    /// # };
    /// #
    /// let rotation = Rotation2::identity();
    /// let point = Point2::new(1_f64, 2_f64);
    ///
    /// assert_eq!(rotation * point, point);
    /// assert_eq!(rotation.inverse(), rotation);
    /// ```
    /// 
    /// An example in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Point3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// #
    /// let rotation = Rotation3::identity();
    /// let point = Point3::new(1_f64, 2_f64, 3_f64);
    ///
    /// assert_eq!(rotation * point, point);
    /// assert_eq!(rotation.inverse(), rotation);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self { 
            matrix: Matrix::identity(),
        }
    }
}

impl<S, const N: usize> fmt::Display for Rotation<S, N> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Rotation{} [{}]", N, self.matrix)
    }
}

impl<S, const N: usize> AsRef<Matrix<S, N, N>> for Rotation<S, N> {
    #[inline]
    fn as_ref(&self) -> &Matrix<S, N, N> {
        &self.matrix
    }
}

impl<S, const N: usize> approx::AbsDiffEq for Rotation<S, N> 
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
        Matrix::abs_diff_eq(&self.matrix, &other.matrix, epsilon)
    }
}

impl<S, const N: usize> approx::RelativeEq for Rotation<S, N> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        Matrix::relative_eq(&self.matrix, &other.matrix, epsilon, max_relative)
    }
}

impl<S, const N: usize> approx::UlpsEq for Rotation<S, N> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        Matrix::ulps_eq(&self.matrix, &other.matrix, epsilon, max_ulps)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for Rotation<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.rotate_point(&other)
    }
}

impl<S, const N: usize> ops::Mul<&Point<S, N>> for Rotation<S, N> 
where 
    S: SimdScalarFloat 
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &Point<S, N>) -> Self::Output {
        self.rotate_point(other)
    }
}

impl<S, const N: usize> ops::Mul<Point<S, N>> for &Rotation<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: Point<S, N>) -> Self::Output {
        self.rotate_point(&other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'a Point<S, N>> for &'b Rotation<S, N> 
where 
    S: SimdScalarFloat
{
    type Output = Point<S, N>;

    #[inline]
    fn mul(self, other: &'a Point<S, N>) -> Self::Output {
        self.rotate_point(other)
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<Rotation<S, N>> for Rotation<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Rotation<S, N>;

    #[inline]
    fn mul(self, other: Rotation<S, N>) -> Self::Output {
        Rotation {
            matrix: self.matrix() * other.matrix()
        }
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<&Rotation<S, N>> for Rotation<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Rotation<S, N>;

    #[inline]
    fn mul(self, other: &Rotation<S, N>) -> Self::Output {
        Rotation {
            matrix: self.matrix() * other.matrix()
        }
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<Rotation<S, N>> for &Rotation<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Rotation<S, N>;

    #[inline]
    fn mul(self, other: Rotation<S, N>) -> Self::Output {
        Rotation {
            matrix: self.matrix() * other.matrix()
        }
    }
}

impl<'a, 'b, S, const N: usize, const NN: usize> ops::Mul<&'a Rotation<S, N>> for &'b Rotation<S, N> 
where 
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Rotation<S, N>;

    #[inline]
    fn mul(self, other: &'a Rotation<S, N>) -> Self::Output {
        Rotation {
            matrix: self.matrix() * other.matrix()
        }
    }
}

impl<S> Rotation2<S> 
where 
    S: SimdScalarFloat 
{
    /// Get the rotation angle of the rotation transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let angle = Degrees(90_f64);
    /// let rotation = Rotation2::from_angle(angle);
    /// let expected = angle.into();
    /// let result = rotation.angle();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn angle(&self) -> Radians<S> {
        Radians::atan2(self.matrix.c0r1, self.matrix.c0r0)
    }

    /// Rotate a two-dimensional vector in the **xy-plane** by an angle `angle`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation2,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let angle = Degrees(90_f64);
    /// let rotation = Rotation2::from_angle(angle);
    /// let unit_x = Vector2::unit_x();
    /// let unit_y = Vector2::unit_y();
    ///
    /// assert_relative_eq!(
    ///     rotation.rotate_vector(&unit_x), unit_y, epsilon = 1e-8
    /// );
    /// ```
    #[inline]
    pub fn from_angle<A: Into<Radians<S>>>(angle: A) -> Self {  
        Self {
            matrix: Matrix2x2::from_angle(angle.into()),
        }
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between two unit vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2,
    /// #     Unit, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation2,
    /// # };
    /// #
    /// let point = Point2::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64);
    /// let vector1 = Unit::from_value(Vector2::unit_y());
    /// let vector2 = Unit::from_value(Vector2::unit_x());
    /// let rotation = Rotation2::rotation_between_axis(&vector1, &vector2);
    /// let expected = Point2::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
    /// let result = rotation.rotate_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn rotation_between_axis(a: &Unit<Vector2<S>>, b: &Unit<Vector2<S>>) -> Self {
        let unit_a = a.as_ref();
        let unit_b = b.as_ref();
        let cos_angle = unit_a.dot(unit_b);
        let sin_angle = unit_a.x * unit_b.y - unit_a.y * unit_b.x;

        Self::from_angle(Radians::atan2(sin_angle, cos_angle))
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Point2,
    /// #     Vector2, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation2,
    /// # };
    /// #
    /// let point = Point2::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64);
    /// let vector1 = 3_f64 * Vector2::unit_y();
    /// let vector2 = 6_f64 * Vector2::unit_x();
    /// let rotation = Rotation2::rotation_between(&vector1, &vector2);
    /// let expected = Point2::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64);
    /// let result = rotation.rotate_point(&point);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn rotation_between(a: &Vector2<S>, b: &Vector2<S>) -> Self {
        if let (Some(unit_a), Some(unit_b)) = (
            Unit::try_from_value(*a, S::zero()), 
            Unit::try_from_value(*b, S::zero()))
        {
            Self::rotation_between_axis(&unit_a, &unit_b)
        } else {
            Self::identity()
        }
    }

    /// Convert a rotation into a generic transformation.
    #[inline]
    pub fn to_transform(&self) -> Transform2<S> {
        Transform2::from_specialized(self.matrix)
    }

    /// Convert a rotation into a generic affine matrix.
    #[rustfmt::skip]
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix3x3<S> {
        let zero = S::zero();
        let one = S::one();

        Matrix3x3::new(
            self.matrix.c0r0, self.matrix.c0r1, zero,
            self.matrix.c1r0, self.matrix.c1r1, zero,
            zero,             zero,             one
        )
    }
}

impl<S> From<Rotation2<S>> for Matrix2x2<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn from(rotation: Rotation2<S>) -> Matrix2x2<S> {
        rotation.matrix
    }
}

impl<S> From<Rotation2<S>> for Matrix3x3<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn from(rotation: Rotation2<S>) -> Matrix3x3<S> {
        Matrix3x3::from(&rotation.matrix)
    }
}


impl<S> Rotation3<S> 
where 
    S: SimdScalarFloat 
{
    /// Get the rotation angle of the rotation transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let angle = Degrees(90_f64);
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let expected = angle.into();
    /// let result = rotation.angle();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn angle(&self) -> Radians<S> {
        let two = cglinalg_numeric::cast(2);
        Radians::acos((
            self.matrix.c0r0 + self.matrix.c1r1 + self.matrix.c2r2 - S::one()) / two
        )
    }

    /// Compute the axis of the rotation if it exists.
    ///
    /// If the rotation angle is zero or `pi`, axis returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let angle = Degrees(90_f64);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let expected = Some(axis);
    /// let result = rotation.axis();
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// If the rotation angle is zero, the function returns `None`.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let angle = Degrees(0_f64);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let expected = None;
    /// let result = rotation.axis();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn axis(&self) -> Option<Unit<Vector3<S>>> {
        let axis = Vector3::new(
            self.matrix.c1r2 - self.matrix.c2r1,
            self.matrix.c2r0 - self.matrix.c0r2,
            self.matrix.c0r1 - self.matrix.c1r0
        );

        Unit::try_from_value(axis, S::default_epsilon())
    }

    /// Compute the axis and angle of the rotation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_6);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// let expected = Some((axis, angle));
    /// let result = rotation.axis_angle();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn axis_angle(&self) -> Option<(Unit<Vector3<S>>, Radians<S>)> {
        self.axis().map(|axis| (axis, self.angle()))
    }

    /// Construct a three-dimensional rotation matrix from a quaternion.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Quaternion,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// # use core::f64;
    /// #
    /// let angle = Radians(f64::consts::FRAC_PI_4);
    /// let axis = Unit::from_value(Vector3::unit_y());
    /// let quaternion = Quaternion::from_axis_angle(&axis, angle);
    /// let expected = Rotation3::from_axis_angle(&axis, angle);
    /// let result = Rotation3::from_quaternion(&quaternion);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn from_quaternion(quaternion: &Quaternion<S>) -> Self {
        Self {
            matrix: quaternion.to_matrix3x3(),
        }
    }

    /// Construct a new three-dimensional rotation about an axis `axis` by 
    /// an angle `angle`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let rotation = Rotation3::from_axis_angle(&axis, angle);
    /// 
    /// let expected_axis_angle = Some((axis, angle));
    /// assert_eq!(rotation.axis_angle(), expected_axis_angle);
    /// 
    /// // Rotate a vector ninety degrees.
    /// let unit_x = Vector3::unit_x();
    /// let expected = Vector3::unit_y();
    /// let result = rotation.rotate_vector(&unit_x);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn from_axis_angle<A: Into<Radians<S>>>(axis: &Unit<Vector3<S>>, angle: A) -> Self {
        Self {
            matrix: Matrix3x3::from_axis_angle(axis, angle.into()),
        }
    }

    /// Construct a new three-dimensional rotation about the **x-axis** in the 
    /// **yz-plane** by an angle `angle`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use core::f64;
    /// # 
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let axis = Unit::from_value(Vector3::unit_x());
    /// let rotation = Rotation3::from_angle_x(angle);
    /// let expected = Some(axis);
    /// let result = rotation.axis();
    ///
    /// assert_eq!(rotation.angle(), angle);
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_angle_x<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_axis_angle(&Unit::from_value_unchecked(Vector3::unit_x()), angle)
    }

    /// Construct a new three-dimensional rotation about the **y-axis** in the 
    /// **zx-plane** by an angle `angle`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use core::f64;
    /// # 
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let axis = Unit::from_value(Vector3::unit_y());
    /// let rotation = Rotation3::from_angle_y(angle);
    /// let expected = Some(axis);
    /// let result = rotation.axis();
    ///
    /// assert_eq!(rotation.angle(), angle);
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_angle_y<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_axis_angle(&Unit::from_value_unchecked(Vector3::unit_y()), angle)
    }

    /// Construct a new three-dimensional rotation about the **z-axis** in the 
    /// **xy-plane** by an angle `angle`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use core::f64;
    /// # 
    /// let angle = Radians(f64::consts::FRAC_PI_2);
    /// let axis = Unit::from_value(Vector3::unit_z());
    /// let rotation = Rotation3::from_angle_z(angle);
    /// let expected = Some(axis);
    /// let result = rotation.axis();
    ///
    /// assert_eq!(rotation.angle(), angle);
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_angle_z<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_axis_angle(&Unit::from_value_unchecked(Vector3::unit_z()), angle)
    }

    /// Construct a coordinate transformation that maps the coordinate system 
    /// of an observer located at the origin facing the direction `direction` 
    /// into a coordinate system of an observer located at the origin facing 
    /// the **positive z-axis**. The resulting coordinate transformation is a 
    /// **left-handed** coordinate transformation.
    ///
    /// The resulting transformation maps `direction` to the **positive z-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,    
    /// # };
    /// # use core::f64;
    /// #
    /// let direction = Vector3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let rotation = Rotation3::look_to_lh(&direction, &up);
    /// let unit_z = Vector3::unit_z();
    ///
    /// assert_relative_eq!(
    ///     rotation.rotate_vector(&direction).normalize(), 
    ///     unit_z, 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_to_lh(direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        Self {
            matrix: Matrix3x3::look_to_lh(direction, up),
        }
    }

    /// Construct a coordinate transformation that maps the coordinate system 
    /// of an observer located at the origin facing the direction `direction` 
    /// into a coordinate system of an observer located at the origin facing 
    /// the **negative z-axis**. The resulting coordinate transformation is a 
    /// **right-handed** coordinate transformation.
    ///
    /// The resulting transformation maps `direction` to the **negative z-axis**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,    
    /// # };
    /// # use core::f64;
    /// #
    /// let direction = Vector3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let rotation = Rotation3::look_to_rh(&direction, &up);
    /// let minus_unit_z = -Vector3::unit_z();
    ///
    /// assert_relative_eq!(
    ///     rotation.rotate_vector(&direction).normalize(), 
    ///     minus_unit_z, 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_to_rh(direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        Self {
            matrix: Matrix3x3::look_to_rh(direction, up),
        }
    }

    /// Construct a coordinate transformation that transforms
    /// a coordinate system of an observer located at the position `eye` facing 
    /// the direction `direction` into the coordinate system of an observer located
    /// at the origin facing the **positive z-axis**.
    ///
    /// The function maps the direction `direction` to the **positive z-axis** and 
    /// locates the `eye` position to the origin in the new the coordinate system.
    /// This transformation is a **left-handed** coordinate transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # 
    /// let eye = Point3::new(0_f64, -5_f64, 0_f64);
    /// let target = Point3::origin();
    /// let up: Vector3<f64> = Vector3::unit_x();
    /// let direction = target - eye;
    /// let rotation = Rotation3::look_at_lh(&eye, &target, &up);
    /// let result = rotation.rotate_vector(&direction).normalize();
    /// let expected = Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn look_at_lh(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        Self {
            matrix: Matrix3x3::look_at_lh(eye, target, up),
        }
    }

    /// Construct a coordinate transformation that transforms
    /// a coordinate system of an observer located at the position `eye` facing 
    /// the direction `direction` into the coordinate system of an observer located
    /// at the origin facing the **negative z-axis**.
    ///
    /// The function maps the direction `direction` to the **negative z-axis** and 
    /// locates the `eye` position to the origin in the new the coordinate system.
    /// This transformation is a **right-handed** coordinate transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// # 
    /// let eye = Point3::new(0_f64, -5_f64, 0_f64);
    /// let target = Point3::origin();
    /// let up: Vector3<f64> = Vector3::unit_x();
    /// let direction = target - eye;
    /// let rotation = Rotation3::look_at_rh(&eye, &target, &up);
    /// let result = rotation.rotate_vector(&direction).normalize();
    /// let expected = -Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn look_at_rh(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        Self {
            matrix: Matrix3x3::look_at_rh(eye, target, up),
        }
    }

    /// Construct a coordinate transformation that maps the coordinate system 
    /// of an observer located at the origin facing the **positive z-axis** into a 
    /// coordinate system of an observer located at the position origin facing 
    /// the direction `direction`. The resulting coordinate transformation is a 
    /// **left-handed** coordinate transformation.
    ///
    /// The resulting transformation maps the **positive z-axis** to `direction`.
    /// This function is the inverse of [`look_to_lh`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,    
    /// # };
    /// # use core::f64;
    /// #
    /// let direction = Vector3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let rotation = Rotation3::look_to_lh_inv(&direction, &up);
    /// let unit_z = Vector3::unit_z();
    ///
    /// assert_relative_eq!(
    ///     rotation.rotate_vector(&unit_z), 
    ///     direction.normalize(), 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_to_lh_inv(direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        Self {
            matrix: Matrix3x3::look_to_lh_inv(direction, up),
        }
    }

    /// Construct a coordinate transformation that maps the coordinate system 
    /// of an observer located at the origin facing the **negative z-axis** into a 
    /// coordinate system of an observer located at the position origin facing 
    /// the direction `direction`. The resulting coordinate transformation is a 
    /// **right-handed** coordinate transformation.
    ///
    /// The resulting transformation maps the **negative z-axis** to `direction`.
    /// This function is the inverse of [`look_to_rh`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,    
    /// # };
    /// # use core::f64;
    /// #
    /// let direction = Vector3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let rotation = Rotation3::look_to_rh_inv(&direction, &up);
    /// let minus_unit_z = -Vector3::unit_z();
    ///
    /// assert_relative_eq!(
    ///     rotation.rotate_vector(&minus_unit_z), 
    ///     direction.normalize(), 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_to_rh_inv(direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        Self {
            matrix: Matrix3x3::look_to_rh_inv(direction, up),
        }
    }

    /// Construct a coordinate transformation that maps the coordinate system 
    /// of an observer located at the origin facing the **positive z-axis** into a 
    /// coordinate system of an observer located at the position origin facing 
    /// the direction `target - eye`. The resulting coordinate transformation is a 
    /// **left-handed** coordinate transformation.
    ///
    /// The resulting transformation maps the **positive z-axis** to `target - eye`.
    /// This function is the inverse of [`look_at_lh`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,    
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(-1_f64, 1_f64, -1_f64);
    /// let target = Point3::origin();
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let rotation = Rotation3::look_at_lh_inv(&eye, &target, &up);
    /// let direction = target - eye;
    /// let unit_z = Vector3::unit_z();
    ///
    /// assert_relative_eq!(
    ///     rotation.rotate_vector(&unit_z), 
    ///     direction.normalize(), 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_at_lh_inv(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        Self {
            matrix: Matrix3x3::look_at_lh_inv(eye, target, up),
        }
    }

    /// Construct a coordinate transformation that maps the coordinate system 
    /// of an observer located at the origin facing the **negative z-axis** into a 
    /// coordinate system of an observer located at the position origin facing 
    /// the direction `target - eye`. The resulting coordinate transformation is a 
    /// **right-handed** coordinate transformation.
    ///
    /// The resulting transformation maps the **negative z-axis** to `target - eye`.
    /// This function is the inverse of [`look_at_rh`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,    
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(-1_f64, 1_f64, -1_f64);
    /// let target = Point3::origin();
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let rotation = Rotation3::look_at_rh_inv(&eye, &target, &up);
    /// let direction = target - eye;
    /// let minus_unit_z = -Vector3::unit_z();
    ///
    /// assert_relative_eq!(
    ///     rotation.rotate_vector(&minus_unit_z), 
    ///     direction.normalize(), 
    ///     epsilon = 1e-10,
    /// );
    /// ```
    #[inline]
    pub fn look_at_rh_inv(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        Self {
            matrix: Matrix3x3::look_at_rh_inv(eye, target, up),
        }
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between two vectors.
    ///
    /// The rotation uses the unit directional vectors of the input vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3, 
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let vector = 3_f64 * Vector3::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64, 0_f64);
    /// let vector1 = 5_f64 * Vector3::unit_y();
    /// let vector2 = 12_f64 * Vector3::unit_x();
    /// let rotation = Rotation3::rotation_between(&vector1, &vector2).unwrap();
    /// let expected = 3_f64 * Vector3::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64, 0_f64);
    /// let result = rotation.rotate_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn rotation_between(v1: &Vector3<S>, v2: &Vector3<S>) -> Option<Rotation3<S>> {
        Quaternion::rotation_between(v1, v2).map(|q| q.into())
    }

    /// Construct a rotation that rotates the shortest angular distance 
    /// between two unit vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use cglinalg_transform::{
    /// #     Rotation3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let vector = 3_f64 * Vector3::new(f64::sqrt(3_f64) / 2_f64, 1_f64 / 2_f64, 0_f64);
    /// let unit1 = Unit::from_value(5_f64 * Vector3::unit_y());
    /// let unit2 = Unit::from_value(12_f64 * Vector3::unit_x());
    /// let rotation = Rotation3::rotation_between_axis(&unit1, &unit2).unwrap();
    /// let expected = 3_f64 * Vector3::new(1_f64 / 2_f64, -f64::sqrt(3_f64) / 2_f64, 0_f64);
    /// let result = rotation.rotate_vector(&vector);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn rotation_between_axis(v1: &Unit<Vector3<S>>, v2: &Unit<Vector3<S>>) -> Option<Self> {
        Quaternion::rotation_between_axis(v1, v2).map(|q| q.into())
    }

    /// Convert a rotation to a generic transformation.
    #[inline]
    pub fn to_transform(&self) -> Transform3<S> {
        Transform3::from_specialized(self.matrix)
    }

    /// Convert a rotation into a generic affine matrix.
    #[rustfmt::skip]
    #[inline]
    pub fn to_affine_matrix(&self) -> Matrix4x4<S> {
        let zero = S::zero();
        let one = S::one();

        Matrix4x4::new(
            self.matrix.c0r0, self.matrix.c0r1, self.matrix.c0r2, zero,
            self.matrix.c1r0, self.matrix.c1r1, self.matrix.c1r2, zero,
            self.matrix.c2r0, self.matrix.c2r1, self.matrix.c2r2, zero,
            zero,             zero,             zero,             one
        )
    }
}

impl<S> From<Rotation3<S>> for Matrix4x4<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn from(rotation: Rotation3<S>) -> Matrix4x4<S> {
        Matrix4x4::from(&rotation.matrix)
    }
}

impl<S> From<Quaternion<S>> for Rotation3<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn from(quaternion: Quaternion<S>) -> Rotation3<S> {
        Rotation3::from_quaternion(&quaternion)
    }
}

impl<S> From<Rotation3<S>> for Quaternion<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn from(rotation: Rotation3<S>) -> Quaternion<S> {
        let matrix = Matrix3x3::new(
            rotation.matrix.c0r0, rotation.matrix.c0r1, rotation.matrix.c0r2,
            rotation.matrix.c1r0, rotation.matrix.c1r1, rotation.matrix.c1r2,
            rotation.matrix.c2r0, rotation.matrix.c2r1, rotation.matrix.c2r2
        );
        Quaternion::from(&matrix)
    }
}

