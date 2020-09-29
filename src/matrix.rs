use approx::{
    ulps_eq,
};
use num_traits::{
    NumCast,
};
use crate::point::{
    Point3,
};
use crate::scalar::{
    Scalar,
    ScalarSigned,
    ScalarFloat,
};
use crate::angle::{
    Angle,
    Radians,
};
use crate::traits::{
    Array,
    CrossProduct,
    DotProduct,
    Identity, 
    AdditiveIdentity, 
    Matrix, 
    SquareMatrix,
    InvertibleSquareMatrix,
    Magnitude,
};
use crate::vector::{
    Vector2,
    Vector3,
    Vector4,
};
use crate::unit::{
    Unit,
};

use core::fmt;
use core::ops;
use core::iter;


macro_rules! impl_mul_operator {
    ($Lhs:ty, $Rhs:ty, $Output:ty, { $($field:ident),* }) => {
        impl ops::Mul<$Rhs> for $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                <$Output>::new( $(self * other.$field),*)
            }
        }

        impl<'a> ops::Mul<$Rhs> for &'a $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                <$Output>::new( $(self * other.$field),*)
            }
        }
    }
}


// Type synonyms for common programming matrix naming conventions.
pub type Matrix2<S> = Matrix2x2<S>;
pub type Matrix3<S> = Matrix3x3<S>;
pub type Matrix4<S> = Matrix4x4<S>;


/// The `Matrix2x2` type represents 2x2 matrices in column-major order.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C)]
pub struct Matrix2x2<S> {
    /// Column 0 of the matrix.
    pub c0r0: S, pub c0r1: S,
    /// Column 1 of the matrix.
    pub c1r0: S, pub c1r1: S,
}

impl<S> Matrix2x2<S> {
    /// Construct a new 2x2 matrix from its field elements.
    #[inline]
    pub const fn new(c0r0: S, c0r1: S, c1r0: S, c1r1: S) -> Matrix2x2<S> {
        Matrix2x2 { 
            c0r0: c0r0, 
            c0r1: c0r1, 
            c1r0: c1r0, 
            c1r1: c1r1 
        }
    }

    /// Construct a 2x2 matrix from a pair of two-dimensional vectors.
    #[inline]
    pub fn from_columns(c0: Vector2<S>, c1: Vector2<S>) -> Matrix2x2<S> {
        Matrix2x2 { 
            c0r0: c0.x, 
            c0r1: c0.y, 
            c1r0: c1.x, 
            c1r1: c1.y 
        }
    }

    /// Map an operation on the elements of a matrix, returning a matrix whose 
    /// elements are elements of the new underlying type.
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Matrix2x2<T> where F: FnMut(S) -> T {
        Matrix2x2 {
            c0r0: op(self.c0r0), 
            c1r0: op(self.c1r0),
            c0r1: op(self.c0r1), 
            c1r1: op(self.c1r1),
        }
    }
}

impl<S> Matrix2x2<S> where S: Copy {
    /// Construct a new matrix from a fill value.
    ///
    /// The resulting matrix is a matrix where each entry is the supplied fill
    /// value.
    #[inline]
    pub fn from_fill(value: S) -> Matrix2x2<S> {
        Matrix2x2::new(value, value, value, value)
    }
}

impl<S> Matrix2x2<S> where S: NumCast + Copy {
    /// Cast a matrix from one type of scalars to another type of scalars.
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Matrix2x2<T>> {
        let c0r0 = match num_traits::cast(self.c0r0) {
            Some(value) => value,
            None => return None,
        };
        let c0r1 = match num_traits::cast(self.c0r1) {
            Some(value) => value,
            None => return None,
        };
        let c1r0 = match num_traits::cast(self.c1r0) {
            Some(value) => value,
            None => return None,
        };
        let c1r1 = match num_traits::cast(self.c1r1) {
            Some(value) => value,
            None => return None,
        };

        Some(Matrix2x2::new(c0r0, c0r1, c1r0, c1r1))
    }
}

impl<S> Matrix2x2<S> where S: Scalar {
    /// Construct a matrix that will cause a vector to point 
    /// at the vector `direction` using up for orientation.
    #[inline]
    pub fn look_at(direction: &Vector2<S>, up: &Vector2<S>) -> Matrix2x2<S> {
        Matrix2x2::from_columns(*up, *direction).transpose()
    }

    /// Construct a shearing matrix along the x-axis, holding the y-axis constant.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the _y-axis_ to shearing along the _x-axis_.
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S) -> Matrix2x2<S> {
        Matrix2x2::new(
            S::one(),       S::zero(),
            shear_x_with_y, S::one(),
        )
    }

    /// Construct a shearing matrix along the y-axis, holding the x-axis constant.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the _x-axis_ to shearing along the _y-axis_.
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_y(shear_y_with_x: S) -> Matrix2x2<S> {
        Matrix2x2::new(
            S::one(),  shear_y_with_x,
            S::zero(), S::one(),
        )
    }
    
    /// Construct a general shearing matrix in two dimensions. There are two 
    /// possible parameters describing a shearing transformation in two 
    /// dimensions.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the _x-axis_ to shearing along the _y-axis_.
    /// The parameter `shear_x_with_y` denotes the factor scaling the 
    /// contribution of the y-component to the shearing of the x-component. 
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear(shear_x_with_y: S, shear_y_with_x: S) -> Matrix2x2<S> {
        let one = S::one();

        Matrix2x2::new(
            one,            shear_y_with_x,
            shear_x_with_y, one
        )
    }

    /// Construct a two-dimensional uniform scaling matrix.
    ///
    /// The matrix applies the same scale factor to all dimensions, so each
    /// component of a vector will be scaled by the same factor. In particular,
    /// calling `from_scale(scale)` is equivalent to calling 
    /// `from_nonuniform_scale(scale, scale)`.
    #[inline]
    pub fn from_scale(scale: S) -> Matrix2x2<S> {
        Matrix2x2::from_nonuniform_scale(scale, scale)
    }
        
    /// Construct two-dimensional general scaling matrix.
    ///
    /// This is the most general case for scaling matrices: the scale factor
    /// in each dimension need not be identical.
    #[rustfmt::skip]
    #[inline]
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S) -> Matrix2x2<S> {
        let zero = S::zero();
        Matrix2x2::new(
            scale_x,   zero,
            zero,      scale_y,
        )
    }
}

impl<S> Matrix2x2<S> where S: ScalarSigned {
    /// Construct a two-dimensional reflection matrix for reflecting through a 
    /// line through the origin in the xy-plane.
    #[rustfmt::skip]
    #[inline]
    pub fn from_reflection(normal: &Unit<Vector2<S>>) -> Matrix2x2<S> {
        let one = S::one();
        let two = one + one;

        Matrix2x2::new(
             one - two * normal.x * normal.x, -two * normal.x * normal.y,
            -two * normal.x * normal.y,        one - two * normal.y * normal.y,
        )
    }
}

impl<S> Matrix2x2<S> where S: ScalarFloat {
    /// Construct a rotation matrix in two-dimensions that rotates a vector
    /// in the xy-plane by an angle `angle`.
    #[rustfmt::skip]
    #[inline]
    pub fn from_angle<A: Into<Radians<S>>>(angle: A) -> Matrix2x2<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Matrix2x2::new(
             cos_angle, sin_angle, 
            -sin_angle, cos_angle
        )
    }

    /// Construct a rotation matrix that rotates the shortest angular distance 
    /// between two vectors.
    #[inline]
    pub fn rotation_between(v1: &Vector2<S>, v2: &Vector2<S>) -> Matrix2x2<S> {
        if let (Some(unit_v1), Some(unit_v2)) = (
            Unit::try_from_value(*v1, S::zero()),
            Unit::try_from_value(*v2, S::zero()),
        ) {
            Self::rotation_between_axis(&unit_v1, &unit_v2)
        } else {
            <Self as SquareMatrix>::identity()
        }
    }

    /// Construct a rotation matrix that rotates the shortest angular distance 
    /// between two unit vectors.
    #[inline]
    pub fn rotation_between_axis(v1: &Unit<Vector2<S>>, v2: &Unit<Vector2<S>>) -> Matrix2x2<S> {
        let cos_angle = v1.as_ref().dot(v2.as_ref());
        let sin_angle = S::sqrt(S::one() - cos_angle * cos_angle);

        Self::from_angle(Radians::atan2(sin_angle, cos_angle))
    }

    /// Linearly interpolate between two matrices.
    #[inline]
    pub fn lerp(&self, other: &Matrix2x2<S>, amount: S) -> Matrix2x2<S> {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of a matrix are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A matrix is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values. For example, when the vector elements are `f64`, the vector is 
    /// finite when the elements are neither `NaN` nor infinite.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.c0r0.is_finite() && self.c0r1.is_finite() &&
        self.c1r0.is_finite() && self.c1r1.is_finite()
    }
}

impl<S> Array for Matrix2x2<S> where S: Copy {
    type Element = S;

    #[inline]
    fn len() -> usize {
        4
    }

    #[inline]
    fn shape() -> (usize, usize) {
        (2, 2)
    }

    #[inline]
    fn as_ptr(&self) -> *const S {
        &self.c0r0
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut S {
        &mut self.c0r0
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Element] {
        <Self as AsRef<[Self::Element; 4]>>::as_ref(self)
    }
}

impl<S> Matrix for Matrix2x2<S> where S: Scalar {
    type Element = S;
    type Row = Vector2<S>;
    type Column = Vector2<S>;
    type Transpose = Matrix2x2<S>;

    #[inline]
    fn row(&self, r: usize) -> Self::Row {
        Vector2::new(self[0][r], self[1][r])
    }
    
    #[inline]
    fn swap_rows(&mut self, row_a: usize, row_b: usize) {
        let c0ra = self[0][row_a];
        let c1ra = self[1][row_a];
        self[0][row_a] = self[0][row_b];
        self[1][row_a] = self[1][row_b];
        self[0][row_b] = c0ra;
        self[1][row_b] = c1ra;
    }
    
    #[inline]
    fn swap_columns(&mut self, col_a: usize, col_b: usize) {
        let car0 = self[col_a][0];
        let car1 = self[col_a][1];
        self[col_a][0] = self[col_b][0];
        self[col_a][1] = self[col_b][1];
        self[col_b][0] = car0;
        self[col_b][1] = car1;
    }
    
    #[inline]
    fn swap_elements(&mut self, a: (usize, usize), b: (usize, usize)) {
        let element_a = self[a.0][a.1];
        self[a.0][a.1] = self[b.0][b.1];
        self[b.0][b.1] = element_a;
    }
    
    #[inline]
    fn transpose(&self) -> Self::Transpose {
        Matrix2x2::new(self.c0r0, self.c1r0, self.c0r1, self.c1r1)
    }
}

impl<S> From<[[S; 2]; 2]> for Matrix2x2<S> where S: Scalar {
    #[inline]
    fn from(array: [[S; 2]; 2]) -> Matrix2x2<S> {
        Matrix2x2::new(array[0][0], array[0][1], array[1][0], array[1][1])
    }
}

impl<'a, S> From<&'a [[S; 2]; 2]> for &'a Matrix2x2<S> where S: Scalar {
    #[inline]
    fn from(array: &'a [[S; 2]; 2]) -> &'a Matrix2x2<S> {
        unsafe { 
            &*(array as *const [[S; 2]; 2] as *const Matrix2x2<S>)
        }
    }    
}

impl<S> From<[S; 4]> for Matrix2x2<S> where S: Scalar {
    #[inline]
    fn from(array: [S; 4]) -> Matrix2x2<S> {
        Matrix2x2::new(array[0], array[1], array[2], array[3])
    }
}

impl<'a, S> From<&'a [S; 4]> for &'a Matrix2x2<S> where S: Scalar {
    #[inline]
    fn from(array: &'a [S; 4]) -> &'a Matrix2x2<S> {
        unsafe { 
            &*(array as *const [S; 4] as *const Matrix2x2<S>)
        }
    }
}

impl<S> fmt::Display for Matrix2x2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // We print the matrix contents in row-major order like mathematical convention.
        writeln!(
            formatter, 
            "Matrix2x2 [[{}, {}], [{}, {}]]", 
            self.c0r0, self.c1r0,
            self.c0r1, self.c1r1,
        )
    }
}

impl<S> AsRef<[S; 4]> for Matrix2x2<S> {
    #[inline]
    fn as_ref(&self) -> &[S; 4] {
        unsafe { 
            &*(self as *const Matrix2x2<S> as *const [S; 4])
        }
    }
}

impl<S> AsRef<[[S; 2]; 2]> for Matrix2x2<S> {
    #[inline]
    fn as_ref(&self) -> &[[S; 2]; 2] {
        unsafe { 
            &*(self as *const Matrix2x2<S> as *const [[S; 2]; 2])
        }
    }
}

impl<S> AsRef<[Vector2<S>; 2]> for Matrix2x2<S> {
    #[inline]
    fn as_ref(&self) -> &[Vector2<S>; 2] {
        unsafe { 
            &*(self as *const Matrix2x2<S> as *const [Vector2<S>; 2])
        }
    }
}

impl<S> AsMut<[S; 4]> for Matrix2x2<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; 4] {
        unsafe { 
            &mut *(self as *mut Matrix2x2<S> as *mut [S; 4])
        }
    }
}

impl<S> AsMut<[[S; 2]; 2]> for Matrix2x2<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [[S; 2]; 2] {
        unsafe { 
            &mut *(self as *mut Matrix2x2<S> as *mut [[S; 2]; 2])
        }
    }
}

impl<S> AsMut<[Vector2<S>; 2]> for Matrix2x2<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [Vector2<S>; 2] {
        unsafe { 
            &mut *(self as *mut Matrix2x2<S> as *mut [Vector2<S>; 2])
        }
    }
}

impl<S> ops::Index<usize> for Matrix2x2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[Vector2<S>; 2] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::IndexMut<usize> for Matrix2x2<S> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Vector2<S> {
        let v: &mut [Vector2<S>; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> AdditiveIdentity for Matrix2x2<S> where S: Scalar {
    #[inline]
    fn zero() -> Matrix2x2<S> {
        Matrix2x2::new(S::zero(), S::zero(), S::zero(), S::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.c0r0 == S::zero() && self.c0r1 == S::zero() &&
        self.c1r0 == S::zero() && self.c1r1 == S::zero()
    }
}

impl<S> Identity for Matrix2x2<S> where S: Scalar {
    #[inline]
    fn identity() -> Matrix2x2<S> {
        Matrix2x2::new(S::one(), S::zero(), S::zero(), S::one())
    }
}

impl<S> ops::Add<Matrix2x2<S>> for Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn add(self, other: Matrix2x2<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Add<&Matrix2x2<S>> for Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn add(self, other: &Matrix2x2<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Add<Matrix2x2<S>> for &Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn add(self, other: Matrix2x2<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a, 'b, S> ops::Add<&'a Matrix2x2<S>> for &'b Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn add(self, other: &'a Matrix2x2<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Sub<Matrix2x2<S>> for Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn sub(self, other: Matrix2x2<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Sub<&Matrix2x2<S>> for Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn sub(self, other: &Matrix2x2<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Sub<Matrix2x2<S>> for &Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn sub(self, other: Matrix2x2<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a, 'b, S> ops::Sub<&'a Matrix2x2<S>> for &'b Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn sub(self, other: &'a Matrix2x2<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Mul<&Matrix2x2<S>> for Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn mul(self, other: &Matrix2x2<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1;
        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Matrix2x2<S>> for &'b Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn mul(self, other: &'a Matrix2x2<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1;
        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Mul<Matrix2x2<S>> for Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn mul(self, other: Matrix2x2<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1;
        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Mul<Matrix2x2<S>> for &Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn mul(self, other: Matrix2x2<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1;
        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Mul<S> for Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Mul<Vector2<S>> for Matrix2x2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, other: Vector2<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y;
        let y = self.c0r1 * other.x + self.c1r1 * other.y;

        Vector2::new(x, y)
    }
}

impl<S> ops::Mul<&Vector2<S>> for Matrix2x2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, other: &Vector2<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y;
        let y = self.c0r1 * other.x + self.c1r1 * other.y;

        Vector2::new(x, y)
    }
}

impl<S> ops::Mul<Vector2<S>> for &Matrix2x2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, other: Vector2<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y;
        let y = self.c0r1 * other.x + self.c1r1 * other.y;

        Vector2::new(x, y)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Vector2<S>> for &'b Matrix2x2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, other: &'a Vector2<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y;
        let y = self.c0r1 * other.x + self.c1r1 * other.y;

        Vector2::new(x, y)
    }
}

impl<S> ops::Mul<S> for &Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Div<S> for Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Div<S> for &Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Neg for Matrix2x2<S> where S: ScalarSigned {
    type Output = Matrix2x2<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        let c0r0 = -self.c0r0;
        let c0r1 = -self.c0r1;
        let c1r0 = -self.c1r0;
        let c1r1 = -self.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Neg for &Matrix2x2<S> where S: ScalarSigned {
    type Output = Matrix2x2<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        let c0r0 = -self.c0r0;
        let c0r1 = -self.c0r1;
        let c1r0 = -self.c1r0;
        let c1r1 = -self.c1r1;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Rem<S> for Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 % other;
        let c0r1 = self.c0r1 % other;
        let c1r0 = self.c1r0 % other;
        let c1r1 = self.c1r1 % other;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Rem<S> for &Matrix2x2<S> where S: Scalar {
    type Output = Matrix2x2<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 % other;
        let c0r1 = self.c0r1 % other;
        let c1r0 = self.c1r0 % other;
        let c1r1 = self.c1r1 % other;

        Matrix2x2::new(c0r0, c0r1, c1r0, c1r1)        
    }
}

impl<S> ops::AddAssign<Matrix2x2<S>> for Matrix2x2<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: Matrix2x2<S>) {
        self.c0r0 += other.c0r0;
        self.c0r1 += other.c0r1;
        self.c1r0 += other.c1r0;
        self.c1r1 += other.c1r1;
    }
}

impl<S> ops::AddAssign<&Matrix2x2<S>> for Matrix2x2<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: &Matrix2x2<S>) {
        self.c0r0 += other.c0r0;
        self.c0r1 += other.c0r1;
        self.c1r0 += other.c1r0;
        self.c1r1 += other.c1r1;
    }
}

impl<S> ops::SubAssign<Matrix2x2<S>> for Matrix2x2<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: Matrix2x2<S>) {
        self.c0r0 -= other.c0r0;
        self.c0r1 -= other.c0r1;
        self.c1r0 -= other.c1r0;
        self.c1r1 -= other.c1r1;
    }
}

impl<S> ops::SubAssign<&Matrix2x2<S>> for Matrix2x2<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: &Matrix2x2<S>) {
        self.c0r0 -= other.c0r0;
        self.c0r1 -= other.c0r1;
        self.c1r0 -= other.c1r0;
        self.c1r1 -= other.c1r1;
    }
}

impl<S> ops::MulAssign<S> for Matrix2x2<S> where S: Scalar {
    #[inline]
    fn mul_assign(&mut self, other: S) {
        self.c0r0 *= other;
        self.c0r1 *= other;
        self.c1r0 *= other;
        self.c1r1 *= other;
    }
}

impl<S> ops::DivAssign<S> for Matrix2x2<S> where S: Scalar {
    #[inline]
    fn div_assign(&mut self, other: S) {
        self.c0r0 /= other;
        self.c0r1 /= other;
        self.c1r0 /= other;
        self.c1r1 /= other;
    }
}

impl<S> ops::RemAssign<S> for Matrix2x2<S> where S: Scalar {
    #[inline]
    fn rem_assign(&mut self, other: S) {
        self.c0r0 %= other;
        self.c0r1 %= other;
        self.c1r0 %= other;
        self.c1r1 %= other;
    }
}

impl<S> approx::AbsDiffEq for Matrix2x2<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.c0r0, &other.c0r0, epsilon) && 
        S::abs_diff_eq(&self.c0r1, &other.c0r1, epsilon) &&
        S::abs_diff_eq(&self.c1r0, &other.c1r0, epsilon) && 
        S::abs_diff_eq(&self.c1r1, &other.c1r1, epsilon)
    }
}

impl<S> approx::RelativeEq for Matrix2x2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.c0r0, &other.c0r0, epsilon, max_relative) &&
        S::relative_eq(&self.c0r1, &other.c0r1, epsilon, max_relative) &&
        S::relative_eq(&self.c1r0, &other.c1r0, epsilon, max_relative) &&
        S::relative_eq(&self.c1r1, &other.c1r1, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Matrix2x2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.c0r0, &other.c0r0, epsilon, max_ulps) &&
        S::ulps_eq(&self.c0r1, &other.c0r1, epsilon, max_ulps) &&
        S::ulps_eq(&self.c1r0, &other.c1r0, epsilon, max_ulps) &&
        S::ulps_eq(&self.c1r1, &other.c1r1, epsilon, max_ulps)
    }
}


impl<S> SquareMatrix for Matrix2x2<S> where S: ScalarFloat {
    type ColumnRow = Vector2<S>;

    #[rustfmt::skip]
    #[inline]
    fn from_diagonal_value(value: Self::Element) -> Self {
        Matrix2x2::new(
            value,     S::zero(),
            S::zero(), value
        )
    }
    
    #[rustfmt::skip]
    #[inline]
    fn from_diagonal(value: Vector2<S>) -> Self {
        Matrix2x2::new(
            value.x,   S::zero(),
            S::zero(), value.y
        )
    }
    
    #[inline]
    fn diagonal(&self) -> Vector2<S> {
        Vector2::new(self.c0r0, self.c1r1)
    }
    
    #[inline]
    fn transpose_in_place(&mut self) {
        self.swap_elements((0, 1), (1, 0));
    }
    
    #[inline]
    fn trace(&self) -> S {
        self.c0r0 + self.c1r1
    }

    fn determinant(&self) -> Self::Element {
        self.c0r0 * self.c1r1 - self.c0r1 * self.c1r0
    }
    
    #[inline]
    fn is_diagonal(&self) -> bool {
        ulps_eq!(self.c0r1, S::zero()) && ulps_eq!(self.c1r0, S::zero())
    }
    
    #[inline]
    fn is_symmetric(&self) -> bool {
        ulps_eq!(self.c0r1, self.c1r0) && ulps_eq!(self.c1r0, self.c0r1)
    }

    #[inline]
    fn is_identity(&self) -> bool {
        ulps_eq!(self, &<Self as Identity>::identity())
    }
}

impl<S> InvertibleSquareMatrix for Matrix2x2<S> where S: ScalarFloat {
    #[rustfmt::skip]
    fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == S::zero() {
            // A matrix with zero determinant has no inverse.
            None
        } else {
            let inv_det = S::one() / det;
            Some(Matrix2x2::new(
                inv_det *  self.c1r1, inv_det * -self.c0r1,
                inv_det * -self.c1r0, inv_det *  self.c0r0
            ))
        }
    }
}

impl<S: Scalar> iter::Sum<Matrix2x2<S>> for Matrix2x2<S> {
    #[inline]
    fn sum<I: Iterator<Item = Matrix2x2<S>>>(iter: I) -> Matrix2x2<S> {
        iter.fold(Matrix2x2::<S>::zero(), ops::Add::add)
    }
}

impl<'a, S: 'a + Scalar> iter::Sum<&'a Matrix2x2<S>> for Matrix2x2<S> {
    #[inline]
    fn sum<I: Iterator<Item = &'a Matrix2x2<S>>>(iter: I) -> Matrix2x2<S> {
        iter.fold(Matrix2x2::<S>::zero(), ops::Add::add)
    }
}

impl<S: Scalar> iter::Product<Matrix2x2<S>> for Matrix2x2<S> {
    #[inline]
    fn product<I: Iterator<Item = Matrix2x2<S>>>(iter: I) -> Matrix2x2<S> {
        iter.fold(Matrix2x2::<S>::identity(), ops::Mul::mul)
    }
}

impl<'a, S: 'a + Scalar> iter::Product<&'a Matrix2x2<S>> for Matrix2x2<S> {
    #[inline]
    fn product<I: Iterator<Item = &'a Matrix2x2<S>>>(iter: I) -> Matrix2x2<S> {
        iter.fold(Matrix2x2::<S>::identity(), ops::Mul::mul)
    }
}



/// The `Matrix3x3` type represents 3x3 matrices in column-major order.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C)]
pub struct Matrix3x3<S> {
    /// Column 0 of the matrix.
    pub c0r0: S, pub c0r1: S, pub c0r2: S,
    /// Column 1 of the matrix.
    pub c1r0: S, pub c1r1: S, pub c1r2: S,
    /// Column 2 of the matrix.
    pub c2r0: S, pub c2r1: S, pub c2r2: S,
}

impl<S> Matrix3x3<S> {
    /// Construct a new 3x3 matrix.
    #[rustfmt::skip]
    #[inline]
    pub const fn new(
        c0r0: S, c0r1: S, c0r2: S,
        c1r0: S, c1r1: S, c1r2: S,
        c2r0: S, c2r1: S, c2r2: S) -> Matrix3x3<S> {

        Matrix3x3 {
            // Column 0 of the matrix.
            c0r0: c0r0, c0r1: c0r1, c0r2: c0r2,
            // Column 1 of the matrix.
            c1r0: c1r0, c1r1: c1r1, c1r2: c1r2,
            // Column 2 of the matrix.
            c2r0: c2r0, c2r1: c2r1, c2r2: c2r2,
        }
    }

    /// Create a 3x3 matrix from a triple of three-dimensional column vectors.
    #[rustfmt::skip]
    #[inline]
    pub fn from_columns(c0: Vector3<S>, c1: Vector3<S>, c2: Vector3<S>) -> Matrix3x3<S> {
        Matrix3x3 {
            c0r0: c0.x, c0r1: c0.y, c0r2: c0.z, 
            c1r0: c1.x, c1r1: c1.y, c1r2: c1.z,
            c2r0: c2.x, c2r1: c2.y, c2r2: c2.z,
        }
    }

    /// Map an operation on the elements of a matrix, returning a matrix whose 
    /// elements are elements of the new underlying type.
    #[rustfmt::skip]
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Matrix3x3<T> where F: FnMut(S) -> T {
        Matrix3x3 {
            c0r0: op(self.c0r0), c1r0: op(self.c1r0), c2r0: op(self.c2r0),
            c0r1: op(self.c0r1), c1r1: op(self.c1r1), c2r1: op(self.c2r1),
            c0r2: op(self.c0r2), c1r2: op(self.c1r2), c2r2: op(self.c2r2),
        }
    }
}

impl<S> Matrix3x3<S> where S: Copy {
    /// Construct a new matrix from a fill value.
    ///
    /// The resulting matrix is a matrix where each entry is the supplied fill
    /// value.
    #[inline]
    pub fn from_fill(value: S) -> Matrix3x3<S> {
        Matrix3x3::new(
            value, value, value,
            value, value, value,
            value, value, value
        )
    }
}

impl<S> Matrix3x3<S> where S: NumCast + Copy {
    /// Cast a matrix from one type of scalars to another type of scalars.
    #[rustfmt::skip]
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Matrix3x3<T>> {
        let c0r0 = match num_traits::cast(self.c0r0) {
            Some(value) => value,
            None => return None,
        };
        let c0r1 = match num_traits::cast(self.c0r1) {
            Some(value) => value,
            None => return None,
        };
        let c0r2 = match num_traits::cast(self.c0r2) {
            Some(value) => value,
            None => return None,
        };
        let c1r0 = match num_traits::cast(self.c1r0) {
            Some(value) => value,
            None => return None,
        };
        let c1r1 = match num_traits::cast(self.c1r1) {
            Some(value) => value,
            None => return None,
        };
        let c1r2 = match num_traits::cast(self.c1r2) {
            Some(value) => value,
            None => return None,
        };
        let c2r0 = match num_traits::cast(self.c2r0) {
            Some(value) => value,
            None => return None,
        };
        let c2r1 = match num_traits::cast(self.c2r1) {
            Some(value) => value,
            None => return None,
        };
        let c2r2 = match num_traits::cast(self.c2r2) {
            Some(value) => value,
            None => return None,
        };

        Some(Matrix3x3::new(
            c0r0, c0r1, c0r2,
            c1r0, c1r1, c1r2, 
            c2r0, c2r1, c2r2,
        ))
    }
}

impl<S> Matrix3x3<S> where S: Scalar {
    /// Construct a two-dimensional affine translation matrix.
    ///
    /// This represents a translation in the xy-plane as an affine 
    /// transformation that displaces a vector along the length of the vector
    /// `distance`.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_translation(distance: Vector2<S>) -> Matrix3x3<S> {
        let one = S::one();
        let zero = S::zero();
        
        Matrix3x3::new(
            one,        zero,       zero,
            zero,       one,        zero,
            distance.x, distance.y, one
        )
    }
    
    /// Construct a three-dimensional uniform scaling matrix.
    ///
    /// The matrix applies the same scale factor to all dimensions, so each
    /// component of a vector will be scaled by the same factor. In particular,
    /// calling `from_scale(scale)` is equivalent to calling 
    /// `from_nonuniform_scale(scale, scale, scale)`.
    #[inline]
    pub fn from_scale(scale: S) -> Matrix3x3<S> {
        Matrix3x3::from_nonuniform_scale(scale, scale, scale)
    }
    
    /// Construct a three-dimensional general scaling matrix.
    ///
    /// This is the most general case for scaling matrices: the scale factor
    /// in each dimension need not be identical.
    #[rustfmt::skip]
    #[inline]
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S, scale_z: S) -> Matrix3x3<S> {
        let zero = S::zero();

        Matrix3x3::new(
            scale_x,   zero,      zero,
            zero,      scale_y,   zero,
            zero,      zero,      scale_z,
        )
    }

    /// Construct a two-dimensional uniform affine scaling matrix.
    ///
    /// The matrix applies the same scale factor to all dimensions, so each
    /// component of a vector will be scaled by the same factor. In particular,
    /// calling `from_scale(scale)` is equivalent to calling 
    /// `from_affine_nonuniform_scale(scale, scale)`. The z-component is 
    /// unaffected since this is an affine matrix.
    #[inline]
    pub fn from_affine_scale(scale: S) -> Matrix3x3<S> {
        Matrix3x3::from_affine_nonuniform_scale(scale, scale)
    }
    
    /// Construct a two-dimensional affine scaling matrix.
    ///
    /// This is the most general case for scaling matrices: the scale factor
    /// in each dimension need not be identical. The z-component is unaffected because
    /// this is an affine matrix.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_nonuniform_scale(scale_x: S, scale_y: S) -> Matrix3x3<S> {
        let zero = S::zero();
        let one = S::one();

        Matrix3x3::new(
            scale_x,   zero,      zero,
            zero,      scale_y,   zero,
            zero,      zero,      one,
        )
    }

    /// Construct a three-dimensional shearing matrix for shearing along the 
    /// _x-axis_, holding the y-axis constant and the _z-axis_ constant.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` are the 
    /// multiplicative factors for the contributions of the _y-axis_ and the 
    /// _z-axis_, respectively to shearing along the _x-axis_. 
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S, shear_x_with_z: S) -> Matrix3x3<S> {
        let one = S::one();
        let zero = S::zero();

        Matrix3x3::new(
            one,            zero, zero,
            shear_x_with_y, one,  zero, 
            shear_x_with_z, zero, one
        )
    }

    /// Construct a three-dimensional shearing matrix for shearing along the 
    /// _y-axis_, holding the x-axis constant and the z-axis constant.
    ///
    /// The parameters `shear_y_with_x` and `shear_y_with_z` are the
    /// multiplicative factors for the contributions of the _x-axis_, and the 
    /// _z-axis_, respectively to shearing along the _y-axis_. 
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_y(shear_y_with_x: S, shear_y_with_z: S) -> Matrix3x3<S> {
        let one = S::one();
        let zero = S::zero();

        Matrix3x3::new(
            one,  shear_y_with_x, zero,
            zero, one,            zero,
            zero, shear_y_with_z, one
        )
    }

    /// Construct a three-dimensional shearing matrix for shearing along the 
    /// _z-axis_, holding the _x-axis_ constant and the _y-axis_ constant.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` are the multiplicative
    /// factors for the contributions of the _x-axis_, and the _y-axis_, respectively to 
    /// shearing along the _z-axis_. 
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_z(shear_z_with_x: S, shear_z_with_y: S) -> Matrix3x3<S> {
        let one = S::one();
        let zero = S::zero();

        Matrix3x3::new(
            one,  zero, shear_z_with_x,
            zero, one,  shear_z_with_y,
            zero, zero, one   
        )
    }

    /// Construct a general shearing matrix in three dimensions. There are six
    /// parameters describing a shearing transformation in three dimensions.
    /// 
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the y-component to shearing of the x-component.
    ///
    /// The parameter `shear_x_with_z` denotes the factor scaling the 
    /// contribution  of the z-component to the shearing of the x-component.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the x-component to shearing of the y-component.
    ///
    /// The parameter `shear_y_with_z` denotes the factor scaling the 
    /// contribution of the _z-axis_ to the shearing of the y-component. 
    ///
    /// The parameter `shear_z_with_x` denotes the factor scaling the
    /// contribution of the _x-axis_ to shearing of the _z-axis_.
    ///
    /// The parameter `shear_z_with_y` denotes the factor scaling the 
    /// contribution of the y-component to the shearing of the z-component. 
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear(
        shear_x_with_y: S, shear_x_with_z: S, 
        shear_y_with_x: S, shear_y_with_z: S, 
        shear_z_with_x: S, shear_z_with_y: S) -> Matrix3x3<S> 
    {
        let one = S::one();

        Matrix3x3::new(
            one,            shear_y_with_x, shear_z_with_x,
            shear_x_with_y, one,            shear_z_with_y,
            shear_x_with_z, shear_y_with_z, one
        )
    }

    /// Construct a two-dimensional affine shearing matrix along the 
    /// _x-axis_, holding the _y-axis_ constant.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the _y-axis_ to shearing along the _x-axis_.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_x(shear_x_with_y: S) -> Matrix3x3<S> {
        let zero = S::zero();
        let one = S::one();

        Matrix3x3::new(
            one,            zero, zero,
            shear_x_with_y, one,  zero,
            zero,           zero, one
        )
    }

    /// Construct a two-dimensional affine shearing matrix along the 
    /// _y-axis_, holding the _x-axis_ constant.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the _y-axis_ to shearing along the _x-axis_.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_y(shear_y_with_x: S) -> Matrix3x3<S> {
        let zero = S::zero();
        let one = S::one();

        Matrix3x3::new(
            one,  shear_y_with_x, zero,
            zero, one,            zero,
            zero, zero,           one
        )
    }

    /// Construct a general affine shearing matrix in two dimensions. There are 
    /// two possible parameters describing a shearing transformation in two 
    /// dimensions.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the _x-axis_ to shearing along the _y-axis_.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the contribution 
    /// of the _y-axis_ to the shearing along the _x-axis_. 
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear(shear_x_with_y: S, shear_y_with_x: S) -> Matrix3x3<S> {
        let zero = S::zero();
        let one = S::one();

        Matrix3x3::new(
            one,            shear_y_with_x, zero,
            shear_x_with_y, one,            zero,
            zero,           zero,           one
        )
    }
}

impl<S> Matrix3x3<S> where S: ScalarSigned {
    /// Construct a two-dimensional affine reflection matrix in the xy-plane 
    /// for a line with normal vector `normal` and bias vector `bias`. The bias 
    /// vector can be any known point on the line of reflection.
    /// 
    /// The affine version of reflection generalizes the two-dimensional 
    /// `from_reflection` function in that `from_reflection` only works for 
    /// lines that cross the origin. If the line does not cross the origin, we 
    /// need to compute a translation in order to calculate the reflection 
    /// matrix. Since translation operations are affine and not linear, 
    /// constructing a general two-dimensional reflection requires an affine 
    /// transformation instead of a linear one.
    ///
    /// In particular, consider a line of the form
    /// ```text
    /// L = { (x, y) | a * (x - x0) + b * (y - y0) == 0 } 
    /// where (x0, x0) is a known point in L.
    /// ```
    /// A bare reflection matrix assumes that we can use the origin 
    /// (x0 = 0, y0 = 0) as a known point, which makes the translation terms 
    /// zero. This yields the matrix formula
    /// ```text
    /// |  1 - 2*nx*nx  -2*nx*ny       0 |
    /// | -2*nx*ny       1 - 2*ny*ny   0 |
    /// |  0             0             1 |
    /// ```
    /// In the case where the the line `L` does not cross the origin, we must 
    /// first do a coordinate transformation to coordinates where the line passes 
    /// through the origin: this is just a shift by the bias `(x0, y0)` from 
    /// `(x, y)` to `(x - x0, y - y0)`. We achieve this transformation in 
    /// homogeneous coordinates by the matrix
    /// ```text
    /// | 1  0  -x0 |
    /// | 0  1  -y0 |
    /// | 0  0   1  |
    /// ```
    /// This puts us in the shifted coordinate system where the line now passes 
    /// through the origin. In this coordinate system, we can now apply the 
    /// reflection matrix, which gives a homogeneous matrix equation 
    /// ```text
    /// | 1 0  -x0 |   |xr|    |  1 - 2*nx*nx   -2*nx*ny      0 |   | 1 0  -x0 |   |x|
    /// | 0 1  -y0 | * |yr| == | -2*nx*ny        1 - 2*ny*ny  0 | * | 0 1  -y0 | * |y|
    /// | 0 0   1  |   |1 |    |  0              0            1 |   | 0 0   1  |   |1|
    /// ```
    /// Then to solve for the reflection components, we invert the translation 
    /// matrix on the left hand side to get an equation of the form
    /// ```text
    /// |xr|    | 1 0  x0 |   |  1 - 2*nx*nx   -2*nx*ny      0 |   | 1 0  -x0 |   |x|
    /// |yr| == | 0 1  y0 | * | -2*nx*ny        1 - 2*ny*ny  0 | * | 0 1  -y0 | * |y|
    /// |1 |    | 0 0  1  |   |  0              0            1 |   | 0 0   1  |   |1|
    ///
    ///         |  1 - 2*nx*nx   -2*nx*ny       2*nx*(nx*n0 + ny*y0) |   |x|
    ///      == | -2*nx*ny        1 - 2*ny*ny   2*ny*(nx*x0 + ny*y0) | * |y|
    ///         |  0              0             1                    |   |1|
    /// ```
    /// Here the terms `xr` and `yr` are the coordinates of the reflected point 
    /// across the line `L`.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_reflection(normal: &Unit<Vector2<S>>, bias: &Vector2<S>) -> Matrix3x3<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;

        let c0r0 = one - two * normal.x * normal.x;
        let c0r1 = -two * normal.x * normal.y;
        let c0r2 = zero;

        let c1r0 = -two * normal.x * normal.y;
        let c1r1 = one - two * normal.y * normal.y;
        let c1r2 = zero;

        let c2r0 = two * normal.x * (normal.x * bias.x + normal.y * bias.y);
        let c2r1 = two * normal.y * (normal.x * bias.x + normal.y * bias.y);
        let c2r2 = one;

        Matrix3x3::new(
            c0r0, c0r1, c0r2,
            c1r0, c1r1, c1r2,
            c2r0, c2r1, c2r2
        )
    }

    /// Construct a three-dimensional reflection matrix for a plane that
    /// crosses the origin.
    #[rustfmt::skip]
    #[inline]
    pub fn from_reflection(normal: &Unit<Vector3<S>>) -> Matrix3x3<S> {
        let one = S::one();
        let two = one + one;

        let c0r0 =  one - two * normal.x * normal.x;
        let c0r1 = -two * normal.x * normal.y;
        let c0r2 = -two * normal.x * normal.z;

        let c1r0 = -two * normal.x * normal.y;
        let c1r1 =  one - two * normal.y * normal.y;
        let c1r2 = -two * normal.y * normal.z;

        let c2r0 = -two * normal.x * normal.z;
        let c2r1 = -two * normal.y * normal.z;
        let c2r2 =  one - two * normal.z * normal.z;
    
        Matrix3x3::new(
            c0r0, c0r1, c0r2,
            c1r0, c1r1, c1r2,
            c2r0, c2r1, c2r2
       )
    }
}

impl<S> Matrix3x3<S> where S: ScalarFloat {
    /// Construct an affine rotation matrix in two dimensions that rotates a 
    /// vector in the xy-plane by an angle `angle`.
    ///
    /// This is the affine matrix counterpart to the 2x2 matrix function 
    /// `from_angle`.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_angle<A: Into<Radians<S>>>(angle: A) -> Matrix3x3<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());
        let zero = S::zero();
        let one =  S::one();

        Matrix3x3::new(
             cos_angle, sin_angle, zero,
            -sin_angle, cos_angle, zero,
             zero,      zero,      one
        )
    }

    /// Construct a rotation matrix about the _x-axis_ by an angle `angle`.
    #[rustfmt::skip]
    #[inline]
    pub fn from_angle_x<A: Into<Radians<S>>>(angle: A) -> Matrix3x3<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Matrix3x3::new(
            S::one(),   S::zero(), S::zero(),
            S::zero(),  cos_angle, sin_angle,
            S::zero(), -sin_angle, cos_angle,
        )
    }

    /// Construct a rotation matrix about the _y-axis_ by an angle `angle`.
    #[rustfmt::skip]
    #[inline]
    pub fn from_angle_y<A: Into<Radians<S>>>(angle: A) -> Matrix3x3<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Matrix3x3::new(
            cos_angle, S::zero(), -sin_angle,
            S::zero(), S::one(),   S::zero(),
            sin_angle, S::zero(),  cos_angle,
        )
    }

    /// Construct a rotation matrix about the _z-axis_ by an angle `angle`.
    #[rustfmt::skip]
    #[inline]
    pub fn from_angle_z<A: Into<Radians<S>>>(angle: A) -> Matrix3x3<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Matrix3x3::new(
             cos_angle, sin_angle, S::zero(),
            -sin_angle, cos_angle, S::zero(),
             S::zero(), S::zero(), S::one(),
        )
    }

    /// Construct a rotation matrix about an arbitrary axis by an angle 
    /// `angle`.
    #[rustfmt::skip]
    #[inline]
    pub fn from_axis_angle<A: Into<Radians<S>>>(axis: &Unit<Vector3<S>>, angle: A) -> Matrix3x3<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());
        let one_minus_cos_angle = S::one() - cos_angle;
        let _axis = axis.as_ref();

        Matrix3x3::new(
            one_minus_cos_angle * _axis.x * _axis.x + cos_angle,
            one_minus_cos_angle * _axis.x * _axis.y + sin_angle * _axis.z,
            one_minus_cos_angle * _axis.x * _axis.z - sin_angle * _axis.y,

            one_minus_cos_angle * _axis.x * _axis.y - sin_angle * _axis.z,
            one_minus_cos_angle * _axis.y * _axis.y + cos_angle,
            one_minus_cos_angle * _axis.y * _axis.z + sin_angle * _axis.x,

            one_minus_cos_angle * _axis.x * _axis.z + sin_angle * _axis.y,
            one_minus_cos_angle * _axis.y * _axis.z - sin_angle * _axis.x,
            one_minus_cos_angle * _axis.z * _axis.z + cos_angle,
        )
    }

    /// Construct a rotation matrix that transforms the coordinate system of
    /// an observer located at the origin facing the _positive z-axis_ into a
    /// coordinate system of an observer located at the origin facing the 
    /// direction `direction`.
    ///
    /// The function maps the _z-axis_ to the direction `direction`.
    #[rustfmt::skip]
    #[inline]
    pub fn face_towards(direction: &Vector3<S>, up: &Vector3<S>) -> Matrix3x3<S> {
        let z_axis = direction.normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        Matrix3x3::new(
            x_axis.x, x_axis.y, x_axis.z,
            y_axis.x, y_axis.y, y_axis.z,
            z_axis.x, z_axis.y, z_axis.z
        )
    }

    /// Construct a coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing 
    /// the direction `direction` into the coordinate system of an observer located
    /// at the origin facing the _negative z-axis_.
    ///
    /// The function maps the direction `direction` to the _negative z-axis_ in 
    /// the new the coordinate system. This corresponds to a rotation matrix.
    /// This transformation is a _right-handed_ coordinate transformation. 
    #[inline]
    pub fn look_at_rh(direction: &Vector3<S>, up: &Vector3<S>) -> Matrix3x3<S> {
        // The inverse of a rotation matrix is its transpose.
        Self::face_towards(&(-direction), up).transpose()
    }

    /// Construct a coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing 
    /// the direction `direction` into the coordinate system of an observer located
    /// at the origin facing the _positive z-axis_.
    ///
    /// The function maps the direction `direction` to the _positive z-axis_ in 
    /// the new the coordinate system. This corresponds to a rotation matrix.
    /// This transformation is a _left-handed_ coordinate transformation. 
    #[inline]
    pub fn look_at_lh(direction: &Vector3<S>, up: &Vector3<S>) -> Matrix3x3<S> {
        // The inverse of a rotation matrix is its transpose.
        Self::face_towards(direction, up).transpose()
    }

    /// Construct a rotation matrix that rotates the shortest angular distance 
    /// between two vectors.
    #[inline]
    pub fn rotation_between(v1: &Vector3<S>, v2: &Vector3<S>) -> Option<Matrix3x3<S>> {
        if let (Some(unit_v1), Some(unit_v2)) = (v1.try_normalize(S::zero()), v2.try_normalize(S::zero())) {
            let cross = unit_v1.cross(&unit_v2);

            if let Some(axis) = Unit::try_from_value(cross, S::default_epsilon()) {
                return Some(
                    Matrix3x3::from_axis_angle(&axis, Radians::acos(unit_v1.dot(&unit_v2)))
                );
            }

            // Zero or PI.
            if unit_v1.dot(&unit_v2) < S::zero() {
                // PI
                //
                // The rotation axis is undefined but the angle not zero. This is not a
                // simple rotation.
                return None;
            }
        }

        Some(<Self as SquareMatrix>::identity())
    }

    /// Construct a rotation matrix that rotates the shortest angular distance 
    /// between two vectors.
    #[inline]
    pub fn rotation_between_axis(unit_v1: &Unit<Vector3<S>>, unit_v2: &Unit<Vector3<S>>) -> Option<Matrix3x3<S>> {
        let cross = unit_v1.as_ref().cross(unit_v2.as_ref());
        let cos_angle = unit_v1.as_ref().dot(unit_v2.as_ref());

        if let Some(axis) = Unit::try_from_value(cross, S::default_epsilon()) {
            return Some(
                Matrix3x3::from_axis_angle(&axis, Radians::acos(cos_angle))
            );
        }

        if cos_angle < S::zero() {
            return None;
        }

        Some(<Self as SquareMatrix>::identity())
    }

    /// Linearly interpolate between two matrices.
    #[inline]
    pub fn lerp(&self, other: &Matrix3x3<S>, amount: S) -> Matrix3x3<S> {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of a matrix are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A matrix is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values. For example, when the vector elements are `f64`, the vector is 
    /// finite when the elements are neither `NaN` nor infinite.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.c0r0.is_finite() && self.c0r1.is_finite() && self.c0r2.is_finite() &&
        self.c1r0.is_finite() && self.c1r1.is_finite() && self.c1r2.is_finite() &&
        self.c2r0.is_finite() && self.c2r1.is_finite() && self.c2r2.is_finite()
    }
}

impl<S> Array for Matrix3x3<S> where S: Copy {
    type Element = S;

    #[inline]
    fn len() -> usize {
        9
    }

    #[inline]
    fn shape() -> (usize, usize) {
        (3, 3)
    }

    #[inline]
    fn as_ptr(&self) -> *const S {
        &self.c0r0
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut S {
        &mut self.c0r0
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Element] {
        <Self as AsRef<[Self::Element; 9]>>::as_ref(self)
    }
}

impl<S> Matrix for Matrix3x3<S> where S: Scalar {
    type Element = S;
    type Row = Vector3<S>;
    type Column = Vector3<S>;
    type Transpose = Matrix3x3<S>;

    #[inline]
    fn row(&self, r: usize) -> Self::Row {
        Vector3::new(self[0][r], self[1][r], self[2][r])
    }
    
    #[inline]
    fn swap_rows(&mut self, row_a: usize, row_b: usize) {
        let c0ra = self[0][row_a];
        let c1ra = self[1][row_a];
        let c2ra = self[2][row_a];
        self[0][row_a] = self[0][row_b];
        self[1][row_a] = self[1][row_b];
        self[2][row_a] = self[2][row_b];
        self[0][row_b] = c0ra;
        self[1][row_b] = c1ra;
        self[2][row_b] = c2ra;
    }
    
    #[inline]
    fn swap_columns(&mut self, col_a: usize, col_b: usize) {
        let car0 = self[col_a][0];
        let car1 = self[col_a][1];
        let car2 = self[col_a][2];
        self[col_a][0] = self[col_b][0];
        self[col_a][1] = self[col_b][1];
        self[col_a][2] = self[col_b][2];
        self[col_b][0] = car0;
        self[col_b][1] = car1;
        self[col_b][2] = car2;
    }
    
    #[inline]
    fn swap_elements(&mut self, a: (usize, usize), b: (usize, usize)) {
        let element_a = self[a.0][a.1];
        self[a.0][a.1] = self[b.0][b.1];
        self[b.0][b.1] = element_a;
    }
    
    #[rustfmt::skip]
    #[inline]
    fn transpose(&self) -> Self::Transpose {
        Matrix3x3::new(
            self.c0r0, self.c1r0, self.c2r0,
            self.c0r1, self.c1r1, self.c2r1,
            self.c0r2, self.c1r2, self.c2r2
        )
    }
}

impl<S> From<[[S; 3]; 3]> for Matrix3x3<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn from(array: [[S; 3]; 3]) -> Matrix3x3<S> {
        Matrix3x3::new(
            array[0][0], array[0][1], array[0][2], 
            array[1][0], array[1][1], array[1][2], 
            array[2][0], array[2][1], array[2][2],
        )
    }
}

impl<'a, S> From<&'a [[S; 3]; 3]> for &'a Matrix3x3<S> where S: Scalar {
    #[inline]
    fn from(array: &'a [[S; 3]; 3]) -> &'a Matrix3x3<S> {
        unsafe { 
            &*(array as *const [[S; 3]; 3] as *const Matrix3x3<S>)
        }
    }    
}

impl<S> From<[S; 9]> for Matrix3x3<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn from(array: [S; 9]) -> Matrix3x3<S> {
        Matrix3x3::new(
            array[0], array[1], array[2], 
            array[3], array[4], array[5], 
            array[6], array[7], array[8]
        )
    }
}

impl<'a, S> From<&'a [S; 9]> for &'a Matrix3x3<S> where S: Scalar {
    #[inline]
    fn from(array: &'a [S; 9]) -> &'a Matrix3x3<S> {
        unsafe { 
            &*(array as *const [S; 9] as *const Matrix3x3<S>)
        }
    }
}

impl<S> From<Matrix2x2<S>> for Matrix3x3<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: Matrix2x2<S>) -> Matrix3x3<S> {
        Matrix3x3::new(
            matrix.c0r0,     matrix.c0r1,     S::zero(),
            matrix.c1r0,     matrix.c1r1,     S::zero(),
            S::zero(), S::zero(), S::one()
        )
    }
}

impl<S> From<&Matrix2x2<S>> for Matrix3x3<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: &Matrix2x2<S>) -> Matrix3x3<S> {
        Matrix3x3::new(
            matrix.c0r0,     matrix.c0r1,     S::zero(),
            matrix.c1r0,     matrix.c1r1,     S::zero(),
            S::zero(), S::zero(), S::one()
        )
    }
}

impl<S> fmt::Display for Matrix3x3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // We print the matrix contents in row-major order like mathematical convention.
        writeln!(
            formatter, 
            "Matrix3x3 [[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]]", 
            self.c0r0, self.c1r0, self.c2r0,
            self.c0r1, self.c1r1, self.c2r1,
            self.c0r2, self.c1r2, self.c2r2,
        )
    }
}

impl<S> AsRef<[S; 9]> for Matrix3x3<S> {
    #[inline]
    fn as_ref(&self) -> &[S; 9] {
        unsafe { 
            &*(self as *const Matrix3x3<S> as *const [S; 9])
        }
    }
}

impl<S> AsRef<[[S; 3]; 3]> for Matrix3x3<S> {
    #[inline]
    fn as_ref(&self) -> &[[S; 3]; 3] {
        unsafe { 
            &*(self as *const Matrix3x3<S> as *const [[S; 3]; 3])
        }
    }
}

impl<S> AsRef<[Vector3<S>; 3]> for Matrix3x3<S> {
    #[inline]
    fn as_ref(&self) -> &[Vector3<S>; 3] {
        unsafe { 
            &*(self as *const Matrix3x3<S> as *const [Vector3<S>; 3])
        }
    }
}

impl<S> AsMut<[S; 9]> for Matrix3x3<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; 9] {
        unsafe { 
            &mut *(self as *mut Matrix3x3<S> as *mut [S; 9])
        }
    }
}

impl<S> AsMut<[[S; 3]; 3]> for Matrix3x3<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [[S; 3];3 ] {
        unsafe { 
            &mut *(self as *mut Matrix3x3<S> as *mut [[S; 3]; 3])
        }
    }
}

impl<S> AsMut<[Vector3<S>; 3]> for Matrix3x3<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [Vector3<S>; 3] {
        unsafe { 
            &mut *(self as *mut Matrix3x3<S> as *mut [Vector3<S>; 3])
        }
    }
}

impl<S> ops::Index<usize> for Matrix3x3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[Vector3<S>; 3] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::IndexMut<usize> for Matrix3x3<S> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Vector3<S> {
        let v: &mut [Vector3<S>; 3] = self.as_mut();
        &mut v[index]
    }
}

impl<S> AdditiveIdentity for Matrix3x3<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn zero() -> Matrix3x3<S> {
        let zero = S::zero();
        Matrix3x3::new(
            zero, zero, zero, 
            zero, zero, zero, 
            zero, zero, zero
        )
    }

    #[inline]
    fn is_zero(&self) -> bool {
        let zero = S::zero();
        self.c0r0 == zero && self.c0r1 == zero && self.c0r2 == zero &&
        self.c1r0 == zero && self.c1r1 == zero && self.c1r2 == zero &&
        self.c2r0 == zero && self.c2r1 == zero && self.c2r2 == zero
    }
}

impl<S> Identity for Matrix3x3<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn identity() -> Matrix3x3<S> {
        let zero = S::zero();
        let one = S::one();
        Matrix3x3::new(
            one,  zero, zero, 
            zero, one,  zero, 
            zero, zero, one
        )
    }
}

impl<S> ops::Add<Matrix3x3<S>> for Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn add(self, other: Matrix3x3<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Add<&Matrix3x3<S>> for Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn add(self, other: &Matrix3x3<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Add<Matrix3x3<S>> for &Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn add(self, other: Matrix3x3<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<'a, 'b, S> ops::Add<&'a Matrix3x3<S>> for &'b Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn add(self, other: &'a Matrix3x3<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Sub<Matrix3x3<S>> for Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn sub(self, other: Matrix3x3<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Sub<&Matrix3x3<S>> for Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn sub(self, other: &Matrix3x3<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Sub<Matrix3x3<S>> for &Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn sub(self, other: Matrix3x3<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<'a, 'b, S> ops::Sub<&'a Matrix3x3<S>> for &'b Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn sub(self, other: &'a Matrix3x3<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Mul<&Matrix3x3<S>> for Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn mul(self, other: &Matrix3x3<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Matrix3x3<S>> for &'b Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn mul(self, other: &'a Matrix3x3<S>) -> Matrix3x3<S> {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Mul<Matrix3x3<S>> for Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn mul(self, other: Matrix3x3<S>) -> Matrix3x3<S> {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Mul<Matrix3x3<S>> for &Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn mul(self, other: Matrix3x3<S>) -> Matrix3x3<S> {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Mul<S> for Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c0r2 = self.c0r2 * other;

        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;
        let c1r2 = self.c1r2 * other;
        
        let c2r0 = self.c2r0 * other;
        let c2r1 = self.c2r1 * other;
        let c2r2 = self.c2r2 * other;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Mul<S> for &Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c0r2 = self.c0r2 * other;

        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;
        let c1r2 = self.c1r2 * other;
        
        let c2r0 = self.c2r0 * other;
        let c2r1 = self.c2r1 * other;
        let c2r2 = self.c2r2 * other;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Mul<Vector3<S>> for Matrix3x3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: Vector3<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y + self.c2r0 * other.z;
        let y = self.c0r1 * other.x + self.c1r1 * other.y + self.c2r1 * other.z;
        let z = self.c0r2 * other.x + self.c1r2 * other.y + self.c2r2 * other.z;

        Vector3::new(x, y, z)
    }
}

impl<S> ops::Mul<&Vector3<S>> for Matrix3x3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: &Vector3<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y + self.c2r0 * other.z;
        let y = self.c0r1 * other.x + self.c1r1 * other.y + self.c2r1 * other.z;
        let z = self.c0r2 * other.x + self.c1r2 * other.y + self.c2r2 * other.z;

        Vector3::new(x, y, z)
    }
}

impl<S> ops::Mul<Vector3<S>> for &Matrix3x3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: Vector3<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y + self.c2r0 * other.z;
        let y = self.c0r1 * other.x + self.c1r1 * other.y + self.c2r1 * other.z;
        let z = self.c0r2 * other.x + self.c1r2 * other.y + self.c2r2 * other.z;

        Vector3::new(x, y, z)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Vector3<S>> for &'b Matrix3x3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: &'a Vector3<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y + self.c2r0 * other.z;
        let y = self.c0r1 * other.x + self.c1r1 * other.y + self.c2r1 * other.z;
        let z = self.c0r2 * other.x + self.c1r2 * other.y + self.c2r2 * other.z;

        Vector3::new(x, y, z)
    }
}

impl<S> ops::Div<S> for Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c0r2 = self.c0r2 / other;

        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;
        let c1r2 = self.c1r2 / other;
        
        let c2r0 = self.c2r0 / other;
        let c2r1 = self.c2r1 / other;
        let c2r2 = self.c2r2 / other;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Div<S> for &Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c0r2 = self.c0r2 / other;

        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;
        let c1r2 = self.c1r2 / other;
        
        let c2r0 = self.c2r0 / other;
        let c2r1 = self.c2r1 / other;
        let c2r2 = self.c2r2 / other;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Neg for Matrix3x3<S> where S: ScalarSigned {
    type Output = Matrix3x3<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        let c0r0 = -self.c0r0;
        let c0r1 = -self.c0r1;
        let c0r2 = -self.c0r2;

        let c1r0 = -self.c1r0;
        let c1r1 = -self.c1r1;
        let c1r2 = -self.c1r2;

        let c2r0 = -self.c2r0;
        let c2r1 = -self.c2r1;
        let c2r2 = -self.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Neg for &Matrix3x3<S> where S: ScalarSigned {
    type Output = Matrix3x3<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        let c0r0 = -self.c0r0;
        let c0r1 = -self.c0r1;
        let c0r2 = -self.c0r2;

        let c1r0 = -self.c1r0;
        let c1r1 = -self.c1r1;
        let c1r2 = -self.c1r2;

        let c2r0 = -self.c2r0;
        let c2r1 = -self.c2r1;
        let c2r2 = -self.c2r2;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Rem<S> for Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 % other;
        let c0r1 = self.c0r1 % other;
        let c0r2 = self.c0r2 % other;

        let c1r0 = self.c1r0 % other;
        let c1r1 = self.c1r1 % other;
        let c1r2 = self.c1r2 % other;
        
        let c2r0 = self.c2r0 % other;
        let c2r1 = self.c2r1 % other;
        let c2r2 = self.c2r2 % other;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Rem<S> for &Matrix3x3<S> where S: Scalar {
    type Output = Matrix3x3<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 % other;
        let c0r1 = self.c0r1 % other;
        let c0r2 = self.c0r2 % other;

        let c1r0 = self.c1r0 % other;
        let c1r1 = self.c1r1 % other;
        let c1r2 = self.c1r2 % other;
        
        let c2r0 = self.c2r0 % other;
        let c2r1 = self.c2r1 % other;
        let c2r2 = self.c2r2 % other;

        Matrix3x3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)     
    }
}

impl<S> ops::AddAssign<Matrix3x3<S>> for Matrix3x3<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: Matrix3x3<S>) {
        self.c0r0 += other.c0r0;
        self.c0r1 += other.c0r1;
        self.c0r2 += other.c0r2;
        
        self.c1r0 += other.c1r0;
        self.c1r1 += other.c1r1;
        self.c1r2 += other.c1r2;

        self.c2r0 += other.c2r0;
        self.c2r1 += other.c2r1;
        self.c2r2 += other.c2r2;
    }
}

impl<S> ops::AddAssign<&Matrix3x3<S>> for Matrix3x3<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: &Matrix3x3<S>) {
        self.c0r0 += other.c0r0;
        self.c0r1 += other.c0r1;
        self.c0r2 += other.c0r2;
        
        self.c1r0 += other.c1r0;
        self.c1r1 += other.c1r1;
        self.c1r2 += other.c1r2;

        self.c2r0 += other.c2r0;
        self.c2r1 += other.c2r1;
        self.c2r2 += other.c2r2;
    }
}

impl<S> ops::SubAssign<Matrix3x3<S>> for Matrix3x3<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: Matrix3x3<S>) {
        self.c0r0 -= other.c0r0;
        self.c0r1 -= other.c0r1;
        self.c0r2 -= other.c0r2;
        
        self.c1r0 -= other.c1r0;
        self.c1r1 -= other.c1r1;
        self.c1r2 -= other.c1r2;

        self.c2r0 -= other.c2r0;
        self.c2r1 -= other.c2r1;
        self.c2r2 -= other.c2r2;
    }
}

impl<S> ops::SubAssign<&Matrix3x3<S>> for Matrix3x3<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: &Matrix3x3<S>) {
        self.c0r0 -= other.c0r0;
        self.c0r1 -= other.c0r1;
        self.c0r2 -= other.c0r2;
        
        self.c1r0 -= other.c1r0;
        self.c1r1 -= other.c1r1;
        self.c1r2 -= other.c1r2;

        self.c2r0 -= other.c2r0;
        self.c2r1 -= other.c2r1;
        self.c2r2 -= other.c2r2;
    }
}

impl<S> ops::MulAssign<S> for Matrix3x3<S> where S: Scalar {
    #[inline]
    fn mul_assign(&mut self, other: S) {
        self.c0r0 *= other;
        self.c0r1 *= other;
        self.c0r2 *= other;
        
        self.c1r0 *= other;
        self.c1r1 *= other;
        self.c1r2 *= other;

        self.c2r0 *= other;
        self.c2r1 *= other;
        self.c2r2 *= other;
    }
}

impl<S> ops::DivAssign<S> for Matrix3x3<S> where S: Scalar {
    #[inline]
    fn div_assign(&mut self, other: S) {
        self.c0r0 /= other;
        self.c0r1 /= other;
        self.c0r2 /= other;
        
        self.c1r0 /= other;
        self.c1r1 /= other;
        self.c1r2 /= other;

        self.c2r0 /= other;
        self.c2r1 /= other;
        self.c2r2 /= other;
    }
}

impl<S> ops::RemAssign<S> for Matrix3x3<S> where S: Scalar {
    #[inline]
    fn rem_assign(&mut self, other: S) {
        self.c0r0 %= other;
        self.c0r1 %= other;
        self.c0r2 %= other;
        
        self.c1r0 %= other;
        self.c1r1 %= other;
        self.c1r2 %= other;

        self.c2r0 %= other;
        self.c2r1 %= other;
        self.c2r2 %= other;
    }
}

impl<S> approx::AbsDiffEq for Matrix3x3<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.c0r0, &other.c0r0, epsilon) && 
        S::abs_diff_eq(&self.c0r1, &other.c0r1, epsilon) &&
        S::abs_diff_eq(&self.c0r2, &other.c0r2, epsilon) &&
        S::abs_diff_eq(&self.c1r0, &other.c1r0, epsilon) && 
        S::abs_diff_eq(&self.c1r1, &other.c1r1, epsilon) &&
        S::abs_diff_eq(&self.c1r2, &other.c1r2, epsilon) &&
        S::abs_diff_eq(&self.c2r0, &other.c2r0, epsilon) && 
        S::abs_diff_eq(&self.c2r1, &other.c2r1, epsilon) &&
        S::abs_diff_eq(&self.c2r2, &other.c2r2, epsilon)
    }
}

impl<S> approx::RelativeEq for Matrix3x3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.c0r0, &other.c0r0, epsilon, max_relative) &&
        S::relative_eq(&self.c0r1, &other.c0r1, epsilon, max_relative) &&
        S::relative_eq(&self.c0r2, &other.c0r2, epsilon, max_relative) &&
        S::relative_eq(&self.c1r0, &other.c1r0, epsilon, max_relative) &&
        S::relative_eq(&self.c1r1, &other.c1r1, epsilon, max_relative) &&
        S::relative_eq(&self.c1r2, &other.c1r2, epsilon, max_relative) &&
        S::relative_eq(&self.c2r0, &other.c2r0, epsilon, max_relative) &&
        S::relative_eq(&self.c2r1, &other.c2r1, epsilon, max_relative) &&
        S::relative_eq(&self.c2r2, &other.c2r2, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Matrix3x3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.c0r0, &other.c0r0, epsilon, max_ulps) &&
        S::ulps_eq(&self.c0r1, &other.c0r1, epsilon, max_ulps) &&
        S::ulps_eq(&self.c0r2, &other.c0r2, epsilon, max_ulps) &&
        S::ulps_eq(&self.c1r0, &other.c1r0, epsilon, max_ulps) &&
        S::ulps_eq(&self.c1r1, &other.c1r1, epsilon, max_ulps) &&
        S::ulps_eq(&self.c1r2, &other.c1r2, epsilon, max_ulps) &&
        S::ulps_eq(&self.c2r0, &other.c2r0, epsilon, max_ulps) &&
        S::ulps_eq(&self.c2r1, &other.c2r1, epsilon, max_ulps) &&
        S::ulps_eq(&self.c2r2, &other.c2r2, epsilon, max_ulps)
    }
}

impl<S> SquareMatrix for Matrix3x3<S> where S: ScalarFloat {
    type ColumnRow = Vector3<S>;

    #[rustfmt::skip]
    #[inline]
    fn from_diagonal_value(value: Self::Element) -> Self {
        Matrix3x3::new(
            value,     S::zero(), S::zero(),
            S::zero(), value,     S::zero(),
            S::zero(), S::zero(), value,
        )
    }
    
    #[rustfmt::skip]
    #[inline]
    fn from_diagonal(value: Self::ColumnRow) -> Self {
        Matrix3x3::new(
            value.x, S::zero(),    S::zero(),
            S::zero(),  value.y,   S::zero(),
            S::zero(),  S::zero(), value.z
        )
    }
    
    #[inline]
    fn diagonal(&self) -> Self::ColumnRow {
        Vector3::new(self.c0r0, self.c1r1, self.c2r2)
    }
    
    #[inline]
    fn transpose_in_place(&mut self) {
        self.swap_elements((0, 1), (1, 0));
        self.swap_elements((0, 2), (2, 0));
        self.swap_elements((1, 2), (2, 1));
    }
    
    #[inline]
    fn trace(&self) -> Self::Element {
        self.c0r0 + self.c1r1 + self.c2r2
    }

    #[rustfmt::skip]
    #[inline]
    fn determinant(&self) -> Self::Element {
        self.c0r0 * self.c1r1 * self.c2r2 - self.c0r0 * self.c1r2 * self.c2r1 -
        self.c1r0 * self.c0r1 * self.c2r2 + self.c1r0 * self.c0r2 * self.c2r1 +
        self.c2r0 * self.c0r1 * self.c1r2 - self.c2r0 * self.c0r2 * self.c1r1
    }
    
    #[inline]
    fn is_diagonal(&self) -> bool {
        ulps_eq!(self.c0r1, S::zero()) &&
        ulps_eq!(self.c0r2, S::zero()) && 
        ulps_eq!(self.c1r0, S::zero()) &&
        ulps_eq!(self.c1r2, S::zero()) &&
        ulps_eq!(self.c2r0, S::zero()) &&
        ulps_eq!(self.c2r1, S::zero())
    }
    
    #[inline]
    fn is_symmetric(&self) -> bool {
        ulps_eq!(self.c0r1, self.c1r0) && ulps_eq!(self.c1r0, self.c0r1) &&
        ulps_eq!(self.c0r2, self.c2r0) && ulps_eq!(self.c2r0, self.c0r2) &&
        ulps_eq!(self.c1r2, self.c2r1) && ulps_eq!(self.c2r1, self.c1r2)
    }

    #[inline]
    fn is_identity(&self) -> bool {
        ulps_eq!(self, &<Self as Identity>::identity())
    }
}

impl<S> InvertibleSquareMatrix for Matrix3x3<S> where S: ScalarFloat {
    #[rustfmt::skip]
    fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == S::zero() {
            // A matrix with zero determinant has no inverse.
            None
        } else {
            let inv_det = S::one() / det;

            Some(Matrix3x3::new(
                inv_det * (self.c1r1 * self.c2r2 - self.c1r2 * self.c2r1), 
                inv_det * (self.c0r2 * self.c2r1 - self.c0r1 * self.c2r2), 
                inv_det * (self.c0r1 * self.c1r2 - self.c0r2 * self.c1r1),
    
                inv_det * (self.c1r2 * self.c2r0 - self.c1r0 * self.c2r2),
                inv_det * (self.c0r0 * self.c2r2 - self.c0r2 * self.c2r0),
                inv_det * (self.c0r2 * self.c1r0 - self.c0r0 * self.c1r2),

                inv_det * (self.c1r0 * self.c2r1 - self.c1r1 * self.c2r0), 
                inv_det * (self.c0r1 * self.c2r0 - self.c0r0 * self.c2r1), 
                inv_det * (self.c0r0 * self.c1r1 - self.c0r1 * self.c1r0)
            ))
        }
    }
}

impl<S: Scalar> iter::Sum<Matrix3x3<S>> for Matrix3x3<S> {
    #[inline]
    fn sum<I: Iterator<Item = Matrix3x3<S>>>(iter: I) -> Matrix3x3<S> {
        iter.fold(Matrix3x3::<S>::zero(), ops::Add::add)
    }
}

impl<'a, S: 'a + Scalar> iter::Sum<&'a Matrix3x3<S>> for Matrix3x3<S> {
    #[inline]
    fn sum<I: Iterator<Item = &'a Matrix3x3<S>>>(iter: I) -> Matrix3x3<S> {
        iter.fold(Matrix3x3::<S>::zero(), ops::Add::add)
    }
}

impl<S: Scalar> iter::Product<Matrix3x3<S>> for Matrix3x3<S> {
    #[inline]
    fn product<I: Iterator<Item = Matrix3x3<S>>>(iter: I) -> Matrix3x3<S> {
        iter.fold(Matrix3x3::<S>::identity(), ops::Mul::mul)
    }
}

impl<'a, S: 'a + Scalar> iter::Product<&'a Matrix3x3<S>> for Matrix3x3<S> {
    #[inline]
    fn product<I: Iterator<Item = &'a Matrix3x3<S>>>(iter: I) -> Matrix3x3<S> {
        iter.fold(Matrix3x3::<S>::identity(), ops::Mul::mul)
    }
}



/// The `Matrix4x4` type represents 4x4 matrices in column-major order.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C)]
pub struct Matrix4x4<S> {
    /// Column 0 of the matrix.
    pub c0r0: S, pub c0r1: S, pub c0r2: S, pub c0r3: S,
    /// Column 1 of the matrix.
    pub c1r0: S, pub c1r1: S, pub c1r2: S, pub c1r3: S,
    /// Column 2 of the matrix.
    pub c2r0: S, pub c2r1: S, pub c2r2: S, pub c2r3: S,
    /// Column 3 of the matrix.
    pub c3r0: S, pub c3r1: S, pub c3r2: S, pub c3r3: S,
}

impl<S> Matrix4x4<S> {
    /// Construct a new 4x4 matrix.
    #[rustfmt::skip]
    #[inline]
    pub const fn new(
        c0r0: S, c0r1: S, c0r2: S, c0r3: S,
        c1r0: S, c1r1: S, c1r2: S, c1r3: S,
        c2r0: S, c2r1: S, c2r2: S, c2r3: S,
        c3r0: S, c3r1: S, c3r2: S, c3r3: S) -> Matrix4x4<S> {

        Matrix4x4 {
            c0r0: c0r0, c0r1: c0r1, c0r2: c0r2, c0r3: c0r3,
            c1r0: c1r0, c1r1: c1r1, c1r2: c1r2, c1r3: c1r3,
            c2r0: c2r0, c2r1: c2r1, c2r2: c2r2, c2r3: c2r3,
            c3r0: c3r0, c3r1: c3r1, c3r2: c3r2, c3r3: c3r3,
        }
    }

    /// Construct a 4x4 matrix from column vectors.
    #[rustfmt::skip]
    #[inline]
    pub fn from_columns(c0: Vector4<S>, c1: Vector4<S>, c2: Vector4<S>, c3: Vector4<S>) -> Matrix4x4<S> {
        Matrix4x4 {
            c0r0: c0.x, c0r1: c0.y, c0r2: c0.z, c0r3: c0.w,
            c1r0: c1.x, c1r1: c1.y, c1r2: c1.z, c1r3: c1.w,
            c2r0: c2.x, c2r1: c2.y, c2r2: c2.z, c2r3: c2.w,
            c3r0: c3.x, c3r1: c3.y, c3r2: c3.z, c3r3: c3.w,
        }
    }

    /// Map an operation on the elements of a matrix, returning a matrix whose elements
    /// are elements of the new underlying type.
    #[rustfmt::skip]
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Matrix4x4<T> where F: FnMut(S) -> T {
        Matrix4x4 {
            c0r0: op(self.c0r0), c1r0: op(self.c1r0), c2r0: op(self.c2r0), c3r0: op(self.c3r0),
            c0r1: op(self.c0r1), c1r1: op(self.c1r1), c2r1: op(self.c2r1), c3r1: op(self.c3r1),
            c0r2: op(self.c0r2), c1r2: op(self.c1r2), c2r2: op(self.c2r2), c3r2: op(self.c3r2),
            c0r3: op(self.c0r3), c1r3: op(self.c1r3), c2r3: op(self.c2r3), c3r3: op(self.c3r3),
        }
    }
}

impl<S> Matrix4x4<S> where S: Copy {
    /// Construct a new matrix from a fill value.
    ///
    /// The resulting matrix is a matrix where each entry is the supplied fill
    /// value.
    #[inline]
    pub fn from_fill(value: S) -> Matrix4x4<S> {
        Matrix4x4::new(
            value, value, value, value,
            value, value, value, value,
            value, value, value, value,
            value, value, value, value
        )
    }
}

impl<S> Matrix4x4<S> where S: NumCast + Copy {
    /// Cast a matrix from one type of scalars to another type of scalars.
    #[rustfmt::skip]
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Matrix4x4<T>> {
        let c0r0 = match num_traits::cast(self.c0r0) {
            Some(value) => value,
            None => return None,
        };
        let c0r1 = match num_traits::cast(self.c0r1) {
            Some(value) => value,
            None => return None,
        };
        let c0r2 = match num_traits::cast(self.c0r2) {
            Some(value) => value,
            None => return None,
        };
        let c0r3 = match num_traits::cast(self.c0r3) {
            Some(value) => value,
            None => return None,
        };
        let c1r0 = match num_traits::cast(self.c1r0) {
            Some(value) => value,
            None => return None,
        };
        let c1r1 = match num_traits::cast(self.c1r1) {
            Some(value) => value,
            None => return None,
        };
        let c1r2 = match num_traits::cast(self.c1r2) {
            Some(value) => value,
            None => return None,
        };
        let c1r3 = match num_traits::cast(self.c1r3) {
            Some(value) => value,
            None => return None,
        };
        let c2r0 = match num_traits::cast(self.c2r0) {
            Some(value) => value,
            None => return None,
        };
        let c2r1 = match num_traits::cast(self.c2r1) {
            Some(value) => value,
            None => return None,
        };
        let c2r2 = match num_traits::cast(self.c2r2) {
            Some(value) => value,
            None => return None,
        };
        let c2r3 = match num_traits::cast(self.c2r3) {
            Some(value) => value,
            None => return None,
        };
        let c3r0 = match num_traits::cast(self.c3r0) {
            Some(value) => value,
            None => return None,
        };
        let c3r1 = match num_traits::cast(self.c3r1) {
            Some(value) => value,
            None => return None,
        };
        let c3r2 = match num_traits::cast(self.c3r2) {
            Some(value) => value,
            None => return None,
        };
        let c3r3 = match num_traits::cast(self.c3r3) {
            Some(value) => value,
            None => return None,
        };

        Some(Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        ))
    }
}

impl<S> Matrix4x4<S> where S: Scalar {
    /// Construct an affine translation matrix in three-dimensions.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_translation(distance: Vector3<S>) -> Matrix4x4<S> {
        let one = S::one();
        let zero = S::zero();

        Matrix4x4::new(
            one,        zero,       zero,       zero,
            zero,       one,        zero,       zero,
            zero,       zero,       one,        zero,
            distance.x, distance.y, distance.z, one
        )
    }

    /// Construct a three-dimensional uniform affine scaling matrix.
    ///
    /// The matrix applies the same scale factor to all dimensions, so each
    /// component of a vector will be scaled by the same factor. In particular,
    /// calling `from_scale(scale)` is equivalent to calling 
    /// `from_nonuniform_scale(scale, scale, scale)`. Since this is an affine 
    /// matrix the `w` component is unaffected.
    #[inline]
    pub fn from_affine_scale(scale: S) -> Matrix4x4<S> {
        Matrix4x4::from_affine_nonuniform_scale(scale, scale, scale)
    }

    /// Construct a three-dimensional affine scaling matrix.
    ///
    /// This is the most general case for affine scaling matrices: the scale 
    /// factor in each dimension need not be identical. Since this is an 
    /// affine matrix, the `w` component is unaffected.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_nonuniform_scale(scale_x: S, scale_y: S, scale_z: S) -> Matrix4x4<S> {
        let one = S::one();
        let zero = S::zero();

        Matrix4x4::new(
            scale_x, zero,    zero,    zero,
            zero,    scale_y, zero,    zero,
            zero,    zero,    scale_z, zero,
            zero,    zero,    zero,    one
        )
    }

    /// Construct a three-dimensional affine shearing matrix for shearing 
    /// along the _x-axis_, holding the _y-axis_ constant and the _z-axis_ 
    /// constant.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` are the 
    /// multiplicative factors for the contributions of the _y-axis_, and the
    /// _z-axis_, respectively to shearing along the _x-axis_. Since this is an 
    /// affine transformation the `w` component of four-dimensional vectors is 
    /// unaffected.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_x(shear_x_with_y: S, shear_x_with_z: S) -> Matrix4x4<S> {
        let one = S::one();
        let zero = S::zero();
        
        Matrix4x4::new(
            one,            zero, zero, zero,
            shear_x_with_y, one,  zero, zero,
            shear_x_with_z, zero, one,  zero,
            zero,           zero, zero, one
        )
    }

    /// Construct a three-dimensional affine shearing matrix for shearing along 
    /// the _y-axis_, holding the _x-axis_ constant and the _z-axis_ constant.
    ///
    /// The parameters `shear_y_with_x` and `shear_y_with_z` are the 
    /// multiplicative factors for the contributions of the _x-axis_, and the 
    /// _z-axis_, respectively to shearing along the _y-axis_. Since this is 
    /// an affine transformation the `w` component of four-dimensional vectors 
    /// is unaffected.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_y(shear_y_with_x: S, shear_y_with_z: S) -> Matrix4x4<S> {
        let one = S::one();
        let zero = S::zero();

        Matrix4x4::new(
            one,  shear_y_with_x, zero, zero,
            zero, one,            zero, zero,
            zero, shear_y_with_z, one,  zero,
            zero, zero,           zero, one
        )
    }

    /// Construct a three-dimensional affine shearing matrix for shearing along 
    /// the _z-axis_, holding the _x-axis_ constant and the _y-axis_ constant.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` are the 
    /// multiplicative factors for the contributions of the _x-axis_, and the 
    /// _y-axis_, respectively to shearing along the _z-axis_. Since this is an 
    /// affine transformation the `w` component of four-dimensional vectors is 
    /// unaffected.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_z(shear_z_with_x: S, shear_z_with_y: S) -> Matrix4x4<S> {
        let one = S::one();
        let zero = S::zero();

        Matrix4x4::new(
            one,  zero, shear_z_with_x, zero,
            zero, one,  shear_z_with_y, zero,
            zero, zero, one,            zero,
            zero, zero, zero,           one
        )
    }

    /// Construct a general shearing affine matrix in three dimensions. 
    ///
    /// There are six parameters describing a shearing transformation in three 
    /// dimensions.
    /// 
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the y-axis to shearing along the _x-axis_.
    ///
    /// The parameter `shear_x_with_z` denotes the factor scaling the 
    /// contribution of the _z-axis_ to the shearing along the _x-axis_. 
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the _x-axis_ to shearing along the _y-axis_.
    ///
    /// The parameter `shear_y_with_z` denotes the factor scaling the 
    /// contribution of the _z-axis_ to the shearing along the _y-axis_. 
    ///
    /// The parameter `shear_z_with_x` denotes the factor scaling the
    /// contribution of the _x-axis_ to shearing along the _z-axis_.
    ///
    /// The parameter `shear_z_with_y` denotes the factor scaling the 
    /// contribution of the y-axis to the shearing along the _z-axis_. 
    ///
    /// Since this is an affine transformation the `w` component
    /// of four-dimensional vectors is unaffected.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear(
        shear_x_with_y: S, shear_x_with_z: S, 
        shear_y_with_x: S, shear_y_with_z: S, 
        shear_z_with_x: S, shear_z_with_y: S) -> Matrix4x4<S> 
    {
        let zero = S::zero();
        let one = S::one();

        Matrix4x4::new(
            one,            shear_y_with_x, shear_z_with_x, zero,
            shear_x_with_y, one,            shear_z_with_y, zero,
            shear_x_with_z, shear_y_with_z, one,            zero,
            zero,           zero,           zero,           one
        )
    }
}

impl<S> Matrix4x4<S> where S: ScalarSigned {
    /// Construct a three-dimensional affine reflection matrix for a plane with
    /// normal vector `normal` and bias vector `bias`. The bias vector can be 
    /// any known point on the plane of reflection.
    /// 
    /// The affine version of reflection generalizes the three-dimensional 
    /// `from_reflection` function in that `from_reflection` only works for 
    /// planes that cross the origin. If the plane does not cross the 
    /// origin, we need to compute a translation for the reflection matrix. 
    /// Since translation operations are affine and not linear, constructing a 
    /// general three-dimensional reflection transformation requires an affine 
    /// transformation instead of a linear one.
    ///
    /// In particular, consider a plane of the form
    /// ```text
    /// P = { (x, y, z) | a * (x - x0) + b * (y - y0) + c * (z - z0) == 0 }
    /// where (x0, y0, z0) is a known point in P.
    /// ```
    /// A bare reflection matrix assumes that the the x-axis intercept `x0` 
    /// and the y-axis intercept `y0` are both zero, in which case the 
    /// translation terms are zero. This yields the matrix formula
    /// ```text
    /// |  1 - 2*nx*nx   -2*nx*ny       -2*nx*nz       0 |
    /// | -2*nx*ny        1 - 2*ny*ny   -2*ny*nz       0 |
    /// | -2*nx*nz       -2*ny*nz        1 - 2*nz*nz   0 |
    /// |  0              0             0              1 |
    /// ```
    /// In the case where the the plane `P` does not cross the origin, we must 
    /// first do a coordinate transformation to coordinates where the line 
    /// passes through the origin; just shift by the bias `(x0, y0)` from 
    /// `(x, y)` to `(x - x0, y - y0)`. We achieve this transformation in 
    /// homogeneous coordinates by the matrix
    /// ```text
    /// | 1  0  0  -x0 |
    /// | 0  1  0  -y0 |
    /// | 0  0  1  -z0 |
    /// | 0  0  0   1  |
    /// ```
    /// This puts us in the shifted coordinate system where the line now passes 
    /// through the origin. In this coordinate system, we can now apply the 
    /// reflection matrix, which gives a homogeneous matrix equation 
    /// ```text
    /// | 1  0  0  -x0 |   |xr|    |  1 - 2*nx*nx   -2*nx*ny       -2*nx*nz       0 |   | 1  0  0  -x0 |   |x|
    /// | 0  1  0  -y0 | * |yr| == | -2*nx*ny        1 - 2*ny*ny   -2*ny*nz       0 | * | 0  1  0  -y0 | * |y|
    /// | 0  0  1  -z0 |   |zr|    | -2*nx*nz       -2*ny*nz        1 - 2*nz*nz   0 |   | 0  0  1  -z0 |   |z|
    /// | 0  0  0   1  |   |1 |    |  0              0             0              1 |   | 0  0  0   1  |   |1| 
    /// ```
    /// Then to solve for the reflection components, we invert the translation 
    /// matrix on the left hand side to get an equation of the form
    /// ```text
    /// |xr|    | 1  0  0  x0 |   |  1 - 2*nx*nx   -2*nx*ny       -2*nx*nz       0 |   | 1  0  0  -x0 |   |x|
    /// |yr| == | 0  1  0  y0 | * | -2*nx*ny        1 - 2*ny*ny   -2*ny*nz       0 | * | 0  1  0  -y0 | * |y|
    /// |zr|    | 0  0  1  z0 |   | -2*nx*nz       -2*ny*nz        1 - 2*nz*nz   0 |   | 0  0  1  -z0 |   |z|
    /// |1 |    | 0  0  0  1  |   |  0              0             0              1 |   | 0  0  0   1  |   |1| 
    ///
    ///         |  1 - 2*nx*nx   -2*nx*ny       -2*nx*xz       2*nx*(nx*n0 + ny*y0 + nz*z0) |   |x|
    ///      == | -2*nx*ny        1 - 2*ny*ny   -2*ny*nz       2*ny*(nx*x0 + ny*y0 + nz*z0) | * |y|
    ///         | -2*nx*nz       -2*ny*nz        1 - 2*nz*nz   2*nz*(nx*x0 + ny*y0 + nz*z0) |   |z|
    ///         |  0              0              0             1                            |   |1|
    /// ```
    /// Here the terms `xr`, `yr`, and `zr` are the coordinates of the 
    /// reflected point across the plane `P`.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_reflection(normal: &Unit<Vector3<S>>, bias: &Vector3<S>) -> Matrix4x4<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;

        let c0r0 =  one - two * normal.x * normal.x;
        let c0r1 = -two * normal.x * normal.y;
        let c0r2 = -two * normal.x * normal.z;
        let c0r3 = zero;

        let c1r0 = -two * normal.x * normal.y;
        let c1r1 =  one - two * normal.y * normal.y;
        let c1r2 = -two * normal.y * normal.z;
        let c1r3 =  zero;

        let c2r0 = -two * normal.x * normal.z;
        let c2r1 = -two * normal.y * normal.z;
        let c2r2 =  one - two * normal.z * normal.z;
        let c2r3 =  zero;

        let c3r0 = two * normal.x * (normal.x * bias.x + normal.y * bias.y + normal.z * bias.z);
        let c3r1 = two * normal.y * (normal.x * bias.x + normal.y * bias.y + normal.z * bias.z);
        let c3r2 = two * normal.z * (normal.x * bias.x + normal.y * bias.y + normal.z * bias.z);
        let c3r3 = one;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> Matrix4x4<S> where S: ScalarFloat {
    /// Construct a three-dimensional affine rotation matrix rotating a vector around the 
    /// x-axis by an angle `angle` radians/degrees.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_angle_x<A: Into<Radians<S>>>(angle: A) -> Matrix4x4<S> {
        let (sin_angle, cos_angle) = angle.into().sin_cos();
        let one = S::one();
        let zero = S::zero();

        Matrix4x4::new(
            one,   zero,      zero,      zero,
            zero,  cos_angle, sin_angle, zero,
            zero, -sin_angle, cos_angle, zero,
            zero,  zero,      zero,      one
        )
    }
        
    /// Construct a three-dimensional affine rotation matrix rotating a vector 
    /// around the y-axis by an angle `angle` radians/degrees.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_angle_y<A: Into<Radians<S>>>(angle: A) -> Matrix4x4<S> {
        let (sin_angle, cos_angle) = angle.into().sin_cos();
        let one = S::one();
        let zero = S::zero();

        Matrix4x4::new(
            cos_angle, zero, -sin_angle, zero,
            zero,      one,   zero,      zero,
            sin_angle, zero,  cos_angle, zero,
            zero,      zero,  zero,      one
        )
    }
    
    /// Construct a three-dimensional affine rotation matrix rotating a vector 
    /// around the z-axis by an angle `angle` radians/degrees.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_angle_z<A: Into<Radians<S>>>(angle: A) -> Matrix4x4<S> {
        let (sin_angle, cos_angle) = angle.into().sin_cos();
        let one = S::one();
        let zero = S::zero();
        
        Matrix4x4::new(
             cos_angle, sin_angle, zero, zero,
            -sin_angle, cos_angle, zero, zero,
             zero,      zero,      one,  zero,
             zero,      zero,      zero, one
        )
    }

    /// Construct a three-dimensional affine rotation matrix rotating a vector 
    /// around the axis `axis` by an angle `angle` radians/degrees.
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_axis_angle<A: Into<Radians<S>>>(axis: &Unit<Vector3<S>>, angle: A) -> Matrix4x4<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());
        let one_minus_cos_angle = S::one() - cos_angle;
        let _axis = axis.as_ref();

        Matrix4x4::new(
            one_minus_cos_angle * _axis.x * _axis.x + cos_angle,
            one_minus_cos_angle * _axis.x * _axis.y + sin_angle * _axis.z,
            one_minus_cos_angle * _axis.x * _axis.z - sin_angle * _axis.y,
            S::zero(),

            one_minus_cos_angle * _axis.x * _axis.y - sin_angle * _axis.z,
            one_minus_cos_angle * _axis.y * _axis.y + cos_angle,
            one_minus_cos_angle * _axis.y * _axis.z + sin_angle * _axis.x,
            S::zero(),

            one_minus_cos_angle * _axis.x * _axis.z + sin_angle * _axis.y,
            one_minus_cos_angle * _axis.y * _axis.z - sin_angle * _axis.x,
            one_minus_cos_angle * _axis.z * _axis.z + cos_angle,
            S::zero(),

            S::zero(), 
            S::zero(), 
            S::zero(), 
            S::one(),
        )
    }

    /// Construct a new three-dimensional orthographic projection matrix.
    #[rustfmt::skip]
    #[inline]
    pub fn from_orthographic(left: S, right: S, bottom: S, top: S, near: S, far: S) -> Matrix4x4<S> {
        let zero = S::zero();
        let one  = S::one();
        let two = one + one;
        let sx =  two / (right - left);
        let sy =  two / (top - bottom);
        let sz = -two / (far - near);
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);
        let tz = -(far + near) / (far - near);

        // We use the same orthographic projection matrix that OpenGL uses.
        Matrix4x4::new(
            sx,   zero, zero, zero,
            zero, sy,   zero, zero,
            zero, zero, sz,   zero,
            tx,   ty,   tz,   one
        )
    }

    /// Construct a new three-dimensional perspective projection matrix based
    /// on arbitrary `left`, `right`, `bottom`, `top`, `near` and `far` planes.
    #[rustfmt::skip]
    #[inline]
    pub fn from_perspective(left: S, right: S, bottom: S, top: S, near: S, far: S) -> Matrix4x4<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;

        let c0r0 = (two * near) / (right - left);
        let c0r1 = zero;
        let c0r2 = zero;
        let c0r3 = zero;

        let c1r0 = zero;
        let c1r1 = (two * near) / (top - bottom);
        let c1r2 = zero;
        let c1r3 = zero;

        let c2r0 =  (right + left)   / (right - left);
        let c2r1 =  (top   + bottom) / (top   - bottom);
        let c2r2 = -(far   + near)   / (far   - near);
        let c2r3 = -one;

        let c3r0 = zero;
        let c3r1 = zero;
        let c3r2 = -(two * far * near) / (far - near);
        let c3r3 = zero;

        // We use the same perspective projection matrix that OpenGL uses.
        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3,
        )
    }

    /// Construct a perspective projection based on the `near` plane, the `far` 
    /// plane and the vertical field of view angle `fovy` and the 
    /// horizontal/vertical aspect ratio `aspect`.
    #[rustfmt::skip]
    #[inline]
    pub fn from_perspective_fov<A: Into<Radians<S>>>(fovy: A, aspect: S, near: S, far: S) -> Matrix4x4<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let range = Angle::tan(fovy.into() / two) * near;
        let sx = (two * near) / (range * aspect + range * aspect);
        let sy = near / range;
        let sz = (far + near) / (near - far);
        let pz = (two * far * near) / (near - far);
        
        // We use the same perspective projection matrix that OpenGL uses.
        Matrix4x4::new(
            sx,    zero,  zero,  zero,
            zero,  sy,    zero,  zero,
            zero,  zero,  sz,   -one,
            zero,  zero,  pz,    zero
        )
    }

    /// Construct an affine coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing the _z-axis_
    /// into a coordinate system of an observer located at the position `eye` facing
    /// the direction `direction`.
    ///
    /// The function maps the _z-axis_ to the direction `direction`, and locates the 
    /// origin of the coordinate system to the `eye` position.
    #[rustfmt::skip]
    #[inline]
    pub fn face_towards(eye: &Point3<S>, direction: &Vector3<S>, up: &Vector3<S>) -> Matrix4x4<S> {
        let zero = S::zero();
        let one = S::one();
        let z_axis = direction.normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        let eye_vec = eye - Point3::origin();
        let eye_x = eye_vec.dot(x_axis);
        let eye_y = eye_vec.dot(y_axis);
        let eye_z = eye_vec.dot(z_axis);
        
        Matrix4x4::new(
             x_axis.x,  x_axis.y,  x_axis.z, zero,
             y_axis.x,  y_axis.y,  y_axis.z, zero,
             z_axis.x,  z_axis.y,  z_axis.z, zero,
             eye_x,     eye_y,     eye_z,    one
        )
    }

    /// Construct an affine coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the position `eye` facing 
    /// the direction `direction` into the coordinate system of an observer located
    /// at the origin facing the _negative z-axis_.
    ///
    /// The function maps the direction `direction` to the _negative z-axis_ and 
    /// locates the `eye` position to the origin in the new the coordinate system.
    /// This transformation is a _right-handed_ coordinate transformation. It is
    /// conventionally used in computer graphics for camera view transformations.
    #[rustfmt::skip]
    #[inline]
    pub fn look_at_rh(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Matrix4x4<S> {
        let direction = -(target - eye);
        
        let zero = S::zero();
        let one = S::one();
        let z_axis = direction.normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        let eye_vec = eye - Point3::origin();
        let neg_eye_x = -eye_vec.dot(x_axis);
        let neg_eye_y = -eye_vec.dot(y_axis);
        let neg_eye_z = -eye_vec.dot(z_axis);
        
        Matrix4x4::new(
            x_axis.x,  y_axis.x,  z_axis.x,  zero,
            x_axis.y,  y_axis.y,  z_axis.y,  zero,
            x_axis.z,  y_axis.z,  z_axis.z,  zero,
            neg_eye_x, neg_eye_y, neg_eye_z, one
        )
    }

    /// Construct an affine coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the position `eye` facing 
    /// the direction `direction` into the coordinate system of an observer located
    /// at the origin facing the _positive z-axis_.
    ///
    /// The function maps the direction `direction` to the _positive z-axis_ and 
    /// locates the `eye` position to the origin in the new the coordinate system.
    /// This transformation is a _left-handed_ coordinate transformation.
    #[rustfmt::skip]
    #[inline]
    pub fn look_at_lh(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Matrix4x4<S> {
        let direction = target - eye;
         
        let zero = S::zero();
        let one = S::one();
        let z_axis = direction.normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        let eye_vec = eye - Point3::origin();
        let neg_eye_x = -eye_vec.dot(x_axis);
        let neg_eye_y = -eye_vec.dot(y_axis);
        let neg_eye_z = -eye_vec.dot(z_axis);
        
        Matrix4x4::new(
            x_axis.x,  y_axis.x,  z_axis.x,  zero,
            x_axis.y,  y_axis.y,  z_axis.y,  zero,
            x_axis.z,  y_axis.z,  z_axis.z,  zero,
            neg_eye_x, neg_eye_y, neg_eye_z, one
        )
    }

    /// Linearly interpolate between two matrices.
    #[inline]
    pub fn lerp(&self, other: &Matrix4x4<S>, amount: S) -> Matrix4x4<S> {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of a matrix are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A matrix is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values. For example, when the vector elements are `f64`, the vector is 
    /// finite when the elements are neither `NaN` nor infinite.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.c0r0.is_finite() && self.c0r1.is_finite() && 
        self.c0r2.is_finite() && self.c0r3.is_finite() &&
        self.c1r0.is_finite() && self.c1r1.is_finite() && 
        self.c1r2.is_finite() && self.c1r3.is_finite() &&
        self.c2r0.is_finite() && self.c2r1.is_finite() && 
        self.c2r2.is_finite() && self.c2r3.is_finite() &&
        self.c3r0.is_finite() && self.c3r1.is_finite() &&
        self.c3r2.is_finite() && self.c3r3.is_finite()
    }
}

impl<S> Array for Matrix4x4<S> where S: Copy {
    type Element = S;

    #[inline]
    fn len() -> usize {
        16
    }

    #[inline]
    fn shape() -> (usize, usize) {
        (4, 4)
    }

    #[inline]
    fn as_ptr(&self) -> *const S {
        &self.c0r0
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut S {
        &mut self.c0r0
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Element] {
        <Self as AsRef<[Self::Element; 16]>>::as_ref(self)
    }
}

impl<S> Matrix for Matrix4x4<S> where S: Scalar {
    type Element = S;
    type Row = Vector4<S>;
    type Column = Vector4<S>;
    type Transpose = Matrix4x4<S>;

    #[inline]
    fn row(&self, r: usize) -> Self::Row {
        Vector4::new(self[0][r], self[1][r], self[2][r], self[3][r])
    }
    
    #[inline]
    fn swap_rows(&mut self, row_a: usize, row_b: usize) {
        let c0ra = self[0][row_a];
        let c1ra = self[1][row_a];
        let c2ra = self[2][row_a];
        let c3ra = self[3][row_a];
        self[0][row_a] = self[0][row_b];
        self[1][row_a] = self[1][row_b];
        self[2][row_a] = self[2][row_b];
        self[3][row_a] = self[3][row_b];
        self[0][row_b] = c0ra;
        self[1][row_b] = c1ra;
        self[2][row_b] = c2ra;
        self[3][row_b] = c3ra;
    }
    
    #[inline]
    fn swap_columns(&mut self, col_a: usize, col_b: usize) {
        let car0 = self[col_a][0];
        let car1 = self[col_a][1];
        let car2 = self[col_a][2];
        let car3 = self[col_a][3];
        self[col_a][0] = self[col_b][0];
        self[col_a][1] = self[col_b][1];
        self[col_a][2] = self[col_b][2];
        self[col_a][3] = self[col_b][3];
        self[col_b][0] = car0;
        self[col_b][1] = car1;
        self[col_b][2] = car2;
        self[col_b][3] = car3;
    }
    
    #[inline]
    fn swap_elements(&mut self, a: (usize, usize), b: (usize, usize)) {
        let element_a = self[a.0][a.1];
        self[a.0][a.1] = self[b.0][b.1];
        self[b.0][b.1] = element_a;
    }
    
    #[rustfmt::skip]
    #[inline]
    fn transpose(&self) -> Self::Transpose {
        Matrix4x4::new(
            self.c0r0, self.c1r0, self.c2r0, self.c3r0,
            self.c0r1, self.c1r1, self.c2r1, self.c3r1, 
            self.c0r2, self.c1r2, self.c2r2, self.c3r2, 
            self.c0r3, self.c1r3, self.c2r3, self.c3r3
        )
    }
}

impl<S> From<[[S; 4]; 4]> for Matrix4x4<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn from(array: [[S; 4]; 4]) -> Matrix4x4<S> {
        Matrix4x4::new(
            array[0][0], array[0][1], array[0][2], array[0][3], 
            array[1][0], array[1][1], array[1][2], array[1][3],
            array[2][0], array[2][1], array[2][2], array[2][3], 
            array[3][0], array[3][1], array[3][2], array[3][3]
        )
    }
}

impl<'a, S> From<&'a [[S; 4]; 4]> for &'a Matrix4x4<S> where S: Scalar {
    #[inline]
    fn from(array: &'a [[S; 4]; 4]) -> &'a Matrix4x4<S> {
        unsafe { 
            &*(array as *const [[S; 4]; 4] as *const Matrix4x4<S>)
        }
    }    
}

impl<S> From<[S; 16]> for Matrix4x4<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn from(array: [S; 16]) -> Matrix4x4<S> {
        Matrix4x4::new(
            array[0],  array[1],  array[2],  array[3], 
            array[4],  array[5],  array[6],  array[7],
            array[8],  array[9],  array[10], array[11], 
            array[12], array[13], array[14], array[15]
        )
    }
}

impl<'a, S> From<&'a [S; 16]> for &'a Matrix4x4<S> where S: Scalar {
    #[inline]
    fn from(array: &'a [S; 16]) -> &'a Matrix4x4<S> {
        unsafe { 
            &*(array as *const [S; 16] as *const Matrix4x4<S>)
        }
    }
}

impl<S> From<Matrix2x2<S>> for Matrix4x4<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: Matrix2x2<S>) -> Matrix4x4<S> {
        let one = S::one();
        let zero = S::zero();
        Matrix4x4::new(
            matrix.c0r0, matrix.c0r1, zero, zero,
            matrix.c1r0, matrix.c1r1, zero, zero,
            zero,  zero,  one,  zero,
            zero,  zero,  zero, one
        )
    }
}

impl<S> From<&Matrix2x2<S>> for Matrix4x4<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: &Matrix2x2<S>) -> Matrix4x4<S> {
        let one = S::one();
        let zero = S::zero();
        Matrix4x4::new(
            matrix.c0r0, matrix.c0r1, zero, zero,
            matrix.c1r0, matrix.c1r1, zero, zero,
            zero,  zero,  one,  zero,
            zero,  zero,  zero, one
        )
    }
}

impl<S> From<Matrix3x3<S>> for Matrix4x4<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: Matrix3x3<S>) -> Matrix4x4<S> {
        let one = S::one();
        let zero = S::zero();
        Matrix4x4::new(
            matrix.c0r0, matrix.c0r1, matrix.c0r2, zero,
            matrix.c1r0, matrix.c1r1, matrix.c1r2, zero,
            matrix.c2r0, matrix.c2r1, matrix.c2r2, zero,
            zero,  zero,  zero,  one
        )
    }
}

impl<S> From<&Matrix3x3<S>> for Matrix4x4<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: &Matrix3x3<S>) -> Matrix4x4<S> {
        let one = S::one();
        let zero = S::zero();
        Matrix4x4::new(
            matrix.c0r0, matrix.c0r1, matrix.c0r2, zero,
            matrix.c1r0, matrix.c1r1, matrix.c1r2, zero,
            matrix.c2r0, matrix.c2r1, matrix.c2r2, zero,
            zero, zero,    zero, one
        )
    }
}

impl<S> fmt::Display for Matrix4x4<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // We print the matrix contents in row-major order like mathematical convention.
        writeln!(
            formatter, 
            "Matrix4x4 [[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]]",
            self.c0r0, self.c1r0, self.c2r0, self.c3r0,
            self.c0r1, self.c1r1, self.c2r1, self.c3r1,
            self.c0r2, self.c1r2, self.c2r2, self.c3r2,
            self.c0r3, self.c1r3, self.c2r3, self.c3r3
        )
    }
}

impl<S> AsRef<[S; 16]> for Matrix4x4<S>  {
    #[inline]
    fn as_ref(&self) -> &[S; 16] {
        unsafe { 
            &*(self as *const Matrix4x4<S> as *const [S; 16])
        }
    }
}

impl<S> AsRef<[[S; 4]; 4]> for Matrix4x4<S> {
    #[inline]
    fn as_ref(&self) -> &[[S; 4]; 4] {
        unsafe { 
            &*(self as *const Matrix4x4<S> as *const [[S; 4]; 4])
        }
    }
}

impl<S> AsRef<[Vector4<S>; 4]> for Matrix4x4<S> {
    #[inline]
    fn as_ref(&self) -> &[Vector4<S>; 4] {
        unsafe { 
            &*(self as *const Matrix4x4<S> as *const [Vector4<S>; 4])
        }
    }
}

impl<S> AsMut<[S; 16]> for Matrix4x4<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; 16] {
        unsafe { 
            &mut *(self as *mut Matrix4x4<S> as *mut [S; 16])
        }
    }
}

impl<S> AsMut<[[S; 4]; 4]> for Matrix4x4<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [[S; 4]; 4] {
        unsafe { 
            &mut *(self as *mut Matrix4x4<S> as *mut [[S; 4]; 4])
        }
    }
}

impl<S> AsMut<[Vector4<S>; 4]> for Matrix4x4<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [Vector4<S>; 4] {
        unsafe { 
            &mut *(self as *mut Matrix4x4<S> as *mut [Vector4<S>; 4])
        }
    }
}

impl<S> ops::Index<usize> for Matrix4x4<S> {
    type Output = Vector4<S>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[Vector4<S>; 4] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::IndexMut<usize> for Matrix4x4<S> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Vector4<S> {
        let v: &mut [Vector4<S>; 4] = self.as_mut();
        &mut v[index]
    }
}

impl<S> AdditiveIdentity for Matrix4x4<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn zero() -> Matrix4x4<S> {
        let zero = S::zero();
        Matrix4x4::new(
            zero, zero, zero, zero, 
            zero, zero, zero, zero, 
            zero, zero, zero, zero, 
            zero, zero, zero, zero
        )
    }

    #[inline]
    fn is_zero(&self) -> bool {
        let zero = S::zero();
        self.c0r0 == zero && self.c0r1 == zero && 
        self.c0r2 == zero && self.c0r3 == zero &&
        self.c1r0 == zero && self.c1r1 == zero && 
        self.c1r2 == zero && self.c1r3 == zero &&
        self.c2r0 == zero && self.c2r1 == zero && 
        self.c2r2 == zero && self.c2r3 == zero &&
        self.c3r0 == zero && self.c3r1 == zero && 
        self.c3r2 == zero && self.c3r3 == zero
    }
}

impl<S> Identity for Matrix4x4<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn identity() -> Matrix4x4<S> {
        let one = S::one();
        let zero = S::zero();
        Matrix4x4::new(
            one,  zero, zero, zero, 
            zero, one,  zero, zero, 
            zero, zero, one,  zero, 
            zero, zero, zero, one
        )
    }
}

impl<S> ops::Add<Matrix4x4<S>> for Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn add(self, other: Matrix4x4<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;
        let c0r3 = self.c0r3 + other.c0r3;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;
        let c1r3 = self.c1r3 + other.c1r3;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;
        let c2r3 = self.c2r3 + other.c2r3;

        let c3r0 = self.c3r0 + other.c3r0;
        let c3r1 = self.c3r1 + other.c3r1;
        let c3r2 = self.c3r2 + other.c3r2;
        let c3r3 = self.c3r3 + other.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Add<&Matrix4x4<S>> for Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn add(self, other: &Matrix4x4<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;
        let c0r3 = self.c0r3 + other.c0r3;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;
        let c1r3 = self.c1r3 + other.c1r3;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;
        let c2r3 = self.c2r3 + other.c2r3;

        let c3r0 = self.c3r0 + other.c3r0;
        let c3r1 = self.c3r1 + other.c3r1;
        let c3r2 = self.c3r2 + other.c3r2;
        let c3r3 = self.c3r3 + other.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Add<Matrix4x4<S>> for &Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn add(self, other: Matrix4x4<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;
        let c0r3 = self.c0r3 + other.c0r3;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;
        let c1r3 = self.c1r3 + other.c1r3;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;
        let c2r3 = self.c2r3 + other.c2r3;

        let c3r0 = self.c3r0 + other.c3r0;
        let c3r1 = self.c3r1 + other.c3r1;
        let c3r2 = self.c3r2 + other.c3r2;
        let c3r3 = self.c3r3 + other.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<'a, 'b, S> ops::Add<&'a Matrix4x4<S>> for &'b Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn add(self, other: &'a Matrix4x4<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;
        let c0r3 = self.c0r3 + other.c0r3;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;
        let c1r3 = self.c1r3 + other.c1r3;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;
        let c2r3 = self.c2r3 + other.c2r3;

        let c3r0 = self.c3r0 + other.c3r0;
        let c3r1 = self.c3r1 + other.c3r1;
        let c3r2 = self.c3r2 + other.c3r2;
        let c3r3 = self.c3r3 + other.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Sub<Matrix4x4<S>> for Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn sub(self, other: Matrix4x4<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;
        let c0r3 = self.c0r3 - other.c0r3;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;
        let c1r3 = self.c1r3 - other.c1r3;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;
        let c2r3 = self.c2r3 - other.c2r3;

        let c3r0 = self.c3r0 - other.c3r0;
        let c3r1 = self.c3r1 - other.c3r1;
        let c3r2 = self.c3r2 - other.c3r2;
        let c3r3 = self.c3r3 - other.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Sub<&Matrix4x4<S>> for Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn sub(self, other: &Matrix4x4<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;
        let c0r3 = self.c0r3 - other.c0r3;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;
        let c1r3 = self.c1r3 - other.c1r3;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;
        let c2r3 = self.c2r3 - other.c2r3;

        let c3r0 = self.c3r0 - other.c3r0;
        let c3r1 = self.c3r1 - other.c3r1;
        let c3r2 = self.c3r2 - other.c3r2;
        let c3r3 = self.c3r3 - other.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Sub<Matrix4x4<S>> for &Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn sub(self, other: Matrix4x4<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;
        let c0r3 = self.c0r3 - other.c0r3;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;
        let c1r3 = self.c1r3 - other.c1r3;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;
        let c2r3 = self.c2r3 - other.c2r3;

        let c3r0 = self.c3r0 - other.c3r0;
        let c3r1 = self.c3r1 - other.c3r1;
        let c3r2 = self.c3r2 - other.c3r2;
        let c3r3 = self.c3r3 - other.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<'a, 'b, S> ops::Sub<&'a Matrix4x4<S>> for &'b Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn sub(self, other: &'a Matrix4x4<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;
        let c0r3 = self.c0r3 - other.c0r3;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;
        let c1r3 = self.c1r3 - other.c1r3;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;
        let c2r3 = self.c2r3 - other.c2r3;

        let c3r0 = self.c3r0 - other.c3r0;
        let c3r1 = self.c3r1 - other.c3r1;
        let c3r2 = self.c3r2 - other.c3r2;
        let c3r3 = self.c3r3 - other.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Mul<Vector4<S>> for Matrix4x4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: Vector4<S>) -> Self::Output {
        let x = self.c0r0 * other[0] + self.c1r0 * other[1] + self.c2r0 * other[2] + self.c3r0 * other[3];
        let y = self.c0r1 * other[0] + self.c1r1 * other[1] + self.c2r1 * other[2] + self.c3r1 * other[3];
        let z = self.c0r2 * other[0] + self.c1r2 * other[1] + self.c2r2 * other[2] + self.c3r2 * other[3];
        let w = self.c0r3 * other[0] + self.c1r3 * other[1] + self.c2r3 * other[2] + self.c3r3 * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<S> ops::Mul<&Vector4<S>> for Matrix4x4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: &Vector4<S>) -> Self::Output {
        let x = self.c0r0 * other[0] + self.c1r0 * other[1] + self.c2r0 * other[2] + self.c3r0 * other[3];
        let y = self.c0r1 * other[0] + self.c1r1 * other[1] + self.c2r1 * other[2] + self.c3r1 * other[3];
        let z = self.c0r2 * other[0] + self.c1r2 * other[1] + self.c2r2 * other[2] + self.c3r2 * other[3];
        let w = self.c0r3 * other[0] + self.c1r3 * other[1] + self.c2r3 * other[2] + self.c3r3 * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<S> ops::Mul<Vector4<S>> for &Matrix4x4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: Vector4<S>) -> Self::Output {
        let x = self.c0r0 * other[0] + self.c1r0 * other[1] + self.c2r0 * other[2] + self.c3r0 * other[3];
        let y = self.c0r1 * other[0] + self.c1r1 * other[1] + self.c2r1 * other[2] + self.c3r1 * other[3];
        let z = self.c0r2 * other[0] + self.c1r2 * other[1] + self.c2r2 * other[2] + self.c3r2 * other[3];
        let w = self.c0r3 * other[0] + self.c1r3 * other[1] + self.c2r3 * other[2] + self.c3r3 * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Vector4<S>> for &'b Matrix4x4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: &'a Vector4<S>) -> Self::Output {
        let x = self.c0r0 * other[0] + self.c1r0 * other[1] + self.c2r0 * other[2] + self.c3r0 * other[3];
        let y = self.c0r1 * other[0] + self.c1r1 * other[1] + self.c2r1 * other[2] + self.c3r1 * other[3];
        let z = self.c0r2 * other[0] + self.c1r2 * other[1] + self.c2r2 * other[2] + self.c3r2 * other[3];
        let w = self.c0r3 * other[0] + self.c1r3 * other[1] + self.c2r3 * other[2] + self.c3r3 * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<S> ops::Mul<Matrix4x4<S>> for Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: Matrix4x4<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2 + self.c3r0 * other.c0r3;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2 + self.c3r1 * other.c0r3;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2 + self.c3r2 * other.c0r3;
        let c0r3 = self.c0r3 * other.c0r0 + self.c1r3 * other.c0r1 + self.c2r3 * other.c0r2 + self.c3r3 * other.c0r3;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2 + self.c3r0 * other.c1r3;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2 + self.c3r1 * other.c1r3;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2 + self.c3r2 * other.c1r3;
        let c1r3 = self.c0r3 * other.c1r0 + self.c1r3 * other.c1r1 + self.c2r3 * other.c1r2 + self.c3r3 * other.c1r3;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2 + self.c3r0 * other.c2r3;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2 + self.c3r1 * other.c2r3;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2 + self.c3r2 * other.c2r3;
        let c2r3 = self.c0r3 * other.c2r0 + self.c1r3 * other.c2r1 + self.c2r3 * other.c2r2 + self.c3r3 * other.c2r3;

        let c3r0 = self.c0r0 * other.c3r0 + self.c1r0 * other.c3r1 + self.c2r0 * other.c3r2 + self.c3r0 * other.c3r3;
        let c3r1 = self.c0r1 * other.c3r0 + self.c1r1 * other.c3r1 + self.c2r1 * other.c3r2 + self.c3r1 * other.c3r3;
        let c3r2 = self.c0r2 * other.c3r0 + self.c1r2 * other.c3r1 + self.c2r2 * other.c3r2 + self.c3r2 * other.c3r3;
        let c3r3 = self.c0r3 * other.c3r0 + self.c1r3 * other.c3r1 + self.c2r3 * other.c3r2 + self.c3r3 * other.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Mul<&Matrix4x4<S>> for Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: &Matrix4x4<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2 + self.c3r0 * other.c0r3;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2 + self.c3r1 * other.c0r3;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2 + self.c3r2 * other.c0r3;
        let c0r3 = self.c0r3 * other.c0r0 + self.c1r3 * other.c0r1 + self.c2r3 * other.c0r2 + self.c3r3 * other.c0r3;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2 + self.c3r0 * other.c1r3;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2 + self.c3r1 * other.c1r3;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2 + self.c3r2 * other.c1r3;
        let c1r3 = self.c0r3 * other.c1r0 + self.c1r3 * other.c1r1 + self.c2r3 * other.c1r2 + self.c3r3 * other.c1r3;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2 + self.c3r0 * other.c2r3;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2 + self.c3r1 * other.c2r3;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2 + self.c3r2 * other.c2r3;
        let c2r3 = self.c0r3 * other.c2r0 + self.c1r3 * other.c2r1 + self.c2r3 * other.c2r2 + self.c3r3 * other.c2r3;

        let c3r0 = self.c0r0 * other.c3r0 + self.c1r0 * other.c3r1 + self.c2r0 * other.c3r2 + self.c3r0 * other.c3r3;
        let c3r1 = self.c0r1 * other.c3r0 + self.c1r1 * other.c3r1 + self.c2r1 * other.c3r2 + self.c3r1 * other.c3r3;
        let c3r2 = self.c0r2 * other.c3r0 + self.c1r2 * other.c3r1 + self.c2r2 * other.c3r2 + self.c3r2 * other.c3r3;
        let c3r3 = self.c0r3 * other.c3r0 + self.c1r3 * other.c3r1 + self.c2r3 * other.c3r2 + self.c3r3 * other.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Mul<Matrix4x4<S>> for &Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: Matrix4x4<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2 + self.c3r0 * other.c0r3;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2 + self.c3r1 * other.c0r3;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2 + self.c3r2 * other.c0r3;
        let c0r3 = self.c0r3 * other.c0r0 + self.c1r3 * other.c0r1 + self.c2r3 * other.c0r2 + self.c3r3 * other.c0r3;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2 + self.c3r0 * other.c1r3;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2 + self.c3r1 * other.c1r3;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2 + self.c3r2 * other.c1r3;
        let c1r3 = self.c0r3 * other.c1r0 + self.c1r3 * other.c1r1 + self.c2r3 * other.c1r2 + self.c3r3 * other.c1r3;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2 + self.c3r0 * other.c2r3;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2 + self.c3r1 * other.c2r3;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2 + self.c3r2 * other.c2r3;
        let c2r3 = self.c0r3 * other.c2r0 + self.c1r3 * other.c2r1 + self.c2r3 * other.c2r2 + self.c3r3 * other.c2r3;

        let c3r0 = self.c0r0 * other.c3r0 + self.c1r0 * other.c3r1 + self.c2r0 * other.c3r2 + self.c3r0 * other.c3r3;
        let c3r1 = self.c0r1 * other.c3r0 + self.c1r1 * other.c3r1 + self.c2r1 * other.c3r2 + self.c3r1 * other.c3r3;
        let c3r2 = self.c0r2 * other.c3r0 + self.c1r2 * other.c3r1 + self.c2r2 * other.c3r2 + self.c3r2 * other.c3r3;
        let c3r3 = self.c0r3 * other.c3r0 + self.c1r3 * other.c3r1 + self.c2r3 * other.c3r2 + self.c3r3 * other.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<'a, 'b, S> ops::Mul<&'a Matrix4x4<S>> for &'b Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: &'a Matrix4x4<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2 + self.c3r0 * other.c0r3;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2 + self.c3r1 * other.c0r3;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2 + self.c3r2 * other.c0r3;
        let c0r3 = self.c0r3 * other.c0r0 + self.c1r3 * other.c0r1 + self.c2r3 * other.c0r2 + self.c3r3 * other.c0r3;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2 + self.c3r0 * other.c1r3;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2 + self.c3r1 * other.c1r3;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2 + self.c3r2 * other.c1r3;
        let c1r3 = self.c0r3 * other.c1r0 + self.c1r3 * other.c1r1 + self.c2r3 * other.c1r2 + self.c3r3 * other.c1r3;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2 + self.c3r0 * other.c2r3;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2 + self.c3r1 * other.c2r3;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2 + self.c3r2 * other.c2r3;
        let c2r3 = self.c0r3 * other.c2r0 + self.c1r3 * other.c2r1 + self.c2r3 * other.c2r2 + self.c3r3 * other.c2r3;

        let c3r0 = self.c0r0 * other.c3r0 + self.c1r0 * other.c3r1 + self.c2r0 * other.c3r2 + self.c3r0 * other.c3r3;
        let c3r1 = self.c0r1 * other.c3r0 + self.c1r1 * other.c3r1 + self.c2r1 * other.c3r2 + self.c3r1 * other.c3r3;
        let c3r2 = self.c0r2 * other.c3r0 + self.c1r2 * other.c3r1 + self.c2r2 * other.c3r2 + self.c3r2 * other.c3r3;
        let c3r3 = self.c0r3 * other.c3r0 + self.c1r3 * other.c3r1 + self.c2r3 * other.c3r2 + self.c3r3 * other.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Mul<S> for Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c0r2 = self.c0r2 * other;
        let c0r3 = self.c0r3 * other;

        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;
        let c1r2 = self.c1r2 * other;
        let c1r3 = self.c1r3 * other;
        
        let c2r0 = self.c2r0 * other;
        let c2r1 = self.c2r1 * other;
        let c2r2 = self.c2r2 * other;
        let c2r3 = self.c2r3 * other;

        let c3r0 = self.c3r0 * other;
        let c3r1 = self.c3r1 * other;
        let c3r2 = self.c3r2 * other;
        let c3r3 = self.c3r3 * other;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Mul<S> for &Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c0r2 = self.c0r2 * other;
        let c0r3 = self.c0r3 * other;

        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;
        let c1r2 = self.c1r2 * other;
        let c1r3 = self.c1r3 * other;
        
        let c2r0 = self.c2r0 * other;
        let c2r1 = self.c2r1 * other;
        let c2r2 = self.c2r2 * other;
        let c2r3 = self.c2r3 * other;

        let c3r0 = self.c3r0 * other;
        let c3r1 = self.c3r1 * other;
        let c3r2 = self.c3r2 * other;
        let c3r3 = self.c3r3 * other;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Div<S> for Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn div(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c0r2 = self.c0r2 / other;
        let c0r3 = self.c0r3 / other;

        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;
        let c1r2 = self.c1r2 / other;
        let c1r3 = self.c1r3 / other;
        
        let c2r0 = self.c2r0 / other;
        let c2r1 = self.c2r1 / other;
        let c2r2 = self.c2r2 / other;
        let c2r3 = self.c2r3 / other;

        let c3r0 = self.c3r0 / other;
        let c3r1 = self.c3r1 / other;
        let c3r2 = self.c3r2 / other;
        let c3r3 = self.c3r3 / other;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Div<S> for &Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn div(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c0r2 = self.c0r2 / other;
        let c0r3 = self.c0r3 / other;

        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;
        let c1r2 = self.c1r2 / other;
        let c1r3 = self.c1r3 / other;
        
        let c2r0 = self.c2r0 / other;
        let c2r1 = self.c2r1 / other;
        let c2r2 = self.c2r2 / other;
        let c2r3 = self.c2r3 / other;

        let c3r0 = self.c3r0 / other;
        let c3r1 = self.c3r1 / other;
        let c3r2 = self.c3r2 / other;
        let c3r3 = self.c3r3 / other;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Neg for Matrix4x4<S> where S: ScalarSigned {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn neg(self) -> Self::Output {
        let c0r0 = -self.c0r0;
        let c0r1 = -self.c0r1;
        let c0r2 = -self.c0r2;
        let c0r3 = -self.c0r3;

        let c1r0 = -self.c1r0;
        let c1r1 = -self.c1r1;
        let c1r2 = -self.c1r2;
        let c1r3 = -self.c1r3;

        let c2r0 = -self.c2r0;
        let c2r1 = -self.c2r1;
        let c2r2 = -self.c2r2;
        let c2r3 = -self.c2r3;

        let c3r0 = -self.c3r0;
        let c3r1 = -self.c3r1;
        let c3r2 = -self.c3r2;
        let c3r3 = -self.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Neg for &Matrix4x4<S> where S: ScalarSigned {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn neg(self) -> Self::Output {
        let c0r0 = -self.c0r0;
        let c0r1 = -self.c0r1;
        let c0r2 = -self.c0r2;
        let c0r3 = -self.c0r3;

        let c1r0 = -self.c1r0;
        let c1r1 = -self.c1r1;
        let c1r2 = -self.c1r2;
        let c1r3 = -self.c1r3;

        let c2r0 = -self.c2r0;
        let c2r1 = -self.c2r1;
        let c2r2 = -self.c2r2;
        let c2r3 = -self.c2r3;

        let c3r0 = -self.c3r0;
        let c3r1 = -self.c3r1;
        let c3r2 = -self.c3r2;
        let c3r3 = -self.c3r3;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Rem<S> for Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 % other;
        let c0r1 = self.c0r1 % other;
        let c0r2 = self.c0r2 % other;
        let c0r3 = self.c0r3 % other;

        let c1r0 = self.c1r0 % other;
        let c1r1 = self.c1r1 % other;
        let c1r2 = self.c1r2 % other;
        let c1r3 = self.c1r3 % other;
        
        let c2r0 = self.c2r0 % other;
        let c2r1 = self.c2r1 % other;
        let c2r2 = self.c2r2 % other;
        let c2r3 = self.c2r3 % other;

        let c3r0 = self.c3r0 % other;
        let c3r1 = self.c3r1 % other;
        let c3r2 = self.c3r2 % other;
        let c3r3 = self.c3r3 % other;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Rem<S> for &Matrix4x4<S> where S: Scalar {
    type Output = Matrix4x4<S>;

    #[rustfmt::skip]
    #[inline]
    fn rem(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 % other;
        let c0r1 = self.c0r1 % other;
        let c0r2 = self.c0r2 % other;
        let c0r3 = self.c0r3 % other;

        let c1r0 = self.c1r0 % other;
        let c1r1 = self.c1r1 % other;
        let c1r2 = self.c1r2 % other;
        let c1r3 = self.c1r3 % other;
        
        let c2r0 = self.c2r0 % other;
        let c2r1 = self.c2r1 % other;
        let c2r2 = self.c2r2 % other;
        let c2r3 = self.c2r3 % other;

        let c3r0 = self.c3r0 % other;
        let c3r1 = self.c3r1 % other;
        let c3r2 = self.c3r2 % other;
        let c3r3 = self.c3r3 % other;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3, 
            c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, 
            c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::AddAssign<Matrix4x4<S>> for Matrix4x4<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: Matrix4x4<S>) {
        self.c0r0 += other.c0r0;
        self.c0r1 += other.c0r1;
        self.c0r2 += other.c0r2;
        self.c0r3 += other.c0r3;
        
        self.c1r0 += other.c1r0;
        self.c1r1 += other.c1r1;
        self.c1r2 += other.c1r2;
        self.c1r3 += other.c1r3;

        self.c2r0 += other.c2r0;
        self.c2r1 += other.c2r1;
        self.c2r2 += other.c2r2;
        self.c2r3 += other.c2r3;

        self.c3r0 += other.c3r0;
        self.c3r1 += other.c3r1;
        self.c3r2 += other.c3r2;
        self.c3r3 += other.c3r3;
    }
}

impl<S> ops::AddAssign<&Matrix4x4<S>> for Matrix4x4<S> where S: Scalar {
    #[inline]
    fn add_assign(&mut self, other: &Matrix4x4<S>) {
        self.c0r0 += other.c0r0;
        self.c0r1 += other.c0r1;
        self.c0r2 += other.c0r2;
        self.c0r3 += other.c0r3;
        
        self.c1r0 += other.c1r0;
        self.c1r1 += other.c1r1;
        self.c1r2 += other.c1r2;
        self.c1r3 += other.c1r3;

        self.c2r0 += other.c2r0;
        self.c2r1 += other.c2r1;
        self.c2r2 += other.c2r2;
        self.c2r3 += other.c2r3;

        self.c3r0 += other.c3r0;
        self.c3r1 += other.c3r1;
        self.c3r2 += other.c3r2;
        self.c3r3 += other.c3r3;
    }
}

impl<S> ops::SubAssign<Matrix4x4<S>> for Matrix4x4<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: Matrix4x4<S>) {
        self.c0r0 -= other.c0r0;
        self.c0r1 -= other.c0r1;
        self.c0r2 -= other.c0r2;
        self.c0r3 -= other.c0r3;
        
        self.c1r0 -= other.c1r0;
        self.c1r1 -= other.c1r1;
        self.c1r2 -= other.c1r2;
        self.c1r3 -= other.c1r3;

        self.c2r0 -= other.c2r0;
        self.c2r1 -= other.c2r1;
        self.c2r2 -= other.c2r2;
        self.c2r3 -= other.c2r3;

        self.c3r0 -= other.c3r0;
        self.c3r1 -= other.c3r1;
        self.c3r2 -= other.c3r2;
        self.c3r3 -= other.c3r3;
    }
}

impl<S> ops::SubAssign<&Matrix4x4<S>> for Matrix4x4<S> where S: Scalar {
    #[inline]
    fn sub_assign(&mut self, other: &Matrix4x4<S>) {
        self.c0r0 -= other.c0r0;
        self.c0r1 -= other.c0r1;
        self.c0r2 -= other.c0r2;
        self.c0r3 -= other.c0r3;
        
        self.c1r0 -= other.c1r0;
        self.c1r1 -= other.c1r1;
        self.c1r2 -= other.c1r2;
        self.c1r3 -= other.c1r3;

        self.c2r0 -= other.c2r0;
        self.c2r1 -= other.c2r1;
        self.c2r2 -= other.c2r2;
        self.c2r3 -= other.c2r3;

        self.c3r0 -= other.c3r0;
        self.c3r1 -= other.c3r1;
        self.c3r2 -= other.c3r2;
        self.c3r3 -= other.c3r3;
    }
}

impl<S> ops::MulAssign<S> for Matrix4x4<S> where S: Scalar {
    #[inline]
    fn mul_assign(&mut self, other: S) {
        self.c0r0 *= other;
        self.c0r1 *= other;
        self.c0r2 *= other;
        self.c0r3 *= other;
        
        self.c1r0 *= other;
        self.c1r1 *= other;
        self.c1r2 *= other;
        self.c1r3 *= other;

        self.c2r0 *= other;
        self.c2r1 *= other;
        self.c2r2 *= other;
        self.c2r3 *= other;

        self.c3r0 *= other;
        self.c3r1 *= other;
        self.c3r2 *= other;
        self.c3r3 *= other;
    }
}

impl<S> ops::DivAssign<S> for Matrix4x4<S> where S: Scalar {
    #[inline]
    fn div_assign(&mut self, other: S) {
        self.c0r0 /= other;
        self.c0r1 /= other;
        self.c0r2 /= other;
        self.c0r3 /= other;
        
        self.c1r0 /= other;
        self.c1r1 /= other;
        self.c1r2 /= other;
        self.c1r3 /= other;

        self.c2r0 /= other;
        self.c2r1 /= other;
        self.c2r2 /= other;
        self.c2r3 /= other;

        self.c3r0 /= other;
        self.c3r1 /= other;
        self.c3r2 /= other;
        self.c3r3 /= other;
    }
}

impl<S> ops::RemAssign<S> for Matrix4x4<S> where S: Scalar {
    #[inline]
    fn rem_assign(&mut self, other: S) {
        self.c0r0 %= other;
        self.c0r1 %= other;
        self.c0r2 %= other;
        self.c0r3 %= other;
        
        self.c1r0 %= other;
        self.c1r1 %= other;
        self.c1r2 %= other;
        self.c1r3 %= other;

        self.c2r0 %= other;
        self.c2r1 %= other;
        self.c2r2 %= other;
        self.c2r3 %= other;

        self.c3r0 %= other;
        self.c3r1 %= other;
        self.c3r2 %= other;
        self.c3r3 %= other;
    }
}

impl<S> approx::AbsDiffEq for Matrix4x4<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.c0r0, &other.c0r0, epsilon) && 
        S::abs_diff_eq(&self.c0r1, &other.c0r1, epsilon) &&
        S::abs_diff_eq(&self.c0r2, &other.c0r2, epsilon) &&
        S::abs_diff_eq(&self.c0r3, &other.c0r3, epsilon) && 
        S::abs_diff_eq(&self.c1r0, &other.c1r0, epsilon) && 
        S::abs_diff_eq(&self.c1r1, &other.c1r1, epsilon) &&
        S::abs_diff_eq(&self.c1r2, &other.c1r2, epsilon) &&
        S::abs_diff_eq(&self.c1r3, &other.c1r3, epsilon) && 
        S::abs_diff_eq(&self.c2r0, &other.c2r0, epsilon) && 
        S::abs_diff_eq(&self.c2r1, &other.c2r1, epsilon) &&
        S::abs_diff_eq(&self.c2r2, &other.c2r2, epsilon) &&
        S::abs_diff_eq(&self.c2r3, &other.c2r3, epsilon) && 
        S::abs_diff_eq(&self.c3r0, &other.c3r0, epsilon) && 
        S::abs_diff_eq(&self.c3r1, &other.c3r1, epsilon) &&
        S::abs_diff_eq(&self.c3r2, &other.c3r2, epsilon) &&
        S::abs_diff_eq(&self.c3r3, &other.c3r3, epsilon) 
    }
}

impl<S> approx::RelativeEq for Matrix4x4<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.c0r0, &other.c0r0, epsilon, max_relative) &&
        S::relative_eq(&self.c0r1, &other.c0r1, epsilon, max_relative) &&
        S::relative_eq(&self.c0r2, &other.c0r2, epsilon, max_relative) &&
        S::relative_eq(&self.c0r3, &other.c0r3, epsilon, max_relative) &&
        S::relative_eq(&self.c1r0, &other.c1r0, epsilon, max_relative) &&
        S::relative_eq(&self.c1r1, &other.c1r1, epsilon, max_relative) &&
        S::relative_eq(&self.c1r2, &other.c1r2, epsilon, max_relative) &&
        S::relative_eq(&self.c1r3, &other.c1r3, epsilon, max_relative) &&
        S::relative_eq(&self.c2r0, &other.c2r0, epsilon, max_relative) &&
        S::relative_eq(&self.c2r1, &other.c2r1, epsilon, max_relative) &&
        S::relative_eq(&self.c2r2, &other.c2r2, epsilon, max_relative) &&
        S::relative_eq(&self.c2r3, &other.c2r3, epsilon, max_relative) &&
        S::relative_eq(&self.c3r0, &other.c3r0, epsilon, max_relative) &&
        S::relative_eq(&self.c3r1, &other.c3r1, epsilon, max_relative) &&
        S::relative_eq(&self.c3r2, &other.c3r2, epsilon, max_relative) &&
        S::relative_eq(&self.c3r3, &other.c3r3, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Matrix4x4<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.c0r0, &other.c0r0, epsilon, max_ulps) &&
        S::ulps_eq(&self.c0r1, &other.c0r1, epsilon, max_ulps) &&
        S::ulps_eq(&self.c0r2, &other.c0r2, epsilon, max_ulps) &&
        S::ulps_eq(&self.c0r3, &other.c0r3, epsilon, max_ulps) &&
        S::ulps_eq(&self.c1r0, &other.c1r0, epsilon, max_ulps) &&
        S::ulps_eq(&self.c1r1, &other.c1r1, epsilon, max_ulps) &&
        S::ulps_eq(&self.c1r2, &other.c1r2, epsilon, max_ulps) &&
        S::ulps_eq(&self.c1r3, &other.c1r3, epsilon, max_ulps) &&
        S::ulps_eq(&self.c2r0, &other.c2r0, epsilon, max_ulps) &&
        S::ulps_eq(&self.c2r1, &other.c2r1, epsilon, max_ulps) &&
        S::ulps_eq(&self.c2r2, &other.c2r2, epsilon, max_ulps) &&
        S::ulps_eq(&self.c2r3, &other.c2r3, epsilon, max_ulps) &&
        S::ulps_eq(&self.c3r0, &other.c3r0, epsilon, max_ulps) &&
        S::ulps_eq(&self.c3r1, &other.c3r1, epsilon, max_ulps) &&
        S::ulps_eq(&self.c3r2, &other.c3r2, epsilon, max_ulps) &&
        S::ulps_eq(&self.c3r3, &other.c3r3, epsilon, max_ulps)
    }
}

impl<S> SquareMatrix for Matrix4x4<S> where S: ScalarFloat {
    type ColumnRow = Vector4<S>;

    #[rustfmt::skip]
    #[inline]
    fn from_diagonal_value(value: Self::Element) -> Self {
        Matrix4x4::new(
            value,     S::zero(), S::zero(), S::zero(),
            S::zero(), value,     S::zero(), S::zero(),
            S::zero(), S::zero(), value,     S::zero(),
            S::zero(), S::zero(), S::zero(), value
        )
    }
    
    #[rustfmt::skip]
    #[inline]
    fn from_diagonal(value: Self::ColumnRow) -> Self {
        Matrix4x4::new(
            value.x,   S::zero(), S::zero(), S::zero(),
            S::zero(), value.y,   S::zero(), S::zero(),
            S::zero(), S::zero(), value.z,   S::zero(),
            S::zero(), S::zero(), S::zero(), value.w,
        )
    }
    
    #[inline]
    fn diagonal(&self) -> Self::ColumnRow {
        Vector4::new(self.c0r0, self.c1r1, self.c2r2, self.c3r3)
    }
    
    #[inline]
    fn transpose_in_place(&mut self) {
        self.swap_elements((0, 1), (1, 0));
        self.swap_elements((0, 2), (2, 0));
        self.swap_elements((1, 2), (2, 1));
        self.swap_elements((0, 3), (3, 0));
        self.swap_elements((1, 3), (3, 1));
        self.swap_elements((2, 3), (3, 2));
    }

    #[rustfmt::skip]
    #[inline]
    fn determinant(&self) -> Self::Element {
        self.c0r0 * self.c1r1 * self.c2r2 * self.c3r3 -
        self.c0r0 * self.c1r1 * self.c2r3 * self.c3r2 -
        self.c0r0 * self.c2r1 * self.c1r2 * self.c3r3 +
        self.c0r0 * self.c2r1 * self.c1r3 * self.c3r2 +
        self.c0r0 * self.c3r1 * self.c1r2 * self.c2r3 -
        self.c0r0 * self.c3r1 * self.c1r3 * self.c2r2 -
        self.c1r0 * self.c0r1 * self.c2r2 * self.c3r3 +
        self.c1r0 * self.c0r1 * self.c2r3 * self.c3r2 +
        self.c1r0 * self.c2r1 * self.c0r2 * self.c3r3 -
        self.c1r0 * self.c2r1 * self.c0r3 * self.c3r2 -
        self.c1r0 * self.c3r1 * self.c0r2 * self.c2r3 +
        self.c1r0 * self.c3r1 * self.c0r3 * self.c2r2 +
        self.c2r0 * self.c0r1 * self.c1r2 * self.c3r3 -
        self.c2r0 * self.c0r1 * self.c1r3 * self.c3r2 -
        self.c2r0 * self.c1r1 * self.c0r2 * self.c3r3 +
        self.c2r0 * self.c1r1 * self.c0r3 * self.c3r2 +
        self.c2r0 * self.c3r1 * self.c0r2 * self.c1r3 -
        self.c2r0 * self.c3r1 * self.c0r3 * self.c1r2 -
        self.c3r0 * self.c0r1 * self.c1r2 * self.c2r3 +
        self.c3r0 * self.c0r1 * self.c1r3 * self.c2r2 +
        self.c3r0 * self.c1r1 * self.c0r2 * self.c2r3 -
        self.c3r0 * self.c1r1 * self.c0r3 * self.c2r2 -
        self.c3r0 * self.c2r1 * self.c0r2 * self.c1r3 +
        self.c3r0 * self.c2r1 * self.c0r3 * self.c1r2
    }
    
    #[inline]
    fn trace(&self) -> Self::Element {
        self.c0r0 + self.c1r1 + self.c2r2 + self.c3r3
    }
    
    #[inline]
    fn is_diagonal(&self) -> bool {
        ulps_eq!(self.c0r1, S::zero()) &&
        ulps_eq!(self.c0r2, S::zero()) && 
        ulps_eq!(self.c1r0, S::zero()) &&
        ulps_eq!(self.c1r2, S::zero()) &&
        ulps_eq!(self.c2r0, S::zero()) &&
        ulps_eq!(self.c2r1, S::zero())
    }
    
    #[inline]
    fn is_symmetric(&self) -> bool {
        ulps_eq!(self.c0r1, self.c1r0) && ulps_eq!(self.c1r0, self.c0r1) &&
        ulps_eq!(self.c0r2, self.c2r0) && ulps_eq!(self.c2r0, self.c0r2) &&
        ulps_eq!(self.c1r2, self.c2r1) && ulps_eq!(self.c2r1, self.c1r2) &&
        ulps_eq!(self.c0r3, self.c3r0) && ulps_eq!(self.c3r0, self.c0r3) &&
        ulps_eq!(self.c1r3, self.c3r1) && ulps_eq!(self.c3r1, self.c1r3) &&
        ulps_eq!(self.c2r3, self.c3r2) && ulps_eq!(self.c3r2, self.c2r3)
    }

    #[inline]
    fn is_identity(&self) -> bool {
        ulps_eq!(self, &<Self as Identity>::identity())
    }
}

impl<S> InvertibleSquareMatrix for Matrix4x4<S> where S: ScalarFloat {
    #[rustfmt::skip]
    fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == S::zero() {
            // A matrix with zero determinant has no inverse.
            None
        } else {
            let det_inv = S::one() / det;
            let _c0r0 = self.c1r1 * self.c2r2 * self.c3r3 + self.c2r1 * self.c3r2 * self.c1r3 + self.c3r1 * self.c1r2 * self.c2r3
                      - self.c3r1 * self.c2r2 * self.c1r3 - self.c2r1 * self.c1r2 * self.c3r3 - self.c1r1 * self.c3r2 * self.c2r3;
            let _c1r0 = self.c3r0 * self.c2r2 * self.c1r3 + self.c2r0 * self.c1r2 * self.c3r3 + self.c1r0 * self.c3r2 * self.c2r3
                      - self.c1r0 * self.c2r2 * self.c3r3 - self.c2r0 * self.c3r2 * self.c1r3 - self.c3r0 * self.c1r2 * self.c2r3;
            let _c2r0 = self.c1r0 * self.c2r1 * self.c3r3 + self.c2r0 * self.c3r1 * self.c1r3 + self.c3r0 * self.c1r1 * self.c2r3
                      - self.c3r0 * self.c2r1 * self.c1r3 - self.c2r0 * self.c1r1 * self.c3r3 - self.c1r0 * self.c3r1 * self.c2r3;
            let _c3r0 = self.c3r0 * self.c2r1 * self.c1r2 + self.c2r0 * self.c1r1 * self.c3r2 + self.c1r0 * self.c3r1 * self.c2r2
                      - self.c1r0 * self.c2r1 * self.c3r2 - self.c2r0 * self.c3r1 * self.c1r2 - self.c3r0 * self.c1r1 * self.c2r2;
            let _c0r1 = self.c3r1 * self.c2r2 * self.c0r3 + self.c2r1 * self.c0r2 * self.c3r3 + self.c0r1 * self.c3r2 * self.c2r3
                      - self.c0r1 * self.c2r2 * self.c3r3 - self.c2r1 * self.c3r2 * self.c0r3 - self.c3r1 * self.c0r2 * self.c2r3;
            let _c1r1 = self.c0r0 * self.c2r2 * self.c3r3 + self.c2r0 * self.c3r2 * self.c0r3 + self.c3r0 * self.c0r2 * self.c2r3
                      - self.c3r0 * self.c2r2 * self.c0r3 - self.c2r0 * self.c0r2 * self.c3r3 - self.c0r0 * self.c3r2 * self.c2r3;
            let _c2r1 = self.c3r0 * self.c2r1 * self.c0r3 + self.c2r0 * self.c0r1 * self.c3r3 + self.c0r0 * self.c3r1 * self.c2r3
                      - self.c0r0 * self.c2r1 * self.c3r3 - self.c2r0 * self.c3r1 * self.c0r3 - self.c3r0 * self.c0r1 * self.c2r3;
            let _c3r1 = self.c0r0 * self.c2r1 * self.c3r2 + self.c2r0 * self.c3r1 * self.c0r2 + self.c3r0 * self.c0r1 * self.c2r2
                      - self.c3r0 * self.c2r1 * self.c0r2 - self.c2r0 * self.c0r1 * self.c3r2 - self.c0r0 * self.c3r1 * self.c2r2;
            let _c0r2 = self.c0r1 * self.c1r2 * self.c3r3 + self.c1r1 * self.c3r2 * self.c0r3 + self.c3r1 * self.c0r2 * self.c1r3 
                      - self.c3r1 * self.c1r2 * self.c0r3 - self.c1r1 * self.c0r2 * self.c3r3 - self.c0r1 * self.c3r2 * self.c1r3;
            let _c1r2 = self.c3r0 * self.c1r2 * self.c0r3 + self.c1r0 * self.c0r2 * self.c3r3 + self.c0r0 * self.c3r2 * self.c1r3
                      - self.c0r0 * self.c1r2 * self.c3r3 - self.c1r0 * self.c3r2 * self.c0r3 - self.c3r0 * self.c0r2 * self.c1r3;
            let _c2r2 = self.c0r0 * self.c1r1 * self.c3r3 + self.c1r0 * self.c3r1 * self.c0r3 + self.c3r0 * self.c0r1 * self.c1r3
                      - self.c3r0 * self.c1r1 * self.c0r3 - self.c1r0 * self.c0r1 * self.c3r3 - self.c0r0 * self.c3r1 * self.c1r3;
            let _c3r2 = self.c3r0 * self.c1r1 * self.c0r2 + self.c1r0 * self.c0r1 * self.c3r2 + self.c0r0 * self.c3r1 * self.c1r2
                      - self.c0r0 * self.c1r1 * self.c3r2 - self.c1r0 * self.c3r1 * self.c0r2 - self.c3r0 * self.c0r1 * self.c1r2;
            let _c0r3 = self.c2r1 * self.c1r2 * self.c0r3 + self.c1r1 * self.c0r2 * self.c2r3 + self.c0r1 * self.c2r2 * self.c1r3
                      - self.c0r1 * self.c1r2 * self.c2r3 - self.c1r1 * self.c2r2 * self.c0r3 - self.c2r1 * self.c0r2 * self.c1r3;  
            let _c1r3 = self.c0r0 * self.c1r2 * self.c2r3 + self.c1r0 * self.c2r2 * self.c0r3 + self.c2r0 * self.c0r2 * self.c1r3
                      - self.c2r0 * self.c1r2 * self.c0r3 - self.c1r0 * self.c0r2 * self.c2r3 - self.c0r0 * self.c2r2 * self.c1r3;
            let _c2r3 = self.c2r0 * self.c1r1 * self.c0r3 + self.c1r0 * self.c0r1 * self.c2r3 + self.c0r0 * self.c2r1 * self.c1r3
                      - self.c0r0 * self.c1r1 * self.c2r3 - self.c1r0 * self.c2r1 * self.c0r3 - self.c2r0 * self.c0r1 * self.c1r3;
            let _c3r3 = self.c0r0 * self.c1r1 * self.c2r2 + self.c1r0 * self.c2r1 * self.c0r2 + self.c2r0 * self.c0r1 * self.c1r2
                      - self.c2r0 * self.c1r1 * self.c0r2 - self.c1r0 * self.c0r1 * self.c2r2 - self.c0r0 * self.c2r1 * self.c1r2; 
            
            let c0r0 = det_inv * _c0r0; 
            let c0r1 = det_inv * _c0r1; 
            let c0r2 = det_inv * _c0r2; 
            let c0r3 = det_inv * _c0r3;

            let c1r0 = det_inv * _c1r0; 
            let c1r1 = det_inv * _c1r1; 
            let c1r2 = det_inv * _c1r2; 
            let c1r3 = det_inv * _c1r3;

            let c2r0 = det_inv * _c2r0; 
            let c2r1 = det_inv * _c2r1; 
            let c2r2 = det_inv * _c2r2; 
            let c2r3 = det_inv * _c2r3;

            let c3r0 = det_inv * _c3r0; 
            let c3r1 = det_inv * _c3r1; 
            let c3r2 = det_inv * _c3r2; 
            let c3r3 = det_inv * _c3r3;

            Some(Matrix4x4::new(
                c0r0, c0r1, c0r2, c0r3,
                c1r0, c1r1, c1r2, c1r3,
                c2r0, c2r1, c2r2, c2r3,
                c3r0, c3r1, c3r2, c3r3
            ))
        }
    }
}

impl<S: Scalar> iter::Sum<Matrix4x4<S>> for Matrix4x4<S> {
    #[inline]
    fn sum<I: Iterator<Item = Matrix4x4<S>>>(iter: I) -> Matrix4x4<S> {
        iter.fold(Matrix4x4::<S>::zero(), ops::Add::add)
    }
}

impl<'a, S: 'a + Scalar> iter::Sum<&'a Matrix4x4<S>> for Matrix4x4<S> {
    #[inline]
    fn sum<I: Iterator<Item = &'a Matrix4x4<S>>>(iter: I) -> Matrix4x4<S> {
        iter.fold(Matrix4x4::<S>::zero(), ops::Add::add)
    }
}

impl<S: Scalar> iter::Product<Matrix4x4<S>> for Matrix4x4<S> {
    #[inline]
    fn product<I: Iterator<Item = Matrix4x4<S>>>(iter: I) -> Matrix4x4<S> {
        iter.fold(Matrix4x4::<S>::identity(), ops::Mul::mul)
    }
}

impl<'a, S: 'a + Scalar> iter::Product<&'a Matrix4x4<S>> for Matrix4x4<S> {
    #[inline]
    fn product<I: Iterator<Item = &'a Matrix4x4<S>>>(iter: I) -> Matrix4x4<S> {
        iter.fold(Matrix4x4::<S>::identity(), ops::Mul::mul)
    }
}


impl_mul_operator!(u8,    Matrix2x2<u8>,    Matrix2x2<u8>,    { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(u16,   Matrix2x2<u16>,   Matrix2x2<u16>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(u32,   Matrix2x2<u32>,   Matrix2x2<u32>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(u64,   Matrix2x2<u64>,   Matrix2x2<u64>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(u128,  Matrix2x2<u128>,  Matrix2x2<u128>,  { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(usize, Matrix2x2<usize>, Matrix2x2<usize>, { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(i8,    Matrix2x2<i8>,    Matrix2x2<i8>,    { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(i16,   Matrix2x2<i16>,   Matrix2x2<i16>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(i32,   Matrix2x2<i32>,   Matrix2x2<i32>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(i64,   Matrix2x2<i64>,   Matrix2x2<i64>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(i128,  Matrix2x2<i128>,  Matrix2x2<i128>,  { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(isize, Matrix2x2<isize>, Matrix2x2<isize>, { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(f32,   Matrix2x2<f32>,   Matrix2x2<f32>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(f64,   Matrix2x2<f64>,   Matrix2x2<f64>,   { c0r0, c0r1, c1r0, c1r1 });

impl_mul_operator!(u8,    Matrix3x3<u8>,    Matrix3x3<u8>,    { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(u16,   Matrix3x3<u16>,   Matrix3x3<u16>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(u32,   Matrix3x3<u32>,   Matrix3x3<u32>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(u64,   Matrix3x3<u64>,   Matrix3x3<u64>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(u128,  Matrix3x3<u128>,  Matrix3x3<u128>,  { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(usize, Matrix3x3<usize>, Matrix3x3<usize>, { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(i8,    Matrix3x3<i8>,    Matrix3x3<i8>,    { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(i16,   Matrix3x3<i16>,   Matrix3x3<i16>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(i32,   Matrix3x3<i32>,   Matrix3x3<i32>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(i64,   Matrix3x3<i64>,   Matrix3x3<i64>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(i128,  Matrix3x3<i128>,  Matrix3x3<i128>,  { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(isize, Matrix3x3<isize>, Matrix3x3<isize>, { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(f32,   Matrix3x3<f32>,   Matrix3x3<f32>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(f64,   Matrix3x3<f64>,   Matrix3x3<f64>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });

impl_mul_operator!(u8,    Matrix4x4<u8>,    Matrix4x4<u8>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(u16,   Matrix4x4<u16>,   Matrix4x4<u16>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(u32,   Matrix4x4<u32>,   Matrix4x4<u32>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(u64,   Matrix4x4<u64>,   Matrix4x4<u64>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(u128,  Matrix4x4<u128>,  Matrix4x4<u128>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(usize, Matrix4x4<usize>, Matrix4x4<usize>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(i8,    Matrix4x4<i8>,    Matrix4x4<i8>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(i16,   Matrix4x4<i16>,   Matrix4x4<i16>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(i32,   Matrix4x4<i32>,   Matrix4x4<i32>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(i64,   Matrix4x4<i64>,   Matrix4x4<i64>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(i128,  Matrix4x4<i128>,  Matrix4x4<i128>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(isize, Matrix4x4<isize>, Matrix4x4<isize>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(f32,   Matrix4x4<f32>,   Matrix4x4<f32>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(f64,   Matrix4x4<f64>,   Matrix4x4<f64>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);

