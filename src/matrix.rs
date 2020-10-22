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
use crate::magnitude::{
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
use crate::array::*;
use approx::{
    ulps_eq,
    ulps_ne,
};
use num_traits::{
    NumCast,
};

use core::fmt;
use core::ops::*;
use core::ops;
use core::iter;
use core::mem;


macro_rules! impl_coords {
    ($T:ident, { $($comps: ident),* }) => {
        /// Data structure used to provide access to matrix and vector coordinates with the dot
        /// notation, e.g., `v.x` is the same as `v[0]` for a vector.
        #[repr(C)]
        #[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
        pub struct $T<S: Copy> {
            $(pub $comps: S),*
        }
    }
}

macro_rules! impl_coords_deref {
    ($Source:ident, $Target:ident) => {
        impl<S> Deref for $Source<S> where S: Copy
        {
            type Target = $Target<S>;

            #[inline]
            fn deref(&self) -> &Self::Target {
                unsafe { mem::transmute(self.data.as_ptr()) }
            }
        }

        impl<S> DerefMut for $Source<S> where S: Copy
        {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { mem::transmute(self.data.as_mut_ptr()) }
            }
        }
    }
}

macro_rules! impl_as_ref_ops {
    ($MatrixType:ty, $RefType:ty) => {
        impl<S> AsRef<$RefType> for $MatrixType {
            #[inline]
            fn as_ref(&self) -> &$RefType {
                unsafe {
                    &*(self as *const $MatrixType as *const $RefType)
                }
            }
        }

        impl<S> AsMut<$RefType> for $MatrixType {
            #[inline]
            fn as_mut(&mut self) -> &mut $RefType {
                unsafe {
                    &mut *(self as *mut $MatrixType as *mut $RefType)
                }
            }
        }
    }
}

macro_rules! impl_matrix_scalar_binary_ops1 {
    ($OpType:ident, $op:ident, $op_impl:ident, $T:ty, $Output:ty, { $( ($col:expr, $row:expr) ),* }) => {
        impl<S> $OpType<S> for $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: S) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(&self.data, other, $col, $row) ),* 
                )
            }
        }

        impl<S> $OpType<S> for &$T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: S) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(&self.data, other, $col, $row) ),* 
                )
            }
        }
    }
}

macro_rules! impl_matrix_matrix_binary_ops1 {
    ($OpType:ident, $op:ident, $op_impl:ident, $T:ty, $Output:ty, { $( ($col:expr, $row:expr) ),* }) => {
        impl<S> $OpType<$T> for $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: $T) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(&self.data, &other.data, $col, $row) ),* 
                )
            }
        }

        impl<S> $OpType<&$T> for $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: &$T) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(&self.data, &other.data, $col, $row) ),* 
                )
            }
        }

        impl<S> $OpType<$T> for &$T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: $T) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(&self.data, &other.data, $col, $row) ),* 
                )
            }
        }

        impl<'a, 'b, S> $OpType<&'a $T> for &'b $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: &'a $T) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(&self.data, &other.data, $col, $row) ),* 
                )
            }
        }
    }
}

macro_rules! impl_matrix_unary_ops1 {
    ($OpType:ident, $op:ident, $op_impl:ident, $T:ty, $Output:ty, { $( ($col:expr, $row:expr) ),* }) => {
        impl<S> $OpType for $T where S: ScalarSigned {
            type Output = $Output;

            #[inline]
            fn $op(self) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(&self.data, $col, $row) ),* 
                )
            }
        }

        impl<S> $OpType for &$T where S: ScalarSigned {
            type Output = $Output;

            #[inline]
            fn $op(self) -> Self::Output {
                Self::Output::new( 
                    $( $op_impl(&self.data, $col, $row) ),* 
                )
            }
        }
    }
}

macro_rules! impl_matrix_binary_assign_ops1 {
    ($T:ty, { $( ($col:expr, $row:expr) ),* }) => {
        impl<S> ops::AddAssign<$T> for $T where S: Scalar {
            #[inline]
            fn add_assign(&mut self, other: $T) {
                $( self.data[$col][$row] += other.data[$col][$row] );*
            }
        }

        impl<S> ops::AddAssign<&$T> for $T where S: Scalar {
            #[inline]
            fn add_assign(&mut self, other: &$T) {
                $( self.data[$col][$row] += other.data[$col][$row] );*
            }
        }

        impl<S> ops::SubAssign<$T> for $T where S: Scalar {
            #[inline]
            fn sub_assign(&mut self, other: $T) {
                $( self.data[$col][$row] -= other.data[$col][$row] );*
            }
        }

        impl<S> ops::SubAssign<&$T> for $T where S: Scalar {
            #[inline]
            fn sub_assign(&mut self, other: &$T) {
                $( self.data[$col][$row] -= other.data[$col][$row] );*
            }
        }

        impl<S> ops::MulAssign<S> for $T where S: Scalar {
            #[inline]
            fn mul_assign(&mut self, other: S) {
                $( self.data[$col][$row] *= other );*
            }
        }
        
        impl<S> ops::DivAssign<S> for $T where S: Scalar {
            #[inline]
            fn div_assign(&mut self, other: S) {
                $( self.data[$col][$row] /= other );*
            }
        }
        
        impl<S> ops::RemAssign<S> for $T where S: Scalar {
            #[inline]
            fn rem_assign(&mut self, other: S) {
                $( self.data[$col][$row] %= other );*
            }
        }
    }
}

macro_rules! impl_matrix_matrix_mul_ops {
    ($MatrixMxN:ident, $MatrixNxK:ident => $Output:ident, $dot_arr_col:ident, { $( ($col:expr, $row:expr) ),* }) => {
        impl<S> ops::Mul<$MatrixNxK<S>> for $MatrixMxN<S> where S: Scalar {
            type Output = $Output<S>;

            #[inline]
            fn mul(self, other: $MatrixNxK<S>) -> Self::Output {
                Self::Output::new(
                    $( $dot_arr_col(&self.data, &other.data[$col], $row) ),*
                )
            }
        }

        impl<S> ops::Mul<&$MatrixNxK<S>> for $MatrixMxN<S> where S: Scalar {
            type Output = $Output<S>;

            #[inline]
            fn mul(self, other: &$MatrixNxK<S>) -> Self::Output {
                Self::Output::new(
                    $( $dot_arr_col(&self.data, &other.data[$col], $row) ),*
                )
            }
        }

        impl<S> ops::Mul<$MatrixNxK<S>> for &$MatrixMxN<S> where S: Scalar {
            type Output = $Output<S>;

            #[inline]
            fn mul(self, other: $MatrixNxK<S>) -> Self::Output {
                Self::Output::new(
                    $( $dot_arr_col(&self.data, &other.data[$col], $row) ),*
                )
            }
        }

        impl<'a, 'b, S> ops::Mul<&'a $MatrixNxK<S>> for &'b $MatrixMxN<S> where S: Scalar {
            type Output = $Output<S>;

            #[inline]
            fn mul(self, other: &'a $MatrixNxK<S>) -> Self::Output {
                Self::Output::new(
                    $( $dot_arr_col(&self.data, &other.data[$col], $row) ),*
                )
            }
        }
    }
}

macro_rules! impl_scalar_matrix_mul_ops1 {
    ($Lhs:ty, $Rhs:ty, $Output:ty, { $( ($col:expr, $row:expr) ),* }) => {
        impl ops::Mul<$Rhs> for $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                <$Output>::new( $(self * other.data[$col][$row]),* )
            }
        }

        impl<'a> ops::Mul<$Rhs> for &'a $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                <$Output>::new( $(self * other.data[$col][$row]),* )
            }
        }
    }
}


/// A Type synonym for `Matrix2x2`.
pub type Matrix2<S> = Matrix2x2<S>;

/// A Type synonym for `Matrix3x3`.
pub type Matrix3<S> = Matrix3x3<S>;

/// A Type synonym for `Matrix4x4`
pub type Matrix4<S> = Matrix4x4<S>;


/// The `Matrix2x2` type represents 2x2 matrices in column-major order.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Matrix2x2<S> {
    data: [[S; 2]; 2],
}

impl_coords!(View2x2, { c0r0, c0r1, c1r0, c1r1 });
impl_coords_deref!(Matrix2x2, View2x2);

impl<S> Matrix2x2<S> {
    /// Construct a new 2x2 matrix from its field elements.
    #[inline]
    pub const fn new(c0r0: S, c0r1: S, c1r0: S, c1r1: S) -> Matrix2x2<S> {
        Matrix2x2 {
            data: [
                [c0r0, c0r1],
                [c1r0, c1r1],
            ]
        }
    }

    /// Construct a 2x2 matrix from a pair of two-dimensional vectors.
    #[inline]
    pub fn from_columns(c0: Vector2<S>, c1: Vector2<S>) -> Matrix2x2<S> {
        Matrix2x2::new(
            c0.x, c0.y, 
            c1.x, c1.y
        )
    }
}

impl<S> Matrix2x2<S> where S: Copy {
    /// Construct a new matrix from a fill value.
    ///
    /// The resulting matrix is a matrix where each entry is the supplied fill
    /// value.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// #
    /// let fill_value: u32 = 3;
    /// let expected = Matrix2x2::new(fill_value, fill_value, fill_value, fill_value);
    /// let result = Matrix2x2::from_fill(fill_value);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Matrix2x2<S> {
        Matrix2x2::new(value, value, value, value)
    }

    /// Get the row of the matrix by value.
    #[inline]
    pub fn row(&self, r: usize) -> Vector2<S> {
        Vector2::new(self[0][r], self[1][r])
    }

    /// Get the column of the matrix by value
    #[inline]
    pub fn column(&self, c: usize) -> Vector2<S> {
        Vector2::new(self[c][0], self[c][1])
    }
    
    /// Swap two rows of a matrix.
    #[inline]
    pub fn swap_rows(&mut self, row_a: usize, row_b: usize) {
        let c0ra = self[0][row_a];
        let c1ra = self[1][row_a];
        self[0][row_a] = self[0][row_b];
        self[1][row_a] = self[1][row_b];
        self[0][row_b] = c0ra;
        self[1][row_b] = c1ra;
    }
    
     /// Swap two columns of a matrix.
    #[inline]
    pub fn swap_columns(&mut self, col_a: usize, col_b: usize) {
        let car0 = self[col_a][0];
        let car1 = self[col_a][1];
        self[col_a][0] = self[col_b][0];
        self[col_a][1] = self[col_b][1];
        self[col_b][0] = car0;
        self[col_b][1] = car1;
    }
    
    /// Swap two elements of a matrix.
    #[inline]
    pub fn swap(&mut self, a: (usize, usize), b: (usize, usize)) {
        let element_a = self[a.0][a.1];
        self[a.0][a.1] = self[b.0][b.1];
        self[b.0][b.1] = element_a;
    }

    /// The length of the the underlying array.
    #[inline]
    pub fn len() -> usize {
        4
    }

    /// The shape of the underlying array.
    #[inline]
    pub fn shape() -> (usize, usize) {
        (2, 2)
    }

    /// Generate a pointer to the underlying array.
    #[inline]
    pub fn as_ptr(&self) -> *const S {
        //&self.c0r0
        &self.data[0][0]
    }

    /// Generate a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        //&mut self.c0r0
        &mut self.data[0][0]
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        <Self as AsRef<[S; 4]>>::as_ref(self)
    }

    /// Map an operation on the elements of a matrix, returning a matrix whose 
    /// elements are elements of the new underlying type.
    #[inline]
    pub fn map<T, F>(&self, mut op: F) -> Matrix2x2<T> where F: FnMut(S) -> T {
        Matrix2x2 {
            data: [
                [op(self.data[0][0]), op(self.data[0][1])],
                [op(self.data[1][0]), op(self.data[1][1])]
            ],
        }
    }
}

impl<S> Matrix2x2<S> where S: NumCast + Copy {
    /// Cast a matrix from one type of scalars to another type of scalars.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,   
    /// # };
    /// # 
    /// let matrix: Matrix2x2<u32> = Matrix2x2::new(1_u32, 2_u32, 3_u32, 4_u32);
    /// let expected: Option<Matrix2x2<i32>> = Some(Matrix2x2::new(1_i32, 2_i32, 3_i32, 4_i32));
    /// let result = matrix.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Matrix2x2<T>> {
        let c0r0 = match num_traits::cast(self.data[0][0]) {
            Some(value) => value,
            None => return None,
        };
        let c0r1 = match num_traits::cast(self.data[0][1]) {
            Some(value) => value,
            None => return None,
        };
        let c1r0 = match num_traits::cast(self.data[1][0]) {
            Some(value) => value,
            None => return None,
        };
        let c1r1 = match num_traits::cast(self.data[1][1]) {
            Some(value) => value,
            None => return None,
        };

        Some(Matrix2x2::new(c0r0, c0r1, c1r0, c1r1))
    }
}

impl<S> Matrix2x2<S> where S: Scalar {
    /// Construct a shearing matrix along the x-axis, holding the **y-axis** constant.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the **y-axis** to shearing along the **x-axis**.
    ///
    /// ## Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// #     Vector2,
    /// # };
    /// #
    /// let shear_x_with_y = 3_u32;
    /// let matrix = Matrix2x2::from_shear_x(shear_x_with_y);
    /// let vector = Vector2::new(1, 1);
    /// let expected = Vector2::new(1 + 3*1, 1);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_x(shear_x_with_y: S) -> Matrix2x2<S> {
        Matrix2x2::new(
            S::one(),       S::zero(),
            shear_x_with_y, S::one(),
        )
    }

    /// Construct a shearing matrix along the y-axis, holding the **x-axis** constant.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the **x-axis** to shearing along the **y-axis**.
    ///
    /// ## Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// #     Vector2,
    /// # };
    /// #
    /// let shear_y_with_x = 3_u32;
    /// let matrix = Matrix2x2::from_shear_y(shear_y_with_x);
    /// let vector = Vector2::new(1, 1);
    /// let expected = Vector2::new(1, 1 + 3*1);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    /// contribution of the **x-axis** to shearing along the **y-axis**.
    /// The parameter `shear_x_with_y` denotes the factor scaling the 
    /// contribution of the `y`-component to the shearing of the `x`-component. 
    ///
    /// ## Example 
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// #     Vector2,
    /// # };
    /// #
    /// let shear_x_with_y = 15_u32;
    /// let shear_y_with_x = 4_u32;
    /// let matrix = Matrix2x2::from_shear(shear_x_with_y, shear_y_with_x);
    /// let vector = Vector2::new(1, 1);
    /// let expected = Vector2::new(1 + 15*1, 1 + 4*1);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    ///
    /// ## Example 
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// #     Vector2,
    /// # };
    /// #
    /// let scale = 11_u32;
    /// let matrix = Matrix2x2::from_scale(scale);
    /// let vector = Vector2::new(1, 2);
    /// let expected = Vector2::new(11, 22);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_scale(scale: S) -> Matrix2x2<S> {
        Matrix2x2::from_nonuniform_scale(scale, scale)
    }
        
    /// Construct two-dimensional general scaling matrix.
    ///
    /// This is the most general case for scaling matrices: the scale factor
    /// in each dimension need not be identical.
    ///
    /// ## Example 
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// #     Vector2,
    /// # };
    /// #
    /// let scale_x = 3_u32;
    /// let scale_y = 5_u32;
    /// let matrix = Matrix2x2::from_nonuniform_scale(scale_x, scale_y);
    /// let vector = Vector2::new(1, 2);
    /// let expected = Vector2::new(3, 10);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S) -> Matrix2x2<S> {
        let zero = S::zero();
        Matrix2x2::new(
            scale_x,   zero,
            zero,      scale_y,
        )
    }

    /// Mutably transpose a square matrix in place.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,    
    /// # };
    /// #
    /// let mut result = Matrix2x2::new(
    ///     1_i32, 1_i32,
    ///     2_i32, 2_i32 
    /// );
    /// let expected = Matrix2x2::new(
    ///     1_i32, 2_i32,
    ///     1_i32, 2_i32 
    /// );
    /// result.transpose_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn transpose_mut(&mut self) {
        self.swap((0, 1), (1, 0));
    }

    /// Transpose a matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// #
    /// let matrix = Matrix2x2::new(
    ///     1_i32, 1_i32,
    ///     2_i32, 2_i32
    /// );
    /// let expected = Matrix2x2::new(
    ///     1_i32, 2_i32,
    ///     1_i32, 2_i32
    /// );
    /// let result = matrix.transpose();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn transpose(&self) -> Matrix2x2<S> {
        Matrix2x2::new(
            self.data[0][0], self.data[1][0], 
            self.data[0][1], self.data[1][1]
        )
    }

    /// Compute a zero matrix.
    ///
    /// A zero matrix is a matrix in which all of its elements are zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// #
    /// let matrix: Matrix2x2<i32> = Matrix2x2::zero();
    ///
    /// assert!(matrix.is_zero());
    /// ```
    #[inline]
    pub fn zero() -> Matrix2x2<S> {
        Matrix2x2::new(S::zero(), S::zero(), S::zero(), S::zero())
    }
    
    /// Determine whether a matrix is a zero matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// #
    /// let matrix: Matrix2x2<i32> = Matrix2x2::zero();
    ///
    /// assert!(matrix.is_zero());
    /// ```
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.data[0][0].is_zero() && self.data[0][1].is_zero() &&
        self.data[1][0].is_zero() && self.data[1][1].is_zero()
    }

    /// Compute an identity matrix.
    ///
    /// An identity matrix is a matrix where the diagonal elements are one
    /// and the off-diagonal elements are zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// #
    /// let result: Matrix2x2<i32> = Matrix2x2::identity();
    /// let expected = Matrix2x2::new(
    ///     1_i32, 0_i32,
    ///     0_i32, 1_i32,  
    /// );
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn identity() -> Matrix2x2<S> {
        Matrix2x2::new(S::one(), S::zero(), S::zero(), S::one())
    }
    
    /// Determine whether a matrix is an identity matrix.
    ///
    /// An identity matrix is a matrix where the diagonal elements are one
    /// and the off-diagonal elements are zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// #
    /// let matrix: Matrix2x2<i32> = Matrix2x2::identity();
    /// 
    /// assert!(matrix.is_identity());
    /// ```
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.data[0][0].is_one()  && self.data[0][1].is_zero() &&
        self.data[1][0].is_zero() && self.data[1][1].is_one()
    }

    /// Construct a new diagonal matrix from a given value where
    /// each element along the diagonal is equal to `value`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// #
    /// let result = Matrix2x2::from_diagonal_value(2_i32);
    /// let expected = Matrix2x2::new(
    ///     2_i32, 0_i32,
    ///     0_i32, 2_i32 
    /// );
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_diagonal_value(value: S) -> Self {
        Matrix2x2::new(
            value,     S::zero(),
            S::zero(), value
        )
    }
    
    /// Construct a new diagonal matrix from a vector of values
    /// representing the elements along the diagonal.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// # };
    /// #
    /// let result = Matrix2x2::from_diagonal(&Vector2::new(2_i32, 3_i32));
    /// let expected = Matrix2x2::new(
    ///     2_i32, 0_i32,
    ///     0_i32, 3_i32 
    /// );
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_diagonal(diagonal: &Vector2<S>) -> Self {
        Matrix2x2::new(
            diagonal.x, S::zero(),
            S::zero(),  diagonal.y
        )
    }

    /// Get the diagonal part of a square matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// # };
    /// #
    /// let matrix = Matrix2x2::new(
    ///     1_i32, 2_i32,
    ///     3_i32, 4_i32,   
    /// );
    /// let expected = Vector2::new(1_i32, 4_i32);
    /// let result = matrix.diagonal();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn diagonal(&self) -> Vector2<S> {
        Vector2::new(self.data[0][0], self.data[1][1])
    }

    /// Compute the trace of a square matrix.
    ///
    /// The trace of a matrix is the sum of the diagonal elements.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// #
    /// let matrix = Matrix2x2::new(
    ///     1_i32, 2_i32,
    ///     3_i32, 4_i32 
    /// );
    ///
    /// assert_eq!(matrix.trace(), 5_i32);
    /// ```
    #[inline]
    pub fn trace(&self) -> S {
        self.data[0][0] + self.data[1][1]
    }
}

impl<S> Matrix2x2<S> where S: ScalarSigned {
    /// Construct a two-dimensional reflection matrix for reflecting through a 
    /// line through the origin in the **xy-plane**.
    ///
    /// ## Example
    ///
    /// Here is an example of reflecting a vector across the **x-axis**.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// #     Unit, 
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector2::unit_y());
    /// let matrix = Matrix2x2::from_reflection(&normal);
    /// let vector = Vector2::new(2_f64, 2_f64);
    /// let expected = Vector2::new(2_f64, -2_f64);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// In two dimensions there is an ambiguity in the choice of normal 
    /// vector, and as a result, a normal vector to the line---or 
    /// its negation---will produce the same reflection.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// #     Unit, 
    /// # };
    /// #
    /// let minus_normal = Unit::from_value(-Vector2::unit_y());
    /// let matrix = Matrix2x2::from_reflection(&minus_normal);
    /// let vector = Vector2::new(2_f64, 2_f64);
    /// let expected = Vector2::new(2_f64, -2_f64);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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

    /// Mutably negate the elements of a matrix in place.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// # 
    /// let mut result = Matrix2x2::new(
    ///     1_i32, 2_i32,
    ///     3_i32, 4_i32
    /// );
    /// let expected = Matrix2x2::new(
    ///     -1_i32, -2_i32,
    ///     -3_i32, -4_i32   
    /// );
    /// result.neg_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn neg_mut(&mut self) {
        self.data[0][0] = -self.data[0][0];
        self.data[0][1] = -self.data[0][1];
        self.data[1][0] = -self.data[1][0];
        self.data[1][1] = -self.data[1][1];
    }

    /// Compute the determinant of a matrix.
    /// 
    /// The determinant of a matrix is the signed volume of the parallelopiped
    /// swept out by the vectors represented by the matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// #
    /// let matrix = Matrix2x2::new(
    ///     1_f64, 3_f64,
    ///     2_f64, 4_f64 
    /// );
    ///
    /// assert_eq!(matrix.determinant(), -2_f64);
    /// ```
    #[inline]
    pub fn determinant(&self) -> S {
        self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
    }
}

impl<S> Matrix2x2<S> where S: ScalarFloat {
    /// Construct a rotation matrix in two-dimensions that rotates a vector
    /// in the **xy-plane** by an angle `angle`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,
    /// #     Radians,
    /// #     Angle,
    /// #     Vector2, 
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let unit_x = Vector2::unit_x();
    /// let unit_y = Vector2::unit_y();
    /// let matrix = Matrix2x2::from_angle(angle);
    /// let expected = unit_y;
    /// let result = matrix * unit_x;
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-10));
    /// ```
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
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,
    /// #     Vector2, 
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let v1 = Vector2::new(1_f64, 1_f64);
    /// let v2 = Vector2::new(-1_f64, 1_f64);
    /// let matrix = Matrix2x2::rotation_between(&v1, &v2);
    /// let vector = Vector2::unit_y();
    /// let expected = -Vector2::unit_x();
    /// let result = matrix * vector;
    ///
    /// assert!(relative_eq!(result, expected));
    /// ```
    /// The matrix returned by `rotation_between` should make `v1` and `v2` collinear.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,
    /// #     Vector2, 
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let v1 = Vector2::new(1_f64, 1_f64);
    /// let v2 = Vector2::new(-1_f64, 1_f64);
    /// let matrix = Matrix2x2::rotation_between(&v1, &v2);
    /// let result = matrix * v1;
    /// let expected = v2;
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-10));
    /// ```
    #[inline]
    pub fn rotation_between(v1: &Vector2<S>, v2: &Vector2<S>) -> Matrix2x2<S> {
        if let (Some(unit_v1), Some(unit_v2)) = (
            Unit::try_from_value(*v1, S::zero()),
            Unit::try_from_value(*v2, S::zero()),
        ) {
            Self::rotation_between_axis(&unit_v1, &unit_v2)
        } else {
            Self::identity()
        }
    }

    /// Construct a rotation matrix that rotates the shortest angular distance 
    /// between two unit vectors.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let v1 = Vector2::new(1_f64, 1_f64);
    /// let v2 = Vector2::new(-1_f64, 1_f64);
    /// let unit_v1 = Unit::from_value(v1);
    /// let unit_v2 = Unit::from_value(v2);
    /// let matrix = Matrix2x2::rotation_between_axis(&unit_v1, &unit_v2);
    /// let vector = Vector2::unit_y();
    /// let expected = -Vector2::unit_x();
    /// let result = matrix * vector;
    ///
    /// assert!(relative_eq!(result, expected));
    /// ```
    /// The matrix returned by `rotation_between` should make `v1` and `v2` collinear.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let v1 = Vector2::new(1_f64, 1_f64);
    /// let v2 = Vector2::new(-1_f64, 1_f64);
    /// let unit_v1 = Unit::from_value(v1);
    /// let unit_v2 = Unit::from_value(v2);
    /// let matrix = Matrix2x2::rotation_between_axis(&unit_v1, &unit_v2);
    /// let result = matrix * v1;
    /// let expected = v2;
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-10));
    /// ```
    #[inline]
    pub fn rotation_between_axis(v1: &Unit<Vector2<S>>, v2: &Unit<Vector2<S>>) -> Matrix2x2<S> {
        let cos_angle = v1.as_ref().dot(v2.as_ref());
        let sin_angle = S::sqrt(S::one() - cos_angle * cos_angle);

        Self::from_angle(Radians::atan2(sin_angle, cos_angle))
    }

    /// Linearly interpolate between two matrices.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,    
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let matrix0 = Matrix2x2::new(0_f64, 0_f64, 1_f64, 1_f64);
    /// let matrix1 = Matrix2x2::new(2_f64, 2_f64, 3_f64, 3_f64);
    /// let amount = 0.5;
    /// let expected = Matrix2x2::new(1_f64, 1_f64, 2_f64, 2_f64);
    /// let result = matrix0.lerp(&matrix1, amount);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Matrix2x2<S>, amount: S) -> Matrix2x2<S> {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of a matrix are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A matrix is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values.
    ///
    /// ## Example (Finite Matrix)
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// #
    /// let matrix = Matrix2x2::new(1_f64, 2_f64, 3_f64, 4_f64);
    ///
    /// assert!(matrix.is_finite());
    /// ```
    ///
    /// ## Example (Not A Finite Matrix)
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// #
    /// let matrix = Matrix2x2::new(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_f64);
    ///
    /// assert!(!matrix.is_finite());
    /// ```
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.data[0][0].is_finite() && self.data[0][1].is_finite() &&
        self.data[1][0].is_finite() && self.data[1][1].is_finite()
    }

    /// Compute the inverse of a square matrix, if the inverse exists. 
    ///
    /// Given a square matrix `self` Compute the matrix `m` if it exists 
    /// such that
    /// ```text
    /// m * self = self * m = 1.
    /// ```
    /// Not every square matrix has an inverse.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2,  
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let matrix = Matrix2x2::new(
    ///     2_f64, 3_f64,
    ///     1_f64, 5_f64 
    /// );
    /// let expected = Matrix2x2::new(
    ///      5_f64 / 7_f64, -3_f64 / 7_f64,
    ///     -1_f64 / 7_f64,  2_f64 / 7_f64
    /// );
    /// let result = matrix.inverse().unwrap();
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.is_zero() {
            None
        } else {
            let inv_det = S::one() / det;
            Some(Matrix2x2::new(
                inv_det *  self.data[1][1], inv_det * -self.data[0][1],
                inv_det * -self.data[1][0], inv_det *  self.data[0][0]
            ))
        }
    }

    /// Determine whether a square matrix has an inverse matrix.
    ///
    /// A matrix is invertible is its determinant is not zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix2x2, 
    /// # };
    /// #
    /// let matrix = Matrix2x2::new(
    ///     1_f64, 2_f64,
    ///     2_f64, 1_f64   
    /// );
    /// 
    /// assert_eq!(matrix.determinant(), -3_f64);
    /// assert!(matrix.is_invertible());
    /// ```
    #[inline]
    pub fn is_invertible(&self) -> bool {
        ulps_ne!(self.determinant(), S::zero())
    }

    /// Determine whether a square matrix is a diagonal matrix. 
    ///
    /// A square matrix is a diagonal matrix if every off-diagonal 
    /// element is zero.
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        ulps_eq!(self.data[0][1], S::zero()) && ulps_eq!(self.data[1][0], S::zero())
    }
    
    /// Determine whether a matrix is symmetric. 
    ///
    /// A matrix is symmetric when element `(i, j)` is equal to element `(j, i)` 
    /// for each row `i` and column `j`. Otherwise, it is not a symmetric matrix. 
    /// Note that every diagonal matrix is a symmetric matrix.
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        ulps_eq!(self.data[0][1], self.data[1][0]) && ulps_eq!(self.data[1][0], self.data[0][1])
    }
}

impl<S> fmt::Display for Matrix2x2<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            formatter, 
            "Matrix2x2 [[{}, {}], [{}, {}]]", 
            self.data[0][0], self.data[1][0],
            self.data[0][1], self.data[1][1],
        )
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


impl_as_ref_ops!(Matrix2x2<S>, [S; 4]);
impl_as_ref_ops!(Matrix2x2<S>, [[S; 2]; 2]);
impl_as_ref_ops!(Matrix2x2<S>, [Vector2<S>; 2]);

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

impl<S> ops::Index<(usize, usize)> for Matrix2x2<S>{
    type Output = S;

    #[inline]
    fn index(&self, (column, row): (usize, usize)) -> &Self::Output {
        let v: &[[S; 2]; 2] = self.as_ref();
        &v[column][row]
    }
}

impl<S> ops::IndexMut<(usize, usize)> for Matrix2x2<S> {
    #[inline]
    fn index_mut(&mut self, (column, row): (usize, usize)) -> &mut S {
        let v: &mut [[S; 2]; 2] = self.as_mut();
        &mut v[column][row]
    }
}

impl_matrix_matrix_mul_ops!(
    Matrix2x2, Matrix2x2 => Matrix2x2, dot_array2x2_col2,
    { (0, 0), (0, 1), (1, 0), (1, 1) }
);

impl<S> ops::Mul<Vector2<S>> for Matrix2x2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, other: Vector2<S>) -> Self::Output {
        let x = self.data[0][0] * other[0] + self.data[1][0] * other[1];
        let y = self.data[0][1] * other[0] + self.data[1][1] * other[1];

        Vector2::new(x, y)
    }
}

impl<S> ops::Mul<&Vector2<S>> for Matrix2x2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, other: &Vector2<S>) -> Self::Output {
        let x = self.data[0][0] * other[0] + self.data[1][0] * other[1];
        let y = self.data[0][1] * other[0] + self.data[1][1] * other[1];

        Vector2::new(x, y)
    }
}

impl<S> ops::Mul<Vector2<S>> for &Matrix2x2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, other: Vector2<S>) -> Self::Output {
        let x = self.data[0][0] * other[0] + self.data[1][0] * other[1];
        let y = self.data[0][1] * other[0] + self.data[1][1] * other[1];

        Vector2::new(x, y)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Vector2<S>> for &'b Matrix2x2<S> where S: Scalar {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, other: &'a Vector2<S>) -> Self::Output {
        let x = self.data[0][0] * other[0] + self.data[1][0] * other[1];
        let y = self.data[0][1] * other[0] + self.data[1][1] * other[1];

        Vector2::new(x, y)
    }
}

impl_matrix_matrix_binary_ops1!(
    Add, add, 
    add_array2x2_array2x2, Matrix2x2<S>, Matrix2x2<S>, 
    { (0, 0), (0, 1), (1, 0), (1, 1) }
);
impl_matrix_matrix_binary_ops1!(
    Sub, sub, sub_array2x2_array2x2, Matrix2x2<S>, Matrix2x2<S>, 
    { (0, 0), (0, 1), (1, 0), (1, 1) }
);
impl_matrix_unary_ops1!(
    Neg, neg, neg_array2x2, Matrix2x2<S>, Matrix2x2<S>,
    { (0, 0), (0, 1), (1, 0), (1, 1) }
);

impl_matrix_scalar_binary_ops1!(
    Mul, mul, mul_array2x2_scalar, Matrix2x2<S>, Matrix2x2<S>, 
    { (0, 0), (0, 1), (1, 0), (1, 1) }
);
impl_matrix_scalar_binary_ops1!(
    Div, div, div_array2x2_scalar, Matrix2x2<S>, Matrix2x2<S>, 
    { (0, 0), (0, 1), (1, 0), (1, 1) }
);
impl_matrix_scalar_binary_ops1!(
    Rem, rem, rem_array2x2_scalar, Matrix2x2<S>, Matrix2x2<S>, 
    { (0, 0), (0, 1), (1, 0), (1, 1) }
);

impl_matrix_binary_assign_ops1!(
    Matrix2x2<S>, { (0, 0), (0, 1), (1, 0), (1, 1) }
);

impl_scalar_matrix_mul_ops1!(u8,    Matrix2x2<u8>,    Matrix2x2<u8>,    { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(u16,   Matrix2x2<u16>,   Matrix2x2<u16>,   { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(u32,   Matrix2x2<u32>,   Matrix2x2<u32>,   { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(u64,   Matrix2x2<u64>,   Matrix2x2<u64>,   { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(u128,  Matrix2x2<u128>,  Matrix2x2<u128>,  { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(usize, Matrix2x2<usize>, Matrix2x2<usize>, { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(i8,    Matrix2x2<i8>,    Matrix2x2<i8>,    { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(i16,   Matrix2x2<i16>,   Matrix2x2<i16>,   { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(i32,   Matrix2x2<i32>,   Matrix2x2<i32>,   { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(i64,   Matrix2x2<i64>,   Matrix2x2<i64>,   { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(i128,  Matrix2x2<i128>,  Matrix2x2<i128>,  { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(isize, Matrix2x2<isize>, Matrix2x2<isize>, { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(f32,   Matrix2x2<f32>,   Matrix2x2<f32>,   { (0, 0), (0, 1), (1, 0), (1, 1) });
impl_scalar_matrix_mul_ops1!(f64,   Matrix2x2<f64>,   Matrix2x2<f64>,   { (0, 0), (0, 1), (1, 0), (1, 1) });


impl<S> approx::AbsDiffEq for Matrix2x2<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.data[0][0], &other.data[0][0], epsilon) && 
        S::abs_diff_eq(&self.data[0][1], &other.data[0][1], epsilon) &&
        S::abs_diff_eq(&self.data[1][0], &other.data[1][0], epsilon) && 
        S::abs_diff_eq(&self.data[1][1], &other.data[1][1], epsilon)
    }
}

impl<S> approx::RelativeEq for Matrix2x2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.data[0][0], &other.data[0][0], epsilon, max_relative) &&
        S::relative_eq(&self.data[0][1], &other.data[0][1], epsilon, max_relative) &&
        S::relative_eq(&self.data[1][0], &other.data[1][0], epsilon, max_relative) &&
        S::relative_eq(&self.data[1][1], &other.data[1][1], epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Matrix2x2<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.data[0][0], &other.data[0][0], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[0][1], &other.data[0][1], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[1][0], &other.data[1][0], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[1][1], &other.data[1][1], epsilon, max_ulps)
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
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Matrix3x3<S> {
    data: [[S; 3]; 3],
}

impl_coords!(View3x3, { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_coords_deref!(Matrix3x3, View3x3);

impl<S> Matrix3x3<S> {
    /// Construct a new 3x3 matrix.
    #[rustfmt::skip]
    #[inline]
    pub const fn new(
        c0r0: S, c0r1: S, c0r2: S,
        c1r0: S, c1r1: S, c1r2: S,
        c2r0: S, c2r1: S, c2r2: S) -> Matrix3x3<S> {

        Matrix3x3 {
            data: [
                [c0r0, c0r1, c0r2],
                [c1r0, c1r1, c1r2],
                [c2r0, c2r1, c2r2],
            ]
        }
    }

    /// Create a 3x3 matrix from a triple of three-dimensional column vectors.
    #[rustfmt::skip]
    #[inline]
    pub fn from_columns(c0: Vector3<S>, c1: Vector3<S>, c2: Vector3<S>) -> Matrix3x3<S> {
        Matrix3x3::new(
            c0.x, c0.y, c0.z, 
            c1.x, c1.y, c1.z,
            c2.x, c2.y, c2.z,
        )
    }
}

impl<S> Matrix3x3<S> where S: Copy {
    /// Construct a new matrix from a fill value.
    ///
    /// The resulting matrix is a matrix where each entry is the supplied fill
    /// value.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,  
    /// # };
    /// #
    /// let fill_value = 3_u32;
    /// let expected = Matrix3x3::new(
    ///     fill_value, fill_value, fill_value,
    ///     fill_value, fill_value, fill_value,
    ///     fill_value, fill_value, fill_value
    /// );
    /// let result = Matrix3x3::from_fill(fill_value);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Matrix3x3<S> {
        Matrix3x3::new(
            value, value, value,
            value, value, value,
            value, value, value
        )
    }

    /// Get the row of the matrix by value.
    #[inline]
    pub fn row(&self, r: usize) -> Vector3<S> {
        Vector3::new(self[0][r], self[1][r], self[2][r])
    }

    /// Get the column of the matrix by value.
    #[inline]
    pub fn column(&self, c: usize) -> Vector3<S> {
        Vector3::new(self[c][0], self[c][1], self[c][2])
    }
    
    /// Swap two rows of a matrix.
    #[inline]
    pub fn swap_rows(&mut self, row_a: usize, row_b: usize) {
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
    
    /// Swap two columns of a matrix.
    #[inline]
    pub fn swap_columns(&mut self, col_a: usize, col_b: usize) {
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
    
    /// Swap two elements of a matrix.
    #[inline]
    pub fn swap(&mut self, a: (usize, usize), b: (usize, usize)) {
        let element_a = self[a.0][a.1];
        self[a.0][a.1] = self[b.0][b.1];
        self[b.0][b.1] = element_a;
    }

    /// The length of the the underlying array.
    #[inline]
    pub fn len() -> usize {
        9
    }

    /// The shape of the underlying array.
    #[inline]
    pub fn shape() -> (usize, usize) {
        (3, 3)
    }

    /// Generate a pointer to the underlying array.
    #[inline]
    pub fn as_ptr(&self) -> *const S {
        &self.data[0][0]
    }

    /// Generate a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        &mut self.data[0][0]
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        <Self as AsRef<[S; 9]>>::as_ref(self)
    }

    /// Map an operation on the elements of a matrix, returning a matrix whose 
    /// elements are elements of the new underlying type.
    #[rustfmt::skip]
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Matrix3x3<T> where F: FnMut(S) -> T {
        Matrix3x3::new(
            op(self.data[0][0]), 
            op(self.data[0][1]), 
            op(self.data[0][2]),
            op(self.data[1][0]), 
            op(self.data[1][1]), 
            op(self.data[1][2]),
            op(self.data[2][0]), 
            op(self.data[2][1]), 
            op(self.data[2][2]),
        )
    }
}

impl<S> Matrix3x3<S> where S: NumCast + Copy {
    /// Cast a matrix from one type of scalars to another type of scalars.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,   
    /// # };
    /// # 
    /// let matrix: Matrix3x3<u32> = Matrix3x3::new(
    ///     1_u32, 2_u32, 3_u32, 
    ///     4_u32, 5_u32, 6_u32,
    ///     7_u32, 8_u32, 9_u32
    /// );
    /// let expected: Option<Matrix3x3<i32>> = Some(Matrix3x3::new(
    ///     1_i32, 2_i32, 3_i32, 
    ///     4_i32, 5_i32, 6_i32,
    ///     7_i32, 8_i32, 9_i32
    /// ));
    /// let result = matrix.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Matrix3x3<T>> {
        let c0r0 = match num_traits::cast(self.data[0][0]) {
            Some(value) => value,
            None => return None,
        };
        let c0r1 = match num_traits::cast(self.data[0][1]) {
            Some(value) => value,
            None => return None,
        };
        let c0r2 = match num_traits::cast(self.data[0][2]) {
            Some(value) => value,
            None => return None,
        };
        let c1r0 = match num_traits::cast(self.data[1][0]) {
            Some(value) => value,
            None => return None,
        };
        let c1r1 = match num_traits::cast(self.data[1][1]) {
            Some(value) => value,
            None => return None,
        };
        let c1r2 = match num_traits::cast(self.data[1][2]) {
            Some(value) => value,
            None => return None,
        };
        let c2r0 = match num_traits::cast(self.data[2][0]) {
            Some(value) => value,
            None => return None,
        };
        let c2r1 = match num_traits::cast(self.data[2][1]) {
            Some(value) => value,
            None => return None,
        };
        let c2r2 = match num_traits::cast(self.data[2][2]) {
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
    /// This represents a translation in the **xy-plane** as an affine 
    /// transformation that displaces a vector along the length of the vector
    /// `distance`.
    ///
    /// ## Example
    /// A homogeneous vector with a zero `z`-component should not translate.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector2,
    /// #     Vector3,
    /// # };
    /// #
    /// let distance = Vector2::new(3_u32, 7_u32);
    /// let matrix = Matrix3x3::from_affine_translation(&distance);
    /// let vector = Vector3::new(1_u32, 1_u32, 0_u32);
    /// let expected = Vector3::new(1_u32, 1_u32, 0_u32);
    /// let result = matrix * vector;
    /// 
    /// assert_eq!(result, expected); 
    /// ```
    /// A homogeneous vector with a unit `z`-component should translate.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector2,
    /// #     Vector3,
    /// # };
    /// #
    /// let distance = Vector2::new(3_u32, 7_u32);
    /// let matrix = Matrix3x3::from_affine_translation(&distance);
    /// let vector = Vector3::new(1_u32, 1_u32, 1_u32);
    /// let expected = Vector3::new(1_u32 + distance.x, 1_u32 + distance.y, 1_u32);
    /// let result = matrix * vector;
    /// 
    /// assert_eq!(result, expected); 
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_translation(distance: &Vector2<S>) -> Matrix3x3<S> {
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
    ///
    /// ## Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3,  
    /// # };
    /// #
    /// let scale = 5_i32;
    /// let vector = Vector3::new(1_i32, 2_i32, 3_i32);
    /// let matrix = Matrix3x3::from_scale(scale);
    /// let expected = Vector3::new(5_i32, 10_i32, 15_i32);
    /// let result = matrix * vector;
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_scale(scale: S) -> Matrix3x3<S> {
        Matrix3x3::from_nonuniform_scale(scale, scale, scale)
    }
    
    /// Construct a three-dimensional general scaling matrix.
    ///
    /// This is the most general case for scaling matrices: the scale factor
    /// in each dimension need not be identical.
    ///
    /// ## Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3,  
    /// # };
    /// #
    /// let scale_x = 5_i32;
    /// let scale_y = 10_i32;
    /// let scale_z = 15_i32;
    /// let vector = Vector3::new(1_i32, 1_i32, 1_i32);
    /// let matrix = Matrix3x3::from_nonuniform_scale(scale_x, scale_y, scale_z);
    /// let expected = Vector3::new(5_i32, 10_i32, 15_i32);
    /// let result = matrix * vector;
    /// 
    /// assert_eq!(result, expected);
    /// ```
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
    /// `from_affine_nonuniform_scale(scale, scale)`. The `z`-component is 
    /// unaffected since this is an affine matrix.
    ///
    /// ## Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3,  
    /// # };
    /// #
    /// let scale = 5_i32;
    /// let vector = Vector3::new(1_i32, 2_i32, 3_i32);
    /// let matrix = Matrix3x3::from_affine_scale(scale);
    /// let expected = Vector3::new(5_i32, 10_i32, 3_i32);
    /// let result = matrix * vector;
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_affine_scale(scale: S) -> Matrix3x3<S> {
        Matrix3x3::from_affine_nonuniform_scale(scale, scale)
    }
    
    /// Construct a two-dimensional affine scaling matrix.
    ///
    /// This is the most general case for scaling matrices: the scale factor
    /// in each dimension need not be identical. The `z`-component is unaffected 
    /// because this is an affine matrix.
    ///
    /// ## Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3,  
    /// # };
    /// #
    /// let scale_x = 5_i32;
    /// let scale_y = 10_i32;
    /// let vector = Vector3::new(1_i32, 1_i32, 3_i32);
    /// let matrix = Matrix3x3::from_affine_nonuniform_scale(scale_x, scale_y);
    /// let expected = Vector3::new(5_i32, 10_i32, 3_i32);
    /// let result = matrix * vector;
    /// 
    /// assert_eq!(result, expected);
    /// ```
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
    /// **x-axis**, holding the **y-axis** constant and the **z-axis** constant.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` are the 
    /// multiplicative factors for the contributions of the **y-axis** and the 
    /// **z-axis**, respectively to shearing along the **x-axis**.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3, 
    /// # };
    /// #
    /// let shear_x_with_y = 3_i32;
    /// let shear_x_with_z = 8_i32;
    /// let matrix = Matrix3x3::from_shear_x(shear_x_with_y, shear_x_with_z);
    /// let vector = Vector3::new(1_i32, 1_i32, 1_i32);
    /// let expected = Vector3::new(12_i32, 1_i32, 1_i32);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    /// **y-axis**, holding the **x-axis** constant and the **z-axis** constant.
    ///
    /// The parameters `shear_y_with_x` and `shear_y_with_z` are the
    /// multiplicative factors for the contributions of the **x-axis**, and the 
    /// **z-axis**, respectively to shearing along the **y-axis**.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3, 
    /// # };
    /// #
    /// let shear_y_with_x = 3_i32;
    /// let shear_y_with_z = 8_i32;
    /// let matrix = Matrix3x3::from_shear_y(shear_y_with_x, shear_y_with_z);
    /// let vector = Vector3::new(1_i32, 1_i32, 1_i32);
    /// let expected = Vector3::new(1_i32, 12_i32, 1_i32);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    /// **z-axis**, holding the **x-axis** constant and the **y-axis** constant.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` are the multiplicative
    /// factors for the contributions of the **x-axis**, and the **y-axis**, 
    /// respectively to shearing along the **z-axis**. 
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3, 
    /// # };
    /// #
    /// let shear_z_with_x = 3_i32;
    /// let shear_z_with_y = 8_i32;
    /// let matrix = Matrix3x3::from_shear_z(shear_z_with_x, shear_z_with_y);
    /// let vector = Vector3::new(1_i32, 1_i32, 1_i32);
    /// let expected = Vector3::new(1_i32, 1_i32, 12_i32);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    /// contribution of the `y`-component to shearing of the `x`-component.
    ///
    /// The parameter `shear_x_with_z` denotes the factor scaling the 
    /// contribution  of the `z`-component to the shearing of the `x`-component.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the `x`-component to shearing of the `y`-component.
    ///
    /// The parameter `shear_y_with_z` denotes the factor scaling the 
    /// contribution of the **z-axis** to the shearing of the `y`-component. 
    ///
    /// The parameter `shear_z_with_x` denotes the factor scaling the
    /// contribution of the **x-axis** to shearing of the **z-axis**.
    ///
    /// The parameter `shear_z_with_y` denotes the factor scaling the 
    /// contribution of the `y`-component to the shearing of the `z`-component. 
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3, 
    /// # };
    /// #
    /// let shear_x_with_y = 1_usize;
    /// let shear_x_with_z = 2_usize;
    /// let shear_y_with_x = 3_usize;
    /// let shear_y_with_z = 4_usize;
    /// let shear_z_with_x = 5_usize;
    /// let shear_z_with_y = 6_usize;
    /// let matrix = Matrix3x3::from_shear(
    ///     shear_x_with_y, 
    ///     shear_x_with_z,
    ///     shear_y_with_x,
    ///     shear_y_with_z,
    ///     shear_z_with_x,
    ///     shear_z_with_y,
    /// );
    /// let vector = Vector3::new(1_usize, 1_usize, 1_usize);
    /// let expected = Vector3::new(
    ///     vector.x + shear_x_with_y * vector.y + shear_x_with_z * vector.z,
    ///     vector.y + shear_y_with_x * vector.x + shear_y_with_z * vector.z,
    ///     vector.z + shear_z_with_x * vector.x + shear_z_with_y * vector.y
    /// );
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ``` 
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
    /// **x-axis**, holding the **y-axis** constant.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the **y-axis** to shearing along the **x-axis**.
    ///
    /// ## Example 
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_x_with_y = 3_u32;
    /// let matrix = Matrix3x3::from_affine_shear_x(shear_x_with_y);
    /// let vector = Vector3::new(1, 1, 0);
    /// let expected = Vector3::new(4, 1, 0);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    /// **y-axis**, holding the **x-axis** constant.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the **y-axis** to shearing along the **x-axis**.
    ///
    /// ## Example 
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_y_with_x = 3_u32;
    /// let matrix = Matrix3x3::from_affine_shear_y(shear_y_with_x);
    /// let vector = Vector3::new(1, 1, 0);
    /// let expected = Vector3::new(1, 4, 0);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    /// contribution of the **x-axis** to shearing along the **y-axis**.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the contribution 
    /// of the **y-axis** to the shearing along the **x-axis**.
    ///
    /// ## Example 
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_x_with_y = 15_u32;
    /// let shear_y_with_x = 4_u32;
    /// let matrix = Matrix3x3::from_affine_shear(shear_x_with_y, shear_y_with_x);
    /// let vector = Vector3::new(1, 1, 0);
    /// let expected = Vector3::new(16, 5, 0);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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

    /// Mutably transpose a square matrix in place.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let mut result = Matrix3x3::new(
    ///     1_i32, 1_i32, 1_i32,
    ///     2_i32, 2_i32, 2_i32,
    ///     3_i32, 3_i32, 3_i32,   
    /// );
    /// let expected = Matrix3x3::new(
    ///     1_i32, 2_i32, 3_i32,
    ///     1_i32, 2_i32, 3_i32,
    ///     1_i32, 2_i32, 3_i32
    /// );
    /// result.transpose_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn transpose_mut(&mut self) {
        self.swap((0, 1), (1, 0));
        self.swap((0, 2), (2, 0));
        self.swap((1, 2), (2, 1));
    }

    /// Transpose a matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_i32, 1_i32, 1_i32,
    ///     2_i32, 2_i32, 2_i32,
    ///     3_i32, 3_i32, 3_i32
    /// );
    /// let expected = Matrix3x3::new(
    ///     1_i32, 2_i32, 3_i32,
    ///     1_i32, 2_i32, 3_i32,
    ///     1_i32, 2_i32, 3_i32, 
    /// );
    /// let result = matrix.transpose();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn transpose(&self) -> Matrix3x3<S> {
        Matrix3x3::new(
            self.data[0][0], self.data[1][0], self.data[2][0],
            self.data[0][1], self.data[1][1], self.data[2][1],
            self.data[0][2], self.data[1][2], self.data[2][2]
        )
    }

    /// Compute a zero matrix.
    ///
    /// A zero matrix is a matrix in which all of its elements are zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let matrix: Matrix3x3<i32> = Matrix3x3::zero();
    ///
    /// assert!(matrix.is_zero());
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn zero() -> Matrix3x3<S> {
            let zero = S::zero();
            Matrix3x3::new(
                zero, zero, zero, 
                zero, zero, zero, 
                zero, zero, zero
            )
    }
    
    /// Determine whether a matrix is a zero matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let matrix: Matrix3x3<i32> = Matrix3x3::zero();
    ///
    /// assert!(matrix.is_zero());
    /// ```
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.data[0][0].is_zero() && 
        self.data[0][1].is_zero() && 
        self.data[0][2].is_zero() &&
        self.data[1][0].is_zero() && 
        self.data[1][1].is_zero() && 
        self.data[1][2].is_zero() &&
        self.data[2][0].is_zero() && 
        self.data[2][1].is_zero() && 
        self.data[2][2].is_zero()
    }
    
    /// Compute an identity matrix.
    ///
    /// An identity matrix is a matrix where the diagonal elements are one
    /// and the off-diagonal elements are zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let result = Matrix3x3::identity();
    /// let expected = Matrix3x3::new(
    ///     1_i32, 0_i32, 0_i32,
    ///     0_i32, 1_i32, 0_i32,
    ///     0_i32, 0_i32, 1_i32
    /// );
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn identity() -> Matrix3x3<S> {
        let zero = S::zero();
        let one = S::one();

        Matrix3x3::new(
            one,  zero, zero, 
            zero, one,  zero, 
            zero, zero, one
        )
    }
    
    /// Determine whether a matrix is an identity matrix.
    ///
    /// An identity matrix is a matrix where the diagonal elements are one
    /// and the off-diagonal elements are zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let matrix: Matrix3x3<i32> = Matrix3x3::identity();
    /// 
    /// assert!(matrix.is_identity());
    /// ```
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.data[0][0].is_one()  && self.data[0][1].is_zero() && self.data[0][2].is_zero() &&
        self.data[1][0].is_zero() && self.data[1][1].is_one()  && self.data[1][2].is_zero() &&
        self.data[2][0].is_zero() && self.data[2][1].is_zero() && self.data[2][2].is_one()
    }

    /// Construct a new diagonal matrix from a given value where
    /// each element along the diagonal is equal to `value`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let result = Matrix3x3::from_diagonal_value(3_i32);
    /// let expected = Matrix3x3::new(
    ///     3_i32, 0_i32, 0_i32,
    ///     0_i32, 3_i32, 0_i32,
    ///     0_i32, 0_i32, 3_i32
    /// );
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_diagonal_value(value: S) -> Self {
        Matrix3x3::new(
            value,     S::zero(), S::zero(),
            S::zero(), value,     S::zero(),
            S::zero(), S::zero(), value,
        )
    }
    
    /// Construct a new diagonal matrix from a given value where
    /// each element along the diagonal is equal to `value`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let result = Matrix3x3::from_diagonal(&Vector3::new(2_i32, 3_i32, 4_i32));
    /// let expected = Matrix3x3::new(
    ///     2_i32, 0_i32, 0_i32,
    ///     0_i32, 3_i32, 0_i32,
    ///     0_i32, 0_i32, 4_i32, 
    /// );
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_diagonal(diagonal: &Vector3<S>) -> Self {
        Matrix3x3::new(
            diagonal.x, S::zero(),  S::zero(),
            S::zero(),  diagonal.y, S::zero(),
            S::zero(),  S::zero(),  diagonal.z
        )
    }

    /// Get the diagonal part of a square matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_i32, 2_i32, 3_i32,
    ///     4_i32, 5_i32, 6_i32,
    ///     7_i32, 8_i32, 9_i32
    /// );
    /// let expected = Vector3::new(1_i32, 5_i32, 9_i32);
    /// let result = matrix.diagonal();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn diagonal(&self) -> Vector3<S> {
        Vector3::new(self.data[0][0], self.data[1][1], self.data[2][2])
    }

    /// Compute the trace of a square matrix.
    ///
    /// The trace of a matrix is the sum of the diagonal elements.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_i32, 2_i32, 3_i32,
    ///     4_i32, 5_i32, 6_i32,
    ///     7_i32, 8_i32, 9_i32 
    /// );
    ///
    /// assert_eq!(matrix.trace(), 15_i32);
    /// ```
    #[inline]
    pub fn trace(&self) -> S {
        self.data[0][0] + self.data[1][1] + self.data[2][2]
    }
}

impl<S> Matrix3x3<S> where S: ScalarSigned {
    /// Construct a two-dimensional affine reflection matrix in the **xy-plane** 
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
    ///
    /// ## Example (Line Through The Origin)
    ///
    /// Here is an example of reflecting a vector across the **x-axis** with 
    /// the line of reflection passing through the origin.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Vector2,
    /// #     Unit, 
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector2::unit_y());
    /// let bias = Vector2::new(0_f64, 0_f64);
    /// let matrix = Matrix3x3::from_affine_reflection(&normal, &bias);
    /// let vector = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let expected = Vector3::new(2_f64, -2_f64, 0_f64);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// In two dimensions there is an ambiguity in the choice of normal 
    /// vector, and as a result, a normal vector to the line---or 
    /// its negation---will produce the same reflection.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Vector2,
    /// #     Unit, 
    /// # };
    /// #
    /// let minus_normal = Unit::from_value(-Vector2::unit_y());
    /// let bias = Vector2::new(0_f64, 0_f64);
    /// let matrix = Matrix3x3::from_affine_reflection(&minus_normal, &bias);
    /// let vector = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let expected = Vector3::new(2_f64, -2_f64, 0_f64);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// ## Example (Line That Does Not Cross The Origin)
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Vector2, 
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     relative_eq,  
    /// # };
    /// #
    /// let bias = Vector2::new(0.0, 2.0);
    /// let normal = Unit::from_value(
    ///     Vector2::new(-1.0 / f64::sqrt(5.0), 2.0 / f64::sqrt(5.0))
    /// );
    /// let matrix = Matrix3x3::from_affine_reflection(&normal, &bias);
    /// let vector = Vector3::new(1.0, 0.0, 1.0);
    /// let expected = Vector3::new(-1.0, 4.0, 1.0);
    /// let result = matrix * vector;
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let expected = Matrix3x3::new(
    ///     1.0, 0.0,  0.0, 
    ///     0.0, 1.0,  0.0,  
    ///     0.0, 0.0, -1.0
    /// );
    /// let result = Matrix3x3::from_reflection(&normal);
    ///
    /// assert_eq!(result, expected);
    /// ```
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

    /// Mutably negate the elements of a matrix in place.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// # 
    /// let mut result = Matrix3x3::new(
    ///     1_i32, 2_i32, 3_i32,
    ///     4_i32, 5_i32, 6_i32,
    ///     7_i32, 8_i32, 9_i32
    /// );
    /// let expected = Matrix3x3::new(
    ///     -1_i32, -2_i32, -3_i32,
    ///     -4_i32, -5_i32, -6_i32,
    ///     -7_i32, -8_i32, -9_i32   
    /// );
    /// result.neg_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn neg_mut(&mut self) {
        self.data[0][0] = -self.data[0][0];
        self.data[0][1] = -self.data[0][1];
        self.data[0][2] = -self.data[0][2];
        self.data[1][0] = -self.data[1][0];
        self.data[1][1] = -self.data[1][1];
        self.data[1][2] = -self.data[1][2];
        self.data[2][0] = -self.data[2][0];
        self.data[2][1] = -self.data[2][1];
        self.data[2][2] = -self.data[2][2];
    }

    /// Compute the determinant of a matrix.
    /// 
    /// The determinant of a matrix is the signed volume of the parallelopiped
    /// swept out by the vectors represented by the matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_f64, 4_f64, 7_f64,
    ///     2_f64, 5_f64, 8_f64,
    ///     3_f64, 6_f64, 9_f64
    /// );
    ///
    /// assert_eq!(matrix.determinant(), 0_f64);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn determinant(&self) -> S {
        self.data[0][0] * self.data[1][1] * self.data[2][2] - 
        self.data[0][0] * self.data[1][2] * self.data[2][1] -
        self.data[1][0] * self.data[0][1] * self.data[2][2] + 
        self.data[1][0] * self.data[0][2] * self.data[2][1] +
        self.data[2][0] * self.data[0][1] * self.data[1][2] - 
        self.data[2][0] * self.data[0][2] * self.data[1][1]
    }
}

impl<S> Matrix3x3<S> where S: ScalarFloat {
    /// Construct an affine rotation matrix in two dimensions that rotates a 
    /// vector in the **xy-plane** by an angle `angle`.
    ///
    /// This is the affine matrix counterpart to the 2x2 matrix function 
    /// `from_angle`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Angle,
    /// #     Radians,
    /// #     Vector3, 
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_affine_angle(angle);
    /// let unit_x = Vector3::unit_x();
    /// let expected = Vector3::unit_y();
    /// let result = matrix * unit_x;
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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

    /// Construct a rotation matrix about the **x-axis** by an angle `angle`.
    ///
    /// ## Example
    /// 
    /// In this example the rotation is in the **yz-plane**.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Angle,
    /// #     Radians,
    /// #     Vector3, 
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_angle_x(angle);
    /// let vector = Vector3::new(0_f64, 1_f64, 1_f64);
    /// let expected = Vector3::new(0_f64, -1_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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

    /// Construct a rotation matrix about the **y-axis** by an angle `angle`.
    ///
    /// ## Example
    /// 
    /// In this example the rotation is in the **zx-plane**.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Angle,
    /// #     Radians,
    /// #     Vector3, 
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_angle_y(angle);
    /// let vector = Vector3::new(1_f64, 0_f64, 1_f64);
    /// let expected = Vector3::new(1_f64, 0_f64, -1_f64);
    /// let result = matrix * vector;
    /// 
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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

    /// Construct a rotation matrix about the **z-axis** by an angle `angle`.
    ///
    /// ## Example
    /// 
    /// In this example the rotation is in the **xy-plane**.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Angle,
    /// #     Radians,
    /// #     Vector3, 
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_angle_z(angle);
    /// let vector = Vector3::new(1_f64, 1_f64, 0_f64);
    /// let expected = Vector3::new(-1_f64, 1_f64, 0_f64);
    /// let result = matrix * vector;
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Angle, 
    /// #     Matrix3x3,
    /// #     Radians,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_axis_angle(&axis, angle);
    /// let unit_x = Vector3::unit_x();
    /// let expected = Vector3::unit_y();
    /// let result = matrix * unit_x;
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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
    /// an observer located at the origin facing the **positive z-axis** into a
    /// coordinate system of an observer located at the origin facing the 
    /// direction `direction`.
    ///
    /// The function maps the **positive z-axis** to the direction `direction`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3,    
    /// # };
    /// # use approx::{
    /// #     relative_eq,    
    /// # };
    /// # use core::f64;
    /// #
    /// let direction = Vector3::new(1_f64, -1_f64, 1_f64) / f64::sqrt(3_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let expected = Matrix3x3::new(
    ///      1_f64 / f64::sqrt(6_f64), -1_f64 / f64::sqrt(6_f64), -2_f64 / f64::sqrt(6_f64),
    ///      1_f64 / f64::sqrt(2_f64),  1_f64 / f64::sqrt(2_f64),  0_f64,
    ///      1_f64 / f64::sqrt(3_f64), -1_f64 / f64::sqrt(3_f64),  1_f64 / f64::sqrt(3_f64)
    /// );
    /// let result = Matrix3x3::face_towards(&direction, &up);
    ///
    /// assert!(relative_eq!(result, expected));
    ///
    /// let transformed_z = result * Vector3::unit_z();
    ///
    /// assert_eq!(transformed_z, direction);
    /// ```
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
    /// at the origin facing the **negative z-axis**.
    ///
    /// The function maps the direction `direction` to the **negative z-axis** in 
    /// the new the coordinate system. This corresponds to a rotation matrix.
    /// This transformation is a **right-handed** coordinate transformation.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Vector3, 
    /// # };
    /// # use approx::{
    /// #     relative_eq,  
    /// # };
    /// # use core::f64;
    /// #
    /// let direction = Vector3::new(1_f64, 1_f64, 0_f64);
    /// let up = Vector3::unit_z();
    /// let expected = Matrix3x3::new(
    ///      1_f64 / f64::sqrt(2_f64), 0_f64, -1_f64 / f64::sqrt(2_f64),
    ///     -1_f64 / f64::sqrt(2_f64), 0_f64, -1_f64 / f64::sqrt(2_f64),
    ///      0_f64,                    1_f64,  0_f64
    /// );
    /// let result = Matrix3x3::look_at_rh(&direction, &up);
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
    #[inline]
    pub fn look_at_rh(direction: &Vector3<S>, up: &Vector3<S>) -> Matrix3x3<S> {
        // The inverse of a rotation matrix is its transpose.
        Self::face_towards(&(-direction), up).transpose()
    }

    /// Construct a coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing 
    /// the direction `direction` into the coordinate system of an observer located
    /// at the origin facing the **positive z-axis**.
    ///
    /// The function maps the direction `direction` to the **positive z-axis** in 
    /// the new the coordinate system. This corresponds to a rotation matrix.
    /// This transformation is a **left-handed** coordinate transformation. 
    #[inline]
    pub fn look_at_lh(direction: &Vector3<S>, up: &Vector3<S>) -> Matrix3x3<S> {
        // The inverse of a rotation matrix is its transpose.
        Self::face_towards(direction, up).transpose()
    }

    /// Construct a rotation matrix that rotates the shortest angular distance 
    /// between two vectors.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Radians,
    /// #     Angle,
    /// #     Vector3,
    /// # };
    /// # use approx::{
    /// #     relative_eq,   
    /// # };
    /// #
    /// let v1: Vector3<f64> = Vector3::unit_x() * 2_f64;
    /// let v2: Vector3<f64> = Vector3::unit_y() * 3_f64;
    /// let matrix = Matrix3x3::rotation_between(&v1, &v2).unwrap();
    /// let expected = Vector3::new(0_f64, 2_f64, 0_f64);
    /// let result = matrix * v1;
    /// 
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
    #[inline]
    pub fn rotation_between(v1: &Vector3<S>, v2: &Vector3<S>) -> Option<Matrix3x3<S>> {
        if let (Some(unit_v1), Some(unit_v2)) = (
            v1.try_normalize(S::zero()), 
            v2.try_normalize(S::zero()))
         {
            let cross = unit_v1.cross(&unit_v2);

            if let Some(axis) = Unit::try_from_value(cross, S::default_epsilon()) {
                return Some(
                    Matrix3x3::from_axis_angle(&axis, Radians::acos(unit_v1.dot(&unit_v2)))
                );
            }

            if unit_v1.dot(&unit_v2) < S::zero() {
                return None;
            }
        }

        Some(Self::identity())
    }

    /// Construct a rotation matrix that rotates the shortest angular distance 
    /// between two vectors.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,
    /// #     Radians,
    /// #     Angle,
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     relative_eq,   
    /// # };
    /// #
    /// let unit_v1: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x() * 2_f64);
    /// let unit_v2: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y() * 3_f64);
    /// let matrix = Matrix3x3::rotation_between_axis(&unit_v1, &unit_v2).unwrap();
    /// let vector = Vector3::unit_x() * 2_f64;
    /// let expected = Vector3::unit_y() * 2_f64;
    /// let result = matrix * vector;
    /// 
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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

        Some(Self::identity())
    }

    /// Linearly interpolate between two matrices.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,    
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let matrix0 = Matrix3x3::new(
    ///     0_f64, 0_f64, 0_f64, 
    ///     1_f64, 1_f64, 1_f64,
    ///     2_f64, 2_f64, 2_f64
    /// );
    /// let matrix1 = Matrix3x3::new(
    ///     3_f64, 3_f64, 3_f64, 
    ///     4_f64, 4_f64, 4_f64,
    ///     5_f64, 5_f64, 5_f64
    /// );
    /// let amount = 0.5;
    /// let expected = Matrix3x3::new(
    ///     1.5_f64, 1.5_f64, 1.5_f64, 
    ///     2.5_f64, 2.5_f64, 2.5_f64,
    ///     3.5_f64, 3.5_f64, 3.5_f64
    /// );
    /// let result = matrix0.lerp(&matrix1, amount);
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    ///
    /// ## Example (Finite Matrix)
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// # use core::f64;
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_f64, 2_f64, 3_f64,
    ///     4_f64, 5_f64, 6_f64,
    ///     7_f64, 8_f64, 9_f64 
    /// );
    /// 
    /// assert!(matrix.is_finite());
    /// ```
    ///
    /// ## Example (Not A Finite Matrix)
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY,
    ///     f64::INFINITY,     f64::INFINITY,     f64::INFINITY,
    ///     7_f64,             8_f64,             9_f64 
    /// );
    /// 
    /// assert!(!matrix.is_finite());
    /// ```
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.data[0][0].is_finite() && 
        self.data[0][1].is_finite() && 
        self.data[0][2].is_finite() &&
        self.data[1][0].is_finite() && 
        self.data[1][1].is_finite() && 
        self.data[1][2].is_finite() &&
        self.data[2][0].is_finite() && 
        self.data[2][1].is_finite() && 
        self.data[2][2].is_finite()
    }

    /// Compute the inverse of a square matrix, if the inverse exists. 
    ///
    /// Given a square matrix `self` Compute the matrix `m` if it exists 
    /// such that
    /// ```text
    /// m * self = self * m = 1.
    /// ```
    /// Not every square matrix has an inverse.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3,  
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_f64, 4_f64, 7_f64,
    ///     2_f64, 5_f64, 8_f64,
    ///     5_f64, 6_f64, 11_f64
    /// );
    /// let expected = Matrix3x3::new(
    ///     -7_f64 / 12_f64,   2_f64 / 12_f64,   3_f64 / 12_f64,
    ///     -18_f64 / 12_f64,  24_f64 / 12_f64, -6_f64 / 12_f64,
    ///      13_f64 / 12_f64, -14_f64 / 12_f64,  3_f64 / 12_f64
    /// );
    /// let result = matrix.inverse().unwrap();
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.is_zero() {
            None
        } else {
            let inv_det = S::one() / det;
    
            Some(Matrix3x3::new(
                inv_det * (self.data[1][1] * self.data[2][2] - self.data[1][2] * self.data[2][1]), 
                inv_det * (self.data[0][2] * self.data[2][1] - self.data[0][1] * self.data[2][2]), 
                inv_det * (self.data[0][1] * self.data[1][2] - self.data[0][2] * self.data[1][1]),
        
                inv_det * (self.data[1][2] * self.data[2][0] - self.data[1][0] * self.data[2][2]),
                inv_det * (self.data[0][0] * self.data[2][2] - self.data[0][2] * self.data[2][0]),
                inv_det * (self.data[0][2] * self.data[1][0] - self.data[0][0] * self.data[1][2]),
    
                inv_det * (self.data[1][0] * self.data[2][1] - self.data[1][1] * self.data[2][0]), 
                inv_det * (self.data[0][1] * self.data[2][0] - self.data[0][0] * self.data[2][1]), 
                inv_det * (self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0])
            ))
        }
    }

    /// Determine whether a square matrix has an inverse matrix.
    ///
    /// A matrix is invertible is its determinant is not zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_f64, 2_f64, 3_f64,
    ///     4_f64, 5_f64, 6_f64,
    ///     7_f64, 8_f64, 9_f64   
    /// );
    /// 
    /// assert_eq!(matrix.determinant(), 0_f64);
    /// assert!(!matrix.is_invertible());
    /// ```
    #[inline]
    pub fn is_invertible(&self) -> bool {
        ulps_ne!(self.determinant(), S::zero())
    }
    
    /// Determine whether a square matrix is a diagonal matrix. 
    ///
    /// A square matrix is a diagonal matrix if every off-diagonal 
    /// element is zero.
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        ulps_eq!(self.data[0][1], S::zero()) &&
        ulps_eq!(self.data[0][2], S::zero()) && 
        ulps_eq!(self.data[1][0], S::zero()) &&
        ulps_eq!(self.data[1][2], S::zero()) &&
        ulps_eq!(self.data[2][0], S::zero()) &&
        ulps_eq!(self.data[2][1], S::zero())
    }
    
    /// Determine whether a matrix is symmetric. 
    ///
    /// A matrix is symmetric when element `(i, j)` is equal to element `(j, i)` 
    /// for each row `i` and column `j`. Otherwise, it is not a symmetric matrix. 
    /// Note that every diagonal matrix is a symmetric matrix.
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        ulps_eq!(self.data[0][1], self.data[1][0]) && 
        ulps_eq!(self.data[1][0], self.data[0][1]) &&
        ulps_eq!(self.data[0][2], self.data[2][0]) && 
        ulps_eq!(self.data[2][0], self.data[0][2]) &&
        ulps_eq!(self.data[1][2], self.data[2][1]) && 
        ulps_eq!(self.data[2][1], self.data[1][2])
    }
}

impl<S> fmt::Display for Matrix3x3<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // We print the matrix contents in row-major order like mathematical convention.
        writeln!(
            formatter, 
            "Matrix3x3 [[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]]", 
            self.data[0][0], self.data[1][0], self.data[2][0],
            self.data[0][1], self.data[1][1], self.data[2][1],
            self.data[0][2], self.data[1][2], self.data[2][2],
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
            matrix[0][0], matrix[0][1], S::zero(),
            matrix[1][0], matrix[1][1], S::zero(),
            S::zero(),    S::zero(),    S::one()
        )
    }
}

impl<S> From<&Matrix2x2<S>> for Matrix3x3<S> where S: Scalar {
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: &Matrix2x2<S>) -> Matrix3x3<S> {
        Matrix3x3::new(
            matrix[0][0], matrix[0][1], S::zero(),
            matrix[1][0], matrix[1][1], S::zero(),
            S::zero(),    S::zero(),    S::one()
        )
    }
}

impl_as_ref_ops!(Matrix3x3<S>, [S; 9]);
impl_as_ref_ops!(Matrix3x3<S>, [[S; 3]; 3]);
impl_as_ref_ops!(Matrix3x3<S>, [Vector3<S>; 3]);

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

impl<S> ops::Index<(usize, usize)> for Matrix3x3<S>{
    type Output = S;

    #[inline]
    fn index(&self, (column, row): (usize, usize)) -> &Self::Output {
        let v: &[[S; 3]; 3] = self.as_ref();
        &v[column][row]
    }
}

impl<S> ops::IndexMut<(usize, usize)> for Matrix3x3<S> {
    #[inline]
    fn index_mut(&mut self, (column, row): (usize, usize)) -> &mut S {
        let v: &mut [[S; 3]; 3] = self.as_mut();
        &mut v[column][row]
    }
}

impl_matrix_matrix_binary_ops1!(
    Add, add, add_array3x3_array3x3, Matrix3x3<S>, Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_matrix_matrix_binary_ops1!(
    Sub, sub, sub_array3x3_array3x3, Matrix3x3<S>, Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});

impl_matrix_matrix_mul_ops!(
    Matrix3x3, Matrix3x3 => Matrix3x3, dot_array3x3_col3, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
});

impl<S> ops::Mul<Vector3<S>> for Matrix3x3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: Vector3<S>) -> Self::Output {
        let x = self.data[0][0] * other[0] + self.data[1][0] * other[1] + self.data[2][0] * other[2];
        let y = self.data[0][1] * other[0] + self.data[1][1] * other[1] + self.data[2][1] * other[2];
        let z = self.data[0][2] * other[0] + self.data[1][2] * other[1] + self.data[2][2] * other[2];

        Vector3::new(x, y, z)
    }
}

impl<S> ops::Mul<&Vector3<S>> for Matrix3x3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: &Vector3<S>) -> Self::Output {
        let x = self.data[0][0] * other[0] + self.data[1][0] * other[1] + self.data[2][0] * other[2];
        let y = self.data[0][1] * other[0] + self.data[1][1] * other[1] + self.data[2][1] * other[2];
        let z = self.data[0][2] * other[0] + self.data[1][2] * other[1] + self.data[2][2] * other[2];

        Vector3::new(x, y, z)
    }
}

impl<S> ops::Mul<Vector3<S>> for &Matrix3x3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: Vector3<S>) -> Self::Output {
        let x = self.data[0][0] * other[0] + self.data[1][0] * other[1] + self.data[2][0] * other[2];
        let y = self.data[0][1] * other[0] + self.data[1][1] * other[1] + self.data[2][1] * other[2];
        let z = self.data[0][2] * other[0] + self.data[1][2] * other[1] + self.data[2][2] * other[2];

        Vector3::new(x, y, z)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Vector3<S>> for &'b Matrix3x3<S> where S: Scalar {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, other: &'a Vector3<S>) -> Self::Output {
        let x = self.data[0][0] * other[0] + self.data[1][0] * other[1] + self.data[2][0] * other[2];
        let y = self.data[0][1] * other[0] + self.data[1][1] * other[1] + self.data[2][1] * other[2];
        let z = self.data[0][2] * other[0] + self.data[1][2] * other[1] + self.data[2][2] * other[2];

        Vector3::new(x, y, z)
    }
}

impl_matrix_scalar_binary_ops1!(
    Mul, mul, mul_array3x3_scalar, Matrix3x3<S>, Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_matrix_scalar_binary_ops1!(
    Div, div, div_array3x3_scalar, Matrix3x3<S>, Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_matrix_scalar_binary_ops1!(
    Rem, rem, rem_array3x3_scalar, Matrix3x3<S>, Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_matrix_unary_ops1!(Neg, neg, neg_array3x3, Matrix3x3<S>, Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_matrix_binary_assign_ops1!(Matrix3x3<S>, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});

impl_scalar_matrix_mul_ops1!(u8,    Matrix3x3<u8>,    Matrix3x3<u8>,    { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(u16,   Matrix3x3<u16>,   Matrix3x3<u16>,   { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(u32,   Matrix3x3<u32>,   Matrix3x3<u32>,   { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(u64,   Matrix3x3<u64>,   Matrix3x3<u64>,   { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(u128,  Matrix3x3<u128>,  Matrix3x3<u128>,  { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(usize, Matrix3x3<usize>, Matrix3x3<usize>, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(i8,    Matrix3x3<i8>,    Matrix3x3<i8>,    { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(i16,   Matrix3x3<i16>,   Matrix3x3<i16>,   { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(i32,   Matrix3x3<i32>,   Matrix3x3<i32>,   { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(i64,   Matrix3x3<i64>,   Matrix3x3<i64>,   { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(i128,  Matrix3x3<i128>,  Matrix3x3<i128>,  { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(isize, Matrix3x3<isize>, Matrix3x3<isize>, { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(f32,   Matrix3x3<f32>,   Matrix3x3<f32>,   { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});
impl_scalar_matrix_mul_ops1!(f64,   Matrix3x3<f64>,   Matrix3x3<f64>,   { 
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) 
});

impl<S> approx::AbsDiffEq for Matrix3x3<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.data[0][0], &other.data[0][0], epsilon) && 
        S::abs_diff_eq(&self.data[0][1], &other.data[0][1], epsilon) &&
        S::abs_diff_eq(&self.data[0][2], &other.data[0][2], epsilon) &&
        S::abs_diff_eq(&self.data[1][0], &other.data[1][0], epsilon) && 
        S::abs_diff_eq(&self.data[1][1], &other.data[1][1], epsilon) &&
        S::abs_diff_eq(&self.data[1][2], &other.data[1][2], epsilon) &&
        S::abs_diff_eq(&self.data[2][0], &other.data[2][0], epsilon) && 
        S::abs_diff_eq(&self.data[2][1], &other.data[2][1], epsilon) &&
        S::abs_diff_eq(&self.data[2][2], &other.data[2][2], epsilon)
    }
}

impl<S> approx::RelativeEq for Matrix3x3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.data[0][0], &other.data[0][0], epsilon, max_relative) &&
        S::relative_eq(&self.data[0][1], &other.data[0][1], epsilon, max_relative) &&
        S::relative_eq(&self.data[0][2], &other.data[0][2], epsilon, max_relative) &&
        S::relative_eq(&self.data[1][0], &other.data[1][0], epsilon, max_relative) &&
        S::relative_eq(&self.data[1][1], &other.data[1][1], epsilon, max_relative) &&
        S::relative_eq(&self.data[1][2], &other.data[1][2], epsilon, max_relative) &&
        S::relative_eq(&self.data[2][0], &other.data[2][0], epsilon, max_relative) &&
        S::relative_eq(&self.data[2][1], &other.data[2][1], epsilon, max_relative) &&
        S::relative_eq(&self.data[2][2], &other.data[2][2], epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Matrix3x3<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.data[0][0], &other.data[0][0], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[0][1], &other.data[0][1], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[0][2], &other.data[0][2], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[1][0], &other.data[1][0], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[1][1], &other.data[1][1], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[1][2], &other.data[1][2], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[2][0], &other.data[2][0], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[2][1], &other.data[2][1], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[2][2], &other.data[2][2], epsilon, max_ulps)
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
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Matrix4x4<S> {
    data: [[S; 4]; 4],
}

impl_coords!(View4x4, { 
    c0r0, c0r1, c0r2, c0r3, 
    c1r0, c1r1, c1r2, c1r3, 
    c2r0, c2r1, c2r2, c2r3, 
    c3r0, c3r1, c3r2, c3r3 
});
impl_coords_deref!(Matrix4x4, View4x4);

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
            data: [
                [c0r0, c0r1, c0r2, c0r3],
                [c1r0, c1r1, c1r2, c1r3],
                [c2r0, c2r1, c2r2, c2r3],
                [c3r0, c3r1, c3r2, c3r3],
            ]
        }
    }

    /// Construct a 4x4 matrix from column vectors.
    #[rustfmt::skip]
    #[inline]
    pub fn from_columns(c0: Vector4<S>, c1: Vector4<S>, c2: Vector4<S>, c3: Vector4<S>) -> Matrix4x4<S> {
        Matrix4x4::new(
            c0.x, c0.y, c0.z, c0.w,
            c1.x, c1.y, c1.z, c1.w,
            c2.x, c2.y, c2.z, c2.w,
            c3.x, c3.y, c3.z, c3.w,
        )
    }
}

impl<S> Matrix4x4<S> where S: Copy {
    /// Construct a new matrix from a fill value.
    ///
    /// The resulting matrix is a matrix where each entry is the supplied fill
    /// value.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let fill_value = 4_u32;
    /// let expected = Matrix4x4::new(
    ///     fill_value, fill_value, fill_value, fill_value,
    ///     fill_value, fill_value, fill_value, fill_value,
    ///     fill_value, fill_value, fill_value, fill_value,
    ///     fill_value, fill_value, fill_value, fill_value
    /// );
    /// let result = Matrix4x4::from_fill(fill_value);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Matrix4x4<S> {
        Matrix4x4::new(
            value, value, value, value,
            value, value, value, value,
            value, value, value, value,
            value, value, value, value
        )
    }

    /// Get the row of the matrix by value.
    #[inline]
    pub fn row(&self, r: usize) -> Vector4<S> {
        Vector4::new(self[0][r], self[1][r], self[2][r], self[3][r])
    }
 
    /// Get the column of the matrix by value.
    #[inline]
    pub fn column(&self, c: usize) -> Vector4<S> {
        Vector4::new(self[c][0], self[c][1], self[c][2], self[c][3])
    }
     
    /// Swap two rows of a matrix.
    #[inline]
    pub fn swap_rows(&mut self, row_a: usize, row_b: usize) {
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
     
    /// Swap two columns of a matrix.
    #[inline]
    pub fn swap_columns(&mut self, col_a: usize, col_b: usize) {
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
     
    /// Swap two elements of a matrix.
    #[inline]
    pub fn swap(&mut self, a: (usize, usize), b: (usize, usize)) {
        let element_a = self[a.0][a.1];
        self[a.0][a.1] = self[b.0][b.1];
        self[b.0][b.1] = element_a;
    }

    /// The length of the the underlying array.
    #[inline]
    pub fn len() -> usize {
        16
    }

    /// The shape of the underlying array.
    #[inline]
    pub fn shape() -> (usize, usize) {
        (4, 4)
    }

    /// Generate a pointer to the underlying array.
    #[inline]
    pub fn as_ptr(&self) -> *const S {
        &self.data[0][0]
    }

    /// Generate a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        &mut self.data[0][0]
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        <Self as AsRef<[S; 16]>>::as_ref(self)
    }

    /// Map an operation on the elements of a matrix, returning a matrix whose 
    /// elements are elements of the new underlying type.
    #[rustfmt::skip]
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Matrix4x4<T> where F: FnMut(S) -> T {
        Matrix4x4::new(
            op(self.data[0][0]), 
            op(self.data[0][1]), 
            op(self.data[0][2]), 
            op(self.data[0][3]),
            op(self.data[1][0]), 
            op(self.data[1][1]), 
            op(self.data[1][2]), 
            op(self.data[3][1]),
            op(self.data[2][0]), 
            op(self.data[2][1]), 
            op(self.data[2][2]), 
            op(self.data[2][3]),
            op(self.data[3][0]), 
            op(self.data[3][1]), 
            op(self.data[3][2]), 
            op(self.data[3][3]),
        )
    }
}

impl<S> Matrix4x4<S> where S: NumCast + Copy {
    /// Cast a matrix from one type of scalars to another type of scalars.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,   
    /// # };
    /// # 
    /// let matrix: Matrix4x4<u32> = Matrix4x4::new(
    ///     1_u32,  2_u32,  3_u32,  4_u32,
    ///     5_u32,  6_u32,  7_u32,  8_u32,
    ///     9_u32,  10_u32, 11_u32, 12_u32,
    ///     13_u32, 14_u32, 15_u32, 16_u32
    /// );
    /// let expected: Option<Matrix4x4<i32>> = Some(Matrix4x4::new(
    ///     1_i32,  2_i32,  3_i32,  4_i32, 
    ///     5_i32,  6_i32,  7_i32,  8_i32, 
    ///     9_i32,  10_i32, 11_i32, 12_i32,
    ///     13_i32, 14_i32, 15_i32, 16_i32
    /// ));
    /// let result = matrix.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Matrix4x4<T>> {
        let c0r0 = match num_traits::cast(self.data[0][0]) {
            Some(value) => value,
            None => return None,
        };
        let c0r1 = match num_traits::cast(self.data[0][1]) {
            Some(value) => value,
            None => return None,
        };
        let c0r2 = match num_traits::cast(self.data[0][2]) {
            Some(value) => value,
            None => return None,
        };
        let c0r3 = match num_traits::cast(self.data[0][3]) {
            Some(value) => value,
            None => return None,
        };
        let c1r0 = match num_traits::cast(self.data[1][0]) {
            Some(value) => value,
            None => return None,
        };
        let c1r1 = match num_traits::cast(self.data[1][1]) {
            Some(value) => value,
            None => return None,
        };
        let c1r2 = match num_traits::cast(self.data[1][2]) {
            Some(value) => value,
            None => return None,
        };
        let c1r3 = match num_traits::cast(self.data[1][3]) {
            Some(value) => value,
            None => return None,
        };
        let c2r0 = match num_traits::cast(self.data[2][0]) {
            Some(value) => value,
            None => return None,
        };
        let c2r1 = match num_traits::cast(self.data[2][1]) {
            Some(value) => value,
            None => return None,
        };
        let c2r2 = match num_traits::cast(self.data[2][2]) {
            Some(value) => value,
            None => return None,
        };
        let c2r3 = match num_traits::cast(self.data[2][3]) {
            Some(value) => value,
            None => return None,
        };
        let c3r0 = match num_traits::cast(self.data[3][0]) {
            Some(value) => value,
            None => return None,
        };
        let c3r1 = match num_traits::cast(self.data[3][1]) {
            Some(value) => value,
            None => return None,
        };
        let c3r2 = match num_traits::cast(self.data[3][2]) {
            Some(value) => value,
            None => return None,
        };
        let c3r3 = match num_traits::cast(self.data[3][3]) {
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
    ///
    ///
    /// ## Example
    /// A homogeneous vector with a zero `w`-component should not translate.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// #     Vector3,
    /// # };
    /// #
    /// let distance = Vector3::new(3_u32, 7_u32, 11_u32);
    /// let matrix = Matrix4x4::from_affine_translation(&distance);
    /// let vector = Vector4::new(1_u32, 1_u32, 1_u32, 0_u32);
    /// let expected = Vector4::new(1_u32, 1_u32, 1_u32, 0_u32);
    /// let result = matrix * vector;
    /// 
    /// assert_eq!(result, expected); 
    /// ```
    /// A homogeneous vector with a unit `w`-component should translate.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// #     Vector3,
    /// # };
    /// #
    /// let distance = Vector3::new(3_u32, 7_u32, 11_u32);
    /// let matrix = Matrix4x4::from_affine_translation(&distance);
    /// let vector = Vector4::new(1_u32, 1_u32, 1_u32, 1_u32);
    /// let expected = Vector4::new(
    ///     1_u32 + distance.x, 
    ///     1_u32 + distance.y, 
    ///     1_u32 + distance.z, 
    ///     1_u32
    /// );
    /// let result = matrix * vector;
    /// 
    /// assert_eq!(result, expected); 
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_translation(distance: &Vector3<S>) -> Matrix4x4<S> {
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
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector4, 
    /// # };
    /// #
    /// let scale = 4_usize;
    /// let matrix = Matrix4x4::from_affine_scale(scale);
    /// let vector = Vector4::new(1_usize, 1_usize, 1_usize, 1_usize);
    /// let expected = Vector4::new(4_usize, 4_usize, 4_usize, 1_usize);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_affine_scale(scale: S) -> Matrix4x4<S> {
        Matrix4x4::from_affine_nonuniform_scale(scale, scale, scale)
    }

    /// Construct a three-dimensional affine scaling matrix.
    ///
    /// This is the most general case for affine scaling matrices: the scale 
    /// factor in each dimension need not be identical. Since this is an 
    /// affine matrix, the `w` component is unaffected.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector4, 
    /// # };
    /// #
    /// let scale_x = 4_usize;
    /// let scale_y = 6_usize;
    /// let scale_z = 8_usize;
    /// let matrix = Matrix4x4::from_affine_nonuniform_scale(
    ///     scale_x,
    ///     scale_y,
    ///     scale_z
    /// );
    /// let vector = Vector4::new(1_usize, 1_usize, 1_usize, 1_usize);
    /// let expected = Vector4::new(4_usize, 6_usize, 8_usize, 1_usize);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    /// along the **x-axis**, holding the **y-axis** constant and the **z-axis** 
    /// constant.
    ///
    /// The parameters `shear_x_with_y` and `shear_x_with_z` are the 
    /// multiplicative factors for the contributions of the **y-axis**, and the
    /// **z-axis**, respectively to shearing along the **x-axis**. Since this is an 
    /// affine transformation the `w` component of four-dimensional vectors is 
    /// unaffected.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector4, 
    /// # };
    /// #
    /// let shear_x_with_y = 3_i32;
    /// let shear_x_with_z = 19_i32;
    /// let matrix = Matrix4x4::from_affine_shear_x(shear_x_with_y, shear_x_with_z);
    /// let vector = Vector4::new(1_i32, 1_i32, 1_i32, 1_i32);
    /// let expected = Vector4::new(
    ///     1_i32 + shear_x_with_y * 1_i32 + shear_x_with_z * 1_i32,
    ///     1_i32,
    ///     1_i32,
    ///     1_i32
    /// );
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    /// the **y-axis**, holding the **x-axis** constant and the **z-axis** constant.
    ///
    /// The parameters `shear_y_with_x` and `shear_y_with_z` are the 
    /// multiplicative factors for the contributions of the **x-axis**, and the 
    /// **z-axis**, respectively to shearing along the **y-axis**. Since this is 
    /// an affine transformation the `w` component of four-dimensional vectors 
    /// is unaffected.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector4, 
    /// # };
    /// #
    /// let shear_y_with_x = 3_i32;
    /// let shear_y_with_z = 19_i32;
    /// let matrix = Matrix4x4::from_affine_shear_y(shear_y_with_x, shear_y_with_z);
    /// let vector = Vector4::new(1_i32, 1_i32, 1_i32, 1_i32);
    /// let expected = Vector4::new(
    ///     1_i32,
    ///     1_i32 + shear_y_with_x * 1_i32 + shear_y_with_z * 1_i32,
    ///     1_i32,
    ///     1_i32
    /// );
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    /// the **z-axis**, holding the **x-axis** constant and the **y-axis** constant.
    ///
    /// The parameters `shear_z_with_x` and `shear_z_with_y` are the 
    /// multiplicative factors for the contributions of the **x-axis**, and the 
    /// **y-axis**, respectively to shearing along the **z-axis**. Since this is an 
    /// affine transformation the `w` component of four-dimensional vectors is 
    /// unaffected.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector4, 
    /// # };
    /// #
    /// let shear_z_with_x = 3_i32;
    /// let shear_z_with_y = 19_i32;
    /// let matrix = Matrix4x4::from_affine_shear_z(shear_z_with_x, shear_z_with_y);
    /// let vector = Vector4::new(1_i32, 1_i32, 1_i32, 1_i32);
    /// let expected = Vector4::new(
    ///     1_i32,
    ///     1_i32,
    ///     1_i32 + shear_z_with_x * 1_i32 + shear_z_with_y * 1_i32,
    ///     1_i32
    /// );
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
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
    /// contribution of the **y-axis** to shearing along the **x-axis**.
    ///
    /// The parameter `shear_x_with_z` denotes the factor scaling the 
    /// contribution of the **z-axis** to the shearing along the **x-axis**. 
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the **x-axis** to shearing along the **y-axis**.
    ///
    /// The parameter `shear_y_with_z` denotes the factor scaling the 
    /// contribution of the **z-axis** to the shearing along the **y-axis**. 
    ///
    /// The parameter `shear_z_with_x` denotes the factor scaling the
    /// contribution of the **x-axis** to shearing along the **z-axis**.
    ///
    /// The parameter `shear_z_with_y` denotes the factor scaling the 
    /// contribution of the **y-axis** to the shearing along the **z-axis**. 
    ///
    /// Since this is an affine transformation the `w` component
    /// of four-dimensional vectors is unaffected.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector4,   
    /// # };
    /// #
    /// let shear_x_with_y = 1_usize;
    /// let shear_x_with_z = 2_usize;
    /// let shear_y_with_x = 3_usize;
    /// let shear_y_with_z = 4_usize;
    /// let shear_z_with_x = 5_usize;
    /// let shear_z_with_y = 6_usize;
    /// let matrix = Matrix4x4::from_affine_shear(
    ///     shear_x_with_y, 
    ///     shear_x_with_z,
    ///     shear_y_with_x,
    ///     shear_y_with_z,
    ///     shear_z_with_x,
    ///     shear_z_with_y,
    /// );
    /// let vector = Vector4::new(1_usize, 1_usize, 1_usize, 1_usize);
    /// let expected = Vector4::new(
    ///     vector.x + shear_x_with_y * vector.y + shear_x_with_z * vector.z,
    ///     vector.y + shear_y_with_x * vector.x + shear_y_with_z * vector.z,
    ///     vector.z + shear_z_with_x * vector.x + shear_z_with_y * vector.y,
    ///     1_usize
    /// );
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ``` 
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

    /// Mutably transpose a square matrix in place.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let mut result = Matrix4x4::new(
    ///     1_i32, 1_i32, 1_i32, 1_i32,
    ///     2_i32, 2_i32, 2_i32, 2_i32,
    ///     3_i32, 3_i32, 3_i32, 3_i32,
    ///     4_i32, 4_i32, 4_i32, 4_i32
    /// );
    /// let expected = Matrix4x4::new(
    ///     1_i32, 2_i32, 3_i32, 4_i32,
    ///     1_i32, 2_i32, 3_i32, 4_i32,
    ///     1_i32, 2_i32, 3_i32, 4_i32,
    ///     1_i32, 2_i32, 3_i32, 4_i32 
    /// );
    /// result.transpose_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn transpose_mut(&mut self) {
        self.swap((0, 1), (1, 0));
        self.swap((0, 2), (2, 0));
        self.swap((1, 2), (2, 1));
        self.swap((0, 3), (3, 0));
        self.swap((1, 3), (3, 1));
        self.swap((2, 3), (3, 2));
    }

    /// Transpose a matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_i32, 1_i32, 1_i32, 1_i32,
    ///     2_i32, 2_i32, 2_i32, 2_i32,
    ///     3_i32, 3_i32, 3_i32, 3_i32,
    ///     4_i32, 4_i32, 4_i32, 4_i32
    /// );
    /// let expected = Matrix4x4::new(
    ///     1_i32, 2_i32, 3_i32, 4_i32,
    ///     1_i32, 2_i32, 3_i32, 4_i32,
    ///     1_i32, 2_i32, 3_i32, 4_i32,
    ///     1_i32, 2_i32, 3_i32, 4_i32   
    /// );
    /// let result = matrix.transpose();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn transpose(&self) -> Matrix4x4<S> {
        Matrix4x4::new(
            self.data[0][0], self.data[1][0], self.data[2][0], self.data[3][0],
            self.data[0][1], self.data[1][1], self.data[2][1], self.data[3][1], 
            self.data[0][2], self.data[1][2], self.data[2][2], self.data[3][2], 
            self.data[0][3], self.data[1][3], self.data[2][3], self.data[3][3]
        )
    }

    /// Compute a zero matrix.
    ///
    /// A zero matrix is a matrix in which all of its elements are zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let matrix: Matrix4x4<i32> = Matrix4x4::zero();
    ///
    /// assert!(matrix.is_zero());
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn zero() -> Matrix4x4<S> {
        let zero = S::zero();
        Matrix4x4::new(
            zero, zero, zero, zero, 
            zero, zero, zero, zero, 
            zero, zero, zero, zero, 
            zero, zero, zero, zero
        )
    }
    
    /// Determine whether a matrix is a zero matrix.
    ///
    /// A zero matrix is a matrix in which all of its elements are zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let matrix: Matrix4x4<i32> = Matrix4x4::zero();
    ///
    /// assert!(matrix.is_zero());
    /// ```
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.data[0][0].is_zero() && self.data[0][1].is_zero() && 
        self.data[0][2].is_zero() && self.data[0][3].is_zero() &&
        self.data[1][0].is_zero() && self.data[1][1].is_zero() && 
        self.data[1][2].is_zero() && self.data[1][3].is_zero() &&
        self.data[2][0].is_zero() && self.data[2][1].is_zero() && 
        self.data[2][2].is_zero() && self.data[2][3].is_zero() &&
        self.data[3][0].is_zero() && self.data[3][1].is_zero() && 
        self.data[3][2].is_zero() && self.data[3][3].is_zero()
    }
    
    /// Compute an identity matrix.
    ///
    /// An identity matrix is a matrix where the diagonal elements are one
    /// and the off-diagonal elements are zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let result: Matrix4x4<i32> = Matrix4x4::identity();
    /// let expected = Matrix4x4::new(
    ///     1_i32, 0_i32, 0_i32, 0_i32,
    ///     0_i32, 1_i32, 0_i32, 0_i32,
    ///     0_i32, 0_i32, 1_i32, 0_i32,
    ///     0_i32, 0_i32, 0_i32, 1_i32
    /// );
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn identity() -> Matrix4x4<S> {
        let one = S::one();
        let zero = S::zero();

        Matrix4x4::new(
            one,  zero, zero, zero, 
            zero, one,  zero, zero, 
            zero, zero, one,  zero, 
            zero, zero, zero, one
        )
    }
    
    /// Determine whether a matrix is an identity matrix.
    ///
    /// An identity matrix is a matrix where the diagonal elements are one
    /// and the off-diagonal elements are zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let matrix: Matrix4x4<i32> = Matrix4x4::identity();
    /// 
    /// assert!(matrix.is_identity());
    /// ```
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.data[0][0].is_one()  && self.data[0][1].is_zero() && 
        self.data[0][2].is_zero() && self.data[0][3].is_zero() &&
        self.data[1][0].is_zero() && self.data[1][1].is_one()  && 
        self.data[1][2].is_zero() && self.data[1][3].is_zero() &&
        self.data[2][0].is_zero() && self.data[2][1].is_zero() && 
        self.data[2][2].is_one()  && self.data[2][3].is_zero() &&
        self.data[3][0].is_zero() && self.data[3][1].is_zero() && 
        self.data[3][2].is_zero() && self.data[3][3].is_one()
    }

    /// Construct a new diagonal matrix from a given value where
    /// each element along the diagonal is equal to `value`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let result = Matrix4x4::from_diagonal_value(4_i32);
    /// let expected = Matrix4x4::new(
    ///     4_i32, 0_i32, 0_i32, 0_i32,
    ///     0_i32, 4_i32, 0_i32, 0_i32,
    ///     0_i32, 0_i32, 4_i32, 0_i32,
    ///     0_i32, 0_i32, 0_i32, 4_i32 
    /// );
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_diagonal_value(value: S) -> Self {
        Matrix4x4::new(
            value,     S::zero(), S::zero(), S::zero(),
            S::zero(), value,     S::zero(), S::zero(),
            S::zero(), S::zero(), value,     S::zero(),
            S::zero(), S::zero(), S::zero(), value
        )
    }
    
    /// Construct a new diagonal matrix from a vector of values
    /// representing the elements along the diagonal.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// #
    /// let result = Matrix4x4::from_diagonal(
    ///     &Vector4::new(2_i32, 3_i32, 4_i32, 5_i32)
    /// );
    /// let expected = Matrix4x4::new(
    ///     2_i32, 0_i32, 0_i32, 0_i32,
    ///     0_i32, 3_i32, 0_i32, 0_i32,
    ///     0_i32, 0_i32, 4_i32, 0_i32,
    ///     0_i32, 0_i32, 0_i32, 5_i32 
    /// );
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_diagonal(value: &Vector4<S>) -> Self {
        Matrix4x4::new(
            value.x,   S::zero(), S::zero(), S::zero(),
            S::zero(), value.y,   S::zero(), S::zero(),
            S::zero(), S::zero(), value.z,   S::zero(),
            S::zero(), S::zero(), S::zero(), value.w,
        )
    }
    
    /// Get the diagonal part of a square matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// #     Vector4,
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_i32,  2_i32,  3_i32,  4_i32,
    ///     5_i32,  6_i32,  7_i32,  8_i32,
    ///     9_i32,  10_i32, 11_i32, 12_i32,
    ///     13_i32, 14_i32, 15_i32, 16_i32
    /// );
    /// let expected = Vector4::new(1_i32, 6_i32, 11_i32, 16_i32);
    /// let result = matrix.diagonal();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn diagonal(&self) -> Vector4<S> {
        Vector4::new(
            self.data[0][0], 
            self.data[1][1], 
            self.data[2][2], 
            self.data[3][3]
        )
    }

    /// Compute the trace of a square matrix.
    ///
    /// The trace of a matrix is the sum of the diagonal elements.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_i32,  2_i32,  3_i32,  4_i32,
    ///     5_i32,  6_i32,  7_i32,  8_i32,
    ///     9_i32,  10_i32, 11_i32, 12_i32,
    ///     13_i32, 14_i32, 15_i32, 16_i32 
    /// );
    ///
    /// assert_eq!(matrix.trace(), 34_i32);
    /// ```
    #[inline]
    pub fn trace(&self) -> S {
        self.data[0][0] + self.data[1][1] + self.data[2][2] + self.data[3][3]
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
    /// A bare reflection matrix assumes that the the **x-axis** intercept `x0` 
    /// and the **y-axis** intercept `y0` are both zero, in which case the 
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
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// #
    /// let bias = Vector3::new(0_f64, 0_f64, 0_f64);
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let expected = Matrix4x4::new(
    ///     1_f64, 0_f64,  0_f64, 0_f64,
    ///     0_f64, 1_f64,  0_f64, 0_f64,
    ///     0_f64, 0_f64, -1_f64, 0_f64,
    ///     0_f64, 0_f64,  0_f64, 1_f64
    /// );
    /// let result = Matrix4x4::from_affine_reflection(&normal, &bias);
    ///
    /// assert_eq!(result, expected);
    /// ```
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

    /// Mutably negate the elements of a matrix in place.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// # 
    /// let mut result = Matrix4x4::new(
    ///     1_i32,  2_i32,  3_i32,  4_i32,
    ///     5_i32,  6_i32,  7_i32,  8_i32,
    ///     9_i32,  10_i32, 11_i32, 12_i32,
    ///     13_i32, 14_i32, 15_i32, 16_i32
    /// );
    /// let expected = Matrix4x4::new(
    ///     -1_i32,  -2_i32,  -3_i32,  -4_i32,
    ///     -5_i32,  -6_i32,  -7_i32,  -8_i32, 
    ///     -9_i32,  -10_i32, -11_i32, -12_i32,
    ///     -13_i32, -14_i32, -15_i32, -16_i32   
    /// );
    /// result.neg_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn neg_mut(&mut self) {
        self.data[0][0] = -self.data[0][0];
        self.data[0][1] = -self.data[0][1];
        self.data[0][2] = -self.data[0][2];
        self.data[0][3] = -self.data[0][3];
        self.data[1][0] = -self.data[1][0];
        self.data[1][1] = -self.data[1][1];
        self.data[1][2] = -self.data[1][2];
        self.data[1][3] = -self.data[1][3];
        self.data[2][0] = -self.data[2][0];
        self.data[2][1] = -self.data[2][1];
        self.data[2][2] = -self.data[2][2];
        self.data[2][3] = -self.data[2][3];
        self.data[3][0] = -self.data[3][0];
        self.data[3][1] = -self.data[3][1];
        self.data[3][2] = -self.data[3][2];
        self.data[3][3] = -self.data[3][3];
    }

    /// Compute the determinant of a matrix.
    /// 
    /// The determinant of a matrix is the signed volume of the parallelopiped
    /// swept out by the vectors represented by the matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     4_f64, 12_f64, 34_f64, 67_f64,
    ///     7_f64, 15_f64, 9_f64,  6_f64,
    ///     1_f64, 3_f64,  3_f64,  7_f64,
    ///     9_f64, 9_f64,  2_f64,  13_f64
    /// );
    ///
    /// assert_eq!(matrix.determinant(), 7854_f64);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn determinant(&self) -> S {
        self.data[0][0] * self.data[1][1] * self.data[2][2] * self.data[3][3] -
        self.data[0][0] * self.data[1][1] * self.data[2][3] * self.data[3][2] -
        self.data[0][0] * self.data[2][1] * self.data[1][2] * self.data[3][3] +
        self.data[0][0] * self.data[2][1] * self.data[1][3] * self.data[3][2] +
        self.data[0][0] * self.data[3][1] * self.data[1][2] * self.data[2][3] -
        self.data[0][0] * self.data[3][1] * self.data[1][3] * self.data[2][2] -
        self.data[1][0] * self.data[0][1] * self.data[2][2] * self.data[3][3] +
        self.data[1][0] * self.data[0][1] * self.data[2][3] * self.data[3][2] +
        self.data[1][0] * self.data[2][1] * self.data[0][2] * self.data[3][3] -
        self.data[1][0] * self.data[2][1] * self.data[0][3] * self.data[3][2] -
        self.data[1][0] * self.data[3][1] * self.data[0][2] * self.data[2][3] +
        self.data[1][0] * self.data[3][1] * self.data[0][3] * self.data[2][2] +
        self.data[2][0] * self.data[0][1] * self.data[1][2] * self.data[3][3] -
        self.data[2][0] * self.data[0][1] * self.data[1][3] * self.data[3][2] -
        self.data[2][0] * self.data[1][1] * self.data[0][2] * self.data[3][3] +
        self.data[2][0] * self.data[1][1] * self.data[0][3] * self.data[3][2] +
        self.data[2][0] * self.data[3][1] * self.data[0][2] * self.data[1][3] -
        self.data[2][0] * self.data[3][1] * self.data[0][3] * self.data[1][2] -
        self.data[3][0] * self.data[0][1] * self.data[1][2] * self.data[2][3] +
        self.data[3][0] * self.data[0][1] * self.data[1][3] * self.data[2][2] +
        self.data[3][0] * self.data[1][1] * self.data[0][2] * self.data[2][3] -
        self.data[3][0] * self.data[1][1] * self.data[0][3] * self.data[2][2] -
        self.data[3][0] * self.data[2][1] * self.data[0][2] * self.data[1][3] +
        self.data[3][0] * self.data[2][1] * self.data[0][3] * self.data[1][2]
    }
}

impl<S> Matrix4x4<S> where S: ScalarFloat {
    /// Construct a three-dimensional affine rotation matrix rotating a vector around the 
    /// **x-axis** by an angle `angle` radians/degrees.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector4, 
    /// #     Radians,
    /// #     Angle, 
    /// # };
    /// # use approx::{
    /// #     relative_eq,   
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix4x4::from_affine_angle_x(angle);
    /// let vector = Vector4::new(0_f64, 1_f64, 1_f64, 1_f64);
    /// let expected = Vector4::new(0_f64, -1_f64, 1_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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
    /// around the **y-axis** by an angle `angle` radians/degrees.
    ///
    /// ## Example
    /// 
    /// In this example the rotation is in the **zx-plane**.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Angle,
    /// #     Radians,
    /// #     Vector4, 
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix4x4::from_affine_angle_y(angle);
    /// let vector = Vector4::new(1_f64, 0_f64, 1_f64, 1_f64);
    /// let expected = Vector4::new(1_f64, 0_f64, -1_f64, 1_f64);
    /// let result = matrix * vector;
    /// 
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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
    /// around the **z-axis** by an angle `angle` radians/degrees.
    ///
    /// ## Example
    /// 
    /// In this example the rotation is in the **xy-plane**.
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Angle,
    /// #     Radians,
    /// #     Vector4, 
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix4x4::from_affine_angle_z(angle);
    /// let vector = Vector4::new(1_f64, 1_f64, 0_f64, 1_f64);
    /// let expected = Vector4::new(-1_f64, 1_f64, 0_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Angle, 
    /// #     Matrix4x4,
    /// #     Radians,
    /// #     Unit,
    /// #     Vector4,
    /// #     Vector3,
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix4x4::from_affine_axis_angle(&axis, angle);
    /// let vector = Vector4::new(1_f64, 0_f64, 0_f64, 1_f64);
    /// let expected = Vector4::new(0_f64, 1_f64, 0_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let left = -4.0;
    /// let right = 4.0;
    /// let bottom = -2.0;
    /// let top = 2.0;
    /// let near = 1.0;
    /// let far = 100.0;
    /// let expected = Matrix4x4::new(
    ///     1.0 / 4.0,  0.0,        0.0,          0.0,
    ///     0.0,        1.0 / 2.0,  0.0,          0.0,
    ///     0.0,        0.0,       -2.0 / 99.0,   0.0,
    ///     0.0,        0.0,       -101.0 / 99.0, 1.0
    /// );
    /// let result = Matrix4x4::from_orthographic(left, right, bottom, top, near, far);
    ///
    /// assert_eq!(result, expected);
    /// ```
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

    /// Construct a new three-dimensional orthographic projection matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Degrees,
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let vfov = Degrees(90.0);
    /// let aspect = 800 as f64 / 600 as f64;
    /// let near = 1.0;
    /// let far = 100.0;
    /// let expected = Matrix4x4::new(
    ///     2.0 / 100.0, 0.0,         0.0,          0.0, 
    ///     0.0,         2.0 / 75.0,  0.0,          0.0, 
    ///     0.0,         0.0,        -2.0 / 99.0,   0.0, 
    ///     0.0,         0.0,        -101.0 / 99.0, 1.0
    /// );
    /// let result = Matrix4x4::from_orthographic_fov(vfov, aspect, near, far);
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_orthographic_fov<A: Into<Radians<S>>>(
        vfov: A, aspect: S, near: S, far: S) -> Matrix4x4<S> 
    {
        let one_half = num_traits::cast(0.5).unwrap();
        let width = far * Angle::tan(vfov.into() * one_half);
        let height = width / aspect;

        Self::from_orthographic(
            -width * one_half, 
            width * one_half,
            -height * one_half,
            height * one_half,
            near,
            far
        )
    }

    /// Construct a new three-dimensional perspective projection matrix based
    /// on arbitrary `left`, `right`, `bottom`, `top`, `near` and `far` planes.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let left = -4.0;
    /// let right = 4.0;
    /// let bottom = -2.0;
    /// let top = 3.0;
    /// let near = 1.0;
    /// let far = 100.0;
    /// let expected = Matrix4x4::new(
    ///     1.0 / 4.0,  0.0,        0.0,           0.0,
    ///     0.0,        2.0 / 5.0,  0.0,           0.0,
    ///     0.0,        1.0 / 5.0, -101.0 / 99.0, -1.0,
    ///     0.0,        0.0,       -200.0 / 99.0,  0.0
    /// );
    /// let result = Matrix4x4::from_perspective(left, right, bottom, top, near, far);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_perspective(
        left: S, right: S, bottom: S, top: S, near: S, far: S) -> Matrix4x4<S> 
    {
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

    /// Construct a perspective projection matrix based on the `near` 
    /// plane, the `far` plane and the vertical field of view angle `vfov` and 
    /// the horizontal/vertical aspect ratio `aspect`.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Degrees,
    /// # };
    /// #
    /// let vfov = Degrees(72.0);
    /// let aspect = 800 as f32 / 600 as f32;
    /// let near = 0.1;
    /// let far = 100.0;
    /// let expected = Matrix4x4::new(
    ///     1.0322863, 0.0,        0.0,       0.0, 
    ///     0.0,       1.3763818,  0.0,       0.0, 
    ///     0.0,       0.0,       -1.002002, -1.0, 
    ///     0.0,       0.0,       -0.2002002, 0.0
    /// );
    /// let result = Matrix4x4::from_perspective_fov(vfov, aspect, near, far);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_perspective_fov<A: Into<Radians<S>>>(vfov: A, aspect: S, near: S, far: S) -> Matrix4x4<S> {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let range = Angle::tan(vfov.into() / two) * near;
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
    /// a coordinate system of an observer located at the origin facing the **z-axis**
    /// into a coordinate system of an observer located at the position `eye` facing
    /// the direction `direction`.
    ///
    /// The function maps the **z-axis** to the direction `direction`, and locates the 
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
        let eye_x = eye_vec.dot(&x_axis);
        let eye_y = eye_vec.dot(&y_axis);
        let eye_z = eye_vec.dot(&z_axis);
        
        Matrix4x4::new(
             x_axis.x,  x_axis.y,  x_axis.z, zero,
             y_axis.x,  y_axis.y,  y_axis.z, zero,
             z_axis.x,  z_axis.y,  z_axis.z, zero,
             eye_x,     eye_y,     eye_z,    one
        )
    }

    /// Construct an affine coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the position `eye` facing 
    /// the position `target` into the coordinate system of an observer located
    /// at the origin facing the **negative z-axis**.
    ///
    /// The function maps the direction of the target `target` to the 
    /// **negative z-axis** and locates the `eye` position to the origin in the
    /// new the coordinate system. This transformation is a **right-handed** 
    /// coordinate transformation. It is conventionally used in computer graphics 
    /// for camera view transformations.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Point3, 
    /// # };
    /// # use approx::{
    /// #     relative_eq,  
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(4_f64, 5_f64, 3_f64);
    /// let up = Vector3::unit_z();
    /// let expected = Matrix4x4::new(
    ///      1_f64 / f64::sqrt(2_f64),  0_f64, -1_f64 / f64::sqrt(2_f64),  0_f64,
    ///     -1_f64 / f64::sqrt(2_f64),  0_f64, -1_f64 / f64::sqrt(2_f64),  0_f64, 
    ///      0_f64,                     1_f64,  0_f64,                     0_f64,
    ///      1_f64 / f64::sqrt(2_f64), -3_f64,  3_f64 / f64::sqrt(2_f64),  1_f64
    /// );
    /// let result = Matrix4x4::look_at_rh(&eye, &target, &up);
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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
        let neg_eye_x = -eye_vec.dot(&x_axis);
        let neg_eye_y = -eye_vec.dot(&y_axis);
        let neg_eye_z = -eye_vec.dot(&z_axis);
        
        Matrix4x4::new(
            x_axis.x,  y_axis.x,  z_axis.x,  zero,
            x_axis.y,  y_axis.y,  z_axis.y,  zero,
            x_axis.z,  y_axis.z,  z_axis.z,  zero,
            neg_eye_x, neg_eye_y, neg_eye_z, one
        )
    }

    /// Construct an affine coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the position `eye` facing 
    /// the position `target` into the coordinate system of an observer located
    /// at the origin facing the **positive z-axis**.
    ///
    /// The function maps the direction of the target `target` to the 
    /// **positive z-axis** and locates the `eye` position to the origin in the
    /// new the coordinate system. This transformation is a **left-handed** 
    /// coordinate transformation. It is conventionally used in computer graphics 
    /// for camera view transformations.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Point3, 
    /// # };
    /// # use approx::{
    /// #     relative_eq,  
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(4_f64, 5_f64, 3_f64);
    /// let up = Vector3::unit_z();
    /// let expected = Matrix4x4::new(
    ///      -1_f64 / f64::sqrt(2_f64),  0_f64, 1_f64 / f64::sqrt(2_f64),  0_f64,
    ///       1_f64 / f64::sqrt(2_f64),  0_f64, 1_f64 / f64::sqrt(2_f64),  0_f64, 
    ///       0_f64,                     1_f64,                    0_f64,  0_f64,
    ///      -1_f64 / f64::sqrt(2_f64), -3_f64,  -3_f64 / f64::sqrt(2_f64),  1_f64
    /// );
    /// let result = Matrix4x4::look_at_lh(&eye, &target, &up);
    /// 
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
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
        let neg_eye_x = -eye_vec.dot(&x_axis);
        let neg_eye_y = -eye_vec.dot(&y_axis);
        let neg_eye_z = -eye_vec.dot(&z_axis);
        
        Matrix4x4::new(
            x_axis.x,  y_axis.x,  z_axis.x,  zero,
            x_axis.y,  y_axis.y,  z_axis.y,  zero,
            x_axis.z,  y_axis.z,  z_axis.z,  zero,
            neg_eye_x, neg_eye_y, neg_eye_z, one
        )
    }

    /// Linearly interpolate between two matrices.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #    Matrix4x4,
    /// # };
    /// #
    /// let matrix0 = Matrix4x4::new(
    ///     0_f64, 0_f64, 0_f64, 0_f64,
    ///     1_f64, 1_f64, 1_f64, 1_f64,
    ///     2_f64, 2_f64, 2_f64, 2_f64,
    ///     3_f64, 3_f64, 3_f64, 3_f64
    /// );
    /// let matrix1 = Matrix4x4::new(
    ///     4_f64, 4_f64, 4_f64, 4_f64,
    ///     5_f64, 5_f64, 5_f64, 5_f64,
    ///     6_f64, 6_f64, 6_f64, 6_f64,
    ///     7_f64, 7_f64, 7_f64, 7_f64
    /// );
    /// let amount = 0.5;
    /// let expected = Matrix4x4::new(
    ///     2_f64, 2_f64, 2_f64, 2_f64,
    ///     3_f64, 3_f64, 3_f64, 3_f64,
    ///     4_f64, 4_f64, 4_f64, 4_f64,
    ///     5_f64, 5_f64, 5_f64, 5_f64
    /// );
    /// let result = matrix0.lerp(&matrix1, amount);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Matrix4x4<S>, amount: S) -> Matrix4x4<S> {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of a matrix are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A matrix is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values.
    ///
    /// ## Example (Finite Matrix)
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,  
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_f64,  2_f64,  3_f64,  4_f64,
    ///     5_f64,  6_f64,  7_f64,  8_f64,
    ///     9_f64,  10_f64, 11_f64, 12_f64,
    ///     13_f64, 14_f64, 15_f64, 16_f64
    /// );
    /// 
    /// assert!(matrix.is_finite());
    /// ```
    ///
    /// ## Example (Not A Finite Matrix)
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,    
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_f64,             2_f64,             3_f64,             4_f64,
    ///     f64::NAN,          f64::NAN,          f64::NAN,          f64::NAN,
    ///     f64::INFINITY,     f64::INFINITY,     f64::INFINITY,     f64::INFINITY,
    ///     f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY
    /// );
    ///
    /// assert!(!matrix.is_finite());
    /// ```
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.data[0][0].is_finite() && self.data[0][1].is_finite() && 
        self.data[0][2].is_finite() && self.data[0][3].is_finite() &&
        self.data[1][0].is_finite() && self.data[1][1].is_finite() && 
        self.data[1][2].is_finite() && self.data[1][3].is_finite() &&
        self.data[2][0].is_finite() && self.data[2][1].is_finite() && 
        self.data[2][2].is_finite() && self.data[2][3].is_finite() &&
        self.data[3][0].is_finite() && self.data[3][1].is_finite() &&
        self.data[3][2].is_finite() && self.data[3][3].is_finite()
    }

    /// Compute the inverse of a square matrix, if the inverse exists. 
    ///
    /// Given a square matrix `self` Compute the matrix `m` if it exists 
    /// such that
    /// ```text
    /// m * self = self * m = 1.
    /// ```
    /// Not every square matrix has an inverse.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4,  
    /// # };
    /// # use approx::{
    /// #     relative_eq, 
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_f64, 4_f64, 7_f64,  8_f64,
    ///     2_f64, 5_f64, 8_f64,  4_f64,
    ///     5_f64, 6_f64, 11_f64, 4_f64,
    ///     9_f64, 3_f64, 13_f64, 5_f64
    /// );
    /// let expected = Matrix4x4::new(
    ///      17_f64 / 60_f64, -41_f64 / 30_f64,  21_f64 / 20_f64, -1_f64 / 5_f64,
    ///      7_f64 / 30_f64,  -16_f64 / 15_f64,  11_f64 / 10_f64, -2_f64 / 5_f64,
    ///     -13_f64 / 36_f64,  25_f64 / 18_f64, -13_f64 / 12_f64,  1_f64 / 3_f64,
    ///      13_f64 / 45_f64, -23_f64 / 45_f64,  4_f64 / 15_f64,  -1_f64 / 15_f64
    /// );
    /// let result = matrix.inverse().unwrap();
    ///
    /// assert!(relative_eq!(result, expected, epsilon = 1e-8));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.is_zero() {
            None
        } else {
            let det_inv = S::one() / det;
            let _c0r0 = self.data[1][1] * self.data[2][2] * self.data[3][3] + self.data[2][1] * self.data[3][2] * self.data[1][3] + self.data[3][1] * self.data[1][2] * self.data[2][3]
                      - self.data[3][1] * self.data[2][2] * self.data[1][3] - self.data[2][1] * self.data[1][2] * self.data[3][3] - self.data[1][1] * self.data[3][2] * self.data[2][3];
            let _c1r0 = self.data[3][0] * self.data[2][2] * self.data[1][3] + self.data[2][0] * self.data[1][2] * self.data[3][3] + self.data[1][0] * self.data[3][2] * self.data[2][3]
                      - self.data[1][0] * self.data[2][2] * self.data[3][3] - self.data[2][0] * self.data[3][2] * self.data[1][3] - self.data[3][0] * self.data[1][2] * self.data[2][3];
            let _c2r0 = self.data[1][0] * self.data[2][1] * self.data[3][3] + self.data[2][0] * self.data[3][1] * self.data[1][3] + self.data[3][0] * self.data[1][1] * self.data[2][3]
                      - self.data[3][0] * self.data[2][1] * self.data[1][3] - self.data[2][0] * self.data[1][1] * self.data[3][3] - self.data[1][0] * self.data[3][1] * self.data[2][3];
            let _c3r0 = self.data[3][0] * self.data[2][1] * self.data[1][2] + self.data[2][0] * self.data[1][1] * self.data[3][2] + self.data[1][0] * self.data[3][1] * self.data[2][2]
                      - self.data[1][0] * self.data[2][1] * self.data[3][2] - self.data[2][0] * self.data[3][1] * self.data[1][2] - self.data[3][0] * self.data[1][1] * self.data[2][2];
            let _c0r1 = self.data[3][1] * self.data[2][2] * self.data[0][3] + self.data[2][1] * self.data[0][2] * self.data[3][3] + self.data[0][1] * self.data[3][2] * self.data[2][3]
                      - self.data[0][1] * self.data[2][2] * self.data[3][3] - self.data[2][1] * self.data[3][2] * self.data[0][3] - self.data[3][1] * self.data[0][2] * self.data[2][3];
            let _c1r1 = self.data[0][0] * self.data[2][2] * self.data[3][3] + self.data[2][0] * self.data[3][2] * self.data[0][3] + self.data[3][0] * self.data[0][2] * self.data[2][3]
                      - self.data[3][0] * self.data[2][2] * self.data[0][3] - self.data[2][0] * self.data[0][2] * self.data[3][3] - self.data[0][0] * self.data[3][2] * self.data[2][3];
            let _c2r1 = self.data[3][0] * self.data[2][1] * self.data[0][3] + self.data[2][0] * self.data[0][1] * self.data[3][3] + self.data[0][0] * self.data[3][1] * self.data[2][3]
                      - self.data[0][0] * self.data[2][1] * self.data[3][3] - self.data[2][0] * self.data[3][1] * self.data[0][3] - self.data[3][0] * self.data[0][1] * self.data[2][3];
            let _c3r1 = self.data[0][0] * self.data[2][1] * self.data[3][2] + self.data[2][0] * self.data[3][1] * self.data[0][2] + self.data[3][0] * self.data[0][1] * self.data[2][2]
                      - self.data[3][0] * self.data[2][1] * self.data[0][2] - self.data[2][0] * self.data[0][1] * self.data[3][2] - self.data[0][0] * self.data[3][1] * self.data[2][2];
            let _c0r2 = self.data[0][1] * self.data[1][2] * self.data[3][3] + self.data[1][1] * self.data[3][2] * self.data[0][3] + self.data[3][1] * self.data[0][2] * self.data[1][3] 
                      - self.data[3][1] * self.data[1][2] * self.data[0][3] - self.data[1][1] * self.data[0][2] * self.data[3][3] - self.data[0][1] * self.data[3][2] * self.data[1][3];
            let _c1r2 = self.data[3][0] * self.data[1][2] * self.data[0][3] + self.data[1][0] * self.data[0][2] * self.data[3][3] + self.data[0][0] * self.data[3][2] * self.data[1][3]
                      - self.data[0][0] * self.data[1][2] * self.data[3][3] - self.data[1][0] * self.data[3][2] * self.data[0][3] - self.data[3][0] * self.data[0][2] * self.data[1][3];
            let _c2r2 = self.data[0][0] * self.data[1][1] * self.data[3][3] + self.data[1][0] * self.data[3][1] * self.data[0][3] + self.data[3][0] * self.data[0][1] * self.data[1][3]
                      - self.data[3][0] * self.data[1][1] * self.data[0][3] - self.data[1][0] * self.data[0][1] * self.data[3][3] - self.data[0][0] * self.data[3][1] * self.data[1][3];
            let _c3r2 = self.data[3][0] * self.data[1][1] * self.data[0][2] + self.data[1][0] * self.data[0][1] * self.data[3][2] + self.data[0][0] * self.data[3][1] * self.data[1][2]
                      - self.data[0][0] * self.data[1][1] * self.data[3][2] - self.data[1][0] * self.data[3][1] * self.data[0][2] - self.data[3][0] * self.data[0][1] * self.data[1][2];
            let _c0r3 = self.data[2][1] * self.data[1][2] * self.data[0][3] + self.data[1][1] * self.data[0][2] * self.data[2][3] + self.data[0][1] * self.data[2][2] * self.data[1][3]
                      - self.data[0][1] * self.data[1][2] * self.data[2][3] - self.data[1][1] * self.data[2][2] * self.data[0][3] - self.data[2][1] * self.data[0][2] * self.data[1][3];  
            let _c1r3 = self.data[0][0] * self.data[1][2] * self.data[2][3] + self.data[1][0] * self.data[2][2] * self.data[0][3] + self.data[2][0] * self.data[0][2] * self.data[1][3]
                      - self.data[2][0] * self.data[1][2] * self.data[0][3] - self.data[1][0] * self.data[0][2] * self.data[2][3] - self.data[0][0] * self.data[2][2] * self.data[1][3];
            let _c2r3 = self.data[2][0] * self.data[1][1] * self.data[0][3] + self.data[1][0] * self.data[0][1] * self.data[2][3] + self.data[0][0] * self.data[2][1] * self.data[1][3]
                      - self.data[0][0] * self.data[1][1] * self.data[2][3] - self.data[1][0] * self.data[2][1] * self.data[0][3] - self.data[2][0] * self.data[0][1] * self.data[1][3];
            let _c3r3 = self.data[0][0] * self.data[1][1] * self.data[2][2] + self.data[1][0] * self.data[2][1] * self.data[0][2] + self.data[2][0] * self.data[0][1] * self.data[1][2]
                      - self.data[2][0] * self.data[1][1] * self.data[0][2] - self.data[1][0] * self.data[0][1] * self.data[2][2] - self.data[0][0] * self.data[2][1] * self.data[1][2]; 

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

    /// Determine whether a square matrix has an inverse matrix.
    ///
    /// A matrix is invertible is its determinant is not zero.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_f64,  2_f64,  3_f64,  4_f64,
    ///     5_f64,  6_f64,  7_f64,  8_f64,
    ///     9_f64,  10_f64, 11_f64, 12_f64,
    ///     13_f64, 14_f64, 15_f64, 16_f64 
    /// );
    /// 
    /// assert_eq!(matrix.determinant(), 0_f64);
    /// eprintln!("{}", matrix.determinant());
    /// assert!(!matrix.is_invertible());
    /// ```
    #[inline]
    pub fn is_invertible(&self) -> bool {
        ulps_ne!(self.determinant(), S::zero())
    }

    /// Determine whether a square matrix is a diagonal matrix. 
    ///
    /// A square matrix is a diagonal matrix if every off-diagonal 
    /// element is zero.
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        ulps_eq!(self.data[0][1], S::zero()) &&
        ulps_eq!(self.data[0][2], S::zero()) && 
        ulps_eq!(self.data[1][0], S::zero()) &&
        ulps_eq!(self.data[1][2], S::zero()) &&
        ulps_eq!(self.data[2][0], S::zero()) &&
        ulps_eq!(self.data[2][1], S::zero())
    }
    
    /// Determine whether a matrix is symmetric. 
    ///
    /// A matrix is symmetric when element `(i, j)` is equal to element `(j, i)` 
    /// for each row `i` and column `j`. Otherwise, it is not a symmetric matrix. 
    /// Note that every diagonal matrix is a symmetric matrix.
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        ulps_eq!(self.data[0][1], self.data[1][0]) && ulps_eq!(self.data[1][0], self.data[0][1]) &&
        ulps_eq!(self.data[0][2], self.data[2][0]) && ulps_eq!(self.data[2][0], self.data[0][2]) &&
        ulps_eq!(self.data[1][2], self.data[2][1]) && ulps_eq!(self.data[2][1], self.data[1][2]) &&
        ulps_eq!(self.data[0][3], self.data[3][0]) && ulps_eq!(self.data[3][0], self.data[0][3]) &&
        ulps_eq!(self.data[1][3], self.data[3][1]) && ulps_eq!(self.data[3][1], self.data[1][3]) &&
        ulps_eq!(self.data[2][3], self.data[3][2]) && ulps_eq!(self.data[3][2], self.data[2][3])
    }
}

impl<S> fmt::Display for Matrix4x4<S> where S: fmt::Display {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // We print the matrix contents in row-major order like mathematical convention.
        writeln!(
            formatter, 
            "Matrix4x4 [[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]]",
            self.data[0][0], self.data[1][0], self.data[2][0], self.data[3][0],
            self.data[0][1], self.data[1][1], self.data[2][1], self.data[3][1],
            self.data[0][2], self.data[1][2], self.data[2][2], self.data[3][2],
            self.data[0][3], self.data[1][3], self.data[2][3], self.data[3][3]
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
            matrix[0][0], matrix[0][1], zero, zero,
            matrix[1][0], matrix[1][1], zero, zero,
            zero,         zero,         one,  zero,
            zero,         zero,         zero, one
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
            matrix[0][0], matrix[0][1], zero, zero,
            matrix[1][0], matrix[1][1], zero, zero,
            zero,         zero,         one,  zero,
            zero,         zero,         zero, one
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
            matrix[0][0], matrix[0][1], matrix[0][2], zero,
            matrix[1][0], matrix[1][1], matrix[1][2], zero,
            matrix[2][0], matrix[2][1], matrix[2][2], zero,
            zero,         zero,         zero,         one
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
            matrix[0][0], matrix[0][1], matrix[0][2], zero,
            matrix[1][0], matrix[1][1], matrix[1][2], zero,
            matrix[2][0], matrix[2][1], matrix[2][2], zero,
            zero,         zero,         zero,         one
        )
    }
}

impl_as_ref_ops!(Matrix4x4<S>, [S; 16]);
impl_as_ref_ops!(Matrix4x4<S>, [[S; 4]; 4]);
impl_as_ref_ops!(Matrix4x4<S>, [Vector4<S>; 4]);

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

impl<S> ops::Index<(usize, usize)> for Matrix4x4<S>{
    type Output = S;

    #[inline]
    fn index(&self, (column, row): (usize, usize)) -> &Self::Output {
        let v: &[[S; 4]; 4] = self.as_ref();
        &v[column][row]
    }
}

impl<S> ops::IndexMut<(usize, usize)> for Matrix4x4<S> {
    #[inline]
    fn index_mut(&mut self, (column, row): (usize, usize)) -> &mut S {
        let v: &mut [[S; 4]; 4] = self.as_mut();
        &mut v[column][row]
    }
}

impl_matrix_matrix_binary_ops1!(
    Add, add, add_array4x4_array4x4, Matrix4x4<S>, Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_matrix_matrix_binary_ops1!(
    Sub, sub, sub_array4x4_array4x4, Matrix4x4<S>, Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3) 
});

impl<S> ops::Mul<Vector4<S>> for Matrix4x4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: Vector4<S>) -> Self::Output {
        let x = self.data[0][0] * other[0] + self.data[1][0] * other[1] + self.data[2][0] * other[2] + self.data[3][0] * other[3];
        let y = self.data[0][1] * other[0] + self.data[1][1] * other[1] + self.data[2][1] * other[2] + self.data[3][1] * other[3];
        let z = self.data[0][2] * other[0] + self.data[1][2] * other[1] + self.data[2][2] * other[2] + self.data[3][2] * other[3];
        let w = self.data[0][3] * other[0] + self.data[1][3] * other[1] + self.data[2][3] * other[2] + self.data[3][3] * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<S> ops::Mul<&Vector4<S>> for Matrix4x4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: &Vector4<S>) -> Self::Output {
        let x = self.data[0][0] * other[0] + self.data[1][0] * other[1] + self.data[2][0] * other[2] + self.data[3][0] * other[3];
        let y = self.data[0][1] * other[0] + self.data[1][1] * other[1] + self.data[2][1] * other[2] + self.data[3][1] * other[3];
        let z = self.data[0][2] * other[0] + self.data[1][2] * other[1] + self.data[2][2] * other[2] + self.data[3][2] * other[3];
        let w = self.data[0][3] * other[0] + self.data[1][3] * other[1] + self.data[2][3] * other[2] + self.data[3][3] * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<S> ops::Mul<Vector4<S>> for &Matrix4x4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: Vector4<S>) -> Self::Output {
        let x = self.data[0][0] * other[0] + self.data[1][0] * other[1] + self.data[2][0] * other[2] + self.data[3][0] * other[3];
        let y = self.data[0][1] * other[0] + self.data[1][1] * other[1] + self.data[2][1] * other[2] + self.data[3][1] * other[3];
        let z = self.data[0][2] * other[0] + self.data[1][2] * other[1] + self.data[2][2] * other[2] + self.data[3][2] * other[3];
        let w = self.data[0][3] * other[0] + self.data[1][3] * other[1] + self.data[2][3] * other[2] + self.data[3][3] * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Vector4<S>> for &'b Matrix4x4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: &'a Vector4<S>) -> Self::Output {
        let x = self.data[0][0] * other[0] + self.data[1][0] * other[1] + self.data[2][0] * other[2] + self.data[3][0] * other[3];
        let y = self.data[0][1] * other[0] + self.data[1][1] * other[1] + self.data[2][1] * other[2] + self.data[3][1] * other[3];
        let z = self.data[0][2] * other[0] + self.data[1][2] * other[1] + self.data[2][2] * other[2] + self.data[3][2] * other[3];
        let w = self.data[0][3] * other[0] + self.data[1][3] * other[1] + self.data[2][3] * other[2] + self.data[3][3] * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl_matrix_matrix_mul_ops!(
    Matrix4x4, Matrix4x4 => Matrix4x4, dot_array4x4_col4, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_matrix_scalar_binary_ops1!(
    Mul, mul, mul_array4x4_scalar, Matrix4x4<S>, Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_matrix_scalar_binary_ops1!(
    Div, div, div_array4x4_scalar, Matrix4x4<S>, Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_matrix_scalar_binary_ops1!(
    Rem, rem, rem_array4x4_scalar, Matrix4x4<S>, Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_matrix_unary_ops1!(
    Neg, neg, neg_array4x4, Matrix4x4<S>, Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_matrix_binary_assign_ops1!(
    Matrix4x4<S>, { 
    (0, 0), (0, 1), (0, 2), (0, 3), 
    (1, 0), (1, 1), (1, 2), (1, 3), 
    (2, 0), (2, 1), (2, 2), (2, 3), 
    (3, 0), (3, 1), (3, 2), (3, 3) 
});

impl<S> approx::AbsDiffEq for Matrix4x4<S> where S: ScalarFloat {
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        S::abs_diff_eq(&self.data[0][0], &other.data[0][0], epsilon) && 
        S::abs_diff_eq(&self.data[0][1], &other.data[0][1], epsilon) &&
        S::abs_diff_eq(&self.data[0][2], &other.data[0][2], epsilon) &&
        S::abs_diff_eq(&self.data[0][3], &other.data[0][3], epsilon) && 
        S::abs_diff_eq(&self.data[1][0], &other.data[1][0], epsilon) && 
        S::abs_diff_eq(&self.data[1][1], &other.data[1][1], epsilon) &&
        S::abs_diff_eq(&self.data[1][2], &other.data[1][2], epsilon) &&
        S::abs_diff_eq(&self.data[1][3], &other.data[1][3], epsilon) && 
        S::abs_diff_eq(&self.data[2][0], &other.data[2][0], epsilon) && 
        S::abs_diff_eq(&self.data[2][1], &other.data[2][1], epsilon) &&
        S::abs_diff_eq(&self.data[2][2], &other.data[2][2], epsilon) &&
        S::abs_diff_eq(&self.data[2][3], &other.data[2][3], epsilon) && 
        S::abs_diff_eq(&self.data[3][0], &other.data[3][0], epsilon) && 
        S::abs_diff_eq(&self.data[3][1], &other.data[3][1], epsilon) &&
        S::abs_diff_eq(&self.data[3][2], &other.data[3][2], epsilon) &&
        S::abs_diff_eq(&self.data[3][3], &other.data[3][3], epsilon) 
    }
}

impl<S> approx::RelativeEq for Matrix4x4<S> where S: ScalarFloat {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.data[0][0], &other.data[0][0], epsilon, max_relative) &&
        S::relative_eq(&self.data[0][1], &other.data[0][1], epsilon, max_relative) &&
        S::relative_eq(&self.data[0][2], &other.data[0][2], epsilon, max_relative) &&
        S::relative_eq(&self.data[0][3], &other.data[0][3], epsilon, max_relative) &&
        S::relative_eq(&self.data[1][0], &other.data[1][0], epsilon, max_relative) &&
        S::relative_eq(&self.data[1][1], &other.data[1][1], epsilon, max_relative) &&
        S::relative_eq(&self.data[1][2], &other.data[1][2], epsilon, max_relative) &&
        S::relative_eq(&self.data[1][3], &other.data[1][3], epsilon, max_relative) &&
        S::relative_eq(&self.data[2][0], &other.data[2][0], epsilon, max_relative) &&
        S::relative_eq(&self.data[2][1], &other.data[2][1], epsilon, max_relative) &&
        S::relative_eq(&self.data[2][2], &other.data[2][2], epsilon, max_relative) &&
        S::relative_eq(&self.data[2][3], &other.data[2][3], epsilon, max_relative) &&
        S::relative_eq(&self.data[3][0], &other.data[3][0], epsilon, max_relative) &&
        S::relative_eq(&self.data[3][1], &other.data[3][1], epsilon, max_relative) &&
        S::relative_eq(&self.data[3][2], &other.data[3][2], epsilon, max_relative) &&
        S::relative_eq(&self.data[3][3], &other.data[3][3], epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Matrix4x4<S> where S: ScalarFloat {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.data[0][0], &other.data[0][0], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[0][1], &other.data[0][1], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[0][2], &other.data[0][2], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[0][3], &other.data[0][3], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[1][0], &other.data[1][0], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[1][1], &other.data[1][1], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[1][2], &other.data[1][2], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[1][3], &other.data[1][3], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[2][0], &other.data[2][0], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[2][1], &other.data[2][1], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[2][2], &other.data[2][2], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[2][3], &other.data[2][3], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[3][0], &other.data[3][0], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[3][1], &other.data[3][1], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[3][2], &other.data[3][2], epsilon, max_ulps) &&
        S::ulps_eq(&self.data[3][3], &other.data[3][3], epsilon, max_ulps)
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

impl_scalar_matrix_mul_ops1!(
    u8,    Matrix4x4<u8>,    Matrix4x4<u8>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    u16,   Matrix4x4<u16>,   Matrix4x4<u16>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    u32,   Matrix4x4<u32>,   Matrix4x4<u32>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    u64,   Matrix4x4<u64>,   Matrix4x4<u64>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    u128,  Matrix4x4<u128>,  Matrix4x4<u128>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    usize, Matrix4x4<usize>, Matrix4x4<usize>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    i8,    Matrix4x4<i8>,    Matrix4x4<i8>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    i16,   Matrix4x4<i16>,   Matrix4x4<i16>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    i32,   Matrix4x4<i32>,   Matrix4x4<i32>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    i64,   Matrix4x4<i64>,   Matrix4x4<i64>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    i128,  Matrix4x4<i128>,  Matrix4x4<i128>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    isize, Matrix4x4<isize>, Matrix4x4<isize>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    f32,   Matrix4x4<f32>,   Matrix4x4<f32>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});
impl_scalar_matrix_mul_ops1!(
    f64,   Matrix4x4<f64>,   Matrix4x4<f64>, { 
        (0, 0), (0, 1), (0, 2), (0, 3), 
        (1, 0), (1, 1), (1, 2), (1, 3), 
        (2, 0), (2, 1), (2, 2), (2, 3), 
        (3, 0), (3, 1), (3, 2), (3, 3) 
});

