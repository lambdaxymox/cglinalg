use crate::constraint::{
    CanMultiply,
    CanTransposeMultiply,
    Const,
    DimAdd,
    DimEq,
    DimLt,
    DimMul,
    DimSub,
    ShapeConstraint,
};
use crate::normed::{
    Norm,
    Normed,
};
use crate::point::{
    Point2,
    Point3,
};
use crate::unit::Unit;
use crate::vector::{
    Vector,
    Vector2,
    Vector3,
    Vector4,
};
use crate::{
    impl_coords,
    impl_coords_deref,
};
use approx_cmp::{
    ulps_eq,
    ulps_ne,
};
use cglinalg_numeric::{
    SimdCast,
    SimdScalar,
    SimdScalarFloat,
    SimdScalarOrd,
    SimdScalarSigned,
};
use cglinalg_trigonometry::{
    Angle,
    Radians,
};

use core::fmt;
use core::ops;


/// A stack-allocated **(1 row, 1 column)** matrix in column-major order.
pub type Matrix1x1<S> = Matrix<S, 1, 1>;

/// A stack-allocated **(2 row, 2 column)** matrix in column-major order.
pub type Matrix2x2<S> = Matrix<S, 2, 2>;

/// A stack-allocated **(3 row, 3 column)** matrix in column-major order.
pub type Matrix3x3<S> = Matrix<S, 3, 3>;

/// A stack-allocated **(4 row, 4 column)** matrix in column-major order.
pub type Matrix4x4<S> = Matrix<S, 4, 4>;

/// A stack-allocated **(1 row, 2 column)** matrix in column-major order.
pub type Matrix1x2<S> = Matrix<S, 1, 2>;

/// A stack-allocated **(1 row, 3 column)** matrix in column-major order.
pub type Matrix1x3<S> = Matrix<S, 1, 3>;

/// A stack-allocated **(1 row, 4 column)** matrix in column-major order.
pub type Matrix1x4<S> = Matrix<S, 1, 4>;

/// A stack-allocated **(2 row, 3 column)** matrix in column-major order.
pub type Matrix2x3<S> = Matrix<S, 2, 3>;

/// A stack-allocated **(3 row, 2 column)** matrix in column-major order.
pub type Matrix3x2<S> = Matrix<S, 3, 2>;

/// A stack-allocated **(2 row, 4 column)** matrix in column-major order.
pub type Matrix2x4<S> = Matrix<S, 2, 4>;

/// A stack-allocated **(4 row, 2 column)** matrix in column-major order.
pub type Matrix4x2<S> = Matrix<S, 4, 2>;

/// A stack-allocated **(3 row, 4 column)** matrix in column-major order.
pub type Matrix3x4<S> = Matrix<S, 3, 4>;

/// A stack-allocated **(4 row, 3 column)** matrix in column-major order.
pub type Matrix4x3<S> = Matrix<S, 4, 3>;


/// A stack-allocated **(1 row, 1 column)** matrix in column-major order.
pub type RowVector1<S> = Matrix1x1<S>;

/// A stack-allocated **(1 row, 2 column)** matrix in column-major order.
pub type RowVector2<S> = Matrix1x2<S>;

/// A stack-allocated **(1 row, 3 column)** matrix in column-major order.
pub type RowVector3<S> = Matrix1x3<S>;

/// A stack-allocated **(1 row, 4 column)** matrix in column-major order.
pub type RowVector4<S> = Matrix1x4<S>;


/// A stack-allocated **(1 row, 1 column)** matrix in column-major order.
pub type Matrix1<S> = Matrix1x1<S>;

/// A stack-allocated **(2 row, 2 column)** matrix in column-major order.
pub type Matrix2<S> = Matrix2x2<S>;

/// A stack-allocated **(3 row, 3 column)** matrix in column-major order.
pub type Matrix3<S> = Matrix3x3<S>;

/// A stack-allocated **(4 row, 4 column)** matrix in column-major order.
pub type Matrix4<S> = Matrix4x4<S>;


#[inline(always)]
fn dot_array_col<S, const R1: usize, const C1: usize, const R2: usize>(arr: &[[S; R1]; C1], col: &[S; R2], r: usize) -> S
where
    S: SimdScalar,
    ShapeConstraint: DimEq<Const<C1>, Const<R2>> + DimEq<Const<R2>, Const<C1>>,
{
    // PERFORMANCE: The const loop should get unrolled during optimization.
    let mut result = S::zero();
    for i in 0..C1 {
        result += arr[i][r] * col[i];
    }

    result
}


/// A stack-allocated **(R row, C column)** matrix in column-major order.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Matrix<S, const R: usize, const C: usize> {
    data: [[S; R]; C],
}

impl<S, const R: usize, const C: usize> Matrix<S, R, C> {
    /// Returns the length of the underlying array storing the matrix omponents.
    #[inline]
    pub const fn len(&self) -> usize {
        R * C
    }

    /// Tests whether the number of elements in the matrix is zero.
    ///
    /// Returns `true` when the matrix is zero-dimensional, i.e. the number of rows is zero, or the
    /// number of columns is zero. It returns `false` otherwise.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The shape of the underlying array storing the matrix components.
    ///
    /// The shape is the equivalent number of columns and rows of the
    /// array as though it represents a matrix. The order of the descriptions
    /// of the shape of the array is **(rows, columns)**.
    #[inline]
    pub const fn shape(&self) -> (usize, usize) {
        (R, C)
    }

    /// Get a pointer to the underlying array.
    #[inline]
    pub const fn as_ptr(&self) -> *const S {
        &self.data[0][0]
    }

    /// Get a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        &mut self.data[0][0]
    }
}

impl<S, const R: usize, const C: usize, const RC: usize> Matrix<S, R, C>
where
    ShapeConstraint: DimMul<Const<R>, Const<C>, Output = Const<RC>>,
    ShapeConstraint: DimMul<Const<C>, Const<R>, Output = Const<RC>>,
{
    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        AsRef::<[S; RC]>::as_ref(self)
    }
}

impl<S, const R: usize, const C: usize> AsRef<[[S; R]; C]> for Matrix<S, R, C> {
    #[inline]
    fn as_ref(&self) -> &[[S; R]; C] {
        &self.data
    }
}

impl<S, const R: usize, const C: usize> AsMut<[[S; R]; C]> for Matrix<S, R, C> {
    #[inline]
    fn as_mut(&mut self) -> &mut [[S; R]; C] {
        &mut self.data
    }
}

impl<S, const R: usize, const C: usize> AsRef<[Vector<S, R>; C]> for Matrix<S, R, C> {
    #[inline]
    fn as_ref(&self) -> &[Vector<S, R>; C] {
        unsafe { &*(self as *const Matrix<S, R, C> as *const [Vector<S, R>; C]) }
    }
}

impl<S, const R: usize, const C: usize> AsMut<[Vector<S, R>; C]> for Matrix<S, R, C> {
    #[inline]
    fn as_mut(&mut self) -> &mut [Vector<S, R>; C] {
        unsafe { &mut *(self as *mut Matrix<S, R, C> as *mut [Vector<S, R>; C]) }
    }
}

impl<S, const R: usize, const C: usize, const RC: usize> AsRef<[S; RC]> for Matrix<S, R, C>
where
    ShapeConstraint: DimMul<Const<R>, Const<C>, Output = Const<RC>>,
    ShapeConstraint: DimMul<Const<C>, Const<R>, Output = Const<RC>>,
{
    #[inline]
    fn as_ref(&self) -> &[S; RC] {
        unsafe { &*(self as *const Matrix<S, R, C> as *const [S; RC]) }
    }
}

impl<S, const R: usize, const C: usize, const RC: usize> AsMut<[S; RC]> for Matrix<S, R, C>
where
    ShapeConstraint: DimMul<Const<R>, Const<C>, Output = Const<RC>>,
    ShapeConstraint: DimMul<Const<C>, Const<R>, Output = Const<RC>>,
{
    #[inline]
    fn as_mut(&mut self) -> &mut [S; RC] {
        unsafe { &mut *(self as *mut Matrix<S, R, C> as *mut [S; RC]) }
    }
}

impl<S, const R: usize, const C: usize> ops::Index<usize> for Matrix<S, R, C> {
    type Output = Vector<S, R>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[Vector<S, R>; C] = self.as_ref();
        &v[index]
    }
}

impl<S, const R: usize, const C: usize> ops::IndexMut<usize> for Matrix<S, R, C> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let v: &mut [Vector<S, R>; C] = self.as_mut();
        &mut v[index]
    }
}

impl<S, const R: usize, const C: usize> ops::Index<(usize, usize)> for Matrix<S, R, C> {
    type Output = S;

    #[inline]
    fn index(&self, (column, row): (usize, usize)) -> &Self::Output {
        let v: &[[S; R]; C] = self.as_ref();
        &v[column][row]
    }
}

impl<S, const R: usize, const C: usize> ops::IndexMut<(usize, usize)> for Matrix<S, R, C> {
    #[inline]
    fn index_mut(&mut self, (column, row): (usize, usize)) -> &mut Self::Output {
        let v: &mut [[S; R]; C] = self.as_mut();
        &mut v[column][row]
    }
}

impl<S, const R: usize, const C: usize> From<[[S; R]; C]> for Matrix<S, R, C>
where
    S: Copy,
{
    #[inline]
    fn from(data: [[S; R]; C]) -> Self {
        Self { data }
    }
}

impl<'a, S, const R: usize, const C: usize> From<&'a [[S; R]; C]> for &'a Matrix<S, R, C>
where
    S: Copy,
{
    #[inline]
    fn from(data: &'a [[S; R]; C]) -> &'a Matrix<S, R, C> {
        unsafe { &*(data as *const [[S; R]; C] as *const Matrix<S, R, C>) }
    }
}

impl<'a, S, const R: usize, const C: usize> From<&'a [Vector<S, R>; C]> for &'a Matrix<S, R, C>
where
    S: Copy,
{
    #[inline]
    fn from(data: &'a [Vector<S, R>; C]) -> &'a Matrix<S, R, C> {
        unsafe { &*(data as *const [Vector<S, R>; C] as *const Matrix<S, R, C>) }
    }
}

impl<S, const R: usize, const C: usize, const RC: usize> From<[S; RC]> for Matrix<S, R, C>
where
    S: Copy,
    ShapeConstraint: DimMul<Const<R>, Const<C>, Output = Const<RC>>,
    ShapeConstraint: DimMul<Const<C>, Const<R>, Output = Const<RC>>,
{
    #[inline]
    fn from(array: [S; RC]) -> Self {
        let data: &[[S; R]; C] = unsafe { core::mem::transmute::<&[S; RC], &[[S; R]; C]>(&array) };

        Self { data: *data }
    }
}

impl<'a, S, const R: usize, const C: usize, const RC: usize> From<&'a [S; RC]> for &'a Matrix<S, R, C>
where
    S: Copy,
    ShapeConstraint: DimMul<Const<R>, Const<C>, Output = Const<RC>>,
    ShapeConstraint: DimMul<Const<C>, Const<R>, Output = Const<RC>>,
{
    #[inline]
    fn from(array: &'a [S; RC]) -> &'a Matrix<S, R, C> {
        unsafe { &*(array as *const [S; RC] as *const Matrix<S, R, C>) }
    }
}


impl<S, const R: usize, const C: usize> Matrix<S, R, C> {
    /// Determine whether this matrix is a square matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x3,
    /// #     Matrix3x3,
    /// # };
    /// #
    /// let zero_matrix2x3: Matrix2x3<i32> = Matrix2x3::zero();
    /// let zero_matrix3x3: Matrix3x3<i32> = Matrix3x3::zero();
    ///
    /// assert!(!zero_matrix2x3.is_square());
    /// assert!(zero_matrix3x3.is_square());
    /// ```
    #[inline]
    pub const fn is_square(&self) -> bool {
        let shape = self.shape();

        shape.0 == shape.1
    }
}

impl<S, const R: usize, const C: usize> Matrix<S, R, C>
where
    S: Copy,
{
    /// Construct a new matrix from a fill value.
    ///
    /// The resulting matrix is a matrix where each entry is the supplied fill
    /// value.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let fill_value = 3_i32;
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
    pub const fn from_fill(value: S) -> Self {
        Self { data: [[value; R]; C] }
    }

    /// Construct a matrix from a set of column vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let c0 = Vector3::new(1_i32, 2_i32, 3_i32);
    /// let c1 = Vector3::new(4_i32, 5_i32, 6_i32);
    /// let c2 = Vector3::new(7_i32, 8_i32, 9_i32);
    /// let matrix = Matrix3x3::from_columns(&[c0, c1, c2]);
    ///
    /// assert_eq!(matrix[0][0], c0[0]); assert_eq!(matrix[0][1], c0[1]); assert_eq!(matrix[0][2], c0[2]);
    /// assert_eq!(matrix[1][0], c1[0]); assert_eq!(matrix[1][1], c1[1]); assert_eq!(matrix[1][2], c1[2]);
    /// assert_eq!(matrix[2][0], c2[0]); assert_eq!(matrix[2][1], c2[1]); assert_eq!(matrix[2][2], c2[2]);
    /// ```
    #[inline]
    pub fn from_columns(columns: &[Vector<S, R>; C]) -> Self {
        let data_ptr = unsafe { &*(columns as *const [Vector<S, R>; C] as *const [[S; R]; C]) };

        Self { data: *data_ptr }
    }

    /// Construct a matrix from a set of row vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let r0 = Vector3::new(1_i32, 2_i32, 3_i32);
    /// let r1 = Vector3::new(4_i32, 5_i32, 6_i32);
    /// let r2 = Vector3::new(7_i32, 8_i32, 9_i32);
    /// let matrix = Matrix3x3::from_rows(&[r0, r1, r2]);
    ///
    /// assert_eq!(matrix[0][0], r0[0]); assert_eq!(matrix[0][1], r1[0]); assert_eq!(matrix[0][2], r2[0]);
    /// assert_eq!(matrix[1][0], r0[1]); assert_eq!(matrix[1][1], r1[1]); assert_eq!(matrix[1][2], r2[1]);
    /// assert_eq!(matrix[2][0], r0[2]); assert_eq!(matrix[2][1], r1[2]); assert_eq!(matrix[2][2], r2[2]);
    /// ```
    #[allow(clippy::needless_range_loop)]
    #[inline]
    pub fn from_rows(rows: &[Vector<S, C>; R]) -> Self {
        // SAFETY: Every location gets written into with a valid value of type `S`.
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut data: [[S; R]; C] = unsafe { core::mem::zeroed() };
        for r in 0..R {
            for c in 0..C {
                data[c][r] = rows[r][c];
            }
        }

        Self { data }
    }

    /// Map an operation on the elements of a matrix, returning a matrix whose
    /// elements are elements of the new underlying type.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_i32,  2_i32,  3_i32,  4_i32,
    ///     5_i32,  6_i32,  7_i32,  8_i32,
    ///     9_i32,  10_i32, 11_i32, 12_i32,
    ///     13_i32, 14_i32, 15_i32, 16_i32,
    /// );
    /// let expected = Matrix4x4::new(
    ///     2_f64,  4_f64,  6_f64,  8_f64,
    ///     10_f64, 12_f64, 14_f64, 16_f64,
    ///     18_f64, 20_f64, 22_f64, 24_f64,
    ///     26_f64, 28_f64, 30_f64, 32_f64,
    /// );
    /// let result = matrix.map(|comp| (2 * comp) as f64);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[allow(clippy::needless_range_loop)]
    #[inline]
    pub fn map<T, F>(&self, mut op: F) -> Matrix<T, R, C>
    where
        F: FnMut(S) -> T,
    {
        // SAFETY: Every location gets written into with a valid value of type `T`.
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut data: [[T; R]; C] = unsafe { core::mem::zeroed() };
        for c in 0..C {
            for r in 0..R {
                data[c][r] = op(self.data[c][r]);
            }
        }

        Matrix { data }
    }

    /// Get the row of the matrix by value.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_i32, 2_i32, 3_i32,
    ///     4_i32, 5_i32, 6_i32,
    ///     7_i32, 8_i32, 9_i32,
    /// );
    /// let expected_0 = Vector3::new(1_i32, 4_i32, 7_i32);
    /// let expected_1 = Vector3::new(2_i32, 5_i32, 8_i32);
    /// let expected_2 = Vector3::new(3_i32, 6_i32, 9_i32);
    ///
    /// assert_eq!(matrix.row(0), expected_0);
    /// assert_eq!(matrix.row(1), expected_1);
    /// assert_eq!(matrix.row(2), expected_2);
    /// ```
    #[allow(clippy::needless_range_loop)]
    #[inline]
    pub fn row(&self, r: usize) -> Vector<S, C> {
        // SAFETY: Every location gets written into with a value value of type `S`.
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut data: [S; C] = unsafe { core::mem::zeroed() };
        for c in 0..C {
            data[c] = self.data[c][r];
        }

        Vector::from(data)
    }

    /// Get the column of the matrix by value.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_i32, 2_i32, 3_i32,
    ///     4_i32, 5_i32, 6_i32,
    ///     7_i32, 8_i32, 9_i32,
    /// );
    /// let expected_0 = Vector3::new(1_i32, 2_i32, 3_i32);
    /// let expected_1 = Vector3::new(4_i32, 5_i32, 6_i32);
    /// let expected_2 = Vector3::new(7_i32, 8_i32, 9_i32);
    ///
    /// assert_eq!(matrix.column(0), expected_0);
    /// assert_eq!(matrix.column(1), expected_1);
    /// assert_eq!(matrix.column(2), expected_2);
    /// ```
    #[inline]
    pub fn column(&self, c: usize) -> Vector<S, R> {
        Vector::from(&self.data[c])
    }

    /// Swap two rows of a matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let mut result = Matrix3x3::new(
    ///     1_i32, 2_i32, 3_i32,
    ///     1_i32, 2_i32, 3_i32,
    ///     1_i32, 2_i32, 3_i32,
    /// );
    /// let expected = Matrix3x3::new(
    ///     3_i32, 2_i32, 1_i32,
    ///     3_i32, 2_i32, 1_i32,
    ///     3_i32, 2_i32, 1_i32,
    /// );
    /// result.swap_rows(0, 2);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn swap_rows(&mut self, row_a: usize, row_b: usize) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for c in 0..C {
            self.data[c].swap(row_a, row_b);
        }
    }

    /// Swap two columns of a matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x4;
    /// #
    /// let mut result = Matrix3x4::new(
    ///     1_i32, 1_i32, 1_i32,
    ///     2_i32, 2_i32, 2_i32,
    ///     3_i32, 3_i32, 3_i32,
    ///     4_i32, 4_i32, 4_i32,
    ///
    /// );
    /// let expected = Matrix3x4::new(
    ///     2_i32, 2_i32, 2_i32,
    ///     4_i32, 4_i32, 4_i32,
    ///     3_i32, 3_i32, 3_i32,
    ///     1_i32, 1_i32, 1_i32,
    /// );
    /// result.swap_columns(0, 1);
    /// result.swap_columns(1, 3);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn swap_columns(&mut self, col_a: usize, col_b: usize) {
        self.data.swap(col_a, col_b);
    }

    /// Swap two elements of a matrix.
    ///
    /// The element order for each element to swap is **(column, row)**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let mut result = Matrix4x4::new(
    ///     1_i32,  2_i32,  3_i32,  4_i32,
    ///     5_i32,  6_i32,  7_i32,  8_i32,
    ///     9_i32,  10_i32, 11_i32, 12_i32,
    ///     13_i32, 14_i32, 15_i32, 16_i32,
    /// );
    /// let expected = Matrix4x4::new(
    ///     1_i32, 2_i32,  3_i32,  13_i32,
    ///     5_i32, 6_i32,  7_i32,  8_i32,
    ///     9_i32, 10_i32, 11_i32, 12_i32,
    ///     4_i32, 14_i32, 15_i32, 16_i32,
    /// );
    /// result.swap((0, 3), (3, 0));
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn swap(&mut self, a: (usize, usize), b: (usize, usize)) {
        let element_a = self.data[a.0][a.1];
        self.data[a.0][a.1] = self.data[b.0][b.1];
        self.data[b.0][b.1] = element_a;
    }
}

impl<S, const R: usize, const C: usize> Matrix<S, R, C>
where
    S: SimdCast + Copy,
{
    /// Cast a matrix from one type of scalars to another type of scalars.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix2x2;
    /// #
    /// let matrix: Matrix2x2<i32> = Matrix2x2::new(1_i32, 2_i32, 3_i32, 4_i32);
    /// let expected: Option<Matrix2x2<f64>> = Some(Matrix2x2::new(1_f64, 2_f64, 3_f64, 4_f64));
    /// let result = matrix.try_cast::<f64>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[allow(clippy::needless_range_loop)]
    #[inline]
    pub fn try_cast<T>(&self) -> Option<Matrix<T, R, C>>
    where
        T: SimdCast,
    {
        // SAFETY: Every location gets written into with a valid value of type `T`.
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut data: [[T; R]; C] = unsafe { core::mem::zeroed() };
        for c in 0..C {
            for r in 0..R {
                data[c][r] = match cglinalg_numeric::try_cast(self.data[c][r]) {
                    Some(value) => value,
                    None => return None,
                };
            }
        }

        Some(Matrix { data })
    }
}

impl<S, const R: usize, const C: usize> Matrix<S, R, C>
where
    S: SimdScalar,
{
    /// Compute the transpose of a matrix.
    ///
    /// Given a matrix `m`, the transpose of `m` is the matrix `m_tr` such that
    /// ```text
    /// forall c in 0..C. forall r in 0..R. m_tr[c][r] == m[r][c]
    /// ```
    /// in other words, every column of `m_tr` is a row of `m`, and every row of
    /// `m_tr` is a column of `m`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x4,
    /// #     Matrix4x2,
    /// # };
    /// #
    /// let matrix = Matrix2x4::new(
    ///     1_i32, 1_i32,
    ///     2_i32, 2_i32,
    ///     3_i32, 3_i32,
    ///     4_i32, 4_i32,
    /// );
    /// let expected = Matrix4x2::new(
    ///     1_i32, 2_i32, 3_i32, 4_i32,
    ///     1_i32, 2_i32, 3_i32, 4_i32,
    /// );
    /// let result = matrix.transpose();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn transpose(&self) -> Matrix<S, C, R> {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::zero();
        for c in 0..C {
            for r in 0..R {
                result.data[r][c] = self.data[c][r];
            }
        }

        result
    }

    /// Construct a zero matrix.
    ///
    /// A zero matrix is a matrix in which all of its elements are zero. In
    /// particular, the **(R row, C column)** zero matrix is the matrix `zero`
    /// such that
    /// ```text
    /// forall c in 0..C. forall r in 0..R. zero[c][r] == 0
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix: Matrix4x4<i32> = Matrix4x4::zero();
    ///
    /// assert!(matrix.is_zero());
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self { data: [[S::zero(); R]; C] }
    }

    /// Determine whether a matrix is a zero matrix.
    ///
    /// A zero matrix is a matrix in which all of its elements are zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix: Matrix4x4<i32> = Matrix4x4::zero();
    ///
    /// assert!(matrix.is_zero());
    /// ```
    #[inline]
    pub fn is_zero(&self) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for c in 0..C {
            for r in 0..R {
                result &= self.data[c][r].is_zero();
            }
        }

        result
    }

    /// Compute the product of two matrices component-wise.
    ///
    /// Given two matrices `m1` and `m2` with `rows` rows and `columns` columns,
    /// the component product of `m1` and `m2` is a matrix `m3` with `rows` rows
    /// and `columns` such that
    /// ```text
    /// forall c in 0..C. forall r in 0..R. m3[c][r] := m1[c][r] * m2[c][r]
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let m1 = Matrix3x3::new(
    ///     0_f64, 1_f64, 2_f64,
    ///     3_f64, 4_f64, 5_f64,
    ///     6_f64, 7_f64, 8_f64,
    /// );
    /// let m2 = Matrix3x3::new(
    ///     9_f64,  10_f64, 11_f64,
    ///     12_f64, 13_f64, 14_f64,
    ///     15_f64, 16_f64, 17_f64,
    /// );
    /// let expected = Matrix3x3::new(
    ///     0_f64,  10_f64,  22_f64,
    ///     36_f64, 52_f64,  70_f64,
    ///     90_f64, 112_f64, 136_f64,
    /// );
    /// let result = m1.component_mul(&m2);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn component_mul(&self, other: &Self) -> Self {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] * other.data[c][r];
            }
        }

        result
    }

    /// Compute the product of two matrices component-wise mutably in place.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let mut result = Matrix3x3::new(
    ///     0_f64, 1_f64, 2_f64,
    ///     3_f64, 4_f64, 5_f64,
    ///     6_f64, 7_f64, 8_f64,
    /// );
    /// let other = Matrix3x3::new(
    ///     9_f64,  10_f64, 11_f64,
    ///     12_f64, 13_f64, 14_f64,
    ///     15_f64, 16_f64, 17_f64,
    /// );
    /// let expected = Matrix3x3::new(
    ///     0_f64,  10_f64,  22_f64,
    ///     36_f64, 52_f64,  70_f64,
    ///     90_f64, 112_f64, 136_f64,
    /// );
    /// result.component_mul_assign(&other);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn component_mul_assign(&mut self, other: &Self) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for c in 0..C {
            for r in 0..R {
                self.data[c][r] *= other.data[c][r];
            }
        }
    }
}

impl<S, const R1: usize, const C1: usize> Matrix<S, R1, C1>
where
    S: SimdScalar,
{
    /// Compute the dot product of two matrices.
    ///
    /// # Example
    ///
    /// An example involving integer scalars.
    /// ```
    /// # use cglinalg_core::Matrix2x3;
    /// #
    /// let matrix1 = Matrix2x3::new(
    ///     2_i32,  1_i32,
    ///     0_i32, -1_i32,
    ///     6_i32,  2_i32,
    /// );
    /// let matrix2 = Matrix2x3::new(
    ///      8_i32,  4_i32,
    ///     -3_i32,  1_i32,
    ///      2_i32, -5_i32,
    /// );
    /// let expected = 21_i32;
    /// let result = Matrix2x3::dot(&matrix1, &matrix2);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn dot<const R2: usize, const C2: usize>(&self, other: &Matrix<S, R2, C2>) -> S
    where
        ShapeConstraint: DimEq<Const<R1>, Const<R2>> + DimEq<Const<R2>, Const<R1>>,
        ShapeConstraint: DimEq<Const<C1>, Const<C2>> + DimEq<Const<C2>, Const<C1>>,
    {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = S::zero();
        for r in 0..R1 {
            for c in 0..C1 {
                result += self.data[c][r] * other.data[c][r]
            }
        }

        result
    }

    /// Compute the matrix product of the transpose of `self` with `other`.
    ///
    /// The function `tr_mul` satisfies the following property: Given a matrix
    /// `m2` with `R1` rows and `C1` columns, and a matrix `m2` with `R2` rows
    /// and `C2` columns, such that `R1 == R2`, the transpose product of `m1`
    /// and `m2` is given by
    /// ```text
    /// tr_mul(m1, m2) := transpose(m1) * m2
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Matrix3x2,
    /// # };
    /// #
    /// let matrix1 = Matrix3x2::new(
    ///     1_i32, 3_i32, 5_i32,
    ///     2_i32, 4_i32, 6_i32,
    /// );
    /// let matrix2 = Matrix3x2::new(
    ///     7_i32, 9_i32,  11_i32,
    ///     8_i32, 10_i32, 12_i32,
    /// );
    /// let expected = Matrix2x2::new(
    ///     89_i32, 116_i32,
    ///     98_i32, 128_i32,
    /// );
    /// let result = matrix1.tr_mul(&matrix2);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let matrix1_tr = matrix1.transpose();
    /// let result_tr = matrix1_tr * matrix2;
    ///
    /// assert_eq!(result_tr, expected);
    /// ```
    #[inline]
    pub fn tr_mul<const R2: usize, const C2: usize, const C1C2: usize>(&self, other: &Matrix<S, R2, C2>) -> Matrix<S, C1, C2>
    where
        ShapeConstraint: CanTransposeMultiply<Const<R1>, Const<C1>, Const<R2>, Const<C2>>,
        ShapeConstraint: DimMul<Const<C1>, Const<C2>, Output = Const<C1C2>>,
        ShapeConstraint: DimMul<Const<C2>, Const<C1>, Output = Const<C1C2>>,
    {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::zero();
        for c1 in 0..C1 {
            for c2 in 0..C2 {
                let mut result_c1c2 = S::zero();
                for r in 0..R1 {
                    result_c1c2 += self.data[c1][r] * other.data[c2][r];
                }

                result[c2][c1] = result_c1c2;
            }
        }

        result
    }

    /// Compute the dot product between the transpose of `self` and `other`.
    ///
    /// Given a matrix `m1` with `R1` rows and `C1` columns, and a matrix `m2` with
    /// `R2` rows and `C2` columns, such that `C1 == R2` and `R1 == C2`, the transpose
    /// dot product of `m1` and `m2` is given by
    /// ```text
    /// tr_dot(m1, m2) := dot(transpose(m1), m2)
    /// ```
    /// where `transpose(m1)` has a shape of `C1` rows and `R1` columns, so that we can
    /// indeed compute the matrix dot product for `transpose(m1)` and `m2`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x3,
    /// #     Matrix3x2,
    /// # };
    /// #
    /// let matrix1 = Matrix2x3::new(
    ///     1_i32, 4_i32,
    ///     2_i32, 5_i32,
    ///     3_i32, 6_i32,
    /// );
    /// let matrix2 = Matrix3x2::new(
    ///     7_i32, 9_i32,  11_i32,
    ///     8_i32, 10_i32, 12_i32,
    /// );
    /// let expected = 212_i32;
    /// let result = matrix1.tr_dot(&matrix2);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let matrix1_tr = matrix1.transpose();
    /// let result_tr = matrix1_tr.dot(&matrix2);
    ///
    /// assert_eq!(result_tr, expected);
    /// ```
    #[inline]
    pub fn tr_dot<const R2: usize, const C2: usize>(&self, other: &Matrix<S, R2, C2>) -> S
    where
        ShapeConstraint: DimEq<Const<R1>, Const<C2>> + DimEq<Const<C2>, Const<R1>>,
        ShapeConstraint: DimEq<Const<R2>, Const<C1>> + DimEq<Const<C1>, Const<R2>>,
    {
        let mut result = S::zero();
        for c in 0..R1 {
            for r in 0..C1 {
                result += self.data[r][c] * other.data[c][r];
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> Matrix<S, R, C>
where
    S: SimdScalarSigned,
{
    /// Mutably negate the elements of a matrix in place.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let mut result = Matrix4x4::new(
    ///     1_i32,  2_i32,  3_i32,  4_i32,
    ///     5_i32,  6_i32,  7_i32,  8_i32,
    ///     9_i32,  10_i32, 11_i32, 12_i32,
    ///     13_i32, 14_i32, 15_i32, 16_i32,
    /// );
    /// let expected = Matrix4x4::new(
    ///     -1_i32,  -2_i32,  -3_i32,  -4_i32,
    ///     -5_i32,  -6_i32,  -7_i32,  -8_i32,
    ///     -9_i32,  -10_i32, -11_i32, -12_i32,
    ///     -13_i32, -14_i32, -15_i32, -16_i32,
    /// );
    /// result.neg_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn neg_mut(&mut self) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for c in 0..C {
            for r in 0..R {
                self.data[c][r] = -self.data[c][r];
            }
        }
    }
}

impl<S, const R: usize, const C: usize> Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    /// Linearly interpolate between two matrices.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let matrix0 = Matrix3x3::new(
    ///     0_f64, 0_f64, 0_f64,
    ///     1_f64, 1_f64, 1_f64,
    ///     2_f64, 2_f64, 2_f64,
    /// );
    /// let matrix1 = Matrix3x3::new(
    ///     3_f64, 3_f64, 3_f64,
    ///     4_f64, 4_f64, 4_f64,
    ///     5_f64, 5_f64, 5_f64,
    /// );
    /// let amount = 0.5_f64;
    /// let expected = Matrix3x3::new(
    ///     1.5_f64, 1.5_f64, 1.5_f64,
    ///     2.5_f64, 2.5_f64, 2.5_f64,
    ///     3.5_f64, 3.5_f64, 3.5_f64,
    /// );
    /// let result = matrix0.lerp(&matrix1, amount);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Self, amount: S) -> Self {
        self + ((other - self) * amount)
    }

    /// Returns `true` if the elements of a matrix are all finite.
    /// Otherwise, it returns `false`.
    ///
    /// A matrix is finite when all of its elements are finite. This is useful
    /// for vector and matrix types working with fixed precision floating point
    /// values.
    ///
    /// # Example (Finite Matrix)
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_f64,  2_f64,  3_f64,  4_f64,
    ///     5_f64,  6_f64,  7_f64,  8_f64,
    ///     9_f64,  10_f64, 11_f64, 12_f64,
    ///     13_f64, 14_f64, 15_f64, 16_f64,
    /// );
    ///
    /// assert!(matrix.is_finite());
    /// ```
    ///
    /// # Example (Not A Finite Matrix)
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_f64,             2_f64,             3_f64,             4_f64,
    ///     f64::NAN,          f64::NAN,          f64::NAN,          f64::NAN,
    ///     f64::INFINITY,     f64::INFINITY,     f64::INFINITY,     f64::INFINITY,
    ///     f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY,
    /// );
    ///
    /// assert!(!matrix.is_finite());
    /// ```
    #[inline]
    pub fn is_finite(&self) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for c in 0..C {
            for r in 0..R {
                result &= self.data[c][r].is_finite();
            }
        }

        result
    }
}

impl<S, const N: usize> Matrix<S, N, N>
where
    S: SimdScalar,
{
    /// Mutably transpose a square matrix in place.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let mut result = Matrix4x4::new(
    ///     1_i32, 1_i32, 1_i32, 1_i32,
    ///     2_i32, 2_i32, 2_i32, 2_i32,
    ///     3_i32, 3_i32, 3_i32, 3_i32,
    ///     4_i32, 4_i32, 4_i32, 4_i32,
    /// );
    /// let expected = Matrix4x4::new(
    ///     1_i32, 2_i32, 3_i32, 4_i32,
    ///     1_i32, 2_i32, 3_i32, 4_i32,
    ///     1_i32, 2_i32, 3_i32, 4_i32,
    ///     1_i32, 2_i32, 3_i32, 4_i32,
    /// );
    /// result.transpose_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn transpose_mut(&mut self) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for i in 0..N {
            for j in 0..i {
                self.swap((i, j), (j, i));
            }
        }
    }

    /// Compute an identity matrix.
    ///
    /// An identity matrix is a matrix where the diagonal elements are one
    /// and the off-diagonal elements are zero. In particular, the identity
    /// matrix is the matrix `identity` such that
    /// ```text
    /// forall i in 0..N. identity[i][i] == 1
    /// forall i, j in 0..N. i != j ==> identity[i][j] == 0
    /// ```
    /// In other words, every off-diagonal element is zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let result = Matrix3x3::identity();
    /// let expected = Matrix3x3::new(
    ///     1_i32, 0_i32, 0_i32,
    ///     0_i32, 1_i32, 0_i32,
    ///     0_i32, 0_i32, 1_i32,
    /// );
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::zero();
        for i in 0..N {
            result[i][i] = S::one();
        }

        result
    }

    /// Determine whether a matrix is an identity matrix.
    ///
    /// An identity matrix is a matrix where the diagonal elements are one
    /// and the off-diagonal elements are zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix: Matrix4x4<i32> = Matrix4x4::identity();
    ///
    /// assert!(matrix.is_identity());
    /// ```
    #[inline]
    pub fn is_identity(&self) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for i in 0..N {
            for j in 0..i {
                result &= self.data[i][j].is_zero() && self.data[j][i].is_zero();
            }
        }

        // PERFORMANCE: The const loop should get unrolled during optimization.
        for i in 0..N {
            result &= self.data[i][i].is_one();
        }

        result
    }

    /// Construct a new diagonal matrix from a given value where
    /// each element along the diagonal is equal to `value`. The resulting
    /// matrix `matrix` satisfies the predicate
    /// ```text
    /// forall i in 0..N. matrix[i][i] == value
    /// forall i, j in 0..N. i != j ==> matrix[i][j] == 0
    /// ```
    /// In other words, every off-diagonal element is zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let result = Matrix4x4::from_diagonal_value(4_i32);
    /// let expected = Matrix4x4::new(
    ///     4_i32, 0_i32, 0_i32, 0_i32,
    ///     0_i32, 4_i32, 0_i32, 0_i32,
    ///     0_i32, 0_i32, 4_i32, 0_i32,
    ///     0_i32, 0_i32, 0_i32, 4_i32,
    /// );
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_diagonal_value(value: S) -> Self {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::zero();
        for i in 0..N {
            result[i][i] = value;
        }

        result
    }

    /// Construct a new diagonal matrix from a vector of values
    /// representing the elements along the diagonal.
    ///
    /// The resulting matrix `matrix` satisfies the predicate. Given a vector of
    /// length `N` `diagonal`
    /// ```text
    /// forall i in 0..N. m[i][i] == diangonal[i]
    /// forall i, j in 0..N. i != j ==> matrix[i][j] == 0
    /// ```
    /// In other words, every off-diagonal element is zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    ///     0_i32, 0_i32, 0_i32, 5_i32,
    /// );
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_diagonal(diagonal: &Vector<S, N>) -> Self {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::zero();
        for i in 0..N {
            result[i][i] = diagonal[i];
        }

        result
    }

    /// Get the diagonal part of a square matrix.
    ///
    /// The resulting vector is a vector of all elements of the matrix `matrix`
    /// on the diagonal. It is the vector `diagonal` such that
    /// ```text
    /// forall i in 0..N. diagonal[i] == matrix[i][i]
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_i32,  2_i32,  3_i32,  4_i32,
    ///     5_i32,  6_i32,  7_i32,  8_i32,
    ///     9_i32,  10_i32, 11_i32, 12_i32,
    ///     13_i32, 14_i32, 15_i32, 16_i32,
    /// );
    /// let expected = Vector4::new(1_i32, 6_i32, 11_i32, 16_i32);
    /// let result = matrix.diagonal();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn diagonal(&self) -> Vector4<S> {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Vector::zero();
        for i in 0..N {
            result[i] = self.data[i][i];
        }

        result
    }

    /// Compute the trace of a square matrix.
    ///
    /// The trace of a matrix is the sum of the diagonal elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_i32, 2_i32, 3_i32,
    ///     4_i32, 5_i32, 6_i32,
    ///     7_i32, 8_i32, 9_i32,
    /// );
    ///
    /// assert_eq!(matrix.trace(), 15_i32);
    /// ```
    #[inline]
    pub fn trace(&self) -> S {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = S::zero();
        for i in 0..N {
            result += self.data[i][i];
        }

        result
    }
}

impl<S, const N: usize> Matrix<S, N, N>
where
    S: SimdScalarFloat,
{
    /// Determine whether a square matrix is a diagonal matrix.
    ///
    /// A square matrix is a diagonal matrix if every off-diagonal
    /// element is zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let diagonal_matrix = Matrix3x3::new(
    ///     1_f32, 0_f32, 0_f32,
    ///     0_f32, 2_f32, 0_f32,
    ///     0_f32, 0_f32, 3_f32,    
    /// );
    /// let zero: Matrix3x3<f32> = Matrix3x3::zero();
    /// let identity: Matrix3x3<f32> = Matrix3x3::identity();
    /// let nondiagonal_matrix = Matrix3x3::new(
    ///     0_f32, 1_f32, 1_f32,
    ///     1_f32, 0_f32, 1_f32,
    ///     1_f32, 1_f32, 0_f32,
    /// );
    ///
    /// assert!(diagonal_matrix.is_diagonal());
    /// assert!(zero.is_diagonal());
    /// assert!(identity.is_diagonal());
    /// assert!(!nondiagonal_matrix.is_diagonal());
    /// ```
    #[inline]
    pub fn is_diagonal(&self) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for i in 0..N {
            for j in 0..i {
                result &= ulps_eq!(self.data[i][j], S::zero(), abs_diff_all <= S::machine_epsilon(), ulps_all <= S::default_ulps()) 
                    && ulps_eq!(self.data[j][i], S::zero(), abs_diff_all <= S::machine_epsilon(), ulps_all <= S::default_ulps());
            }
        }

        result
    }

    /// Determine whether a matrix is symmetric.
    ///
    /// A matrix is symmetric when element `(i, j)` is equal to element `(j, i)`
    /// for each row `i` and column `j`. Otherwise, it is not a symmetric matrix.
    /// Every diagonal matrix is a symmetric matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let symmetric_matrix = Matrix3x3::new(
    ///     1_f32, 2_f32, 3_f32,
    ///     2_f32, 1_f32, 2_f32,
    ///     3_f32, 2_f32, 1_f32,
    /// );
    /// let asymmetric_matrix = Matrix3x3::new(
    ///     1_f32, 2_f32, 3_f32,
    ///     4_f32, 5_f32, 6_f32,
    ///     7_f32, 8_f32, 9_f32,
    /// );
    /// let zero: Matrix3x3<f32> = Matrix3x3::zero();
    /// let identity: Matrix3x3<f32> = Matrix3x3::identity();
    ///
    /// assert!(symmetric_matrix.is_symmetric());
    /// assert!(!asymmetric_matrix.is_symmetric());
    /// assert!(zero.is_symmetric());
    /// assert!(identity.is_symmetric());
    /// ```
    #[inline]
    pub fn is_symmetric(&self) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for i in 0..N {
            for j in 0..i {
                result &= ulps_eq!(self.data[i][j], self.data[j][i], abs_diff_all <= S::machine_epsilon(), ulps_all <= S::default_ulps());
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> Default for Matrix<S, R, C>
where
    S: SimdScalar,
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<S, const R: usize, const C: usize> fmt::Display for Matrix<S, R, C>
where
    S: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Matrix{}x{} [", R, C).unwrap();

        for c in 0..(C - 1) {
            write!(formatter, "[").unwrap();
            for r in 0..(R - 1) {
                write!(formatter, "{}, ", self.data[c][r]).unwrap();
            }
            write!(formatter, "{}], ", self.data[c][R - 1]).unwrap();
        }

        write!(formatter, "[").unwrap();
        for r in 0..(R - 1) {
            write!(formatter, "{}, ", self.data[C - 1][r]).unwrap();
        }
        write!(formatter, "{}]", self.data[C - 1][R - 1]).unwrap();

        write!(formatter, "]")
    }
}


impl<S, const R: usize, const C: usize> Matrix<S, R, C>
where
    S: SimdScalarSigned,
{
    /// Calculate the norm of a matrix with respect to the supplied [`Norm`] type.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     FrobeniusNorm,
    /// #     L1MatrixNorm,
    /// #     LinfMatrixNorm,
    /// #     Matrix3x3,
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_f64, 2_f64, 3_f64,
    ///     5_f64, 6_f64, 7_f64,
    ///     8_f64, 9_f64, 10_f64,
    /// );
    /// let l1_norm = L1MatrixNorm::new();
    /// let linf_norm = LinfMatrixNorm::new();
    /// let frobenius_norm = FrobeniusNorm::new();
    ///
    /// assert_eq!(matrix.apply_norm(&l1_norm), 27_f64);
    /// assert_eq!(matrix.apply_norm(&linf_norm), 20_f64);
    /// 
    /// let result = matrix.apply_norm(&frobenius_norm);
    /// let expected = 19.209372712298546_f64;
    /// 
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn apply_norm(&self, norm: &impl Norm<Matrix<S, R, C>, Output = S>) -> S {
        norm.norm(self)
    }

    /// Calculate the metric distance between two matrices with respect to the
    /// supplied [`Norm`] type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     FrobeniusNorm,
    /// #     L1MatrixNorm,
    /// #     LinfMatrixNorm,
    /// #     Matrix3x3,
    /// # };
    /// #
    /// let matrix1 = Matrix3x3::new(
    ///     1_f64, 2_f64, 3_f64,
    ///     5_f64, 6_f64, 7_f64,
    ///     8_f64, 9_f64, 10_f64,
    /// );
    /// let matrix2 = Matrix3x3::new(
    ///     0_f64, -5_f64,  6_f64,
    ///    -3_f64,  1_f64,  20_f64,
    ///     7_f64,  12_f64, 4_f64,
    /// );
    /// let l1_norm = L1MatrixNorm::new();
    /// let linf_norm = LinfMatrixNorm::new();
    /// let frobenius_norm = FrobeniusNorm::new();
    ///
    /// assert_eq!(matrix1.apply_metric_distance(&matrix2, &l1_norm), 26_f64);
    /// assert_eq!(matrix1.apply_metric_distance(&matrix2, &linf_norm), 22_f64);
    /// 
    /// let result = matrix1.apply_metric_distance(&matrix2, &frobenius_norm);
    /// let expected = 19.05255888325765_f64;
    /// 
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn apply_metric_distance(&self, other: &Self, norm: &impl Norm<Matrix<S, R, C>, Output = S>) -> S {
        norm.metric_distance(self, other)
    }

    /// Compute the squared **Frobenius** norm of a matrix.
    ///
    /// The squared **Frobenius** norm of a matrix is the sum of the squares of all
    /// the elements of the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix = Matrix4x4::new(
    ///     -7_i32,  1_i32, 5_i32, 0_i32,
    ///      1_i32, -5_i32, 8_i32, 2_i32,
    ///      5_i32,  6_i32, 3_i32, 6_i32,
    ///      0_i32,  4_i32, 0_i32, 15_i32,
    /// );
    /// let expected = 516_i32;
    /// let result = matrix.norm_squared();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn norm_squared(&self) -> S {
        self.dot(self)
    }

    /// Compute the squared **Frobenius** norm of a matrix.
    ///
    /// This is a synonym for [`Matrix::norm_squared`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix = Matrix4x4::new(
    ///     -7_i32,  1_i32, 5_i32, 0_i32,
    ///      1_i32, -5_i32, 8_i32, 2_i32,
    ///      5_i32,  6_i32, 3_i32, 6_i32,
    ///      0_i32,  4_i32, 0_i32, 15_i32,
    /// );
    /// let expected = 516_i32;
    /// let result = matrix.norm_squared();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn magnitude_squared(&self) -> S {
        self.norm_squared()
    }

    /// Compute the squared metric distance between two matrices with respect
    /// to the metric induced by the **Frobenius** norm.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let matrix1 = Matrix3x3::new(
    ///     -7_i32,  5_i32, -9_i32,
    ///      1_i32, -5_i32,  8_i32,
    ///      5_i32,  6_i32,  3_i32,
    /// );
    /// let matrix2 = Matrix3x3::new(
    ///     2_i32, 6_i32, 1_i32,
    ///     1_i32, 2_i32, 8_i32,
    ///     3_i32, 1_i32, 3_i32,
    /// );
    /// let expected = 260_i32;
    /// let result = matrix1.metric_distance_squared(&matrix2);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn metric_distance_squared(&self, other: &Self) -> S {
        (self - other).norm_squared()
    }
}

impl<S, const R: usize, const C: usize> Matrix<S, R, C>
where
    S: SimdScalarSigned + SimdScalarOrd,
{
    /// Compute the **L1** norm of a matrix.
    ///
    /// The matrix **L1** norm is also called the **maximum column sum norm**.
    ///
    /// # Example
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let matrix = Matrix3x3::new(
    ///     -3_i32, 2_i32, 0_i32,
    ///      5_i32, 6_i32, 2_i32,
    ///      7_i32, 4_i32, 8_i32,
    /// );
    /// let expected = 19_i32;
    /// let result = matrix.l1_norm();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn l1_norm(&self) -> S {
        let mut result = S::zero();
        for c in 0..C {
            result = S::max(result, self[c].l1_norm());
        }

        result
    }

    /// Compute the **L-infinity** norm of a matrix.
    ///
    /// The matrix **L-infinity** norm is also called the **maximum row sum norm**.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix = Matrix4x4::new(
    ///     -7_i32,  1_i32, 5_i32, 0_i32,
    ///      1_i32, -5_i32, 8_i32, 2_i32,
    ///      5_i32,  6_i32, 3_i32, 6_i32,
    ///      0_i32,  4_i32, 0_i32, 15_i32,
    /// );
    /// let expected = 23_i32;
    /// let result = matrix.linf_norm();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn linf_norm(&self) -> S {
        let mut result = S::zero();
        for r in 0..R {
            let mut row_sum = S::zero();
            for c in 0..C {
                row_sum += self.data[c][r].abs();
            }

            result = S::max(result, row_sum);
        }

        result
    }
}

impl<S, const R: usize, const C: usize> Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    /// Compute the **Frobenius** norm of a matrix.
    ///
    /// The squared **Frobenius** norm of a matrix is the sum of the squares of all
    /// the elements of the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix = Matrix4x4::new(
    ///     -7_f64,  1_f64, 5_f64, 0_f64,
    ///      1_f64, -5_f64, 8_f64, 2_f64,
    ///      5_f64,  6_f64, 3_f64, 6_f64,
    ///      0_f64,  4_f64, 0_f64, 15_f64,
    /// );
    /// let expected = 22.715633383201094;
    /// let result = matrix.norm();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn norm(&self) -> S {
        self.norm_squared().sqrt()
    }

    /// Compute the **Frobenius** norm of a matrix.
    ///
    /// This is a synonym for [`Matrix::norm`].
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix = Matrix4x4::new(
    ///     -7_f64,  1_f64, 5_f64, 0_f64,
    ///      1_f64, -5_f64, 8_f64, 2_f64,
    ///      5_f64,  6_f64, 3_f64, 6_f64,
    ///      0_f64,  4_f64, 0_f64, 15_f64,
    /// );
    /// let expected = 22.715633383201094;
    /// let result = matrix.magnitude();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn magnitude(&self) -> S {
        self.norm()
    }

    /// Compute the metric distance between two matrices with respect to the
    /// metric induced by the **Frobenius** norm.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let matrix1 = Matrix3x3::new(
    ///     -7_f64,  5_f64, -9_f64,
    ///      1_f64, -5_f64,  8_f64,
    ///      5_f64,  6_f64,  3_f64,
    /// );
    /// let matrix2 = Matrix3x3::new(
    ///     2_f64, 6_f64, 1_f64,
    ///     1_f64, 2_f64, 8_f64,
    ///     3_f64, 1_f64, 3_f64,
    /// );
    /// let expected = 16.124515496597099;
    /// let result = matrix1.metric_distance(&matrix2);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn metric_distance(&self, other: &Self) -> S {
        (self - other).norm()
    }
}


#[derive(Copy, Clone, Debug)]
pub struct L1MatrixNorm {}

impl L1MatrixNorm {
    #[inline]
    pub const fn new() -> Self {
        Self {}
    }
}

impl<S, const R: usize, const C: usize> Norm<Matrix<S, R, C>> for L1MatrixNorm
where
    S: SimdScalarSigned + SimdScalarOrd,
{
    type Output = S;

    #[inline]
    fn norm(&self, rhs: &Matrix<S, R, C>) -> Self::Output {
        rhs.l1_norm()
    }

    #[inline]
    fn metric_distance(&self, lhs: &Matrix<S, R, C>, rhs: &Matrix<S, R, C>) -> Self::Output {
        self.norm(&(lhs - rhs))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FrobeniusNorm {}

impl FrobeniusNorm {
    #[inline]
    pub const fn new() -> Self {
        Self {}
    }
}

impl<S, const R: usize, const C: usize> Norm<Matrix<S, R, C>> for FrobeniusNorm
where
    S: SimdScalarFloat,
{
    type Output = S;

    #[inline]
    fn norm(&self, rhs: &Matrix<S, R, C>) -> Self::Output {
        rhs.norm()
    }

    #[inline]
    fn metric_distance(&self, lhs: &Matrix<S, R, C>, rhs: &Matrix<S, R, C>) -> Self::Output {
        lhs.metric_distance(rhs)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LinfMatrixNorm {}

impl LinfMatrixNorm {
    #[inline]
    pub const fn new() -> Self {
        Self {}
    }
}

impl<S, const R: usize, const C: usize> Norm<Matrix<S, R, C>> for LinfMatrixNorm
where
    S: SimdScalarSigned + SimdScalarOrd,
{
    type Output = S;

    #[inline]
    fn norm(&self, rhs: &Matrix<S, R, C>) -> Self::Output {
        rhs.linf_norm()
    }

    #[inline]
    fn metric_distance(&self, lhs: &Matrix<S, R, C>, rhs: &Matrix<S, R, C>) -> Self::Output {
        self.norm(&(lhs - rhs))
    }
}

impl<S, const R: usize, const C: usize> Normed for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    type Output = S;

    #[inline]
    fn norm_squared(&self) -> Self::Output {
        self.norm_squared()
    }

    #[inline]
    fn norm(&self) -> Self::Output {
        self.norm()
    }

    #[inline]
    fn scale(&self, scale: Self::Output) -> Self {
        self * scale
    }

    #[inline]
    fn scale_mut(&mut self, scale: Self::Output) {
        *self = self.scale(scale);
    }

    #[inline]
    fn unscale(&self, scale: Self::Output) -> Self {
        self * (Self::Output::one() / scale)
    }

    #[inline]
    fn unscale_mut(&mut self, scale: Self::Output) {
        *self = self.unscale(scale);
    }

    #[inline]
    fn normalize(&self) -> Self {
        self * (S::one() / self.norm())
    }

    #[inline]
    fn normalize_mut(&mut self) -> Self::Output {
        let norm = self.norm();
        *self = self.normalize();

        norm
    }

    #[inline]
    fn try_normalize(&self, threshold: Self::Output) -> Option<Self> {
        let norm = self.norm();
        if norm <= threshold {
            None
        } else {
            Some(self.normalize())
        }
    }

    #[inline]
    fn try_normalize_mut(&mut self, threshold: Self::Output) -> Option<Self::Output> {
        let norm = self.norm();
        if norm <= threshold {
            None
        } else {
            Some(self.normalize_mut())
        }
    }

    #[inline]
    fn distance_squared(&self, other: &Self) -> Self::Output {
        self.metric_distance_squared(other)
    }

    #[inline]
    fn distance(&self, other: &Self) -> Self::Output {
        self.metric_distance(other)
    }
}

impl<S, const N: usize> Matrix<S, N, N>
where
    S: SimdScalar,
{
    /// Construct a uniform scaling matrix.
    ///
    /// The matrix applies the same scale factor to all dimensions, so each
    /// component of a vector will be scaled by the same factor. In particular,
    /// calling [`Matrix::from_scale(scale)`] is equivalent to calling
    /// [`Matrix::from_nonuniform_scale`] with `scale` as each entry in the vector
    /// of scale factors.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// # };
    /// #
    /// let scale = 11_i32;
    /// let matrix = Matrix2x2::from_scale(scale);
    /// let vector = Vector2::new(1_i32, 2_i32);
    /// let expected = Vector2::new(11_i32, 22_i32);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_scale(scale: S) -> Self {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::identity();
        for i in 0..N {
            result[i][i] = scale;
        }

        result
    }

    /// Construct a general scaling matrix.
    ///
    /// This is the most general case for scaling matrices: the scale factor
    /// in each dimension need not be identical.
    ///
    /// # Example (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// # };
    /// #
    /// let scale_x = 3_i32;
    /// let scale_y = 5_i32;
    /// let scale_vector = Vector2::new(scale_x, scale_y);
    /// let matrix = Matrix2x2::from_nonuniform_scale(&scale_vector);
    /// let vector = Vector2::new(1_i32, 2_i32);
    /// let expected = Vector2::new(3_i32, 10_i32);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// # Example (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let scale_x = 5_i32;
    /// let scale_y = 10_i32;
    /// let scale_z = 15_i32;
    /// let scale_vector = Vector3::new(scale_x, scale_y, scale_z);
    /// let vector = Vector3::new(1_i32, 1_i32, 1_i32);
    /// let matrix = Matrix3x3::from_nonuniform_scale(&scale_vector);
    /// let expected = Vector3::new(5_i32, 10_i32, 15_i32);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_nonuniform_scale(scale: &Vector<S, N>) -> Self {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::identity();
        for i in 0..N {
            result[i][i] = scale[i];
        }

        result
    }

    /// Construct a uniform affine scaling matrix.
    ///
    /// The matrix applies the same scale factor to all dimensions, so each
    /// component of a vector will be scaled by the same factor. In particular,
    /// given a vector `v`, the resulting matrix `m` satisfies
    /// ```text
    /// forall i in 0..(N - 1). (m * v)[i] == scale * v[i]
    /// ```
    /// where `scale` is the uniform scaling factor, and `N` is the dimensionality
    /// of the matrix. Moreover, the matrix `m` has a form that satisfies
    /// ```text
    /// forall i in 0..(N - 1). m[i][i] == scale
    /// m[N - 1][N - 1] == 1
    /// forall i, j in 0..N. i != j ==> m[i][j] == 0
    /// ```
    /// The uniform affine scaling matrix has the general form
    /// ```text
    /// | scale  0      ...  0       0 |
    /// | 0      scale  ...  0       0 |
    /// | .      .           .       . |
    /// | .      .           .       . |
    /// | 0      ...    ...  scale   0 |
    /// | 0      ...    ...  0       1 |
    /// ```
    /// In particular, this is a special case of the more general form in
    /// [`Self::from_affine_nonuniform_scale`].
    ///
    /// # Examples (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
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
    ///
    /// The form of the uniform affine scaling matrix in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let scale = 5_i32;
    /// let expected = Matrix3x3::new(
    ///     scale, 0_i32, 0_i32,
    ///     0_i32, scale, 0_i32,
    ///     0_i32, 0_i32, 1_i32,
    /// );
    /// let result = Matrix3x3::from_affine_scale(scale);
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// # Examples (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// #
    /// let scale = 4_i32;
    /// let matrix = Matrix4x4::from_affine_scale(scale);
    /// let vector = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
    /// let expected = Vector4::new(4_i32, 8_i32, 12_i32, 4_i32);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// The form of the uniform affine scaling matrix in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// #
    /// let scale = 4_i32;
    /// let expected = Matrix4x4::new(
    ///     scale, 0_i32, 0_i32, 0_i32,
    ///     0_i32, scale, 0_i32, 0_i32,
    ///     0_i32, 0_i32, scale, 0_i32,
    ///     0_i32, 0_i32, 0_i32, 1_i32,
    /// );
    /// let result = Matrix4x4::from_affine_scale(scale);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_affine_scale(scale: S) -> Self {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::identity();
        for i in 0..(N - 1) {
            result[i][i] = scale;
        }

        result
    }
}

impl<S, const N: usize, const NMINUS1: usize> Matrix<S, N, N>
where
    S: SimdScalar,
    ShapeConstraint: DimAdd<Const<NMINUS1>, Const<1>, Output = Const<N>>,
    ShapeConstraint: DimSub<Const<N>, Const<1>, Output = Const<NMINUS1>>,
{
    /// Construct an affine scaling matrix.
    ///
    /// This is the most general case for affine scaling matrices: the scale
    /// factor in each dimension need not be identical. Since this is an
    /// affine matrix, the `w` component is unaffected.
    ///
    /// Let `scale` be a vector of scale factors, such that `scale[i]` is the
    /// scale factor for component `i` of a vector. Let `m` be the affine
    /// scaling matrix corresponding to `scale`. The matrix `m` satisfies the
    /// following: given a vector `v`
    /// ```text
    /// forall i in 0..(N - 1). (m * v)[i] == scale[i] * v[i]
    /// ```
    /// where `N` is the dimensionality of `m`. Since `m` is affine, `v` has
    /// dimension `N - 1`. Morever, the matrix `m` has a form that satisfies
    /// ```text
    /// forall i in 0..(N - 1). m[i][i] == scale[i]
    /// m[N - 1][N - 1] == 1
    /// forall i, j in 0..N. i != j ==> m[i][j] == 0
    /// ```
    /// The affine scaling matrix has the general form
    /// ```text
    /// | scale[0] 0         ...  0             0 |
    /// | 0        scale[1]  ...  0             0 |
    /// | .        .              .             . |
    /// | .        .              .             . |
    /// | 0        ...       ...  scale[N - 2]  0 |
    /// | 0        ...       ...  0             1 |
    /// ```
    ///
    /// # Examples (Two Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Vector2,
    /// # };
    /// #
    /// let scale_x = 5_i32;
    /// let scale_y = 10_i32;
    /// let vector = Vector3::new(1_i32, 1_i32, 3_i32);
    /// let matrix = Matrix3x3::from_affine_nonuniform_scale(&Vector2::new(
    ///     scale_x,
    ///     scale_y,
    /// ));
    /// let expected = Vector3::new(5_i32, 10_i32, 3_i32);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// The form of the affine scaling matrix in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector2,
    /// # };
    /// #
    /// let scale_x = 5_i32;
    /// let scale_y = 10_i32;
    /// let expected = Matrix3x3::new(
    ///     scale_x, 0_i32,   0_i32,
    ///     0_i32,   scale_y, 0_i32,
    ///     0_i32,   0_i32,   1_i32
    /// );
    /// let result = Matrix3x3::from_affine_nonuniform_scale(&Vector2::new(
    ///     scale_x,
    ///     scale_y,
    /// ));
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// # Examples (Three Dimensions)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// #
    /// let scale_x = 4_i32;
    /// let scale_y = 6_i32;
    /// let scale_z = 8_i32;
    /// let matrix = Matrix4x4::from_affine_nonuniform_scale(&Vector3::new(
    ///     scale_x,
    ///     scale_y,
    ///     scale_z,
    /// ));
    /// let vector = Vector4::new(1_i32, 1_i32, 1_i32, 1_i32);
    /// let expected = Vector4::new(4_i32, 6_i32, 8_i32, 1_i32);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// The form of the affine scaling matrix in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// # };
    /// let scale_x = 4_i32;
    /// let scale_y = 6_i32;
    /// let scale_z = 8_i32;
    /// let expected = Matrix4x4::new(
    ///     scale_x, 0_i32,   0_i32,   0_i32,
    ///     0_i32,   scale_y, 0_i32,   0_i32,
    ///     0_i32,   0_i32,   scale_z, 0_i32,
    ///     0_i32,   0_i32,   0_i32,   1_i32,
    /// );
    /// let result = Matrix4x4::from_affine_nonuniform_scale(&Vector3::new(
    ///     scale_x,
    ///     scale_y,
    ///     scale_z,
    /// ));
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_affine_nonuniform_scale(scale: &Vector<S, NMINUS1>) -> Self {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::identity();
        for i in 0..(N - 1) {
            result[i][i] = scale[i];
        }

        result
    }

    /// Construct an affine translation matrix.
    ///
    /// Let `distance` be a vector of displacements: component `i` of `distance`
    /// adds `distance[i]` to component `i` of a vector. Let `m` be the affine
    /// translation matrix corresponding to `distance`. Then the matrix `m` satisfies
    /// the following: given a vector `v`
    /// ```text
    /// forall i in 0..(N - 1). (m * v)[i] == v[i] + distance[i]
    /// ```
    /// where `N` is the dimensionality of `m`. Since `m` is affine, `distance`
    /// and `v` have dimensionality `N - 1`. Moreover, form of the matrix `m` is all `
    /// 1`'s along the diagonal, and `0` among all over elements except for the last
    /// column, where each row except the last is the corresponding entry of `distance`.
    /// More precisely, the form of the matrix `m` satisfies
    /// ```text
    /// forall i in 0..N. m[i][i] == 1
    /// forall r in 0..(N - 1). m[N - 1][r] == distance[r]
    /// forall c in 0..(N - 1). forall r in 0..N. c != r ==> m[c][r] == 0
    /// ```
    /// Note that we are indexing in column-major order, so that the last constraint clause
    /// indicates that every entry except the last one in the bottom row is zero. In
    /// particular, the affine translation matrix has the form
    /// ```text
    /// | 1  0   ...   distance[0]     |
    /// | 0  1   ...   distance[1]     |
    /// | .  .   ...   .               |
    /// | .  .   ...   .               |
    /// | 0  0   ...   distance[N - 2] |
    /// | 0  0   ...   1               |
    /// ```
    ///
    /// # Examples (Two Dimensions)
    ///
    /// A homogeneous vector with a zero **z-component** should not translate.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector2,
    /// #     Vector3,
    /// # };
    /// #
    /// let distance = Vector2::new(3_i32, 7_i32);
    /// let matrix = Matrix3x3::from_affine_translation(&distance);
    /// let vector = Vector3::new(1_i32, 1_i32, 0_i32);
    /// let expected = Vector3::new(1_i32, 1_i32, 0_i32);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// A homogeneous vector with a unit **z-component** should translate.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector2,
    /// #     Vector3,
    /// # };
    /// #
    /// let distance = Vector2::new(3_i32, 7_i32);
    /// let matrix = Matrix3x3::from_affine_translation(&distance);
    /// let vector = Vector3::new(1_i32, 1_i32, 1_i32);
    /// let expected = Vector3::new(1_i32 + distance.x, 1_i32 + distance.y, 1_i32);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// The form of the affine translation matrix in two dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector2,
    /// #     Vector3,
    /// # };
    /// #
    /// let distance = Vector2::new(3_i32, 7_i32);
    /// let expected = Matrix3x3::new(
    ///     1_i32,       0_i32,       0_i32,
    ///     0_i32,       1_i32,       0_i32,
    ///     distance[0], distance[1], 1_i32,
    /// );
    /// let result = Matrix3x3::from_affine_translation(&distance);
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// # Examples (Three Dimensions)
    ///
    /// A homogeneous vector with a zero **w-component** should not translate.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// #
    /// let distance = Vector3::new(3_i32, 7_i32, 11_i32);
    /// let matrix = Matrix4x4::from_affine_translation(&distance);
    /// let vector = Vector4::new(1_i32, 1_i32, 1_i32, 0_i32);
    /// let expected = Vector4::new(1_i32, 1_i32, 1_i32, 0_i32);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// A homogeneous vector with a unit **w-component** should translate.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// #
    /// let distance = Vector3::new(3_i32, 7_i32, 11_i32);
    /// let matrix = Matrix4x4::from_affine_translation(&distance);
    /// let vector = Vector4::new(1_i32, 1_i32, 1_i32, 1_i32);
    /// let expected = Vector4::new(
    ///     1_i32 + distance.x,
    ///     1_i32 + distance.y,
    ///     1_i32 + distance.z,
    ///     1_i32,
    /// );
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// The form of the affine translation matrix in three dimensions.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// #
    /// let distance = Vector3::new(3_i32, 7_i32, 11_i32);
    /// let expected = Matrix4x4::new(
    ///     1_i32,       0_i32,       0_i32,       0_i32,
    ///     0_i32,       1_i32,       0_i32,       0_i32,
    ///     0_i32,       0_i32,       1_i32,       0_i32,
    ///     distance[0], distance[1], distance[2], 1_i32,
    /// );
    /// let result = Matrix4x4::from_affine_translation(&distance);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_affine_translation(distance: &Vector<S, NMINUS1>) -> Self {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Matrix::identity();
        for r in 0..(N - 1) {
            result[N - 1][r] = distance[r];
        }

        result
    }
}


impl<S> Matrix1x1<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix1x1;
    /// #
    /// let c0r0 = 1_i32;
    /// let matrix = Matrix1x1::new(c0r0);
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(c0r0: S) -> Self {
        Self {
            data: [[c0r0]],
        }
    }
}

impl<S> Matrix1x1<S>
where
    S: Copy,
{
    /// Convert this 1x1 matrix into a scalar.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix1x1;
    /// #
    /// let vector = Matrix1x1::new(1_i32);
    ///
    /// assert_eq!(vector.to_scalar(), 1_i32);
    /// ```
    #[inline]
    pub const fn to_scalar(&self) -> S {
        self.data[0][0]
    }
}

impl<S> Matrix1x1<S>
where
    S: SimdScalarSigned,
{
    /// Compute the determinant of a matrix.
    ///
    /// The determinant of a matrix is the signed volume of the parallelepiped
    /// swept out by the vectors represented by the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix1x1;
    /// #
    /// let matrix = Matrix1x1::new(-3_f64);
    ///
    /// assert_eq!(matrix.determinant(), -3_f64);
    /// ```
    #[inline]
    pub fn determinant(&self) -> S {
        self.data[0][0]
    }
}

impl<S> Matrix1x1<S>
where
    S: SimdScalarFloat,
{
    /// Compute the inverse of a square matrix, if the inverse exists.
    ///
    /// Given a square matrix `self` Compute the matrix `m` if it exists
    /// such that
    /// ```text
    /// m * self == self * m == 1.
    /// ```
    /// Not every square matrix has an inverse.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix1x1;
    /// #
    /// let matrix = Matrix1x1::new(5_f64);
    /// let expected = Matrix1x1::new(1_f64 / 5_f64);
    /// let result = matrix.inverse().unwrap();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.is_zero() {
            None
        } else {
            let det_inv = S::one() / det;

            Some(Self::new(det_inv))
        }
    }

    /// Determine whether a square matrix has an inverse matrix.
    ///
    /// A matrix is invertible if its determinant is not zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix1x1;
    /// #
    /// let matrix = Matrix1x1::new(-2_f64);
    ///
    /// assert_eq!(matrix.determinant(), -2_f64);
    /// assert!(matrix.is_invertible());
    /// ```
    #[inline]
    pub fn is_invertible(&self) -> bool {
        ulps_ne!(self.determinant(), S::zero(), abs_diff_all <= S::machine_epsilon(), ulps_all <= S::default_ulps())
    }
}


impl<S> Matrix2x2<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix2x2;
    /// #
    /// let c0r0 = 1_i32; let c0r1 = 2_i32;
    /// let c1r0 = 2_i32; let c1r1 = 3_i32;
    /// let matrix = Matrix2x2::new(
    ///     c0r0, c0r1,
    ///     c1r0, c1r1,
    /// );
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// assert_eq!(matrix[0][1], c0r1);
    /// assert_eq!(matrix[1][0], c1r0);
    /// assert_eq!(matrix[1][1], c1r1);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(c0r0: S, c0r1: S, c1r0: S, c1r1: S) -> Self {
        Self {
            data: [
                [c0r0, c0r1],
                [c1r0, c1r1],
            ],
        }
    }
}

impl<S> Matrix2x2<S>
where
    S: SimdScalar,
{
    /// Construct a shearing matrix in two dimensions with respect to
    /// a line passing through the origin `[0, 0]`, using the **x-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    ///
    /// This version of the shearing transformation is a linear transformation because
    /// the origin of the coordinate frame for applying the shearing transformation
    /// is `[0, 0]` so there is no translation term.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the shearing
    /// transformation in general, see [`Matrix2x2::from_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// # };
    /// #
    /// let shear_factor = 4_i32;
    /// let matrix = Matrix2x2::from_shear_xy(shear_factor);
    /// let vertices = [
    ///     Vector2::new( 1_i32,  1_i32),
    ///     Vector2::new(-1_i32,  1_i32),
    ///     Vector2::new(-1_i32, -1_i32),
    ///     Vector2::new( 1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Vector2::new( 1_i32 + shear_factor,  1_i32),
    ///     Vector2::new(-1_i32 + shear_factor,  1_i32),
    ///     Vector2::new(-1_i32 - shear_factor, -1_i32),
    ///     Vector2::new( 1_i32 - shear_factor, -1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_line = [
    ///     Vector2::new( 1_i32, 0_i32),
    ///     Vector2::new(-1_i32, 0_i32),
    ///     Vector2::new( 0_i32, 0_i32),
    /// ];
    /// let expected_in_line = vertices_in_line;
    /// let result_in_line = vertices_in_line.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_line, expected_in_line);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_xy(shear_factor: S) -> Self {
        Self::new(
            S::one(),     S::zero(),
            shear_factor, S::one(),
        )
    }

    /// Construct a shearing matrix in two dimensions with respect to
    /// a line passing through the origin `[0, 0]`, using the **y-axis**
    /// as the shearing direction, and the **x-axis** as the normal vector.
    ///
    /// This version of the shearing transformation is a linear transformation because
    /// the origin of the coordinate frame for applying the shearing transformation
    /// is `[0, 0]` so there is no translation term.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the shearing
    /// transformation in general, see [`Matrix2x2::from_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// # };
    /// #
    /// let shear_factor = 4_i32;
    /// let matrix = Matrix2x2::from_shear_yx(shear_factor);
    /// let vertices = [
    ///     Vector2::new( 1_i32,  1_i32),
    ///     Vector2::new(-1_i32,  1_i32),
    ///     Vector2::new(-1_i32, -1_i32),
    ///     Vector2::new( 1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Vector2::new( 1_i32,  1_i32 + shear_factor),
    ///     Vector2::new(-1_i32,  1_i32 - shear_factor),
    ///     Vector2::new(-1_i32, -1_i32 - shear_factor),
    ///     Vector2::new( 1_i32, -1_i32 + shear_factor),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_line = [
    ///     Vector2::new(0_i32,  1_i32),
    ///     Vector2::new(0_i32, -1_i32),
    ///     Vector2::new(0_i32,  0_i32),
    /// ];
    /// let expected_in_line = vertices_in_line;
    /// let result_in_line = vertices_in_line.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_line, expected_in_line);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_yx(shear_factor: S) -> Self {
        Self::new(
            S::one(),  shear_factor,
            S::zero(), S::one(),
        )
    }
}

impl<S> Matrix2x2<S>
where
    S: SimdScalarFloat,
{
    /// Construct a general shearing matrix in two dimensions with respect to
    /// a line passing through the origin `[0, 0]`.
    ///
    /// This version of the shearing transformation is a linear transformation because
    /// the origin of the coordinate frame for applying the shearing transformation
    /// is `[0, 0]` so there is no translation term to account for.
    ///
    /// # Parameters
    ///
    /// The shearing matrix constructor has the following parameters
    /// * `shear_factor`: The amount by which a point in a line parallel to the shearing
    ///    line gets sheared.
    /// * `direction`: The direction along which the shearing happens.
    /// * `normal`: The normal vector to the shearing line.
    ///
    /// # Discussion
    ///
    /// The displacement of a point with respect to the shearing line is a function
    /// of the signed distance of the point from the shearing line. In particular, it
    /// is a function of the value of the component of the point projected along the
    /// normal vector of the shearing line.
    ///
    /// More precisely, let `v` be the shearing direction, let `n` be a vector normal
    /// to `v`, and let `p` be a point. In two dimensions, the unit vectors `v` and `n`
    /// form a coordinate frame in conjunction with the origin. Let `m` be the shearing
    /// factor. Let `q` be the point that results from applying the shearing
    /// transformation to `p`. The point `q` is defined precisely as
    /// ```text
    /// q := p + (m * p_n) * v
    /// ```
    /// where `p_n` is the component of `p` projected onto the normal vector `n`.
    /// In particular, `p_n := dot(p, n)`, so `q` is given by
    /// ```text
    /// q == p + (m * dot(p, n)) * v
    ///   == I * p + (m * dot(p, n)) * v == I * p + m * (dot(p, n) * v)
    ///   == I * p + m * (v * n^T) * p
    ///   == (I + m * (v * n^T)) * p
    /// ```
    /// where `v * n^T` denotes the outer product of `v` and `n`. The shearing matrix
    /// in geometric form is given by
    /// ```text
    /// M := I + m * (v * n^T) == I + m * outer(v, n)
    /// ```
    /// where `I` denotes the identity matrix. In the standard basis in Euclidean
    /// space, the outer product of `v` and `n` is given by
    /// ```text
    /// outer(v, n) := | v.x * n.x   v.x * n.y |
    ///                | v.y * n.x   v.y * n.y |
    /// ```
    /// so the right-hand side of the expression for the shearing matrix is
    /// ```text
    /// I + m * outer(v, n) == | 1 0 | + m * | v.x * n.x   v.x * n.y |
    ///                        | 0 1 |       | v.y * n.x   v.y * n.y |
    ///
    ///                     == | 1 + m * v.x * n.x   m * v.x * n.y     |
    ///                        | m * v.y * n.x       1 + m * v.y * n.y |
    /// ```
    /// which leads to the formula used to implement the shearing transformation
    /// ```text
    /// M == | 1 + m * v.x * n.x   m * v.x * n.y     |
    ///      | m * v.y * n.x       1 + m * v.y * n.y |
    /// ```
    ///
    /// # An Equivalent Interpretation Of The Shearing Factor
    ///
    /// The projection of the vector `p` onto `v` is given by `p_v := dot(p, v) * v`.
    /// Observe that
    /// ```text
    /// q - p == m * dot(p, n) * v
    /// p - p_v == p_n * n == dot(p, n) * n
    /// ```
    /// With the two vectors define above as a vector with a purely `v` component,
    /// and a purely `n` component, respectively, the three points form a triangle.
    /// The tangent of the angle `phi` with respect to the normal vector at `p_v`
    /// is then given by
    /// ```text
    /// tan(phi) := (q - p)_v / (p - p_v)_n
    ///          == dot(q - p, v) / dot(p - p_v, n)
    ///          == (m * dot(p, n) * dot(v, v)) / (dot(p, n) * dot(n, n))
    ///          == m * (dot(p, n) / dot(p, n))
    ///          == m
    /// ```
    /// so the shearing factor `m` represents the tangent of the shearing angle `phi`
    /// with respect to the unit normal `n`.
    ///
    ///
    /// # Example
    ///
    /// Shearing a rotated square parallel to the line `y == (1 / 2) * x` along the
    /// line `y == (1 / 2) * x`.
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Unit,
    /// #     Vector2,
    /// # };
    /// # use core::f64;
    /// #
    /// let shear_factor = 3_f64;
    /// let direction = Unit::from_value(Vector2::new(2_f64, 1_f64));
    /// let normal = Unit::from_value(Vector2::new(-1_f64, 2_f64));
    /// let matrix = Matrix2x2::from_shear(shear_factor, &direction, &normal);
    ///
    /// // The square's top and bottom sides run parallel to the line `y == (1 / 2) * x`.
    /// // The square's left and right sides run perpendicular to the line `y == (1 / 2) * x`.
    /// let vertices = [
    ///     Vector2::new( 1_f64 / f64::sqrt(5_f64),  3_f64 / f64::sqrt(5_f64)),
    ///     Vector2::new(-3_f64 / f64::sqrt(5_f64),  1_f64 / f64::sqrt(5_f64)),
    ///     Vector2::new(-1_f64 / f64::sqrt(5_f64), -3_f64 / f64::sqrt(5_f64)),
    ///     Vector2::new( 3_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64)),
    /// ];
    /// let expected = [
    ///     Vector2::new(
    ///         (2_f64 / f64::sqrt(5_f64)) * (1_f64 + shear_factor) - 1_f64 / f64::sqrt(5_f64),
    ///         (1_f64 / f64::sqrt(5_f64)) * (1_f64 + shear_factor) + 2_f64 / f64::sqrt(5_f64),
    ///     ),
    ///     Vector2::new(
    ///         (2_f64 / f64::sqrt(5_f64)) * (-1_f64 + shear_factor) - 1_f64 / f64::sqrt(5_f64),
    ///         (1_f64 / f64::sqrt(5_f64)) * (-1_f64 + shear_factor) + 2_f64 / f64::sqrt(5_f64),
    ///     ),
    ///     Vector2::new(
    ///         (2_f64 / f64::sqrt(5_f64)) * (-1_f64 - shear_factor) + 1_f64 / f64::sqrt(5_f64),
    ///         (1_f64 / f64::sqrt(5_f64)) * (-1_f64 - shear_factor) - 2_f64 / f64::sqrt(5_f64),
    ///     ),
    ///     Vector2::new(
    ///         (2_f64 / f64::sqrt(5_f64)) * (1_f64 - shear_factor) + 1_f64 / f64::sqrt(5_f64),
    ///         (1_f64 / f64::sqrt(5_f64)) * (1_f64 - shear_factor) - 2_f64 / f64::sqrt(5_f64),
    ///     ),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_relative_eq!(result[0], expected[0], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[1], expected[1], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[2], expected[2], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[3], expected[3], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    ///
    /// let vertices_in_line = [
    ///     Vector2::new( 1_f64 / f64::sqrt(5_f64),  1_f64 / (2_f64 * f64::sqrt(5_f64))),
    ///     Vector2::new(-3_f64 / f64::sqrt(5_f64), -3_f64 / (2_f64 * f64::sqrt(5_f64))),
    ///     Vector2::new(-1_f64 / f64::sqrt(5_f64), -1_f64 / (2_f64 * f64::sqrt(5_f64))),
    ///     Vector2::new( 3_f64 / f64::sqrt(5_f64),  3_f64 / (2_f64 * f64::sqrt(5_f64))),
    ///     Vector2::new( 0_f64, 0_f64),
    /// ];
    /// let expected_in_line = vertices_in_line;
    /// let result_in_line = vertices_in_line.map(|v| matrix * v);
    ///
    /// assert_relative_eq!(result_in_line[0], expected_in_line[0], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_line[1], expected_in_line[1], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_line[2], expected_in_line[2], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_line[3], expected_in_line[3], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_line[4], expected_in_line[4], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear(shear_factor: S, direction: &Unit<Vector2<S>>, normal: &Unit<Vector2<S>>) -> Self {
        let one = S::one();
        let c0r0 = one + shear_factor * direction[0] * normal[0];
        let c0r1 = shear_factor * direction[1] * normal[0];
        let c1r0 = shear_factor * direction[0] * normal[1];
        let c1r1 = one + shear_factor * direction[1] * normal[1];

        Self::new(
            c0r0, c0r1,
            c1r0, c1r1,
        )
    }
}

impl<S> Matrix2x2<S>
where
    S: SimdScalarSigned,
{
    /// Construct a two-dimensional reflection matrix for reflecting through a
    /// line through the origin in the **xy-plane**.
    ///
    /// # Example
    ///
    /// Here is an example of reflecting a vector across the **x-axis**.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Unit,
    /// #     Vector2,
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
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Unit,
    /// #     Vector2,
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
    pub fn from_reflection(normal: &Unit<Vector2<S>>) -> Self {
        let one = S::one();
        let two = one + one;

        let c0r0 = one - two * normal[0] * normal[0];
        let c0r1 = -two * normal[1] * normal[0];
        let c1r0 = -two * normal[0] * normal[1];
        let c1r1 = one - two * normal[1] * normal[1];


        Self::new(
            c0r0, c0r1,
            c1r0, c1r1,
        )
    }

    /// Compute the determinant of a matrix.
    ///
    /// The determinant of a matrix is the signed volume of the parallelepiped
    /// swept out by the vectors represented by the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix2x2;
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

impl<S> Matrix2x2<S>
where
    S: SimdScalarFloat,
{
    /// Construct a rotation matrix in two dimensions that rotates a vector
    /// in the **xy-plane** by an angle `angle`.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let unit_x = Vector2::unit_x();
    /// let unit_y = Vector2::unit_y();
    /// let matrix = Matrix2x2::from_angle(angle);
    /// let expected = unit_y;
    /// let result = matrix * unit_x;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_angle<A>(angle: A) -> Self
    where
        A: Into<Radians<S>>,
    {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Self::new(
             cos_angle, sin_angle,
            -sin_angle, cos_angle,
        )
    }

    /// Construct a rotation matrix that rotates the shortest angular distance
    /// between two vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// # };
    /// #
    /// let v1 = Vector2::new(1_f64, 1_f64);
    /// let v2 = Vector2::new(-1_f64, 1_f64);
    /// let matrix = Matrix2x2::rotation_between(&v1, &v2);
    /// let vector = Vector2::unit_y();
    /// let expected = -Vector2::unit_x();
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    /// The matrix returned by `rotation_between` should make `v1` and `v2` collinear.
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// # };
    /// #
    /// let v1 = Vector2::new(1_f64, 1_f64);
    /// let v2 = Vector2::new(-1_f64, 1_f64);
    /// let matrix = Matrix2x2::rotation_between(&v1, &v2);
    /// let result = matrix * v1;
    /// let expected = v2;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn rotation_between(v1: &Vector2<S>, v2: &Vector2<S>) -> Self {
        if let (Some(unit_v1), Some(unit_v2)) = (Unit::try_from_value(*v1, S::zero()), Unit::try_from_value(*v2, S::zero())) {
            Self::rotation_between_axis(&unit_v1, &unit_v2)
        } else {
            Self::identity()
        }
    }

    /// Construct a rotation matrix that rotates the shortest angular distance
    /// between two unit vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Unit,
    /// #     Vector2,
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
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    /// The matrix returned by `rotation_between` should make `v1` and `v2` collinear.
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Unit,
    /// #     Vector2,
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
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn rotation_between_axis(v1: &Unit<Vector2<S>>, v2: &Unit<Vector2<S>>) -> Self {
        let cos_angle = v1.as_ref().dot(v2.as_ref());
        let sin_angle = S::sqrt(S::one() - cos_angle * cos_angle);

        Self::from_angle(Radians::atan2(sin_angle, cos_angle))
    }

    /// Compute the inverse of a square matrix, if the inverse exists.
    ///
    /// Given a square matrix `self` Compute the matrix `m` if it exists
    /// such that
    /// ```text
    /// m * self == self * m == 1.
    /// ```
    /// Not every square matrix has an inverse.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix2x2;
    /// #
    /// let matrix = Matrix2x2::new(
    ///     2_f64, 3_f64,
    ///     1_f64, 5_f64,
    /// );
    /// let expected = Matrix2x2::new(
    ///      5_f64 / 7_f64, -3_f64 / 7_f64,
    ///     -1_f64 / 7_f64,  2_f64 / 7_f64,
    /// );
    /// let result = matrix.inverse().unwrap();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.is_zero() {
            None
        } else {
            let det_inv = S::one() / det;

            Some(Matrix2x2::new(
                det_inv *  self.data[1][1], det_inv * -self.data[0][1],
                det_inv * -self.data[1][0], det_inv *  self.data[0][0],
            ))
        }
    }

    /// Determine whether a square matrix has an inverse matrix.
    ///
    /// A matrix is invertible if its determinant is not zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix2x2;
    /// #
    /// let matrix = Matrix2x2::new(
    ///     1_f64, 2_f64,
    ///     2_f64, 1_f64,
    /// );
    ///
    /// assert_eq!(matrix.determinant(), -3_f64);
    /// assert!(matrix.is_invertible());
    /// ```
    #[inline]
    pub fn is_invertible(&self) -> bool {
        ulps_ne!(self.determinant(), S::zero(), abs_diff_all <= S::machine_epsilon(), ulps_all <= S::default_ulps())
    }
}

impl<S> Matrix3x3<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let c0r0 = 1_i32; let c0r1 = 2_i32; let c0r2 = 3_i32;
    /// let c1r0 = 4_i32; let c1r1 = 5_i32; let c1r2 = 6_i32;
    /// let c2r0 = 7_i32; let c2r1 = 8_i32; let c2r2 = 9_i32;
    /// let matrix = Matrix3x3::new(
    ///     c0r0, c0r1, c0r2,
    ///     c1r0, c1r1, c1r2,
    ///     c2r0, c2r1, c2r2,
    /// );
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// assert_eq!(matrix[0][1], c0r1);
    /// assert_eq!(matrix[0][2], c0r2);
    /// assert_eq!(matrix[1][0], c1r0);
    /// assert_eq!(matrix[1][1], c1r1);
    /// assert_eq!(matrix[1][2], c1r2);
    /// assert_eq!(matrix[2][0], c2r0);
    /// assert_eq!(matrix[2][1], c2r1);
    /// assert_eq!(matrix[2][2], c2r2);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(
        c0r0: S, c0r1: S, c0r2: S,
        c1r0: S, c1r1: S, c1r2: S,
        c2r0: S, c2r1: S, c2r2: S) -> Self {

        Self {
            data: [
                [c0r0, c0r1, c0r2],
                [c1r0, c1r1, c1r2],
                [c2r0, c2r1, c2r2],
            ],
        }
    }
}

impl<S> Matrix3x3<S>
where
    S: SimdScalar,
{
    /// Construct a shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **x-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    ///
    /// This version of the shearing transformation is a linear transformation because
    /// the origin of the coordinate frame for applying the shearing transformation
    /// is `[0, 0, 0]` so there is no translation term.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the shearing
    /// transformation in general, see [`Matrix3x3::from_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_factor = 8_i32;
    /// let matrix = Matrix3x3::from_shear_xy(shear_factor);
    /// let vertices = [
    ///     Vector3::new( 1_i32,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32, -1_i32, -1_i32),
    ///     Vector3::new( 1_i32, -1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Vector3::new( 1_i32 + shear_factor,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32 + shear_factor,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32 - shear_factor, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32 - shear_factor, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32 + shear_factor,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32 + shear_factor,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32 - shear_factor, -1_i32, -1_i32),
    ///     Vector3::new( 1_i32 - shear_factor, -1_i32, -1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector3::new( 1_i32, 0_i32,  1_i32),
    ///     Vector3::new(-1_i32, 0_i32,  1_i32),
    ///     Vector3::new(-1_i32, 0_i32, -1_i32),
    ///     Vector3::new( 1_i32, 0_i32, -1_i32),
    ///     Vector3::new( 0_i32, 0_i32,  0_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_xy(shear_factor: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,          zero, zero,
            shear_factor, one,  zero,
            zero,         zero, one,
        )
    }

    /// Construct a shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **x-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    ///
    /// This version of the shearing transformation is a linear transformation because
    /// the origin of the coordinate frame for applying the shearing transformation
    /// is `[0, 0, 0]` so there is no translation term.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the shearing
    /// transformation in general, see [`Matrix3x3::from_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_factor = 8_i32;
    /// let matrix = Matrix3x3::from_shear_xz(shear_factor);
    /// let vertices = [
    ///     Vector3::new( 1_i32,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32, -1_i32, -1_i32),
    ///     Vector3::new( 1_i32, -1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Vector3::new( 1_i32 + shear_factor,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32 + shear_factor,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32 + shear_factor, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32 + shear_factor, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32 - shear_factor,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32 - shear_factor,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32 - shear_factor, -1_i32, -1_i32),
    ///     Vector3::new( 1_i32 - shear_factor, -1_i32, -1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector3::new( 1_i32,  1_i32, 0_i32),
    ///     Vector3::new(-1_i32,  1_i32, 0_i32),
    ///     Vector3::new(-1_i32, -1_i32, 0_i32),
    ///     Vector3::new( 1_i32, -1_i32, 0_i32),
    ///     Vector3::new( 0_i32,  0_i32, 0_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_xz(shear_factor: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,          zero, zero,
            zero,         one,  zero,
            shear_factor, zero, one,
        )
    }

    /// Construct a shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **y-axis**
    /// as the shearing direction, and the **x-axis** as the normal vector.
    ///
    /// This version of the shearing transformation is a linear transformation because
    /// the origin of the coordinate frame for applying the shearing transformation
    /// is `[0, 0, 0]` so there is no translation term.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the shearing
    /// transformation in general, see [`Matrix3x3::from_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_factor = 8_i32;
    /// let matrix = Matrix3x3::from_shear_yx(shear_factor);
    /// let vertices = [
    ///     Vector3::new( 1_i32,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32, -1_i32, -1_i32),
    ///     Vector3::new( 1_i32, -1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Vector3::new( 1_i32,  1_i32 + shear_factor,  1_i32),
    ///     Vector3::new(-1_i32,  1_i32 - shear_factor,  1_i32),
    ///     Vector3::new(-1_i32, -1_i32 - shear_factor,  1_i32),
    ///     Vector3::new( 1_i32, -1_i32 + shear_factor,  1_i32),
    ///     Vector3::new( 1_i32,  1_i32 + shear_factor, -1_i32),
    ///     Vector3::new(-1_i32,  1_i32 - shear_factor, -1_i32),
    ///     Vector3::new(-1_i32, -1_i32 - shear_factor, -1_i32),
    ///     Vector3::new( 1_i32, -1_i32 + shear_factor, -1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector3::new(0_i32,  1_i32,  1_i32),
    ///     Vector3::new(0_i32, -1_i32,  1_i32),
    ///     Vector3::new(0_i32, -1_i32, -1_i32),
    ///     Vector3::new(0_i32,  1_i32, -1_i32),
    ///     Vector3::new(0_i32,  0_i32,  0_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_yx(shear_factor: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,  shear_factor, zero,
            zero, one,          zero,
            zero, zero,         one,
        )
    }

    /// Construct a shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **y-axis**
    /// as the shearing direction, and the **z-axis** as the normal vector.
    ///
    /// This version of the shearing transformation is a linear transformation because
    /// the origin of the coordinate frame for applying the shearing transformation
    /// is `[0, 0, 0]` so there is no translation term.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the shearing
    /// transformation in general, see [`Matrix3x3::from_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_factor = 8_i32;
    /// let matrix = Matrix3x3::from_shear_yz(shear_factor);
    /// let vertices = [
    ///     Vector3::new( 1_i32,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32, -1_i32, -1_i32),
    ///     Vector3::new( 1_i32, -1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Vector3::new( 1_i32,  1_i32 + shear_factor,  1_i32),
    ///     Vector3::new(-1_i32,  1_i32 + shear_factor,  1_i32),
    ///     Vector3::new(-1_i32, -1_i32 + shear_factor,  1_i32),
    ///     Vector3::new( 1_i32, -1_i32 + shear_factor,  1_i32),
    ///     Vector3::new( 1_i32,  1_i32 - shear_factor, -1_i32),
    ///     Vector3::new(-1_i32,  1_i32 - shear_factor, -1_i32),
    ///     Vector3::new(-1_i32, -1_i32 - shear_factor, -1_i32),
    ///     Vector3::new( 1_i32, -1_i32 - shear_factor, -1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector3::new( 1_i32,  1_i32, 0_i32),
    ///     Vector3::new(-1_i32,  1_i32, 0_i32),
    ///     Vector3::new(-1_i32, -1_i32, 0_i32),
    ///     Vector3::new( 1_i32, -1_i32, 0_i32),
    ///     Vector3::new( 0_i32,  0_i32, 0_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_yz(shear_factor: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,  zero,         zero,
            zero, one,          zero,
            zero, shear_factor, one,
        )
    }

    /// Construct a shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **z-axis**
    /// as the shearing direction, and the **x-axis** as the normal vector.
    ///
    /// This version of the shearing transformation is a linear transformation because
    /// the origin of the coordinate frame for applying the shearing transformation
    /// is `[0, 0, 0]` so there is no translation term.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the shearing
    /// transformation in general, see [`Matrix3x3::from_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_factor = 8_i32;
    /// let matrix = Matrix3x3::from_shear_zx(shear_factor);
    /// let vertices = [
    ///     Vector3::new( 1_i32,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32, -1_i32, -1_i32),
    ///     Vector3::new( 1_i32, -1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Vector3::new( 1_i32,  1_i32,  1_i32 + shear_factor),
    ///     Vector3::new(-1_i32,  1_i32,  1_i32 - shear_factor),
    ///     Vector3::new(-1_i32, -1_i32,  1_i32 - shear_factor),
    ///     Vector3::new( 1_i32, -1_i32,  1_i32 + shear_factor),
    ///     Vector3::new( 1_i32,  1_i32, -1_i32 + shear_factor),
    ///     Vector3::new(-1_i32,  1_i32, -1_i32 - shear_factor),
    ///     Vector3::new(-1_i32, -1_i32, -1_i32 - shear_factor),
    ///     Vector3::new( 1_i32, -1_i32, -1_i32 + shear_factor),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector3::new(0_i32,  1_i32,  1_i32),
    ///     Vector3::new(0_i32, -1_i32,  1_i32),
    ///     Vector3::new(0_i32, -1_i32, -1_i32),
    ///     Vector3::new(0_i32,  1_i32, -1_i32),
    ///     Vector3::new(0_i32,  0_i32,  0_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_zx(shear_factor: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,  zero, shear_factor,
            zero, one,  zero,
            zero, zero, one,
        )
    }

    /// Construct a shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **z-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    ///
    /// This version of the shearing transformation is a linear transformation because
    /// the origin of the coordinate frame for applying the shearing transformation
    /// is `[0, 0, 0]` so there is no translation term.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the shearing
    /// transformation in general, see [`Matrix3x3::from_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_factor = 8_i32;
    /// let matrix = Matrix3x3::from_shear_zy(shear_factor);
    /// let vertices = [
    ///     Vector3::new( 1_i32,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32,  1_i32,  1_i32),
    ///     Vector3::new(-1_i32, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32, -1_i32,  1_i32),
    ///     Vector3::new( 1_i32,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32,  1_i32, -1_i32),
    ///     Vector3::new(-1_i32, -1_i32, -1_i32),
    ///     Vector3::new( 1_i32, -1_i32, -1_i32),
    /// ];
    /// let expected = [
    ///     Vector3::new( 1_i32,  1_i32,  1_i32 + shear_factor),
    ///     Vector3::new(-1_i32,  1_i32,  1_i32 + shear_factor),
    ///     Vector3::new(-1_i32, -1_i32,  1_i32 - shear_factor),
    ///     Vector3::new( 1_i32, -1_i32,  1_i32 - shear_factor),
    ///     Vector3::new( 1_i32,  1_i32, -1_i32 + shear_factor),
    ///     Vector3::new(-1_i32,  1_i32, -1_i32 + shear_factor),
    ///     Vector3::new(-1_i32, -1_i32, -1_i32 - shear_factor),
    ///     Vector3::new( 1_i32, -1_i32, -1_i32 - shear_factor),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector3::new( 1_i32, 0_i32,  1_i32),
    ///     Vector3::new(-1_i32, 0_i32,  1_i32),
    ///     Vector3::new(-1_i32, 0_i32, -1_i32),
    ///     Vector3::new( 1_i32, 0_i32, -1_i32),
    ///     Vector3::new( 0_i32, 0_i32,  0_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear_zy(shear_factor: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,  zero, zero,
            zero, one,  shear_factor,
            zero, zero, one,
        )
    }
}

impl<S> Matrix3x3<S>
where
    S: SimdScalarFloat,
{
    /// Construct a general shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`.
    ///
    /// This version of the shearing transformation is a linear transformation because
    /// the origin of the coordinate frame for applying the shearing transformation
    /// is `[0, 0, 0]` so there is no translation term to account for.
    ///
    /// # Parameters
    ///
    /// The shearing matrix constructor has the following parameters
    /// * `shear_factor`: The amount by which a point in a plane parallel to the shearing
    ///    plane gets sheared.
    /// * `direction`: The direction along which the shearing happens.
    /// * `normal`: The normal vector to the shearing plane.
    ///
    /// # Discussion
    ///
    /// The displacement of a point with respect to the shearing plane is a function
    /// of the signed distance of the point from the shearing plane. In particular, it
    /// is a function of the value of the component of the point projected along the
    /// normal vector of the shearing plane.
    ///
    /// More precisely, let `v` be the shearing direction, let `n` be a vector normal
    /// to `v`, and let `p` be a point. In three dimensions, the unit vectors `v`, `n`, and
    /// `v x n` form a coordinate frame in conjunction with the origin. Let `m` be the
    /// shearing factor. Let `q` be the point that results from applying the shearing
    /// transformation to `p`. The point `q` is defined precisely as
    /// ```text
    /// q := p + (m * p_n) * v
    /// ```
    /// where `p_n` is the component of `p` projected onto the normal vector `n`.
    /// In particular, `p_n := dot(p, n)`, so `q` is given by
    /// ```text
    /// q == p + (m * dot(p, n)) * v
    ///   == I * p + (m * dot(p, n)) * v == I * p + m * (dot(p, n) * v)
    ///   == I * p + m * (v * n^T) * p
    ///   == (I + m * (v * n^T)) * p
    /// ```
    /// where `v * n^T` denotes the outer product of `v` and `n`. The shearing matrix
    /// in geometric form is given by
    /// ```text
    /// M := I + m * (v * n^T) == I + m * outer(v, n)
    /// ```
    /// where `I` denotes the identity matrix. In the standard basis in Euclidean
    /// space, the outer product of `v` and `n` is given by
    /// ```text
    ///                | v.x * n.x   v.x * n.y   v.x * n.z |
    /// outer(v, n) := | v.y * n.x   v.y * n.y   v.y * n.z |
    ///                | v.z * n.x   v.z * n.y   v.z * n.z |
    /// ```
    /// so the right-hand side of the expression for the shearing matrix is
    /// ```text
    ///                        | 1 0 0 |       | v.x * n.x   v.x * n.y   v.x * n.z |
    /// I + m * outer(v, n) == | 0 1 0 | + m * | v.y * n.x   v.y * n.y   v.y * n.z |
    ///                        | 0 0 1 |       | v.z * n.x   v.z * n.y   v.z * n.z |
    ///
    ///                        | 1 + m * v.x * n.x   m * v.x * n.y       m * v.x * n.z     |
    ///                     == | m * v.y * n.x       1 + m * v.y * n.y   m * v.y * n.z     |
    ///                        | m * v.z * n.x       m * v.z * n.y       1 + m * v.z * n.z |
    /// ```
    /// which leads to the formula used to implement the shearing transformation
    /// ```text
    ///      | 1 + m * v.x * n.x   m * v.x * n.y       m * v.x * n.z     |
    /// M == | m * v.y * n.x       1 + m * v.y * n.y   m * v.y * n.z     |
    ///      | m * v.z * n.x       m * v.z * n.y       1 + m * v.z * n.z |
    /// ```
    ///
    /// # An Equivalent Interpretation Of The Shearing Factor
    ///
    /// The projection of the vector `p` onto `v` is given by `p_v := dot(p, v) * v`.
    /// Observe that
    /// ```text
    /// q - p == m * dot(p, n) * v
    /// p - p_v == p_n * n == dot(p, n) * n
    /// ```
    /// With the two vectors define above as a vector with a purely `v` component,
    /// and a purely `n` component, respectively, the three points form a triangle.
    /// The tangent of the angle `phi` with respect to the normal vector at `p_v`
    /// is then given by
    /// ```text
    /// tan(phi) := (q - p)_v / (p - p_v)_n
    ///          == dot(q - p, v) / dot(p - p_v, n)
    ///          == (m * dot(p, n) * dot(v, v)) / (dot(p, n) * dot(n, n))
    ///          == m * (dot(p, n) / dot(p, n))
    ///          == m
    /// ```
    /// so the shearing factor `m` represents the tangent of the shearing angle `phi`
    /// with respect to the unit normal `n`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_factor = 8_f64;
    /// let direction = Unit::from_value(Vector3::unit_x());
    /// let normal = Unit::from_value(-Vector3::unit_y());
    /// let matrix = Matrix3x3::new(
    ///      1_f64,        0_f64, 0_f64,
    ///     -shear_factor, 1_f64, 0_f64,
    ///      0_f64,        0_f64, 1_f64,
    /// );
    /// let expected_matrix = matrix;
    /// let result_matrix = Matrix3x3::from_shear(shear_factor, &direction, &normal);
    ///
    /// assert_eq!(result_matrix, expected_matrix);
    ///
    /// let vertices = [
    ///     Vector3::new( 1_f64,  1_f64,  1_f64),
    ///     Vector3::new(-1_f64,  1_f64,  1_f64),
    ///     Vector3::new(-1_f64, -1_f64,  1_f64),
    ///     Vector3::new( 1_f64, -1_f64,  1_f64),
    ///     Vector3::new( 1_f64,  1_f64, -1_f64),
    ///     Vector3::new(-1_f64,  1_f64, -1_f64),
    ///     Vector3::new(-1_f64, -1_f64, -1_f64),
    ///     Vector3::new( 1_f64, -1_f64, -1_f64),
    /// ];
    /// let expected = [
    ///     Vector3::new( 1_f64 - shear_factor,  1_f64,  1_f64),
    ///     Vector3::new(-1_f64 - shear_factor,  1_f64,  1_f64),
    ///     Vector3::new(-1_f64 + shear_factor, -1_f64,  1_f64),
    ///     Vector3::new( 1_f64 + shear_factor, -1_f64,  1_f64),
    ///     Vector3::new( 1_f64 - shear_factor,  1_f64, -1_f64),
    ///     Vector3::new(-1_f64 - shear_factor,  1_f64, -1_f64),
    ///     Vector3::new(-1_f64 + shear_factor, -1_f64, -1_f64),
    ///     Vector3::new( 1_f64 + shear_factor, -1_f64, -1_f64),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector3::new( 1_f64, 0_f64,  1_f64),
    ///     Vector3::new(-1_f64, 0_f64,  1_f64),
    ///     Vector3::new(-1_f64, 0_f64, -1_f64),
    ///     Vector3::new( 1_f64, 0_f64, -1_f64),
    ///     Vector3::new( 0_f64, 0_f64,  0_f64),
    /// ];
    /// // Points in the shearing plane don't move.
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_shear(shear_factor: S, direction: &Unit<Vector3<S>>, normal: &Unit<Vector3<S>>) -> Self {
        let one = S::one();

        let c0r0 = one + shear_factor * direction[0] * normal[0];
        let c0r1 = shear_factor * direction[1] * normal[0];
        let c0r2 = shear_factor * direction[2] * normal[0];

        let c1r0 = shear_factor * direction[0] * normal[1];
        let c1r1 = one + shear_factor * direction[1] * normal[1];
        let c1r2 = shear_factor * direction[2] * normal[1];

        let c2r0 = shear_factor * direction[0] * normal[2];
        let c2r1 = shear_factor * direction[1] * normal[2];
        let c2r2 = one + shear_factor * direction[2] * normal[2];

        Self::new(
            c0r0, c0r1, c0r2,
            c1r0, c1r1, c1r2,
            c2r0, c2r1, c2r2,
        )
    }
}

impl<S> Matrix3x3<S>
where
    S: SimdScalar,
{
    /// Construct an affine shearing matrix in two dimensions with respect to
    /// a line passing through the origin `[0, 0]`, using the **x-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the affine
    /// shearing transformation in general, see [`Matrix3x3::from_affine_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_factor = 3_i32;
    /// let matrix = Matrix3x3::from_affine_shear_xy(shear_factor);
    /// let vertices = [
    ///     Vector3::new( 1_i32,  1_i32, 1_i32),
    ///     Vector3::new(-1_i32,  1_i32, 1_i32),
    ///     Vector3::new(-1_i32, -1_i32, 1_i32),
    ///     Vector3::new( 1_i32, -1_i32, 1_i32),
    /// ];
    /// let expected = [
    ///     Vector3::new( 1_i32 + shear_factor,  1_i32, 1_i32),
    ///     Vector3::new(-1_i32 + shear_factor,  1_i32, 1_i32),
    ///     Vector3::new(-1_i32 - shear_factor, -1_i32, 1_i32),
    ///     Vector3::new( 1_i32 - shear_factor, -1_i32, 1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_line = [
    ///     Vector3::new( 1_i32, 0_i32, 1_i32),
    ///     Vector3::new(-1_i32, 0_i32, 1_i32),
    ///     Vector3::new( 0_i32, 0_i32, 1_i32),
    /// ];
    /// let expected_in_line = vertices_in_line;
    /// let result_in_line = vertices_in_line.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_line, expected_in_line);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_xy(shear_factor: S) -> Self {
        let zero = S::zero();
        let one = S::one();

        Self::new(
            one,          zero, zero,
            shear_factor, one,  zero,
            zero,         zero, one,
        )
    }

    /// Construct an affine shearing matrix in two dimensions with respect to
    /// a line passing through the origin `[0, 0]`, using the **y-axis**
    /// as the shearing direction, and the **x-axis** as the normal vector.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the affine
    /// shearing transformation in general, see [`Matrix3x3::from_affine_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_factor = 3_i32;
    /// let matrix = Matrix3x3::from_affine_shear_yx(shear_factor);
    /// let vertices = [
    ///     Vector3::new( 1_i32,  1_i32, 1_i32),
    ///     Vector3::new(-1_i32,  1_i32, 1_i32),
    ///     Vector3::new(-1_i32, -1_i32, 1_i32),
    ///     Vector3::new( 1_i32, -1_i32, 1_i32),
    /// ];
    /// let expected = [
    ///     Vector3::new( 1_i32,  1_i32 + shear_factor, 1_i32),
    ///     Vector3::new(-1_i32,  1_i32 - shear_factor, 1_i32),
    ///     Vector3::new(-1_i32, -1_i32 - shear_factor, 1_i32),
    ///     Vector3::new( 1_i32, -1_i32 + shear_factor, 1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_line = [
    ///     Vector3::new(0_i32,  1_i32, 1_i32),
    ///     Vector3::new(0_i32, -1_i32, 1_i32),
    ///     Vector3::new(0_i32,  0_i32, 1_i32),
    /// ];
    /// let expected_in_line = vertices_in_line;
    /// let result_in_line = vertices_in_line.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_line, expected_in_line);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_yx(shear_factor: S) -> Self {
        let zero = S::zero();
        let one = S::one();

        Self::new(
            one,  shear_factor, zero,
            zero, one,          zero,
            zero, zero,         one,
        )
    }
}

impl<S> Matrix3x3<S>
where
    S: SimdScalarFloat,
{
    /// Construct a general affine shearing matrix in two dimensions with respect to
    /// a line passing through the origin `origin`, not necessarily `[0, 0]`.
    ///
    /// # Parameters
    ///
    /// The affine shearing matrix constructor has four parameters
    /// * `origin`: The origin of the affine frame for the shearing transformation.
    /// * `shear_factor`: The amount by which a point in a plane parallel to the shearing
    ///    line gets sheared.
    /// * `direction`: The direction along which the shearing happens in the shearing line.
    /// * `normal`: The normal vector to the shearing line.
    ///
    /// # Discussion
    ///
    /// The displacement of a point with respect to the shearing line is a function
    /// of the signed distance of the point from the shearing line. In particular, it
    /// is a function of the value of the component of the difference between the point
    /// and the origin of the affine frame projected along the normal vector of the
    /// shearing line.
    ///
    /// More precisely, let `Q` be the origin of the affine frame for the shearing
    /// transformation, let `v` be the shearing direction, let `n` be a vector normal
    /// to `v`, and let `p` be a point. In two dimensions, the unit vectors `v` and `n`
    /// form a coordinate frame in conjunction with the origin `Q`. Let `m` be the shearing
    /// factor. Let `q` be the point that results from applying the shearing
    /// transformation to `p`. The point `q` is defined precisely as
    /// ```text
    /// q := p + (m * (p - Q)_n) * v
    /// ```
    /// where `(p - Q)_n` is the component of `p - Q` projected onto the normal vector `n`.
    /// In particular, `(p - Q)_n := dot(p - Q, n)`, so `q` is given by
    /// ```text
    /// q == p + (m * dot(p - Q, n)) * v
    ///   == I * p + (m * dot(p - Q, n)) * v == I * p + m * (dot(p - Q, n) * v)
    ///   == I * p + m * dot(p, n) * v - m * dot(Q, n) * v
    ///   == I * p + m * (v * n^T) * p - m * dot(Q, n) * v
    ///   == (I + m * (v * n^T)) * p - m * dot(Q, n) * v
    /// ```
    /// where `v * n^T` denotes the outer product of `v` and `n`. The shearing matrix
    /// in geometric form is given by
    /// ```text
    /// M := | I + m * (v * n^T)   -m * dot(Q, n) * v |
    ///      | 0^T                  1                 |
    ///
    ///   == | I + m * outer(v, t)   -m * dot(Q, n) * v |
    ///      | 0^T                    1                 |
    /// ```
    /// where `I` denotes the identity matrix, and `0^T` denotes the transpose of
    /// the zero vector. In the standard basis in Euclidean space, the outer product
    /// of `v` and `n` is given by
    /// ```text
    /// outer(v, n) := | v.x * n.x   v.x * n.y |
    ///                | v.y * n.x   v.y * n.y |
    /// ```
    /// so the right-hand side of the expression for the shearing matrix is
    /// ```text
    /// I + m * outer(v, n) == | 1 0 | + m * | v.x * n.x   v.x * n.y |
    ///                        | 0 1 |       | v.y * n.x   v.y * n.y |
    ///
    ///                     == | 1 + m * v.x * n.x   m * v.x * n.y     |
    ///                        | m * v.y * n.x       1 + m * v.y * n.y |
    /// ```
    /// which leads to the formula used to implement the shearing transformation
    /// ```text
    ///      | 1 + m * v.x * n.x   m * v.x * n.y      -m * dot(Q, n) * v |
    /// M == | m * v.y * n.x       1 + m* v.y * n.y   -m * dot(Q, n) * v |
    ///      | 0                   0                   1                 |
    /// ```
    ///
    /// # An Equivalent Interpretation Of The Shearing Factor
    ///
    /// The projection of the vector `p - Q` onto `v` is given by `p_v := dot(p - Q, v) * v`.
    /// Observe that
    /// ```text
    /// q - p == m * dot(p - Q, n) * v
    /// p - p_v == (p - Q)_n * n == dot(p - Q, n) * n
    /// ```
    /// With the two vectors define above as a vector with a purely `v` component,
    /// and a purely `n` component, respectively, the three points form a triangle.
    /// The tangent of the angle `phi` with respect to the normal vector at `p_v`
    /// is then given by
    /// ```text
    /// tan(phi) := (q - p)_v / (p - p_v)_n
    ///          == dot(q - p, v) / dot(p - p_v, n)
    ///          == (m * dot(p - Q, n) * dot(v, v)) / (dot(p - Q, n) * dot(n, n))
    ///          == m * (dot(p - Q, n) / dot(p - Q, n))
    ///          == m
    /// ```
    /// so the shearing factor `m` represents the tangent of the shearing angle `phi`
    /// with respect to the unit normal `n`.
    ///
    /// # Examples
    ///
    /// Shearing along the **x-axis** with a non-zero origin on the **x-axis**.
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// #     Vector3,
    /// # };
    /// #
    /// let shear_factor = 15_f64;
    /// let origin = Point2::new(-2_f64, 0_f64);
    /// let direction = Unit::from_value(Vector2::unit_x());
    /// let normal = Unit::from_value(Vector2::unit_y());
    /// let matrix = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);
    /// let vertices = [
    ///     Vector3::new( 1_f64,  1_f64, 1_f64),
    ///     Vector3::new(-1_f64,  1_f64, 1_f64),
    ///     Vector3::new(-1_f64, -1_f64, 1_f64),
    ///     Vector3::new( 1_f64, -1_f64, 1_f64),
    /// ];
    /// let expected = [
    ///     Vector3::new( 1_f64 + shear_factor,  1_f64, 1_f64),
    ///     Vector3::new(-1_f64 + shear_factor,  1_f64, 1_f64),
    ///     Vector3::new(-1_f64 - shear_factor, -1_f64, 1_f64),
    ///     Vector3::new( 1_f64 - shear_factor, -1_f64, 1_f64),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_relative_eq!(result[0], expected[0], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[1], expected[1], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[2], expected[2], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[3], expected[3], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    ///
    /// let vertices_in_line = [
    ///     Vector3::new( 1_f64, 0_f64, 1_f64),
    ///     Vector3::new(-1_f64, 0_f64, 1_f64),
    ///     Vector3::new( 0_f64, 0_f64, 1_f64),
    /// ];
    /// let expected_in_line = vertices_in_line;
    /// let result_in_line = vertices_in_line.map(|v| matrix * v);
    ///
    /// assert_relative_eq!(result_in_line[0], expected_in_line[0], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_line[1], expected_in_line[1], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_line[2], expected_in_line[2], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    ///
    /// Shearing along the line `y == (1 / 2) * x + 1` using the origin `(2, 2)`.
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// #     Vector3,
    /// # };
    /// # use core::f64;
    /// #
    /// let shear_factor = 7_f64;
    /// let origin = Point2::new(2_f64, 2_f64);
    /// let direction = Unit::from_value(Vector2::new(2_f64, 1_f64));
    /// let normal = Unit::from_value(Vector2::new(-1_f64, 2_f64));
    /// let matrix = Matrix3x3::from_affine_shear(shear_factor, &origin, &direction, &normal);
    ///
    /// // The square's top and bottom sides run parallel to the line `y == (1 / 2) * x + 1`.
    /// // The square's left and right sides run perpendicular to the line `y == (1 / 2) * x + 1`.
    /// let vertices = [
    ///     Vector3::new( 1_f64 / f64::sqrt(5_f64),  3_f64 / f64::sqrt(5_f64) + 1_f64, 1_f64),
    ///     Vector3::new(-3_f64 / f64::sqrt(5_f64),  1_f64 / f64::sqrt(5_f64) + 1_f64, 1_f64),
    ///     Vector3::new(-1_f64 / f64::sqrt(5_f64), -3_f64 / f64::sqrt(5_f64) + 1_f64, 1_f64),
    ///     Vector3::new( 3_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64) + 1_f64, 1_f64),
    /// ];
    /// let rotated_origin = Vector3::new(f64::sqrt(5_f64), 0_f64, 1_f64);
    /// let expected = [
    ///     Vector3::new(
    ///          (1_f64 / f64::sqrt(5_f64)) + (2_f64 / f64::sqrt(5_f64)) * shear_factor,
    ///          (3_f64 / f64::sqrt(5_f64)) + (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
    ///          1_f64,
    ///     ),
    ///     Vector3::new(
    ///         -(3_f64 / f64::sqrt(5_f64)) + (2_f64 / f64::sqrt(5_f64)) * shear_factor,
    ///          (1_f64 / f64::sqrt(5_f64))  + (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
    ///          1_f64,
    ///     ),
    ///     Vector3::new(
    ///         -(1_f64 / f64::sqrt(5_f64)) - (2_f64 / f64::sqrt(5_f64)) * shear_factor,
    ///         -(3_f64 / f64::sqrt(5_f64)) - (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
    ///          1_f64,
    ///     ),
    ///     Vector3::new(
    ///          (3_f64 / f64::sqrt(5_f64)) - (2_f64 / f64::sqrt(5_f64)) * shear_factor,
    ///         -(1_f64 / f64::sqrt(5_f64)) - (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
    ///          1_f64,
    ///     ),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_relative_eq!(result[0], expected[0], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[1], expected[1], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[2], expected[2], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[3], expected[3], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    ///
    /// let vertices_in_plane = [
    ///     Vector3::new( 1_f64 / f64::sqrt(5_f64),  1_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64, 1_f64),
    ///     Vector3::new(-3_f64 / f64::sqrt(5_f64), -3_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64, 1_f64),
    ///     Vector3::new(-1_f64 / f64::sqrt(5_f64), -1_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64, 1_f64),
    ///     Vector3::new( 3_f64 / f64::sqrt(5_f64),  3_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64, 1_f64),
    ///     Vector3::new( 0_f64, 1_f64, 1_f64),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_relative_eq!(result_in_plane[0], expected_in_plane[0], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_plane[1], expected_in_plane[1], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_plane[2], expected_in_plane[2], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_plane[3], expected_in_plane[3], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_plane[4], expected_in_plane[4], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear(
        shear_factor: S,
        origin: &Point2<S>,
        direction: &Unit<Vector2<S>>,
        normal: &Unit<Vector2<S>>
    ) -> Self
    {
        let zero = S::zero();
        let one = S::one();
        let translation = direction.into_inner() * (-shear_factor * origin.to_vector().dot(normal));

        let c0r0 = one + shear_factor * direction[0] * normal[0];
        let c0r1 = shear_factor * direction[1] * normal[0];
        let c0r2 = zero;

        let c1r0 = shear_factor * direction[0] * normal[1];
        let c1r1 = one + shear_factor * direction[1] * normal[1];
        let c1r2 = zero;

        let c2r0 = translation[0];
        let c2r1 = translation[1];
        let c2r2 = one;

        Self::new(
            c0r0, c0r1, c0r2,
            c1r0, c1r1, c1r2,
            c2r0, c2r1, c2r2,
        )
    }
}

impl<S> Matrix3x3<S>
where
    S: SimdScalarSigned,
{
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
    /// # Discussion
    ///
    /// The reflection of a point is defined as follows. Let `M` be the plane of
    /// reflection, also known as the **mirror plane**. Let `n` be a vector normal
    /// to the mirror plane `M`. Since `n` is normal to `M`, reflected points are
    /// reflected in a direction parallel to `n`, i.e. perpendicular to the mirror
    /// plane `M`. To reflect points correctly, we need a known point `Q` in the plane
    /// of reflection.
    ///
    /// For a vector `v`, we can choose vectors `v_per` and `v_par` such that
    /// `v == v_per + v_par`, `v_per` is perpendicular to the `n` and `v_par` is
    /// parallel to `n`. Stated different, `v_per` is parallel to the mirror plane `M`
    /// and `v_par` is perpendicular to the mirror plane `M`. The reflection `Ref` acts
    /// on `v_per` and `v_par` as follows
    /// ```text
    /// Ref(v_per) :=  v_per
    /// Ref(v_par) := -v_par
    /// ```
    /// by definition. This means that the reflection on vectors is defined by
    /// ```text
    /// Ref(v) := Ref(v_per + v_par)
    ///        := Ref(v_per) + Ref(v_par)
    ///        := Ref(v_per) - v_par
    ///        == v_per - v_par
    ///        == v - v_par - v_par
    ///        == v - 2 * v_par
    ///        == v - (2 * dot(v, n)) * n
    /// ```
    /// and reflection on points is defined by
    /// ```text
    /// Ref(P) := Ref(Q + (P - Q))
    ///        := Q + Ref(P - Q)
    ///        == Q + [(P - Q) - 2 * dot(P - Q, n) * n]
    ///        == P - 2 * dot(P - Q, n) * n
    ///        == I * P - (2 * dot(P, n)) * n + (2 * dot(Q, n)) * n
    ///        == [I - 2 * outer(n, n)] * P + (2 * dot(Q, n)) * n
    /// ```
    /// and the corresponding affine matrix has the form
    /// ```text
    /// M := | I - 2 * outer(n, n)   2 * dot(Q, n) * n |
    ///      | 0^T                   1                 |
    /// ```
    /// geometrically. In the standard basis in two-dimensional Euclidean space, we
    /// have
    /// ```text
    ///      |  1 - 2 * n.x * n.x   -2 * n.x * n.y       2 * dot(Q, n) * n.x |
    /// M == | -2 * n.y * n.x        1 - 2 * n.y * n.y   2 * dot(Q, n) * n.y |
    ///      |  0                    0                   1                   |
    /// ```
    /// and in three-dimensional Euclidean space we have
    /// ```text
    ///      |  1 - 2 * n.x * n.x   -2 * n.x * n.y       -2 * n.x * n.z        2 * dot(Q, n) * n.x |
    /// M == | -2 * n.y * n.x        1 - 2 * n.y * n.y   -2 * n.y * n.z        2 * dot(Q, n) * n.y |
    ///      | -2 * n.z * n.x       -2 * n.z * n.y        1 - 2 * n.z * n.z    2 * dot(Q, n) * n.z |
    ///      |  0                    0                    0                    1                   |
    /// ```
    /// which correspond exactly the how the respective matrices are implemented.
    ///
    /// # Example (Line Through The Origin)
    ///
    /// Here is an example of reflecting a vector across the **x-axis** with
    /// the line of reflection passing through the origin.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// #     Vector3,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector2::unit_y());
    /// let bias = Point2::new(0_f64, 0_f64);
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// #     Vector3,
    /// # };
    /// #
    /// let minus_normal = Unit::from_value(-Vector2::unit_y());
    /// let bias = Point2::new(0_f64, 0_f64);
    /// let matrix = Matrix3x3::from_affine_reflection(&minus_normal, &bias);
    /// let vector = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let expected = Vector3::new(2_f64, -2_f64, 0_f64);
    /// let result = matrix * vector;
    ///
    /// assert_eq!(result, expected);
    /// ```
    ///
    /// # Example (Line That Does Not Cross The Origin)
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Point2,
    /// #     Unit,
    /// #     Vector2,
    /// #     Vector3,
    /// # };
    /// #
    /// let bias = Point2::new(0_f64, 2_f64);
    /// let normal = Unit::from_value(
    ///     Vector2::new(-1_f64 / f64::sqrt(5_f64), 2_f64 / f64::sqrt(5_f64))
    /// );
    /// let matrix = Matrix3x3::from_affine_reflection(&normal, &bias);
    /// let vector = Vector3::new(1_f64, 0_f64, 1_f64);
    /// let expected = Vector3::new(-1_f64, 4_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_reflection(normal: &Unit<Vector2<S>>, bias: &Point2<S>) -> Self {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;

        let c0r0 = one - two * normal[0] * normal[0];
        let c0r1 = -two * normal[1] * normal[0];
        let c0r2 = zero;

        let c1r0 = -two * normal[0] * normal[1];
        let c1r1 = one - two * normal[1] * normal[1];
        let c1r2 = zero;

        let c2r0 = two * normal[0] * (normal[0] * bias[0] + normal[1] * bias[1]);
        let c2r1 = two * normal[1] * (normal[0] * bias[0] + normal[1] * bias[1]);
        let c2r2 = one;

        Self::new(
            c0r0, c0r1, c0r2,
            c1r0, c1r1, c1r2,
            c2r0, c2r1, c2r2,
        )
    }

    /// Construct a three-dimensional reflection matrix for a plane that
    /// crosses the origin.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let expected = Matrix3x3::new(
    ///     1_f64, 0_f64,  0_f64,
    ///     0_f64, 1_f64,  0_f64,
    ///     0_f64, 0_f64, -1_f64,
    /// );
    /// let result = Matrix3x3::from_reflection(&normal);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_reflection(normal: &Unit<Vector3<S>>) -> Self {
        let one = S::one();
        let two = one + one;

        let c0r0 =  one - two * normal[0] * normal[0];
        let c0r1 = -two * normal[1] * normal[0];
        let c0r2 = -two * normal[2] * normal[0];

        let c1r0 = -two * normal[0] * normal[1];
        let c1r1 =  one - two * normal[1] * normal[1];
        let c1r2 = -two * normal[2] * normal[1];

        let c2r0 = -two * normal[0] * normal[2];
        let c2r1 = -two * normal[1] * normal[2];
        let c2r2 =  one - two * normal[2] * normal[2];

        Self::new(
            c0r0, c0r1, c0r2,
            c1r0, c1r1, c1r2,
            c2r0, c2r1, c2r2,
       )
    }

    /// Compute the determinant of a matrix.
    ///
    /// The determinant of a matrix is the signed volume of the parallelepiped
    /// swept out by the vectors represented by the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_f64, 4_f64, 7_f64,
    ///     2_f64, 5_f64, 8_f64,
    ///     3_f64, 6_f64, 9_f64,
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

    /// Compute the cross product matrix for a given vector.
    ///
    /// The cross matrix for a vector `a` is the matrix `A` such that for any
    /// vector `v`, `A` satisfies
    /// ```text
    /// A * v == cross(a, v)
    /// ```
    /// In Euclidean space in the standard basis, the cross matrix has the
    /// following form. Given a vector `a`, the cross matrix for `a` is
    /// ```text
    ///                  | 0    -a.z   a.y |
    /// A := cross(a) := | a.z   0    -a.x |
    ///                  | -a.y  a.x   0   |
    /// ```
    /// where `a.*` denote the components of the vector `a` in the standard Euclidean
    /// basis.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// #
    /// let a = Vector3::new(2_i32, 3_i32, 4_i32);
    /// let v = Vector3::new(43_i32, 5_i32, 89_i32);
    /// let cross_a = Matrix3x3::cross_matrix(&a);
    ///
    /// assert_eq!(cross_a * v, a.cross(&v));
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn cross_matrix(vector: &Vector3<S>) -> Self {
        Matrix3x3::new(
             S::zero(),  vector[2], -vector[1],
            -vector[2],  S::zero(),  vector[0],
             vector[1], -vector[0],  S::zero(),
        )
    }
}

impl<S> Matrix3x3<S>
where
    S: SimdScalarFloat,
{
    /// Construct an affine rotation matrix in two dimensions that rotates a
    /// vector in the **xy-plane** by an angle `angle`.
    ///
    /// This is the affine matrix counterpart to the 2x2 matrix function
    /// `from_angle`.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_affine_angle(angle);
    /// let unit_x = Vector3::unit_x();
    /// let expected = Vector3::unit_y();
    /// let result = matrix * unit_x;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_angle<A>(angle: A) -> Self
    where
        A: Into<Radians<S>>,
    {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());
        let zero = S::zero();
        let one =  S::one();

        Self::new(
             cos_angle, sin_angle, zero,
            -sin_angle, cos_angle, zero,
             zero,      zero,      one,
        )
    }

    /// Construct a rotation matrix about the **x-axis** by an angle `angle`.
    ///
    /// # Example
    ///
    /// In this example the rotation is in the **yz-plane**.
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_angle_x(angle);
    /// let vector = Vector3::new(0_f64, 1_f64, 1_f64);
    /// let expected = Vector3::new(0_f64, -1_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_angle_x<A>(angle: A) -> Self
    where
        A: Into<Radians<S>>,
    {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Self::new(
            S::one(),   S::zero(), S::zero(),
            S::zero(),  cos_angle, sin_angle,
            S::zero(), -sin_angle, cos_angle,
        )
    }

    /// Construct a rotation matrix about the **y-axis** by an angle `angle`.
    ///
    /// # Example
    ///
    /// In this example the rotation is in the **zx-plane**.
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_angle_y(angle);
    /// let vector = Vector3::new(1_f64, 0_f64, 1_f64);
    /// let expected = Vector3::new(1_f64, 0_f64, -1_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_angle_y<A>(angle: A) -> Self
    where
        A: Into<Radians<S>>,
    {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Self::new(
            cos_angle, S::zero(), -sin_angle,
            S::zero(), S::one(),   S::zero(),
            sin_angle, S::zero(),  cos_angle,
        )
    }

    /// Construct a rotation matrix about the **z-axis** by an angle `angle`.
    ///
    /// # Example
    ///
    /// In this example the rotation is in the **xy-plane**.
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_angle_z(angle);
    /// let vector = Vector3::new(1_f64, 1_f64, 0_f64);
    /// let expected = Vector3::new(-1_f64, 1_f64, 0_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_angle_z<A>(angle: A) -> Self
    where
        A: Into<Radians<S>>,
    {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Self::new(
             cos_angle, sin_angle, S::zero(),
            -sin_angle, cos_angle, S::zero(),
             S::zero(), S::zero(), S::one(),
        )
    }

    /// Construct a rotation matrix about an arbitrary axis by an angle
    /// `angle`.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_axis_angle(&axis, angle);
    /// let unit_x = Vector3::unit_x();
    /// let expected = Vector3::unit_y();
    /// let result = matrix * unit_x;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_axis_angle<A>(axis: &Unit<Vector3<S>>, angle: A) -> Self
    where
        A: Into<Radians<S>>,
    {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());
        let one_minus_cos_angle = S::one() - cos_angle;
        let _axis = axis.as_ref();

        Self::new(
            one_minus_cos_angle * _axis[0] * _axis[0] + cos_angle,
            one_minus_cos_angle * _axis[0] * _axis[1] + sin_angle * _axis[2],
            one_minus_cos_angle * _axis[0] * _axis[2] - sin_angle * _axis[1],

            one_minus_cos_angle * _axis[0] * _axis[1] - sin_angle * _axis[2],
            one_minus_cos_angle * _axis[1] * _axis[1] + cos_angle,
            one_minus_cos_angle * _axis[1] * _axis[2] + sin_angle * _axis[0],

            one_minus_cos_angle * _axis[0] * _axis[2] + sin_angle * _axis[1],
            one_minus_cos_angle * _axis[1] * _axis[2] - sin_angle * _axis[0],
            one_minus_cos_angle * _axis[2] * _axis[2] + cos_angle,
        )
    }

    /// Construct a coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing
    /// the direction `direction` into the coordinate system of an observer
    /// located at the origin facing the **positive z-axis**.
    ///
    /// The function maps the direction `direction` to the **positive z-axis**
    /// in the new coordinate system. This corresponds to a rotation matrix.
    /// This transformation is a **left-handed** coordinate transformation.
    ///
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Normed,
    /// #     Vector3,
    /// # };
    /// # use core::f64;
    /// #
    /// let direction = Vector3::new(1_f64, 1_f64, 0_f64);
    /// let up = Vector3::unit_z();
    /// let expected = Matrix3x3::new(
    ///     -1_f64 / f64::sqrt(2_f64), 0_f64,  1_f64 / f64::sqrt(2_f64),
    ///      1_f64 / f64::sqrt(2_f64), 0_f64,  1_f64 / f64::sqrt(2_f64),
    ///      0_f64,                    1_f64,  0_f64,
    /// );
    /// let result = Matrix3x3::look_to_lh(&direction, &up);
    /// let unit_z = Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * direction.normalize(), unit_z, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn look_to_lh(direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let z_axis = direction.normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        Self::new(
            x_axis[0], y_axis[0], z_axis[0],
            x_axis[1], y_axis[1], z_axis[1],
            x_axis[2], y_axis[2], z_axis[2],
        )
    }

    /// Construct a coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing
    /// the direction `direction` into the coordinate system of an observer
    /// located at the origin facing the **negative z-axis**.
    ///
    /// The function maps the direction `direction` to the **negative z-axis**
    /// in the new coordinate system. This corresponds to a rotation matrix.
    /// This transformation is a **right-handed** coordinate transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Normed,
    /// #     Vector3,
    /// # };
    /// # use core::f64;
    /// #
    /// let direction = Vector3::new(1_f64, 1_f64, 0_f64);
    /// let up = Vector3::unit_z();
    /// let expected = Matrix3x3::new(
    ///      1_f64 / f64::sqrt(2_f64), 0_f64, -1_f64 / f64::sqrt(2_f64),
    ///     -1_f64 / f64::sqrt(2_f64), 0_f64, -1_f64 / f64::sqrt(2_f64),
    ///      0_f64,                    1_f64,  0_f64,
    /// );
    /// let result = Matrix3x3::look_to_rh(&direction, &up);
    /// let minus_unit_z = -Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * direction.normalize(), minus_unit_z, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn look_to_rh(direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let z_axis = -direction.normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        Self::new(
            x_axis[0], y_axis[0], z_axis[0],
            x_axis[1], y_axis[1], z_axis[1],
            x_axis[2], y_axis[2], z_axis[2],
        )
    }

    /// Construct a coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the position `eye` facing
    /// the position `target` into the coordinate system of an observer
    /// located at the origin facing the **positive z-axis**.
    ///
    /// The function maps the direction `target - eye` to the **positive z-axis**
    /// in the new coordinate system. This corresponds to a rotation matrix.
    /// This transformation is a **left-handed** coordinate transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(-1_f64, -1_f64, 0_f64);
    /// let target = Point3::origin();
    /// let up = Vector3::unit_z();
    /// let expected = Matrix3x3::new(
    ///     -1_f64 / f64::sqrt(2_f64), 0_f64,  1_f64 / f64::sqrt(2_f64),
    ///      1_f64 / f64::sqrt(2_f64), 0_f64,  1_f64 / f64::sqrt(2_f64),
    ///      0_f64,                    1_f64,  0_f64,
    /// );
    /// let result = Matrix3x3::look_at_lh(&eye, &target, &up);
    /// let direction = (target - eye).normalize();
    /// let unit_z = Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * direction, unit_z, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn look_at_lh(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        Self::look_to_lh(&(target - eye), up)
    }

    /// Construct a coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing
    /// the direction `direction` into the coordinate system of an observer
    /// located at the origin facing the **negative z-axis**.
    ///
    /// The function maps the direction `target - eye` to the **negative z-axis**
    /// in the new coordinate system. This corresponds to a rotation matrix.
    /// This transformation is a **right-handed** coordinate transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(-1_f64, -1_f64, 0_f64);
    /// let target = Point3::origin();
    /// let up = Vector3::unit_z();
    /// let expected = Matrix3x3::new(
    ///      1_f64 / f64::sqrt(2_f64), 0_f64, -1_f64 / f64::sqrt(2_f64),
    ///     -1_f64 / f64::sqrt(2_f64), 0_f64, -1_f64 / f64::sqrt(2_f64),
    ///      0_f64,                    1_f64,  0_f64,
    /// );
    /// let result = Matrix3x3::look_at_rh(&eye, &target, &up);
    /// let direction = (target - eye).normalize();
    /// let minus_unit_z = -Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * direction, minus_unit_z, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn look_at_rh(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        Self::look_to_rh(&(target - eye), up)
    }

    /// Construct a rotation matrix that transforms the coordinate system of
    /// an observer located at the origin facing the **positive z-axis** into a
    /// coordinate system of an observer located at the origin facing the
    /// direction `direction`. The resulting coordinate transformation is a
    /// **left-handed** coordinate transformation.
    ///
    /// The function maps the **positive z-axis** to the direction `direction`.
    /// This function is the inverse of [`look_to_lh`].
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Normed,
    /// #     Vector3,
    /// # };
    /// # use core::f64;
    /// #
    /// let direction = Vector3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let expected = Matrix3x3::new(
    ///      1_f64 / f64::sqrt(6_f64), -1_f64 / f64::sqrt(6_f64), -2_f64 / f64::sqrt(6_f64),
    ///      1_f64 / f64::sqrt(2_f64),  1_f64 / f64::sqrt(2_f64),  0_f64,
    ///      1_f64 / f64::sqrt(3_f64), -1_f64 / f64::sqrt(3_f64),  1_f64 / f64::sqrt(3_f64),
    /// );
    /// let result = Matrix3x3::look_to_lh_inv(&direction, &up);
    /// let unit_z = Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * unit_z, direction.normalize(), abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn look_to_lh_inv(direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        Self::look_to_lh(direction, up).transpose()
    }

    /// Construct a rotation matrix that transforms the coordinate system of
    /// an observer located at the origin facing the **negative z-axis** into a
    /// coordinate system of an observer located at the origin facing the
    /// direction `direction`. The resulting coordinate transformation is a
    /// **right-handed** coordinate transformation.
    ///
    /// The function maps the **negative z-axis** to the direction `direction`.
    /// This function is the inverse of [`look_to_rh`].
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Normed,
    /// #     Vector3,
    /// # };
    /// # use core::f64;
    /// #
    /// let direction = Vector3::new(1_f64, -1_f64, 1_f64);
    /// let up = Vector3::new(2_f64, 2_f64, 0_f64);
    /// let expected = Matrix3x3::new(
    ///     -1_f64 / f64::sqrt(6_f64),  1_f64 / f64::sqrt(6_f64),  2_f64 / f64::sqrt(6_f64),
    ///      1_f64 / f64::sqrt(2_f64),  1_f64 / f64::sqrt(2_f64),  0_f64,
    ///     -1_f64 / f64::sqrt(3_f64),  1_f64 / f64::sqrt(3_f64), -1_f64 / f64::sqrt(3_f64),
    /// );
    /// let result = Matrix3x3::look_to_rh_inv(&direction, &up);
    /// let minus_unit_z = -Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * minus_unit_z, direction.normalize(), abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn look_to_rh_inv(direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        Self::look_to_rh(direction, up).transpose()
    }

    /// Construct a coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing
    /// the direction `target - eye` into the coordinate system of an observer
    /// located at the origin facing the **positive z-axis**.
    ///
    /// The function maps the direction `target - eyey` to the **positive z-axis**
    /// in the new coordinate system. This corresponds to a rotation matrix.
    /// This transformation is a **left-handed** coordinate transformation.
    /// This function is the inverse of [`look_at_lh`].
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(-1_f64, -1_f64, 0_f64);
    /// let target = Point3::origin();
    /// let up = Vector3::unit_z();
    /// let expected = Matrix3x3::new(
    ///     -1_f64 / f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64), 0_f64,
    ///      0_f64,                    0_f64,                    1_f64,
    ///      1_f64 / f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64), 0_f64,
    /// );
    /// let result = Matrix3x3::look_at_lh_inv(&eye, &target, &up);
    /// let direction = (target - eye).normalize();
    /// let unit_z = Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * unit_z, direction, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn look_at_lh_inv(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        Self::look_at_lh(eye, target, up).transpose()
    }

    /// Construct a coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing
    /// the direction `target - eye` into the coordinate system of an observer
    /// located at the origin facing the **negative z-axis**.
    ///
    /// The function maps the direction `target - eye` to the **negative z-axis**
    /// in the new coordinate system. This corresponds to a rotation matrix.
    /// This transformation is a **right-handed** coordinate transformation.
    /// This function is the inverse of [`look_at_rh`].
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(-1_f64, -1_f64, 0_f64);
    /// let target = Point3::origin();
    /// let up = Vector3::unit_z();
    /// let expected = Matrix3x3::new(
    ///      1_f64 / f64::sqrt(2_f64), -1_f64 / f64::sqrt(2_f64), 0_f64,
    ///      0_f64,                     0_f64,                    1_f64,
    ///     -1_f64 / f64::sqrt(2_f64), -1_f64 / f64::sqrt(2_f64), 0_f64,
    /// );
    /// let result = Matrix3x3::look_at_rh_inv(&eye, &target, &up);
    /// let direction = (target - eye).normalize();
    /// let minus_unit_z = -Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * minus_unit_z, direction, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn look_at_rh_inv(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        Self::look_at_rh(eye, target, up).transpose()
    }

    /// Construct a rotation matrix that rotates the shortest angular distance
    /// between two vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// #
    /// let v1: Vector3<f64> = Vector3::unit_x() * 2_f64;
    /// let v2: Vector3<f64> = Vector3::unit_y() * 3_f64;
    /// let matrix = Matrix3x3::rotation_between(&v1, &v2).unwrap();
    /// let expected = Vector3::new(0_f64, 2_f64, 0_f64);
    /// let result = matrix * v1;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn rotation_between(v1: &Vector3<S>, v2: &Vector3<S>) -> Option<Self> {
        Self::rotation_between_eps(v1, v2, S::machine_epsilon())
    }

    #[inline]
    fn rotation_between_eps(v1: &Vector3<S>, v2: &Vector3<S>, threshold: S) -> Option<Self> {
        if let (Some(unit_v1), Some(unit_v2)) = (v1.try_normalize(S::zero()), v2.try_normalize(S::zero())) {
            let cross = unit_v1.cross(&unit_v2);

            if let Some(axis) = Unit::try_from_value(cross, threshold) {
                return Some(Self::from_axis_angle(&axis, Radians::acos(unit_v1.dot(&unit_v2))));
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
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// #
    /// let unit_v1: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x() * 2_f64);
    /// let unit_v2: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y() * 3_f64);
    /// let matrix = Matrix3x3::rotation_between_axis(&unit_v1, &unit_v2).unwrap();
    /// let vector = Vector3::unit_x() * 2_f64;
    /// let expected = Vector3::unit_y() * 2_f64;
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn rotation_between_axis(unit_v1: &Unit<Vector3<S>>, unit_v2: &Unit<Vector3<S>>) -> Option<Self> {
        Self::rotation_between_axis_eps(unit_v1, unit_v2, S::machine_epsilon())
    }

    #[inline]
    fn rotation_between_axis_eps(unit_v1: &Unit<Vector3<S>>, unit_v2: &Unit<Vector3<S>>, threshold: S) -> Option<Self> {
        let cross = unit_v1.as_ref().cross(unit_v2.as_ref());
        let cos_angle = unit_v1.as_ref().dot(unit_v2.as_ref());

        if let Some(axis) = Unit::try_from_value(cross, threshold) {
            return Some(Self::from_axis_angle(&axis, Radians::acos(cos_angle)));
        }

        if cos_angle < S::zero() {
            return None;
        }

        Some(Self::identity())
    }

    /// Compute the inverse of a square matrix, if the inverse exists.
    ///
    /// Given a square matrix `self` Compute the matrix `m` if it exists
    /// such that
    /// ```text
    /// m * self == self * m == 1.
    /// ```
    /// Not every square matrix has an inverse.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_f64, 4_f64, 7_f64,
    ///     2_f64, 5_f64, 8_f64,
    ///     5_f64, 6_f64, 11_f64,
    /// );
    /// let expected = Matrix3x3::new(
    ///     -7_f64 / 12_f64,   2_f64 / 12_f64,   3_f64 / 12_f64,
    ///     -18_f64 / 12_f64,  24_f64 / 12_f64, -6_f64 / 12_f64,
    ///      13_f64 / 12_f64, -14_f64 / 12_f64,  3_f64 / 12_f64,
    /// );
    /// let result = matrix.inverse().unwrap();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.is_zero() {
            None
        } else {
            let det_inv = S::one() / det;

            Some(Self::new(
                det_inv * (self.data[1][1] * self.data[2][2] - self.data[1][2] * self.data[2][1]),
                det_inv * (self.data[0][2] * self.data[2][1] - self.data[0][1] * self.data[2][2]),
                det_inv * (self.data[0][1] * self.data[1][2] - self.data[0][2] * self.data[1][1]),

                det_inv * (self.data[1][2] * self.data[2][0] - self.data[1][0] * self.data[2][2]),
                det_inv * (self.data[0][0] * self.data[2][2] - self.data[0][2] * self.data[2][0]),
                det_inv * (self.data[0][2] * self.data[1][0] - self.data[0][0] * self.data[1][2]),

                det_inv * (self.data[1][0] * self.data[2][1] - self.data[1][1] * self.data[2][0]),
                det_inv * (self.data[0][1] * self.data[2][0] - self.data[0][0] * self.data[2][1]),
                det_inv * (self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]),
            ))
        }
    }

    /// Determine whether a square matrix has an inverse matrix.
    ///
    /// A matrix is invertible if its determinant is not zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x3;
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_f64, 2_f64, 3_f64,
    ///     4_f64, 5_f64, 6_f64,
    ///     7_f64, 8_f64, 9_f64,
    /// );
    ///
    /// assert_eq!(matrix.determinant(), 0_f64);
    /// assert!(!matrix.is_invertible());
    /// ```
    #[inline]
    pub fn is_invertible(&self) -> bool {
        ulps_ne!(self.determinant(), S::zero(), abs_diff_all <= S::machine_epsilon(), ulps_all <= S::default_ulps())
    }
}

impl<S, const M: usize, const N: usize> From<Matrix<S, M, M>> for Matrix<S, N, N>
where
    S: SimdScalar,
    ShapeConstraint: DimLt<Const<M>, Const<N>>,
{
    #[inline]
    fn from(matrix: Matrix<S, M, M>) -> Matrix<S, N, N> {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        // SAFETY: M < N so the conversion cannot fail.
        let mut result = Matrix::identity();
        for c in 0..M {
            for r in 0..M {
                result[c][r] = matrix[c][r];
            }
        }

        result
    }
}

impl<S, const M: usize, const N: usize> From<&Matrix<S, M, M>> for Matrix<S, N, N>
where
    S: SimdScalar,
    ShapeConstraint: DimLt<Const<M>, Const<N>>,
{
    #[inline]
    fn from(matrix: &Matrix<S, M, M>) -> Matrix<S, N, N> {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        // SAFETY: M < N so the conversion cannot fail.
        let mut result = Matrix::identity();
        for c in 0..M {
            for r in 0..M {
                result[c][r] = matrix[c][r];
            }
        }

        result
    }
}

impl<S> Matrix4x4<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let c0r0 = 1_i32;  let c0r1 = 2_i32;  let c0r2 = 3_i32;  let c0r3 = 4_i32;
    /// let c1r0 = 5_i32;  let c1r1 = 6_i32;  let c1r2 = 7_i32;  let c1r3 = 8_i32;
    /// let c2r0 = 9_i32;  let c2r1 = 10_i32; let c2r2 = 11_i32; let c2r3 = 12_i32;
    /// let c3r0 = 13_i32; let c3r1 = 14_i32; let c3r2 = 15_i32; let c3r3 = 16_i32;
    /// let matrix = Matrix4x4::new(
    ///     c0r0, c0r1, c0r2, c0r3,
    ///     c1r0, c1r1, c1r2, c1r3,
    ///     c2r0, c2r1, c2r2, c2r3,
    ///     c3r0, c3r1, c3r2, c3r3,
    /// );
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// assert_eq!(matrix[0][1], c0r1);
    /// assert_eq!(matrix[0][2], c0r2);
    /// assert_eq!(matrix[0][3], c0r3);
    /// assert_eq!(matrix[1][0], c1r0);
    /// assert_eq!(matrix[1][1], c1r1);
    /// assert_eq!(matrix[1][2], c1r2);
    /// assert_eq!(matrix[1][3], c1r3);
    /// assert_eq!(matrix[2][0], c2r0);
    /// assert_eq!(matrix[2][1], c2r1);
    /// assert_eq!(matrix[2][2], c2r2);
    /// assert_eq!(matrix[2][3], c2r3);
    /// assert_eq!(matrix[3][0], c3r0);
    /// assert_eq!(matrix[3][1], c3r1);
    /// assert_eq!(matrix[3][2], c3r2);
    /// assert_eq!(matrix[3][3], c3r3);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(
        c0r0: S, c0r1: S, c0r2: S, c0r3: S,
        c1r0: S, c1r1: S, c1r2: S, c1r3: S,
        c2r0: S, c2r1: S, c2r2: S, c2r3: S,
        c3r0: S, c3r1: S, c3r2: S, c3r3: S) -> Self {

        Self {
            data: [
                [c0r0, c0r1, c0r2, c0r3],
                [c1r0, c1r1, c1r2, c1r3],
                [c2r0, c2r1, c2r2, c2r3],
                [c3r0, c3r1, c3r2, c3r3],
            ],
        }
    }
}

impl<S> Matrix4x4<S>
where
    S: SimdScalar,
{
    /// Construct an affine shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **x-axis**
    /// as the shearing direction, and the **y-axis** as the normal vector.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the affine
    /// shearing transformation in general, see [`Matrix4x4::from_affine_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// #
    /// let shear_factor = 19_i32;
    /// let matrix = Matrix4x4::from_affine_shear_xy(shear_factor);
    /// let vertices = [
    ///     Vector4::new( 1_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32, -1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32, -1_i32, 1_i32),
    /// ];
    /// let expected = [
    ///     Vector4::new( 1_i32 + shear_factor,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32 + shear_factor,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32 - shear_factor, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32 - shear_factor, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32 + shear_factor,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32 + shear_factor,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32 - shear_factor, -1_i32, -1_i32, 1_i32),
    ///     Vector4::new( 1_i32 - shear_factor, -1_i32, -1_i32, 1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector4::new( 1_i32, 0_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32, 0_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32, 0_i32, -1_i32, 1_i32),
    ///     Vector4::new( 1_i32, 0_i32, -1_i32, 1_i32),
    ///     Vector4::new( 0_i32, 0_i32,  0_i32, 1_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_xy(shear_factor: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,          zero, zero, zero,
            shear_factor, one,  zero, zero,
            zero,         zero, one,  zero,
            zero,         zero, zero, one,
        )
    }

    /// Construct an affine shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **x-axis**
    /// as the shearing direction, and the **z-axis** as the normal vector.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the affine
    /// shearing transformation in general, see [`Matrix4x4::from_affine_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// #
    /// let shear_factor = 19_i32;
    /// let matrix = Matrix4x4::from_affine_shear_xz(shear_factor);
    /// let vertices = [
    ///     Vector4::new( 1_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32, -1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32, -1_i32, 1_i32),
    /// ];
    /// let expected = [
    ///     Vector4::new( 1_i32 + shear_factor,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32 + shear_factor,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32 + shear_factor, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32 + shear_factor, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32 - shear_factor,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32 - shear_factor,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32 - shear_factor, -1_i32, -1_i32, 1_i32),
    ///     Vector4::new( 1_i32 - shear_factor, -1_i32, -1_i32, 1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector4::new( 1_i32,  1_i32, 0_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32, 0_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32, 0_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32, 0_i32, 1_i32),
    ///     Vector4::new( 0_i32,  0_i32, 0_i32, 1_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_xz(shear_factor: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,          zero, zero, zero,
            zero,         one,  zero, zero,
            shear_factor, zero, one,  zero,
            zero,         zero, zero, one,
        )
    }

    /// Construct an affine shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **y-axis**
    /// as the shearing direction, and the **x-axis** as the normal vector.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the affine
    /// shearing transformation in general, see [`Matrix4x4::from_affine_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// #
    /// let shear_factor = 19_i32;
    /// let matrix = Matrix4x4::from_affine_shear_yx(shear_factor);
    /// let vertices = [
    ///     Vector4::new( 1_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32, -1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32, -1_i32, 1_i32),
    /// ];
    /// let expected = [
    ///     Vector4::new( 1_i32,  1_i32 + shear_factor,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32 - shear_factor,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32 - shear_factor,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32 + shear_factor,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32,  1_i32 + shear_factor, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32 - shear_factor, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32 - shear_factor, -1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32 + shear_factor, -1_i32, 1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector4::new(0_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(0_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new(0_i32, -1_i32, -1_i32, 1_i32),
    ///     Vector4::new(0_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(0_i32,  0_i32,  0_i32, 1_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_yx(shear_factor: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,  shear_factor, zero, zero,
            zero, one,          zero, zero,
            zero, zero,         one,  zero,
            zero, zero,         zero, one,
        )
    }

    /// Construct an affine shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **y-axis**
    /// as the shearing direction, and the **z-axis** as the normal vector.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the affine
    /// shearing transformation in general, see [`Matrix4x4::from_affine_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// #
    /// let shear_factor = 19_i32;
    /// let matrix = Matrix4x4::from_affine_shear_yz(shear_factor);
    /// let vertices = [
    ///     Vector4::new( 1_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32, -1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32, -1_i32, 1_i32),
    /// ];
    /// let expected = [
    ///     Vector4::new( 1_i32,  1_i32 + shear_factor,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32 + shear_factor,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32 + shear_factor,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32 + shear_factor,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32,  1_i32 - shear_factor, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32 - shear_factor, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32 - shear_factor, -1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32 - shear_factor, -1_i32, 1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector4::new( 1_i32,  1_i32, 0_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32, 0_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32, 0_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32, 0_i32, 1_i32),
    ///     Vector4::new( 0_i32,  0_i32, 0_i32, 1_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_yz(shear_factor: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,  zero,         zero, zero,
            zero, one,          zero, zero,
            zero, shear_factor, one,  zero,
            zero, zero,         zero, one,
        )
    }

    /// Construct an affine shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **z-axis**
    /// as the shearing direction, and the **x-axis** as the normal vector.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the affine
    /// shearing transformation in general, see [`Matrix4x4::from_affine_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// #
    /// let shear_factor = 19_i32;
    /// let matrix = Matrix4x4::from_affine_shear_zx(shear_factor);
    /// let vertices = [
    ///     Vector4::new( 1_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32, -1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32, -1_i32, 1_i32),
    /// ];
    /// let expected = [
    ///     Vector4::new( 1_i32,  1_i32,  1_i32 + shear_factor, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32,  1_i32 - shear_factor, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32,  1_i32 - shear_factor, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32,  1_i32 + shear_factor, 1_i32),
    ///     Vector4::new( 1_i32,  1_i32, -1_i32 + shear_factor, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32, -1_i32 - shear_factor, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32, -1_i32 - shear_factor, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32, -1_i32 + shear_factor, 1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector4::new(0_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(0_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new(0_i32, -1_i32, -1_i32, 1_i32),
    ///     Vector4::new(0_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(0_i32,  0_i32,  0_i32, 1_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_zx(shear_factor: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,  zero, shear_factor, zero,
            zero, one,  zero,         zero,
            zero, zero, one,          zero,
            zero, zero, zero,         one,
        )
    }

    /// Construct an affine shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `[0, 0, 0]`, using the **z-axis**
    /// as the shearing direction, and the **x-axis** as the normal vector.
    ///
    /// For a more in depth exposition on the geometrical underpinnings of the affine
    /// shearing transformation in general, see [`Matrix4x4::from_affine_shear`].
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// #
    /// let shear_factor = 19_i32;
    /// let matrix = Matrix4x4::from_affine_shear_zy(shear_factor);
    /// let vertices = [
    ///     Vector4::new( 1_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32,  1_i32, 1_i32),
    ///     Vector4::new( 1_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32, -1_i32, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32, -1_i32, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32, -1_i32, 1_i32),
    /// ];
    /// let expected = [
    ///     Vector4::new( 1_i32,  1_i32,  1_i32 + shear_factor, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32,  1_i32 + shear_factor, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32,  1_i32 - shear_factor, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32,  1_i32 - shear_factor, 1_i32),
    ///     Vector4::new( 1_i32,  1_i32, -1_i32 + shear_factor, 1_i32),
    ///     Vector4::new(-1_i32,  1_i32, -1_i32 + shear_factor, 1_i32),
    ///     Vector4::new(-1_i32, -1_i32, -1_i32 - shear_factor, 1_i32),
    ///     Vector4::new( 1_i32, -1_i32, -1_i32 - shear_factor, 1_i32),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vertices_in_plane = [
    ///     Vector4::new( 1_i32, 0_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32, 0_i32,  1_i32, 1_i32),
    ///     Vector4::new(-1_i32, 0_i32, -1_i32, 1_i32),
    ///     Vector4::new( 1_i32, 0_i32, -1_i32, 1_i32),
    ///     Vector4::new( 0_i32, 0_i32,  0_i32, 1_i32),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_eq!(result_in_plane, expected_in_plane);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear_zy(shear_factor: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,  zero, zero,         zero,
            zero, one,  shear_factor, zero,
            zero, zero, one,          zero,
            zero, zero, zero,         one,
        )
    }
}

impl<S> Matrix4x4<S>
where
    S: SimdScalarFloat,
{
    /// Construct a general affine shearing matrix in three dimensions with respect to
    /// a plane passing through the origin `origin`, not necessarily `[0, 0, 0]`.
    ///
    /// # Parameters
    ///
    /// The affine shearing matrix constructor has four parameters
    /// * `origin`: The origin of the affine frame for the shearing transformation.
    /// * `shear_factor`: The amount by which a point in a plane parallel to the shearing
    ///    plane gets sheared.
    /// * `direction`: The direction along which the shearing happens in the shearing plane.
    /// * `normal`: The normal vector to the shearing plane.
    ///
    /// # Discussion
    ///
    /// The displacement of a point with respect to the shearing plane is a function
    /// of the signed distance of the point from the shearing plane. In particular, it
    /// is a function of the value of the component of the difference between the point
    /// and the origin of the affine frame projected along the normal vector of the
    /// shearing plane.
    ///
    /// More precisely, let `Q` be the origin of the affine frame for the shearing
    /// transformation, let `v` be the shearing direction, let `n` be a vector normal
    /// to `v`, and let `p` be a point. In three dimensions, the unit vectors `v`, `n`, and
    /// `v x n` form a coordinate frame in conjunction with the origin `Q`. Let `m` be the
    /// shearing factor. Let `q` be the point that results from applying the shearing
    /// transformation to `p`. The point `q` is defined precisely as
    /// ```text
    /// q := p + (m * (p - Q)_n) * v
    /// ```
    /// where `(p - Q)_n` is the component of `p - Q` projected onto the normal vector `n`.
    /// In particular, `(p - Q)_n := dot(p - Q, n)`, so `q` is given by
    /// ```text
    /// q == p + (m * dot(p - Q, n)) * v
    ///   == I * p + (m * dot(p - Q, n)) * v == I * p + m * (dot(p - Q, n) * v)
    ///   == I * p + m * dot(p, n) * v - m * dot(Q, n) * v
    ///   == I * p + m * (v * n^T) * p - m * dot(Q, n) * v
    ///   == (I + m * (v * n^T)) * p - m * dot(Q, n) * v
    /// ```
    /// where `v * n^T` denotes the outer product of `v` and `n`. The shearing matrix
    /// in geometric form is given by
    /// ```text
    /// M := | I + m * (v * n^T)   -m * dot(Q, n) * v |
    ///      | 0^T                  1                 |
    ///
    ///   == | I + m * outer(v, t)   -m * dot(Q, n) * v |
    ///      | 0^T                    1                 |
    /// ```
    /// where `I` denotes the identity matrix, and `0^T` denotes the transpose of
    /// the zero vector. In the standard basis in Euclidean space, the outer product
    /// of `v` and `n` is given by
    /// ```text
    ///                | v.x * n.x   v.x * n.y   v.x * n.z |
    /// outer(v, n) := | v.y * n.x   v.y * n.y   v.y * n.z |
    ///                | v.z * n.x   v.z * n.y   v.z * n.z |
    /// ```
    /// so the right-hand side of the expression for the shearing matrix is
    /// ```text
    ///                        | 1 0 0 |       | v.x * n.x   v.x * n.y   v.x * n.z |
    /// I + m * outer(v, n) == | 0 1 0 | + m * | v.y * n.x   v.y * n.y   v.y * n.z |
    ///                        | 0 0 1 |       | v.z * n.x   v.z * n.y   v.z * n.z |
    ///
    ///                        | 1 + m * v.x * n.x   m * v.x * n.y       m * v.x * n.z     |
    ///                     == | m * v.y * n.x       1 + m * v.y * n.y   m * v.y * n.z     |
    ///                        | m * v.z * n.x       m * v.z * n.y       1 + m * v.z * n.z |
    /// ```
    /// which leads to the formula used to implement the shearing transformation
    /// ```text
    ///      | 1 + m * v.x * n.x   m * v.x * n.y       m * v.x * n.z       -m * dot(Q, n) * v |
    /// M == | m * v.y * n.x       1 + m * v.y * n.y   m * v.y * n.z       -m * dot(Q, n) * v |
    ///      | m * v.z * n.x       m * v.z * n.y       1 + m * v.z * n.z   -m * dot(Q, n) * v |
    ///      | 0                   0                   0                    1                 |
    /// ```
    ///
    /// # An Equivalent Interpretation Of The Shearing Factor
    ///
    /// The projection of the vector `p - Q` onto `v` is given by `p_v := dot(p - Q, v) * v`.
    /// Observe that
    /// ```text
    /// q - p == m * dot(p - Q, n) * v
    /// p - p_v == (p - Q)_n * n == dot(p - Q, n) * n
    /// ```
    /// With the two vectors define above as a vector with a purely `v` component,
    /// and a purely `n` component, respectively, the three points form a triangle.
    /// The tangent of the angle `phi` with respect to the normal vector at `p_v`
    /// is then given by
    /// ```text
    /// tan(phi) := (q - p)_v / (p - p_v)_n
    ///          == dot(q - p, v) / dot(p - p_v, n)
    ///          == (m * dot(p - Q, n) * dot(v, v)) / (dot(p - Q, n) * dot(n, n))
    ///          == m * (dot(p - Q, n) / dot(p - Q, n))
    ///          == m
    /// ```
    /// so the shearing factor `m` represents the tangent of the shearing angle `phi`
    /// with respect to the unit normal `n`.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// # use core::f64;
    /// #
    /// let shear_factor = 15_f64;
    /// let origin = Point3::origin();
    /// let direction = Unit::from_value(Vector3::new(
    ///     1_f64 / f64::sqrt(2_f64),
    ///     1_f64 / f64::sqrt(2_f64),
    ///     0_f64,
    /// ));
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let matrix = Matrix4x4::from_affine_shear(shear_factor, &origin, &direction, &normal);
    /// let vertices = [
    ///     Vector4::new( 1_f64,  1_f64,  1_f64, 1_f64),
    ///     Vector4::new(-1_f64,  1_f64,  1_f64, 1_f64),
    ///     Vector4::new(-1_f64, -1_f64,  1_f64, 1_f64),
    ///     Vector4::new( 1_f64, -1_f64,  1_f64, 1_f64),
    ///     Vector4::new( 1_f64,  1_f64, -1_f64, 1_f64),
    ///     Vector4::new(-1_f64,  1_f64, -1_f64, 1_f64),
    ///     Vector4::new(-1_f64, -1_f64, -1_f64, 1_f64),
    ///     Vector4::new( 1_f64, -1_f64, -1_f64, 1_f64),
    /// ];
    /// let expected = [
    ///     Vector4::new( 1_f64 + shear_factor / f64::sqrt(2_f64),  1_f64 + shear_factor / f64::sqrt(2_f64),  1_f64, 1_f64),
    ///     Vector4::new(-1_f64 + shear_factor / f64::sqrt(2_f64),  1_f64 + shear_factor / f64::sqrt(2_f64),  1_f64, 1_f64),
    ///     Vector4::new(-1_f64 + shear_factor / f64::sqrt(2_f64), -1_f64 + shear_factor / f64::sqrt(2_f64),  1_f64, 1_f64),
    ///     Vector4::new( 1_f64 + shear_factor / f64::sqrt(2_f64), -1_f64 + shear_factor / f64::sqrt(2_f64),  1_f64, 1_f64),
    ///     Vector4::new( 1_f64 - shear_factor / f64::sqrt(2_f64),  1_f64 - shear_factor / f64::sqrt(2_f64), -1_f64, 1_f64),
    ///     Vector4::new(-1_f64 - shear_factor / f64::sqrt(2_f64),  1_f64 - shear_factor / f64::sqrt(2_f64), -1_f64, 1_f64),
    ///     Vector4::new(-1_f64 - shear_factor / f64::sqrt(2_f64), -1_f64 - shear_factor / f64::sqrt(2_f64), -1_f64, 1_f64),
    ///     Vector4::new( 1_f64 - shear_factor / f64::sqrt(2_f64), -1_f64 - shear_factor / f64::sqrt(2_f64), -1_f64, 1_f64),
    /// ];
    /// let result = vertices.map(|v| matrix * v);
    ///
    /// assert_relative_eq!(result[0], expected[0], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[1], expected[1], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[2], expected[2], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[3], expected[3], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[4], expected[4], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[5], expected[5], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[6], expected[6], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result[7], expected[7], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    ///
    /// let vertices_in_plane = [
    ///     Vector4::new( 1_f64,  1_f64, 0_f64, 1_f64),
    ///     Vector4::new(-1_f64,  1_f64, 0_f64, 1_f64),
    ///     Vector4::new(-1_f64, -1_f64, 0_f64, 1_f64),
    ///     Vector4::new( 1_f64, -1_f64, 0_f64, 1_f64),
    ///     Vector4::new( 0_f64,  0_f64, 0_f64, 1_f64),
    /// ];
    /// let expected_in_plane = vertices_in_plane;
    /// let result_in_plane = vertices_in_plane.map(|v| matrix * v);
    ///
    /// assert_relative_eq!(result_in_plane[0], expected_in_plane[0], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_plane[1], expected_in_plane[1], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_plane[2], expected_in_plane[2], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_plane[3], expected_in_plane[3], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result_in_plane[4], expected_in_plane[4], abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_shear(
        shear_factor: S,
        origin: &Point3<S>,
        direction: &Unit<Vector3<S>>,
        normal: &Unit<Vector3<S>>
    ) -> Self
    {
        let zero = S::zero();
        let one = S::one();
        let translation = direction.into_inner() * (-shear_factor * origin.to_vector().dot(normal));

        let c0r0 = one + shear_factor * direction[0] * normal[0];
        let c0r1 = shear_factor * direction[1] * normal[0];
        let c0r2 = shear_factor * direction[2] * normal[0];
        let c0r3 = zero;

        let c1r0 = shear_factor * direction[0] * normal[1];
        let c1r1 = one + shear_factor * direction[1] * normal[1];
        let c1r2 = shear_factor * direction[2] * normal[1];
        let c1r3 = zero;

        let c2r0 = shear_factor * direction[0] * normal[2];
        let c2r1 = shear_factor * direction[1] * normal[2];
        let c2r2 = one + shear_factor * direction[2] * normal[2];
        let c2r3 = zero;

        let c3r0 = translation[0];
        let c3r1 = translation[1];
        let c3r2 = translation[2];
        let c3r3 = one;

        Self::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3,
        )
    }
}

impl<S> Matrix4x4<S>
where
    S: SimdScalarSigned,
{
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
    /// # Discussion
    ///
    /// The reflection of a point is defined as follows. Let `M` be the plane of
    /// reflection, also known as the **mirror plane**. Let `n` be a vector normal
    /// to the mirror plane `M`. Since `n` is normal to `M`, reflected points are
    /// reflected in a direction parallel to `n`, i.e. perpendicular to the mirror
    /// plane `M`. To reflect points correctly, we need a known point `Q` in the plane
    /// of reflection.
    ///
    /// For a vector `v`, we can choose vectors `v_per` and `v_par` such that
    /// `v == v_per + v_par`, `v_per` is perpendicular to the `n` and `v_par` is
    /// parallel to `n`. Stated different, `v_per` is parallel to the mirror plane `M`
    /// and `v_par` is perpendicular to the mirror plane `M`. The reflection `Ref` acts
    /// on `v_per` and `v_par` as follows
    /// ```text
    /// Ref(v_per) :=  v_per
    /// Ref(v_par) := -v_par
    /// ```
    /// by definition. This means that the reflection on vectors is defined by
    /// ```text
    /// Ref(v) := Ref(v_per + v_par)
    ///        := Ref(v_per) + Ref(v_par)
    ///        := Ref(v_per) - v_par
    ///        == v_per - v_par
    ///        == v - v_par - v_par
    ///        == v - 2 * v_par
    ///        == v - (2 * dot(v, n)) * n
    /// ```
    /// and reflection on points is defined by
    /// ```text
    /// Ref(P) := Ref(Q + (P - Q))
    ///        := Q + Ref(P - Q)
    ///        == Q + [(P - Q) - 2 * dot(P - Q, n) * n]
    ///        == P - 2 * dot(P - Q, n) * n
    ///        == I * P - (2 * dot(P, n)) * n + (2 * dot(Q, n)) * n
    ///        == [I - 2 * outer(n, n)] * P + (2 * dot(Q, n)) * n
    /// ```
    /// and the corresponding affine matrix has the form
    /// ```text
    /// M := | I - 2 * outer(n, n)   2 * dot(Q, n) * n |
    ///      | 0^T                   1                 |
    /// ```
    /// geometrically. In the standard basis in two-dimensional Euclidean space, we
    /// have
    /// ```text
    ///      |  1 - 2 * n.x * n.x   -2 * n.x * n.y       2 * dot(Q, n) * n.x |
    /// M == | -2 * n.y * n.x        1 - 2 * n.y * n.y   2 * dot(Q, n) * n.y |
    ///      |  0                    0                   1                   |
    /// ```
    /// and in three-dimensional Euclidean space we have
    /// ```text
    ///      |  1 - 2 * n.x * n.x   -2 * n.x * n.y       -2 * n.x * n.z        2 * dot(Q, n) * n.x |
    /// M == | -2 * n.y * n.x        1 - 2 * n.y * n.y   -2 * n.y * n.z        2 * dot(Q, n) * n.y |
    ///      | -2 * n.z * n.x       -2 * n.z * n.y        1 - 2 * n.z * n.z    2 * dot(Q, n) * n.z |
    ///      |  0                    0                    0                    1                   |
    /// ```
    /// which correspond exactly the how the respective matrices are implemented.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Point3,
    /// #     Unit,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// #
    /// let bias = Point3::new(0_f64, 0_f64, 0_f64);
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let expected = Matrix4x4::new(
    ///     1_f64, 0_f64,  0_f64, 0_f64,
    ///     0_f64, 1_f64,  0_f64, 0_f64,
    ///     0_f64, 0_f64, -1_f64, 0_f64,
    ///     0_f64, 0_f64,  0_f64, 1_f64,
    /// );
    /// let result = Matrix4x4::from_affine_reflection(&normal, &bias);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_reflection(normal: &Unit<Vector3<S>>, bias: &Point3<S>) -> Self {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;

        let c0r0 =  one - two * normal[0] * normal[0];
        let c0r1 = -two * normal[1] * normal[0];
        let c0r2 = -two * normal[2] * normal[0];
        let c0r3 = zero;

        let c1r0 = -two * normal[0] * normal[1];
        let c1r1 =  one - two * normal[1] * normal[1];
        let c1r2 = -two * normal[2] * normal[1];
        let c1r3 =  zero;

        let c2r0 = -two * normal[0] * normal[2];
        let c2r1 = -two * normal[1] * normal[2];
        let c2r2 =  one - two * normal[2] * normal[2];
        let c2r3 =  zero;

        let c3r0 = two * normal[0] * (normal[0] * bias[0] + normal[1] * bias[1] + normal[2] * bias[2]);
        let c3r1 = two * normal[1] * (normal[0] * bias[0] + normal[1] * bias[1] + normal[2] * bias[2]);
        let c3r2 = two * normal[2] * (normal[0] * bias[0] + normal[1] * bias[1] + normal[2] * bias[2]);
        let c3r3 = one;

        Self::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3,
        )
    }

    /// Compute the determinant of a matrix.
    ///
    /// The determinant of a matrix is the signed volume of the parallelepiped
    /// swept out by the vectors represented by the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix = Matrix4x4::new(
    ///     4_f64, 12_f64, 34_f64, 67_f64,
    ///     7_f64, 15_f64, 9_f64,  6_f64,
    ///     1_f64, 3_f64,  3_f64,  7_f64,
    ///     9_f64, 9_f64,  2_f64,  13_f64,
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

impl<S> Matrix4x4<S>
where
    S: SimdScalarFloat,
{
    /// Construct a three-dimensional affine rotation matrix rotating a vector around the
    /// **x-axis** by an angle `angle` radians/degrees.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix4x4::from_affine_angle_x(angle);
    /// let vector = Vector4::new(0_f64, 1_f64, 1_f64, 1_f64);
    /// let expected = Vector4::new(0_f64, -1_f64, 1_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_angle_x<A>(angle: A) -> Self
    where
        A: Into<Radians<S>>,
    {
        let (sin_angle, cos_angle) = angle.into().sin_cos();
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,   zero,      zero,      zero,
            zero,  cos_angle, sin_angle, zero,
            zero, -sin_angle, cos_angle, zero,
            zero,  zero,      zero,      one,
        )
    }

    /// Construct a three-dimensional affine rotation matrix rotating a vector
    /// around the **y-axis** by an angle `angle` radians/degrees.
    ///
    /// # Example
    ///
    /// In this example the rotation is in the **zx-plane**.
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix4x4::from_affine_angle_y(angle);
    /// let vector = Vector4::new(1_f64, 0_f64, 1_f64, 1_f64);
    /// let expected = Vector4::new(1_f64, 0_f64, -1_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_angle_y<A>(angle: A) -> Self
    where
        A: Into<Radians<S>>,
    {
        let (sin_angle, cos_angle) = angle.into().sin_cos();
        let one = S::one();
        let zero = S::zero();

        Self::new(
            cos_angle, zero, -sin_angle, zero,
            zero,      one,   zero,      zero,
            sin_angle, zero,  cos_angle, zero,
            zero,      zero,  zero,      one,
        )
    }

    /// Construct a three-dimensional affine rotation matrix rotating a vector
    /// around the **z-axis** by an angle `angle` radians/degrees.
    ///
    /// # Example
    ///
    /// In this example the rotation is in the **xy-plane**.
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix4x4::from_affine_angle_z(angle);
    /// let vector = Vector4::new(1_f64, 1_f64, 0_f64, 1_f64);
    /// let expected = Vector4::new(-1_f64, 1_f64, 0_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_angle_z<A>(angle: A) -> Self
    where
        A: Into<Radians<S>>,
    {
        let (sin_angle, cos_angle) = angle.into().sin_cos();
        let one = S::one();
        let zero = S::zero();

        Self::new(
             cos_angle, sin_angle, zero, zero,
            -sin_angle, cos_angle, zero, zero,
             zero,      zero,      one,  zero,
             zero,      zero,      zero, one,
        )
    }

    /// Construct a three-dimensional affine rotation matrix rotating a vector
    /// around the axis `axis` by an angle `angle` radians/degrees.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Unit,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix4x4::from_affine_axis_angle(&axis, angle);
    /// let vector = Vector4::new(1_f64, 0_f64, 0_f64, 1_f64);
    /// let expected = Vector4::new(0_f64, 1_f64, 0_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_axis_angle<A>(axis: &Unit<Vector3<S>>, angle: A) -> Self
    where
        A: Into<Radians<S>>,
    {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());
        let one_minus_cos_angle = S::one() - cos_angle;
        let _axis = axis.as_ref();

        Self::new(
            one_minus_cos_angle * _axis[0] * _axis[0] + cos_angle,
            one_minus_cos_angle * _axis[0] * _axis[1] + sin_angle * _axis[2],
            one_minus_cos_angle * _axis[0] * _axis[2] - sin_angle * _axis[1],
            S::zero(),

            one_minus_cos_angle * _axis[0] * _axis[1] - sin_angle * _axis[2],
            one_minus_cos_angle * _axis[1] * _axis[1] + cos_angle,
            one_minus_cos_angle * _axis[1] * _axis[2] + sin_angle * _axis[0],
            S::zero(),

            one_minus_cos_angle * _axis[0] * _axis[2] + sin_angle * _axis[1],
            one_minus_cos_angle * _axis[1] * _axis[2] - sin_angle * _axis[0],
            one_minus_cos_angle * _axis[2] * _axis[2] + cos_angle,
            S::zero(),

            S::zero(),
            S::zero(),
            S::zero(),
            S::one(),
        )
    }

    /// Construct a new orthographic projection matrix.
    ///
    /// The `near` and `far` parameters are the absolute values of the positions
    /// of the **near plane** and the **far** plane, respectively, along the
    /// **negative z-axis**. In particular, the position of the **near plane** is
    /// `z == -near` and the position of the **far plane** is `z == -far`.
    ///
    /// This function returns a homogeneous matrix representing an orthographic
    /// projection transformation with a right-handed coordinate system where the
    /// orthographic camera faces the **negative z-axis** with the **positive x-axis**
    /// going to the right, and the **positive y-axis** going up. The orthographic view
    /// volume is the box `[left, right] x [bottom, top] x [-near, -far]`. The
    /// normalized device coordinates this transformation maps to are
    /// `[-1, 1] x [-1, 1] x [-1, 1]`.
    ///
    /// The resulting matrix is identical to the one used by OpenGL. We provide
    /// it here for reference
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
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let left = -4_f64;
    /// let right = 4_f64;
    /// let bottom = -2_f64;
    /// let top = 2_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let expected = Matrix4x4::new(
    ///     1_f64 / 4_f64,  0_f64,          0_f64,            0_f64,
    ///     0_f64,          1_f64 / 2_f64,  0_f64,            0_f64,
    ///     0_f64,          0_f64,         -2_f64 / 99_f64,   0_f64,
    ///     0_f64,          0_f64,         -101_f64 / 99_f64, 1_f64,
    /// );
    /// let result = Matrix4x4::from_orthographic(left, right, bottom, top, near, far);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_orthographic(left: S, right: S, bottom: S, top: S, near: S, far: S) -> Self {
        let zero = S::zero();
        let one  = S::one();
        let two = one + one;

        let c0r0 =  two / (right - left);
        let c0r1 = zero;
        let c0r2 = zero;
        let c0r3 = zero;

        let c1r0 = zero;
        let c1r1 = two / (top - bottom);
        let c1r2 = zero;
        let c1r3 = zero;

        let c2r0 = zero;
        let c2r1 = zero;
        let c2r2 = -two / (far - near);
        let c2r3 = zero;

        let c3r0 = -(right + left) / (right - left);
        let c3r1 = -(top + bottom) / (top - bottom);
        let c3r2 = -(far + near) / (far - near);
        let c3r3 = one;

        Self::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3,
        )
    }

    /// Construct a new possibly off-center perspective projection matrix based
    /// on arbitrary `left`, `right`, `bottom`, `top`, `near` and `far` planes.
    ///
    /// The `near` and `far` parameters are the absolute values of the positions
    /// of the **near plane** and the **far** plane, respectively, along the
    /// **negative z-axis**. In particular, the position of the **near plane** is
    /// `z == -near` and the position of the **far plane** is `z == -far`.
    ///
    /// This function returns a homogeneous matrix representing a perspective
    /// projection transformation with a right-handed coordinate system where the
    /// perspective camera faces the **negative z-axis** with the **positive x-axis**
    /// going to the right, and the **positive y-axis** going up. The perspective view
    /// volume is the frustum contained in
    /// `[left, right] x [bottom, top] x [-near, -far]`. The normalized device
    /// coordinates this transformation maps to are `[-1, 1] x [-1, 1] x [-1, 1]`.
    ///
    /// The resulting matrix is identical to the one used by OpenGL, provided here for
    /// reference
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
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let left = -4_f64;
    /// let right = 4_f64;
    /// let bottom = -2_f64;
    /// let top = 3_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let expected = Matrix4x4::new(
    ///     1_f64 / 4_f64,  0_f64,          0_f64,             0_f64,
    ///     0_f64,          2_f64 / 5_f64,  0_f64,             0_f64,
    ///     0_f64,          1_f64 / 5_f64, -101_f64 / 99_f64, -1_f64,
    ///     0_f64,          0_f64,         -200_f64 / 99_f64,  0_f64,
    /// );
    /// let result = Matrix4x4::from_perspective(left, right, bottom, top, near, far);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_perspective(left: S, right: S, bottom: S, top: S, near: S, far: S) -> Self {
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

        Self::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3,
        )
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
    /// This function returns a homogeneous matrix representing a perspective
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
    /// The resulting matrix is identical to the one used by OpenGL, provided here for
    /// reference
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
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// # use cglinalg_trigonometry::Degrees;
    /// #
    /// let vfov = Degrees(72_f32);
    /// let aspect_ratio = 800_f32 / 600_f32;
    /// let near = 0.1_f32;
    /// let far = 100_f32;
    /// let expected = Matrix4x4::new(
    ///     1.0322863_f32, 0_f32,          0_f32,          0_f32,
    ///     0_f32,         1.3763818_f32,  0_f32,          0_f32,
    ///     0_f32,         0_f32,         -1.002002_f32,  -1_f32,
    ///     0_f32,         0_f32,         -0.2002002_f32,  0_f32,
    /// );
    /// let result = Matrix4x4::from_perspective_fov(vfov, aspect_ratio, near, far);
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-6, relative_all <= f32::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_perspective_fov<A>(vfov: A, aspect_ratio: S, near: S, far: S) -> Self
    where
        A: Into<Radians<S>>,
    {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let range = Angle::tan(vfov.into() / two) * near;

        let c0r0 = (two * near) / (range * aspect_ratio + range * aspect_ratio);
        let c0r1 = zero;
        let c0r2 = zero;
        let c0r3 = zero;

        let c1r0 = zero;
        let c1r1 = near / range;
        let c1r2 = zero;
        let c1r3 = zero;

        let c2r0 = zero;
        let c2r1 = zero;
        let c2r2 = -(far + near) / (far - near);
        let c2r3 = -one;

        let c3r0 = zero;
        let c3r1 = zero;
        let c3r2 = -(two * far * near) / (far - near);
        let c3r3 = zero;

        Self::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3,
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
    /// coordinate transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(4_f64, 5_f64, 3_f64);
    /// let direction = target - eye;
    /// let up = Vector3::unit_z();
    /// let expected = Matrix4x4::new(
    ///      -1_f64 / f64::sqrt(2_f64),  0_f64,  1_f64 / f64::sqrt(2_f64),  0_f64,
    ///       1_f64 / f64::sqrt(2_f64),  0_f64,  1_f64 / f64::sqrt(2_f64),  0_f64,
    ///       0_f64,                     1_f64,  0_f64,                     0_f64,
    ///      -1_f64 / f64::sqrt(2_f64), -3_f64, -3_f64 / f64::sqrt(2_f64),  1_f64,
    /// );
    /// let result = Matrix4x4::look_to_lh(&eye, &direction, &up);
    /// let unit_z = Vector4::unit_z();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(
    ///     (result * direction.to_homogeneous()).normalize(),
    ///     unit_z,
    ///     abs_diff_all <= 1e-10,
    ///     relative_all <= f64::EPSILON,
    /// );
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn look_to_lh(eye: &Point3<S>, direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let zero = S::zero();
        let one = S::one();
        let z_axis = direction.normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        let eye_vec = eye - Point3::origin();
        let neg_eye_x = -eye_vec.dot(&x_axis);
        let neg_eye_y = -eye_vec.dot(&y_axis);
        let neg_eye_z = -eye_vec.dot(&z_axis);

        Self::new(
            x_axis[0], y_axis[0], z_axis[0], zero,
            x_axis[1], y_axis[1], z_axis[1], zero,
            x_axis[2], y_axis[2], z_axis[2], zero,
            neg_eye_x, neg_eye_y, neg_eye_z, one,
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
    /// coordinate transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(4_f64, 5_f64, 3_f64);
    /// let direction = target - eye;
    /// let up = Vector3::unit_z();
    /// let expected = Matrix4x4::new(
    ///      1_f64 / f64::sqrt(2_f64),  0_f64, -1_f64 / f64::sqrt(2_f64),  0_f64,
    ///     -1_f64 / f64::sqrt(2_f64),  0_f64, -1_f64 / f64::sqrt(2_f64),  0_f64,
    ///      0_f64,                     1_f64,  0_f64,                     0_f64,
    ///      1_f64 / f64::sqrt(2_f64), -3_f64,  3_f64 / f64::sqrt(2_f64),  1_f64,
    /// );
    /// let result = Matrix4x4::look_to_rh(&eye, &direction, &up);
    /// let minus_unit_z = -Vector4::unit_z();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(
    ///     (result * direction.to_homogeneous()).normalize(),
    ///     minus_unit_z,
    ///     abs_diff_all <= 1e-10,
    ///     relative_all <= f64::EPSILON,
    /// );
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn look_to_rh(eye: &Point3<S>, direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let zero = S::zero();
        let one = S::one();
        let z_axis = (-direction).normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        let eye_vec = eye - Point3::origin();
        let neg_eye_x = -eye_vec.dot(&x_axis);
        let neg_eye_y = -eye_vec.dot(&y_axis);
        let neg_eye_z = -eye_vec.dot(&z_axis);

        Self::new(
            x_axis[0], y_axis[0], z_axis[0], zero,
            x_axis[1], y_axis[1], z_axis[1], zero,
            x_axis[2], y_axis[2], z_axis[2], zero,
            neg_eye_x, neg_eye_y, neg_eye_z, one,
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
    /// coordinate transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(1_f64, 2_f64, 3_f64);
    /// let target = Point3::new(4_f64, 5_f64, 3_f64);
    /// let up = Vector3::unit_z();
    /// let expected = Matrix4x4::new(
    ///      -1_f64 / f64::sqrt(2_f64),  0_f64,  1_f64 / f64::sqrt(2_f64),  0_f64,
    ///       1_f64 / f64::sqrt(2_f64),  0_f64,  1_f64 / f64::sqrt(2_f64),  0_f64,
    ///       0_f64,                     1_f64,  0_f64,                     0_f64,
    ///      -1_f64 / f64::sqrt(2_f64), -3_f64, -3_f64 / f64::sqrt(2_f64),  1_f64,
    /// );
    /// let result = Matrix4x4::look_at_lh(&eye, &target, &up);
    /// let direction = (target - eye).to_homogeneous();
    /// let unit_z = Vector4::unit_z();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(
    ///     (result * direction).normalize(),
    ///     unit_z,
    ///     abs_diff_all <= 1e-10,
    ///     relative_all <= f64::EPSILON,
    /// );
    /// ```
    #[inline]
    pub fn look_at_lh(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        Self::look_to_lh(eye, &(target - eye), up)
    }

    /// Construct an affine coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the position `eye` facing
    /// the position `target` into the coordinate system of an observer located
    /// at the origin facing the **negative z-axis**.
    ///
    /// The function maps the direction of the target `target` to the
    /// **negative z-axis** and locates the `eye` position to the origin in the
    /// new the coordinate system. This transformation is a **right-handed**
    /// coordinate transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// #     Vector4,
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
    ///      1_f64 / f64::sqrt(2_f64), -3_f64,  3_f64 / f64::sqrt(2_f64),  1_f64,
    /// );
    /// let result = Matrix4x4::look_at_rh(&eye, &target, &up);
    /// let direction = (target - eye).to_homogeneous();
    /// let minus_unit_z = -Vector4::unit_z();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(
    ///     (result * direction).normalize(),
    ///     minus_unit_z,
    ///     abs_diff_all <= 1e-10,
    ///     relative_all <= f64::EPSILON,
    /// );
    /// ```
    #[inline]
    pub fn look_at_rh(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        Self::look_to_rh(eye, &(target - eye), up)
    }

    /// Construct an affine coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing the
    /// **positive z-axis** into a coordinate system where the observer is
    /// located at the position `eye` facing the direction `direction`. The
    /// resulting coordinate transformation is a **left-handed** coordinate
    /// transformation.
    ///
    /// The function maps the **positive z-axis** to the direction `direction`,
    /// and locates the origin of the coordinate system to the `eye` position.
    /// This function is the inverse of `look_to_lh`.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Point3,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// #
    /// let eye = Point3::new(-2_f64, 3_f64, -4_f64);
    /// let direction: Vector3<f64> = -Vector3::unit_x();
    /// let up = Vector3::unit_z();
    /// let expected = Matrix4x4::new(
    ///      0_f64, -1_f64, 0_f64, 0_f64,
    ///      0_f64,  0_f64, 1_f64, 0_f64,
    ///     -1_f64,  0_f64, 0_f64, 0_f64,
    ///     -3_f64, -4_f64, 2_f64, 1_f64,
    /// );
    /// let result = Matrix4x4::look_to_lh_inv(&eye, &direction, &up);
    /// let direction = direction.to_homogeneous();
    /// let unit_z = Vector4::unit_z();
    /// let minus_unit_z = -unit_z;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * unit_z, direction, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * minus_unit_z, -direction, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn look_to_lh_inv(eye: &Point3<S>, direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let zero = S::zero();
        let one = S::one();
        let z_axis = direction.normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        let eye_vec = eye - Point3::origin();
        let eye_x = eye_vec.dot(&x_axis);
        let eye_y = eye_vec.dot(&y_axis);
        let eye_z = eye_vec.dot(&z_axis);

        Self::new(
            x_axis[0], x_axis[1], x_axis[2], zero,
            y_axis[0], y_axis[1], y_axis[2], zero,
            z_axis[0], z_axis[1], z_axis[2], zero,
            eye_x,     eye_y,     eye_z,     one,
        )
    }

    /// Construct an affine coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing the
    /// **negative z-axis** into a coordinate system where the observer is
    /// located at the position `eye` facing the direction `direction`. The
    /// resulting coordinate transformation is a **right-handed** coordinate
    /// transformation.
    ///
    /// The function maps the **negative z-axis** to the direction `direction`,
    /// and locates the origin of the coordinate system to the `eye` position.
    /// This function is the inverse of `look_to_rh`.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// #
    /// let eye = Point3::new(-2_f64, 3_f64, -4_f64);
    /// let direction: Vector3<f64> = -Vector3::unit_x();
    /// let up = Vector3::unit_z();
    /// let expected = Matrix4x4::new(
    ///     0_f64,  1_f64,  0_f64, 0_f64,
    ///     0_f64,  0_f64,  1_f64, 0_f64,
    ///     1_f64,  0_f64,  0_f64, 0_f64,
    ///     3_f64, -4_f64, -2_f64, 1_f64,
    /// );
    /// let result = Matrix4x4::look_to_rh_inv(&eye, &direction, &up);
    /// let direction = direction.to_homogeneous().normalize();
    /// let unit_z = Vector4::unit_z();
    /// let minus_unit_z = -unit_z;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * unit_z, -direction, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * minus_unit_z, direction, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn look_to_rh_inv(eye: &Point3<S>, direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let zero = S::zero();
        let one = S::one();
        let z_axis = (-direction).normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        let eye_vec = eye - Point3::origin();
        let eye_x = eye_vec.dot(&x_axis);
        let eye_y = eye_vec.dot(&y_axis);
        let eye_z = eye_vec.dot(&z_axis);

        Self::new(
            x_axis[0], x_axis[1], x_axis[2], zero,
            y_axis[0], y_axis[1], y_axis[2], zero,
            z_axis[0], z_axis[1], z_axis[2], zero,
            eye_x,     eye_y,     eye_z,     one,
        )
    }

    /// Construct an affine coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing the
    /// **positive z-axis** into a coordinate system where the observer is
    /// located at the position `eye` facing the direction `direction`. The
    /// resulting coordinate transformation is a **left-handed** coordinate
    /// transformation.
    ///
    /// The function maps the **positive z-axis** to the direction `direction`,
    /// and locates the origin of the coordinate system to the `eye` position.
    /// This function is the inverse of `look_at_lh`.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// #     Vector4
    /// # };
    /// #
    /// let eye = Point3::new(-2_f64, 3_f64, -4_f64);
    /// let target = Point3::new(-4_f64, 3_f64, -4_f64);
    /// let up = Vector3::unit_z();
    /// let expected = Matrix4x4::new(
    ///      0_f64, -1_f64, 0_f64, 0_f64,
    ///      0_f64,  0_f64, 1_f64, 0_f64,
    ///     -1_f64,  0_f64, 0_f64, 0_f64,
    ///     -3_f64, -4_f64, 2_f64, 1_f64,
    /// );
    /// let result = Matrix4x4::look_at_lh_inv(&eye, &target, &up);
    /// let direction = (target - eye).to_homogeneous().normalize();
    /// let unit_z = Vector4::unit_z();
    /// let minus_unit_z = -unit_z;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * unit_z, direction, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * minus_unit_z, -direction, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn look_at_lh_inv(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        Self::look_to_lh_inv(eye, &(target - eye), up)
    }

    /// Construct an affine coordinate transformation matrix that transforms
    /// a coordinate system of an observer located at the origin facing the
    /// **negative z-axis** into a coordinate system where the observer is
    /// located at the position `eye` facing the direction `direction`. The
    /// resulting coordinate transformation is a **right-handed** coordinate
    /// transformation.
    ///
    /// The function maps the **negative z-axis** to the direction `direction`,
    /// and locates the origin of the coordinate system to the `eye` position.
    /// This function is the inverse of `look_at_rh`.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Normed,
    /// #     Point3,
    /// #     Vector3,
    /// #     Vector4,
    /// # };
    /// #
    /// let eye = Point3::new(-2_f64, 3_f64, -4_f64);
    /// let target = Point3::new(-4_f64, 3_f64, -4_f64);
    /// let up = Vector3::unit_z();
    /// let expected = Matrix4x4::new(
    ///     0_f64,  1_f64,  0_f64, 0_f64,
    ///     0_f64,  0_f64,  1_f64, 0_f64,
    ///     1_f64,  0_f64,  0_f64, 0_f64,
    ///     3_f64, -4_f64, -2_f64, 1_f64,
    /// );
    /// let result = Matrix4x4::look_at_rh_inv(&eye, &target, &up);
    /// let direction = (target - eye).to_homogeneous().normalize();
    /// let unit_z = Vector4::unit_z();
    /// let minus_unit_z = -unit_z;
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * unit_z, -direction, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// assert_relative_eq!(result * minus_unit_z, direction, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    /// ```
    #[inline]
    pub fn look_at_rh_inv(eye: &Point3<S>, target: &Point3<S>, up: &Vector3<S>) -> Self {
        Self::look_to_rh_inv(eye, &(target - eye), up)
    }


    /// Compute the inverse of a square matrix, if the inverse exists.
    ///
    /// Given a square matrix `self` Compute the matrix `m` if it exists
    /// such that
    /// ```text
    /// m * self == self * m == 1.
    /// ```
    /// Not every square matrix has an inverse.
    ///
    /// # Example
    ///
    /// ```
    /// # use approx_cmp::assert_relative_eq;
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_f64, 4_f64, 7_f64,  8_f64,
    ///     2_f64, 5_f64, 8_f64,  4_f64,
    ///     5_f64, 6_f64, 11_f64, 4_f64,
    ///     9_f64, 3_f64, 13_f64, 5_f64,
    /// );
    /// let expected = Matrix4x4::new(
    ///      17_f64 / 60_f64, -41_f64 / 30_f64,  21_f64 / 20_f64, -1_f64 / 5_f64,
    ///      7_f64 / 30_f64,  -16_f64 / 15_f64,  11_f64 / 10_f64, -2_f64 / 5_f64,
    ///     -13_f64 / 36_f64,  25_f64 / 18_f64, -13_f64 / 12_f64,  1_f64 / 3_f64,
    ///      13_f64 / 45_f64, -23_f64 / 45_f64,  4_f64 / 15_f64,  -1_f64 / 15_f64,
    /// );
    /// let result = matrix.inverse().unwrap();
    ///
    /// assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
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

            Some(Self::new(
                c0r0, c0r1, c0r2, c0r3,
                c1r0, c1r1, c1r2, c1r3,
                c2r0, c2r1, c2r2, c2r3,
                c3r0, c3r1, c3r2, c3r3,
            ))
        }
    }

    /// Determine whether a square matrix has an inverse matrix.
    ///
    /// A matrix is invertible if its determinant is not zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x4;
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_f64,  2_f64,  3_f64,  4_f64,
    ///     5_f64,  6_f64,  7_f64,  8_f64,
    ///     9_f64,  10_f64, 11_f64, 12_f64,
    ///     13_f64, 14_f64, 15_f64, 16_f64,
    /// );
    ///
    /// assert_eq!(matrix.determinant(), 0_f64);
    /// assert!(!matrix.is_invertible());
    /// ```
    #[inline]
    pub fn is_invertible(&self) -> bool {
        ulps_ne!(self.determinant(), S::zero(), abs_diff_all <= S::machine_epsilon(), ulps_all <= S::default_ulps())
    }
}

impl<S> Matrix1x2<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix1x2;
    /// #
    /// let c0r0 = 1_i32;
    /// let c1r0 = 2_i32;
    /// let matrix = Matrix1x2::new(
    ///     c0r0,
    ///     c1r0,
    /// );
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// assert_eq!(matrix[1][0], c1r0);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(c0r0: S, c1r0: S) -> Self {
        Self {
            data: [
                [c0r0],
                [c1r0],
            ],
        }
    }
}

impl<S> Matrix1x3<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix1x3;
    /// #
    /// let c0r0 = 1_i32;
    /// let c1r0 = 2_i32;
    /// let c2r0 = 3_i32;
    /// let matrix = Matrix1x3::new(
    ///     c0r0,
    ///     c1r0,
    ///     c2r0,
    /// );
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// assert_eq!(matrix[1][0], c1r0);
    /// assert_eq!(matrix[2][0], c2r0);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(c0r0: S, c1r0: S, c2r0: S) -> Self {
        Self {
            data: [
                [c0r0],
                [c1r0],
                [c2r0],
            ],
        }
    }
}

impl<S> Matrix1x4<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix1x4;
    /// #
    /// let c0r0 = 1_i32;
    /// let c1r0 = 2_i32;
    /// let c2r0 = 3_i32;
    /// let c3r0 = 4_i32;
    /// let matrix = Matrix1x4::new(
    ///     c0r0,
    ///     c1r0,
    ///     c2r0,
    ///     c3r0,
    /// );
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// assert_eq!(matrix[1][0], c1r0);
    /// assert_eq!(matrix[2][0], c2r0);
    /// assert_eq!(matrix[3][0], c3r0);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(c0r0: S, c1r0: S, c2r0: S, c3r0: S) -> Self {
        Self {
            data: [
                [c0r0],
                [c1r0],
                [c2r0],
                [c3r0],
            ],
        }
    }
}

impl<S> Matrix2x3<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix2x3;
    /// #
    /// let c0r0 = 1_i32; let c0r1 = 2_i32;
    /// let c1r0 = 3_i32; let c1r1 = 4_i32;
    /// let c2r0 = 5_i32; let c2r1 = 6_i32;
    /// let matrix = Matrix2x3::new(
    ///     c0r0, c0r1,
    ///     c1r0, c1r1,
    ///     c2r0, c2r1,
    /// );
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// assert_eq!(matrix[0][1], c0r1);
    /// assert_eq!(matrix[1][0], c1r0);
    /// assert_eq!(matrix[1][1], c1r1);
    /// assert_eq!(matrix[2][0], c2r0);
    /// assert_eq!(matrix[2][1], c2r1);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(
        c0r0: S, c0r1: S,
        c1r0: S, c1r1: S,
        c2r0: S, c2r1: S) -> Self
    {
        Self {
            data: [
                [c0r0, c0r1],
                [c1r0, c1r1],
                [c2r0, c2r1],
            ],
        }
    }
}

impl<S> Matrix3x2<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x2;
    /// #
    /// let c0r0 = 1_i32; let c0r1 = 2_i32; let c0r2 = 3_i32;
    /// let c1r0 = 4_i32; let c1r1 = 5_i32; let c1r2 = 6_i32;
    /// let matrix = Matrix3x2::new(
    ///     c0r0, c0r1, c0r2,
    ///     c1r0, c1r1, c1r2,
    /// );
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// assert_eq!(matrix[0][1], c0r1);
    /// assert_eq!(matrix[0][2], c0r2);
    /// assert_eq!(matrix[1][0], c1r0);
    /// assert_eq!(matrix[1][1], c1r1);
    /// assert_eq!(matrix[1][2], c1r2);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(
        c0r0: S, c0r1: S, c0r2: S,
        c1r0: S, c1r1: S, c1r2: S) -> Self
    {
        Self {
            data: [
                [c0r0, c0r1, c0r2],
                [c1r0, c1r1, c1r2],
            ],
        }
    }
}

impl<S> Matrix2x4<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix2x4;
    /// #
    /// let c0r0 = 1_i32; let c0r1 = 2_i32;
    /// let c1r0 = 3_i32; let c1r1 = 4_i32;
    /// let c2r0 = 5_i32; let c2r1 = 6_i32;
    /// let c3r0 = 7_i32; let c3r1 = 8_i32;
    /// let matrix = Matrix2x4::new(
    ///     c0r0, c0r1,
    ///     c1r0, c1r1,
    ///     c2r0, c2r1,
    ///     c3r0, c3r1,
    /// );
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// assert_eq!(matrix[0][1], c0r1);
    /// assert_eq!(matrix[1][0], c1r0);
    /// assert_eq!(matrix[1][1], c1r1);
    /// assert_eq!(matrix[2][0], c2r0);
    /// assert_eq!(matrix[2][1], c2r1);
    /// assert_eq!(matrix[3][0], c3r0);
    /// assert_eq!(matrix[3][1], c3r1);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(
        c0r0: S, c0r1: S,
        c1r0: S, c1r1: S,
        c2r0: S, c2r1: S,
        c3r0: S, c3r1: S) -> Self
    {
        Self {
            data: [
                [c0r0, c0r1],
                [c1r0, c1r1],
                [c2r0, c2r1],
                [c3r0, c3r1],
            ],
        }
    }
}

impl<S> Matrix4x2<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x2;
    /// #
    /// let c0r0 = 1_i32;
    /// let c0r1 = 2_i32;
    /// let c0r2 = 3_i32;
    /// let c0r3 = 4_i32;
    /// let c1r0 = 5_i32;
    /// let c1r1 = 6_i32;
    /// let c1r2 = 7_i32;
    /// let c1r3 = 8_i32;
    /// let matrix = Matrix4x2::new(
    ///     c0r0, c0r1, c0r2, c0r3,
    ///     c1r0, c1r1, c1r2, c1r3,
    /// );
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// assert_eq!(matrix[0][1], c0r1);
    /// assert_eq!(matrix[0][2], c0r2);
    /// assert_eq!(matrix[0][3], c0r3);
    /// assert_eq!(matrix[1][0], c1r0);
    /// assert_eq!(matrix[1][1], c1r1);
    /// assert_eq!(matrix[1][2], c1r2);
    /// assert_eq!(matrix[1][3], c1r3);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(
        c0r0: S, c0r1: S, c0r2: S, c0r3: S,
        c1r0: S, c1r1: S, c1r2: S, c1r3: S) -> Self
    {
        Self {
            data: [
                [c0r0, c0r1, c0r2, c0r3],
                [c1r0, c1r1, c1r2, c1r3],
            ]
        }
    }
}

impl<S> Matrix3x4<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix3x4;
    /// #
    /// let c0r0 = 1_i32;  let c0r1 = 2_i32;  let c0r2 = 3_i32;
    /// let c1r0 = 4_i32;  let c1r1 = 5_i32;  let c1r2 = 6_i32;
    /// let c2r0 = 7_i32;  let c2r1 = 8_i32;  let c2r2 = 9_i32;
    /// let c3r0 = 10_i32; let c3r1 = 11_i32; let c3r2 = 12_i32;
    /// let matrix = Matrix3x4::new(
    ///     c0r0, c0r1, c0r2,
    ///     c1r0, c1r1, c1r2,
    ///     c2r0, c2r1, c2r2,
    ///     c3r0, c3r1, c3r2,
    /// );
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// assert_eq!(matrix[0][1], c0r1);
    /// assert_eq!(matrix[1][0], c1r0);
    /// assert_eq!(matrix[1][1], c1r1);
    /// assert_eq!(matrix[2][0], c2r0);
    /// assert_eq!(matrix[2][1], c2r1);
    /// assert_eq!(matrix[3][0], c3r0);
    /// assert_eq!(matrix[3][1], c3r1);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(
        c0r0: S, c0r1: S, c0r2: S,
        c1r0: S, c1r1: S, c1r2: S,
        c2r0: S, c2r1: S, c2r2: S,
        c3r0: S, c3r1: S, c3r2: S) -> Self
    {
        Self {
            data: [
                [c0r0, c0r1, c0r2],
                [c1r0, c1r1, c1r2],
                [c2r0, c2r1, c2r2],
                [c3r0, c3r1, c3r2],
            ],
        }
    }
}

impl<S> Matrix4x3<S> {
    /// Construct a new matrix from its elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::Matrix4x3;
    /// #
    /// let c0r0 = 1_i32; let c0r1 = 2_i32;  let c0r2 = 3_i32;  let c0r3 = 4_i32;
    /// let c1r0 = 5_i32; let c1r1 = 6_i32;  let c1r2 = 7_i32;  let c1r3 = 8_i32;
    /// let c2r0 = 9_i32; let c2r1 = 10_i32; let c2r2 = 11_i32; let c2r3 = 12_i32;
    /// let matrix = Matrix4x3::new(
    ///     c0r0, c0r1, c0r2, c0r3,
    ///     c1r0, c1r1, c1r2, c1r3,
    ///     c2r0, c2r1, c2r2, c2r3,
    /// );
    ///
    /// assert_eq!(matrix[0][0], c0r0);
    /// assert_eq!(matrix[0][1], c0r1);
    /// assert_eq!(matrix[0][2], c0r2);
    /// assert_eq!(matrix[0][3], c0r3);
    /// assert_eq!(matrix[1][0], c1r0);
    /// assert_eq!(matrix[1][1], c1r1);
    /// assert_eq!(matrix[1][2], c1r2);
    /// assert_eq!(matrix[1][3], c1r3);
    /// assert_eq!(matrix[2][0], c2r0);
    /// assert_eq!(matrix[2][1], c2r1);
    /// assert_eq!(matrix[2][2], c2r2);
    /// assert_eq!(matrix[2][3], c2r3);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub const fn new(
        c0r0: S, c0r1: S, c0r2: S, c0r3: S,
        c1r0: S, c1r1: S, c1r2: S, c1r3: S,
        c2r0: S, c2r1: S, c2r2: S, c2r3: S) -> Self
    {
        Self {
            data: [
                [c0r0, c0r1, c0r2, c0r3],
                [c1r0, c1r1, c1r2, c1r3],
                [c2r0, c2r1, c2r2, c2r3],
            ],
        }
    }
}

impl_coords!(View1x1, { c0r0, });
impl_coords!(View2x2, { c0r0, c0r1, c1r0, c1r1, });
impl_coords!(View3x3, { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2, });
impl_coords!(View4x4, {
    c0r0, c0r1, c0r2, c0r3,
    c1r0, c1r1, c1r2, c1r3,
    c2r0, c2r1, c2r2, c2r3,
    c3r0, c3r1, c3r2, c3r3,
});
impl_coords!(View1x2, { c0r0, c1r0, });
impl_coords!(View1x3, { c0r0, c1r0, c2r0, });
impl_coords!(View1x4, { c0r0, c1r0, c2r0, c3r0, });
impl_coords!(View2x3, { c0r0, c0r1, c1r0, c1r1, c2r0, c2r1, });
impl_coords!(View2x4, { c0r0, c0r1, c1r0, c1r1, c2r0, c2r1, c3r0, c3r1, });
impl_coords!(View3x4, {
    c0r0, c0r1, c0r2,
    c1r0, c1r1, c1r2,
    c2r0, c2r1, c2r2,
    c3r0, c3r1, c3r2,
});
impl_coords!(View3x2, { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, });
impl_coords!(View4x2, { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, });
impl_coords!(View4x3, {
    c0r0, c0r1, c0r2, c0r3,
    c1r0, c1r1, c1r2, c1r3,
    c2r0, c2r1, c2r2, c2r3,
});

impl_coords_deref!(Matrix1x1, View1x1);
impl_coords_deref!(Matrix2x2, View2x2);
impl_coords_deref!(Matrix3x3, View3x3);
impl_coords_deref!(Matrix4x4, View4x4);
impl_coords_deref!(Matrix1x2, View1x2);
impl_coords_deref!(Matrix1x3, View1x3);
impl_coords_deref!(Matrix1x4, View1x4);
impl_coords_deref!(Matrix2x3, View2x3);
impl_coords_deref!(Matrix2x4, View2x4);
impl_coords_deref!(Matrix3x2, View3x2);
impl_coords_deref!(Matrix3x4, View3x4);
impl_coords_deref!(Matrix4x2, View4x2);
impl_coords_deref!(Matrix4x3, View4x3);


impl<S, const R: usize, const C: usize> ops::Add<Matrix<S, R, C>> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn add(self, other: Matrix<S, R, C>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] + other.data[c][r];
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Add<&Matrix<S, R, C>> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn add(self, other: &Matrix<S, R, C>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] + other.data[c][r];
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Add<Matrix<S, R, C>> for &Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn add(self, other: Matrix<S, R, C>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] + other.data[c][r];
            }
        }

        result
    }
}

impl<'a, 'b, S, const R: usize, const C: usize> ops::Add<&'b Matrix<S, R, C>> for &'a Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn add(self, other: &'b Matrix<S, R, C>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] + other.data[c][r];
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Sub<Matrix<S, R, C>> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn sub(self, other: Matrix<S, R, C>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] - other.data[c][r];
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Sub<&Matrix<S, R, C>> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn sub(self, other: &Matrix<S, R, C>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] - other.data[c][r];
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Sub<Matrix<S, R, C>> for &Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn sub(self, other: Matrix<S, R, C>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] - other.data[c][r];
            }
        }

        result
    }
}

impl<'a, 'b, S, const R: usize, const C: usize> ops::Sub<&'b Matrix<S, R, C>> for &'a Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn sub(self, other: &'b Matrix<S, R, C>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] - other.data[c][r];
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Neg for Matrix<S, R, C>
where
    S: SimdScalarSigned,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn neg(self) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = -self.data[c][r];
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Neg for &Matrix<S, R, C>
where
    S: SimdScalarSigned,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn neg(self) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = -self.data[c][r];
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Mul<S> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] * other;
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Mul<S> for &Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] * other;
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Div<S> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] / other;
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Div<S> for &Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] / other;
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Rem<S> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] % other;
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Rem<S> for &Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Matrix<S, R, C>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                result[c][r] = self.data[c][r] % other;
            }
        }

        result
    }
}


macro_rules! impl_scalar_matrix_mul_ops {
    ($($Lhs:ty),* $(,)*) => {$(
        impl<const R: usize, const C: usize> ops::Mul<Matrix<$Lhs, R, C>> for $Lhs {
            type Output = Matrix<$Lhs, R, C>;

            #[inline]
            fn mul(self, other: Matrix<$Lhs, R, C>) -> Self::Output {
                // PERFORMANCE: The const loop should get unrolled during optimization.
                let mut result = Self::Output::zero();
                for c in 0..C {
                    for r in 0..R {
                        result[c][r] = self * other.data[c][r];
                    }
                }

                result
            }
        }

        impl<const R: usize, const C: usize> ops::Mul<&Matrix<$Lhs, R, C>> for $Lhs {
            type Output = Matrix<$Lhs, R, C>;

            #[inline]
            fn mul(self, other: &Matrix<$Lhs, R, C>) -> Self::Output {
                // PERFORMANCE: The const loop should get unrolled during optimization.
                let mut result = Self::Output::zero();
                for c in 0..C {
                    for r in 0..R {
                        result[c][r] = self * other.data[c][r];
                    }
                }

                result
            }
        }

        impl<'a, const R: usize, const C: usize> ops::Mul<Matrix<$Lhs, R, C>> for &'a $Lhs {
            type Output = Matrix<$Lhs, R, C>;

            #[inline]
            fn mul(self, other: Matrix<$Lhs, R, C>) -> Self::Output {
                // PERFORMANCE: The const loop should get unrolled during optimization.
                let mut result = Self::Output::zero();
                for c in 0..C {
                    for r in 0..R {
                        result[c][r] = self * other.data[c][r];
                    }
                }

                result
            }
        }

        impl<'a, 'b, const R: usize, const C: usize> ops::Mul<&'b Matrix<$Lhs, R, C>> for &'a $Lhs {
            type Output = Matrix<$Lhs, R, C>;

            #[inline]
            fn mul(self, other: &'b Matrix<$Lhs, R, C>) -> Self::Output {
                // PERFORMANCE: The const loop should get unrolled during optimization.
                let mut result = Self::Output::zero();
                for c in 0..C {
                    for r in 0..R {
                        result[c][r] = self * other.data[c][r];
                    }
                }

                result
            }
        }
    )*}
}

impl_scalar_matrix_mul_ops!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);


impl<S, const R: usize, const C: usize> ops::Mul<Vector<S, C>> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Vector<S, R>;

    #[inline]
    fn mul(self, other: Vector<S, C>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for r in 0..R {
            result[r] = dot_array_col(self.as_ref(), other.as_ref(), r);
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Mul<&Vector<S, C>> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Vector<S, R>;

    #[inline]
    fn mul(self, other: &Vector<S, C>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for r in 0..R {
            result[r] = dot_array_col(self.as_ref(), other.as_ref(), r);
        }

        result
    }
}

impl<S, const R: usize, const C: usize> ops::Mul<Vector<S, C>> for &Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Vector<S, R>;

    #[inline]
    fn mul(self, other: Vector<S, C>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for r in 0..R {
            result[r] = dot_array_col(self.as_ref(), other.as_ref(), r);
        }

        result
    }
}

impl<'a, 'b, S, const R: usize, const C: usize> ops::Mul<&'b Vector<S, C>> for &'a Matrix<S, R, C>
where
    S: SimdScalar,
{
    type Output = Vector<S, R>;

    #[inline]
    fn mul(self, other: &'b Vector<S, C>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for r in 0..R {
            result[r] = dot_array_col(self.as_ref(), other.as_ref(), r);
        }

        result
    }
}

impl<S, const R1: usize, const C1: usize, const R2: usize, const C2: usize, const R1C2: usize> ops::Mul<Matrix<S, R2, C2>>
    for Matrix<S, R1, C1>
where
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<R1>, Const<C1>, Const<R2>, Const<C2>>,
    ShapeConstraint: DimMul<Const<R1>, Const<C2>, Output = Const<R1C2>>,
    ShapeConstraint: DimMul<Const<C2>, Const<R1>, Output = Const<R1C2>>,
{
    type Output = Matrix<S, R1, C2>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: Matrix<S, R2, C2>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C2 {
            for r in 0..R1 {
                result[c][r] = dot_array_col(
                    self.as_ref(),
                    &<Matrix<S, R2, C2> as AsRef<[[S; R2]; C2]>>::as_ref(&other)[c],
                    r,
                );
            }
        }

        result
    }
}

impl<S, const R1: usize, const C1: usize, const R2: usize, const C2: usize, const R1C2: usize> ops::Mul<&Matrix<S, R2, C2>>
    for Matrix<S, R1, C1>
where
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<R1>, Const<C1>, Const<R2>, Const<C2>>,
    ShapeConstraint: DimMul<Const<R1>, Const<C2>, Output = Const<R1C2>>,
    ShapeConstraint: DimMul<Const<C2>, Const<R1>, Output = Const<R1C2>>,
{
    type Output = Matrix<S, R1, C2>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: &Matrix<S, R2, C2>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C2 {
            for r in 0..R1 {
                result[c][r] = dot_array_col(
                    self.as_ref(),
                    &<Matrix<S, R2, C2> as AsRef<[[S; R2]; C2]>>::as_ref(other)[c],
                    r,
                );
            }
        }

        result
    }
}

impl<S, const R1: usize, const C1: usize, const R2: usize, const C2: usize, const R1C2: usize> ops::Mul<Matrix<S, R2, C2>>
    for &Matrix<S, R1, C1>
where
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<R1>, Const<C1>, Const<R2>, Const<C2>>,
    ShapeConstraint: DimMul<Const<R1>, Const<C2>, Output = Const<R1C2>>,
    ShapeConstraint: DimMul<Const<C2>, Const<R1>, Output = Const<R1C2>>,
{
    type Output = Matrix<S, R1, C2>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: Matrix<S, R2, C2>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C2 {
            for r in 0..R1 {
                result[c][r] = dot_array_col(
                    self.as_ref(),
                    &<Matrix<S, R2, C2> as AsRef<[[S; R2]; C2]>>::as_ref(&other)[c],
                    r,
                );
            }
        }

        result
    }
}

impl<'a, 'b, S, const R1: usize, const C1: usize, const R2: usize, const C2: usize, const R1C2: usize> ops::Mul<&'b Matrix<S, R2, C2>>
    for &'a Matrix<S, R1, C1>
where
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<R1>, Const<C1>, Const<R2>, Const<C2>>,
    ShapeConstraint: DimMul<Const<R1>, Const<C2>, Output = Const<R1C2>>,
    ShapeConstraint: DimMul<Const<C2>, Const<R1>, Output = Const<R1C2>>,
{
    type Output = Matrix<S, R1, C2>;

    #[rustfmt::skip]
    #[inline]
    fn mul(self, other: &'b Matrix<S, R2, C2>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C2 {
            for r in 0..R1 {
                result[c][r] = dot_array_col(
                    self.as_ref(),
                    &<Matrix<S, R2, C2> as AsRef<[[S; R2]; C2]>>::as_ref(other)[c],
                    r,
                );
            }
        }

        result
    }
}


impl<S, const R: usize, const C: usize> ops::AddAssign<Matrix<S, R, C>> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    #[inline]
    fn add_assign(&mut self, other: Matrix<S, R, C>) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for c in 0..C {
            for r in 0..R {
                self.data[c][r] += other.data[c][r];
            }
        }
    }
}

impl<S, const R: usize, const C: usize> ops::AddAssign<&Matrix<S, R, C>> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    #[inline]
    fn add_assign(&mut self, other: &Matrix<S, R, C>) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for c in 0..C {
            for r in 0..R {
                self.data[c][r] += other.data[c][r];
            }
        }
    }
}

impl<S, const R: usize, const C: usize> ops::SubAssign<Matrix<S, R, C>> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    #[inline]
    fn sub_assign(&mut self, other: Matrix<S, R, C>) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for c in 0..C {
            for r in 0..R {
                self.data[c][r] -= other.data[c][r];
            }
        }
    }
}

impl<S, const R: usize, const C: usize> ops::SubAssign<&Matrix<S, R, C>> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    #[inline]
    fn sub_assign(&mut self, other: &Matrix<S, R, C>) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for c in 0..C {
            for r in 0..R {
                self.data[c][r] -= other.data[c][r];
            }
        }
    }
}

impl<S, const R: usize, const C: usize> ops::MulAssign<S> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    #[inline]
    fn mul_assign(&mut self, other: S) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for c in 0..C {
            for r in 0..R {
                self.data[c][r] *= other;
            }
        }
    }
}

impl<S, const R: usize, const C: usize> ops::DivAssign<S> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    #[inline]
    fn div_assign(&mut self, other: S) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for c in 0..C {
            for r in 0..R {
                self.data[c][r] /= other;
            }
        }
    }
}

impl<S, const R: usize, const C: usize> ops::RemAssign<S> for Matrix<S, R, C>
where
    S: SimdScalar,
{
    #[inline]
    fn rem_assign(&mut self, other: S) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for c in 0..C {
            for r in 0..R {
                self.data[c][r] %= other;
            }
        }
    }
}
/*
impl<S, const R: usize, const C: usize> approx::AbsDiffEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    type Epsilon = <S as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for c in 0..C {
            for r in 0..R {
                result &= S::abs_diff_eq(&self.data[c][r], &other.data[c][r], epsilon);
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> approx::RelativeEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for c in 0..C {
            for r in 0..R {
                result &= S::relative_eq(&self.data[c][r], &other.data[c][r], epsilon, max_relative);
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> approx::UlpsEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for c in 0..C {
            for r in 0..R {
                result &= S::ulps_eq(&self.data[c][r], &other.data[c][r], epsilon, max_ulps);
            }
        }

        result
    }
}
*/
impl<S, const R: usize, const C: usize> approx_cmp::AbsDiffEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    type Tolerance = Matrix<<S as approx_cmp::AbsDiffEq>::Tolerance, R, C>;

    #[inline]
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for c in 0..C {
            for r in 0..R {
                result &= S::abs_diff_eq(&self.data[c][r], &other.data[c][r], &max_abs_diff.data[c][r]);
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> approx_cmp::AbsDiffAllEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::AbsDiffAllEq>::AllTolerance;

    #[inline]
    fn abs_diff_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for c in 0..C {
            for r in 0..R {
                result &= S::abs_diff_all_eq(&self.data[c][r], &other.data[c][r], max_abs_diff);
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> approx_cmp::AssertAbsDiffEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = Matrix<<S as approx_cmp::AssertAbsDiffEq>::DebugAbsDiff, R, C>;
    type DebugTolerance = Matrix<<S as approx_cmp::AssertAbsDiffEq>::DebugTolerance, R, C>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let result = approx_cmp::AssertAbsDiffEq::debug_abs_diff(
            &self.data,
            &other.data,
        );

        Matrix::from(result)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let result = approx_cmp::AssertAbsDiffEq::debug_abs_diff_tolerance(
            &self.data,
            &other.data,
            &max_abs_diff.data,
        );

        Matrix::from(result)
    }
}

impl<S, const R: usize, const C: usize> approx_cmp::AssertAbsDiffAllEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = Matrix<<S as approx_cmp::AssertAbsDiffAllEq>::AllDebugTolerance, R, C>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let result = approx_cmp::AssertAbsDiffAllEq::debug_abs_diff_all_tolerance(
            &self.data,
            &other.data,
            max_abs_diff,
        );

        Matrix::from(result)
    }
}

impl<S, const R: usize, const C: usize> approx_cmp::RelativeEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    type Tolerance = Matrix<<S as approx_cmp::RelativeEq>::Tolerance, R, C>;

    #[inline]
    fn relative_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_relative: &Self::Tolerance) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for c in 0..C {
            for r in 0..R {
                result &= S::relative_eq(
                    &self.data[c][r],
                    &other.data[c][r],
                    &max_abs_diff.data[c][r],
                    &max_relative.data[c][r],
                );
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> approx_cmp::RelativeAllEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::RelativeAllEq>::AllTolerance;

    #[inline]
    fn relative_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_relative: &Self::AllTolerance) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for c in 0..C {
            for r in 0..R {
                result &= S::relative_all_eq(&self.data[c][r], &other.data[c][r], max_abs_diff, max_relative);
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> approx_cmp::AssertRelativeEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    type DebugAbsDiff = Matrix<<S as approx_cmp::AssertRelativeEq>::DebugAbsDiff, R, C>;
    type DebugTolerance = Matrix<<S as approx_cmp::AssertRelativeEq>::DebugTolerance, R, C>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let result = approx_cmp::AssertRelativeEq::debug_abs_diff(&self.data, &other.data);

        Matrix::from(result)
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let result = approx_cmp::AssertRelativeEq::debug_abs_diff_tolerance(
            &self.data,
            &other.data,
            &max_abs_diff.data,
        );

        Matrix::from(result)
    }

    #[inline]
    fn debug_relative_tolerance(&self, other: &Self, max_relative: &Self::Tolerance) -> Self::DebugTolerance {
        let result = approx_cmp::AssertRelativeEq::debug_relative_tolerance(
            &self.data,
            &other.data,
            &max_relative.data,
        );

        Matrix::from(result)
    }
}

impl<S, const R: usize, const C: usize> approx_cmp::AssertRelativeAllEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = Matrix<<S as approx_cmp::AssertRelativeAllEq>::AllDebugTolerance, R, C>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let result = approx_cmp::AssertRelativeAllEq::debug_abs_diff_all_tolerance(
            &self.data,
            &other.data,
            max_abs_diff,
        );

        Matrix::from(result)
    }

    #[inline]
    fn debug_relative_all_tolerance(&self, other: &Self, max_relative: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let result = approx_cmp::AssertRelativeAllEq::debug_relative_all_tolerance(
            &self.data,
            &other.data,
            max_relative,
        );

        Matrix::from(result)
    }
}

impl<S, const R: usize, const C: usize> approx_cmp::UlpsEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
    S::UlpsTolerance: Sized,
{
    type Tolerance = Matrix<<S as approx_cmp::UlpsEq>::Tolerance, R, C>;
    type UlpsTolerance = Matrix<<S as approx_cmp::UlpsEq>::UlpsTolerance, R, C>;

    #[inline]
    fn ulps_eq(&self, other: &Self, max_abs_diff: &Self::Tolerance, max_ulps: &Self::UlpsTolerance) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for c in 0..C {
            for r in 0..R {
                result &= S::ulps_eq(
                    &self.data[c][r],
                    &other.data[c][r],
                    &max_abs_diff.data[c][r],
                    &max_ulps.data[c][r],
                );
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> approx_cmp::UlpsAllEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    type AllTolerance = <S as approx_cmp::UlpsAllEq>::AllTolerance;
    type AllUlpsTolerance = <S as approx_cmp::UlpsAllEq>::AllUlpsTolerance;

    #[inline]
    fn ulps_all_eq(&self, other: &Self, max_abs_diff: &Self::AllTolerance, max_ulps: &Self::AllUlpsTolerance) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for c in 0..C {
            for r in 0..R {
                result &= S::ulps_all_eq(&self.data[c][r], &other.data[c][r], max_abs_diff, max_ulps);
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> approx_cmp::AssertUlpsEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
    S::UlpsTolerance: Sized,
{
    type DebugAbsDiff = Matrix<<S as approx_cmp::AssertUlpsEq>::DebugAbsDiff, R, C>;
    type DebugUlpsDiff = Matrix<<S as approx_cmp::AssertUlpsEq>::DebugUlpsDiff, R, C>;
    type DebugTolerance = Matrix<<S as approx_cmp::AssertUlpsEq>::DebugTolerance, R, C>;
    type DebugUlpsTolerance = Matrix<<S as approx_cmp::AssertUlpsEq>::DebugUlpsTolerance, R, C>;

    #[inline]
    fn debug_abs_diff(&self, other: &Self) -> Self::DebugAbsDiff {
        let data = approx_cmp::AssertUlpsEq::debug_abs_diff(&self.data, &other.data);

        Matrix::from(data)
    }

    #[inline]
    fn debug_ulps_diff(&self, other: &Self) -> Self::DebugUlpsDiff {
        let data = approx_cmp::AssertUlpsEq::debug_ulps_diff(&self.data, &other.data);

        Matrix { data, }
    }

    #[inline]
    fn debug_abs_diff_tolerance(&self, other: &Self, max_abs_diff: &Self::Tolerance) -> Self::DebugTolerance {
        let data = approx_cmp::AssertUlpsEq::debug_abs_diff_tolerance(
            &self.data,
            &other.data,
            &max_abs_diff.data,
        );

        Matrix { data, }
    }

    #[inline]
    fn debug_ulps_tolerance(&self, other: &Self, max_ulps: &Self::UlpsTolerance) -> Self::DebugUlpsTolerance {
        let data = approx_cmp::AssertUlpsEq::debug_ulps_tolerance(
            &self.data,
            &other.data,
            &max_ulps.data,
        );

        Matrix { data, }
    }
}

impl<S, const R: usize, const C: usize> approx_cmp::AssertUlpsAllEq for Matrix<S, R, C>
where
    S: SimdScalarFloat,
{
    type AllDebugTolerance = Matrix<<S as approx_cmp::AssertUlpsAllEq>::AllDebugTolerance, R, C>;
    type AllDebugUlpsTolerance = Matrix<<S as approx_cmp::AssertUlpsAllEq>::AllDebugUlpsTolerance, R, C>;

    #[inline]
    fn debug_abs_diff_all_tolerance(&self, other: &Self, max_abs_diff: &Self::AllTolerance) -> Self::AllDebugTolerance {
        let result = approx_cmp::AssertUlpsAllEq::debug_abs_diff_all_tolerance(
            &self.data,
            &other.data,
            max_abs_diff,
        );

        Matrix::from(result)
    }

    #[inline]
    fn debug_ulps_all_tolerance(&self, other: &Self, max_ulps: &Self::AllUlpsTolerance) -> Self::AllDebugUlpsTolerance {
        let data = approx_cmp::AssertUlpsAllEq::debug_ulps_all_tolerance(
            &self.data,
            &other.data,
            max_ulps,
        );

        Matrix { data, }
    }
}

impl<S, const R: usize, const C: usize> ops::Neg for Unit<Matrix<S, R, C>>
where
    S: SimdScalarFloat,
{
    type Output = Unit<Matrix<S, R, C>>;

    #[inline]
    fn neg(self) -> Self::Output {
        Unit::from_value_unchecked(-self.into_inner())
    }
}

impl<S, const R: usize, const C: usize> ops::Neg for &Unit<Matrix<S, R, C>>
where
    S: SimdScalarFloat,
{
    type Output = Unit<Matrix<S, R, C>>;

    #[inline]
    fn neg(self) -> Self::Output {
        Unit::from_value_unchecked(-self.into_inner())
    }
}
