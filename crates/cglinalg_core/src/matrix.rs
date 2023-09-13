use cglinalg_numeric::{
    SimdCast,
    SimdScalar,
    SimdScalarSigned,
    SimdScalarOrd,
    SimdScalarFloat,
};
use cglinalg_trigonometry::{
    Angle,
    Radians,
};
use crate::constraints::{
    Const,
    DimMul,
    DimEq,
    CanMultiply,
    ShapeConstraint,
};
use crate::normed::{
    Normed,
    Norm,
};
use crate::unit::{
    Unit,
};
use crate::point::{
    Point3,
};
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
use approx::{
    ulps_eq,
    ulps_ne,
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
    ShapeConstraint: DimEq<Const<C1>, Const<R2>> + DimEq<Const<R2>, Const<C1>>
{
    // PERFORMANCE: The const loop should get unrolled during optimization.
    let mut result = unsafe { core::mem::zeroed() };
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
    ShapeConstraint: DimMul<Const<C>, Const<R>, Output = Const<RC>>
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
        unsafe {
            &*(self as *const Matrix<S, R, C> as *const [[S; R]; C])
        }
    }
}

impl<S, const R: usize, const C: usize> AsMut<[[S; R]; C]> for Matrix<S, R, C> {
    #[inline]
    fn as_mut(&mut self) -> &mut [[S; R]; C] {
        unsafe {
            &mut *(self as *mut Matrix<S, R, C> as *mut [[S; R]; C])
        }
    }
}

impl<S, const R: usize, const C: usize> AsRef<[Vector<S, R>; C]> for Matrix<S, R, C> {
    #[inline]
    fn as_ref(&self) -> &[Vector<S, R>; C] {
        unsafe {
            &*(self as *const Matrix<S, R, C> as *const [Vector<S, R>; C])
        }
    }
}

impl<S, const R: usize, const C: usize> AsMut<[Vector<S, R>; C]> for Matrix<S, R, C> {
    #[inline]
    fn as_mut(&mut self) -> &mut [Vector<S, R>; C] {
        unsafe {
            &mut *(self as *mut Matrix<S, R, C> as *mut [Vector<S, R>; C])
        }
    }
}

impl<S, const R: usize, const C: usize, const RC: usize> AsRef<[S; RC]> for Matrix<S, R, C> 
where
    ShapeConstraint: DimMul<Const<R>, Const<C>, Output = Const<RC>>,
    ShapeConstraint: DimMul<Const<C>, Const<R>, Output = Const<RC>>
{
    #[inline]
    fn as_ref(&self) -> &[S; RC] {
        unsafe {
            &*(self as *const Matrix<S, R, C> as *const [S; RC])
        }
    }
}

impl<S, const R: usize, const C: usize, const RC: usize> AsMut<[S; RC]> for Matrix<S, R, C> 
where
    ShapeConstraint: DimMul<Const<R>, Const<C>, Output = Const<RC>>,
    ShapeConstraint: DimMul<Const<C>, Const<R>, Output = Const<RC>>
{
    #[inline]
    fn as_mut(&mut self) -> &mut [S; RC] {
        unsafe {
            &mut *(self as *mut Matrix<S, R, C> as *mut [S; RC])
        }
    }
}

impl<S, const R: usize, const C: usize> ops::Index<usize> for Matrix<S, R, C> 
{
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
    S: Copy
{
    #[inline]
    fn from(data: [[S; R]; C]) -> Self {
        Self { data }
    }
}

impl<'a, S, const R: usize, const C: usize> From<&'a [[S; R]; C]> for &'a Matrix<S, R, C>
where
    S: Copy
{
    #[inline]
    fn from(data: &'a [[S; R]; C]) -> &'a Matrix<S, R, C> {
        unsafe { 
            &*(data as *const [[S; R]; C] as *const Matrix<S, R, C>)
        }
    }    
}

impl<'a, S, const R: usize, const C: usize> From<&'a [Vector<S, R>; C]> for &'a Matrix<S, R, C>
where
    S: Copy
{
    #[inline]
    fn from(data: &'a [Vector<S, R>; C]) -> &'a Matrix<S, R, C> {
        unsafe { 
            &*(data as *const [Vector<S, R>; C] as *const Matrix<S, R, C>)
        }
    }  
}

impl<S, const R: usize, const C: usize, const RC: usize> From<[S; RC]> for Matrix<S, R, C> 
where
    S: Copy,
    ShapeConstraint: DimMul<Const<R>, Const<C>, Output = Const<RC>>,
    ShapeConstraint: DimMul<Const<C>, Const<R>, Output = Const<RC>>
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
    ShapeConstraint: DimMul<Const<C>, Const<R>, Output = Const<RC>>
{
    #[inline]
    fn from(array: &'a [S; RC]) -> &'a Matrix<S, R, C> {
        unsafe { 
            &*(array as *const [S; RC] as *const Matrix<S, R, C>)
        }
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
    S: Copy
{
    /// Construct a new matrix from a fill value.
    ///
    /// The resulting matrix is a matrix where each entry is the supplied fill
    /// value.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    pub const fn from_fill(value: S) -> Self {
        Self {
            data: [[value; R]; C],
        }
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
        let data_ptr = unsafe {
            &*(columns as *const [Vector<S, R>; C] as *const [[S; R]; C])
        };
    
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
    /// # use cglinalg_core::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     1_u32,  2_u32,  3_u32,  4_u32, 
    ///     5_u32,  6_u32,  7_u32,  8_u32,
    ///     9_u32,  10_u32, 11_u32, 12_u32,
    ///     13_u32, 14_u32, 15_u32, 16_u32
    /// );
    /// let expected = Matrix4x4::new(
    ///     2_i32,  4_i32,  6_i32,  8_i32,
    ///     10_i32, 12_i32, 14_i32, 16_i32,
    ///     18_i32, 20_i32, 22_i32, 24_i32,
    ///     26_i32, 28_i32, 30_i32, 32_i32
    /// );
    /// let result = matrix.map(|comp| (2 * comp) as i32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[allow(clippy::needless_range_loop)]
    #[inline]
    pub fn map<T, F>(&self, mut op: F) -> Matrix<T, R, C> 
    where 
        F: FnMut(S) -> T
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
    ///     7_i32, 8_i32, 9_i32
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
    ///     7_i32, 8_i32, 9_i32
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3, 
    /// # };
    /// #
    /// let mut result = Matrix3x3::new(
    ///     1_i32, 2_i32, 3_i32,
    ///     1_i32, 2_i32, 3_i32,
    ///     1_i32, 2_i32, 3_i32
    /// );
    /// let expected = Matrix3x3::new(
    ///     3_i32, 2_i32, 1_i32,
    ///     3_i32, 2_i32, 1_i32,
    ///     3_i32, 2_i32, 1_i32
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
    /// # use cglinalg_core::{
    /// #     Matrix3x4, 
    /// # };
    /// #
    /// let mut result = Matrix3x4::new(
    ///     1_i32, 1_i32, 1_i32,
    ///     2_i32, 2_i32, 2_i32,
    ///     3_i32, 3_i32, 3_i32,
    ///     4_i32, 4_i32, 4_i32
    ///
    /// );
    /// let expected = Matrix3x4::new(
    ///     2_i32, 2_i32, 2_i32,
    ///     4_i32, 4_i32, 4_i32,
    ///     3_i32, 3_i32, 3_i32,
    ///     1_i32, 1_i32, 1_i32
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
    /// # use cglinalg_core::{
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
    ///     1_i32, 2_i32,  3_i32,  13_i32, 
    ///     5_i32, 6_i32,  7_i32,  8_i32,
    ///     9_i32, 10_i32, 11_i32, 12_i32,
    ///     4_i32, 14_i32, 15_i32, 16_i32
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
    S: SimdCast + Copy 
{
    /// Cast a matrix from one type of scalars to another type of scalars.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x2,   
    /// # };
    /// # 
    /// let matrix: Matrix2x2<u32> = Matrix2x2::new(1_u32, 2_u32, 3_u32, 4_u32);
    /// let expected: Option<Matrix2x2<i32>> = Some(Matrix2x2::new(1_i32, 2_i32, 3_i32, 4_i32));
    /// let result = matrix.try_cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[allow(clippy::needless_range_loop)]
    #[inline]
    pub fn try_cast<T>(&self) -> Option<Matrix<T, R, C>> 
    where
        T: SimdCast
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
    S: SimdScalar
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
    ///     4_i32, 4_i32
    /// );
    /// let expected = Matrix4x2::new(
    ///     1_i32, 2_i32, 3_i32, 4_i32, 
    ///     1_i32, 2_i32, 3_i32, 4_i32
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
    /// # use cglinalg_core::{
    /// #     Matrix4x4, 
    /// # };
    /// #
    /// let matrix: Matrix4x4<i32> = Matrix4x4::zero();
    ///
    /// assert!(matrix.is_zero());
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self { 
            data: [[S::zero(); R]; C],
        }
    }

    /// Determine whether a matrix is a zero matrix.
    ///
    /// A zero matrix is a matrix in which all of its elements are zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4, 
    /// # };
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// # };
    /// #
    /// let m1 = Matrix3x3::new(
    ///     0_f64, 1_f64, 2_f64,
    ///     3_f64, 4_f64, 5_f64,
    ///     6_f64, 7_f64, 8_f64
    /// );
    /// let m2 = Matrix3x3::new(
    ///     9_f64,  10_f64, 11_f64,
    ///     12_f64, 13_f64, 14_f64,
    ///     15_f64, 16_f64, 17_f64
    /// );
    /// let expected = Matrix3x3::new(
    ///     0_f64,  10_f64,  22_f64,
    ///     36_f64, 52_f64,  70_f64,
    ///     90_f64, 112_f64, 136_f64
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// # };
    /// #
    /// let mut result = Matrix3x3::new(
    ///     0_f64, 1_f64, 2_f64,
    ///     3_f64, 4_f64, 5_f64,
    ///     6_f64, 7_f64, 8_f64
    /// );
    /// let other = Matrix3x3::new(
    ///     9_f64,  10_f64, 11_f64,
    ///     12_f64, 13_f64, 14_f64,
    ///     15_f64, 16_f64, 17_f64
    /// );
    /// let expected = Matrix3x3::new(
    ///     0_f64,  10_f64,  22_f64,
    ///     36_f64, 52_f64,  70_f64,
    ///     90_f64, 112_f64, 136_f64
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

    /// Compute the dot product of two matrices.
    /// 
    /// # Example
    /// 
    /// An example involving integer scalars.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x3,
    /// # };
    /// #
    /// let matrix1 = Matrix2x3::new(
    ///     2_i32,  1_i32,
    ///     0_i32, -1_i32, 
    ///     6_i32,  2_i32,
    /// );
    /// let matrix2 = Matrix2x3::new(
    ///      8_i32,  4_i32,
    ///     -3_i32,  1_i32, 
    ///      2_i32, -5_i32
    /// );
    /// let expected = 21_i32;
    /// let result = Matrix2x3::dot(&matrix1, &matrix2);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn dot(&self, other: &Self) -> S {
        let mut result = S::zero();
        for r in 0..R {
            for c in 0..C {
                result += self.data[c][r] * other.data[c][r]
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> Matrix<S, R, C>
where
    S: SimdScalarSigned
{
    /// Mutably negate the elements of a matrix in place.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    S: SimdScalarFloat
{
    /// Linearly interpolate between two matrices.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,    
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
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
    /// let amount = 0.5_f64;
    /// let expected = Matrix3x3::new(
    ///     1.5_f64, 1.5_f64, 1.5_f64, 
    ///     2.5_f64, 2.5_f64, 2.5_f64,
    ///     3.5_f64, 3.5_f64, 3.5_f64
    /// );
    /// let result = matrix0.lerp(&matrix1, amount);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
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
    /// # use cglinalg_core::{
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
    /// # Example (Not A Finite Matrix)
    ///
    /// ```
    /// # use cglinalg_core::{
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
    S: SimdScalar
{
    /// Mutably transpose a square matrix in place.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    /// # use cglinalg_core::{
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
    /// # use cglinalg_core::{
    /// #     Matrix4x4, 
    /// # };
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
    /// # use cglinalg_core::{
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
    ///     0_i32, 0_i32, 0_i32, 5_i32 
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
    ///     13_i32, 14_i32, 15_i32, 16_i32
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
    /// # use cglinalg_core::{
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
    S: SimdScalarFloat
{
    /// Determine whether a square matrix is a diagonal matrix. 
    ///
    /// A square matrix is a diagonal matrix if every off-diagonal 
    /// element is zero.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// # };
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
                result &= ulps_eq!(self.data[i][j], S::zero()) && ulps_eq!(self.data[j][i], S::zero());
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// # };
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
                result &= ulps_eq!(self.data[i][j], self.data[j][i]);
            }
        }

        result
    }
}

impl<S, const R: usize, const C: usize> Default for Matrix<S, R, C>
where
    S: SimdScalar
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<S, const R: usize, const C: usize> fmt::Display for Matrix<S, R, C>
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Matrix{}x{} [", R, C).unwrap();
        for c in 0..(C - 1) {
            write!(formatter, "[").unwrap();
            for r in 0..(R - 1) {
                write!(formatter, "{}, ", self.data[c][r]).unwrap();
            }
            write!(formatter, "{}]", self.data[c][R - 1]).unwrap();
        }
        write!(formatter, "{}]", self.data[C - 1][R - 1])
    }
}


impl<S, const R: usize, const C: usize> Matrix<S, R, C>
where
    S: SimdScalarSigned
{
    /// Calculate the norm of a matrix with respect to the supplied [`Norm`] type.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     L1MatrixNorm,
    /// #     LinfMatrixNorm,
    /// #     FrobeniusNorm,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     1_f64, 2_f64, 3_f64,
    ///     5_f64, 6_f64, 7_f64,
    ///     8_f64, 9_f64, 10_f64
    /// );
    /// let l1_norm = L1MatrixNorm::new();
    /// let linf_norm = LinfMatrixNorm::new();
    /// let frobenius_norm = FrobeniusNorm::new();
    /// 
    /// assert_eq!(matrix.apply_norm(&l1_norm), 27_f64);
    /// assert_eq!(matrix.apply_norm(&linf_norm), 20_f64);
    /// assert_relative_eq!(matrix.apply_norm(&frobenius_norm), 19.209372712298546, epsilon = 1e-10);
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     L1MatrixNorm,
    /// #     LinfMatrixNorm,
    /// #     FrobeniusNorm,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let matrix1 = Matrix3x3::new(
    ///     1_f64, 2_f64, 3_f64,
    ///     5_f64, 6_f64, 7_f64,
    ///     8_f64, 9_f64, 10_f64
    /// );
    /// let matrix2 = Matrix3x3::new(
    ///     0_f64, -5_f64,  6_f64,
    ///    -3_f64,  1_f64,  20_f64,
    ///     7_f64,  12_f64, 4_f64
    /// );
    /// let l1_norm = L1MatrixNorm::new();
    /// let linf_norm = LinfMatrixNorm::new();
    /// let frobenius_norm = FrobeniusNorm::new();
    /// 
    /// assert_eq!(matrix1.apply_metric_distance(&matrix2, &l1_norm), 26_f64);
    /// assert_eq!(matrix1.apply_metric_distance(&matrix2, &linf_norm), 22_f64);
    /// assert_relative_eq!(matrix1.apply_metric_distance(&matrix2, &frobenius_norm), 19.05255888325765, epsilon = 1e-10);
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
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     -7_i32,  1_i32, 5_i32, 0_i32,
    ///      1_i32, -5_i32, 8_i32, 2_i32,
    ///      5_i32,  6_i32, 3_i32, 6_i32,
    ///      0_i32,  4_i32, 0_i32, 15_i32
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
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     -7_i32,  1_i32, 5_i32, 0_i32,
    ///      1_i32, -5_i32, 8_i32, 2_i32,
    ///      5_i32,  6_i32, 3_i32, 6_i32,
    ///      0_i32,  4_i32, 0_i32, 15_i32
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3
    /// # };
    /// #
    /// let matrix1 = Matrix3x3::new(
    ///     -7_i32,  5_i32, -9_i32,
    ///      1_i32, -5_i32,  8_i32,
    ///      5_i32,  6_i32,  3_i32
    /// );
    /// let matrix2 = Matrix3x3::new(
    ///     2_i32, 6_i32, 1_i32,
    ///     1_i32, 2_i32, 8_i32,
    ///     3_i32, 1_i32, 3_i32
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
    S: SimdScalarSigned + SimdScalarOrd
{
    /// Compute the **L1** norm of a matrix.
    /// 
    /// The matrix **L1** norm is also called the **maximum column sum norm**.
    /// 
    /// # Example
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// # };
    /// #
    /// let matrix = Matrix3x3::new(
    ///     -3_i32, 2_i32, 0_i32,
    ///      5_i32, 6_i32, 2_i32,
    ///      7_i32, 4_i32, 8_i32
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
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     -7_i32,  1_i32, 5_i32, 0_i32,
    ///      1_i32, -5_i32, 8_i32, 2_i32,
    ///      5_i32,  6_i32, 3_i32, 6_i32,
    ///      0_i32,  4_i32, 0_i32, 15_i32
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
    S: SimdScalarFloat
{
    /// Compute the **Frobenius** norm of a matrix.
    /// 
    /// The squared **Frobenius** norm of a matrix is the sum of the squares of all 
    /// the elements of the matrix.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     -7_f64,  1_f64, 5_f64, 0_f64,
    ///      1_f64, -5_f64, 8_f64, 2_f64,
    ///      5_f64,  6_f64, 3_f64, 6_f64,
    ///      0_f64,  4_f64, 0_f64, 15_f64
    /// );
    /// let expected = 22.715633383201094;
    /// let result = matrix.norm();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
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
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let matrix = Matrix4x4::new(
    ///     -7_f64,  1_f64, 5_f64, 0_f64,
    ///      1_f64, -5_f64, 8_f64, 2_f64,
    ///      5_f64,  6_f64, 3_f64, 6_f64,
    ///      0_f64,  4_f64, 0_f64, 15_f64
    /// );
    /// let expected = 22.715633383201094;
    /// let result = matrix.magnitude();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let matrix1 = Matrix3x3::new(
    ///     -7_f64,  5_f64, -9_f64,
    ///      1_f64, -5_f64,  8_f64,
    ///      5_f64,  6_f64,  3_f64
    /// );
    /// let matrix2 = Matrix3x3::new(
    ///     2_f64, 6_f64, 1_f64,
    ///     1_f64, 2_f64, 8_f64,
    ///     3_f64, 1_f64, 3_f64
    /// );
    /// let expected = 16.124515496597099;
    /// let result = matrix1.metric_distance(&matrix2);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
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
    S: SimdScalarSigned + SimdScalarOrd
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
    S: SimdScalarFloat
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
    S: SimdScalarSigned + SimdScalarOrd
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
    S: SimdScalarFloat
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


impl<S> Matrix1x1<S> {
    /// Construct a new matrix from its elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix1x1,
    /// # };
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
    S: SimdScalarSigned 
{
    /// Compute the determinant of a matrix.
    /// 
    /// The determinant of a matrix is the signed volume of the parallelepiped
    /// swept out by the vectors represented by the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix1x1, 
    /// # };
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
    S: SimdScalarFloat
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
    /// # use cglinalg_core::{
    /// #     Matrix1x1,  
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let matrix = Matrix1x1::new(5_f64);
    /// let expected = Matrix1x1::new(1_f64 / 5_f64);
    /// let result = matrix.inverse().unwrap();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
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
    /// # use cglinalg_core::{
    /// #     Matrix1x1, 
    /// # };
    /// #
    /// let matrix = Matrix1x1::new(-2_f64);
    /// 
    /// assert_eq!(matrix.determinant(), -2_f64);
    /// assert!(matrix.is_invertible());
    /// ```
    #[inline]
    pub fn is_invertible(&self) -> bool {
        ulps_ne!(self.determinant(), S::zero())
    }
}


impl<S> Matrix2x2<S> {
    /// Construct a new matrix from its elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// # };
    /// #
    /// let c0r0 = 1_i32; let c0r1 = 2_i32;
    /// let c1r0 = 2_i32; let c1r1 = 3_i32;
    /// let matrix = Matrix2x2::new(
    ///     c0r0, c0r1,
    ///     c1r0, c1r1
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
            ]
        }
    }
}

impl<S> Matrix2x2<S> 
where 
    S: SimdScalar 
{
    /// Construct a shearing matrix along the x-axis, holding the **y-axis** constant.
    ///
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the **y-axis** to shearing along the **x-axis**.
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_shear_x(shear_x_with_y: S) -> Self {
        Self::new(
            S::one(),       S::zero(),
            shear_x_with_y, S::one(),
        )
    }

    /// Construct a shearing matrix along the y-axis, holding the **x-axis** constant.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the **x-axis** to shearing along the **y-axis**.
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_shear_y(shear_y_with_x: S) -> Self {
        Self::new(
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
    /// contribution of the **y-component** to the shearing of the **x-component**. 
    ///
    /// # Example 
    /// 
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_shear(shear_x_with_y: S, shear_y_with_x: S) -> Self {
        let one = S::one();

        Self::new(
            one,            shear_y_with_x,
            shear_x_with_y, one
        )
    }

    /// Construct a two-dimensional uniform scaling matrix.
    ///
    /// The matrix applies the same scale factor to all dimensions, so each
    /// component of a vector will be scaled by the same factor. In particular,
    /// calling `from_scale(scale)` is equivalent to calling 
    /// `Self::from_nonuniform_scale(scale, scale)`.
    ///
    /// # Example 
    /// 
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_scale(scale: S) -> Self {
        Self::from_nonuniform_scale(scale, scale)
    }
        
    /// Construct two-dimensional general scaling matrix.
    ///
    /// This is the most general case for scaling matrices: the scale factor
    /// in each dimension need not be identical.
    ///
    /// # Example 
    /// 
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S) -> Self {
        let zero = S::zero();

        Self::new(
            scale_x,   zero,
            zero,      scale_y,
        )
    }
}

impl<S> Matrix2x2<S> 
where 
    S: SimdScalarSigned 
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
    /// # use cglinalg_core::{
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
    pub fn from_reflection(normal: &Unit<Vector2<S>>) -> Self {
        let one = S::one();
        let two = one + one;

        Self::new(
             one - two * normal.x * normal.x, -two * normal.x * normal.y,
            -two * normal.x * normal.y,        one - two * normal.y * normal.y,
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
    /// # use cglinalg_core::{
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

impl<S> Matrix2x2<S> 
where 
    S: SimdScalarFloat
{
    /// Construct a rotation matrix in two-dimensions that rotates a vector
    /// in the **xy-plane** by an angle `angle`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Vector2, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let unit_x = Vector2::unit_x();
    /// let unit_y = Vector2::unit_y();
    /// let matrix = Matrix2x2::from_angle(angle);
    /// let expected = unit_y;
    /// let result = matrix * unit_x;
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_angle<A: Into<Radians<S>>>(angle: A) -> Self {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Self::new(
             cos_angle, sin_angle, 
            -sin_angle, cos_angle
        )
    }

    /// Construct a rotation matrix that rotates the shortest angular distance 
    /// between two vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Vector2, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let v1 = Vector2::new(1_f64, 1_f64);
    /// let v2 = Vector2::new(-1_f64, 1_f64);
    /// let matrix = Matrix2x2::rotation_between(&v1, &v2);
    /// let vector = Vector2::unit_y();
    /// let expected = -Vector2::unit_x();
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    /// The matrix returned by `rotation_between` should make `v1` and `v2` collinear.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Vector2, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let v1 = Vector2::new(1_f64, 1_f64);
    /// let v2 = Vector2::new(-1_f64, 1_f64);
    /// let matrix = Matrix2x2::rotation_between(&v1, &v2);
    /// let result = matrix * v1;
    /// let expected = v2;
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn rotation_between(v1: &Vector2<S>, v2: &Vector2<S>) -> Self {
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    /// The matrix returned by `rotation_between` should make `v1` and `v2` collinear.
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x2,
    /// #     Vector2,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
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
    /// # use cglinalg_core::{
    /// #     Matrix2x2,  
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
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
                det_inv * -self.data[1][0], det_inv *  self.data[0][0]
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
    /// # use cglinalg_core::{
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
}

impl<S> Matrix3x3<S> {
    /// Construct a new matrix from its elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// # };
    /// #
    /// let c0r0 = 1_i32; let c0r1 = 2_i32; let c0r2 = 3_i32;
    /// let c1r0 = 4_i32; let c1r1 = 5_i32; let c1r2 = 6_i32;
    /// let c2r0 = 7_i32; let c2r1 = 8_i32; let c2r2 = 9_i32;
    /// let matrix = Matrix3x3::new(
    ///     c0r0, c0r1, c0r2,
    ///     c1r0, c1r1, c1r2,
    ///     c2r0, c2r1, c2r2
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
            ]
        }
    }
}

impl<S> Matrix3x3<S> 
where 
    S: SimdScalar
{
    /// Construct a two-dimensional affine translation matrix.
    ///
    /// This represents a translation in the **xy-plane** as an affine 
    /// transformation that displaces a vector along the length of the vector
    /// `distance`.
    ///
    /// # Example
    /// A homogeneous vector with a zero **z-component** should not translate.
    /// ```
    /// # use cglinalg_core::{
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
    /// A homogeneous vector with a unit **z-component** should translate.
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_affine_translation(distance: &Vector2<S>) -> Self {
        let one = S::one();
        let zero = S::zero();
        
        Self::new(
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
    /// `Self::from_nonuniform_scale(scale, scale, scale)`.
    ///
    /// # Example
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
        Self::from_nonuniform_scale(scale, scale, scale)
    }
    
    /// Construct a three-dimensional general scaling matrix.
    ///
    /// This is the most general case for scaling matrices: the scale factor
    /// in each dimension need not be identical.
    ///
    /// # Example
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
    /// let vector = Vector3::new(1_i32, 1_i32, 1_i32);
    /// let matrix = Matrix3x3::from_nonuniform_scale(scale_x, scale_y, scale_z);
    /// let expected = Vector3::new(5_i32, 10_i32, 15_i32);
    /// let result = matrix * vector;
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_nonuniform_scale(scale_x: S, scale_y: S, scale_z: S) -> Self {
        let zero = S::zero();

        Self::new(
            scale_x,   zero,      zero,
            zero,      scale_y,   zero,
            zero,      zero,      scale_z,
        )
    }

    /// Construct a two-dimensional uniform affine scaling matrix.
    ///
    /// The matrix applies the same scale factor to all dimensions, so each
    /// component of a vector will be scaled by the same factor. In particular,
    /// calling `Self::from_scale(scale)` is equivalent to calling 
    /// `Self::from_affine_nonuniform_scale(scale, scale)`. The **z-component** is 
    /// unaffected since this is an affine matrix.
    ///
    /// # Example
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
    #[inline]
    pub fn from_affine_scale(scale: S) -> Self {
        Self::from_affine_nonuniform_scale(scale, scale)
    }
    
    /// Construct a two-dimensional affine scaling matrix.
    ///
    /// This is the most general case for scaling matrices: the scale factor
    /// in each dimension need not be identical. The **z-component** is unaffected 
    /// because this is an affine matrix.
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_affine_nonuniform_scale(scale_x: S, scale_y: S) -> Self {
        let zero = S::zero();
        let one = S::one();

        Self::new(
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_shear_x(shear_x_with_y: S, shear_x_with_z: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_shear_y(shear_y_with_x: S, shear_y_with_z: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_shear_z(shear_z_with_x: S, shear_z_with_y: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,  zero, shear_z_with_x,
            zero, one,  shear_z_with_y,
            zero, zero, one   
        )
    }

    /// Construct a general shearing matrix in three dimensions. There are six
    /// parameters describing a shearing transformation in three dimensions.
    /// 
    /// The parameter `shear_x_with_y` denotes the factor scaling the
    /// contribution of the **y-component** to shearing of the **x-component**.
    ///
    /// The parameter `shear_x_with_z` denotes the factor scaling the 
    /// contribution  of the **z-component** to the shearing of the **x-component**.
    ///
    /// The parameter `shear_y_with_x` denotes the factor scaling the
    /// contribution of the **x-component** to shearing of the **y-component**.
    ///
    /// The parameter `shear_y_with_z` denotes the factor scaling the 
    /// contribution of the **z-axis** to the shearing of the **y-component**. 
    ///
    /// The parameter `shear_z_with_x` denotes the factor scaling the
    /// contribution of the **x-axis** to shearing of the **z-axis**.
    ///
    /// The parameter `shear_z_with_y` denotes the factor scaling the 
    /// contribution of the **y-component** to the shearing of the **z-component**. 
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
        shear_z_with_x: S, shear_z_with_y: S) -> Self 
    {
        let one = S::one();

        Self::new(
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
    /// # Example 
    /// 
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_affine_shear_x(shear_x_with_y: S) -> Self {
        let zero = S::zero();
        let one = S::one();

        Self::new(
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
    /// # Example 
    /// 
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_affine_shear_y(shear_y_with_x: S) -> Self {
        let zero = S::zero();
        let one = S::one();

        Self::new(
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
    /// # Example 
    /// 
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_affine_shear(shear_x_with_y: S, shear_y_with_x: S) -> Self {
        let zero = S::zero();
        let one = S::one();

        Self::new(
            one,            shear_y_with_x, zero,
            shear_x_with_y, one,            zero,
            zero,           zero,           one
        )
    }
}

impl<S> Matrix3x3<S> 
where 
    S: SimdScalarSigned
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
    /// In particular, consider a line of the form
    /// ```text
    /// L := { (x, y) | a * (x - x0) + b * (y - y0) == 0 } 
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
    /// # Example (Line Through The Origin)
    ///
    /// Here is an example of reflecting a vector across the **x-axis** with 
    /// the line of reflection passing through the origin.
    /// ```
    /// # use cglinalg_core::{
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
    /// # use cglinalg_core::{
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
    /// # Example (Line That Does Not Cross The Origin)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Vector2, 
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
    /// # };
    /// #
    /// let bias = Vector2::new(0_f64, 2_f64);
    /// let normal = Unit::from_value(
    ///     Vector2::new(-1_f64 / f64::sqrt(5_f64), 2_f64 / f64::sqrt(5_f64))
    /// );
    /// let matrix = Matrix3x3::from_affine_reflection(&normal, &bias);
    /// let vector = Vector3::new(1_f64, 0_f64, 1_f64);
    /// let expected = Vector3::new(-1_f64, 4_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_reflection(normal: &Unit<Vector2<S>>, bias: &Vector2<S>) -> Self {
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

        Self::new(
            c0r0, c0r1, c0r2,
            c1r0, c1r1, c1r2,
            c2r0, c2r1, c2r2
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
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// #
    /// let normal = Unit::from_value(Vector3::unit_z());
    /// let expected = Matrix3x3::new(
    ///     1_f64, 0_f64,  0_f64, 
    ///     0_f64, 1_f64,  0_f64,  
    ///     0_f64, 0_f64, -1_f64
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

        let c0r0 =  one - two * normal.x * normal.x;
        let c0r1 = -two * normal.x * normal.y;
        let c0r2 = -two * normal.x * normal.z;

        let c1r0 = -two * normal.x * normal.y;
        let c1r1 =  one - two * normal.y * normal.y;
        let c1r2 = -two * normal.y * normal.z;

        let c2r0 = -two * normal.x * normal.z;
        let c2r1 = -two * normal.y * normal.z;
        let c2r2 =  one - two * normal.z * normal.z;
    
        Self::new(
            c0r0, c0r1, c0r2,
            c1r0, c1r1, c1r2,
            c2r0, c2r1, c2r2
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
    /// # use cglinalg_core::{
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

    /// Compute the cross product matrix for a given vector.
    /// 
    /// The cross matrix for a vector `a` is the matrix `A` such that for any
    /// vector `v`, `A` satisfies
    /// ```text
    /// A * v == cross(a, v)
    /// ```
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
             vector[1], -vector[0],  S::zero()
        )
    }
}

impl<S> Matrix3x3<S> 
where 
    S: SimdScalarFloat
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
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_affine_angle(angle);
    /// let unit_x = Vector3::unit_x();
    /// let expected = Vector3::unit_y();
    /// let result = matrix * unit_x;
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_angle<A: Into<Radians<S>>>(angle: A) -> Self {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());
        let zero = S::zero();
        let one =  S::one();

        Self::new(
             cos_angle, sin_angle, zero,
            -sin_angle, cos_angle, zero,
             zero,      zero,      one
        )
    }

    /// Construct a rotation matrix about the **x-axis** by an angle `angle`.
    ///
    /// # Example
    /// 
    /// In this example the rotation is in the **yz-plane**.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_angle_x(angle);
    /// let vector = Vector3::new(0_f64, 1_f64, 1_f64);
    /// let expected = Vector3::new(0_f64, -1_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_angle_x<A: Into<Radians<S>>>(angle: A) -> Self {
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
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_angle_y(angle);
    /// let vector = Vector3::new(1_f64, 0_f64, 1_f64);
    /// let expected = Vector3::new(1_f64, 0_f64, -1_f64);
    /// let result = matrix * vector;
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_angle_y<A: Into<Radians<S>>>(angle: A) -> Self {
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
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_angle_z(angle);
    /// let vector = Vector3::new(1_f64, 1_f64, 0_f64);
    /// let expected = Vector3::new(-1_f64, 1_f64, 0_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_angle_z<A: Into<Radians<S>>>(angle: A) -> Self {
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
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix3x3::from_axis_angle(&axis, angle);
    /// let unit_x = Vector3::unit_x();
    /// let expected = Vector3::unit_y();
    /// let result = matrix * unit_x;
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_axis_angle<A: Into<Radians<S>>>(axis: &Unit<Vector3<S>>, angle: A) -> Self {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());
        let one_minus_cos_angle = S::one() - cos_angle;
        let _axis = axis.as_ref();

        Self::new(
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result * direction.normalize(), unit_z, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn look_to_lh(direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let z_axis = direction.normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        Self::new(
            x_axis.x, y_axis.x, z_axis.x,
            x_axis.y, y_axis.y, z_axis.y,
            x_axis.z, y_axis.z, z_axis.z,
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result * direction.normalize(), minus_unit_z, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn look_to_rh(direction: &Vector3<S>, up: &Vector3<S>) -> Self {
        let z_axis = -direction.normalize();
        let x_axis = up.cross(&z_axis).normalize();
        let y_axis = z_axis.cross(&x_axis).normalize();

        Self::new(
            x_axis.x, y_axis.x, z_axis.x,
            x_axis.y, y_axis.y, z_axis.y,
            x_axis.z, y_axis.z, z_axis.z,
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result * direction, unit_z, epsilon = 1e-10);
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result * direction, minus_unit_z, epsilon = 1e-10);
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result * unit_z, direction.normalize(), epsilon = 1e-10);
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,    
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result * minus_unit_z, direction.normalize(), epsilon = 1e-10);
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result * unit_z, direction, epsilon = 1e-10);
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
    /// # };
    /// # use core::f64;
    /// #
    /// let eye = Point3::new(-1_f64, -1_f64, 0_f64);
    /// let target = Point3::origin();
    /// let up = Vector3::unit_z();
    /// let expected = Matrix3x3::new(
    ///      1_f64 / f64::sqrt(2_f64), -1_f64 / f64::sqrt(2_f64), 0_f64,
    ///      0_f64,                     0_f64,                    1_f64,
    ///     -1_f64 / f64::sqrt(2_f64), -1_f64 / f64::sqrt(2_f64), 0_f64
    /// );
    /// let result = Matrix3x3::look_at_rh_inv(&eye, &target, &up);
    /// let direction = (target - eye).normalize();
    /// let minus_unit_z = -Vector3::unit_z();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result * minus_unit_z, direction, epsilon = 1e-10);
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
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,   
    /// # };
    /// #
    /// let v1: Vector3<f64> = Vector3::unit_x() * 2_f64;
    /// let v2: Vector3<f64> = Vector3::unit_y() * 3_f64;
    /// let matrix = Matrix3x3::rotation_between(&v1, &v2).unwrap();
    /// let expected = Vector3::new(0_f64, 2_f64, 0_f64);
    /// let result = matrix * v1;
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn rotation_between(v1: &Vector3<S>, v2: &Vector3<S>) -> Option<Self> {
        if let (Some(unit_v1), Some(unit_v2)) = (
            v1.try_normalize(S::zero()), 
            v2.try_normalize(S::zero()))
         {
            let cross = unit_v1.cross(&unit_v2);

            if let Some(axis) = Unit::try_from_value(cross, S::default_epsilon()) {
                return Some(
                    Self::from_axis_angle(&axis, Radians::acos(unit_v1.dot(&unit_v2)))
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix3x3,
    /// #     Vector3,
    /// #     Unit,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,   
    /// # };
    /// #
    /// let unit_v1: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x() * 2_f64);
    /// let unit_v2: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y() * 3_f64);
    /// let matrix = Matrix3x3::rotation_between_axis(&unit_v1, &unit_v2).unwrap();
    /// let vector = Vector3::unit_x() * 2_f64;
    /// let expected = Vector3::unit_y() * 2_f64;
    /// let result = matrix * vector;
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn rotation_between_axis(unit_v1: &Unit<Vector3<S>>, unit_v2: &Unit<Vector3<S>>) -> Option<Self> {
        let cross = unit_v1.as_ref().cross(unit_v2.as_ref());
        let cos_angle = unit_v1.as_ref().dot(unit_v2.as_ref());

        if let Some(axis) = Unit::try_from_value(cross, S::default_epsilon()) {
            return Some(
                Self::from_axis_angle(&axis, Radians::acos(cos_angle))
            );
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
    /// # use cglinalg_core::{
    /// #     Matrix3x3,  
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
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
                det_inv * (self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0])
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
    /// # use cglinalg_core::{
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
}

impl<S> From<Matrix2x2<S>> for Matrix3x3<S> 
where 
    S: SimdScalar
{
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: Matrix2x2<S>) -> Self {
        Self::new(
            matrix[0][0], matrix[0][1], S::zero(),
            matrix[1][0], matrix[1][1], S::zero(),
            S::zero(),    S::zero(),    S::one()
        )
    }
}

impl<S> From<&Matrix2x2<S>> for Matrix3x3<S> 
where 
    S: SimdScalar
{
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: &Matrix2x2<S>) -> Self {
        Self::new(
            matrix[0][0], matrix[0][1], S::zero(),
            matrix[1][0], matrix[1][1], S::zero(),
            S::zero(),    S::zero(),    S::one()
        )
    }
}

impl<S> Matrix4x4<S> {
    /// Construct a new matrix from its elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// # };
    /// #
    /// let c0r0 = 1_i32;  let c0r1 = 2_i32;  let c0r2 = 3_i32;  let c0r3 = 4_i32;
    /// let c1r0 = 5_i32;  let c1r1 = 6_i32;  let c1r2 = 7_i32;  let c1r3 = 8_i32;
    /// let c2r0 = 9_i32;  let c2r1 = 10_i32; let c2r2 = 11_i32; let c2r3 = 12_i32;
    /// let c3r0 = 13_i32; let c3r1 = 14_i32; let c3r2 = 15_i32; let c3r3 = 16_i32;
    /// let matrix = Matrix4x4::new(
    ///     c0r0, c0r1, c0r2, c0r3,
    ///     c1r0, c1r1, c1r2, c1r3,
    ///     c2r0, c2r1, c2r2, c2r3,
    ///     c3r0, c3r1, c3r2, c3r3
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
            ]
        }
    }
}

impl<S> Matrix4x4<S>
where 
    S: SimdScalar
{
    /// Construct an affine translation matrix in three-dimensions.
    ///
    ///
    /// # Example
    /// A homogeneous vector with a zero **w-component** should not translate.
    /// ```
    /// # use cglinalg_core::{
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
    /// A homogeneous vector with a unit **w-component** should translate.
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_affine_translation(distance: &Vector3<S>) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
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
    /// calling `Sel::from_scale(scale)` is equivalent to calling 
    /// `Self::from_nonuniform_scale(scale, scale, scale)`. Since this is an affine 
    /// matrix the `w` component is unaffected.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_affine_scale(scale: S) -> Self {
        Self::from_affine_nonuniform_scale(scale, scale, scale)
    }

    /// Construct a three-dimensional affine scaling matrix.
    ///
    /// This is the most general case for affine scaling matrices: the scale 
    /// factor in each dimension need not be identical. Since this is an 
    /// affine matrix, the `w` component is unaffected.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_affine_nonuniform_scale(scale_x: S, scale_y: S, scale_z: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_affine_shear_x(shear_x_with_y: S, shear_x_with_z: S) -> Self {
        let one = S::one();
        let zero = S::zero();
        
        Self::new(
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_affine_shear_y(shear_y_with_x: S, shear_y_with_z: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_affine_shear_z(shear_z_with_x: S, shear_z_with_y: S) -> Self {
        let one = S::one();
        let zero = S::zero();

        Self::new(
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
        shear_z_with_x: S, shear_z_with_y: S) -> Self
    {
        let zero = S::zero();
        let one = S::one();

        Self::new(
            one,            shear_y_with_x, shear_z_with_x, zero,
            shear_x_with_y, one,            shear_z_with_y, zero,
            shear_x_with_z, shear_y_with_z, one,            zero,
            zero,           zero,           zero,           one
        )
    }
}

impl<S> Matrix4x4<S> 
where 
    S: SimdScalarSigned
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
    /// In particular, consider a plane of the form
    /// ```text
    /// P := { (x, y, z) | a * (x - x0) + b * (y - y0) + c * (z - z0) == 0 }
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
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
    pub fn from_affine_reflection(normal: &Unit<Vector3<S>>, bias: &Vector3<S>) -> Self {
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

        Self::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
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
    /// # use cglinalg_core::{
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

impl<S> Matrix4x4<S> 
where 
    S: SimdScalarFloat
{
    /// Construct a three-dimensional affine rotation matrix rotating a vector around the 
    /// **x-axis** by an angle `angle` radians/degrees.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,   
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix4x4::from_affine_angle_x(angle);
    /// let vector = Vector4::new(0_f64, 1_f64, 1_f64, 1_f64);
    /// let expected = Vector4::new(0_f64, -1_f64, 1_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_angle_x<A: Into<Radians<S>>>(angle: A) -> Self {
        let (sin_angle, cos_angle) = angle.into().sin_cos();
        let one = S::one();
        let zero = S::zero();

        Self::new(
            one,   zero,      zero,      zero,
            zero,  cos_angle, sin_angle, zero,
            zero, -sin_angle, cos_angle, zero,
            zero,  zero,      zero,      one
        )
    }
        
    /// Construct a three-dimensional affine rotation matrix rotating a vector 
    /// around the **y-axis** by an angle `angle` radians/degrees.
    ///
    /// # Example
    /// 
    /// In this example the rotation is in the **zx-plane**.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix4x4::from_affine_angle_y(angle);
    /// let vector = Vector4::new(1_f64, 0_f64, 1_f64, 1_f64);
    /// let expected = Vector4::new(1_f64, 0_f64, -1_f64, 1_f64);
    /// let result = matrix * vector;
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_angle_y<A: Into<Radians<S>>>(angle: A) -> Self {
        let (sin_angle, cos_angle) = angle.into().sin_cos();
        let one = S::one();
        let zero = S::zero();

        Self::new(
            cos_angle, zero, -sin_angle, zero,
            zero,      one,   zero,      zero,
            sin_angle, zero,  cos_angle, zero,
            zero,      zero,  zero,      one
        )
    }
    
    /// Construct a three-dimensional affine rotation matrix rotating a vector 
    /// around the **z-axis** by an angle `angle` radians/degrees.
    ///
    /// # Example
    /// 
    /// In this example the rotation is in the **xy-plane**.
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector4, 
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix4x4::from_affine_angle_z(angle);
    /// let vector = Vector4::new(1_f64, 1_f64, 0_f64, 1_f64);
    /// let expected = Vector4::new(-1_f64, 1_f64, 0_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_angle_z<A: Into<Radians<S>>>(angle: A) -> Self {
        let (sin_angle, cos_angle) = angle.into().sin_cos();
        let one = S::one();
        let zero = S::zero();
        
        Self::new(
             cos_angle, sin_angle, zero, zero,
            -sin_angle, cos_angle, zero, zero,
             zero,      zero,      one,  zero,
             zero,      zero,      zero, one
        )
    }

    /// Construct a three-dimensional affine rotation matrix rotating a vector 
    /// around the axis `axis` by an angle `angle` radians/degrees.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Angle,
    /// #     Radians,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Unit,
    /// #     Vector4,
    /// #     Vector3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let axis: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
    /// let angle: Radians<f64> = Radians::full_turn_div_4();
    /// let matrix = Matrix4x4::from_affine_axis_angle(&axis, angle);
    /// let vector = Vector4::new(1_f64, 0_f64, 0_f64, 1_f64);
    /// let expected = Vector4::new(0_f64, 1_f64, 0_f64, 1_f64);
    /// let result = matrix * vector;
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_affine_axis_angle<A: Into<Radians<S>>>(axis: &Unit<Vector3<S>>, angle: A) -> Self {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());
        let one_minus_cos_angle = S::one() - cos_angle;
        let _axis = axis.as_ref();

        Self::new(
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4, 
    /// # };
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
    ///     0_f64,          0_f64,         -101_f64 / 99_f64, 1_f64
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
        let sx =  two / (right - left);
        let sy =  two / (top - bottom);
        let sz = -two / (far - near);
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);
        let tz = -(far + near) / (far - near);

        // We use the same orthographic projection matrix that OpenGL uses.
        Self::new(
            sx,   zero, zero, zero,
            zero, sy,   zero, zero,
            zero, zero, sz,   zero,
            tx,   ty,   tz,   one
        )
    }

    /// Construct a new three-dimensional orthographic projection matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let vfov = Degrees(90_f64);
    /// let aspect = 800_f64 / 600_f64;
    /// let near = 1_f64;
    /// let far = 100_f64;
    /// let expected = Matrix4x4::new(
    ///     2_f64 / 100_f64, 0_f64,           0_f64,            0_f64, 
    ///     0_f64,           2_f64 / 75_f64,  0_f64,            0_f64, 
    ///     0_f64,           0_f64,          -2_f64 / 99_f64,   0_f64, 
    ///     0_f64,           0_f64,          -101_f64 / 99_f64, 1_f64
    /// );
    /// let result = Matrix4x4::from_orthographic_fov(vfov, aspect, near, far);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_orthographic_fov<A: Into<Radians<S>>>(vfov: A, aspect: S, near: S, far: S) -> Self {
        let one = S::one();
        let one_half = one / (one + one);
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4, 
    /// # };
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
    ///     0_f64,          0_f64,         -200_f64 / 99_f64,  0_f64
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

        // We use the same perspective projection matrix that OpenGL uses.
        Self::new(
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
    /// # Example
    ///
    /// ```
    /// # use cglinalg_trigonometry::{
    /// #     Degrees,
    /// # };
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let vfov = Degrees(72_f32);
    /// let aspect = 800_f32 / 600_f32;
    /// let near = 0.1_f32;
    /// let far = 100_f32;
    /// let expected = Matrix4x4::new(
    ///     1.0322863_f32, 0_f32,          0_f32,          0_f32,
    ///     0_f32,         1.3763818_f32,  0_f32,          0_f32,
    ///     0_f32,         0_f32,         -1.002002_f32,  -1_f32,
    ///     0_f32,         0_f32,         -0.2002002_f32,  0_f32
    /// );
    /// let result = Matrix4x4::from_perspective_fov(vfov, aspect, near, far);
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-6);
    /// ```
    #[rustfmt::skip]
    #[inline]
    pub fn from_perspective_fov<A: Into<Radians<S>>>(vfov: A, aspect: S, near: S, far: S) -> Self {
        let zero = S::zero();
        let one = S::one();
        let two = one + one;
        let range = Angle::tan(vfov.into() / two) * near;
        let sx = (two * near) / (range * aspect + range * aspect);
        let sy = near / range;
        let sz = (far + near) / (near - far);
        let pz = (two * far * near) / (near - far);
        
        // We use the same perspective projection matrix that OpenGL uses.
        Self::new(
            sx,    zero,  zero,  zero,
            zero,  sy,    zero,  zero,
            zero,  zero,  sz,   -one,
            zero,  zero,  pz,    zero
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
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Vector4,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
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
    ///      -1_f64 / f64::sqrt(2_f64), -3_f64, -3_f64 / f64::sqrt(2_f64),  1_f64
    /// );
    /// let result = Matrix4x4::look_to_lh(&eye, &direction, &up);
    /// let unit_z = Vector4::unit_z();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(
    ///     (result * direction.to_homogeneous()).normalize(), 
    ///     unit_z, 
    ///     epsilon = 1e-10,
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
            x_axis.x,  y_axis.x,  z_axis.x,  zero,
            x_axis.y,  y_axis.y,  z_axis.y,  zero,
            x_axis.z,  y_axis.z,  z_axis.z,  zero,
            neg_eye_x, neg_eye_y, neg_eye_z, one
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
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Vector4,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
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
    ///      1_f64 / f64::sqrt(2_f64), -3_f64,  3_f64 / f64::sqrt(2_f64),  1_f64
    /// );
    /// let result = Matrix4x4::look_to_rh(&eye, &direction, &up);
    /// let minus_unit_z = -Vector4::unit_z();
    ///
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(
    ///     (result * direction.to_homogeneous()).normalize(), 
    ///     minus_unit_z, 
    ///     epsilon = 1e-10
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
    /// coordinate transformation.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Vector4,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
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
    ///      -1_f64 / f64::sqrt(2_f64), -3_f64, -3_f64 / f64::sqrt(2_f64),  1_f64
    /// );
    /// let result = Matrix4x4::look_at_lh(&eye, &target, &up);
    /// let direction = (target - eye).to_homogeneous();
    /// let unit_z = Vector4::unit_z();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!((result * direction).normalize(), unit_z, epsilon = 1e-10);
    /// ```
    #[rustfmt::skip]
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
    /// # use cglinalg_core::{
    /// #     Matrix4x4,
    /// #     Vector3,
    /// #     Vector4,
    /// #     Point3, 
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
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
    /// let direction = (target - eye).to_homogeneous();
    /// let minus_unit_z = -Vector4::unit_z();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!((result * direction).normalize(), minus_unit_z, epsilon = 1e-10);
    /// ```
    #[rustfmt::skip]
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
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Vector4,
    /// #     Matrix4x4,
    /// #     Point3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,  
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result * unit_z, direction, epsilon = 1e-10);
    /// assert_relative_eq!(result * minus_unit_z, -direction, epsilon = 1e-10);
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
            x_axis.x,  x_axis.y,  x_axis.z, zero,
            y_axis.x,  y_axis.y,  y_axis.z, zero,
            z_axis.x,  z_axis.y,  z_axis.z, zero,
            eye_x,     eye_y,     eye_z,    one
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
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Vector4,
    /// #     Matrix4x4,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result * unit_z, -direction, epsilon = 1e-10);
    /// assert_relative_eq!(result * minus_unit_z, direction, epsilon = 1e-10);
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
            x_axis.x,  x_axis.y,  x_axis.z, zero,
            y_axis.x,  y_axis.y,  y_axis.z, zero,
            z_axis.x,  z_axis.y,  z_axis.z, zero,
            eye_x,     eye_y,     eye_z,    one,
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
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Vector4,
    /// #     Matrix4x4,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result * unit_z, direction, epsilon = 1e-10);
    /// assert_relative_eq!(result * minus_unit_z, -direction, epsilon = 1e-10);
    /// ```
    #[rustfmt::skip]
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
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Vector4,
    /// #     Matrix4x4,
    /// #     Point3,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result * unit_z, -direction, epsilon = 1e-10);
    /// assert_relative_eq!(result * minus_unit_z, direction, epsilon = 1e-10);
    /// ```
    #[rustfmt::skip]
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
    /// # use cglinalg_core::{
    /// #     Matrix4x4,  
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
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
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
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
                c3r0, c3r1, c3r2, c3r3
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
    /// # use cglinalg_core::{
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
    /// assert!(!matrix.is_invertible());
    /// ```
    #[inline]
    pub fn is_invertible(&self) -> bool {
        ulps_ne!(self.determinant(), S::zero())
    }
}

impl<S> From<Matrix2x2<S>> for Matrix4x4<S> 
where 
    S: SimdScalar 
{
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: Matrix2x2<S>) -> Self {
        let one = S::one();
        let zero = S::zero();
        Self::new(
            matrix[0][0], matrix[0][1], zero, zero,
            matrix[1][0], matrix[1][1], zero, zero,
            zero,         zero,         one,  zero,
            zero,         zero,         zero, one
        )
    }
}

impl<S> From<&Matrix2x2<S>> for Matrix4x4<S> 
where 
    S: SimdScalar
{
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: &Matrix2x2<S>) -> Self {
        let one = S::one();
        let zero = S::zero();
        Self::new(
            matrix[0][0], matrix[0][1], zero, zero,
            matrix[1][0], matrix[1][1], zero, zero,
            zero,         zero,         one,  zero,
            zero,         zero,         zero, one
        )
    }
}

impl<S> From<Matrix3x3<S>> for Matrix4x4<S> 
where 
    S: SimdScalar
{
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: Matrix3x3<S>) -> Self {
        let one = S::one();
        let zero = S::zero();
        Self::new(
            matrix[0][0], matrix[0][1], matrix[0][2], zero,
            matrix[1][0], matrix[1][1], matrix[1][2], zero,
            matrix[2][0], matrix[2][1], matrix[2][2], zero,
            zero,         zero,         zero,         one
        )
    }
}

impl<S> From<&Matrix3x3<S>> for Matrix4x4<S> 
where 
    S: SimdScalar
{
    #[rustfmt::skip]
    #[inline]
    fn from(matrix: &Matrix3x3<S>) -> Self {
        let one = S::one();
        let zero = S::zero();
        Self::new(
            matrix[0][0], matrix[0][1], matrix[0][2], zero,
            matrix[1][0], matrix[1][1], matrix[1][2], zero,
            matrix[2][0], matrix[2][1], matrix[2][2], zero,
            zero,         zero,         zero,         one
        )
    }
}

impl<S> Matrix1x2<S> {
    /// Construct a new matrix from its elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix1x2,
    /// # };
    /// #
    /// let c0r0 = 1_i32;
    /// let c1r0 = 2_i32;
    /// let matrix = Matrix1x2::new(
    ///     c0r0,
    ///     c1r0
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
            ]
        }
    }
}

impl<S> Matrix1x3<S> {
    /// Construct a new matrix from its elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix1x3,
    /// # };
    /// #
    /// let c0r0 = 1_i32;
    /// let c1r0 = 2_i32;
    /// let c2r0 = 3_i32;
    /// let matrix = Matrix1x3::new(
    ///     c0r0,
    ///     c1r0,
    ///     c2r0
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
            ]
        }
    }
}

impl<S> Matrix1x4<S> {
    /// Construct a new matrix from its elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix1x4,
    /// # };
    /// #
    /// let c0r0 = 1_i32;
    /// let c1r0 = 2_i32;
    /// let c2r0 = 3_i32;
    /// let c3r0 = 4_i32;
    /// let matrix = Matrix1x4::new(
    ///     c0r0,
    ///     c1r0,
    ///     c2r0,
    ///     c3r0
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
            ]
        }
    }
}

impl<S> Matrix2x3<S> {
    /// Construct a new matrix from its elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x3,
    /// # };
    /// #
    /// let c0r0 = 1_i32; let c0r1 = 2_i32;
    /// let c1r0 = 3_i32; let c1r1 = 4_i32;
    /// let c2r0 = 5_i32; let c2r1 = 6_i32;
    /// let matrix = Matrix2x3::new(
    ///     c0r0, c0r1,
    ///     c1r0, c1r1,
    ///     c2r0, c2r1
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
            ]
        }
    }
}

impl<S> Matrix3x2<S> {
    /// Construct a new matrix from its elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix3x2,
    /// # };
    /// #
    /// let c0r0 = 1_i32; let c0r1 = 2_i32; let c0r2 = 3_i32;
    /// let c1r0 = 4_i32; let c1r1 = 5_i32; let c1r2 = 6_i32;
    /// let matrix = Matrix3x2::new(
    ///     c0r0, c0r1, c0r2,
    ///     c1r0, c1r1, c1r2
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
            ]
        }
    }
}

impl<S> Matrix2x4<S> {
    /// Construct a new matrix from its elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix2x4,
    /// # };
    /// #
    /// let c0r0 = 1_i32; let c0r1 = 2_i32;
    /// let c1r0 = 3_i32; let c1r1 = 4_i32;
    /// let c2r0 = 5_i32; let c2r1 = 6_i32;
    /// let c3r0 = 7_i32; let c3r1 = 8_i32;
    /// let matrix = Matrix2x4::new(
    ///     c0r0, c0r1,
    ///     c1r0, c1r1,
    ///     c2r0, c2r1,
    ///     c3r0, c3r1
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
            ]
        }
    }
}

impl<S> Matrix4x2<S> {
    /// Construct a new matrix from its elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x2,
    /// # };
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
    ///     c1r0, c1r1, c1r2, c1r3
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
    /// # use cglinalg_core::{
    /// #     Matrix3x4,
    /// # };
    /// #
    /// let c0r0 = 1_i32;  let c0r1 = 2_i32;  let c0r2 = 3_i32;
    /// let c1r0 = 4_i32;  let c1r1 = 5_i32;  let c1r2 = 6_i32;
    /// let c2r0 = 7_i32;  let c2r1 = 8_i32;  let c2r2 = 9_i32;
    /// let c3r0 = 10_i32; let c3r1 = 11_i32; let c3r2 = 12_i32;
    /// let matrix = Matrix3x4::new(
    ///     c0r0, c0r1, c0r2,
    ///     c1r0, c1r1, c1r2,
    ///     c2r0, c2r1, c2r2,
    ///     c3r0, c3r1, c3r2
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
            ]
        }
    }
}

impl<S> Matrix4x3<S> {
    /// Construct a new matrix from its elements.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Matrix4x3,
    /// # };
    /// #
    /// let c0r0 = 1_i32; let c0r1 = 2_i32;  let c0r2 = 3_i32;  let c0r3 = 4_i32;
    /// let c1r0 = 5_i32; let c1r1 = 6_i32;  let c1r2 = 7_i32;  let c1r3 = 8_i32;
    /// let c2r0 = 9_i32; let c2r1 = 10_i32; let c2r2 = 11_i32; let c2r3 = 12_i32;
    /// let matrix = Matrix4x3::new(
    ///     c0r0, c0r1, c0r2, c0r3,
    ///     c1r0, c1r1, c1r2, c1r3,
    ///     c2r0, c2r1, c2r2, c2r3
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
            ]
        }
    }
}

impl_coords!(View1x1, { c0r0 });
impl_coords!(View2x2, { c0r0, c0r1, c1r0, c1r1 });
impl_coords!(View3x3, { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_coords!(View4x4, { 
    c0r0, c0r1, c0r2, c0r3, 
    c1r0, c1r1, c1r2, c1r3, 
    c2r0, c2r1, c2r2, c2r3, 
    c3r0, c3r1, c3r2, c3r3 
});
impl_coords!(View1x2, { c0r0, c1r0 });
impl_coords!(View1x3, { c0r0, c1r0, c2r0 });
impl_coords!(View1x4, { c0r0, c1r0, c2r0, c3r0 });
impl_coords!(View2x3, { c0r0, c0r1, c1r0, c1r1, c2r0, c2r1 });
impl_coords!(View2x4, { c0r0, c0r1, c1r0, c1r1, c2r0, c2r1, c3r0, c3r1 });
impl_coords!(View3x4, { 
    c0r0, c0r1, c0r2, 
    c1r0, c1r1, c1r2, 
    c2r0, c2r1, c2r2, 
    c3r0, c3r1, c3r2
});
impl_coords!(View3x2, { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2 });
impl_coords!(View4x2, { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3 });
impl_coords!(View4x3, { 
    c0r0, c0r1, c0r2, c0r3, 
    c1r0, c1r1, c1r2, c1r3,
    c2r0, c2r1, c2r2, c2r3
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
    S: SimdScalar
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
    S: SimdScalar 
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar 
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalarSigned
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
    S: SimdScalarSigned 
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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

impl<S, const R1: usize, const C1: usize, const R2: usize, const C2: usize, const R1C2: usize> ops::Mul<Matrix<S, R2, C2>> for Matrix<S, R1, C1>
where 
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<R1>, Const<C1>, Const<R2>, Const<C2>>,
    ShapeConstraint: DimMul<Const<R1>, Const<C2>, Output = Const<R1C2>>,
    ShapeConstraint: DimMul<Const<C2>, Const<R1>, Output = Const<R1C2>>
{
    type Output = Matrix<S, R1, C2>;

    #[inline]
    fn mul(self, other: Matrix<S, R2, C2>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C2 {
            for r in 0..R1 {
                result[c][r] = dot_array_col(
                    self.as_ref(), 
                    &<Matrix<S, R2, C2> as AsRef<[[S; R2]; C2]>>::as_ref(&other)[c], 
                    r
                );
            }
        }

        result
    }
}

impl<S, const R1: usize, const C1: usize, const R2: usize, const C2: usize, const R1C2: usize> ops::Mul<&Matrix<S, R2, C2>> for Matrix<S, R1, C1>
where 
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<R1>, Const<C1>, Const<R2>, Const<C2>>,
    ShapeConstraint: DimMul<Const<R1>, Const<C2>, Output = Const<R1C2>>,
    ShapeConstraint: DimMul<Const<C2>, Const<R1>, Output = Const<R1C2>>
{
    type Output = Matrix<S, R1, C2>;

    #[inline]
    fn mul(self, other: &Matrix<S, R2, C2>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C2 {
            for r in 0..R1 {
                result[c][r] = dot_array_col(
                    self.as_ref(), 
                    &<Matrix<S, R2, C2> as AsRef<[[S; R2]; C2]>>::as_ref(other)[c], 
                    r
                );
            }
        }

        result
    }
}

impl<S, const R1: usize, const C1: usize, const R2: usize, const C2: usize, const R1C2: usize> ops::Mul<Matrix<S, R2, C2>> for &Matrix<S, R1, C1>
where 
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<R1>, Const<C1>, Const<R2>, Const<C2>>,
    ShapeConstraint: DimMul<Const<R1>, Const<C2>, Output = Const<R1C2>>,
    ShapeConstraint: DimMul<Const<C2>, Const<R1>, Output = Const<R1C2>>
{
    type Output = Matrix<S, R1, C2>;

    #[inline]
    fn mul(self, other: Matrix<S, R2, C2>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C2 {
            for r in 0..R1 {
                result[c][r] = dot_array_col(
                    self.as_ref(), 
                    &<Matrix<S, R2, C2> as AsRef<[[S; R2]; C2]>>::as_ref(&other)[c], 
                    r
                );
            }
        }

        result
    }
}

impl<'a, 'b, S, const R1: usize, const C1: usize, const R2: usize, const C2: usize, const R1C2: usize> ops::Mul<&'b Matrix<S, R2, C2>> for &'a Matrix<S, R1, C1>
where 
    S: SimdScalar,
    ShapeConstraint: CanMultiply<Const<R1>, Const<C1>, Const<R2>, Const<C2>>,
    ShapeConstraint: DimMul<Const<R1>, Const<C2>, Output = Const<R1C2>>,
    ShapeConstraint: DimMul<Const<C2>, Const<R1>, Output = Const<R1C2>>
{
    type Output = Matrix<S, R1, C2>;

    #[inline]
    fn mul(self, other: &'b Matrix<S, R2, C2>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for c in 0..C2 {
            for r in 0..R1 {
                result[c][r] = dot_array_col(
                    self.as_ref(), 
                    &<Matrix<S, R2, C2> as AsRef<[[S; R2]; C2]>>::as_ref(other)[c], 
                    r
                );
            }
        }

        result
    }
}


impl<S, const R: usize, const C: usize> ops::AddAssign<Matrix<S, R, C>> for Matrix<S, R, C>
where 
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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
    S: SimdScalar
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

impl<S, const R: usize, const C: usize> approx::AbsDiffEq for Matrix<S, R, C>
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
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
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
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
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

