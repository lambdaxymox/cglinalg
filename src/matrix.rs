use std::fmt;
use std::mem;
use std::ops;
use std::iter;


use approx::{
    ulps_eq,
};
use num_traits::{
    NumCast,
};
use scalar::{
    Scalar,
    ScalarSigned,
    ScalarFloat,
};
use angle::Radians;
use structure::{
    Angle,
    Array, 
    One, 
    Zero, 
    Matrix, 
    Lerp,
    Sum,
    Product,
    SquareMatrix,
    SkewSymmetricMatrix,
    InvertibleSquareMatrix,
};
use vector::*;


/// The `Matrix2` type represents 2x2 matrices in column-major order.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C)]
pub struct Matrix2<S> {
    /// Column 0 of the matrix.
    pub c0r0: S, pub c0r1: S,
    /// Column 1 of the matrix.
    pub c1r0: S, pub c1r1: S,
}

impl<S> Matrix2<S> {
    /// Construct a new 2x2 matrix from its field elements.
    pub const fn new(c0r0: S, c0r1: S, c1r0: S, c1r1: S) -> Matrix2<S> {
        Matrix2 { c0r0: c0r0, c0r1: c0r1, c1r0: c1r0, c1r1: c1r1 }
    }

    /// Construct a 2x2 matrix from a pair of two-dimensional vectors.
    pub fn from_columns(c0: Vector2<S>, c1: Vector2<S>) -> Matrix2<S> {
        Matrix2 { c0r0: c0.x, c0r1: c0.y, c1r0: c1.x, c1r1: c1.y }
    }

    /// Map an operation on the elements of a matrix, returning a matrix whose elements
    /// are elements of the new underlying type.
    pub fn map<T, F>(self, mut op: F) -> Matrix2<T> where F: FnMut(S) -> T {
        Matrix2 {
            c0r0: op(self.c0r0), c1r0: op(self.c1r0),
            c0r1: op(self.c0r1), c1r1: op(self.c1r1),
        }
    }
}

impl<S> Matrix2<S> where S: NumCast + Copy {
    /// Cast a matrix from one type of scalars to another type of scalars.
    pub fn cast<T: NumCast>(&self) -> Option<Matrix2<T>> {
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

        Some(Matrix2::new(c0r0, c0r1, c1r0, c1r1))
    }
}

impl<S> Matrix2<S> where S: ScalarFloat {
    pub fn from_angle<A: Into<Radians<S>>>(angle: A) -> Matrix2<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Matrix2::new(
             cos_angle, sin_angle, 
            -sin_angle, cos_angle
        )
    }
}

impl<S> Array for Matrix2<S> where S: Scalar {
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
    fn from_value(value: Self::Element) -> Self {
        Matrix2::new(value, value, value, value)
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

impl<S> Sum for Matrix2<S> where S: Scalar {
    #[inline]
    fn sum(&self) -> S {
        self.c0r0 + self.c1r0 + self.c0r1 + self.c1r1
    }
}

impl<S> Product for Matrix2<S> where S: Scalar {
    #[inline]
    fn product(&self) -> S {
        self.c0r0 * self.c1r0 * self.c0r1 * self.c1r1
    }
}

impl<S> Matrix for Matrix2<S> where S: Scalar {
    type Element = S;
    type Row = Vector2<S>;
    type Column = Vector2<S>;
    type Transpose = Matrix2<S>;

    fn row(&self, r: usize) -> Self::Row {
        Vector2::new(self[0][r], self[1][r])
    }
    
    fn swap_rows(&mut self, row_a: usize, row_b: usize) {
        let c0ra = self[0][row_a];
        let c1ra = self[1][row_a];
        self[0][row_a] = self[0][row_b];
        self[1][row_a] = self[1][row_b];
        self[0][row_b] = c0ra;
        self[1][row_b] = c1ra;
    }
    
    fn swap_columns(&mut self, col_a: usize, col_b: usize) {
        let car0 = self[col_a][0];
        let car1 = self[col_a][1];
        self[col_a][0] = self[col_b][0];
        self[col_a][1] = self[col_b][1];
        self[col_b][0] = car0;
        self[col_b][1] = car1;
    }
    
    fn swap_elements(&mut self, a: (usize, usize), b: (usize, usize)) {
        let element_a = self[a.0][a.1];
        self[a.0][a.1] = self[b.0][b.1];
        self[b.0][b.1] = element_a;
    }
    
    fn transpose(&self) -> Self::Transpose {
        Matrix2::new(self.c0r0, self.c1r0, self.c0r1, self.c1r1)
    }
}

impl<S> From<[[S; 2]; 2]> for Matrix2<S> where S: Scalar {
    #[inline]
    fn from(m: [[S; 2]; 2]) -> Matrix2<S> {
        Matrix2::new(m[0][0], m[0][1], m[1][0], m[1][1])
    }
}

impl<'a, S> From<&'a [[S; 2]; 2]> for &'a Matrix2<S> where S: Scalar {
    #[inline]
    fn from(m: &'a [[S; 2]; 2]) -> &'a Matrix2<S> {
        unsafe { mem::transmute(m) }
    }    
}

impl<S> From<[S; 4]> for Matrix2<S> where S: Scalar {
    #[inline]
    fn from(m: [S; 4]) -> Matrix2<S> {
        Matrix2::new(m[0], m[1], m[2], m[3])
    }
}

impl<'a, S> From<&'a [S; 4]> for &'a Matrix2<S> where S: Scalar {
    #[inline]
    fn from(m: &'a [S; 4]) -> &'a Matrix2<S> {
        unsafe { mem::transmute(m) }
    }
}

impl<S> fmt::Display for Matrix2<S> where S: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, 
            "\n[{:.2}][{:.2}]\n[{:.2}][{:.2}]", 
            self.c0r0, self.c1r0,
            self.c0r1, self.c1r1,
        )
    }
}

impl<S> AsRef<[S; 4]> for Matrix2<S> {
    fn as_ref(&self) -> &[S; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsRef<[[S; 2]; 2]> for Matrix2<S> {
    fn as_ref(&self) -> &[[S; 2]; 2] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsRef<[Vector2<S>; 2]> for Matrix2<S> {
    fn as_ref(&self) -> &[Vector2<S>; 2] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[S; 4]> for Matrix2<S> {
    fn as_mut(&mut self) -> &mut [S; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[[S; 2]; 2]> for Matrix2<S> {
    fn as_mut(&mut self) -> &mut [[S; 2]; 2] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[Vector2<S>; 2]> for Matrix2<S> {
    fn as_mut(&mut self) -> &mut [Vector2<S>; 2] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> ops::Index<usize> for Matrix2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[Vector2<S>; 2] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::IndexMut<usize> for Matrix2<S> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Vector2<S> {
        let v: &mut [Vector2<S>; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> Zero for Matrix2<S> where S: Scalar {
    fn zero() -> Matrix2<S> {
        Matrix2::new(S::zero(), S::zero(), S::zero(), S::zero())
    }

    fn is_zero(&self) -> bool {
        self.c0r0 == S::zero() && self.c0r1 == S::zero() &&
        self.c1r0 == S::zero() && self.c1r1 == S::zero()
    }
}

impl<S> One for Matrix2<S> where S: Scalar {
    fn one() -> Matrix2<S> {
        Matrix2::new(S::one(), S::zero(), S::zero(), S::one())
    }
}

impl<S> ops::Add<Matrix2<S>> for Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn add(self, other: Matrix2<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Add<&Matrix2<S>> for Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn add(self, other: &Matrix2<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Add<Matrix2<S>> for &Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn add(self, other: Matrix2<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a, 'b, S> ops::Add<&'a Matrix2<S>> for &'b Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn add(self, other: &'a Matrix2<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Sub<Matrix2<S>> for Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn sub(self, other: Matrix2<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Sub<&Matrix2<S>> for Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn sub(self, other: &Matrix2<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Sub<Matrix2<S>> for &Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn sub(self, other: Matrix2<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a, 'b, S> ops::Sub<&'a Matrix2<S>> for &'b Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn sub(self, other: &'a Matrix2<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Mul<&Matrix2<S>> for Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn mul(self, other: &Matrix2<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1;
        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Matrix2<S>> for &'b Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn mul(self, other: &'a Matrix2<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1;
        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Mul<Matrix2<S>> for Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn mul(self, other: Matrix2<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1;
        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Mul<Matrix2<S>> for &Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn mul(self, other: Matrix2<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1;
        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Mul<S> for Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn mul(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Mul<Vector2<S>> for Matrix2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn mul(self, other: Vector2<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y;
        let y = self.c1r0 * other.x + self.c1r1 * other.y;

        Vector2::new(x, y)
    }
}

impl<S> ops::Mul<&Vector2<S>> for Matrix2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn mul(self, other: &Vector2<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y;
        let y = self.c1r0 * other.x + self.c1r1 * other.y;

        Vector2::new(x, y)
    }
}

impl<S> ops::Mul<Vector2<S>> for &Matrix2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn mul(self, other: Vector2<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y;
        let y = self.c1r0 * other.x + self.c1r1 * other.y;

        Vector2::new(x, y)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Vector2<S>> for &'b Matrix2<S> where S: Scalar {
    type Output = Vector2<S>;

    fn mul(self, other: &'a Vector2<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y;
        let y = self.c1r0 * other.x + self.c1r1 * other.y;

        Vector2::new(x, y)
    }
}

impl<S> ops::Mul<S> for &Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn mul(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 * other;
        let c0r1 = self.c0r1 * other;
        let c1r0 = self.c1r0 * other;
        let c1r1 = self.c1r1 * other;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Div<S> for Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn div(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Div<S> for &Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn div(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 / other;
        let c0r1 = self.c0r1 / other;
        let c1r0 = self.c1r0 / other;
        let c1r1 = self.c1r1 / other;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Neg for Matrix2<S> where S: ScalarSigned {
    type Output = Matrix2<S>;

    fn neg(self) -> Self::Output {
        let c0r0 = -self.c0r0;
        let c0r1 = -self.c0r1;
        let c1r0 = -self.c1r0;
        let c1r1 = -self.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Neg for &Matrix2<S> where S: ScalarSigned {
    type Output = Matrix2<S>;

    fn neg(self) -> Self::Output {
        let c0r0 = -self.c0r0;
        let c0r1 = -self.c0r1;
        let c1r0 = -self.c1r0;
        let c1r1 = -self.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Rem<S> for Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn rem(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 % other;
        let c0r1 = self.c0r1 % other;
        let c1r0 = self.c1r0 % other;
        let c1r1 = self.c1r1 % other;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Rem<S> for &Matrix2<S> where S: Scalar {
    type Output = Matrix2<S>;

    fn rem(self, other: S) -> Self::Output {
        let c0r0 = self.c0r0 % other;
        let c0r1 = self.c0r1 % other;
        let c1r0 = self.c1r0 % other;
        let c1r1 = self.c1r1 % other;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)        
    }
}

impl<S> ops::AddAssign<Matrix2<S>> for Matrix2<S> where S: Scalar {
    fn add_assign(&mut self, other: Matrix2<S>) {
        self.c0r0 += other.c0r0;
        self.c0r1 += other.c0r1;
        self.c1r0 += other.c1r0;
        self.c1r1 += other.c1r1;
    }
}

impl<S> ops::AddAssign<&Matrix2<S>> for Matrix2<S> where S: Scalar {
    fn add_assign(&mut self, other: &Matrix2<S>) {
        self.c0r0 += other.c0r0;
        self.c0r1 += other.c0r1;
        self.c1r0 += other.c1r0;
        self.c1r1 += other.c1r1;
    }
}

impl<S> ops::SubAssign<Matrix2<S>> for Matrix2<S> where S: Scalar {
    fn sub_assign(&mut self, other: Matrix2<S>) {
        self.c0r0 -= other.c0r0;
        self.c0r1 -= other.c0r1;
        self.c1r0 -= other.c1r0;
        self.c1r1 -= other.c1r1;
    }
}

impl<S> ops::SubAssign<&Matrix2<S>> for Matrix2<S> where S: Scalar {
    fn sub_assign(&mut self, other: &Matrix2<S>) {
        self.c0r0 -= other.c0r0;
        self.c0r1 -= other.c0r1;
        self.c1r0 -= other.c1r0;
        self.c1r1 -= other.c1r1;
    }
}

impl<S> ops::MulAssign<S> for Matrix2<S> where S: Scalar {
    fn mul_assign(&mut self, other: S) {
        self.c0r0 *= other;
        self.c0r1 *= other;
        self.c1r0 *= other;
        self.c1r1 *= other;
    }
}

impl<S> ops::DivAssign<S> for Matrix2<S> where S: Scalar {
    fn div_assign(&mut self, other: S) {
        self.c0r0 /= other;
        self.c0r1 /= other;
        self.c1r0 /= other;
        self.c1r1 /= other;
    }
}

impl<S> ops::RemAssign<S> for Matrix2<S> where S: Scalar {
    fn rem_assign(&mut self, other: S) {
        self.c0r0 %= other;
        self.c0r1 %= other;
        self.c1r0 %= other;
        self.c1r1 %= other;
    }
}

impl<S> Lerp<Matrix2<S>> for Matrix2<S> where S: Scalar {
    type Scalar = S;
    type Output = Matrix2<S>;

    fn lerp(self, other: Matrix2<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<&Matrix2<S>> for Matrix2<S> where S: Scalar {
    type Scalar = S;
    type Output = Matrix2<S>;

    fn lerp(self, other: &Matrix2<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<Matrix2<S>> for &Matrix2<S> where S: Scalar {
    type Scalar = S;
    type Output = Matrix2<S>;

    fn lerp(self, other: Matrix2<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<'a, 'b, S> Lerp<&'a Matrix2<S>> for &'b Matrix2<S> where S: Scalar {
    type Scalar = S;
    type Output = Matrix2<S>;

    fn lerp(self, other: &'a Matrix2<S>, amount: S) -> Self::Output {
        self + ((other - self) * amount)
    }
}

impl<S> approx::AbsDiffEq for Matrix2<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Matrix2<S> where S: ScalarFloat {
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

impl<S> approx::UlpsEq for Matrix2<S> where S: ScalarFloat {
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


impl<S> SquareMatrix for Matrix2<S> where S: ScalarFloat {
    type ColumnRow = Vector2<S>;

    #[inline]
    fn from_value(value: Self::Element) -> Self {
        Matrix2::new(
            value,     S::zero(),
            S::zero(), value
        )
    }
    
    #[inline]
    fn from_diagonal(value: Vector2<S>) -> Self {
        Matrix2::new(
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
        ulps_eq!(self, &Self::one())
    }
}

impl<S> SkewSymmetricMatrix for Matrix2<S> where S: ScalarFloat {
    fn is_skew_symmetric(&self) -> bool {
        ulps_eq!(self.c0r1, -self.c1r0) && ulps_eq!(self.c1r0, -self.c0r1)
    }
}

impl<S> InvertibleSquareMatrix for Matrix2<S> where S: ScalarFloat {
    fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == S::zero() {
            // A matrix with zero determinant has no inverse.
            None
        } else {
            let inv_det = S::one() / det;
            Some(Matrix2::new(
                inv_det *  self.c1r1, inv_det * -self.c0r1,
                inv_det * -self.c1r0, inv_det *  self.c0r0
            ))
        }
    }
}

impl<S: Scalar> iter::Sum<Matrix2<S>> for Matrix2<S> {
    #[inline]
    fn sum<I: Iterator<Item = Matrix2<S>>>(iter: I) -> Matrix2<S> {
        iter.fold(Matrix2::<S>::zero(), ops::Add::add)
    }
}

impl<'a, S: 'a + Scalar> iter::Sum<&'a Matrix2<S>> for Matrix2<S> {
    #[inline]
    fn sum<I: Iterator<Item = &'a Matrix2<S>>>(iter: I) -> Matrix2<S> {
        iter.fold(Matrix2::<S>::zero(), ops::Add::add)
    }
}

impl<S: Scalar> iter::Product<Matrix2<S>> for Matrix2<S> {
    #[inline]
    fn product<I: Iterator<Item = Matrix2<S>>>(iter: I) -> Matrix2<S> {
        iter.fold(Matrix2::<S>::one(), ops::Mul::mul)
    }
}

impl<'a, S: 'a + Scalar> iter::Product<&'a Matrix2<S>> for Matrix2<S> {
    #[inline]
    fn product<I: Iterator<Item = &'a Matrix2<S>>>(iter: I) -> Matrix2<S> {
        iter.fold(Matrix2::<S>::one(), ops::Mul::mul)
    }
}



/// The `Matrix3` type represents 3x3 matrices in column-major order.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C)]
pub struct Matrix3<S> {
    /// Column 0 of the matrix.
    pub c0r0: S, pub c0r1: S, pub c0r2: S,
    /// Column 1 of the matrix.
    pub c1r0: S, pub c1r1: S, pub c1r2: S,
    /// Column 2 of the matrix.
    pub c2r0: S, pub c2r1: S, pub c2r2: S,
}

impl<S> Matrix3<S> {
    pub const fn new(
        c0r0: S, c0r1: S, c0r2: S,
        c1r0: S, c1r1: S, c1r2: S,
        c2r0: S, c2r1: S, c2r2: S) -> Matrix3<S> {

        Matrix3 {
            // Column 0 of the matrix.
            c0r0: c0r0, c0r1: c0r1, c0r2: c0r2,
            // Column 1 of the matrix.
            c1r0: c1r0, c1r1: c1r1, c1r2: c1r2,
            // Column 2 of the matrix.
            c2r0: c2r0, c2r1: c2r1, c2r2: c2r2,
        }
    }

    /// Create a 3x3 matrix from a triple of three-dimensional column vectors.
    pub fn from_columns(c0: Vector3<S>, c1: Vector3<S>, c2: Vector3<S>) -> Matrix3<S> {
        Matrix3 {
            c0r0: c0.x, c0r1: c0.y, c0r2: c0.z, 
            c1r0: c1.x, c1r1: c1.y, c1r2: c1.z,
            c2r0: c2.x, c2r1: c2.y, c2r2: c2.z,
        }
    }

    /// Map an operation on the elements of a matrix, returning a matrix whose elements
    /// are elements of the new underlying type.
    pub fn map<T, F>(self, mut op: F) -> Matrix3<T> where F: FnMut(S) -> T {
        Matrix3 {
            c0r0: op(self.c0r0), c1r0: op(self.c1r0), c2r0: op(self.c2r0),
            c0r1: op(self.c0r1), c1r1: op(self.c1r1), c2r1: op(self.c2r1),
            c0r2: op(self.c0r2), c1r2: op(self.c1r2), c2r2: op(self.c2r2),
        }
    }
}

impl<S> Matrix3<S> where S: NumCast + Copy {
    /// Cast a matrix from one type of scalars to another type of scalars.
    pub fn cast<T: NumCast>(&self) -> Option<Matrix3<T>> {
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

        Some(Matrix3::new(
            c0r0, c0r1, c0r2,
            c1r0, c1r1, c1r2, 
            c2r0, c2r1, c2r2,
        ))
    }
}

impl<S> Matrix3<S> where S: ScalarFloat {
    /// Compute a rotation matrix about the x-axis by an angle `angle`.
    pub fn from_angle_x<A: Into<Radians<S>>>(angle: A) -> Matrix3<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Matrix3::new(
            S::one(),   S::zero(), S::zero(),
            S::zero(),  cos_angle, sin_angle,
            S::zero(), -sin_angle, cos_angle,
        )
    }

    /// Compute a rotation matrix about the y-axis by an angle `angle`.
    pub fn from_angle_y<A: Into<Radians<S>>>(angle: A) -> Matrix3<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Matrix3::new(
            cos_angle, S::zero(), -sin_angle,
            S::zero(), S::one(),   S::zero(),
            sin_angle, S::zero(),  cos_angle,
        )
    }

    /// Compute a rotation matrix about the z-axis by an angle `angle`.
    pub fn from_angle_z<A: Into<Radians<S>>>(angle: A) -> Matrix3<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());

        Matrix3::new(
             cos_angle, sin_angle, S::zero(),
            -sin_angle, cos_angle, S::zero(),
             S::zero(), S::zero(), S::one(),
        )
    }

    pub fn from_axis_angle<A: Into<Radians<S>>>(axis: Vector3<S>, angle: A) -> Matrix3<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());
        let one_minus_cos_angle = S::one() - cos_angle;

        Matrix3::new(
            one_minus_cos_angle * axis.x * axis.x + cos_angle,
            one_minus_cos_angle * axis.x * axis.y + sin_angle * axis.z,
            one_minus_cos_angle * axis.x * axis.z - sin_angle * axis.y,

            one_minus_cos_angle * axis.x * axis.y - sin_angle * axis.z,
            one_minus_cos_angle * axis.y * axis.y + cos_angle,
            one_minus_cos_angle * axis.y * axis.z + sin_angle * axis.x,

            one_minus_cos_angle * axis.x * axis.z + sin_angle * axis.y,
            one_minus_cos_angle * axis.y * axis.z - sin_angle * axis.x,
            one_minus_cos_angle * axis.z * axis.z + cos_angle,
        )
    }
}

impl<S> Array for Matrix3<S> where S: Scalar {
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
    fn from_value(value: Self::Element) -> Self {
        Matrix3::new(
            value, value, value, 
            value, value, value, 
            value, value, value
        )
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

impl<S> Sum for Matrix3<S> where S: Scalar {
    #[inline]
    fn sum(&self) -> S {
        self.c0r0 + self.c1r0 + self.c2r0 +
        self.c0r1 + self.c1r1 + self.c2r1 +
        self.c0r2 + self.c1r2 + self.c2r2
    }
}

impl<S> Product for Matrix3<S> where S: Scalar {
    #[inline]
    fn product(&self) -> S {
        self.c0r0 * self.c1r0 * self.c2r0 *
        self.c0r1 * self.c1r1 * self.c2r1 *
        self.c0r2 * self.c1r2 * self.c2r2
    }
}

impl<S> Matrix for Matrix3<S> where S: Scalar {
    type Element = S;
    type Row = Vector3<S>;
    type Column = Vector3<S>;
    type Transpose = Matrix3<S>;

    fn row(&self, r: usize) -> Self::Row {
        Vector3::new(self[0][r], self[1][r], self[2][r])
    }
    
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
    
    fn swap_elements(&mut self, a: (usize, usize), b: (usize, usize)) {
        let element_a = self[a.0][a.1];
        self[a.0][a.1] = self[b.0][b.1];
        self[b.0][b.1] = element_a;
    }
    
    fn transpose(&self) -> Self::Transpose {
        Matrix3::new(
            self.c0r0, self.c1r0, self.c2r0,
            self.c0r1, self.c1r1, self.c2r1,
            self.c0r2, self.c1r2, self.c2r2
        )
    }
}

impl<S> From<[[S; 3]; 3]> for Matrix3<S> where S: Scalar {
    #[inline]
    fn from(m: [[S; 3]; 3]) -> Matrix3<S> {
        Matrix3::new(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2])
    }
}

impl<'a, S> From<&'a [[S; 3]; 3]> for &'a Matrix3<S> where S: Scalar {
    #[inline]
    fn from(m: &'a [[S; 3]; 3]) -> &'a Matrix3<S> {
        unsafe { mem::transmute(m) }
    }    
}

impl<S> From<[S; 9]> for Matrix3<S> where S: Scalar {
    #[inline]
    fn from(m: [S; 9]) -> Matrix3<S> {
        Matrix3::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8])
    }
}

impl<'a, S> From<&'a [S; 9]> for &'a Matrix3<S> where S: Scalar {
    #[inline]
    fn from(m: &'a [S; 9]) -> &'a Matrix3<S> {
        unsafe { mem::transmute(m) }
    }
}

impl<S> From<Matrix2<S>> for Matrix3<S> where S: Scalar {
    #[inline]
    fn from(m: Matrix2<S>) -> Matrix3<S> {
        Matrix3::new(
            m.c0r0,    m.c0r1,    S::zero(),
            m.c1r0,    m.c1r1,    S::zero(),
            S::zero(), S::zero(), S::one()
        )
    }
}

impl<S> From<&Matrix2<S>> for Matrix3<S> where S: Scalar {
    #[inline]
    fn from(m: &Matrix2<S>) -> Matrix3<S> {
        Matrix3::new(
            m.c0r0,    m.c0r1,    S::zero(),
            m.c1r0,    m.c1r1,    S::zero(),
            S::zero(), S::zero(), S::one()
        )
    }
}

impl<S> fmt::Display for Matrix3<S> where S: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, 
            "\n[{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}]", 
            self.c0r0, self.c1r0, self.c2r0,
            self.c0r1, self.c1r1, self.c2r1,
            self.c0r2, self.c1r2, self.c2r2,
        )
    }
}

impl<S> AsRef<[S; 9]> for Matrix3<S> {
    fn as_ref(&self) -> &[S; 9] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsRef<[[S; 3]; 3]> for Matrix3<S> {
    fn as_ref(&self) -> &[[S; 3]; 3] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsRef<[Vector3<S>; 3]> for Matrix3<S> {
    fn as_ref(&self) -> &[Vector3<S>; 3] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[S; 9]> for Matrix3<S> {
    fn as_mut(&mut self) -> &mut [S; 9] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[[S; 3]; 3]> for Matrix3<S> {
    fn as_mut(&mut self) -> &mut [[S; 3];3 ] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[Vector3<S>; 3]> for Matrix3<S> {
    fn as_mut(&mut self) -> &mut [Vector3<S>; 3] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> ops::Index<usize> for Matrix3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[Vector3<S>; 3] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::IndexMut<usize> for Matrix3<S> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Vector3<S> {
        let v: &mut [Vector3<S>; 3] = self.as_mut();
        &mut v[index]
    }
}

impl<S> Zero for Matrix3<S> where S: Scalar {
    fn zero() -> Matrix3<S> {
        let zero = S::zero();
        Matrix3::new(zero, zero, zero, zero, zero, zero, zero, zero, zero)
    }

    fn is_zero(&self) -> bool {
        let zero = S::zero();
        self.c0r0 == zero && self.c0r1 == zero && self.c0r2 == zero &&
        self.c1r0 == zero && self.c1r1 == zero && self.c1r2 == zero &&
        self.c2r0 == zero && self.c2r1 == zero && self.c2r2 == zero
    }
}

impl<S> One for Matrix3<S> where S: Scalar {
    fn one() -> Matrix3<S> {
        let zero = S::zero();
        let one = S::one();
        Matrix3::new(one, zero, zero, zero, one, zero, zero, zero, one)
    }
}

impl<S> ops::Add<Matrix3<S>> for Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

    fn add(self, other: Matrix3<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Add<&Matrix3<S>> for Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

    fn add(self, other: &Matrix3<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Add<Matrix3<S>> for &Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

    fn add(self, other: Matrix3<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<'a, 'b, S> ops::Add<&'a Matrix3<S>> for &'b Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

    fn add(self, other: &'a Matrix3<S>) -> Self::Output {
        let c0r0 = self.c0r0 + other.c0r0;
        let c0r1 = self.c0r1 + other.c0r1;
        let c0r2 = self.c0r2 + other.c0r2;

        let c1r0 = self.c1r0 + other.c1r0;
        let c1r1 = self.c1r1 + other.c1r1;
        let c1r2 = self.c1r2 + other.c1r2;

        let c2r0 = self.c2r0 + other.c2r0;
        let c2r1 = self.c2r1 + other.c2r1;
        let c2r2 = self.c2r2 + other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Sub<Matrix3<S>> for Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

    fn sub(self, other: Matrix3<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Sub<&Matrix3<S>> for Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

    fn sub(self, other: &Matrix3<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Sub<Matrix3<S>> for &Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

    fn sub(self, other: Matrix3<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<'a, 'b, S> ops::Sub<&'a Matrix3<S>> for &'b Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

    fn sub(self, other: &'a Matrix3<S>) -> Self::Output {
        let c0r0 = self.c0r0 - other.c0r0;
        let c0r1 = self.c0r1 - other.c0r1;
        let c0r2 = self.c0r2 - other.c0r2;

        let c1r0 = self.c1r0 - other.c1r0;
        let c1r1 = self.c1r1 - other.c1r1;
        let c1r2 = self.c1r2 - other.c1r2;

        let c2r0 = self.c2r0 - other.c2r0;
        let c2r1 = self.c2r1 - other.c2r1;
        let c2r2 = self.c2r2 - other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Mul<&Matrix3<S>> for Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

    fn mul(self, other: &Matrix3<S>) -> Self::Output {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Matrix3<S>> for &'b Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

    fn mul(self, other: &'a Matrix3<S>) -> Matrix3<S> {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Mul<Matrix3<S>> for Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

    fn mul(self, other: Matrix3<S>) -> Matrix3<S> {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Mul<Matrix3<S>> for &Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

    fn mul(self, other: Matrix3<S>) -> Matrix3<S> {
        let c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2;
        let c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2;
        let c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2;

        let c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2;
        let c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2;
        let c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2;

        let c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2;
        let c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2;
        let c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2;

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Mul<S> for Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

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

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Mul<S> for &Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

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

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Mul<Vector3<S>> for Matrix3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn mul(self, other: Vector3<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y + self.c2r0 * other.z;
        let y = self.c0r1 * other.x + self.c1r1 * other.y + self.c2r1 * other.z;
        let z = self.c0r2 * other.x + self.c1r2 * other.y + self.c2r2 * other.z;

        Vector3::new(x, y, z)
    }
}

impl<S> ops::Mul<&Vector3<S>> for Matrix3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn mul(self, other: &Vector3<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y + self.c2r0 * other.z;
        let y = self.c0r1 * other.x + self.c1r1 * other.y + self.c2r1 * other.z;
        let z = self.c0r2 * other.x + self.c1r2 * other.y + self.c2r2 * other.z;

        Vector3::new(x, y, z)
    }
}

impl<S> ops::Mul<Vector3<S>> for &Matrix3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn mul(self, other: Vector3<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y + self.c2r0 * other.z;
        let y = self.c0r1 * other.x + self.c1r1 * other.y + self.c2r1 * other.z;
        let z = self.c0r2 * other.x + self.c1r2 * other.y + self.c2r2 * other.z;

        Vector3::new(x, y, z)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Vector3<S>> for &'b Matrix3<S> where S: Scalar {
    type Output = Vector3<S>;

    fn mul(self, other: &'a Vector3<S>) -> Self::Output {
        let x = self.c0r0 * other.x + self.c1r0 * other.y + self.c2r0 * other.z;
        let y = self.c0r1 * other.x + self.c1r1 * other.y + self.c2r1 * other.z;
        let z = self.c0r2 * other.x + self.c1r2 * other.y + self.c2r2 * other.z;

        Vector3::new(x, y, z)
    }
}

impl<S> ops::Div<S> for Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

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

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Div<S> for &Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

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

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Neg for Matrix3<S> where S: ScalarSigned {
    type Output = Matrix3<S>;

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

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Neg for &Matrix3<S> where S: ScalarSigned {
    type Output = Matrix3<S>;

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

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Rem<S> for Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

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

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
    }
}

impl<S> ops::Rem<S> for &Matrix3<S> where S: Scalar {
    type Output = Matrix3<S>;

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

        Matrix3::new(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)     
    }
}

impl<S> ops::AddAssign<Matrix3<S>> for Matrix3<S> where S: Scalar {
    fn add_assign(&mut self, other: Matrix3<S>) {
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

impl<S> ops::AddAssign<&Matrix3<S>> for Matrix3<S> where S: Scalar {
    fn add_assign(&mut self, other: &Matrix3<S>) {
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

impl<S> ops::SubAssign<Matrix3<S>> for Matrix3<S> where S: Scalar {
    fn sub_assign(&mut self, other: Matrix3<S>) {
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

impl<S> ops::SubAssign<&Matrix3<S>> for Matrix3<S> where S: Scalar {
    fn sub_assign(&mut self, other: &Matrix3<S>) {
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

impl<S> ops::MulAssign<S> for Matrix3<S> where S: Scalar {
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

impl<S> ops::DivAssign<S> for Matrix3<S> where S: Scalar {
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

impl<S> ops::RemAssign<S> for Matrix3<S> where S: Scalar {
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

impl<S> Lerp<Matrix3<S>> for Matrix3<S> where S: Scalar {
    type Scalar = S;
    type Output = Matrix3<S>;

    fn lerp(self, other: Matrix3<S>, amount: Self::Scalar) -> Matrix3<S> {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<&Matrix3<S>> for Matrix3<S> where S: Scalar {
    type Scalar = S;
    type Output = Matrix3<S>;

    fn lerp(self, other: &Matrix3<S>, amount: Self::Scalar) -> Matrix3<S> {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<Matrix3<S>> for &Matrix3<S> where S: Scalar {
    type Scalar = S;
    type Output = Matrix3<S>;

    fn lerp(self, other: Matrix3<S>, amount: Self::Scalar) -> Matrix3<S> {
        self + ((other - self) * amount)
    }
}

impl<'a, 'b, S> Lerp<&'a Matrix3<S>> for &'b Matrix3<S> where S: Scalar {
    type Scalar = S;
    type Output = Matrix3<S>;

    fn lerp(self, other: &'a Matrix3<S>, amount: Self::Scalar) -> Matrix3<S> {
        self + ((other - self) * amount)
    }
}

impl<S> approx::AbsDiffEq for Matrix3<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Matrix3<S> where S: ScalarFloat {
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

impl<S> approx::UlpsEq for Matrix3<S> where S: ScalarFloat {
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

impl<S> SquareMatrix for Matrix3<S> where S: ScalarFloat {
    type ColumnRow = Vector3<S>;

    #[inline]
    fn from_value(value: Self::Element) -> Self {
        Matrix3::new(
            value,     S::zero(), S::zero(),
            S::zero(), value,     S::zero(),
            S::zero(), S::zero(), value,
        )
    }
    
    #[inline]
    fn from_diagonal(value: Self::ColumnRow) -> Self {
        Matrix3::new(
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
        ulps_eq!(self, &Self::one())
    }
}

impl<S> SkewSymmetricMatrix for Matrix3<S> where S: ScalarFloat {
    fn is_skew_symmetric(&self) -> bool {
        ulps_eq!(self.c0r1, -self.c1r0) && ulps_eq!(self.c1r0, -self.c0r1) &&
        ulps_eq!(self.c0r2, -self.c2r0) && ulps_eq!(self.c2r0, -self.c0r2) &&
        ulps_eq!(self.c1r2, -self.c2r1) && ulps_eq!(self.c2r1, -self.c1r2)
    }
}

impl<S> InvertibleSquareMatrix for Matrix3<S> where S: ScalarFloat {
    fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == S::zero() {
            // A matrix with zero determinant has no inverse.
            None
        } else {
            let inv_det = S::one() / det;

            Some(Matrix3::new(
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

impl<S: Scalar> iter::Sum<Matrix3<S>> for Matrix3<S> {
    #[inline]
    fn sum<I: Iterator<Item = Matrix3<S>>>(iter: I) -> Matrix3<S> {
        iter.fold(Matrix3::<S>::zero(), ops::Add::add)
    }
}

impl<'a, S: 'a + Scalar> iter::Sum<&'a Matrix3<S>> for Matrix3<S> {
    #[inline]
    fn sum<I: Iterator<Item = &'a Matrix3<S>>>(iter: I) -> Matrix3<S> {
        iter.fold(Matrix3::<S>::zero(), ops::Add::add)
    }
}

impl<S: Scalar> iter::Product<Matrix3<S>> for Matrix3<S> {
    #[inline]
    fn product<I: Iterator<Item = Matrix3<S>>>(iter: I) -> Matrix3<S> {
        iter.fold(Matrix3::<S>::one(), ops::Mul::mul)
    }
}

impl<'a, S: 'a + Scalar> iter::Product<&'a Matrix3<S>> for Matrix3<S> {
    #[inline]
    fn product<I: Iterator<Item = &'a Matrix3<S>>>(iter: I) -> Matrix3<S> {
        iter.fold(Matrix3::<S>::one(), ops::Mul::mul)
    }
}



/// The `Matrix4` type represents 4x4 matrices in column-major order.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C)]
pub struct Matrix4<S> {
    /// Column 0 of the matrix.
    pub c0r0: S, pub c0r1: S, pub c0r2: S, pub c0r3: S,
    /// Column 1 of the matrix.
    pub c1r0: S, pub c1r1: S, pub c1r2: S, pub c1r3: S,
    /// Column 2 of the matrix.
    pub c2r0: S, pub c2r1: S, pub c2r2: S, pub c2r3: S,
    /// Column 3 of the matrix.
    pub c3r0: S, pub c3r1: S, pub c3r2: S, pub c3r3: S,
}

impl<S> Matrix4<S> {
    pub const fn new(
        c0r0: S, c0r1: S, c0r2: S, c0r3: S,
        c1r0: S, c1r1: S, c1r2: S, c1r3: S,
        c2r0: S, c2r1: S, c2r2: S, c2r3: S,
        c3r0: S, c3r1: S, c3r2: S, c3r3: S) -> Matrix4<S> {

        Matrix4 {
            c0r0: c0r0, c0r1: c0r1, c0r2: c0r2, c0r3: c0r3,
            c1r0: c1r0, c1r1: c1r1, c1r2: c1r2, c1r3: c1r3,
            c2r0: c2r0, c2r1: c2r1, c2r2: c2r2, c2r3: c2r3,
            c3r0: c3r0, c3r1: c3r1, c3r2: c3r2, c3r3: c3r3,
        }
    }

    /// Construct a 4x4 matrix from column vectors.
    pub fn from_columns(c0: Vector4<S>, c1: Vector4<S>, c2: Vector4<S>, c3: Vector4<S>) -> Matrix4<S> {
        Matrix4 {
            c0r0: c0.x, c0r1: c0.y, c0r2: c0.z, c0r3: c0.w,
            c1r0: c1.x, c1r1: c1.y, c1r2: c1.z, c1r3: c1.w,
            c2r0: c2.x, c2r1: c2.y, c2r2: c2.z, c2r3: c2.w,
            c3r0: c3.x, c3r1: c3.y, c3r2: c3.z, c3r3: c3.w,
        }
    }

    /// Map an operation on the elements of a matrix, returning a matrix whose elements
    /// are elements of the new underlying type.
    pub fn map<T, F>(self, mut op: F) -> Matrix4<T> where F: FnMut(S) -> T {
        Matrix4 {
            c0r0: op(self.c0r0), c1r0: op(self.c1r0), c2r0: op(self.c2r0), c3r0: op(self.c3r0),
            c0r1: op(self.c0r1), c1r1: op(self.c1r1), c2r1: op(self.c2r1), c3r1: op(self.c3r1),
            c0r2: op(self.c0r2), c1r2: op(self.c1r2), c2r2: op(self.c2r2), c3r2: op(self.c3r2),
            c0r3: op(self.c0r3), c1r3: op(self.c1r3), c2r3: op(self.c2r3), c3r3: op(self.c3r3),
        }
    }
}

impl<S> Matrix4<S> where S: NumCast + Copy {
    /// Cast a matrix from one type of scalars to another type of scalars.
    pub fn cast<T: NumCast>(&self) -> Option<Matrix4<T>> {
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

        Some(Matrix4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        ))
    }
}

impl<S> Matrix4<S> where S: Scalar {
    /// Create a affine translation matrix.
    #[inline]
    pub fn from_translation(distance: Vector3<S>) -> Matrix4<S> {
        let one = S::one();
        let zero = S::zero();
        Matrix4::new(
            one,        zero,       zero,       zero,
            zero,       one,        zero,       zero,
            zero,       zero,       one,        zero,
            distance.x, distance.y, distance.z, one
        )
    }

    /// Scale a matrix uniformly.
    #[inline]
    pub fn from_scale(value: S) -> Matrix4<S> {
        Matrix4::from_nonuniform_scale(value, value, value)
    }

    /// Scale a matrix in a nonuniform fashion.
    #[inline]
    pub fn from_nonuniform_scale(sx: S, sy: S, sz: S) -> Matrix4<S> {
        let one = S::one();
        let zero = S::zero();
        Matrix4::new(
            sx,   zero, zero, zero,
            zero, sy,   zero, zero,
            zero, zero, sz,   zero,
            zero, zero, zero, one
        )
    }
}

impl<S> Matrix4<S> where S: ScalarFloat {
    /// Create a rotation matrix around the x axis by an angle in `angle` radians/degrees.
    pub fn from_angle_x<A: Into<Radians<S>>>(angle: A) -> Matrix4<S> {
        let (sin_angle, cos_angle) = angle.into().sin_cos();
        let one = S::one();
        let zero = S::zero();

        Matrix4::new(
            one,   zero,      zero,      zero,
            zero,  cos_angle, sin_angle, zero,
            zero, -sin_angle, cos_angle, zero,
            zero,  zero,      zero,      one
        )
    }
        
    /// Create a rotation matrix around the y axis by an angle in `angle` radians/degrees.
    pub fn from_angle_y<A: Into<Radians<S>>>(angle: A) -> Matrix4<S> {
        let (sin_angle, cos_angle) = angle.into().sin_cos();
        let one = S::one();
        let zero = S::zero();

        Matrix4::new(
            cos_angle, zero, -sin_angle, zero,
            zero,      one,   zero,      zero,
            sin_angle, zero,  cos_angle, zero,
            zero,      zero,  zero,      one
        )
    }
    
    /// Create a rotation matrix around the z axis by an angle in `angle` radians/degrees.
    pub fn from_angle_z<A: Into<Radians<S>>>(angle: A) -> Matrix4<S> {
        let (sin_angle, cos_angle) = angle.into().sin_cos();
        let one = S::one();
        let zero = S::zero();
        
        Matrix4::new(
             cos_angle, sin_angle, zero, zero,
            -sin_angle, cos_angle, zero, zero,
             zero,      zero,      one,  zero,
             zero,      zero,      zero, one
        )
    }

    pub fn from_axis_angle<A: Into<Radians<S>>>(axis: Vector3<S>, angle: A) -> Matrix4<S> {
        let (sin_angle, cos_angle) = Radians::sin_cos(angle.into());
        let one_minus_cos_angle = S::one() - cos_angle;

        Matrix4::new(
            one_minus_cos_angle * axis.x * axis.x + cos_angle,
            one_minus_cos_angle * axis.x * axis.y + sin_angle * axis.z,
            one_minus_cos_angle * axis.x * axis.z - sin_angle * axis.y,
            S::zero(),

            one_minus_cos_angle * axis.x * axis.y - sin_angle * axis.z,
            one_minus_cos_angle * axis.y * axis.y + cos_angle,
            one_minus_cos_angle * axis.y * axis.z + sin_angle * axis.x,
            S::zero(),

            one_minus_cos_angle * axis.x * axis.z + sin_angle * axis.y,
            one_minus_cos_angle * axis.y * axis.z - sin_angle * axis.x,
            one_minus_cos_angle * axis.z * axis.z + cos_angle,
            S::zero(),

            S::zero(), S::zero(), S::zero(), S::one(),
        )
    }
}

impl<S> Array for Matrix4<S> where S: Scalar {
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
    fn from_value(value: Self::Element) -> Self {
        Matrix4::new(
            value, value, value, value, 
            value, value, value, value, 
            value, value, value, value, 
            value, value, value, value
        )
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

impl<S> Sum for Matrix4<S> where S: Scalar {
    #[inline]
    fn sum(&self) -> S {
        self.c0r0 + self.c1r0 + self.c2r0 + self.c3r0 +
        self.c0r1 + self.c1r1 + self.c2r1 + self.c3r1 +
        self.c0r2 + self.c1r2 + self.c2r2 + self.c3r2 +
        self.c0r3 + self.c1r3 + self.c2r3 + self.c3r3
    }
}

impl<S> Product for Matrix4<S> where S: Scalar {
    #[inline]
    fn product(&self) -> S {
        self.c0r0 * self.c1r0 * self.c2r0 * self.c3r0 *
        self.c0r1 * self.c1r1 * self.c2r1 * self.c3r1 *
        self.c0r2 * self.c1r2 * self.c2r2 * self.c3r2 *
        self.c0r3 * self.c1r3 * self.c2r3 * self.c3r3
    }
}

impl<S> Matrix for Matrix4<S> where S: Scalar {
    type Element = S;
    type Row = Vector4<S>;
    type Column = Vector4<S>;
    type Transpose = Matrix4<S>;

    fn row(&self, r: usize) -> Self::Row {
        Vector4::new(self[0][r], self[1][r], self[2][r], self[3][r])
    }
    
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
    
    fn swap_elements(&mut self, a: (usize, usize), b: (usize, usize)) {
        let element_a = self[a.0][a.1];
        self[a.0][a.1] = self[b.0][b.1];
        self[b.0][b.1] = element_a;
    }
    
    fn transpose(&self) -> Self::Transpose {
        Matrix4::new(
            self.c0r0, self.c1r0, self.c2r0, self.c3r0,
            self.c0r1, self.c1r1, self.c2r1, self.c3r1, 
            self.c0r2, self.c1r2, self.c2r2, self.c3r2, 
            self.c0r3, self.c1r3, self.c2r3, self.c3r3
        )
    }
}

impl<S> From<[[S; 4]; 4]> for Matrix4<S> where S: Scalar {
    #[inline]
    fn from(m: [[S; 4]; 4]) -> Matrix4<S> {
        Matrix4::new(
            m[0][0], m[0][1], m[0][2], m[0][3], 
            m[1][0], m[1][1], m[1][2], m[1][3],
            m[2][0], m[2][1], m[2][2], m[2][3], 
            m[3][0], m[3][1], m[3][2], m[3][3]
        )
    }
}

impl<'a, S> From<&'a [[S; 4]; 4]> for &'a Matrix4<S> where S: Scalar {
    #[inline]
    fn from(m: &'a [[S; 4]; 4]) -> &'a Matrix4<S> {
        unsafe { mem::transmute(m) }
    }    
}

impl<S> From<[S; 16]> for Matrix4<S> where S: Scalar {
    #[inline]
    fn from(m: [S; 16]) -> Matrix4<S> {
        Matrix4::new(
            m[0],  m[1],  m[2],  m[3], 
            m[4],  m[5],  m[6],  m[7],
            m[8],  m[9],  m[10], m[11], 
            m[12], m[13], m[14], m[15]
        )
    }
}

impl<'a, S> From<&'a [S; 16]> for &'a Matrix4<S> where S: Scalar {
    #[inline]
    fn from(m: &'a [S; 16]) -> &'a Matrix4<S> {
        unsafe { mem::transmute(m) }
    }
}

impl<S> From<Matrix2<S>> for Matrix4<S> where S: Scalar {
    #[inline]
    fn from(m: Matrix2<S>) -> Matrix4<S> {
        let one = S::one();
        let zero = S::zero();
        Matrix4::new(
            m.c0r0, m.c0r1,   zero, zero,
            m.c1r0, m.c1r1,   zero, zero,
      zero, zero, one,  zero,
      zero, zero, zero, one
        )
    }
}

impl<S> From<&Matrix2<S>> for Matrix4<S> where S: Scalar {
    #[inline]
    fn from(m: &Matrix2<S>) -> Matrix4<S> {
        let one = S::one();
        let zero = S::zero();
        Matrix4::new(
            m.c0r0, m.c0r1,   zero, zero,
            m.c1r0, m.c1r1,   zero, zero,
      zero, zero, one,  zero,
      zero, zero, zero, one
        )
    }
}

impl<S> From<Matrix3<S>> for Matrix4<S> where S: Scalar {
    #[inline]
    fn from(m: Matrix3<S>) -> Matrix4<S> {
        let one = S::one();
        let zero = S::zero();
        Matrix4::new(
            m.c0r0, m.c0r1, m.c0r2, zero,
            m.c1r0, m.c1r1, m.c1r2, zero,
            m.c2r0, m.c2r1, m.c2r2, zero,
      zero, zero, zero, one
        )
    }
}

impl<S> From<&Matrix3<S>> for Matrix4<S> where S: Scalar {
    #[inline]
    fn from(m: &Matrix3<S>) -> Matrix4<S> {
        let one = S::one();
        let zero = S::zero();
        Matrix4::new(
            m.c0r0, m.c0r1, m.c0r2, zero,
            m.c1r0, m.c1r1, m.c1r2, zero,
            m.c2r0, m.c2r1, m.c2r2, zero,
          zero, zero,    zero, one
        )
    }
}

impl<S> fmt::Display for Matrix4<S> where S: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, 
            "\n[{:.2}][{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}][{:.2}]\n[{:.2}][{:.2}][{:.2}][{:.2}]", 
            self.c0r0, self.c1r0, self.c2r0, self.c3r0,
            self.c0r1, self.c1r1, self.c2r1, self.c3r1,
            self.c0r2, self.c1r2, self.c2r2, self.c3r2,
            self.c0r3, self.c1r3, self.c2r3, self.c3r3
        )
    }
}

impl<S> AsRef<[S; 16]> for Matrix4<S> where S: Scalar {
    fn as_ref(&self) -> &[S; 16] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsRef<[[S; 4]; 4]> for Matrix4<S> where S: Scalar {
    fn as_ref(&self) -> &[[S; 4]; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsRef<[Vector4<S>; 4]> for Matrix4<S> where S: Scalar {
    fn as_ref(&self) -> &[Vector4<S>; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[S; 16]> for Matrix4<S> where S: Scalar {
    fn as_mut(&mut self) -> &mut [S; 16] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[[S; 4]; 4]> for Matrix4<S> where S: Scalar {
    fn as_mut(&mut self) -> &mut [[S; 4]; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> AsMut<[Vector4<S>; 4]> for Matrix4<S> where S: Scalar {
    fn as_mut(&mut self) -> &mut [Vector4<S>; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl<S> ops::Index<usize> for Matrix4<S> where S: Scalar {
    type Output = Vector4<S>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[Vector4<S>; 4] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::IndexMut<usize> for Matrix4<S> where S: Scalar {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Vector4<S> {
        let v: &mut [Vector4<S>; 4] = self.as_mut();
        &mut v[index]
    }
}

impl<S> Zero for Matrix4<S> where S: Scalar {
    fn zero() -> Matrix4<S> {
        let zero = S::zero();
        Matrix4::new(
            zero, zero, zero, zero, zero, zero, zero, zero, 
            zero, zero, zero, zero, zero, zero, zero, zero
        )
    }

    fn is_zero(&self) -> bool {
        let zero = S::zero();
        self.c0r0 == zero && self.c0r1 == zero && self.c0r2 == zero && self.c0r3 == zero &&
        self.c1r0 == zero && self.c1r1 == zero && self.c1r2 == zero && self.c1r3 == zero &&
        self.c2r0 == zero && self.c2r1 == zero && self.c2r2 == zero && self.c2r3 == zero &&
        self.c3r0 == zero && self.c3r1 == zero && self.c3r2 == zero && self.c3r3 == zero
    }
}

impl<S> One for Matrix4<S> where S: Scalar {
    fn one() -> Matrix4<S> {
        let one = S::one();
        let zero = S::zero();
        Matrix4::new(
             one, zero, zero, zero, 
            zero,  one, zero, zero, 
            zero, zero,  one, zero, 
            zero, zero, zero,  one
        )
    }
}

impl<S> ops::Add<Matrix4<S>> for Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

    fn add(self, other: Matrix4<S>) -> Self::Output {
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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Add<&Matrix4<S>> for Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

    fn add(self, other: &Matrix4<S>) -> Self::Output {
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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Add<Matrix4<S>> for &Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

    fn add(self, other: Matrix4<S>) -> Self::Output {
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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<'a, 'b, S> ops::Add<&'a Matrix4<S>> for &'b Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

    fn add(self, other: &'a Matrix4<S>) -> Self::Output {
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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Sub<Matrix4<S>> for Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

    fn sub(self, other: Matrix4<S>) -> Self::Output {
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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Sub<&Matrix4<S>> for Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

    fn sub(self, other: &Matrix4<S>) -> Self::Output {
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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Sub<Matrix4<S>> for &Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

    fn sub(self, other: Matrix4<S>) -> Self::Output {
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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<'a, 'b, S> ops::Sub<&'a Matrix4<S>> for &'b Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

    fn sub(self, other: &'a Matrix4<S>) -> Self::Output {
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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Mul<Vector4<S>> for Matrix4<S> where S: Scalar {
    type Output = Vector4<S>;

    fn mul(self, other: Vector4<S>) -> Self::Output {
        let x = self.c0r0 * other[0] + self.c1r0 * other[1] + self.c2r0 * other[2] + self.c3r0 * other[3];
        let y = self.c0r1 * other[0] + self.c1r1 * other[1] + self.c2r1 * other[2] + self.c3r1 * other[3];
        let z = self.c0r2 * other[0] + self.c1r2 * other[1] + self.c2r2 * other[2] + self.c3r2 * other[3];
        let w = self.c0r3 * other[0] + self.c1r3 * other[1] + self.c2r3 * other[2] + self.c3r3 * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<S> ops::Mul<&Vector4<S>> for Matrix4<S> where S: Scalar {
    type Output = Vector4<S>;

    fn mul(self, other: &Vector4<S>) -> Self::Output {
        let x = self.c0r0 * other[0] + self.c1r0 * other[1] + self.c2r0 * other[2] + self.c3r0 * other[3];
        let y = self.c0r1 * other[0] + self.c1r1 * other[1] + self.c2r1 * other[2] + self.c3r1 * other[3];
        let z = self.c0r2 * other[0] + self.c1r2 * other[1] + self.c2r2 * other[2] + self.c3r2 * other[3];
        let w = self.c0r3 * other[0] + self.c1r3 * other[1] + self.c2r3 * other[2] + self.c3r3 * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<S> ops::Mul<Vector4<S>> for &Matrix4<S> where S: Scalar {
    type Output = Vector4<S>;

    fn mul(self, other: Vector4<S>) -> Self::Output {
        let x = self.c0r0 * other[0] + self.c1r0 * other[1] + self.c2r0 * other[2] + self.c3r0 * other[3];
        let y = self.c0r1 * other[0] + self.c1r1 * other[1] + self.c2r1 * other[2] + self.c3r1 * other[3];
        let z = self.c0r2 * other[0] + self.c1r2 * other[1] + self.c2r2 * other[2] + self.c3r2 * other[3];
        let w = self.c0r3 * other[0] + self.c1r3 * other[1] + self.c2r3 * other[2] + self.c3r3 * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<'a, 'b, S> ops::Mul<&'a Vector4<S>> for &'b Matrix4<S> where S: Scalar {
    type Output = Vector4<S>;

    fn mul(self, other: &'a Vector4<S>) -> Self::Output {
        let x = self.c0r0 * other[0] + self.c1r0 * other[1] + self.c2r0 * other[2] + self.c3r0 * other[3];
        let y = self.c0r1 * other[0] + self.c1r1 * other[1] + self.c2r1 * other[2] + self.c3r1 * other[3];
        let z = self.c0r2 * other[0] + self.c1r2 * other[1] + self.c2r2 * other[2] + self.c3r2 * other[3];
        let w = self.c0r3 * other[0] + self.c1r3 * other[1] + self.c2r3 * other[2] + self.c3r3 * other[3];
        
        Vector4::new(x, y, z, w)
    }
}

impl<S> ops::Mul<Matrix4<S>> for Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

    fn mul(self, other: Matrix4<S>) -> Self::Output {
        let mut m = Matrix4::zero();

        m.c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2 + self.c3r0 * other.c0r3;
        m.c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2 + self.c3r1 * other.c0r3;
        m.c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2 + self.c3r2 * other.c0r3;
        m.c0r3 = self.c0r3 * other.c0r0 + self.c1r3 * other.c0r1 + self.c2r3 * other.c0r2 + self.c3r3 * other.c0r3;

        m.c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2 + self.c3r0 * other.c1r3;
        m.c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2 + self.c3r1 * other.c1r3;
        m.c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2 + self.c3r2 * other.c1r3;
        m.c1r3 = self.c0r3 * other.c1r0 + self.c1r3 * other.c1r1 + self.c2r3 * other.c1r2 + self.c3r3 * other.c1r3;

        m.c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2 + self.c3r0 * other.c2r3;
        m.c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2 + self.c3r1 * other.c2r3;
        m.c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2 + self.c3r2 * other.c2r3;
        m.c2r3 = self.c0r3 * other.c2r0 + self.c1r3 * other.c2r1 + self.c2r3 * other.c2r2 + self.c3r3 * other.c2r3;

        m.c3r0 = self.c0r0 * other.c3r0 + self.c1r0 * other.c3r1 + self.c2r0 * other.c3r2 + self.c3r0 * other.c3r3;
        m.c3r1 = self.c0r1 * other.c3r0 + self.c1r1 * other.c3r1 + self.c2r1 * other.c3r2 + self.c3r1 * other.c3r3;
        m.c3r2 = self.c0r2 * other.c3r0 + self.c1r2 * other.c3r1 + self.c2r2 * other.c3r2 + self.c3r2 * other.c3r3;
        m.c3r3 = self.c0r3 * other.c3r0 + self.c1r3 * other.c3r1 + self.c2r3 * other.c3r2 + self.c3r3 * other.c3r3;

        m
    }
}

impl<S> ops::Mul<&Matrix4<S>> for Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

    fn mul(self, other: &Matrix4<S>) -> Self::Output {
        let mut m = Matrix4::zero();

        m.c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2 + self.c3r0 * other.c0r3;
        m.c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2 + self.c3r1 * other.c0r3;
        m.c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2 + self.c3r2 * other.c0r3;
        m.c0r3 = self.c0r3 * other.c0r0 + self.c1r3 * other.c0r1 + self.c2r3 * other.c0r2 + self.c3r3 * other.c0r3;

        m.c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2 + self.c3r0 * other.c1r3;
        m.c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2 + self.c3r1 * other.c1r3;
        m.c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2 + self.c3r2 * other.c1r3;
        m.c1r3 = self.c0r3 * other.c1r0 + self.c1r3 * other.c1r1 + self.c2r3 * other.c1r2 + self.c3r3 * other.c1r3;

        m.c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2 + self.c3r0 * other.c2r3;
        m.c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2 + self.c3r1 * other.c2r3;
        m.c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2 + self.c3r2 * other.c2r3;
        m.c2r3 = self.c0r3 * other.c2r0 + self.c1r3 * other.c2r1 + self.c2r3 * other.c2r2 + self.c3r3 * other.c2r3;

        m.c3r0 = self.c0r0 * other.c3r0 + self.c1r0 * other.c3r1 + self.c2r0 * other.c3r2 + self.c3r0 * other.c3r3;
        m.c3r1 = self.c0r1 * other.c3r0 + self.c1r1 * other.c3r1 + self.c2r1 * other.c3r2 + self.c3r1 * other.c3r3;
        m.c3r2 = self.c0r2 * other.c3r0 + self.c1r2 * other.c3r1 + self.c2r2 * other.c3r2 + self.c3r2 * other.c3r3;
        m.c3r3 = self.c0r3 * other.c3r0 + self.c1r3 * other.c3r1 + self.c2r3 * other.c3r2 + self.c3r3 * other.c3r3;

        m
    }
}

impl<S> ops::Mul<Matrix4<S>> for &Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

    fn mul(self, other: Matrix4<S>) -> Self::Output {
        let mut m = Matrix4::zero();

        m.c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2 + self.c3r0 * other.c0r3;
        m.c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2 + self.c3r1 * other.c0r3;
        m.c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2 + self.c3r2 * other.c0r3;
        m.c0r3 = self.c0r3 * other.c0r0 + self.c1r3 * other.c0r1 + self.c2r3 * other.c0r2 + self.c3r3 * other.c0r3;

        m.c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2 + self.c3r0 * other.c1r3;
        m.c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2 + self.c3r1 * other.c1r3;
        m.c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2 + self.c3r2 * other.c1r3;
        m.c1r3 = self.c0r3 * other.c1r0 + self.c1r3 * other.c1r1 + self.c2r3 * other.c1r2 + self.c3r3 * other.c1r3;

        m.c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2 + self.c3r0 * other.c2r3;
        m.c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2 + self.c3r1 * other.c2r3;
        m.c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2 + self.c3r2 * other.c2r3;
        m.c2r3 = self.c0r3 * other.c2r0 + self.c1r3 * other.c2r1 + self.c2r3 * other.c2r2 + self.c3r3 * other.c2r3;

        m.c3r0 = self.c0r0 * other.c3r0 + self.c1r0 * other.c3r1 + self.c2r0 * other.c3r2 + self.c3r0 * other.c3r3;
        m.c3r1 = self.c0r1 * other.c3r0 + self.c1r1 * other.c3r1 + self.c2r1 * other.c3r2 + self.c3r1 * other.c3r3;
        m.c3r2 = self.c0r2 * other.c3r0 + self.c1r2 * other.c3r1 + self.c2r2 * other.c3r2 + self.c3r2 * other.c3r3;
        m.c3r3 = self.c0r3 * other.c3r0 + self.c1r3 * other.c3r1 + self.c2r3 * other.c3r2 + self.c3r3 * other.c3r3;

        m
    }
}

impl<'a, 'b, S> ops::Mul<&'a Matrix4<S>> for &'b Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

    fn mul(self, other: &'a Matrix4<S>) -> Self::Output {
        let mut m = Matrix4::zero();

        m.c0r0 = self.c0r0 * other.c0r0 + self.c1r0 * other.c0r1 + self.c2r0 * other.c0r2 + self.c3r0 * other.c0r3;
        m.c0r1 = self.c0r1 * other.c0r0 + self.c1r1 * other.c0r1 + self.c2r1 * other.c0r2 + self.c3r1 * other.c0r3;
        m.c0r2 = self.c0r2 * other.c0r0 + self.c1r2 * other.c0r1 + self.c2r2 * other.c0r2 + self.c3r2 * other.c0r3;
        m.c0r3 = self.c0r3 * other.c0r0 + self.c1r3 * other.c0r1 + self.c2r3 * other.c0r2 + self.c3r3 * other.c0r3;

        m.c1r0 = self.c0r0 * other.c1r0 + self.c1r0 * other.c1r1 + self.c2r0 * other.c1r2 + self.c3r0 * other.c1r3;
        m.c1r1 = self.c0r1 * other.c1r0 + self.c1r1 * other.c1r1 + self.c2r1 * other.c1r2 + self.c3r1 * other.c1r3;
        m.c1r2 = self.c0r2 * other.c1r0 + self.c1r2 * other.c1r1 + self.c2r2 * other.c1r2 + self.c3r2 * other.c1r3;
        m.c1r3 = self.c0r3 * other.c1r0 + self.c1r3 * other.c1r1 + self.c2r3 * other.c1r2 + self.c3r3 * other.c1r3;

        m.c2r0 = self.c0r0 * other.c2r0 + self.c1r0 * other.c2r1 + self.c2r0 * other.c2r2 + self.c3r0 * other.c2r3;
        m.c2r1 = self.c0r1 * other.c2r0 + self.c1r1 * other.c2r1 + self.c2r1 * other.c2r2 + self.c3r1 * other.c2r3;
        m.c2r2 = self.c0r2 * other.c2r0 + self.c1r2 * other.c2r1 + self.c2r2 * other.c2r2 + self.c3r2 * other.c2r3;
        m.c2r3 = self.c0r3 * other.c2r0 + self.c1r3 * other.c2r1 + self.c2r3 * other.c2r2 + self.c3r3 * other.c2r3;

        m.c3r0 = self.c0r0 * other.c3r0 + self.c1r0 * other.c3r1 + self.c2r0 * other.c3r2 + self.c3r0 * other.c3r3;
        m.c3r1 = self.c0r1 * other.c3r0 + self.c1r1 * other.c3r1 + self.c2r1 * other.c3r2 + self.c3r1 * other.c3r3;
        m.c3r2 = self.c0r2 * other.c3r0 + self.c1r2 * other.c3r1 + self.c2r2 * other.c3r2 + self.c3r2 * other.c3r3;
        m.c3r3 = self.c0r3 * other.c3r0 + self.c1r3 * other.c3r1 + self.c2r3 * other.c3r2 + self.c3r3 * other.c3r3;

        m
    }
}

impl<S> ops::Mul<S> for Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Mul<S> for &Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Div<S> for Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Div<S> for &Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Neg for Matrix4<S> where S: ScalarSigned {
    type Output = Matrix4<S>;

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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Neg for &Matrix4<S> where S: ScalarSigned {
    type Output = Matrix4<S>;

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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Rem<S> for Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::Rem<S> for &Matrix4<S> where S: Scalar {
    type Output = Matrix4<S>;

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

        Matrix4::new(
            c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, 
            c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3
        )
    }
}

impl<S> ops::AddAssign<Matrix4<S>> for Matrix4<S> where S: Scalar {
    fn add_assign(&mut self, other: Matrix4<S>) {
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

impl<S> ops::AddAssign<&Matrix4<S>> for Matrix4<S> where S: Scalar {
    fn add_assign(&mut self, other: &Matrix4<S>) {
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

impl<S> ops::SubAssign<Matrix4<S>> for Matrix4<S> where S: Scalar {
    fn sub_assign(&mut self, other: Matrix4<S>) {
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

impl<S> ops::SubAssign<&Matrix4<S>> for Matrix4<S> where S: Scalar {
    fn sub_assign(&mut self, other: &Matrix4<S>) {
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

impl<S> ops::MulAssign<S> for Matrix4<S> where S: Scalar {
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

impl<S> ops::DivAssign<S> for Matrix4<S> where S: Scalar {
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

impl<S> ops::RemAssign<S> for Matrix4<S> where S: Scalar {
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

impl<S> Lerp<Matrix4<S>> for Matrix4<S> where S: Scalar {
    type Scalar = S;
    type Output = Matrix4<S>;

    fn lerp(self, other: Matrix4<S>, amount: S) -> Matrix4<S> {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<&Matrix4<S>> for Matrix4<S> where S: Scalar {
    type Scalar = S;
    type Output = Matrix4<S>;

    fn lerp(self, other: &Matrix4<S>, amount: S) -> Matrix4<S> {
        self + ((other - self) * amount)
    }
}

impl<S> Lerp<Matrix4<S>> for &Matrix4<S> where S: Scalar {
    type Scalar = S;
    type Output = Matrix4<S>;

    fn lerp(self, other: Matrix4<S>, amount: S) -> Matrix4<S> {
        self + ((other - self) * amount)
    }
}

impl<'a, 'b, S> Lerp<&'a Matrix4<S>> for &'b Matrix4<S> where S: Scalar {
    type Scalar = S;
    type Output = Matrix4<S>;

    fn lerp(self, other: &'a Matrix4<S>, amount: S) -> Matrix4<S> {
        self + ((other - self) * amount)
    }
}

impl<S> approx::AbsDiffEq for Matrix4<S> where S: ScalarFloat {
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

impl<S> approx::RelativeEq for Matrix4<S> where S: ScalarFloat {
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

impl<S> approx::UlpsEq for Matrix4<S> where S: ScalarFloat {
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

impl<S> SquareMatrix for Matrix4<S> where S: ScalarFloat {
    type ColumnRow = Vector4<S>;

    #[inline]
    fn from_value(value: Self::Element) -> Self {
        Matrix4::new(
            value,     S::zero(), S::zero(), S::zero(),
            S::zero(), value,     S::zero(), S::zero(),
            S::zero(), S::zero(), value,     S::zero(),
            S::zero(), S::zero(), S::zero(), value
        )
    }
    
    #[inline]
    fn from_diagonal(value: Self::ColumnRow) -> Self {
        Matrix4::new(
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
        ulps_eq!(self, &Self::one())
    }
}

impl<S> SkewSymmetricMatrix for Matrix4<S> where S: ScalarFloat {
    fn is_skew_symmetric(&self) -> bool {
        ulps_eq!(self.c0r1, -self.c1r0) && ulps_eq!(self.c1r0, -self.c0r1) &&
        ulps_eq!(self.c0r2, -self.c2r0) && ulps_eq!(self.c2r0, -self.c0r2) &&
        ulps_eq!(self.c1r2, -self.c2r1) && ulps_eq!(self.c2r1, -self.c1r2) &&
        ulps_eq!(self.c0r3, -self.c3r0) && ulps_eq!(self.c3r0, -self.c0r3) &&
        ulps_eq!(self.c1r3, -self.c3r1) && ulps_eq!(self.c3r1, -self.c1r3) &&
        ulps_eq!(self.c2r3, -self.c3r2) && ulps_eq!(self.c3r2, -self.c2r3)
    }
}

impl<S> InvertibleSquareMatrix for Matrix4<S> where S: ScalarFloat {
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

            Some(Matrix4::new(
                c0r0, c0r1, c0r2, c0r3,
                c1r0, c1r1, c1r2, c1r3,
                c2r0, c2r1, c2r2, c2r3,
                c3r0, c3r1, c3r2, c3r3
            ))
        }
    }
}

impl<S: Scalar> iter::Sum<Matrix4<S>> for Matrix4<S> {
    #[inline]
    fn sum<I: Iterator<Item = Matrix4<S>>>(iter: I) -> Matrix4<S> {
        iter.fold(Matrix4::<S>::zero(), ops::Add::add)
    }
}

impl<'a, S: 'a + Scalar> iter::Sum<&'a Matrix4<S>> for Matrix4<S> {
    #[inline]
    fn sum<I: Iterator<Item = &'a Matrix4<S>>>(iter: I) -> Matrix4<S> {
        iter.fold(Matrix4::<S>::zero(), ops::Add::add)
    }
}

impl<S: Scalar> iter::Product<Matrix4<S>> for Matrix4<S> {
    #[inline]
    fn product<I: Iterator<Item = Matrix4<S>>>(iter: I) -> Matrix4<S> {
        iter.fold(Matrix4::<S>::one(), ops::Mul::mul)
    }
}

impl<'a, S: 'a + Scalar> iter::Product<&'a Matrix4<S>> for Matrix4<S> {
    #[inline]
    fn product<I: Iterator<Item = &'a Matrix4<S>>>(iter: I) -> Matrix4<S> {
        iter.fold(Matrix4::<S>::one(), ops::Mul::mul)
    }
}



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

impl_mul_operator!(u8,    Matrix2<u8>,    Matrix2<u8>,    { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(u16,   Matrix2<u16>,   Matrix2<u16>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(u32,   Matrix2<u32>,   Matrix2<u32>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(u64,   Matrix2<u64>,   Matrix2<u64>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(u128,  Matrix2<u128>,  Matrix2<u128>,  { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(usize, Matrix2<usize>, Matrix2<usize>, { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(i8,    Matrix2<i8>,    Matrix2<i8>,    { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(i16,   Matrix2<i16>,   Matrix2<i16>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(i32,   Matrix2<i32>,   Matrix2<i32>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(i64,   Matrix2<i64>,   Matrix2<i64>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(i128,  Matrix2<i128>,  Matrix2<i128>,  { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(isize, Matrix2<isize>, Matrix2<isize>, { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(f32,   Matrix2<f32>,   Matrix2<f32>,   { c0r0, c0r1, c1r0, c1r1 });
impl_mul_operator!(f64,   Matrix2<f64>,   Matrix2<f64>,   { c0r0, c0r1, c1r0, c1r1 });

impl_mul_operator!(u8,    Matrix3<u8>,    Matrix3<u8>,    { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(u16,   Matrix3<u16>,   Matrix3<u16>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(u32,   Matrix3<u32>,   Matrix3<u32>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(u64,   Matrix3<u64>,   Matrix3<u64>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(u128,  Matrix3<u128>,  Matrix3<u128>,  { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(usize, Matrix3<usize>, Matrix3<usize>, { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(i8,    Matrix3<i8>,    Matrix3<i8>,    { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(i16,   Matrix3<i16>,   Matrix3<i16>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(i32,   Matrix3<i32>,   Matrix3<i32>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(i64,   Matrix3<i64>,   Matrix3<i64>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(i128,  Matrix3<i128>,  Matrix3<i128>,  { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(isize, Matrix3<isize>, Matrix3<isize>, { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(f32,   Matrix3<f32>,   Matrix3<f32>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });
impl_mul_operator!(f64,   Matrix3<f64>,   Matrix3<f64>,   { c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2 });

impl_mul_operator!(u8,    Matrix4<u8>,    Matrix4<u8>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(u16,   Matrix4<u16>,   Matrix4<u16>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(u32,   Matrix4<u32>,   Matrix4<u32>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(u64,   Matrix4<u64>,   Matrix4<u64>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(u128,  Matrix4<u128>,  Matrix4<u128>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(usize, Matrix4<usize>, Matrix4<usize>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(i8,    Matrix4<i8>,    Matrix4<i8>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(i16,   Matrix4<i16>,   Matrix4<i16>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(i32,   Matrix4<i32>,   Matrix4<i32>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(i64,   Matrix4<i64>,   Matrix4<i64>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(i128,  Matrix4<i128>,  Matrix4<i128>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(isize, Matrix4<isize>, Matrix4<isize>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(f32,   Matrix4<f32>,   Matrix4<f32>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);
impl_mul_operator!(f64,   Matrix4<f64>,   Matrix4<f64>, 
    { c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, c2r0, c2r1, c2r2, c2r3, c3r0, c3r1, c3r2, c3r3 }
);

