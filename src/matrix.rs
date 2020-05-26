use std::fmt;
use std::mem;
use std::ops;

use base::{
    Scalar,
    ScalarFloat,
};
use structure::{
    Storage, 
    One, 
    Zero, 
    Matrix, 
    Lerp
};
use vector::*;


const EPSILON: f32 = 0.00001;
const M_PI: f32 = 3.14159265358979323846264338327950288;
const ONE_DEG_IN_RAD: f32 = (2.0 * M_PI) / 360.0; // == 0.017444444


/// The `Matrix2` type represents 2x2 matrices in column-major order.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C)]
pub struct Matrix2<S> {
    /// Column 1 of the matrix.
    pub c0r0: S, pub c0r1: S,
    /// Column 2 of the matrix.
    pub c1r0: S, pub c1r1: S,
}

impl<S> Matrix2<S> where S: Scalar {
    /// Construct a new 2x2 matrix from its field elements.
    pub fn new(c0r0: S, c0r1: S, c1r0: S, c1r1: S) -> Matrix2<S> {
        Matrix2 { c0r0: c0r0, c0r1: c0r1, c1r0: c1r0, c1r1: c1r1 }
    }

    /// Construct a 2x2 matrix from a pair of two-dimensional vectors.
    pub fn from_cols(c0: Vector2<S>, c1: Vector2<S>) -> Matrix2<S> {
        Matrix2 { c0r0: c0.x, c0r1: c0.y, c1r0: c1.x, c1r1: c1.y }
    }
}

impl<S> Matrix2<S> where S: ScalarFloat {
    /// Compute the determinant of a 2x2 matrix.
    pub fn determinant(&self) -> S {
        self.c0r0 * self.c1r1 - self.c0r1 * self.c1r0
    }

    /// Determine whether a 2x2 matrix is invertible.
    pub fn is_invertible(&self) -> bool {
        self.determinant() != S::zero()
    }

    /// Compute the inverse of a 2x2 matrix.
    pub fn inverse(&self) -> Option<Matrix2<S>> {
        let det = self.determinant();
        // A matrix with zero determinant has no inverse.
        // TODO: Make a more careful check for when the deterimant is very close to zero.
        if det == S::zero() {
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

impl<S> Storage for Matrix2<S> where S: Scalar {
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
        self[1][row_a] = self[0][row_b];
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

impl<S> ops::Neg for Matrix2<S> where S: ScalarFloat {
    type Output = Matrix2<S>;

    fn neg(self) -> Self::Output {
        let c0r0 = -self.c0r0;
        let c0r1 = -self.c0r1;
        let c1r0 = -self.c1r0;
        let c1r1 = -self.c1r1;

        Matrix2::new(c0r0, c0r1, c1r0, c1r1)
    }
}

impl<S> ops::Neg for &Matrix2<S> where S: ScalarFloat {
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


/// The `Matrix3` type represents 3x3 matrices in column-major order.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(C)]
pub struct Matrix3<S> {
    /// Column 1 of the matrix.
    pub c0r0: S, pub c0r1: S, pub c0r2: S,
    /// Column 2 of the matrix.
    pub c1r0: S, pub c1r1: S, pub c1r2: S,
    /// Column 3 of the matrix.
    pub c2r0: S, pub c2r1: S, pub c2r2: S,
}

impl<S> Matrix3<S> {
    pub fn new(
        c0r0: S, c0r1: S, c0r2: S,
        c1r0: S, c1r1: S, c1r2: S,
        c2r0: S, c2r1: S, c2r2: S) -> Matrix3<S> {

        Matrix3 {
            // Column 1 of the matrix.
            c0r0: c0r0, c0r1: c0r1, c0r2: c0r2,
            // Column 2 of the matrix.
            c1r0: c1r0, c1r1: c1r1, c1r2: c1r2,
            // Column 3 of the matrix.
            c2r0: c2r0, c2r1: c2r1, c2r2: c2r2,
        }
    }

    /// Create a 3x3 matrix from a triple of three-dimensional column vectors.
    pub fn from_cols(c0: Vector3<S>, c1: Vector3<S>, c2: Vector3<S>) -> Matrix3<S> {
        Matrix3 {
            c0r0: c0.x, c0r1: c0.y, c0r2: c0.z, 
            c1r0: c1.x, c1r1: c1.y, c1r2: c1.z,
            c2r0: c2.x, c2r1: c2.y, c2r2: c2.z,
        }
    }
}

impl<S> Matrix3<S> where S: ScalarFloat {
    /// Calculate the determinant of a 3x3 matrix.
    pub fn determinant(&self) -> S {
        self.c0r0 * self.c1r1 * self.c2r2 - self.c0r0 * self.c1r2 * self.c2r1 -
        self.c1r0 * self.c0r1 * self.c2r2 + self.c1r0 * self.c0r2 * self.c2r1 +
        self.c2r0 * self.c0r1 * self.c1r2 - self.c2r0 * self.c0r2 * self.c1r1
    }

    /// Determine whether a 3x3 matrix is invertible.
    pub fn is_invertible(&self) -> bool {
        self.determinant() != S::zero()
    }

    /// Calculate the inverser of a 3x3 matrix, if it exists.
    /// A matrix with zero determinant has no inverse.
    pub fn inverse(&self) -> Option<Matrix3<S>> {
        let det = self.determinant();
        if det == S::zero() {
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

impl<S> Storage for Matrix3<S> where S: Scalar {
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
        Matrix3::new(value, value, value, value, value, value, value, value, value)
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
        self[1][row_a] = self[0][row_b];
        self[2][row_a] = self[0][row_b];
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
            m.c0r0, m.c0r1, S::zero(),
            m.c1r0, m.c1r1, S::zero(),
            S::zero(),    S::zero(), S::one()
        )
    }
}

impl<S> From<&Matrix2<S>> for Matrix3<S> where S: Scalar {
    #[inline]
    fn from(m: &Matrix2<S>) -> Matrix3<S> {
        Matrix3::new(
            m.c0r0, m.c0r1, S::zero(),
            m.c1r0, m.c1r1, S::zero(),
            S::zero(),    S::zero(), S::one()
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

impl<S> ops::Neg for Matrix3<S> where S: ScalarFloat {
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

impl<S> ops::Neg for &Matrix3<S> where S: ScalarFloat {
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


/// The `Matrix4` type represents 4x4 matrices in column-major order.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Matrix4<S> {
    /// Column 1 of the matrix.
    pub c0r0: S, pub c0r1: S, pub c0r2: S, pub c0r3: S,
    /// Column 2 of the matrix.
    pub c1r0: S, pub c1r1: S, pub c1r2: S, pub c1r3: S,
    /// Column 3 of the matrix.
    pub c2r0: S, pub c2r1: S, pub c2r2: S, pub c2r3: S,
    /// Column 4 of the matrix.
    pub c3r0: S, pub c3r1: S, pub c3r2: S, pub c3r3: S,
}

impl<S> Matrix4<S> {
    pub fn new(
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
    pub fn from_cols(c0: Vector4<S>, c1: Vector4<S>, c2: Vector4<S>, c3: Vector4<S>) -> Matrix4<S> {
        Matrix4 {
            c0r0: c0.x, c0r1: c0.y, c0r2: c0.z, c0r3: c0.w,
            c1r0: c1.x, c1r1: c1.y, c1r2: c1.z, c1r3: c1.w,
            c2r0: c2.x, c2r1: c2.y, c2r2: c2.z, c2r3: c2.w,
            c3r0: c3.x, c3r1: c3.y, c3r2: c3.z, c3r3: c3.w,
        }
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
    /*
    /// Create a rotation matrix around the x axis by an angle in `degrees` degrees.
    pub fn from_rotation_x(degrees: S) -> Matrix4<S> {
        // Convert to radians.
        let radians = degrees * ONE_DEG_IN_RAD;
        let mut rot_mat = Matrix4::one();
        rot_mat.c1r1 =  f32::cos(radians);
        rot_mat.c2r1 = -f32::sin(radians);
        rot_mat.c1r2 =  f32::sin(radians);
        rot_mat.c2r2 =  f32::cos(radians);
    
        rot_mat
    }
    
    /// Create a rotation matrix around the y axis by an angle in `degrees` degrees.
    pub fn from_rotation_y(degrees: S) -> Matrix4<S> {
        // Convert to radians.
        let radians = degrees * ONE_DEG_IN_RAD;
        let mut rot_mat = Matrix4::one();
        rot_mat.c0r0 =  f32::cos(radians);
        rot_mat.c2r0 =  f32::sin(radians);
        rot_mat.c0r2 = -f32::sin(radians);
        rot_mat.c2r2 =  f32::cos(radians);
    
        rot_mat
    }

    /// Create a rotation matrix around the z axis by an angle in `degrees` degrees.
    pub fn from_rotation_z(degrees: S) -> Matrix4<S> {
        // Convert to radians.
        let radians = degrees * ONE_DEG_IN_RAD;
        let mut rot_mat = Matrix4::one();
        rot_mat.c0r0 =  f32::cos(radians);
        rot_mat.c1r0 = -f32::sin(radians);
        rot_mat.c0r1 =  f32::sin(radians);
        rot_mat.c1r1 =  f32::cos(radians);
    
        rot_mat
    }
    */
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

    /// Computes the determinant of a 4x4 matrix.
    pub fn determinant(&self) -> S {
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

    pub fn is_invertible(&self) -> bool {
        !self.determinant().is_zero()
    }

    /// Compute the inverse of a 4x4 matrix.
    pub fn inverse(&self) -> Option<Matrix4<S>> {
        let det = self.determinant();
        // A matrix with zero determinant has no inverse.
        if det == S::zero() {
            None
        } else {
            let inv_det = S::one() / det;

            Some(Matrix4::new(
                inv_det * ( self.c2r1 * self.c3r2 * self.c1r3 - self.c3r1 * self.c2r2 * self.c1r3 +
                                        self.c3r1 * self.c1r2 * self.c2r3 - self.c1r1 * self.c3r2 * self.c2r3 -
                                        self.c2r1 * self.c1r2 * self.c3r3 + self.c1r1 * self.c2r2 * self.c3r3 ),
                inv_det * ( self.c3r1 * self.c2r2 * self.c0r3 - self.c2r1 * self.c3r2 * self.c0r3 -
                                        self.c3r1 * self.c0r2 * self.c2r3 + self.c0r1 * self.c3r2 * self.c2r3 +
                                        self.c2r1 * self.c0r2 * self.c3r3 - self.c0r1 * self.c2r2 * self.c3r3 ),
                inv_det * ( self.c1r1 * self.c3r2 * self.c0r3 - self.c3r1 * self.c1r1 * self.c0r3 +
                                        self.c3r1 * self.c0r2 * self.c1r2 - self.c0r1 * self.c3r2 * self.c1r3 -
                                        self.c1r1 * self.c0r2 * self.c3r3 + self.c0r1 * self.c1r2 * self.c3r3 ),
                inv_det * ( self.c2r1 * self.c1r2 * self.c0r3 - self.c1r1 * self.c2r2 * self.c0r3 -
                                        self.c2r1 * self.c0r2 * self.c1r3 + self.c0r1 * self.c2r2 * self.c1r3 +
                                        self.c1r1 * self.c0r2 * self.c2r3 - self.c0r1 * self.c1r2 * self.c2r3 ),
                inv_det * ( self.c3r0 * self.c2r2 * self.c1r3 - self.c2r0 * self.c3r2 * self.c1r3 -
                                        self.c3r0 * self.c1r2 * self.c2r3 + self.c1r0 * self.c3r2 * self.c2r3 +
                                        self.c2r0 * self.c1r2 * self.c3r3 - self.c1r0 * self.c2r2 * self.c3r3 ),
                inv_det * ( self.c2r0 * self.c3r2 * self.c0r3 - self.c3r0 * self.c2r2 * self.c0r3 +
                                        self.c3r0 * self.c0r2 * self.c2r3 - self.c0r0 * self.c3r2 * self.c2r3 -
                                        self.c2r0 * self.c0r2 * self.c3r3 + self.c0r0 * self.c2r2 * self.c3r3 ),
                inv_det * ( self.c3r0 * self.c1r2 * self.c0r3 - self.c1r0 * self.c3r2 * self.c0r3 -
                                        self.c3r0 * self.c0r2 * self.c1r3 + self.c0r0 * self.c3r2 * self.c1r3 +
                                        self.c1r0 * self.c0r2 * self.c3r3 - self.c0r0 * self.c1r2 * self.c3r3 ),
                inv_det * ( self.c1r0 * self.c2r2 * self.c0r3 - self.c2r0 * self.c1r2 * self.c0r3 +
                                        self.c2r0 * self.c0r2 * self.c1r3 - self.c0r0 * self.c2r2 * self.c1r3 -
                                        self.c1r0 * self.c0r2 * self.c2r3 + self.c0r0 * self.c1r2 * self.c2r3 ),
                inv_det * ( self.c2r0 * self.c3r1 * self.c1r3 - self.c3r0 * self.c2r1 * self.c1r3 +
                                        self.c3r0 * self.c1r1 * self.c2r3 - self.c1r0 * self.c3r1 * self.c2r3 -
                                        self.c2r0 * self.c1r1 * self.c3r3 + self.c1r0 * self.c2r1 * self.c3r3 ),
                inv_det * ( self.c3r0 * self.c2r1 * self.c0r3 - self.c2r0 * self.c3r1 * self.c0r3 -
                                        self.c3r0 * self.c0r1 * self.c2r3 + self.c0r0 * self.c3r1 * self.c2r3 +
                                        self.c2r0 * self.c0r1 * self.c3r3 - self.c0r0 * self.c2r1 * self.c3r3 ),
                inv_det * ( self.c1r0 * self.c3r1 * self.c0r3 - self.c3r0 * self.c1r1 * self.c0r3 +
                                        self.c3r0 * self.c0r1 * self.c1r3 - self.c0r0 * self.c3r1 * self.c1r3 -
                                        self.c1r0 * self.c0r1 * self.c3r3 + self.c0r0 * self.c1r1 * self.c3r3 ),
                inv_det * ( self.c2r0 * self.c1r1 * self.c0r3 - self.c1r0 * self.c2r1 * self.c0r3 -
                                        self.c2r0 * self.c0r1 * self.c1r3 + self.c0r0 * self.c2r1 * self.c1r3 +
                                        self.c1r0 * self.c0r1 * self.c2r3 - self.c0r0 * self.c1r1 * self.c2r3 ),
                inv_det * ( self.c3r0 * self.c2r1 * self.c1r2 - self.c2r0 * self.c3r1 * self.c1r2 -
                                        self.c3r0 * self.c1r1 * self.c2r2 + self.c1r0 * self.c3r1 * self.c2r2 +
                                        self.c2r0 * self.c1r1 * self.c3r2 - self.c1r0 * self.c2r1 * self.c3r2 ),
                inv_det * ( self.c2r0 * self.c3r1 * self.c0r2 - self.c3r0 * self.c2r1 * self.c0r2 +
                                        self.c3r0 * self.c0r1 * self.c2r2 - self.c0r0 * self.c3r1 * self.c2r2 -
                                        self.c2r0 * self.c0r1 * self.c3r2 + self.c0r0 * self.c2r1 * self.c3r2 ),
                inv_det * ( self.c3r0 * self.c1r1 * self.c0r2 - self.c1r0 * self.c3r1 * self.c0r2 -
                                        self.c3r0 * self.c0r1 * self.c1r2 + self.c0r0 * self.c3r1 * self.c1r2 +
                                        self.c1r0 * self.c0r1 * self.c3r2 - self.c0r0 * self.c1r1 * self.c3r2 ),
                inv_det * ( self.c1r0 * self.c2r1 * self.c0r2 - self.c2r0 * self.c1r1 * self.c0r2 +
                                        self.c2r0 * self.c0r1 * self.c1r2 - self.c0r0 * self.c2r1 * self.c1r2 -
                                        self.c1r0 * self.c0r1 * self.c2r2 + self.c0r0 * self.c1r1 * self.c2r2 ) ) )
        }
    }
}

impl<S> Storage for Matrix4<S> where S: Scalar {
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
            value, value, value, value, value, value, value, value, 
            value, value, value, value, value, value, value, value
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
        self[1][row_a] = self[0][row_b];
        self[2][row_a] = self[0][row_b];
        self[3][row_a] = self[0][row_b];
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
            m[0], m[1], m[2], m[3], 
            m[4], m[5], m[6], m[7],
            m[8], m[9], m[10], m[11], 
            m[12],m[13], m[14], m[15]
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
            m.c0r0, m.c0r1, zero, zero,
            m.c1r0, m.c1r1, zero, zero,
                zero, zero,  one, zero,
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
            m.c0r0, m.c0r1, zero, zero,
            m.c1r0, m.c1r1, zero, zero,
        zero,zero,  one, zero,
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
               zero,    zero,    zero, one
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
               zero,    zero,    zero, one
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

impl<'a, 'b,S> ops::Mul<&'a Matrix4<S>> for &'b Matrix4<S> where S: Scalar {
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

impl<S> ops::Neg for Matrix4<S> where S: ScalarFloat {
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

impl<S> ops::Neg for &Matrix4<S> where S: ScalarFloat {
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


#[cfg(test)]
mod matrix2_tests {
    use std::slice::Iter;
    use vector::Vector2;
    use super::Matrix2;
    use structure::{One, Zero, Matrix};
    use crate::approx::relative_eq;


    struct TestCase {
        a_mat: Matrix2<f32>,
        b_mat: Matrix2<f32>,
        expected: Matrix2<f32>,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase {
                    a_mat: Matrix2::new(80.0,  23.43,     426.1,   23.5724),
                    b_mat: Matrix2::new(36.84, 427.46894, 7.04217, 61.891390),
                    expected: Matrix2::new(185091.72, 10939.63, 26935.295, 1623.9266),
                },
                TestCase {
                    a_mat: Matrix2::one(),
                    b_mat: Matrix2::one(),
                    expected: Matrix2::one(),
                },
                TestCase {
                    a_mat: Matrix2::zero(),
                    b_mat: Matrix2::zero(),
                    expected: Matrix2::zero(),
                },
                TestCase {
                    a_mat: Matrix2::new(68.32, 0.0, 0.0, 37.397),
                    b_mat: Matrix2::new(57.72, 0.0, 0.0, 9.5433127),
                    expected: Matrix2::new(3943.4304, 0.0, 0.0, 356.89127),
                },
            ]
        }
    }

    #[test]
    fn test_mat_times_identity_equals_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_identity = test.a_mat * Matrix2::one();
            let b_mat_times_identity = test.b_mat * Matrix2::one();

            assert_eq!(a_mat_times_identity, test.a_mat);
            assert_eq!(b_mat_times_identity, test.b_mat);
        })
    }

    #[test]
    fn test_mat_times_zero_equals_zero() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_zero = test.a_mat * Matrix2::zero();
            let b_mat_times_zero = test.b_mat * Matrix2::zero();

            assert_eq!(a_mat_times_zero, Matrix2::zero());
            assert_eq!(b_mat_times_zero, Matrix2::zero());
        })
    }

    #[test]
    fn test_zero_times_mat_equals_zero() {
        test_cases().iter().for_each(|test| {
            let zero_times_a_mat = Matrix2::zero() * test.a_mat;
            let zero_times_b_mat = Matrix2::zero() * test.b_mat;

            assert_eq!(zero_times_a_mat, Matrix2::zero());
            assert_eq!(zero_times_b_mat, Matrix2::zero());
        })
    }

    #[test]
    fn test_mat_times_identity_equals_identity_times_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_identity = test.a_mat * Matrix2::one();
            let identity_times_a_mat = Matrix2::one() * test.a_mat;
            let b_mat_times_identity = test.b_mat * Matrix2::one();
            let identity_times_b_mat = Matrix2::one() * test.b_mat;

            assert_eq!(a_mat_times_identity, identity_times_a_mat);
            assert_eq!(b_mat_times_identity, identity_times_b_mat);
        })
    }

    #[test]
    fn test_mat_transpose_transpose_equals_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_tr_tr = test.a_mat.transpose().transpose();
            let b_mat_tr_tr = test.b_mat.transpose().transpose();
            
            assert_eq!(a_mat_tr_tr, test.a_mat);
            assert_eq!(b_mat_tr_tr, test.b_mat);
        })
    }

    #[test]
    fn test_identity_transpose_equals_identity() {
        let identity = Matrix2::<f32>::one();
        let identity_tr = identity.transpose();
            
        assert_eq!(identity, identity_tr);
    }

    #[test]
    fn test_matrix_multiplication() {
        test_cases().iter().for_each(|test| {
            let result = test.a_mat * test.b_mat;
            let expected = test.expected;

            assert_eq!(result, expected);
        })
    }

    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector2::new(1.0, 2.0);
        let c1 = Vector2::new(3.0, 4.0);
        let expected = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        let result = Matrix2::from_cols(c0, c1);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_constant_times_identity_is_constant_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix2::one();
        let expected = Matrix2::new(c, 0.0, 0.0, c);

        assert_eq!(id * c, expected);
    }

    #[test]
    fn test_identity_divide_constant_is_constant_inverse_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix2::one();
        let expected = Matrix2::new(1.0/c, 0.0, 0.0, 1.0/c);

        assert_eq!(id / c, expected);
    }

    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero = Matrix2::zero();
        let matrix = Matrix2::new(36.84, 427.46, 7.47, 61.89);

        assert_eq!(matrix + zero, matrix);
    }

    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero = Matrix2::zero();
        let matrix = Matrix2::new(36.84, 427.46, 7.47, 61.89);

        assert_eq!(zero + matrix, matrix);
    }

    #[test]
    fn test_matrix_with_zero_determinant() {
        let matrix = Matrix2::new(1f32, 2f32, 4f32, 8f32);
        
        assert_eq!(matrix.determinant(), 0.0);
    }

    #[test]
    fn test_matrix_with_nonzero_determinant_is_invertible() {
        let matrix = Matrix2::new(1f32, 2f32, 3f32, 4f32);
        
        assert!(matrix.is_invertible());
    }

    #[test]
    fn test_matrix_with_zero_determinant_is_not_invertible() {
        let matrix = Matrix2::new(1f32, 2f32, 4f32, 8f32);
        
        assert!(!matrix.is_invertible());
    }

    #[test]
    fn test_noninvertible_matrix_returns_none() {
        let matrix = Matrix2::new(1f32, 2f32, 4f32, 8f32);
        
        assert!(matrix.inverse().is_none());
    }


    #[test]
    fn test_matrix_times_inverse_is_identity() {
        let matrix = Matrix2::new(36.84, 427.46, 7.47, 61.89);
        let matrix_inv = matrix.inverse().unwrap();
        let one = Matrix2::one();

        assert!(relative_eq!(matrix * matrix_inv, one, epsilon = 1e-7));
    }

    #[test]
    fn test_inverse_times_matrix_is_identity() {
        let matrix = Matrix2::new(36.84, 427.46, 7.47, 61.89);
        let matrix_inv = matrix.inverse().unwrap();
        let one = Matrix2::one();

        assert!(relative_eq!(matrix_inv * matrix, one, epsilon = 1e-7));        
    }

    #[test]
    fn test_matrix_elements_should_be_column_major_order() {
        let matrix = Matrix2::new(1, 2, 3, 4);
        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c0r1, matrix[0][1]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c1r1, matrix[1][1]);
    }
}


#[cfg(test)]
mod matrix3_tests {
    use std::slice::Iter;
    use vector::Vector3;
    use super::Matrix3;
    use structure::{One, Zero, Matrix};
    use crate::approx::relative_eq;


    struct TestCase {
        a_mat: Matrix3<f32>,
        b_mat: Matrix3<f32>,
        expected: Matrix3<f32>,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase {
                    a_mat: Matrix3::new(80.0, 426.1, 43.393, 23.43, 23.5724, 1.27, 81.439, 12.19, 43.36),
                    b_mat: Matrix3::new(36.84, 7.04217, 5.74, 427.46894, 61.89139, 96.27, 152.66, 86.333, 26.71),
                    expected: Matrix3::new(3579.6579, 15933.496, 1856.4281, 43487.7660, 184776.9752, 22802.0289, 16410.8178, 67409.1000, 7892.1646),
                },
                TestCase {
                    a_mat: Matrix3::one(),
                    b_mat: Matrix3::one(),
                    expected: Matrix3::one(),
                },
                TestCase {
                    a_mat: Matrix3::zero(),
                    b_mat: Matrix3::zero(),
                    expected: Matrix3::zero(),
                },
                TestCase {
                    a_mat: Matrix3::new(68.32, 0.0, 0.0, 0.0, 37.397, 0.0, 0.0, 0.0, 43.393),
                    b_mat: Matrix3::new(57.72, 0.0, 0.0, 0.0, 9.5433127, 0.0, 0.0, 0.0, 12.19),
                    expected: Matrix3::new(3943.4304, 0.0, 0.0, 0.0, 356.89127, 0.0, 0.0, 0.0, 528.96067),
                },
            ]
        }
    }

    #[test]
    fn test_mat_times_identity_equals_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_identity = test.a_mat * Matrix3::one();
            let b_mat_times_identity = test.b_mat * Matrix3::one();

            assert_eq!(a_mat_times_identity, test.a_mat);
            assert_eq!(b_mat_times_identity, test.b_mat);
        })
    }

    #[test]
    fn test_mat_times_zero_equals_zero() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_zero = test.a_mat * Matrix3::zero();
            let b_mat_times_zero = test.b_mat * Matrix3::zero();

            assert_eq!(a_mat_times_zero, Matrix3::zero());
            assert_eq!(b_mat_times_zero, Matrix3::zero());
        })
    }

    #[test]
    fn test_zero_times_mat_equals_zero() {
        test_cases().iter().for_each(|test| {
            let zero_times_a_mat = Matrix3::zero() * test.a_mat;
            let zero_times_b_mat = Matrix3::zero() * test.b_mat;

            assert_eq!(zero_times_a_mat, Matrix3::zero());
            assert_eq!(zero_times_b_mat, Matrix3::zero());
        })
    }

    #[test]
    fn test_mat_times_identity_equals_identity_times_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_times_identity = test.a_mat * Matrix3::one();
            let identity_times_a_mat = Matrix3::one() * test.a_mat;
            let b_mat_times_identity = test.b_mat * Matrix3::one();
            let identity_times_b_mat = Matrix3::one() * test.b_mat;

            assert_eq!(a_mat_times_identity, identity_times_a_mat);
            assert_eq!(b_mat_times_identity, identity_times_b_mat);
        })
    }

    #[test]
    fn test_mat_transpose_transpose_equals_mat() {
        test_cases().iter().for_each(|test| {
            let a_mat_tr_tr = test.a_mat.transpose().transpose();
            let b_mat_tr_tr = test.b_mat.transpose().transpose();
            
            assert_eq!(a_mat_tr_tr, test.a_mat);
            assert_eq!(b_mat_tr_tr, test.b_mat);
        })
    }

    #[test]
    fn test_identity_transpose_equals_identity() {
        let identity = Matrix3::<f32>::one();
        let identity_tr = identity.transpose();
            
        assert_eq!(identity, identity_tr);
    }

    #[test]
    fn test_matrix_multiplication() {
        test_cases().iter().for_each(|test| {
            let result = test.a_mat * test.b_mat;
            let expected = test.expected;

            assert_eq!(result, expected);
        })
    }

    #[test]
    fn test_construction_from_cols() {
        let c0 = Vector3::new(1.0, 2.0, 3.0);
        let c1 = Vector3::new(4.0, 5.0, 6.0);
        let c2 = Vector3::new(7.0, 8.0, 9.0);
        let expected = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let result = Matrix3::from_cols(c0, c1, c2);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_constant_times_identity_is_constant_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix3::one();
        let expected = Matrix3::new(c, 0.0, 0.0, 0.0, c, 0.0, 0.0, 0.0, c);

        assert_eq!(id * c, expected);
    }

    #[test]
    fn test_identity_divide_constant_is_constant_inverse_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix3::one();
        let expected = Matrix3::new(1.0/c, 0.0, 0.0, 0.0, 1.0/c, 0.0, 0.0, 0.0, 1.0/c);

        assert_eq!(id / c, expected);
    }

    #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero = Matrix3::zero();
        let matrix = Matrix3::new(80.0, 426.1, 43.393, 23.43, 23.5724, 1.27, 81.439, 12.19, 43.36);

        assert_eq!(matrix + zero, matrix);
    }

    #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero = Matrix3::zero();
        let matrix = Matrix3::new(80.0, 426.1, 43.393, 23.43, 23.5724, 1.27, 81.439, 12.19, 43.36);

        assert_eq!(zero + matrix, matrix);
    }

    #[test]
    fn test_matrix_with_zero_determinant() {
        let matrix = Matrix3::new(1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 4f32, 5f32, 6f32);
        
        assert_eq!(matrix.determinant(), 0.0);
    }

    #[test]
    fn test_matrix_with_nonzero_determinant_is_invertible() {
        let matrix = Matrix3::new(1f32, 2f32, 3f32, 0f32, 4f32, 5f32, 0f32, 0f32, 6f32);
        
        assert!(matrix.is_invertible());
    }

    #[test]
    fn test_matrix_with_zero_determinant_is_not_invertible() {
        let matrix = Matrix3::new(1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 4f32, 5f32, 6f32);
        
        assert!(!matrix.is_invertible());
    }

    #[test]
    fn test_noninvertible_matrix_returns_none() {
        let matrix = Matrix3::new(1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 4f32, 5f32, 6f32);
        
        assert!(matrix.inverse().is_none());
    }

    #[test]
    fn test_matrix_times_inverse_is_identity() {
        let matrix = Matrix3::new(80.0, 426.1, 43.393, 23.43, 23.5724, 1.27, 81.439, 12.19, 43.36);
        let matrix_inv = matrix.inverse().unwrap();
        let one = Matrix3::one();

        assert!(relative_eq!(matrix * matrix_inv, one, epsilon = 1e-7));
    }

    #[test]
    fn test_inverse_times_matrix_is_identity() {
        let matrix = Matrix3::new(80.0, 426.1, 43.393, 23.43, 23.5724, 1.27, 81.439, 12.19, 43.36);
        let matrix_inv = matrix.inverse().unwrap();
        let one = Matrix3::one();

        assert!(relative_eq!(matrix_inv * matrix, one, epsilon = 1e-7));
    }

    #[test]
    fn test_matrix_elements_should_be_column_major_order() {
        let matrix = Matrix3::new(1, 2, 3, 4, 5, 6, 7, 8, 9);
        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c0r1, matrix[0][1]);
        assert_eq!(matrix.c0r2, matrix[0][2]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c1r1, matrix[1][1]);
        assert_eq!(matrix.c1r2, matrix[1][2]);
        assert_eq!(matrix.c2r0, matrix[2][0]);
        assert_eq!(matrix.c2r1, matrix[2][1]);
        assert_eq!(matrix.c2r2, matrix[2][2]);
    }
}

#[cfg(test)]
mod matrix4_tests {
    use std::slice::Iter;
    use vector::{Vector3, Vector4};
    use super::{Matrix4};
    use structure::{One, Zero, Matrix};
    use approx::assert_relative_eq;


    struct TestCase {
        a_mat: Matrix4<f64>,
        b_mat: Matrix4<f64>,
        expected: Matrix4<f64>,
    }

    struct Test {
        tests: Vec<TestCase>,
    }

    impl Test {
        fn iter(&self) -> TestIter {
            TestIter {
                inner: self.tests.iter()
            }
        }
    }

    struct TestIter<'a> {
        inner: Iter<'a, TestCase>,
    }

    impl<'a> Iterator for TestIter<'a> {
        type Item = &'a TestCase;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next()
        }
    }

    fn test_cases() -> Test {
        Test {
            tests: vec![
                TestCase {
                    a_mat: Matrix4::new(
                        80.0,   23.43,   43.569,  6.741, 
                        426.1,  23.5724, 27.6189, 13.90,
                        4.2219, 258.083, 31.70,   42.17, 
                        70.0,   49.0,    95.0,    89.9138
                    ),
                    b_mat: Matrix4::new(
                        36.84,   427.4689, 882.1983, 89.5049, 
                        7.0421,  61.8913,  56.31,    89.0, 
                        72.0,    936.5,    413.80,   50.3111,  
                        37.6985, 311.8,    60.81,    73.8393
                    ),
                    expected: Matrix4::new(
                        195081.59429, 243005.75305, 49879.95437, 51440.18413,
                        33402.98611,  20517.57661,  12256.21388, 11284.68076,
                        410071.44922, 133022.5019,  46899.08235, 35476.31018,
                        141299.34473, 27545.30310,  19196.46946, 13791.67534
                    ),
                },
                TestCase {
                    a_mat: Matrix4::one(),
                    b_mat: Matrix4::one(),
                    expected: Matrix4::one(),
                },
                TestCase {
                    a_mat: Matrix4::zero(),
                    b_mat: Matrix4::zero(),
                    expected: Matrix4::zero(),
                },
                TestCase {
                    a_mat: Matrix4::new(
                        68.32, 0.0,    0.0,   0.0,
                        0.0,   37.397, 0.0,   0.0,
                        0.0,   0.0,    9.483, 0.0,
                        0.0,   0.0,    0.0,   887.710
                    ),
                    b_mat: Matrix4::new(
                        57.72, 0.0,    0.0,     0.0, 
                        0.0,   9.5433, 0.0,     0.0, 
                        0.0,   0.0,    86.7312, 0.0,
                        0.0,   0.0,    0.0,     269.1134
                    ),
                    expected: Matrix4::new(
                        3943.4304, 0.0,         0.0,         0.0,
                        0.0,       356.8907901, 0.0,         0.0,
                        0.0,       0.0,         822.4719696, 0.0,
                        0.0,       0.0,         0.0,         238894.65631
                    ),
                },
            ]
        }
    }

    #[test]
    fn test_mat_times_identity_equals_mat() {
        for test in test_cases().iter() {
            let a_mat_times_identity = test.a_mat * Matrix4::one();
            let b_mat_times_identity = test.b_mat * Matrix4::one();

            assert_eq!(a_mat_times_identity, test.a_mat);
            assert_eq!(b_mat_times_identity, test.b_mat);
        }
    }

    #[test]
    fn test_mat_times_zero_equals_zero() {
        for test in test_cases().iter() {
            let a_mat_times_zero = test.a_mat * Matrix4::zero();
            let b_mat_times_zero = test.b_mat * Matrix4::zero();

            assert_eq!(a_mat_times_zero, Matrix4::zero());
            assert_eq!(b_mat_times_zero, Matrix4::zero());
        }
    }

    #[test]
    fn test_zero_times_mat_equals_zero() {
        for test in test_cases().iter() {
            let zero_times_a_mat = Matrix4::zero() * test.a_mat;
            let zero_times_b_mat = Matrix4::zero() * test.b_mat;

            assert_eq!(zero_times_a_mat, Matrix4::zero());
            assert_eq!(zero_times_b_mat, Matrix4::zero());
        }
    }

    #[test]
    fn test_mat_times_identity_equals_identity_times_mat() {
        for test in test_cases().iter() {
            let a_mat_times_identity = test.a_mat * Matrix4::one();
            let identity_times_a_mat = Matrix4::one() * test.a_mat;
            let b_mat_times_identity = test.b_mat * Matrix4::one();
            let identity_times_b_mat = Matrix4::one() * test.b_mat;

            assert_eq!(a_mat_times_identity, identity_times_a_mat);
            assert_eq!(b_mat_times_identity, identity_times_b_mat);
        }
    }

    #[test]
    fn test_mat_transpose_transpose_equals_mat() {
        for test in test_cases().iter() {
            let a_mat_tr_tr = test.a_mat.transpose().transpose();
            let b_mat_tr_tr = test.b_mat.transpose().transpose();
            
            assert_eq!(a_mat_tr_tr, test.a_mat);
            assert_eq!(b_mat_tr_tr, test.b_mat);
        }
    }

    #[test]
    fn test_identity_transpose_equals_identity() {
        let identity = Matrix4::<f32>::one();
        let identity_tr = identity.transpose();
            
        assert_eq!(identity, identity_tr);
    }

    #[test]
    fn test_matrix_multiplication() {
        test_cases().iter().for_each(|test| {
            let result = test.a_mat * test.b_mat;
            let expected = test.expected;
            let epsilon = 1e-7;

            assert_relative_eq!(result, expected, epsilon = epsilon);
        })
    }

    #[test]
    fn test_identity_mat4_translates_vector_along_vector() {
        let v = Vector3::from((2.0, 2.0, 2.0));
        let trans_mat = Matrix4::from_translation(v);
        let zero_vec4 = Vector4::from((0.0, 0.0, 0.0, 1.0));
        let zero_vec3 = Vector3::from((0.0, 0.0, 0.0));

        let result = trans_mat * zero_vec4;
        assert_eq!(result, Vector4::from((zero_vec3 + v, 1.0)));
    }

    #[test]
    fn test_constant_times_identity_is_identity_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix4::one();
        let expected = Matrix4::new(
            c, 0.0, 0.0, 0.0, 0.0, c, 0.0, 0.0, 0.0, 0.0, c, 0.0, 0.0, 0.0, 0.0, c
        );

        assert_eq!(id * c, expected);
    }

    #[test]
    fn test_identity_divide_constant_is_constant_inverse_along_diagonal() {
        let c = 802.3435169;
        let id = Matrix4::one();
        let expected = Matrix4::new(
            1.0/c, 0.0, 0.0, 0.0, 0.0, 1.0/c, 0.0, 0.0, 0.0, 0.0, 1.0/c, 0.0, 0.0, 0.0, 0.0, 1.0/c
        );

        assert_eq!(id / c, expected);
    }

        #[test]
    fn test_matrix_plus_zero_equals_matrix() {
        let zero = Matrix4::zero();
        let matrix = Matrix4::new(
            36.84,   427.46894, 8827.1983, 89.5049494, 
            7.04217, 61.891390, 56.31,     89.0, 
            72.0,    936.5,     413.80,    50.311160,  
            37.6985,  311.8,    60.81,     73.8393
        );

        assert_eq!(matrix + zero, matrix);
    }

        #[test]
    fn test_zero_plus_matrix_equals_matrix() {
        let zero = Matrix4::zero();
        let matrix = Matrix4::new(
            36.84,   427.46894, 8827.1983, 89.5049494, 
            7.04217, 61.891390, 56.31,     89.0, 
            72.0,    936.5,     413.80,    50.311160,  
            37.6985,  311.8,    60.81,     73.8393
        );

        assert_eq!(zero + matrix, matrix);
    }

    #[test]
    fn test_matrix_with_zero_determinant() {
        let matrix: Matrix4<f64> = Matrix4::new(
            1_f64,  2_f64,  3_f64,  4_f64, 5_f64,  6_f64,  7_f64,  8_f64,
            5_f64,  6_f64,  7_f64,  8_f64, 9_f64, 10_f64, 11_f64, 12_f64
        );
        
        assert_eq!(matrix.determinant(), 0.0);
    }

    #[test]
    fn test_matrix_with_zero_determinant_is_not_invertible() {
        let matrix: Matrix4<f64> = Matrix4::new(
            1_f64,  2_f64,  3_f64,  4_f64, 5_f64,  6_f64,  7_f64,  8_f64,
            5_f64,  6_f64,  7_f64,  8_f64, 9_f64, 10_f64, 11_f64, 12_f64
        );
        
        assert!(!matrix.is_invertible());
    }

    #[test]
    fn test_noninvertible_matrix_returns_none() {
        let matrix: Matrix4<f64> = Matrix4::new(
            1_f64,  2_f64,  3_f64,  4_f64, 
            5_f64,  6_f64,  7_f64,  8_f64,
            5_f64,  6_f64,  7_f64,  8_f64, 
            9_f64,  10_f64, 11_f64, 12_f64
        );
        
        assert!(matrix.inverse().is_none());
    }

    #[test]
    fn test_matrix_times_inverse_is_identity() {
        let matrix: Matrix4<f64> = Matrix4::new(
            36.84,   427.4689, 827.1983, 89.5049, 
            7.0421 , 61.8913,  56.31,    89.0, 
            72.0,    936.5,    413.80,   50.3111,  
            37.6985, 311.8,    60.81,    73.8393
        );
        let matrix_inv = matrix.inverse().unwrap();
        let one = Matrix4::one();
        let epsilon = 1e-7;

        assert_relative_eq!(matrix_inv * matrix, one, epsilon = epsilon);
    }

    #[test]
    fn test_inverse_times_matrix_is_identity() {
        let matrix: Matrix4<f64> = Matrix4::new(
            36.84,   427.468, 827.198,  89.504, 
            7.042,   61.891,  56.31,    89.0, 
            72.0,    936.5,   413.80,   50.311,  
            37.698,  311.8,   60.81,    73.839
        );
        let matrix_inv = matrix.inverse().unwrap();
        let one = Matrix4::one();
        let epsilon = 1e-7;
        
        assert_relative_eq!(matrix_inv * matrix, one, epsilon = epsilon);        
    }

    #[test]
    fn test_matrix_elements_should_be_column_major_order() {
        let matrix = Matrix4::new(
            1, 2, 3, 4, 5, 6, 7, 8, 
            9, 10, 11, 12, 13, 14, 15, 16
        );
        assert_eq!(matrix.c0r0, matrix[0][0]);
        assert_eq!(matrix.c0r1, matrix[0][1]);
        assert_eq!(matrix.c0r2, matrix[0][2]);
        assert_eq!(matrix.c0r3, matrix[0][3]);
        assert_eq!(matrix.c1r0, matrix[1][0]);
        assert_eq!(matrix.c1r1, matrix[1][1]);
        assert_eq!(matrix.c1r2, matrix[1][2]);
        assert_eq!(matrix.c1r3, matrix[1][3]);
        assert_eq!(matrix.c2r0, matrix[2][0]);
        assert_eq!(matrix.c2r1, matrix[2][1]);
        assert_eq!(matrix.c2r2, matrix[2][2]);
        assert_eq!(matrix.c2r3, matrix[2][3]);
        assert_eq!(matrix.c3r0, matrix[3][0]);
        assert_eq!(matrix.c3r1, matrix[3][1]);
        assert_eq!(matrix.c3r2, matrix[3][2]);
        assert_eq!(matrix.c3r3, matrix[3][3]);
    }
}
