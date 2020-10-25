use crate::base::{
    ScalarFloat,
};
use crate::matrix::{
    Matrix2x2,
    Matrix3x3,
    Matrix4x4,
};


impl<S> Matrix2x2<S> where S: ScalarFloat {
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
}

impl<S> Matrix3x3<S> where S: ScalarFloat {
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
}

impl<S> Matrix4x4<S> where S: ScalarFloat {
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
}

