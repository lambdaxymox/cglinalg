use crate::core_numeric::{
    SimdScalar,
    SimdScalarSigned,
    SimdScalarOrd,
    SimdScalarFloat,
};
use crate::constraints::{
    Const,
    DimAdd,
    DimSub,
    CanExtend,
    CanContract,
    ShapeConstraint,
};
use crate::norm::{
    Normed,
    Norm,
};
use crate::{
    impl_coords,
    impl_coords_deref,
};
use num_traits::{
    NumCast,
};

use core::fmt;
use core::ops;


/// A stack-allocated one-dimensional vector.
pub type Vector1<S> = Vector<S, 1>;

/// A stack-allocated two-dimensional vector.
pub type Vector2<S> = Vector<S, 2>;

/// A stack-allocated three-dimensional vector.
pub type Vector3<S> = Vector<S, 3>;

/// A stack-allocated four-dimensional vector.
pub type Vector4<S> = Vector<S, 4>;


#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
/// A stack-allocated vector.
pub struct Vector<S, const N: usize> {
    data: [S; N],
}

impl<S, const N: usize> Vector<S, N> {
    /// Returns the length of the the underlying array storing the vector components.
    #[inline]
    pub const fn len(&self) -> usize {
        N
    }

    /// Tests whether the number of elements in the vector is zero.
    /// 
    /// Returns `true` when the vector is zero-dimensional. Returns `false` 
    /// otherwise.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The shape of the underlying array storing the vector components.
    ///
    /// The shape is the equivalent number of columns and rows of the 
    /// array as though it represents a matrix. The order of the descriptions 
    /// of the shape of the array is **(rows, columns)**.
    #[inline]
    pub const fn shape(&self) -> (usize, usize) {
        (N, 1)
    }

    /// Get a pointer to the underlying array.
    #[inline]
    pub const fn as_ptr(&self) -> *const S {
        &self.data[0]
    }

    /// Get a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        &mut self.data[0]
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        AsRef::<[S; N]>::as_ref(self)
    }
}

impl<S, const N: usize> Vector<S, N> 
where 
    S: NumCast + Copy
{
    /// Cast a vector from one type of scalars to another type of scalars.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector4,   
    /// # };
    /// #
    /// let vector: Vector4<u32> = Vector4::new(1_u32, 2_u32, 3_u32, 4_u32);
    /// let expected: Option<Vector4<i32>> = Some(Vector4::new(1_i32, 2_i32, 3_i32, 4_i32));
    /// let result = vector.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[allow(clippy::needless_range_loop)]
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Vector<T, N>> {
        // SAFETY: Every location gets written into with a valid value of type `T`.
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut data: [T; N] = unsafe { core::mem::zeroed() };
        for i in 0..N {
            data[i] = match num_traits::cast(self.data[i]) {
                Some(value) => value,
                None => return None,
            };
        }

        Some(Vector { data })
    }
}

impl<S, const N: usize> Vector<S, N>
where
    S: Copy
{
    /// Construct a vector from a fill value.
    ///
    /// Every component of the resulting vector will have the same value
    /// supplied by the `value` argument.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,   
    /// # };
    /// #
    /// let fill_value = 3_f64;
    /// let result = Vector3::from_fill(fill_value);
    /// let expected = Vector3::new(3_f64, 3_f64, 3_f64);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub const fn from_fill(value: S) -> Self {
        Self { 
            data: [value; N],
        }
    }

    /// Map an operation on that acts on the components of a vector, returning 
    /// a vector whose components are of the new scalar type.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector4,  
    /// # };
    /// #
    /// let vector: Vector4<u32> = Vector4::new(1_u32, 2_u32, 3_u32, 4_u32);
    /// let expected: Vector4<i32> = Vector4::new(2_i32, 3_i32, 4_i32, 5_i32);
    /// let result: Vector4<i32> = vector.map(|comp| (comp + 1) as i32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[allow(unused_mut)]
    #[inline]
    pub fn map<T, F>(&self, mut op: F) -> Vector<T, N> 
    where 
        F: FnMut(S) -> T
    {
        Vector {
            data: self.data.map(op),
        }
    }
}

impl<S, const N: usize> Vector<S, N>
where
    S: SimdScalar
{
    /// Construct the zero vector.
    ///
    /// The zero vector is the vector in which all of its elements are zero.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// #
    /// let vector: Vector3<i32> = Vector3::zero();
    /// 
    /// assert_eq!(vector[0], 0);
    /// assert_eq!(vector[1], 0);
    /// assert_eq!(vector[2], 0);
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self {
            data: [S::zero(); N],
        }
    }

    /// Determine whether a vector is the zero vector.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector4,
    /// # };
    /// #
    /// let zero: Vector4<i32> = Vector4::zero();
    /// let non_zero = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
    /// 
    /// assert!(zero.is_zero());
    /// assert!(!non_zero.is_zero());
    /// ```
    #[inline]
    pub fn is_zero(&self) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for i in 0..N {
            result &= self.data[i].is_zero();
        }

        result
    }

    /// Compute the Euclidean dot product (inner product) of two vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3, 
    /// # };
    /// #
    /// let vector1 = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let vector2 = Vector3::new(4_f64, 5_f64, 6_f64);
    /// 
    /// assert_eq!(vector1.dot(&vector2), 32_f64);
    /// ```
    #[inline]
    pub fn dot(&self, other: &Self) -> S {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = S::zero();
        for i in 0..N {
            result += self.data[i] * other.data[i];
        }

        result
    }

    /// Compute the product of two vectors component-wise.
    /// 
    /// Given `N`-dimensional vectors `v1` and `v2`, the component product of `v1` and `v2` is a 
    /// `N`-dimensional vector `v3` such that
    /// ```text
    /// for all i in 0..N. v3[i] := v1[i] * v2[i]
    /// ```
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// #
    /// let v1 = Vector3::new(0_f64, 1_f64, 4_f64);
    /// let v2 = Vector3::new(5_f64, 8_f64, 3_f64);
    /// let expected = Vector3::new(0_f64, 8_f64, 12_f64);
    /// let result = v1.component_mul(&v2);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn component_mul(&self, other: &Self) -> Self {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::zero();
        for i in 0..N {
            result[i] = self.data[i] * other.data[i];
        }
 
        result
    }

    /// Compute the product of two vectors component-wise mutably in place.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// #
    /// let mut result = Vector3::new(0_f64, 1_f64, 4_f64);
    /// let other = Vector3::new(5_f64, 8_f64, 3_f64);
    /// let expected = Vector3::new(0_f64, 8_f64, 12_f64);
    /// result.component_mul_assign(&other);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn component_mul_assign(&mut self, other: &Self) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for i in 0..N {
            self.data[i] *= other.data[i];
        }
    }
}

impl<S, const N: usize> Vector<S, N> 
where 
    S: SimdScalarSigned
{
    /// Compute the negation of a vector mutably in place.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector4, 
    /// # };
    /// #
    /// let mut result = Vector4::new(1_i32, 2_i32, 3_i32, 4_i32);
    /// let expected = -result;
    /// result.neg_mut();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn neg_mut(&mut self) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for i in 0..N {
            self.data[i] = -self.data[i];
        }
    }
}

impl<S, const N: usize> Vector<S, N>
where
    S: SimdScalar
{
        /// Calculate the squared norm of a vector with respect to the **L2** (Euclidean) norm.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #      Vector3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let value = 1_f64 / f64::sqrt(3_f64);
    /// let vector = Vector3::from_fill(value);
    /// let expected = 1_f64;
    /// let result = vector.norm_squared();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn norm_squared(&self) -> S {
        self.dot(self)
    }

    /// Calculate the squared metric distance between two vectors with respect 
    /// to the metric induced by the **L2** norm.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// #
    /// let vector1 = Vector3::new(0_f64, -5_f64, 6_f64);
    /// let vector2 = Vector3::new(-3_f64, 1_f64, 2_f64);
    /// let expected = 61_f64;
    /// let result = vector1.metric_distance_squared(&vector2);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn metric_distance_squared(&self, other: &Self) -> S {
        (self - other).norm_squared()
    }

    /// Calculate the squared norm of a vector with respect to the **L2** (Euclidean) norm.
    /// 
    /// This is a synonym for [`Vector::norm_squared`].
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #      Vector3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let value = 1_f64 / f64::sqrt(3_f64);
    /// let vector = Vector3::from_fill(value);
    /// let expected = 1_f64;
    /// let result = vector.magnitude_squared();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn magnitude_squared(&self) -> S {
        self.norm_squared()
    }
}

impl<S, const N: usize> Vector<S, N>
where
    S: SimdScalarSigned
{
    /// Calculate the norm of a vector with respect to the supplied [`Norm`] type.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     L1Norm,
    /// # };
    /// #
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let norm = L1Norm::new();
    /// 
    /// assert_eq!(vector.apply_norm(&norm), 6_f64);
    /// ```
    #[inline]
    pub fn apply_norm(&self, norm: &impl Norm<Vector<S, N>, Output = S>) -> S {
        norm.norm(self)
    }

    /// Calculate the metric distance between two vectors with respect to the 
    /// supplied [`Norm`] type.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     L1Norm,
    /// # };
    /// #
    /// let vector1 = Vector3::new(0_f64, -5_f64, 6_f64);
    /// let vector2 = Vector3::new(-3_f64, 1_f64, 2_f64);
    /// let norm = L1Norm::new();
    /// let expected = 13_f64;
    /// let result = vector1.apply_metric_distance(&vector2, &norm);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn apply_metric_distance(&self, other: &Self, norm: &impl Norm<Vector<S, N>, Output = S>) -> S {
        norm.metric_distance(self, other)
    }

    /// Calculate the norm of a vector with respect to the **L1** norm.
    /// 
    /// # Examples
    /// 
    /// An example computing the **L1** norm of a vector of [`f64`] scalars.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// #
    /// let vector = Vector3::new(-2_f64, 7_f64, 8_f64);
    /// let expected = 17_f64;
    /// let result = vector.l1_norm();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// An example of computing the **L1** norm of a vector of [`i32`] scalars.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// #
    /// let vector = Vector3::new(-2_i32, 7_i32, 8_i32);
    /// let expected = 17_i32;
    /// let result = vector.l1_norm();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn l1_norm(&self) -> S {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = S::zero();
        for i in 0..N {
            result += self.data[i].abs();
        }

        result
    }
}

impl<S, const N: usize> Vector<S, N>
where
    S: SimdScalarSigned + SimdScalarOrd
{
    /// Calculate the norm of a vector with respect to the **L-infinity** norm.
    /// 
    /// # Examples
    /// 
    /// An example of computing the **L-infinity** norm of a vector of [`f64`] scalars.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector4,
    /// # };
    /// #
    /// let vector = Vector4::new(1_f64, 100_f64, 3_f64, 4_f64);
    /// let expected = 100_f64;
    /// let result = vector.linf_norm();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    /// 
    /// An example of computing the **L-infinity** norm of a vector of [`i32`] scalars.
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector4,
    /// # };
    /// #
    /// let vector = Vector4::new(1_i32, 100_i32, 3_i32, 4_i32);
    /// let expected = 100_i32;
    /// let result = vector.linf_norm();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn linf_norm(&self) -> S {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = S::zero();
        for i in 0..N {
            result = result.max(self.data[i].abs());
        }

        result
    }
}

impl<S, const N: usize> Vector<S, N>
where
    S: SimdScalarFloat
{
    /// Calculate the norm of a vector with respect to the **L2** (Euclidean) norm.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #      Vector3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let value = 1_f64 / f64::sqrt(3_f64);
    /// let vector = Vector3::from_fill(value);
    /// let expected = 1_f64;
    /// let result = vector.norm();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn norm(&self) -> S {
        self.norm_squared().sqrt()
    }

    /// Calculate the metric distance between two vectors with respect 
    /// to the metric induced by the **L2** norm.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// #
    /// let vector1 = Vector3::new(1_f64, 4_f64, 6_f64);
    /// let vector2 = Vector3::new(1_f64, 8_f64, -2_f64);
    /// let expected = f64::sqrt(80_f64);
    /// let result = vector1.metric_distance(&vector2);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn metric_distance(&self, other: &Self) -> S {
        (self - other).norm()
    }

    /// Calculate the norm of a vector with respect to the **L2** (Euclidean) norm.
    /// 
    /// This is a synonym for [`Vector::norm`].
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #      Vector3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let value = 1_f64 / f64::sqrt(3_f64);
    /// let vector = Vector3::from_fill(value);
    /// let expected = 1_f64;
    /// let result = vector.magnitude();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn magnitude(&self) -> S {
        self.norm()
    }

    /// Calculate the norm of a vector with respect to the **L2** (Euclidean) norm.
    /// 
    /// This is a synonym for [`Vector::norm`].
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #      Vector3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let value = 1_f64 / f64::sqrt(3_f64);
    /// let vector = Vector3::from_fill(value);
    /// let expected = 1_f64;
    /// let result = vector.l2_norm();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn l2_norm(&self) -> S {
        self.norm()
    }

    /// Calculate the norm of a vector with respect to the **Lp** norm, where
    /// `p` is an integer.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #      Vector3,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let value = 1_f64 / f64::sqrt(3_f64);
    /// let vector = Vector3::from_fill(value);
    /// let expected = 1_f64 / f64::powf(3_f64, 3_f64 / 10_f64);
    /// let result = vector.lp_norm(5);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn lp_norm(&self, p: u32) -> S {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = S::zero();
        for i in 0..N {
            result += self.data[i].abs().powi(p as i32);
        }
        
        result.powf(num_traits::cast((p as f64).recip()).unwrap())
    }
}

pub type UniformNorm = LinfNorm;
pub type EuclideanNorm = L2Norm;

#[derive(Copy, Clone, Debug)]
pub struct L1Norm {}

impl L1Norm {
    #[inline]
    pub const fn new() -> Self {
        Self {}
    }
}

impl<S, const N: usize> Norm<Vector<S, N>> for L1Norm 
where
    S: SimdScalarSigned
{
    type Output = S;

    #[inline]
    fn norm(&self, rhs: &Vector<S, N>) -> Self::Output {
        rhs.l1_norm()
    }

    #[inline]
    fn metric_distance(&self, lhs: &Vector<S, N>, rhs: &Vector<S, N>) -> Self::Output {
        self.norm(&(rhs - lhs))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct L2Norm {}

impl L2Norm {
    #[inline]
    pub const fn new() -> Self {
        Self {}
    }
}

impl<S, const N: usize> Norm<Vector<S, N>> for L2Norm
where 
    S: SimdScalarFloat
{
    type Output = S;

    #[inline]
    fn norm(&self, lhs: &Vector<S, N>) -> Self::Output {
        Self::Output::sqrt(Vector::dot(lhs, lhs))
    }

    #[inline]
    fn metric_distance(&self, lhs: &Vector<S, N>, rhs: &Vector<S, N>) -> Self::Output {
        self.norm(&(rhs - lhs))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LinfNorm {}

impl LinfNorm {
    #[inline]
    pub const fn new() -> Self {
        Self {}
    }
}

impl<S, const N: usize> Norm<Vector<S, N>> for LinfNorm 
where
    S: SimdScalarSigned + SimdScalarOrd
{
    type Output = S;

    #[inline]
    fn norm(&self, lhs: &Vector<S, N>) -> Self::Output {
        lhs.linf_norm()
    }

    #[inline]
    fn metric_distance(&self, lhs: &Vector<S, N>, rhs: &Vector<S, N>) -> Self::Output {
        self.norm(&(rhs - lhs))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LpNorm {
    pub p: u32,
}

impl LpNorm {
    #[inline]
    pub const fn new(p: u32) -> Self {
        Self { 
            p,
        }
    }
}

impl<S, const N: usize> Norm<Vector<S, N>> for LpNorm 
where
    S: SimdScalarFloat
{
    type Output = S;

    #[inline]
    fn norm(&self, lhs: &Vector<S, N>) -> Self::Output {
        lhs.lp_norm(self.p)
    }

    #[inline]
    fn metric_distance(&self, lhs: &Vector<S, N>, rhs: &Vector<S, N>) -> Self::Output {
        self.norm(&(rhs - lhs))
    }
}


impl<S, const N: usize> Vector<S, N>
where
    S: SimdScalarFloat
{
    /// Returns `true` if the elements of a vector are all finite. 
    /// Otherwise, it returns `false`. 
    ///
    /// A vector is finite when all of its elements are finite. This is useful 
    /// for vector and matrix types working with fixed precision floating point 
    /// values.
    ///
    /// # Example (Finite Vector)
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector4,
    /// # };
    /// #
    /// let v = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
    ///
    /// assert!(v.is_finite()); 
    /// ```
    ///
    /// # Example (Not A Finite Vector)
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector4,
    /// # };
    /// #
    /// let w = Vector4::new(1_f64, f64::NAN, f64::NEG_INFINITY, 4_f64);
    ///
    /// assert!(!w.is_finite()); 
    /// ```
    #[inline]
    pub fn is_finite(&self) -> bool {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = true;
        for i in 0..N {
            result &= self.data[i].is_finite();
        }

        result
    }

    /// Linearly interpolate between the two vectors `self` and `other`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,  
    /// # };
    /// #
    /// let v0 = Vector3::new(0_f64, 0_f64, 0_f64);
    /// let v1 = Vector3::new(10_f64, 20_f64, 30_f64);
    /// let amount = 0.7_f64;
    /// let expected = Vector3::new(7_f64, 14_f64, 21_f64);
    /// let result = v0.lerp(&v1, amount);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Self, amount: S) -> Self {
        self + ((other - self) * amount)
    }

    /// Compute the projection of the vector `self` onto the vector
    /// `other`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3, 
    /// #     Normed,
    /// # };
    /// # 
    /// let vector = Vector3::new(1_f64 / 2_f64, f64::sqrt(3_f64) / 2_f64, 2_f64);
    /// let unit_x = Vector3::unit_x();
    /// let unit_y = Vector3::unit_y();
    /// let unit_z = Vector3::unit_z();
    /// let projected_x = vector.project(&unit_x);
    /// let projected_y = vector.project(&unit_y);
    /// let projected_z = vector.project(&unit_z);
    ///
    /// assert_eq!(projected_x, vector.x * unit_x);
    /// assert_eq!(projected_y, vector.y * unit_y);
    /// assert_eq!(projected_z, vector.z * unit_z);
    /// ```
    #[inline]
    pub fn project(&self, other: &Self) -> Self {
        other * (self.dot(other) / other.norm_squared())
    }

    /// Reflect a vector about a normal vector.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Normed,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq, 
    /// # };
    /// #
    /// let vector = Vector2::new(1_f64, 1_f64);
    /// let normal = Vector2::new(-1_f64 / 2_f64, 1_f64);
    /// let expected = Vector2::new(7_f64 / 5_f64, 1_f64 / 5_f64);
    /// let result = vector.reflect(&normal);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// assert_relative_eq!(result.norm(), expected.norm(), epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn reflect(&self, normal: &Self) -> Self {
        let two = S::one() + S::one();
        let unit_normal = normal.normalize();

        self - unit_normal * (two * self.dot(&unit_normal))
    }

    /// Compute the component-wise minimum of two vectors.
    /// 
    /// Given two vectors `v1` and `v2`, the minimum of `v1` and `v2` is a vector `v3`
    /// such that
    /// ```text
    /// for all i in 0..N. v3[i] := min(v1[i], v2[i])
    /// ```
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// #
    /// let v1 = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let v2 = Vector3::new(-1_f64, 5_f64, 0_f64);
    /// let expected = Vector3::new(-1_f64, 2_f64, 0_f64);
    /// let result = v1.component_min(&v2);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn component_min(&self, other: &Self) -> Self {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::zero();
        for i in 0..N {
            result[i] = S::min(self.data[i], other.data[i]);
        }

        result
    }

    /// Compute the component-wise maximum of two vectors.
    /// 
    /// Given two vectors `v1` and `v2`, the minimum of `v1` and `v2` is a vector `v3`
    /// such that
    /// ```text
    /// for all i in 0..N. v3[i] := max(v1[i], v2[i])
    /// ```
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// #
    /// let v1 = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let v2 = Vector3::new(-1_f64, 5_f64, 0_f64);
    /// let expected = Vector3::new(1_f64, 5_f64, 3_f64);
    /// let result = v1.component_max(&v2);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn component_max(&self, other: &Self) -> Self {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::zero();
        for i in 0..N {
            result[i] = S::max(self.data[i], other.data[i]);
        }

        result
    }
}

impl<S, const N: usize, const NPLUS1: usize> Vector<S, N>
where
    S: SimdScalar,
    ShapeConstraint: CanExtend<Const<N>, Const<NPLUS1>>,
    ShapeConstraint: DimAdd<Const<N>, Const<1>, Output = Const<NPLUS1>>,
    ShapeConstraint: DimSub<Const<NPLUS1>, Const<1>, Output = Const<N>>
{
    /// Extend a vector into a vector one dimension higher using the supplied 
    /// last element `last_element`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let v = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let expected = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
    /// let result = v.extend(4_f64);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn extend(&self, last_element: S) -> Vector<S, NPLUS1> {
        // SAFETY: The output vector has length `N + 1` with `last_element` in the 
        // component `N` of the output vector.
        let mut result = Vector::default();
        for i in 0..N {
            result.data[i] = self.data[i];
        }

        result.data[N] = last_element;

        result
    }

    /// Compute the coordinates of a vector in projective space.
    ///
    /// The function appends a `0` to the vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let vector = Vector3::new(1_i32, 2_i32, 3_i32);
    /// let expected = Vector4::new(1_i32, 2_i32, 3_i32, 0_i32);
    /// let result = vector.to_homogeneous();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_homogeneous(&self) -> Vector<S, NPLUS1> {
        self.extend(S::zero())
    }
}

impl<S, const N: usize, const NMINUS1: usize> Vector<S, N>
where
    S: SimdScalar,
    ShapeConstraint: CanContract<Const<N>, Const<NMINUS1>>,
    ShapeConstraint: DimSub<Const<N>, Const<1>, Output = Const<NMINUS1>>,
    ShapeConstraint: DimAdd<Const<NMINUS1>, Const<1>, Output = Const<N>>
{
    /// Contract a vector to a vector one dimension smaller the last component.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let v = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
    /// let expected = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let result = v.contract();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn contract(&self) -> Vector<S, NMINUS1> {
        // SAFETY: The output vector has length `N - 1`.
        let mut result = Vector::default();
        for i in 0..(N - 1) {
            result.data[i] = self.data[i];
        }

        result
    }

    /// Compute the coordinates of a projective vector in Euclidean space.
    ///
    /// The function removes a `0` from the end of the vector, otherwise it
    /// returns `None`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector2,
    /// #     Vector3, 
    /// # };
    /// #
    /// let vector = Vector3::new(1_i32, 2_i32, 0_i32);
    /// let expected = Some(Vector2::new(1_i32, 2_i32));
    /// let result = vector.from_homogeneous();
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vector = Vector3::new(1_i32, 2_i32, 1_i32);
    /// let expected: Option<Vector2<i32>> = None;
    /// let result = vector.from_homogeneous();
    ///
    /// assert!(result.is_none());
    /// ```
    #[inline]
    pub fn from_homogeneous(&self) -> Option<Vector<S, NMINUS1>> {
        if self.data[N - 1].is_zero() {
            Some(self.contract())
        } else {
            None
        }
    }
}

impl<S, const N: usize> AsRef<[S; N]> for Vector<S, N> {
    #[inline]
    fn as_ref(&self) -> &[S; N] {
        unsafe {
            &*(self as *const Vector<S, N> as *const [S; N])
        }
    }
}

impl<S, const N: usize> AsMut<[S; N]> for Vector<S, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; N] {
        unsafe {
            &mut *(self as *mut Vector<S, N> as *mut [S; N])
        }
    }
}

impl<S, const N: usize> AsRef<[[S; N]; 1]> for Vector<S, N> {
    #[inline]
    fn as_ref(&self) -> &[[S; N]; 1] {
        unsafe {
            &*(self as *const Vector<S, N> as *const [[S; N]; 1])
        }
    }
}

impl<S, const N: usize> AsMut<[[S; N]; 1]> for Vector<S, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [[S; N]; 1] {
        unsafe {
            &mut *(self as *mut Vector<S, N> as *mut [[S; N]; 1])
        }
    }
}

impl<S, const N: usize> Default for Vector<S, N>
where
    S: SimdScalar
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<S, const N: usize> From<[S; N]> for Vector<S, N>
where
    S: Copy
{
    #[inline]
    fn from(data: [S; N]) -> Self {
        Self { data }
    }
}

impl<S, const N: usize> From<&[S; N]> for Vector<S, N> 
where 
    S: Copy
{
    #[inline]
    fn from(data: &[S; N]) -> Self {
        Self {
            data: *data
        }
    }
}

impl<'a, S, const N: usize> From<&'a [S; N]> for &'a Vector<S, N> 
where 
    S: Copy
{
    #[inline]
    fn from(data: &'a [S; N]) -> &'a Vector<S, N> {
        unsafe { 
            &*(data as *const [S; N] as *const Vector<S, N>)    
        }
    }
}

macro_rules! impl_vector_index_ops {
    ($IndexType:ty, $Output:ty) => {
        impl<S, const N: usize> ops::Index<$IndexType> for Vector<S, N> {
            type Output = $Output;

            #[inline]
            fn index(&self, index: $IndexType) -> &Self::Output {
                let v: &[S; N] = self.as_ref();
                &v[index]
            }
        }

        impl<S, const N: usize> ops::IndexMut<$IndexType> for Vector<S, N> {
            #[inline]
            fn index_mut(&mut self, index: $IndexType) -> &mut Self::Output {
                let v: &mut [S; N] = self.as_mut();
                &mut v[index]
            }
        }
    }
}

impl_vector_index_ops!(usize, S);
impl_vector_index_ops!(ops::Range<usize>, [S]);
impl_vector_index_ops!(ops::RangeTo<usize>, [S]);
impl_vector_index_ops!(ops::RangeFrom<usize>, [S]);
impl_vector_index_ops!(ops::RangeFull, [S]);


impl<S, const N: usize> fmt::Display for Vector<S, N> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Vector{} [", N).unwrap();
        for i in 0..(N - 1) {
            write!(formatter, "{}, ", self.data[i]).unwrap();
        }
        write!(formatter, "{}]", self.data[N - 1])
    }
}


impl<S, const N: usize> approx::AbsDiffEq for Vector<S, N> 
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
        for i in 0..N {
            result &= S::abs_diff_eq(&self.data[i], &other.data[i], epsilon);
        }

        result
    }
}

impl<S, const N: usize> approx::RelativeEq for Vector<S, N> 
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
        for i in 0..N {
            result &= S::relative_eq(&self.data[i], &other.data[i], epsilon, max_relative);
        }

        result
    }
}

impl<S, const N: usize> approx::UlpsEq for Vector<S, N> 
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
        for i in 0..N {
            result &= S::ulps_eq(&self.data[i], &other.data[i], epsilon, max_ulps);
        }

        result
    }
}

impl<S, const N: usize> Normed for Vector<S, N> 
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
    fn scale(&self, norm: Self::Output) -> Self {
        self * (norm / self.norm())
    }

    #[inline]
    fn scale_mut(&mut self, norm: Self::Output) {
        *self = self.scale(norm);
    }

    #[inline]
    fn unscale(&self, norm: Self::Output) -> Self {
        self * (Self::Output::one() / norm)
    }

    #[inline]
    fn unscale_mut(&mut self, norm: Self::Output) {
        *self = self.unscale(norm);
    }
    
    #[inline]
    fn normalize(&self) -> Self {
        self * (Self::Output::one() / self.norm())
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
    fn distance_squared(&self, other: &Vector<S, N>) -> Self::Output {
        self.metric_distance_squared(other)
    }

    #[inline]
    fn distance(&self, other: &Self) -> Self::Output {
        self.metric_distance(other)
    }
}

macro_rules! impl_scalar_vector_mul_ops {
    ($($Lhs:ty),* $(,)*) => {$(
        impl<const N: usize> ops::Mul<Vector<$Lhs, N>> for $Lhs {
            type Output = Vector<$Lhs, N>;

            #[inline]
            fn mul(self, other: Vector<$Lhs, N>) -> Self::Output {
                // PERFORMANCE: The const loop should get unrolled during optimization.
                let mut result = Vector::zero();
                for i in 0..N {
                    result[i] = self * other.data[i];
                }

                result
            }
        }

        impl<const N: usize> ops::Mul<&Vector<$Lhs, N>> for $Lhs {
            type Output = Vector<$Lhs, N>;

            #[inline]
            fn mul(self, other: &Vector<$Lhs, N>) -> Self::Output {
                // PERFORMANCE: The const loop should get unrolled during optimization.
                let mut result = Vector::zero();
                for i in 0..N {
                    result[i] = self * other.data[i];
                }

                result
            }
        }

        impl<const N: usize> ops::Mul<Vector<$Lhs, N>> for &$Lhs {
            type Output = Vector<$Lhs, N>;

            #[inline]
            fn mul(self, other: Vector<$Lhs, N>) -> Self::Output {
                // PERFORMANCE: The const loop should get unrolled during optimization.
                let mut result = Vector::zero();
                for i in 0..N {
                    result[i] = self * other.data[i];
                }

                result
            }
        }

        impl<'a, 'b, const N: usize> ops::Mul<&'b Vector<$Lhs, N>> for &'a $Lhs {
            type Output = Vector<$Lhs, N>;

            #[inline]
            fn mul(self, other: &'b Vector<$Lhs, N>) -> Self::Output {
                // PERFORMANCE: The const loop should get unrolled during optimization.
                let mut result = Vector::zero();
                for i in 0..N {
                    result[i] = self * other.data[i];
                }

                result
            }
        }
    )*}
}

impl_scalar_vector_mul_ops!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);


macro_rules! impl_vector_scalar_binary_ops {
    ($OpType:ident, $op:ident) => {
        impl<S, const N: usize> ops::$OpType<S> for Vector<S, N> 
        where 
            S: SimdScalar 
        {
            type Output = Vector<S, N>;

            #[inline]
            fn $op(self, other: S) -> Self::Output {
                // PERFORMANCE: The const loop should get unrolled during optimization.
                let mut result = Vector::zero();
                for i in 0..N {
                    result[i] = self.data[i].$op(other);
                }

                result
            }
        }

        impl<S, const N: usize> ops::$OpType<S> for &Vector<S, N> 
        where 
            S: SimdScalar 
        {
            type Output = Vector<S, N>;

            #[inline]
            fn $op(self, other: S) -> Self::Output {
                // PERFORMANCE: The const loop should get unrolled during optimization.
                let mut result = Vector::zero();
                for i in 0..N {
                    result[i] = self.data[i].$op(other);
                }

                result
            }
        }
    }
}

impl_vector_scalar_binary_ops!(Mul, mul);
impl_vector_scalar_binary_ops!(Div, div);
impl_vector_scalar_binary_ops!(Rem, rem);


impl<S, const N: usize> ops::Add<Vector<S, N>> for Vector<S, N> 
where 
    S: SimdScalar 
{
    type Output = Vector<S, N>;

    #[inline]
    fn add(self, other: Vector<S, N>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for i in 0..N {
            result[i] = self.data[i] + other.data[i];
        }

        result
    }
}

impl<S, const N: usize> ops::Add<&Vector<S, N>> for Vector<S, N> 
where 
    S: SimdScalar 
{
    type Output = Vector<S, N>;

    #[inline]
    fn add(self, other: &Vector<S, N>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for i in 0..N {
            result[i] = self.data[i] + other.data[i];
        }

        result
    }
}

impl<S, const N: usize> ops::Add<Vector<S, N>> for &Vector<S, N> 
where 
    S: SimdScalar 
{
    type Output = Vector<S, N>;

    #[inline]
    fn add(self, other: Vector<S, N>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for i in 0..N {
            result[i] = self.data[i] + other.data[i];
        }

        result
    }
}

impl<'a, 'b, S, const N: usize> ops::Add<&'b Vector<S, N>> for &'a Vector<S, N> 
where 
    S: SimdScalar 
{
    type Output = Vector<S, N>;

    #[inline]
    fn add(self, other: &'b Vector<S, N>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for i in 0..N {
            result[i] = self.data[i] + other.data[i];
        }

        result
    }
}

impl<S, const N: usize> ops::Sub<Vector<S, N>> for Vector<S, N> 
where 
    S: SimdScalar 
{
    type Output = Vector<S, N>;

    #[inline]
    fn sub(self, other: Vector<S, N>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for i in 0..N {
            result[i] = self.data[i] - other.data[i];
        }

        result
    }
}

impl<S, const N: usize> ops::Sub<&Vector<S, N>> for Vector<S, N> 
where 
    S: SimdScalar 
{
    type Output = Vector<S, N>;

    #[inline]
    fn sub(self, other: &Vector<S, N>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for i in 0..N {
            result[i] = self.data[i] - other.data[i];
        }

        result
    }
}

impl<S, const N: usize> ops::Sub<Vector<S, N>> for &Vector<S, N> 
where 
    S: SimdScalar 
{
    type Output = Vector<S, N>;

    #[inline]
    fn sub(self, other: Vector<S, N>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for i in 0..N {
            result[i] = self.data[i] - other.data[i];
        }

        result
    }
}

impl<'a, 'b, S, const N: usize> ops::Sub<&'b Vector<S, N>> for &'a Vector<S, N> 
where 
    S: SimdScalar 
{
    type Output = Vector<S, N>;

    #[inline]
    fn sub(self, other: &'b Vector<S, N>) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for i in 0..N {
            result[i] = self.data[i] - other.data[i];
        }

        result
    }
}

impl<S, const N: usize> ops::AddAssign<Vector<S, N>> for Vector<S, N> 
where 
    S: SimdScalar
{
    #[inline]
    fn add_assign(&mut self, other: Vector<S, N>) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for i in 0..N {
            self.data[i] += other.data[i];
        }
    }
}

impl<S, const N: usize> ops::AddAssign<&Vector<S, N>> for Vector<S, N> 
where 
    S: SimdScalar
{
    #[inline]
    fn add_assign(&mut self, other: &Vector<S, N>) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for i in 0..N {
            self.data[i] += other.data[i];
        }
    }
}

impl<S, const N: usize> ops::SubAssign<Vector<S, N>> for Vector<S, N> 
where 
    S: SimdScalar
{
    #[inline]
    fn sub_assign(&mut self, other: Vector<S, N>) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for i in 0..N {
            self.data[i] -= other.data[i];
        }
    }
}

impl<S, const N: usize> ops::SubAssign<&Vector<S, N>> for Vector<S, N> 
where 
    S: SimdScalar
{
    #[inline]
    fn sub_assign(&mut self, other: &Vector<S, N>) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for i in 0..N {
            self.data[i] -= other.data[i];
        }
    }
}

impl<S, const N: usize> ops::MulAssign<S> for Vector<S, N> 
where 
    S: SimdScalar
{
    #[inline]
    fn mul_assign(&mut self, other: S) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for i in 0..N {
            self.data[i] *= other;
        }
    }
}

impl<S, const N: usize> ops::DivAssign<S> for Vector<S, N> 
where 
    S: SimdScalar 
{
    #[inline]
    fn div_assign(&mut self, other: S) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for i in 0..N {
            self.data[i] /= other;
        }
    }
}

impl<S, const N: usize> ops::RemAssign<S> for Vector<S, N> 
where 
    S: SimdScalar 
{
    #[inline]
    fn rem_assign(&mut self, other: S) {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        for i in 0..N {
            self.data[i] %= other;
        }
    }
}

impl<S, const N: usize> ops::Neg for Vector<S, N> 
where 
    S: SimdScalarSigned
{
    type Output = Vector<S, N>;

    #[inline]
    fn neg(self) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for i in 0..N {
            result[i] = -self.data[i];
        }

        result
    }
}

impl<S, const N: usize> ops::Neg for &Vector<S, N>
where 
    S: SimdScalarSigned
{
    type Output =  Vector<S, N>;

    #[inline]
    fn neg(self) -> Self::Output {
        // PERFORMANCE: The const loop should get unrolled during optimization.
        let mut result = Self::Output::zero();
        for i in 0..N {
            result[i] = -self.data[i];
        }

        result
    }
}


impl<S> Vector1<S> {
    /// Construct a new vector.
    #[inline]
    pub const fn new(x: S) -> Self {
        Self { 
            data: [x], 
        }
    }
}

impl<S> Vector1<S> 
where 
    S: SimdScalar
{
    /// Returns the **x-axis** unit vector, a unit vector with the **x-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_x() -> Self {
        Vector1::new(S::one())
    }
}


impl<S> Vector2<S> {
    /// Construct a new vector.
    #[inline]
    pub const fn new(x: S, y: S) -> Self {
        Self { 
            data: [x, y],
        }
    }
}

impl<S> Vector2<S> 
where 
    S: SimdScalar 
{
    /// Returns the **x-axis** unit vector, a unit vector with the **x-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_x() -> Self {
        Self::new(S::one(), S::zero())
    }

    /// Returns the **y-axis** unit vector, a unit vector with the **y-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_y() -> Self {
        Self::new(S::zero(), S::one())
    }
}

impl<S> Vector3<S> {
    /// Construct a new vector.
    #[inline]
    pub const fn new(x: S, y: S, z: S) -> Self {
        Self { 
            data: [x, y, z],
        }
    }
}

impl<S> Vector3<S> 
where 
    S: SimdScalar
{
    /// Returns the **x-axis** unit vector, a unit vector with the **x-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_x() -> Self {
        Self::new(S::one(), S::zero(), S::zero())
    }

    /// Returns the **y-axis** unit vector, a unit vector with the **y-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_y() -> Self {
        Self::new(S::zero(), S::one(), S::zero())
    }
    
    /// Returns the **z-axis** unit vector, a unit vector with the **z-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_z() -> Self {
        Self::new(S::zero(), S::zero(), S::one())
    }

    /// Compute the cross product of two three-dimensional vectors. 
    ///
    /// For the vector dimensions used in computer graphics 
    /// (up to four dimensions), the cross product is well-defined only in 
    /// three dimensions. The cross product is a form of vector 
    /// multiplication that computes a vector normal to the plane swept out by 
    /// the two vectors. The norm of this vector is the area of the 
    /// parallelogram swept out by the two vectors. 
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,  
    /// # };
    /// #
    /// let vector1 = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let vector2 = Vector3::new(4_f64, 5_f64, 6_f64);
    /// let cross = vector1.cross(&vector2);
    ///
    /// assert_eq!(cross, Vector3::new(-3_f64, 6_f64, -3_f64));
    /// // The cross product is perpendicular to the two input vectors.
    /// assert_eq!(cross.dot(&vector1), 0_f64);
    /// assert_eq!(cross.dot(&vector2), 0_f64);
    ///
    /// // The cross product of a vector with itself is zero.
    /// let vector = Vector3::new(1_f64, 2_f64, 3_f64);
    ///
    /// assert_eq!(vector.cross(&vector), Vector3::zero());
    /// ```
    #[inline]
    pub fn cross(&self, other: &Self) -> Self {
        let x = self.data[1] * other.data[2] - self.data[2] * other.data[1];
        let y = self.data[2] * other.data[0] - self.data[0] * other.data[2];
        let z = self.data[0] * other.data[1] - self.data[1] * other.data[0];
    
        Vector3::new(x, y, z)
    }

    /// Compute the scalar triple product of three three-dimensional vectors.
    /// 
    /// The scalar triple product of three vectors `u`, `v`, and `w` is the 
    /// signed volume of the parallelepiped formed by the three vectors. In
    /// symbols, the triple product is given by
    /// ```text
    /// triple(u, v, w) = dot(cross(u, v), w)
    /// ```
    /// The order of the dot product and the cross product can be interchanged
    /// in the triple product without affecting the end result
    /// ```text
    /// dot(cross(u, v), w) = dot(u, cross(v, w))
    /// ```
    /// The scalar triple product also remains constant under cyclic permutations
    /// of its three arguments. Given that
    /// ```text
    /// dot(cross(u, v), w) = dot(cross(v, w), u) = dot(cross(w, v), u)
    /// ```
    /// it follows that
    /// ```text
    /// triple(u, v, w) = triple(v, w, u) = triple(w, u, v)
    /// ```
    /// Under noncyclic permutations, the scalar triple product changes sign
    /// due to the cross product
    /// ```text
    /// triple(u, v, w) = -triple(u, w, v) = -triple(v, u, w) = -triple(w, v, u)
    /// ```
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Vector3,
    /// # };
    /// #
    /// let u = Vector3::new(1_f64, 2_f64, 3_f64);
    /// let v = Vector3::new(6_f64, 87_f64, 222_f64);
    /// let w = Vector3::new(52_f64, 85_f64, 108_f64);
    /// let expected = 276_f64;
    /// let result = u.triple(&v, &w);
    /// 
    /// assert_eq!(result, expected);
    /// 
    /// assert_eq!(u.triple(&v, &w), v.triple(&w, &u));
    /// assert_eq!(u.triple(&v, &w), w.triple(&u, &v));
    /// assert_eq!(u.triple(&v, &w), -u.triple(&w, &v));
    /// assert_eq!(u.triple(&v, &w), -v.triple(&u, &w));
    /// assert_eq!(u.triple(&v, &w), -w.triple(&v, &u));
    /// ```
    #[inline]
    pub fn triple(&self, other1: &Vector3<S>, other2: &Vector3<S>) -> S {
        self.cross(other1).dot(other2)
    }
}


impl<S> Vector4<S> {
    /// Construct a new vector.
    #[inline]
    pub const fn new(x: S, y: S, z: S, w: S) -> Self {
        Self { 
            data: [x, y, z, w],
        }
    }
}

impl<S> Vector4<S> 
where 
    S: SimdScalar
{
    /// Returns the **x-axis** unit vector, a unit vector with the **x-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_x() -> Self {
        Self::new(S::one(), S::zero(), S::zero(), S::zero())
    }

    /// Returns the **y-axis** unit vector, a unit vector with the **y-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_y() -> Self {
        Self::new(S::zero(), S::one(), S::zero(), S::zero())
    }
    
    /// Returns the **z-axis** unit vector, a unit vector with the **z-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_z() -> Self {
        Self::new(S::zero(), S::zero(), S::one(), S::zero())
    }

    /// Returns the **w-axis** unit vector, a unit vector with the **w-component**
    /// component as a `1` and the rest of the components are zero.
    #[inline]
    pub fn unit_w() -> Self {
        Self::new(S::zero(), S::zero(), S::zero(), S::one())
    }
}

impl<S> From<S> for Vector1<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: S) -> Self {
        Self::new(v)
    }
}

impl<S> From<(S,)> for Vector1<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: (S,)) -> Self {
        Self::new(v.0)
    }
}

impl<S> From<&(S,)> for Vector1<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: &(S,)) -> Self  {
        Self::new(v.0)
    }
}

impl<'a, S> From<&'a (S,)> for &'a Vector1<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: &'a (S,)) -> &'a Vector1<S> {
        unsafe { 
            &*(v as *const (S,) as *const Vector1<S>)
        }
    }
}

impl<S> From<(S, S)> for Vector2<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: (S, S)) -> Self {
        Self::new(v.0, v.1)
    }
}

impl<S> From<&(S, S)> for Vector2<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: &(S, S)) -> Self {
        Self::new(v.0, v.1)
    }
}

impl<'a, S> From<&'a (S, S)> for &'a Vector2<S> 
where
    S: Copy
{
    #[inline]
    fn from(v: &'a (S, S)) -> &'a Vector2<S> {
        unsafe {
            &*(v as *const (S, S) as *const Vector2<S>)
        }
    }
}

impl<S> From<(S, S, S)> for Vector3<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: (S, S, S)) -> Self {
        Self::new(v.0, v.1, v.2)
    }
}

impl<S> From<&(S, S, S)> for Vector3<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: &(S, S, S)) -> Self {
        Self::new(v.0, v.1, v.2)
    }
}

impl<'a, S> From<&'a (S, S, S)> for &'a Vector3<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: &'a (S, S, S)) -> &'a Vector3<S> {
        unsafe { 
            &*(v as *const (S, S, S) as *const Vector3<S>)
        }
    }
}

impl<S> From<Vector4<S>> for Vector3<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: Vector4<S>) -> Self {
        Self::new(v.data[0], v.data[1], v.data[2])
    }
}

impl<S> From<&Vector4<S>> for Vector3<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: &Vector4<S>) -> Self {
        Self::new(v.data[0], v.data[1], v.data[2])
    }
}

impl<S> From<(S, S, S, S)> for Vector4<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: (S, S, S, S)) -> Self {
        Self::new(v.0, v.1, v.2, v.3)
    }
}

impl<S> From<&(S, S, S, S)> for Vector4<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: &(S, S, S, S)) -> Self {
        Self::new(v.0, v.1, v.2, v.3)
    }
}

impl<'a, S> From<&'a (S, S, S, S)> for &'a Vector4<S> 
where 
    S: Copy
{
    #[inline]
    fn from(v: &'a (S, S, S, S)) -> &'a Vector4<S> {
        unsafe { 
            &*(v as *const (S, S, S, S) as *const Vector4<S>)
        }
    }
}

impl_coords!(VectorCoordsX, { x });
impl_coords_deref!(Vector1, VectorCoordsX);

impl_coords!(VectorCoordsXY, { x, y });
impl_coords_deref!(Vector2, VectorCoordsXY);

impl_coords!(VectorCoordsXYZ, { x, y, z });
impl_coords_deref!(Vector3, VectorCoordsXYZ);

impl_coords!(VectorCoordsXYZW, { x, y, z, w });
impl_coords_deref!(Vector4, VectorCoordsXYZW);


macro_rules! impl_as_ref_ops {
    ($VecType:ty, $RefType:ty) => {
        impl<S> AsRef<$RefType> for $VecType {
            #[inline]
            fn as_ref(&self) -> &$RefType {
                unsafe {
                    &*(self as *const $VecType as *const $RefType)
                }
            }
        }

        impl<S> AsMut<$RefType> for $VecType {
            #[inline]
            fn as_mut(&mut self) -> &mut $RefType {
                unsafe {
                    &mut *(self as *mut $VecType as *mut $RefType)
                }
            }
        }
    }
}

impl_as_ref_ops!(Vector1<S>, S);
impl_as_ref_ops!(Vector1<S>, (S,));
impl_as_ref_ops!(Vector2<S>, (S, S));
impl_as_ref_ops!(Vector3<S>, (S, S, S));
impl_as_ref_ops!(Vector4<S>, (S, S, S, S));


macro_rules! impl_swizzle {
    ($name:ident() => $VectorN:ident => $Output:ident { $($i:expr),+ }) => {
        impl<S> $VectorN<S> where S: Copy {
            /// Construct a new vector from the components of the input vector.
            #[inline]
            pub fn $name(&self) -> $Output<S> {
                $Output::new(
                    $(self.data[$i]),*
                )
            }
        }
    }
}

impl_swizzle!(x() => Vector1 => Vector1 { 0 });

impl_swizzle!(x()  => Vector2 => Vector1 { 0 });
impl_swizzle!(y()  => Vector2 => Vector1 { 1 });
impl_swizzle!(xx() => Vector2 => Vector2 { 0, 0 });
impl_swizzle!(xy() => Vector2 => Vector2 { 0, 1 });
impl_swizzle!(yx() => Vector2 => Vector2 { 1, 0 });
impl_swizzle!(yy() => Vector2 => Vector2 { 1, 1 });

impl_swizzle!(x()  => Vector3 => Vector1 { 0 });
impl_swizzle!(y()  => Vector3 => Vector1 { 1 });
impl_swizzle!(z()  => Vector3 => Vector1 { 2 });
impl_swizzle!(xx() => Vector3 => Vector2 { 0, 0 });
impl_swizzle!(xy() => Vector3 => Vector2 { 0, 1 });
impl_swizzle!(xz() => Vector3 => Vector2 { 0, 2 });
impl_swizzle!(yx() => Vector3 => Vector2 { 1, 0 });
impl_swizzle!(yy() => Vector3 => Vector2 { 1, 1 });
impl_swizzle!(yz() => Vector3 => Vector2 { 1, 2 });
impl_swizzle!(zx() => Vector3 => Vector2 { 2, 0 });
impl_swizzle!(zy() => Vector3 => Vector2 { 2, 1 });
impl_swizzle!(zz() => Vector3 => Vector2 { 2, 2 });

impl_swizzle!(xxx() => Vector3 => Vector3 { 0, 0, 0 });
impl_swizzle!(xxy() => Vector3 => Vector3 { 0, 0, 1 });
impl_swizzle!(xxz() => Vector3 => Vector3 { 0, 0, 2 });
impl_swizzle!(xyx() => Vector3 => Vector3 { 0, 1, 0 });
impl_swizzle!(xyy() => Vector3 => Vector3 { 0, 1, 1 });
impl_swizzle!(xyz() => Vector3 => Vector3 { 0, 1, 2 });
impl_swizzle!(xzx() => Vector3 => Vector3 { 0, 2, 0 });
impl_swizzle!(xzy() => Vector3 => Vector3 { 0, 2, 1 });
impl_swizzle!(xzz() => Vector3 => Vector3 { 0, 2, 2 });
impl_swizzle!(yxx() => Vector3 => Vector3 { 1, 0, 0 });
impl_swizzle!(yxy() => Vector3 => Vector3 { 1, 0, 1 });
impl_swizzle!(yxz() => Vector3 => Vector3 { 1, 0, 2 });
impl_swizzle!(yyx() => Vector3 => Vector3 { 1, 1, 0 });
impl_swizzle!(yyy() => Vector3 => Vector3 { 1, 1, 1 });
impl_swizzle!(yyz() => Vector3 => Vector3 { 1, 1, 2 });
impl_swizzle!(yzx() => Vector3 => Vector3 { 1, 2, 0 });
impl_swizzle!(yzy() => Vector3 => Vector3 { 1, 2, 1 });
impl_swizzle!(yzz() => Vector3 => Vector3 { 1, 2, 2 });
impl_swizzle!(zxx() => Vector3 => Vector3 { 2, 0, 0 });
impl_swizzle!(zxy() => Vector3 => Vector3 { 2, 0, 1 });
impl_swizzle!(zxz() => Vector3 => Vector3 { 2, 0, 2 });
impl_swizzle!(zyx() => Vector3 => Vector3 { 2, 1, 0 });
impl_swizzle!(zyy() => Vector3 => Vector3 { 2, 1, 1 });
impl_swizzle!(zyz() => Vector3 => Vector3 { 2, 1, 2 });
impl_swizzle!(zzx() => Vector3 => Vector3 { 2, 2, 0 });
impl_swizzle!(zzy() => Vector3 => Vector3 { 2, 2, 1 });
impl_swizzle!(zzz() => Vector3 => Vector3 { 2, 2, 2 });


impl_swizzle!(x() => Vector4 => Vector1 { 0 });
impl_swizzle!(y() => Vector4 => Vector1 { 1 });
impl_swizzle!(z() => Vector4 => Vector1 { 2 });
impl_swizzle!(w() => Vector4 => Vector1 { 3 });

impl_swizzle!(xx() => Vector4 => Vector2 { 0, 0 });
impl_swizzle!(xy() => Vector4 => Vector2 { 0, 1 });
impl_swizzle!(xz() => Vector4 => Vector2 { 0, 2 });
impl_swizzle!(xw() => Vector4 => Vector2 { 0, 3 });
impl_swizzle!(yx() => Vector4 => Vector2 { 1, 0 });
impl_swizzle!(yy() => Vector4 => Vector2 { 1, 1 });
impl_swizzle!(yz() => Vector4 => Vector2 { 1, 2 });
impl_swizzle!(yw() => Vector4 => Vector2 { 1, 3 });
impl_swizzle!(zx() => Vector4 => Vector2 { 2, 0 });
impl_swizzle!(zy() => Vector4 => Vector2 { 2, 1 });
impl_swizzle!(zz() => Vector4 => Vector2 { 2, 2 });
impl_swizzle!(zw() => Vector4 => Vector2 { 2, 3 });
impl_swizzle!(wx() => Vector4 => Vector2 { 3, 0 });
impl_swizzle!(wy() => Vector4 => Vector2 { 3, 1 });
impl_swizzle!(wz() => Vector4 => Vector2 { 3, 2 });
impl_swizzle!(ww() => Vector4 => Vector2 { 3, 3 });

impl_swizzle!(xxx() => Vector4 => Vector3 { 0, 0, 0 });
impl_swizzle!(xxy() => Vector4 => Vector3 { 0, 0, 1 });
impl_swizzle!(xxz() => Vector4 => Vector3 { 0, 0, 2 });
impl_swizzle!(xxw() => Vector4 => Vector3 { 0, 0, 3 });
impl_swizzle!(xyx() => Vector4 => Vector3 { 0, 1, 0 });
impl_swizzle!(xyy() => Vector4 => Vector3 { 0, 1, 1 });
impl_swizzle!(xyz() => Vector4 => Vector3 { 0, 1, 2 });
impl_swizzle!(xyw() => Vector4 => Vector3 { 0, 1, 3 });
impl_swizzle!(xzx() => Vector4 => Vector3 { 0, 2, 0 });
impl_swizzle!(xzy() => Vector4 => Vector3 { 0, 2, 1 });
impl_swizzle!(xzz() => Vector4 => Vector3 { 0, 2, 2 });
impl_swizzle!(xzw() => Vector4 => Vector3 { 0, 2, 3 });
impl_swizzle!(xwx() => Vector4 => Vector3 { 0, 3, 0 });
impl_swizzle!(xwy() => Vector4 => Vector3 { 0, 3, 1 });
impl_swizzle!(xwz() => Vector4 => Vector3 { 0, 3, 2 });
impl_swizzle!(xww() => Vector4 => Vector3 { 0, 3, 3 });
impl_swizzle!(yxx() => Vector4 => Vector3 { 1, 0, 0 });
impl_swizzle!(yxy() => Vector4 => Vector3 { 1, 0, 1 });
impl_swizzle!(yxz() => Vector4 => Vector3 { 1, 0, 2 });
impl_swizzle!(yxw() => Vector4 => Vector3 { 1, 0, 3 });
impl_swizzle!(yyx() => Vector4 => Vector3 { 1, 1, 0 });
impl_swizzle!(yyy() => Vector4 => Vector3 { 1, 1, 1 });
impl_swizzle!(yyz() => Vector4 => Vector3 { 1, 1, 2 });
impl_swizzle!(yyw() => Vector4 => Vector3 { 1, 1, 3 });
impl_swizzle!(yzx() => Vector4 => Vector3 { 1, 2, 0 });
impl_swizzle!(yzy() => Vector4 => Vector3 { 1, 2, 1 });
impl_swizzle!(yzz() => Vector4 => Vector3 { 1, 2, 2 });
impl_swizzle!(yzw() => Vector4 => Vector3 { 1, 2, 3 });
impl_swizzle!(ywx() => Vector4 => Vector3 { 1, 3, 0 });
impl_swizzle!(ywy() => Vector4 => Vector3 { 1, 3, 1 });
impl_swizzle!(ywz() => Vector4 => Vector3 { 1, 3, 2 });
impl_swizzle!(yww() => Vector4 => Vector3 { 1, 3, 3 });
impl_swizzle!(zxx() => Vector4 => Vector3 { 2, 0, 0 });
impl_swizzle!(zxy() => Vector4 => Vector3 { 2, 0, 1 });
impl_swizzle!(zxz() => Vector4 => Vector3 { 2, 0, 2 });
impl_swizzle!(zxw() => Vector4 => Vector3 { 2, 0, 3 });
impl_swizzle!(zyx() => Vector4 => Vector3 { 2, 1, 0 });
impl_swizzle!(zyy() => Vector4 => Vector3 { 2, 1, 1 });
impl_swizzle!(zyz() => Vector4 => Vector3 { 2, 1, 2 });
impl_swizzle!(zyw() => Vector4 => Vector3 { 2, 1, 3 });
impl_swizzle!(zzx() => Vector4 => Vector3 { 2, 2, 0 });
impl_swizzle!(zzy() => Vector4 => Vector3 { 2, 2, 1 });
impl_swizzle!(zzz() => Vector4 => Vector3 { 2, 2, 2 });
impl_swizzle!(zzw() => Vector4 => Vector3 { 2, 2, 3 });
impl_swizzle!(zwx() => Vector4 => Vector3 { 2, 3, 0 });
impl_swizzle!(zwy() => Vector4 => Vector3 { 2, 3, 1 });
impl_swizzle!(zwz() => Vector4 => Vector3 { 2, 3, 2 });
impl_swizzle!(zww() => Vector4 => Vector3 { 2, 3, 3 });
impl_swizzle!(wxx() => Vector4 => Vector3 { 3, 0, 0 });
impl_swizzle!(wxy() => Vector4 => Vector3 { 3, 0, 1 });
impl_swizzle!(wxz() => Vector4 => Vector3 { 3, 0, 2 });
impl_swizzle!(wxw() => Vector4 => Vector3 { 3, 0, 3 });
impl_swizzle!(wyx() => Vector4 => Vector3 { 3, 1, 0 });
impl_swizzle!(wyy() => Vector4 => Vector3 { 3, 1, 1 });
impl_swizzle!(wyz() => Vector4 => Vector3 { 3, 1, 2 });
impl_swizzle!(wyw() => Vector4 => Vector3 { 3, 1, 3 });
impl_swizzle!(wzx() => Vector4 => Vector3 { 3, 2, 0 });
impl_swizzle!(wzy() => Vector4 => Vector3 { 3, 2, 1 });
impl_swizzle!(wzz() => Vector4 => Vector3 { 3, 2, 2 });
impl_swizzle!(wzw() => Vector4 => Vector3 { 3, 2, 3 });
impl_swizzle!(wwx() => Vector4 => Vector3 { 3, 3, 0 });
impl_swizzle!(wwy() => Vector4 => Vector3 { 3, 3, 1 });
impl_swizzle!(wwz() => Vector4 => Vector3 { 3, 3, 2 });
impl_swizzle!(www() => Vector4 => Vector3 { 3, 3, 3 });


impl_swizzle!(xxxx() => Vector4 => Vector4 { 0, 0, 0, 0 });
impl_swizzle!(xxxy() => Vector4 => Vector4 { 0, 0, 0, 1 });
impl_swizzle!(xxxz() => Vector4 => Vector4 { 0, 0, 0, 2 });
impl_swizzle!(xxxw() => Vector4 => Vector4 { 0, 0, 0, 3 });
impl_swizzle!(xxyx() => Vector4 => Vector4 { 0, 0, 1, 0 });
impl_swizzle!(xxyy() => Vector4 => Vector4 { 0, 0, 1, 1 });
impl_swizzle!(xxyz() => Vector4 => Vector4 { 0, 0, 1, 2 });
impl_swizzle!(xxyw() => Vector4 => Vector4 { 0, 0, 1, 3 });
impl_swizzle!(xxzx() => Vector4 => Vector4 { 0, 0, 2, 0 });
impl_swizzle!(xxzy() => Vector4 => Vector4 { 0, 0, 2, 1 });
impl_swizzle!(xxzz() => Vector4 => Vector4 { 0, 0, 2, 2 });
impl_swizzle!(xxzw() => Vector4 => Vector4 { 0, 0, 2, 3 });
impl_swizzle!(xxwx() => Vector4 => Vector4 { 0, 0, 3, 0 });
impl_swizzle!(xxwy() => Vector4 => Vector4 { 0, 0, 3, 1 });
impl_swizzle!(xxwz() => Vector4 => Vector4 { 0, 0, 3, 2 });
impl_swizzle!(xxww() => Vector4 => Vector4 { 0, 0, 3, 3 });
impl_swizzle!(xyxx() => Vector4 => Vector4 { 0, 1, 0, 0 });
impl_swizzle!(xyxy() => Vector4 => Vector4 { 0, 1, 0, 1 });
impl_swizzle!(xyxz() => Vector4 => Vector4 { 0, 1, 0, 2 });
impl_swizzle!(xyxw() => Vector4 => Vector4 { 0, 1, 0, 3 });
impl_swizzle!(xyyx() => Vector4 => Vector4 { 0, 1, 1, 0 });
impl_swizzle!(xyyy() => Vector4 => Vector4 { 0, 1, 1, 1 });
impl_swizzle!(xyyz() => Vector4 => Vector4 { 0, 1, 1, 2 });
impl_swizzle!(xyyw() => Vector4 => Vector4 { 0, 1, 1, 3 });
impl_swizzle!(xyzx() => Vector4 => Vector4 { 0, 1, 2, 0 });
impl_swizzle!(xyzy() => Vector4 => Vector4 { 0, 1, 2, 1 });
impl_swizzle!(xyzz() => Vector4 => Vector4 { 0, 1, 2, 2 });
impl_swizzle!(xyzw() => Vector4 => Vector4 { 0, 1, 2, 3 });
impl_swizzle!(xywx() => Vector4 => Vector4 { 0, 1, 3, 0 });
impl_swizzle!(xywy() => Vector4 => Vector4 { 0, 1, 3, 1 });
impl_swizzle!(xywz() => Vector4 => Vector4 { 0, 1, 3, 2 });
impl_swizzle!(xyww() => Vector4 => Vector4 { 0, 1, 3, 3 });
impl_swizzle!(xzxx() => Vector4 => Vector4 { 0, 2, 0, 0 });
impl_swizzle!(xzxy() => Vector4 => Vector4 { 0, 2, 0, 1 });
impl_swizzle!(xzxz() => Vector4 => Vector4 { 0, 2, 0, 2 });
impl_swizzle!(xzxw() => Vector4 => Vector4 { 0, 2, 0, 3 });
impl_swizzle!(xzyx() => Vector4 => Vector4 { 0, 2, 1, 0 });
impl_swizzle!(xzyy() => Vector4 => Vector4 { 0, 2, 1, 1 });
impl_swizzle!(xzyz() => Vector4 => Vector4 { 0, 2, 1, 2 });
impl_swizzle!(xzyw() => Vector4 => Vector4 { 0, 2, 1, 3 });
impl_swizzle!(xzzx() => Vector4 => Vector4 { 0, 2, 2, 0 });
impl_swizzle!(xzzy() => Vector4 => Vector4 { 0, 2, 2, 1 });
impl_swizzle!(xzzz() => Vector4 => Vector4 { 0, 2, 2, 2 });
impl_swizzle!(xzzw() => Vector4 => Vector4 { 0, 2, 2, 3 });
impl_swizzle!(xzwx() => Vector4 => Vector4 { 0, 2, 3, 0 });
impl_swizzle!(xzwy() => Vector4 => Vector4 { 0, 2, 3, 1 });
impl_swizzle!(xzwz() => Vector4 => Vector4 { 0, 2, 3, 2 });
impl_swizzle!(xzww() => Vector4 => Vector4 { 0, 2, 3, 3 });
impl_swizzle!(xwxx() => Vector4 => Vector4 { 0, 3, 0, 0 });
impl_swizzle!(xwxy() => Vector4 => Vector4 { 0, 3, 0, 1 });
impl_swizzle!(xwxz() => Vector4 => Vector4 { 0, 3, 0, 2 });
impl_swizzle!(xwxw() => Vector4 => Vector4 { 0, 3, 0, 3 });
impl_swizzle!(xwyx() => Vector4 => Vector4 { 0, 3, 1, 0 });
impl_swizzle!(xwyy() => Vector4 => Vector4 { 0, 3, 1, 1 });
impl_swizzle!(xwyz() => Vector4 => Vector4 { 0, 3, 1, 2 });
impl_swizzle!(xwyw() => Vector4 => Vector4 { 0, 3, 1, 3 });
impl_swizzle!(xwzx() => Vector4 => Vector4 { 0, 3, 2, 0 });
impl_swizzle!(xwzy() => Vector4 => Vector4 { 0, 3, 2, 1 });
impl_swizzle!(xwzz() => Vector4 => Vector4 { 0, 3, 2, 2 });
impl_swizzle!(xwzw() => Vector4 => Vector4 { 0, 3, 2, 3 });
impl_swizzle!(xwwx() => Vector4 => Vector4 { 0, 3, 3, 0 });
impl_swizzle!(xwwy() => Vector4 => Vector4 { 0, 3, 3, 1 });
impl_swizzle!(xwwz() => Vector4 => Vector4 { 0, 3, 3, 2 });
impl_swizzle!(xwww() => Vector4 => Vector4 { 0, 3, 3, 3 });
impl_swizzle!(yxxx() => Vector4 => Vector4 { 1, 0, 0, 0 });
impl_swizzle!(yxxy() => Vector4 => Vector4 { 1, 0, 0, 1 });
impl_swizzle!(yxxz() => Vector4 => Vector4 { 1, 0, 0, 2 });
impl_swizzle!(yxxw() => Vector4 => Vector4 { 1, 0, 0, 3 });
impl_swizzle!(yxyx() => Vector4 => Vector4 { 1, 0, 1, 0 });
impl_swizzle!(yxyy() => Vector4 => Vector4 { 1, 0, 1, 1 });
impl_swizzle!(yxyz() => Vector4 => Vector4 { 1, 0, 1, 2 });
impl_swizzle!(yxyw() => Vector4 => Vector4 { 1, 0, 1, 3 });
impl_swizzle!(yxzx() => Vector4 => Vector4 { 1, 0, 2, 0 });
impl_swizzle!(yxzy() => Vector4 => Vector4 { 1, 0, 2, 1 });
impl_swizzle!(yxzz() => Vector4 => Vector4 { 1, 0, 2, 2 });
impl_swizzle!(yxzw() => Vector4 => Vector4 { 1, 0, 2, 3 });
impl_swizzle!(yxwx() => Vector4 => Vector4 { 1, 0, 3, 0 });
impl_swizzle!(yxwy() => Vector4 => Vector4 { 1, 0, 3, 1 });
impl_swizzle!(yxwz() => Vector4 => Vector4 { 1, 0, 3, 2 });
impl_swizzle!(yxww() => Vector4 => Vector4 { 1, 0, 3, 3 });
impl_swizzle!(yyxx() => Vector4 => Vector4 { 1, 1, 0, 0 });
impl_swizzle!(yyxy() => Vector4 => Vector4 { 1, 1, 0, 1 });
impl_swizzle!(yyxz() => Vector4 => Vector4 { 1, 1, 0, 2 });
impl_swizzle!(yyxw() => Vector4 => Vector4 { 1, 1, 0, 3 });
impl_swizzle!(yyyx() => Vector4 => Vector4 { 1, 1, 1, 0 });
impl_swizzle!(yyyy() => Vector4 => Vector4 { 1, 1, 1, 1 });
impl_swizzle!(yyyz() => Vector4 => Vector4 { 1, 1, 1, 2 });
impl_swizzle!(yyyw() => Vector4 => Vector4 { 1, 1, 1, 3 });
impl_swizzle!(yyzx() => Vector4 => Vector4 { 1, 1, 2, 0 });
impl_swizzle!(yyzy() => Vector4 => Vector4 { 1, 1, 2, 1 });
impl_swizzle!(yyzz() => Vector4 => Vector4 { 1, 1, 2, 2 });
impl_swizzle!(yyzw() => Vector4 => Vector4 { 1, 1, 2, 3 });
impl_swizzle!(yywx() => Vector4 => Vector4 { 1, 1, 3, 0 });
impl_swizzle!(yywy() => Vector4 => Vector4 { 1, 1, 3, 1 });
impl_swizzle!(yywz() => Vector4 => Vector4 { 1, 1, 3, 2 });
impl_swizzle!(yyww() => Vector4 => Vector4 { 1, 1, 3, 3 });
impl_swizzle!(yzxx() => Vector4 => Vector4 { 1, 2, 0, 0 });
impl_swizzle!(yzxy() => Vector4 => Vector4 { 1, 2, 0, 1 });
impl_swizzle!(yzxz() => Vector4 => Vector4 { 1, 2, 0, 2 });
impl_swizzle!(yzxw() => Vector4 => Vector4 { 1, 2, 0, 3 });
impl_swizzle!(yzyx() => Vector4 => Vector4 { 1, 2, 1, 0 });
impl_swizzle!(yzyy() => Vector4 => Vector4 { 1, 2, 1, 1 });
impl_swizzle!(yzyz() => Vector4 => Vector4 { 1, 2, 1, 2 });
impl_swizzle!(yzyw() => Vector4 => Vector4 { 1, 2, 1, 3 });
impl_swizzle!(yzzx() => Vector4 => Vector4 { 1, 2, 2, 0 });
impl_swizzle!(yzzy() => Vector4 => Vector4 { 1, 2, 2, 1 });
impl_swizzle!(yzzz() => Vector4 => Vector4 { 1, 2, 2, 2 });
impl_swizzle!(yzzw() => Vector4 => Vector4 { 1, 2, 2, 3 });
impl_swizzle!(yzwx() => Vector4 => Vector4 { 1, 2, 3, 0 });
impl_swizzle!(yzwy() => Vector4 => Vector4 { 1, 2, 3, 1 });
impl_swizzle!(yzwz() => Vector4 => Vector4 { 1, 2, 3, 2 });
impl_swizzle!(yzww() => Vector4 => Vector4 { 1, 2, 3, 3 });
impl_swizzle!(ywxx() => Vector4 => Vector4 { 1, 3, 0, 0 });
impl_swizzle!(ywxy() => Vector4 => Vector4 { 1, 3, 0, 1 });
impl_swizzle!(ywxz() => Vector4 => Vector4 { 1, 3, 0, 2 });
impl_swizzle!(ywxw() => Vector4 => Vector4 { 1, 3, 0, 3 });
impl_swizzle!(ywyx() => Vector4 => Vector4 { 1, 3, 1, 0 });
impl_swizzle!(ywyy() => Vector4 => Vector4 { 1, 3, 1, 1 });
impl_swizzle!(ywyz() => Vector4 => Vector4 { 1, 3, 1, 2 });
impl_swizzle!(ywyw() => Vector4 => Vector4 { 1, 3, 1, 3 });
impl_swizzle!(ywzx() => Vector4 => Vector4 { 1, 3, 2, 0 });
impl_swizzle!(ywzy() => Vector4 => Vector4 { 1, 3, 2, 1 });
impl_swizzle!(ywzz() => Vector4 => Vector4 { 1, 3, 2, 2 });
impl_swizzle!(ywzw() => Vector4 => Vector4 { 1, 3, 2, 3 });
impl_swizzle!(ywwx() => Vector4 => Vector4 { 1, 3, 3, 0 });
impl_swizzle!(ywwy() => Vector4 => Vector4 { 1, 3, 3, 1 });
impl_swizzle!(ywwz() => Vector4 => Vector4 { 1, 3, 3, 2 });
impl_swizzle!(ywww() => Vector4 => Vector4 { 1, 3, 3, 3 });
impl_swizzle!(zxxx() => Vector4 => Vector4 { 2, 0, 0, 0 });
impl_swizzle!(zxxy() => Vector4 => Vector4 { 2, 0, 0, 1 });
impl_swizzle!(zxxz() => Vector4 => Vector4 { 2, 0, 0, 2 });
impl_swizzle!(zxxw() => Vector4 => Vector4 { 2, 0, 0, 3 });
impl_swizzle!(zxyx() => Vector4 => Vector4 { 2, 0, 1, 0 });
impl_swizzle!(zxyy() => Vector4 => Vector4 { 2, 0, 1, 1 });
impl_swizzle!(zxyz() => Vector4 => Vector4 { 2, 0, 1, 2 });
impl_swizzle!(zxyw() => Vector4 => Vector4 { 2, 0, 1, 3 });
impl_swizzle!(zxzx() => Vector4 => Vector4 { 2, 0, 2, 0 });
impl_swizzle!(zxzy() => Vector4 => Vector4 { 2, 0, 2, 1 });
impl_swizzle!(zxzz() => Vector4 => Vector4 { 2, 0, 2, 2 });
impl_swizzle!(zxzw() => Vector4 => Vector4 { 2, 0, 2, 3 });
impl_swizzle!(zxwx() => Vector4 => Vector4 { 2, 0, 3, 0 });
impl_swizzle!(zxwy() => Vector4 => Vector4 { 2, 0, 3, 1 });
impl_swizzle!(zxwz() => Vector4 => Vector4 { 2, 0, 3, 2 });
impl_swizzle!(zxww() => Vector4 => Vector4 { 2, 0, 3, 3 });
impl_swizzle!(zyxx() => Vector4 => Vector4 { 2, 1, 0, 0 });
impl_swizzle!(zyxy() => Vector4 => Vector4 { 2, 1, 0, 1 });
impl_swizzle!(zyxz() => Vector4 => Vector4 { 2, 1, 0, 2 });
impl_swizzle!(zyxw() => Vector4 => Vector4 { 2, 1, 0, 3 });
impl_swizzle!(zyyx() => Vector4 => Vector4 { 2, 1, 1, 0 });
impl_swizzle!(zyyy() => Vector4 => Vector4 { 2, 1, 1, 1 });
impl_swizzle!(zyyz() => Vector4 => Vector4 { 2, 1, 1, 2 });
impl_swizzle!(zyyw() => Vector4 => Vector4 { 2, 1, 1, 3 });
impl_swizzle!(zyzx() => Vector4 => Vector4 { 2, 1, 2, 0 });
impl_swizzle!(zyzy() => Vector4 => Vector4 { 2, 1, 2, 1 });
impl_swizzle!(zyzz() => Vector4 => Vector4 { 2, 1, 2, 2 });
impl_swizzle!(zyzw() => Vector4 => Vector4 { 2, 1, 2, 3 });
impl_swizzle!(zywx() => Vector4 => Vector4 { 2, 1, 3, 0 });
impl_swizzle!(zywy() => Vector4 => Vector4 { 2, 1, 3, 1 });
impl_swizzle!(zywz() => Vector4 => Vector4 { 2, 1, 3, 2 });
impl_swizzle!(zyww() => Vector4 => Vector4 { 2, 1, 3, 3 });
impl_swizzle!(zzxx() => Vector4 => Vector4 { 2, 2, 0, 0 });
impl_swizzle!(zzxy() => Vector4 => Vector4 { 2, 2, 0, 1 });
impl_swizzle!(zzxz() => Vector4 => Vector4 { 2, 2, 0, 2 });
impl_swizzle!(zzxw() => Vector4 => Vector4 { 2, 2, 0, 3 });
impl_swizzle!(zzyx() => Vector4 => Vector4 { 2, 2, 1, 0 });
impl_swizzle!(zzyy() => Vector4 => Vector4 { 2, 2, 1, 1 });
impl_swizzle!(zzyz() => Vector4 => Vector4 { 2, 2, 1, 2 });
impl_swizzle!(zzyw() => Vector4 => Vector4 { 2, 2, 1, 3 });
impl_swizzle!(zzzx() => Vector4 => Vector4 { 2, 2, 2, 0 });
impl_swizzle!(zzzy() => Vector4 => Vector4 { 2, 2, 2, 1 });
impl_swizzle!(zzzz() => Vector4 => Vector4 { 2, 2, 2, 2 });
impl_swizzle!(zzzw() => Vector4 => Vector4 { 2, 2, 2, 3 });
impl_swizzle!(zzwx() => Vector4 => Vector4 { 2, 2, 3, 0 });
impl_swizzle!(zzwy() => Vector4 => Vector4 { 2, 2, 3, 1 });
impl_swizzle!(zzwz() => Vector4 => Vector4 { 2, 2, 3, 2 });
impl_swizzle!(zzww() => Vector4 => Vector4 { 2, 2, 3, 3 });
impl_swizzle!(zwxx() => Vector4 => Vector4 { 2, 3, 0, 0 });
impl_swizzle!(zwxy() => Vector4 => Vector4 { 2, 3, 0, 1 });
impl_swizzle!(zwxz() => Vector4 => Vector4 { 2, 3, 0, 2 });
impl_swizzle!(zwxw() => Vector4 => Vector4 { 2, 3, 0, 3 });
impl_swizzle!(zwyx() => Vector4 => Vector4 { 2, 3, 1, 0 });
impl_swizzle!(zwyy() => Vector4 => Vector4 { 2, 3, 1, 1 });
impl_swizzle!(zwyz() => Vector4 => Vector4 { 2, 3, 1, 2 });
impl_swizzle!(zwyw() => Vector4 => Vector4 { 2, 3, 1, 3 });
impl_swizzle!(zwzx() => Vector4 => Vector4 { 2, 3, 2, 0 });
impl_swizzle!(zwzy() => Vector4 => Vector4 { 2, 3, 2, 1 });
impl_swizzle!(zwzz() => Vector4 => Vector4 { 2, 3, 2, 2 });
impl_swizzle!(zwzw() => Vector4 => Vector4 { 2, 3, 2, 3 });
impl_swizzle!(zwwx() => Vector4 => Vector4 { 2, 3, 3, 0 });
impl_swizzle!(zwwy() => Vector4 => Vector4 { 2, 3, 3, 1 });
impl_swizzle!(zwwz() => Vector4 => Vector4 { 2, 3, 3, 2 });
impl_swizzle!(zwww() => Vector4 => Vector4 { 2, 3, 3, 3 });
impl_swizzle!(wxxx() => Vector4 => Vector4 { 3, 0, 0, 0 });
impl_swizzle!(wxxy() => Vector4 => Vector4 { 3, 0, 0, 1 });
impl_swizzle!(wxxz() => Vector4 => Vector4 { 3, 0, 0, 2 });
impl_swizzle!(wxxw() => Vector4 => Vector4 { 3, 0, 0, 3 });
impl_swizzle!(wxyx() => Vector4 => Vector4 { 3, 0, 1, 0 });
impl_swizzle!(wxyy() => Vector4 => Vector4 { 3, 0, 1, 1 });
impl_swizzle!(wxyz() => Vector4 => Vector4 { 3, 0, 1, 2 });
impl_swizzle!(wxyw() => Vector4 => Vector4 { 3, 0, 1, 3 });
impl_swizzle!(wxzx() => Vector4 => Vector4 { 3, 0, 2, 0 });
impl_swizzle!(wxzy() => Vector4 => Vector4 { 3, 0, 2, 1 });
impl_swizzle!(wxzz() => Vector4 => Vector4 { 3, 0, 2, 2 });
impl_swizzle!(wxzw() => Vector4 => Vector4 { 3, 0, 2, 3 });
impl_swizzle!(wxwx() => Vector4 => Vector4 { 3, 0, 3, 0 });
impl_swizzle!(wxwy() => Vector4 => Vector4 { 3, 0, 3, 1 });
impl_swizzle!(wxwz() => Vector4 => Vector4 { 3, 0, 3, 2 });
impl_swizzle!(wxww() => Vector4 => Vector4 { 3, 0, 3, 3 });
impl_swizzle!(wyxx() => Vector4 => Vector4 { 3, 1, 0, 0 });
impl_swizzle!(wyxy() => Vector4 => Vector4 { 3, 1, 0, 1 });
impl_swizzle!(wyxz() => Vector4 => Vector4 { 3, 1, 0, 2 });
impl_swizzle!(wyxw() => Vector4 => Vector4 { 3, 1, 0, 3 });
impl_swizzle!(wyyx() => Vector4 => Vector4 { 3, 1, 1, 0 });
impl_swizzle!(wyyy() => Vector4 => Vector4 { 3, 1, 1, 1 });
impl_swizzle!(wyyz() => Vector4 => Vector4 { 3, 1, 1, 2 });
impl_swizzle!(wyyw() => Vector4 => Vector4 { 3, 1, 1, 3 });
impl_swizzle!(wyzx() => Vector4 => Vector4 { 3, 1, 2, 0 });
impl_swizzle!(wyzy() => Vector4 => Vector4 { 3, 1, 2, 1 });
impl_swizzle!(wyzz() => Vector4 => Vector4 { 3, 1, 2, 2 });
impl_swizzle!(wyzw() => Vector4 => Vector4 { 3, 1, 2, 3 });
impl_swizzle!(wywx() => Vector4 => Vector4 { 3, 1, 3, 0 });
impl_swizzle!(wywy() => Vector4 => Vector4 { 3, 1, 3, 1 });
impl_swizzle!(wywz() => Vector4 => Vector4 { 3, 1, 3, 2 });
impl_swizzle!(wyww() => Vector4 => Vector4 { 3, 1, 3, 3 });
impl_swizzle!(wzxx() => Vector4 => Vector4 { 3, 2, 0, 0 });
impl_swizzle!(wzxy() => Vector4 => Vector4 { 3, 2, 0, 1 });
impl_swizzle!(wzxz() => Vector4 => Vector4 { 3, 2, 0, 2 });
impl_swizzle!(wzxw() => Vector4 => Vector4 { 3, 2, 0, 3 });
impl_swizzle!(wzyx() => Vector4 => Vector4 { 3, 2, 1, 0 });
impl_swizzle!(wzyy() => Vector4 => Vector4 { 3, 2, 1, 1 });
impl_swizzle!(wzyz() => Vector4 => Vector4 { 3, 2, 1, 2 });
impl_swizzle!(wzyw() => Vector4 => Vector4 { 3, 2, 1, 3 });
impl_swizzle!(wzzx() => Vector4 => Vector4 { 3, 2, 2, 0 });
impl_swizzle!(wzzy() => Vector4 => Vector4 { 3, 2, 2, 1 });
impl_swizzle!(wzzz() => Vector4 => Vector4 { 3, 2, 2, 2 });
impl_swizzle!(wzzw() => Vector4 => Vector4 { 3, 2, 2, 3 });
impl_swizzle!(wzwx() => Vector4 => Vector4 { 3, 2, 3, 0 });
impl_swizzle!(wzwy() => Vector4 => Vector4 { 3, 2, 3, 1 });
impl_swizzle!(wzwz() => Vector4 => Vector4 { 3, 2, 3, 2 });
impl_swizzle!(wzww() => Vector4 => Vector4 { 3, 2, 3, 3 });
impl_swizzle!(wwxx() => Vector4 => Vector4 { 3, 3, 0, 0 });
impl_swizzle!(wwxy() => Vector4 => Vector4 { 3, 3, 0, 1 });
impl_swizzle!(wwxz() => Vector4 => Vector4 { 3, 3, 0, 2 });
impl_swizzle!(wwxw() => Vector4 => Vector4 { 3, 3, 0, 3 });
impl_swizzle!(wwyx() => Vector4 => Vector4 { 3, 3, 1, 0 });
impl_swizzle!(wwyy() => Vector4 => Vector4 { 3, 3, 1, 1 });
impl_swizzle!(wwyz() => Vector4 => Vector4 { 3, 3, 1, 2 });
impl_swizzle!(wwyw() => Vector4 => Vector4 { 3, 3, 1, 3 });
impl_swizzle!(wwzx() => Vector4 => Vector4 { 3, 3, 2, 0 });
impl_swizzle!(wwzy() => Vector4 => Vector4 { 3, 3, 2, 1 });
impl_swizzle!(wwzz() => Vector4 => Vector4 { 3, 3, 2, 2 });
impl_swizzle!(wwzw() => Vector4 => Vector4 { 3, 3, 2, 3 });
impl_swizzle!(wwwx() => Vector4 => Vector4 { 3, 3, 3, 0 });
impl_swizzle!(wwwy() => Vector4 => Vector4 { 3, 3, 3, 1 });
impl_swizzle!(wwwz() => Vector4 => Vector4 { 3, 3, 3, 2 });
impl_swizzle!(wwww() => Vector4 => Vector4 { 3, 3, 3, 3 });

