use crate::core_numeric::{
    SimdScalarFloat,   
};


/// A type with this trait acts as a vector with a notion of norm (magnitude, length).
///
/// Examples of types that can be made into Euclidean normed spaces include 
/// vectors, quaternions, complex numbers, points, and scalar numbers. In the 
/// scalar case, the Euclidean norm is the absolute value of the scalar.
///
/// # Examples
///
/// ```
/// # use cglinalg_core::{
/// #     Vector4,
/// #     Normed,  
/// # };
/// #
/// // The norm of the vector.
/// let vector = Vector4::new(1_f64, 2_f64, 3_f64, 4_f64);
/// assert_eq!(vector.norm_squared(), 30_f64);
/// assert_eq!(vector.norm(), 30_f64.sqrt());
/// 
/// // Nomalizing a vector.
/// let vector = Vector4::new(1_f64, 1_f64, 1_f64, 1_f64);
/// let expected = vector / 2_f64;
/// let result = vector.normalize();
/// assert_eq!(result, expected);
///
/// // Scale a vector to a specific norm.
/// let norm = 5_f64;
/// let expected = (5_f64 / 2_f64) * vector;
/// let result = vector.scale(norm);
/// assert_eq!(result, expected);
///
/// // Normalize a vector whose norm is close to zero.
/// let vector = Vector4::new(1e-11_f64, 1e-11_f64, 1e-11_f64, 1e-11_f64);
/// let threshold = 1e-10;
/// assert!(vector.try_normalize(threshold).is_none());
///
/// // The Euclidean distance between two vectors.
/// let vector1 = Vector4::new(1_f64, 1_f64, 1_f64, 1_f64);
/// let vector2 = Vector4::new(2_f64, 2_f64, 2_f64, 2_f64);
/// assert_eq!(vector1.distance_squared(&vector2), 4_f64);
/// assert_eq!(vector1.distance(&vector2), 2_f64);
/// ```
pub trait Normed 
where 
    Self: Sized
{
    type Output: SimdScalarFloat;

    /// Compute the squared norm of a vector.
    fn norm_squared(&self) -> Self::Output;

    /// Compute the norm of a vector.
    fn norm(&self) -> Self::Output;

    /// Scale a vector to a specified norm `norm`.
    fn scale(&self, norm: Self::Output) -> Self;

    /// Scale a vector mutably in place to a specified norm `norm`.
    fn scale_mut(&mut self, norm: Self::Output);

    /// Scale a vector to a specified norm `1 / norm`.
    fn unscale(&self, norm: Self::Output) -> Self;

    /// Scale a vector mutably in place to a specified norm `1 / norm`.
    fn unscale_mut(&mut self, norm: Self::Output);

    /// Normalize a vector to a unit vector.
    fn normalize(&self) -> Self;

    /// Normalize a vector to a unit vector mutably in place and return its norm 
    /// prior to normalization.
    fn normalize_mut(&mut self) -> Self::Output;

    /// Attempt to normalize a vector, but give up if the norm
    /// is too small.
    ///
    /// If the norm of the vector is smaller than the threshold
    /// `threshold`, the function returns `None`.
    fn try_normalize(&self, threshold: Self::Output) -> Option<Self>;

    /// Attempt to normalize a vector in place, but give up if the norm is too small.
    /// 
    /// If the norm of the vector is smaller than the threshold 
    /// `threshold`, the function does not mutate `self` and returns `None`.
    fn try_normalize_mut(&mut self, threshold: Self::Output) -> Option<Self::Output>;

    /// Compute the squared Euclidean distance between two vectors.
    fn distance_squared(&self, other: &Self) -> Self::Output;

    /// Compute the Euclidean distance between two vectors.
    fn distance(&self, other: &Self) -> Self::Output;
}

pub trait Norm<V>
where
{
    type Output: SimdScalarFloat;

    fn norm(&self, rhs: &V) -> Self::Output;

    fn metric_distance(&self, lhs: &V, rhs: &V) -> Self::Output;
}

pub type UniformNorm<V> = LinfNorm<V>;
pub type EuclideanNorm<V> = L2Norm<V>;

#[derive(Copy, Clone, Debug)]
pub struct L1Norm<V> {
    _marker: core::marker::PhantomData<V>,
}

impl<V> L1Norm<V> {
    #[inline]
    pub const fn new() -> Self {
        Self { 
            _marker: core::marker::PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct L2Norm<V> {
    _marker: core::marker::PhantomData<V>,
}

impl<V> L2Norm<V> {
    #[inline]
    pub const fn new() -> Self {
        Self { 
            _marker: core::marker::PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LpNorm<V> {
    pub p: u32,
    _marker: core::marker::PhantomData<V>,
}

impl<V> LpNorm<V> {
    #[inline]
    pub const fn new(p: u32) -> Self {
        Self { 
            p,
            _marker: core::marker::PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LinfNorm<V> {
    _marker: core::marker::PhantomData<V>,
}

impl<V> LinfNorm<V> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

