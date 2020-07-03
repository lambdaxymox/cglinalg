use base::{
    Scalar,
    ScalarFloat,   
};
use num_traits::Float;
use std::ops;


/// A type implementing this trait has the structure of an array
/// of its elements in its underlying storage. In this way we can manipulate
/// underlying storage directly for operations such as passing geometric data 
/// across an API boundary to the GPU, or other external hardware.
pub trait Storage {
    /// The elements of an array.
    type Element: Copy;

    /// The length of the the underlying array.
    fn len() -> usize;
    
    /// The shape of the underlying storage.
    fn shape() -> (usize, usize);

    /// Construct an array whose entries are all an input value.
    fn from_value(value: Self::Element) -> Self;

    /// Generate a pointer to the underlying array for passing a
    /// matrix or vector to the graphics hardware.
    fn as_ptr(&self) -> *const Self::Element; 

    /// Generate a mutable pointer to the underlying array for passing a
    /// matrix or vector to the graphics hardware.
    fn as_mut_ptr(&mut self) -> *mut Self::Element; 

    /// Get a slice of the underlying elements of the data type.
    fn as_slice(&self) -> &[Self::Element];
}

/// This trait indicates that a type has an arithmetical zero element.
pub trait Zero where Self: Sized + ops::Add<Self, Output = Self> {
    /// Create a zero element.
    fn zero() -> Self;

    /// Test whether an element is equal to the zero element.
    fn is_zero(&self) -> bool;
}

/// This trait indicates that a type has a multiplicative unit element.
pub trait One where Self: Sized + ops::Mul<Self, Output = Self> {
    /// Create a multiplicative unit element.
    fn one() -> Self;

    /// Determine whether an element is equal to the multiplicative unit element.
    #[inline]
    fn is_one(&self) -> bool where Self: PartialEq<Self> {
        *self == Self::one()
    }
}

/// A trait that defines features for specifying when a vector is finite. A vector is finite when
/// all of its elements are finite. This is useful for vector and matrix types working with fixed 
/// precision floating point values.
pub trait Finite {
    /// Returns `true` if the elements of this vector are all finite. Otherwise, it returns `false`. 
    ///
    /// For example, when the vector elements are `f64`, the vector is finite when the elements are
    /// neither `NaN` nor infinite.
    fn is_finite(self) -> bool;

    /// Returns `true` if any of the elements of this vector are not finite. 
    /// Otherwise, it returns `false`.
    fn is_not_finite(self) -> bool;
}

pub trait VectorSpace where
    Self: Copy + Clone,
    Self: Zero,
    Self: ops::Add<Self, Output = Self>, 
    Self: ops::Sub<Self, Output = Self>,
    Self: ops::Mul<<Self as VectorSpace>::Scalar, Output = Self>,
    Self: ops::Div<<Self as VectorSpace>::Scalar, Output = Self>,
    Self: ops::Rem<<Self as VectorSpace>::Scalar, Output = Self>
{
    type Scalar: Scalar;
}

/// A type with this trait has a notion of comparing the distance (metric) between
/// two elements of that type. For example, one can use this trait to compute the 
/// Euclidean distance between two vectors. 
pub trait Metric<V: Sized>: Sized {
    type Output: ScalarFloat;

    /// Compute the squared distance between two vectors.
    fn distance_squared(self, other: V) -> Self::Output;

    /// Compute the Euclidean distance between two vectors.
    fn distance(self, other: V) -> Self::Output {
        Self::Output::sqrt(Self::distance_squared(self, other))
    }
}

/// This trait enables one to define the dot product of two elements of a vector space.
pub trait DotProduct<V: Copy + Clone> where Self: Copy + Clone {
    type Output: Scalar;

    /// Compute the inner product (dot product) of two vectors.
    fn dot(self, other: V) -> Self::Output;
}

/// This trait enables one to assign lengths to vectors.
pub trait Magnitude {
    type Output: Scalar;

    /// Compute the norm (length) of a vector.
    fn magnitude(&self) -> Self::Output;

    /// Compute the squared length of a vector.
    fn magnitude_squared(&self) -> Self::Output;

    /// Convert a vector into a unit vector.
    fn normalize(&self) -> Self;

    /// Normalize a vector with a specified magnitude.
    fn normalize_to(&self, magnitude: Self::Output) -> Self;
}

/// A vector type with the `Lerp` trait has the ability to linearly interpolate between two elements
/// of that type. 
pub trait Lerp<V: Copy + Clone> {
    type Scalar: Scalar;
    type Output;

    /// Linearly interpolate between two vectors.
    fn lerp(self, other: V, amount: Self::Scalar) -> Self::Output;
}

/// A vector type with this trait can perform normalized linear interpolations between two elements
/// of that type.
pub trait Nlerp<V: Copy + Clone> {
    type Scalar: Scalar;
    type Output;

    /// Compute the normalized linear interpolation between two vectors.
    fn nlerp(self, other: V, amount: Self::Scalar) -> Self::Output;
}

/// A vector or quaternion with this trait can perform spherical linear interpolation
/// between two elements of that type on the unit sphere.
pub trait Slerp<V: Copy + Clone> {
    type Scalar: Scalar;
    type Output;

    /// Spherically linearly interpolate between two unit elements.
    fn slerp(self, other: V, amount: Self::Scalar) -> Self::Output;
} 

pub trait ProjectOn<V> where Self: DotProduct<V>, V: Copy + Clone {
    type Output;

    /// Compute the projection for a vector onto another vector.
    fn project_on(self, onto: V) -> <Self as ProjectOn<V>>::Output;
}

/// A data type implementing the `Matrix` trait has the structure of a matrix 
/// in column major order. If a type represents a matrix, we can perform 
/// operations such as swapping rows, swapping columns, getting a row of 
/// the the matrix, or swapping elements.
pub trait Matrix {
    type Element: Copy;

    /// The row vector of a matrix.
    type Row: Storage<Element = Self::Element>;

    /// The column vector of a matrix.
    type Column: Storage<Element = Self::Element>;

    /// The type signature of the tranpose of the matrix.
    type Transpose: Matrix<Element = Self::Element, Row = Self::Column, Column = Self::Row>;

    /// Get the row of the matrix by value.
    fn row(&self, r: usize) -> Self::Row;
    
    /// Swap two rows of a matrix.
    fn swap_rows(&mut self, row_a: usize, row_b: usize);
    
    /// Swap two columns of a matrix.
    fn swap_columns(&mut self, col_a: usize, col_b: usize);
    
    /// Swap two elements of a matrix.
    fn swap_elements(&mut self, a: (usize, usize), b: (usize, usize));
    
    /// Transpose a matrix.
    fn transpose(&self) -> Self::Transpose;
}

/// A trait that enables trigonometry for typed angles. This enables a rigorous distinction between 
/// different units of angles to prevent trigonometric errors that arise from using incorrect angular units.
/// For example, adding radians to degrees, or passing an angle in degrees to a trigonometric function 
/// when one meant to pass an angle in units of radians.
pub trait Angle where 
    Self: Copy + Clone,
    Self: PartialEq + PartialOrd,
    Self: Zero,
    Self: ops::Neg<Output = Self>,
    Self: ops::Add<Self, Output = Self>,
    Self: ops::Sub<Self, Output = Self>,
    Self: ops::Mul<<Self as Angle>::Scalar, Output = Self>,
    Self: ops::Div<Self, Output = Self>,
    Self: ops::Div<<Self as Angle>::Scalar, Output = Self>,
    Self: ops::Rem<Self, Output = Self>,
    Self: approx::AbsDiffEq<Epsilon = <Self as Angle>::Scalar>,
    Self: approx::RelativeEq<Epsilon = <Self as Angle>::Scalar>,
    Self: approx::UlpsEq<Epsilon = <Self as Angle>::Scalar>,
{
    type Scalar: ScalarFloat;

    /// The value of a full rotation around the unit circle for a typed angle.
    fn full_turn() -> Self;

    /// Compute the sine of a typed angle.
    fn sin(self) -> Self::Scalar;

    /// Compute the cosine of a typed angle.
    fn cos(self) -> Self::Scalar;

    /// Compute the tangent of a typed angle.
    fn tan(self) -> Self::Scalar;

    /// Compute the arcsin of a scalar value, returning a corresponding typed angle.
    fn asin(ratio: Self::Scalar) -> Self;

    /// Compute the arc cosine of a scalar value, returning a corresponding typed angle.
    fn acos(ratio: Self::Scalar) -> Self;

    /// Compute the arc tangent of a scalar value, returning a corresponding typed angle.
    fn atan(ratio: Self::Scalar) -> Self;

    /// Compute the four quadrant arc tangent of two angles, returning a typed angle.
    /// The return values fall into the following value ranges.
    /// ```text
    /// x = 0 and y = 0 -> 0
    /// x >= 0          -> arctan(y / x) in [-pi / 2, pi / 2]
    /// y >= 0          -> (arctan(y / x) + pi) in (pi / 2, pi]
    /// y < 0           -> (arctan(y / x) - pi) in (-pi, -pi / 2)
    /// ```
    fn atan2(a: Self::Scalar, b: Self::Scalar) -> Self;

    /// Simultaneously compute the sin and cosine of an angle. In applications
    /// there are frequently computed together.
    #[inline]
    fn sin_cos(self) -> (Self::Scalar, Self::Scalar) {
        (Self::sin(self), Self::cos(self))
    }

    /// The value of half of a full turn around the unit circle.
    #[inline]
    fn full_turn_div_2() -> Self {
        let denominator: Self::Scalar = num_traits::cast(2).unwrap();
        Self::full_turn() / denominator
    }

    /// The value of a one fourth of a full turn around the unit circle.
    #[inline]
    fn full_turn_div_4() -> Self {
        let denominator: Self::Scalar = num_traits::cast(4).unwrap();
        Self::full_turn() / denominator
    }

    /// The value of one sixth of a full turn around the unit circle.
    #[inline]
    fn full_turn_div_6() -> Self {
        let denominator: Self::Scalar = num_traits::cast(6).unwrap();
        Self::full_turn() / denominator
    }

    /// The value of one eighth of a full turn around the unit circle.
    #[inline]
    fn full_turn_div_8() -> Self {
        let denominator: Self::Scalar = num_traits::cast(8).unwrap();
        Self::full_turn() / denominator
    }

    /// Map an angle to its smallest congruent angle in the range `[0, full_turn)`.
    #[inline]
    fn normalize(self) -> Self {
        let remainder = self % Self::full_turn();
        if remainder < Self::zero() {
            remainder + Self::full_turn()
        } else {
            remainder
        }
    }

    /// Map an angle to its smallest congruent angle in the range `[-full_turn / 2, full_turn / 2)`.
    #[inline]
    fn normalize_signed(self) -> Self {
        let remainder = self.normalize();
        if remainder > Self::full_turn_div_2() {
            remainder - Self::full_turn()
        } else {
            remainder
        }
    }

    /// Compute the angle rotated by half of a turn.
    #[inline]
    fn opposite(self) -> Self {
        Self::normalize(self + Self::full_turn_div_2())
    }

    /// Compute the interior bisector (the angle that is half-way between the angles) 
    /// of `self` and `other`.
    #[inline]
    fn bisect(self, other: Self) -> Self {
        let one_half = num_traits::cast(0.5_f64).unwrap();
        Self::normalize((other - self) * one_half + self)
    }

    /// Compute the cosecant of a typed angle.
    #[inline]
    fn csc(self) -> Self::Scalar {
        Self::sin(self).recip()
    }

    /// Compute the cotangent of a typed angle.
    #[inline]
    fn cot(self) -> Self::Scalar {
        Self::tan(self).recip()
    }

    /// Compute the secant of a typed angle.
    #[inline]
    fn sec(self) -> Self::Scalar {
        Self::cos(self).recip()
    }
}
