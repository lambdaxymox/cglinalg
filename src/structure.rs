use base::{
    Scalar,
    ScalarFloat,   
};
use std::ops;


/// A type implementing the Array trait has the structure of an array
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

pub trait Zero where Self: Sized + ops::Add<Self, Output = Self> {
    /// Create a zero element.
    fn zero() -> Self;

    /// Test whether an element is equal to the zero element.
    fn is_zero(&self) -> bool;
}

pub trait One where Self: Sized + ops::Mul<Self, Output = Self> {
    /// Create a multiplicative unit element.
    fn one() -> Self;

    /// Determine whether an element is equal to the multiplicative unit element.
    #[inline]
    fn is_one(&self) -> bool where Self: PartialEq<Self> {
        *self == Self::one()
    }
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

pub trait Metric<V: Sized>: Sized {
    type Output: ScalarFloat;

    /// Compute the squared distance between two vectors.
    fn distance_squared(self, other: V) -> Self::Output;

    /// Compute the Euclidean distance between two vectors.
    fn distance(self, other: V) -> Self::Output {
        use num_traits::Float;
        Self::Output::sqrt(Self::distance_squared(self, other))
    }
}

pub trait DotProduct<V: Copy + Clone> where Self: Copy + Clone {
    type Output: Scalar;

    /// Compute the inner product (dot product) of two vectors.
    fn dot(self, other: V) -> Self::Output;
}


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

pub trait Lerp<V: Copy + Clone> {
    type Scalar: Scalar;
    type Output;

    fn lerp(self, other: V, amount: Self::Scalar) -> Self::Output;
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
