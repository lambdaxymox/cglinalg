use crate::scalar::{
    Scalar,
    ScalarFloat,   
};
use num_traits::{
    Float,
};

use core::ops;


/// A type implementing this trait has the structure of an array
/// of its elements in its underlying storage. 
/// 
/// We can manipulate underlying storage directly for operations such as 
/// passing geometric data across an API boundary to the GPU, or other 
/// external hardware.
pub trait Array { 
    /// The elements of an array.
    type Element: Copy;

    /// The length of the the underlying array.
    fn len() -> usize;
    
    /// The shape of the underlying array.
    fn shape() -> (usize, usize);

    /// Generate a pointer to the underlying array for passing a
    /// matrix or vector to the graphics hardware.
    fn as_ptr(&self) -> *const Self::Element; 

    /// Generate a mutable pointer to the underlying array for passing a
    /// matrix or vector to the graphics hardware.
    fn as_mut_ptr(&mut self) -> *mut Self::Element; 

    /// Get a slice of the underlying elements of the data type.
    fn as_slice(&self) -> &[Self::Element];
}

/// A type with this trait has a notion of comparing the distance (metric) 
/// between two elements of that type. For example, one can use this trait 
/// to compute the Euclidean distance between two vectors. 
pub trait Metric<V: Sized>: Sized {
    type Output: ScalarFloat;

    /// Compute the squared Eucliean distance between two vectors.
    fn distance_squared(self, other: V) -> Self::Output;

    /// Compute the Euclidean distance between two vectors.
    fn distance(self, other: V) -> Self::Output {
        Self::Output::sqrt(Self::distance_squared(self, other))
    }
}

/// This trait enables one to assign lengths to vectors.
pub trait Magnitude where Self: Sized {
    type Output: Scalar;

    /// Compute the norm (length) of a vector.
    fn magnitude(&self) -> Self::Output;

    /// Compute the squared length of a vector.
    fn magnitude_squared(&self) -> Self::Output;

    /// Convert a vector into a unit vector.
    fn normalize(&self) -> Self;

    /// Normalize a vector with a specified magnitude.
    fn normalize_to(&self, magnitude: Self::Output) -> Self;

    /// Attempt to normalize a vector, but give up if the norm
    /// is too small.
    fn try_normalize(&self, threshold: Self::Output) -> Option<Self>;
}

/// A data type implementing the `Matrix` trait has the structure of a matrix 
/// in column major order. 
///
/// If a type represents a matrix, we can perform operations such as swapping 
/// rows, swapping columns, getting a row of  the the matrix, or swapping 
/// elements.
pub trait Matrix {
    /// The type of the underlying scalars of the matrix.
    type Element: Scalar;

    /// The row vector of a matrix.
    type Row: Array<Element = Self::Element>;

    /// The column vector of a matrix.
    type Column: Array<Element = Self::Element>;

    /// The type signature of the transpose of the matrix.
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
/*
/// Define operations on square matrices. 
///
/// A matrix is said to be square if the number of rows and 
/// the number of columns are are the same.
pub trait SquareMatrix where
    Self: Sized,
    Self: Matrix<
        Column = <Self as SquareMatrix>::ColumnRow,
        Row = <Self as SquareMatrix>::ColumnRow,
        Transpose = Self,
    >,
    Self: ops::Mul<<Self as SquareMatrix>::ColumnRow, Output = <Self as SquareMatrix>::ColumnRow>,
    Self: ops::Mul<Self, Output = Self>,
{
    /// The type of the columns are rows of the square matrix.
    type ColumnRow: Array<Element = Self::Element>;

    /// Construct a new diagonal matrix from a given value where
    /// each element along the diagonal is equal to `value`.
    fn from_diagonal_value(value: Self::Element) -> Self;

    /// Construct a new diagonal matrix from a vector of values
    /// representing the elements along the diagonal.
    fn from_diagonal(value: Self::ColumnRow) -> Self;

    /// Get the diagonal part of a square matrix.
    fn diagonal(&self) -> Self::ColumnRow;

    /// Compute the trace of a square matrix.
    fn trace(&self) -> Self::Element;

    /// Determine whether a square matrix is a diagonal matrix. 
    ///
    /// A square matrix is a diagonal matrix if every off-diagonal 
    /// element is zero.
    fn is_diagonal(&self) -> bool;

    /// Determine whether a matrix is symmetric. 
    ///
    /// A matrix is symmetric when element `(i, j)` is equal to element `(j, i)` 
    /// for each row `i` and column `j`. Otherwise, it is not a symmetric matrix. 
    /// Note that every diagonal matrix is a symmetric matrix.
    fn is_symmetric(&self) -> bool;
}

*/