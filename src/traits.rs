use std::ops;


pub trait Array {
    type Element: Copy;

    ///
    /// The length of the the underlying array.
    ///
    fn len() -> usize;

    /// 
    /// Generate a pointer to the underlying array for passing a
    /// matrix or vector to the graphics hardware.
    ///
    fn as_ptr(&self) -> *const Self::Element; 

    /// 
    /// Generate a mutable pointer to the underlying array for passing a
    /// matrix or vector to the graphics hardware.
    ///
    fn as_mut_ptr(&mut self) -> *mut Self::Element; 
}

pub trait VectorSpace: Copy + Clone where
    Self: ops::Add<Self, Output = Self>,
    Self: ops::Sub<Self, Output = Self>,
    Self: ops::Mul<f32, Output = Self>,
    Self: ops::Div<f32, Output = Self>,
    Self: ops::Rem<f32, Output = Self> 
{

}

pub trait MetricSpace: Sized {
    ///
    /// Compute the squared distance between two vectors.
    ///
    fn distance2(self, to: Self) -> f32;

    ///
    /// Compute the Euclidean distance between two vectors.
    ///
    fn distance(self, to: Self) -> f32 {
        f32::sqrt(self.distance2(to))
    }
}

pub trait DotProduct where Self: MetricSpace + VectorSpace {
    ///
    /// Compute the dot product of two vectors.
    ///
    fn dot(self, other: Self) -> f32;

    ///
    /// Compute the norm (length) of a vector.
    ///
    fn norm(self) -> f32 {
        f32::sqrt(self.dot(self))
    }

    ///
    /// Compute the squared norm (length) of a vector.
    ///
    fn norm2(self) -> f32 {
        self.dot(self)
    }

    ///
    /// Convert an arbitrary vector into a unit vector.
    ///
    fn normalize(self) -> Self {
        self / self.norm()
    }
}
