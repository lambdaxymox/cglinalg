use crate::scalar::{
    ScalarFloat,
};
use crate::traits::{
    Magnitude,
};

use num_traits::{
    Float,
};

use core::fmt;


/// A type that represents unit normalized values. 
///
/// This type enforces the requirement that values have a unit magnitude. This 
/// is useful when one needs to know the direction a vector is pointing for 
/// operations like constructing rotations. The unit type statically enforces the 
/// requirement that an input argument be normalized. This reduces the chance of
/// errors from passing an unnormalized vector or a zero vector into a calculation
/// involving unit vectors.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Unit<T> {
    /// The underlying normalized value.
    value: T,
}

impl<T> Unit<T> {
    /// Unwraps the underlying value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T> AsRef<T> for Unit<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T: fmt::Display> fmt::Display for Unit<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.value.fmt(f)
    }
}

impl<T> Unit<T> where T: Magnitude {
    /// Construct a new unit value, normalizing the input value.
    #[inline]
    pub fn new(value: T) -> Self {
        Self::new_with_magnitude(value).0
    }

    /// Construct a new normalized unit value along with its unnormalized magnitude.
    ///
    /// ### Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Magnitude,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// let vector: Vector3<f64> = Vector3::new(0.0, 2.0, 0.0);
    /// let (wrapped, norm) = Unit::new_with_magnitude(vector);
    /// let unit_vector: &Vector3<f64> = wrapped.as_ref();
    /// 
    /// assert_eq!(norm, 2.0);
    /// assert_eq!(unit_vector.magnitude_squared(), 1.0, "unit_vector = {}", unit_vector);
    /// assert_eq!(unit_vector.magnitude(), 1.0, "unit_vector = {}", unit_vector);
    /// ```
    #[inline]
    pub fn new_with_magnitude(value: T) -> (Self, T::Output) {
        let magnitude = value.magnitude();
        let normalized_value = value.normalize();
        let unit = Unit {
            value: normalized_value,
        };
        
        (unit, magnitude)
    }

    /// Construct a new normalized unit value along with its unnormalized magnitude, 
    /// provided that its magnitude is larger than `threshold`. 
    ///
    /// The argument `threshold` argument exists to check for vectors that may be
    /// very close to zero length.
    ///
    /// ### Example
    ///
    /// Here is an example where the function returns `None` because the vector 
    /// magnitude is too small.
    /// ```
    /// # use cglinalg::{
    /// #     Magnitude,
    /// #     Unit,
    /// #     Vector3,
    /// # };
    /// let vector: Vector3<f64> = Vector3::new(0.0, 1e-20, 0.0);
    /// let threshold = 1e-10;
    /// let result = Unit::try_new_with_magnitude(vector, threshold);
    /// 
    /// assert!(result.is_none());
    /// ```
    #[inline]
    pub fn try_new_with_magnitude(value: T, threshold: T::Output) -> Option<(Self, T::Output)>
        where T::Output: ScalarFloat, 
    {
        let magnitude_squared = value.magnitude_squared();

        if magnitude_squared > threshold * threshold {
            let magnitude = magnitude_squared.sqrt();
            let normalized_value = value.normalize();
            let unit = Unit {
                value: normalized_value,
            };

            Some((unit, magnitude))
        } else {
            None
        }
    }

    /// Construct a new normalized unit value, provided that its magnitude is 
    /// larger than `threshold`. 
    ///
    /// The argument `threshold` argument exists to check for vectors that may be
    /// very close to zero length.
    #[inline]
    pub fn try_new(value: T, threshold: T::Output) -> Option<Self>
        where T::Output: ScalarFloat,
    {
        Self::try_new_with_magnitude(value, threshold).map(|(unit, _)| unit)
    }
}

