use crate::core_numeric::{
    SimdScalar,
    SimdScalarSigned,
    SimdScalarFloat,
};
use crate::angle::{
    Angle,
    Radians,
};
use crate::norm::{
    Normed,
};
use num_traits::{
    NumCast,
};

use core::fmt;
use core::ops;


/// A complex number in Cartesian form.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Complex<S> {
    /// The real part of a complex number.
    pub re: S,
    /// The imaginary part of a complex number.
    pub im: S,
}

impl<S> Complex<S> {
    /// Construct a new complex number from its real and imaginary components.
    #[inline]
    pub const fn new(re: S, im: S) -> Self {
        Self { re, im }
    }

    /// The shape of the underlying array storing the complex number components.
    ///
    /// The shape is the equivalent number of columns and rows of the 
    /// array as though it represents a matrix. The order of the descriptions 
    /// of the shape of the array is **(rows, columns)**.
    #[inline]
    pub const fn shape(&self) -> (usize, usize) {
        (2, 1)
    }

    /// The length of the the underlying array storing the complex number components.
    #[inline]
    pub const fn len(&self) -> usize {
        2
    }

    /// Tests whether the number of elements in the complex number is zero.
    /// 
    /// Always returns `false` since a complex number has two components.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Get a pointer to the underlying array.
    #[inline]
    pub const fn as_ptr(&self) -> *const S {
        &self.re
    }

    /// Get a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        &mut self.re
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        AsRef::<[S; 2]>::as_ref(self)
    }
}

impl<S> Complex<S> 
where 
    S: NumCast + Copy 
{
    /// Cast a complex number from one type of scalars to another type of scalars.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,   
    /// # };
    /// #
    /// let z: Complex<u32> = Complex::new(1_u32, 2_u32);
    /// let expected: Option<Complex<i32>> = Some(Complex::new(1_i32, 2_i32));
    /// let result = z.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn cast<T: NumCast>(self) -> Option<Complex<T>> {
        let re = match num_traits::cast(self.re) {
            Some(value) => value,
            None => return None,
        };
        let im = match num_traits::cast(self.im) {
            Some(value) => value,
            None => return None,
        };

        Some(Complex::new(re, im))
    }
}

impl<S> Complex<S>
where
    S: Copy
{
    /// Get the real part of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(1_i32, 2_i32);
    /// 
    /// assert_eq!(z.real(), 1_i32);
    /// ```
    #[inline]
    pub const fn real(self) -> S {
        self.re
    }

    /// Get the imaginary part of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(1_i32, 2_i32);
    /// 
    /// assert_eq!(z.imaginary(), 2_i32);
    /// ```
    #[inline]
    pub const fn imaginary(self) -> S {
        self.im
    }

    /// Map an operation on that acts on the components of a complex number, returning 
    /// a complex number whose coordinates are of the new scalar type.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,  
    /// # };
    /// #
    /// let z: Complex<f32> = Complex::new(1_f32, 2_f32);
    /// let expected: Complex<f64> = Complex::new(-2_f64, -3_f64);
    /// let result: Complex<f64> = z.map(|comp| -(comp + 1_f32) as f64);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn map<T, F>(self, mut op: F) -> Complex<T> 
    where 
        F: FnMut(S) -> T 
    {
        Complex::new(op(self.re), op(self.im),)
    }
}

impl<S> Complex<S>
where
    S: SimdScalar
{
    /// Construct a new complex number from its real part.
    /// 
    /// The resulting complex number has a zero imaginary part.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::from_real(1_i32);
    /// 
    /// assert_eq!(z.real(), 1_i32);
    /// assert_eq!(z.imaginary(), 0_i32);
    /// ```
    #[inline]
    pub fn from_real(value: S) -> Self {
        Self::new(value, S::zero())
    }

    /// Construct a new complex number from its imaginary part.
    /// 
    /// The resulting complex number has a zero real part.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::from_imaginary(1_i32);
    /// 
    /// assert_eq!(z.real(), 0_i32);
    /// assert_eq!(z.imaginary(), 1_i32);
    /// ```
    #[inline]
    pub fn from_imaginary(value: S) -> Self {
        Self::new(S::zero(), value)
    }

    /// Construct a new additive unit (zero) complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let zero: Complex<i32> = Complex::zero();
    /// let other = Complex::new(92_i32, 137_i32);
    /// 
    /// assert_eq!(other + zero, other);
    /// assert_eq!(zero + other, other);
    /// assert_eq!(zero * other, zero);
    /// assert_eq!(other * zero, zero);
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self::new(S::zero(), S::zero())
    }

    /// Determine whether a complex number is equal to the zero complex number.
    ///  
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let zero: Complex<i32> = Complex::zero();
    /// let non_zero = Complex::new(92_i32, 137_i32);
    /// 
    /// assert!(zero.is_zero());
    /// assert!(!non_zero.is_zero());
    /// ```
    #[inline]
    pub fn is_zero(self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    /// Get the multiplicative unit complex number.
    /// 
    /// This is the unit real complex number `1`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let unit_complex: Complex<i32> = Complex::identity();
    /// let z = Complex::new(3_i32, 7_i32);
    /// 
    /// assert_eq!(z * unit_complex, z);
    /// assert_eq!(unit_complex * z, z);
    /// assert_eq!(unit_complex * unit_complex, unit_complex);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Self::new(S::one(), S::zero())
    }

    /// Get the multiplicative unit complex number.
    /// 
    /// This is a synonym for [`Complex::identity`].
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let unit_complex: Complex<i32> = Complex::one();
    /// let z = Complex::new(3_i32, 7_i32);
    /// 
    /// assert_eq!(z * unit_complex, z);
    /// assert_eq!(unit_complex * z, z);
    /// assert_eq!(unit_complex * unit_complex, unit_complex);
    /// ```
    #[inline]
    pub fn one() -> Self {
        Self::identity()
    }

    /// Determine whether a complex number is the identity complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let unit_complex: Complex<i32> = Complex::identity();
    /// 
    /// assert!(unit_complex.is_identity());
    /// 
    /// let z = Complex::new(5_i32, 6_i32);
    /// 
    /// assert!(!z.is_identity());
    /// ```
    #[inline]
    pub fn is_identity(self) -> bool {
        self.re.is_one() && self.im.is_zero()
    }

    /// Determine whether a complex number is the identity complex number.
    /// 
    /// This is a synonym for [`Complex::is_identity`].
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let unit_complex: Complex<i32> = Complex::one();
    /// 
    /// assert!(unit_complex.is_one());
    /// 
    /// let z = Complex::new(5_i32, 6_i32);
    /// 
    /// assert!(!z.is_one());
    /// ```
    #[inline]
    pub fn is_one(self) -> bool {
        self.re.is_one() && self.im.is_zero()
    }

    /// Get the multiplicative unit complex number.
    /// 
    /// This is a synonym for the identity complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let unit_re: Complex<i32> = Complex::unit_re();
    /// let identity: Complex<i32> = Complex::identity();
    /// 
    /// assert_eq!(unit_re, identity);
    /// ```
    #[inline]
    pub fn unit_re() -> Self {
        Self::identity()
    }

    /// Get the unit imaginary complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let i: Complex<i32> = Complex::unit_im();
    /// 
    /// assert_eq!(i.real(), 0_i32);
    /// assert_eq!(i.imaginary(), 1_i32);
    /// ```
    #[inline]
    pub fn unit_im() -> Self {
        Self::new(S::zero(), S::one())
    }

    /// Determine whether a complex number is real.
    /// 
    /// A complex number is real if its imaginary part is zero, i.e.
    /// it lies on the real line in the complex plane.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::from_real(2_i32);
    /// 
    /// assert!(z.is_real());
    /// assert!(!z.is_imaginary());
    /// ```
    #[inline]
    pub fn is_real(self) -> bool {
        self.im.is_zero()
    }

    /// Determine whether a complex number is imaginary.
    /// 
    /// A complex number is an imaginary number if its real part is zero, i.e.
    /// it lies on the imaginary line in the complex plane.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::from_imaginary(2_i32);
    /// 
    /// assert!(!z.is_real());
    /// assert!(z.is_imaginary());
    /// ```
    #[inline]
    pub fn is_imaginary(self) -> bool {
        self.re.is_zero()
    }

    /// Calculate the squared modulus of a complex number.
    /// 
    /// The modulus of a complex number is its Euclidean norm, defined as follows.
    /// Given a complex number `z`
    /// ```text
    /// modulus(z) = sqrt(re(z) * re(z) + im(z) * im(z))
    /// ```
    /// where `re(z)` is the real part of `z`, and `im(z)` is the imaginary part
    /// of `z`. The squared modulus of `z` is then defined to be
    /// ```text
    /// modulus_squared(z) = modulus(z) * modulus(z)
    ///                    = sqrt(re(z) * re(z) + im(z) * im(z)) * sqrt(re(z) * re(z) + im(z) * im(z))
    ///                    = re(z) * re(z) + im(z) * im(z)
    /// ```
    /// 
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(2_f32, 5_f32);
    /// 
    /// assert_eq!(z.modulus_squared(), 29_f32);
    /// ```
    #[inline]
    pub fn modulus_squared(self) -> S {
        self.norm_squared()
    }

    /// Calculate the squared norm of a complex number.
    /// 
    /// This is a synonym for [`Complex::modulus_squared`].
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(2_f32, 5_f32);
    /// 
    /// assert_eq!(z.norm_squared(), 29_f32);
    /// ```
    #[inline]
    pub fn norm_squared(self) -> S {
        self.re * self.re + self.im * self.im
    }

    /// Calculate the squared norm of a complex number.
    /// 
    /// This is a synonym for [`Complex::modulus_squared`].
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(2_f32, 5_f32);
    /// 
    /// assert_eq!(z.magnitude_squared(), 29_f32);
    /// ```
    #[inline]
    pub fn magnitude_squared(self) -> S {
        self.norm_squared()
    }

    /// Scale a complex number `self` by multiplying it by the scalar `scale`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 2_f64);
    /// let expected = Complex::new(3_f64, 6_f64);
    /// let result = z.scale(3_f64);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn scale(self, scale: S) -> Self {
        Self::new(self.re * scale, self.im * scale)
    }

    /// Unscale a complex number `self` by dividing it by the scalar `scale`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(3_f64, 6_f64);
    /// let expected = Complex::new(1_f64, 2_f64);
    /// let result = z.unscale(3_f64);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn unscale(self, scale: S) -> Self {
        Self::new(self.re / scale, self.im / scale)
    }
}

impl<S> Complex<S>
where
    S: SimdScalarSigned
{
    /// Calculate the complex conjugate of a complex number.
    /// 
    /// Given a complex number `z`, the complex conjugate of `z`
    /// is `z` with the sign of the imaginary part flipped, i.e.
    /// let `z := a + ib`, then `z* := a - ib`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(1_i32, 2_i32);
    /// let expected = Complex::new(1_i32, -2_i32);
    /// let result = z.conjugate();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(self.re, -self.im)
    }
}

impl<S> Complex<S>
where
    S: SimdScalarSigned
{
    /// Calculate the **L1** norm of a complex number.
    /// 
    /// # Examples
    /// 
    /// An example computing the **L1** norm of a [`f32`] complex number.
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// let z = Complex::new(2_f32, -5_f32);
    /// 
    /// assert_eq!(z.l1_norm(), 7_f32);
    /// ```
    /// 
    /// An example computing the **L1** norm of an [`i32`] complex number.
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// let z = Complex::new(2_i32, -5_i32);
    /// 
    /// assert_eq!(z.l1_norm(), 7_i32);
    /// ```
    #[inline]
    pub fn l1_norm(self) -> S { 
        self.re.abs() + self.im.abs()
    }
}

impl<S> Complex<S>
where
    S: SimdScalarFloat
{
    /// Returns `true` if all of the elements of a complex number are finite.
    /// Otherwise, it returns `false`.
    /// 
    /// # Example (Finite Complex Number)
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 2_f64);
    /// 
    /// assert!(z.is_finite());
    /// ```
    /// 
    /// # Example (Not A Finite Complex Number)
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, f64::NAN);
    /// assert!(!z.is_finite());
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Returns `true` if at least one of the elements of a complex number is `NaN`.
    /// Otherwise, it returns `false`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z1 = Complex::new(1_f64, 2_f64);
    /// let z2 = Complex::new(1_f64, f64::NAN);
    /// 
    /// assert!(!z1.is_nan());
    /// assert!(z2.is_nan());
    /// 
    /// ```
    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// Returns `true` is at least one of the elements of a complex number is infinite.
    /// Otherwise, it returns false.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z1 = Complex::new(1_f64, 2_f64);
    /// let z2 = Complex::new(1_f64, f64::NEG_INFINITY);
    /// 
    /// assert!(!z1.is_infinite());
    /// assert!(z2.is_infinite());
    /// ```
    #[inline]
    pub fn is_infinite(self) -> bool {
        !self.is_nan() && (self.re.is_infinite() || self.im.is_infinite())
    }

    /// Calculate the modulus of a complex number.
    /// 
    /// The modulus of a complex number is its Euclidean norm, defined as follows.
    /// Given a complex number `z`
    /// ```text
    /// modulus(z) = sqrt(re(z) * re(z) + im(z) * im(z))
    /// ```
    /// where `re(z)` is the real part of `z`, and `im(z)` is the imaginary part
    /// of `z`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(7_f64, 3_f64);
    /// 
    /// assert_eq!(z.modulus(), f64::sqrt(58_f64));
    /// ```
    #[inline]
    pub fn modulus(self) -> S {
        self.norm()
    }

    /// Calculate the **L2** (Euclidean) norm of a complex number.
    /// 
    /// This is a synonym for [`Complex::modulus`].
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(7_f64, 3_f64);
    /// 
    /// assert_eq!(z.norm(), f64::sqrt(58_f64));
    /// ```
    #[inline]
    pub fn norm(self) -> S {
        self.norm_squared().sqrt()
    }

    /// Calculate the L2 (Euclidean) norm of a complex number.
    /// 
    /// This is a synonym for [`Complex::modulus`].
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(7_f64, 3_f64);
    /// 
    /// assert_eq!(z.magnitude(), f64::sqrt(58_f64));
    /// ```
    #[inline]
    pub fn magnitude(self) -> S {
        self.norm()
    }

    /// Calculate the **L2** (Euclidean) norm of a complex number.
    /// 
    /// This is a synonuym for [`Complex::norm`].
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(7_f64, 3_f64);
    /// 
    /// assert_eq!(z.l2_norm(), f64::sqrt(58_f64));
    /// ```
    #[inline]
    pub fn l2_norm(self) -> S {
        self.norm()
    }

    /// Calculate the principal argument of a complex number.
    /// 
    /// In polar form, a complex number `z` can be written as 
    /// ```text
    /// z := |z| * exp(i * angle) := |z| * (cos(angle) + i * sin(angle))
    /// ```
    /// Consequently there is an ambiguity in choosing the angle for `z` in its 
    /// polar form; two complex numbers in polar form are equal if they have 
    /// identical norms and they differ by a factor of `2 * pi` in their 
    /// arguments. Let `z1` be another complex number. Then `z == z1` if and only 
    /// if `|z| == |z1|` and `angle1 == angle + 2 * pi * n1` where `n1` is an integer. 
    /// In order to resolve this ambiguity and make equality of complex numbers in polar 
    /// form well-defined, we restrict our choice of angle to be `-pi < angle <= pi` 
    /// (notice the open lower bound). This angle is called the principal argument of `z`, 
    /// the value returned by the function. Indeed, let `angle` be the principal 
    /// argument of `z`, and let `angle1` be the argument of `z1` that we defined 
    /// earlier, such that `angle1 == angle + 2 * pi * n1` for some integer `n1`. 
    /// We have
    /// ```text
    /// z1 = |z1| * exp(i * angle1) 
    ///    = |z1| * ( cos(angle1) + i * sin(angle1) )
    ///    = |z1| * ( cos(angle + 2 * pi * n1) + i * sin(angle + 2 * pi * n1) )
    ///    = |z1| * ( cos(angle) + i * sin(angle) )
    ///    = |z| * ( cos(angle) + i * sin(angle) )
    ///    = |z| * exp(i * angle)
    ///    = z
    /// ```
    /// as desired. Incidentally, the principal argument is given by 
    /// ```text
    /// Arg(z) = Arg(a + ib) := atan(b / a)`
    /// ```
    /// where `a + ib` is the complex number `z` written out in cartesian form. 
    /// This can be obtained from polar form by writing
    /// ```text
    /// z = |z| * (cos(angle) + i * sin(angle)) = |z| * cos(angle) + i * |z| * sin(angle)
    /// ```
    /// and reading off the resulting components.
    ///  
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let pi = core::f64::consts::PI;
    /// let angle = pi / 4_f64;
    /// let z1 = Complex::from_polar_decomposition(2_f64, Radians(angle));
    /// let z2 = Complex::from_polar_decomposition(2_f64, Radians(angle + 2_f64 * pi));
    /// 
    /// assert_relative_eq!(z1, z2, epsilon = 1e-10);
    /// assert_relative_eq!(z1.arg(), z2.arg(), epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn arg(self) -> S {
        self.im.atan2(self.re)
    }

    /// Construct a complex number from its polar form.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let pi_over_four = core::f64::consts::FRAC_PI_4;
    /// let expected = Complex::new(2_f64 / f64::sqrt(2_f64), 2_f64 / f64::sqrt(2_f64));
    /// let result = Complex::from_polar_decomposition(2_f64, Radians(pi_over_four));
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ``` 
    #[inline]
    pub fn from_polar_decomposition<A: Into<Radians<S>>>(scale: S, angle: A) -> Self {
        let _angle: Radians<S> = angle.into();
        Self::new(scale * _angle.cos(), scale * _angle.sin())
    }

    /// Construct a unit complex number from its polar form.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let pi_over_four = core::f64::consts::FRAC_PI_4;
    /// let expected = Complex::new(1_f64 / f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64));
    /// let result = Complex::from_angle(Radians(pi_over_four));
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn from_angle<A: Into<Radians<S>>>(angle: A) -> Self {
        Self::from_polar_decomposition(S::one(), angle)
    }

    /// Convert a complex number to its polar form.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// #     Radians,
    /// # };
    /// #
    /// let z = Complex::from_polar_decomposition(3_f64, Radians(2_f64));
    /// let (scale, angle) = z.polar_decomposition();
    /// 
    /// assert_eq!(scale, 3_f64);
    /// assert_eq!(angle, Radians(2_f64));
    /// ```
    #[inline]
    pub fn polar_decomposition(self) -> (S, Radians<S>) {
        (self.norm(), Radians(self.arg()))
    }

    /// Compute the exponential of a complex number.
    /// 
    /// Given a complex number `z = a + ib`, the exponential of z is given by
    /// ```text
    /// exp(z) := exp(a + ib) 
    ///         = exp(a) * exp(ib) 
    ///         = exp(a) * (cos(b) + i * sin(b))
    ///         = exp(a) * cos(b) + i * exp(a) * sin(b)
    /// ```
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(2_f64, 5_f64);
    /// let expected = Complex::new(2.09599580151_f64, -7.08554526009_f64);
    /// let result = z.exp();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn exp(self) -> Self {
        let exp_re = self.re.exp();
        let (sin_im, cos_im) = self.im.sin_cos();

        Self::new(exp_re * cos_im, exp_re * sin_im)
    }

    /// Calculate the principal value of the natural logarithm of a complex number.
    /// 
    /// Given a complex number `z`, the principal value of the natural logarithm 
    /// of `z` is the logarithm whose imaginary part lies in `-pi < Im(z) <= pi` 
    /// (notice the open lower bound). The natural logarithm of zero is undefined 
    /// since there is no complex number `w` satisfying `exp(w) = 0`. 
    /// 
    /// We need the notion of the principal value because given the polar form of
    /// a complex number `z`, the are infinitely many angles that have the same
    /// natural logarithm, since they haven identical cosines and sines. We need a
    /// way to choose a single canonical value in order to make the natural logarithm
    /// well-defined. That canonical value is the principal value. The principal value
    /// has the form
    /// ```text
    /// Log(z) := ln(|z|) + i * Arg(z)
    /// ```
    /// where `Arg(z)` is the principal argument of `z`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// #     Radians,
    /// # };
    /// #
    /// let pi = core::f64::consts::PI;
    /// let z = Complex::from_polar_decomposition(
    ///     3_f64, Radians(pi / 6_f64) + Radians((2_f64 * pi))
    /// );
    /// let expected = Complex::new(z.norm().ln(), pi / 6_f64);
    /// let result = z.ln();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn ln(self) -> Self {
        let norm_self = self.norm();
        let arg_self = self.arg();

        Self::new(norm_self.ln(), arg_self)
    }

    /// Calculate the positive square root of a complex number.
    /// 
    /// Given a complex number `z`, the square root of `z` is a complex number 
    /// `w` such that `w * w = z`. The formula for `sqrt(z)` is given by the following
    /// formula. Let `z := a + ib` where `a` is the real parts of `z`, and `b` is the 
    /// imaginary part of `z`.
    /// ```text
    ///                              t + 2 * pi * n              t + 2 * pi * n
    /// sqrt(z) = sqrt(|z|) * ( cos(----------------) + i * sin(----------------) )
    ///                                     2                           2
    /// ```
    /// where `|z|` is the modulus (norm) of `z`, `t` is the principal argument of `z`, and `n`
    /// is the nth angle satisfying the above equation. In the case of the square root, there
    /// are two solutions: `n = 0` and `n = 1`. The `n = 0` solution corresponds 
    /// to the solution returned by the function, and the `n = 1` case corresponds to the
    /// solution `-w` which differs only by a sign. Indeed, let
    /// ```text
    ///                                        t               t
    /// w0 := w = sqrt(z) = sqrt(|z|) * ( cos(---) + i * sin (---) )
    ///                                        2               2
    /// ```
    /// which is the `n = 0` solution. Let
    /// ```text
    ///                         t + 2 * pi              t + 2 * pi
    /// w1 = sqrt(|z|) * ( cos(------------) + i * sin(------------) )
    ///                              2                       2
    /// ```
    /// Observe that
    /// ```text
    /// cos((t + 2 * pi) / 2) = cos((t / 2) + pi) = -cos(t / 2)
    /// sin((t + 2 * pi) / 2) = sin((t / 2) + pi) = -sin(t / 2)
    /// ```
    /// so that
    /// ```text
    ///                          t                 t
    /// w1 = sqrt(|z|) * ( -cos(---) + i * ( -sin(---) )
    ///                          2                 2
    ///
    ///                          t              t
    ///    = -sqrt(|z|) * ( cos(---) + i * sin(---) )
    ///                          2              2
    /// 
    ///    = -w
    /// ```
    /// Thus the complex number square root is indeed a proper square root with two 
    /// solutions given by `p` and `-p`. We illustate this with an example.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 4_f64);
    /// let sqrt_z = z.sqrt();
    ///
    /// assert_relative_eq!(sqrt_z * sqrt_z, z, epsilon = 1e-10);
    /// 
    /// let minus_sqrt_z = -sqrt_z;
    /// 
    /// assert_relative_eq!(minus_sqrt_z * minus_sqrt_z, z, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn sqrt(self) -> Self {
        let two = S::one() + S::one();
        let sqrt_norm = self.norm().sqrt();
        let angle = self.arg();
        let (sin_angle_over_two, cos_angle_over_two) = (angle / two).sin_cos();

        Self::new(sqrt_norm * cos_angle_over_two, sqrt_norm * sin_angle_over_two)
    }

    /// Calculate the power of a complex number where the exponent is a floating 
    /// point number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let pi_over_four = core::f64::consts::FRAC_PI_4;
    /// let z = Complex::from_polar_decomposition(2_f64, Radians(pi_over_four));
    /// let exponent = 5_f64;
    /// let expected = Complex::from_polar_decomposition(32_f64, Radians(exponent * pi_over_four));
    /// let result = z.powf(exponent);
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-10);
    /// ```
    #[inline]
    pub fn powf(self, exponent: S) -> Self {
        (self.ln() * exponent).exp()
    }

    /// Calculate the multiplicative inverse of a complex number.
    /// 
    /// The multiplicative inverse of a complex number `z` is a complex 
    /// number `w` such that `w * z = z * w = 1`.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(2_f64, 3_f64);
    /// let expected = Some(Complex::new(2_f64 / 13_f64, -3_f64 / 13_f64));
    /// let result = z.inverse();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn inverse(self) -> Option<Self> {
        let norm_squared = self.norm_squared();
        if norm_squared.is_zero() {
            None
        } else {
            Some(Self::new(
                 self.re / norm_squared,
                -self.im / norm_squared
            ))
        }
    }

    /// Determine whether a complex number is invertible.
    /// 
    /// Returns `false` if the modulus of the complex number is sufficiently
    /// close to zero.
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// #
    /// let z: Complex<f64> = Complex::unit_im();
    /// 
    /// assert!(z.is_invertible());
    /// ```
    #[inline]
    pub fn is_invertible(self) -> bool {
        self.is_invertible_eps(S::default_epsilon())
    }

    #[inline]
    fn is_invertible_eps(self, epsilon: S) -> bool {
        self.norm_squared() >= epsilon * epsilon
    }

    /// Compute the complex cosine of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 2_f64);
    /// let expected = Complex::new(2.032723007_f64, -3.051897799_f64);
    /// let result = z.cos();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn cos(self) -> Self {
        Self::new(
             self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh()
        )
    }

    /// Compute the complex arccosine of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 3_f64);
    /// let expected = Complex::new(1.2631926773_f64, -1.8641615442_f64);
    /// let result = z.acos();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// 
    /// let z1 = Complex::new(-2_f64, 5_f64);
    /// 
    /// assert_relative_eq!(z1.acos().cos(), z1, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn acos(self) -> Self {
        let i = Self::unit_im();

        -i * Self::ln((i * Self::sqrt(Self::one() - self * self)) + self)
    }

    /// Compute the complex sine of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 2_f64);
    /// let expected = Complex::new(3.165778513_f64, 1.959601041_f64);
    /// let result = z.sin();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn sin(self) -> Self {
        Self::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh()
        )
    }

    /// Compute the complex arcsine of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 3_f64);
    /// let expected = Complex::new(0.307603650_f64, 1.864161544_f64);
    /// let result = z.asin();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// 
    /// let z1 = Complex::new(-2_f64, 5_f64);
    /// 
    /// assert_relative_eq!(z1.asin().sin(), z1, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn asin(self) -> Self {
        let i = Self::unit_im();

        -i * ((Self::one() - self * self).sqrt() + i * self).ln()
    }

    /// Compute the complex tangent of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 1_f64);
    /// let expected = Complex::new(0.2717525853_f64, 1.0839233273_f64);
    /// let result = z.tan();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn tan(self) -> Self {
        let two_times_re = self.re + self.re;
        let two_times_im = self.im + self.im;
        let denominator = two_times_re.cos() + two_times_im.cosh();
        let tan_re = two_times_re.sin() / denominator;
        let tan_im = two_times_im.sinh() / denominator;

        Self::new(tan_re, tan_im)
    }

    /// Compute the complex arctangent of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 1_f64);
    /// let expected = Complex::new(1.0172219679_f64, 0.4023594781_f64);
    /// let result = z.atan();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// 
    /// let z1 = Complex::new(-2_f64, 5_f64);
    /// 
    /// assert_relative_eq!(z1.atan().tan(), z1, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn atan(self) -> Self {
        let i = Self::unit_im();
        let one = Self::one();
        let two = one + one;
        if self == i {
            return Self::new(S::zero(), S::infinity());
        } else if self == -i {
            return Self::new(S::zero(), S::neg_infinity());
        }

        (Self::ln(one + i * self) - Self::ln(one - i * self)) / (two * i)
    }

    /// Compute the complex secant of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(2_f64, 3_f64);
    /// let expected = Complex::new(-0.0416749644_f64, 0.0906111371_f64);
    /// let result = z.sec();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn sec(self) -> Self {
        let one = S::one();
        let two = one + one;
        let denominator = S::cos(self.re + self.re) + S::cosh(self.im + self.im);
        let sec_re = two * S::cos(self.re) * S::cosh(self.im); 
        let sec_im = two * S::sin(self.re) * S::sinh(self.im);

        Self::new(sec_re, sec_im) / denominator
    }

    /// Compute the complex arcsecant of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(2_f64, 3_f64);
    /// let expected = Complex::new(1.4204107224_f64, 0.2313346985_f64);
    /// let result = z.asec();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// 
    /// let z1 = Complex::new(1_f64, -2_f64);
    /// 
    /// assert_relative_eq!(z1.asec().sec(), z1, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn asec(self) -> Self {
        let one = Self::unit_re();
        let i = Self::unit_im();
        let one_over_self = one / self;
        let one_over_self_squared = one / (self * self);
        
        -i * (i * (one - one_over_self_squared).sqrt() + one_over_self).ln()
    }

    /// Compute the complex cosecant of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(2_f64, 3_f64);
    /// let expected = Complex::new(0.0904732097, 0.0412009862);
    /// let result = z.csc();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn csc(self) -> Self {
        let one = S::one();
        let two = one + one;
        let denominator = S::cos(two * self.re) - S::cosh(two * self.im);
        let csc_re = -two * S::sin(self.re) * S::cosh(self.im);
        let csc_im = two * S::cos(self.re) * S::sinh(self.im);

        Self::new(csc_re, csc_im) / denominator
    }

    /// Compute the complex arccosecant of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(2_f64, 3_f64);
    /// let expected = Complex::new(0.1503856043_f64, -0.2313346985_f64);
    /// let result = z.acsc();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// 
    /// let z1 = Complex::new(1_f64, -2_f64);
    /// 
    /// assert_relative_eq!(z1.acsc().csc(), z1, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn acsc(self) -> Self {
        let one = Complex::unit_re();
        let i = Complex::unit_im();
        let i_over_self = i / self;
        let one_over_self_squared = one / (self * self);
        
        -i * ((one - one_over_self_squared).sqrt() + i_over_self).ln()
    }

    /// Compute the complex cotangent of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(4_f64, 7_f64);
    /// let expected = Complex::new(0.00000164535_f64, -0.9999997580_f64);
    /// let result = z.cot();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn cot(self) -> Self {
        let two_times_re = self.re + self.re;
        let two_times_im = self.im + self.im;
        let denominator = S::cos(two_times_re) - S::cosh(two_times_im);
        let cot_re = -S::sin(two_times_re);
        let cot_im = S::sinh(two_times_im);

        Self::new(cot_re, cot_im) / denominator
    }

    /// Compute the complex arccotangent of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(4_f64, 7_f64);
    /// let expected = Complex::new(0.0621774972_f64, -0.1076957290_f64);
    /// let result = z.acot();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// 
    /// let z1 = Complex::new(1_f64, 1_f64);
    /// 
    /// assert_relative_eq!(z1.acot().cot(), z1, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn acot(self) -> Self {
        let one = S::one();
        let two = one + one;
        let i = Self::unit_im();
        if self == i {
            return Self::new(S::zero(), S::neg_infinity());
        } else if self == -i {
            return Self::new(S::zero(), S::infinity());
        }
        
        (-i / two) * Self::ln((i * self - one) / (i * self + one))
    }


    /// Compute the complex hyperbolic cosine of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, -5_f64);
    /// let expected = Complex::new(0.4377136252_f64, 1.1269289521_f64);
    /// let result = z.cosh();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn cosh(self) -> Self {
        Self::new(
            self.re.cosh() * self.im.cos(),
            self.re.sinh() * self.im.sin()
        )
    }

    /// Compute the complex hyperbolic arccosine of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, -5_f64);
    /// let expected = Complex::new(2.3309746530_f64, -1.3770031902_f64);
    /// let result = z.acosh();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// 
    /// let z1 = Complex::new(2_f64, 3_f64);
    /// 
    /// assert_relative_eq!(z1.acosh().cosh(), z1, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn acosh(self) -> Self {
        let one = Self::one();
        let two = one + one;
        let z = Self::sqrt((self + one) / two) + Self::sqrt((self - one) / two);

        two * Self::ln(z)
    }

    /// Compute the complex hyperbolic sine of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, -5_f64);
    /// let expected = Complex::new(0.3333601389_f64, 1.4796974784_f64);
    /// let result = z.sinh();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn sinh(self) -> Self {
        Self::new(
            self.re.sinh() * self.im.cos(),
            self.re.cosh() * self.im.sin()
        )
    }

    /// Compute the complex hyperbolic arcsine of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, -5_f64);
    /// let expected = Complex::new(2.3132209417_f64, -1.3696012470_f64);
    /// let result = z.asinh();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// 
    /// let z1 = Complex::new(2_f64, 3_f64);
    /// 
    /// assert_relative_eq!(z1.asinh().sinh(), z1, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn asinh(self) -> Self {
        let one = Self::one();

        Self::ln(self + Self::sqrt(one + self * self))
    }

    /// Compute the complex hyperbolic tangent of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(-3_f64, 4_f64);
    /// let expected = Complex::new(-1.0007095360_f64, 0.0049082580_f64);
    /// let result = z.tanh();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn tanh(self) -> Self {
        let remainder = (self.im - S::frac_pi_2()) % S::two_pi();
        if remainder.is_zero() && remainder.is_sign_positive() {
            return Complex::new(S::zero(), S::infinity());
        }

        if remainder.is_zero() && remainder.is_sign_negative() {
            return Complex::new(S::zero(), S::neg_infinity());
        }

        let two_times_re = self.re + self.re;
        let two_times_im = self.im + self.im;
        let denominator = S::cosh(two_times_re) + S::cos(two_times_im);
        let tanh_re = S::sinh(two_times_re);
        let tanh_im = S::sin(two_times_im);

        Self::new(tanh_re, tanh_im) / denominator
    }

    /// Compute the principal value complex hyperbolic arctangent of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(-3_f64, 4_f64);
    /// let expected = Complex::new(-0.1175009073_f64, 1.4099210495_f64);
    /// let result = z.atanh();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// 
    /// let z1 = Complex::new(2_f64, 3_f64);
    /// 
    /// assert_relative_eq!(z1.atanh().tanh(), z1, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn atanh(self) -> Self {
        let one = Self::one();
        let two = one + one;
        if self == one {
            return Self::new(S::infinity(), S::zero());
        } else if self == -one {
            return Self::new(S::neg_infinity(), S::zero());
        }
        
        (Self::ln(one + self) - Self::ln(one - self)) / two
    }

    /// Compute the complex hyperbolic secant of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(-3_f64, 4_f64);
    /// let expected = Complex::new(-0.0652940278_f64, -0.0752249603_f64);
    /// let result = z.sech();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn sech(self) -> Self {
        let one = S::one();
        let two = one + one;
        let two_times_re = self.re + self.re;
        let two_times_im = self.im + self.im;
        let denominator = S::cosh(two_times_re) + S::cos(two_times_im);
        let sech_re = two * S::cosh(self.re) * S::cos(self.im);
        let sech_im = -two * S::sinh(self.re) * S::sin(self.im);

        Self::new(sech_re, sech_im) / denominator
    }

    /// Compute the complex hyperbolic arcsecant of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(-3_f64, 4_f64);
    /// let expected = Complex::new(0.1604455337_f64, -1.6895470581_f64);
    /// let result = z.asech();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// 
    /// let z1 = Complex::new(1_f64, 2_f64);
    /// 
    /// assert_relative_eq!(z1.asech().sech(), z1, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn asech(self) -> Self {
        let one = Self::one();
        let one_over_self = one / self;
        let z1 = Self::sqrt(one_over_self + one);
        let z2 = Self::sqrt(one_over_self - one);
        
        Self::ln(one_over_self + z1 * z2)
    }

    /// Compute the complex hyperbolic cosecant of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 1_f64);
    /// let expected = Complex::new(0.3039310016_f64, -0.6215180171_f64);
    /// let result = z.csch();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn csch(self) -> Self {
        let one = S::one();
        let two = one + one;
        let two_times_re = self.re + self.re;
        let two_times_im = self.im + self.im;
        let denominator = S::cos(two_times_im) - S::cosh(two_times_re);
        let csch_re = -two * S::sinh(self.re) * S::cos(self.im);
        let csch_im = two * S::cosh(self.re) * S::sin(self.im);

        Self::new(csch_re, csch_im) / denominator
    }

    /// Compute the complex hyperbolic arccosecant of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 1_f64);
    /// let expected = Complex::new(0.5306375309_f64, -0.4522784471_f64);
    /// let result = z.acsch();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// 
    /// let z1 = Complex::new(1_f64, 2_f64);
    /// 
    /// assert_relative_eq!(z1.acsch().csch(), z1, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn acsch(self) -> Self {
        let one = Self::one();
        let one_over_self = one / self;
        let one_over_self_squared = one / (self * self);

        Self::ln(one_over_self + Self::sqrt(one_over_self_squared + one))
    }

    /// Compute the complex hyperbolic cotangent of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 2_f64);
    /// let expected = Complex::new(0.8213297974_f64, 0.1713836129_f64);
    /// let result = z.coth();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn coth(self) -> Self {
        let two_times_re = self.re + self.re;
        let two_times_im = self.im + self.im;
        let denominator = S::cos(two_times_im) - S::cosh(two_times_re);
        let coth_re = -S::sinh(two_times_re);
        let coth_im = S::sin(two_times_im);

        Self::new(coth_re, coth_im) / denominator
    }

    /// Compute the complex hyperbolic arccotangent of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg_core::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     assert_relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 2_f64);
    /// let expected = Complex::new(0.1732867951_f64, -0.3926990816_f64);
    /// let result = z.acoth();
    /// 
    /// assert_relative_eq!(result, expected, epsilon = 1e-8);
    /// 
    /// let z1 = Complex::new(2_f64, 3_f64);
    /// 
    /// assert_relative_eq!(z1.atanh().tanh(), z1, epsilon = 1e-8);
    /// ```
    #[inline]
    pub fn acoth(self) -> Self {
        let unit_re = Self::unit_re();
        let one = S::one();
        let two = one + one;
        let one_half = one / two;
        if self == unit_re {
            return Self::new(S::infinity(), S::zero());
        } else if self == -unit_re {
            return Self::new(S::neg_infinity(), S::zero());
        }
        
        Self::ln((self + one) / (self - one)) * one_half
    }
}

impl<S> AsRef<[S; 2]> for Complex<S> {
    #[inline]
    fn as_ref(&self) -> &[S; 2] {
        unsafe { 
            &*(self as *const Complex<S> as *const [S; 2])
        }
    }
}

impl<S> AsRef<(S, S)> for Complex<S> {
    #[inline]
    fn as_ref(&self) -> &(S, S) {
        unsafe { 
            &*(self as *const Complex<S> as *const (S, S))
        }
    }
}

impl<S> AsMut<[S; 2]> for Complex<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; 2] {
        unsafe { 
            &mut *(self as *mut Complex<S> as *mut [S; 2])
        }
    }
}

impl<S> AsMut<(S, S)> for Complex<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut (S, S) {
        unsafe { 
            &mut *(self as *mut Complex<S> as *mut (S, S))
        }
    }
}

impl<S> ops::Index<usize> for Complex<S> 
where 
    S: SimdScalar 
{
    type Output = S;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let v: &[S; 2] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::Range<usize>> for Complex<S> 
where 
    S: SimdScalar 
{
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        let v: &[S; 2] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeTo<usize>> for Complex<S> 
where 
    S: SimdScalar 
{
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        let v: &[S; 2] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeFrom<usize>> for Complex<S> 
where 
    S: SimdScalar 
{
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        let v: &[S; 2] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::Index<ops::RangeFull> for Complex<S> 
where 
    S: SimdScalar 
{
    type Output = [S];

    #[inline]
    fn index(&self, index: ops::RangeFull) -> &Self::Output {
        let v: &[S; 2] = self.as_ref();
        &v[index]
    }
}

impl<S> ops::IndexMut<usize> for Complex<S> 
where 
    S: SimdScalar 
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut S {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::Range<usize>> for Complex<S> 
where 
    S: SimdScalar 
{
    #[inline]
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut [S] {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeTo<usize>> for Complex<S> 
where 
    S: SimdScalar 
{
    #[inline]
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut [S] {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFrom<usize>> for Complex<S> 
where 
    S: SimdScalar 
{
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut [S] {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFull> for Complex<S> 
where 
    S: SimdScalar 
{
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFull) -> &mut [S] {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> Default for Complex<S>
where
    S: SimdScalar
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<S> From<S> for Complex<S>
where
    S: SimdScalar
{
    #[inline]
    fn from(re: S) -> Self {
        Self::new(re, S::zero())
    }
}

impl<S> From<&S> for Complex<S>
where
    S: SimdScalar
{
    #[inline]
    fn from(re: &S) -> Self {
        Self::new(*re, S::zero())
    }
}

impl<S> From<(S, S)> for Complex<S>
where
    S: SimdScalar
{
    #[inline]
    fn from(v: (S, S)) -> Self {
        Self::new(v.0, v.1)
    }
}

impl<S> From<[S; 2]> for Complex<S>
where
    S: SimdScalar
{
    #[inline]
    fn from(v: [S; 2]) -> Self {
        Self::new(v[0], v[1])
    }
}

impl<S> From<&(S, S)> for Complex<S>
where
    S: SimdScalar
{
    #[inline]
    fn from(v: &(S, S)) -> Self {
        Self::new(v.0, v.1)
    }
}

impl<S> From<&[S; 2]> for Complex<S>
where
    S: SimdScalar
{
    #[inline]
    fn from(v: &[S; 2]) -> Self {
        Self::new(v[0], v[1])
    }
}

impl<'a, S> From<&'a (S, S)> for &'a Complex<S>
where
    S: SimdScalar
{
    #[inline]
    fn from(v: &'a (S, S)) -> &'a Complex<S> {
        unsafe {
            &*(v as *const (S, S) as *const Complex<S>)
        }
    }
}

impl<'a, S> From<&'a [S; 2]> for &'a Complex<S>
where
    S: SimdScalar
{
    #[inline]
    fn from(v: &'a [S; 2]) -> &'a Complex<S> {
        unsafe {
            &*(v as *const [S; 2] as *const Complex<S>)
        }
    }
}

impl<S> fmt::Display for Complex<S>
where
    S: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{} + i{}", self.re, self.im)
    }
}

impl<S> ops::Neg for Complex<S>
where
    S: SimdScalarSigned
{
    type Output = Complex<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::new(-self.re, -self.im)
    }
}

impl<S> ops::Neg for &Complex<S>
where
    S: SimdScalarSigned
{
    type Output = Complex<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::new(-self.re, -self.im)
    }
}

impl<S> ops::Add<Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: Complex<S>) -> Self::Output {
        Self::Output::new(self.re + other.re, self.im + other.im)
    }
}

impl<S> ops::Add<&Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: &Complex<S>) -> Self::Output {
        Self::Output::new(self.re + other.re, self.im + other.im)
    }
}

impl<S> ops::Add<Complex<S>> for &Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: Complex<S>) -> Self::Output {
        Self::Output::new(self.re + other.re, self.im + other.im)
    }
}

impl<'a, 'b, S> ops::Add<&'b Complex<S>> for &'a Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: &'b Complex<S>) -> Self::Output {
        Self::Output::new(self.re + other.re, self.im + other.im)
    }
}

impl<S> ops::Add<S> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: S) -> Self::Output {
        Self::Output::new(self.re + other, self.im)
    }
}

impl<S> ops::Add<&S> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: &S) -> Self::Output {
        Self::Output::new(self.re + *other, self.im)
    }
}

impl<S> ops::Add<S> for &Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: S) -> Self::Output {
        Self::Output::new(self.re + other, self.im)
    }
}

impl<'a, 'b, S> ops::Add<&'b S> for &'a Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: &'b S) -> Self::Output {
        Self::Output::new(self.re + *other, self.im)
    }
}

impl<S> ops::Sub<Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: Complex<S>) -> Self::Output {
        Self::Output::new(self.re - other.re, self.im - other.im)
    }
}

impl<S> ops::Sub<&Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: &Complex<S>) -> Self::Output {
        Self::Output::new(self.re - other.re, self.im - other.im)
    }
}

impl<S> ops::Sub<Complex<S>> for &Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: Complex<S>) -> Self::Output {
        Self::Output::new(self.re - other.re, self.im - other.im)
    }
}

impl<'a, 'b, S> ops::Sub<&'b Complex<S>> for &'a Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: &'b Complex<S>) -> Self::Output {
        Self::Output::new(self.re - other.re, self.im - other.im)
    }
}

impl<S> ops::Sub<S> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: S) -> Self::Output {
        Self::Output::new(self.re - other, self.im)
    }
}

impl<S> ops::Sub<&S> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: &S) -> Self::Output {
        Self::Output::new(self.re - *other, self.im)
    }
}

impl<S> ops::Sub<S> for &Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: S) -> Self::Output {
        Self::Output::new(self.re - other, self.im)
    }
}

impl<'a, 'b, S> ops::Sub<&'b S> for &'a Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: &'b S) -> Self::Output {
        Self::Output::new(self.re - *other, self.im)
    }
}

impl<S> ops::Mul<Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn mul(self, other: Complex<S>) -> Self::Output {
        let real_part = self.re * other.re - self.im * other.im;
        let imaginary_part = self.re * other.im + self.im * other.re;

        Self::Output::new(real_part, imaginary_part)
    }
}

impl<S> ops::Mul<&Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn mul(self, other: &Complex<S>) -> Self::Output {
        let real_part = self.re * other.re - self.im * other.im;
        let imaginary_part = self.re * other.im + self.im * other.re;

        Self::Output::new(real_part, imaginary_part)
    }
}

impl<S> ops::Mul<Complex<S>> for &Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn mul(self, other: Complex<S>) -> Self::Output {
        let real_part = self.re * other.re - self.im * other.im;
        let imaginary_part = self.re * other.im + self.im * other.re;

        Self::Output::new(real_part, imaginary_part)
    }
}

impl<'a, 'b, S> ops::Mul<&'b Complex<S>> for &'a Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn mul(self, other: &'b Complex<S>) -> Self::Output {
        let real_part = self.re * other.re - self.im * other.im;
        let imaginary_part = self.re * other.im + self.im * other.re;

        Self::Output::new(real_part, imaginary_part)
    }
}

impl<S> ops::Mul<S> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Self::Output::new(self.re * other, self.im * other)
    }
}

impl<S> ops::Mul<&S> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn mul(self, other: &S) -> Self::Output {
        Self::Output::new(self.re * *other, self.im * *other)
    }
}

impl<S> ops::Mul<S> for &Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Self::Output::new(self.re * other, self.im * other)
    }
}

impl<'a, 'b, S> ops::Mul<&'b S> for &'a Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn mul(self, other: &'b S) -> Self::Output {
        Self::Output::new(self.re * *other, self.im * *other)
    }
}

impl<S> ops::Div<Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn div(self, other: Complex<S>) -> Self::Output {
        let denominator = other.re * other.re + other.im * other.im;
        let re = (self.re * other.re + self.im * other.im) / denominator;
        let im = (self.im * other.re - self.re * other.im) / denominator;

        Self::Output::new(re, im)
    }
}

impl<S> ops::Div<&Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn div(self, other: &Complex<S>) -> Self::Output {
        let denominator = other.re * other.re + other.im * other.im;
        let re = (self.re * other.re + self.im * other.im) / denominator;
        let im = (self.im * other.re - self.re * other.im) / denominator;

        Self::Output::new(re, im)
    }
}

impl<S> ops::Div<Complex<S>> for &Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn div(self, other: Complex<S>) -> Self::Output {
        let denominator = other.re * other.re + other.im * other.im;
        let re = (self.re * other.re + self.im * other.im) / denominator;
        let im = (self.im * other.re - self.re * other.im) / denominator;

        Self::Output::new(re, im)
    }
}

impl<'a, 'b, S> ops::Div<&'b Complex<S>> for &'a Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn div(self, other: &'b Complex<S>) -> Self::Output {
        let denominator = other.re * other.re + other.im * other.im;
        let re = (self.re * other.re + self.im * other.im) / denominator;
        let im = (self.im * other.re - self.re * other.im) / denominator;

        Self::Output::new(re, im)
    }
}

impl<S> ops::Div<S> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Self::Output::new(self.re / other, self.im / other)
    }
}

impl<S> ops::Div<&S> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn div(self, other: &S) -> Self::Output {
        Self::Output::new(self.re / *other, self.im / *other)
    }
}

impl<S> ops::Div<S> for &Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Self::Output::new(self.re / other, self.im / other)
    }
}

impl<'a, 'b, S> ops::Div<&'b S> for &'a Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn div(self, other: &'b S) -> Self::Output {
        Self::Output::new(self.re / *other, self.im / *other)
    }
}

impl<S> ops::Rem<S> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        Self::Output::new(self.re % other, self.im % other)
    }
}

impl<S> ops::Rem<&S> for Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn rem(self, other: &S) -> Self::Output {
        Self::Output::new(self.re % *other, self.im % *other)
    }
}

impl<S> ops::Rem<S> for &Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        Self::Output::new(self.re % other, self.im % other)
    }
}

impl<'a, 'b, S> ops::Rem<&'b S> for &'a Complex<S>
where
    S: SimdScalar
{
    type Output = Complex<S>;

    #[inline]
    fn rem(self, other: &'b S) -> Self::Output {
        Self::Output::new(self.re % *other, self.im % *other)
    }
}

impl<S> ops::AddAssign<Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    fn add_assign(&mut self, other: Complex<S>) {
        self.re += other.re;
        self.im += other.im;
    }
}

impl<S> ops::AddAssign<&Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    fn add_assign(&mut self, other: &Complex<S>) {
        self.re += other.re;
        self.im += other.im;
    }
}

impl<S> ops::AddAssign<S> for Complex<S>
where
    S: SimdScalar
{
    fn add_assign(&mut self, other: S) {
        self.re += other;
    }
}

impl<S> ops::AddAssign<&S> for Complex<S>
where
    S: SimdScalar
{
    fn add_assign(&mut self, other: &S) {
        self.re += *other;
    }
}

impl<S> ops::SubAssign<Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    fn sub_assign(&mut self, other: Complex<S>) {
        self.re -= other.re;
        self.im -= other.im;
    }
}

impl<S> ops::SubAssign<&Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    fn sub_assign(&mut self, other: &Complex<S>) {
        self.re -= other.re;
        self.im -= other.im;
    }
}

impl<S> ops::SubAssign<S> for Complex<S>
where
    S: SimdScalar
{
    fn sub_assign(&mut self, other: S) {
        self.re -= other;
    }
}

impl<S> ops::SubAssign<&S> for Complex<S>
where
    S: SimdScalar
{
    fn sub_assign(&mut self, other: &S) {
        self.re -= *other;
    }
}

impl<S> ops::MulAssign<Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    fn mul_assign(&mut self, other: Complex<S>) {
        let a = self.re;

        self.re *= other.re;
        self.re -= self.im * other.im;

        self.im *= other.re;
        self.im += a * other.im;
    }
}

impl<S> ops::MulAssign<&Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    fn mul_assign(&mut self, other: &Complex<S>) {
        let a = self.re;

        self.re *= other.re;
        self.re -= self.im * other.im;

        self.im *= other.re;
        self.im += a * other.im;
    }
}

impl<S> ops::MulAssign<S> for Complex<S>
where
    S: SimdScalar
{
    fn mul_assign(&mut self, other: S) {
        self.re *= other;
        self.im *= other;
    }
}

impl<S> ops::MulAssign<&S> for Complex<S>
where
    S: SimdScalar
{
    fn mul_assign(&mut self, other: &S) {
        self.re *= *other;
        self.im *= *other;
    }
}

impl<S> ops::DivAssign<Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    fn div_assign(&mut self, other: Complex<S>) {
        let a = self.re;
        let norm_squared = other.norm_squared();

        self.re *= other.re;
        self.re += self.im * other.im;
        self.re /= norm_squared;

        self.im *= other.re;
        self.im -= a * other.im;
        self.im /= norm_squared;
    }
}

impl<S> ops::DivAssign<&Complex<S>> for Complex<S>
where
    S: SimdScalar
{
    fn div_assign(&mut self, other: &Complex<S>) {
        let a = self.re;
        let norm_squared = other.norm_squared();

        self.re *= other.re;
        self.re += self.im * other.im;
        self.re /= norm_squared;

        self.im *= other.re;
        self.im -= a * other.im;
        self.im /= norm_squared;
    }
}

impl<S> ops::DivAssign<S> for Complex<S>
where
    S: SimdScalar
{
    fn div_assign(&mut self, other: S) {
        self.re /= other;
        self.im /= other;
    }
}

impl<S> ops::DivAssign<&S> for Complex<S>
where
    S: SimdScalar
{
    fn div_assign(&mut self, other: &S) {
        self.re /= *other;
        self.im /= *other;
    }
}

impl<S> ops::RemAssign<S> for Complex<S>
where
    S: SimdScalar
{
    fn rem_assign(&mut self, other: S) {
        self.re %= other;
        self.im %= other;
    }
}

impl<S> ops::RemAssign<&S> for Complex<S>
where
    S: SimdScalar
{
    fn rem_assign(&mut self, other: &S) {
        self.re %= *other;
        self.im %= *other;
    }
}

macro_rules! impl_scalar_complex_add_ops {
    ($($Lhs:ty),* $(,)*) => {$(
        impl ops::Add<Complex<$Lhs>> for $Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn add(self, other: Complex<$Lhs>) -> Self::Output {
                Self::Output::new(self + other.re, other.im)
            }
        }

        impl ops::Add<&Complex<$Lhs>> for $Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn add(self, other: &Complex<$Lhs>) -> Self::Output {
                Self::Output::new(self + other.re, other.im)
            }
        }

        impl ops::Add<Complex<$Lhs>> for &$Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn add(self, other: Complex<$Lhs>) -> Self::Output {
                Self::Output::new(self + other.re, other.im)
            }
        }

        impl<'a, 'b> ops::Add<&'a Complex<$Lhs>> for &'b $Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn add(self, other: &'a Complex<$Lhs>) -> Self::Output {
                Self::Output::new(self + other.re, other.im)
            }
        }
    )*}
}

impl_scalar_complex_add_ops!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);


macro_rules! impl_scalar_complex_sub_ops {
    ($($Lhs:ty),* $(,)*) => {$(
        impl ops::Sub<Complex<$Lhs>> for $Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn sub(self, other: Complex<$Lhs>) -> Self::Output {
                Self::Output::new(self - other.re, other.im)
            }
        }

        impl ops::Sub<&Complex<$Lhs>> for $Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn sub(self, other: &Complex<$Lhs>) -> Self::Output {
                Self::Output::new(self - other.re, other.im)
            }
        }

        impl ops::Sub<Complex<$Lhs>> for &$Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn sub(self, other: Complex<$Lhs>) -> Self::Output {
                Self::Output::new(self - other.re, other.im)
            }
        }

        impl<'a, 'b> ops::Sub<&'a Complex<$Lhs>> for &'b $Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn sub(self, other: &'a Complex<$Lhs>) -> Self::Output {
                Self::Output::new(self - other.re, other.im)
            }
        }
    )*}
}

impl_scalar_complex_sub_ops!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);


macro_rules! impl_scalar_complex_mul_ops {
    ($($Lhs:ty),* $(,)*) => {$(
        impl ops::Mul<Complex<$Lhs>> for $Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn mul(self, other: Complex<$Lhs>) -> Self::Output {
                Self::Output::new(self * other.re, self * other.im)
            }
        }

        impl ops::Mul<&Complex<$Lhs>> for $Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn mul(self, other: &Complex<$Lhs>) -> Self::Output {
                Self::Output::new(self * other.re, self * other.im)
            }
        }

        impl ops::Mul<Complex<$Lhs>> for &$Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn mul(self, other: Complex<$Lhs>) -> Self::Output {
                Self::Output::new(self * other.re, self * other.im)
            }
        }

        impl<'a, 'b> ops::Mul<&'a Complex<$Lhs>> for &'b $Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn mul(self, other: &'a Complex<$Lhs>) -> Self::Output {
                Self::Output::new(self * other.re, self * other.im)
            }
        }
    )*}
}

impl_scalar_complex_mul_ops!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);


macro_rules! impl_scalar_complex_div_ops {
    ($($Lhs:ty),* $(,)*) => {$(
        impl ops::Div<Complex<$Lhs>> for $Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn div(self, other: Complex<$Lhs>) -> Self::Output {
                let denominator = other.re * other.re + other.im * other.im;
                let re = (self * other.re) / denominator;
                let im = -(self * other.im) / denominator;

                Self::Output::new(re, im)
            }
        }

        impl ops::Div<&Complex<$Lhs>> for $Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn div(self, other: &Complex<$Lhs>) -> Self::Output {
                let denominator = other.re * other.re + other.im * other.im;
                let re = (self * other.re) / denominator;
                let im = -(self * other.im) / denominator;
                
                Self::Output::new(re, im)
            }
        }

        impl ops::Div<Complex<$Lhs>> for &$Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn div(self, other: Complex<$Lhs>) -> Self::Output {
                let denominator = other.re * other.re + other.im * other.im;
                let re = (self * other.re) / denominator;
                let im = -(self * other.im) / denominator;
                
                Self::Output::new(re, im)
            }
        }

        impl<'a, 'b> ops::Div<&'a Complex<$Lhs>> for &'b $Lhs {
            type Output = Complex<$Lhs>;

            #[inline]
            fn div(self, other: &'a Complex<$Lhs>) -> Self::Output {
                let denominator = other.re * other.re + other.im * other.im;
                let re = (self * other.re) / denominator;
                let im = -(self * other.im) / denominator;
                
                Self::Output::new(re, im)
            }
        }
    )*}
}

impl_scalar_complex_div_ops!(i8, i16, i32, i64, i128, isize, f32, f64);


impl<S> approx::AbsDiffEq for Complex<S> 
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
        S::abs_diff_eq(&self.re, &other.re, epsilon) &&
        S::abs_diff_eq(&self.im, &other.im, epsilon)
    }
}

impl<S> approx::RelativeEq for Complex<S> 
where 
    S: SimdScalarFloat 
{
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.re, &other.re, epsilon, max_relative) &&
        S::relative_eq(&self.im, &other.im, epsilon, max_relative)
    }
}

impl<S> approx::UlpsEq for Complex<S> 
where 
    S: SimdScalarFloat
{
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.re, &other.re, epsilon, max_ulps) &&
        S::ulps_eq(&self.im, &other.im, epsilon, max_ulps)
    }
}

impl<S> Normed for Complex<S>
where
    S: SimdScalarFloat
{
    type Output = S;

    fn norm_squared(&self) -> Self::Output {
        self.modulus_squared()
    }
    
    fn norm(&self) -> Self::Output {
        self.modulus()
    }

    fn scale(&self, norm: Self::Output) -> Self {
        self * (norm / self.modulus())
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

    fn normalize(&self) -> Self {
        self * (Self::Output::one() / self.modulus())
    }

    fn normalize_mut(&mut self) -> Self::Output {
        let norm = self.modulus();
        *self = self.normalize();

        norm
    }

    fn try_normalize(&self, threshold: Self::Output) -> Option<Self> {
        let norm = self.modulus();
        if norm <= threshold {
            None
        } else {
            Some(self.normalize())
        }
    }
        
    fn try_normalize_mut(&mut self, threshold: Self::Output) -> Option<Self::Output> {
        let norm = self.modulus();
        if norm <= threshold {
            None
        } else {
            Some(self.normalize_mut())
        }
    }

    fn distance_squared(&self, other: &Self) -> Self::Output {
        (self - other).modulus_squared()
    }

    fn distance(&self, other: &Self) -> Self::Output {
        (self - other).modulus()
    }
}

