use crate::base::{
    Magnitude,
    Scalar,
    ScalarSigned,
    ScalarFloat,
};
use crate::angle::{
    Angle,
    Radians,
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
    pub re: S,
    pub im: S,
}

impl<S> Complex<S> {
    /// Construct a new complex number from its real and imaginary components.
    #[inline]
    pub const fn new(re: S, im: S) -> Self {
        Self {
            re: re,
            im: im,
        }
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
        <Self as AsRef<[S; 2]>>::as_ref(self)
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
    /// # use cglinalg::{
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
    /// # use cglinalg::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(1_i32, 2_i32);
    /// 
    /// assert_eq!(z.real(), 1_i32);
    /// ```
    #[inline]
    pub fn real(self) -> S {
        self.re
    }

    /// Get the imaginary part of a complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(1_i32, 2_i32);
    /// 
    /// assert_eq!(z.imaginary(), 2_i32);
    /// ```
    #[inline]
    pub fn imaginary(self) -> S {
        self.im
    }

    /// Map an operation on that acts on the components of a complex number, returning 
    /// a complex number whose coordinates are of the new scalar type.
    ///
    /// # Example
    ///
    /// ```
    /// # use cglinalg::{
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
    S: Scalar
{
    /// Construct a new complex number from its real part.
    /// 
    /// The resulting complex number has a zero imaginary part.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
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
    /// # use cglinalg::{
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
    /// # use cglinalg::{
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
    /// # use cglinalg::{
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
    /// # use cglinalg::{
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
    /// This is a synonym for `identity`.
    #[inline]
    pub fn one() -> Self {
        Self::identity()
    }

    /// Determine whether a complex number is the identity complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
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

    /// Get the multiplicative unit complex number.
    /// 
    /// This is a synonym for the identity complex number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
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
    /// # use cglinalg::{
    /// #     Complex,
    /// # };
    /// #
    /// let i: Complex<i32> = Complex::unit_im();
    /// 
    /// assert_eq!(i.real(), 0);
    /// assert_eq!(i.imaginary(), 1);
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
    /// # use cglinalg::{
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
    /// # use cglinalg::{
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
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(2_i32, 5_i32);
    /// 
    /// assert_eq!(z.magnitude_squared(), 29);
    /// ```
    #[inline]
    pub fn magnitude_squared(self) -> S {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }
}

impl<S> Complex<S>
where
    S: ScalarSigned
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
    /// # use cglinalg::{
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
    S: ScalarFloat
{
    /// Returns `true` if all of the elements of a complex number are finite.
    /// Otherwise, it returns `false`.
    /// 
    /// # Example (Finite Complex Number)
    /// 
    /// ```
    /// # use cglinalg::{
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
    /// # use cglinalg::{
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
    /// # use cglinalg::{
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
    /// # use cglinalg::{
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

    /// Compute the magnitude (modulus, length) of a complex number in Euclidean
    /// space.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Complex,
    /// # };
    /// #
    /// let z = Complex::new(7_f64, 3_f64);
    /// 
    /// assert_eq!(z.magnitude(), f64::sqrt(58_f64));
    /// ```
    #[inline]
    pub fn magnitude(self) -> S {
        self.magnitude_squared().sqrt()
    }

    /// Calculate the principal argument of a complex number.
    /// 
    /// In polar form, a complex number `z` can be written as 
    /// ```text
    /// z := |z| * exp(i * angle) := |z| * (cos(angle) + i * sin(angle))
    /// ```
    /// Consequently there is an ambiguity in choosing the angle for `z` in its 
    /// polar form; two complex numbers in polar form are equal if they have 
    /// identical magnitudes and they differ by a factor of `2 * pi` in their 
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
    /// # use cglinalg::{
    /// #     Complex,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     relative_eq,
    /// # };
    /// #
    /// let pi = core::f64::consts::PI;
    /// let angle = pi / 4_f64;
    /// let z1 = Complex::from_polar_decomposition(2_f64, Radians(angle));
    /// let z2 = Complex::from_polar_decomposition(2_f64, Radians(angle + 2_f64 * pi));
    /// 
    /// assert!(relative_eq!(z1, z2, epsilon = 1e-10));
    /// assert!(relative_eq!(z1.arg(), z2.arg(), epsilon = 1e-10));
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
    /// # use cglinalg::{
    /// #     Complex,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     relative_eq,
    /// # };
    /// #
    /// let pi_over_four = core::f64::consts::FRAC_PI_4;
    /// let expected = Complex::new(2_f64 / f64::sqrt(2_f64), 2_f64 / f64::sqrt(2_f64));
    /// let result = Complex::from_polar_decomposition(2_f64, Radians(pi_over_four));
    /// 
    /// assert!(relative_eq!(result, expected, epsilon = 1e-10));
    /// ``` 
    #[inline]
    pub fn from_polar_decomposition<A: Into<Radians<S>>>(radius: S, angle: A) -> Self {
        let _angle: Radians<S> = angle.into();
        Self::new(radius * _angle.cos(), radius * _angle.sin())
    }

    /// Construct a unit complex number from its polar form.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Complex,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     relative_eq,
    /// # };
    /// #
    /// let pi_over_four = core::f64::consts::FRAC_PI_4;
    /// let expected = Complex::new(1_f64 / f64::sqrt(2_f64), 1_f64 / f64::sqrt(2_f64));
    /// let result = Complex::from_angle(Radians(pi_over_four));
    /// 
    /// assert!(relative_eq!(result, expected, epsilon = 1e-10));
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
    /// # use cglinalg::{
    /// #     Complex,
    /// #     Radians,
    /// # };
    /// #
    /// let z = Complex::from_polar_decomposition(3_f64, Radians(2_f64));
    /// let (radius, angle) = z.polar_decomposition();
    /// 
    /// assert_eq!(radius, 3_f64);
    /// assert_eq!(angle, Radians(2_f64));
    /// ```
    #[inline]
    pub fn polar_decomposition(self) -> (S, Radians<S>) {
        (self.magnitude(), Radians(self.arg()))
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
    /// # use cglinalg::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(2_f64, 5_f64);
    /// let expected = Complex::new(2.09599580151, -7.08554526009);
    /// let result = z.exp();
    /// 
    /// assert!(relative_eq!(result, expected, epsilon = 1e-10));
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
    /// # use cglinalg::{
    /// #     Complex,
    /// #     Radians,
    /// # };
    /// #
    /// let pi = core::f64::consts::PI;
    /// let z = Complex::from_polar_decomposition(
    ///     3_f64, Radians(pi / 6_f64) + Radians((2_f64 * pi))
    /// );
    /// let expected = Complex::new(z.magnitude().ln(), pi / 6_f64);
    /// let result = z.ln();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn ln(self) -> Self {
        let magnitude_self = self.magnitude();
        let arg_self = self.arg();

        Self::new(magnitude_self.ln(), arg_self)
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
    /// where `|z|` is the magnitude of `z`, `t` is the principal argument of `z`, and `n`
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
    /// # use cglinalg::{
    /// #     Complex,
    /// # };
    /// # use approx::{
    /// #     relative_eq,
    /// # };
    /// #
    /// let z = Complex::new(1_f64, 4_f64);
    /// let sqrt_z = z.sqrt();
    ///
    /// assert!(relative_eq!(sqrt_z * sqrt_z, z, epsilon = 1e-10));
    /// 
    /// let minus_sqrt_z = -sqrt_z;
    /// 
    /// assert!(relative_eq!(minus_sqrt_z * minus_sqrt_z, z, epsilon = 1e-10));
    /// ```
    #[inline]
    pub fn sqrt(self) -> Self {
        let two = S::one() + S::one();
        let sqrt_magnitude = self.magnitude().sqrt();
        let angle = self.arg();
        let (sin_angle_over_two, cos_angle_over_two) = (angle / two).sin_cos();

        Self::new(sqrt_magnitude * cos_angle_over_two, sqrt_magnitude * sin_angle_over_two)
    }

    /// Calculate the power of a complex number where the exponent is a floating 
    /// point number.
    /// 
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Complex,
    /// #     Radians,
    /// # };
    /// # use approx::{
    /// #     relative_eq,
    /// # };
    /// #
    /// let pi_over_four = core::f64::consts::FRAC_PI_4;
    /// let z = Complex::from_polar_decomposition(2_f64, Radians(pi_over_four));
    /// let exponent = 5_f64;
    /// let expected = Complex::from_polar_decomposition(32_f64, Radians(exponent * pi_over_four));
    /// let result = z.powf(exponent);
    /// 
    /// assert!(relative_eq!(result, expected, epsilon = 1e-10));
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
    /// # use cglinalg::{
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
        let magnitude_squared = self.magnitude_squared();
        if magnitude_squared.is_zero() {
            None
        } else {
            Some(Self::new(
                 self.re / magnitude_squared,
                -self.im / magnitude_squared
            ))
        }
    }

    /// Determine whether a complex number is invertible.
    /// 
    /// Returns `false` if the magnitude of the complex number is sufficiently
    /// close to zero.
    ///
    /// # Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Complex,
    /// # };
    /// #
    /// let z: Complex<f64> = Complex::unit_im();
    /// 
    /// assert!(z.is_invertible());
    /// ```
    #[inline]
    pub fn is_invertible(&self) -> bool {
        self.is_invertible_eps(S::default_epsilon())
    }

    #[inline]
    fn is_invertible_eps(&self, epsilon: S) -> bool {
        self.magnitude_squared() >= epsilon * epsilon
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
    S: Scalar 
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
    S: Scalar 
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
    S: Scalar 
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
    S: Scalar 
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
    S: Scalar 
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
    S: Scalar 
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut S {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::Range<usize>> for Complex<S> 
where 
    S: Scalar 
{
    #[inline]
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut [S] {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeTo<usize>> for Complex<S> 
where 
    S: Scalar 
{
    #[inline]
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut [S] {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFrom<usize>> for Complex<S> 
where 
    S: Scalar 
{
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut [S] {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> ops::IndexMut<ops::RangeFull> for Complex<S> 
where 
    S: Scalar 
{
    #[inline]
    fn index_mut(&mut self, index: ops::RangeFull) -> &mut [S] {
        let v: &mut [S; 2] = self.as_mut();
        &mut v[index]
    }
}

impl<S> Default for Complex<S>
where
    S: Scalar
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<S> From<S> for Complex<S>
where
    S: Scalar
{
    #[inline]
    fn from(re: S) -> Self {
        Self::new(re, S::zero())
    }
}

impl<S> From<&S> for Complex<S>
where
    S: Scalar
{
    #[inline]
    fn from(re: &S) -> Self {
        Self::new(*re, S::zero())
    }
}

impl<S> From<(S, S)> for Complex<S>
where
    S: Scalar
{
    #[inline]
    fn from(v: (S, S)) -> Self {
        Self::new(v.0, v.1)
    }
}

impl<S> From<[S; 2]> for Complex<S>
where
    S: Scalar
{
    #[inline]
    fn from(v: [S; 2]) -> Self {
        Self::new(v[0], v[1])
    }
}

impl<S> From<&(S, S)> for Complex<S>
where
    S: Scalar
{
    #[inline]
    fn from(v: &(S, S)) -> Self {
        Self::new(v.0, v.1)
    }
}

impl<S> From<&[S; 2]> for Complex<S>
where
    S: Scalar
{
    #[inline]
    fn from(v: &[S; 2]) -> Self {
        Self::new(v[0], v[1])
    }
}

impl<'a, S> From<&'a (S, S)> for &'a Complex<S>
where
    S: Scalar
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
    S: Scalar
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
    S: ScalarSigned
{
    type Output = Complex<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::new(-self.re, -self.im)
    }
}

impl<S> ops::Neg for &Complex<S>
where
    S: ScalarSigned
{
    type Output = Complex<S>;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::Output::new(-self.re, -self.im)
    }
}

impl<S> ops::Add<Complex<S>> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: Complex<S>) -> Self::Output {
        Self::Output::new(self.re + other.re, self.im + other.im)
    }
}

impl<S> ops::Add<&Complex<S>> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: &Complex<S>) -> Self::Output {
        Self::Output::new(self.re + other.re, self.im + other.im)
    }
}

impl<S> ops::Add<Complex<S>> for &Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: Complex<S>) -> Self::Output {
        Self::Output::new(self.re + other.re, self.im + other.im)
    }
}

impl<'a, 'b, S> ops::Add<&'b Complex<S>> for &'a Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: &'b Complex<S>) -> Self::Output {
        Self::Output::new(self.re + other.re, self.im + other.im)
    }
}

impl<S> ops::Add<S> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: S) -> Self::Output {
        Self::Output::new(self.re + other, self.im)
    }
}

impl<S> ops::Add<&S> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: &S) -> Self::Output {
        Self::Output::new(self.re + *other, self.im)
    }
}

impl<S> ops::Add<S> for &Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: S) -> Self::Output {
        Self::Output::new(self.re + other, self.im)
    }
}

impl<'a, 'b, S> ops::Add<&'b S> for &'a Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn add(self, other: &'b S) -> Self::Output {
        Self::Output::new(self.re + *other, self.im)
    }
}

impl<S> ops::Sub<Complex<S>> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: Complex<S>) -> Self::Output {
        Self::Output::new(self.re - other.re, self.im - other.im)
    }
}

impl<S> ops::Sub<&Complex<S>> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: &Complex<S>) -> Self::Output {
        Self::Output::new(self.re - other.re, self.im - other.im)
    }
}

impl<S> ops::Sub<Complex<S>> for &Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: Complex<S>) -> Self::Output {
        Self::Output::new(self.re - other.re, self.im - other.im)
    }
}

impl<'a, 'b, S> ops::Sub<&'b Complex<S>> for &'a Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: &'b Complex<S>) -> Self::Output {
        Self::Output::new(self.re - other.re, self.im - other.im)
    }
}

impl<S> ops::Sub<S> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: S) -> Self::Output {
        Self::Output::new(self.re - other, self.im)
    }
}

impl<S> ops::Sub<&S> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: &S) -> Self::Output {
        Self::Output::new(self.re - *other, self.im)
    }
}

impl<S> ops::Sub<S> for &Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: S) -> Self::Output {
        Self::Output::new(self.re - other, self.im)
    }
}

impl<'a, 'b, S> ops::Sub<&'b S> for &'a Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn sub(self, other: &'b S) -> Self::Output {
        Self::Output::new(self.re - *other, self.im)
    }
}

impl<S> ops::Mul<Complex<S>> for Complex<S>
where
    S: Scalar
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
    S: Scalar
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
    S: Scalar
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
    S: Scalar
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
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Self::Output::new(self.re * other, self.im * other)
    }
}

impl<S> ops::Mul<&S> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn mul(self, other: &S) -> Self::Output {
        Self::Output::new(self.re * *other, self.im * *other)
    }
}

impl<S> ops::Mul<S> for &Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn mul(self, other: S) -> Self::Output {
        Self::Output::new(self.re * other, self.im * other)
    }
}

impl<'a, 'b, S> ops::Mul<&'b S> for &'b Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn mul(self, other: &'b S) -> Self::Output {
        Self::Output::new(self.re * *other, self.im * *other)
    }
}

impl<S> ops::Div<Complex<S>> for Complex<S>
where
    S: Scalar
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
    S: Scalar
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
    S: Scalar
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
    S: Scalar
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
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Self::Output::new(self.re / other, self.im / other)
    }
}

impl<S> ops::Div<&S> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn div(self, other: &S) -> Self::Output {
        Self::Output::new(self.re / *other, self.im / *other)
    }
}

impl<S> ops::Div<S> for &Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn div(self, other: S) -> Self::Output {
        Self::Output::new(self.re / other, self.im / other)
    }
}

impl<'a, 'b, S> ops::Div<&'b S> for &'a Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn div(self, other: &'b S) -> Self::Output {
        Self::Output::new(self.re / *other, self.im / *other)
    }
}

impl<S> ops::Rem<S> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        Self::Output::new(self.re % other, self.im % other)
    }
}

impl<S> ops::Rem<&S> for Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn rem(self, other: &S) -> Self::Output {
        Self::Output::new(self.re % *other, self.im % *other)
    }
}

impl<S> ops::Rem<S> for &Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn rem(self, other: S) -> Self::Output {
        Self::Output::new(self.re % other, self.im % other)
    }
}

impl<'a, 'b, S> ops::Rem<&'b S> for &'a Complex<S>
where
    S: Scalar
{
    type Output = Complex<S>;

    #[inline]
    fn rem(self, other: &'b S) -> Self::Output {
        Self::Output::new(self.re % *other, self.im % *other)
    }
}

impl<S> ops::AddAssign<Complex<S>> for Complex<S>
where
    S: Scalar
{
    fn add_assign(&mut self, other: Complex<S>) {
        self.re += other.re;
        self.im += other.im;
    }
}

impl<S> ops::AddAssign<&Complex<S>> for Complex<S>
where
    S: Scalar
{
    fn add_assign(&mut self, other: &Complex<S>) {
        self.re += other.re;
        self.im += other.im;
    }
}

impl<S> ops::AddAssign<S> for Complex<S>
where
    S: Scalar
{
    fn add_assign(&mut self, other: S) {
        self.re += other;
    }
}

impl<S> ops::AddAssign<&S> for Complex<S>
where
    S: Scalar
{
    fn add_assign(&mut self, other: &S) {
        self.re += *other;
    }
}

impl<S> ops::SubAssign<Complex<S>> for Complex<S>
where
    S: Scalar
{
    fn sub_assign(&mut self, other: Complex<S>) {
        self.re -= other.re;
        self.im -= other.im;
    }
}

impl<S> ops::SubAssign<&Complex<S>> for Complex<S>
where
    S: Scalar
{
    fn sub_assign(&mut self, other: &Complex<S>) {
        self.re -= other.re;
        self.im -= other.im;
    }
}

impl<S> ops::SubAssign<S> for Complex<S>
where
    S: Scalar
{
    fn sub_assign(&mut self, other: S) {
        self.re -= other;
    }
}

impl<S> ops::SubAssign<&S> for Complex<S>
where
    S: Scalar
{
    fn sub_assign(&mut self, other: &S) {
        self.re -= *other;
    }
}

impl<S> ops::MulAssign<Complex<S>> for Complex<S>
where
    S: Scalar
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
    S: Scalar
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
    S: Scalar
{
    fn mul_assign(&mut self, other: S) {
        self.re *= other;
        self.im *= other;
    }
}

impl<S> ops::MulAssign<&S> for Complex<S>
where
    S: Scalar
{
    fn mul_assign(&mut self, other: &S) {
        self.re *= *other;
        self.im *= *other;
    }
}

impl<S> ops::DivAssign<Complex<S>> for Complex<S>
where
    S: Scalar
{
    fn div_assign(&mut self, other: Complex<S>) {
        let a = self.re;
        let magnitude_squared = other.magnitude_squared();

        self.re *= other.re;
        self.re += self.im * other.im;
        self.re /= magnitude_squared;

        self.im *= other.re;
        self.im -= a * other.im;
        self.im /= magnitude_squared;
    }
}

impl<S> ops::DivAssign<&Complex<S>> for Complex<S>
where
    S: Scalar
{
    fn div_assign(&mut self, other: &Complex<S>) {
        let a = self.re;
        let magnitude_squared = other.magnitude_squared();

        self.re *= other.re;
        self.re += self.im * other.im;
        self.re /= magnitude_squared;

        self.im *= other.re;
        self.im -= a * other.im;
        self.im /= magnitude_squared;
    }
}

impl<S> ops::DivAssign<S> for Complex<S>
where
    S: Scalar
{
    fn div_assign(&mut self, other: S) {
        self.re /= other;
        self.im /= other;
    }
}

impl<S> ops::DivAssign<&S> for Complex<S>
where
    S: Scalar
{
    fn div_assign(&mut self, other: &S) {
        self.re /= *other;
        self.im /= *other;
    }
}

impl<S> ops::RemAssign<S> for Complex<S>
where
    S: Scalar
{
    fn rem_assign(&mut self, other: S) {
        self.re %= other;
        self.im %= other;
    }
}

impl<S> ops::RemAssign<&S> for Complex<S>
where
    S: Scalar
{
    fn rem_assign(&mut self, other: &S) {
        self.re %= *other;
        self.im %= *other;
    }
}

macro_rules! impl_scalar_complex_add_ops {
    ($Lhs:ty) => {
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
    }
}

impl_scalar_complex_add_ops!(u8);
impl_scalar_complex_add_ops!(u16);
impl_scalar_complex_add_ops!(u32);
impl_scalar_complex_add_ops!(u64);
impl_scalar_complex_add_ops!(u128);
impl_scalar_complex_add_ops!(usize);
impl_scalar_complex_add_ops!(i8);
impl_scalar_complex_add_ops!(i16);
impl_scalar_complex_add_ops!(i32);
impl_scalar_complex_add_ops!(i64);
impl_scalar_complex_add_ops!(i128);
impl_scalar_complex_add_ops!(isize);
impl_scalar_complex_add_ops!(f32);
impl_scalar_complex_add_ops!(f64);


macro_rules! impl_scalar_complex_sub_ops {
    ($Lhs:ty) => {
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
    }
}

impl_scalar_complex_sub_ops!(u8);
impl_scalar_complex_sub_ops!(u16);
impl_scalar_complex_sub_ops!(u32);
impl_scalar_complex_sub_ops!(u64);
impl_scalar_complex_sub_ops!(u128);
impl_scalar_complex_sub_ops!(usize);
impl_scalar_complex_sub_ops!(i8);
impl_scalar_complex_sub_ops!(i16);
impl_scalar_complex_sub_ops!(i32);
impl_scalar_complex_sub_ops!(i64);
impl_scalar_complex_sub_ops!(i128);
impl_scalar_complex_sub_ops!(isize);
impl_scalar_complex_sub_ops!(f32);
impl_scalar_complex_sub_ops!(f64);


macro_rules! impl_scalar_complex_mul_ops {
    ($Lhs:ty) => {
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
    }
}

impl_scalar_complex_mul_ops!(u8);
impl_scalar_complex_mul_ops!(u16);
impl_scalar_complex_mul_ops!(u32);
impl_scalar_complex_mul_ops!(u64);
impl_scalar_complex_mul_ops!(u128);
impl_scalar_complex_mul_ops!(usize);
impl_scalar_complex_mul_ops!(i8);
impl_scalar_complex_mul_ops!(i16);
impl_scalar_complex_mul_ops!(i32);
impl_scalar_complex_mul_ops!(i64);
impl_scalar_complex_mul_ops!(i128);
impl_scalar_complex_mul_ops!(isize);
impl_scalar_complex_mul_ops!(f32);
impl_scalar_complex_mul_ops!(f64);

macro_rules! impl_scalar_complex_div_ops {
    ($Lhs:ty) => {
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
    }
}

impl_scalar_complex_div_ops!(i8);
impl_scalar_complex_div_ops!(i16);
impl_scalar_complex_div_ops!(i32);
impl_scalar_complex_div_ops!(i64);
impl_scalar_complex_div_ops!(i128);
impl_scalar_complex_div_ops!(isize);
impl_scalar_complex_div_ops!(f32);
impl_scalar_complex_div_ops!(f64);


impl<S> approx::AbsDiffEq for Complex<S> 
where 
    S: ScalarFloat 
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
    S: ScalarFloat 
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
    S: ScalarFloat
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

