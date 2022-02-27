use crate::base::{
    Magnitude,
    Scalar,
    ScalarSigned,
    ScalarFloat,
};
use crate::vector::{
    Vector1,
    Vector2,
    Vector3,
    Vector4,
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


/// A point is a location in a one-dimensional Euclidean space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Point1<S> {
    data: Vector1<S>,
}

impl<S> Point1<S> {
    /// Construct a new point in one-dimensional Euclidean space.
    #[inline]
    pub const fn new(x: S) -> Self {
        Point1 { 
            data: Vector1::new(x), 
        }
    }
}

impl<S> Point1<S> 
where 
    S: Copy
{
    /// Construct a new two-dimensional point from a one-dimensional point by
    /// supplying the y-coordinate.
    /// 
    /// ## Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Point1,
    /// #     Point2, 
    /// # };
    /// #
    /// let point = Point1::new(1_u32);
    /// let expected = Point2::new(1_u32, 2_u32);
    /// let result = point.extend(2_u32);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn extend(self, y: S) -> Point2<S> {
        Point2::new(self.data[0], y)
    }

    /// Construct a new point from a fill value.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1,
    /// # };
    /// 
    /// let fill_value = 1_u32;
    /// let expected = Point1::new(1_u32);
    /// let result = Point1::from_fill(fill_value);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Self {
        Self::new(value)
    }

    /// The length of the the underlying array storing the point components.
    #[inline]
    pub fn len(&self) -> usize {
        1
    }

    /// The shape of the underlying array storing the point components.
    ///
    /// The shape is the equivalent number of columns and rows of the 
    /// array as though it represents a matrix. The order of the descriptions 
    /// of the shape of the array is **(rows, columns)**.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (1, 1)
    }

    /// Get a pointer to the underlying array.
    #[inline]
    pub fn as_ptr(&self) -> *const S {
        self.data.as_ptr()
    }

    /// Get a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        self.data.as_mut_ptr()
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        <Self as AsRef<[S; 1]>>::as_ref(self)
    }

    /// Map an operation on that acts on the coordinates of a point, returning 
    /// a point whose coordinates are of the new scalar type.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1,  
    /// # };
    /// #
    /// let vector: Point1<u32> = Point1::new(1_u32);
    /// let expected: Point1<i32> = Point1::new(2_i32);
    /// let result: Point1<i32> = vector.map(|comp| (comp + 1) as i32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn map<T, F>(self, op: F) -> Point1<T> 
    where 
        F: FnMut(S) -> T 
    {
        Point1 { 
            data: self.data.map(op),
        }
    }
}

impl<S> Point1<S> 
where 
    S: NumCast + Copy 
{
    /// Cast a point of one type of scalars to a point of another type of scalars.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1,   
    /// # };
    /// #
    /// let point: Point1<u32> = Point1::new(1_u32);
    /// let expected: Option<Point1<i32>> = Some(Point1::new(1_i32));
    /// let result = point.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Point1<T>> {
        let x = match num_traits::cast(self.data[0]) {
            Some(value) => value,
            None => return None,
        };

        Some(Point1::new(x))
    }
}

impl<S> Point1<S> 
where 
    S: Scalar
{
    /// Compute the origin of the Euclidean vector space.
    #[inline]
    pub fn origin() -> Self {
        Self::new(S::zero())
    }

    /// Convert a vector to a point. 
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1, 
    /// #     Vector1,
    /// # };
    /// #
    /// let vector = Vector1::new(1_u32);
    /// let expected = Point1::new(1_u32);
    /// let result = Point1::from_vector(vector);
    ///
    /// assert_eq!(result, expected);
    /// ``` 
    #[inline]
    pub fn from_vector(vector: Vector1<S>) -> Self {
        Self {
            data: vector,
        }
    }
    
    /// Convert a point to a vector.
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1, 
    /// #     Vector1,
    /// # };
    /// #
    /// let point = Point1::new(1_u32);
    /// let expected = Vector1::new(1_u32);
    /// let result = point.to_vector();
    ///
    /// assert_eq!(result, expected);
    /// ``` 
    #[inline]
    pub fn to_vector(self) -> Vector1<S> {
        self.data
    }

    /// Compute the dot product (inner product) of two points.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1, 
    /// # };
    /// #
    /// let point1 = Point1::new(1_f64);
    /// let point2 = Point1::new(2_f64);
    /// 
    /// assert_eq!(point1.dot(&point2), 2_f64);
    /// ```
    #[inline]
    pub fn dot(&self, other: &Self) -> S {
        self.data.dot(&other.data)
    }
}

impl<S> fmt::Display for Point1<S> 
where
    S: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Point1 [{}]", self.data[0])
    }
}

impl<S> From<S> for Point1<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: S) -> Self {
        Self::new(v)
    }
}

impl<S> From<[S; 1]> for Point1<S>
where 
    S: Scalar
{
    #[inline]
    fn from(v: [S; 1]) -> Self {
        Self::new(v[0])
    }
}

impl<S> From<&[S; 1]> for Point1<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &[S; 1]) -> Self {
        Self::new(v[0])
    }
}

impl<'a, S> From<&'a [S; 1]> for &'a Point1<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &'a [S; 1]) -> &'a Point1<S> {
        unsafe { 
            &*(v as *const [S; 1] as *const Point1<S>)
        }
    }
}


/// A point is a location in a two-dimensional Euclidean space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Point2<S> {
    data: Vector2<S>,
}

impl<S> Point2<S> {
    /// Construct a new two-dimensional point.
    #[inline]
    pub const fn new(x: S, y: S) -> Self {
        Self { 
            data: Vector2::new(x, y) 
        }
    }
}

impl<S> Point2<S> 
where 
    S: Copy
{
    /// Expand a two-dimensional point to a three-dimensional point using
    /// the supplied z-value.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2,
    /// #     Point3,    
    /// # };
    /// #
    /// let point = Point2::new(1_u32, 2_u32);
    /// let expected = Point3::new(1_u32, 2_u32, 3_u32);
    /// let result = point.extend(3_u32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn extend(self, z: S) -> Point3<S> {
        Point3::new(self.data[0], self.data[1], z)
    }

    /// Contract a two-dimensional point to a one-dimensional point by
    /// removing its **y-component**.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point1,
    /// #     Point2, 
    /// # };
    /// #
    /// let point = Point2::new(1_u32, 2_u32);
    /// let expected = Point1::new(1_u32);
    /// let result = point.contract();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn contract(self) -> Point1<S> {
        Point1::new(self.data[0])
    }

    /// Construct a new point from a fill value.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2, 
    /// # };
    /// #
    /// let fill_value = 2_u32;
    /// let expected = Point2::new(2_u32, 2_u32);
    /// let result = Point2::from_fill(fill_value);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Self {
        Self::new(value, value)
    }

    /// The length of the the underlying array storing the point components.
    #[inline]
    pub fn len(&self) -> usize {
        2
    }

    /// The shape of the underlying array storing the point components.
    ///
    /// The shape is the equivalent number of columns and rows of the 
    /// array as though it represents a matrix. The order of the descriptions 
    /// of the shape of the array is **(rows, columns)**.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (2, 1)
    }

    /// Get a pointer to the underlying array.
    #[inline]
    pub fn as_ptr(&self) -> *const S {
        self.data.as_ptr()
    }

    /// Get a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        self.data.as_mut_ptr()
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        <Self as AsRef<[S; 2]>>::as_ref(self)
    }

    /// Map an operation on that acts on the coordinates of a point, returning 
    /// a point whose coordinates are of the new scalar type.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2,  
    /// # };
    /// #
    /// let vector: Point2<u32> = Point2::new(1_u32, 2_u32);
    /// let expected: Point2<i32> = Point2::new(2_i32, 3_i32);
    /// let result: Point2<i32> = vector.map(|comp| (comp + 1) as i32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn map<T, F>(self, op: F) -> Point2<T> 
    where 
        F: FnMut(S) -> T 
    {
        Point2 {
            data: self.data.map(op),
        }
    }
}

impl<S> Point2<S> 
where 
    S: NumCast + Copy
{
    /// Cast a point of one type of scalars to a point of another type of scalars.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2,   
    /// # };
    /// #
    /// let point: Point2<u32> = Point2::new(1_u32, 2_u32);
    /// let expected: Option<Point2<i32>> = Some(Point2::new(1_i32, 2_i32));
    /// let result = point.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Point2<T>> {
        let x = match num_traits::cast(self.data[0]) {
            Some(value) => value,
            None => return None,
        };
        let y = match num_traits::cast(self.data[1]) {
            Some(value) => value,
            None => return None,
        };

        Some(Point2::new(x, y))
    }
}

impl<S> Point2<S> 
where 
    S: Scalar
{
    /// Convert a homogeneous vector into a point.
    ///
    /// ## Example
    /// 
    /// ```
    /// # use cglinalg::{
    /// #     Point2, 
    /// #     Vector3,
    /// # };
    /// #
    /// let vector = Vector3::new(3_f64, 6_f64, 3_f64);
    /// let expected = Some(Point2::new(1_f64, 2_f64));
    /// let result = Point2::from_homogeneous(vector);
    ///
    /// assert_eq!(result, expected);
    ///
    /// let vector_z_zero = Vector3::new(3_f64, 6_f64, 0_f64);
    /// let result = Point2::from_homogeneous(vector_z_zero);
    ///
    /// assert!(result.is_none());
    /// ```
    #[inline]
    pub fn from_homogeneous(vector: Vector3<S>) -> Option<Self> {
        if !vector.z.is_zero() {
            Some(Point2::new(vector.x / vector.z, vector.y / vector.z))
        } else {
            None
        }
    }

    /// Convert a point to a vector in homogeneous coordinates.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2,
    /// #     Vector3,
    /// # };
    /// #
    /// let point = Point2::new(1_f64, 2_f64);
    /// let expected = Vector3::new(1_f64, 2_f64, 1_f64);
    /// let result = point.to_homogeneous();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_homogeneous(self) -> Vector3<S> {
        self.data.extend(S::one())
    }

    /// Compute the origin of the Euclidean vector space.
    #[inline]
    pub fn origin() -> Self {
        Self::new(S::zero(), S::zero())
    }

    /// Convert a vector to a point. 
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2,
    /// #     Vector2, 
    /// # };
    /// #
    /// let vector = Vector2::new(1_u32, 2_u32);
    /// let expected = Point2::new(1_u32, 2_u32);
    /// let result = Point2::from_vector(vector);
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_vector(vector: Vector2<S>) -> Self {
        Self {
            data: vector,
        }
    }

    /// Convert a point to a vector.
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2,
    /// #     Vector2, 
    /// # };
    /// #
    /// let point = Point2::new(1_u32, 2_u32);
    /// let expected = Vector2::new(1_u32, 2_u32);
    /// let result = point.to_vector();
    /// 
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_vector(self) -> Vector2<S> {
        self.data
    }

    /// Compute the dot product (inner product) of two points.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2, 
    /// # };
    /// #
    /// let point1 = Point2::new(1_f64, 2_f64);
    /// let point2 = Point2::new(3_f64, 4_f64);
    /// 
    /// assert_eq!(point1.dot(&point2), 11_f64);
    /// ```
    #[inline]
    pub fn dot(&self, other: &Self) -> S {
        self.data.dot(&other.data)
    }
}

impl<S> fmt::Display for Point2<S> 
where 
    S: fmt::Display 
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Point2 [{}, {}]", self.data[0], self.data[1])
    }
}

impl<S> From<(S, S)> for Point2<S> 
where 
    S: Scalar
{
    #[inline]
    fn from((x, y): (S, S)) -> Self {
        Self::new(x, y)
    }
}

impl<S> From<[S; 2]> for Point2<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: [S; 2]) -> Self {
        Self::new(v[0], v[1])
    }
}

impl<S> From<&[S; 2]> for Point2<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &[S; 2]) -> Self {
        Self::new(v[0], v[1])
    }
}

impl<'a, S> From<&'a [S; 2]> for &'a Point2<S> 
where 
    S: Scalar 
{
    #[inline]
    fn from(v: &'a [S; 2]) -> &'a Point2<S> {
        unsafe { 
            &*(v as *const [S; 2] as *const Point2<S>)
        }
    }
}


/// A representation of three-dimensional points in a Euclidean space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Point3<S> {
    data: Vector3<S>,
}

impl<S> Point3<S> {
    /// Construct a new point in three-dimensional Euclidean space.
    #[inline]
    pub const fn new(x: S, y: S, z: S) -> Self {
        Self { 
            data: Vector3::new(x, y, z),
        }
    }
}

impl<S> Point3<S> 
where 
    S: Copy
{
    /// Contract a three-dimensional point, removing its **z-component**.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point2,
    /// #     Point3, 
    /// # };
    /// #
    /// let point = Point3::new(1_u32, 2_u32, 3_u32);
    /// let expected = Point2::new(1_u32, 2_u32);
    /// let result = point.contract();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn contract(self) -> Point2<S> {
        Point2::new(self.data[0], self.data[1])
    }

    /// Construct a new point from a fill value.
    /// 
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3, 
    /// # };
    /// #
    /// let fill_value = 3_u32;
    /// let expected = Point3::new(3_u32, 3_u32, 3_u32);
    /// let result = Point3::from_fill(fill_value);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_fill(value: S) -> Self {
        Self::new(value, value, value)
    }

    /// The length of the the underlying array storing the point components.
    #[inline]
    pub fn len(&self) -> usize {
        3
    }

    /// The shape of the underlying array storing the point components.
    ///
    /// The shape is the equivalent number of columns and rows of the 
    /// array as though it represents a matrix. The order of the descriptions 
    /// of the shape of the array is **(rows, columns)**.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (3, 1)
    }

    /// Get a pointer to the underlying array.
    #[inline]
    pub fn as_ptr(&self) -> *const S {
        self.data.as_ptr()
    }

    /// Get a mutable pointer to the underlying array.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S {
        self.data.as_mut_ptr()
    }

    /// Get a slice of the underlying elements of the data type.
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        <Self as AsRef<[S; 3]>>::as_ref(self)
    }

    /// Map an operation on that acts on the coordinates of a point, returning 
    /// a point whose coordinates are of the new scalar type.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,  
    /// # };
    /// #
    /// let vector: Point3<u32> = Point3::new(1_u32, 2_u32, 3_u32);
    /// let expected: Point3<i32> = Point3::new(2_i32, 3_i32, 4_i32);
    /// let result: Point3<i32> = vector.map(|comp| (comp + 1) as i32);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn map<T, F>(self, op: F) -> Point3<T> 
    where 
        F: FnMut(S) -> T
    {
        Point3 {
            data: self.data.map(op),
        }
    }
}

impl<S> Point3<S> 
where 
    S: NumCast + Copy
{
    /// Cast a point from one type of scalars to another type of scalars.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,   
    /// # };
    /// #
    /// let point: Point3<u32> = Point3::new(1_u32, 2_u32, 3_u32);
    /// let expected: Option<Point3<i32>> = Some(Point3::new(1_i32, 2_i32, 3_i32));
    /// let result = point.cast::<i32>();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Point3<T>> {
        let x = match num_traits::cast(self.data[0]) {
            Some(value) => value,
            None => return None,
        };
        let y = match num_traits::cast(self.data[1]) {
            Some(value) => value,
            None => return None,
        };
        let z = match num_traits::cast(self.data[2]) {
            Some(value) => value,
            None => return None,
        };

        Some(Point3::new(x, y, z))
    }
}

impl<S> Point3<S> 
where 
    S: Scalar
{
    /// Convert a vector in homogeneous coordinates into a point.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let vector = Vector4::new(5_f64, 10_f64, 15_f64, 5_f64);
    /// let expected = Some(Point3::new(1_f64, 2_f64, 3_f64));
    /// let result = Point3::from_homogeneous(vector);
    ///
    /// assert!(result.is_some());
    /// assert_eq!(result, expected);
    ///
    /// let vector_w_zero = Vector4::new(5_f64, 10_f64, 15_f64, 0_f64);
    /// let result = Point3::from_homogeneous(vector_w_zero);
    ///
    /// assert!(result.is_none());
    /// ```
    #[inline]
    pub fn from_homogeneous(vector: Vector4<S>) -> Option<Self> {
        if !vector.w.is_zero() {
            Some(Self::new(
                vector.x / vector.w, 
                vector.y / vector.w, 
                vector.z / vector.w
            ))
        } else {
            None
        }
    }

    /// Convert a point to a vector in homogeneous coordinates.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,
    /// #     Vector4, 
    /// # };
    /// #
    /// let point = Point3::new(1_u32, 2_u32, 3_u32);
    /// let expected = Vector4::new(1_u32, 2_u32, 3_u32, 1_u32);
    /// let result = point.to_homogeneous();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_homogeneous(self) -> Vector4<S> {
        self.data.extend(S::one())
    }

    /// Compute the origin of the Euclidean vector space.
    #[inline]
    pub fn origin() -> Self {
        Self::new(S::zero(), S::zero(), S::zero())
    }

    /// Convert a vector to a point. 
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// #
    /// let vector = Vector3::new(1_u32, 2_u32, 3_u32);
    /// let expected = Point3::new(1_u32, 2_u32, 3_u32);
    /// let result = Point3::from_vector(vector);
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn from_vector(vector: Vector3<S>) -> Self {
        Self {
            data: vector,
        }
    }

    /// Convert a point to a vector.
    /// 
    /// Points are locations in Euclidean space, whereas vectors
    /// are displacements relative to the origin in Euclidean space.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3,
    /// #     Vector3,
    /// # };
    /// #
    /// let point = Point3::new(1_u32, 2_u32, 3_u32);
    /// let expected = Vector3::new(1_u32, 2_u32, 3_u32);
    /// let result = point.to_vector();
    ///
    /// assert_eq!(result, expected);
    /// ```
    #[inline]
    pub fn to_vector(self) -> Vector3<S> {
        self.data
    }

    /// Compute the dot product (inner product) of two points.
    ///
    /// ## Example
    ///
    /// ```
    /// # use cglinalg::{
    /// #     Point3, 
    /// # };
    /// #
    /// let point1 = Point3::new(1_f64, 2_f64, 3_f64);
    /// let point2 = Point3::new(4_f64, 5_f64, 6_f64);
    /// 
    /// assert_eq!(point1.dot(&point2), 32_f64);
    /// ```
    #[inline]
    pub fn dot(&self, other: &Self) -> S {
        self.data.dot(&other.data)
    }
}

impl<S> fmt::Display for Point3<S> 
where 
    S: fmt::Display
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter, 
            "Point3 [{}, {}, {}]", 
            self.data[0], self.data[1], self.data[2]
        )
    }
}

impl<S> From<(S, S, S)> for Point3<S> 
where 
    S: Scalar
{
    #[inline]
    fn from((x, y, z): (S, S, S)) -> Self {
        Self::new(x, y, z)
    }
}

impl<S> From<[S; 3]> for Point3<S> 
where 
    S: Scalar 
{
    #[inline]
    fn from(v: [S; 3]) -> Self {
        Point3::new(v[0], v[1], v[2])
    }
}

impl<'a, S> From<&'a [S; 3]> for &'a Point3<S> 
where 
    S: Scalar 
{
    #[inline]
    fn from(v: &'a [S; 3]) -> &'a Point3<S> {
        unsafe { 
            &*(v as *const [S; 3] as *const Point3<S>)
        }
    }
}

impl<'a, S> From<&'a (S, S, S)> for &'a Point3<S> 
where 
    S: Scalar
{
    #[inline]
    fn from(v: &'a (S, S, S)) -> &'a Point3<S> {
        unsafe { 
            &*(v as *const (S, S, S) as *const Point3<S>)
        }
    }
}


impl_coords!(X, { x });
impl_coords_deref!(Point1, X);

impl_coords!(XY, { x, y });
impl_coords_deref!(Point2, XY);

impl_coords!(XYZ, { x, y, z });
impl_coords_deref!(Point3, XYZ);


macro_rules! impl_as_ref_ops {
    ($PointType:ty, $RefType:ty) => {
        impl<S> AsRef<$RefType> for $PointType {
            #[inline]
            fn as_ref(&self) -> &$RefType {
                unsafe {
                    &*(self as *const $PointType as *const $RefType)
                }
            }
        }

        impl<S> AsMut<$RefType> for $PointType {
            #[inline]
            fn as_mut(&mut self) -> &mut $RefType {
                unsafe {
                    &mut *(self as *mut $PointType as *mut $RefType)
                }
            }
        }
    }
}

impl_as_ref_ops!(Point1<S>, S);
impl_as_ref_ops!(Point1<S>, (S,));
impl_as_ref_ops!(Point1<S>, [S; 1]);

impl_as_ref_ops!(Point2<S>, (S, S));
impl_as_ref_ops!(Point2<S>, [S; 2]);

impl_as_ref_ops!(Point3<S>, (S, S, S));
impl_as_ref_ops!(Point3<S>, [S; 3]);


macro_rules! impl_point_index_ops {
    ($T:ty, $n:expr, $IndexType:ty, $Output:ty) => {
        impl<S> ops::Index<$IndexType> for $T {
            type Output = $Output;

            #[inline]
            fn index(&self, index: $IndexType) -> &Self::Output {
                let v: &[S; $n] = self.as_ref();
                &v[index]
            }
        }

        impl<S> ops::IndexMut<$IndexType> for $T {
            #[inline]
            fn index_mut(&mut self, index: $IndexType) -> &mut Self::Output {
                let v: &mut [S; $n] = self.as_mut();
                &mut v[index]
            }
        }
    }
}

impl_point_index_ops!(Point1<S>, 1, usize, S);
impl_point_index_ops!(Point1<S>, 1, ops::Range<usize>, [S]);
impl_point_index_ops!(Point1<S>, 1, ops::RangeTo<usize>, [S]);
impl_point_index_ops!(Point1<S>, 1, ops::RangeFrom<usize>, [S]);
impl_point_index_ops!(Point1<S>, 1, ops::RangeFull, [S]);

impl_point_index_ops!(Point2<S>, 2, usize, S);
impl_point_index_ops!(Point2<S>, 2, ops::Range<usize>, [S]);
impl_point_index_ops!(Point2<S>, 2, ops::RangeTo<usize>, [S]);
impl_point_index_ops!(Point2<S>, 2, ops::RangeFrom<usize>, [S]);
impl_point_index_ops!(Point2<S>, 2, ops::RangeFull, [S]);

impl_point_index_ops!(Point3<S>, 3, usize, S);
impl_point_index_ops!(Point3<S>, 3, ops::Range<usize>, [S]);
impl_point_index_ops!(Point3<S>, 3, ops::RangeTo<usize>, [S]);
impl_point_index_ops!(Point3<S>, 3, ops::RangeFrom<usize>, [S]);
impl_point_index_ops!(Point3<S>, 3, ops::RangeFull, [S]);


macro_rules! impl_point_vector_binary_ops {
    ($OpType:ident, $op:ident, $T1:ty, $T2:ty, $Output:ty, { $($index:expr),* }) => {
        impl<S> ops::$OpType<$T2> for $T1 where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: $T2) -> Self::Output {
                Self::Output::new( 
                    $( self[$index].$op(other[$index]) ),* 
                )
            }
        }

        impl<S> ops::$OpType<&$T2> for $T1 where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: &$T2) -> Self::Output {
                Self::Output::new( 
                    $( self[$index].$op(other[$index]) ),* 
                )
            }
        }

        impl<S> ops::$OpType<$T2> for &$T1 where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: $T2) -> Self::Output {
                Self::Output::new( 
                    $( self[$index].$op(other[$index]) ),* 
                )
            }
        }

        impl<'a, 'b, S> ops::$OpType<&'a $T2> for &'b $T1 where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: &'a $T2) -> Self::Output {
                Self::Output::new( 
                    $( self[$index].$op(other[$index]) ),* 
                )
            }
        }
    }
}

impl_point_vector_binary_ops!(Add, add, Point1<S>, Vector1<S>, Point1<S>, { 0 });
impl_point_vector_binary_ops!(Sub, sub, Point1<S>, Vector1<S>, Point1<S>, { 0 });
impl_point_vector_binary_ops!(Sub, sub, Point1<S>, Point1<S>, Vector1<S>, { 0 });

impl_point_vector_binary_ops!(Add, add, Point2<S>, Vector2<S>, Point2<S>, { 0, 1 });
impl_point_vector_binary_ops!(Sub, sub, Point2<S>, Vector2<S>, Point2<S>, { 0, 1 });
impl_point_vector_binary_ops!(Sub, sub, Point2<S>, Point2<S>, Vector2<S>, { 0, 1 });

impl_point_vector_binary_ops!(Add, add, Point3<S>, Vector3<S>, Point3<S>, { 0, 1, 2 });
impl_point_vector_binary_ops!(Sub, sub, Point3<S>, Vector3<S>, Point3<S>, { 0, 1, 2 });
impl_point_vector_binary_ops!(Sub, sub, Point3<S>, Point3<S>, Vector3<S>, { 0, 1, 2 });


macro_rules! impl_point_scalar_binary_ops {
    ($OpType:ident, $op:ident, $T:ty, $Output:ty, { $($index:expr),* }) => {
        impl<S> ops::$OpType<S> for $T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: S) -> Self::Output {
                Self::Output::new( 
                    $( self[$index].$op(other) ),* 
                )
            }
        }

        impl<S> ops::$OpType<S> for &$T where S: Scalar {
            type Output = $Output;

            #[inline]
            fn $op(self, other: S) -> Self::Output {
                Self::Output::new( 
                    $( self[$index].$op(other) ),* 
                )
            }
        }
    }
}

impl_point_scalar_binary_ops!(Mul, mul, Point1<S>, Point1<S>, { 0 });
impl_point_scalar_binary_ops!(Div, div, Point1<S>, Point1<S>, { 0 });
impl_point_scalar_binary_ops!(Rem, rem, Point1<S>, Point1<S>, { 0 });

impl_point_scalar_binary_ops!(Mul, mul, Point2<S>, Point2<S>, { 0, 1 });
impl_point_scalar_binary_ops!(Div, div, Point2<S>, Point2<S>, { 0, 1 });
impl_point_scalar_binary_ops!(Rem, rem, Point2<S>, Point2<S>, { 0, 1 });

impl_point_scalar_binary_ops!(Mul, mul, Point3<S>, Point3<S>, { 0, 1, 2 });
impl_point_scalar_binary_ops!(Div, div, Point3<S>, Point3<S>, { 0, 1, 2 });
impl_point_scalar_binary_ops!(Rem, rem, Point3<S>, Point3<S>, { 0, 1, 2 });


macro_rules! impl_scalar_point_mul_ops {
    ($Lhs:ty, $Rhs:ty, $Output:ty, { $($index:expr),* }) => {
        impl ops::Mul<$Rhs> for $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                Self::Output::new( $(self * other[$index]),*)
            }
        }

        impl<'a> ops::Mul<$Rhs> for &'a $Lhs {
            type Output = $Output;

            #[inline]
            fn mul(self, other: $Rhs) -> $Output {
                Self::Output::new( $(self * other[$index]),*)
            }
        }
    }
}

impl_scalar_point_mul_ops!(u8,    Point1<u8>,    Point1<u8>,    { 0 });
impl_scalar_point_mul_ops!(u16,   Point1<u16>,   Point1<u16>,   { 0 });
impl_scalar_point_mul_ops!(u32,   Point1<u32>,   Point1<u32>,   { 0 });
impl_scalar_point_mul_ops!(u64,   Point1<u64>,   Point1<u64>,   { 0 });
impl_scalar_point_mul_ops!(u128,  Point1<u128>,  Point1<u128>,  { 0 });
impl_scalar_point_mul_ops!(usize, Point1<usize>, Point1<usize>, { 0 });
impl_scalar_point_mul_ops!(i8,    Point1<i8>,    Point1<i8>,    { 0 });
impl_scalar_point_mul_ops!(i16,   Point1<i16>,   Point1<i16>,   { 0 });
impl_scalar_point_mul_ops!(i32,   Point1<i32>,   Point1<i32>,   { 0 });
impl_scalar_point_mul_ops!(i64,   Point1<i64>,   Point1<i64>,   { 0 });
impl_scalar_point_mul_ops!(i128,  Point1<i128>,  Point1<i128>,  { 0 });
impl_scalar_point_mul_ops!(isize, Point1<isize>, Point1<isize>, { 0 });
impl_scalar_point_mul_ops!(f32,   Point1<f32>,   Point1<f32>,   { 0 });
impl_scalar_point_mul_ops!(f64,   Point1<f64>,   Point1<f64>,   { 0 });

impl_scalar_point_mul_ops!(u8,    Point2<u8>,    Point2<u8>,    { 0, 1 });
impl_scalar_point_mul_ops!(u16,   Point2<u16>,   Point2<u16>,   { 0, 1 });
impl_scalar_point_mul_ops!(u32,   Point2<u32>,   Point2<u32>,   { 0, 1 });
impl_scalar_point_mul_ops!(u64,   Point2<u64>,   Point2<u64>,   { 0, 1 });
impl_scalar_point_mul_ops!(u128,  Point2<u128>,  Point2<u128>,  { 0, 1 });
impl_scalar_point_mul_ops!(usize, Point2<usize>, Point2<usize>, { 0, 1 });
impl_scalar_point_mul_ops!(i8,    Point2<i8>,    Point2<i8>,    { 0, 1 });
impl_scalar_point_mul_ops!(i16,   Point2<i16>,   Point2<i16>,   { 0, 1 });
impl_scalar_point_mul_ops!(i32,   Point2<i32>,   Point2<i32>,   { 0, 1 });
impl_scalar_point_mul_ops!(i64,   Point2<i64>,   Point2<i64>,   { 0, 1 });
impl_scalar_point_mul_ops!(i128,  Point2<i128>,  Point2<i128>,  { 0, 1 });
impl_scalar_point_mul_ops!(isize, Point2<isize>, Point2<isize>, { 0, 1 });
impl_scalar_point_mul_ops!(f32,   Point2<f32>,   Point2<f32>,   { 0, 1 });
impl_scalar_point_mul_ops!(f64,   Point2<f64>,   Point2<f64>,   { 0, 1 });

impl_scalar_point_mul_ops!(u8,    Point3<u8>,    Point3<u8>,    { 0, 1, 2 });
impl_scalar_point_mul_ops!(u16,   Point3<u16>,   Point3<u16>,   { 0, 1, 2 });
impl_scalar_point_mul_ops!(u32,   Point3<u32>,   Point3<u32>,   { 0, 1, 2 });
impl_scalar_point_mul_ops!(u64,   Point3<u64>,   Point3<u64>,   { 0, 1, 2 });
impl_scalar_point_mul_ops!(u128,  Point3<u128>,  Point3<u128>,  { 0, 1, 2 });
impl_scalar_point_mul_ops!(usize, Point3<usize>, Point3<usize>, { 0, 1, 2 });
impl_scalar_point_mul_ops!(i8,    Point3<i8>,    Point3<i8>,    { 0, 1, 2 });
impl_scalar_point_mul_ops!(i16,   Point3<i16>,   Point3<i16>,   { 0, 1, 2 });
impl_scalar_point_mul_ops!(i32,   Point3<i32>,   Point3<i32>,   { 0, 1, 2 });
impl_scalar_point_mul_ops!(i64,   Point3<i64>,   Point3<i64>,   { 0, 1, 2 });
impl_scalar_point_mul_ops!(i128,  Point3<i128>,  Point3<i128>,  { 0, 1, 2 });
impl_scalar_point_mul_ops!(isize, Point3<isize>, Point3<isize>, { 0, 1, 2 });
impl_scalar_point_mul_ops!(f32,   Point3<f32>,   Point3<f32>,   { 0, 1, 2 });
impl_scalar_point_mul_ops!(f64,   Point3<f64>,   Point3<f64>,   { 0, 1, 2 });


macro_rules! impl_point_unary_ops {
    ($OpType:ident, $op:ident, $T:ty, $Output:ty, { $($index:expr),* }) => {
        impl<S> ops::$OpType for $T where S: ScalarSigned {
            type Output = $Output;

            #[inline]
            fn $op(self) -> Self::Output {
                Self::Output::new( 
                    $( self[$index].$op() ),* 
                )
            }
        }

        impl<S> ops::$OpType for &$T where S: ScalarSigned {
            type Output = $Output;

            #[inline]
            fn $op(self) -> Self::Output {
                Self::Output::new( 
                    $( self[$index].$op() ),* 
                )
            }
        }
    }
}

impl_point_unary_ops!(Neg, neg, Point1<S>, Point1<S>, { 0 });
impl_point_unary_ops!(Neg, neg, Point2<S>, Point2<S>, { 0, 1 });
impl_point_unary_ops!(Neg, neg, Point3<S>, Point3<S>, { 0, 1, 2 });


macro_rules! impl_point_binary_assign_ops {
    ($PointType:ty, $VectorType:ty, { $($index:expr),* }) => {
        impl<S> ops::AddAssign<$VectorType> for $PointType where S: Scalar {
            #[inline]
            fn add_assign(&mut self, other: $VectorType) {
                $(self[$index] += other[$index]);*
            }
        }

        impl<S> ops::AddAssign<&$VectorType> for $PointType where S: Scalar {
            #[inline]
            fn add_assign(&mut self, other: &$VectorType) {
                $(self[$index] += other[$index]);*
            }
        }

        impl<S> ops::SubAssign<$VectorType> for $PointType where S: Scalar {
            #[inline]
            fn sub_assign(&mut self, other: $VectorType) {
                $(self[$index] -= other[$index]);*
            }
        }

        impl<S> ops::SubAssign<&$VectorType> for $PointType where S: Scalar {
            #[inline]
            fn sub_assign(&mut self, other: &$VectorType) {
                $(self[$index] -= other[$index]);*
            }
        }

        impl<S> ops::MulAssign<S> for $PointType where S: Scalar {
            #[inline]
            fn mul_assign(&mut self, other: S) {
                $(self[$index] *= other);*
            }
        }
        
        impl<S> ops::DivAssign<S> for $PointType where S: Scalar {
            #[inline]
            fn div_assign(&mut self, other: S) {
                $(self[$index] /= other);*
            }
        }
        
        impl<S> ops::RemAssign<S> for $PointType where S: Scalar {
            #[inline]
            fn rem_assign(&mut self, other: S) {
                $(self[$index] %= other);*
            }
        }
    }
}

impl_point_binary_assign_ops!(Point1<S>, Vector1<S>, { 0 });
impl_point_binary_assign_ops!(Point2<S>, Vector2<S>, { 0, 1 });
impl_point_binary_assign_ops!(Point3<S>, Vector3<S>, { 0, 1, 2 });


macro_rules! impl_magnitude {
    ($PointN:ident) => {
        impl<S> Magnitude for $PointN<S> where S: ScalarFloat {
            type Output = S;

            #[inline]
            fn magnitude_squared(&self) -> Self::Output {
                self.dot(self)
            }
        
            #[inline]
            fn magnitude(&self) -> Self::Output {
                self.magnitude_squared().sqrt()
            }
            
            #[inline]
            fn normalize(&self) -> Self {
                self / self.magnitude()
            }
            
            #[inline]
            fn normalize_to(&self, magnitude: Self::Output) -> Self {
                self * (magnitude / self.magnitude())
            }
        
            #[inline]
            fn try_normalize(&self, threshold: Self::Output) -> Option<Self> {
                let magnitude = self.magnitude();
        
                if magnitude <= threshold {
                    None
                } else {
                    Some(self.normalize())
                }
            }
        
            #[inline]
            fn distance_squared(&self, other: &$PointN<S>) -> Self::Output {
                (self - other).magnitude_squared()
            }
        
            #[inline]
            fn distance(&self, other: &Self) -> Self::Output {
                self.distance_squared(other).sqrt()
            }
        }
    }
}

impl_magnitude!(Point1);
impl_magnitude!(Point2);
impl_magnitude!(Point3);


macro_rules! impl_approx_eq_ops {
    ($PointN:ident, $VectorN:ident) => {
        impl<S> approx::AbsDiffEq for $PointN<S> where S: ScalarFloat {
            type Epsilon = <S as approx::AbsDiffEq>::Epsilon;
        
            #[inline]
            fn default_epsilon() -> Self::Epsilon {
                S::default_epsilon()
            }
        
            #[inline]
            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                $VectorN::abs_diff_eq(&self.data, &other.data, epsilon)
            }
        }
        
        impl<S> approx::RelativeEq for $PointN<S> where S: ScalarFloat {
            #[inline]
            fn default_max_relative() -> S::Epsilon {
                S::default_max_relative()
            }
        
            #[inline]
            fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
                $VectorN::relative_eq(&self.data, &other.data, epsilon, max_relative)
            }
        }
        
        impl<S> approx::UlpsEq for $PointN<S> where S: ScalarFloat {
            #[inline]
            fn default_max_ulps() -> u32 {
                S::default_max_ulps()
            }
        
            #[inline]
            fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
                $VectorN::ulps_eq(&self.data, &other.data, epsilon, max_ulps)
            }
        }
    }
}

impl_approx_eq_ops!(Point1, Vector1);
impl_approx_eq_ops!(Point2, Vector2);
impl_approx_eq_ops!(Point3, Vector3);


macro_rules! impl_swizzle {
    ($name:ident() => $PointN:ident => $Output:ident { $($i:expr),+ }) => {
        impl<S> $PointN<S> where S: Copy {
            /// Construct a new point from the components of the input point.
            #[inline]
            pub fn $name(&self) -> $Output<S> {
                $Output::new(
                    $(self.data[$i]),*
                )
            }
        }
    }
}

impl_swizzle!(x() => Point1 => Point1 { 0 });

impl_swizzle!(x() => Point2 => Point1 { 0 });
impl_swizzle!(y() => Point2 => Point1 { 1 });

impl_swizzle!(xx() => Point2 => Point2 { 0, 0 });
impl_swizzle!(xy() => Point2 => Point2 { 0, 1 });
impl_swizzle!(yx() => Point2 => Point2 { 1, 0 });
impl_swizzle!(yy() => Point2 => Point2 { 1, 1 });

impl_swizzle!(x() => Point3 => Point1 { 0 });
impl_swizzle!(y() => Point3 => Point1 { 1 });
impl_swizzle!(z() => Point3 => Point1 { 2 });

impl_swizzle!(xx() => Point3 => Point2 { 0, 0 });
impl_swizzle!(xy() => Point3 => Point2 { 0, 1 });
impl_swizzle!(xz() => Point3 => Point2 { 0, 2 });
impl_swizzle!(yx() => Point3 => Point2 { 1, 0 });
impl_swizzle!(yy() => Point3 => Point2 { 1, 1 });
impl_swizzle!(yz() => Point3 => Point2 { 1, 2 });
impl_swizzle!(zx() => Point3 => Point2 { 2, 0 });
impl_swizzle!(zy() => Point3 => Point2 { 2, 1 });
impl_swizzle!(zz() => Point3 => Point2 { 2, 2 });

impl_swizzle!(xxx() => Point3 => Point3 { 0, 0, 0 });
impl_swizzle!(xxy() => Point3 => Point3 { 0, 0, 1 });
impl_swizzle!(xxz() => Point3 => Point3 { 0, 0, 2 });
impl_swizzle!(xyx() => Point3 => Point3 { 0, 1, 0 });
impl_swizzle!(xyy() => Point3 => Point3 { 0, 1, 1 });
impl_swizzle!(xyz() => Point3 => Point3 { 0, 1, 2 });
impl_swizzle!(xzx() => Point3 => Point3 { 0, 2, 0 });
impl_swizzle!(xzy() => Point3 => Point3 { 0, 2, 1 });
impl_swizzle!(xzz() => Point3 => Point3 { 0, 2, 2 });
impl_swizzle!(yxx() => Point3 => Point3 { 1, 0, 0 });
impl_swizzle!(yxy() => Point3 => Point3 { 1, 0, 1 });
impl_swizzle!(yxz() => Point3 => Point3 { 1, 0, 2 });
impl_swizzle!(yyx() => Point3 => Point3 { 1, 1, 0 });
impl_swizzle!(yyy() => Point3 => Point3 { 1, 1, 1 });
impl_swizzle!(yyz() => Point3 => Point3 { 1, 1, 2 });
impl_swizzle!(yzx() => Point3 => Point3 { 1, 2, 0 });
impl_swizzle!(yzy() => Point3 => Point3 { 1, 2, 1 });
impl_swizzle!(yzz() => Point3 => Point3 { 1, 2, 2 });
impl_swizzle!(zxx() => Point3 => Point3 { 2, 0, 0 });
impl_swizzle!(zxy() => Point3 => Point3 { 2, 0, 1 });
impl_swizzle!(zxz() => Point3 => Point3 { 2, 0, 2 });
impl_swizzle!(zyx() => Point3 => Point3 { 2, 1, 0 });
impl_swizzle!(zyy() => Point3 => Point3 { 2, 1, 1 });
impl_swizzle!(zyz() => Point3 => Point3 { 2, 1, 2 });
impl_swizzle!(zzx() => Point3 => Point3 { 2, 2, 0 });
impl_swizzle!(zzy() => Point3 => Point3 { 2, 2, 1 });
impl_swizzle!(zzz() => Point3 => Point3 { 2, 2, 2 });

