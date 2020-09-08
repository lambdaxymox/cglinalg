use num_traits::NumCast;


#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(C)]
pub struct Point1<S> {
    pub x: S,
}

impl<S> Point1<S> {
    /// Construct a new point.
    pub const fn new(x: S) -> Point1<S> {
        Point1 { x: x }
    }

    /// Map an operation on the elements of a point, returning a point of the 
    /// new underlying type.
    pub fn map<T, F>(self, mut op: F) -> Point1<T> where F: FnMut(S) -> T {
        Point1 { x: op(self.x) }
    }
}

impl<S> Point1<S> where S: NumCast + Copy {
    /// Cast a point of one type of scalars to a point of another type of scalars.
    pub fn cast<T: NumCast>(&self) -> Option<Point1<T>> {
        let x = match num_traits::cast(self.x) {
            Some(value) => value,
            None => return None,
        };

        Some(Point1::new(x))
    }
}


/// A representation of two-dimensional points with a Euclidean metric.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(C)]
pub struct Point2<S> {
   pub x: S,
   pub y: S,
}

impl<S> Point2<S> {
    /// Construct a new point.
    pub const fn new(x: S, y: S) -> Point2<S> {
        Point2 { x: x, y: y }
    }

    /// Map an operation on the elements of a point, returning a point of the 
    /// new underlying type.
    pub fn map<T, F>(self, mut op: F) -> Point2<T> where F: FnMut(S) -> T {
        Point2 {
            x: op(self.x),
            y: op(self.y),
        }
    }
}

impl<S> Point2<S> where S: NumCast + Copy {
    /// Cast a point of one type of scalars to a point of another type of scalars.
    pub fn cast<T: NumCast>(&self) -> Option<Point2<T>> {
        let x = match num_traits::cast(self.x) {
            Some(value) => value,
            None => return None,
        };
        let y = match num_traits::cast(self.y) {
            Some(value) => value,
            None => return None,
        };

        Some(Point2::new(x, y))
    }
}


/// A representation of three-dimensional points in a Euclidean space.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(C)]
pub struct Point3<S> {
    pub x: S,
    pub y: S,
    pub z: S,
}

impl<S> Point3<S> {
    /// Construct a new point.
    pub const fn new(x: S, y: S, z: S) -> Point3<S> {
        Point3 { x: x, y: y, z: z }
    }

    /// Map an operation on the elements of a point, returning a point of the 
    /// new underlying type.
    pub fn map<T, F>(self, mut op: F) -> Point3<T> where F: FnMut(S) -> T {
        Point3 {
            x: op(self.x),
            y: op(self.y),
            z: op(self.z),
        }
    }
}

impl<S> Point3<S> where S: NumCast + Copy {
    /// Cast a point from one type of scalars to another type of scalars.
    pub fn cast<T: NumCast>(&self) -> Option<Point3<T>> {
        let x = match num_traits::cast(self.x) {
            Some(value) => value,
            None => return None,
        };
        let y = match num_traits::cast(self.y) {
            Some(value) => value,
            None => return None,
        };
        let z = match num_traits::cast(self.z) {
            Some(value) => value,
            None => return None,
        };

        Some(Point3::new(x, y, z))
    }
}
