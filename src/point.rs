use num_traits::NumCast;


#[repr(C)]
pub struct Point1<S> {
    pub x: S,
}

impl<S> Point1<S> {
    /// Construct a new vector.
    pub const fn new(x: S) -> Point1<S> {
        Point1 { x: x }
    }

    /// Map an operation on the elements of a vector, returning a vector of the 
    /// new underlying type.
    pub fn map<T, F>(self, mut op: F) -> Point1<T> where F: FnMut(S) -> T {
        Point1 { x: op(self.x) }
    }
}

impl<S> Point1<S> where S: NumCast + Copy {
    /// Cast a vector from one type of scalars to another type of scalars.
    pub fn cast<T: NumCast>(&self) -> Option<Point1<T>> {
        let x = match NumCast::from(self.x) {
            Some(value) => value,
            None => return None,
        };

        Some(Point1::new(x))
    }
}


