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


pub trait MetricSpace: Sized {
    fn distance2(self, to: Self) -> f32;

    fn distance(self, to: Self) -> f32 {
        f32::sqrt(self.distance2(to))
    }
}
