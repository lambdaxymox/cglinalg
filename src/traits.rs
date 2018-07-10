pub trait AsArray {
    type Element: Copy;

    fn len() -> usize;

    fn as_ptr(&self) -> *const Self::Element; 

    fn as_mut_ptr(&mut self) -> *mut Self::Element; 
}