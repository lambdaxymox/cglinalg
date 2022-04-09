/*
 * Generate a view into a vector or matrix type that accesses the components 
 * of the vector or matrix type by name.
 *
 * The component names are used in conjunction with a `Deref` implementation
 * such that one can access the components of a matrix or vector by name instead
 * of only by index. For example, the components names for the  three-dimensional
 * vector are `x`, `y` and `z`. The underlying data structure for the vector is
 * an array of length three, and each component name corresponds to an element 
 * of the array:
 * 
 * ```text
 * vector.x <--> vector[0]
 * vector.y <--> vector[1]
 * vector.z <--> vector[2]
 * ```
 */
#[macro_export]
macro_rules! impl_coords {
    ($T:ident, { $($comps: ident),* }) => {
        #[allow(clippy::upper_case_acronyms)]
        #[repr(C)]
        #[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
        pub struct $T<S: Copy> {
            $(pub $comps: S),*
        }
    }
}

/*
 *
 * Generate the component accessors for a vector or matrix type.
 *
 */
#[macro_export]
macro_rules! impl_coords_deref {
    ($Source:ident, $Target:ident) => {
        impl<S> core::ops::Deref for $Source<S> where S: Copy
        {
            type Target = $Target<S>;

            #[inline]
            fn deref(&self) -> &Self::Target {
                unsafe { 
                    &*(self.as_ptr() as *const $Target<S>) 
                }
            }
        }

        impl<S> core::ops::DerefMut for $Source<S> where S: Copy
        {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { 
                    &mut *(self.as_mut_ptr() as *mut $Target<S>) 
                }
            }
        }
    }
}

