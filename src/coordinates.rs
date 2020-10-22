#[macro_export]
macro_rules! impl_coords {
    ($T:ident, { $($comps: ident),* }) => {
        #[repr(C)]
        #[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
        pub struct $T<S: Copy> {
            $(pub $comps: S),*
        }
    }
}

#[macro_export]
macro_rules! impl_coords_deref {
    ($Source:ident, $Target:ident) => {
        impl<S> Deref for $Source<S> where S: Copy
        {
            type Target = $Target<S>;

            #[inline]
            fn deref(&self) -> &Self::Target {
                unsafe { mem::transmute(self.as_ptr()) }
            }
        }

        impl<S> DerefMut for $Source<S> where S: Copy
        {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { mem::transmute(self.as_mut_ptr()) }
            }
        }
    }
}

