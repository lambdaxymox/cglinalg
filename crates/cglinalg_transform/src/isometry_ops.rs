use cglinalg_numeric::{
    SimdScalarFloat
};
use cglinalg_core::{
    DimMul,
    Const,
    ShapeConstraint,
};
use crate::isometry::{
    Isometry,
};
use crate::rotation::{
    Rotation,
};
use crate::translation::{
    Translation,
};

use core::ops;


impl<S, const N: usize> ops::Mul<Translation<S, N>> for Rotation<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: Translation<S, N>) -> Self::Output {
        let distance = self * other.vector;
        let translation = Translation::from_vector(&distance);

        Isometry::from_parts(&translation, &self)
    }
}

impl<S, const N: usize> ops::Mul<&Translation<S, N>> for Rotation<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: &Translation<S, N>) -> Self::Output {
        let distance = self * other.vector;
        let translation = Translation::from_vector(&distance);

        Isometry::from_parts(&translation, &self)
    }
}

impl<S, const N: usize> ops::Mul<Translation<S, N>> for &Rotation<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: Translation<S, N>) -> Self::Output {
        let distance = self * other.vector;
        let translation = Translation::from_vector(&distance);

        Isometry::from_parts(&translation, self)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'b Translation<S, N>> for &'a Rotation<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: &'b Translation<S, N>) -> Self::Output {
        let distance = self * other.vector;
        let translation = Translation::from_vector(&distance);

        Isometry::from_parts(&translation, self)
    }
}

impl<S, const N: usize> ops::Mul<Rotation<S, N>> for Translation<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: Rotation<S, N>) -> Self::Output {
        Isometry::from_parts(&self, &other)
    }
}

impl<S, const N: usize> ops::Mul<&Rotation<S, N>> for Translation<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: &Rotation<S, N>) -> Self::Output {
        Isometry::from_parts(&self, other)
    }
}

impl<S, const N: usize> ops::Mul<Rotation<S, N>> for &Translation<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: Rotation<S, N>) -> Self::Output {
        Isometry::from_parts(self, &other)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'b Rotation<S, N>> for &'a Translation<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: &'b Rotation<S, N>) -> Self::Output {
        Isometry::from_parts(self, other)
    }
}

impl<S, const N: usize> ops::Mul<Translation<S, N>> for Isometry<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn mul(self, other: Translation<S, N>) -> Self::Output {
        let new_vector = self.translation.vector + self.rotation.apply_vector(&other.vector);
        let new_translation = Translation::from_vector(&new_vector);
        
        Isometry::from_parts(&new_translation, &self.rotation)
    }
}

impl<S, const N: usize> ops::Mul<&Translation<S, N>> for Isometry<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn mul(self, other: &Translation<S, N>) -> Self::Output {
        let new_vector = self.translation.vector + self.rotation.apply_vector(&other.vector);
        let new_translation = Translation::from_vector(&new_vector);
        
        Isometry::from_parts(&new_translation, &self.rotation)
    }
}

impl<S, const N: usize> ops::Mul<Translation<S, N>> for &Isometry<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn mul(self, other: Translation<S, N>) -> Self::Output {
        let new_vector = self.translation.vector + self.rotation.apply_vector(&other.vector);
        let new_translation = Translation::from_vector(&new_vector);
        
        Isometry::from_parts(&new_translation, &self.rotation)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'b Translation<S, N>> for &'a Isometry<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn mul(self, other: &'b Translation<S, N>) -> Self::Output {
        let new_vector = self.translation.vector + self.rotation.apply_vector(&other.vector);
        let new_translation = Translation::from_vector(&new_vector);
        
        Isometry::from_parts(&new_translation, &self.rotation)
    }
}

impl<S, const N: usize> ops::Mul<Isometry<S, N>> for Translation<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: Isometry<S, N>) -> Self::Output {
        let new_translation = self * other.translation;

        Isometry::from_parts(&new_translation, &other.rotation)
    }
}

impl<S, const N: usize> ops::Mul<&Isometry<S, N>> for Translation<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: &Isometry<S, N>) -> Self::Output {
        let new_translation = self * other.translation;

        Isometry::from_parts(&new_translation, &other.rotation)
    }
}

impl<S, const N: usize> ops::Mul<Isometry<S, N>> for &Translation<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: Isometry<S, N>) -> Self::Output {
        let new_translation = self * other.translation;

        Isometry::from_parts(&new_translation, &other.rotation)
    }
}

impl<'a, 'b, S, const N: usize> ops::Mul<&'b Isometry<S, N>> for &'a Translation<S, N>
where
    S: SimdScalarFloat
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: &'b Isometry<S, N>) -> Self::Output {
        let new_translation = self * other.translation;

        Isometry::from_parts(&new_translation, &other.rotation)
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<Rotation<S, N>> for Isometry<S, N>
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: Rotation<S, N>) -> Self::Output {
        let new_rotation = self.rotation * other;

        Isometry::from_parts(&self.translation, &new_rotation)
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<&Rotation<S, N>> for Isometry<S, N>
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: &Rotation<S, N>) -> Self::Output {
        let new_rotation = self.rotation * other;

        Isometry::from_parts(&self.translation, &new_rotation)
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<Rotation<S, N>> for &Isometry<S, N>
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: Rotation<S, N>) -> Self::Output {
        let new_rotation = self.rotation * other;

        Isometry::from_parts(&self.translation, &new_rotation)
    }
}

impl<'a, 'b, S, const N: usize, const NN: usize> ops::Mul<&'b Rotation<S, N>> for &'a Isometry<S, N>
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: &'b Rotation<S, N>) -> Self::Output {
        let new_rotation = self.rotation * other;

        Isometry::from_parts(&self.translation, &new_rotation)
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<Isometry<S, N>> for Rotation<S, N>
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: Isometry<S, N>) -> Self::Output {
        let distance = self * other.translation.vector;
        let new_translation = Translation::from_vector(&distance);
        let new_rotation = self * other.rotation;

        Isometry::from_parts(&new_translation, &new_rotation)
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<&Isometry<S, N>> for Rotation<S, N>
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: &Isometry<S, N>) -> Self::Output {
        let distance = self * other.translation.vector;
        let new_translation = Translation::from_vector(&distance);
        let new_rotation = self * other.rotation;

        Isometry::from_parts(&new_translation, &new_rotation)
    }
}

impl<S, const N: usize, const NN: usize> ops::Mul<Isometry<S, N>> for &Rotation<S, N>
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: Isometry<S, N>) -> Self::Output {
        let distance = self * other.translation.vector;
        let new_translation = Translation::from_vector(&distance);
        let new_rotation = self * other.rotation;

        Isometry::from_parts(&new_translation, &new_rotation)
    }
}

impl<'a, 'b, S, const N: usize, const NN: usize> ops::Mul<&'b Isometry<S, N>> for &'a Rotation<S, N>
where
    S: SimdScalarFloat,
    ShapeConstraint: DimMul<Const<N>, Const<N>, Output = Const<NN>>
{
    type Output = Isometry<S, N>;

    #[inline]
    fn mul(self, other: &'b Isometry<S, N>) -> Self::Output {
        let distance = self * other.translation.vector;
        let new_translation = Translation::from_vector(&distance);
        let new_rotation = self * other.rotation;

        Isometry::from_parts(&new_translation, &new_rotation)
    }
}

