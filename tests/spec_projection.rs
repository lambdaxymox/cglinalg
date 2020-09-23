extern crate cglinalg;
extern crate num_traits;
extern crate proptest;


use cglinalg::{
    Vector3,
    Point3,
    PerspectiveSpec,
    PerspectiveProjection3D,
    OrthographicSpec,
    OrthographicProjection3D,
    ScalarFloat,
};

use proptest::prelude::*;


fn any_vector3<S>() -> impl Strategy<Value = Vector3<S>> 
    where S: ScalarFloat + Arbitrary
{
    any::<(S, S, S)>()
        .prop_map(|(x, y, z)| Vector3::new(x, y, z))
}

fn any_point3<S>() -> impl Strategy<Value = Point3<S>> 
    where S: ScalarFloat + Arbitrary 
{
    any::<(S, S, S)>()
        .prop_map(|(x, y, z)| Point3::new(x, y, z))
}

fn any_perspective_projection<S>() -> impl Strategy<Value = PerspectiveProjection3D<S, PerspectiveSpec<S>>> 
    where S: ScalarFloat + Arbitrary
{
    any::<(S, S, S, S, S, S)>()
        .prop_filter("", |(left, right, bottom, top, near, far)| {
            left.is_finite()   &&
            right.is_finite()  &&
            bottom.is_finite() &&
            top.is_finite()    &&
            near.is_finite()   &&
            far.is_finite()
        })
        .prop_map(|(left, right, bottom, top, near, far)| {
            (-S::abs(left) - S::one(),
              S::abs(right) + S::one(),
             -S::abs(bottom) - S::one(),
              S::abs(top) + S::one(),
              S::abs(near),
              S::abs(far) + S::one(),
            )
        })    
        .prop_map(|(left, right, bottom, top, near, far)| {
            let (spec_left, spec_right) = if left > right {
                (right, left)
            } else {
                (left, right)
            };
            let (spec_bottom, spec_top) = if bottom > top {
                (top, bottom)
            } else {
                (bottom, top)
            };
            let (spec_near, spec_far) = if near > far {
                (far, near)
            } else {
                (near, far)
            };
            let spec = PerspectiveSpec::new(
                spec_left, spec_right, spec_bottom, spec_top, spec_near, spec_far
            );

            PerspectiveProjection3D::new(spec)
        })
        .no_shrink()
}

fn any_orthographic_projection<S>() -> impl Strategy<Value = OrthographicProjection3D<S>>
    where S: ScalarFloat + Arbitrary
{
    any::<(S, S, S, S, S, S)>()
        .prop_filter("", |(left, right, bottom, top, near, far)| {
            left.is_finite()   &&
            right.is_finite()  &&
            bottom.is_finite() &&
            top.is_finite()    &&
            near.is_finite()   &&
            far.is_finite()
        })
        .prop_map(|(left, right, bottom, top, near, far)| {
            (-S::abs(left) - S::one(),
              S::abs(right) + S::one(),
             -S::abs(bottom) - S::one(),
              S::abs(top) + S::one(),
              S::abs(near),
              S::abs(far) + S::one(),
            )
        })    
        .prop_map(|(left, right, bottom, top, near, far)| {
            let (spec_left, spec_right) = if left > right {
                (right, left)
            } else {
                (left, right)
            };
            let (spec_bottom, spec_top) = if bottom > top {
                (top, bottom)
            } else {
                (bottom, top)
            };
            let (spec_near, spec_far) = if near > far {
                (far, near)
            } else {
                (near, far)
            };
            let spec = OrthographicSpec::new(
                spec_left, spec_right, spec_bottom, spec_top, spec_near, spec_far
            );

            OrthographicProjection3D::new(spec)
        })
        .no_shrink()
}

/// Generates the properties tests for perspective projection testing.
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name of the module we place the tests in to 
///    separate them from each other for each scalar type to prevent namespace
///    collisions.
/// * `$VectorN` denotes the name of the vector type.
/// * `$ScalarType` denotes the underlying system of numbers.
/// * `$ProjGen` denotes the type of the projection transformation.
/// * `$VecGen` is the name of a function or closure for generating vector 
///    examples.
/// * `$PointGen` is the name of a function for generating point examples.
/// * `$tolerance` specifies the highest amount of acceptable error in the 
///    floating point computations that defines a correct computation. We cannot 
///    guarantee floating point computations will be exact since the underlying 
///    floating point arithmetic is not exact.
macro_rules! perspective_projection_props {
    ($TestModuleName:ident, $ScalarType:ty, $ProjGen:ident, $VecGen:ident, $PointGen:ident, $tolerance:expr) => {
        #[cfg(test)]
        mod $TestModuleName {
            use cglinalg::approx::{
                relative_eq,
            };
            use super::{
                $ProjGen,
                $VecGen,
                $PointGen,
            };
            use proptest::prelude::*;

            proptest! {
                #[test]
                fn prop_perspective_projection_project_vector(
                    projection in $ProjGen::<$ScalarType>(), vector in $VecGen::<$ScalarType>()) {
                    
                    let projected_vector = projection.project_vector(&vector);
                    let unprojected_vector = projection.unproject_vector(&projected_vector);

                    prop_assert!(
                        relative_eq!(unprojected_vector, vector, epsilon = $tolerance),
                        "projection = {}\nvector = {}\nprojected_vector = {}\nunprojected_vector={}",
                        projection, vector, projected_vector, unprojected_vector
                    );
                }

                #[test]
                fn prop_perspective_projection_project_point(
                    projection in $ProjGen::<$ScalarType>(), point in $PointGen::<$ScalarType>()) {
                    
                    let projected_point = projection.project_point(&point);
                    let unprojected_point = projection.unproject_point(&projected_point);

                    prop_assert!(
                        relative_eq!(unprojected_point, point, epsilon = $tolerance),
                        "projection = {}\npoint = {}\nprojected_point = {}\nunprojected_point={}",
                        projection, point, projected_point, unprojected_point
                    );
                }
            
                #[test]
                fn prop_perspective_projection_inverse_unproject_vector(
                    projection in $ProjGen::<$ScalarType>(), 
                    vector in $VecGen::<$ScalarType>()) {
                    
                    let unprojected_vector = projection.unproject_vector(&vector);
                    let projected_vector = projection.project_vector(&unprojected_vector);

                    prop_assert!(
                        relative_eq!(projected_vector, vector, epsilon = $tolerance),
                        "projection = {}\nvector = {}\nunprojected_vector = {}\nprojected_vector={}",
                        projection, vector, unprojected_vector, projected_vector
                    );

                }

                #[test]
                fn prop_perspective_projection_inverse_unproject_point(
                    projection in $ProjGen::<$ScalarType>(), point in $PointGen::<$ScalarType>()) {
                    
                    let unprojected_point = projection.unproject_point(&point);
                    let projected_point = projection.project_point(&unprojected_point);

                    prop_assert!(
                        relative_eq!(projected_point, point, epsilon = $tolerance),
                        "projection = {}\npoint = {}\nunprojected_point = {}\nprojected_point={}",
                        projection, point, unprojected_point, projected_point
                    );
                }
            }
        }
    }
}

perspective_projection_props!(
    perspective_f64_props, 
    f64, 
    any_perspective_projection, 
    any_vector3, 
    any_point3, 
    1e-7
);


/// Generates the properties tests for orthographic projection testing.
///
/// The macro parameters are the following:
/// * `$TestModuleName` is a name of the module we place the tests in to 
///    separate them from each other for each scalar type to prevent namespace
///    collisions.
/// * `$VectorN` denotes the name of the vector type.
/// * `$ScalarType` denotes the underlying system of numbers.
/// * `$ProjGen` denotes the type of the projection transformation.
/// * `$VecGen` is the name of a function or closure for generating vector 
///    examples.
/// * `$PointGen` is the name of a function for generating point examples.
/// * `$tolerance` specifies the highest amount of acceptable error in the 
///    floating point computations that defines a correct computation. We cannot 
///    guarantee floating point computations will be exact since the underlying 
///    floating point arithmetic is not exact.
macro_rules! orthographic_projection_props {
    ($TestModuleName:ident, $ScalarType:ty, $ProjGen:ident, $VecGen:ident, $PointGen:ident, $tolerance:expr) => {
        #[cfg(test)]
        mod $TestModuleName {
            use cglinalg::approx::{
                relative_eq,
            };
            use super::{
                $ProjGen,
                $VecGen,
                $PointGen,
            };
            use proptest::prelude::*;

            proptest! {
                #[test]
                fn prop_orthographic_projection_project_vector(
                    projection in $ProjGen::<$ScalarType>(), vector in $VecGen::<$ScalarType>()) {
                    
                    let projected_vector = projection.project_vector(&vector);
                    let unprojected_vector = projection.unproject_vector(&projected_vector);

                    prop_assert!(
                        relative_eq!(unprojected_vector, vector, epsilon = $tolerance),
                        "projection = {}\nvector = {}\nprojected_vector = {}\nunprojected_vector={}",
                        projection, vector, projected_vector, unprojected_vector
                    );
                }

                #[test]
                fn prop_orthographic_projection_project_point(
                    projection in $ProjGen::<$ScalarType>(), point in $PointGen::<$ScalarType>()) {
                    
                    let projected_point = projection.project_point(&point);
                    let unprojected_point = projection.unproject_point(&projected_point);

                    prop_assert!(
                        relative_eq!(unprojected_point, point, epsilon = $tolerance),
                        "projection = {}\npoint = {}\nprojected_point = {}\nunprojected_point={}",
                        projection, point, projected_point, unprojected_point
                    );
                }
            
                #[test]
                fn prop_orthographic_projection_inverse_unproject_vector(
                    projection in $ProjGen::<$ScalarType>(), 
                    vector in $VecGen::<$ScalarType>()) {
                    
                    let unprojected_vector = projection.unproject_vector(&vector);
                    let projected_vector = projection.project_vector(&unprojected_vector);

                    prop_assert!(
                        relative_eq!(projected_vector, vector, epsilon = $tolerance),
                        "projection = {}\nvector = {}\nunprojected_vector = {}\nprojected_vector={}",
                        projection, vector, unprojected_vector, projected_vector
                    );

                }

                #[test]
                fn prop_orthographic_projection_inverse_unproject_point(
                    projection in $ProjGen::<$ScalarType>(), point in $PointGen::<$ScalarType>()) {
                    
                    let unprojected_point = projection.unproject_point(&point);
                    let projected_point = projection.project_point(&unprojected_point);

                    prop_assert!(
                        relative_eq!(projected_point, point, epsilon = $tolerance),
                        "projection = {}\npoint = {}\nunprojected_point = {}\nprojected_point={}",
                        projection, point, unprojected_point, projected_point
                    );
                }
            }
        }
    }
}

orthographic_projection_props!(
    orthographic_f64_props, 
    f64, 
    any_orthographic_projection, 
    any_vector3, 
    any_point3, 
    1e-7
);

