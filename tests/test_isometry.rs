extern crate cglinalg;


#[cfg(test)]
mod isometry2_tests {
    use cglinalg::{
        Isometry2,
        Degrees,
        Point2,
        Vector2,
        Unit,
    };
    use cglinalg::approx::{
        relative_eq,
    };



}


#[cfg(test)]
mod rotation3_tests {
    use cglinalg::{
        Isometry3,
        Angle,
        Degrees,
        Radians,
        Point3,
        Vector3,
        Unit,
    };
    use cglinalg::approx::{
        relative_eq,
    };

}