#[cfg(test)]
mod shear2_tests {
    use cglinalg_core::{
        Point2,
        Unit,
        Vector2,
    };
    use cglinalg_transform::Shear2;

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_point() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_xy(shear_factor);
        let vertices = [
            Point2::new( 1_i32,  1_i32),
            Point2::new(-1_i32,  1_i32),
            Point2::new(-1_i32, -1_i32),
            Point2::new( 1_i32, -1_i32),
        ];
        let expected = [
            Point2::new( 1_i32 + shear_factor,  1_i32),
            Point2::new(-1_i32 + shear_factor,  1_i32),
            Point2::new(-1_i32 - shear_factor, -1_i32),
            Point2::new( 1_i32 - shear_factor, -1_i32),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_vector() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_xy(shear_factor);
        let vertices = [
            Vector2::new( 1_i32,  1_i32),
            Vector2::new(-1_i32,  1_i32),
            Vector2::new(-1_i32, -1_i32),
            Vector2::new( 1_i32, -1_i32),
        ];
        let expected = [
            Vector2::new( 1_i32 + shear_factor,  1_i32),
            Vector2::new(-1_i32 + shear_factor,  1_i32),
            Vector2::new(-1_i32 - shear_factor, -1_i32),
            Vector2::new( 1_i32 - shear_factor, -1_i32),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_xy(shear_factor);
        let vertices = [
            Point2::new( 1_i32, 0_i32),
            Point2::new( 0_i32, 0_i32),
            Point2::new(-1_i32, 0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_xy(shear_factor);
        let vertices = [
            Vector2::new( 1_i32, 0_i32),
            Vector2::new( 0_i32, 0_i32),
            Vector2::new(-1_i32, 0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_point() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_yx(shear_factor);
        let vertices = [
            Point2::new( 1_i32,  1_i32),
            Point2::new(-1_i32,  1_i32),
            Point2::new(-1_i32, -1_i32),
            Point2::new( 1_i32, -1_i32),
        ];
        let expected = [
            Point2::new( 1_i32,  1_i32 + shear_factor),
            Point2::new(-1_i32,  1_i32 - shear_factor),
            Point2::new(-1_i32, -1_i32 - shear_factor),
            Point2::new( 1_i32, -1_i32 + shear_factor),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_vector() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_yx(shear_factor);
        let vertices = [
            Vector2::new( 1_i32,  1_i32),
            Vector2::new(-1_i32,  1_i32),
            Vector2::new(-1_i32, -1_i32),
            Vector2::new( 1_i32, -1_i32),
        ];
        let expected = [
            Vector2::new( 1_i32,  1_i32 + shear_factor),
            Vector2::new(-1_i32,  1_i32 - shear_factor),
            Vector2::new(-1_i32, -1_i32 - shear_factor),
            Vector2::new( 1_i32, -1_i32 + shear_factor),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_yx(shear_factor);
        let vertices = [
            Point2::new(0_i32,  1_i32),
            Point2::new(0_i32 , 0_i32),
            Point2::new(0_i32, -1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_yx(shear_factor);
        let vertices = [
            Vector2::new(0_i32,  1_i32),
            Vector2::new(0_i32 , 0_i32),
            Vector2::new(0_i32, -1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_xy() {
        let shear_factor = 7_f64;
        let direction = Unit::from_value(Vector2::unit_x());
        let normal = Unit::from_value(Vector2::unit_y());
        let expected = Shear2::from_shear_xy(shear_factor);
        let result = Shear2::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_yx() {
        let shear_factor = 7_f64;
        let direction = Unit::from_value(Vector2::unit_y());
        let normal = Unit::from_value(Vector2::unit_x());
        let expected = Shear2::from_shear_yx(shear_factor);
        let result = Shear2::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_identity_shear_point() {
        let shear = Shear2::identity();
        let vertices = [
            Point2::new( 0_f64 , 0_f64),
            Point2::new( 0_f64,  1_f64),
            Point2::new( 1_f64,  0_f64),
            Point2::new( 1_f64,  1_f64),
            Point2::new( 0_f64, -1_f64),
            Point2::new(-1_f64,  0_f64),
            Point2::new(-1_f64, -1_f64),
            Point2::new( 1_f64, -1_f64),
            Point2::new(-1_f64,  1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_identity_shear_vector() {
        let shear = Shear2::identity();
        let vertices = [
            Vector2::new( 0_f64 , 0_f64),
            Vector2::new( 0_f64,  1_f64),
            Vector2::new( 1_f64,  0_f64),
            Vector2::new( 1_f64,  1_f64),
            Vector2::new( 0_f64, -1_f64),
            Vector2::new(-1_f64,  0_f64),
            Vector2::new(-1_f64, -1_f64),
            Vector2::new( 1_f64, -1_f64),
            Vector2::new(-1_f64,  1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_xy_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector2::unit_x());
        let normal = Unit::from_value(Vector2::unit_y());
        let shear = Shear2::from_shear(shear_factor, &direction, &normal);
        let expected = 3_f64;
        let result = shear.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_yx_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector2::unit_y());
        let normal = Unit::from_value(Vector2::unit_x());
        let shear = Shear2::from_shear(shear_factor, &direction, &normal);
        let expected = 3_f64;
        let result = shear.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector2::new(
            -f64::sqrt(9_f64 / 13_f64), 
            -f64::sqrt(4_f64 / 13_f64),
        ));
        let normal = Unit::from_value(Vector2::new(
             f64::sqrt(4_f64 / 13_f64),
            -f64::sqrt(9_f64 / 13_f64),
        ));
        let shear = Shear2::from_shear(shear_factor, &direction, &normal);
        let expected = 3_f64;
        let result = shear.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod shear2_inverse_tests {
    use cglinalg_core::{
        Point2,
        Unit,
        Vector2,
    };
    use cglinalg_transform::Shear2;

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_inverse_point() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_xy(shear_factor);
        let vertices = [
            Point2::new( 1_i32,  1_i32),
            Point2::new(-1_i32,  1_i32),
            Point2::new(-1_i32, -1_i32),
            Point2::new( 1_i32, -1_i32),
        ];
        let expected = [
            Point2::new( 1_i32 - shear_factor,  1_i32),
            Point2::new(-1_i32 - shear_factor,  1_i32),
            Point2::new(-1_i32 + shear_factor, -1_i32),
            Point2::new( 1_i32 + shear_factor, -1_i32),
        ];
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_inverse_vector() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_xy(shear_factor);
        let vertices = [
            Vector2::new( 1_i32,  1_i32),
            Vector2::new(-1_i32,  1_i32),
            Vector2::new(-1_i32, -1_i32),
            Vector2::new( 1_i32, -1_i32),
        ];
        let expected = [
            Vector2::new( 1_i32 - shear_factor,  1_i32),
            Vector2::new(-1_i32 - shear_factor,  1_i32),
            Vector2::new(-1_i32 + shear_factor, -1_i32),
            Vector2::new( 1_i32 + shear_factor, -1_i32),
        ];
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_inverse_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_xy(shear_factor);
        let vertices = [
            Point2::new( 1_i32, 0_i32),
            Point2::new( 0_i32, 0_i32),
            Point2::new(-1_i32, 0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_inverse_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_xy(shear_factor);
        let vertices = [
            Vector2::new( 1_i32, 0_i32),
            Vector2::new( 0_i32, 0_i32),
            Vector2::new(-1_i32, 0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_inverse_point() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_yx(shear_factor);
        let vertices = [
            Point2::new( 1_i32,  1_i32),
            Point2::new(-1_i32,  1_i32),
            Point2::new(-1_i32, -1_i32),
            Point2::new( 1_i32, -1_i32),
        ];
        let expected = [
            Point2::new( 1_i32,  1_i32 - shear_factor),
            Point2::new(-1_i32,  1_i32 + shear_factor),
            Point2::new(-1_i32, -1_i32 + shear_factor),
            Point2::new( 1_i32, -1_i32 - shear_factor),
        ];
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_inverse_vector() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_yx(shear_factor);
        let vertices = [
            Vector2::new( 1_i32,  1_i32),
            Vector2::new(-1_i32,  1_i32),
            Vector2::new(-1_i32, -1_i32),
            Vector2::new( 1_i32, -1_i32),
        ];
        let expected = [
            Vector2::new( 1_i32,  1_i32 - shear_factor),
            Vector2::new(-1_i32,  1_i32 + shear_factor),
            Vector2::new(-1_i32, -1_i32 + shear_factor),
            Vector2::new( 1_i32, -1_i32 - shear_factor),
        ];
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_inverse_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_yx(shear_factor);
        let vertices = [
            Point2::new(0_i32,  1_i32),
            Point2::new(0_i32 , 0_i32),
            Point2::new(0_i32, -1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_inverse_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear2::from_shear_yx(shear_factor);
        let vertices = [
            Vector2::new(0_i32,  1_i32),
            Vector2::new(0_i32 , 0_i32),
            Vector2::new(0_i32, -1_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_inverse_from_shear_xy_inverse() {
        let shear_factor = 7_f64;
        let direction = Unit::from_value(Vector2::unit_x());
        let normal = Unit::from_value(Vector2::unit_y());
        let shear = Shear2::from_shear(shear_factor, &direction, &normal);
        let expected = Shear2::from_shear_xy(-shear_factor);
        let result = shear.inverse();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_inverse_from_shear_yx_inverse() {
        let shear_factor = 7_f64;
        let direction = Unit::from_value(Vector2::unit_y());
        let normal = Unit::from_value(Vector2::unit_x());
        let shear = Shear2::from_shear(shear_factor, &direction, &normal);
        let expected = Shear2::from_shear_yx(-shear_factor);
        let result = shear.inverse();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_identity_shear_inverse_point() {
        let shear = Shear2::identity();
        let vertices = [
            Point2::new( 0_f64 , 0_f64),
            Point2::new( 0_f64,  1_f64),
            Point2::new( 1_f64,  0_f64),
            Point2::new( 1_f64,  1_f64),
            Point2::new( 0_f64, -1_f64),
            Point2::new(-1_f64,  0_f64),
            Point2::new(-1_f64, -1_f64),
            Point2::new( 1_f64, -1_f64),
            Point2::new(-1_f64,  1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_identity_shear_inverse_vector() {
        let shear = Shear2::identity();
        let vertices = [
            Vector2::new( 0_f64 , 0_f64),
            Vector2::new( 0_f64,  1_f64),
            Vector2::new( 1_f64,  0_f64),
            Vector2::new( 1_f64,  1_f64),
            Vector2::new( 0_f64, -1_f64),
            Vector2::new(-1_f64,  0_f64),
            Vector2::new(-1_f64, -1_f64),
            Vector2::new( 1_f64, -1_f64),
            Vector2::new(-1_f64,  1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_xy_inverse_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector2::unit_x());
        let normal = Unit::from_value(Vector2::unit_y());
        let shear = Shear2::from_shear(shear_factor, &direction, &normal);
        let shear_inv = shear.inverse();
        let expected = 3_f64;
        let result = shear_inv.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_yx_inverse_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector2::unit_y());
        let normal = Unit::from_value(Vector2::unit_x());
        let shear = Shear2::from_shear(shear_factor, &direction, &normal);
        let shear_inv = shear.inverse();
        let expected = 3_f64;
        let result = shear_inv.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_inverse_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector2::new(
            -f64::sqrt(9_f64 / 13_f64),
            -f64::sqrt(4_f64 / 13_f64)
        ));
        let normal = Unit::from_value(Vector2::new(
             f64::sqrt(4_f64 / 13_f64),
            -f64::sqrt(9_f64 / 13_f64)
        ));
        let shear = Shear2::from_shear(shear_factor, &direction, &normal);
        let shear_inv = shear.inverse();
        let expected = 3_f64;
        let result = shear_inv.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod shear2_coordinate_plane_tests {
    use cglinalg_core::{
        Matrix3x3,
        Point2,
        Unit,
        Vector2,
    };
    use cglinalg_transform::Shear2;

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy() {
        let shear_factor = 7_f64;
        let origin = Point2::new(-2_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_x());
        let normal = Unit::from_value(Vector2::unit_y());
        let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector2::new( 1_f64,  1_f64),
            Vector2::new(-1_f64,  1_f64),
            Vector2::new(-1_f64, -1_f64),
            Vector2::new( 1_f64, -1_f64),
        ];
        let expected = [
            Vector2::new( 1_f64 + shear_factor,  1_f64),
            Vector2::new(-1_f64 + shear_factor,  1_f64),
            Vector2::new(-1_f64 - shear_factor, -1_f64),
            Vector2::new( 1_f64 - shear_factor, -1_f64),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy_matrix() {
        let shear_factor = 7_f64;
        let origin = Point2::new(-2_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_x());
        let normal = Unit::from_value(Vector2::unit_y());
        let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let expected = Matrix3x3::new(
            1_f64,        0_f64, 0_f64,
            shear_factor, 1_f64, 0_f64,
            0_f64,        0_f64, 1_f64,
        );
        let result = shear.to_affine_matrix();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy_shearing_plane() {
        let shear_factor = 7_f64;
        let origin = Point2::new(-2_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_x());
        let normal = Unit::from_value(Vector2::unit_y());
        let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector2::new( 1_f64, 0_f64),
            Vector2::new(-1_f64, 0_f64),
            Vector2::new( 0_f64, 0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx() {
        let shear_factor = 7_f64;
        let origin = Point2::new(-2_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_y());
        let normal = Unit::from_value(Vector2::unit_x());
        let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector2::new( 1_f64,  1_f64),
            Vector2::new(-1_f64,  1_f64),
            Vector2::new(-1_f64, -1_f64),
            Vector2::new( 1_f64, -1_f64),
        ];
        let expected = [
            Vector2::new( 1_f64,  1_f64 + 3_f64 * shear_factor),
            Vector2::new(-1_f64,  1_f64 + shear_factor),
            Vector2::new(-1_f64, -1_f64 + shear_factor),
            Vector2::new( 1_f64, -1_f64 + 3_f64 * shear_factor),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx_matrix() {
        let shear_factor = 7_f64;
        let origin = Point2::new(-2_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_y());
        let normal = Unit::from_value(Vector2::unit_x());
        let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let expected = Matrix3x3::new(
            1_f64,  shear_factor,             0_f64,
            0_f64,  1_f64,                    0_f64,
            0_f64, -origin[0] * shear_factor, 1_f64,
        );
        let result = shear.to_affine_matrix();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx_shearing_plane() {
        let shear_factor = 7_f64;
        let origin = Point2::new(-2_f64, 0_f64);
        let direction = Unit::from_value(Vector2::unit_y());
        let normal = Unit::from_value(Vector2::unit_x());
        let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector2::new(-2_f64,  1_f64),
            Vector2::new(-2_f64, -1_f64),
            Vector2::new(-2_f64,  0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }
}

/// Shearing along the plane `(1 / 2) * x + 1 - y == 0`
/// with origin `[2, 2]`, direction `[2 / sqrt(5), 1 / sqrt(5)]`, and
/// normal `[-1 / sqrt(5), 2 / sqrt(5)]`.
#[cfg(test)]
mod shear2_noncoordinate_plane_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix3x3,
        Point2,
        Unit,
        Vector2,
    };
    use cglinalg_transform::{
        Rotation2,
        Shear2,
        Translation2,
    };
    use cglinalg_trigonometry::{
        Angle,
        Radians,
    };

    fn shear_factor() -> f64 {
        7_f64
    }

    fn rotation_angle() -> Radians<f64> {
        Radians(f64::atan2(1_f64, 2_f64))
    }

    fn origin() -> Point2<f64> {
        Point2::new(2_f64, 2_f64)
    }

    fn direction() -> Unit<Vector2<f64>> {
        Unit::from_value(Vector2::new(2_f64 / f64::sqrt(5_f64), 1_f64 / f64::sqrt(5_f64)))
    }

    fn normal() -> Unit<Vector2<f64>> {
        Unit::from_value(Vector2::new(-1_f64 / f64::sqrt(5_f64), 2_f64 / f64::sqrt(5_f64)))
    }

    fn translation() -> Translation2<f64> {
        Translation2::from_vector(&Vector2::new(0_f64, 1_f64))
    }

    fn translation_inv() -> Translation2<f64> {
        Translation2::from_vector(&Vector2::new(0_f64, -1_f64))
    }

    fn rotation() -> Rotation2<f64> {
        let rotation_angle = rotation_angle();

        Rotation2::from_angle(rotation_angle)
    }

    fn rotation_inv() -> Rotation2<f64> {
        let rotation_angle = rotation_angle();

        Rotation2::from_angle(-rotation_angle)
    }

    #[rustfmt::skip]
    fn rotation_matrix() -> Matrix3x3<f64> {
        Matrix3x3::new(
            2_f64 / f64::sqrt(5_f64), 1_f64 / f64::sqrt(5_f64), 0_f64,
           -1_f64 / f64::sqrt(5_f64), 2_f64 / f64::sqrt(5_f64), 0_f64,
            0_f64,                    0_f64,                    1_f64,
        )
    }

    #[rustfmt::skip]
    fn rotation_matrix_inv() -> Matrix3x3<f64> {
        Matrix3x3::new(
            2_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64), 0_f64,
            1_f64 / f64::sqrt(5_f64),  2_f64 / f64::sqrt(5_f64), 0_f64,
            0_f64,                     0_f64,                    1_f64,
        )
    }

    #[rustfmt::skip]
    fn shear_matrix_xy() -> Shear2<f64> {
        let shear_factor = shear_factor();

        Shear2::from_shear_xy(shear_factor)
    }

    #[test]
    fn test_from_affine_shear_rotation_angle() {
        let rotation_angle = rotation_angle();

        assert_relative_eq!(
            rotation_angle.cos(),
            2_f64 / f64::sqrt(5_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
        assert_relative_eq!(
            rotation_angle.sin(),
            1_f64 / f64::sqrt(5_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
    }

    #[test]
    fn test_from_affine_rotation_matrix() {
        let rotation = rotation();
        let expected = rotation_matrix();
        let result = rotation.to_affine_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_rotation_inv_matrix() {
        let rotation_inv = rotation_inv();
        let expected = rotation_matrix_inv();
        let result = rotation_inv.to_affine_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn tests_from_affine_shear_coordinates() {
        let translation = translation();
        let rotation = rotation();
        let vertices = [
            Point2::new( 1_f64 / f64::sqrt(5_f64),  3_f64 / f64::sqrt(5_f64) + 1_f64),
            Point2::new(-3_f64 / f64::sqrt(5_f64),  1_f64 / f64::sqrt(5_f64) + 1_f64),
            Point2::new(-1_f64 / f64::sqrt(5_f64), -3_f64 / f64::sqrt(5_f64) + 1_f64),
            Point2::new( 3_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64) + 1_f64),
        ];
        let rotated_vertices = [
            Point2::new( 1_f64,  1_f64),
            Point2::new(-1_f64,  1_f64),
            Point2::new(-1_f64, -1_f64),
            Point2::new( 1_f64, -1_f64),
        ];
        let expected = vertices;
        let result = rotated_vertices.map(|v| translation * rotation * v);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_origin() {
        let origin = origin();
        let translation = translation();
        let rotation = rotation();
        let rotated_origin = Point2::new(f64::sqrt(5_f64), 0_f64);
        let result_rotated_translated_origin = translation * rotation * rotated_origin;

        assert_relative_eq!(
            result_rotated_translated_origin,
            origin,
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
    }

    #[test]
    fn test_from_affine_shear_direction() {
        let direction = direction();
        let translation_inv = translation_inv();
        let rotation_inv = rotation_inv();
        let expected = Vector2::unit_x();
        let result = {
            let _direction = direction.into_inner();
            translation_inv * rotation_inv * _direction
        };

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_normal() {
        let normal = normal();
        let translation_inv = translation_inv();
        let rotation_inv = rotation_inv();
        let expected = Vector2::unit_y();
        let result = {
            let _normal = normal.into_inner();
            translation_inv * rotation_inv * _normal
        };

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_vertices() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector2::new( 1_f64 / f64::sqrt(5_f64),  3_f64 / f64::sqrt(5_f64) + 1_f64),
            Vector2::new(-3_f64 / f64::sqrt(5_f64),  1_f64 / f64::sqrt(5_f64) + 1_f64),
            Vector2::new(-1_f64 / f64::sqrt(5_f64), -3_f64 / f64::sqrt(5_f64) + 1_f64),
            Vector2::new( 3_f64 / f64::sqrt(5_f64), -1_f64 / f64::sqrt(5_f64) + 1_f64),
        ];
        let expected = [
            Vector2::new(
                 (1_f64 / f64::sqrt(5_f64)) + (2_f64 / f64::sqrt(5_f64)) * shear_factor,
                 (3_f64 / f64::sqrt(5_f64)) + (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
            ),
            Vector2::new(
                -(3_f64 / f64::sqrt(5_f64)) + (2_f64 / f64::sqrt(5_f64)) * shear_factor,
                 (1_f64 / f64::sqrt(5_f64))  + (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
            ),
            Vector2::new(
                -(1_f64 / f64::sqrt(5_f64)) - (2_f64 / f64::sqrt(5_f64)) * shear_factor,
                -(3_f64 / f64::sqrt(5_f64)) - (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
            ),
            Vector2::new(
                 (3_f64 / f64::sqrt(5_f64)) - (2_f64 / f64::sqrt(5_f64)) * shear_factor,
                -(1_f64 / f64::sqrt(5_f64)) - (1_f64 / f64::sqrt(5_f64)) * shear_factor + 1_f64,
            ),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));
    
        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_matrix() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let expected = Matrix3x3::new(
             1_f64 - (2_f64 / 5_f64) * shear_factor, -(1_f64 / 5_f64) * shear_factor,         0_f64,
             (4_f64 / 5_f64) * shear_factor,          1_f64 + (2_f64 / 5_f64) * shear_factor, 0_f64,
            -(4_f64 / 5_f64) * shear_factor,         -(2_f64 / 5_f64) * shear_factor,         1_f64
        );
        let result = shear.to_affine_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_matrix_alternative_path() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let translation = translation();
        let translation_inv = translation_inv();
        let rotation = rotation();
        let rotation_inv = rotation_inv();
        let shear_matrix_xy = shear_matrix_xy();
        let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let expected = shear.to_transform();
        let result = {
            let isometry = (translation * rotation).to_transform();
            let isometry_inv = (rotation_inv * translation_inv).to_transform();
            let _shear_matrix_xy = shear_matrix_xy.to_transform();
            isometry * _shear_matrix_xy * isometry_inv
        };

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_shearing_plane() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let shear = Shear2::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector2::new( 1_f64 / f64::sqrt(5_f64),  1_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64),
            Vector2::new(-3_f64 / f64::sqrt(5_f64), -3_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64),
            Vector2::new(-1_f64 / f64::sqrt(5_f64), -1_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64),
            Vector2::new( 3_f64 / f64::sqrt(5_f64),  3_f64 / (2_f64 * f64::sqrt(5_f64)) + 1_f64),
            Vector2::new( 0_f64, 1_f64),

        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));
    
        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod shear3_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Point3,
        Unit,
        Vector3,
    };
    use cglinalg_transform::Shear3;

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xy(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Point3::new( 1_i32 + shear_factor,  1_i32,  1_i32),
            Point3::new(-1_i32 + shear_factor,  1_i32,  1_i32),
            Point3::new(-1_i32 - shear_factor, -1_i32,  1_i32),
            Point3::new( 1_i32 - shear_factor, -1_i32,  1_i32),
            Point3::new( 1_i32 + shear_factor,  1_i32, -1_i32),
            Point3::new(-1_i32 + shear_factor,  1_i32, -1_i32),
            Point3::new(-1_i32 - shear_factor, -1_i32, -1_i32),
            Point3::new( 1_i32 - shear_factor, -1_i32, -1_i32),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32 + shear_factor,  1_i32,  1_i32),
            Vector3::new(-1_i32 + shear_factor,  1_i32,  1_i32),
            Vector3::new(-1_i32 - shear_factor, -1_i32,  1_i32),
            Vector3::new( 1_i32 - shear_factor, -1_i32,  1_i32),
            Vector3::new( 1_i32 + shear_factor,  1_i32, -1_i32),
            Vector3::new(-1_i32 + shear_factor,  1_i32, -1_i32),
            Vector3::new(-1_i32 - shear_factor, -1_i32, -1_i32),
            Vector3::new( 1_i32 - shear_factor, -1_i32, -1_i32),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xy(shear_factor);
        let vertices = [
            Point3::new( 1_i32, 0_i32,  1_i32),
            Point3::new(-1_i32, 0_i32,  1_i32),
            Point3::new(-1_i32, 0_i32, -1_i32),
            Point3::new( 1_i32, 0_i32, -1_i32),
            Point3::new( 0_i32, 0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32, 0_i32,  1_i32),
            Vector3::new(-1_i32, 0_i32,  1_i32),
            Vector3::new(-1_i32, 0_i32, -1_i32),
            Vector3::new( 1_i32, 0_i32, -1_i32),
            Vector3::new( 0_i32, 0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xz_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xz(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Point3::new( 1_i32 + shear_factor,  1_i32,  1_i32),
            Point3::new(-1_i32 + shear_factor,  1_i32,  1_i32),
            Point3::new(-1_i32 + shear_factor, -1_i32,  1_i32),
            Point3::new( 1_i32 + shear_factor, -1_i32,  1_i32),
            Point3::new( 1_i32 - shear_factor,  1_i32, -1_i32),
            Point3::new(-1_i32 - shear_factor,  1_i32, -1_i32),
            Point3::new(-1_i32 - shear_factor, -1_i32, -1_i32),
            Point3::new( 1_i32 - shear_factor, -1_i32, -1_i32),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xz_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xz(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32 + shear_factor,  1_i32,  1_i32),
            Vector3::new(-1_i32 + shear_factor,  1_i32,  1_i32),
            Vector3::new(-1_i32 + shear_factor, -1_i32,  1_i32),
            Vector3::new( 1_i32 + shear_factor, -1_i32,  1_i32),
            Vector3::new( 1_i32 - shear_factor,  1_i32, -1_i32),
            Vector3::new(-1_i32 - shear_factor,  1_i32, -1_i32),
            Vector3::new(-1_i32 - shear_factor, -1_i32, -1_i32),
            Vector3::new( 1_i32 - shear_factor, -1_i32, -1_i32),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));
        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xz_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xz(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  0_i32),
            Point3::new(-1_i32,  1_i32,  0_i32),
            Point3::new(-1_i32, -1_i32,  0_i32),
            Point3::new( 1_i32, -1_i32,  0_i32),
            Point3::new( 0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xz_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xz(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  0_i32),
            Vector3::new(-1_i32,  1_i32,  0_i32),
            Vector3::new(-1_i32, -1_i32,  0_i32),
            Vector3::new( 1_i32, -1_i32,  0_i32),
            Vector3::new( 0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yx(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Point3::new( 1_i32,  1_i32 + shear_factor,  1_i32),
            Point3::new(-1_i32,  1_i32 - shear_factor,  1_i32),
            Point3::new(-1_i32, -1_i32 - shear_factor,  1_i32),
            Point3::new( 1_i32, -1_i32 + shear_factor,  1_i32),
            Point3::new( 1_i32,  1_i32 + shear_factor, -1_i32),
            Point3::new(-1_i32,  1_i32 - shear_factor, -1_i32),
            Point3::new(-1_i32, -1_i32 - shear_factor, -1_i32),
            Point3::new( 1_i32, -1_i32 + shear_factor, -1_i32),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yx(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32 + shear_factor,  1_i32),
            Vector3::new(-1_i32,  1_i32 - shear_factor,  1_i32),
            Vector3::new(-1_i32, -1_i32 - shear_factor,  1_i32),
            Vector3::new( 1_i32, -1_i32 + shear_factor,  1_i32),
            Vector3::new( 1_i32,  1_i32 + shear_factor, -1_i32),
            Vector3::new(-1_i32,  1_i32 - shear_factor, -1_i32),
            Vector3::new(-1_i32, -1_i32 - shear_factor, -1_i32),
            Vector3::new( 1_i32, -1_i32 + shear_factor, -1_i32),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yx(shear_factor);
        let vertices = [
            Point3::new(0_i32,  1_i32,  1_i32),
            Point3::new(0_i32, -1_i32,  1_i32),
            Point3::new(0_i32,  1_i32, -1_i32),
            Point3::new(0_i32, -1_i32, -1_i32),
            Point3::new(0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yx(shear_factor);
        let vertices = [
            Vector3::new(0_i32,  1_i32,  1_i32),
            Vector3::new(0_i32, -1_i32,  1_i32),
            Vector3::new(0_i32,  1_i32, -1_i32),
            Vector3::new(0_i32, -1_i32, -1_i32),
            Vector3::new(0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yz_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yz(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Point3::new( 1_i32,  1_i32 + shear_factor,  1_i32),
            Point3::new(-1_i32,  1_i32 + shear_factor,  1_i32),
            Point3::new(-1_i32, -1_i32 + shear_factor,  1_i32),
            Point3::new( 1_i32, -1_i32 + shear_factor,  1_i32),
            Point3::new( 1_i32,  1_i32 - shear_factor, -1_i32),
            Point3::new(-1_i32,  1_i32 - shear_factor, -1_i32),
            Point3::new(-1_i32, -1_i32 - shear_factor, -1_i32),
            Point3::new( 1_i32, -1_i32 - shear_factor, -1_i32),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yz_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yz(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32 + shear_factor,  1_i32),
            Vector3::new(-1_i32,  1_i32 + shear_factor,  1_i32),
            Vector3::new(-1_i32, -1_i32 + shear_factor,  1_i32),
            Vector3::new( 1_i32, -1_i32 + shear_factor,  1_i32),
            Vector3::new( 1_i32,  1_i32 - shear_factor, -1_i32),
            Vector3::new(-1_i32,  1_i32 - shear_factor, -1_i32),
            Vector3::new(-1_i32, -1_i32 - shear_factor, -1_i32),
            Vector3::new( 1_i32, -1_i32 - shear_factor, -1_i32),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yz_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yz(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32, 0_i32),
            Point3::new(-1_i32,  1_i32, 0_i32),
            Point3::new(-1_i32, -1_i32, 0_i32),
            Point3::new( 1_i32, -1_i32, 0_i32),
            Point3::new( 0_i32,  0_i32, 0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yz_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yz(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32, 0_i32),
            Vector3::new(-1_i32,  1_i32, 0_i32),
            Vector3::new(-1_i32, -1_i32, 0_i32),
            Vector3::new( 1_i32, -1_i32, 0_i32),
            Vector3::new( 0_i32,  0_i32, 0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zx_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zx(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Point3::new( 1_i32,  1_i32,  1_i32 + shear_factor),
            Point3::new(-1_i32,  1_i32,  1_i32 - shear_factor),
            Point3::new(-1_i32, -1_i32,  1_i32 - shear_factor),
            Point3::new( 1_i32, -1_i32,  1_i32 + shear_factor),
            Point3::new( 1_i32,  1_i32, -1_i32 + shear_factor),
            Point3::new(-1_i32,  1_i32, -1_i32 - shear_factor),
            Point3::new(-1_i32, -1_i32, -1_i32 - shear_factor),
            Point3::new( 1_i32, -1_i32, -1_i32 + shear_factor),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zx_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zx(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32,  1_i32 + shear_factor),
            Vector3::new(-1_i32,  1_i32,  1_i32 - shear_factor),
            Vector3::new(-1_i32, -1_i32,  1_i32 - shear_factor),
            Vector3::new( 1_i32, -1_i32,  1_i32 + shear_factor),
            Vector3::new( 1_i32,  1_i32, -1_i32 + shear_factor),
            Vector3::new(-1_i32,  1_i32, -1_i32 - shear_factor),
            Vector3::new(-1_i32, -1_i32, -1_i32 - shear_factor),
            Vector3::new( 1_i32, -1_i32, -1_i32 + shear_factor),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zx_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zx(shear_factor);
        let vertices = [
            Point3::new(0_i32,  1_i32,  1_i32),
            Point3::new(0_i32, -1_i32,  1_i32),
            Point3::new(0_i32, -1_i32, -1_i32),
            Point3::new(0_i32,  1_i32, -1_i32),
            Point3::new(0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zx_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zx(shear_factor);
        let vertices = [
            Vector3::new(0_i32,  1_i32,  1_i32),
            Vector3::new(0_i32, -1_i32,  1_i32),
            Vector3::new(0_i32, -1_i32, -1_i32),
            Vector3::new(0_i32,  1_i32, -1_i32),
            Vector3::new(0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zy_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zy(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Point3::new( 1_i32,  1_i32,  1_i32 + shear_factor),
            Point3::new(-1_i32,  1_i32,  1_i32 + shear_factor),
            Point3::new(-1_i32, -1_i32,  1_i32 - shear_factor),
            Point3::new( 1_i32, -1_i32,  1_i32 - shear_factor),
            Point3::new( 1_i32,  1_i32, -1_i32 + shear_factor),
            Point3::new(-1_i32,  1_i32, -1_i32 + shear_factor),
            Point3::new(-1_i32, -1_i32, -1_i32 - shear_factor),
            Point3::new( 1_i32, -1_i32, -1_i32 - shear_factor),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zy_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32,  1_i32 + shear_factor),
            Vector3::new(-1_i32,  1_i32,  1_i32 + shear_factor),
            Vector3::new(-1_i32, -1_i32,  1_i32 - shear_factor),
            Vector3::new( 1_i32, -1_i32,  1_i32 - shear_factor),
            Vector3::new( 1_i32,  1_i32, -1_i32 + shear_factor),
            Vector3::new(-1_i32,  1_i32, -1_i32 + shear_factor),
            Vector3::new(-1_i32, -1_i32, -1_i32 - shear_factor),
            Vector3::new( 1_i32, -1_i32, -1_i32 - shear_factor),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zy_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zy(shear_factor);
        let vertices = [
            Point3::new( 1_i32, 0_i32,  1_i32),
            Point3::new(-1_i32, 0_i32,  1_i32),
            Point3::new(-1_i32, 0_i32, -1_i32),
            Point3::new( 1_i32, 0_i32, -1_i32),
            Point3::new( 0_i32, 0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zy_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32, 0_i32,  1_i32),
            Vector3::new(-1_i32, 0_i32,  1_i32),
            Vector3::new(-1_i32, 0_i32, -1_i32),
            Vector3::new( 1_i32, 0_i32, -1_i32),
            Vector3::new( 0_i32, 0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_xy() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let expected = Shear3::from_shear_xy(shear_factor);
        let result = Shear3::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_xz() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let expected = Shear3::from_shear_xz(shear_factor);
        let result = Shear3::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_yx() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let expected = Shear3::from_shear_yx(shear_factor);
        let result = Shear3::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_yz() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let expected = Shear3::from_shear_yz(shear_factor);
        let result = Shear3::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_zx() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_x());
        let expected = Shear3::from_shear_zx(shear_factor);
        let result = Shear3::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_from_shear_zy() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let expected = Shear3::from_shear_zy(shear_factor);
        let result = Shear3::from_shear(shear_factor, &direction, &normal);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_identity_shear_point() {
        let shear = Shear3::identity();
        let vertices = [
            Point3::new( 0_f64 , 0_f64,  0_f64),
            Point3::new( 0_f64,  1_f64,  0_f64),
            Point3::new( 1_f64,  0_f64,  0_f64),
            Point3::new( 1_f64,  1_f64,  0_f64),
            Point3::new( 0_f64, -1_f64,  0_f64),
            Point3::new(-1_f64,  0_f64,  0_f64),
            Point3::new(-1_f64, -1_f64,  0_f64),
            Point3::new( 1_f64, -1_f64,  0_f64),
            Point3::new(-1_f64,  1_f64,  0_f64),
            Point3::new( 0_f64 , 0_f64,  1_f64),
            Point3::new( 0_f64,  1_f64,  1_f64),
            Point3::new( 1_f64,  0_f64,  1_f64),
            Point3::new( 1_f64,  1_f64,  1_f64),
            Point3::new( 0_f64, -1_f64,  1_f64),
            Point3::new(-1_f64,  0_f64,  1_f64),
            Point3::new(-1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64, -1_f64,  1_f64),
            Point3::new(-1_f64,  1_f64,  1_f64),
            Point3::new( 0_f64 , 0_f64, -1_f64),
            Point3::new( 0_f64,  1_f64, -1_f64),
            Point3::new( 1_f64,  0_f64, -1_f64),
            Point3::new( 1_f64,  1_f64, -1_f64),
            Point3::new( 0_f64, -1_f64, -1_f64),
            Point3::new(-1_f64,  0_f64, -1_f64),
            Point3::new(-1_f64, -1_f64, -1_f64),
            Point3::new( 1_f64, -1_f64, -1_f64),
            Point3::new(-1_f64,  1_f64, -1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_identity_shear_vector() {
        let shear = Shear3::identity();
        let vertices = [
            Vector3::new( 0_f64 , 0_f64,  0_f64),
            Vector3::new( 0_f64,  1_f64,  0_f64),
            Vector3::new( 1_f64,  0_f64,  0_f64),
            Vector3::new( 1_f64,  1_f64,  0_f64),
            Vector3::new( 0_f64, -1_f64,  0_f64),
            Vector3::new(-1_f64,  0_f64,  0_f64),
            Vector3::new(-1_f64, -1_f64,  0_f64),
            Vector3::new( 1_f64, -1_f64,  0_f64),
            Vector3::new(-1_f64,  1_f64,  0_f64),
            Vector3::new( 0_f64 , 0_f64,  1_f64),
            Vector3::new( 0_f64,  1_f64,  1_f64),
            Vector3::new( 1_f64,  0_f64,  1_f64),
            Vector3::new( 1_f64,  1_f64,  1_f64),
            Vector3::new( 0_f64, -1_f64,  1_f64),
            Vector3::new(-1_f64,  0_f64,  1_f64),
            Vector3::new(-1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64, -1_f64,  1_f64),
            Vector3::new(-1_f64,  1_f64,  1_f64),
            Vector3::new( 0_f64 , 0_f64, -1_f64),
            Vector3::new( 0_f64,  1_f64, -1_f64),
            Vector3::new( 1_f64,  0_f64, -1_f64),
            Vector3::new( 1_f64,  1_f64, -1_f64),
            Vector3::new( 0_f64, -1_f64, -1_f64),
            Vector3::new(-1_f64,  0_f64, -1_f64),
            Vector3::new(-1_f64, -1_f64, -1_f64),
            Vector3::new( 1_f64, -1_f64, -1_f64),
            Vector3::new(-1_f64,  1_f64, -1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_xy_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = 4_f64;
        let result = shear.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_xz_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = 4_f64;
        let result = shear.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_yx_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = 4_f64;
        let result = shear.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_yz_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = 4_f64;
        let result = shear.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_zx_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = 4_f64;
        let result = shear.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_zy_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = 4_f64;
        let result = shear.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::new(
            -f64::sqrt(4_f64 / 10_f64),
            -f64::sqrt(1_f64 / 10_f64),
             f64::sqrt(5_f64 / 10_f64),
        ));
        let normal = Unit::from_value(Vector3::new(
            f64::sqrt(4_f64 / 10_f64),
            f64::sqrt(1_f64 / 10_f64),
            f64::sqrt(5_f64 / 10_f64),
        ));
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = 4_f64;
        let result = shear.to_affine_matrix().trace();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod shear3_inverse_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Point3,
        Unit,
        Vector3,
    };
    use cglinalg_transform::Shear3;

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_inverse_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xy(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Point3::new( 1_i32 - shear_factor,  1_i32,  1_i32),
            Point3::new(-1_i32 - shear_factor,  1_i32,  1_i32),
            Point3::new(-1_i32 + shear_factor, -1_i32,  1_i32),
            Point3::new( 1_i32 + shear_factor, -1_i32,  1_i32),
            Point3::new( 1_i32 - shear_factor,  1_i32, -1_i32),
            Point3::new(-1_i32 - shear_factor,  1_i32, -1_i32),
            Point3::new(-1_i32 + shear_factor, -1_i32, -1_i32),
            Point3::new( 1_i32 + shear_factor, -1_i32, -1_i32),
        ];
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_inverse_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32 - shear_factor,  1_i32,  1_i32),
            Vector3::new(-1_i32 - shear_factor,  1_i32,  1_i32),
            Vector3::new(-1_i32 + shear_factor, -1_i32,  1_i32),
            Vector3::new( 1_i32 + shear_factor, -1_i32,  1_i32),
            Vector3::new( 1_i32 - shear_factor,  1_i32, -1_i32),
            Vector3::new(-1_i32 - shear_factor,  1_i32, -1_i32),
            Vector3::new(-1_i32 + shear_factor, -1_i32, -1_i32),
            Vector3::new( 1_i32 + shear_factor, -1_i32, -1_i32),
        ];
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_inverse_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xy(shear_factor);
        let vertices = [
            Point3::new( 1_i32, 0_i32,  1_i32),
            Point3::new(-1_i32, 0_i32,  1_i32),
            Point3::new(-1_i32, 0_i32, -1_i32),
            Point3::new( 1_i32, 0_i32, -1_i32),
            Point3::new( 0_i32, 0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xy_inverse_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32, 0_i32,  1_i32),
            Vector3::new(-1_i32, 0_i32,  1_i32),
            Vector3::new(-1_i32, 0_i32, -1_i32),
            Vector3::new( 1_i32, 0_i32, -1_i32),
            Vector3::new( 0_i32, 0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xz_inverse_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xz(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Point3::new( 1_i32 - shear_factor,  1_i32,  1_i32),
            Point3::new(-1_i32 - shear_factor,  1_i32,  1_i32),
            Point3::new(-1_i32 - shear_factor, -1_i32,  1_i32),
            Point3::new( 1_i32 - shear_factor, -1_i32,  1_i32),
            Point3::new( 1_i32 + shear_factor,  1_i32, -1_i32),
            Point3::new(-1_i32 + shear_factor,  1_i32, -1_i32),
            Point3::new(-1_i32 + shear_factor, -1_i32, -1_i32),
            Point3::new( 1_i32 + shear_factor, -1_i32, -1_i32),
        ];
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xz_inverse_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xz(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32 - shear_factor,  1_i32,  1_i32),
            Vector3::new(-1_i32 - shear_factor,  1_i32,  1_i32),
            Vector3::new(-1_i32 - shear_factor, -1_i32,  1_i32),
            Vector3::new( 1_i32 - shear_factor, -1_i32,  1_i32),
            Vector3::new( 1_i32 + shear_factor,  1_i32, -1_i32),
            Vector3::new(-1_i32 + shear_factor,  1_i32, -1_i32),
            Vector3::new(-1_i32 + shear_factor, -1_i32, -1_i32),
            Vector3::new( 1_i32 + shear_factor, -1_i32, -1_i32),
        ];
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));
        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xz_inverse_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xz(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  0_i32),
            Point3::new(-1_i32,  1_i32,  0_i32),
            Point3::new(-1_i32, -1_i32,  0_i32),
            Point3::new( 1_i32, -1_i32,  0_i32),
            Point3::new( 0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_xz_inverse_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_xz(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  0_i32),
            Vector3::new(-1_i32,  1_i32,  0_i32),
            Vector3::new(-1_i32, -1_i32,  0_i32),
            Vector3::new( 1_i32, -1_i32,  0_i32),
            Vector3::new( 0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_inverse_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yx(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Point3::new( 1_i32,  1_i32 - shear_factor,  1_i32),
            Point3::new(-1_i32,  1_i32 + shear_factor,  1_i32),
            Point3::new(-1_i32, -1_i32 + shear_factor,  1_i32),
            Point3::new( 1_i32, -1_i32 - shear_factor,  1_i32),
            Point3::new( 1_i32,  1_i32 - shear_factor, -1_i32),
            Point3::new(-1_i32,  1_i32 + shear_factor, -1_i32),
            Point3::new(-1_i32, -1_i32 + shear_factor, -1_i32),
            Point3::new( 1_i32, -1_i32 - shear_factor, -1_i32),
        ];
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_inverse_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yx(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32 - shear_factor,  1_i32),
            Vector3::new(-1_i32,  1_i32 + shear_factor,  1_i32),
            Vector3::new(-1_i32, -1_i32 + shear_factor,  1_i32),
            Vector3::new( 1_i32, -1_i32 - shear_factor,  1_i32),
            Vector3::new( 1_i32,  1_i32 - shear_factor, -1_i32),
            Vector3::new(-1_i32,  1_i32 + shear_factor, -1_i32),
            Vector3::new(-1_i32, -1_i32 + shear_factor, -1_i32),
            Vector3::new( 1_i32, -1_i32 - shear_factor, -1_i32),
        ];
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_inverse_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yx(shear_factor);
        let vertices = [
            Point3::new(0_i32,  1_i32,  1_i32),
            Point3::new(0_i32, -1_i32,  1_i32),
            Point3::new(0_i32,  1_i32, -1_i32),
            Point3::new(0_i32, -1_i32, -1_i32),
            Point3::new(0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yx_inverse_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yx(shear_factor);
        let vertices = [
            Vector3::new(0_i32,  1_i32,  1_i32),
            Vector3::new(0_i32, -1_i32,  1_i32),
            Vector3::new(0_i32,  1_i32, -1_i32),
            Vector3::new(0_i32, -1_i32, -1_i32),
            Vector3::new(0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yz_inverse_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yz(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Point3::new( 1_i32,  1_i32 - shear_factor,  1_i32),
            Point3::new(-1_i32,  1_i32 - shear_factor,  1_i32),
            Point3::new(-1_i32, -1_i32 - shear_factor,  1_i32),
            Point3::new( 1_i32, -1_i32 - shear_factor,  1_i32),
            Point3::new( 1_i32,  1_i32 + shear_factor, -1_i32),
            Point3::new(-1_i32,  1_i32 + shear_factor, -1_i32),
            Point3::new(-1_i32, -1_i32 + shear_factor, -1_i32),
            Point3::new( 1_i32, -1_i32 + shear_factor, -1_i32),
        ];
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yz_inverse_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yz(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32 - shear_factor,  1_i32),
            Vector3::new(-1_i32,  1_i32 - shear_factor,  1_i32),
            Vector3::new(-1_i32, -1_i32 - shear_factor,  1_i32),
            Vector3::new( 1_i32, -1_i32 - shear_factor,  1_i32),
            Vector3::new( 1_i32,  1_i32 + shear_factor, -1_i32),
            Vector3::new(-1_i32,  1_i32 + shear_factor, -1_i32),
            Vector3::new(-1_i32, -1_i32 + shear_factor, -1_i32),
            Vector3::new( 1_i32, -1_i32 + shear_factor, -1_i32),
        ];
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yz_inverse_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yz(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32, 0_i32),
            Point3::new(-1_i32,  1_i32, 0_i32),
            Point3::new(-1_i32, -1_i32, 0_i32),
            Point3::new( 1_i32, -1_i32, 0_i32),
            Point3::new( 0_i32,  0_i32, 0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_yz_inverse_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_yz(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32, 0_i32),
            Vector3::new(-1_i32,  1_i32, 0_i32),
            Vector3::new(-1_i32, -1_i32, 0_i32),
            Vector3::new( 1_i32, -1_i32, 0_i32),
            Vector3::new( 0_i32,  0_i32, 0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zx_inverse_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zx(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Point3::new( 1_i32,  1_i32,  1_i32 - shear_factor),
            Point3::new(-1_i32,  1_i32,  1_i32 + shear_factor),
            Point3::new(-1_i32, -1_i32,  1_i32 + shear_factor),
            Point3::new( 1_i32, -1_i32,  1_i32 - shear_factor),
            Point3::new( 1_i32,  1_i32, -1_i32 - shear_factor),
            Point3::new(-1_i32,  1_i32, -1_i32 + shear_factor),
            Point3::new(-1_i32, -1_i32, -1_i32 + shear_factor),
            Point3::new( 1_i32, -1_i32, -1_i32 - shear_factor),
        ];
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zx_inverse_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zx(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32,  1_i32 - shear_factor),
            Vector3::new(-1_i32,  1_i32,  1_i32 + shear_factor),
            Vector3::new(-1_i32, -1_i32,  1_i32 + shear_factor),
            Vector3::new( 1_i32, -1_i32,  1_i32 - shear_factor),
            Vector3::new( 1_i32,  1_i32, -1_i32 - shear_factor),
            Vector3::new(-1_i32,  1_i32, -1_i32 + shear_factor),
            Vector3::new(-1_i32, -1_i32, -1_i32 + shear_factor),
            Vector3::new( 1_i32, -1_i32, -1_i32 - shear_factor),
        ];
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zx_inverse_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zx(shear_factor);
        let vertices = [
            Point3::new(0_i32,  1_i32,  1_i32),
            Point3::new(0_i32, -1_i32,  1_i32),
            Point3::new(0_i32, -1_i32, -1_i32),
            Point3::new(0_i32,  1_i32, -1_i32),
            Point3::new(0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zx_inverse_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zx(shear_factor);
        let vertices = [
            Vector3::new(0_i32,  1_i32,  1_i32),
            Vector3::new(0_i32, -1_i32,  1_i32),
            Vector3::new(0_i32, -1_i32, -1_i32),
            Vector3::new(0_i32,  1_i32, -1_i32),
            Vector3::new(0_i32,  0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zy_inverse_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zy(shear_factor);
        let vertices = [
            Point3::new( 1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32,  1_i32,  1_i32),
            Point3::new(-1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32, -1_i32,  1_i32),
            Point3::new( 1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32,  1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Point3::new( 1_i32,  1_i32,  1_i32 - shear_factor),
            Point3::new(-1_i32,  1_i32,  1_i32 - shear_factor),
            Point3::new(-1_i32, -1_i32,  1_i32 + shear_factor),
            Point3::new( 1_i32, -1_i32,  1_i32 + shear_factor),
            Point3::new( 1_i32,  1_i32, -1_i32 - shear_factor),
            Point3::new(-1_i32,  1_i32, -1_i32 - shear_factor),
            Point3::new(-1_i32, -1_i32, -1_i32 + shear_factor),
            Point3::new( 1_i32, -1_i32, -1_i32 + shear_factor),
        ];
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zy_inverse_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32,  1_i32,  1_i32),
            Vector3::new(-1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32, -1_i32,  1_i32),
            Vector3::new( 1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32,  1_i32, -1_i32),
            Vector3::new(-1_i32, -1_i32, -1_i32),
            Vector3::new( 1_i32, -1_i32, -1_i32),
        ];
        let expected = [
            Vector3::new( 1_i32,  1_i32,  1_i32 - shear_factor),
            Vector3::new(-1_i32,  1_i32,  1_i32 - shear_factor),
            Vector3::new(-1_i32, -1_i32,  1_i32 + shear_factor),
            Vector3::new( 1_i32, -1_i32,  1_i32 + shear_factor),
            Vector3::new( 1_i32,  1_i32, -1_i32 - shear_factor),
            Vector3::new(-1_i32,  1_i32, -1_i32 - shear_factor),
            Vector3::new(-1_i32, -1_i32, -1_i32 + shear_factor),
            Vector3::new( 1_i32, -1_i32, -1_i32 + shear_factor),
        ];
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zy_inverse_shearing_plane_point() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zy(shear_factor);
        let vertices = [
            Point3::new( 1_i32, 0_i32,  1_i32),
            Point3::new(-1_i32, 0_i32,  1_i32),
            Point3::new(-1_i32, 0_i32, -1_i32),
            Point3::new( 1_i32, 0_i32, -1_i32),
            Point3::new( 0_i32, 0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_zy_inverse_shearing_plane_vector() {
        let shear_factor = 5_i32;
        let shear = Shear3::from_shear_zy(shear_factor);
        let vertices = [
            Vector3::new( 1_i32, 0_i32,  1_i32),
            Vector3::new(-1_i32, 0_i32,  1_i32),
            Vector3::new(-1_i32, 0_i32, -1_i32),
            Vector3::new( 1_i32, 0_i32, -1_i32),
            Vector3::new( 0_i32, 0_i32,  0_i32),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_inverse_from_shear_xy_inverse() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = Shear3::from_shear_xy(-shear_factor);
        let result = shear.inverse();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_inverse_from_shear_xz_inverse() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = Shear3::from_shear_xz(-shear_factor);
        let result = shear.inverse();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_inverse_from_shear_yx_inverse() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = Shear3::from_shear_yx(-shear_factor);
        let result = shear.inverse();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_inverse_from_shear_yz_inverse() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = Shear3::from_shear_yz(-shear_factor);
        let result = shear.inverse();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_inverse_from_shear_zx_inverse() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_x());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = Shear3::from_shear_zx(-shear_factor);
        let result = shear.inverse();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_inverse_from_shear_zy_inverse() {
        let shear_factor = 15_f64;
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = Shear3::from_shear_zy(-shear_factor);
        let result = shear.inverse();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_identity_shear_inverse_point() {
        let shear = Shear3::identity();
        let vertices = [
            Point3::new( 0_f64 , 0_f64,  0_f64),
            Point3::new( 0_f64,  1_f64,  0_f64),
            Point3::new( 1_f64,  0_f64,  0_f64),
            Point3::new( 1_f64,  1_f64,  0_f64),
            Point3::new( 0_f64, -1_f64,  0_f64),
            Point3::new(-1_f64,  0_f64,  0_f64),
            Point3::new(-1_f64, -1_f64,  0_f64),
            Point3::new( 1_f64, -1_f64,  0_f64),
            Point3::new(-1_f64,  1_f64,  0_f64),
            Point3::new( 0_f64 , 0_f64,  1_f64),
            Point3::new( 0_f64,  1_f64,  1_f64),
            Point3::new( 1_f64,  0_f64,  1_f64),
            Point3::new( 1_f64,  1_f64,  1_f64),
            Point3::new( 0_f64, -1_f64,  1_f64),
            Point3::new(-1_f64,  0_f64,  1_f64),
            Point3::new(-1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64, -1_f64,  1_f64),
            Point3::new(-1_f64,  1_f64,  1_f64),
            Point3::new( 0_f64 , 0_f64, -1_f64),
            Point3::new( 0_f64,  1_f64, -1_f64),
            Point3::new( 1_f64,  0_f64, -1_f64),
            Point3::new( 1_f64,  1_f64, -1_f64),
            Point3::new( 0_f64, -1_f64, -1_f64),
            Point3::new(-1_f64,  0_f64, -1_f64),
            Point3::new(-1_f64, -1_f64, -1_f64),
            Point3::new( 1_f64, -1_f64, -1_f64),
            Point3::new(-1_f64,  1_f64, -1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.inverse_apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_identity_shear_inverse_vector() {
        let shear = Shear3::identity();
        let vertices = [
            Vector3::new( 0_f64 , 0_f64,  0_f64),
            Vector3::new( 0_f64,  1_f64,  0_f64),
            Vector3::new( 1_f64,  0_f64,  0_f64),
            Vector3::new( 1_f64,  1_f64,  0_f64),
            Vector3::new( 0_f64, -1_f64,  0_f64),
            Vector3::new(-1_f64,  0_f64,  0_f64),
            Vector3::new(-1_f64, -1_f64,  0_f64),
            Vector3::new( 1_f64, -1_f64,  0_f64),
            Vector3::new(-1_f64,  1_f64,  0_f64),
            Vector3::new( 0_f64 , 0_f64,  1_f64),
            Vector3::new( 0_f64,  1_f64,  1_f64),
            Vector3::new( 1_f64,  0_f64,  1_f64),
            Vector3::new( 1_f64,  1_f64,  1_f64),
            Vector3::new( 0_f64, -1_f64,  1_f64),
            Vector3::new(-1_f64,  0_f64,  1_f64),
            Vector3::new(-1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64, -1_f64,  1_f64),
            Vector3::new(-1_f64,  1_f64,  1_f64),
            Vector3::new( 0_f64 , 0_f64, -1_f64),
            Vector3::new( 0_f64,  1_f64, -1_f64),
            Vector3::new( 1_f64,  0_f64, -1_f64),
            Vector3::new( 1_f64,  1_f64, -1_f64),
            Vector3::new( 0_f64, -1_f64, -1_f64),
            Vector3::new(-1_f64,  0_f64, -1_f64),
            Vector3::new(-1_f64, -1_f64, -1_f64),
            Vector3::new( 1_f64, -1_f64, -1_f64),
            Vector3::new(-1_f64,  1_f64, -1_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.inverse_apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_xy_inverse_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = 4_f64;
        let result = shear.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_xz_inverse_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = 4_f64;
        let result = shear.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_yx_inverse_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let expected = 4_f64;
        let result = shear.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_yz_inverse_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let shear_inv = shear.inverse();
        let expected = 4_f64;
        let result = shear_inv.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_zx_inverse_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let shear_inv = shear.inverse();
        let expected = 4_f64;
        let result = shear_inv.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_shear_zy_inverse_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let shear_inv = shear.inverse();
        let expected = 4_f64;
        let result = shear_inv.to_affine_matrix().trace();

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_shear_inverse_trace() {
        let shear_factor = 10_f64;
        let direction = Unit::from_value(Vector3::new(
            -f64::sqrt(4_f64 / 10_f64),
            -f64::sqrt(1_f64 / 10_f64),
             f64::sqrt(5_f64 / 10_f64),
        ));
        let normal = Unit::from_value(Vector3::new(
            f64::sqrt(4_f64 / 10_f64),
            f64::sqrt(1_f64 / 10_f64),
            f64::sqrt(5_f64 / 10_f64),
        ));
        let shear = Shear3::from_shear(shear_factor, &direction, &normal);
        let shear_inv = shear.inverse();
        let expected = 4_f64;
        let result = shear_inv.to_affine_matrix().trace();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod shear3_coordinate_plane_tests {
    use cglinalg_core::{
        Point3,
        Unit,
        Vector3,
    };
    use cglinalg_transform::Shear3;

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy_point() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new( 1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64, -1_f64, -1_f64),
            Point3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = [
            Point3::new( 1_f64 + shear_factor,  1_f64,  1_f64),
            Point3::new(-1_f64 + shear_factor,  1_f64,  1_f64),
            Point3::new(-1_f64 - shear_factor, -1_f64,  1_f64),
            Point3::new( 1_f64 - shear_factor, -1_f64,  1_f64),
            Point3::new( 1_f64 + shear_factor,  1_f64, -1_f64),
            Point3::new(-1_f64 + shear_factor,  1_f64, -1_f64),
            Point3::new(-1_f64 - shear_factor, -1_f64, -1_f64),
            Point3::new( 1_f64 - shear_factor, -1_f64, -1_f64),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy_vector() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64, -1_f64, -1_f64),
            Vector3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = [
            Vector3::new( 1_f64 + shear_factor,  1_f64,  1_f64),
            Vector3::new(-1_f64 + shear_factor,  1_f64,  1_f64),
            Vector3::new(-1_f64 - shear_factor, -1_f64,  1_f64),
            Vector3::new( 1_f64 - shear_factor, -1_f64,  1_f64),
            Vector3::new( 1_f64 + shear_factor,  1_f64, -1_f64),
            Vector3::new(-1_f64 + shear_factor,  1_f64, -1_f64),
            Vector3::new(-1_f64 - shear_factor, -1_f64, -1_f64),
            Vector3::new( 1_f64 - shear_factor, -1_f64, -1_f64),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy_shearing_plane_point() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new( 1_f64, 0_f64,  1_f64),
            Point3::new(-1_f64, 0_f64,  1_f64),
            Point3::new(-1_f64, 0_f64, -1_f64),
            Point3::new( 1_f64, 0_f64, -1_f64),
            Point3::new( 0_f64, 0_f64,  0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xy_shearing_plane_vector() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64, 0_f64,  1_f64),
            Vector3::new(-1_f64, 0_f64,  1_f64),
            Vector3::new(-1_f64, 0_f64, -1_f64),
            Vector3::new( 1_f64, 0_f64, -1_f64),
            Vector3::new( 0_f64, 0_f64,  0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xz_point() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new( 1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64, -1_f64, -1_f64),
            Point3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = [
            Point3::new( 1_f64 + shear_factor,  1_f64,  1_f64),
            Point3::new(-1_f64 + shear_factor,  1_f64,  1_f64),
            Point3::new(-1_f64 + shear_factor, -1_f64,  1_f64),
            Point3::new( 1_f64 + shear_factor, -1_f64,  1_f64),
            Point3::new( 1_f64 - shear_factor,  1_f64, -1_f64),
            Point3::new(-1_f64 - shear_factor,  1_f64, -1_f64),
            Point3::new(-1_f64 - shear_factor, -1_f64, -1_f64),
            Point3::new( 1_f64 - shear_factor, -1_f64, -1_f64),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xz_vector() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64, -1_f64, -1_f64),
            Vector3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = [
            Vector3::new( 1_f64 + shear_factor,  1_f64,  1_f64),
            Vector3::new(-1_f64 + shear_factor,  1_f64,  1_f64),
            Vector3::new(-1_f64 + shear_factor, -1_f64,  1_f64),
            Vector3::new( 1_f64 + shear_factor, -1_f64,  1_f64),
            Vector3::new( 1_f64 - shear_factor,  1_f64, -1_f64),
            Vector3::new(-1_f64 - shear_factor,  1_f64, -1_f64),
            Vector3::new(-1_f64 - shear_factor, -1_f64, -1_f64),
            Vector3::new( 1_f64 - shear_factor, -1_f64, -1_f64),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xz_shearing_plane_point() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new( 1_f64,  1_f64,  0_f64),
            Point3::new(-1_f64,  1_f64,  0_f64),
            Point3::new(-1_f64, -1_f64,  0_f64),
            Point3::new( 1_f64, -1_f64,  0_f64),
            Point3::new( 0_f64,  0_f64,  0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_xz_shearing_plane_vector() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_x());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64,  1_f64,  0_f64),
            Vector3::new(-1_f64,  1_f64,  0_f64),
            Vector3::new(-1_f64, -1_f64,  0_f64),
            Vector3::new( 1_f64, -1_f64,  0_f64),
            Vector3::new( 0_f64,  0_f64,  0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx_point() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new( 1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64, -1_f64, -1_f64),
            Point3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = [
            Point3::new( 1_f64,  1_f64 + shear_factor,  1_f64),
            Point3::new(-1_f64,  1_f64 - shear_factor,  1_f64),
            Point3::new(-1_f64, -1_f64 - shear_factor,  1_f64),
            Point3::new( 1_f64, -1_f64 + shear_factor,  1_f64),
            Point3::new( 1_f64,  1_f64 + shear_factor, -1_f64),
            Point3::new(-1_f64,  1_f64 - shear_factor, -1_f64),
            Point3::new(-1_f64, -1_f64 - shear_factor, -1_f64),
            Point3::new( 1_f64, -1_f64 + shear_factor, -1_f64),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx_vector() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64, -1_f64, -1_f64),
            Vector3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = [
            Vector3::new( 1_f64,  1_f64 + shear_factor,  1_f64),
            Vector3::new(-1_f64,  1_f64 - shear_factor,  1_f64),
            Vector3::new(-1_f64, -1_f64 - shear_factor,  1_f64),
            Vector3::new( 1_f64, -1_f64 + shear_factor,  1_f64),
            Vector3::new( 1_f64,  1_f64 + shear_factor, -1_f64),
            Vector3::new(-1_f64,  1_f64 - shear_factor, -1_f64),
            Vector3::new(-1_f64, -1_f64 - shear_factor, -1_f64),
            Vector3::new( 1_f64, -1_f64 + shear_factor, -1_f64),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx_shearing_plane_point() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new(0_f64,  1_f64,  1_f64),
            Point3::new(0_f64, -1_f64,  1_f64),
            Point3::new(0_f64,  1_f64, -1_f64),
            Point3::new(0_f64, -1_f64, -1_f64),
            Point3::new(0_f64,  0_f64,  0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yx_shearing_plane_vector() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_x());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new(0_f64,  1_f64,  1_f64),
            Vector3::new(0_f64, -1_f64,  1_f64),
            Vector3::new(0_f64,  1_f64, -1_f64),
            Vector3::new(0_f64, -1_f64, -1_f64),
            Vector3::new(0_f64,  0_f64,  0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yz_point() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new( 1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64, -1_f64, -1_f64),
            Point3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = [
            Point3::new( 1_f64,  1_f64 + shear_factor,  1_f64),
            Point3::new(-1_f64,  1_f64 + shear_factor,  1_f64),
            Point3::new(-1_f64, -1_f64 + shear_factor,  1_f64),
            Point3::new( 1_f64, -1_f64 + shear_factor,  1_f64),
            Point3::new( 1_f64,  1_f64 - shear_factor, -1_f64),
            Point3::new(-1_f64,  1_f64 - shear_factor, -1_f64),
            Point3::new(-1_f64, -1_f64 - shear_factor, -1_f64),
            Point3::new( 1_f64, -1_f64 - shear_factor, -1_f64),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yz_vector() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64, -1_f64, -1_f64),
            Vector3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = [
            Vector3::new( 1_f64,  1_f64 + shear_factor,  1_f64),
            Vector3::new(-1_f64,  1_f64 + shear_factor,  1_f64),
            Vector3::new(-1_f64, -1_f64 + shear_factor,  1_f64),
            Vector3::new( 1_f64, -1_f64 + shear_factor,  1_f64),
            Vector3::new( 1_f64,  1_f64 - shear_factor, -1_f64),
            Vector3::new(-1_f64,  1_f64 - shear_factor, -1_f64),
            Vector3::new(-1_f64, -1_f64 - shear_factor, -1_f64),
            Vector3::new( 1_f64, -1_f64 - shear_factor, -1_f64),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yz_shearing_plane_point() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new( 1_f64,  1_f64, 0_f64),
            Point3::new(-1_f64,  1_f64, 0_f64),
            Point3::new(-1_f64, -1_f64, 0_f64),
            Point3::new( 1_f64, -1_f64, 0_f64),
            Point3::new( 0_f64,  0_f64, 0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_yz_shearing_plane_vector() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 2_f64, 0_f64);
        let direction = Unit::from_value(Vector3::unit_y());
        let normal = Unit::from_value(Vector3::unit_z());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64,  1_f64, 0_f64),
            Vector3::new(-1_f64,  1_f64, 0_f64),
            Vector3::new(-1_f64, -1_f64, 0_f64),
            Vector3::new( 1_f64, -1_f64, 0_f64),
            Vector3::new( 0_f64,  0_f64, 0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zx_point() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_x());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new( 1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64, -1_f64, -1_f64),
            Point3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = [
            Point3::new( 1_f64,  1_f64,  1_f64 + shear_factor),
            Point3::new(-1_f64,  1_f64,  1_f64 - shear_factor),
            Point3::new(-1_f64, -1_f64,  1_f64 - shear_factor),
            Point3::new( 1_f64, -1_f64,  1_f64 + shear_factor),
            Point3::new( 1_f64,  1_f64, -1_f64 + shear_factor),
            Point3::new(-1_f64,  1_f64, -1_f64 - shear_factor),
            Point3::new(-1_f64, -1_f64, -1_f64 - shear_factor),
            Point3::new( 1_f64, -1_f64, -1_f64 + shear_factor),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zx_vector() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_x());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64, -1_f64, -1_f64),
            Vector3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = [
            Vector3::new( 1_f64,  1_f64,  1_f64 + shear_factor),
            Vector3::new(-1_f64,  1_f64,  1_f64 - shear_factor),
            Vector3::new(-1_f64, -1_f64,  1_f64 - shear_factor),
            Vector3::new( 1_f64, -1_f64,  1_f64 + shear_factor),
            Vector3::new( 1_f64,  1_f64, -1_f64 + shear_factor),
            Vector3::new(-1_f64,  1_f64, -1_f64 - shear_factor),
            Vector3::new(-1_f64, -1_f64, -1_f64 - shear_factor),
            Vector3::new( 1_f64, -1_f64, -1_f64 + shear_factor),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zx_shearing_plane_point() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_x());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new(0_f64,  1_f64,  1_f64),
            Point3::new(0_f64, -1_f64,  1_f64),
            Point3::new(0_f64, -1_f64, -1_f64),
            Point3::new(0_f64,  1_f64, -1_f64),
            Point3::new(0_f64,  0_f64,  0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zx_shearing_plane_vector() {
        let shear_factor = 11_f64;
        let origin = Point3::new(0_f64, 2_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_x());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new(0_f64,  1_f64,  1_f64),
            Vector3::new(0_f64, -1_f64,  1_f64),
            Vector3::new(0_f64, -1_f64, -1_f64),
            Vector3::new(0_f64,  1_f64, -1_f64),
            Vector3::new(0_f64,  0_f64,  0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zy_point() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new( 1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64, -1_f64, -1_f64),
            Point3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = [
            Point3::new( 1_f64,  1_f64,  1_f64 + shear_factor),
            Point3::new(-1_f64,  1_f64,  1_f64 + shear_factor),
            Point3::new(-1_f64, -1_f64,  1_f64 - shear_factor),
            Point3::new( 1_f64, -1_f64,  1_f64 - shear_factor),
            Point3::new( 1_f64,  1_f64, -1_f64 + shear_factor),
            Point3::new(-1_f64,  1_f64, -1_f64 + shear_factor),
            Point3::new(-1_f64, -1_f64, -1_f64 - shear_factor),
            Point3::new( 1_f64, -1_f64, -1_f64 - shear_factor),
        ];
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zy_vector() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64,  1_f64,  1_f64),
            Vector3::new(-1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64, -1_f64,  1_f64),
            Vector3::new( 1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64,  1_f64, -1_f64),
            Vector3::new(-1_f64, -1_f64, -1_f64),
            Vector3::new( 1_f64, -1_f64, -1_f64),
        ];
        let expected = [
            Vector3::new( 1_f64,  1_f64,  1_f64 + shear_factor),
            Vector3::new(-1_f64,  1_f64,  1_f64 + shear_factor),
            Vector3::new(-1_f64, -1_f64,  1_f64 - shear_factor),
            Vector3::new( 1_f64, -1_f64,  1_f64 - shear_factor),
            Vector3::new( 1_f64,  1_f64, -1_f64 + shear_factor),
            Vector3::new(-1_f64,  1_f64, -1_f64 + shear_factor),
            Vector3::new(-1_f64, -1_f64, -1_f64 - shear_factor),
            Vector3::new( 1_f64, -1_f64, -1_f64 - shear_factor),
        ];
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zy_shearing_plane_point() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new( 1_f64, 0_f64,  1_f64),
            Point3::new(-1_f64, 0_f64,  1_f64),
            Point3::new(-1_f64, 0_f64, -1_f64),
            Point3::new( 1_f64, 0_f64, -1_f64),
            Point3::new( 0_f64, 0_f64,  0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|p| shear.apply_point(&p));

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_zy_shearing_plane_vector() {
        let shear_factor = 11_f64;
        let origin = Point3::new(2_f64, 0_f64, 2_f64);
        let direction = Unit::from_value(Vector3::unit_z());
        let normal = Unit::from_value(Vector3::unit_y());
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Vector3::new( 1_f64, 0_f64,  1_f64),
            Vector3::new(-1_f64, 0_f64,  1_f64),
            Vector3::new(-1_f64, 0_f64, -1_f64),
            Vector3::new( 1_f64, 0_f64, -1_f64),
            Vector3::new( 0_f64, 0_f64,  0_f64),
        ];
        let expected = vertices;
        let result = vertices.map(|v| shear.apply_vector(&v));

        assert_eq!(result, expected);
    }
}

/// Shearing along the plane `(1 / 2) * x + (1 / 3) * y - z + 1 == 0`
/// with origin `[2, 3, 3]`, direction `[2 / sqrt(17), 3 / sqrt(17), 2 / sqrt(17)]`,
/// and normal `[0, -2 / sqrt(13), 3 / sqrt(13)]`.
#[cfg(test)]
mod shear3_noncoordinate_plane_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix4x4,
        Point3,
        Unit,
        Vector3,
    };
    use cglinalg_transform::{
        Rotation3,
        Shear3,
        Translation3,
    };
    use cglinalg_trigonometry::{
        Angle,
        Radians,
    };

    fn shear_factor() -> f64 {
        11_f64
    }

    fn origin() -> Point3<f64> {
        Point3::new(2_f64, 3_f64, 3_f64)
    }

    #[rustfmt::skip]
    fn direction() -> Unit<Vector3<f64>> {
        Unit::from_value(Vector3::new(
            2_f64 / f64::sqrt(17_f64),
            3_f64 / f64::sqrt(17_f64),
            2_f64 / f64::sqrt(17_f64),
        ))
    }

    #[rustfmt::skip]
    fn normal() -> Unit<Vector3<f64>> {
        Unit::from_value(Vector3::new(
            0_f64,
           -2_f64 / f64::sqrt(13_f64),
            3_f64 / f64::sqrt(13_f64),
       ))
    }

    fn rotation_angle_x_yz() -> Radians<f64> {
        Radians(f64::atan2(2_f64 / 3_f64, 1_f64))
    }

    fn rotation_angle_z_xy() -> Radians<f64> {
        Radians(f64::atan2(13_f64 / (2_f64 * f64::sqrt(13_f64)), 1_f64))
    }

    fn translation() -> Translation3<f64> {
        Translation3::from_vector(&Vector3::new(0_f64, 0_f64, 1_f64))
    }

    fn translation_inv() -> Translation3<f64> {
        Translation3::from_vector(&Vector3::new(0_f64, 0_f64, -1_f64))
    }

    fn rotation_x_yz() -> Rotation3<f64> {
        let rotation_angle_x_yz = rotation_angle_x_yz();

        Rotation3::from_angle_x(rotation_angle_x_yz)
    }

    fn rotation_x_yz_inv() -> Rotation3<f64> {
        let rotation_angle_x_yz = rotation_angle_x_yz();

        Rotation3::from_angle_x(-rotation_angle_x_yz)
    }

    fn rotation_z_xy() -> Rotation3<f64> {
        let rotation_angle_z_xy = rotation_angle_z_xy();

        Rotation3::from_angle_z(rotation_angle_z_xy)
    }

    fn rotation_z_xy_inv() -> Rotation3<f64> {
        let rotation_angle_z_xy = rotation_angle_z_xy();

        Rotation3::from_angle_z(-rotation_angle_z_xy)
    }

    fn rotation() -> Rotation3<f64> {
        let rotation_x_yz = rotation_x_yz();
        let rotation_z_xy = rotation_z_xy();

        rotation_x_yz * rotation_z_xy
    }

    fn rotation_inv() -> Rotation3<f64> {
        let rotation_x_yz_inv = rotation_x_yz_inv();
        let rotation_z_xy_inv = rotation_z_xy_inv();

        rotation_z_xy_inv * rotation_x_yz_inv
    }

    #[rustfmt::skip]
    fn rotation_x_yz_matrix() -> Matrix4x4<f64> {
        Matrix4x4::new(
            1_f64,  0_f64,                     0_f64,                     0_f64,
            0_f64,  f64::sqrt(9_f64 / 13_f64), f64::sqrt(4_f64 / 13_f64), 0_f64,
            0_f64, -f64::sqrt(4_f64 / 13_f64), f64::sqrt(9_f64 / 13_f64), 0_f64,
            0_f64,  0_f64,                     0_f64,                     1_f64,
        )
    }

    #[rustfmt::skip]
    fn rotation_x_yz_inv_matrix() -> Matrix4x4<f64> {
        Matrix4x4::new(
            1_f64, 0_f64,                      0_f64,                     0_f64,
            0_f64, f64::sqrt(9_f64 / 13_f64), -f64::sqrt(4_f64 / 13_f64), 0_f64,
            0_f64, f64::sqrt(4_f64 / 13_f64),  f64::sqrt(9_f64 / 13_f64), 0_f64,
            0_f64, 0_f64,                      0_f64,                     1_f64,
        )
    }

    #[rustfmt::skip]
    fn rotation_z_xy_matrix() -> Matrix4x4<f64> {
        Matrix4x4::new(
            f64::sqrt(4_f64 / 17_f64),  f64::sqrt(13_f64 / 17_f64), 0_f64, 0_f64,
           -f64::sqrt(13_f64 / 17_f64), f64::sqrt(4_f64 / 17_f64),  0_f64, 0_f64,
            0_f64,                      0_f64,                      1_f64, 0_f64,
            0_f64,                      0_f64,                      0_f64, 1_f64,
        )
    }

    #[rustfmt::skip]
    fn rotation_z_xy_inv_matrix() -> Matrix4x4<f64> {
        Matrix4x4::new(
            f64::sqrt(4_f64 / 17_f64),  -f64::sqrt(13_f64 / 17_f64), 0_f64, 0_f64,
            f64::sqrt(13_f64 / 17_f64),  f64::sqrt(4_f64 / 17_f64),  0_f64, 0_f64,
            0_f64,                       0_f64,                      1_f64, 0_f64,
            0_f64,                       0_f64,                      0_f64, 1_f64,
        )
    }

    #[rustfmt::skip]
    fn rotation_matrix() -> Matrix4x4<f64> {
        let c0r0 = f64::sqrt(4_f64 / 17_f64);
        let c0r1 = f64::sqrt(9_f64 / 17_f64);
        let c0r2 = f64::sqrt(4_f64 / 17_f64);
        let c0r3 = 0_f64;

        let c1r0 = -f64::sqrt(13_f64 / 17_f64);
        let c1r1 = 6_f64 / f64::sqrt(221_f64);
        let c1r2 = 4_f64 / f64::sqrt(221_f64);
        let c1r3 = 0_f64;

        let c2r0 = 0_f64;
        let c2r1 = -f64::sqrt(4_f64 / 13_f64);
        let c2r2 = f64::sqrt(9_f64 / 13_f64);
        let c2r3 = 0_f64;

        let c3r0 = 0_f64;
        let c3r1 = 0_f64;
        let c3r2 = 0_f64;
        let c3r3 = 1_f64;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        )
    }

    #[rustfmt::skip]
    fn rotation_inv_matrix() -> Matrix4x4<f64> {
        let c0r0 = f64::sqrt(4_f64 / 17_f64);
        let c0r1 = -f64::sqrt(13_f64 / 17_f64);
        let c0r2 = 0_f64;
        let c0r3 = 0_f64;

        let c1r0 = f64::sqrt(9_f64 / 17_f64);
        let c1r1 = 6_f64 / f64::sqrt(221_f64);
        let c1r2 = -f64::sqrt(4_f64 / 13_f64);
        let c1r3 = 0_f64;

        let c2r0 = f64::sqrt(4_f64 / 17_f64);
        let c2r1 = 4_f64 / f64::sqrt(221_f64);
        let c2r2 = f64::sqrt(9_f64 / 13_f64);
        let c2r3 = 0_f64;

        let c3r0 = 0_f64;
        let c3r1 = 0_f64;
        let c3r2 = 0_f64;
        let c3r3 = 1_f64;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        )
    }

    fn shear_matrix_xz() -> Shear3<f64> {
        let shear_factor = shear_factor();

        Shear3::from_shear_xz(shear_factor)
    }

    #[rustfmt::skip]
    fn shear_matrix_xz_matrix() -> Matrix4x4<f64> {
        let shear_factor = shear_factor();

        Matrix4x4::new(
            1_f64,        0_f64, 0_f64, 0_f64,
            0_f64,        1_f64, 0_f64, 0_f64,
            shear_factor, 0_f64, 1_f64, 0_f64,
            0_f64,        0_f64, 0_f64, 1_f64,
        )
    }

    #[test]
    fn test_from_affine_shear_rotation_angle_x_yz() {
        let rotation_angle_x_yz = rotation_angle_x_yz();

        assert_relative_eq!(
            rotation_angle_x_yz.cos(),
            3_f64 / f64::sqrt(13_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
        assert_relative_eq!(
            rotation_angle_x_yz.sin(),
            2_f64 / f64::sqrt(13_f64),
            abs_diff_all <= 1e-10,
            relative_all <= f64::EPSILON
        );
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_rotation_angle_z_xy() {
        let rotation_angle_z_xy = rotation_angle_z_xy();

        assert_relative_eq!(rotation_angle_z_xy.cos(), f64::sqrt(4_f64 / 17_f64),  abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
        assert_relative_eq!(rotation_angle_z_xy.sin(), f64::sqrt(13_f64 / 17_f64), abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_translation_inv() {
        let translation = translation();
        let expected = translation_inv();
        let result = translation.inverse();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_affine_shear_rotation_x_yz() {
        let rotation_angle_x_yz = rotation_angle_x_yz();
        let expected = rotation_x_yz();
        let result = Rotation3::from_angle_x(rotation_angle_x_yz);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation_x_yz_inv() {
        let rotation_angle_x_yz = rotation_angle_x_yz();
        let expected = rotation_x_yz_inv();
        let result = Rotation3::from_angle_x(-rotation_angle_x_yz);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation_z_xy() {
        let rotation_angle_z_xy = rotation_angle_z_xy();
        let expected = rotation_z_xy();
        let result = Rotation3::from_angle_z(rotation_angle_z_xy);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation_z_xy_inv() {
        let rotation_angle_z_xy = rotation_angle_z_xy();
        let expected = rotation_z_xy_inv();
        let result = Rotation3::from_angle_z(-rotation_angle_z_xy);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation() {
        let expected = rotation_matrix();
        let rotation = rotation();
        let result = rotation.to_affine_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation_inv() {
        let expected = rotation_inv_matrix();
        let rotation_inv = rotation_inv();
        let result = rotation_inv.to_affine_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation_x_yz_matrix() {
        let rotation_angle_x_yz = rotation_angle_x_yz();
        let expected = rotation_x_yz_matrix();
        let rotation = Rotation3::from_angle_x(rotation_angle_x_yz);
        let result = rotation.to_affine_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation_x_yz_inv_matrix() {
        let rotation_angle_x_yz = rotation_angle_x_yz();
        let expected = rotation_x_yz_inv_matrix();
        let rotation = Rotation3::from_angle_x(-rotation_angle_x_yz);
        let result = rotation.to_affine_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation_z_xy_matrix() {
        let rotation_angle_z_xy = rotation_angle_z_xy();
        let expected = rotation_z_xy_matrix();
        let rotation = Rotation3::from_angle_z(rotation_angle_z_xy);
        let result = rotation.to_affine_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_rotation_z_xy_inv_matrix() {
        let rotation_angle_z_xy = rotation_angle_z_xy();
        let expected = rotation_z_xy_inv_matrix();
        let rotation = Rotation3::from_angle_z(-rotation_angle_z_xy);
        let result = rotation.to_affine_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_xz_matrix() {
        let shear = shear_matrix_xz();
        let expected = shear_matrix_xz_matrix();
        let result = shear.to_affine_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_origin_xz() {
        let translation = translation();
        let rotation = rotation();
        let origin = origin();
        let origin_xz = Point3::new(f64::sqrt(17_f64), 0_f64, 0_f64);
        let expected = origin;
        let result = translation * rotation * origin_xz;

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_direction_xz() {
        let direction = direction();
        let rotation_inv = rotation_inv();
        let expected = Vector3::unit_x();
        let result = {
            let _direction = direction.into_inner();
            rotation_inv * _direction
        };

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_normal_xz() {
        let normal = normal();
        let rotation_inv = rotation_inv();
        let expected = Vector3::unit_z();
        let result = {
            let _normal = normal.into_inner();
            rotation_inv * _normal
        };

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_coordinates_vertices() {
        let vertices = [
            Point3::new( 
                -f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                -f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                 f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                 f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                -f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                -f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                 f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                 f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
        ];
        let translation = translation();
        let rotation = rotation();
        let vertices_xz = [
            Point3::new( 1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64,  1_f64,  1_f64),
            Point3::new(-1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64, -1_f64,  1_f64),
            Point3::new( 1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64,  1_f64, -1_f64),
            Point3::new(-1_f64, -1_f64, -1_f64),
            Point3::new( 1_f64, -1_f64, -1_f64),
        ];
        let result = vertices_xz.map(|vertex_xz| translation * rotation * vertex_xz);

        assert_relative_eq!(result, vertices, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_vertices() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let vertices = [
            Point3::new( 
                -f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                -f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                 f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                 f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                -f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                 f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                -f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                -f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                 f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
            Point3::new(
                 f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64),
                 f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64),
                -f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64,
            ),
        ];
        let expected = [
            Point3::new(
                 f64::sqrt(4_f64 / 17_f64) - f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64) * shear_factor,
                -f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64) + f64::sqrt(9_f64 / 17_f64) * shear_factor,
                 f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64 + f64::sqrt(4_f64 / 17_f64) * shear_factor,
            ),
            Point3::new(
                -f64::sqrt(4_f64 / 17_f64) - f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64) * shear_factor,
                -f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64) + f64::sqrt(9_f64 / 17_f64) * shear_factor,
                 f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64 + f64::sqrt(4_f64 / 17_f64) * shear_factor,
            ),
            Point3::new(
                -f64::sqrt(4_f64 / 17_f64) + f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64) * shear_factor,
                -f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64) + f64::sqrt(9_f64 / 17_f64) * shear_factor,
                 f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64 + f64::sqrt(4_f64 / 17_f64) * shear_factor,
            ),
            Point3::new(
                 f64::sqrt(4_f64 / 17_f64) + f64::sqrt(13_f64 / 17_f64) + f64::sqrt(4_f64 / 17_f64) * shear_factor,
                -f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64) + f64::sqrt(9_f64 / 17_f64) * shear_factor,
                 f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64 + f64::sqrt(4_f64 / 17_f64) * shear_factor,
            ),
            Point3::new(
                 f64::sqrt(4_f64 / 17_f64) - f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64) - f64::sqrt(9_f64 / 17_f64) * shear_factor,
                -f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64 - f64::sqrt(4_f64 / 17_f64) * shear_factor,
            ),
            Point3::new(
                -f64::sqrt(4_f64 / 17_f64) - f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) + 6_f64 / f64::sqrt(221_f64) - f64::sqrt(9_f64 / 17_f64) * shear_factor,
                -f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) + 4_f64 / f64::sqrt(221_f64) + 1_f64 - f64::sqrt(4_f64 / 17_f64) * shear_factor,
            ),
            Point3::new(
                -f64::sqrt(4_f64 / 17_f64) + f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 f64::sqrt(4_f64 / 13_f64) - f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64) - f64::sqrt(9_f64 / 17_f64) * shear_factor,
                -f64::sqrt(9_f64 / 13_f64) - f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64 - f64::sqrt(4_f64 / 17_f64) * shear_factor,
            ),
            Point3::new(
                 f64::sqrt(4_f64 / 17_f64) + f64::sqrt(13_f64 / 17_f64) - f64::sqrt(4_f64 / 17_f64) * shear_factor,
                 f64::sqrt(4_f64 / 13_f64) + f64::sqrt(9_f64 / 17_f64) - 6_f64 / f64::sqrt(221_f64) - f64::sqrt(9_f64 / 17_f64) * shear_factor,
                -f64::sqrt(9_f64 / 13_f64) + f64::sqrt(4_f64 / 17_f64) - 4_f64 / f64::sqrt(221_f64) + 1_f64 - f64::sqrt(4_f64 / 17_f64) * shear_factor,
            ),
        ];
        let result = vertices.map(|v| shear * v);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[rustfmt::skip]
    #[test]
    fn test_from_affine_shear_matrix() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let expected = {
            let c0r0 = 1_f64;
            let c0r1 = 0_f64;
            let c0r2 = 0_f64;
            let c0r3 = 0_f64;

            let c1r0 = (-4_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c1r1 = 1_f64 - (6_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c1r2 = (-4_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c1r3 = 0_f64;

            let c2r0 = (6_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c2r1 = (9_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c2r2 = 1_f64 + (6_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c2r3 = 0_f64;

            let c3r0 = (-6_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c3r1 = (-9_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c3r2 = (-6_f64 / f64::sqrt(221_f64)) * shear_factor;
            let c3r3 = 1_f64;

            Matrix4x4::new(
                c0r0, c0r1, c0r2, c0r3,
                c1r0, c1r1, c1r2, c1r3,
                c2r0, c2r1, c2r2, c2r3,
                c3r0, c3r1, c3r2, c3r3
            )
        };
        let result = shear.to_affine_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_from_affine_shear_matrix_alternative_path() {
        let shear_factor = shear_factor();
        let origin = origin();
        let direction = direction();
        let normal = normal();
        let translation = translation();
        let translation_inv = translation_inv();
        let rotation = rotation();
        let rotation_inv = rotation_inv();
        let shear_matrix_xz = shear_matrix_xz();
        let shear = Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal);
        let expected = shear.to_transform();
        let result = {
            let isometry = (translation * rotation).to_transform();
            let isometry_inv = (rotation_inv * translation_inv).to_transform();
            let _shear_matrix_xz = shear_matrix_xz.to_transform();

            isometry * _shear_matrix_xz * isometry_inv
        };

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod shear3_trace_determinant_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Matrix4x4,
        Point3,
        Unit,
        Vector3,
    };
    use cglinalg_transform::Shear3;

    fn shear_factor() -> f64 {
        -372203_f64
    }

    #[rustfmt::skip]
    fn direction() -> Unit<Vector3<f64>> {
        Unit::from_value(Vector3::new(
            f64::sqrt(1_f64 / 3_f64),
            f64::sqrt(1_f64 / 3_f64),
            f64::sqrt(1_f64 / 3_f64),
        ))
    }

    #[rustfmt::skip]
    fn normal() -> Unit<Vector3<f64>> {
        Unit::from_value(Vector3::new(
             f64::sqrt(1_f64 / 6_f64),
             f64::sqrt(1_f64 / 6_f64),
            -f64::sqrt(4_f64 / 6_f64),
        ))
    }

    fn shear_transformation() -> Shear3<f64> {
        let shear_factor = shear_factor();
        let origin = Point3::new(1_f64, 1_f64, 1_f64);
        let direction = direction();
        let normal = normal();

        Shear3::from_affine_shear(shear_factor, &origin, &direction, &normal)
    }

    #[rustfmt::skip]
    fn expected_matrix() -> Matrix4x4<f64> {
        let shear_factor = shear_factor();

        let c0r0 = 1_f64 + f64::sqrt(1_f64 / 18_f64) * shear_factor;
        let c0r1 = f64::sqrt(1_f64 / 18_f64) * shear_factor;
        let c0r2 = f64::sqrt(1_f64 / 18_f64) * shear_factor;
        let c0r3 = 0_f64;

        let c1r0 = f64::sqrt(1_f64 / 18_f64) * shear_factor;
        let c1r1 = 1_f64 + f64::sqrt(1_f64 / 18_f64) * shear_factor;
        let c1r2 = f64::sqrt(1_f64 / 18_f64) * shear_factor;
        let c1r3 = 0_f64;

        let c2r0 = -f64::sqrt(4_f64 / 18_f64) * shear_factor;
        let c2r1 = -f64::sqrt(4_f64 / 18_f64) * shear_factor;
        let c2r2 = 1_f64 - f64::sqrt(4_f64 / 18_f64) * shear_factor;
        let c2r3 = 0_f64;

        let c3r0 = 0_f64;
        let c3r1 = 0_f64;
        let c3r2 = 0_f64;
        let c3r3 = 1_f64;

        Matrix4x4::new(
            c0r0, c0r1, c0r2, c0r3,
            c1r0, c1r1, c1r2, c1r3,
            c2r0, c2r1, c2r2, c2r3,
            c3r0, c3r1, c3r2, c3r3
        )
    }

    #[test]
    fn test_shear_matrix_direction_normal() {
        let direction = direction();
        let normal = normal();

        assert_relative_eq!(direction.dot(&normal), 0_f64, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_shear_matrix() {
        let expected = expected_matrix();
        let shear = shear_transformation();
        let result = shear.to_affine_matrix();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-15, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_shear_matrix_trace() {
        let shear = shear_transformation();
        let matrix = shear.to_affine_matrix();
        let expected = 4_f64;
        let result = matrix.trace();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    #[test]
    fn test_shear_expected_matrix_trace() {
        let matrix = expected_matrix();
        let expected = 4_f64;
        let result = matrix.trace();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    /// Mathematically, the shearing transformation has a determinant of `1`,
    /// but numerically the shearing transformation is not always well-behaved.
    #[test]
    fn test_shear_matrix_determinant() {
        let shear = shear_transformation();
        let matrix = shear.to_affine_matrix();
        let expected = 1.25_f64;
        let result = matrix.determinant();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }

    /// Mathematically, the shearing transformation has a determinant of `1`,
    /// but numerically the shearing transformation is not always well-behaved.
    #[test]
    fn test_shear_expected_matrix_determinant() {
        let matrix = expected_matrix();
        let expected = 0.75_f64;
        let result = matrix.determinant();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}
