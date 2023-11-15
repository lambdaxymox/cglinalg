extern crate cglinalg_transform;


#[cfg(test)]
mod reflection2_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Point2,
        Unit,
        Vector2,
    };
    use cglinalg_transform::Reflection2;


    #[rustfmt::skip]
    #[test]
    fn test_reflection_normal_x_basis_vectors() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
        let bias = Point2::origin();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let unit_x = Vector2::unit_x();
        let unit_y = Vector2::unit_y();

        assert_eq!(reflection.apply_vector(&unit_x),    -unit_x);
        assert_eq!(reflection.apply_vector(&(-unit_x)),  unit_x);
        assert_eq!(reflection.apply_vector(&unit_y),     unit_y);
        assert_eq!(reflection.apply_vector(&(-unit_y)), -unit_y);
    }

    #[rustfmt::skip]
    #[test]
    fn test_reflection_normal_y_basis_vectors() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_y());
        let bias = Point2::origin();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let unit_x = Vector2::unit_x();
        let unit_y = Vector2::unit_y();

        assert_eq!(reflection.apply_vector(&unit_x),     unit_x);
        assert_eq!(reflection.apply_vector(&(-unit_x)), -unit_x);
        assert_eq!(reflection.apply_vector(&unit_y),    -unit_y);
        assert_eq!(reflection.apply_vector(&(-unit_y)),  unit_y);
    }

    #[test]
    fn test_reflection_x_line_point1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_y());
        let bias = Point2::origin();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(1_f64, 1_f64);
        let expected = Point2::new(1_f64, -1_f64);
        let result = reflection.apply_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_reflection_x_line_vector1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_y());
        let bias = Point2::origin();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(1_f64, 1_f64);
        let expected = Vector2::new(1_f64, -1_f64);
        let result = reflection.apply_vector(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_reflection_y_line_point1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
        let bias = Point2::origin();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(1_f64, 1_f64);
        let expected = Point2::new(-1_f64, 1_f64);
        let result = reflection.apply_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_reflection_y_line_vector1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
        let bias = Point2::origin();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(1_f64, 1_f64);
        let expected = Vector2::new(-1_f64, 1_f64);
        let result = reflection.apply_vector(&vector);

        assert_eq!(result, expected);
    }

    /// Test the reflection through the line `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a line in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_line_through_origin_point1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / f64::sqrt(2_f64),
             1_f64 / f64::sqrt(2_f64),
        ));
        let bias = Point2::origin();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(2_f64, 1_f64);
        let expected = Point2::new(1_f64, 2_f64);
        let result = reflection.apply_point(&point);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the line `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a line in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_line_through_origin_vector1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / f64::sqrt(2_f64),
             1_f64 / f64::sqrt(2_f64),
        ));
        let bias = Point2::origin();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(2_f64, 1_f64);
        let expected = Vector2::new(1_f64, 2_f64);
        let result = reflection.apply_vector(&vector);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the line `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a line in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_line_through_origin_point2() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(-Vector2::new(
            -1_f64 / f64::sqrt(2_f64),
             1_f64 / f64::sqrt(2_f64),
        ));
        let bias = Point2::origin();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(2_f64, 1_f64);
        let expected = Point2::new(1_f64, 2_f64);
        let result = reflection.apply_point(&point);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the line `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a line in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_line_through_origin_vector2() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(-Vector2::new(
            -1_f64 / f64::sqrt(2_f64),
             1_f64 / f64::sqrt(2_f64),
        ));
        let bias = Point2::origin();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(2_f64, 1_f64);
        let expected = Vector2::new(1_f64, 2_f64);
        let result = reflection.apply_vector(&vector);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the line `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a line in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_line_through_origin_point3() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / f64::sqrt(2_f64),
             1_f64 / f64::sqrt(2_f64),
        ));
        let bias = Point2::origin();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(2_f64, 2_f64);
        let expected = Point2::new(2_f64, 2_f64);
        let result = reflection.apply_point(&point);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the line `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a line in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_line_through_origin_vector3() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / f64::sqrt(2_f64),
             1_f64 / f64::sqrt(2_f64),
        ));
        let bias = Point2::origin();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(2_f64, 2_f64);
        let expected = Vector2::new(2_f64, 2_f64);
        let result = reflection.apply_vector(&vector);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the line `y = (1/2)*x + 1`.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_arbitrary_line_point() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / 2_f64,
             1_f64,
        ));
        let bias = Point2::new(0_f64, 1_f64);
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(1_f64, 1_f64);
        let expected = Point2::new(3_f64 / 5_f64, 9_f64 / 5_f64);
        let result = reflection.apply_point(&point);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the line `y = (1/2)*x + 1.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_arbitrary_line_vector() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / 2_f64,
             1_f64,
        ));
        let bias = Point2::new(0_f64, 1_f64);
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(1_f64, 1_f64);
        let expected = Vector2::new(7_f64 / 5_f64, 1_f64 / 5_f64);
        let result = reflection.apply_vector(&vector);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the line `y = (1/2)*x + 1.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_arbitrary_line_matrix_determinant() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / 2_f64,
            1_f64,
        ));
        let bias = Point2::new(0_f64, 1_f64);
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let expected = -1_f64;
        let result = reflection.to_affine_matrix().determinant();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}

#[cfg(test)]
mod reflection3_tests {
    use approx_cmp::assert_relative_eq;
    use cglinalg_core::{
        Point3,
        Unit,
        Vector3,
    };
    use cglinalg_transform::Reflection3;


    #[rustfmt::skip]
    #[test]
    fn test_reflection_normal_x_basis_vectors() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x());
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let unit_x = Vector3::unit_x();
        let unit_y = Vector3::unit_y();
        let unit_z = Vector3::unit_z();

        assert_eq!(reflection.apply_vector(&unit_x),    -unit_x);
        assert_eq!(reflection.apply_vector(&(-unit_x)),  unit_x);
        assert_eq!(reflection.apply_vector(&unit_y),     unit_y);
        assert_eq!(reflection.apply_vector(&(-unit_y)), -unit_y);
        assert_eq!(reflection.apply_vector(&unit_z),     unit_z);
        assert_eq!(reflection.apply_vector(&(-unit_z)), -unit_z);
    }

    #[rustfmt::skip]
    #[test]
    fn test_reflection_normal_y_basis_vectors() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y());
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let unit_x = Vector3::unit_x();
        let unit_y = Vector3::unit_y();
        let unit_z = Vector3::unit_z();

        assert_eq!(reflection.apply_vector(&unit_x),     unit_x);
        assert_eq!(reflection.apply_vector(&(-unit_x)), -unit_x);
        assert_eq!(reflection.apply_vector(&unit_y),    -unit_y);
        assert_eq!(reflection.apply_vector(&(-unit_y)),  unit_y);
        assert_eq!(reflection.apply_vector(&unit_z),     unit_z);
        assert_eq!(reflection.apply_vector(&(-unit_z)), -unit_z);
    }

    #[rustfmt::skip]
    #[test]
    fn test_reflection_normal_z_basis_vectors() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let unit_x = Vector3::unit_x();
        let unit_y = Vector3::unit_y();
        let unit_z = Vector3::unit_z();

        assert_eq!(reflection.apply_vector(&unit_x),     unit_x);
        assert_eq!(reflection.apply_vector(&(-unit_x)), -unit_x);
        assert_eq!(reflection.apply_vector(&unit_y),     unit_y);
        assert_eq!(reflection.apply_vector(&(-unit_y)), -unit_y);
        assert_eq!(reflection.apply_vector(&unit_z),    -unit_z);
        assert_eq!(reflection.apply_vector(&(-unit_z)),  unit_z);
    }

    /// Test reflecting points through the plane `x = 0`.
    #[test]
    fn test_reflection_x_plane_point1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x());
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(1_f64, 1_f64, 1_f64);
        let expected = Point3::new(-1_f64, 1_f64, 1_f64);
        let result = reflection.apply_point(&point);

        assert_eq!(result, expected);
    }

    /// Test reflecting vectors through the plane `x = 0`.
    #[test]
    fn test_reflection_x_plane_vector1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x());
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(1_f64, 1_f64, 1_f64);
        let expected = Vector3::new(-1_f64, 1_f64, 1_f64);
        let result = reflection.apply_vector(&vector);

        assert_eq!(result, expected);
    }

    /// Test reflecting points through the plane `y = 0`.
    #[test]
    fn test_reflection_y_plane_point1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y());
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(1_f64, 1_f64, 1_f64);
        let expected = Point3::new(1_f64, -1_f64, 1_f64);
        let result = reflection.apply_point(&point);

        assert_eq!(result, expected);
    }

    /// Test reflecting vectors through the plane `y = 0`.
    #[test]
    fn test_reflection_y_plane_vector1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y());
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(1_f64, 1_f64, 1_f64);
        let expected = Vector3::new(1_f64, -1_f64, 1_f64);
        let result = reflection.apply_vector(&vector);

        assert_eq!(result, expected);
    }

    /// Test reflecting points through the plane `z = 0`.
    #[test]
    fn test_reflection_z_plane_point1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(1_f64, 1_f64, 1_f64);
        let expected = Point3::new(1_f64, 1_f64, -1_f64);
        let result = reflection.apply_point(&point);

        assert_eq!(result, expected);
    }

    /// Test reflecting vectors through the plane `z = 0`.
    #[test]
    fn test_reflection_z_plane_vector1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(1_f64, 1_f64, 1_f64);
        let expected = Vector3::new(1_f64, 1_f64, -1_f64);
        let result = reflection.apply_vector(&vector);

        assert_eq!(result, expected);
    }

    /// Test the reflection through the plane `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a plane in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_plane_through_origin_point1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / f64::sqrt(2_f64),
             1_f64 / f64::sqrt(2_f64),
             0_f64,
        ));
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(2_f64, 1_f64, 1_f64);
        let expected = Point3::new(1_f64, 2_f64, 1_f64);
        let result = reflection.apply_point(&point);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the plane `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a plane in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_plane_through_origin_vector1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / f64::sqrt(2_f64),
             1_f64 / f64::sqrt(2_f64),
             0_f64,
        ));
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(2_f64, 1_f64, 1_f64);
        let expected = Vector3::new(1_f64, 2_f64, 1_f64);
        let result = reflection.apply_vector(&vector);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON); 
    }

    /// Test the reflection through the plane `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a plane in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_plane_through_origin_point2() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(-Vector3::new(
            -1_f64 / f64::sqrt(2_f64),
             1_f64 / f64::sqrt(2_f64),
             0_f64,
        ));
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(2_f64, 1_f64, 1_f64);
        let expected = Point3::new(1_f64, 2_f64, 1_f64);
        let result = reflection.apply_point(&point);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the plane `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a plane in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_plane_through_origin_vector2() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(-Vector3::new(
            -1_f64 / f64::sqrt(2_f64),
             1_f64 / f64::sqrt(2_f64),
             0_f64,
        ));
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(2_f64, 1_f64, 1_f64);
        let expected = Vector3::new(1_f64, 2_f64, 1_f64);
        let result = reflection.apply_vector(&vector);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the plane `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a plane in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_plane_through_origin_point3() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
             -1_f64 / f64::sqrt(2_f64),
              1_f64 / f64::sqrt(2_f64),
              0_f64,
        ));
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(2_f64, 2_f64, 1_f64);
        let expected = Point3::new(2_f64, 2_f64, 1_f64);
        let result = reflection.apply_point(&point);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the plane `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a plane in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_plane_through_origin_vector3() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / f64::sqrt(2_f64),
             1_f64 / f64::sqrt(2_f64),
             0_f64,
        ));
        let bias = Point3::origin();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(2_f64, 2_f64, 1_f64);
        let expected = Vector3::new(2_f64, 2_f64, 1_f64);
        let result = reflection.apply_vector(&vector);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the plane `y = (1/2)*x + 1`.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_arbitrary_plane_point1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / 2_f64,
             1_f64,
             0_f64,
        ));
        let bias = Point3::new(0_f64, 1_f64, 0_f64);
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(1_f64, 1_f64, 20_f64);
        let expected = Point3::new(3_f64 / 5_f64, 9_f64 / 5_f64, 20_f64);
        let result = reflection.apply_point(&point);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the plane `y = (1/2)*x + 1.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_arbitrary_plane_vector1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / 2_f64,
             1_f64,
             0_f64,
        ));
        let bias = Point3::new(0_f64, 1_f64, 0_f64);
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(1_f64, 1_f64, 20_f64);
        let expected = Vector3::new(7_f64 / 5_f64, 1_f64 / 5_f64, 20_f64);
        let result = reflection.apply_vector(&vector);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the plane `(z - 2) + (y - 0) + (x - 0) == 0`.
    /// Note that this equation has a bias of `[0   0   2]^T`.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_arbitrary_plane_point2() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / f64::sqrt(3_f64),
            -1_f64 / f64::sqrt(3_f64),
             1_f64 / f64::sqrt(3_f64),
        ));
        let bias = Point3::new(0_f64, 0_f64, 2_f64);
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(1_f64, 1_f64, 1_f64);
        let expected = Point3::new(-1_f64, -1_f64, 3_f64);
        let result = reflection.apply_point(&point);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the plane `(z - 2) + (y - 0) + (x - 0) == 0`.
    /// Note that this equation has a bias of `[0   0   2]^T`.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_arbitrary_plane_vector2() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / f64::sqrt(3_f64),
            -1_f64 / f64::sqrt(3_f64),
             1_f64 / f64::sqrt(3_f64),
        ));
        let bias = Point3::new(0_f64, 0_f64, 2_f64);
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(1_f64, 1_f64, 1_f64);
        let expected = Vector3::new(1_f64 / 3_f64, 1_f64 / 3_f64, 5_f64 / 3_f64);
        let result = reflection.apply_vector(&vector);

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-8, relative_all <= f64::EPSILON);
    }

    /// Test the reflection through the plane `y = (1/2)*x + 1.
    #[rustfmt::skip]
    #[test]
    fn test_reflection_arbitrary_plane_matrix_determinant() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / 2_f64,
             1_f64,
             0_f64,
        ));
        let bias = Point3::new(0_f64, 1_f64, 0_f64);
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let expected = -1_f64;
        let result = reflection.to_affine_matrix().determinant();

        assert_relative_eq!(result, expected, abs_diff_all <= 1e-10, relative_all <= f64::EPSILON);
    }
}
