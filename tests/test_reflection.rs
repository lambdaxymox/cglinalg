extern crate cglinalg;


#[cfg(test)]
mod reflection2_tests {
    use cglinalg::{
        Reflection2,
        Point2,
        Vector2,
        Unit,
        Zero,
    };
    use approx::{
        relative_eq,
    };


    #[test]
    fn test_reflection_x_line_point1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_y());
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(1_f64, 1_f64);
        let expected = Point2::new(1_f64, -1_f64);
        let result = reflection.reflect_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_reflection_x_line_vector1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_y());
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(1_f64, 1_f64);
        let expected = Vector2::new(1_f64, -1_f64);
        let result = reflection.reflect_vector(&vector);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_reflection_y_line_point1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(1_f64, 1_f64);
        let expected = Point2::new(-1_f64, 1_f64);
        let result = reflection.reflect_point(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_reflection_y_line_vector1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::unit_x());
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(1_f64, 1_f64);
        let expected = Vector2::new(-1_f64, 1_f64);
        let result = reflection.reflect_vector(&vector);

        assert_eq!(result, expected);
    }

    /// Test the reflection through the line `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a line in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[test]
    fn test_reflection_line_through_origin_point1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / f64::sqrt(2_f64), 
             1_f64 / f64::sqrt(2_f64)
        ));
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(2_f64, 1_f64);
        let expected = Point2::new(1_f64, 2_f64);
        let result = reflection.reflect_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the line `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a line in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[test]
    fn test_reflection_line_through_origin_vector1() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / f64::sqrt(2_f64), 
             1_f64 / f64::sqrt(2_f64)
        ));
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(2_f64, 1_f64);
        let expected = Vector2::new(1_f64, 2_f64);
        let result = reflection.reflect_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the line `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a line in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[test]
    fn test_reflection_line_through_origin_point2() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(-Vector2::new(
            -1_f64 / f64::sqrt(2_f64), 
             1_f64 / f64::sqrt(2_f64)
        ));
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(2_f64, 1_f64);
        let expected = Point2::new(1_f64, 2_f64);
        let result = reflection.reflect_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the line `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a line in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[test]
    fn test_reflection_line_through_origin_vector2() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(-Vector2::new(
            -1_f64 / f64::sqrt(2_f64), 
             1_f64 / f64::sqrt(2_f64)
        ));
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(2_f64, 1_f64);
        let expected = Vector2::new(1_f64, 2_f64);
        let result = reflection.reflect_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the line `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a line in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[test]
    fn test_reflection_line_through_origin_point3() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / f64::sqrt(2_f64), 
             1_f64 / f64::sqrt(2_f64)
        ));
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(2_f64, 2_f64);
        let expected = Point2::new(2_f64, 2_f64);
        let result = reflection.reflect_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the line `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a line in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[test]
    fn test_reflection_line_through_origin_vector3() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / f64::sqrt(2_f64), 
             1_f64 / f64::sqrt(2_f64)
        ));
        let bias = Vector2::zero();
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(2_f64, 2_f64);
        let expected = Vector2::new(2_f64, 2_f64);
        let result = reflection.reflect_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the line `y = (1/2)*x + 1`.
    #[test]
    fn test_reflection_arbitrary_line_point() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / 2_f64, 
             1_f64
        ));
        let bias = Vector2::new(0_f64, 1_f64);
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let point = Point2::new(1_f64, 1_f64);
        let expected = Point2::new(3_f64 / 5_f64, 9_f64 / 5_f64);
        let result = reflection.reflect_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the line `y = (1/2)*x + 1.
    #[test]
    fn test_reflection_arbitrary_line_vector() {
        let normal: Unit<Vector2<f64>> = Unit::from_value(Vector2::new(
            -1_f64 / 2_f64, 
             1_f64
        ));
        let bias = Vector2::new(0_f64, 1_f64);
        let reflection = Reflection2::from_normal_bias(&normal, &bias);
        let vector = Vector2::new(1_f64, 1_f64);
        let expected = Vector2::new(7_f64 / 5_f64, 1_f64 / 5_f64);
        let result = reflection.reflect_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }
}

#[cfg(test)]
mod reflection3_tests {
    use cglinalg::{
        Reflection3,
        Point3,
        Vector3,
        Unit,
        Zero,
    };
    use approx::{
        relative_eq,
    };


    /// Test reflecting points through the plane `x = 0`.
    #[test]
    fn test_reflection_x_plane_point1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x());
        let bias = Vector3::zero();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(1_f64, 1_f64, 1_f64);
        let expected = Point3::new(-1_f64, 1_f64, 1_f64);
        let result = reflection.reflect_point(&point);

        assert_eq!(result, expected);
    }

    /// Test reflecting vectors through the plane `x = 0`.
    #[test]
    fn test_reflection_x_plane_vector1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_x());
        let bias = Vector3::zero();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(1_f64, 1_f64, 1_f64);
        let expected = Vector3::new(-1_f64, 1_f64, 1_f64);
        let result = reflection.reflect_vector(&vector);

        assert_eq!(result, expected);
    }

    /// Test reflecting points through the plane `y = 0`.
    #[test]
    fn test_reflection_y_plane_point1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y());
        let bias = Vector3::zero();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(1_f64, 1_f64, 1_f64);
        let expected = Point3::new(1_f64, -1_f64, 1_f64);
        let result = reflection.reflect_point(&point);

        assert_eq!(result, expected);
    }

    /// Test reflecting vectors through the plane `y = 0`.
    #[test]
    fn test_reflection_y_plane_vector1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_y());
        let bias = Vector3::zero();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(1_f64, 1_f64, 1_f64);
        let expected = Vector3::new(1_f64, -1_f64, 1_f64);
        let result = reflection.reflect_vector(&vector);

        assert_eq!(result, expected);
    }

    /// Test reflecting points through the plane `z = 0`.
    #[test]
    fn test_reflection_z_plane_point1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
        let bias = Vector3::zero();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(1_f64, 1_f64, 1_f64);
        let expected = Point3::new(1_f64, 1_f64, -1_f64);
        let result = reflection.reflect_point(&point);

        assert_eq!(result, expected);
    }

    /// Test reflecting vectors through the plane `z = 0`.
    #[test]
    fn test_reflection_z_plane_vector1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::unit_z());
        let bias = Vector3::zero();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(1_f64, 1_f64, 1_f64);
        let expected = Vector3::new(1_f64, 1_f64, -1_f64);
        let result = reflection.reflect_vector(&vector);

        assert_eq!(result, expected);
    }

    /// Test the reflection through the plane `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a plane in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[test]
    fn test_reflection_plane_through_origin_point1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / f64::sqrt(2_f64), 
            1_f64 / f64::sqrt(2_f64),
            0_f64
        ));
        let bias = Vector3::zero();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(2_f64, 1_f64, 1_f64);
        let expected = Point3::new(1_f64, 2_f64, 1_f64);
        let result = reflection.reflect_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the plane `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a plane in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[test]
    fn test_reflection_plane_through_origin_vector1() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / f64::sqrt(2_f64), 
            1_f64 / f64::sqrt(2_f64),
            0_f64
        ));
        let bias = Vector3::zero();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(2_f64, 1_f64, 1_f64);
        let expected = Vector3::new(1_f64, 2_f64, 1_f64);
        let result = reflection.reflect_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8)); 
    }

    /// Test the reflection through the plane `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a plane in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[test]
    fn test_reflection_plane_through_origin_point2() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(-Vector3::new(
            -1_f64 / f64::sqrt(2_f64), 
            1_f64 / f64::sqrt(2_f64),
            0_f64
        ));
        let bias = Vector3::zero();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(2_f64, 1_f64, 1_f64);
        let expected = Point3::new(1_f64, 2_f64, 1_f64);
        let result = reflection.reflect_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the plane `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a plane in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[test]
    fn test_reflection_plane_through_origin_vector2() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(-Vector3::new(
        -1_f64 / f64::sqrt(2_f64), 
         1_f64 / f64::sqrt(2_f64),
         0_f64
        ));
        let bias = Vector3::zero();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(2_f64, 1_f64, 1_f64);
        let expected = Vector3::new(1_f64, 2_f64, 1_f64);
        let result = reflection.reflect_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the plane `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a plane in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[test]
    fn test_reflection_plane_through_origin_point3() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
             -1_f64 / f64::sqrt(2_f64), 
             1_f64 / f64::sqrt(2_f64),
             0_f64
        ));
        let bias = Vector3::zero();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(2_f64, 2_f64, 1_f64);
        let expected = Point3::new(2_f64, 2_f64, 1_f64);
        let result = reflection.reflect_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the plane `y = x`.
    /// Note that there is an ambiguity in the choice of normal to a plane in
    /// two dimensions. We can choose either a normal vector or its negation
    /// to construct the reflection and get the same reflection.
    #[test]
    fn test_reflection_plane_through_origin_vector3() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / f64::sqrt(2_f64), 
            1_f64 / f64::sqrt(2_f64),
            0_f64
        ));
        let bias = Vector3::zero();
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(2_f64, 2_f64, 1_f64);
        let expected = Vector3::new(2_f64, 2_f64, 1_f64);
        let result = reflection.reflect_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the plane `y = (1/2)*x + 1`.
    #[test]
    fn test_reflection_arbitrary_plane_point() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / 2_f64, 
            1_f64,
            0_f64
        ));
        let bias = Vector3::new(0_f64, 1_f64, 0_f64);
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let point = Point3::new(1_f64, 1_f64, 20_f64);
        let expected = Point3::new(3_f64 / 5_f64, 9_f64 / 5_f64, 20_f64);
        let result = reflection.reflect_point(&point);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }

    /// Test the reflection through the plane `y = (1/2)*x + 1.
    #[test]
    fn test_reflection_arbitrary_plane_vector() {
        let normal: Unit<Vector3<f64>> = Unit::from_value(Vector3::new(
            -1_f64 / 2_f64, 
            1_f64,
            0_f64
        ));
        let bias = Vector3::new(0_f64, 1_f64, 0_f64);
        let reflection = Reflection3::from_normal_bias(&normal, &bias);
        let vector = Vector3::new(1_f64, 1_f64, 20_f64);
        let expected = Vector3::new(7_f64 / 5_f64, 1_f64 / 5_f64, 20_f64);
        let result = reflection.reflect_vector(&vector);

        assert!(relative_eq!(result, expected, epsilon = 1e-8));
    }
}

