extern crate cglinalg;


#[cfg(test)]
mod vector_constructor_tests {
    use cglinalg::glm;
    use cglinalg::{
        Vector1,
        Vector2,
        Vector3,
        Vector4,
    };


    #[test]
    fn test_vec1() {
        let expected = Vector1::new(1);
        let result = glm::vec1(1);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_vec2() {
        let expected = Vector2::new(1, 2);
        let result = glm::vec2(1, 2);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_vec3() {
        let expected = Vector3::new(1, 2, 3);
        let result = glm::vec3(1, 2, 3);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_vec4() {
        let expected = Vector4::new(1, 2, 3, 4);
        let result = glm::vec4(1, 2, 3, 4);

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod matrix_constructor_tests {
    use cglinalg::glm;
    use cglinalg::{
        Matrix2x2, 
        Matrix3x3, 
        Matrix4x4
    };


    #[rustfmt::skip]
    #[test]
    fn test_mat2() {
        let expected = Matrix2x2::new(
            1_f32, 2_f32, 
            3_f32, 4_f32
        );
        let result = glm::mat2(1_f32, 2_f32, 3_f32, 4_f32);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_mat3() {
        let expected = Matrix3x3::new(
            1_f32, 2_f32, 3_f32, 
            4_f32, 5_f32, 6_f32, 
            7_f32, 8_f32, 9_f32
        );
        let result = glm::mat3(
            1_f32, 2_f32, 3_f32, 
            4_f32, 5_f32, 6_f32, 
            7_f32, 8_f32, 9_f32
        );

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_mat4() {
        let expected = Matrix4x4::new(
            1_f32,  2_f32,  3_f32,  4_f32,  
            5_f32,  6_f32,  7_f32,  8_f32, 
            9_f32,  10_f32, 11_f32, 12_f32, 
            13_f32, 14_f32, 15_f32, 15_f32
        );
        let result = glm::mat4(
            1_f32,  2_f32,  3_f32,  4_f32,  
            5_f32,  6_f32,  7_f32,  8_f32, 
            9_f32,  10_f32, 11_f32, 12_f32, 
            13_f32, 14_f32, 15_f32, 15_f32
        );

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod quaternion_constructor_tests {
    use cglinalg::glm;
    use cglinalg::{
        Quaternion,
    };


    #[test]
    fn test_quat() {
        let expected = Quaternion::new(1, 2, 3 , 4);
        let result = glm::quat(1, 2, 3, 4);

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod vector_product_tests {
    use cglinalg::glm;
    use cglinalg::{
        CrossProduct,
        DotProduct,
        Vector3,
    };
    

    #[test]
    fn test_dot() {
        let v = Vector3::new(1, 2, 3);
        let w = Vector3::new(4, 5, 6);
        let expected = v.dot(w);
        let result = glm::dot(v, w);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_cross() {
        let v1 = Vector3::new(1, 2, 3);
        let v2 = Vector3::new(4, 5, 6);
        let expected = v1.cross(v2);
        let result = glm::cross(&v1, &v2);

        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod projection_tests {
    use cglinalg::glm;
    use cglinalg::{
        Degrees,
        Matrix4x4,
        OrthographicSpec,
        PerspectiveFovSpec,
    };


    #[test]
    fn test_orthogrpahic() {
        let near = 0.1;
        let far = 100.0;
        let left = -1.0;
        let right = 1.0;
        let top = 1.0;
        let bottom = -1.0;
        let spec = OrthographicSpec::new(left, right, bottom, top, near, far);

        let expected = Matrix4x4::from(spec);
        let result = glm::ortho(spec);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_perspective_fov() {
        let near = 0.1;
        let far = 100.0;
        let fovy = Degrees(67.0);
        let aspect = 1280 as f32 / 720 as f32;
        let spec = PerspectiveFovSpec::new(fovy, aspect, near, far);
        let expected = Matrix4x4::from(spec);
        let result = glm::perspective(fovy, aspect, near, far);

        assert_eq!(result, expected);
    }
}

