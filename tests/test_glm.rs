extern crate cglinalg;


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
        let result = glm::mat2([
            1_f32, 2_f32, 3_f32, 4_f32
        ]);

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
        let result = glm::mat3([
            1_f32, 2_f32, 3_f32, 
            4_f32, 5_f32, 6_f32, 
            7_f32, 8_f32, 9_f32
        ]);

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
        let result = glm::mat4([
            1_f32,  2_f32,  3_f32,  4_f32,  
            5_f32,  6_f32,  7_f32,  8_f32, 
            9_f32,  10_f32, 11_f32, 12_f32, 
            13_f32, 14_f32, 15_f32, 15_f32
        ]);

        assert_eq!(result, expected);
    }
}
