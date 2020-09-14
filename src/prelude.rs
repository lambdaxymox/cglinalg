
use structure::*;
use vector::*;
use matrix::*;
use quaternion::*;


#[cfg(test)]
mod tests {
    use matrix::{Matrix2, Matrix3, Matrix4};

    #[rustfmt::skip]
    #[test]
    fn test_mat2() {
        let expected = Matrix2::new(
            1_f32, 2_f32, 
            3_f32, 4_f32
        );
        let result = super::mat2([
            1_f32, 2_f32, 3_f32, 4_f32
        ]);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_mat3() {
        let expected = Matrix3::new(
            1_f32, 2_f32, 3_f32, 
            4_f32, 5_f32, 6_f32, 
            7_f32, 8_f32, 9_f32
        );
        let result = super::mat3([
            1_f32, 2_f32, 3_f32, 
            4_f32, 5_f32, 6_f32, 
            7_f32, 8_f32, 9_f32
        ]);

        assert_eq!(result, expected);
    }

    #[rustfmt::skip]
    #[test]
    fn test_mat4() {
        let expected = Matrix4::new(
            1_f32,  2_f32,  3_f32,  4_f32,  
            5_f32,  6_f32,  7_f32,  8_f32, 
            9_f32,  10_f32, 11_f32, 12_f32, 
            13_f32, 14_f32, 15_f32, 15_f32
        );
        let result = super::mat4([
            1_f32,  2_f32,  3_f32,  4_f32,  
            5_f32,  6_f32,  7_f32,  8_f32, 
            9_f32, 10_f32, 11_f32, 12_f32, 
            13_f32, 14_f32, 15_f32, 15_f32
        ]);

        assert_eq!(result, expected);
    }
}
