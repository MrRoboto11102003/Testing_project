import pytest
from openlens.transform import Matrix4x4
from openlens.vector3 import Vector3


class TestFromTranslation:
    """Tests for Matrix4x4.from_translation static method."""
    
    def test_from_translation_basic_structure(self):
        """Verify translation values placed in correct matrix positions."""
        mat = Matrix4x4.from_translation(5, -2, 3.5)
        
        # Translation in column 3
        assert mat.m[0][3] == 5
        assert mat.m[1][3] == -2
        assert mat.m[2][3] == 3.5
        
        # Rest is identity
        assert mat.m[0][0] == 1.0
        assert mat.m[1][1] == 1.0
        assert mat.m[2][2] == 1.0
        assert mat.m[3][3] == 1.0
        assert mat.m[0][1] == 0.0
        assert mat.m[1][0] == 0.0
    
    def test_from_translation_distinct_values_catch_index_errors(self):
        """Use distinct x,y,z to catch row/column swaps."""
        mat = Matrix4x4.from_translation(1, 2, 3)
        assert mat.m[0][3] == 1
        assert mat.m[1][3] == 2
        assert mat.m[2][3] == 3
    
    def test_from_translation_zero_equals_identity(self):
        """Zero translation should produce identity matrix."""
        mat = Matrix4x4.from_translation(0, 0, 0)
        identity = Matrix4x4()
        for i in range(4):
            for j in range(4):
                assert mat.m[i][j] == identity.m[i][j]
    
    def test_from_translation_returns_fresh_independent_objects(self):
        """Each call creates new object; mutations don't cross-pollinate."""
        mat1 = Matrix4x4.from_translation(1, 2, 3)
        mat2 = Matrix4x4.from_translation(1, 2, 3)
        
        assert mat1 is not mat2
        assert mat1.m[0] is not mat2.m[0]
        
        mat1.m[0][3] = 999
        assert mat2.m[0][3] == 1
    
    def test_from_translation_stores_negative_and_float_unchanged(self):
        """Verify exact storage of negative and float values."""
        mat = Matrix4x4.from_translation(-1.5, 0, 2.7)
        assert mat.m[0][3] == -1.5
        assert mat.m[1][3] == 0
        assert mat.m[2][3] == 2.7


class TestMultiplyPoint:
    """Tests for Matrix4x4.multiply_point method."""
    
    def test_multiply_point_identity_unchanged(self):
        """Identity matrix should return point unchanged."""
        mat = Matrix4x4()
        point = Vector3(1, 2, 3)
        result = mat.multiply_point(point)
        assert result.x == 1
        assert result.y == 2
        assert result.z == 3
    
    def test_multiply_point_translation_applied(self):
        """Translation matrix adds to point coordinates."""
        mat = Matrix4x4.from_translation(10, -5, 2.5)
        point = Vector3(1, 2, 3)
        result = mat.multiply_point(point)
        assert result.x == 11
        assert result.y == -3
        assert result.z == 5.5
    
    def test_multiply_point_translation_to_origin(self):
        """Translation applied to origin gives translation vector."""
        mat = Matrix4x4.from_translation(5, -3, 7)
        result = mat.multiply_point(Vector3(0, 0, 0))
        assert result.x == 5
        assert result.y == -3
        assert result.z == 7
    
    def test_multiply_point_non_symmetric_matrix_catches_transposition(self):
        """Non-symmetric matrix detects row/column bugs."""
        mat_data = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]
        mat = Matrix4x4(mat_data)
        result = mat.multiply_point(Vector3(1, 1, 1))
        # x = 1*1 + 2*1 + 3*1 + 4 = 10
        # y = 5*1 + 6*1 + 7*1 + 8 = 26
        # z = 9*1 + 10*1 + 11*1 + 12 = 42
        assert result.x == 10
        assert result.y == 26
        assert result.z == 42
    
    def test_multiply_point_specific_coordinates_verify_indexing(self):
        """Test with selective coordinates to verify correct access patterns."""
        mat_data = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [0, 0, 0, 1]
        ]
        mat = Matrix4x4(mat_data)
        result = mat.multiply_point(Vector3(1, 0, 0))
        # x = 1*1 + 2*0 + 3*0 + 4 = 5
        # y = 5*1 + 6*0 + 7*0 + 8 = 13
        # z = 9*1 + 10*0 + 11*0 + 12 = 21
        assert result.x == 5
        assert result.y == 13
        assert result.z == 21
    
    def test_multiply_point_row_3_ignored_no_perspective_divide(self):
        """Row 3 of matrix is ignored; no perspective division."""
        mat_data = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 2]
        ]
        mat = Matrix4x4(mat_data)
        result = mat.multiply_point(Vector3(2, 4, 6))
        # Row 3 ignored, so no perspective divide
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6
    
    def test_multiply_point_returns_new_vector(self):
        """Result is a new Vector3 object, not the input."""
        mat = Matrix4x4()
        point = Vector3(1, 2, 3)
        result = mat.multiply_point(point)
        assert result is not point


class TestMultiplyVector:
    """Tests for Matrix4x4.multiply_vector method."""
    
    def test_multiply_vector_identity_unchanged(self):
        """Identity matrix returns vector unchanged."""
        mat = Matrix4x4()
        vec = Vector3(1, 2, 3)
        result = mat.multiply_vector(vec)
        assert result.x == 1
        assert result.y == 2
        assert result.z == 3
    
    def test_multiply_vector_translation_ignored(self):
        """Translation matrix does not affect direction vectors (w=0)."""
        mat = Matrix4x4.from_translation(10, 20, 30)
        vec = Vector3(1, 2, 3)
        result = mat.multiply_vector(vec)
        assert result.x == 1
        assert result.y == 2
        assert result.z == 3
    
    def test_multiply_vector_vs_multiply_point_translation_difference(self):
        """Vectors ignore translation; points include it."""
        mat = Matrix4x4.from_translation(5, 10, 15)
        
        result_vec = mat.multiply_vector(Vector3(1, 1, 1))
        result_point = mat.multiply_point(Vector3(1, 1, 1))
        
        # Vector unaffected
        assert result_vec.x == 1
        assert result_vec.y == 1
        assert result_vec.z == 1
        
        # Point affected by translation
        assert result_point.x == 6
        assert result_point.y == 11
        assert result_point.z == 16
    
    def test_multiply_vector_non_symmetric_matrix_linear_transformation(self):
        """Non-symmetric matrix applies correct 3x3 linear transformation."""
        mat_data = [
            [1, 2, 3, 10],
            [4, 5, 6, 20],
            [7, 8, 9, 30],
            [0, 0, 0, 1]
        ]
        mat = Matrix4x4(mat_data)
        result = mat.multiply_vector(Vector3(1, 2, 3))
        # x = 1*1 + 2*2 + 3*3 = 14
        # y = 4*1 + 5*2 + 6*3 = 32
        # z = 7*1 + 8*2 + 9*3 = 50
        assert result.x == 14
        assert result.y == 32
        assert result.z == 50
    
    def test_multiply_vector_translation_column_has_no_effect(self):
        """Large translation column should not appear in vector result."""
        mat_data = [
            [1, 2, 3, 999],
            [4, 5, 6, 999],
            [7, 8, 9, 999],
            [0, 0, 0, 1]
        ]
        mat = Matrix4x4(mat_data)
        result = mat.multiply_vector(Vector3(1, 1, 1))
        # Only 3x3 block used
        assert result.x == 6   # 1+2+3
        assert result.y == 15  # 4+5+6
        assert result.z == 24  # 7+8+9
    
    def test_multiply_vector_zero_vector_gives_zero(self):
        """Any matrix applied to zero vector yields zero vector."""
        mat_data = [
            [2, 3, 4, 10],
            [5, 6, 7, 20],
            [8, 9, 10, 30],
            [1, 1, 1, 1]
        ]
        mat = Matrix4x4(mat_data)
        result = mat.multiply_vector(Vector3(0, 0, 0))
        assert result.x == 0
        assert result.y == 0
        assert result.z == 0
    
    def test_multiply_vector_returns_new_vector(self):
        """Result is a new Vector3 object."""
        mat = Matrix4x4()
        vec = Vector3(1, 2, 3)
        result = mat.multiply_vector(vec)
        assert result is not vec
