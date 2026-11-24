import pytest
from openlens.transform import Matrix4x4, Vector3


class TestMatrix4x4FromTranslation:
    """Test Matrix4x4.from_translation static method."""
    
    def test_from_translation_basic(self):
        """Test creating a basic translation matrix."""
        mat = Matrix4x4.from_translation(1.0, 2.0, 3.0)
        assert mat.m[0][3] == 1.0
        assert mat.m[1][3] == 2.0
        assert mat.m[2][3] == 3.0
    
    def test_from_translation_zero(self):
        """Test translation matrix with zero values."""
        mat = Matrix4x4.from_translation(0.0, 0.0, 0.0)
        assert mat.m[0][3] == 0.0
        assert mat.m[1][3] == 0.0
        assert mat.m[2][3] == 0.0
    
    def test_from_translation_negative(self):
        """Test translation matrix with negative values."""
        mat = Matrix4x4.from_translation(-5.0, -10.0, -15.0)
        assert mat.m[0][3] == -5.0
        assert mat.m[1][3] == -10.0
        assert mat.m[2][3] == -15.0
    
    def test_from_translation_large_values(self):
        """Test translation matrix with large values."""
        mat = Matrix4x4.from_translation(1e6, 1e6, 1e6)
        assert mat.m[0][3] == 1e6
        assert mat.m[1][3] == 1e6
        assert mat.m[2][3] == 1e6
    
    def test_from_translation_preserves_identity_diagonal(self):
        """Test that translation matrix preserves identity in 3x3 part."""
        mat = Matrix4x4.from_translation(2.0, 3.0, 4.0)
        # Diagonal should be 1 for identity rotation/scale
        assert mat.m[0][0] == 1.0
        assert mat.m[1][1] == 1.0
        assert mat.m[2][2] == 1.0
        assert mat.m[3][3] == 1.0


class TestMatrix4x4MultiplyPoint:
    """Test Matrix4x4.multiply_point method."""
    
    def test_multiply_point_with_identity(self):
        """Test that identity matrix preserves the point."""
        mat = Matrix4x4()
        point = Vector3(1.0, 2.0, 3.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)
    
    def test_multiply_point_with_translation(self):
        """Test that translation is applied to points."""
        mat = Matrix4x4.from_translation(2.0, 3.0, 4.0)
        point = Vector3(1.0, 1.0, 1.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(3.0)
        assert result.y == pytest.approx(4.0)
        assert result.z == pytest.approx(5.0)
    
    def test_multiply_point_origin_translated(self):
        """Test that the origin point is translated to the translation vector."""
        mat = Matrix4x4.from_translation(2.0, 3.0, 4.0)
        point = Vector3(0.0, 0.0, 0.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(2.0)
        assert result.y == pytest.approx(3.0)
        assert result.z == pytest.approx(4.0)
    
    def test_multiply_point_with_negative_translation(self):
        """Test translation with negative displacement."""
        mat = Matrix4x4.from_translation(-1.0, -2.0, -3.0)
        point = Vector3(5.0, 5.0, 5.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(4.0)
        assert result.y == pytest.approx(3.0)
        assert result.z == pytest.approx(2.0)
    
    def test_multiply_point_sequential_translations(self):
        """Test applying sequential translations."""
        mat1 = Matrix4x4.from_translation(1.0, 2.0, 3.0)
        mat2 = Matrix4x4.from_translation(2.0, 3.0, 4.0)
        point = Vector3(0.0, 0.0, 0.0)
        result1 = mat1.multiply_point(point)
        result2 = mat2.multiply_point(result1)
        assert result2.x == pytest.approx(3.0)
        assert result2.y == pytest.approx(5.0)
        assert result2.z == pytest.approx(7.0)


class TestMatrix4x4MultiplyVector:
    """Test Matrix4x4.multiply_vector method."""
    
    def test_multiply_vector_with_identity(self):
        """Test that identity matrix preserves the vector."""
        mat = Matrix4x4()
        vector = Vector3(1.0, 2.0, 3.0)
        result = mat.multiply_vector(vector)
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)
    
    def test_multiply_vector_translation_ignored(self):
        """Test that translation is ignored for direction vectors."""
        mat = Matrix4x4.from_translation(10.0, 20.0, 30.0)
        vector = Vector3(1.0, 2.0, 3.0)
        result = mat.multiply_vector(vector)
        # Translation should not affect vectors
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)
    
    def test_multiply_vector_zero_vector(self):
        """Test that zero vector remains zero."""
        mat = Matrix4x4.from_translation(5.0, 5.0, 5.0)
        vector = Vector3(0.0, 0.0, 0.0)
        result = mat.multiply_vector(vector)
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)
    
    def test_multiply_point_vs_vector_translation_difference(self):
        """Test that points and vectors differ only in translation behavior."""
        mat = Matrix4x4.from_translation(5.0, 10.0, 15.0)
        point = Vector3(1.0, 2.0, 3.0)
        vector = Vector3(1.0, 2.0, 3.0)
        
        point_result = mat.multiply_point(point)
        vector_result = mat.multiply_vector(vector)
        
        # Point should be translated
        assert point_result.x == pytest.approx(6.0)
        assert point_result.y == pytest.approx(12.0)
        assert point_result.z == pytest.approx(18.0)
        
        # Vector should not be translated
        assert vector_result.x == pytest.approx(1.0)
        assert vector_result.y == pytest.approx(2.0)
        assert vector_result.z == pytest.approx(3.0)
