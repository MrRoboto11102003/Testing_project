import pytest
from openlens.transform import Matrix4x4, Vector3


class TestMatrix4x4FromTranslation:
    """Test Matrix4x4.from_translation static method."""
    
    def test_from_translation_basic(self):
        """Test creating a translation matrix with basic positive values."""
        mat = Matrix4x4.from_translation(1.0, 2.0, 3.0)
        assert mat.m[0][3] == 1.0
        assert mat.m[1][3] == 2.0
        assert mat.m[2][3] == 3.0
    
    def test_from_translation_zero(self):
        """Test creating a translation matrix with zero values."""
        mat = Matrix4x4.from_translation(0.0, 0.0, 0.0)
        assert mat.m[0][3] == 0.0
        assert mat.m[1][3] == 0.0
        assert mat.m[2][3] == 0.0
    
    def test_from_translation_negative(self):
        """Test creating a translation matrix with negative values."""
        mat = Matrix4x4.from_translation(-5.0, -10.0, -15.0)
        assert mat.m[0][3] == -5.0
        assert mat.m[1][3] == -10.0
        assert mat.m[2][3] == -15.0
    
    def test_from_translation_identity_rotation_scale(self):
        """Test that from_translation preserves identity for rotation/scale part."""
        mat = Matrix4x4.from_translation(2.0, 3.0, 4.0)
        # Check diagonal is 1 (from Matrix4x4 default constructor)
        assert mat.m[0][0] == 1.0
        assert mat.m[1][1] == 1.0
        assert mat.m[2][2] == 1.0
        assert mat.m[3][3] == 1.0


class TestMatrix4x4MultiplyPoint:
    """Test Matrix4x4.multiply_point method."""
    
    def test_multiply_point_identity(self):
        """Test multiplying a point by identity matrix is unchanged."""
        mat = Matrix4x4()  # Default constructor should be identity
        point = Vector3(1.0, 2.0, 3.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)
    
    def test_multiply_point_with_translation(self):
        """Test that translation matrix translates a point correctly."""
        mat = Matrix4x4.from_translation(10.0, 20.0, 30.0)
        point = Vector3(1.0, 2.0, 3.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(11.0)
        assert result.y == pytest.approx(22.0)
        assert result.z == pytest.approx(33.0)
    
    def test_multiply_point_origin(self):
        """Test that translation matrix translates origin point."""
        mat = Matrix4x4.from_translation(5.0, 10.0, 15.0)
        point = Vector3(0.0, 0.0, 0.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(10.0)
        assert result.z == pytest.approx(15.0)
    
    def test_multiply_point_negative_coordinates(self):
        """Test multiplying a point with negative input coordinates."""
        mat = Matrix4x4.from_translation(5.0, 5.0, 5.0)
        point = Vector3(-2.0, -3.0, -4.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(3.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(1.0)
    
    def test_multiply_point_large_values(self):
        """Test multiplying point with large coordinate values."""
        mat = Matrix4x4.from_translation(1000.0, 2000.0, 3000.0)
        point = Vector3(500.0, 1000.0, 1500.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(1500.0)
        assert result.y == pytest.approx(3000.0)
        assert result.z == pytest.approx(4500.0)


class TestMatrix4x4MultiplyVector:
    """Test Matrix4x4.multiply_vector method."""
    
    def test_multiply_vector_identity(self):
        """Test multiplying a vector by identity matrix is unchanged."""
        mat = Matrix4x4()  # Default constructor should be identity
        vector = Vector3(1.0, 2.0, 3.0)
        result = mat.multiply_vector(vector)
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)
    
    def test_multiply_vector_translation_ignored(self):
        """Test that translation is ignored when multiplying a vector."""
        mat = Matrix4x4.from_translation(100.0, 200.0, 300.0)
        vector = Vector3(1.0, 2.0, 3.0)
        result = mat.multiply_vector(vector)
        # Vector should be unchanged because translation is ignored
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)
    
    def test_multiply_vector_zero(self):
        """Test multiplying a zero vector yields zero."""
        mat = Matrix4x4.from_translation(10.0, 20.0, 30.0)
        vector = Vector3(0.0, 0.0, 0.0)
        result = mat.multiply_vector(vector)
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)
    
    def test_multiply_vector_negative(self):
        """Test multiplying a vector with negative components."""
        mat = Matrix4x4()  # Identity
        vector = Vector3(-1.0, -2.0, -3.0)
        result = mat.multiply_vector(vector)
        assert result.x == pytest.approx(-1.0)
        assert result.y == pytest.approx(-2.0)
        assert result.z == pytest.approx(-3.0)
