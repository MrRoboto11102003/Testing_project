import pytest
from openlens.transform import Matrix4x4, Vector3


class TestMatrix4x4FromTranslation:
    """Tests for Matrix4x4.from_translation static method."""

    def test_from_translation_basic(self):
        """Test creating a translation matrix with simple values."""
        mat = Matrix4x4.from_translation(1.0, 2.0, 3.0)
        assert mat.m[0][3] == 1.0
        assert mat.m[1][3] == 2.0
        assert mat.m[2][3] == 3.0

    def test_from_translation_negative_values(self):
        """Test translation with negative values."""
        mat = Matrix4x4.from_translation(-5.0, -10.0, -15.0)
        assert mat.m[0][3] == -5.0
        assert mat.m[1][3] == -10.0
        assert mat.m[2][3] == -15.0

    def test_from_translation_zero(self):
        """Test zero translation matrix."""
        mat = Matrix4x4.from_translation(0.0, 0.0, 0.0)
        assert mat.m[0][3] == 0.0
        assert mat.m[1][3] == 0.0
        assert mat.m[2][3] == 0.0


class TestMatrix4x4MultiplyPoint:
    """Tests for Matrix4x4.multiply_point method (transforms points with w=1)."""

    def test_multiply_point_origin_translated(self):
        """Test that origin point is translated correctly."""
        mat = Matrix4x4.from_translation(1.0, 2.0, 3.0)
        point = Vector3(0.0, 0.0, 0.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)

    def test_multiply_point_non_origin(self):
        """Test translation of a non-origin point."""
        mat = Matrix4x4.from_translation(10.0, 20.0, 30.0)
        point = Vector3(1.0, 2.0, 3.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(11.0)
        assert result.y == pytest.approx(22.0)
        assert result.z == pytest.approx(33.0)

    def test_multiply_point_negative_translation(self):
        """Test point transformation with negative translation."""
        mat = Matrix4x4.from_translation(-5.0, -10.0, -15.0)
        point = Vector3(5.0, 10.0, 15.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_multiply_point_zero_translation(self):
        """Test that zero translation leaves point unchanged."""
        mat = Matrix4x4.from_translation(0.0, 0.0, 0.0)
        point = Vector3(5.0, 7.0, 9.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(7.0)
        assert result.z == pytest.approx(9.0)

    def test_multiply_point_large_values(self):
        """Test with large translation and point values."""
        mat = Matrix4x4.from_translation(1000.0, 2000.0, 3000.0)
        point = Vector3(100.0, 200.0, 300.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(1100.0)
        assert result.y == pytest.approx(2200.0)
        assert result.z == pytest.approx(2300.0)


class TestMatrix4x4MultiplyVector:
    """Tests for Matrix4x4.multiply_vector method (transforms vectors with w=0, ignores translation)."""

    def test_multiply_vector_ignores_translation(self):
        """Test that translation is ignored for direction vectors (w=0)."""
        mat = Matrix4x4.from_translation(10.0, 20.0, 30.0)
        vector = Vector3(1.0, 0.0, 0.0)
        result = mat.multiply_vector(vector)
        # With identity rotation part, vector should be unchanged
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_multiply_vector_arbitrary_components(self):
        """Test vector transformation with arbitrary non-unit components."""
        mat = Matrix4x4.from_translation(100.0, 200.0, 300.0)
        vector = Vector3(2.0, 3.0, 4.0)
        result = mat.multiply_vector(vector)
        # Translation should be ignored
        assert result.x == pytest.approx(2.0)
        assert result.y == pytest.approx(3.0)
        assert result.z == pytest.approx(4.0)

    def test_multiply_vector_zero_vector(self):
        """Test that zero vector remains zero regardless of translation."""
        mat = Matrix4x4.from_translation(10.0, 20.0, 30.0)
        vector = Vector3(0.0, 0.0, 0.0)
        result = mat.multiply_vector(vector)
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(0.0)

    def test_multiply_vector_negative_components(self):
        """Test vector with negative components ignores translation."""
        mat = Matrix4x4.from_translation(50.0, 50.0, 50.0)
        vector = Vector3(-1.0, -2.0, -3.0)
        result = mat.multiply_vector(vector)
        assert result.x == pytest.approx(-1.0)
        assert result.y == pytest.approx(-2.0)
        assert result.z == pytest.approx(-3.0)


class TestMatrix4x4PointVectorDistinction:
    """Tests verifying the critical distinction between point and vector transformation."""

    def test_point_vs_vector_at_origin(self):
        """Test that points and vectors behave differently at origin with translation."""
        mat = Matrix4x4.from_translation(5.0, 5.0, 5.0)
        
        # Origin point should be translated
        point = Vector3(0.0, 0.0, 0.0)
        point_result = mat.multiply_point(point)
        
        # Origin vector should not be translated
        vector = Vector3(0.0, 0.0, 0.0)
        vector_result = mat.multiply_vector(vector)
        
        # Points get translated, vectors do not
        assert point_result.x == pytest.approx(5.0)
        assert vector_result.x == pytest.approx(0.0)

    def test_point_and_vector_difference(self):
        """Test that point and vector results differ by exactly the translation amount."""
        mat = Matrix4x4.from_translation(10.0, 20.0, 30.0)
        pos = Vector3(1.0, 2.0, 3.0)
        
        point_result = mat.multiply_point(pos)
        vector_result = mat.multiply_vector(pos)
        
        # Difference should equal translation
        assert (point_result.x - vector_result.x) == pytest.approx(10.0)
        assert (point_result.y - vector_result.y) == pytest.approx(20.0)
        assert (point_result.z - vector_result.z) == pytest.approx(30.0)

    def test_very_small_translation(self):
        """Test precision with very small translation values."""
        mat = Matrix4x4.from_translation(0.001, 0.002, 0.003)
        point = Vector3(1.0, 1.0, 1.0)
        result = mat.multiply_point(point)
        assert result.x == pytest.approx(1.001, abs=1e-9)
        assert result.y == pytest.approx(1.002, abs=1e-9)
        assert result.z == pytest.approx(1.003, abs=1e-9)
