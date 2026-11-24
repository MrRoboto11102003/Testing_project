import pytest
from openlens.transform import Matrix4x4
from openlens.vector3 import Vector3


class TestMatrix4x4FromTranslation:
    """Tests for Matrix4x4.from_translation static method."""

    def test_from_translation_basic_placement(self):
        """Translation values placed in m[i][3] positions; rest is identity."""
        mat = Matrix4x4.from_translation(1, 2, 3)
        assert mat.m[0][3] == 1
        assert mat.m[1][3] == 2
        assert mat.m[2][3] == 3
        # Rest is identity
        assert mat.m[0][0] == 1.0 and mat.m[1][1] == 1.0 and mat.m[2][2] == 1.0
        assert mat.m[3] == [0.0, 0.0, 0.0, 1.0]

    def test_from_translation_returns_new_instance(self):
        """Each call returns a new matrix; row lists are not shared."""
        mat1 = Matrix4x4.from_translation(1, 2, 3)
        mat2 = Matrix4x4.from_translation(1, 2, 3)
        assert mat1 is not mat2
        assert mat1.m[0] is not mat2.m[0]

    def test_from_translation_various_values(self):
        """Works with zero, negative, and float values."""
        # Zero translation equals identity
        assert Matrix4x4.from_translation(0, 0, 0).m == Matrix4x4().m
        # Negative values
        mat_neg = Matrix4x4.from_translation(-10, -20, -30)
        assert mat_neg.m[0][3] == -10 and mat_neg.m[1][3] == -20 and mat_neg.m[2][3] == -30
        # Float values
        mat_fl = Matrix4x4.from_translation(1.5, -2.7, 3.14)
        assert mat_fl.m[0][3] == 1.5 and mat_fl.m[1][3] == -2.7 and mat_fl.m[2][3] == 3.14


class TestMatrix4x4MultiplyPoint:
    """Tests for Matrix4x4.multiply_point method."""

    def test_multiply_point_identity(self):
        """Identity matrix preserves point coordinates."""
        result = Matrix4x4().multiply_point(Vector3(1, 2, 3))
        assert result.x == 1 and result.y == 2 and result.z == 3

    def test_multiply_point_with_translation(self):
        """Translation is properly applied to points."""
        mat = Matrix4x4.from_translation(10, -5, 2.5)
        result = mat.multiply_point(Vector3(1, 2, 3))
        assert result.x == 11 and result.y == -3 and result.z == 5.5

    def test_multiply_point_origin_gives_translation(self):
        """Translation applied to origin gives translation vector."""
        result = Matrix4x4.from_translation(10, 20, 30).multiply_point(Vector3(0, 0, 0))
        assert result.x == 10 and result.y == 20 and result.z == 30

    def test_multiply_point_non_symmetric_matrix(self):
        """Non-symmetric matrix detects row vs column ordering errors."""
        m = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        result = Matrix4x4(m).multiply_point(Vector3(1, 1, 1))
        # x = 1*1 + 2*1 + 3*1 + 4 = 10; y = 26; z = 42
        assert result.x == 10 and result.y == 26 and result.z == 42

    def test_multiply_point_ignores_row3_no_perspective_divide(self):
        """Row 3 is ignored; no perspective divide applied."""
        m = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [999, 999, 999, 999]]
        result = Matrix4x4(m).multiply_point(Vector3(2, 4, 6))
        assert result.x == 2 and result.y == 4 and result.z == 6

    def test_multiply_point_no_mutation_returns_new(self):
        """Input not mutated; result is new Vector3 object."""
        mat = Matrix4x4.from_translation(1, 2, 3)
        point = Vector3(5, 6, 7)
        result = mat.multiply_point(point)
        assert result is not point
        assert point.x == 5  # original unchanged


class TestMatrix4x4MultiplyVector:
    """Tests for Matrix4x4.multiply_vector method."""

    def test_multiply_vector_identity(self):
        """Identity matrix preserves vector coordinates."""
        result = Matrix4x4().multiply_vector(Vector3(1, 2, 3))
        assert result.x == 1 and result.y == 2 and result.z == 3

    def test_multiply_vector_translation_ignored(self):
        """Translation is completely ignored for direction vectors."""
        result = Matrix4x4.from_translation(10, 20, 30).multiply_vector(Vector3(1, 2, 3))
        assert result.x == 1 and result.y == 2 and result.z == 3

    def test_multiply_vector_non_symmetric_matrix(self):
        """Non-symmetric matrix detects row vs column ordering errors."""
        m = [[1, 2, 3, 10], [4, 5, 6, 20], [7, 8, 9, 30], [0, 0, 0, 1]]
        result = Matrix4x4(m).multiply_vector(Vector3(1, 2, 3))
        # x = 1*1 + 2*2 + 3*3 = 14; y = 32; z = 50
        assert result.x == 14 and result.y == 32 and result.z == 50

    def test_multiply_vector_ignores_row3_and_translation_column(self):
        """Row 3 and translation column (m[i][3]) are ignored."""
        m = [[1, 0, 0, 999], [0, 1, 0, 999], [0, 0, 1, 999], [999, 999, 999, 999]]
        result = Matrix4x4(m).multiply_vector(Vector3(1, 2, 3))
        assert result.x == 1 and result.y == 2 and result.z == 3

    def test_multiply_vector_no_mutation_returns_new(self):
        """Input not mutated; result is new Vector3 object."""
        mat = Matrix4x4.from_translation(1, 2, 3)
        vector = Vector3(1, 2, 3)
        result = mat.multiply_vector(vector)
        assert result is not vector
        assert vector.x == 1  # original unchanged
