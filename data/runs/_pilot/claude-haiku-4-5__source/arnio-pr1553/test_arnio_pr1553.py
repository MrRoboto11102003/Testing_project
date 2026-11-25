import pytest
import arnio as ar


class TestSchemaValidate:
    """Tests for arnio.schema.Schema.validate method"""

    @pytest.fixture
    def test_schema(self):
        """Create a schema instance for testing"""
        return ar.schema.Schema()

    @pytest.fixture
    def simple_frame(self):
        """Create a simple ArFrame for testing"""
        return ar.from_dict({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })

    @pytest.fixture
    def empty_frame(self):
        """Create an empty ArFrame"""
        return ar.from_dict({})

    @pytest.fixture
    def frame_with_nulls(self):
        """Create an ArFrame with null values"""
        return ar.from_dict({
            "col1": [1, None, 3],
            "col2": ["a", None, "c"],
            "col3": [None, None, None]
        })

    def test_validate_returns_validation_result(self, test_schema, simple_frame):
        """Test that validate returns a ValidationResult instance"""
        result = test_schema.validate(simple_frame)
        assert hasattr(result, '__class__')
        assert 'ValidationResult' in type(result).__name__

    def test_validate_with_frame_positional_arg(self, test_schema, simple_frame):
        """Test validate accepts frame as positional argument"""
        result = test_schema.validate(simple_frame)
        assert result is not None

    def test_validate_with_frame_keyword_arg(self, test_schema, simple_frame):
        """Test validate accepts frame as keyword argument"""
        result = test_schema.validate(frame=simple_frame)
        assert result is not None

    def test_validate_max_errors_none(self, test_schema, simple_frame):
        """Test validate with max_errors=None (no limit)"""
        result = test_schema.validate(simple_frame, max_errors=None)
        assert result is not None

    def test_validate_max_errors_zero(self, test_schema, simple_frame):
        """Test validate with max_errors=0"""
        result = test_schema.validate(simple_frame, max_errors=0)
        assert result is not None

    def test_validate_max_errors_positive(self, test_schema, simple_frame):
        """Test validate with positive max_errors value"""
        result = test_schema.validate(simple_frame, max_errors=5)
        assert result is not None

    def test_validate_empty_frame(self, test_schema, empty_frame):
        """Test validate handles empty ArFrame"""
        result = test_schema.validate(empty_frame)
        assert result is not None

    def test_validate_single_row_frame(self, test_schema):
        """Test validate with single-row ArFrame"""
        frame = ar.from_dict({"col": [42]})
        result = test_schema.validate(frame)
        assert result is not None

    def test_validate_single_column_frame(self, test_schema):
        """Test validate with single-column ArFrame"""
        frame = ar.from_dict({"col": [1, 2, 3, 4, 5]})
        result = test_schema.validate(frame)
        assert result is not None

    def test_validate_frame_with_nulls(self, test_schema, frame_with_nulls):
        """Test validate handles frames with null/None values"""
        result = test_schema.validate(frame_with_nulls)
        assert result is not None

    def test_validate_mixed_data_types(self, test_schema):
        """Test validate with mixed data types"""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        result = test_schema.validate(frame)
        assert result is not None

    def test_validate_large_frame(self, test_schema):
        """Test validate on large ArFrame"""
        frame = ar.from_dict({
            "col1": list(range(10000)),
            "col2": list(range(10000, 20000))
        })
        result = test_schema.validate(frame)
        assert result is not None

    def test_validate_with_keyword_and_positional_max_errors(self, test_schema, simple_frame):
        """Test validate with max_errors specified by keyword argument"""
        result1 = test_schema.validate(simple_frame, max_errors=10)
        result2 = test_schema.validate(simple_frame, max_errors=1)
        assert result1 is not None
        assert result2 is not None

    def test_validate_duplicate_rows(self, test_schema):
        """Test validate with duplicate rows"""
        frame = ar.from_dict({
            "col1": [1, 1, 2, 2, 3],
            "col2": ["a", "a", "b", "b", "c"]
        })
        result = test_schema.validate(frame)
        assert result is not None
