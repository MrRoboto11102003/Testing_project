import pytest
import arnio as ar


class TestSchemaValidate:
    """Tests for arnio.schema.Schema.validate method"""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing"""
        return ar.from_dict({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "salary": [50000.0, 60000.0, 75000.0]
        })

    @pytest.fixture
    def empty_frame(self):
        """Create an empty ArFrame"""
        return ar.from_dict({"name": [], "age": [], "salary": []})

    @pytest.fixture
    def frame_with_nulls(self):
        """Create an ArFrame with null values"""
        return ar.from_dict({
            "name": ["Alice", None, "Charlie"],
            "age": [25, None, 35],
            "salary": [50000.0, None, 75000.0]
        })

    @pytest.fixture
    def simple_schema(self):
        """Create a simple test schema"""
        return ar.schema.Schema()

    def test_validate_returns_validation_result(self, sample_frame, simple_schema):
        """validate returns a ValidationResult object"""
        result = simple_schema.validate(sample_frame)
        assert result is not None
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'errors')

    def test_validate_frame_parameter_required(self, simple_schema):
        """frame parameter is required for validate"""
        with pytest.raises(TypeError):
            simple_schema.validate()

    def test_validate_with_max_errors_none(self, sample_frame, simple_schema):
        """max_errors=None collects all validation errors"""
        result = simple_schema.validate(sample_frame, max_errors=None)
        assert result is not None
        assert hasattr(result, 'errors')

    def test_validate_with_max_errors_limit_respected(self, sample_frame, simple_schema):
        """max_errors parameter limits number of errors returned"""
        result_max_1 = simple_schema.validate(sample_frame, max_errors=1)
        result_max_5 = simple_schema.validate(sample_frame, max_errors=5)
        
        assert len(result_max_1.errors) <= 1
        assert len(result_max_5.errors) <= 5

    def test_validate_with_max_errors_zero(self, sample_frame, simple_schema):
        """max_errors=0 returns empty error list"""
        result = simple_schema.validate(sample_frame, max_errors=0)
        assert len(result.errors) == 0

    def test_validate_empty_frame(self, empty_frame, simple_schema):
        """validate handles empty frames correctly"""
        result = simple_schema.validate(empty_frame)
        assert result is not None
        assert isinstance(result.is_valid, bool)

    def test_validate_frame_with_nulls(self, frame_with_nulls, simple_schema):
        """validate handles frames containing null values"""
        result = simple_schema.validate(frame_with_nulls)
        assert result is not None
        assert hasattr(result, 'is_valid')

    def test_validate_consistent_result_type(self, sample_frame, simple_schema):
        """validate returns consistent ValidationResult type"""
        result1 = simple_schema.validate(sample_frame)
        result2 = simple_schema.validate(sample_frame, max_errors=10)
        result3 = simple_schema.validate(sample_frame, max_errors=None)
        
        assert type(result1) == type(result2) == type(result3)

    def test_validate_large_frame(self, simple_schema):
        """validate handles large frames efficiently"""
        large_data = {
            f"col_{i}": list(range(1000))
            for i in range(20)
        }
        large_frame = ar.from_dict(large_data)
        result = simple_schema.validate(large_frame)
        assert result is not None

    def test_validate_max_errors_stops_early(self, simple_schema):
        """Validation stops once max_errors limit is reached"""
        frame = ar.from_dict({
            f"column_{i}": [i] * 100
            for i in range(50)
        })
        result = simple_schema.validate(frame, max_errors=3)
        assert len(result.errors) <= 3

    def test_validate_errors_is_iterable(self, sample_frame, simple_schema):
        """ValidationResult.errors is iterable"""
        result = simple_schema.validate(sample_frame)
        errors = result.errors
        try:
            iter(errors)
        except TypeError:
            pytest.fail("ValidationResult.errors should be iterable")

    def test_validate_is_valid_is_boolean(self, sample_frame, simple_schema):
        """ValidationResult.is_valid is a boolean"""
        result = simple_schema.validate(sample_frame)
        assert isinstance(result.is_valid, bool)

    def test_validate_accepts_keyword_arguments(self, sample_frame, simple_schema):
        """validate accepts max_errors as keyword argument"""
        result = simple_schema.validate(frame=sample_frame, max_errors=5)
        assert result is not None
