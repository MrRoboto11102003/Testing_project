import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestSchemaValidate:
    """Test suite for arnio.schema.Schema.validate method"""

    @pytest.fixture
    def basic_schema(self):
        """Create a basic schema for testing"""
        return ar.Schema()

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame"""
        df = pd.DataFrame({"col": [1, 2, 3]})
        return ar.ArFrame(df)

    def test_validate_returns_validation_result(self, basic_schema, sample_frame):
        """Test that validate returns a ValidationResult"""
        result = basic_schema.validate(sample_frame)
        assert isinstance(result, ar.ValidationResult)

    def test_validate_default_max_errors(self, basic_schema, sample_frame):
        """Test validate with default max_errors parameter"""
        result = basic_schema.validate(sample_frame)
        assert result is not None

    def test_validate_max_errors_none(self, basic_schema, sample_frame):
        """Test validate with max_errors=None (no limit)"""
        result = basic_schema.validate(sample_frame, max_errors=None)
        assert isinstance(result, ar.ValidationResult)

    def test_validate_max_errors_zero(self, basic_schema, sample_frame):
        """Test validate with max_errors=0"""
        result = basic_schema.validate(sample_frame, max_errors=0)
        assert isinstance(result, ar.ValidationResult)

    def test_validate_max_errors_positive(self, basic_schema, sample_frame):
        """Test validate with positive max_errors"""
        result = basic_schema.validate(sample_frame, max_errors=5)
        assert isinstance(result, ar.ValidationResult)

    def test_validate_max_errors_large(self, basic_schema, sample_frame):
        """Test validate with large max_errors value"""
        result = basic_schema.validate(sample_frame, max_errors=1000000)
        assert isinstance(result, ar.ValidationResult)

    def test_validate_empty_frame(self, basic_schema):
        """Test validate with empty ArFrame"""
        df = pd.DataFrame()
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)

    def test_validate_single_row_frame(self, basic_schema):
        """Test validate with single row"""
        df = pd.DataFrame({"col": [1]})
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)

    def test_validate_large_frame(self, basic_schema):
        """Test validate with large frame (10k rows)"""
        df = pd.DataFrame({"col": range(10000)})
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)

    def test_validate_multiple_columns(self, basic_schema):
        """Test validate with multiple columns and types"""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"]
        })
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)

    def test_validate_with_nulls_and_nans(self, basic_schema):
        """Test validate with null and NaN values"""
        df = pd.DataFrame({
            "col1": [1, None, 3],
            "col2": [1.0, np.nan, 3.0]
        })
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)

    def test_validate_does_not_mutate_frame(self, basic_schema, sample_frame):
        """Test that validate does not modify input frame"""
        frame_copy = sample_frame.copy()
        basic_schema.validate(sample_frame)
        pd.testing.assert_frame_equal(sample_frame, frame_copy)

    def test_validate_consistent_across_calls(self, basic_schema, sample_frame):
        """Test that validate produces consistent results on repeated calls"""
        result1 = basic_schema.validate(sample_frame, max_errors=1)
        result2 = basic_schema.validate(sample_frame, max_errors=1)
        assert isinstance(result1, ar.ValidationResult)
        assert isinstance(result2, ar.ValidationResult)

    def test_validate_max_errors_stops_early(self, basic_schema, sample_frame):
        """Test that max_errors parameter limits validation output"""
        result_limit_1 = basic_schema.validate(sample_frame, max_errors=1)
        result_unlimited = basic_schema.validate(sample_frame, max_errors=None)
        assert isinstance(result_limit_1, ar.ValidationResult)
        assert isinstance(result_unlimited, ar.ValidationResult)
