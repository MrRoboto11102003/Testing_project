import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestSchemaValidate:
    """Tests for arnio.schema.Schema.validate method."""

    @pytest.fixture
    def basic_schema(self):
        """Create a basic schema with int and string columns."""
        schema = ar.Schema()
        schema.add_column("id", int, nullable=False)
        schema.add_column("name", str, nullable=False)
        schema.add_column("age", int, nullable=True)
        return schema

    @pytest.fixture
    def valid_frame(self):
        """Create a valid frame matching schema."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        return ar.ArFrame(df)

    def test_validate_valid_frame_succeeds(self, basic_schema, valid_frame):
        """Test that a valid frame passes validation."""
        result = basic_schema.validate(valid_frame)
        assert result.is_valid() is True
        assert len(result.errors) == 0

    def test_validate_returns_validation_result(self, basic_schema, valid_frame):
        """Test that validate returns a ValidationResult object."""
        result = basic_schema.validate(valid_frame)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert callable(result.is_valid)

    def test_validate_missing_required_column(self, basic_schema):
        """Test validation fails when required column is missing."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "age": [25, 30, 35]
        })
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        assert result.is_valid() is False
        assert len(result.errors) > 0

    def test_validate_null_in_non_nullable_column(self, basic_schema):
        """Test that null values fail in non-nullable columns."""
        df = pd.DataFrame({
            "id": [1, None, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        assert result.is_valid() is False

    def test_validate_null_in_nullable_column_succeeds(self, basic_schema):
        """Test that null values are allowed in nullable columns."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, None, 35]
        })
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        assert result.is_valid() is True

    def test_validate_wrong_data_type(self, basic_schema):
        """Test validation fails with incorrect data types."""
        df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        assert result.is_valid() is False

    def test_validate_max_errors_limits_results(self, basic_schema):
        """Test that max_errors parameter limits number of errors returned."""
        df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "name": [1, 2, 3],
            "age": [25, 30, 35]
        })
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame, max_errors=1)
        assert len(result.errors) <= 1

    def test_validate_max_errors_none_returns_all_errors(self, basic_schema):
        """Test that max_errors=None returns all validation errors."""
        df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "name": [1, 2, 3]
        })
        frame = ar.ArFrame(df)
        result_limited = basic_schema.validate(frame, max_errors=1)
        result_unlimited = basic_schema.validate(frame, max_errors=None)
        assert len(result_unlimited.errors) >= len(result_limited.errors)

    def test_validate_empty_frame(self, basic_schema):
        """Test validation on empty DataFrame."""
        df = pd.DataFrame({
            "id": pd.Series(dtype="int64"),
            "name": pd.Series(dtype="object"),
            "age": pd.Series(dtype="int64")
        })
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        assert result.is_valid() is True

    def test_validate_large_frame(self, basic_schema):
        """Test validation on large DataFrame."""
        df = pd.DataFrame({
            "id": range(10000),
            "name": ["name_" + str(i) for i in range(10000)],
            "age": np.random.randint(18, 80, 10000)
        })
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        assert result.is_valid() is True

    def test_validate_max_errors_zero(self, basic_schema, valid_frame):
        """Test that max_errors=0 returns no errors in result."""
        result = basic_schema.validate(valid_frame, max_errors=0)
        assert len(result.errors) == 0

    def test_validate_with_nan_values(self, basic_schema):
        """Test validation handles NaN values correctly."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25.0, np.nan, 35.0]
        })
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        # NaN should be treated as nullable
        assert result.is_valid() is True

    def test_validate_multiple_issues_reported(self, basic_schema):
        """Test that validation reports multiple errors when present."""
        df = pd.DataFrame({
            "id": ["invalid", "invalid", "invalid"],
            "name": [None, None, None],
            "age": ["invalid", "invalid", "invalid"]
        })
        frame = ar.ArFrame(df)
        result = basic_schema.validate(frame)
        assert result.is_valid() is False
        assert len(result.errors) > 0
