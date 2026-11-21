import pytest
import pandas as pd
import arnio as ar


class TestValidateBasic:
    """Test basic validation functionality."""
    
    def test_validate_returns_validation_result(self):
        """Test that validate returns a ValidationResult."""
        df = pd.DataFrame({"email": ["test@example.com"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)
        assert hasattr(result, "passed")
        assert hasattr(result, "issue_count")
        assert hasattr(result, "issues")
        assert hasattr(result, "bad_rows")
    
    def test_validate_passing_frame(self):
        """Test validation of a frame that passes all checks."""
        df = pd.DataFrame({"email": ["test@example.com"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)
        assert result.passed
        assert result.issue_count == 0
    
    def test_validate_with_dict_schema(self):
        """Test validation using dict instead of Schema object."""
        df = pd.DataFrame({"email": ["test@example.com"]})
        frame = ar.from_pandas(df)
        schema_dict = {"email": ar.Email()}
        result = ar.validate(frame, schema_dict)
        assert isinstance(result, ar.ValidationResult)


class TestValidateMissingColumns:
    """Test validation of missing required columns."""
    
    def test_missing_required_column(self):
        """Test detection of missing required column."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "required_column" for issue in result.issues)


class TestValidateStrictMode:
    """Test strict mode validation."""
    
    def test_strict_mode_rejects_extra_columns(self):
        """Test that strict mode flags unexpected columns."""
        df = pd.DataFrame({"email": ["test@example.com"], "extra": [1]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email()}, strict=True)
        result = ar.validate(frame, schema)
        assert any(issue.rule == "unexpected_column" for issue in result.issues)
    
    def test_non_strict_allows_extra_columns(self):
        """Test that non-strict mode allows extra columns."""
        df = pd.DataFrame({"email": ["test@example.com"], "extra": [1]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email()}, strict=False)
        result = ar.validate(frame, schema)
        unexpected_issues = [i for i in result.issues if i.rule == "unexpected_column"]
        assert len(unexpected_issues) == 0


class TestValidateUnique:
    """Test unique constraint validation."""
    
    def test_unique_no_duplicates(self):
        """Test unique constraint with unique values."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int()}, unique=["id"])
        result = ar.validate(frame, schema)
        assert not any(issue.rule == "composite_unique" for issue in result.issues)
    
    def test_unique_detects_duplicates(self):
        """Test unique constraint detects duplicate values."""
        df = pd.DataFrame({"id": [1, 1, 2]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int()}, unique=["id"])
        result = ar.validate(frame, schema)
        assert any(issue.rule == "composite_unique" for issue in result.issues)
    
    def test_unique_empty_list_error(self):
        """Test that empty unique list produces an error."""
        df = pd.DataFrame({"id": [1, 2]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int()}, unique=[])
        result = ar.validate(frame, schema)
        assert any(issue.rule == "composite_unique" for issue in result.issues)
    
    def test_unique_missing_column_error(self):
        """Test unique constraint with missing column."""
        df = pd.DataFrame({"id": [1, 2]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int()}, unique=["id", "missing"])
        result = ar.validate(frame, schema)
        assert any(issue.rule == "missing_column" for issue in result.issues)
    
    def test_unique_type_error_not_list(self):
        """Test TypeError when unique is not list or tuple."""
        df = pd.DataFrame({"id": [1, 2]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int()}, unique="id")
        with pytest.raises(TypeError, match="unique.*must be a list or tuple"):
            ar.validate(frame, schema)
    
    def test_unique_type_error_non_string(self):
        """Test TypeError when unique contains non-strings."""
        df = pd.DataFrame({"id": [1, 2]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int()}, unique=["id", 123])
        with pytest.raises(TypeError, match="unique.*must be strings"):
            ar.validate(frame, schema)


class TestValidateMaxErrors:
    """Test max_errors parameter."""
    
    def test_max_errors_limit_enforced(self):
        """Test that max_errors limit is respected."""
        df = pd.DataFrame({"a": [None, None], "b": [None, None]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"a": ar.String(nullable=False), "b": ar.String(nullable=False)})
        result = ar.validate(frame, schema, max_errors=1)
        assert result.issue_count <= 1
    
    def test_max_errors_type_error_string(self):
        """Test TypeError for string max_errors."""
        df = pd.DataFrame({"id": [1]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int()})
        with pytest.raises(TypeError, match="max_errors must be an int"):
            ar.validate(frame, schema, max_errors="5")
    
    def test_max_errors_type_error_bool(self):
        """Test TypeError for bool max_errors."""
        df = pd.DataFrame({"id": [1]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int()})
        with pytest.raises(TypeError, match="max_errors must be an int"):
            ar.validate(frame, schema, max_errors=True)
    
    def test_max_errors_value_error_zero(self):
        """Test ValueError for max_errors=0."""
        df = pd.DataFrame({"id": [1]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int()})
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, schema, max_errors=0)
    
    def test_max_errors_value_error_negative(self):
        """Test ValueError for negative max_errors."""
        df = pd.DataFrame({"id": [1]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int()})
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, schema, max_errors=-1)


class TestValidateTypeErrors:
    """Test type error handling."""
    
    def test_non_arframe_type_error(self):
        """Test TypeError when frame is not ArFrame."""
        df = pd.DataFrame({"id": [1]})
        schema = ar.Schema({"id": ar.Int()})
        with pytest.raises(TypeError, match="expects a single ArFrame"):
            ar.validate(df, schema)


class TestValidateResultProperties:
    """Test ValidationResult properties."""
    
    def test_result_row_count(self):
        """Test that row_count is correct."""
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int()})
        result = ar.validate(frame, schema)
        assert result.row_count == 5
    
    def test_result_bad_rows_sorted(self):
        """Test that bad_rows is sorted list of indices."""
        df = pd.DataFrame({"id": [1, None, None, 4]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int(nullable=False)})
        result = ar.validate(frame, schema)
        assert isinstance(result.bad_rows, list)
        if len(result.bad_rows) > 1:
            assert result.bad_rows == sorted(result.bad_rows)
