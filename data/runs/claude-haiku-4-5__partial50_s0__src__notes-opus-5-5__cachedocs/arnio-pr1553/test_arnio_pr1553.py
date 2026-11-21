import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestSchemaValidateBasic:
    """Test basic validation parameter handling."""
    
    def test_valid_frame_passes(self):
        """A valid frame with matching schema should pass."""
        df = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}))
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        result = schema.validate(df)
        assert result.passed
        assert result.issue_count == 0
    
    def test_pandas_dataframe_raises_typeerror(self):
        """Passing a pandas DataFrame (not ArFrame) should raise TypeError."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        schema = ar.Schema({"id": ar.Int64()})
        with pytest.raises(TypeError, match="expects a single ArFrame"):
            schema.validate(df)
    
    def test_max_errors_bool_raises_typeerror(self):
        """max_errors parameter as bool should raise TypeError."""
        df = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))
        schema = ar.Schema({"id": ar.Int64()})
        with pytest.raises(TypeError, match="max_errors must be an int or None"):
            schema.validate(df, max_errors=True)
    
    def test_max_errors_float_raises_typeerror(self):
        """max_errors parameter as float should raise TypeError."""
        df = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))
        schema = ar.Schema({"id": ar.Int64()})
        with pytest.raises(TypeError, match="max_errors must be an int or None"):
            schema.validate(df, max_errors=1.5)
    
    def test_max_errors_zero_raises_valueerror(self):
        """max_errors=0 should raise ValueError."""
        df = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))
        schema = ar.Schema({"id": ar.Int64()})
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            schema.validate(df, max_errors=0)
    
    def test_max_errors_negative_raises_valueerror(self):
        """max_errors=-1 should raise ValueError."""
        df = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))
        schema = ar.Schema({"id": ar.Int64()})
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            schema.validate(df, max_errors=-1)


class TestSchemaValidateConstraints:
    """Test dtype, nullable, and unique constraints."""
    
    def test_dtype_mismatch_fails(self):
        """Column with mismatched dtype should produce dtype issue."""
        df = ar.from_pandas(pd.DataFrame({"id": ["1", "2", "3"]}))  # string, not int
        schema = ar.Schema({"id": ar.Int64()})
        result = schema.validate(df)
        assert not result.passed
        assert any(i.rule == "dtype" for i in result.issues)
    
    def test_non_nullable_with_nulls_fails(self):
        """Non-nullable column with null values should fail."""
        df = ar.from_pandas(pd.DataFrame({"id": [1, None, 3]}))
        schema = ar.Schema({"id": ar.Int64(nullable=False)})
        result = schema.validate(df)
        assert not result.passed
        assert any(i.rule == "nullable" for i in result.issues)
    
    def test_nullable_with_nulls_passes(self):
        """Nullable column with null values should pass."""
        df = ar.from_pandas(pd.DataFrame({"id": [1, None, 3]}))
        schema = ar.Schema({"id": ar.Int64(nullable=True)})
        result = schema.validate(df)
        assert result.passed
    
    def test_unique_with_duplicates_fails(self):
        """Column with duplicate values should fail unique check."""
        df = ar.from_pandas(pd.DataFrame({"id": [1, 2, 2, 3]}))
        schema = ar.Schema({"id": ar.Int64(unique=True)})
        result = schema.validate(df)
        assert not result.passed
        # unique check reports all occurrences of duplicates
        duplicate_issues = [i for i in result.issues if i.rule == "unique"]
        assert len(duplicate_issues) >= 2


class TestSchemaValidateBounds:
    """Test min/max bounds validation."""
    
    def test_value_below_min_fails(self):
        """Value below min constraint should fail."""
        df = ar.from_pandas(pd.DataFrame({"age": [1, 5, 10]}))
        schema = ar.Schema({"age": ar.Int64(min=2)})
        result = schema.validate(df)
        assert not result.passed
        # row_index is 1-based, so pandas row 0 becomes row_index 1
        assert any(i.rule == "min" and i.row_index == 1 for i in result.issues)
    
    def test_value_above_max_fails(self):
        """Value above max constraint should fail."""
        df = ar.from_pandas(pd.DataFrame({"age": [1, 5, 100]}))
        schema = ar.Schema({"age": ar.Int64(max=99)})
        result = schema.validate(df)
        assert not result.passed
        # pandas row 2 becomes row_index 3
        assert any(i.rule == "max" and i.row_index == 3 for i in result.issues)
    
    def test_value_at_bounds_passes(self):
        """Values equal to bounds should pass (bounds are inclusive)."""
        df = ar.from_pandas(pd.DataFrame({"age": [2, 5, 9]}))
        schema = ar.Schema({"age": ar.Int64(min=2, max=9)})
        result = schema.validate(df)
        assert result.passed


class TestSchemaValidateRowIndexAndBadRows:
    """Test row_index (1-based) and bad_rows tracking."""
    
    def test_row_index_is_one_based(self):
        """row_index values should be 1-based (pandas row 0 = row_index 1)."""
        df = ar.from_pandas(pd.DataFrame({"id": [10, 5, 20]}))
        schema = ar.Schema({"id": ar.Int64(min=10, max=15)})
        result = schema.validate(df)
        # Pandas row 0: id=10 ✓
        # Pandas row 1: id=5 ✗ (below min) → row_index 2
        # Pandas row 2: id=20 ✗ (above max) → row_index 3
        assert sorted(result.bad_rows) == [2, 3]
    
    def test_bad_rows_deduplicated(self):
        """bad_rows should be deduplicated."""
        df = ar.from_pandas(pd.DataFrame({"id": [5, 20], "age": [5, 20]}))
        # Both constraints will fail on both rows, but bad_rows should be deduplicated
        schema = ar.Schema({"id": ar.Int64(min=10), "age": ar.Int64(min=10)})
        result = schema.validate(df)
        assert result.bad_rows == [1, 2]  # No duplicates


class TestSchemaValidateMaxErrors:
    """Test max_errors limiting behavior."""
    
    def test_max_errors_stops_at_limit(self):
        """Validation should stop and return exactly max_errors issues."""
        df = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3, 4, 5]}))
        # Rows 1 and 5 fail (too low), plus check runs would find other issues
        schema = ar.Schema({"id": ar.Int64(min=2, max=4)})
        result = schema.validate(df, max_errors=2)
        assert result.issue_count == 2


class TestSchemaValidateSeverity:
    """Test severity handling."""
    
    def test_warning_severity_passes(self):
        """Warning-severity issues should not prevent passed=True."""
        df = ar.from_pandas(pd.DataFrame({"id": [1, 100]}))
        schema = ar.Schema({"id": ar.Int64(max=50, severity="warning")})
        result = schema.validate(df)
        assert result.passed  # Warnings don't fail validation
        assert result.issue_count > 0  # But issues are recorded


class TestSchemaValidateEmptyFrame:
    """Test edge cases with empty or minimal frames."""
    
    def test_empty_frame_no_row_issues(self):
        """Empty frame should have no row-level issues."""
        df = ar.from_pandas(pd.DataFrame({"id": pd.Series([], dtype="int64")}))
        schema = ar.Schema({"id": ar.Int64(min=0, max=100)})
        result = schema.validate(df)
        assert result.row_count == 0
        # Row-level issues have row_index != None
        row_issues = [i for i in result.issues if i.row_index is not None]
        assert len(row_issues) == 0
    
    def test_null_values_skipped_after_nullable_check(self):
        """Null values should be skipped by min/max checks after nullable passes."""
        df = ar.from_pandas(pd.DataFrame({"id": [5, None, 15]}))
        # With nullable=True, the None should not trigger min/max issues
        schema = ar.Schema({"id": ar.Int64(nullable=True, min=10, max=20)})
        result = schema.validate(df)
        # Only row_index 1 (id=5) should fail min check; None should be skipped
        min_issues = [i for i in result.issues if i.rule == "min"]
        assert len(min_issues) == 1 and min_issues[0].row_index == 1
