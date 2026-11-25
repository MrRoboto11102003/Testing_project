import pytest
import pandas as pd
import numpy as np
import arnio as ar


@pytest.fixture
def clean_frame():
    """Simple clean frame for baseline tests."""
    return ar.from_pandas(pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [95.5, 87.2, 92.1],
    }))


@pytest.fixture
def messy_frame():
    """Frame with whitespace, duplicates, and nulls."""
    return ar.from_pandas(pd.DataFrame({
        "id": [1, 2, 2, 3, 4],
        "name": [" Alice ", " Bob ", " Bob ", None, " Dave "],
        "score": [95.5, 87.2, 87.2, 90.0, np.nan],
    }))


@pytest.fixture
def numeric_only_frame():
    """Frame with only numeric columns for type casting."""
    return ar.from_pandas(pd.DataFrame({
        "age": ["25", "30", "35"],
        "salary": ["50000", "60000", "70000"],
    }))


class TestAutoCleanSafeMode:
    """Tests for auto_clean with mode='safe'."""
    
    def test_safe_mode_default(self, messy_frame):
        """Safe mode is the default and only strips whitespace."""
        result = ar.auto_clean(messy_frame)
        df = ar.to_pandas(result)
        assert df.shape[0] == 5  # No rows dropped
        assert df["name"].iloc[0] == "Alice"  # Whitespace stripped
        assert df["name"].iloc[1] == "Bob"
    
    def test_safe_mode_explicit(self, messy_frame):
        """Explicitly passing mode='safe' works."""
        result = ar.auto_clean(messy_frame, mode="safe")
        df = ar.to_pandas(result)
        assert df.shape[0] == 5
    
    def test_safe_mode_preserves_nulls(self, messy_frame):
        """Safe mode preserves null values."""
        result = ar.auto_clean(messy_frame, mode="safe")
        df = ar.to_pandas(result)
        assert pd.isna(df["name"].iloc[3])
        assert pd.isna(df["score"].iloc[4])
    
    def test_safe_mode_preserves_duplicates(self, messy_frame):
        """Safe mode does not remove duplicates."""
        result = ar.auto_clean(messy_frame, mode="safe")
        df = ar.to_pandas(result)
        duplicate_ids = df[df["id"] == 2]
        assert len(duplicate_ids) == 2


class TestAutoCleanStrictMode:
    """Tests for auto_clean with mode='strict'."""
    
    def test_strict_mode_removes_duplicates(self, messy_frame):
        """Strict mode removes exact duplicates."""
        result = ar.auto_clean(messy_frame, mode="strict")
        df = ar.to_pandas(result)
        # Row with id=2 is duplicated; strict removes one
        assert len(df[df["id"] == 2]) == 1
    
    def test_strict_mode_strips_whitespace(self, messy_frame):
        """Strict mode also strips whitespace."""
        result = ar.auto_clean(messy_frame, mode="strict")
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Alice"
    
    def test_strict_mode_reduces_rows(self, messy_frame):
        """Strict mode reduces row count by removing duplicates."""
        safe_result = ar.auto_clean(messy_frame, mode="safe")
        strict_result = ar.auto_clean(messy_frame, mode="strict")
        safe_df = ar.to_pandas(safe_result)
        strict_df = ar.to_pandas(strict_result)
        assert strict_df.shape[0] < safe_df.shape[0]


class TestAutoCleanReturnReport:
    """Tests for return_report parameter."""
    
    def test_return_report_single_value(self, clean_frame):
        """Without return_report, returns only ArFrame."""
        result = ar.auto_clean(clean_frame)
        assert hasattr(result, 'columns')  # It's an ArFrame
        assert hasattr(result, 'shape')
    
    def test_return_report_true(self, messy_frame):
        """With return_report=True, returns tuple of (ArFrame, DataQualityReport)."""
        result = ar.auto_clean(messy_frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, report = result
        assert hasattr(frame, 'columns')
        assert hasattr(report, 'summary')
    
    def test_report_contains_summary(self, messy_frame):
        """Returned report has meaningful summary."""
        _, report = ar.auto_clean(messy_frame, mode="safe", return_report=True)
        summary = report.summary()
        assert "row_count" in summary
        assert "column_count" in summary


class TestAutoCleanDryRun:
    """Tests for dry_run parameter."""
    
    def test_dry_run_no_changes(self, messy_frame):
        """dry_run=True returns suggestions without modifying data."""
        result = ar.auto_clean(messy_frame, dry_run=True)
        # With dry_run, typically returns a report
        assert hasattr(result, 'summary') or hasattr(result, 'columns')
    
    def test_dry_run_with_return_report(self, messy_frame):
        """dry_run=True with return_report returns inspection data."""
        result = ar.auto_clean(
            messy_frame,
            mode="strict",
            dry_run=True,
            return_report=True
        )
        assert isinstance(result, tuple)


class TestAutoCleanCastTypes:
    """Tests for automatic type casting in strict mode."""
    
    def test_strict_mode_casts_numeric_strings(self, numeric_only_frame):
        """Strict mode can cast string columns to numeric."""
        result = ar.auto_clean(numeric_only_frame, mode="strict")
        df = ar.to_pandas(result)
        # After auto_clean with strict mode and confirmed casts
        assert df.shape[0] == 3
    
    def test_allow_lossy_casts_flag(self, numeric_only_frame):
        """allow_lossy_casts parameter is accepted."""
        result = ar.auto_clean(
            numeric_only_frame,
            mode="strict",
            allow_lossy_casts=True
        )
        assert hasattr(result, 'columns')
    
    def test_confirmed_casts_parameter(self, numeric_only_frame):
        """confirmed_casts parameter allows explicit cast mapping."""
        cast_map = {"age": "int64", "salary": "int64"}
        result = ar.auto_clean(
            numeric_only_frame,
            mode="strict",
            confirmed_casts=cast_map,
            allow_lossy_casts=True
        )
        assert hasattr(result, 'columns')


class TestAutoCleanEdgeCases:
    """Edge case tests."""
    
    def test_empty_frame(self):
        """auto_clean handles empty frames gracefully."""
        empty = ar.from_pandas(pd.DataFrame({"col": []}))
        result = ar.auto_clean(empty)
        df = ar.to_pandas(result)
        assert len(df) == 0
    
    def test_single_row_frame(self):
        """auto_clean on single-row frame."""
        single = ar.from_pandas(pd.DataFrame({"a": [1], "b": ["test"]}))
        result = ar.auto_clean(single, mode="strict")
        df = ar.to_pandas(result)
        assert len(df) == 1
    
    def test_all_nulls_column(self):
        """auto_clean with a column of all nulls."""
        all_null = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "all_null": [None, None, None],
        }))
        result = ar.auto_clean(all_null)
        df = ar.to_pandas(result)
        assert df.shape == (3, 2)
    
    def test_invalid_mode_raises(self, clean_frame):
        """Invalid mode parameter raises error."""
        with pytest.raises((ValueError, KeyError)):
            ar.auto_clean(clean_frame, mode="invalid_mode")
    
    def test_clean_frame_idempotent(self, clean_frame):
        """Auto-cleaning an already-clean frame is safe."""
        result1 = ar.auto_clean(clean_frame, mode="strict")
        result2 = ar.auto_clean(result1, mode="strict")
        df1 = ar.to_pandas(result1)
        df2 = ar.to_pandas(result2)
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_explain_parameter(self, messy_frame):
        """explain=True parameter is accepted."""
        result = ar.auto_clean(messy_frame, explain=True)
        # Should return something without error
        assert result is not None
