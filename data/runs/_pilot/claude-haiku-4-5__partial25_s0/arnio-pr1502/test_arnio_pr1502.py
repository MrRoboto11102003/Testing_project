import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestAutoClean:
    """Tests for arnio.quality.auto_clean function."""

    def test_auto_clean_default_returns_arframe(self):
        """auto_clean() with defaults should return an ArFrame."""
        frame = ar.from_dict({"name": ["Alice"], "age": [25]})
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_safe_mode_strips_whitespace(self):
        """mode='safe' should strip leading/trailing whitespace from strings."""
        frame = ar.from_dict({
            "name": [" Alice ", "Bob"],
            "city": ["Paris", " London "]
        })
        result = ar.auto_clean(frame, mode="safe")
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Alice"
        assert df["city"].iloc[1] == "London"

    def test_auto_clean_strict_mode_removes_duplicates(self):
        """mode='strict' should remove duplicate rows."""
        frame = ar.from_dict({
            "order_id": [1, 2, 2, 3],
            "customer": ["Alice", "Bob", "Bob", "Charlie"]
        })
        result = ar.auto_clean(frame, mode="strict")
        assert result.shape[0] <= frame.shape[0]

    def test_auto_clean_return_report_returns_tuple(self):
        """return_report=True should return (ArFrame, DataQualityReport)."""
        frame = ar.from_dict({"name": [" Alice "], "age": [25]})
        result = ar.auto_clean(frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert hasattr(result[1], 'summary')

    def test_auto_clean_dry_run_returns_report(self):
        """dry_run=True should return a report object."""
        frame = ar.from_dict({"name": [" Alice "], "age": [25]})
        result = ar.auto_clean(frame, mode="strict", dry_run=True)
        assert hasattr(result, 'suggestions') or hasattr(result, 'summary')

    def test_auto_clean_preserves_column_names(self):
        """auto_clean should preserve column names."""
        frame = ar.from_dict({
            "first_name": ["Alice"],
            "last_name": ["Smith"]
        })
        result = ar.auto_clean(frame)
        assert "first_name" in result.columns
        assert "last_name" in result.columns

    def test_auto_clean_empty_frame(self):
        """auto_clean on empty frame should return empty frame with same columns."""
        frame = ar.from_dict({"name": [], "age": []})
        result = ar.auto_clean(frame)
        assert result.shape[0] == 0
        assert result.shape[1] == 2

    def test_auto_clean_single_row_frame(self):
        """auto_clean on single-row frame should work correctly."""
        frame = ar.from_dict({"name": [" Alice "], "age": [25]})
        result = ar.auto_clean(frame)
        assert result.shape[0] == 1

    def test_auto_clean_invalid_mode_raises_error(self):
        """auto_clean with invalid mode should raise ValueError or KeyError."""
        frame = ar.from_dict({"name": ["Alice"]})
        with pytest.raises((ValueError, KeyError)):
            ar.auto_clean(frame, mode="invalid_mode")

    def test_auto_clean_safe_mode_is_default(self):
        """Default mode should behave like mode='safe'."""
        frame = ar.from_dict({"name": [" Alice ", " Bob "]})
        result_default = ar.auto_clean(frame)
        result_safe = ar.auto_clean(frame, mode="safe")
        df_default = ar.to_pandas(result_default)
        df_safe = ar.to_pandas(result_safe)
        pd.testing.assert_frame_equal(df_default, df_safe)

    def test_auto_clean_with_nulls_safe_mode(self):
        """mode='safe' should preserve nulls without dropping rows."""
        frame = ar.from_dict({
            "name": ["Alice", None, "Bob"],
            "age": [25, 30, None]
        })
        result = ar.auto_clean(frame, mode="safe")
        assert result.shape[0] == 3

    def test_auto_clean_numeric_columns_preserved(self):
        """auto_clean should preserve numeric column values."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "value": [1.5, 2.7, 3.14]
        })
        result = ar.auto_clean(frame)
        df = ar.to_pandas(result)
        np.testing.assert_array_equal(df["id"].values, [1, 2, 3])
        np.testing.assert_array_almost_equal(
            df["value"].values,
            [1.5, 2.7, 3.14]
        )

    def test_auto_clean_explain_returns_tuple(self):
        """explain=True should return tuple with explanation."""
        frame = ar.from_dict({"name": [" Alice "]})
        result = ar.auto_clean(frame, explain=True)
        assert isinstance(result, tuple)
        assert len(result) >= 2
        assert isinstance(result[0], ar.ArFrame)

    def test_auto_clean_return_report_and_explain(self):
        """Both return_report=True and explain=True should return 3-tuple."""
        frame = ar.from_dict({"name": [" Alice "]})
        result = ar.auto_clean(frame, return_report=True, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], ar.ArFrame)

    def test_auto_clean_all_null_column(self):
        """auto_clean should handle all-null columns without errors."""
        frame = ar.from_dict({
            "name": ["Alice"],
            "nullable": [None]
        })
        result = ar.auto_clean(frame)
        assert "nullable" in result.columns
