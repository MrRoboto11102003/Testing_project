import pytest
import numpy as np
import pandas as pd
import arnio as ar
import re


class TestFilterRows:
    """Tests for arnio.cleaning.filter_rows"""

    def test_filter_rows_arframe_basic(self):
        """Filter ArFrame returns ArFrame with reset index"""
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(pdf)
        result = ar.filter_rows(frame, "a", ">", 1)
        
        assert isinstance(result, ar.ArFrame)
        out = ar.to_pandas(result)
        assert out["a"].tolist() == [2, 3]
        assert list(out.index) == [0, 1]

    def test_filter_rows_dataframe_basic(self):
        """Filter DataFrame keeps original index"""
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = ar.filter_rows(pdf, "a", ">", 1)
        
        assert isinstance(result, pd.DataFrame)
        assert result["a"].tolist() == [2, 3]
        assert list(result.index) == [1, 2]

    def test_filter_rows_all_operators(self):
        """All six operators work correctly"""
        pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        
        assert ar.filter_rows(pdf, "a", ">", 2)["a"].tolist() == [3, 4, 5]
        assert ar.filter_rows(pdf, "a", "<", 3)["a"].tolist() == [1, 2]
        assert ar.filter_rows(pdf, "a", ">=", 3)["a"].tolist() == [3, 4, 5]
        assert ar.filter_rows(pdf, "a", "<=", 2)["a"].tolist() == [1, 2]
        assert ar.filter_rows(pdf, "a", "==", 2)["a"].tolist() == [2]
        assert ar.filter_rows(pdf, "a", "!=", 2)["a"].tolist() == [1, 3, 4, 5]

    def test_filter_rows_no_matches(self):
        """No matching rows returns empty frame with same columns"""
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = ar.filter_rows(pdf, "a", ">", 100)
        
        assert len(result) == 0
        assert list(result.columns) == ["a", "b"]

    def test_filter_rows_error_invalid_frame_type(self):
        """TypeError for invalid frame type"""
        with pytest.raises(TypeError, match="must be an ArFrame or a pandas DataFrame"):
            ar.filter_rows({"a": [1]}, "a", ">", 1)

    def test_filter_rows_error_column_not_string(self):
        """TypeError for non-string column"""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(pdf, 5, ">", 1)

    def test_filter_rows_error_column_empty_or_whitespace(self):
        """TypeError for empty or whitespace-only column"""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(pdf, "", ">", 1)
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(pdf, "   ", ">", 1)

    def test_filter_rows_error_op_not_string(self):
        """TypeError for non-string op"""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="op must be a string"):
            ar.filter_rows(pdf, "a", 123, 1)

    def test_filter_rows_error_unsupported_operator(self):
        """ValueError for unsupported operator (checked before column exists)"""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match=re.escape("Unsupported operator: =")):
            ar.filter_rows(pdf, "missing_col", "=", 1)

    def test_filter_rows_error_missing_column(self):
        """ValueError for column not in DataFrame"""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown column: zzz"):
            ar.filter_rows(pdf, "zzz", ">", 1)

    def test_filter_rows_error_non_scalar_value(self):
        """TypeError for non-scalar value"""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="value must be a scalar"):
            ar.filter_rows(pdf, "a", ">", [1, 2])

    def test_filter_rows_error_incompatible_comparison(self):
        """TypeError when comparing incompatible types"""
        pdf = pd.DataFrame({"s": ["a", "b"]})
        with pytest.raises(TypeError, match="cannot compare"):
            ar.filter_rows(pdf, "s", ">", 1)

    def test_filter_rows_float_nan_excluded_by_operators(self):
        """NaN in float column is excluded by comparison operators"""
        pdf = pd.DataFrame({"f": [1.0, float("nan"), 3.0]})
        result = ar.filter_rows(pdf, "f", ">", 0)
        assert len(result) == 2

    def test_filter_rows_float_nan_with_not_equal(self):
        """NaN is kept with != operator (NaN != x is True in numpy)"""
        pdf = pd.DataFrame({"f": [1.0, float("nan"), 3.0]})
        result = ar.filter_rows(pdf, "f", "!=", 1.0)
        assert len(result) == 2

    def test_filter_rows_preserves_dtypes_and_columns(self):
        """Filtering preserves column order and dtypes"""
        pdf = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.5, 20.3, 30.1],
            "name": ["a", "b", "c"]
        })
        result = ar.filter_rows(pdf, "id", ">", 1)
        
        assert list(result.columns) == ["id", "value", "name"]
        assert len(result) == 2
