import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDataQualityReportToDict:
    """Test suite for DataQualityReport.to_dict method."""

    def test_to_dict_returns_dict(self):
        """Test that to_dict returns a dictionary."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_required_top_level_keys(self):
        """Test that to_dict includes expected top-level keys."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        required_keys = {"row_count", "column_count", "columns"}
        assert required_keys.issubset(set(result.keys()))

    def test_to_dict_columns_dict_structure(self):
        """Test that columns key contains dict of column profiles."""
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["x", "y"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert isinstance(result["columns"], dict)
        assert set(result["columns"].keys()) == {"a", "b", "c"}

    def test_to_dict_numeric_dtype_preserved(self):
        """Test that numeric column dtypes are preserved in output."""
        df = pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.5, 2.5, 3.5]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert result["columns"]["int_col"]["dtype"] == "int64"
        assert result["columns"]["float_col"]["dtype"] == "float64"

    def test_to_dict_string_dtype_preserved(self):
        """Test that string column dtype is preserved in output."""
        df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert result["columns"]["name"]["dtype"] == "string"

    def test_to_dict_null_count_recorded(self):
        """Test that null counts are recorded in column profiles."""
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert result["columns"]["a"]["null_count"] == 1
        assert result["columns"]["b"]["null_count"] == 1

    def test_to_dict_all_nulls(self):
        """Test to_dict with a column that is all nulls."""
        df = pd.DataFrame({"col": [None, None, None]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert result["columns"]["col"]["null_count"] == 3

    def test_to_dict_row_count_matches_data(self):
        """Test that row_count matches the actual number of rows."""
        for n_rows in [0, 1, 5, 100]:
            df = pd.DataFrame({"col": range(n_rows)})
            frame = ar.from_pandas(df)
            report = ar.profile(frame)
            result = report.to_dict()
            assert result["row_count"] == n_rows

    def test_to_dict_column_count_matches_data(self):
        """Test that column_count matches the actual number of columns."""
        for n_cols in [1, 3, 5]:
            data = {f"col_{i}": [1, 2] for i in range(n_cols)}
            df = pd.DataFrame(data)
            frame = ar.from_pandas(df)
            report = ar.profile(frame)
            result = report.to_dict()
            assert result["column_count"] == n_cols

    def test_to_dict_redact_sample_values_false_default(self):
        """Test that sample_values are present by default (not redacted)."""
        df = pd.DataFrame({"col": ["alice", "bob", "carol"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=2)
        result = report.to_dict()
        
        col_profile = result["columns"]["col"]
        if "sample_values" in col_profile:
            # At least one sample value should not be redacted
            assert any(val != "[REDACTED]" for val in col_profile["sample_values"])

    def test_to_dict_redact_sample_values_true(self):
        """Test that sample_values are redacted when flag is True."""
        df = pd.DataFrame({"col": ["alice", "bob", "carol"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=2)
        result = report.to_dict(redact_sample_values=True)
        
        col_profile = result["columns"]["col"]
        if "sample_values" in col_profile:
            for val in col_profile["sample_values"]:
                assert val == "[REDACTED]"

    def test_to_dict_redact_top_values(self):
        """Test that top_values are redacted when redact_sample_values=True."""
        df = pd.DataFrame({"status": ["active", "active", "inactive", "pending"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict(redact_sample_values=True)
        
        col_profile = result["columns"]["status"]
        if "top_values" in col_profile and col_profile["top_values"]:
            for item in col_profile["top_values"]:
                assert item["value"] == "[REDACTED]"
                assert isinstance(item.get("count", 0), (int, float))
                assert isinstance(item.get("ratio", 0.0), (int, float))

    def test_to_dict_exclude_columns_single_string(self):
        """Test excluding a single column using list."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=["a"])
        
        assert "a" not in result["columns"]
        assert "b" in result["columns"]
        assert "c" in result["columns"]

    def test_to_dict_exclude_columns_multiple_list(self):
        """Test excluding multiple columns using a list."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=["a", "c"])
        
        assert "a" not in result["columns"]
        assert "b" in result["columns"]
        assert "c" not in result["columns"]
        assert "d" in result["columns"]

    def test_to_dict_exclude_columns_set(self):
        """Test excluding columns using a set."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns={"a", "c"})
        
        assert "a" not in result["columns"]
        assert "b" in result["columns"]
        assert "c" not in result["columns"]

    def test_to_dict_exclude_columns_tuple(self):
        """Test excluding columns using a tuple."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=("a", "c"))
        
        assert "a" not in result["columns"]
        assert "b" in result["columns"]
        assert "c" not in result["columns"]

    def test_to_dict_exclude_columns_empty_list(self):
        """Test exclude_columns with empty list includes all columns."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=[])
        
        assert "a" in result["columns"]
        assert "b" in result["columns"]

    def test_to_dict_exclude_columns_none_includes_all(self):
        """Test exclude_columns=None includes all columns (default behavior)."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=None)
        
        assert "a" in result["columns"]
        assert "b" in result["columns"]
        assert "c" in result["columns"]

    def test_to_dict_exclude_nonexistent_column(self):
        """Test that excluding non-existent columns is handled gracefully."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=["a", "nonexistent"])
        
        assert "a" not in result["columns"]
        assert "b" in result["columns"]

    def test_to_dict_exclude_all_columns(self):
        """Test excluding all columns results in empty columns dict."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=["a", "b"])
        
        assert result["columns"] == {}

    def test_to_dict_combined_redact_and_exclude(self):
        """Test combining redact_sample_values and exclude_columns."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "email": ["alice@example.com", "bob@example.com"],
            "age": [25, 30]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=1)
        result = report.to_dict(
            redact_sample_values=True,
            exclude_columns=["email"]
        )
        
        assert "email" not in result["columns"]
        assert "name" in result["columns"]
        assert "age" in result["columns"]
        if "sample_values" in result["columns"]["name"]:
            for val in result["columns"]["name"]["sample_values"]:
                assert val == "[REDACTED]"

    def test_to_dict_quality_score_present(self):
        """Test that quality_score is in the output."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert "quality_score" in result
        assert isinstance(result["quality_score"], (int, float))

    def test_to_dict_boolean_column(self):
        """Test to_dict with boolean column."""
        df = pd.DataFrame({"flag": [True, False, True]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert result["columns"]["flag"]["dtype"] == "bool"

    def test_to_dict_mixed_column_types(self):
        """Test to_dict preserves all column type information."""
        df = pd.DataFrame({
            "ints": [1, 2],
            "floats": [1.5, 2.5],
            "strings": ["a", "b"],
            "bools": [True, False]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert result["columns"]["ints"]["dtype"] == "int64"
        assert result["columns"]["floats"]["dtype"] == "float64"
        assert result["columns"]["strings"]["dtype"] == "string"
        assert result["columns"]["bools"]["dtype"] == "bool"

    def test_to_dict_duplicate_rows_recorded(self):
        """Test that duplicate row information is recorded."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert "duplicate_rows" in result

    def test_to_dict_json_serializable(self):
        """Test that to_dict output is JSON-serializable."""
        import json
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_to_dict_with_sample_size_zero(self):
        """Test to_dict when profile was created with sample_size=0."""
        df = pd.DataFrame({"col": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=0)
        result = report.to_dict()
        
        col_profile = result["columns"]["col"]
        if "sample_values" in col_profile:
            assert col_profile["sample_values"] == []

    def test_to_dict_empty_dataframe(self):
        """Test to_dict with an empty DataFrame."""
        df = pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="float64")})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert result["row_count"] == 0
        assert "a" in result["columns"]
        assert "b" in result["columns"]

    def test_to_dict_keyword_only_arguments(self):
        """Test that arguments must be passed as keywords."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        # Positional arguments should raise TypeError
        with pytest.raises(TypeError):
            report.to_dict(True)

    def test_to_dict_preserves_reported_metrics(self):
        """Test that numeric metrics in output are consistent."""
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        # Call twice to ensure consistency
        result2 = report.to_dict()
        assert result["row_count"] == result2["row_count"]
        assert result["column_count"] == result2["column_count"]
