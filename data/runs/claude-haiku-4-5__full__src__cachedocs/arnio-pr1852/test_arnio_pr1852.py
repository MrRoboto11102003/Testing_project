import pytest
import pandas as pd
import numpy as np
import re
import arnio as ar


class TestCoalesceColumns:
    def test_coalesce_selects_first_non_null(self):
        """Coalesce selects first non-null value from columns."""
        df = pd.DataFrame({"a": [None, 2, None], "b": [10, None, 30], "c": [100, 200, 300]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b", "c"], output_column="result")
        result_df = ar.to_pandas(result)
        assert result_df["result"].tolist() == [10.0, 2.0, 30.0]

    def test_coalesce_raises_empty_subset(self):
        """Coalesce raises ValueError on empty subset."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="subset must contain at least one column"):
            ar.coalesce_columns(frame, subset=[])


class TestCombineColumns:
    def test_combine_with_separator(self):
        """Combine concatenates columns with separator."""
        df = pd.DataFrame({"first": ["John", "Jane"], "last": ["Doe", "Smith"]})
        frame = ar.from_pandas(df)
        result = ar.combine_columns(frame, subset=["first", "last"], separator=" ", output_column="full_name")
        result_df = ar.to_pandas(result)
        assert result_df["full_name"].tolist() == ["John Doe", "Jane Smith"]

    def test_combine_raises_on_existing_output(self):
        """Combine raises ValueError if output column already exists."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
        with pytest.raises(ValueError, match="already exists"):
            ar.combine_columns(frame, subset=["a"], output_column="a")


class TestDropColumnsMatching:
    def test_drop_matching_regex_pattern(self):
        """Drop columns matching regex pattern."""
        df = pd.DataFrame({"temp_a": [1], "temp_b": [2], "keep_c": [3]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, r"^temp_")
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["keep_c"]

    def test_drop_matching_raises_on_all_match(self):
        """Drop matching raises ValueError if pattern matches all columns."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
        with pytest.raises(ValueError, match="At least one column must remain"):
            ar.drop_columns_matching(frame, ".*")

    def test_drop_matching_invalid_regex_raises(self):
        """Drop matching raises re.error on invalid regex."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(re.error):
            ar.drop_columns_matching(frame, "[invalid")


class TestDropConstantColumns:
    def test_drop_constant_single_value(self):
        """Drop constant columns with single unique value."""
        df = pd.DataFrame({"const": [5, 5, 5], "var": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["var"]

    def test_keep_column_with_one_value_and_nulls(self):
        """Keep columns with one non-null value and nulls."""
        df = pd.DataFrame({"const_null": [1, 1, None], "var": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        assert "const_null" in result_df.columns


class TestDropNulls:
    def test_drop_rows_with_nulls(self):
        """Drop rows containing null values."""
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})
        frame = ar.from_pandas(df)
        result = ar.drop_nulls(frame)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 2
        assert result_df["a"].tolist() == [1.0, 3.0]

    def test_drop_nulls_with_subset(self):
        """Drop nulls checks only subset columns."""
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})
        frame = ar.from_pandas(df)
        result = ar.drop_nulls(frame, subset=["b"])
        result_df = ar.to_pandas(result)
        assert len(result_df) == 3  # No nulls in 'b'


class TestFilterRows:
    def test_filter_with_gt_operator(self):
        """Filter rows using > operator."""
        df = pd.DataFrame({"age": [18, 25, 16, 30]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op=">", value=18)
        result_df = ar.to_pandas(result)
        assert result_df["age"].tolist() == [25, 30]

    def test_filter_invalid_operator_raises(self):
        """Filter raises ValueError on unsupported operator."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(frame, column="a", op=">>>", value=1)


class TestKeepRowsWithNulls:
    def test_keep_rows_with_any_null(self):
        """Keep only rows with at least one null."""
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 5.0, 6.0]})
        frame = ar.from_pandas(df)
        result = ar.keep_rows_with_nulls(frame)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 2


class TestReplaceValues:
    def test_replace_in_specific_column(self):
        """Replace values in specific column."""
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})
        frame = ar.from_pandas(df)
        result = ar.replace_values(frame, {"active": "A", "inactive": "I"}, column="status")
        result_df = ar.to_pandas(result)
        assert result_df["status"].tolist() == ["A", "I", "A"]

    def test_replace_missing_column_raises(self):
        """Replace raises KeyError on missing column."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(KeyError):
            ar.replace_values(frame, {1: 2}, column="missing")


class TestSafeDivideColumns:
    def test_safe_divide_normal_division(self):
        """Safe divide performs normal division."""
        df = pd.DataFrame({"revenue": [100.0, 200.0, 300.0], "cost": [10.0, 20.0, 30.0]})
        frame = ar.from_pandas(df)
        result = ar.safe_divide_columns(frame, numerator="revenue", denominator="cost", output_column="ratio")
        result_df = ar.to_pandas(result)
        assert np.allclose(result_df["ratio"].tolist(), [10.0, 10.0, 10.0])

    def test_safe_divide_zero_denominator_uses_fill_value(self):
        """Safe divide uses fill_value for zero denominators."""
        df = pd.DataFrame({"num": [10.0, 20.0], "denom": [0.0, 5.0]})
        frame = ar.from_pandas(df)
        result = ar.safe_divide_columns(frame, numerator="num", denominator="denom", 
                                       output_column="ratio", fill_value=999.0)
        result_df = ar.to_pandas(result)
        assert result_df["ratio"].tolist() == [999.0, 4.0]

    def test_safe_divide_existing_output_raises(self):
        """Safe divide raises ValueError if output column exists."""
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="already exists"):
            ar.safe_divide_columns(frame, numerator="a", denominator="b", output_column="a")


class TestStandardizeMissingTokens:
    def test_standardize_default_tokens(self):
        """Standardize converts default missing tokens to NaN."""
        df = pd.DataFrame({"value": ["1", "2", "N/A", "null"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df["value"].iloc[2])
        assert pd.isna(result_df["value"].iloc[3])

    def test_standardize_custom_tokens(self):
        """Standardize converts custom tokens to NaN."""
        df = pd.DataFrame({"value": ["1", "2", "MISSING"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=["MISSING"])
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df["value"].iloc[2])

    def test_standardize_case_insensitive(self):
        """Standardize matching is case-insensitive."""
        df = pd.DataFrame({"value": ["1", "n/a", "N/A", "nA"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        assert sum(pd.isna(result_df["value"])) == 3
