import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCombineColumns:
    """Test arnio.cleaning.combine_columns"""

    def test_combine_columns_basic(self):
        """Test combining multiple columns with separator."""
        df = pd.DataFrame({
            "first": ["Alice", "Bob"],
            "last": ["Smith", "Jones"]
        })
        frame = ar.from_pandas(df)
        result = ar.combine_columns(
            frame, subset=["first", "last"], 
            separator=" ", output_column="full_name"
        )
        result_df = ar.to_pandas(result)
        
        assert "full_name" in result_df.columns
        assert result_df["full_name"].tolist() == ["Alice Smith", "Bob Jones"]

    def test_combine_columns_error_existing_output(self):
        """Test ValueError when output_column already exists."""
        df = pd.DataFrame({"a": ["x"], "b": ["y"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.combine_columns(frame, subset=["a", "b"], output_column="a")


class TestFilterRows:
    """Test arnio.cleaning.filter_rows"""

    def test_filter_rows_multiple_operators(self):
        """Test filtering with various comparison operators."""
        df = pd.DataFrame({"age": [18, 25, 30, 16]})
        frame = ar.from_pandas(df)
        
        # Test > operator
        result = ar.filter_rows(frame, column="age", op=">", value=18)
        assert len(ar.to_pandas(result)) == 2
        
        # Test == operator
        result = ar.filter_rows(frame, column="age", op="==", value=18)
        assert len(ar.to_pandas(result)) == 1
        
        # Test >= operator
        result = ar.filter_rows(frame, column="age", op=">=", value=25)
        assert len(ar.to_pandas(result)) == 2

    def test_filter_rows_string_column(self):
        """Test filtering on string values."""
        df = pd.DataFrame({"status": ["active", "inactive", "active", "pending"]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="status", op="==", value="active")
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 2
        assert all(result_df["status"] == "active")


class TestKeepRowsWithNulls:
    """Test arnio.cleaning.keep_rows_with_nulls"""

    def test_keep_rows_with_nulls_any_column(self):
        """Test keeping rows with null values in any column."""
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [4, 5, None]
        })
        frame = ar.from_pandas(df)
        result = ar.keep_rows_with_nulls(frame)
        result_df = ar.to_pandas(result)
        
        # Should keep 2 rows (rows with any null)
        assert len(result_df) == 2
        assert result_df["a"].isna().any() or result_df["b"].isna().any()

    def test_keep_rows_with_nulls_subset(self):
        """Test keeping rows with nulls in specific columns."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, None, 6],
            "c": [None, 8, 9]
        })
        frame = ar.from_pandas(df)
        result = ar.keep_rows_with_nulls(frame, subset=["b"])
        result_df = ar.to_pandas(result)
        
        # Should keep only row where b is null
        assert len(result_df) == 1
        assert pd.isna(result_df["b"].iloc[0])


class TestSafeDivideColumns:
    """Test arnio.cleaning.safe_divide_columns"""

    def test_safe_divide_basic(self):
        """Test basic division of two numeric columns."""
        df = pd.DataFrame({
            "revenue": [100.0, 200.0],
            "cost": [10.0, 20.0]
        })
        frame = ar.from_pandas(df)
        result = ar.safe_divide_columns(
            frame, numerator="revenue", denominator="cost", 
            output_column="ratio"
        )
        result_df = ar.to_pandas(result)
        
        assert "ratio" in result_df.columns
        np.testing.assert_array_almost_equal(
            result_df["ratio"].tolist(), [10.0, 10.0]
        )

    def test_safe_divide_zero_denominator(self):
        """Test that division by zero uses fill_value instead of inf/nan."""
        df = pd.DataFrame({
            "num": [100.0, 200.0],
            "denom": [10.0, 0.0]
        })
        frame = ar.from_pandas(df)
        result = ar.safe_divide_columns(
            frame, numerator="num", denominator="denom",
            output_column="result", fill_value=999.0
        )
        result_df = ar.to_pandas(result)
        
        # When denominator is 0, result should be fill_value
        assert result_df["result"].iloc[1] == 999.0


class TestCoalesceColumns:
    """Test arnio.cleaning.coalesce_columns"""

    def test_coalesce_columns_first_non_null(self):
        """Test selecting first non-null value across columns."""
        df = pd.DataFrame({
            "a": [None, 2, 3],
            "b": [1, None, 4],
            "c": [5, 6, None]
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b", "c"])
        result_df = ar.to_pandas(result)
        
        assert "coalesced" in result_df.columns
        # First row: a is None, b is 1, so result is 1
        assert result_df["coalesced"].iloc[0] == 1.0
        # Second row: a is 2, so result is 2
        assert result_df["coalesced"].iloc[1] == 2.0


class TestDropColumnsMatching:
    """Test arnio.cleaning.drop_columns_matching"""

    def test_drop_columns_matching_regex(self):
        """Test dropping columns by regex pattern."""
        df = pd.DataFrame({
            "temp_1": [1, 2],
            "temp_2": [3, 4],
            "value": [5, 6]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        result_df = ar.to_pandas(result)
        
        assert "value" in result_df.columns
        assert "temp_1" not in result_df.columns
        assert "temp_2" not in result_df.columns


class TestDropConstantColumns:
    """Test arnio.cleaning.drop_constant_columns"""

    def test_drop_constant_columns_single_value(self):
        """Test removing columns with only one unique value."""
        df = pd.DataFrame({
            "constant": [1, 1, 1],
            "variable": [1, 2, 3],
            "another_const": ["a", "a", "a"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert "variable" in result_df.columns
        assert "constant" not in result_df.columns
        assert "another_const" not in result_df.columns


class TestDropNulls:
    """Test arnio.cleaning.drop_nulls"""

    def test_drop_nulls_basic(self):
        """Test removing rows with any null values."""
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [4, 5, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_nulls(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 1
        assert result_df["a"].iloc[0] == 1.0

    def test_drop_nulls_subset(self):
        """Test dropping rows with nulls in specific columns only."""
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [4, 5, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_nulls(frame, subset=["a"])
        result_df = ar.to_pandas(result)
        
        # Should remove only row where column a is None
        assert len(result_df) == 2
        assert result_df["a"].notna().all()


class TestReplaceValues:
    """Test arnio.cleaning.replace_values"""

    def test_replace_values_single_column(self):
        """Test replacing values in a specific column."""
        df = pd.DataFrame({
            "status": ["active", "inactive", "active"],
            "other": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.replace_values(
            frame, {"active": "A", "inactive": "I"}, 
            column="status"
        )
        result_df = ar.to_pandas(result)
        
        assert result_df["status"].tolist() == ["A", "I", "A"]
        assert result_df["other"].tolist() == ["a", "b", "c"]

    def test_replace_values_whole_frame(self):
        """Test replacing values across all columns when no column specified."""
        df = pd.DataFrame({
            "a": ["yes", "no"],
            "b": ["yes", "no"]
        })
        frame = ar.from_pandas(df)
        result = ar.replace_values(frame, {"yes": "Y", "no": "N"})
        result_df = ar.to_pandas(result)
        
        assert result_df["a"].tolist() == ["Y", "N"]
        assert result_df["b"].tolist() == ["Y", "N"]


class TestStandardizeMissingTokens:
    """Test arnio.cleaning.standardize_missing_tokens"""

    def test_standardize_missing_tokens_default(self):
        """Test standardizing common missing value strings to NaN."""
        df = pd.DataFrame({
            "value": ["1", "MISSING", "unknown", "N/A", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        # Common missing tokens should be converted to NaN
        assert pd.isna(result_df["value"].iloc[1])
        assert pd.isna(result_df["value"].iloc[2])
        assert pd.isna(result_df["value"].iloc[3])
        assert result_df["value"].iloc[4] == "valid"
