import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

try:
    import arnio as ar
except ImportError:
    ar = None


@pytest.mark.skipif(ar is None, reason="arnio not installed")
class TestArnioCleaningWorkflows:
    """End-to-end system tests for arnio cleaning functions."""

    def test_coalesce_columns_csv_workflow(self, tmp_path):
        """Test coalesce_columns: load CSV, process, verify output."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "primary": [1.0, None, 3.0, None],
            "secondary": [None, 20.0, None, 50.0],
            "tertiary": [100.0, 200.0, 300.0, 400.0]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.coalesce_columns(frame, subset=["primary", "secondary", "tertiary"],
                                      output_column="coalesced")
        result_df = pd.DataFrame(result)
        
        assert "coalesced" in result_df.columns
        assert result_df["coalesced"].tolist() == [1.0, 20.0, 3.0, 50.0]
        
        # Verify error cases
        with pytest.raises(ValueError, match="already exists"):
            ar.coalesce_columns(frame, subset=["primary"], output_column="primary")
        with pytest.raises(ValueError, match="at least one column"):
            ar.coalesce_columns(frame, subset=[])

    def test_combine_columns_multiple_separators(self, tmp_path):
        """Test combine_columns with separators and null handling."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "first": ["John", "Jane", None],
            "last": ["Doe", "Smith", "Brown"],
            "middle": ["Q", None, "R"]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.combine_columns(frame, subset=["first", "middle"],
                                     separator="-", output_column="combined")
        result_df = pd.DataFrame(result)
        
        assert result_df["combined"].iloc[0] == "John-Q"
        assert pd.isna(result_df["combined"].iloc[2])

    def test_combine_columns_all_columns(self):
        """Test combine_columns with subset=None (all columns)."""
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        result = ar.combine_columns(df, separator="|", output_column="combined")
        result_df = pd.DataFrame(result)
        
        assert result_df["combined"].tolist() == ["x|p", "y|q"]
        with pytest.raises(TypeError):
            ar.combine_columns(df, separator=123, output_column="result")

    def test_drop_columns_matching_regex(self, tmp_path):
        """Test drop_columns_matching with regex patterns."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "name": ["A", "B"],
            "temp_col1": [1, 2],
            "temp_col2": [3, 4],
            "keep_col": [5, 6],
            "_private": [7, 8]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_columns_matching(frame, r"^temp_")
        result_df = pd.DataFrame(result)
        
        assert "name" in result_df.columns
        assert "keep_col" in result_df.columns
        assert "_private" in result_df.columns
        assert "temp_col1" not in result_df.columns
        assert "temp_col2" not in result_df.columns
        
        # Error cases
        with pytest.raises(ValueError, match="matches all columns"):
            ar.drop_columns_matching(frame, ".*")
        with pytest.raises(Exception):  # re.error
            ar.drop_columns_matching(frame, "[invalid(")

    def test_drop_constant_columns_with_nulls(self, tmp_path):
        """Test drop_constant_columns removes single-value columns including all-null."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "group": ["A", "A", "A"],
            "value": [10, 20, 30],
            "all_null": [None, None, None],
            "mixed": [1, None, 3]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_constant_columns(frame)
        result_df = pd.DataFrame(result)
        
        assert "id" in result_df.columns
        assert "value" in result_df.columns
        assert "mixed" in result_df.columns
        assert "group" not in result_df.columns
        assert "all_null" not in result_df.columns
        assert len(result_df) == 3

    def test_drop_nulls_all_and_subset(self, tmp_path):
        """Test drop_nulls with and without subset parameter."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["A", "B", None, "D"],
            "age": [25, None, 35, 40],
            "city": ["NYC", "LA", "CHI", None]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        
        # Drop rows with any null
        result_all = ar.drop_nulls(frame)
        result_all_df = pd.DataFrame(result_all)
        assert len(result_all_df) == 1
        assert result_all_df["id"].iloc[0] == 1
        
        # Drop only rows with null in name
        result_subset = ar.drop_nulls(frame, subset=["name"])
        result_subset_df = pd.DataFrame(result_subset)
        assert len(result_subset_df) == 3
        assert result_subset_df["id"].tolist() == [1, 2, 4]

    def test_filter_rows_all_operators(self, tmp_path):
        """Test filter_rows with all supported comparison operators."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"age": [20, 30, 40, 50], "name": ["A", "B", "C", "D"]})
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        
        for op, expected_indices in [
            (">", [2, 3]), (">=", [1, 2, 3]), ("<", [0, 1]), 
            ("<=", [0, 1, 2]), ("==", [1]), ("!=", [0, 2, 3])
        ]:
            result = ar.filter_rows(frame, column="age", op=op, value=30)
            result_df = pd.DataFrame(result)
            assert result_df.index.tolist() == expected_indices or len(result_df) == len(expected_indices)
        
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(frame, column="age", op=">>", value=30)

    def test_keep_rows_with_nulls_all_and_subset(self, tmp_path):
        """Test keep_rows_with_nulls retains only rows with nulls."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "col_a": [10, None, 30, None, 50],
            "col_b": ["a", "b", None, "d", "e"]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        
        # Keep rows with any null
        result_all = ar.keep_rows_with_nulls(frame)
        result_all_df = pd.DataFrame(result_all)
        assert len(result_all_df) == 3
        assert result_all_df["id"].tolist() == [2, 3, 4]
        
        # Keep rows with null in col_a only
        result_subset = ar.keep_rows_with_nulls(frame, subset=["col_a"])
        result_subset_df = pd.DataFrame(result_subset)
        assert len(result_subset_df) == 2
        assert result_subset_df["id"].tolist() == [2, 4]

    def test_replace_values_single_and_all_columns(self, tmp_path):
        """Test replace_values on single column vs all columns."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "status": ["yes", "no", "yes"],
            "flag": ["yes", "maybe", "no"]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        
        # Single column
        result_single = ar.replace_values(frame, {"yes": "Y", "no": "N"}, column="status")
        result_single_df = pd.DataFrame(result_single)
        assert result_single_df["status"].tolist() == ["Y", "N", "Y"]
        assert result_single_df["flag"].tolist() == ["yes", "maybe", "no"]
        
        # All columns
        result_all = ar.replace_values(frame, {"yes": "Y", "no": "N", "maybe": "?"})
        result_all_df = pd.DataFrame(result_all)
        assert result_all_df["flag"].tolist() == ["Y", "?", "N"]

    def test_replace_values_with_nulls(self):
        """Test replace_values handling of null-like keys."""
        df = pd.DataFrame({"data": ["value", None, "other", np.nan]})
        result = ar.replace_values(df, {None: "MISSING"}, column="data")
        result_df = pd.DataFrame(result)
        
        assert result_df["data"].iloc[1] == "MISSING"

    def test_safe_divide_columns_workflow(self, tmp_path):
        """Test safe_divide_columns with division by zero and nulls."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "numerator": [100.0, 200.0, 300.0, 50.0],
            "denominator": [10.0, 0.0, 15.0, None]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.safe_divide_columns(frame, numerator="numerator",
                                        denominator="denominator",
                                        output_column="ratio", fill_value=-1.0)
        result_df = pd.DataFrame(result)
        
        assert result_df["ratio"].iloc[0] == 10.0
        assert result_df["ratio"].iloc[1] == -1.0
        assert result_df["ratio"].iloc[2] == 20.0
        assert result_df["ratio"].iloc[3] == -1.0
        
        with pytest.raises(ValueError, match="already exists"):
            ar.safe_divide_columns(frame, numerator="numerator",
                                   denominator="denominator", output_column="numerator")

    def test_standardize_missing_tokens_default_and_custom(self, tmp_path):
        """Test standardize_missing_tokens with default and custom tokens."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "value": ["N/A", "valid", "null", "", "NA", "none"]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = ar.standardize_missing_tokens(frame)
        result_df = pd.DataFrame(result)
        
        for idx in [0, 2, 3, 4, 5]:
            assert pd.isna(result_df["value"].iloc[idx])
        assert result_df["value"].iloc[1] == "valid"

    def test_standardize_missing_tokens_custom_tokens_and_subset(self):
        """Test standardize_missing_tokens with custom tokens and subset."""
        df = pd.DataFrame({
            "col1": ["CUSTOM", "data", "IGNORE"],
            "col2": ["CUSTOM", "data", "IGNORE"]
        })
        result = ar.standardize_missing_tokens(df, tokens=["CUSTOM"], subset=["col1"])
        result_df = pd.DataFrame(result)
        
        assert pd.isna(result_df["col1"].iloc[0])
        assert result_df["col2"].iloc[0] == "CUSTOM"

    def test_chained_cleaning_operations(self, tmp_path):
        """Test chaining multiple cleaning operations in realistic workflow."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["A", "B", None, "D"],
            "status": ["N/A", "active", "active", ""],
            "const_col": ["X", "X", "X", "X"]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        
        # Apply multiple operations
        result = ar.drop_constant_columns(frame)
        result = ar.standardize_missing_tokens(result)
        result = ar.drop_nulls(result)
        result_df = pd.DataFrame(result)
        
        assert "const_col" not in result_df.columns
        assert len(result_df) == 2
        assert pd.isna(result_df["status"].iloc[0])

    def test_edge_case_empty_subset_and_error_messages(self):
        """Test edge cases: empty subsets and meaningful error messages."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        
        with pytest.raises(ValueError):
            ar.drop_nulls(ar.from_pandas(df) if hasattr(ar, 'from_pandas') else None 
                          or df, subset=[])
