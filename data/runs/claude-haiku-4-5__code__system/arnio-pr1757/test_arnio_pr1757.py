import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
from pathlib import Path


import arnio as ar


class TestDropColumnsBasic:
    """Basic functionality tests for drop_columns."""

    def test_drop_single_column(self, tmp_path):
        """Test dropping a single column from a frame."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "debug": ["x", "y", "z"]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        result = ar.drop_columns(frame, ["debug"])
        result_df = ar.to_pandas(result)

        assert list(result_df.columns) == ["id", "name"]
        assert len(result_df) == 3

    def test_drop_multiple_columns(self, tmp_path):
        """Test dropping multiple columns at once."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
            "d": [10, 11, 12]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        result = ar.drop_columns(frame, ["b", "d"])
        result_df = ar.to_pandas(result)

        assert list(result_df.columns) == ["a", "c"]
        assert result_df.shape == (3, 2)

    def test_drop_columns_data_integrity(self, tmp_path):
        """Test that data values are preserved after dropping columns."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.5, 20.3, 30.1],
            "junk": ["x", "y", "z"]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        result = ar.drop_columns(frame, ["junk"])
        result_df = ar.to_pandas(result)

        assert result_df["id"].tolist() == [1, 2, 3]
        assert np.allclose(result_df["value"].values, [10.5, 20.3, 30.1])

    def test_drop_columns_save_to_file(self, tmp_path):
        """Test dropping columns and saving result to file."""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9]
        })
        df.to_csv(input_file, index=False)

        frame = ar.read_csv(input_file)
        result = ar.drop_columns(frame, ["b"])
        result_df = ar.to_pandas(result)
        result_df.to_csv(output_file, index=False)

        saved_df = pd.read_csv(output_file)
        assert list(saved_df.columns) == ["a", "c"]
        assert len(saved_df) == 3

    def test_drop_columns_original_frame_unchanged(self, tmp_path):
        """Test that the original frame is not modified."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        original_cols = list(ar.to_pandas(frame).columns)

        ar.drop_columns(frame, ["b"])

        frame_after = ar.to_pandas(frame)
        assert list(frame_after.columns) == original_cols


class TestDropColumnsErrors:
    """Test error handling in drop_columns."""

    def test_drop_nonexistent_column_raises_valueerror(self, tmp_path):
        """Test that dropping non-existent column raises ValueError."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        with pytest.raises(ValueError, match="Columns not found in frame"):
            ar.drop_columns(frame, ["missing"])

    def test_drop_all_columns_raises_valueerror(self, tmp_path):
        """Test that dropping all columns raises ValueError."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        with pytest.raises(ValueError, match="cannot remove all columns"):
            ar.drop_columns(frame, ["a", "b"])

    def test_drop_columns_string_input_raises_typeerror(self, tmp_path):
        """Test that passing string instead of sequence raises TypeError."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        with pytest.raises(TypeError):
            ar.drop_columns(frame, "a")

    def test_drop_columns_nonstring_item_raises_typeerror(self, tmp_path):
        """Test that non-string items in sequence raise TypeError."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        with pytest.raises(TypeError):
            ar.drop_columns(frame, ["a", 1])

    def test_drop_partial_missing_columns_raises_valueerror(self, tmp_path):
        """Test that partially invalid column list raises ValueError."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({"a": [1], "b": [2], "c": [3]}).to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        with pytest.raises(ValueError, match="Columns not found in frame"):
            ar.drop_columns(frame, ["a", "missing"])


class TestDropColumnsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_drop_empty_column_list(self, tmp_path):
        """Test that empty column list returns all columns unchanged."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        result = ar.drop_columns(frame, [])
        result_df = ar.to_pandas(result)

        assert list(result_df.columns) == ["a", "b"]
        assert len(result_df) == 2

    def test_drop_columns_with_special_characters(self, tmp_path):
        """Test dropping columns with special characters in names."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "col_1": [1],
            "col-2": [2],
            "col.3": [3]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        result = ar.drop_columns(frame, ["col-2"])
        result_df = ar.to_pandas(result)

        assert "col-2" not in result_df.columns
        assert "col_1" in result_df.columns
        assert "col.3" in result_df.columns

    def test_drop_columns_with_spaces_in_names(self, tmp_path):
        """Test dropping columns with spaces in names."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "first name": [1, 2],
            "last name": [3, 4],
            "debug col": [5, 6]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        result = ar.drop_columns(frame, ["debug col"])
        result_df = ar.to_pandas(result)

        assert list(result_df.columns) == ["first name", "last name"]

    def test_drop_columns_case_sensitive(self, tmp_path):
        """Test that column matching is case-sensitive."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"Name": [1], "name": [2]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        result = ar.drop_columns(frame, ["Name"])
        result_df = ar.to_pandas(result)

        assert list(result_df.columns) == ["name"]

    def test_drop_columns_duplicate_names_in_list(self, tmp_path):
        """Test dropping with duplicate column names in drop list."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        result = ar.drop_columns(frame, ["a", "a", "b"])
        result_df = ar.to_pandas(result)

        assert list(result_df.columns) == ["c"]

    def test_drop_columns_with_numeric_data(self, tmp_path):
        """Test dropping columns preserves numeric types."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "junk": ["a", "b", "c"]
        })
        df.to_csv(csv_file, index=False)

        frame = ar.read_csv(csv_file)
        result = ar.drop_columns(frame, ["junk"])
        result_df = ar.to_pandas(result)

        assert "int_col" in result_df.columns
        assert "float_col" in result_df.columns
        assert len(result_df) == 3
