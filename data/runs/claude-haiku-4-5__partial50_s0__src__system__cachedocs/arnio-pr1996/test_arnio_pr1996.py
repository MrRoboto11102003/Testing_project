import pytest
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path
import csv


class TestDropEmptyColumnsE2E:
    """End-to-end tests for ar.drop_empty_columns() removing fully empty/null columns"""

    def test_basic_drop_single_null_column(self, tmp_path):
        """Drop a single column containing only null values"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,empty_col\n1,Alice,\n2,Bob,\n3,Charlie,\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["id", "name"]
        assert result.shape == (3, 2)
        
        # Verify via pandas roundtrip
        df = ar.to_pandas(result)
        assert list(df.columns) == ["id", "name"]
        assert len(df) == 3

    def test_drop_whitespace_only_columns(self, tmp_path):
        """Remove columns containing only whitespace and empty strings"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,spaces,tabs\n1,Alice,   ,\n2,Bob, ,\t\n3,Charlie,,  \n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert "spaces" not in result.columns
        assert "tabs" not in result.columns
        assert set(result.columns) == {"id", "name"}
        assert result.shape[0] == 3

    def test_keep_partial_empty_columns(self, tmp_path):
        """Keep columns with at least one non-null/non-empty value"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,partial1,partial2,full\n1,,Alice,value\n2,text,,value\n3,,,value\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert "partial1" in result.columns
        assert "partial2" in result.columns
        assert "full" in result.columns
        assert result.shape == (3, 4)

    def test_no_columns_dropped_when_all_populated(self, tmp_path):
        """Return unchanged frame when no columns are entirely empty"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,Alice,95\n2,Bob,87\n3,Charlie,92\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == frame.shape
        assert result.columns == frame.columns

    def test_all_columns_empty_preserves_rows(self, tmp_path):
        """When all columns are empty, return frame with zero columns but same row count"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\n,  ,\n, \t ,\n, ,  \n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert len(result.columns) == 0
        assert result.shape[0] == 3  # row count preserved

    def test_zero_row_frame_preserves_all_columns(self, tmp_path):
        """Frame with header only (no data rows) preserves all columns"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,empty_col\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        # Cannot determine emptiness without data, so keep all
        assert result.shape == (0, 3)

    def test_multiple_scattered_empty_columns(self, tmp_path):
        """Drop multiple empty columns interspersed with populated ones"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,empty1,name,empty2,score,empty3\n"
                           "1,,Alice,,95,\n"
                           "2,,Bob,,87,\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["id", "name", "score"]
        assert result.shape == (2, 3)

    def test_roundtrip_csv_write_read(self, tmp_path):
        """Full workflow: load CSV, drop empty columns, write CSV, read back"""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        input_file.write_text("customer,empty1,amount,empty2\nAlice,, 100,\nBob,,200,\nCharlie,,150,\n")
        
        frame = ar.read_csv(str(input_file))
        cleaned = ar.drop_empty_columns(frame)
        ar.write_csv(cleaned, str(output_file))
        
        # Verify output file
        frame_reread = ar.read_csv(str(output_file))
        assert frame_reread.columns == ["customer", "amount"]
        assert frame_reread.shape == (3, 2)

    def test_pandas_from_to_roundtrip(self, tmp_path):
        """Convert pandas DataFrame → ArFrame → clean → back to pandas"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "empty_col": [None, None, None],
            "score": [95, 87, 92]
        })
        
        frame = ar.from_pandas(df)
        cleaned = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(cleaned)
        
        assert list(result_df.columns) == ["id", "name", "score"]
        assert len(result_df) == 3
        assert result_df["id"].tolist() == [1, 2, 3]

    def test_mixed_null_and_whitespace_removed(self, tmp_path):
        """Column with mix of null and whitespace-only entries is dropped"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,mixed\n1,Alice,\n2,Bob,   \n3,Charlie,\t\n4,Diana,\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert "mixed" not in result.columns
        assert result.shape == (4, 2)

    def test_numeric_column_with_one_value_kept(self, tmp_path):
        """Numeric column with at least one non-empty value is kept"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score\n1,100\n2,\n3,95\n4,\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert "score" in result.columns
        assert result.shape == (4, 2)

    def test_column_order_preserved(self, tmp_path):
        """Remaining columns maintain original order"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("z,empty1,y,empty2,x\nz_val,,y_val,,x_val\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["z", "y", "x"]

    def test_single_row_dataset(self, tmp_path):
        """Handle single-row dataset correctly"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,empty1,name,empty2\n42,,Alice,\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["id", "name"]
        assert result.shape == (1, 2)
        
        df = ar.to_pandas(result)
        assert df.loc[0, "id"] == 42
        assert df.loc[0, "name"] == "Alice"

    def test_idempotent_double_application(self, tmp_path):
        """Applying drop_empty_columns twice gives same result as once"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,empty,name\n1,,Alice\n2,,Bob\n3,,Charlie\n")
        
        frame = ar.read_csv(str(csv_file))
        result_once = ar.drop_empty_columns(frame)
        result_twice = ar.drop_empty_columns(result_once)
        
        assert result_once.columns == result_twice.columns
        assert result_once.shape == result_twice.shape

    def test_large_dataset_performance(self, tmp_path):
        """Verify correctness with moderately large dataset"""
        csv_file = tmp_path / "large.csv"
        
        # Create CSV with 10K rows, mixed empty and populated columns
        lines = ["id,empty1,value,empty2,name,empty3"]
        for i in range(10000):
            lines.append(f"{i},,val_{i},,Name_{i},")
        csv_file.write_text("\n".join(lines))
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (10000, 3)
        assert result.columns == ["id", "value", "name"]
        
        # Spot check with pandas
        df = ar.to_pandas(result)
        assert len(df) == 10000
        assert df.loc[0, "id"] == 0
        assert df.loc[9999, "name"] == "Name_9999"

    def test_error_on_non_arframe_input(self):
        """drop_empty_columns raises TypeError on non-ArFrame input"""
        with pytest.raises(Exception):  # Should raise TypeError or similar
            ar.drop_empty_columns("not a frame")

    def test_special_characters_in_data_preserved(self, tmp_path):
        """Non-empty columns with special characters are preserved"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text('name,desc,empty\n"Alice ""The Great""","Has\ttabs",\n"Bob, Jr.","Line\nbreak",\n')
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["name", "desc"]
        assert result.shape == (2, 2)
