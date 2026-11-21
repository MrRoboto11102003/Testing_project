import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
import os
from pathlib import Path


class TestDropConstantColumns:
    """System tests for ar.drop_constant_columns()"""

    def test_single_constant_column_removed(self, tmp_path):
        """Single column with one unique value is removed."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,status\n1,alice,active\n2,bob,active\n3,charlie,active")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)
        
        assert "status" not in df.columns
        assert set(df.columns) == {"id", "name"}
        assert len(df) == 3

    def test_multiple_constant_columns(self, tmp_path):
        """Multiple constant columns are all removed."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,const1,const2,name\n1,x,y,alice\n2,x,y,bob\n3,x,y,charlie")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)
        
        assert set(df.columns) == {"id", "name"}
        assert len(df) == 3

    def test_no_constant_columns_unchanged(self, tmp_path):
        """Frame with no constant columns remains unchanged."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,score\n1,alice,95\n2,bob,87\n3,charlie,92")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)
        
        assert set(df.columns) == {"id", "name", "score"}
        assert len(df) == 3

    def test_all_constant_columns_zero_column_frame(self, tmp_path):
        """When all columns are constant, result is zero-column frame preserving row count."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("const1,const2\nx,y\nx,y\nx,y")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)
        
        assert len(df.columns) == 0
        assert len(df) == 3

    def test_all_null_column_is_constant(self, tmp_path):
        """Column with all nulls (one unique value) is dropped."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,empty\n1,alice,\n2,bob,\n3,charlie,")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)
        
        assert "empty" not in df.columns
        assert set(df.columns) == {"id", "name"}

    def test_mixed_nulls_not_constant(self, tmp_path):
        """Column with both values and nulls (two unique) is kept."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,mixed\n1,a\n2,\n3,a")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)
        
        assert "mixed" in df.columns
        assert len(df.columns) == 2

    def test_empty_frame_zero_rows_keeps_all_columns(self, tmp_path):
        """Empty frame (zero rows) preserves all columns."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,status")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)
        
        assert set(df.columns) == {"id", "name", "status"}
        assert len(df) == 0

    def test_single_row_frame_all_constant(self, tmp_path):
        """Single row: all columns have nunique=1, all are dropped."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,score\n1,alice,95")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)
        
        # All columns have exactly one unique value
        assert len(df.columns) == 0
        assert len(df) == 1

    def test_pandas_dataframe_input(self, tmp_path):
        """Accept pandas DataFrame and return pandas DataFrame."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "const": ["x", "x", "x"],
            "name": ["alice", "bob", "charlie"]
        })
        
        result = ar.drop_constant_columns(df)
        
        assert isinstance(result, pd.DataFrame)
        assert "const" not in result.columns
        assert set(result.columns) == {"id", "name"}

    def test_in_pipeline(self, tmp_path):
        """Use drop_constant_columns within ar.pipeline()."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,const,name\n1,x,alice\n2,x,bob\n3,x,charlie")
        
        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_constant_columns",),
            ("drop_duplicates",),
        ])
        df = ar.to_pandas(clean)
        
        assert "const" not in df.columns
        assert set(df.columns) == {"id", "name"}
        assert len(df) == 3

    def test_numeric_constant_column(self, tmp_path):
        """Constant numeric column is dropped."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score,flag\n1,95,1\n2,87,1\n3,92,1")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)
        
        assert "flag" not in df.columns
        assert set(df.columns) == {"id", "score"}

    def test_roundtrip_write_csv_read(self, tmp_path):
        """Write cleaned result to CSV and read it back."""
        csv_in = tmp_path / "input.csv"
        csv_out = tmp_path / "output.csv"
        csv_in.write_text("id,const,name\n1,x,alice\n2,x,bob")
        
        frame = ar.read_csv(str(csv_in))
        result = ar.drop_constant_columns(frame)
        ar.write_csv(result, str(csv_out))
        
        df_out = pd.read_csv(csv_out)
        assert "const" not in df_out.columns
        assert set(df_out.columns) == {"id", "name"}
        assert len(df_out) == 2


class TestDropEmptyColumns:
    """System tests for ar.drop_empty_columns()"""

    def test_all_null_column_removed(self, tmp_path):
        """Column with all null values is removed."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,empty\n1,alice,\n2,bob,\n3,charlie,")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_empty_columns(frame)
        df = ar.to_pandas(result)
        
        assert "empty" not in df.columns
        assert set(df.columns) == {"id", "name"}
        assert len(df) == 3

    def test_all_whitespace_column_removed(self, tmp_path):
        """Column with only whitespace strings is removed."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,whitespace\n1,alice,   \n2,bob,  \n3,charlie,\t")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_empty_columns(frame)
        df = ar.to_pandas(result)
        
        assert "whitespace" not in df.columns
        assert set(df.columns) == {"id", "name"}

    def test_mixed_null_and_whitespace_removed(self, tmp_path):
        """Column with mix of null and whitespace is removed."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,empty_col\n1,\n2,  \n3,")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_empty_columns(frame)
        df = ar.to_pandas(result)
        
        assert "empty_col" not in df.columns
        assert set(df.columns) == {"id"}

    def test_no_empty_columns_unchanged(self, tmp_path):
        """Frame with no empty columns remains unchanged."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name,score\n1,alice,95\n2,bob,87\n3,charlie,92")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_empty_columns(frame)
        df = ar.to_pandas(result)
        
        assert set(df.columns) == {"id", "name", "score"}
        assert len(df) == 3

    def test_multiple_empty_columns(self, tmp_path):
        """Multiple empty columns are all removed."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,empty1,name,empty2\n1,,alice,\n2,,bob,\n3,,charlie,")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_empty_columns(frame)
        df = ar.to_pandas(result)
        
        assert set(df.columns) == {"id", "name"}
        assert len(df) == 3

    def test_all_columns_empty_zero_column_result(self, tmp_path):
        """When all columns are empty, result has no columns."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("empty1,empty2\n,\n,\n,")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_empty_columns(frame)
        df = ar.to_pandas(result)
        
        assert len(df.columns) == 0
        assert len(df) == 3

    def test_zero_row_frame_returns_zero_column(self, tmp_path):
        """Zero-row frame with empty columns returns zero-column frame."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,empty,name")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_empty_columns(frame)
        df = ar.to_pandas(result)
        
        assert len(df) == 0
        assert len(df.columns) == 0

    def test_column_with_some_values_kept(self, tmp_path):
        """Column with at least one non-empty value is kept."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,mostly_empty\n1,\n2,x\n3,")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_empty_columns(frame)
        df = ar.to_pandas(result)
        
        assert "mostly_empty" in df.columns
        assert set(df.columns) == {"id", "mostly_empty"}

    def test_column_with_zero_string_kept(self, tmp_path):
        """Column with '0' string values is kept (not considered empty)."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,values\n1,0\n2,0\n3,0")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.drop_empty_columns(frame)
        df = ar.to_pandas(result)
        
        assert "values" in df.columns
        assert len(df) == 3

    def test_in_pipeline(self, tmp_path):
        """Use drop_empty_columns within ar.pipeline()."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,empty,name\n1,,alice\n2,,bob\n3,,charlie")
        
        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_empty_columns",),
        ])
        df = ar.to_pandas(clean)
        
        assert "empty" not in df.columns
        assert set(df.columns) == {"id", "name"}
        assert len(df) == 3

    def test_roundtrip_write_csv_read(self, tmp_path):
        """Write cleaned result to CSV and read it back."""
        csv_in = tmp_path / "input.csv"
        csv_out = tmp_path / "output.csv"
        csv_in.write_text("id,empty,name\n1,,alice\n2,,bob")
        
        frame = ar.read_csv(str(csv_in))
        result = ar.drop_empty_columns(frame)
        ar.write_csv(result, str(csv_out))
        
        df_out = pd.read_csv(csv_out)
        assert "empty" not in df_out.columns
        assert set(df_out.columns) == {"id", "name"}
        assert len(df_out) == 2

    def test_combined_with_drop_constant_columns(self, tmp_path):
        """Combine drop_empty_columns and drop_constant_columns in pipeline."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,empty,const,name\n1,,x,alice\n2,,x,bob")
        
        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("drop_empty_columns",),
            ("drop_constant_columns",),
        ])
        df = ar.to_pandas(clean)
        
        assert "empty" not in df.columns
        assert "const" not in df.columns
        assert set(df.columns) == {"id", "name"}
        assert len(df) == 2
