"""
System (end-to-end) tests for arnio.cleaning.drop_empty_columns.

Tests real CSV workflows through Arnio's public API, including file I/O,
frame transformations, pipeline integration, and pandas conversion.
"""

import pytest
import pandas as pd
import arnio as ar


class TestDropEmptyColumnsBasic:
    """Core functionality: identifying and removing empty columns."""

    def test_drop_completely_null_column(self, tmp_path):
        """Remove column where all values are null/empty."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,empty,value\nAlice,,100\nBob,,200\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["name", "value"]
        assert result.shape == (2, 2)

    def test_drop_whitespace_only_column(self, tmp_path):
        """Remove column where all values are whitespace only."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,spaces,value\nAlice,   ,100\nBob,  ,200\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert "spaces" not in result.columns
        assert result.columns == ["name", "value"]
        assert result.shape == (2, 2)

    def test_keep_columns_with_mixed_values(self, tmp_path):
        """Keep columns that have at least one non-empty value."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,phone,value\nAlice,555-1234,100\nBob,,200\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert "phone" in result.columns
        assert result.shape[1] == 3
        assert result.shape[0] == 2

    def test_no_empty_columns_returns_all(self, tmp_path):
        """Return all columns unchanged when none are empty."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["a", "b", "c"]
        assert result.shape == (2, 3)

    def test_all_columns_empty(self, tmp_path):
        """Handle case where all columns are empty."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2,col3\n,,\n,,\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape[1] == 0
        assert result.shape[0] == 2


class TestDropEmptyColumnsPreservation:
    """Test preservation of data and structure."""

    def test_preserve_column_order(self, tmp_path):
        """Preserve the order of remaining columns."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("z,empty1,y,empty2,x\n1,,2,,3\n4,,5,,6\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["z", "y", "x"]
        assert result.shape == (2, 3)

    def test_original_frame_unmodified(self, tmp_path):
        """Ensure original frame is not mutated."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,empty,value\nAlice,,100\n")
        
        frame = ar.read_csv(str(csv_file))
        original_cols = list(frame.columns)
        original_shape = frame.shape
        
        result = ar.drop_empty_columns(frame)
        
        assert frame.columns == original_cols
        assert frame.shape == original_shape
        assert len(result.columns) < len(frame.columns)

    def test_data_integrity_in_kept_columns(self, tmp_path):
        """Verify data values are preserved in non-empty columns."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,empty,value\nAlice,,100\nBob,,200\nCharlie,,300\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        df = ar.to_pandas(result)
        
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]
        assert df["value"].tolist() == [100, 200, 300]


class TestDropEmptyColumnsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_frame_zero_rows(self, tmp_path):
        """Handle frame with header only (0 data rows)."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,empty,value\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape[0] == 0

    def test_single_empty_column(self, tmp_path):
        """Handle single column that is entirely empty."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("empty\n\n\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape[1] == 0
        assert result.shape[0] == 2

    def test_single_nonempty_column(self, tmp_path):
        """Handle single column that is not empty."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("data\nA\nB\nC\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["data"]
        assert result.shape == (3, 1)

    def test_mixed_empty_representations(self, tmp_path):
        """Handle various empty representations: null, whitespace, tabs."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,empty_col,value\nAlice,,100\nBob,   ,200\nCharlie,\t,300\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert "empty_col" not in result.columns
        assert result.shape == (3, 2)


class TestDropEmptyColumnsIntegration:
    """Test integration with other Arnio features and workflows."""

    def test_to_pandas_conversion(self, tmp_path):
        """Full workflow: drop empty columns and convert to pandas."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,empty,value\nAlice,,100\nBob,,200\n")
        
        frame = ar.read_csv(str(csv_file))
        cleaned = ar.drop_empty_columns(frame)
        df = ar.to_pandas(cleaned)
        
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["name", "value"]
        assert len(df) == 2

    def test_write_csv_roundtrip(self, tmp_path):
        """Full workflow: load CSV, drop empty, write CSV, reload."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("name,empty,value\nAlice,,100\nBob,,200\n")
        
        output_csv = tmp_path / "output.csv"
        
        frame = ar.read_csv(str(input_csv))
        cleaned = ar.drop_empty_columns(frame)
        ar.write_csv(cleaned, str(output_csv))
        
        result = ar.read_csv(str(output_csv))
        assert result.columns == ["name", "value"]
        assert result.shape == (2, 2)

    def test_pipeline_integration_single_step(self, tmp_path):
        """Use drop_empty_columns as a standalone pipeline step."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,empty,value\nAlice,,100\nBob,,200\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("drop_empty_columns",),
        ])
        
        assert result.columns == ["name", "value"]

    def test_pipeline_integration_with_other_steps(self, tmp_path):
        """Integrate drop_empty_columns into a multi-step cleaning pipeline."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,empty,city\n ALICE ,,NEW YORK\n BOB ,,LOS ANGELES\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("drop_empty_columns",),
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
        ])
        
        df = ar.to_pandas(result)
        assert "empty" not in df.columns
        assert df["name"].iloc[0] == "alice"
        assert df["city"].iloc[1] == "los angeles"

    def test_from_pandas_roundtrip(self, tmp_path):
        """Create frame from pandas, drop empty, convert back."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "empty": [None, None],
            "value": [100, 200],
        })
        
        frame = ar.from_pandas(df)
        cleaned = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(cleaned)
        
        assert list(result_df.columns) == ["name", "value"]
        assert len(result_df) == 2

    def test_numeric_null_column(self, tmp_path):
        """Remove numeric column with all null values."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,empty_num,amount\n1,,50\n2,,75\n3,,100\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert "empty_num" not in result.columns
        assert result.shape == (3, 2)

    def test_multiple_consecutive_empty_columns(self, tmp_path):
        """Remove multiple consecutive empty columns."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,empty1,empty2,empty3,b\n1,,,, 10\n2,,,, 20\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["a", "b"]
        assert result.shape == (2, 2)

    def test_scan_then_drop_empty(self, tmp_path):
        """Use scan_csv to preview, then drop empty columns."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,empty,value\nAlice,,100\nBob,,200\n")
        
        # First scan to see the schema
        schema = ar.scan_csv(str(csv_file))
        assert "empty" in schema
        
        # Then load and clean
        frame = ar.read_csv(str(csv_file))
        cleaned = ar.drop_empty_columns(frame)
        
        schema_after = cleaned.dtypes
        assert "empty" not in schema_after
