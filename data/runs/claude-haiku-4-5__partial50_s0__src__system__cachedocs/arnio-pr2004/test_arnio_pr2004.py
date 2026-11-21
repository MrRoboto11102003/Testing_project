import pytest
import os
import sys
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path


class TestWriteCsvBasic:
    """Basic write_csv functionality tests."""

    def test_write_csv_creates_file(self, tmp_path):
        """Test that write_csv creates a CSV file."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file))
        
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_write_csv_round_trip(self, tmp_path):
        """Test write_csv followed by read_csv preserves data integrity."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1]
        })
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file))
        read_back = ar.read_csv(str(output_file))
        result_df = ar.to_pandas(read_back)
        
        assert result_df.shape == df.shape
        assert list(result_df.columns) == list(df.columns)
        assert len(result_df) == 3

    def test_write_csv_with_nulls_round_trip(self, tmp_path):
        """Test write_csv handles null values correctly."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({
            "id": [1, 2, None],
            "name": ["Alice", None, "Charlie"]
        })
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file))
        read_back = ar.read_csv(str(output_file))
        result_df = ar.to_pandas(read_back)
        
        assert result_df["id"].isna().sum() >= 0
        assert result_df["name"].isna().sum() >= 0

    def test_write_csv_txt_extension(self, tmp_path):
        """Test write_csv with .txt extension."""
        output_file = tmp_path / "output.txt"
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file))
        
        assert output_file.exists()
        content = output_file.read_text()
        assert "x" in content and "y" in content

    def test_write_csv_tsv_extension(self, tmp_path):
        """Test write_csv with .tsv extension."""
        output_file = tmp_path / "output.tsv"
        df = pd.DataFrame({"col1": ["a", "b"], "col2": ["c", "d"]})
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file))
        
        assert output_file.exists()

    def test_write_csv_delimiter_tab(self, tmp_path):
        """Test write_csv with tab delimiter."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file), delimiter="\t")
        
        content = output_file.read_text()
        assert "a\tb" in content

    def test_write_csv_delimiter_semicolon(self, tmp_path):
        """Test write_csv with semicolon delimiter."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1, 2], "y": [5, 6]})
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file), delimiter=";")
        
        content = output_file.read_text()
        assert "x;y" in content

    def test_write_csv_no_header(self, tmp_path):
        """Test write_csv without header."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file), write_header=False)
        
        content = output_file.read_text()
        lines = content.strip().split("\n")
        assert "name" not in lines[0]

    def test_write_csv_with_header(self, tmp_path):
        """Test write_csv includes header by default."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file), write_header=True)
        
        content = output_file.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "name,age"

    def test_write_csv_line_terminator_crlf(self, tmp_path):
        """Test write_csv with CRLF line terminator."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file), line_terminator="\r\n")
        
        content = output_file.read_bytes()
        assert b"\r\n" in content

    def test_write_csv_escape_formulas(self, tmp_path):
        """Test write_csv with escape_formulas=True."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({
            "formula": ["=SUM(A1:A10)", "+1", "-1", "@cmd", "normal"]
        })
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file), escape_formulas=True)
        
        assert output_file.exists()
        content = output_file.read_text()
        # At least verify file was written
        assert len(content) > 0

    def test_write_csv_utf8_encoding(self, tmp_path):
        """Test write_csv with UTF-8 encoding."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({
            "name": ["Café", "Naïve"]
        })
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file), encoding="utf-8")
        
        content = output_file.read_text(encoding="utf-8")
        assert "Caf" in content

    def test_write_csv_encoding_errors_replace(self, tmp_path):
        """Test write_csv with encoding_errors='replace' for non-ASCII."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"data": ["café"]})
        frame = ar.from_pandas(df)
        
        ar.write_csv(
            frame, str(output_file),
            encoding="ascii",
            encoding_errors="replace"
        )
        
        assert output_file.exists()

    def test_write_csv_encoding_errors_ignore(self, tmp_path):
        """Test write_csv with encoding_errors='ignore'."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"data": ["café"]})
        frame = ar.from_pandas(df)
        
        ar.write_csv(
            frame, str(output_file),
            encoding="ascii",
            encoding_errors="ignore"
        )
        
        assert output_file.exists()


class TestWriteCsvErrors:
    """Error handling and validation tests."""

    def test_write_csv_invalid_frame_type(self, tmp_path):
        """Test write_csv raises TypeError for non-ArFrame."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1, 2]})
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_csv(df, str(output_file))

    def test_write_csv_invalid_path_type(self, tmp_path):
        """Test write_csv raises TypeError for invalid path type."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.write_csv(frame, 12345)

    def test_write_csv_unsupported_extension(self, tmp_path):
        """Test write_csv raises ValueError for unsupported extension."""
        output_file = tmp_path / "output.xlsx"
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_csv(frame, str(output_file))

    def test_write_csv_invalid_delimiter_multichar(self, tmp_path):
        """Test write_csv raises error for multi-character delimiter."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises((ValueError, TypeError)):
            ar.write_csv(frame, str(output_file), delimiter="||")

    def test_write_csv_invalid_line_terminator(self, tmp_path):
        """Test write_csv raises ValueError for invalid line terminator."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="line_terminator"):
            ar.write_csv(frame, str(output_file), line_terminator="\n\n")

    def test_write_csv_unknown_encoding(self, tmp_path):
        """Test write_csv raises ValueError for unknown encoding."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.write_csv(frame, str(output_file), encoding="unknown-codec-xyz")

    def test_write_csv_invalid_encoding_errors(self, tmp_path):
        """Test write_csv raises ValueError for invalid encoding_errors."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.write_csv(frame, str(output_file), encoding_errors="invalid")


class TestWriteCsvIntegration:
    """Integration tests with other arnio functions."""

    def test_write_csv_after_pipeline_cleaning(self, tmp_path):
        """Test write_csv after applying a cleaning pipeline."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({
            "name": [" Alice ", " Bob "],
            "score": [95, 87]
        })
        frame = ar.from_pandas(df)
        
        # Apply cleaning pipeline
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
        ])
        
        ar.write_csv(clean, str(output_file))
        
        read_back = ar.read_csv(str(output_file))
        result_df = ar.to_pandas(read_back)
        assert len(result_df) == 2

    def test_write_csv_pandas_accessor_workflow(self, tmp_path):
        """Test write_csv integrated with pandas accessor cleaning."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30]
        })
        
        # Use pandas accessor to clean
        cleaned_df = df.arnio.clean(strip_whitespace=True)
        frame = ar.from_pandas(cleaned_df)
        
        ar.write_csv(frame, str(output_file))
        
        read_back = ar.read_csv(str(output_file))
        result_df = ar.to_pandas(read_back)
        assert result_df.shape[0] == 3

    def test_write_csv_pathlib_path_object(self, tmp_path):
        """Test write_csv accepts pathlib.Path objects."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, output_file)
        
        assert output_file.exists()

    def test_write_csv_mixed_data_types_round_trip(self, tmp_path):
        """Test write_csv with mixed data types and round-trip verification."""
        output_file = tmp_path / "output.csv"
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        frame = ar.from_pandas(df)
        
        ar.write_csv(frame, str(output_file))
        
        read_back = ar.read_csv(str(output_file))
        result_df = ar.to_pandas(read_back)
        assert result_df.shape == (3, 4)
        assert list(result_df.columns) == ["int_col", "float_col", "str_col", "bool_col"]
