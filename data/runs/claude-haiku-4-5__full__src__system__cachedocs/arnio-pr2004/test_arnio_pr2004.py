import pytest
import tempfile
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import arnio as ar


class TestWriteCsvBasic:
    """Basic write_csv functionality."""

    def test_write_csv_default_format(self, tmp_path):
        """Write a simple frame with default comma-delimited UTF-8."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1]
        })
        frame = ar.from_pandas(df)
        output_path = tmp_path / "output.csv"
        
        ar.write_csv(frame, str(output_path))
        
        assert output_path.exists()
        result_df = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(result_df, df)

    def test_write_csv_with_pathlib(self, tmp_path):
        """Write using pathlib.Path object instead of string."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        frame = ar.from_pandas(df)
        output_path = tmp_path / "data.csv"
        
        ar.write_csv(frame, output_path)  # Pass Path directly
        
        assert output_path.exists()
        result = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(result, df)

    def test_write_csv_roundtrip(self, tmp_path):
        """Write CSV and read back to verify complete data preservation."""
        original = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.5, 20.3, 30.1],
            "label": ["A", "B", "C"]
        })
        
        frame = ar.from_pandas(original)
        path = tmp_path / "roundtrip.csv"
        ar.write_csv(frame, str(path))
        
        read_frame = ar.read_csv(str(path))
        result = ar.to_pandas(read_frame)
        pd.testing.assert_frame_equal(result, original)


class TestWriteCsvFormatOptions:
    """Test delimiter, header, and line terminator options."""

    def test_write_csv_tab_delimiter(self, tmp_path):
        """Write with tab delimiter (TSV format)."""
        df = pd.DataFrame({"id": [1, 2], "value": [100, 200]})
        frame = ar.from_pandas(df)
        path = tmp_path / "data.tsv"
        
        ar.write_csv(frame, str(path), delimiter="\t")
        
        result = pd.read_csv(path, delimiter="\t")
        pd.testing.assert_frame_equal(result, df)

    def test_write_csv_semicolon_delimiter(self, tmp_path):
        """Write with semicolon delimiter (European CSV)."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        frame = ar.from_pandas(df)
        path = tmp_path / "data.csv"
        
        ar.write_csv(frame, str(path), delimiter=";")
        
        result = pd.read_csv(path, delimiter=";")
        pd.testing.assert_frame_equal(result, df)

    def test_write_csv_without_header(self, tmp_path):
        """Write CSV without column header row."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        frame = ar.from_pandas(df)
        path = tmp_path / "no_header.csv"
        
        ar.write_csv(frame, str(path), write_header=False)
        
        content = path.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "1,3"  # First data row, not header
        assert len(lines) == 2

    def test_write_csv_windows_line_terminator(self, tmp_path):
        """Write with Windows CRLF line terminator."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        path = tmp_path / "windows.csv"
        
        ar.write_csv(frame, str(path), line_terminator="\r\n")
        
        content = path.read_bytes()
        assert b"\r\n" in content
        assert content.count(b"\r\n") >= 2  # Header + at least 1 data row


class TestWriteCsvFormulaEscaping:
    """Test formula injection prevention."""

    def test_escape_formulas_equals(self, tmp_path):
        """Escape formula-injection strings starting with =."""
        df = pd.DataFrame({
            "user_input": ["=SUM(A1:A100)", "normal_text", "data"]
        })
        frame = ar.from_pandas(df)
        path = tmp_path / "escaped.csv"
        
        ar.write_csv(frame, str(path), escape_formulas=True)
        
        content = path.read_text()
        assert "'=SUM(A1:A100)" in content
        assert "normal_text" in content

    def test_escape_formulas_all_triggers(self, tmp_path):
        """Escape all formula trigger characters: =, +, -, @."""
        df = pd.DataFrame({
            "values": ["=FORMULA", "+ADD", "-SUB", "@REF"]
        })
        frame = ar.from_pandas(df)
        path = tmp_path / "escaped.csv"
        
        ar.write_csv(frame, str(path), escape_formulas=True)
        
        content = path.read_text()
        lines = content.strip().split("\n")
        data_line = lines[1]
        # All should be escaped with leading single quote
        assert data_line.count("'") >= 4

    def test_no_escape_by_default(self, tmp_path):
        """Verify that formulas are NOT escaped by default."""
        df = pd.DataFrame({"formula": ["=SUM(1,2)"]})
        frame = ar.from_pandas(df)
        path = tmp_path / "not_escaped.csv"
        
        ar.write_csv(frame, str(path), escape_formulas=False)
        
        content = path.read_text()
        assert "=SUM(1,2)" in content
        assert "'=SUM" not in content


class TestWriteCsvEncoding:
    """Test encoding and character handling."""

    def test_write_csv_utf8_unicode(self, tmp_path):
        """Write frame with Unicode characters in UTF-8 (default)."""
        df = pd.DataFrame({
            "city": ["Paris", "München", "São Paulo", "北京"],
            "emoji": ["🌍", "🏖️", "🎭", "🏔️"]
        })
        frame = ar.from_pandas(df)
        path = tmp_path / "unicode.csv"
        
        ar.write_csv(frame, str(path))  # Default UTF-8
        
        result = pd.read_csv(path, encoding="utf-8")
        pd.testing.assert_frame_equal(result, df)

    def test_write_csv_latin1_encoding(self, tmp_path):
        """Write with latin-1 encoding for Western European text."""
        df = pd.DataFrame({
            "text": ["café", "naïve", "résumé"]
        })
        frame = ar.from_pandas(df)
        path = tmp_path / "latin1.csv"
        
        ar.write_csv(frame, str(path), encoding="latin-1")
        
        result = pd.read_csv(path, encoding="latin-1")
        pd.testing.assert_frame_equal(result, df)

    def test_write_csv_encoding_errors_replace(self, tmp_path):
        """Handle unencodable characters by replacing them."""
        df = pd.DataFrame({
            "mixed": ["Hello", "世界", "café"]
        })
        frame = ar.from_pandas(df)
        path = tmp_path / "replaced.csv"
        
        # ASCII can't encode the Chinese characters or accented é
        ar.write_csv(
            frame,
            str(path),
            encoding="ascii",
            encoding_errors="replace"
        )
        
        assert path.exists()
        # File should be readable with replacement characters
        content = path.read_text(encoding="ascii", errors="replace")
        assert "Hello" in content


class TestWriteCsvFileExtensions:
    """Test support for different file extensions."""

    def test_write_csv_supported_extensions(self, tmp_path):
        """Write to .csv, .txt, and .tsv extensions."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        for ext in ["csv", "txt", "tsv"]:
            path = tmp_path / f"file.{ext}"
            ar.write_csv(frame, str(path))
            assert path.exists()

    def test_write_csv_unsupported_extension(self, tmp_path):
        """Reject unsupported file extensions."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        path = tmp_path / "file.json"
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_csv(frame, str(path))


class TestWriteCsvErrorHandling:
    """Test error cases and validation."""

    def test_write_csv_invalid_frame(self, tmp_path):
        """Raise TypeError if frame is not an ArFrame."""
        df = pd.DataFrame({"x": [1, 2]})
        path = tmp_path / "out.csv"
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_csv(df, str(path))

    def test_write_csv_invalid_path_type(self, tmp_path):
        """Raise TypeError for invalid path type."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.write_csv(frame, 12345)

    def test_write_csv_invalid_encoding(self, tmp_path):
        """Raise ValueError for unknown encoding."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        path = tmp_path / "out.csv"
        
        with pytest.raises(ValueError, match="unknown encoding"):
            ar.write_csv(frame, str(path), encoding="not-a-codec")

    def test_write_csv_invalid_encoding_errors(self, tmp_path):
        """Raise ValueError for invalid encoding_errors option."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        path = tmp_path / "out.csv"
        
        with pytest.raises(ValueError):
            ar.write_csv(frame, str(path), encoding_errors="invalid")


class TestWriteCsvWithPipeline:
    """Integration with cleaning pipelines."""

    def test_write_csv_after_pipeline(self, tmp_path):
        """Write output after applying a cleaning pipeline."""
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "  Charlie  "],
            "age": [30, 25, 35]
        })
        frame = ar.from_pandas(df)
        
        # Apply strip_whitespace pipeline
        clean = ar.pipeline(frame, [("strip_whitespace",)])
        
        path = tmp_path / "clean.csv"
        ar.write_csv(clean, str(path))
        
        result = pd.read_csv(path)
        assert result["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_write_csv_full_workflow(self, tmp_path):
        """Full workflow: create messy CSV, load, clean, write."""
        # Create messy input
        input_path = tmp_path / "messy.csv"
        input_path.write_text(
            "id,name,score\n"
            "1, Alice ,95\n"
            "2, Bob ,87\n"
            "1, Alice ,95\n"  # Duplicate
        )
        
        # Load, clean, write
        frame = ar.read_csv(str(input_path))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",)
        ])
        
        output_path = tmp_path / "clean.csv"
        ar.write_csv(clean, str(output_path))
        
        # Verify
        result = pd.read_csv(output_path)
        assert len(result) == 2
        assert result["name"].tolist() == ["Alice", "Bob"]


class TestWriteCsvMixedData:
    """Test writing frames with various data types."""

    def test_write_csv_mixed_types_roundtrip(self, tmp_path):
        """Write and read back frame with int, float, string, bool columns."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["x", "y", "z"],
            "bool_col": [True, False, True]
        })
        frame = ar.from_pandas(df)
        path = tmp_path / "mixed.csv"
        
        ar.write_csv(frame, str(path))
        
        result = pd.read_csv(path)
        assert list(result.columns) == ["int_col", "float_col", "str_col", "bool_col"]
        assert len(result) == 3

    def test_write_csv_with_nulls(self, tmp_path):
        """Write frame containing null/missing values."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", None, "Charlie"],
            "value": [100.0, 200.0, None]
        })
        frame = ar.from_pandas(df)
        path = tmp_path / "nulls.csv"
        
        ar.write_csv(frame, str(path))
        
        result = pd.read_csv(path)
        assert pd.isna(result.loc[1, "name"])
        assert pd.isna(result.loc[2, "value"])

    def test_write_csv_empty_frame(self, tmp_path):
        """Write empty frame (header only, zero rows)."""
        df = pd.DataFrame({"col1": [], "col2": [], "col3": []})
        frame = ar.from_pandas(df)
        path = tmp_path / "empty.csv"
        
        ar.write_csv(frame, str(path))
        
        result = pd.read_csv(path)
        assert list(result.columns) == ["col1", "col2", "col3"]
        assert len(result) == 0
