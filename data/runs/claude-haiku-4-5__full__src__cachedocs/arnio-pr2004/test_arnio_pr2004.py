import pytest
import tempfile
import os
import pandas as pd
import arnio as ar
from pathlib import Path


class TestWriteCsvBasic:
    """Basic write_csv functionality."""
    
    def test_write_csv_creates_file(self):
        """Test that write_csv creates a CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            frame = ar.from_pandas(pd.DataFrame({
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"]
            }))
            ar.write_csv(frame, output_path)
            
            assert os.path.exists(output_path)
            df = pd.read_csv(output_path)
            assert list(df.columns) == ["id", "name"]
            assert len(df) == 3
    
    def test_write_tsv_format(self):
        """Test writing TSV format (.tsv extension)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.tsv")
            frame = ar.from_pandas(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
            ar.write_csv(frame, output_path)
            
            assert os.path.exists(output_path)
            df = pd.read_csv(output_path, delimiter="\t")
            assert list(df.columns) == ["a", "b"]
    
    def test_write_txt_format(self):
        """Test writing TXT format (.txt extension)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.txt")
            frame = ar.from_pandas(pd.DataFrame({"col": [1, 2]}))
            ar.write_csv(frame, output_path)
            
            assert os.path.exists(output_path)


class TestWriteCsvOptions:
    """Test write_csv configuration options."""
    
    def test_write_without_header(self):
        """Test write_header=False omits column headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "no_header.csv")
            frame = ar.from_pandas(pd.DataFrame({
                "id": [1, 2],
                "name": ["Alice", "Bob"]
            }))
            ar.write_csv(frame, output_path, write_header=False)
            
            with open(output_path, "r") as f:
                first_line = f.readline()
            assert "id" not in first_line
            assert "name" not in first_line
    
    def test_custom_delimiter(self):
        """Test custom delimiter parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "semicolon.csv")
            frame = ar.from_pandas(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
            ar.write_csv(frame, output_path, delimiter=";")
            
            with open(output_path, "r") as f:
                content = f.read()
            assert "a;b" in content
    
    def test_line_terminator_crlf(self):
        """Test CRLF line terminator (Windows-style)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "crlf.csv")
            frame = ar.from_pandas(pd.DataFrame({"col": [1, 2]}))
            ar.write_csv(frame, output_path, line_terminator="\r\n")
            
            with open(output_path, "rb") as f:
                content = f.read()
            assert b"\r\n" in content
    
    def test_line_terminator_cr(self):
        """Test CR line terminator (old Mac-style)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "cr.csv")
            frame = ar.from_pandas(pd.DataFrame({"col": [1, 2]}))
            ar.write_csv(frame, output_path, line_terminator="\r")
            
            with open(output_path, "rb") as f:
                content = f.read()
            assert b"\r" in content


class TestWriteCsvFormulaEscaping:
    """Test formula escaping security feature."""
    
    def test_escape_formulas_equals(self):
        """Test that escape_formulas=True prefixes = with quote."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "formula.csv")
            frame = ar.from_pandas(pd.DataFrame({
                "formula": ["=SUM(A1:A10)", "normal"]
            }))
            ar.write_csv(frame, output_path, escape_formulas=True)
            
            with open(output_path, "r") as f:
                content = f.read()
            # Should contain escaped formula with leading quote
            assert "'" in content or "SUM" in content
    
    def test_no_escape_formulas_default(self):
        """Test that formulas are not escaped by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "formula.csv")
            frame = ar.from_pandas(pd.DataFrame({
                "formula": ["=SUM(A1:A10)"]
            }))
            ar.write_csv(frame, output_path, escape_formulas=False)
            
            # Should write without error
            assert os.path.exists(output_path)


class TestWriteCsvEncoding:
    """Test encoding support and error handling."""
    
    def test_utf8_encoding_default(self):
        """Test UTF-8 encoding is used by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "utf8.csv")
            frame = ar.from_pandas(pd.DataFrame({
                "text": ["Hello", "Café"]
            }))
            ar.write_csv(frame, output_path)
            
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Caf" in content
    
    def test_latin1_encoding(self):
        """Test alternative encoding (Latin-1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "latin1.csv")
            frame = ar.from_pandas(pd.DataFrame({
                "text": ["Hello", "Café"]
            }))
            ar.write_csv(frame, output_path, encoding="latin-1")
            
            with open(output_path, "r", encoding="latin-1") as f:
                content = f.read()
            assert "Caf" in content
    
    def test_encoding_errors_replace(self):
        """Test encoding_errors='replace' substitutes unencodable chars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "replace.csv")
            frame = ar.from_pandas(pd.DataFrame({
                "text": ["Hello", "日本語"]
            }))
            # ASCII doesn't support Japanese; should replace
            ar.write_csv(frame, output_path, encoding="ascii", 
                        encoding_errors="replace")
            
            assert os.path.exists(output_path)
    
    def test_encoding_errors_ignore(self):
        """Test encoding_errors='ignore' drops unencodable chars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "ignore.csv")
            frame = ar.from_pandas(pd.DataFrame({
                "text": ["Hello", "日本語"]
            }))
            ar.write_csv(frame, output_path, encoding="ascii", 
                        encoding_errors="ignore")
            
            assert os.path.exists(output_path)


class TestWriteCsvValidation:
    """Test input validation and error handling."""
    
    def test_invalid_frame_type_raises_typeerror(self):
        """Test that non-ArFrame raises TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            df = pd.DataFrame({"col": [1, 2]})
            
            with pytest.raises(TypeError, match="frame must be an ArFrame"):
                ar.write_csv(df, output_path)
    
    def test_invalid_path_type_raises_typeerror(self):
        """Test that non-string/PathLike path raises TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"col": [1, 2]}))
        
        with pytest.raises(TypeError):
            ar.write_csv(frame, 123)
    
    def test_unsupported_file_extension_raises_valueerror(self):
        """Test that unsupported file extensions raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.json")
            frame = ar.from_pandas(pd.DataFrame({"col": [1, 2]}))
            
            with pytest.raises(ValueError, match="Unsupported file format"):
                ar.write_csv(frame, output_path)
    
    def test_invalid_line_terminator_raises_valueerror(self):
        """Test that invalid line_terminator raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            frame = ar.from_pandas(pd.DataFrame({"col": [1, 2]}))
            
            with pytest.raises(ValueError, match="line_terminator"):
                ar.write_csv(frame, output_path, line_terminator="\n\r")
    
    def test_invalid_encoding_raises_valueerror(self):
        """Test that invalid encoding name raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            frame = ar.from_pandas(pd.DataFrame({"col": [1, 2]}))
            
            with pytest.raises(ValueError):
                ar.write_csv(frame, output_path, encoding="fake-encoding-xyz")
    
    def test_invalid_encoding_errors_raises_valueerror(self):
        """Test that invalid encoding_errors raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            frame = ar.from_pandas(pd.DataFrame({"col": [1, 2]}))
            
            with pytest.raises(ValueError):
                ar.write_csv(frame, output_path, encoding_errors="invalid-mode")


class TestWriteCsvEdgeCases:
    """Test edge cases."""
    
    def test_write_empty_frame(self):
        """Test writing an empty frame with only column headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "empty.csv")
            frame = ar.from_pandas(pd.DataFrame({
                "a": pd.Series([], dtype="int64"),
                "b": pd.Series([], dtype="object")
            }))
            ar.write_csv(frame, output_path)
            
            df = pd.read_csv(output_path)
            assert list(df.columns) == ["a", "b"]
            assert len(df) == 0
    
    def test_pathlib_path_support(self):
        """Test that pathlib.Path objects are supported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.csv"
            frame = ar.from_pandas(pd.DataFrame({"col": [1, 2]}))
            ar.write_csv(frame, output_path)
            
            assert output_path.exists()
    
    def test_write_mixed_types(self):
        """Test writing frame with mixed data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "mixed.csv")
            frame = ar.from_pandas(pd.DataFrame({
                "int": [1, 2],
                "float": [1.5, 2.5],
                "str": ["a", "b"],
                "bool": [True, False]
            }))
            ar.write_csv(frame, output_path)
            
            df = pd.read_csv(output_path)
            assert len(df) == 2
            assert len(df.columns) == 4
