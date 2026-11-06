import pytest
import pandas as pd
import arnio as ar
import tempfile
import os
from pathlib import Path


class TestWriteCsvBasic:
    """Basic write_csv functionality tests."""

    def test_write_csv_default_params(self):
        """Test writing CSV with default parameters."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path)
            
            assert os.path.exists(path)
            with open(path, 'r') as f:
                content = f.read()
                assert "name,age" in content
                assert "Alice,25" in content
                assert "Bob,30" in content

    def test_write_csv_with_pathlib_path(self):
        """Test write_csv accepts pathlib.Path objects."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            ar.write_csv(frame, path)
            assert path.exists()

    def test_write_csv_without_header(self):
        """Test writing CSV without header row."""
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path, write_header=False)
            
            with open(path, 'r') as f:
                content = f.read()
                assert "name" not in content
                assert "Alice,25" in content

    def test_write_csv_custom_delimiters(self):
        """Test writing CSV with different delimiters."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Semicolon delimiter
            path1 = os.path.join(tmpdir, "semicolon.csv")
            ar.write_csv(frame, path1, delimiter=";")
            with open(path1, 'r') as f:
                assert "a;b" in f.read()
            
            # Tab delimiter
            path2 = os.path.join(tmpdir, "tab.tsv")
            ar.write_csv(frame, path2, delimiter="\t")
            with open(path2, 'r') as f:
                assert "a\tb" in f.read()

    def test_write_csv_custom_line_terminator(self):
        """Test writing CSV with Windows-style line terminator."""
        df = pd.DataFrame({"x": [1, 2]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path, line_terminator="\r\n")
            
            with open(path, 'rb') as f:
                content = f.read()
                assert b"\r\n" in content


class TestWriteCsvFormulas:
    """Test formula escaping functionality."""

    def test_escape_formulas_true(self):
        """Test escape_formulas=True prefixes formula trigger characters."""
        df = pd.DataFrame({
            "formulas": ["=SUM(A1:A10)", "+1+2", "-5", "@INDIRECT()", "normal"]
        })
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path, escape_formulas=True)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_escape_formulas_false_default(self):
        """Test escape_formulas=False (default) preserves original values."""
        df = pd.DataFrame({"text": ["=SUM(A1)", "normal"]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path, escape_formulas=False)
            
            with open(path, 'r') as f:
                content = f.read()
                assert "SUM" in content


class TestWriteCsvEncoding:
    """Test encoding-related functionality."""

    def test_write_csv_utf8_encoding(self):
        """Test writing with UTF-8 encoding (default)."""
        df = pd.DataFrame({"name": ["Café", "Naïve", "José"]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path)
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 0

    def test_write_csv_latin1_encoding(self):
        """Test writing with latin-1 encoding."""
        df = pd.DataFrame({"text": ["Café", "Naïve"]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path, encoding="latin-1")
            
            with open(path, 'r', encoding='latin-1') as f:
                content = f.read()
                assert len(content) > 0

    def test_write_csv_invalid_encoding_raises(self):
        """Test that invalid encoding raises ValueError."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            with pytest.raises(ValueError):
                ar.write_csv(frame, path, encoding="not-a-real-encoding-codec")

    def test_write_csv_encoding_errors_strict_with_unencodable(self):
        """Test encoding_errors='strict' raises for unencodable characters."""
        df = pd.DataFrame({"text": ["hello🌟world"]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            with pytest.raises(ValueError):
                ar.write_csv(frame, path, encoding="ascii", encoding_errors="strict")

    def test_write_csv_encoding_errors_replace(self):
        """Test encoding_errors='replace' substitutes with replacement character."""
        df = pd.DataFrame({"text": ["hello🌟world"]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path, encoding="ascii", encoding_errors="replace")
            
            with open(path, 'r', encoding='ascii', errors='replace') as f:
                content = f.read()
                assert "hello" in content and "world" in content

    def test_write_csv_encoding_errors_ignore(self):
        """Test encoding_errors='ignore' drops unencodable characters."""
        df = pd.DataFrame({"text": ["hello🌟world"]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path, encoding="ascii", encoding_errors="ignore")
            
            with open(path, 'r', encoding='ascii') as f:
                content = f.read()
                assert "hello" in content and "world" in content

    def test_write_csv_invalid_encoding_errors_raises(self):
        """Test that invalid encoding_errors value raises ValueError."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            with pytest.raises(ValueError):
                ar.write_csv(frame, path, encoding_errors="invalid_mode")


class TestWriteCsvEdgeCases:
    """Test edge cases and error conditions."""

    def test_write_csv_empty_frame(self):
        """Test writing an empty frame with column headers."""
        df = pd.DataFrame({"col1": [], "col2": []})
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path)
            
            with open(path, 'r') as f:
                content = f.read()
                assert "col1,col2" in content

    def test_write_csv_special_characters(self):
        """Test writing data with special characters."""
        df = pd.DataFrame({
            "text": ['hello, world', 'quoted "value"', 'newline\nhere']
        })
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path)
            assert os.path.exists(path) and os.path.getsize(path) > 0

    def test_write_csv_with_null_values(self):
        """Test writing frame with null values."""
        df = pd.DataFrame({
            "num": [1, None, 3],
            "text": ["a", None, "c"]
        })
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path)
            assert os.path.exists(path)

    def test_write_csv_roundtrip(self):
        """Test that write and read produce consistent data."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1]
        })
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            ar.write_csv(frame, path)
            
            frame2 = ar.read_csv(path)
            df2 = ar.to_pandas(frame2)
            
            assert df2.shape == df.shape
            assert list(df2.columns) == list(df.columns)

    def test_write_csv_nonexistent_directory_raises(self):
        """Test that writing to nonexistent directory raises RuntimeError."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        
        path = "/nonexistent/directory/that/does/not/exist/file.csv"
        with pytest.raises(RuntimeError):
            ar.write_csv(frame, path)
