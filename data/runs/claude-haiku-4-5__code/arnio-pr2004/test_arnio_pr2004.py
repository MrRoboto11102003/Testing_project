import pytest
import os
import tempfile
import pandas as pd
from pathlib import Path
import arnio as ar


@pytest.fixture
def simple_frame():
    """Create a simple test ArFrame."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    return ar.from_pandas(df)


class TestWriteCsvBasic:
    """Test basic write_csv functionality."""
    
    def test_write_csv_basic(self, tmp_path, simple_frame):
        """Test basic CSV writing with default parameters."""
        path = tmp_path / "test.csv"
        ar.write_csv(simple_frame, str(path))
        assert path.exists()
        df = pd.read_csv(path)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 3
    
    def test_write_csv_with_pathlib_path(self, tmp_path, simple_frame):
        """Test write_csv accepts pathlib.Path objects."""
        path = tmp_path / "test.csv"
        ar.write_csv(simple_frame, path)
        assert path.exists()
    
    def test_write_csv_csv_extension(self, tmp_path, simple_frame):
        """Test write_csv with .csv extension."""
        path = tmp_path / "output.csv"
        ar.write_csv(simple_frame, str(path))
        assert path.exists()
    
    def test_write_csv_txt_extension(self, tmp_path, simple_frame):
        """Test write_csv with .txt extension."""
        path = tmp_path / "output.txt"
        ar.write_csv(simple_frame, str(path))
        assert path.exists()
    
    def test_write_csv_tsv_extension(self, tmp_path, simple_frame):
        """Test write_csv with .tsv extension."""
        path = tmp_path / "output.tsv"
        ar.write_csv(simple_frame, str(path))
        assert path.exists()


class TestWriteCsvDelimiter:
    """Test delimiter parameter."""
    
    def test_write_csv_tab_delimiter(self, tmp_path):
        """Test writing with tab delimiter."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.tsv"
        ar.write_csv(frame, str(path), delimiter="\t")
        content = path.read_text()
        assert "\t" in content
    
    def test_write_csv_pipe_delimiter(self, tmp_path):
        """Test writing with pipe delimiter."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        ar.write_csv(frame, str(path), delimiter="|")
        content = path.read_text()
        assert "|" in content


class TestWriteCsvHeader:
    """Test write_header parameter."""
    
    def test_write_csv_with_header_default(self, tmp_path):
        """Test header is written by default."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        ar.write_csv(frame, str(path))
        lines = path.read_text().strip().split("\n")
        assert "x,y" in lines[0]
    
    def test_write_csv_without_header(self, tmp_path):
        """Test header can be omitted."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        ar.write_csv(frame, str(path), write_header=False)
        lines = path.read_text().strip().split("\n")
        assert lines[0] != "x,y"


class TestWriteCsvLineTerminator:
    """Test line_terminator parameter."""
    
    def test_write_csv_line_terminator_lf(self, tmp_path):
        """Test LF line terminator."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        ar.write_csv(frame, str(path), line_terminator="\n")
        content = path.read_bytes()
        assert b"\n" in content
    
    def test_write_csv_line_terminator_crlf(self, tmp_path):
        """Test CRLF line terminator."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        ar.write_csv(frame, str(path), line_terminator="\r\n")
        content = path.read_bytes()
        assert b"\r\n" in content
    
    def test_write_csv_line_terminator_cr(self, tmp_path):
        """Test CR line terminator."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        ar.write_csv(frame, str(path), line_terminator="\r")
        content = path.read_bytes()
        assert b"\r" in content


class TestWriteCsvFormulas:
    """Test escape_formulas parameter."""
    
    def test_write_csv_escape_formulas_false(self, tmp_path):
        """Test formula characters are not escaped by default."""
        df = pd.DataFrame({"formula": ["=SUM(1,2)"]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        ar.write_csv(frame, str(path), escape_formulas=False)
        assert path.exists()
    
    def test_write_csv_escape_formulas_true(self, tmp_path):
        """Test formula characters can be escaped."""
        df = pd.DataFrame({"data": ["=start", "+start", "-start", "@start"]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        ar.write_csv(frame, str(path), escape_formulas=True)
        assert path.exists()


class TestWriteCsvEncoding:
    """Test encoding parameter."""
    
    def test_write_csv_utf8_encoding(self, tmp_path):
        """Test UTF-8 encoding (default fast path)."""
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        ar.write_csv(frame, str(path), encoding="utf-8")
        content = path.read_text(encoding="utf-8")
        assert "Alice" in content
    
    def test_write_csv_latin1_encoding(self, tmp_path):
        """Test latin-1 encoding with transcoding."""
        df = pd.DataFrame({"text": ["hello"]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        ar.write_csv(frame, str(path), encoding="latin-1")
        assert path.exists()
    
    def test_write_csv_encoding_errors_replace(self, tmp_path):
        """Test encoding_errors='replace'."""
        df = pd.DataFrame({"text": ["test"]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        ar.write_csv(frame, str(path), encoding="ascii", encoding_errors="replace")
        assert path.exists()
    
    def test_write_csv_encoding_errors_ignore(self, tmp_path):
        """Test encoding_errors='ignore'."""
        df = pd.DataFrame({"text": ["test"]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        ar.write_csv(frame, str(path), encoding="ascii", encoding_errors="ignore")
        assert path.exists()


class TestWriteCsvErrors:
    """Test error conditions and validation."""
    
    def test_write_csv_not_arframe_raises_typeerror(self, tmp_path):
        """Test TypeError when frame is not an ArFrame."""
        path = tmp_path / "test.csv"
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_csv(pd.DataFrame({"a": [1]}), str(path))
    
    def test_write_csv_unsupported_extension_raises_valueerror(self, tmp_path):
        """Test ValueError for unsupported file extension."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.json"
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_csv(frame, str(path))
    
    def test_write_csv_invalid_line_terminator_raises_valueerror(self, tmp_path):
        """Test ValueError for invalid line_terminator."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        with pytest.raises(ValueError, match="line_terminator must be"):
            ar.write_csv(frame, str(path), line_terminator="\n\r")
    
    def test_write_csv_invalid_encoding_errors_raises_valueerror(self, tmp_path):
        """Test ValueError for invalid encoding_errors."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        with pytest.raises(ValueError):
            ar.write_csv(frame, str(path), encoding_errors="invalid")
    
    def test_write_csv_unknown_encoding_raises_error(self, tmp_path):
        """Test error for unknown encoding codec."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        path = tmp_path / "test.csv"
        with pytest.raises((ValueError, LookupError)):
            ar.write_csv(frame, str(path), encoding="nonexistent_xyz")
