import pytest
import pandas as pd
from pathlib import Path
import arnio as ar


class TestWriteCsvTypeValidation:
    """Test TypeError cases for write_csv input validation."""

    def test_frame_not_arframe(self):
        """TypeError if frame is not an ArFrame."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_csv(df, "output.csv")

    def test_path_invalid_type(self):
        """TypeError if path is not str, bytes, or PathLike."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="path must be a string, bytes, or os.PathLike"):
            ar.write_csv(frame, 123)

    def test_delimiter_not_string(self):
        """TypeError if delimiter is not a string."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="delimiter must be a string"):
            ar.write_csv(frame, "output.csv", delimiter=123)

    def test_line_terminator_not_string(self):
        """TypeError if line_terminator is not a string."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="line_terminator must be a string"):
            ar.write_csv(frame, "output.csv", line_terminator=123)

    def test_encoding_not_string(self):
        """TypeError if encoding is not a string."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="encoding must be a string"):
            ar.write_csv(frame, "output.csv", encoding=123)

    def test_encoding_errors_not_string(self):
        """TypeError if encoding_errors is not a string."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="encoding_errors must be a string"):
            ar.write_csv(frame, "output.csv", encoding_errors=123)

    def test_write_header_not_bool(self):
        """TypeError if write_header is not a bool (rejects int 1/0)."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="write_header must be True or False"):
            ar.write_csv(frame, "output.csv", write_header=1)

    def test_escape_formulas_not_bool(self):
        """TypeError if escape_formulas is not a bool (rejects string)."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="escape_formulas must be True or False"):
            ar.write_csv(frame, "output.csv", escape_formulas="yes")


class TestWriteCsvValueValidation:
    """Test ValueError cases for write_csv parameters."""

    def test_path_unsupported_extension(self):
        """ValueError for unsupported file extensions."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_csv(frame, "output.json")

    def test_delimiter_empty_or_multiple_chars(self):
        """ValueError if delimiter is not exactly one character."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="exactly one character"):
            ar.write_csv(frame, "output.csv", delimiter="")
        with pytest.raises(ValueError, match="exactly one character"):
            ar.write_csv(frame, "output.csv", delimiter=";;")

    def test_delimiter_newline_or_quote(self):
        """ValueError if delimiter is newline or quote character."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="newline"):
            ar.write_csv(frame, "output.csv", delimiter="\n")
        with pytest.raises(ValueError, match="quote character"):
            ar.write_csv(frame, "output.csv", delimiter='"')

    def test_line_terminator_invalid_value(self):
        """ValueError if line_terminator is not \\n, \\r\\n, or \\r."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="line_terminator must be one of"):
            ar.write_csv(frame, "output.csv", line_terminator=";")

    def test_encoding_unknown_codec(self):
        """ValueError for unknown encoding."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="Unknown encoding"):
            ar.write_csv(frame, "output.csv", encoding="nope")

    def test_encoding_errors_invalid_value(self):
        """ValueError for invalid encoding_errors value."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="encoding_errors must be one of"):
            ar.write_csv(frame, "output.csv", encoding_errors="bad")


class TestWriteCsvFunctionality:
    """Test CSV writing functionality."""

    def test_basic_write(self, tmp_path):
        """Write CSV file successfully."""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        frame = ar.from_pandas(df)
        path = tmp_path / "output.csv"
        ar.write_csv(frame, str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_pathlib_path_accepted(self, tmp_path):
        """Accept pathlib.Path as path argument."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        path = tmp_path / "output.csv"
        ar.write_csv(frame, path)  # Pass Path object directly
        assert path.exists()

    def test_custom_delimiter(self, tmp_path):
        """Custom delimiter is written to file."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        path = tmp_path / "output.csv"
        ar.write_csv(frame, str(path), delimiter=";")
        content = path.read_bytes()
        assert b";" in content

    def test_line_terminator_crlf(self, tmp_path):
        """CRLF line terminator is written when specified."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        path = tmp_path / "output.csv"
        ar.write_csv(frame, str(path), line_terminator="\r\n")
        content = path.read_bytes()
        assert b"\r\n" in content

    def test_encoding_latin1(self, tmp_path):
        """Write file with latin-1 encoding (non-UTF-8 path)."""
        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        path = tmp_path / "output.csv"
        ar.write_csv(frame, str(path), encoding="latin-1")
        content = path.read_bytes()
        # Verify latin-1 encoded 'é' byte (0xe9)
        assert b"\xe9" in content

    def test_encoding_strict_unencodable_char(self, tmp_path):
        """encoding_errors='strict' raises ValueError for unencodable chars."""
        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        path = tmp_path / "output.csv"
        with pytest.raises(ValueError, match="character cannot be encoded"):
            ar.write_csv(
                frame, str(path), encoding="ascii", encoding_errors="strict"
            )
        # File should not be created on encode error
        assert not path.exists()

    def test_encoding_replace_unencodable_char(self, tmp_path):
        """encoding_errors='replace' substitutes unencodable chars."""
        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        path = tmp_path / "output.csv"
        ar.write_csv(frame, str(path), encoding="ascii", encoding_errors="replace")
        assert path.exists()
        content = path.read_text()
        # ASCII replacement character for é is '?'
        assert "?" in content

    def test_txt_extension_supported(self, tmp_path):
        """Write to .txt file (supported extension)."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        path = tmp_path / "output.txt"
        ar.write_csv(frame, str(path))
        assert path.exists()

    def test_tsv_extension_no_auto_delimiter(self, tmp_path):
        """TSV extension does not auto-change delimiter to tab."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        path = tmp_path / "output.tsv"
        ar.write_csv(frame, str(path))
        content = path.read_bytes()
        # Default delimiter is comma, not tab
        lines = content.split(b"\n")
        if lines and lines[0]:
            assert b"," in lines[0]

    def test_case_insensitive_extension(self, tmp_path):
        """File extension check is case-insensitive."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        path = tmp_path / "output.CSV"
        ar.write_csv(frame, str(path))
        assert path.exists()

    def test_validation_before_file_creation(self, tmp_path):
        """Validation errors occur before any file is created."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        path = tmp_path / "output.json"
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_csv(frame, str(path))
        # File should not be created on validation error
        assert not path.exists()
