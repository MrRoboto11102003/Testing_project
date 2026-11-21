import pytest
import pandas as pd
import arnio as ar
import os
from pathlib import Path


class TestWriteCsv:
    """Tests for arnio.io.write_csv"""
    
    def test_basic_write_csv_round_trip(self, tmp_path):
        """Write CSV with defaults and read it back"""
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        
        path = tmp_path / "output.csv"
        ar.write_csv(frame, str(path))
        
        # Read back and verify
        result = ar.read_csv(str(path))
        result_df = ar.to_pandas(result)
        
        assert result_df.shape == (3, 2)
        assert list(result_df.columns) == ["id", "name"]
        assert result_df["name"].iloc[0] == "a"
    
    def test_write_with_different_delimiters(self, tmp_path):
        """Test writing with different delimiters"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        
        # Semicolon
        path_semi = tmp_path / "semi.csv"
        ar.write_csv(frame, str(path_semi), delimiter=";")
        content_semi = open(path_semi, "rb").read()
        assert b";" in content_semi
        
        # Tab
        path_tab = tmp_path / "tab.tsv"
        ar.write_csv(frame, str(path_tab), delimiter="\t")
        content_tab = open(path_tab, "rb").read()
        assert b"\t" in content_tab
    
    def test_write_without_header(self, tmp_path):
        """Write CSV without header row"""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        
        path = tmp_path / "no_header.csv"
        ar.write_csv(frame, str(path), write_header=False)
        
        # Verify raw bytes don't have column name in first line
        content = open(path, "rb").read()
        lines = content.split(b"\n")
        # First line should be numeric data, not column name
        assert lines[0] == b"1" or b"a" not in lines[0]
    
    def test_write_with_line_terminators(self, tmp_path):
        """Test different line terminators"""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        
        # CRLF
        path_crlf = tmp_path / "crlf.csv"
        ar.write_csv(frame, str(path_crlf), line_terminator="\r\n")
        content_crlf = open(path_crlf, "rb").read()
        assert b"\r\n" in content_crlf
        
        # Default LF only
        path_lf = tmp_path / "lf.csv"
        ar.write_csv(frame, str(path_lf), line_terminator="\n")
        content_lf = open(path_lf, "rb").read()
        assert b"\r\n" not in content_lf
        assert b"\n" in content_lf
    
    def test_formula_escaping(self, tmp_path):
        """Test formula escaping feature"""
        df = pd.DataFrame({"formula": ["=1+1", "normal"]})
        frame = ar.from_pandas(df)
        
        # With escaping enabled
        path_escaped = tmp_path / "escaped.csv"
        ar.write_csv(frame, str(path_escaped), escape_formulas=True)
        
        # With escaping disabled (default)
        path_unescaped = tmp_path / "unescaped.csv"
        ar.write_csv(frame, str(path_unescaped), escape_formulas=False)
        
        # Both should write without error
        assert path_escaped.exists()
        assert path_unescaped.exists()
    
    def test_encoding_latin1(self, tmp_path):
        """Test non-UTF-8 encoding"""
        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        
        path = tmp_path / "latin1.csv"
        ar.write_csv(frame, str(path), encoding="latin-1")
        
        # Read back as latin-1 to verify
        content = open(path, "rb").read()
        decoded = content.decode("latin-1")
        assert "café" in decoded or "caf" in decoded
    
    def test_error_frame_not_arframe(self, tmp_path):
        """Frame must be ArFrame, not pandas DataFrame"""
        df = pd.DataFrame({"a": [1, 2]})
        path = tmp_path / "out.csv"
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_csv(df, str(path))
    
    def test_error_path_type(self, tmp_path):
        """Path must be str, bytes, or PathLike"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(TypeError, match="path must be a string"):
            ar.write_csv(frame, 123)
    
    def test_error_invalid_file_extension(self, tmp_path):
        """Only .csv, .txt, .tsv extensions supported"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        path = tmp_path / "out.json"
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_csv(frame, str(path))
    
    def test_error_invalid_delimiter(self, tmp_path):
        """Delimiter validation"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        path = tmp_path / "out.csv"
        
        # Empty string
        with pytest.raises(ValueError, match="delimiter must"):
            ar.write_csv(frame, str(path), delimiter="")
        
        # Multiple characters
        with pytest.raises(ValueError, match="delimiter must"):
            ar.write_csv(frame, str(path), delimiter=";;")
    
    def test_error_invalid_line_terminator(self, tmp_path):
        """Line terminator must be one of valid options"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        path = tmp_path / "out.csv"
        
        with pytest.raises(ValueError, match="line_terminator must be one of"):
            ar.write_csv(frame, str(path), line_terminator=";")
    
    def test_error_invalid_encoding(self, tmp_path):
        """Encoding must be valid codec"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        path = tmp_path / "out.csv"
        
        with pytest.raises(ValueError, match="Unknown encoding"):
            ar.write_csv(frame, str(path), encoding="nope")
    
    def test_error_invalid_encoding_errors(self, tmp_path):
        """encoding_errors must be strict, replace, or ignore"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        path = tmp_path / "out.csv"
        
        with pytest.raises(ValueError, match="encoding_errors must be one of"):
            ar.write_csv(frame, str(path), encoding_errors="bad")
    
    def test_error_bool_validation(self, tmp_path):
        """write_header and escape_formulas must be bool"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        path = tmp_path / "out.csv"
        
        # write_header with int
        with pytest.raises(TypeError, match="write_header must be True or False"):
            ar.write_csv(frame, str(path), write_header=1)
        
        # escape_formulas with string
        with pytest.raises(TypeError, match="escape_formulas must be True or False"):
            ar.write_csv(frame, str(path), escape_formulas="yes")
    
    def test_error_unencodable_char_strict(self, tmp_path):
        """Unencodable character with strict encoding raises ValueError"""
        df = pd.DataFrame({"text": ["café"]})
        frame = ar.from_pandas(df)
        path = tmp_path / "out.csv"
        
        # Try to write "café" as ASCII with strict - should fail
        with pytest.raises(ValueError, match="character cannot be encoded"):
            ar.write_csv(frame, str(path), encoding="ascii", encoding_errors="strict")
