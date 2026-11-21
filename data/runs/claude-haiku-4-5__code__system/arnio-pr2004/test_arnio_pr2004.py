import pytest
import sys
import os
import tempfile
import subprocess
from pathlib import Path
import arnio as ar
import pandas as pd
import numpy as np


class TestWriteCsv:
    """System tests for arnio.io.write_csv"""

    def test_basic_csv_write(self, tmp_path):
        """Test basic CSV writing to file"""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'score': [95.5, 87.3, 92.1]
        })
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "output.csv"
        ar.write_csv(frame, str(output_file))
        
        assert output_file.exists()
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 4  # header + 3 rows
        assert 'name,age,score' in lines[0]
        assert 'Alice,25,95.5' in lines[1]

    def test_csv_with_pathlib_path(self, tmp_path):
        """Test that pathlib.Path works as path argument"""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "data.csv"
        ar.write_csv(frame, output_file)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert 'x,y' in content

    def test_csv_tsv_delimiter(self, tmp_path):
        """Test TSV output with tab delimiter"""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "output.tsv"
        ar.write_csv(frame, str(output_file), delimiter="\t")
        
        content = output_file.read_text()
        assert 'col1\tcol2' in content

    def test_csv_custom_delimiter_pipe(self, tmp_path):
        """Test custom delimiter (pipe character)"""
        df = pd.DataFrame({'a': [10, 20], 'b': [30, 40]})
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "output.csv"
        ar.write_csv(frame, str(output_file), delimiter="|")
        
        content = output_file.read_text()
        assert 'a|b' in content

    def test_csv_no_header(self, tmp_path):
        """Test CSV writing without header row"""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "no_header.csv"
        ar.write_csv(frame, str(output_file), write_header=False)
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 3  # only data rows, no header

    def test_csv_line_terminator_crlf(self, tmp_path):
        """Test Windows line terminator (CRLF)"""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "crlf.csv"
        ar.write_csv(frame, str(output_file), line_terminator="\r\n")
        
        with open(output_file, 'rb') as f:
            content = f.read()
        
        assert b'\r\n' in content

    def test_csv_escape_formulas(self, tmp_path):
        """Test formula escaping for spreadsheet security"""
        df = pd.DataFrame({
            'formula': ['=SUM(1,2)', '+10', '-5', '@host', 'normal']
        })
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "formulas.csv"
        ar.write_csv(frame, str(output_file), escape_formulas=True)
        
        content = output_file.read_text()
        # Formula-triggering characters should be escaped
        assert "'" in content

    def test_csv_txt_extension_support(self, tmp_path):
        """Test that .txt extension is supported"""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "output.txt"
        ar.write_csv(frame, str(output_file))
        
        assert output_file.exists()
        content = output_file.read_text()
        assert 'a,b' in content

    def test_csv_encoding_utf8_unicode(self, tmp_path):
        """Test UTF-8 encoding with unicode characters"""
        df = pd.DataFrame({
            'name': ['Alice', 'François', '日本'],
            'city': ['NYC', 'Paris', 'Tokyo']
        })
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "utf8.csv"
        ar.write_csv(frame, str(output_file), encoding="utf-8")
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'François' in content
        assert '日本' in content

    def test_csv_encoding_latin1(self, tmp_path):
        """Test latin-1 encoding for Western European characters"""
        df = pd.DataFrame({
            'name': ['Alice', 'François'],
            'city': ['NYC', 'Paris']
        })
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "latin1.csv"
        ar.write_csv(frame, str(output_file), encoding="latin-1")
        
        with open(output_file, 'r', encoding='latin-1') as f:
            content = f.read()
        
        assert 'François' in content

    def test_csv_encoding_errors_replace(self, tmp_path):
        """Test encoding_errors='replace' substitutes unencodable characters"""
        df = pd.DataFrame({'text': ['hello', '日本語', 'world']})
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "replace.csv"
        ar.write_csv(
            frame, 
            str(output_file), 
            encoding="latin-1", 
            encoding_errors="replace"
        )
        
        assert output_file.exists()
        with open(output_file, 'r', encoding='latin-1') as f:
            content = f.read()
        assert 'hello' in content and 'world' in content

    def test_error_non_arframe(self, tmp_path):
        """Test error when passing non-ArFrame object"""
        df = pd.DataFrame({'x': [1, 2]})
        output_file = tmp_path / "output.csv"
        
        with pytest.raises(TypeError, match="ArFrame"):
            ar.write_csv(df, str(output_file))

    def test_error_unsupported_file_extension(self, tmp_path):
        """Test error for unsupported file extension"""
        df = pd.DataFrame({'x': [1, 2]})
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "output.xlsx"
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_csv(frame, str(output_file))

    def test_error_invalid_delimiter(self, tmp_path):
        """Test error for empty delimiter string"""
        df = pd.DataFrame({'x': [1, 2]})
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "output.csv"
        
        with pytest.raises(ValueError, match="delimiter"):
            ar.write_csv(frame, str(output_file), delimiter="")

    def test_error_invalid_line_terminator(self, tmp_path):
        """Test error for invalid line terminator"""
        df = pd.DataFrame({'x': [1, 2]})
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "output.csv"
        
        with pytest.raises(ValueError, match="line_terminator"):
            ar.write_csv(frame, str(output_file), line_terminator="\n\r")

    def test_error_unknown_encoding_strict(self, tmp_path):
        """Test error when encoding_errors='strict' with unencodable characters"""
        df = pd.DataFrame({'text': ['こんにちは']})
        frame = ar.from_pandas(df)
        
        output_file = tmp_path / "output.csv"
        
        with pytest.raises(ValueError, match="cannot be encoded"):
            ar.write_csv(
                frame,
                str(output_file),
                encoding="ascii",
                encoding_errors="strict"
            )
