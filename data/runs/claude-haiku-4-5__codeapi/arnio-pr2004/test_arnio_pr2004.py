import pytest
import os
import tempfile
from pathlib import Path
import pandas as pd
import arnio as ar


class TestWriteCsv:
    """Tests for arnio.io.write_csv function."""

    @pytest.fixture
    def temp_dir(self):
        """Create and clean up temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame with mixed data types."""
        data = {
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'Chicago']
        }
        return ar.from_pandas(pd.DataFrame(data))

    def test_basic_write_and_pathlib(self, sample_frame, temp_dir):
        """Test basic CSV writing and pathlib.Path support."""
        # Test string path
        path_str = os.path.join(temp_dir, 'output.csv')
        ar.write_csv(sample_frame, path_str)
        assert os.path.exists(path_str)
        
        # Test pathlib.Path
        path_pathlib = Path(temp_dir) / 'output_pathlib.csv'
        ar.write_csv(sample_frame, path_pathlib)
        assert path_pathlib.exists()

    def test_delimiter_variants(self, sample_frame, temp_dir):
        """Test CSV writing with various delimiters."""
        for delimiter, fname in [(',', 'comma.csv'), ('\t', 'tab.tsv'), ('|', 'pipe.csv')]:
            path = os.path.join(temp_dir, fname)
            ar.write_csv(sample_frame, path, delimiter=delimiter)
            with open(path, 'r') as f:
                content = f.read()
            assert delimiter in content

    def test_write_header_control(self, sample_frame, temp_dir):
        """Test write_header parameter controls header presence."""
        path_with = os.path.join(temp_dir, 'with_header.csv')
        path_without = os.path.join(temp_dir, 'without_header.csv')
        
        ar.write_csv(sample_frame, path_with, write_header=True)
        ar.write_csv(sample_frame, path_without, write_header=False)
        
        with open(path_with, 'r') as f:
            assert 'name' in f.readline()
        with open(path_without, 'r') as f:
            assert 'Alice' in f.readline()

    def test_line_terminators(self, sample_frame, temp_dir):
        """Test all valid line terminators: LF, CRLF, CR."""
        for terminator in ['\n', '\r\n', '\r']:
            path = os.path.join(temp_dir, f'term_{hash(terminator)}.csv')
            ar.write_csv(sample_frame, path, line_terminator=terminator)
            with open(path, 'rb') as f:
                content = f.read()
            assert terminator.encode() in content

    def test_line_terminator_validation(self, sample_frame, temp_dir):
        """Test that invalid line_terminator raises ValueError."""
        path = os.path.join(temp_dir, 'output.csv')
        with pytest.raises(ValueError):
            ar.write_csv(sample_frame, path, line_terminator='INVALID')

    def test_unicode_encodings(self, temp_dir):
        """Test UTF-8 and latin-1 encodings with Unicode characters."""
        data = {'text': ['Hello', 'Привет', '你好', 'Café']}
        frame = ar.from_pandas(pd.DataFrame(data))
        
        # UTF-8
        path_utf8 = os.path.join(temp_dir, 'utf8.csv')
        ar.write_csv(frame, path_utf8, encoding='utf-8')
        with open(path_utf8, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'Привет' in content and '你好' in content
        
        # latin-1 (for compatible characters)
        data_latin1 = {'text': ['Café', 'Naïve']}
        frame_latin1 = ar.from_pandas(pd.DataFrame(data_latin1))
        path_latin1 = os.path.join(temp_dir, 'latin1.csv')
        ar.write_csv(frame_latin1, path_latin1, encoding='latin-1')
        with open(path_latin1, 'r', encoding='latin-1') as f:
            assert 'Café' in f.read()

    def test_encoding_errors_handling(self, temp_dir):
        """Test encoding_errors modes: strict fails, replace/ignore succeed."""
        data = {'text': ['ASCII', '你好']}
        frame = ar.from_pandas(pd.DataFrame(data))
        
        # strict should raise
        path_strict = os.path.join(temp_dir, 'strict.csv')
        with pytest.raises(ValueError):
            ar.write_csv(frame, path_strict, encoding='ascii', encoding_errors='strict')
        
        # replace and ignore should succeed
        for mode in ['replace', 'ignore']:
            path = os.path.join(temp_dir, f'{mode}.csv')
            ar.write_csv(frame, path, encoding='ascii', encoding_errors=mode)
            assert os.path.exists(path)

    def test_encoding_validation(self, sample_frame, temp_dir):
        """Test that invalid encoding and encoding_errors raise ValueError."""
        path = os.path.join(temp_dir, 'output.csv')
        
        # Invalid encoding
        with pytest.raises(ValueError):
            ar.write_csv(sample_frame, path, encoding='nonexistent-codec')
        
        # Invalid encoding_errors
        with pytest.raises(ValueError):
            ar.write_csv(sample_frame, path, encoding_errors='invalid_mode')

    def test_file_extensions(self, sample_frame, temp_dir):
        """Test supported extensions and reject unsupported ones."""
        # Supported
        for ext in ['.csv', '.txt', '.tsv']:
            path = os.path.join(temp_dir, f'output{ext}')
            ar.write_csv(sample_frame, path)
            assert os.path.exists(path)
        
        # Unsupported
        path_unsupported = os.path.join(temp_dir, 'output.json')
        with pytest.raises(ValueError):
            ar.write_csv(sample_frame, path_unsupported)

    def test_extension_case_insensitive(self, sample_frame, temp_dir):
        """Test that file extension matching is case-insensitive."""
        for ext in ['.CSV', '.Csv', '.TXT', '.Tsv']:
            path = os.path.join(temp_dir, f'output{ext}')
            ar.write_csv(sample_frame, path)
            assert os.path.exists(path)

    def test_type_validation(self, temp_dir):
        """Test TypeError for invalid frame type or path type."""
        path = os.path.join(temp_dir, 'output.csv')
        df = pd.DataFrame({'a': [1, 2, 3]})
        ar_frame = ar.from_pandas(df)
        
        # Non-ArFrame
        with pytest.raises(TypeError):
            ar.write_csv(df, path)
        
        # Invalid path type
        with pytest.raises(TypeError):
            ar.write_csv(ar_frame, 12345)

    def test_null_and_special_values(self, temp_dir):
        """Test writing frames with null/None values."""
        data = {'col1': [1, None, 3], 'col2': ['a', 'b', None], 'col3': [1.5, None, 2.5]}
        frame = ar.from_pandas(pd.DataFrame(data))
        path = os.path.join(temp_dir, 'nulls.csv')
        
        ar.write_csv(frame, path)
        assert os.path.exists(path)
        with open(path, 'r') as f:
            content = f.read()
        assert 'col1' in content and 'col2' in content

    def test_escape_formulas(self, temp_dir):
        """Test escape_formulas parameter for formula injection prevention."""
        data = {
            'values': ['=1+1', '+SUM(A1)', '-5', '@TODAY()', 'normal'],
            'numbers': [1, 2, 3, 4, 5]
        }
        frame = ar.from_pandas(pd.DataFrame(data))
        
        # Both should succeed and create valid files
        for escape in [False, True]:
            path = os.path.join(temp_dir, f'escape_{escape}.csv')
            ar.write_csv(frame, path, escape_formulas=escape)
            assert os.path.exists(path)
        
        # Numeric columns should not be affected
        path_no_escape = os.path.join(temp_dir, 'escape_False.csv')
        with open(path_no_escape, 'r') as f:
            content = f.read()
        assert '1' in content and '2' in content

    def test_empty_and_large_frames(self, temp_dir):
        """Test writing empty frames and frames with many rows."""
        # Empty frame
        empty_frame = ar.from_pandas(pd.DataFrame())
        path_empty = os.path.join(temp_dir, 'empty.csv')
        ar.write_csv(empty_frame, path_empty)
        assert os.path.exists(path_empty)
        
        # Large frame
        large_data = {'col' + str(i): list(range(1000)) for i in range(5)}
        large_frame = ar.from_pandas(pd.DataFrame(large_data))
        path_large = os.path.join(temp_dir, 'large.csv')
        ar.write_csv(large_frame, path_large)
        assert os.path.exists(path_large)
        
        with open(path_large, 'r') as f:
            lines = f.readlines()
        assert len(lines) > 1000
