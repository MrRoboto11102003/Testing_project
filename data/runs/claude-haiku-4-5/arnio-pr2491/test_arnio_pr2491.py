import pytest
import tempfile
import os
from pathlib import Path
import arnio as ar
import pandas as pd


class TestScanCsv:
    """Tests for arnio.scan_csv"""

    @pytest.fixture
    def basic_csv(self):
        """Create a basic CSV file"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline=''
        ) as f:
            f.write("name,age,salary\n")
            f.write("Alice,30,50000\n")
            f.write("Bob,25,45000\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def tsv_file(self):
        """Create a TSV file"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.tsv', delete=False, newline=''
        ) as f:
            f.write("name\tage\tsalary\n")
            f.write("Alice\t30\t50000\n")
            f.write("Bob\t25\t45000\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def no_header_csv(self):
        """Create a CSV without header"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline=''
        ) as f:
            f.write("Alice,30,50000\n")
            f.write("Bob,25,45000\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def float_csv(self):
        """Create a CSV with float values"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline=''
        ) as f:
            f.write("item,price\n")
            f.write("A,19.99\n")
            f.write("B,25.50\n")
            path = f.name
        yield path
        os.unlink(path)

    def test_scan_csv_returns_dict(self, basic_csv):
        """Test that scan_csv returns a dictionary"""
        schema = ar.scan_csv(basic_csv)
        assert isinstance(schema, dict)

    def test_scan_csv_type_inference_int(self, basic_csv):
        """Test integer type inference"""
        schema = ar.scan_csv(basic_csv)
        assert schema['age'] == 'int64'
        assert schema['salary'] == 'int64'

    def test_scan_csv_type_inference_string(self, basic_csv):
        """Test string type inference"""
        schema = ar.scan_csv(basic_csv)
        assert schema['name'] == 'string'

    def test_scan_csv_tsv_auto_delimiter(self, tsv_file):
        """Test that .tsv files auto-detect tab delimiter"""
        schema = ar.scan_csv(tsv_file)
        assert 'name' in schema
        assert len(schema) == 3

    def test_scan_csv_explicit_delimiter_precedence(self, tsv_file):
        """Test explicit delimiter overrides auto-detection"""
        schema = ar.scan_csv(tsv_file, delimiter=",")
        assert isinstance(schema, dict)

    def test_scan_csv_file_object(self, basic_csv):
        """Test scan_csv accepts file-like objects"""
        with open(basic_csv, 'r') as f:
            schema = ar.scan_csv(f)
            assert 'name' in schema

    def test_scan_csv_pathlib_path(self, basic_csv):
        """Test scan_csv accepts pathlib.Path objects"""
        path = Path(basic_csv)
        schema = ar.scan_csv(path)
        assert 'age' in schema

    def test_scan_csv_no_header(self, no_header_csv):
        """Test has_header=False generates synthetic column names"""
        schema = ar.scan_csv(no_header_csv, has_header=False)
        assert any('col_' in str(k) for k in schema.keys())

    def test_scan_csv_sample_size(self, basic_csv):
        """Test sample_size parameter"""
        schema = ar.scan_csv(basic_csv, sample_size=1)
        assert isinstance(schema, dict)

    def test_scan_csv_invalid_delimiter_type(self, basic_csv):
        """Test non-string delimiter raises TypeError"""
        with pytest.raises(TypeError):
            ar.scan_csv(basic_csv, delimiter=123)

    def test_scan_csv_invalid_thousands_sep_type(self, basic_csv):
        """Test non-string thousands_separator raises TypeError"""
        with pytest.raises(TypeError):
            ar.scan_csv(basic_csv, thousands_separator=456)


class TestWriteCsv:
    """Tests for arnio.write_csv"""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame"""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [30, 25, 35],
            'salary': [50000, 45000, 60000]
        })
        return ar.ArFrame(df)

    @pytest.fixture
    def output_path(self):
        """Provide temporary output file path"""
        path = tempfile.mktemp(suffix='.csv')
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_write_csv_creates_file(self, sample_frame, output_path):
        """Test write_csv creates output file"""
        ar.write_csv(sample_frame, output_path)
        assert os.path.exists(output_path)

    def test_write_csv_with_header(self, sample_frame, output_path):
        """Test write_header=True includes column names"""
        ar.write_csv(sample_frame, output_path, write_header=True)
        with open(output_path, 'r') as f:
            first_line = f.readline()
            assert 'name' in first_line

    def test_write_csv_without_header(self, sample_frame, output_path):
        """Test write_header=False omits column names"""
        ar.write_csv(sample_frame, output_path, write_header=False)
        with open(output_path, 'r') as f:
            first_line = f.readline()
            assert 'Alice' in first_line or 'Bob' in first_line

    def test_write_csv_custom_delimiter(self, sample_frame, output_path):
        """Test custom delimiter parameter"""
        ar.write_csv(sample_frame, output_path, delimiter=";")
        with open(output_path, 'r') as f:
            first_line = f.readline()
            assert ';' in first_line

    def test_write_csv_encoding(self, sample_frame, output_path):
        """Test encoding parameter"""
        ar.write_csv(sample_frame, output_path, encoding='utf-8')
        assert os.path.exists(output_path)

    def test_write_csv_invalid_encoding_type(self, sample_frame, output_path):
        """Test non-string encoding raises TypeError"""
        with pytest.raises(TypeError):
            ar.write_csv(sample_frame, output_path, encoding=123)
