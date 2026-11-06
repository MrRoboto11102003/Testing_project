import pytest
import io
from pathlib import Path
import arnio as ar
import pandas as pd


class TestReadCsvChunked:
    """Test arnio.read_csv_chunked for chunked CSV iteration."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a test CSV file with 10 data rows."""
        content = "id,name,age,city\n"
        for i in range(1, 11):
            content += f"{i},user{i},{20+i},city{i}\n"
        
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(content)
        return str(csv_path)

    def test_chunked_reading_returns_iterator(self, sample_csv):
        """Test that read_csv_chunked returns an iterator."""
        result = ar.read_csv_chunked(sample_csv, chunksize=3)
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_chunked_basic_iteration(self, sample_csv):
        """Test basic chunked iteration with default parameters."""
        chunks = list(ar.read_csv_chunked(sample_csv, chunksize=3))
        assert len(chunks) == 4  # 10 rows / 3 = 4 chunks
        assert all(isinstance(c, ar.ArFrame) for c in chunks)

    def test_chunk_sizes_correct(self, sample_csv):
        """Test individual chunk sizes match expected values."""
        chunks = list(ar.read_csv_chunked(sample_csv, chunksize=3))
        
        assert chunks[0].shape[0] == 3
        assert chunks[1].shape[0] == 3
        assert chunks[2].shape[0] == 3
        assert chunks[3].shape[0] == 1  # remainder

    def test_chunked_preserves_columns_across_chunks(self, sample_csv):
        """Test columns are identical in all chunks."""
        chunks = list(ar.read_csv_chunked(sample_csv, chunksize=2))
        
        expected_cols = ["id", "name", "age", "city"]
        for chunk in chunks:
            assert chunk.columns == expected_cols

    def test_custom_chunksize_smaller(self, sample_csv):
        """Test chunking with smaller chunksize (1 row per chunk)."""
        chunks = list(ar.read_csv_chunked(sample_csv, chunksize=1))
        assert len(chunks) == 10
        assert all(c.shape[0] == 1 for c in chunks)

    def test_custom_chunksize_larger(self, sample_csv):
        """Test chunking when chunksize exceeds total rows."""
        chunks = list(ar.read_csv_chunked(sample_csv, chunksize=20))
        assert len(chunks) == 1
        assert chunks[0].shape[0] == 10

    def test_nrows_parameter_limits_total(self, sample_csv):
        """Test nrows parameter limits total rows read."""
        chunks = list(ar.read_csv_chunked(sample_csv, chunksize=2, nrows=5))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 5

    def test_skip_rows_parameter(self, sample_csv):
        """Test skip_rows parameter skips initial rows."""
        chunks = list(ar.read_csv_chunked(sample_csv, chunksize=10, skip_rows=3))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 7  # 10 - 3 skipped

    def test_usecols_selects_columns(self, sample_csv):
        """Test usecols parameter selects specific columns."""
        chunks = list(ar.read_csv_chunked(
            sample_csv, chunksize=3, usecols=["id", "name"]
        ))
        
        for chunk in chunks:
            assert chunk.columns == ["id", "name"]
            assert chunk.shape[1] == 2

    def test_file_like_object_stringio(self):
        """Test reading from file-like object (StringIO)."""
        content = "x,y,z\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n"
        f = io.StringIO(content)
        
        chunks = list(ar.read_csv_chunked(f, chunksize=2))
        assert len(chunks) == 2
        assert chunks[0].shape[0] == 2
        assert chunks[1].shape[0] == 2

    def test_pathlike_object_input(self, sample_csv):
        """Test that pathlib.Path objects work as input."""
        p = Path(sample_csv)
        chunks = list(ar.read_csv_chunked(p, chunksize=3))
        
        assert len(chunks) == 4
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 10

    def test_custom_delimiter(self, tmp_path):
        """Test reading CSV with custom delimiter."""
        content = "a;b;c\n1;2;3\n4;5;6\n7;8;9\n"
        csv_path = tmp_path / "semicolon.csv"
        csv_path.write_text(content)
        
        chunks = list(ar.read_csv_chunked(str(csv_path), chunksize=2, delimiter=";"))
        
        assert chunks[0].columns == ["a", "b", "c"]
        assert chunks[0].shape[1] == 3

    def test_total_rows_across_chunks(self, sample_csv):
        """Test row counts sum correctly across chunks."""
        chunks = list(ar.read_csv_chunked(sample_csv, chunksize=2))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 10

    def test_null_values_parameter(self, tmp_path):
        """Test null_values parameter recognizes sentinels."""
        content = "id,status,note\n1,OK,good\n2,NA,missing\n3,NONE,null\n"
        csv_path = tmp_path / "nulls.csv"
        csv_path.write_text(content)
        
        chunks = list(ar.read_csv_chunked(
            str(csv_path),
            chunksize=10,
            null_values=["NA", "NONE"]
        ))
        
        assert len(chunks) == 1
        assert chunks[0].shape[0] == 3

    def test_decimal_separator(self, tmp_path):
        """Test decimal_separator parameter for non-dot decimals."""
        content = "id,price\n1,10,50\n2,20,99\n"
        csv_path = tmp_path / "decimal.csv"
        csv_path.write_text(content)
        
        chunks = list(ar.read_csv_chunked(
            str(csv_path),
            chunksize=10,
            delimiter=",",
            decimal_separator=","
        ))
        
        assert chunks[0].shape[0] == 2
