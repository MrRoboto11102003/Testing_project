import pytest
import io
import tempfile
import pandas as pd
import numpy as np
import arnio as ar


class TestReadCsvChunked:
    """Test suite for arnio.read_csv_chunked functionality."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV with 10 rows."""
        csv_content = """id,name,age,score
1,Alice,30,95.5
2,Bob,25,87.3
3,Charlie,35,92.1
4,Diana,28,88.9
5,Eve,32,91.2
6,Frank,29,86.5
7,Grace,31,93.8
8,Henry,26,89.1
9,Iris,34,94.2
10,Jack,27,85.6
"""
        path = tmp_path / "sample.csv"
        path.write_text(csv_content)
        return path

    @pytest.fixture
    def large_csv(self, tmp_path):
        """Create a CSV with 100 rows."""
        path = tmp_path / "large.csv"
        with open(path, "w") as f:
            f.write("id,value\n")
            for i in range(100):
                f.write(f"{i},{i*10}\n")
        return path

    @pytest.fixture
    def csv_with_nulls(self, tmp_path):
        """Create a CSV with missing values."""
        csv_content = """id,name,age
1,Alice,30
2,,25
3,Charlie,
4,Diana,28
5,,
"""
        path = tmp_path / "nulls.csv"
        path.write_text(csv_content)
        return path

    @pytest.fixture
    def semicolon_csv(self, tmp_path):
        """Create a semicolon-delimited CSV."""
        csv_content = "id;name;age\n1;Alice;30\n2;Bob;25\n3;Charlie;35\n"
        path = tmp_path / "semicolon.csv"
        path.write_text(csv_content)
        return path

    def test_chunked_basic_iteration(self, sample_csv):
        """Test basic chunked iteration with default chunksize."""
        chunks = list(ar.read_csv_chunked(str(sample_csv)))
        
        assert len(chunks) == 1  # 10 rows < 10k default
        assert chunks[0].shape == (10, 4)
        assert chunks[0].columns == ["id", "name", "age", "score"]

    def test_chunked_custom_chunksize(self, large_csv):
        """Test custom chunksize breaks file into expected chunks."""
        chunks = list(ar.read_csv_chunked(str(large_csv), chunksize=25))
        
        assert len(chunks) == 4  # 100 rows / 25 = 4 chunks
        assert all(c.shape[0] == 25 for c in chunks)

    def test_chunked_nrows_limit(self, large_csv):
        """Test nrows parameter limits total rows across all chunks."""
        chunks = list(ar.read_csv_chunked(
            str(large_csv),
            chunksize=15,
            nrows=40
        ))
        
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 40

    def test_chunked_usecols_selection(self, sample_csv):
        """Test usecols parameter selects only specified columns."""
        chunks = list(ar.read_csv_chunked(
            str(sample_csv),
            usecols=["id", "name"]
        ))
        
        assert chunks[0].columns == ["id", "name"]
        assert chunks[0].shape[1] == 2
        df = ar.to_pandas(chunks[0])
        assert "age" not in df.columns

    def test_chunked_skiprows(self, sample_csv):
        """Test skiprows parameter skips initial data rows."""
        chunks = list(ar.read_csv_chunked(
            str(sample_csv),
            skiprows=2
        ))
        
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 8  # 10 - 2 skipped

    def test_chunked_skip_rows_alias(self, sample_csv):
        """Test skip_rows as alternative to skiprows."""
        chunks_skip = list(ar.read_csv_chunked(
            str(sample_csv),
            skip_rows=3
        ))
        
        total_rows = sum(c.shape[0] for c in chunks_skip)
        assert total_rows == 7  # 10 - 3 skipped

    def test_chunked_custom_delimiter(self, semicolon_csv):
        """Test delimiter parameter for non-comma-separated files."""
        chunks = list(ar.read_csv_chunked(
            str(semicolon_csv),
            delimiter=";"
        ))
        
        assert chunks[0].columns == ["id", "name", "age"]
        df = ar.to_pandas(chunks[0])
        assert df.iloc[0]["name"] == "Alice"

    def test_chunked_no_header(self, tmp_path):
        """Test has_header=False for headerless CSV."""
        csv_content = "1,Alice,30\n2,Bob,25\n"
        path = tmp_path / "no_header.csv"
        path.write_text(csv_content)
        
        chunks = list(ar.read_csv_chunked(
            str(path),
            has_header=False
        ))
        
        assert len(chunks[0].columns) == 3
        assert all(col.startswith("col") or col.isdigit() for col in chunks[0].columns)

    def test_chunked_null_values_handling(self, csv_with_nulls):
        """Test custom null_values parameter."""
        chunks = list(ar.read_csv_chunked(
            str(csv_with_nulls),
            null_values=[""]
        ))
        
        df = ar.to_pandas(chunks[0])
        # Should have null entries
        assert df.isna().any().any()

    def test_chunked_stringio_input(self, sample_csv):
        """Test reading from file-like object (StringIO)."""
        with open(sample_csv) as f:
            content = f.read()
        
        string_io = io.StringIO(content)
        chunks = list(ar.read_csv_chunked(string_io, chunksize=5))
        
        assert len(chunks) == 2  # 10 rows / 5 per chunk
        assert chunks[0].shape[0] == 5
        assert chunks[1].shape[0] == 5

    def test_chunked_large_chunksize(self, sample_csv):
        """Test when chunksize exceeds total rows."""
        chunks = list(ar.read_csv_chunked(
            str(sample_csv),
            chunksize=500
        ))
        
        assert len(chunks) == 1
        assert chunks[0].shape[0] == 10

    def test_chunked_empty_file(self, tmp_path):
        """Test reading CSV with only header (no data rows)."""
        path = tmp_path / "empty.csv"
        path.write_text("id,name,age\n")
        
        chunks = list(ar.read_csv_chunked(str(path)))
        
        assert len(chunks) == 0

    def test_chunked_dtype_parameter(self, sample_csv):
        """Test dtype parameter for explicit column type hints."""
        chunks = list(ar.read_csv_chunked(
            str(sample_csv),
            dtype={"id": "int64", "score": "float64"}
        ))
        
        dtypes = chunks[0].dtypes
        assert dtypes["id"] == "int64"
        assert dtypes["score"] == "float64"

    def test_chunked_iterator_lazy_evaluation(self, large_csv):
        """Test that chunks are lazily evaluated (iterator, not list)."""
        chunked_iter = ar.read_csv_chunked(str(large_csv), chunksize=20)
        
        # Should be an iterator
        assert hasattr(chunked_iter, '__iter__')
        assert hasattr(chunked_iter, '__next__')
        
        # Consume one chunk
        first_chunk = next(chunked_iter)
        assert first_chunk.shape[0] == 20

    def test_chunked_decimal_separator(self, tmp_path):
        """Test decimal_separator parameter for European decimal format."""
        csv_content = "id,price\n1,19,99\n2,29,95\n"
        path = tmp_path / "decimal.csv"
        path.write_text(csv_content)
        
        chunks = list(ar.read_csv_chunked(
            str(path),
            decimal_separator=","
        ))
        
        # Verify numeric parsing occurred
        df = ar.to_pandas(chunks[0])
        assert len(df) == 2
