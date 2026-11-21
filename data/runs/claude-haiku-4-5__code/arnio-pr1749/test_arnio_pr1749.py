import pytest
import os
import tempfile
import gzip
import pandas as pd
import arnio as ar


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def create_csv(path: str, content: str) -> None:
    """Helper to create a CSV file."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        f.write(content)


def create_gz(path: str, content: str) -> None:
    """Helper to create a gzipped CSV file."""
    with gzip.open(path, 'wt', newline='', encoding='utf-8') as f:
        f.write(content)


class TestReadCsvChunkedBasic:
    """Basic functionality tests."""

    def test_yields_arframe_objects(self, temp_dir):
        """Test that read_csv_chunked yields ArFrame objects."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, "a,b,c\n1,2,3\n4,5,6\n")
        
        chunks = list(ar.read_csv_chunked(path, chunksize=10))
        
        assert len(chunks) == 1
        assert all(isinstance(chunk, ar.ArFrame) for chunk in chunks)

    def test_chunks_data_correctly(self, temp_dir):
        """Test that chunking splits data into correct chunk sizes."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, "a,b\n1,2\n3,4\n5,6\n7,8\n9,10\n")
        
        chunks = list(ar.read_csv_chunked(path, chunksize=2))
        
        assert len(chunks) == 3
        assert chunks[0].shape[0] == 2
        assert chunks[1].shape[0] == 2
        assert chunks[2].shape[0] == 1


class TestReadCsvChunkedDelimiter:
    """Delimiter handling tests."""

    def test_tsv_auto_detection_from_extension(self, temp_dir):
        """Test that .tsv files automatically use tab delimiter."""
        path = os.path.join(temp_dir, "test.tsv")
        create_csv(path, "a\tb\tc\n1\t2\t3\n4\t5\t6\n")
        
        chunks = list(ar.read_csv_chunked(path, chunksize=10))
        df = ar.to_pandas(chunks[0])
        
        assert df.shape[1] == 3
        assert list(df.columns) == ['a', 'b', 'c']

    def test_explicit_delimiter_overrides_extension(self, temp_dir):
        """Test that explicit delimiter overrides extension-based detection."""
        path = os.path.join(temp_dir, "test.tsv")
        create_csv(path, "a,b,c\n1,2,3\n4,5,6\n")
        
        chunks = list(ar.read_csv_chunked(path, delimiter=",", chunksize=10))
        df = ar.to_pandas(chunks[0])
        
        assert df.shape[1] == 3
        assert list(df.columns) == ['a', 'b', 'c']

    def test_txt_defaults_to_comma(self, temp_dir):
        """Test that .txt files default to comma delimiter."""
        path = os.path.join(temp_dir, "test.txt")
        create_csv(path, "a,b,c\n1,2,3\n4,5,6\n")
        
        chunks = list(ar.read_csv_chunked(path, chunksize=10))
        df = ar.to_pandas(chunks[0])
        
        assert df.shape[1] == 3


class TestReadCsvChunkedHeader:
    """Header handling tests."""

    def test_has_header_true_extracts_column_names(self, temp_dir):
        """Test reading with header row extracts column names."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, "name,age,city\n1,2,3\n4,5,6\n")
        
        chunks = list(ar.read_csv_chunked(path, has_header=True, chunksize=10))
        df = ar.to_pandas(chunks[0])
        
        assert list(df.columns) == ['name', 'age', 'city']
        assert df.shape[0] == 2

    def test_has_header_false_treats_first_row_as_data(self, temp_dir):
        """Test reading without header row treats first row as data."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, "1,2,3\n4,5,6\n")
        
        chunks = list(ar.read_csv_chunked(path, has_header=False, chunksize=10))
        df = ar.to_pandas(chunks[0])
        
        assert df.shape[0] == 2
        assert df.iloc[0, 0] == 1

    def test_trim_headers_removes_whitespace(self, temp_dir):
        """Test that trim_headers removes leading/trailing whitespace."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, " a , b , c \n1,2,3\n")
        
        chunks = list(ar.read_csv_chunked(path, trim_headers=True, chunksize=10))
        df = ar.to_pandas(chunks[0])
        
        assert set(df.columns) == {'a', 'b', 'c'}


class TestReadCsvChunkedRowSelection:
    """Row and column selection tests."""

    def test_usecols_filters_columns(self, temp_dir):
        """Test usecols parameter selects specific columns."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, "a,b,c,d\n1,2,3,4\n5,6,7,8\n")
        
        chunks = list(ar.read_csv_chunked(path, usecols=['a', 'c'], chunksize=10))
        df = ar.to_pandas(chunks[0])
        
        assert set(df.columns) == {'a', 'c'}
        assert df.shape[0] == 2

    def test_nrows_limits_total_data_rows(self, temp_dir):
        """Test nrows parameter limits total rows across all chunks."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, "a,b\n1,2\n3,4\n5,6\n7,8\n9,10\n")
        
        chunks = list(ar.read_csv_chunked(path, nrows=3, chunksize=2))
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        
        assert total_rows == 3

    def test_skip_rows_skips_data_rows_after_header(self, temp_dir):
        """Test skip_rows skips data rows (not header line)."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, "a,b\n1,2\n3,4\n5,6\n")
        
        chunks = list(ar.read_csv_chunked(path, skip_rows=1, chunksize=10))
        df = ar.to_pandas(chunks[0])
        
        assert df.iloc[0, 0] == 3
        assert df.shape[0] == 2

    def test_skiprows_alias_for_skip_rows(self, temp_dir):
        """Test skiprows parameter as alias for skip_rows."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, "a,b\n1,2\n3,4\n5,6\n")
        
        chunks = list(ar.read_csv_chunked(path, skiprows=2, chunksize=10))
        df = ar.to_pandas(chunks[0])
        
        assert df.iloc[0, 0] == 5

    def test_skiprows_conflicting_with_skip_rows_raises_error(self, temp_dir):
        """Test that conflicting skip_rows and skiprows raise ValueError."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, "a,b\n1,2\n3,4\n")
        
        with pytest.raises(ValueError, match="Conflicting values"):
            list(ar.read_csv_chunked(
                path, skip_rows=1, skiprows=2, chunksize=10
            ))


class TestReadCsvChunkedNullValues:
    """Null value handling tests."""

    def test_null_values_recognizes_null_strings(self, temp_dir):
        """Test null_values parameter recognizes specified null strings."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, "a,b,c\n1,NA,3\n4,,6\n7,N/A,9\n")
        
        chunks = list(ar.read_csv_chunked(
            path, null_values=['NA', '', 'N/A'], chunksize=10
        ))
        df = ar.to_pandas(chunks[0])
        
        assert pd.isna(df.iloc[0, 1])
        assert pd.isna(df.iloc[1, 1])
        assert pd.isna(df.iloc[2, 1])


class TestReadCsvChunkedFileFormats:
    """File format and compression tests."""

    def test_reads_gzipped_csv(self, temp_dir):
        """Test reading gzipped CSV files."""
        path = os.path.join(temp_dir, "test.csv.gz")
        create_gz(path, "a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
        
        chunks = list(ar.read_csv_chunked(path, chunksize=2))
        
        assert len(chunks) == 2
        df = ar.to_pandas(chunks[0])
        assert list(df.columns) == ['a', 'b', 'c']

    def test_unsupported_file_format_raises_error(self, temp_dir):
        """Test that unsupported file formats raise ValueError."""
        path = os.path.join(temp_dir, "test.json")
        create_csv(path, "{}")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            list(ar.read_csv_chunked(path, chunksize=10))


class TestReadCsvChunkedEdgeCases:
    """Edge case tests."""

    def test_nrows_zero_yields_no_chunks(self, temp_dir):
        """Test that nrows=0 yields no chunks."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, "a,b\n1,2\n3,4\n5,6\n")
        
        chunks = list(ar.read_csv_chunked(path, nrows=0, chunksize=10))
        
        assert len(chunks) == 0

    def test_nonexistent_file_raises_error(self, temp_dir):
        """Test that missing file raises an error."""
        path = os.path.join(temp_dir, "nonexistent.csv")
        
        with pytest.raises(Exception):
            list(ar.read_csv_chunked(path, chunksize=10))

    def test_single_row_file(self, temp_dir):
        """Test reading file with header and single data row."""
        path = os.path.join(temp_dir, "test.csv")
        create_csv(path, "a,b,c\n1,2,3\n")
        
        chunks = list(ar.read_csv_chunked(path, chunksize=10))
        
        assert len(chunks) == 1
        assert chunks[0].shape[0] == 1
