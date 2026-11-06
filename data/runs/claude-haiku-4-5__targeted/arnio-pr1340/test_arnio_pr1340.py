import pytest
import arnio as ar
import pandas as pd
import tempfile
import os
from pathlib import Path
import io


class TestReadCsv:
    """Tests for arnio.io.read_csv function"""

    @pytest.fixture
    def simple_csv(self):
        """Create a simple CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            f.write("id,name,score\n")
            f.write("1,Alice,95.5\n")
            f.write("2,Bob,87.3\n")
            f.write("3,Charlie,92.1\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def simple_tsv(self):
        """Create a simple TSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, newline='') as f:
            f.write("id\tname\tscore\n")
            f.write("1\tAlice\t95.5\n")
            f.write("2\tBob\t87.3\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def csv_with_metadata(self):
        """Create a CSV with metadata lines before header."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            f.write("# Version 1.0\n")
            f.write("# Date: 2025-11-03\n")
            f.write("id,name,score\n")
            f.write("1,Alice,95.5\n")
            f.write("2,Bob,87.3\n")
            path = f.name
        yield path
        os.unlink(path)

    def test_read_csv_basic(self, simple_csv):
        """Test basic CSV reading."""
        frame = ar.read_csv(simple_csv)
        df = ar.to_pandas(frame)
        assert df.shape == (3, 3)
        assert list(df.columns) == ["id", "name", "score"]

    def test_read_csv_usecols_subset(self, simple_csv):
        """Test reading specific columns with usecols."""
        frame = ar.read_csv(simple_csv, usecols=["id", "name"])
        df = ar.to_pandas(frame)
        assert df.shape[1] == 2
        assert "score" not in df.columns

    def test_read_csv_nrows_limit(self, simple_csv):
        """Test nrows parameter limits rows to read."""
        frame = ar.read_csv(simple_csv, nrows=2)
        df = ar.to_pandas(frame)
        assert df.shape[0] == 2

    def test_read_csv_skiprows_metadata(self, csv_with_metadata):
        """Test skiprows skips lines before header."""
        frame = ar.read_csv(csv_with_metadata, skiprows=2)
        df = ar.to_pandas(frame)
        assert df.shape[0] == 2
        assert "id" in df.columns

    def test_read_csv_tsv_auto_delimiter(self, simple_tsv):
        """Test .tsv extension auto-detects tab delimiter."""
        frame = ar.read_csv(simple_tsv)
        df = ar.to_pandas(frame)
        assert df.shape == (2, 3)
        assert "id" in df.columns

    def test_read_csv_explicit_delimiter_override(self, simple_tsv):
        """Test explicit delimiter overrides auto-detection."""
        frame = ar.read_csv(simple_tsv, delimiter=",")
        df = ar.to_pandas(frame)
        # Comma delimiter on tab-separated file yields single column
        assert df.shape[1] == 1

    def test_read_csv_no_header(self, simple_csv):
        """Test has_header=False treats first row as data."""
        frame = ar.read_csv(simple_csv, has_header=False)
        df = ar.to_pandas(frame)
        # 4 rows: header row + 3 data rows all treated as data
        assert df.shape[0] == 4

    def test_read_csv_pathlib_path(self, simple_csv):
        """Test reading with pathlib.Path object."""
        path = Path(simple_csv)
        frame = ar.read_csv(path)
        df = ar.to_pandas(frame)
        assert df.shape[0] == 3

    def test_read_csv_file_like_object(self, simple_csv):
        """Test reading from file-like object."""
        with open(simple_csv, 'r') as f:
            frame = ar.read_csv(f)
        df = ar.to_pandas(frame)
        assert df.shape[0] == 3

    def test_read_csv_encoding_parameter(self, simple_csv):
        """Test encoding parameter."""
        frame = ar.read_csv(simple_csv, encoding="utf-8")
        df = ar.to_pandas(frame)
        assert df.shape == (3, 3)

    def test_read_csv_mode_strict(self, simple_csv):
        """Test strict mode (default)."""
        frame = ar.read_csv(simple_csv, mode="strict")
        df = ar.to_pandas(frame)
        assert df.shape == (3, 3)

    def test_read_csv_returns_arframe(self, simple_csv):
        """Test read_csv returns ArFrame convertible to pandas."""
        frame = ar.read_csv(simple_csv)
        # Should be convertible via to_pandas
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)


class TestReadCsvChunked:
    """Tests for arnio.io.read_csv_chunked function"""

    @pytest.fixture
    def large_csv(self):
        """Create a larger CSV file for chunking tests."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            f.write("id,value\n")
            for i in range(25):
                f.write(f"{i},{i*10}\n")
            path = f.name
        yield path
        os.unlink(path)

    def test_read_csv_chunked_returns_iterator(self, large_csv):
        """Test read_csv_chunked returns an iterator."""
        result = ar.read_csv_chunked(large_csv, chunksize=5)
        assert hasattr(result, '__iter__')

    def test_read_csv_chunked_yields_arframes(self, large_csv):
        """Test each chunk is convertible to pandas DataFrame."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=10))
        assert len(chunks) > 0
        for chunk in chunks:
            df = ar.to_pandas(chunk)
            assert isinstance(df, pd.DataFrame)

    def test_read_csv_chunked_respects_chunksize(self, large_csv):
        """Test chunksize parameter is respected."""
        for chunk in ar.read_csv_chunked(large_csv, chunksize=10):
            df = ar.to_pandas(chunk)
            assert df.shape[0] <= 10

    def test_read_csv_chunked_nrows_total_limit(self, large_csv):
        """Test nrows limits total rows across all chunks."""
        total_rows = 0
        for chunk in ar.read_csv_chunked(large_csv, chunksize=5, nrows=12):
            df = ar.to_pandas(chunk)
            total_rows += df.shape[0]
        assert total_rows <= 12

    def test_read_csv_chunked_usecols(self, large_csv):
        """Test usecols parameter with chunked reading."""
        for chunk in ar.read_csv_chunked(large_csv, chunksize=5, usecols=["id"]):
            df = ar.to_pandas(chunk)
            assert df.shape[1] == 1
            assert "id" in df.columns

    def test_read_csv_chunked_multiple_chunks(self, large_csv):
        """Test that large file produces multiple chunks."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=10))
        # 25 rows with chunksize=10 should produce 3 chunks
        assert len(chunks) >= 2


class TestArnioCleaner:
    """Tests for arnio.integrations.sklearn.ArnioCleaner.get_feature_names_out"""

    def test_get_feature_names_out_method_exists(self):
        """Test get_feature_names_out method is available."""
        try:
            from arnio.integrations.sklearn import ArnioCleaner
            cleaner = ArnioCleaner()
            assert hasattr(cleaner, 'get_feature_names_out')
            assert callable(cleaner.get_feature_names_out)
        except ImportError:
            pytest.skip("sklearn integration not available")

    def test_get_feature_names_out_accepts_none(self):
        """Test get_feature_names_out accepts input_features=None."""
        try:
            from arnio.integrations.sklearn import ArnioCleaner
            cleaner = ArnioCleaner()
            # Method should accept input_features=None per documentation
            import inspect
            sig = inspect.signature(cleaner.get_feature_names_out)
            assert 'input_features' in sig.parameters
            param = sig.parameters['input_features']
            assert param.default is None
        except ImportError:
            pytest.skip("sklearn integration not available")
