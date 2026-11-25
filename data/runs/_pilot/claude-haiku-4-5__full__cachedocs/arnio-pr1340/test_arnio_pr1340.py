import pytest
import pandas as pd
import numpy as np
import arnio as ar
import io
import tempfile
from pathlib import Path


def _sklearn_available():
    """Check if sklearn integration is available."""
    try:
        from arnio.integrations.sklearn import ArnioCleaner
        return True
    except (ImportError, AttributeError):
        return False


class TestReadCSV:
    """Test arnio.io.read_csv function."""

    def test_read_csv_from_stringio(self):
        """Test reading CSV from file-like object (StringIO)."""
        csv_data = "name,age,city\nAlice,30,Paris\nBob,25,London\n"
        frame = ar.read_csv(io.StringIO(csv_data))
        assert frame.shape == (2, 3)
        assert frame.columns == ["name", "age", "city"]

    def test_read_csv_usecols(self):
        """Test usecols parameter filters to specified columns."""
        csv_data = "name,age,city\nAlice,30,Paris\nBob,25,London\n"
        frame = ar.read_csv(io.StringIO(csv_data), usecols=["name", "age"])
        assert frame.columns == ["name", "age"]
        assert frame.shape[1] == 2

    def test_read_csv_nrows(self):
        """Test nrows parameter limits number of rows."""
        csv_data = "name,age\nAlice,30\nBob,25\nCharlie,35\n"
        frame = ar.read_csv(io.StringIO(csv_data), nrows=2)
        assert frame.shape[0] == 2

    def test_read_csv_trim_headers(self):
        """Test trim_headers removes leading/trailing whitespace from column names."""
        csv_data = " name , age \nAlice,30\nBob,25\n"
        frame = ar.read_csv(io.StringIO(csv_data), trim_headers=True)
        assert frame.columns == ["name", "age"]

    def test_read_csv_dtype(self):
        """Test dtype parameter specifies explicit column types."""
        csv_data = "id,value\n1,10.5\n2,20.5\n"
        frame = ar.read_csv(io.StringIO(csv_data), dtype={"id": "int64"})
        assert frame.dtypes["id"] == "int64"

    def test_read_csv_tsv_auto_detection(self):
        """Test TSV delimiter is auto-detected from .tsv file extension."""
        csv_data = "name\tage\tcity\nAlice\t30\tParis\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write(csv_data)
            f.flush()
            fname = f.name
        try:
            frame = ar.read_csv(fname)
            assert frame.shape == (1, 3)
            assert "name" in frame.columns
        finally:
            Path(fname).unlink()

    def test_read_csv_delimiter_override(self):
        """Test explicit delimiter takes precedence over extension-based auto-detection."""
        csv_data = "name;age;city\nAlice;30;Paris\n"
        frame = ar.read_csv(io.StringIO(csv_data), delimiter=";")
        assert frame.shape == (1, 3)
        assert "name" in frame.columns

    def test_read_csv_skiprows(self):
        """Test skiprows parameter skips lines before header."""
        csv_data = "# Metadata\nname,age\nAlice,30\nBob,25\n"
        frame = ar.read_csv(io.StringIO(csv_data), skiprows=1)
        assert frame.shape == (2, 2)
        assert "name" in frame.columns


class TestReadCSVChunked:
    """Test arnio.io.read_csv_chunked function."""

    def test_read_csv_chunked_basic(self):
        """Test basic chunked reading with specified chunksize."""
        lines = ["id,value"]
        for i in range(100):
            lines.append(f"{i},{i*10}")
        csv_data = "\n".join(lines) + "\n"

        chunks = list(ar.read_csv_chunked(io.StringIO(csv_data), chunksize=25))
        assert len(chunks) == 4  # 100 rows / 25 per chunk
        assert chunks[0].shape[0] == 25

    def test_read_csv_chunked_type_consistency(self):
        """Test column types are consistent across all chunks."""
        lines = ["id,value,name"]
        for i in range(50):
            lines.append(f"{i},{i*10},val_{i}")
        csv_data = "\n".join(lines) + "\n"

        chunks = list(ar.read_csv_chunked(io.StringIO(csv_data), chunksize=20))
        first_dtypes = chunks[0].dtypes
        for chunk in chunks[1:]:
            assert dict(chunk.dtypes) == dict(first_dtypes)

    def test_read_csv_chunked_last_chunk_smaller(self):
        """Test last chunk is smaller when rows don't divide evenly by chunksize."""
        lines = ["id"]
        for i in range(55):
            lines.append(str(i))
        csv_data = "\n".join(lines) + "\n"

        chunks = list(ar.read_csv_chunked(io.StringIO(csv_data), chunksize=20))
        assert chunks[-1].shape[0] == 15  # 55 % 20 = 15

    def test_read_csv_chunked_nrows(self):
        """Test nrows parameter limits total rows across all chunks."""
        lines = ["id"]
        for i in range(100):
            lines.append(str(i))
        csv_data = "\n".join(lines) + "\n"

        chunks = list(ar.read_csv_chunked(
            io.StringIO(csv_data),
            chunksize=20,
            nrows=50
        ))
        total = sum(chunk.shape[0] for chunk in chunks)
        assert total == 50


@pytest.mark.skipif(
    not _sklearn_available(),
    reason="sklearn integration not available"
)
class TestArnioCleaner:
    """Test arnio.integrations.sklearn.ArnioCleaner.get_feature_names_out."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "col_a": [1, 2, 3],
            "col_b": [4, 5, 6],
            "col_c": [7, 8, 9]
        })

    @pytest.fixture
    def fitted_cleaner(self, sample_df):
        """Create a fitted ArnioCleaner instance."""
        from arnio.integrations.sklearn import ArnioCleaner
        cleaner = ArnioCleaner(steps=[("strip_whitespace",)])
        cleaner.fit(sample_df)
        return cleaner

    def test_get_feature_names_out_with_none(self, fitted_cleaner):
        """Test get_feature_names_out returns fitted feature names when input_features is None."""
        names = fitted_cleaner.get_feature_names_out(input_features=None)
        assert isinstance(names, np.ndarray)
        expected = np.array(["col_a", "col_b", "col_c"])
        np.testing.assert_array_equal(names, expected)

    def test_get_feature_names_out_with_matching_features(self, fitted_cleaner):
        """Test get_feature_names_out with input_features matching fitted names."""
        input_features = ["col_a", "col_b", "col_c"]
        names = fitted_cleaner.get_feature_names_out(input_features=input_features)
        assert isinstance(names, np.ndarray)
        np.testing.assert_array_equal(names, input_features)

    def test_get_feature_names_out_returns_ndarray(self, fitted_cleaner):
        """Test get_feature_names_out returns ndarray type."""
        result = fitted_cleaner.get_feature_names_out()
        assert isinstance(result, np.ndarray)
