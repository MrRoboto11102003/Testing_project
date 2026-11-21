import pytest
import tempfile
import os
import io
from pathlib import Path
import numpy as np

try:
    import arnio as ar
    from arnio.integrations.sklearn import ArnioCleaner
    HAS_ARNIO = True
except ImportError:
    HAS_ARNIO = False

pytestmark = pytest.mark.skipif(not HAS_ARNIO, reason="arnio not installed")


class TestArnoiCleanerGetFeatureNamesOut:
    """Tests for ArnioCleaner.get_feature_names_out method."""

    @pytest.fixture
    def fitted_cleaner(self):
        """Create a fitted ArnioCleaner instance."""
        csv_str = "col_a,col_b,col_c\n1,2,3\n4,5,6\n7,8,9"
        df = ar.read_csv(io.StringIO(csv_str))
        cleaner = ArnioCleaner()
        cleaner.fit(df)
        return cleaner

    def test_get_feature_names_out_with_none_returns_fitted_names(self, fitted_cleaner):
        """Test that None input returns the fitted feature names."""
        result = fitted_cleaner.get_feature_names_out(None)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, fitted_cleaner.feature_names_out_)

    def test_get_feature_names_out_default_returns_fitted_names(self, fitted_cleaner):
        """Test that default call returns the fitted feature names."""
        result = fitted_cleaner.get_feature_names_out()
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, fitted_cleaner.feature_names_out_)

    def test_get_feature_names_out_raises_typeerror_on_string(self, fitted_cleaner):
        """Test that string input raises TypeError."""
        with pytest.raises(TypeError, match="must be a sequence of strings, not str"):
            fitted_cleaner.get_feature_names_out("col_a")

    def test_get_feature_names_out_raises_typeerror_on_bytes(self, fitted_cleaner):
        """Test that bytes input raises TypeError."""
        with pytest.raises(TypeError, match="must be a sequence of strings, not str"):
            fitted_cleaner.get_feature_names_out(b"col_a")

    def test_get_feature_names_out_raises_typeerror_on_non_sequence(self, fitted_cleaner):
        """Test that non-sequence input raises TypeError."""
        with pytest.raises(TypeError, match="must be a sequence"):
            fitted_cleaner.get_feature_names_out(42)

    def test_get_feature_names_out_raises_typeerror_on_non_string_elements(self, fitted_cleaner):
        """Test that non-string elements raise TypeError."""
        with pytest.raises(TypeError, match="All input_features must be strings"):
            fitted_cleaner.get_feature_names_out(["col_a", 1, "col_c"])

    def test_get_feature_names_out_raises_valueerror_on_length_mismatch(self, fitted_cleaner):
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="should have length"):
            fitted_cleaner.get_feature_names_out(["col_a", "col_b"])

    def test_get_feature_names_out_raises_valueerror_on_name_mismatch(self, fitted_cleaner):
        """Test that column name mismatch raises ValueError."""
        with pytest.raises(ValueError, match="must match columns seen during fit"):
            fitted_cleaner.get_feature_names_out(["wrong_a", "wrong_b", "wrong_c"])

    def test_get_feature_names_out_accepts_matching_column_names(self, fitted_cleaner):
        """Test that matching column names are accepted."""
        fitted_names = list(fitted_cleaner.feature_names_out_)
        result = fitted_cleaner.get_feature_names_out(fitted_names)
        np.testing.assert_array_equal(result, fitted_cleaner.feature_names_out_)

    def test_get_feature_names_out_raises_notfittederror_before_fit(self):
        """Test that calling before fit raises NotFittedError."""
        cleaner = ArnioCleaner()
        with pytest.raises(Exception):
            cleaner.get_feature_names_out()


class TestReadCsv:
    """Tests for arnio.io.read_csv function."""

    @pytest.fixture
    def basic_csv(self):
        """Create a basic CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            f.write("id,name,value\n1,alice,10.5\n2,bob,20.3\n3,charlie,15.8\n")
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def tsv_file(self):
        """Create a TSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, newline='') as f:
            f.write("id\tname\tvalue\n1\talice\t10.5\n2\tbob\t20.3\n")
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_read_csv_basic_csv_file(self, basic_csv):
        """Test reading a basic CSV file."""
        df = ar.read_csv(basic_csv)
        assert df.shape[0] == 3
        assert df.shape[1] == 3

    def test_read_csv_with_usecols(self, basic_csv):
        """Test reading specific columns with usecols."""
        df = ar.read_csv(basic_csv, usecols=["id", "name"])
        assert df.shape[1] == 2

    def test_read_csv_with_nrows(self, basic_csv):
        """Test reading limited rows with nrows."""
        df = ar.read_csv(basic_csv, nrows=2)
        assert df.shape[0] == 2

    def test_read_csv_tsv_auto_detection(self, tsv_file):
        """Test automatic tab delimiter detection for .tsv files."""
        df = ar.read_csv(tsv_file)
        assert df.shape[1] == 3
        assert df.shape[0] == 2

    def test_read_csv_explicit_delimiter_overrides_extension(self, tsv_file):
        """Test that explicit delimiter overrides file extension."""
        with pytest.raises(Exception):
            ar.read_csv(tsv_file, delimiter=",")

    def test_read_csv_from_stringio(self):
        """Test reading CSV from StringIO object."""
        csv_data = "x,y,z\n1,2,3\n4,5,6\n"
        df = ar.read_csv(io.StringIO(csv_data))
        assert df.shape == (2, 3)

    def test_read_csv_with_skiprows(self, basic_csv):
        """Test skiprows parameter skips lines before header."""
        df = ar.read_csv(basic_csv, skiprows=1)
        assert df.shape[0] == 2

    def test_read_csv_with_pathlib_path(self, basic_csv):
        """Test reading with pathlib.Path object."""
        df = ar.read_csv(Path(basic_csv))
        assert df.shape[0] == 3

    def test_read_csv_no_header_mode(self):
        """Test reading CSV without header row."""
        csv_data = "1,2,3\n4,5,6\n"
        df = ar.read_csv(io.StringIO(csv_data), has_header=False)
        assert df.shape == (2, 3)


class TestReadCsvChunked:
    """Tests for arnio.io.read_csv_chunked function."""

    @pytest.fixture
    def large_csv(self):
        """Create a CSV file with many rows."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            f.write("id,value\n")
            for i in range(100):
                f.write(f"{i},{i*10}\n")
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def tsv_large(self):
        """Create a large TSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, newline='') as f:
            f.write("id\tvalue\n")
            for i in range(60):
                f.write(f"{i}\t{i*5}\n")
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_read_csv_chunked_yields_arframes(self, large_csv):
        """Test that read_csv_chunked yields ArFrame objects."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=25))
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'shape') for chunk in chunks)

    def test_read_csv_chunked_respects_chunksize(self, large_csv):
        """Test that chunk sizes are respected."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=20))
        for chunk in chunks[:-1]:
            assert chunk.shape[0] <= 20

    def test_read_csv_chunked_total_rows_equals_input(self, large_csv):
        """Test that total rows across chunks equals file size."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=30))
        total = sum(chunk.shape[0] for chunk in chunks)
        assert total == 100

    def test_read_csv_chunked_with_nrows_limit(self, large_csv):
        """Test nrows parameter limits total rows read."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=30, nrows=50))
        total = sum(chunk.shape[0] for chunk in chunks)
        assert total <= 50

    def test_read_csv_chunked_tsv_auto_detection(self, tsv_large):
        """Test automatic tab detection for .tsv in chunked mode."""
        chunks = list(ar.read_csv_chunked(tsv_large, chunksize=15))
        assert len(chunks) > 0
        assert chunks[0].shape[1] == 2

    def test_read_csv_chunked_from_stringio(self):
        """Test chunked reading from StringIO."""
        csv_data = "a,b\n" + "\n".join(f"{i},{i*2}" for i in range(50))
        chunks = list(ar.read_csv_chunked(io.StringIO(csv_data), chunksize=10))
        total = sum(chunk.shape[0] for chunk in chunks)
        assert total == 50

    def test_read_csv_chunked_with_usecols(self, large_csv):
        """Test usecols parameter in chunked reading."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=25, usecols=["id"]))
        assert all(chunk.shape[1] == 1 for chunk in chunks)

    def test_read_csv_chunked_with_pathlib_path(self, large_csv):
        """Test chunked reading with pathlib.Path."""
        chunks = list(ar.read_csv_chunked(Path(large_csv), chunksize=30))
        assert len(chunks) > 0

    def test_read_csv_chunked_with_skiprows(self, large_csv):
        """Test skiprows parameter in chunked reading."""
        chunks_no_skip = list(ar.read_csv_chunked(large_csv, chunksize=50))
        chunks_skip = list(ar.read_csv_chunked(large_csv, chunksize=50, skiprows=10))
        total_no_skip = sum(c.shape[0] for c in chunks_no_skip)
        total_skip = sum(c.shape[0] for c in chunks_skip)
        assert total_skip < total_no_skip
