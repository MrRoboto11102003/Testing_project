import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

try:
    import arnio as ar
    from arnio.integrations.sklearn import ArnioCleaner
except ImportError:
    pytest.skip("arnio not installed", allow_module_level=True)


class TestReadCsv:
    """Test arnio.io.read_csv functionality."""

    @pytest.fixture
    def temp_csv(self):
        """Create a temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('id,name,age\n')
            f.write('1,Alice,30\n')
            f.write('2,Bob,25\n')
            f.write('3,Charlie,35\n')
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def temp_tsv(self):
        """Create a temporary TSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write('id\tname\tage\n')
            f.write('1\tAlice\t30\n')
            f.write('2\tBob\t25\n')
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def temp_csv_with_nulls(self):
        """Create a CSV with null/empty values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('id,name,score\n')
            f.write('1,Alice,95\n')
            f.write('2,,85\n')
            f.write('3,Charlie,\n')
            path = f.name
        yield path
        os.unlink(path)

    def test_read_csv_basic(self, temp_csv):
        """Test reading a basic CSV file returns ArFrame with correct shape."""
        frame = ar.read_csv(temp_csv)
        assert frame.shape == (3, 3)
        assert frame.columns == ['id', 'name', 'age']

    def test_read_csv_usecols_subset(self, temp_csv):
        """Test usecols parameter selects correct columns."""
        frame = ar.read_csv(temp_csv, usecols=['id', 'name'])
        assert frame.shape == (3, 2)
        assert frame.columns == ['id', 'name']

    def test_read_csv_nrows_limit(self, temp_csv):
        """Test nrows parameter limits row count."""
        frame = ar.read_csv(temp_csv, nrows=2)
        assert frame.shape == (2, 3)

    def test_read_csv_skiprows_parameter(self, temp_csv):
        """Test skiprows parameter skips data rows."""
        frame = ar.read_csv(temp_csv, skiprows=1)
        assert frame.shape == (2, 3)

    def test_read_csv_tsv_delimiter_auto_detect(self, temp_tsv):
        """Test that .tsv extension auto-detects tab delimiter."""
        frame = ar.read_csv(temp_tsv)
        assert frame.shape == (2, 3)
        assert 'id' in frame.columns

    def test_read_csv_explicit_delimiter(self, temp_csv):
        """Test explicit delimiter parameter overrides extension."""
        frame = ar.read_csv(temp_csv, delimiter=',')
        assert frame.shape == (3, 3)

    def test_read_csv_null_values(self, temp_csv_with_nulls):
        """Test reading CSV with null/empty values."""
        frame = ar.read_csv(temp_csv_with_nulls)
        df = ar.to_pandas(frame)
        assert pd.isna(df.iloc[1, 1])  # null in second row, name column

    def test_read_csv_file_not_found(self):
        """Test that reading non-existent file raises error."""
        with pytest.raises(Exception):
            ar.read_csv('/nonexistent/path/file.csv')

    def test_read_csv_returns_arframe(self, temp_csv):
        """Test that read_csv returns an ArFrame object."""
        frame = ar.read_csv(temp_csv)
        assert hasattr(frame, 'shape')
        assert hasattr(frame, 'columns')
        assert hasattr(frame, 'dtypes')


class TestReadCsvChunked:
    """Test arnio.io.read_csv_chunked functionality."""

    @pytest.fixture
    def temp_large_csv(self):
        """Create a CSV with 25 data rows for chunking tests."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('id,name,value\n')
            for i in range(1, 26):
                f.write(f'{i},name_{i},{i*10}\n')
            path = f.name
        yield path
        os.unlink(path)

    def test_read_csv_chunked_yields_arframes(self, temp_large_csv):
        """Test that read_csv_chunked yields ArFrame objects."""
        chunks = list(ar.read_csv_chunked(temp_large_csv, chunksize=5))
        assert len(chunks) > 0
        for chunk in chunks:
            assert hasattr(chunk, 'shape')
            assert hasattr(chunk, 'columns')

    def test_read_csv_chunked_respects_chunksize(self, temp_large_csv):
        """Test that chunk sizes do not exceed chunksize parameter."""
        chunksize = 10
        chunks = list(ar.read_csv_chunked(temp_large_csv, chunksize=chunksize))
        for chunk in chunks[:-1]:  # All but last chunk
            assert chunk.shape[0] == chunksize
        assert chunks[-1].shape[0] <= chunksize  # Last chunk may be smaller

    def test_read_csv_chunked_column_consistency(self, temp_large_csv):
        """Test that column names are identical across all chunks."""
        chunks = list(ar.read_csv_chunked(temp_large_csv, chunksize=8))
        first_cols = chunks[0].columns
        for chunk in chunks[1:]:
            assert chunk.columns == first_cols

    def test_read_csv_chunked_dtype_consistency(self, temp_large_csv):
        """Test that dtypes are consistent across all chunks."""
        chunks = list(ar.read_csv_chunked(temp_large_csv, chunksize=8))
        first_dtypes = chunks[0].dtypes
        for chunk in chunks[1:]:
            assert chunk.dtypes == first_dtypes

    def test_read_csv_chunked_with_usecols(self, temp_large_csv):
        """Test that usecols parameter applies to all chunks."""
        chunks = list(ar.read_csv_chunked(temp_large_csv, chunksize=10, usecols=['id', 'name']))
        assert all(chunk.shape[1] == 2 for chunk in chunks)
        assert all(chunk.columns == ['id', 'name'] for chunk in chunks)

    def test_read_csv_chunked_with_nrows(self, temp_large_csv):
        """Test that nrows limit applies across all chunks."""
        chunks = list(ar.read_csv_chunked(temp_large_csv, chunksize=10, nrows=15))
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 15

    def test_read_csv_chunked_iterator_protocol(self, temp_large_csv):
        """Test that read_csv_chunked returns a proper iterator."""
        chunks_iter = ar.read_csv_chunked(temp_large_csv, chunksize=5)
        assert hasattr(chunks_iter, '__iter__')
        first_chunk = next(chunks_iter)
        assert first_chunk.shape[0] <= 5


class TestArnioCleaner:
    """Test arnio.integrations.sklearn.ArnioCleaner.get_feature_names_out."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [10, 20, 30, 40, 50],
            'feature_3': ['a', 'b', 'c', 'd', 'e'],
        })

    @pytest.fixture
    def fitted_cleaner(self, sample_dataframe):
        """Create and fit an ArnioCleaner instance."""
        cleaner = ArnioCleaner(steps=[])
        cleaner.fit(sample_dataframe)
        return cleaner

    def test_get_feature_names_out_returns_ndarray(self, fitted_cleaner):
        """Test that get_feature_names_out returns ndarray."""
        result = fitted_cleaner.get_feature_names_out(input_features=None)
        assert isinstance(result, np.ndarray)

    def test_get_feature_names_out_with_none_input_features(self, fitted_cleaner):
        """Test get_feature_names_out with input_features=None returns fitted names."""
        result = fitted_cleaner.get_feature_names_out(input_features=None)
        assert len(result) == 3
        assert 'feature_1' in result
        assert 'feature_2' in result
        assert 'feature_3' in result

    def test_get_feature_names_out_with_input_features(self, fitted_cleaner):
        """Test get_feature_names_out with explicit input_features array."""
        input_features = ['feature_1', 'feature_2', 'feature_3']
        result = fitted_cleaner.get_feature_names_out(input_features=input_features)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_get_feature_names_out_maintains_order(self, fitted_cleaner):
        """Test that feature name order is preserved."""
        result = fitted_cleaner.get_feature_names_out(input_features=None)
        assert result[0] == 'feature_1'
        assert result[1] == 'feature_2'
        assert result[2] == 'feature_3'

    def test_get_feature_names_out_sklearn_compatible(self, sample_dataframe):
        """Test sklearn ColumnTransformer compatibility interface."""
        cleaner = ArnioCleaner(steps=[('strip_whitespace',)])
        cleaner.fit(sample_dataframe)
        names = cleaner.get_feature_names_out()
        assert callable(getattr(cleaner, 'get_feature_names_out', None))
        assert isinstance(names, np.ndarray)
