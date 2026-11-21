import pytest
import os
import tempfile
from pathlib import Path

import arnio as ar


class TestWriteParquet:
    """Test suite for arnio.io.write_parquet"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def simple_frame(self):
        """Create a simple ArFrame for testing."""
        data = {
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'score': [85.5, 92.0, 78.5]
        }
        return ar.from_dict(data)

    @pytest.fixture
    def frame_with_nulls(self):
        """Create an ArFrame with null values."""
        data = {
            'col_a': [1, None, 3],
            'col_b': ['x', 'y', None]
        }
        return ar.from_dict(data)

    @pytest.fixture
    def empty_frame(self):
        """Create an empty ArFrame."""
        return ar.from_dict({'col1': [], 'col2': []})

    def test_write_and_read_parquet(self, simple_frame, temp_dir):
        """Test basic write and read roundtrip."""
        path = os.path.join(temp_dir, 'test.parquet')
        ar.write_parquet(simple_frame, path)
        
        assert os.path.exists(path)
        result = ar.read_parquet(path)
        assert result.shape == simple_frame.shape
        assert result.columns == simple_frame.columns

    def test_write_parquet_pq_extension(self, simple_frame, temp_dir):
        """Test writing with .pq extension."""
        path = os.path.join(temp_dir, 'test.pq')
        ar.write_parquet(simple_frame, path)
        assert os.path.exists(path)
        result = ar.read_parquet(path)
        assert result.shape == simple_frame.shape

    def test_write_parquet_with_pathlib_path(self, simple_frame, temp_dir):
        """Test writing with pathlib.Path object."""
        path = Path(temp_dir) / 'test.parquet'
        ar.write_parquet(simple_frame, path)
        assert path.exists()

    @pytest.mark.parametrize('compression', ['snappy', 'gzip', 'none'])
    def test_write_parquet_compression(self, simple_frame, temp_dir, compression):
        """Test various compression codecs."""
        path = os.path.join(temp_dir, f'test_{compression}.parquet')
        ar.write_parquet(simple_frame, path, compression=compression)
        assert os.path.exists(path)
        result = ar.read_parquet(path)
        assert result.shape == simple_frame.shape

    def test_write_parquet_row_group_size(self, simple_frame, temp_dir):
        """Test writing with custom row group size."""
        path = os.path.join(temp_dir, 'test_rg.parquet')
        ar.write_parquet(simple_frame, path, row_group_size=10)
        assert os.path.exists(path)

    def test_write_parquet_preserve_attrs_true(self, simple_frame, temp_dir):
        """Test preserving JSON-serializable attributes."""
        path = os.path.join(temp_dir, 'test_attrs.parquet')
        df = ar.to_pandas(simple_frame)
        df.attrs = {'key': 'value', 'number': 42}
        frame_with_attrs = ar.from_pandas(df)
        ar.write_parquet(frame_with_attrs, path, preserve_attrs=True)
        assert os.path.exists(path)

    def test_write_parquet_invalid_frame_type(self, temp_dir):
        """Test that TypeError is raised for non-ArFrame input."""
        path = os.path.join(temp_dir, 'test.parquet')
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet({'data': [1, 2, 3]}, path)

    def test_write_parquet_invalid_extension(self, simple_frame, temp_dir):
        """Test that ValueError is raised for unsupported file extensions."""
        path = os.path.join(temp_dir, 'test.csv')
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(simple_frame, path)

    def test_write_parquet_invalid_compression(self, simple_frame, temp_dir):
        """Test that ValueError is raised for unknown compression codec."""
        path = os.path.join(temp_dir, 'test.parquet')
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(simple_frame, path, compression='invalid')

    def test_write_parquet_invalid_row_group_size_negative(self, simple_frame, temp_dir):
        """Test that ValueError is raised for negative row_group_size."""
        path = os.path.join(temp_dir, 'test.parquet')
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(simple_frame, path, row_group_size=-1)

    def test_write_parquet_invalid_row_group_size_type(self, simple_frame, temp_dir):
        """Test that TypeError is raised for non-integer row_group_size."""
        path = os.path.join(temp_dir, 'test.parquet')
        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(simple_frame, path, row_group_size='invalid')

    def test_write_parquet_non_serializable_attrs(self, simple_frame, temp_dir):
        """Test that TypeError is raised for non-JSON-serializable attrs."""
        path = os.path.join(temp_dir, 'test.parquet')
        df = ar.to_pandas(simple_frame)
        df.attrs = {'object': object()}
        frame_with_attrs = ar.from_pandas(df)
        with pytest.raises(TypeError, match="JSON-serializable values"):
            ar.write_parquet(frame_with_attrs, path, preserve_attrs=True)

    def test_write_parquet_preserve_attrs_false(self, simple_frame, temp_dir):
        """Test that attrs are dropped when preserve_attrs=False."""
        path = os.path.join(temp_dir, 'test_no_attrs.parquet')
        df = ar.to_pandas(simple_frame)
        df.attrs = {'key': 'value'}
        frame_with_attrs = ar.from_pandas(df)
        ar.write_parquet(frame_with_attrs, path, preserve_attrs=False)
        assert os.path.exists(path)

    def test_write_parquet_with_nulls(self, frame_with_nulls, temp_dir):
        """Test that null values are preserved."""
        path = os.path.join(temp_dir, 'test_nulls.parquet')
        ar.write_parquet(frame_with_nulls, path)
        result = ar.read_parquet(path)
        assert result.shape == frame_with_nulls.shape

    def test_write_parquet_empty_frame(self, empty_frame, temp_dir):
        """Test writing an empty frame."""
        path = os.path.join(temp_dir, 'test_empty.parquet')
        ar.write_parquet(empty_frame, path)
        result = ar.read_parquet(path)
        assert result.shape == (0, 2)
