import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

import arnio as ar


@pytest.fixture
def sample_arframe():
    """Create a basic ArFrame for testing"""
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'col3': [1.1, 2.2, 3.3]
    })
    return ar.ArFrame(df)


@pytest.fixture
def arframe_with_attrs():
    """Create an ArFrame with JSON-serializable attributes"""
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    frame = ar.ArFrame(df)
    frame.attrs = {'name': 'test', 'version': 1, 'tags': ['a', 'b']}
    return frame


@pytest.fixture
def temp_dir():
    """Provide a temporary directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Basic functionality tests
def test_write_parquet_basic(sample_arframe, temp_dir):
    """Test basic write_parquet functionality"""
    path = os.path.join(temp_dir, 'test.parquet')
    ar.write_parquet(sample_arframe, path)
    assert os.path.exists(path)
    df = pd.read_parquet(path)
    assert len(df) == 3
    assert list(df.columns) == ['col1', 'col2', 'col3']


def test_write_parquet_pq_extension(sample_arframe, temp_dir):
    """Test write_parquet with .pq extension"""
    path = os.path.join(temp_dir, 'test.pq')
    ar.write_parquet(sample_arframe, path)
    assert os.path.exists(path)


def test_write_parquet_compression_codecs(sample_arframe, temp_dir):
    """Test different compression codecs"""
    for codec in ['snappy', 'gzip', 'zstd', 'none']:
        path = os.path.join(temp_dir, f'test_{codec}.parquet')
        ar.write_parquet(sample_arframe, path, compression=codec)
        assert os.path.exists(path)


def test_write_parquet_row_group_size(sample_arframe, temp_dir):
    """Test write_parquet with specified row_group_size"""
    path = os.path.join(temp_dir, 'test.parquet')
    ar.write_parquet(sample_arframe, path, row_group_size=1)
    assert os.path.exists(path)


def test_write_parquet_preserve_attrs(arframe_with_attrs, temp_dir):
    """Test preserve_attrs parameter"""
    path_true = os.path.join(temp_dir, 'test_true.parquet')
    path_false = os.path.join(temp_dir, 'test_false.parquet')
    ar.write_parquet(arframe_with_attrs, path_true, preserve_attrs=True)
    ar.write_parquet(arframe_with_attrs, path_false, preserve_attrs=False)
    assert os.path.exists(path_true)
    assert os.path.exists(path_false)


def test_write_parquet_path_like(sample_arframe, temp_dir):
    """Test write_parquet with Path object"""
    path = Path(temp_dir) / 'test.parquet'
    ar.write_parquet(sample_arframe, path)
    assert path.exists()


# ValueError tests
def test_write_parquet_invalid_extension(sample_arframe, temp_dir):
    """Test ValueError for invalid file extension"""
    path = os.path.join(temp_dir, 'test.csv')
    with pytest.raises(ValueError, match="Unsupported file format"):
        ar.write_parquet(sample_arframe, path)


def test_write_parquet_invalid_compression(sample_arframe, temp_dir):
    """Test ValueError for invalid compression codec"""
    path = os.path.join(temp_dir, 'test.parquet')
    with pytest.raises(ValueError, match="Unknown compression codec"):
        ar.write_parquet(sample_arframe, path, compression='invalid')


def test_write_parquet_invalid_row_group_size(sample_arframe, temp_dir):
    """Test ValueError for invalid row_group_size"""
    path = os.path.join(temp_dir, 'test.parquet')
    with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
        ar.write_parquet(sample_arframe, path, row_group_size=0)


def test_write_parquet_zero_cols_nonzero_rows(temp_dir):
    """Test ValueError for zero-column frame with rows"""
    path = os.path.join(temp_dir, 'test.parquet')
    frame = ar.ArFrame(pd.DataFrame(index=[0, 1, 2]))
    with pytest.raises(ValueError, match="zero-column"):
        ar.write_parquet(frame, path)


# TypeError tests
def test_write_parquet_not_arframe(temp_dir):
    """Test TypeError when frame is not an ArFrame"""
    path = os.path.join(temp_dir, 'test.parquet')
    df = pd.DataFrame({'a': [1, 2, 3]})
    with pytest.raises(TypeError, match="frame must be an ArFrame"):
        ar.write_parquet(df, path)


def test_write_parquet_invalid_compression_type(sample_arframe, temp_dir):
    """Test TypeError for non-string compression"""
    path = os.path.join(temp_dir, 'test.parquet')
    with pytest.raises(TypeError, match="compression must be a string"):
        ar.write_parquet(sample_arframe, path, compression=123)


def test_write_parquet_invalid_row_group_size_type(sample_arframe, temp_dir):
    """Test TypeError for non-integer row_group_size"""
    path = os.path.join(temp_dir, 'test.parquet')
    with pytest.raises(TypeError, match="row_group_size must be an integer"):
        ar.write_parquet(sample_arframe, path, row_group_size="invalid")


def test_write_parquet_invalid_preserve_attrs_type(sample_arframe, temp_dir):
    """Test TypeError for non-bool preserve_attrs"""
    path = os.path.join(temp_dir, 'test.parquet')
    with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
        ar.write_parquet(sample_arframe, path, preserve_attrs="True")


def test_write_parquet_non_serializable_attrs(sample_arframe, temp_dir):
    """Test TypeError for non-JSON-serializable attrs with preserve_attrs=True"""
    path = os.path.join(temp_dir, 'test.parquet')
    sample_arframe.attrs = {'bad': lambda x: x}
    with pytest.raises(TypeError, match="JSON-serializable"):
        ar.write_parquet(sample_arframe, path, preserve_attrs=True)
