import pytest
import pandas as pd
import arnio as ar
import tempfile
import os


def create_test_frame():
    """Create a simple test frame."""
    return ar.from_dict({
        'col_a': [1, 2, 3, 4, 5],
        'col_b': ['a', 'b', 'c', 'd', 'e'],
        'col_c': [1.1, 2.2, 3.3, 4.4, 5.5]
    })


def test_write_parquet_basic():
    """Test basic write_parquet functionality."""
    frame = create_test_frame()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.parquet')
        ar.write_parquet(frame, path)
        assert os.path.exists(path)
        result = ar.read_parquet(path)
        assert result.shape == frame.shape
        assert result.columns == frame.columns


def test_write_parquet_extensions():
    """Test write_parquet with both .parquet and .pq extensions."""
    frame = create_test_frame()
    with tempfile.TemporaryDirectory() as tmpdir:
        for ext in ['.parquet', '.pq']:
            path = os.path.join(tmpdir, f'test{ext}')
            ar.write_parquet(frame, path)
            assert os.path.exists(path)


def test_write_parquet_invalid_extension():
    """Test write_parquet with invalid file extension."""
    frame = create_test_frame()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.csv')
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, path)


@pytest.mark.parametrize("compression", ["snappy", "gzip", "brotli", "zstd", "none"])
def test_write_parquet_compressions(compression):
    """Test write_parquet with different compression codecs."""
    frame = create_test_frame()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f'test.parquet')
        ar.write_parquet(frame, path, compression=compression)
        assert os.path.exists(path)
        result = ar.read_parquet(path)
        assert result.shape == frame.shape


def test_write_parquet_invalid_compression():
    """Test write_parquet with invalid compression codec."""
    frame = create_test_frame()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.parquet')
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, path, compression="invalid_codec")


def test_write_parquet_row_group_size():
    """Test write_parquet with valid row_group_size."""
    frame = create_test_frame()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.parquet')
        ar.write_parquet(frame, path, row_group_size=2)
        assert os.path.exists(path)


@pytest.mark.parametrize("invalid_rgs", [0, -1])
def test_write_parquet_row_group_size_invalid(invalid_rgs):
    """Test write_parquet with invalid row_group_size values."""
    frame = create_test_frame()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.parquet')
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(frame, path, row_group_size=invalid_rgs)


def test_write_parquet_row_group_size_type_error():
    """Test write_parquet with non-integer row_group_size."""
    frame = create_test_frame()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.parquet')
        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(frame, path, row_group_size="invalid")


def test_write_parquet_preserve_attrs_valid():
    """Test write_parquet with preserve_attrs=True and valid JSON-serializable attrs."""
    frame = create_test_frame()
    df = ar.to_pandas(frame)
    df.attrs = {'key': 'value', 'number': 42, 'nested': {'a': 1}}
    frame_with_attrs = ar.from_pandas(df)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.parquet')
        ar.write_parquet(frame_with_attrs, path, preserve_attrs=True)
        assert os.path.exists(path)


def test_write_parquet_preserve_attrs_false():
    """Test write_parquet with preserve_attrs=False."""
    frame = create_test_frame()
    df = ar.to_pandas(frame)
    df.attrs = {'key': 'value'}
    frame_with_attrs = ar.from_pandas(df)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.parquet')
        ar.write_parquet(frame_with_attrs, path, preserve_attrs=False)
        assert os.path.exists(path)


def test_write_parquet_preserve_attrs_invalid():
    """Test write_parquet with non-JSON-serializable attrs."""
    frame = create_test_frame()
    df = ar.to_pandas(frame)
    df.attrs = {'key': set([1, 2, 3])}
    frame_with_attrs = ar.from_pandas(df)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.parquet')
        with pytest.raises(TypeError, match="JSON-serializable"):
            ar.write_parquet(frame_with_attrs, path, preserve_attrs=True)


def test_write_parquet_not_arframe():
    """Test write_parquet with non-ArFrame input."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.parquet')
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, path)


def test_write_parquet_empty_frame():
    """Test write_parquet with an empty frame."""
    frame = ar.from_dict({'col_a': [], 'col_b': []})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.parquet')
        ar.write_parquet(frame, path)
        assert os.path.exists(path)
        result = ar.read_parquet(path)
        assert result.shape[0] == 0
