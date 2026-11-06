import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a simple ArFrame for testing."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "value": [10.5, 20.3, 30.1],
    })
    return ar.from_pandas(df)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_write_parquet_default_parameters(sample_frame, temp_dir):
    """Write with default parameters creates valid .parquet file."""
    output_path = os.path.join(temp_dir, "output.parquet")
    ar.write_parquet(sample_frame, output_path)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_write_parquet_pq_extension(sample_frame, temp_dir):
    """Write to .pq extension works."""
    output_path = os.path.join(temp_dir, "output.pq")
    ar.write_parquet(sample_frame, output_path)
    assert os.path.exists(output_path)


def test_write_parquet_pathlib_path(sample_frame, temp_dir):
    """Write using pathlib.Path object."""
    output_path = Path(temp_dir) / "output.parquet"
    ar.write_parquet(sample_frame, output_path)
    assert output_path.exists()


def test_write_parquet_data_integrity(sample_frame, temp_dir):
    """Verify written file contains correct data."""
    output_path = os.path.join(temp_dir, "output.parquet")
    ar.write_parquet(sample_frame, output_path)
    
    df_read = pd.read_parquet(output_path)
    df_original = ar.to_pandas(sample_frame)
    pd.testing.assert_frame_equal(df_read, df_original)


@pytest.mark.parametrize("codec", ["snappy", "gzip", "zstd", "brotli", "none"])
def test_write_parquet_compression_codecs(sample_frame, temp_dir, codec):
    """All supported compression codecs work."""
    output_path = os.path.join(temp_dir, f"output_{codec}.parquet")
    ar.write_parquet(sample_frame, output_path, compression=codec)
    assert os.path.exists(output_path)


def test_write_parquet_invalid_compression(sample_frame, temp_dir):
    """Invalid compression codec raises ValueError."""
    output_path = os.path.join(temp_dir, "output.parquet")
    with pytest.raises(ValueError):
        ar.write_parquet(sample_frame, output_path, compression="invalid_codec")


def test_write_parquet_row_group_size(sample_frame, temp_dir):
    """Custom row_group_size parameter works."""
    output_path = os.path.join(temp_dir, "output.parquet")
    ar.write_parquet(sample_frame, output_path, row_group_size=1000)
    assert os.path.exists(output_path)


def test_write_parquet_row_group_size_zero(sample_frame, temp_dir):
    """row_group_size=0 raises ValueError."""
    output_path = os.path.join(temp_dir, "output.parquet")
    with pytest.raises(ValueError):
        ar.write_parquet(sample_frame, output_path, row_group_size=0)


def test_write_parquet_row_group_size_negative(sample_frame, temp_dir):
    """Negative row_group_size raises ValueError."""
    output_path = os.path.join(temp_dir, "output.parquet")
    with pytest.raises(ValueError):
        ar.write_parquet(sample_frame, output_path, row_group_size=-100)


def test_write_parquet_preserve_attrs_false(sample_frame, temp_dir):
    """preserve_attrs=False silently drops attrs without error."""
    output_path = os.path.join(temp_dir, "output.parquet")
    ar.write_parquet(sample_frame, output_path, preserve_attrs=False)
    assert os.path.exists(output_path)


def test_write_parquet_preserve_attrs_invalid_type(sample_frame, temp_dir):
    """Non-boolean preserve_attrs raises TypeError."""
    output_path = os.path.join(temp_dir, "output.parquet")
    with pytest.raises(TypeError):
        ar.write_parquet(sample_frame, output_path, preserve_attrs="yes")


def test_write_parquet_non_serializable_attrs(temp_dir):
    """Non-JSON-serializable attrs with preserve_attrs=True raises TypeError."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.attrs = {"func": lambda x: x}  # lambdas are not JSON-serializable
    frame = ar.from_pandas(df)
    output_path = os.path.join(temp_dir, "output.parquet")
    with pytest.raises(TypeError):
        ar.write_parquet(frame, output_path, preserve_attrs=True)


def test_write_parquet_invalid_extension(sample_frame, temp_dir):
    """Invalid file extension raises ValueError."""
    output_path = os.path.join(temp_dir, "output.csv")
    with pytest.raises(ValueError):
        ar.write_parquet(sample_frame, output_path)
