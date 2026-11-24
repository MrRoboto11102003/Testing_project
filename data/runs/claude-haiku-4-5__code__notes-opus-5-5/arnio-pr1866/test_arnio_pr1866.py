import pytest
import pandas as pd
import re
from pathlib import Path

import arnio as ar


@pytest.fixture
def simple_frame():
    """Create a simple ArFrame for testing."""
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': ['x', 'y', 'z']
    })
    return ar.from_pandas(df)


def test_write_basic(simple_frame, tmp_path):
    """Write a simple frame to parquet file."""
    output = tmp_path / "output.parquet"
    result = ar.write_parquet(simple_frame, str(output))
    assert result is None
    assert output.exists()
    read_df = pd.read_parquet(output)
    assert list(read_df.columns) == ['a', 'b']
    assert len(read_df) == 3


def test_write_pq_extension(simple_frame, tmp_path):
    """Support .pq file extension."""
    output = tmp_path / "output.pq"
    ar.write_parquet(simple_frame, str(output))
    assert output.exists()


def test_path_types(simple_frame, tmp_path):
    """Path can be string, bytes, or PathLike."""
    # String
    output1 = tmp_path / "output1.parquet"
    ar.write_parquet(simple_frame, str(output1))
    assert output1.exists()
    
    # Bytes
    output2 = tmp_path / "output2.parquet"
    ar.write_parquet(simple_frame, str(output2).encode())
    assert output2.exists()
    
    # PathLike
    output3 = tmp_path / "output3.parquet"
    ar.write_parquet(simple_frame, output3)
    assert output3.exists()


def test_frame_must_be_arframe(tmp_path):
    """Frame parameter must be an ArFrame, not DataFrame."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    output = tmp_path / "output.parquet"
    with pytest.raises(TypeError, match="frame must be an ArFrame"):
        ar.write_parquet(df, str(output))


def test_invalid_path_type(simple_frame):
    """Path must be string, bytes, or PathLike."""
    with pytest.raises(TypeError, match=r"path must be a string, bytes, or os\.PathLike object"):
        ar.write_parquet(simple_frame, path=123)


def test_invalid_file_extension(simple_frame, tmp_path):
    """File must have .parquet or .pq extension."""
    output = tmp_path / "output.csv"
    with pytest.raises(ValueError, match="Unsupported file format"):
        ar.write_parquet(simple_frame, str(output))


def test_case_insensitive_extension(simple_frame, tmp_path):
    """File extension check is case-insensitive."""
    output = tmp_path / "output.PARQUET"
    ar.write_parquet(simple_frame, str(output))
    assert output.exists()


def test_compression_codec_validation(simple_frame, tmp_path):
    """Compression must be one of the valid codecs."""
    output = tmp_path / "output.parquet"
    with pytest.raises(ValueError, match=re.escape("Unknown compression codec: 'lz4'")):
        ar.write_parquet(simple_frame, str(output), compression="lz4")


def test_compression_case_sensitive(simple_frame, tmp_path):
    """Compression codec names are case-sensitive."""
    output = tmp_path / "output.parquet"
    with pytest.raises(ValueError, match=re.escape("Unknown compression codec: 'SNAPPY'")):
        ar.write_parquet(simple_frame, str(output), compression="SNAPPY")


def test_compression_not_string(simple_frame, tmp_path):
    """Compression must be a string."""
    output = tmp_path / "output.parquet"
    with pytest.raises(TypeError, match="compression must be a string"):
        ar.write_parquet(simple_frame, str(output), compression=None)


def test_row_group_size_validation(simple_frame, tmp_path):
    """row_group_size must be a positive integer (not bool or zero)."""
    output = tmp_path / "output.parquet"
    
    # Cannot be a bool
    with pytest.raises(TypeError, match="row_group_size must be an integer"):
        ar.write_parquet(simple_frame, str(output), row_group_size=True)
    
    # Must be positive
    with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
        ar.write_parquet(simple_frame, str(output), row_group_size=0)


def test_preserve_attrs_must_be_bool(simple_frame, tmp_path):
    """preserve_attrs must be a boolean."""
    output = tmp_path / "output.parquet"
    with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
        ar.write_parquet(simple_frame, str(output), preserve_attrs=1)


def test_all_valid_compressions(simple_frame, tmp_path):
    """All documented compression codecs are accepted."""
    for codec in ["snappy", "gzip", "brotli", "zstd", "none"]:
        output = tmp_path / f"output_{codec}.parquet"
        ar.write_parquet(simple_frame, str(output), compression=codec)
        assert output.exists()


def test_valid_row_group_size(simple_frame, tmp_path):
    """Valid row_group_size parameter is accepted."""
    output = tmp_path / "output.parquet"
    ar.write_parquet(simple_frame, str(output), row_group_size=1000)
    assert output.exists()


def test_empty_dataframe(tmp_path):
    """Empty dataframe (0 rows) is allowed."""
    df = pd.DataFrame({'a': []})
    frame = ar.from_pandas(df)
    output = tmp_path / "output.parquet"
    ar.write_parquet(frame, str(output))
    assert output.exists()


def test_no_file_created_on_validation_error(simple_frame, tmp_path):
    """File is not created if validation fails."""
    output = tmp_path / "output.csv"
    with pytest.raises(ValueError):
        ar.write_parquet(simple_frame, str(output))
    assert not output.exists()
