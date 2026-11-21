import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import arnio as ar


def test_write_parquet_basic(tmp_path):
    """Test basic write_parquet with default parameters and data roundtrip"""
    df = pd.DataFrame({
        'ints': [1, 2, 3],
        'floats': [1.1, 2.2, 3.3],
        'strings': ['x', 'y', 'z'],
    })
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.parquet"
    ar.write_parquet(frame, str(output_path))
    
    assert output_path.exists()
    df_read = pd.read_parquet(output_path)
    assert len(df_read) == 3
    assert list(df_read.columns) == ['ints', 'floats', 'strings']


def test_write_parquet_pq_extension(tmp_path):
    """Test write_parquet with .pq file extension"""
    df = pd.DataFrame({'x': [10, 20], 'y': [100, 200]})
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.pq"
    ar.write_parquet(frame, str(output_path))
    
    assert output_path.exists()
    df_read = pd.read_parquet(output_path)
    assert len(df_read) == 2


def test_write_parquet_compression_codecs(tmp_path):
    """Test write_parquet with different compression codecs"""
    df = pd.DataFrame({'data': np.arange(100)})
    frame = ar.from_pandas(df)
    
    for compression in ["snappy", "gzip", "none"]:
        output_path = tmp_path / f"output_{compression}.parquet"
        ar.write_parquet(frame, str(output_path), compression=compression)
        assert output_path.exists()
        df_read = pd.read_parquet(output_path)
        assert len(df_read) == 100


def test_write_parquet_row_group_size(tmp_path):
    """Test write_parquet with explicit row_group_size parameter"""
    df = pd.DataFrame({'values': np.arange(200)})
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.parquet"
    ar.write_parquet(frame, str(output_path), row_group_size=50)
    
    assert output_path.exists()
    df_read = pd.read_parquet(output_path)
    assert len(df_read) == 200


def test_write_parquet_preserve_attrs_valid_json(tmp_path):
    """Test write_parquet preserves JSON-serializable attrs"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    df.attrs = {'name': 'test', 'version': 1, 'tags': ['a', 'b'], 'enabled': True, 'score': None}
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.parquet"
    ar.write_parquet(frame, str(output_path), preserve_attrs=True)
    
    assert output_path.exists()


def test_write_parquet_preserve_attrs_false(tmp_path):
    """Test write_parquet with preserve_attrs=False drops attributes"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    df.attrs = {'metadata': 'value', 'number': 42}
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.parquet"
    ar.write_parquet(frame, str(output_path), preserve_attrs=False)
    
    assert output_path.exists()


def test_write_parquet_preserve_attrs_non_serializable_fails(tmp_path):
    """Test write_parquet raises TypeError for non-JSON-serializable attrs"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    df.attrs = {'func': lambda x: x}  # Non-serializable
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.parquet"
    with pytest.raises(TypeError, match="JSON-serializable"):
        ar.write_parquet(frame, str(output_path), preserve_attrs=True)


def test_write_parquet_invalid_frame_type(tmp_path):
    """Test write_parquet rejects non-ArFrame input"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    
    output_path = tmp_path / "output.parquet"
    with pytest.raises(TypeError, match="must be an ArFrame"):
        ar.write_parquet(df, str(output_path))


def test_write_parquet_invalid_path_type(tmp_path):
    """Test write_parquet rejects invalid path type"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError):
        ar.write_parquet(frame, 12345)


def test_write_parquet_invalid_extension(tmp_path):
    """Test write_parquet rejects unsupported file extensions"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.csv"
    with pytest.raises(ValueError, match="Unsupported file format"):
        ar.write_parquet(frame, str(output_path))


def test_write_parquet_invalid_compression_codec(tmp_path):
    """Test write_parquet rejects unrecognized compression codec"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.parquet"
    with pytest.raises(ValueError, match="Unknown compression codec"):
        ar.write_parquet(frame, str(output_path), compression="invalid_codec")


def test_write_parquet_compression_not_string(tmp_path):
    """Test write_parquet rejects non-string compression parameter"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.parquet"
    with pytest.raises(TypeError, match="compression must be a string"):
        ar.write_parquet(frame, str(output_path), compression=123)


def test_write_parquet_row_group_size_validation(tmp_path):
    """Test write_parquet validates row_group_size parameter"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    frame = ar.from_pandas(df)
    output_path = tmp_path / "output.parquet"
    
    # Non-integer value
    with pytest.raises(TypeError, match="row_group_size must be an integer"):
        ar.write_parquet(frame, str(output_path), row_group_size="100")
    
    # Zero value
    with pytest.raises(ValueError, match="positive integer"):
        ar.write_parquet(frame, str(output_path), row_group_size=0)
    
    # Negative value
    with pytest.raises(ValueError, match="positive integer"):
        ar.write_parquet(frame, str(output_path), row_group_size=-1)


def test_write_parquet_preserve_attrs_invalid_type(tmp_path):
    """Test write_parquet rejects non-bool preserve_attrs parameter"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.parquet"
    with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
        ar.write_parquet(frame, str(output_path), preserve_attrs="false")


def test_write_parquet_pathlib_path_object(tmp_path):
    """Test write_parquet accepts pathlib.Path objects"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.parquet"
    ar.write_parquet(frame, output_path)
    
    assert output_path.exists()
    df_read = pd.read_parquet(output_path)
    assert len(df_read) == 3


def test_write_parquet_case_insensitive_extension(tmp_path):
    """Test write_parquet accepts uppercase file extensions"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.PARQUET"
    ar.write_parquet(frame, str(output_path))
    
    assert output_path.exists()


def test_write_parquet_zero_columns_many_rows_fails(tmp_path):
    """Test write_parquet rejects zero-column frames with rows"""
    df = pd.DataFrame(index=np.arange(5))  # 5 rows, 0 columns
    frame = ar.from_pandas(df)
    
    output_path = tmp_path / "output.parquet"
    with pytest.raises(ValueError, match="zero-column"):
        ar.write_parquet(frame, str(output_path))
