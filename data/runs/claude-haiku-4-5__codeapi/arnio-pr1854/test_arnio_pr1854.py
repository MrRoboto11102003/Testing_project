import pytest
import io
import os
import tempfile
import pandas as pd
import arnio as ar
from pathlib import Path


@pytest.fixture
def simple_csv():
    """Create a simple CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write("name,age,score\n")
        f.write("Alice,30,85.5\n")
        f.write("Bob,25,90.0\n")
        f.write("Charlie,35,88.5\n")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def large_csv():
    """Create a large CSV file with 100 rows."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write("id,name,value\n")
        for i in range(100):
            f.write(f"{i},user_{i},{i * 10}\n")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def tsv_file():
    """Create a TSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, encoding='utf-8') as f:
        f.write("col1\tcol2\tcol3\n")
        f.write("a\tb\tc\n")
        f.write("d\te\tf\n")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def no_header_csv():
    """Create a CSV without header."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write("Alice,30\n")
        f.write("Bob,25\n")
        f.write("Charlie,35\n")
        path = f.name
    yield path
    os.unlink(path)


def test_read_csv_chunked_basic_iteration(simple_csv):
    """Test basic chunking returns iterator of ArFrames."""
    chunks = list(ar.read_csv_chunked(simple_csv))
    assert len(chunks) == 1
    assert isinstance(chunks[0], ar.ArFrame)
    assert chunks[0].shape == (3, 3)


def test_read_csv_chunked_returns_iterator(simple_csv):
    """Test that read_csv_chunked returns an iterator."""
    result = ar.read_csv_chunked(simple_csv)
    assert hasattr(result, '__iter__')
    assert hasattr(result, '__next__')


def test_read_csv_chunked_small_chunksize(large_csv):
    """Test multiple chunks with small chunksize."""
    chunks = list(ar.read_csv_chunked(large_csv, chunksize=25))
    assert len(chunks) == 4
    assert all(isinstance(c, ar.ArFrame) for c in chunks)
    assert chunks[0].shape[0] == 25
    assert chunks[-1].shape[0] == 25


def test_read_csv_chunked_columns_preserved(large_csv):
    """Test that column names are preserved across chunks."""
    chunks = list(ar.read_csv_chunked(large_csv, chunksize=30))
    expected_cols = {'id', 'name', 'value'}
    for chunk in chunks:
        assert set(chunk.columns) == expected_cols


def test_read_csv_chunked_nrows_limit(large_csv):
    """Test nrows parameter limits total rows."""
    chunks = list(ar.read_csv_chunked(large_csv, nrows=35, chunksize=20))
    total_rows = sum(c.shape[0] for c in chunks)
    assert total_rows == 35


def test_read_csv_chunked_skip_rows(large_csv):
    """Test skip_rows parameter."""
    chunks = list(ar.read_csv_chunked(large_csv, skip_rows=10))
    total_rows = sum(c.shape[0] for c in chunks)
    assert total_rows == 90


def test_read_csv_chunked_skiprows_alias(large_csv):
    """Test skiprows as alias for skip_rows."""
    chunks = list(ar.read_csv_chunked(large_csv, skiprows=10))
    total_rows = sum(c.shape[0] for c in chunks)
    assert total_rows == 90


def test_read_csv_chunked_conflicting_skip_skiprows(simple_csv):
    """Test that conflicting skip_rows and skiprows raise ValueError."""
    with pytest.raises(ValueError, match="Conflicting values"):
        list(ar.read_csv_chunked(simple_csv, skip_rows=5, skiprows=10))


def test_read_csv_chunked_usecols(large_csv):
    """Test usecols parameter selects specific columns."""
    chunks = list(ar.read_csv_chunked(large_csv, usecols=['id', 'value']))
    assert set(chunks[0].columns) == {'id', 'value'}
    assert chunks[0].shape[1] == 2


def test_read_csv_chunked_tsv_auto_detection(tsv_file):
    """Test automatic tab delimiter detection for .tsv files."""
    chunks = list(ar.read_csv_chunked(tsv_file))
    assert len(chunks) == 1
    assert set(chunks[0].columns) == {'col1', 'col2', 'col3'}
    assert chunks[0].shape[0] == 2


def test_read_csv_chunked_explicit_delimiter(tsv_file):
    """Test explicit delimiter parameter."""
    chunks = list(ar.read_csv_chunked(tsv_file, delimiter='\t'))
    assert set(chunks[0].columns) == {'col1', 'col2', 'col3'}


def test_read_csv_chunked_no_header(no_header_csv):
    """Test has_header=False parameter."""
    chunks = list(ar.read_csv_chunked(no_header_csv, has_header=False))
    assert len(chunks) == 1
    assert chunks[0].shape[0] == 3
    assert chunks[0].shape[1] == 2


def test_read_csv_chunked_pathlib_path(simple_csv):
    """Test reading from pathlib.Path."""
    chunks = list(ar.read_csv_chunked(Path(simple_csv)))
    assert len(chunks) == 1
    assert isinstance(chunks[0], ar.ArFrame)


def test_read_csv_chunked_file_like_object(simple_csv):
    """Test reading from file-like object (StringIO)."""
    with open(simple_csv, 'r') as f:
        content = f.read()
    file_obj = io.StringIO(content)
    chunks = list(ar.read_csv_chunked(file_obj))
    assert len(chunks) >= 1
    assert isinstance(chunks[0], ar.ArFrame)


def test_read_csv_chunked_combined_equals_single_read(large_csv):
    """Test that concatenating chunks equals single read."""
    chunks = list(ar.read_csv_chunked(large_csv, chunksize=30))
    combined_dfs = [ar.to_pandas(c) for c in chunks]
    combined = pd.concat(combined_dfs, ignore_index=True)
    
    full_frame = ar.read_csv(large_csv)
    full_df = ar.to_pandas(full_frame)
    
    assert len(combined) == len(full_df)
    assert list(combined.columns) == list(full_df.columns)


def test_read_csv_chunked_empty_csv():
    """Test reading CSV with only headers."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("col1,col2,col3\n")
        path = f.name
    
    try:
        chunks = list(ar.read_csv_chunked(path))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 0
    finally:
        os.unlink(path)


def test_read_csv_chunked_dtype_parameter(simple_csv):
    """Test dtype parameter for type specification."""
    chunks = list(ar.read_csv_chunked(
        simple_csv,
        dtype={'age': 'int64', 'score': 'float64'}
    ))
    assert len(chunks) == 1
    dtypes = chunks[0].dtypes
    assert 'age' in dtypes
    assert 'score' in dtypes


def test_read_csv_chunked_unsupported_format():
    """Test that unsupported file format raises ValueError."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"test": "data"}')
        path = f.name
    
    try:
        with pytest.raises(ValueError, match="Unsupported file format"):
            list(ar.read_csv_chunked(path))
    finally:
        os.unlink(path)
