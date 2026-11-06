import pytest
import arnio as ar
import pandas as pd
import tempfile
import os
from pathlib import Path
import io


def test_read_csv_basic():
    """Test basic CSV reading with standard format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("name,age,city\nAlice,25,NYC\nBob,30,LA\n")
        
        frame = ar.read_csv(csv_path)
        assert frame.shape == (2, 3)
        assert "name" in frame.columns


def test_read_csv_file_not_found():
    """Test CsvReadError raised for non-existent file."""
    with pytest.raises(ar.CsvReadError, match="Cannot open file"):
        ar.read_csv("/nonexistent/path/file.csv")


def test_read_csv_empty_file():
    """Test CsvReadError raised for zero-byte file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "empty.csv")
        open(csv_path, 'w').close()
        
        with pytest.raises(ar.CsvReadError, match="CSV file is empty"):
            ar.read_csv(csv_path)


def test_read_csv_blank_header():
    """Test CsvReadError raised for blank column name in header."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("name,,city\nAlice,25,NYC\n")
        
        with pytest.raises(ar.CsvReadError, match="CSV header contains an empty column name"):
            ar.read_csv(csv_path)


def test_read_csv_with_usecols():
    """Test read_csv selects specified columns only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("name,age,city\nAlice,25,NYC\nBob,30,LA\n")
        
        frame = ar.read_csv(csv_path, usecols=["name", "city"])
        assert frame.shape[1] == 2
        assert "age" not in frame.columns


def test_read_csv_with_nrows():
    """Test read_csv limits row count with nrows parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("name,age\n")
            for i in range(5):
                f.write(f"Person{i},{20+i}\n")
        
        frame = ar.read_csv(csv_path, nrows=2)
        assert frame.shape[0] == 2


def test_read_csv_with_skiprows():
    """Test read_csv skips initial rows with skiprows parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("comment_line\nname,age\nAlice,25\nBob,30\n")
        
        frame = ar.read_csv(csv_path, skiprows=1)
        assert frame.shape[0] == 2
        assert "name" in frame.columns


def test_read_csv_pathlike_input():
    """Test read_csv accepts pathlib.Path objects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"
        with open(csv_path, 'w') as f:
            f.write("name,age\nAlice,25\nBob,30\n")
        
        frame = ar.read_csv(csv_path)
        assert frame.shape[0] == 2


def test_read_csv_filelike_input():
    """Test read_csv accepts file-like objects (StringIO)."""
    csv_content = "name,age\nAlice,25\nBob,30\n"
    file_obj = io.StringIO(csv_content)
    
    frame = ar.read_csv(file_obj)
    assert frame.shape == (2, 2)


def test_read_csv_with_custom_delimiter():
    """Test read_csv with non-default delimiter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("name;age;city\nAlice;25;NYC\nBob;30;LA\n")
        
        frame = ar.read_csv(csv_path, delimiter=";")
        assert frame.shape[0] == 2
        assert "name" in frame.columns


def test_read_csv_chunked_basic():
    """Test read_csv_chunked yields chunks of specified size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("name,age\n")
            for i in range(25):
                f.write(f"Person{i},{20+i}\n")
        
        chunks = list(ar.read_csv_chunked(csv_path, chunksize=10))
        assert len(chunks) >= 2
        assert chunks[0].shape[0] == 10


def test_read_csv_chunked_returns_iterator():
    """Test read_csv_chunked returns an iterator of ArFrame objects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("name,age\n")
            for i in range(10):
                f.write(f"Person{i},{20+i}\n")
        
        result = ar.read_csv_chunked(csv_path, chunksize=5)
        chunk = next(iter(result))
        assert chunk.shape[0] == 5
        assert hasattr(chunk, 'columns')


def test_read_csv_chunked_with_usecols():
    """Test read_csv_chunked respects usecols parameter across chunks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("name,age,city\n")
            for i in range(20):
                f.write(f"Person{i},{20+i},City{i}\n")
        
        chunks = list(ar.read_csv_chunked(csv_path, chunksize=5, usecols=["name", "age"]))
        assert all(chunk.shape[1] == 2 for chunk in chunks)


def test_arniocleaner_get_feature_names_out():
    """Test ArnioCleaner.get_feature_names_out returns column names after fit."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [25, 30],
        "city": ["NYC", "LA"]
    })
    
    cleaner = ar.integrations.sklearn.ArnioCleaner(
        steps=[("strip_whitespace",)]
    )
    cleaner.fit(df)
    feature_names = cleaner.get_feature_names_out()
    
    assert len(feature_names) == 3
    assert "name" in feature_names


def test_arniocleaner_get_feature_names_out_with_input_features():
    """Test ArnioCleaner.get_feature_names_out with input_features parameter."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [25, 30]
    })
    
    cleaner = ar.integrations.sklearn.ArnioCleaner(
        steps=[("strip_whitespace",)]
    )
    cleaner.fit(df)
    
    input_features = ["name", "age"]
    feature_names = cleaner.get_feature_names_out(input_features=input_features)
    assert len(feature_names) == 2
