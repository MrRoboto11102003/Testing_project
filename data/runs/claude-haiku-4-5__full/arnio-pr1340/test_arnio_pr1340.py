import pytest
import tempfile
import os
import io
import numpy as np
import pandas as pd
from pathlib import Path

import arnio as ar


class TestReadCSV:
    """Test arnio.io.read_csv basic functionality and documented edge cases."""
    
    def test_read_csv_basic_comma_delimited(self):
        """Test basic CSV reading with comma delimiter."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name,value\n1,alice,100\n2,bob,200\n")
            f.flush()
            path = f.name
        
        try:
            frame = ar.read_csv(path)
            assert frame.shape == (2, 3)
            assert frame.columns == ["id", "name", "value"]
            df = ar.to_pandas(frame)
            assert df["id"].tolist() == [1, 2]
            assert df["name"].tolist() == ["alice", "bob"]
        finally:
            os.unlink(path)
    
    def test_read_csv_tab_delimiter_auto_detect(self):
        """Test .tsv auto-detects tab delimiter without explicit parameter."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("id\tname\tvalue\n1\talice\t100\n")
            f.flush()
            path = f.name
        
        try:
            frame = ar.read_csv(path)
            assert frame.columns == ["id", "name", "value"]
            assert frame.shape == (1, 3)
        finally:
            os.unlink(path)
    
    def test_read_csv_explicit_delimiter_overrides_extension(self):
        """Test that explicit delimiter takes precedence over extension."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("id,name,value\n1,alice,100\n")
            f.flush()
            path = f.name
        
        try:
            frame = ar.read_csv(path, delimiter=",")
            assert frame.shape == (1, 3)
            assert frame.columns == ["id", "name", "value"]
        finally:
            os.unlink(path)
    
    def test_read_csv_with_usecols_subset(self):
        """Test reading only selected columns with usecols."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name,age,city\n1,alice,25,paris\n2,bob,30,london\n")
            f.flush()
            path = f.name
        
        try:
            frame = ar.read_csv(path, usecols=["id", "name"])
            assert frame.columns == ["id", "name"]
            assert frame.shape == (2, 2)
        finally:
            os.unlink(path)
    
    def test_read_csv_with_nrows_limit(self):
        """Test nrows parameter limits number of rows read."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,value\n1,100\n2,200\n3,300\n4,400\n")
            f.flush()
            path = f.name
        
        try:
            frame = ar.read_csv(path, nrows=2)
            assert frame.shape == (2, 2)
            df = ar.to_pandas(frame)
            assert df["id"].tolist() == [1, 2]
        finally:
            os.unlink(path)
    
    def test_read_csv_trim_headers_removes_whitespace(self):
        """Test trim_headers=True removes leading/trailing whitespace from column names."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("  id  , name , value  \n1,alice,100\n")
            f.flush()
            path = f.name
        
        try:
            frame = ar.read_csv(path, trim_headers=True)
            assert frame.columns == ["id", "name", "value"]
        finally:
            os.unlink(path)
    
    def test_read_csv_null_values_custom_strings(self):
        """Test null_values recognizes custom sentinel strings as nulls."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,value\n1,100\n2,N/A\n3,MISSING\n")
            f.flush()
            path = f.name
        
        try:
            frame = ar.read_csv(path, null_values=["N/A", "MISSING"])
            df = ar.to_pandas(frame)
            assert pd.isna(df.iloc[1, 1])
            assert pd.isna(df.iloc[2, 1])
        finally:
            os.unlink(path)
    
    def test_read_csv_decimal_separator_european(self):
        """Test decimal_separator handles European-style decimals."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,price\n1,\"19,99\"\n2,\"29,95\"\n")
            f.flush()
            path = f.name
        
        try:
            frame = ar.read_csv(path, decimal_separator=",")
            df = ar.to_pandas(frame)
            assert df["price"].dtype in [np.float64, float]
        finally:
            os.unlink(path)
    
    def test_read_csv_file_not_found_raises_error(self):
        """Test reading non-existent file raises CsvReadError."""
        with pytest.raises(ar.CsvReadError, match="Cannot open file"):
            ar.read_csv("/nonexistent/path/data.csv")
    
    def test_read_csv_empty_file_raises_error(self):
        """Test reading empty file raises CsvReadError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.flush()
            path = f.name
        
        try:
            with pytest.raises(ar.CsvReadError, match="empty"):
                ar.read_csv(path)
        finally:
            os.unlink(path)


class TestReadCSVChunked:
    """Test arnio.io.read_csv_chunked streaming functionality."""
    
    def test_read_csv_chunked_basic_iteration(self):
        """Test basic chunked reading yields ArFrame objects."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,value\n")
            for i in range(1, 11):
                f.write(f"{i},{i*100}\n")
            f.flush()
            path = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=3))
            assert len(chunks) == 4  # 3+3+3+1
            assert all(hasattr(c, 'shape') for c in chunks)
            assert chunks[0].shape[0] == 3
            assert chunks[-1].shape[0] == 1
        finally:
            os.unlink(path)
    
    def test_read_csv_chunked_consistent_column_types(self):
        """Test column types are consistent across all chunks."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,value\n")
            for i in range(1, 21):
                f.write(f"{i},{i*100}\n")
            f.flush()
            path = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=5))
            first_dtypes = chunks[0].dtypes
            for chunk in chunks[1:]:
                assert chunk.dtypes == first_dtypes
        finally:
            os.unlink(path)
    
    def test_read_csv_chunked_nrows_limits_total(self):
        """Test nrows parameter limits total rows read across all chunks."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,value\n")
            for i in range(1, 21):
                f.write(f"{i},{i*100}\n")
            f.flush()
            path = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(path, chunksize=5, nrows=12))
            total_rows = sum(chunk.shape[0] for chunk in chunks)
            assert total_rows == 12
        finally:
            os.unlink(path)
    
    def test_read_csv_chunked_returns_iterator(self):
        """Test that read_csv_chunked returns a lazy iterator."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id\n1\n2\n")
            f.flush()
            path = f.name
        
        try:
            result = ar.read_csv_chunked(path, chunksize=1)
            assert hasattr(result, '__iter__')
            assert hasattr(result, '__next__')
            first_chunk = next(result)
            assert hasattr(first_chunk, 'shape')
        finally:
            os.unlink(path)
    
    def test_read_csv_chunked_preserves_column_selection(self):
        """Test usecols applies consistently to all chunks."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,name,value,extra\n")
            for i in range(1, 6):
                f.write(f"{i},user{i},{i*100},x\n")
            f.flush()
            path = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(path, usecols=["id", "value"], chunksize=2))
            for chunk in chunks:
                assert chunk.columns == ["id", "value"]
        finally:
            os.unlink(path)


class TestArnioCleaner:
    """Test arnio.integrations.sklearn.ArnioCleaner.get_feature_names_out."""
    
    def test_get_feature_names_out_returns_ndarray(self):
        """Test that get_feature_names_out returns numpy ndarray."""
        from arnio.integrations.sklearn import ArnioCleaner
        
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        cleaner = ArnioCleaner(steps=[("strip_whitespace",)])
        cleaner.fit(df)
        
        result = cleaner.get_feature_names_out()
        assert isinstance(result, np.ndarray)
        assert result.dtype == object
    
    def test_get_feature_names_out_with_none_returns_fitted_names(self):
        """Test get_feature_names_out(None) returns fitted output feature names."""
        from arnio.integrations.sklearn import ArnioCleaner
        
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        cleaner = ArnioCleaner(steps=[("strip_whitespace",)])
        cleaner.fit(df)
        
        result = cleaner.get_feature_names_out(input_features=None)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert set(result) == {"col_a", "col_b"}
    
    def test_get_feature_names_out_with_valid_input_features(self):
        """Test get_feature_names_out with matching input_features array."""
        from arnio.integrations.sklearn import ArnioCleaner
        
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        cleaner = ArnioCleaner(steps=[("strip_whitespace",)])
        cleaner.fit(df)
        
        result = cleaner.get_feature_names_out(input_features=["x", "y"])
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
