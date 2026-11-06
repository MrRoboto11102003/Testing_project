import pytest
import pandas as pd
import numpy as np
import io
import tempfile
import os
from pathlib import Path

import arnio as ar
from arnio.integrations.sklearn import ArnioCleaner
from arnio.io import read_csv, read_csv_chunked


class TestArnioCleaner:
    """Tests for ArnioCleaner.get_feature_names_out"""
    
    def test_get_feature_names_out_default(self):
        """Test get_feature_names_out with no input_features."""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        cleaner = ArnioCleaner()
        cleaner.fit(df)
        
        result = cleaner.get_feature_names_out()
        assert result is not None
        assert len(result) == 2
    
    def test_get_feature_names_out_with_input_features(self):
        """Test get_feature_names_out with input_features array."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        cleaner = ArnioCleaner()
        cleaner.fit(df)
        
        input_features = np.array(['x', 'y'])
        result = cleaner.get_feature_names_out(input_features)
        assert result is not None
        assert len(result) == 2
    
    def test_get_feature_names_out_with_none_input(self):
        """Test get_feature_names_out explicitly with None."""
        df = pd.DataFrame({'feat1': [1, 2], 'feat2': [3, 4], 'feat3': [5, 6]})
        cleaner = ArnioCleaner()
        cleaner.fit(df)
        
        result = cleaner.get_feature_names_out(input_features=None)
        assert result is not None
        assert len(result) == 3


class TestReadCSV:
    """Tests for read_csv function"""
    
    def test_read_csv_basic(self):
        """Test basic CSV reading with header."""
        csv_content = "a,b,c\n1,2,3\n4,5,6"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            result = read_csv(temp_path)
            assert len(result) == 2
            assert list(result.columns) == ['a', 'b', 'c']
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_file_object(self):
        """Test reading from file-like object."""
        csv_content = "x,y,z\n10,20,30"
        file_obj = io.StringIO(csv_content)
        
        result = read_csv(file_obj)
        assert len(result) == 1
        assert len(result.columns) == 3
    
    def test_read_csv_custom_delimiter(self):
        """Test reading with custom delimiter."""
        csv_content = "col1|col2|col3\n1|2|3\n4|5|6"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            result = read_csv(temp_path, delimiter='|')
            assert len(result.columns) == 3
            assert len(result) == 2
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_no_header(self):
        """Test reading CSV without header."""
        csv_content = "1,2,3\n4,5,6"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            result = read_csv(temp_path, has_header=False)
            assert len(result) == 2
            assert len(result.columns) == 3
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_usecols(self):
        """Test column selection with usecols parameter."""
        csv_content = "a,b,c,d,e\n1,2,3,4,5\n6,7,8,9,10"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            result = read_csv(temp_path, usecols=['a', 'c', 'e'])
            assert len(result.columns) == 3
            assert list(result.columns) == ['a', 'c', 'e']
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_nrows(self):
        """Test nrows parameter to limit row count."""
        csv_content = "x,y\n1,2\n3,4\n5,6\n7,8\n9,10"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            result = read_csv(temp_path, nrows=3)
            assert len(result) == 3
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_skiprows(self):
        """Test skiprows parameter."""
        csv_content = "a,b\n0,0\n1,2\n3,4\n5,6"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            result = read_csv(temp_path, skiprows=1)
            assert len(result) == 3
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_trim_headers(self):
        """Test header trimming behavior."""
        csv_content = " col1 , col2 , col3 \n1,2,3"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            result_trimmed = read_csv(temp_path, trim_headers=True)
            result_untrimmed = read_csv(temp_path, trim_headers=False)
            # Trimmed should not have leading/trailing spaces in column names
            assert all(str(c).strip() == str(c) for c in result_trimmed.columns)
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_decimal_separator(self):
        """Test decimal separator parameter."""
        csv_content = "value\n1,5\n2,3"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            result = read_csv(temp_path, decimal_separator=',')
            # Values should be parsed as floats with comma as decimal
            assert len(result) == 2
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_null_values(self):
        """Test null_values parameter."""
        csv_content = "data\n1\nNA\n3\nNone"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            result = read_csv(temp_path, null_values=['NA', 'None'])
            assert len(result) == 4
        finally:
            os.unlink(temp_path)


class TestReadCSVChunked:
    """Tests for read_csv_chunked function"""
    
    def test_read_csv_chunked_returns_iterator(self):
        """Test that read_csv_chunked returns an iterator."""
        csv_content = "a,b\n" + "\n".join([f"{i},{i+1}" for i in range(50)])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            result = read_csv_chunked(temp_path, chunksize=10)
            assert hasattr(result, '__iter__')
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_chunked_produces_chunks(self):
        """Test that chunks are produced from iterator."""
        csv_content = "x,y\n" + "\n".join([f"{i},{i*2}" for i in range(1, 51)])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            chunks = list(read_csv_chunked(temp_path, chunksize=10))
            assert len(chunks) > 1
            assert all(len(c) > 0 for c in chunks)
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_chunked_chunksize_respected(self):
        """Test that chunksize parameter is respected."""
        csv_content = "col\n" + "\n".join(str(i) for i in range(100))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            chunks = list(read_csv_chunked(temp_path, chunksize=15))
            # Chunks except last should be chunksize
            for chunk in chunks[:-1]:
                assert len(chunk) == 15
            # Last chunk should be <= chunksize
            assert len(chunks[-1]) <= 15
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_chunked_file_object(self):
        """Test read_csv_chunked with file-like object."""
        csv_content = "data\n" + "\n".join(str(i) for i in range(30))
        file_obj = io.StringIO(csv_content)
        
        chunks = list(read_csv_chunked(file_obj, chunksize=10))
        assert len(chunks) >= 2
        assert all(len(c) > 0 for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
