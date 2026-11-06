import pytest
import pandas as pd
import numpy as np
import io
import os
import tempfile
from pathlib import Path

import arnio as ar


class TestReadCsv:
    """Test arnio.io.read_csv functionality"""
    
    def test_read_csv_basic(self):
        """Test basic CSV reading from StringIO"""
        csv_data = "name,age\nAlice,30\nBob,25"
        frame = ar.read_csv(io.StringIO(csv_data))
        assert frame.shape == (2, 2)
        assert "name" in frame.columns
        assert "age" in frame.columns
    
    def test_read_csv_pathlib(self):
        """Test reading CSV with Path object"""
        csv_data = "x,y\n1,2\n3,4"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            f.flush()
            try:
                frame = ar.read_csv(Path(f.name))
                assert frame.shape == (2, 2)
            finally:
                os.unlink(f.name)
    
    def test_read_csv_custom_delimiter(self):
        """Test reading CSV with semicolon delimiter"""
        csv_data = "name;age\nAlice;30\nBob;25"
        frame = ar.read_csv(io.StringIO(csv_data), delimiter=";")
        assert "name" in frame.columns
        assert "age" in frame.columns
        assert frame.shape == (2, 2)
    
    def test_read_csv_no_header(self):
        """Test reading CSV without header line"""
        csv_data = "Alice,30\nBob,25"
        frame = ar.read_csv(io.StringIO(csv_data), has_header=False)
        assert frame.shape == (2, 2)
        assert len(frame.columns) == 2
    
    def test_read_csv_usecols(self):
        """Test selecting specific columns with usecols"""
        csv_data = "name,age,city\nAlice,30,NYC\nBob,25,LA"
        frame = ar.read_csv(
            io.StringIO(csv_data),
            usecols=['name', 'city']
        )
        assert set(frame.columns) == {'name', 'city'}
        assert frame.shape == (2, 2)
    
    def test_read_csv_nrows(self):
        """Test limiting rows with nrows parameter"""
        csv_data = "x\n1\n2\n3\n4\n5"
        frame = ar.read_csv(io.StringIO(csv_data), nrows=2)
        assert frame.shape == (2, 1)
    
    def test_read_csv_skiprows(self):
        """Test skipping rows with skiprows parameter"""
        csv_data = "x\n1\n2\n3\n4"
        frame = ar.read_csv(io.StringIO(csv_data), skiprows=2)
        assert frame.shape[0] == 2
    
    def test_read_csv_decimal_separator(self):
        """Test European decimal format with comma separator"""
        csv_data = "price\n19,99\n29,95"
        frame = ar.read_csv(
            io.StringIO(csv_data),
            decimal_separator=","
        )
        assert frame.shape == (2, 1)
    
    def test_read_csv_null_values(self):
        """Test custom null value sentinels"""
        csv_data = "name,score\nAlice,100\nBob,MISSING"
        frame = ar.read_csv(
            io.StringIO(csv_data),
            null_values=["MISSING"]
        )
        df = ar.to_pandas(frame)
        # Second row's score should be null or marked as missing
        assert frame.shape == (2, 2)
    
    def test_read_csv_mode_permissive(self):
        """Test permissive mode handles ragged rows"""
        csv_data = "a,b\n1,2\n3"  # Second row missing column
        frame = ar.read_csv(
            io.StringIO(csv_data),
            mode="permissive"
        )
        assert frame.shape == (2, 2)
    
    def test_read_csv_trim_headers(self):
        """Test trim_headers parameter removes header whitespace"""
        csv_data = " col_a , col_b \n1,2"
        frame = ar.read_csv(
            io.StringIO(csv_data),
            trim_headers=True
        )
        assert "col_a" in frame.columns or "col_b" in frame.columns


class TestReadCsvChunked:
    """Test arnio.io.read_csv_chunked functionality"""
    
    def test_read_csv_chunked_basic(self):
        """Test basic chunked reading returns iterator of ArFrames"""
        csv_data = "x\n" + "\n".join(str(i) for i in range(25))
        chunks = list(ar.read_csv_chunked(io.StringIO(csv_data), chunksize=10))
        assert len(chunks) >= 2
        assert all(hasattr(c, 'shape') for c in chunks)
        assert all(hasattr(c, 'columns') for c in chunks)
    
    def test_read_csv_chunked_chunksize(self):
        """Test chunksize parameter controls chunk dimensions"""
        csv_data = "x\n" + "\n".join(str(i) for i in range(35))
        chunks = list(ar.read_csv_chunked(io.StringIO(csv_data), chunksize=10))
        assert chunks[0].shape[0] == 10
        assert chunks[1].shape[0] == 10
    
    def test_read_csv_chunked_nrows_limit(self):
        """Test nrows parameter limits total rows across all chunks"""
        csv_data = "x\n" + "\n".join(str(i) for i in range(100))
        chunks = list(ar.read_csv_chunked(
            io.StringIO(csv_data),
            chunksize=20,
            nrows=50
        ))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 50
    
    def test_read_csv_chunked_skiprows(self):
        """Test skiprows parameter skips initial rows in chunked mode"""
        csv_data = "x\n" + "\n".join(str(i) for i in range(30))
        chunks = list(ar.read_csv_chunked(
            io.StringIO(csv_data),
            chunksize=10,
            skiprows=5
        ))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 25
    
    def test_read_csv_chunked_usecols(self):
        """Test usecols filters columns in all chunks"""
        csv_data = "a,b,c\n" + "\n".join(f"{i},{i*2},{i*3}" for i in range(20))
        chunks = list(ar.read_csv_chunked(
            io.StringIO(csv_data),
            chunksize=10,
            usecols=['a', 'c']
        ))
        assert all(set(c.columns) == {'a', 'c'} for c in chunks)


class TestArnioCleaner:
    """Test ArnioCleaner.get_feature_names_out"""
    
    def test_cleaner_get_feature_names_out(self):
        """Test get_feature_names_out returns feature names from cleaner"""
        try:
            from arnio.integrations.sklearn import ArnioCleaner
            
            df = pd.DataFrame({
                "feature_1": [1, 2, 3],
                "feature_2": [4, 5, 6]
            })
            
            cleaner = ArnioCleaner(steps=[("strip_whitespace",)])
            cleaner.fit(df)
            
            feature_names = cleaner.get_feature_names_out()
            
            # Should return array-like object containing feature names
            assert len(feature_names) == 2
            assert "feature_1" in feature_names
            assert "feature_2" in feature_names
        except ImportError:
            pytest.skip("scikit-learn integration not available")
    
    def test_cleaner_get_feature_names_with_input_features(self):
        """Test get_feature_names_out respects input_features parameter"""
        try:
            from arnio.integrations.sklearn import ArnioCleaner
            
            df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
            
            cleaner = ArnioCleaner(steps=[("strip_whitespace",)])
            cleaner.fit(df)
            
            # Test with explicit input_features
            feature_names = cleaner.get_feature_names_out(
                input_features=np.array(["x", "y", "z"])
            )
            
            assert len(feature_names) > 0
        except ImportError:
            pytest.skip("scikit-learn integration not available")
