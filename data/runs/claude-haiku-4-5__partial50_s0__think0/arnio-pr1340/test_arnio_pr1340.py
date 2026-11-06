import pytest
import pandas as pd
import numpy as np
import io
import tempfile
import os
from pathlib import Path
import arnio as ar


class TestArnioCleaner:
    """Tests for ArnioCleaner.get_feature_names_out"""
    
    def test_get_feature_names_out_returns_column_names(self):
        """get_feature_names_out returns the input feature names"""
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "salary": [50000, 60000, 70000],
            "name": ["Alice", "Bob", "Charlie"]
        })
        frame = ar.from_pandas(df)
        
        cleaner = ar.integrations.sklearn.ArnioCleaner(
            steps=[("strip_whitespace",)]
        )
        cleaner.fit(frame)
        
        feature_names = cleaner.get_feature_names_out()
        assert list(feature_names) == ["age", "salary", "name"]
    
    def test_get_feature_names_out_with_input_features(self):
        """get_feature_names_out respects input_features parameter"""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        frame = ar.from_pandas(df)
        
        cleaner = ar.integrations.sklearn.ArnioCleaner(
            steps=[("strip_whitespace",)]
        )
        cleaner.fit(frame)
        
        custom_features = ["feature_a", "feature_b"]
        result = cleaner.get_feature_names_out(input_features=custom_features)
        assert list(result) == custom_features
    
    def test_get_feature_names_out_single_column(self):
        """get_feature_names_out works with single column frames"""
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        frame = ar.from_pandas(df)
        
        cleaner = ar.integrations.sklearn.ArnioCleaner(steps=[])
        cleaner.fit(frame)
        
        feature_names = cleaner.get_feature_names_out()
        assert list(feature_names) == ["value"]


class TestReadCsv:
    """Tests for ar.read_csv"""
    
    def test_read_csv_basic(self):
        """read_csv loads a basic CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,name,age\n1,Alice,30\n2,Bob,25\n")
            fpath = f.name
        
        try:
            frame = ar.read_csv(fpath)
            assert frame.shape == (2, 3)
            assert frame.columns == ["id", "name", "age"]
        finally:
            os.unlink(fpath)
    
    def test_read_csv_with_usecols(self):
        """read_csv respects usecols parameter"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,name,age,city\n1,Alice,30,NYC\n2,Bob,25,LA\n")
            fpath = f.name
        
        try:
            frame = ar.read_csv(fpath, usecols=["id", "name"])
            assert frame.columns == ["id", "name"]
            assert frame.shape == (2, 2)
        finally:
            os.unlink(fpath)
    
    def test_read_csv_with_nrows(self):
        """read_csv respects nrows parameter"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,name\n1,Alice\n2,Bob\n3,Charlie\n4,Dave\n")
            fpath = f.name
        
        try:
            frame = ar.read_csv(fpath, nrows=2)
            assert frame.shape == (2, 2)
        finally:
            os.unlink(fpath)
    
    def test_read_csv_with_delimiter(self):
        """read_csv accepts custom delimiter"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id;name;age\n1;Alice;30\n2;Bob;25\n")
            fpath = f.name
        
        try:
            frame = ar.read_csv(fpath, delimiter=";")
            assert frame.columns == ["id", "name", "age"]
            assert frame.shape == (2, 3)
        finally:
            os.unlink(fpath)
    
    def test_read_csv_with_null_values(self):
        """read_csv treats specified strings as null"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,name,age\n1,Alice,NA\n2,MISSING,25\n")
            fpath = f.name
        
        try:
            frame = ar.read_csv(fpath, null_values=["NA", "MISSING"])
            df = ar.to_pandas(frame)
            assert pd.isna(df.iloc[0, 2]) or str(df.iloc[0, 2]).upper() in ["NA", "NAN"]
        finally:
            os.unlink(fpath)
    
    def test_read_csv_pathlike(self):
        """read_csv accepts Path objects"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,name\n1,Alice\n2,Bob\n")
            fpath = Path(f.name)
        
        try:
            frame = ar.read_csv(fpath)
            assert frame.shape == (2, 2)
        finally:
            os.unlink(fpath)
    
    def test_read_csv_stringio(self):
        """read_csv accepts StringIO objects"""
        csv_data = "id,name,age\n1,Alice,30\n2,Bob,25\n"
        sio = io.StringIO(csv_data)
        frame = ar.read_csv(sio)
        assert frame.shape == (2, 3)
    
    def test_read_csv_strict_mode_default(self):
        """read_csv defaults to strict mode"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,name,age\n1,Alice,30\n2,Bob\n")
            fpath = f.name
        
        try:
            with pytest.raises(ar.exceptions.CsvReadError):
                ar.read_csv(fpath, mode="strict")
        finally:
            os.unlink(fpath)
    
    def test_read_csv_permissive_mode(self):
        """read_csv permissive mode fills missing values with nulls"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,name,age\n1,Alice,30\n2,Bob\n")
            fpath = f.name
        
        try:
            frame = ar.read_csv(fpath, mode="permissive")
            assert frame.shape[0] == 2
        finally:
            os.unlink(fpath)


class TestReadCsvChunked:
    """Tests for ar.read_csv_chunked"""
    
    def test_read_csv_chunked_basic(self):
        """read_csv_chunked yields ArFrame chunks"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,name,value\n")
            for i in range(100):
                f.write(f"{i},name_{i},{i*10}\n")
            fpath = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(fpath, chunksize=30))
            assert len(chunks) == 4  # 100 rows / 30 = 3.33, so 4 chunks
            assert all(isinstance(c, ar.ArFrame) for c in chunks)
        finally:
            os.unlink(fpath)
    
    def test_read_csv_chunked_chunksize(self):
        """read_csv_chunked respects chunksize parameter"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,value\n")
            for i in range(50):
                f.write(f"{i},{i*2}\n")
            fpath = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(fpath, chunksize=20))
            assert chunks[0].shape[0] == 20
            assert chunks[1].shape[0] == 20
            assert chunks[2].shape[0] == 10
        finally:
            os.unlink(fpath)
    
    def test_read_csv_chunked_with_usecols(self):
        """read_csv_chunked respects usecols parameter"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,name,age,city\n1,Alice,30,NYC\n2,Bob,25,LA\n3,Charlie,35,SF\n")
            fpath = f.name
        
        try:
            for chunk in ar.read_csv_chunked(fpath, usecols=["id", "name"], chunksize=2):
                assert chunk.columns == ["id", "name"]
        finally:
            os.unlink(fpath)
    
    def test_read_csv_chunked_with_nrows(self):
        """read_csv_chunked respects nrows parameter"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,value\n")
            for i in range(100):
                f.write(f"{i},{i*2}\n")
            fpath = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(fpath, nrows=25, chunksize=10))
            total_rows = sum(c.shape[0] for c in chunks)
            assert total_rows == 25
        finally:
            os.unlink(fpath)
    
    def test_read_csv_chunked_is_iterator(self):
        """read_csv_chunked returns an iterator"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id\n1\n2\n3\n")
            fpath = f.name
        
        try:
            result = ar.read_csv_chunked(fpath, chunksize=1)
            assert hasattr(result, '__iter__')
            first_chunk = next(result)
            assert isinstance(first_chunk, ar.ArFrame)
        finally:
            os.unlink(fpath)
    
    def test_read_csv_chunked_with_delimiter(self):
        """read_csv_chunked accepts custom delimiter"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id;value;name\n1;10;Alice\n2;20;Bob\n3;30;Charlie\n")
            fpath = f.name
        
        try:
            chunks = list(ar.read_csv_chunked(fpath, delimiter=";", chunksize=2))
            assert chunks[0].columns == ["id", "value", "name"]
        finally:
            os.unlink(fpath)
    
    def test_read_csv_chunked_stringio(self):
        """read_csv_chunked accepts StringIO objects"""
        csv_data = "id,value\n" + "\n".join(f"{i},{i*2}" for i in range(25))
        sio = io.StringIO(csv_data)
        chunks = list(ar.read_csv_chunked(sio, chunksize=10))
        assert len(chunks) == 3
        assert chunks[0].shape[0] == 10
    
    def test_read_csv_chunked_lazy_evaluation(self):
        """read_csv_chunked is lazy and doesn't load entire file upfront"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("id,value\n")
            for i in range(1000):
                f.write(f"{i},{i*2}\n")
            fpath = f.name
        
        try:
            chunked_iter = ar.read_csv_chunked(fpath, chunksize=100)
            first_chunk = next(chunked_iter)
            assert first_chunk.shape[0] == 100
        finally:
            os.unlink(fpath)
