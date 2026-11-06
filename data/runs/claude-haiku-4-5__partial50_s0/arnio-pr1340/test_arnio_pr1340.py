import pytest
import pandas as pd
import tempfile
import os
import arnio as ar
from pathlib import Path


class TestReadCsv:
    """Tests for arnio.io.read_csv"""
    
    def test_read_csv_basic(self):
        """Test basic CSV reading"""
        csv_content = "name,age\nAlice,30\nBob,25"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            frame = ar.read_csv(f.name)
            assert frame.shape == (2, 2)
            assert frame.columns == ['name', 'age']
        os.unlink(f.name)
    
    def test_read_csv_with_usecols(self):
        """Test reading only selected columns"""
        csv_content = "id,name,age,city\n1,Alice,30,NYC\n2,Bob,25,LA"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            frame = ar.read_csv(f.name, usecols=['name', 'age'])
            assert frame.shape == (2, 2)
            assert frame.columns == ['name', 'age']
        os.unlink(f.name)
    
    def test_read_csv_with_nrows(self):
        """Test limiting number of rows read"""
        csv_content = "id,value\n1,10\n2,20\n3,30\n4,40"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            frame = ar.read_csv(f.name, nrows=2)
            assert frame.shape == (2, 2)
        os.unlink(f.name)
    
    def test_read_csv_with_skiprows(self):
        """Test skipping initial rows"""
        csv_content = "skip\nid,value\n1,10\n2,20"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            frame = ar.read_csv(f.name, skiprows=1)
            assert frame.shape == (2, 2)
            assert frame.columns == ['id', 'value']
        os.unlink(f.name)
    
    def test_read_csv_custom_delimiter(self):
        """Test reading with custom delimiter"""
        csv_content = "name;age;city\nAlice;30;NYC\nBob;25;LA"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            frame = ar.read_csv(f.name, delimiter=';')
            assert frame.shape == (2, 3)
            assert frame.columns == ['name', 'age', 'city']
        os.unlink(f.name)
    
    def test_read_csv_null_values_default_empty(self):
        """Test that empty cells are treated as null by default"""
        csv_content = "name,age\nAlice,30\nBob,"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            frame = ar.read_csv(f.name)
            df = ar.to_pandas(frame)
            assert pd.isna(df.iloc[1, 1])
        os.unlink(f.name)
    
    def test_read_csv_custom_null_values(self):
        """Test custom null value sentinels (case-insensitive)"""
        csv_content = "name,age\nAlice,30\nBob,MISSING\nCharlie,25"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            frame = ar.read_csv(f.name, null_values=["", "MISSING"])
            df = ar.to_pandas(frame)
            assert pd.isna(df.iloc[1, 1])
        os.unlink(f.name)
    
    def test_read_csv_mode_strict_fails_inconsistent_width(self):
        """Test that strict mode fails on inconsistent row widths"""
        csv_content = "name,age\nAlice,30\nBob"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            with pytest.raises(ar.CsvReadError):
                ar.read_csv(f.name, mode="strict")
        os.unlink(f.name)
    
    def test_read_csv_mode_permissive_fills_nulls(self):
        """Test that permissive mode fills missing trailing values with nulls"""
        csv_content = "name,age,city\nAlice,30,NYC\nBob,25"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            frame = ar.read_csv(f.name, mode="permissive")
            df = ar.to_pandas(frame)
            assert pd.isna(df.iloc[1, 2])
        os.unlink(f.name)
    
    def test_read_csv_file_not_found_raises_csvreaderror(self):
        """Test that missing file raises CsvReadError with path message"""
        with pytest.raises(ar.CsvReadError):
            ar.read_csv("/nonexistent/path/file.csv")
    
    def test_read_csv_pathlib_path_accepted(self):
        """Test that pathlib.Path is accepted as input"""
        csv_content = "id,value\n1,10"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            path = Path(f.name)
        
        frame = ar.read_csv(path)
        assert frame.shape == (1, 2)
        os.unlink(path)


class TestReadCsvChunked:
    """Tests for arnio.io.read_csv_chunked"""
    
    def test_read_csv_chunked_basic(self):
        """Test basic chunked reading returns correct chunk sizes"""
        csv_content = "id,value\n" + "\n".join(f"{i},{i*10}" for i in range(1, 26))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            chunks = list(ar.read_csv_chunked(f.name, chunksize=5))
            assert len(chunks) == 5
            assert all(chunk.shape == (5, 2) for chunk in chunks)
        os.unlink(f.name)
    
    def test_read_csv_chunked_uneven_remainder(self):
        """Test chunked reading with remainder rows less than chunksize"""
        csv_content = "id,value\n" + "\n".join(f"{i},{i*10}" for i in range(1, 7))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            chunks = list(ar.read_csv_chunked(f.name, chunksize=2))
            assert len(chunks) == 3
            assert chunks[0].shape[0] == 2
            assert chunks[2].shape[0] == 2
        os.unlink(f.name)
    
    def test_read_csv_chunked_with_usecols(self):
        """Test chunked reading respects column selection"""
        csv_content = "a,b,c,d\n" + "\n".join(f"{i},{i*2},{i*3},{i*4}" for i in range(1, 11))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            chunks = list(ar.read_csv_chunked(f.name, chunksize=3, usecols=['a', 'c']))
            assert all(chunk.columns == ['a', 'c'] for chunk in chunks)
        os.unlink(f.name)
    
    def test_read_csv_chunked_is_lazy_iterator(self):
        """Test that read_csv_chunked returns a lazy iterator"""
        csv_content = "id,value\n1,10\n2,20\n3,30"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            iterator = ar.read_csv_chunked(f.name, chunksize=2)
            chunk1 = next(iterator)
            assert chunk1.shape == (2, 2)
            chunk2 = next(iterator)
            assert chunk2.shape == (1, 2)
        os.unlink(f.name)
    
    def test_read_csv_chunked_with_nrows_limit(self):
        """Test chunked reading respects nrows parameter"""
        csv_content = "id,value\n" + "\n".join(f"{i},{i*10}" for i in range(1, 21))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            chunks = list(ar.read_csv_chunked(f.name, chunksize=5, nrows=12))
            total_rows = sum(chunk.shape[0] for chunk in chunks)
            assert total_rows == 12
        os.unlink(f.name)
    
    def test_read_csv_chunked_with_skiprows(self):
        """Test chunked reading skips initial rows consistently"""
        csv_content = "skip\nid,value\n1,10\n2,20\n3,30"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            chunks = list(ar.read_csv_chunked(f.name, chunksize=2, skiprows=1))
            assert chunks[0].columns == ['id', 'value']
            total_rows = sum(chunk.shape[0] for chunk in chunks)
            assert total_rows == 3
        os.unlink(f.name)


class TestArnioCleaner:
    """Tests for arnio.integrations.sklearn.ArnioCleaner.get_feature_names_out"""
    
    def test_get_feature_names_out_preserves_schema(self):
        """Test get_feature_names_out returns original column names for schema-preserving steps"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [30, 25, 35]
        })
        
        cleaner = ar.integrations.sklearn.ArnioCleaner(
            steps=[("strip_whitespace",), ("fill_nulls", {"value": 0})]
        )
        
        cleaner.fit(df)
        feature_names = cleaner.get_feature_names_out()
        assert list(feature_names) == ['id', 'name', 'age']
    
    def test_get_feature_names_out_accepts_input_features_parameter(self):
        """Test get_feature_names_out accepts input_features parameter"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
        })
        
        cleaner = ar.integrations.sklearn.ArnioCleaner(
            steps=[("strip_whitespace",)]
        )
        cleaner.fit(df)
        
        feature_names = cleaner.get_feature_names_out(input_features=['col1', 'col2'])
        assert len(feature_names) == 2
    
    def test_get_feature_names_out_after_rename_with_schema_changes_enabled(self):
        """Test get_feature_names_out reflects column renames when allow_schema_changes=True"""
        df = pd.DataFrame({
            'old_col': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        cleaner = ar.integrations.sklearn.ArnioCleaner(
            steps=[("rename_columns", {"old_col": "new_col"})],
            allow_schema_changes=True
        )
        
        cleaner.fit(df)
        feature_names = list(cleaner.get_feature_names_out())
        assert 'new_col' in feature_names
        assert 'old_col' not in feature_names
    
    def test_get_feature_names_out_column_order_preserved(self):
        """Test get_feature_names_out preserves column order from fitted data"""
        df = pd.DataFrame({
            'z_col': [1, 2, 3],
            'a_col': ['x', 'y', 'z'],
            'm_col': [1.1, 2.2, 3.3]
        })
        
        cleaner = ar.integrations.sklearn.ArnioCleaner(
            steps=[("strip_whitespace",)]
        )
        cleaner.fit(df)
        
        feature_names = list(cleaner.get_feature_names_out())
        assert feature_names == ['z_col', 'a_col', 'm_col']
