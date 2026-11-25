import pytest
import pandas as pd
import numpy as np
import duckdb
import arnio as ar


class TestRegisterDuckDB:
    """Tests for arnio.integrations.duckdb.register_duckdb function"""
    
    @pytest.fixture
    def duckdb_conn(self):
        """Create a fresh DuckDB connection for each test"""
        return duckdb.connect(memory=True)
    
    def test_register_basic(self, duckdb_conn):
        """Test basic registration of an ArFrame with a DuckDB connection"""
        df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        frame = ar.from_pandas(df)
        
        ar.integrations.duckdb.register_duckdb(frame, duckdb_conn, 'test_table')
        
        result = duckdb_conn.execute('SELECT COUNT(*) FROM test_table').fetchall()
        assert result[0][0] == 3
    
    def test_register_returns_none(self, duckdb_conn):
        """Test that register_duckdb returns None"""
        df = pd.DataFrame({'x': [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.integrations.duckdb.register_duckdb(frame, duckdb_conn, 'table1')
        assert result is None
    
    def test_data_integrity_after_registration(self, duckdb_conn):
        """Test that data is preserved exactly after registration"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c']
        })
        frame = ar.from_pandas(df)
        ar.integrations.duckdb.register_duckdb(frame, duckdb_conn, 'data')
        
        result = duckdb_conn.execute('SELECT * FROM data ORDER BY int_col').fetchall()
        assert len(result) == 3
        assert result[0] == (1, 1.5, 'a')
        assert result[2] == (3, 3.5, 'c')
    
    def test_register_with_various_dtypes(self, duckdb_conn):
        """Test registration with various numpy/pandas dtypes"""
        df = pd.DataFrame({
            'int64': np.array([100, 200, 300], dtype=np.int64),
            'float32': np.array([1.1, 2.2, 3.3], dtype=np.float32),
            'float64': np.array([10.1, 20.2, 30.3], dtype=np.float64),
            'string': ['x', 'y', 'z'],
            'bool': [True, False, True]
        })
        frame = ar.from_pandas(df)
        ar.integrations.duckdb.register_duckdb(frame, duckdb_conn, 'dtypes_table')
        
        result = duckdb_conn.execute('SELECT COUNT(*) FROM dtypes_table').fetchall()
        assert result[0][0] == 3
    
    def test_register_empty_frame(self, duckdb_conn):
        """Test registration of empty ArFrame"""
        df = pd.DataFrame({'col1': pd.Series([], dtype='int64'), 
                           'col2': pd.Series([], dtype='object')})
        frame = ar.from_pandas(df)
        ar.integrations.duckdb.register_duckdb(frame, duckdb_conn, 'empty')
        
        result = duckdb_conn.execute('SELECT COUNT(*) FROM empty').fetchall()
        assert result[0][0] == 0
    
    def test_register_with_null_values(self, duckdb_conn):
        """Test registration correctly handles pandas NULL/None values"""
        df = pd.DataFrame({
            'id': [1, 2, None, 4],
            'name': ['Alice', None, 'Charlie', 'David']
        })
        frame = ar.from_pandas(df)
        ar.integrations.duckdb.register_duckdb(frame, duckdb_conn, 'nulls')
        
        null_count = duckdb_conn.execute('SELECT COUNT(*) FROM nulls WHERE id IS NULL').fetchall()
        assert null_count[0][0] == 1
        
        not_null_count = duckdb_conn.execute('SELECT COUNT(*) FROM nulls WHERE name IS NOT NULL').fetchall()
        assert not_null_count[0][0] == 3
    
    def test_register_with_nan_values(self, duckdb_conn):
        """Test registration converts pandas NaN to DuckDB NULL"""
        df = pd.DataFrame({
            'x': [1.0, np.nan, 3.0, np.nan],
            'y': [np.nan, 2.0, 3.0, 4.0]
        })
        frame = ar.from_pandas(df)
        ar.integrations.duckdb.register_duckdb(frame, duckdb_conn, 'nans')
        
        result = duckdb_conn.execute('SELECT COUNT(*) FROM nans WHERE x IS NOT NULL').fetchall()
        assert result[0][0] == 2
    
    def test_register_multiple_tables(self, duckdb_conn):
        """Test registering multiple ArFrames with different names"""
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6, 7]})
        
        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        
        ar.integrations.duckdb.register_duckdb(frame1, duckdb_conn, 'table_a')
        ar.integrations.duckdb.register_duckdb(frame2, duckdb_conn, 'table_b')
        
        result_a = duckdb_conn.execute('SELECT COUNT(*) FROM table_a').fetchall()
        result_b = duckdb_conn.execute('SELECT COUNT(*) FROM table_b').fetchall()
        
        assert result_a[0][0] == 3
        assert result_b[0][0] == 4
    
    def test_register_overwrite_existing_name(self, duckdb_conn):
        """Test that re-registering with same name replaces previous table"""
        df1 = pd.DataFrame({'val': [1, 2]})
        df2 = pd.DataFrame({'val': [10, 20, 30, 40, 50]})
        
        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        
        ar.integrations.duckdb.register_duckdb(frame1, duckdb_conn, 'dynamic')
        ar.integrations.duckdb.register_duckdb(frame2, duckdb_conn, 'dynamic')
        
        result = duckdb_conn.execute('SELECT COUNT(*) FROM dynamic').fetchall()
        assert result[0][0] == 5
    
    def test_register_with_datetime_columns(self, duckdb_conn):
        """Test registration with datetime columns"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=3),
            'value': [10, 20, 30]
        })
        frame = ar.from_pandas(df)
        ar.integrations.duckdb.register_duckdb(frame, duckdb_conn, 'dates')
        
        result = duckdb_conn.execute('SELECT COUNT(*) FROM dates').fetchall()
        assert result[0][0] == 3
    
    def test_register_large_dataframe(self, duckdb_conn):
        """Test registration of large ArFrame"""
        n = 5000
        df = pd.DataFrame({
            'id': range(n),
            'value': np.random.randn(n),
            'category': np.random.choice(['A', 'B', 'C'], n)
        })
        frame = ar.from_pandas(df)
        ar.integrations.duckdb.register_duckdb(frame, duckdb_conn, 'large')
        
        result = duckdb_conn.execute('SELECT COUNT(*) FROM large').fetchall()
        assert result[0][0] == n
    
    def test_register_preserves_column_order(self, duckdb_conn):
        """Test that column order is preserved after registration"""
        df = pd.DataFrame({
            'z': [1, 2, 3],
            'a': [4, 5, 6],
            'b': [7, 8, 9]
        })
        frame = ar.from_pandas(df)
        ar.integrations.duckdb.register_duckdb(frame, duckdb_conn, 'cols')
        
        result = duckdb_conn.execute("PRAGMA table_info(cols)").fetchall()
        col_names = [row[1] for row in result]
        
        assert col_names == ['z', 'a', 'b']
    
    def test_register_with_mixed_types_and_nulls(self, duckdb_conn):
        """Test registration with mixed types including nulls and NaNs"""
        df = pd.DataFrame({
            'int_col': [1, None, 3],
            'float_col': [1.1, 2.2, np.nan],
            'str_col': ['a', None, 'c'],
            'bool_col': [True, False, None]
        })
        frame = ar.from_pandas(df)
        ar.integrations.duckdb.register_duckdb(frame, duckdb_conn, 'mixed')
        
        result = duckdb_conn.execute('SELECT COUNT(*) FROM mixed').fetchall()
        assert result[0][0] == 3
        
        null_counts = duckdb_conn.execute(
            'SELECT COUNT(*) FILTER (WHERE int_col IS NULL), '
            '       COUNT(*) FILTER (WHERE float_col IS NULL), '
            '       COUNT(*) FILTER (WHERE str_col IS NULL) FROM mixed'
        ).fetchall()[0]
        assert null_counts[0] == 1
        assert null_counts[1] == 1
        assert null_counts[2] == 1
