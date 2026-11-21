import pytest
import duckdb
import pandas as pd
import numpy as np
import arnio as ar


class TestRegisterDuckDB:
    """Tests for arnio.integrations.duckdb.register_duckdb function."""
    
    @pytest.fixture
    def duckdb_conn(self):
        """Provide a fresh in-memory DuckDB connection."""
        return duckdb.connect(':memory:')
    
    @pytest.fixture
    def sample_arframe(self):
        """Provide a simple ArFrame for testing."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        return ar.read_pandas(df)
    
    def test_successful_registration(self, duckdb_conn, sample_arframe):
        """Test successful registration and querying of an ArFrame."""
        ar.register_duckdb(sample_arframe, duckdb_conn, "users")
        
        result = duckdb_conn.execute("SELECT * FROM users").fetchdf()
        
        assert len(result) == 3
        assert list(result.columns) == ['id', 'name']
        assert result['id'].tolist() == [1, 2, 3]
    
    def test_raises_typeerror_when_frame_not_arframe(self, duckdb_conn):
        """Test that TypeError is raised when frame is not an ArFrame."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.register_duckdb(df, duckdb_conn, "table")
    
    def test_raises_typeerror_when_name_is_int(self, duckdb_conn, sample_arframe):
        """Test that TypeError is raised when name is an integer."""
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(sample_arframe, duckdb_conn, 123)
    
    def test_raises_typeerror_when_name_is_list(self, duckdb_conn, sample_arframe):
        """Test that TypeError is raised when name is a list."""
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(sample_arframe, duckdb_conn, ["table"])
    
    def test_raises_valueerror_when_name_empty(self, duckdb_conn, sample_arframe):
        """Test that ValueError is raised when name is an empty string."""
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(sample_arframe, duckdb_conn, "")
    
    def test_raises_valueerror_when_name_spaces_only(self, duckdb_conn, sample_arframe):
        """Test that ValueError is raised when name contains only spaces."""
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(sample_arframe, duckdb_conn, "   ")
    
    def test_raises_valueerror_when_name_whitespace_only(self, duckdb_conn, sample_arframe):
        """Test that ValueError is raised for whitespace-only names."""
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(sample_arframe, duckdb_conn, "\t\n")
    
    def test_raises_typeerror_when_conn_no_register(self, sample_arframe):
        """Test that TypeError is raised when conn lacks register method."""
        with pytest.raises(TypeError, match="conn must be a DuckDB connection"):
            ar.register_duckdb(sample_arframe, object(), "table")
    
    def test_raises_typeerror_when_register_not_callable(self, sample_arframe):
        """Test that TypeError is raised when register is not callable."""
        class MockConn:
            register = "not_a_method"
        
        with pytest.raises(TypeError, match="conn must be a DuckDB connection"):
            ar.register_duckdb(sample_arframe, MockConn(), "table")
    
    def test_registers_frame_with_mixed_dtypes(self, duckdb_conn):
        """Test registration of frame with various data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        frame = ar.read_pandas(df)
        
        ar.register_duckdb(frame, duckdb_conn, "mixed_data")
        result = duckdb_conn.execute("SELECT * FROM mixed_data").fetchdf()
        
        assert len(result) == 3
        assert len(result.columns) == 4
    
    def test_registers_frame_with_nan_and_none(self, duckdb_conn):
        """Test registration of frame with NaN and None values."""
        df = pd.DataFrame({
            'floats': [1.0, np.nan, 3.0],
            'strings': ['a', None, 'c']
        })
        frame = ar.read_pandas(df)
        
        ar.register_duckdb(frame, duckdb_conn, "nulls_data")
        result = duckdb_conn.execute("SELECT COUNT(*) as cnt FROM nulls_data").fetchone()
        
        assert result[0] == 3
    
    def test_registers_empty_arframe(self, duckdb_conn):
        """Test registration of an empty ArFrame."""
        df = pd.DataFrame({'col1': [], 'col2': []})
        frame = ar.read_pandas(df)
        
        ar.register_duckdb(frame, duckdb_conn, "empty_data")
        result = duckdb_conn.execute("SELECT * FROM empty_data").fetchdf()
        
        assert len(result) == 0
        assert list(result.columns) == ['col1', 'col2']
    
    def test_overwrites_existing_table(self, duckdb_conn):
        """Test that registering with same name overwrites previous table."""
        df1 = pd.DataFrame({'x': [1, 2]})
        frame1 = ar.read_pandas(df1)
        ar.register_duckdb(frame1, duckdb_conn, "mydata")
        
        df2 = pd.DataFrame({'y': [10, 20, 30]})
        frame2 = ar.read_pandas(df2)
        ar.register_duckdb(frame2, duckdb_conn, "mydata")
        
        result = duckdb_conn.execute("SELECT * FROM mydata").fetchdf()
        assert len(result) == 3
        assert list(result.columns) == ['y']
