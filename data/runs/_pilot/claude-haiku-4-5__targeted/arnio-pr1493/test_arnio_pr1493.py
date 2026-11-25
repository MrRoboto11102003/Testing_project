import pytest
import pandas as pd
import numpy as np
import duckdb

try:
    import arnio as ar
except ImportError:
    pytest.skip("arnio not installed", allow_module_level=True)


class TestRegisterDuckdb:
    """Tests for arnio.integrations.duckdb.register_duckdb"""
    
    @pytest.fixture
    def duckdb_connection(self):
        """Create a DuckDB connection for testing"""
        conn = duckdb.connect(":memory:")
        yield conn
        conn.close()
    
    @pytest.fixture
    def simple_frame(self):
        """Create a simple ArFrame for testing"""
        data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
        return ar.from_dict(data)
    
    def test_register_duckdb_basic(self, simple_frame, duckdb_connection):
        """Test basic registration and querying"""
        ar.register_duckdb(simple_frame, duckdb_connection, "test_table")
        result = duckdb_connection.execute("SELECT * FROM test_table").fetchdf()
        assert len(result) == 3
        assert list(result.columns) == ["name", "age"]
    
    def test_register_duckdb_query_filtering(self, simple_frame, duckdb_connection):
        """Test that registered table returns correct filtered data"""
        ar.register_duckdb(simple_frame, duckdb_connection, "people")
        result = duckdb_connection.execute("SELECT * FROM people WHERE age > 25").fetchdf()
        assert len(result) == 2
    
    def test_register_duckdb_empty_string_name(self, simple_frame, duckdb_connection):
        """Test that empty string name raises ValueError"""
        with pytest.raises(ValueError):
            ar.register_duckdb(simple_frame, duckdb_connection, "")
    
    def test_register_duckdb_whitespace_only_name(self, simple_frame, duckdb_connection):
        """Test that whitespace-only name raises ValueError"""
        with pytest.raises(ValueError):
            ar.register_duckdb(simple_frame, duckdb_connection, "   ")
    
    def test_register_duckdb_whitespace_tabs_newlines(self, simple_frame, duckdb_connection):
        """Test that name with only tabs and newlines raises ValueError"""
        with pytest.raises(ValueError):
            ar.register_duckdb(simple_frame, duckdb_connection, "\t\n")
    
    def test_register_duckdb_non_arframe_pandas_df(self, duckdb_connection):
        """Test that pandas DataFrame raises TypeError"""
        pandas_df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError):
            ar.register_duckdb(pandas_df, duckdb_connection, "invalid")
    
    def test_register_duckdb_non_arframe_dict(self, duckdb_connection):
        """Test that dict raises TypeError"""
        with pytest.raises(TypeError):
            ar.register_duckdb({"a": [1, 2]}, duckdb_connection, "invalid")
    
    def test_register_duckdb_name_not_string_int(self, simple_frame, duckdb_connection):
        """Test that integer name raises TypeError"""
        with pytest.raises(TypeError):
            ar.register_duckdb(simple_frame, duckdb_connection, 123)
    
    def test_register_duckdb_name_not_string_none(self, simple_frame, duckdb_connection):
        """Test that None name raises TypeError"""
        with pytest.raises(TypeError):
            ar.register_duckdb(simple_frame, duckdb_connection, None)
    
    def test_register_duckdb_invalid_connection_type(self, simple_frame):
        """Test that invalid connection type raises TypeError"""
        with pytest.raises(TypeError):
            ar.register_duckdb(simple_frame, "not_a_connection", "table")
    
    def test_register_duckdb_multiple_registrations(self, duckdb_connection):
        """Test registering multiple frames with different names"""
        frame1 = ar.from_dict({"x": [1, 2]})
        frame2 = ar.from_dict({"y": [3, 4]})
        
        ar.register_duckdb(frame1, duckdb_connection, "table1")
        ar.register_duckdb(frame2, duckdb_connection, "table2")
        
        result1 = duckdb_connection.execute("SELECT * FROM table1").fetchdf()
        result2 = duckdb_connection.execute("SELECT * FROM table2").fetchdf()
        
        assert len(result1) == 2
        assert len(result2) == 2
    
    def test_register_duckdb_with_null_values(self, duckdb_connection):
        """Test registering frame with null values"""
        frame = ar.from_dict({"a": [1, None, 3], "b": ["x", "y", None]})
        ar.register_duckdb(frame, duckdb_connection, "nulls_table")
        result = duckdb_connection.execute("SELECT * FROM nulls_table").fetchdf()
        assert len(result) == 3
    
    def test_register_duckdb_empty_frame(self, duckdb_connection):
        """Test registering an empty frame"""
        frame = ar.from_dict({"col1": [], "col2": []})
        ar.register_duckdb(frame, duckdb_connection, "empty_table")
        result = duckdb_connection.execute("SELECT * FROM empty_table").fetchdf()
        assert len(result) == 0
        assert list(result.columns) == ["col1", "col2"]
    
    def test_register_duckdb_complex_query(self, simple_frame, duckdb_connection):
        """Test complex SQL query on registered frame"""
        ar.register_duckdb(simple_frame, duckdb_connection, "test_data")
        result = duckdb_connection.execute(
            "SELECT name, age FROM test_data WHERE age >= 30 ORDER BY age DESC"
        ).fetchdf()
        assert len(result) == 2
        assert result.iloc[0]["age"] == 35
