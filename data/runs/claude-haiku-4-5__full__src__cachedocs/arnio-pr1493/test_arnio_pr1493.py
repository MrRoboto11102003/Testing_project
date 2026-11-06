"""Tests for arnio.integrations.duckdb.register_duckdb"""

import pytest
import duckdb
import pandas as pd
import arnio as ar


class TestRegisterDuckDB:
    """Test suite for ar.register_duckdb()"""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a simple test ArFrame"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1]
        })
        return ar.from_pandas(df)
    
    @pytest.fixture
    def duckdb_conn(self):
        """Create an in-memory DuckDB connection"""
        return duckdb.connect(":memory:")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Basic functionality tests
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_register_duckdb_basic_registration(self, sample_frame, duckdb_conn):
        """Test basic registration of an ArFrame as a DuckDB relation"""
        ar.register_duckdb(sample_frame, duckdb_conn, "test_table")
        result = duckdb_conn.execute("SELECT * FROM test_table").fetchdf()
        
        assert len(result) == 3
        assert list(result.columns) == ["id", "name", "score"]
    
    def test_register_duckdb_data_integrity(self, sample_frame, duckdb_conn):
        """Test that data is correctly transferred to DuckDB"""
        ar.register_duckdb(sample_frame, duckdb_conn, "test_table")
        result = duckdb_conn.execute("SELECT * FROM test_table").fetchdf()
        
        assert result["id"].tolist() == [1, 2, 3]
        assert result["name"].tolist() == ["Alice", "Bob", "Charlie"]
    
    def test_register_duckdb_with_sql_query(self, sample_frame, duckdb_conn):
        """Test that registered table can be queried with WHERE clause"""
        ar.register_duckdb(sample_frame, duckdb_conn, "students")
        result = duckdb_conn.execute(
            "SELECT name FROM students WHERE score > 90"
        ).fetchdf()
        
        assert len(result) == 2
        assert set(result["name"].tolist()) == {"Alice", "Charlie"}
    
    def test_register_duckdb_aggregation_query(self, sample_frame, duckdb_conn):
        """Test that aggregate queries work on registered table"""
        ar.register_duckdb(sample_frame, duckdb_conn, "scores")
        result = duckdb_conn.execute(
            "SELECT AVG(score) as avg_score FROM scores"
        ).fetchval()
        
        expected_avg = (95.5 + 87.3 + 92.1) / 3
        assert abs(result - expected_avg) < 0.01
    
    # ─────────────────────────────────────────────────────────────────────────
    # Edge cases
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_register_duckdb_empty_frame(self, duckdb_conn):
        """Test registering an empty ArFrame"""
        df = pd.DataFrame({"col1": [], "col2": []})
        frame = ar.from_pandas(df)
        
        ar.register_duckdb(frame, duckdb_conn, "empty_table")
        result = duckdb_conn.execute("SELECT * FROM empty_table").fetchdf()
        
        assert len(result) == 0
        assert list(result.columns) == ["col1", "col2"]
    
    def test_register_duckdb_single_row(self, duckdb_conn):
        """Test registering a frame with a single row"""
        df = pd.DataFrame({"id": [42], "value": ["single"]})
        frame = ar.from_pandas(df)
        
        ar.register_duckdb(frame, duckdb_conn, "single_row")
        result = duckdb_conn.execute("SELECT * FROM single_row").fetchdf()
        
        assert len(result) == 1
        assert result["id"].iloc[0] == 42
    
    def test_register_duckdb_with_nulls(self, duckdb_conn):
        """Test registering a frame with null values"""
        df = pd.DataFrame({
            "id": [1, 2, None],
            "value": ["a", None, "c"]
        })
        frame = ar.from_pandas(df)
        
        ar.register_duckdb(frame, duckdb_conn, "nulls_table")
        result = duckdb_conn.execute("SELECT * FROM nulls_table").fetchdf()
        
        assert len(result) == 3
        assert pd.isna(result.iloc[2, 0])
        assert pd.isna(result.iloc[1, 1])
    
    def test_register_duckdb_multiple_tables(self, sample_frame, duckdb_conn):
        """Test registering multiple frames with different names"""
        ar.register_duckdb(sample_frame, duckdb_conn, "table1")
        ar.register_duckdb(sample_frame, duckdb_conn, "table2")
        
        result1 = duckdb_conn.execute("SELECT COUNT(*) as cnt FROM table1").fetchval()
        result2 = duckdb_conn.execute("SELECT COUNT(*) as cnt FROM table2").fetchval()
        
        assert result1 == 3
        assert result2 == 3
    
    def test_register_duckdb_various_dtypes(self, duckdb_conn):
        """Test registering a frame with various data types"""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        frame = ar.from_pandas(df)
        
        ar.register_duckdb(frame, duckdb_conn, "dtype_table")
        result = duckdb_conn.execute("SELECT * FROM dtype_table").fetchdf()
        
        assert len(result) == 3
        assert list(result.columns) == ["int_col", "float_col", "str_col", "bool_col"]
    
    def test_register_duckdb_special_characters_in_data(self, duckdb_conn):
        """Test registering a frame with special characters in data"""
        df = pd.DataFrame({
            "text": ["hello", "world's", 'quote"test', "comma,here"]
        })
        frame = ar.from_pandas(df)
        
        ar.register_duckdb(frame, duckdb_conn, "special_text")
        result = duckdb_conn.execute("SELECT * FROM special_text").fetchdf()
        
        assert result["text"].tolist() == ["hello", "world's", 'quote"test', "comma,here"]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Error cases (TypeError/ValueError)
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_register_duckdb_frame_not_arframe(self, duckdb_conn):
        """Test that TypeError is raised when frame is not an ArFrame"""
        df = pd.DataFrame({"col": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.register_duckdb(df, duckdb_conn, "table")
    
    def test_register_duckdb_name_not_string(self, sample_frame, duckdb_conn):
        """Test that TypeError is raised when name is not a string"""
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(sample_frame, duckdb_conn, 123)
    
    def test_register_duckdb_empty_name(self, sample_frame, duckdb_conn):
        """Test that ValueError is raised when name is empty string"""
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(sample_frame, duckdb_conn, "")
    
    def test_register_duckdb_whitespace_name(self, sample_frame, duckdb_conn):
        """Test that ValueError is raised when name is whitespace-only"""
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(sample_frame, duckdb_conn, "   ")
    
    def test_register_duckdb_invalid_connection(self, sample_frame):
        """Test that TypeError is raised when connection lacks register method"""
        invalid_conn = object()
        
        with pytest.raises(TypeError, match="conn must be a DuckDB connection"):
            ar.register_duckdb(sample_frame, invalid_conn, "table")
