import pytest
import pandas as pd
import duckdb
import arnio as ar
from unittest.mock import Mock


class TestRegisterDuckdb:
    """Tests for arnio.integrations.duckdb.register_duckdb"""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing"""
        return ar.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})

    @pytest.fixture
    def duckdb_connection(self):
        """Create an in-memory DuckDB connection"""
        return duckdb.connect(":memory:")

    def test_register_success(self, sample_frame, duckdb_connection):
        """Test successful registration of an ArFrame"""
        ar.register_duckdb(sample_frame, duckdb_connection, "test_table")
        result = duckdb_connection.execute("SELECT * FROM test_table").fetchdf()
        assert len(result) == 2
        assert set(result.columns) == {"name", "age"}

    def test_frame_not_arframe_dict(self, duckdb_connection):
        """Test TypeError when frame is a dict"""
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.register_duckdb({}, duckdb_connection, "test")

    def test_frame_not_arframe_pandas(self, duckdb_connection):
        """Test TypeError when frame is a pandas DataFrame"""
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.register_duckdb(pd.DataFrame(), duckdb_connection, "test")

    def test_name_not_string_int(self, sample_frame, duckdb_connection):
        """Test TypeError when name is an int"""
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(sample_frame, duckdb_connection, 123)

    def test_name_not_string_list(self, sample_frame, duckdb_connection):
        """Test TypeError when name is a list"""
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(sample_frame, duckdb_connection, [])

    def test_name_empty_string(self, sample_frame, duckdb_connection):
        """Test ValueError when name is empty string"""
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(sample_frame, duckdb_connection, "")

    def test_name_whitespace_only(self, sample_frame, duckdb_connection):
        """Test ValueError when name is whitespace-only"""
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(sample_frame, duckdb_connection, "   \t\n")

    def test_conn_no_register_method(self, sample_frame):
        """Test TypeError when conn has no register method"""
        mock_conn = Mock(spec=[])
        with pytest.raises(TypeError, match="conn must be a DuckDB connection"):
            ar.register_duckdb(sample_frame, mock_conn, "test")

    def test_conn_register_not_callable(self, sample_frame):
        """Test TypeError when register is not callable"""
        mock_conn = Mock()
        mock_conn.register = "not_callable"
        with pytest.raises(TypeError, match="conn must be a DuckDB connection"):
            ar.register_duckdb(sample_frame, mock_conn, "test")

    def test_data_integrity(self, duckdb_connection):
        """Test that data is preserved after registration"""
        frame = ar.from_dict({
            "name": ["Alice", "Bob"],
            "age": [25, 30],
            "score": [95.5, 87.3]
        })
        ar.register_duckdb(frame, duckdb_connection, "data_test")
        result = duckdb_connection.execute("SELECT * FROM data_test ORDER BY name").fetchdf()
        
        assert list(result["name"]) == ["Alice", "Bob"]
        assert list(result["age"]) == [25, 30]
        assert list(result["score"]) == [95.5, 87.3]

    def test_multiple_registrations(self, duckdb_connection):
        """Test registering multiple frames with different names"""
        frame1 = ar.from_dict({"id": [1, 2]})
        frame2 = ar.from_dict({"id": [10, 20]})
        
        ar.register_duckdb(frame1, duckdb_connection, "table1")
        ar.register_duckdb(frame2, duckdb_connection, "table2")
        
        result1 = duckdb_connection.execute("SELECT * FROM table1").fetchdf()
        result2 = duckdb_connection.execute("SELECT * FROM table2").fetchdf()
        
        assert list(result1["id"]) == [1, 2]
        assert list(result2["id"]) == [10, 20]

    def test_empty_frame(self, duckdb_connection):
        """Test registering an empty ArFrame"""
        frame = ar.from_dict({"col1": [], "col2": []})
        ar.register_duckdb(frame, duckdb_connection, "empty_table")
        result = duckdb_connection.execute("SELECT * FROM empty_table").fetchdf()
        
        assert len(result) == 0
        assert list(result.columns) == ["col1", "col2"]

    def test_mixed_data_types(self, duckdb_connection):
        """Test registration with mixed data types"""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"]
        })
        ar.register_duckdb(frame, duckdb_connection, "mixed_types")
        result = duckdb_connection.execute("SELECT * FROM mixed_types").fetchdf()
        
        assert len(result) == 3
        assert set(result.columns) == {"int_col", "float_col", "str_col"}

    def test_register_same_frame_multiple_names(self, duckdb_connection):
        """Test registering the same frame with different names"""
        frame = ar.from_dict({"id": [1, 2, 3]})
        
        ar.register_duckdb(frame, duckdb_connection, "first")
        ar.register_duckdb(frame, duckdb_connection, "second")
        
        result1 = duckdb_connection.execute("SELECT COUNT(*) as cnt FROM first").fetchdf()
        result2 = duckdb_connection.execute("SELECT COUNT(*) as cnt FROM second").fetchdf()
        
        assert result1["cnt"][0] == 3
        assert result2["cnt"][0] == 3
