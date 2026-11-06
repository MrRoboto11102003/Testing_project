import pytest
import pandas as pd
import duckdb
import arnio as ar


class TestRegisterDuckdb:
    """Tests for arnio.integrations.duckdb.register_duckdb function."""
    
    def setup_method(self):
        """Create a fresh in-memory DuckDB connection for each test."""
        self.conn = duckdb.connect(":memory:")
    
    def test_basic_registration_and_query(self):
        """Test that a frame can be registered and queried from DuckDB."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        
        ar.register_duckdb(frame, self.conn, "test_table")
        
        result = self.conn.execute("SELECT * FROM test_table").fetchdf()
        assert len(result) == 2
        assert set(result.columns) == {"name", "age"}
        assert result["age"].tolist() == [25, 30]
    
    def test_type_error_invalid_frame(self):
        """Test TypeError when frame is not an ArFrame."""
        with pytest.raises(TypeError):
            ar.register_duckdb({"invalid": "dict"}, self.conn, "table_name")
    
    def test_type_error_invalid_name_type(self):
        """Test TypeError when name is not a string."""
        frame = ar.from_dict({"col": [1, 2]})
        with pytest.raises(TypeError):
            ar.register_duckdb(frame, self.conn, 123)
    
    def test_value_error_empty_string_name(self):
        """Test ValueError when name is an empty string."""
        frame = ar.from_dict({"col": [1]})
        with pytest.raises(ValueError):
            ar.register_duckdb(frame, self.conn, "")
    
    def test_value_error_whitespace_only_name(self):
        """Test ValueError when name contains only whitespace."""
        frame = ar.from_dict({"col": [1]})
        with pytest.raises(ValueError):
            ar.register_duckdb(frame, self.conn, "   \t\n  ")
    
    def test_multiple_frames_registered(self):
        """Test registering multiple distinct frames with different names."""
        frame1 = ar.from_dict({"x": [1, 2, 3]})
        frame2 = ar.from_dict({"y": [10, 20, 30, 40]})
        
        ar.register_duckdb(frame1, self.conn, "table1")
        ar.register_duckdb(frame2, self.conn, "table2")
        
        result1 = self.conn.execute("SELECT COUNT(*) as cnt FROM table1").fetchdf()
        result2 = self.conn.execute("SELECT COUNT(*) as cnt FROM table2").fetchdf()
        
        assert result1["cnt"][0] == 3
        assert result2["cnt"][0] == 4
    
    def test_preserves_column_names(self):
        """Test that column names are exactly preserved in the registered table."""
        data = {"first_name": [1], "UPPER_CASE": [2], "snake_case_col": [3]}
        frame = ar.from_dict(data)
        
        ar.register_duckdb(frame, self.conn, "cols")
        
        result = self.conn.execute("SELECT * FROM cols").fetchdf()
        assert set(result.columns) == {"first_name", "UPPER_CASE", "snake_case_col"}
    
    def test_mixed_numeric_string_types(self):
        """Test registration with mixed data types (int, float, string)."""
        data = {
            "integers": [1, 2, 3],
            "floats": [1.1, 2.2, 3.3],
            "strings": ["a", "b", "c"],
        }
        frame = ar.from_dict(data)
        
        ar.register_duckdb(frame, self.conn, "mixed_data")
        
        result = self.conn.execute("SELECT * FROM mixed_data").fetchdf()
        assert len(result) == 3
        assert result["integers"].tolist() == [1, 2, 3]
        assert result["strings"].tolist() == ["a", "b", "c"]
    
    def test_aggregation_query_on_registered_frame(self):
        """Test that aggregation queries work correctly on registered frame."""
        data = {"amount": [100, 200, 300]}
        frame = ar.from_dict(data)
        
        ar.register_duckdb(frame, self.conn, "amounts")
        
        result = self.conn.execute("SELECT SUM(amount) as total FROM amounts").fetchdf()
        assert result["total"][0] == 600
    
    def test_where_clause_filtering(self):
        """Test WHERE clause filtering on a registered frame."""
        data = {
            "id": [1, 2, 3, 4, 5],
            "status": ["active", "inactive", "active", "inactive", "active"]
        }
        frame = ar.from_dict(data)
        
        ar.register_duckdb(frame, self.conn, "items")
        
        result = self.conn.execute("SELECT * FROM items WHERE status = 'active'").fetchdf()
        assert len(result) == 3
        assert all(result["status"] == "active")
    
    def test_join_multiple_registered_frames(self):
        """Test joining two registered frames in a SQL query."""
        frame1 = ar.from_dict({"id": [1, 2], "name": ["Alice", "Bob"]})
        frame2 = ar.from_dict({"id": [1, 2], "score": [95, 87]})
        
        ar.register_duckdb(frame1, self.conn, "users")
        ar.register_duckdb(frame2, self.conn, "scores")
        
        result = self.conn.execute(
            "SELECT u.name, s.score FROM users u JOIN scores s ON u.id = s.id"
        ).fetchdf()
        
        assert len(result) == 2
        assert result["name"].tolist() == ["Alice", "Bob"]
        assert result["score"].tolist() == [95, 87]
    
    def test_empty_frame_registration(self):
        """Test registering an empty frame (columns present, no rows)."""
        frame = ar.from_dict({"col1": [], "col2": []})
        
        ar.register_duckdb(frame, self.conn, "empty_table")
        
        result = self.conn.execute("SELECT COUNT(*) as cnt FROM empty_table").fetchdf()
        assert result["cnt"][0] == 0
    
    def test_valid_name_with_numbers_and_underscore(self):
        """Test that table names with underscores and numbers are accepted."""
        frame = ar.from_dict({"value": [1, 2]})
        
        ar.register_duckdb(frame, self.conn, "table_v2_test_123")
        
        result = self.conn.execute("SELECT COUNT(*) as cnt FROM table_v2_test_123").fetchdf()
        assert result["cnt"][0] == 2
    
    def test_reregister_overwrites_existing_table(self):
        """Test that registering with same name overwrites the previous table."""
        frame1 = ar.from_dict({"value": [1, 2, 3]})
        frame2 = ar.from_dict({"value": [10, 20, 30, 40, 50]})
        
        ar.register_duckdb(frame1, self.conn, "mytable")
        result_before = self.conn.execute("SELECT COUNT(*) as cnt FROM mytable").fetchdf()
        
        ar.register_duckdb(frame2, self.conn, "mytable")
        result_after = self.conn.execute("SELECT COUNT(*) as cnt FROM mytable").fetchdf()
        
        assert result_before["cnt"][0] == 3
        assert result_after["cnt"][0] == 5
