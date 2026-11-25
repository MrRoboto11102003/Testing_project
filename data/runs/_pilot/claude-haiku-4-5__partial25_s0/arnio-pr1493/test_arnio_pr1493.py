import pytest
import duckdb
import pandas as pd
import arnio as ar


class TestRegisterDuckdb:
    """Test suite for ar.register_duckdb integration."""

    @pytest.fixture
    def duckdb_conn(self):
        """Create an in-memory DuckDB connection."""
        return duckdb.connect(":memory:")

    @pytest.fixture
    def simple_frame(self):
        """Create a simple test ArFrame."""
        return ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1],
        })

    @pytest.fixture
    def frame_with_nulls(self):
        """Create an ArFrame with null values."""
        return ar.from_dict({
            "id": [1, 2, None],
            "value": [10.5, None, 30.2],
        })

    @pytest.fixture
    def empty_frame(self):
        """Create an empty ArFrame."""
        return ar.from_dict({"col1": [], "col2": []})

    def test_register_duckdb_basic_registration(self, simple_frame, duckdb_conn):
        """Test basic ArFrame registration as a DuckDB table."""
        ar.register_duckdb(simple_frame, duckdb_conn, "test_table")
        
        result = duckdb_conn.execute("SELECT * FROM test_table").fetchdf()
        assert len(result) == 3
        assert list(result.columns) == ["id", "name", "score"]

    def test_register_duckdb_data_integrity(self, simple_frame, duckdb_conn):
        """Test that registered table data matches the original frame."""
        ar.register_duckdb(simple_frame, duckdb_conn, "my_table")
        
        result_df = duckdb_conn.execute("SELECT * FROM my_table").fetchdf()
        original_df = ar.to_pandas(simple_frame)
        
        pd.testing.assert_frame_equal(result_df, original_df)

    def test_register_duckdb_returns_none(self, simple_frame, duckdb_conn):
        """Test that register_duckdb returns None."""
        result = ar.register_duckdb(simple_frame, duckdb_conn, "table")
        assert result is None

    def test_register_duckdb_invalid_frame_type_pandas_df(self, duckdb_conn):
        """Test that TypeError is raised if frame is a pandas DataFrame."""
        invalid_frame = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(TypeError):
            ar.register_duckdb(invalid_frame, duckdb_conn, "table")

    def test_register_duckdb_invalid_frame_type_dict(self, duckdb_conn):
        """Test that TypeError is raised if frame is a dict."""
        with pytest.raises(TypeError):
            ar.register_duckdb({"a": [1, 2, 3]}, duckdb_conn, "table")

    def test_register_duckdb_invalid_name_type_int(self, simple_frame, duckdb_conn):
        """Test that TypeError is raised if name is not a string."""
        with pytest.raises(TypeError):
            ar.register_duckdb(simple_frame, duckdb_conn, 123)

    def test_register_duckdb_invalid_name_type_none(self, simple_frame, duckdb_conn):
        """Test that TypeError is raised if name is None."""
        with pytest.raises(TypeError):
            ar.register_duckdb(simple_frame, duckdb_conn, None)

    def test_register_duckdb_empty_string_name(self, simple_frame, duckdb_conn):
        """Test that ValueError is raised for empty string name."""
        with pytest.raises(ValueError):
            ar.register_duckdb(simple_frame, duckdb_conn, "")

    def test_register_duckdb_whitespace_only_name(self, simple_frame, duckdb_conn):
        """Test that ValueError is raised for whitespace-only name."""
        with pytest.raises(ValueError):
            ar.register_duckdb(simple_frame, duckdb_conn, "   ")

    def test_register_duckdb_tab_only_name(self, simple_frame, duckdb_conn):
        """Test that ValueError is raised for tab-only name."""
        with pytest.raises(ValueError):
            ar.register_duckdb(simple_frame, duckdb_conn, "\t\t")

    def test_register_duckdb_with_nulls(self, frame_with_nulls, duckdb_conn):
        """Test registration of frame with null values."""
        ar.register_duckdb(frame_with_nulls, duckdb_conn, "nulls_table")
        
        result = duckdb_conn.execute(
            "SELECT COUNT(*) as cnt FROM nulls_table"
        ).fetchdf()
        assert result["cnt"][0] == 3

    def test_register_duckdb_empty_frame(self, empty_frame, duckdb_conn):
        """Test registration of an empty ArFrame."""
        ar.register_duckdb(empty_frame, duckdb_conn, "empty_table")
        
        result = duckdb_conn.execute(
            "SELECT COUNT(*) as cnt FROM empty_table"
        ).fetchdf()
        assert result["cnt"][0] == 0

    def test_register_duckdb_queryable(self, simple_frame, duckdb_conn):
        """Test that registered table can be queried with filters and aggregates."""
        ar.register_duckdb(simple_frame, duckdb_conn, "scores")
        
        # Test WHERE clause
        filtered = duckdb_conn.execute(
            "SELECT name FROM scores WHERE score > 90"
        ).fetchdf()
        assert len(filtered) == 2
        
        # Test aggregate
        agg = duckdb_conn.execute(
            "SELECT AVG(score) as avg_score FROM scores"
        ).fetchdf()
        expected_avg = (95.5 + 87.3 + 92.1) / 3
        assert abs(agg["avg_score"][0] - expected_avg) < 0.01

    def test_register_duckdb_multiple_tables(self, simple_frame, duckdb_conn):
        """Test registering multiple tables with different names."""
        ar.register_duckdb(simple_frame, duckdb_conn, "table1")
        ar.register_duckdb(simple_frame, duckdb_conn, "table2")
        
        count1 = duckdb_conn.execute("SELECT COUNT(*) FROM table1").fetchval()
        count2 = duckdb_conn.execute("SELECT COUNT(*) FROM table2").fetchval()
        
        assert count1 == 3
        assert count2 == 3

    def test_register_duckdb_table_overwrite(self, duckdb_conn):
        """Test that re-registering with same name overwrites previous table."""
        frame1 = ar.from_dict({"id": [1, 2, 3]})
        frame2 = ar.from_dict({"id": [100, 200]})
        
        ar.register_duckdb(frame1, duckdb_conn, "data")
        count1 = duckdb_conn.execute("SELECT COUNT(*) FROM data").fetchval()
        assert count1 == 3
        
        ar.register_duckdb(frame2, duckdb_conn, "data")
        count2 = duckdb_conn.execute("SELECT COUNT(*) FROM data").fetchval()
        assert count2 == 2
