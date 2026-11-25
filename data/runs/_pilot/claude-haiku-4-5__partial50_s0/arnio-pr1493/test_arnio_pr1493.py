import pytest
import pandas as pd
import arnio as ar

pytest.importorskip("duckdb")
import duckdb


class TestRegisterDuckDB:
    """Tests for arnio.integrations.duckdb.register_duckdb."""

    @pytest.fixture
    def conn(self):
        """Create an in-memory DuckDB connection."""
        connection = duckdb.connect(":memory:")
        yield connection
        connection.close()

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame with mixed data types."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "salary": [50000.0, 60000.0, 55000.0]
        })
        return ar.from_pandas(df)

    def test_register_returns_none(self, sample_frame, conn):
        """Test that register_duckdb returns None."""
        result = ar.register_duckdb(sample_frame, conn, "table1")
        assert result is None

    def test_registered_table_is_queryable(self, sample_frame, conn):
        """Test that registered table can be queried with SQL."""
        ar.register_duckdb(sample_frame, conn, "users")
        result = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchall()
        assert result[0][0] == 3

    def test_data_preserved_after_registration(self, sample_frame, conn):
        """Test that data values are correctly preserved."""
        ar.register_duckdb(sample_frame, conn, "users")
        result = conn.execute(
            "SELECT id, name, salary FROM users ORDER BY id"
        ).fetchdf()
        
        assert list(result["id"]) == [1, 2, 3]
        assert list(result["name"]) == ["Alice", "Bob", "Charlie"]
        assert list(result["salary"]) == [50000.0, 60000.0, 55000.0]

    def test_column_names_preserved(self, sample_frame, conn):
        """Test that column names match the original frame."""
        ar.register_duckdb(sample_frame, conn, "test")
        result = conn.execute("SELECT * FROM test LIMIT 1").fetchdf()
        assert list(result.columns) == ["id", "name", "salary"]

    def test_register_empty_frame(self, conn):
        """Test registering an empty ArFrame."""
        df = pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="string")})
        frame = ar.from_pandas(df)
        ar.register_duckdb(frame, conn, "empty")
        
        result = conn.execute("SELECT COUNT(*) as cnt FROM empty").fetchall()
        assert result[0][0] == 0

    def test_frame_with_nulls(self, conn):
        """Test registration handles null values correctly."""
        df = pd.DataFrame({
            "id": [1, 2, None],
            "value": [10, None, 30]
        })
        frame = ar.from_pandas(df)
        ar.register_duckdb(frame, conn, "nulls")
        
        result = conn.execute("SELECT COUNT(*) FROM nulls WHERE id IS NULL").fetchall()
        assert result[0][0] == 1

    def test_multiple_table_registrations(self, conn):
        """Test registering multiple frames as different tables."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"y": [3, 4, 5]})
        
        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        
        ar.register_duckdb(frame1, conn, "t1")
        ar.register_duckdb(frame2, conn, "t2")
        
        r1 = conn.execute("SELECT COUNT(*) as cnt FROM t1").fetchall()
        r2 = conn.execute("SELECT COUNT(*) as cnt FROM t2").fetchall()
        
        assert r1[0][0] == 2
        assert r2[0][0] == 3

    def test_sql_aggregation_on_registered_table(self, sample_frame, conn):
        """Test that SQL aggregations work on registered table."""
        ar.register_duckdb(sample_frame, conn, "employees")
        
        result = conn.execute(
            "SELECT COUNT(*) as cnt, AVG(salary) as avg_sal FROM employees"
        ).fetchall()
        
        assert result[0][0] == 3
        assert abs(result[0][1] - 55000.0) < 1

    def test_column_order_preserved(self, conn):
        """Test that column order is preserved in registered table."""
        df = pd.DataFrame({"z": [1], "a": [2], "m": [3]})
        frame = ar.from_pandas(df)
        ar.register_duckdb(frame, conn, "ordered")
        
        result = conn.execute("SELECT * FROM ordered").fetchdf()
        assert list(result.columns) == ["z", "a", "m"]

    def test_error_frame_not_arframe(self, conn):
        """Test TypeError when frame is not an ArFrame."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        with pytest.raises(TypeError):
            ar.register_duckdb(df, conn, "table")

    def test_error_name_not_string(self, sample_frame, conn):
        """Test TypeError when name is not a string."""
        with pytest.raises(TypeError):
            ar.register_duckdb(sample_frame, conn, 42)

    def test_error_empty_name_string(self, sample_frame, conn):
        """Test ValueError when name is an empty string."""
        with pytest.raises(ValueError):
            ar.register_duckdb(sample_frame, conn, "")

    def test_error_whitespace_only_name(self, sample_frame, conn):
        """Test ValueError when name is whitespace-only."""
        with pytest.raises(ValueError):
            ar.register_duckdb(sample_frame, conn, "   ")

    def test_error_whitespace_variations(self, sample_frame, conn):
        """Test ValueError with tabs and newlines."""
        with pytest.raises(ValueError):
            ar.register_duckdb(sample_frame, conn, "\t")
        with pytest.raises(ValueError):
            ar.register_duckdb(sample_frame, conn, " \n \t ")
