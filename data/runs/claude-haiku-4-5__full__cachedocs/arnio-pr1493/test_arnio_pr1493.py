import pytest
import pandas as pd
import numpy as np
import arnio as ar

# Skip tests if duckdb is not installed
duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def sample_frame():
    """Create a sample ArFrame for testing."""
    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [95.5, 87.3, 92.1, 88.5, 91.2],
        "active": [True, False, True, True, False],
    }
    return ar.from_pandas(pd.DataFrame(data))


@pytest.fixture
def duckdb_connection():
    """Create an in-memory DuckDB connection."""
    return duckdb.connect(":memory:")


class TestRegisterDuckdb:
    """Test suite for ar.register_duckdb()"""

    def test_basic_registration_and_query(self, sample_frame, duckdb_connection):
        """Test that a frame can be registered and queried."""
        ar.register_duckdb(sample_frame, duckdb_connection, "test_table")
        result = duckdb_connection.execute("SELECT * FROM test_table").fetchdf()

        assert len(result) == 5
        assert list(result.columns) == ["id", "name", "score", "active"]
        assert result["id"].tolist() == [1, 2, 3, 4, 5]

    def test_query_with_where_clause(self, sample_frame, duckdb_connection):
        """Test filtering with WHERE clause on registered table."""
        ar.register_duckdb(sample_frame, duckdb_connection, "users")
        result = duckdb_connection.execute(
            "SELECT * FROM users WHERE score > 90"
        ).fetchdf()

        assert len(result) == 4
        assert all(result["score"] > 90)

    def test_aggregate_functions(self, sample_frame, duckdb_connection):
        """Test aggregate SQL functions on registered table."""
        ar.register_duckdb(sample_frame, duckdb_connection, "scores")
        result = duckdb_connection.execute(
            "SELECT COUNT(*) as count, AVG(score) as avg_score FROM scores"
        ).fetchdf()

        assert result["count"][0] == 5
        assert 90 < result["avg_score"][0] < 91

    def test_multiple_table_registration(self, duckdb_connection):
        """Test registering multiple frames as different tables."""
        df1 = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
        df2 = ar.from_pandas(pd.DataFrame({"y": [4, 5, 6]}))

        ar.register_duckdb(df1, duckdb_connection, "table_a")
        ar.register_duckdb(df2, duckdb_connection, "table_b")

        result_a = duckdb_connection.execute("SELECT COUNT(*) as cnt FROM table_a").fetchdf()
        result_b = duckdb_connection.execute("SELECT COUNT(*) as cnt FROM table_b").fetchdf()

        assert result_a["cnt"][0] == 3
        assert result_b["cnt"][0] == 3

    def test_join_two_tables(self, duckdb_connection):
        """Test joining two registered tables."""
        df1 = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]}))
        df2 = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3], "value": [100, 200, 300]}))

        ar.register_duckdb(df1, duckdb_connection, "left_tbl")
        ar.register_duckdb(df2, duckdb_connection, "right_tbl")

        result = duckdb_connection.execute(
            "SELECT l.name, r.value FROM left_tbl l JOIN right_tbl r ON l.id = r.id"
        ).fetchdf()

        assert len(result) == 3
        assert result["value"].tolist() == [100, 200, 300]

    def test_overwrite_existing_table(self, duckdb_connection):
        """Test that registering same name overwrites previous table."""
        df1 = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        df2 = ar.from_pandas(pd.DataFrame({"a": [3, 4, 5]}))

        ar.register_duckdb(df1, duckdb_connection, "data")
        ar.register_duckdb(df2, duckdb_connection, "data")

        result = duckdb_connection.execute("SELECT COUNT(*) as cnt FROM data").fetchdf()
        assert result["cnt"][0] == 3

    def test_type_preservation(self, duckdb_connection):
        """Test that column types are preserved through registration."""
        df = pd.DataFrame({
            "int_col": np.array([1, 2, 3], dtype=np.int64),
            "float_col": np.array([1.5, 2.5, 3.5], dtype=np.float64),
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })
        frame = ar.from_pandas(df)
        ar.register_duckdb(frame, duckdb_connection, "types")

        result = duckdb_connection.execute("SELECT * FROM types").fetchdf()

        assert np.issubdtype(result["int_col"].dtype, np.integer)
        assert np.issubdtype(result["float_col"].dtype, np.floating)
        assert result["str_col"].dtype == object
        assert result["bool_col"].dtype == bool

    def test_null_values_handling(self, duckdb_connection):
        """Test that NULL values are handled correctly."""
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, np.nan, 30.0]})
        frame = ar.from_pandas(df)
        ar.register_duckdb(frame, duckdb_connection, "nulls")

        result = duckdb_connection.execute(
            "SELECT COUNT(*) as null_count FROM nulls WHERE value IS NULL"
        ).fetchdf()

        assert result["null_count"][0] == 1

    def test_large_dataset(self, duckdb_connection):
        """Test registration of a larger dataset."""
        df = pd.DataFrame({"id": range(5000), "value": np.random.randn(5000)})
        frame = ar.from_pandas(df)
        ar.register_duckdb(frame, duckdb_connection, "big_data")

        result = duckdb_connection.execute("SELECT COUNT(*) as cnt FROM big_data").fetchdf()

        assert result["cnt"][0] == 5000

    def test_empty_dataframe(self, duckdb_connection):
        """Test registration of an empty DataFrame."""
        df = pd.DataFrame({"a": pd.Series([], dtype=int), "b": pd.Series([], dtype=str)})
        frame = ar.from_pandas(df)
        ar.register_duckdb(frame, duckdb_connection, "empty")

        result = duckdb_connection.execute("SELECT COUNT(*) as cnt FROM empty").fetchdf()
        assert result["cnt"][0] == 0

    def test_error_not_arframe(self, duckdb_connection):
        """Test TypeError when frame is not an ArFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(TypeError):
            ar.register_duckdb(df, duckdb_connection, "test")

    def test_error_name_not_string(self, sample_frame, duckdb_connection):
        """Test TypeError when name is not a string."""
        with pytest.raises(TypeError):
            ar.register_duckdb(sample_frame, duckdb_connection, 123)

    def test_error_empty_name(self, sample_frame, duckdb_connection):
        """Test ValueError when name is empty string."""
        with pytest.raises(ValueError):
            ar.register_duckdb(sample_frame, duckdb_connection, "")

    def test_error_whitespace_only_name(self, sample_frame, duckdb_connection):
        """Test ValueError when name is whitespace-only."""
        with pytest.raises(ValueError):
            ar.register_duckdb(sample_frame, duckdb_connection, "   \t\n")

    def test_special_characters_in_table_name(self, sample_frame, duckdb_connection):
        """Test that table names with special characters are handled."""
        ar.register_duckdb(sample_frame, duckdb_connection, "my_table_1")

        result = duckdb_connection.execute(
            "SELECT COUNT(*) as cnt FROM my_table_1"
        ).fetchdf()

        assert result["cnt"][0] == 5
