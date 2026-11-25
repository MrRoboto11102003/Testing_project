import pytest
import pandas as pd
import numpy as np

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

import arnio as ar


pytestmark = pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")


class TestRegisterDuckDB:
    """Test suite for arnio.register_duckdb function."""

    def test_basic_registration_and_query(self):
        """Test basic registration and querying a registered relation."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        conn = duckdb.connect()

        ar.register_duckdb(frame, conn, "test_table")
        result = conn.execute("SELECT * FROM test_table").fetchdf()

        assert result.shape == (3, 2)
        assert list(result.columns) == ["id", "name"]

    def test_data_integrity_after_registration(self):
        """Test that registered data matches the original frame."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.5, 20.3, 30.1]
        })
        frame = ar.from_pandas(df)
        conn = duckdb.connect()

        ar.register_duckdb(frame, conn, "data")
        result = conn.execute("SELECT * FROM data ORDER BY id").fetchdf()

        assert result["id"].tolist() == [1, 2, 3]
        assert np.allclose(result["value"].tolist(), [10.5, 20.3, 30.1])

    def test_registration_with_null_values(self):
        """Test registration of frame containing null values."""
        frame = ar.from_dict({
            "col1": [1, None, 3],
            "col2": ["a", "b", None]
        })
        conn = duckdb.connect()

        ar.register_duckdb(frame, conn, "nulls_table")
        result = conn.execute("SELECT * FROM nulls_table").fetchdf()

        assert result.shape == (3, 2)
        assert pd.isna(result["col1"].iloc[1])
        assert pd.isna(result["col2"].iloc[2])

    def test_registration_of_empty_frame(self):
        """Test registration of an empty frame with columns."""
        frame = ar.from_dict({"a": [], "b": []})
        conn = duckdb.connect()

        ar.register_duckdb(frame, conn, "empty_table")
        result = conn.execute("SELECT * FROM empty_table").fetchdf()

        assert result.shape == (0, 2)
        assert list(result.columns) == ["a", "b"]

    def test_aggregation_query_on_registered_relation(self):
        """Test that aggregation queries work on registered relations."""
        frame = ar.from_dict({
            "category": ["A", "A", "B", "B"],
            "amount": [10, 20, 30, 40]
        })
        conn = duckdb.connect()

        ar.register_duckdb(frame, conn, "sales")
        result = conn.execute(
            "SELECT category, SUM(amount) as total FROM sales GROUP BY category"
        ).fetchdf()

        assert result.shape[0] == 2
        assert all(col in result.columns for col in ["category", "total"])

    def test_filtering_query_on_registered_relation(self):
        """Test that WHERE clause filtering works on registered relations."""
        frame = ar.from_dict({
            "id": [1, 2, 3, 4],
            "status": ["active", "inactive", "active", "inactive"]
        })
        conn = duckdb.connect()

        ar.register_duckdb(frame, conn, "records")
        result = conn.execute(
            "SELECT * FROM records WHERE status = 'active'"
        ).fetchdf()

        assert result.shape[0] == 2
        assert all(result["status"] == "active")

    def test_registration_with_mixed_dtypes(self):
        """Test registration with various column data types."""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        conn = duckdb.connect()

        ar.register_duckdb(frame, conn, "mixed_types")
        result = conn.execute("SELECT * FROM mixed_types").fetchdf()

        assert result.shape == (3, 4)
        assert all(col in result.columns for col in ["int_col", "float_col", "str_col", "bool_col"])

    def test_type_error_when_frame_not_arframe(self):
        """Test that TypeError is raised when frame is not an ArFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        conn = duckdb.connect()

        with pytest.raises(TypeError):
            ar.register_duckdb(df, conn, "table")

    def test_type_error_when_name_not_string(self):
        """Test that TypeError is raised when name is not a string."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        conn = duckdb.connect()

        with pytest.raises(TypeError):
            ar.register_duckdb(frame, conn, 123)

    def test_value_error_with_empty_string_name(self):
        """Test that ValueError is raised for empty string name."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        conn = duckdb.connect()

        with pytest.raises(ValueError):
            ar.register_duckdb(frame, conn, "")

    def test_value_error_with_whitespace_only_name(self):
        """Test that ValueError is raised for whitespace-only name."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        conn = duckdb.connect()

        with pytest.raises(ValueError):
            ar.register_duckdb(frame, conn, "   ")

    def test_multiple_relations_in_same_connection(self):
        """Test registering multiple relations in the same connection."""
        frame1 = ar.from_dict({"id": [1, 2], "name": ["a", "b"]})
        frame2 = ar.from_dict({"id": [3, 4], "name": ["c", "d"]})
        conn = duckdb.connect()

        ar.register_duckdb(frame1, conn, "table1")
        ar.register_duckdb(frame2, conn, "table2")

        result1 = conn.execute("SELECT * FROM table1").fetchdf()
        result2 = conn.execute("SELECT * FROM table2").fetchdf()

        assert result1.shape == (2, 2)
        assert result2.shape == (2, 2)
        assert result1["name"].tolist() == ["a", "b"]
        assert result2["name"].tolist() == ["c", "d"]

    def test_join_across_registered_relations(self):
        """Test that registered relations can be joined together."""
        users = ar.from_dict({"user_id": [1, 2], "name": ["Alice", "Bob"]})
        orders = ar.from_dict({"order_id": [101, 102], "user_id": [1, 2]})
        conn = duckdb.connect()

        ar.register_duckdb(users, conn, "users")
        ar.register_duckdb(orders, conn, "orders")

        result = conn.execute(
            "SELECT u.name, o.order_id FROM users u JOIN orders o ON u.user_id = o.user_id"
        ).fetchdf()

        assert result.shape == (2, 2)
        assert list(result.columns) == ["name", "order_id"]

    def test_column_selection_on_registered_relation(self):
        """Test that column selection queries work on registered relations."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "value": [10, 20, 30]
        })
        conn = duckdb.connect()

        ar.register_duckdb(frame, conn, "data")
        result = conn.execute("SELECT name, value FROM data").fetchdf()

        assert result.shape == (3, 2)
        assert list(result.columns) == ["name", "value"]
        assert "id" not in result.columns

    def test_ordering_on_registered_relation(self):
        """Test that ORDER BY works on registered relations."""
        frame = ar.from_dict({
            "name": ["Charlie", "Alice", "Bob"],
            "score": [85, 95, 88]
        })
        conn = duckdb.connect()

        ar.register_duckdb(frame, conn, "scores")
        result = conn.execute("SELECT name FROM scores ORDER BY score DESC").fetchdf()

        assert result["name"].tolist() == ["Alice", "Bob", "Charlie"]
