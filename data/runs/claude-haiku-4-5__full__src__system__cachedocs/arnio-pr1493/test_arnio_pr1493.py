import pytest
from pathlib import Path

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

import arnio as ar
import pandas as pd


pytestmark = pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")


class TestRegisterDuckDBIntegration:
    """End-to-end system tests for ar.register_duckdb()"""

    def test_basic_registration_and_query(self, tmp_path):
        """User workflow: load CSV, register with DuckDB, query it"""
        # Setup: create real CSV file
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,Alice,95\n2,Bob,87\n3,Charlie,92\n")
        
        # Workflow: load, register, query
        frame = ar.read_csv(csv_file)
        conn = duckdb.connect(":memory:")
        ar.register_duckdb(frame, conn, "users")
        
        # Verify: check query results
        result = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchdf()
        assert result["cnt"].values[0] == 3
        
        result = conn.execute(
            "SELECT name FROM users WHERE score > 90 ORDER BY name"
        ).fetchdf()
        assert result["name"].tolist() == ["Alice", "Charlie"]

    def test_registration_with_cleaned_data(self, tmp_path):
        """User workflow: load messy CSV, clean with pipeline, register with DuckDB"""
        # Setup: create messy CSV
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("name,city\n  Alice  , Paris \n Bob , London \n")
        
        # Workflow: load, clean, register
        frame = ar.read_csv(csv_file)
        clean_frame = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
        ])
        
        conn = duckdb.connect(":memory:")
        ar.register_duckdb(clean_frame, conn, "clean_users")
        
        # Verify: check cleaned data in DuckDB
        result = conn.execute(
            "SELECT name, city FROM clean_users WHERE name = 'alice'"
        ).fetchdf()
        assert len(result) == 1
        assert result["city"].values[0] == "paris"

    def test_multiple_registrations_and_join(self, tmp_path):
        """User workflow: register multiple frames and perform SQL join"""
        # Setup: create two CSV files
        users_file = tmp_path / "users.csv"
        users_file.write_text("user_id,name\n1,Alice\n2,Bob\n3,Charlie\n")
        
        orders_file = tmp_path / "orders.csv"
        orders_file.write_text("order_id,user_id,amount\n101,1,100\n102,2,200\n103,1,150\n")
        
        # Workflow: load both, register both, join
        users_frame = ar.read_csv(users_file)
        orders_frame = ar.read_csv(orders_file)
        
        conn = duckdb.connect(":memory:")
        ar.register_duckdb(users_frame, conn, "users")
        ar.register_duckdb(orders_frame, conn, "orders")
        
        # Verify: check join result
        result = conn.execute(
            "SELECT u.name, COUNT(*) as num_orders, SUM(o.amount) as total "
            "FROM users u JOIN orders o ON u.user_id = o.user_id "
            "GROUP BY u.name ORDER BY u.name"
        ).fetchdf()
        
        assert len(result) == 2
        alice_total = result[result["name"] == "Alice"]["total"].values[0]
        assert alice_total == 250

    def test_registration_with_filtered_data(self, tmp_path):
        """User workflow: load CSV, filter with pipeline, register subset"""
        # Setup: create CSV
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,status,value\n1,active,100\n2,inactive,200\n3,active,150\n")
        
        # Workflow: load, filter, register
        frame = ar.read_csv(csv_file)
        active_frame = ar.pipeline(frame, [
            ("filter_rows", {"column": "status", "op": "==", "value": "active"}),
        ])
        
        conn = duckdb.connect(":memory:")
        ar.register_duckdb(active_frame, conn, "active_records")
        
        # Verify: check only active records present
        result = conn.execute("SELECT COUNT(*) as cnt FROM active_records").fetchdf()
        assert result["cnt"].values[0] == 2
        
        result = conn.execute("SELECT SUM(value) as total FROM active_records").fetchdf()
        assert result["total"].values[0] == 250

    def test_registration_preserves_data_types(self, tmp_path):
        """User workflow: load CSV with mixed types, verify types in DuckDB"""
        # Setup: create CSV with various types
        csv_file = tmp_path / "types.csv"
        csv_file.write_text("int_col,float_col,text_col\n1,1.5,hello\n2,2.7,world\n")
        
        # Workflow: load, register, query with type operations
        frame = ar.read_csv(csv_file)
        conn = duckdb.connect(":memory:")
        ar.register_duckdb(frame, conn, "typed_data")
        
        # Verify: perform type-specific operations
        result = conn.execute(
            "SELECT int_col * 2 as doubled, float_col + 0.5 as adjusted FROM typed_data"
        ).fetchdf()
        
        assert result["doubled"].tolist() == [2, 4]
        assert result["adjusted"].iloc[0] == pytest.approx(2.0, 0.01)

    def test_null_values_preserved(self, tmp_path):
        """User workflow: load CSV with nulls, verify nulls preserved in DuckDB"""
        # Setup: create CSV with missing values
        csv_file = tmp_path / "nulls.csv"
        csv_file.write_text("id,value\n1,100\n2,\n3,300\n4,\n")
        
        # Workflow: load, register
        frame = ar.read_csv(csv_file)
        conn = duckdb.connect(":memory:")
        ar.register_duckdb(frame, conn, "null_data")
        
        # Verify: check null handling in queries
        result = conn.execute(
            "SELECT COUNT(*) as total, COUNT(value) as non_null FROM null_data"
        ).fetchdf()
        assert result["total"].values[0] == 4
        assert result["non_null"].values[0] == 2

    def test_error_on_non_arframe(self, tmp_path):
        """User error: passing pandas DataFrame instead of ArFrame raises TypeError"""
        conn = duckdb.connect(":memory:")
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.register_duckdb(df, conn, "test_table")

    def test_error_on_non_string_name(self, tmp_path):
        """User error: passing non-string name raises TypeError"""
        # Setup: create and load CSV
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,10\n")
        frame = ar.read_csv(csv_file)
        
        conn = duckdb.connect(":memory:")
        
        # Test: passing integer name
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(frame, conn, 123)
        
        # Test: passing list name
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(frame, conn, ["table"])

    def test_error_on_empty_name(self, tmp_path):
        """User error: passing empty string as name raises ValueError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,10\n")
        frame = ar.read_csv(csv_file)
        
        conn = duckdb.connect(":memory:")
        
        with pytest.raises(ValueError, match="name must not be empty"):
            ar.register_duckdb(frame, conn, "")

    def test_error_on_whitespace_only_name(self, tmp_path):
        """User error: passing whitespace-only name raises ValueError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,10\n")
        frame = ar.read_csv(csv_file)
        
        conn = duckdb.connect(":memory:")
        
        with pytest.raises(ValueError, match="name must not be empty"):
            ar.register_duckdb(frame, conn, "   \t  \n")

    def test_error_on_invalid_connection(self, tmp_path):
        """User error: passing object without register() method raises TypeError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,10\n")
        frame = ar.read_csv(csv_file)
        
        fake_conn = object()  # No register method
        
        with pytest.raises(TypeError, match="DuckDB connection"):
            ar.register_duckdb(frame, fake_conn, "test")

    def test_aggregation_query(self, tmp_path):
        """User workflow: register data and perform aggregation query"""
        # Setup: create sales CSV
        csv_file = tmp_path / "sales.csv"
        csv_file.write_text("product,amount\nApple,100\nBanana,200\nApple,150\nBanana,100\n")
        
        # Workflow: load, register, aggregate
        frame = ar.read_csv(csv_file)
        conn = duckdb.connect(":memory:")
        ar.register_duckdb(frame, conn, "sales")
        
        # Verify: check aggregation results
        result = conn.execute(
            "SELECT product, SUM(amount) as total, AVG(amount) as avg_amount "
            "FROM sales GROUP BY product ORDER BY product"
        ).fetchdf()
        
        assert len(result) == 2
        apple_row = result[result["product"] == "Apple"].iloc[0]
        assert apple_row["total"] == 250
        assert apple_row["avg_amount"] == pytest.approx(125.0)

    def test_complex_pipeline_and_register(self, tmp_path):
        """User workflow: load, clean, filter, register, complex query"""
        # Setup: create messy data
        csv_file = tmp_path / "records.csv"
        csv_file.write_text(
            "id,category,value,status\n"
            "1, A , 100 , active \n"
            "2,B,200,inactive\n"
            "3,  A  ,  150  , active \n"
            "4,B,250,active\n"
        )
        
        # Workflow: clean and filter
        frame = ar.read_csv(csv_file)
        clean_frame = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("filter_rows", {"column": "status", "op": "==", "value": "active"}),
        ])
        
        conn = duckdb.connect(":memory:")
        ar.register_duckdb(clean_frame, conn, "clean_records")
        
        # Verify: complex query on cleaned, filtered data
        result = conn.execute(
            "SELECT category, COUNT(*) as count, AVG(value) as avg_value "
            "FROM clean_records GROUP BY category ORDER BY category"
        ).fetchdf()
        
        assert len(result) == 2
        a_row = result[result["category"] == "A"].iloc[0]
        assert a_row["count"] == 2
        assert a_row["avg_value"] == pytest.approx(125.0)

    def test_data_integrity_round_trip(self, tmp_path):
        """User verification: data integrity through register->query round trip"""
        # Setup: create reference CSV
        csv_file = tmp_path / "reference.csv"
        csv_file.write_text("name,score\nAlice,95\nBob,87\nCharlie,92\n")
        
        # Workflow: load, register, query back
        frame = ar.read_csv(csv_file)
        df_original = ar.to_pandas(frame)
        
        conn = duckdb.connect(":memory:")
        ar.register_duckdb(frame, conn, "reference")
        df_from_duckdb = conn.execute("SELECT * FROM reference ORDER BY name").fetchdf()
        
        # Verify: data matches (order may differ, so sort)
        df_original_sorted = df_original.sort_values("name").reset_index(drop=True)
        df_from_duckdb_sorted = df_from_duckdb.sort_values("name").reset_index(drop=True)
        pd.testing.assert_frame_equal(df_original_sorted, df_from_duckdb_sorted)

    def test_special_characters_in_data(self, tmp_path):
        """User workflow: load data with special characters, query in DuckDB"""
        # Setup: create CSV with special characters
        csv_file = tmp_path / "special.csv"
        csv_file.write_text(
            'name,description\n'
            '"Alice, Jr.","Likes ""coding"" & databases"\n'
            '"Bob Smith","Test with; semicolon"\n'
        )
        
        # Workflow: load, register
        frame = ar.read_csv(csv_file)
        conn = duckdb.connect(":memory:")
        ar.register_duckdb(frame, conn, "special_data")
        
        # Verify: query and verify special characters preserved
        result = conn.execute(
            "SELECT name FROM special_data WHERE name LIKE '%Jr%'"
        ).fetchdf()
        assert len(result) == 1
        assert "Jr" in result["name"].values[0]
