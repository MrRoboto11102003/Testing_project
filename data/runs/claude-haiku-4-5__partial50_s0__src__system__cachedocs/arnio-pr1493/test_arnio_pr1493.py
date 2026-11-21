"""
System (end-to-end) tests for arnio.integrations.duckdb.register_duckdb.

These tests verify complete user workflows: loading CSVs, cleaning/validating
data, registering with DuckDB, and querying registered tables.
"""

import pytest
import sys
import pandas as pd
import arnio as ar

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


pytestmark = pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")


class TestRegisterDuckDBBasic:
    """Test basic registration and querying workflow."""

    def test_register_and_query_simple_csv(self, tmp_path):
        """End-to-end: load CSV, register with DuckDB, query results."""
        # Create test CSV
        csv_file = tmp_path / "sales.csv"
        csv_file.write_text("region,amount\nUS,100\nEU,150\nUS,120\n")

        # Load into ArFrame
        frame = ar.read_csv(str(csv_file))
        assert frame.shape == (3, 2)

        # Register with DuckDB
        conn = duckdb.connect()
        ar.register_duckdb(frame, conn, "sales_data")

        # Query the registered table
        result = conn.execute(
            "SELECT region, SUM(amount) as total FROM sales_data GROUP BY region ORDER BY region"
        ).fetchdf()

        assert len(result) == 2
        assert result.iloc[0]["region"] == "EU"
        assert result.iloc[1]["region"] == "US"

    def test_register_cleaned_frame_pipeline(self, tmp_path):
        """End-to-end: load, clean with pipeline, register, verify duplicates removed."""
        # Messy CSV with duplicates and whitespace
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("id,name,value\n1,  alice  ,100\n2,  bob  ,200\n1,  alice  ,100\n")

        # Load and clean
        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])

        # Register cleaned frame
        conn = duckdb.connect()
        ar.register_duckdb(clean, conn, "cleaned")

        # Verify duplicates are gone
        result = conn.execute("SELECT COUNT(*) as cnt FROM cleaned").fetchdf()
        assert result.iloc[0]["cnt"] == 2

    def test_register_multiple_frames_in_connection(self, tmp_path):
        """End-to-end: load multiple CSVs, register each, join in DuckDB."""
        # Create two CSV files
        users_csv = tmp_path / "users.csv"
        users_csv.write_text("user_id,user_name\n1,Alice\n2,Bob\n")

        orders_csv = tmp_path / "orders.csv"
        orders_csv.write_text("order_id,user_id,amount\n101,1,500\n102,2,750\n103,1,250\n")

        # Load both into ArFrames
        users_frame = ar.read_csv(str(users_csv))
        orders_frame = ar.read_csv(str(orders_csv))

        # Register both
        conn = duckdb.connect()
        ar.register_duckdb(users_frame, conn, "users")
        ar.register_duckdb(orders_frame, conn, "orders")

        # Join query
        result = conn.execute(
            "SELECT u.user_name, COUNT(o.order_id) as order_count "
            "FROM users u LEFT JOIN orders o ON u.user_id = o.user_id "
            "GROUP BY u.user_name ORDER BY u.user_name"
        ).fetchdf()

        assert len(result) == 2
        assert result.iloc[0]["user_name"] == "Alice"
        assert result.iloc[0]["order_count"] == 2

    def test_register_from_pandas_dataframe(self):
        """End-to-end: create ArFrame from pandas, register, query."""
        # Create pandas DataFrame
        pdf = pd.DataFrame({
            "id": [1, 2, 3],
            "category": ["A", "B", "A"],
            "value": [100, 200, 150]
        })

        # Convert to ArFrame
        frame = ar.from_pandas(pdf)

        # Register with DuckDB
        conn = duckdb.connect()
        ar.register_duckdb(frame, conn, "from_pandas_table")

        # Query
        result = conn.execute(
            "SELECT category, SUM(value) as total FROM from_pandas_table "
            "GROUP BY category ORDER BY category"
        ).fetchdf()

        assert len(result) == 2
        assert result.iloc[0]["total"] == 250  # Category A
        assert result.iloc[1]["total"] == 200  # Category B

    def test_register_from_dict_source(self):
        """End-to-end: create ArFrame from dict, register, verify data."""
        # Create ArFrame from dictionary
        data = {"product": ["Apple", "Banana", "Orange"], "price": [1.5, 0.75, 2.0]}
        frame = ar.from_dict(data)

        # Register with DuckDB
        conn = duckdb.connect()
        ar.register_duckdb(frame, conn, "products")

        # Query with aggregation
        result = conn.execute(
            "SELECT COUNT(*) as cnt, AVG(price) as avg_price FROM products"
        ).fetchdf()

        assert result.iloc[0]["cnt"] == 3
        assert abs(result.iloc[0]["avg_price"] - 1.4166667) < 0.01


class TestRegisterDuckDBValidation:
    """Test schema validation combined with DuckDB registration."""

    def test_validate_then_register_with_duckdb(self, tmp_path):
        """End-to-end: validate schema, register valid frame, query verified data."""
        # Create CSV with predictable data types
        csv_file = tmp_path / "customers.csv"
        csv_file.write_text("id,email,score\n1,alice@test.com,95\n2,bob@test.com,87\n")

        # Load and validate
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "email": ar.String(nullable=False),
            "score": ar.Int64(min=0, max=100),
        })
        result = ar.validate(frame, schema)
        assert result.passed

        # Register validated frame
        conn = duckdb.connect()
        ar.register_duckdb(frame, conn, "customers")

        # Query validated data
        query_result = conn.execute(
            "SELECT COUNT(*) as cnt FROM customers WHERE score >= 90"
        ).fetchdf()
        assert query_result.iloc[0]["cnt"] == 1


class TestRegisterDuckDBDataTypes:
    """Test that data types are correctly preserved through registration."""

    def test_null_values_preserved_in_duckdb(self, tmp_path):
        """End-to-end: load CSV with nulls, register, query nulls correctly."""
        csv_file = tmp_path / "data_with_nulls.csv"
        csv_file.write_text("id,value,note\n1,100,present\n2,,missing\n3,300,\n")

        frame = ar.read_csv(str(csv_file))
        conn = duckdb.connect()
        ar.register_duckdb(frame, conn, "nulls_table")

        # Count rows with NULL values
        result = conn.execute(
            "SELECT COUNT(*) as cnt FROM nulls_table WHERE value IS NULL OR note IS NULL"
        ).fetchdf()

        assert result.iloc[0]["cnt"] == 2

    def test_numeric_types_in_duckdb_query(self, tmp_path):
        """End-to-end: load numeric CSV, register, perform numeric aggregations."""
        csv_file = tmp_path / "numbers.csv"
        csv_file.write_text("int_col,float_col\n1,1.5\n2,2.5\n3,3.5\n")

        frame = ar.read_csv(str(csv_file))
        conn = duckdb.connect()
        ar.register_duckdb(frame, conn, "numbers")

        # Numeric query
        result = conn.execute(
            "SELECT SUM(int_col) as int_sum, AVG(float_col) as float_avg FROM numbers"
        ).fetchdf()

        assert result.iloc[0]["int_sum"] == 6
        assert abs(result.iloc[0]["float_avg"] - 2.5) < 0.01


class TestRegisterDuckDBErrors:
    """Test error handling and validation."""

    def test_error_on_non_arframe_input(self):
        """Verify TypeError when frame is not an ArFrame."""
        conn = duckdb.connect()
        pdf = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.register_duckdb(pdf, conn, "table")

    def test_error_on_non_string_table_name(self, tmp_path):
        """Verify TypeError when name is not a string."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        frame = ar.read_csv(str(csv_file))
        conn = duckdb.connect()

        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(frame, conn, 123)

    def test_error_on_empty_table_name(self, tmp_path):
        """Verify ValueError when name is empty string."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        frame = ar.read_csv(str(csv_file))
        conn = duckdb.connect()

        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(frame, conn, "")

    def test_error_on_whitespace_table_name(self, tmp_path):
        """Verify ValueError when name is only whitespace."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        frame = ar.read_csv(str(csv_file))
        conn = duckdb.connect()

        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(frame, conn, "   \t  ")

    def test_error_on_invalid_connection_object(self, tmp_path):
        """Verify TypeError when conn is not a valid DuckDB connection."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        frame = ar.read_csv(str(csv_file))

        invalid_conn = {"not": "a_connection"}
        with pytest.raises(TypeError, match="conn must be a DuckDB connection"):
            ar.register_duckdb(frame, invalid_conn, "table")


class TestRegisterDuckDBComplexWorkflows:
    """Test complex, realistic workflows combining multiple arnio operations."""

    def test_load_clean_validate_register_query(self, tmp_path):
        """End-to-end: CSV → clean → validate → register → complex query."""
        csv_file = tmp_path / "transactions.csv"
        csv_file.write_text(
            "tx_id,customer_id,amount,status\n"
            "1,  101  ,500,completed\n"
            "2,  102  ,750,completed\n"
            "1,  101  ,500,completed\n"
            "3,,1000,pending\n"
        )

        # Load and clean
        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
            ("drop_nulls", {"subset": ["customer_id"]}),
        ])

        # Validate
        schema = ar.Schema({
            "tx_id": ar.Int64(),
            "customer_id": ar.Int64(),
            "amount": ar.Int64(min=0),
            "status": ar.String(),
        })
        validation = ar.validate(clean, schema)
        assert validation.passed

        # Register and query
        conn = duckdb.connect()
        ar.register_duckdb(clean, conn, "transactions")

        result = conn.execute(
            "SELECT customer_id, COUNT(*) as tx_count, SUM(amount) as total_amount "
            "FROM transactions WHERE status = 'completed' "
            "GROUP BY customer_id ORDER BY total_amount DESC"
        ).fetchdf()

        assert len(result) == 2
        assert result.iloc[0]["total_amount"] == 750

    def test_profile_then_register_and_verify(self, tmp_path):
        """End-to-end: profile data, register, verify profile stats match DuckDB."""
        csv_file = tmp_path / "profile_test.csv"
        csv_file.write_text("id,name,age\n1,Alice,30\n2,Bob,25\n3,Charlie,35\n")

        frame = ar.read_csv(str(csv_file))

        # Profile
        report = ar.profile(frame)
        assert report.row_count == 3

        # Register
        conn = duckdb.connect()
        ar.register_duckdb(frame, conn, "profile_test")

        # Verify DuckDB counts match profile
        duckdb_count = conn.execute("SELECT COUNT(*) as cnt FROM profile_test").fetchdf()
        assert duckdb_count.iloc[0]["cnt"] == report.row_count
