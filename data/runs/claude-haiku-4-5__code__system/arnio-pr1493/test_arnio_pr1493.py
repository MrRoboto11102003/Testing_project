import pytest
import sys
import pandas as pd
import numpy as np

try:
    import duckdb
except ImportError:
    pytest.skip("duckdb not installed", allow_module_level=True)

try:
    import arnio as ar
except ImportError:
    pytest.skip("arnio not installed", allow_module_level=True)


class TestRegisterDuckDB:
    """End-to-end tests for arnio.integrations.duckdb.register_duckdb function."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300\n")
        return csv_file

    @pytest.fixture
    def sample_csv_typed(self, tmp_path):
        """Create a CSV with multiple data types."""
        csv_file = tmp_path / "typed.csv"
        csv_file.write_text(
            "id,name,amount,rating\n"
            "1,Alice,100.5,4.5\n"
            "2,Bob,200.0,3.8\n"
            "3,Charlie,300.5,4.2\n"
        )
        return csv_file

    @pytest.fixture
    def duckdb_conn(self):
        """Create a DuckDB connection."""
        conn = duckdb.connect(":memory:")
        yield conn
        conn.close()

    def test_basic_registration_and_query(self, sample_csv, duckdb_conn):
        """Test: Read CSV as ArFrame, register, and query via SQL."""
        frame = ar.read_csv(sample_csv)
        ar.register_duckdb(frame, duckdb_conn, "my_table")
        
        result = duckdb_conn.execute("SELECT * FROM my_table").fetchdf()
        
        assert len(result) == 3
        assert list(result.columns) == ["id", "name", "value"]
        assert result["name"].tolist() == ["Alice", "Bob", "Charlie"]
        assert result["value"].tolist() == [100, 200, 300]

    def test_registered_table_aggregation(self, sample_csv, duckdb_conn):
        """Test: Registered table can be used for SQL aggregations."""
        frame = ar.read_csv(sample_csv)
        ar.register_duckdb(frame, duckdb_conn, "sales")
        
        result = duckdb_conn.execute(
            "SELECT COUNT(*) as cnt, SUM(value) as total, AVG(value) as avg FROM sales"
        ).fetchdf()
        
        assert result.loc[0, "cnt"] == 3
        assert result.loc[0, "total"] == 600
        assert result.loc[0, "avg"] == 200.0

    def test_multiple_registrations_with_join(self, tmp_path, sample_csv, duckdb_conn):
        """Test: Multiple frames registered and joined in SQL."""
        csv2 = tmp_path / "meta.csv"
        csv2.write_text("id,category\n1,A\n2,B\n3,C\n")
        
        frame1 = ar.read_csv(sample_csv)
        frame2 = ar.read_csv(csv2)
        ar.register_duckdb(frame1, duckdb_conn, "data1")
        ar.register_duckdb(frame2, duckdb_conn, "data2")
        
        result = duckdb_conn.execute(
            "SELECT d1.name, d1.value, d2.category FROM data1 d1 "
            "JOIN data2 d2 ON d1.id = d2.id"
        ).fetchdf()
        
        assert len(result) == 3
        assert result["category"].tolist() == ["A", "B", "C"]

    def test_data_type_preservation(self, sample_csv_typed, duckdb_conn):
        """Test: Data types are preserved through registration."""
        frame = ar.read_csv(sample_csv_typed)
        ar.register_duckdb(frame, duckdb_conn, "typed_table")
        
        result = duckdb_conn.execute("SELECT * FROM typed_table").fetchdf()
        
        assert len(result) == 3
        assert result["name"].tolist() == ["Alice", "Bob", "Charlie"]
        assert np.isclose(result["amount"].iloc[0], 100.5)
        assert np.isclose(result["rating"].iloc[2], 4.2)

    def test_error_frame_not_arframe(self, duckdb_conn):
        """Test: TypeError if frame is not ArFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.register_duckdb(df, duckdb_conn, "table")

    def test_error_name_not_string(self, sample_csv, duckdb_conn):
        """Test: TypeError if name is not a string."""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(frame, duckdb_conn, 123)

    def test_error_name_is_none(self, sample_csv, duckdb_conn):
        """Test: TypeError if name is None."""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(frame, duckdb_conn, None)

    def test_error_empty_name(self, sample_csv, duckdb_conn):
        """Test: ValueError if name is empty string."""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(frame, duckdb_conn, "")

    def test_error_whitespace_name(self, sample_csv, duckdb_conn):
        """Test: ValueError if name is whitespace-only."""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(frame, duckdb_conn, "   \t\n  ")

    def test_error_invalid_connection(self, sample_csv):
        """Test: TypeError if connection lacks register method."""
        frame = ar.read_csv(sample_csv)
        fake_conn = {}
        
        with pytest.raises(TypeError, match="DuckDB connection"):
            ar.register_duckdb(frame, fake_conn, "table")

    def test_registration_with_special_name(self, sample_csv, duckdb_conn):
        """Test: Registration with underscores in table name."""
        frame = ar.read_csv(sample_csv)
        ar.register_duckdb(frame, duckdb_conn, "my_data_2024_v1")
        
        result = duckdb_conn.execute("SELECT COUNT(*) FROM my_data_2024_v1").fetchdf()
        assert result.iloc[0, 0] == 3

    def test_same_frame_different_names(self, sample_csv, duckdb_conn):
        """Test: Same frame registered with multiple names."""
        frame = ar.read_csv(sample_csv)
        
        ar.register_duckdb(frame, duckdb_conn, "version_1")
        ar.register_duckdb(frame, duckdb_conn, "version_2")
        
        for name in ["version_1", "version_2"]:
            result = duckdb_conn.execute(f"SELECT COUNT(*) FROM {name}").fetchdf()
            assert result.iloc[0, 0] == 3

    def test_registered_data_filtering(self, sample_csv, duckdb_conn):
        """Test: Registered data can be filtered via SQL."""
        frame = ar.read_csv(sample_csv)
        ar.register_duckdb(frame, duckdb_conn, "products")
        
        result = duckdb_conn.execute(
            "SELECT name FROM products WHERE value > 150 ORDER BY name"
        ).fetchdf()
        
        assert len(result) == 2
        assert result["name"].tolist() == ["Bob", "Charlie"]

    def test_registered_large_dataset(self, tmp_path, duckdb_conn):
        """Test: Registration works with larger dataset."""
        csv_file = tmp_path / "large.csv"
        with open(csv_file, "w") as f:
            f.write("id,value\n")
            for i in range(1000):
                f.write(f"{i},{i*2}\n")
        
        frame = ar.read_csv(csv_file)
        ar.register_duckdb(frame, duckdb_conn, "large_table")
        
        result = duckdb_conn.execute("SELECT COUNT(*) FROM large_table").fetchdf()
        assert result.iloc[0, 0] == 1000

    def test_error_connection_no_register_method(self, sample_csv):
        """Test: TypeError if connection has no register method."""
        frame = ar.read_csv(sample_csv)
        
        class FakeConnection:
            def execute(self, sql):
                pass
        
        with pytest.raises(TypeError, match="register\\(\\) method"):
            ar.register_duckdb(frame, FakeConnection(), "table")
