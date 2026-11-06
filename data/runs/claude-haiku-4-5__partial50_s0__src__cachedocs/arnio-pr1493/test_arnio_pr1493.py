import pytest
import pandas as pd
import duckdb
import arnio as ar


class TestRegisterDuckdb:
    """Test suite for arnio.integrations.duckdb.register_duckdb"""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1]
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def duckdb_conn(self):
        """Create a DuckDB connection for testing."""
        return duckdb.connect(":memory:")

    def test_register_duckdb_basic(self, sample_frame, duckdb_conn):
        """Test basic registration of an ArFrame."""
        ar.register_duckdb(sample_frame, duckdb_conn, "test_table")
        
        result = duckdb_conn.execute("SELECT * FROM test_table").fetchdf()
        assert len(result) == 3
        assert list(result.columns) == ["id", "name", "score"]

    def test_register_duckdb_data_integrity(self, sample_frame, duckdb_conn):
        """Test that registered data matches the original frame."""
        ar.register_duckdb(sample_frame, duckdb_conn, "test_table")
        
        result = duckdb_conn.execute("SELECT * FROM test_table").fetchdf()
        original = ar.to_pandas(sample_frame)
        
        pd.testing.assert_frame_equal(result.reset_index(drop=True), 
                                      original.reset_index(drop=True))

    def test_register_duckdb_query_after_registration(self, sample_frame, duckdb_conn):
        """Test that registered table can be queried."""
        ar.register_duckdb(sample_frame, duckdb_conn, "users")
        
        result = duckdb_conn.execute(
            "SELECT name FROM users WHERE score > 90"
        ).fetchdf()
        
        assert len(result) == 2
        assert set(result["name"].tolist()) == {"Alice", "Charlie"}

    def test_register_duckdb_invalid_frame_type(self, duckdb_conn):
        """Test TypeError when frame is not an ArFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.register_duckdb(df, duckdb_conn, "table")

    def test_register_duckdb_invalid_name_type(self, sample_frame, duckdb_conn):
        """Test TypeError when name is not a string."""
        with pytest.raises(TypeError, match="name must be a string"):
            ar.register_duckdb(sample_frame, duckdb_conn, 123)

    def test_register_duckdb_empty_name(self, sample_frame, duckdb_conn):
        """Test ValueError when name is empty string."""
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(sample_frame, duckdb_conn, "")

    def test_register_duckdb_whitespace_only_name(self, sample_frame, duckdb_conn):
        """Test ValueError when name is only whitespace."""
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            ar.register_duckdb(sample_frame, duckdb_conn, "   ")

    def test_register_duckdb_invalid_connection_no_register(self, sample_frame):
        """Test TypeError when conn lacks register method."""
        class FakeConn:
            pass
        
        with pytest.raises(TypeError, match="conn must be a DuckDB connection"):
            ar.register_duckdb(sample_frame, FakeConn(), "table")

    def test_register_duckdb_connection_with_non_callable_register(self, sample_frame):
        """Test TypeError when register attribute is not callable."""
        class FakeConnBadRegister:
            register = "not_callable"
        
        with pytest.raises(TypeError, match="conn must be a DuckDB connection"):
            ar.register_duckdb(sample_frame, FakeConnBadRegister(), "table")

    def test_register_duckdb_multiple_tables(self, sample_frame, duckdb_conn):
        """Test registering multiple tables in same connection."""
        ar.register_duckdb(sample_frame, duckdb_conn, "table1")
        ar.register_duckdb(sample_frame, duckdb_conn, "table2")
        
        result1 = duckdb_conn.execute("SELECT COUNT(*) as cnt FROM table1").fetchdf()
        result2 = duckdb_conn.execute("SELECT COUNT(*) as cnt FROM table2").fetchdf()
        
        assert result1["cnt"].iloc[0] == 3
        assert result2["cnt"].iloc[0] == 3

    def test_register_duckdb_with_null_values(self, duckdb_conn):
        """Test registering a frame containing null values."""
        df = pd.DataFrame({
            "id": [1, 2, None],
            "value": [10.5, None, 30.5]
        })
        frame = ar.from_pandas(df)
        
        ar.register_duckdb(frame, duckdb_conn, "nullable_table")
        result = duckdb_conn.execute("SELECT * FROM nullable_table").fetchdf()
        
        assert len(result) == 3
        assert result.isna().sum().sum() > 0

    def test_register_duckdb_with_mixed_dtypes(self, duckdb_conn):
        """Test registering a frame with various column types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        frame = ar.from_pandas(df)
        
        ar.register_duckdb(frame, duckdb_conn, "mixed_types")
        result = duckdb_conn.execute("SELECT COUNT(*) as cnt FROM mixed_types").fetchdf()
        
        assert result["cnt"].iloc[0] == 3

    def test_register_duckdb_table_name_with_underscores(self, sample_frame, duckdb_conn):
        """Test registering with valid alphanumeric table name."""
        ar.register_duckdb(sample_frame, duckdb_conn, "my_test_table_123")
        
        result = duckdb_conn.execute("SELECT COUNT(*) as cnt FROM my_test_table_123").fetchdf()
        assert result["cnt"].iloc[0] == 3

    def test_register_duckdb_overwrite_existing_table(self, duckdb_conn):
        """Test that registering with same name overwrites the previous table."""
        df1 = pd.DataFrame({"col": [1, 2, 3]})
        df2 = pd.DataFrame({"col": [4, 5]})
        
        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        
        ar.register_duckdb(frame1, duckdb_conn, "same_name")
        ar.register_duckdb(frame2, duckdb_conn, "same_name")
        
        result = duckdb_conn.execute("SELECT COUNT(*) as cnt FROM same_name").fetchdf()
        assert result["cnt"].iloc[0] == 2
