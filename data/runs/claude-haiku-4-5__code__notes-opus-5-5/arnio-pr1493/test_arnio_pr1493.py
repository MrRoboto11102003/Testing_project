import pytest
import pandas as pd
from unittest.mock import MagicMock
import arnio as ar
from arnio.integrations.duckdb import register_duckdb


class TestRegisterDuckDB:
    """Tests for arnio.integrations.duckdb.register_duckdb"""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def empty_frame(self):
        """Create an empty ArFrame for testing."""
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        return ar.from_pandas(df)

    # --- Valid cases ---

    def test_valid_call_returns_none(self, sample_frame):
        """Valid call returns None."""
        conn = MagicMock()
        result = register_duckdb(sample_frame, conn, "my_table")
        assert result is None

    def test_register_called_with_correct_args(self, sample_frame):
        """register is called exactly once with name and DataFrame."""
        conn = MagicMock()
        register_duckdb(sample_frame, conn, "my_table")
        
        conn.register.assert_called_once()
        name_arg, df = conn.register.call_args.args
        assert name_arg == "my_table"
        assert isinstance(df, pd.DataFrame)
        pd.testing.assert_frame_equal(df, ar.to_pandas(sample_frame))

    def test_with_real_duckdb(self, sample_frame):
        """Integration test with real DuckDB."""
        duckdb = pytest.importorskip("duckdb")
        conn = duckdb.connect()
        
        register_duckdb(sample_frame, conn, "test_table")
        
        result = conn.execute("SELECT * FROM test_table ORDER BY a").fetchall()
        assert len(result) == 3
        assert result[0][0] == 1
        assert result[0][1] == "x"

    def test_name_with_spaces_accepted(self, sample_frame):
        """Name with surrounding spaces is accepted and passed verbatim."""
        conn = MagicMock()
        register_duckdb(sample_frame, conn, "  my_table  ")
        
        name_arg = conn.register.call_args.args[0]
        assert name_arg == "  my_table  "

    def test_single_character_name(self, sample_frame):
        """Single non-space character names are accepted."""
        conn = MagicMock()
        register_duckdb(sample_frame, conn, "x")
        conn.register.assert_called_once()

    def test_zero_row_frame(self, empty_frame):
        """Zero-row frames are accepted and registered."""
        conn = MagicMock()
        register_duckdb(empty_frame, conn, "empty_table")
        conn.register.assert_called_once()

    # --- Type validation for frame ---

    def test_frame_not_arframe_raises_typeerror(self):
        """frame that is not an ArFrame raises TypeError."""
        conn = MagicMock()
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            register_duckdb(None, conn, "t")
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            register_duckdb(pd.DataFrame({"a": [1]}), conn, "t")

    # --- Type validation for name ---

    def test_name_not_string_raises_typeerror(self, sample_frame):
        """name that is not a string raises TypeError."""
        conn = MagicMock()
        
        with pytest.raises(TypeError, match="name must be a string"):
            register_duckdb(sample_frame, conn, 123)
        
        with pytest.raises(TypeError, match="name must be a string"):
            register_duckdb(sample_frame, conn, None)
        
        with pytest.raises(TypeError, match="name must be a string"):
            register_duckdb(sample_frame, conn, True)

    def test_name_empty_or_whitespace_raises_valueerror(self, sample_frame):
        """name that is empty or whitespace-only raises ValueError."""
        conn = MagicMock()
        
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            register_duckdb(sample_frame, conn, "")
        
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            register_duckdb(sample_frame, conn, "   \t\n  ")

    # --- Conn validation ---

    def test_conn_missing_or_non_callable_register_raises_typeerror(self, sample_frame):
        """conn without callable register method raises TypeError."""
        
        with pytest.raises(TypeError, match=r"callable register\(\) method"):
            register_duckdb(sample_frame, None, "t")
        
        with pytest.raises(TypeError, match=r"callable register\(\) method"):
            register_duckdb(sample_frame, object(), "t")
        
        conn = MagicMock(register=5)
        with pytest.raises(TypeError, match=r"callable register\(\) method"):
            register_duckdb(sample_frame, conn, "t")

    # --- Validation order and side effects ---

    def test_validation_order_frame_before_name(self):
        """frame is validated before name when both are invalid."""
        conn = MagicMock()
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            register_duckdb(None, conn, 123)

    def test_validation_order_name_type_before_conn(self, sample_frame):
        """name type is validated before conn when both are invalid."""
        with pytest.raises(TypeError, match="name must be a string"):
            register_duckdb(sample_frame, None, 123)

    def test_validation_order_name_value_before_conn(self, sample_frame):
        """name value is validated before conn when both are invalid."""
        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            register_duckdb(sample_frame, None, "")

    def test_register_not_called_on_validation_failure(self, sample_frame):
        """register is not called if validation fails."""
        conn = MagicMock()
        
        with pytest.raises(ValueError):
            register_duckdb(sample_frame, conn, "")
        
        conn.register.assert_not_called()

    def test_propagates_exception_from_register(self, sample_frame):
        """Exceptions from conn.register are propagated."""
        conn = MagicMock()
        conn.register.side_effect = RuntimeError("custom error")
        
        with pytest.raises(RuntimeError, match="custom error"):
            register_duckdb(sample_frame, conn, "t")
