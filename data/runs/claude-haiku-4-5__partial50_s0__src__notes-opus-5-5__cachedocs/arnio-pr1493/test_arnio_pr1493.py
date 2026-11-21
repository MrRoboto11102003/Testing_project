import pytest
import pandas as pd
import arnio as ar
from unittest.mock import MagicMock
from arnio.integrations.duckdb import register_duckdb


class TestRegisterDuckdb:
    """Test suite for arnio.integrations.duckdb.register_duckdb."""

    def test_valid_call_with_mock_connection(self):
        """Test valid registration with mocked DuckDB connection."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        mock_conn = MagicMock()

        result = register_duckdb(frame, mock_conn, "test_table")

        assert result is None
        mock_conn.register.assert_called_once()
        call_args = mock_conn.register.call_args
        assert call_args.args[0] == "test_table"
        assert isinstance(call_args.args[1], pd.DataFrame)

    def test_valid_call_with_real_duckdb(self):
        """Test valid registration with real DuckDB connection."""
        duckdb = pytest.importorskip("duckdb")

        frame = ar.from_pandas(
            pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        )
        conn = duckdb.connect(":memory:")

        register_duckdb(frame, conn, "my_table")

        result = conn.execute("SELECT COUNT(*) as cnt FROM my_table").fetchall()
        assert result[0][0] == 3

    def test_frame_must_be_arframe_rejects_dataframe(self):
        """Test that pandas DataFrame is rejected."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        mock_conn = MagicMock()

        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            register_duckdb(df, mock_conn, "test")

        mock_conn.register.assert_not_called()

    def test_frame_must_be_arframe_rejects_none(self):
        """Test that None is rejected for frame."""
        mock_conn = MagicMock()

        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            register_duckdb(None, mock_conn, "test")

        mock_conn.register.assert_not_called()

    @pytest.mark.parametrize("bad_name", [123, None, b"bytes", True])
    def test_name_must_be_string(self, bad_name):
        """Test that name must be a string."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        mock_conn = MagicMock()

        with pytest.raises(TypeError, match="name must be a string"):
            register_duckdb(frame, mock_conn, bad_name)

        mock_conn.register.assert_not_called()

    def test_name_cannot_be_empty(self):
        """Test that empty name is rejected."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        mock_conn = MagicMock()

        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            register_duckdb(frame, mock_conn, "")

        mock_conn.register.assert_not_called()

    @pytest.mark.parametrize("whitespace_name", ["   ", "\t", "\n"])
    def test_name_cannot_be_whitespace_only(self, whitespace_name):
        """Test that whitespace-only name is rejected."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        mock_conn = MagicMock()

        with pytest.raises(ValueError, match="name must not be empty or whitespace-only"):
            register_duckdb(frame, mock_conn, whitespace_name)

        mock_conn.register.assert_not_called()

    def test_connection_must_have_callable_register_method(self):
        """Test that connection without callable register method is rejected."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        bad_conn = object()

        with pytest.raises(TypeError, match=r"callable register\(\) method"):
            register_duckdb(frame, bad_conn, "test")

    def test_connection_register_must_be_callable(self):
        """Test that connection with non-callable register is rejected."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        bad_conn = MagicMock()
        bad_conn.register = 42  # Not callable

        with pytest.raises(TypeError, match=r"callable register\(\) method"):
            register_duckdb(frame, bad_conn, "test")

    def test_frame_validation_before_name_validation(self):
        """Test that frame validation occurs before name validation."""
        df = pd.DataFrame({"a": [1]})
        mock_conn = MagicMock()

        # Both frame and name are invalid; frame error should come first
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            register_duckdb(df, mock_conn, 123)

    def test_name_validation_before_connection_validation(self):
        """Test that name validation occurs before connection validation."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        bad_conn = object()

        # Both name and connection are invalid; name error should come first
        with pytest.raises(ValueError, match="name must not be empty"):
            register_duckdb(frame, bad_conn, "")

    def test_name_with_spaces_preserved_verbatim(self):
        """Test that name with surrounding spaces is preserved verbatim."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        mock_conn = MagicMock()

        register_duckdb(frame, mock_conn, "  spaced_name  ")

        call_args = mock_conn.register.call_args
        assert call_args.args[0] == "  spaced_name  "

    def test_zero_row_frame_handled_correctly(self):
        """Test that zero-row frame is handled correctly."""
        frame = ar.from_pandas(pd.DataFrame({"a": pd.Series([], dtype="int64")}))
        mock_conn = MagicMock()

        result = register_duckdb(frame, mock_conn, "empty_table")

        assert result is None
        mock_conn.register.assert_called_once()

    def test_dataframe_passed_to_register_matches_to_pandas(self):
        """Test that the DataFrame passed to register matches ar.to_pandas output."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        mock_conn = MagicMock()

        register_duckdb(frame, mock_conn, "test")

        call_args = mock_conn.register.call_args
        df_passed = call_args.args[1]
        expected_df = ar.to_pandas(frame)

        pd.testing.assert_frame_equal(df_passed, expected_df)

    def test_always_returns_none(self):
        """Test that function always returns None regardless of conn.register return value."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        mock_conn = MagicMock()
        mock_conn.register.return_value = "some_value"

        result = register_duckdb(frame, mock_conn, "test")

        assert result is None
