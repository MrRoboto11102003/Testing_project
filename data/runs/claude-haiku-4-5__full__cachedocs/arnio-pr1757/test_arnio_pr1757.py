import pytest
import pandas as pd
import arnio as ar


class TestDropColumns:
    """Test suite for ar.drop_columns()"""

    def test_drop_single_column(self):
        """Drop a single column from a frame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["b"])
        assert result.columns == ["a", "c"]

    def test_drop_multiple_columns(self):
        """Drop multiple columns from a frame."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["a", "c"])
        assert result.columns == ["b", "d"]

    def test_preserves_column_order(self):
        """Remaining columns preserve their original order."""
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3], "w": [4]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["y", "w"])
        assert result.columns == ["x", "z"]

    def test_drop_empty_list_returns_all_columns(self):
        """Dropping no columns returns all original columns."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, [])
        assert result.columns == ["a", "b", "c"]

    def test_drop_all_columns(self):
        """Dropping all columns returns an empty column list."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["a", "b"])
        assert result.columns == []

    def test_drop_from_single_column_frame(self):
        """Drop the only column from a frame."""
        df = pd.DataFrame({"only": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["only"])
        assert result.columns == []

    def test_raises_type_error_if_columns_is_string(self):
        """Raises TypeError if columns argument is a string, not a sequence."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.drop_columns(frame, "a")

    def test_raises_type_error_if_columns_is_bytes(self):
        """Raises TypeError if columns argument is bytes."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.drop_columns(frame, b"a")

    def test_raises_type_error_if_columns_contains_non_strings(self):
        """Raises TypeError if any item in columns is not a string."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.drop_columns(frame, ["a", 1, "c"])

    def test_raises_value_error_if_column_missing(self):
        """Raises ValueError if any requested column doesn't exist."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.drop_columns(frame, ["nonexistent"])

    def test_raises_value_error_if_any_column_missing(self):
        """Raises ValueError if any of multiple requested columns don't exist."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.drop_columns(frame, ["a", "missing"])

    def test_returns_new_arframe(self):
        """drop_columns returns a new ArFrame object."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["a"])
        assert result is not frame
        assert isinstance(result, ar.ArFrame)

    def test_does_not_modify_original_frame(self):
        """Original frame is not modified by drop_columns."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        frame = ar.from_pandas(df)
        ar.drop_columns(frame, ["b"])
        assert frame.columns == ["a", "b", "c"]

    def test_preserves_data_in_remaining_columns(self):
        """Data in remaining columns is preserved correctly."""
        df = pd.DataFrame({
            "keep1": [1, 2, 3],
            "drop": [10, 20, 30],
            "keep2": [100, 200, 300]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["drop"])
        result_df = ar.to_pandas(result)
        
        pd.testing.assert_series_equal(result_df["keep1"], df["keep1"])
        pd.testing.assert_series_equal(result_df["keep2"], df["keep2"])

    def test_case_sensitive_column_names(self):
        """Column name matching is case-sensitive."""
        df = pd.DataFrame({"Name": [1], "name": [2]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["Name"])
        assert result.columns == ["name"]
