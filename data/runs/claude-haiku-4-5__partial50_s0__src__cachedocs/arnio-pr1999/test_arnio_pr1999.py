import pytest
import pandas as pd
import arnio as ar


class TestDropColumns:
    """Tests for ArFrame.drop_columns()"""

    def test_drop_single_column(self):
        """Drop a single column."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        result = frame.drop_columns(["b"])
        assert result.columns == ["a", "c"]

    def test_drop_duplicates_silently_ignored(self):
        """Duplicate column names in drop list are silently ignored."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        result = frame.drop_columns(["b", "b", "c"])
        assert result.columns == ["a"]

    def test_drop_empty_list_returns_copy(self):
        """Empty column list returns unchanged copy."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        result = frame.drop_columns([])
        assert result.columns == ["a", "b"]
        assert result is not frame

    def test_drop_invalid_type_raises_typeerror(self):
        """Raises TypeError if cols is not list or tuple."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            frame.drop_columns("a")

    def test_drop_unknown_column_raises_valueerror(self):
        """Raises ValueError if any column doesn't exist."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            frame.drop_columns(["unknown"])

    def test_drop_all_columns_raises_valueerror(self):
        """Cannot drop all columns from the frame."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            frame.drop_columns(["a"])


class TestSelectColumns:
    """Tests for ArFrame.select_columns()"""

    def test_select_single_column(self):
        """Select a single column."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        result = frame.select_columns(["b"])
        assert result.columns == ["b"]

    def test_select_preserves_requested_order(self):
        """Selected columns appear in requested order, not original order."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        result = frame.select_columns(["c", "a", "b"])
        assert result.columns == ["c", "a", "b"]

    def test_select_duplicate_raises_valueerror(self):
        """Raises ValueError if selection has duplicates."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            frame.select_columns(["a", "a"])

    def test_select_unknown_column_raises_valueerror(self):
        """Raises ValueError if any column doesn't exist."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            frame.select_columns(["unknown"])


class TestSelectDtypes:
    """Tests for ArFrame.select_dtypes()"""

    def test_select_include_numeric(self):
        """Select numeric columns by include."""
        df = pd.DataFrame(
            {"a": [1, 2], "b": [1.5, 2.5], "c": ["x", "y"]}
        )
        frame = ar.from_pandas(df)
        result = frame.select_dtypes(include=["int64", "float64"])
        assert set(result.columns) == {"a", "b"}

    def test_select_exclude_string(self):
        """Exclude string columns."""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        frame = ar.from_pandas(df)
        result = frame.select_dtypes(exclude="string")
        assert result.columns == ["a"]

    def test_select_neither_include_nor_exclude_raises_valueerror(self):
        """Raises ValueError if neither include nor exclude provided."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            frame.select_dtypes()

    def test_select_invalid_dtype_raises_valueerror(self):
        """Raises ValueError for unrecognized dtype string."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            frame.select_dtypes(include="invalid_dtype")

    def test_select_overlapping_include_exclude_raises_valueerror(self):
        """Raises ValueError if include and exclude overlap."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            frame.select_dtypes(include="int64", exclude="int64")
