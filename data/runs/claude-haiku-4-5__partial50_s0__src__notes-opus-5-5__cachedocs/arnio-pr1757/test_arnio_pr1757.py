import pytest
import pandas as pd
import re
import arnio as ar


class TestDropColumns:
    """Tests for arnio.cleaning.drop_columns"""

    def test_drop_single_column(self):
        """Drop a single column from frame"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
        result = ar.drop_columns(frame, ["b"])
        assert result.columns == ["a", "c"]
        assert result.shape == (1, 2)

    def test_drop_multiple_columns(self):
        """Drop multiple columns from frame"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]}))
        result = ar.drop_columns(frame, ["b", "d"])
        assert result.columns == ["a", "c"]

    def test_column_order_preserved(self):
        """Remaining columns preserve original order regardless of request order"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
        result = ar.drop_columns(frame, ["c", "a"])
        assert result.columns == ["b"]

    def test_missing_column_raises_valueerror(self):
        """ValueError raised when requesting non-existent column"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
        with pytest.raises(ValueError, match=re.escape("Columns not found in frame: ['zz']")):
            ar.drop_columns(frame, ["zz"])

    def test_drop_all_columns_raises_valueerror(self):
        """ValueError raised when attempting to drop all columns"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
        with pytest.raises(ValueError, match="drop_columns cannot remove all columns from the frame"):
            ar.drop_columns(frame, ["a", "b"])

    def test_all_columns_with_duplicates_raises_valueerror(self):
        """Duplicates count toward all-columns check"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
        with pytest.raises(ValueError, match="drop_columns cannot remove all columns from the frame"):
            ar.drop_columns(frame, ["a", "a"])

    def test_string_input_raises_typeerror(self):
        """TypeError raised when columns argument is a string"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="columns must be a sequence of column names, not a string"):
            ar.drop_columns(frame, "a")

    def test_set_input_raises_typeerror(self):
        """TypeError raised when columns argument is a set"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="columns must be a sequence of column names"):
            ar.drop_columns(frame, {"a"})

    def test_non_string_item_raises_typeerror(self):
        """TypeError raised when columns contains non-string items"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
        with pytest.raises(TypeError, match="columns must contain only string column names"):
            ar.drop_columns(frame, ["a", 1])

    def test_non_arframe_input_raises_typeerror(self):
        """TypeError raised when frame is not an ArFrame"""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(TypeError):
            ar.drop_columns(df, ["a"])

    def test_empty_columns_list(self):
        """Empty columns list returns frame unchanged"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
        result = ar.drop_columns(frame, [])
        assert result.columns == ["a", "b"]
        assert result.shape == (1, 2)

    def test_pandas_index_input(self):
        """pd.Index can be used as columns argument"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
        result = ar.drop_columns(frame, pd.Index(["b"]))
        assert result.columns == ["a", "c"]

    def test_tuple_input(self):
        """Tuple can be used as columns argument"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
        result = ar.drop_columns(frame, ("b",))
        assert result.columns == ["a", "c"]

    def test_input_frame_not_mutated(self):
        """Original frame is not modified by drop_columns"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
        original_columns = frame.columns.copy()
        original_shape = frame.shape
        ar.drop_columns(frame, ["a"])
        assert frame.columns == original_columns
        assert frame.shape == original_shape

    def test_null_values_preserved(self):
        """Null values in remaining columns are preserved"""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2], "b": [None, 3], "c": [4.5, None]}))
        result = ar.drop_columns(frame, ["b"])
        result_df = ar.to_pandas(result)
        assert result_df["a"].tolist() == [1, 2]
        assert result_df["c"][0] == 4.5
        assert pd.isna(result_df["c"][1])
