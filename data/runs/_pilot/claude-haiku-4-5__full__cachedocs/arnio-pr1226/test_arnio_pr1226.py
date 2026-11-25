import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestToPandas:
    """Tests for arnio.convert.to_pandas"""

    def test_basic_conversion(self):
        """Convert a simple ArFrame to pandas DataFrame."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"]
        }))
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["id", "name"]
        assert len(df) == 3

    def test_dtype_preservation_int(self):
        """Verify int dtypes are preserved during conversion."""
        original = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        assert result["col"].dtype == original["col"].dtype

    def test_dtype_preservation_float(self):
        """Verify float dtypes are preserved during conversion."""
        original = pd.DataFrame({"col": [1.5, 2.5, 3.5]})
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        assert result["col"].dtype == original["col"].dtype

    def test_null_values_handled(self):
        """Verify null/NaN values are handled correctly."""
        original = pd.DataFrame({
            "int_col": pd.array([1, None, 3], dtype="Int64"),
            "float_col": [1.5, np.nan, 3.5],
        })
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        
        assert pd.isna(result["int_col"].iloc[1])
        assert pd.isna(result["float_col"].iloc[1])

    def test_copy_false_default(self):
        """Verify copy=False is the default behavior."""
        original = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)  # no copy parameter
        assert result["col"].tolist() == [1, 2, 3]

    def test_copy_true_returns_dataframe(self):
        """Verify copy=True returns a defensive DataFrame copy."""
        original = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame, copy=True)
        assert isinstance(result, pd.DataFrame)
        assert result["col"].tolist() == [1, 2, 3]

    def test_column_order_preserved(self):
        """Verify column order is preserved from ArFrame."""
        original = pd.DataFrame({
            "z": [1, 2],
            "a": [3, 4],
            "m": [5, 6]
        })
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        assert list(result.columns) == ["z", "a", "m"]

    def test_metadata_attrs_restored(self):
        """Verify DataFrame attrs metadata is restored."""
        original = pd.DataFrame({"col": [1, 2, 3]})
        original.attrs["source"] = "test_data"
        original.attrs["version"] = 42
        
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        
        assert result.attrs.get("source") == "test_data"
        assert result.attrs.get("version") == 42

    def test_round_trip_conversion(self):
        """Verify values survive round-trip conversion."""
        original = pd.DataFrame({
            "int": [1, 2, 3],
            "float": [1.5, 2.5, 3.5],
            "str": ["x", "y", "z"]
        })
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        
        pd.testing.assert_frame_equal(original, result)

    def test_empty_frame(self):
        """Convert an empty ArFrame with no rows."""
        original = pd.DataFrame({"col": pd.Series([], dtype="int64")})
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        
        assert len(result) == 0
        assert "col" in result.columns

    def test_single_row(self):
        """Convert an ArFrame with a single row."""
        original = pd.DataFrame({"a": [1], "b": ["x"]})
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        
        assert len(result) == 1
        assert result["a"].iloc[0] == 1
        assert result["b"].iloc[0] == "x"

    def test_single_column(self):
        """Convert an ArFrame with a single column."""
        original = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        
        assert len(result.columns) == 1
        assert result["col"].tolist() == [1, 2, 3]

    def test_boolean_column_with_nulls(self):
        """Handle boolean columns with null values correctly."""
        original = pd.DataFrame({
            "bool_col": pd.array([True, None, False], dtype="boolean")
        })
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        
        assert result["bool_col"].iloc[0] == True
        assert pd.isna(result["bool_col"].iloc[1])
        assert result["bool_col"].iloc[2] == False

    def test_string_column_with_nulls(self):
        """Handle string columns with null values correctly."""
        original = pd.DataFrame({
            "str_col": pd.array(["a", None, "c"], dtype="string")
        })
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        
        assert result["str_col"].iloc[0] == "a"
        assert pd.isna(result["str_col"].iloc[1])
        assert result["str_col"].iloc[2] == "c"

    def test_multiple_dtypes_mixed(self):
        """Convert frame with multiple mixed data types."""
        original = pd.DataFrame({
            "int": [1, 2],
            "float": [1.1, 2.2],
            "str": ["a", "b"],
            "bool": [True, False],
        })
        frame = ar.from_pandas(original)
        result = ar.to_pandas(frame)
        
        assert len(result) == 2
        assert len(result.columns) == 4
        assert list(result.columns) == ["int", "float", "str", "bool"]
