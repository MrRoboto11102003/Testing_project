import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
import arnio as ar


class TestToPandas:
    """Unit tests for arnio.convert.to_pandas function."""

    def test_basic_conversion_returns_dataframe(self):
        """Test that to_pandas returns a pandas DataFrame."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)

    def test_conversion_preserves_shape(self):
        """Test that shape is preserved during conversion."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.shape == (3, 2)

    def test_conversion_preserves_column_names(self):
        """Test that column names are preserved."""
        df = pd.DataFrame({"id": [1], "name": ["test"], "score": [99.5]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert list(result.columns) == ["id", "name", "score"]

    def test_conversion_preserves_column_order(self):
        """Test that column order is preserved."""
        df = pd.DataFrame({"z": [1], "a": [2], "m": [3]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert list(result.columns) == ["z", "a", "m"]

    def test_conversion_int64_dtype(self):
        """Test that int64 columns maintain their dtype."""
        df = pd.DataFrame({"values": [1, 2, 3, 100]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["values"].dtype == np.int64
        assert list(result["values"]) == [1, 2, 3, 100]

    def test_conversion_float64_dtype(self):
        """Test that float64 columns maintain their dtype."""
        df = pd.DataFrame({"values": [1.5, 2.7, 3.14]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["values"].dtype == np.float64
        assert list(result["values"]) == pytest.approx([1.5, 2.7, 3.14])

    def test_conversion_bool_dtype(self):
        """Test that boolean columns are handled correctly."""
        df = pd.DataFrame({"flags": [True, False, True]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert list(result["flags"]) == [True, False, True]

    def test_conversion_string_column(self):
        """Test that string columns are preserved."""
        df = pd.DataFrame({"names": ["alice", "bob", "charlie"]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert list(result["names"]) == ["alice", "bob", "charlie"]

    def test_conversion_with_nulls_in_integers(self):
        """Test that nullable integer columns preserve nulls."""
        df = pd.DataFrame({"ids": pd.array([1, None, 3], dtype="Int64")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert pd.isna(result["ids"].iloc[1])
        assert result["ids"].iloc[0] == 1
        assert result["ids"].iloc[2] == 3

    def test_conversion_with_nulls_in_floats(self):
        """Test that float columns with NaN preserve nulls."""
        df = pd.DataFrame({"scores": [1.5, np.nan, 3.5]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert pd.isna(result["scores"].iloc[1])
        assert result["scores"].iloc[0] == 1.5

    def test_conversion_with_nulls_in_strings(self):
        """Test that string columns with nulls preserve them."""
        df = pd.DataFrame({"names": pd.array(["alice", None, "bob"], dtype="string")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert pd.isna(result["names"].iloc[1])
        assert result["names"].iloc[0] == "alice"

    def test_conversion_with_null_booleans(self):
        """Test that nullable boolean columns preserve nulls."""
        df = pd.DataFrame({"flags": pd.array([True, None, False], dtype="boolean")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert pd.isna(result["flags"].iloc[1])
        assert result["flags"].iloc[0] is True

    def test_conversion_decimal_as_string(self):
        """Test that Decimal objects are converted to strings."""
        df = pd.DataFrame({
            "prices": [Decimal("19.99"), Decimal("29.95")]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        # Decimals should be preserved as strings
        assert result["prices"].iloc[0] == "19.99"
        assert result["prices"].iloc[1] == "29.95"

    def test_conversion_empty_frame(self):
        """Test conversion of an empty frame."""
        df = pd.DataFrame({"a": [], "b": []})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.shape == (0, 2)
        assert list(result.columns) == ["a", "b"]

    def test_conversion_single_row(self):
        """Test conversion of a frame with a single row."""
        df = pd.DataFrame({"x": [42], "y": [3.14]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.shape == (1, 2)
        assert result["x"].iloc[0] == 42

    def test_conversion_single_column(self):
        """Test conversion of a frame with a single column."""
        df = pd.DataFrame({"values": [10, 20, 30]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.shape == (3, 1)

    def test_copy_parameter_false(self):
        """Test that copy=False works and returns valid DataFrame."""
        df = pd.DataFrame({"numbers": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame, copy=False)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result["numbers"]) == [1, 2, 3, 4, 5]

    def test_copy_parameter_true(self):
        """Test that copy=True returns defensive pandas-owned buffers."""
        df = pd.DataFrame({"numbers": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame, copy=True)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result["numbers"]) == [1, 2, 3, 4, 5]

    def test_result_has_default_range_index(self):
        """Test that result has a default RangeIndex."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result.index, pd.RangeIndex)

    def test_whitespace_preservation_in_strings(self):
        """Test that whitespace in strings is preserved."""
        df = pd.DataFrame({"text": ["  hello  ", "world ", " test"]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert list(result["text"]) == ["  hello  ", "world ", " test"]

    def test_special_characters_in_strings(self):
        """Test that special characters are preserved in strings."""
        df = pd.DataFrame({
            "data": ["hello@world", "test#123", "foo$bar&baz"]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert list(result["data"]) == ["hello@world", "test#123", "foo$bar&baz"]

    def test_negative_numbers_preserved(self):
        """Test that negative numbers are preserved correctly."""
        df = pd.DataFrame({
            "ints": [-1, -100, -999],
            "floats": [-1.5, -99.99, -0.001]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert list(result["ints"]) == [-1, -100, -999]
        assert result["floats"].iloc[0] == -1.5

    def test_zero_values_preserved(self):
        """Test that zero values are preserved correctly."""
        df = pd.DataFrame({
            "int_zeros": [0, 0, 0],
            "float_zeros": [0.0, 0.0, 0.0]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert list(result["int_zeros"]) == [0, 0, 0]
        assert list(result["float_zeros"]) == [0.0, 0.0, 0.0]

    def test_large_integers_preserved(self):
        """Test that large integer values are preserved."""
        df = pd.DataFrame({
            "large": [9223372036854775807, -9223372036854775808, 0]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["large"].iloc[0] == 9223372036854775807
