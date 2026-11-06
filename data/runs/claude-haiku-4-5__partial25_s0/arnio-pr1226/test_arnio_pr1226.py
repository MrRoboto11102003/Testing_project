import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
import arnio as ar


class TestToPandas:
    """Test suite for arnio.convert.to_pandas function."""

    def test_returns_pandas_dataframe(self):
        """Test that to_pandas returns a pandas DataFrame."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        result = ar.to_pandas(frame)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_shape(self):
        """Test that shape is preserved during conversion."""
        frame = ar.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = ar.to_pandas(frame)
        assert result.shape == (3, 2)

    def test_preserves_column_names_and_order(self):
        """Test that column names and order are preserved."""
        frame = ar.from_dict({"z_col": [1], "a_col": [2], "m_col": [3]})
        result = ar.to_pandas(frame)
        assert list(result.columns) == ["z_col", "a_col", "m_col"]

    def test_preserves_integer_values(self):
        """Test that integer values are preserved correctly."""
        frame = ar.from_dict({"ints": [1, 2, 3, 100, -5]})
        result = ar.to_pandas(frame)
        assert result["ints"].tolist() == [1, 2, 3, 100, -5]

    def test_preserves_float_values(self):
        """Test that float values are preserved correctly."""
        frame = ar.from_dict({"floats": [1.1, 2.5, -3.7]})
        result = ar.to_pandas(frame)
        np.testing.assert_array_almost_equal(result["floats"].values, [1.1, 2.5, -3.7])

    def test_preserves_string_values(self):
        """Test that string values are preserved correctly."""
        frame = ar.from_dict({"strings": ["hello", "world", "test"]})
        result = ar.to_pandas(frame)
        assert result["strings"].tolist() == ["hello", "world", "test"]

    def test_preserves_boolean_values(self):
        """Test that boolean values are preserved correctly."""
        frame = ar.from_dict({"bools": [True, False, True, False]})
        result = ar.to_pandas(frame)
        assert result["bools"].tolist() == [True, False, True, False]

    def test_preserves_null_values(self):
        """Test that null values are handled and preserved."""
        frame = ar.from_dict({"values": [1, None, 3, None, 5]})
        result = ar.to_pandas(frame)
        assert pd.isna(result["values"].iloc[1])
        assert pd.isna(result["values"].iloc[3])
        assert result["values"].iloc[0] == 1
        assert result["values"].iloc[2] == 3

    def test_decimal_converted_to_string(self):
        """Test that Decimal objects are converted to strings."""
        df = pd.DataFrame({"price": [Decimal("19.99"), Decimal("29.95"), Decimal("99.99")]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        assert result["price"].dtype == object
        assert result["price"].tolist() == ["19.99", "29.95", "99.99"]

    def test_decimal_with_nulls_converted_to_string(self):
        """Test that Decimal with nulls are converted to strings."""
        df = pd.DataFrame({"price": [Decimal("19.99"), None, Decimal("50.00")]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        assert result["price"].dtype == object
        assert pd.isna(result["price"].iloc[1])
        assert result["price"].iloc[0] == "19.99"

    def test_copy_parameter_false_default(self):
        """Test to_pandas with copy=False (default behavior)."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        result = ar.to_pandas(frame, copy=False)
        assert isinstance(result, pd.DataFrame)
        assert result["col"].tolist() == [1, 2, 3]

    def test_copy_parameter_true(self):
        """Test to_pandas with copy=True."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        result = ar.to_pandas(frame, copy=True)
        assert isinstance(result, pd.DataFrame)
        assert result["col"].tolist() == [1, 2, 3]

    def test_mixed_data_types(self):
        """Test frame with mixed data types."""
        frame = ar.from_dict({
            "ints": [1, 2],
            "floats": [1.5, 2.5],
            "strings": ["a", "b"],
            "bools": [True, False]
        })
        result = ar.to_pandas(frame)
        assert result.shape == (2, 4)
        assert len(result.columns) == 4

    def test_unicode_and_special_characters_preserved(self):
        """Test that unicode and special characters are preserved."""
        frame = ar.from_dict({"text": ["café", "naïve", "test@123", "emoji😀"]})
        result = ar.to_pandas(frame)
        assert result["text"].tolist() == ["café", "naïve", "test@123", "emoji😀"]

    def test_empty_frame_conversion(self):
        """Test conversion of an empty frame."""
        frame = ar.from_dict({})
        result = ar.to_pandas(frame)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
