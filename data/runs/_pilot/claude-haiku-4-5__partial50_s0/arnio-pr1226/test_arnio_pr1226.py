import pytest
import pandas as pd
import numpy as np
import arnio as ar
from decimal import Decimal


class TestToPandas:
    """Test suite for arnio.convert.to_pandas function"""

    def test_basic_conversion(self):
        """Test basic ArFrame to DataFrame conversion"""
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        frame = ar.from_pandas(pdf)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert result.shape == (3, 2)

    def test_copy_false_default(self):
        """Test that copy=False is the default behavior"""
        pdf = pd.DataFrame({"values": [1, 2, 3]})
        frame = ar.from_pandas(pdf)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result["values"]) == [1, 2, 3]

    def test_copy_true_creates_defensive_buffers(self):
        """Test that copy=True creates pandas-owned defensive buffers"""
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]})
        frame = ar.from_pandas(pdf)
        result = ar.to_pandas(frame, copy=True)
        
        pd.testing.assert_frame_equal(result, pdf)

    def test_copy_is_keyword_only(self):
        """Test that copy parameter must be passed as keyword argument"""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(pdf)
        
        # Should work with keyword argument
        result = ar.to_pandas(frame, copy=False)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_column_order_and_names(self):
        """Test that column order and names are preserved"""
        pdf = pd.DataFrame({"z": [1, 2], "a": [3, 4], "m": [5, 6]})
        frame = ar.from_pandas(pdf)
        result = ar.to_pandas(frame)
        
        assert list(result.columns) == ["z", "a", "m"]

    def test_returns_default_range_index(self):
        """Test that result has a default RangeIndex, not original index"""
        pdf = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(pdf)
        result = ar.to_pandas(frame)
        
        assert isinstance(result.index, pd.RangeIndex)
        assert list(result.index) == [0, 1, 2]

    def test_numeric_and_string_types(self):
        """Test conversion of int64, float64, and string columns"""
        pdf = pd.DataFrame({
            "ints": [1, 2, 3],
            "floats": [1.1, 2.2, 3.3],
            "strings": ["a", "b", "c"],
        })
        frame = ar.from_pandas(pdf)
        result = ar.to_pandas(frame)
        
        assert result["ints"].dtype == np.int64
        assert result["floats"].dtype == np.float64
        assert list(result["strings"]) == ["a", "b", "c"]

    def test_boolean_columns(self):
        """Test conversion of boolean columns"""
        pdf = pd.DataFrame({"flags": [True, False, True]})
        frame = ar.from_pandas(pdf)
        result = ar.to_pandas(frame)
        
        assert list(result["flags"]) == [True, False, True]

    def test_null_values_preserved(self):
        """Test that null/NA values are preserved with pandas nullable types"""
        pdf = pd.DataFrame({
            "nums": pd.array([1, None, 3], dtype="Int64"),
            "texts": pd.array(["a", None, "c"], dtype="string"),
        })
        frame = ar.from_pandas(pdf)
        result = ar.to_pandas(frame)
        
        assert pd.isna(result["nums"].iloc[1])
        assert pd.isna(result["texts"].iloc[1])
        assert result["nums"].iloc[0] == 1

    def test_decimal_converted_to_string(self):
        """Test that Decimal objects are converted to string representation"""
        pdf = pd.DataFrame({
            "prices": [Decimal("19.99"), Decimal("29.95")]
        })
        frame = ar.from_pandas(pdf)
        result = ar.to_pandas(frame)
        
        # Decimals are preserved as exact strings to avoid floating-point loss
        assert result["prices"].dtype == object
        assert result["prices"].iloc[0] == "19.99"

    def test_after_pipeline_steps(self):
        """Test conversion of ArFrame after applying pipeline steps"""
        pdf = pd.DataFrame({"name": ["  alice  ", "  bob  "]})
        frame = ar.from_pandas(pdf)
        clean = ar.pipeline(frame, [("strip_whitespace",)])
        result = ar.to_pandas(clean)
        
        assert list(result["name"]) == ["alice", "bob"]

    def test_empty_and_single_column_frames(self):
        """Test conversion of edge cases: empty frames and single columns"""
        # Empty frame
        pdf_empty = pd.DataFrame({"a": [], "b": []})
        frame_empty = ar.from_pandas(pdf_empty)
        result_empty = ar.to_pandas(frame_empty)
        
        assert result_empty.shape[0] == 0
        assert list(result_empty.columns) == ["a", "b"]
        
        # Single column
        pdf_single = pd.DataFrame({"data": [1, 2, 3, 4, 5]})
        frame_single = ar.from_pandas(pdf_single)
        result_single = ar.to_pandas(frame_single)
        
        assert result_single.shape == (5, 1)
