import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestToPandas:
    """Test suite for arnio.convert.to_pandas"""

    def test_to_pandas_basic_conversion(self):
        """Test basic ArFrame to DataFrame conversion"""
        data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1", "col2"]
        assert len(result) == 3

    def test_to_pandas_preserves_column_names(self):
        """Test that column names are preserved during conversion"""
        data = {"name": [1, 2], "age": [25, 30]}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert list(result.columns) == ["name", "age"]

    def test_to_pandas_preserves_attrs_metadata(self):
        """Test that attrs metadata is restored from original DataFrame"""
        data = {"col1": [1, 2, 3]}
        df = pd.DataFrame(data)
        df.attrs = {"source": "test_data", "version": 2}
        
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.attrs == {"source": "test_data", "version": 2}

    def test_to_pandas_copy_false_default(self):
        """Test that copy=False is the default"""
        data = {"col1": [1.0, 2.0, 3.0]}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)

    def test_to_pandas_copy_true(self):
        """Test that copy=True creates defensive copies"""
        data = {"col1": [1, 2, 3]}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame, copy=True)
        
        assert isinstance(result, pd.DataFrame)
        assert (result["col1"].values == [1, 2, 3]).all()

    def test_to_pandas_invalid_copy_type_string(self):
        """Test that non-bool copy parameter raises TypeError"""
        data = {"col1": [1, 2, 3]}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy="true")

    def test_to_pandas_invalid_copy_type_int(self):
        """Test that int copy parameter raises TypeError"""
        data = {"col1": [1, 2, 3]}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy=1)

    def test_to_pandas_invalid_frame_type_pandas_dataframe(self):
        """Test that pandas DataFrame input raises TypeError"""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="to_pandas\\(\\) expects an ArFrame"):
            ar.to_pandas(df)

    def test_to_pandas_invalid_frame_type_dict(self):
        """Test that dict input raises TypeError"""
        with pytest.raises(TypeError, match="to_pandas\\(\\) expects an ArFrame"):
            ar.to_pandas({"col1": [1, 2, 3]})

    def test_to_pandas_with_int64_dtype(self):
        """Test conversion preserves INT64 dtype"""
        data = {"col1": [1, 2, 3]}
        df = pd.DataFrame(data, dtype="int64")
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert pd.api.types.is_integer_dtype(result["col1"])
        assert (result["col1"].values == [1, 2, 3]).all()

    def test_to_pandas_with_float64_dtype(self):
        """Test conversion preserves FLOAT64 dtype"""
        data = {"col1": [1.5, 2.5, 3.7]}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert pd.api.types.is_float_dtype(result["col1"])

    def test_to_pandas_with_bool_dtype(self):
        """Test conversion preserves BOOL dtype"""
        data = {"col1": [True, False, True]}
        df = pd.DataFrame(data, dtype="bool")
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert pd.api.types.is_bool_dtype(result["col1"])

    def test_to_pandas_with_nulls_in_int_column(self):
        """Test handling of nulls in INT64 columns uses pd.NA"""
        data = {"col1": pd.array([1, pd.NA, 3], dtype="Int64")}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert pd.isna(result["col1"].iloc[1])
        assert result["col1"].iloc[0] == 1
        assert result["col1"].iloc[2] == 3

    def test_to_pandas_with_nulls_in_float_column(self):
        """Test handling of nulls in FLOAT64 columns uses np.nan"""
        data = {"col1": [1.0, np.nan, 3.0]}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert np.isnan(result["col1"].iloc[1])
        assert result["col1"].iloc[0] == 1.0

    def test_to_pandas_mixed_dtypes_and_nulls(self):
        """Test conversion with mixed dtypes and nulls"""
        data = {
            "ints": pd.array([1, pd.NA, 3], dtype="Int64"),
            "floats": [1.0, np.nan, 3.0],
            "bools": pd.array([True, pd.NA, False], dtype="boolean"),
            "strs": pd.array(["a", pd.NA, "c"], dtype="string")
        }
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert len(result) == 3
        assert pd.isna(result["ints"].iloc[1])
        assert np.isnan(result["floats"].iloc[1])
        assert pd.isna(result["bools"].iloc[1])
        assert pd.isna(result["strs"].iloc[1])
