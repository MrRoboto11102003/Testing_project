import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestToPandas:
    """Test suite for arnio.convert.to_pandas"""

    def test_basic_conversion_from_dict(self):
        """Test basic conversion of ArFrame to pandas DataFrame"""
        frame = ar.from_dict({"x": [1, 2, 3], "y": [4, 5, 6]})
        df = ar.to_pandas(frame)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["x", "y"]

    def test_int64_dtype_conversion(self):
        """Test INT64 columns convert to nullable Int64Dtype"""
        frame = ar.from_dict({"id": [1, 2, 3]})
        df = ar.to_pandas(frame)

        assert df["id"].dtype == pd.Int64Dtype()
        assert list(df["id"]) == [1, 2, 3]

    def test_float64_dtype_conversion(self):
        """Test FLOAT64 columns convert to native float64"""
        frame = ar.from_dict({"price": [1.5, 2.5, 3.5]})
        df = ar.to_pandas(frame)

        assert df["price"].dtype == np.float64
        assert np.allclose(df["price"].values, [1.5, 2.5, 3.5])

    def test_bool_dtype_conversion(self):
        """Test BOOL columns convert to nullable BooleanDtype"""
        frame = ar.from_dict({"active": [True, False, True]})
        df = ar.to_pandas(frame)

        assert df["active"].dtype == pd.BooleanDtype()
        assert list(df["active"]) == [True, False, True]

    def test_string_dtype_conversion(self):
        """Test STRING columns convert to StringDtype"""
        frame = ar.from_dict({"name": ["Alice", "Bob", "Charlie"]})
        df = ar.to_pandas(frame)

        assert df["name"].dtype == pd.StringDtype()
        assert list(df["name"]) == ["Alice", "Bob", "Charlie"]

    def test_null_handling_with_nullable_int(self):
        """Test null values in INT64 columns map to pd.NA"""
        df_in = pd.DataFrame({"val": pd.array([1, 2, None], dtype="Int64")})
        frame = ar.from_pandas(df_in)
        df = ar.to_pandas(frame)

        assert df["val"].iloc[0] == 1
        assert df["val"].iloc[1] == 2
        assert pd.isna(df["val"].iloc[2])
        assert df["val"].dtype == pd.Int64Dtype()

    def test_null_handling_with_nullable_bool(self):
        """Test null values in BOOL columns map to pd.NA"""
        df_in = pd.DataFrame({"flag": pd.array([True, False, None], dtype="boolean")})
        frame = ar.from_pandas(df_in)
        df = ar.to_pandas(frame)

        assert df["flag"].iloc[0] == True
        assert df["flag"].iloc[1] == False
        assert pd.isna(df["flag"].iloc[2])
        assert df["flag"].dtype == pd.BooleanDtype()

    def test_null_handling_with_float(self):
        """Test null values in FLOAT64 columns map to np.nan"""
        df_in = pd.DataFrame({"score": [1.0, np.nan, 3.0]})
        frame = ar.from_pandas(df_in)
        df = ar.to_pandas(frame)

        assert df["score"].iloc[0] == 1.0
        assert np.isnan(df["score"].iloc[1])
        assert df["score"].iloc[2] == 3.0
        assert df["score"].dtype == np.float64

    def test_null_handling_with_string(self):
        """Test null values in STRING columns map to pd.NA"""
        df_in = pd.DataFrame({"text": pd.array(["a", None, "c"], dtype="string")})
        frame = ar.from_pandas(df_in)
        df = ar.to_pandas(frame)

        assert df["text"].iloc[0] == "a"
        assert pd.isna(df["text"].iloc[1])
        assert df["text"].iloc[2] == "c"
        assert df["text"].dtype == pd.StringDtype()

    def test_copy_parameter_false_default(self):
        """Test default behavior with copy=False"""
        frame = ar.from_dict({"x": [1, 2, 3]})
        df = ar.to_pandas(frame)  # copy defaults to False

        assert df.shape == (3, 1)
        assert df["x"].dtype == pd.Int64Dtype()

    def test_copy_parameter_true_returns_copy(self):
        """Test copy=True returns defensive pandas-owned copies"""
        frame = ar.from_dict({"val": [1.0, 2.0, 3.0]})
        df = ar.to_pandas(frame, copy=True)

        assert df.shape == (3, 1)
        assert df["val"].dtype == np.float64

    def test_empty_frame_conversion(self):
        """Test conversion of empty ArFrame"""
        frame = ar.from_dict({})
        df = ar.to_pandas(frame)

        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 0

    def test_attrs_metadata_restoration(self):
        """Test that attrs from original DataFrame are restored"""
        df_in = pd.DataFrame({"x": [1, 2, 3]})
        df_in.attrs = {"source": "test_data", "version": "1.0"}

        frame = ar.from_pandas(df_in)
        df = ar.to_pandas(frame)

        assert df.attrs == {"source": "test_data", "version": "1.0"}

    def test_copy_parameter_not_bool_raises_typeerror(self):
        """Test TypeError when copy parameter is not a boolean"""
        frame = ar.from_dict({"x": [1, 2, 3]})

        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy="true")

        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy=1)

    def test_non_arframe_input_raises_typeerror(self):
        """Test TypeError when input is not an ArFrame"""
        df = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(TypeError, match="expects an ArFrame"):
            ar.to_pandas(df)

    def test_mixed_dtype_conversion(self):
        """Test conversion with mixed column types"""
        df_in = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "score": [1.5, 2.5, 3.5],
            "active": [True, False, True]
        })

        frame = ar.from_pandas(df_in)
        df = ar.to_pandas(frame)

        assert df["id"].dtype == pd.Int64Dtype()
        assert df["name"].dtype == pd.StringDtype()
        assert df["score"].dtype == np.float64
        assert df["active"].dtype == pd.BooleanDtype()
        assert df.shape == (3, 4)
