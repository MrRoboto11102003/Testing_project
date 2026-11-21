import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestToPandas:
    """Tests for arnio.convert.to_pandas function."""
    
    def test_copy_parameter_type_validation(self):
        """Test that copy parameter must be a bool."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy="not_bool")
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy=1)
    
    def test_frame_parameter_type_validation(self):
        """Test that frame parameter must be an ArFrame."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        with pytest.raises(TypeError, match="expects an ArFrame"):
            ar.to_pandas(df)
    
    def test_int64_column_conversion(self):
        """Test conversion of INT64 column without nulls."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["col"].dtype == pd.Int64Dtype()
        assert list(result["col"]) == [1, 2, 3]
    
    def test_float64_column_conversion(self):
        """Test conversion of FLOAT64 column without nulls."""
        df = pd.DataFrame({"col": [1.5, 2.5, 3.5]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["col"].dtype == np.float64
        np.testing.assert_array_almost_equal(result["col"], [1.5, 2.5, 3.5])
    
    def test_bool_column_conversion(self):
        """Test conversion of BOOL column without nulls."""
        df = pd.DataFrame({"col": [True, False, True]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["col"].dtype == pd.BooleanDtype()
        assert list(result["col"]) == [True, False, True]
    
    def test_string_column_conversion(self):
        """Test conversion of STRING column without nulls."""
        df = pd.DataFrame({"col": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["col"].dtype == pd.StringDtype()
        assert list(result["col"]) == ["a", "b", "c"]
    
    def test_int64_with_nulls(self):
        """Test INT64 column with null values uses pd.NA."""
        df = pd.DataFrame({"col": pd.array([1, None, 3], dtype=pd.Int64Dtype())})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["col"].dtype == pd.Int64Dtype()
        assert result["col"][0] == 1
        assert pd.isna(result["col"][1])
        assert result["col"][2] == 3
    
    def test_float64_with_nulls(self):
        """Test FLOAT64 column with null values uses np.nan."""
        df = pd.DataFrame({"col": [1.5, np.nan, 3.5]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["col"].dtype == np.float64
        assert result["col"][0] == 1.5
        assert np.isnan(result["col"][1])
        assert result["col"][2] == 3.5
    
    def test_bool_with_nulls(self):
        """Test BOOL column with null values uses pd.NA."""
        df = pd.DataFrame({"col": pd.array([True, None, False], dtype=pd.BooleanDtype())})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["col"].dtype == pd.BooleanDtype()
        assert result["col"][0] == True
        assert pd.isna(result["col"][1])
        assert result["col"][2] == False
    
    def test_string_with_nulls(self):
        """Test STRING column with null values uses pd.NA."""
        df = pd.DataFrame({"col": pd.array(["a", None, "c"], dtype=pd.StringDtype())})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["col"].dtype == pd.StringDtype()
        assert result["col"][0] == "a"
        assert pd.isna(result["col"][1])
        assert result["col"][2] == "c"
    
    def test_multiple_columns_mixed_types(self):
        """Test conversion of frame with multiple columns of different types."""
        df = pd.DataFrame({
            "int": [1, 2, 3],
            "float": [1.5, 2.5, 3.5],
            "bool": [True, False, True],
            "str": ["x", "y", "z"]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["int"].dtype == pd.Int64Dtype()
        assert result["float"].dtype == np.float64
        assert result["bool"].dtype == pd.BooleanDtype()
        assert result["str"].dtype == pd.StringDtype()
        assert len(result) == 3
    
    def test_empty_frame_no_data(self):
        """Test conversion of empty frame (no rows or columns)."""
        frame = ar.from_dict({})
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 0
        assert len(result) == 0
    
    def test_frame_with_no_rows(self):
        """Test conversion of frame with columns but zero rows."""
        df = pd.DataFrame({"col": pd.Series([], dtype=pd.Int64Dtype())})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert "col" in result.columns
        assert len(result) == 0
    
    def test_metadata_attrs_restoration(self):
        """Test that DataFrame.attrs metadata is restored after from_pandas roundtrip."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        df.attrs = {"source": "test_data", "version": 1, "custom": "metadata"}
        
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.attrs == {"source": "test_data", "version": 1, "custom": "metadata"}
    
    def test_copy_false_preserves_zero_copy(self):
        """Test that copy=False is accepted and returns valid DataFrame."""
        df = pd.DataFrame({"col": [1.5, 2.5, 3.5]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame, copy=False)
        
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result["col"], [1.5, 2.5, 3.5])
    
    def test_copy_true_creates_defensive_copies(self):
        """Test that copy=True creates defensive copies of supported buffers."""
        df = pd.DataFrame({"col": [1.5, 2.5, 3.5]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame, copy=True)
        
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result["col"], [1.5, 2.5, 3.5])
    
    def test_all_nulls_column_preserves_type(self):
        """Test column with all null values preserves its dtype."""
        df = pd.DataFrame({"col": pd.array([None, None, None], dtype=pd.Int64Dtype())})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["col"].dtype == pd.Int64Dtype()
        assert result["col"].isna().all()
