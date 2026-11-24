import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestToPandasBasicDtypes:
    """Test basic dtype conversions."""
    
    def test_int64_no_nulls(self):
        """INT64 converts to Int64 dtype."""
        df = pd.DataFrame({"col": pd.array([1, 2, 3], dtype="Int64")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert str(result["col"].dtype) == "Int64"
        assert result["col"].tolist() == [1, 2, 3]
    
    def test_int64_with_nulls(self):
        """INT64 with nulls represented as pd.NA."""
        df = pd.DataFrame({"col": pd.array([1, None, 3], dtype="Int64")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert str(result["col"].dtype) == "Int64"
        assert result["col"].isna().tolist() == [False, True, False]
        assert result["col"].iloc[1] is pd.NA
    
    def test_float64_no_nulls(self):
        """FLOAT64 stays numpy float64."""
        df = pd.DataFrame({"col": np.array([1.5, 2.5, 3.5], dtype="float64")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["col"].dtype == np.float64
        assert result["col"].tolist() == [1.5, 2.5, 3.5]
    
    def test_float64_with_nulls(self):
        """FLOAT64 with nulls represented as np.nan."""
        df = pd.DataFrame({"col": np.array([1.5, np.nan, 3.5], dtype="float64")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["col"].dtype == np.float64
        assert result["col"].isna().tolist() == [False, True, False]
        assert np.isnan(result["col"].iloc[1])
    
    def test_bool_no_nulls(self):
        """BOOL converts to boolean dtype."""
        df = pd.DataFrame({"col": pd.array([True, False, True], dtype="boolean")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert str(result["col"].dtype) == "boolean"
        assert result["col"].tolist() == [True, False, True]
    
    def test_bool_with_nulls(self):
        """BOOL with nulls represented as pd.NA."""
        df = pd.DataFrame({"col": pd.array([True, None, False], dtype="boolean")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert str(result["col"].dtype) == "boolean"
        assert result["col"].isna().tolist() == [False, True, False]
        assert result["col"].iloc[1] is pd.NA
    
    def test_string_no_nulls(self):
        """STRING uses StringDtype."""
        df = pd.DataFrame({"col": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result["col"].dtype, pd.StringDtype)
        assert result["col"].tolist() == ["a", "b", "c"]
    
    def test_string_with_nulls(self):
        """STRING with nulls represented as pd.NA."""
        df = pd.DataFrame({"col": ["a", None, "c"]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result["col"].dtype, pd.StringDtype)
        assert result["col"].isna().tolist() == [False, True, False]
        assert result["col"].iloc[1] is pd.NA


class TestToPandasParameterValidation:
    """Test parameter validation."""
    
    def test_copy_not_bool_raises(self):
        """copy must be exactly bool, not int or None."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy=1)
    
    def test_frame_not_arframe_raises(self):
        """frame must be ArFrame, not DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="to_pandas\\(\\) expects an ArFrame"):
            ar.to_pandas(df)
    
    def test_copy_checked_before_frame(self):
        """copy is validated before frame type."""
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas("not_frame", copy=None)


class TestToPandasCopyBehavior:
    """Test copy parameter behavior."""
    
    def test_copy_true_independent(self):
        """copy=True produces independent data."""
        df = pd.DataFrame({"f": np.array([1.5, 2.5, 3.5], dtype="float64")})
        frame = ar.from_pandas(df)
        
        r1 = ar.to_pandas(frame, copy=True)
        r1["f"].iloc[0] = 99.0
        
        r2 = ar.to_pandas(frame, copy=False)
        assert r2["f"].iloc[0] == 1.5


class TestToPandasAttrs:
    """Test attrs preservation."""
    
    def test_attrs_deep_copied(self):
        """Attrs from from_pandas are deep copied to result."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.attrs = {"nested": {"k": 1}}
        
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.attrs == {"nested": {"k": 1}}
        result.attrs["nested"]["k"] = 99
        
        result2 = ar.to_pandas(frame)
        assert result2.attrs["nested"]["k"] == 1


class TestToPandasEdgeCases:
    """Test edge cases."""
    
    def test_empty_frame(self):
        """Empty frame stays empty with RangeIndex."""
        df = pd.DataFrame()
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.shape == (0, 0)
        assert isinstance(result.index, pd.RangeIndex)
    
    def test_zero_rows_multiple_dtypes(self):
        """Zero rows preserves column dtypes."""
        df = pd.DataFrame({
            "i": pd.array([], dtype="Int64"),
            "f": np.array([], dtype="float64"),
            "b": pd.array([], dtype="boolean"),
            "s": pd.array([], dtype=pd.StringDtype())
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.shape == (0, 4)
        assert str(result["i"].dtype) == "Int64"
        assert result["f"].dtype == np.float64
        assert str(result["b"].dtype) == "boolean"
        assert isinstance(result["s"].dtype, pd.StringDtype)
