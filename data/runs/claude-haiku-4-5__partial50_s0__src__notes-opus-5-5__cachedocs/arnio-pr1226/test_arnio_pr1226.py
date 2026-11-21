import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestToPandasInputValidation:
    """Test input validation for to_pandas()."""

    @pytest.mark.parametrize("bad_copy", [0, 1, None, "yes"])
    def test_copy_must_be_bool(self, bad_copy):
        """copy parameter must be exactly bool, not int, None, or str."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas(frame, copy=bad_copy)

    def test_copy_validation_runs_first(self):
        """copy validation runs before frame validation."""
        with pytest.raises(TypeError, match="copy must be a bool"):
            ar.to_pandas("not_a_frame", copy=1)

    @pytest.mark.parametrize("bad_frame", [
        pd.DataFrame({"a": [1]}),
        {"a": [1]},
        None,
    ])
    def test_frame_must_be_arframe(self, bad_frame):
        """frame parameter must be an ArFrame instance."""
        with pytest.raises(TypeError, match="expects an ArFrame"):
            ar.to_pandas(bad_frame)


class TestToPandasDtypeConversion:
    """Test dtype mappings from ArFrame to pandas."""

    @pytest.mark.parametrize("dtype,expected_dtype", [
        ("Int64", "Int64"),
        ("boolean", "boolean"),
    ])
    def test_nullable_dtypes_preserved(self, dtype, expected_dtype):
        """Nullable INT64 and BOOL become pd.Int64Dtype and pd.BooleanDtype."""
        if dtype == "Int64":
            df = pd.DataFrame({"c": pd.array([1, 2], dtype=dtype)})
        else:
            df = pd.DataFrame({"c": pd.array([True, False], dtype=dtype)})
        
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert str(result["c"].dtype) == expected_dtype

    def test_float64_stays_numpy_float64(self):
        """FLOAT64 stays as numpy float64."""
        df = pd.DataFrame({"f": [1.5, 2.5, 3.5]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["f"].dtype == np.float64

    def test_string_becomes_string_dtype(self):
        """String columns become pd.StringDtype()."""
        df = pd.DataFrame({"s": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result["s"].dtype, pd.StringDtype)


class TestToPandasNullHandling:
    """Test null value handling."""

    @pytest.mark.parametrize("dtype", ["Int64", "boolean"])
    def test_nullable_dtype_nulls_become_pd_na(self, dtype):
        """INT64 and BOOL nulls are represented as pd.NA."""
        if dtype == "Int64":
            df = pd.DataFrame({"c": pd.array([1, None, 3], dtype=dtype)})
        else:
            df = pd.DataFrame({"c": pd.array([True, None, False], dtype=dtype)})
        
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["c"][1] is pd.NA

    def test_float64_nulls_become_nan(self):
        """FLOAT64 nulls are represented as np.nan."""
        df = pd.DataFrame({"f": [1.5, np.nan, 3.5]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert np.isnan(result["f"][1])

    def test_string_nulls_become_pd_na(self):
        """String nulls are represented as pd.NA."""
        df = pd.DataFrame({"s": ["a", None, "c"]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["s"][1] is pd.NA


class TestToPandasAttrs:
    """Test attrs metadata restoration."""

    def test_attrs_restored_as_deep_copy(self):
        """attrs from frame are restored as a deep copy."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.attrs = {"src": {"key": "value"}}
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.attrs == {"src": {"key": "value"}}
        assert result.attrs is not df.attrs

    def test_attrs_mutation_isolation(self):
        """Mutating result.attrs does not affect future conversions."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.attrs = {"nested": {"k": 1}}
        frame = ar.from_pandas(df)
        
        result1 = ar.to_pandas(frame)
        result1.attrs["nested"]["k"] = 999
        
        result2 = ar.to_pandas(frame)
        assert result2.attrs["nested"]["k"] == 1


class TestToPandasShape:
    """Test shape, structure, and copy behavior."""

    def test_empty_dataframe_zero_zero(self):
        """Empty DataFrame (0, 0) round-trips correctly."""
        frame = ar.from_pandas(pd.DataFrame())
        result = ar.to_pandas(frame)
        
        assert result.shape == (0, 0)
        assert isinstance(result.index, pd.RangeIndex)

    def test_zero_rows_preserves_column_dtypes(self):
        """Zero-row DataFrame preserves column dtypes."""
        df = pd.DataFrame({
            "i": pd.Series([], dtype="Int64"),
            "f": pd.Series([], dtype="float64"),
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.shape[0] == 0
        assert len(result.columns) == 2
        assert str(result["i"].dtype) == "Int64"
        assert result["f"].dtype == np.float64

    def test_column_order_preserved(self):
        """Column order is preserved through conversion."""
        df = pd.DataFrame({"z": [1], "a": [2], "m": [3]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.columns.tolist() == ["z", "a", "m"]

    def test_copy_false_and_copy_true_equivalent(self):
        """copy=False and copy=True produce equivalent DataFrames."""
        df = pd.DataFrame({
            "i": pd.array([1, None, 3], dtype="Int64"),
            "f": [1.5, np.nan, 3.5],
            "b": pd.array([True, None, False], dtype="boolean"),
            "s": ["a", None, "c"]
        })
        frame = ar.from_pandas(df)
        
        result_shallow = ar.to_pandas(frame, copy=False)
        result_deep = ar.to_pandas(frame, copy=True)
        
        pd.testing.assert_frame_equal(result_shallow, result_deep)
