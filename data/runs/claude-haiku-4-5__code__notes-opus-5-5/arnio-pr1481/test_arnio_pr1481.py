import pytest
import pandas as pd
import numpy as np
import re
import arnio as ar


class TestFilterRows:
    """Tests for arnio.cleaning.filter_rows"""

    @pytest.fixture
    def df(self):
        """Sample DataFrame for testing"""
        return pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["x", "y", "z", "w", "v"],
            "c": [1.5, 2.5, 3.5, 4.5, 5.5]
        })

    @pytest.fixture
    def arf(self, df):
        """Sample ArFrame for testing"""
        return ar.from_pandas(df)

    def test_all_operators(self, df):
        """Test all six comparison operators"""
        assert list(ar.filter_rows(df, "a", ">", 2)["a"]) == [3, 4, 5]
        assert list(ar.filter_rows(df, "a", "<", 3)["a"]) == [1, 2]
        assert list(ar.filter_rows(df, "a", ">=", 3)["a"]) == [3, 4, 5]
        assert list(ar.filter_rows(df, "a", "<=", 2)["a"]) == [1, 2]
        assert list(ar.filter_rows(df, "a", "==", 2)["a"]) == [2]
        assert list(ar.filter_rows(df, "a", "!=", 2)["a"]) == [1, 3, 4, 5]

    def test_type_preservation_and_index(self, df, arf):
        """Input type matches output type; index handling differs for ArFrame vs DataFrame"""
        df_result = ar.filter_rows(df, "a", ">", 2)
        assert isinstance(df_result, pd.DataFrame)
        assert not isinstance(df_result, ar.ArFrame)
        assert list(df_result.index) == [2, 3, 4]  # DataFrame preserves original index
        
        arf_result = ar.filter_rows(arf, "a", ">", 2)
        assert isinstance(arf_result, ar.ArFrame)
        arf_df = ar.to_pandas(arf_result)
        assert list(arf_df.index) == [0, 1, 2]  # ArFrame has reset index

    def test_string_and_float_columns(self, df):
        """Filter on string and float columns"""
        str_result = ar.filter_rows(df, "b", "==", "y")
        assert len(str_result) == 1
        assert str_result["b"].iloc[0] == "y"
        
        float_result = ar.filter_rows(df, "c", ">", 2.5)
        assert len(float_result) == 3
        assert list(float_result["c"]) == [3.5, 4.5, 5.5]

    def test_no_matches_and_all_matches(self, df):
        """Test edge cases: no matches and all matches"""
        no_match = ar.filter_rows(df, "a", ">", 100)
        assert len(no_match) == 0
        assert list(no_match.columns) == ["a", "b", "c"]
        
        all_match = ar.filter_rows(df, "a", ">", 0)
        assert len(all_match) == 5

    def test_immutability(self, df):
        """Input DataFrame is not mutated"""
        original = df.copy()
        ar.filter_rows(df, "a", ">", 2)
        assert df.equals(original)

    def test_error_invalid_frame_type(self):
        """TypeError when frame is not DataFrame or ArFrame"""
        with pytest.raises(TypeError, match=re.escape("frame must be an ArFrame or a pandas DataFrame")):
            ar.filter_rows({"a": [1]}, "a", ">", 1)

    def test_error_column_type_and_format(self, df):
        """TypeError when column is not a non-empty string"""
        with pytest.raises(TypeError, match=r"filter_rows: column must be a non-empty string, got 'int'"):
            ar.filter_rows(df, 5, ">", 1)
        
        with pytest.raises(TypeError, match=r"filter_rows: column must be a non-empty string, got 'str'"):
            ar.filter_rows(df, "", ">", 1)

    def test_error_op_type(self, df):
        """TypeError when op is not a string"""
        with pytest.raises(TypeError, match=r"filter_rows: op must be a string, got 'NoneType'"):
            ar.filter_rows(df, "a", None, 1)

    def test_error_unsupported_operator(self, df):
        """ValueError when operator is not supported"""
        with pytest.raises(ValueError, match=re.escape("Unsupported operator: =")):
            ar.filter_rows(df, "a", "=", 1)

    def test_error_unknown_column(self, df):
        """ValueError when column does not exist"""
        with pytest.raises(ValueError, match=re.escape("Unknown column: zzz")):
            ar.filter_rows(df, "zzz", ">", 1)

    def test_error_non_scalar_value(self, df):
        """TypeError when value is not scalar"""
        with pytest.raises(TypeError, match=re.escape("filter_rows value must be a scalar")):
            ar.filter_rows(df, "a", ">", [1, 2])

    def test_error_comparison_type_mismatch(self):
        """TypeError when comparison is not supported for the data type"""
        df = pd.DataFrame({"s": ["a", "b"]})
        with pytest.raises(TypeError, match=r"filter_rows: cannot compare column 's' with value 1 using operator '>':"):
            ar.filter_rows(df, "s", ">", 1)

    def test_error_validation_order(self, df):
        """Validation order: op before column existence, column before value type"""
        # Op validated before column existence
        with pytest.raises(ValueError, match=re.escape("Unsupported operator: =")):
            ar.filter_rows(df, "missing", "=", 1)
        
        # Column existence checked before value type
        with pytest.raises(ValueError, match=re.escape("Unknown column: missing")):
            ar.filter_rows(df, "missing", ">", [1])

    def test_nan_with_not_equal(self):
        """NaN != value returns True"""
        df = pd.DataFrame({"f": [1.0, float("nan"), 3.0]})
        result = ar.filter_rows(df, "f", "!=", 1.0)
        assert 1 in result.index

    def test_nan_with_comparison(self):
        """NaN in comparison operators returns False (filtered out)"""
        df = pd.DataFrame({"f": [1.0, float("nan"), 3.0]})
        result = ar.filter_rows(df, "f", ">", 0)
        assert len(result) == 2
        assert 1 not in result.index
