import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Test suite for arnio.cleaning.filter_rows function."""
    
    @pytest.fixture
    def df_integers(self):
        """DataFrame with integer column."""
        return pd.DataFrame({"age": [20, 25, 30, 35, 40]})
    
    @pytest.fixture
    def df_floats(self):
        """DataFrame with float column."""
        return pd.DataFrame({"score": [1.5, 2.5, 3.5, 4.5, 5.5]})
    
    @pytest.fixture
    def df_strings(self):
        """DataFrame with string column."""
        return pd.DataFrame({"name": ["Alice", "Bob", "Charlie", "Diana", "Eve"]})
    
    @pytest.fixture
    def df_bools(self):
        """DataFrame with boolean column."""
        return pd.DataFrame({"active": [True, False, True, False, True]})
    
    def test_filter_greater_than(self, df_integers):
        """Test > operator filters correctly."""
        result = ar.filter_rows(df_integers, column="age", op=">", value=30)
        assert len(result) == 2
        assert list(result["age"]) == [35, 40]
    
    def test_filter_less_than(self, df_integers):
        """Test < operator filters correctly."""
        result = ar.filter_rows(df_integers, column="age", op="<", value=30)
        assert len(result) == 2
        assert list(result["age"]) == [20, 25]
    
    def test_filter_greater_equal(self, df_integers):
        """Test >= operator filters correctly."""
        result = ar.filter_rows(df_integers, column="age", op=">=", value=30)
        assert len(result) == 3
        assert list(result["age"]) == [30, 35, 40]
    
    def test_filter_less_equal(self, df_integers):
        """Test <= operator filters correctly."""
        result = ar.filter_rows(df_integers, column="age", op="<=", value=30)
        assert len(result) == 3
        assert list(result["age"]) == [20, 25, 30]
    
    def test_filter_equal(self, df_integers):
        """Test == operator filters correctly."""
        result = ar.filter_rows(df_integers, column="age", op="==", value=30)
        assert len(result) == 1
        assert result["age"].iloc[0] == 30
    
    def test_filter_not_equal(self, df_integers):
        """Test != operator filters correctly."""
        result = ar.filter_rows(df_integers, column="age", op="!=", value=30)
        assert len(result) == 4
        assert 30 not in result["age"].values
    
    def test_filter_float_column(self, df_floats):
        """Test filtering float column works."""
        result = ar.filter_rows(df_floats, column="score", op=">=", value=3.0)
        assert len(result) == 3
        assert all(result["score"] >= 3.0)
    
    def test_filter_string_column(self, df_strings):
        """Test filtering string column works."""
        result = ar.filter_rows(df_strings, column="name", op="==", value="Alice")
        assert len(result) == 1
        assert result["name"].iloc[0] == "Alice"
    
    def test_filter_boolean_column(self, df_bools):
        """Test filtering boolean column works."""
        result = ar.filter_rows(df_bools, column="active", op="==", value=True)
        assert len(result) == 3
        assert all(result["active"])
    
    def test_filter_returns_dataframe_type_when_given_dataframe(self, df_integers):
        """Result type matches input: DataFrame returns DataFrame."""
        result = ar.filter_rows(df_integers, column="age", op=">", value=25)
        assert isinstance(result, pd.DataFrame)
    
    def test_filter_returns_arframe_type_when_given_arframe(self, df_integers):
        """Result type matches input: ArFrame returns ArFrame."""
        frame = ar.from_pandas(df_integers)
        result = ar.filter_rows(frame, column="age", op=">", value=25)
        assert isinstance(result, ar.ArFrame)
    
    def test_filter_no_matches_returns_empty_frame(self, df_integers):
        """Filter with no matches returns empty frame with preserved columns."""
        result = ar.filter_rows(df_integers, column="age", op=">", value=100)
        assert len(result) == 0
        assert list(result.columns) == ["age"]
    
    def test_filter_invalid_column_raises_error(self, df_integers):
        """Invalid column name raises KeyError or ValueError."""
        with pytest.raises((KeyError, ValueError)):
            ar.filter_rows(df_integers, column="missing_col", op=">", value=25)
    
    def test_filter_invalid_operator_raises_error(self, df_integers):
        """Invalid operator raises ValueError."""
        with pytest.raises(ValueError):
            ar.filter_rows(df_integers, column="age", op="invalid_op", value=25)
    
    def test_filter_empty_dataframe(self):
        """Filter on empty DataFrame returns empty result."""
        empty_df = pd.DataFrame({"value": pd.Series([], dtype="int64")})
        result = ar.filter_rows(empty_df, column="value", op=">", value=0)
        assert len(result) == 0
        assert list(result.columns) == ["value"]
