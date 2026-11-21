import pytest
import numpy as np
import pandas as pd
import arnio as ar


class TestFilterRowsOperators:
    """Test all supported operators."""
    
    def test_all_operators(self):
        """Test >, <, >=, <=, ==, != operators."""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        
        assert ar.filter_rows(pdf, "a", ">", 1)["a"].tolist() == [2, 3]
        assert ar.filter_rows(pdf, "a", "<", 2)["a"].tolist() == [1]
        assert ar.filter_rows(pdf, "a", ">=", 2)["a"].tolist() == [2, 3]
        assert ar.filter_rows(pdf, "a", "<=", 2)["a"].tolist() == [1, 2]
        assert ar.filter_rows(pdf, "a", "==", 2)["a"].tolist() == [2]
        assert ar.filter_rows(pdf, "a", "!=", 2)["a"].tolist() == [1, 3]


class TestFilterRowsTypes:
    """Test return types with ArFrame and pandas."""
    
    def test_pandas_dataframe_input_output(self):
        """pandas DataFrame input returns pandas DataFrame with preserved index."""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        result = ar.filter_rows(pdf, "a", ">", 1)
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == [1, 2]
    
    def test_arframe_input_output(self):
        """ArFrame input returns ArFrame with reset index."""
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(pdf)
        result = ar.filter_rows(frame, "a", ">", 1)
        assert isinstance(result, ar.ArFrame)
        assert result.columns == ["a", "b"]
        result_pdf = ar.to_pandas(result)
        assert list(result_pdf.index) == [0, 1]


class TestFilterRowsValidation:
    """Test input validation and error handling."""
    
    def test_invalid_frame_type(self):
        """Invalid frame type raises TypeError."""
        with pytest.raises(TypeError, match="must be an ArFrame or a pandas DataFrame"):
            ar.filter_rows({"a": [1]}, "a", ">", 1)
    
    def test_invalid_column_type_or_empty(self):
        """Invalid column type or empty column name raises TypeError."""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(pdf, 5, ">", 1)
        
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(pdf, "", ">", 1)
        
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(pdf, "   ", ">", 1)
    
    def test_invalid_op_type(self):
        """Non-string op raises TypeError."""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="op must be a string"):
            ar.filter_rows(pdf, "a", None, 1)
    
    def test_unsupported_operator(self):
        """Unsupported operator raises ValueError."""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(pdf, "a", "=", 1)
    
    def test_missing_column(self):
        """Missing column raises ValueError."""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(pdf, "missing", ">", 1)
    
    def test_non_scalar_value(self):
        """Non-scalar value raises TypeError."""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="filter_rows value must be a scalar"):
            ar.filter_rows(pdf, "a", ">", [1, 2])
    
    def test_incompatible_comparison(self):
        """Incompatible type comparison raises TypeError."""
        pdf = pd.DataFrame({"s": ["a", "b", "c"]})
        with pytest.raises(TypeError, match="cannot compare column"):
            ar.filter_rows(pdf, "s", ">", 1)


class TestFilterRowsEdgeCases:
    """Test edge cases and special behaviors."""
    
    def test_no_matching_rows(self):
        """Filter with no matches returns empty frame with same columns."""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        result = ar.filter_rows(pdf, "a", ">", 100)
        assert len(result) == 0
        assert result.columns.tolist() == ["a"]
    
    def test_all_matching_rows(self):
        """Filter matching all rows."""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        result = ar.filter_rows(pdf, "a", ">", 0)
        assert result["a"].tolist() == [1, 2, 3]
    
    def test_input_immutability(self):
        """Input frame should not be mutated."""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        original = pdf.copy()
        ar.filter_rows(pdf, "a", ">", 1)
        assert pdf.equals(original)
    
    def test_string_column_filtering(self):
        """Filter on string column with equality and comparison."""
        pdf = pd.DataFrame({"s": ["x", "y", "z"]})
        result = ar.filter_rows(pdf, "s", "==", "y")
        assert result["s"].tolist() == ["y"]
        
        result2 = ar.filter_rows(pdf, "s", ">=", "y")
        assert result2["s"].tolist() == ["y", "z"]
    
    def test_float_nan_handling(self):
        """NaN in float columns with != keeps NaN, other ops drop it."""
        pdf = pd.DataFrame({"f": [1.0, float("nan"), 3.0]})
        
        # NaN != 1.0 is True, so NaN row is included
        result = ar.filter_rows(pdf, "f", "!=", 1.0)
        assert list(result.index) == [1, 2]
        
        # NaN > 0.0 is False (fillna(False) drops it)
        result2 = ar.filter_rows(pdf, "f", ">", 0.0)
        assert result2["f"].tolist() == [1.0, 3.0]
