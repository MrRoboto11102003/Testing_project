import pytest
import pandas as pd
import numpy as np
import re
import arnio as ar


def test_filter_rows_greater_than():
    """Test > operator with numeric column."""
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = ar.filter_rows(pdf, "a", ">", 1)
    assert list(result["a"]) == [2, 3]
    assert list(result.index) == [1, 2]  # pandas preserves original index


def test_filter_rows_all_operators():
    """Test all supported operators."""
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    
    assert list(ar.filter_rows(pdf, "a", "<", 2)["a"]) == [1]
    assert list(ar.filter_rows(pdf, "a", "<=", 2)["a"]) == [1, 2]
    assert list(ar.filter_rows(pdf, "a", ">=", 2)["a"]) == [2, 3]
    assert list(ar.filter_rows(pdf, "a", "==", 2)["a"]) == [2]
    assert list(ar.filter_rows(pdf, "a", "!=", 2)["a"]) == [1, 3]


def test_filter_rows_arframe_returns_arframe_with_reset_index():
    """Test that ArFrame input returns ArFrame with reset index."""
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    fr = ar.from_pandas(pdf)
    result = ar.filter_rows(fr, "a", ">", 1)
    
    assert isinstance(result, ar.ArFrame)
    result_pdf = ar.to_pandas(result)
    assert list(result_pdf["a"]) == [2, 3]
    assert list(result_pdf.index) == [0, 1]  # ArFrame resets index


def test_filter_rows_string_column():
    """Test filtering on string columns with equality and lexicographic comparison."""
    pdf = pd.DataFrame({"b": ["x", "y", "z"]})
    
    result_eq = ar.filter_rows(pdf, "b", "==", "y")
    assert list(result_eq["b"]) == ["y"]
    
    result_lex = ar.filter_rows(pdf, "b", ">=", "y")
    assert list(result_lex["b"]) == ["y", "z"]


def test_filter_rows_no_matches():
    """Test that no matching rows returns empty frame with same columns."""
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = ar.filter_rows(pdf, "a", ">", 100)
    
    assert len(result) == 0
    assert list(result.columns) == ["a", "b"]


def test_filter_rows_nan_handling():
    """Test NaN handling in comparisons."""
    pdf = pd.DataFrame({"f": [1.0, float("nan"), 3.0]})
    
    # NaN with != keeps the NaN row (NaN != 1.0 is True)
    result_ne = ar.filter_rows(pdf, "f", "!=", 1.0)
    assert len(result_ne) == 2
    assert list(result_ne.index) == [1, 2]
    
    # NaN with > drops the NaN row (comparison is fillna'd to False)
    result_gt = ar.filter_rows(pdf, "f", ">", 0)
    assert len(result_gt) == 2
    assert list(result_gt.index) == [0, 2]


def test_filter_rows_immutability():
    """Test that input frames are not modified."""
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    original_copy = pdf.copy()
    
    ar.filter_rows(pdf, "a", ">", 1)
    
    assert pdf.equals(original_copy)


def test_filter_rows_invalid_frame_type():
    """Test that non-frame input raises TypeError."""
    with pytest.raises(TypeError, match="must be an ArFrame or a pandas DataFrame"):
        ar.filter_rows({"a": [1]}, "a", ">", 1)


def test_filter_rows_invalid_column():
    """Test that invalid column raises TypeError."""
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    
    # Non-string column
    with pytest.raises(TypeError, match=re.escape("filter_rows: column must be a non-empty string, got 'int'")):
        ar.filter_rows(pdf, 5, ">", 1)
    
    # Empty string
    with pytest.raises(TypeError, match=re.escape("filter_rows: column must be a non-empty string, got 'str'")):
        ar.filter_rows(pdf, "", ">", 1)


def test_filter_rows_invalid_op():
    """Test that invalid operator raises appropriate errors."""
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    
    # Non-string op
    with pytest.raises(TypeError, match=re.escape("filter_rows: op must be a string, got 'NoneType'")):
        ar.filter_rows(pdf, "a", None, 1)
    
    # Unsupported operator
    with pytest.raises(ValueError, match=re.escape("Unsupported operator: =")):
        ar.filter_rows(pdf, "a", "=", 1)


def test_filter_rows_column_not_found():
    """Test that non-existent column raises ValueError."""
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match=re.escape("Unknown column: zzz")):
        ar.filter_rows(pdf, "zzz", ">", 1)


def test_filter_rows_non_scalar_value():
    """Test that non-scalar value raises TypeError."""
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(TypeError, match="filter_rows value must be a scalar"):
        ar.filter_rows(pdf, "a", ">", [1, 2])


def test_filter_rows_incompatible_comparison():
    """Test that incompatible type comparison raises TypeError."""
    pdf = pd.DataFrame({"s": ["a", "b"]})
    with pytest.raises(TypeError, match=re.escape("filter_rows: cannot compare column 's' with value 1 using operator '>'")):
        ar.filter_rows(pdf, "s", ">", 1)


def test_filter_rows_validation_order_op_before_column():
    """Test that op is validated before column existence."""
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match=re.escape("Unsupported operator: =")):
        ar.filter_rows(pdf, "missing", "=", 1)


def test_filter_rows_validation_order_column_before_value():
    """Test that column is validated before value type."""
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match=re.escape("Unknown column: missing")):
        ar.filter_rows(pdf, "missing", ">", [1])
