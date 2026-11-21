import pytest
import pandas as pd
import numpy as np
import arnio as ar


@pytest.fixture
def sample_arframe():
    """Create a sample ArFrame for testing."""
    data = {
        "age": [18, 25, 30, 35, 40],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "salary": [50000.0, 60000.0, 75000.0, 80000.0, 95000.0],
    }
    return ar.from_dict(data)


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame({
        "age": [18, 25, 30, 35, 40],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "salary": [50000.0, 60000.0, 75000.0, 80000.0, 95000.0],
    })


def test_filter_rows_greater_than(sample_arframe):
    """Test filtering with > operator."""
    result = ar.filter_rows(sample_arframe, column="age", op=">", value=25)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 3
    assert all(result_df["age"] > 25)


def test_filter_rows_less_than(sample_arframe):
    """Test filtering with < operator."""
    result = ar.filter_rows(sample_arframe, column="age", op="<", value=30)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 2
    assert all(result_df["age"] < 30)


def test_filter_rows_greater_equal(sample_arframe):
    """Test filtering with >= operator."""
    result = ar.filter_rows(sample_arframe, column="age", op=">=", value=30)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 3
    assert all(result_df["age"] >= 30)


def test_filter_rows_less_equal(sample_arframe):
    """Test filtering with <= operator."""
    result = ar.filter_rows(sample_arframe, column="age", op="<=", value=25)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 2
    assert all(result_df["age"] <= 25)


def test_filter_rows_equal(sample_arframe):
    """Test filtering with == operator."""
    result = ar.filter_rows(sample_arframe, column="age", op="==", value=30)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 1
    assert result_df["age"].iloc[0] == 30


def test_filter_rows_not_equal(sample_arframe):
    """Test filtering with != operator."""
    result = ar.filter_rows(sample_arframe, column="age", op="!=", value=30)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 4
    assert all(result_df["age"] != 30)


def test_filter_rows_returns_arframe(sample_arframe):
    """Test that ArFrame input returns ArFrame output."""
    result = ar.filter_rows(sample_arframe, column="age", op=">", value=25)
    assert isinstance(result, ar.ArFrame)


def test_filter_rows_returns_dataframe(sample_dataframe):
    """Test that DataFrame input returns DataFrame output."""
    result = ar.filter_rows(sample_dataframe, column="age", op=">", value=25)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_filter_rows_invalid_operator(sample_arframe):
    """Test that invalid operator raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported operator"):
        ar.filter_rows(sample_arframe, column="age", op=">>", value=25)


def test_filter_rows_column_not_found(sample_arframe):
    """Test that non-existent column raises ValueError."""
    with pytest.raises(ValueError, match="Unknown column"):
        ar.filter_rows(sample_arframe, column="nonexistent", op=">", value=25)


def test_filter_rows_non_scalar_value(sample_arframe):
    """Test that non-scalar value raises TypeError."""
    with pytest.raises(TypeError, match="must be a scalar"):
        ar.filter_rows(sample_arframe, column="age", op=">", value=[25, 30])


def test_filter_rows_empty_column_name(sample_arframe):
    """Test that empty column name raises TypeError."""
    with pytest.raises(TypeError, match="must be a non-empty string"):
        ar.filter_rows(sample_arframe, column="", op=">", value=25)


def test_filter_rows_type_mismatch(sample_arframe):
    """Test that comparing incompatible types raises TypeError."""
    with pytest.raises(TypeError):
        ar.filter_rows(sample_arframe, column="name", op=">", value=25)


def test_filter_rows_with_nulls():
    """Test filtering with NaN values - they should be treated as False."""
    data = {
        "value": [10.0, np.nan, 30.0, np.nan, 50.0]
    }
    frame = ar.from_dict(data)
    result = ar.filter_rows(frame, column="value", op=">", value=25)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 1
    assert result_df["value"].iloc[0] == 50.0


def test_filter_rows_no_rows_match(sample_arframe):
    """Test filtering where no rows match."""
    result = ar.filter_rows(sample_arframe, column="age", op=">", value=100)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 0
