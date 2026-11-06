import pytest
import arnio as ar


@pytest.fixture
def simple_frame():
    """Create a simple 3-column ArFrame."""
    data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.1, 2.2, 3.3]}
    return ar.from_dict(data)


@pytest.fixture
def multi_column_frame():
    """Create a 5-column ArFrame."""
    data = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9],
        "d": [10, 11, 12],
        "e": [13, 14, 15],
    }
    return ar.from_dict(data)


def test_drop_single_column_list(simple_frame):
    """Test dropping a single column with list input."""
    result = simple_frame.drop_columns(["col1"])
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["col2", "col3"]
    assert len(result_df) == 3


def test_drop_multiple_columns_list(simple_frame):
    """Test dropping multiple columns with list."""
    result = simple_frame.drop_columns(["col1", "col3"])
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["col2"]


def test_drop_columns_with_tuple(simple_frame):
    """Test drop_columns accepts tuple input."""
    result = simple_frame.drop_columns(("col1", "col2"))
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["col3"]


def test_drop_columns_preserves_column_order(multi_column_frame):
    """Test that original column order is preserved."""
    result = multi_column_frame.drop_columns(["b", "d"])
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["a", "c", "e"]


def test_drop_columns_empty_list(simple_frame):
    """Test that empty list returns unchanged copy."""
    result = simple_frame.drop_columns([])
    result_df = ar.to_pandas(result)
    original_df = ar.to_pandas(simple_frame)
    assert list(result_df.columns) == list(original_df.columns)
    assert result is not simple_frame


def test_drop_columns_duplicate_names_silently_ignored(simple_frame):
    """Test that duplicate names in cols are silently ignored."""
    result = simple_frame.drop_columns(["col1", "col1", "col1"])
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["col2", "col3"]


def test_drop_columns_data_preserved(simple_frame):
    """Test that data in remaining columns is preserved."""
    result = simple_frame.drop_columns(["col1"])
    result_df = ar.to_pandas(result)
    original_df = ar.to_pandas(simple_frame)
    
    assert (result_df["col2"] == original_df["col2"]).all()
    assert (result_df["col3"] == original_df["col3"]).all()


def test_drop_columns_original_unchanged(simple_frame):
    """Test that original frame is not modified."""
    original_cols = set(ar.to_pandas(simple_frame).columns)
    simple_frame.drop_columns(["col1"])
    current_cols = set(ar.to_pandas(simple_frame).columns)
    assert original_cols == current_cols


def test_drop_columns_nonexistent_raises_value_error(simple_frame):
    """Test ValueError when column doesn't exist."""
    with pytest.raises(ValueError):
        simple_frame.drop_columns(["nonexistent"])


def test_drop_columns_mixed_existent_nonexistent_raises(simple_frame):
    """Test ValueError when any column in list doesn't exist."""
    with pytest.raises(ValueError):
        simple_frame.drop_columns(["col1", "nonexistent"])


def test_drop_columns_non_string_int_raises_type_error(simple_frame):
    """Test TypeError when cols contains non-string elements (int)."""
    with pytest.raises(TypeError):
        simple_frame.drop_columns(["col1", 123])


def test_drop_columns_non_string_float_raises_type_error(simple_frame):
    """Test TypeError when cols contains float."""
    with pytest.raises(TypeError):
        simple_frame.drop_columns([1.5])


def test_drop_columns_not_list_or_tuple_raises_type_error(simple_frame):
    """Test TypeError when cols is not list or tuple."""
    with pytest.raises(TypeError):
        simple_frame.drop_columns("col1")


def test_drop_columns_dict_raises_type_error(simple_frame):
    """Test TypeError when cols is dict instead of list/tuple."""
    with pytest.raises(TypeError):
        simple_frame.drop_columns({"col1": True})


def test_drop_columns_all_but_one(multi_column_frame):
    """Test dropping all but one column preserves data."""
    result = multi_column_frame.drop_columns(["a", "b", "c", "d"])
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["e"]
    assert len(result_df) == 3
