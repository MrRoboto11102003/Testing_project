import pytest
import pandas as pd
import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a sample ArFrame with various data types."""
    data = {
        'int_col': pd.array([1, 2, 3], dtype='int64'),
        'float_col': pd.array([1.1, 2.2, 3.3], dtype='float64'),
        'str_col': pd.array(['a', 'b', 'c'], dtype='string'),
        'bool_col': pd.array([True, False, True], dtype='bool'),
        'int_col2': pd.array([4, 5, 6], dtype='int64'),
        'float_col2': pd.array([4.4, 5.5, 6.6], dtype='float64'),
    }
    return ar.ArFrame(pd.DataFrame(data))


def test_include_single_string(sample_frame):
    """Test include with single dtype string."""
    result = sample_frame.select_dtypes(include="int64")
    assert list(result.columns) == ['int_col', 'int_col2']


def test_include_list(sample_frame):
    """Test include with list of dtype strings."""
    result = sample_frame.select_dtypes(include=["int64", "float64"])
    assert list(result.columns) == ['int_col', 'float_col', 'int_col2', 'float_col2']


def test_include_tuple(sample_frame):
    """Test include with tuple of dtype strings."""
    result = sample_frame.select_dtypes(include=("string", "bool"))
    assert set(result.columns) == {'str_col', 'bool_col'}


def test_exclude_single_string(sample_frame):
    """Test exclude with single dtype string."""
    result = sample_frame.select_dtypes(exclude="int64")
    assert 'int_col' not in result.columns
    assert 'int_col2' not in result.columns


def test_exclude_list(sample_frame):
    """Test exclude with list of dtype strings."""
    result = sample_frame.select_dtypes(exclude=["int64", "float64"])
    assert set(result.columns) == {'str_col', 'bool_col'}


def test_include_and_exclude(sample_frame):
    """Test include and exclude together without overlap."""
    result = sample_frame.select_dtypes(
        include=["int64", "float64", "string"],
        exclude="string"
    )
    assert set(result.columns) == {'int_col', 'float_col', 'int_col2', 'float_col2'}


def test_column_order_preserved(sample_frame):
    """Test that column order is preserved."""
    result = sample_frame.select_dtypes(include=["float64", "int64"])
    assert list(result.columns) == ['int_col', 'float_col', 'int_col2', 'float_col2']


def test_no_matches_returns_empty_frame(sample_frame):
    """Test when no columns match."""
    result = sample_frame.select_dtypes(include="null")
    assert len(result.columns) == 0
    assert len(result) == len(sample_frame)


def test_missing_include_and_exclude_raises_valueerror(sample_frame):
    """ValueError when neither include nor exclude is provided."""
    with pytest.raises(ValueError, match="requires at least one"):
        sample_frame.select_dtypes()


def test_overlap_include_exclude_raises_valueerror(sample_frame):
    """ValueError when include and exclude overlap."""
    with pytest.raises(ValueError, match="overlap"):
        sample_frame.select_dtypes(include="int64", exclude="int64")


def test_unrecognized_dtype_raises_valueerror(sample_frame):
    """ValueError for unrecognized dtype."""
    with pytest.raises(ValueError, match="Unrecognised dtype"):
        sample_frame.select_dtypes(include="invalid_type")


def test_invalid_type_for_include_raises_typeerror(sample_frame):
    """TypeError when include is not string/list/tuple."""
    with pytest.raises(TypeError, match="'include' must be a string"):
        sample_frame.select_dtypes(include=123)


def test_invalid_type_for_exclude_raises_typeerror(sample_frame):
    """TypeError when exclude is not string/list/tuple."""
    with pytest.raises(TypeError, match="'exclude' must be a string"):
        sample_frame.select_dtypes(exclude={"string": True})


def test_non_string_in_list_raises_typeerror(sample_frame):
    """TypeError when list contains non-strings."""
    with pytest.raises(TypeError, match="must contain only strings"):
        sample_frame.select_dtypes(include=["int64", 123])


def test_returns_new_arframe_instance(sample_frame):
    """Test that a new ArFrame instance is returned."""
    result = sample_frame.select_dtypes(include="int64")
    assert isinstance(result, ar.ArFrame)
    assert result is not sample_frame
