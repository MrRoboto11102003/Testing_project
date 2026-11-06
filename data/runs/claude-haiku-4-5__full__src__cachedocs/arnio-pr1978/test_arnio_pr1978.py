import pytest
import pandas as pd
import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a sample ArFrame with various dtypes for testing."""
    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.5, 2.5, 3.5],
        'string_col': ['alice', 'bob', 'charlie'],
        'bool_col': [True, False, True],
    })
    return ar.from_pandas(df)


def test_include_single_dtype_string(sample_frame):
    """Include a single dtype as string."""
    result = sample_frame.select_dtypes(include='int64')
    assert result.columns == ['int_col']


def test_include_multiple_dtypes_list(sample_frame):
    """Include multiple dtypes as list."""
    result = sample_frame.select_dtypes(include=['int64', 'float64'])
    assert set(result.columns) == {'int_col', 'float_col'}


def test_include_multiple_dtypes_tuple(sample_frame):
    """Include multiple dtypes as tuple."""
    result = sample_frame.select_dtypes(include=('string', 'bool'))
    assert set(result.columns) == {'string_col', 'bool_col'}


def test_include_preserves_column_order(sample_frame):
    """Included columns preserve original order."""
    result = sample_frame.select_dtypes(include=['float64', 'int64'])
    assert result.columns == ['int_col', 'float_col']


def test_include_no_matches_returns_empty_frame(sample_frame):
    """No columns match include filter returns empty frame."""
    result = sample_frame.select_dtypes(include='null')
    assert result.shape[0] == sample_frame.shape[0]
    assert result.shape[1] == 0


def test_exclude_single_dtype_string(sample_frame):
    """Exclude a single dtype as string."""
    result = sample_frame.select_dtypes(exclude='string')
    assert set(result.columns) == {'int_col', 'float_col', 'bool_col'}


def test_exclude_multiple_dtypes(sample_frame):
    """Exclude multiple dtypes as list."""
    result = sample_frame.select_dtypes(exclude=['int64', 'float64'])
    assert set(result.columns) == {'string_col', 'bool_col'}


def test_exclude_preserves_column_order(sample_frame):
    """Excluded columns preserve original order."""
    result = sample_frame.select_dtypes(exclude=['string', 'bool'])
    assert result.columns == ['int_col', 'float_col']


def test_include_and_exclude_together(sample_frame):
    """Include and exclude can be used together."""
    result = sample_frame.select_dtypes(
        include=['int64', 'float64', 'string'],
        exclude=['string']
    )
    assert set(result.columns) == {'int_col', 'float_col'}


def test_neither_include_nor_exclude_raises_value_error(sample_frame):
    """Neither include nor exclude raises ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        sample_frame.select_dtypes()


def test_include_exclude_overlap_raises_value_error(sample_frame):
    """Overlapping include and exclude raises ValueError."""
    with pytest.raises(ValueError, match="overlap"):
        sample_frame.select_dtypes(include='int64', exclude='int64')


def test_unknown_dtype_raises_value_error(sample_frame):
    """Unknown dtype raises ValueError."""
    with pytest.raises(ValueError, match="Unrecognised dtype"):
        sample_frame.select_dtypes(include='unknown_dtype')


def test_include_wrong_type_raises_type_error(sample_frame):
    """Include with wrong type raises TypeError."""
    with pytest.raises(TypeError, match="must be a string, list, or tuple"):
        sample_frame.select_dtypes(include=123)


def test_include_list_with_non_string_raises_type_error(sample_frame):
    """Include list with non-string elements raises TypeError."""
    with pytest.raises(TypeError, match="must contain only strings"):
        sample_frame.select_dtypes(include=['int64', 123])


def test_all_columns_excluded(sample_frame):
    """Excluding all columns returns empty frame with same row count."""
    result = sample_frame.select_dtypes(
        exclude=['int64', 'float64', 'string', 'bool']
    )
    assert result.shape[0] == sample_frame.shape[0]
    assert result.shape[1] == 0
