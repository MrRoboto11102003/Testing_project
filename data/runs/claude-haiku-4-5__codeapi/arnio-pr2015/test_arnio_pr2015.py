import pytest
import pandas as pd
import numpy as np
import arnio as ar


def test_default_true_values():
    """Test conversion with default true_values {'true', 'yes', 'y', '1'}."""
    frame = ar.from_dict({'col': ['true', 'yes', 'y', '1']})
    result = ar.parse_bool_strings(frame)
    df = ar.to_pandas(result)
    assert all(df['col'] == True)


def test_default_false_values():
    """Test conversion with default false_values {'false', 'no', 'n', '0'}."""
    frame = ar.from_dict({'col': ['false', 'no', 'n', '0']})
    result = ar.parse_bool_strings(frame)
    df = ar.to_pandas(result)
    assert all(df['col'] == False)


def test_custom_true_values():
    """Test conversion with custom true_values."""
    frame = ar.from_dict({'col': ['on', 'enabled', 'other']})
    result = ar.parse_bool_strings(frame, true_values={'on', 'enabled'})
    df = ar.to_pandas(result)
    assert df['col'][0] is True
    assert df['col'][1] is True
    assert df['col'][2] == 'other'


def test_custom_false_values():
    """Test conversion with custom false_values."""
    frame = ar.from_dict({'col': ['off', 'disabled', 'other']})
    result = ar.parse_bool_strings(frame, false_values={'off', 'disabled'})
    df = ar.to_pandas(result)
    assert df['col'][0] is False
    assert df['col'][1] is False
    assert df['col'][2] == 'other'


def test_subset_parameter():
    """Test conversion on subset of columns."""
    frame = ar.from_dict({
        'col1': ['true', 'false'],
        'col2': ['yes', 'no']
    })
    result = ar.parse_bool_strings(frame, subset=['col1'])
    df = ar.to_pandas(result)
    assert df['col1'][0] is True
    assert df['col1'][1] is False
    assert df['col2'][0] == 'yes'
    assert df['col2'][1] == 'no'


def test_case_insensitivity():
    """Test that parsing is case-insensitive."""
    frame = ar.from_dict({'col': ['TRUE', 'False', 'YES', 'no']})
    result = ar.parse_bool_strings(frame)
    df = ar.to_pandas(result)
    assert df['col'][0] is True
    assert df['col'][1] is False
    assert df['col'][2] is True
    assert df['col'][3] is False


def test_whitespace_normalization():
    """Test that leading/trailing whitespace is stripped before conversion."""
    frame = ar.from_dict({'col': [' true ', ' false ', '  yes  ', '  no  ']})
    result = ar.parse_bool_strings(frame)
    df = ar.to_pandas(result)
    assert df['col'][0] is True
    assert df['col'][1] is False
    assert df['col'][2] is True
    assert df['col'][3] is False


def test_unrecognized_values_preserved():
    """Test that unrecognized values are preserved as-is."""
    frame = ar.from_dict({'col': ['true', 'maybe', 'false', 'unknown']})
    result = ar.parse_bool_strings(frame)
    df = ar.to_pandas(result)
    assert df['col'][0] is True
    assert df['col'][1] == 'maybe'
    assert df['col'][2] is False
    assert df['col'][3] == 'unknown'


def test_null_values_preserved():
    """Test that null and NaN values are preserved unchanged."""
    frame = ar.from_dict({'col': ['true', None, 'false', np.nan]})
    result = ar.parse_bool_strings(frame)
    df = ar.to_pandas(result)
    assert df['col'][0] is True
    assert pd.isna(df['col'][1])
    assert df['col'][2] is False
    assert pd.isna(df['col'][3])


def test_list_and_tuple_sequences():
    """Test that list and tuple can be used instead of set."""
    frame = ar.from_dict({'col': ['yes', 'no']})
    result = ar.parse_bool_strings(frame, true_values=['yes'], false_values=('no',))
    df = ar.to_pandas(result)
    assert df['col'][0] is True
    assert df['col'][1] is False


def test_overlapping_values_raises_error():
    """Test that overlapping true_values and false_values raises ValueError."""
    frame = ar.from_dict({'col': ['true', 'false']})
    with pytest.raises(ValueError, match="overlap"):
        ar.parse_bool_strings(
            frame,
            true_values={'yes', 'true'},
            false_values={'no', 'true'}
        )


def test_bare_string_raises_typeerror():
    """Test that bare string for true_values or false_values raises TypeError."""
    frame = ar.from_dict({'col': ['true', 'false']})
    with pytest.raises(TypeError, match="true_values must be a set"):
        ar.parse_bool_strings(frame, true_values='true')
    with pytest.raises(TypeError, match="false_values must be a set"):
        ar.parse_bool_strings(frame, false_values='false')


def test_non_string_in_values_raises_typeerror():
    """Test that non-string items in true/false values raise TypeError."""
    frame = ar.from_dict({'col': ['true', 'false']})
    with pytest.raises(TypeError, match="true_values must contain only strings"):
        ar.parse_bool_strings(frame, true_values={'true', 1})
    with pytest.raises(TypeError, match="false_values must contain only strings"):
        ar.parse_bool_strings(frame, false_values={'false', 2})


def test_boolean_columns_skipped():
    """Test that existing boolean columns are skipped when subset is None."""
    df = pd.DataFrame({
        'bool_col': [True, False, True],
        'str_col': ['true', 'false', 'true']
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    assert result_df['bool_col'].dtype == bool
    assert result_df['str_col'][0] is True
    assert result_df['str_col'][1] is False


def test_all_string_columns_converted_by_default():
    """Test that all object/string columns are converted when subset is None."""
    frame = ar.from_dict({
        'col1': ['true', 'false'],
        'col2': ['yes', 'no'],
        'col3': ['maybe', 'unknown']
    })
    result = ar.parse_bool_strings(frame)
    df = ar.to_pandas(result)
    assert df['col1'][0] is True
    assert df['col2'][0] is True
    assert df['col3'][0] == 'maybe'


def test_empty_dataframe():
    """Test with an empty dataframe."""
    frame = ar.from_dict({'col': []})
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 0
