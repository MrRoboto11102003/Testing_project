import pytest
import pandas as pd
import arnio as ar


def test_validate_valid_frame_passes():
    """Valid frame against matching schema passes validation."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    frame = ar.from_pandas(df)
    schema = ar.Schema({"a": ar.Int64(), "b": ar.String()})
    result = schema.validate(frame)
    assert result.passed
    assert result.issue_count == 0


def test_validate_non_arframe_raises_typeerror():
    """Passing DataFrame instead of ArFrame raises TypeError."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    schema = ar.Schema({"a": ar.Int64()})
    with pytest.raises(TypeError, match="expects a single ArFrame"):
        schema.validate(df)


def test_validate_max_errors_bool_raises_typeerror():
    """max_errors=True raises TypeError."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    schema = ar.Schema({"a": ar.Int64()})
    with pytest.raises(TypeError, match="max_errors must be an int or None"):
        schema.validate(frame, max_errors=True)


def test_validate_max_errors_zero_raises_valueerror():
    """max_errors=0 raises ValueError."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    schema = ar.Schema({"a": ar.Int64()})
    with pytest.raises(ValueError, match="max_errors must be >= 1"):
        schema.validate(frame, max_errors=0)


def test_validate_missing_required_column():
    """Missing required column generates required_column issue."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    schema = ar.Schema({"a": ar.Int64(), "b": ar.String()})
    result = schema.validate(frame)
    assert not result.passed
    assert any(i.rule == "required_column" for i in result.issues)


def test_validate_dtype_mismatch():
    """Column with wrong dtype generates dtype issue."""
    frame = ar.from_pandas(pd.DataFrame({"a": ["1", "2", "3"]}))
    schema = ar.Schema({"a": ar.Int64()})
    result = schema.validate(frame)
    assert not result.passed
    assert any(i.rule == "dtype" for i in result.issues)


def test_validate_nullable_false_rejects_null():
    """nullable=False rejects null/NA values in column."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, None, 3]}))
    schema = ar.Schema({"a": ar.Int64(nullable=False)})
    result = schema.validate(frame)
    assert not result.passed
    assert any(i.rule == "nullable" for i in result.issues)


def test_validate_unique_true_rejects_duplicates():
    """unique=True rejects duplicate values."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 1]}))
    schema = ar.Schema({"a": ar.Int64(unique=True)})
    result = schema.validate(frame)
    assert not result.passed
    assert any(i.rule == "unique" for i in result.issues)


def test_validate_allowed_values():
    """Values outside allowed set fail validation."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    schema = ar.Schema({"a": ar.Int64(allowed=[1, 2])})
    result = schema.validate(frame)
    assert not result.passed
    assert any(i.rule == "allowed" for i in result.issues)


def test_validate_min_max_bounds():
    """Values outside min/max bounds fail validation."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 5, 10]}))
    schema = ar.Schema({"a": ar.Int64(min=2, max=9)})
    result = schema.validate(frame)
    assert not result.passed
    assert len([i for i in result.issues if i.rule in ("min", "max")]) > 0


def test_validate_max_errors_limit_stops_validation():
    """max_errors parameter limits number of issues returned."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, None, None, None]}))
    schema = ar.Schema({"a": ar.Int64(nullable=False)})
    result = schema.validate(frame, max_errors=1)
    assert result.issue_count == 1


def test_validate_row_index_one_based():
    """row_index in issues is 1-based, not 0-based."""
    frame = ar.from_pandas(pd.DataFrame({"a": [None, 2, 3]}))
    schema = ar.Schema({"a": ar.Int64(nullable=False)})
    result = schema.validate(frame)
    assert result.issues[0].row_index == 1


def test_validate_bad_rows_sorted_and_deduplicated():
    """bad_rows is sorted and deduplicated."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, None], "b": [None, 2]}))
    schema = ar.Schema({
        "a": ar.Int64(nullable=False),
        "b": ar.Int64(nullable=False),
    })
    result = schema.validate(frame)
    assert result.bad_rows == sorted(set(result.bad_rows))
    assert len(result.bad_rows) == len(set(result.bad_rows))


def test_validate_result_row_count_and_issue_count():
    """ValidationResult row_count and issue_count are correct."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3, 4, 5]}))
    schema = ar.Schema({"a": ar.Int64()})
    result = schema.validate(frame)
    assert result.row_count == 5
    assert result.issue_count == len(result.issues)
    assert result.passed


def test_validate_zero_row_frame():
    """Zero-row frame with matching columns passes validation."""
    df = pd.DataFrame({"a": pd.Series([], dtype='int64')})
    frame = ar.from_pandas(df)
    schema = ar.Schema({"a": ar.Int64()})
    result = schema.validate(frame)
    assert result.row_count == 0
    assert result.passed
