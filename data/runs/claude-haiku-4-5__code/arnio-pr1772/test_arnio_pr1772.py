import pytest
import arnio as ar


def test_from_records_dict_without_columns():
    """Test creating ArFrame from list of dicts without specifying columns."""
    records = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]
    result = ar.frame.ArFrame.from_records(records)
    assert isinstance(result, ar.frame.ArFrame)
    assert set(result.columns) == {"a", "b"}
    assert len(result) == 2


def test_from_records_dict_with_columns():
    """Test creating ArFrame from list of dicts with columns specified."""
    records = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]
    result = ar.frame.ArFrame.from_records(records, columns=["a", "b"])
    assert set(result.columns) == {"a", "b"}
    assert len(result) == 2


def test_from_records_lists_with_columns():
    """Test creating ArFrame from list of lists with columns specified."""
    records = [
        [1, 2],
        [3, 4],
    ]
    result = ar.frame.ArFrame.from_records(records, columns=["a", "b"])
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 2


def test_from_records_tuples_with_columns():
    """Test creating ArFrame from list of tuples with columns specified."""
    records = [
        (1, 2),
        (3, 4),
    ]
    result = ar.frame.ArFrame.from_records(records, columns=["a", "b"])
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 2


def test_from_records_records_not_list():
    """Test that TypeError is raised when records is not a list."""
    with pytest.raises(TypeError, match="records must be a list"):
        ar.frame.ArFrame.from_records(({"a": 1},))


def test_from_records_empty_records():
    """Test that ValueError is raised for empty records list."""
    with pytest.raises(ValueError, match="records must be non-empty"):
        ar.frame.ArFrame.from_records([])


def test_from_records_mixed_types():
    """Test that TypeError is raised for mixed record types."""
    records = [
        {"a": 1},
        [2],
    ]
    with pytest.raises(TypeError, match="all records must be"):
        ar.frame.ArFrame.from_records(records, columns=["a"])


def test_from_records_nested_list():
    """Test that TypeError is raised for nested list in cell."""
    records = [
        {"a": [1, 2]},
    ]
    with pytest.raises(TypeError, match="nested values are not supported"):
        ar.frame.ArFrame.from_records(records)


def test_from_records_nested_dict():
    """Test that TypeError is raised for nested dict in cell."""
    records = [
        {"a": {"nested": 1}},
    ]
    with pytest.raises(TypeError, match="nested values are not supported"):
        ar.frame.ArFrame.from_records(records)


def test_from_records_columns_string():
    """Test that TypeError is raised when columns is a string."""
    records = [{"a": 1}]
    with pytest.raises(TypeError, match="columns must be a list or tuple"):
        ar.frame.ArFrame.from_records(records, columns="a")


def test_from_records_columns_bytes():
    """Test that TypeError is raised when columns is bytes."""
    records = [{"a": 1}]
    with pytest.raises(TypeError, match="columns must be a list or tuple"):
        ar.frame.ArFrame.from_records(records, columns=b"a")


def test_from_records_columns_non_strings():
    """Test that TypeError is raised for non-string column names."""
    records = [{"a": 1}]
    with pytest.raises(TypeError, match="columns must contain only strings"):
        ar.frame.ArFrame.from_records(records, columns=[1, "b"])


def test_from_records_lists_without_columns():
    """Test that ValueError is raised when columns missing for list records."""
    records = [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="columns must be provided"):
        ar.frame.ArFrame.from_records(records)


def test_from_records_row_length_mismatch():
    """Test that ValueError is raised for row length mismatch."""
    records = [
        [1, 2],
        [3],
    ]
    with pytest.raises(ValueError, match="has .* value.*but .* column"):
        ar.frame.ArFrame.from_records(records, columns=["a", "b"])


def test_from_records_dict_empty_columns():
    """Test that ValueError is raised for empty columns list with dict records."""
    records = [{"a": 1}]
    with pytest.raises(ValueError, match="columns must not be empty"):
        ar.frame.ArFrame.from_records(records, columns=[])
