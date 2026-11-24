import pytest
import re
import arnio as ar


def test_dict_records_no_columns():
    """Happy path: infer columns from dict keys."""
    records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    frame = ar.ArFrame.from_records(records)
    assert frame.shape == (2, 2)
    assert list(frame["a"]) == [1, 2]


def test_dict_records_with_columns():
    """Dict records with explicit columns."""
    records = [{"a": 1}, {"a": 2}]
    frame = ar.ArFrame.from_records(records, columns=["a"])
    assert frame.shape == (2, 1)


def test_list_records():
    """Happy path: list of lists with columns."""
    records = [[1, "x"], [2, "y"]]
    frame = ar.ArFrame.from_records(records, columns=["id", "name"])
    assert frame.shape == (2, 2)


def test_tuple_records():
    """Tuple records with tuple columns."""
    records = [(1, "x"), (2, "y")]
    frame = ar.ArFrame.from_records(records, columns=("id", "name"))
    assert frame.shape == (2, 2)


def test_mixed_list_tuple_rows():
    """List and tuple rows can be mixed."""
    records = [(1, "x"), [2, "y"]]
    frame = ar.ArFrame.from_records(records, columns=["a", "b"])
    assert frame.shape == (2, 2)


def test_records_not_list():
    """TypeError if records is not a list."""
    with pytest.raises(TypeError, match="records must be a list, got 'tuple'"):
        ar.ArFrame.from_records(({"a": 1},))


def test_records_empty():
    """ValueError if records is empty."""
    with pytest.raises(ValueError, match="records must be non-empty"):
        ar.ArFrame.from_records([])


def test_columns_is_string():
    """TypeError if columns is a string."""
    with pytest.raises(TypeError, match="not a string or bytes"):
        ar.ArFrame.from_records([{"a": 1}], columns="a")


def test_columns_not_list_tuple():
    """TypeError if columns is not list or tuple."""
    with pytest.raises(TypeError, match="columns must be a list or tuple"):
        ar.ArFrame.from_records([{"a": 1}], columns={"a"})


def test_columns_non_string_element():
    """TypeError if columns contains non-string."""
    with pytest.raises(TypeError, match="columns must contain only strings"):
        ar.ArFrame.from_records([{"a": 1}], columns=["a", 1])


def test_dict_records_mixed_types():
    """TypeError if dict records mixed with other types."""
    with pytest.raises(TypeError, match="all records must be dicts, but row 1 is 'list'"):
        ar.ArFrame.from_records([{"a": 1}, [1]])


def test_dict_records_nested_values():
    """TypeError for nested list in dict records."""
    with pytest.raises(TypeError, match="nested values are not supported.*'list'"):
        ar.ArFrame.from_records([{"a": [1]}])


def test_dict_records_empty_columns():
    """ValueError if columns is empty list for dicts."""
    with pytest.raises(ValueError, match="columns must not be empty"):
        ar.ArFrame.from_records([{"a": 1}], columns=[])


def test_sequence_records_no_columns():
    """ValueError if columns not provided for sequences."""
    with pytest.raises(ValueError, match="columns must be provided"):
        ar.ArFrame.from_records([[1, 2]])


def test_sequence_records_length_mismatch():
    """ValueError if row length doesn't match columns."""
    with pytest.raises(ValueError, match="row 1 has 1 value"):
        ar.ArFrame.from_records([[1, 2], [3]], columns=["a", "b"])
