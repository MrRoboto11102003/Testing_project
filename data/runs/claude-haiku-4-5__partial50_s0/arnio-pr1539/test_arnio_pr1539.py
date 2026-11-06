import pytest
import arnio as ar


def test_drop_single_column():
    """Test dropping a single column"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"], "age": [25, 30, 35]})
    result = frame.drop_columns(["name"])
    assert result.columns == ["id", "age"]
    assert result.shape == (3, 2)


def test_drop_multiple_columns():
    """Test dropping multiple columns at once"""
    frame = ar.from_dict({
        "id": [1, 2, 3],
        "name": ["A", "B", "C"],
        "age": [25, 30, 35],
        "city": ["NYC", "LA", "SF"],
    })
    result = frame.drop_columns(["name", "city"])
    assert result.columns == ["id", "age"]


def test_drop_columns_tuple_input():
    """Test that tuple input is accepted"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"], "age": [25, 30, 35]})
    result = frame.drop_columns(("name", "age"))
    assert result.columns == ["id"]


def test_drop_columns_empty_list():
    """Test empty list returns unchanged copy"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    result = frame.drop_columns([])
    assert result.columns == frame.columns
    assert result.shape == frame.shape


def test_drop_columns_preserves_order():
    """Test that column order is preserved for remaining columns"""
    frame = ar.from_dict({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]})
    result = frame.drop_columns(["b", "d"])
    assert result.columns == ["a", "c"]


def test_drop_columns_silently_ignores_duplicates():
    """Test duplicate column names in input are silently ignored"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"], "age": [25, 30, 35]})
    result = frame.drop_columns(["name", "name", "age", "age"])
    assert result.columns == ["id"]


def test_drop_columns_nonexistent_raises_valueerror():
    """Test ValueError is raised for non-existent column"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    with pytest.raises(ValueError):
        frame.drop_columns(["nonexistent"])


def test_drop_columns_partial_invalid_raises_valueerror():
    """Test ValueError when one of multiple columns doesn't exist"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    with pytest.raises(ValueError):
        frame.drop_columns(["name", "nonexistent"])


def test_drop_columns_string_input_raises_typeerror():
    """Test TypeError for string input instead of list/tuple"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    with pytest.raises(TypeError):
        frame.drop_columns("name")


def test_drop_columns_dict_input_raises_typeerror():
    """Test TypeError for dict input"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    with pytest.raises(TypeError):
        frame.drop_columns({"name": None})


def test_drop_columns_nonstring_element_raises_typeerror():
    """Test TypeError for non-string elements in list"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    with pytest.raises(TypeError):
        frame.drop_columns(["name", 123])


def test_drop_columns_returns_arframe():
    """Test return type is ArFrame"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    result = frame.drop_columns(["name"])
    assert isinstance(result, ar.ArFrame)


def test_drop_columns_does_not_modify_original():
    """Test original frame is not modified (immutability)"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"], "age": [25, 30, 35]})
    original_cols = frame.columns
    result = frame.drop_columns(["name"])
    assert frame.columns == original_cols
    assert "name" in frame.columns


def test_drop_columns_all_columns():
    """Test dropping all columns"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    result = frame.drop_columns(["id", "name"])
    assert result.columns == []
    assert result.shape[0] == 3


def test_drop_columns_data_preservation():
    """Test data integrity in remaining columns after drop"""
    frame = ar.from_dict({"id": [1, 2, 3], "name": ["A", "B", "C"], "age": [25, 30, 35]})
    result = frame.drop_columns(["name"])
    df = ar.to_pandas(result)
    assert list(df["id"]) == [1, 2, 3]
    assert list(df["age"]) == [25, 30, 35]
