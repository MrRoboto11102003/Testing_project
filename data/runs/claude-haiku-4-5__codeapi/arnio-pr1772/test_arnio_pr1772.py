import pytest
import pandas as pd
import arnio as ar


class TestArFrameFromRecords:
    """Test ArFrame.from_records() classmethod."""

    def test_from_records_with_dicts_infers_columns(self):
        """Test creating ArFrame from dicts with automatic column inference."""
        records = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        frame = ar.ArFrame.from_records(records)
        assert frame.shape == (2, 2)
        assert set(frame.columns) == {"name", "age"}

    def test_from_records_with_dicts_explicit_columns(self):
        """Test creating ArFrame from dicts with explicit column subset."""
        records = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
        ]
        frame = ar.ArFrame.from_records(records, columns=["name", "age"])
        assert frame.shape == (2, 2)
        assert set(frame.columns) == {"name", "age"}

    def test_from_records_with_lists(self):
        """Test creating ArFrame from a list of lists."""
        records = [[1, "Alice"], [2, "Bob"]]
        columns = ["id", "name"]
        frame = ar.ArFrame.from_records(records, columns=columns)
        assert frame.shape == (2, 2)
        assert frame.columns == columns

    def test_from_records_with_tuples(self):
        """Test creating ArFrame from a list of tuples."""
        records = [(1, "Alice"), (2, "Bob")]
        columns = ["id", "name"]
        frame = ar.ArFrame.from_records(records, columns=columns)
        assert frame.shape == (2, 2)
        assert frame.columns == columns

    def test_from_records_error_not_a_list(self):
        """Test TypeError when records is not a list."""
        with pytest.raises(TypeError, match="records must be a list"):
            ar.ArFrame.from_records({"a": 1})

    def test_from_records_error_empty_list(self):
        """Test ValueError when records is empty."""
        with pytest.raises(ValueError, match="records must be non-empty"):
            ar.ArFrame.from_records([])

    def test_from_records_error_mixed_record_types(self):
        """Test TypeError when record types are mixed."""
        records = [{"x": 1}, [2, 3]]
        with pytest.raises(TypeError, match="all records must be"):
            ar.ArFrame.from_records(records)

    def test_from_records_error_nested_list_value(self):
        """Test TypeError when records contain nested lists."""
        records = [{"x": 1, "nested": [1, 2]}]
        with pytest.raises(TypeError, match="nested values are not supported"):
            ar.ArFrame.from_records(records)

    def test_from_records_error_nested_dict_value(self):
        """Test TypeError when records contain nested dicts."""
        records = [{"x": 1, "nested": {"a": 1}}]
        with pytest.raises(TypeError, match="nested values are not supported"):
            ar.ArFrame.from_records(records)

    def test_from_records_error_columns_missing_for_lists(self):
        """Test ValueError when columns not provided for list records."""
        records = [[1, 2], [3, 4]]
        with pytest.raises(ValueError, match="columns must be provided"):
            ar.ArFrame.from_records(records)

    def test_from_records_error_columns_is_string(self):
        """Test TypeError when columns is a string instead of list."""
        with pytest.raises(TypeError, match="columns must be a list or tuple"):
            ar.ArFrame.from_records([[1, 2]], columns="x,y")

    def test_from_records_error_columns_contains_non_string(self):
        """Test TypeError when columns contains non-string values."""
        with pytest.raises(TypeError, match="columns must contain only strings"):
            ar.ArFrame.from_records([{"a": 1}], columns=["a", 2])

    def test_from_records_error_row_length_mismatch(self):
        """Test ValueError when row length doesn't match columns."""
        records = [[1, 2, 3], [4, 5]]
        columns = ["a", "b", "c"]
        with pytest.raises(ValueError, match="row .* has"):
            ar.ArFrame.from_records(records, columns=columns)

    def test_from_records_error_empty_columns_list_with_dicts(self):
        """Test ValueError when columns list is empty for dict records."""
        with pytest.raises(ValueError, match="columns must not be empty"):
            ar.ArFrame.from_records([{"x": 1}], columns=[])

    def test_from_records_error_invalid_record_type(self):
        """Test TypeError when records contain invalid type (not dict/list/tuple)."""
        with pytest.raises(TypeError, match="records must contain dicts, lists, or tuples"):
            ar.ArFrame.from_records([42])

    def test_from_records_preserves_data(self):
        """Test that values are correctly preserved in the frame."""
        records = [
            {"id": 1, "name": "Alice", "score": 95.5},
            {"id": 2, "name": "Bob", "score": 87.3},
        ]
        frame = ar.ArFrame.from_records(records)
        df = ar.to_pandas(frame)
        assert df["id"].tolist() == [1, 2]
        assert df["name"].tolist() == ["Alice", "Bob"]
        assert df["score"].tolist() == [95.5, 87.3]
