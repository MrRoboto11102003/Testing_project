import pytest
import re
import arnio as ar


class TestArFrameFromRecords:
    """Test ArFrame.from_records() classmethod."""

    # ---- Valid basic cases ----

    def test_dict_records_basic(self):
        """Build ArFrame from simple dict records."""
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        frame = ar.from_records(records)
        assert frame.shape == (2, 2)
        assert frame.columns == ["a", "b"]

    def test_list_records_with_columns(self):
        """Build ArFrame from list records with columns parameter."""
        records = [[1, "x"], [2, "y"]]
        frame = ar.from_records(records, columns=["id", "name"])
        assert frame.shape == (2, 2)
        assert frame.columns == ["id", "name"]

    def test_mixed_list_tuple_rows_and_tuple_columns(self):
        """Accept mixed list/tuple rows and tuple columns parameter."""
        records = [[1, "x"], (2, "y")]
        frame = ar.from_records(records, columns=("id", "name"))
        assert frame.shape == (2, 2)
        assert frame.columns == ["id", "name"]

    # ---- Error: records parameter validation ----

    def test_records_not_list_raises_typeerror(self):
        """Reject records that is not a list."""
        with pytest.raises(TypeError, match=r"records must be a list, got 'tuple'"):
            ar.from_records(({"a": 1},))

    def test_empty_records_raises_valueerror(self):
        """Reject empty records list."""
        with pytest.raises(ValueError, match="records must be non-empty"):
            ar.from_records([])

    # ---- Error: columns parameter validation ----

    def test_columns_string_raises_typeerror(self):
        """Reject columns as a string."""
        with pytest.raises(
            TypeError,
            match=re.escape("columns must be a list or tuple of strings, not a string or bytes")
        ):
            ar.from_records([{"a": 1}], columns="a")

    def test_columns_with_non_string_element_raises_typeerror(self):
        """Reject columns containing non-string elements."""
        with pytest.raises(
            TypeError,
            match="columns must contain only strings"
        ):
            ar.from_records([{"a": 1}], columns=["a", 1])

    # ---- Error: dict records validation ----

    def test_dict_nested_list_value_raises_typeerror(self):
        """Reject nested list in dict record value."""
        with pytest.raises(
            TypeError,
            match=re.escape("nested values are not supported; column 'a' at row 0 contains a 'list'")
        ):
            ar.from_records([{"a": [1]}])

    def test_dict_mixed_record_types_raises_typeerror(self):
        """Reject non-dict row when dict records expected."""
        with pytest.raises(
            TypeError,
            match=re.escape("all records must be dicts, but row 1 is 'list'")
        ):
            ar.from_records([{"a": 1}, [1]])

    def test_dict_empty_columns_raises_valueerror(self):
        """Reject empty columns list with dict records."""
        with pytest.raises(
            ValueError,
            match=re.escape("columns must not be empty when records are dicts")
        ):
            ar.from_records([{"a": 1}], columns=[])

    # ---- Error: sequence records validation ----

    def test_sequence_records_no_columns_raises_valueerror(self):
        """Require columns parameter for sequence records."""
        with pytest.raises(
            ValueError,
            match="columns must be provided when records are lists or tuples"
        ):
            ar.from_records([[1, 2]])

    def test_sequence_row_length_mismatch_raises_valueerror(self):
        """Reject row with mismatched column count."""
        with pytest.raises(
            ValueError,
            match=re.escape("row 1 has 1 value(s) but 2 column(s) were provided")
        ):
            ar.from_records([[1, 2], [3]], columns=["a", "b"])

    def test_sequence_nested_list_value_raises_typeerror(self):
        """Reject nested list in sequence record value."""
        with pytest.raises(
            TypeError,
            match=re.escape("nested values are not supported; column 'b' at row 0 contains a 'list'")
        ):
            ar.from_records([[1, [2]]], columns=["a", "b"])

    def test_sequence_mixed_record_types_raises_typeerror(self):
        """Reject non-sequence row when sequence records expected."""
        with pytest.raises(
            TypeError,
            match=re.escape("all records must be the same type, but row 1 is 'dict'")
        ):
            ar.from_records([[1, 2], {"a": 1}], columns=["a", "b"])

    # ---- Error: invalid record type ----

    def test_invalid_record_type_raises_typeerror(self):
        """Reject unsupported record type."""
        with pytest.raises(
            TypeError,
            match=re.escape("records must contain dicts, lists, or tuples, got 'int'")
        ):
            ar.from_records([1])
