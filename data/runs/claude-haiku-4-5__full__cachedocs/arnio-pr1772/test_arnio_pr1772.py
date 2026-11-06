import pytest
import pandas as pd
import arnio as ar


class TestArFrameFromRecords:
    """Tests for ArFrame.from_records() - builds ArFrame from list of records."""

    # Valid dict-based records
    def test_dict_records_infers_columns(self):
        """Dict records automatically infer columns from keys."""
        records = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]
        frame = ar.from_records(records)
        assert frame.shape == (2, 2)
        assert set(frame.columns) == {"id", "name"}

    def test_dict_records_with_explicit_columns(self):
        """Dict records respect explicit column parameter."""
        records = [{"id": 1, "name": "alice"}]
        frame = ar.from_records(records, columns=["id", "name"])
        assert frame.columns == ["id", "name"]

    def test_dict_records_missing_keys_become_null(self):
        """Missing keys in dict records are filled with None."""
        records = [
            {"id": 1, "name": "alice", "age": 30},
            {"id": 2, "name": "bob"},  # missing 'age'
        ]
        frame = ar.from_records(records)
        df = ar.to_pandas(frame)
        # Row 1 should have null for age
        assert pd.isna(df.iloc[1]["age"])

    # Valid sequence-based records
    def test_list_records_require_columns(self):
        """List records must be accompanied by column names."""
        records = [[1, "alice"], [2, "bob"]]
        columns = ["id", "name"]
        frame = ar.from_records(records, columns=columns)
        assert frame.shape == (2, 2)
        assert frame.columns == columns

    def test_tuple_records_require_columns(self):
        """Tuple records must be accompanied by column names."""
        records = [(1, "alice"), (2, "bob")]
        columns = ["id", "name"]
        frame = ar.from_records(records, columns=columns)
        assert frame.shape == (2, 2)

    # Edge cases: valid
    def test_single_row(self):
        """Single row record is valid."""
        records = [{"id": 1, "name": "alice"}]
        frame = ar.from_records(records)
        assert frame.shape == (1, 2)

    def test_single_column(self):
        """Single column from records is valid."""
        records = [{"id": 1}, {"id": 2}, {"id": 3}]
        frame = ar.from_records(records)
        assert frame.shape == (3, 1)

    def test_converts_to_pandas_dataframe(self):
        """Result converts cleanly to pandas DataFrame."""
        records = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        frame = ar.from_records(records)
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    # Error cases: ValueError
    def test_empty_records_raises_valueerror(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError):
            ar.from_records([])

    def test_list_records_without_columns_raises_valueerror(self):
        """List records without columns parameter raises ValueError."""
        with pytest.raises(ValueError):
            ar.from_records([[1, "alice"], [2, "bob"]])

    def test_tuple_records_without_columns_raises_valueerror(self):
        """Tuple records without columns parameter raises ValueError."""
        with pytest.raises(ValueError):
            ar.from_records([(1, "alice"), (2, "bob")])

    def test_row_length_mismatch_raises_valueerror(self):
        """Row with length not matching columns raises ValueError."""
        records = [[1, "alice"], [2, "bob", "extra"]]
        with pytest.raises(ValueError):
            ar.from_records(records, columns=["id", "name"])

    # Error cases: TypeError
    def test_mixed_record_types_raises_typeerror(self):
        """Mixing dict and non-dict records raises TypeError."""
        records = [
            {"id": 1, "name": "alice"},
            [2, "bob"],
        ]
        with pytest.raises(TypeError):
            ar.from_records(records)

    def test_nested_list_cell_raises_typeerror(self):
        """Nested list as cell value raises TypeError."""
        records = [{"id": 1, "tags": ["a", "b"]}]
        with pytest.raises(TypeError):
            ar.from_records(records)

    def test_nested_dict_cell_raises_typeerror(self):
        """Nested dict as cell value raises TypeError."""
        records = [{"id": 1, "meta": {"key": "val"}}]
        with pytest.raises(TypeError):
            ar.from_records(records)
