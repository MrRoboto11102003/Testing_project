import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestArFrameFromRecords:
    """Tests for ArFrame.from_records class method."""

    # ========== Normal cases: dict records ==========

    def test_from_records_dict_basic(self):
        """Create ArFrame from a list of dicts."""
        records = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]
        frame = ar.from_records(records)
        assert frame.shape == (2, 2)
        assert set(frame.columns) == {"id", "name"}

    def test_from_records_dict_with_explicit_columns(self):
        """Create ArFrame from dicts with explicit column order."""
        records = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]
        columns = ["name", "id"]
        frame = ar.from_records(records, columns=columns)
        assert frame.columns == ["name", "id"]

    def test_from_records_dict_with_missing_keys_becomes_null(self):
        """Dict records with missing keys become nulls."""
        records = [
            {"id": 1, "name": "alice"},
            {"id": 2},  # missing 'name'
        ]
        frame = ar.from_records(records)
        df = ar.to_pandas(frame)
        assert pd.isna(df.iloc[1]["name"])

    def test_from_records_single_dict_record(self):
        """Single dict record is valid."""
        records = [{"id": 1, "name": "alice"}]
        frame = ar.from_records(records)
        assert frame.shape == (1, 2)

    def test_from_records_dict_numeric_types_preserved(self):
        """Numeric columns are inferred and preserved."""
        records = [
            {"id": 1, "score": 95.5},
            {"id": 2, "score": 88.3},
        ]
        frame = ar.from_records(records)
        dtypes = frame.dtypes
        assert dtypes["id"] == "int64"
        assert dtypes["score"] == "float64"

    # ========== Normal cases: list/tuple records ==========

    def test_from_records_list_basic(self):
        """Create ArFrame from a list of lists with columns."""
        records = [
            [1, "alice"],
            [2, "bob"],
        ]
        columns = ["id", "name"]
        frame = ar.from_records(records, columns=columns)
        assert frame.shape == (2, 2)
        assert frame.columns == columns

    def test_from_records_tuple_basic(self):
        """Create ArFrame from a list of tuples with columns."""
        records = [
            (1, "alice"),
            (2, "bob"),
        ]
        columns = ["id", "name"]
        frame = ar.from_records(records, columns=columns)
        assert frame.shape == (2, 2)
        assert frame.columns == columns

    def test_from_records_list_single_row(self):
        """Single list record is valid."""
        records = [[1, "alice"]]
        columns = ["id", "name"]
        frame = ar.from_records(records, columns=columns)
        assert frame.shape == (1, 2)

    # ========== Error cases: TypeError ==========

    def test_from_records_not_list_raises_typeerror(self):
        """Records must be a list, not dict or tuple."""
        with pytest.raises(TypeError, match="records must be a list"):
            ar.from_records({"id": 1})

    def test_from_records_columns_as_string_raises_typeerror(self):
        """Columns must be list/tuple, not a string."""
        records = [{"id": 1, "name": "alice"}]
        with pytest.raises(TypeError, match="columns must be a list or tuple"):
            ar.from_records(records, columns="id,name")

    def test_from_records_columns_with_non_string_raises_typeerror(self):
        """Columns must contain only strings, not integers."""
        records = [{"id": 1, "name": "alice"}]
        with pytest.raises(TypeError, match="columns must contain only strings"):
            ar.from_records(records, columns=["id", 1])

    def test_from_records_mixed_dict_and_list_raises_typeerror(self):
        """All records must be same type (all dicts or all lists/tuples)."""
        records = [
            {"id": 1, "name": "alice"},
            [2, "bob"],
        ]
        with pytest.raises(TypeError, match="all records must be"):
            ar.from_records(records)

    def test_from_records_nested_list_in_cell_raises_typeerror(self):
        """Nested lists in cell values are not allowed."""
        records = [
            {"id": 1, "tags": ["python", "data"]},
        ]
        with pytest.raises(TypeError, match="nested values are not supported"):
            ar.from_records(records)

    def test_from_records_nested_dict_in_cell_raises_typeerror(self):
        """Nested dicts in cell values are not allowed."""
        records = [
            {"id": 1, "meta": {"key": "value"}},
        ]
        with pytest.raises(TypeError, match="nested values are not supported"):
            ar.from_records(records)

    # ========== Error cases: ValueError ==========

    def test_from_records_empty_list_raises_valueerror(self):
        """Empty records list is not allowed."""
        with pytest.raises(ValueError, match="records must be non-empty"):
            ar.from_records([])

    def test_from_records_list_without_columns_raises_valueerror(self):
        """Columns must be provided when records are lists or tuples."""
        records = [[1, "alice"], [2, "bob"]]
        with pytest.raises(ValueError, match="columns must be provided"):
            ar.from_records(records)

    def test_from_records_row_length_mismatch_raises_valueerror(self):
        """Row length must match columns length."""
        records = [
            [1, "alice"],
            [2, "bob", "extra"],
        ]
        columns = ["id", "name"]
        with pytest.raises(ValueError, match="has.*value.*but.*column"):
            ar.from_records(records, columns=columns)

    def test_from_records_dict_empty_columns_raises_valueerror(self):
        """Empty columns list with dict records should raise ValueError."""
        records = [{"id": 1, "name": "alice"}]
        with pytest.raises(ValueError, match="columns must not be empty"):
            ar.from_records(records, columns=[])
