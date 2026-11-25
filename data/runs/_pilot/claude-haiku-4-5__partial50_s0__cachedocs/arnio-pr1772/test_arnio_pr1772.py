import pytest
import arnio as ar
import pandas as pd


class TestFromRecords:
    """Tests for arnio.from_records and ArFrame.from_records"""

    # === Basic functionality ===

    def test_from_records_dict_automatic_columns(self):
        """Build ArFrame from list of dicts with automatic column inference."""
        records = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]
        frame = ar.from_records(records)
        assert frame.shape == (2, 2)
        assert set(frame.columns) == {"id", "name"}

    def test_from_records_list_with_columns(self):
        """Build ArFrame from list of lists with explicit columns."""
        records = [
            [1, "alice"],
            [2, "bob"],
        ]
        frame = ar.from_records(records, columns=["id", "name"])
        assert frame.shape == (2, 2)
        assert frame.columns == ["id", "name"]

    def test_from_records_tuple_with_columns(self):
        """Build ArFrame from list of tuples with explicit columns."""
        records = [
            (1, "alice"),
            (2, "bob"),
        ]
        frame = ar.from_records(records, columns=["id", "name"])
        assert frame.shape == (2, 2)
        assert frame.columns == ["id", "name"]

    def test_from_records_single_record(self):
        """Build ArFrame from a single record."""
        records = [{"id": 1, "name": "alice"}]
        frame = ar.from_records(records)
        assert frame.shape == (1, 2)

    def test_from_records_returns_arframe_instance(self):
        """from_records returns an ArFrame instance."""
        records = [{"id": 1, "name": "alice"}]
        frame = ar.from_records(records)
        assert isinstance(frame, ar.ArFrame)

    # === Dict-specific behavior ===

    def test_from_records_dict_missing_keys_filled_with_none(self):
        """Missing keys in dict records are filled with None."""
        records = [
            {"id": 1, "name": "alice", "age": 30},
            {"id": 2, "name": "bob"},  # missing 'age'
        ]
        frame = ar.from_records(records)
        assert frame.shape == (2, 3)
        df = ar.to_pandas(frame)
        assert pd.isna(df.loc[1, "age"])

    def test_from_records_dict_with_explicit_columns_ordering(self):
        """Dict records respect explicit columns parameter for ordering."""
        records = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]
        frame = ar.from_records(records, columns=["name", "id"])
        assert frame.columns == ["name", "id"]

    # === Error conditions ===

    def test_from_records_empty_list_raises_valueerror(self):
        """Empty records list raises ValueError."""
        with pytest.raises(ValueError):
            ar.from_records([])

    def test_from_records_list_without_columns_raises_valueerror(self):
        """List/tuple records without columns parameter raises ValueError."""
        records = [[1, "alice"], [2, "bob"]]
        with pytest.raises(ValueError):
            ar.from_records(records)

    def test_from_records_nested_list_raises_typeerror(self):
        """Cell value that is a list raises TypeError."""
        records = [{"id": 1, "data": [1, 2, 3]}]
        with pytest.raises(TypeError):
            ar.from_records(records)

    def test_from_records_nested_dict_raises_typeerror(self):
        """Cell value that is a dict raises TypeError."""
        records = [{"id": 1, "nested": {"key": "value"}}]
        with pytest.raises(TypeError):
            ar.from_records(records)

    def test_from_records_mixed_record_types_raises_typeerror(self):
        """Mixed element types (dict and list in same list) raises TypeError."""
        records = [
            {"id": 1, "name": "alice"},
            [2, "bob"],
        ]
        with pytest.raises(TypeError):
            ar.from_records(records)

    def test_from_records_row_length_mismatch_raises_valueerror(self):
        """Row with mismatched length vs columns raises ValueError."""
        records = [
            [1, "alice"],
            [2, "bob", "extra"],
        ]
        with pytest.raises(ValueError):
            ar.from_records(records, columns=["id", "name"])

    def test_from_records_mixed_data_types(self):
        """Records with mixed numeric and string types are handled correctly."""
        records = [
            {"id": 1, "score": 95.5, "name": "alice"},
            {"id": 2, "score": 88.0, "name": "bob"},
        ]
        frame = ar.from_records(records)
        assert frame.shape == (2, 3)
        assert "id" in frame.columns
        assert "score" in frame.columns
        assert "name" in frame.columns

    def test_from_records_with_none_values(self):
        """Records with None values are handled correctly."""
        records = [
            {"id": 1, "name": None},
            {"id": 2, "name": "bob"},
        ]
        frame = ar.from_records(records)
        assert frame.shape == (2, 2)
