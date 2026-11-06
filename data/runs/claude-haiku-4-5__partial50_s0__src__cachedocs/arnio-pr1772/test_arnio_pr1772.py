import pytest
import pandas as pd
import arnio as ar


class TestFromRecords:
    """Tests for ar.from_records (arnio.frame.ArFrame.from_records)"""
    
    def test_dict_records_inferred_columns(self):
        """Build frame from dict records with inferred columns"""
        records = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        frame = ar.from_records(records)
        df = ar.to_pandas(frame)
        assert df.shape == (2, 2)
        assert set(df.columns) == {"id", "name"}
    
    def test_dict_records_explicit_columns(self):
        """Build frame from dict records with explicit columns"""
        records = [
            {"id": 1, "name": "Alice", "extra": "x"},
            {"id": 2, "name": "Bob", "extra": "y"},
        ]
        frame = ar.from_records(records, columns=["id", "name"])
        df = ar.to_pandas(frame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["id", "name"]
    
    def test_dict_records_missing_keys_filled_with_nulls(self):
        """Missing keys in dicts are filled with null"""
        records = [
            {"id": 1, "name": "Alice"},
            {"id": 2},
        ]
        frame = ar.from_records(records)
        df = ar.to_pandas(frame)
        assert df.shape == (2, 2)
        assert pd.isna(df.loc[1, "name"])
    
    def test_list_records_with_columns(self):
        """Build frame from list records"""
        records = [[1, "Alice"], [2, "Bob"]]
        frame = ar.from_records(records, columns=["id", "name"])
        df = ar.to_pandas(frame)
        assert df.shape == (2, 2)
        assert df["id"].tolist() == [1, 2]
    
    def test_tuple_records_with_columns(self):
        """Build frame from tuple records"""
        records = [(1, "Alice"), (2, "Bob")]
        frame = ar.from_records(records, columns=["id", "name"])
        df = ar.to_pandas(frame)
        assert df.shape == (2, 2)
        assert df["id"].tolist() == [1, 2]
    
    def test_records_not_list_raises_typeerror(self):
        """TypeError when records is not a list"""
        with pytest.raises(TypeError, match="records must be a list"):
            ar.from_records({"id": 1})
    
    def test_empty_records_raises_valueerror(self):
        """ValueError when records is empty"""
        with pytest.raises(ValueError, match="records must be non-empty"):
            ar.from_records([])
    
    def test_columns_string_raises_typeerror(self):
        """TypeError when columns is a string"""
        records = [{"id": 1}]
        with pytest.raises(TypeError, match="columns must be a list or tuple"):
            ar.from_records(records, columns="id")
    
    def test_columns_contains_non_strings_raises_typeerror(self):
        """TypeError when columns contains non-string elements"""
        records = [{"id": 1}]
        with pytest.raises(TypeError, match="columns must contain only strings"):
            ar.from_records(records, columns=["id", 1])
    
    def test_empty_columns_list_for_dicts_raises_valueerror(self):
        """ValueError when columns is empty list for dict records"""
        records = [{"id": 1}]
        with pytest.raises(ValueError, match="columns must not be empty"):
            ar.from_records(records, columns=[])
    
    def test_mixed_dict_and_list_records_raises_typeerror(self):
        """TypeError when mixing dict and list records"""
        records = [
            {"id": 1},
            [2],
        ]
        with pytest.raises(TypeError, match="all records must be"):
            ar.from_records(records)
    
    def test_row_length_mismatch_raises_valueerror(self):
        """ValueError when row length doesn't match columns"""
        records = [
            [1, "Alice"],
            [2],
        ]
        with pytest.raises(ValueError, match="has.*value.*column"):
            ar.from_records(records, columns=["id", "name"])
    
    def test_nested_list_in_cell_raises_typeerror(self):
        """TypeError when cell contains a list"""
        records = [{"id": 1, "tags": ["a", "b"]}]
        with pytest.raises(TypeError, match="nested values are not supported"):
            ar.from_records(records)
    
    def test_nested_dict_in_cell_raises_typeerror(self):
        """TypeError when cell contains a dict"""
        records = [{"id": 1, "meta": {"key": "value"}}]
        with pytest.raises(TypeError, match="nested values are not supported"):
            ar.from_records(records)
    
    def test_list_records_without_columns_raises_valueerror(self):
        """ValueError when columns not provided for list records"""
        records = [[1, "Alice"], [2, "Bob"]]
        with pytest.raises(ValueError, match="columns must be provided"):
            ar.from_records(records)
