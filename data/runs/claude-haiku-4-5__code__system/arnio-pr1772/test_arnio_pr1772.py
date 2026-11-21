import pytest
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import arnio as ar


class TestArFrameFromRecordsBasicSuccess:
    """Test successful creation of ArFrame from various record formats."""
    
    def test_from_dicts_inferred_columns(self):
        """Create ArFrame from dicts with inferred columns."""
        records = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        frame = ar.frame.ArFrame.from_records(records)
        assert isinstance(frame, ar.frame.ArFrame)
        df = frame.to_pandas()
        assert "name" in df.columns
        assert "age" in df.columns
        assert len(df) == 2
        assert df.iloc[0]["name"] == "Alice"
    
    def test_from_dicts_explicit_columns(self):
        """Create ArFrame from dicts with explicit column subset."""
        records = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"},
        ]
        columns = ["name", "age"]
        frame = ar.frame.ArFrame.from_records(records, columns=columns)
        df = frame.to_pandas()
        assert list(df.columns) == ["name", "age"]
        assert "city" not in df.columns
        assert len(df) == 2
    
    def test_from_lists_with_columns(self):
        """Create ArFrame from list records with columns."""
        records = [
            ["Alice", 30, "NYC"],
            ["Bob", 25, "LA"],
            ["Charlie", 35, "SF"],
        ]
        columns = ["name", "age", "city"]
        frame = ar.frame.ArFrame.from_records(records, columns=columns)
        df = frame.to_pandas()
        assert list(df.columns) == ["name", "age", "city"]
        assert len(df) == 3
        assert df.iloc[2]["name"] == "Charlie"
    
    def test_from_tuples_with_columns(self):
        """Create ArFrame from tuple records with columns."""
        records = [
            ("Alice", 30),
            ("Bob", 25),
        ]
        columns = ["name", "age"]
        frame = ar.frame.ArFrame.from_records(records, columns=columns)
        df = frame.to_pandas()
        assert list(df.columns) == ["name", "age"]
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[1]["age"] == 25
    
    def test_single_record(self):
        """Create ArFrame from single record."""
        records = [{"x": 1, "y": 2.5, "z": "text"}]
        frame = ar.frame.ArFrame.from_records(records)
        df = frame.to_pandas()
        assert len(df) == 1
        assert df.iloc[0]["x"] == 1
    
    def test_various_scalar_types(self):
        """Test records with various scalar types (int, float, str, bool, None)."""
        records = [
            {"int_val": 42, "float_val": 3.14, "str_val": "hello", "bool_val": True, "none_val": None},
            {"int_val": -1, "float_val": 2.71, "str_val": "world", "bool_val": False, "none_val": None},
        ]
        frame = ar.frame.ArFrame.from_records(records)
        df = frame.to_pandas()
        assert df.iloc[0]["int_val"] == 42
        assert abs(df.iloc[0]["float_val"] - 3.14) < 0.01
        assert df.iloc[0]["str_val"] == "hello"
        assert df.iloc[0]["bool_val"] is True
    
    def test_unicode_data(self):
        """Test records with unicode column names and values."""
        records = [
            {"名前": "太郎", "年齢": 30},
            {"名前": "花子", "年齢": 25},
        ]
        frame = ar.frame.ArFrame.from_records(records)
        df = frame.to_pandas()
        assert df.iloc[0]["名前"] == "太郎"
        assert df.iloc[1]["年齢"] == 25
    
    def test_large_dataset(self):
        """Test from_records with larger dataset (100 rows)."""
        records = [
            {"id": i, "value": i * 2.5, "category": f"cat_{i % 3}"}
            for i in range(100)
        ]
        frame = ar.frame.ArFrame.from_records(records)
        df = frame.to_pandas()
        assert len(df) == 100
        assert df.iloc[50]["id"] == 50


class TestArFrameFromRecordsErrors:
    """Test error conditions and validation."""
    
    def test_records_not_list(self):
        """TypeError when records is not a list."""
        with pytest.raises(TypeError, match="records must be a list"):
            ar.frame.ArFrame.from_records(("a", "b"))
    
    def test_records_empty(self):
        """ValueError when records is empty."""
        with pytest.raises(ValueError, match="records must be non-empty"):
            ar.frame.ArFrame.from_records([])
    
    def test_columns_is_string(self):
        """TypeError when columns is passed as string."""
        records = [["a", "b"]]
        with pytest.raises(TypeError, match="columns must be a list or tuple"):
            ar.frame.ArFrame.from_records(records, columns="ab")
    
    def test_columns_is_dict(self):
        """TypeError when columns is not list/tuple."""
        records = [["a", "b"]]
        with pytest.raises(TypeError, match="columns must be a list or tuple"):
            ar.frame.ArFrame.from_records(records, columns={"x": 1})
    
    def test_columns_has_non_string(self):
        """TypeError when columns contains non-string elements."""
        records = [["a", "b"]]
        with pytest.raises(TypeError, match="columns must contain only strings"):
            ar.frame.ArFrame.from_records(records, columns=["name", 42])
    
    def test_mixed_dict_and_list_records(self):
        """TypeError when mixing dict and list record types."""
        records = [
            {"name": "Alice"},
            ["Bob", "value"],
        ]
        with pytest.raises(TypeError, match="all records must be dicts"):
            ar.frame.ArFrame.from_records(records)
    
    def test_nested_list_in_value(self):
        """TypeError when record value contains nested list."""
        records = [
            {"data": [1, 2, 3]},
        ]
        with pytest.raises(TypeError, match="nested values are not supported"):
            ar.frame.ArFrame.from_records(records)
    
    def test_nested_dict_in_value(self):
        """TypeError when record value contains nested dict."""
        records = [
            {"metadata": {"key": "value"}},
        ]
        with pytest.raises(TypeError, match="nested values are not supported"):
            ar.frame.ArFrame.from_records(records)
    
    def test_missing_columns_for_lists(self):
        """ValueError when columns not provided for list records."""
        records = [["Alice", 30]]
        with pytest.raises(ValueError, match="columns must be provided"):
            ar.frame.ArFrame.from_records(records)
    
    def test_row_length_mismatch(self):
        """ValueError when row length differs from columns count."""
        records = [
            ["Alice", 30],
            ["Bob"],
        ]
        with pytest.raises(ValueError, match="row 1 has 1 value"):
            ar.frame.ArFrame.from_records(records, columns=["name", "age"])
    
    def test_empty_columns_with_dicts(self):
        """ValueError when empty columns list provided for dicts."""
        records = [{"x": 1}]
        with pytest.raises(ValueError, match="columns must not be empty"):
            ar.frame.ArFrame.from_records(records, columns=[])
    
    def test_invalid_first_record_type(self):
        """TypeError when first record is not dict/list/tuple."""
        records = [42]
        with pytest.raises(TypeError, match="records must contain dicts, lists, or tuples"):
            ar.frame.ArFrame.from_records(records)
    
    def test_mixed_list_and_tuple(self):
        """TypeError when mixing list and tuple records."""
        records = [
            ["Alice", 30],
            ("Bob", 25),
        ]
        with pytest.raises(TypeError, match="all records must be the same type"):
            ar.frame.ArFrame.from_records(records, columns=["name", "age"])


class TestArFrameFromRecordsEndToEnd:
    """Test complete workflows and file I/O."""
    
    def test_create_and_save_csv(self, tmp_path):
        """Create ArFrame from records and save to CSV."""
        records = [
            {"product": "Apple", "price": 1.5, "qty": 100},
            {"product": "Banana", "price": 0.75, "qty": 200},
            {"product": "Orange", "price": 2.0, "qty": 150},
        ]
        
        frame = ar.frame.ArFrame.from_records(records)
        df = frame.to_pandas()
        
        csv_file = tmp_path / "products.csv"
        df.to_csv(csv_file, index=False)
        
        assert csv_file.exists()
        saved_df = pd.read_csv(csv_file)
        assert len(saved_df) == 3
        assert "product" in saved_df.columns
        assert saved_df.iloc[0]["product"] == "Apple"
    
    def test_convert_to_pandas_dataframe(self):
        """Convert ArFrame to pandas DataFrame."""
        records = [
            ["id", "name", "score"],
            [1, "Alice", 95.5],
            [2, "Bob", 87.3],
        ]
        columns = ["col1", "col2", "col3"]
        frame = ar.frame.ArFrame.from_records(records[1:], columns=columns)
        df = frame.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.iloc[0]["col2"] == "Alice"
