import pytest
import pandas as pd
import arnio as ar


class TestFromDict:
    """Tests for arnio.from_dict and ArFrame.from_dict"""

    def test_basic_from_dict_with_mixed_types(self):
        """Test basic from_dict with mixed data types."""
        data = {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "salary": [50000.0, 60000.0, 75000.0],
        }
        frame = ar.from_dict(data)

        assert frame.shape == (3, 3)
        assert frame.columns == ["name", "age", "salary"]

        df = ar.to_pandas(frame)
        assert list(df["name"]) == ["Alice", "Bob", "Charlie"]
        assert list(df["age"]) == [25, 30, 35]

    def test_from_dict_class_method(self):
        """Test calling from_dict as a class method on ArFrame."""
        data = {"id": [1, 2, 3], "value": [10, 20, 30]}
        frame = ar.ArFrame.from_dict(data)

        assert frame.shape == (3, 2)
        df = ar.to_pandas(frame)
        assert list(df["id"]) == [1, 2, 3]

    def test_from_dict_single_column(self):
        """Test from_dict with a single column."""
        data = {"values": [1, 2, 3, 4, 5]}
        frame = ar.from_dict(data)

        assert frame.shape == (5, 1)
        assert frame.columns == ["values"]

    def test_from_dict_single_row(self):
        """Test from_dict with a single row."""
        data = {"a": [1], "b": ["hello"], "c": [3.14]}
        frame = ar.from_dict(data)

        assert frame.shape == (1, 3)

    def test_from_dict_with_none_values(self):
        """Test from_dict with None values in columns."""
        data = {"name": ["Alice", None, "Charlie"], "age": [25, 30, None]}
        frame = ar.from_dict(data)

        assert frame.shape == (3, 2)
        df = ar.to_pandas(frame)
        assert pd.isna(df.loc[1, "name"])
        assert pd.isna(df.loc[2, "age"])

    def test_from_dict_empty_lists(self):
        """Test from_dict with empty lists for each column."""
        data = {"a": [], "b": [], "c": []}
        frame = ar.from_dict(data)

        assert frame.shape == (0, 3)
        assert frame.columns == ["a", "b", "c"]

    def test_from_dict_unequal_column_lengths_raises(self):
        """Test that unequal column lengths raise an error."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30, 35]}

        with pytest.raises((ValueError, TypeError)):
            ar.from_dict(data)

    def test_from_dict_empty_dict(self):
        """Test from_dict with an empty dictionary."""
        frame = ar.from_dict({})

        assert frame.shape[0] == 0
        assert len(frame.columns) == 0

    def test_from_dict_with_boolean_values(self):
        """Test from_dict with boolean column."""
        data = {"is_active": [True, False, True]}
        frame = ar.from_dict(data)

        assert frame.shape == (3, 1)
        df = ar.to_pandas(frame)
        assert df["is_active"].dtype in (bool, object)

    def test_from_dict_with_tuples(self):
        """Test from_dict accepts tuples as column values."""
        data = {"x": (1, 2, 3), "y": (4, 5, 6)}
        frame = ar.from_dict(data)

        assert frame.shape == (3, 2)
        df = ar.to_pandas(frame)
        assert list(df["x"]) == [1, 2, 3]

    def test_from_dict_preserves_column_order(self):
        """Test that column order is preserved (Python 3.7+)."""
        data = {"z": [1], "a": [2], "m": [3]}
        frame = ar.from_dict(data)

        assert frame.columns == ["z", "a", "m"]

    def test_from_dict_numeric_strings_preserved(self):
        """Test that numeric-looking strings are preserved as strings."""
        data = {"id_str": ["001", "002", "003"], "value": [10, 20, 30]}
        frame = ar.from_dict(data)

        df = ar.to_pandas(frame)
        assert list(df["id_str"]) == ["001", "002", "003"]

    def test_from_dict_with_nested_structures_raises(self):
        """Test that nested structures raise an error."""
        data = {"simple": [1, 2], "nested": [[1, 2], [3, 4]]}

        with pytest.raises((TypeError, ValueError)):
            ar.from_dict(data)

    def test_from_dict_roundtrip_pandas(self):
        """Test roundtrip: dict → ArFrame → pandas → ArFrame → pandas."""
        original_data = {"name": ["Alice", "Bob"], "age": [25, 30]}

        frame1 = ar.from_dict(original_data)
        df1 = ar.to_pandas(frame1)
        frame2 = ar.from_pandas(df1)
        df2 = ar.to_pandas(frame2)

        assert list(df2["name"]) == ["Alice", "Bob"]
        assert list(df2["age"]) == [25, 30]

    def test_from_dict_usable_in_pipeline(self):
        """Test that from_dict result can be used directly in a pipeline."""
        data = {"value": [1, 2, 3], "name": ["a", "b", "c"]}
        frame = ar.from_dict(data)

        clean = ar.pipeline(frame, [("strip_whitespace",)])
        df = ar.to_pandas(clean)

        assert frame.shape == clean.shape
