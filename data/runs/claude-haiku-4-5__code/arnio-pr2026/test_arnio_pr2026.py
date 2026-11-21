import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropConstantColumns:
    """Tests for arnio.cleaning.drop_constant_columns"""
    
    def test_drop_single_constant_column(self):
        """Column with one unique value should be dropped"""
        df = pd.DataFrame({'A': [1, 1, 1], 'B': [1, 2, 3]})
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ['B']
        assert len(result) == 3
        assert isinstance(result, pd.DataFrame)
    
    def test_keep_column_with_null_and_value(self):
        """Column with value and None should be kept (2 unique values)"""
        df = pd.DataFrame({'A': [1, 1, None], 'B': [1, 2, 3]})
        result = ar.drop_constant_columns(df)
        assert set(result.columns) == {'A', 'B'}
    
    def test_drop_column_all_nulls(self):
        """Column with all None/NaN should be dropped (one unique value: None)"""
        df = pd.DataFrame({'A': [None, None, None], 'B': [1, 2, 3]})
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ['B']
    
    def test_drop_multiple_constant_columns(self):
        """Multiple constant columns should all be dropped"""
        df = pd.DataFrame({
            'const1': [1, 1, 1],
            'const2': [2, 2, 2],
            'varying': [1, 2, 3],
            'const3': ['x', 'x', 'x']
        })
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ['varying']
    
    def test_no_constant_columns(self):
        """DataFrame with no constant columns should be unchanged"""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ['A', 'B']
    
    def test_all_columns_constant_preserves_row_count(self):
        """All constant columns results in zero-column frame preserving row count"""
        df = pd.DataFrame({'A': [1, 1], 'B': [2, 2]})
        result = ar.drop_constant_columns(df)
        assert len(result.columns) == 0
        assert len(result) == 2
    
    def test_empty_dataframe_zero_rows_keeps_columns(self):
        """Empty (zero-row) DataFrame keeps all columns"""
        df = pd.DataFrame({'A': [], 'B': []})
        result = ar.drop_constant_columns(df)
        assert set(result.columns) == {'A', 'B'}
        assert len(result) == 0
    
    def test_single_row_all_columns_constant(self):
        """Single row DataFrame - all columns have 1 unique value"""
        df = pd.DataFrame({'A': [1], 'B': [2]})
        result = ar.drop_constant_columns(df)
        assert len(result.columns) == 0
        assert len(result) == 1
    
    def test_string_constant_column(self):
        """String columns can be detected as constant"""
        df = pd.DataFrame({'A': ['x', 'x', 'x'], 'B': ['a', 'b', 'c']})
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ['B']
    
    def test_float_constant_column(self):
        """Float constant columns can be detected"""
        df = pd.DataFrame({'A': [1.5, 1.5, 1.5], 'B': [1.1, 2.2, 3.3]})
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ['B']
    
    def test_returns_same_type_as_input(self):
        """Returns pd.DataFrame when given pd.DataFrame"""
        df = pd.DataFrame({'A': [1, 1], 'B': [1, 2]})
        result = ar.drop_constant_columns(df)
        assert isinstance(result, pd.DataFrame)


class TestDropEmptyColumns:
    """Tests for arnio.cleaning.drop_empty_columns"""
    
    def test_drop_all_none_column(self):
        """Column with all None values should be dropped"""
        frame = ar.from_dict({'A': [None, None, None], 'B': [1, 2, 3]})
        result = ar.drop_empty_columns(frame)
        assert list(result.columns) == ['B']
    
    def test_drop_all_empty_strings(self):
        """Column with all empty strings should be dropped"""
        frame = ar.from_dict({'A': ['', '', ''], 'B': ['a', 'b', 'c']})
        result = ar.drop_empty_columns(frame)
        assert list(result.columns) == ['B']
    
    def test_drop_whitespace_only_strings(self):
        """Column with only whitespace strings should be dropped"""
        frame = ar.from_dict({'A': ['  ', '\t', '\n'], 'B': ['a', 'b', 'c']})
        result = ar.drop_empty_columns(frame)
        assert list(result.columns) == ['B']
    
    def test_keep_column_with_one_non_empty_value(self):
        """Column with at least one non-empty/non-null value should be kept"""
        frame = ar.from_dict({'A': [None, None, 'x'], 'B': ['', '', '']})
        result = ar.drop_empty_columns(frame)
        assert list(result.columns) == ['A']
    
    def test_mixed_empty_and_nonempty_columns(self):
        """Multiple empty columns dropped, non-empty kept"""
        frame = ar.from_dict({
            'empty_null': [None, None],
            'empty_strings': ['', ''],
            'empty_whitespace': ['  ', '  '],
            'has_value': ['a', 'b']
        })
        result = ar.drop_empty_columns(frame)
        assert list(result.columns) == ['has_value']
    
    def test_zero_row_frame_preserves_columns(self):
        """Zero-row frame preserves all columns"""
        frame = ar.from_dict({'A': [], 'B': []})
        result = ar.drop_empty_columns(frame)
        assert set(result.columns) == {'A', 'B'}
        assert len(result) == 0
    
    def test_all_columns_empty_preserves_row_count(self):
        """All empty columns results in zero-column frame with row count preserved"""
        frame = ar.from_dict({'A': [None, None], 'B': ['', '']})
        result = ar.drop_empty_columns(frame)
        assert len(result.columns) == 0
        assert len(result) == 2
