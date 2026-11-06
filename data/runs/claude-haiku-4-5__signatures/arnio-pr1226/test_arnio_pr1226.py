import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestToPandas:
    """Test suite for arnio.convert.to_pandas function"""

    def test_basic_conversion_returns_dataframe(self):
        """Test that to_pandas returns a pandas DataFrame"""
        arframe = ar.ArFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = ar.convert.to_pandas(arframe)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_column_names(self):
        """Test that column names are preserved during conversion"""
        arframe = ar.ArFrame({'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]})
        result = ar.convert.to_pandas(arframe)
        assert list(result.columns) == ['col1', 'col2', 'col3']

    def test_preserves_data_values(self):
        """Test that data values are correctly transferred"""
        data = {'a': [1, 2, 3], 'b': [4.5, 5.5, 6.5]}
        arframe = ar.ArFrame(data)
        result = ar.convert.to_pandas(arframe)
        assert result['a'].tolist() == [1, 2, 3]
        assert result['b'].tolist() == [4.5, 5.5, 6.5]

    def test_copy_parameter_keyword_only(self):
        """Test that copy parameter must be passed as keyword argument"""
        arframe = ar.ArFrame({'a': [1, 2, 3]})
        # Should work with keyword argument
        result = ar.convert.to_pandas(arframe, copy=False)
        assert isinstance(result, pd.DataFrame)
        # Should raise TypeError when passed positionally
        with pytest.raises(TypeError):
            ar.convert.to_pandas(arframe, False)

    def test_copy_true_creates_independent_copy(self):
        """Test that copy=True creates data independent from original"""
        arframe = ar.ArFrame({'a': [1, 2, 3]})
        result = ar.convert.to_pandas(arframe, copy=True)
        # Modify the DataFrame
        result.iloc[0, 0] = 999
        # Re-convert original to verify it wasn't affected
        result_check = ar.convert.to_pandas(arframe, copy=False)
        assert result_check.iloc[0, 0] == 1

    def test_copy_false_default(self):
        """Test that copy=False is the default behavior"""
        arframe = ar.ArFrame({'a': [1, 2, 3]})
        result1 = ar.convert.to_pandas(arframe)
        result2 = ar.convert.to_pandas(arframe, copy=False)
        # Both should be DataFrames
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)

    def test_empty_arframe(self):
        """Test conversion of empty ArFrame"""
        arframe = ar.ArFrame({})
        result = ar.convert.to_pandas(arframe)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_column(self):
        """Test conversion with single column"""
        arframe = ar.ArFrame({'single': [10, 20, 30]})
        result = ar.convert.to_pandas(arframe)
        assert len(result.columns) == 1
        assert result.columns[0] == 'single'
        assert len(result) == 3

    def test_single_row(self):
        """Test conversion with single row"""
        arframe = ar.ArFrame({'a': [1], 'b': [2], 'c': [3]})
        result = ar.convert.to_pandas(arframe)
        assert len(result) == 1
        assert list(result.iloc[0]) == [1, 2, 3]

    def test_preserves_shape(self):
        """Test that DataFrame shape matches ArFrame shape"""
        arframe = ar.ArFrame({
            'x': [1, 2, 3, 4],
            'y': [5, 6, 7, 8],
            'z': [9, 10, 11, 12]
        })
        result = ar.convert.to_pandas(arframe)
        assert result.shape == (4, 3)

    def test_preserves_index(self):
        """Test that custom index is preserved"""
        arframe = ar.ArFrame({'a': [100, 200, 300]}, index=['x', 'y', 'z'])
        result = ar.convert.to_pandas(arframe)
        assert list(result.index) == ['x', 'y', 'z']

    def test_multiple_data_types(self):
        """Test conversion with mixed data types"""
        arframe = ar.ArFrame({
            'ints': [1, 2, 3],
            'floats': [1.5, 2.5, 3.5],
            'strings': ['a', 'b', 'c'],
            'bools': [True, False, True]
        })
        result = ar.convert.to_pandas(arframe)
        assert len(result.columns) == 4
        assert len(result) == 3

    def test_handles_null_values(self):
        """Test that null/NA values are preserved"""
        arframe = ar.ArFrame({
            'with_nulls': [1, None, 3],
            'also_nulls': [None, 2.5, None]
        })
        result = ar.convert.to_pandas(arframe)
        assert pd.isna(result.iloc[1, 0])
        assert pd.isna(result.iloc[0, 1])
        assert pd.isna(result.iloc[2, 1])

    def test_result_is_mutable(self):
        """Test that returned DataFrame can be modified"""
        arframe = ar.ArFrame({'a': [1, 2, 3]})
        result = ar.convert.to_pandas(arframe)
        result.iloc[0, 0] = 999
        assert result.iloc[0, 0] == 999

    def test_large_frame_conversion(self):
        """Test conversion of larger frames"""
        data = {f'col_{i}': list(range(100)) for i in range(10)}
        arframe = ar.ArFrame(data)
        result = ar.convert.to_pandas(arframe)
        assert result.shape == (100, 10)
