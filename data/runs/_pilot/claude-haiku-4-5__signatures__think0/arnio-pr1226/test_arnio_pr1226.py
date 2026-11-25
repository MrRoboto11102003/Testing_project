import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestToPandas:
    """Test suite for arnio.convert.to_pandas function."""

    def test_to_pandas_basic_conversion(self):
        """Test basic conversion of ArFrame to pandas DataFrame."""
        # Create a simple DataFrame and wrap it as ArFrame
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame)
        
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)

    def test_to_pandas_preserves_dtypes(self):
        """Test that to_pandas preserves column data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c']
        })
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame)
        
        assert result['int_col'].dtype == df['int_col'].dtype
        assert result['float_col'].dtype == df['float_col'].dtype
        assert result['str_col'].dtype == df['str_col'].dtype

    def test_to_pandas_preserves_index(self):
        """Test that to_pandas preserves the DataFrame index."""
        df = pd.DataFrame(
            {'a': [1, 2, 3], 'b': [4, 5, 6]},
            index=['x', 'y', 'z']
        )
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame)
        
        pd.testing.assert_index_equal(result.index, df.index)

    def test_to_pandas_empty_dataframe(self):
        """Test conversion of empty DataFrame."""
        df = pd.DataFrame()
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame)
        
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_to_pandas_single_column(self):
        """Test conversion with single column."""
        df = pd.DataFrame({'col': [1, 2, 3]})
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame)
        
        pd.testing.assert_frame_equal(result, df)

    def test_to_pandas_single_row(self):
        """Test conversion with single row."""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame)
        
        pd.testing.assert_frame_equal(result, df)

    def test_to_pandas_with_nan_values(self):
        """Test conversion preserves NaN values."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame)
        
        pd.testing.assert_frame_equal(result, df)

    def test_to_pandas_copy_false_default(self):
        """Test that copy=False is the default (no copy made)."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame, copy=False)
        
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)

    def test_to_pandas_copy_true(self):
        """Test that copy=True creates an independent copy."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame, copy=True)
        
        pd.testing.assert_frame_equal(result, df)
        # Verify it's a copy by modifying and checking original is unaffected
        result.iloc[0, 0] = 999
        assert ar.to_pandas(ar_frame).iloc[0, 0] != 999

    def test_to_pandas_with_multiindex(self):
        """Test conversion preserves MultiIndex."""
        arrays = [['a', 'a', 'b', 'b'], [1, 2, 1, 2]]
        index = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame({'col': [10, 20, 30, 40]}, index=index)
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame)
        
        pd.testing.assert_index_equal(result.index, df.index)

    def test_to_pandas_with_datetime_index(self):
        """Test conversion with DatetimeIndex."""
        dates = pd.date_range('2020-01-01', periods=3)
        df = pd.DataFrame({'a': [1, 2, 3]}, index=dates)
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame)
        
        pd.testing.assert_index_equal(result.index, df.index)

    def test_to_pandas_with_categorical_dtype(self):
        """Test conversion preserves categorical dtype."""
        df = pd.DataFrame({'cat_col': pd.Categorical(['a', 'b', 'a', 'c'])})
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame)
        
        assert pd.api.types.is_categorical_dtype(result['cat_col'])
        pd.testing.assert_frame_equal(result, df)

    def test_to_pandas_with_bool_dtype(self):
        """Test conversion preserves boolean dtype."""
        df = pd.DataFrame({'bool_col': [True, False, True]})
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame)
        
        assert result['bool_col'].dtype == bool
        pd.testing.assert_frame_equal(result, df)

    def test_to_pandas_large_dataframe(self):
        """Test conversion of larger DataFrame."""
        df = pd.DataFrame({
            'col_' + str(i): np.random.randn(1000)
            for i in range(10)
        })
        ar_frame = ar.from_pandas(df)
        
        result = ar.to_pandas(ar_frame)
        
        assert result.shape == df.shape
        pd.testing.assert_frame_equal(result, df)
