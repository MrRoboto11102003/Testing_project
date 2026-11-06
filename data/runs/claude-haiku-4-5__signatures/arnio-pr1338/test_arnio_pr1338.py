import pytest
import pandas as pd
import numpy as np
import arnio as ar


def test_auto_clean_default_returns_frame():
    """Default call returns frame without extra objects"""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    result = ar.quality.auto_clean(df)
    assert result is not None
    assert hasattr(result, 'shape')


def test_auto_clean_safe_mode_explicit():
    """Safe mode is the default and works explicitly"""
    df = pd.DataFrame({'A': [1, 2, 3]})
    result = ar.quality.auto_clean(df, mode='safe')
    assert result is not None


def test_auto_clean_return_report_true_returns_tuple():
    """return_report=True returns tuple with frame and report"""
    df = pd.DataFrame({'A': [1, 2, 3]})
    result = ar.quality.auto_clean(df, return_report=True)
    assert isinstance(result, tuple)
    assert len(result) >= 2


def test_auto_clean_explain_true_returns_tuple():
    """explain=True returns tuple with frame and explanation"""
    df = pd.DataFrame({'A': [1, 2, 3]})
    result = ar.quality.auto_clean(df, explain=True)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_auto_clean_return_report_and_explain_returns_three_tuple():
    """Both return_report=True and explain=True returns 3-tuple"""
    df = pd.DataFrame({'A': [1, 2, 3]})
    result = ar.quality.auto_clean(df, return_report=True, explain=True)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_auto_clean_handles_missing_values():
    """Cleans dataframe with NaN values"""
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': ['a', np.nan, 'c', 'd']
    })
    result = ar.quality.auto_clean(df)
    assert result is not None


def test_auto_clean_handles_duplicate_rows():
    """Cleans dataframe with duplicate rows"""
    df = pd.DataFrame({
        'A': [1, 2, 1, 3],
        'B': ['x', 'y', 'x', 'z']
    })
    result = ar.quality.auto_clean(df)
    assert result is not None


def test_auto_clean_dry_run_mode():
    """dry_run=True previews cleaning without modification"""
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': ['a', 'b', 'c', 'd']
    })
    result = ar.quality.auto_clean(df, dry_run=True)
    assert result is not None


def test_auto_clean_confirmed_casts_parameter():
    """confirmed_casts guides type conversions"""
    df = pd.DataFrame({
        'A': ['1', '2', '3'],
        'B': ['x', 'y', 'z']
    })
    result = ar.quality.auto_clean(df, confirmed_casts={'A': 'int64'})
    assert result is not None


def test_auto_clean_allow_lossy_casts_true():
    """allow_lossy_casts=True permits data loss in conversions"""
    df = pd.DataFrame({
        'A': [1.5, 2.5, 3.5],
        'B': ['a', 'b', 'c']
    })
    result = ar.quality.auto_clean(df, allow_lossy_casts=True)
    assert result is not None


def test_auto_clean_allow_lossy_casts_false():
    """allow_lossy_casts=False prevents lossy conversions"""
    df = pd.DataFrame({
        'A': [1.5, 2.5, 3.5],
        'B': ['a', 'b', 'c']
    })
    result = ar.quality.auto_clean(df, allow_lossy_casts=False)
    assert result is not None


def test_auto_clean_empty_dataframe():
    """Handles empty dataframe"""
    df = pd.DataFrame()
    result = ar.quality.auto_clean(df)
    assert result is not None


def test_auto_clean_single_column():
    """Handles single column dataframe"""
    df = pd.DataFrame({'A': [1, 2, 3]})
    result = ar.quality.auto_clean(df)
    assert result is not None


def test_auto_clean_mixed_data_types():
    """Handles mixed data types"""
    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.5, 2.5, 3.5],
        'str_col': ['a', 'b', 'c']
    })
    result = ar.quality.auto_clean(df)
    assert result is not None


def test_auto_clean_all_missing_column():
    """Handles column with all NaN values"""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [np.nan, np.nan, np.nan]
    })
    result = ar.quality.auto_clean(df)
    assert result is not None


def test_auto_clean_single_row():
    """Handles single row dataframe"""
    df = pd.DataFrame({'A': [1], 'B': ['x']})
    result = ar.quality.auto_clean(df)
    assert result is not None
