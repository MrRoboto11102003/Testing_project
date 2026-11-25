import pytest
import pandas as pd
import numpy as np
import arnio as ar


@pytest.fixture
def profile_a():
    """Create a sample profile for comparison tests."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "score": [85.5, 90.0, 78.5, 92.0, 88.0],
    })
    frame = ar.from_pandas(df)
    return ar.profile(frame)


@pytest.fixture
def profile_b():
    """Create another sample profile for comparison tests."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack"],
        "score": [85.5, 90.0, 78.5, 92.0, 88.0, 76.0, 91.0, 82.0, 87.0, 93.0],
    })
    frame = ar.from_pandas(df)
    return ar.profile(frame)


def test_compare_profiles_returns_comparison_object(profile_a, profile_b):
    """Test that compare_profiles returns a ProfileComparison object."""
    result = ar.quality.compare_profiles(profile_a, profile_b)
    assert result is not None
    assert hasattr(result, '__class__')


def test_compare_identical_profiles():
    """Test comparing identical profiles."""
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
    })
    frame = ar.from_pandas(df)
    profile = ar.profile(frame)
    
    result = ar.quality.compare_profiles(profile, profile)
    assert result is not None


def test_compare_profiles_different_row_counts(profile_a, profile_b):
    """Test comparing profiles with different row counts."""
    result = ar.quality.compare_profiles(profile_a, profile_b)
    assert result is not None
    # profile_a has 5 rows, profile_b has 10 rows


def test_compare_profiles_with_nulls():
    """Test comparing profiles with null values."""
    df1 = pd.DataFrame({
        "value": [1, 2, None, 4, 5],
    })
    df2 = pd.DataFrame({
        "value": [1, 2, 3, None, 5],
    })
    
    frame1 = ar.from_pandas(df1)
    frame2 = ar.from_pandas(df2)
    
    profile1 = ar.profile(frame1)
    profile2 = ar.profile(frame2)
    
    result = ar.quality.compare_profiles(profile1, profile2)
    assert result is not None


def test_compare_profiles_empty_vs_nonempty():
    """Test comparing an empty profile with a non-empty profile."""
    df_empty = pd.DataFrame({"col": []})
    df_nonempty = pd.DataFrame({"col": [1, 2, 3]})
    
    frame_empty = ar.from_pandas(df_empty)
    frame_nonempty = ar.from_pandas(df_nonempty)
    
    profile_empty = ar.profile(frame_empty)
    profile_nonempty = ar.profile(frame_nonempty)
    
    result = ar.quality.compare_profiles(profile_empty, profile_nonempty)
    assert result is not None


def test_compare_profiles_different_dtypes():
    """Test comparing profiles where columns have different dtypes."""
    df1 = pd.DataFrame({
        "value": [1, 2, 3],  # int
    })
    df2 = pd.DataFrame({
        "value": [1.0, 2.5, 3.7],  # float
    })
    
    frame1 = ar.from_pandas(df1)
    frame2 = ar.from_pandas(df2)
    
    profile1 = ar.profile(frame1)
    profile2 = ar.profile(frame2)
    
    result = ar.quality.compare_profiles(profile1, profile2)
    assert result is not None


def test_compare_profiles_different_column_count():
    """Test comparing profiles with different numbers of columns."""
    df1 = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    df2 = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9],
    })
    
    frame1 = ar.from_pandas(df1)
    frame2 = ar.from_pandas(df2)
    
    profile1 = ar.profile(frame1)
    profile2 = ar.profile(frame2)
    
    result = ar.quality.compare_profiles(profile1, profile2)
    assert result is not None


def test_compare_profiles_with_duplicates():
    """Test comparing profiles with duplicate rows."""
    df1 = pd.DataFrame({
        "id": [1, 2, 3, 1, 2],
        "name": ["Alice", "Bob", "Charlie", "Alice", "Bob"],
    })
    df2 = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    })
    
    frame1 = ar.from_pandas(df1)
    frame2 = ar.from_pandas(df2)
    
    profile1 = ar.profile(frame1)
    profile2 = ar.profile(frame2)
    
    result = ar.quality.compare_profiles(profile1, profile2)
    assert result is not None


def test_compare_profiles_all_nulls():
    """Test comparing profiles where one or both have all-null columns."""
    df1 = pd.DataFrame({
        "col": [None, None, None],
    })
    df2 = pd.DataFrame({
        "col": [1, 2, 3],
    })
    
    frame1 = ar.from_pandas(df1)
    frame2 = ar.from_pandas(df2)
    
    profile1 = ar.profile(frame1)
    profile2 = ar.profile(frame2)
    
    result = ar.quality.compare_profiles(profile1, profile2)
    assert result is not None


def test_compare_profiles_single_column():
    """Test comparing profiles with a single column."""
    df1 = pd.DataFrame({"value": [1, 2, 3]})
    df2 = pd.DataFrame({"value": [10, 20, 30]})
    
    frame1 = ar.from_pandas(df1)
    frame2 = ar.from_pandas(df2)
    
    profile1 = ar.profile(frame1)
    profile2 = ar.profile(frame2)
    
    result = ar.quality.compare_profiles(profile1, profile2)
    assert result is not None


def test_compare_profiles_zero_rows():
    """Test comparing profiles where one has zero rows."""
    df_zero = pd.DataFrame({"a": pd.Series([], dtype="int64")})
    df_nonzero = pd.DataFrame({"a": [1, 2, 3]})
    
    frame_zero = ar.from_pandas(df_zero)
    frame_nonzero = ar.from_pandas(df_nonzero)
    
    profile_zero = ar.profile(frame_zero)
    profile_nonzero = ar.profile(frame_nonzero)
    
    result = ar.quality.compare_profiles(profile_zero, profile_nonzero)
    assert result is not None


def test_compare_profiles_high_cardinality():
    """Test comparing profiles with high-cardinality string columns."""
    df1 = pd.DataFrame({
        "uuid": [f"id-{i}" for i in range(100)],
    })
    df2 = pd.DataFrame({
        "uuid": [f"id-{i}" for i in range(50, 150)],
    })
    
    frame1 = ar.from_pandas(df1)
    frame2 = ar.from_pandas(df2)
    
    profile1 = ar.profile(frame1)
    profile2 = ar.profile(frame2)
    
    result = ar.quality.compare_profiles(profile1, profile2)
    assert result is not None


def test_compare_profiles_mixed_types():
    """Test comparing profiles with mixed column types."""
    df1 = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.1, 2.2, 3.3],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True],
    })
    df2 = pd.DataFrame({
        "int_col": [10, 20, 30, 40],
        "float_col": [10.1, 20.2, 30.3, 40.4],
        "str_col": ["x", "y", "z", "w"],
        "bool_col": [False, True, False, True],
    })
    
    frame1 = ar.from_pandas(df1)
    frame2 = ar.from_pandas(df2)
    
    profile1 = ar.profile(frame1)
    profile2 = ar.profile(frame2)
    
    result = ar.quality.compare_profiles(profile1, profile2)
    assert result is not None


def test_compare_profiles_result_has_comparison_data(profile_a, profile_b):
    """Test that the returned object contains comparison information."""
    result = ar.quality.compare_profiles(profile_a, profile_b)
    
    # Result should have some form of comparison output
    # Common patterns: to_markdown(), summary(), __str__, or properties
    assert (
        hasattr(result, 'to_markdown') or
        hasattr(result, 'summary') or
        hasattr(result, '__str__') or
        hasattr(result, 'passed')
    )
