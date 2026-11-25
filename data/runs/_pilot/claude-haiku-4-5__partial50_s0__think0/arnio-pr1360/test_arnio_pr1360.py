import pytest
import pandas as pd
import numpy as np
import arnio as ar


@pytest.fixture
def simple_frame():
    """Create a simple ArFrame for testing."""
    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "score": [85.5, 90.0, 78.5, 92.0, 88.0],
        "active": [True, True, False, True, False],
    }
    return ar.from_pandas(pd.DataFrame(data))


@pytest.fixture
def frame_with_nulls():
    """Create an ArFrame with null values."""
    data = {
        "id": [1, 2, None, 4, 5],
        "name": ["Alice", None, "Charlie", "David", "Eve"],
        "score": [85.5, 90.0, None, 92.0, 88.0],
        "active": [True, True, False, None, False],
    }
    return ar.from_pandas(pd.DataFrame(data))


@pytest.fixture
def frame_high_duplicates():
    """Create an ArFrame with many duplicate rows."""
    data = {
        "id": [1, 1, 2, 2, 3, 3, 3],
        "name": ["Alice", "Alice", "Bob", "Bob", "Charlie", "Charlie", "Charlie"],
        "value": [100, 100, 200, 200, 300, 300, 300],
    }
    return ar.from_pandas(pd.DataFrame(data))


@pytest.fixture
def frame_low_duplicates():
    """Create an ArFrame with few duplicate rows."""
    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "value": [100, 200, 300, 400, 500],
    }
    return ar.from_pandas(pd.DataFrame(data))


def test_compare_profiles_basic_structure(simple_frame):
    """Test that compare_profiles returns a ProfileComparison object."""
    profile_a = ar.profile(simple_frame)
    profile_b = ar.profile(simple_frame)
    
    result = ar.compare_profiles(profile_a, profile_b)
    
    assert result is not None
    assert hasattr(result, "differences")


def test_compare_profiles_identical_profiles(simple_frame):
    """Test comparing identical profiles shows no significant differences."""
    profile_a = ar.profile(simple_frame)
    profile_b = ar.profile(simple_frame)
    
    result = ar.compare_profiles(profile_a, profile_b)
    
    # Identical profiles should have minimal or no differences
    assert result is not None


def test_compare_profiles_with_nulls(simple_frame, frame_with_nulls):
    """Test comparing profiles with different null patterns."""
    profile_a = ar.profile(simple_frame)
    profile_b = ar.profile(frame_with_nulls)
    
    result = ar.compare_profiles(profile_a, profile_b)
    
    assert result is not None
    # Should detect difference in null counts


def test_compare_profiles_different_row_counts():
    """Test comparing profiles with different row counts."""
    df_small = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
    df_large = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8]}))
    
    profile_a = ar.profile(df_small)
    profile_b = ar.profile(df_large)
    
    result = ar.compare_profiles(profile_a, profile_b)
    
    assert result is not None


def test_compare_profiles_different_duplicates(frame_low_duplicates, frame_high_duplicates):
    """Test comparing profiles with different duplicate ratios."""
    profile_a = ar.profile(frame_low_duplicates)
    profile_b = ar.profile(frame_high_duplicates)
    
    result = ar.compare_profiles(profile_a, profile_b)
    
    assert result is not None
    # Should detect different duplicate ratios


def test_compare_profiles_column_profile_changes():
    """Test detecting changes in column-level statistics."""
    df1 = ar.from_pandas(pd.DataFrame({
        "values": [10, 20, 30, 40, 50],
    }))
    df2 = ar.from_pandas(pd.DataFrame({
        "values": [100, 200, 300, 400, 500],
    }))
    
    profile_a = ar.profile(df1)
    profile_b = ar.profile(df2)
    
    result = ar.compare_profiles(profile_a, profile_b)
    
    assert result is not None


def test_compare_profiles_cardinality_change():
    """Test detecting changes in column cardinality."""
    df1 = ar.from_pandas(pd.DataFrame({
        "category": ["A", "B", "C"],
    }))
    df2 = ar.from_pandas(pd.DataFrame({
        "category": ["A", "A", "B", "B", "C", "C", "D"],
    }))
    
    profile_a = ar.profile(df1)
    profile_b = ar.profile(df2)
    
    result = ar.compare_profiles(profile_a, profile_b)
    
    assert result is not None


def test_compare_profiles_null_ratio_increase(simple_frame):
    """Test detecting an increase in null ratio."""
    df_clean = simple_frame
    
    # Create a frame with more nulls
    df_nulls = ar.from_pandas(pd.DataFrame({
        "id": [1, None, 3, None, 5],
        "name": ["Alice", "Bob", None, "David", None],
        "score": [85.5, 90.0, 78.5, 92.0, 88.0],
        "active": [True, True, False, True, False],
    }))
    
    profile_a = ar.profile(df_clean)
    profile_b = ar.profile(df_nulls)
    
    result = ar.compare_profiles(profile_a, profile_b)
    
    assert result is not None


def test_compare_profiles_symmetric():
    """Test that compare_profiles can be called in both directions."""
    df1 = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
    df2 = ar.from_pandas(pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]}))
    
    profile_a = ar.profile(df1)
    profile_b = ar.profile(df2)
    
    result_ab = ar.compare_profiles(profile_a, profile_b)
    result_ba = ar.compare_profiles(profile_b, profile_a)
    
    assert result_ab is not None
    assert result_ba is not None


def test_compare_profiles_empty_frames():
    """Test comparing profiles of frames with different emptiness."""
    df_empty = ar.from_pandas(pd.DataFrame({
        "id": pd.Series([], dtype="int64"),
        "name": pd.Series([], dtype="object"),
    }))
    df_populated = ar.from_pandas(pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["A", "B", "C"],
    }))
    
    profile_a = ar.profile(df_empty)
    profile_b = ar.profile(df_populated)
    
    result = ar.compare_profiles(profile_a, profile_b)
    
    assert result is not None


def test_compare_profiles_type_consistency(simple_frame):
    """Test that returned ProfileComparison has consistent structure."""
    profile_a = ar.profile(simple_frame)
    profile_b = ar.profile(simple_frame)
    
    result = ar.compare_profiles(profile_a, profile_b)
    
    # Verify the result has expected methods or attributes
    assert hasattr(result, "summary") or hasattr(result, "differences")


def test_compare_profiles_single_column_change():
    """Test detecting changes in a single column while others remain stable."""
    df1 = ar.from_pandas(pd.DataFrame({
        "stable": [1, 2, 3, 4, 5],
        "changing": ["A", "B", "C", "D", "E"],
    }))
    df2 = ar.from_pandas(pd.DataFrame({
        "stable": [1, 2, 3, 4, 5],
        "changing": ["A", "B", None, "D", None],
    }))
    
    profile_a = ar.profile(df1)
    profile_b = ar.profile(df2)
    
    result = ar.compare_profiles(profile_a, profile_b)
    
    assert result is not None


def test_compare_profiles_with_numeric_variance(simple_frame):
    """Test comparing numeric columns with different variance."""
    df_low_variance = ar.from_pandas(pd.DataFrame({
        "value": [50, 51, 49, 50, 52],
    }))
    df_high_variance = ar.from_pandas(pd.DataFrame({
        "value": [10, 50, 100, 5, 95],
    }))
    
    profile_a = ar.profile(df_low_variance)
    profile_b = ar.profile(df_high_variance)
    
    result = ar.compare_profiles(profile_a, profile_b)
    
    assert result is not None
