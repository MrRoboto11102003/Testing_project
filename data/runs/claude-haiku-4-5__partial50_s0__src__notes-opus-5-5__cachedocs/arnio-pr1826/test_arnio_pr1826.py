import pytest
import pandas as pd
import numpy as np
import arnio as ar


def test_profile_simple_frame():
    """Test profiling a simple frame."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["alice", "bob", "carol"],
    })
    frame = ar.from_pandas(df)
    report = ar.profile(frame)
    
    assert report.row_count == 3
    assert report.column_count == 2
    assert "id" in report.columns
    assert "name" in report.columns
    assert isinstance(report.quality_score, float)


def test_profile_sample_size_must_be_integer():
    """sample_size must be an integer, not a float or bool."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(TypeError, match="sample_size must be an integer"):
        ar.profile(frame, sample_size=1.5)
    with pytest.raises(TypeError, match="sample_size must be an integer"):
        ar.profile(frame, sample_size=True)


def test_profile_sample_size_non_negative():
    """sample_size must be >= 0."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(ValueError, match="sample_size must be non-negative"):
        ar.profile(frame, sample_size=-1)


def test_profile_approx_top_values_bool():
    """approx_top_values must be a boolean."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(TypeError, match="approx_top_values must be a bool"):
        ar.profile(frame, approx_top_values=1)


def test_profile_approx_top_values_min_unique_integer():
    """approx_top_values_min_unique must be an integer, not bool or string."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(TypeError, match="approx_top_values_min_unique must be an integer"):
        ar.profile(frame, approx_top_values_min_unique="5")
    with pytest.raises(TypeError, match="approx_top_values_min_unique must be an integer"):
        ar.profile(frame, approx_top_values_min_unique=True)


def test_profile_approx_top_values_min_unique_non_negative():
    """approx_top_values_min_unique must be >= 0."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(ValueError, match="approx_top_values_min_unique must be non-negative"):
        ar.profile(frame, approx_top_values_min_unique=-1)


def test_profile_approx_top_values_min_ratio_numeric():
    """approx_top_values_min_ratio must be numeric, not string or bool."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(TypeError, match="approx_top_values_min_ratio must be a float"):
        ar.profile(frame, approx_top_values_min_ratio="0.5")
    with pytest.raises(TypeError, match="approx_top_values_min_ratio must be a float"):
        ar.profile(frame, approx_top_values_min_ratio=True)


def test_profile_approx_top_values_min_ratio_finite():
    """approx_top_values_min_ratio must be finite (not inf or nan)."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(ValueError, match="finite"):
        ar.profile(frame, approx_top_values_min_ratio=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        ar.profile(frame, approx_top_values_min_ratio=float("nan"))


def test_profile_approx_top_values_min_ratio_range():
    """approx_top_values_min_ratio must be between 0 and 1 inclusive."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(ValueError, match="between 0 and 1"):
        ar.profile(frame, approx_top_values_min_ratio=1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ar.profile(frame, approx_top_values_min_ratio=-0.1)


def test_profile_approx_top_values_sample_size_integer():
    """approx_top_values_sample_size must be an integer."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(TypeError, match="approx_top_values_sample_size must be an integer"):
        ar.profile(frame, approx_top_values_sample_size=2.0)


def test_profile_approx_top_values_sample_size_positive():
    """approx_top_values_sample_size must be > 0."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(ValueError, match="approx_top_values_sample_size must be positive"):
        ar.profile(frame, approx_top_values_sample_size=0)


def test_profile_zero_rows():
    """Profile a frame with zero rows returns quality_score=100 and empty components."""
    df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
    frame = ar.from_pandas(df)
    report = ar.profile(frame)
    
    assert report.row_count == 0
    assert report.duplicate_rows == 0
    assert report.duplicate_ratio == 0.0
    assert report.quality_score == 100.0
    assert report.score_components == {}


def test_profile_exclude_columns():
    """Excluding columns removes them from the report."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["alice", "bob", "carol"],
        "debug": ["x", "y", "z"],
    })
    frame = ar.from_pandas(df)
    report = ar.profile(frame, exclude_columns=["debug"])
    
    assert report.column_count == 2
    assert "id" in report.columns
    assert "name" in report.columns
    assert "debug" not in report.columns


def test_profile_duplicate_rows():
    """Duplicate rows are counted correctly."""
    df = pd.DataFrame({
        "id": [1, 1, 2],
        "name": ["alice", "alice", "bob"],
    })
    frame = ar.from_pandas(df)
    report = ar.profile(frame)
    
    assert report.row_count == 3
    assert report.duplicate_rows == 1
    assert report.duplicate_ratio == pytest.approx(1 / 3, abs=0.01)
    assert report.quality_score < 100.0


def test_profile_string_column_properties():
    """String columns report unique count and top values."""
    df = pd.DataFrame({
        "name": ["bob", "bob", "alice", "alice", "carol"],
    })
    frame = ar.from_pandas(df)
    report = ar.profile(frame)
    
    col_profile = report.columns["name"]
    assert col_profile.dtype == "string"
    assert col_profile.unique_count == 3
    assert col_profile.top_values is not None
    assert len(col_profile.top_values) > 0
    # Top value should be one of the tied values (bob or alice appear twice)
    assert col_profile.top_values[0][1] == 2


def test_profile_numeric_column_statistics():
    """Numeric columns report mean, min, max, and quantiles."""
    df = pd.DataFrame({
        "age": [10, 20, 30, 40, 50],
    })
    frame = ar.from_pandas(df)
    report = ar.profile(frame)
    
    col_profile = report.columns["age"]
    assert col_profile.min == 10
    assert col_profile.max == 50
    assert col_profile.mean == pytest.approx(30.0)
    assert col_profile.std > 0
    assert col_profile.q50 == pytest.approx(30.0)
