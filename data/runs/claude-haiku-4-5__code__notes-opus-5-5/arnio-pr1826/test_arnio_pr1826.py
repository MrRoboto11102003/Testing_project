import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.quality import profile


def test_profile_sample_size_must_be_int():
    """sample_size must be int, not float or bool."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(TypeError, match="sample_size must be an integer"):
        profile(frame, sample_size=1.5)
    with pytest.raises(TypeError, match="sample_size must be an integer"):
        profile(frame, sample_size=True)


def test_profile_sample_size_non_negative():
    """sample_size must be >= 0."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(ValueError, match="sample_size must be non-negative"):
        profile(frame, sample_size=-1)


def test_profile_approx_top_values_bool_validation():
    """approx_top_values must be bool."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(TypeError, match="approx_top_values must be a bool"):
        profile(frame, approx_top_values=1)


def test_profile_approx_top_values_min_unique_int_validation():
    """approx_top_values_min_unique must be int, not bool or str."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(TypeError, match="approx_top_values_min_unique must be an integer"):
        profile(frame, approx_top_values_min_unique="5")
    with pytest.raises(TypeError, match="approx_top_values_min_unique must be an integer"):
        profile(frame, approx_top_values_min_unique=True)
    with pytest.raises(ValueError, match="approx_top_values_min_unique must be non-negative"):
        profile(frame, approx_top_values_min_unique=-1)


def test_profile_approx_top_values_min_ratio_validation():
    """approx_top_values_min_ratio must be numeric, finite, and in [0, 1]."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(TypeError, match="approx_top_values_min_ratio must be a float"):
        profile(frame, approx_top_values_min_ratio="0.2")
    with pytest.raises(TypeError, match="approx_top_values_min_ratio must be a float"):
        profile(frame, approx_top_values_min_ratio=True)
    with pytest.raises(ValueError, match="finite number"):
        profile(frame, approx_top_values_min_ratio=float("nan"))
    with pytest.raises(ValueError, match="finite number"):
        profile(frame, approx_top_values_min_ratio=float("inf"))
    with pytest.raises(ValueError, match="between 0 and 1"):
        profile(frame, approx_top_values_min_ratio=1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        profile(frame, approx_top_values_min_ratio=-0.1)


def test_profile_approx_top_values_sample_size_validation():
    """approx_top_values_sample_size must be int > 0."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(TypeError, match="approx_top_values_sample_size must be an integer"):
        profile(frame, approx_top_values_sample_size=2.0)
    with pytest.raises(TypeError, match="approx_top_values_sample_size must be an integer"):
        profile(frame, approx_top_values_sample_size=True)
    with pytest.raises(ValueError, match="approx_top_values_sample_size must be positive"):
        profile(frame, approx_top_values_sample_size=0)


def test_profile_basic_numeric_frame():
    """Test basic profiling of a numeric frame."""
    df = pd.DataFrame({"age": [20, 30, 40, 50, 60]})
    frame = ar.from_pandas(df)
    report = profile(frame)
    
    assert report.row_count == 5
    assert report.column_count == 1
    assert report.duplicate_rows == 0
    assert report.duplicate_ratio == 0.0
    assert report.quality_score == 100.0
    assert "age" in report.columns
    assert report.columns["age"].min == 20
    assert report.columns["age"].max == 60
    assert report.columns["age"].mean == 40.0


def test_profile_duplicate_rows():
    """Test duplicate row detection and ratio calculation."""
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "b": [1, 1, 2]
    })
    frame = ar.from_pandas(df)
    report = profile(frame)
    
    assert report.row_count == 3
    assert report.duplicate_rows == 1
    assert pytest.approx(report.duplicate_ratio, 0.001) == 1/3


def test_profile_exclude_columns():
    """Test that columns are correctly excluded from profiling."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9]
    })
    frame = ar.from_pandas(df)
    report = profile(frame, exclude_columns=["b"])
    
    assert report.column_count == 2
    assert "a" in report.columns
    assert "b" not in report.columns
    assert "c" in report.columns


def test_profile_sample_values():
    """Test that sample_values respects sample_size limit."""
    df = pd.DataFrame({"vals": [1, 2, 3, 4, 5]})
    frame = ar.from_pandas(df)
    report = profile(frame, sample_size=0)
    
    assert report.columns["vals"].sample_values == []
    
    report = profile(frame, sample_size=2)
    assert len(report.columns["vals"].sample_values) <= 2


def test_profile_null_handling():
    """Test null value counting and ratio."""
    df = pd.DataFrame({"vals": [1, 2, None, 4, 5]})
    frame = ar.from_pandas(df)
    report = profile(frame)
    
    col = report.columns["vals"]
    assert col.null_count == 1
    assert pytest.approx(col.null_ratio, 0.001) == 0.2
    assert col.unique_count == 4


def test_profile_all_null_column():
    """Test a column with all null values."""
    df = pd.DataFrame({"vals": [None, None, None]})
    frame = ar.from_pandas(df)
    report = profile(frame)
    
    col = report.columns["vals"]
    assert col.null_count == 3
    assert col.null_ratio == 1.0
    assert col.semantic_type == "empty"


def test_profile_string_column_whitespace():
    """Test whitespace detection in string columns."""
    df = pd.DataFrame({"name": [" alice", "bob", "  charlie"]})
    frame = ar.from_pandas(df)
    report = profile(frame)
    
    col = report.columns["name"]
    assert col.whitespace_count == 2


def test_profile_string_column_length_stats():
    """Test length statistics for string columns."""
    df = pd.DataFrame({"text": ["a", "bb", "ccc"]})
    frame = ar.from_pandas(df)
    report = profile(frame)
    
    col = report.columns["text"]
    assert col.min == 1
    assert col.max == 3
    assert col.mean == 2.0


def test_profile_numeric_column_quantiles():
    """Test quantile computation for numeric columns."""
    df = pd.DataFrame({"vals": list(range(1, 101))})
    frame = ar.from_pandas(df)
    report = profile(frame)
    
    col = report.columns["vals"]
    assert col.q25 is not None
    assert col.q50 is not None
    assert col.q75 is not None
    assert col.q95 is not None
    assert col.q25 < col.q50 < col.q75 < col.q95
    assert col.histogram is not None
    assert len(col.histogram) == 10


def test_profile_zero_row_frame():
    """Test profiling a frame with zero rows."""
    df = pd.DataFrame({"a": [], "b": []})
    frame = ar.from_pandas(df)
    report = profile(frame)
    
    assert report.row_count == 0
    assert report.column_count == 2
    assert report.duplicate_rows == 0
    assert report.duplicate_ratio == 0.0
    assert report.quality_score == 100.0
    assert report.score_components == {}
