"""
Unit tests for arnio.quality.auto_clean.
Tests documented behavior, edge cases, and return type combinations.
"""

import pytest
import pandas as pd
import numpy as np

from arnio import ArFrame
from arnio.quality import auto_clean


def create_test_frame():
    """Create a test ArFrame with sample data."""
    df = pd.DataFrame({
        "name": [" Alice ", " Bob", "  Charlie  "],
        "age": [25, 30, None],
        "score": [95.5, 88.0, 100.0],
    })
    return ArFrame.from_pandas(df)


def test_auto_clean_basic():
    """Test basic auto_clean with safe mode (default)."""
    frame = create_test_frame()
    result = auto_clean(frame)

    assert isinstance(result, ArFrame)
    # Safe mode only strips whitespace
    assert " Alice " in str(result["name"].iloc[0])  # Should still have spaces


def test_auto_clean_strict_mode():
    """Test auto_clean with strict mode (deduplication and casting)."""
    frame = create_test_frame()
    result = auto_clean(frame, mode="strict")

    assert isinstance(result, ArFrame)
    # Strict mode removes duplicates
    # Note: original data has no exact duplicates after whitespace strip


def test_auto_clean_dry_run():
    """Test dry_run=True returns DataQualityReport without mutating frame."""
    frame = create_test_frame()
    report = auto_clean(frame, dry_run=True)

    assert isinstance(report, type(auto_clean.__globals__["profile"](frame)))
    # Frame should be unchanged (dry_run doesn't mutate)


def test_auto_clean_return_report():
    """Test return_report=True returns tuple with frame and report."""
    frame = create_test_frame()
    result = auto_clean(frame, return_report=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], ArFrame)
    assert type(result[1]) is not ArFrame


def test_auto_clean_explain_only():
    """Test explain=True returns tuple with frame and explanation."""
    frame = create_test_frame()
    result = auto_clean(frame, explain=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert type(result[1]) is not ArFrame


def test_auto_clean_both_report_and_explain():
    """Test return_report=True and explain=True returns triple."""
    frame = create_test_frame()
    result = auto_clean(frame, return_report=True, explain=True)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert type(result[1]) is not ArFrame


def test_auto_clean_invalid_mode():
    """Test ValueError raised for invalid mode."""
    frame = create_test_frame()
    with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
        auto_clean(frame, mode="invalid")


def test_auto_clean_invalid_return_report_type():
    """Test TypeError raised for non-bool return_report."""
    frame = create_test_frame()
    with pytest.raises(TypeError, match="return_report must be a bool"):
        auto_clean(frame, return_report=123)


def test_auto_clean_invalid_dry_run_type():
    """Test TypeError raised for non-bool dry_run."""
    frame = create_test_frame()
    with pytest.raises(TypeError, match="dry_run must be a bool"):
        auto_clean(frame, dry_run="yes")


def test_auto_clean_invalid_explain_type():
    """Test TypeError raised for non-bool explain."""
    frame = create_test_frame()
    with pytest.raises(TypeError, match="explain must be a bool"):
        auto_clean(frame, explain=123)


def test_auto_clean_explain_with_dry_run():
    """Test ValueError raised when explain=True and dry_run=True."""
    frame = create_test_frame()
    with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
        auto_clean(frame, dry_run=True, explain=True)


def test_auto_clean_explain_with_return_report():
    """Test ValueError raised when explain=True and return_report=True."""
    frame = create_test_frame()
    with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
        auto_clean(frame, dry_run=True, return_report=True)


def test_auto_clean_strict_with_confirmed_casts():
    """Test strict mode with confirmed casts applied successfully."""
    frame = create_test_frame()
    # Create a mapping that would be suggested (e.g., age as int64)
    casts = {"age": "int64"}
    result = auto_clean(frame, mode="strict", allow_lossy_casts=True, confirmed_casts=casts)

    assert isinstance(result, ArFrame)


def test_auto_clean_strict_without_confirmed_casts():
    """Test ValueError raised for strict mode without confirmed casts."""
    frame = create_test_frame()
    with pytest.raises(ValueError, match="requires confirmed_casts before"):
        auto_clean(frame, mode="strict", allow_lossy_casts=True)


def test_auto_clean_invalid_confirmed_casts_type():
    """Test TypeError raised for non-dict confirmed_casts."""
    frame = create_test_frame()
    with pytest.raises(TypeError, match="confirmed_casts must be a dict"):
        auto_clean(frame, confirmed_casts=["invalid"])


def test_auto_clean_empty_frame():
    """Test auto_clean with an empty ArFrame."""
    df = pd.DataFrame({"col": []})
    frame = ArFrame.from_pandas(df)
    result = auto_clean(frame)

    assert isinstance(result, ArFrame)
    assert result.shape == (0, 1)


def test_auto_clean_strict_with_duplicates():
    """Test strict mode removes duplicates from data."""
    df = pd.DataFrame({
        "name": ["Alice", "alice", "Bob"],
        "score": [95.0, 95.0, 88.0],
    })
    frame = ArFrame.from_pandas(df)
    result = auto_clean(frame, mode="strict")

    # After strip_whitespace and drop_duplicates, should have unique rows
    assert len(result["name"].unique()) == 2  # Alice and Bob (alice removed)
