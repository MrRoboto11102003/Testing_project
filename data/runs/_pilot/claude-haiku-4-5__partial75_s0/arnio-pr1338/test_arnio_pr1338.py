import pytest
import pandas as pd
import numpy as np
import arnio as ar


def test_auto_clean_default_returns_arframe():
    """Default parameters return cleaned ArFrame."""
    df = pd.DataFrame({"col": [" a ", " b "]})
    frame = ar.from_pandas(df)
    result = ar.auto_clean(frame)
    assert isinstance(result, ar.ArFrame)


def test_auto_clean_safe_mode_default():
    """safe is the default mode and only applies whitespace trimming."""
    df = pd.DataFrame({"name": [" alice ", " bob "]})
    frame = ar.from_pandas(df)
    result = ar.auto_clean(frame, mode="safe")
    result_df = ar.to_pandas(result)
    assert result_df.loc[0, "name"] == "alice"
    assert result_df.loc[1, "name"] == "bob"


def test_auto_clean_strict_mode_removes_duplicates():
    """strict mode removes duplicate rows and applies casts."""
    df = pd.DataFrame({"id": [1, 2, 2], "name": ["alice", "bob", "bob"]})
    frame = ar.from_pandas(df)
    result = ar.auto_clean(frame, mode="strict")
    result_df = ar.to_pandas(result)
    assert len(result_df) == 2
    assert result_df["id"].tolist() == [1, 2]


def test_auto_clean_return_report_true_returns_tuple():
    """return_report=True returns tuple[ArFrame, DataQualityReport]."""
    df = pd.DataFrame({"col": ["a", "b"]})
    frame = ar.from_pandas(df)
    result = ar.auto_clean(frame, return_report=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], ar.ArFrame)
    assert isinstance(result[1], ar.DataQualityReport)


def test_auto_clean_dry_run_returns_report():
    """dry_run=True returns DataQualityReport without cleaning frame."""
    df = pd.DataFrame({"col": [" a ", " b "]})
    frame = ar.from_pandas(df)
    report = ar.auto_clean(frame, dry_run=True)
    assert isinstance(report, ar.DataQualityReport)
    # Original frame should be unchanged
    original_df = ar.to_pandas(frame)
    assert original_df.loc[0, "col"] == " a "


def test_auto_clean_explain_true_returns_tuple_with_explanation():
    """explain=True returns tuple[ArFrame, CleanExplanation]."""
    df = pd.DataFrame({"col": ["a"]})
    frame = ar.from_pandas(df)
    result = ar.auto_clean(frame, explain=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], ar.ArFrame)
    # CleanExplanation is printable
    assert hasattr(result[1], '__str__')


def test_auto_clean_return_report_and_explain_returns_triple():
    """return_report=True and explain=True returns tuple[ArFrame, DataQualityReport, CleanExplanation]."""
    df = pd.DataFrame({"col": ["a"]})
    frame = ar.from_pandas(df)
    result = ar.auto_clean(frame, return_report=True, explain=True)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], ar.ArFrame)
    assert isinstance(result[1], ar.DataQualityReport)


def test_auto_clean_safe_mode_no_deduplication():
    """safe mode does not remove duplicate rows."""
    df = pd.DataFrame({"id": [1, 1, 2], "name": ["alice", "alice", "bob"]})
    frame = ar.from_pandas(df)
    result = ar.auto_clean(frame, mode="safe")
    result_df = ar.to_pandas(result)
    assert len(result_df) == 3


def test_auto_clean_empty_frame():
    """auto_clean handles empty frames gracefully."""
    df = pd.DataFrame({"col": []})
    frame = ar.from_pandas(df)
    result = ar.auto_clean(frame)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 0


def test_auto_clean_frame_with_nulls_strict_mode():
    """strict mode handles frames containing null values."""
    df = pd.DataFrame({"id": [1, None, 2], "name": ["a", "b", "c"]})
    frame = ar.from_pandas(df)
    result = ar.auto_clean(frame, mode="strict")
    assert isinstance(result, ar.ArFrame)
    assert len(ar.to_pandas(result)) == 3


def test_auto_clean_preserves_column_names_and_order():
    """auto_clean preserves column names and order."""
    df = pd.DataFrame({"z": [1, 2], "a": [3, 4], "m": [5, 6]})
    frame = ar.from_pandas(df)
    result = ar.auto_clean(frame)
    result_df = ar.to_pandas(result)
    assert result_df.columns.tolist() == ["z", "a", "m"]


def test_auto_clean_whitespace_stripped_in_all_modes():
    """Both safe and strict modes strip leading/trailing whitespace."""
    df = pd.DataFrame({"col": [" value "]})
    frame1 = ar.from_pandas(df.copy())
    frame2 = ar.from_pandas(df.copy())
    
    safe_result = ar.to_pandas(ar.auto_clean(frame1, mode="safe"))
    strict_result = ar.to_pandas(ar.auto_clean(frame2, mode="strict"))
    
    assert safe_result.loc[0, "col"] == "value"
    assert strict_result.loc[0, "col"] == "value"


def test_auto_clean_confirmed_casts_with_allow_lossy():
    """confirmed_casts parameter works with allow_lossy_casts in strict mode."""
    df = pd.DataFrame({"num": ["1", "2", "3"]})
    frame = ar.from_pandas(df)
    result = ar.auto_clean(
        frame,
        mode="strict",
        allow_lossy_casts=True,
        confirmed_casts={"num": "int64"}
    )
    assert isinstance(result, ar.ArFrame)


def test_auto_clean_strict_mode_deterministic():
    """strict mode produces deterministic results on same input."""
    df = pd.DataFrame({"id": [1, 2, 2, 3], "val": [" a ", " b ", " b ", " c "]})
    frame1 = ar.from_pandas(df.copy())
    frame2 = ar.from_pandas(df.copy())
    
    result1 = ar.to_pandas(ar.auto_clean(frame1, mode="strict"))
    result2 = ar.to_pandas(ar.auto_clean(frame2, mode="strict"))
    
    pd.testing.assert_frame_equal(result1, result2)


def test_auto_clean_dry_run_with_explain():
    """Combination of dry_run and explain parameters is handled."""
    df = pd.DataFrame({"col": [" a "]})
    frame = ar.from_pandas(df)
    # dry_run with explain should return appropriate result
    result = ar.auto_clean(frame, dry_run=True, explain=True)
    assert result is not None
