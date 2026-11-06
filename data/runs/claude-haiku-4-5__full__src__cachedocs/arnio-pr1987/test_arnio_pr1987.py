import pytest
from arnio.quality import CleanStepRecord


def test_valid_creation():
    """Create a valid CleanStepRecord."""
    record = CleanStepRecord(
        step="strip_whitespace",
        kwargs={"subset": ["name"]},
        rows_before=100,
        rows_after=100,
        rows_removed=0,
        reason="Remove whitespace"
    )
    assert record.step == "strip_whitespace"
    assert record.kwargs == {"subset": ["name"]}
    assert record.rows_before == 100
    assert record.rows_after == 100
    assert record.rows_removed == 0
    assert record.reason == "Remove whitespace"


def test_valid_creation_with_row_removal():
    """Create a valid record where rows are removed."""
    record = CleanStepRecord(
        step="drop_duplicates",
        kwargs={"keep": "first"},
        rows_before=100,
        rows_after=85,
        rows_removed=15,
        reason="Remove duplicates"
    )
    assert record.rows_removed == 15


def test_valid_creation_zero_rows():
    """Create a valid record with zero rows."""
    record = CleanStepRecord(
        step="drop_nulls",
        kwargs={},
        rows_before=0,
        rows_after=0,
        rows_removed=0,
        reason="Empty dataset"
    )
    assert record.rows_before == 0


def test_step_type_validation():
    """step must be a string."""
    with pytest.raises(TypeError, match="CleanStepRecord.step must be a str"):
        CleanStepRecord(
            step=123,  # type: ignore
            kwargs={},
            rows_before=10,
            rows_after=10,
            rows_removed=0,
            reason="Test"
        )


def test_reason_type_validation():
    """reason must be a string."""
    with pytest.raises(TypeError, match="CleanStepRecord.reason must be a str"):
        CleanStepRecord(
            step="test",
            kwargs={},
            rows_before=10,
            rows_after=10,
            rows_removed=0,
            reason=42  # type: ignore
        )


def test_kwargs_type_validation():
    """kwargs must be a dictionary."""
    with pytest.raises(TypeError, match="CleanStepRecord.kwargs must be a dict"):
        CleanStepRecord(
            step="test",
            kwargs="not a dict",  # type: ignore
            rows_before=10,
            rows_after=10,
            rows_removed=0,
            reason="Test"
        )


def test_rows_before_type_validation():
    """rows_before must be an integer."""
    with pytest.raises(TypeError, match="CleanStepRecord.rows_before must be an int"):
        CleanStepRecord(
            step="test",
            kwargs={},
            rows_before="10",  # type: ignore
            rows_after=10,
            rows_removed=0,
            reason="Test"
        )


def test_rows_after_type_validation():
    """rows_after must be an integer."""
    with pytest.raises(TypeError, match="CleanStepRecord.rows_after must be an int"):
        CleanStepRecord(
            step="test",
            kwargs={},
            rows_before=10,
            rows_after=10.5,  # type: ignore
            rows_removed=0,
            reason="Test"
        )


def test_rows_removed_type_validation():
    """rows_removed must be an integer."""
    with pytest.raises(TypeError, match="CleanStepRecord.rows_removed must be an int"):
        CleanStepRecord(
            step="test",
            kwargs={},
            rows_before=10,
            rows_after=10,
            rows_removed=None,  # type: ignore
            reason="Test"
        )


def test_bool_rejected_for_row_counts():
    """Boolean values must be rejected for row count fields (bool is subclass of int)."""
    # Test rows_before
    with pytest.raises(TypeError, match="CleanStepRecord.rows_before must be an int"):
        CleanStepRecord(
            step="test", kwargs={}, rows_before=True,  # type: ignore
            rows_after=10, rows_removed=0, reason="Test"
        )
    
    # Test rows_after
    with pytest.raises(TypeError, match="CleanStepRecord.rows_after must be an int"):
        CleanStepRecord(
            step="test", kwargs={}, rows_before=10, rows_after=False,  # type: ignore
            rows_removed=0, reason="Test"
        )
    
    # Test rows_removed
    with pytest.raises(TypeError, match="CleanStepRecord.rows_removed must be an int"):
        CleanStepRecord(
            step="test", kwargs={}, rows_before=10, rows_after=10,
            rows_removed=True,  # type: ignore
            reason="Test"
        )


def test_negative_row_counts_rejected():
    """Row counts cannot be negative."""
    # Test rows_before
    with pytest.raises(ValueError, match="CleanStepRecord.rows_before cannot be negative"):
        CleanStepRecord(
            step="test", kwargs={}, rows_before=-1,
            rows_after=10, rows_removed=0, reason="Test"
        )
    
    # Test rows_after
    with pytest.raises(ValueError, match="CleanStepRecord.rows_after cannot be negative"):
        CleanStepRecord(
            step="test", kwargs={}, rows_before=10, rows_after=-5,
            rows_removed=0, reason="Test"
        )
    
    # Test rows_removed
    with pytest.raises(ValueError, match="CleanStepRecord.rows_removed cannot be negative"):
        CleanStepRecord(
            step="test", kwargs={}, rows_before=10, rows_after=10,
            rows_removed=-1, reason="Test"
        )


def test_frozen_dataclass():
    """CleanStepRecord is frozen and cannot be modified."""
    record = CleanStepRecord(
        step="test",
        kwargs={},
        rows_before=10,
        rows_after=10,
        rows_removed=0,
        reason="Test"
    )
    
    with pytest.raises(AttributeError):
        record.step = "modified"  # type: ignore
    
    with pytest.raises(AttributeError):
        record.reason = "modified"  # type: ignore


def test_edge_case_empty_strings():
    """Empty strings are allowed for step and reason."""
    record = CleanStepRecord(
        step="",
        kwargs={},
        rows_before=0,
        rows_after=0,
        rows_removed=0,
        reason=""
    )
    assert record.step == ""
    assert record.reason == ""


def test_complex_kwargs():
    """kwargs can contain nested structures."""
    kwargs = {
        "subset": ["col1", "col2"],
        "params": {"nested": {"deep": "value"}},
        "values": [1, 2, 3]
    }
    record = CleanStepRecord(
        step="complex",
        kwargs=kwargs,
        rows_before=100,
        rows_after=100,
        rows_removed=0,
        reason="Complex operation"
    )
    assert record.kwargs == kwargs


def test_large_row_counts():
    """Large row counts are supported."""
    record = CleanStepRecord(
        step="test",
        kwargs={},
        rows_before=1_000_000_000,
        rows_after=999_999_999,
        rows_removed=1,
        reason="Large dataset"
    )
    assert record.rows_before == 1_000_000_000
    assert record.rows_after == 999_999_999
