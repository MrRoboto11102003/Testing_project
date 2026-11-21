import pytest
from arnio.quality import CleanStepRecord


def test_valid_basic_creation():
    """Test basic valid CleanStepRecord creation."""
    record = CleanStepRecord(
        step="strip_whitespace",
        kwargs={},
        rows_before=100,
        rows_after=95,
        rows_removed=5,
        reason="Removed rows with all NaN values"
    )
    assert record.step == "strip_whitespace"
    assert record.rows_before == 100
    assert record.rows_after == 95
    assert record.rows_removed == 5
    assert record.reason == "Removed rows with all NaN values"


def test_valid_with_zero_rows_removed():
    """Test valid creation with zero rows_removed (non-row-dropping step)."""
    record = CleanStepRecord(
        step="normalize_types",
        kwargs={"strict": True},
        rows_before=50,
        rows_after=50,
        rows_removed=0,
        reason="No rows were removed"
    )
    assert record.rows_removed == 0
    assert record.rows_before == record.rows_after


def test_valid_with_complex_kwargs():
    """Test valid creation with complex kwargs dict."""
    kwargs = {"threshold": 0.5, "method": "median", "columns": ["A", "B"]}
    record = CleanStepRecord(
        step="fill_missing",
        kwargs=kwargs,
        rows_before=1000,
        rows_after=1000,
        rows_removed=0,
        reason="Filled missing values"
    )
    assert record.kwargs == kwargs


def test_valid_with_large_numbers():
    """Test valid creation with large row counts."""
    record = CleanStepRecord(
        step="dedup",
        kwargs={},
        rows_before=1_000_000,
        rows_after=999_000,
        rows_removed=1000,
        reason="Removed duplicates"
    )
    assert record.rows_before == 1_000_000
    assert record.rows_after == 999_000
    assert record.rows_removed == 1000


def test_invalid_step_not_string():
    """Test TypeError when step is not a string."""
    with pytest.raises(TypeError, match="CleanStepRecord.step must be a str"):
        CleanStepRecord(
            step=123,
            kwargs={},
            rows_before=100,
            rows_after=100,
            rows_removed=0,
            reason="Test"
        )


def test_invalid_step_none():
    """Test TypeError when step is None."""
    with pytest.raises(TypeError, match="CleanStepRecord.step must be a str"):
        CleanStepRecord(
            step=None,
            kwargs={},
            rows_before=100,
            rows_after=100,
            rows_removed=0,
            reason="Test"
        )


def test_invalid_reason_not_string():
    """Test TypeError when reason is not a string."""
    with pytest.raises(TypeError, match="CleanStepRecord.reason must be a str"):
        CleanStepRecord(
            step="test_step",
            kwargs={},
            rows_before=100,
            rows_after=100,
            rows_removed=0,
            reason=123
        )


def test_invalid_kwargs_not_dict():
    """Test TypeError when kwargs is not a dict."""
    with pytest.raises(TypeError, match="CleanStepRecord.kwargs must be a dict"):
        CleanStepRecord(
            step="test_step",
            kwargs=["a", "b"],
            rows_before=100,
            rows_after=100,
            rows_removed=0,
            reason="Test"
        )


def test_invalid_rows_before_not_int():
    """Test TypeError when rows_before is not an int."""
    with pytest.raises(TypeError, match="CleanStepRecord.rows_before must be an int"):
        CleanStepRecord(
            step="test_step",
            kwargs={},
            rows_before="100",
            rows_after=100,
            rows_removed=0,
            reason="Test"
        )


def test_invalid_rows_after_not_int():
    """Test TypeError when rows_after is not an int."""
    with pytest.raises(TypeError, match="CleanStepRecord.rows_after must be an int"):
        CleanStepRecord(
            step="test_step",
            kwargs={},
            rows_before=100,
            rows_after=50.5,
            rows_removed=0,
            reason="Test"
        )


def test_invalid_rows_removed_bool():
    """Test TypeError when rows_removed is a bool (not accepted as int)."""
    with pytest.raises(TypeError, match="CleanStepRecord.rows_removed must be an int"):
        CleanStepRecord(
            step="test_step",
            kwargs={},
            rows_before=100,
            rows_after=100,
            rows_removed=True,
            reason="Test"
        )


def test_negative_rows_before():
    """Test ValueError when rows_before is negative."""
    with pytest.raises(ValueError, match="CleanStepRecord.rows_before cannot be negative"):
        CleanStepRecord(
            step="test_step",
            kwargs={},
            rows_before=-1,
            rows_after=100,
            rows_removed=0,
            reason="Test"
        )


def test_negative_rows_after():
    """Test ValueError when rows_after is negative."""
    with pytest.raises(ValueError, match="CleanStepRecord.rows_after cannot be negative"):
        CleanStepRecord(
            step="test_step",
            kwargs={},
            rows_before=100,
            rows_after=-5,
            rows_removed=0,
            reason="Test"
        )


def test_negative_rows_removed():
    """Test ValueError when rows_removed is negative."""
    with pytest.raises(ValueError, match="CleanStepRecord.rows_removed cannot be negative"):
        CleanStepRecord(
            step="test_step",
            kwargs={},
            rows_before=100,
            rows_after=95,
            rows_removed=-1,
            reason="Test"
        )


def test_frozen_immutable():
    """Test that CleanStepRecord is immutable (frozen dataclass)."""
    record = CleanStepRecord(
        step="test_step",
        kwargs={},
        rows_before=100,
        rows_after=100,
        rows_removed=0,
        reason="Test"
    )
    with pytest.raises(AttributeError):
        record.step = "new_step"
