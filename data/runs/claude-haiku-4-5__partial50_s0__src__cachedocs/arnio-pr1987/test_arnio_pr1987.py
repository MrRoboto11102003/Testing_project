import pytest
from arnio.quality import CleanStepRecord


class TestCleanStepRecord:
    """Test arnio.quality.CleanStepRecord dataclass."""

    def test_valid_creation(self):
        """Test creating a valid CleanStepRecord with all fields."""
        record = CleanStepRecord(
            step="strip_whitespace",
            kwargs={},
            rows_before=100,
            rows_after=100,
            rows_removed=0,
            reason="Remove leading/trailing whitespace",
        )
        assert record.step == "strip_whitespace"
        assert record.kwargs == {}
        assert record.rows_before == 100
        assert record.rows_after == 100
        assert record.rows_removed == 0
        assert record.reason == "Remove leading/trailing whitespace"

    def test_step_must_be_str(self):
        """Test that step field must be a string, not other types."""
        with pytest.raises(TypeError, match="CleanStepRecord.step must be a str"):
            CleanStepRecord(
                step=123,
                kwargs={},
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason="test",
            )

    def test_reason_must_be_str(self):
        """Test that reason field must be a string, not other types."""
        with pytest.raises(TypeError, match="CleanStepRecord.reason must be a str"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=5,
                rows_removed=5,
                reason=123,
            )

    def test_kwargs_must_be_dict(self):
        """Test that kwargs field must be a dictionary, not other types."""
        with pytest.raises(TypeError, match="CleanStepRecord.kwargs must be a dict"):
            CleanStepRecord(
                step="normalize_case",
                kwargs="not_a_dict",
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason="test",
            )

    def test_rows_before_must_be_int_not_bool(self):
        """Test that rows_before must be int and explicitly rejects bool."""
        with pytest.raises(TypeError, match="CleanStepRecord.rows_before must be an int"):
            CleanStepRecord(
                step="drop_duplicates",
                kwargs={},
                rows_before=True,
                rows_after=10,
                rows_removed=0,
                reason="test",
            )

    def test_rows_after_must_be_int_not_bool(self):
        """Test that rows_after must be int and explicitly rejects bool."""
        with pytest.raises(TypeError, match="CleanStepRecord.rows_after must be an int"):
            CleanStepRecord(
                step="drop_duplicates",
                kwargs={},
                rows_before=10,
                rows_after=False,
                rows_removed=0,
                reason="test",
            )

    def test_rows_removed_must_be_int_not_bool(self):
        """Test that rows_removed must be int and explicitly rejects bool."""
        with pytest.raises(TypeError, match="CleanStepRecord.rows_removed must be an int"):
            CleanStepRecord(
                step="drop_duplicates",
                kwargs={},
                rows_before=10,
                rows_after=8,
                rows_removed=True,
                reason="test",
            )

    def test_rows_before_cannot_be_negative(self):
        """Test that rows_before must be non-negative."""
        with pytest.raises(ValueError, match="CleanStepRecord.rows_before cannot be negative"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=-1,
                rows_after=10,
                rows_removed=0,
                reason="test",
            )

    def test_rows_after_cannot_be_negative(self):
        """Test that rows_after must be non-negative."""
        with pytest.raises(ValueError, match="CleanStepRecord.rows_after cannot be negative"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=-1,
                rows_removed=0,
                reason="test",
            )

    def test_rows_removed_cannot_be_negative(self):
        """Test that rows_removed must be non-negative."""
        with pytest.raises(ValueError, match="CleanStepRecord.rows_removed cannot be negative"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=5,
                rows_removed=-1,
                reason="test",
            )

    def test_frozen_dataclass_prevents_mutation(self):
        """Test that frozen dataclass prevents modification after creation."""
        record = CleanStepRecord(
            step="strip_whitespace",
            kwargs={},
            rows_before=100,
            rows_after=100,
            rows_removed=0,
            reason="test reason",
        )
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            record.step = "drop_nulls"

    def test_empty_kwargs_dict(self):
        """Test creating a record with an empty kwargs dictionary."""
        record = CleanStepRecord(
            step="drop_empty_columns",
            kwargs={},
            rows_before=100,
            rows_after=95,
            rows_removed=0,
            reason="Removed 5 empty columns",
        )
        assert record.kwargs == {}
        assert len(record.kwargs) == 0

    def test_complex_kwargs_with_nested_values(self):
        """Test creating a record with complex kwargs containing various types."""
        kwargs = {
            "subset": ["col1", "col2"],
            "threshold": 0.5,
            "keep": "first",
            "nested_list": [1, 2, 3],
        }
        record = CleanStepRecord(
            step="filter_rows",
            kwargs=kwargs,
            rows_before=100,
            rows_after=75,
            rows_removed=25,
            reason="Filtered rows based on threshold",
        )
        assert record.kwargs == kwargs
        assert record.kwargs["subset"] == ["col1", "col2"]

    def test_zero_row_counts_allowed(self):
        """Test creating a record with zero row counts (edge case but valid)."""
        record = CleanStepRecord(
            step="drop_duplicates",
            kwargs={"keep": "first"},
            rows_before=0,
            rows_after=0,
            rows_removed=0,
            reason="No rows to process",
        )
        assert record.rows_before == 0
        assert record.rows_after == 0
        assert record.rows_removed == 0
