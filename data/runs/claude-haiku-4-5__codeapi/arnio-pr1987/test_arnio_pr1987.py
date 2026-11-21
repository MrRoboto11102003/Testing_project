import pytest
from arnio.quality import CleanStepRecord


class TestCleanStepRecord:
    """Test CleanStepRecord validation and behavior."""
    
    def test_valid_creation(self):
        """Test creating a valid CleanStepRecord with realistic values."""
        record = CleanStepRecord(
            step="drop_nulls",
            kwargs={"subset": ["col1"]},
            rows_before=100,
            rows_after=90,
            rows_removed=10,
            reason="Removed rows with null values"
        )
        assert record.step == "drop_nulls"
        assert record.kwargs == {"subset": ["col1"]}
        assert record.rows_before == 100
        assert record.rows_after == 90
        assert record.rows_removed == 10
        assert record.reason == "Removed rows with null values"
    
    def test_zero_rows_removed_non_dropping_step(self):
        """Test that zero rows removed is valid for non-row-dropping steps."""
        record = CleanStepRecord(
            step="strip_whitespace",
            kwargs={},
            rows_before=100,
            rows_after=100,
            rows_removed=0,
            reason="No rows affected"
        )
        assert record.rows_removed == 0
    
    def test_empty_kwargs_dict(self):
        """Test that empty kwargs dict is valid."""
        record = CleanStepRecord(
            step="some_step",
            kwargs={},
            rows_before=10,
            rows_after=10,
            rows_removed=0,
            reason="No kwargs needed"
        )
        assert record.kwargs == {}
    
    def test_step_not_string_raises_typeerror(self):
        """Test that non-string step raises TypeError."""
        with pytest.raises(TypeError, match="CleanStepRecord.step must be a str"):
            CleanStepRecord(
                step=123,
                kwargs={},
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason="Valid reason"
            )
    
    def test_reason_not_string_raises_typeerror(self):
        """Test that non-string reason raises TypeError."""
        with pytest.raises(TypeError, match="CleanStepRecord.reason must be a str"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason=None
            )
    
    def test_kwargs_not_dict_raises_typeerror(self):
        """Test that non-dict kwargs raises TypeError."""
        with pytest.raises(TypeError, match="CleanStepRecord.kwargs must be a dict"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs="invalid",
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason="Valid reason"
            )
    
    def test_rows_before_not_int_raises_typeerror(self):
        """Test that non-int rows_before raises TypeError."""
        with pytest.raises(TypeError, match="CleanStepRecord.rows_before must be an int"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before="10",
                rows_after=10,
                rows_removed=0,
                reason="Valid reason"
            )
    
    def test_rows_after_not_int_raises_typeerror(self):
        """Test that non-int rows_after raises TypeError."""
        with pytest.raises(TypeError, match="CleanStepRecord.rows_after must be an int"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=5.5,
                rows_removed=0,
                reason="Valid reason"
            )
    
    def test_rows_removed_not_int_raises_typeerror(self):
        """Test that non-int rows_removed raises TypeError."""
        with pytest.raises(TypeError, match="CleanStepRecord.rows_removed must be an int"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=5,
                rows_removed=[],
                reason="Valid reason"
            )
    
    def test_bool_rejected_for_numeric_fields(self):
        """Test that bool is rejected for numeric fields despite being int subclass."""
        with pytest.raises(TypeError, match="CleanStepRecord.rows_before must be an int"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=True,
                rows_after=10,
                rows_removed=0,
                reason="Valid reason"
            )
    
    def test_negative_rows_before_raises_valueerror(self):
        """Test that negative rows_before raises ValueError."""
        with pytest.raises(ValueError, match="CleanStepRecord.rows_before cannot be negative"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=-1,
                rows_after=0,
                rows_removed=0,
                reason="Valid reason"
            )
    
    def test_negative_rows_after_raises_valueerror(self):
        """Test that negative rows_after raises ValueError."""
        with pytest.raises(ValueError, match="CleanStepRecord.rows_after cannot be negative"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=-1,
                rows_removed=0,
                reason="Valid reason"
            )
    
    def test_negative_rows_removed_raises_valueerror(self):
        """Test that negative rows_removed raises ValueError."""
        with pytest.raises(ValueError, match="CleanStepRecord.rows_removed cannot be negative"):
            CleanStepRecord(
                step="drop_nulls",
                kwargs={},
                rows_before=10,
                rows_after=10,
                rows_removed=-5,
                reason="Valid reason"
            )
    
    def test_frozen_dataclass_immutable(self):
        """Test that CleanStepRecord is frozen and cannot be modified."""
        record = CleanStepRecord(
            step="drop_nulls",
            kwargs={},
            rows_before=10,
            rows_after=10,
            rows_removed=0,
            reason="Valid reason"
        )
        with pytest.raises((AttributeError, ValueError)):
            record.step = "modified"
    
    def test_complex_kwargs_with_nested_structures(self):
        """Test that complex nested kwargs are preserved correctly."""
        complex_kwargs = {
            "subset": ["col1", "col2"],
            "nested": {"key": "value"},
            "list": [1, 2, 3]
        }
        record = CleanStepRecord(
            step="normalize_case",
            kwargs=complex_kwargs,
            rows_before=100,
            rows_after=100,
            rows_removed=0,
            reason="Normalized case with complex config"
        )
        assert record.kwargs == complex_kwargs
