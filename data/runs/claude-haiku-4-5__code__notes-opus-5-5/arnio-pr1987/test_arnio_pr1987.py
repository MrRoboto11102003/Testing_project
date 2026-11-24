import pytest
import re
from collections import OrderedDict
import numpy as np
from arnio.quality import CleanStepRecord
import dataclasses


class TestCleanStepRecordValidation:
    """Test CleanStepRecord validation and behavior."""

    def test_valid_minimal_record(self):
        """Test that a minimal valid record can be created."""
        rec = CleanStepRecord(step="", kwargs={}, rows_before=0, rows_after=0, rows_removed=0, reason="")
        assert rec.step == ""
        assert rec.kwargs == {}
        assert rec.rows_before == 0

    def test_valid_record_with_complex_kwargs_and_dict_subclass(self):
        """Test that complex kwargs and dict subclasses are accepted."""
        od = OrderedDict([("subset", ["col1", "col2"]), ("keep", "first")])
        rec = CleanStepRecord(
            step="drop_duplicates",
            kwargs=od,
            rows_before=100,
            rows_after=95,
            rows_removed=5,
            reason="Low duplicate ratio"
        )
        assert rec.kwargs is od

    def test_valid_record_with_large_integers(self):
        """Test that large integers are accepted."""
        rec = CleanStepRecord(
            step="test",
            kwargs={},
            rows_before=10**12,
            rows_after=10**12,
            rows_removed=0,
            reason="test"
        )
        assert rec.rows_before == 10**12

    def test_type_error_step_not_str(self):
        """Test that step must be a str."""
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.step must be a str, got int")):
            CleanStepRecord(step=1, kwargs={}, rows_before=0, rows_after=0, rows_removed=0, reason="r")

    def test_type_error_reason_checked_before_kwargs(self):
        """Test that reason is checked before kwargs."""
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.reason must be a str, got NoneType")):
            CleanStepRecord(step="s", kwargs=[], rows_before=0, rows_after=0, rows_removed=0, reason=None)

    def test_type_error_kwargs_not_dict(self):
        """Test that kwargs must be a dict."""
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.kwargs must be a dict, got list")):
            CleanStepRecord(step="s", kwargs=[], rows_before=0, rows_after=0, rows_removed=0, reason="r")

    def test_type_error_rows_before_bool_rejected(self):
        """Test that bool is rejected for rows_before even though bool is int in Python."""
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.rows_before must be an int, got bool")):
            CleanStepRecord(step="s", kwargs={}, rows_before=True, rows_after=0, rows_removed=0, reason="r")

    def test_type_error_rows_before_float(self):
        """Test that float is rejected for rows_before."""
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.rows_before must be an int, got float")):
            CleanStepRecord(step="s", kwargs={}, rows_before=1.0, rows_after=0, rows_removed=0, reason="r")

    def test_type_error_rows_after_string_rejected(self):
        """Test that non-int types are rejected for rows_after."""
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.rows_after must be an int, got str")):
            CleanStepRecord(step="s", kwargs={}, rows_before=0, rows_after="1", rows_removed=0, reason="r")

    def test_type_error_rows_removed_numpy_int64_rejected(self):
        """Test that numpy.int64 is rejected for rows_removed."""
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.rows_removed must be an int, got int64")):
            CleanStepRecord(step="s", kwargs={}, rows_before=0, rows_after=0, rows_removed=np.int64(1), reason="r")

    def test_value_error_rows_before_negative(self):
        """Test that negative rows_before raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("CleanStepRecord.rows_before cannot be negative: -1")):
            CleanStepRecord(step="s", kwargs={}, rows_before=-1, rows_after=0, rows_removed=0, reason="r")

    def test_value_error_rows_after_negative(self):
        """Test that negative rows_after raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("CleanStepRecord.rows_after cannot be negative: -1")):
            CleanStepRecord(step="s", kwargs={}, rows_before=0, rows_after=-1, rows_removed=0, reason="r")

    def test_value_error_rows_removed_negative(self):
        """Test that negative rows_removed raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("CleanStepRecord.rows_removed cannot be negative: -5")):
            CleanStepRecord(step="s", kwargs={}, rows_before=10, rows_after=10, rows_removed=-5, reason="r")

    def test_no_consistency_check_rows_after_greater_than_before(self):
        """Test that rows_after > rows_before is allowed (no consistency check)."""
        rec = CleanStepRecord(
            step="x",
            kwargs={},
            rows_before=1,
            rows_after=5,
            rows_removed=0,
            reason="r"
        )
        assert rec.rows_after > rec.rows_before

    def test_frozen_instance_and_equality(self):
        """Test that the dataclass is frozen and equality works correctly."""
        rec1 = CleanStepRecord(step="a", kwargs={"k": 1}, rows_before=2, rows_after=2, rows_removed=0, reason="r")
        rec2 = CleanStepRecord(step="a", kwargs={"k": 1}, rows_before=2, rows_after=2, rows_removed=0, reason="r")
        
        assert rec1 == rec2
        
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec1.step = "new"
