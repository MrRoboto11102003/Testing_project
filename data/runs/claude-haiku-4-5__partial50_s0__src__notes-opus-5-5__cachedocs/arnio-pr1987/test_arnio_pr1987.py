import re
import pytest
from dataclasses import FrozenInstanceError
from arnio.quality import CleanStepRecord


class TestCleanStepRecordConstruction:
    """Test valid construction of CleanStepRecord."""

    def test_minimal_valid_record(self):
        """All-valid minimal record with empty strings and zeros."""
        record = CleanStepRecord("", {}, 0, 0, 0, "")
        assert record.step == ""
        assert record.kwargs == {}
        assert record.rows_before == 0
        assert record.rows_after == 0
        assert record.rows_removed == 0
        assert record.reason == ""

    def test_typical_record_with_kwargs_preservation(self):
        """Typical record preserves kwargs dict object (not copied)."""
        kwargs = {"subset": ["name", "city"]}
        record = CleanStepRecord(
            "strip_whitespace",
            kwargs,
            100,
            100,
            0,
            "Trimming leading/trailing whitespace"
        )
        assert record.step == "strip_whitespace"
        assert record.kwargs is kwargs  # Same object, not copied
        assert record.rows_removed == 0

    def test_large_counts_and_dict_subclass(self):
        """Large row counts and dict subclasses like OrderedDict accepted."""
        from collections import OrderedDict
        d = OrderedDict([("a", 1), ("b", 2)])
        big = 10**12
        record = CleanStepRecord("step", d, big, big, 0, "reason")
        assert record.rows_before == big
        assert record.kwargs is d


class TestCleanStepRecordTypeValidation:
    """Test type validation in __post_init__, in correct order."""

    def test_step_must_be_str(self):
        """step type checked first."""
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.step must be a str, got int")):
            CleanStepRecord(42, {}, 0, 0, 0, "r")

    def test_reason_checked_before_kwargs(self):
        """reason type is checked before kwargs type."""
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.reason must be a str, got NoneType")):
            CleanStepRecord("s", "not_a_dict", 0, 0, 0, None)

    def test_kwargs_must_be_dict(self):
        """kwargs must be dict (not list, str, None, etc.)."""
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.kwargs must be a dict, got list")):
            CleanStepRecord("s", [], 0, 0, 0, "r")

    def test_row_fields_must_be_int_not_bool(self):
        """All row fields must be int; bool (even though bool is int in Python) is rejected."""
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.rows_before must be an int, got bool")):
            CleanStepRecord("s", {}, True, 0, 0, "r")
        
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.rows_after must be an int, got bool")):
            CleanStepRecord("s", {}, 0, False, 0, "r")
        
        with pytest.raises(TypeError, match=re.escape("CleanStepRecord.rows_removed must be an int, got bool")):
            CleanStepRecord("s", {}, 0, 0, True, "r")

    def test_row_fields_must_be_nonnegative(self):
        """All row fields must be >= 0; type checks run before negative checks."""
        with pytest.raises(ValueError, match=re.escape("CleanStepRecord.rows_before cannot be negative: -1")):
            CleanStepRecord("s", {}, -1, 0, 0, "r")
        
        with pytest.raises(ValueError, match=re.escape("CleanStepRecord.rows_after cannot be negative: -100")):
            CleanStepRecord("s", {}, 0, -100, 0, "r")
        
        with pytest.raises(ValueError, match=re.escape("CleanStepRecord.rows_removed cannot be negative: -5")):
            CleanStepRecord("s", {}, 0, 0, -5, "r")


class TestCleanStepRecordNoConsistencyCheck:
    """Test that there is NO consistency check between row fields."""

    def test_rows_after_can_exceed_rows_before(self):
        """rows_after > rows_before is allowed (no consistency enforced)."""
        record = CleanStepRecord("s", {}, 1, 5, 0, "r")
        assert record.rows_after == 5

    def test_rows_removed_independent_of_before_after(self):
        """rows_removed need not equal rows_before - rows_after."""
        record = CleanStepRecord("s", {}, 100, 50, 10, "r")
        assert record.rows_removed == 10  # Not corrected to 50


class TestCleanStepRecordFrozen:
    """Test frozen dataclass behavior."""

    def test_field_cannot_be_modified(self):
        """Frozen dataclass prevents field reassignment."""
        record = CleanStepRecord("s", {}, 0, 0, 0, "r")
        with pytest.raises(FrozenInstanceError):
            record.step = "new"

    def test_kwargs_dict_mutable_but_field_frozen(self):
        """The kwargs dict contents can be mutated, but the field itself cannot be reassigned."""
        d = {"key": "value"}
        record = CleanStepRecord("s", d, 0, 0, 0, "r")
        
        # Can mutate dict contents
        record.kwargs["key"] = "modified"
        assert record.kwargs["key"] == "modified"
        
        # Cannot reassign the field
        with pytest.raises(FrozenInstanceError):
            record.kwargs = {}


class TestCleanStepRecordEquality:
    """Test equality and hashing behavior."""

    def test_equal_records(self):
        """Records with identical fields are equal."""
        r1 = CleanStepRecord("s", {"k": 1}, 10, 10, 0, "r")
        r2 = CleanStepRecord("s", {"k": 1}, 10, 10, 0, "r")
        assert r1 == r2

    def test_different_field_makes_unequal(self):
        """Changing any field makes records unequal."""
        r1 = CleanStepRecord("step1", {}, 0, 0, 0, "r")
        r2 = CleanStepRecord("step2", {}, 0, 0, 0, "r")
        assert r1 != r2

    def test_unhashable_due_to_dict_field(self):
        """Record is unhashable because kwargs is an unhashable dict."""
        record = CleanStepRecord("s", {}, 0, 0, 0, "r")
        with pytest.raises(TypeError, match="unhashable type"):
            hash(record)
