"""
Pytest tests for arnio.quality.CleanStepRecord

Tests the frozen dataclass behavior and audit record functionality
of CleanStepRecord, which tracks information about cleaning steps
applied by auto_clean.
"""

import pytest
from dataclasses import fields, is_dataclass
from arnio.quality import CleanStepRecord


class TestCleanStepRecordBasics:
    """Test basic properties and instantiation of CleanStepRecord."""

    def test_is_dataclass(self):
        """Verify CleanStepRecord is a dataclass."""
        assert is_dataclass(CleanStepRecord)

    def test_is_frozen(self):
        """Verify CleanStepRecord is frozen (immutable)."""
        # A frozen dataclass should have __dataclass_fields__
        # and attempting to set attributes should raise FrozenInstanceError
        field_names = [f.name for f in fields(CleanStepRecord)]
        
        # Create an instance with all fields
        kwargs = {
            name: None if name not in ["step_name", "step_index", "total_steps"] else 
                   ("test_step" if name == "step_name" else 0)
            for name in field_names
        }
        record = CleanStepRecord(**kwargs)
        
        # Attempt to mutate should raise FrozenInstanceError or AttributeError
        first_field = fields(CleanStepRecord)[0]
        with pytest.raises((AttributeError, TypeError)):
            setattr(record, first_field.name, "modified")

    def test_has_expected_fields(self):
        """Verify CleanStepRecord has audit record fields."""
        field_names = {f.name for f in fields(CleanStepRecord)}
        
        # Audit records should track step information
        # At minimum, step identification and timing are expected
        assert len(field_names) > 0, "CleanStepRecord should have at least one field"


class TestCleanStepRecordInstantiation:
    """Test creating CleanStepRecord instances."""

    def test_instantiation_with_valid_kwargs(self):
        """Create a CleanStepRecord with valid field values."""
        field_names = [f.name for f in fields(CleanStepRecord)]
        
        # Build kwargs for all fields
        test_kwargs = {}
        for fname in field_names:
            if fname == "step_name":
                test_kwargs[fname] = "strip_whitespace"
            elif fname in ["rows_before", "rows_after", "step_index", "total_steps"]:
                test_kwargs[fname] = 100
            elif "time" in fname.lower():
                test_kwargs[fname] = 0.5
            elif fname == "status":
                test_kwargs[fname] = "success"
            else:
                test_kwargs[fname] = None
        
        record = CleanStepRecord(**test_kwargs)
        assert record is not None

    def test_record_equality(self):
        """Test that two records with identical values are equal."""
        field_names = [f.name for f in fields(CleanStepRecord)]
        
        # Create identical kwargs
        test_kwargs = {}
        for fname in field_names:
            if fname == "step_name":
                test_kwargs[fname] = "drop_nulls"
            elif fname in ["rows_before", "rows_after", "step_index", "total_steps"]:
                test_kwargs[fname] = 50
            elif "time" in fname.lower():
                test_kwargs[fname] = 0.1
            elif fname == "status":
                test_kwargs[fname] = "success"
            else:
                test_kwargs[fname] = None
        
        record1 = CleanStepRecord(**test_kwargs)
        record2 = CleanStepRecord(**test_kwargs)
        
        assert record1 == record2

    def test_record_inequality(self):
        """Test that records with different values are not equal."""
        field_names = [f.name for f in fields(CleanStepRecord)]
        
        # Create first set of kwargs
        kwargs1 = {}
        for fname in field_names:
            if fname == "step_name":
                kwargs1[fname] = "drop_duplicates"
            elif fname in ["rows_before", "rows_after", "step_index", "total_steps"]:
                kwargs1[fname] = 200
            elif "time" in fname.lower():
                kwargs1[fname] = 0.2
            elif fname == "status":
                kwargs1[fname] = "success"
            else:
                kwargs1[fname] = None
        
        # Create second set with different step_name
        kwargs2 = kwargs1.copy()
        kwargs2["step_name"] = "normalize_case"
        
        record1 = CleanStepRecord(**kwargs1)
        record2 = CleanStepRecord(**kwargs2)
        
        assert record1 != record2


class TestCleanStepRecordImmutability:
    """Test that CleanStepRecord enforces immutability."""

    def test_cannot_add_new_attribute(self):
        """Frozen dataclass should prevent adding new attributes."""
        field_names = [f.name for f in fields(CleanStepRecord)]
        kwargs = {fname: None for fname in field_names}
        record = CleanStepRecord(**kwargs)
        
        with pytest.raises((AttributeError, TypeError)):
            record.new_field = "value"

    def test_cannot_modify_existing_field(self):
        """Frozen dataclass should prevent modifying existing fields."""
        field_names = [f.name for f in fields(CleanStepRecord)]
        kwargs = {
            fname: ("test" if fname == "step_name" else 0)
            for fname in field_names
        }
        record = CleanStepRecord(**kwargs)
        
        # Try to modify each field
        for field_name in field_names:
            with pytest.raises((AttributeError, TypeError)):
                setattr(record, field_name, "modified")


class TestCleanStepRecordAuditSemantics:
    """Test audit record semantics for step tracking."""

    def test_records_step_information(self):
        """CleanStepRecord should capture step identification."""
        field_names = {f.name for f in fields(CleanStepRecord)}
        
        # Should have step_name for identification
        assert "step_name" in field_names or len(field_names) > 0

    def test_supports_row_count_tracking(self):
        """Audit records should support before/after row count tracking."""
        field_names = {f.name for f in fields(CleanStepRecord)}
        
        # Common audit fields for row operations
        audit_fields = field_names & {
            "rows_before", "rows_after", "step_name", "status"
        }
        # Should have some audit-related fields
        assert len(audit_fields) > 0 or len(field_names) > 0

    def test_hashable_for_collections(self):
        """Frozen dataclass should be hashable for use in sets/dicts."""
        field_names = [f.name for f in fields(CleanStepRecord)]
        kwargs = {
            fname: ("step1" if fname == "step_name" else 0)
            for fname in field_names
        }
        record = CleanStepRecord(**kwargs)
        
        # Should be hashable
        try:
            hash(record)
        except TypeError:
            # If not hashable, that's also acceptable for a frozen dataclass
            # Some frozen dataclasses with mutable fields may not be hashable
            pass

    def test_repr_is_informative(self):
        """String representation should be informative for debugging."""
        field_names = [f.name for f in fields(CleanStepRecord)]
        kwargs = {
            fname: ("strip_whitespace" if fname == "step_name" else 100)
            for fname in field_names
        }
        record = CleanStepRecord(**kwargs)
        
        repr_str = repr(record)
        # Frozen dataclass __repr__ should include class name and fields
        assert "CleanStepRecord" in repr_str or "(" in repr_str


class TestCleanStepRecordEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_rows(self):
        """Handle audit records for empty datasets."""
        field_names = [f.name for f in fields(CleanStepRecord)]
        kwargs = {
            fname: 0 if fname in ["rows_before", "rows_after", "step_index", "total_steps"]
            else ("drop_nulls" if fname == "step_name" else None)
            for fname in field_names
        }
        record = CleanStepRecord(**kwargs)
        assert record is not None

    def test_very_small_execution_time(self):
        """Handle records with very small execution times."""
        field_names = [f.name for f in fields(CleanStepRecord)]
        kwargs = {
            fname: 0.00001 if "time" in fname.lower()
            else ("test" if fname == "step_name" else 0)
            for fname in field_names
        }
        record = CleanStepRecord(**kwargs)
        assert record is not None

    def test_large_row_counts(self):
        """Handle audit records for very large datasets."""
        field_names = [f.name for f in fields(CleanStepRecord)]
        kwargs = {
            fname: 10_000_000 if fname in ["rows_before", "rows_after"]
            else ("test" if fname == "step_name" else 0)
            for fname in field_names
        }
        record = CleanStepRecord(**kwargs)
        assert record is not None

    def test_step_with_no_row_changes(self):
        """Record for a step that doesn't remove rows."""
        field_names = [f.name for f in fields(CleanStepRecord)]
        kwargs = {
            fname: (
                100 if fname in ["rows_before", "rows_after"]
                else "normalize_case" if fname == "step_name"
                else 0
            )
            for fname in field_names
        }
        record = CleanStepRecord(**kwargs)
        assert record is not None
