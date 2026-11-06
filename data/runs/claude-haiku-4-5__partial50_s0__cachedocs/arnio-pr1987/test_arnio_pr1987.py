"""pytest unit tests for arnio.quality.CleanStepRecord"""

import pytest
from dataclasses import FrozenInstanceError
import arnio as ar
from arnio.quality import CleanStepRecord


class TestCleanStepRecord:
    """Test suite for arnio.quality.CleanStepRecord frozen dataclass."""

    def test_instantiation_basic(self):
        """Test basic instantiation with typical step metadata."""
        record = CleanStepRecord(
            step_name="strip_whitespace",
            elapsed_time=0.001,
            rows_before=100,
            rows_after=100
        )
        assert record.step_name == "strip_whitespace"
        assert record.elapsed_time == 0.001
        assert record.rows_before == 100
        assert record.rows_after == 100

    def test_frozen_dataclass_immutability(self):
        """Test that CleanStepRecord is immutable (frozen)."""
        record = CleanStepRecord(
            step_name="drop_nulls",
            elapsed_time=0.002,
            rows_before=100,
            rows_after=95
        )
        with pytest.raises(FrozenInstanceError):
            record.step_name = "modified"

    def test_frozen_prevents_attribute_addition(self):
        """Test that frozen dataclass prevents adding new attributes."""
        record = CleanStepRecord(
            step_name="test_step",
            elapsed_time=0.001,
            rows_before=100,
            rows_after=100
        )
        with pytest.raises(FrozenInstanceError):
            record.new_attribute = "value"

    def test_equality_identical_records(self):
        """Test that two records with identical values are equal."""
        record1 = CleanStepRecord(
            step_name="normalize_case",
            elapsed_time=0.0005,
            rows_before=500,
            rows_after=500
        )
        record2 = CleanStepRecord(
            step_name="normalize_case",
            elapsed_time=0.0005,
            rows_before=500,
            rows_after=500
        )
        assert record1 == record2

    def test_inequality_different_step_names(self):
        """Test that records with different step names are not equal."""
        record1 = CleanStepRecord(
            step_name="strip_whitespace",
            elapsed_time=0.001,
            rows_before=100,
            rows_after=100
        )
        record2 = CleanStepRecord(
            step_name="drop_nulls",
            elapsed_time=0.001,
            rows_before=100,
            rows_after=95
        )
        assert record1 != record2

    def test_inequality_different_timing(self):
        """Test that records with different elapsed times are not equal."""
        record1 = CleanStepRecord(
            step_name="same_step",
            elapsed_time=0.001,
            rows_before=100,
            rows_after=100
        )
        record2 = CleanStepRecord(
            step_name="same_step",
            elapsed_time=0.002,
            rows_before=100,
            rows_after=100
        )
        assert record1 != record2

    def test_hash_enables_set_operations(self):
        """Test that hashing allows records to be used in sets."""
        record1 = CleanStepRecord(
            step_name="fill_nulls",
            elapsed_time=0.003,
            rows_before=200,
            rows_after=200
        )
        record2 = CleanStepRecord(
            step_name="drop_duplicates",
            elapsed_time=0.004,
            rows_before=200,
            rows_after=180
        )
        records_set = {record1, record2}
        assert len(records_set) == 2
        assert record1 in records_set

    def test_zero_elapsed_time(self):
        """Test with zero elapsed time (very fast step)."""
        record = CleanStepRecord(
            step_name="fast_step",
            elapsed_time=0.0,
            rows_before=1000,
            rows_after=1000
        )
        assert record.elapsed_time == 0.0

    def test_zero_rows_after_drop(self):
        """Test with zero rows remaining after filtering step."""
        record = CleanStepRecord(
            step_name="drop_nulls",
            elapsed_time=0.01,
            rows_before=100,
            rows_after=0
        )
        assert record.rows_after == 0

    def test_row_count_reduction_deduplication(self):
        """Test typical drop_duplicates scenario with row reduction."""
        record = CleanStepRecord(
            step_name="drop_duplicates",
            elapsed_time=0.008,
            rows_before=10000,
            rows_after=9500
        )
        assert record.rows_before > record.rows_after
        row_reduction = record.rows_before - record.rows_after
        assert row_reduction == 500

    def test_row_count_unchanged(self):
        """Test steps that don't change row count like strip_whitespace."""
        record = CleanStepRecord(
            step_name="strip_whitespace",
            elapsed_time=0.0015,
            rows_before=5000,
            rows_after=5000
        )
        assert record.rows_before == record.rows_after

    def test_large_row_counts(self):
        """Test with realistically large row counts."""
        record = CleanStepRecord(
            step_name="large_file_process",
            elapsed_time=1.5,
            rows_before=1_000_000,
            rows_after=999_000
        )
        assert record.rows_before == 1_000_000
        assert record.rows_after == 999_000

    def test_high_precision_timing(self):
        """Test with microsecond-level precision in elapsed time."""
        record = CleanStepRecord(
            step_name="micro_step",
            elapsed_time=0.000001,
            rows_before=10,
            rows_after=10
        )
        assert record.elapsed_time == 0.000001

    def test_repr_contains_step_info(self):
        """Test that repr includes readable step information."""
        record = CleanStepRecord(
            step_name="rename_columns",
            elapsed_time=0.0002,
            rows_before=300,
            rows_after=300
        )
        record_repr = repr(record)
        # Frozen dataclass should include class name and field values in repr
        assert "CleanStepRecord" in record_repr or "rename_columns" in record_repr
