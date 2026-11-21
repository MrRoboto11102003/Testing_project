import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
import tempfile
from pathlib import Path
from arnio.quality import CleanStepRecord


class TestCleanStepRecordValidation:
    """Test CleanStepRecord validation and type checking."""

    def test_valid_clean_step_record_creation(self):
        """Test creating a valid CleanStepRecord with all fields."""
        record = CleanStepRecord(
            step="strip_whitespace",
            kwargs={"columns": ["name", "email"]},
            rows_before=1000,
            rows_after=1000,
            rows_removed=0,
            reason="Strip leading/trailing whitespace from text columns"
        )
        assert record.step == "strip_whitespace"
        assert record.kwargs == {"columns": ["name", "email"]}
        assert record.rows_before == 1000
        assert record.rows_after == 1000
        assert record.rows_removed == 0

    def test_step_must_be_string_type(self):
        """Test that step field must be a string, not int or other type."""
        with pytest.raises(TypeError, match="CleanStepRecord.step must be a str"):
            CleanStepRecord(
                step=123,
                kwargs={},
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason="test reason"
            )

    def test_reason_must_be_string_type(self):
        """Test that reason field must be a string."""
        with pytest.raises(TypeError, match="CleanStepRecord.reason must be a str"):
            CleanStepRecord(
                step="filter",
                kwargs={},
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason=None
            )

    def test_kwargs_must_be_dict_type(self):
        """Test that kwargs field must be a dict, not list or other type."""
        with pytest.raises(TypeError, match="CleanStepRecord.kwargs must be a dict"):
            CleanStepRecord(
                step="filter",
                kwargs=["invalid"],
                rows_before=10,
                rows_after=10,
                rows_removed=0,
                reason="test"
            )

    def test_rows_before_rejects_bool_type(self):
        """Test that rows_before rejects bool (bool is subclass of int in Python)."""
        with pytest.raises(TypeError, match="CleanStepRecord.rows_before must be an int"):
            CleanStepRecord(
                step="filter",
                kwargs={},
                rows_before=True,
                rows_after=10,
                rows_removed=0,
                reason="test"
            )

    def test_rows_after_rejects_bool_type(self):
        """Test that rows_after rejects bool."""
        with pytest.raises(TypeError, match="CleanStepRecord.rows_after must be an int"):
            CleanStepRecord(
                step="filter",
                kwargs={},
                rows_before=10,
                rows_after=False,
                rows_removed=0,
                reason="test"
            )

    def test_rows_removed_rejects_bool_type(self):
        """Test that rows_removed rejects bool."""
        with pytest.raises(TypeError, match="CleanStepRecord.rows_removed must be an int"):
            CleanStepRecord(
                step="filter",
                kwargs={},
                rows_before=10,
                rows_after=10,
                rows_removed=True,
                reason="test"
            )

    def test_rows_before_cannot_be_negative(self):
        """Test that rows_before cannot be negative."""
        with pytest.raises(ValueError, match="CleanStepRecord.rows_before cannot be negative"):
            CleanStepRecord(
                step="filter",
                kwargs={},
                rows_before=-1,
                rows_after=10,
                rows_removed=0,
                reason="test"
            )

    def test_rows_after_cannot_be_negative(self):
        """Test that rows_after cannot be negative."""
        with pytest.raises(ValueError, match="CleanStepRecord.rows_after cannot be negative"):
            CleanStepRecord(
                step="filter",
                kwargs={},
                rows_before=10,
                rows_after=-5,
                rows_removed=0,
                reason="test"
            )

    def test_rows_removed_cannot_be_negative(self):
        """Test that rows_removed cannot be negative."""
        with pytest.raises(ValueError, match="CleanStepRecord.rows_removed cannot be negative"):
            CleanStepRecord(
                step="filter",
                kwargs={},
                rows_before=10,
                rows_after=5,
                rows_removed=-1,
                reason="test"
            )

    def test_clean_step_record_immutability(self):
        """Test that CleanStepRecord is frozen and immutable."""
        record = CleanStepRecord(
            step="filter",
            kwargs={},
            rows_before=10,
            rows_after=5,
            rows_removed=5,
            reason="test"
        )
        with pytest.raises(Exception):  # dataclass frozen raises FrozenInstanceError
            record.step = "new_step"

    def test_clean_step_record_with_complex_kwargs(self):
        """Test CleanStepRecord with nested dict kwargs."""
        complex_kwargs = {
            "columns": ["col1", "col2"],
            "threshold": 0.95,
            "options": {"drop": True, "inplace": False}
        }
        record = CleanStepRecord(
            step="remove_duplicates",
            kwargs=complex_kwargs,
            rows_before=1000,
            rows_after=950,
            rows_removed=50,
            reason="Removed duplicate rows based on all columns"
        )
        assert record.kwargs == complex_kwargs
        assert record.kwargs["threshold"] == 0.95
        assert record.rows_removed == 50

    def test_all_rows_removed_edge_case(self):
        """Test CleanStepRecord when all rows are removed."""
        record = CleanStepRecord(
            step="filter_empty",
            kwargs={"column": "value"},
            rows_before=100,
            rows_after=0,
            rows_removed=100,
            reason="All rows had empty values in required column"
        )
        assert record.rows_after == 0
        assert record.rows_removed == 100
        assert record.rows_before == 100

    def test_zero_rows_removed_edge_case(self):
        """Test CleanStepRecord when no rows are removed."""
        record = CleanStepRecord(
            step="normalize",
            kwargs={"method": "standardize"},
            rows_before=500,
            rows_after=500,
            rows_removed=0,
            reason="Normalization step does not remove rows"
        )
        assert record.rows_removed == 0
        assert record.rows_before == record.rows_after

    def test_empty_dataframe_edge_case(self):
        """Test CleanStepRecord with empty dataframe (zero rows)."""
        record = CleanStepRecord(
            step="filter",
            kwargs={},
            rows_before=0,
            rows_after=0,
            rows_removed=0,
            reason="Input dataframe was already empty"
        )
        assert record.rows_before == 0
        assert record.rows_after == 0
        assert record.rows_removed == 0
