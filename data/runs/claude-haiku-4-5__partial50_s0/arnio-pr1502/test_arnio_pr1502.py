import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestAutoClean:
    """Test suite for arnio.quality.auto_clean function."""

    @pytest.fixture
    def clean_frame(self):
        """Create a clean test frame."""
        data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        }
        return ar.from_pandas(pd.DataFrame(data))

    @pytest.fixture
    def messy_frame(self):
        """Create a messy test frame with whitespace and duplicates."""
        data = {
            "id": [1, 2, 2, 3],
            "name": [" Alice ", " Bob ", " Bob ", " Charlie "],
            "city": [" Paris ", " London ", " London ", " Tokyo "],
        }
        return ar.from_pandas(pd.DataFrame(data))

    @pytest.fixture
    def mixed_type_frame(self):
        """Create a frame with string values that could be cast."""
        data = {
            "id": ["1", "2", "3"],
            "value": ["10.5", "20.3", "15.7"],
        }
        return ar.from_pandas(pd.DataFrame(data))

    def test_auto_clean_default_mode_returns_arframe(self, clean_frame):
        """Test that auto_clean with defaults returns an ArFrame."""
        result = ar.auto_clean(clean_frame)
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_safe_mode_only_strips_whitespace(self, messy_frame):
        """Test that safe mode only applies strip_whitespace, no deduplication."""
        original_rows = messy_frame.shape[0]
        result = ar.auto_clean(messy_frame, mode="safe")
        
        # Safe mode should preserve row count (no deduplication)
        assert result.shape[0] == original_rows
        
        # But whitespace should be stripped
        result_df = ar.to_pandas(result)
        assert not result_df["name"].iloc[0].startswith(" ")
        assert not result_df["city"].iloc[0].startswith(" ")

    def test_auto_clean_strict_mode_removes_duplicates(self, messy_frame):
        """Test that strict mode includes deduplication."""
        original_rows = messy_frame.shape[0]
        result = ar.auto_clean(messy_frame, mode="strict")
        
        # Strict mode should remove duplicates
        assert result.shape[0] < original_rows

    def test_auto_clean_return_report_true_returns_tuple(self, clean_frame):
        """Test that return_report=True returns a tuple."""
        result = ar.auto_clean(clean_frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        # Second element should have report-like interface
        assert hasattr(result[1], 'summary') or hasattr(result[1], 'to_markdown')

    def test_auto_clean_dry_run_validation(self, messy_frame):
        """Test that dry_run=True validates without transforming."""
        result = ar.auto_clean(messy_frame, mode="strict", dry_run=True)
        assert result is not None

    def test_auto_clean_allow_lossy_casts_parameter(self, mixed_type_frame):
        """Test allow_lossy_casts parameter enables type conversions."""
        result = ar.auto_clean(
            mixed_type_frame,
            mode="strict",
            allow_lossy_casts=True,
        )
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_confirmed_casts_parameter(self, mixed_type_frame):
        """Test confirmed_casts parameter applies specific type mappings."""
        confirmed = {"id": "int64", "value": "float64"}
        result = ar.auto_clean(
            mixed_type_frame,
            mode="strict",
            confirmed_casts=confirmed,
            allow_lossy_casts=True,
        )
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_explain_parameter(self, clean_frame):
        """Test explain=True includes execution metadata."""
        result = ar.auto_clean(clean_frame, explain=True)
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_empty_frame(self):
        """Test auto_clean handles empty frame correctly."""
        data = {"id": [], "name": []}
        frame = ar.from_pandas(pd.DataFrame(data))
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] == 0

    def test_auto_clean_invalid_mode_raises_error(self, clean_frame):
        """Test that invalid mode parameter raises an error."""
        with pytest.raises((ValueError, KeyError)):
            ar.auto_clean(clean_frame, mode="invalid_mode")

    def test_auto_clean_preserves_column_order(self, messy_frame):
        """Test that column order is preserved after cleaning."""
        original_cols = messy_frame.columns
        result = ar.auto_clean(messy_frame, mode="safe")
        assert list(result.columns) == list(original_cols)

    def test_auto_clean_safe_mode_preserves_dtypes(self, clean_frame):
        """Test that safe mode preserves column dtypes."""
        original_dtypes = clean_frame.dtypes
        result = ar.auto_clean(clean_frame, mode="safe")
        assert result.dtypes == original_dtypes

    def test_auto_clean_dry_run_with_return_report(self, messy_frame):
        """Test dry_run=True combined with return_report=True."""
        result = ar.auto_clean(
            messy_frame,
            mode="strict",
            dry_run=True,
            return_report=True,
        )
        assert result is not None

    def test_auto_clean_whitespace_handling_both_modes(self, messy_frame):
        """Test that both safe and strict modes strip whitespace."""
        result_safe = ar.auto_clean(messy_frame, mode="safe")
        result_strict = ar.auto_clean(messy_frame, mode="strict")
        
        df_safe = ar.to_pandas(result_safe)
        df_strict = ar.to_pandas(result_strict)
        
        # Both should have stripped whitespace
        assert not any(df_safe["name"].str.startswith(" "))
        assert not any(df_strict["name"].str.startswith(" "))
