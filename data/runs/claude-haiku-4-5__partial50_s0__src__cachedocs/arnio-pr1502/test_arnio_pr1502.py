import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestAutoCleanReturnTypes:
    """Test different return type combinations."""

    def test_auto_clean_default_returns_arframe(self):
        """Default call returns ArFrame."""
        frame = ar.from_pandas(pd.DataFrame({"name": [" Alice ", " Bob "]}))
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)

    def test_auto_clean_dry_run_returns_report(self):
        """dry_run=True returns DataQualityReport without modifying frame."""
        frame = ar.from_pandas(pd.DataFrame({"name": [" Alice ", " Bob "]}))
        report = ar.auto_clean(frame, dry_run=True)
        assert isinstance(report, ar.DataQualityReport)
        # Original frame unchanged
        assert ar.to_pandas(frame)["name"].tolist() == [" Alice ", " Bob "]

    def test_auto_clean_return_report_true(self):
        """return_report=True returns (frame, report) tuple."""
        frame = ar.from_pandas(pd.DataFrame({"name": [" Alice "]}))
        result, report = ar.auto_clean(frame, return_report=True)
        assert isinstance(result, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)

    def test_auto_clean_explain_true(self):
        """explain=True returns (frame, explanation) tuple."""
        frame = ar.from_pandas(pd.DataFrame({"name": [" Alice "]}))
        result, explanation = ar.auto_clean(frame, explain=True)
        assert isinstance(result, ar.ArFrame)
        assert hasattr(explanation, "steps")
        assert hasattr(explanation, "mode")
        assert explanation.mode == "safe"

    def test_auto_clean_return_report_and_explain(self):
        """Both flags returns (frame, report, explanation) tuple."""
        frame = ar.from_pandas(pd.DataFrame({"name": [" Alice "]}))
        result, report, explanation = ar.auto_clean(
            frame, return_report=True, explain=True
        )
        assert isinstance(result, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)
        assert hasattr(explanation, "steps")


class TestAutoCleanModes:
    """Test safe vs strict mode behavior."""

    def test_safe_mode_only_strips_whitespace(self):
        """Safe mode applies only strip_whitespace, preserves duplicates."""
        frame = ar.from_pandas(
            pd.DataFrame({"name": [" Alice ", " Alice "]})
        )
        result = ar.auto_clean(frame, mode="safe")
        df = ar.to_pandas(result)
        # Whitespace stripped but duplicates kept in safe mode
        assert df["name"].tolist() == ["Alice", "Alice"]
        assert len(df) == 2

    def test_strict_mode_with_confirmed_casts(self):
        """Strict mode applies all steps when confirmed_casts provided."""
        frame = ar.from_pandas(
            pd.DataFrame({"name": [" Alice ", " Bob "]})
        )
        result = ar.auto_clean(
            frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={},
        )
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        assert df["name"].tolist() == ["Alice", "Bob"]


class TestAutoCleanErrorHandling:
    """Test error conditions."""

    def test_invalid_mode_raises_valueerror(self):
        """Invalid mode string raises ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
            ar.auto_clean(frame, mode="invalid")

    def test_non_arframe_input_raises_error(self):
        """Non-ArFrame input raises TypeError or AttributeError."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises((TypeError, AttributeError)):
            ar.auto_clean(df)

    def test_dry_run_and_explain_mutually_exclusive(self):
        """dry_run=True and explain=True together raise ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
            ar.auto_clean(frame, dry_run=True, explain=True)

    def test_dry_run_and_return_report_mutually_exclusive(self):
        """dry_run=True and return_report=True together raise ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
            ar.auto_clean(frame, dry_run=True, return_report=True)

    def test_invalid_return_report_type(self):
        """Non-bool return_report raises TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="return_report must be a bool"):
            ar.auto_clean(frame, return_report="invalid")

    def test_invalid_dry_run_type(self):
        """Non-bool dry_run raises TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.auto_clean(frame, dry_run="invalid")

    def test_invalid_explain_type(self):
        """Non-bool explain raises TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="explain must be a bool"):
            ar.auto_clean(frame, explain="invalid")

    def test_strict_mode_requires_allow_lossy_casts_when_needed(self):
        """Strict mode raises error if type casts needed but allow_lossy_casts=False."""
        frame = ar.from_pandas(
            pd.DataFrame({"value": ["1", "2", "3"]})
        )
        with pytest.raises(ValueError, match="allow_lossy_casts=True"):
            ar.auto_clean(frame, mode="strict", allow_lossy_casts=False)

    def test_strict_mode_requires_confirmed_casts_match_exactly(self):
        """confirmed_casts must match proposed mapping exactly."""
        frame = ar.from_pandas(
            pd.DataFrame({"name": [" Alice ", " Bob "]})
        )
        with pytest.raises(ValueError, match="confirmed_casts must match"):
            ar.auto_clean(
                frame,
                mode="strict",
                allow_lossy_casts=True,
                confirmed_casts={"wrong": "mapping"},
            )


class TestAutoCleanEdgeCases:
    """Test edge cases and data preservation."""

    def test_empty_frame(self):
        """Empty frame handled gracefully."""
        frame = ar.from_pandas(pd.DataFrame({"a": []}))
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] == 0

    def test_column_order_preserved(self):
        """Column order is preserved after cleaning."""
        frame = ar.from_pandas(
            pd.DataFrame({"z": [1], "a": [2], "m": [3]})
        )
        result = ar.auto_clean(frame)
        assert ar.to_pandas(result).columns.tolist() == ["z", "a", "m"]

    def test_numeric_types_preserved(self):
        """Numeric types preserved in safe mode."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
            })
        )
        result = ar.auto_clean(frame, mode="safe")
        df = ar.to_pandas(result)
        assert pd.api.types.is_integer_dtype(df["int_col"])
        assert pd.api.types.is_float_dtype(df["float_col"])
