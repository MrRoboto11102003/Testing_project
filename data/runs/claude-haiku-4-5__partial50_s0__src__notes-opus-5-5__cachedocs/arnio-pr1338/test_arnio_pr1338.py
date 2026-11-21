import pytest
import pandas as pd
import arnio as ar
from arnio.quality import DataQualityReport, CleanExplanation


class TestAutoCleanReturnTypes:
    """Test auto_clean return types for different flag combinations."""

    def test_default_returns_arframe(self):
        """Default call returns an ArFrame."""
        df = pd.DataFrame({"name": [" alice", "bob"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)

    def test_dry_run_returns_report(self):
        """dry_run=True returns DataQualityReport without modifications."""
        df = pd.DataFrame({"name": [" alice"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, dry_run=True)
        assert isinstance(result, DataQualityReport)

    def test_return_report_returns_tuple(self):
        """return_report=True returns (ArFrame, DataQualityReport)."""
        df = pd.DataFrame({"name": [" alice"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, return_report=True)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], DataQualityReport)

    def test_explain_returns_explanation_tuple(self):
        """explain=True returns (ArFrame, CleanExplanation)."""
        df = pd.DataFrame({"name": [" alice"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, explain=True)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], CleanExplanation)

    def test_both_report_and_explain_returns_triple(self):
        """Both flags return (ArFrame, DataQualityReport, CleanExplanation)."""
        df = pd.DataFrame({"name": [" alice"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, return_report=True, explain=True)
        assert isinstance(result, tuple) and len(result) == 3
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], DataQualityReport)
        assert isinstance(result[2], CleanExplanation)


class TestAutoCleanValidation:
    """Test input validation and error handling."""

    def test_mode_case_sensitive(self):
        """Mode must be 'safe' or 'strict' (lowercase, case-sensitive)."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
            ar.auto_clean(frame, mode="SAFE")

    def test_mode_invalid_value(self):
        """Invalid mode string raises ValueError."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
            ar.auto_clean(frame, mode="fast")

    def test_return_report_must_be_bool(self):
        """return_report must be bool, not int."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="return_report must be a bool"):
            ar.auto_clean(frame, return_report=1)

    def test_dry_run_must_be_bool(self):
        """dry_run must be bool, not int."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.auto_clean(frame, dry_run=0)

    def test_allow_lossy_casts_must_be_bool(self):
        """allow_lossy_casts must be bool."""
        df = pd.DataFrame({"num": ["1"]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="allow_lossy_casts must be a bool"):
            ar.auto_clean(frame, allow_lossy_casts=1)

    def test_explain_must_be_bool(self):
        """explain must be bool."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="explain must be a bool"):
            ar.auto_clean(frame, explain="yes")

    def test_dry_run_and_explain_conflict(self):
        """dry_run=True and explain=True cannot be used together."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
            ar.auto_clean(frame, dry_run=True, explain=True)

    def test_dry_run_and_return_report_conflict(self):
        """dry_run=True and return_report=True cannot be used together."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
            ar.auto_clean(frame, dry_run=True, return_report=True)

    def test_confirmed_casts_must_be_dict_or_none(self):
        """confirmed_casts must be dict[str, str] or None."""
        df = pd.DataFrame({"x": ["1"]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="confirmed_casts must be a dict"):
            ar.auto_clean(frame, confirmed_casts=[("x", "int64")])

    def test_confirmed_casts_keys_must_be_strings(self):
        """confirmed_casts dict keys must be strings."""
        df = pd.DataFrame({"x": ["1"]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="confirmed_casts keys must be str"):
            ar.auto_clean(frame, confirmed_casts={1: "int64"})

    def test_confirmed_casts_values_must_be_strings(self):
        """confirmed_casts dict values must be strings."""
        df = pd.DataFrame({"x": ["1"]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="confirmed_casts values must be str"):
            ar.auto_clean(frame, confirmed_casts={"x": 123})


class TestAutoCleanModes:
    """Test safe and strict mode behavior."""

    def test_safe_mode_strips_whitespace(self):
        """Safe mode applies strip_whitespace to string columns."""
        df = pd.DataFrame({"name": [" alice ", " bob "]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        result_df = ar.to_pandas(result)
        assert result_df["name"].tolist() == ["alice", "bob"]

    def test_safe_mode_no_duplicates_removed(self):
        """Safe mode does not remove duplicates."""
        df = pd.DataFrame({"x": ["a", "a", "b"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        # Duplicates should remain in safe mode
        assert result.shape[0] == 3

    def test_strict_mode_requires_cast_confirmation(self):
        """Strict mode with numeric strings requires cast confirmation."""
        df = pd.DataFrame({"num": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="would apply type casts"):
            ar.auto_clean(frame, mode="strict")

    def test_strict_mode_requires_allow_lossy_flag(self):
        """Strict mode with allow_lossy_casts=False raises for casts."""
        df = pd.DataFrame({"num": ["10", "20"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="allow_lossy_casts=True"):
            ar.auto_clean(frame, mode="strict", allow_lossy_casts=False)

    def test_strict_mode_requires_confirmed_casts_dict(self):
        """Strict mode with allow_lossy_casts=True but no confirmed_casts raises."""
        df = pd.DataFrame({"num": ["1", "2"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="requires confirmed_casts"):
            ar.auto_clean(frame, mode="strict", allow_lossy_casts=True)

    def test_strict_mode_rejects_mismatched_casts(self):
        """confirmed_casts must match proposed mapping exactly."""
        df = pd.DataFrame({"num": ["1", "2"]})
        frame = ar.from_pandas(df)
        report = ar.auto_clean(frame, mode="strict", dry_run=True)
        proposed = dict(report.suggestions).get("cast_types", {})
        
        if proposed:
            wrong_casts = {k: "float64" for k in proposed}
            with pytest.raises(ValueError, match="must match the proposed cast mapping exactly"):
                ar.auto_clean(
                    frame,
                    mode="strict",
                    allow_lossy_casts=True,
                    confirmed_casts=wrong_casts,
                )

    def test_strict_mode_accepts_correct_casts(self):
        """Strict mode works with correct confirmed_casts."""
        df = pd.DataFrame({"num": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        report = ar.auto_clean(frame, mode="strict", dry_run=True)
        casts = dict(report.suggestions).get("cast_types", {})
        
        if casts:
            result = ar.auto_clean(
                frame,
                mode="strict",
                allow_lossy_casts=True,
                confirmed_casts=casts,
            )
            assert isinstance(result, ar.ArFrame)

    def test_strict_mode_removes_duplicates(self):
        """Strict mode removes exact duplicate rows."""
        df = pd.DataFrame({"x": ["a", "a", "b"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="strict")
        assert result.shape[0] < frame.shape[0]


class TestAutoCleanFinalDedup:
    """Test final deduplication in strict mode."""

    def test_strict_final_dedup_after_strip(self):
        """Strict mode dedupes rows made identical by stripping."""
        df = pd.DataFrame({"x": [" a", "a"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="strict")
        # After strip, " a" becomes "a", so rows are identical and deduped
        assert result.shape[0] == 1
        result_df = ar.to_pandas(result)
        assert result_df["x"].iloc[0] == "a"


class TestAutoCleanExplanation:
    """Test CleanExplanation output and metadata."""

    def test_explanation_mode_field(self):
        """CleanExplanation.mode matches the input mode."""
        df = pd.DataFrame({"x": [" a"]})
        frame = ar.from_pandas(df)
        _, expl = ar.auto_clean(frame, mode="safe", explain=True)
        assert expl.mode == "safe"

    def test_explanation_rows_tracking(self):
        """CleanExplanation tracks before/after row counts."""
        df = pd.DataFrame({"x": ["a", "a"]})
        frame = ar.from_pandas(df)
        _, expl = ar.auto_clean(frame, mode="strict", explain=True)
        assert expl.rows_before == 2
        assert expl.rows_removed == expl.rows_before - expl.rows_after
        assert expl.rows_removed > 0

    def test_explanation_steps_list(self):
        """CleanExplanation.steps contains applied cleaning steps."""
        df = pd.DataFrame({"x": [" a"]})
        frame = ar.from_pandas(df)
        _, expl = ar.auto_clean(frame, mode="safe", explain=True)
        assert isinstance(expl.steps, list)


class TestAutoCleanEdgeCases:
    """Test edge cases and special scenarios."""

    def test_no_modifications_needed(self):
        """Frame with no issues returns valid result."""
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        assert isinstance(result, ar.ArFrame)
        result_df = ar.to_pandas(result)
        assert result_df["x"].tolist() == ["a", "b", "c"]

    def test_safe_mode_ignores_cast_flags(self):
        """Safe mode does not apply casts regardless of flags."""
        df = pd.DataFrame({"x": [" a"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe", allow_lossy_casts=False)
        assert isinstance(result, ar.ArFrame)

    def test_confirmed_casts_ignored_without_casts(self):
        """confirmed_casts ignored if no casts are suggested."""
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        # Non-numeric strings, so no casts suggested
        result = ar.auto_clean(frame, mode="strict", confirmed_casts={"x": "int64"})
        assert isinstance(result, ar.ArFrame)

    def test_report_describes_original_frame(self):
        """Returned report describes original frame before cleaning."""
        df = pd.DataFrame({"x": [" a", " a"]})
        frame = ar.from_pandas(df)
        result_frame, report = ar.auto_clean(frame, mode="strict", return_report=True)
        # Report row count is original
        assert report.row_count == 2
        # Result frame is cleaned
        assert result_frame.shape[0] <= 2

    def test_dry_run_does_not_modify(self):
        """dry_run=True returns report without modifying input."""
        df = pd.DataFrame({"x": [" a", " a"]})
        frame = ar.from_pandas(df)
        original_rows = frame.shape[0]
        report = ar.auto_clean(frame, mode="strict", dry_run=True)
        assert isinstance(report, DataQualityReport)
        # Input frame unchanged
        assert frame.shape[0] == original_rows
