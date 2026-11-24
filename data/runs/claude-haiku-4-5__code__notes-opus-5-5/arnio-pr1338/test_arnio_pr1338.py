import pytest
import pandas as pd
import arnio as ar
from arnio.quality.auto_clean import CleanExplanation
import re


class TestAutoCleanValidation:
    """Test input validation and error handling."""
    
    def test_mode_invalid_value(self):
        """mode must be exactly 'safe' or 'strict'."""
        df = pd.DataFrame({"a": ["x", "y"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("mode must be 'safe' or 'strict'")):
            ar.auto_clean(frame, mode="fast")
    
    def test_mode_case_sensitive(self):
        """mode is case-sensitive."""
        df = pd.DataFrame({"a": ["x", "y"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("mode must be 'safe' or 'strict'")):
            ar.auto_clean(frame, mode="SAFE")
    
    def test_return_report_not_bool(self):
        """return_report must be bool, not int (0/1 raises TypeError)."""
        df = pd.DataFrame({"a": ["x", "y"]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="return_report must be a bool"):
            ar.auto_clean(frame, return_report=1)
    
    def test_bool_params_strict_type_checking(self):
        """dry_run, allow_lossy_casts, explain must be bool."""
        df = pd.DataFrame({"a": ["x", "y"]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.auto_clean(frame, dry_run=0)
    
    def test_dry_run_and_explain_conflict(self):
        """dry_run and explain cannot both be True."""
        df = pd.DataFrame({"a": ["x", "y"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("explain=True cannot be used with dry_run=True")):
            ar.auto_clean(frame, dry_run=True, explain=True)
    
    def test_dry_run_and_return_report_conflict(self):
        """dry_run and return_report cannot both be True."""
        df = pd.DataFrame({"a": ["x", "y"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("return_report=True cannot be used with dry_run=True")):
            ar.auto_clean(frame, dry_run=True, return_report=True)


class TestAutoCleanReturnTypes:
    """Test return value types for different flag combinations."""
    
    def test_default_returns_frame_only(self):
        """Default call returns just an ArFrame."""
        df = pd.DataFrame({"a": ["x", "y"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert not isinstance(result, tuple)
        assert hasattr(result, 'shape')
    
    def test_dry_run_returns_report_only(self):
        """dry_run=True returns DataQualityReport, not frame."""
        df = pd.DataFrame({"a": [" x", "y"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, dry_run=True)
        assert hasattr(result, 'suggestions')
    
    def test_return_report_returns_tuple(self):
        """return_report=True returns (ArFrame, DataQualityReport)."""
        df = pd.DataFrame({"a": [" x", "y"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame_out, report = result
        assert hasattr(frame_out, 'shape')
        assert hasattr(report, 'suggestions')
    
    def test_explain_returns_tuple_with_explanation(self):
        """explain=True returns (ArFrame, CleanExplanation)."""
        df = pd.DataFrame({"a": [" x", "y"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame_out, explanation = result
        assert isinstance(explanation, CleanExplanation)
    
    def test_both_return_report_and_explain_returns_triple(self):
        """Both flags True returns (ArFrame, DataQualityReport, CleanExplanation)."""
        df = pd.DataFrame({"a": [" x", "y"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, return_report=True, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        frame_out, report, explanation = result
        assert isinstance(explanation, CleanExplanation)


class TestAutoCleanSafeMode:
    """Test safe mode behavior (whitespace stripping only)."""
    
    def test_safe_mode_strips_whitespace(self):
        """Safe mode applies strip_whitespace."""
        df = pd.DataFrame({"name": [" alice ", "bob", " carol"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        result_df = ar.to_pandas(result)
        assert result_df["name"].tolist() == ["alice", "bob", "carol"]
    
    def test_safe_mode_no_type_casting(self):
        """Safe mode does not apply type casts, even if suggested."""
        df = pd.DataFrame({"n": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        assert result.dtypes["n"] == "string"


class TestAutoCleanStrictMode:
    """Test strict mode behavior (casts and duplicates)."""
    
    def test_strict_mode_requires_allow_lossy_casts_flag(self):
        """Strict mode with numeric cast requires allow_lossy_casts=True."""
        df = pd.DataFrame({"n": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="allow_lossy_casts=True"):
            ar.auto_clean(frame, mode="strict")
    
    def test_strict_mode_requires_confirmed_casts_dict(self):
        """Strict mode with numeric cast requires confirmed_casts dict."""
        df = pd.DataFrame({"n": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="confirmed_casts"):
            ar.auto_clean(frame, mode="strict", allow_lossy_casts=True, confirmed_casts=None)
    
    def test_strict_mode_casts_must_match_proposed_exactly(self):
        """Strict mode confirmed_casts must match proposed mapping exactly."""
        df = pd.DataFrame({"n": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        wrong_casts = {"n": "float64"}
        with pytest.raises(ValueError, match="confirmed_casts must match the proposed cast mapping exactly"):
            ar.auto_clean(frame, mode="strict", allow_lossy_casts=True, confirmed_casts=wrong_casts)
    
    def test_strict_mode_applies_confirmed_casts(self):
        """Strict mode applies type casts when both flags set and mapping matches."""
        df = pd.DataFrame({"n": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        # Get proposed mapping via dry_run
        report = ar.auto_clean(frame, mode="strict", dry_run=True)
        casts = dict(report.suggestions).get("cast_types", {})
        
        if casts:  # Only if casts were suggested
            result = ar.auto_clean(
                frame,
                mode="strict",
                allow_lossy_casts=True,
                confirmed_casts=casts
            )
            # Verify cast was applied
            assert result.dtypes["n"] in ("int64", "float64")


class TestAutoCleanExplanation:
    """Test CleanExplanation structure and content."""
    
    def test_clean_explanation_has_required_fields(self):
        """CleanExplanation has mode, row counts, and steps."""
        df = pd.DataFrame({"a": [" x", " y"]})
        frame = ar.from_pandas(df)
        result, explanation = ar.auto_clean(frame, explain=True)
        assert explanation.mode in ("safe", "strict")
        assert explanation.rows_before >= 0
        assert explanation.rows_after >= 0
        assert explanation.rows_removed == explanation.rows_before - explanation.rows_after
        assert isinstance(explanation.steps, list)
    
    def test_clean_explanation_records_applied_steps(self):
        """CleanExplanation.steps contains applied cleaning steps."""
        df = pd.DataFrame({"a": [" x", " y"]})
        frame = ar.from_pandas(df)
        result, explanation = ar.auto_clean(frame, explain=True)
        # With whitespace data in safe mode, should have at least strip_whitespace
        assert len(explanation.steps) >= 1
        assert explanation.steps[0].step == "strip_whitespace"
        assert hasattr(explanation.steps[0], 'rows_before')
        assert hasattr(explanation.steps[0], 'rows_after')


class TestAutoCleanEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_frame_unchanged(self):
        """Empty frame returns with no modifications."""
        df = pd.DataFrame({"a": []})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert result.shape[0] == 0
    
    def test_no_cleaning_returns_same_frame_in_safe_mode(self):
        """When no cleaning applies, safe mode may return input object."""
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        # No whitespace to strip, so may be same object
        assert result.shape == frame.shape
