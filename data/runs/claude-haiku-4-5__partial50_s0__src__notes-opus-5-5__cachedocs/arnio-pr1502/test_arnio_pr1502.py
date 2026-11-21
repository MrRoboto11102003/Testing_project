import pytest
import pandas as pd
import arnio as ar
import re


class TestAutoCleanBasic:
    """Basic functionality and return types."""

    def test_safe_mode_strips_whitespace_default(self):
        """Safe mode (default) strips whitespace from string columns."""
        df = pd.DataFrame({"name": [" alice", "bob ", " carol "]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        
        result_df = ar.to_pandas(result)
        assert result_df["name"].tolist() == ["alice", "bob", "carol"]

    def test_safe_mode_explicit(self):
        """Safe mode explicitly specified."""
        df = pd.DataFrame({"name": [" alice "]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        
        result_df = ar.to_pandas(result)
        assert result_df["name"].iloc[0] == "alice"

    def test_dry_run_returns_report_only(self):
        """dry_run=True returns DataQualityReport without modifications."""
        df = pd.DataFrame({"name": [" alice"]})
        frame = ar.from_pandas(df)
        
        report = ar.auto_clean(frame, dry_run=True)
        
        assert hasattr(report, 'summary')
        assert hasattr(report, 'suggestions')
        # Original unchanged
        assert ar.to_pandas(frame)["name"].iloc[0] == " alice"

    def test_return_report_with_cleaned_frame(self):
        """return_report=True returns (ArFrame, DataQualityReport)."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        
        result = ar.auto_clean(frame, return_report=True)
        
        assert isinstance(result, tuple) and len(result) == 2
        cleaned_frame, report = result
        assert hasattr(report, 'summary')

    def test_explain_returns_explanation(self):
        """explain=True returns (ArFrame, CleanExplanation)."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        
        result = ar.auto_clean(frame, explain=True)
        
        assert isinstance(result, tuple) and len(result) == 2
        _, explanation = result
        assert hasattr(explanation, 'mode')
        assert hasattr(explanation, 'steps')

    def test_all_flags_return_triple(self):
        """return_report + explain returns (ArFrame, Report, Explanation)."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        
        result = ar.auto_clean(frame, return_report=True, explain=True)
        
        assert isinstance(result, tuple) and len(result) == 3


class TestAutoCleanValidation:
    """Input validation in documented order."""

    def test_invalid_mode_string(self):
        """Invalid mode raises ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        with pytest.raises(ValueError, match=re.escape("mode must be 'safe' or 'strict'")):
            ar.auto_clean(frame, mode="invalid")

    def test_mode_case_sensitive(self):
        """Mode is case-sensitive."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        with pytest.raises(ValueError, match=re.escape("mode must be 'safe' or 'strict'")):
            ar.auto_clean(frame, mode="SAFE")

    def test_return_report_type_check(self):
        """return_report must be bool, not int."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        with pytest.raises(TypeError, match=re.escape("return_report must be a bool")):
            ar.auto_clean(frame, return_report=1)

    def test_dry_run_type_check(self):
        """dry_run must be bool."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        with pytest.raises(TypeError, match=re.escape("dry_run must be a bool")):
            ar.auto_clean(frame, dry_run=0)

    def test_allow_lossy_casts_type_check(self):
        """allow_lossy_casts must be bool."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        with pytest.raises(TypeError, match=re.escape("allow_lossy_casts must be a bool")):
            ar.auto_clean(frame, allow_lossy_casts="yes")

    def test_explain_type_check(self):
        """explain must be bool."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        with pytest.raises(TypeError, match=re.escape("explain must be a bool")):
            ar.auto_clean(frame, explain=1)

    def test_dry_run_and_explain_conflict(self):
        """dry_run and explain cannot both be True."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        with pytest.raises(ValueError, match=re.escape("explain=True cannot be used with dry_run=True")):
            ar.auto_clean(frame, dry_run=True, explain=True)

    def test_dry_run_and_return_report_conflict(self):
        """dry_run and return_report cannot both be True."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        with pytest.raises(ValueError, match=re.escape("return_report=True cannot be used with dry_run=True")):
            ar.auto_clean(frame, dry_run=True, return_report=True)


class TestAutoCleanStrictMode:
    """Strict mode casting and deduplication behavior."""

    def test_strict_mode_removes_exact_duplicates(self):
        """Strict mode removes exact duplicate rows."""
        df = pd.DataFrame({"name": ["alice", "alice"]})
        frame = ar.from_pandas(df)
        
        result = ar.auto_clean(frame, mode="strict")
        
        assert result.shape[0] == 1

    def test_strict_mode_whitespace_creates_duplicates_then_dedups(self):
        """Strict mode: whitespace stripping can create duplicates that are then removed."""
        df = pd.DataFrame({"name": ["alice", " alice"]})
        frame = ar.from_pandas(df)
        
        result = ar.auto_clean(frame, mode="strict")
        
        assert result.shape[0] == 1
        assert ar.to_pandas(result)["name"].iloc[0] == "alice"

    def test_strict_mode_requires_cast_confirmation(self):
        """Strict mode with numeric string column requires cast confirmation."""
        df = pd.DataFrame({"score": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="auto_clean\\(mode='strict'\\) would apply type casts"):
            ar.auto_clean(frame, mode="strict")

    def test_strict_mode_allow_lossy_without_confirmed_mapping_fails(self):
        """Strict mode: allow_lossy_casts=True alone is insufficient."""
        df = pd.DataFrame({"score": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="requires confirmed_casts"):
            ar.auto_clean(frame, mode="strict", allow_lossy_casts=True)

    def test_strict_mode_confirmed_casts_must_match_exactly(self):
        """Confirmed casts must exactly match the proposed mapping."""
        df = pd.DataFrame({"score": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        
        # This will fail because confirmed_casts doesn't match proposed
        with pytest.raises(ValueError, match="must match the proposed cast mapping exactly"):
            ar.auto_clean(
                frame,
                mode="strict",
                allow_lossy_casts=True,
                confirmed_casts={"score": "float64"}  # Likely proposed is int64
            )

    def test_strict_mode_safe_workflow_with_dry_run(self):
        """Recommended workflow: dry_run to get mapping, then apply with confirmation."""
        df = pd.DataFrame({"score": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        
        # Step 1: get proposed mapping via dry_run
        report = ar.auto_clean(frame, mode="strict", dry_run=True)
        suggestions = dict(report.suggestions)
        casts = suggestions.get("cast_types", {})
        
        # Step 2: apply with exact confirmation
        if casts:
            result = ar.auto_clean(
                frame,
                mode="strict",
                allow_lossy_casts=True,
                confirmed_casts=casts
            )
            assert result.dtypes["score"] == "int64"


class TestAutoCleanExplanation:
    """CleanExplanation object fields and content."""

    def test_explanation_mode_field(self):
        """CleanExplanation.mode reflects the mode parameter."""
        df = pd.DataFrame({"x": [1]})
        frame = ar.from_pandas(df)
        
        _, explanation = ar.auto_clean(frame, mode="safe", explain=True)
        assert explanation.mode == "safe"

    def test_explanation_row_counts(self):
        """CleanExplanation records before/after row counts."""
        df = pd.DataFrame({"name": [" a", " a"]})
        frame = ar.from_pandas(df)
        
        _, explanation = ar.auto_clean(frame, mode="strict", explain=True)
        
        assert explanation.rows_before == 2
        assert explanation.rows_after == 1
        assert explanation.rows_removed == 1

    def test_explanation_steps_recorded(self):
        """CleanExplanation.steps lists applied steps with metadata."""
        df = pd.DataFrame({"name": [" alice", " alice"]})
        frame = ar.from_pandas(df)
        
        _, explanation = ar.auto_clean(frame, mode="strict", explain=True)
        
        assert len(explanation.steps) >= 1
        for step_record in explanation.steps:
            assert hasattr(step_record, 'step')
            assert hasattr(step_record, 'kwargs')
            assert hasattr(step_record, 'rows_before')
            assert hasattr(step_record, 'rows_after')
            assert hasattr(step_record, 'reason')
            assert isinstance(step_record.reason, str)

    def test_explanation_identity_when_no_changes(self):
        """When no changes apply, the input frame object is returned unchanged."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        result = ar.auto_clean(frame)
        
        # Same object when no changes
        assert result is frame


class TestAutoCleanEdgeCases:
    """Edge cases and special scenarios."""

    def test_safe_mode_no_deduplication(self):
        """Safe mode does not deduplicate."""
        df = pd.DataFrame({"x": ["a", "a"]})
        frame = ar.from_pandas(df)
        
        result = ar.auto_clean(frame, mode="safe")
        
        assert result.shape[0] == 2

    def test_numeric_columns_no_stripping(self):
        """Numeric columns are not stripped for whitespace."""
        df = pd.DataFrame({"score": [1, 2, 3], "value": [1.5, 2.5, 3.5]})
        frame = ar.from_pandas(df)
        
        result = ar.auto_clean(frame, mode="strict")
        
        result_df = ar.to_pandas(result)
        assert result_df["score"].tolist() == [1, 2, 3]
        assert result_df["value"].tolist() == [1.5, 2.5, 3.5]

    def test_empty_dataframe(self):
        """Empty DataFrame is handled gracefully."""
        df = pd.DataFrame({"x": pd.Series([], dtype="object")})
        frame = ar.from_pandas(df)
        
        result = ar.auto_clean(frame, mode="strict")
        
        assert result.shape[0] == 0

    def test_report_describes_original_frame(self):
        """Report describes the original frame, not the result."""
        df = pd.DataFrame({"name": [" alice", " alice"]})
        frame = ar.from_pandas(df)
        
        result, report = ar.auto_clean(frame, mode="strict", return_report=True)
        
        # Report row count is from original
        assert report.row_count == 2
        # Result has fewer rows after dedup
        assert result.shape[0] == 1
