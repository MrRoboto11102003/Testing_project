import pytest
import pandas as pd
import arnio as ar


def make_frame(data):
    """Convert dict to ArFrame via pandas."""
    return ar.from_pandas(pd.DataFrame(data))


def to_pandas(frame):
    """Convert ArFrame to pandas DataFrame."""
    return ar.to_pandas(frame)


class TestAutoCleanDefaultBehavior:
    """Test default behavior."""
    
    def test_default_safe_mode_strips_whitespace(self):
        """Test that default (safe) mode strips whitespace."""
        frame = make_frame({
            "name": ["  Alice  ", "  Bob  "],
            "age": [25, 30],
        })
        result = ar.auto_clean(frame)
        df = to_pandas(result)
        assert df["name"].tolist() == ["Alice", "Bob"]


class TestAutoCleanSafeMode:
    """Test safe mode behavior."""
    
    def test_safe_mode_explicit(self):
        """Test explicit safe mode parameter."""
        frame = make_frame({"col": ["  test  "]})
        result = ar.auto_clean(frame, mode="safe")
        assert isinstance(result, ar.ArFrame)
    
    def test_safe_mode_does_not_remove_duplicates(self):
        """Test that safe mode preserves duplicates."""
        frame = make_frame({"id": [1, 1, 2]})
        result = ar.auto_clean(frame, mode="safe")
        assert result.shape[0] == 3


class TestAutoCleanStrictMode:
    """Test strict mode behavior and requirements."""
    
    def test_strict_requires_allow_lossy_casts(self):
        """Test that strict mode without allow_lossy_casts raises ValueError."""
        frame = make_frame({"id": [1, 2, 3]})
        with pytest.raises(ValueError, match="allow_lossy_casts=True"):
            ar.auto_clean(frame, mode="strict")
    
    def test_strict_requires_confirmed_casts(self):
        """Test that strict mode requires confirmed_casts when allow_lossy_casts=True."""
        frame = make_frame({"id": [1, 2, 3]})
        with pytest.raises(ValueError, match="confirmed_casts"):
            ar.auto_clean(frame, mode="strict", allow_lossy_casts=True)
    
    def test_strict_confirmed_casts_must_match_exactly(self):
        """Test that confirmed_casts must exactly match proposed mapping."""
        frame = make_frame({"id": [1, 2, 3]})
        with pytest.raises(ValueError, match="confirmed_casts must match"):
            ar.auto_clean(
                frame,
                mode="strict",
                allow_lossy_casts=True,
                confirmed_casts={"wrong_col": "int64"},
            )


class TestAutoCleanDryRun:
    """Test dry_run behavior."""
    
    def test_dry_run_returns_report_only(self):
        """Test that dry_run=True returns only DataQualityReport."""
        frame = make_frame({"col": ["  test  "]})
        result = ar.auto_clean(frame, dry_run=True)
        assert isinstance(result, ar.DataQualityReport)
    
    def test_dry_run_incompatible_with_return_report(self):
        """Test that dry_run and return_report together raise error."""
        frame = make_frame({"col": [1, 2]})
        with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
            ar.auto_clean(frame, dry_run=True, return_report=True)
    
    def test_dry_run_incompatible_with_explain(self):
        """Test that dry_run and explain together raise error."""
        frame = make_frame({"col": [1, 2]})
        with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
            ar.auto_clean(frame, dry_run=True, explain=True)


class TestAutoCleanReturnReport:
    """Test return_report parameter."""
    
    def test_return_report_true_returns_tuple(self):
        """Test that return_report=True returns (frame, report) tuple."""
        frame = make_frame({"col": [1, 2]})
        result = ar.auto_clean(frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], ar.DataQualityReport)
    
    def test_return_report_frame_is_cleaned(self):
        """Test that frame in tuple is actually cleaned."""
        frame = make_frame({"name": ["  test  "]})
        cleaned_frame, _ = ar.auto_clean(frame, return_report=True)
        df = to_pandas(cleaned_frame)
        assert df["name"].iloc[0] == "test"


class TestAutoCleanExplain:
    """Test explain parameter."""
    
    def test_explain_true_returns_tuple(self):
        """Test that explain=True returns (frame, explanation) tuple."""
        frame = make_frame({"col": [1, 2]})
        result = ar.auto_clean(frame, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert hasattr(result[1], 'mode')
        assert hasattr(result[1], 'steps')
    
    def test_explain_and_return_report_returns_triple(self):
        """Test that both explain and return_report return 3-tuple."""
        frame = make_frame({"col": [1, 2]})
        result = ar.auto_clean(frame, explain=True, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], ar.DataQualityReport)
        assert hasattr(result[2], 'mode')


class TestAutoCleanParameterValidation:
    """Test parameter type validation."""
    
    def test_invalid_mode_raises_value_error(self):
        """Test that invalid mode value raises ValueError."""
        frame = make_frame({"col": [1, 2]})
        with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
            ar.auto_clean(frame, mode="invalid")
    
    def test_return_report_must_be_bool(self):
        """Test that return_report must be boolean."""
        frame = make_frame({"col": [1, 2]})
        with pytest.raises(TypeError, match="return_report must be a bool"):
            ar.auto_clean(frame, return_report="yes")
    
    def test_explain_must_be_bool(self):
        """Test that explain must be boolean."""
        frame = make_frame({"col": [1, 2]})
        with pytest.raises(TypeError, match="explain must be a bool"):
            ar.auto_clean(frame, explain=[])


class TestAutoCleanEdgeCases:
    """Test edge cases and unusual inputs."""
    
    def test_empty_frame(self):
        """Test auto_clean on empty frame."""
        frame = make_frame({"col": []})
        result = ar.auto_clean(frame)
        assert result.shape[0] == 0
    
    def test_single_row_frame(self):
        """Test auto_clean on single-row frame."""
        frame = make_frame({"col": [1]})
        result = ar.auto_clean(frame)
        assert result.shape[0] == 1
    
    def test_frame_needing_no_cleaning(self):
        """Test frame with no whitespace or other issues in safe mode."""
        frame = make_frame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        result = ar.auto_clean(frame, mode="safe")
        assert result.shape[0] == 3
        df = to_pandas(result)
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]
