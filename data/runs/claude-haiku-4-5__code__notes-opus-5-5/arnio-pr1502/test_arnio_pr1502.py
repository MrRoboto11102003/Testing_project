import pytest
import pandas as pd
import arnio as ar
import re


class TestAutoCleanBasicReturns:
    """Test return types based on parameter combinations."""
    
    def test_default_returns_frame(self):
        """Test that with no flags, returns ArFrame."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        f = ar.from_pandas(df)
        result = ar.auto_clean(f)
        assert hasattr(result, 'dtypes')
    
    def test_dry_run_returns_report(self):
        """Test that dry_run=True returns DataQualityReport."""
        df = pd.DataFrame({"x": ["1"]})
        f = ar.from_pandas(df)
        result = ar.auto_clean(f, dry_run=True)
        assert hasattr(result, 'suggestions')
    
    def test_return_report_returns_tuple(self):
        """Test that return_report=True returns (frame, report)."""
        df = pd.DataFrame({"x": [1]})
        f = ar.from_pandas(df)
        result = ar.auto_clean(f, return_report=True)
        assert isinstance(result, tuple) and len(result) == 2
    
    def test_explain_returns_tuple(self):
        """Test that explain=True returns (frame, explanation)."""
        df = pd.DataFrame({"x": [1]})
        f = ar.from_pandas(df)
        result = ar.auto_clean(f, explain=True)
        assert isinstance(result, tuple) and len(result) == 2
        assert hasattr(result[1], 'steps')
    
    def test_return_report_and_explain_returns_triple(self):
        """Test that both flags return (frame, report, explanation)."""
        df = pd.DataFrame({"x": [1]})
        f = ar.from_pandas(df)
        result = ar.auto_clean(f, return_report=True, explain=True)
        assert isinstance(result, tuple) and len(result) == 3


class TestAutoCleanParameterValidation:
    """Test parameter validation and error handling."""
    
    def test_mode_invalid_raises_valueerror(self):
        """Test that invalid mode string raises ValueError."""
        df = pd.DataFrame({"x": [1]})
        f = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("mode must be 'safe' or 'strict'")):
            ar.auto_clean(f, mode="fast")
    
    def test_mode_case_sensitive(self):
        """Test that mode is case-sensitive."""
        df = pd.DataFrame({"x": [1]})
        f = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("mode must be 'safe' or 'strict'")):
            ar.auto_clean(f, mode="SAFE")
    
    def test_return_report_requires_bool(self):
        """Test that return_report must be bool, not int."""
        df = pd.DataFrame({"x": [1]})
        f = ar.from_pandas(df)
        with pytest.raises(TypeError, match=re.escape("return_report must be a bool")):
            ar.auto_clean(f, return_report=1)
    
    def test_dry_run_requires_bool(self):
        """Test that dry_run must be bool, not int."""
        df = pd.DataFrame({"x": [1]})
        f = ar.from_pandas(df)
        with pytest.raises(TypeError, match=re.escape("dry_run must be a bool")):
            ar.auto_clean(f, dry_run=0)
    
    def test_allow_lossy_casts_requires_bool(self):
        """Test that allow_lossy_casts must be bool, not int."""
        df = pd.DataFrame({"x": [1]})
        f = ar.from_pandas(df)
        with pytest.raises(TypeError, match=re.escape("allow_lossy_casts must be a bool")):
            ar.auto_clean(f, allow_lossy_casts=1)
    
    def test_explain_requires_bool(self):
        """Test that explain must be bool, not int."""
        df = pd.DataFrame({"x": [1]})
        f = ar.from_pandas(df)
        with pytest.raises(TypeError, match=re.escape("explain must be a bool")):
            ar.auto_clean(f, explain=0)
    
    def test_confirmed_casts_validates_dict_type(self):
        """Test that confirmed_casts must be dict or None."""
        df = pd.DataFrame({"x": ["1"]})
        f = ar.from_pandas(df)
        with pytest.raises(TypeError, match=re.escape("confirmed_casts must be a dict")):
            ar.auto_clean(f, confirmed_casts=[("x", "int64")])
    
    def test_dry_run_explain_conflict(self):
        """Test that dry_run and explain cannot both be True."""
        df = pd.DataFrame({"x": [1]})
        f = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("explain=True cannot be used with dry_run=True")):
            ar.auto_clean(f, dry_run=True, explain=True)
    
    def test_dry_run_return_report_conflict(self):
        """Test that dry_run and return_report cannot both be True."""
        df = pd.DataFrame({"x": [1]})
        f = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("return_report=True cannot be used with dry_run=True")):
            ar.auto_clean(f, dry_run=True, return_report=True)


class TestAutoCleanSafeModeFunction:
    """Test safe mode functionality."""
    
    def test_safe_mode_strips_whitespace(self):
        """Test that safe mode strips leading and trailing whitespace."""
        df = pd.DataFrame({"name": [" alice", "bob ", " carol "]})
        f = ar.from_pandas(df)
        result = ar.auto_clean(f, mode="safe")
        values = ar.to_pandas(result)["name"].tolist()
        assert values == ["alice", "bob", "carol"]
    
    def test_safe_mode_default(self):
        """Test that safe is the default mode."""
        df = pd.DataFrame({"name": [" alice", "bob"]})
        f = ar.from_pandas(df)
        result = ar.auto_clean(f)
        values = ar.to_pandas(result)["name"].tolist()
        assert values == ["alice", "bob"]
    
    def test_safe_mode_preserves_duplicates(self):
        """Test that safe mode does not remove exact duplicates."""
        df = pd.DataFrame({"x": ["a", "a", "b"]})
        f = ar.from_pandas(df)
        result = ar.auto_clean(f, mode="safe")
        assert ar.to_pandas(result).shape[0] == 3


class TestAutoCleanStrictMode:
    """Test strict mode functionality."""
    
    def test_strict_requires_allow_lossy_casts_for_casts(self):
        """Test strict mode with cast suggestion requires allow_lossy_casts."""
        df = pd.DataFrame({"num": ["1", "2", "3"]})
        f = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("auto_clean(mode='strict') would apply type casts")):
            ar.auto_clean(f, mode="strict")
    
    def test_strict_requires_confirmed_casts(self):
        """Test strict mode requires confirmed_casts when allow_lossy_casts=True."""
        df = pd.DataFrame({"num": ["1", "2", "3"]})
        f = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("auto_clean(mode='strict') requires confirmed_casts")):
            ar.auto_clean(f, mode="strict", allow_lossy_casts=True)
    
    def test_strict_requires_exact_cast_mapping(self):
        """Test that confirmed_casts must match proposed mapping exactly."""
        df = pd.DataFrame({"num": ["1", "2", "3"]})
        f = ar.from_pandas(df)
        # Try with wrong mapping
        with pytest.raises(ValueError, match=re.escape("confirmed_casts must match the proposed cast mapping exactly")):
            ar.auto_clean(f, mode="strict", allow_lossy_casts=True, confirmed_casts={"num": "float64"})
    
    def test_strict_with_valid_confirmed_casts(self):
        """Test strict mode succeeds with valid confirmed_casts."""
        df = pd.DataFrame({"num": ["1", "2", "3"]})
        f = ar.from_pandas(df)
        # Get proposed mapping via dry_run
        report = ar.auto_clean(f, mode="strict", dry_run=True)
        
        # Extract casts from suggestions
        casts = None
        for suggestion in report.suggestions:
            if hasattr(suggestion, 'step') and suggestion.step == "cast_types":
                casts = suggestion.kwargs
                break
            elif isinstance(suggestion, tuple) and suggestion[0] == "cast_types":
                casts = suggestion[1]
                break
        
        if casts:
            result = ar.auto_clean(f, mode="strict", allow_lossy_casts=True, confirmed_casts=casts)
            assert result is not None


class TestAutoCleanDryRun:
    """Test dry_run behavior."""
    
    def test_dry_run_does_not_modify(self):
        """Test that dry_run does not modify the input frame."""
        df = pd.DataFrame({"name": [" alice", " bob"]})
        f = ar.from_pandas(df)
        original_values = ar.to_pandas(f)["name"].tolist()
        
        ar.auto_clean(f, dry_run=True)
        
        # Original frame should be unchanged
        assert ar.to_pandas(f)["name"].tolist() == original_values
    
    def test_dry_run_returns_report(self):
        """Test that dry_run returns a DataQualityReport."""
        df = pd.DataFrame({"name": [" alice"]})
        f = ar.from_pandas(df)
        result = ar.auto_clean(f, dry_run=True)
        assert hasattr(result, 'suggestions')


class TestAutoCleanExplain:
    """Test explain functionality."""
    
    def test_explain_returns_explanation_object(self):
        """Test that explain returns CleanExplanation with expected fields."""
        df = pd.DataFrame({"name": [" alice"]})
        f = ar.from_pandas(df)
        frame, explanation = ar.auto_clean(f, explain=True)
        assert hasattr(explanation, 'steps')
        assert hasattr(explanation, 'mode')
        assert hasattr(explanation, 'rows_before')
        assert hasattr(explanation, 'rows_after')
        assert hasattr(explanation, 'rows_removed')
    
    def test_explain_mode_field(self):
        """Test that explanation.mode matches input mode."""
        df = pd.DataFrame({"x": [1]})
        f = ar.from_pandas(df)
        _, explanation = ar.auto_clean(f, mode="safe", explain=True)
        assert explanation.mode == "safe"
    
    def test_explain_row_counts(self):
        """Test that explanation tracks row counts correctly."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        f = ar.from_pandas(df)
        _, explanation = ar.auto_clean(f, explain=True)
        assert explanation.rows_before == 3
        assert explanation.rows_removed == explanation.rows_before - explanation.rows_after


class TestAutoCleanEdgeCases:
    """Test edge cases."""
    
    def test_zero_row_frame(self):
        """Test that zero-row frame is handled correctly."""
        df = pd.DataFrame({"x": pd.Series([], dtype="string")})
        f = ar.from_pandas(df)
        result = ar.auto_clean(f)
        assert result.shape[0] == 0
    
    def test_no_whitespace_to_strip(self):
        """Test frame with no whitespace returns unchanged."""
        df = pd.DataFrame({"x": ["alice", "bob"]})
        f = ar.from_pandas(df)
        result = ar.auto_clean(f)
        values = ar.to_pandas(result)["x"].tolist()
        assert values == ["alice", "bob"]
    
    def test_multiple_columns_with_whitespace(self):
        """Test whitespace stripping in multiple columns."""
        df = pd.DataFrame({
            "first": [" alice", "bob"],
            "last": ["smith ", " jones"]
        })
        f = ar.from_pandas(df)
        result = ar.auto_clean(f)
        result_df = ar.to_pandas(result)
        assert result_df["first"].tolist() == ["alice", "bob"]
        assert result_df["last"].tolist() == ["smith", "jones"]
