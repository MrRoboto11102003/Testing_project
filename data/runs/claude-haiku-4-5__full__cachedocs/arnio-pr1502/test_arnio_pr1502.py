import pytest
import pandas as pd
import arnio as ar


class TestAutoCleanBasic:
    """Test basic auto_clean functionality."""

    def test_safe_mode_returns_frame(self):
        """Safe mode should return a cleaned ArFrame by default."""
        df = pd.DataFrame({
            "name": ["  alice  ", "  bob  "],
            "age": [25, 30]
        })
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        assert isinstance(result, ar.ArFrame)

    def test_safe_mode_strips_whitespace(self):
        """Safe mode should strip whitespace from strings."""
        df = pd.DataFrame({"name": ["  alice  ", "  bob  "]})
        frame = ar.from_pandas(df)
        clean = ar.auto_clean(frame, mode="safe")
        result_df = ar.to_pandas(clean)
        assert result_df["name"].iloc[0] == "alice"
        assert result_df["name"].iloc[1] == "bob"

    def test_empty_frame(self):
        """Should handle empty frames gracefully."""
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        frame = ar.from_pandas(df)
        clean = ar.auto_clean(frame, mode="safe")
        assert ar.to_pandas(clean).empty

    def test_no_changes_preserves_data(self):
        """Should preserve frames that need no cleaning."""
        df = pd.DataFrame({
            "name": ["alice", "bob"],
            "age": [25, 30]
        })
        frame = ar.from_pandas(df)
        clean = ar.auto_clean(frame, mode="safe")
        result = ar.to_pandas(clean)
        assert len(result) == 2


class TestAutoCleanReturnTypes:
    """Test different return type combinations."""

    def test_return_report_returns_tuple(self):
        """return_report=True should return (ArFrame, DataQualityReport)."""
        df = pd.DataFrame({"col": ["  a  ", "  b  "]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe", return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], ar.DataQualityReport)

    def test_dry_run_returns_report(self):
        """dry_run=True should return DataQualityReport without cleaning."""
        df = pd.DataFrame({"col": ["  a  ", "  b  "]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe", dry_run=True)
        assert isinstance(result, ar.DataQualityReport)

    def test_explain_returns_tuple(self):
        """explain=True should return (ArFrame, CleanExplanation)."""
        df = pd.DataFrame({"col": ["  a  ", "  b  "]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe", explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)

    def test_return_report_and_explain_returns_triple(self):
        """Both return_report and explain should return triple tuple."""
        df = pd.DataFrame({"col": ["  a  ", "  b  "]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe", return_report=True, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], ar.DataQualityReport)


class TestAutoCleanStrict:
    """Test strict mode behavior and cast confirmation."""

    def test_strict_mode_requires_confirmation(self):
        """Strict mode should require allow_lossy_casts or confirmed_casts."""
        df = pd.DataFrame({"id": ["1", "2"], "name": ["a", "b"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.auto_clean(frame, mode="strict")

    def test_strict_mode_with_confirmed_casts(self):
        """Strict mode should apply when confirmed_casts provided."""
        df = pd.DataFrame({"id": ["1", "2"], "name": ["a", "b"]})
        frame = ar.from_pandas(df)
        
        clean = ar.auto_clean(
            frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={}
        )
        assert isinstance(clean, ar.ArFrame)

    def test_strict_mode_removes_duplicates(self):
        """Strict mode should remove exact duplicate rows."""
        df = pd.DataFrame({
            "name": ["alice", "alice", "bob"],
            "age": [25, 25, 30]
        })
        frame = ar.from_pandas(df)
        clean = ar.auto_clean(
            frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={}
        )
        result = ar.to_pandas(clean)
        # Duplicate row should be removed
        assert len(result) < len(df)


class TestAutoCleanDryRun:
    """Test dry_run mode behavior."""

    def test_dry_run_does_not_modify_frame(self):
        """dry_run=True should not modify input frame."""
        df = pd.DataFrame({"col": ["  a  ", "  b  "]})
        frame = ar.from_pandas(df)
        original_shape = frame.shape
        
        ar.auto_clean(frame, mode="safe", dry_run=True)
        
        # Frame shape should be unchanged
        assert frame.shape == original_shape

    def test_dry_run_ignores_return_report(self):
        """dry_run=True should return report regardless of return_report."""
        df = pd.DataFrame({"col": ["a", "b"]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(
            frame, 
            mode="safe", 
            dry_run=True, 
            return_report=True
        )
        assert isinstance(result, ar.DataQualityReport)


class TestAutoCleanEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_mode_raises_value_error(self):
        """Invalid mode parameter should raise ValueError."""
        df = pd.DataFrame({"col": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.auto_clean(frame, mode="invalid_mode")

    def test_frame_with_nulls(self):
        """Should handle frames containing null values."""
        df = pd.DataFrame({
            "name": ["  alice  ", None, "  bob  "],
            "age": [25, 30, None]
        })
        frame = ar.from_pandas(df)
        clean = ar.auto_clean(frame, mode="safe")
        result = ar.to_pandas(clean)
        # Nulls should be preserved, only whitespace stripped
        assert result.isnull().any().any()

    def test_single_row_frame(self):
        """Should handle single-row frames."""
        df = pd.DataFrame({"col": ["  value  "]})
        frame = ar.from_pandas(df)
        clean = ar.auto_clean(frame, mode="safe")
        result = ar.to_pandas(clean)
        assert len(result) == 1
        assert result["col"].iloc[0] == "value"

    def test_multiple_whitespace_and_duplicates(self):
        """Should handle combined whitespace and duplicates in strict mode."""
        df = pd.DataFrame({
            "a": ["  x  ", "  x  ", "  y  "],
            "b": [1, 1, 2]
        })
        frame = ar.from_pandas(df)
        clean = ar.auto_clean(
            frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={}
        )
        result = ar.to_pandas(clean)
        # Should strip whitespace and remove one duplicate
        assert result["a"].iloc[0] == "x"
        assert len(result) < len(df)

    def test_confirmed_casts_empty_dict(self):
        """Empty confirmed_casts dict should be valid in strict mode."""
        df = pd.DataFrame({"col": ["a", "b"]})
        frame = ar.from_pandas(df)
        clean = ar.auto_clean(
            frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={}
        )
        assert isinstance(clean, ar.ArFrame)

    def test_mode_case_sensitive(self):
        """Mode parameter should be case-sensitive."""
        df = pd.DataFrame({"col": [1]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.auto_clean(frame, mode="Safe")
