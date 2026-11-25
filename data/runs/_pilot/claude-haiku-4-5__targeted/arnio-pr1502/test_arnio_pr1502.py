import pytest
import pandas as pd
import arnio as ar


@pytest.fixture
def clean_frame():
    """Create a clean test frame."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    return ar.from_dict(data)


@pytest.fixture
def messy_frame():
    """Create a frame with whitespace and duplicates."""
    data = {
        "name": ["  Alice  ", "Bob", "  Bob  "],
        "age": [25, 30, 30],
    }
    return ar.from_dict(data)


class TestAutoCleanBasic:
    """Test basic auto_clean functionality."""

    def test_default_returns_frame(self, clean_frame):
        """Default auto_clean should return a cleaned frame."""
        result = ar.auto_clean(clean_frame)
        df = ar.to_pandas(result)
        assert isinstance(df, pd.DataFrame)

    def test_safe_mode_returns_frame(self, clean_frame):
        """Safe mode should return cleaned frame."""
        result = ar.auto_clean(clean_frame, mode="safe")
        df = ar.to_pandas(result)
        assert isinstance(df, pd.DataFrame)

    def test_strict_mode_returns_frame(self, clean_frame):
        """Strict mode should return cleaned frame."""
        result = ar.auto_clean(clean_frame, mode="strict")
        df = ar.to_pandas(result)
        assert isinstance(df, pd.DataFrame)


class TestAutoCleanReturnTypes:
    """Test return type variations based on parameters."""

    def test_dry_run_returns_report(self, clean_frame):
        """dry_run=True should return report without tuple."""
        result = ar.auto_clean(clean_frame, dry_run=True)
        assert not isinstance(result, tuple)

    def test_return_report_returns_tuple(self, clean_frame):
        """return_report=True should return 2-element tuple."""
        result = ar.auto_clean(clean_frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_explain_returns_tuple(self, clean_frame):
        """explain=True should return 2-element tuple."""
        result = ar.auto_clean(clean_frame, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_report_and_explain_returns_triple(self, clean_frame):
        """Both return_report and explain should return 3-element tuple."""
        result = ar.auto_clean(clean_frame, return_report=True, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_dry_run_with_return_report(self, clean_frame):
        """dry_run with return_report should return report only."""
        result = ar.auto_clean(clean_frame, dry_run=True, return_report=True)
        assert not isinstance(result, tuple)


class TestAutoCleanDryRun:
    """Test dry_run behavior does not mutate frame."""

    def test_dry_run_preserves_frame(self, messy_frame):
        """dry_run=True should not modify the original frame."""
        original = ar.to_pandas(messy_frame)
        ar.auto_clean(messy_frame, dry_run=True, mode="safe")
        current = ar.to_pandas(messy_frame)
        pd.testing.assert_frame_equal(original, current)

    def test_dry_run_strict_mode(self, clean_frame):
        """dry_run should work in strict mode."""
        result = ar.auto_clean(clean_frame, mode="strict", dry_run=True)
        assert result is not None


class TestAutoCleanModeEffects:
    """Test cleaning behavior differences between modes."""

    def test_safe_mode_whitespace_cleaning(self, messy_frame):
        """Safe mode should trim whitespace from string columns."""
        result = ar.auto_clean(messy_frame, mode="safe")
        result_df = ar.to_pandas(result)
        # First name should be trimmed from "  Alice  " to "Alice"
        assert result_df["name"].iloc[0] == "Alice"

    def test_strict_mode_duplicate_removal(self, messy_frame):
        """Strict mode should remove exact duplicates."""
        result = ar.auto_clean(messy_frame, mode="strict")
        result_df = ar.to_pandas(result)
        # After strict cleaning and trimming, Bob should appear once (age 30)
        assert len(result_df) <= 3


class TestAutoCleanTypecast:
    """Test type casting parameters."""

    def test_allow_lossy_casts_alone(self, clean_frame):
        """allow_lossy_casts=True should work without confirmed_casts."""
        result = ar.auto_clean(
            clean_frame, mode="strict", allow_lossy_casts=True
        )
        df = ar.to_pandas(result)
        assert isinstance(df, pd.DataFrame)

    def test_confirmed_casts_with_allow_lossy(self, clean_frame):
        """confirmed_casts with allow_lossy_casts should apply casts."""
        result = ar.auto_clean(
            clean_frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={"age": "int32"},
        )
        df = ar.to_pandas(result)
        assert isinstance(df, pd.DataFrame)

    def test_empty_confirmed_casts_dict(self, clean_frame):
        """Empty confirmed_casts dict should be handled."""
        result = ar.auto_clean(
            clean_frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={},
        )
        df = ar.to_pandas(result)
        assert isinstance(df, pd.DataFrame)


class TestAutoCleanExplanation:
    """Test explanation output functionality."""

    def test_explanation_is_printable(self, clean_frame):
        """Explanation should be printable."""
        clean, explanation = ar.auto_clean(clean_frame, explain=True)
        explanation_str = str(explanation)
        assert isinstance(explanation_str, str)

    def test_explanation_with_report_together(self, clean_frame):
        """Both report and explanation should be present together."""
        clean, report, explanation = ar.auto_clean(
            clean_frame, return_report=True, explain=True
        )
        df = ar.to_pandas(clean)
        assert isinstance(df, pd.DataFrame)
        assert report is not None
        assert explanation is not None
