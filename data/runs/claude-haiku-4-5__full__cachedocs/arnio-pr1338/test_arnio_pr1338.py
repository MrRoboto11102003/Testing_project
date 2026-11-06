import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestAutoCleanBasicBehavior:
    """Test basic auto_clean functionality and return types."""

    @pytest.fixture
    def messy_frame(self):
        """Frame with whitespace and duplicates."""
        df = pd.DataFrame({
            "name": [" Alice ", " Bob ", " Alice "],
            "age": ["25", "30", "25"],
            "city": [" NYC ", " LA ", " NYC "],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def clean_frame(self):
        """Frame with no quality issues."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1],
        })
        return ar.from_pandas(df)

    def test_auto_clean_default_returns_arframe(self, messy_frame):
        """Default auto_clean returns ArFrame."""
        result = ar.auto_clean(messy_frame)
        assert isinstance(result, ar.ArFrame)
        assert not isinstance(result, tuple)

    def test_auto_clean_safe_strips_whitespace(self, messy_frame):
        """Safe mode (default) removes leading/trailing whitespace."""
        result = ar.auto_clean(messy_frame, mode="safe")
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Alice"
        assert df["city"].iloc[0] == "NYC"
        assert df["city"].iloc[1] == "LA"

    def test_auto_clean_preserves_frame_shape(self, clean_frame):
        """Auto_clean preserves frame structure on clean data."""
        result = ar.auto_clean(clean_frame, mode="safe")
        assert result.shape == clean_frame.shape
        assert result.columns == clean_frame.columns


class TestAutoCleanReturnTypes:
    """Test various parameter combinations and return types."""

    @pytest.fixture
    def test_frame(self):
        df = pd.DataFrame({
            "col": [" a ", " b ", " a "],
        })
        return ar.from_pandas(df)

    def test_dry_run_returns_report_not_frame(self, test_frame):
        """dry_run=True returns report when return_report not set."""
        result = ar.auto_clean(test_frame, dry_run=True)
        assert not isinstance(result, ar.ArFrame)
        # Report has these attributes
        assert hasattr(result, 'row_count') or hasattr(result, 'columns')

    def test_return_report_true_returns_tuple(self, test_frame):
        """return_report=True returns (ArFrame, DataQualityReport)."""
        result = ar.auto_clean(test_frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)

    def test_explain_true_returns_tuple(self, test_frame):
        """explain=True returns (ArFrame, CleanExplanation)."""
        result = ar.auto_clean(test_frame, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)

    def test_report_and_explain_returns_triple(self, test_frame):
        """return_report=True and explain=True returns 3-tuple."""
        result = ar.auto_clean(test_frame, return_report=True, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], ar.ArFrame)


class TestAutoCleanModes:
    """Test safe vs strict mode behavior."""

    @pytest.fixture
    def frame_with_dupes(self):
        df = pd.DataFrame({
            "id": [1, 1, 2, 3],
            "value": ["x", "x", "y", "z"],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def frame_with_nulls(self):
        df = pd.DataFrame({
            "col": [1, None, 3],
        })
        return ar.from_pandas(df)

    def test_safe_mode_does_not_deduplicate(self, frame_with_dupes):
        """Safe mode preserves duplicate rows."""
        result = ar.auto_clean(frame_with_dupes, mode="safe")
        # Safe mode only strips whitespace, doesn't drop duplicates
        assert len(result) == len(frame_with_dupes)

    def test_strict_mode_can_reduce_rows(self, frame_with_dupes):
        """Strict mode may reduce rows through deduplication."""
        # Get suggested casts first
        report = ar.auto_clean(frame_with_dupes, mode="strict", dry_run=True)
        suggestions_dict = dict(report.suggestions) if report.suggestions else {}
        casts = suggestions_dict.get("cast_types", {})
        
        result = ar.auto_clean(frame_with_dupes, mode="strict",
                             allow_lossy_casts=True,
                             confirmed_casts=casts)
        # Strict mode includes drop_duplicates, so rows may be <= original
        assert len(result) <= len(frame_with_dupes)

    def test_nulls_preserved_in_safe_mode(self, frame_with_nulls):
        """Safe mode preserves null values by default."""
        result = ar.auto_clean(frame_with_nulls, mode="safe")
        # Safe mode doesn't drop nulls
        assert len(result) == len(frame_with_nulls)


class TestAutoCleanParameterCombinations:
    """Test parameter combinations and interactions."""

    @pytest.fixture
    def test_frame(self):
        df = pd.DataFrame({
            "x": [1, 1, 2],
            "y": [" a ", " a ", " b "],
        })
        return ar.from_pandas(df)

    def test_dry_run_does_not_modify_input(self, test_frame):
        """dry_run=True doesn't modify input frame."""
        original_shape = test_frame.shape
        ar.auto_clean(test_frame, dry_run=True)
        assert test_frame.shape == original_shape

    def test_strict_with_empty_confirmed_casts(self, test_frame):
        """Strict mode accepts empty confirmed_casts dict."""
        result = ar.auto_clean(test_frame, mode="strict",
                             allow_lossy_casts=True,
                             confirmed_casts={})
        assert isinstance(result, ar.ArFrame)

    def test_safe_mode_ignores_cast_parameters(self, test_frame):
        """Safe mode ignores allow_lossy_casts and confirmed_casts."""
        result = ar.auto_clean(test_frame, mode="safe",
                             allow_lossy_casts=True,
                             confirmed_casts={"x": "int64"})
        assert isinstance(result, ar.ArFrame)


class TestAutoCleanEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_frame(self):
        """auto_clean on empty frame."""
        df = pd.DataFrame({"col": []})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert len(result) == 0
        assert result.shape[1] == 1

    def test_single_row_frame(self):
        """auto_clean on single row frame."""
        df = pd.DataFrame({"col": [" value "]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert len(result) == 1

    def test_frame_all_nulls(self):
        """auto_clean on frame with all null values."""
        df = pd.DataFrame({
            "col1": [None, None],
            "col2": [None, None],
        })
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        assert result.shape == (2, 2)

    def test_frame_mixed_types(self):
        """auto_clean on frame with mixed column types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "str_col": [" a ", " b ", " c "],
            "float_col": [1.1, 2.2, 3.3],
        })
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame, mode="safe")
        assert result.shape == frame.shape

    def test_frame_single_column(self):
        """auto_clean on single column frame."""
        df = pd.DataFrame({"only_col": [1, 2, 2]})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert result.shape[1] == 1


class TestAutoCleanModeValidation:
    """Test mode parameter handling."""

    @pytest.fixture
    def test_frame(self):
        df = pd.DataFrame({"col": [1, 2]})
        return ar.from_pandas(df)

    def test_valid_modes(self, test_frame):
        """Both 'safe' and 'strict' modes work."""
        safe_result = ar.auto_clean(test_frame, mode="safe")
        strict_result = ar.auto_clean(test_frame, mode="strict",
                                     allow_lossy_casts=True,
                                     confirmed_casts={})
        assert isinstance(safe_result, ar.ArFrame)
        assert isinstance(strict_result, ar.ArFrame)
