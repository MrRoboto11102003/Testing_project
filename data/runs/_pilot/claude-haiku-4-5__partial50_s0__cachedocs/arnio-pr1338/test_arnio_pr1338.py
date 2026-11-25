import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestAutoClean:
    """Unit tests for ar.auto_clean() function."""

    @pytest.fixture
    def messy_frame(self):
        """Frame with leading/trailing whitespace and duplicates."""
        df = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "name": [" Alice ", " Bob ", " Bob ", " Charlie "],
            "city": [" Paris ", "London", "London", " Tokyo "],
            "score": [95.5, 88.0, 88.0, 92.5],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def clean_frame(self):
        """Already clean frame with no whitespace or duplicates."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def numeric_string_frame(self):
        """Frame with numeric strings that could be cast."""
        df = pd.DataFrame({
            "id": ["1", "2", "3"],
            "age": ["25", "30", "35"],
            "active": ["true", "false", "true"],
        })
        return ar.from_pandas(df)

    def test_default_returns_arframe(self, messy_frame):
        """By default, auto_clean returns just an ArFrame."""
        result = ar.auto_clean(messy_frame)
        assert isinstance(result, ar.ArFrame)

    def test_safe_mode_strips_whitespace(self, messy_frame):
        """Safe mode applies strip_whitespace only."""
        result = ar.auto_clean(messy_frame, mode="safe")
        df = ar.to_pandas(result)
        # Names should be stripped of leading/trailing whitespace
        assert "Alice" in df["name"].values
        assert " Alice " not in df["name"].values
        assert "Bob" in df["city"].values
        # But duplicates should still exist
        assert len(df) == 4

    def test_safe_mode_preserves_duplicates(self, messy_frame):
        """Safe mode does not remove duplicate rows."""
        result = ar.auto_clean(messy_frame, mode="safe")
        df = ar.to_pandas(result)
        # Row count unchanged
        assert len(df) == 4

    def test_strict_mode_removes_duplicates(self, messy_frame):
        """Strict mode applies strip_whitespace + drop_duplicates."""
        result = ar.auto_clean(messy_frame, mode="strict")
        df = ar.to_pandas(result)
        # Should have fewer rows due to deduplication
        assert len(df) < 4
        # All rows should now be unique
        assert len(df) == df.drop_duplicates().shape[0]

    def test_return_report_true_returns_tuple(self, messy_frame):
        """When return_report=True, returns (ArFrame, DataQualityReport)."""
        result = ar.auto_clean(messy_frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, report = result
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)

    def test_dry_run_true_returns_report_only(self, messy_frame):
        """When dry_run=True, returns DataQualityReport only."""
        result = ar.auto_clean(messy_frame, dry_run=True)
        assert isinstance(result, ar.DataQualityReport)

    def test_explain_true_returns_tuple(self, messy_frame):
        """When explain=True, returns (ArFrame, explanation)."""
        result = ar.auto_clean(messy_frame, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, explanation = result
        assert isinstance(frame, ar.ArFrame)

    def test_return_report_and_explain_returns_triple(self, messy_frame):
        """When both return_report=True and explain=True, returns triple."""
        result = ar.auto_clean(
            messy_frame,
            return_report=True,
            explain=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        frame, report, explanation = result
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(report, ar.DataQualityReport)

    def test_works_on_already_clean_frame(self, clean_frame):
        """auto_clean handles already-clean data without error."""
        result = ar.auto_clean(clean_frame)
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        assert df.shape[0] == 3

    def test_works_on_empty_frame(self):
        """auto_clean handles empty frames gracefully."""
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        frame = ar.from_pandas(df)
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)

    def test_column_order_preserved(self, messy_frame):
        """Column order and names are preserved after cleaning."""
        original_cols = messy_frame.columns
        result = ar.auto_clean(messy_frame)
        assert list(result.columns) == list(original_cols)

    def test_confirmed_casts_workflow(self, numeric_string_frame):
        """Recommended workflow: dry_run then confirmed_casts."""
        # Step 1: dry run to get suggestions
        report = ar.auto_clean(numeric_string_frame, mode="strict", dry_run=True)
        assert isinstance(report, ar.DataQualityReport)

        # Step 2: apply with confirmed casts if needed
        result = ar.auto_clean(
            numeric_string_frame,
            mode="strict",
            allow_lossy_casts=False,
            confirmed_casts={"id": "int64", "age": "int64"},
        )
        assert isinstance(result, ar.ArFrame)

    def test_allow_lossy_casts_accepted(self, messy_frame):
        """allow_lossy_casts=True permits precision loss during casting."""
        result = ar.auto_clean(
            messy_frame,
            allow_lossy_casts=True,
            confirmed_casts={"score": "int64"},
        )
        assert isinstance(result, ar.ArFrame)

    def test_report_contains_summary(self, messy_frame):
        """DataQualityReport from return_report has summary() method."""
        frame, report = ar.auto_clean(messy_frame, return_report=True)
        summary = report.summary()
        assert isinstance(summary, dict)
        assert "row_count" in summary or "rows" in summary.keys() or len(summary) > 0

    def test_strict_mode_idempotent(self, messy_frame):
        """Applying strict mode twice produces same result (idempotent)."""
        result1 = ar.auto_clean(messy_frame, mode="strict")
        result2 = ar.auto_clean(result1, mode="strict")
        df1 = ar.to_pandas(result1)
        df2 = ar.to_pandas(result2)
        assert df1.shape == df2.shape
        pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))
