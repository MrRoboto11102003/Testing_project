import pytest
import arnio as ar
import pandas as pd
import numpy as np


class TestCompareProfiles:
    """Tests for ar.compare_profiles() function."""

    def test_identical_profiles_all_ok_status(self):
        """Identical profiles should have all columns marked 'ok'."""
        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)

        profile1 = ar.profile(frame)
        profile2 = ar.profile(frame)

        comparison = ar.compare_profiles(profile1, profile2)

        assert comparison.status_counts["ok"] == 2
        assert comparison.status_counts["warning"] == 0
        assert comparison.status_counts["changed"] == 0

    def test_returns_profile_comparison_with_required_attributes(self):
        """ProfileComparison should contain expected attributes."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        result = ar.compare_profiles(profile, profile)

        assert hasattr(result, "left_profile")
        assert hasattr(result, "right_profile")
        assert hasattr(result, "drift_report")
        assert hasattr(result, "status_counts")

    def test_preserves_input_profiles(self):
        """Input profiles should be stored unchanged in result."""
        df = pd.DataFrame({"col": [1, 2]})
        frame = ar.from_pandas(df)
        profile1 = ar.profile(frame)
        profile2 = ar.profile(frame)

        comparison = ar.compare_profiles(profile1, profile2)

        assert comparison.left_profile is profile1
        assert comparison.right_profile is profile2

    def test_drift_report_contains_all_columns(self):
        """Drift report should include all columns from profiles."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
            "col3": [1.5, 2.5, 3.5],
        })
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        assert set(comparison.drift_report.keys()) == {"col1", "col2", "col3"}

    def test_each_drift_entry_has_status_field(self):
        """Each column in drift_report should have a 'status' field."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        for entry in comparison.drift_report.values():
            assert "status" in entry
            assert entry["status"] in {"ok", "warning", "changed"}

    def test_status_counts_sum_equals_column_count(self):
        """Status counts should sum to the total number of columns."""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        })
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        total = (
            comparison.status_counts["ok"]
            + comparison.status_counts["warning"]
            + comparison.status_counts["changed"]
        )
        assert total == len(comparison.drift_report)

    def test_missing_column_in_first_profile_raises_value_error(self):
        """ValueError when first profile is missing a column."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        profile1 = ar.profile(ar.from_pandas(df1))
        profile2 = ar.profile(ar.from_pandas(df2))

        with pytest.raises(ValueError, match="incompatible schemas"):
            ar.compare_profiles(profile1, profile2)

    def test_missing_column_in_second_profile_raises_value_error(self):
        """ValueError when second profile is missing a column."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        profile1 = ar.profile(ar.from_pandas(df1))
        profile2 = ar.profile(ar.from_pandas(df2))

        with pytest.raises(ValueError, match="incompatible schemas"):
            ar.compare_profiles(profile1, profile2)

    def test_type_error_when_first_arg_not_report(self):
        """TypeError when first argument is not a DataQualityReport."""
        df = pd.DataFrame({"x": [1]})
        profile = ar.profile(ar.from_pandas(df))

        with pytest.raises(TypeError):
            ar.compare_profiles("not a report", profile)

    def test_type_error_when_second_arg_not_report(self):
        """TypeError when second argument is not a DataQualityReport."""
        df = pd.DataFrame({"x": [1]})
        profile = ar.profile(ar.from_pandas(df))

        with pytest.raises(TypeError):
            ar.compare_profiles(profile, 123)

    def test_single_column_dataframe(self):
        """Should handle DataFrames with a single column."""
        df = pd.DataFrame({"single_col": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        profile1 = ar.profile(frame)
        profile2 = ar.profile(frame)

        comparison = ar.compare_profiles(profile1, profile2)

        assert len(comparison.drift_report) == 1
        assert "single_col" in comparison.drift_report

    def test_different_null_ratio_detected_as_not_ok(self):
        """Profiles with different null ratios should not be 'ok'."""
        df1 = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        df2 = pd.DataFrame({"col": [1, 2, None, None, 5]})

        profile1 = ar.profile(ar.from_pandas(df1))
        profile2 = ar.profile(ar.from_pandas(df2))

        comparison = ar.compare_profiles(profile1, profile2)

        # Status should reflect the difference in null counts
        assert comparison.drift_report["col"]["status"] != "ok"

    def test_dtype_difference_detected(self):
        """Profiles with different dtypes should be detected."""
        # Create two profiles with different dtypes for the same data
        df1 = pd.DataFrame({"col": [1, 2, 3]})
        df2 = pd.DataFrame({"col": ["1", "2", "3"]})

        profile1 = ar.profile(ar.from_pandas(df1))
        profile2 = ar.profile(ar.from_pandas(df2))

        comparison = ar.compare_profiles(profile1, profile2)

        # Dtype mismatch should be detected
        assert comparison.drift_report["col"]["status"] != "ok"

    def test_multiple_columns_with_mixed_differences(self):
        """Should correctly report mixed drift statuses across columns."""
        df1 = pd.DataFrame({
            "unchanged": [1, 2, 3],
            "changed": [10, 20, 30],
        })
        df2 = pd.DataFrame({
            "unchanged": [1, 2, 3],
            "changed": [100, 200, 300],
        })

        profile1 = ar.profile(ar.from_pandas(df1))
        profile2 = ar.profile(ar.from_pandas(df2))

        comparison = ar.compare_profiles(profile1, profile2)

        assert "unchanged" in comparison.drift_report
        assert "changed" in comparison.drift_report
        # At least one column should show drift
        assert sum(comparison.status_counts.values()) == 2

    def test_error_message_includes_missing_columns_info(self):
        """ValueError message should indicate which columns are missing."""
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [1], "c": [3]})

        profile1 = ar.profile(ar.from_pandas(df1))
        profile2 = ar.profile(ar.from_pandas(df2))

        with pytest.raises(ValueError) as exc_info:
            ar.compare_profiles(profile1, profile2)

        error_msg = str(exc_info.value)
        assert "missing_from" in error_msg or "incompatible" in error_msg
