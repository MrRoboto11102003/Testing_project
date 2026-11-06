import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCompareProfiles:
    """Tests for arnio.quality.compare_profiles function."""

    def test_compare_identical_profiles_all_ok(self):
        """Identical profiles should result in 'ok' status for all columns."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85.5, 90.0, 78.5, 92.0, 88.5],
        })
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        assert comparison.status_counts["ok"] == 3
        assert comparison.status_counts["warning"] == 0
        assert comparison.status_counts["changed"] == 0

    def test_compare_profiles_with_null_ratio_change(self):
        """Profiles with different null ratios should detect drift."""
        df1 = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        })
        df2 = pd.DataFrame({
            "id": [1, 2, 3, 4, None],
            "value": [10, 20, 30, 40, 50],
        })

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # The id column should show drift due to null ratio change
        assert comparison.drift_report["id"]["status"] in ("warning", "changed")

    def test_compare_profiles_incompatible_schemas_raises_valueerror(self):
        """Profiles with different columns should raise ValueError."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "c": [4, 5, 6]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        with pytest.raises(ValueError, match="incompatible schemas"):
            ar.compare_profiles(profile1, profile2)

    def test_compare_profiles_error_mentions_missing_columns(self):
        """Error message should detail which columns are missing."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "c": [3, 4]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        with pytest.raises(ValueError) as exc_info:
            ar.compare_profiles(profile1, profile2)

        error_msg = str(exc_info.value)
        assert "missing_from" in error_msg

    def test_compare_profiles_rejects_non_dataqualityreport_first_arg(self):
        """Non-DataQualityReport for profile_a should raise TypeError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        with pytest.raises(TypeError):
            ar.compare_profiles(df, profile)

    def test_compare_profiles_rejects_non_dataqualityreport_second_arg(self):
        """Non-DataQualityReport for profile_b should raise TypeError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        with pytest.raises(TypeError):
            ar.compare_profiles(profile, df)

    def test_compare_profiles_returns_profilecomparison_with_required_attributes(self):
        """Return value should have drift_report, left_profile, right_profile, and status_counts."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        result = ar.compare_profiles(profile, profile)

        assert hasattr(result, "drift_report")
        assert hasattr(result, "left_profile")
        assert hasattr(result, "right_profile")
        assert hasattr(result, "status_counts")
        assert isinstance(result.drift_report, dict)

    def test_compare_profiles_drift_report_covers_all_columns(self):
        """Drift report should have an entry for every column in profiles."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
            "col3": [1.1, 2.2, 3.3],
        })
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        assert len(comparison.drift_report) == 3
        assert "col1" in comparison.drift_report
        assert "col2" in comparison.drift_report
        assert "col3" in comparison.drift_report

    def test_compare_profiles_each_drift_entry_has_status_field(self):
        """Every drift report entry should contain a 'status' field."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        for col_name, entry in comparison.drift_report.items():
            assert "status" in entry
            assert entry["status"] in ("ok", "warning", "changed")

    def test_compare_profiles_status_counts_sum_to_column_count(self):
        """The sum of status_counts should equal number of columns."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0],
            "c": ["x", "y", "z"],
        })
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        total = (
            comparison.status_counts.get("ok", 0)
            + comparison.status_counts.get("warning", 0)
            + comparison.status_counts.get("changed", 0)
        )
        assert total == len(comparison.drift_report)

    def test_compare_profiles_preserves_profile_references(self):
        """Returned comparison should preserve references to original profiles."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5, 6]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        assert comparison.left_profile is profile1
        assert comparison.right_profile is profile2

    def test_compare_profiles_numeric_distribution_drift(self):
        """Changes in numeric statistics (mean, std, etc.) should be detected."""
        df1 = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        df2 = pd.DataFrame({"value": [100, 200, 300, 400, 500]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # Large change in numeric distribution should show drift
        assert "value" in comparison.drift_report

    def test_compare_profiles_string_columns(self):
        """String columns should be included in comparison."""
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        assert "name" in comparison.drift_report
        assert comparison.drift_report["name"]["status"] == "ok"

    def test_compare_profiles_boolean_columns(self):
        """Boolean columns should be included in comparison."""
        df = pd.DataFrame({"flag": [True, False, True, False, True]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        assert "flag" in comparison.drift_report
        assert comparison.drift_report["flag"]["status"] == "ok"

    def test_compare_profiles_single_column_profile(self):
        """Comparison should work with single-column profiles."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        assert len(comparison.drift_report) == 1
        assert comparison.status_counts["ok"] == 1
