import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCompareProfiles:
    """Test arnio.quality.compare_profiles function."""

    def test_identical_profiles_all_ok(self):
        """Identical profiles should report 'ok' status for all columns."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [85.5, 90.0, 88.2],
        })
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        assert all(
            info["status"] == "ok"
            for info in comparison.drift_report.values()
        )
        assert comparison.status_counts.get("ok", 0) == len(comparison.drift_report)

    def test_different_null_ratios_detected(self):
        """Changes in null ratio should be detected as warning or changed."""
        df1 = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        df2 = pd.DataFrame({"value": [1, 2, None, 4, 5]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        assert comparison.drift_report["value"]["status"] in ["warning", "changed"]

    def test_different_column_sets_raises_valueerror(self):
        """Profiles with non-matching columns should raise ValueError."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "c": [5, 6]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        with pytest.raises(ValueError):
            ar.compare_profiles(profile1, profile2)

    def test_numeric_mean_change_detected(self):
        """Significant change in numeric mean should be detected."""
        df1 = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        df2 = pd.DataFrame({"value": [10.0, 20.0, 30.0, 40.0, 50.0]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        assert comparison.drift_report["value"]["status"] in ["warning", "changed"]

    def test_drift_report_structure(self):
        """drift_report should have expected structure with status field."""
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        assert isinstance(comparison.drift_report, dict)
        for col_name, col_report in comparison.drift_report.items():
            assert isinstance(col_report, dict)
            assert "status" in col_report
            assert col_report["status"] in ["ok", "warning", "changed"]

    def test_status_counts_present_and_valid(self):
        """status_counts should summarize drift statuses."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        assert hasattr(comparison, "status_counts")
        assert isinstance(comparison.status_counts, dict)
        assert all(
            status in ["ok", "warning", "changed"]
            for status in comparison.status_counts.keys()
        )

    def test_multiple_columns_mixed_drift(self):
        """Handle profiles with some columns drifted, others unchanged."""
        df1 = pd.DataFrame({
            "stable": [1, 2, 3],
            "drifted": [1, 2, 3],
        })
        df2 = pd.DataFrame({
            "stable": [1, 2, 3],
            "drifted": [100, 200, 300],
        })

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        assert comparison.drift_report["stable"]["status"] == "ok"
        assert comparison.drift_report["drifted"]["status"] in ["warning", "changed"]

    def test_all_columns_present_in_drift_report(self):
        """All columns from profiles should appear in drift_report."""
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": ["a", "b"],
            "col3": [1.5, 2.5],
        })
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        assert set(comparison.drift_report.keys()) == {"col1", "col2", "col3"}

    def test_single_column_profiles(self):
        """Single-column profiles should compare correctly."""
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"x": [1, 2, 3]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        assert len(comparison.drift_report) == 1
        assert "x" in comparison.drift_report

    def test_string_column_null_ratio_change(self):
        """String column null ratio changes should be detected."""
        df1 = pd.DataFrame({"text": ["hello", "world", "test"]})
        df2 = pd.DataFrame({"text": ["hello", None, "test"]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        assert comparison.drift_report["text"]["status"] in ["warning", "changed"]

    def test_asymmetric_column_subset_raises_error(self):
        """Profiles with asymmetric column sets should raise ValueError."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [1], "b": [2]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        with pytest.raises(ValueError):
            ar.compare_profiles(profile1, profile2)

    def test_numeric_std_deviation_change(self):
        """Changes in numeric standard deviation should be detected."""
        df1 = pd.DataFrame({"value": [1.0, 2.0, 3.0]})  # std ≈ 1.0
        df2 = pd.DataFrame({"value": [1.0, 50.0, 99.0]})  # std ≈ 50.0

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        assert comparison.drift_report["value"]["status"] in ["warning", "changed"]

    def test_empty_dataframe_comparison(self):
        """Comparing profiles of empty frames should work."""
        df = pd.DataFrame({"col": pd.Series([], dtype="int64")})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        comparison = ar.compare_profiles(profile, profile)

        assert "col" in comparison.drift_report

    def test_profiles_with_boolean_columns(self):
        """Boolean columns should be compared correctly."""
        df1 = pd.DataFrame({"flag": [True, False, True]})
        df2 = pd.DataFrame({"flag": [True, False, True]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        assert comparison.drift_report["flag"]["status"] == "ok"

    def test_column_name_order_independent(self):
        """Column order should not affect comparison (only presence matters)."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"b": [3, 4], "a": [1, 2]})  # Different order

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        # Should succeed since columns match
        comparison = ar.compare_profiles(profile1, profile2)

        assert "a" in comparison.drift_report
        assert "b" in comparison.drift_report

    def test_return_type_has_required_attributes(self):
        """Returned ProfileComparison should have drift_report and status_counts."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)

        result = ar.compare_profiles(profile, profile)

        assert hasattr(result, "drift_report")
        assert hasattr(result, "status_counts")
