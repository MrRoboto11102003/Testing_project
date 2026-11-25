import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCompareProfiles:
    """Test suite for ar.compare_profiles()"""

    def test_identical_profiles_all_ok(self):
        """Identical profiles should show 'ok' status for all columns."""
        data = {"id": [1, 2, 3], "name": ["alice", "bob", "charlie"]}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)

        profile_a = ar.profile(frame)
        profile_b = ar.profile(frame)

        comparison = ar.compare_profiles(profile_a, profile_b)

        # All columns should have 'ok' status
        for col_status in comparison.drift_report.values():
            assert col_status["status"] == "ok"

        # Verify status_counts reflects this
        assert comparison.status_counts.get("ok") == len(comparison.drift_report)

    def test_different_columns_raises_value_error(self):
        """Comparing profiles with different column sets should raise ValueError."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        with pytest.raises(ValueError):
            ar.compare_profiles(profile1, profile2)

    def test_null_ratio_increase_detected(self):
        """Increased null ratio should be detected as drift."""
        df1 = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        df2 = pd.DataFrame({"value": [1.0, 2.0, np.nan, np.nan, np.nan]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # Null ratio change should trigger warning or changed status
        assert comparison.drift_report["value"]["status"] in ["warning", "changed"]

    def test_numeric_mean_drift_detected(self):
        """Significant changes in numeric mean should be detected."""
        df1 = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"value": [10.0, 20.0, 30.0]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # Mean change should be detected as drift
        assert comparison.drift_report["value"]["status"] in ["warning", "changed"]

    def test_dtype_change_detected(self):
        """Changes in column dtype should be detected."""
        df1 = pd.DataFrame({"value": [1, 2, 3]})
        df2 = pd.DataFrame({"value": ["1", "2", "3"]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # Dtype change should be detected
        assert comparison.drift_report["value"]["status"] in ["warning", "changed"]

    def test_status_counts_structure_and_validity(self):
        """status_counts should contain valid status strings with correct counts."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, np.nan]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # status_counts should be a dict
        assert isinstance(comparison.status_counts, dict)

        # Total counts should equal number of columns
        total_status_count = sum(comparison.status_counts.values())
        assert total_status_count == len(comparison.drift_report)

        # All keys should be valid status strings
        valid_statuses = {"ok", "warning", "changed"}
        assert all(
            status in valid_statuses for status in comparison.status_counts.keys()
        )

    def test_drift_report_column_names_match(self):
        """drift_report keys should match the profiled column names."""
        df1 = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4], "col_c": [5, 6]})
        df2 = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4], "col_c": [5, 6]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # drift_report should contain all columns
        assert set(comparison.drift_report.keys()) == {"col_a", "col_b", "col_c"}

    def test_mixed_drift_some_ok_some_drift(self):
        """Some columns drifted, some unchanged."""
        df1 = pd.DataFrame(
            {"stable": [1, 2, 3], "drifting": [1.0, 2.0, 3.0], "changing": ["a", "b", "c"]}
        )
        df2 = pd.DataFrame(
            {"stable": [1, 2, 3], "drifting": [10.0, 20.0, 30.0], "changing": ["x", "y", "z"]}
        )

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # At least one should be ok, at least one should show drift
        statuses = [col["status"] for col in comparison.drift_report.values()]
        assert "ok" in statuses
        assert any(status != "ok" for status in statuses)

    def test_string_column_null_increase_detected(self):
        """String columns with increased null ratio should show drift."""
        df1 = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        df2 = pd.DataFrame({"name": ["alice", None, None]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        assert comparison.drift_report["name"]["status"] in ["warning", "changed"]

    def test_all_null_column_comparison(self):
        """Identical all-null columns should compare without errors."""
        df1 = pd.DataFrame({"value": [None, None, None]})
        df2 = pd.DataFrame({"value": [None, None, None]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # Should complete without error and show no drift
        assert comparison.drift_report["value"]["status"] == "ok"

    def test_zero_nulls_to_some_nulls(self):
        """Introducing nulls where there were none should be detected."""
        df1 = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        df2 = pd.DataFrame({"value": [1, 2, None, 4, 5]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # Going from 0% to 20% null should be detected
        assert comparison.drift_report["value"]["status"] in ["warning", "changed"]

    def test_large_numeric_distribution_shift(self):
        """Large shift in numeric distribution should be detected."""
        df1 = pd.DataFrame({"value": list(range(50))})
        df2 = pd.DataFrame({"value": list(range(100, 150))})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # Shift from 0-50 to 100-150 should be detected
        assert comparison.drift_report["value"]["status"] in ["warning", "changed"]

    def test_symmetric_comparison_both_directions(self):
        """Comparing A→B and B→A should both detect drift (though direction differs)."""
        df1 = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"value": [1.0, 2.0, np.nan]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison_ab = ar.compare_profiles(profile1, profile2)
        comparison_ba = ar.compare_profiles(profile2, profile1)

        # Both directions should detect drift
        assert comparison_ab.drift_report["value"]["status"] != "ok"
        assert comparison_ba.drift_report["value"]["status"] != "ok"

    def test_single_row_frame_comparison(self):
        """Single-row frames should be comparable without errors."""
        df1 = pd.DataFrame({"value": [1], "name": ["a"]})
        df2 = pd.DataFrame({"value": [2], "name": ["b"]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        # Should not raise error
        comparison = ar.compare_profiles(profile1, profile2)
        assert "value" in comparison.drift_report
        assert "name" in comparison.drift_report

    def test_numeric_std_deviation_change(self):
        """Changes in standard deviation should be detected."""
        # Low variance data
        df1 = pd.DataFrame({"value": [1.0, 1.1, 1.0, 1.1, 1.0]})
        # High variance data (same mean ~1.04)
        df2 = pd.DataFrame({"value": [0.0, 0.5, 1.0, 1.5, 2.0]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # Variance change should be detected
        assert comparison.drift_report["value"]["status"] in ["warning", "changed"]

    def test_profile_comparison_return_type(self):
        """Returned object should have drift_report and status_counts attributes."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 3]})

        frame1 = ar.from_pandas(df1)
        frame2 = ar.from_pandas(df2)

        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)

        comparison = ar.compare_profiles(profile1, profile2)

        # Should have required attributes
        assert hasattr(comparison, "drift_report")
        assert hasattr(comparison, "status_counts")

        # Types should be correct
        assert isinstance(comparison.drift_report, dict)
        assert isinstance(comparison.status_counts, dict)
