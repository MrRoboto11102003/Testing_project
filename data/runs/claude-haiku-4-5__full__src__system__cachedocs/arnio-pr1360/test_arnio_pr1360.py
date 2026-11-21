"""
System (end-to-end) tests for arnio.quality.compare_profiles().

These tests drive complete workflows through arnio's public API:
- Real CSV files on disk (via pytest tmp_path)
- Loading and profiling with ar.read_csv() and ar.profile()
- Comparing profiles with ar.compare_profiles()
- Verifying results and error handling
"""

import pytest
import pandas as pd
import arnio as ar


class TestCompareProfiles:
    """End-to-end system tests for ar.compare_profiles()"""

    def test_identical_profiles_all_ok(self, tmp_path):
        """Comparing identical profiles should mark all columns as 'ok'"""
        csv_content = "id,name,score\n1,alice,95\n2,bob,88\n3,charlie,92\n"
        
        path_a = tmp_path / "data_a.csv"
        path_b = tmp_path / "data_b.csv"
        path_a.write_text(csv_content)
        path_b.write_text(csv_content)
        
        frame_a = ar.read_csv(str(path_a))
        frame_b = ar.read_csv(str(path_b))
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        # All columns should be "ok" when profiles are identical
        assert all(entry["status"] == "ok" for entry in comparison.drift_report.values())
        assert comparison.status_counts["ok"] > 0
        assert comparison.status_counts["warning"] == 0
        assert comparison.status_counts["changed"] == 0

    def test_null_ratio_increase_triggers_drift(self, tmp_path):
        """Significant increase in null ratio should trigger warning/changed"""
        csv_no_nulls = "id,score\n1,100\n2,95\n3,90\n4,85\n"
        csv_many_nulls = "id,score\n1,100\n2,\n3,\n4,\n"
        
        path_base = tmp_path / "base.csv"
        path_nulls = tmp_path / "nulls.csv"
        path_base.write_text(csv_no_nulls)
        path_nulls.write_text(csv_many_nulls)
        
        profile_base = ar.profile(ar.read_csv(str(path_base)))
        profile_nulls = ar.profile(ar.read_csv(str(path_nulls)))
        
        comparison = ar.compare_profiles(profile_base, profile_nulls)
        
        # score column should show drift (75% increase in nulls)
        assert comparison.drift_report["score"]["status"] in ("warning", "changed")

    def test_dtype_mismatch_detected(self, tmp_path):
        """When a column changes dtype, drift should be detected"""
        csv_string_ids = "id,value\na,100\nb,200\nc,300\n"
        csv_int_ids = "id,value\n1,100\n2,200\n3,300\n"
        
        path_str = tmp_path / "string_ids.csv"
        path_int = tmp_path / "int_ids.csv"
        path_str.write_text(csv_string_ids)
        path_int.write_text(csv_int_ids)
        
        profile_str = ar.profile(ar.read_csv(str(path_str)))
        profile_int = ar.profile(ar.read_csv(str(path_int)))
        
        comparison = ar.compare_profiles(profile_str, profile_int)
        
        # id column should show dtype change
        assert comparison.drift_report["id"]["status"] in ("warning", "changed")

    def test_numeric_distribution_drift(self, tmp_path):
        """Changes in numeric column statistics should be detected"""
        csv_small_range = "amount\n10\n20\n30\n"
        csv_large_range = "amount\n1000\n2000\n3000\n"
        
        path_small = tmp_path / "small.csv"
        path_large = tmp_path / "large.csv"
        path_small.write_text(csv_small_range)
        path_large.write_text(csv_large_range)
        
        profile_small = ar.profile(ar.read_csv(str(path_small)))
        profile_large = ar.profile(ar.read_csv(str(path_large)))
        
        comparison = ar.compare_profiles(profile_small, profile_large)
        
        # Mean, min, max differ significantly
        assert comparison.drift_report["amount"]["status"] in ("warning", "changed")

    def test_mismatched_column_schemas_raises_error(self, tmp_path):
        """Should raise ValueError when profiles have different column sets"""
        csv_a = "id,name\n1,alice\n"
        csv_b = "id,score\n1,95\n"
        
        path_a = tmp_path / "a.csv"
        path_b = tmp_path / "b.csv"
        path_a.write_text(csv_a)
        path_b.write_text(csv_b)
        
        profile_a = ar.profile(ar.read_csv(str(path_a)))
        profile_b = ar.profile(ar.read_csv(str(path_b)))
        
        with pytest.raises(ValueError, match="incompatible schemas"):
            ar.compare_profiles(profile_a, profile_b)

    def test_type_error_on_non_report_input(self):
        """Passing non-DataQualityReport objects should raise TypeError"""
        with pytest.raises(TypeError, match="DataQualityReport"):
            ar.compare_profiles("not a report", "also not")
        
        with pytest.raises(TypeError, match="DataQualityReport"):
            ar.compare_profiles({}, {})

    def test_status_counts_sum_to_column_count(self, tmp_path):
        """status_counts dictionary should sum to total number of columns"""
        csv = "a,b,c\n1,x,10.5\n2,y,20.3\n"
        
        path_a = tmp_path / "a.csv"
        path_b = tmp_path / "b.csv"
        path_a.write_text(csv)
        path_b.write_text(csv)
        
        profile_a = ar.profile(ar.read_csv(str(path_a)))
        profile_b = ar.profile(ar.read_csv(str(path_b)))
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        total = (comparison.status_counts["ok"] + 
                 comparison.status_counts["warning"] + 
                 comparison.status_counts["changed"])
        assert total == len(comparison.drift_report)

    def test_all_columns_included_in_drift_report(self, tmp_path):
        """drift_report should contain exactly the columns from the profiles"""
        csv = "id,name,score,active\n1,alice,95,true\n2,bob,88,false\n"
        
        path_a = tmp_path / "a.csv"
        path_b = tmp_path / "b.csv"
        path_a.write_text(csv)
        path_b.write_text(csv)
        
        profile_a = ar.profile(ar.read_csv(str(path_a)))
        profile_b = ar.profile(ar.read_csv(str(path_b)))
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        expected_columns = {"id", "name", "score", "active"}
        assert set(comparison.drift_report.keys()) == expected_columns

    def test_empty_dataframes_comparison(self, tmp_path):
        """Should handle profiles of empty datasets (header only)"""
        csv_header_only = "id,name,score\n"
        
        path_a = tmp_path / "empty_a.csv"
        path_b = tmp_path / "empty_b.csv"
        path_a.write_text(csv_header_only)
        path_b.write_text(csv_header_only)
        
        profile_a = ar.profile(ar.read_csv(str(path_a)))
        profile_b = ar.profile(ar.read_csv(str(path_b)))
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert isinstance(comparison.drift_report, dict)
        assert len(comparison.drift_report) > 0

    def test_unique_value_ratio_changes(self, tmp_path):
        """Changes in column uniqueness should be detected"""
        csv_varied = "value\na\nb\nc\n"
        csv_repeated = "value\na\na\na\n"
        
        path_varied = tmp_path / "varied.csv"
        path_repeated = tmp_path / "repeated.csv"
        path_varied.write_text(csv_varied)
        path_repeated.write_text(csv_repeated)
        
        profile_varied = ar.profile(ar.read_csv(str(path_varied)))
        profile_repeated = ar.profile(ar.read_csv(str(path_repeated)))
        
        comparison = ar.compare_profiles(profile_varied, profile_repeated)
        
        # value column should detect unique ratio drop from 100% to 33%
        assert comparison.drift_report["value"]["status"] in ("warning", "changed")

    def test_exclude_columns_mismatch_raises_error(self, tmp_path):
        """Profiles created with different exclude_columns should raise ValueError"""
        csv = "id,name,debug\n1,alice,xyz\n2,bob,abc\n"
        
        path_a = tmp_path / "a.csv"
        path_b = tmp_path / "b.csv"
        path_a.write_text(csv)
        path_b.write_text(csv)
        
        profile_full = ar.profile(ar.read_csv(str(path_a)))
        profile_excluded = ar.profile(ar.read_csv(str(path_b)), exclude_columns=["debug"])
        
        with pytest.raises(ValueError, match="incompatible schemas"):
            ar.compare_profiles(profile_full, profile_excluded)

    def test_drift_report_status_field_validation(self, tmp_path):
        """Each drift_report entry must have a valid 'status' field"""
        csv = "x,y,z\n1,2,3\n4,5,6\n"
        
        path_a = tmp_path / "a.csv"
        path_b = tmp_path / "b.csv"
        path_a.write_text(csv)
        path_b.write_text(csv)
        
        profile_a = ar.profile(ar.read_csv(str(path_a)))
        profile_b = ar.profile(ar.read_csv(str(path_b)))
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        for col_name, col_entry in comparison.drift_report.items():
            assert "status" in col_entry, f"Missing status for column {col_name}"
            assert col_entry["status"] in ("ok", "warning", "changed")

    def test_pandas_accessor_workflow(self, tmp_path):
        """Compare profiles created via pandas accessor"""
        csv = "id,value\n1,100\n2,200\n"
        
        path_a = tmp_path / "a.csv"
        path_b = tmp_path / "b.csv"
        path_a.write_text(csv)
        path_b.write_text(csv)
        
        df_a = pd.read_csv(str(path_a))
        df_b = pd.read_csv(str(path_b))
        
        profile_a = df_a.arnio.profile()
        profile_b = df_b.arnio.profile()
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert isinstance(comparison.drift_report, dict)
        assert all(entry["status"] == "ok" for entry in comparison.drift_report.values())

    def test_comparison_result_contains_profiles(self, tmp_path):
        """ProfileComparison should retain references to both input profiles"""
        csv = "a,b\n1,2\n3,4\n"
        
        path_a = tmp_path / "a.csv"
        path_b = tmp_path / "b.csv"
        path_a.write_text(csv)
        path_b.write_text(csv)
        
        profile_a = ar.profile(ar.read_csv(str(path_a)))
        profile_b = ar.profile(ar.read_csv(str(path_b)))
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        # ProfileComparison should include the input profiles
        assert hasattr(result, "left_profile")
        assert hasattr(result, "right_profile")
        assert result.left_profile is profile_a
        assert result.right_profile is profile_b

    def test_comparison_with_quality_gates(self, tmp_path):
        """Demonstrate compare_profiles integration with quality gate checks"""
        # Baseline dataset
        csv_baseline = "id,amount,status\n1,100,active\n2,200,active\n3,300,active\n"
        # Current dataset (with one null in amount)
        csv_current = "id,amount,status\n1,100,active\n2,,inactive\n3,300,active\n"
        
        path_baseline = tmp_path / "baseline.csv"
        path_current = tmp_path / "current.csv"
        path_baseline.write_text(csv_baseline)
        path_current.write_text(csv_current)
        
        profile_baseline = ar.profile(ar.read_csv(str(path_baseline)))
        profile_current = ar.profile(ar.read_csv(str(path_current)))
        
        # Compare profiles first
        comparison = ar.compare_profiles(profile_baseline, profile_current)
        
        # Both columns should show some drift
        assert len(comparison.drift_report) > 0
        
        # Can be used with quality gates
        gate_result = ar.check_quality_gates(
            profile_baseline,
            profile_current,
            max_null_ratio_delta=0.10,
        )
        
        # Null ratio increased in one column; gates may fail or pass depending on threshold
        assert hasattr(gate_result, "passed")
