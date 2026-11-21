import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
from pathlib import Path


class TestCompareProfiles:
    """End-to-end tests for arnio.quality.compare_profiles."""

    def test_compare_identical_datasets(self, tmp_path):
        """Identical datasets should show 'ok' status for all columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,alice,95\n2,bob,88\n3,charlie,92\n")
        
        frame = ar.read_csv(str(csv_file))
        profile_a = ar.profile(frame)
        profile_b = ar.profile(frame)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert comparison.status_counts["ok"] == 3
        assert comparison.status_counts["warning"] == 0
        assert comparison.status_counts["changed"] == 0
        
        for col_status in comparison.drift_report.values():
            assert col_status["status"] == "ok"

    def test_compare_different_null_ratios(self, tmp_path):
        """Different null ratios should trigger warning or changed status."""
        csv_a = tmp_path / "data_a.csv"
        csv_b = tmp_path / "data_b.csv"
        
        csv_a.write_text("id,score\n1,95\n2,88\n3,92\n")
        csv_b.write_text("id,score\n1,95\n2,\n3,92\n")  # One null in score
        
        frame_a = ar.read_csv(str(csv_a))
        frame_b = ar.read_csv(str(csv_b))
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        # Score column should show change due to null ratio difference
        assert "score" in comparison.drift_report
        assert comparison.drift_report["score"]["status"] in ["warning", "changed"]

    def test_compare_incompatible_schemas_raises_valueerror(self, tmp_path):
        """Profiles with different column sets should raise ValueError."""
        csv_a = tmp_path / "data_a.csv"
        csv_b = tmp_path / "data_b.csv"
        
        csv_a.write_text("id,name\n1,alice\n2,bob\n")
        csv_b.write_text("id,score\n1,95\n2,88\n")
        
        frame_a = ar.read_csv(str(csv_a))
        frame_b = ar.read_csv(str(csv_b))
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        with pytest.raises(ValueError, match="Profiles have incompatible schemas"):
            ar.compare_profiles(profile_a, profile_b)

    def test_compare_wrong_input_types_raises_typeerror(self):
        """Non-DataQualityReport inputs should raise TypeError."""
        with pytest.raises(TypeError, match="compare_profiles expects two DataQualityReport"):
            ar.compare_profiles("not a report", "also not a report")
        
        # Even one wrong type should fail
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        
        with pytest.raises(TypeError, match="compare_profiles expects two DataQualityReport"):
            ar.compare_profiles(profile, None)

    def test_compare_numeric_stats_changed(self, tmp_path):
        """Different numeric statistics should be detected."""
        csv_a = tmp_path / "data_a.csv"
        csv_b = tmp_path / "data_b.csv"
        
        # Dataset A: scores 90-100
        csv_a.write_text("score\n95\n92\n98\n96\n94\n")
        # Dataset B: scores 50-60
        csv_b.write_text("score\n55\n52\n58\n56\n54\n")
        
        frame_a = ar.read_csv(str(csv_a))
        frame_b = ar.read_csv(str(csv_b))
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        # Score column should show changes due to different mean/min/max
        assert comparison.drift_report["score"]["status"] in ["warning", "changed"]

    def test_compare_dtype_change_detection(self, tmp_path):
        """Type change should be detected and marked as changed."""
        csv_a = tmp_path / "data_a.csv"
        csv_b = tmp_path / "data_b.csv"
        
        csv_a.write_text("value\n10\n20\n30\n")
        csv_b.write_text("value\napple\nbanana\ncherry\n")
        
        frame_a = ar.read_csv(str(csv_a))
        frame_b = ar.read_csv(str(csv_b))
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        # Dtype change should be detected
        assert comparison.drift_report["value"]["status"] == "changed"

    def test_compare_multiple_columns_ok_status(self, tmp_path):
        """Multiple identical columns should all show 'ok' status."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c,d\n1,x,10,10.5\n2,y,20,20.5\n3,z,30,30.5\n")
        
        frame = ar.read_csv(str(csv_file))
        profile_a = ar.profile(frame)
        profile_b = ar.profile(frame)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert comparison.status_counts["ok"] == 4
        assert comparison.status_counts["warning"] == 0
        assert comparison.status_counts["changed"] == 0
        assert sum(comparison.status_counts.values()) == 4

    def test_compare_mixed_statuses(self, tmp_path):
        """Dataset with some changed and some unchanged columns."""
        csv_a = tmp_path / "data_a.csv"
        csv_b = tmp_path / "data_b.csv"
        
        # First dataset
        csv_a.write_text("id,name,score\n1,alice,95\n2,bob,88\n3,charlie,92\n")
        # Second: same names, different scores
        csv_b.write_text("id,name,score\n1,alice,50\n2,bob,60\n3,charlie,70\n4,dave,80\n")
        
        frame_a = ar.read_csv(str(csv_a))
        frame_b = ar.read_csv(str(csv_b))
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        # score should change (different distribution), name may be ok
        assert comparison.drift_report["score"]["status"] in ["warning", "changed"]
        assert sum(comparison.status_counts.values()) == 3

    def test_compare_single_column_dataset(self, tmp_path):
        """Minimal case with single column should work correctly."""
        csv_a = tmp_path / "data_a.csv"
        csv_b = tmp_path / "data_b.csv"
        
        csv_a.write_text("value\n10\n20\n30\n")
        csv_b.write_text("value\n10\n20\n30\n")
        
        frame_a = ar.read_csv(str(csv_a))
        frame_b = ar.read_csv(str(csv_b))
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert len(comparison.drift_report) == 1
        assert comparison.drift_report["value"]["status"] == "ok"
        assert comparison.status_counts["ok"] == 1

    def test_compare_via_pandas_accessor(self, tmp_path):
        """Compare profiles generated via pandas .arnio accessor."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score\n1,95\n2,88\n3,92\n")
        
        df = pd.read_csv(str(csv_file))
        
        # Use pandas accessor to generate profiles
        profile_a = df.arnio.profile()
        profile_b = df.arnio.profile()
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert comparison.status_counts["ok"] == 2
        assert comparison.status_counts["changed"] == 0
        assert comparison.status_counts["warning"] == 0

    def test_compare_preserves_profile_references(self, tmp_path):
        """ProfileComparison should preserve references to both input profiles."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y\n1,2\n3,4\n")
        
        frame = ar.read_csv(str(csv_file))
        profile_a = ar.profile(frame)
        profile_b = ar.profile(frame)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert comparison.left_profile is profile_a
        assert comparison.right_profile is profile_b

    def test_compare_empty_frames_same_schema(self, tmp_path):
        """Empty frames with same schema should compare without error."""
        csv_a = tmp_path / "empty_a.csv"
        csv_b = tmp_path / "empty_b.csv"
        
        csv_a.write_text("id,name,score\n")
        csv_b.write_text("id,name,score\n")
        
        frame_a = ar.read_csv(str(csv_a))
        frame_b = ar.read_csv(str(csv_b))
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        # Should not raise, should handle empty frames gracefully
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert isinstance(comparison.drift_report, dict)
        assert len(comparison.drift_report) == 3

    def test_compare_status_counts_sum_to_total_columns(self, tmp_path):
        """Status counts should always sum to total number of columns."""
        csv_a = tmp_path / "data_a.csv"
        csv_b = tmp_path / "data_b.csv"
        
        csv_a.write_text("a,b,c,d,e\n1,2,3,4,5\n6,7,8,9,10\n")
        csv_b.write_text("a,b,c,d,e\n1,2,99,4,5\n6,7,88,9,10\n")
        
        frame_a = ar.read_csv(str(csv_a))
        frame_b = ar.read_csv(str(csv_b))
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        total_columns = len(profile_a.columns)
        total_status_count = sum(comparison.status_counts.values())
        assert total_status_count == total_columns
        assert total_columns == 5

    def test_compare_drift_report_structure(self, tmp_path):
        """Verify drift_report contains expected structure for each column."""
        csv_a = tmp_path / "data_a.csv"
        csv_b = tmp_path / "data_b.csv"
        
        csv_a.write_text("id,value\n1,100\n2,200\n")
        csv_b.write_text("id,value\n1,100\n2,200\n")
        
        frame_a = ar.read_csv(str(csv_a))
        frame_b = ar.read_csv(str(csv_b))
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        # Each column in drift_report should have required structure
        for col_name, col_entry in comparison.drift_report.items():
            assert isinstance(col_entry, dict)
            assert "status" in col_entry
            assert col_entry["status"] in ["ok", "warning", "changed"]
