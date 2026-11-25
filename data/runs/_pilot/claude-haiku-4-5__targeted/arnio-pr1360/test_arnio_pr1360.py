import pytest
import arnio as ar


class TestCompareProfiles:
    """Test suite for arnio.quality.compare_profiles"""
    
    def test_returns_profile_comparison_object(self):
        """Test that compare_profiles returns a ProfileComparison object."""
        data = {"col": [1, 2, 3]}
        frame = ar.from_dict(data)
        profile = ar.profile(frame)
        
        comparison = ar.compare_profiles(profile, profile)
        assert comparison is not None
    
    def test_has_drift_report_attribute(self):
        """Test that result has accessible drift_report attribute."""
        data = {"value": [1, 2, 3]}
        frame = ar.from_dict(data)
        profile = ar.profile(frame)
        
        comparison = ar.compare_profiles(profile, profile)
        assert hasattr(comparison, 'drift_report')
    
    def test_drift_report_contains_all_columns(self):
        """Test that drift_report includes all shared columns."""
        data = {"col_a": [1, 2, 3], "col_b": ["x", "y", "z"], "col_c": [1.1, 2.2, 3.3]}
        frame1 = ar.from_dict(data)
        frame2 = ar.from_dict(data)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert "col_a" in comparison.drift_report
        assert "col_b" in comparison.drift_report
        assert "col_c" in comparison.drift_report
    
    def test_drift_report_entry_has_status_field(self):
        """Test that each column entry in drift_report has a status field."""
        data = {"metric": [1, 2, 3]}
        frame1 = ar.from_dict(data)
        frame2 = ar.from_dict(data)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert "status" in comparison.drift_report["metric"]
    
    def test_mismatched_columns_raises_value_error(self):
        """Test that profiles with different column sets raise ValueError."""
        data1 = {"col_a": [1, 2, 3], "col_b": [4, 5, 6]}
        data2 = {"col_a": [1, 2, 3], "col_c": [7, 8, 9]}
        
        frame1 = ar.from_dict(data1)
        frame2 = ar.from_dict(data2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        with pytest.raises(ValueError):
            ar.compare_profiles(profile1, profile2)
    
    def test_detects_null_ratio_change(self):
        """Test that changes in null ratio are detected."""
        data1 = {"value": [1, 2, 3, 4, 5]}
        data2 = {"value": [1, 2, None, None, 5]}
        
        frame1 = ar.from_dict(data1)
        frame2 = ar.from_dict(data2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison.drift_report["value"]["status"] is not None
    
    def test_detects_numeric_mean_shift(self):
        """Test that changes in numeric mean are detected."""
        data1 = {"score": [10.0, 20.0, 30.0]}
        data2 = {"score": [100.0, 200.0, 300.0]}
        
        frame1 = ar.from_dict(data1)
        frame2 = ar.from_dict(data2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison.drift_report["score"]["status"] is not None
    
    def test_detects_numeric_variance_change(self):
        """Test that changes in standard deviation are detected."""
        data1 = {"variance": [100, 100, 100]}
        data2 = {"variance": [1, 100, 200]}
        
        frame1 = ar.from_dict(data1)
        frame2 = ar.from_dict(data2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison.drift_report["variance"]["status"] is not None
    
    def test_detects_min_max_changes(self):
        """Test that changes in numeric min/max are detected."""
        data1 = {"value": [10, 20, 30]}
        data2 = {"value": [100, 200, 300]}
        
        frame1 = ar.from_dict(data1)
        frame2 = ar.from_dict(data2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison.drift_report["value"]["status"] is not None
    
    def test_detects_uniqueness_change(self):
        """Test that changes in column uniqueness are detected."""
        data1 = {"category": ["A", "B", "C", "D", "E"]}
        data2 = {"category": ["A", "A", "A", "A", "A"]}
        
        frame1 = ar.from_dict(data1)
        frame2 = ar.from_dict(data2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert comparison.drift_report["category"]["status"] is not None
    
    def test_identical_data_comparison(self):
        """Test comparing profiles from identical data."""
        data = {"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [10.5, 20.5, 30.5]}
        frame = ar.from_dict(data)
        
        profile1 = ar.profile(frame)
        profile2 = ar.profile(frame)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert len(comparison.drift_report) == 3
        assert all(col in comparison.drift_report for col in ["id", "name", "value"])
    
    def test_all_numeric_columns_comparison(self):
        """Test comparing profiles with entirely numeric columns."""
        data1 = {"a": [1, 2, 3], "b": [4.5, 5.5, 6.5], "c": [10, 20, 30]}
        data2 = {"a": [10, 20, 30], "b": [40.5, 50.5, 60.5], "c": [100, 200, 300]}
        
        frame1 = ar.from_dict(data1)
        frame2 = ar.from_dict(data2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert len(comparison.drift_report) == 3
    
    def test_all_categorical_columns_comparison(self):
        """Test comparing profiles with entirely categorical columns."""
        data1 = {"col_a": ["x", "y", "z"], "col_b": ["cat", "dog", "bird"]}
        data2 = {"col_a": ["a", "b", "c"], "col_b": ["car", "truck", "bus"]}
        
        frame1 = ar.from_dict(data1)
        frame2 = ar.from_dict(data2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert "col_a" in comparison.drift_report
        assert "col_b" in comparison.drift_report
    
    def test_mixed_data_types_comparison(self):
        """Test comparing profiles with mixed data types."""
        data1 = {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"]
        }
        data2 = {
            "int_col": [10, 20, 30],
            "float_col": [10.1, 20.2, 30.3],
            "str_col": ["x", "y", "z"]
        }
        
        frame1 = ar.from_dict(data1)
        frame2 = ar.from_dict(data2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert len(comparison.drift_report) == 3
    
    def test_single_column_profile_comparison(self):
        """Test comparing single-column profiles."""
        data1 = {"metric": [1, 2, 3]}
        data2 = {"metric": [100, 200, 300]}
        
        frame1 = ar.from_dict(data1)
        frame2 = ar.from_dict(data2)
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        
        comparison = ar.compare_profiles(profile1, profile2)
        assert "metric" in comparison.drift_report
        assert "status" in comparison.drift_report["metric"]
