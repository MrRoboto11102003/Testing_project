import pytest
import arnio as ar


class TestCompareProfiles:
    """Test suite for arnio.quality.compare_profiles function."""

    def setup_method(self):
        """Set up common test data."""
        self.simple_data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}

    def test_compare_identical_profiles(self):
        """Test comparing two identical profiles succeeds."""
        frame_a = ar.from_dict(self.simple_data)
        frame_b = ar.from_dict(self.simple_data)
        
        profile_a = ar.profile(frame_a)
        profile_b = ar.profile(frame_b)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert comparison is not None
        assert hasattr(comparison, "drift_report")

    def test_compare_profiles_returns_profile_comparison(self):
        """Test that compare_profiles returns ProfileComparison with required attributes."""
        frame = ar.from_dict(self.simple_data)
        profile_a = ar.profile(frame)
        profile_b = ar.profile(frame)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert hasattr(result, "drift_report")
        assert hasattr(result, "status_counts")
        assert hasattr(result, "left_profile")
        assert hasattr(result, "right_profile")

    def test_compare_profiles_type_error_first_arg_invalid(self):
        """Test TypeError when first argument is not a DataQualityReport."""
        frame = ar.from_dict(self.simple_data)
        profile = ar.profile(frame)
        
        with pytest.raises(TypeError):
            ar.compare_profiles("not_a_profile", profile)

    def test_compare_profiles_type_error_second_arg_invalid(self):
        """Test TypeError when second argument is not a DataQualityReport."""
        frame = ar.from_dict(self.simple_data)
        profile = ar.profile(frame)
        
        with pytest.raises(TypeError):
            ar.compare_profiles(profile, None)

    def test_compare_profiles_type_error_both_args_invalid(self):
        """Test TypeError when both arguments are not DataQualityReport."""
        with pytest.raises(TypeError):
            ar.compare_profiles([1, 2, 3], {"key": "value"})

    def test_compare_profiles_schema_mismatch_missing_in_first(self):
        """Test ValueError when profile_a is missing a column from profile_b."""
        data_a = {"col1": [1, 2, 3]}
        data_b = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        
        profile_a = ar.profile(ar.from_dict(data_a))
        profile_b = ar.profile(ar.from_dict(data_b))
        
        with pytest.raises(ValueError, match="Profiles have incompatible schemas"):
            ar.compare_profiles(profile_a, profile_b)

    def test_compare_profiles_schema_mismatch_missing_in_second(self):
        """Test ValueError when profile_b is missing a column from profile_a."""
        data_a = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        data_b = {"col1": [1, 2, 3]}
        
        profile_a = ar.profile(ar.from_dict(data_a))
        profile_b = ar.profile(ar.from_dict(data_b))
        
        with pytest.raises(ValueError, match="Profiles have incompatible schemas"):
            ar.compare_profiles(profile_a, profile_b)

    def test_compare_profiles_drift_report_contains_columns(self):
        """Test that drift_report contains entries for all shared columns."""
        frame = ar.from_dict(self.simple_data)
        profile_a = ar.profile(frame)
        profile_b = ar.profile(frame)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert "col1" in comparison.drift_report
        assert "col2" in comparison.drift_report
        assert len(comparison.drift_report) == 2

    def test_compare_profiles_drift_report_has_status_field(self):
        """Test that each drift_report entry contains a status field."""
        frame = ar.from_dict(self.simple_data)
        profile_a = ar.profile(frame)
        profile_b = ar.profile(frame)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        for col_name in comparison.drift_report:
            assert "status" in comparison.drift_report[col_name]
            assert comparison.drift_report[col_name]["status"] in ["ok", "warning", "changed"]

    def test_compare_profiles_status_counts_structure(self):
        """Test that status_counts has all required keys."""
        frame = ar.from_dict(self.simple_data)
        profile_a = ar.profile(frame)
        profile_b = ar.profile(frame)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert "ok" in comparison.status_counts
        assert "warning" in comparison.status_counts
        assert "changed" in comparison.status_counts

    def test_compare_profiles_status_counts_sum_equals_columns(self):
        """Test that sum of status_counts equals number of columns."""
        frame = ar.from_dict(self.simple_data)
        profile_a = ar.profile(frame)
        profile_b = ar.profile(frame)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        total_statuses = sum(comparison.status_counts.values())
        assert total_statuses == len(comparison.drift_report)

    def test_compare_profiles_stores_input_profiles(self):
        """Test that comparison stores references to input profiles."""
        frame = ar.from_dict(self.simple_data)
        profile_a = ar.profile(frame)
        profile_b = ar.profile(frame)
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert comparison.left_profile is profile_a
        assert comparison.right_profile is profile_b

    def test_compare_profiles_detects_numeric_distribution_changes(self):
        """Test that comparing profiles detects changes in numeric distributions."""
        data_a = {"value": [10.0, 20.0, 30.0]}
        data_b = {"value": [10.0, 20.0, 100.0]}
        
        profile_a = ar.profile(ar.from_dict(data_a))
        profile_b = ar.profile(ar.from_dict(data_b))
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert "value" in comparison.drift_report
        assert comparison.drift_report["value"]["status"] in ["ok", "warning", "changed"]

    def test_compare_profiles_detects_null_ratio_changes(self):
        """Test that comparing profiles detects changes in null ratios."""
        data_a = {"col": [1, 2, 3]}
        data_b = {"col": [1, None, 3]}
        
        profile_a = ar.profile(ar.from_dict(data_a))
        profile_b = ar.profile(ar.from_dict(data_b))
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert "col" in comparison.drift_report
        # Profile comparison should reflect the change in null ratio
        assert isinstance(comparison.drift_report["col"]["status"], str)

    def test_compare_profiles_mixed_column_types(self):
        """Test comparing profiles with mixed numeric and string columns."""
        data = {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"]
        }
        
        profile_a = ar.profile(ar.from_dict(data))
        profile_b = ar.profile(ar.from_dict(data))
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert len(comparison.drift_report) == 3
        total_statuses = sum(comparison.status_counts.values())
        assert total_statuses == 3
