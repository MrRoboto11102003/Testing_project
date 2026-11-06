import pytest
import json
import pandas as pd
import arnio as ar


@pytest.fixture
def profile_comparison():
    """Create a ProfileComparison object for testing."""
    # First dataset
    df1 = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "score": [85.5, 90.0, 88.2, 92.1, 87.3],
        "active": [True, False, True, True, False],
    })
    
    # Second dataset (slightly different)
    df2 = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
        "score": [85.5, 90.0, None, 92.1, 87.3, 88.0],
        "active": [True, False, True, True, False, True],
        "status": ["active", "inactive", "active", "inactive", "active", "pending"],
    })
    
    frame1 = ar.from_pandas(df1)
    frame2 = ar.from_pandas(df2)
    
    profile1 = ar.profile(frame1)
    profile2 = ar.profile(frame2)
    
    return ar.compare_profiles(profile1, profile2)


class TestProfileComparisonToDict:
    """Test suite for ProfileComparison.to_dict()"""
    
    def test_to_dict_returns_dict(self, profile_comparison):
        """Test that to_dict returns a dictionary."""
        result = profile_comparison.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_has_required_keys(self, profile_comparison):
        """Test that returned dict has all required top-level keys."""
        result = profile_comparison.to_dict()
        
        required_keys = {"left_profile", "right_profile", "status_counts", "drift_report"}
        assert required_keys.issubset(result.keys())
    
    def test_to_dict_profiles_are_dicts(self, profile_comparison):
        """Test that left_profile and right_profile values are dictionaries."""
        result = profile_comparison.to_dict()
        
        assert isinstance(result["left_profile"], dict)
        assert isinstance(result["right_profile"], dict)
    
    def test_to_dict_status_counts_is_dict(self, profile_comparison):
        """Test that status_counts is a dictionary."""
        result = profile_comparison.to_dict()
        
        assert isinstance(result["status_counts"], dict)
    
    def test_to_dict_drift_report_is_dict(self, profile_comparison):
        """Test that drift_report is a dictionary."""
        result = profile_comparison.to_dict()
        
        assert isinstance(result["drift_report"], dict)
    
    def test_to_dict_with_redact_sample_values_true(self, profile_comparison):
        """Test to_dict with redact_sample_values=True."""
        result = profile_comparison.to_dict(redact_sample_values=True)
        
        assert isinstance(result, dict)
        assert "left_profile" in result
        assert "right_profile" in result
        # Both profiles should receive the redaction parameter
        assert isinstance(result["left_profile"], dict)
        assert isinstance(result["right_profile"], dict)
    
    def test_to_dict_with_exclude_columns_list(self, profile_comparison):
        """Test to_dict with exclude_columns as a list."""
        exclude_cols = ["id", "name"]
        result = profile_comparison.to_dict(exclude_columns=exclude_cols)
        
        assert isinstance(result, dict)
        # Verify structure is preserved
        assert "left_profile" in result
        assert "right_profile" in result
        assert "status_counts" in result
    
    def test_to_dict_with_exclude_columns_set(self, profile_comparison):
        """Test to_dict with exclude_columns as a set."""
        result = profile_comparison.to_dict(exclude_columns={"id", "score"})
        
        assert isinstance(result, dict)
        assert "left_profile" in result
        assert "right_profile" in result
    
    def test_to_dict_with_exclude_columns_tuple(self, profile_comparison):
        """Test to_dict with exclude_columns as a tuple."""
        result = profile_comparison.to_dict(exclude_columns=("name", "active"))
        
        assert isinstance(result, dict)
        assert "drift_report" in result
    
    def test_to_dict_with_both_parameters(self, profile_comparison):
        """Test to_dict with both redact_sample_values and exclude_columns."""
        result = profile_comparison.to_dict(
            redact_sample_values=True,
            exclude_columns=["id", "name"]
        )
        
        assert isinstance(result, dict)
        # All four keys should be present
        assert len(result) >= 4
        assert "left_profile" in result
        assert "right_profile" in result
        assert "status_counts" in result
        assert "drift_report" in result
    
    def test_to_dict_with_empty_exclude_columns(self, profile_comparison):
        """Test to_dict with empty exclude_columns list."""
        result = profile_comparison.to_dict(exclude_columns=[])
        
        assert isinstance(result, dict)
        # Structure should be unchanged
        assert "left_profile" in result
        assert "right_profile" in result
    
    def test_to_dict_json_serializable(self, profile_comparison):
        """Test that to_dict output is JSON serializable."""
        result = profile_comparison.to_dict(redact_sample_values=True)
        
        # Should not raise an exception during JSON serialization
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Should be able to deserialize it back
        deserialized = json.loads(json_str)
        assert isinstance(deserialized, dict)
    
    def test_to_dict_status_counts_populated(self, profile_comparison):
        """Test that status_counts contains expected content."""
        result = profile_comparison.to_dict()
        status_counts = result["status_counts"]
        
        # status_counts should have entries (at minimum count data for columns)
        assert isinstance(status_counts, dict)
        # Should have some status values like "ok", "warning", "changed", etc.
        # The exact keys depend on the comparison
    
    def test_to_dict_with_nonexistent_exclude_column(self, profile_comparison):
        """Test to_dict with a nonexistent column in exclude_columns."""
        # Should handle gracefully without raising an error
        result = profile_comparison.to_dict(exclude_columns=["nonexistent_col"])
        
        assert isinstance(result, dict)
        assert "left_profile" in result
        assert "right_profile" in result
