import pytest
import pandas as pd
import numpy as np
import json
import arnio as ar


class TestProfileComparisonToDict:
    """Test arnio.quality.ProfileComparison.to_dict method."""

    @pytest.fixture
    def left_frame(self):
        """Create left frame for comparison."""
        return ar.from_pandas(
            pd.DataFrame({
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "score": [95.5, 87.3, 91.2, 88.9, 92.1],
                "active": [True, False, True, True, False],
            })
        )

    @pytest.fixture
    def right_frame(self):
        """Create right frame for comparison (slightly different)."""
        return ar.from_pandas(
            pd.DataFrame({
                "id": [1, 2, 3, 4, 5, 6],
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
                "score": [95.5, 87.3, 91.2, 88.9, 92.1, 89.7],
                "active": [True, False, True, True, False, True],
            })
        )

    @pytest.fixture
    def comparison(self, left_frame, right_frame):
        """Create a ProfileComparison from two frames."""
        left_profile = ar.profile(left_frame)
        right_profile = ar.profile(right_frame)
        # Access ProfileComparison class from arnio.quality module
        from arnio.quality import ProfileComparison
        return ProfileComparison(left_profile, right_profile)

    def test_to_dict_returns_dict(self, comparison):
        """Test that to_dict returns a dictionary."""
        result = comparison.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_required_keys(self, comparison):
        """Test that returned dict contains all required keys."""
        result = comparison.to_dict()
        required_keys = {"left_profile", "right_profile", "status_counts", "drift_report"}
        assert required_keys.issubset(result.keys())

    def test_to_dict_nested_values_are_dicts(self, comparison):
        """Test that all nested values are dictionaries."""
        result = comparison.to_dict()
        
        assert isinstance(result["left_profile"], dict)
        assert isinstance(result["right_profile"], dict)
        assert isinstance(result["status_counts"], dict)
        assert isinstance(result["drift_report"], dict)

    def test_to_dict_default_parameters(self, comparison):
        """Test to_dict with no parameters (defaults)."""
        result = comparison.to_dict()
        
        assert isinstance(result, dict)
        assert len(result) >= 4
        assert "left_profile" in result

    def test_to_dict_redact_sample_values_false(self, comparison):
        """Test to_dict with redact_sample_values=False."""
        result = comparison.to_dict(redact_sample_values=False)
        
        assert isinstance(result, dict)
        assert "left_profile" in result
        assert "right_profile" in result

    def test_to_dict_redact_sample_values_true(self, comparison):
        """Test to_dict with redact_sample_values=True."""
        result = comparison.to_dict(redact_sample_values=True)
        
        assert isinstance(result, dict)
        assert "left_profile" in result
        assert "right_profile" in result

    def test_to_dict_exclude_columns_as_list(self, comparison):
        """Test exclude_columns parameter with list."""
        result = comparison.to_dict(exclude_columns=["name"])
        
        assert isinstance(result, dict)
        assert "left_profile" in result

    def test_to_dict_exclude_columns_as_set(self, comparison):
        """Test exclude_columns parameter with set."""
        result = comparison.to_dict(exclude_columns={"name", "active"})
        
        assert isinstance(result, dict)
        assert "right_profile" in result

    def test_to_dict_exclude_columns_as_tuple(self, comparison):
        """Test exclude_columns parameter with tuple."""
        result = comparison.to_dict(exclude_columns=("name", "score", "id"))
        
        assert isinstance(result, dict)
        assert "status_counts" in result

    def test_to_dict_exclude_empty_sequence(self, comparison):
        """Test exclude_columns with empty sequence."""
        result = comparison.to_dict(exclude_columns=[])
        
        assert isinstance(result, dict)
        assert all(k in result for k in ["left_profile", "right_profile", "status_counts", "drift_report"])

    def test_to_dict_both_parameters_set(self, comparison):
        """Test to_dict with both redact_sample_values and exclude_columns."""
        result = comparison.to_dict(
            redact_sample_values=True,
            exclude_columns=["name", "id"]
        )
        
        assert isinstance(result, dict)
        assert "left_profile" in result
        assert "right_profile" in result
        assert "status_counts" in result
        assert "drift_report" in result

    def test_to_dict_result_json_serializable(self, comparison):
        """Test that returned dictionary is JSON-serializable."""
        result = comparison.to_dict()
        
        # Should not raise
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)

    def test_to_dict_exclude_nonexistent_columns(self, comparison):
        """Test that excluding non-existent columns does not raise error."""
        result = comparison.to_dict(exclude_columns=["does_not_exist", "also_missing"])
        
        assert isinstance(result, dict)
        assert "drift_report" in result

    def test_to_dict_status_counts_valid_dict(self, comparison):
        """Test that status_counts is a valid dictionary structure."""
        result = comparison.to_dict()
        
        status_counts = result["status_counts"]
        assert isinstance(status_counts, dict)
        # Should be a regular dict (converted from dict-like if needed)
        assert all(isinstance(k, str) for k in status_counts.keys())
