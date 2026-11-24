import pytest
import json
import pandas as pd
import numpy as np
import arnio as ar


@pytest.fixture
def simple_comparison():
    """Create a ProfileComparison from two identical DataFrames."""
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "name": ["alice", "bob", "charlie", "david", "eve"],
    })
    frame = ar.from_pandas(df)
    profile = ar.profile(frame)
    return ar.compare_profiles(profile, profile)


@pytest.fixture
def different_comparison():
    """Create a ProfileComparison from two different DataFrames."""
    df1 = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "name": ["alice", "bob", "charlie", "david", "eve"],
    })
    df2 = pd.DataFrame({
        "x": [1, 2, 3, 4, 10],
        "name": ["alice", "bob", "charlie", "david", "frank"],
    })
    frame1 = ar.from_pandas(df1)
    frame2 = ar.from_pandas(df2)
    profile1 = ar.profile(frame1)
    profile2 = ar.profile(frame2)
    return ar.compare_profiles(profile1, profile2)


class TestProfileComparisonToDict:
    """Test suite for ProfileComparison.to_dict()"""

    def test_basic_structure_key_order(self, simple_comparison):
        """Test that result has correct keys in correct order."""
        result = simple_comparison.to_dict()
        assert list(result.keys()) == ["left_profile", "right_profile", "status_counts", "drift_report"]

    def test_json_serializable(self, simple_comparison):
        """Test that result is JSON-serializable."""
        result = simple_comparison.to_dict()
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_redact_sample_values_true(self, simple_comparison):
        """Test redact_sample_values=True redacts sample values."""
        result = simple_comparison.to_dict(redact_sample_values=True)
        for col_data in result["left_profile"]["columns"].values():
            if col_data.get("sample_values"):
                for sample in col_data["sample_values"]:
                    assert sample == "[REDACTED]"

    def test_exclude_columns_removes_from_profiles(self, simple_comparison):
        """Test exclude_columns removes columns from profile exports."""
        result = simple_comparison.to_dict(exclude_columns=["name"])
        assert "name" not in result["left_profile"]["columns"]
        assert "name" not in result["right_profile"]["columns"]
        assert "x" in result["left_profile"]["columns"]

    def test_exclude_columns_does_not_affect_drift_report(self, simple_comparison):
        """Test exclude_columns doesn't affect drift_report."""
        result = simple_comparison.to_dict(exclude_columns=["name"])
        assert "name" in result["drift_report"]

    def test_exclude_columns_accepts_tuple_and_set(self, simple_comparison):
        """Test exclude_columns accepts tuple and set types."""
        result_tuple = simple_comparison.to_dict(exclude_columns=("name",))
        result_set = simple_comparison.to_dict(exclude_columns={"name"})
        assert result_tuple["left_profile"]["columns"] == result_set["left_profile"]["columns"]

    def test_identical_frames_profiles_equal(self, simple_comparison):
        """Test left and right profiles are equal for identical input."""
        result = simple_comparison.to_dict()
        assert result["left_profile"] == result["right_profile"]

    def test_identical_frames_status_counts(self, simple_comparison):
        """Test status_counts structure for identical frames."""
        result = simple_comparison.to_dict()
        assert "ok" in result["status_counts"]
        assert "warning" in result["status_counts"]
        assert "changed" in result["status_counts"]

    def test_different_frames_shows_changes(self, different_comparison):
        """Test that different frames produce changes."""
        result = different_comparison.to_dict()
        has_changes = any(
            entry["status"] != "ok" or entry["changes"]
            for entry in result["drift_report"].values()
        )
        assert has_changes

    def test_error_redact_sample_values_not_bool(self, simple_comparison):
        """Test TypeError when redact_sample_values is not a bool."""
        with pytest.raises(TypeError, match="redact_sample_values must be a bool"):
            simple_comparison.to_dict(redact_sample_values=1)
        with pytest.raises(TypeError, match="redact_sample_values must be a bool"):
            simple_comparison.to_dict(redact_sample_values="yes")

    def test_error_exclude_columns_wrong_type(self, simple_comparison):
        """Test TypeError when exclude_columns has wrong type."""
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            simple_comparison.to_dict(exclude_columns="name")

    def test_error_exclude_columns_non_string_item(self, simple_comparison):
        """Test TypeError when exclude_columns contains non-string items."""
        with pytest.raises(TypeError, match="exclude_columns must contain only string column names"):
            simple_comparison.to_dict(exclude_columns=[1])

    def test_error_exclude_columns_unknown_column(self, simple_comparison):
        """Test KeyError for unknown column in exclude_columns."""
        with pytest.raises(KeyError, match=r"Unknown exclude_columns"):
            simple_comparison.to_dict(exclude_columns=["unknown"])

    def test_delegation_to_profiles(self, simple_comparison):
        """Test that result correctly delegates to profile.to_dict()."""
        result = simple_comparison.to_dict(redact_sample_values=True, exclude_columns=["name"])
        expected_left = simple_comparison.left_profile.to_dict(
            redact_sample_values=True, exclude_columns=["name"]
        )
        expected_right = simple_comparison.right_profile.to_dict(
            redact_sample_values=True, exclude_columns=["name"]
        )
        assert result["left_profile"] == expected_left
        assert result["right_profile"] == expected_right

    def test_numeric_vs_string_columns(self, simple_comparison):
        """Test quantile fields in numeric but not string columns."""
        result = simple_comparison.to_dict()
        assert "q25" in result["left_profile"]["columns"]["x"]
        assert "q25" not in result["left_profile"]["columns"]["name"]
