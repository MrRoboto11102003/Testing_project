import pytest
import pandas as pd
import arnio as ar
import json


class TestProfileComparisonToDict:
    """Test suite for arnio.quality.ProfileComparison.to_dict method."""

    @pytest.fixture
    def baseline_frame(self):
        """Create a baseline ArFrame for profiling."""
        return ar.from_pandas(
            pd.DataFrame({
                "user_id": [1, 2, 3, 4, 5],
                "email": [
                    "alice@example.com",
                    "bob@example.com",
                    "charlie@example.com",
                    "david@example.com",
                    "eve@example.com",
                ],
                "score": [95.5, 87.3, 91.2, 88.9, 92.1],
                "active": [True, False, True, True, False],
            })
        )

    @pytest.fixture
    def current_frame(self):
        """Create a current ArFrame with slight differences."""
        return ar.from_pandas(
            pd.DataFrame({
                "user_id": [1, 2, 3, 4, 5, 6],
                "email": [
                    "alice@example.com",
                    "bob@example.com",
                    "charlie@example.com",
                    "david@example.com",
                    "eve@example.com",
                    "frank@example.com",
                ],
                "score": [95.5, 87.3, 91.2, 88.9, 92.1, 89.0],
                "active": [True, False, True, True, False, True],
            })
        )

    @pytest.fixture
    def profile_comparison(self, baseline_frame, current_frame):
        """Create a ProfileComparison object from two profiles."""
        baseline_report = ar.profile(baseline_frame)
        current_report = ar.profile(current_frame)

        # Import and instantiate ProfileComparison
        from arnio.quality import ProfileComparison

        return ProfileComparison(baseline_report, current_report)

    def test_to_dict_method_exists(self, profile_comparison):
        """Test that to_dict method exists and is callable."""
        assert hasattr(profile_comparison, "to_dict")
        assert callable(profile_comparison.to_dict)

    def test_to_dict_returns_dict(self, profile_comparison):
        """Test that to_dict returns a dictionary."""
        result = profile_comparison.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_default_parameters(self, profile_comparison):
        """Test to_dict with all default parameters."""
        result = profile_comparison.to_dict()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_to_dict_redact_sample_values_false(self, profile_comparison):
        """Test to_dict with redact_sample_values=False (default)."""
        result = profile_comparison.to_dict(redact_sample_values=False)
        assert isinstance(result, dict)

    def test_to_dict_redact_sample_values_true(self, profile_comparison):
        """Test to_dict with redact_sample_values=True."""
        result = profile_comparison.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)
        # Verify [REDACTED] marker is present in string representation
        result_json = json.dumps(result, default=str)
        assert "[REDACTED]" in result_json

    def test_to_dict_exclude_columns_empty_list(self, profile_comparison):
        """Test to_dict with empty exclude_columns list."""
        result = profile_comparison.to_dict(exclude_columns=[])
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_single_column_list(self, profile_comparison):
        """Test to_dict excluding a single column as list."""
        result = profile_comparison.to_dict(exclude_columns=["user_id"])
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_multiple_list(self, profile_comparison):
        """Test to_dict excluding multiple columns as list."""
        result = profile_comparison.to_dict(exclude_columns=["user_id", "email"])
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_set(self, profile_comparison):
        """Test to_dict with exclude_columns as a set."""
        result = profile_comparison.to_dict(exclude_columns={"user_id", "score"})
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_tuple(self, profile_comparison):
        """Test to_dict with exclude_columns as a tuple."""
        result = profile_comparison.to_dict(exclude_columns=("user_id", "active"))
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_none(self, profile_comparison):
        """Test to_dict with exclude_columns=None (default)."""
        result = profile_comparison.to_dict(exclude_columns=None)
        assert isinstance(result, dict)

    def test_to_dict_redact_and_exclude_combined(self, profile_comparison):
        """Test to_dict with both redact_sample_values=True and exclude_columns."""
        result = profile_comparison.to_dict(
            redact_sample_values=True, exclude_columns=["user_id", "email"]
        )
        assert isinstance(result, dict)
        result_json = json.dumps(result, default=str)
        assert "[REDACTED]" in result_json

    def test_to_dict_result_is_json_serializable(self, profile_comparison):
        """Test that the returned dict is JSON-serializable."""
        result = profile_comparison.to_dict()
        # Should not raise TypeError
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)
        # Verify it can be parsed back
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_to_dict_all_columns_excluded(self, profile_comparison):
        """Test to_dict excluding all columns."""
        all_cols = ["user_id", "email", "score", "active"]
        result = profile_comparison.to_dict(exclude_columns=all_cols)
        assert isinstance(result, dict)
