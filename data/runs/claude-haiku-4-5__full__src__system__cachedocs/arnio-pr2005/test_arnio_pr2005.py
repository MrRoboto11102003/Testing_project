import pytest
import pandas as pd
import arnio as ar
import json


class TestProfileComparisonToDict:
    """System tests for ProfileComparison.to_dict() with real file workflows"""

    @pytest.fixture
    def sample_csv_files(self, tmp_path):
        """Create realistic sample CSV files for testing"""
        # Baseline CSV
        baseline_path = tmp_path / "baseline.csv"
        baseline_df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com",
                      "diana@example.com", "eve@example.com"],
            "age": [25, 30, 35, 28, 32],
            "salary": [50000.0, 60000.0, 75000.0, 65000.0, 72000.0],
        })
        baseline_df.to_csv(baseline_path, index=False)

        # Current CSV (slightly different - more rows, different values)
        current_path = tmp_path / "current.csv"
        current_df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5, 6, 7],
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com",
                      "diana@example.com", "eve@example.com", "frank@example.com", "grace@example.com"],
            "age": [25, 31, 35, 28, 33, 29, 40],
            "salary": [51000.0, 62000.0, 76000.0, 66000.0, 74000.0, 61000.0, 80000.0],
        })
        current_df.to_csv(current_path, index=False)

        return baseline_path, current_path

    def test_basic_to_dict_from_csv_files(self, sample_csv_files):
        """Test basic to_dict workflow loading profiles from CSV files"""
        baseline_path, current_path = sample_csv_files

        # Load CSVs via arnio
        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        # Profile them
        baseline_report = ar.profile(baseline_frame)
        current_report = ar.profile(current_frame)

        # Compare and convert to dict
        comparison = ar.compare_profiles(baseline_report, current_report)
        result = comparison.to_dict()

        # Verify structure
        assert isinstance(result, dict)
        assert "left_profile" in result
        assert "right_profile" in result
        assert "status_counts" in result
        assert "drift_report" in result
        assert isinstance(result["status_counts"], dict)

    def test_to_dict_json_roundtrip(self, sample_csv_files):
        """Test that to_dict output can be serialized and deserialized"""
        baseline_path, current_path = sample_csv_files

        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        baseline_report = ar.profile(baseline_frame)
        current_report = ar.profile(current_frame)

        comparison = ar.compare_profiles(baseline_report, current_report)
        result = comparison.to_dict()

        # Serialize to JSON
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Deserialize back
        restored = json.loads(json_str)

        # Verify structure is preserved
        assert set(restored.keys()) == set(result.keys())
        assert "left_profile" in restored
        assert "right_profile" in restored

    def test_to_dict_with_redact_sample_values_true(self, sample_csv_files):
        """Test that redact_sample_values=True produces valid redacted dict"""
        baseline_path, current_path = sample_csv_files

        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        baseline_report = ar.profile(baseline_frame, sample_size=5)
        current_report = ar.profile(current_frame, sample_size=5)

        comparison = ar.compare_profiles(baseline_report, current_report)

        # Get redacted version
        result_redacted = comparison.to_dict(redact_sample_values=True)

        # Verify it's still a valid dict structure
        assert isinstance(result_redacted["left_profile"], dict)
        assert isinstance(result_redacted["right_profile"], dict)
        assert "status_counts" in result_redacted

        # Verify it's JSON serializable
        json_str = json.dumps(result_redacted)
        assert isinstance(json_str, str)

    def test_to_dict_exclude_single_column(self, sample_csv_files):
        """Test excluding a single column from comparison dict"""
        baseline_path, current_path = sample_csv_files

        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        baseline_report = ar.profile(baseline_frame)
        current_report = ar.profile(current_frame)

        comparison = ar.compare_profiles(baseline_report, current_report)
        result = comparison.to_dict(exclude_columns=["email"])

        # Email column should be excluded from both profiles
        left_cols = result["left_profile"].get("columns", {})
        right_cols = result["right_profile"].get("columns", {})

        assert "email" not in left_cols
        assert "email" not in right_cols

        # Other columns should be present
        assert "user_id" in left_cols
        assert "age" in left_cols
        assert "salary" in left_cols

    def test_to_dict_exclude_multiple_columns_as_list(self, sample_csv_files):
        """Test excluding multiple columns as a list"""
        baseline_path, current_path = sample_csv_files

        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        baseline_report = ar.profile(baseline_frame)
        current_report = ar.profile(current_frame)

        comparison = ar.compare_profiles(baseline_report, current_report)
        result = comparison.to_dict(exclude_columns=["email", "salary"])

        left_cols = result["left_profile"].get("columns", {})

        assert "email" not in left_cols
        assert "salary" not in left_cols
        assert "user_id" in left_cols
        assert "age" in left_cols

    def test_to_dict_exclude_columns_as_set(self, sample_csv_files):
        """Test that exclude_columns accepts set type"""
        baseline_path, current_path = sample_csv_files

        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        baseline_report = ar.profile(baseline_frame)
        current_report = ar.profile(current_frame)

        comparison = ar.compare_profiles(baseline_report, current_report)
        result = comparison.to_dict(exclude_columns={"email"})

        left_cols = result["left_profile"].get("columns", {})
        assert "email" not in left_cols
        assert "user_id" in left_cols

    def test_to_dict_exclude_columns_as_tuple(self, sample_csv_files):
        """Test that exclude_columns accepts tuple type"""
        baseline_path, current_path = sample_csv_files

        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        baseline_report = ar.profile(baseline_frame)
        current_report = ar.profile(current_frame)

        comparison = ar.compare_profiles(baseline_report, current_report)
        result = comparison.to_dict(exclude_columns=("email", "age"))

        left_cols = result["left_profile"].get("columns", {})
        assert "email" not in left_cols
        assert "age" not in left_cols
        assert "user_id" in left_cols
        assert "salary" in left_cols

    def test_to_dict_redact_and_exclude_combined(self, sample_csv_files):
        """Test using both redact_sample_values and exclude_columns together"""
        baseline_path, current_path = sample_csv_files

        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        baseline_report = ar.profile(baseline_frame, sample_size=3)
        current_report = ar.profile(current_frame, sample_size=3)

        comparison = ar.compare_profiles(baseline_report, current_report)
        result = comparison.to_dict(
            redact_sample_values=True,
            exclude_columns=["email"]
        )

        # Verify exclusion in both profiles
        left_cols = result["left_profile"].get("columns", {})
        right_cols = result["right_profile"].get("columns", {})
        assert "email" not in left_cols
        assert "email" not in right_cols

        # Verify it's still a valid dict structure
        assert isinstance(result["left_profile"], dict)
        assert isinstance(result["right_profile"], dict)
        assert isinstance(result["status_counts"], dict)

    def test_to_dict_with_drift_report(self, sample_csv_files):
        """Test that drift_report is populated in to_dict output"""
        baseline_path, current_path = sample_csv_files

        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        baseline_report = ar.profile(baseline_frame)
        current_report = ar.profile(current_frame)

        comparison = ar.compare_profiles(baseline_report, current_report)
        result = comparison.to_dict()

        # drift_report should exist and be a dict
        assert "drift_report" in result
        assert isinstance(result["drift_report"], dict)

        # If there are entries, they should have valid structure
        for col_name, entry in result["drift_report"].items():
            assert isinstance(col_name, str)
            assert isinstance(entry, dict)

    def test_to_dict_status_counts_valid(self, sample_csv_files):
        """Test that status_counts contains valid integer values"""
        baseline_path, current_path = sample_csv_files

        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        baseline_report = ar.profile(baseline_frame)
        current_report = ar.profile(current_frame)

        comparison = ar.compare_profiles(baseline_report, current_report)
        result = comparison.to_dict()

        # status_counts should have valid structure
        status_counts = result["status_counts"]
        assert isinstance(status_counts, dict)

        # All values should be non-negative integers
        for status_name, count in status_counts.items():
            assert isinstance(status_name, str)
            assert isinstance(count, int)
            assert count >= 0

    def test_to_dict_empty_exclude_columns(self, sample_csv_files):
        """Test that empty exclude_columns list includes all columns"""
        baseline_path, current_path = sample_csv_files

        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        baseline_report = ar.profile(baseline_frame)
        current_report = ar.profile(current_frame)

        comparison = ar.compare_profiles(baseline_report, current_report)
        result = comparison.to_dict(exclude_columns=[])

        left_cols = result["left_profile"].get("columns", {})

        # All columns should be present
        assert "user_id" in left_cols
        assert "email" in left_cols
        assert "age" in left_cols
        assert "salary" in left_cols

    def test_to_dict_with_null_values(self, tmp_path):
        """Test to_dict with data containing null values"""
        # Create CSV with nulls
        baseline_path = tmp_path / "baseline_nulls.csv"
        baseline_df = pd.DataFrame({
            "id": [1, 2, 3, None, 5],
            "name": ["Alice", None, "Charlie", "Diana", None],
            "score": [100.0, 200.0, None, 400.0, 500.0],
        })
        baseline_df.to_csv(baseline_path, index=False)

        current_path = tmp_path / "current_nulls.csv"
        current_df = pd.DataFrame({
            "id": [1, 2, None, 4, 5, 6],
            "name": ["Alice", None, "Charlie", "Diana", "Eve", None],
            "score": [100.0, 200.0, 300.0, None, 500.0, 600.0],
        })
        current_df.to_csv(current_path, index=False)

        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        baseline_report = ar.profile(baseline_frame)
        current_report = ar.profile(current_frame)

        comparison = ar.compare_profiles(baseline_report, current_report)
        result = comparison.to_dict()

        # Should handle nulls without error
        assert isinstance(result, dict)
        assert "left_profile" in result
        assert "right_profile" in result

        # Should be JSON serializable even with null handling
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_to_dict_preserves_profile_structure(self, sample_csv_files):
        """Test that to_dict preserves nested profile dict structures"""
        baseline_path, current_path = sample_csv_files

        baseline_frame = ar.read_csv(str(baseline_path))
        current_frame = ar.read_csv(str(current_path))

        baseline_report = ar.profile(baseline_frame)
        current_report = ar.profile(current_frame)

        comparison = ar.compare_profiles(baseline_report, current_report)
        result = comparison.to_dict()

        # Nested profiles should have dict structure
        left_profile = result["left_profile"]
        right_profile = result["right_profile"]

        assert isinstance(left_profile, dict)
        assert isinstance(right_profile, dict)

        # drift_report should contain column status entries
        drift_report = result["drift_report"]
        assert isinstance(drift_report, dict)
