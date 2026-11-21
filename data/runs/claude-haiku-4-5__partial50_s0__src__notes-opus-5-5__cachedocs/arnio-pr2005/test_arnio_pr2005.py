import pytest
import json
import pandas as pd
import numpy as np
import arnio as ar


class TestProfileComparisonToDict:
    """Tests for arnio.quality.ProfileComparison.to_dict"""

    def test_basic_structure_and_key_order(self):
        """Test that to_dict returns the correct top-level structure with keys in order."""
        df = pd.DataFrame({"x": [1, 2, 3], "name": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        result = cmp.to_dict()

        assert isinstance(result, dict)
        assert list(result.keys()) == [
            "left_profile",
            "right_profile",
            "status_counts",
            "drift_report",
        ]

    def test_json_serializable(self):
        """Test that the returned dict is JSON-serializable."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        result = cmp.to_dict()

        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_identical_frames_no_drift(self):
        """Test comparing identical frames shows no drift."""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "name": ["a", "b", "c", "d"]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        result = cmp.to_dict()

        assert result["status_counts"]["ok"] == 2
        assert result["status_counts"]["warning"] == 0
        assert result["status_counts"]["changed"] == 0

        for col_name, entry in result["drift_report"].items():
            assert entry["status"] == "ok"
            assert entry["changes"] == {}
            assert len(entry["reasons"]) > 0

        assert result["left_profile"] == result["right_profile"]

    def test_redact_sample_values_true(self):
        """Test that redact_sample_values=True redacts sample values in both profiles."""
        df = pd.DataFrame({"text": ["hello", "world", "test", "data", "value"]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        result = cmp.to_dict(redact_sample_values=True)

        left_col = result["left_profile"]["columns"]["text"]
        right_col = result["right_profile"]["columns"]["text"]

        assert all(v == "[REDACTED]" for v in left_col["sample_values"])
        assert all(v == "[REDACTED]" for v in right_col["sample_values"])

    def test_redact_sample_values_false_default(self):
        """Test that redact_sample_values=False (default) preserves sample values."""
        df = pd.DataFrame({"text": ["hello", "world"]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        result = cmp.to_dict(redact_sample_values=False)
        left_col = result["left_profile"]["columns"]["text"]

        assert any(isinstance(v, str) and v != "[REDACTED]" for v in left_col["sample_values"])

    def test_exclude_columns_list(self):
        """Test that exclude_columns removes those columns from both profiles only."""
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "name": ["a", "b", "c"],
            "age": [25, 30, 35],
        })
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        result = cmp.to_dict(exclude_columns=["name"])

        assert "name" not in result["left_profile"]["columns"]
        assert "name" not in result["right_profile"]["columns"]

        assert "x" in result["left_profile"]["columns"]
        assert "age" in result["left_profile"]["columns"]

        assert result["left_profile"]["column_count"] == 3

        assert "name" in result["drift_report"]

    def test_exclude_columns_tuple(self):
        """Test that exclude_columns accepts tuples."""
        df = pd.DataFrame({"x": [1, 2, 3], "name": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        result = cmp.to_dict(exclude_columns=("name",))

        assert "name" not in result["left_profile"]["columns"]
        assert "x" in result["left_profile"]["columns"]

    def test_exclude_columns_set(self):
        """Test that exclude_columns accepts sets."""
        df = pd.DataFrame({"x": [1, 2, 3], "name": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        result = cmp.to_dict(exclude_columns={"name"})

        assert "name" not in result["left_profile"]["columns"]
        assert "x" in result["left_profile"]["columns"]

    def test_exclude_columns_empty_behaves_like_none(self):
        """Test that exclude_columns=[] or set() behaves like None."""
        df = pd.DataFrame({"x": [1, 2, 3], "name": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        result_none = cmp.to_dict(exclude_columns=None)
        result_empty = cmp.to_dict(exclude_columns=[])
        result_empty_set = cmp.to_dict(exclude_columns=set())

        assert set(result_none["left_profile"]["columns"].keys()) == set(
            result_empty["left_profile"]["columns"].keys()
        )
        assert set(result_none["left_profile"]["columns"].keys()) == set(
            result_empty_set["left_profile"]["columns"].keys()
        )

    def test_redact_sample_values_not_bool_raises_typeerror(self):
        """Test that non-bool redact_sample_values raises TypeError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        with pytest.raises(TypeError, match="redact_sample_values must be a bool"):
            cmp.to_dict(redact_sample_values=1)

        with pytest.raises(TypeError, match="redact_sample_values must be a bool"):
            cmp.to_dict(redact_sample_values="yes")

    def test_exclude_columns_invalid_type_raises_typeerror(self):
        """Test that invalid exclude_columns type raises TypeError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        with pytest.raises(
            TypeError, match="exclude_columns must be a list, tuple, set, or None"
        ):
            cmp.to_dict(exclude_columns="name")

        with pytest.raises(
            TypeError, match="exclude_columns must be a list, tuple, set, or None"
        ):
            cmp.to_dict(exclude_columns={"name": "value"})

    def test_exclude_columns_non_string_items_raises_typeerror(self):
        """Test that non-string items in exclude_columns raise TypeError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        with pytest.raises(
            TypeError, match="exclude_columns must contain only string column names"
        ):
            cmp.to_dict(exclude_columns=[1])

        with pytest.raises(
            TypeError, match="exclude_columns must contain only string column names"
        ):
            cmp.to_dict(exclude_columns=["x", 2])

    def test_exclude_columns_unknown_column_raises_keyerror(self):
        """Test that unknown column names in exclude_columns raise KeyError."""
        df = pd.DataFrame({"x": [1, 2, 3], "name": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        with pytest.raises(KeyError, match=r"Unknown exclude_columns"):
            cmp.to_dict(exclude_columns=["unknown_col"])

    def test_status_counts_is_new_dict(self):
        """Test that status_counts in result is a new dict object."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        profile = ar.profile(frame)
        cmp = ar.compare_profiles(profile, profile)

        result = cmp.to_dict()

        assert result["status_counts"] is not cmp.status_counts
        assert result["status_counts"] == dict(cmp.status_counts)
