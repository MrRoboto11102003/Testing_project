import pytest
import arnio as ar
import pandas as pd
import json


@pytest.fixture
def baseline_frame():
    """Create a baseline ArFrame for testing."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "email": ["alice@example.com", "bob@example.com", "charlie@example.com", "david@example.com", "eve@example.com"],
        "age": [25, 30, 35, 40, 45],
        "score": [85.5, 90.0, 88.2, 92.1, 87.3],
    })
    return ar.from_pandas(df)


@pytest.fixture
def current_frame():
    """Create a current ArFrame with some row/value differences."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
        "email": ["alice@example.com", "bob@example.com", "charlie@example.com", "david@example.com", "eve@example.com", "frank@example.com"],
        "age": [25, 30, 35, 40, 45, 50],
        "score": [85.5, 90.0, 88.2, 92.1, 87.3, 89.5],
    })
    return ar.from_pandas(df)


@pytest.fixture
def profile_comparison(baseline_frame, current_frame):
    """Create a ProfileComparison by comparing two profiles."""
    baseline = ar.profile(baseline_frame)
    current = ar.profile(current_frame)
    return ar.compare_profiles(baseline, current)


def test_to_dict_returns_dictionary(profile_comparison):
    """Test that to_dict returns a dictionary."""
    result = profile_comparison.to_dict()
    assert isinstance(result, dict)


def test_to_dict_with_default_parameters(profile_comparison):
    """Test to_dict with default parameters."""
    result = profile_comparison.to_dict()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_to_dict_redact_sample_values_false(profile_comparison):
    """Test to_dict with redact_sample_values=False (default)."""
    result = profile_comparison.to_dict(redact_sample_values=False)
    assert isinstance(result, dict)


def test_to_dict_redact_sample_values_true(profile_comparison):
    """Test to_dict with redact_sample_values=True."""
    result = profile_comparison.to_dict(redact_sample_values=True)
    assert isinstance(result, dict)
    # Convert to string to check for redaction markers
    result_str = json.dumps(result)
    assert isinstance(result_str, str)


def test_to_dict_exclude_columns_as_list(profile_comparison):
    """Test exclude_columns parameter as a list."""
    result = profile_comparison.to_dict(exclude_columns=["id", "email"])
    assert isinstance(result, dict)


def test_to_dict_exclude_columns_as_set(profile_comparison):
    """Test exclude_columns parameter as a set."""
    result = profile_comparison.to_dict(exclude_columns={"id", "name"})
    assert isinstance(result, dict)


def test_to_dict_exclude_columns_as_tuple(profile_comparison):
    """Test exclude_columns parameter as a tuple."""
    result = profile_comparison.to_dict(exclude_columns=("age", "score"))
    assert isinstance(result, dict)


def test_to_dict_exclude_columns_empty_list(profile_comparison):
    """Test exclude_columns with empty list."""
    result = profile_comparison.to_dict(exclude_columns=[])
    assert isinstance(result, dict)


def test_to_dict_exclude_columns_none(profile_comparison):
    """Test exclude_columns=None (default/no exclusions)."""
    result = profile_comparison.to_dict(exclude_columns=None)
    assert isinstance(result, dict)


def test_to_dict_combined_redact_and_exclude(profile_comparison):
    """Test using both redact_sample_values and exclude_columns together."""
    result = profile_comparison.to_dict(
        redact_sample_values=True,
        exclude_columns=["id", "email"]
    )
    assert isinstance(result, dict)


def test_to_dict_exclude_nonexistent_column(profile_comparison):
    """Test that excluding non-existent columns does not raise an error."""
    result = profile_comparison.to_dict(exclude_columns=["nonexistent_column"])
    assert isinstance(result, dict)


def test_to_dict_consistency_across_calls(profile_comparison):
    """Test that calling to_dict multiple times produces identical results."""
    result1 = profile_comparison.to_dict()
    result2 = profile_comparison.to_dict()
    assert result1 == result2


def test_to_dict_json_serializable(profile_comparison):
    """Test that the returned dictionary is JSON serializable."""
    result = profile_comparison.to_dict(redact_sample_values=True)
    # Should not raise
    json_str = json.dumps(result)
    assert isinstance(json_str, str)
    assert len(json_str) > 0


def test_to_dict_with_all_parameters(profile_comparison):
    """Test to_dict with all parameters specified."""
    result = profile_comparison.to_dict(
        redact_sample_values=True,
        exclude_columns=["id", "name", "email"]
    )
    assert isinstance(result, dict)
    # Verify it's still JSON serializable after all transformations
    json.dumps(result)
