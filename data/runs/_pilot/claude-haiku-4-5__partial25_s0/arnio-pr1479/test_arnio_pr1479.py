import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDataQualityReportToDict:
    """Test suite for DataQualityReport.to_dict() method."""

    @pytest.fixture
    def simple_frame(self):
        """Create a simple test frame."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [95.5, 87.3, 92.1, 88.9, 91.2],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def frame_with_nulls(self):
        """Create a frame with null values."""
        df = pd.DataFrame({
            "id": [1, 2, None, 4, 5],
            "name": ["Alice", None, "Charlie", None, "Eve"],
            "score": [95.5, 87.3, None, 88.9, 91.2],
        })
        return ar.from_pandas(df)

    def test_to_dict_returns_dict_type(self, simple_frame):
        """Test that to_dict() returns a dict object."""
        report = ar.profile(simple_frame)
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_returns_nonempty_dict(self, simple_frame):
        """Test that to_dict() returns non-empty dictionary."""
        report = ar.profile(simple_frame)
        result = report.to_dict()
        assert len(result) > 0

    def test_to_dict_with_redact_sample_values_false(self, simple_frame):
        """Test to_dict() with redact_sample_values=False (default)."""
        report = ar.profile(simple_frame)
        result = report.to_dict(redact_sample_values=False)
        assert isinstance(result, dict)

    def test_to_dict_with_redact_sample_values_true(self, simple_frame):
        """Test to_dict() with redact_sample_values=True."""
        report = ar.profile(simple_frame)
        result = report.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_list(self, simple_frame):
        """Test to_dict() with exclude_columns as list."""
        report = ar.profile(simple_frame)
        result = report.to_dict(exclude_columns=["name"])
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_set(self, simple_frame):
        """Test to_dict() with exclude_columns as set."""
        report = ar.profile(simple_frame)
        result = report.to_dict(exclude_columns={"name", "score"})
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_tuple(self, simple_frame):
        """Test to_dict() with exclude_columns as tuple."""
        report = ar.profile(simple_frame)
        result = report.to_dict(exclude_columns=("id", "name"))
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_none_default(self, simple_frame):
        """Test to_dict() with exclude_columns=None (default behavior)."""
        report = ar.profile(simple_frame)
        result = report.to_dict(exclude_columns=None)
        assert isinstance(result, dict)

    def test_to_dict_with_frame_containing_nulls(self, frame_with_nulls):
        """Test to_dict() on frame with null values."""
        report = ar.profile(frame_with_nulls)
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_both_parameters_together(self, simple_frame):
        """Test to_dict() with both redact_sample_values and exclude_columns."""
        report = ar.profile(simple_frame)
        result = report.to_dict(
            redact_sample_values=True,
            exclude_columns=["name", "score"]
        )
        assert isinstance(result, dict)

    def test_to_dict_exclude_nonexistent_columns(self, simple_frame):
        """Test to_dict() excluding columns that don't exist in frame."""
        report = ar.profile(simple_frame)
        result = report.to_dict(exclude_columns=["nonexistent", "fake_column"])
        assert isinstance(result, dict)

    def test_to_dict_single_column_frame(self):
        """Test to_dict() on a minimal single-column frame."""
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_with_mixed_data_types(self):
        """Test to_dict() with various column data types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_exclude_all_columns(self, simple_frame):
        """Test to_dict() when excluding all columns from the frame."""
        report = ar.profile(simple_frame)
        result = report.to_dict(exclude_columns=["id", "name", "score"])
        # Should still return a dict (possibly empty or with metadata)
        assert isinstance(result, dict)

    def test_to_dict_keyword_only_arguments(self, simple_frame):
        """Test that to_dict() enforces keyword-only arguments."""
        report = ar.profile(simple_frame)
        # Arguments must be passed as keywords, not positional
        with pytest.raises(TypeError):
            report.to_dict(False)  # type: ignore
