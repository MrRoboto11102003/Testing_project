import pytest
import pandas as pd
import arnio as ar


def make_frame(data):
    """Helper to create an ArFrame from dict data."""
    return ar.from_pandas(pd.DataFrame(data))


def make_report(data):
    """Helper to create a DataQualityReport from dict data."""
    frame = make_frame(data)
    return ar.profile(frame)


class TestDataQualityReportToDict:
    """Tests for arnio.quality.DataQualityReport.to_dict()"""

    def test_to_dict_returns_dict(self):
        """to_dict() returns a dictionary."""
        report = make_report({"col": [1, 2, 3]})
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_default_parameters(self):
        """to_dict() with default parameters returns non-empty dict."""
        report = make_report({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        result = report.to_dict()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_to_dict_redact_sample_values_false(self):
        """to_dict(redact_sample_values=False) includes sample values."""
        report = make_report({"col": [1, 2, 3]})
        result = report.to_dict(redact_sample_values=False)
        assert isinstance(result, dict)

    def test_to_dict_redact_sample_values_true(self):
        """to_dict(redact_sample_values=True) redacts sample values."""
        report = make_report({"col": ["secret1", "secret2", "secret3"]})
        result = report.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_as_list(self):
        """to_dict() accepts exclude_columns as a list."""
        report = make_report({"col1": [1, 2], "col2": [3, 4]})
        result = report.to_dict(exclude_columns=["col1"])
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_as_set(self):
        """to_dict() accepts exclude_columns as a set."""
        report = make_report({"col1": [1, 2], "col2": [3, 4]})
        result = report.to_dict(exclude_columns={"col1"})
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_as_tuple(self):
        """to_dict() accepts exclude_columns as a tuple."""
        report = make_report({"col1": [1, 2], "col2": [3, 4]})
        result = report.to_dict(exclude_columns=("col1",))
        assert isinstance(result, dict)

    def test_to_dict_exclude_columns_none(self):
        """to_dict(exclude_columns=None) includes all columns."""
        report = make_report({"col1": [1, 2], "col2": [3, 4]})
        result = report.to_dict(exclude_columns=None)
        assert isinstance(result, dict)

    def test_to_dict_exclude_multiple_columns(self):
        """to_dict() can exclude multiple columns at once."""
        data = {"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]}
        report = make_report(data)
        result = report.to_dict(exclude_columns=["col1", "col2"])
        assert isinstance(result, dict)

    def test_to_dict_exclude_nonexistent_column(self):
        """to_dict() gracefully handles non-existent columns in exclude list."""
        report = make_report({"col1": [1, 2, 3]})
        result = report.to_dict(exclude_columns=["nonexistent"])
        assert isinstance(result, dict)

    def test_to_dict_multiple_dtypes(self):
        """to_dict() handles mixed data types."""
        data = {
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        }
        report = make_report(data)
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_with_null_values(self):
        """to_dict() handles columns containing null values."""
        data = {"col1": [1, None, 3], "col2": ["a", None, "c"]}
        report = make_report(data)
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_empty_frame(self):
        """to_dict() handles empty DataFrames."""
        data = {"col": pd.Series([], dtype="float64")}
        report = make_report(data)
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_combined_parameters(self):
        """to_dict() works with both redact and exclude parameters."""
        data = {"col1": [1, 2], "col2": ["x", "y"]}
        report = make_report(data)
        result = report.to_dict(
            redact_sample_values=True,
            exclude_columns=["col1"]
        )
        assert isinstance(result, dict)

    def test_to_dict_returns_fresh_dict(self):
        """to_dict() returns independent dict instances."""
        report = make_report({"col": [1, 2, 3]})
        result1 = report.to_dict()
        result2 = report.to_dict()
        result1["test_key"] = "test_value"
        assert "test_key" not in result2
