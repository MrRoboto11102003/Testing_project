"""
End-to-end pytest tests for arnio.quality.DataQualityReport.to_dict()

These tests exercise the public API: creating CSV files, loading them with arnio,
profiling to get a DataQualityReport, and calling to_dict() with various parameter
combinations. They verify JSON serialization, error handling, and core functionality.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import arnio as ar


class TestDataQualityReportToDict:
    """End-to-end tests for DataQualityReport.to_dict() public API"""

    def test_basic_to_dict_workflow(self, tmp_path):
        """Basic workflow: create CSV → profile → to_dict()"""
        csv_file = tmp_path / "customers.csv"
        csv_file.write_text(
            "customer_id,name,email,signup_date\n"
            "1,Alice Smith,alice@example.com,2024-01-15\n"
            "2,Bob Jones,bob@example.com,2024-02-20\n"
            "3,Charlie Brown,charlie@example.com,2024-03-10\n"
        )

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert report_dict["row_count"] == 3
        assert report_dict["column_count"] == 4
        assert "quality_score" in report_dict
        assert "columns" in report_dict
        assert all(
            col in report_dict["columns"]
            for col in ["customer_id", "name", "email", "signup_date"]
        )

    def test_to_dict_json_roundtrip(self, tmp_path):
        """Verify to_dict() output is JSON-serializable and can roundtrip"""
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "product_id,price,quantity\n"
            "1,29.99,100\n"
            "2,19.99,50\n"
            "3,9.99,200\n"
        )

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        report_dict = report.to_dict()

        # Should be JSON-serializable using default=str for special types
        json_str = json.dumps(report_dict, default=str)
        assert isinstance(json_str, str)

        # Should deserialize correctly
        restored = json.loads(json_str)
        assert restored["row_count"] == 3
        assert restored["column_count"] == 3
        assert "product_id" in restored["columns"]

    def test_exclude_columns_single(self, tmp_path):
        """Exclude a single column from to_dict() output"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "user_id,username,password_hash,email\n"
            "1,alice,secret123,alice@test.com\n"
            "2,bob,secret456,bob@test.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        report_dict = report.to_dict(exclude_columns=["password_hash"])

        assert "user_id" in report_dict["columns"]
        assert "username" in report_dict["columns"]
        assert "email" in report_dict["columns"]
        assert "password_hash" not in report_dict["columns"]

    def test_exclude_columns_multiple(self, tmp_path):
        """Exclude multiple columns from to_dict() output"""
        csv_file = tmp_path / "pii.csv"
        csv_file.write_text(
            "id,first_name,last_name,ssn,phone,email\n"
            "1,John,Doe,123-45-6789,555-1234,john@example.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        report_dict = report.to_dict(exclude_columns=["ssn", "phone", "email"])

        assert "id" in report_dict["columns"]
        assert "first_name" in report_dict["columns"]
        assert "last_name" in report_dict["columns"]
        assert len(report_dict["columns"]) == 3
        for excluded in ["ssn", "phone", "email"]:
            assert excluded not in report_dict["columns"]

    def test_exclude_columns_as_set(self, tmp_path):
        """exclude_columns parameter accepts set type"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c,d\n1,2,3,4\n5,6,7,8\n")

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns={"b", "d"})

        assert "a" in result["columns"]
        assert "c" in result["columns"]
        assert "b" not in result["columns"]
        assert "d" not in result["columns"]

    def test_exclude_columns_as_tuple(self, tmp_path):
        """exclude_columns parameter accepts tuple type"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\nval1,val2,val3\n")

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=("col2",))

        assert "col1" in result["columns"]
        assert "col3" in result["columns"]
        assert "col2" not in result["columns"]

    def test_redact_sample_values_true(self, tmp_path):
        """redact_sample_values=True redacts sensitive data in output"""
        csv_file = tmp_path / "contacts.csv"
        csv_file.write_text(
            "email,phone\n"
            "alice@test.com,555-1111\n"
            "bob@test.com,555-2222\n"
        )

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)

        result_redacted = report.to_dict(redact_sample_values=True)

        # Verify sample values are redacted if present
        for col_name, col_profile in result_redacted["columns"].items():
            if "sample_values" in col_profile and col_profile["sample_values"]:
                for val in col_profile.get("sample_values", []):
                    assert val == "[REDACTED]" or val is None

    def test_redact_and_exclude_together(self, tmp_path):
        """Combine redaction and column exclusion in single call"""
        csv_file = tmp_path / "mixed_pii.csv"
        csv_file.write_text(
            "id,name,email,ssn\n"
            "1,Alice,alice@test.com,111-11-1111\n"
            "2,Bob,bob@test.com,222-22-2222\n"
        )

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)

        result = report.to_dict(redact_sample_values=True, exclude_columns=["ssn"])

        # ssn should be excluded entirely
        assert "ssn" not in result["columns"]
        # Other columns should be present
        assert "id" in result["columns"]
        assert "name" in result["columns"]
        assert "email" in result["columns"]

    def test_error_exclude_nonexistent_column(self, tmp_path):
        """Excluding a nonexistent column raises KeyError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n")

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)

        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            report.to_dict(exclude_columns=["nonexistent"])

    def test_error_exclude_columns_wrong_type(self, tmp_path):
        """exclude_columns with string instead of list/set/tuple raises TypeError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)

        with pytest.raises(TypeError):
            report.to_dict(exclude_columns="a")

    def test_error_exclude_columns_non_string_items(self, tmp_path):
        """exclude_columns with non-string items raises TypeError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)

        with pytest.raises(TypeError):
            report.to_dict(exclude_columns=["a", 123])

    def test_error_redact_invalid_type(self, tmp_path):
        """redact_sample_values with non-bool type raises TypeError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col\nval\n")

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)

        with pytest.raises(TypeError):
            report.to_dict(redact_sample_values="true")

    def test_core_metrics_preserved_with_exclusions(self, tmp_path):
        """Core report metrics remain consistent regardless of column exclusions"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c,d\n1,2,3,4\n1,2,3,4\n5,6,7,8\n")

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)

        result_full = report.to_dict()
        result_partial = report.to_dict(exclude_columns=["a", "b", "c"])

        # Overall metrics must stay the same
        assert result_full["row_count"] == result_partial["row_count"] == 3
        assert result_full["column_count"] == result_partial["column_count"] == 4
        assert result_full["duplicate_ratio"] == result_partial["duplicate_ratio"]
        assert (
            result_full["quality_score"] == result_partial["quality_score"]
        )

    def test_empty_exclude_columns_list(self, tmp_path):
        """Empty exclude_columns list behaves identically to no exclusion"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y,z\n1,2,3\n")

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)

        result_no_exclude = report.to_dict()
        result_empty_exclude = report.to_dict(exclude_columns=[])

        # Columns should be identical
        assert (
            result_no_exclude["columns"].keys()
            == result_empty_exclude["columns"].keys()
        )
        assert result_no_exclude["row_count"] == result_empty_exclude["row_count"]
        assert (
            result_no_exclude["quality_score"]
            == result_empty_exclude["quality_score"]
        )

    def test_to_dict_with_missing_values(self, tmp_path):
        """to_dict() handles DataFrames with many missing values correctly"""
        csv_file = tmp_path / "sparse.csv"
        csv_file.write_text(
            "col_a,col_b,col_c\n"
            "1,,\n"
            ",,2\n"
            ",,\n"
            "4,5,6\n"
        )

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        result = report.to_dict()

        # Should handle nulls gracefully
        assert result["row_count"] == 4
        assert "col_a" in result["columns"]
        assert "col_b" in result["columns"]
        assert "col_c" in result["columns"]
        # Null counts should be reported
        assert result["columns"]["col_a"].get("null_count", 0) >= 0
        assert result["columns"]["col_b"].get("null_count", 0) >= 0
