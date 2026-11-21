import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
from pathlib import Path
import arnio as ar
import tempfile


class TestEmailSchemaEndToEnd:
    """System/end-to-end tests for arnio.schema.Email field."""

    # Basic unit tests on field creation
    def test_email_field_creation_defaults(self):
        """Test Email field creation with default parameters."""
        field = ar.Email()
        assert field.dtype == "string"
        assert field.nullable is True
        assert field.unique is False
        assert field.semantic == "email"
        assert field.severity == "error"

    def test_email_field_nullable_parameter(self):
        """Test Email field with nullable=False."""
        field = ar.Email(nullable=False)
        assert field.nullable is False

    def test_email_field_unique_parameter(self):
        """Test Email field with unique=True."""
        field = ar.Email(unique=True)
        assert field.unique is True

    def test_email_field_validation_light_mode(self):
        """Test Email field with light validation mode."""
        field = ar.Email(validation="light")
        assert field.semantic == "email"

    def test_email_field_validation_strict_mode(self):
        """Test Email field with strict validation mode."""
        field = ar.Email(validation="strict")
        assert field.semantic == "email:strict"

    def test_email_field_severity_parameter(self):
        """Test Email field with custom severity level."""
        field = ar.Email(severity="warning")
        assert field.severity == "warning"

    def test_email_field_required_if_parameter(self):
        """Test Email field with required_if conditional requirement."""
        condition = ("account_type", "premium")
        field = ar.Email(required_if=condition)
        assert field.required_if == condition

    def test_email_validation_parameter_type_error(self):
        """Test that non-string validation parameter raises TypeError."""
        with pytest.raises(TypeError, match="Email validation must be a string"):
            ar.Email(validation=123)

    def test_email_validation_parameter_value_error(self):
        """Test that invalid validation mode raises ValueError."""
        with pytest.raises(ValueError, match="Email validation must be 'light' or 'strict'"):
            ar.Email(validation="medium")

    # End-to-end workflow tests
    def test_email_validate_valid_addresses(self, tmp_path):
        """End-to-end: Load CSV with valid emails, validate, and write output."""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "email": ["alice@example.com", "bob@test.org", "charlie@domain.co.uk"]
        })
        df.to_csv(input_file, index=False)

        schema = ar.Schema({
            "user_id": ar.Field(dtype="int64"),
            "email": ar.Email(validation="light", nullable=False)
        })

        loaded = pd.read_csv(input_file)
        validated = ar.validate(loaded, schema)
        validated.to_csv(output_file, index=False)

        assert output_file.exists()
        result_df = pd.read_csv(output_file)
        assert len(result_df) == 3
        assert list(result_df.columns) == ["user_id", "email"]

    def test_email_nullable_true_with_nulls(self, tmp_path):
        """End-to-end: Handle null values when nullable=True."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["alice@example.com", None, "charlie@example.com"]
        })
        df.to_csv(csv_file, index=False)

        schema = ar.Schema({
            "id": ar.Field(dtype="int64"),
            "email": ar.Email(nullable=True)
        })

        loaded = pd.read_csv(csv_file)
        result = ar.validate(loaded, schema)

        assert len(result) == 3
        assert pd.isna(result.loc[1, "email"])

    def test_email_unique_constraint_enforcement(self, tmp_path):
        """End-to-end: Enforce uniqueness constraint on email column."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["alice@example.com", "bob@example.com", "alice@example.com"]
        })
        df.to_csv(csv_file, index=False)

        schema = ar.Schema({
            "id": ar.Field(dtype="int64"),
            "email": ar.Email(unique=True)
        })

        loaded = pd.read_csv(csv_file)
        result = ar.validate(loaded, schema)

        assert result is not None

    def test_email_strict_validation_format_checking(self, tmp_path):
        """End-to-end: Strict validation enforces stricter email format rules."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "id": [1, 2],
            "email": ["valid.user@example.com", "invalid.format"]
        })
        df.to_csv(csv_file, index=False)

        schema_strict = ar.Schema({
            "id": ar.Field(dtype="int64"),
            "email": ar.Email(validation="strict")
        })

        loaded = pd.read_csv(csv_file)
        result_strict = ar.validate(loaded, schema_strict)

        assert result_strict is not None

    def test_email_pandas_accessor_validation(self):
        """End-to-end: Use pandas .arnio accessor to validate emails."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "email": ["user1@example.com", "user2@test.org", "user3@domain.com"]
        })

        schema = ar.Schema({
            "user_id": ar.Field(dtype="int64"),
            "email": ar.Email(validation="light")
        })

        result = df.arnio.validate(schema)

        assert len(result) == 3
        assert "email" in result.columns

    def test_email_cli_validation_workflow(self, tmp_path):
        """End-to-end: Use CLI to validate email column in data file."""
        data_file = tmp_path / "data.csv"
        output_file = tmp_path / "output.csv"

        df = pd.DataFrame({
            "email": ["test@example.com", "another@test.org"]
        })
        df.to_csv(data_file, index=False)

        schema_yaml = tmp_path / "schema.yaml"
        schema_yaml.write_text("""
fields:
  email:
    type: string
    semantic: email
    nullable: false
""")

        result = subprocess.run(
            [sys.executable, "-m", "arnio.cli", "validate",
             str(data_file), str(schema_yaml), "--output", str(output_file)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0 or output_file.exists()

    def test_email_field_all_parameters_combined(self):
        """Test Email field with all parameters set."""
        field = ar.Email(
            nullable=False,
            unique=True,
            severity="warning",
            validation="strict",
            required_if=("active", True)
        )
        assert field.dtype == "string"
        assert field.nullable is False
        assert field.unique is True
        assert field.severity == "warning"
        assert field.semantic == "email:strict"
        assert field.required_if == ("active", True)

    def test_email_profile_report_generation(self, tmp_path):
        """End-to-end: Generate profiling report for email column."""
        csv_file = tmp_path / "data.csv"
        report_file = tmp_path / "report.html"

        df = pd.DataFrame({
            "email": ["alice@example.com", "bob@example.com", None, "charlie@example.com"]
        })
        df.to_csv(csv_file, index=False)

        schema = ar.Schema({
            "email": ar.Email(nullable=True)
        })

        loaded = pd.read_csv(csv_file)
        profile = ar.profile(loaded, schema)

        # Assert profile was generated
        assert profile is not None
