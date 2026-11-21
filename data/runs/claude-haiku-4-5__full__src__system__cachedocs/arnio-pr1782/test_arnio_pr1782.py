"""
End-to-end tests for arnio.schema.Email field validator.

Tests cover the complete user workflow: loading CSV data, creating schemas
with Email fields, validating data, and inspecting results.
"""

import pytest
import pandas as pd
import arnio as ar
from pathlib import Path


class TestEmailValidationBasics:
    """Core email validation functionality with light and strict modes."""

    def test_valid_emails_pass_light_validation(self, tmp_path):
        """Valid email addresses should pass light validation."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\nalice@example.com\nbob@example.org\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email(validation="light")})
        result = ar.validate(frame, schema)

        assert result.passed
        assert result.issue_count == 0

    def test_invalid_emails_fail_validation(self, tmp_path):
        """Invalid email addresses should fail validation."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\nalice@example.com\nnot-an-email\nbob@example.org\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)

        assert not result.passed
        email_issues = [i for i in result.issues if i.column == "email"]
        assert len(email_issues) > 0
        # Verify 1-based row indexing: invalid email is row 2
        assert any(issue.row_index == 2 for issue in email_issues)

    def test_strict_mode_validation(self, tmp_path):
        """Strict validation mode should enforce stricter email format rules."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\nalice@example.com\nbob@example.org\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email(validation="strict")})
        result = ar.validate(frame, schema)

        # Valid emails should pass strict mode too
        assert result.passed

    def test_whitespace_in_email_fails_validation(self, tmp_path):
        """Emails with whitespace should fail validation."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\nalice@example.com\n  invalid@test.com  \n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)

        assert not result.passed


class TestEmailNullableConstraint:
    """Test nullable behavior with Email schema."""

    def test_null_emails_allowed_by_default(self, tmp_path):
        """Null values should be allowed when nullable=True (default)."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\nalice@example.com\n\nbob@example.org\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email(nullable=True)})
        result = ar.validate(frame, schema)

        assert result.passed

    def test_null_emails_rejected_when_not_nullable(self, tmp_path):
        """Null values should be rejected when nullable=False."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\nalice@example.com\n\nbob@example.org\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame, schema)

        assert not result.passed
        email_issues = [i for i in result.issues if i.column == "email"]
        assert len(email_issues) > 0
        # Null is in row 2
        assert any(issue.row_index == 2 for issue in email_issues)


class TestEmailUniquenessConstraint:
    """Test unique constraint with Email schema."""

    def test_all_unique_emails_pass_validation(self, tmp_path):
        """All unique emails should pass unique constraint."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text(
            "email\nalice@example.com\nbob@example.org\ncharlie@example.net\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email(unique=True)})
        result = ar.validate(frame, schema)

        assert result.passed

    def test_duplicate_emails_fail_unique_constraint(self, tmp_path):
        """Duplicate emails should fail unique constraint."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text(
            "email\nalice@example.com\nbob@example.org\nalice@example.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email(unique=True)})
        result = ar.validate(frame, schema)

        assert not result.passed
        unique_issues = [i for i in result.issues if "unique" in i.rule.lower()]
        assert len(unique_issues) > 0


class TestEmailConditionalRequired:
    """Test required_if conditional requirement with Email schema."""

    def test_email_required_when_condition_met(self, tmp_path):
        """Email should be required when conditional requirement is met."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("user_type,email\npremium,alice@example.com\nfree,\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {
                "user_type": ar.String(),
                "email": ar.Email(required_if=("user_type", "premium")),
            }
        )
        result = ar.validate(frame, schema)

        # First row: premium user with email (OK)
        # Second row: free user without email (OK, not required)
        assert result.passed

    def test_missing_email_fails_conditional_requirement(self, tmp_path):
        """Email should fail when condition is met but email is null."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("user_type,email\npremium,\nfree,optional@example.com\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {
                "user_type": ar.String(),
                "email": ar.Email(required_if=("user_type", "premium")),
            }
        )
        result = ar.validate(frame, schema)

        assert not result.passed
        # Row 1 has premium with no email
        email_issues = [i for i in result.issues if i.column == "email"]
        assert len(email_issues) > 0


class TestEmailSeverityLevels:
    """Test warning vs error severity with Email schema."""

    def test_warning_severity_does_not_fail_overall(self, tmp_path):
        """Warning severity should not fail overall validation."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\ninvalid-email\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email(severity="warning")})
        result = ar.validate(frame, schema)

        # With warning severity, validation should still pass
        assert result.passed

    def test_error_severity_fails_on_invalid_email(self, tmp_path):
        """Error severity should fail validation on invalid email."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\ninvalid-email\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email(severity="error")})
        result = ar.validate(frame, schema)

        assert not result.passed


class TestEmailParameterValidation:
    """Test Email factory parameter validation."""

    def test_invalid_validation_mode_raises_value_error(self):
        """Invalid validation mode should raise ValueError."""
        with pytest.raises(ValueError, match="light.*strict"):
            ar.Email(validation="invalid_mode")

    def test_non_string_validation_raises_type_error(self):
        """Non-string validation parameter should raise TypeError."""
        with pytest.raises(TypeError, match="validation must be a string"):
            ar.Email(validation=123)

    def test_valid_parameters_create_field(self):
        """Valid Email parameters should create a Field without error."""
        field1 = ar.Email(validation="light", nullable=False, unique=True)
        field2 = ar.Email(validation="strict", nullable=True, unique=False)
        assert field1 is not None
        assert field2 is not None


class TestEmailMultiColumnSchema:
    """Test Email field within multi-column schemas."""

    def test_email_validation_with_other_field_types(self, tmp_path):
        """Email field should work alongside other field types."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,email,active\n1,alice@example.com,true\n2,bob@example.org,false\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {
                "id": ar.Int64(),
                "email": ar.Email(),
                "active": ar.Bool(),
            }
        )
        result = ar.validate(frame, schema)

        assert result.passed

    def test_validation_report_identifies_correct_column(self, tmp_path):
        """Validation report should identify email column in failures."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,email\n1,alice@example.com\n2,invalid\n3,bob@example.org\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"id": ar.Int64(), "email": ar.Email()})
        result = ar.validate(frame, schema)

        assert not result.passed
        # Check that issues are attributed to email column
        email_issues = [i for i in result.issues if i.column == "email"]
        assert len(email_issues) > 0
        assert any(issue.row_index == 2 for issue in email_issues)


class TestEmailWithPandasAccessor:
    """Test Email validation through pandas .arnio accessor."""

    def test_pandas_accessor_validate_with_email_schema(self):
        """Email validation should work through pandas accessor."""
        df = pd.DataFrame(
            {"email": ["alice@example.com", "bob@example.org", "charlie@example.net"]}
        )

        schema = ar.Schema({"email": ar.Email()})
        result = df.arnio.validate(schema)

        assert result.passed

    def test_pandas_accessor_validate_invalid_emails(self):
        """Invalid emails should fail through pandas accessor."""
        df = pd.DataFrame(
            {"email": ["alice@example.com", "not-an-email", "bob@example.org"]}
        )

        schema = ar.Schema({"email": ar.Email()})
        result = df.arnio.validate(schema)

        assert not result.passed
        email_issues = [i for i in result.issues if i.column == "email"]
        assert len(email_issues) > 0


class TestEmailRoundTrip:
    """Test Email validation with data round-tripping."""

    def test_valid_data_roundtrip_preserves_emails(self, tmp_path):
        """Valid email data should round-trip through pandas cleanly."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\nalice@example.com\nbob@example.org\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)

        # Validate before round-trip
        assert result.passed

        # Convert to pandas and back
        df = ar.to_pandas(frame)
        assert len(df) == 2
        assert df["email"].tolist() == ["alice@example.com", "bob@example.org"]

        # Validate again after round-trip
        frame2 = ar.from_pandas(df)
        result2 = ar.validate(frame2, schema)
        assert result2.passed

    def test_validation_error_messages_are_informative(self, tmp_path):
        """Validation error messages should be clear and informative."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\nalice@example.com\nbad@email\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)

        assert not result.passed
        # Check that error messages are non-empty and descriptive
        for issue in result.issues:
            assert issue.message
            assert len(issue.message) > 0


class TestEmailSpecialCases:
    """Test special cases and boundary conditions."""

    def test_emails_with_plus_addressing(self, tmp_path):
        """Emails with plus addressing should pass validation."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text(
            "email\nalice+tag@example.com\nbob.smith+2024@sub.example.org\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)

        assert result.passed

    def test_empty_csv_header_only(self, tmp_path):
        """CSV with only header (no data rows) should pass."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)

        assert result.passed
        assert result.issue_count == 0

    def test_validation_result_summary_includes_email_issues(self, tmp_path):
        """ValidationResult summary should include email-specific issues."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\nalice@example.com\nbad-email\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)

        assert not result.passed
        summary = result.summary()
        # Summary should contain information about failures
        assert "issues" in summary or result.issue_count > 0
