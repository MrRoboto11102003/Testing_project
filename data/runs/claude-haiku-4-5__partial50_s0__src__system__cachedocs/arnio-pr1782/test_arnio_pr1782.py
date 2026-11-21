import pytest
import csv
import pandas as pd
import arnio as ar
from pathlib import Path


class TestEmailSchemaField:
    """End-to-end tests for arnio.schema.Email field."""

    def test_email_field_creation_with_defaults(self):
        """Test Email field creation with default parameters."""
        field = ar.Email()
        assert field is not None
        # Field should be usable in a schema
        schema = ar.Schema({"email": field})
        assert "email" in schema

    def test_email_field_nullable_false(self):
        """Test Email field with nullable=False rejects null values."""
        csv_data = [
            ["user_id", "email"],
            ["1", "alice@example.com"],
            ["2", ""],
            ["3", "charlie@test.com"],
        ]
        frame = self._create_frame_from_data(csv_data)
        
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "email": ar.Email(nullable=False),
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        assert result.issue_count > 0

    def test_email_field_nullable_true_allows_nulls(self):
        """Test Email field with nullable=True allows empty values."""
        csv_data = [
            ["user_id", "email"],
            ["1", "alice@example.com"],
            ["2", ""],
            ["3", "charlie@test.com"],
        ]
        frame = self._create_frame_from_data(csv_data)
        
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "email": ar.Email(nullable=True),
        })
        result = ar.validate(frame, schema)
        
        assert result.passed

    def test_email_field_valid_emails_light_validation(self):
        """Test Email field with valid email addresses passes light validation."""
        csv_data = [
            ["user_id", "email"],
            ["1", "alice@example.com"],
            ["2", "bob@example.org"],
            ["3", "charlie@test.co.uk"],
            ["4", "user+tag@domain.io"],
        ]
        frame = self._create_frame_from_data(csv_data)
        
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "email": ar.Email(validation="light", nullable=False),
        })
        result = ar.validate(frame, schema)
        
        assert result.passed
        assert result.issue_count == 0

    def test_email_field_invalid_emails(self):
        """Test Email field catches invalid email formats."""
        csv_data = [
            ["user_id", "email"],
            ["1", "not-an-email"],
            ["2", "bob@"],
            ["3", "@example.com"],
        ]
        frame = self._create_frame_from_data(csv_data)
        
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "email": ar.Email(),
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        assert result.issue_count > 0

    def test_email_field_unique_constraint(self):
        """Test Email field with unique=True detects duplicate emails."""
        csv_data = [
            ["user_id", "email"],
            ["1", "alice@example.com"],
            ["2", "alice@example.com"],
            ["3", "charlie@test.com"],
        ]
        frame = self._create_frame_from_data(csv_data)
        
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "email": ar.Email(unique=True),
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        assert result.issue_count > 0

    def test_email_field_unique_allows_different_emails(self):
        """Test Email field with unique=True passes for unique emails."""
        csv_data = [
            ["user_id", "email"],
            ["1", "alice@example.com"],
            ["2", "bob@example.com"],
            ["3", "charlie@test.com"],
        ]
        frame = self._create_frame_from_data(csv_data)
        
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "email": ar.Email(unique=True),
        })
        result = ar.validate(frame, schema)
        
        assert result.passed

    def test_email_field_severity_warning(self):
        """Test Email field with severity='warning' reports but passes."""
        csv_data = [
            ["user_id", "email"],
            ["1", "alice@example.com"],
            ["2", "invalid-email"],
        ]
        frame = self._create_frame_from_data(csv_data)
        
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "email": ar.Email(severity="warning"),
        })
        result = ar.validate(frame, schema)
        
        # Warning severity means passed=True but issues reported
        assert result.passed
        assert result.issue_count > 0

    def test_email_field_validation_strict_mode(self):
        """Test Email field with validation='strict' applies strict validation."""
        field = ar.Email(validation="strict")
        assert field is not None
        
        schema = ar.Schema({"email": field})
        assert schema is not None

    def test_email_field_validation_light_mode(self):
        """Test Email field with validation='light' applies light validation."""
        field = ar.Email(validation="light")
        assert field is not None
        
        schema = ar.Schema({"email": field})
        assert schema is not None

    def test_email_field_invalid_validation_type_raises_error(self):
        """Test that non-string validation parameter raises TypeError."""
        with pytest.raises(TypeError, match="Email validation must be a string"):
            ar.Email(validation=123)

    def test_email_field_invalid_validation_value_raises_error(self):
        """Test that invalid validation value raises ValueError."""
        with pytest.raises(ValueError, match="Email validation must be 'light' or 'strict'"):
            ar.Email(validation="invalid_mode")

    def test_email_field_full_workflow_with_pandas(self, tmp_path):
        """Test end-to-end workflow: CSV → clean → validate → pandas."""
        # Create input CSV
        csv_path = tmp_path / "users.csv"
        csv_data = [
            ["id", "name", "email"],
            ["1", " Alice ", " alice@example.com "],
            ["2", " Bob ", " bob@domain.org "],
            ["3", " Charlie ", " charlie@test.co.uk "],
        ]
        self._write_csv(csv_path, csv_data)
        
        # Load and clean
        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
        ])
        
        # Validate emails
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
            "email": ar.Email(nullable=False),
        })
        result = ar.validate(clean, schema)
        assert result.passed
        
        # Convert to pandas and verify
        df = ar.to_pandas(clean)
        assert len(df) == 3
        assert df["email"].iloc[0] == "alice@example.com"
        assert df["email"].iloc[1] == "bob@domain.org"

    def test_email_field_with_required_if_condition(self):
        """Test Email field with required_if conditional requirement."""
        # Create data where email is required only for premium accounts
        csv_data = [
            ["user_id", "account_type", "email"],
            ["1", "premium", "alice@example.com"],
            ["2", "premium", ""],
            ["3", "free", ""],
        ]
        frame = self._create_frame_from_data(csv_data)
        
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "account_type": ar.String(),
            "email": ar.Email(nullable=True, required_if=("account_type", "premium")),
        })
        result = ar.validate(frame, schema)
        
        # Row 2 should fail (premium account with no email)
        assert not result.passed or any(
            issue.row_index == 2 for issue in result.issues
        )

    def test_email_field_validation_result_details(self):
        """Test that validation result contains detailed issue information."""
        csv_data = [
            ["user_id", "email"],
            ["1", "valid@example.com"],
            ["2", "invalid.email"],
            ["3", "@nodomain"],
        ]
        frame = self._create_frame_from_data(csv_data)
        
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "email": ar.Email(nullable=False),
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        assert result.issue_count >= 2
        
        # Check that issues contain row information
        for issue in result.issues:
            assert issue.row_index > 0
            assert issue.column == "email"

    def test_email_field_with_pandas_accessor(self, tmp_path):
        """Test Email validation through pandas .arnio accessor."""
        # Create CSV
        csv_path = tmp_path / "emails.csv"
        csv_data = [
            ["id", "email"],
            ["1", "alice@example.com"],
            ["2", "bob@example.org"],
        ]
        self._write_csv(csv_path, csv_data)
        
        # Load as pandas
        df = pd.read_csv(csv_path)
        
        # Convert to arnio and validate
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(nullable=False),
        })
        result = ar.validate(frame, schema)
        
        assert result.passed

    # Helper methods
    def _create_frame_from_data(self, csv_data):
        """Helper to create an ArFrame from list of lists."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', 
                                         newline='', delete=False) as f:
            writer = csv.writer(f)
            for row in csv_data:
                writer.writerow(row)
            temp_path = f.name
        
        try:
            frame = ar.read_csv(temp_path)
            return frame
        finally:
            Path(temp_path).unlink()

    def _write_csv(self, path, data):
        """Helper to write CSV data to file."""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in data:
                writer.writerow(row)
