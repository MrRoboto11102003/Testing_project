import pytest
import pandas as pd
import arnio as ar
from pathlib import Path


class TestStringSchemaEndToEnd:
    """End-to-end tests for arnio.schema.String field validation."""

    def test_string_basic_valid(self, tmp_path):
        """Test basic string field validation passes for valid data."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,city\nAlice,Paris\nBob,London\nCharlie,Tokyo\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "name": ar.String(nullable=False),
            "city": ar.String(nullable=False),
        })
        
        result = ar.validate(frame, schema)
        assert result.passed
        assert len(result.issues) == 0

    def test_string_nullable_false_fails_on_empty(self, tmp_path):
        """Test that nullable=False rejects null/empty values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,\n3,Charlie\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "name": ar.String(nullable=False),
        })
        
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.column == "name" and issue.row_index == 2 for issue in result.issues)

    def test_string_pattern_regex_validation(self, tmp_path):
        """Test string field with regex pattern matching."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\nalice@test.com\nbob@example.org\ninvalid.email\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "email": ar.String(
                pattern=r"^[^@]+@[^@]+\.[^@]+$",
                nullable=False
            ),
        })
        
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any("pattern" in str(issue).lower() for issue in result.issues)

    def test_string_allowed_values_case_sensitive(self, tmp_path):
        """Test allowed values list with case sensitivity enabled."""
        csv_file = tmp_path / "status.csv"
        csv_file.write_text("status\nactive\ninactive\npending\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "status": ar.String(
                allowed=["active", "inactive", "pending"],
                case_sensitive=True,
                nullable=False
            ),
        })
        
        result = ar.validate(frame, schema)
        assert result.passed

    def test_string_allowed_values_case_insensitive(self, tmp_path):
        """Test allowed values with case_sensitive=False allows different cases."""
        csv_file = tmp_path / "country.csv"
        csv_file.write_text("country\nUS\nuk\nFr\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "country": ar.String(
                allowed=["US", "UK", "FR"],
                case_sensitive=False,
                nullable=False
            ),
        })
        
        result = ar.validate(frame, schema)
        assert result.passed

    def test_string_min_length_constraint(self, tmp_path):
        """Test minimum string length constraint."""
        csv_file = tmp_path / "usernames.csv"
        csv_file.write_text("username\nalice\nab\ncharlie\nxyz\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "username": ar.String(min_length=3, nullable=False),
        })
        
        result = ar.validate(frame, schema)
        assert not result.passed
        assert len(result.issues) == 1
        assert result.issues[0].row_index == 2  # "ab" is on row 2

    def test_string_max_length_constraint(self, tmp_path):
        """Test maximum string length constraint."""
        csv_file = tmp_path / "codes.csv"
        csv_file.write_text("code\nABC\nABCDEF\nXY\nVWXYZ\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "code": ar.String(max_length=5, nullable=False),
        })
        
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_string_min_and_max_length_combined(self, tmp_path):
        """Test both min and max length constraints together."""
        csv_file = tmp_path / "ids.csv"
        csv_file.write_text("id\nUSR001\nAA\nUSR00000\nX123Y\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "id": ar.String(min_length=4, max_length=7, nullable=False),
        })
        
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_string_unique_constraint_violation(self, tmp_path):
        """Test unique constraint detects duplicate string values."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\nalice@test.com\nbob@test.com\nalice@test.com\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "email": ar.String(unique=True, nullable=False),
        })
        
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_string_required_if_conditional_requirement(self, tmp_path):
        """Test conditional requirement based on another column value."""
        csv_file = tmp_path / "users.csv"
        csv_file.write_text("user_type,company_name\nindividual,\nbusiness,Acme Inc\nindividual,\nbusiness,\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "user_type": ar.String(nullable=False),
            "company_name": ar.String(
                required_if=("user_type", "business"),
                nullable=True
            ),
        })
        
        result = ar.validate(frame, schema)
        assert not result.passed
        # Row 4 has user_type="business" but company_name is empty

    def test_string_severity_warning_non_fatal(self, tmp_path):
        """Test that warning severity doesn't fail overall validation."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("code\nABCDEFGHI\nXYZ\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "code": ar.String(max_length=5, severity="warning", nullable=False),
        })
        
        result = ar.validate(frame, schema)
        assert result.passed  # Validation passed despite warnings
        assert len(result.issues) > 0  # But has warning issues

    def test_string_invalid_regex_pattern_raises_valueerror(self):
        """Test that invalid regex pattern raises ValueError during schema creation."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            ar.String(pattern="[invalid(regex")

    def test_string_min_greater_than_max_raises_valueerror(self):
        """Test that min_length > max_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be less than or equal to max_length"):
            ar.String(min_length=10, max_length=5)

    def test_string_allowed_bare_string_raises_typeerror(self):
        """Test that passing bare string to allowed raises TypeError."""
        with pytest.raises(TypeError, match="allowed must be a sequence"):
            ar.String(allowed="active,inactive")

    def test_string_full_workflow_read_validate_clean_write(self, tmp_path):
        """End-to-end workflow: read CSV, validate, clean, and write output."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("name,status\nAlice,ACTIVE\nBob,active\nCharlie,PENDING\n")
        
        output_csv = tmp_path / "output.csv"
        
        # Read CSV
        frame = ar.read_csv(str(input_csv))
        
        # Validate with case-sensitive allowed values (should fail on lowercase "active")
        schema = ar.Schema({
            "name": ar.String(min_length=1, nullable=False),
            "status": ar.String(
                allowed=["ACTIVE", "INACTIVE", "PENDING"],
                case_sensitive=True,
                nullable=False
            ),
        })
        
        result = ar.validate(frame, schema)
        assert not result.passed
        
        # Clean: normalize to uppercase
        clean_frame = ar.pipeline(frame, [
            ("normalize_case", {"case_type": "upper"}),
        ])
        
        # Validate again (should pass now)
        result2 = ar.validate(clean_frame, schema)
        assert result2.passed
        
        # Write output
        ar.write_csv(clean_frame, str(output_csv))
        
        # Verify file was written
        assert output_csv.exists()
        content = output_csv.read_text()
        assert "ACTIVE" in content
        assert "PENDING" in content

    def test_string_validation_via_pandas_accessor(self):
        """Test string validation using pandas DataFrame .arnio accessor."""
        df = pd.DataFrame({
            "username": ["alice_user", "bob", "charlie_name"],
            "status": ["active", "inactive", "active"],
        })
        
        schema = ar.Schema({
            "username": ar.String(min_length=5, nullable=False),
            "status": ar.String(allowed=["active", "inactive"], nullable=False),
        })
        
        result = df.arnio.validate(schema)
        assert not result.passed
        # "bob" (3 chars) fails min_length=5 constraint

    def test_string_pattern_and_length_combined_constraints(self, tmp_path):
        """Test string field with both pattern and length constraints."""
        csv_file = tmp_path / "codes.csv"
        csv_file.write_text("code\nUSR-001\nABC\nUSR-12345\nXY-Z\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "code": ar.String(
                pattern=r"^[A-Z]{3}-\d{3}$",
                min_length=7,
                max_length=7,
                nullable=False
            ),
        })
        
        result = ar.validate(frame, schema)
        assert not result.passed
