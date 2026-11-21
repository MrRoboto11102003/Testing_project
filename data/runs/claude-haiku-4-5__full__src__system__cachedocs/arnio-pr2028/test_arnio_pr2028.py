import pytest
import pandas as pd
import arnio as ar
import tempfile
from pathlib import Path


class TestStringFieldValidation:
    """End-to-end tests for arnio.schema.String field validation."""

    def test_string_basic_validation_passes(self, tmp_path):
        """Valid string data passes basic schema validation."""
        csv_file = tmp_path / "users.csv"
        csv_file.write_text("name,city\nAlice,Paris\nBob,London\nCharlie,Tokyo\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "name": ar.String(nullable=False),
            "city": ar.String(nullable=False)
        })
        result = ar.validate(frame, schema)
        
        assert result.passed
        assert result.issue_count == 0

    def test_string_nullable_false_rejects_nulls(self, tmp_path):
        """String with nullable=False rejects rows with null values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,email\nAlice,alice@example.com\n,bob@example.com\nCharlie,\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "name": ar.String(nullable=False),
            "email": ar.String(nullable=False)
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        assert result.issue_count >= 1
        assert any(issue.row_index == 2 and issue.column == "name" for issue in result.issues)
        assert any(issue.row_index == 3 and issue.column == "email" for issue in result.issues)

    def test_string_nullable_true_allows_nulls(self, tmp_path):
        """String with nullable=True allows null values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,notes\nAlice,active user\n,no notes\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "name": ar.String(nullable=True),
            "notes": ar.String(nullable=True)
        })
        result = ar.validate(frame, schema)
        
        assert result.passed

    def test_string_pattern_matching(self, tmp_path):
        """String pattern validates against regex."""
        csv_file = tmp_path / "emails.csv"
        csv_file.write_text("email\nalice@example.com\nbob@example.com\ninvalid.email\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "email": ar.String(pattern=r"^[^@]+@[^@]+\.[^@]+$")
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        assert any(issue.row_index == 3 and issue.column == "email" for issue in result.issues)

    def test_string_allowed_values_enum(self, tmp_path):
        """String allowed parameter enforces enumeration."""
        csv_file = tmp_path / "statuses.csv"
        csv_file.write_text("status\nactive\ninactive\npending\ninvalid_status\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "status": ar.String(allowed=["active", "inactive", "pending"])
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        issue = next((i for i in result.issues if i.row_index == 4), None)
        assert issue is not None
        assert issue.column == "status"

    def test_string_allowed_case_sensitive_true(self, tmp_path):
        """Case sensitive matching for allowed values."""
        csv_file = tmp_path / "codes.csv"
        csv_file.write_text("code\nABC\nabc\nABC\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "code": ar.String(allowed=["ABC"], case_sensitive=True)
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        assert any(issue.row_index == 2 for issue in result.issues)

    def test_string_allowed_case_sensitive_false(self, tmp_path):
        """Case insensitive matching for allowed values."""
        csv_file = tmp_path / "codes.csv"
        csv_file.write_text("code\nABC\nabc\nAbC\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "code": ar.String(allowed=["ABC"], case_sensitive=False)
        })
        result = ar.validate(frame, schema)
        
        assert result.passed

    def test_string_min_length(self, tmp_path):
        """String min_length validates minimum string length."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("username\nAlice\nBob\nAl\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "username": ar.String(min_length=3)
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        assert any(issue.row_index == 3 for issue in result.issues)

    def test_string_max_length(self, tmp_path):
        """String max_length validates maximum string length."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("code\nAB\nABC\nABCD\nABCDE\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "code": ar.String(max_length=4)
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        assert any(issue.row_index == 5 for issue in result.issues)

    def test_string_unique_constraint(self, tmp_path):
        """String unique=True detects duplicate values."""
        csv_file = tmp_path / "users.csv"
        csv_file.write_text("user_id\nalice123\nbob456\nalice123\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "user_id": ar.String(unique=True)
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        assert any(issue.row_index == 3 for issue in result.issues)

    def test_string_severity_warning_still_reports_issues(self, tmp_path):
        """Severity=warning reports issues but doesn't fail overall validation."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("code\nAB\nABC\nABCD\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "code": ar.String(min_length=3, severity="warning")
        })
        result = ar.validate(frame, schema)
        
        assert result.passed
        assert result.issue_count > 0

    def test_string_required_if_conditional_requirement(self, tmp_path):
        """required_if makes field required based on another column value."""
        csv_file = tmp_path / "accounts.csv"
        csv_file.write_text("type,company\npersonal,\nbusiness,ACME Corp\nbusiness,\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "type": ar.String(),
            "company": ar.String(nullable=True, required_if=("type", "business"))
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        assert any(issue.row_index == 3 for issue in result.issues)

    def test_string_invalid_regex_pattern_raises_error(self):
        """Invalid regex pattern raises ValueError at schema creation time."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            ar.String(pattern="[unclosed(pattern")

    def test_string_min_max_conflict_raises_error(self):
        """min_length > max_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be less than or equal"):
            ar.String(min_length=10, max_length=5)

    def test_string_validation_with_pandas_accessor(self, tmp_path):
        """Validation works through pandas .arnio accessor."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "status": ["active", "inactive", "pending"],
            "code": ["A", "B", "TOOLONG"]
        })
        
        schema = ar.Schema({
            "name": ar.String(nullable=False),
            "status": ar.String(allowed=["active", "inactive", "pending"]),
            "code": ar.String(max_length=1)
        })
        
        result = df.arnio.validate(schema)
        
        assert not result.passed
        assert any(issue.column == "code" for issue in result.issues)

    def test_string_allowed_with_tuple_set_list(self, tmp_path):
        """Allowed parameter accepts tuple, set, and list."""
        for allowed_input, label in [
            (["red", "green", "blue"], "list"),
            (("red", "green", "blue"), "tuple"),
            ({"red", "green", "blue"}, "set")
        ]:
            csv_file = tmp_path / f"colors_{label}.csv"
            csv_file.write_text("color\nred\ngreen\n")
            
            frame = ar.read_csv(str(csv_file))
            schema = ar.Schema({
                "color": ar.String(allowed=allowed_input)
            })
            result = ar.validate(frame, schema)
            
            assert result.passed, f"Failed for {label}"

    def test_string_pattern_with_special_characters(self, tmp_path):
        """Pattern matching handles special regex characters."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("code\nUSA-001\nCAN-002\nUSA001\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "code": ar.String(pattern=r"^[A-Z]{3}-\d{3}$")
        })
        result = ar.validate(frame, schema)
        
        assert not result.passed
        assert any(issue.row_index == 3 for issue in result.issues)

    def test_string_round_trip_validate_and_convert_to_pandas(self, tmp_path):
        """Validate a frame and convert to pandas for downstream use."""
        csv_file = tmp_path / "customers.csv"
        csv_file.write_text("customer_id,name,email\n1,Alice,alice@ex.com\n2,Bob,bob@ex.com\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "customer_id": ar.String(pattern=r"^\d+$"),
            "name": ar.String(min_length=1, nullable=False),
            "email": ar.String(pattern=r".+@.+")
        })
        result = ar.validate(frame, schema)
        
        assert result.passed
        
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "name" in df.columns
