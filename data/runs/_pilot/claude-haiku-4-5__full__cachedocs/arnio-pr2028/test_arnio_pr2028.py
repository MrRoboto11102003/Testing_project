import pytest
import pandas as pd
import arnio as ar


class TestStringField:
    """Tests for arnio.schema.String field validation."""

    def make_frame(self, data):
        """Helper to create an ArFrame from a dict."""
        return ar.from_pandas(pd.DataFrame(data))

    def test_string_basic_valid(self):
        """Basic string field should accept valid strings."""
        schema = ar.Schema({"name": ar.String(nullable=False)})
        frame = self.make_frame({"name": ["Alice", "Bob", "Charlie"]})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_string_nullable_true_allows_nulls(self):
        """String field with nullable=True should allow null values."""
        schema = ar.Schema({"name": ar.String(nullable=True)})
        frame = self.make_frame({"name": ["Alice", None, "Bob"]})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_string_nullable_false_rejects_nulls(self):
        """String field with nullable=False should reject null values."""
        schema = ar.Schema({"name": ar.String(nullable=False)})
        frame = self.make_frame({"name": ["Alice", None, "Bob"]})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert len(result.issues) == 1
        assert result.issues[0].row_index == 2

    def test_string_pattern_validation(self):
        """String field with pattern regex should validate correctly."""
        schema = ar.Schema({"code": ar.String(pattern=r"^[A-Z]{3}$", nullable=False)})
        frame = self.make_frame({"code": ["ABC", "XYZ", "DEF"]})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_string_pattern_rejection(self):
        """String field with pattern should reject non-matching values."""
        schema = ar.Schema({"code": ar.String(pattern=r"^[A-Z]{3}$", nullable=False)})
        frame = self.make_frame({"code": ["ABC", "12X", "XYZ"]})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert len(result.issues) == 1
        assert result.issues[0].row_index == 2

    def test_string_allowed_values(self):
        """String field with allowed list should validate membership."""
        schema = ar.Schema({"status": ar.String(allowed=["active", "inactive", "pending"], nullable=False)})
        frame = self.make_frame({"status": ["active", "inactive", "pending"]})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_string_allowed_values_rejection(self):
        """String field with allowed list should reject non-member values."""
        schema = ar.Schema({"status": ar.String(allowed=["active", "inactive"], nullable=False)})
        frame = self.make_frame({"status": ["active", "unknown", "inactive"]})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.row_index == 2 for issue in result.issues)

    def test_string_case_insensitive_matching(self):
        """String field with case_sensitive=False should match case-insensitively."""
        schema = ar.Schema({
            "status": ar.String(
                allowed=["active", "inactive"],
                case_sensitive=False,
                nullable=False
            )
        })
        frame = self.make_frame({"status": ["ACTIVE", "Inactive", "active"]})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_string_case_sensitive_matching(self):
        """String field with case_sensitive=True should match case-sensitively."""
        schema = ar.Schema({
            "status": ar.String(
                allowed=["active", "inactive"],
                case_sensitive=True,
                nullable=False
            )
        })
        frame = self.make_frame({"status": ["ACTIVE", "inactive"]})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert len(result.issues) == 1

    def test_string_min_length(self):
        """String field with min_length should reject shorter strings."""
        schema = ar.Schema({"name": ar.String(min_length=3, nullable=False)})
        frame = self.make_frame({"name": ["Alice", "Bob", "Al"]})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.row_index == 3 for issue in result.issues)

    def test_string_max_length(self):
        """String field with max_length should reject longer strings."""
        schema = ar.Schema({"code": ar.String(max_length=5, nullable=False)})
        frame = self.make_frame({"code": ["ABC", "DEFGH", "HIJKLMNOP"]})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert len(result.issues) == 1

    def test_string_length_range(self):
        """String field with min and max length should validate range."""
        schema = ar.Schema({"code": ar.String(min_length=2, max_length=4, nullable=False)})
        frame = self.make_frame({"code": ["A", "AB", "ABC", "ABCDE"]})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert len(result.issues) == 2  # Too short and too long

    def test_string_unique_constraint(self):
        """String field with unique=True should reject duplicate values."""
        schema = ar.Schema({"email": ar.String(unique=True, nullable=False)})
        frame = self.make_frame({"email": ["alice@example.com", "bob@example.com", "alice@example.com"]})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any("unique" in issue.rule.lower() for issue in result.issues)

    def test_string_severity_warning(self):
        """String field with severity='warning' should report issues but pass."""
        schema = ar.Schema({
            "name": ar.String(min_length=10, severity="warning", nullable=False)
        })
        frame = self.make_frame({"name": ["Alice", "Bob"]})
        result = ar.validate(frame, schema)
        assert result.passed  # Warnings don't fail validation
        assert len(result.issues) > 0  # But issues are still recorded

    def test_string_empty_string_not_null(self):
        """Empty strings should be valid strings, not treated as null."""
        schema = ar.Schema({"name": ar.String(nullable=False)})
        frame = self.make_frame({"name": ["Alice", "", "Bob"]})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_string_combined_constraints(self):
        """String field with multiple constraints should validate all."""
        schema = ar.Schema({
            "username": ar.String(
                nullable=False,
                pattern=r"^[a-z_]+$",
                min_length=3,
                max_length=20,
                unique=True,
            )
        })
        frame = self.make_frame({
            "username": ["alice_smith", "bob_jones", "charlie_brown"]
        })
        result = ar.validate(frame, schema)
        assert result.passed
