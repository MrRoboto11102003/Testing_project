import pytest
import pandas as pd
import arnio as ar


class TestEmailSchema:
    """Tests for ar.Email() factory function."""

    def test_default_parameters(self):
        """Email() should return Field with defaults: light semantic, nullable, non-unique."""
        field = ar.Email()
        assert field.dtype == "string"
        assert field.semantic == "email"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
        assert field.required_if is None

    def test_strict_validation_mode(self):
        """Email(validation='strict') should use email:strict semantic."""
        field = ar.Email(validation="strict")
        assert field.semantic == "email:strict"

    def test_validation_parameter_case_sensitive(self):
        """Email(validation='LIGHT') should fail (case-sensitive)."""
        with pytest.raises(ValueError, match="Email validation must be 'light' or 'strict'"):
            ar.Email(validation="LIGHT")

    def test_validation_parameter_not_string(self):
        """Email(validation=None) should raise TypeError."""
        with pytest.raises(TypeError, match="Email validation must be a string"):
            ar.Email(validation=None)

    def test_validation_parameter_invalid_value(self):
        """Email(validation='invalid') should raise ValueError."""
        with pytest.raises(ValueError, match="Email validation must be 'light' or 'strict'"):
            ar.Email(validation="invalid")

    def test_nullable_parameter_propagates(self):
        """Email(nullable=False) should propagate to Field."""
        field = ar.Email(nullable=False)
        assert field.nullable is False

    def test_unique_parameter_propagates(self):
        """Email(unique=True) should propagate to Field."""
        field = ar.Email(unique=True)
        assert field.unique is True

    def test_severity_parameter_propagates(self):
        """Email(severity='warning') should propagate to Field."""
        field = ar.Email(severity="warning")
        assert field.severity == "warning"

    def test_required_if_parameter_propagates(self):
        """Email(required_if=(...)) should propagate to Field."""
        field = ar.Email(required_if=("status", "active"))
        assert field.required_if == ("status", "active")

    def test_returned_field_is_frozen(self):
        """Email() returns frozen Field; modifications should raise."""
        field = ar.Email()
        with pytest.raises(Exception):  # FrozenInstanceError
            field.dtype = "int64"

    def test_light_mode_accepts_underscores_in_domain(self):
        """Light validation should accept underscores in domain."""
        frame = ar.from_pandas(pd.DataFrame({"email": ["user@exa_mple.com"]}))
        result = ar.validate(frame, ar.Schema({"email": ar.Email(validation="light")}))
        email_issues = [i for i in result.issues if i.rule == "email"]
        assert len(email_issues) == 0

    def test_strict_mode_rejects_underscores_in_domain(self):
        """Strict validation should reject underscores in domain."""
        frame = ar.from_pandas(pd.DataFrame({"email": ["user@exa_mple.com"]}))
        result = ar.validate(frame, ar.Schema({"email": ar.Email(validation="strict")}))
        email_issues = [i for i in result.issues if i.rule == "email:strict"]
        assert len(email_issues) == 1
        assert email_issues[0].row_index == 1

    def test_light_vs_strict_fields_not_equal(self):
        """Email(validation='light') and Email(validation='strict') should differ."""
        light = ar.Email(validation="light")
        strict = ar.Email(validation="strict")
        assert light != strict

    def test_valid_email_addresses_pass(self):
        """Valid email addresses should produce no email issues."""
        frame = ar.from_pandas(pd.DataFrame({
            "email": ["a@b.co", "user@sub.example.com"]
        }))
        result = ar.validate(frame, ar.Schema({"email": ar.Email()}))
        email_issues = [i for i in result.issues if i.rule == "email"]
        assert len(email_issues) == 0

    def test_invalid_email_addresses_fail(self):
        """Invalid emails (no dot, double @, with spaces) should produce email issues."""
        frame = ar.from_pandas(pd.DataFrame({
            "email": ["a@b", "a@@b.co", "a b@c.de"]
        }))
        result = ar.validate(frame, ar.Schema({"email": ar.Email()}))
        email_issues = [i for i in result.issues if i.rule == "email"]
        assert len(email_issues) == 3
