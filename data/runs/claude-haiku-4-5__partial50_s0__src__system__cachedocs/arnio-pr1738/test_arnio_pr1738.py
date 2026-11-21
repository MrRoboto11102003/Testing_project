import pytest
import pandas as pd
import arnio as ar
from pathlib import Path


class TestPhoneNumberValidation:
    """Test arnio.schema.PhoneNumber validator in end-to-end workflows."""

    def test_phone_valid_formats_pass(self, tmp_path):
        """Valid phone number formats from README should pass validation."""
        csv_file = tmp_path / "phones.csv"
        csv_file.write_text(
            "phone\n"
            "+1-555-123-4567\n"
            "+91 9876543210\n"
            "5551234567\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = ar.validate(frame, schema)

        assert result.passed
        assert result.issue_count == 0

    def test_phone_invalid_format_fails(self, tmp_path):
        """Invalid phone formats should fail validation."""
        csv_file = tmp_path / "phones.csv"
        csv_file.write_text("phone\nabc123\nnot_a_phone\n!!!\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = ar.validate(frame, schema)

        assert not result.passed
        assert result.issue_count == 3
        assert all(issue.column == "phone" for issue in result.issues)

    def test_phone_nullable_allows_empty(self, tmp_path):
        """Nullable phone field should allow null/empty values."""
        csv_file = tmp_path / "phones.csv"
        csv_file.write_text(
            "id,phone\n"
            "1,+1-555-123-4567\n"
            "2,\n"
            "3,5551234567\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "id": ar.Int64(),
            "phone": ar.PhoneNumber(nullable=True),
        })
        result = ar.validate(frame, schema)

        assert result.passed

    def test_phone_non_nullable_rejects_empty(self, tmp_path):
        """Non-nullable phone field should reject null values."""
        csv_file = tmp_path / "phones.csv"
        csv_file.write_text("phone\n+1-555-123-4567\n\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = ar.validate(frame, schema)

        assert not result.passed
        assert any(issue.rule == "required_value" for issue in result.issues)

    def test_phone_unique_constraint(self, tmp_path):
        """Phone field with unique=True should reject duplicates."""
        csv_file = tmp_path / "phones.csv"
        csv_file.write_text(
            "phone\n"
            "+1-555-123-4567\n"
            "+91 9876543210\n"
            "+1-555-123-4567\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"phone": ar.PhoneNumber(unique=True)})
        result = ar.validate(frame, schema)

        assert not result.passed
        assert any(issue.rule == "unique" for issue in result.issues)

    def test_phone_severity_warning_does_not_fail(self, tmp_path):
        """Phone validation with severity='warning' should not fail overall."""
        csv_file = tmp_path / "phones.csv"
        csv_file.write_text("phone\ninvalid\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"phone": ar.PhoneNumber(severity="warning")})
        result = ar.validate(frame, schema)

        assert result.passed
        assert result.issue_count > 0  # Issues recorded but not fatal

    def test_phone_validation_to_pandas(self, tmp_path):
        """Validated phone frame converts cleanly to pandas DataFrame."""
        csv_file = tmp_path / "phones.csv"
        csv_file.write_text("phone\n+1-555-123-4567\n+91 9876543210\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"phone": ar.PhoneNumber()})
        result = ar.validate(frame, schema)

        assert result.passed
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df["phone"].iloc[0] == "+1-555-123-4567"


class TestURLValidation:
    """Test arnio.schema.URL validator in end-to-end workflows."""

    def test_url_valid_formats_pass(self, tmp_path):
        """Valid http/https URLs should pass validation."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text(
            "url\n"
            "https://example.com\n"
            "http://google.com\n"
            "https://github.com/user/repo?query=value#anchor\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"url": ar.URL(nullable=False)})
        result = ar.validate(frame, schema)

        assert result.passed
        assert result.issue_count == 0

    def test_url_invalid_format_fails(self, tmp_path):
        """Invalid URLs should fail validation."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text(
            "url\n"
            "not_a_url\n"
            "just text\n"
            "ftp://unsupported.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"url": ar.URL(nullable=False)})
        result = ar.validate(frame, schema)

        assert not result.passed
        assert result.issue_count > 0

    def test_url_allowed_schemes_https_only(self, tmp_path):
        """URL field with allowed_schemes=['https'] should reject http."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text(
            "url\n"
            "https://secure.com\n"
            "http://insecure.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"url": ar.URL(allowed_schemes=["https"])})
        result = ar.validate(frame, schema)

        assert not result.passed
        assert any(issue.column == "url" for issue in result.issues)

    def test_url_allowed_schemes_multiple(self, tmp_path):
        """URL field can restrict to multiple allowed schemes."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text(
            "url\n"
            "https://secure.com\n"
            "http://public.com\n"
            "ftp://files.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"url": ar.URL(allowed_schemes=["http", "https"])})
        result = ar.validate(frame, schema)

        assert not result.passed
        assert len([i for i in result.issues if i.column == "url"]) >= 1

    def test_url_nullable_allows_empty(self, tmp_path):
        """Nullable URL field should allow null/empty values."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text(
            "id,url\n"
            "1,https://example.com\n"
            "2,\n"
            "3,https://google.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "id": ar.Int64(),
            "url": ar.URL(nullable=True),
        })
        result = ar.validate(frame, schema)

        assert result.passed

    def test_url_non_nullable_rejects_empty(self, tmp_path):
        """Non-nullable URL field should reject null values."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text("url\nhttps://example.com\n\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"url": ar.URL(nullable=False)})
        result = ar.validate(frame, schema)

        assert not result.passed

    def test_url_unique_constraint(self, tmp_path):
        """URL field with unique=True should reject duplicate values."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text(
            "url\n"
            "https://example.com\n"
            "https://google.com\n"
            "https://example.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"url": ar.URL(unique=True)})
        result = ar.validate(frame, schema)

        assert not result.passed
        assert any(issue.rule == "unique" for issue in result.issues)

    def test_url_allowed_schemes_validation_error_string(self):
        """URL with bare string allowed_schemes should raise TypeError."""
        with pytest.raises(TypeError, match="must be a sequence"):
            ar.URL(allowed_schemes="https")

    def test_url_allowed_schemes_validation_error_empty(self):
        """URL with empty allowed_schemes list should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            ar.URL(allowed_schemes=[])

    def test_url_validation_result_markdown(self, tmp_path):
        """ValidationResult with URL errors should format to markdown."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text("url\nnot_a_url\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"url": ar.URL()})
        result = ar.validate(frame, schema)

        markdown = result.to_markdown()
        assert isinstance(markdown, str)
        assert len(markdown) > 0
        assert not result.passed


class TestPhoneAndURLCombined:
    """Test PhoneNumber and URL validators in realistic combined workflows."""

    def test_customer_contact_data_validation(self, tmp_path):
        """Validate customer records with both phone and URL fields."""
        csv_file = tmp_path / "customers.csv"
        csv_file.write_text(
            "customer_id,name,phone,website\n"
            "1,Acme Corp,+1-555-123-4567,https://acme.example.com\n"
            "2,Beta Inc,,https://beta.example.com\n"
            "3,Gamma Ltd,+91 9876543210,\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "customer_id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
            "phone": ar.PhoneNumber(nullable=True),
            "website": ar.URL(nullable=True),
        })
        result = ar.validate(frame, schema)

        assert result.passed

        df = ar.to_pandas(frame)
        assert len(df) == 3
        assert df["name"].tolist() == ["Acme Corp", "Beta Inc", "Gamma Ltd"]

    def test_mixed_contact_data_profiling(self, tmp_path):
        """Profile data containing both phone and URL columns."""
        csv_file = tmp_path / "contacts.csv"
        csv_file.write_text(
            "phone,website,contact_type\n"
            "+1-555-123-4567,https://example.com,primary\n"
            "+91 9876543210,https://google.com,secondary\n"
            ",https://backup.com,tertiary\n"
        )

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)

        assert report.row_count == 3
        assert "phone" in report.columns
        assert "website" in report.columns
        assert report.columns["phone"].null_count == 1
        assert report.columns["website"].null_count == 0

    def test_validation_summary_output(self, tmp_path):
        """ValidationResult.summary() includes phone and URL validation metrics."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "phone,url\n"
            "+1-555-123-4567,https://example.com\n"
            "invalid,not-url\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "phone": ar.PhoneNumber(),
            "url": ar.URL(),
        })
        result = ar.validate(frame, schema)

        assert not result.passed
        summary = result.summary()
        assert isinstance(summary, dict)
        assert summary.get("total_issues", 0) > 0
