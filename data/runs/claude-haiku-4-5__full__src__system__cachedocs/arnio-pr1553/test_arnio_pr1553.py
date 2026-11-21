"""
System (end-to-end) tests for arnio.schema.Schema.validate.

Tests drive complete workflows through arnio's public API, including:
- Reading CSV files from disk
- Creating schemas with various field types and constraints
- Validating ArFrames
- Checking ValidationResult outputs
- Converting between ArFrame, pandas, and CSV formats
"""

import pytest
import pandas as pd
import arnio as ar


class TestSchemaValidateBasicWorkflow:
    """Core validation workflows."""

    def test_validate_passing_data(self, tmp_path):
        """Complete workflow: CSV → ArFrame → validate → passes."""
        csv_path = tmp_path / "valid.csv"
        csv_path.write_text("id,name,score\n1,Alice,95\n2,Bob,87\n3,Charlie,92\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "name": ar.String(nullable=False, min_length=1),
            "score": ar.Int64(nullable=False, min=0, max=100),
        })

        result = ar.validate(frame, schema)

        assert result.passed is True
        assert result.issue_count == 0
        assert len(result.issues) == 0

    def test_validate_failing_dtype(self, tmp_path):
        """Validation fails on dtype mismatch."""
        csv_path = tmp_path / "bad_dtype.csv"
        csv_path.write_text("id,score\n1,ninety\n2,eighty\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "score": ar.Int64(nullable=False),
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert result.issue_count > 0
        assert any(issue.column == "score" for issue in result.issues)

    def test_validate_failing_constraints(self, tmp_path):
        """Validation detects constraint violations (min/max, length)."""
        csv_path = tmp_path / "constraints.csv"
        csv_path.write_text("user_id,username,age\n1,ab,25\n2,validname,150\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "user_id": ar.Int64(nullable=False),
            "username": ar.String(nullable=False, min_length=3),
            "age": ar.Int64(nullable=False, min=0, max=120),
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        issues_by_col = {i.column: i for i in result.issues}
        assert "username" in issues_by_col or "age" in issues_by_col

    def test_validate_nullable_field_pass(self, tmp_path):
        """Nullable fields accept null values."""
        csv_path = tmp_path / "nullable.csv"
        csv_path.write_text("id,name,notes\n1,Alice,has notes\n2,Bob,\n3,Charlie,\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
            "notes": ar.String(nullable=True),
        })

        result = ar.validate(frame, schema)

        assert result.passed is True

    def test_validate_nullable_field_fail(self, tmp_path):
        """Non-nullable fields reject null values."""
        csv_path = tmp_path / "required.csv"
        csv_path.write_text("id,name\n1,Alice\n2,\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.column == "name" and issue.row_index == 2 for issue in result.issues)


class TestSchemaValidateFieldTypes:
    """Field type validation."""

    def test_validate_email_format(self, tmp_path):
        """Email field validates format."""
        csv_path = tmp_path / "emails.csv"
        csv_path.write_text("id,email\n1,valid@example.com\n2,not-an-email\n3,bob@test.org\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.column == "email" and issue.row_index == 2 for issue in result.issues)

    def test_validate_url_format(self, tmp_path):
        """URL field validates format."""
        csv_path = tmp_path / "urls.csv"
        csv_path.write_text("id,website\n1,https://example.com\n2,not a url\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "website": ar.URL(nullable=False),
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.column == "website" for issue in result.issues)

    def test_validate_float_field(self, tmp_path):
        """Float field validation with constraints."""
        csv_path = tmp_path / "prices.csv"
        csv_path.write_text("id,price\n1,19.99\n2,-5.00\n3,100.00\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "price": ar.Float64(nullable=False, min=0.0),
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.column == "price" and issue.row_index == 2 for issue in result.issues)


class TestSchemaValidateUniqueConstraints:
    """Uniqueness constraints."""

    def test_validate_unique_column(self, tmp_path):
        """Uniqueness constraint detects duplicates."""
        csv_path = tmp_path / "users.csv"
        csv_path.write_text("id,username\n1,alice\n2,bob\n3,alice\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "username": ar.String(nullable=False, unique=True),
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.rule == "unique" for issue in result.issues)

    def test_validate_missing_required_column(self, tmp_path):
        """Schema validation fails when required column is missing."""
        csv_path = tmp_path / "incomplete.csv"
        csv_path.write_text("id,name\n1,Alice\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
            "email": ar.Email(nullable=False),  # Not in CSV
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.column == "email" and issue.rule == "required_column" for issue in result.issues)


class TestSchemaValidateMaxErrors:
    """max_errors parameter controls result size."""

    def test_validate_max_errors_limits_issues(self, tmp_path):
        """max_errors parameter limits number of reported issues."""
        csv_path = tmp_path / "many_errors.csv"
        lines = ["id,score\n"]
        for i in range(1, 21):
            lines.append(f"{i},invalid\n")
        csv_path.write_text("".join(lines))

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "score": ar.Int64(nullable=False),
        })

        result = ar.validate(frame, schema, max_errors=5)

        assert result.issue_count <= 5

    def test_validate_max_errors_none_returns_all(self, tmp_path):
        """max_errors=None returns all issues."""
        csv_path = tmp_path / "all_errors.csv"
        lines = ["id,score\n"]
        for i in range(1, 6):
            lines.append(f"{i},invalid\n")
        csv_path.write_text("".join(lines))

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "score": ar.Int64(nullable=False),
        })

        result = ar.validate(frame, schema, max_errors=None)

        assert result.issue_count == 5


class TestSchemaValidateOutputs:
    """ValidationResult output formats."""

    def test_validate_result_to_markdown(self, tmp_path):
        """ValidationResult produces markdown output."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,age\n1,25\n2,-5\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "age": ar.Int64(nullable=False, min=0, max=150),
        })

        result = ar.validate(frame, schema)
        markdown = result.to_markdown()

        assert isinstance(markdown, str)
        assert len(markdown) > 0
        assert "age" in markdown.lower() or "validation" in markdown.lower()

    def test_validate_result_to_pandas(self, tmp_path):
        """ValidationResult exports issues to pandas DataFrame."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score\n1,invalid\n2,99\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "score": ar.Int64(nullable=False),
        })

        result = ar.validate(frame, schema)
        df = result.to_pandas()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_validate_result_summary(self, tmp_path):
        """ValidationResult.summary() provides structured output."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,name\n1,\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })

        result = ar.validate(frame, schema)
        summary = result.summary()

        assert isinstance(summary, dict)


class TestSchemaValidateIntegration:
    """Integration with other arnio workflows."""

    def test_validate_after_pipeline_cleanup(self, tmp_path):
        """Validate cleaned data after applying pipeline."""
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text("id,name\n1, Alice \n2, Bob \n")

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [("strip_whitespace",)])

        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })

        result = ar.validate(clean, schema)

        assert result.passed is True

    def test_validate_roundtrip_csv_export(self, tmp_path):
        """Validate CSV that was exported and re-imported."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("id,name\n1,Alice\n2,Bob\n")

        frame = ar.read_csv(str(input_csv))
        output_csv = tmp_path / "output.csv"
        ar.write_csv(frame, str(output_csv))

        reloaded = ar.read_csv(str(output_csv))

        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })

        result = ar.validate(reloaded, schema)

        assert result.passed is True

    def test_validate_with_pandas_roundtrip(self, tmp_path):
        """Validate ArFrame created from pandas DataFrame."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95, 87, 92],
        })

        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
            "score": ar.Int64(nullable=False, min=0, max=100),
        })

        result = ar.validate(frame, schema)

        assert result.passed is True
        # Convert back to verify round-trip
        result_df = ar.to_pandas(frame)
        assert len(result_df) == 3
