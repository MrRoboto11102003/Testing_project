import pytest
import pandas as pd
import arnio as ar
import tempfile
from pathlib import Path


class TestValidateBasicWorkflows:
    """Core validation workflows matching real user patterns."""

    def test_validate_passing_schema(self, tmp_path):
        """User loads CSV, validates against schema that passes."""
        csv_path = tmp_path / "customers.csv"
        csv_path.write_text("id,name,age\n1,Alice,30\n2,Bob,25\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
            "age": ar.Int64(),
        })

        result = ar.validate(frame, schema)

        assert result.passed is True
        assert result.issue_count == 0
        assert result.row_count == 2

    def test_validate_dtype_mismatch(self, tmp_path):
        """User detects type violations in data."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score\n1,not_a_number\n2,99.5\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "score": ar.Float64(),
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert result.issue_count > 0
        assert any(issue.rule == "dtype" for issue in result.issues)

    def test_validate_missing_required_column(self, tmp_path):
        """User defines schema with column that doesn't exist in CSV."""
        csv_path = tmp_path / "incomplete.csv"
        csv_path.write_text("id,name\n1,Alice\n2,Bob\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
            "email": ar.Email(nullable=False),  # This column is missing
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.rule == "required_column" for issue in result.issues)
        assert any("email" in issue.message for issue in result.issues)

    def test_validate_strict_mode_rejects_extra_columns(self, tmp_path):
        """Strict schema rejects unexpected columns in data."""
        csv_path = tmp_path / "extra_cols.csv"
        csv_path.write_text("id,name,debug_col\n1,Alice,xyz\n2,Bob,abc\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema(
            {"id": ar.Int64(), "name": ar.String()},
            strict=True,
        )

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.rule == "unexpected_column" for issue in result.issues)

    def test_validate_nullable_false_catches_nulls(self, tmp_path):
        """User requires non-null values and validation catches missing data."""
        csv_path = tmp_path / "sparse.csv"
        csv_path.write_text("id,name\n1,Alice\n2,\n3,Charlie\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(nullable=False),
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert 2 in result.bad_rows  # Row 2 (1-indexed) is bad
        assert 1 not in result.bad_rows
        assert 3 not in result.bad_rows

    def test_validate_unique_single_column(self, tmp_path):
        """User enforces uniqueness on a single column."""
        csv_path = tmp_path / "users.csv"
        csv_path.write_text("id,name\n1,Alice\n2,Bob\n1,Charlie\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(unique=True),
            "name": ar.String(),
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.rule == "unique" for issue in result.issues)
        assert 3 in result.bad_rows

    def test_validate_composite_unique_constraint(self, tmp_path):
        """User enforces composite uniqueness across multiple columns."""
        csv_path = tmp_path / "enrollments.csv"
        csv_path.write_text(
            "user_id,course_id,grade\n1,101,A\n1,102,B\n1,101,A\n"
        )

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema(
            {
                "user_id": ar.Int64(),
                "course_id": ar.Int64(),
                "grade": ar.String(),
            },
            unique=["user_id", "course_id"],
        )

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.rule == "composite_unique" for issue in result.issues)
        assert 3 in result.bad_rows

    def test_validate_email_field(self, tmp_path):
        """User validates email field format."""
        csv_path = tmp_path / "emails.csv"
        csv_path.write_text("email\nvalid@example.com\ninvalid_email\ntest@domain.co\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({"email": ar.Email(nullable=False)})

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.rule == "email" for issue in result.issues)
        assert 2 in result.bad_rows

    def test_validate_int_bounds(self, tmp_path):
        """User enforces min/max bounds on numeric fields."""
        csv_path = tmp_path / "ages.csv"
        csv_path.write_text("user_id,age\n1,25\n2,150\n3,30\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "age": ar.Int64(min=0, max=120),
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.rule == "max" for issue in result.issues)
        assert 2 in result.bad_rows

    def test_validate_cross_field_rule(self, tmp_path):
        """User defines custom rule validating relationships across columns."""
        csv_path = tmp_path / "dates.csv"
        csv_path.write_text("start_date,end_date\n2024-01-01,2024-01-05\n2024-02-01,2024-01-20\n")

        def end_after_start(df):
            """Custom rule: end_date must be >= start_date."""
            issues = []
            for i, row in df.iterrows():
                if str(row["end_date"]) < str(row["start_date"]):
                    issues.append(
                        ar.ValidationIssue(
                            column="end_date",
                            rule="cross_field",
                            message="end_date must be >= start_date",
                            row_index=int(i) + 1,
                        )
                    )
            return issues

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema(
            {"start_date": ar.String(), "end_date": ar.String()},
            rules=[end_after_start],
        )

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert any(issue.rule == "cross_field" for issue in result.issues)
        assert 2 in result.bad_rows

    def test_validate_max_errors_limits_issues(self, tmp_path):
        """User limits validation to first N issues for large datasets."""
        csv_path = tmp_path / "emails.csv"
        csv_path.write_text("email\ninvalid1\ninvalid2\ninvalid3\ninvalid4\ninvalid5\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({"email": ar.Email(nullable=False)})

        result = ar.validate(frame, schema, max_errors=2)

        assert result.passed is False
        assert result.issue_count == 2
        assert len(result.issues) == 2

    def test_validate_validation_result_metadata(self, tmp_path):
        """ValidationResult provides accurate metadata about issues."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score\n1,10\n2,\n3,30\n4,\n5,50\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "score": ar.Int64(nullable=False),
        })

        result = ar.validate(frame, schema)

        assert result.passed is False
        assert result.row_count == 5
        assert result.issue_count == 2
        assert 2 in result.bad_rows
        assert 4 in result.bad_rows
        assert 1 not in result.bad_rows

    def test_validate_via_pandas_accessor(self, tmp_path):
        """User validates pandas DataFrame using .arnio accessor."""
        csv_path = tmp_path / "customers.csv"
        csv_path.write_text("id,name,age\n1,Alice,30\n2,Bob,999\n")

        df = pd.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
            "age": ar.Int64(max=120),
        })

        result = df.arnio.validate(schema)

        assert result.passed is False
        assert any(issue.rule == "max" for issue in result.issues)
        assert 2 in result.bad_rows

    def test_validate_requires_arframe(self):
        """validate() raises TypeError if passed a plain DataFrame."""
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        schema = ar.Schema({"id": ar.Int64()})

        with pytest.raises(TypeError, match="single ArFrame"):
            ar.validate(df, schema)

    def test_validate_max_errors_validation(self, tmp_path):
        """validate() rejects invalid max_errors values."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id\n1\n2\n")
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({"id": ar.Int64()})

        with pytest.raises(TypeError, match="max_errors must be an int"):
            ar.validate(frame, schema, max_errors="invalid")

        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, schema, max_errors=0)

    def test_validate_empty_dataframe(self, tmp_path):
        """Validation handles empty CSV correctly."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("id,name\n")

        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})

        result = ar.validate(frame, schema)

        assert result.passed is True
        assert result.row_count == 0
        assert result.issue_count == 0
