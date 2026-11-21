import pytest
import pandas as pd
import arnio as ar
from pathlib import Path


class TestValidateSchemaSystem:
    """System tests for arnio.schema.validate() covering end-to-end workflows."""

    def test_validate_passing_schema(self, tmp_path):
        """Test that validation passes when data matches schema."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,age\n1,Alice,30\n2,Bob,25\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {
                "id": ar.Int64(),
                "name": ar.String(),
                "age": ar.Int64(),
            }
        )

        result = ar.validate(frame, schema)

        assert result.passed
        assert result.issue_count == 0
        assert len(result.issues) == 0
        assert result.row_count == 2

    def test_validate_failing_dtype_mismatch(self, tmp_path):
        """Test that validation catches dtype mismatches."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\nabc,Bob\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})

        result = ar.validate(frame, schema)

        assert not result.passed
        assert result.issue_count > 0
        assert any(i.column == "id" for i in result.issues)

    def test_validate_email_field(self, tmp_path):
        """Test email field validation with valid and invalid formats."""
        csv_file = tmp_path / "users.csv"
        csv_file.write_text(
            "id,email\n1,alice@example.com\n2,invalid-email\n3,bob@example.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"id": ar.Int64(), "email": ar.Email()})

        result = ar.validate(frame, schema)

        assert not result.passed
        email_issues = [i for i in result.issues if i.column == "email"]
        assert len(email_issues) > 0
        assert any("invalid-email" in str(i.message) for i in email_issues)

    def test_validate_nullable_and_required(self, tmp_path):
        """Test nullable=False constraint catches null values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,required_name,optional_desc\n1,Alice,desc1\n2,,desc2\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {
                "id": ar.Int64(),
                "required_name": ar.String(nullable=False),
                "optional_desc": ar.String(nullable=True),
            }
        )

        result = ar.validate(frame, schema)

        assert not result.passed
        # Should report issue on row 2 for required_name
        assert any(
            i.row_index == 2 and i.column == "required_name" for i in result.issues
        )

    def test_validate_unique_single_column(self, tmp_path):
        """Test single-column uniqueness constraint."""
        csv_file = tmp_path / "users.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n1,Charlie\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"id": ar.Int64(unique=True), "name": ar.String()})

        result = ar.validate(frame, schema)

        assert not result.passed
        # Row 3 has duplicate id=1
        assert 3 in result.bad_rows

    def test_validate_unique_composite_key(self, tmp_path):
        """Test composite (multi-column) uniqueness constraint."""
        csv_file = tmp_path / "enrollments.csv"
        csv_file.write_text(
            "student_id,course_id,semester\n"
            "1,101,2024-01\n"
            "1,102,2024-01\n"
            "1,101,2024-01\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {
                "student_id": ar.Int64(),
                "course_id": ar.Int64(),
                "semester": ar.String(),
            },
            unique=["student_id", "course_id", "semester"],
        )

        result = ar.validate(frame, schema)

        assert not result.passed
        composite_issues = [i for i in result.issues if i.rule == "composite_unique"]
        assert len(composite_issues) > 0

    def test_validate_max_errors_parameter(self, tmp_path):
        """Test max_errors parameter limits issue collection."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("value\naaa\nbbb\nccc\nddd\neee\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"value": ar.Int64()})

        result_all = ar.validate(frame, schema)
        result_limited = ar.validate(frame, schema, max_errors=2)

        assert result_limited.issue_count == 2
        assert result_all.issue_count == 5

    def test_validate_missing_required_column(self, tmp_path):
        """Test that validation fails when required columns are missing."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n")

        frame = ar.read_csv(str(csv_file))
        # Schema requires 'email' column that doesn't exist in CSV
        schema = ar.Schema(
            {"id": ar.Int64(), "name": ar.String(), "email": ar.Email()}
        )

        result = ar.validate(frame, schema)

        assert not result.passed
        assert any(i.rule == "required_column" and i.column == "email" for i in result.issues)

    def test_validate_strict_mode_unexpected_columns(self, tmp_path):
        """Test strict=True rejects unexpected columns in data."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,debug_col\n1,Alice,extra\n2,Bob,extra\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()}, strict=True)

        result = ar.validate(frame, schema)

        assert not result.passed
        assert any(
            i.rule == "unexpected_column" and i.column == "debug_col"
            for i in result.issues
        )

    def test_validate_custom_cross_field_rule(self, tmp_path):
        """Test custom validation rule for cross-field constraints."""
        csv_file = tmp_path / "orders.csv"
        csv_file.write_text(
            "order_id,quantity,price\n1,5,100.0\n2,-1,50.0\n3,0,25.0\n4,2,0.0\n"
        )

        frame = ar.read_csv(str(csv_file))

        def positive_amounts(df):
            """Custom rule: quantity and price must both be positive."""
            issues = []
            for idx, row in df.iterrows():
                qty = row.get("quantity")
                price = row.get("price")
                if pd.notna(qty) and qty <= 0:
                    issues.append(
                        ar.ValidationIssue(
                            column="quantity",
                            rule="positive_amount",
                            message="Quantity must be positive",
                            row_index=int(idx) + 1,
                        )
                    )
                if pd.notna(price) and price <= 0:
                    issues.append(
                        ar.ValidationIssue(
                            column="price",
                            rule="positive_amount",
                            message="Price must be positive",
                            row_index=int(idx) + 1,
                        )
                    )
            return issues

        schema = ar.Schema(
            {
                "order_id": ar.Int64(),
                "quantity": ar.Int64(),
                "price": ar.Float64(),
            },
            rules=[positive_amounts],
        )

        result = ar.validate(frame, schema)

        assert not result.passed
        custom_issues = [i for i in result.issues if i.rule == "positive_amount"]
        assert len(custom_issues) > 0

    def test_validate_pandas_accessor_integration(self):
        """Test validation through pandas .arnio accessor."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "email": ["alice@example.com", "not-an-email", "bob@test.org"],
                "age": [25, 30, 28],
            }
        )

        schema = ar.Schema(
            {
                "id": ar.Int64(),
                "email": ar.Email(),
                "age": ar.Int64(min=18, max=120),
            }
        )

        result = df.arnio.validate(schema)

        assert isinstance(result, ar.ValidationResult)
        assert not result.passed
        email_issues = [i for i in result.issues if "not-an-email" in str(i.message)]
        assert len(email_issues) > 0

    def test_validate_result_to_markdown_format(self, tmp_path):
        """Test ValidationResult.to_markdown() generates valid markdown."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,abc\n2,def\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"id": ar.Int64(), "value": ar.Int64()})

        result = ar.validate(frame, schema)
        markdown_output = result.to_markdown()

        assert isinstance(markdown_output, str)
        assert len(markdown_output) > 0
        # Markdown report should contain table or header markers
        assert any(marker in markdown_output for marker in ["|", "#", "-"])

    def test_validate_result_to_pandas_dataframe(self, tmp_path):
        """Test ValidationResult.to_pandas() converts issues to DataFrame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("code\nabc\ndef\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"code": ar.Int64(nullable=False)})

        result = ar.validate(frame, schema)
        issues_df = result.to_pandas()

        assert isinstance(issues_df, pd.DataFrame)
        assert len(issues_df) == result.issue_count
        assert all(col in issues_df.columns for col in ["column", "rule", "message"])

    def test_validate_url_field_validation(self, tmp_path):
        """Test URL field validation with valid and invalid URLs."""
        csv_file = tmp_path / "websites.csv"
        csv_file.write_text(
            "id,url\n"
            "1,https://example.com\n"
            "2,not-a-url\n"
            "3,http://test.org/path\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"id": ar.Int64(), "url": ar.URL()})

        result = ar.validate(frame, schema)

        assert not result.passed
        url_issues = [i for i in result.issues if i.column == "url"]
        assert len(url_issues) > 0

    def test_validate_error_non_arframe_input(self):
        """Test that validate() raises TypeError when given non-ArFrame input."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        schema = ar.Schema({"id": ar.Int64()})

        with pytest.raises(TypeError, match="single ArFrame"):
            ar.validate(df, schema)

    def test_validate_error_invalid_max_errors(self, tmp_path):
        """Test that validate() raises ValueError for invalid max_errors values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id\n1\n2\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"id": ar.Int64()})

        # max_errors <= 0 should raise ValueError
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, schema, max_errors=0)

        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, schema, max_errors=-1)

        # max_errors as non-int should raise TypeError
        with pytest.raises(TypeError, match="max_errors must be an int"):
            ar.validate(frame, schema, max_errors="invalid")

    def test_validate_numeric_constraints(self, tmp_path):
        """Test min/max constraints on numeric fields."""
        csv_file = tmp_path / "scores.csv"
        csv_file.write_text("test_id,score\n1,95\n2,105\n3,50\n4,-10\n5,85\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {"test_id": ar.Int64(), "score": ar.Int64(min=0, max=100)}
        )

        result = ar.validate(frame, schema)

        assert not result.passed
        # Rows 2 and 4 should have violations
        assert len(result.bad_rows) >= 2

    def test_validate_empty_frame(self, tmp_path):
        """Test validation on empty frame (header only)."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("id,name,value\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {
                "id": ar.Int64(),
                "name": ar.String(),
                "value": ar.Float64(nullable=True),
            }
        )

        result = ar.validate(frame, schema)

        # Empty frame with valid schema should pass
        assert result.passed
        assert result.row_count == 0
        assert result.issue_count == 0
