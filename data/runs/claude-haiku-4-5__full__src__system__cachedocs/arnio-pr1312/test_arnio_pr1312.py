import json
import pytest
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import arnio as ar


class TestBootstrapFromReport:
    """Test Schema.bootstrap_from_report() with profiling workflows."""

    def test_bootstrap_basic_and_validate(self, tmp_path):
        """Profile a frame, bootstrap schema, validate same frame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,Alice,95\n2,Bob,87\n3,Charlie,92\n")

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)

        result = schema.validate(frame)
        assert result.passed
        assert len(result.issues) == 0

    def test_bootstrap_marks_nullable_columns(self, tmp_path):
        """Columns with nulls should be marked nullable=True in bootstrapped schema."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,Alice,95\n2,,\n3,Charlie,92\n")

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)

        # Columns with observed nulls should allow nulls
        assert schema.fields["name"].nullable
        assert schema.fields["score"].nullable
        # id has no nulls
        assert not schema.fields["id"].nullable

        result = schema.validate(frame)
        assert result.passed

    def test_bootstrap_error_on_non_report(self):
        """bootstrap_from_report should raise TypeError for non-report input."""
        with pytest.raises(TypeError, match="Expected DataQualityReport"):
            ar.Schema.bootstrap_from_report({"not": "a report"})

    def test_bootstrap_error_on_empty_report(self):
        """bootstrap_from_report should raise ValueError when report has no columns."""
        from arnio.quality import DataQualityReport

        empty_report = DataQualityReport(
            row_count=0,
            column_count=0,
            columns={},
            memory_usage=0,
            duplicate_rows=0,
            quality_score=100.0,
        )
        with pytest.raises(ValueError, match="no columns"):
            ar.Schema.bootstrap_from_report(empty_report)

    def test_bootstrap_then_clean_then_validate(self, tmp_path):
        """End-to-end: profile → bootstrap → clean → validate."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("id,name,score\n 1 , Alice , 95 \n 2 , Bob , 87 \n")

        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)

        # Clean whitespace
        clean_frame = ar.pipeline(frame, [("strip_whitespace",)])

        # Validate cleaned data
        result = schema.validate(clean_frame)
        assert result.passed


class TestSchemaJsonRoundTrip:
    """Test Schema.to_json() and Schema.from_json() serialization."""

    def test_simple_schema_roundtrip(self):
        """Basic schema should serialize and deserialize correctly."""
        schema = ar.Schema(
            {
                "id": ar.Int64(),
                "name": ar.String(),
            }
        )

        json_str = schema.to_json()
        restored = ar.Schema.from_json(json_str)

        assert set(restored.fields.keys()) == {"id", "name"}
        assert restored.fields["id"].dtype == "int64"
        assert restored.fields["name"].dtype == "string"

    def test_schema_with_constraints_roundtrip(self):
        """Schema with nullable, unique, min/max should roundtrip."""
        schema = ar.Schema(
            {
                "user_id": ar.Int64(nullable=False),
                "email": ar.Email(nullable=False),
                "age": ar.Int64(nullable=True, min=0, max=150),
            },
            strict=True,
            unique=["user_id"],
        )

        json_str = schema.to_json()
        restored = ar.Schema.from_json(json_str)

        assert restored.strict
        assert restored.unique == ["user_id"]
        assert not restored.fields["user_id"].nullable
        assert restored.fields["age"].nullable
        assert restored.fields["age"].min == 0
        assert restored.fields["age"].max == 150

    def test_to_json_warns_when_rules_present(self):
        """to_json should warn and mark rules_omitted when rules exist."""

        def dummy_rule(df):
            return []

        schema = ar.Schema({"a": ar.String()}, rules=[dummy_rule])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            json_str = schema.to_json()
            assert len(w) == 1
            assert "rules cannot be serialized" in str(w[0].message)

        payload = json.loads(json_str)
        assert payload.get("rules_omitted") is True

    def test_from_json_invalid_json_raises_error(self):
        """from_json should raise ValueError on malformed JSON."""
        with pytest.raises(ValueError, match="Invalid schema JSON"):
            ar.Schema.from_json("{not valid json")

    def test_from_json_missing_fields_raises_error(self):
        """from_json should raise TypeError when 'fields' key is missing."""
        with pytest.raises(TypeError, match="fields"):
            ar.Schema.from_json(json.dumps({"strict": True}))

    def test_from_json_rejects_unknown_keys(self):
        """from_json should raise ValueError for unrecognized schema keys."""
        with pytest.raises(ValueError, match="unknown key"):
            ar.Schema.from_json(
                json.dumps(
                    {
                        "fields": {"a": {"dtype": "string"}},
                        "strict": True,
                        "unknown_key": "value",
                    }
                )
            )

    def test_json_persist_to_file_and_reload(self, tmp_path):
        """Schema can be persisted to JSON file and reloaded."""
        schema_file = tmp_path / "schema.json"

        # Save schema
        schema = ar.Schema(
            {
                "product_id": ar.Int64(nullable=False),
                "price": ar.Float64(min=0.0),
            }
        )
        schema_json = schema.to_json()
        schema_file.write_text(schema_json)

        # Reload and verify
        reloaded_json = schema_file.read_text()
        reloaded_schema = ar.Schema.from_json(reloaded_json)

        assert reloaded_schema.fields["product_id"].dtype == "int64"
        assert not reloaded_schema.fields["product_id"].nullable
        assert reloaded_schema.fields["price"].min == 0.0


class TestSchemaValidate:
    """Test Schema.validate() method."""

    def test_validate_passing_data(self, tmp_path):
        """validate should return passed=True for valid data."""
        csv_file = tmp_path / "users.csv"
        csv_file.write_text(
            "user_id,email,age\n1,alice@example.com,30\n2,bob@example.com,25\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {
                "user_id": ar.Int64(nullable=False),
                "email": ar.Email(nullable=False),
                "age": ar.Int64(nullable=True, min=0, max=150),
            }
        )

        result = schema.validate(frame)
        assert result.passed
        assert result.issue_count == 0

    def test_validate_detects_invalid_email(self, tmp_path):
        """validate should detect email format violations."""
        csv_file = tmp_path / "users.csv"
        csv_file.write_text(
            "user_id,email,age\n1,not-an-email,30\n2,bob@example.com,25\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {
                "user_id": ar.Int64(),
                "email": ar.Email(nullable=False),
                "age": ar.Int64(nullable=True),
            }
        )

        result = schema.validate(frame)
        assert not result.passed
        assert result.issue_count > 0
        email_issues = [i for i in result.issues if i.column == "email"]
        assert len(email_issues) > 0

    def test_validate_detects_numeric_constraint_violations(self, tmp_path):
        """validate should detect min/max violations on numeric fields."""
        csv_file = tmp_path / "scores.csv"
        csv_file.write_text("id,score\n1,500\n2,10\n3,-5\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {
                "id": ar.Int64(),
                "score": ar.Int64(min=0, max=100),
            }
        )

        result = schema.validate(frame)
        assert not result.passed
        score_issues = [i for i in result.issues if i.column == "score"]
        assert len(score_issues) > 0

    def test_validate_max_errors_parameter(self, tmp_path):
        """validate should stop after reaching max_errors limit."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("score\n500\n-10\n999\n-50\n100\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"score": ar.Int64(min=0, max=100)})

        result_limited = schema.validate(frame, max_errors=2)
        result_unlimited = schema.validate(frame, max_errors=None)

        assert result_limited.issue_count <= 2
        assert result_unlimited.issue_count >= result_limited.issue_count

    def test_validate_empty_frame_passes(self, tmp_path):
        """validate should pass on empty frame (no data rows)."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("id,name\n")

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {
                "id": ar.Int64(),
                "name": ar.String(),
            }
        )

        result = schema.validate(frame)
        assert result.passed

    def test_validate_via_pandas_accessor(self):
        """validate should work through pandas .arnio accessor."""
        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
            }
        )

        schema = ar.Schema(
            {
                "user_id": ar.Int64(nullable=False),
                "email": ar.Email(nullable=False),
            }
        )

        result = df.arnio.validate(schema)
        assert result.passed

    def test_validate_with_cross_field_rule(self, tmp_path):
        """validate should apply custom cross-field validation rules."""
        csv_file = tmp_path / "dates.csv"
        csv_file.write_text("start_date,end_date\n2024-01-01,2024-01-31\n2024-02-15,2024-02-01\n")

        def end_after_start(df):
            issues = []
            for i, row in df.iterrows():
                if row["end_date"] < row["start_date"]:
                    issues.append(
                        ar.ValidationIssue(
                            column="end_date",
                            rule="cross_field",
                            message="end_date must be >= start_date",
                            row_index=int(i) + 1,
                        )
                    )
            return issues

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema(
            {
                "start_date": ar.String(),
                "end_date": ar.String(),
            },
            rules=[end_after_start],
        )

        result = schema.validate(frame)
        assert not result.passed
        cross_field_issues = [i for i in result.issues if i.rule == "cross_field"]
        assert len(cross_field_issues) > 0


class TestEndToEndIntegration:
    """Test complete workflows combining schema, profiling, and validation."""

    def test_profile_bootstrap_clean_validate_write(self, tmp_path):
        """Complete workflow: profile → bootstrap → clean → validate → write."""
        raw_csv = tmp_path / "raw.csv"
        raw_csv.write_text("id,name,score\n 1 , Alice , 95 \n 2 , Bob , 88 \n")

        frame = ar.read_csv(str(raw_csv))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)

        # Clean the frame
        clean_frame = ar.pipeline(frame, [("strip_whitespace",)])

        # Validate cleaned data
        result = schema.validate(clean_frame)
        assert result.passed

        # Write output
        output_csv = tmp_path / "clean.csv"
        ar.write_csv(clean_frame, str(output_csv))
        assert output_csv.exists()

        output_text = output_csv.read_text()
        assert "Alice" in output_text
        assert "Bob" in output_text

    def test_define_schema_persist_validate_workflow(self, tmp_path):
        """Workflow: define schema → serialize → load → validate."""
        csv_file = tmp_path / "products.csv"
        csv_file.write_text("product_id,price\n1,29.99\n2,49.99\n3,-5.00\n")

        # Define and persist schema
        schema = ar.Schema(
            {
                "product_id": ar.Int64(nullable=False),
                "price": ar.Float64(nullable=False, min=0.0),
            }
        )
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(schema.to_json())

        # Load schema and validate
        loaded_schema_json = schema_file.read_text()
        loaded_schema = ar.Schema.from_json(loaded_schema_json)

        frame = ar.read_csv(str(csv_file))
        result = loaded_schema.validate(frame)

        assert not result.passed
        price_issues = [i for i in result.issues if i.column == "price"]
        assert len(price_issues) > 0
        assert any("min" in issue.message.lower() for issue in price_issues)

    def test_pandas_accessor_full_workflow(self, tmp_path):
        """Test full workflow using pandas accessor."""
        df = pd.DataFrame(
            {
                "order_id": [1, 2, 3],
                "customer_name": ["  Alice  ", "Bob", "  Charlie  "],
                "amount": [100.5, 200.0, 50.75],
            }
        )

        # Clean via accessor
        clean_df = df.arnio.clean(strip_whitespace=True)

        # Validate via accessor
        schema = ar.Schema(
            {
                "order_id": ar.Int64(nullable=False),
                "customer_name": ar.String(nullable=False, min_length=1),
                "amount": ar.Float64(nullable=False, min=0.0),
            }
        )

        result = clean_df.arnio.validate(schema)
        assert result.passed
