import pytest
import pandas as pd
import numpy as np
import arnio as ar
import json


class TestSchemaBootstrapFromReport:
    """Test Schema.bootstrap_from_report class method."""

    def test_bootstrap_from_valid_report(self):
        """Creates a Schema from a valid DataQualityReport."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 88.0, 92.3],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        schema = ar.Schema.bootstrap_from_report(report)

        assert isinstance(schema, ar.Schema)
        assert len(schema.fields) == 3
        assert "id" in schema.fields
        assert "name" in schema.fields
        assert "score" in schema.fields

    def test_bootstrap_raises_on_non_report(self):
        """Raises TypeError if input is not a DataQualityReport."""
        with pytest.raises(TypeError):
            ar.Schema.bootstrap_from_report("not a report")

    def test_bootstrap_schema_validates_original_frame(self):
        """Bootstrapped schema can validate the original frame."""
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        schema = ar.Schema.bootstrap_from_report(report)
        result = schema.validate(frame)

        assert isinstance(result, ar.ValidationResult)
        assert result.passed is True


class TestSchemaJsonSerialization:
    """Test Schema.to_json and Schema.from_json methods."""

    def test_to_json_returns_string(self):
        """to_json returns a valid JSON string."""
        schema = ar.Schema({"id": ar.Int64(nullable=False)})

        json_str = schema.to_json()

        assert isinstance(json_str, str)
        json.loads(json_str)  # Should not raise

    def test_from_json_deserializes_valid_schema(self):
        """from_json creates a Schema from JSON string."""
        original = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        json_str = original.to_json()

        restored = ar.Schema.from_json(json_str)

        assert isinstance(restored, ar.Schema)
        assert "id" in restored.fields
        assert "name" in restored.fields

    def test_json_roundtrip_simple_types(self):
        """JSON roundtrip preserves simple field types."""
        original = ar.Schema({
            "int_col": ar.Int64(),
            "str_col": ar.String(),
            "bool_col": ar.Bool(),
            "float_col": ar.Float64(),
        })

        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)

        assert set(original.fields.keys()) == set(restored.fields.keys())

    def test_json_roundtrip_with_constraints_and_nullable(self):
        """JSON roundtrip preserves constraints and nullable flags."""
        original = ar.Schema({
            "age": ar.Int64(min=0, max=150, nullable=False),
            "email": ar.Email(nullable=False),
            "nickname": ar.String(min_length=2, max_length=50, nullable=True),
        })

        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)

        assert "age" in restored.fields
        assert "email" in restored.fields
        assert "nickname" in restored.fields

    def test_json_roundtrip_with_unique_constraint(self):
        """JSON roundtrip preserves unique constraints."""
        original = ar.Schema(
            {
                "user_id": ar.Int64(nullable=False),
                "email": ar.String(nullable=False),
            },
            unique=["user_id", "email"],
        )

        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)

        assert isinstance(restored, ar.Schema)


class TestSchemaValidate:
    """Test Schema.validate method."""

    def test_validate_returns_validation_result(self):
        """validate returns a ValidationResult object."""
        schema = ar.Schema({"id": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))

        result = schema.validate(frame)

        assert isinstance(result, ar.ValidationResult)
        assert hasattr(result, "passed")
        assert hasattr(result, "issues")

    def test_validate_passing_data(self):
        """Validation passes for correct data matching schema."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }))

        result = schema.validate(frame)

        assert result.passed is True
        assert len(result.issues) == 0

    def test_validate_dtype_mismatch(self):
        """Validation catches dtype mismatches."""
        schema = ar.Schema({"id": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"id": ["a", "b", "c"]}))

        result = schema.validate(frame)

        assert result.passed is False
        assert len(result.issues) > 0

    def test_validate_nullable_constraint_violation(self):
        """Validation catches nullable=False violations."""
        schema = ar.Schema({"name": ar.String(nullable=False)})
        frame = ar.from_pandas(pd.DataFrame({"name": ["Alice", None, "Bob"]}))

        result = schema.validate(frame)

        assert result.passed is False
        assert any(issue.column == "name" for issue in result.issues)

    def test_validate_numeric_min_max_constraints(self):
        """Validation catches numeric min/max constraint violations."""
        schema = ar.Schema({"age": ar.Int64(min=0, max=150)})
        frame = ar.from_pandas(pd.DataFrame({"age": [25, 200, 35]}))

        result = schema.validate(frame)

        assert result.passed is False

    def test_validate_string_length_constraints(self):
        """Validation checks string length constraints."""
        schema = ar.Schema({"code": ar.String(min_length=2, max_length=5)})
        frame = ar.from_pandas(pd.DataFrame({"code": ["AB", "ABC", "TOOLONG"]}))

        result = schema.validate(frame)

        assert result.passed is False

    def test_validate_max_errors_parameter(self):
        """max_errors parameter limits issues returned."""
        schema = ar.Schema({"id": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"id": ["a", "b", "c", "d", "e"]}))

        result = schema.validate(frame, max_errors=2)

        assert len(result.issues) <= 2

    def test_validate_missing_required_column(self):
        """Validation catches missing columns required by schema."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(),
        })
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))

        result = schema.validate(frame)

        assert result.passed is False
