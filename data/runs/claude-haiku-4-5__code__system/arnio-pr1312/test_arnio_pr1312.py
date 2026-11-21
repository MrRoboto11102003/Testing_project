import json
import warnings

import pandas as pd
import pytest

import arnio as ar


class TestBootstrapFromReport:
    """Test Schema.bootstrap_from_report"""

    def test_bootstrap_basic_profile(self):
        """Bootstrap schema from a profile of a simple DataFrame."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        
        report = ar.profile(df)
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert isinstance(schema, ar.Schema)
        assert "id" in schema.fields
        assert "name" in schema.fields

    def test_bootstrap_detects_nullable_columns(self):
        """Bootstrap sets nullable=True for columns with nulls."""
        df = pd.DataFrame({
            "no_nulls": [1, 2, 3],
            "has_nulls": [1, None, 3]
        })
        
        report = ar.profile(df)
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert schema.fields["no_nulls"].nullable is False
        assert schema.fields["has_nulls"].nullable is True

    def test_bootstrap_raises_typeerror_for_non_report(self):
        """bootstrap_from_report raises TypeError for non-DataQualityReport."""
        with pytest.raises(TypeError, match="Expected DataQualityReport"):
            ar.Schema.bootstrap_from_report("not a report")

    def test_bootstrap_raises_valueerror_for_empty_report(self):
        """bootstrap_from_report raises ValueError for empty report."""
        empty_df = pd.DataFrame()
        report = ar.profile(empty_df)
        
        with pytest.raises(ValueError, match="Cannot bootstrap schema from an empty report"):
            ar.Schema.bootstrap_from_report(report)


class TestToJsonFromJson:
    """Test Schema serialization and deserialization"""

    def test_to_json_produces_valid_json(self):
        """to_json() produces parseable JSON with required fields."""
        schema = ar.Schema(fields={
            "id": ar.Field(dtype="int64"),
            "name": ar.Field(dtype="string")
        })
        
        json_str = schema.to_json()
        payload = json.loads(json_str)
        
        assert "fields" in payload
        assert "strict" in payload
        assert isinstance(payload["fields"], dict)

    def test_to_json_preserves_schema_properties(self):
        """to_json() includes strict and unique properties."""
        schema = ar.Schema(
            fields={"col": ar.Field(dtype="int64")},
            strict=True,
            unique=["col"]
        )
        
        payload = json.loads(schema.to_json())
        
        assert payload["strict"] is True
        assert payload.get("unique") == ["col"]

    def test_to_json_warns_about_omitted_rules(self):
        """to_json() emits UserWarning when rules are present."""
        schema = ar.Schema(
            fields={"id": ar.Field(dtype="int64")},
            rules=["some_rule"]
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            json_str = schema.to_json()
            
            assert len(w) >= 1
            assert issubclass(w[0].category, UserWarning)
            assert "rules" in str(w[0].message).lower()
            
            payload = json.loads(json_str)
            assert payload.get("rules_omitted") is True

    def test_from_json_basic_schema(self):
        """from_json() deserializes basic schema correctly."""
        json_str = json.dumps({
            "fields": {"id": {"dtype": "int64", "nullable": False}},
            "strict": False,
            "unique": None
        })
        
        schema = ar.Schema.from_json(json_str)
        
        assert isinstance(schema, ar.Schema)
        assert "id" in schema.fields
        assert schema.fields["id"].nullable is False

    def test_from_json_raises_on_invalid_json(self):
        """from_json() raises ValueError for malformed JSON."""
        with pytest.raises(ValueError, match="Invalid schema JSON"):
            ar.Schema.from_json("{not valid")

    def test_from_json_raises_on_non_dict_json(self):
        """from_json() raises TypeError if JSON is not a dict."""
        with pytest.raises(TypeError, match="must decode to an object"):
            ar.Schema.from_json(json.dumps([]))

    def test_from_json_raises_on_missing_fields_key(self):
        """from_json() raises TypeError if 'fields' key is missing."""
        with pytest.raises(TypeError, match="must be an object"):
            ar.Schema.from_json(json.dumps({"strict": False}))

    def test_from_json_raises_on_unknown_keys(self):
        """from_json() raises ValueError for unknown schema keys."""
        with pytest.raises(ValueError, match="unknown key"):
            ar.Schema.from_json(json.dumps({
                "fields": {},
                "strict": False,
                "badkey": True
            }))

    def test_from_json_raises_on_non_bool_strict(self):
        """from_json() raises TypeError if strict is not bool."""
        with pytest.raises(TypeError, match="must be a boolean"):
            ar.Schema.from_json(json.dumps({
                "fields": {},
                "strict": "yes"
            }))

    def test_from_json_ignores_rules_omitted_marker(self):
        """from_json() silently ignores rules_omitted marker."""
        schema = ar.Schema.from_json(json.dumps({
            "fields": {},
            "strict": False,
            "rules_omitted": True
        }))
        
        assert isinstance(schema, ar.Schema)

    def test_roundtrip_preserves_schema(self):
        """Schema survives to_json() and from_json() roundtrip."""
        original = ar.Schema(
            fields={
                "id": ar.Field(dtype="int64", nullable=False),
                "name": ar.Field(dtype="string", nullable=True)
            },
            strict=True,
            unique=["id"]
        )
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        assert restored.strict == original.strict
        assert set(restored.fields.keys()) == set(original.fields.keys())
        assert restored.fields["name"].nullable == original.fields["name"].nullable
        assert restored.unique == original.unique


class TestValidate:
    """Test Schema.validate"""

    def test_validate_returns_result(self):
        """validate() returns a ValidationResult."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        schema = ar.Schema(fields={"id": ar.Field(dtype="int64")})
        
        result = schema.validate(df)
        
        assert result is not None

    def test_validate_with_max_errors_parameter(self):
        """validate() accepts max_errors parameter."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        schema = ar.Schema(fields={"value": ar.Field(dtype="int64")})
        
        result = schema.validate(df, max_errors=5)
        
        assert result is not None


class TestCompleteWorkflow:
    """Test complete user data-preparation workflow"""

    def test_full_data_pipeline_profile_bootstrap_persist_validate(self, tmp_path):
        """Profile data, bootstrap schema, persist to JSON, reload, and validate."""
        # Create test CSV
        csv_path = tmp_path / "input.csv"
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "username": ["alice", "bob", "charlie", "david", "eve"],
            "score": [95.5, None, 87.3, 92.1, 88.9]
        })
        df.to_csv(csv_path, index=False)
        
        # Load data
        data = pd.read_csv(csv_path)
        
        # Profile
        report = ar.profile(data)
        
        # Bootstrap schema from report
        schema = ar.Schema.bootstrap_from_report(report)
        assert len(schema.fields) == 3
        assert schema.fields["score"].nullable is True
        
        # Persist schema to JSON
        schema_path = tmp_path / "schema.json"
        schema_json = schema.to_json()
        schema_path.write_text(schema_json)
        
        # Reload schema from JSON
        schema_json_restored = schema_path.read_text()
        schema_restored = ar.Schema.from_json(schema_json_restored)
        
        # Validate original data against restored schema
        result = schema_restored.validate(data)
        assert result is not None
        
        # Verify file artifacts
        assert schema_path.exists()
        assert len(schema_json) > 0
