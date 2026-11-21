import pytest
import json
import pandas as pd
import warnings
from unittest.mock import Mock, MagicMock
import arnio as ar


class TestBootstrapFromReport:
    """Test Schema.bootstrap_from_report() class method."""

    def test_bootstrap_valid_report_creates_schema(self):
        """Bootstrap from valid DataQualityReport creates Schema with correct fields."""
        from arnio.quality import DataQualityReport
        report = MagicMock(spec=DataQualityReport)
        
        profile_col1 = MagicMock()
        profile_col1.dtype = "int64"
        profile_col1.null_count = 0
        
        profile_col2 = MagicMock()
        profile_col2.dtype = "object"
        profile_col2.null_count = 3
        
        report.columns = {"col1": profile_col1, "col2": profile_col2}
        
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert isinstance(schema, ar.Schema)
        assert "col1" in schema.fields
        assert "col2" in schema.fields
        assert schema.fields["col1"].nullable is False
        assert schema.fields["col2"].nullable is True

    def test_bootstrap_raises_type_error_on_invalid_input(self):
        """Raises TypeError when input is not DataQualityReport."""
        with pytest.raises(TypeError, match="Expected DataQualityReport"):
            ar.Schema.bootstrap_from_report({"not": "report"})

    def test_bootstrap_raises_value_error_on_empty_report(self):
        """Raises ValueError when report has no columns."""
        from arnio.quality import DataQualityReport
        report = MagicMock(spec=DataQualityReport)
        report.columns = {}
        
        with pytest.raises(ValueError, match="empty report"):
            ar.Schema.bootstrap_from_report(report)

    def test_bootstrap_raises_value_error_on_missing_dtype(self):
        """Raises ValueError when column profile missing dtype."""
        from arnio.quality import DataQualityReport
        report = MagicMock(spec=DataQualityReport)
        
        profile = MagicMock(spec=[])
        report.columns = {"col1": profile}
        
        with pytest.raises(ValueError, match="missing 'dtype'"):
            ar.Schema.bootstrap_from_report(report)


class TestFromJson:
    """Test Schema.from_json() deserialization method."""

    def test_from_json_valid_minimal_schema(self):
        """Deserialize minimal valid schema JSON."""
        json_str = '{"fields": {}, "strict": false}'
        schema = ar.Schema.from_json(json_str)
        
        assert isinstance(schema, ar.Schema)
        assert schema.fields == {}
        assert schema.strict is False
        assert schema.unique is None

    def test_from_json_with_unique_constraint(self):
        """Deserialize schema with unique constraint."""
        json_str = '{"fields": {}, "strict": false, "unique": ["col1", "col2"]}'
        schema = ar.Schema.from_json(json_str)
        
        assert schema.unique == ["col1", "col2"]

    def test_from_json_invalid_syntax_raises_value_error(self):
        """Raises ValueError on invalid JSON syntax."""
        with pytest.raises(ValueError, match="Invalid schema JSON"):
            ar.Schema.from_json("{not valid json}")

    def test_from_json_root_not_dict_raises_type_error(self):
        """Raises TypeError if JSON root is not a dict."""
        with pytest.raises(TypeError):
            ar.Schema.from_json('["array", "not", "object"]')

    def test_from_json_unknown_keys_raises_value_error(self):
        """Raises ValueError for unknown keys in schema JSON."""
        with pytest.raises(ValueError, match="unknown key"):
            ar.Schema.from_json('{"fields": {}, "strict": false, "mystery_key": 123}')

    def test_from_json_fields_not_dict_raises_type_error(self):
        """Raises TypeError if 'fields' is not a dict."""
        with pytest.raises(TypeError):
            ar.Schema.from_json('{"fields": [], "strict": false}')

    def test_from_json_strict_not_bool_raises_type_error(self):
        """Raises TypeError if 'strict' is not boolean."""
        with pytest.raises(TypeError):
            ar.Schema.from_json('{"fields": {}, "strict": "false"}')

    def test_from_json_unique_not_list_raises_type_error(self):
        """Raises TypeError if 'unique' is not list or null."""
        with pytest.raises(TypeError):
            ar.Schema.from_json('{"fields": {}, "strict": false, "unique": "col1"}')

    def test_from_json_with_rules_omitted_marker_ignored(self):
        """Silently ignores 'rules_omitted' marker in JSON."""
        json_str = '{"fields": {}, "strict": false, "rules_omitted": true}'
        schema = ar.Schema.from_json(json_str)
        assert isinstance(schema, ar.Schema)


class TestToJson:
    """Test Schema.to_json() serialization method."""

    def test_to_json_produces_valid_json(self):
        """to_json() produces valid JSON that can be parsed."""
        schema = ar.Schema(fields={}, strict=False)
        json_str = schema.to_json()
        
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "fields" in parsed
        assert "strict" in parsed

    def test_to_json_roundtrip_preserves_schema(self):
        """Schema survives roundtrip through to_json/from_json."""
        schema1 = ar.Schema(fields={}, strict=True, unique=None)
        json_str = schema1.to_json()
        schema2 = ar.Schema.from_json(json_str)
        
        assert schema2.strict == schema1.strict
        assert schema2.unique == schema1.unique

    def test_to_json_with_rules_emits_user_warning(self):
        """to_json() emits UserWarning when rules are present."""
        schema = ar.Schema(fields={})
        schema.rules = [Mock()]
        
        with pytest.warns(UserWarning, match="rules"):
            schema.to_json()

    def test_to_json_includes_rules_omitted_when_rules_present(self):
        """JSON includes 'rules_omitted: true' when rules were present."""
        schema = ar.Schema(fields={})
        schema.rules = [Mock()]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            json_str = schema.to_json()
        
        parsed = json.loads(json_str)
        assert parsed.get("rules_omitted") is True

    def test_to_json_no_rules_omitted_when_empty_rules(self):
        """JSON does not include 'rules_omitted' when rules list empty."""
        schema = ar.Schema(fields={}, strict=False)
        schema.rules = []
        
        json_str = schema.to_json()
        parsed = json.loads(json_str)
        
        assert "rules_omitted" not in parsed or parsed.get("rules_omitted") is False


class TestValidate:
    """Test Schema.validate() method."""

    def test_validate_returns_validation_result(self):
        """validate() returns a ValidationResult object."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        schema = ar.Schema(fields={})
        
        result = schema.validate(df)
        
        assert result is not None

    def test_validate_with_max_errors_none(self):
        """validate() accepts max_errors=None (unlimited)."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        schema = ar.Schema(fields={})
        
        result = schema.validate(df, max_errors=None)
        assert result is not None

    def test_validate_with_max_errors_int(self):
        """validate() accepts max_errors as integer."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        schema = ar.Schema(fields={})
        
        result = schema.validate(df, max_errors=5)
        assert result is not None

    def test_validate_with_pandas_dataframe(self):
        """validate() works with pandas DataFrame."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
            "float_col": [1.1, 2.2, 3.3]
        })
        schema = ar.Schema(fields={})
        
        result = schema.validate(df)
        assert result is not None
