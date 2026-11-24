import pytest
import pandas as pd
import json
import re
import warnings
import arnio as ar
from arnio.schema import Int64, String, Float64


class TestBootstrapFromReport:
    """Test Schema.bootstrap_from_report"""
    
    def test_bootstrap_valid_report(self):
        """Test bootstrapping from a valid report with multiple columns"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", None],
        })
        report = ar.profile(ar.from_pandas(df))
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert "id" in schema.fields
        assert "name" in schema.fields
        assert schema.fields["name"].nullable is True
        assert schema.fields["id"].nullable is False
    
    def test_bootstrap_type_error(self):
        """Test TypeError when input is not a DataQualityReport"""
        with pytest.raises(TypeError, match="Expected DataQualityReport, got dict"):
            ar.Schema.bootstrap_from_report({})
    
    def test_bootstrap_empty_report(self):
        """Test ValueError when report has no columns"""
        empty_df = pd.DataFrame()
        report = ar.profile(ar.from_pandas(empty_df))
        with pytest.raises(ValueError, match="Cannot bootstrap schema from an empty report"):
            ar.Schema.bootstrap_from_report(report)


class TestFromJson:
    """Test Schema.from_json"""
    
    def test_from_json_valid(self):
        """Test valid JSON deserialization"""
        json_str = '{"fields": {"a": {"dtype": "int64", "nullable": false}}, "strict": false, "unique": null}'
        schema = ar.Schema.from_json(json_str)
        assert schema.fields["a"].dtype == "int64"
        assert schema.fields["a"].nullable is False
    
    def test_from_json_invalid_json(self):
        """Test ValueError for invalid JSON"""
        with pytest.raises(ValueError, match="Invalid schema JSON"):
            ar.Schema.from_json("not valid json {")
    
    def test_from_json_unknown_keys(self):
        """Test ValueError for unknown keys"""
        with pytest.raises(ValueError, match="unknown key"):
            ar.Schema.from_json('{"fields": {}, "unknown_key": 1}')


class TestToJson:
    """Test Schema.to_json"""
    
    def test_to_json_basic(self):
        """Test basic JSON serialization"""
        schema = ar.Schema({"a": Int64(), "b": String()})
        json_str = schema.to_json()
        payload = json.loads(json_str)
        assert "fields" in payload
        assert payload["strict"] is False
    
    def test_to_json_field_order(self):
        """Test that fields are sorted in JSON output"""
        schema = ar.Schema({"z": String(), "a": Int64()})
        payload = json.loads(schema.to_json())
        assert list(payload["fields"].keys()) == ["a", "z"]
    
    def test_to_json_with_rules_warning(self):
        """Test warning when rules are present"""
        schema = ar.Schema({"a": Int64()}, rules=[lambda df: []])
        with pytest.warns(UserWarning, match="cross-field rules cannot be serialized"):
            schema.to_json()


class TestRoundTrip:
    """Test round-trip serialization"""
    
    def test_round_trip(self):
        """Test round-trip from to_json and from_json"""
        original = ar.Schema({
            "id": Int64(nullable=False, min=1, max=100),
            "name": String()
        }, strict=True)
        restored = ar.Schema.from_json(original.to_json())
        assert restored.fields["id"].dtype == "int64"
        assert restored.fields["id"].min == 1
        assert restored.strict is True


class TestValidate:
    """Test Schema.validate"""
    
    def test_validate_pass(self):
        """Test basic validation that passes"""
        df = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.Schema({"a": Int64()}).validate(df)
        assert result.passed is True
    
    def test_validate_not_arframe(self):
        """Test TypeError when frame is not an ArFrame"""
        with pytest.raises(TypeError, match="validate\\(\\) expects a single ArFrame"):
            ar.Schema({"a": Int64()}).validate(pd.DataFrame({"a": [1]}))
    
    def test_validate_dtype_mismatch(self):
        """Test detection of dtype mismatch"""
        df = ar.from_pandas(pd.DataFrame({"a": ["x", "y"]}))
        result = ar.Schema({"a": Int64()}).validate(df)
        assert not result.passed
        assert any(issue.rule == "dtype" for issue in result.issues)
    
    def test_validate_nullable(self):
        """Test detection of null values"""
        df = ar.from_pandas(pd.DataFrame({"a": [1.0, None, 3.0]}))
        result = ar.Schema({"a": Int64(nullable=False)}).validate(df)
        assert not result.passed
        assert any(issue.rule == "nullable" for issue in result.issues)
    
    def test_validate_strict_mode(self):
        """Test strict mode detects unexpected columns"""
        df = ar.from_pandas(pd.DataFrame({"a": [1], "b": ["x"]}))
        result = ar.Schema({"a": Int64()}, strict=True).validate(df)
        assert not result.passed
        assert any(issue.rule == "unexpected_column" for issue in result.issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
