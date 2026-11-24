import pytest
import json
import re
from arnio.quality import DataQualityReport, ColumnProfile, CleaningSuggestion


class TestDataQualityReportToDict:
    """Tests for DataQualityReport.to_dict() method."""
    
    def test_basic_to_dict_structure_and_key_order(self):
        """Test that to_dict returns dict with correct keys in exact order."""
        cp = ColumnProfile(
            name="col1", dtype="string", semantic_type="text", row_count=5,
            null_count=0, null_ratio=0.0, unique_count=5, unique_ratio=1.0,
            sample_values=["a", "b"]
        )
        rep = DataQualityReport(
            row_count=5, column_count=1, memory_usage=100, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=85.0, columns={"col1": cp},
            score_components={}, suggestions=[]
        )
        
        result = rep.to_dict()
        
        expected_keys = [
            "row_count", "column_count", "memory_usage", "duplicate_rows",
            "duplicate_ratio", "quality_score", "score_components", "columns", "suggestions"
        ]
        assert list(result.keys()) == expected_keys
    
    def test_score_components_same_object_not_copied(self):
        """Test that score_components in output is the same object (aliased)."""
        score_comp = {"penalty": -15.0, "bonus": 5.0}
        cp = ColumnProfile(
            name="x", dtype="int64", semantic_type="numeric", row_count=3,
            null_count=0, null_ratio=0.0, unique_count=3, unique_ratio=1.0,
            q25=1.0, q50=2.0, q75=3.0
        )
        rep = DataQualityReport(
            row_count=3, column_count=1, memory_usage=50, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=90.0, columns={"x": cp},
            score_components=score_comp, suggestions=[]
        )
        
        result = rep.to_dict()
        assert result["score_components"] is score_comp
    
    def test_columns_sorted_alphabetically(self):
        """Test that columns in output dict are sorted by name."""
        cp_b = ColumnProfile(
            name="b", dtype="string", semantic_type="text", row_count=2,
            null_count=0, null_ratio=0.0, unique_count=2, unique_ratio=1.0
        )
        cp_a = ColumnProfile(
            name="a", dtype="int64", semantic_type="numeric", row_count=2,
            null_count=0, null_ratio=0.0, unique_count=2, unique_ratio=1.0,
            q25=1.0, q50=2.0, q75=3.0
        )
        rep = DataQualityReport(
            row_count=2, column_count=2, memory_usage=100, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=80.0, columns={"b": cp_b, "a": cp_a},
            score_components={}, suggestions=[]
        )
        
        result = rep.to_dict()
        assert list(result["columns"].keys()) == ["a", "b"]
    
    def test_exclude_columns_filters_from_output(self):
        """Test that exclude_columns removes specified columns."""
        cp_a = ColumnProfile(
            name="a", dtype="string", semantic_type="text", row_count=2,
            null_count=0, null_ratio=0.0, unique_count=2, unique_ratio=1.0
        )
        cp_b = ColumnProfile(
            name="b", dtype="int64", semantic_type="numeric", row_count=2,
            null_count=0, null_ratio=0.0, unique_count=2, unique_ratio=1.0,
            q25=1.0, q50=2.0, q75=3.0
        )
        rep = DataQualityReport(
            row_count=2, column_count=2, memory_usage=100, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=85.0, columns={"a": cp_a, "b": cp_b},
            score_components={}, suggestions=[]
        )
        
        result = rep.to_dict(exclude_columns=["a"])
        assert list(result["columns"].keys()) == ["b"]
        assert result["column_count"] == 2  # Not recomputed
    
    def test_exclude_columns_filters_suggestion_kwargs(self):
        """Test that excluded columns are removed from suggestion kwargs."""
        cp_a = ColumnProfile(
            name="a", dtype="int64", semantic_type="numeric", row_count=2,
            null_count=0, null_ratio=0.0, unique_count=2, unique_ratio=1.0,
            q25=1.0, q50=2.0, q75=3.0
        )
        cp_b = ColumnProfile(
            name="b", dtype="string", semantic_type="text", row_count=2,
            null_count=0, null_ratio=0.0, unique_count=2, unique_ratio=1.0
        )
        
        sugg = CleaningSuggestion(
            "cast_types", {"a": "int64", "b": "string"}, 0.9, "Cast"
        )
        
        rep = DataQualityReport(
            row_count=2, column_count=2, memory_usage=100, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=80.0, columns={"a": cp_a, "b": cp_b},
            score_components={}, suggestions=[sugg]
        )
        
        result = rep.to_dict(exclude_columns=["a"])
        assert result["suggestions"][0]["kwargs"] == {"b": "string"}
    
    def test_redact_sample_values_true(self):
        """Test that redact_sample_values=True masks sample values."""
        cp = ColumnProfile(
            name="col", dtype="string", semantic_type="text", row_count=2,
            null_count=0, null_ratio=0.0, unique_count=2, unique_ratio=1.0,
            sample_values=["secret1", "secret2"]
        )
        rep = DataQualityReport(
            row_count=2, column_count=1, memory_usage=50, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=90.0, columns={"col": cp},
            score_components={}, suggestions=[]
        )
        
        result = rep.to_dict(redact_sample_values=True)
        assert result["columns"]["col"]["sample_values"] == ["[REDACTED]", "[REDACTED]"]
    
    def test_redact_sample_values_false(self):
        """Test that redact_sample_values=False preserves sample values."""
        cp = ColumnProfile(
            name="col", dtype="string", semantic_type="text", row_count=2,
            null_count=0, null_ratio=0.0, unique_count=2, unique_ratio=1.0,
            sample_values=["public1", "public2"]
        )
        rep = DataQualityReport(
            row_count=2, column_count=1, memory_usage=50, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=90.0, columns={"col": cp},
            score_components={}, suggestions=[]
        )
        
        result = rep.to_dict(redact_sample_values=False)
        assert result["columns"]["col"]["sample_values"] == ["public1", "public2"]
    
    def test_suggestions_sorted_by_step_then_kwargs(self):
        """Test that suggestions are sorted by step name."""
        cp = ColumnProfile(
            name="col", dtype="string", semantic_type="text", row_count=1,
            null_count=0, null_ratio=0.0, unique_count=1, unique_ratio=1.0
        )
        
        sugg1 = CleaningSuggestion("strip_whitespace", {"subset": ["col"]}, 0.8, "S")
        sugg2 = CleaningSuggestion("cast_types", {"col": "string"}, 0.9, "C")
        
        rep = DataQualityReport(
            row_count=1, column_count=1, memory_usage=50, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=85.0, columns={"col": cp},
            score_components={}, suggestions=[sugg1, sugg2]
        )
        
        result = rep.to_dict()
        steps = [s["step"] for s in result["suggestions"]]
        assert steps == ["cast_types", "strip_whitespace"]
    
    def test_plain_tuple_suggestion_confidence_fields_none(self):
        """Test that plain 2-tuple suggestions have None confidence fields."""
        cp = ColumnProfile(
            name="col", dtype="string", semantic_type="text", row_count=1,
            null_count=0, null_ratio=0.0, unique_count=1, unique_ratio=1.0
        )
        
        sugg = ("drop_duplicates", {"keep": "first"})
        
        rep = DataQualityReport(
            row_count=1, column_count=1, memory_usage=50, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=80.0, columns={"col": cp},
            score_components={}, suggestions=[sugg]
        )
        
        result = rep.to_dict()
        assert result["suggestions"][0]["confidence_score"] is None
        assert result["suggestions"][0]["confidence_reason"] is None
    
    def test_redact_column_name_in_suggestion_reason(self):
        """Test that column names in suggestion reasons are redacted."""
        cp_a = ColumnProfile(
            name="a", dtype="int64", semantic_type="numeric", row_count=1,
            null_count=0, null_ratio=0.0, unique_count=1, unique_ratio=1.0,
            q25=1.0, q50=2.0, q75=3.0
        )
        
        sugg = CleaningSuggestion(
            "cast_types", {"a": "int64"}, 0.95,
            "Column 'a' conforms perfectly to int64 structure."
        )
        
        rep = DataQualityReport(
            row_count=1, column_count=1, memory_usage=50, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=90.0, columns={"a": cp_a},
            score_components={}, suggestions=[sugg]
        )
        
        result = rep.to_dict(exclude_columns=["a"])
        reason = result["suggestions"][0]["confidence_reason"]
        assert "[REDACTED]" in reason
        assert "'a'" not in reason
    
    def test_exclude_columns_type_error_when_string(self):
        """Test that exclude_columns as string raises TypeError."""
        cp = ColumnProfile(
            name="col", dtype="string", semantic_type="text", row_count=1,
            null_count=0, null_ratio=0.0, unique_count=1, unique_ratio=1.0
        )
        rep = DataQualityReport(
            row_count=1, column_count=1, memory_usage=10, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=80.0, columns={"col": cp},
            score_components={}, suggestions=[]
        )
        
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            rep.to_dict(exclude_columns="col")
    
    def test_exclude_columns_type_error_non_string_elements(self):
        """Test that non-string elements in exclude_columns raises TypeError."""
        cp = ColumnProfile(
            name="col", dtype="string", semantic_type="text", row_count=1,
            null_count=0, null_ratio=0.0, unique_count=1, unique_ratio=1.0
        )
        rep = DataQualityReport(
            row_count=1, column_count=1, memory_usage=10, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=80.0, columns={"col": cp},
            score_components={}, suggestions=[]
        )
        
        with pytest.raises(TypeError, match="exclude_columns must contain only string column names"):
            rep.to_dict(exclude_columns=["col", 1])
    
    def test_exclude_columns_key_error_unknown_column(self):
        """Test that unknown exclude_columns raises KeyError."""
        cp = ColumnProfile(
            name="col", dtype="string", semantic_type="text", row_count=1,
            null_count=0, null_ratio=0.0, unique_count=1, unique_ratio=1.0
        )
        rep = DataQualityReport(
            row_count=1, column_count=1, memory_usage=10, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=80.0, columns={"col": cp},
            score_components={}, suggestions=[]
        )
        
        with pytest.raises(KeyError, match=re.escape("Unknown exclude_columns: ['unknown']")):
            rep.to_dict(exclude_columns=["unknown"])
    
    def test_redact_sample_values_type_error_not_bool(self):
        """Test that redact_sample_values=1 (int) raises TypeError."""
        cp = ColumnProfile(
            name="col", dtype="string", semantic_type="text", row_count=1,
            null_count=0, null_ratio=0.0, unique_count=1, unique_ratio=1.0
        )
        rep = DataQualityReport(
            row_count=1, column_count=1, memory_usage=10, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=80.0, columns={"col": cp},
            score_components={}, suggestions=[]
        )
        
        with pytest.raises(TypeError, match="redact_sample_values must be a bool"):
            rep.to_dict(redact_sample_values=1)
    
    def test_empty_report_to_dict(self):
        """Test to_dict with no columns."""
        rep = DataQualityReport(
            row_count=0, column_count=0, memory_usage=0, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=0.0, columns={},
            score_components={}, suggestions=[]
        )
        
        result = rep.to_dict()
        assert result["columns"] == {}
        assert result["column_count"] == 0
    
    def test_json_serializable_output(self):
        """Test that to_dict output is JSON-serializable."""
        cp = ColumnProfile(
            name="col", dtype="int64", semantic_type="numeric", row_count=3,
            null_count=0, null_ratio=0.0, unique_count=3, unique_ratio=1.0,
            q25=1.0, q50=2.0, q75=3.0
        )
        rep = DataQualityReport(
            row_count=3, column_count=1, memory_usage=50, duplicate_rows=0,
            duplicate_ratio=0.0, quality_score=90.0, columns={"col": cp},
            score_components={"penalty": -10.0}, suggestions=[]
        )
        
        result = rep.to_dict()
        # Should not raise
        json_str = json.dumps(result, default=str)
        assert json_str is not None
