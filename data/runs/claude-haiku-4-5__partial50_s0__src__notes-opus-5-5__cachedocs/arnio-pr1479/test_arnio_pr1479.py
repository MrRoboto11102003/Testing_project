import pytest
import re
import json
import numpy as np
import pandas as pd
import arnio as ar
from arnio.quality import DataQualityReport, ColumnProfile, CleaningSuggestion


def _make_column_profile(name, dtype="string", **kwargs):
    """Helper to create a ColumnProfile with minimal defaults."""
    defaults = {
        "name": name,
        "dtype": dtype,
        "semantic_type": "text" if dtype == "string" else "numeric",
        "row_count": 10,
        "null_count": 0,
        "null_ratio": 0.0,
        "unique_count": 5,
        "unique_ratio": 0.5,
        "empty_string_count": 0,
        "whitespace_count": 0,
        "sample_values": ["s1", "s2"],
    }
    if dtype in ("int64", "float64"):
        defaults.update({"q25": 1.0, "q50": 2.0, "q75": 3.0, "q95": 4.0})
    defaults.update(kwargs)
    return ColumnProfile(**defaults)


class TestToDict:
    """Test suite for DataQualityReport.to_dict()"""

    def test_basic_to_dict(self):
        """Test basic to_dict with minimal report."""
        rep = DataQualityReport(
            row_count=100, column_count=2, memory_usage=1024,
            duplicate_rows=5, duplicate_ratio=0.05, quality_score=85.0,
            score_components={"dup": -15.0},
            columns={"a": _make_column_profile("a"), "b": _make_column_profile("b")},
            suggestions=[]
        )
        result = rep.to_dict()
        
        assert result["row_count"] == 100
        assert result["column_count"] == 2
        assert result["duplicate_ratio"] == 0.05
        assert isinstance(result["columns"], dict)
        assert isinstance(result["suggestions"], list)

    def test_to_dict_key_order(self):
        """Test that returned dict has keys in correct order."""
        rep = DataQualityReport(
            row_count=1, column_count=0, memory_usage=0,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={}, columns={}, suggestions=[]
        )
        keys = list(rep.to_dict().keys())
        expected = [
            "row_count", "column_count", "memory_usage", "duplicate_rows",
            "duplicate_ratio", "quality_score", "score_components", "columns", "suggestions"
        ]
        assert keys == expected

    def test_redact_sample_values_type_validation(self):
        """Test that redact_sample_values must be a bool, not int/str/None."""
        rep = DataQualityReport(
            row_count=1, column_count=1, memory_usage=0,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={},
            columns={"a": _make_column_profile("a")},
            suggestions=[]
        )
        
        # These should raise TypeError
        with pytest.raises(TypeError, match="redact_sample_values must be a bool"):
            rep.to_dict(redact_sample_values=1)
        
        with pytest.raises(TypeError, match="redact_sample_values must be a bool"):
            rep.to_dict(redact_sample_values="true")
        
        with pytest.raises(TypeError, match="redact_sample_values must be a bool"):
            rep.to_dict(redact_sample_values=None)

    def test_exclude_columns_type_validation(self):
        """Test that exclude_columns accepts only list/tuple/set/None."""
        rep = DataQualityReport(
            row_count=1, column_count=1, memory_usage=0,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={},
            columns={"a": _make_column_profile("a")},
            suggestions=[]
        )
        
        # str is not allowed
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            rep.to_dict(exclude_columns="a")
        
        # dict is not allowed
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            rep.to_dict(exclude_columns={"a": 1})

    def test_exclude_columns_element_type_validation(self):
        """Test that all elements in exclude_columns must be strings."""
        rep = DataQualityReport(
            row_count=1, column_count=1, memory_usage=0,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={},
            columns={"a": _make_column_profile("a")},
            suggestions=[]
        )
        
        with pytest.raises(TypeError, match="exclude_columns must contain only string column names"):
            rep.to_dict(exclude_columns=["a", 1])
        
        with pytest.raises(TypeError, match="exclude_columns must contain only string column names"):
            rep.to_dict(exclude_columns=[None])

    def test_exclude_columns_unknown_columns(self):
        """Test that unknown column names in exclude_columns raise KeyError."""
        rep = DataQualityReport(
            row_count=1, column_count=2, memory_usage=0,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={},
            columns={"b": _make_column_profile("b"), "a": _make_column_profile("a")},
            suggestions=[]
        )
        
        # Single unknown column
        with pytest.raises(KeyError, match=re.escape("Unknown exclude_columns: ['z']")):
            rep.to_dict(exclude_columns=["z"])
        
        # Multiple unknown columns should be sorted in error
        with pytest.raises(KeyError, match=re.escape("['x', 'z']")):
            rep.to_dict(exclude_columns=["z", "x"])
        
        # Available columns should be in insertion order
        with pytest.raises(KeyError, match=re.escape("Available columns: b, a")):
            rep.to_dict(exclude_columns=["unknown"])

    def test_exclude_columns_empty_report(self):
        """Test error message when report is empty."""
        rep = DataQualityReport(
            row_count=0, column_count=0, memory_usage=0,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={}, columns={}, suggestions=[]
        )
        
        with pytest.raises(KeyError, match=re.escape("Available columns: <none>")):
            rep.to_dict(exclude_columns=["a"])

    def test_columns_sorted_and_excluded(self):
        """Test that columns are sorted alphabetically and exclusions work."""
        rep = DataQualityReport(
            row_count=10, column_count=3, memory_usage=100,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={},
            columns={
                "z": _make_column_profile("z"),
                "a": _make_column_profile("a"),
                "m": _make_column_profile("m"),
            },
            suggestions=[]
        )
        
        # Without exclusion: columns sorted
        result = rep.to_dict()
        assert list(result["columns"].keys()) == ["a", "m", "z"]
        
        # With exclusion: sorted and filtered
        result = rep.to_dict(exclude_columns=["m"])
        assert list(result["columns"].keys()) == ["a", "z"]
        
        # column_count not recomputed
        assert result["column_count"] == 3

    def test_redact_sample_values_true(self):
        """Test that sample_values are redacted when flag is True."""
        cp = _make_column_profile("a", sample_values=["val1", "val2", "val3"])
        rep = DataQualityReport(
            row_count=10, column_count=1, memory_usage=100,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={},
            columns={"a": cp},
            suggestions=[]
        )
        
        # Without redaction
        result = rep.to_dict(redact_sample_values=False)
        assert result["columns"]["a"]["sample_values"] == ["val1", "val2", "val3"]
        
        # With redaction: same length, all "[REDACTED]"
        result = rep.to_dict(redact_sample_values=True)
        assert result["columns"]["a"]["sample_values"] == ["[REDACTED]", "[REDACTED]", "[REDACTED]"]

    def test_numeric_columns_have_quantiles(self):
        """Test that numeric columns have q25/q50/etc but string columns don't."""
        int_col = _make_column_profile("int_col", dtype="int64", q25=1.0, q50=2.0, q75=3.0, q95=4.0)
        str_col = _make_column_profile("str_col", dtype="string")
        
        rep = DataQualityReport(
            row_count=10, column_count=2, memory_usage=100,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={},
            columns={"int_col": int_col, "str_col": str_col},
            suggestions=[]
        )
        
        result = rep.to_dict()
        assert "q25" in result["columns"]["int_col"]
        assert "q50" in result["columns"]["int_col"]
        assert "q25" not in result["columns"]["str_col"]

    def test_score_components_object_identity(self):
        """Test that score_components is the same object (not a copy)."""
        score_comp = {"dup_penalty": -15.0, "null_penalty": -5.0}
        rep = DataQualityReport(
            row_count=10, column_count=1, memory_usage=100,
            duplicate_rows=5, duplicate_ratio=0.05, quality_score=80.0,
            score_components=score_comp,
            columns={"a": _make_column_profile("a")},
            suggestions=[]
        )
        
        result = rep.to_dict()
        assert result["score_components"] is score_comp

    def test_suggestions_sorted_by_step_and_kwargs(self):
        """Test that suggestions are sorted by step name, then by JSON kwargs."""
        sugg1 = CleaningSuggestion("strip_whitespace", {"subset": ["b"]}, 0.95, "r1")
        sugg2 = CleaningSuggestion("drop_duplicates", {"keep": "first"}, 0.90, "r2")
        sugg3 = CleaningSuggestion("cast_types", {"a": "int64"}, 0.85, "r3")
        
        rep = DataQualityReport(
            row_count=10, column_count=2, memory_usage=100,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={},
            columns={"a": _make_column_profile("a"), "b": _make_column_profile("b")},
            suggestions=[sugg1, sugg2, sugg3]  # Unsorted input
        )
        
        result = rep.to_dict()
        steps = [s["step"] for s in result["suggestions"]]
        # cast_types < drop_duplicates < strip_whitespace (alphabetical)
        assert steps == ["cast_types", "drop_duplicates", "strip_whitespace"]

    def test_suggestion_subset_filtering(self):
        """Test that excluded columns are filtered from suggestion kwargs."""
        sugg = CleaningSuggestion("strip_whitespace", {"subset": ["a", "b", "c"]}, 0.95, "reason")
        rep = DataQualityReport(
            row_count=10, column_count=3, memory_usage=100,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={},
            columns={
                "a": _make_column_profile("a"),
                "b": _make_column_profile("b"),
                "c": _make_column_profile("c"),
            },
            suggestions=[sugg]
        )
        
        result = rep.to_dict(exclude_columns=["b"])
        suggestion_dict = result["suggestions"][0]
        assert suggestion_dict["kwargs"]["subset"] == ["a", "c"]

    def test_suggestion_reason_redaction(self):
        """Test that quoted column names in reasons are redacted."""
        reason = "Column 'a' conforms to int64 and column 'b' needs cleaning."
        sugg = CleaningSuggestion("strip_whitespace", {"subset": ["a", "b"]}, 0.95, reason)
        
        rep = DataQualityReport(
            row_count=10, column_count=2, memory_usage=100,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={},
            columns={"a": _make_column_profile("a"), "b": _make_column_profile("b")},
            suggestions=[sugg]
        )
        
        result = rep.to_dict(exclude_columns=["a"])
        suggestion_dict = result["suggestions"][0]
        expected = "Column '[REDACTED]' conforms to int64 and column 'b' needs cleaning."
        assert suggestion_dict["confidence_reason"] == expected

    def test_plain_tuple_suggestion_confidence_fields_none(self):
        """Test that plain 2-tuple suggestions have None for confidence fields."""
        # Plain tuple (not CleaningSuggestion object)
        plain_sugg = ("drop_duplicates", {"keep": "first"})
        
        rep = DataQualityReport(
            row_count=10, column_count=1, memory_usage=100,
            duplicate_rows=5, duplicate_ratio=0.05, quality_score=90.0,
            score_components={},
            columns={"a": _make_column_profile("a")},
            suggestions=[plain_sugg]
        )
        
        result = rep.to_dict()
        sugg = result["suggestions"][0]
        assert sugg["step"] == "drop_duplicates"
        assert sugg["confidence_score"] is None
        assert sugg["confidence_reason"] is None

    def test_exclude_columns_duplicates_handled(self):
        """Test that duplicate values in exclude_columns don't cause issues."""
        rep = DataQualityReport(
            row_count=10, column_count=2, memory_usage=100,
            duplicate_rows=0, duplicate_ratio=0.0, quality_score=100.0,
            score_components={},
            columns={"a": _make_column_profile("a"), "b": _make_column_profile("b")},
            suggestions=[]
        )
        
        # Duplicate exclusions
        result = rep.to_dict(exclude_columns=["a", "a", "a"])
        assert list(result["columns"].keys()) == ["b"]
