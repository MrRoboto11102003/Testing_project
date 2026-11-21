import pytest
import json
import io
import math
import numpy as np
import pandas as pd
import arnio as ar
from arnio.quality import ColumnProfile, DataQualityReport, CleaningSuggestion


class TestColumnProfileToDict:
    """Tests for ColumnProfile.to_dict()"""

    def test_numeric_dtype_includes_quantiles(self):
        """Numeric dtypes include quantile keys; non-numeric exclude them."""
        # Numeric: int64 should include q25, q50, etc.
        cp_num = ColumnProfile(
            name="age",
            dtype="int64",
            semantic_type="numeric",
            row_count=10,
            null_count=0,
            null_ratio=0.0,
            unique_count=10,
            unique_ratio=1.0,
            min=1,
            max=100,
            q25=25.0,
            q50=50.0,
            q75=75.0,
            q95=95.0,
            iqr=50.0,
        )
        d_num = cp_num.to_dict()
        assert "q25" in d_num
        assert "q50" in d_num
        assert d_num["q25"] == 25.0

        # Non-numeric: string should NOT include quantiles
        cp_str = ColumnProfile(
            name="name",
            dtype="string",
            semantic_type="text",
            row_count=5,
            null_count=0,
            null_ratio=0.0,
            unique_count=5,
            unique_ratio=1.0,
        )
        d_str = cp_str.to_dict()
        assert "q25" not in d_str
        assert "q50" not in d_str

    def test_redact_sample_values_keyword_only(self):
        """redact_sample_values must be keyword-only."""
        cp = ColumnProfile(
            name="x",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0,
        )
        # Positional call should fail
        with pytest.raises(TypeError):
            cp.to_dict(True)

    def test_redact_sample_values_must_be_bool(self):
        """redact_sample_values must be exactly bool, not 1, 0, or None."""
        cp = ColumnProfile(
            name="x",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0,
        )
        with pytest.raises(TypeError, match="must be a bool"):
            cp.to_dict(redact_sample_values=1)

        with pytest.raises(TypeError, match="must be a bool"):
            cp.to_dict(redact_sample_values=None)

    def test_redact_sample_values_true(self):
        """With redaction, sample_values become [REDACTED] but length preserved."""
        cp = ColumnProfile(
            name="text",
            dtype="string",
            semantic_type="text",
            row_count=3,
            null_count=0,
            null_ratio=0.0,
            unique_count=3,
            unique_ratio=1.0,
            sample_values=["secret1", "secret2", "secret3"],
        )
        d = cp.to_dict(redact_sample_values=True)
        assert d["sample_values"] == ["[REDACTED]", "[REDACTED]", "[REDACTED]"]
        assert len(d["sample_values"]) == 3

    def test_redact_top_values(self):
        """With redaction, top_values items have [REDACTED] value but counts/ratios kept."""
        cp = ColumnProfile(
            name="status",
            dtype="string",
            semantic_type="text",
            row_count=4,
            null_count=0,
            null_ratio=0.0,
            unique_count=2,
            unique_ratio=0.5,
            top_values=[("active", 3, 0.75), ("inactive", 1, 0.25)],
        )
        d = cp.to_dict(redact_sample_values=True)
        assert d["top_values"][0]["value"] == "[REDACTED]"
        assert d["top_values"][0]["count"] == 3
        assert d["top_values"][0]["ratio"] == 0.75

    def test_clean_scalar_nan_inf_to_none(self):
        """NaN, inf, and -inf are cleaned to None."""
        cp = ColumnProfile(
            name="value",
            dtype="float64",
            semantic_type="numeric",
            row_count=3,
            null_count=0,
            null_ratio=0.0,
            unique_count=3,
            unique_ratio=1.0,
            min=np.float64(np.nan),
            max=np.float64(np.inf),
            sample_values=[1.5, float("nan"), float("inf")],
        )
        d = cp.to_dict()
        assert d["min"] is None
        assert d["max"] is None
        assert d["sample_values"][1] is None
        assert d["sample_values"][2] is None

    def test_numpy_scalar_unwrapping(self):
        """Numpy scalars are unwrapped to Python scalars."""
        cp = ColumnProfile(
            name="count",
            dtype="int64",
            semantic_type="numeric",
            row_count=2,
            null_count=0,
            null_ratio=0.0,
            unique_count=2,
            unique_ratio=1.0,
            min=np.int64(10),
            max=np.int64(99),
            sample_values=[np.int64(42)],
        )
        d = cp.to_dict()
        assert type(d["min"]) is int
        assert type(d["max"]) is int
        assert type(d["sample_values"][0]) is int
        assert d["min"] == 10


class TestDataQualityReportScoreBreakdown:
    """Tests for DataQualityReport.score_breakdown()"""

    def test_score_breakdown_structure(self):
        """score_breakdown returns 4-key dict with penalties and final_score."""
        rep = DataQualityReport(
            row_count=10,
            column_count=1,
            memory_usage=100,
            duplicate_rows=2,
            duplicate_ratio=0.2,
            columns={},
            quality_score=70.0,
            score_components={
                "duplicate_penalty": -20.0,
                "null_penalty": -10.0,
            },
        )
        result = rep.score_breakdown()

        assert list(result.keys()) == [
            "null_penalty",
            "duplicate_penalty",
            "type_mismatch_penalty",
            "final_score",
        ]
        assert result["duplicate_penalty"] == -20.0
        assert result["null_penalty"] == -10.0
        assert result["type_mismatch_penalty"] == 0.0
        assert result["final_score"] == 70.0

    def test_score_breakdown_missing_penalties_default_zero(self):
        """Missing penalty keys default to 0.0."""
        rep = DataQualityReport(
            row_count=5,
            column_count=1,
            memory_usage=50,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
            quality_score=100.0,
            score_components={},
        )
        result = rep.score_breakdown()

        assert result["null_penalty"] == 0.0
        assert result["duplicate_penalty"] == 0.0
        assert result["type_mismatch_penalty"] == 0.0

    def test_score_breakdown_new_dict_each_call(self):
        """score_breakdown returns a new dict each time."""
        rep = DataQualityReport(
            row_count=1,
            column_count=1,
            memory_usage=10,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
            quality_score=100.0,
            score_components={"null_penalty": -5.0},
        )
        d1 = rep.score_breakdown()
        d2 = rep.score_breakdown()

        assert d1 == d2
        assert d1 is not d2


class TestDataQualityReportToDict:
    """Tests for DataQualityReport.to_dict()"""

    def test_to_dict_top_level_keys_and_order(self):
        """to_dict has 9 top-level keys in exact order."""
        rep = DataQualityReport(
            row_count=2,
            column_count=1,
            memory_usage=20,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
            quality_score=100.0,
        )
        d = rep.to_dict()

        expected_keys = [
            "row_count",
            "column_count",
            "memory_usage",
            "duplicate_rows",
            "duplicate_ratio",
            "quality_score",
            "score_components",
            "columns",
            "suggestions",
        ]
        assert list(d.keys()) == expected_keys

    def test_to_dict_columns_sorted_alphabetically(self):
        """Columns in output are sorted by name."""
        cols = {
            "zebra": ColumnProfile(
                name="zebra",
                dtype="string",
                semantic_type="text",
                row_count=1,
                null_count=0,
                null_ratio=0.0,
                unique_count=1,
                unique_ratio=1.0,
            ),
            "apple": ColumnProfile(
                name="apple",
                dtype="string",
                semantic_type="text",
                row_count=1,
                null_count=0,
                null_ratio=0.0,
                unique_count=1,
                unique_ratio=1.0,
            ),
        }
        rep = DataQualityReport(
            row_count=1,
            column_count=2,
            memory_usage=20,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns=cols,
        )
        d = rep.to_dict()

        assert list(d["columns"].keys()) == ["apple", "zebra"]

    def test_to_dict_exclude_columns(self):
        """exclude_columns removes specified columns from output."""
        col_a = ColumnProfile(
            name="a",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0,
        )
        col_b = ColumnProfile(
            name="b",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0,
        )
        rep = DataQualityReport(
            row_count=1,
            column_count=2,
            memory_usage=20,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={"a": col_a, "b": col_b},
        )
        d = rep.to_dict(exclude_columns=["a"])

        assert "b" in d["columns"]
        assert "a" not in d["columns"]
        assert d["column_count"] == 2  # Not changed

    def test_to_dict_exclude_columns_invalid_type(self):
        """exclude_columns must be None, list, tuple, or set."""
        rep = DataQualityReport(
            row_count=1,
            column_count=1,
            memory_usage=10,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
        )
        with pytest.raises(TypeError, match="must be a list, tuple, set, or None"):
            rep.to_dict(exclude_columns="a")

    def test_to_dict_exclude_columns_unknown(self):
        """exclude_columns with unknown column names raises KeyError."""
        rep = DataQualityReport(
            row_count=1,
            column_count=1,
            memory_usage=10,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={
                "x": ColumnProfile(
                    name="x",
                    dtype="string",
                    semantic_type="text",
                    row_count=1,
                    null_count=0,
                    null_ratio=0.0,
                    unique_count=1,
                    unique_ratio=1.0,
                )
            },
        )
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            rep.to_dict(exclude_columns=["y"])

    def test_to_dict_score_components_aliased(self):
        """score_components is the same object (not copied)."""
        sc = {"null_penalty": -5.0}
        rep = DataQualityReport(
            row_count=1,
            column_count=1,
            memory_usage=10,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
            score_components=sc,
        )
        d = rep.to_dict()

        assert d["score_components"] is sc

    def test_to_dict_redact_sample_values_in_columns(self):
        """redact_sample_values redacts samples in all column profiles."""
        col = ColumnProfile(
            name="text",
            dtype="string",
            semantic_type="text",
            row_count=2,
            null_count=0,
            null_ratio=0.0,
            unique_count=2,
            unique_ratio=1.0,
            sample_values=["password123", "secret456"],
        )
        rep = DataQualityReport(
            row_count=2,
            column_count=1,
            memory_usage=20,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={"text": col},
        )
        d = rep.to_dict(redact_sample_values=True)

        assert d["columns"]["text"]["sample_values"] == ["[REDACTED]", "[REDACTED]"]


class TestDataQualityReportToJson:
    """Tests for DataQualityReport.to_json()"""

    def test_to_json_returns_string(self):
        """With output=None, to_json returns a JSON string."""
        rep = DataQualityReport(
            row_count=1,
            column_count=0,
            memory_usage=10,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
        )
        result = rep.to_json()

        assert isinstance(result, str)
        assert result.startswith("{")
        # Should be parseable JSON
        d = json.loads(result)
        assert d["row_count"] == 1

    def test_to_json_writes_to_stream(self):
        """With output given, to_json writes to stream and returns None."""
        rep = DataQualityReport(
            row_count=5,
            column_count=0,
            memory_usage=50,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
        )
        stream = io.StringIO()
        result = rep.to_json(output=stream)

        assert result is None
        output_str = stream.getvalue()
        assert isinstance(output_str, str)
        d = json.loads(output_str)
        assert d["row_count"] == 5

    def test_to_json_output_must_have_write_method(self):
        """output parameter must have a write method."""
        rep = DataQualityReport(
            row_count=1,
            column_count=0,
            memory_usage=10,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
        )
        with pytest.raises(TypeError, match="must be a writable text stream"):
            rep.to_json(output=123)

    def test_to_json_indent_parameter(self):
        """indent parameter affects formatting."""
        rep = DataQualityReport(
            row_count=1,
            column_count=0,
            memory_usage=10,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
        )
        json_compact = rep.to_json(indent=None)
        json_pretty = rep.to_json(indent=2)

        # Compact should not have newlines
        assert "\n" not in json_compact
        # Pretty should have newlines
        assert "\n" in json_pretty
        # Both should parse to the same dict
        assert json.loads(json_compact) == json.loads(json_pretty)

    def test_to_json_exclude_columns_filters_suggestions(self):
        """exclude_columns filters column names from suggestion kwargs."""
        col_a = ColumnProfile(
            name="a",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0,
        )
        col_b = ColumnProfile(
            name="b",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0,
        )
        rep = DataQualityReport(
            row_count=1,
            column_count=2,
            memory_usage=20,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={"a": col_a, "b": col_b},
            suggestions=[
                CleaningSuggestion(
                    "strip_whitespace", {"subset": ["a", "b"]}, 0.95, "reason"
                )
            ],
        )
        result_json = rep.to_json(exclude_columns=["a"])
        d = json.loads(result_json)

        # Subset should only have "b"
        assert d["suggestions"][0]["kwargs"]["subset"] == ["b"]
        # Column "a" should not be in columns
        assert "a" not in d["columns"]
