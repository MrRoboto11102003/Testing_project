import json
import io
import math
import numpy as np
import pytest
from arnio.quality import ColumnProfile, DataQualityReport, CleaningSuggestion


class TestColumnProfileToDict:
    """Test ColumnProfile.to_dict()"""
    
    def test_string_dtype_no_quantiles(self):
        """String column should not include quantile keys."""
        cp = ColumnProfile(
            name="text_col",
            dtype="string",
            semantic_type="text",
            row_count=2,
            null_count=0,
            null_ratio=0.0,
            unique_count=2,
            unique_ratio=1.0
        )
        d = cp.to_dict()
        assert "q25" not in d
        assert "q50" not in d
        assert "outlier_count" not in d
        
    def test_int64_dtype_includes_quantiles(self):
        """int64 column should include quantile keys."""
        cp = ColumnProfile(
            name="num_col",
            dtype="int64",
            semantic_type="numeric",
            row_count=4,
            null_count=0,
            null_ratio=0.0,
            unique_count=4,
            unique_ratio=1.0,
            min=np.int64(1),
            max=np.int64(4),
            q25=1.0,
            q50=2.0,
            q75=3.0,
            q95=3.95
        )
        d = cp.to_dict()
        assert "q25" in d
        assert d["q25"] == 1.0
        assert "outlier_lower_bound" in d
        
    def test_redact_sample_values_true(self):
        """With redact=True, sample_values become [REDACTED] strings."""
        cp = ColumnProfile(
            name="col",
            dtype="string",
            semantic_type="text",
            row_count=3,
            null_count=0,
            null_ratio=0.0,
            unique_count=2,
            unique_ratio=0.667,
            sample_values=["secret1", "secret2", "secret3"]
        )
        d = cp.to_dict(redact_sample_values=True)
        assert d["sample_values"] == ["[REDACTED]", "[REDACTED]", "[REDACTED]"]
        
    def test_redact_top_values(self):
        """With redact=True, top_values items get value=[REDACTED]."""
        cp = ColumnProfile(
            name="col",
            dtype="string",
            semantic_type="text",
            row_count=2,
            null_count=0,
            null_ratio=0.0,
            unique_count=2,
            unique_ratio=1.0,
            top_values=[("secret", 2, 1.0)]
        )
        d_redacted = cp.to_dict(redact_sample_values=True)
        d_plain = cp.to_dict(redact_sample_values=False)
        assert d_redacted["top_values"][0]["value"] == "[REDACTED]"
        assert d_plain["top_values"][0]["value"] == "secret"
        assert d_redacted["top_values"][0]["count"] == d_plain["top_values"][0]["count"]
        
    def test_redact_sample_values_invalid_type(self):
        """Non-bool redact_sample_values raises TypeError."""
        cp = ColumnProfile(
            name="col",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0
        )
        with pytest.raises(TypeError, match="redact_sample_values must be a bool"):
            cp.to_dict(redact_sample_values=1)
        with pytest.raises(TypeError, match="redact_sample_values must be a bool"):
            cp.to_dict(redact_sample_values=None)
            
    def test_numpy_scalar_unwrapping(self):
        """Numpy scalars are unwrapped via _clean_scalar."""
        cp = ColumnProfile(
            name="col",
            dtype="int64",
            semantic_type="numeric",
            row_count=2,
            null_count=0,
            null_ratio=0.0,
            unique_count=2,
            unique_ratio=1.0,
            min=np.int64(5),
            max=np.float64(10.0),
            sample_values=[np.int64(7), np.float64(8.5)]
        )
        d = cp.to_dict()
        assert type(d["min"]) is int and d["min"] == 5
        assert type(d["max"]) is float and d["max"] == 10.0
        assert type(d["sample_values"][0]) is int
        assert type(d["sample_values"][1]) is float
        
    def test_nan_becomes_none(self):
        """NaN values become None via _clean_scalar."""
        cp = ColumnProfile(
            name="col",
            dtype="float64",
            semantic_type="numeric",
            row_count=2,
            null_count=0,
            null_ratio=0.0,
            unique_count=2,
            unique_ratio=1.0,
            min=float("nan"),
            max=1.0,
            sample_values=[float("nan"), 2.0]
        )
        d = cp.to_dict()
        assert d["min"] is None
        assert d["sample_values"][0] is None
        
    def test_inf_becomes_none(self):
        """Inf values become None via _clean_scalar."""
        cp = ColumnProfile(
            name="col",
            dtype="float64",
            semantic_type="numeric",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0,
            max=float("inf"),
            sample_values=[float("-inf")]
        )
        d = cp.to_dict()
        assert d["max"] is None
        assert d["sample_values"][0] is None
        
    def test_mean_std_not_cleaned(self):
        """mean and std are NOT cleaned; NaN stays as NaN."""
        cp = ColumnProfile(
            name="col",
            dtype="float64",
            semantic_type="numeric",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0,
            mean=float("nan"),
            std=float("inf")
        )
        d = cp.to_dict()
        assert math.isnan(d["mean"])
        assert math.isinf(d["std"])
        
    def test_histogram_formatting(self):
        """Histogram tuples are converted to dicts with clean scalars."""
        cp = ColumnProfile(
            name="col",
            dtype="float64",
            semantic_type="numeric",
            row_count=3,
            null_count=0,
            null_ratio=0.0,
            unique_count=3,
            unique_ratio=1.0,
            histogram=[(0.0, 1.0, 2, 0.667), (np.float64(np.inf), 2.0, 0, 0.0)]
        )
        d = cp.to_dict()
        assert len(d["histogram"]) == 2
        assert d["histogram"][0]["bucket_start"] == 0.0
        assert d["histogram"][0]["bucket_end"] == 1.0
        assert d["histogram"][1]["bucket_start"] is None  # inf cleaned
        
    def test_top_values_none_stays_none(self):
        """When top_values is None, it stays None."""
        cp = ColumnProfile(
            name="col",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0,
            top_values=None
        )
        d = cp.to_dict()
        assert d["top_values"] is None
        
    def test_warnings_is_copy(self):
        """Warnings list is copied, not aliased."""
        cp = ColumnProfile(
            name="col",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0,
            warnings=["warn1", "warn2"]
        )
        d = cp.to_dict()
        assert d["warnings"] == ["warn1", "warn2"]
        assert d["warnings"] is not cp.warnings


class TestDataQualityReportScoreBreakdown:
    """Test DataQualityReport.score_breakdown()"""
    
    def test_default_report_all_zeros(self):
        """Default report returns all 0.0 penalties and score 100.0."""
        rep = DataQualityReport(
            row_count=0,
            column_count=0,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={}
        )
        breakdown = rep.score_breakdown()
        assert breakdown == {
            "null_penalty": 0.0,
            "duplicate_penalty": 0.0,
            "type_mismatch_penalty": 0.0,
            "final_score": 100.0
        }
        
    def test_with_actual_penalties(self):
        """score_breakdown returns stored penalties and final score."""
        rep = DataQualityReport(
            row_count=10,
            column_count=1,
            memory_usage=100,
            duplicate_rows=1,
            duplicate_ratio=0.1,
            columns={},
            quality_score=72.5,
            score_components={
                "duplicate_penalty": -10.0,
                "null_penalty": -17.5
            }
        )
        breakdown = rep.score_breakdown()
        assert breakdown["null_penalty"] == -17.5
        assert breakdown["duplicate_penalty"] == -10.0
        assert breakdown["type_mismatch_penalty"] == 0.0
        assert breakdown["final_score"] == 72.5
        
    def test_extra_component_keys_ignored(self):
        """Extra keys in score_components don't appear in breakdown."""
        rep = DataQualityReport(
            row_count=1,
            column_count=1,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
            score_components={"custom_key": -99.0, "null_penalty": -5.0}
        )
        breakdown = rep.score_breakdown()
        assert len(breakdown) == 4
        assert "custom_key" not in breakdown
        
    def test_key_order(self):
        """Keys appear in exact order."""
        rep = DataQualityReport(
            row_count=0,
            column_count=0,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={}
        )
        keys = list(rep.score_breakdown().keys())
        assert keys == ["null_penalty", "duplicate_penalty", "type_mismatch_penalty", "final_score"]
        
    def test_no_recomputation(self):
        """final_score is stored quality_score, not computed from components."""
        rep = DataQualityReport(
            row_count=0,
            column_count=0,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
            quality_score=55.0,
            score_components={}  # Empty, so sum would be 100
        )
        breakdown = rep.score_breakdown()
        assert breakdown["final_score"] == 55.0  # Not 100


class TestDataQualityReportToDict:
    """Test DataQualityReport.to_dict()"""
    
    def test_minimal_report_structure(self):
        """Minimal report has correct top-level structure."""
        rep = DataQualityReport(
            row_count=5,
            column_count=1,
            memory_usage=50,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={}
        )
        d = rep.to_dict()
        assert d["row_count"] == 5
        assert d["column_count"] == 1
        assert d["memory_usage"] == 50
        assert isinstance(d["columns"], dict)
        assert isinstance(d["suggestions"], list)
        
    def test_redact_sample_values_invalid(self):
        """Non-bool redact_sample_values raises TypeError."""
        rep = DataQualityReport(
            row_count=0,
            column_count=0,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={}
        )
        with pytest.raises(TypeError, match="redact_sample_values must be a bool"):
            rep.to_dict(redact_sample_values=1)
            
    def test_exclude_columns_not_list_tuple_set(self):
        """exclude_columns must be list/tuple/set, not string or dict."""
        rep = DataQualityReport(
            row_count=0,
            column_count=0,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={}
        )
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            rep.to_dict(exclude_columns="col")
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            rep.to_dict(exclude_columns={"col"})
            
    def test_exclude_columns_non_string_elements(self):
        """exclude_columns elements must be strings."""
        rep = DataQualityReport(
            row_count=0,
            column_count=0,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={}
        )
        with pytest.raises(TypeError, match="exclude_columns must contain only string column names"):
            rep.to_dict(exclude_columns=["a", 1, "b"])
            
    def test_exclude_columns_unknown_name(self):
        """Unknown column name in exclude_columns raises KeyError."""
        cp = ColumnProfile(
            name="a",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0
        )
        rep = DataQualityReport(
            row_count=1,
            column_count=1,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={"a": cp}
        )
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            rep.to_dict(exclude_columns=["unknown"])
            
    def test_columns_are_sorted(self):
        """Columns in output are sorted by name."""
        cp_a = ColumnProfile(
            name="a",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0
        )
        cp_z = ColumnProfile(
            name="z",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0
        )
        rep = DataQualityReport(
            row_count=1,
            column_count=2,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={"z": cp_z, "a": cp_a}  # Inserted z first
        )
        d = rep.to_dict()
        assert list(d["columns"].keys()) == ["a", "z"]
        
    def test_exclude_columns_filters_output(self):
        """Excluded columns don't appear in output."""
        cp_a = ColumnProfile(
            name="a",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0
        )
        cp_b = ColumnProfile(
            name="b",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0
        )
        rep = DataQualityReport(
            row_count=1,
            column_count=2,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={"a": cp_a, "b": cp_b}
        )
        d = rep.to_dict(exclude_columns=["a"])
        assert "a" not in d["columns"]
        assert "b" in d["columns"]
        
    def test_score_components_aliased(self):
        """score_components is same object (not copied)."""
        components = {"key": -5.0}
        rep = DataQualityReport(
            row_count=0,
            column_count=0,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
            score_components=components
        )
        d = rep.to_dict()
        assert d["score_components"] is components


class TestDataQualityReportToJson:
    """Test DataQualityReport.to_json()"""
    
    def test_to_json_returns_string(self):
        """to_json() without output returns a JSON string."""
        rep = DataQualityReport(
            row_count=1,
            column_count=0,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={}
        )
        result = rep.to_json()
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["row_count"] == 1
        
    def test_to_json_with_indent(self):
        """indent parameter affects formatting."""
        rep = DataQualityReport(
            row_count=1,
            column_count=0,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={}
        )
        json_no_indent = rep.to_json(indent=None)
        json_with_indent = rep.to_json(indent=2)
        # With indent should have newlines
        assert "\n" in json_with_indent
        # Both should parse to same dict
        assert json.loads(json_no_indent) == json.loads(json_with_indent)
        
    def test_to_json_writes_to_stream(self):
        """With output, writes to stream and returns None."""
        rep = DataQualityReport(
            row_count=1,
            column_count=0,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={}
        )
        stream = io.StringIO()
        result = rep.to_json(output=stream)
        assert result is None
        written = stream.getvalue()
        assert isinstance(written, str)
        assert json.loads(written)["row_count"] == 1
        
    def test_to_json_invalid_stream(self):
        """Non-writable output raises TypeError."""
        rep = DataQualityReport(
            row_count=1,
            column_count=0,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={}
        )
        with pytest.raises(TypeError, match="output must be a writable text stream"):
            rep.to_json(output=42)
            
    def test_to_json_respects_exclude_columns(self):
        """exclude_columns is passed through to to_dict."""
        cp = ColumnProfile(
            name="secret",
            dtype="string",
            semantic_type="text",
            row_count=1,
            null_count=0,
            null_ratio=0.0,
            unique_count=1,
            unique_ratio=1.0
        )
        rep = DataQualityReport(
            row_count=1,
            column_count=1,
            memory_usage=0,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={"secret": cp}
        )
        json_str = rep.to_json(exclude_columns=["secret"])
        parsed = json.loads(json_str)
        assert "secret" not in parsed["columns"]
        
    def test_to_json_matches_to_dict_json_dumps(self):
        """to_json() output equals json.dumps(to_dict())."""
        rep = DataQualityReport(
            row_count=5,
            column_count=0,
            memory_usage=42,
            duplicate_rows=1,
            duplicate_ratio=0.2,
            columns={},
            quality_score=85.0
        )
        json_str = rep.to_json()
        dict_obj = rep.to_dict()
        expected_json = json.dumps(dict_obj)
        assert json_str == expected_json
