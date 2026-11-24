import pytest
from arnio.quality import ColumnProfile, DataQualityReport, compare_profiles


def col(name="x", dtype="float64", null_ratio=0.0, unique_count=5,
        unique_ratio=0.5, row_count=10, semantic_type=None, **kw):
    """Helper to create a ColumnProfile."""
    if semantic_type is None:
        semantic_type = "numeric" if dtype in ("int64", "float64") else "string"
    return ColumnProfile(
        name=name,
        dtype=dtype,
        semantic_type=semantic_type,
        row_count=row_count,
        null_count=0,
        null_ratio=null_ratio,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        **kw
    )


def rep(*cols):
    """Helper to create a DataQualityReport."""
    return DataQualityReport(
        row_count=10,
        column_count=len(cols),
        memory_usage=0,
        duplicate_rows=0,
        duplicate_ratio=0.0,
        columns={c.name: c for c in cols}
    )


class TestCompareProfilesTypeValidation:
    """Test type checking."""

    def test_raises_typeerror_first_arg_not_report(self):
        """Should raise TypeError if profile_a is not a DataQualityReport."""
        profile_b = rep(col("x"))
        with pytest.raises(TypeError, match="compare_profiles expects two DataQualityReport instances"):
            compare_profiles({}, profile_b)

    def test_raises_typeerror_second_arg_not_report(self):
        """Should raise TypeError if profile_b is not a DataQualityReport."""
        profile_a = rep(col("x"))
        with pytest.raises(TypeError, match="compare_profiles expects two DataQualityReport instances"):
            compare_profiles(profile_a, [])


class TestCompareProfilesSchemaValidation:
    """Test schema (column set) validation."""

    def test_raises_valueerror_mismatched_columns(self):
        """Should raise ValueError if column sets don't match."""
        profile_a = rep(col("x"))
        profile_b = rep(col("x"), col("y"))
        with pytest.raises(ValueError, match="Profiles have incompatible schemas"):
            compare_profiles(profile_a, profile_b)

    def test_valueerror_message_missing_from_b(self):
        """Error message lists columns missing from each profile."""
        profile_a = rep(col("a"))
        profile_b = rep()
        with pytest.raises(ValueError) as exc_info:
            compare_profiles(profile_a, profile_b)
        assert "missing from profile_b=['a']" in str(exc_info.value)


class TestCompareProfilesBasic:
    """Test basic functionality and return structure."""

    def test_identical_profiles_all_ok(self):
        """Identical profiles should have all 'ok' status with no changes."""
        profile_a = rep(col("x"))
        profile_b = rep(col("x"))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.drift_report["x"]["status"] == "ok"
        assert comparison.drift_report["x"]["changes"] == {}
        assert comparison.drift_report["x"]["reasons"] == ["no drift detected"]
        assert comparison.status_counts == {"ok": 1, "warning": 0, "changed": 0}

    def test_empty_profiles(self):
        """Empty profiles should have empty drift_report."""
        profile_a = rep()
        profile_b = rep()
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.drift_report == {}
        assert comparison.status_counts == {"ok": 0, "warning": 0, "changed": 0}

    def test_profiles_stored_by_reference(self):
        """left_profile and right_profile should be the input objects."""
        profile_a = rep(col("x"))
        profile_b = rep(col("x"))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.left_profile is profile_a
        assert comparison.right_profile is profile_b

    def test_drift_report_columns_sorted(self):
        """drift_report keys should be in sorted column order."""
        profile_a = rep(col("c"), col("a"), col("b"))
        profile_b = rep(col("c"), col("a"), col("b"))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert list(comparison.drift_report.keys()) == ["a", "b", "c"]


class TestCompareProfilesDtype:
    """Test dtype change detection."""

    def test_dtype_changed_status_changed(self):
        """Dtype mismatch should produce 'changed' status."""
        profile_a = rep(col("x", dtype="int64"))
        profile_b = rep(col("x", dtype="string"))
        comparison = compare_profiles(profile_a, profile_b)
        
        entry = comparison.drift_report["x"]
        assert entry["status"] == "changed"
        assert "dtype changed from int64 to string" in entry["reasons"]
        assert entry["changes"]["dtype"]["baseline"] == "int64"
        assert entry["changes"]["dtype"]["comparison"] == "string"
        assert entry["changes"]["dtype"]["delta"] is None

    def test_dtype_no_change_no_entry_in_changes(self):
        """Same dtype should not appear in changes dict."""
        profile_a = rep(col("x", dtype="float64"))
        profile_b = rep(col("x", dtype="float64"))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert "dtype" not in comparison.drift_report["x"]["changes"]


class TestCompareProfilesNullRatio:
    """Test null_ratio drift detection."""

    def test_null_ratio_ok_below_warning_threshold(self):
        """null_ratio delta <= 0.1 should be 'ok'."""
        profile_a = rep(col("x", null_ratio=0.0))
        profile_b = rep(col("x", null_ratio=0.05))
        comparison = compare_profiles(profile_a, profile_b)
        
        entry = comparison.drift_report["x"]
        assert entry["status"] == "ok"
        assert "null ratio changed" in entry["reasons"]
        assert entry["changes"]["null_ratio"]["delta"] == pytest.approx(0.05)

    def test_null_ratio_warning_above_warning_threshold(self):
        """null_ratio delta in (0.1, 0.25] should be 'warning'."""
        profile_a = rep(col("x", null_ratio=0.0))
        profile_b = rep(col("x", null_ratio=0.2))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.drift_report["x"]["status"] == "warning"

    def test_null_ratio_changed_above_changed_threshold(self):
        """null_ratio delta > 0.25 should be 'changed'."""
        profile_a = rep(col("x", null_ratio=0.0))
        profile_b = rep(col("x", null_ratio=0.3))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.drift_report["x"]["status"] == "changed"


class TestCompareProfilesNumericMetrics:
    """Test numeric metric drift detection (mean, std, min, max)."""

    def test_numeric_metric_ok_below_threshold(self):
        """Numeric metric delta <= scale*0.1 should be 'ok'."""
        profile_a = rep(col("x", dtype="float64", mean=10.0))
        profile_b = rep(col("x", dtype="float64", mean=10.5))
        comparison = compare_profiles(profile_a, profile_b)
        
        entry = comparison.drift_report["x"]
        assert entry["status"] == "ok"
        assert "numeric mean changed" in entry["reasons"]
        assert entry["changes"]["mean"]["delta"] == pytest.approx(0.5)

    def test_numeric_metric_warning_in_range(self):
        """Numeric metric delta in (scale*0.1, scale*0.25] should be 'warning'."""
        profile_a = rep(col("x", dtype="float64", mean=10.0))
        profile_b = rep(col("x", dtype="float64", mean=11.5))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.drift_report["x"]["status"] == "warning"
        assert "numeric mean changed" in comparison.drift_report["x"]["reasons"]

    def test_numeric_metric_changed_above_threshold(self):
        """Numeric metric delta > scale*0.25 should be 'changed'."""
        profile_a = rep(col("x", dtype="float64", mean=10.0))
        profile_b = rep(col("x", dtype="float64", mean=13.0))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.drift_report["x"]["status"] == "changed"
        assert "numeric mean changed" in comparison.drift_report["x"]["reasons"]

    def test_numeric_metrics_skipped_for_string_dtype(self):
        """Numeric metrics should not be compared for string dtypes."""
        profile_a = rep(col("x", dtype="string"))
        profile_b = rep(col("x", dtype="string", mean=5.0))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert "mean" not in comparison.drift_report["x"]["changes"]
        assert comparison.drift_report["x"]["status"] == "ok"

    def test_numeric_metrics_skipped_if_none(self):
        """Numeric metrics should be skipped if either value is None."""
        profile_a = rep(col("x", dtype="float64", std=None))
        profile_b = rep(col("x", dtype="float64", std=1.0))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert "std" not in comparison.drift_report["x"]["changes"]


class TestCompareProfilesUniqueCount:
    """Test unique_count drift detection with row_count scaling."""

    def test_unique_count_ok_below_scaled_threshold(self):
        """unique_count delta below scaled threshold should be 'ok'."""
        # row_count=10, warning threshold = max(1.0, 10*0.1) = 1.0
        # delta = 1.0 is not > 1.0, so ok
        profile_a = rep(col("x", unique_count=5, row_count=10))
        profile_b = rep(col("x", unique_count=6, row_count=10))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.drift_report["x"]["status"] == "ok"
        assert "unique count changed" in comparison.drift_report["x"]["reasons"]

    def test_unique_count_warning_scaled(self):
        """unique_count delta in (warning, changed] with scaling should be 'warning'."""
        # row_count=10, threshold_w=1.0, threshold_c=max(2.0, 2.5)=2.5
        # delta=1.5 is > 1.0 and <= 2.5, so warning
        profile_a = rep(col("x", unique_count=5, row_count=10))
        profile_b = rep(col("x", unique_count=6.5, row_count=10))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.drift_report["x"]["status"] == "warning"

    def test_unique_count_changed_scaled(self):
        """unique_count delta > scaled changed threshold should be 'changed'."""
        # row_count=10, threshold_c=max(2.0, 10*0.25)=2.5
        # delta=3 > 2.5, so changed
        profile_a = rep(col("x", unique_count=5, row_count=10))
        profile_b = rep(col("x", unique_count=8, row_count=10))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.drift_report["x"]["status"] == "changed"


class TestCompareProfilesUniqueRatio:
    """Test unique_ratio drift detection."""

    def test_unique_ratio_ok(self):
        """unique_ratio delta <= 0.1 should be 'ok'."""
        profile_a = rep(col("x", unique_ratio=0.5))
        profile_b = rep(col("x", unique_ratio=0.55))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.drift_report["x"]["status"] == "ok"
        assert "unique ratio changed" in comparison.drift_report["x"]["reasons"]

    def test_unique_ratio_changed(self):
        """unique_ratio delta > 0.25 should be 'changed'."""
        profile_a = rep(col("x", unique_ratio=0.5))
        profile_b = rep(col("x", unique_ratio=0.8))
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.drift_report["x"]["status"] == "changed"
        assert "unique ratio changed" in comparison.drift_report["x"]["reasons"]


class TestCompareProfilesMultipleColumns:
    """Test comparisons with multiple columns and status aggregation."""

    def test_multiple_columns_status_counts(self):
        """Status counts should correctly aggregate all columns."""
        profile_a = rep(
            col("a", null_ratio=0.0),
            col("b", null_ratio=0.0),
            col("c", null_ratio=0.0)
        )
        profile_b = rep(
            col("a", null_ratio=0.05),   # ok (delta 0.05 <= 0.1)
            col("b", null_ratio=0.2),    # warning (0.1 < delta 0.2 <= 0.25)
            col("c", null_ratio=0.3)     # changed (delta 0.3 > 0.25)
        )
        
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.status_counts == {"ok": 1, "warning": 1, "changed": 1}
        assert len(comparison.drift_report) == 3


class TestCompareProfilesReportLevelIgnored:
    """Test that report-level fields do not affect comparisons."""

    def test_report_level_fields_ignored(self):
        """Report-level metrics should not appear in drift detection."""
        profile_a = rep(col("x"))
        profile_a.row_count = 100
        profile_a.duplicate_rows = 10
        profile_a.duplicate_ratio = 0.1
        
        profile_b = rep(col("x"))
        profile_b.row_count = 50
        profile_b.duplicate_rows = 0
        profile_b.duplicate_ratio = 0.0
        
        comparison = compare_profiles(profile_a, profile_b)
        
        assert comparison.drift_report["x"]["status"] == "ok"
        assert comparison.status_counts == {"ok": 1, "warning": 0, "changed": 0}
