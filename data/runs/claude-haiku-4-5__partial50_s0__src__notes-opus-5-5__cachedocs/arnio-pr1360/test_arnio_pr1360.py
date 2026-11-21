import pytest
import pandas as pd
import arnio as ar
from arnio.quality import ColumnProfile, DataQualityReport


def col(name="x", dtype="float64", null_ratio=0.0, uc=5, ur=0.5, rc=10, **kw):
    """Helper to create a ColumnProfile."""
    return ColumnProfile(
        name, 
        dtype, 
        "numeric", 
        rc, 
        0, 
        null_ratio, 
        uc, 
        ur, 
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
    
    def test_raises_typeerror_if_first_arg_not_report(self):
        """Should raise TypeError if profile_a is not a DataQualityReport."""
        profile_b = rep(col("x"))
        with pytest.raises(TypeError, match="compare_profiles expects two DataQualityReport instances"):
            ar.compare_profiles({}, profile_b)
    
    def test_raises_typeerror_if_second_arg_not_report(self):
        """Should raise TypeError if profile_b is not a DataQualityReport."""
        profile_a = rep(col("x"))
        with pytest.raises(TypeError, match="compare_profiles expects two DataQualityReport instances"):
            ar.compare_profiles(profile_a, None)


class TestCompareProfilesSchemaValidation:
    """Test schema compatibility checks."""
    
    def test_raises_valueerror_if_columns_differ(self):
        """Should raise ValueError if column sets don't match."""
        profile_a = rep(col("x"))
        profile_b = rep(col("x"), col("y"))
        with pytest.raises(ValueError, match="Profiles have incompatible schemas"):
            ar.compare_profiles(profile_a, profile_b)
    
    def test_error_message_shows_sorted_missing_columns(self):
        """Missing columns in error message should be sorted."""
        profile_a = rep(col("x"))
        profile_b = rep(col("x"), col("b"), col("a"))
        with pytest.raises(ValueError) as exc:
            ar.compare_profiles(profile_a, profile_b)
        assert "missing from profile_a=['a', 'b']" in str(exc.value)
        assert "missing from profile_b=[]" in str(exc.value)


class TestCompareProfilesReturnStructure:
    """Test return value structure and identity."""
    
    def test_returns_profilecomparison_with_required_fields(self):
        """Should return ProfileComparison with all required fields."""
        profile_a = rep(col("x"))
        profile_b = rep(col("x"))
        result = ar.compare_profiles(profile_a, profile_b)
        assert hasattr(result, "left_profile")
        assert hasattr(result, "right_profile")
        assert hasattr(result, "drift_report")
        assert hasattr(result, "status_counts")
    
    def test_profiles_are_stored_by_identity(self):
        """left_profile and right_profile should be the same objects."""
        profile_a = rep(col("x"))
        profile_b = rep(col("x"))
        result = ar.compare_profiles(profile_a, profile_b)
        assert result.left_profile is profile_a
        assert result.right_profile is profile_b
    
    def test_status_counts_always_has_three_keys(self):
        """status_counts must have ok, warning, changed keys."""
        profile_a = rep(col("x"))
        profile_b = rep(col("x"))
        result = ar.compare_profiles(profile_a, profile_b)
        assert result.status_counts.keys() == {"ok", "warning", "changed"}
    
    def test_drift_report_columns_sorted(self):
        """drift_report keys should be in sorted order."""
        profile_a = rep(col("b"), col("a"))
        profile_b = rep(col("b"), col("a"))
        result = ar.compare_profiles(profile_a, profile_b)
        assert list(result.drift_report.keys()) == ["a", "b"]


class TestCompareProfilesIdenticalProfiles:
    """Test behavior when profiles are identical."""
    
    def test_identical_profiles_all_ok(self):
        """Identical profiles should produce all ok status."""
        profile = rep(col("x", rc=10, null_ratio=0.1, uc=5, ur=0.5))
        result = ar.compare_profiles(profile, profile)
        assert result.status_counts == {"ok": 1, "warning": 0, "changed": 0}
        assert result.drift_report["x"]["status"] == "ok"
        assert result.drift_report["x"]["changes"] == {}
        assert result.drift_report["x"]["reasons"] == ["no drift detected"]
    
    def test_empty_profiles_produce_empty_report(self):
        """Empty profiles should produce empty drift_report."""
        profile_a = rep()
        profile_b = rep()
        result = ar.compare_profiles(profile_a, profile_b)
        assert result.drift_report == {}
        assert result.status_counts == {"ok": 0, "warning": 0, "changed": 0}


class TestCompareProfilesDtypeChanges:
    """Test dtype change detection."""
    
    def test_dtype_change_raises_changed_status(self):
        """dtype change should result in changed status."""
        profile_a = rep(col("x", dtype="float64"))
        profile_b = rep(col("x", dtype="string"))
        result = ar.compare_profiles(profile_a, profile_b)
        entry = result.drift_report["x"]
        assert entry["status"] == "changed"
        assert entry["changes"]["dtype"]["baseline"] == "float64"
        assert entry["changes"]["dtype"]["comparison"] == "string"
        assert entry["changes"]["dtype"]["delta"] is None


class TestCompareProfilesNullRatioDrift:
    """Test null_ratio threshold detection."""
    
    def test_null_ratio_delta_exceeds_warning_threshold(self):
        """null_ratio delta > 0.1 should trigger warning."""
        profile_a = rep(col("x", null_ratio=0.0))
        profile_b = rep(col("x", null_ratio=0.15))
        result = ar.compare_profiles(profile_a, profile_b)
        assert result.drift_report["x"]["status"] == "warning"
        assert result.drift_report["x"]["changes"]["null_ratio"]["delta"] == pytest.approx(0.15)
    
    def test_null_ratio_delta_exceeds_changed_threshold(self):
        """null_ratio delta > 0.25 should trigger changed."""
        profile_a = rep(col("x", null_ratio=0.0))
        profile_b = rep(col("x", null_ratio=0.3))
        result = ar.compare_profiles(profile_a, profile_b)
        assert result.drift_report["x"]["status"] == "changed"
    
    def test_null_ratio_delta_exactly_at_threshold_is_ok(self):
        """null_ratio delta exactly 0.1 should not trigger warning."""
        profile_a = rep(col("x", null_ratio=0.0))
        profile_b = rep(col("x", null_ratio=0.1))
        result = ar.compare_profiles(profile_a, profile_b)
        # Delta is in changes but status remains ok (not > 0.1)
        assert result.drift_report["x"]["status"] == "ok"
        assert "null_ratio" in result.drift_report["x"]["changes"]


class TestCompareProfilesUniqueCountDrift:
    """Test unique_count threshold detection."""
    
    def test_unique_count_warning_threshold(self):
        """unique_count delta > max(1.0, rcA*0.1, rcB*0.1) triggers warning."""
        profile_a = rep(col("x", rc=10, uc=5, ur=0.5))
        profile_b = rep(col("x", rc=10, uc=7, ur=0.7))
        result = ar.compare_profiles(profile_a, profile_b)
        entry = result.drift_report["x"]
        assert entry["status"] == "warning"
        assert entry["changes"]["unique_count"]["delta"] == 2
    
    def test_unique_count_changed_threshold(self):
        """unique_count delta > max(2.0, rcA*0.25, rcB*0.25) triggers changed."""
        profile_a = rep(col("x", rc=10, uc=5))
        profile_b = rep(col("x", rc=10, uc=8))
        result = ar.compare_profiles(profile_a, profile_b)
        entry = result.drift_report["x"]
        assert entry["status"] == "changed"
        assert entry["changes"]["unique_count"]["delta"] == 3


class TestCompareProfilesNumericMetricsDrift:
    """Test numeric metrics (mean, std, min, max)."""
    
    def test_mean_change_triggers_warning(self):
        """mean delta > scale*0.1 triggers warning."""
        profile_a = rep(col("x", dtype="float64", mean=10.0))
        profile_b = rep(col("x", dtype="float64", mean=12.0))
        result = ar.compare_profiles(profile_a, profile_b)
        entry = result.drift_report["x"]
        assert entry["status"] == "warning"
        assert "mean" in entry["changes"]
    
    def test_mean_change_triggers_changed(self):
        """mean delta > scale*0.25 triggers changed."""
        profile_a = rep(col("x", dtype="float64", mean=10.0))
        profile_b = rep(col("x", dtype="float64", mean=14.0))
        result = ar.compare_profiles(profile_a, profile_b)
        entry = result.drift_report["x"]
        assert entry["status"] == "changed"
    
    def test_numeric_metrics_only_compared_for_numeric_dtypes(self):
        """Numeric metrics should not be compared for string columns."""
        profile_a = rep(col("x", dtype="string", mean=5.0))
        profile_b = rep(col("x", dtype="string", mean=10.0))
        result = ar.compare_profiles(profile_a, profile_b)
        entry = result.drift_report["x"]
        # Numeric metrics should not appear for string dtype
        assert "mean" not in entry["changes"]


class TestCompareProfilesMultipleColumns:
    """Test with multiple columns."""
    
    def test_multiple_columns_status_counts(self):
        """Multiple columns with different statuses should be counted correctly."""
        profile_a = rep(
            col("a", null_ratio=0.0),
            col("b", null_ratio=0.0),
            col("c", dtype="int64")
        )
        profile_b = rep(
            col("a", null_ratio=0.0),
            col("b", null_ratio=0.15),
            col("c", dtype="string")
        )
        result = ar.compare_profiles(profile_a, profile_b)
        assert result.status_counts["ok"] == 1
        assert result.status_counts["warning"] == 1
        assert result.status_counts["changed"] == 1


class TestCompareProfilesEndToEnd:
    """End-to-end tests with real pandas data."""
    
    def test_end_to_end_with_real_dataframes(self):
        """Test with real pandas dataframes converted to profiles."""
        df_a = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        df_b = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 10.0]})
        
        profile_a = ar.profile(ar.from_pandas(df_a))
        profile_b = ar.profile(ar.from_pandas(df_b))
        
        result = ar.compare_profiles(profile_a, profile_b)
        assert "x" in result.drift_report
        assert result.drift_report["x"]["status"] in ["ok", "warning", "changed"]
