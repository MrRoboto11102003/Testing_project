import pytest
import arnio as ar


class TestProfileBasics:
    """Test basic profile functionality."""
    
    def test_profile_simple_frame(self):
        """Test profiling a simple frame."""
        frame = ar.from_dict({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.column_count == 2
        assert "col1" in report.columns
        assert "col2" in report.columns
        assert hasattr(report, 'quality_score')


class TestExcludeColumns:
    """Test exclude_columns parameter."""
    
    def test_exclude_columns_basic(self):
        """Test that specified columns are excluded from report."""
        frame = ar.from_dict({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        report = ar.profile(frame, exclude_columns=["b"])
        
        assert report.column_count == 2
        assert "b" not in report.columns
        assert "a" in report.columns
    
    def test_exclude_nonexistent_column_raises(self):
        """Test that excluding nonexistent column raises error."""
        frame = ar.from_dict({"a": [1, 2]})
        
        with pytest.raises(Exception):
            ar.profile(frame, exclude_columns=["nonexistent"])


class TestSampleSizeValidation:
    """Test sample_size parameter validation."""
    
    def test_sample_size_zero_valid(self):
        """Test that sample_size=0 is valid."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        ar.profile(frame, sample_size=0)  # Should not raise
    
    def test_sample_size_negative_raises(self):
        """Test that negative sample_size raises ValueError."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="sample_size must be non-negative"):
            ar.profile(frame, sample_size=-1)
    
    def test_sample_size_float_raises(self):
        """Test that float sample_size raises TypeError."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(frame, sample_size=5.5)
    
    def test_sample_size_bool_raises(self):
        """Test that bool sample_size raises TypeError."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(frame, sample_size=True)


class TestApproxTopValuesValidation:
    """Test approx_top_values and related parameters."""
    
    def test_approx_top_values_not_bool_raises(self):
        """Test that approx_top_values must be bool."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="approx_top_values must be a bool"):
            ar.profile(frame, approx_top_values="yes")
    
    def test_min_unique_negative_raises(self):
        """Test that negative min_unique raises ValueError."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="approx_top_values_min_unique must be non-negative"):
            ar.profile(frame, approx_top_values_min_unique=-1)
    
    def test_min_unique_bool_raises(self):
        """Test that bool min_unique raises TypeError."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="approx_top_values_min_unique must be an integer"):
            ar.profile(frame, approx_top_values_min_unique=True)
    
    def test_min_ratio_bounds(self):
        """Test that min_ratio must be between 0 and 1."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        
        ar.profile(frame, approx_top_values_min_ratio=0.5)
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            ar.profile(frame, approx_top_values_min_ratio=1.1)
    
    def test_min_ratio_finite(self):
        """Test that min_ratio must be finite."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="finite"):
            ar.profile(frame, approx_top_values_min_ratio=float('nan'))
    
    def test_sample_size_must_be_positive(self):
        """Test that approx_top_values_sample_size must be positive."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="approx_top_values_sample_size must be positive"):
            ar.profile(frame, approx_top_values_sample_size=0)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_frame(self):
        """Test profiling an empty frame."""
        frame = ar.from_dict({"col": []})
        report = ar.profile(frame)
        
        assert report.row_count == 0
    
    def test_with_null_values(self):
        """Test profiling with null values."""
        frame = ar.from_dict({"col": [1, None, 3, None, 5]})
        report = ar.profile(frame)
        
        assert report.row_count == 5
        assert "col" in report.columns
    
    def test_duplicate_rows_counted(self):
        """Test that duplicate rows are detected and counted."""
        frame = ar.from_dict({"a": [1, 1, 2], "b": [3, 3, 4]})
        report = ar.profile(frame)
        
        assert report.duplicate_rows == 1
        assert 0 < report.duplicate_ratio < 1
