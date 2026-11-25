import pytest
import arnio as ar


class TestProfile:
    """Tests for arnio.quality.profile function"""

    def test_profile_basic_returns_report(self):
        """Test basic profiling returns a DataQualityReport"""
        frame = ar.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})
        report = ar.profile(frame)
        
        assert report is not None
        assert hasattr(report, 'summary')
        assert hasattr(report, 'to_markdown')
        assert hasattr(report, 'to_html')

    def test_profile_summary_returns_dict(self):
        """Test that summary() returns a dictionary"""
        frame = ar.from_dict({"col": [1, 2, 3]})
        report = ar.profile(frame)
        summary = report.summary()
        
        assert isinstance(summary, dict)

    def test_profile_exclude_columns(self):
        """Test profiling with excluded columns parameter"""
        frame = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        report = ar.profile(frame, exclude_columns=["col3"])
        
        assert report is not None
        assert report.summary() is not None

    def test_profile_sample_size_parameter(self):
        """Test sample_size parameter accepts different values"""
        frame = ar.from_dict({"vals": list(range(100))})
        
        report1 = ar.profile(frame, sample_size=5)
        report2 = ar.profile(frame, sample_size=20)
        
        assert report1 is not None
        assert report2 is not None

    def test_profile_empty_frame(self):
        """Test profiling an empty frame"""
        frame = ar.from_dict({"col1": [], "col2": []})
        report = ar.profile(frame)
        
        assert report is not None
        assert isinstance(report.summary(), dict)

    def test_profile_single_column(self):
        """Test profiling a single-column frame"""
        frame = ar.from_dict({"values": [1, 2, 3, 4, 5]})
        report = ar.profile(frame)
        
        assert report is not None

    def test_profile_all_null_column(self):
        """Test profiling a column with all null values"""
        frame = ar.from_dict({"col": [None, None, None]})
        report = ar.profile(frame)
        
        assert report is not None

    def test_profile_mixed_dtypes(self):
        """Test profiling mixed data types"""
        frame = ar.from_dict({
            "ints": [1, 2, 3],
            "floats": [1.5, 2.5, 3.5],
            "strs": ["a", "b", "c"],
            "bools": [True, False, True]
        })
        report = ar.profile(frame)
        
        assert report is not None
        summary = report.summary()
        assert isinstance(summary, dict)

    def test_profile_approx_top_values_disabled(self):
        """Test profiling with approximation disabled (default)"""
        frame = ar.from_dict({
            "cats": ["cat_" + str(i % 100) for i in range(1000)]
        })
        report = ar.profile(frame, approx_top_values=False)
        
        assert report is not None

    def test_profile_approx_top_values_enabled(self):
        """Test profiling with approximation enabled and thresholds"""
        frame = ar.from_dict({
            "vals": ["val_" + str(i % 2000) for i in range(5000)]
        })
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_unique=1000,
            approx_top_values_min_ratio=0.2,
            approx_top_values_sample_size=2000
        )
        
        assert report is not None

    def test_profile_markdown_output(self):
        """Test markdown output is a non-empty string"""
        frame = ar.from_dict({"col": [1, 2, 3]})
        report = ar.profile(frame)
        markdown = report.to_markdown()
        
        assert isinstance(markdown, str)
        assert len(markdown) > 0

    def test_profile_html_output(self):
        """Test HTML output is a non-empty string"""
        frame = ar.from_dict({"col": [1, 2, 3]})
        report = ar.profile(frame)
        html = report.to_html()
        
        assert isinstance(html, str)
        assert len(html) > 0

    def test_profile_with_nulls_and_duplicates(self):
        """Test profiling with null values and duplicate rows"""
        frame = ar.from_dict({
            "id": [1, 2, 2, 3, None],
            "val": [10, 20, 20, 30, 40]
        })
        report = ar.profile(frame)
        
        assert report is not None

    def test_profile_numeric_only_frame(self):
        """Test profiling frame with only numeric columns"""
        frame = ar.from_dict({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        report = ar.profile(frame)
        
        assert report is not None

    def test_profile_string_only_frame(self):
        """Test profiling frame with only string columns"""
        frame = ar.from_dict({
            "col1": ["apple", "banana", "cherry"],
            "col2": ["x", "y", "z", "x"]
        })
        report = ar.profile(frame)
        
        assert report is not None
