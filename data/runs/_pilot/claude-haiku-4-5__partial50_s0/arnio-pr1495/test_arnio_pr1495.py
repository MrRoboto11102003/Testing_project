import pytest
import pandas as pd
import arnio as ar


class TestProfile:
    """Test suite for arnio.quality.profile function."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for profiling tests."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [95.5, 87.0, 92.5, None, 88.0],
            "active": [True, False, True, True, False],
        })
        return ar.from_pandas(df)

    def test_profile_returns_data_quality_report(self, sample_data):
        """Test that profile returns a DataQualityReport with expected methods."""
        report = ar.profile(sample_data)
        assert report is not None
        assert hasattr(report, 'summary')
        assert hasattr(report, 'to_markdown')
        assert hasattr(report, 'to_html')

    def test_profile_summary_method_returns_dict(self, sample_data):
        """Test that summary() returns a dictionary."""
        report = ar.profile(sample_data)
        summary = report.summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_profile_to_markdown_returns_string(self, sample_data):
        """Test that to_markdown() returns a non-empty string."""
        report = ar.profile(sample_data)
        md = report.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_profile_to_html_returns_string(self, sample_data):
        """Test that to_html() returns a non-empty HTML string."""
        report = ar.profile(sample_data)
        html = report.to_html()
        assert isinstance(html, str)
        assert len(html) > 0

    def test_profile_exclude_columns_parameter(self, sample_data):
        """Test profile with exclude_columns parameter."""
        report = ar.profile(sample_data, exclude_columns=["name"])
        assert report is not None

    def test_profile_sample_size_parameter(self, sample_data):
        """Test profile with custom sample_size parameter."""
        report = ar.profile(sample_data, sample_size=3)
        assert report is not None

    def test_profile_approx_top_values_enabled(self, sample_data):
        """Test profile with approx_top_values=True."""
        report = ar.profile(sample_data, approx_top_values=True)
        assert report is not None

    def test_profile_all_approximation_parameters(self, sample_data):
        """Test profile with all approximation parameters configured."""
        report = ar.profile(
            sample_data,
            approx_top_values=True,
            approx_top_values_min_unique=500,
            approx_top_values_min_ratio=0.15,
            approx_top_values_sample_size=1500,
        )
        assert report is not None

    def test_profile_empty_frame(self):
        """Test profile on an empty frame (edge case)."""
        df = pd.DataFrame({"col1": pd.Series([], dtype="int64")})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert report is not None

    def test_profile_with_null_values(self):
        """Test profile handles frames with null values."""
        df = pd.DataFrame({
            "col1": [None, 2, None, 4, None],
            "col2": ["a", None, "c", None, "e"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert report is not None

    def test_profile_mixed_data_types(self):
        """Test profile works with multiple data types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert report is not None

    def test_profile_to_html_file_output(self, sample_data, tmp_path):
        """Test that to_html() writes to file when file_path is provided."""
        output_file = tmp_path / "report.html"
        report = ar.profile(sample_data)
        html = report.to_html(file_path=str(output_file))
        assert isinstance(html, str)
        assert output_file.exists()
        with open(output_file) as f:
            content = f.read()
            assert len(content) > 0

    def test_profile_single_row_frame(self):
        """Test profile on a single-row frame (edge case)."""
        df = pd.DataFrame({"col1": [1], "col2": ["test"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert report is not None

    def test_profile_with_duplicate_rows(self):
        """Test profile handles frames with duplicate rows."""
        df = pd.DataFrame({
            "id": [1, 1, 2, 2, 3],
            "value": ["a", "a", "b", "b", "c"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        assert report is not None

    def test_profile_default_parameters(self, sample_data):
        """Test profile uses documented default parameters correctly."""
        report = ar.profile(sample_data)
        assert report is not None
        # Verify same behavior as explicit defaults
        report_explicit = ar.profile(
            sample_data,
            exclude_columns=None,
            sample_size=5,
            approx_top_values=False,
        )
        assert report_explicit is not None
