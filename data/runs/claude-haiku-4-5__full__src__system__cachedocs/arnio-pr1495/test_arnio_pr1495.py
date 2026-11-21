import pytest
import pandas as pd
import tempfile
from pathlib import Path

import arnio as ar


class TestProfileBasic:
    """Basic profiling functionality."""

    def test_profile_simple_dataframe(self):
        """Profile a simple DataFrame with mixed types."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1],
        })

        report = ar.profile(frame)

        assert report.row_count == 3
        assert report.column_count == 3
        assert report.duplicate_rows == 0
        assert 0 <= report.quality_score <= 100.0

    def test_profile_csv_file(self, tmp_path):
        """Profile data loaded from CSV file."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,name,active\n"
            "1,Alice,true\n"
            "2,Bob,false\n"
            "3,Charlie,true\n"
        )

        frame = ar.read_csv(str(csv_path))
        report = ar.profile(frame)

        assert report.row_count == 3
        assert report.column_count == 3

    def test_profile_with_nulls_and_duplicates(self):
        """Profile data with nulls and duplicate rows."""
        frame = ar.from_dict({
            "id": [1, 2, 2, 3, None],
            "value": [100, 200, 200, 300, 400],
        })

        report = ar.profile(frame)

        assert report.row_count == 5
        assert report.duplicate_rows == 1
        assert report.columns["id"].null_count == 1

    def test_profile_parameters_exclude_columns(self):
        """Exclude specific columns from profiling."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "sensitive": ["x", "y", "z"],
            "value": [100, 200, 300],
        })

        report = ar.profile(frame, exclude_columns=["sensitive"])

        assert report.column_count == 2
        assert "sensitive" not in report.columns

    def test_profile_parameters_sample_size(self):
        """Test sample_size parameter behavior."""
        frame = ar.from_dict({
            "col": ["a", "b", "c", "d", "e"],
        })

        report = ar.profile(frame, sample_size=2)

        assert len(report.columns["col"].sample_values) <= 2

    def test_profile_invalid_parameters(self):
        """Invalid parameters raise appropriate errors."""
        frame = ar.from_dict({"col": [1, 2, 3]})

        with pytest.raises(ValueError, match="sample_size must be non-negative"):
            ar.profile(frame, sample_size=-1)

        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(frame, sample_size=1.5)

        with pytest.raises(ValueError, match="column 'nonexistent'"):
            ar.profile(frame, exclude_columns=["nonexistent"])

    def test_profile_quality_score(self):
        """Quality score reflects data quality."""
        clean_frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })

        dirty_frame = ar.from_dict({
            "id": [1, 1, 2, None],
            "value": [100, 100, 200, None],
        })

        clean_report = ar.profile(clean_frame)
        dirty_report = ar.profile(dirty_frame)

        assert clean_report.quality_score >= 95.0
        assert dirty_report.quality_score < clean_report.quality_score
        assert isinstance(dirty_report.score_components, dict)

    def test_profile_numeric_statistics(self):
        """Numeric columns have correct min/max/mean/std."""
        frame = ar.from_dict({
            "values": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        report = ar.profile(frame)
        col = report.columns["values"]

        assert col.dtype == "float64"
        assert col.min == 10.0
        assert col.max == 50.0
        assert col.mean == 30.0
        assert col.std > 0

    def test_profile_string_statistics(self):
        """String columns show top values and frequencies."""
        frame = ar.from_dict({
            "category": ["A", "B", "A", "C", "A", "B"],
        })

        report = ar.profile(frame)
        col = report.columns["category"]

        assert col.dtype == "string"
        assert len(col.top_values) > 0
        # "A" should be the most frequent
        assert col.top_values[0].value == "A"

    def test_profile_report_exports(self):
        """Report can be exported to multiple formats."""
        frame = ar.from_dict({"id": [1, 2, 3], "value": [100, 200, 300]})
        report = ar.profile(frame)

        # Test dict export
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["row_count"] == 3
        assert "columns" in d

        # Test markdown export
        md = report.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

        # Test summary
        summary = report.summary()
        assert isinstance(summary, dict)
        assert "row_count" in summary

    def test_profile_pandas_accessor(self):
        """Profile using pandas .arnio accessor."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [100, 200, 300],
        })

        report = df.arnio.profile()

        assert report.row_count == 3
        assert report.column_count == 2

    def test_profile_edge_cases(self):
        """Profile handles edge cases correctly."""
        # Empty frame
        empty = ar.from_dict({"col1": [], "col2": []})
        empty_report = ar.profile(empty)
        assert empty_report.row_count == 0
        assert empty_report.duplicate_rows == 0

        # Single row
        single = ar.from_dict({"id": [1], "value": [100]})
        single_report = ar.profile(single)
        assert single_report.row_count == 1

        # All nulls column
        nulls = ar.from_dict({
            "id": [1, 2, 3],
            "nulls": [None, None, None],
        })
        nulls_report = ar.profile(nulls)
        assert nulls_report.columns["nulls"].null_count == 3
        assert nulls_report.columns["nulls"].null_ratio == 1.0

    def test_profile_before_after_cleaning(self):
        """Compare profiles before and after cleaning pipeline."""
        raw = ar.from_dict({
            "name": [" Alice ", " Bob ", " Charlie "],
            "value": [100, 200, 300],
        })

        raw_report = ar.profile(raw)

        clean = ar.pipeline(raw, [("strip_whitespace",)])
        clean_report = ar.profile(clean)

        # Same row count after cleaning
        assert raw_report.row_count == clean_report.row_count
        # Quality should be acceptable
        assert clean_report.quality_score >= 0

    def test_profile_with_cleaning_suggestions(self):
        """Profile integrates with cleaning suggestion system."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "value": [100, 200, 300],
        })

        report = ar.profile(frame)
        suggestions = ar.suggest_cleaning(report)

        assert isinstance(suggestions, list)
        # Clean data should have minimal suggestions
        assert len(suggestions) <= 5

    def test_profile_report_to_pandas(self):
        """Report can provide statistics as pandas DataFrame."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "value": [100, 200, 300],
        })

        report = ar.profile(frame)
        df = report.to_pandas()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
