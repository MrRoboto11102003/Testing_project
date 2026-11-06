import pytest
import pandas as pd
import arnio as ar


class TestValidate:
    """Tests for arnio.schema.validate function."""

    @pytest.fixture
    def valid_frame(self):
        """Basic valid frame for testing."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["a@example.com", "b@example.com", "c@example.com"],
            "age": [25, 30, 35],
        })
        return ar.from_pandas(df)

    def test_validate_with_valid_data(self, valid_frame):
        """Valid data passes validation."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
            "age": ar.Int64(),
        })
        result = ar.validate(valid_frame, schema)
        assert result.passed
        assert result.issue_count == 0
        assert len(result.issues) == 0

    def test_validate_with_dict_schema(self, valid_frame):
        """Schema can be passed as dict instead of Schema object."""
        result = ar.validate(valid_frame, {
            "id": ar.Int64(),
            "email": ar.Email(),
            "age": ar.Int64(),
        })
        assert result.passed

    def test_validate_missing_required_column(self, valid_frame):
        """Missing column in data produces required_column issue."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
            "missing_col": ar.String(),
        })
        result = ar.validate(valid_frame, schema)
        assert not result.passed
        assert any(issue.rule == "required_column" for issue in result.issues)

    def test_validate_nullable_violation(self):
        """nullable=False rejects null values."""
        df = pd.DataFrame({
            "id": [1, None, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(),
        })
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "nullable" for issue in result.issues)

    def test_validate_unique_violation(self):
        """unique=True rejects duplicate values."""
        df = pd.DataFrame({
            "id": [1, 2, 1],
            "name": ["Alice", "Bob", "Charlie"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(unique=True),
            "name": ar.String(),
        })
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "unique" for issue in result.issues)

    def test_validate_composite_unique(self):
        """Composite unique checks multiple columns together."""
        df = pd.DataFrame({
            "user_id": [1, 1, 2],
            "course_id": [10, 10, 10],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {
                "user_id": ar.Int64(),
                "course_id": ar.Int64(),
            },
            unique=["user_id", "course_id"],
        )
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "composite_unique" for issue in result.issues)

    def test_validate_strict_mode_rejects_extra_columns(self, valid_frame):
        """strict=True rejects unexpected columns."""
        schema = ar.Schema(
            {
                "id": ar.Int64(),
                "email": ar.Email(),
            },
            strict=True,
        )
        result = ar.validate(valid_frame, schema)
        assert not result.passed
        assert any(issue.rule == "unexpected_column" for issue in result.issues)

    def test_validate_max_errors_limits_result_count(self):
        """max_errors stops validation early."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["invalid", "bad", "worse"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
        })
        result = ar.validate(frame, schema, max_errors=2)
        assert len(result.issues) <= 2

    def test_validate_max_errors_type_validation(self, valid_frame):
        """max_errors must be int or None."""
        schema = ar.Schema({"id": ar.Int64()})
        with pytest.raises(TypeError):
            ar.validate(valid_frame, schema, max_errors="two")
        with pytest.raises(TypeError):
            ar.validate(valid_frame, schema, max_errors=2.5)

    def test_validate_max_errors_positive(self, valid_frame):
        """max_errors must be >= 1."""
        schema = ar.Schema({"id": ar.Int64()})
        with pytest.raises(ValueError):
            ar.validate(valid_frame, schema, max_errors=0)
        with pytest.raises(ValueError):
            ar.validate(valid_frame, schema, max_errors=-1)

    def test_validate_non_arframe_raises_error(self):
        """Non-ArFrame input raises TypeError."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        schema = ar.Schema({"id": ar.Int64()})
        with pytest.raises(TypeError, match="single ArFrame"):
            ar.validate(df, schema)

    def test_validate_cross_field_rule(self):
        """Cross-field rules are evaluated on full DataFrame."""
        def dates_ordered(df):
            issues = []
            for i, row in df.iterrows():
                if row["end"] < row["start"]:
                    issues.append(
                        ar.ValidationIssue(
                            column="end",
                            rule="cross_field",
                            message="end must be >= start",
                            row_index=int(i) + 1,
                        )
                    )
            return issues

        df = pd.DataFrame({
            "start": ["2024-01-01", "2024-02-01"],
            "end": ["2024-12-31", "2024-01-15"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {
                "start": ar.String(),
                "end": ar.String(),
            },
            rules=[dates_ordered],
        )
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "cross_field" for issue in result.issues)

    def test_validate_unique_empty_list_fails(self):
        """Empty unique list produces validation issue."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {"id": ar.Int64()},
            unique=[],
        )
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "composite_unique" for issue in result.issues)

    def test_validate_unique_must_contain_strings(self, valid_frame):
        """unique list items must be strings."""
        schema = ar.Schema(
            {"id": ar.Int64()},
            unique=[123],
        )
        with pytest.raises(TypeError):
            ar.validate(valid_frame, schema)

    def test_validate_bad_rows_tracking(self):
        """bad_rows field tracks rows with validation issues."""
        df = pd.DataFrame({
            "id": [1, None, 3],
            "name": ["A", "B", "C"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(),
        })
        result = ar.validate(frame, schema)
        assert len(result.bad_rows) > 0
        assert any(issue.row_index for issue in result.issues if issue.row_index)
