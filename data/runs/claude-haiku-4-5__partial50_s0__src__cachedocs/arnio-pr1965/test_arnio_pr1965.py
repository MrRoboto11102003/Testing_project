import pytest
import pandas as pd
import arnio as ar


class TestValidateBasics:
    """Test basic validate() functionality."""

    def test_validate_accepts_arframe_with_schema_object(self):
        """validate() should accept ArFrame and Schema object."""
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        result = ar.validate(frame, schema)
        assert result.passed is True
        assert result.issue_count == 0

    def test_validate_accepts_dict_schema(self):
        """validate() should accept dict[str, Field] as schema."""
        df = pd.DataFrame({"id": [1, 2]})
        frame = ar.from_pandas(df)
        schema_dict = {"id": ar.Int64()}
        result = ar.validate(frame, schema_dict)
        assert isinstance(result, ar.ValidationResult)
        assert result.passed is True

    def test_validate_rejects_non_arframe(self):
        """validate() should reject non-ArFrame input."""
        df = pd.DataFrame({"id": [1, 2]})
        schema = ar.Schema({"id": ar.Int64()})
        with pytest.raises(TypeError, match="single ArFrame"):
            ar.validate(df, schema)

    def test_validate_returns_validation_result_with_attributes(self):
        """ValidationResult should have required attributes."""
        df = pd.DataFrame({"id": [1, 2]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64()})
        result = ar.validate(frame, schema)
        assert hasattr(result, "passed")
        assert hasattr(result, "issues")
        assert hasattr(result, "bad_rows")
        assert hasattr(result, "row_count")
        assert hasattr(result, "issue_count")
        assert result.row_count == 2


class TestMissingColumns:
    """Test handling of missing required columns."""

    def test_missing_required_column_reported(self):
        """Should report missing required columns as issues."""
        df = pd.DataFrame({"id": [1, 2]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(), "email": ar.Email()})
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issue_count > 0
        email_issues = [i for i in result.issues if i.column == "email"]
        assert len(email_issues) > 0
        assert "required_column" in email_issues[0].rule


class TestStrictMode:
    """Test schema.strict mode for unexpected columns."""

    def test_strict_mode_rejects_unexpected_columns(self):
        """When strict=True, should report unexpected columns."""
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "extra": [999]})
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {"id": ar.Int64(), "name": ar.String()},
            strict=True
        )
        result = ar.validate(frame, schema)
        assert result.passed is False
        unexpected = [i for i in result.issues if "unexpected_column" in i.rule]
        assert len(unexpected) > 0

    def test_non_strict_mode_allows_extra_columns(self):
        """When strict=False, should ignore extra columns."""
        df = pd.DataFrame({"id": [1], "name": ["Alice"], "extra": [999]})
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {"id": ar.Int64(), "name": ar.String()},
            strict=False
        )
        result = ar.validate(frame, schema)
        assert result.passed is True


class TestUniqueConstraints:
    """Test unique and composite unique constraints."""

    def test_composite_unique_detects_duplicates(self):
        """Composite unique should detect duplicate rows on specified columns."""
        df = pd.DataFrame({
            "user_id": [1, 1, 2],
            "course_id": [101, 101, 101],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {"user_id": ar.Int64(), "course_id": ar.Int64()},
            unique=["user_id", "course_id"]
        )
        result = ar.validate(frame, schema)
        assert result.passed is False
        composite_issues = [i for i in result.issues if "composite_unique" in i.rule]
        assert len(composite_issues) > 0

    def test_composite_unique_with_missing_column(self):
        """Should report missing columns referenced in unique constraint."""
        df = pd.DataFrame({"user_id": [1, 2]})
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {"user_id": ar.Int64()},
            unique=["user_id", "course_id"]
        )
        result = ar.validate(frame, schema)
        assert result.passed is False
        missing = [i for i in result.issues if i.column == "course_id"]
        assert len(missing) > 0

    def test_unique_empty_list_error(self):
        """Empty unique constraint list should report error."""
        df = pd.DataFrame({"id": [1, 2]})
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {"id": ar.Int64()},
            unique=[]
        )
        result = ar.validate(frame, schema)
        assert result.passed is False
        composite_issues = [i for i in result.issues if "composite_unique" in i.rule]
        assert len(composite_issues) > 0


class TestMaxErrors:
    """Test max_errors parameter to limit issue count."""

    def test_max_errors_limits_issues(self):
        """max_errors should limit number of returned issues."""
        df = pd.DataFrame({"col1": ["x"], "col2": ["y"], "col3": ["z"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "col1": ar.Int64(),
            "col2": ar.Int64(),
            "col3": ar.Int64()
        })
        result = ar.validate(frame, schema, max_errors=2)
        assert result.issue_count <= 2

    def test_max_errors_invalid_type_raises(self):
        """max_errors with non-int type should raise TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1]}))
        schema = ar.Schema({"id": ar.Int64()})
        with pytest.raises(TypeError, match="max_errors must be an int"):
            ar.validate(frame, schema, max_errors="invalid")

    def test_max_errors_boolean_raises(self):
        """max_errors as bool should raise TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1]}))
        schema = ar.Schema({"id": ar.Int64()})
        with pytest.raises(TypeError, match="max_errors must be an int"):
            ar.validate(frame, schema, max_errors=True)

    def test_max_errors_zero_or_negative_raises(self):
        """max_errors <= 0 should raise ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1]}))
        schema = ar.Schema({"id": ar.Int64()})
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, schema, max_errors=0)
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, schema, max_errors=-1)


class TestSchemaUniqueValidation:
    """Test validation of schema.unique parameter itself."""

    def test_unique_not_list_or_tuple_raises(self):
        """schema.unique that is not list/tuple should raise TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1]}))
        schema = ar.Schema({"id": ar.Int64()}, unique="id")
        with pytest.raises(TypeError, match="list or tuple"):
            ar.validate(frame, schema)

    def test_unique_with_non_string_member_raises(self):
        """schema.unique with non-string members should raise TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1]}))
        schema = ar.Schema({"id": ar.Int64()}, unique=[1, "id"])
        with pytest.raises(TypeError, match="members must be strings"):
            ar.validate(frame, schema)


class TestValidationResult:
    """Test ValidationResult properties."""

    def test_passed_true_when_no_issues(self):
        """ValidationResult.passed should be True when no issues exist."""
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        result = ar.validate(frame, schema)
        assert result.passed is True
        assert result.issue_count == 0

    def test_passed_false_when_issues_exist(self):
        """ValidationResult.passed should be False when issues exist."""
        df = pd.DataFrame({"id": [1, 2]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issue_count > 0

    def test_bad_rows_is_sorted(self):
        """ValidationResult.bad_rows should be sorted."""
        df = pd.DataFrame({"id": [1, "invalid", 1]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(unique=True)})
        result = ar.validate(frame, schema)
        if result.bad_rows:
            assert result.bad_rows == sorted(result.bad_rows)

    def test_row_index_is_one_based(self):
        """ValidationIssue.row_index should be 1-based (header not counted)."""
        df = pd.DataFrame({
            "user_id": [1, 1],
            "course_id": [101, 101]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {"user_id": ar.Int64(), "course_id": ar.Int64()},
            unique=["user_id", "course_id"]
        )
        result = ar.validate(frame, schema)
        # Rows with duplicates should have 1-based indices
        for issue in result.issues:
            if issue.row_index is not None:
                assert issue.row_index >= 1
