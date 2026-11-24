import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestValidateInputValidation:
    """Test input validation and error conditions."""
    
    def test_non_arframe_raises_typeerror(self):
        """Non-ArFrame input raises TypeError."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(TypeError, match="validate\\(\\) expects a single ArFrame"):
            ar.validate(df, {"a": ar.Int64()})
    
    def test_max_errors_true_raises_typeerror(self):
        """max_errors=True raises TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="max_errors must be an int or None"):
            ar.validate(frame, {"a": ar.Int64()}, max_errors=True)
    
    def test_max_errors_string_raises_typeerror(self):
        """max_errors as string raises TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="max_errors must be an int or None"):
            ar.validate(frame, {"a": ar.Int64()}, max_errors="1")
    
    def test_max_errors_zero_raises_valueerror(self):
        """max_errors=0 raises ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, {"a": ar.Int64()}, max_errors=0)
    
    def test_max_errors_negative_raises_valueerror(self):
        """max_errors=-1 raises ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            ar.validate(frame, {"a": ar.Int64()}, max_errors=-1)


class TestValidateBasic:
    """Test basic validation behavior."""
    
    def test_valid_data_passes(self):
        """Valid data against schema passes."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.validate(frame, {"a": ar.Int64()})
        assert result.passed is True
        assert result.issue_count == 0
        assert result.row_count == 3
    
    def test_missing_required_column(self):
        """Missing column produces required_column issue."""
        frame = ar.from_pandas(pd.DataFrame({"b": [1, 2]}))
        result = ar.validate(frame, {"a": ar.Int64()})
        assert result.passed is False
        assert result.issue_count == 1
        assert result.issues[0].rule == "required_column"
        assert result.issues[0].column == "a"
        assert "Missing required column: a" in result.issues[0].message


class TestValidateNullability:
    """Test null handling."""
    
    def test_nullable_false_flags_null_values(self):
        """nullable=False creates issue for null rows."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, None, 3]}))
        result = ar.validate(frame, {"a": ar.Int64(nullable=False)})
        assert result.passed is False
        nullable_issues = [i for i in result.issues if i.rule == "nullable"]
        assert len(nullable_issues) >= 1
        assert 2 in result.bad_rows
    
    def test_nullable_true_allows_nulls(self):
        """nullable=True allows null values."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, None, 3]}))
        result = ar.validate(frame, {"a": ar.Int64(nullable=True)})
        assert result.passed is True
        assert result.issue_count == 0


class TestValidateUnique:
    """Test uniqueness constraints."""
    
    def test_unique_field_detects_duplicates(self):
        """unique=True detects duplicate values in field."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        result = ar.validate(frame, {"a": ar.Int64(unique=True)})
        assert result.passed is False
        unique_issues = [i for i in result.issues if i.rule == "unique"]
        assert len(unique_issues) >= 1


class TestValidateStrict:
    """Test strict mode."""
    
    def test_strict_detects_unexpected_columns(self):
        """strict=True reports unexpected columns."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        result = ar.validate(
            frame,
            ar.Schema({"a": ar.Int64()}, strict=True)
        )
        assert result.passed is False
        unexpected = [i for i in result.issues if i.rule == "unexpected_column"]
        assert len(unexpected) >= 1
        assert "b" in unexpected[0].message
    
    def test_non_strict_allows_extra_columns(self):
        """strict=False allows extra columns."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        result = ar.validate(
            frame,
            ar.Schema({"a": ar.Int64()}, strict=False)
        )
        assert result.passed is True


class TestValidateMaxErrors:
    """Test max_errors truncation."""
    
    def test_max_errors_limits_issues(self):
        """max_errors truncates result to specified count."""
        frame = ar.from_pandas(pd.DataFrame({
            "a": [None, None, 3],
            "b": [None, 2, 3]
        }))
        result = ar.validate(
            frame,
            ar.Schema({
                "a": ar.Int64(nullable=False),
                "b": ar.Int64(nullable=False)
            }),
            max_errors=2
        )
        assert result.issue_count == 2
    
    def test_max_errors_one_returns_single_issue(self):
        """max_errors=1 returns exactly one issue."""
        frame = ar.from_pandas(pd.DataFrame({"a": [None, None, None]}))
        result = ar.validate(
            frame,
            {"a": ar.Int64(nullable=False)},
            max_errors=1
        )
        assert result.issue_count == 1


class TestValidateDictSchema:
    """Test dict to Schema conversion."""
    
    def test_dict_converted_to_schema(self):
        """Dict argument is converted to Schema."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.validate(frame, {"a": ar.Int64()})
        assert result.passed is True


class TestValidateRules:
    """Test custom rule validation."""
    
    def test_rule_returning_non_list_raises_typeerror(self):
        """Rule returning non-list raises TypeError."""
        def bad_rule(df):
            return None
        
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="must return a list"):
            ar.validate(
                frame,
                ar.Schema({"a": ar.Int64()}, rules=[bad_rule])
            )
    
    def test_rule_with_non_validationissue_raises_typeerror(self):
        """Rule returning non-ValidationIssue item raises TypeError."""
        def bad_rule(df):
            return [42]
        
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="non-ValidationIssue"):
            ar.validate(
                frame,
                ar.Schema({"a": ar.Int64()}, rules=[bad_rule])
            )
    
    def test_valid_rule_produces_issues(self):
        """Valid rule produces issues."""
        def rule_check(df):
            if len(df) > 1:
                return [ar.ValidationIssue(
                    column=None,
                    rule="custom",
                    message="Too many rows"
                )]
            return []
        
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.validate(
            frame,
            ar.Schema({"a": ar.Int64()}, rules=[rule_check])
        )
        custom_issues = [i for i in result.issues if i.rule == "custom"]
        assert len(custom_issues) >= 1


class TestValidateEdgeCases:
    """Test edge cases and special behaviors."""
    
    def test_empty_frame(self):
        """Empty frame passes with valid schema."""
        frame = ar.from_pandas(pd.DataFrame({"a": pd.Series([], dtype="int64")}))
        result = ar.validate(frame, {"a": ar.Int64()})
        assert result.passed is True
        assert result.row_count == 0
    
    def test_composite_unique_empty_list_raises_issue(self):
        """Empty unique list produces issue."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        result = ar.validate(
            frame,
            ar.Schema({"a": ar.Int64()}, unique=[])
        )
        composite_issues = [i for i in result.issues if i.rule == "composite_unique"]
        assert len(composite_issues) >= 1
    
    def test_composite_unique_missing_column(self):
        """Composite unique with missing column creates issue."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        result = ar.validate(
            frame,
            ar.Schema({"a": ar.Int64()}, unique=["a", "missing"])
        )
        missing_issues = [i for i in result.issues if i.rule == "missing_column"]
        assert len(missing_issues) >= 1
        assert "missing" in missing_issues[0].message
    
    def test_bad_rows_one_based_indices(self):
        """bad_rows contains 1-based row indices."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, None, 3]}))
        result = ar.validate(frame, {"a": ar.Int64(nullable=False)})
        # Second row (pandas index 1) is row 2 in 1-based indexing
        assert all(isinstance(idx, int) and idx > 0 for idx in result.bad_rows)
        assert 2 in result.bad_rows
    
    def test_whitespace_treated_as_null(self):
        """Whitespace-only strings treated as null."""
        frame = ar.from_pandas(pd.DataFrame({"a": ["  ", "text"]}))
        result = ar.validate(frame, {"a": ar.String(nullable=False)})
        nullable_issues = [i for i in result.issues if i.rule == "nullable"]
        assert len(nullable_issues) >= 1
        assert 1 in result.bad_rows
