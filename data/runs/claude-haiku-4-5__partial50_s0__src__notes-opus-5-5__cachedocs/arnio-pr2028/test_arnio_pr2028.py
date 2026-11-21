import pytest
import re
import numpy as np
import pandas as pd
import arnio as ar


class TestStringFieldCreation:
    """Test ar.String factory - basic creation and parameter handling."""
    
    def test_string_basic_defaults(self):
        """Test String with default parameters."""
        field = ar.String()
        assert field.dtype == "string"
        assert field.nullable is True
        assert field.unique is False
        assert field.pattern is None
        assert field.allowed is None
        assert field.min_length is None
        assert field.max_length is None
        assert field.case_sensitive is True
        assert field.severity == "error"
    
    def test_string_with_custom_parameters(self):
        """Test String with custom parameters."""
        field = ar.String(
            nullable=False,
            unique=True,
            case_sensitive=False,
            min_length=3,
            max_length=10,
            severity="warning"
        )
        assert field.nullable is False
        assert field.unique is True
        assert field.case_sensitive is False
        assert field.min_length == 3
        assert field.max_length == 10
        assert field.severity == "warning"
    
    def test_string_allowed_list_becomes_set(self):
        """Test that allowed list is converted to set, collapsing duplicates."""
        field = ar.String(allowed=["red", "green", "blue", "red"])
        assert field.allowed == {"red", "green", "blue"}
        assert isinstance(field.allowed, set)
    
    def test_string_allowed_empty_list(self):
        """Test that allowed=[] creates empty set, not None."""
        field = ar.String(allowed=[])
        assert field.allowed == set()
        assert field.allowed is not None
    
    def test_string_allowed_dict_becomes_key_set(self):
        """Test that allowed dict becomes set of keys."""
        field = ar.String(allowed={"x": 1, "y": 2})
        assert field.allowed == {"x", "y"}
    
    def test_string_pattern_valid_regex(self):
        """Test valid regex patterns are accepted."""
        field = ar.String(pattern=r"^[A-Z][a-z]+$")
        assert field.pattern == r"^[A-Z][a-z]+$"
    
    def test_string_min_max_length_zero_allowed(self):
        """Test that min_length=0 and max_length=0 are valid."""
        field = ar.String(min_length=0, max_length=0)
        assert field.min_length == 0
        assert field.max_length == 0


class TestStringFieldErrorHandling:
    """Test argument validation and error conditions."""
    
    def test_string_min_length_negative_raises_valueerror(self):
        """Test that negative min_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be greater than or equal to 0"):
            ar.String(min_length=-1)
    
    def test_string_min_max_inverted_raises_valueerror(self):
        """Test that min_length > max_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be less than or equal to max_length"):
            ar.String(min_length=10, max_length=5)
    
    def test_string_min_length_bool_raises_typeerror(self):
        """Test that bool min_length raises TypeError."""
        with pytest.raises(TypeError, match="min_length must be an integer"):
            ar.String(min_length=True)
    
    def test_string_min_length_numpy_bool_raises_typeerror(self):
        """Test that numpy.bool_ raises TypeError."""
        with pytest.raises(TypeError, match="min_length must be an integer"):
            ar.String(min_length=np.bool_(False))
    
    def test_string_min_length_numpy_int_accepted(self):
        """Test that numpy int types are accepted and converted to Python int."""
        field = ar.String(min_length=np.int64(3))
        assert field.min_length == 3
        assert isinstance(field.min_length, int)
    
    def test_string_allowed_bare_string_raises_typeerror(self):
        """Test that allowed as bare string raises TypeError."""
        with pytest.raises(
            TypeError,
            match="allowed must be a sequence of allowed values, not a bare string"
        ):
            ar.String(allowed="abc")
    
    def test_string_allowed_bare_bytes_raises_typeerror(self):
        """Test that allowed as bare bytes raises TypeError."""
        with pytest.raises(
            TypeError,
            match="allowed must be a sequence of allowed values, not a bare string"
        ):
            ar.String(allowed=b"xyz")
    
    def test_string_pattern_not_string_raises_typeerror(self):
        """Test that non-string pattern raises TypeError."""
        with pytest.raises(TypeError, match="pattern must be a string or None"):
            ar.String(pattern=123)
    
    def test_string_pattern_invalid_regex_raises_valueerror(self):
        """Test that invalid regex raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("Invalid regex pattern: '('")):
            ar.String(pattern="(")
    
    def test_string_pattern_unsafe_nested_quantifiers_raises_valueerror(self):
        """Test that nested quantifiers are rejected."""
        with pytest.raises(ValueError, match="Unsafe regex pattern rejected"):
            ar.String(pattern="(a+)+")
    
    def test_string_pattern_unsafe_nested_star_plus_raises_valueerror(self):
        """Test that (a+)* pattern is rejected."""
        with pytest.raises(ValueError, match="Unsafe regex pattern rejected"):
            ar.String(pattern="(a+)*")
    
    def test_string_pattern_safe_fixed_quantifier_accepted(self):
        """Test that fixed-count quantifiers are safe."""
        field = ar.String(pattern=r"(\d{3})+")
        assert field.pattern == r"(\d{3})+"


class TestStringValidation:
    """Test String field behavior with ar.validate."""
    
    def test_string_min_length_validation_raises_issue(self):
        """Test that min_length violation is caught by validate."""
        frame = ar.from_pandas(pd.DataFrame({"s": ["ab", "abcd"]}))
        result = ar.validate(frame, ar.Schema({"s": ar.String(min_length=3)}))
        assert not result.passed
        issues = [i for i in result.issues if i.rule == "min_length"]
        assert len(issues) == 1
        assert issues[0].row_index == 1  # 1-based: first row
        assert issues[0].value == "ab"
    
    def test_string_max_length_validation_raises_issue(self):
        """Test that max_length violation is caught by validate."""
        frame = ar.from_pandas(pd.DataFrame({"s": ["ab", "abcde"]}))
        result = ar.validate(frame, ar.Schema({"s": ar.String(max_length=3)}))
        assert not result.passed
        issues = [i for i in result.issues if i.rule == "max_length"]
        assert len(issues) == 1
        assert issues[0].row_index == 2  # 1-based: second row
        assert issues[0].value == "abcde"
    
    def test_string_pattern_validation_uses_fullmatch(self):
        """Test that pattern uses fullmatch (not partial match)."""
        frame = ar.from_pandas(pd.DataFrame({"s": ["A1", "A1extra"]}))
        result = ar.validate(frame, ar.Schema({"s": ar.String(pattern=r"^[A-Z]\d$")}))
        assert not result.passed
        issues = [i for i in result.issues if i.rule == "pattern"]
        assert len(issues) == 1
        assert issues[0].row_index == 2
    
    def test_string_allowed_validation_raises_issue(self):
        """Test that disallowed values are caught."""
        frame = ar.from_pandas(pd.DataFrame({"s": ["red", "green", "purple"]}))
        result = ar.validate(frame, ar.Schema({"s": ar.String(
            allowed=["red", "green", "blue"]
        )}))
        assert not result.passed
        issues = [i for i in result.issues if i.rule == "allowed"]
        assert len(issues) == 1
        assert issues[0].value == "purple"
    
    def test_string_allowed_empty_rejects_all_values(self):
        """Test that allowed=[] rejects all non-null values."""
        frame = ar.from_pandas(pd.DataFrame({"s": ["a", "b"]}))
        result = ar.validate(frame, ar.Schema({"s": ar.String(allowed=[])}))
        assert not result.passed
        issues = [i for i in result.issues if i.rule == "allowed"]
        assert len(issues) == 2  # Both rows fail
    
    def test_string_case_insensitive_allowed_matching(self):
        """Test case-insensitive allowed value matching."""
        frame = ar.from_pandas(pd.DataFrame({"s": ["RED", "Green", "BLUE"]}))
        result = ar.validate(frame, ar.Schema({"s": ar.String(
            allowed=["red", "green", "blue"],
            case_sensitive=False
        )}))
        assert result.passed
    
    def test_string_unique_validation_flags_duplicates(self):
        """Test that unique=True flags duplicate values."""
        frame = ar.from_pandas(pd.DataFrame({"s": ["a", "b", "a"]}))
        result = ar.validate(frame, ar.Schema({"s": ar.String(unique=True)}))
        assert not result.passed
        issues = [i for i in result.issues if i.rule == "unique"]
        assert any(i.row_index == 3 for i in issues)  # Duplicate at row 3


class TestStringFieldImmutability:
    """Test that returned Field objects are frozen."""
    
    def test_string_field_is_immutable(self):
        """Test that Field returned by String is frozen."""
        field = ar.String(min_length=5)
        with pytest.raises(Exception):  # FrozenInstanceError
            field.min_length = 10
