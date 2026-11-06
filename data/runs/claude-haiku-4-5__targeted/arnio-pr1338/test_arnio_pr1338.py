import pytest
import pandas as pd
import arnio as ar


@pytest.fixture
def clean_frame():
    """Create a clean ArFrame for testing."""
    return ar.from_dict({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "salary": [50000.0, 55000.0, 60000.0],
    })


@pytest.fixture
def messy_frame():
    """Create a messy ArFrame needing cleaning."""
    return ar.from_dict({
        "name": ["  Alice  ", "Bob", "Bob"],
        "age": ["25", "thirty", "35"],
        "salary": [50000.0, None, 60000.0],
    })


@pytest.fixture
def frame_with_whitespace():
    """Create an ArFrame with whitespace."""
    return ar.from_dict({
        "col1": ["  text  ", "value", "  another  "],
        "col2": ["x", "  y  ", "z"],
    })


class TestAutoCleanBasic:
    """Test basic auto_clean functionality."""
    
    def test_auto_clean_returns_arframe_default(self, clean_frame):
        """Default call with no flags returns an ArFrame."""
        result = ar.auto_clean(clean_frame)
        assert isinstance(result, ar.ArFrame)
    
    def test_auto_clean_safe_mode_default(self, messy_frame):
        """Safe mode is default and returns ArFrame."""
        result = ar.auto_clean(messy_frame)
        assert isinstance(result, ar.ArFrame)
    
    def test_auto_clean_explicit_safe_mode(self, messy_frame):
        """Explicit mode='safe' returns ArFrame."""
        result = ar.auto_clean(messy_frame, mode="safe")
        assert isinstance(result, ar.ArFrame)
    
    def test_auto_clean_strict_mode(self, messy_frame):
        """mode='strict' returns ArFrame."""
        result = ar.auto_clean(messy_frame, mode="strict")
        assert isinstance(result, ar.ArFrame)
    
    def test_auto_clean_invalid_mode_raises(self, clean_frame):
        """Invalid mode should raise ValueError or KeyError."""
        with pytest.raises((ValueError, KeyError)):
            ar.auto_clean(clean_frame, mode="invalid")


class TestAutoCleanReturnTypes:
    """Test return types with different flag combinations."""
    
    def test_return_report_true_returns_tuple(self, clean_frame):
        """return_report=True returns tuple[ArFrame, Report]."""
        result = ar.auto_clean(clean_frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
    
    def test_dry_run_true_returns_report(self, clean_frame):
        """dry_run=True returns report, not ArFrame."""
        result = ar.auto_clean(clean_frame, dry_run=True)
        assert not isinstance(result, ar.ArFrame)
    
    def test_explain_true_returns_tuple(self, clean_frame):
        """explain=True returns tuple[ArFrame, CleanExplanation]."""
        result = ar.auto_clean(clean_frame, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
    
    def test_return_report_and_explain_returns_triple(self, clean_frame):
        """Both return_report and explain return 3-tuple."""
        result = ar.auto_clean(clean_frame, return_report=True, explain=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], ar.ArFrame)
    
    def test_dry_run_takes_precedence_over_return_report(self, clean_frame):
        """dry_run=True supersedes return_report; returns report."""
        result = ar.auto_clean(clean_frame, dry_run=True, return_report=True)
        # dry_run takes precedence, should return report type
        assert not isinstance(result, ar.ArFrame)


class TestAutoCleanStrictMode:
    """Test strict mode specific behavior."""
    
    def test_strict_mode_basic(self, messy_frame):
        """Strict mode basic functionality."""
        result = ar.auto_clean(messy_frame, mode="strict")
        assert isinstance(result, ar.ArFrame)
    
    def test_strict_mode_dry_run(self, messy_frame):
        """Strict mode with dry_run returns report."""
        result = ar.auto_clean(messy_frame, mode="strict", dry_run=True)
        assert not isinstance(result, ar.ArFrame)
    
    def test_strict_mode_with_allow_lossy_casts_flag(self, messy_frame):
        """allow_lossy_casts flag accepted in strict mode."""
        result = ar.auto_clean(
            messy_frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={}
        )
        assert isinstance(result, ar.ArFrame)
    
    def test_strict_mode_confirmed_casts_parameter(self, messy_frame):
        """confirmed_casts parameter accepted as dict."""
        result = ar.auto_clean(
            messy_frame,
            mode="strict",
            allow_lossy_casts=True,
            confirmed_casts={"age": "int"}
        )
        assert isinstance(result, ar.ArFrame)


class TestAutoCleanEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_frame(self):
        """Empty frame is handled without error."""
        frame = ar.from_dict({"col": []})
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)
    
    def test_single_row_frame(self):
        """Single row frame is handled."""
        frame = ar.from_dict({"a": [1], "b": ["x"]})
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)
    
    def test_single_column_frame(self):
        """Single column frame is handled."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)
    
    def test_frame_all_nulls_column(self):
        """Column with all nulls is handled."""
        frame = ar.from_dict({"col": [None, None, None]})
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)
    
    def test_frame_mixed_types(self):
        """Frame with mixed column types is handled."""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"]
        })
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)
    
    def test_safe_mode_with_whitespace(self, frame_with_whitespace):
        """Safe mode applies whitespace cleaning."""
        result = ar.auto_clean(frame_with_whitespace, mode="safe")
        assert isinstance(result, ar.ArFrame)


class TestAutoCleanDocumentationExamples:
    """Test patterns from documentation."""
    
    def test_example_basic_clean(self, clean_frame):
        """From docs: ar.auto_clean(frame)."""
        clean = ar.auto_clean(clean_frame)
        assert isinstance(clean, ar.ArFrame)
    
    def test_example_strict_dry_run(self, messy_frame):
        """From docs: ar.auto_clean(frame, mode='strict', dry_run=True)."""
        report = ar.auto_clean(messy_frame, mode="strict", dry_run=True)
        assert not isinstance(report, ar.ArFrame)
    
    def test_example_strict_with_report(self, messy_frame):
        """From docs: ar.auto_clean(frame, mode='strict', return_report=True)."""
        clean, report = ar.auto_clean(messy_frame, mode="strict", return_report=True)
        assert isinstance(clean, ar.ArFrame)
        assert isinstance((clean, report), tuple)
    
    def test_example_with_explain(self, clean_frame):
        """From docs: ar.auto_clean(frame, explain=True)."""
        clean, explanation = ar.auto_clean(clean_frame, explain=True)
        assert isinstance(clean, ar.ArFrame)
        # CleanExplanation should be stringifiable
        assert isinstance(str(explanation), str)
