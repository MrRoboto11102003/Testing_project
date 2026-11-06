import pytest
import arnio as ar


@pytest.fixture
def clean_frame():
    """Create a clean ArFrame for testing."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    return ar.from_dict(data)


@pytest.fixture
def messy_frame():
    """Create a messy ArFrame with whitespace and duplicates."""
    data = {
        "name": [" Alice ", "Bob", " Bob "],
        "age": [25, 30, 30]
    }
    return ar.from_dict(data)


class TestAutoCleanReturnTypes:
    """Test return type behavior for different parameter combinations."""

    def test_default_returns_arframe(self, clean_frame):
        """Default call returns ArFrame."""
        result = ar.auto_clean(clean_frame)
        assert isinstance(result, ar.ArFrame)

    def test_dry_run_returns_report(self, clean_frame):
        """dry_run=True returns report, not ArFrame."""
        result = ar.auto_clean(clean_frame, dry_run=True)
        assert not isinstance(result, ar.ArFrame)

    def test_return_report_returns_tuple(self, clean_frame):
        """return_report=True returns (ArFrame, Report)."""
        result = ar.auto_clean(clean_frame, return_report=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)

    def test_explain_returns_tuple(self, clean_frame):
        """explain=True returns (ArFrame, CleanExplanation)."""
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


class TestAutoCleanModeParameter:
    """Test mode parameter validation and behavior."""

    def test_safe_mode_explicit(self, clean_frame):
        """safe mode can be set explicitly."""
        result = ar.auto_clean(clean_frame, mode="safe")
        assert isinstance(result, ar.ArFrame)

    def test_strict_mode(self, clean_frame):
        """strict mode can be set."""
        result = ar.auto_clean(clean_frame, mode="strict")
        assert isinstance(result, ar.ArFrame)

    def test_invalid_mode_raises_valueerror(self, clean_frame):
        """Invalid mode value raises ValueError."""
        with pytest.raises(ValueError):
            ar.auto_clean(clean_frame, mode="invalid")

    def test_non_string_mode_raises_valueerror(self, clean_frame):
        """Non-string mode raises ValueError."""
        with pytest.raises(ValueError):
            ar.auto_clean(clean_frame, mode=123)


class TestAutoCleanParameterValidation:
    """Test type validation of boolean parameters."""

    def test_return_report_must_be_bool(self, clean_frame):
        """return_report must be bool, not string."""
        with pytest.raises(TypeError):
            ar.auto_clean(clean_frame, return_report="yes")

    def test_dry_run_must_be_bool(self, clean_frame):
        """dry_run must be bool, not int."""
        with pytest.raises(TypeError):
            ar.auto_clean(clean_frame, dry_run=1)

    def test_allow_lossy_casts_must_be_bool(self, clean_frame):
        """allow_lossy_casts must be bool."""
        with pytest.raises(TypeError):
            ar.auto_clean(clean_frame, allow_lossy_casts="yes")

    def test_explain_must_be_bool(self, clean_frame):
        """explain must be bool, not int."""
        with pytest.raises(TypeError):
            ar.auto_clean(clean_frame, explain=1)


class TestAutoCleanConflictingParameters:
    """Test mutually exclusive parameter combinations."""

    def test_dry_run_and_explain_conflict(self, clean_frame):
        """dry_run=True and explain=True raises ValueError."""
        with pytest.raises(ValueError):
            ar.auto_clean(clean_frame, dry_run=True, explain=True)

    def test_dry_run_and_return_report_conflict(self, clean_frame):
        """dry_run=True and return_report=True raises ValueError."""
        with pytest.raises(ValueError):
            ar.auto_clean(clean_frame, dry_run=True, return_report=True)


class TestAutoCleanEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_row_frame(self):
        """auto_clean handles single-row frame."""
        frame = ar.from_dict({"col": ["value"]})
        result = ar.auto_clean(frame)
        assert isinstance(result, ar.ArFrame)

    def test_messy_frame_in_strict_mode(self, messy_frame):
        """strict mode processes frames with duplicates and whitespace."""
        result = ar.auto_clean(messy_frame, mode="strict")
        assert isinstance(result, ar.ArFrame)
