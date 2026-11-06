import pytest
import arnio as ar


@pytest.fixture
def sample_frame():
    """Sample ArFrame for testing."""
    return ar.from_dict({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "score": [95.5, 87.3, 92.1]
    })


@pytest.fixture
def messy_frame():
    """Messy ArFrame with whitespace and duplicates."""
    return ar.from_dict({
        "name": ["  Alice  ", " Bob ", "  Alice  "],
        "age": [25, 30, 25],
        "value": ["100", "200", "100"]
    })


def test_auto_clean_default_returns_arframe(sample_frame):
    """Default call returns cleaned ArFrame."""
    result = ar.auto_clean(sample_frame)
    assert isinstance(result, ar.ArFrame)


def test_auto_clean_safe_mode(sample_frame):
    """safe mode returns ArFrame."""
    result = ar.auto_clean(sample_frame, mode="safe")
    assert isinstance(result, ar.ArFrame)


def test_auto_clean_strict_mode(sample_frame):
    """strict mode returns ArFrame."""
    result = ar.auto_clean(sample_frame, mode="strict")
    assert isinstance(result, ar.ArFrame)


def test_auto_clean_dry_run_returns_report(sample_frame):
    """dry_run=True returns DataQualityReport."""
    result = ar.auto_clean(sample_frame, dry_run=True)
    assert isinstance(result, ar.DataQualityReport)


def test_auto_clean_return_report_true(sample_frame):
    """return_report=True returns (ArFrame, DataQualityReport)."""
    result = ar.auto_clean(sample_frame, return_report=True)
    assert isinstance(result, tuple) and len(result) == 2
    assert isinstance(result[0], ar.ArFrame)
    assert isinstance(result[1], ar.DataQualityReport)


def test_auto_clean_explain_true(sample_frame):
    """explain=True returns (ArFrame, CleanExplanation)."""
    result = ar.auto_clean(sample_frame, explain=True)
    assert isinstance(result, tuple) and len(result) == 2
    assert isinstance(result[0], ar.ArFrame)
    assert isinstance(result[1], ar.CleanExplanation)


def test_auto_clean_return_report_and_explain(sample_frame):
    """Both return_report and explain returns (ArFrame, DataQualityReport, CleanExplanation)."""
    result = ar.auto_clean(sample_frame, return_report=True, explain=True)
    assert isinstance(result, tuple) and len(result) == 3
    assert isinstance(result[0], ar.ArFrame)
    assert isinstance(result[1], ar.DataQualityReport)
    assert isinstance(result[2], ar.CleanExplanation)


def test_auto_clean_invalid_mode_raises(sample_frame):
    """Invalid mode raises ValueError."""
    with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
        ar.auto_clean(sample_frame, mode="invalid")


def test_auto_clean_invalid_return_report_type_raises(sample_frame):
    """Non-bool return_report raises TypeError."""
    with pytest.raises(TypeError, match="return_report must be a bool"):
        ar.auto_clean(sample_frame, return_report="true")


def test_auto_clean_invalid_dry_run_type_raises(sample_frame):
    """Non-bool dry_run raises TypeError."""
    with pytest.raises(TypeError, match="dry_run must be a bool"):
        ar.auto_clean(sample_frame, dry_run=1)


def test_auto_clean_invalid_explain_type_raises(sample_frame):
    """Non-bool explain raises TypeError."""
    with pytest.raises(TypeError, match="explain must be a bool"):
        ar.auto_clean(sample_frame, explain=1)


def test_auto_clean_dry_run_with_explain_raises(sample_frame):
    """dry_run=True with explain=True raises ValueError."""
    with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
        ar.auto_clean(sample_frame, dry_run=True, explain=True)


def test_auto_clean_dry_run_with_return_report_raises(sample_frame):
    """dry_run=True with return_report=True raises ValueError."""
    with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
        ar.auto_clean(sample_frame, dry_run=True, return_report=True)


def test_auto_clean_explanation_attributes(messy_frame):
    """CleanExplanation contains expected attributes."""
    _, explanation = ar.auto_clean(messy_frame, explain=True)
    assert hasattr(explanation, 'mode')
    assert hasattr(explanation, 'rows_before')
    assert hasattr(explanation, 'rows_after')
    assert hasattr(explanation, 'rows_removed')
    assert hasattr(explanation, 'steps')
    assert explanation.rows_before >= 0
    assert explanation.rows_after >= 0
    assert explanation.rows_removed >= 0
