import pytest
import pandas as pd
import arnio as ar


def test_auto_clean_default_returns_arframe():
    """Test that auto_clean returns an ArFrame by default."""
    data = {"name": [" alice ", " bob "], "age": [25, 30]}
    frame = ar.from_pandas(pd.DataFrame(data))
    result = ar.auto_clean(frame)
    assert isinstance(result, ar.ArFrame)


def test_auto_clean_safe_mode_strips_whitespace():
    """Test that safe mode applies strip_whitespace."""
    data = {"name": [" alice ", "  bob  "], "city": [" paris ", " london "]}
    frame = ar.from_pandas(pd.DataFrame(data))
    result = ar.auto_clean(frame, mode="safe")
    df = ar.to_pandas(result)
    assert df["name"].iloc[0] == "alice"
    assert df["name"].iloc[1] == "bob"
    assert df["city"].iloc[0] == "paris"


def test_auto_clean_invalid_mode_raises_value_error():
    """Test that invalid mode raises ValueError."""
    data = {"name": ["alice"]}
    frame = ar.from_pandas(pd.DataFrame(data))
    with pytest.raises(ValueError, match="mode must be 'safe' or 'strict'"):
        ar.auto_clean(frame, mode="invalid")


def test_auto_clean_invalid_return_report_type():
    """Test that non-bool return_report raises TypeError."""
    data = {"name": ["alice"]}
    frame = ar.from_pandas(pd.DataFrame(data))
    with pytest.raises(TypeError, match="return_report must be a bool"):
        ar.auto_clean(frame, return_report="yes")


def test_auto_clean_invalid_dry_run_type():
    """Test that non-bool dry_run raises TypeError."""
    data = {"name": ["alice"]}
    frame = ar.from_pandas(pd.DataFrame(data))
    with pytest.raises(TypeError, match="dry_run must be a bool"):
        ar.auto_clean(frame, dry_run="yes")


def test_auto_clean_invalid_explain_type():
    """Test that non-bool explain raises TypeError."""
    data = {"name": ["alice"]}
    frame = ar.from_pandas(pd.DataFrame(data))
    with pytest.raises(TypeError, match="explain must be a bool"):
        ar.auto_clean(frame, explain="yes")


def test_auto_clean_dry_run_returns_report():
    """Test that dry_run=True returns DataQualityReport without modifying frame."""
    data = {"name": [" alice "]}
    frame = ar.from_pandas(pd.DataFrame(data))
    original_shape = frame.shape
    result = ar.auto_clean(frame, dry_run=True)
    assert isinstance(result, ar.DataQualityReport)
    assert frame.shape == original_shape


def test_auto_clean_return_report_returns_tuple():
    """Test that return_report=True returns (ArFrame, DataQualityReport)."""
    data = {"name": [" alice "]}
    frame = ar.from_pandas(pd.DataFrame(data))
    result = ar.auto_clean(frame, return_report=True)
    assert isinstance(result, tuple) and len(result) == 2
    assert isinstance(result[0], ar.ArFrame)
    assert isinstance(result[1], ar.DataQualityReport)


def test_auto_clean_explain_returns_tuple():
    """Test that explain=True returns (ArFrame, CleanExplanation)."""
    data = {"name": [" alice "]}
    frame = ar.from_pandas(pd.DataFrame(data))
    result = ar.auto_clean(frame, explain=True)
    assert isinstance(result, tuple) and len(result) == 2
    assert isinstance(result[0], ar.ArFrame)
    assert hasattr(result[1], 'mode')


def test_auto_clean_explain_and_return_report_returns_triple():
    """Test that explain=True and return_report=True returns (ArFrame, Report, Explanation)."""
    data = {"name": [" alice "]}
    frame = ar.from_pandas(pd.DataFrame(data))
    result = ar.auto_clean(frame, explain=True, return_report=True)
    assert isinstance(result, tuple) and len(result) == 3
    assert isinstance(result[0], ar.ArFrame)
    assert isinstance(result[1], ar.DataQualityReport)
    assert hasattr(result[2], 'mode')


def test_auto_clean_dry_run_and_explain_raises_error():
    """Test that dry_run=True and explain=True cannot be used together."""
    data = {"name": ["alice"]}
    frame = ar.from_pandas(pd.DataFrame(data))
    with pytest.raises(ValueError, match="explain=True cannot be used with dry_run=True"):
        ar.auto_clean(frame, dry_run=True, explain=True)


def test_auto_clean_dry_run_and_return_report_raises_error():
    """Test that dry_run=True and return_report=True cannot be used together."""
    data = {"name": ["alice"]}
    frame = ar.from_pandas(pd.DataFrame(data))
    with pytest.raises(ValueError, match="return_report=True cannot be used with dry_run=True"):
        ar.auto_clean(frame, dry_run=True, return_report=True)


def test_auto_clean_strict_mode_applies_cleaning():
    """Test that strict mode applies cleaning steps including strip_whitespace."""
    data = {"name": [" alice ", "  bob  "]}
    frame = ar.from_pandas(pd.DataFrame(data))
    result = ar.auto_clean(frame, mode="strict")
    assert isinstance(result, ar.ArFrame)
    df = ar.to_pandas(result)
    assert df["name"].iloc[0] == "alice"
    assert df["name"].iloc[1] == "bob"


def test_auto_clean_strict_mode_without_allow_lossy_casts_raises_on_type_casts():
    """Test that strict mode raises error if type casts are needed but not confirmed."""
    # Create data where type casts might be suggested
    data = {"id": ["1", "2", "3"], "value": ["10.5", "20.3", "15.1"]}
    frame = ar.from_pandas(pd.DataFrame(data))
    
    # Check if strict mode would propose casts
    report = ar.auto_clean(frame, mode="strict", dry_run=True)
    has_casts = any(
        (isinstance(s, tuple) and s[0] == "cast_types")
        or (hasattr(s, 'step') and s.step == "cast_types")
        for s in report.suggestions
    )
    
    if has_casts:
        # Should raise because allow_lossy_casts is False
        with pytest.raises(ValueError, match="allow_lossy_casts"):
            ar.auto_clean(frame, mode="strict")


def test_auto_clean_strict_mode_confirmed_casts_must_match():
    """Test that confirmed_casts mapping must exactly match the proposed mapping."""
    data = {"id": ["1", "2"], "value": ["10.5", "20.3"]}
    frame = ar.from_pandas(pd.DataFrame(data))
    
    # Get the proposed mapping
    report = ar.auto_clean(frame, mode="strict", dry_run=True)
    proposed = None
    for s in report.suggestions:
        if isinstance(s, tuple) and s[0] == "cast_types":
            proposed = s[1]
            break
        elif hasattr(s, 'step') and s.step == "cast_types":
            proposed = s.kwargs
            break
    
    if proposed:
        # Try with wrong mapping
        wrong_mapping = {"id": "float64"} if proposed != {"id": "float64"} else {"id": "int64"}
        with pytest.raises(ValueError, match="confirmed_casts must match"):
            ar.auto_clean(
                frame,
                mode="strict",
                allow_lossy_casts=True,
                confirmed_casts=wrong_mapping
            )
