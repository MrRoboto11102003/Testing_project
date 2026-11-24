import pytest
from unittest.mock import Mock
from openlens.performance_metrics import PerformanceMetrics


class TestPerformanceMetricsCalculateBackFocalLength:
    """Test suite for PerformanceMetrics.calculate_back_focal_length()"""

    @pytest.fixture
    def mock_lens(self):
        """Create a mock Lens object"""
        lens = Mock()
        lens.calculate_back_focal_length = Mock()
        return lens

    @pytest.fixture
    def mock_system(self):
        """Create a mock OpticalSystem object"""
        system = Mock()
        system.calculate_back_focal_length = Mock()
        return system

    def test_returns_none_when_no_lens_or_system(self):
        """Test returns None when neither lens nor system is set"""
        metrics = PerformanceMetrics(lens=None, system=None)
        result = metrics.calculate_back_focal_length()
        assert result is None

    def test_delegates_to_lens_when_lens_exists(self, mock_lens):
        """Test that lens.calculate_back_focal_length() is called when lens exists"""
        mock_lens.calculate_back_focal_length.return_value = 42.5
        metrics = PerformanceMetrics(lens=mock_lens, system=None)

        result = metrics.calculate_back_focal_length()

        assert result == 42.5
        mock_lens.calculate_back_focal_length.assert_called_once()

    def test_delegates_to_system_when_system_exists(self, mock_system):
        """Test that system.calculate_back_focal_length() is called when system exists"""
        mock_system.calculate_back_focal_length.return_value = 37.5
        metrics = PerformanceMetrics(lens=None, system=mock_system)

        result = metrics.calculate_back_focal_length()

        assert result == 37.5
        mock_system.calculate_back_focal_length.assert_called_once()

    def test_lens_takes_precedence_over_system(self, mock_lens, mock_system):
        """Test that lens is prioritized when both lens and system are set"""
        mock_lens.calculate_back_focal_length.return_value = 42.5
        mock_system.calculate_back_focal_length.return_value = 37.5
        metrics = PerformanceMetrics(lens=mock_lens, system=mock_system)

        result = metrics.calculate_back_focal_length()

        assert result == 42.5
        mock_lens.calculate_back_focal_length.assert_called_once()
        mock_system.calculate_back_focal_length.assert_not_called()

    def test_returns_none_when_lens_returns_none(self, mock_lens):
        """Test returns None when lens.calculate_back_focal_length() returns None"""
        mock_lens.calculate_back_focal_length.return_value = None
        metrics = PerformanceMetrics(lens=mock_lens, system=None)

        result = metrics.calculate_back_focal_length()

        assert result is None

    def test_returns_none_when_system_returns_none(self, mock_system):
        """Test returns None when system.calculate_back_focal_length() returns None"""
        mock_system.calculate_back_focal_length.return_value = None
        metrics = PerformanceMetrics(lens=None, system=mock_system)

        result = metrics.calculate_back_focal_length()

        assert result is None

    def test_positive_bfl_for_converging_lens(self, mock_lens):
        """Test positive BFL value for converging (positive) lens"""
        mock_lens.calculate_back_focal_length.return_value = 50.0
        metrics = PerformanceMetrics(lens=mock_lens)

        result = metrics.calculate_back_focal_length()

        assert result == 50.0
        assert result > 0

    def test_negative_bfl_for_diverging_lens(self, mock_lens):
        """Test negative BFL value for diverging (negative) lens"""
        mock_lens.calculate_back_focal_length.return_value = -50.0
        metrics = PerformanceMetrics(lens=mock_lens)

        result = metrics.calculate_back_focal_length()

        assert result == -50.0
        assert result < 0

    def test_zero_bfl_for_afocal_lens(self, mock_lens):
        """Test zero BFL for afocal lens (infinite focal length)"""
        mock_lens.calculate_back_focal_length.return_value = 0.0
        metrics = PerformanceMetrics(lens=mock_lens)

        result = metrics.calculate_back_focal_length()

        assert result == 0.0

    def test_very_small_bfl_value(self, mock_lens):
        """Test with very small BFL value"""
        mock_lens.calculate_back_focal_length.return_value = 1e-6
        metrics = PerformanceMetrics(lens=mock_lens)

        result = metrics.calculate_back_focal_length()

        assert result == pytest.approx(1e-6, abs=1e-9)

    def test_very_large_bfl_value(self, mock_lens):
        """Test with very large BFL value"""
        mock_lens.calculate_back_focal_length.return_value = 1e6
        metrics = PerformanceMetrics(lens=mock_lens)

        result = metrics.calculate_back_focal_length()

        assert result == 1e6

    def test_fractional_bfl_value(self, mock_lens):
        """Test with fractional BFL value"""
        mock_lens.calculate_back_focal_length.return_value = 42.75
        metrics = PerformanceMetrics(lens=mock_lens)

        result = metrics.calculate_back_focal_length()

        assert result == pytest.approx(42.75)

    def test_multiple_invocations_call_delegate_each_time(self, mock_lens):
        """Test that multiple invocations call the delegate each time (no caching)"""
        mock_lens.calculate_back_focal_length.return_value = 42.5
        metrics = PerformanceMetrics(lens=mock_lens)

        result1 = metrics.calculate_back_focal_length()
        result2 = metrics.calculate_back_focal_length()

        assert result1 == result2 == 42.5
        assert mock_lens.calculate_back_focal_length.call_count == 2

    def test_system_bfl_for_multi_element_assembly(self, mock_system):
        """Test BFL calculation for multi-element optical system"""
        mock_system.calculate_back_focal_length.return_value = 37.5
        metrics = PerformanceMetrics(lens=None, system=mock_system)

        result = metrics.calculate_back_focal_length()

        assert result == 37.5
        mock_system.calculate_back_focal_length.assert_called_once()
