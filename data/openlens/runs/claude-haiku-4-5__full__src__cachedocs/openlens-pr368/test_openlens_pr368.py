import pytest
import math
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.spot_diagram import SpotDiagram
from openlens.tolerancing import InverseSensitivityAnalyzer, Tolerance, ParamType
from openlens.constants import WAVELENGTH_GREEN


@pytest.fixture
def simple_lens():
    """Create a basic biconvex lens for testing."""
    return Lens(
        name="Test Biconvex",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7",
    )


@pytest.fixture
def simple_system(simple_lens):
    """Create a simple optical system with one lens."""
    system = OpticalSystem(name="Test System")
    system.add_lens(simple_lens)
    return system


@pytest.fixture
def spot_diagram(simple_system):
    """Create a SpotDiagram instance."""
    return SpotDiagram(simple_system)


class TestSpotDiagramTraceSpot:
    """Test SpotDiagram.trace_spot method."""

    def test_trace_spot_on_axis(self, spot_diagram):
        """Test tracing rays on-axis with default parameters."""
        result = spot_diagram.trace_spot(
            field_angle_x_deg=0.0,
            field_angle_y_deg=0.0,
            num_rings=4,
        )
        
        assert isinstance(result, dict)
        assert "rms_radius" in result
        assert "geo_radius" in result
        assert "centroid" in result
        assert "points" in result
        assert "valid_rays" in result
        assert "image_plane_x" in result

    def test_trace_spot_returns_dict_structure(self, spot_diagram):
        """Verify return dict has all documented keys."""
        result = spot_diagram.trace_spot()
        
        required_keys = {"rms_radius", "geo_radius", "centroid", "points", 
                        "valid_rays", "image_plane_x"}
        assert required_keys.issubset(result.keys())

    def test_trace_spot_centroid_is_tuple(self, spot_diagram):
        """Verify centroid is a tuple of floats."""
        result = spot_diagram.trace_spot(num_rings=3)
        
        if result["valid_rays"] >= 2:
            centroid = result["centroid"]
            assert isinstance(centroid, tuple)
            assert len(centroid) == 2
            assert isinstance(centroid[0], float)
            assert isinstance(centroid[1], float)

    def test_trace_spot_with_field_angle(self, spot_diagram):
        """Test tracing with non-zero field angles."""
        result_on_axis = spot_diagram.trace_spot(
            field_angle_x_deg=0.0,
            field_angle_y_deg=0.0,
            num_rings=4,
        )
        result_off_axis = spot_diagram.trace_spot(
            field_angle_x_deg=5.0,
            field_angle_y_deg=5.0,
            num_rings=4,
        )
        
        # Both should return valid results
        assert result_on_axis["valid_rays"] > 0
        assert result_off_axis["valid_rays"] > 0

    def test_trace_spot_with_wavelength(self, spot_diagram):
        """Test tracing at different wavelengths."""
        result_green = spot_diagram.trace_spot(
            wavelength_nm=WAVELENGTH_GREEN * 1e9,
            num_rings=4,
        )
        result_red = spot_diagram.trace_spot(
            wavelength_nm=650.0,
            num_rings=4,
        )
        
        # Both should return valid structure
        assert "rms_radius" in result_green
        assert "rms_radius" in result_red

    def test_trace_spot_with_defocus(self, spot_diagram):
        """Test tracing with focus shift (defocus)."""
        result_nominal = spot_diagram.trace_spot(focus_shift_mm=0.0, num_rings=4)
        result_defocus = spot_diagram.trace_spot(focus_shift_mm=1.0, num_rings=4)
        
        # Defocused result should have different image plane
        assert result_defocus["image_plane_x"] > result_nominal["image_plane_x"]

    def test_trace_spot_lens_state_restoration(self, spot_diagram, simple_lens):
        """Test that lens wavelength/index is restored after trace."""
        original_wavelength = simple_lens.wavelength
        original_index = simple_lens.refractive_index
        
        spot_diagram.trace_spot(wavelength_nm=650.0, num_rings=4)
        
        # Lens state should be restored
        assert simple_lens.wavelength == original_wavelength
        assert simple_lens.refractive_index == original_index

    def test_trace_spot_invalid_rays_returns_none_radii(self, spot_diagram):
        """Test that vignetted system returns None for radii."""
        # Create a highly vignetted condition - very small rays with large field angle
        result = spot_diagram.trace_spot(
            field_angle_x_deg=90.0,  # Extreme angle
            field_angle_y_deg=90.0,
            num_rings=1,
        )
        
        # If rays are vignetted (< 2 valid), rms_radius should be None
        if result["valid_rays"] < 2:
            assert result["rms_radius"] is None
            assert result["geo_radius"] is None
            assert "error" in result

    def test_trace_spot_points_list(self, spot_diagram):
        """Test that spot points are returned as list."""
        result = spot_diagram.trace_spot(num_rings=3)
        
        points = result["points"]
        assert isinstance(points, list)
        for point in points:
            assert isinstance(point, tuple)
            assert len(point) == 2  # (y, z)

    def test_trace_spot_explicit_image_plane(self, spot_diagram):
        """Test tracing with explicit image plane position."""
        custom_plane = 100.0
        result = spot_diagram.trace_spot(
            image_plane_x_mm=custom_plane,
            num_rings=4,
        )
        
        # Image plane should match requested (plus focus_shift if any)
        assert result["image_plane_x"] == custom_plane

    def test_trace_spot_num_rings_affects_ray_count(self, spot_diagram):
        """Test that num_rings parameter affects number of rays."""
        result_sparse = spot_diagram.trace_spot(num_rings=2)
        result_dense = spot_diagram.trace_spot(num_rings=6)
        
        # More rings should generally produce more points (if not all vignetted)
        assert len(result_dense["points"]) >= len(result_sparse["points"])


class TestInverseSensitivityAnalyzer:
    """Test InverseSensitivityAnalyzer.calculate_sensitivities method."""

    @pytest.fixture
    def analyzer(self, simple_system):
        """Create an analyzer with mock tolerances."""
        analyzer = InverseSensitivityAnalyzer(simple_system)
        
        # Create simple mock tolerance objects
        tol1 = Mock()
        tol1.max_val = 0.1
        tol1.param_type = ParamType.RADIUS
        tol1.element_index = 0
        
        tol2 = Mock()
        tol2.max_val = 0.05
        tol2.param_type = ParamType.THICKNESS
        tol2.element_index = 0
        
        analyzer.tolerances = [tol1, tol2]
        
        # Mock the internal methods to avoid complex setup
        analyzer._get_system_state = Mock(return_value={})
        analyzer._set_system_state = Mock()
        analyzer._apply_single_tolerance = Mock()
        
        return analyzer

    def test_calculate_sensitivities_valid_criterion(self, analyzer):
        """Test with valid criterion 'rms_spot_radius'."""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            mock_instance = MagicMock()
            mock_instance.trace_spot.return_value = {
                "rms_radius": 0.5,
                "valid_rays": 10,
            }
            mock_spot.return_value = mock_instance
            
            results = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
            
            assert isinstance(results, list)
            assert len(results) == 2  # Two tolerances

    def test_calculate_sensitivities_invalid_criterion_raises(self, analyzer):
        """Test that invalid criterion raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            analyzer.calculate_sensitivities(criterion="invalid_metric")
        
        assert "Unsupported criterion" in str(exc_info.value)

    def test_calculate_sensitivities_result_structure(self, analyzer):
        """Test that results have correct structure."""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            mock_instance = MagicMock()
            mock_instance.trace_spot.return_value = {
                "rms_radius": 0.5,
                "valid_rays": 10,
            }
            mock_spot.return_value = mock_instance
            
            results = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
            
            for result in results:
                assert isinstance(result, dict)
                required_keys = {
                    "operand_index",
                    "type",
                    "element",
                    "test_value",
                    "result_value",
                    "change",
                    "sensitivity",
                }
                assert required_keys.issubset(result.keys())

    def test_calculate_sensitivities_handles_vignetted_nominal(self, analyzer):
        """Test sensitivity calculation when nominal is vignetted (None rms_radius)."""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            nominal_result = {"rms_radius": None, "valid_rays": 0}
            perturbed_result = {"rms_radius": 0.5, "valid_rays": 10}
            
            mock_instance = MagicMock()
            mock_instance.trace_spot.side_effect = [nominal_result, perturbed_result]
            mock_spot.return_value = mock_instance
            
            results = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
            
            # First result should handle inf nominal
            first_result = results[0]
            assert isinstance(first_result["sensitivity"], float)

    def test_calculate_sensitivities_state_restoration(self, analyzer):
        """Test that system state is restored after each tolerance test."""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            mock_instance = MagicMock()
            mock_instance.trace_spot.return_value = {"rms_radius": 0.5}
            mock_spot.return_value = mock_instance
            
            analyzer.calculate_sensitivities(criterion="rms_spot_radius")
            
            # _set_system_state should be called to restore state
            assert analyzer._set_system_state.called

    def test_calculate_sensitivities_sensitivity_calculation(self, analyzer):
        """Test sensitivity is calculated as change/delta."""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            nominal_result = {"rms_radius": 1.0}
            perturbed_result = {"rms_radius": 1.2}
            
            mock_instance = MagicMock()
            mock_instance.trace_spot.side_effect = [
                nominal_result,
                perturbed_result,
                perturbed_result,
            ]
            mock_spot.return_value = mock_instance
            
            analyzer.tolerances[0].max_val = 0.1
            results = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
            
            first_result = results[0]
            # sensitivity = change / delta = (1.2 - 1.0) / 0.1 = 2.0
            expected_sensitivity = 0.2 / 0.1
            assert abs(first_result["sensitivity"] - expected_sensitivity) < 1e-6

    def test_calculate_sensitivities_handles_exception_in_trace(self, analyzer):
        """Test that exceptions during trace are caught and handled."""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            nominal_result = {"rms_radius": 1.0}
            mock_instance = MagicMock()
            mock_instance.trace_spot.side_effect = [
                nominal_result,
                RuntimeError("Trace failed"),
            ]
            mock_spot.return_value = mock_instance
            
            results = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
            
            # Should return results without raising exception
            assert len(results) > 0
            # Failed trace should result in inf value
            assert any(math.isinf(r["result_value"]) for r in results)

    def test_calculate_sensitivities_zero_delta_handled(self, analyzer):
        """Test sensitivity calculation when delta is zero."""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            mock_instance = MagicMock()
            mock_instance.trace_spot.return_value = {"rms_radius": 1.0}
            mock_spot.return_value = mock_instance
            
            analyzer.tolerances[0].max_val = 0.0
            results = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
            
            first_result = results[0]
            # Division by zero should be handled gracefully
            assert isinstance(first_result["sensitivity"], float)

    def test_calculate_sensitivities_all_tolerances_tested(self, analyzer):
        """Test that all tolerances are included in results."""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            mock_instance = MagicMock()
            mock_instance.trace_spot.return_value = {"rms_radius": 0.5}
            mock_spot.return_value = mock_instance
            
            num_tolerances = len(analyzer.tolerances)
            results = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
            
            # Should have result for each tolerance plus nominal
            assert len(results) == num_tolerances
            # Operand indices should cover all tolerances
            indices = [r["operand_index"] for r in results]
            assert set(indices) == set(range(num_tolerances))

    def test_calculate_sensitivities_nan_for_undefined_sensitivity(self, analyzer):
        """Test that undefined sensitivities return NaN."""
        with patch('openlens.tolerancing.SpotDiagram') as mock_spot:
            # Both nominal and perturbed vignetted -> inf - inf = undefined
            mock_instance = MagicMock()
            mock_instance.trace_spot.side_effect = [
                {"rms_radius": None},  # nominal
                {"rms_radius": None},  # perturbed
            ]
            mock_spot.return_value = mock_instance
            
            results = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
            
            first_result = results[0]
            # Should have NaN when both are vignetted
            assert math.isnan(first_result["sensitivity"]) or math.isnan(first_result["change"])
