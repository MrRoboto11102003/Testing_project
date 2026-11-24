import pytest
import math
from openlens.lens import Lens
from openlens.analysis.spot_diagram import SpotDiagram
from openlens.tolerancing import InverseSensitivityAnalyzer, Tolerance, ToleranceParamType
from openlens.optical_system import OpticalSystem, OpticalElement


@pytest.fixture
def simple_lens():
    """Create a simple biconvex test lens."""
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
    """Create a simple single-element optical system."""
    system = OpticalSystem(name="Simple System")
    system.add_element(OpticalElement(lens=simple_lens, position=0.0))
    return system


class TestSpotDiagramTraceSpot:
    """Tests for SpotDiagram.trace_spot method."""
    
    def test_trace_spot_returns_dict_with_required_keys(self, simple_system):
        """Test that trace_spot returns dict with all documented keys."""
        spot_diag = SpotDiagram(simple_system)
        result = spot_diag.trace_spot()
        
        required_keys = {
            "rms_radius", "geo_radius", "centroid", 
            "points", "valid_rays", "image_plane_x"
        }
        assert required_keys.issubset(result.keys())
    
    def test_trace_spot_centroid_is_tuple(self, simple_system):
        """Test that centroid is returned as a 2-tuple of floats."""
        spot_diag = SpotDiagram(simple_system)
        result = spot_diag.trace_spot()
        
        centroid = result["centroid"]
        assert isinstance(centroid, tuple)
        assert len(centroid) == 2
        assert all(isinstance(c, (int, float)) for c in centroid)
    
    def test_trace_spot_points_list_structure(self, simple_system):
        """Test that points are returned as list of tuples."""
        spot_diag = SpotDiagram(simple_system)
        result = spot_diag.trace_spot()
        
        assert isinstance(result["points"], list)
        for point in result["points"]:
            assert isinstance(point, tuple)
            assert len(point) == 2
            assert all(isinstance(p, (int, float)) for p in point)
    
    def test_trace_spot_valid_rays_matches_points_count(self, simple_system):
        """Test that valid_rays count equals the number of spot points."""
        spot_diag = SpotDiagram(simple_system)
        result = spot_diag.trace_spot()
        
        assert result["valid_rays"] == len(result["points"])
    
    def test_trace_spot_geo_radius_ge_rms_radius(self, simple_system):
        """Test that geometric radius >= RMS radius when both are defined."""
        spot_diag = SpotDiagram(simple_system)
        result = spot_diag.trace_spot()
        
        if result["rms_radius"] is not None and result["geo_radius"] is not None:
            assert result["geo_radius"] >= result["rms_radius"]
    
    def test_trace_spot_with_field_angle_x(self, simple_system):
        """Test spot tracing with non-zero X field angle."""
        spot_diag = SpotDiagram(simple_system)
        result = spot_diag.trace_spot(field_angle_x_deg=5.0)
        
        assert isinstance(result, dict)
        assert result["valid_rays"] >= 0
        assert "error" not in result or result.get("valid_rays", 0) >= 2
    
    def test_trace_spot_with_field_angle_y(self, simple_system):
        """Test spot tracing with non-zero Y field angle."""
        spot_diag = SpotDiagram(simple_system)
        result = spot_diag.trace_spot(field_angle_y_deg=3.0)
        
        assert isinstance(result, dict)
        assert result["valid_rays"] >= 0
    
    def test_trace_spot_with_custom_image_plane(self, simple_system):
        """Test that custom image_plane_x_mm is used."""
        spot_diag = SpotDiagram(simple_system)
        custom_plane = 120.0
        result = spot_diag.trace_spot(image_plane_x_mm=custom_plane)
        
        assert result["image_plane_x"] == custom_plane
    
    def test_trace_spot_with_focus_shift(self, simple_system):
        """Test that focus_shift_mm offsets the image plane."""
        spot_diag = SpotDiagram(simple_system)
        result_no_shift = spot_diag.trace_spot(image_plane_x_mm=100.0, focus_shift_mm=0.0)
        result_shifted = spot_diag.trace_spot(image_plane_x_mm=100.0, focus_shift_mm=5.0)
        
        expected_plane = result_no_shift["image_plane_x"] + 5.0
        assert result_shifted["image_plane_x"] == pytest.approx(expected_plane)
    
    def test_trace_spot_different_wavelengths(self, simple_system):
        """Test spot tracing at different wavelengths."""
        spot_diag = SpotDiagram(simple_system)
        result_550 = spot_diag.trace_spot(wavelength_nm=550.0)
        result_650 = spot_diag.trace_spot(wavelength_nm=650.0)
        
        # Both should return valid dictionaries with proper structure
        assert isinstance(result_550, dict)
        assert isinstance(result_650, dict)
        assert result_550["valid_rays"] >= 0
        assert result_650["valid_rays"] >= 0
    
    def test_trace_spot_different_sampling_densities(self, simple_system):
        """Test that num_rings affects ray count."""
        spot_diag = SpotDiagram(simple_system)
        result_sparse = spot_diag.trace_spot(num_rings=2)
        result_dense = spot_diag.trace_spot(num_rings=10)
        
        # Higher sampling should produce more or equal rays
        assert result_dense["valid_rays"] >= result_sparse["valid_rays"]


class TestInverseSensitivityAnalyzer:
    """Tests for InverseSensitivityAnalyzer.calculate_sensitivities method."""
    
    @pytest.fixture
    def tolerance_list(self):
        """Create a list of tolerances for testing."""
        return [
            Tolerance(
                param_type=ToleranceParamType.RADIUS_1,
                element_index=0,
                min_val=-0.1,
                max_val=0.1,
            )
        ]
    
    def test_calculate_sensitivities_returns_list(self, simple_system, tolerance_list):
        """Test that calculate_sensitivities returns a list."""
        analyzer = InverseSensitivityAnalyzer(simple_system, tolerance_list)
        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_calculate_sensitivities_dict_structure(self, simple_system, tolerance_list):
        """Test that each result dict has required keys."""
        analyzer = InverseSensitivityAnalyzer(simple_system, tolerance_list)
        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        
        required_keys = {
            "operand_index", "type", "element",
            "test_value", "result_value", "change", "sensitivity"
        }
        
        for item in result:
            assert isinstance(item, dict)
            assert required_keys.issubset(item.keys())
    
    def test_calculate_sensitivities_unsupported_criterion_raises_error(
        self, simple_system, tolerance_list
    ):
        """Test that ValueError is raised for unsupported criterion."""
        analyzer = InverseSensitivityAnalyzer(simple_system, tolerance_list)
        
        with pytest.raises(ValueError, match="Unsupported criterion"):
            analyzer.calculate_sensitivities(criterion="unsupported_metric")
    
    def test_calculate_sensitivities_sensitivity_calculation(
        self, simple_system, tolerance_list
    ):
        """Test that sensitivity = change / test_value."""
        analyzer = InverseSensitivityAnalyzer(simple_system, tolerance_list)
        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        
        for item in result:
            test_val = item["test_value"]
            change = item["change"]
            sensitivity = item["sensitivity"]
            
            # When values are finite and test_val is non-zero
            if test_val != 0 and math.isfinite(change) and math.isfinite(sensitivity):
                expected = change / test_val
                assert sensitivity == pytest.approx(expected, rel_tol=1e-5)
    
    def test_calculate_sensitivities_multiple_tolerances(self, simple_system):
        """Test with multiple tolerances."""
        tolerances = [
            Tolerance(
                param_type=ToleranceParamType.RADIUS_1,
                element_index=0,
                min_val=-0.1,
                max_val=0.1,
            ),
            Tolerance(
                param_type=ToleranceParamType.THICKNESS,
                element_index=0,
                min_val=-0.05,
                max_val=0.05,
            ),
        ]
        analyzer = InverseSensitivityAnalyzer(simple_system, tolerances)
        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        
        # Should have one result per tolerance
        assert len(result) == len(tolerances)
    
    def test_calculate_sensitivities_system_state_restored(self, simple_system, tolerance_list):
        """Test that system state is restored after sensitivity analysis."""
        original_r1 = simple_system.elements[0].lens.radius_of_curvature_1
        
        analyzer = InverseSensitivityAnalyzer(simple_system, tolerance_list)
        analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        
        # System state should be unchanged
        assert simple_system.elements[0].lens.radius_of_curvature_1 == original_r1
