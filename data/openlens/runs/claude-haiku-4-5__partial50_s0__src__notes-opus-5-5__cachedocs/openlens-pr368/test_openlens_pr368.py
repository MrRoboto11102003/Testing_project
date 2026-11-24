import pytest
import math
from unittest.mock import patch
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.spot_diagram import SpotDiagram
from openlens.tolerancing import InverseSensitivityAnalyzer, ToleranceOperand, ToleranceType


# ============= FIXTURES =============

@pytest.fixture
def simple_lens():
    """Create a simple biconvex lens for testing."""
    return Lens(
        name="test_lens",
        radius_of_curvature_1=50.0,
        radius_of_curvature_2=-50.0,
        thickness=5.0,
        diameter=25.0,
        material="BK7",
    )


@pytest.fixture
def simple_system(simple_lens):
    """Create a simple optical system with one lens."""
    system = OpticalSystem()
    system.add_lens(simple_lens)
    return system


@pytest.fixture
def spot_diagram(simple_system):
    """Create a SpotDiagram for the simple system."""
    return SpotDiagram(simple_system)


# ============= SpotDiagram.trace_spot TESTS =============

class TestSpotDiagramTraceSpot:
    """Test suite for SpotDiagram.trace_spot"""

    def test_basic_trace_returns_valid_result_dict(self, spot_diagram, simple_system):
        """Test basic on-axis ray tracing returns valid result with 6 keys."""
        result = spot_diagram.trace_spot(
            field_angle_x_deg=0.0,
            field_angle_y_deg=0.0,
            image_plane_x_mm=simple_system.elements[0].position + 100.0,
            num_rings=2,
        )
        
        # Check success structure: exactly 6 keys, no "error" key
        assert len(result) == 6
        assert set(result.keys()) == {
            "rms_radius", "geo_radius", "centroid", "points",
            "valid_rays", "image_plane_x"
        }
        assert result["rms_radius"] is not None
        assert result["geo_radius"] is not None
        assert isinstance(result["centroid"], tuple)
        assert len(result["centroid"]) == 2
        assert result["valid_rays"] > 0

    def test_empty_system_raises_indexerror(self):
        """Empty system should raise IndexError."""
        empty_system = OpticalSystem()
        spot_diagram = SpotDiagram(empty_system)
        
        with pytest.raises(IndexError):
            spot_diagram.trace_spot(image_plane_x_mm=100.0)

    def test_num_rings_zero_returns_failure_dict(self, spot_diagram, simple_system):
        """num_rings=0 produces fewer than 2 rays, returning 7-key failure dict."""
        result = spot_diagram.trace_spot(
            num_rings=0,
            image_plane_x_mm=simple_system.elements[0].position + 100.0,
        )
        
        # Check failure structure: exactly 7 keys with "error"
        assert len(result) == 7
        assert "error" in result
        assert result["rms_radius"] is None
        assert result["geo_radius"] is None
        assert result["centroid"] == (0.0, 0.0)
        assert result["valid_rays"] < 2
        assert "Fewer than 2 rays" in result["error"]

    def test_on_axis_centroid_near_origin(self, spot_diagram, simple_system):
        """On-axis rays should have centroid very close to origin."""
        result = spot_diagram.trace_spot(
            field_angle_x_deg=0.0,
            field_angle_y_deg=0.0,
            image_plane_x_mm=simple_system.elements[0].position + 100.0,
            num_rings=3,
        )
        
        cent_y, cent_z = result["centroid"]
        assert abs(cent_y) < 1e-6
        assert abs(cent_z) < 1e-6

    def test_field_angle_y_affects_centroid_y(self, spot_diagram, simple_system):
        """Field angle in Y should move centroid in Y direction."""
        result_on_axis = spot_diagram.trace_spot(
            field_angle_x_deg=0.0,
            field_angle_y_deg=0.0,
            image_plane_x_mm=simple_system.elements[0].position + 100.0,
            num_rings=2,
        )
        
        result_off_axis = spot_diagram.trace_spot(
            field_angle_x_deg=0.0,
            field_angle_y_deg=3.0,
            image_plane_x_mm=simple_system.elements[0].position + 100.0,
            num_rings=2,
        )
        
        on_axis_y = result_on_axis["centroid"][0]
        off_axis_y = result_off_axis["centroid"][0]
        
        # Off-axis Y field should shift Y centroid significantly
        assert abs(off_axis_y - on_axis_y) > 0.1

    def test_field_angle_x_affects_centroid_z(self, spot_diagram, simple_system):
        """Field angle in X should move centroid in Z direction."""
        result_on_axis = spot_diagram.trace_spot(
            field_angle_x_deg=0.0,
            field_angle_y_deg=0.0,
            image_plane_x_mm=simple_system.elements[0].position + 100.0,
            num_rings=2,
        )
        
        result_off_axis = spot_diagram.trace_spot(
            field_angle_x_deg=3.0,
            field_angle_y_deg=0.0,
            image_plane_x_mm=simple_system.elements[0].position + 100.0,
            num_rings=2,
        )
        
        on_axis_z = result_on_axis["centroid"][1]
        off_axis_z = result_off_axis["centroid"][1]
        
        # Off-axis X field should shift Z centroid significantly
        assert abs(off_axis_z - on_axis_z) > 0.1

    def test_explicit_image_plane_x_applied(self, spot_diagram, simple_system):
        """Explicit image_plane_x_mm should be recorded in result."""
        plane_x = simple_system.elements[0].position + 50.0
        result = spot_diagram.trace_spot(
            image_plane_x_mm=plane_x,
            num_rings=2,
        )
        
        assert result["image_plane_x"] == plane_x

    def test_focus_shift_applied_to_image_plane(self, spot_diagram, simple_system):
        """focus_shift_mm should offset the image plane in result."""
        plane_x = simple_system.elements[0].position + 50.0
        shift = 5.0
        result = spot_diagram.trace_spot(
            image_plane_x_mm=plane_x,
            focus_shift_mm=shift,
            num_rings=2,
        )
        
        assert result["image_plane_x"] == pytest.approx(plane_x + shift)

    def test_rms_radius_less_or_equal_geo_radius(self, spot_diagram, simple_system):
        """RMS radius must be <= geometric (max) radius."""
        result = spot_diagram.trace_spot(
            image_plane_x_mm=simple_system.elements[0].position + 100.0,
            num_rings=3,
        )
        
        if result["rms_radius"] is not None and result["geo_radius"] is not None:
            assert result["rms_radius"] <= result["geo_radius"] + 1e-9

    def test_lens_state_restored_after_trace(self, simple_system):
        """Lens wavelength and refractive_index must be restored after trace."""
        original_wavelength = simple_system.elements[0].lens.wavelength
        original_index = simple_system.elements[0].lens.refractive_index
        
        spot_diagram = SpotDiagram(simple_system)
        result = spot_diagram.trace_spot(
            wavelength_nm=486.1,
            image_plane_x_mm=simple_system.elements[0].position + 100.0,
        )
        
        assert simple_system.elements[0].lens.wavelength == original_wavelength
        assert simple_system.elements[0].lens.refractive_index == pytest.approx(original_index)

    def test_failure_case_logs_warning(self, spot_diagram, simple_system, caplog):
        """Failure (< 2 rays) should log WARNING with specific message."""
        import logging
        caplog.set_level(logging.WARNING)
        
        result = spot_diagram.trace_spot(
            num_rings=0,
            image_plane_x_mm=simple_system.elements[0].position + 100.0,
        )
        
        assert "Only" in caplog.text
        assert "ray(s) reached the image plane" in caplog.text


# ============= InverseSensitivityAnalyzer.calculate_sensitivities TESTS =============

class TestInverseSensitivityAnalyzer:
    """Test suite for InverseSensitivityAnalyzer.calculate_sensitivities"""

    def test_unsupported_criterion_raises_valueerror_first(self, simple_system):
        """Unsupported criterion must raise ValueError before any operations."""
        tol = ToleranceOperand(
            element_index=0,
            param_type=ToleranceType.RADIUS_1,
            min_val=-0.1,
            max_val=0.1,
        )
        analyzer = InverseSensitivityAnalyzer(simple_system, [tol])
        
        with patch('openlens.tolerancing.SpotDiagram') as mock_sd:
            with pytest.raises(ValueError, match="Unsupported criterion"):
                analyzer.calculate_sensitivities(criterion="mtf")
            
            # SpotDiagram should NOT be instantiated when criterion check fails
            mock_sd.assert_not_called()

    def test_empty_tolerances_returns_empty_list(self, simple_system):
        """Empty tolerances list should return empty list."""
        analyzer = InverseSensitivityAnalyzer(simple_system, [])
        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        
        assert result == []
        assert isinstance(result, list)

    def test_result_dict_has_exactly_seven_keys(self, simple_system):
        """Each result entry must have exactly 7 keys."""
        tol = ToleranceOperand(
            element_index=0,
            param_type=ToleranceType.RADIUS_1,
            min_val=-0.1,
            max_val=0.1,
        )
        analyzer = InverseSensitivityAnalyzer(simple_system, [tol])
        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        
        assert len(result) == 1
        entry = result[0]
        
        expected_keys = {
            "operand_index", "type", "element", "test_value",
            "result_value", "change", "sensitivity"
        }
        assert set(entry.keys()) == expected_keys

    def test_type_is_enum_name_not_value(self, simple_system):
        """'type' field must be enum .name (e.g., 'RADIUS_1'), not .value."""
        tol = ToleranceOperand(
            element_index=0,
            param_type=ToleranceType.THICKNESS,
            min_val=-0.1,
            max_val=0.1,
        )
        analyzer = InverseSensitivityAnalyzer(simple_system, [tol])
        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        
        entry = result[0]
        assert entry["type"] == ToleranceType.THICKNESS.name
        assert entry["type"] == "THICKNESS"

    def test_zero_max_val_gives_int_zero_sensitivity(self, simple_system):
        """When max_val=0, sensitivity must be the int 0, not 0.0."""
        tol = ToleranceOperand(
            element_index=0,
            param_type=ToleranceType.THICKNESS,
            min_val=0.0,
            max_val=0.0,
        )
        analyzer = InverseSensitivityAnalyzer(simple_system, [tol])
        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        
        entry = result[0]
        assert entry["sensitivity"] == 0
        assert isinstance(entry["sensitivity"], int)
        assert entry["test_value"] == 0.0

    def test_multiple_tolerances_returned_in_order(self, simple_system):
        """Multiple tolerances must be returned in input order."""
        tols = [
            ToleranceOperand(0, ToleranceType.RADIUS_1, -0.1, 0.1),
            ToleranceOperand(0, ToleranceType.THICKNESS, -0.05, 0.05),
            ToleranceOperand(0, ToleranceType.RADIUS_2, -0.15, 0.15),
        ]
        analyzer = InverseSensitivityAnalyzer(simple_system, tols)
        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        
        assert len(result) == 3
        assert result[0]["operand_index"] == 0
        assert result[1]["operand_index"] == 1
        assert result[2]["operand_index"] == 2
        assert result[0]["type"] == "RADIUS_1"
        assert result[1]["type"] == "THICKNESS"
        assert result[2]["type"] == "RADIUS_2"

    def test_sensitivity_equation_finite_case(self, simple_system):
        """When values are finite, sensitivity must equal change / test_value."""
        tol = ToleranceOperand(
            element_index=0,
            param_type=ToleranceType.THICKNESS,
            min_val=-0.1,
            max_val=0.2,
        )
        analyzer = InverseSensitivityAnalyzer(simple_system, [tol])
        result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
        
        entry = result[0]
        change = entry["change"]
        sensitivity = entry["sensitivity"]
        test_value = entry["test_value"]
        
        # Only check equation when both are finite
        if math.isfinite(change) and math.isfinite(sensitivity) and test_value != 0:
            assert sensitivity == pytest.approx(change / test_value)

    def test_non_finite_values_become_nan(self, simple_system):
        """Non-finite change/sensitivity must both become NaN."""
        with patch('openlens.tolerancing.SpotDiagram') as mock_sd:
            # Nominal finite, perturbed infinite
            nominal_result = {"rms_radius": 1.0}
            perturbed_result = {"rms_radius": float('inf')}
            
            mock_sd.return_value.trace_spot.side_effect = [
                nominal_result,
                perturbed_result,
            ]
            
            tol = ToleranceOperand(
                element_index=0,
                param_type=ToleranceType.THICKNESS,
                min_val=-0.1,
                max_val=0.1,
            )
            analyzer = InverseSensitivityAnalyzer(simple_system, [tol])
            result = analyzer.calculate_sensitivities(criterion="rms_spot_radius")
            
            entry = result[0]
            assert math.isnan(entry["change"])
            assert math.isnan(entry["sensitivity"])
            assert entry["result_value"] == float('inf')
