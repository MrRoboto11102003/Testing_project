import pytest
import math
from unittest.mock import MagicMock, patch
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem
from openlens.analysis.spot_diagram import SpotDiagram
from openlens.tolerancing import InverseSensitivityAnalyzer, ToleranceOperand, ToleranceType


class TestSpotDiagramTraceSpot:
    """Tests for SpotDiagram.trace_spot"""

    @pytest.fixture
    def system_with_lens(self):
        sys = OpticalSystem()
        sys.add_lens(Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=5,
            diameter=25,
            material='BK7'
        ))
        return sys

    @pytest.fixture
    def spot_diagram(self, system_with_lens):
        return SpotDiagram(system_with_lens)

    def test_trace_spot_success_returns_six_keys(self, spot_diagram, system_with_lens):
        """Successful trace returns dict with exactly 6 keys"""
        image_plane = system_with_lens.elements[0].position + 50.0
        result = spot_diagram.trace_spot(image_plane_x_mm=image_plane)
        
        if result["valid_rays"] >= 2:
            assert len(result) == 6
            assert set(result.keys()) == {
                "rms_radius", "geo_radius", "centroid", "points", 
                "valid_rays", "image_plane_x"
            }

    def test_trace_spot_failure_returns_seven_keys_with_error(self, spot_diagram, system_with_lens):
        """Failure (num_rings=0) returns dict with 7 keys including error"""
        image_plane = system_with_lens.elements[0].position + 50.0
        result = spot_diagram.trace_spot(num_rings=0, image_plane_x_mm=image_plane)
        
        assert len(result) == 7
        assert result["error"] == "Fewer than 2 rays reached the image plane"
        assert result["rms_radius"] is None
        assert result["geo_radius"] is None
        assert result["centroid"] == (0.0, 0.0)

    def test_centroid_calculation_verified_from_points(self, spot_diagram, system_with_lens):
        """Centroid matches manual calculation from spot points"""
        image_plane = system_with_lens.elements[0].position + 50.0
        result = spot_diagram.trace_spot(image_plane_x_mm=image_plane)
        
        if result["valid_rays"] >= 2:
            points = result["points"]
            cent_y, cent_z = result["centroid"]
            
            manual_cent_y = sum(p[0] for p in points) / len(points)
            manual_cent_z = sum(p[1] for p in points) / len(points)
            
            assert cent_y == pytest.approx(manual_cent_y, abs=1e-8)
            assert cent_z == pytest.approx(manual_cent_z, abs=1e-8)

    def test_rms_radius_calculation_verified(self, spot_diagram, system_with_lens):
        """RMS radius matches formula from centroid and points"""
        image_plane = system_with_lens.elements[0].position + 50.0
        result = spot_diagram.trace_spot(image_plane_x_mm=image_plane)
        
        if result["valid_rays"] >= 2:
            points = result["points"]
            cent_y, cent_z = result["centroid"]
            
            manual_rms = math.sqrt(
                sum((p[0]-cent_y)**2 + (p[1]-cent_z)**2 for p in points) / len(points)
            )
            assert result["rms_radius"] == pytest.approx(manual_rms, abs=1e-8)

    def test_geo_radius_is_maximum_distance(self, spot_diagram, system_with_lens):
        """Geometric radius equals maximum distance from centroid"""
        image_plane = system_with_lens.elements[0].position + 50.0
        result = spot_diagram.trace_spot(image_plane_x_mm=image_plane)
        
        if result["valid_rays"] >= 2:
            points = result["points"]
            cent_y, cent_z = result["centroid"]
            
            manual_geo = max(
                math.sqrt((p[0]-cent_y)**2 + (p[1]-cent_z)**2) for p in points
            )
            assert result["geo_radius"] == pytest.approx(manual_geo, abs=1e-8)
            assert result["geo_radius"] >= result["rms_radius"]

    def test_valid_rays_equals_points_length(self, spot_diagram, system_with_lens):
        """valid_rays count matches length of points list"""
        image_plane = system_with_lens.elements[0].position + 50.0
        result = spot_diagram.trace_spot(image_plane_x_mm=image_plane)
        
        assert result["valid_rays"] == len(result["points"])

    def test_focus_shift_offsets_image_plane_x(self, spot_diagram, system_with_lens):
        """focus_shift_mm correctly offsets image_plane_x in result"""
        base_plane = system_with_lens.elements[0].position + 50.0
        shift = 5.5
        
        result = spot_diagram.trace_spot(
            image_plane_x_mm=base_plane,
            focus_shift_mm=shift
        )
        
        assert result["image_plane_x"] == pytest.approx(base_plane + shift, abs=1e-9)

    def test_lens_state_restored_after_trace(self, spot_diagram, system_with_lens):
        """Lens refractive_index and wavelength restored after trace"""
        lens = system_with_lens.elements[0].lens
        original_n = lens.refractive_index
        original_wl = lens.wavelength
        
        image_plane = system_with_lens.elements[0].position + 50.0
        spot_diagram.trace_spot(wavelength_nm=486.1, image_plane_x_mm=image_plane)
        
        assert lens.refractive_index == original_n
        assert lens.wavelength == original_wl

    def test_empty_system_raises_index_error(self):
        """Empty system raises IndexError"""
        empty_sys = OpticalSystem()
        spot = SpotDiagram(empty_sys)
        
        with pytest.raises(IndexError):
            spot.trace_spot(image_plane_x_mm=100.0)


class TestInverseSensitivityAnalyzerCalculateSensitivities:
    """Tests for InverseSensitivityAnalyzer.calculate_sensitivities"""

    @pytest.fixture
    def system_with_lens(self):
        sys = OpticalSystem()
        sys.add_lens(Lens(
            radius_of_curvature_1=100,
            radius_of_curvature_2=-100,
            thickness=5,
            diameter=25,
            material='BK7'
        ))
        return sys

    def test_unsupported_criterion_raises_value_error(self, system_with_lens):
        """Unsupported criterion raises ValueError immediately"""
        tolerances = [ToleranceOperand(0, ToleranceType.RADIUS_1, -0.5, 0.5)]
        analyzer = InverseSensitivityAnalyzer(system_with_lens, tolerances)
        
        with pytest.raises(ValueError, match="Unsupported criterion 'mtf'"):
            analyzer.calculate_sensitivities(criterion="mtf")

    def test_returns_list_with_one_dict_per_tolerance(self, system_with_lens):
        """Result is list with one dict per tolerance"""
        tolerances = [
            ToleranceOperand(0, ToleranceType.RADIUS_1, -0.5, 0.5),
            ToleranceOperand(0, ToleranceType.THICKNESS, -0.1, 0.1),
        ]
        analyzer = InverseSensitivityAnalyzer(system_with_lens, tolerances)
        
        with patch('openlens.tolerancing.SpotDiagram') as mock_sd:
            mock_inst = MagicMock()
            mock_inst.trace_spot.side_effect = [
                {"rms_radius": 1.0},
                {"rms_radius": 1.1},
                {"rms_radius": 1.2},
            ]
            mock_sd.return_value = mock_inst
            
            result = analyzer.calculate_sensitivities()
            
            assert isinstance(result, list)
            assert len(result) == 2

    def test_result_dict_has_seven_keys(self, system_with_lens):
        """Each result dict has exactly 7 keys"""
        tolerances = [ToleranceOperand(0, ToleranceType.RADIUS_1, -0.5, 0.5)]
        analyzer = InverseSensitivityAnalyzer(system_with_lens, tolerances)
        
        with patch('openlens.tolerancing.SpotDiagram') as mock_sd:
            mock_inst = MagicMock()
            mock_inst.trace_spot.side_effect = [
                {"rms_radius": 1.0},
                {"rms_radius": 1.5},
            ]
            mock_sd.return_value = mock_inst
            
            result = analyzer.calculate_sensitivities()
            
            expected_keys = {
                "operand_index", "type", "element", "test_value",
                "result_value", "change", "sensitivity"
            }
            assert set(result[0].keys()) == expected_keys

    def test_sensitivity_calculation_change_per_max_val(self, system_with_lens):
        """Sensitivity correctly computed as change / test_value"""
        tolerances = [ToleranceOperand(0, ToleranceType.RADIUS_1, -0.5, 0.5)]
        analyzer = InverseSensitivityAnalyzer(system_with_lens, tolerances)
        
        with patch('openlens.tolerancing.SpotDiagram') as mock_sd:
            mock_inst = MagicMock()
            mock_inst.trace_spot.side_effect = [
                {"rms_radius": 1.0},
                {"rms_radius": 1.25},
            ]
            mock_sd.return_value = mock_inst
            
            result = analyzer.calculate_sensitivities()
            
            entry = result[0]
            assert entry["change"] == pytest.approx(0.25, abs=1e-9)
            assert entry["sensitivity"] == pytest.approx(0.5, abs=1e-9)

    def test_max_val_zero_gives_int_zero_sensitivity(self, system_with_lens):
        """max_val=0 results in sensitivity being int 0"""
        tolerances = [ToleranceOperand(0, ToleranceType.RADIUS_1, -0.1, 0.0)]
        analyzer = InverseSensitivityAnalyzer(system_with_lens, tolerances)
        
        with patch('openlens.tolerancing.SpotDiagram') as mock_sd:
            mock_inst = MagicMock()
            mock_inst.trace_spot.side_effect = [
                {"rms_radius": 1.0},
                {"rms_radius": 1.0},
            ]
            mock_sd.return_value = mock_inst
            
            result = analyzer.calculate_sensitivities()
            
            assert result[0]["sensitivity"] == 0
            assert isinstance(result[0]["sensitivity"], int)

    def test_vignetted_nominal_gives_nan_change_sensitivity(self, system_with_lens):
        """Nominal None (vignetted) gives NaN for change and sensitivity"""
        tolerances = [ToleranceOperand(0, ToleranceType.RADIUS_1, -0.5, 0.5)]
        analyzer = InverseSensitivityAnalyzer(system_with_lens, tolerances)
        
        with patch('openlens.tolerancing.SpotDiagram') as mock_sd:
            mock_inst = MagicMock()
            mock_inst.trace_spot.side_effect = [
                {"rms_radius": None},
                {"rms_radius": 1.5},
            ]
            mock_sd.return_value = mock_inst
            
            result = analyzer.calculate_sensitivities()
            
            entry = result[0]
            assert math.isnan(entry["change"])
            assert math.isnan(entry["sensitivity"])

    def test_exception_during_operand_trace_gives_inf(self, system_with_lens):
        """Exception during operand trace sets result_value to inf"""
        tolerances = [ToleranceOperand(0, ToleranceType.RADIUS_1, -0.5, 0.5)]
        analyzer = InverseSensitivityAnalyzer(system_with_lens, tolerances)
        
        with patch('openlens.tolerancing.SpotDiagram') as mock_sd:
            mock_inst = MagicMock()
            mock_inst.trace_spot.side_effect = [
                {"rms_radius": 1.0},
                RuntimeError("Trace failed"),
            ]
            mock_sd.return_value = mock_inst
            
            result = analyzer.calculate_sensitivities()
            
            entry = result[0]
            assert entry["result_value"] == float("inf")
            assert math.isnan(entry["change"])
            assert math.isnan(entry["sensitivity"])

    def test_empty_tolerances_returns_empty_list(self, system_with_lens):
        """Empty tolerances list returns empty list"""
        analyzer = InverseSensitivityAnalyzer(system_with_lens, [])
        
        with patch('openlens.tolerancing.SpotDiagram') as mock_sd:
            mock_inst = MagicMock()
            mock_inst.trace_spot.return_value = {"rms_radius": 1.0}
            mock_sd.return_value = mock_inst
            
            result = analyzer.calculate_sensitivities()
            
            assert result == []
            assert mock_inst.trace_spot.call_count == 1
