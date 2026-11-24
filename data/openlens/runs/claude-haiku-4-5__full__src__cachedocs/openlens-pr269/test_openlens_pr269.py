import pytest
import numpy as np
from openlens.chromatic_analyzer import ChromaticResult


class TestChromaticResultToDict:
    """Tests for ChromaticResult.to_dict() method"""

    def test_to_dict_returns_dict(self):
        """Test that to_dict returns a dictionary object"""
        result = ChromaticResult(
            wavelengths=[550],
            focal_lengths=[100.0],
            focal_shift=0.0,
            lateral_color=0.0,
            spot_sizes=[0.0],
            transverse_aberration=0.0,
            axial_chromatic_aberration=0.0,
        )
        output = result.to_dict()
        assert isinstance(output, dict)

    def test_to_dict_has_all_required_keys(self):
        """Test that to_dict contains all expected keys"""
        result = ChromaticResult(
            wavelengths=[550],
            focal_lengths=[100.0],
            focal_shift=0.0,
            lateral_color=0.0,
            spot_sizes=[0.0],
            transverse_aberration=0.0,
            axial_chromatic_aberration=0.0,
        )
        output = result.to_dict()
        expected_keys = {
            "wavelengths",
            "focal_lengths",
            "focal_shift",
            "lateral_color",
            "spot_sizes",
            "transverse_aberration",
            "axial_chromatic_aberration",
        }
        assert set(output.keys()) == expected_keys

    def test_to_dict_preserves_basic_values(self):
        """Test to_dict preserves scalar and list values exactly"""
        result = ChromaticResult(
            wavelengths=[450, 550, 650],
            focal_lengths=[100.1, 100.0, 99.9],
            focal_shift=0.2,
            lateral_color=0.001,
            spot_sizes=[0.5, 0.3, 0.6],
            transverse_aberration=0.002,
            axial_chromatic_aberration=0.1,
        )
        output = result.to_dict()
        
        assert output["wavelengths"] == [450, 550, 650]
        assert output["focal_lengths"] == [100.1, 100.0, 99.9]
        assert output["focal_shift"] == 0.2
        assert output["lateral_color"] == 0.001
        assert output["spot_sizes"] == [0.5, 0.3, 0.6]
        assert output["transverse_aberration"] == 0.002
        assert output["axial_chromatic_aberration"] == 0.1

    def test_to_dict_with_numpy_arrays(self):
        """Test to_dict correctly handles numpy array inputs"""
        result = ChromaticResult(
            wavelengths=np.array([450, 550, 650]),
            focal_lengths=np.array([100.1, 100.0, 99.9]),
            focal_shift=0.2,
            lateral_color=0.001,
            spot_sizes=np.array([0.5, 0.3, 0.6]),
            transverse_aberration=0.002,
            axial_chromatic_aberration=0.1,
        )
        output = result.to_dict()
        
        # Verify array values are accessible
        assert len(output["wavelengths"]) == 3
        assert len(output["focal_lengths"]) == 3
        assert len(output["spot_sizes"]) == 3

    def test_to_dict_with_none_aberration_values(self):
        """Test to_dict when trace or calculation fails (returns None)"""
        result = ChromaticResult(
            wavelengths=[550],
            focal_lengths=[100.0],
            focal_shift=None,
            lateral_color=None,
            spot_sizes=[0.0],
            transverse_aberration=None,
            axial_chromatic_aberration=None,
        )
        output = result.to_dict()
        
        assert output["focal_shift"] is None
        assert output["lateral_color"] is None
        assert output["transverse_aberration"] is None
        assert output["axial_chromatic_aberration"] is None

    def test_to_dict_with_empty_spectrum(self):
        """Test to_dict with empty wavelength list (edge case)"""
        result = ChromaticResult(
            wavelengths=[],
            focal_lengths=[],
            focal_shift=0.0,
            lateral_color=0.0,
            spot_sizes=[],
            transverse_aberration=0.0,
            axial_chromatic_aberration=0.0,
        )
        output = result.to_dict()
        
        assert output["wavelengths"] == []
        assert output["focal_lengths"] == []
        assert output["spot_sizes"] == []

    def test_to_dict_with_negative_aberrations(self):
        """Test to_dict preserves negative aberration values"""
        result = ChromaticResult(
            wavelengths=[550],
            focal_lengths=[-100.0],
            focal_shift=-0.1,
            lateral_color=-0.005,
            spot_sizes=[-0.2],
            transverse_aberration=-0.001,
            axial_chromatic_aberration=-0.05,
        )
        output = result.to_dict()
        
        assert output["focal_shift"] == -0.1
        assert output["lateral_color"] == -0.005
        assert output["transverse_aberration"] == -0.001
        assert output["axial_chromatic_aberration"] == -0.05

    def test_to_dict_ideal_achromatic_lens(self):
        """Test to_dict with zero aberrations (perfect achromatic lens)"""
        result = ChromaticResult(
            wavelengths=[550],
            focal_lengths=[100.0],
            focal_shift=0.0,
            lateral_color=0.0,
            spot_sizes=[0.0],
            transverse_aberration=0.0,
            axial_chromatic_aberration=0.0,
        )
        output = result.to_dict()
        
        assert all(val == 0.0 for val in [
            output["focal_shift"],
            output["lateral_color"],
            output["transverse_aberration"],
            output["axial_chromatic_aberration"],
        ])

    def test_to_dict_full_visible_spectrum(self):
        """Test to_dict with complete visible spectrum (400-700 nm)"""
        wavelengths = [400, 450, 500, 550, 600, 650, 700]
        focal_lengths = [99.5, 99.7, 99.9, 100.0, 100.1, 100.3, 100.5]
        spot_sizes = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        
        result = ChromaticResult(
            wavelengths=wavelengths,
            focal_lengths=focal_lengths,
            focal_shift=1.0,
            lateral_color=0.05,
            spot_sizes=spot_sizes,
            transverse_aberration=0.01,
            axial_chromatic_aberration=0.5,
        )
        output = result.to_dict()
        
        assert len(output["wavelengths"]) == 7
        assert len(output["focal_lengths"]) == 7
        assert len(output["spot_sizes"]) == 7
        assert output["wavelengths"] == wavelengths

    def test_to_dict_does_not_mutate_original(self):
        """Test that modifying returned dict does not affect ChromaticResult"""
        original_wavelengths = [450, 550, 650]
        result = ChromaticResult(
            wavelengths=original_wavelengths,
            focal_lengths=[100.1, 100.0, 99.9],
            focal_shift=0.2,
            lateral_color=0.001,
            spot_sizes=[0.5, 0.3, 0.6],
            transverse_aberration=0.002,
            axial_chromatic_aberration=0.1,
        )
        
        output = result.to_dict()
        output["focal_shift"] = 999.0
        
        # Original ChromaticResult should remain unchanged
        assert result.focal_shift == 0.2

    def test_to_dict_large_chromatic_aberration(self):
        """Test to_dict with significant chromatic aberration (poor lens design)"""
        result = ChromaticResult(
            wavelengths=[400, 550, 700],
            focal_lengths=[95.0, 100.0, 105.0],
            focal_shift=10.0,
            lateral_color=1.0,
            spot_sizes=[5.0, 2.0, 5.0],
            transverse_aberration=0.5,
            axial_chromatic_aberration=5.0,
        )
        output = result.to_dict()
        
        assert output["focal_shift"] == 10.0
        assert output["axial_chromatic_aberration"] == 5.0
        assert output["lateral_color"] == 1.0

    def test_to_dict_very_small_aberrations(self):
        """Test to_dict with very small aberration values (high-precision lens)"""
        result = ChromaticResult(
            wavelengths=[550],
            focal_lengths=[100.0],
            focal_shift=1e-6,
            lateral_color=1e-7,
            spot_sizes=[1e-5],
            transverse_aberration=1e-8,
            axial_chromatic_aberration=1e-6,
        )
        output = result.to_dict()
        
        assert output["focal_shift"] == pytest.approx(1e-6)
        assert output["lateral_color"] == pytest.approx(1e-7)
        assert output["transverse_aberration"] == pytest.approx(1e-8)

    def test_to_dict_multiple_calls_consistent(self):
        """Test that multiple calls to to_dict return consistent results"""
        result = ChromaticResult(
            wavelengths=[450, 550, 650],
            focal_lengths=[100.1, 100.0, 99.9],
            focal_shift=0.2,
            lateral_color=0.001,
            spot_sizes=[0.5, 0.3, 0.6],
            transverse_aberration=0.002,
            axial_chromatic_aberration=0.1,
        )
        
        output1 = result.to_dict()
        output2 = result.to_dict()
        
        assert output1 == output2
