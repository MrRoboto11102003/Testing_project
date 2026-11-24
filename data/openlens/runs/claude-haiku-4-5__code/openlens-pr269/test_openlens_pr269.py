import pytest
import numpy as np
from openlens.chromatic_analyzer import ChromaticResult


class TestChromaticResultToDict:
    """Tests for ChromaticResult.to_dict() method"""

    def test_to_dict_returns_dict_type(self):
        """Test that to_dict() returns a dictionary"""
        result = ChromaticResult(
            wavelengths=np.array([400, 500, 600]),
            focal_lengths=np.array([10.0, 10.1, 10.2]),
            focal_shift=np.array([0.0, 0.1, 0.2]),
            lateral_color=np.array([0.0, 0.01, 0.02]),
            spot_sizes=np.array([0.5, 0.6, 0.7]),
            transverse_aberration=np.array([0.0, 0.001, 0.002]),
            axial_chromatic_aberration=np.array([0.0, 0.1, 0.2])
        )
        output = result.to_dict()
        assert isinstance(output, dict)

    def test_to_dict_has_all_required_keys(self):
        """Test that to_dict() contains all required keys"""
        result = ChromaticResult(
            wavelengths=np.array([550]),
            focal_lengths=np.array([10.0]),
            focal_shift=np.array([0.0]),
            lateral_color=np.array([0.0]),
            spot_sizes=np.array([0.5]),
            transverse_aberration=np.array([0.0]),
            axial_chromatic_aberration=np.array([0.0])
        )
        output = result.to_dict()
        required_keys = {
            "wavelengths",
            "focal_lengths",
            "focal_shift",
            "lateral_color",
            "spot_sizes",
            "transverse_aberration",
            "axial_chromatic_aberration",
        }
        assert set(output.keys()) == required_keys

    def test_to_dict_preserves_wavelengths(self):
        """Test that wavelengths array is correctly preserved"""
        wavelengths = np.array([380, 450, 550, 650, 780])
        result = ChromaticResult(
            wavelengths=wavelengths,
            focal_lengths=np.ones(5) * 10.0,
            focal_shift=np.zeros(5),
            lateral_color=np.zeros(5),
            spot_sizes=np.ones(5) * 0.5,
            transverse_aberration=np.zeros(5),
            axial_chromatic_aberration=np.zeros(5)
        )
        output = result.to_dict()
        np.testing.assert_array_equal(output["wavelengths"], wavelengths)

    def test_to_dict_preserves_focal_lengths(self):
        """Test that focal_lengths array is correctly preserved"""
        focal_lengths = np.array([9.8, 9.9, 10.0, 10.1, 10.2])
        result = ChromaticResult(
            wavelengths=np.arange(5),
            focal_lengths=focal_lengths,
            focal_shift=np.zeros(5),
            lateral_color=np.zeros(5),
            spot_sizes=np.ones(5) * 0.5,
            transverse_aberration=np.zeros(5),
            axial_chromatic_aberration=np.zeros(5)
        )
        output = result.to_dict()
        np.testing.assert_array_equal(output["focal_lengths"], focal_lengths)

    def test_to_dict_preserves_all_aberration_arrays(self):
        """Test that all aberration-related arrays are preserved"""
        focal_shift = np.array([-0.1, 0.0, 0.1])
        lateral_color = np.array([-0.02, 0.0, 0.02])
        transverse_aberration = np.array([-0.001, 0.0, 0.001])
        axial_chromatic_aberration = np.array([-0.2, 0.0, 0.2])
        
        result = ChromaticResult(
            wavelengths=np.array([400, 550, 700]),
            focal_lengths=np.array([10.0, 10.0, 10.0]),
            focal_shift=focal_shift,
            lateral_color=lateral_color,
            spot_sizes=np.array([0.5, 0.5, 0.5]),
            transverse_aberration=transverse_aberration,
            axial_chromatic_aberration=axial_chromatic_aberration
        )
        output = result.to_dict()
        
        np.testing.assert_array_equal(output["focal_shift"], focal_shift)
        np.testing.assert_array_equal(output["lateral_color"], lateral_color)
        np.testing.assert_array_equal(output["transverse_aberration"], transverse_aberration)
        np.testing.assert_array_equal(output["axial_chromatic_aberration"], axial_chromatic_aberration)

    def test_to_dict_with_single_element(self):
        """Test to_dict with single wavelength/measurement"""
        result = ChromaticResult(
            wavelengths=np.array([550.0]),
            focal_lengths=np.array([10.0]),
            focal_shift=np.array([0.0]),
            lateral_color=np.array([0.0]),
            spot_sizes=np.array([0.5]),
            transverse_aberration=np.array([0.0]),
            axial_chromatic_aberration=np.array([0.0])
        )
        output = result.to_dict()
        assert len(output["wavelengths"]) == 1
        assert all(len(output[key]) == 1 for key in output.keys())

    def test_to_dict_with_large_array(self):
        """Test to_dict with large arrays"""
        n = 1000
        result = ChromaticResult(
            wavelengths=np.linspace(300, 1000, n),
            focal_lengths=np.full(n, 10.0),
            focal_shift=np.random.randn(n) * 0.5,
            lateral_color=np.random.randn(n) * 0.1,
            spot_sizes=np.full(n, 0.5),
            transverse_aberration=np.random.randn(n) * 0.01,
            axial_chromatic_aberration=np.random.randn(n) * 0.5
        )
        output = result.to_dict()
        assert all(len(output[key]) == n for key in output.keys())

    def test_to_dict_with_negative_aberrations(self):
        """Test to_dict correctly handles negative aberration values"""
        result = ChromaticResult(
            wavelengths=np.array([400, 550, 700]),
            focal_lengths=np.array([9.9, 10.0, 10.1]),
            focal_shift=np.array([-0.1, 0.0, 0.1]),
            lateral_color=np.array([-0.01, 0.0, 0.01]),
            spot_sizes=np.array([0.4, 0.5, 0.6]),
            transverse_aberration=np.array([-0.001, 0.0, 0.001]),
            axial_chromatic_aberration=np.array([-0.1, 0.0, 0.1])
        )
        output = result.to_dict()
        assert np.any(output["focal_shift"] < 0)
        assert np.any(output["lateral_color"] < 0)
        assert np.any(output["axial_chromatic_aberration"] < 0)

    def test_to_dict_with_zero_arrays(self):
        """Test to_dict with all zero aberration arrays"""
        result = ChromaticResult(
            wavelengths=np.array([400, 550, 700]),
            focal_lengths=np.array([10.0, 10.0, 10.0]),
            focal_shift=np.zeros(3),
            lateral_color=np.zeros(3),
            spot_sizes=np.array([0.5, 0.5, 0.5]),
            transverse_aberration=np.zeros(3),
            axial_chromatic_aberration=np.zeros(3)
        )
        output = result.to_dict()
        assert np.allclose(output["focal_shift"], 0.0)
        assert np.allclose(output["lateral_color"], 0.0)
        assert np.allclose(output["transverse_aberration"], 0.0)
        assert np.allclose(output["axial_chromatic_aberration"], 0.0)

    def test_to_dict_multiple_calls_consistency(self):
        """Test that multiple calls to to_dict return consistent results"""
        result = ChromaticResult(
            wavelengths=np.array([400, 500, 600]),
            focal_lengths=np.array([10.0, 10.1, 10.2]),
            focal_shift=np.array([0.0, 0.1, 0.2]),
            lateral_color=np.array([0.0, 0.01, 0.02]),
            spot_sizes=np.array([0.5, 0.6, 0.7]),
            transverse_aberration=np.array([0.0, 0.001, 0.002]),
            axial_chromatic_aberration=np.array([0.0, 0.1, 0.2])
        )
        dict1 = result.to_dict()
        dict2 = result.to_dict()
        for key in dict1.keys():
            np.testing.assert_array_equal(dict1[key], dict2[key])

    def test_to_dict_with_inf_values(self):
        """Test to_dict correctly handles infinite values"""
        result = ChromaticResult(
            wavelengths=np.array([400, 500, 600]),
            focal_lengths=np.array([10.0, np.inf, 10.2]),
            focal_shift=np.array([0.0, 0.1, np.inf]),
            lateral_color=np.array([0.0, 0.01, 0.02]),
            spot_sizes=np.array([0.5, 0.6, 0.7]),
            transverse_aberration=np.array([0.0, 0.001, 0.002]),
            axial_chromatic_aberration=np.array([0.0, 0.1, 0.2])
        )
        output = result.to_dict()
        assert np.inf in output["focal_lengths"]
        assert np.inf in output["focal_shift"]

    def test_to_dict_with_nan_values(self):
        """Test to_dict correctly handles NaN values"""
        result = ChromaticResult(
            wavelengths=np.array([400, 500, 600]),
            focal_lengths=np.array([10.0, np.nan, 10.2]),
            focal_shift=np.array([0.0, 0.1, 0.2]),
            lateral_color=np.array([np.nan, 0.01, 0.02]),
            spot_sizes=np.array([0.5, 0.6, 0.7]),
            transverse_aberration=np.array([0.0, 0.001, 0.002]),
            axial_chromatic_aberration=np.array([0.0, 0.1, 0.2])
        )
        output = result.to_dict()
        assert np.isnan(output["focal_lengths"][1])
        assert np.isnan(output["lateral_color"][0])

    def test_to_dict_array_data_types_preserved(self):
        """Test that numpy array data types are preserved"""
        wavelengths_float = np.array([400.0, 500.0, 600.0], dtype=np.float32)
        focal_lengths_float64 = np.array([10.0, 10.1, 10.2], dtype=np.float64)
        
        result = ChromaticResult(
            wavelengths=wavelengths_float,
            focal_lengths=focal_lengths_float64,
            focal_shift=np.array([0.0, 0.1, 0.2]),
            lateral_color=np.array([0.0, 0.01, 0.02]),
            spot_sizes=np.array([0.5, 0.6, 0.7]),
            transverse_aberration=np.array([0.0, 0.001, 0.002]),
            axial_chromatic_aberration=np.array([0.0, 0.1, 0.2])
        )
        output = result.to_dict()
        # Arrays should be preserved with their values
        assert len(output["wavelengths"]) == 3
        assert len(output["focal_lengths"]) == 3
