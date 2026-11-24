import pytest
import numpy as np
from openlens.lens import Lens
from openlens.polarization import (
    PolarizationState,
    PolarizationCalculator,
    JonesMatrices,
)
from openlens.tolerancing import ToleranceType


class TestLensFocalLengths:
    """Test Lens BFL and FFL calculations with edge cases"""

    def test_bfl_converging_biconvex(self):
        """Test BFL for typical biconvex lens (positive)"""
        lens = Lens(
            name="Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        bfl = lens.calculate_back_focal_length()
        assert bfl is not None
        assert isinstance(bfl, float)
        assert bfl > 0

    def test_bfl_afocal_returns_none(self):
        """Test BFL returns None (not inf) for afocal plane-parallel plate"""
        lens = Lens(
            name="Plate",
            radius_of_curvature_1=float('inf'),
            radius_of_curvature_2=float('inf'),
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        bfl = lens.calculate_back_focal_length()
        assert bfl is None

    def test_ffl_converging_lens_negative(self):
        """Test FFL is negative for converging lens (front focus on object side)"""
        lens = Lens(
            name="Biconvex",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        ffl = lens.calculate_front_focal_length()
        assert ffl is not None
        assert ffl < 0

    def test_ffl_afocal_returns_none(self):
        """Test FFL returns None for afocal element"""
        lens = Lens(
            name="Plate",
            radius_of_curvature_1=float('inf'),
            radius_of_curvature_2=float('inf'),
            thickness=5.0,
            diameter=25.0,
            material="BK7",
        )
        ffl = lens.calculate_front_focal_length()
        assert ffl is None


class TestJonesMatricesLinearPolarizer:
    """Test Jones matrix for linear polarizers at various angles"""

    def test_linear_polarizer_0_degrees(self):
        """Test 0° polarizer transmits x-component only"""
        matrix = JonesMatrices.linear_polarizer(0.0)
        expected = np.array([[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(matrix, expected, atol=1e-10)

    def test_linear_polarizer_90_degrees(self):
        """Test 90° polarizer transmits y-component only"""
        matrix = JonesMatrices.linear_polarizer(90.0)
        expected = np.array([[0.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(matrix, expected, atol=1e-10)

    def test_linear_polarizer_45_degrees(self):
        """Test 45° polarizer has equal x/y transmission"""
        matrix = JonesMatrices.linear_polarizer(45.0)
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_allclose(matrix, expected, atol=1e-10)


class TestFresnelCoefficients:
    """Test Fresnel reflection/transmission coefficients"""

    def setup_method(self):
        self.calc = PolarizationCalculator()

    def test_fresnel_normal_incidence(self):
        """Test Fresnel coefficients at normal incidence"""
        coeffs = self.calc.fresnel_coefficients(n1=1.0, n2=1.5, angle_deg=0.0)
        assert coeffs["total_internal_reflection"] is False
        expected_r = (1.0 - 1.5) / (1.0 + 1.5)
        np.testing.assert_allclose(coeffs["r_s"], expected_r, atol=1e-10)
        np.testing.assert_allclose(coeffs["r_p"], expected_r, atol=1e-10)

    def test_fresnel_total_internal_reflection(self):
        """Test Fresnel with TIR: transmission zero, reflection unit magnitude"""
        coeffs = self.calc.fresnel_coefficients(n1=1.5, n2=1.0, angle_deg=50.0)
        assert coeffs["total_internal_reflection"] is True
        assert coeffs["t_s"] == 0.0 + 0j
        assert coeffs["t_p"] == 0.0 + 0j
        assert np.abs(coeffs["r_s"]) == pytest.approx(1.0, abs=1e-10)
        assert np.abs(coeffs["r_p"]) == pytest.approx(1.0, abs=1e-10)


class TestReflectanceTransmittance:
    """Test power reflectance and transmittance calculations"""

    def setup_method(self):
        self.calc = PolarizationCalculator()

    def test_reflectance_transmittance_energy_conservation(self):
        """Test R + T = 1 for normal incidence (energy conservation)"""
        rt = self.calc.reflectance_transmittance(n1=1.0, n2=1.5, angle_deg=0.0)
        assert (rt["R_s"] + rt["T_s"]) == pytest.approx(1.0, abs=1e-10)
        assert (rt["R_p"] + rt["T_p"]) == pytest.approx(1.0, abs=1e-10)

    def test_reflectance_transmittance_tir(self):
        """Test TIR gives R=1, T=0"""
        rt = self.calc.reflectance_transmittance(n1=1.5, n2=1.0, angle_deg=50.0)
        assert rt["R_s"] == pytest.approx(1.0, abs=1e-10)
        assert rt["R_p"] == pytest.approx(1.0, abs=1e-10)
        assert rt["T_s"] == pytest.approx(0.0, abs=1e-10)
        assert rt["T_p"] == pytest.approx(0.0, abs=1e-10)


class TestPolarizationState:
    """Test PolarizationState methods: elliptical, intensity, Stokes, matrix application"""

    def test_elliptical_polarization_normalized(self):
        """Test elliptical polarization is normalized to unit intensity"""
        state = PolarizationState.elliptical(a=2.0, b=1.0, angle_deg=0.0)
        intensity = state.intensity()
        assert intensity == pytest.approx(1.0, abs=1e-10)

    def test_intensity_linear_polarization(self):
        """Test intensity calculation for linear x-polarized state"""
        jones = np.array([1.0, 0.0], dtype=complex)
        state = PolarizationState(jones)
        intensity = state.intensity()
        assert intensity == pytest.approx(1.0, abs=1e-10)

    def test_stokes_parameters_linear_x(self):
        """Test Stokes parameters [S0,S1,S2,S3] for linear x polarization"""
        jones = np.array([1.0, 0.0], dtype=complex)
        state = PolarizationState(jones)
        stokes = state.stokes_parameters()
        assert stokes[0] == pytest.approx(1.0, abs=1e-10)
        assert stokes[1] == pytest.approx(1.0, abs=1e-10)
        assert stokes[2] == pytest.approx(0.0, abs=1e-10)
        assert stokes[3] == pytest.approx(0.0, abs=1e-10)

    def test_apply_jones_matrix(self):
        """Test applying Jones matrix transformation to polarization state"""
        jones = np.array([1.0, 0.0], dtype=complex)
        state = PolarizationState(jones)
        polarizer = JonesMatrices.linear_polarizer(90.0)
        result = state.apply_matrix(polarizer)
        result_intensity = result.intensity()
        assert result_intensity == pytest.approx(0.0, abs=1e-10)


class TestToleranceType:
    """Test ToleranceType enum"""

    def test_tolerance_type_enum_members(self):
        """Test ToleranceType has expected members with native units"""
        assert ToleranceType.RADIUS_1.value == "Radius 1"
        assert ToleranceType.THICKNESS.value == "Thickness"
        assert ToleranceType.REFRACTIVE_INDEX.value == "Refractive Index"
        assert ToleranceType.FOCUS.value == "Focus"
        assert ToleranceType.WEDGE.value == "Wedge"
