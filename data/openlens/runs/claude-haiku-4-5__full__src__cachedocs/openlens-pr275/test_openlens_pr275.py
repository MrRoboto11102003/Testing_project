import pytest
import numpy as np
from openlens.lens import Lens
from openlens.polarization import (
    PolarizationState,
    PolarizationCalculator,
    JonesMatrices,
)
from openlens.tolerancing import ToleranceType


class TestBackFocalLength:
    """Test Lens.calculate_back_focal_length()"""

    def test_bfl_normal_thick_lens(self):
        """Test BFL for a normal biconvex thick lens"""
        lens = Lens(
            name="Biconvex",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5168,
        )
        bfl = lens.calculate_back_focal_length()
        assert bfl is not None
        assert isinstance(bfl, float)
        assert bfl > 0

    def test_bfl_returns_none_for_afocal_lens(self):
        """Test that BFL returns None when focal length is undefined"""
        lens = Lens(
            name="Plano-plano",
            radius_of_curvature_1=float("inf"),
            radius_of_curvature_2=float("inf"),
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        bfl = lens.calculate_back_focal_length()
        assert bfl is None


class TestFrontFocalLength:
    """Test Lens.calculate_front_focal_length()"""

    def test_ffl_normal_thick_lens_negative(self):
        """Test that FFL is negative for converging lens (focus on object side)"""
        lens = Lens(
            name="Biconvex",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5168,
        )
        ffl = lens.calculate_front_focal_length()
        assert ffl is not None
        assert isinstance(ffl, float)
        assert ffl < 0

    def test_ffl_returns_none_for_afocal_lens(self):
        """Test that FFL returns None when focal length is undefined"""
        lens = Lens(
            name="Plano-plano",
            radius_of_curvature_1=float("inf"),
            radius_of_curvature_2=float("inf"),
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        ffl = lens.calculate_front_focal_length()
        assert ffl is None

    def test_bfl_ffl_symmetric_lens(self):
        """Test BFL and FFL relationship for symmetric lens"""
        lens = Lens(
            name="Symmetric",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        bfl = lens.calculate_back_focal_length()
        ffl = lens.calculate_front_focal_length()
        assert bfl is not None
        assert ffl is not None
        # For symmetric lens, magnitudes should be close
        assert abs(abs(ffl) - bfl) < abs(bfl) * 0.15


class TestJonesMatrices:
    """Test JonesMatrices.linear_polarizer()"""

    def test_linear_polarizer_key_angles(self):
        """Test linear polarizer at 0°, 45°, and 90°"""
        # 0 degrees (horizontal)
        m0 = JonesMatrices.linear_polarizer(0)
        np.testing.assert_array_almost_equal(m0, np.array([[1, 0], [0, 0]]))

        # 90 degrees (vertical)
        m90 = JonesMatrices.linear_polarizer(90)
        np.testing.assert_array_almost_equal(m90, np.array([[0, 0], [0, 1]]))

        # 45 degrees
        m45 = JonesMatrices.linear_polarizer(45)
        np.testing.assert_array_almost_equal(m45, np.array([[0.5, 0.5], [0.5, 0.5]]))

    def test_linear_polarizer_idempotent(self):
        """Test that M² = M (projection idempotence)"""
        matrix = JonesMatrices.linear_polarizer(30)
        result = matrix @ matrix
        np.testing.assert_array_almost_equal(result, matrix)


class TestFresselCoefficients:
    """Test PolarizationCalculator.fresnel_coefficients()"""

    def test_fresnel_normal_incidence(self):
        """Test Fresnel at normal incidence (θ=0°)"""
        calc = PolarizationCalculator()
        coeffs = calc.fresnel_coefficients(n1=1.0, n2=1.5, angle_deg=0)

        assert coeffs["total_internal_reflection"] is False
        assert "r_s" in coeffs and "r_p" in coeffs
        # At normal incidence, r_s == r_p
        assert np.abs(coeffs["r_s"] - coeffs["r_p"]) < 1e-10
        assert np.abs(coeffs["r_s"]) > 0  # Non-zero reflection at interface

    def test_fresnel_total_internal_reflection(self):
        """Test that TIR occurs and gives |r|=1"""
        calc = PolarizationCalculator()
        # Dense medium (n=1.5) to air (n=1) at large angle
        coeffs = calc.fresnel_coefficients(n1=1.5, n2=1.0, angle_deg=50)

        assert coeffs["total_internal_reflection"] is True
        assert np.abs(coeffs["r_s"]) == pytest.approx(1.0, abs=1e-6)
        assert np.abs(coeffs["r_p"]) == pytest.approx(1.0, abs=1e-6)
        assert coeffs["t_s"] == 0
        assert coeffs["t_p"] == 0


class TestReflectanceTransmittance:
    """Test PolarizationCalculator.reflectance_transmittance()"""

    def test_energy_conservation_normal_incidence(self):
        """Test R + T ≈ 1 at normal incidence (no absorption)"""
        calc = PolarizationCalculator()
        result = calc.reflectance_transmittance(n1=1.0, n2=1.5, angle_deg=0)

        # At normal incidence, R_s == R_p and T_s == T_p
        R_total = result["R_s"]
        T_total = result["T_s"]
        assert R_total + T_total == pytest.approx(1.0, abs=1e-10)

    def test_total_internal_reflection_full_power(self):
        """Test that TIR gives R=1, T=0"""
        calc = PolarizationCalculator()
        result = calc.reflectance_transmittance(n1=1.5, n2=1.0, angle_deg=50)

        assert result["R_s"] == pytest.approx(1.0, abs=1e-10)
        assert result["R_p"] == pytest.approx(1.0, abs=1e-10)
        assert result["T_s"] == pytest.approx(0.0, abs=1e-10)
        assert result["T_p"] == pytest.approx(0.0, abs=1e-10)


class TestPolarizationState:
    """Test PolarizationState methods"""

    def test_elliptical_intensity_normalized(self):
        """Test elliptical polarization creation and intensity"""
        state = PolarizationState.elliptical(a=1.0, b=0.5, angle_deg=0)
        intensity = state.intensity()
        assert 0.9 < intensity < 1.1  # Approximately normalized

    def test_apply_matrix_transforms_state(self):
        """Test that Jones matrix application transforms polarization"""
        # Horizontal polarization
        state = PolarizationState(np.array([1.0 + 0j, 0.0 + 0j]))

        # 45-degree linear polarizer
        matrix = JonesMatrices.linear_polarizer(45)
        rotated = state.apply_matrix(matrix)

        # After 45-degree polarizer applied to horizontal, both components present
        assert rotated.jones_vector is not None
        assert len(rotated.jones_vector) == 2

    def test_stokes_parameters_linear_horizontal(self):
        """Test Stokes parameters for horizontal linear polarization"""
        state = PolarizationState(np.array([1.0 + 0j, 0.0 + 0j]))
        stokes = state.stokes_parameters()

        assert len(stokes) == 4
        assert stokes[0] == pytest.approx(1.0)  # S0 = total intensity
        assert stokes[1] == pytest.approx(1.0)  # S1 = horizontal preference
        assert stokes[2] == pytest.approx(0.0)  # S2 = 0 for linear
        assert stokes[3] == pytest.approx(0.0)  # S3 = 0 for linear


class TestToleranceType:
    """Test ToleranceType enum"""

    def test_tolerance_type_enum_members(self):
        """Test that ToleranceType has all expected members"""
        expected_members = {
            "RADIUS_1",
            "RADIUS_2",
            "THICKNESS",
            "REFRACTIVE_INDEX",
            "DECENTER_X",
            "DECENTER_Y",
            "TILT_X",
            "TILT_Y",
            "DECENTER_Z",
            "AIR_GAP",
            "ABBE_NUMBER",
            "IRREGULARITY",
            "WEDGE",
            "FOCUS",
        }
        actual_members = {member.name for member in ToleranceType}
        assert actual_members == expected_members

    def test_tolerance_type_descriptions_in_values(self):
        """Test that tolerance type enum has human-readable descriptions"""
        assert ToleranceType.RADIUS_1.value == "Radius 1"
        assert ToleranceType.FOCUS.value == "Focus"
        # All values should be non-empty strings
        for tol_type in ToleranceType:
            assert isinstance(tol_type.value, str)
            assert len(tol_type.value) > 0
