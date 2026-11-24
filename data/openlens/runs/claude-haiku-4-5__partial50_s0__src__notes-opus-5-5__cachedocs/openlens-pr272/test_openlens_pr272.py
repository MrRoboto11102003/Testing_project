import pytest
from openlens.lens import Lens
from openlens.optical_system import OpticalSystem


class TestOpticalSystemGetSystemFocalLength:
    """Tests for OpticalSystem.get_system_focal_length()."""

    def test_empty_system_returns_none(self):
        """An empty system has no focal length."""
        system = OpticalSystem()
        assert system.get_system_focal_length() is None

    def test_single_lens_returns_its_focal_length(self):
        """A single-lens system returns that lens's focal length."""
        lens = Lens(
            name="single",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system = OpticalSystem()
        system.add_lens(lens)

        expected = lens.calculate_focal_length()
        assert system.get_system_focal_length() == expected

    def test_single_lens_ignores_air_gap_before(self):
        """The air gap before the first lens has no effect on focal length."""
        lens = Lens(
            name="single",
            radius_of_curvature_1=50.0,
            radius_of_curvature_2=-50.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system = OpticalSystem()
        system.add_lens(lens, air_gap_before=10.0)

        expected = lens.calculate_focal_length()
        assert system.get_system_focal_length() == expected

    def test_two_lens_system_with_gap_worked_example(self):
        """Two identical thick lenses with 10 mm gap compute EFL via ABCD matrix."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system = OpticalSystem()
        system.add_lens(L1)
        system.add_lens(L2, air_gap_before=10.0)

        # Expected from ABCD: C = -0.018519375 -> EFL = -1/C ≈ 53.9975
        expected_efl = pytest.approx(53.9975, rel=1e-4)
        assert system.get_system_focal_length() == expected_efl

    def test_air_gap_affects_focal_length(self):
        """Changing the air gap between elements changes the system focal length."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system = OpticalSystem()
        system.add_lens(L1)
        system.add_lens(L2, air_gap_before=10.0)

        efl_10mm = system.get_system_focal_length()

        # Change gap to 20 mm
        system.air_gaps[0].thickness = 20.0
        efl_20mm = system.get_system_focal_length()

        # EFLs must differ due to gap sensitivity
        assert efl_10mm != efl_20mm

    def test_cemented_lenses_zero_gap(self):
        """Cemented lenses (zero air gap) compute a focal length."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system = OpticalSystem()
        system.add_lens(L1)
        system.add_lens(L2, air_gap_before=0.0)

        efl = system.get_system_focal_length()
        assert efl is not None
        assert isinstance(efl, float)

    def test_negative_power_system(self):
        """A negative-power system returns a negative focal length."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=-100.0,
            radius_of_curvature_2=100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=-100.0,
            radius_of_curvature_2=100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system = OpticalSystem()
        system.add_lens(L1)
        system.add_lens(L2, air_gap_before=5.0)

        efl = system.get_system_focal_length()
        assert efl is not None
        assert efl < 0

    def test_afocal_system_returns_none(self, monkeypatch):
        """An afocal system (where C ≈ 0 in ABCD matrix) returns None."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system = OpticalSystem()
        system.add_lens(L1)
        system.add_lens(L2, air_gap_before=5.0)

        # Monkeypatch matrix to have C = 0 (afocal case)
        def afocal_matrix(n_overrides=None):
            return (1.0, 0.0, 0.0, 1.0)

        monkeypatch.setattr(system, '_calculate_system_matrix', afocal_matrix)

        assert system.get_system_focal_length() is None

    def test_three_element_system(self):
        """A three-lens system computes focal length using the ABCD matrix."""
        L1 = Lens(
            name="L1",
            radius_of_curvature_1=80.0,
            radius_of_curvature_2=-80.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        L2 = Lens(
            name="L2",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=4.0,
            diameter=20.0,
            refractive_index=1.6,
        )
        L3 = Lens(
            name="L3",
            radius_of_curvature_1=90.0,
            radius_of_curvature_2=-90.0,
            thickness=5.0,
            diameter=22.0,
            refractive_index=1.5,
        )
        system = OpticalSystem()
        system.add_lens(L1)
        system.add_lens(L2, air_gap_before=8.0)
        system.add_lens(L3, air_gap_before=12.0)

        efl = system.get_system_focal_length()
        assert efl is not None
        assert isinstance(efl, float)

    def test_tilt_does_not_affect_focal_length(self):
        """Element tilt is ignored in focal length calculation."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system = OpticalSystem()
        system.add_lens(L1)
        system.add_lens(L2, air_gap_before=10.0)

        efl_no_tilt = system.get_system_focal_length()

        # Add tilt to second element
        system.elements[1].tilt_x = 5.0
        efl_with_tilt = system.get_system_focal_length()

        assert efl_no_tilt == efl_with_tilt

    def test_decenter_does_not_affect_focal_length(self):
        """Element decenter is ignored in focal length calculation."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system = OpticalSystem()
        system.add_lens(L1)
        system.add_lens(L2, air_gap_before=10.0)

        efl_no_decenter = system.get_system_focal_length()

        # Add decenter to second element
        system.elements[1].decenter_x = 2.0
        efl_with_decenter = system.get_system_focal_length()

        assert efl_no_decenter == efl_with_decenter

    def test_focal_length_not_cached_recomputed_on_each_call(self):
        """Focal length is recomputed on each call; changes are reflected immediately."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system = OpticalSystem()
        system.add_lens(L1)
        system.add_lens(L2, air_gap_before=10.0)

        efl1 = system.get_system_focal_length()

        # Modify gap
        system.air_gaps[0].thickness = 15.0
        efl2 = system.get_system_focal_length()

        # Restore gap
        system.air_gaps[0].thickness = 10.0
        efl3 = system.get_system_focal_length()

        # First and third should match
        assert efl1 == efl3
        # Second should be different
        assert efl1 != efl2

    def test_thickness_affects_focal_length_via_abcd_matrix(self):
        """Changing lens thickness affects the system focal length."""
        L1 = Lens(
            name="a",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        L2 = Lens(
            name="b",
            radius_of_curvature_1=100.0,
            radius_of_curvature_2=-100.0,
            thickness=5.0,
            diameter=25.0,
            refractive_index=1.5,
        )
        system = OpticalSystem()
        system.add_lens(L1)
        system.add_lens(L2, air_gap_before=10.0)

        efl_thin = system.get_system_focal_length()

        # Increase thickness of second lens
        L2.thickness = 10.0
        efl_thick = system.get_system_focal_length()

        # Should be different due to thickness term in ABCD matrix
        assert efl_thin != efl_thick
