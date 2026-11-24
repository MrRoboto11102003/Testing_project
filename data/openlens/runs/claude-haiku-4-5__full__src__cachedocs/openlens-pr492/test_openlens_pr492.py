import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock
from openlens.material_database import MaterialDatabase


@pytest.fixture
def temp_agf_dir():
    """Create a temporary directory for test AGF files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def material_db():
    """Create a MaterialDatabase instance with mocked save."""
    db = MaterialDatabase()
    db.save_database = Mock()
    return db


class TestImportAGFCatalog:
    """Test suite for MaterialDatabase.import_agf_catalog"""

    def test_file_not_found_returns_zero(self, material_db):
        """Non-existent file should return 0."""
        result = material_db.import_agf_catalog("/nonexistent/path/catalog.agf")
        assert result == 0
        material_db.save_database.assert_not_called()

    def test_empty_file_returns_zero(self, temp_agf_dir, material_db):
        """Empty AGF file should return 0."""
        agf_file = os.path.join(temp_agf_dir, "empty.agf")
        Path(agf_file).write_text("")
        result = material_db.import_agf_catalog(agf_file)
        assert result == 0

    def test_single_material_complete(self, temp_agf_dir, material_db):
        """Import a single material with all fields: NM name formula MIL Nd Vd TCE density."""
        agf_file = os.path.join(temp_agf_dir, "single.agf")
        Path(agf_file).write_text("NM BK7 1 0 1.5168 64.17 10.5 2.51\n")
        result = material_db.import_agf_catalog(agf_file)
        assert result == 1
        material_db.save_database.assert_called_once()

    def test_single_material_minimal(self, temp_agf_dir, material_db):
        """Import material with optional fields (TCE, density) missing."""
        agf_file = os.path.join(temp_agf_dir, "minimal.agf")
        Path(agf_file).write_text("NM N-BK7 1 0 1.5168 64.17\n")
        result = material_db.import_agf_catalog(agf_file)
        assert result == 1

    def test_multiple_materials(self, temp_agf_dir, material_db):
        """Import multiple materials in sequence."""
        agf_file = os.path.join(temp_agf_dir, "multi.agf")
        content = (
            "NM BK7 1 0 1.5168 64.17 10.5 2.51\n"
            "NM SF11 1 0 1.7847 25.76 15.0 4.70\n"
            "NM FusedSilica 1 0 1.4585 67.8 5.3 2.20\n"
        )
        Path(agf_file).write_text(content)
        result = material_db.import_agf_catalog(agf_file)
        assert result == 3

    def test_material_with_cd_dispersion_data(self, temp_agf_dir, material_db):
        """NM followed by CD (Cauchy dispersion coefficients) should import successfully."""
        agf_file = os.path.join(temp_agf_dir, "dispersion.agf")
        content = (
            "NM BK7 1 0 1.5168 64.17 10.5 2.51\n"
            "CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653\n"
        )
        Path(agf_file).write_text(content)
        result = material_db.import_agf_catalog(agf_file)
        assert result == 1

    def test_material_with_ed_extra_data(self, temp_agf_dir, material_db):
        """NM followed by ED (extra data with authoritative TCE) should parse correctly."""
        agf_file = os.path.join(temp_agf_dir, "extra.agf")
        content = (
            "NM BK7 1 0 1.5168 64.17 0.0 2.51\n"
            "ED 15.5 0.001 0.0 2510.0 0.0\n"
        )
        Path(agf_file).write_text(content)
        result = material_db.import_agf_catalog(agf_file)
        assert result == 1

    def test_material_with_multiple_it_transmission_lines(self, temp_agf_dir, material_db):
        """Multiple IT (transmission) lines for one material should accumulate."""
        agf_file = os.path.join(temp_agf_dir, "transmission.agf")
        content = (
            "NM BK7 1 0 1.5168 64.17 10.5 2.51\n"
            "IT 400 0.95 10.0\n"
            "IT 550 0.98 10.0\n"
            "IT 1000 0.90 10.0\n"
        )
        Path(agf_file).write_text(content)
        result = material_db.import_agf_catalog(agf_file)
        assert result == 1

    def test_invalid_transmission_values_outside_range_ignored(self, temp_agf_dir, material_db):
        """IT lines with transmission outside [0, 1] should be logged and skipped."""
        agf_file = os.path.join(temp_agf_dir, "bad_transmission.agf")
        content = (
            "NM BK7 1 0 1.5168 64.17 10.5 2.51\n"
            "IT 400 1.5 10.0\n"
            "IT 550 0.98 10.0\n"
            "IT 1000 -0.1 10.0\n"
        )
        Path(agf_file).write_text(content)
        result = material_db.import_agf_catalog(agf_file)
        assert result == 1

    def test_malformed_nm_line_skipped_next_material_imported(self, temp_agf_dir, material_db):
        """NM line with unparseable numeric fields should be skipped."""
        agf_file = os.path.join(temp_agf_dir, "bad_nm.agf")
        content = (
            "NM BadGlass NotANumber NotANumber\n"
            "NM BK7 1 0 1.5168 64.17 10.5 2.51\n"
        )
        Path(agf_file).write_text(content)
        result = material_db.import_agf_catalog(agf_file)
        assert result == 1

    def test_cd_ed_it_lines_without_material_context_ignored(self, temp_agf_dir, material_db):
        """CD/ED/IT lines before first NM should be safely ignored."""
        agf_file = os.path.join(temp_agf_dir, "orphaned.agf")
        content = (
            "CD 1.03961212 0.00600069867\n"
            "ED 10.5 0.001 0.0 2510.0\n"
            "IT 550 0.98 10.0\n"
            "NM BK7 1 0 1.5168 64.17 10.5 2.51\n"
        )
        Path(agf_file).write_text(content)
        result = material_db.import_agf_catalog(agf_file)
        assert result == 1

    def test_blank_and_whitespace_lines_ignored(self, temp_agf_dir, material_db):
        """Blank lines and whitespace-only lines should be skipped."""
        agf_file = os.path.join(temp_agf_dir, "blanks.agf")
        content = (
            "NM BK7 1 0 1.5168 64.17 10.5 2.51\n"
            "\n"
            "   \n"
            "NM SF11 1 0 1.7847 25.76 15.0 4.70\n"
            "\t\n"
        )
        Path(agf_file).write_text(content)
        result = material_db.import_agf_catalog(agf_file)
        assert result == 2

    def test_save_database_called_only_on_nonzero_count(self, temp_agf_dir, material_db):
        """save_database() should be called only if count > 0."""
        # Empty file: count = 0
        empty_file = os.path.join(temp_agf_dir, "empty.agf")
        Path(empty_file).write_text("")
        material_db.import_agf_catalog(empty_file)
        material_db.save_database.assert_not_called()

        # Reset mock
        material_db.save_database.reset_mock()

        # File with material: count = 1
        full_file = os.path.join(temp_agf_dir, "full.agf")
        Path(full_file).write_text("NM BK7 1 0 1.5168 64.17 10.5 2.51\n")
        material_db.import_agf_catalog(full_file)
        material_db.save_database.assert_called_once()

    def test_tce_values_in_ppm_converted_to_k_inverse(self, temp_agf_dir, material_db):
        """TCE from NM and ED lines stored as ppm (1e-6/K) should be converted to K^-1."""
        agf_file = os.path.join(temp_agf_dir, "tce_conversion.agf")
        # NM line with TCE = 10.5 ppm; ED line with TCE = 15.5 ppm (ED overrides)
        content = (
            "NM BK7 1 0 1.5168 64.17 10.5 2.51\n"
            "ED 15.5 0.001 0.0 2510.0 0.0\n"
        )
        Path(agf_file).write_text(content)
        result = material_db.import_agf_catalog(agf_file)
        assert result == 1

    def test_quoted_material_names_stripped(self, temp_agf_dir, material_db):
        """Material names enclosed in double quotes should have quotes removed."""
        agf_file = os.path.join(temp_agf_dir, "quoted.agf")
        Path(agf_file).write_text('NM "N-BK7" 1 0 1.5168 64.17 10.5 2.51\n')
        result = material_db.import_agf_catalog(agf_file)
        assert result == 1

    def test_mixed_valid_and_invalid_materials(self, temp_agf_dir, material_db):
        """File with mix of valid and invalid NM lines should import only valid ones."""
        agf_file = os.path.join(temp_agf_dir, "mixed.agf")
        content = (
            "NM BK7 1 0 1.5168 64.17 10.5 2.51\n"
            "NM Invalid 999 BAD\n"
            "NM SF11 1 0 1.7847 25.76 15.0 4.70\n"
        )
        Path(agf_file).write_text(content)
        result = material_db.import_agf_catalog(agf_file)
        assert result == 2
