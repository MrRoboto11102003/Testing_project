import pytest
import os
import tempfile
from pathlib import Path
from openlens.material_database import MaterialDatabase


class TestMaterialDatabaseImportAGF:
    """Tests for MaterialDatabase.import_agf_catalog method."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path for isolated testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "materials.json"

    @pytest.fixture
    def db(self, temp_db_path):
        """Create a fresh MaterialDatabase instance with temp storage."""
        return MaterialDatabase(db_path=str(temp_db_path))

    def test_nonexistent_file_returns_zero(self, db):
        """Non-existent file returns 0 without raising."""
        result = db.import_agf_catalog("/nonexistent/path/catalog.agf")
        assert result == 0

    def test_empty_file_returns_zero_no_save(self, db, temp_db_path):
        """Empty AGF file returns 0 and does not create database file."""
        agf_file = temp_db_path.parent / "empty.agf"
        agf_file.write_text("", encoding="utf-8")
        
        result = db.import_agf_catalog(str(agf_file))
        assert result == 0
        assert not os.path.exists(str(temp_db_path))

    def test_single_valid_glass_imported(self, db, temp_db_path):
        """Single valid glass is imported and counts as 1."""
        agf_file = temp_db_path.parent / "single.agf"
        agf_file.write_text(
            "NM TESTGLASS 2 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n",
            encoding="utf-8"
        )
        
        result = db.import_agf_catalog(str(agf_file))
        assert result == 1
        assert os.path.exists(str(temp_db_path))
        
        glass = db.get_material("TESTGLASS")
        assert glass is not None
        assert glass.nd == pytest.approx(1.5168)
        assert glass.vd == pytest.approx(64.17)

    def test_multiple_glasses_imported(self, db, temp_db_path):
        """Multiple valid glasses all imported and counted."""
        agf_file = temp_db_path.parent / "multi.agf"
        agf_file.write_text(
            "NM GLASS1 2 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n"
            "NM GLASS2 2 517643 1.6 50.0 8.0 2.5\n"
            "CD 1.1 0.007 0.25 0.02 1.0 100.0\n",
            encoding="utf-8"
        )
        
        result = db.import_agf_catalog(str(agf_file))
        assert result == 2
        
        assert db.get_material("GLASS1") is not None
        assert db.get_material("GLASS2") is not None

    def test_dispersion_coefficients_deinterleaved(self, db, temp_db_path):
        """CD line coefficients are correctly de-interleaved to B/C pairs."""
        agf_file = temp_db_path.parent / "disp.agf"
        agf_file.write_text(
            "NM DISPGLASS 2 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n",
            encoding="utf-8"
        )
        
        db.import_agf_catalog(str(agf_file))
        glass = db.get_material("DISPGLASS")
        
        assert glass.B1 == pytest.approx(1.03961212)
        assert glass.C1 == pytest.approx(6.00069867e-3)
        assert glass.B2 == pytest.approx(0.231792344)
        assert glass.C2 == pytest.approx(2.00179144e-2)
        assert glass.B3 == pytest.approx(1.01046945)
        assert glass.C3 == pytest.approx(103.560653)

    def test_thermal_expansion_converted_from_ppm_to_per_kelvin(self, db, temp_db_path):
        """TCE from NM line converted from 1e-6/K units to K^-1."""
        agf_file = temp_db_path.parent / "thermal.agf"
        agf_file.write_text(
            "NM THERMGLASS 2 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n",
            encoding="utf-8"
        )
        
        db.import_agf_catalog(str(agf_file))
        glass = db.get_material("THERMGLASS")
        assert glass.thermal_expansion == pytest.approx(7.1e-6)

    def test_ed_line_overrides_tce_and_density(self, db, temp_db_path):
        """ED line TCE and density override NM line values."""
        agf_file = temp_db_path.parent / "ed.agf"
        agf_file.write_text(
            "NM EDGLASS 2 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n"
            "ED 8.5 0 0 2.6\n",
            encoding="utf-8"
        )
        
        db.import_agf_catalog(str(agf_file))
        glass = db.get_material("EDGLASS")
        
        assert glass.thermal_expansion == pytest.approx(8.5e-6)
        assert glass.density == pytest.approx(2.6)

    def test_transmission_data_accumulates_per_wavelength(self, db, temp_db_path):
        """Multiple IT lines accumulate transmission data with last thickness wins."""
        agf_file = temp_db_path.parent / "trans.agf"
        agf_file.write_text(
            "NM TRANSGLASS 2 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n"
            "IT 400 0.9 10\n"
            "IT 500 0.95 25\n",
            encoding="utf-8"
        )
        
        db.import_agf_catalog(str(agf_file))
        glass = db.get_material("TRANSGLASS")
        
        assert len(glass.transmission_data) == 2
        assert glass.transmission_data[0][0] == pytest.approx(400.0)
        assert glass.transmission_data[0][1] == pytest.approx(0.9)
        assert glass.transmission_data[1][0] == pytest.approx(500.0)
        assert glass.transmission_data[1][1] == pytest.approx(0.95)
        assert glass.transmission_thickness == pytest.approx(25.0)

    def test_invalid_transmission_values_rejected_in_range_check(self, db, temp_db_path):
        """Transmission values outside [0, 1] are rejected and logged."""
        agf_file = temp_db_path.parent / "badtrans.agf"
        agf_file.write_text(
            "NM BADTRANS 2 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n"
            "IT 400 1.5 10\n"
            "IT 500 0.95 25\n",
            encoding="utf-8"
        )
        
        db.import_agf_catalog(str(agf_file))
        glass = db.get_material("BADTRANS")
        
        assert len(glass.transmission_data) == 1
        assert glass.transmission_data[0][0] == pytest.approx(500.0)

    def test_insufficient_cd_coefficients_glass_skipped(self, db, temp_db_path):
        """Glass with fewer than 6 CD coefficients is skipped, count=0."""
        agf_file = temp_db_path.parent / "badcd.agf"
        agf_file.write_text(
            "NM BADCD 2 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344\n",
            encoding="utf-8"
        )
        
        result = db.import_agf_catalog(str(agf_file))
        assert result == 0
        assert db.get_material("BADCD") is None

    def test_invalid_formula_glass_skipped(self, db, temp_db_path):
        """Glass with formula not in {1, 2} is skipped."""
        agf_file = temp_db_path.parent / "badformula.agf"
        agf_file.write_text(
            "NM BADFORMULA 3 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n",
            encoding="utf-8"
        )
        
        result = db.import_agf_catalog(str(agf_file))
        assert result == 0
        assert db.get_material("BADFORMULA") is None

    def test_case_insensitive_line_keys(self, db, temp_db_path):
        """Line keys (NM, CD, ED, IT) are case-insensitive."""
        agf_file = temp_db_path.parent / "case.agf"
        agf_file.write_text(
            "nm CASETEST 2 517642 1.5168 64.17 7.1 2.51\n"
            "cd 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n"
            "it 500 0.95\n",
            encoding="utf-8"
        )
        
        result = db.import_agf_catalog(str(agf_file))
        assert result == 1
        assert db.get_material("CASETEST") is not None

    def test_duplicate_names_latest_overwrites(self, db, temp_db_path):
        """Later glass definition with same name overwrites earlier one."""
        agf_file = temp_db_path.parent / "dup.agf"
        agf_file.write_text(
            "NM DUPGLASS 2 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n"
            "NM DUPGLASS 2 517643 1.6 50.0 8.0 2.5\n"
            "CD 1.1 0.007 0.25 0.02 1.0 100.0\n",
            encoding="utf-8"
        )
        
        result = db.import_agf_catalog(str(agf_file))
        assert result == 2
        
        glass = db.get_material("DUPGLASS")
        assert glass.nd == pytest.approx(1.6)
        assert glass.vd == pytest.approx(50.0)

    def test_catalog_name_set_to_agf_import(self, db, temp_db_path):
        """Imported glasses have catalog='AGF_Import'."""
        agf_file = temp_db_path.parent / "catalog.agf"
        agf_file.write_text(
            "NM CATTEST 2 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n",
            encoding="utf-8"
        )
        
        db.import_agf_catalog(str(agf_file))
        glass = db.get_material("CATTEST")
        assert glass.catalog == "AGF_Import"

    def test_persistence_across_database_instances(self, db, temp_db_path):
        """Imported glasses persist when database is reloaded from JSON."""
        agf_file = temp_db_path.parent / "persist.agf"
        agf_file.write_text(
            "NM PERSIST 2 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 6.00069867e-3 0.231792344 2.00179144e-2 1.01046945 103.560653\n",
            encoding="utf-8"
        )
        
        db.import_agf_catalog(str(agf_file))
        
        db2 = MaterialDatabase(db_path=str(temp_db_path))
        glass = db2.get_material("PERSIST")
        assert glass is not None
        assert glass.nd == pytest.approx(1.5168)
