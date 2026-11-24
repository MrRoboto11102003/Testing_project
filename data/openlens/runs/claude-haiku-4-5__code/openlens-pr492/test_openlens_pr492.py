import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
import openlens
from openlens.material_database import MaterialDatabase


class TestImportAGFCatalog:
    """Test suite for MaterialDatabase.import_agf_catalog"""

    @pytest.fixture
    def db(self):
        """Create a MaterialDatabase instance"""
        return MaterialDatabase()

    def test_import_nonexistent_file_returns_zero(self, db):
        """Test importing from non-existent file returns 0"""
        result = db.import_agf_catalog("/nonexistent/path/file.agf")
        assert result == 0

    def test_import_empty_file_returns_zero(self, db):
        """Test importing an empty AGF file returns 0"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            temp_file = f.name
        try:
            result = db.import_agf_catalog(temp_file)
            assert result == 0
        finally:
            os.unlink(temp_file)

    def test_import_single_valid_material(self, db):
        """Test importing a single valid material with NM line"""
        agf_content = "NM N-BK7 1 517642 1.5168 64.17 7.1 2.51\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(agf_content)
            temp_file = f.name
        try:
            with patch.object(db, '_process_agf_glass', return_value=True):
                result = db.import_agf_catalog(temp_file)
            assert result == 1
        finally:
            os.unlink(temp_file)

    def test_import_multiple_materials(self, db):
        """Test importing multiple materials from one file"""
        agf_content = (
            "NM N-BK7 1 517642 1.5168 64.17 7.1 2.51\n"
            "NM N-SF5 1 517642 1.6727 32.25 7.3 3.75\n"
            "NM N-FK51A 1 517642 1.4860 84.47 10.1 2.33\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(agf_content)
            temp_file = f.name
        try:
            with patch.object(db, '_process_agf_glass', return_value=True):
                result = db.import_agf_catalog(temp_file)
            assert result == 3
        finally:
            os.unlink(temp_file)

    def test_import_material_with_quoted_name(self, db):
        """Test importing material with quoted name is properly stripped"""
        agf_content = 'NM "N-BK7" 1 517642 1.5168 64.17 7.1 2.51\n'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(agf_content)
            temp_file = f.name
        try:
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                db.import_agf_catalog(temp_file)
            call_args = mock_process.call_args[0][0]
            assert call_args['name'] == 'N-BK7'
        finally:
            os.unlink(temp_file)

    def test_import_material_with_dispersion_coefficients(self, db):
        """Test CD line parsing and storage in dispersion field"""
        agf_content = (
            "NM N-BK7 1 517642 1.5168 64.17 7.1 2.51\n"
            "CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(agf_content)
            temp_file = f.name
        try:
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                db.import_agf_catalog(temp_file)
            call_args = mock_process.call_args[0][0]
            assert len(call_args['dispersion']) == 6
            assert call_args['dispersion'][0] == 1.03961212
        finally:
            os.unlink(temp_file)

    def test_import_ed_line_overrides_tce(self, db):
        """Test ED line overrides TCE from NM line"""
        agf_content = (
            "NM N-BK7 1 517642 1.5168 64.17 0.0 2.51\n"
            "ED 7.5 0.0 0.0 1.5168\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(agf_content)
            temp_file = f.name
        try:
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                db.import_agf_catalog(temp_file)
            call_args = mock_process.call_args[0][0]
            # TCE converted from 1e-6/K to K^-1
            assert call_args['tce'] == 7.5e-6
        finally:
            os.unlink(temp_file)

    def test_import_multiple_transmission_it_lines(self, db):
        """Test IT line parsing accumulates transmission data"""
        agf_content = (
            "NM N-BK7 1 517642 1.5168 64.17 7.1 2.51\n"
            "IT 365 0.95 10.0\n"
            "IT 405 0.98 10.0\n"
            "IT 546 0.99 10.0\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(agf_content)
            temp_file = f.name
        try:
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                db.import_agf_catalog(temp_file)
            call_args = mock_process.call_args[0][0]
            assert len(call_args['transmission']) == 3
            assert call_args['transmission'][0] == (365.0, 0.95)
        finally:
            os.unlink(temp_file)

    def test_import_ignores_invalid_transmission_values(self, db):
        """Test transmission values outside [0, 1] are ignored with warning"""
        agf_content = (
            "NM N-BK7 1 517642 1.5168 64.17 7.1 2.51\n"
            "IT 365 1.5 10.0\n"
            "IT 405 0.98 10.0\n"
            "IT 546 -0.1 10.0\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(agf_content)
            temp_file = f.name
        try:
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                db.import_agf_catalog(temp_file)
            call_args = mock_process.call_args[0][0]
            # Only valid transmission (0.98) should be included
            assert len(call_args['transmission']) == 1
            assert call_args['transmission'][0][1] == 0.98
        finally:
            os.unlink(temp_file)

    def test_import_missing_optional_fields_defaults_to_zero(self, db):
        """Test missing optional fields in NM line default to 0.0"""
        agf_content = "NM N-BK7 1 517642 1.5168 64.17\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(agf_content)
            temp_file = f.name
        try:
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                db.import_agf_catalog(temp_file)
            call_args = mock_process.call_args[0][0]
            assert call_args['tce'] == 0.0
            assert call_args['density'] == 0.0
        finally:
            os.unlink(temp_file)

    def test_import_skips_malformed_nm_lines(self, db):
        """Test malformed NM lines are skipped, valid ones imported"""
        agf_content = (
            "NM N-BK7 INVALID 517642 1.5168 64.17\n"
            "NM N-SF5 1 517642 1.6727 32.25 7.3 3.75\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(agf_content)
            temp_file = f.name
        try:
            with patch.object(db, '_process_agf_glass', return_value=True):
                result = db.import_agf_catalog(temp_file)
            # Only valid NM line should result in call to _process_agf_glass
            assert result == 1
        finally:
            os.unlink(temp_file)

    def test_import_skips_blank_lines_and_continues(self, db):
        """Test blank and whitespace lines are skipped"""
        agf_content = (
            "NM N-BK7 1 517642 1.5168 64.17 7.1 2.51\n"
            "\n"
            "   \n"
            "NM N-SF5 1 517642 1.6727 32.25 7.3 3.75\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(agf_content)
            temp_file = f.name
        try:
            with patch.object(db, '_process_agf_glass', return_value=True):
                result = db.import_agf_catalog(temp_file)
            assert result == 2
        finally:
            os.unlink(temp_file)

    def test_import_saves_database_after_successful_import(self, db):
        """Test save_database is called once after import completes"""
        agf_content = "NM N-BK7 1 517642 1.5168 64.17 7.1 2.51\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(agf_content)
            temp_file = f.name
        try:
            with patch.object(db, '_process_agf_glass', return_value=True):
                with patch.object(db, 'save_database') as mock_save:
                    db.import_agf_catalog(temp_file)
            mock_save.assert_called_once()
        finally:
            os.unlink(temp_file)

    def test_import_does_not_save_when_count_is_zero(self, db):
        """Test save_database is not called when no materials imported"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            temp_file = f.name
        try:
            with patch.object(db, 'save_database') as mock_save:
                result = db.import_agf_catalog(temp_file)
            assert result == 0
            mock_save.assert_not_called()
        finally:
            os.unlink(temp_file)

    def test_import_handles_missing_dispersion_gracefully(self, db):
        """Test materials without CD lines still import successfully"""
        agf_content = (
            "NM N-BK7 1 517642 1.5168 64.17 7.1 2.51\n"
            "NM N-SF5 1 517642 1.6727 32.25 7.3 3.75\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.agf', delete=False) as f:
            f.write(agf_content)
            temp_file = f.name
        try:
            with patch.object(db, '_process_agf_glass', return_value=True) as mock_process:
                result = db.import_agf_catalog(temp_file)
            assert result == 2
            # Both materials should have empty dispersion lists
            for call in mock_process.call_args_list:
                assert call[0][0]['dispersion'] == []
        finally:
            os.unlink(temp_file)
