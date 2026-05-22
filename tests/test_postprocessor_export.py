"""
Tests for HydroPhonoKit Postprocessor export functionality.

Run with: pytest tests/test_postprocessor_export.py -v
"""
import pytest
import os
import json
import csv
import tempfile
from unittest.mock import Mock, patch


# ============================================================================
# Test Export Templates
# ============================================================================

class TestExportTemplates:
    """Test export template definitions."""

    def test_minimal_template_exists(self):
        """Minimal template should be defined."""
        from hydrophonokit.postprocessor.enhanced_export import EXPORT_TEMPLATES
        assert 'minimal' in EXPORT_TEMPLATES

    def test_full_template_exists(self):
        """Full template should be defined."""
        from hydrophonokit.postprocessor.enhanced_export import EXPORT_TEMPLATES
        assert 'full' in EXPORT_TEMPLATES

    def test_publication_template_exists(self):
        """Publication template should be defined."""
        from hydrophonokit.postprocessor.enhanced_export import EXPORT_TEMPLATES
        assert 'publication' in EXPORT_TEMPLATES

    def test_template_has_required_attributes(self):
        """Each template should have all required attributes."""
        from hydrophonokit.postprocessor.enhanced_export import EXPORT_TEMPLATES, ExportTemplate
        
        required_attrs = ['name', 'description', 'include_metadata', 
                         'include_raw_data', 'include_validations',
                         'include_phase_status', 'include_scientific']
        
        for name, template in EXPORT_TEMPLATES.items():
            for attr in required_attrs:
                assert hasattr(template, attr), f"Template '{name}' missing '{attr}'"

    def test_minimal_template_excludes_raw_data(self):
        """Minimal template should exclude raw data."""
        from hydrophonokit.postprocessor.enhanced_export import EXPORT_TEMPLATES
        assert EXPORT_TEMPLATES['minimal'].include_raw_data is False

    def test_full_template_includes_all(self):
        """Full template should include everything."""
        from hydrophonokit.postprocessor.enhanced_export import EXPORT_TEMPLATES
        tmpl = EXPORT_TEMPLATES['full']
        assert tmpl.include_metadata is True
        assert tmpl.include_raw_data is True
        assert tmpl.include_validations is True
        assert tmpl.include_phase_status is True
        assert tmpl.include_scientific is True


# ============================================================================
# Test Export Validator
# ============================================================================

class TestExportValidator:
    """Test export data validation."""

    def test_valid_data_passes_validation(self):
        """Valid data should pass validation."""
        from hydrophonokit.postprocessor.enhanced_export import ExportValidator
        
        valid_data = {
            'metadata': {
                'hydrophonokit_version': '2.7.0',
                'timestamp': '2026-04-12T12:00:00',
                'formula': 'Si',
            },
            'band_structure': {
                'min_frequency_THz': 0.5,
                'is_stable': True,
            },
            'thermodynamics': {
                'ZPE_kJ_mol': 10.5,
                'at_300K': {'F': -5.0, 'S': 50.0, 'Cv': 25.0},
            }
        }
        
        validator = ExportValidator()
        result = validator.validate(valid_data)
        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_missing_metadata_fails_validation(self):
        """Missing metadata should fail validation."""
        from hydrophonokit.postprocessor.enhanced_export import ExportValidator
        
        invalid_data = {
            'band_structure': {'min_frequency_THz': 0.5},
        }
        
        validator = ExportValidator()
        result = validator.validate(invalid_data)
        assert result['valid'] is False
        assert any('metadata' in err for err in result['errors'])

    def test_missing_required_fields_fails(self):
        """Missing required fields should fail validation."""
        from hydrophonokit.postprocessor.enhanced_export import ExportValidator
        
        invalid_data = {
            'metadata': {'hydrophonokit_version': '2.7.0'},  # Missing timestamp, formula
            'band_structure': {},  # Missing required fields
            'thermodynamics': {},  # Missing required fields
        }
        
        validator = ExportValidator()
        result = validator.validate(invalid_data)
        assert result['valid'] is False


# ============================================================================
# Test PhononResultsExporter
# ============================================================================

class TestPhononResultsExporter:
    """Test PhononResultsExporter functionality."""

    def _make_mock_results(self):
        """Create mock results dictionary."""
        profile = Mock()
        profile.formula = "Si"
        profile.space_group = "Fd-3m (227)"
        profile.crystal_system = "Cubic"
        profile.n_atoms = 2
        profile.density = 2.33

        return {
            'profile': profile,
            'ifc': {
                'method': 'symfc',
                'validation': {
                    'quality': 'GOOD',
                    'max_fc_eV_A2': 45.6,
                    'asr_error_eV_A3': 1e-6,
                }
            },
            'bands_dos': {
                'min_freq': 0.5,
                'stability': {'stable': True, 'n_imaginary': 0},
            },
            'thermodynamics': {
                'zpe': 10.5,
                'validations': {'third_law_ok': True},
                'at_300K': {'F_kJ_mol': -5.0, 'S_J_molK': 50.0, 'Cv_J_molK': 25.0},
                'thermo_data': {
                    'temperatures': [0, 100, 200, 300],
                    'free_energy': [-10.5, -8.0, -6.0, -5.0],
                    'entropy': [0.0, 20.0, 35.0, 50.0],
                    'heat_capacity': [0.0, 15.0, 22.0, 25.0],
                }
            },
            'hydrogen': None,
            'group_velocities': {
                'avg_sound_velocity': 5800.0,
                'max_group_velocity': 9400.0,
            },
            'debye_waller': {
                'temperature': 300,
                'isotropic_u2': [0.01, 0.02],
                'B_factor': [0.3, 0.6],
                'zero_point_u2': [0.005, 0.01],
            },
            'mode_resolved_thermo': {
                'element_fractions': {'Si': 100.0},
                'hydrogen_entropy_contribution': 0.0,
            },
            'phase_status': {
                'data_collection': {'completed': True},
                'force_constants': {'completed': True},
            },
        }

    def test_init_default_template(self):
        """Should initialize with 'full' template by default."""
        from hydrophonokit.postprocessor.enhanced_export import PhononResultsExporter
        exporter = PhononResultsExporter()
        assert exporter.template.name == 'full'

    def test_init_custom_template(self):
        """Should initialize with specified template."""
        from hydrophonokit.postprocessor.enhanced_export import PhononResultsExporter
        exporter = PhononResultsExporter(template='minimal')
        assert exporter.template.name == 'minimal'

    def test_export_json(self):
        """Should export results as JSON."""
        from hydrophonokit.postprocessor.enhanced_export import PhononResultsExporter
        
        exporter = PhononResultsExporter()
        results = self._make_mock_results()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exported_files = exporter.export(results, tmpdir)
            
            assert 'main_json' in exported_files
            assert os.path.exists(exported_files['main_json'])
            
            # Verify JSON is valid
            with open(exported_files['main_json']) as f:
                data = json.load(f)
            
            assert 'metadata' in data
            assert data['metadata']['formula'] == 'Si'

    def test_export_csv(self):
        """Should export thermodynamic data as CSV."""
        from hydrophonokit.postprocessor.enhanced_export import PhononResultsExporter
        
        exporter = PhononResultsExporter()
        results = self._make_mock_results()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exported_files = exporter.export(results, tmpdir)
            
            assert 'thermo_csv' in exported_files
            assert os.path.exists(exported_files['thermo_csv'])
            
            # Verify CSV is valid
            with open(exported_files['thermo_csv'], encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # Header row + data rows
            assert len(rows) == 5  # Header + 4 data rows
            assert rows[0] == ['T(K)', 'F(kJ/mol)', 'S(J/mol·K)', 'Cv(J/mol·K)']

    def test_export_minimal_template(self):
        """Minimal template should exclude raw data."""
        from hydrophonokit.postprocessor.enhanced_export import PhononResultsExporter
        
        exporter = PhononResultsExporter(template='minimal')
        results = self._make_mock_results()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exported_files = exporter.export(results, tmpdir)
            
            # Should still export JSON
            assert 'main_json' in exported_files
            
            # Verify minimal content
            with open(exported_files['main_json']) as f:
                data = json.load(f)
            
            # Should have metadata but no raw arrays
            assert 'metadata' in data

    def test_export_publication_template(self):
        """Publication template should exclude phase status."""
        from hydrophonokit.postprocessor.enhanced_export import PhononResultsExporter
        
        exporter = PhononResultsExporter(template='publication')
        results = self._make_mock_results()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exported_files = exporter.export(results, tmpdir)
            
            with open(exported_files['main_json']) as f:
                data = json.load(f)
            
            # Should not have phase_status
            assert 'phase_status' not in data
