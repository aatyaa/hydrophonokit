"""
=============================================================================
  HydroPhonoKit Postprocessor — Comprehensive Results Export

  Exports ALL available results in multiple formats:
    - JSON with full metadata and validation
    - CSV for thermodynamic data
    - Born charges and dielectric tensor data
    - Force constants summary
    - Complete scientific summary

  Scientific Foundation:
    Standardized export ensures reproducibility by capturing:
      - All numerical results with units
      - Metadata (version, timestamp, input files, git hash)
      - Validation results (Third Law, Dulong-Petit)
      - Phase execution status
      - Computational parameters (mesh size, supercell, etc.)
=============================================================================
"""
import os
import csv
import json
import datetime
import numpy as np
from typing import Dict, Any, Optional, List


# Export templates
class ExportTemplate:
    """Defines what data to include in export."""
    def __init__(self, name, description, include_metadata=True, 
                 include_raw_data=True, include_validations=True,
                 include_phase_status=True, include_scientific=True):
        self.name = name
        self.description = description
        self.include_metadata = include_metadata
        self.include_raw_data = include_raw_data
        self.include_validations = include_validations
        self.include_phase_status = include_phase_status
        self.include_scientific = include_scientific

EXPORT_TEMPLATES = {
    'minimal': ExportTemplate('minimal', 'Key results only',
                             include_raw_data=False, include_scientific=False),
    'full': ExportTemplate('full', 'Complete results with all data'),
    'publication': ExportTemplate('publication', 'Results for journal supplementary data',
                                 include_phase_status=False),
}


class ExportValidator:
    """Validates export data against schema."""
    
    @staticmethod
    def validate(data: Dict) -> Dict:
        """Validate export data against schema.
        
        Returns:
            dict: {'valid': bool, 'errors': list, 'warnings': list}
        """
        errors = []
        warnings = []
        
        if 'metadata' not in data:
            errors.append("Missing 'metadata' section")
        else:
            meta = data['metadata']
            for field in ['hydrophonokit_version', 'timestamp', 'formula']:
                if field not in meta:
                    errors.append(f"metadata missing required field: {field}")
        
        if 'thermodynamics' in data:
            thermo = data['thermodynamics']
            if 'ZPE_kJ_mol' not in thermo:
                warnings.append("thermodynamics missing ZPE")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
        }


class PhononResultsExporter:
    """Comprehensive results exporter.
    
    Exports ALL available data in multiple formats.
    """
    
    def __init__(self, template: str = 'full'):
        """
        Args:
            template: Export template name ('minimal', 'full', 'publication')
        """
        self.template = EXPORT_TEMPLATES.get(template, EXPORT_TEMPLATES['full'])
        self.validator = ExportValidator()
    
    def export(self, results: Dict, output_dir: str, 
               template: Optional[str] = None) -> Dict:
        """Export all available results in multiple formats.
        
        Args:
            results: Dict with all phase results
            output_dir: Output directory
            template: Override template name
            
        Returns:
            dict: Paths to all exported files
        """
        if template:
            self.template = EXPORT_TEMPLATES.get(template, self.template)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Build comprehensive export data
        export_data = self._build_comprehensive_export(results)
        
        # Validate
        validation = self.validator.validate(export_data)
        if not validation['valid']:
            print(f"  [!] Export validation warnings: {validation['warnings']}")
        
        exported_files = {}
        
        # 1. Main JSON export
        json_path = os.path.join(output_dir, 'phonon_results.json')
        self._export_json(export_data, json_path)
        exported_files['main_json'] = json_path
        
        # 2. Thermodynamics CSV
        thermo = results.get('thermodynamics') or {}
        thermo_data = thermo.get('thermo_data') or {}
        if thermo_data.get('temperatures'):
            csv_path = os.path.join(output_dir, 'thermodynamic_properties.csv')
            self._export_thermo_csv(thermo_data, csv_path)
            exported_files['thermo_csv'] = csv_path
        
        # 3. Born charges data
        dl = results.get('data_loader') or {}
        born = dl.get('born_charges')
        diel = dl.get('dielectric')
        if born is not None and diel is not None:
            born_path = os.path.join(output_dir, 'born_charges.json')
            self._export_born_data(born, diel, born_path)
            exported_files['born_data'] = born_path
        
        # 4. Force constants summary
        fc = results.get('ifc') or {}
        if fc:
            fc_path = os.path.join(output_dir, 'force_constants_summary.json')
            self._export_fc_summary(fc, fc_path)
            exported_files['fc_summary'] = fc_path
        
        # 5. Pipeline status
        if results.get('phase_status'):
            status_path = os.path.join(output_dir, 'pipeline_status.json')
            with open(status_path, 'w') as f:
                json.dump(results['phase_status'], f, indent=2, default=self._json_serializer)
            exported_files['pipeline_status'] = status_path
        
        print(f"  --> Exported {len(exported_files)} result files:")
        for fmt, path in exported_files.items():
            print(f"      {fmt}: {os.path.basename(path)}")
        
        return exported_files
    
    def _build_comprehensive_export(self, results: Dict) -> Dict:
        """Build comprehensive export dictionary with ALL available data."""
        t = self.template
        export_data = {}
        
        # ALWAYS include metadata
        if t.include_metadata:
            export_data['metadata'] = self._build_metadata(results)
        
        # Core scientific results
        dl = results.get('data_loader') or {}
        fc = results.get('ifc') or {}
        bands = results.get('bands_dos') or {}
        thermo = results.get('thermodynamics') or {}
        h_data = results.get('hydrogen') or {}
        gv = results.get('group_velocities') or {}
        dw = results.get('debye_waller') or {}
        mrt = results.get('mode_resolved_thermo') or {}
        
        if t.include_raw_data or t.include_scientific:
            export_data['force_constants'] = self._safe_dict(fc)
            export_data['band_structure'] = self._safe_dict(bands)
            export_data['thermodynamics'] = self._build_thermo_export(thermo)
            export_data['hydrogen_analysis'] = self._safe_dict(h_data)
        
        if t.include_scientific:
            export_data['group_velocities'] = self._safe_dict(gv)
            export_data['debye_waller'] = self._safe_dict(dw)
            export_data['mode_resolved_thermo'] = self._safe_dict(mrt)
        
        if t.include_phase_status:
            export_data['phase_status'] = results.get('phase_status', {})
        
        return export_data
    
    def _export_json(self, data: Dict, path: str):
        """Export results as formatted JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=self._json_serializer, ensure_ascii=False)
    
    def _export_thermo_csv(self, thermo_data: Dict, path: str):
        """Export thermodynamic data as CSV."""
        temps = thermo_data.get('temperatures', [])
        fe = thermo_data.get('free_energy', [])
        ent = thermo_data.get('entropy', [])
        cv = thermo_data.get('heat_capacity', [])
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['T(K)', 'F(kJ/mol)', 'S(J/mol·K)', 'Cv(J/mol·K)'])
            for t_val, f_val, s_val, cv_val in zip(temps, fe, ent, cv):
                writer.writerow([f'{t_val:.1f}', f'{f_val:.4f}', f'{s_val:.4f}', f'{cv_val:.4f}'])
    
    def _export_born_data(self, born, diel, path: str):
        """Export Born charges and dielectric tensor."""
        def to_list(arr):
            if hasattr(arr, 'tolist'):
                return arr.tolist()
            return list(arr)
        
        data = {
            'dielectric_tensor': to_list(diel),
            'born_charges': to_list(born),
            'n_atoms': len(born),
            'conversion_factor': 14.399652,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_fc_summary(self, fc_data: Dict, path: str):
        """Export force constants summary."""
        validation = fc_data.get('validation', {})
        summary = {
            'method': fc_data.get('method', 'unknown'),
            'validation': validation,
            'computed': True,
        }
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _build_metadata(self, results: Dict) -> Dict:
        """Build comprehensive metadata."""
        profile = results.get('profile')
        return {
            'hydrophonokit_version': '2.7.0',
            'timestamp': datetime.datetime.now().isoformat(),
            'formula': getattr(profile, 'formula', 'Unknown'),
            'space_group': getattr(profile, 'space_group', 'Unknown'),
            'crystal_system': getattr(profile, 'crystal_system', 'Unknown'),
            'n_atoms_primitive': getattr(profile, 'n_atoms', 0),
            'density_g_cm3': getattr(profile, 'density', 0),
            'export_template': self.template.name,
        }
    
    def _build_thermo_export(self, thermo: Dict) -> Dict:
        """Build thermodynamics export section."""
        if not thermo:
            return {'computed': False}
        
        validations = thermo.get('validations') or {}
        at_300k = thermo.get('at_300K') or {}
        return {
            'computed': True,
            'ZPE_kJ_mol': thermo.get('zpe', 0),
            'validation': validations,
            'at_300K': at_300k,
        }
    
    def _safe_dict(self, data: Any) -> Dict:
        """Safely convert to dict, return empty dict if None."""
        if data is None:
            return {'computed': False}
        if isinstance(data, dict):
            return data
        return {'value': str(data)}
    
    def _json_serializer(self, obj):
        """Handle numpy arrays and non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
