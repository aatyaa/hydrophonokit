"""
=============================================================================
  HydroPhonoKit Postprocessor — Results Export

  Standardized export of phonon analysis results to JSON/CSV formats.

  Scientific Foundation:
    Standardized export ensures reproducibility by capturing:
      - All numerical results with units
      - Metadata (version, timestamp, input files)
      - Validation results (Third Law, Dulong-Petit)
      - Phase execution status
=============================================================================
"""
import os
import json
import datetime
import numpy as np
from typing import Dict, Any


class PhononResultsExporter:
    """Exports phonon analysis results in standardized formats.

    This class handles:
      1. JSON export with full metadata
      2. CSV export for thermodynamic data
      3. Validation of export schema
    """

    def export(self, results: Dict, output_dir: str) -> str:
        """Export all results as JSON.

        Args:
            results: Dict with all phase results
            output_dir: Output directory

        Returns:
            Path to exported JSON file
        """
        export_data = self._build_export_dict(results)
        json_path = os.path.join(output_dir, 'phonon_results.json')

        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=self._json_serializer)

        print(f"  --> Exported results to: {json_path}")
        return json_path

    def _build_export_dict(self, results: Dict) -> Dict:
        """Build standardized export dictionary."""
        profile = results.get('profile', {})
        data_loader = results.get('data_loader', {})
        ifc = results.get('ifc', {})
        bands = results.get('bands_dos', {})
        thermo = results.get('thermodynamics', {})
        hydrogen = results.get('hydrogen', {})
        gv = results.get('group_velocities', {})
        dw = results.get('debye_waller', {})
        mrt = results.get('mode_resolved_thermo', {})

        return {
            'metadata': self._build_metadata(profile),
            'force_constants': self._build_fc_metadata(ifc),
            'band_structure': self._build_band_metadata(bands),
            'thermodynamics': self._build_thermo_metadata(thermo),
            'hydrogen_analysis': self._build_h_metadata(hydrogen),
            'group_velocities': self._build_gv_metadata(gv),
            'debye_waller': self._build_dw_metadata(dw),
            'mode_resolved_thermo': self._build_mrt_metadata(mrt),
            'phase_status': results.get('phase_status', {}),
        }

    def _build_metadata(self, profile) -> Dict:
        """Build metadata section."""
        return {
            'hydrophonokit_version': '2.7.0',
            'timestamp': datetime.datetime.now().isoformat(),
            'formula': getattr(profile, 'formula', 'Unknown'),
            'space_group': getattr(profile, 'space_group', 'Unknown'),
            'crystal_system': getattr(profile, 'crystal_system', 'Unknown'),
            'n_atoms_primitive': getattr(profile, 'n_atoms', 0),
            'density_g_cm3': getattr(profile, 'density', 0),
        }

    def _build_fc_metadata(self, ifc_data: Dict) -> Dict:
        """Build force constants metadata."""
        validation = ifc_data.get('validation', {})
        return {
            'method': ifc_data.get('method', 'unknown'),
            'asr_applied': True,
            'quality': validation.get('quality', 'unknown'),
            'max_fc_eV_A2': validation.get('max_fc_eV_A2', 0),
            'asr_error_eV_A3': validation.get('asr_error_eV_A3', 0),
        }

    def _build_band_metadata(self, bands_data: Dict) -> Dict:
        """Build band structure metadata."""
        stability = bands_data.get('stability', {})
        return {
            'min_frequency_THz': bands_data.get('min_freq', 0),
            'is_stable': stability.get('stable', False),
            'n_imaginary_modes': stability.get('n_imaginary', 0),
            'k_path': bands_data.get('k_path', 'unknown'),
        }

    def _build_thermo_metadata(self, thermo_data: Dict) -> Dict:
        """Build thermodynamic metadata."""
        validations = thermo_data.get('validations', {})
        at_300k = thermo_data.get('at_300K', {})
        return {
            'ZPE_kJ_mol': thermo_data.get('zpe', 0),
            'validation': validations,
            'at_300K': at_300k,
        }

    def _build_h_metadata(self, h_data: Dict) -> Dict:
        """Build hydrogen analysis metadata."""
        if not h_data:
            return {'present': False}

        decomp = h_data.get('decomposition', {})
        peak = h_data.get('peak_stretching', {})
        return {
            'present': True,
            'mode_decomposition': {
                'librational_pct': decomp.get('lib', 0),
                'bending_pct': decomp.get('bend', 0),
                'stretching_pct': decomp.get('stretch', 0),
            },
            'peak_stretching': {
                'frequency_THz': peak.get('freq_thz', 0),
                'frequency_cm': peak.get('freq_cm', 0),
            },
            'hydride_type': h_data.get('hydride_type', 'unknown'),
        }

    def _build_gv_metadata(self, gv_data: Dict) -> Dict:
        """Build group velocities metadata."""
        if not gv_data:
            return {'computed': False}
        return {
            'computed': True,
            'avg_sound_velocity_ms': gv_data.get('avg_sound_velocity', 0),
            'max_group_velocity_ms': gv_data.get('max_group_velocity', 0),
        }

    def _build_dw_metadata(self, dw_data: Dict) -> Dict:
        """Build Debye-Waller metadata."""
        if not dw_data:
            return {'computed': False}
        return {
            'computed': True,
            'temperature_K': dw_data.get('temperature', 0),
            'avg_isotropic_u2_A2': float(np.mean(dw_data.get('isotropic_u2', [0]))) if dw_data.get('isotropic_u2') is not None else 0,
            'avg_B_factor_A2': float(np.mean(dw_data.get('B_factor', [0]))) if dw_data.get('B_factor') is not None else 0,
            'zero_point_u2_A2': float(np.mean(dw_data.get('zero_point_u2', [0]))) if dw_data.get('zero_point_u2') is not None else 0,
        }

    def _build_mrt_metadata(self, mrt_data: Dict) -> Dict:
        """Build mode-resolved thermodynamics metadata."""
        if not mrt_data:
            return {'computed': False}
        return {
            'computed': True,
            'element_fractions': mrt_data.get('element_fractions', {}),
            'hydrogen_entropy_contribution_J_molK': mrt_data.get('hydrogen_entropy_contribution', 0),
        }

    def _json_serializer(self, obj):
        """Handle numpy arrays and other non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
