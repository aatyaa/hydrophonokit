"""
=============================================================================
  HydroPhonoKit Postprocessor — Report Generator

  Generates comprehensive HTML summary reports with all available results.

  Scientific Foundation:
    Reports provide a human-readable summary of phonon analysis results
    including stability assessment, thermodynamic properties, and
    hydrogen storage analytics when applicable.
=============================================================================
"""
import os
import numpy as np
from typing import Dict, Optional


class ReportGenerator:
    """Generates comprehensive HTML summary reports.

    This class handles:
      1. HTML report generation with all available data
      2. Stability assessment display
      3. Thermodynamic property summary
      4. Scientific enhancements summary
      5. H-storage analytics (if available)
    """

    def generate(self, results: Dict, output_dir: str) -> str:
        """Generate HTML report with all available results.

        Args:
            results: Dict with all phase results (some may be None)
            output_dir: Output directory

        Returns:
            Path to saved HTML report
        """
        print("\n[Phase 6] Report Generation")

        html_path = os.path.join(output_dir, 'Phonon_Analysis_Report.html')

        # Safely extract all data
        dl = results.get('data_loader') or {}
        phonon = dl.get('phonon') if dl else None
        profile = results.get('profile')
        bands = results.get('bands_dos') or {}
        thermo = results.get('thermodynamics') or {}
        h_data = results.get('hydrogen') or {}
        gv = results.get('group_velocities') or {}
        dw = results.get('debye_waller') or {}
        mrt = results.get('mode_resolved_thermo') or {}
        fc = results.get('ifc') or {}

        # Stability
        stability_text = "Dynamically Stable at 0K"
        stability_color = "#059669"
        min_freq = bands.get('min_freq', 0) if bands else 0
        if min_freq < -0.2:
            stability_text = f"Dynamically Unstable (min: {min_freq:.2f} THz)"
            stability_color = "#DC2626"

        # Basic info
        n_prim = len(phonon.unitcell.numbers) if phonon else 0
        n_sc = len(phonon.supercell.numbers) if phonon else 0
        formula = getattr(profile, 'formula', 'Unknown')
        space_group = getattr(profile, 'space_group', 'Unknown')
        crystal_sys = getattr(profile, 'crystal_system', 'Unknown')

        # Thermodynamics
        thermo_data = thermo.get('thermo_data') or {}
        at_300k = thermo.get('at_300K') or {}
        validations = thermo.get('validations') or {}
        temps = thermo_data.get('temperatures', [])
        idx_300 = int(np.argmin(np.abs(np.array(temps) - 300))) if temps else 30
        fe_list = thermo_data.get('free_energy', [])
        s_list = thermo_data.get('entropy', [])
        cv_list = thermo_data.get('heat_capacity', [])

        zpe = thermo.get('zpe', 0)
        fe_300 = fe_list[idx_300] if fe_list else 0
        s_300 = s_list[idx_300] if s_list else 0
        cv_300 = cv_list[idx_300] if cv_list else 0

        # Build HTML
        html = self._build_html(
            formula=formula, space_group=space_group, crystal_sys=crystal_sys,
            n_prim=n_prim, n_sc=n_sc, min_freq=min_freq,
            stability_text=stability_text, stability_color=stability_color,
            zpe=zpe, fe_300=fe_300, s_300=s_300, cv_300=cv_300,
            validations=validations, bands=bands, fc=fc,
            gv=gv, dw=dw, mrt=mrt, h_data=h_data
        )

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"  --> HTML report completed: {html_path}")
        return html_path

    def _build_html(self, **kwargs) -> str:
        """Build complete HTML document."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Phonon Analysis: {kwargs['formula']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; }}
        .header {{ background: #1e3a8a; color: white; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; }}
        .card {{ background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); margin-bottom: 1.5rem; }}
        .card h2 {{ margin-top: 0; color: #1e3a8a; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 8px 12px; border: 1px solid #e2e8f0; text-align: left; }}
        th {{ background: #f1f5f9; font-weight: bold; }}
        .stability {{ font-size: 1.25rem; font-weight: bold; color: {kwargs['stability_color']}; padding: 1rem; border-left: 4px solid {kwargs['stability_color']}; background: #f1f5f9; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; }}
        .pass {{ background: #d1fae5; color: #059669; }}
        .warn {{ background: #fef3c7; color: #d97706; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Phonon Post-Processing Report: {kwargs['formula']}</h1>
        <p>Space Group: {kwargs['space_group']} | Crystal System: {kwargs['crystal_sys']}</p>
        <p>Atoms: {kwargs['n_prim']} primitive / {kwargs['n_sc']} supercell</p>
    </div>

    <div class="card stability">
        {kwargs['stability_text']}
    </div>

    <div class="card">
        <h2>Thermodynamics @ 300K</h2>
        <table>
            <tr><th>Property</th><th>Value</th><th>Unit</th></tr>
            <tr><td>Zero Point Energy</td><td>{kwargs['zpe']:.2f}</td><td>kJ/mol</td></tr>
            <tr><td>Helmholtz Free Energy F</td><td>{kwargs['fe_300']:.2f}</td><td>kJ/mol</td></tr>
            <tr><td>Entropy S</td><td>{kwargs['s_300']:.2f}</td><td>J/(mol·K)</td></tr>
            <tr><td>Heat Capacity Cv</td><td>{kwargs['cv_300']:.2f}</td><td>J/(mol·K)</td></tr>
        </table>
    </div>
"""
        # Force Constants
        fc = kwargs.get('fc') or {}
        if fc:
            html += f"""
    <div class="card">
        <h2>Force Constants</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Method</td><td>{fc.get('method', 'unknown')}</td></tr>
            <tr><td>Quality</td><td>{fc.get('validation', {}).get('quality', 'unknown')}</td></tr>
            <tr><td>Max FC Magnitude</td><td>{fc.get('validation', {}).get('max_fc_eV_A2', 0):.2f} eV/A^2</td></tr>
        </table>
    </div>
"""
        # Group Velocities
        gv = kwargs.get('gv') or {}
        if gv.get('avg_sound_velocity', 0) > 0:
            html += f"""
    <div class="card">
        <h2>Group Velocities</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Avg Sound Velocity</td><td>{gv['avg_sound_velocity']:.1f} m/s</td></tr>
            <tr><td>Max Group Velocity</td><td>{gv['max_group_velocity']:.1f} m/s</td></tr>
        </table>
    </div>
"""
        # Debye-Waller
        dw = kwargs.get('dw') or {}
        if dw.get('computed'):
            html += f"""
    <div class="card">
        <h2>Debye-Waller Factors (@ {dw.get('temperature', 300)}K)</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Avg Mean Square Displacement</td><td>{dw.get('avg_isotropic_u2_A2', 0):.4f} A^2</td></tr>
            <tr><td>Avg B-Factor</td><td>{dw.get('avg_B_factor_A2', 0):.2f} A^2</td></tr>
            <tr><td>Zero-Point <u^2></td><td>{dw.get('zero_point_u2_A2', 0):.4f} A^2</td></tr>
        </table>
    </div>
"""
        # Mode-Resolved Thermo
        mrt = kwargs.get('mrt') or {}
        if mrt.get('computed'):
            fracs = mrt.get('element_fractions', {})
            rows = ''.join(f"<tr><td>{e}</td><td>{f:.1f}%</td></tr>" for e, f in fracs.items())
            html += f"""
    <div class="card">
        <h2>Mode-Resolved Thermodynamics</h2>
        <table>
            <tr><th>Element</th><th>Entropy Contribution @ 300K</th></tr>
            {rows}
            <tr><td>H Entropy Contribution</td><td>{mrt.get('hydrogen_entropy_contribution_J_molK', 0):.2f} J/(mol·K)</td></tr>
        </table>
    </div>
"""
        # Hydrogen
        h_data = kwargs.get('h_data') or {}
        if h_data.get('peak_stretching'):
            ps = h_data['peak_stretching']
            decomp = h_data.get('decomposition', {})
            html += f"""
    <div class="card">
        <h2>Hydrogen Storage Analytics</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Principal Stretch Frequency</td><td>{ps['freq_thz']:.2f} THz ({ps['freq_cm']:.0f} cm^-1)</td></tr>
            <tr><td>Hydride Type</td><td>{h_data.get('hydride_type', 'unknown')}</td></tr>
            <tr><td>Librational Fraction</td><td>{decomp.get('lib', 0):.1f}%</td></tr>
            <tr><td>Bending Fraction</td><td>{decomp.get('bend', 0):.1f}%</td></tr>
            <tr><td>Stretching Fraction</td><td>{decomp.get('stretch', 0):.1f}%</td></tr>
        </table>
    </div>
"""
        html += "</body></html>"
        return html
