"""
=============================================================================
  HydroPhonoKit Postprocessor — Visualization Engine

  Generates publication-quality plots for phonon analysis results.
  Uses themes from advanced_plotting.py for consistent styling.

  Scientific Foundation:
    Phonon visualization follows standard conventions in condensed matter
    physics for band structure and DOS presentation:
      - Band structure: frequency vs q-path with high-symmetry points
      - DOS: total + element-projected with consistent color scheme
      - Thermodynamics: F(T), S(T), C_v(T) with Dulong-Petit limit
      - H-analysis: mode decomposition with labeled peaks

  References:
    [1] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy plotting
    [2] APS Style Guide -- Figure preparation standards
=============================================================================
"""
import os
import warnings
import numpy as np
from typing import Dict, Optional, Tuple

# Import themes and advanced plotting
from .advanced_plotting import THEMES, ELEMENT_COLORS, PhononPlotter as AdvancedPlotter
from .specialized_plots import SpecializedPlots

# Configure matplotlib backend
import os as _os
if not _os.environ.get('MPLBACKEND'):
    import matplotlib
    matplotlib.use('Agg')
del _os

import matplotlib.pyplot as plt


class PhononPlotter:
    """Generates publication-quality phonon plots.

    Wraps AdvancedPlotter with theme support for backward compatibility.
    """

    def __init__(self, theme: str = 'nature'):
        """
        Args:
            theme: Plot theme name ('nature', 'science', 'dark', 'minimal')
        """
        self.theme = theme
        self.advanced = AdvancedPlotter(theme=theme)

    def plot_band_structure(self, band_dict: Dict, formula: str,
                           output_dir: str) -> str:
        """Plot phonon band structure with stability badge."""
        fig, ax = self.advanced.plot_band_structure(band_dict, formula)
        path = os.path.join(output_dir, 'phonon_band_structure.png')
        self.advanced.save(path, fig=fig)
        plt.close(fig)
        print(f"  --> Saved: {path}")
        return path

    def plot_fat_bands(self, band_dict: Dict, pdos_data: Tuple,
                      formula: str, output_dir: str) -> str:
        """Plot element-projected ('fat') band structure."""
        fig, ax = self.advanced.plot_fat_bands(band_dict, pdos_data, formula)
        path = os.path.join(output_dir, 'phonon_fat_bands.png')
        self.advanced.save(path, fig=fig)
        plt.close(fig)
        print(f"  --> Saved: {path}")
        return path

    def plot_partial_dos(self, pdos_data, formula: str,
                        output_dir: str) -> str:
        """Plot element-projected partial DOS."""
        fig, axes = self.advanced.plot_partial_dos(pdos_data, formula)
        path = os.path.join(output_dir, 'phonon_dos_partial.png')
        self.advanced.save(path, fig=fig)
        plt.close(fig)
        print(f"  --> Saved: {path}")
        return path

    def plot_thermodynamics(self, thermo_data: Dict, formula: str,
                           n_atoms: int, output_dir: str) -> str:
        """Plot F(T), S(T), C_v(T) with Dulong-Petit limit."""
        fig, axes = self.advanced.plot_thermodynamics(thermo_data, formula, n_atoms)
        path = os.path.join(output_dir, 'phonon_thermodynamics.png')
        self.advanced.save(path, fig=fig)
        plt.close(fig)
        print(f"  --> Saved: {path}")
        return path

    def plot_h_modes(self, pdos_data, h_metrics: Dict, formula: str,
                    output_dir: str) -> str:
        """Plot hydrogen vibrational mode decomposition."""
        fig, ax = self.advanced.plot_h_modes(pdos_data, h_metrics, formula)
        path = os.path.join(output_dir, 'H_mode_analysis.png')
        self.advanced.save(path, fig=fig)
        plt.close(fig)
        print(f"  --> Saved: {path}")
        return path

    def plot_cumulative_kappa(self, mfp_data: Dict, formula: str,
                             output_dir: str) -> str:
        """Plot cumulative thermal conductivity vs mean free path."""
        fig, ax = SpecializedPlots.plot_cumulative_kappa(
            mfp_data['mean_free_paths'],
            mfp_data['kappa_contributions'],
            formula
        )
        path = os.path.join(output_dir, 'cumulative_kappa.png')
        self.advanced.save(path, fig=fig)
        plt.close(fig)
        print(f"  --> Saved: {path}")
        return path

