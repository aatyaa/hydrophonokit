"""
=============================================================================
  HydroPhonoKit Visualization — Comparison & Overlay Plots

  Implements comparison and overlay plot types:
    1. Multiple Materials Overlay - Compare band structures/DOS of different materials
    2. Theory vs Experiment - Overlay calculated with experimental data
    3. Convergence: Supercell Size - Compare different supercell sizes
    4. Convergence: q-Mesh Density - Compare different q-mesh densities

  Scientific Foundation:
    Comparison plots are essential for:
      - Validating calculations against experiment
      - Checking convergence with respect to computational parameters
      - Understanding trends across material families
      - Identifying systematic errors

  References:
    [1] Setyawan & Curtarolo, Comput. Mater. Sci. 49, 2010 -- High-symmetry paths
    [2] Hinuma et al., Comput. Mater. Sci. 140, 2017 -- seekpath paths
=============================================================================
"""
import os
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

# Configure matplotlib backend
import os as _os
if not _os.environ.get('MPLBACKEND'):
    import matplotlib
    matplotlib.use('Agg')
del _os

import matplotlib.pyplot as plt
import matplotlib.axes as mpl_axes
from matplotlib.figure import Figure

from .base_plotter import BasePlotter


# ============================================================================
# COMPARISON PLOTTER CLASS
# ============================================================================

class ComparisonPlotter(BasePlotter):
    """Specialized plotter for comparison and overlay plots.
    
    Provides methods for multi-material and convergence visualization.
    """
    
    def plot_multi_material_bands(self, band_data: Dict[str, Dict],
                                 formula: str = 'Materials',
                                 ax: mpl_axes.Axes = None,
                                 figsize: Tuple[float, float] = (8, 6),
                                 title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot multiple materials' band structures overlaid.
        
        Args:
            band_data: Dict mapping formula -> band_dict
            formula: Overall title
            ax: Existing axis
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        colors = ['#1E3A8A', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
        
        for i, (mat_formula, band_dict) in enumerate(band_data.items()):
            color = colors[i % len(colors)]
            
            distances = band_dict.get('distances', [])
            frequencies = band_dict.get('frequencies', [])
            
            for dist, freq in zip(distances, frequencies):
                for b in range(freq.shape[1]):
                    ax.plot(dist, freq[:, b], color=color, lw=0.8, alpha=0.85,
                           label=mat_formula if b == 0 else "")
        
        ax.set_ylabel('Frequency (THz)')
        ax.set_xlabel('')
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        ax.legend(loc='upper right', fontsize=9)
        
        if title is None:
            title = f'{formula} - Multi-Material Band Comparison'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_multi_material_dos(self, dos_data: Dict[str, Tuple],
                               formula: str = 'Materials',
                               normalize: bool = True,
                               ax: mpl_axes.Axes = None,
                               figsize: Tuple[float, float] = (8, 6),
                               title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot multiple materials' DOS overlaid.
        
        Args:
            dos_data: Dict mapping formula -> (freq, total_dos)
            formula: Overall title
            normalize: Normalize each DOS to unit area
            ax: Existing axis
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        colors = ['#1E3A8A', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
        
        for i, (mat_formula, (freq, dos)) in enumerate(dos_data.items()):
            color = colors[i % len(colors)]
            
            dos_plot = dos.copy()
            if normalize:
                integral = np.trapz(dos_plot, freq)
                if integral > 0:
                    dos_plot = dos_plot / integral
            
            ax.plot(freq, dos_plot, color=color, lw=1.5, label=mat_formula)
        
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('DOS (normalized)' if normalize else 'DOS (states/THz)')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if title is None:
            title = f'{formula} - Multi-Material DOS Comparison'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_theory_vs_experiment(self, calc_freq: np.ndarray,
                                 calc_intensity: np.ndarray,
                                 exp_freq: np.ndarray,
                                 exp_intensity: np.ndarray,
                                 exp_errors: np.ndarray = None,
                                 formula: str = 'Material',
                                 ax: mpl_axes.Axes = None,
                                 figsize: Tuple[float, float] = (8, 6),
                                 title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot calculated vs experimental spectra.
        
        Args:
            calc_freq: Calculated frequencies (THz)
            calc_intensity: Calculated intensities
            exp_freq: Experimental frequencies (THz)
            exp_intensity: Experimental intensities
            exp_errors: Experimental error bars (optional)
            formula: Material formula
            ax: Existing axis
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        # Experimental data with error bars
        if exp_errors is not None:
            ax.errorbar(exp_freq, exp_intensity, yerr=exp_errors,
                       fmt='o', color='#E74C3C', markersize=5, capsize=3,
                       label='Experiment', zorder=5)
        else:
            ax.plot(exp_freq, exp_intensity, 'o', color='#E74C3C',
                   markersize=5, label='Experiment', zorder=5)
        
        # Calculated spectrum
        ax.plot(calc_freq, calc_intensity, '-', color='#1E3A8A', lw=1.5,
               label='Calculated', zorder=3)
        
        # Correlation annotation
        if len(calc_freq) > 1 and len(exp_freq) > 1:
            from scipy import stats
            # Interpolate calculated to experimental frequencies
            calc_interp = np.interp(exp_freq, calc_freq, calc_intensity)
            r_value, p_value = stats.pearsonr(exp_intensity, calc_interp)
            
            ax.text(0.02, 0.95, f'R = {r_value:.3f}\np = {p_value:.2e}',
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('Intensity (arb. units)')
        ax.legend()
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if title is None:
            title = f'{formula} - Theory vs Experiment'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_convergence_supercell(self, supercell_sizes: List[str],
                                  frequencies: List[np.ndarray],
                                  property_name: str = 'Frequency at Γ',
                                  ax: mpl_axes.Axes = None,
                                  figsize: Tuple[float, float] = (8, 6),
                                  title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot convergence vs supercell size.
        
        Args:
            supercell_sizes: List of supercell size strings (e.g., ['2x2x2', '3x3x3'])
            frequencies: List of frequency arrays for each size
            property_name: Y-axis label
            ax: Existing axis
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        n_sizes = len(supercell_sizes)
        x = np.arange(n_sizes)
        
        # Plot each band
        if len(frequencies) > 0:
            n_bands = len(frequencies[0])
            for b in range(min(n_bands, 10)):  # Plot first 10 bands
                freq_vals = [freqs[b] for freqs in frequencies if len(freqs) > b]
                ax.plot(x[:len(freq_vals)], freq_vals, 'o-', markersize=4,
                       label=f'Band {b+1}')
        
        ax.set_xticks(x)
        ax.set_xticklabels(supercell_sizes)
        ax.set_ylabel(property_name + ' (THz)')
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if n_sizes > 1:
            ax.legend(fontsize=8, loc='upper right')
        
        if title is None:
            title = 'Convergence vs Supercell Size'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_convergence_qmesh(self, mesh_sizes: List[str],
                              property_values: List[float],
                              property_name: str = 'ZPE (kJ/mol)',
                              ax: mpl_axes.Axes = None,
                              figsize: Tuple[float, float] = (8, 6),
                              title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot convergence vs q-mesh density.
        
        Args:
            mesh_sizes: List of mesh size strings (e.g., ['10x10x10', '15x15x15'])
            property_values: Property value for each mesh
            property_name: Y-axis label
            ax: Existing axis
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        x = np.arange(len(mesh_sizes))
        
        ax.plot(x, property_values, 'o-', color='#1E3A8A', markersize=6, lw=1.5)
        ax.fill_between(x, property_values, alpha=0.2, color='#1E3A8A')
        
        ax.set_xticks(x)
        ax.set_xticklabels(mesh_sizes, rotation=45)
        ax.set_ylabel(property_name)
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        # Convergence annotation
        if len(property_values) > 1:
            final_val = property_values[-1]
            prev_val = property_values[-2]
            deviation = abs(final_val - prev_val) / abs(final_val) * 100
            
            ax.text(0.02, 0.95,
                   f'Final: {final_val:.3f}\n'
                   f'Last change: {deviation:.2f}%',
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if title is None:
            title = 'Convergence vs q-Mesh Density'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_multi_property_comparison(self, materials: List[str],
                                      properties: Dict[str, List[float]],
                                      ax: mpl_axes.Axes = None,
                                      figsize: Tuple[float, float] = (10, 6),
                                      title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot multiple properties for multiple materials as grouped bar chart.
        
        Args:
            materials: List of material formulas
            properties: Dict mapping property name -> list of values
            ax: Existing axis
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        n_materials = len(materials)
        n_properties = len(properties)
        x = np.arange(n_materials)
        width = 0.8 / n_properties
        
        colors = ['#1E3A8A', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
        
        for i, (prop_name, values) in enumerate(properties.items()):
            ax.bar(x + i * width, values, width, label=prop_name,
                  color=colors[i % len(colors)])
        
        ax.set_xticks(x + width * (n_properties - 1) / 2)
        ax.set_xticklabels(materials, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=self.theme['axes']['grid_alpha'], axis='y')
        
        if title is None:
            title = 'Multi-Material Property Comparison'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
