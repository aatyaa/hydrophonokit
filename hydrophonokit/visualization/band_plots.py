"""
=============================================================================
  HydroPhonoKit Visualization — Band Structure Plots

  Implements all band structure plot types for phonon analysis:
    1. Standard Band Structure - High-symmetry path with stability badge
    2. Fat Band Structure - Element-projected band widths
    3. Orbital-Projected Bands - Mode character analysis
    4. Band Unfolding - Supercell to primitive cell mapping
    5. Overlay with Experiment - Theory vs IR/Raman data
    6. Imaginary Mode Highlight - Detailed unstable mode analysis
    7. Zoomed Regions - Focused views of specific frequency ranges

  Scientific Foundation:
    Phonon band structures reveal dynamical stability, mode characters,
    and vibrational properties along high-symmetry paths in the Brillouin zone.

    Key features:
      - Acoustic modes: 3 branches approaching zero at Gamma
      - Optical modes: Higher frequency branches
      - Imaginary modes: Indicate dynamical instability (negative freq^2)
      - Mode gaps: Frequency ranges with no allowed modes
      - Van Hove singularities: Flat band regions (high DOS)

  References:
    [1] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy
    [2] Hinuma et al., Comput. Mater. Sci. 140, 2017 -- seekpath paths
    [3] Setyawan & Curtarolo, Comput. Mater. Sci. 49, 2010 -- High-symmetry paths
=============================================================================
"""
import os
import warnings
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
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from .base_plotter import (
    BasePlotter,
    freq_to_cm,
    freq_to_meV,
    THZ_TO_CM,
    HIGH_SYMMETRY_LABELS,
)
from .themes import get_element_color, ELEMENT_COLORS


# ============================================================================
# HIGH-SYMMETRY PATH UTILITIES
# ============================================================================

def parse_band_labels(labels: List[str]) -> Tuple[List[str], List[float]]:
    """Parse band structure labels and positions.
    
    Args:
        labels: List of high-symmetry point labels
    
    Returns:
        (unique_labels, positions) tuple
    """
    unique_labels = []
    positions = []
    seen = set()
    
    for i, label in enumerate(labels):
        if label not in seen:
            # Convert GAMMA to Greek Gamma symbol
            display = HIGH_SYMMETRY_LABELS.get(label, label)
            unique_labels.append(display)
            positions.append(i)
            seen.add(label)
    
    return unique_labels, positions


# ============================================================================
# BAND STRUCTURE PLOTTER CLASS
# ============================================================================

class BandStructurePlotter(BasePlotter):
    """Specialized plotter for phonon band structures.
    
    Provides methods for all band structure plot types.
    """
    
    def plot_standard(self, band_dict: Dict, formula: str = 'Material',
                     unit: str = 'THz', highlight_imaginary: bool = True,
                     show_acoustic_gap: bool = False, ax: mpl_axes.Axes = None,
                     figsize: Tuple[float, float] = (8, 6),
                     title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot standard phonon band structure.
        
        Features:
          - High-symmetry path with labeled points
          - Zero frequency reference line
          - Stability badge (stable/imaginary)
          - Optional acoustic mode gap highlighting
          - Frequency unit conversion (THz, cm^-1, meV)
        
        Args:
            band_dict: Phonopy band structure dictionary
            formula: Material formula for title
            unit: Frequency unit ('THz', 'cm^-1', 'meV')
            highlight_imaginary: Color imaginary modes red
            show_acoustic_gap: Highlight gap between acoustic and optical modes
            ax: Existing axis (creates new if None)
            figsize: Figure size if creating new axis
            title: Custom title (defaults to '{formula} - Phonon Band Structure')
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        distances = band_dict['distances']
        frequencies = band_dict['frequencies']
        
        # Extract all frequencies for stability check
        all_freqs = np.concatenate([f.flatten() for f in frequencies])
        min_freq = np.min(all_freqs)
        max_freq = np.max(all_freqs)
        has_imaginary = min_freq < -0.1
        
        # Collect high-symmetry positions
        all_positions = []
        for dist_arr in distances:
            all_positions.extend([dist_arr[0], dist_arr[-1]])
        unique_positions = sorted(set(all_positions))
        
        # Plot bands
        for dist, freq in zip(distances, frequencies):
            for b in range(freq.shape[1]):
                if highlight_imaginary and has_imaginary:
                    # Color segments based on frequency sign
                    for i in range(len(dist) - 1):
                        f_avg = (freq[i, b] + freq[i+1, b]) / 2
                        if f_avg < -0.1:
                            color = self.get_color('band_imaginary')
                            alpha = 0.9
                        else:
                            color = self.get_color('band')
                            alpha = 0.85
                        ax.plot(dist[i:i+2], freq[i:i+2, b],
                               color=color, lw=0.8, alpha=alpha)
                else:
                    ax.plot(dist, freq[:, b],
                           color=self.get_color('band'), lw=0.8, alpha=0.85)
        
        # Add high-symmetry markers
        self.add_high_symmetry_markers(ax, unique_positions)
        
        # Add zero line
        self.add_zero_line(ax)
        
        # Highlight acoustic-optical gap if requested
        if show_acoustic_gap and not has_imaginary:
            # Find max acoustic frequency at Gamma
            acoustic_max = 0
            for freq in frequencies:
                for b in range(min(3, freq.shape[1])):
                    acoustic_max = max(acoustic_max, freq[0, b])
            
            if acoustic_max > 0:
                ax.axhspan(0, acoustic_max, alpha=0.1, color='green',
                          label='Acoustic modes')
        
        # Add stability badge
        self.add_stability_badge(ax, min_freq)
        
        # Format axes
        self._format_band_axes(ax, unit=unit, y_limits=(-0.5, max_freq * 1.05))
        
        # Title
        if title is None:
            title = f'{formula} - Phonon Band Structure'
        ax.set_title(title, fontweight='bold')
        
        # Set x-ticks at high-symmetry points
        self._set_high_symmetry_ticks(ax, distances)
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_fat_bands(self, band_dict: Dict, pdos_data: Tuple,
                      formula: str = 'Material',
                      elements: List[str] = None,
                      max_width: float = 4.0,
                      unit: str = 'THz',
                      background_bands: bool = True,
                      ax: mpl_axes.Axes = None,
                      figsize: Tuple[float, float] = (10, 7),
                      title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot element-projected ('fat') band structure.
        
        Band width is proportional to element's contribution to each mode,
        calculated from partial DOS at that frequency.
        
        Args:
            band_dict: Phonopy band structure dictionary
            pdos_data: (freq, pdos, sym_to_idx) tuple from DOS calculation
            formula: Material formula for title
            elements: List of elements to show (None = all)
            max_width: Maximum band width in points
            unit: Frequency unit ('THz', 'cm^-1', 'meV')
            background_bands: Show thin background bands for reference
            ax: Existing axis (creates new if None)
            figsize: Figure size if creating new axis
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        distances = band_dict['distances']
        frequencies = band_dict['frequencies']
        pdos_freq, pdos_arr, sym_to_idx = pdos_data
        
        elements_to_show = elements or list(sym_to_idx.keys())
        
        # Plot fat bands
        for dist, freq_arr in zip(distances, frequencies):
            for b in range(freq_arr.shape[1]):
                f_val = freq_arr[0, b]
                if f_val < 0:
                    continue
                
                # Find DOS contribution at this frequency
                freq_idx = np.argmin(np.abs(pdos_freq - abs(f_val)))
                total_dos = pdos_arr[:, freq_idx].sum()
                
                if total_dos < 1e-10:
                    continue
                
                for elem in elements_to_show:
                    if elem not in sym_to_idx:
                        continue
                    
                    indices = sym_to_idx[elem]
                    elem_dos = sum(
                        pdos_arr[i, freq_idx]
                        for i in indices
                        if i < pdos_arr.shape[0]
                    )
                    fraction = elem_dos / total_dos
                    width = max_width * fraction
                    color = self.get_element_color(elem)
                    
                    for i in range(len(dist) - 1):
                        ax.plot(dist[i:i+2], freq_arr[i:i+2, b],
                               color=color, lw=width, alpha=0.7)
        
        # Background bands
        if background_bands:
            for dist, freq_arr in zip(distances, frequencies):
                for b in range(freq_arr.shape[1]):
                    ax.plot(dist, freq_arr[:, b],
                           color='black', lw=0.5, alpha=0.3, zorder=0)
        
        # High-symmetry markers and zero line
        all_positions = []
        for dist_arr in distances:
            all_positions.extend([dist_arr[0], dist_arr[-1]])
        unique_positions = sorted(set(all_positions))
        self.add_high_symmetry_markers(ax, unique_positions)
        self.add_zero_line(ax)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=self.get_element_color(e), lw=3, label=e)
            for e in elements_to_show
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Format axes
        max_freq = max(np.max(f) for f in frequencies)
        self._format_band_axes(ax, unit=unit, y_limits=(-0.5, max_freq * 1.05))
        
        if title is None:
            title = f'{formula} - Fat Band Structure'
        ax.set_title(title, fontweight='bold')
        
        self._set_high_symmetry_ticks(ax, distances)
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_orbital_projected(self, band_dict: Dict, mode_characters: Dict,
                              formula: str = 'Material',
                              mode_types: List[str] = None,
                              unit: str = 'THz',
                              ax: mpl_axes.Axes = None,
                              figsize: Tuple[float, float] = (10, 7),
                              title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot orbital/mode-character projected bands.
        
        Shows the character of each phonon mode (e.g., stretching, bending,
        librational, acoustic, optical) through color coding.
        
        Args:
            band_dict: Phonopy band structure dictionary
            mode_characters: Dict mapping mode index to character string
            formula: Material formula
            mode_types: List of mode types to show (None = all)
            unit: Frequency unit
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
        
        distances = band_dict['distances']
        frequencies = band_dict['frequencies']
        
        # Color map for mode types
        mode_colors = {
            'acoustic': '#2ECC71',      # Green
            'optical': '#3498DB',       # Blue
            'stretching': '#E74C3C',    # Red
            'bending': '#F39C12',       # Orange
            'librational': '#9B59B6',   # Purple
            'mixed': '#95A5A6',         # Gray
        }
        
        mode_types_to_show = mode_types or list(mode_colors.keys())
        
        # Plot bands with mode character colors
        for dist, freq in zip(distances, frequencies):
            for b in range(freq.shape[1]):
                # Determine mode character
                mode_char = mode_characters.get(b, 'mixed')
                color = mode_colors.get(mode_char, mode_colors['mixed'])
                
                if mode_char in mode_types_to_show:
                    ax.plot(dist, freq[:, b], color=color, lw=0.8, alpha=0.85,
                           label=mode_char if b == 0 else "")
        
        # High-symmetry markers and zero line
        all_positions = []
        for dist_arr in distances:
            all_positions.extend([dist_arr[0], dist_arr[-1]])
        unique_positions = sorted(set(all_positions))
        self.add_high_symmetry_markers(ax, unique_positions)
        self.add_zero_line(ax)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color=c, lw=2, label=t)
            for t, c in mode_colors.items()
            if t in mode_types_to_show
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Format axes
        all_freqs = np.concatenate([f.flatten() for f in frequencies])
        max_freq = np.max(all_freqs)
        self._format_band_axes(ax, unit=unit, y_limits=(-0.5, max_freq * 1.05))
        
        if title is None:
            title = f'{formula} - Mode-Projected Bands'
        ax.set_title(title, fontweight='bold')
        
        self._set_high_symmetry_ticks(ax, distances)
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_with_experiment(self, band_dict: Dict, exp_data: Dict,
                            formula: str = 'Material',
                            unit: str = 'THz',
                            show_error_bars: bool = True,
                            ax: mpl_axes.Axes = None,
                            figsize: Tuple[float, float] = (8, 6),
                            title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot band structure overlaid with experimental data.
        
        Args:
            band_dict: Phonopy band structure dictionary
            exp_data: Dict with experimental data:
                - 'q_points': Array of q-point positions (fractional)
                - 'frequencies': Array of experimental frequencies (THz)
                - 'errors': Array of error bars (optional, THz)
                - 'method': String ('INS', 'IXS', 'Raman', 'IR')
            formula: Material formula
            unit: Frequency unit
            show_error_bars: Show experimental error bars
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
        
        # Plot calculated bands
        distances = band_dict['distances']
        frequencies = band_dict['frequencies']
        
        for dist, freq in zip(distances, frequencies):
            for b in range(freq.shape[1]):
                ax.plot(dist, freq[:, b], color=self.get_color('band'),
                       lw=0.8, alpha=0.85)
        
        # Plot experimental data
        exp_q = exp_data.get('q_points', [])
        exp_freq = exp_data.get('frequencies', [])
        exp_errors = exp_data.get('errors', None)
        method = exp_data.get('method', 'Experiment')
        
        if len(exp_q) > 0 and len(exp_freq) > 0:
            if show_error_bars and exp_errors is not None:
                ax.errorbar(exp_q, exp_freq, yerr=exp_errors,
                           fmt='ro', markersize=6, capsize=3,
                           label=f'{method}', zorder=5)
            else:
                ax.plot(exp_q, exp_freq, 'ro', markersize=6,
                       label=f'{method}', zorder=5)
        
        # High-symmetry markers and zero line
        all_positions = []
        for dist_arr in distances:
            all_positions.extend([dist_arr[0], dist_arr[-1]])
        unique_positions = sorted(set(all_positions))
        self.add_high_symmetry_markers(ax, unique_positions)
        self.add_zero_line(ax)
        
        ax.legend()
        
        # Format axes
        all_freqs = np.concatenate([f.flatten() for f in frequencies])
        max_freq = max(np.max(all_freqs), max(exp_freq) if exp_freq else 0)
        self._format_band_axes(ax, unit=unit, y_limits=(-0.5, max_freq * 1.1))
        
        if title is None:
            title = f'{formula} - Theory vs Experiment'
        ax.set_title(title, fontweight='bold')
        
        self._set_high_symmetry_ticks(ax, distances)
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_imaginary_modes(self, band_dict: Dict, formula: str = 'Material',
                            threshold: float = -0.1,
                            highlight_regions: bool = True,
                            unit: str = 'THz',
                            ax: mpl_axes.Axes = None,
                            figsize: Tuple[float, float] = (8, 6),
                            title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot band structure with detailed imaginary mode analysis.
        
        Highlights:
          - All imaginary frequency modes in red
          - q-point ranges where imaginary modes occur
          - Most unstable mode (minimum frequency)
          - Acoustic mode region near Gamma
        
        Args:
            band_dict: Phonopy band structure dictionary
            formula: Material formula
            threshold: Frequency below which is considered imaginary (THz)
            highlight_regions: Highlight q-ranges with imaginary modes
            unit: Frequency unit
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
        
        distances = band_dict['distances']
        frequencies = band_dict['frequencies']
        
        # Find all imaginary modes
        all_freqs = np.concatenate([f.flatten() for f in frequencies])
        min_freq = np.min(all_freqs)
        
        # Plot bands
        for dist, freq in zip(distances, frequencies):
            for b in range(freq.shape[1]):
                # Check if this band has imaginary modes
                has_imag = np.any(freq < threshold)
                
                if has_imag:
                    # Color by frequency: more negative = more red
                    for i in range(len(dist) - 1):
                        f_avg = (freq[i, b] + freq[i+1, b]) / 2
                        if f_avg < threshold:
                            # Gradient from orange to red based on instability
                            intensity = min(1.0, abs(f_avg) / (abs(min_freq) + 0.1))
                            color = (1.0, 1.0 - intensity, 0)  # Yellow to Red
                            alpha = 0.9
                        else:
                            color = self.get_color('band')
                            alpha = 0.85
                        ax.plot(dist[i:i+2], freq[i:i+2, b],
                               color=color, lw=0.8, alpha=alpha)
                else:
                    ax.plot(dist, freq[:, b],
                           color=self.get_color('band'), lw=0.8, alpha=0.85)
        
        # Highlight imaginary regions
        if highlight_regions:
            # Find q-points with imaginary modes
            imag_q_ranges = []
            current_start = None
            
            for dist_arr, freq_arr in zip(distances, frequencies):
                for i in range(len(dist_arr)):
                    if np.any(freq_arr[i] < threshold):
                        if current_start is None:
                            current_start = dist_arr[i]
                    else:
                        if current_start is not None:
                            imag_q_ranges.append((current_start, dist_arr[i]))
                            current_start = None
            
            if current_start is not None:
                imag_q_ranges.append((current_start, dist_arr[-1]))
            
            # Add shaded regions
            for start, end in imag_q_ranges:
                ax.axvspan(start, end, alpha=0.1, color='red')
        
        # High-symmetry markers and zero line
        all_positions = []
        for dist_arr in distances:
            all_positions.extend([dist_arr[0], dist_arr[-1]])
        unique_positions = sorted(set(all_positions))
        self.add_high_symmetry_markers(ax, unique_positions)
        self.add_zero_line(ax, color='#DC2626', lw=1.5)
        
        # Stability badge
        n_imaginary = np.sum(all_freqs < threshold)
        badge_text = f'{n_imaginary} Imaginary Modes (min: {min_freq:.2f} THz)'
        ax.text(0.02, 0.02, badge_text,
                transform=ax.transAxes, color='#DC2626', fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#FEE2E2', alpha=0.9))
        
        # Format axes
        max_freq = np.max(all_freqs[all_freqs > 0]) if np.any(all_freqs > 0) else 5
        self._format_band_axes(ax, unit=unit, y_limits=(min_freq * 1.2, max_freq * 1.1))
        
        if title is None:
            title = f'{formula} - Imaginary Mode Analysis'
        ax.set_title(title, fontweight='bold')
        
        self._set_high_symmetry_ticks(ax, distances)
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_zoomed(self, band_dict: Dict, formula: str = 'Material',
                   freq_range: Tuple[float, float] = (0, 5),
                   unit: str = 'THz',
                   highlight_region: bool = True,
                   ax: mpl_axes.Axes = None,
                   figsize: Tuple[float, float] = (8, 6),
                   title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot zoomed-in view of specific frequency range.
        
        Useful for:
          - Acoustic modes near Gamma (0-2 THz)
          - Low-energy optical modes
          - Specific mode groups (e.g., H-stretching)
        
        Args:
            band_dict: Phonopy band structure dictionary
            formula: Material formula
            freq_range: (min_freq, max_freq) in THz
            unit: Frequency unit
            highlight_region: Shade the zoomed region
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
        
        distances = band_dict['distances']
        frequencies = band_dict['frequencies']
        
        f_min, f_max = freq_range
        
        # Plot bands (only within range)
        for dist, freq in zip(distances, frequencies):
            for b in range(freq.shape[1]):
                # Clip to range for display
                freq_clipped = np.clip(freq[:, b], f_min, f_max)
                ax.plot(dist, freq_clipped,
                       color=self.get_color('band'), lw=0.8, alpha=0.85)
        
        # Highlight region
        if highlight_region:
            ax.axhspan(f_min, f_max, alpha=0.05, color='blue')
        
        # High-symmetry markers
        all_positions = []
        for dist_arr in distances:
            all_positions.extend([dist_arr[0], dist_arr[-1]])
        unique_positions = sorted(set(all_positions))
        self.add_high_symmetry_markers(ax, unique_positions)
        
        # Add range indicators
        ax.axhline(y=f_min, color='gray', ls='--', alpha=0.5)
        ax.axhline(y=f_max, color='gray', ls='--', alpha=0.5)
        
        # Format axes
        self._format_band_axes(ax, unit=unit, y_limits=(f_min, f_max))
        
        if title is None:
            title = f'{formula} - Zoomed ({f_min}-{f_max} THz)'
        ax.set_title(title, fontweight='bold')
        
        self._set_high_symmetry_ticks(ax, distances)
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _format_band_axes(self, ax: mpl_axes.Axes, unit: str = 'THz',
                         y_limits: Tuple[float, float] = None):
        """Format band structure axes with proper labels and limits.
        
        Args:
            ax: Matplotlib axis
            unit: Frequency unit
            y_limits: (min, max) frequency limits
        """
        if unit == 'THz':
            ax.set_ylabel('Frequency (THz)', fontsize=12)
        elif unit == 'cm^-1':
            ax.set_ylabel('Frequency (cm$^{-1}$)', fontsize=12)
            # Convert y-ticks
            ticks = ax.get_yticks()
            ax.set_yticklabels([f'{t * THZ_TO_CM:.0f}' for t in ticks])
        elif unit == 'meV':
            from .base_plotter import THZ_TO_MEV
            ax.set_ylabel('Frequency (meV)', fontsize=12)
            ticks = ax.get_yticks()
            ax.set_yticklabels([f'{t * THZ_TO_MEV:.1f}' for t in ticks])
        
        ax.set_xlabel('')  # High-symmetry path labels go here
        ax.grid(False)
        
        if y_limits:
            ax.set_ylim(y_limits)
    
    def _set_high_symmetry_ticks(self, ax: mpl_axes.Axes,
                                distances: List[np.ndarray]):
        """Set x-ticks and labels at high-symmetry points.
        
        Args:
            ax: Matplotlib axis
            distances: List of distance arrays from band_dict
        """
        # Collect unique positions and their labels
        positions = []
        labels = []
        seen_positions = set()
        
        for dist_arr in distances:
            # Start of segment
            start = dist_arr[0]
            if start not in seen_positions:
                positions.append(start)
                seen_positions.add(start)
            # End of segment
            end = dist_arr[-1]
            if end not in seen_positions:
                positions.append(end)
                seen_positions.add(end)
        
        # Set ticks
        ax.set_xticks(positions)
        ax.set_xticklabels([''] * len(positions))  # Labels added via annotations if needed
        
        # Add vertical lines at high-symmetry points
        for pos in positions:
            ax.axvline(x=pos, color=self.get_color('grid'), lw=0.5, alpha=0.5)
