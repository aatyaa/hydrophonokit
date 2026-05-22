"""
=============================================================================
  HydroPhonoKit Visualization — Density of States (DOS) Plots

  Implements all DOS plot types for phonon analysis:
    1. Total DOS - Complete phonon density of states
    2. Partial DOS (per element) - Element-projected contributions
    3. Stacked DOS - Stacked area plot of element contributions
    4. Cumulative DOS - Integrated DOS for ZPE and thermodynamics
    5. Mode-Projected DOS - Acoustic vs optical mode separation
    6. Log-Scale DOS - Logarithmic scale for fine detail visibility
    7. Hydrogen-Specific DOS - H-only DOS with mode labels
    8. H-Mode Decomposition - Librational/bending/stretching fractions

  Scientific Foundation:
    The phonon density of states g(omega) describes the number of vibrational
    modes per unit frequency interval. It's fundamental for:
      - Thermodynamic properties (F, S, Cv via g(omega) integrals)
      - Zero-point energy calculation
      - Debye temperature estimation
      - Mode character analysis (acoustic vs optical)
      - Element-specific vibrational analysis
      - Hydrogen bonding environment characterization

    Key features:
      - Acoustic modes: Low-frequency region (0 to few THz)
      - Optical modes: Higher frequency bands
      - Van Hove singularities: Peaks in DOS (flat band regions)
      - Mode gaps: Frequency ranges with no allowed modes
      - Element contributions: Mass-weighted projection onto DOS

  References:
    [1] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy DOS
    [2] Grimvall, Thermophysical Properties of Materials (1999)
    [3] Nakamoto, IR/Raman Spectra of Inorganic Compounds (2009)
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

from .base_plotter import BasePlotter, freq_to_cm, freq_to_meV, THZ_TO_CM
from .themes import get_element_color, ELEMENT_COLORS


# ============================================================================
# DOS PLOTTER CLASS
# ============================================================================

class DOSPlotter(BasePlotter):
    """Specialized plotter for phonon density of states.
    
    Provides methods for all DOS plot types.
    """
    
    def plot_total_dos(self, freq: np.ndarray, total_dos: np.ndarray,
                      formula: str = 'Material',
                      unit: str = 'THz',
                      show_cumulative: bool = False,
                      show_zpe_marker: bool = False,
                      ax: mpl_axes.Axes = None,
                      figsize: Tuple[float, float] = (8, 6),
                      title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot total phonon density of states.
        
        Args:
            freq: Frequency array (THz)
            total_dos: Total DOS array (states/THz)
            formula: Material formula for title
            unit: Frequency unit ('THz', 'cm^-1', 'meV')
            show_cumulative: Overlay cumulative DOS
            show_zpe_marker: Mark zero-point energy frequency
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
        
        # Main DOS plot
        ax.fill_between(freq, total_dos, alpha=0.3, color=self.get_color('dos_fill'))
        ax.plot(freq, total_dos, color=self.get_color('dos_line'), lw=1.5, label='Total DOS')
        
        # Cumulative DOS overlay
        if show_cumulative:
            cum_dos = np.cumsum(total_dos) * np.mean(np.diff(freq))
            ax2 = ax.twinx()
            ax2.plot(freq, cum_dos, color=self.get_color('secondary'), lw=1.0, ls='--',
                    label='Cumulative DOS')
            ax2.set_ylabel('Cumulative DOS (states)', color=self.get_color('secondary'))
            ax2.tick_params(axis='y', labelcolor=self.get_color('secondary'))
            ax2.legend(loc='upper left')
        
        # ZPE marker
        if show_zpe_marker:
            # Find frequency with maximum DOS contribution to ZPE
            zpe_contrib = total_dos * freq
            zpe_peak_idx = np.argmax(zpe_contrib)
            zpe_freq = freq[zpe_peak_idx]
            ax.axvline(x=zpe_freq, color=self.get_color('tertiary'), ls=':', alpha=0.7)
            ax.text(zpe_freq, max(total_dos) * 0.9,
                   f'ZPE peak: {zpe_freq:.2f} THz',
                   color=self.get_color('tertiary'), fontsize=9, ha='center')
        
        # Format axes
        self._format_dos_axes(ax, freq, unit=unit, dos_max=np.max(total_dos))
        
        # Title and labels
        if title is None:
            title = f'{formula} - Total Phonon DOS'
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_partial_dos(self, freq: np.ndarray, pdos: np.ndarray,
                        sym_to_idx: Dict[str, List[int]],
                        formula: str = 'Material',
                        unit: str = 'THz',
                        stacked: bool = False,
                        elements: List[str] = None,
                        normalize: bool = False,
                        ax: mpl_axes.Axes = None,
                        figsize: Tuple[float, float] = (8, 6),
                        title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot element-projected partial DOS.
        
        Args:
            freq: Frequency array (THz)
            pdos: Partial DOS array (n_atoms, n_freq)
            sym_to_idx: Dict mapping element symbol to atom indices
            formula: Material formula
            unit: Frequency unit
            stacked: Stack element contributions (True) or overlay (False)
            elements: List of elements to show (None = all)
            normalize: Normalize each element's DOS to unit area
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
        
        elements_to_show = elements or list(sym_to_idx.keys())
        
        if stacked:
            # Stacked area plot
            dos_components = []
            labels = []
            
            for elem in elements_to_show:
                if elem not in sym_to_idx:
                    continue
                
                indices = sym_to_idx[elem]
                elem_dos = np.sum([pdos[i] for i in indices if i < pdos.shape[0]], axis=0)
                
                if normalize:
                    integral = np.trapz(elem_dos, freq)
                    if integral > 0:
                        elem_dos = elem_dos / integral
                
                dos_components.append(elem_dos)
                labels.append(elem)
            
            if dos_components:
                colors = [self.get_element_color(e) for e in labels]
                ax.stackplot(freq, dos_components, labels=labels, colors=colors, alpha=0.7)
                ax.legend(loc='upper right')
        else:
            # Overlay plot
            for elem in elements_to_show:
                if elem not in sym_to_idx:
                    continue
                
                indices = sym_to_idx[elem]
                elem_dos = np.sum([pdos[i] for i in indices if i < pdos.shape[0]], axis=0)
                
                if normalize:
                    integral = np.trapz(elem_dos, freq)
                    if integral > 0:
                        elem_dos = elem_dos / integral
                
                c = self.get_element_color(elem)
                ax.fill_between(freq, elem_dos, alpha=0.2, color=c)
                ax.plot(freq, elem_dos, color=c, lw=1.5, label=elem)
            
            ax.legend(ncol=min(len(elements_to_show), 4), loc='upper right')
        
        # Format axes
        total_dos = np.sum([pdos[i] for i in range(pdos.shape[0])], axis=0)
        self._format_dos_axes(ax, freq, unit=unit, dos_max=np.max(total_dos))
        
        if title is None:
            title = f'{formula} - Partial Phonon DOS'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_cumulative_dos(self, freq: np.ndarray, total_dos: np.ndarray,
                           formula: str = 'Material',
                           unit: str = 'THz',
                           show_zpe: bool = True,
                           show_debye: bool = False,
                           ax: mpl_axes.Axes = None,
                           figsize: Tuple[float, float] = (8, 6),
                           title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot cumulative (integrated) DOS.
        
        Useful for:
          - Zero-point energy calculation
          - Counting modes below a frequency
          - Debye frequency estimation
        
        Args:
            freq: Frequency array (THz)
            total_dos: Total DOS array (states/THz)
            formula: Material formula
            unit: Frequency unit
            show_zpe: Mark ZPE frequency (where integral = 0.5 * total modes)
            show_debye: Mark Debye frequency estimate
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
        
        # Calculate cumulative DOS
        df = np.mean(np.diff(freq))
        cum_dos = np.cumsum(total_dos) * df
        total_modes = cum_dos[-1]
        
        # Main plot
        ax.plot(freq, cum_dos, color=self.get_color('primary'), lw=2)
        ax.fill_between(freq, 0, cum_dos, alpha=0.2, color=self.get_color('dos_fill'))
        
        # Total modes line
        ax.axhline(y=total_modes, color='gray', ls='--', alpha=0.5,
                  label=f'Total modes: {total_modes:.0f}')
        
        # ZPE marker
        if show_zpe:
            # ZPE corresponds to half the modes (roughly)
            zpe_idx = np.argmin(np.abs(cum_dos - total_modes * 0.5))
            zpe_freq = freq[zpe_idx]
            ax.axvline(x=zpe_freq, color=self.get_color('tertiary'), ls=':', alpha=0.7)
            ax.text(zpe_freq, total_modes * 0.52,
                   f'ZPE: {zpe_freq:.2f} THz',
                   color=self.get_color('tertiary'), fontsize=9)
        
        # Debye frequency estimate
        if show_debye:
            # Debye frequency where cumulative = 3N (acoustic modes)
            n_acoustic = 3 * np.round(total_modes / 3)  # Rough estimate
            debye_idx = np.argmin(np.abs(cum_dos - n_acoustic))
            debye_freq = freq[debye_idx]
            ax.axvline(x=debye_freq, color=self.get_color('secondary'), ls='-.', alpha=0.7)
            ax.text(debye_freq, n_acoustic * 1.05,
                   f'Debye: {debye_freq:.2f} THz',
                   color=self.get_color('secondary'), fontsize=9)
        
        # Format axes
        ax.set_xlabel(f'Frequency ({unit})')
        ax.set_ylabel('Cumulative DOS (states)')
        ax.legend(loc='lower right')
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        ax.set_xlim(0, np.max(freq) * 1.05)
        ax.set_ylim(0, total_modes * 1.1)
        
        if title is None:
            title = f'{formula} - Cumulative Phonon DOS'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_mode_projected_dos(self, freq: np.ndarray, pdos: np.ndarray,
                               sym_to_idx: Dict[str, List[int]],
                               formula: str = 'Material',
                               unit: str = 'THz',
                               acoustic_cutoff: float = 2.0,
                               ax: mpl_axes.Axes = None,
                               figsize: Tuple[float, float] = (8, 6),
                               title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot mode-projected DOS (acoustic vs optical).
        
        Separates DOS into:
          - Acoustic modes: Low-frequency (< acoustic_cutoff THz)
          - Optical modes: Higher frequency
        
        Args:
            freq: Frequency array (THz)
            pdos: Partial DOS array (n_atoms, n_freq)
            sym_to_idx: Dict mapping element symbol to atom indices
            formula: Material formula
            unit: Frequency unit
            acoustic_cutoff: Frequency cutoff for acoustic modes (THz)
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
        
        total_dos = np.sum([pdos[i] for i in range(pdos.shape[0])], axis=0)
        
        # Separate acoustic and optical
        acoustic_mask = freq < acoustic_cutoff
        optical_mask = freq >= acoustic_cutoff
        
        acoustic_dos = np.zeros_like(total_dos)
        optical_dos = np.zeros_like(total_dos)
        acoustic_dos[acoustic_mask] = total_dos[acoustic_mask]
        optical_dos[optical_mask] = total_dos[optical_mask]
        
        # Plot
        ax.fill_between(freq, acoustic_dos, alpha=0.5, color=self.get_color('tertiary'),
                       label=f'Acoustic (< {acoustic_cutoff} THz)')
        ax.fill_between(freq, acoustic_dos + optical_dos, acoustic_dos,
                       alpha=0.5, color=self.get_color('secondary'),
                       label='Optical')
        
        ax.plot(freq, total_dos, color='black', lw=1.0, alpha=0.7, label='Total')
        
        # Separator line
        ax.axvline(x=acoustic_cutoff, color='gray', ls='--', alpha=0.5)
        
        # Format axes
        self._format_dos_axes(ax, freq, unit=unit, dos_max=np.max(total_dos))
        
        if title is None:
            title = f'{formula} - Mode-Projected DOS'
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_log_dos(self, freq: np.ndarray, total_dos: np.ndarray,
                    formula: str = 'Material',
                    unit: str = 'THz',
                    min_dos: float = 1e-3,
                    ax: mpl_axes.Axes = None,
                    figsize: Tuple[float, float] = (8, 6),
                    title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot DOS on logarithmic scale.
        
        Useful for seeing fine details in low-DOS regions.
        
        Args:
            freq: Frequency array (THz)
            total_dos: Total DOS array (states/THz)
            formula: Material formula
            unit: Frequency unit
            min_dos: Minimum DOS value for log scale
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
        
        # Ensure positive values for log
        dos_plot = np.maximum(total_dos, min_dos)
        
        ax.semilogy(freq, dos_plot, color=self.get_color('primary'), lw=1.5)
        ax.fill_between(freq, min_dos, dos_plot, alpha=0.2, color=self.get_color('dos_fill'))
        
        # Format axes
        ax.set_xlabel(f'Frequency ({unit})')
        ax.set_ylabel('DOS (states/THz, log scale)')
        ax.grid(alpha=self.theme['axes']['grid_alpha'], which='both')
        ax.set_xlim(0, np.max(freq) * 1.05)
        
        if title is None:
            title = f'{formula} - Phonon DOS (Log Scale)'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_hydrogen_dos(self, freq: np.ndarray, pdos: np.ndarray,
                         sym_to_idx: Dict[str, List[int]],
                         formula: str = 'Material',
                         unit: str = 'THz',
                         show_mode_regions: bool = True,
                         ax: mpl_axes.Axes = None,
                         figsize: Tuple[float, float] = (10, 6),
                         title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot hydrogen-specific DOS with mode decomposition.
        
        Highlights:
          - Librational modes (5-21 THz)
          - Bending modes (21-50 THz)
          - Stretching modes (50-100 THz)
        
        Args:
            freq: Frequency array (THz)
            pdos: Partial DOS array (n_atoms, n_freq)
            sym_to_idx: Dict mapping element symbol to atom indices
            formula: Material formula
            unit: Frequency unit
            show_mode_regions: Highlight H-mode regions
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
        
        # Extract H DOS
        if 'H' not in sym_to_idx:
            raise ValueError("No hydrogen atoms found in sym_to_idx")
        
        h_indices = sym_to_idx['H']
        h_dos = np.sum([pdos[i] for i in h_indices if i < pdos.shape[0]], axis=0)
        
        # Main H DOS plot
        ax.fill_between(freq, h_dos, alpha=0.4, color='#F39C12')
        ax.plot(freq, h_dos, color='#E67E22', lw=2, label='H DOS')
        
        # Highlight mode regions
        if show_mode_regions:
            mode_regions = [
                (5, 21, 'Librational', '#9B59B6'),
                (21, 50, 'Bending', '#3498DB'),
                (50, 100, 'Stretching', '#E74C3C'),
            ]
            
            for f_min, f_max, label, color in mode_regions:
                mask = (freq >= f_min) & (freq <= f_max)
                if np.any(mask):
                    ax.axvspan(f_min, f_max, alpha=0.1, color=color)
                    # Label at peak
                    peak_idx = np.argmax(h_dos[mask])
                    peak_freq = freq[mask][peak_idx]
                    peak_height = h_dos[mask][peak_idx]
                    ax.text(peak_freq, peak_height * 1.1, label,
                           color=color, fontsize=9, ha='center', fontweight='bold')
        
        # Format axes
        total_dos_max = np.max(h_dos)
        self._format_dos_axes(ax, freq, unit=unit, dos_max=total_dos_max)
        
        if title is None:
            title = f'{formula} - Hydrogen DOS with Mode Decomposition'
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_h_mode_decomposition(self, freq: np.ndarray, h_dos: np.ndarray,
                                 formula: str = 'Material',
                                 unit: str = 'THz',
                                 show_fractions: bool = True,
                                 ax: mpl_axes.Axes = None,
                                 figsize: Tuple[float, float] = (10, 6),
                                 title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot hydrogen mode decomposition as filled regions.
        
        Shows:
          - Librational fraction
          - Bending fraction
          - Stretching fraction
        
        Args:
            freq: Frequency array (THz)
            h_dos: Hydrogen DOS array (states/THz)
            formula: Material formula
            unit: Frequency unit
            show_fractions: Show percentage labels
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
        
        # Mode regions
        lib_mask = (freq > 5) & (freq < 21)
        bend_mask = (freq > 21) & (freq < 50)
        stretch_mask = (freq > 50) & (freq < 100)
        
        # Calculate fractions
        if show_fractions:
            h_total = np.trapz(h_dos[freq > 0], freq[freq > 0])
            h_lib = np.trapz(h_dos[lib_mask], freq[lib_mask]) if np.any(lib_mask) else 0
            h_bend = np.trapz(h_dos[bend_mask], freq[bend_mask]) if np.any(bend_mask) else 0
            h_stretch = np.trapz(h_dos[stretch_mask], freq[stretch_mask]) if np.any(stretch_mask) else 0
            
            if h_total > 0:
                lib_pct = h_lib / h_total * 100
                bend_pct = h_bend / h_total * 100
                stretch_pct = h_stretch / h_total * 100
            else:
                lib_pct = bend_pct = stretch_pct = 0
        
        # Plot filled regions
        ax.fill_between(freq[lib_mask], h_dos[lib_mask], alpha=0.4,
                       color='#9B59B6', label=f'Librational ({lib_pct:.1f}%)' if show_fractions else 'Librational')
        ax.fill_between(freq[bend_mask], h_dos[bend_mask], alpha=0.4,
                       color='#3498DB', label=f'Bending ({bend_pct:.1f}%)' if show_fractions else 'Bending')
        ax.fill_between(freq[stretch_mask], h_dos[stretch_mask], alpha=0.4,
                       color='#E74C3C', label=f'Stretching ({stretch_pct:.1f}%)' if show_fractions else 'Stretching')
        ax.plot(freq, h_dos, color='#F39C12', lw=1.5, label='H Total')
        
        # Find peak
        if np.any(stretch_mask):
            peak_idx = np.argmax(h_dos[stretch_mask])
            peak_freq = freq[stretch_mask][peak_idx]
            ax.axvline(x=peak_freq, color='#E74C3C', ls='--', alpha=0.7)
            ax.text(peak_freq, max(h_dos) * 0.85,
                   f'Peak: {peak_freq:.1f} THz\n({peak_freq * THZ_TO_CM:.0f} cm$^{{-1}}$)',
                   fontsize=9, color='#E74C3C', fontweight='bold', ha='center')
        
        # Format axes
        total_dos_max = np.max(h_dos)
        self._format_dos_axes(ax, freq, unit=unit, dos_max=total_dos_max)
        
        if title is None:
            title = f'{formula} - Hydrogen Mode Decomposition'
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _format_dos_axes(self, ax: mpl_axes.Axes, freq: np.ndarray,
                        unit: str = 'THz', dos_max: float = None):
        """Format DOS axes with proper labels and limits.
        
        Args:
            ax: Matplotlib axis
            freq: Frequency array
            unit: Frequency unit
            dos_max: Maximum DOS value for y-limit
        """
        if unit == 'THz':
            ax.set_xlabel('Frequency (THz)')
        elif unit == 'cm^-1':
            ax.set_xlabel('Frequency (cm$^{-1}$)')
            # Convert x-ticks
            ticks = ax.get_xticks()
            ax.set_xticklabels([f'{t * THZ_TO_CM:.0f}' for t in ticks])
        elif unit == 'meV':
            from .base_plotter import THZ_TO_MEV
            ax.set_xlabel('Frequency (meV)')
            ticks = ax.get_xticks()
            ax.set_xticklabels([f'{t * THZ_TO_MEV:.1f}' for t in ticks])
        
        ax.set_ylabel('DOS (states/THz)')
        ax.set_xlim(0, np.max(freq) * 1.05)
        
        if dos_max is not None:
            ax.set_ylim(0, dos_max * 1.1)
        
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
