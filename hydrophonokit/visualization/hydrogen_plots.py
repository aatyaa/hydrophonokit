"""
=============================================================================
  HydroPhonoKit Visualization — Hydrogen-Specific Plots

  Implements specialized hydrogen storage analysis plots:
    1. H-DOS with Mode Labels - Hydrogen DOS with lib/bend/stretch regions
    2. B-H Stretch Region Zoom - Focused view of stretching modes
    3. H-Mode Fraction Pie Chart - Percentage breakdown of modes
    4. Hydride Type Identification - Comparison with known hydrides

  Scientific Foundation:
    Hydrogen in solids exhibits characteristic vibrational modes that are
    diagnostic of the bonding environment:

    Librational modes (5-21 THz, ~167-700 cm⁻¹):
      - Hindered rotation of H₂ or BH₄ units
      - Sensitive to local symmetry and crystal field

    Bending modes (21-50 THz, ~700-1668 cm⁻¹):
      - H-X-H bending vibrations
      - Bond angle information

    Stretching modes (50-100 THz, ~1668-3336 cm⁻¹):
      - X-H stretching vibrations
      - Direct measure of bond strength
      - B-H: 2200-2600 cm⁻¹ (borohydrides)
      - Al-H: 1700-1850 cm⁻¹ (alanates)
      - N-H: 3100-3500 cm⁻¹ (amides)
      - Mg-H: 1100-1400 cm⁻¹ (magnesium hydride)

  References:
    [1] Nakamoto, IR/Raman Spectra of Inorganic Compounds (2009)
    [2] Bogdanovic et al., J. Alloys Compd. 382, 1 (2004) -- MgH₂
    [3] Züttel et al., Nature Mater. 4, 673 (2005) -- LiBH₄
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

from .base_plotter import BasePlotter, THZ_TO_CM


# ============================================================================
# HYDRIDE REFERENCE DATA
# ============================================================================

HYDRIDE_REFERENCES = {
    'MgH₂': {'stretch_cm': (1100, 1400), 'type': 'Ionic', 'color': '#E74C3C'},
    'LiBH₄': {'stretch_cm': (2200, 2600), 'type': 'Borohydride', 'color': '#3498DB'},
    'NaAlH₄': {'stretch_cm': (1700, 1850), 'type': 'Alanate', 'color': '#2ECC71'},
    'LiNH₂': {'stretch_cm': (3100, 3500), 'type': 'Amide', 'color': '#9B59B6'},
    'CaH₂': {'stretch_cm': (1200, 1400), 'type': 'Ionic', 'color': '#E67E22'},
    'NaH': {'stretch_cm': (1150, 1250), 'type': 'Ionic', 'color': '#F39C12'},
}


# ============================================================================
# HYDROGEN PLOTTER CLASS
# ============================================================================

class HydrogenPlotter(BasePlotter):
    """Specialized plotter for hydrogen storage analysis.
    
    Provides methods for hydrogen-specific visualization.
    """
    
    def plot_h_dos_with_modes(self, freq: np.ndarray, h_dos: np.ndarray,
                             formula: str = 'Material',
                             unit: str = 'THz',
                             show_mode_regions: bool = True,
                             ax: mpl_axes.Axes = None,
                             figsize: Tuple[float, float] = (10, 6),
                             title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot hydrogen DOS with mode region labels.
        
        Highlights:
          - Librational region (5-21 THz)
          - Bending region (21-50 THz)
          - Stretching region (50-100 THz)
        
        Args:
            freq: Frequency array (THz)
            h_dos: Hydrogen DOS array (states/THz)
            formula: Material formula
            unit: Frequency unit
            show_mode_regions: Highlight mode regions
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
        
        # Main H DOS plot
        ax.fill_between(freq, h_dos, alpha=0.5, color='#F39C12')
        ax.plot(freq, h_dos, color='#E67E22', lw=2, label='H DOS')
        
        # Mode regions
        if show_mode_regions:
            mode_regions = [
                (5, 21, 'Librational', '#9B59B6'),
                (21, 50, 'Bending', '#3498DB'),
                (50, 100, 'Stretching', '#E74C3C'),
            ]
            
            for f_min, f_max, label, color in mode_regions:
                mask = (freq >= f_min) & (freq <= f_max)
                if np.any(mask):
                    ax.axvspan(f_min, f_max, alpha=0.15, color=color)
                    
                    # Find peak in this region
                    region_dos = h_dos[mask]
                    region_freq = freq[mask]
                    if len(region_dos) > 0 and np.max(region_dos) > 0:
                        peak_idx = np.argmax(region_dos)
                        peak_freq = region_freq[peak_idx]
                        peak_height = region_dos[peak_idx]
                        
                        ax.annotate(label, xy=(peak_freq, peak_height),
                                   xytext=(peak_freq, peak_height * 1.3),
                                   ha='center', fontsize=10, color=color,
                                   fontweight='bold',
                                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        
        # Format axes
        self._format_h_dos_axes(ax, freq, unit=unit, dos_max=np.max(h_dos))
        
        if title is None:
            title = f'{formula} - Hydrogen DOS with Mode Decomposition'
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_stretch_zoom(self, freq: np.ndarray, h_dos: np.ndarray,
                         formula: str = 'Material',
                         stretch_range: Tuple[float, float] = (50, 100),
                         unit: str = 'THz',
                         ax: mpl_axes.Axes = None,
                         figsize: Tuple[float, float] = (8, 6),
                         title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot zoomed view of stretching region.
        
        Focuses on the X-H stretching modes to identify hydride type.
        
        Args:
            freq: Frequency array (THz)
            h_dos: Hydrogen DOS array
            formula: Material formula
            stretch_range: (min, max) THz for stretching region
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
        
        # Filter to stretch range
        mask = (freq >= stretch_range[0]) & (freq <= stretch_range[1])
        freq_stretch = freq[mask]
        h_dos_stretch = h_dos[mask]
        
        # Plot
        ax.fill_between(freq_stretch, h_dos_stretch, alpha=0.5, color='#E74C3C')
        ax.plot(freq_stretch, h_dos_stretch, color='#C0392B', lw=2)
        
        # Find peak
        if len(h_dos_stretch) > 0:
            peak_idx = np.argmax(h_dos_stretch)
            peak_freq = freq_stretch[peak_idx]
            peak_height = h_dos_stretch[peak_idx]
            peak_cm = peak_freq * THZ_TO_CM
            
            ax.axvline(x=peak_freq, color='#E74C3C', ls='--', alpha=0.7)
            ax.text(peak_freq, peak_height * 0.9,
                   f'Peak: {peak_freq:.1f} THz\n({peak_cm:.0f} cm$^{{-1}}$)',
                   ha='center', fontsize=10, color='#E74C3C', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hydride type annotation
            hydride_type = self._identify_hydride_type(peak_cm)
            ax.text(0.02, 0.95, f'Hydride Type: {hydride_type}',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'Frequency ({unit})')
        ax.set_ylabel('H DOS (states/THz)')
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if title is None:
            title = f'{formula} - H Stretching Region'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_h_mode_pie_chart(self, freq: np.ndarray, h_dos: np.ndarray,
                             formula: str = 'Material',
                             ax: mpl_axes.Axes = None,
                             figsize: Tuple[float, float] = (6, 6),
                             title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot hydrogen mode fractions as pie chart.
        
        Shows the percentage of H-DOS in each mode region.
        
        Args:
            freq: Frequency array (THz)
            h_dos: Hydrogen DOS array
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
        
        # Calculate fractions
        lib_mask = (freq > 5) & (freq < 21)
        bend_mask = (freq > 21) & (freq < 50)
        stretch_mask = (freq > 50) & (freq < 100)
        
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
        
        # Pie chart
        labels = [f'Librational\n{lib_pct:.1f}%',
                 f'Bending\n{bend_pct:.1f}%',
                 f'Stretching\n{stretch_pct:.1f}%']
        sizes = [lib_pct, bend_pct, stretch_pct]
        colors = ['#9B59B6', '#3498DB', '#E74C3C']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90,
                                          pctdistance=0.85,
                                          wedgeprops=dict(width=0.5, edgecolor='white'))
        
        # Center text
        ax.text(0, 0, f'{formula}\nH-Modes', ha='center', va='center',
               fontsize=11, fontweight='bold')
        
        ax.set_aspect('equal')
        
        if title is None:
            title = f'{formula} - Hydrogen Mode Fractions'
        ax.set_title(title, fontweight='bold', pad=20)
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_hydride_type_comparison(self, peak_freq_cm: float,
                                    formula: str = 'Material',
                                    ax: mpl_axes.Axes = None,
                                    figsize: Tuple[float, float] = (8, 6),
                                    title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot hydride type identification with reference ranges.
        
        Compares the material's stretch frequency with known hydrides.
        
        Args:
            peak_freq_cm: Peak stretching frequency (cm⁻¹)
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
        
        # Plot reference ranges
        y_pos = 1
        for name, ref in HYDRIDE_REFERENCES.items():
            f_min, f_max = ref['stretch_cm']
            color = ref['color']
            
            ax.barh(y_pos, f_max - f_min, left=f_min, height=0.6,
                   color=color, alpha=0.6, edgecolor='black',
                   label=f'{name} ({ref["type"]})')
            y_pos += 1
        
        # Plot material's peak
        hydride_type = self._identify_hydride_type(peak_freq_cm)
        ax.axvline(x=peak_freq_cm, color='red', lw=2.5, ls='--',
                  label=f'{formula}: {peak_freq_cm:.0f} cm⁻¹')
        ax.text(peak_freq_cm, y_pos + 0.5, f'→ {hydride_type}',
               fontsize=10, ha='center', color='red', fontweight='bold')
        
        ax.set_xlabel('Stretching Frequency (cm⁻¹)')
        ax.set_yticks(range(1, y_pos))
        ax.set_yticklabels([''] * (y_pos - 1))
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=self.theme['axes']['grid_alpha'], axis='x')
        ax.set_xlim(800, 3800)
        
        if title is None:
            title = f'{formula} - Hydride Type Identification'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _identify_hydride_type(self, freq_cm: float) -> str:
        """Identify hydride type from stretching frequency.
        
        Args:
            freq_cm: Stretching frequency (cm⁻¹)
        
        Returns:
            Hydride type string
        """
        for name, ref in HYDRIDE_REFERENCES.items():
            f_min, f_max = ref['stretch_cm']
            if f_min <= freq_cm <= f_max:
                return f'{name} ({ref["type"]})'
        
        if freq_cm < 1100:
            return 'Metallic Hydride'
        elif freq_cm > 3500:
            return 'Molecular H₂'
        else:
            return 'Unknown'
    
    def _format_h_dos_axes(self, ax: mpl_axes.Axes, freq: np.ndarray,
                          unit: str = 'THz', dos_max: float = None):
        """Format H-DOS axes with proper labels and limits."""
        if unit == 'THz':
            ax.set_xlabel('Frequency (THz)')
        elif unit == 'cm^-1':
            ax.set_xlabel('Frequency (cm$^{-1}$)')
            ticks = ax.get_xticks()
            ax.set_xticklabels([f'{t * THZ_TO_CM:.0f}' for t in ticks])
        
        ax.set_ylabel('H DOS (states/THz)')
        ax.set_xlim(0, np.min([np.max(freq) * 1.05, 110]))
        
        if dos_max is not None:
            ax.set_ylim(0, dos_max * 1.1)
        
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
