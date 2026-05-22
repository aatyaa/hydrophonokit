"""
=============================================================================
  HydroPhonoKit Visualization — Base Plotter

  Provides base class with common utilities for all phonon plot types:
    - Theme application
    - Figure/axis management
    - Label formatting (THz, cm^-1, meV)
    - Stability badges
    - High-symmetry point markers
    - Multi-panel composition
    - Save/export utilities

  Scientific Foundation:
    All plots follow standard conventions in condensed matter physics:
      - Frequency units: THz (primary), cm^-1 (spectroscopy), meV (energy)
      - High-symmetry points: Gamma, X, M, K, L, W, U (standard notation)
      - Stability indicators: imaginary modes highlighted
      - Zero-frequency reference line for acoustic modes

  References:
    [1] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy plotting
    [2] Hinuma et al., Comput. Mater. Sci. 140, 2017 -- seekpath paths
    [3] APS Style Guide -- Figure preparation standards
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

from .themes import get_theme, get_element_color, ELEMENT_COLORS, DEFAULT_THEME


# ============================================================================
# UNIT CONVERSIONS
# ============================================================================

THZ_TO_CM = 33.356409519815  # 1 THz = 33.356 cm^-1
THZ_TO_MEV = 4.135667696     # 1 THz = 4.136 meV
CM_TO_THZ = 1.0 / THZ_TO_CM
MEV_TO_THZ = 1.0 / THZ_TO_MEV


def freq_to_cm(freq_thz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert frequency from THz to cm^-1."""
    return freq_thz * THZ_TO_CM


def freq_to_meV(freq_thz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert frequency from THz to meV."""
    return freq_thz * THZ_TO_MEV


# ============================================================================
# HIGH-SYMMETRY POINTS
# ============================================================================

# Standard high-symmetry point labels
HIGH_SYMMETRY_LABELS = {
    'GAMMA': r'$\Gamma$',
    'X': 'X', 'Y': 'Y', 'Z': 'Z',
    'M': 'M', 'K': 'K', 'L': 'L',
    'W': 'W', 'U': 'U', 'A': 'A',
    'H': 'H', 'P': 'P', 'N': 'N',
    'S': 'S', 'R': 'R', 'T': 'T',
}


# ============================================================================
# BASE PLOTTER CLASS
# ============================================================================

class BasePlotter:
    """Base class for all phonon plotters.
    
    Provides common utilities:
      - Theme management
      - Figure/axis creation
      - Unit conversion
      - Label formatting
      - Save/export
      - High-symmetry markers
      - Stability badges
    
    Usage:
        plotter = BasePlotter(theme='nature')
        fig, ax = plotter.create_figure(figsize=(8, 6))
        ax.plot(x, y)
        plotter.save(fig, 'output.png')
    """
    
    def __init__(self, theme: str = DEFAULT_THEME):
        """
        Args:
            theme: Theme name ('nature', 'science', 'prl', 'acs', 'presentation', 'minimal')
        """
        self.theme_name = theme
        self.theme = get_theme(theme)
        self._figures: List[Figure] = []
        self._apply_theme()
    
    def _apply_theme(self):
        """Apply theme settings to matplotlib rcParams."""
        t = self.theme
        
        # Build rcParams update dict
        rc_updates = {
            'font.family': t['font']['family'],
            'font.size': t['font']['size'],
            'font.weight': t['font']['weight'],
            'figure.dpi': t['figure']['dpi'],
            'figure.facecolor': t['figure']['facecolor'],
            'figure.edgecolor': t['figure']['edgecolor'],
            'axes.linewidth': t['axes']['linewidth'],
            'axes.labelsize': t['axes']['labelsize'],
            'axes.titlesize': t['axes']['titlesize'],
            'axes.labelcolor': t['axes']['labelcolor'],
            'axes.titlecolor': t['axes']['titlecolor'],
            'legend.fontsize': t['legend']['fontsize'],
            'legend.framealpha': t['legend']['framealpha'],
            'lines.linewidth': t['lines']['linewidth'],
            'lines.markersize': t['lines']['markersize'],
            'savefig.dpi': t['savefig']['dpi'],
            'savefig.format': t['savefig']['format'],
        }
        
        # Add optional legend settings
        if 'facecolor' in t['legend']:
            rc_updates['legend.facecolor'] = t['legend']['facecolor']
        if 'edgecolor' in t['legend']:
            rc_updates['legend.edgecolor'] = t['legend']['edgecolor']
        
        plt.rcParams.update(rc_updates)
    
    def create_figure(self, figsize: Tuple[float, float] = (8, 6),
                     nrows: int = 1, ncols: int = 1,
                     sharex: bool = False, sharey: bool = False,
                     **kwargs) -> Tuple[Figure, Any]:
        """Create a new figure with theme settings applied.
        
        Args:
            figsize: Figure size (width, height) in inches
            nrows: Number of rows
            ncols: Number of columns
            sharex: Share x-axis between subplots
            sharey: Share y-axis between subplots
            **kwargs: Additional kwargs for plt.subplots()
        
        Returns:
            (figure, axes) tuple
        """
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=figsize,
            sharex=sharex, sharey=sharey,
            **kwargs
        )
        
        self._figures.append(fig)
        return fig, axes
    
    def add_high_symmetry_markers(self, ax: mpl_axes.Axes,
                                  positions: List[float],
                                  color: str = None):
        """Add vertical lines at high-symmetry q-points.
        
        Args:
            ax: Matplotlib axis
            positions: List of q-point positions
            color: Line color (defaults to theme grid color)
        """
        if color is None:
            color = self.theme['colors']['grid']
        
        for pos in positions:
            ax.axvline(x=pos, color=color, lw=0.5, ls='-', alpha=0.5)
    
    def add_zero_line(self, ax: mpl_axes.Axes, color: str = None,
                     lw: float = 1.0, alpha: float = 0.7):
        """Add horizontal line at zero frequency.
        
        Args:
            ax: Matplotlib axis
            color: Line color (defaults to theme zero_line)
            lw: Line width
            alpha: Line transparency
        """
        if color is None:
            color = self.theme['colors']['zero_line']
        
        ax.axhline(y=0, color=color, lw=lw, ls='--', alpha=alpha)
    
    def add_stability_badge(self, ax: mpl_axes.Axes, min_freq: float,
                           text_pos: Tuple[float, float] = (0.02, 0.02),
                           fontsize: int = None):
        """Add dynamical stability badge to plot.
        
        Args:
            ax: Matplotlib axis
            min_freq: Minimum phonon frequency (THz)
            text_pos: Text position (x, y) in axes coordinates
            fontsize: Font size (defaults to theme size)
        """
        if fontsize is None:
            fontsize = self.theme['legend']['fontsize']
        
        if min_freq < -0.5:
            text = f'Imaginary: {min_freq:.2f} THz'
            color = self.theme['colors']['band_imaginary']
            bg_color = '#FEE2E2' if self.theme_name != 'presentation' else '#3D0000'
        else:
            text = 'Dynamically Stable'
            color = self.theme['colors']['tertiary']
            bg_color = '#D1FAE5' if self.theme_name != 'presentation' else '#003D1D'
        
        ax.text(text_pos[0], text_pos[1], text,
                transform=ax.transAxes, color=color, fontsize=fontsize,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.9))
    
    def format_frequency_axis(self, ax: mpl_axes.Axes, unit: str = 'THz',
                             ylabel: str = 'Frequency'):
        """Format y-axis with appropriate frequency unit.
        
        Args:
            ax: Matplotlib axis
            unit: Frequency unit ('THz', 'cm^-1', 'meV')
            ylabel: Y-axis label text
        """
        if unit == 'THz':
            ax.set_ylabel(f'{ylabel} (THz)')
        elif unit == 'cm^-1':
            ax.set_ylabel(f'{ylabel} (cm$^{{-1}}$)')
            # Convert existing y-ticks
            ticks = ax.get_yticks()
            ax.set_yticklabels([f'{t * THZ_TO_CM:.0f}' for t in ticks])
        elif unit == 'meV':
            ax.set_ylabel(f'{ylabel} (meV)')
            ticks = ax.get_yticks()
            ax.set_yticklabels([f'{t * THZ_TO_MEV:.1f}' for t in ticks])
    
    def get_color(self, name: str) -> str:
        """Get color from current theme.
        
        Args:
            name: Color name ('primary', 'secondary', 'band', etc.)
        
        Returns:
            Hex color code
        """
        return self.theme['colors'].get(name, '#888888')
    
    def get_element_color(self, element: str) -> str:
        """Get color for an element symbol.
        
        Args:
            element: Element symbol (e.g., 'H', 'Mg', 'B')
        
        Returns:
            Hex color code
        """
        return get_element_color(element)
    
    def save(self, fig: Figure, path: str, **kwargs):
        """Save figure with theme settings.
        
        Args:
            fig: Matplotlib figure
            path: Output file path (extension determines format if not specified)
            **kwargs: Additional kwargs for fig.savefig()
        """
        # Merge theme savefig settings with kwargs
        save_kwargs = {**self.theme['savefig'], **kwargs}
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        fig.savefig(path, **save_kwargs)
    
    def save_all(self, output_dir: str, base_names: List[str] = None,
                format: str = None):
        """Save all tracked figures.
        
        Args:
            output_dir: Directory to save figures
            base_names: List of base names (defaults to 'figure_N')
            format: Override format (defaults to theme format)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if base_names is None:
            base_names = [f'figure_{i}' for i in range(len(self._figures))]
        
        fmt = format or self.theme['savefig']['format']
        
        for i, fig in enumerate(self._figures):
            name = base_names[i] if i < len(base_names) else f'figure_{i}'
            path = os.path.join(output_dir, f'{name}.{fmt}')
            self.save(fig, path)
    
    def close_all(self):
        """Close all tracked figures."""
        for fig in self._figures:
            plt.close(fig)
        self._figures.clear()
    
    def get_figures(self) -> List[Figure]:
        """Get list of tracked figures.
        
        Returns:
            List of matplotlib Figure objects
        """
        return self._figures.copy()
