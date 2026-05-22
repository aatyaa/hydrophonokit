"""
=============================================================================
  HydroPhonoKit Visualization — Multi-Panel Composition

  Implements multi-panel figure composition:
    1. Dashboard Layout - 2x2, 3x2, custom grid layouts
    2. Composite Figure Builder - Combine bands + DOS + thermo
    3. Shared Axes Support - Common x/y axes for related plots
    4. Figure Assembly - Final publication-ready composite figures

  Scientific Foundation:
    Composite figures are standard in publications:
      - Nature/Science: 2-4 panel figures
      - PRL: 2-3 panel figures
      - PRB: Up to 6 panels

    Key principles:
      - Consistent color scheme across panels
      - Shared axes where appropriate
      - Clear panel labels (a), (b), (c), ...
      - Balanced aspect ratios

  References:
    [1] Nature Publishing Guide -- Multi-panel figure standards
    [2] matplotlib.gridspec -- Grid layout management
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
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .base_plotter import BasePlotter


# ============================================================================
# MULTI-PANEL COMPOSER CLASS
# ============================================================================

class MultiPanelComposer(BasePlotter):
    """Composer for multi-panel publication-ready figures.
    
    Provides methods for:
      - Dashboard layouts (2x2, 3x2, custom)
      - Composite figure assembly
      - Shared axes configuration
      - Panel labeling
    """
    
    def create_dashboard_layout(self, n_panels: int = 4,
                               layout: str = '2x2',
                               figsize: Tuple[float, float] = (12, 10),
                               share_x: bool = False,
                               share_y: bool = False) -> Tuple[Figure, List[Axes]]:
        """Create multi-panel dashboard layout.
        
        Args:
            n_panels: Number of panels
            layout: Layout string ('2x2', '3x2', '1x3', etc.)
            figsize: Figure size (width, height) in inches
            share_x: Share x-axis between panels
            share_y: Share y-axis between panels
        
        Returns:
            (figure, list_of_axes) tuple
        """
        rows, cols = map(int, layout.split('x'))
        
        fig, axes = plt.subplots(
            nrows=rows, ncols=cols,
            figsize=figsize,
            sharex=share_x, sharey=share_y
        )
        
        # Ensure axes is always a list
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        self._figures.append(fig)
        return fig, axes
    
    def create_custom_grid(self, specs: List[Tuple[int, int, int, int]],
                          figsize: Tuple[float, float] = (12, 10)) -> Tuple[Figure, List[Axes]]:
        """Create custom grid layout with GridSpec.
        
        Args:
            specs: List of (row_start, col_start, row_span, col_span)
            figsize: Figure size
        
        Returns:
            (figure, list_of_axes) tuple
        """
        fig = plt.figure(figsize=figsize)
        
        # Determine grid size
        max_row = max(s[0] + s[2] for s in specs)
        max_col = max(s[1] + s[3] for s in specs)
        
        gs = gridspec.GridSpec(max_row, max_col, figure=fig,
                               hspace=0.3, wspace=0.3)
        
        axes = []
        for spec in specs:
            ax = fig.add_subplot(gs[spec[0]:spec[0]+spec[2], spec[1]:spec[1]+spec[3]])
            axes.append(ax)
        
        self._figures.append(fig)
        return fig, axes
    
    def create_composite_figure(self, plot_funcs: List[callable],
                               layout: str = '1x3',
                               figsize: Tuple[float, float] = (14, 5),
                               titles: List[str] = None,
                               share_x: bool = False) -> Tuple[Figure, List[Axes]]:
        """Create composite figure from plotting functions.
        
        Args:
            plot_funcs: List of functions that take (ax) and plot
            layout: Layout string
            figsize: Figure size
            titles: Panel titles
            share_x: Share x-axis
        
        Returns:
            (figure, list_of_axes) tuple
        """
        fig, axes = self.create_dashboard_layout(
            n_panels=len(plot_funcs),
            layout=layout,
            figsize=figsize,
            share_x=share_x
        )
        
        titles = titles or [''] * len(plot_funcs)
        
        for ax, func, title in zip(axes, plot_funcs, titles):
            func(ax)
            if title:
                ax.set_title(title, fontweight='bold', fontsize=12)
        
        fig.tight_layout()
        return fig, axes
    
    def add_panel_labels(self, fig: Figure, axes: List[Axes],
                        labels: List[str] = None,
                        fontsize: int = 12):
        """Add panel labels (a), (b), (c), ... to axes.
        
        Args:
            fig: Matplotlib Figure object
            axes: List of Axes objects
            labels: Custom labels (default: a, b, c, ...)
            fontsize: Label font size
        """
        if labels is None:
            labels = [chr(ord('a') + i) for i in range(len(axes))]
        
        for ax, label in zip(axes, labels):
            ax.text(0.02, 0.98, f'({label})',
                   transform=ax.transAxes,
                   fontsize=fontsize,
                   fontweight='bold',
                   va='top',
                   ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def create_standard_phonon_dashboard(self, formula: str = 'Material',
                                        plot_band: bool = True,
                                        plot_dos: bool = True,
                                        plot_thermo: bool = True,
                                        plot_gruneisen: bool = False) -> Tuple[Figure, List[Axes]]:
        """Create standard phonon analysis dashboard.
        
        Standard layout:
          [Band Structure] [DOS] [if 3-panel]
          [Thermodynamics] [Grüneisen] [if 4-panel]
        
        Args:
            formula: Material formula
            plot_band: Include band structure panel
            plot_dos: Include DOS panel
            plot_thermo: Include thermodynamics panel
            plot_gruneisen: Include Grüneisen panel
        
        Returns:
            (figure, list_of_axes) tuple
        """
        n_panels = sum([plot_band, plot_dos, plot_thermo, plot_gruneisen])
        
        if n_panels <= 2:
            layout = '1x2'
            figsize = (12, 5)
        elif n_panels <= 4:
            layout = '2x2'
            figsize = (12, 10)
        else:
            layout = '2x3'
            figsize = (18, 10)
        
        fig, axes = self.create_dashboard_layout(
            n_panels=n_panels,
            layout=layout,
            figsize=figsize
        )
        
        titles = []
        if plot_band:
            titles.append('Phonon Band Structure')
        if plot_dos:
            titles.append('Density of States')
        if plot_thermo:
            titles.append('Thermodynamics')
        if plot_gruneisen:
            titles.append('Grüneisen Parameters')
        
        self.add_panel_labels(fig, axes[:n_panels])
        
        fig.suptitle(f'{formula} - Phonon Analysis Dashboard',
                    fontsize=16, fontweight='bold', y=1.01)
        fig.tight_layout()
        
        return fig, axes
