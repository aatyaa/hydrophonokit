"""
=============================================================================
  HydroPhonoKit Visualization — Interactive Plots (Plotly)

  Implements interactive visualization with plotly:
    1. Interactive Band Structure - Hover for freq, q-point, mode index
    2. Interactive DOS Explorer - Toggle elements, zoom, hover values
    3. Interactive Thermodynamics - Hover for F/S/Cv at any T
    4. Interactive Dashboard - All key plots in one interactive view

  Scientific Foundation:
    Interactive plots enable:
      - Precise value reading at any point
      - Dynamic element toggling
      - Zoom into regions of interest
      - Cross-referencing between plots
      - Export to HTML for sharing

  Dependencies:
    - plotly (pip install plotly)
    - pandas (optional, for data export)

  References:
    [1] Plotly Documentation -- https://plotly.com/python/
=============================================================================
"""
import os
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

# Try to import plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .base_plotter import BasePlotter
from .themes import get_element_color


# ============================================================================
# INTERACTIVE PLOTTER CLASS
# ============================================================================

class InteractivePlotter(BasePlotter):
    """Interactive plotter using plotly for web-based visualization.
    
    Provides methods for interactive phonon plots that work in browsers.
    """
    
    def __init__(self, theme: str = 'nature'):
        """Initialize interactive plotter.
        
        Args:
            theme: Theme name (for color scheme)
        """
        super().__init__(theme=theme)
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly is required for interactive plots. "
                "Install with: pip install plotly"
            )
    
    def plot_interactive_bands(self, band_dict: Dict,
                              formula: str = 'Material',
                              highlight_imaginary: bool = True) -> go.Figure:
        """Plot interactive phonon band structure.
        
        Features:
          - Hover for frequency, q-point, band index
          - Zoom into specific regions
          - Imaginary modes highlighted
          - Export to HTML
        
        Args:
            band_dict: Phonopy band structure dictionary
            formula: Material formula
            highlight_imaginary: Color imaginary modes differently
        
        Returns:
            plotly Figure object
        """
        fig = go.Figure()
        
        distances = band_dict['distances']
        frequencies = band_dict['frequencies']
        n_bands = frequencies[0].shape[1]
        
        # Plot each band
        for b in range(n_bands):
            x_vals = []
            y_vals = []
            hover_texts = []
            
            for dist, freq in zip(distances, frequencies):
                x_vals.extend(dist)
                y_vals.extend(freq[:, b])
                for i in range(len(dist)):
                    hover_texts.append(f'Band: {b+1}<br>q: {dist[i]:.3f}<br>ω: {freq[i, b]:.2f} THz')
            
            # Determine color based on imaginary modes
            has_imaginary = any(np.min(f[:, b]) < -0.1 for f in frequencies)
            color = '#DC2626' if has_imaginary else '#1E3A8A'
            
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                line=dict(color=color, width=1.5),
                hovertext=hover_texts,
                hoverinfo='text',
                name=f'Band {b+1}',
                showlegend=(b == 0)
            ))
        
        # High-symmetry lines
        all_positions = []
        for dist_arr in distances:
            all_positions.extend([dist_arr[0], dist_arr[-1]])
        unique_positions = sorted(set(all_positions))
        
        for pos in unique_positions:
            fig.add_vline(x=pos, line_dash='dash', line_color='gray', opacity=0.5)
        
        # Zero line
        fig.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.7)
        
        # Layout
        fig.update_layout(
            title=f'{formula} - Interactive Phonon Band Structure',
            xaxis_title='Wave Vector q',
            yaxis_title='Frequency (THz)',
            template='plotly_white',
            hovermode='closest',
            showlegend=False,
            height=600,
            width=800
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        
        return fig
    
    def plot_interactive_dos(self, freq: np.ndarray,
                            pdos: np.ndarray,
                            sym_to_idx: Dict[str, List[int]],
                            formula: str = 'Material') -> go.Figure:
        """Plot interactive DOS with element toggling.
        
        Features:
          - Toggle elements on/off
          - Hover for DOS values
          - Stacked or overlay mode
          - Zoom to specific frequency ranges
        
        Args:
            freq: Frequency array (THz)
            pdos: Partial DOS array (n_atoms, n_freq)
            sym_to_idx: Element to atom indices mapping
            formula: Material formula
        
        Returns:
            plotly Figure object
        """
        fig = go.Figure()
        
        elements = list(sym_to_idx.keys())
        colors = [get_element_color(e) for e in elements]
        
        # Total DOS
        total_dos = np.sum([pdos[i] for i in range(pdos.shape[0])], axis=0)
        fig.add_trace(go.Scatter(
            x=freq, y=total_dos,
            mode='lines',
            line=dict(color='black', width=2),
            name='Total DOS',
            hovertemplate='ω: %{x:.2f} THz<br>Total DOS: %{y:.2f}<extra></extra>'
        ))
        
        # Partial DOS
        for elem, color in zip(elements, colors):
            if elem not in sym_to_idx:
                continue
            
            indices = sym_to_idx[elem]
            elem_dos = np.sum([pdos[i] for i in indices if i < pdos.shape[0]], axis=0)
            
            fig.add_trace(go.Scatter(
                x=freq, y=elem_dos,
                mode='lines',
                line=dict(color=color, width=1.5),
                fill='tozeroy',
                fillcolor=color,
                opacity=0.3,
                name=elem,
                hovertemplate=f'ω: %{{x:.2f}} THz<br>{elem} DOS: %{{y:.2f}}<extra></extra>',
                visible='legendonly' if len(elements) > 4 else True
            ))
        
        # Layout
        fig.update_layout(
            title=f'{formula} - Interactive Phonon DOS',
            xaxis_title='Frequency (THz)',
            yaxis_title='DOS (states/THz)',
            template='plotly_white',
            hovermode='x unified',
            height=500,
            width=800,
            legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
        )
        
        return fig
    
    def plot_interactive_thermo(self, temps: np.ndarray,
                               free_energy: np.ndarray,
                               entropy: np.ndarray,
                               cv: np.ndarray,
                               formula: str = 'Material') -> go.Figure:
        """Plot interactive thermodynamics.
        
        Features:
          - Hover for exact values at any T
          - Multiple y-axes for different properties
          - Zoom into specific temperature ranges
        
        Args:
            temps: Temperature array (K)
            free_energy: F(T) (kJ/mol)
            entropy: S(T) (J/(mol·K))
            cv: Cv(T) (J/(mol·K))
            formula: Material formula
        
        Returns:
            plotly Figure object
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Free Energy F(T)', 'Entropy S(T)', 'Heat Capacity Cv(T)'),
            horizontal_spacing=0.1
        )
        
        # Free Energy
        fig.add_trace(go.Scatter(
            x=temps, y=free_energy,
            mode='lines',
            line=dict(color='#1E3A8A', width=2),
            name='F(T)',
            hovertemplate='T: %{x:.0f} K<br>F: %{y:.2f} kJ/mol<extra></extra>'
        ), row=1, col=1)
        
        # Entropy
        fig.add_trace(go.Scatter(
            x=temps, y=entropy,
            mode='lines',
            line=dict(color='#E74C3C', width=2),
            name='S(T)',
            hovertemplate='T: %{x:.0f} K<br>S: %{y:.2f} J/(mol·K)<extra></extra>'
        ), row=1, col=2)
        
        # Heat Capacity
        fig.add_trace(go.Scatter(
            x=temps, y=cv,
            mode='lines',
            line=dict(color='#2ECC71', width=2),
            name='Cv(T)',
            hovertemplate='T: %{x:.0f} K<br>Cv: %{y:.2f} J/(mol·K)<extra></extra>'
        ), row=1, col=3)
        
        # Layout
        fig.update_layout(
            title=f'{formula} - Interactive Thermodynamic Properties',
            template='plotly_white',
            height=500,
            width=1200,
            showlegend=False
        )
        
        for i in range(1, 4):
            fig.update_xaxes(title_text='T (K)', row=1, col=i)
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray', row=1, col=i)
        
        return fig
    
    def plot_interactive_dashboard(self, band_dict: Dict = None,
                                  freq: np.ndarray = None,
                                  pdos: np.ndarray = None,
                                  sym_to_idx: Dict = None,
                                  temps: np.ndarray = None,
                                  free_energy: np.ndarray = None,
                                  entropy: np.ndarray = None,
                                  cv: np.ndarray = None,
                                  formula: str = 'Material') -> go.Figure:
        """Create interactive dashboard with all key plots.
        
        Features:
          - All major plots in one view
          - Cross-referencing via hover
          - Export to standalone HTML
        
        Args:
            band_dict: Band structure data (optional)
            freq: Frequency array (optional)
            pdos: Partial DOS array (optional)
            sym_to_idx: Element mapping (optional)
            temps: Temperature array (optional)
            free_energy: F(T) (optional)
            entropy: S(T) (optional)
            cv: Cv(T) (optional)
            formula: Material formula
        
        Returns:
            plotly Figure object
        """
        # Determine layout based on available data
        n_rows = 0
        has_bands = band_dict is not None
        has_dos = (freq is not None and pdos is not None)
        has_thermo = (temps is not None and cv is not None)
        
        if has_bands:
            n_rows += 1
        if has_dos:
            n_rows += 1
        if has_thermo:
            n_rows += 1
        
        if n_rows == 0:
            raise ValueError("No data available for dashboard")
        
        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=['Phonon Band Structure' if has_bands else '',
                           'Density of States' if has_dos else '',
                           'Thermodynamics' if has_thermo else ''][:n_rows],
            vertical_spacing=0.15,
            specs=[[{'type': 'scatter'}] for _ in range(n_rows)]
        )
        
        current_row = 1
        
        # Band structure
        if has_bands:
            distances = band_dict['distances']
            frequencies = band_dict['frequencies']
            for b in range(min(frequencies[0].shape[1], 10)):  # First 10 bands
                x_vals, y_vals = [], []
                for dist, freq in zip(distances, frequencies):
                    x_vals.extend(dist)
                    y_vals.extend(freq[:, b])
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines', line=dict(width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=current_row, col=1)
            current_row += 1
        
        # DOS
        if has_dos:
            total_dos = np.sum([pdos[i] for i in range(pdos.shape[0])], axis=0)
            fig.add_trace(go.Scatter(
                x=freq, y=total_dos,
                mode='lines', line=dict(width=2),
                showlegend=False
            ), row=current_row, col=1)
            current_row += 1
        
        # Thermodynamics
        if has_thermo:
            fig.add_trace(go.Scatter(
                x=temps, y=cv,
                mode='lines', line=dict(width=2),
                showlegend=False
            ), row=current_row, col=1)
            current_row += 1
        
        fig.update_layout(
            title=f'{formula} - Phonon Analysis Dashboard',
            template='plotly_white',
            height=400 * n_rows,
            width=900,
            hovermode='x unified'
        )
        
        return fig
    
    def save_interactive_html(self, fig: go.Figure, path: str):
        """Save interactive figure to HTML file.
        
        Args:
            fig: Plotly Figure object
            path: Output HTML file path
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.write_html(path, include_plotlyjs=True)
