"""
=============================================================================
  HydroPhonoKit Visualization — Grüneisen Parameters & Thermal Transport

  Implements Grüneisen parameter and thermal conductivity plots:
    1. Mode Grüneisen γ(q,j) - Per-mode anharmonicity
    2. Average Grüneisen γ(T) - Temperature-dependent average
    3. Grüneisen Distribution - Histogram of γ values
    4. Thermal Conductivity κ(T) - Lattice thermal conductivity
    5. Cumulative κ vs MFP - Mean free path analysis
    6. κ Decomposition by Frequency - Frequency-resolved transport
    7. κ Decomposition by MFP - MFP-resolved transport
    8. Slack Model Comparison - Analytical model comparison

  Scientific Foundation:
    Mode Grüneisen parameter:
      γ(q,j) = -∂lnω(q,j)/∂lnV = -V/ω(q,j) × ∂ω(q,j)/∂V

    Average Grüneisen parameter:
      γ(T) = Σ C_v(q,j) × γ(q,j) / Σ C_v(q,j)

    Lattice thermal conductivity (RTA):
      κ = 1/(3V) Σ C_v(q,j) × v_g²(q,j) × τ(q,j)

    Slack model (high-T limit):
      κ = A × M × Θ_D³ × δ / (γ² × T × n^(2/3))

  References:
    [1] Grimvall, Thermophysical Properties of Materials (1999)
    [2] Broido et al., Appl. Phys. Lett. 91, 231922 (2007) -- First-principles κ
    [3] Slack, J. Phys. Chem. Solids 34, 321 (1973) -- Slack model
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
# CONSTANTS
# ============================================================================

R_GAS = 8.314462618  # J/(mol·K)


# ============================================================================
# GRUNEISEN & TRANSPORT PLOTTER CLASS
# ============================================================================

class TransportPlotter(BasePlotter):
    """Specialized plotter for Grüneisen parameters and thermal transport.
    
    Provides methods for Grüneisen and thermal conductivity visualization.
    """
    
    def plot_mode_gruneisen(self, freq: np.ndarray,
                           gruneisen: np.ndarray,
                           formula: str = 'Material',
                           ax: mpl_axes.Axes = None,
                           figsize: Tuple[float, float] = (8, 6),
                           title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot mode Grüneisen parameter vs frequency.
        
        Shows γ(q,j) for each phonon mode, revealing:
          - Acoustic mode behavior (typically γ > 0)
          - Optical mode variations
          - Negative Grüneisen modes (if any)
        
        Args:
            freq: Phonon frequencies (THz)
            gruneisen: Mode Grüneisen parameters
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
        
        # Scatter plot
        mask_pos = gruneisen >= 0
        mask_neg = gruneisen < 0
        
        ax.scatter(freq[mask_pos], gruneisen[mask_pos],
                  color=self.get_color('primary'), s=10, alpha=0.7,
                  label='γ ≥ 0')
        if np.any(mask_neg):
            ax.scatter(freq[mask_neg], gruneisen[mask_neg],
                      color=self.get_color('secondary'), s=10, alpha=0.7,
                      label='γ < 0')
        
        # Zero line
        ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
        
        # Statistics
        avg_gamma = np.mean(gruneisen)
        ax.axhline(y=avg_gamma, color=self.get_color('tertiary'), ls=':',
                  alpha=0.7, label=f'<γ> = {avg_gamma:.2f}')
        ax.legend(fontsize=9)
        
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('Mode Grüneisen γ(q,j)')
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if title is None:
            title = f'{formula} - Mode Grüneisen Parameters'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_average_gruneisen(self, temps: np.ndarray,
                              avg_gruneisen: np.ndarray,
                              formula: str = 'Material',
                              ax: mpl_axes.Axes = None,
                              figsize: Tuple[float, float] = (8, 6),
                              title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot temperature-dependent average Grüneisen parameter.
        
        γ(T) = Σ C_v(q,j) × γ(q,j) / Σ C_v(q,j)
        
        Args:
            temps: Temperature array (K)
            avg_gruneisen: Average Grüneisen parameter γ(T)
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
        
        ax.plot(temps, avg_gruneisen, color=self.get_color('primary'), lw=2)
        
        # High-T limit annotation
        if len(avg_gruneisen) > 0:
            gamma_high_t = avg_gruneisen[-1]
            ax.text(0.98, 0.02, f'γ(@{temps[-1]:.0f}K) = {gamma_high_t:.3f}',
                   transform=ax.transAxes, fontsize=9, ha='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('T (K)')
        ax.set_ylabel('Average Grüneisen γ(T)')
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if title is None:
            title = f'{formula} - Average Grüneisen Parameter vs Temperature'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_gruneisen_distribution(self, gruneisen: np.ndarray,
                                   formula: str = 'Material',
                                   ax: mpl_axes.Axes = None,
                                   figsize: Tuple[float, float] = (8, 6),
                                   title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot histogram of Grüneisen parameter values.
        
        Shows the distribution of γ across all modes.
        
        Args:
            gruneisen: Mode Grüneisen parameters
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
        
        # Histogram
        ax.hist(gruneisen, bins=50, color=self.get_color('primary'),
               alpha=0.7, edgecolor='black')
        
        # Statistics
        mean_gamma = np.mean(gruneisen)
        std_gamma = np.std(gruneisen)
        
        ax.axvline(x=mean_gamma, color=self.get_color('secondary'),
                  lw=2, ls='--', label=f'<γ> = {mean_gamma:.2f}')
        ax.axvline(x=mean_gamma + std_gamma, color=self.get_color('tertiary'),
                  lw=1, ls=':', label=f'σ = {std_gamma:.2f}')
        ax.axvline(x=mean_gamma - std_gamma, color=self.get_color('tertiary'),
                  lw=1, ls=':')
        
        ax.axvline(x=0, color='gray', lw=1, ls='--', alpha=0.5)
        ax.legend(fontsize=9)
        
        ax.set_xlabel('Grüneisen Parameter γ')
        ax.set_ylabel('Count')
        ax.grid(alpha=self.theme['axes']['grid_alpha'], axis='y')
        
        if title is None:
            title = f'{formula} - Grüneisen Parameter Distribution'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_thermal_conductivity(self, temps: np.ndarray,
                                 kappa: np.ndarray,
                                 formula: str = 'Material',
                                 ax: mpl_axes.Axes = None,
                                 figsize: Tuple[float, float] = (8, 6),
                                 title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot lattice thermal conductivity κ(T).
        
        Args:
            temps: Temperature array (K)
            kappa: Thermal conductivity (W/(m·K))
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
        
        # Main κ(T) plot
        ax.plot(temps, kappa, color=self.get_color('primary'), lw=2,
               label='κ (calculated)')
        
        # 1/T fit for high-T
        if len(kappa) > 2 and temps[-1] > 300:
            high_t_mask = temps > 300
            if np.sum(high_t_mask) > 2:
                from scipy import optimize
                def inv_t(t, A):
                    return A / t
                try:
                    popt, _ = optimize.curve_fit(
                        inv_t, temps[high_t_mask], kappa[high_t_mask],
                        p0=[kappa[-1] * temps[-1]]
                    )
                    t_fit = np.linspace(300, temps[-1], 100)
                    ax.plot(t_fit, inv_t(t_fit, *popt),
                           color=self.get_color('secondary'), lw=1.5, ls='--',
                           label=f'1/T fit')
                    ax.legend()
                except Exception:
                    pass
        
        # Annotation
        if len(kappa) > 0:
            ax.text(0.02, 0.95,
                   f'κ(@300K) = {kappa[np.argmin(np.abs(temps-300))]:.1f} W/(m·K)\n'
                   f'κ(@{temps[-1]:.0f}K) = {kappa[-1]:.1f} W/(m·K)',
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('T (K)')
        ax.set_ylabel('κ (W/(m·K))')
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if title is None:
            title = f'{formula} - Lattice Thermal Conductivity'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_cumulative_kappa(self, mfp: np.ndarray,
                             kappa_cumulative: np.ndarray,
                             formula: str = 'Material',
                             ax: mpl_axes.Axes = None,
                             figsize: Tuple[float, float] = (8, 6),
                             title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot cumulative thermal conductivity vs mean free path.
        
        Shows what fraction of κ comes from phonons with ℓ < x.
        Essential for nanostructuring analysis.
        
        Args:
            mfp: Mean free path array (nm)
            kappa_cumulative: Cumulative κ array (W/(m·K))
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
        
        # Main plot (log x-axis)
        ax.semilogx(mfp, kappa_cumulative, color=self.get_color('primary'), lw=2)
        ax.fill_between(mfp, 0, kappa_cumulative, alpha=0.2,
                       color=self.get_color('primary'))
        
        # Total κ annotation
        if len(kappa_cumulative) > 0:
            total_kappa = kappa_cumulative[-1]
            ax.axhline(y=total_kappa, color='gray', ls='--', alpha=0.5,
                      label=f'Total κ = {total_kappa:.1f}')
            
            # 50% and 90% markers
            for pct, color in [(50, 'green'), (90, 'red')]:
                target = total_kappa * pct / 100
                idx = np.argmin(np.abs(kappa_cumulative - target))
                mfp_val = mfp[idx]
                ax.axvline(x=mfp_val, color=color, ls=':', alpha=0.7)
                ax.text(mfp_val, target * 1.05,
                       f'{pct}%: {mfp_val:.1f} nm',
                       color=color, fontsize=9, ha='center',
                       rotation=90, va='bottom')
            
            ax.legend(fontsize=9)
        
        ax.set_xlabel('Mean Free Path (nm)')
        ax.set_ylabel('Cumulative κ (W/(m·K))')
        ax.grid(alpha=self.theme['axes']['grid_alpha'], which='both')
        
        if title is None:
            title = f'{formula} - Cumulative κ vs MFP'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_kappa_decomposition(self, freq: np.ndarray,
                                kappa_by_freq: np.ndarray,
                                formula: str = 'Material',
                                ax: mpl_axes.Axes = None,
                                figsize: Tuple[float, float] = (8, 6),
                                title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot thermal conductivity decomposition by frequency.
        
        Shows which frequency ranges contribute most to κ.
        
        Args:
            freq: Frequency array (THz)
            kappa_by_freq: κ contribution per frequency (W/(m·K·THz))
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
        
        # Bar/step plot
        ax.bar(freq, kappa_by_freq, width=np.mean(np.diff(freq)) if len(freq) > 1 else 0.5,
              color=self.get_color('primary'), alpha=0.7, edgecolor='black')
        
        # Total annotation
        total_kappa = np.sum(kappa_by_freq) * (np.mean(np.diff(freq)) if len(freq) > 1 else 1)
        ax.text(0.02, 0.95, f'Total κ = {total_kappa:.1f} W/(m·K)',
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('dκ/dω (W/(m·K·THz))')
        ax.grid(alpha=self.theme['axes']['grid_alpha'], axis='y')
        
        if title is None:
            title = f'{formula} - κ Decomposition by Frequency'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_slack_comparison(self, temps: np.ndarray,
                             kappa_calc: np.ndarray,
                             theta_D: float,
                             avg_gamma: float,
                             formula: str = 'Material',
                             ax: mpl_axes.Axes = None,
                             figsize: Tuple[float, float] = (8, 6),
                             title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot calculated κ vs Slack model prediction.
        
        Slack model: κ = A × M × Θ_D³ × δ / (γ² × T × n^(2/3))
        
        Args:
            temps: Temperature array (K)
            kappa_calc: Calculated thermal conductivity (W/(m·K))
            theta_D: Debye temperature (K)
            avg_gamma: Average Grüneisen parameter
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
        
        # Calculated κ
        ax.plot(temps, kappa_calc, 'o-', color=self.get_color('primary'),
               markersize=4, lw=1.5, label='Calculated')
        
        # Slack model (1/T behavior)
        if theta_D > 0 and avg_gamma > 0 and len(kappa_calc) > 0:
            # Fit Slack model to high-T data
            high_t_mask = temps > 300
            if np.sum(high_t_mask) > 1:
                A_slack = np.mean(kappa_calc[high_t_mask] * temps[high_t_mask])
                kappa_slack = A_slack / temps
                
                ax.plot(temps, kappa_slack, '--',
                       color=self.get_color('secondary'), lw=2,
                       label=f'Slack model (1/T)')
                
                # Deviation annotation
                if len(kappa_calc) > 0:
                    deviation = np.mean(np.abs(kappa_calc - kappa_slack) / kappa_calc) * 100
                    ax.text(0.02, 0.95, f'Mean deviation: {deviation:.1f}%',
                           transform=ax.transAxes, fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.legend()
        
        ax.set_xlabel('T (K)')
        ax.set_ylabel('κ (W/(m·K))')
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if title is None:
            title = f'{formula} - κ: Calculated vs Slack Model'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
